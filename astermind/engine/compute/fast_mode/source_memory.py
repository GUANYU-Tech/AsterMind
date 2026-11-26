"""Source-backed weight and workspace budgets for the native CPU Scheduler."""

import hashlib
import json
import os
from pathlib import Path

from .v3_residency import weight_bytes_by_rank
from .hardware import KV_HBM_UTILIZATION_PERCENT, KV_RUNTIME_RESERVE_BYTES, fast_hbm_capacity_bytes

from src.serving.kv_cache.model_residency import (
    glm_block_fp8_constructor_buffer_bytes,
    glm_block_fp8_postload_parameter_bytes,
    kimi_k25_outer_loader_payload_bytes,
    kimi_k3_source_workspace_bytes,
    model_rank_residencies,
)

K3_PORTABLE_MEMORY_PROFILE = "kimi-k3-mxfp4-bf16-hybrid-reference"
FP8_REFERENCE_MEMORY_PROFILE = "fp8-widened-bf16-reference"


def _projection_limits(config, model, hardware, parallel, profile):
    # K2 reuses grouped INT4 checkpoint storage only. Vendor workspaces and
    # allocator overhead are accounted for by each target's explicit reserve.
    portable_boards = {"nvidia_a800": "a800", "ascend_910b": "910b", "metax_c500": "c500"}
    portable_k3 = profile == K3_PORTABLE_MEMORY_PROFILE
    portable_fp8 = profile == FP8_REFERENCE_MEMORY_PROFILE
    card = portable_boards.get(hardware.hardware_id, "h200") if (
        portable_k3 or portable_fp8 or profile == "kimi-k2-int4-text") else "h200"
    if config.memory_policy_file or portable_k3 or portable_fp8 or card in ("910b", "c500"):
        policy_file = config.root / config.memory_policy_file if config.memory_policy_file else None
    else:
        policy_file = os.environ.get(f"ORBITServe_{card.upper()}_EXPERIMENTAL_MEMORY_PROJECTION")
    if not policy_file:
        raise ValueError(f"{profile} needs an explicit {card.upper()} experimental memory projection")
    policy = json.loads(Path(policy_file).read_text(encoding="utf-8"))
    expected = {
        "schema": f"{card}-native-source-projection-v1",
        "profile": profile,
        "model_config_sha256": model.config_sha256,
        "hardware": hardware.hardware_id,
        "hardware_config_sha256": hashlib.sha256((config.root / config.hardware_file).read_bytes()).hexdigest(),
        "parallel": [parallel.tensor_parallel_size, parallel.data_parallel_size,
                     parallel.pipeline_parallel_size, parallel.expert_parallel_size],
        "limits": [config.max_model_len, config.max_batched_tokens, config.max_requests,
                   config.kv_block_size_tokens],
    }
    if any(policy.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{profile} memory projection identity or workload limits disagree")
    percentage = policy.get("hbm_utilization_percent")
    reserve = policy.get("additional_non_kv_reserve_bytes_per_rank")
    if (type(percentage) is not int or not 0 < percentage <= 100
            or type(reserve) is not int or reserve <= 0):
        raise ValueError(f"{profile} memory projection requires a positive explicit reserve")
    return percentage, reserve


def projected_budgets(config, model, hardware, parallel, layers, profile):
    if config.compute_mode == "fast_mode":
        if config.memory_policy_file:
            _projection_limits(config, model, hardware, parallel, profile)
        percentage, reserve = KV_HBM_UTILIZATION_PERCENT, KV_RUNTIME_RESERVE_BYTES
        capacity = fast_hbm_capacity_bytes(hardware.hardware_id)
    else:
        percentage, reserve = _projection_limits(config, model, hardware, parallel, profile)
        capacity = hardware.resource_by_role("dram").size_bytes
    portable_k3 = profile == K3_PORTABLE_MEMORY_PROFILE
    portable_fp8 = profile == FP8_REFERENCE_MEMORY_PROFILE
    base = capacity * percentage // 100 - reserve
    options = dict(max_batched_tokens=config.max_batched_tokens, max_requests=config.max_requests,
                   pipeline_layer_partition=layers)
    if portable_fp8 and model.model_type == "deepseek_v4":
        from src.serving.kv_cache.model_residency import dsv4_parallel_model_buffer_bytes
        ranks = model_rank_residencies(
            model, parallel, worker_role="colocated", model_execution_version="vllm_0_30_0",
            pipeline_layer_partition=layers, v4_weight_storage_profile="sm90_deepgemm_marlin",
        )
        budgets = tuple(base - entry.weight_resident_bytes - sum(dsv4_parallel_model_buffer_bytes(
            model, parallel, rank, max_batched_tokens=config.max_batched_tokens,
            pipeline_layer_partition=layers).values()) for rank, entry in enumerate(ranks))
    elif profile == "glm-5.3-fp8-sm90-deepgemm" or portable_fp8 and model.model_type == "glm_moe_dsa":
        budgets = tuple(base - sum(glm_block_fp8_postload_parameter_bytes(
            model, parallel, hardware_id="nvidia_h200" if portable_fp8 else hardware.hardware_id,
            linear_backend="deep_gemm",
            routed_backend="deep_gemm", include_mla_absorption=True, logical_rank=rank,
            pipeline_layer_partition=layers).values()) - sum(glm_block_fp8_constructor_buffer_bytes(
                model, parallel, logical_rank=rank, **options).values())
            for rank in range(parallel.worker_count))
    elif profile == "kimi-k3-mxfp4-hybrid" or portable_k3:
        # The workspace formula models the pinned H200 Marlin reference.
        # Keep its 132-SM envelope separate from the target's HBM and reserve.
        source_multiprocessors = 132 if portable_k3 else hardware.num_compute_units
        workspace = kimi_k3_source_workspace_bytes(
            model, parallel, source_multiprocessors=source_multiprocessors, **options)
        ranks = model_rank_residencies(
            model, parallel, worker_role="colocated", model_execution_version="vllm_0_30_0",
            pipeline_layer_partition=layers)
        common = workspace["activation_workspace_upper_bound_bytes"] + workspace["worker_persistent_bytes"]
        budgets = tuple(base - rank.weight_resident_bytes - common - persistent - transient
                        for rank, persistent, transient in zip(
                            ranks, workspace["pipeline_intermediate_persistent_bytes_by_rank"],
                            workspace["pipeline_transient_upper_bound_bytes_by_rank"], strict=True))
    elif profile == "deepseek-v3-fp8-text" or portable_fp8 and model.model_type == "deepseek_v3":
        budgets = tuple(base - weight for weight in weight_bytes_by_rank(model, parallel, layers))
    elif profile == "kimi-k2-int4-text":
        ranks = model_rank_residencies(
            model, parallel, worker_role="colocated",
            model_execution_version="vllm_0_30_0",
            pipeline_layer_partition=layers)
        # Bound the enabled K2 vision tower and projector by a complete
        # replicated copy on every rank, including in the text-only simulation.
        outer = sum(kimi_k25_outer_loader_payload_bytes(
            model, type(parallel)(), vision_tensor_parallel_mode="data").values())
        budgets = tuple(base - rank.weight_resident_bytes - outer for rank in ranks)
    else:
        raise ValueError(f"unsupported experimental memory projection: {profile}")
    return budgets, percentage


def native_kv_event_ceiling(backend, *, hybrid):
    # The event stream counts logical group pages across independent DP pools.
    # The native allocator separately validates physical bytes per rank.
    info = backend.runtime_info
    replicas = info.get("replicas", (info,))
    if len(replicas) != backend.data_parallel_size:
        raise ValueError("native Scheduler returned incomplete DP KV geometry")
    ceiling = 0
    for replica in replicas:
        groups = replica["kv_group_geometry"]
        allocation = replica["native_allocation"]
        blocks = allocation["num_kv_blocks"]
        if (type(blocks) is not int or blocks <= 0 or not groups
                or any(type(tokens) is not int or tokens <= 0 or type(page) is not int or page < 0
                       for tokens, page in groups)):
            raise ValueError("native Scheduler returned invalid KV group geometry")
        if hybrid:
            capacities = replica.get("k3_source_capacities", ())
            supported = (("UniformTypeKVCacheSpecs", "MambaSpec", "MLAAttentionSpec")
                         if replica.get("session_storage") else ("UniformTypeKVCacheSpecs",))
            if (allocation["kv_spec_type"] not in supported or len(capacities) != 2
                    or capacities[1] != blocks or type(capacities[0]) is not int or capacities[0] < blocks):
                raise ValueError("K3 native Scheduler returned invalid hybrid KV capacities")
        page_bytes = sum(page for _, page in groups)
        if page_bytes <= 0:
            raise ValueError("native Scheduler returned no live cache pages")
        ceiling += blocks * page_bytes
    return ceiling
