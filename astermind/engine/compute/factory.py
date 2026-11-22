"""Bind a Cell to its supported upstream CPU Scheduler."""

from dataclasses import replace
import os
import json
from pathlib import Path

from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig


def create_scheduler(config, backend_config, model, hardware, *, export_embedding_token_ids=True):
    if config.data_parallel_size > 1 and config.scheduler_adapter != "vllm_0_30":
        raise NotImplementedError(
            f"Cell DP{config.data_parallel_size} batch execution is supported; online requests require "
            "a coordinated multi-replica vLLM backend, not a single-replica scheduler")
    source_options = {}
    if export_embedding_token_ids and "export_embedding_token_ids" in VLLMSchedulerConfig.__dataclass_fields__:
        source_options["export_embedding_token_ids"] = True
    if config.enable_chunked_prefill is not None or config.long_prefill_token_threshold:
        source_options.update(
            chunked_prefill_override=config.enable_chunked_prefill,
            long_prefill_token_threshold=config.long_prefill_token_threshold,
        )
    requested = VLLMSchedulerConfig(
        str(config.root / config.model_file), dtype={"bf16": "bfloat16", "fp16": "float16"}[model.compute_dtype],
        max_model_len=config.max_model_len, enable_chunked_prefill=True,
        enable_prefix_caching=config.enable_prefix_caching, async_scheduling=False,
        tensor_parallel_size=config.tensor_parallel_size, pipeline_parallel_size=config.pipeline_parallel_size,
        enable_expert_parallel=config.enable_expert_parallel,
        **source_options,
    )
    if config.scheduler_adapter == "vllm_0_11":
        if config.enable_chunked_prefill is False or config.long_prefill_token_threshold:
            raise NotImplementedError(
                "the vLLM 0.11 adapter keeps chunked prefill enabled without a per-request cap")
        return VLLMBackend.from_model(backend_config, requested)
    if config.scheduler_adapter == "vllm_0_30":
        return _vllm030_with_byte_budget(config, backend_config, requested, model, hardware)

    from .cell_storage import capacity_inputs
    from .latest_vllm_backend import LatestSchedulerProcess

    process = LatestSchedulerProcess(backend_config, requested, storage_options={
        "watermark": config.watermark,
        "capacity": capacity_inputs(config, model, hardware),
    })
    try:
        actual_config = replace(backend_config, num_kv_blocks=process.runtime_info.capacity["num_kv_blocks"])
        return VLLMBackend(actual_config, process)
    except BaseException:
        process.close()
        raise


def _vllm030_with_byte_budget(config, backend_config, requested, model, hardware):
    from src.serving.kv_cache.model_residency import model_rank_residencies
    from src.serving.parallelism.parallel_config import ParallelConfig
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
    from src.serving.workload.model_config import (
        is_glm_mtp_tensor_model_scope, is_glm_mtp_data_parallel_model_scope,
        is_dense_parallel_model_scope, is_qwen3_moe_source_model_scope,
        is_kimi_k2_parallel_model_scope, is_kimi_k2_data_parallel_model_scope,
        is_deepseek_v3_parallel_model_scope, is_deepseek_v3_data_parallel_model_scope,
        is_kimi_k3_parallel_model_scope, is_kimi_k3_data_parallel_model_scope,
        is_glm_fp8_tensor_model_scope, is_glm_fp8_data_parallel_model_scope,
    )

    parallelism = ParallelConfig(tensor_parallel_size=config.tensor_parallel_size,
                                 data_parallel_size=config.data_parallel_size,
                                 sequence_parallel_size=config.sequence_parallel_size,
                                 pipeline_parallel_size=config.pipeline_parallel_size,
                                 enable_expert_parallel=config.enable_expert_parallel)
    # KV layout depends on the model, TP sharding and PP layer placement;
    # its byte budget uses the configured device's HBM. The residency
    # calculator below validates the complete model/parallel structure.
    qwen = (model.model_type in ("qwen3", "qwen3_moe")
            and config.data_parallel_size == config.sequence_parallel_size == 1)
    # The native FullAttentionSpec path already supports BF16 Llama and
    # unquantized BF16/FP16 Qwen. Retain each model's configured precision.
    dense = (config.data_parallel_size == config.sequence_parallel_size == 1
             and not config.enable_expert_parallel
             and is_dense_parallel_model_scope(model, config.tensor_parallel_size,
                                               config.pipeline_parallel_size))
    glm = (model.model_type == "glm_moe_dsa" and config.sequence_parallel_size == 1
           and (is_glm_mtp_data_parallel_model_scope(model, parallelism)
                if config.data_parallel_size > 1 else
                is_glm_mtp_tensor_model_scope(model, config.tensor_parallel_size,
                    expert_parallel=config.enable_expert_parallel)))
    from src.serving.workload.model_config import (
        is_deepseek_v4_data_parallel_model_scope, is_deepseek_v4_parallel_model_scope,
    )
    from src.serving.kv_cache.model_residency import dsv4_parallel_model_buffer_bytes
    from src.kernel.hardware_spec import HWSpec
    portable_v4 = (config.compute_mode == "roofline"
        and HWSpec.source_contract_supported("deepseek_v4_portable_roofline", config.hardware_id)
        and (config.data_parallel_size == 1 or config.enable_expert_parallel))
    v4 = ((config.hardware_id == "nvidia_h200" or portable_v4) and model.model_type == "deepseek_v4"
          and model.weight_dtype == "fp8" and model.compute_dtype == "bf16"
          and config.sequence_parallel_size == 1
          and (is_deepseek_v4_data_parallel_model_scope(model, parallelism)
               if config.data_parallel_size > 1 else
               is_deepseek_v4_parallel_model_scope(model, config.tensor_parallel_size,
                   config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel)))
    if v4 and config.kv_block_size_tokens != 256:
        raise ValueError("DeepSeek-V4 indexer requires a 256-token kernel KV block")
    h200_text_source = config.hardware_id == "nvidia_h200" and config.sequence_parallel_size == 1
    a800_text_source = config.hardware_id == "nvidia_a800" and config.sequence_parallel_size == 1
    portable_v3 = (config.compute_mode == "roofline" and config.sequence_parallel_size == 1
        and (HWSpec.source_contract_supported("deepseek_v3_expert_roofline", config.hardware_id)
             if config.enable_expert_parallel else config.data_parallel_size == 1
             and HWSpec.source_contract_supported("deepseek_v3_portable_roofline", config.hardware_id))
        and (is_deepseek_v3_data_parallel_model_scope(model, parallelism) if config.data_parallel_size > 1
             else is_deepseek_v3_parallel_model_scope(model, config.tensor_parallel_size,
                 config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel)))
    qwen_wide_ep = (config.sequence_parallel_size == 1 and config.data_parallel_size > 1
        and config.enable_expert_parallel and is_qwen3_moe_source_model_scope(model)
        and (h200_text_source or a800_text_source
            or HWSpec.source_contract_supported("qwen3_moe_eager_source", config.hardware_id)))
    plain_mla = h200_text_source and (
        (is_kimi_k2_data_parallel_model_scope(model, parallelism)
         or is_deepseek_v3_data_parallel_model_scope(model, parallelism)) if config.data_parallel_size > 1 else
        (is_kimi_k2_parallel_model_scope(model, config.tensor_parallel_size,
            config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel)
         or is_deepseek_v3_parallel_model_scope(model, config.tensor_parallel_size,
            config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel)))
    kimi_text_source = a800_text_source or (config.compute_mode == "roofline"
        and config.sequence_parallel_size == 1
        and HWSpec.source_contract_supported("kimi_k2_portable_roofline", config.hardware_id))
    kimi = kimi_text_source and (
        config.enable_expert_parallel and is_kimi_k2_data_parallel_model_scope(model, parallelism)
        if config.data_parallel_size > 1 else is_kimi_k2_parallel_model_scope(
            model, config.tensor_parallel_size, config.pipeline_parallel_size,
            expert_parallel=config.enable_expert_parallel))
    portable_k3 = (config.compute_mode == "roofline"
        and HWSpec.source_contract_supported("kimi_k3_portable_roofline", config.hardware_id)
        and config.sequence_parallel_size == 1
        and config.tensor_parallel_size in (1, 2, 4, 8, 16, 32)
        and (config.tensor_parallel_size > 1 or config.pipeline_parallel_size > 1)
        and (config.data_parallel_size == 1 or config.enable_expert_parallel))
    k3 = (h200_text_source or portable_k3) and (
        is_kimi_k3_data_parallel_model_scope(model, parallelism) if config.data_parallel_size > 1 else
        is_kimi_k3_parallel_model_scope(model, config.tensor_parallel_size,
            config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel))
    glm_fp8 = h200_text_source and model.model_id == "zai-org/GLM-5.3" and (
        is_glm_fp8_data_parallel_model_scope(model, parallelism) if config.data_parallel_size > 1 else
        is_glm_fp8_tensor_model_scope(model, config.tensor_parallel_size))
    portable_glm_fp8 = (config.compute_mode == "roofline" and config.sequence_parallel_size == 1
        and HWSpec.source_contract_supported("glm_fp8_weight_only_roofline", config.hardware_id)
        and (model.model_id == "zai-org/GLM-5.3" or config.hardware_id == "nvidia_a800"
             or config.data_parallel_size == 1 and not config.enable_expert_parallel)
        and (config.enable_expert_parallel and is_glm_fp8_data_parallel_model_scope(model, parallelism)
             if config.data_parallel_size > 1 else
             is_glm_fp8_tensor_model_scope(model, config.tensor_parallel_size)
             and (not config.enable_expert_parallel or model.model_id == "zai-org/GLM-5.3")))
    glm_fp8 = glm_fp8 or portable_glm_fp8
    if config.deployment != "colocated" or not (v4 or dense or plain_mla or portable_v3 or kimi or qwen_wide_ep or k3 or glm_fp8
            or model.weight_dtype == "bf16" and (glm or qwen)):
        raise NotImplementedError("unsupported model or placement for the explicit 0.30 KV budget adapter")
    python = os.environ.get("ORBITServe_VLLM030_PYTHON")
    source = os.environ.get("ORBITServe_VLLM030_SOURCE")
    if not python or not source:
        raise ValueError("set ORBITServe_VLLM030_PYTHON and ORBITServe_VLLM030_SOURCE")
    ranges = pipeline_layer_ranges_for_model(model, parallelism.pipeline_parallel_size)
    layers = tuple(end - start for start, end in ranges)
    if not glm_fp8:
        residencies = model_rank_residencies(
            model, parallelism, worker_role="colocated", model_execution_version="vllm_0_30_0",
            pipeline_layer_partition=layers,
            v4_weight_storage_profile="sm90_deepgemm_marlin" if v4 else None,
        )
    # The portable V4 simulator reserves the padded packed-weight inventory
    # (including FP32 scales) without claiming a resident CUDA layout.
    # Keep the native H200 experiment's explicit memory policy.
    percentage, extra_reserve = 90, 2 * 1024**3
    if v4 and not portable_v4:
        policy_file = os.environ.get("ORBITServe_V4_EXPERIMENTAL_MEMORY_PROJECTION")
        if not policy_file:
            raise ValueError("explicit H200 V4 experimental memory projection is required")
        policy = json.loads(Path(policy_file).read_text(encoding="utf-8"))
        if (policy.get("schema") != "h200-v4-native-projection-v1"
                or policy.get("hardware") != config.hardware_id
                or policy.get("model_file") != str(config.model_file)
                or policy.get("weight_storage_profile") != "sm90_deepgemm_marlin"):
            raise ValueError("H200 V4 experimental memory projection identity mismatch")
        percentage = policy.get("hbm_utilization_percent")
        extra_reserve = policy.get("additional_non_kv_reserve_bytes_per_rank")
        if (type(percentage) is not int or not 0 < percentage <= 100
                or type(extra_reserve) is not int or extra_reserve < 0):
            raise ValueError("invalid experimental memory projection parameters")
    hbm_budget = hardware.resource_by_role("dram").size_bytes * percentage // 100 - extra_reserve
    budgets = (_project_glm53_kv_budgets(config, model, hardware, parallelism, layers,
                                      reserve=extra_reserve if portable_glm_fp8 else None) if glm_fp8
               else tuple(hbm_budget - rank.weight_resident_bytes for rank in residencies))
    if v4:
        budgets = tuple(budget - sum(dsv4_parallel_model_buffer_bytes(
            model, parallelism, rank, max_batched_tokens=config.max_batched_tokens,
            pipeline_layer_partition=layers).values()) for rank, budget in enumerate(budgets))
    if k3:
        from src.serving.kv_cache.model_residency import kimi_k3_source_workspace_bytes
        workspace = kimi_k3_source_workspace_bytes(model, parallelism,
            max_batched_tokens=config.max_batched_tokens, max_requests=config.max_requests,
            source_multiprocessors=hardware.num_compute_units, pipeline_layer_partition=layers)
        working_bytes = (workspace["activation_workspace_upper_bound_bytes"]
            + workspace["worker_persistent_bytes"] + workspace["pipeline_p2p_working_bytes"])
        budgets = tuple(budget - working_bytes for budget in budgets)
    if min(budgets) <= 0:
        raise ValueError("weights and non-KV reserve exceed the configured HBM budget")
    settings = replace(requested, scheduler_version="0.30.0", data_parallel_size=config.data_parallel_size,pipeline_layer_partition=layers,
                       enable_chunked_prefill=config.enable_chunked_prefill is True, gpu_memory_utilization=percentage / 100)
    return VLLMBackend.from_vllm030_byte_budget(
        settings, python_executable=Path(python), source_root=Path(source),
        pool_name=backend_config.pool_name, max_batched_tokens=config.max_batched_tokens,
        max_requests=config.max_requests, kv_block_size_tokens=config.kv_block_size_tokens,
        available_memory_by_rank=budgets,
        supported_layouts_by_rank=(("LBHNC",) if glm or portable_glm_fp8 else ("BLHNC",) if k3 else ("BLHNC", "BLNHC"),) * len(budgets),
    )


def _project_glm53_kv_budgets(config, model, hardware, parallelism, layers, *, reserve=None):
    """Bound KV by source payloads and an explicit per-rank runtime reserve."""
    from src.serving.kv_cache.model_residency import (
        glm_block_fp8_constructor_buffer_bytes, glm_block_fp8_loader_payload_bytes,
    )

    if reserve is None:
        path = os.environ.get("ORBITServe_GLM53_MEMORY_PROJECTION")
        if not path:
            raise ValueError("explicit H200 GLM-5.3 memory projection is required")
        policy = json.loads(Path(path).read_text(encoding="utf-8"))
        if (policy.get("schema") != "h200-glm53-native-projection-v1"
                or policy.get("model_config_sha256") != model.config_sha256
                or policy.get("hardware") != config.hardware_id):
            raise ValueError("H200 GLM-5.3 memory projection identity mismatch")
        reserve = policy.get("additional_non_kv_reserve_bytes_per_rank")
    if type(reserve) is not int or reserve <= 0:
        raise ValueError("GLM-5.3 requires a positive runtime workspace and non-KV reserve")
    budget = hardware.resource_by_role("dram").size_bytes * 90 // 100 - reserve
    result = []
    for rank in range(parallelism.worker_count):
        placement = dict(logical_rank=rank, pipeline_layer_partition=layers)
        weights = glm_block_fp8_loader_payload_bytes(model, parallelism,
            include_mla_absorption=True, **placement)
        buffers = glm_block_fp8_constructor_buffer_bytes(model, parallelism,
            max_batched_tokens=config.max_batched_tokens, max_requests=config.max_requests, **placement)
        result.append(budget - sum(weights.values()) - sum(buffers.values()))
    return tuple(result)
