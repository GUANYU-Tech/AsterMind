"""Bind a Cell to its supported upstream CPU Scheduler."""

from dataclasses import replace
import os
import json
from pathlib import Path

from astermind.common.hardware_capabilities import PORTABLE_BATCH_HARDWARE
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig


from .fast_mode.admission import (
    supports_portable_batch_k3, supports_portable_batch_kimi, supports_portable_fp8_reference,
)


def standard_vllm030_budgets(model, hardware, parallel, layers, *, percentage=90,
                              extra_reserve=2 * 1024**3, v4_weight_storage_profile=None):
    from src.serving.kv_cache.model_residency import model_rank_residencies

    residencies = model_rank_residencies(
        model, parallel, worker_role="colocated", model_execution_version="vllm_0_30_0",
        pipeline_layer_partition=layers, v4_weight_storage_profile=v4_weight_storage_profile,
    )
    hbm_budget = hardware.resource_by_role("dram").size_bytes * percentage // 100 - extra_reserve
    return tuple(hbm_budget - rank.weight_resident_bytes for rank in residencies)


def create_scheduler(config, backend_config, model, hardware, *, export_embedding_token_ids=True):
    if config.data_parallel_size > 1 and config.scheduler_adapter != "vllm_0_30":
        raise NotImplementedError(
            f"Cell DP{config.data_parallel_size} batch execution is supported; online requests require "
            "a coordinated multi-replica vLLM backend, not a single-replica scheduler")
    source_options = {}
    if export_embedding_token_ids and "export_embedding_token_ids" in VLLMSchedulerConfig.__dataclass_fields__:
        source_options["export_embedding_token_ids"] = True
    requested = VLLMSchedulerConfig(
        str(config.root / config.model_file), dtype={"bf16": "bfloat16", "fp16": "float16"}[model.compute_dtype],
        max_model_len=config.max_model_len, enable_chunked_prefill=True,
        enable_prefix_caching=config.enable_prefix_caching, async_scheduling=False,
        tensor_parallel_size=config.tensor_parallel_size, pipeline_parallel_size=config.pipeline_parallel_size,
        enable_expert_parallel=config.enable_expert_parallel,
        **source_options,
    )
    if config.scheduler_adapter == "vllm_0_11":
        return VLLMBackend.from_model(backend_config, requested)
    if config.scheduler_adapter == "vllm_0_30":
        return _vllm030_with_byte_budget(config, backend_config.pool_name, model, hardware, requested)

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


def _vllm030_with_byte_budget(config, pool_name, model, hardware, requested=None):
    from src.serving.parallelism.parallel_config import ParallelConfig
    from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
    from src.serving.parallelism.tp import validate_gqa_parallelism
    from src.serving.workload.model_config import (
        is_glm_mtp_tensor_model_scope, is_glm_mtp_data_parallel_model_scope,
        GLM_MOE_DSA_5_3_CONFIG_SHA256, is_dense_parallel_model_scope, is_glm_fp8_data_parallel_model_scope,
        is_kimi_k3_data_parallel_model_scope, is_kimi_k3_parallel_model_scope, is_glm_fp8_tensor_model_scope,
        is_kimi_k2_parallel_model_scope, is_kimi_k2_data_parallel_model_scope,
        is_deepseek_v3_parallel_model_scope, is_deepseek_v3_data_parallel_model_scope,
        is_qwen3_moe_source_model_scope, is_qwen3_moe_pipeline_model_scope,
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
    fast_mode = config.compute_mode == "fast_mode"
    qwen_wide_ep = (fast_mode and config.sequence_parallel_size == 1
        and (config.hardware_id == "nvidia_h200" or config.hardware_id in PORTABLE_BATCH_HARDWARE)
        and config.data_parallel_size > 1 and config.enable_expert_parallel
        and is_qwen3_moe_source_model_scope(model)
        and model.num_experts % parallelism.expert_parallel_size == 0
        and (config.pipeline_parallel_size == 1 or is_qwen3_moe_pipeline_model_scope(
            model, config.tensor_parallel_size, config.pipeline_parallel_size)))
    if qwen_wide_ep:
        validate_gqa_parallelism(model, parallelism)
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
    portable_fp8 = supports_portable_fp8_reference(config, model, parallelism)
    v4 = ((config.hardware_id == "nvidia_h200" or portable_fp8) and model.model_type == "deepseek_v4"
          and model.weight_dtype == "fp8" and model.compute_dtype == "bf16"
          and config.sequence_parallel_size == 1
          and (is_deepseek_v4_data_parallel_model_scope(model, parallelism)
               if config.data_parallel_size > 1 else
               is_deepseek_v4_parallel_model_scope(model, config.tensor_parallel_size,
                   config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel)))
    if v4 and config.kv_block_size_tokens != 256:
        reference = "" if portable_fp8 else "H200 "
        raise ValueError(f"{reference}DeepSeek-V4 indexer requires a 256-token kernel KV block")
    v3 = portable_fp8 and model.model_type == "deepseek_v3"
    glm_fp8 = portable_fp8 and model.model_type == "glm_moe_dsa"
    portable_k3 = supports_portable_batch_k3(config, model, parallelism)
    k3 = portable_k3
    k2 = supports_portable_batch_kimi(config, model, parallelism)
    if fast_mode and config.hardware_id == "nvidia_h200" and config.sequence_parallel_size == 1:
        glm_fp8 = (model.config_sha256 == GLM_MOE_DSA_5_3_CONFIG_SHA256
                   and (is_glm_fp8_data_parallel_model_scope(model, parallelism)
                        if config.data_parallel_size > 1 else
                        is_glm_fp8_tensor_model_scope(model, config.tensor_parallel_size)))
        k3 = (config.enable_expert_parallel and is_kimi_k3_data_parallel_model_scope(model, parallelism)
              if config.data_parallel_size > 1 else
              is_kimi_k3_parallel_model_scope(model, config.tensor_parallel_size,
                  config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel))
        k2 = (is_kimi_k2_data_parallel_model_scope(model, parallelism)
              if config.data_parallel_size > 1 else
              is_kimi_k2_parallel_model_scope(model, config.tensor_parallel_size,
                  config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel))
        v3 = (is_deepseek_v3_data_parallel_model_scope(model, parallelism)
              if config.data_parallel_size > 1 else
              is_deepseek_v3_parallel_model_scope(model, config.tensor_parallel_size,
                  config.pipeline_parallel_size, expert_parallel=config.enable_expert_parallel))
    if config.deployment != "colocated" or not (
            glm_fp8 or k3 or k2 or v3 or v4 or dense or qwen_wide_ep
            or model.weight_dtype == "bf16" and (glm or qwen)):
        raise NotImplementedError("unsupported model or placement for the explicit 0.30 KV budget adapter")
    if requested is None:
        if not k3:
            raise ValueError("hybrid native budget requires the supported K3 source")
        if config.kv_block_size_tokens != config.max_model_len:
            raise ValueError("K3 hybrid Scheduler block size must equal the configured model context")
        requested = VLLMSchedulerConfig(
            str(config.root / config.model_file), dtype="bfloat16", scheduler_version="0.30.0",
            max_model_len=config.max_model_len, enable_chunked_prefill=False,
            enable_prefix_caching=False, async_scheduling=False,
            tensor_parallel_size=config.tensor_parallel_size,
            pipeline_parallel_size=config.pipeline_parallel_size,
            enable_expert_parallel=config.enable_expert_parallel,
        )
    python = os.environ.get("ORBITServe_VLLM030_PYTHON")
    source = os.environ.get("ORBITServe_VLLM030_SOURCE")
    if not python or not source:
        raise ValueError("set ORBITServe_VLLM030_PYTHON and ORBITServe_VLLM030_SOURCE")
    ranges = pipeline_layer_ranges_for_model(model, parallelism.pipeline_parallel_size)
    layers = tuple(end - start for start, end in ranges)
    # Established profiles retain their original policy. V4 requires an
    # explicitly supplied experimental memory projection.
    percentage, extra_reserve = 90, 2 * 1024**3
    if v4 and not portable_fp8:
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
    if portable_fp8:
        from .fast_mode.source_memory import FP8_REFERENCE_MEMORY_PROFILE, projected_budgets
        budgets, percentage = projected_budgets(
            config, model, hardware, parallelism, layers, FP8_REFERENCE_MEMORY_PROFILE)
    elif glm_fp8 or k3 or k2 or v3:
        from .fast_mode.source_memory import K3_PORTABLE_MEMORY_PROFILE, projected_budgets
        profile = (K3_PORTABLE_MEMORY_PROFILE if portable_k3
                   else "glm-5.3-fp8-sm90-deepgemm" if glm_fp8 else "kimi-k3-mxfp4-hybrid" if k3
                   else "kimi-k2-int4-text" if k2 else "deepseek-v3-fp8-text")
        budgets, percentage = projected_budgets(config, model, hardware, parallelism, layers, profile)
    else:
        budgets = standard_vllm030_budgets(
            model, hardware, parallelism, layers, percentage=percentage, extra_reserve=extra_reserve,
            v4_weight_storage_profile="sm90_deepgemm_marlin" if v4 else None,
        )
    if v4 and not portable_fp8:
        budgets = tuple(budget - sum(dsv4_parallel_model_buffer_bytes(
            model, parallelism, rank, max_batched_tokens=config.max_batched_tokens,
            pipeline_layer_partition=layers).values()) for rank, budget in enumerate(budgets))
    if min(budgets) <= 0:
        raise ValueError("weights and non-KV reserve exceed the configured HBM budget")
    settings = replace(requested, scheduler_version="0.30.0", data_parallel_size=config.data_parallel_size,pipeline_layer_partition=layers,
                       enable_chunked_prefill=False, gpu_memory_utilization=percentage / 100)
    layouts = ("LBHNC",) if glm or glm_fp8 else ("BLHNC",) if k3 else ("BLHNC", "BLNHC")
    return VLLMBackend.from_vllm030_byte_budget(
        settings, python_executable=Path(python), source_root=Path(source),
        pool_name=pool_name, max_batched_tokens=config.max_batched_tokens,
        max_requests=config.max_requests, kv_block_size_tokens=config.kv_block_size_tokens,
        available_memory_by_rank=budgets,
        supported_layouts_by_rank=(layouts,) * len(budgets),
    )
