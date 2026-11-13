"""Bind a Cell to its supported upstream CPU Scheduler."""

from dataclasses import replace
import os
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

    legacy = (config.hardware_id == "nvidia_a800" and config.pipeline_parallel_size == 4
              and config.data_parallel_size == 1)
    wide = (config.hardware_id == "nvidia_h200" and config.tensor_parallel_size == 8
            and config.data_parallel_size == 4 and config.pipeline_parallel_size in (1, 2)
            and config.enable_expert_parallel)
    qwen_pp = (model.model_type in ("qwen3", "qwen3_moe") and config.hardware_id == "nvidia_h200"
               and ((config.tensor_parallel_size in (1, 2) and config.pipeline_parallel_size == 2)
                    or (model.model_type == "qwen3" and config.tensor_parallel_size == 8 and config.pipeline_parallel_size == 1))
               and config.data_parallel_size == 1)
    glm = model.model_type == "glm_moe_dsa" and (legacy or wide)
    if model.weight_dtype != "bf16" or config.deployment != "colocated" or not (glm or qwen_pp):
        raise NotImplementedError("unsupported model or placement for the explicit 0.30 KV budget adapter")
    python = os.environ.get("ORBITServe_VLLM030_PYTHON")
    source = os.environ.get("ORBITServe_VLLM030_SOURCE")
    if not python or not source:
        raise ValueError("set ORBITServe_VLLM030_PYTHON and ORBITServe_VLLM030_SOURCE")
    parallelism = ParallelConfig(tensor_parallel_size=config.tensor_parallel_size,
                                 data_parallel_size=config.data_parallel_size,
                                 sequence_parallel_size=config.sequence_parallel_size,
                                 pipeline_parallel_size=config.pipeline_parallel_size,
                                 enable_expert_parallel=config.enable_expert_parallel)
    ranges = pipeline_layer_ranges_for_model(model, parallelism.pipeline_parallel_size)
    layers = tuple(end - start for start, end in ranges)
    residencies = model_rank_residencies(
        model, parallelism, worker_role="colocated", model_execution_version="vllm_0_30_0",
        pipeline_layer_partition=layers,
    )
    # Use the projected A800 policy's 90% utilization and 2 GiB non-KV reserve.
    hbm_budget = hardware.resource_by_role("dram").size_bytes * 9 // 10 - 2 * 1024**3
    budgets = tuple(hbm_budget - rank.weight_resident_bytes for rank in residencies)
    if min(budgets) <= 0:
        raise ValueError("weights and non-KV reserve exceed the configured HBM budget")
    settings = replace(requested, scheduler_version="0.30.0", data_parallel_size=config.data_parallel_size,pipeline_layer_partition=layers,
                       enable_chunked_prefill=False, gpu_memory_utilization=0.9)
    return VLLMBackend.from_vllm030_byte_budget(
        settings, python_executable=Path(python), source_root=Path(source),
        pool_name=backend_config.pool_name, max_batched_tokens=config.max_batched_tokens,
        max_requests=config.max_requests, kv_block_size_tokens=config.kv_block_size_tokens,
        available_memory_by_rank=budgets,
        supported_layouts_by_rank=(("LBHNC",) if glm else ("BLHNC", "BLNHC"),) * len(budgets),
    )
