"""Bind a Cell to its supported upstream CPU Scheduler."""

from dataclasses import replace

from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig


def create_scheduler(config, backend_config, model, hardware, *, export_embedding_token_ids=True):
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
    if config.scheduler_adapter == "upstream_exact":
        return VLLMBackend.from_model(backend_config, requested)

    from ...cell_storage import capacity_inputs
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
