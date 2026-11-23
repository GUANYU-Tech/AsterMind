"""Fast Mode-only admission for ported model sources."""

from astermind.common.hardware_capabilities import PORTABLE_BATCH_HARDWARE


def supports_fast_session_storage(config, model):
    """The GLM-5.2 configuration verified with real 0.30 session storage."""
    from src.serving.workload.model_config import GLM_MOE_DSA_CONFIG_SHA256

    return (
        config.compute_mode == "fast_mode" and config.scheduler_adapter == "vllm_0_30"
        and config.hardware_id == "nvidia_h200" and config.deployment == "colocated"
        and model.config_sha256 == GLM_MOE_DSA_CONFIG_SHA256
        and model.model_type == "glm_moe_dsa" and model.weight_dtype == model.compute_dtype == "bf16"
        and (config.tensor_parallel_size, config.pipeline_parallel_size, config.data_parallel_size,
             config.sequence_parallel_size) == (8, 4, 1, 1)
        and not config.enable_expert_parallel and not config.watermark
    )


def _portable_scope(config, parallel):
    return (config.compute_mode == "fast_mode" and config.hardware_id in PORTABLE_BATCH_HARDWARE
            and config.scheduler_adapter == "vllm_0_30" and config.deployment == "colocated"
            and parallel.context_parallel_size == parallel.sequence_parallel_size == 1)


KIMI_VENDOR_PROJECTION_HARDWARE = frozenset(("ascend_910b", "metax_c500"))


def supports_portable_fp8_reference(config, model, parallel):
    """Allow only the explicit software-widened reference for the pinned sources."""
    if (not _portable_scope(config, parallel) or not config.memory_policy_file):
        return False
    from src.serving.workload.model_config import (
        DEEPSEEK_V3_CONFIG_SHA256, DEEPSEEK_V3_L32_CONFIG_SHA256,
        DEEPSEEK_V4_FLASH_CONFIG_SHA256, DEEPSEEK_V4_PRO_CONFIG_SHA256, GLM_MOE_DSA_5_3_CONFIG_SHA256,
        is_deepseek_v3_data_parallel_model_scope, is_deepseek_v3_parallel_model_scope,
        is_deepseek_v4_data_parallel_model_scope, is_deepseek_v4_parallel_model_scope,
        is_glm_fp8_data_parallel_model_scope, is_glm_fp8_tensor_model_scope,
    )
    tp, pp = parallel.tensor_parallel_size, parallel.pipeline_parallel_size
    if parallel.data_parallel_size > 1 and not parallel.enable_expert_parallel:
        return False
    if model.config_sha256 == GLM_MOE_DSA_5_3_CONFIG_SHA256:
        return (is_glm_fp8_data_parallel_model_scope(model, parallel) if parallel.data_parallel_size > 1
                else is_glm_fp8_tensor_model_scope(model, tp))
    if model.config_sha256 in (DEEPSEEK_V3_CONFIG_SHA256, DEEPSEEK_V3_L32_CONFIG_SHA256):
        return (is_deepseek_v3_data_parallel_model_scope(model, parallel) if parallel.data_parallel_size > 1
                else is_deepseek_v3_parallel_model_scope(
                    model, tp, pp, expert_parallel=parallel.enable_expert_parallel))
    if model.config_sha256 in (DEEPSEEK_V4_PRO_CONFIG_SHA256, DEEPSEEK_V4_FLASH_CONFIG_SHA256):
        return (is_deepseek_v4_data_parallel_model_scope(model, parallel) if parallel.data_parallel_size > 1
                else is_deepseek_v4_parallel_model_scope(
                    model, tp, pp, expert_parallel=parallel.enable_expert_parallel))
    return False


def supports_portable_batch_kimi(config, model, parallel):
    if (not _portable_scope(config, parallel) or model.model_type != "kimi_k2" or model.weight_group_size != 32):
        return False
    from src.serving.workload.model_config import (
        is_kimi_k2_data_parallel_model_scope, is_kimi_k2_parallel_model_scope,
    )
    if parallel.data_parallel_size > 1:
        return parallel.enable_expert_parallel and is_kimi_k2_data_parallel_model_scope(model, parallel)
    return parallel.is_rank_local and is_kimi_k2_parallel_model_scope(
        model, parallel.tensor_parallel_size, parallel.pipeline_parallel_size,
        expert_parallel=parallel.enable_expert_parallel,
    )


def supports_portable_batch_k3(config, model, parallel):
    from src.serving.workload.model_config import (
        KIMI_K3_CONFIG_SHA256, is_kimi_k3_data_parallel_model_scope,
        is_kimi_k3_parallel_model_scope,
    )

    if (not _portable_scope(config, parallel) or model.config_sha256 != KIMI_K3_CONFIG_SHA256):
        return False
    if parallel.data_parallel_size > 1:
        return parallel.enable_expert_parallel and is_kimi_k3_data_parallel_model_scope(model, parallel)
    return parallel.is_rank_local and is_kimi_k3_parallel_model_scope(
        model, parallel.tensor_parallel_size, parallel.pipeline_parallel_size,
        expert_parallel=parallel.enable_expert_parallel,
    )
