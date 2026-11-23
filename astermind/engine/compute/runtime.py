"""Translate one logical Cell deployment into upstream Mid-Layer inputs."""

from dataclasses import dataclass, replace
from inspect import getfile, signature
import math
from pathlib import Path
from typing import Literal

from src.kernel.hardware_spec import HWSpec
from src.serving.deployment.deployment_config import (
    CollectiveConfig, DeploymentConfig, PipelineP2PConfig, RouteConfig, WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.deployment.pd_disagg import build_pd_disagg_plan
from src.serving.execution import ServingExecutionContext
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.backend.vllm import BackendConfig, VLLMSchedulerConfig
from src.serving.simulation import validate_workload_emission_parallelism
from astermind.common.hardware_capabilities import PORTABLE_BATCH_HARDWARE
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs

from .factory import create_scheduler, resolve_batch_token_budget
from .fast_mode.admission import (
    KIMI_VENDOR_PROJECTION_HARDWARE, supports_fast_session_storage, supports_portable_batch_k3,
    supports_portable_batch_kimi, supports_portable_fp8_reference,
)


_STEP_PARAMETERS = signature(ServingExecutionContext.execute_step).parameters
_SOURCE_INPUTS = "embedding_token_ids" in _STEP_PARAMETERS
_HAS_PLAN_BUILDER = hasattr(ServingExecutionContext, "prepare_step")


def prepare_scheduled_step(execution, step, source_prompt_lengths):
    """Adapt the Scheduler batch to the dev plan builder."""
    if not _HAS_PLAN_BUILDER or not _SOURCE_INPUTS or "source_prompt_lengths" in _STEP_PARAMETERS:
        raise NotImplementedError("strict surrogate run requires the unpriced prepare_step interface")
    source_inputs = {
        "embedding_token_ids": step.embedding_token_ids,
        "model_source_inputs": step.model_source_inputs,
    }

    # AsterMind supplies length-only requests to the CPU scheduler. When its
    # MLA bridge omits embedding IDs, use the same synthetic ID as that bridge.
    if source_inputs["embedding_token_ids"] is None and execution.model.attention_layout != "standard_gqa":
        source_inputs["embedding_token_ids"] = (
            (VLLMSchedulerConfig.dummy_token_id,) * step.schedule.token_count
        )

    runtime = execution.runtime_inputs
    if execution.model.model_type == "deepseek_v4":
        execution.runtime_inputs = replace(runtime, v4_source_prompt_lengths=source_prompt_lengths)
    elif execution.model.model_type in ("glm_moe_dsa", "kimi_k2"):
        request_ids = (row.request_id for row in step.schedule.requests)
        prompt_lengths = dict(zip(request_ids, source_prompt_lengths, strict=True))
        execution.runtime_inputs = replace(
            runtime, symbolic_source_prompt_lengths_by_request=prompt_lengths,
        )
    try:
        return execution.prepare_step(step.schedule, kv_cache_events=step.kv_cache_events, **source_inputs)
    finally:
        execution.runtime_inputs = runtime


@dataclass(frozen=True)
class MidlayerCellConfig:
    root: Path
    model_file: str = "configs/model/qwen3-4b.json"
    hardware_file: str = "configs/hardware/nvidia_a800.yaml"
    hardware_id: str = "nvidia_a800"
    max_model_len: int = 128
    max_batched_tokens: int | None = None
    max_requests: int = 4
    num_kv_blocks: int = 64
    kv_block_size_tokens: int = 16
    scheduler_adapter: Literal["auto", "vllm_0_11", "vllm_0_30", "vllm_0_31"] = "auto"
    enable_prefix_caching: bool = False
    watermark: float = 0.0
    memory_policy_file: str | None = None
    data_parallel_size: int = 1
    sequence_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    enable_expert_parallel: bool = False
    expert_routing_seed: int | None = None
    host_launch_profile_file: str | None = None
    deployment: Literal["colocated", "pd_disagg"] = "colocated"
    execution_mode: Literal["accelerated", "fast", "reference"] = "accelerated"
    compute_mode: Literal["midlayer", "roofline", "fast_mode"] = "midlayer"
    enable_chunked_prefill: bool | None = None
    long_prefill_token_threshold: int = 0
    fast_glm_dsa_score: Literal["bf16_timing_proxy"] | None = None


def _runtime_inputs(config, model, *, portable_fp8=False):
    seed = config.expert_routing_seed
    if model.num_experts:
        if type(seed) is not int or seed < 0:
            raise ValueError("a MoE Cell requires a nonnegative expert_routing_seed")
        routing = ExpertRoutingInput.seeded_assumption("orbitserve-explicit-routing-assumption", seed)
    else:
        if config.enable_expert_parallel or seed is not None:
            raise ValueError("dense Cells cannot carry expert parallelism or routing inputs")
        routing = ExpertRoutingInput.not_applicable()

    if not _SOURCE_INPUTS:
        layout = "paged_varlen" if model.attention_layout == "standard_gqa" else "mla"
        return RuntimeInputs(routing, layout, "uniform")

    if model.attention_layout == "standard_gqa":
        portable_program = _uses_portable_qwen_program(config, model)
        if config.compute_mode == "fast_mode":
            # The portable Fast program only uses logical rank semantics.
            vendor_forward = not portable_program and config.data_parallel_size == 1 and (
                config.hardware_id == "ascend_910b"
                or config.hardware_id == "metax_c500" and model.model_type == "qwen3_moe")
        else:
            # The CPU scheduler version can differ from the modeled device forward.
            vendor_ep2 = (config.hardware_id in ("metax_c500", "ascend_910b")
                and model.model_type == "qwen3_moe" and config.data_parallel_size == 1
                and config.tensor_parallel_size == 2 and config.enable_expert_parallel)
            ascend_dense = config.hardware_id == "ascend_910b" and model.num_experts == 0
            vendor_forward = model.compute_dtype == model.weight_dtype == "bf16" and (vendor_ep2 or ascend_dense)
        if config.scheduler_adapter == "vllm_0_30" and not vendor_forward:
            return RuntimeInputs(routing, "paged_varlen", "uniform",
                model_execution_version="vllm_0_30_0",
                symbolic_forward_source="cuda_eager_inplace" if portable_program else None,
                target_sampling_source=f"vllm_0_30_plain_greedy_{model.compute_dtype}")
        version, sampler = {
            "ascend_910b": ("vllm_0_18_0", "vllm_0_18"),
            "metax_c500": ("vllm_0_24_0", "vllm_0_24"),
        }.get(config.hardware_id, ("vllm_0_11_0", "vllm_0_11"))
        return RuntimeInputs(routing, "paged_varlen", "uniform",
            model_execution_version=version, target_sampling_source=f"{sampler}_plain_greedy_{model.compute_dtype}")
    if model.model_type == "deepseek_v4":
        return RuntimeInputs(routing, "mla", "uniform", model_execution_version="vllm_0_30_0",
            target_sampling_source="vllm_0_30_plain_greedy_bf16",
            symbolic_forward_source="vllm_0_30_v4_sync_fp8_ds_mla",
            v4_eager_metadata_source="vllm_0_30_eager_dict_all_valid",
            v4_routing_source="vllm_0_30_modular_cuda_top6_pure_text",
            v4_source_max_model_len=config.max_model_len,
            v4_source_max_batched_tokens=config.max_batched_tokens,
            v4_indexer_selection_source="declared_causal_topk",
            v4_source_use_deep_gemm=False)
    if model.model_type == "deepseek_v3":
        return RuntimeInputs(routing, "mla", "uniform", model_execution_version="vllm_0_27_1",
            target_sampling_source="vllm_0_27_1_plain_greedy_bf16",
            symbolic_forward_source="vllm_0_27_1_deepseek_v3_sync_fp8",
            symbolic_source_max_model_len=config.max_model_len,
            symbolic_source_max_batched_tokens=config.max_batched_tokens)
    runtime = RuntimeInputs(routing, "mla", "uniform", model_execution_version="vllm_0_30_0",
        target_sampling_source="vllm_0_30_plain_greedy_bf16",
        symbolic_source_max_model_len=config.max_model_len,
        symbolic_source_max_batched_tokens=config.max_batched_tokens)
    if portable_fp8:
        ranks = config.tensor_parallel_size * config.pipeline_parallel_size * config.data_parallel_size
        return replace(runtime, symbolic_forward_source="vllm_0_30_v32_fp8_sync_explicit_allreduce",
                       glm_indexer_selection_source="declared_causal_topk",
                       symbolic_workspace_initial_capacity_bytes=(0,) * ranks)
    if (model.model_type == "kimi_linear" and config.compute_mode == "fast_mode"
            and config.hardware_id in PORTABLE_BATCH_HARDWARE):
        # The Fast executor uses the same K3 forward semantics without invoking
        # the detailed NVIDIA kernel provider.
        return replace(runtime, symbolic_forward_source="vllm_0_30_k3_mxfp4_sync")
    if (model.model_type == "kimi_k2" and config.compute_mode == "fast_mode"
            and config.hardware_id in KIMI_VENDOR_PROJECTION_HARDWARE):
        # Record the logical source used by the Fast formulas. This mode
        # does not emit the NVIDIA-specific detailed forward.
        return replace(runtime, symbolic_forward_source="vllm_0_30_k2_6_int4_sync_explicit_allreduce")
    if config.fast_glm_dsa_score is not None:
        ranks = config.tensor_parallel_size * config.pipeline_parallel_size * config.data_parallel_size
        return replace(runtime, symbolic_forward_source="vllm_0_30_v32_bf16_sync_explicit_allreduce",
                       glm_indexer_selection_source="declared_causal_topk",
                       glm_prefill_attention_source=("force_sparse_mqa" if config.data_parallel_size > 1
                                                     else runtime.glm_prefill_attention_source),
                       symbolic_workspace_initial_capacity_bytes=(0,) * ranks)
    if (model.model_type == "glm_moe_dsa" and model.weight_dtype == "bf16"
            and config.data_parallel_size > 1 and config.pipeline_parallel_size == 1
            and (config.hardware_id == "nvidia_a800" or config.compute_mode != "fast_mode"
                and HWSpec.source_contract_supported("glm_bf16_portable_roofline", config.hardware_id))):
        # The existing distributed BF16 forward requires sparse MQA during prefill.
        return replace(runtime, glm_prefill_attention_source="force_sparse_mqa")
    if (config.data_parallel_size > 1 and model.model_type == "glm_moe_dsa"
            and model.weight_dtype == "bf16" and config.hardware_id == "nvidia_h200"):
        ranks = config.tensor_parallel_size * config.pipeline_parallel_size * config.data_parallel_size
        return replace(runtime, symbolic_forward_source="vllm_0_30_v32_bf16_sync_explicit_allreduce",
                       glm_indexer_selection_source="declared_causal_topk",
                       glm_prefill_attention_source="force_sparse_mqa",
                       symbolic_workspace_initial_capacity_bytes=(0,) * ranks)
    if (config.compute_mode == "fast_mode" and config.scheduler_adapter == "vllm_0_30"
            and config.hardware_id == "nvidia_h200" and model.model_type == "glm_moe_dsa" and model.weight_dtype == "fp8"
            and config.data_parallel_size == 1):
        from src.serving.workload.model_config import GLM_MOE_DSA_5_3_CONFIG_SHA256
        if model.config_sha256 == GLM_MOE_DSA_5_3_CONFIG_SHA256:
            ranks = config.tensor_parallel_size * config.pipeline_parallel_size
            return replace(runtime, symbolic_forward_source="vllm_0_30_v32_fp8_sync_explicit_allreduce",
                           glm_indexer_selection_source="declared_causal_topk",
                           symbolic_workspace_initial_capacity_bytes=(0,) * ranks)
    if (model.model_type == "glm_moe_dsa" and config.hardware_id == "nvidia_a800"
            and config.pipeline_parallel_size == 4):
        # Declare a cold source workspace for every rank in this new Cell.
        ranks = config.tensor_parallel_size * config.data_parallel_size * config.pipeline_parallel_size
        return replace(runtime, glm_indexer_selection_source="declared_causal_topk",
                       symbolic_workspace_initial_capacity_bytes=(0,) * ranks)
    return runtime


def create_midlayer_runtime(config: MidlayerCellConfig, *, with_scheduler: bool = True):
    if config.compute_mode not in ("midlayer", "roofline", "fast_mode"):
        raise ValueError("compute_mode must be midlayer, roofline or fast_mode")
    loaded_root = Path(getfile(HWSpec)).resolve().parents[2]
    if config.root.resolve() != loaded_root:
        raise ValueError("Cell root differs from the selected Mid-Layer source")
    model = load_model_config(config.root / config.model_file)
    if config.compute_mode != "fast_mode":
        if (model.model_type == "glm_moe_dsa" and model.weight_dtype == "fp8"
                and HWSpec.source_contract_supported("glm_fp8_weight_only_roofline", config.hardware_id)
                and config.compute_mode != "roofline"):
            raise NotImplementedError("the portable GLM FP8 projection requires Roofline")
        if (model.model_type == "deepseek_v4"
                and HWSpec.source_contract_supported("deepseek_v4_portable_roofline", config.hardware_id)
                and config.compute_mode != "roofline"):
            raise NotImplementedError("the portable DeepSeek-V4 projection requires Roofline")
        if (model.model_type == "kimi_linear"
                and HWSpec.source_contract_supported("kimi_k3_portable_roofline", config.hardware_id)
                and config.compute_mode != "roofline"):
            raise NotImplementedError("the portable Kimi K3 projection requires Roofline")
        if (model.model_type == "deepseek_v3"
                and HWSpec.source_contract_supported("deepseek_v3_portable_roofline", config.hardware_id)
                and config.compute_mode != "roofline"):
            raise NotImplementedError("the portable DeepSeek-V3 projection requires Roofline")
    if not _SOURCE_INPUTS and (config.deployment != "colocated" or config.scheduler_adapter in ("vllm_0_30", "vllm_0_31")
          or config.host_launch_profile_file is not None):
        raise NotImplementedError("the logical main uses colocated CPU scheduling and device timing")
    if config.scheduler_adapter == "auto":
        adapter = ("vllm_0_30" if config.data_parallel_size>1 and model.model_type=="glm_moe_dsa"
                   else "vllm_0_31" if _SOURCE_INPUTS and model.attention_layout == "standard_gqa"
                   else "vllm_0_11")
        config = replace(config, scheduler_adapter=adapter)
    if config.scheduler_adapter not in ("vllm_0_31", "vllm_0_30", "vllm_0_11"):
        raise ValueError("unknown Scheduler adapter")
    config = resolve_batch_token_budget(config)
    if isinstance(config.watermark, bool) or not math.isfinite(config.watermark) or not 0 <= config.watermark < 1:
        raise ValueError("watermark must be in [0, 1)")
    hardware = HWSpec(config.root / config.hardware_file)
    if hardware.hardware_id != config.hardware_id:
        raise ValueError("Cell hardware_id does not match the selected hardware profile")
    parallel = ParallelConfig(data_parallel_size=config.data_parallel_size,
                              sequence_parallel_size=config.sequence_parallel_size, tensor_parallel_size=config.tensor_parallel_size,
                              pipeline_parallel_size=config.pipeline_parallel_size,
                              enable_expert_parallel=config.enable_expert_parallel)
    _validate_batch_hardware(config, model, parallel)
    vendor_kimi = (config.hardware_id in KIMI_VENDOR_PROJECTION_HARDWARE
                   and supports_portable_batch_kimi(config, model, parallel))
    portable_k3 = supports_portable_batch_k3(config, model, parallel)
    portable_fp8 = supports_portable_fp8_reference(config, model, parallel)
    fast_reference_memory = (config.compute_mode == "fast_mode" and config.hardware_id == "nvidia_h200"
                             and config.scheduler_adapter == "vllm_0_30"
                             and (model.model_type in ("deepseek_v3", "deepseek_v4", "kimi_k2", "kimi_linear")
                                  or model.model_type == "glm_moe_dsa" and model.weight_dtype == "fp8"))
    fast_session = (config.compute_mode == "fast_mode" and config.scheduler_adapter == "vllm_0_30"
                    and config.enable_prefix_caching)
    if fast_session and not supports_fast_session_storage(config, model):
        raise NotImplementedError(
            "Fast Mode session storage requires vLLM 0.30, H200 GLM-5.2 TP8xPP4, DP1, EP off")
    storage_enabled = config.scheduler_adapter == "vllm_0_31" or fast_session
    if not storage_enabled and (config.enable_prefix_caching or config.watermark
                                or config.memory_policy_file and not (vendor_kimi or portable_k3
                                                                      or portable_fp8 or fast_reference_memory)):
        raise NotImplementedError("the current upstream Cell adapter requires explicit KV blocks and no session caching")
    layer_ranges = pipeline_layer_ranges_for_model(model, config.pipeline_parallel_size)
    plan_options = {}
    if config.deployment == "pd_disagg":
        if (model.model_id, config.hardware_id, config.tensor_parallel_size,
                config.pipeline_parallel_size, config.enable_expert_parallel) != (
                "moonshotai/Kimi-K2.6", "nvidia_h200", 8, 1, True) or storage_enabled or config.data_parallel_size != 1:
            raise NotImplementedError("PD smoke supports H200 Kimi-K2.6 TP8/EP8 per pool")
        pools = ("prefill", "decode")
        roles = pools
        routes = (RouteConfig("prefill", "decode", "kv", "send_recv", "infiniband",
                             resource_id="pd-kv", backend="analytical_roofline_v1"),)
        build_plan = build_pd_disagg_plan
    else:
        pools, roles, routes = ("main",), ("colocated",), ()
        build_plan = build_deployment_plan
        plan_options["pipeline_layer_partition"] = tuple(end - start for start, end in layer_ranges)
    # Preserve the colocated default's local-fabric assumption with the correct
    # hardware label. This does not resolve physical cross-domain PP routes.
    local_link = {"ascend_910b": "hccs", "metax_c500": "metaxlink"}.get(config.hardware_id, "nvlink")
    collectives = tuple(CollectiveConfig(op, "ring", local_link, "ring", f"main-{local_link}-fabric",
                                        "analytical_roofline_v1")
                        for op in ("all_reduce", "all_gather")) if parallel.worker_count > 1 else ()
    if config.data_parallel_size > 1 and config.enable_expert_parallel:
        collectives += tuple(CollectiveConfig(op, "pairwise", local_link, "fully_connected", "main-ep-fabric",
                            "analytical_roofline_v1") for op in ("all_gatherv", "reduce_scatterv"))
    pipeline_p2p = tuple(PipelineP2PConfig(stage, stage + 1, "send_recv", local_link,
                                         "point_to_point", "main-pipeline-fabric", "analytical_roofline_v1")
                         for stage in range(config.pipeline_parallel_size - 1))
    deployment = build_plan(
        DeploymentConfig(config.deployment, tuple(WorkerPoolConfig(pool, role, config.hardware_id, parallel,
            ranks=tuple(range(index * parallel.worker_count, (index + 1) * parallel.worker_count)),
            collectives=collectives, pipeline_p2p=pipeline_p2p)
            for index, (pool, role) in enumerate(zip(pools, roles, strict=True))), routes=routes),
        num_layers=model.num_hidden_layers, num_experts=model.num_experts, **plan_options,
    )
    host_profiles = None
    if config.host_launch_profile_file is not None:
        from src.serving.host_timing import load_host_launch_profile
        profile = load_host_launch_profile(config.root / config.host_launch_profile_file)
        if profile.hardware_id != config.hardware_id:
            raise ValueError("Host launch profile hardware differs from the Cell")
        host_profiles = {config.hardware_id: profile}
    execution = ServingExecutionContext(
        model, deployment, {config.hardware_id: hardware}, "analytical_launch",
        "roofline" if config.compute_mode == "fast_mode" else config.compute_mode,
        _runtime_inputs(config, model, portable_fp8=portable_fp8), host_launch_profiles=host_profiles,
        execution_mode=config.execution_mode,
        # Older explicit snapshots provide eager Host intervals with the device timeline.
        materialize_timeline_details=host_profiles is not None,
    )
    if (not vendor_kimi and not portable_k3 and not portable_fp8 and config.fast_glm_dsa_score is None
            and not _uses_glm_batch_expert_program(config, model)
            and not _uses_portable_qwen_program(config, model)):
        validate_workload_emission_parallelism(deployment, model, allow_single_pool_tp2=True,
            allow_single_pool_ep2=True, allow_single_pool_tp1_pp=True, allow_single_pool_tp2_pp2=True,
            runtime_inputs=execution.runtime_inputs)
    if not with_scheduler:
        return {}, execution, storage_enabled

    if (config.compute_mode == "fast_mode" and model.attention_layout == "hybrid_kda_mla"
            and config.scheduler_adapter == "vllm_0_30"
            and config.deployment == "colocated"):
        from .factory import _fast_vllm030_with_byte_budget
        return {"main": _fast_vllm030_with_byte_budget(config, "main", model, hardware)}, execution, storage_enabled

    if config.scheduler_adapter == "vllm_0_30" and model.attention_layout == "hybrid_kda_mla":
        # The native allocator supplies hybrid per-group geometry. A single
        # byte-per-token rate cannot describe recurrent state and MLA pages.
        kv_bytes = 0
    elif config.pipeline_parallel_size == 1:
        kv_bytes = math.ceil(model.kv_bytes_per_token_per_parallel_rank(config.tensor_parallel_size, 1))
    else:
        kv_bytes = math.ceil(max(model.kv_bytes_per_token_for_layer_range(config.tensor_parallel_size, start, end)
                       for start, end in layer_ranges))
    backends = {}
    try:
        for pool, role in zip(pools, roles, strict=True):
            backend_config = BackendConfig(pool, role, config.max_batched_tokens, config.max_requests,
                config.num_kv_blocks, config.kv_block_size_tokens, kv_bytes, scheduler_pipeline_depth=1)
            backends[pool] = create_scheduler(
                config, backend_config, model, hardware, export_embedding_token_ids=_SOURCE_INPUTS)
    except BaseException:
        for backend in backends.values():
            backend.close()
        raise
    return backends, execution, storage_enabled


def _uses_glm_batch_expert_program(config, model):
    if (config.compute_mode != "fast_mode" or config.hardware_id != "nvidia_h200"
            or config.scheduler_adapter != "vllm_0_30" or config.deployment != "colocated"
            or config.data_parallel_size != 1 or config.sequence_parallel_size != 1
            or not config.enable_expert_parallel):
        return False
    from src.serving.workload.model_config import (
        GLM_MOE_DSA_5_3_CONFIG_SHA256, is_glm_fp8_tensor_model_scope,
    )
    # Fast Mode owns the rank program while the native runtime still checks
    # the TP geometry, PP partition, and the per-rank Scheduler capacity.
    return (model.config_sha256 == GLM_MOE_DSA_5_3_CONFIG_SHA256
            and is_glm_fp8_tensor_model_scope(model, config.tensor_parallel_size))


def _uses_portable_qwen_program(config, model):
    if config.compute_mode != "fast_mode" or config.hardware_id not in ("ascend_910b", "metax_c500"):
        return False
    # Preserve the vendor emitter for its existing precision and parallel
    # scope. The Fast Mode formulas only need the shared logical rank plan.
    if model.model_type == "qwen3_moe":
        return not (config.data_parallel_size == 1 and config.tensor_parallel_size == 2
                    and config.enable_expert_parallel)
    return (model.model_type == "qwen3" and model.weight_dtype == model.compute_dtype == "fp16"
            and (config.hardware_id == "ascend_910b" or config.pipeline_parallel_size > 1))


def _validate_batch_hardware(config, model, parallel):
    if config.fast_glm_dsa_score is not None:
        from src.serving.workload.model_config import (
            GLM_MOE_DSA_CONFIG_SHA256, is_glm_mtp_data_parallel_model_scope, is_glm_mtp_tensor_model_scope,
        )
        if (config.fast_glm_dsa_score != "bf16_timing_proxy" or config.compute_mode != "fast_mode"
                or config.hardware_id not in PORTABLE_BATCH_HARDWARE or config.scheduler_adapter != "vllm_0_30"
                or config.deployment != "colocated" or parallel.sequence_parallel_size != 1
                or parallel.context_parallel_size != 1 or model.config_sha256 != GLM_MOE_DSA_CONFIG_SHA256):
            raise ValueError("bf16_timing_proxy requires the supplied GLM-5.2 BF16 and non-H200 colocated Fast Mode with vLLM 0.30 and CP1/SP1")
        if parallel.data_parallel_size > 1:
            supported = parallel.enable_expert_parallel and is_glm_mtp_data_parallel_model_scope(model, parallel)
        else:
            supported = is_glm_mtp_tensor_model_scope(model, parallel.tensor_parallel_size,
                                                      expert_parallel=parallel.enable_expert_parallel)
        if not supported:
            raise ValueError("GLM-5.2 DSA timing proxy requires supported model sharding and explicit uniform WideEP for DP > 1")
        return
    if config.compute_mode != "fast_mode" or config.hardware_id not in PORTABLE_BATCH_HARDWARE:
        return
    if (config.scheduler_adapter != "vllm_0_30" or config.deployment != "colocated"
            or parallel.context_parallel_size != 1 or parallel.sequence_parallel_size != 1):
        raise ValueError(f"{config.hardware_id} fast_mode currently requires colocated CP1, SP1 and vllm_0_30")
    tp, pp = parallel.tensor_parallel_size, parallel.pipeline_parallel_size
    if supports_portable_fp8_reference(config, model, parallel):
        return
    if model.model_type == "kimi_linear":
        if not supports_portable_batch_k3(config, model, parallel):
            raise ValueError("portable Kimi-K3 requires the exact BF16/MXFP4 model and legal hybrid TP/PP and EP geometry")
        if not config.memory_policy_file:
            raise ValueError("portable Kimi-K3 Fast Mode requires an explicit per-Cell memory_policy_file for the BF16/W4A16 reference")
        return
    if model.model_type == "kimi_k2":
        if not supports_portable_batch_kimi(config, model, parallel):
            raise ValueError("portable Kimi-K2 requires legal group-32 INT4 sharding and explicit uniform EP for DP > 1")
        if config.hardware_id in KIMI_VENDOR_PROJECTION_HARDWARE and not config.memory_policy_file:
            raise ValueError("910B/C500 Kimi-K2 Fast Mode requires an explicit per-Cell memory_policy_file")
        return
    if model.model_type == "deepseek_v4":
        raise NotImplementedError(
            f"{config.hardware_id} has no supported DeepSeek-V4 Fast Mode profile for the pinned "
            "FP8 projections, FP4 expert residency and compressed FP8 DS-MLA KV layout"
        )
    if model.model_type == "glm_moe_dsa" and model.weight_dtype == "fp8":
        raise NotImplementedError(
            f"{config.hardware_id} has no supported block-FP8 GLM Fast Mode profile for the pinned "
            "E4M3 projections, experts and DSA score with the source KV layout"
        )
    bf16 = model.weight_dtype == model.compute_dtype == "bf16"
    fp16_qwen = model.model_type == "qwen3" and model.weight_dtype == model.compute_dtype == "fp16"
    if model.attention_layout != "standard_gqa" or not (bf16 or fp16_qwen):
        raise ValueError(f"{config.hardware_id} fast_mode has no reviewed forward profile for the model precision and attention")
    from src.serving.parallelism.tp import validate_gqa_parallelism
    from src.serving.workload.model_config import is_dense_parallel_model_scope, is_qwen3_moe_source_model_scope
    validate_gqa_parallelism(model, parallel)
    if parallel.is_rank_local and is_dense_parallel_model_scope(model, tp, pp) and not parallel.enable_expert_parallel:
        return
    if (is_qwen3_moe_source_model_scope(model) and model.hidden_act == "silu"
            and model.rope_scaling_type == "default"):
        if parallel.data_parallel_size > 1 and not parallel.enable_expert_parallel:
            raise ValueError("portable Qwen3 DP greater than one requires explicit WideEP")
        if parallel.enable_expert_parallel and model.num_experts % parallel.expert_parallel_size != 0:
            raise ValueError("portable Qwen3 EP requires even expert ownership")
        return
    raise ValueError("selected portable Dense or MoE model source does not support the TP/PP/EP layout")
