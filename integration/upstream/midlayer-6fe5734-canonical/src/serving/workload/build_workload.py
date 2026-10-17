"""Build a logical transformer DAG from one normalized ScheduleStep."""

import dataclasses
from dataclasses import dataclass
from typing import Iterable

from src.serving.deployment.deployment_plan import (
    DeploymentPlan,
    pd_kv_transfer_rank_pairs,
)
from src.serving.deployment.deployment_config import (
    worker_role_operator_group,
    worker_role_phase,
)
from src.serving.deployment.validated_scope import (
    is_validated_target_construction_scope,
)
from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.kernel.kernel_spec import PAGED_KV_CACHE_ATTRIBUTES
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.schedule.schedule_step import (
    ScheduledRequest,
    ScheduleStep,
    schedule_step_sha256,
)

from .model_config import (
    DEEPSEEK_V3_CONFIG_SHA256,
    DEEPSEEK_V3_L32_CONFIG_SHA256,
    GLM_MOE_DSA_CONFIG_SHA256,
    GLM_MOE_DSA_FP8_CONFIG_SHA256,
    KIMI_K2_CONFIG_SHA256,
    ModelConfig,
    is_validated_llama_tp2_model_scope,
    is_validated_qwen3_moe_ep2_model_scope,
    is_validated_qwen3_tp2_host_kv_model_scope,
)
from .rope_dispatch import (
    ROPE_PROGRAM_FUSED,
    ROPE_PROGRAM_NATIVE_COMPOSITE,
    modelled_rope_program,
    native_composite_rope_calls,
    rope_composite_semantics,
)
from .helper_launch import (
    RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED,
    ROUTER_MASK_RUN_CLASS,
    ROUTER_MASK_STRING_CALL_POINTS,
    boundary_logits_gather_source,
    helper_emission_count,
    moe_routing_mask_emissions,
    moe_routing_mask_string_complete,
    router_mask_source,
)
from .runtime_inputs import ExpertRoutingMemo, RuntimeInputs
from .shape_inference import infer_gemm_dimensions
from .workload_graph import (
    WorkloadBlockRepeat,
    WorkloadEdge,
    WorkloadGraph,
    WorkloadKind,
    WorkloadNode,
)


_LAUNCH_SELECTED_WORKLOAD_KINDS = {
    WorkloadKind.LINEAR,
    WorkloadKind.LM_HEAD,
    WorkloadKind.ROUTER,
    WorkloadKind.FLASH_ATTN,
    WorkloadKind.EAGER_ATTN,
    WorkloadKind.FUSED_MOE,
    WorkloadKind.NORM,
    WorkloadKind.ACTIVATION,
    WorkloadKind.ROTARY,
    WorkloadKind.ELEMENTWISE,
    WorkloadKind.REDUCE,
    WorkloadKind.GATHER,
    WorkloadKind.VECTOR,
    WorkloadKind.KV_UPDATE,
}


def _collective_ring_order(
    algorithm: str,
    participants: tuple[int, ...],
) -> tuple[int, ...]:
    if algorithm == "ring":
        return participants
    if algorithm in ("fc2", "mesh", "tree"):
        return ()
    raise ValueError(f"unsupported Serving collective algorithm: {algorithm!r}")


@dataclass(frozen=True)
class BlockExecutionSignature:
    """All construction inputs that can change one decoder block."""

    model_identity: int
    deployment_identity: int
    launch_policy: str
    phase: str
    token_count: int
    query_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    attention_kv_layout: str
    attention_kv_distribution: str
    graph_runtime_mode: str
    uses_moe: bool
    routing_policy_sha256: str | None
    routing_counts: tuple[int, ...] | None
    input_norm_has_residual: bool
    pipeline_stage: int
    previous_pipeline_stage: int | None
    previous_pool_name: str | None
    previous_rank_frontier_widths: tuple[int, ...]
    has_previous: bool
    has_residual_source: bool
    residual_rank_frontier_widths: tuple[int, ...]


@dataclass(frozen=True)
class _BlockTemplate:
    step_id: str
    prefix: str
    nodes: tuple[WorkloadNode, ...]
    edges: tuple[WorkloadEdge, ...]
    previous_before: str | None
    previous_after: str | None
    previous_rank_frontiers_before: tuple[tuple[str, ...], ...]
    previous_rank_frontiers_after: tuple[tuple[str, ...], ...]
    residual_source_before: str | None
    residual_source_after: str | None
    residual_rank_frontiers_before: tuple[tuple[str, ...], ...]
    residual_rank_frontiers_after: tuple[tuple[str, ...], ...]
    previous_pool_name_after: str | None
    previous_pipeline_stage_after: int | None
    transfer_count: int


@dataclass
class _BlockRunDraft:
    signature: BlockExecutionSignature
    block_id: str
    node_ids: tuple[str, ...]
    feedback_edges: tuple[WorkloadEdge, ...]
    state_after: tuple[object, ...]
    trace_node_ids: dict[str, list[str]]
    repeat_count: int = 1


@dataclass(frozen=True)
class TargetBlockExecutionSignature:
    """All inputs that can change one fixed-target decoder block."""

    model_identity: int
    deployment_identity: int
    launch_policy: str
    phase: str
    token_count: int
    request_phases: tuple[str, ...]
    query_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    attention_kv_layout: str
    attention_kv_distribution: str
    pipeline_stage: int
    block_kind: str
    attention_kind: str
    layer_weight_dtypes: tuple[str, ...]
    routing_policy_sha256: str | None
    routing_counts: tuple[int, ...] | None
    # SGLang flashinfer all-reduce + residual RMSNorm fusion at this layer:
    # (attention all-reduce fused, input norm absorbed, MLP all-reduce fused).
    fused_all_reduce: tuple[bool, bool, bool] = (False, False, False)


@dataclass(frozen=True)
class _TargetBlockTemplate:
    step_id: str
    prefix: str
    layer_index: int
    nodes: tuple[WorkloadNode, ...]
    edges: tuple[WorkloadEdge, ...]
    previous_before: str
    previous_after: str
    topk_source_before: str | None
    topk_source_after: str | None


class BlockConstructionMemo:
    """Reuse decoder block expansion templates within one simulation run."""

    def __init__(self) -> None:
        self._templates: dict[BlockExecutionSignature, _BlockTemplate] = {}
        self._target_templates: dict[
            TargetBlockExecutionSignature, _TargetBlockTemplate
        ] = {}
        self._hit_count = 0
        self._miss_count = 0

    @property
    def entry_count(self) -> int:
        return len(self._templates) + len(self._target_templates)

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def miss_count(self) -> int:
        return self._miss_count

    def template_for(
        self,
        signature: BlockExecutionSignature,
    ) -> _BlockTemplate | None:
        if not isinstance(signature, BlockExecutionSignature):
            raise TypeError(
                "block construction memo requires BlockExecutionSignature")
        if signature in self._templates:
            self._hit_count += 1
            return self._templates[signature]
        self._miss_count += 1
        return None

    def store(
        self,
        signature: BlockExecutionSignature,
        template: _BlockTemplate,
    ) -> None:
        if signature in self._templates:
            raise RuntimeError("block construction signature was stored twice")
        self._templates[signature] = template

    def target_template_for(
        self,
        signature: TargetBlockExecutionSignature,
    ) -> _TargetBlockTemplate | None:
        if not isinstance(signature, TargetBlockExecutionSignature):
            raise TypeError(
                "target block construction memo requires "
                "TargetBlockExecutionSignature")
        if signature in self._target_templates:
            self._hit_count += 1
            return self._target_templates[signature]
        self._miss_count += 1
        return None

    def store_target(
        self,
        signature: TargetBlockExecutionSignature,
        template: _TargetBlockTemplate,
    ) -> None:
        if signature in self._target_templates:
            raise RuntimeError(
                "target block construction signature was stored twice")
        self._target_templates[signature] = template


def _with_launch_policy(kind, attributes, policy):
    normalized = {} if attributes is None else dict(attributes)
    if kind in _LAUNCH_SELECTED_WORKLOAD_KINDS:
        if "launch_policy" in normalized and normalized["launch_policy"] != policy:
            raise ValueError(
                "workload node launch_policy disagrees with the requested "
                f"policy: {normalized['launch_policy']!r} != {policy!r}")
        normalized["launch_policy"] = policy
    return normalized


def _with_gemm_semantics(kind, attributes, tensor_parallel_size,
                         expert_parallel_size):
    """Attach the explicit row-major linear-algebra contract to GEMM nodes."""

    if kind not in (
            WorkloadKind.LINEAR, WorkloadKind.LM_HEAD,
            WorkloadKind.ROUTER):
        return attributes
    normalized = dict(attributes)
    if "has_bias" not in normalized or "position" not in normalized:
        raise ValueError(
            f"{kind.value} workload requires has_bias and position")
    has_bias = normalized["has_bias"]
    if not isinstance(has_bias, bool):
        raise TypeError(f"{kind.value} has_bias must be boolean")
    fixed = {
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "bias" if has_bias else "none",
        "tensor_parallel_size": tensor_parallel_size,
        "expert_parallel_size": expert_parallel_size,
    }
    for name, value in fixed.items():
        if name in normalized and normalized[name] != value:
            raise ValueError(
                f"{kind.value} workload {name} conflicts with the "
                f"row-major linear contract: {normalized[name]!r} != "
                f"{value!r}")
        normalized[name] = value
    return normalized


def _with_eager_attention_parallelism(
        kind, attributes, tensor_parallel_size, expert_parallel_size):
    if kind is not WorkloadKind.EAGER_ATTN:
        return attributes
    normalized = dict(attributes)
    normalized["tensor_parallel_size"] = tensor_parallel_size
    normalized["expert_parallel_size"] = expert_parallel_size
    return normalized


def is_validated_colocated_tp2_host_kv_scope(
    deployment: DeploymentPlan,
    model: ModelConfig,
) -> bool:
    """Return whether rank-local Host-KV fan-out is source-backed."""

    if not isinstance(deployment, DeploymentPlan):
        raise TypeError("TP2 Host-KV scope requires a DeploymentPlan")
    if not isinstance(model, ModelConfig):
        raise TypeError("TP2 Host-KV scope requires a ModelConfig")
    if deployment.mode != "colocated" or len(deployment.worker_pools) != 1:
        return False
    pool = next(iter(deployment.worker_pools.values()))
    parallelism = pool.parallel_plan.config
    return (
        pool.hardware_id == "nvidia_a800"
        and pool.ranks == (0, 1)
        and pool.kv_cache is not None
        and parallelism.worker_count == 2
        and parallelism.tensor_parallel_size == 2
        and parallelism.pipeline_parallel_size == 1
        and parallelism.expert_parallel_size == 1
        and parallelism.is_rank_local
        and len(pool.collectives) == 2
        and {item.operation for item in pool.collectives}
        == {"all_reduce", "all_gather"}
        and is_validated_qwen3_tp2_host_kv_model_scope(model)
    )


def is_validated_tp1_tp2_multi_pool_scope(
    deployment: DeploymentPlan,
    model: ModelConfig,
) -> bool:
    """Recognize dense TP1/TP2 pools and the matching TP2 MoE slice."""

    if not isinstance(deployment, DeploymentPlan):
        raise TypeError("TP1/TP2 multi-pool scope requires a DeploymentPlan")
    if not isinstance(model, ModelConfig):
        raise TypeError("TP1/TP2 multi-pool scope requires a ModelConfig")
    pools = tuple(deployment.worker_pools.values())
    if len(pools) < 2 or deployment.mode not in ("colocated", "pd_disagg"):
        return False
    if deployment.mode == "colocated":
        if any(pool.role != "colocated" for pool in pools):
            return False
    elif deployment.mode == "pd_disagg":
        phases = {worker_role_phase(pool.role) for pool in pools}
        if phases != {"prefill", "decode"}:
            return False
        kv_routes = tuple(
            route for route in deployment.routes if route.payload == "kv")
        if not kv_routes:
            return False
        for route in kv_routes:
            source = deployment.worker_pools[route.source_pool]
            destination = deployment.worker_pools[route.destination_pool]
            if source.hardware_id != destination.hardware_id:
                return False
            source_tp = source.parallel_plan.config.tensor_parallel_size
            destination_tp = (
                destination.parallel_plan.config.tensor_parallel_size)
            if (model.num_key_value_heads < max(source_tp, destination_tp)
                    or model.num_key_value_heads % source_tp
                    or model.num_key_value_heads % destination_tp):
                return False
    else:
        raise AssertionError("validated multi-pool deployment mode vanished")
    parallelisms = tuple(pool.parallel_plan.config for pool in pools)
    if not all(
            parallelism.tensor_parallel_size in (1, 2)
            and len(pool.ranks) == parallelism.tensor_parallel_size
            and parallelism.worker_count == parallelism.tensor_parallel_size
            and parallelism.pipeline_parallel_size == 1
            and parallelism.is_rank_local
            and {item.operation for item in pool.collectives} == (
                set() if parallelism.tensor_parallel_size == 1
                else {"all_reduce", "all_gather"})
            for pool, parallelism in zip(
                pools, parallelisms, strict=True)):
        return False
    dense_tp1_tp2 = (
        model.model_type == "qwen3"
        and model.attention_layout == "standard_gqa"
        and model.num_experts == 0
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
        and all(
            not parallelism.enable_expert_parallel
            for parallelism in parallelisms)
    )
    ep2 = (
        is_validated_qwen3_moe_ep2_model_scope(model)
        and all(parallelism.tensor_parallel_size == 2
                for parallelism in parallelisms)
        and all(
            parallelism.enable_expert_parallel
            and parallelism.expert_parallel_size == 2
            for parallelism in parallelisms)
    )
    return dense_tp1_tp2 or ep2


def is_validated_standalone_af_tp2_scope(
    deployment: DeploymentPlan,
    model: ModelConfig,
) -> bool:
    """Return whether the exact source-backed standalone AF+TP2 slice applies."""

    if not isinstance(deployment, DeploymentPlan):
        raise TypeError("AF+TP2 scope requires a DeploymentPlan")
    if not isinstance(model, ModelConfig):
        raise TypeError("AF+TP2 scope requires a ModelConfig")
    if deployment.mode != "af_disagg" or len(deployment.worker_pools) != 2:
        return False
    attention_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "attention"
        and worker_role_phase(pool.role) == "colocated"
    )
    ffn_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "ffn"
        and worker_role_phase(pool.role) == "colocated"
    )
    if len(attention_pools) != 1 or len(ffn_pools) != 1:
        return False
    pools = attention_pools + ffn_pools
    for pool in pools:
        parallelism = pool.parallel_plan.config
        if not (
            pool.hardware_id == "nvidia_a800"
            and len(pool.ranks) == 2
            and parallelism.worker_count == 2
            and parallelism.tensor_parallel_size == 2
            and parallelism.expert_parallel_size == 1
            and parallelism.pipeline_parallel_size == 1
            and parallelism.is_rank_local
            and {
                collective.operation for collective in pool.collectives
            } == {"all_reduce", "all_gather"}
            and len(pool.collectives) == 2
        ):
            return False
    attention_pool = attention_pools[0]
    ffn_pool = ffn_pools[0]
    if attention_pool.kv_cache is not None or ffn_pool.kv_cache is not None:
        return False
    expected_routes = {
        (attention_pool.name, ffn_pool.name),
        (ffn_pool.name, attention_pool.name),
    }
    if (
        len(deployment.routes) != 2
        or {
            (route.source_pool, route.destination_pool)
            for route in deployment.routes
        } != expected_routes
        or any(
            route.payload != "activation"
            or route.algorithm != "send_recv"
            or route.link != "nvlink"
            or route.topology != "point_to_point"
            for route in deployment.routes
        )
    ):
        return False
    return (
        (
            model.model_type == "qwen3"
            or is_validated_llama_tp2_model_scope(model)
        )
        and model.attention_layout == "standard_gqa"
        and model.num_experts == 0
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
    )


def is_validated_standalone_af_ep2_scope(
    deployment: DeploymentPlan,
    model: ModelConfig,
) -> bool:
    """Return whether the exact source-backed standalone AF+EP2 slice applies."""

    if not isinstance(deployment, DeploymentPlan):
        raise TypeError("AF+EP2 scope requires a DeploymentPlan")
    if not isinstance(model, ModelConfig):
        raise TypeError("AF+EP2 scope requires a ModelConfig")
    if deployment.mode != "af_disagg" or len(deployment.worker_pools) != 2:
        return False
    attention_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "attention"
        and worker_role_phase(pool.role) == "colocated"
    )
    ffn_pools = tuple(
        pool for pool in deployment.worker_pools.values()
        if worker_role_operator_group(pool.role) == "ffn"
        and worker_role_phase(pool.role) == "colocated"
    )
    if len(attention_pools) != 1 or len(ffn_pools) != 1:
        return False
    attention_pool = attention_pools[0]
    ffn_pool = ffn_pools[0]
    if (attention_pool.ranks != (0, 1)
            or ffn_pool.ranks != (2, 3)
            or attention_pool.kv_cache is not None
            or ffn_pool.kv_cache is not None):
        return False
    for pool, expected_ep_size in (
            (attention_pool, 1), (ffn_pool, 2)):
        parallelism = pool.parallel_plan.config
        if not (
            pool.hardware_id == "nvidia_a800"
            and len(pool.ranks) == 2
            and parallelism.worker_count == 2
            and parallelism.tensor_parallel_size == 2
            and parallelism.expert_parallel_size == expected_ep_size
            and parallelism.pipeline_parallel_size == 1
            and parallelism.is_rank_local
            and {
                collective.operation for collective in pool.collectives
            } == {"all_reduce", "all_gather"}
            and len(pool.collectives) == 2
        ):
            return False
    expected_routes = {
        (attention_pool.name, ffn_pool.name),
        (ffn_pool.name, attention_pool.name),
    }
    if (
        len(deployment.routes) != 2
        or {
            (route.source_pool, route.destination_pool)
            for route in deployment.routes
        } != expected_routes
        or any(
            route.payload != "activation"
            or route.algorithm != "send_recv"
            or route.link != "nvlink"
            or route.topology != "point_to_point"
            for route in deployment.routes
        )
    ):
        return False
    return (
        model.architecture == "Qwen3MoeForCausalLM"
        and model.model_type == "qwen3_moe"
        and model.attention_layout == "standard_gqa"
        and model.hidden_size == 128
        and model.intermediate_size == 256
        and model.moe_intermediate_size == 128
        and model.num_attention_heads == 2
        and model.num_key_value_heads == 2
        and model.num_hidden_layers == 2
        and model.head_dim == 64
        and model.vocab_size == 128
        and model.max_position_embeddings == 128
        and model.num_experts == 2
        and model.num_experts_per_tok == 1
        and model.num_dense_layers == 0
        and model.num_shared_experts == 0
        and model.num_mtp_layers == 0
        and model.mtp_architecture == "none"
        and model.routed_scaling_factor == 1.0
        and not model.tie_word_embeddings
        and not model.attention_bias
        and not model.mlp_bias
        and model.qk_norm
        and model.compute_dtype == "bf16"
        and model.weight_dtype == "bf16"
        and model.weight_group_size == 0
        and model.quantized_weight_targets == ()
        and model.quantized_weight_ignore == ()
    )


def tp_collective_payload_bytes(
    model: ModelConfig,
    operation: str,
    token_count: int,
    *,
    local_output_width: int | None = None,
) -> int:
    """Return canonical per-rank TP collective input bytes."""

    if not isinstance(model, ModelConfig):
        raise TypeError("TP collective bytes require ModelConfig")
    if isinstance(token_count, bool) or not isinstance(token_count, int):
        raise TypeError("TP collective token_count must be an integer")
    if token_count <= 0:
        raise ValueError("TP collective token_count must be positive")
    if operation == "all_reduce":
        if local_output_width is not None:
            raise ValueError(
                "TP all-reduce derives output width from model.hidden_size")
        output_width = model.hidden_size
    elif operation == "all_gather":
        if (isinstance(local_output_width, bool)
                or not isinstance(local_output_width, int)
                or local_output_width <= 0):
            raise ValueError(
                "TP all-gather requires a positive local_output_width")
        output_width = local_output_width
    else:
        raise ValueError(
            f"unsupported TP collective operation: {operation!r}")
    return token_count * output_width * model.dtype_bytes


def routed_expert_local_intermediate(
    model: ModelConfig,
    tensor_parallel_size: int,
    expert_parallel_size: int,
) -> int:
    """Return one rank's routed-expert intermediate extent.

    The residency contract (src/serving/kv_cache/model_residency.py) places
    N/EP experts on each rank and tensor-shards every expert linear across
    the TP group, so one rank's GEMM n/k for a single expert spans
    intermediate * EP / TP: the full extent when EP == TP, a TP shard when
    EP == 1.  The forward FUSED_MOE shape must quote the same per-rank
    extent or every rank would be charged the whole unsharded expert weight.
    """

    product = model.moe_intermediate_size * expert_parallel_size
    if product % tensor_parallel_size:
        raise ValueError(
            "routed-expert intermediate extent does not divide across the "
            f"TP/EP deployment: {model.moe_intermediate_size} * "
            f"{expert_parallel_size} % {tensor_parallel_size} != 0")
    return product // tensor_parallel_size


def build_pd_kv_transfer_workload(
    request_id: str,
    context_tokens: int,
    model: ModelConfig,
    deployment: DeploymentPlan,
    source_pool_name: str,
    destination_pool_name: str,
) -> WorkloadGraph:
    """Build one request-scoped P-to-D KV transfer after routing."""

    if not isinstance(request_id, str) or not request_id:
        raise ValueError("KV transfer request_id must be non-empty")
    if isinstance(context_tokens, bool) or not isinstance(context_tokens, int):
        raise TypeError("KV transfer context_tokens must be an integer")
    if context_tokens <= 0:
        raise ValueError("KV transfer context_tokens must be positive")
    if deployment.mode not in ("pd_disagg", "pd_af_disagg"):
        raise ValueError("KV transfer workload requires a PD deployment")
    if source_pool_name not in deployment.worker_pools:
        raise ValueError(f"unknown KV source pool {source_pool_name!r}")
    if destination_pool_name not in deployment.worker_pools:
        raise ValueError(
            f"unknown KV destination pool {destination_pool_name!r}")
    source = deployment.worker_pools[source_pool_name]
    destination = deployment.worker_pools[destination_pool_name]
    if worker_role_phase(source.role) != "prefill":
        raise ValueError("KV transfer source must have a prefill role")
    if worker_role_phase(destination.role) != "decode":
        raise ValueError("KV transfer destination must have a decode role")
    if (model.model_type in ("glm_moe_dsa", "deepseek_v3")
            and source.parallel_plan.config.tensor_parallel_size
                != destination.parallel_plan.config.tensor_parallel_size):
        raise NotImplementedError(
            "heterogeneous PD tensor parallel KV transfer has no explicit "
            "payload contract; residency is not a transfer-size substitute")
    if model.attention_layout == "dsa_mla":
        source_parallel = source.parallel_plan.config
        destination_parallel = destination.parallel_plan.config
        # The DSA transfer payload is EP-independent: the compressed MLA
        # ``kv_c||k_pe`` vector and the FP8 index cache are both
        # num_kv_heads=1 caches replicated on every TP rank and transferred
        # REPLICATE-style per TP rank pair, while expert parallelism only
        # shards MoE expert weights and never enters the KV/index-cache path
        # (pinned vLLM GLM-5.2/DeepSeek-V3.2 semantics: MLAAttentionSpec
        # num_kv_heads=1, indexer cache 132 B/token per full-indexer layer,
        # NIXL "cache is duplicated" rank mapping).  EP8 and EP1 therefore
        # share the identical per-rank payload and identity rank mapping.
        tp8_ep8 = (
            source_parallel.enable_expert_parallel
            and destination_parallel.enable_expert_parallel
            and source_parallel.expert_parallel_size == 8
            and destination_parallel.expert_parallel_size == 8
        )
        tp8_ep1 = (
            not source_parallel.enable_expert_parallel
            and not destination_parallel.enable_expert_parallel
            and source_parallel.expert_parallel_size == 1
            and destination_parallel.expert_parallel_size == 1
        )
        supported_pd_scope = (
            source_parallel.tensor_parallel_size == 8
            and destination_parallel.tensor_parallel_size == 8
            and (tp8_ep8 or tp8_ep1)
        )
        if not supported_pd_scope:
            raise NotImplementedError(
                "DSA PD transfer requires a combined compressed MLA and "
                "FP8 index-cache transfer contract")
    elif model.model_type == "deepseek_v3":
        source_parallel = source.parallel_plan.config
        destination_parallel = destination.parallel_plan.config
        # DeepSeek-V3 plain-MLA PD transfer payload: the compressed MLA
        # ``kv_c||k_pe`` vector is a num_kv_heads=1 cache replicated on
        # every TP rank and transferred REPLICATE-style per TP rank pair,
        # exactly like the DSA MLA subset above but without the GLM-5.2
        # FP8 index cache (DeepSeek-V3 has no DSA indexer).  Expert
        # parallelism only shards MoE expert weights and never enters the
        # KV path (pinned vLLM DeepSeek-V3 MLAAttentionSpec num_kv_heads=1,
        # NIXL "cache is duplicated" rank mapping); the validated
        # DeepSeek-V3 scope is the H200 InferenceX TP8/PP1/EP1 topology.
        supported_pd_scope = (
            source_parallel.tensor_parallel_size == 8
            and destination_parallel.tensor_parallel_size == 8
            and not source_parallel.enable_expert_parallel
            and not destination_parallel.enable_expert_parallel
            and source_parallel.expert_parallel_size == 1
            and destination_parallel.expert_parallel_size == 1
        )
        if not supported_pd_scope:
            raise NotImplementedError(
                "DeepSeek-V3 PD transfer requires a compressed MLA "
                "kv_c||k_pe transfer contract over TP8/EP1 pools")
    rank_pairs = pd_kv_transfer_rank_pairs(
        source, destination, model.num_key_value_heads)
    routes = [
        route for route in deployment.routes
        if route.payload == "kv"
        and route.source_pool == source_pool_name
        and route.destination_pool == destination_pool_name
    ]
    if len(routes) != 1:
        raise ValueError(
            "request-scoped KV transfer requires exactly one route from "
            f"{source_pool_name!r} to {destination_pool_name!r}")
    route = routes[0]
    source_tensor_parallel_size = (
        source.parallel_plan.config.tensor_parallel_size)
    destination_tensor_parallel_size = (
        destination.parallel_plan.config.tensor_parallel_size)
    base_node_id = (
        f"{request_id}:pd-kv-transfer:"
        f"{source_pool_name}-to-{destination_pool_name}"
    )
    nodes = []
    for edge_index, rank_pair in enumerate(rank_pairs):
        source_rank = rank_pair.source_rank
        destination_rank = rank_pair.destination_rank
        overlapping_kv_heads = (
            rank_pair.kv_head_end - rank_pair.kv_head_start)
        if model.attention_layout == "standard_gqa":
            bytes_per_token = (
                2 * model.num_hidden_layers * overlapping_kv_heads
                * model.head_dim * model.dtype_bytes)
        else:
            bytes_per_token = model.kv_bytes_per_token_per_parallel_rank(
                source_tensor_parallel_size,
                source.parallel_plan.config.pipeline_parallel_size)
        node_id = (
            base_node_id
            if len(rank_pairs) == 1
            else f"{base_node_id}:kv-edge-{edge_index}"
        )
        nodes.append(WorkloadNode(
            node_id=node_id,
            kind=WorkloadKind.PD_KV_TRANSFER,
            pool_name=source_pool_name,
            shape={},
            rank=source_rank,
            attributes={
                "algorithm": route.algorithm,
                "bytes": context_tokens * bytes_per_token,
                "bytes_semantics": "payload",
                "backend": route.backend,
                "backend_config_sha256": route.backend_config_sha256,
                "direction": "point_to_point",
                "link": route.link,
                "topology": route.topology,
                "participants": (source_rank, destination_rank),
                "operation": "send_recv",
                "source_rank": source_rank,
                "destination_rank": destination_rank,
                "ring_order": (),
                "reduction": None,
                "resource_id": route.resource_id,
                "destination_pool": destination_pool_name,
                "rank_mapping": "kv_head_overlap",
                "source_logical_rank": rank_pair.source_logical_rank,
                "destination_logical_rank": (
                    rank_pair.destination_logical_rank),
                "source_tensor_parallel_size": source_tensor_parallel_size,
                "destination_tensor_parallel_size": (
                    destination_tensor_parallel_size),
                "kv_head_start": rank_pair.kv_head_start,
                "kv_head_end": rank_pair.kv_head_end,
                "num_key_value_heads": model.num_key_value_heads,
            },
        ))
    return WorkloadGraph(tuple(nodes))


def _build_target_model_logical_workload(
    step: ScheduleStep,
    model: ModelConfig,
    deployment: DeploymentPlan,
    *,
    launch_policy: str,
    kv_cache_events: tuple[KVCacheEvent, ...],
    runtime: RuntimeInputs,
    routing_memo: ExpertRoutingMemo | None,
    block_memo: BlockConstructionMemo | None,
) -> WorkloadGraph:
    """Build the rank-local graph for the two fixed target configurations.

    Kimi-K2.6 and GLM-5.2 execution semantics are pinned to audited vLLM
    sources.  Unsupported physical kernels remain explicit at the lowering
    and prediction boundary instead of being replaced with another operator.
    """

    fixed_contracts = {
        "kimi_k2": {
            "model_id": "moonshotai/Kimi-K2.6",
            "model_revision": "7eb5002f6aadc958aed6a9177b7ed26bb94011bb",
            "config_sha256": KIMI_K2_CONFIG_SHA256,
            "attention_layout": "mla",
            "runtime_kv_layout": "mla",
            "contract": "kimi_k2_6_vllm_0_27_1_logical_v1",
        },
        "glm_moe_dsa": {
            "model_id": "zai-org/GLM-5.2",
            "model_revision": "b4734de4facf877f85769a911abafc5283eab3d9",
            "config_sha256": (
                GLM_MOE_DSA_CONFIG_SHA256,
                GLM_MOE_DSA_FP8_CONFIG_SHA256,
            ),
            "attention_layout": "dsa_mla",
            "runtime_kv_layout": "mla",
            "contract": "glm_5_2_vllm_0_27_1_logical_v1",
        },
        "deepseek_v3": {
            "model_id": "deepseek-ai/DeepSeek-V3",
            "model_revision": "e815299b0bcbac849fa540c768ef21845365c9eb",
            "config_sha256": (
                DEEPSEEK_V3_CONFIG_SHA256,
                DEEPSEEK_V3_L32_CONFIG_SHA256,
            ),
            "attention_layout": "mla",
            "runtime_kv_layout": "mla",
            "contract": "deepseek_v3_vllm_0_27_1_logical_v1",
        },
    }
    if model.model_type not in fixed_contracts:
        raise ValueError(
            "target logical workload requires Kimi-K2.6, GLM-5.2, or "
            "DeepSeek-V3")
    if model.target_semantics is None:
        raise ValueError(
            "target logical workload requires source-pinned model semantics")
    contract = fixed_contracts[model.model_type]
    actual_identity = (
        model.model_id,
        model.model_revision,
        model.config_sha256,
        model.attention_layout,
        model.target_semantics.workload_contract,
    )
    expected_identity = (
        contract["model_id"],
        contract["model_revision"],
        contract["config_sha256"],
        contract["attention_layout"],
        contract["contract"],
    )
    identity_without_config_sha = (
        actual_identity[0],
        actual_identity[1],
        actual_identity[3],
        actual_identity[4],
    )
    expected_without_config_sha = (
        expected_identity[0],
        expected_identity[1],
        expected_identity[3],
        expected_identity[4],
    )
    allowed_config_sha256 = contract["config_sha256"]
    if isinstance(allowed_config_sha256, str):
        allowed_config_sha256 = (allowed_config_sha256,)
    if (identity_without_config_sha != expected_without_config_sha
            or model.config_sha256 not in allowed_config_sha256):
        raise ValueError(
            "target logical workload requires the fixed official model "
            f"identity: actual={actual_identity!r}, "
            f"expected={expected_identity!r}")
    if launch_policy != "analytical_launch":
        if launch_policy == "injected_launch":
            raise NotImplementedError(
                "Kimi-K2.6 and GLM-5.2 currently support only "
                "analytical_launch; injected launch observations are not "
                "available")
        raise ValueError(
            f"unknown launch_policy {launch_policy!r}; expected "
            "analytical_launch")
    if deployment.mode not in ("colocated", "pd_disagg"):
        raise NotImplementedError(
            "target-model logical workload requires colocated or PD pools")
    if deployment.mode == "colocated" and len(deployment.worker_pools) != 1:
        raise NotImplementedError(
            "target-model colocated workload requires exactly one worker pool")
    if step.pool_name not in deployment.worker_pools:
        raise ValueError(
            f"schedule step references unknown pool {step.pool_name!r}")
    pool = deployment.worker_pools[step.pool_name]
    parallelism = pool.parallel_plan.config
    # The validated target scopes are one shared table (DeepSeek-V3 has no
    # source-backed A800 production layout; its validated production scopes
    # are the H200 TP8/PP1/EP1 colocated and PD rows).  The pool-level rank
    # count and rank locality stay as this gate's own structural checks.
    target_scope = is_validated_target_construction_scope(
        model,
        deployment_mode=deployment.mode,
        hardware_id=pool.hardware_id,
        tensor_parallel_size=parallelism.tensor_parallel_size,
        pipeline_parallel_size=parallelism.pipeline_parallel_size,
        expert_parallel=parallelism.enable_expert_parallel,
        expert_parallel_size=parallelism.expert_parallel_size,
    )
    if (len(pool.ranks) != parallelism.worker_count
            or not target_scope
            or not parallelism.is_rank_local):
        raise NotImplementedError(
            "target-model logical workload requires the "
            "TP=PP=EP=SP=CP=DP=1 fixture "
            "scope, TP=EP=8 with PP2 for Kimi-K2.6 / PP4 for GLM-5.2, "
            "H200 GLM-5.2 TP8/PP1/EP1 aggregate, H200 GLM-5.2 PD "
            "TP8/PP1/EP1, H200 Kimi-K2.6 colocated TP8/PP1/EP8, "
            "H200 DeepSeek-V3 TP8/PP1/EP1 aggregate, or H200 DeepSeek-V3 "
            "PD TP8/PP1/EP1, and SP=CP=DP=1")
    layer_ranges = pipeline_layer_ranges_for_model(
        model, parallelism.pipeline_parallel_size)
    expected_layer_stages = tuple(
        stage
        for stage, (layer_start, layer_end) in enumerate(layer_ranges)
        for unused_layer in range(layer_start, layer_end)
    )
    if pool.parallel_plan.layer_to_pipeline_stage != expected_layer_stages:
        raise RuntimeError(
            "target-model pipeline placement disagrees with pinned vLLM")
    for stage in range(parallelism.pipeline_parallel_size):
        if (len(pool.physical_ranks_for_pipeline_stage(stage))
                != parallelism.tensor_parallel_size):
            raise RuntimeError(
                "target PP stage must map to exactly one TP rank group")
    if any(
            event.action in (KVCacheAction.LOAD, KVCacheAction.EVICT)
            for event in kv_cache_events):
        raise NotImplementedError(
            "target-model logical workload does not yet support KV offload")
    if step.attention_kv_layout is None:
        attention_kv_layout = runtime.attention_kv_layout
    elif step.attention_kv_layout == runtime.attention_kv_layout:
        attention_kv_layout = step.attention_kv_layout
    else:
        raise ValueError(
            "scheduler attention KV layout disagrees with RuntimeInputs: "
            f"scheduler={step.attention_kv_layout!r}, "
            f"runtime={runtime.attention_kv_layout!r}")
    expected_kv_layout = contract["runtime_kv_layout"]
    if attention_kv_layout != expected_kv_layout:
        raise NotImplementedError(
            f"{contract['model_id']} logical workload requires "
            f"attention_kv_layout={expected_kv_layout!r}, got "
            f"{attention_kv_layout!r}")
    if runtime.attention_kv_distribution != "uniform":
        raise NotImplementedError(
            "target-model logical attention currently requires a uniform "
            "KV distribution assumption")
    if runtime.expert_routing.source_kind == "not_applicable":
        raise ValueError(
            "target MoE workload requires explicit expert routing input")
    if model.target_semantics.use_cache is not True:
        raise ValueError(
            "target logical workload requires the fixed cache-enabled model "
            "contract")
    step_uses_mtp = any(
            any((
                request.mtp_proposed_tokens,
                request.mtp_verified_tokens,
                request.mtp_accepted_tokens,
                request.mtp_proposer_tokens,
                request.mtp_next_proposed_tokens,
                request.mtp_proposer_mode != "disabled",
                request.mtp_acceptance_source_kind != "disabled",
                request.mtp_acceptance_source_id != "disabled",
            ))
            for request in step.requests)
    if step_uses_mtp:
        if model.model_type == "glm_moe_dsa":
            if (model.mtp_architecture != "glm_moe_dsa_mtp"
                    or model.num_mtp_layers != 1
                    or not model.index_share_for_mtp_iteration):
                raise NotImplementedError(
                    "GLM-5.2 MTP requires one glm_moe_dsa_mtp layer with "
                    "index_share_for_mtp_iteration enabled")
        elif model.model_type == "deepseek_v3":
            if (model.mtp_architecture != "deepseek_mtp"
                    or model.num_mtp_layers != 1):
                raise NotImplementedError(
                    "DeepSeek-V3 MTP requires its single source-backed "
                    "deepseek_mtp NextN layer")
        else:
            raise NotImplementedError(
                "Kimi-K2.6 fixed text config declares zero MTP layers")

    phases = {request.phase for request in step.requests}
    if phases == {"prefill"}:
        phase = "prefill"
    elif phases == {"decode"}:
        phase = "decode"
    elif phases == {"prefill", "decode"}:
        phase = "mixed"
    else:
        raise ValueError(
            f"unknown workload request phases: {sorted(phases)}")
    token_count = step.token_count
    query_lengths = tuple(
        request.scheduled_tokens for request in step.requests)
    kv_lengths = tuple(
        request.computed_tokens + request.scheduled_tokens
        for request in step.requests)
    if step_uses_mtp and any(
            request.phase == "decode"
            and request.recomputed_tokens == 0
            and request.scheduled_tokens != request.mtp_verified_tokens + 1
            for request in step.requests):
        raise ValueError(
            "MTP target verification must schedule one terminal token after "
            "the verified draft tokens")
    if sum(query_lengths) != token_count:
        raise ValueError(
            "target-model query lengths must conserve scheduled tokens")
    if any(length <= 0 for length in query_lengths + kv_lengths):
        raise ValueError(
            "target-model attention lengths must be positive")

    nodes = []
    edges = []
    block_repeats = []
    previous = None
    pending_stage_entry_predecessors = ()
    active_pipeline_stage = 0
    rank = pool.physical_ranks_for_pipeline_stage(active_pipeline_stage)[0]
    tp_size = parallelism.tensor_parallel_size
    ep_size = parallelism.expert_parallel_size
    active_token_count = token_count
    step_sha256 = schedule_step_sha256(step)
    topk_source = None

    def enter_pipeline_stage(next_stage, next_layer_index):
        nonlocal active_pipeline_stage, previous, rank, topk_source
        nonlocal pending_stage_entry_predecessors
        if next_stage == active_pipeline_stage:
            return
        if next_stage != active_pipeline_stage + 1:
            raise RuntimeError(
                "target-model pipeline stages must advance contiguously")
        if previous is None:
            raise RuntimeError("target-model pipeline boundary lacks input")
        if pending_stage_entry_predecessors:
            raise RuntimeError(
                "target-model pipeline boundary has unconsumed stage inputs")
        source_frontier = previous
        config = pool.pipeline_p2p_for(active_pipeline_stage, next_stage)
        source_ranks = pool.physical_ranks_for_pipeline_stage(
            active_pipeline_stage)
        destination_ranks = pool.physical_ranks_for_pipeline_stage(next_stage)
        if len(source_ranks) != tp_size or len(destination_ranks) != tp_size:
            raise RuntimeError("target PP boundary lost a TP rank group")
        total_tensor_elements = token_count * model.hidden_size
        if total_tensor_elements % tp_size:
            raise ValueError("target PP activation must divide by TP size")
        wire_elements = total_tensor_elements // tp_size
        wire_bytes = wire_elements * model.dtype_bytes
        all_gather = pool.collective_for("all_gather")
        boundary_collective_ids = []
        for tensor_name in ("hidden_states", "residual"):
            transfer_ids = []
            for lane, (source_rank, destination_rank) in enumerate(zip(
                    source_ranks, destination_ranks, strict=True)):
                transfer_id = (
                    f"{step.step_id}:pp-{active_pipeline_stage}-to-"
                    f"{next_stage}:{tensor_name.replace('_', '-')}:"
                    f"tp-rank-{lane}")
                nodes.append(WorkloadNode(
                    transfer_id,
                    WorkloadKind.P2P,
                    pool.name,
                    {},
                    rank=source_rank,
                    attributes={
                        "algorithm": config.algorithm,
                        "bytes": wire_bytes,
                        "bytes_semantics": "payload",
                        "backend": config.backend,
                        "backend_config_sha256": config.backend_config_sha256,
                        "direction": "point_to_point",
                        "link": config.link,
                        "topology": config.topology,
                        "participants": (source_rank, destination_rank),
                        "operation": "send_recv",
                        "source_rank": source_rank,
                        "destination_rank": destination_rank,
                        "ring_order": (),
                        "reduction": None,
                        "resource_id": config.resource_id,
                        "source_stage": active_pipeline_stage,
                        "destination_stage": next_stage,
                        "payload_tensor": {
                            "name": tensor_name,
                            "shape": (wire_elements,),
                            "dtype": model.compute_dtype,
                            "bytes": wire_bytes,
                        },
                    },
                ))
                edges.append(WorkloadEdge(
                    source_frontier, transfer_id, "ready"))
                transfer_ids.append(transfer_id)
            collective_id = (
                f"{step.step_id}:pp-{active_pipeline_stage}-to-{next_stage}:"
                f"{tensor_name.replace('_', '-')}:tp-all-gather")
            nodes.append(WorkloadNode(
                collective_id,
                WorkloadKind.COLLECTIVE,
                pool.name,
                {},
                rank=destination_ranks[0],
                attributes={
                    "algorithm": all_gather.algorithm,
                    "bytes": wire_bytes,
                    "bytes_semantics": "per_rank_input",
                    "backend": all_gather.backend,
                    "backend_config_sha256": (
                        all_gather.backend_config_sha256),
                    "direction": "collective",
                    "link": all_gather.link,
                    "topology": all_gather.topology,
                    "participants": destination_ranks,
                    "operation": "all_gather",
                    "source_rank": None,
                    "destination_rank": None,
                    "ring_order": _collective_ring_order(
                        all_gather.algorithm, destination_ranks),
                    "reduction": None,
                    "resource_id": all_gather.resource_id,
                },
            ))
            for transfer_id in transfer_ids:
                edges.append(WorkloadEdge(
                    transfer_id, collective_id, "barrier"))
            boundary_collective_ids.append(collective_id)
        previous = boundary_collective_ids[0]
        pending_stage_entry_predecessors = tuple(
            boundary_collective_ids[1:])
        if (model.model_type == "glm_moe_dsa"
                and model.indexer_types[next_layer_index] != "full"):
            raise RuntimeError(
                "GLM pipeline stages must begin with a full local DSA "
                "indexer; top-k buffers are not pipeline payloads")
        topk_source = None
        active_pipeline_stage = next_stage
        rank = destination_ranks[0]

    # sglang deepseek_v2.py MoEGate.forward: above the DSV3 router kernel's
    # 16 tokens a CUDA batch runs linear_bf16_fp32 (torch.mm, FP32 output).
    sglang_router_runner = (
        {"router_linear_runner": "sglang_linear_bf16_fp32_cublas"}
        if (runtime.serving_framework == "sglang"
            and model.model_type == "deepseek_v3") else {})
    if model.model_type in ("kimi_k2", "deepseek_v3"):
        dense_activation_source = "DeepseekV2MLP.forward"
        dense_down_source = "vllm.DeepseekV2MLP.down_proj"
        moe_router_source = "vllm.DeepseekV2MoE.gate"
        moe_router_dtype = model.target_semantics.moe_router_dtype
        moe_router_weight_dtype = model.compute_dtype
        moe_expert_source = "vllm.FusedMoE.forward"
        moe_shared_add_source = "DeepseekV2MoE.forward"
        input_norm_source = "DeepseekV2DecoderLayer.input_layernorm"
        rotary_source = "vllm.DeepseekV2MLAAttention.rotary_emb"
        output_projection_source = "vllm.DeepseekV2MLAAttention.o_proj"
        post_attention_norm_source = (
            "DeepseekV2DecoderLayer.post_attention_layernorm")
        final_norm_source = "DeepseekV2Model.norm"
        lm_head_source = "vllm.DeepseekV3ForCausalLM.compute_logits"
        compressed_cache_variant = "mla_kv_cache"
    elif model.model_type == "glm_moe_dsa":
        dense_activation_source = "vllm.DeepseekV2MLP.forward"
        dense_down_source = "vllm.DeepseekV2MLP.down_proj"
        moe_router_source = "vllm.DeepseekV2MoE.gate"
        moe_router_dtype = model.target_semantics.moe_router_dtype
        moe_router_weight_dtype = model.compute_dtype
        moe_expert_source = "vllm.FusedMoE.forward"
        moe_shared_add_source = "vllm.DeepseekV2MoE.forward"
        input_norm_source = "vllm.DeepseekV2DecoderLayer.input_layernorm"
        rotary_source = "vllm.MultiHeadLatentAttentionWrapper.rotary_emb"
        output_projection_source = "vllm.DeepseekV2MLAAttention.o_proj"
        post_attention_norm_source = (
            "vllm.DeepseekV2DecoderLayer.post_attention_layernorm")
        final_norm_source = "vllm.DeepseekV2Model.norm"
        lm_head_source = "vllm.GlmMoeDsaForCausalLM.compute_logits"
        compressed_cache_variant = "mla_kv_cache"
    else:
        raise AssertionError(
            f"unhandled target model type: {model.model_type!r}")

    def weight_group_size(weight_dtype):
        if weight_dtype == "int4":
            return model.weight_group_size
        if weight_dtype in (model.compute_dtype, "fp8", "fp32"):
            return 0
        raise ValueError(
            f"unsupported target-model module weight dtype: {weight_dtype!r}")

    def rope_attributes(source_symbol, rope_layout):
        semantics = model.target_semantics
        if semantics is None:
            raise AssertionError("target RoPE requires target model semantics")
        return {
            "layout": f"{rope_layout}_{semantics.rope_type}",
            "rope_layout": rope_layout,
            "rope_type": semantics.rope_type,
            "rope_theta": semantics.rope_theta,
            "rope_factor": semantics.rope_factor,
            "rope_original_max_position_embeddings": (
                semantics.rope_original_max_position_embeddings),
            "rope_beta_fast": semantics.rope_beta_fast,
            "rope_beta_slow": semantics.rope_beta_slow,
            "rope_mscale": semantics.rope_mscale,
            "rope_mscale_all_dim": semantics.rope_mscale_all_dim,
            "source_symbol": source_symbol,
        }

    def append(
        kind,
        suffix,
        shape,
        *,
        variant,
        attributes,
        dtype=None,
        extra_predecessors=(),
        use_previous=True,
    ):
        nonlocal previous, pending_stage_entry_predecessors
        attributes = _with_launch_policy(kind, attributes, launch_policy)
        attributes = _with_gemm_semantics(kind, attributes, tp_size, ep_size)
        attributes = _with_eager_attention_parallelism(
            kind, attributes, tp_size, ep_size)
        node_id = f"{step.step_id}:{suffix}"
        node = WorkloadNode(
            node_id,
            kind,
            pool.name,
            shape,
            dtype=model.compute_dtype if dtype is None else dtype,
            variant=variant,
            rank=rank,
            attributes=attributes,
        )
        nodes.append(node)
        predecessors = []
        if use_previous and previous is not None:
            predecessors.append(previous)
            predecessors.extend(pending_stage_entry_predecessors)
            pending_stage_entry_predecessors = ()
        for predecessor in extra_predecessors:
            if predecessor not in predecessors:
                predecessors.append(predecessor)
        for predecessor in predecessors:
            edges.append(WorkloadEdge(predecessor, node_id))
        previous = node_id
        return node_id

    # SGLang 0.5.12 layers/communicator.py: with the auto-enabled flashinfer
    # all-reduce fusion (both InferenceX server logs) and batch_size <=
    # FUSE_ALLREDUCE_MAX_BATCH_SIZE (2048), the attention all-reduce is fused
    # with post_attention_layernorm, and the MLP/MoE all-reduce with the next
    # layer's input_layernorm except on the last layer of its module
    # (should_fuse_mlp_allreduce_with_next_layer).  Above 2048 tokens SGLang
    # runs the unfused all-reduce and RMSNorm.
    fused_all_reduce_suffixes = set()

    def sglang_fused_all_reduce(layer_index, block_token_count):
        if (runtime.serving_framework != "sglang" or tp_size == 1
                or model.model_type != "deepseek_v3"
                or block_token_count > 2048):
            return (False, False, False)
        target_layer = layer_index < model.num_hidden_layers
        return (
            True,
            target_layer and layer_index >= 1,
            target_layer and layer_index < model.num_hidden_layers - 1,
        )

    def append_tp_collective(
            suffix, operation, *, predecessors=(), payload_bytes=None):
        nonlocal previous
        if tp_size == 1:
            incoming = tuple(predecessors)
            if incoming:
                previous = incoming[-1]
            if previous is None:
                raise RuntimeError(
                    "TP1 collective elision requires an input predecessor")
            return previous
        collective = pool.collective_for(operation)
        participants = pool.physical_ranks_for_pipeline_stage(
            active_pipeline_stage)
        if len(participants) != tp_size:
            raise RuntimeError("target collective lost its TP rank group")
        if operation == "all_reduce":
            reduction = "sum"
        elif operation == "all_gather":
            reduction = None
        else:
            raise ValueError(
                f"unsupported target tensor collective: {operation!r}")
        node_id = f"{step.step_id}:{suffix}"
        if payload_bytes is None:
            payload_bytes = (
                active_token_count * model.hidden_size * model.dtype_bytes)
        if (isinstance(payload_bytes, bool)
                or not isinstance(payload_bytes, int)
                or payload_bytes <= 0):
            raise ValueError("target collective payload bytes must be positive")
        nodes.append(WorkloadNode(
            node_id,
            WorkloadKind.COLLECTIVE,
            pool.name,
            {},
            rank=participants[0],
            attributes={
                "algorithm": collective.algorithm,
                "bytes": payload_bytes,
                "bytes_semantics": "per_rank_input",
                "backend": collective.backend,
                "backend_config_sha256": collective.backend_config_sha256,
                "direction": "collective",
                "link": collective.link,
                "topology": collective.topology,
                "participants": participants,
                "operation": operation,
                "source_rank": None,
                "destination_rank": None,
                "ring_order": _collective_ring_order(
                    collective.algorithm, participants),
                "reduction": reduction,
                "resource_id": collective.resource_id,
                # sglang parallel_state tensor_model_parallel_all_reduce ->
                # CustomAllReduceV2 on a fully NVLink-connected TP group,
                # or flashinfer's all-reduce fused with the next residual
                # RMSNorm where the decoder layer fuses it.
                **({"implementation": (
                    "sglang_flashinfer_allreduce_residual_rmsnorm"
                    if suffix in fused_all_reduce_suffixes
                    else "sglang_custom_all_reduce_v2")}
                   if (runtime.serving_framework == "sglang"
                       and operation == "all_reduce"
                       and collective.link == "nvlink") else {}),
            },
        ))
        incoming = list(predecessors)
        if not incoming and previous is not None:
            incoming.append(previous)
        for predecessor in incoming:
            edges.append(WorkloadEdge(predecessor, node_id, "barrier"))
        previous = node_id
        return node_id

    def linear(
        suffix,
        position,
        block_tokens,
        *,
        module_path,
        source_symbol,
        weight_dtype=None,
        dtype=None,
        extra_predecessors=(),
        use_previous=True,
    ):
        m, n, k = infer_gemm_dimensions(
            model, position, block_tokens, tp_size=tp_size)
        if weight_dtype is None:
            resolved_weight_dtype = model.weight_dtype_for_module(
                module_path, "Linear")
        else:
            resolved_weight_dtype = weight_dtype
        node_id = append(
            WorkloadKind.LINEAR,
            suffix,
            {"b": 1, "m": m, "n": n, "k": k},
            variant="gemm",
            attributes={
                "has_bias": False,
                "launch_policy": launch_policy,
                "module_path": module_path,
                "position": position,
                "source_symbol": source_symbol,
                "weight_dtype": resolved_weight_dtype,
                "weight_group_size": weight_group_size(
                    resolved_weight_dtype),
                # sglang fp8_utils.deepgemm_w8a8_block_fp8_linear_with_fallback:
                # per_token_group_quant + DeepGEMM 1d2d at every M.
                **({"fp8_linear_runner": "sglang_deepgemm"}
                   if (runtime.serving_framework == "sglang"
                       and model.model_type == "deepseek_v3") else {}),
            },
            dtype=dtype,
            extra_predecessors=extra_predecessors,
            use_previous=use_previous,
        )
        if position == "o_proj":
            return append_tp_collective(
                f"{suffix}:tp-all-reduce", "all_reduce")
        return node_id

    def dense_mlp(
        prefix,
        block_tokens,
        intermediate,
        *,
        source_prefix,
        input_predecessors=(),
        use_previous=True,
    ):
        mlp_predecessors = list(input_predecessors)
        if use_previous and previous is not None:
            if previous not in mlp_predecessors:
                mlp_predecessors.append(previous)
        mlp_predecessors = tuple(mlp_predecessors)
        if intermediate % tp_size:
            raise ValueError(
                "target dense intermediate size must divide TP size")
        local_intermediate = intermediate // tp_size
        if model.model_type in ("kimi_k2", "glm_moe_dsa", "deepseek_v3"):
            weight_dtype = model.weight_dtype_for_module(
                f"{source_prefix}.gate_proj", "Linear")
            append(
                WorkloadKind.LINEAR,
                f"{prefix}:gate-up",
                {
                    "b": 1,
                    "m": block_tokens,
                    "n": 2 * local_intermediate,
                    "k": model.hidden_size,
                },
                variant="gemm",
                attributes={
                    "has_bias": False,
                    "launch_policy": launch_policy,
                    "module_path": source_prefix,
                    "position": "gate_up_proj",
                    "source_symbol": "vllm.DeepseekV2MLP.gate_up_proj",
                    "weight_dtype": weight_dtype,
                    "weight_group_size": weight_group_size(weight_dtype),
                },
                extra_predecessors=mlp_predecessors,
                use_previous=False,
            )
            activation_elements = block_tokens * local_intermediate
            activation_extra_predecessors = ()
        else:
            raise AssertionError(
                f"unhandled target model type: {model.model_type!r}")
        append(
            WorkloadKind.ACTIVATION,
            f"{prefix}:silu-mul",
            {
                "elements": activation_elements,
                "tok": block_tokens,
                "intermediate": local_intermediate,
            },
            variant="silu",
            attributes={
                "operation": "silu_and_mul",
                "source_symbol": dense_activation_source,
            },
            extra_predecessors=activation_extra_predecessors,
        )
        append(
            WorkloadKind.LINEAR,
            f"{prefix}:down-proj",
            {
                "b": 1,
                "m": block_tokens,
                "n": model.hidden_size,
                "k": local_intermediate,
            },
            variant="gemm",
            attributes={
                "has_bias": False,
                "launch_policy": launch_policy,
                "module_path": f"{source_prefix}.down_proj",
                "position": "down_proj",
                "source_symbol": dense_down_source,
                "weight_dtype": model.weight_dtype_for_module(
                    f"{source_prefix}.down_proj", "Linear"),
                "weight_group_size": weight_group_size(
                    model.weight_dtype_for_module(
                        f"{source_prefix}.down_proj", "Linear")),
            },
        )
        append_tp_collective(
            f"{prefix}:down-proj:tp-all-reduce", "all_reduce")

    def resolve_target_routing(block_tokens, layer_index, invocation_id):
        stage_logical_rank = active_pipeline_stage * tp_size
        expert_group = pool.parallel_plan.rank_groups.for_rank(
            "expert", stage_logical_rank)
        physical_expert_group = tuple(
            pool.physical_rank_for_logical_rank(logical_rank)
            for logical_rank in expert_group)
        return runtime.expert_routing.resolve(
            step_id=step.step_id,
            layer_index=layer_index,
            moe_invocation_id=invocation_id,
            schedule_step_sha256=step_sha256,
            tokens=block_tokens,
            num_experts=model.num_experts,
            topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256,
            pool_name=pool.name,
            tensor_parallel_size=tp_size,
            expert_parallel_size=ep_size,
            expert_rank_group=physical_expert_group,
            expert_placement_strategy="linear",
            memo=routing_memo,
            moe_layer_ordinal=(
                None if layer_index is None
                else model.moe_layer_ordinal(layer_index)),
        )

    def append_sglang_shared_expert_topk_glue(
            append_rank_node, prefix, rank_suffix, block_tokens,
            physical_rank, router_id):
        """SGLang DeepSeek top-k glue of the fused shared expert.

        sglang ``layers/moe/topk.py`` ``biased_grouped_topk_gpu`` (v0.5.12
        lines 1050-1072, v0.5.19 lines 1676-1688) takes the flashinfer
        ``fused_topk_deepseek`` branch for DeepSeek-V3 (256 experts, 8
        groups, top-4 groups, 8 routed) and, with shared-experts fusion
        enabled (server log "Shared experts fusion optimization enabled" in
        both the InferenceX v0.5.19 MTP run and the v0.5.12 H200 run),
        appends the shared-expert column in PyTorch:
        ``F.pad(topk_ids)`` and ``F.pad(topk_weights)`` (``constant_pad_nd``:
        one fill of the padded tensor and one copy of the routed columns
        each), ``topk_weights[:, :8].sum(-1)`` (one reduce),
        ``/ routed_scaling_factor`` (one scalar multiply) and the slice
        assignment (one copy).  Only launches and element counts are
        emitted; pricing reuses the vector families.
        """

        if (model.model_type != "deepseek_v3"
                or model.num_shared_experts != 1
                or model.target_semantics.moe_topk_method != "noaux_tc"):
            raise NotImplementedError(
                "SGLang fused shared-expert top-k glue is source-backed only "
                "for DeepSeek-V3 noaux_tc routing with one shared expert")
        routed = model.num_experts_per_tok
        padded = routed + model.num_shared_experts
        source = "sglang.srt.layers.moe.topk.biased_grouped_topk_gpu"
        chain = (
            ("pad-ids-fill", WorkloadKind.ELEMENTWISE, "int32", "fill",
             padded, {"reads": 0, "writes": 1}),
            ("pad-ids-copy", WorkloadKind.ELEMENTWISE, "int32", "copy",
             routed, {"reads": 1, "writes": 1}),
            ("pad-weights-fill", WorkloadKind.ELEMENTWISE, "float32", "fill",
             padded, {"reads": 0, "writes": 1}),
            ("pad-weights-copy", WorkloadKind.ELEMENTWISE, "float32", "copy",
             routed, {"reads": 1, "writes": 1}),
            ("shared-weight-sum", WorkloadKind.REDUCE, "float32", None,
             1, {"reduction_factor": routed}),
            ("shared-weight-scale", WorkloadKind.ELEMENTWISE, "float32",
             "unary", 1, {"reads": 1, "writes": 1}),
            ("shared-weight-store", WorkloadKind.ELEMENTWISE, "float32",
             "copy", 1, {"reads": 1, "writes": 1}),
        )
        previous_id = router_id
        for name, kind, dtype, operation, per_token, fields in chain:
            attributes = {"source_symbol": source, **fields}
            if operation is not None:
                attributes["operation"] = operation
            previous_id = append_rank_node(
                kind,
                f"{prefix}:sglang-topk-{name}:{rank_suffix}",
                {"elements": block_tokens * per_token},
                physical_rank=physical_rank,
                variant=(
                    "reduce" if kind is WorkloadKind.REDUCE
                    else "elementwise"),
                dtype=dtype,
                attributes=attributes,
                predecessors=(previous_id,),
            )
        return previous_id

    def moe(prefix, block_tokens, module_layer_index, routing_layer_index):
        nonlocal previous
        moe_input = previous
        if moe_input is None:
            raise ValueError("MoE workload requires an input predecessor")
        if tp_size == 1 and runtime.serving_framework == "sglang":
            raise NotImplementedError(
                "SGLang MoE glue is emitted only on the tensor-parallel "
                "per-rank path")
        if tp_size > 1:
            invocation_id = f"{step.step_id}:{prefix}:moe"
            resolution = resolve_target_routing(
                block_tokens, routing_layer_index, invocation_id)
            routed_module = (
                f"model.layers.{module_layer_index}.mlp.experts.0.gate_proj")
            routed_weight_dtype = model.weight_dtype_for_module(
                routed_module, "Linear")
            expert_group = pool.parallel_plan.rank_groups.for_rank(
                "expert", active_pipeline_stage * tp_size)
            shared_intermediate = (
                model.moe_intermediate_size * model.num_shared_experts)
            if shared_intermediate % tp_size:
                raise ValueError(
                    "target shared-expert intermediate size must divide TP")
            local_shared_intermediate = shared_intermediate // tp_size
            shared_source_prefix = (
                f"model.layers.{module_layer_index}.mlp.shared_experts")
            shared_weight_dtype = model.weight_dtype_for_module(
                f"{shared_source_prefix}.gate_proj", "Linear")
            shared_down_weight_dtype = model.weight_dtype_for_module(
                f"{shared_source_prefix}.down_proj", "Linear")
            local_add_nodes = []

            def append_rank_node(
                    kind, suffix, shape, *, physical_rank, variant,
                    attributes, predecessors, dtype=None):
                normalized_attributes = _with_launch_policy(
                    kind, attributes, launch_policy)
                normalized_attributes = _with_gemm_semantics(
                    kind, normalized_attributes, tp_size, ep_size)
                normalized_attributes = _with_eager_attention_parallelism(
                    kind, normalized_attributes, tp_size, ep_size)
                node_id = f"{step.step_id}:{suffix}"
                nodes.append(WorkloadNode(
                    node_id,
                    kind,
                    pool.name,
                    shape,
                    dtype=(model.compute_dtype if dtype is None else dtype),
                    variant=variant,
                    rank=physical_rank,
                    attributes=normalized_attributes,
                ))
                for predecessor in predecessors:
                    edges.append(WorkloadEdge(predecessor, node_id))
                return node_id

            for expert_rank, logical_rank in enumerate(expert_group):
                physical_rank = pool.physical_rank_for_logical_rank(logical_rank)
                rank_suffix = f"tp-rank-{expert_rank}"
                router_id = append_rank_node(
                    WorkloadKind.ROUTER,
                    f"{prefix}:router:{rank_suffix}",
                    {
                        "b": 1,
                        "m": block_tokens,
                        "n": model.num_experts,
                        "k": model.hidden_size,
                    },
                    physical_rank=physical_rank,
                    variant="gemm",
                    dtype=moe_router_dtype,
                    attributes={
                        "has_bias": False,
                        "launch_policy": launch_policy,
                        "position": "router",
                        "score_function": (
                            model.target_semantics.moe_score_function),
                        "topk_method": model.target_semantics.moe_topk_method,
                        "expert_group_count": (
                            model.target_semantics.moe_group_count),
                        "topk_group_count": (
                            model.target_semantics.moe_topk_group_count),
                        "normalize_topk_probability": (
                            model.target_semantics
                            .moe_normalize_topk_probability),
                        "source_symbol": moe_router_source,
                        "weight_dtype": moe_router_weight_dtype,
                        "weight_group_size": weight_group_size(
                            moe_router_weight_dtype),
                        **sglang_router_runner,
                    },
                    predecessors=(moe_input,),
                )
                if runtime.serving_framework == "sglang":
                    router_id = append_sglang_shared_expert_topk_glue(
                        append_rank_node, prefix, rank_suffix,
                        block_tokens, physical_rank, router_id)
                owned_experts = tuple(
                    expert_id
                    for expert_id, owner_rank in enumerate(
                        pool.parallel_plan.expert_to_shard)
                    if owner_rank == expert_rank
                )
                local_counts = tuple(
                    count if expert_id in owned_experts else 0
                    for expert_id, count in enumerate(
                        resolution.tokens_per_expert)
                )
                routed_id = None
                if sum(local_counts) > 0:
                    routed_id = append_rank_node(
                        WorkloadKind.FUSED_MOE,
                        f"{prefix}:routed-experts:{rank_suffix}",
                        {
                            "tokens": block_tokens,
                            "experts": model.num_experts,
                            "topk": model.num_experts_per_tok,
                            "hidden": model.hidden_size,
                            "intermediate": (
                                routed_expert_local_intermediate(
                                    model, tp_size, ep_size)),
                        },
                        physical_rank=physical_rank,
                        variant="full",
                        attributes={
                            "expert_parallel_size": ep_size,
                            "expert_rank": expert_rank,
                            "owned_experts": owned_experts,
                            "local_tokens_per_expert": local_counts,
                            "expert_placement_strategy": "linear",
                            "launch_policy": launch_policy,
                            "moe_invocation_id": invocation_id,
                            "output_is_reduced": False,
                            "routed_scaling_factor": (
                                model.routed_scaling_factor),
                            "routed_weight_dtype": routed_weight_dtype,
                            "routing_provenance": dict(
                                resolution.routing_provenance),
                            "source_symbol": moe_expert_source,
                            "tokens_per_expert": (
                                resolution.tokens_per_expert),
                            "weight_group_size": weight_group_size(
                                routed_weight_dtype),
                            # SGLang shared-experts fusion (server log
                            # "Shared experts fusion optimization enabled"):
                            # the shared expert is expert E of the routed
                            # kernel; no separate shared MLP or add runs.
                            **({"fused_shared_experts": (
                                model.num_shared_experts)}
                               if runtime.serving_framework == "sglang"
                               else {}),
                        },
                        predecessors=(router_id, moe_input),
                    )
                if runtime.serving_framework == "sglang":
                    if routed_id is None:
                        raise ValueError(
                            "SGLang fused shared experts need a routed "
                            "kernel on every rank")
                    local_add_nodes.append(routed_id)
                    continue
                gate_up_id = append_rank_node(
                    WorkloadKind.LINEAR,
                    f"{prefix}:shared-experts:gate-up:{rank_suffix}",
                    {
                        "b": 1,
                        "m": block_tokens,
                        "n": 2 * local_shared_intermediate,
                        "k": model.hidden_size,
                    },
                    physical_rank=physical_rank,
                    variant="gemm",
                    attributes={
                        "has_bias": False,
                        "launch_policy": launch_policy,
                        "module_path": shared_source_prefix,
                        "position": "gate_up_proj",
                        "source_symbol": (
                            "vllm.DeepseekV2MLP.gate_up_proj"),
                        "weight_dtype": shared_weight_dtype,
                        "weight_group_size": weight_group_size(
                            shared_weight_dtype),
                    },
                    predecessors=(moe_input,),
                )
                activation_id = append_rank_node(
                    WorkloadKind.ACTIVATION,
                    f"{prefix}:shared-experts:silu-mul:{rank_suffix}",
                    {
                        "elements": (
                            block_tokens * local_shared_intermediate),
                        "tok": block_tokens,
                        "intermediate": local_shared_intermediate,
                    },
                    physical_rank=physical_rank,
                    variant="silu",
                    attributes={
                        "operation": "silu_and_mul",
                        "source_symbol": dense_activation_source,
                    },
                    predecessors=(gate_up_id,),
                )
                down_id = append_rank_node(
                    WorkloadKind.LINEAR,
                    f"{prefix}:shared-experts:down-proj:{rank_suffix}",
                    {
                        "b": 1,
                        "m": block_tokens,
                        "n": model.hidden_size,
                        "k": local_shared_intermediate,
                    },
                    physical_rank=physical_rank,
                    variant="gemm",
                    attributes={
                        "has_bias": False,
                        "launch_policy": launch_policy,
                        "module_path": f"{shared_source_prefix}.down_proj",
                        "position": "down_proj",
                        "reduce_results": False,
                        "source_symbol": dense_down_source,
                        "weight_dtype": shared_down_weight_dtype,
                        "weight_group_size": weight_group_size(
                            shared_down_weight_dtype),
                    },
                    predecessors=(activation_id,),
                )
                local_output_id = append_rank_node(
                    WorkloadKind.ELEMENTWISE,
                    f"{prefix}:shared-add:{rank_suffix}",
                    {"elements": block_tokens * model.hidden_size},
                    physical_rank=physical_rank,
                    variant="add",
                    attributes={
                        "operation": "add",
                        "reads": 2,
                        "routed_input": (
                            "materialized" if routed_id is not None
                            else "implicit_zero"),
                        "source_symbol": moe_shared_add_source,
                        "writes": 1,
                    },
                    predecessors=(
                        (down_id, routed_id)
                        if routed_id is not None
                        else (down_id,)
                    ),
                )
                local_add_nodes.append(local_output_id)
            previous = append_tp_collective(
                f"{prefix}:moe-output:tp-all-reduce",
                "all_reduce",
                predecessors=tuple(local_add_nodes),
            )
            return
        router_id = append(
            WorkloadKind.ROUTER,
            f"{prefix}:router",
            {
                "b": 1,
                "m": block_tokens,
                "n": model.num_experts,
                "k": model.hidden_size,
            },
            variant="gemm",
            dtype=moe_router_dtype,
            attributes={
                "has_bias": False,
                "launch_policy": launch_policy,
                "position": "router",
                "score_function": (
                    model.target_semantics.moe_score_function),
                "topk_method": model.target_semantics.moe_topk_method,
                "expert_group_count": (
                    model.target_semantics.moe_group_count),
                "topk_group_count": (
                    model.target_semantics.moe_topk_group_count),
                "normalize_topk_probability": (
                    model.target_semantics.moe_normalize_topk_probability),
                "source_symbol": moe_router_source,
                "weight_dtype": moe_router_weight_dtype,
                "weight_group_size": weight_group_size(
                    moe_router_weight_dtype),
                **sglang_router_runner,
            },
        )
        invocation_id = f"{step.step_id}:{prefix}:moe"
        resolution = resolve_target_routing(
            block_tokens, routing_layer_index, invocation_id)
        routed_module = (
            f"model.layers.{module_layer_index}.mlp.experts.0.gate_proj")
        routed_weight_dtype = model.weight_dtype_for_module(
            routed_module, "Linear")
        expert_group = pool.parallel_plan.rank_groups.for_rank(
            "expert", active_pipeline_stage * tp_size)
        routed_nodes = []
        for expert_rank, logical_rank in enumerate(expert_group):
            physical_rank = pool.physical_rank_for_logical_rank(logical_rank)
            owned_experts = tuple(
                expert_id
                for expert_id, owner_rank in enumerate(
                    pool.parallel_plan.expert_to_shard)
                if owner_rank == expert_rank
            )
            local_counts = tuple(
                count if expert_id in owned_experts else 0
                for expert_id, count in enumerate(
                    resolution.tokens_per_expert)
            )
            if sum(local_counts) == 0:
                continue
            if ep_size == 1:
                node_id = f"{step.step_id}:{prefix}:routed-experts"
            else:
                node_id = (
                    f"{step.step_id}:{prefix}:routed-experts:"
                    f"ep-rank-{expert_rank}")
            nodes.append(WorkloadNode(
                node_id,
                WorkloadKind.FUSED_MOE,
                pool.name,
                {
                    "tokens": block_tokens,
                    "experts": model.num_experts,
                    "topk": model.num_experts_per_tok,
                    "hidden": model.hidden_size,
                    "intermediate": routed_expert_local_intermediate(
                        model, tp_size, ep_size),
                },
                dtype=model.compute_dtype,
                variant="full",
                rank=physical_rank,
                attributes={
                    "expert_parallel_size": ep_size,
                    "expert_rank": expert_rank,
                    "owned_experts": owned_experts,
                    "local_tokens_per_expert": local_counts,
                    "expert_placement_strategy": "linear",
                    "launch_policy": launch_policy,
                    "moe_invocation_id": invocation_id,
                    "routed_scaling_factor": model.routed_scaling_factor,
                    "routed_weight_dtype": routed_weight_dtype,
                    "routing_provenance": dict(
                        resolution.routing_provenance),
                    "source_symbol": moe_expert_source,
                    "tokens_per_expert": resolution.tokens_per_expert,
                    "weight_group_size": weight_group_size(
                        routed_weight_dtype),
                },
            ))
            edges.append(WorkloadEdge(router_id, node_id))
            edges.append(WorkloadEdge(moe_input, node_id))
            routed_nodes.append(node_id)
        routed_expert_id = append_tp_collective(
            f"{prefix}:routed-experts:tp-all-reduce",
            "all_reduce",
            predecessors=tuple(routed_nodes),
        )
        dense_mlp(
            f"{prefix}:shared-experts",
            block_tokens,
            model.moe_intermediate_size * model.num_shared_experts,
            source_prefix=(
                f"model.layers.{module_layer_index}.mlp.shared_experts"),
            input_predecessors=(moe_input,),
            use_previous=False,
        )
        append(
            WorkloadKind.ELEMENTWISE,
            f"{prefix}:shared-add",
            {"elements": block_tokens * model.hidden_size},
            variant="add",
            attributes={
                "operation": "add",
                "reads": 2,
                "source_symbol": moe_shared_add_source,
                "writes": 1,
            },
            extra_predecessors=(routed_expert_id,),
        )

    def target_layer_weight_dtypes(layer_index, uses_moe):
        if uses_moe:
            module_paths = (
                f"model.layers.{layer_index}.mlp.experts.0.gate_proj",
                f"model.layers.{layer_index}.mlp.shared_experts.gate_proj",
                f"model.layers.{layer_index}.mlp.shared_experts.down_proj",
            )
        else:
            module_paths = (
                f"model.layers.{layer_index}.mlp.gate_proj",
                f"model.layers.{layer_index}.mlp.down_proj",
            )
        return tuple(
            model.weight_dtype_for_module(module_path, "Linear")
            for module_path in module_paths
        )

    def append_cached_target_layer(
        template,
        prefix,
        layer_index,
        current_request_ids,
        current_request_phases,
        routing_layer_index,
    ):
        nonlocal previous, topk_source, pending_stage_entry_predecessors
        if pending_stage_entry_predecessors:
            raise RuntimeError(
                "target block clone cannot consume multiple stage inputs")
        if previous is None:
            raise RuntimeError("target block clone lost its predecessor")
        external_id_map = {template.previous_before: previous}
        if template.topk_source_before is not None:
            if topk_source is None:
                raise RuntimeError(
                    "target shared-indexer block lost its top-k source")
            external_id_map[template.topk_source_before] = topk_source

        def rewrite_id(old_id):
            if old_id in external_id_map:
                return external_id_map[old_id]
            old_step_prefix = f"{template.step_id}:"
            if not old_id.startswith(old_step_prefix):
                raise RuntimeError(
                    f"target block template ID is outside its step: {old_id!r}")
            rewritten = f"{step.step_id}:{old_id[len(old_step_prefix):]}"
            return rewritten.replace(
                f":{template.prefix}", f":{prefix}", 1)

        internal_id_map = {
            node.node_id: rewrite_id(node.node_id)
            for node in template.nodes
        }
        routing_resolution = None
        for template_node in template.nodes:
            attributes = dict(template_node.attributes)
            if "module_path" in attributes:
                old_layer_prefix = f"model.layers.{template.layer_index}."
                module_path = attributes["module_path"]
                if (isinstance(module_path, str)
                        and module_path.startswith(old_layer_prefix)):
                    attributes["module_path"] = (
                        f"model.layers.{layer_index}."
                        + module_path[len(old_layer_prefix):]
                    )
            if "topk_source_node_id" in attributes:
                attributes["topk_source_node_id"] = rewrite_id(
                    attributes["topk_source_node_id"])
            if "request_ids" in attributes:
                attributes["request_ids"] = current_request_ids
            if "request_phases" in attributes:
                attributes["request_phases"] = current_request_phases
            if template_node.kind is WorkloadKind.FUSED_MOE:
                invocation_id = f"{step.step_id}:{prefix}:moe"
                if routing_resolution is None:
                    routing_resolution = resolve_target_routing(
                        token_count, routing_layer_index, invocation_id)
                attributes["moe_invocation_id"] = invocation_id
                attributes["tokens_per_expert"] = (
                    routing_resolution.tokens_per_expert)
                attributes["routing_provenance"] = dict(
                    routing_resolution.routing_provenance)
                owned_experts = tuple(attributes["owned_experts"])
                attributes["local_tokens_per_expert"] = tuple(
                    count if expert_id in owned_experts else 0
                    for expert_id, count in enumerate(
                        routing_resolution.tokens_per_expert)
                )
            nodes.append(WorkloadNode(
                internal_id_map[template_node.node_id],
                template_node.kind,
                template_node.pool_name,
                template_node.shape,
                dtype=template_node.dtype,
                variant=template_node.variant,
                rank=template_node.rank,
                attributes=attributes,
            ))
        for template_edge in template.edges:
            edges.append(WorkloadEdge(
                internal_id_map[template_edge.source]
                if template_edge.source in internal_id_map
                else rewrite_id(template_edge.source),
                internal_id_map[template_edge.destination]
                if template_edge.destination in internal_id_map
                else rewrite_id(template_edge.destination),
                template_edge.kind,
            ))

        previous = rewrite_id(template.previous_after)
        if template.topk_source_after is None:
            topk_source = None
        else:
            topk_source = rewrite_id(template.topk_source_after)

    append(
        WorkloadKind.GATHER,
        "token-embedding",
        {
            "elements": token_count * model.hidden_size,
            "rows": token_count,
            "hidden": model.hidden_size,
        },
        variant="embedding",
        attributes={
            "gather_factor": 1,
            "index_bytes": 8,
            "index_count": token_count,
            "model_semantics_contract": contract["contract"],
            "model_source_revision": model.model_revision,
            "scattered": True,
        },
    )
    append_tp_collective("token-embedding:tp-all-reduce", "all_reduce")

    def append_target_decoder_layer(
        layer_index,
        prefix,
        block_token_count,
        block_query_lengths,
        block_kv_lengths,
        block_phase,
        block_request_ids,
        block_request_phases,
        pipeline_stage,
        attention_kind,
        uses_moe,
        routing_layer_index,
        cache_allowed,
        prepared_signature=None,
    ):
        nonlocal active_token_count, topk_source
        token_count = block_token_count
        active_token_count = block_token_count
        fuse_attention_ar, absorb_input_norm, fuse_mlp_ar = (
            sglang_fused_all_reduce(layer_index, block_token_count))
        if fuse_attention_ar:
            fused_all_reduce_suffixes.add(f"{prefix}:o-proj:tp-all-reduce")
        if fuse_mlp_ar:
            fused_all_reduce_suffixes.update((
                f"{prefix}:moe-output:tp-all-reduce",
                f"{prefix}:down-proj:tp-all-reduce"))
        query_lengths = block_query_lengths
        kv_lengths = block_kv_lengths
        phase = block_phase
        # Attention-group truth table, keyed by the real per-row scheduler
        # phase.  Both backends derive that phase from prompt consumption
        # (src/serving/backend/vllm.py: "prefill" iff
        # computed_tokens < prompt_tokens; src/serving/backend/sglang.py:
        # extend mode is "prefill", decode mode is "decode"):
        #   request_phase | q_len | attention group
        #   prefill       | > 1   | prefill (genuine chunked prefill/resume)
        #   prefill       | 1     | prefill (single-token prefill tail chunk)
        #   decode        | 1     | decode  (plain decode)
        #   decode        | > 1   | decode  (MTP/EAGLE verify window)
        # q_len never reclassifies a row: verify windows stay in the decode
        # group and take the MLA decode absorbed path (q-side batched GEMMs
        # plus a decode attention kernel with q_len > 1, matching how real
        # backends run spec-verify), never _forward_prefill's full-prefix
        # kv_b_proj; only rows still consuming their prompt take the prefill
        # path.
        attention_request_phases = tuple(block_request_phases)
        attention_phase_set = set(attention_request_phases)
        if attention_phase_set == {"prefill"}:
            indexer_phase = "prefill"
        elif attention_phase_set == {"decode"}:
            indexer_phase = "decode"
        elif attention_phase_set == {"prefill", "decode"}:
            indexer_phase = "mixed"
        else:
            raise ValueError("target attention contains an unknown phase")
        prefill_request_indices = tuple(
            index for index, request_phase in enumerate(attention_request_phases)
            if request_phase == "prefill"
        )
        decode_request_indices = tuple(
            index for index, request_phase in enumerate(attention_request_phases)
            if request_phase == "decode"
        )
        prefill_token_count = sum(
            query_lengths[index] for index in prefill_request_indices)
        decode_token_count = sum(
            query_lengths[index] for index in decode_request_indices)
        enter_pipeline_stage(pipeline_stage, layer_index)
        target_signature = None
        cache_current_layer = (
            cache_allowed
            and block_memo is not None
            and not pending_stage_entry_predecessors
        )
        if cache_current_layer:
            if uses_moe:
                invocation_id = f"{step.step_id}:{prefix}:moe"
                routing_counts = resolve_target_routing(
                    token_count, routing_layer_index, invocation_id
                ).tokens_per_expert
                routing_policy_sha256 = (
                    runtime.expert_routing.policy_descriptor_sha256)
            else:
                routing_counts = None
                routing_policy_sha256 = None
            if uses_moe:
                block_kind = "moe"
            else:
                block_kind = "dense"
            target_signature = prepared_signature
            if target_signature is None:
                target_signature = TargetBlockExecutionSignature(
                    model_identity=id(model),
                    deployment_identity=id(deployment),
                    launch_policy=launch_policy,
                    phase=phase,
                    token_count=token_count,
                    request_phases=tuple(block_request_phases),
                    query_lengths=query_lengths,
                    kv_lengths=kv_lengths,
                    attention_kv_layout=attention_kv_layout,
                    attention_kv_distribution=(
                        runtime.attention_kv_distribution),
                    pipeline_stage=pipeline_stage,
                    block_kind=block_kind,
                    attention_kind=attention_kind,
                    layer_weight_dtypes=target_layer_weight_dtypes(
                        layer_index, uses_moe),
                    routing_policy_sha256=routing_policy_sha256,
                    routing_counts=routing_counts,
                )
            template = block_memo.target_template_for(target_signature)
            if template is not None:
                append_cached_target_layer(
                    template,
                    prefix,
                    layer_index,
                    tuple(block_request_ids),
                    attention_request_phases,
                    routing_layer_index,
                )
                return
        node_start = len(nodes)
        edge_start = len(edges)
        if previous is None:
            raise RuntimeError("target decoder block lacks its predecessor")
        previous_before = previous
        topk_source_before = {
            "mla": None,
            "dsa_full": None,
            "dsa_shared": topk_source,
        }[attention_kind]
        if absorb_input_norm:
            # Fused into the previous layer's flashinfer all-reduce.
            attention_input = previous
        else:
            attention_input = append(
                WorkloadKind.NORM,
                f"{prefix}:input-norm",
                {"rows": token_count, "hidden": model.hidden_size},
                variant="rmsnorm",
                attributes={
                    "residual_input": True,
                    "source_symbol": input_norm_source,
                },
            )
        fused_qkv_a = linear(
            f"{prefix}:fused-qkv-a-proj",
            "fused_qkv_a_proj",
            token_count,
            module_path=(
                f"model.layers.{layer_index}.self_attn.fused_qkv_a_proj"),
            source_symbol="vllm.DeepseekV2MLAAttention.fused_qkv_a_proj",
            weight_dtype="bf16",
        )
        q_a_norm = append(
            WorkloadKind.NORM,
            f"{prefix}:q-a-norm",
            {"rows": token_count, "hidden": model.q_lora_rank},
            variant="rmsnorm",
            attributes={
                "source_symbol": (
                    "vllm.DeepseekV2MLAAttention.q_a_layernorm"),
            },
        )
        q_b = linear(
            f"{prefix}:q-b-proj",
            "q_b_proj",
            token_count,
            module_path=f"model.layers.{layer_index}.self_attn.q_b_proj",
            source_symbol="vllm.DeepseekV2MLAAttention.q_b_proj",
            weight_dtype="bf16",
        )
        kv_a_norm = append(
            WorkloadKind.NORM,
            f"{prefix}:kv-a-norm",
            {"rows": token_count, "hidden": model.kv_lora_rank},
            variant="rmsnorm",
            attributes={
                "source_symbol": (
                    "vllm.DeepseekV2MLAAttention.kv_a_layernorm"),
            },
            extra_predecessors=(fused_qkv_a,),
            use_previous=False,
        )
        main_rotary_predecessors = (q_b, fused_qkv_a)

        append(
            WorkloadKind.ROTARY,
            f"{prefix}:main-rotary",
            {
                "elements": (
                    token_count
                    * (model.num_attention_heads // tp_size + 1)
                    * model.qk_rope_head_dim
                ),
                "seq_len": token_count,
                "num_query_heads": model.num_attention_heads // tp_size,
                "num_key_heads": 1,
                "head_dim": model.qk_rope_head_dim,
            },
            variant="rotary",
            attributes=rope_attributes(
                rotary_source, model.target_semantics.rope_layout),
            extra_predecessors=main_rotary_predecessors,
            use_previous=False,
        )
        compressed_kv_update = append(
            WorkloadKind.KV_UPDATE,
            f"{prefix}:compressed-kv-update",
            {
                "elements": token_count * (
                    model.kv_lora_rank + model.qk_rope_head_dim),
                "seq_len": token_count,
                "compressed_dim": (
                    model.kv_lora_rank + model.qk_rope_head_dim),
            },
            variant=compressed_cache_variant,
            attributes={
                "cache_layout": "paged_mla",
                "cache_write_semantics": "append",
                "source_symbol": "vllm.MLAAttention.forward",
            },
            extra_predecessors=(kv_a_norm,),
        )
        if model.model_type in ("kimi_k2", "deepseek_v3"):
            attention_output_predecessors = []
            # Deployment-level CUDA-graph facts the MLA backends read when
            # they build attention metadata (vLLM FlashAttnMLA sets its
            # split bound from ``use_full_cuda_graph`` and
            # ``max_cudagraph_capture_size``); config level, not the replay
            # mode of this step.
            graph_execution = runtime.graph_execution
            mla_graph_attributes = {
                "full_cuda_graphs": "FULL" in (
                    graph_execution.uniform_decode_mode,
                    graph_execution.non_uniform_mode),
                "max_cudagraph_capture_size": (
                    max(graph_execution.capture_sizes)
                    if graph_execution.enabled else 0),
            }
            prefill_prefix_tokens = 0
            if prefill_token_count:
                prefill_q_lens = tuple(
                    query_lengths[index]
                    for index in prefill_request_indices)
                prefill_kv_lens = tuple(
                    kv_lengths[index]
                    for index in prefill_request_indices)
                prefill_prefix_tokens = sum(prefill_kv_lens) - sum(
                    prefill_q_lens)
            if (prefill_token_count and prefill_prefix_tokens
                    and runtime.serving_framework == "sglang"):
                # SGLang attention_backend_handler._handle_attention_backend:
                # an extend batch whose summed prefix is below
                # SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD runs absorbed MLA
                # (flashinfer_mla_backend.forward_extend, paged prefill
                # wrapper) for the whole batch; a fresh batch runs MHA.
                if (prefill_prefix_tokens
                        >= _SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD):
                    raise NotImplementedError(
                        "SGLang chunked-prefix-cache MHA prefill is not "
                        "derived")
                attention_output_predecessors.append(
                    _append_absorbed_mla_attention(
                        append, step, prefix, model, tp_size,
                        prefill_token_count, prefill_q_lens,
                        prefill_kv_lens, launch_policy,
                        compressed_kv_update, runtime,
                        attention_kv_layout, mla_graph_attributes,
                        name="prefill-absorbed",
                        variant="mla_absorbed_prefill",
                        source_prefix="sglang.DeepseekV2AttentionMLA"))
            elif prefill_token_count:
                prefill_projected_kv_tokens = sum(prefill_kv_lens)
                prefill_kv_b = linear(
                    f"{prefix}:prefill-kv-b-proj",
                    "kv_b_proj",
                    prefill_projected_kv_tokens,
                    module_path=(
                        f"model.layers.{layer_index}.self_attn.kv_b_proj"),
                    source_symbol=(
                        "vllm.MLACommonImpl._forward_prefill.kv_b_proj"),
                    weight_dtype="bf16",
                    extra_predecessors=(compressed_kv_update,),
                    use_previous=False,
                )
                prefill_attention = append(
                    WorkloadKind.FLASH_ATTN,
                    f"{prefix}:prefill-mla-attention",
                    {
                        "batch": len(prefill_q_lens),
                        "q_len": max(prefill_q_lens),
                        "kv_len": max(prefill_kv_lens),
                        "qk_head_dim": model.head_dim,
                        "v_head_dim": model.v_head_dim,
                        "num_query_heads": (
                            model.num_attention_heads // tp_size),
                        "kv_lora_rank": model.kv_lora_rank,
                    },
                    variant="mla_prefill",
                    attributes={
                        "causal": True,
                        "kv_dist": runtime.attention_kv_distribution,
                        "kv_layout": attention_kv_layout,
                        "kv_lens": prefill_kv_lens,
                        "q_lens": prefill_q_lens,
                        "source_symbol": (
                            "vllm.MLACommonImpl._forward_prefill"),
                        **mla_graph_attributes,
                    },
                    extra_predecessors=(
                        compressed_kv_update, prefill_kv_b),
                    use_previous=False,
                )
                attention_output_predecessors.append(prefill_attention)
            if decode_token_count:
                decode_q_lens = tuple(
                    query_lengths[index]
                    for index in decode_request_indices)
                decode_kv_lens = tuple(
                    kv_lengths[index]
                    for index in decode_request_indices)
                decode_q_down = append(
                    WorkloadKind.LINEAR,
                    f"{prefix}:decode-q-down-proj",
                    {
                        "b": model.num_attention_heads // tp_size,
                        "m": decode_token_count,
                        "n": model.kv_lora_rank,
                        "k": model.qk_nope_head_dim,
                    },
                    variant="batched_gemm",
                    attributes={
                        "a_batch_stride_elements": (
                            decode_token_count * model.qk_nope_head_dim),
                        "b_batch_stride_elements": (
                            model.qk_nope_head_dim * model.kv_lora_rank),
                        "batch_layout": "strided",
                        "c_batch_stride_elements": (
                            decode_token_count * model.kv_lora_rank),
                        "has_bias": False,
                        "launch_policy": launch_policy,
                        "position": "mla_decode_q_down",
                        "source_symbol": (
                            "vllm.MLACommonImpl.forward.W_UK_T"),
                        "weight_dtype": "bf16",
                        "weight_group_size": 0,
                    },
                    extra_predecessors=(
                        f"{step.step_id}:{prefix}:main-rotary",),
                    use_previous=False,
                )
                decode_attention = append(
                    WorkloadKind.FLASH_ATTN,
                    f"{prefix}:decode-mla-attention",
                    {
                        "batch": len(decode_q_lens),
                        "q_len": max(decode_q_lens),
                        "kv_len": max(decode_kv_lens),
                        "qk_head_dim": (
                            model.kv_lora_rank + model.qk_rope_head_dim),
                        "v_head_dim": model.kv_lora_rank,
                        "num_query_heads": (
                            model.num_attention_heads // tp_size),
                        "kv_lora_rank": model.kv_lora_rank,
                    },
                    variant="mla_decode",
                    attributes={
                        "causal": True,
                        "kv_dist": runtime.attention_kv_distribution,
                        "kv_layout": attention_kv_layout,
                        "kv_lens": decode_kv_lens,
                        "q_lens": decode_q_lens,
                        "source_symbol": (
                            "vllm.MLACommonImpl._forward_decode"),
                        **mla_graph_attributes,
                        # sglang flashinfer_mla_backend: decode and target
                        # verify run BatchMLAPagedAttentionWrapper (fa3).
                        **({"attention_backend": "flashinfer"}
                           if runtime.serving_framework == "sglang" else {}),
                    },
                    extra_predecessors=(
                        compressed_kv_update, decode_q_down),
                    use_previous=False,
                )
                decode_v_up = append(
                    WorkloadKind.LINEAR,
                    f"{prefix}:decode-v-up-proj",
                    {
                        "b": model.num_attention_heads // tp_size,
                        "m": decode_token_count,
                        "n": model.v_head_dim,
                        "k": model.kv_lora_rank,
                    },
                    variant="batched_gemm",
                    attributes={
                        "a_batch_stride_elements": (
                            decode_token_count * model.kv_lora_rank),
                        "b_batch_stride_elements": (
                            model.kv_lora_rank * model.v_head_dim),
                        "batch_layout": "strided",
                        "c_batch_stride_elements": (
                            decode_token_count * model.v_head_dim),
                        "has_bias": False,
                        "launch_policy": launch_policy,
                        "position": "mla_decode_v_up",
                        "source_symbol": (
                            "vllm.MLACommonBaseImpl._v_up_proj.W_UV"),
                        "weight_dtype": "bf16",
                        "weight_group_size": 0,
                    },
                    extra_predecessors=(decode_attention,),
                    use_previous=False,
                )
                attention_output_predecessors.append(decode_v_up)
            if not attention_output_predecessors:
                raise AssertionError(
                    "target MLA step has no attention phase")
        elif model.model_type == "glm_moe_dsa":
            if attention_kind == "dsa_full":
                indexer_query = linear(
                    f"{prefix}:indexer-wq-b",
                    "indexer_wq_b",
                    token_count,
                    module_path=(
                        f"model.layers.{layer_index}.self_attn.indexer.wq_b"),
                    source_symbol="vllm.Indexer.wq_b",
                    extra_predecessors=(q_a_norm,),
                    use_previous=False,
                )
                indexer_key_weights = linear(
                    f"{prefix}:indexer-wk-weights-proj",
                    "indexer_wk_weights_proj",
                    token_count,
                    module_path=(
                        f"model.layers.{layer_index}.self_attn.indexer."
                        "wk_weights_proj"),
                    source_symbol="vllm.Indexer.wk_weights_proj",
                    weight_dtype="bf16",
                    extra_predecessors=(attention_input,),
                    use_previous=False,
                )
                indexer_key_norm = append(
                    WorkloadKind.NORM,
                    f"{prefix}:indexer-k-norm",
                    {"rows": token_count, "hidden": model.index_head_dim},
                    variant="layernorm",
                    attributes={
                        "affine": True,
                        "epsilon": 1e-5,
                        "source_symbol": "vllm.Indexer.k_norm",
                    },
                    extra_predecessors=(indexer_key_weights,),
                    use_previous=False,
                )
                indexer_rotary = append(
                    WorkloadKind.ROTARY,
                    f"{prefix}:indexer-rotary",
                    {
                        "elements": token_count * (
                            model.index_n_heads + 1)
                        * model.qk_rope_head_dim,
                        "seq_len": token_count,
                        "num_query_heads": model.index_n_heads,
                        "num_key_heads": 1,
                        "head_dim": model.qk_rope_head_dim,
                    },
                    variant="rotary",
                    attributes=rope_attributes(
                        "vllm.Indexer.indexer_rope_emb",
                        model.target_semantics.indexer_rope_layout),
                    extra_predecessors=(indexer_query, indexer_key_norm),
                    use_previous=False,
                )
                indexer_k_update = append(
                    WorkloadKind.KV_UPDATE,
                    f"{prefix}:indexer-k-update",
                    {
                        "elements": token_count * (
                            model.index_head_dim
                            + model.index_head_dim // 128 * 4),
                        "seq_len": token_count,
                        "key_dim": model.index_head_dim,
                        "scale_bytes_per_token": (
                            model.index_head_dim // 128 * 4),
                    },
                    variant="dsa_fp8_index_cache",
                    dtype="uint8",
                    attributes={
                        "cache_layout": "paged_mla_indexer",
                        "cache_write_semantics": "append",
                        "quantization": "fp8_e4m3_with_ue8m0_scale",
                        "source_symbol": "vllm.DeepseekV32IndexerCache",
                    },
                    extra_predecessors=(indexer_rotary,),
                    use_previous=False,
                )
                topk_source = append(
                    WorkloadKind.EAGER_ATTN,
                    f"{prefix}:indexer-score-topk",
                    {
                        "batch": len(query_lengths),
                        "q_len": max(query_lengths),
                        "kv_len": max(kv_lengths),
                        "num_query_heads": model.index_n_heads,
                        "head_dim": model.index_head_dim,
                        "topk": min(model.index_topk, max(kv_lengths)),
                    },
                    variant=f"dsa_indexer_{indexer_phase}",
                    dtype="int32",
                    attributes={
                        "causal": True,
                        "cache_layout": "paged_mla_indexer",
                        "cache_quantization": (
                            "fp8_e4m3_with_ue8m0_scale"),
                        "score_accumulator_dtype": "fp32",
                        "kv_lens": kv_lengths,
                        "q_lens": query_lengths,
                        "request_ids": tuple(block_request_ids),
                        "request_phases": attention_request_phases,
                        "score_function": "weighted_dot_product_topk",
                        "source_symbol": "vllm.SparseAttnIndexer.forward",
                        "topk_capacity": model.index_topk,
                        "topk_index_dtype": "int32",
                        "topk_sentinel": -1,
                    },
                    extra_predecessors=(
                        indexer_k_update, indexer_key_weights),
                    use_previous=False,
                )
            elif attention_kind == "dsa_shared":
                if topk_source is None:
                    raise ValueError(
                        "GLM shared DSA layer lacks a prior full indexer")
                topk_source = append(
                    WorkloadKind.GATHER,
                    f"{prefix}:indexer-topk-reuse",
                    {
                        "elements": token_count * min(
                            model.index_topk, max(kv_lengths)),
                    },
                    variant="dsa_index_reuse",
                    dtype="int32",
                    attributes={
                        "reuse_semantics": "alias",
                        "source_symbol": (
                            "vllm.MultiHeadLatentAttentionWrapper.skip_topk"),
                        "topk_capacity": model.index_topk,
                        "topk_index_dtype": "int32",
                        "topk_source_node_id": topk_source,
                    },
                    extra_predecessors=(topk_source,),
                    use_previous=False,
                )
            else:
                raise ValueError(
                    "unknown GLM DSA indexer type: "
                    f"{attention_kind!r}")
            attention_output_predecessors = []
            phase_inputs = (
                (
                    "prefill",
                    prefill_token_count,
                    prefill_request_indices,
                    "vllm.SparseMLACommonImpl.forward_mha",
                ),
                (
                    "decode",
                    decode_token_count,
                    decode_request_indices,
                    "vllm.FlashMLASparseImpl._forward_bf16_kv",
                ),
            )
            for attention_phase, phase_token_count, request_indices, source in (
                    phase_inputs):
                if not phase_token_count:
                    continue
                phase_q_lens = tuple(
                    query_lengths[index] for index in request_indices)
                phase_kv_lens = tuple(
                    kv_lengths[index] for index in request_indices)
                phase_topk_lens = tuple(
                    min(model.index_topk, length)
                    for length in phase_kv_lens)
                phase_predecessors = [compressed_kv_update, topk_source]
                if attention_phase == "prefill":
                    prefill_projected_kv_tokens = sum(phase_kv_lens)
                    prefill_kv_b = linear(
                        f"{prefix}:prefill-kv-b-proj",
                        "kv_b_proj",
                        prefill_projected_kv_tokens,
                        module_path=(
                            f"model.layers.{layer_index}.self_attn.kv_b_proj"),
                        source_symbol=(
                            "vllm.SparseMLACommonImpl._project_kv"),
                        weight_dtype="bf16",
                        extra_predecessors=(compressed_kv_update,),
                        use_previous=False,
                    )
                    phase_predecessors.append(prefill_kv_b)
                    attention_qk_head_dim = model.head_dim
                    attention_v_head_dim = model.v_head_dim
                elif attention_phase == "decode":
                    decode_q_down = append(
                        WorkloadKind.LINEAR,
                        f"{prefix}:decode-q-down-proj",
                        {
                            "b": model.num_attention_heads // tp_size,
                            "m": phase_token_count,
                            "n": model.kv_lora_rank,
                            "k": model.qk_nope_head_dim,
                        },
                        variant="batched_gemm",
                        attributes={
                            "a_batch_stride_elements": (
                                phase_token_count * model.qk_nope_head_dim),
                            "b_batch_stride_elements": (
                                model.qk_nope_head_dim * model.kv_lora_rank),
                            "batch_layout": "strided",
                            "c_batch_stride_elements": (
                                phase_token_count * model.kv_lora_rank),
                            "has_bias": False,
                            "launch_policy": launch_policy,
                            "position": "mla_decode_q_down",
                            "source_symbol": (
                                "vllm.MLACommonBaseImpl.forward_impl.W_UK_T"),
                            "weight_dtype": "bf16",
                            "weight_group_size": 0,
                        },
                        extra_predecessors=(
                            f"{step.step_id}:{prefix}:main-rotary",),
                        use_previous=False,
                    )
                    phase_predecessors.append(decode_q_down)
                    attention_qk_head_dim = (
                        model.kv_lora_rank + model.qk_rope_head_dim)
                    attention_v_head_dim = model.kv_lora_rank
                else:
                    raise AssertionError(
                        f"unhandled DSA attention phase: {attention_phase!r}")
                attention = append(
                    WorkloadKind.EAGER_ATTN,
                    f"{prefix}:{attention_phase}-dsa-attention",
                    {
                        "batch": len(phase_q_lens),
                        "q_len": max(phase_q_lens),
                        "kv_len": max(phase_kv_lens),
                        "num_query_heads": (
                            model.num_attention_heads // tp_size),
                        "qk_head_dim": attention_qk_head_dim,
                        "v_head_dim": attention_v_head_dim,
                        "topk": max(phase_topk_lens),
                    },
                    variant=f"dsa_sparse_{attention_phase}",
                    attributes={
                        "causal": True,
                        "indexer_type": attention_kind.removeprefix("dsa_"),
                        "kv_dist": runtime.attention_kv_distribution,
                        "kv_layout": attention_kv_layout,
                        "kv_lens": phase_kv_lens,
                        "q_lens": phase_q_lens,
                        "selected_kv_lens": phase_topk_lens,
                        "source_symbol": source,
                        "topk_capacity": model.index_topk,
                        "topk_index_dtype": "int32",
                        "topk_source_node_id": topk_source,
                    },
                    extra_predecessors=tuple(phase_predecessors),
                    use_previous=False,
                )
                if attention_phase == "decode":
                    attention = append(
                        WorkloadKind.LINEAR,
                        f"{prefix}:decode-v-up-proj",
                        {
                            "b": model.num_attention_heads // tp_size,
                            "m": phase_token_count,
                            "n": model.v_head_dim,
                            "k": model.kv_lora_rank,
                        },
                        variant="batched_gemm",
                        attributes={
                            "a_batch_stride_elements": (
                                phase_token_count * model.kv_lora_rank),
                            "b_batch_stride_elements": (
                                model.kv_lora_rank * model.v_head_dim),
                            "batch_layout": "strided",
                            "c_batch_stride_elements": (
                                phase_token_count * model.v_head_dim),
                            "has_bias": False,
                            "launch_policy": launch_policy,
                            "position": "mla_decode_v_up",
                            "source_symbol": (
                                "vllm.MLACommonBaseImpl._v_up_proj.W_UV"),
                            "weight_dtype": "bf16",
                            "weight_group_size": 0,
                        },
                        extra_predecessors=(attention,),
                        use_previous=False,
                    )
                attention_output_predecessors.append(attention)
            if not attention_output_predecessors:
                raise AssertionError("GLM DSA step has no attention phase")
        else:
            raise AssertionError(
                f"unhandled target model: {model.model_type!r}")

        if model.model_type not in ("kimi_k2", "glm_moe_dsa", "deepseek_v3"):
            raise AssertionError(
                f"unhandled target model: {model.model_type!r}")
        output_predecessors = tuple(attention_output_predecessors)
        linear(
            f"{prefix}:o-proj",
            "o_proj",
            token_count,
            module_path=f"model.layers.{layer_index}.self_attn.o_proj",
            source_symbol=output_projection_source,
            weight_dtype="bf16",
            extra_predecessors=output_predecessors,
            use_previous=False,
        )
        if not fuse_attention_ar:
            append(
                WorkloadKind.NORM,
                f"{prefix}:post-attention-norm",
                {"rows": token_count, "hidden": model.hidden_size},
                variant="rmsnorm",
                attributes={
                    "residual_input": True,
                    "source_symbol": post_attention_norm_source,
                },
            )
        if uses_moe:
            moe(prefix, token_count, layer_index, routing_layer_index)
        else:
            dense_mlp(
                prefix,
                token_count,
                model.intermediate_size,
                source_prefix=f"model.layers.{layer_index}.mlp",
            )
        if cache_current_layer:
            if target_signature is None or previous is None:
                raise RuntimeError(
                    "target block construction lost its cache contract")
            block_memo.store_target(
                target_signature,
                _TargetBlockTemplate(
                    step_id=step.step_id,
                    prefix=prefix,
                    layer_index=layer_index,
                    nodes=tuple(nodes[node_start:]),
                    edges=tuple(edges[edge_start:]),
                    previous_before=previous_before,
                    previous_after=previous,
                    topk_source_before=topk_source_before,
                    topk_source_after=topk_source,
                ),
            )

    def target_layer_signature(layer_index):
        if model.model_type in ("kimi_k2", "deepseek_v3"):
            attention_kind = "mla"
        elif model.model_type == "glm_moe_dsa":
            attention_kind = f"dsa_{model.indexer_types[layer_index]}"
        else:
            raise AssertionError(
                f"unhandled target model type: {model.model_type!r}")
        uses_moe = model.layer_uses_moe(layer_index)
        if uses_moe:
            routing_counts = resolve_target_routing(
                token_count,
                layer_index,
                f"{step.step_id}:layer-{layer_index}:moe",
            ).tokens_per_expert
        else:
            routing_counts = None
        return TargetBlockExecutionSignature(
            model_identity=id(model),
            deployment_identity=id(deployment),
            launch_policy=launch_policy,
            phase=phase,
            token_count=token_count,
            request_phases=main_request_phases,
            query_lengths=query_lengths,
            kv_lengths=kv_lengths,
            attention_kv_layout=attention_kv_layout,
            attention_kv_distribution=runtime.attention_kv_distribution,
            pipeline_stage=expected_layer_stages[layer_index],
            block_kind="moe" if uses_moe else "dense",
            attention_kind=attention_kind,
            layer_weight_dtypes=target_layer_weight_dtypes(
                layer_index, uses_moe),
            routing_policy_sha256=(
                runtime.expert_routing.policy_descriptor_sha256
                if uses_moe else None),
            routing_counts=routing_counts,
            fused_all_reduce=sglang_fused_all_reduce(layer_index, token_count),
        )

    def target_repeat_groups(keys):
        if not keys:
            return {}
        groups = {}
        for stage, (stage_start, stage_end) in enumerate(layer_ranges):
            minimum_start = stage_start if stage == 0 else stage_start + 1
            materialized_cost = {stage_end: 0}
            action = {}
            for start in range(stage_end - 1, minimum_start - 1, -1):
                best_cost = 1 + materialized_cost[start + 1]
                best_action = (1, 1)
                if keys[start].attention_kind in ("mla", "dsa_full"):
                    for period in range(1, (stage_end - start) // 2 + 1):
                        repeat_count = 1
                        while (start + (repeat_count + 1) * period <= stage_end
                               and keys[start:start + period]
                               == keys[
                                   start + repeat_count * period:
                                   start + (repeat_count + 1) * period]):
                            repeat_count += 1
                        if repeat_count < 2:
                            continue
                        end = start + repeat_count * period
                        candidate_cost = period + materialized_cost[end]
                        if candidate_cost < best_cost:
                            best_cost = candidate_cost
                            best_action = (period, repeat_count)
                materialized_cost[start] = best_cost
                action[start] = best_action
            layer_index = minimum_start
            while layer_index < stage_end:
                period, repeat_count = action[layer_index]
                if repeat_count > 1:
                    groups[layer_index] = (period, repeat_count)
                layer_index += period * repeat_count
        return groups

    def append_target_repeat(
        node_start,
        edge_start,
        layer_start,
        period,
        repeat_count,
        previous_before,
        topk_source_before,
    ):
        internal_node_ids = {
            node.node_id for node in nodes[node_start:]
        }
        feedback_source_by_external_id = {
            previous_before: previous,
        }
        if topk_source_before is not None:
            if topk_source == topk_source_before:
                raise RuntimeError(
                    "target repeat cannot retain an external top-k source")
            feedback_source_by_external_id[topk_source_before] = topk_source
        feedback_edges = []
        for edge in edges[edge_start:]:
            if (edge.destination not in internal_node_ids
                    or edge.source in internal_node_ids):
                continue
            if edge.source not in feedback_source_by_external_id:
                raise RuntimeError(
                    "target repeat has an unmapped external dependency: "
                    f"{edge.source!r}")
            feedback_edges.append(WorkloadEdge(
                feedback_source_by_external_id[edge.source],
                edge.destination,
                edge.kind,
            ))
        if not feedback_edges:
            raise RuntimeError(
                "target repeat requires at least one recurrence edge")
        block_repeats.append(WorkloadBlockRepeat(
            f"{step.step_id}:layers-{layer_start}-"
            f"{layer_start + period - 1}",
            tuple(node.node_id for node in nodes[node_start:]),
            repeat_count,
            tuple(feedback_edges),
        ))

    main_request_ids = tuple(request.request_id for request in step.requests)
    main_request_phases = tuple(request.phase for request in step.requests)
    target_layer_signatures = (
        tuple(
            target_layer_signature(index)
            for index in range(model.num_hidden_layers)
        )
        if block_memo is not None else ()
    )
    repeat_groups = target_repeat_groups(target_layer_signatures)
    layer_index = 0
    while layer_index < model.num_hidden_layers:
        if layer_index in repeat_groups:
            period, repeat_count = repeat_groups[layer_index]
        else:
            period, repeat_count = 1, 1
        node_start = len(nodes)
        edge_start = len(edges)
        previous_before = previous
        topk_source_before = topk_source
        for materialized_layer_index in range(
                layer_index, layer_index + period):
            if model.model_type in ("kimi_k2", "deepseek_v3"):
                attention_kind = "mla"
            elif model.model_type == "glm_moe_dsa":
                attention_kind = (
                    f"dsa_{model.indexer_types[materialized_layer_index]}")
            else:
                raise AssertionError(
                    f"unhandled target model: {model.model_type!r}")
            append_target_decoder_layer(
                materialized_layer_index,
                f"layer-{materialized_layer_index}",
                token_count,
                query_lengths,
                kv_lengths,
                phase,
                main_request_ids,
                main_request_phases,
                expected_layer_stages[materialized_layer_index],
                attention_kind,
                model.layer_uses_moe(materialized_layer_index),
                materialized_layer_index,
                True,
                (
                    target_layer_signatures[materialized_layer_index]
                    if target_layer_signatures else None
                ),
            )
        if repeat_count > 1:
            append_target_repeat(
                node_start,
                edge_start,
                layer_index,
                period,
                repeat_count,
                previous_before,
                topk_source_before,
            )
            for repeated_layer_index in range(
                    layer_index + period,
                    layer_index + period * repeat_count):
                if block_memo.target_template_for(
                        target_layer_signatures[repeated_layer_index]) is None:
                    raise RuntimeError(
                        "target repeat lost a cached layer template")
        layer_index += period * repeat_count

    append(
        WorkloadKind.NORM,
        "final-norm",
        {"rows": token_count, "hidden": model.hidden_size},
        variant="rmsnorm",
        attributes={
            "source_symbol": final_norm_source,
        },
    )
    target_logit_rows = (
        len(step.requests)
        + sum(request.mtp_verified_tokens for request in step.requests)
    )
    m, n, k = infer_gemm_dimensions(
        model, "lm_head", target_logit_rows, tp_size=tp_size)
    append(
        WorkloadKind.LM_HEAD,
        "lm-head",
        {"b": 1, "m": m, "n": n, "k": k},
        variant="gemv" if m == 1 else "gemm",
        attributes={
            "has_bias": False,
            "launch_policy": launch_policy,
            "module_path": "lm_head",
            "position": "lm_head",
            "source_symbol": lm_head_source,
            "weight_dtype": model.weight_dtype_for_module(
                "lm_head", "Linear"),
            "weight_group_size": 0,
        },
    )
    append_tp_collective(
        "lm-head:tp-all-gather",
        "all_gather",
        payload_bytes=m * n * model.dtype_bytes,
    )

    mtp_requests = tuple(
        request for request in step.requests if request.mtp_proposer_tokens)
    if mtp_requests:
        if model.mtp_architecture not in ("glm_moe_dsa_mtp", "deepseek_mtp"):
            raise NotImplementedError(
                "target MTP schedule requires glm_moe_dsa_mtp or "
                "deepseek_mtp")
        proposal_depths = {
            request.mtp_proposer_tokens for request in mtp_requests}
        if len(proposal_depths) != 1:
            raise ValueError(
                "one vLLM MTP step must use one global proposal depth")
        proposal_depth = next(iter(proposal_depths))
        proposer_modes = {
            request.mtp_proposer_mode for request in mtp_requests}
        if proposer_modes != {"padded"}:
            raise NotImplementedError(
                "target MTP requires pinned padded proposer batches")
        mtp_request_ids = tuple(
            request.request_id for request in mtp_requests)
        initial_query_lengths = tuple(
            request.scheduled_tokens for request in mtp_requests)
        if any(length <= 0 for length in initial_query_lengths):
            raise ValueError("MTP proposer requires a positive accepted query")
        initial_kv_lengths = tuple(
            request.computed_tokens + request.scheduled_tokens
            for request in mtp_requests)
        initial_request_phases = tuple(
            request.phase for request in mtp_requests)
        mtp_layer_index = model.num_hidden_layers
        topk_source = None
        # SGLang never mixes prefill and decode requests in one batch
        # (enable_mixed_chunk off).  Its EAGLE V2 prefill batch runs only the
        # draft extend and its first top-1 (eagle_worker_v2.py:864-882); the
        # draft decode passes run in the next decode iteration.
        sglang_prefill = False
        if runtime.serving_framework == "sglang":
            initial_phases = set(initial_request_phases)
            if initial_phases == {"prefill", "decode"}:
                raise NotImplementedError(
                    "SGLang EAGLE V2 batches are all-prefill or all-decode")
            sglang_prefill = initial_phases == {"prefill"}
        proposer_pass_count = 1 if sglang_prefill else proposal_depth
        for proposer_pass in range(proposer_pass_count):
            if proposer_pass == 0:
                proposer_query_lengths = initial_query_lengths
                proposer_kv_lengths = initial_kv_lengths
                proposer_request_phases = initial_request_phases
            else:
                proposer_query_lengths = (1,) * len(mtp_requests)
                proposer_kv_lengths = tuple(
                    length + proposer_pass for length in initial_kv_lengths)
                proposer_request_phases = ("decode",) * len(mtp_requests)
            proposer_token_count = sum(proposer_query_lengths)
            proposer_phases = set(proposer_request_phases)
            if proposer_phases == {"prefill"}:
                proposer_phase = "prefill"
            elif proposer_phases == {"decode"}:
                proposer_phase = "decode"
            elif proposer_phases == {"prefill", "decode"}:
                proposer_phase = "mixed"
            else:
                raise ValueError(
                    "MTP proposer contains an unknown request phase")
            prefix = f"mtp-pass-{proposer_pass}"
            append(
                WorkloadKind.GATHER,
                f"{prefix}:token-embedding",
                {
                    "elements": proposer_token_count * model.hidden_size,
                    "rows": proposer_token_count,
                    "hidden": model.hidden_size,
                },
                variant="embedding",
                attributes={
                    "gather_factor": 1,
                    "index_bytes": 8,
                    "index_count": proposer_token_count,
                    "scattered": True,
                    "source_symbol": "vllm.DeepSeekMTP.embed_input_ids",
                },
            )
            append_tp_collective(
                f"{prefix}:token-embedding:tp-all-reduce",
                "all_reduce",
                payload_bytes=(
                    proposer_token_count
                    * model.hidden_size
                    * model.dtype_bytes),
            )
            append(
                WorkloadKind.ELEMENTWISE,
                f"{prefix}:position-zero-mask",
                {"elements": proposer_token_count * model.hidden_size},
                variant="masked-fill-zero",
                attributes={
                    "operation": "copy",
                    "reads": 1,
                    "source_symbol": (
                        "vllm.DeepSeekMultiTokenPredictorLayer.forward"),
                    "writes": 1,
                },
            )
            append(
                WorkloadKind.NORM,
                f"{prefix}:embedding-norm",
                {"rows": proposer_token_count, "hidden": model.hidden_size},
                variant="rmsnorm",
                attributes={
                    "source_symbol": (
                        "vllm.DeepSeekMultiTokenPredictorLayer.enorm"),
                },
            )
            append(
                WorkloadKind.NORM,
                f"{prefix}:hidden-norm",
                {"rows": proposer_token_count, "hidden": model.hidden_size},
                variant="rmsnorm",
                attributes={
                    "source_symbol": (
                        "vllm.DeepSeekMultiTokenPredictorLayer.hnorm"),
                },
            )
            append(
                WorkloadKind.LINEAR,
                f"{prefix}:merge-proj",
                {
                    "b": 1,
                    "m": proposer_token_count,
                    "n": model.hidden_size,
                    "k": 2 * model.hidden_size,
                },
                variant="gemm",
                attributes={
                    "has_bias": False,
                    "launch_policy": launch_policy,
                    "module_path": f"model.layers.{mtp_layer_index}.eh_proj",
                    "position": "mtp_merge_proj",
                    "source_symbol": (
                        "vllm.DeepSeekMultiTokenPredictorLayer.eh_proj"),
                    "weight_dtype": model.compute_dtype,
                    "weight_group_size": 0,
                },
            )
            if model.mtp_architecture == "glm_moe_dsa_mtp":
                mtp_attention_kind = (
                    "dsa_full" if proposer_pass == 0 else "dsa_shared")
            elif model.mtp_architecture == "deepseek_mtp":
                # DeepSeek-V3 has no DSA indexer; its NextN draft layer runs
                # the same plain-MLA attention as a main decoder layer.
                mtp_attention_kind = "mla"
            else:
                raise AssertionError(
                    "unhandled MTP architecture: "
                    f"{model.mtp_architecture!r}")
            append_target_decoder_layer(
                mtp_layer_index,
                f"{prefix}:decoder",
                proposer_token_count,
                proposer_query_lengths,
                proposer_kv_lengths,
                proposer_phase,
                mtp_request_ids,
                proposer_request_phases,
                parallelism.pipeline_parallel_size - 1,
                mtp_attention_kind,
                model.mtp_layer_uses_moe(0),
                None,
                False,
            )
            append(
                WorkloadKind.ELEMENTWISE,
                f"{prefix}:residual-add",
                {"elements": proposer_token_count * model.hidden_size},
                variant="add",
                attributes={
                    "operation": "add",
                    "reads": 2,
                    "source_symbol": (
                        "vllm.DeepSeekMultiTokenPredictorLayer.forward"),
                    "writes": 1,
                },
            )
            # SGLang's EAGLE V2 decode-path draft extend computes logits for
            # every extended row (``select_index`` is None until after the
            # lm_head, eagle_worker_v2.py:1010); its prefill-batch draft
            # extend prunes to each request's last row before the lm_head
            # (logits_processor.py:561-564), as vLLM always does.
            head_rows = (
                proposer_token_count
                if (runtime.serving_framework == "sglang"
                    and proposer_pass == 0 and not sglang_prefill)
                else len(mtp_requests))
            append(
                WorkloadKind.NORM,
                f"{prefix}:head-norm",
                {"rows": head_rows, "hidden": model.hidden_size},
                variant="rmsnorm",
                attributes={
                    "source_symbol": "vllm.SharedHead.norm",
                },
            )
            m, n, k = infer_gemm_dimensions(
                model, "lm_head", head_rows, tp_size=tp_size)
            append(
                WorkloadKind.LM_HEAD,
                f"{prefix}:head",
                {"b": 1, "m": m, "n": n, "k": k},
                variant="gemv" if m == 1 else "gemm",
                attributes={
                    "has_bias": False,
                    "launch_policy": launch_policy,
                    "module_path": "lm_head",
                    "position": "lm_head",
                    "source_symbol": "vllm.DeepSeekMTP.compute_logits",
                    "weight_dtype": model.weight_dtype_for_module(
                        "lm_head", "Linear"),
                    "weight_group_size": 0,
                },
            )
            append_tp_collective(
                f"{prefix}:head:tp-all-gather",
                "all_gather",
                payload_bytes=m * n * model.dtype_bytes,
            )
        if runtime.serving_framework == "sglang":
            for name, kind, dtype, operation, elements, fields in (
                    _sglang_eagle_v2_iteration_launches(
                        model, mtp_requests, proposal_depth)):
                attributes = {
                    "source_symbol": "sglang.srt.speculative.eagle_worker_v2",
                    **fields,
                }
                if operation is not None:
                    attributes["operation"] = operation
                append(
                    kind,
                    f"sglang-eagle:{name}",
                    {"elements": elements},
                    variant=(
                        "reduce" if kind is WorkloadKind.REDUCE
                        else "elementwise"),
                    dtype=dtype,
                    attributes=attributes,
                )
            for name, direction, algorithm, payload in (
                    _sglang_eagle_v2_forward_stream_copies(
                        model, mtp_requests, proposal_depth)):
                append(
                    WorkloadKind.MEMCPY,
                    f"sglang-eagle:{name}",
                    {},
                    variant=None,
                    attributes={
                        "algorithm": algorithm,
                        "bytes": payload,
                        "direction": direction,
                        "link": "dram" if direction == "d2d" else "pcie",
                        "source_symbol": (
                            "sglang.srt.speculative.eagle_worker_v2"),
                    },
                )
    return WorkloadGraph(nodes, edges, block_repeats)


def _sglang_eagle_v2_forward_stream_copies(model, mtp_requests,
                                           proposal_depth):
    """Eager cudaMemcpyAsync copies of one EAGLE V2 iteration on the stream.

    Same source enumeration as ``_sglang_eagle_v2_iteration_launches``: the
    host-to-device copies of kv lengths (eagle_utils.py:968-969, pageable),
    the flashinfer MLA plan H2D copies (fast_mla_decode_plan /
    fast_mla_prefill_plan, flashinfer mla/_core.py .to(device), pageable),
    the pinned extend lengths (forward_batch_info.py:901-906), and the
    device-to-device copies (hidden states, kv indices, graph input ids and
    positions).  The D2H copies run on SGLang's private and copy streams
    (overlap_utils.py:562, scheduler.py:4124-4129) and do not serialize the
    forward stream, so they are not emitted.  Plan-info payloads are
    represented at 256 B (below the first sampled payload step).  Returns
    (name, direction, algorithm, bytes); a prefill batch returns ().
    """

    phases = {request.phase for request in mtp_requests}
    if phases == {"prefill"}:
        return ()
    batch = len(mtp_requests)
    verify = (proposal_depth + 1) * batch
    kv = sum(request.computed_tokens for request in mtp_requests)
    int32, int64 = 4, 8
    pageable, pinned, device = (
        "stream_copy_pageable", "stream_copy_pinned", "stream_copy")
    return (
        ("h2d-cur-kv-lens", "h2d", pageable, int32 * batch),
        ("h2d-next-kv-lens", "h2d", pageable, int32 * batch),
        ("d2d-draft-hidden", "d2d", device,
         model.hidden_size * model.dtype_bytes * batch),
        ("h2d-draft-plan-info", "h2d", pageable, 256),
        ("h2d-verify-qo-indptr", "h2d", pageable, int32 * (batch + 1)),
        ("h2d-verify-kv-indptr", "h2d", pageable, int32 * (batch + 1)),
        ("d2d-verify-kv-indices", "d2d", device, int32 * (kv + verify)),
        ("h2d-verify-kv-len", "h2d", pageable, int32 * batch),
        ("h2d-verify-plan-info", "h2d", pageable, 256),
        ("d2d-verify-input-ids", "d2d", device, int64 * verify),
        ("d2d-verify-positions", "d2d", device, int64 * verify),
        ("h2d-extend-seq-lens", "h2d", pinned, int32 * batch),
        ("h2d-extend-prefix-lens", "h2d", pinned, int32 * batch),
        ("h2d-extend-qo-indptr", "h2d", pageable, int32 * (batch + 1)),
        ("h2d-extend-kv-indptr", "h2d", pageable, int32 * (batch + 1)),
        ("h2d-extend-kv-len", "h2d", pageable, int32 * batch),
        ("h2d-extend-plan-info", "h2d", pageable, 256),
    )


def _sglang_eagle_v2_iteration_launches(model, mtp_requests, proposal_depth):
    """Non-model launches of one SGLang v0.5.19 EAGLE V2 iteration.

    Source: ``speculative/eagle_worker_v2.py``, ``eagle_worker_common.py``,
    ``eagle_utils.py``, ``eagle_info.py``, ``eagle_draft_cuda_graph_runner.py``,
    ``managers/overlap_utils.py``, ``layers/attention/flashinfer_mla_backend.py``
    and ``layers/logits_processor.py`` for DeepSeek NextN EAGLE with
    ``speculative_num_steps`` 2, ``eagle_topk`` 1, 3 draft tokens, greedy
    verification, the overlap (V2) worker, flashinfer MLA and a captured
    batch size (no padding launches).  Per iteration: 53 eager kernels
    (draft prep, verify metadata, greedy verify, draft-extend metadata and
    the FutureMap stash), 4 non-model kernels inside the draft-decode graph
    (token copy, two-stage top-1 argmax, positions - 1) and the FP32 logits
    materialization after each of the three lm_head calls.  Triton/CUDA
    helper kernels are represented by their dominant copy or fill; the
    eager memcpys (H2D/D2H/D2D) are not emitted, nor is the different glue
    of a prefill batch's draft extend.  Returns (name, kind,
    dtype, operation, elements, fields) in stream order.
    """

    phases = {request.phase for request in mtp_requests}
    if phases == {"prefill"}:
        # The prefill-batch draft extend runs a different, per-prefill glue
        # set that is not enumerated here; it is not emitted.
        return ()
    batch = len(mtp_requests)
    verify_lengths = {request.scheduled_tokens for request in mtp_requests}
    if (phases != {"decode"} or len(verify_lengths) != 1
            or proposal_depth != 2
            or next(iter(verify_lengths)) != proposal_depth + 1):
        raise NotImplementedError(
            "SGLang EAGLE V2 launches are source-derived for 2 draft steps, "
            "top-1 and 3 verified tokens per request: "
            f"depth={proposal_depth} scheduled={sorted(verify_lengths)} "
            f"phases={sorted({r.phase for r in mtp_requests})} "
            f"verified={sorted({r.mtp_verified_tokens for r in mtp_requests})}")
    per_request = proposal_depth + 1
    verify = per_request * batch
    kv = sum(request.computed_tokens for request in mtp_requests)
    vocab = model.vocab_size
    hidden = model.hidden_size
    partials = 16
    if vocab % partials:
        raise ValueError("draft top-1 partial argmax needs vocab % 16 == 0")
    fill = {"reads": 0, "writes": 1}
    one = {"reads": 1, "writes": 1}
    two = {"reads": 2, "writes": 1}
    ew = WorkloadKind.ELEMENTWISE
    red = WorkloadKind.REDUCE
    return (
        # schedule stream: req_to_token slots, published seq_lens gather
        ("assign-req-to-token-pool", ew, "int32", "copy", verify, one),
        ("new-seq-lens-gather", ew, "int64", "copy", batch, one),
        # draft decode inputs (FutureMap rows, cache locs, graph buffers)
        ("gather-spec-rows", ew, "bf16", "copy", batch * hidden, one),
        ("assign-draft-cache-locs", ew, "int64", "copy",
         proposal_depth * batch, one),
        ("draft-positions", ew, "int64", "copy", batch, one),
        ("draft-graph-inputs-int64", ew, "int64", "copy", 6 * batch, one),
        ("draft-graph-inputs-fp32", ew, "float32", "copy", batch, one),
        ("draft-decode-kv-indices", ew, "int32", "copy",
         proposal_depth * (kv + batch), one),
        ("draft-seq-lens-int32", ew, "int32", "copy", batch, one),
        # inside the draft-decode graph
        ("draft-token-copy", ew, "int64", "copy", batch, one),
        ("draft-logits-fp32", ew, "float32", "cast_bf16_float32",
         batch * vocab, one),
        ("draft-top1-partial-argmax", red, "float32", None,
         batch * partials, {"reduction_factor": vocab // partials}),
        ("draft-top1-finalize", red, "float32", None, batch,
         {"reduction_factor": partials}),
        ("draft-positions-minus-one", ew, "int64", "unary", batch, one),
        # verify metadata (tree, cache locs, flashinfer indptr/indices)
        ("verify-input-ids-cat", ew, "int64", "copy", verify, one),
        ("tree-mask-full", ew, "bool", "fill",
         per_request * kv + per_request * verify, fill),
        ("retrieve-buf-full", ew, "int64", "fill", per_request * verify,
         fill),
        ("build-tree", ew, "int64", "copy", 4 * verify, one),
        ("assign-extend-cache-locs", ew, "int64", "copy", verify, one),
        ("verify-graph-inputs", ew, "int64", "copy",
         3 * verify + 2 * batch, one),
        ("verify-qo-indptr-arange", ew, "int32", "fill", batch + 1, fill),
        ("verify-kv-indptr-zeros", ew, "int32", "fill", batch, fill),
        ("verify-kv-lens", ew, "int32", "unary", batch, one),
        ("verify-kv-indptr-cumsum", ew, "int32", "unary", batch, one),
        ("verify-kv-indptr-copy", ew, "int32", "copy", batch, one),
        ("verify-kv-indices", ew, "int32", "copy", kv + verify, one),
        # target verify graph logits, greedy verification
        ("verify-logits-fp32", ew, "float32", "cast_bf16_float32",
         verify * vocab, one),
        ("predict-zeros", ew, "int32", "fill", verify, fill),
        ("accept-index-full", ew, "int32", "fill", verify, fill),
        ("verify-argmax", red, "float32", None, verify,
         {"reduction_factor": vocab}),
        ("verify-tree-greedy", ew, "int32", "copy", verify, one),
        ("accept-length-plus-one", ew, "int32", "unary", batch, one),
        ("seq-lens-accept", ew, "int64", "add", batch, two),
        ("accepted-token-gather", ew, "int32", "copy", verify, one),
        ("fill-bonus-tokens", ew, "int32", "copy", batch, one),
        ("publish-seq-lens", ew, "int64", "copy", batch, one),
        ("select-accept-minus-one", ew, "int32", "unary", batch, one),
        ("select-arange", ew, "int64", "fill", batch, fill),
        ("select-add", ew, "int64", "add", batch, two),
        ("select-minus-one", ew, "int64", "unary", batch, one),
        ("predict-int64", ew, "int64", "copy", verify, one),
        # draft-extend metadata
        ("draft-extend-positions", ew, "int64", "copy", verify, one),
        ("draft-extend-seq-lens", ew, "int64", "unary", batch, one),
        ("extend-kv-indptr-cumsum", ew, "int32", "unary", batch, one),
        ("extend-kv-indptr-copy", ew, "int32", "copy", batch + 1, one),
        ("extend-kv-indices", ew, "int32", "copy", kv + verify, one),
        ("extend-qo-lens", ew, "int32", "add", batch, two),
        ("extend-qo-indptr-cumsum", ew, "int32", "unary", batch, one),
        ("extend-qo-indptr-copy", ew, "int32", "copy", batch + 1, one),
        ("draft-extend-inputs", ew, "int64", "copy",
         3 * verify + 2 * batch, one),
        # draft-extend logits over every row (select_index is None)
        ("draft-extend-logits-fp32", ew, "float32", "cast_bf16_float32",
         verify * vocab, one),
        ("select-logits", ew, "float32", "copy", batch * vocab, one),
        ("select-hidden", ew, "bf16", "copy", batch * hidden, one),
        ("draft-extend-argmax", red, "float32", None, batch,
         {"reduction_factor": vocab}),
        ("topk-p-ones", ew, "float32", "fill", batch, fill),
        # FutureMap stash of the next iteration's draft inputs
        ("bonus-int64", ew, "int64", "copy", batch, one),
        ("stash-tokens", ew, "int64", "copy", batch, one),
        ("stash-topk-p", ew, "float32", "copy", batch, one),
        ("stash-topk-index", ew, "int64", "copy", batch, one),
        ("stash-hidden", ew, "bf16", "copy", batch * hidden, one),
    )


def is_non_mtp_async_terminal_placeholder(request: ScheduledRequest) -> bool:
    """Return whether a scheduled row is vLLM's discarded async tail shape."""

    return (
        request.phase == "decode"
        and request.input_tokens == 0
        and request.output_tokens == 1
        and request.recomputed_tokens == 0
        and request.prefix_replay_tokens == 0
        and not request.resumed_from_preemption
        and request.raw_model_output_tokens == 1
        and request.mtp_proposer_mode == "disabled"
        and request.mtp_acceptance_source_kind == "disabled"
        and request.mtp_acceptance_source_id == "disabled"
        and request.mtp_proposed_tokens == 0
        and request.mtp_verified_tokens == 0
        and request.mtp_accepted_tokens == 0
        and request.mtp_proposer_tokens == 0
        and request.mtp_next_proposed_tokens == 0
    )


# Target-model node classes under CUDA-graph execution (vLLM 0.24
# gpu_model_runner / cudagraph_dispatcher / MLAAttention.forward):
# * activations inside the model forward run on the padded token count, so
#   their token-shaped dimensions become model-input rows with a graph
#   token domain;
# * lm_head (and its all-gather) consume ``hidden_states[logits_indices]``
#   outside the captured model and stay logical and eager;
# * the MLA op (``unified_mla_attention_with_output``) sees padded
#   attention metadata only under FULL (``pad_attn``): its decode
#   absorption bmms and attention run on the padded rows under FULL and on
#   the logical rows under PIECEWISE (eager, unpadded metadata);
# * the MLA KV-cache write (``unified_mla_kv_cache_update``) always uses the
#   padded slot mapping (``has_separate_kv_update``).
_TARGET_GRAPH_SCALED_FIELDS = {
    ("gather", "embedding"): (("elements", "rows"), ("index_count",)),
    ("norm", "rmsnorm"): (("rows",), ()),
    ("linear", "gemm"): (("m",), ()),
    ("router", "gemm"): (("m",), ()),
    ("rotary", "rotary"): (("elements", "seq_len"), ()),
    ("kv_update", "mla_kv_cache"): (("elements", "seq_len"), ()),
    ("activation", "silu"): (("elements", "tok"), ()),
    ("elementwise", "add"): (("elements",), ()),
    ("elementwise", "elementwise"): (("elements",), ()),
    ("reduce", "reduce"): (("elements",), ()),
    ("fused_moe", "full"): (("tokens",), ()),
    ("collective", None): ((), ("bytes",)),
}
# sglang environ.py SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD (v0.5.12, v0.5.19).
_SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD = 8192


def _append_absorbed_mla_attention(
        append, step, prefix, model, tp_size, token_count, q_lens, kv_lens,
        launch_policy, compressed_kv_update, runtime, attention_kv_layout,
        mla_graph_attributes, *, name, variant, source_prefix):
    """W_UK absorption, latent-space MLA attention, W_UV up-projection."""

    heads = model.num_attention_heads // tp_size
    q_down = append(
        WorkloadKind.LINEAR,
        f"{prefix}:{name}-q-down-proj",
        {"b": heads, "m": token_count, "n": model.kv_lora_rank,
         "k": model.qk_nope_head_dim},
        variant="batched_gemm",
        attributes={
            "a_batch_stride_elements": token_count * model.qk_nope_head_dim,
            "b_batch_stride_elements": (
                model.qk_nope_head_dim * model.kv_lora_rank),
            "batch_layout": "strided",
            "c_batch_stride_elements": token_count * model.kv_lora_rank,
            "has_bias": False,
            "launch_policy": launch_policy,
            "position": "mla_decode_q_down",
            "source_symbol": f"{source_prefix}.forward_absorb_prepare.w_kc",
            "weight_dtype": "bf16",
            "weight_group_size": 0,
        },
        extra_predecessors=(f"{step.step_id}:{prefix}:main-rotary",),
        use_previous=False,
    )
    attention = append(
        WorkloadKind.FLASH_ATTN,
        f"{prefix}:{name}-mla-attention",
        {
            "batch": len(q_lens),
            "q_len": max(q_lens),
            "kv_len": max(kv_lens),
            "qk_head_dim": model.kv_lora_rank + model.qk_rope_head_dim,
            "v_head_dim": model.kv_lora_rank,
            "num_query_heads": heads,
            "kv_lora_rank": model.kv_lora_rank,
        },
        variant=variant,
        attributes={
            "causal": True,
            "kv_dist": runtime.attention_kv_distribution,
            "kv_layout": attention_kv_layout,
            "kv_lens": kv_lens,
            "q_lens": q_lens,
            "source_symbol": f"{source_prefix}.forward_absorb_core",
            **mla_graph_attributes,
        },
        extra_predecessors=(compressed_kv_update, q_down),
        use_previous=False,
    )
    return append(
        WorkloadKind.LINEAR,
        f"{prefix}:{name}-v-up-proj",
        {"b": heads, "m": token_count, "n": model.v_head_dim,
         "k": model.kv_lora_rank},
        variant="batched_gemm",
        attributes={
            "a_batch_stride_elements": token_count * model.kv_lora_rank,
            "b_batch_stride_elements": model.kv_lora_rank * model.v_head_dim,
            "batch_layout": "strided",
            "c_batch_stride_elements": token_count * model.v_head_dim,
            "has_bias": False,
            "launch_policy": launch_policy,
            "position": "mla_decode_v_up",
            "source_symbol": f"{source_prefix}.forward_absorb_core.w_vc",
            "weight_dtype": "bf16",
            "weight_group_size": 0,
        },
        extra_predecessors=(attention,),
        use_previous=False,
    )


_TARGET_MLA_OP_LINEAR_POSITIONS = frozenset({
    "mla_decode_q_down", "mla_decode_v_up", "kv_b_proj",
})


def _scaled_token_value(node_id, name, value, logical, physical):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"target graph node {node_id!r} field {name} must be positive")
    if value % logical:
        raise NotImplementedError(
            f"target graph node {node_id!r} field {name}={value} is not a "
            f"multiple of the {logical} logical tokens and cannot represent "
            "graph padding")
    return value // logical * physical


def _apply_target_graph_token_domains(
        graph, step, graph_execution, serving_framework):
    """Emit a target-model step with its runtime CUDA-graph token domains.

    SGLang's EAGLE V2 target verify replays the decode CUDA graph captured at
    ``captured_req_width`` tokens per request, padding the batch to the next
    capture size (v0.5.19 ``decode_cuda_graph_runner.py:323``,
    ``:1346-1351``; ragged verify is DSPARK-only, ``spec_info.py:145-149``).
    Padded requests carry the attention backend's ``seq_len_fill_value``
    (``:1055``) and add no attention work.  Its draft extend and the EAGLE V2
    glue run outside that graph, so their nodes keep eager execution.
    """

    if not graph_execution.enabled:
        return graph
    uniform_decode = all(
        request.scheduled_tokens == graph_execution.uniform_decode_query_len
        for request in step.requests)
    shape = graph_execution.resolve(
        step.token_count, uniform_decode=uniform_decode)
    if shape.runtime_mode == "NONE":
        return graph
    if shape.runtime_mode not in ("FULL", "PIECEWISE"):
        raise ValueError(f"unknown graph runtime mode {shape.runtime_mode!r}")
    sglang_verify = False
    if any(request.mtp_proposer_mode != "disabled"
           for request in step.requests):
        if serving_framework != "sglang":
            raise NotImplementedError(
                "target-model graph padding with MTP is not source-closed")
        sglang_verify = True
    logical = shape.logical_tokens
    physical = shape.model_input_tokens
    full = shape.runtime_mode == "FULL"
    domain = {
        "graph_logical_tokens": logical,
        "graph_model_input_tokens": physical,
        "graph_runtime_mode": shape.runtime_mode,
    }
    nodes = []
    for node in graph.nodes:
        kind = node.kind.value
        attributes = dict(node.attributes)
        node_id = node.node_id
        if sglang_verify and (
                ":mtp-pass-" in node_id or ":sglang-eagle:" in node_id):
            nodes.append(node)
            continue
        if kind == "lm_head" or node_id.endswith(":lm-head:tp-all-gather"):
            nodes.append(node)
            continue
        if kind == "flash_attn":
            if node.variant == "mla_decode" and full:
                if (graph_execution.uniform_decode_query_len != 1
                        and not sglang_verify):
                    raise NotImplementedError(
                        "padded MLA decode requests are derived for one "
                        "query token per request")
                # cudagraph_dispatcher: num_reqs_padded = padded tokens /
                # uniform_decode_query_len (bounded by max_num_seqs, which
                # the capture sizes a decode step can select never exceed
                # here); padded requests carry query length 0 and seq_len 0.
                attributes.update(domain)
                nodes.append(dataclasses.replace(
                    node, attributes=attributes))
                continue
            if full:
                raise NotImplementedError(
                    f"{node.variant} under a FULL CUDA graph is not a vLLM "
                    "uniform-decode step")
            nodes.append(node)
            continue
        if (kind == "linear" and "position" in attributes
                and attributes["position"] in _TARGET_MLA_OP_LINEAR_POSITIONS):
            if not full:
                nodes.append(node)
                continue
            if attributes["position"] == "kv_b_proj":
                raise NotImplementedError(
                    "prefill kv_b_proj under a FULL CUDA graph is not a "
                    "vLLM uniform-decode step")
        key = (kind, None if kind == "collective" else node.variant)
        if key not in _TARGET_GRAPH_SCALED_FIELDS and (
                kind, node.variant) != ("linear", "batched_gemm"):
            raise NotImplementedError(
                f"target-model node {node_id!r} ({kind}/{node.variant}) "
                "cannot represent CUDA-graph padding")
        if (kind, node.variant) == ("linear", "batched_gemm"):
            shape_fields, attribute_fields = ("m",), (
                "a_batch_stride_elements", "c_batch_stride_elements")
        else:
            shape_fields, attribute_fields = _TARGET_GRAPH_SCALED_FIELDS[key]
        new_shape = dict(node.shape)
        for name in shape_fields:
            new_shape[name] = _scaled_token_value(
                node_id, name, new_shape[name], logical, physical)
        for name in attribute_fields:
            attributes[name] = _scaled_token_value(
                node_id, name, attributes[name], logical, physical)
        attributes.update(domain)
        nodes.append(dataclasses.replace(
            node, shape=new_shape, attributes=attributes))
    return WorkloadGraph(nodes, graph.edges, graph.block_repeats)


def build_workload(step: ScheduleStep, model: ModelConfig,
                   deployment: DeploymentPlan,
                   *, launch_policy: str,
                   kv_cache_events: Iterable[KVCacheEvent] = (),
                   runtime_inputs: RuntimeInputs,
                   routing_memo: ExpertRoutingMemo | None = None,
                   block_memo: BlockConstructionMemo | None = None) -> WorkloadGraph:
    if routing_memo is not None and not isinstance(
            routing_memo, ExpertRoutingMemo):
        raise TypeError("routing_memo must be ExpertRoutingMemo")
    if block_memo is not None and not isinstance(
            block_memo, BlockConstructionMemo):
        raise TypeError("block_memo must be BlockConstructionMemo")
    normalized_kv_cache_events = tuple(kv_cache_events)
    if model.model_type in ("kimi_k2", "glm_moe_dsa", "deepseek_v3"):
        logical_graph = _build_target_model_logical_workload(
            step,
            model,
            deployment,
            launch_policy=launch_policy,
            kv_cache_events=normalized_kv_cache_events,
            runtime=runtime_inputs,
            routing_memo=routing_memo,
            block_memo=block_memo,
        )
        return _apply_target_graph_token_domains(
            logical_graph, step, runtime_inputs.graph_execution,
            runtime_inputs.serving_framework)
    if model.attention_layout != "standard_gqa":
        raise NotImplementedError(
            "MLA/DSA model configs are parsed but workload emission requires "
            "the canonical MLA/DSA KernelSpec launch contract")
    if model.weight_dtype != model.compute_dtype:
        raise NotImplementedError(
            "quantized model weights require a canonical quantized kernel "
            "contract")
    if launch_policy not in ("injected_launch", "analytical_launch"):
        raise ValueError(
            f"unknown launch_policy {launch_policy!r}; expected "
            "injected_launch or analytical_launch")
    if step.pool_name not in deployment.worker_pools:
        raise ValueError(f"schedule step references unknown pool {step.pool_name!r}")
    scheduler_pool = deployment.worker_pools[step.pool_name]
    runtime = runtime_inputs
    kv_cache_events = normalized_kv_cache_events
    step_sha256 = schedule_step_sha256(step)
    if (model.num_experts > 0
            and runtime.expert_routing.source_kind == "not_applicable"):
        raise ValueError(
            "MoE model workload requires explicit expert routing input")
    if step.attention_kv_layout is None:
        attention_kv_layout = runtime.attention_kv_layout
    elif step.attention_kv_layout == runtime.attention_kv_layout:
        attention_kv_layout = step.attention_kv_layout
    else:
        raise ValueError(
            "scheduler attention KV layout disagrees with RuntimeInputs: "
            f"scheduler={step.attention_kv_layout!r}, "
            f"runtime={runtime.attention_kv_layout!r}")
    plan = scheduler_pool.parallel_plan
    logical_token_count = step.token_count
    uniform_decode = all(
        request.scheduled_tokens
        == runtime.graph_execution.uniform_decode_query_len
        for request in step.requests)
    graph_shape = runtime.graph_execution.resolve(
        logical_token_count,
        uniform_decode=uniform_decode,
    )
    target_query_lengths = tuple(
        request.scheduled_tokens for request in step.requests)
    target_kv_lengths = tuple(
        request.computed_tokens + request.scheduled_tokens
        for request in step.requests)
    attention_shape = graph_shape.attention_shape(
        target_query_lengths,
        target_kv_lengths,
        uniform_decode_query_len=(
            runtime.graph_execution.uniform_decode_query_len),
    )
    if graph_shape.padding_tokens:
        if model.attention_layout != "standard_gqa":
            raise NotImplementedError(
                "Graph padding currently supports standard GQA")
        if any(
                request.mtp_proposer_mode != "disabled"
                for request in step.requests):
            raise NotImplementedError(
                "Graph padding with MTP is not source-closed")
    token_count = graph_shape.model_input_tokens

    def graph_token_domain(
        logical_tokens: int,
        model_input_tokens: int,
    ) -> dict[str, object]:
        """Declare one node's effective-work and captured-tensor domains."""

        for name, value in (
                ("logical_tokens", logical_tokens),
                ("model_input_tokens", model_input_tokens)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"graph {name} must be an integer")
            if value <= 0:
                raise ValueError(f"graph {name} must be positive")
        if model_input_tokens < logical_tokens:
            raise ValueError("graph model-input tokens cannot shrink work")
        if graph_shape.runtime_mode == "NONE" and (
                logical_tokens != model_input_tokens):
            raise ValueError("eager execution cannot declare graph padding")
        # A replayed graph is a replayed graph whether or not the selected
        # capture size happens to equal the logical token count.  Bucket
        # rounding decides ``padding_tokens``; it does not decide whether the
        # runtime launches this call one at a time or replays it as a node of
        # a captured program.  Only ``runtime_mode == "NONE"`` means eager,
        # so only that case has no graph domain to declare.
        if graph_shape.runtime_mode == "NONE":
            return {}
        return {
            "graph_logical_tokens": logical_tokens,
            "graph_model_input_tokens": model_input_tokens,
            "graph_runtime_mode": graph_shape.runtime_mode,
        }
    dtype = model.compute_dtype
    if (model.model_type == "glm4_moe"
            and model.routed_scaling_factor != 1.0):
        raise NotImplementedError(
            "GLM routed_scaling_factor != 1.0 requires a versioned fused-MoE "
            "launch observation; Serving will not invent a scale kernel")
    phases = {request.phase for request in step.requests}
    if phases == {"prefill"}:
        phase = "prefill"
    elif phases == {"decode"}:
        phase = "decode"
    elif phases == {"prefill", "decode"}:
        phase = "mixed"
    else:
        raise ValueError(f"unknown workload request phases: {sorted(phases)}")
    placement_phase = worker_role_phase(scheduler_pool.role)
    # vLLM labels replay of evicted decode KV as prefill-shaped compute.  The
    # work remains placed on the decode Attention/FFN pair after PD handoff.
    pd_af_decode_recompute = (
        deployment.mode == "pd_af_disagg"
        and placement_phase == "decode"
        and phase in ("prefill", "mixed")
        and all(
            request.phase != "prefill"
            or request.resumed_from_preemption
            or request.recomputed_tokens > 0
            for request in step.requests
        )
    )
    pd_prefill_async_tail_batch = (
        deployment.mode == "pd_disagg"
        and placement_phase == "prefill"
        and phase == "mixed"
        and all(
            request.phase == "prefill"
            or is_non_mtp_async_terminal_placeholder(request)
            for request in step.requests
        )
    )
    if (phase == "mixed"
            and deployment.mode not in ("colocated", "af_disagg")
            and not pd_af_decode_recompute
            and not pd_prefill_async_tail_batch):
        raise NotImplementedError(
            "mixed prefill/decode batches are currently supported only for "
            "colocated, standalone Attention/FFN-disaggregated deployment, "
            "or validated decode-side recompute in PD+AF")
    nodes = []
    edges = []
    block_run_drafts = []
    active_block_run = None
    previous = None
    previous_rank_frontiers = ()
    residual_source = None
    residual_rank_frontiers = ()
    previous_pool_name = None
    previous_pipeline_stage = None
    active_pipeline_stage = 0
    transfer_index = 0
    # Flips to False only when the versioned discrete emission observation
    # licenses a zero-count step-head input (helper contract §2.3): the step
    # input boundary is then not a graph node, so the first RMSNorm has no
    # in-graph producer and the initial-residual checks must accept that.
    step_input_embedding_emitted = True

    def pool_for(operator_group):
        if deployment.mode not in ("af_disagg", "pd_af_disagg"):
            return scheduler_pool
        scheduler_group = worker_role_operator_group(scheduler_pool.role)
        if scheduler_group != "attention":
            raise ValueError(
                f"{deployment.mode} Scheduler pool must own Attention; "
                f"{scheduler_pool.name!r} owns {scheduler_group!r}")
        if deployment.mode == "pd_af_disagg":
            if placement_phase != phase and not pd_af_decode_recompute:
                raise ValueError(
                    "PD+AF Scheduler pool placement phase disagrees with "
                    "scheduled compute phase: "
                    f"pool={scheduler_pool.name!r}, compute_phase={phase!r}")
        if operator_group == "attention":
            return scheduler_pool
        if operator_group != "ffn":
            raise ValueError(
                f"unknown AF operator group: {operator_group!r}")
        routes = [
            route for route in deployment.routes
            if route.payload == "activation"
            and route.source_pool == scheduler_pool.name
            and worker_role_operator_group(
                deployment.worker_pools[route.destination_pool].role) == "ffn"
        ]
        if len(routes) != 1:
            raise ValueError(
                "AF Scheduler pool requires exactly one outgoing route to "
                f"its FFN peer: {scheduler_pool.name!r}; found {len(routes)}")
        return deployment.worker_pools[routes[0].destination_pool]

    def activation_route(source_pool_name, destination_pool_name):
        routes = [
            route for route in deployment.routes
            if route.source_pool == source_pool_name
            and route.destination_pool == destination_pool_name
            and route.payload == "activation"
        ]
        if len(routes) != 1:
            raise ValueError(
                "AF pool transition requires exactly one activation route from "
                f"{source_pool_name!r} to {destination_pool_name!r}")
        return routes[0]

    attention_plan = pool_for("attention").parallel_plan
    ffn_plan = pool_for("ffn").parallel_plan
    attention_tp = attention_plan.config.tensor_parallel_size
    ffn_tp = ffn_plan.config.tensor_parallel_size
    ep_rank_local_emission = ffn_plan.config.enable_expert_parallel
    tp_rank_local_emission = attention_tp > 1 or ffn_tp > 1
    af_tp2_rank_local_emission = is_validated_standalone_af_tp2_scope(
        deployment, model)
    af_ep2_rank_local_emission = is_validated_standalone_af_ep2_scope(
        deployment, model)
    af_rank_local_emission = (
        af_tp2_rank_local_emission or af_ep2_rank_local_emission)
    tp1_tp2_multi_pool_emission = (
        is_validated_tp1_tp2_multi_pool_scope(deployment, model))
    pipeline_size = plan.config.pipeline_parallel_size
    pp_rank_local_emission = pipeline_size > 1
    combined_tp_pp_emission = (
        tp_rank_local_emission and pp_rank_local_emission)
    if combined_tp_pp_emission:
        if deployment.mode != "colocated" or len(deployment.worker_pools) != 1:
            raise NotImplementedError(
                "rank-local TP=2, PP=2 emission requires one colocated pool")
        parallelism = scheduler_pool.parallel_plan.config
        if (scheduler_pool.hardware_id != "nvidia_a800"
                or parallelism.tensor_parallel_size != 2
                or parallelism.pipeline_parallel_size != 2
                or parallelism.expert_parallel_size != 1
                or not parallelism.is_rank_local
                or model.model_type != "qwen3"
                or model.num_experts != 0
                or model.attention_layout != "standard_gqa"
                or model.compute_dtype != "bf16"
                or model.weight_dtype != "bf16"):
            raise NotImplementedError(
                "rank-local combined emission supports exactly nvidia_a800 "
                "dense BF16 Qwen3 with TP=2, PP=2 and EP=SP=CP=DP=1")
        expected_tensor_groups = ((0, 1), (2, 3))
        expected_pipeline_groups = ((0, 2), (1, 3))
        if (plan.rank_groups.tensor != expected_tensor_groups
                or plan.rank_groups.pipeline != expected_pipeline_groups):
            raise RuntimeError(
                "TP=2, PP=2 logical rank groups disagree with pinned vLLM "
                "DP x PP x CP x TP ordering")
        for stage_index in range(pipeline_size):
            if len(scheduler_pool.physical_ranks_for_pipeline_stage(
                    stage_index)) != 2:
                raise RuntimeError(
                    "TP=2, PP=2 emission requires two physical ranks per "
                    "pipeline stage")
        scheduler_pool.pipeline_p2p_for(0, 1)
    elif tp_rank_local_emission:
        if (not af_rank_local_emission
                and not tp1_tp2_multi_pool_emission
                and (deployment.mode != "colocated"
                     or len(deployment.worker_pools) != 1)):
            raise NotImplementedError(
                "rank-local TP emission currently requires one colocated "
                "pool or the validated standalone AF TP=2 slice, or matching "
                "TP1/TP2 PD or replicated pools")
        if (not af_rank_local_emission
                and not ep_rank_local_emission
                and model.model_type != "qwen3"
                and scheduler_pool.hardware_id != "nvidia_a800"):
            raise NotImplementedError(
                "rank-local TP emission for non-Qwen3 models currently "
                "requires nvidia_a800")
        parallelism = scheduler_pool.parallel_plan.config
        if (parallelism.tensor_parallel_size != 2
                or parallelism.pipeline_parallel_size != 1
                or not parallelism.is_rank_local):
            raise NotImplementedError(
                "rank-local TP/EP workload emission currently requires "
                "TP=2 and PP=SP=CP=DP=1")
        if attention_tp != 2 or ffn_tp != 2:
            raise RuntimeError(
                "TP plan disagrees between Attention and FFN")
        if af_rank_local_emission:
            for pool in (pool_for("attention"), pool_for("ffn")):
                for operation in ("all_reduce", "all_gather"):
                    pool.collective_for(operation)
        if ep_rank_local_emission:
            ep_parallelism = ffn_plan.config
            if (ep_parallelism.expert_parallel_size != 2
                    or not is_validated_qwen3_moe_ep2_model_scope(model)):
                raise NotImplementedError(
                    "rank-local EP workload emission currently supports "
                    "BF16 Qwen3MoE with TP=EP=2, an even expert count, "
                    "valid top-k, all-sparse decoder layers, and no shared "
                    "experts")
            expected_expert_mapping = tuple(
                expert_rank
                for expert_rank in range(2)
                for _ in range(model.num_experts // 2)
            )
            if (ffn_plan.expert_to_shard != expected_expert_mapping
                    or ffn_plan.rank_groups.expert != ((0, 1),)
                    or ffn_plan.rank_groups.tensor != ((0, 1),)):
                raise NotImplementedError(
                    "rank-local EP workload emission requires the pinned "
                    "linear expert placement on the TP rank group")
        elif (parallelism.expert_parallel_size != 1
              or not (
                  (
                      model.model_type == "qwen3"
                      and model.compute_dtype in ("bf16", "fp16")
                      and model.weight_dtype == model.compute_dtype
                  )
                  or (
                      is_validated_llama_tp2_model_scope(model)
                      and model.compute_dtype == "bf16"
                      and model.weight_dtype == "bf16"
                  )
              )
              or model.num_experts != 0
              or model.attention_layout != "standard_gqa"):
            raise NotImplementedError(
                "rank-local TP emission currently supports dense BF16 or "
                "FP16 Qwen3 with matching compute and weight dtypes, or the "
                "exact source-backed Llama fixture in BF16, with standard "
                "GQA")
    if pp_rank_local_emission and not combined_tp_pp_emission:
        if deployment.mode != "colocated" or len(deployment.worker_pools) != 1:
            raise NotImplementedError(
                "rank-local PP emission currently requires one colocated pool")
        if scheduler_pool.hardware_id != "nvidia_a800":
            raise NotImplementedError(
                "rank-local PP emission currently requires nvidia_a800")
        parallelism = scheduler_pool.parallel_plan.config
        if (parallelism.pipeline_parallel_size < 2
                or parallelism.tensor_parallel_size != 1
                or parallelism.expert_parallel_size != 1
                or not parallelism.is_rank_local):
            raise NotImplementedError(
                "rank-local PP emission currently supports the validated "
                "uniform TP=1 path with PP>=2 and EP=SP=CP=DP=1")
        if (model.model_type != "qwen3"
                or model.num_experts != 0
                or model.tie_word_embeddings
                or model.attention_layout != "standard_gqa"
                or model.compute_dtype != "bf16"
                or model.weight_dtype != "bf16"):
            raise NotImplementedError(
                "rank-local PP emission currently supports untied dense BF16 "
                "Qwen3 with standard GQA")
        if model.num_hidden_layers % pipeline_size:
            raise NotImplementedError(
                "the validated uniform TP=1 PP workload path requires "
                "decoder layers divisible by pipeline parallel size")
        layers_per_stage = model.num_hidden_layers // pipeline_size
        expected_placement = tuple(
            layer // layers_per_stage
            for layer in range(model.num_hidden_layers)
        )
        if plan.layer_to_pipeline_stage != expected_placement:
            raise NotImplementedError(
                "the validated uniform TP=1 PP workload path requires "
                "contiguous equal-sized layer placement")
        for stage_index in range(pipeline_size):
            if len(scheduler_pool.physical_ranks_for_pipeline_stage(
                    stage_index)) != 1:
                raise NotImplementedError(
                    "rank-local PP emission requires exactly one physical "
                    "rank per pipeline stage")
        tuple(
            scheduler_pool.pipeline_p2p_for(
                source_stage, source_stage + 1)
            for source_stage in range(pipeline_size - 1)
        )
    rank_local_emission = tp_rank_local_emission or pp_rank_local_emission
    rank_local_host_kv_emission = (
        is_validated_colocated_tp2_host_kv_scope(deployment, model))
    if rank_local_emission and any(
            any((
                request.mtp_proposed_tokens,
                request.mtp_verified_tokens,
                request.mtp_accepted_tokens,
                request.mtp_proposer_tokens,
                request.mtp_next_proposed_tokens,
                request.mtp_proposer_mode != "disabled",
                request.mtp_acceptance_source_kind != "disabled",
                request.mtp_acceptance_source_id != "disabled",
            ))
            for request in step.requests):
        raise NotImplementedError(
            "rank-local TP/EP/PP emission does not yet support MTP")
    if (rank_local_emission
            and not rank_local_host_kv_emission
            and any(
            event.action in (KVCacheAction.LOAD, KVCacheAction.EVICT)
            for event in kv_cache_events)):
        raise NotImplementedError(
            "rank-local TP/EP/PP emission does not yet support KV offload")
    if tp_rank_local_emission:
        scheduler_pool.collective_for("all_reduce")
        scheduler_pool.collective_for("all_gather")

    def kv_heads_per_rank() -> int:
        return model.kv_heads_per_tensor_parallel_rank(attention_tp)

    def append(kind, suffix, shape, *, variant=None, attributes=None,
               operator_group="attention", pool_override=None,
               activation_token_count=None, tp_collective_operation=None,
               tp_collective_bytes=None, establishes_residual=False,
               token_domain=None):
        nonlocal previous, previous_rank_frontiers, previous_pool_name
        nonlocal residual_source, residual_rank_frontiers
        nonlocal previous_pipeline_stage
        nonlocal transfer_index
        previous_before_append = previous
        rank_frontiers_before_append = previous_rank_frontiers
        pool = (
            pool_override if pool_override is not None
            else pool_for(operator_group))
        attributes = _with_launch_policy(kind, attributes, launch_policy)
        attributes = _with_gemm_semantics(
            kind,
            attributes,
            pool.parallel_plan.config.tensor_parallel_size,
            pool.parallel_plan.config.expert_parallel_size,
        )
        attributes = _with_eager_attention_parallelism(
            kind,
            attributes,
            pool.parallel_plan.config.tensor_parallel_size,
            pool.parallel_plan.config.expert_parallel_size,
        )
        if token_domain is not None:
            if (not isinstance(token_domain, tuple)
                    or len(token_domain) != 2):
                raise TypeError(
                    "workload graph token domain must be a two-integer tuple")
            domain_attributes = graph_token_domain(*token_domain)
            conflicting = {
                name: (attributes[name], value)
                for name, value in domain_attributes.items()
                if name in attributes and attributes[name] != value
            }
            if conflicting:
                raise ValueError(
                    "workload graph token-domain attributes conflict: "
                    f"{conflicting!r}")
            attributes.update(domain_attributes)
        graph_domain_attributes = {
            name: attributes[name]
            for name in (
                "graph_logical_tokens", "graph_model_input_tokens",
                "graph_runtime_mode",
            )
            if name in attributes
        }
        if activation_token_count is None:
            transfer_token_count = token_count
        else:
            transfer_token_count = activation_token_count
        if tp_rank_local_emission:
            if (pool.name != scheduler_pool.name
                    and not af_rank_local_emission):
                raise NotImplementedError(
                    "rank-local TP emission does not support cross-pool nodes")
            if (tp_collective_operation is None) != (
                    tp_collective_bytes is None):
                raise ValueError(
                    "TP collective operation and bytes must be supplied "
                    "together")
            if combined_tp_pp_emission:
                stage_logical_ranks = plan.logical_ranks_for_pipeline_stage(
                    active_pipeline_stage)
            else:
                stage_logical_ranks = tuple(range(attention_tp))
            if len(stage_logical_ranks) != attention_tp:
                raise RuntimeError(
                    "active pipeline stage does not contain one logical rank "
                    "per tensor-parallel lane")
            if (af_rank_local_emission
                    and previous_rank_frontiers
                    and previous_pool_name != pool.name):
                if previous_pool_name is None:
                    raise RuntimeError(
                        "AF TP=2 workload lost the source worker pool")
                residual_transfer_frontiers = residual_rank_frontiers
                residual_frontiers_observation_suppressed = False
                if (len(residual_transfer_frontiers) != attention_tp
                        or any(len(frontier) != 1
                               for frontier in residual_transfer_frontiers)):
                    if (residual_transfer_frontiers == ()
                            and not step_input_embedding_emitted):
                        # Observation-licensed zero-count step-head input
                        # (helper contract §2.3): the residual buffer
                        # originates in an earlier capture window, so the
                        # first per-lane residual transfer anchors its ready
                        # edge on the lane's hidden-state producer.
                        residual_transfer_frontiers = previous_rank_frontiers
                        residual_frontiers_observation_suppressed = True
                    else:
                        raise RuntimeError(
                            "AF TP=2 workload lost one residual producer "
                            "per tensor-parallel lane")
                source_pool = deployment.worker_pools[previous_pool_name]
                route = activation_route(source_pool.name, pool.name)
                transfer_bytes = (
                    transfer_token_count
                    * model.hidden_size
                    * model.dtype_bytes
                )
                transfer_frontiers = []
                boundary_index = transfer_index
                transfer_index += 1
                for lane_index, logical_rank in enumerate(
                        stage_logical_ranks):
                    if len(previous_rank_frontiers[lane_index]) != 1:
                        raise RuntimeError(
                            "AF TP=2 hidden-state boundary requires exactly "
                            "one producer per tensor-parallel lane")
                    source_rank = source_pool.physical_rank_for_logical_rank(
                        logical_rank)
                    destination_rank = pool.physical_rank_for_logical_rank(
                        logical_rank)
                    lane_transfer_ids = []
                    tensor_producers = (
                        (
                            "hidden_states",
                            previous_rank_frontiers[lane_index][0],
                        ),
                        (
                            "residual",
                            residual_transfer_frontiers[lane_index][0],
                        ),
                    )
                    for tensor_name, producer_id in tensor_producers:
                        transfer_id = (
                            f"{step.step_id}:af-transfer-{boundary_index}:"
                            f"{tensor_name.replace('_', '-')}:"
                            f"tp-rank-{lane_index}"
                        )
                        nodes.append(WorkloadNode(
                            transfer_id,
                            WorkloadKind.AF_ACTIVATION_TRANSFER,
                            source_pool.name,
                            {},
                            rank=source_rank,
                            attributes={
                                "algorithm": route.algorithm,
                                "bytes": transfer_bytes,
                                "bytes_semantics": "payload",
                                "backend": route.backend,
                                "backend_config_sha256": (
                                    route.backend_config_sha256),
                                "direction": "point_to_point",
                                "link": route.link,
                                "topology": route.topology,
                                "participants": (
                                    source_rank, destination_rank),
                                "operation": "send_recv",
                                "source_rank": source_rank,
                                "destination_rank": destination_rank,
                                "ring_order": (),
                                "reduction": None,
                                "resource_id": route.resource_id,
                                "destination_pool": pool.name,
                                "payload_tensor": {
                                    "name": tensor_name,
                                    "shape": (
                                        transfer_token_count,
                                        model.hidden_size,
                                    ),
                                    "dtype": dtype,
                                    "bytes": transfer_bytes,
                                },
                                **({
                                    "residual_anchor": (
                                        RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED
                                    ),
                                } if (
                                    tensor_name == "residual"
                                    and residual_frontiers_observation_suppressed
                                ) else {}),
                                **graph_domain_attributes,
                            },
                        ))
                        edges.append(WorkloadEdge(
                            producer_id, transfer_id, "ready"))
                        lane_transfer_ids.append(transfer_id)
                    transfer_frontiers.append(tuple(lane_transfer_ids))
                previous_rank_frontiers = tuple(transfer_frontiers)
            if (combined_tp_pp_emission
                    and previous_rank_frontiers
                    and previous_pipeline_stage != active_pipeline_stage):
                if (previous_pipeline_stage, active_pipeline_stage) != (0, 1):
                    raise RuntimeError(
                        "TP=2, PP=2 emission supports one forward 0->1 "
                        "pipeline boundary")
                source_logical_ranks = (
                    plan.logical_ranks_for_pipeline_stage(
                        previous_pipeline_stage))
                if (len(source_logical_ranks) != attention_tp
                        or len(previous_rank_frontiers) != attention_tp):
                    raise RuntimeError(
                        "TP=2, PP=2 source frontier disagrees with TP size")
                p2p = pool.pipeline_p2p_for(
                    previous_pipeline_stage, active_pipeline_stage)
                total_tensor_elements = (
                    transfer_token_count * model.hidden_size)
                if total_tensor_elements % attention_tp:
                    raise ValueError(
                        "pipeline tensor elements must divide TP size")
                wire_elements = total_tensor_elements // attention_tp
                wire_bytes = wire_elements * model.dtype_bytes
                p2p_ids_by_tensor = {}
                for tensor_name in ("hidden_states", "residual"):
                    p2p_ids = []
                    for lane_index, (
                            source_logical_rank,
                            destination_logical_rank,
                    ) in enumerate(zip(
                        source_logical_ranks,
                        stage_logical_ranks,
                    )):
                        source_rank = pool.physical_rank_for_logical_rank(
                            source_logical_rank)
                        destination_rank = pool.physical_rank_for_logical_rank(
                            destination_logical_rank)
                        transfer_id = (
                            f"{step.step_id}:pp-{previous_pipeline_stage}-to-"
                            f"{active_pipeline_stage}:"
                            f"{tensor_name.replace('_', '-')}:"
                            f"tp-rank-{lane_index}")
                        transfer = WorkloadNode(
                            transfer_id,
                            WorkloadKind.P2P,
                            pool.name,
                            {},
                            rank=source_rank,
                            attributes={
                                "algorithm": p2p.algorithm,
                                "bytes": wire_bytes,
                                "bytes_semantics": "payload",
                                "backend": p2p.backend,
                                "backend_config_sha256": (
                                    p2p.backend_config_sha256),
                                "direction": "point_to_point",
                                "link": p2p.link,
                                "topology": p2p.topology,
                                "participants": (
                                    source_rank, destination_rank),
                                "operation": "send_recv",
                                "source_rank": source_rank,
                                "destination_rank": destination_rank,
                                "ring_order": (),
                                "reduction": None,
                                "resource_id": p2p.resource_id,
                                "source_stage": previous_pipeline_stage,
                                "destination_stage": active_pipeline_stage,
                                "payload_tensor": {
                                    "name": tensor_name,
                                    "shape": (wire_elements,),
                                    "dtype": dtype,
                                    "bytes": wire_bytes,
                                },
                                **graph_domain_attributes,
                            },
                        )
                        nodes.append(transfer)
                        for predecessor in previous_rank_frontiers[lane_index]:
                            edges.append(WorkloadEdge(
                                predecessor, transfer_id, "ready"))
                        p2p_ids.append(transfer_id)
                    p2p_ids_by_tensor[tensor_name] = tuple(p2p_ids)
                destination_participants = pool.physical_rank_group(
                    "tensor", stage_logical_ranks[0])
                expected_destination_ranks = (
                    pool.physical_ranks_for_pipeline_stage(
                        active_pipeline_stage))
                if destination_participants != expected_destination_ranks:
                    raise RuntimeError(
                        "destination TP group disagrees with pipeline stage "
                        "rank ordering")
                all_gather = pool.collective_for("all_gather")
                boundary_collective_ids = []
                for tensor_name in ("hidden_states", "residual"):
                    collective_id = (
                        f"{step.step_id}:pp-{previous_pipeline_stage}-to-"
                        f"{active_pipeline_stage}:"
                        f"{tensor_name.replace('_', '-')}:tp-all-gather")
                    nodes.append(WorkloadNode(
                        collective_id,
                        WorkloadKind.COLLECTIVE,
                        pool.name,
                        {},
                        rank=destination_participants[0],
                        attributes={
                            "algorithm": all_gather.algorithm,
                            "bytes": wire_bytes,
                            "bytes_semantics": "per_rank_input",
                            "backend": all_gather.backend,
                            "backend_config_sha256": (
                                all_gather.backend_config_sha256),
                            "direction": "collective",
                            "link": all_gather.link,
                            "topology": all_gather.topology,
                            "participants": destination_participants,
                            "operation": all_gather.operation,
                            "source_rank": None,
                            "destination_rank": None,
                            "ring_order": _collective_ring_order(
                                all_gather.algorithm,
                                destination_participants,
                            ),
                            "reduction": None,
                            "resource_id": all_gather.resource_id,
                            **graph_domain_attributes,
                        },
                    ))
                    for transfer_id in p2p_ids_by_tensor[tensor_name]:
                        edges.append(WorkloadEdge(
                            transfer_id, collective_id, "barrier"))
                    boundary_collective_ids.append(collective_id)
                previous_rank_frontiers = tuple(
                    tuple(boundary_collective_ids)
                    for _ in stage_logical_ranks)

            local_node_ids = []
            for lane_index, logical_rank in enumerate(stage_logical_ranks):
                physical_rank = pool.physical_rank_for_logical_rank(logical_rank)
                node_id = (
                    f"{step.step_id}:{suffix}:tp-rank-{lane_index}")
                node = WorkloadNode(
                    node_id,
                    kind,
                    pool.name,
                    shape,
                    dtype=dtype,
                    variant=variant,
                    rank=physical_rank,
                    attributes=attributes,
                )
                nodes.append(node)
                if previous_rank_frontiers:
                    for predecessor in previous_rank_frontiers[lane_index]:
                        edges.append(WorkloadEdge(predecessor, node_id))
                local_node_ids.append(node_id)
            if establishes_residual:
                if (kind is not WorkloadKind.NORM
                        or variant not in ("rmsnorm", "add_rmsnorm")
                        or tp_collective_operation is not None):
                    raise ValueError(
                        "residual producer must be a rank-local RMSNorm "
                        "without a following collective")
                if variant == "rmsnorm":
                    if (len(rank_frontiers_before_append) != attention_tp
                            or any(len(frontier) != 1 for frontier in (
                                rank_frontiers_before_append))):
                        if (rank_frontiers_before_append == ()
                                and not step_input_embedding_emitted):
                            # Observation-licensed zero-count step-head
                            # input: no in-graph producer exists to anchor
                            # the initial residual.
                            residual_rank_frontiers = ()
                        else:
                            raise RuntimeError(
                                "initial residual requires exactly one "
                                "input producer per tensor-parallel lane")
                    else:
                        residual_rank_frontiers = (
                            rank_frontiers_before_append)
                elif variant == "add_rmsnorm":
                    residual_rank_frontiers = tuple(
                        (node_id,) for node_id in local_node_ids)
                else:
                    raise AssertionError(
                        "validated residual RMSNorm variant vanished")
            if tp_collective_operation is None:
                previous_rank_frontiers = tuple(
                    (node_id,) for node_id in local_node_ids)
            else:
                if (isinstance(tp_collective_bytes, bool)
                        or not isinstance(tp_collective_bytes, int)
                        or tp_collective_bytes <= 0):
                    raise ValueError(
                        "TP collective bytes must be a positive integer")
                collective = pool.collective_for(tp_collective_operation)
                participants = pool.physical_rank_group(
                    "tensor", stage_logical_ranks[0])
                if len(participants) != attention_tp:
                    raise RuntimeError(
                        "TP collective participants disagree with TP size")
                collective_id = (
                    f"{step.step_id}:{suffix}:tp-"
                    f"{tp_collective_operation.replace('_', '-')}")
                if collective.operation == "all_reduce":
                    reduction = "sum"
                elif collective.operation == "all_gather":
                    reduction = None
                else:
                    raise ValueError(
                        "TP emission supports only all_reduce and all_gather")
                collective_node = WorkloadNode(
                    collective_id,
                    WorkloadKind.COLLECTIVE,
                    pool.name,
                    {},
                    rank=participants[0],
                    attributes={
                        "algorithm": collective.algorithm,
                        "bytes": tp_collective_bytes,
                        "bytes_semantics": "per_rank_input",
                        "backend": collective.backend,
                        "backend_config_sha256": (
                            collective.backend_config_sha256),
                        "direction": "collective",
                        "link": collective.link,
                        "topology": collective.topology,
                        "participants": participants,
                        "operation": collective.operation,
                        "source_rank": None,
                        "destination_rank": None,
                        "ring_order": _collective_ring_order(
                            collective.algorithm, participants),
                        "reduction": reduction,
                        "resource_id": collective.resource_id,
                        **graph_domain_attributes,
                    },
                )
                nodes.append(collective_node)
                for local_node_id in local_node_ids:
                    edges.append(WorkloadEdge(
                        local_node_id, collective_id, "barrier"))
                previous_rank_frontiers = tuple(
                    (collective_id,) for _ in range(attention_tp))
            previous_pool_name = pool.name
            previous_pipeline_stage = active_pipeline_stage
            return
        if pp_rank_local_emission:
            if pool.name != scheduler_pool.name:
                raise NotImplementedError(
                    "rank-local PP emission does not support cross-pool nodes")
            if (tp_collective_operation is not None
                    or tp_collective_bytes is not None):
                raise ValueError(
                    "PP-only emission cannot contain TP collectives")
            stage_ranks = pool.physical_ranks_for_pipeline_stage(
                active_pipeline_stage)
            if len(stage_ranks) != 1:
                raise RuntimeError(
                    "validated PP stage must resolve to one physical rank")
            if (previous is not None
                    and previous_pipeline_stage != active_pipeline_stage):
                if previous_pipeline_stage is None:
                    raise RuntimeError(
                        "PP workload lost the previous pipeline stage")
                config = pool.pipeline_p2p_for(
                    previous_pipeline_stage, active_pipeline_stage)
                source_ranks = pool.physical_ranks_for_pipeline_stage(
                    previous_pipeline_stage)
                if len(source_ranks) != 1:
                    raise RuntimeError(
                        "validated PP source stage must resolve to one rank")
                source_rank = source_ranks[0]
                destination_rank = stage_ranks[0]
                tensor_bytes = (
                    transfer_token_count
                    * model.hidden_size
                    * model.dtype_bytes
                )
                for tensor_name in ("hidden_states", "residual"):
                    transfer_id = (
                        f"{step.step_id}:pp-{previous_pipeline_stage}-to-"
                        f"{active_pipeline_stage}:"
                        f"{tensor_name.replace('_', '-')}")
                    transfer = WorkloadNode(
                        transfer_id,
                        WorkloadKind.P2P,
                        pool.name,
                        {},
                        rank=source_rank,
                        attributes={
                            "algorithm": config.algorithm,
                            "bytes": tensor_bytes,
                            "bytes_semantics": "payload",
                            "backend": config.backend,
                            "backend_config_sha256": (
                                config.backend_config_sha256),
                            "direction": "point_to_point",
                            "link": config.link,
                            "topology": config.topology,
                            "participants": (
                                source_rank, destination_rank),
                            "operation": "send_recv",
                            "source_rank": source_rank,
                            "destination_rank": destination_rank,
                            "ring_order": (),
                            "reduction": None,
                            "resource_id": config.resource_id,
                            "source_stage": previous_pipeline_stage,
                            "destination_stage": active_pipeline_stage,
                            "payload_tensor": {
                                "name": tensor_name,
                                "shape": (
                                    transfer_token_count,
                                    model.hidden_size,
                                ),
                                "dtype": dtype,
                                "bytes": tensor_bytes,
                            },
                            **graph_domain_attributes,
                        },
                    )
                    nodes.append(transfer)
                    edges.append(WorkloadEdge(previous, transfer_id, "ready"))
                    previous = transfer_id
            node_id = f"{step.step_id}:{suffix}"
            node = WorkloadNode(
                node_id,
                kind,
                pool.name,
                shape,
                dtype=dtype,
                variant=variant,
                rank=stage_ranks[0],
                attributes=attributes,
            )
            nodes.append(node)
            if previous is not None:
                edges.append(WorkloadEdge(previous, node_id))
            previous = node_id
            previous_pool_name = pool.name
            previous_pipeline_stage = active_pipeline_stage
            return
        incoming_predecessors = () if previous is None else (previous,)
        if previous is not None and previous_pool_name != pool.name:
            route = activation_route(previous_pool_name, pool.name)
            source_pool = deployment.worker_pools[previous_pool_name]
            if source_pool.hardware_id != pool.hardware_id:
                raise NotImplementedError(
                    "cross-hardware AF activation transfer requires a "
                    "bilateral endpoint/link contract")
            if len(source_pool.ranks) != 1 or len(pool.ranks) != 1:
                raise NotImplementedError(
                    "multi-rank AF activation transfer requires explicit "
                    "rank-pair or repartition semantics")
            residual_anchor_observation_suppressed = False
            if residual_source is None:
                if not step_input_embedding_emitted:
                    # Observation-licensed zero-count step-head input
                    # (helper contract §2.3): the residual buffer originates
                    # in an earlier capture window, so the first residual
                    # transfer still launches and anchors its ready edge on
                    # the source pool's last in-window producer.
                    residual_anchor = previous
                    residual_anchor_observation_suppressed = True
                else:
                    raise RuntimeError(
                        "AF pool transition requires an explicit residual "
                        "producer")
            else:
                residual_anchor = residual_source
            boundary_index = transfer_index
            transfer_index += 1
            transfer_bytes = (
                transfer_token_count
                * model.hidden_size
                * model.dtype_bytes
            )
            transfer_ids = []
            for tensor_name, producer_id in (
                ("hidden_states", previous),
                ("residual", residual_anchor),
            ):
                transfer_id = (
                    f"{step.step_id}:af-transfer-{boundary_index}:"
                    f"{tensor_name.replace('_', '-')}"
                )
                transfer = WorkloadNode(
                    transfer_id,
                    WorkloadKind.AF_ACTIVATION_TRANSFER,
                    source_pool.name,
                    {},
                    rank=source_pool.ranks[0],
                    attributes={
                        "algorithm": route.algorithm,
                        "bytes": transfer_bytes,
                        "bytes_semantics": "payload",
                        "backend": route.backend,
                        "backend_config_sha256": route.backend_config_sha256,
                        "direction": "point_to_point",
                        "link": route.link,
                        "topology": route.topology,
                        "participants": source_pool.ranks + pool.ranks,
                        "operation": "send_recv",
                        "source_rank": source_pool.ranks[0],
                        "destination_rank": pool.ranks[0],
                        "ring_order": (),
                        "reduction": None,
                        "resource_id": route.resource_id,
                        "destination_pool": pool.name,
                        "payload_tensor": {
                            "name": tensor_name,
                            "shape": (
                                transfer_token_count,
                                model.hidden_size,
                            ),
                            "dtype": dtype,
                            "bytes": transfer_bytes,
                        },
                        **({
                            "residual_anchor": (
                                RESIDUAL_ANCHOR_OBSERVATION_SUPPRESSED),
                        } if (tensor_name == "residual"
                              and residual_anchor_observation_suppressed)
                        else {}),
                        **graph_domain_attributes,
                    },
                )
                nodes.append(transfer)
                edges.append(WorkloadEdge(
                    producer_id, transfer_id, "ready"))
                transfer_ids.append(transfer_id)
            incoming_predecessors = tuple(transfer_ids)
        node_id = f"{step.step_id}:{suffix}"
        node = WorkloadNode(
            node_id, kind, pool.name, shape, dtype=dtype,
            variant=variant, rank=pool.ranks[0], attributes=attributes)
        nodes.append(node)
        for predecessor in incoming_predecessors:
            edges.append(WorkloadEdge(predecessor, node_id))
        previous = node_id
        previous_pool_name = pool.name
        if establishes_residual:
            if (kind is not WorkloadKind.NORM
                    or variant not in ("rmsnorm", "add_rmsnorm")):
                raise ValueError(
                    "residual producer must be an RMSNorm workload node")
            if variant == "rmsnorm":
                if (previous_before_append is None
                        and step_input_embedding_emitted):
                    raise RuntimeError(
                        "initial residual requires an input producer")
                residual_source = previous_before_append
            elif variant == "add_rmsnorm":
                residual_source = node_id
            else:
                raise AssertionError(
                    "validated residual RMSNorm variant vanished")

    kv_transfer_ready_events = set()
    pending_evict_attributes = []
    scheduled_request_ids = {
        request.request_id for request in step.requests
    }
    for event in kv_cache_events:
        if event.action in (KVCacheAction.LOAD, KVCacheAction.EVICT):
            if scheduler_pool.kv_cache is None:
                raise ValueError(
                    "KV offload event requires a configured cache tier plan")
            if (len(scheduler_pool.ranks) != 1
                    and not rank_local_host_kv_emission):
                raise NotImplementedError(
                    "modeled Host KV offload currently requires one physical "
                    "rank per scheduler pool or the validated colocated "
                    "Qwen3 TP=2 rank-local scope")
            model_kv_bytes_per_token = (
                model.kv_bytes_per_token_per_parallel_rank(
                    scheduler_pool.parallel_plan.config.tensor_parallel_size,
                    scheduler_pool.parallel_plan.config.pipeline_parallel_size,
                )
            )
            if (scheduler_pool.kv_cache.bytes_per_token
                    != model_kv_bytes_per_token):
                raise ValueError(
                    "KV offload bytes_per_token disagrees with the model KV "
                    "layout")
            source = event.source_tier
            destination = event.destination_tier
            if source is None or destination is None:
                raise RuntimeError("validated KV offload event lacks endpoints")
            source_is_hbm = source.lower() == "hbm"
            destination_is_hbm = destination.lower() == "hbm"
            if (event.action is KVCacheAction.EVICT
                    and source_is_hbm and not destination_is_hbm):
                offload_tier_name = destination.lower()
            elif (event.action is KVCacheAction.LOAD
                    and destination_is_hbm and not source_is_hbm):
                offload_tier_name = source.lower()
            else:
                raise ValueError(
                    "KV load/evict direction disagrees with its tier endpoints")
            if offload_tier_name != "host":
                raise NotImplementedError(
                    "modeled KV offload currently supports only the Host tier")
            if event.tier.lower() != offload_tier_name:
                raise ValueError(
                    "KV offload event tier disagrees with its non-HBM endpoint")
            matching_tiers = [
                tier for tier in scheduler_pool.kv_cache.tiers
                if tier.name.lower() == offload_tier_name
            ]
            if len(matching_tiers) != 1:
                raise ValueError(
                    "KV offload event must match exactly one configured tier")
            offload_tier = matching_tiers[0]
            if offload_tier.link is None:
                raise ValueError("KV offload tier requires a physical link")
            hbm_tiers = [
                tier for tier in scheduler_pool.kv_cache.tiers
                if tier.name.lower() == "hbm"
            ]
            if len(hbm_tiers) != 1:
                raise ValueError(
                    "KV offload event requires exactly one configured HBM tier")
            hbm_tier = hbm_tiers[0]
            if offload_tier.block_size_tokens != hbm_tier.block_size_tokens:
                raise NotImplementedError(
                    "modeled Host KV offload currently requires equal HBM "
                    "and Host block sizes")
            transfer_token_count = event.token_end - event.token_start
            if (event.token_start % hbm_tier.block_size_tokens
                    or transfer_token_count % hbm_tier.block_size_tokens):
                raise ValueError(
                    "KV offload event must cover aligned full HBM blocks")
            expected_bytes = (
                transfer_token_count
                * scheduler_pool.kv_cache.bytes_per_token
            )
            if event.bytes != expected_bytes:
                raise ValueError(
                    "KV offload event bytes disagree with its token range")
            if event.ready_event in kv_transfer_ready_events:
                raise ValueError("KV offload ready_event values must be unique")
            kv_transfer_ready_events.add(event.ready_event)
            if (event.action is KVCacheAction.LOAD
                    and event.request_id not in scheduled_request_ids):
                raise ValueError(
                    "modeled KV demand load must target an active request in "
                    "the same scheduler step")
            attributes = {
                "bytes": event.bytes,
                "direction": event.action.value,
                "source_tier": event.source_tier,
                "destination_tier": event.destination_tier,
                "tier": event.tier,
                "algorithm": "dma",
                "link": offload_tier.link,
                "request_id": event.request_id,
                "token_start": event.token_start,
                "token_end": event.token_end,
                "ready_event": event.ready_event,
            }
            if event.action is KVCacheAction.LOAD:
                append(
                    WorkloadKind.KV_OFFLOAD,
                    f"kv-offload:{event.ready_event}",
                    {},
                    attributes=attributes,
                    pool_override=scheduler_pool,
                )
            elif event.action is KVCacheAction.EVICT:
                pending_evict_attributes.append((event.ready_event, attributes))
            else:
                raise RuntimeError(
                    f"unknown KV transfer action: {event.action.value!r}")

    query_heads = model.num_attention_heads // attention_tp
    kv_heads = kv_heads_per_rank()

    # Helper contract §2.3 (model input boundary): the vLLM source path
    # embeds every scheduler-step input, so the source predicate count is one
    # launch per step. Where the vendor dispatch layer measurably diverges
    # (A800 prefill capture windows show no step-head input-prep launches),
    # the versioned discrete emission observation licenses the divergence —
    # including an authorized zero — and the embedding node plus its TP
    # collective are not emitted for that (hardware, phase).
    embedding_count = helper_emission_count(
        pool_for("attention").hardware_id,
        "token_embedding_gather",
        phase,
        source_count=1,
        model_type=model.model_type,
    )
    if embedding_count != 0:
        if embedding_count != 1:
            raise ValueError(
                "token embedding helper emission count must be 0 or 1; "
                f"resolved {embedding_count}")
        append(WorkloadKind.GATHER, "token-embedding", {
            "elements": token_count * model.hidden_size,
            "rows": token_count,
            "hidden": model.hidden_size,
        }, variant="embedding", attributes={
            "gather_factor": 1,
            "index_count": token_count,
            "index_bytes": 8,
            "predicate": "per-step scheduler input embedding",
            "scattered": True,
        }, activation_token_count=token_count,
               token_domain=(logical_token_count, token_count),
               tp_collective_operation=(
                   "all_reduce" if tp_rank_local_emission else None),
               tp_collective_bytes=(
                   tp_collective_payload_bytes(
                       model, "all_reduce", token_count)
                   if tp_rank_local_emission else None))
    else:
        step_input_embedding_emitted = False

    def append_dense_mlp(
        prefix,
        block_token_count,
        logical_block_token_count,
        intermediate_size,
    ):
        if intermediate_size % ffn_tp:
            raise ValueError(
                "dense MLP intermediate size must divide FFN TP size")
        local_intermediate = intermediate_size // ffn_tp
        append(WorkloadKind.LINEAR, f"{prefix}:gate-up", {
            "b": 1,
            "m": block_token_count,
            "n": 2 * local_intermediate,
            "k": model.hidden_size,
        }, variant="gemm", attributes={
            "has_bias": model.mlp_bias,
            "position": "gate_up_proj",
            "launch_policy": launch_policy,
        }, operator_group="ffn", activation_token_count=block_token_count,
               token_domain=(logical_block_token_count, block_token_count))
        append(WorkloadKind.ACTIVATION, f"{prefix}:activation", {
            "elements": block_token_count * local_intermediate,
            "tok": block_token_count,
            "intermediate": local_intermediate,
        }, variant="silu", operator_group="ffn",
               activation_token_count=block_token_count,
               token_domain=(logical_block_token_count, block_token_count))
        append(WorkloadKind.LINEAR, f"{prefix}:down", {
            "b": 1,
            "m": block_token_count,
            "n": model.hidden_size,
            "k": local_intermediate,
        }, variant="gemm", attributes={
            "has_bias": model.mlp_bias,
            "position": "down_proj",
            "launch_policy": launch_policy,
        }, operator_group="ffn", activation_token_count=block_token_count,
               token_domain=(logical_block_token_count, block_token_count),
               tp_collective_operation=(
                   "all_reduce" if tp_rank_local_emission else None),
               tp_collective_bytes=(
                   tp_collective_payload_bytes(
                       model, "all_reduce", block_token_count)
                   if tp_rank_local_emission else None))

    def append_expert_parallel_moe(
        prefix,
        block_token_count,
        logical_block_token_count,
        tokens_per_expert,
        invocation_id,
        routing_provenance,
    ):
        """Emit the full-input and local-routing EP workload contract."""

        nonlocal previous_rank_frontiers, previous_pool_name
        pool = pool_for("ffn")
        if (pool.name != scheduler_pool.name
                and not af_ep2_rank_local_emission):
            raise NotImplementedError(
                "rank-local EP workload emission does not support AF pools")
        expert_group = ffn_plan.rank_groups.for_rank("expert", 0)
        tensor_group = ffn_plan.rank_groups.for_rank("tensor", 0)
        if (expert_group != tensor_group
                or len(expert_group)
                != ffn_plan.config.expert_parallel_size):
            raise RuntimeError(
                "EP workload requires one shared TP/EP rank group")
        if len(previous_rank_frontiers) != len(expert_group):
            raise RuntimeError(
                "EP MoE workload requires one router frontier per rank")
        moe_node_ids = []
        for expert_rank, logical_rank in enumerate(expert_group):
            physical_rank = pool.physical_rank_for_logical_rank(logical_rank)
            owned_experts = tuple(
                expert_id
                for expert_id, owner_rank in enumerate(
                    ffn_plan.expert_to_shard)
                if owner_rank == expert_rank
            )
            local_tokens_per_expert = tuple(
                count if expert_id in owned_experts else 0
                for expert_id, count in enumerate(tokens_per_expert)
            )
            node_id = f"{invocation_id}:ep-rank-{expert_rank}"
            nodes.append(WorkloadNode(
                node_id,
                WorkloadKind.FUSED_MOE,
                pool.name,
                {
                    "tokens": block_token_count,
                    "experts": model.num_experts,
                    "topk": model.num_experts_per_tok,
                    "hidden": model.hidden_size,
                    "intermediate": routed_expert_local_intermediate(
                        model,
                        ffn_plan.config.tensor_parallel_size,
                        ffn_plan.config.expert_parallel_size),
                },
                dtype=dtype,
                variant="full",
                rank=physical_rank,
                attributes={
                    "tokens_per_expert": tokens_per_expert,
                    "local_tokens_per_expert": local_tokens_per_expert,
                    "expert_parallel_size": len(expert_group),
                    "expert_rank": expert_rank,
                    "owned_experts": owned_experts,
                    "moe_invocation_id": invocation_id,
                    "expert_placement_strategy": "linear",
                    "routing_provenance": routing_provenance,
                    "launch_policy": launch_policy,
                    **graph_token_domain(
                        logical_block_token_count, block_token_count),
                },
            ))
            for predecessor in previous_rank_frontiers[logical_rank]:
                edges.append(WorkloadEdge(predecessor, node_id))
            moe_node_ids.append(node_id)

        collective = pool.collective_for("all_reduce")
        participants = pool.physical_rank_group("tensor", expert_group[0])
        expert_participants = pool.physical_rank_group(
            "expert", expert_group[0])
        if participants != expert_participants:
            raise RuntimeError(
                "EP output reduction requires identical TP/EP "
                "physical rank groups")
        collective_id = f"{invocation_id}:ep-output-all-reduce"
        nodes.append(WorkloadNode(
            collective_id,
            WorkloadKind.COLLECTIVE,
            pool.name,
            {},
            rank=participants[0],
            attributes={
                "algorithm": collective.algorithm,
                "bytes": (
                    block_token_count
                    * model.hidden_size
                    * model.dtype_bytes
                ),
                "bytes_semantics": "per_rank_input",
                "backend": collective.backend,
                "backend_config_sha256": (
                    collective.backend_config_sha256),
                "direction": "collective",
                "link": collective.link,
                "topology": collective.topology,
                "participants": participants,
                "operation": collective.operation,
                "source_rank": None,
                "destination_rank": None,
                "ring_order": _collective_ring_order(
                    collective.algorithm, participants),
                "reduction": "sum",
                "resource_id": collective.resource_id,
                "moe_invocation_id": invocation_id,
                **graph_token_domain(
                    logical_block_token_count, block_token_count),
            },
        ))
        for node_id in moe_node_ids:
            edges.append(WorkloadEdge(node_id, collective_id, "barrier"))
        previous_rank_frontiers = tuple(
            (collective_id,) for _ in expert_group)
        previous_pool_name = pool.name

    def resolve_moe_routing(
        block_token_count,
        *,
        layer_index,
        invocation_id,
    ):
        routing_pool = pool_for("ffn")
        logical_expert_group = ffn_plan.rank_groups.for_rank("expert", 0)
        physical_expert_group = tuple(
            routing_pool.physical_rank_for_logical_rank(logical_rank)
            for logical_rank in logical_expert_group
        )
        resolution = runtime.expert_routing.resolve(
            step_id=step.step_id,
            layer_index=layer_index,
            moe_invocation_id=invocation_id,
            schedule_step_sha256=step_sha256,
            tokens=block_token_count,
            num_experts=model.num_experts,
            topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256,
            pool_name=routing_pool.name,
            tensor_parallel_size=ffn_tp,
            expert_parallel_size=(
                ffn_plan.config.expert_parallel_size),
            expert_rank_group=physical_expert_group,
            expert_placement_strategy="linear",
            memo=routing_memo,
            moe_layer_ordinal=(
                None if layer_index is None
                else model.moe_layer_ordinal(layer_index)),
        )
        return resolution

    def append_moe(
        prefix,
        block_token_count,
        logical_block_token_count,
        *,
        layer_index,
    ):
        # Router and expert launch tensors retain the captured extent, while
        # routing is resolved only over valid Scheduler rows.  Padding slots
        # have no expert assignment and cannot create MoE work.
        moe_hardware_id = pool_for("ffn").hardware_id
        mask_entries = block_token_count * model.num_experts_per_tok
        # Helper contract §2.2: the MoE routing-mask call points whose count
        # the vLLM source predicate fixes (expert_map is not None).  The
        # per-card emission set comes from helper_launch; every call point
        # resolves its count through the versioned observation seam so a
        # vendor-layer divergence stays a registered entry instead of an ad
        # hoc constant.  The counts resolve once here because the router-run
        # stamp must land on the immutable ROUTER node below, before the
        # per-call-point emission loop reuses them.
        mask_emission_counts = {}
        for call_point in moe_routing_mask_emissions(moe_hardware_id):
            emission_count = helper_emission_count(
                moe_hardware_id, call_point, phase,
                source_count=(
                    1 if ffn_plan.config.enable_expert_parallel else 0),
                model_type=model.model_type,
            )
            if emission_count not in (0, 1):
                raise ValueError(
                    "MoE routing-mask helper emission count must be 0 or 1 "
                    f"per layer; resolved {emission_count}")
            mask_emission_counts[call_point] = emission_count
        router_attributes = {
            "has_bias": False,
            "position": "router",
            "launch_policy": launch_policy,
        }
        if moe_routing_mask_string_complete(moe_hardware_id):
            # A card whose emission set forms the whole 4-kernel mask string
            # additionally carries the string count as the router run
            # contract: the trace-side counter observes one contiguous
            # Index->NotEqual->Cast->Mul string per MoE layer, so the run
            # count is exactly the shared per-layer emission count of the
            # four call points.  Structural membership keeps a partial
            # emitter from ever claiming a run count.
            string_counts = {
                mask_emission_counts[call_point]
                for call_point in ROUTER_MASK_STRING_CALL_POINTS}
            if len(string_counts) != 1:
                raise ValueError(
                    "MoE routing-mask string call points must resolve one "
                    f"shared per-layer count; got {sorted(string_counts)}")
            if string_counts == {1}:
                router_attributes["helper_launch"] = {
                    "class": ROUTER_MASK_RUN_CLASS,
                }
        append(WorkloadKind.ROUTER, f"{prefix}:router", {
            "b": 1,
            "m": block_token_count,
            "n": model.num_experts,
            "k": model.hidden_size,
        }, variant="gemm", attributes=router_attributes,
               operator_group="ffn", activation_token_count=block_token_count,
               token_domain=(logical_block_token_count, block_token_count))
        invocation_id = f"{step.step_id}:{prefix}:moe"
        resolution = resolve_moe_routing(
            logical_block_token_count,
            layer_index=layer_index,
            invocation_id=invocation_id,
        )
        tokens_per_expert = resolution.tokens_per_expert
        routing_provenance = dict(resolution.routing_provenance)
        if ep_rank_local_emission:
            append_expert_parallel_moe(
                prefix,
                block_token_count,
                logical_block_token_count,
                tokens_per_expert,
                invocation_id,
                routing_provenance,
            )
        else:
            append(WorkloadKind.FUSED_MOE, f"{prefix}:moe", {
                "tokens": block_token_count,
                "experts": model.num_experts,
                "topk": model.num_experts_per_tok,
                "hidden": model.hidden_size,
                "intermediate": routed_expert_local_intermediate(
                    model,
                    ffn_plan.config.tensor_parallel_size,
                    ffn_plan.config.expert_parallel_size),
            }, variant="full", attributes={
                "tokens_per_expert": tokens_per_expert,
                "expert_parallel_size": ffn_plan.config.expert_parallel_size,
                "moe_invocation_id": invocation_id,
                "expert_placement_strategy": "linear",
                "routing_provenance": routing_provenance,
                "launch_policy": launch_policy,
            }, operator_group="ffn",
                   activation_token_count=block_token_count,
                   token_domain=(
                       logical_block_token_count, block_token_count))
        # Helper contract §2.2 emission loop: the launches are emitted after
        # the MoE block rather than between router and FUSED_MOE because the
        # EP graph contract requires the FUSED_MOE node's single predecessor
        # to stay the same-rank router and its output to converge on the
        # matching all-reduce; chained emission after that block preserves
        # both.
        for call_point, emission_count in mask_emission_counts.items():
            if emission_count == 0:
                continue
            mask_source = router_mask_source(moe_hardware_id)
            if call_point == "moe_expert_map_index":
                append(WorkloadKind.GATHER, f"{prefix}:expert-map-index", {
                    "elements": mask_entries,
                }, variant="expert_map_index", attributes={
                    "gather_factor": 1,
                    "index_count": mask_entries,
                    "index_bytes": 4,
                    "predicate": "expert_map is not None (EP active)",
                    "scattered": True,
                    "source_symbol": mask_source,
                }, operator_group="ffn",
                       activation_token_count=block_token_count,
                       token_domain=(
                           logical_block_token_count, block_token_count))
            elif call_point == "router_mask_index":
                append(WorkloadKind.GATHER, f"{prefix}:router-mask-index", {
                    "elements": mask_entries,
                }, variant="router_mask_index", attributes={
                    "gather_factor": 1,
                    "index_count": mask_entries,
                    "index_bytes": 4,
                    "predicate": "expert_map is not None (EP active)",
                    "scattered": True,
                    "source_symbol": mask_source,
                }, operator_group="ffn",
                       activation_token_count=block_token_count,
                       token_domain=(
                           logical_block_token_count, block_token_count))
            elif call_point == "router_mask_notequal":
                append(WorkloadKind.ELEMENTWISE,
                       f"{prefix}:router-mask-notequal", {
                           "elements": mask_entries,
                       }, variant="elementwise", attributes={
                           "operation": "notequal_int32_bool",
                           "reads": 1,
                           "writes": 1,
                           "helper_launch": {
                               "class": "class2_router_notequal"},
                           "predicate": (
                               "mask = expert_map[topk_ids] != -1 "
                               "(int32 compare to a bool mask)"),
                           "source_symbol": mask_source,
                       }, operator_group="ffn",
                       activation_token_count=block_token_count,
                       token_domain=(
                           logical_block_token_count, block_token_count))
            elif call_point == "router_mask_cast":
                append(WorkloadKind.ELEMENTWISE,
                       f"{prefix}:router-mask-cast", {
                           "elements": mask_entries,
                       }, variant="elementwise", attributes={
                           "operation": "cast_bool_bf16",
                           "reads": 1,
                           "writes": 1,
                           "helper_launch": {
                               "class": "class2_router_mask_cast"},
                           "predicate": (
                               "topk_weights = topk_weights * mask (the bool "
                               "mask is promoted to BF16 before the mul)"),
                           "source_symbol": mask_source,
                       }, operator_group="ffn",
                       activation_token_count=block_token_count,
                       token_domain=(
                           logical_block_token_count, block_token_count))
            elif call_point == "router_mask_mul":
                append(WorkloadKind.ELEMENTWISE, f"{prefix}:router-mask-mul", {
                    "elements": mask_entries,
                }, variant="elementwise", attributes={
                    "operation": "mul",
                    "reads": 2,
                    "writes": 1,
                    "predicate": "topk_weights = topk_weights * mask",
                    "source_symbol": mask_source,
                }, operator_group="ffn",
                       activation_token_count=block_token_count,
                       token_domain=(
                           logical_block_token_count, block_token_count))
            else:
                raise ValueError(
                    f"unknown MoE routing-mask call point: {call_point!r}")
        if model.num_shared_experts:
            append_dense_mlp(
                f"{prefix}:shared-experts",
                block_token_count,
                logical_block_token_count,
                model.moe_intermediate_size * model.num_shared_experts,
            )
            append(WorkloadKind.ELEMENTWISE, f"{prefix}:shared-add", {
                "elements": block_token_count * model.hidden_size,
            }, variant="add", attributes={
                "reads": 2, "writes": 1, "operation": "add"},
               operator_group="ffn",
                   activation_token_count=block_token_count,
                   token_domain=(logical_block_token_count, block_token_count))

    def append_decoder_layer(
        prefix,
        block_token_count,
        query_lengths,
        kv_lengths,
        *,
        uses_moe,
        attention_phase,
        layer_index,
        input_norm_has_residual,
    ):
        logical_block_token_count = sum(query_lengths)
        if logical_block_token_count <= 0:
            raise ValueError(
                "decoder query lengths must contain logical token work")
        if logical_block_token_count > block_token_count:
            raise ValueError(
                "decoder logical query rows cannot exceed model tensor rows")
        if (logical_block_token_count != block_token_count
                and graph_shape.runtime_mode not in ("FULL", "PIECEWISE")):
            raise ValueError(
                "decoder token shapes differ without graph padding")
        if len(query_lengths) != len(kv_lengths) or not query_lengths:
            raise ValueError(
                "decoder query and KV lengths must be non-empty and aligned")
        append(WorkloadKind.NORM, f"{prefix}:input-norm", {
            "rows": block_token_count,
            "hidden": model.hidden_size,
        }, variant=(
            "add_rmsnorm" if input_norm_has_residual else "rmsnorm"
        ), activation_token_count=block_token_count,
               establishes_residual=True,
               token_domain=(logical_block_token_count, block_token_count))
        m, n, k = infer_gemm_dimensions(
            model, "qkv_proj", block_token_count, tp_size=attention_tp)
        append(WorkloadKind.LINEAR, f"{prefix}:qkv", {
            "b": 1, "m": m, "n": n, "k": k,
        }, variant="gemm", attributes={
            "has_bias": model.attention_bias,
            "position": "qkv_proj",
            "launch_policy": launch_policy,
        }, activation_token_count=block_token_count,
               token_domain=(logical_block_token_count, block_token_count))
        if model.qk_norm:
            append(WorkloadKind.NORM, f"{prefix}:q-norm", {
                "rows": block_token_count * query_heads,
                "hidden": model.head_dim,
            }, variant="rmsnorm", attributes={
                "input_materialization": "grouped_strided_to_contiguous",
                "input_outer_groups": block_token_count,
                "input_rows_per_group": query_heads,
                "input_group_stride_elements": (
                    (query_heads + 2 * kv_heads) * model.head_dim),
                "input_row_stride_elements": model.head_dim,
            }, activation_token_count=block_token_count,
                   token_domain=(
                       logical_block_token_count, block_token_count))
            append(WorkloadKind.NORM, f"{prefix}:k-norm", {
                "rows": block_token_count * kv_heads,
                "hidden": model.head_dim,
            }, variant="rmsnorm", attributes={
                "input_materialization": "grouped_strided_to_contiguous",
                "input_outer_groups": block_token_count,
                "input_rows_per_group": kv_heads,
                "input_group_stride_elements": (
                    (query_heads + 2 * kv_heads) * model.head_dim),
                "input_row_stride_elements": model.head_dim,
            }, activation_token_count=block_token_count,
                   token_domain=(
                       logical_block_token_count, block_token_count))
        rope_program, rope_program_divergence = modelled_rope_program(
            pool_for("attention").hardware_id, model.rope_scaling_type)
        if (rope_program_divergence is not None
                and rope_program != ROPE_PROGRAM_FUSED):
            raise AssertionError(
                "a declared rope lowering gap must fall back to the fused "
                f"custom op, got {rope_program!r}")
        if rope_program == ROPE_PROGRAM_FUSED:
            # §2.1 class1b: beyond the two source-authored q/k Norm input
            # materializations, the 910B torch_npu/aclnn dispatch layer
            # launches one additional V materialization copy per attention
            # layer in prefill (contract §14 phase law).  The vLLM-Ascend
            # python source authors no third copy (source_count 0), so the
            # count resolves through the versioned emission observation and
            # the Ascend QKV-RMSNorm-RoPE fusion lowering materializes it.
            rotary_attributes = (
                {"qkv_v_materialization_count": helper_emission_count(
                    pool_for("attention").hardware_id,
                    "qkv_v_materialization",
                    phase,
                    source_count=0,
                    model_type=model.model_type)}
                if pool_for("attention").hardware_id == "ascend_910b"
                else None)
            append(WorkloadKind.ROTARY, f"{prefix}:rotary", {
                "elements": (
                    block_token_count
                    * (query_heads + kv_heads)
                    * model.head_dim
                ),
                "seq_len": block_token_count,
                "num_query_heads": query_heads,
                "num_kv_heads": kv_heads,
                "head_dim": model.head_dim,
            }, variant="rotary", attributes=rotary_attributes,
                   activation_token_count=block_token_count,
                   token_domain=(
                       logical_block_token_count, block_token_count))
        elif rope_program == ROPE_PROGRAM_NATIVE_COMPOSITE:
            # The pinned runtime does not register this rope class as an
            # out-of-tree custom op, so vLLM runs RotaryEmbedding.forward_native
            # and every framework tensor op below is a separate device kernel.
            composite_rope_calls = native_composite_rope_calls(
                block_token_count, query_heads, kv_heads, model.head_dim,
                rope_composite_semantics(pool_for("attention").hardware_id))
            for rope_index, rope_call in enumerate(composite_rope_calls):
                # The vLLM operator capture hooks the rotary module, so the
                # whole composite is one observed `rotary` record; carry the
                # position so the alignment projection can collapse it back.
                composite_attributes = {
                    "rope_composite_position": rope_index,
                    "rope_composite_tokens": block_token_count,
                    "iteration_tokens": rope_call.tokens,
                    "iteration_rows": rope_call.rows,
                    "iteration_inner": rope_call.inner,
                }
                if rope_call.operation == "gather":
                    append(
                        WorkloadKind.GATHER,
                        f"{prefix}:rotary-{rope_call.name}",
                        {
                            "elements": rope_call.elements,
                            "rows": block_token_count,
                            "hidden": model.head_dim,
                        },
                        variant="rope_cos_sin",
                        attributes={
                            "gather_factor": 1,
                            "index_count": block_token_count,
                            "index_bytes": 8,
                            "scattered": True,
                            "source_symbol": rope_call.source_symbol,
                            **composite_attributes,
                        },
                        activation_token_count=block_token_count,
                        token_domain=(
                            logical_block_token_count, block_token_count))
                elif rope_call.operation in ("copy", "mul", "binary"):
                    # CANN routes aclnnSub through the same two-input BF16
                    # elementwise TBE template as aclnnAdd, so both rotate-half
                    # reductions price as the shared "add" elementwise class;
                    # the node id keeps the source op distinguishable.
                    operation = (
                        "add" if rope_call.operation == "binary"
                        else rope_call.operation)
                    reads = 1 if operation == "copy" else 2
                    layout_attributes = {}
                    if rope_call.operand_layout == "concatenated_last_dim":
                        layout_attributes["input_materialization"] = (
                            "concatenated_last_dim")
                    elif rope_call.operand_layout != "contiguous":
                        layout_attributes["operand_layout"] = (
                            rope_call.operand_layout)
                    append(
                        WorkloadKind.ELEMENTWISE,
                        f"{prefix}:rotary-{rope_call.name}",
                        {"elements": rope_call.elements},
                        variant="elementwise",
                        attributes={
                            "reads": reads,
                            "writes": 1,
                            "operation": operation,
                            "source_symbol": rope_call.source_symbol,
                            **layout_attributes,
                            **composite_attributes,
                        },
                        activation_token_count=block_token_count,
                        token_domain=(
                            logical_block_token_count, block_token_count))
                else:
                    raise AssertionError(
                        "unhandled composite rope operation: "
                        f"{rope_call.operation!r}")
        else:
            raise AssertionError(
                f"unhandled rope execution program: {rope_program!r}")
        append(WorkloadKind.KV_UPDATE, f"{prefix}:kv-update", {
            "elements": (
                logical_block_token_count * kv_heads * model.head_dim),
            "seq_len": logical_block_token_count,
            "num_kv_heads": kv_heads,
            "head_dim": model.head_dim,
        }, variant="paged_kv_cache", attributes=PAGED_KV_CACHE_ATTRIBUTES,
               activation_token_count=block_token_count,
               token_domain=(
                   logical_block_token_count, logical_block_token_count))
        if graph_shape.padding_tokens:
            if tuple(query_lengths) != attention_shape.logical_q_lens:
                raise RuntimeError(
                    "decoder logical query lengths disagree with graph contract")
            if tuple(kv_lengths) != attention_shape.logical_kv_lens:
                raise RuntimeError(
                    "decoder logical KV lengths disagree with graph contract")
            attention_batch = len(attention_shape.model_input_q_lens)
            attention_q_len = max(attention_shape.model_input_q_lens)
            max_context = max(attention_shape.model_input_kv_lens)
            attention_attributes = {
                "model_input_q_lens": attention_shape.model_input_q_lens,
                "model_input_kv_lens": attention_shape.model_input_kv_lens,
            }
        else:
            attention_batch = len(query_lengths)
            attention_q_len = max(query_lengths)
            max_context = max(kv_lengths)
            attention_attributes = {}
        if max_context <= 0:
            raise ValueError("attention requires a positive context length")
        if attention_phase in ("prefill", "mixed") or any(
                length > 1 for length in query_lengths):
            query_length = max(query_lengths)
            attention_variant = "prefill"
        elif attention_phase in ("decode", "mtp"):
            if any(length != 1 for length in query_lengths):
                raise ValueError(
                    "decode-shaped attention requires one query token per "
                    f"request, got {query_lengths!r}")
            query_length = 1
            attention_variant = "decode"
        else:
            raise ValueError(
                f"unknown decoder attention phase: {attention_phase!r}")
        # Ascend FULL adds one dummy request whose query rows occupy the padded
        # tensor but whose KV length is zero.  The current FLASH_ATTN contract
        # requires positive KV lengths, so preserve only source-proven active
        # request work here instead of inventing dummy KV traffic.  The output
        # activation remains in the physical block_token_count domain.
        append(WorkloadKind.FLASH_ATTN, f"{prefix}:attention", {
            "batch": attention_batch,
            "q_len": attention_q_len,
            "kv_len": max_context,
            "head_dim": model.head_dim,
            "num_query_heads": query_heads,
            "num_kv_heads": kv_heads,
        }, variant=attention_variant, attributes={
            "causal": True,
            "kv_layout": attention_kv_layout,
            "kv_dist": runtime.attention_kv_distribution,
            "q_lens": tuple(query_lengths),
            "kv_lens": tuple(kv_lengths),
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
            # vLLM-Ascend's attention wrapper contains two source-level
            # output assignments around the fused FIA call.  The first
            # decoder layer elides the inner assignment because its output
            # buffer is already the framework destination; subsequent
            # layers retain the logical-row inner move and the physical-row
            # outer move.  Keep this as source-authored workload metadata so
            # the 910B lowering can materialize the two TensorMove helpers
            # without deriving a residual from measured duration.  The
            # per-role count resolves through the versioned emission
            # observation, so the measured vendor dispatch-layer phase laws
            # (decode 0 on every case; qwen3_moe prefill 0) suppress the
            # copies instead of leaving them silently emitted.
            "launch_policy": launch_policy,
            "split_policy": "serving",
            "graph_runtime_mode": graph_shape.runtime_mode,
            **({
                "source_output_copy_plan": (
                    ("physical",) if layer_index == 0
                    else ("logical", "physical")
                ),
            } if (
                pool_for("attention").hardware_id == "ascend_910b"
                and helper_emission_count(
                    pool_for("attention").hardware_id,
                    "attention_output_writeback",
                    phase,
                    source_count=1,
                    model_type=model.model_type,
                ) != 0
            ) else {}),
            **attention_attributes,
        }, activation_token_count=block_token_count,
               token_domain=(logical_block_token_count, block_token_count))
        m, n, k = infer_gemm_dimensions(
            model, "o_proj", block_token_count, tp_size=attention_tp)
        append(WorkloadKind.LINEAR, f"{prefix}:o-proj", {
            "b": 1, "m": m, "n": n, "k": k,
        }, variant="gemm", attributes={
            "has_bias": model.attention_bias,
            "position": "o_proj",
            "launch_policy": launch_policy,
        }, activation_token_count=block_token_count,
               token_domain=(logical_block_token_count, block_token_count),
               tp_collective_operation=(
                   "all_reduce" if tp_rank_local_emission else None),
               tp_collective_bytes=(
                   tp_collective_payload_bytes(
                       model, "all_reduce", block_token_count)
                   if tp_rank_local_emission else None))
        append(WorkloadKind.NORM, f"{prefix}:post-attn-norm", {
            "rows": block_token_count,
            "hidden": model.hidden_size,
        }, variant="add_rmsnorm", operator_group="ffn",
               activation_token_count=block_token_count,
               establishes_residual=True,
               token_domain=(logical_block_token_count, block_token_count))
        if uses_moe:
            append_moe(
                prefix,
                block_token_count,
                logical_block_token_count,
                layer_index=layer_index,
            )
        else:
            append_dense_mlp(
                prefix,
                block_token_count,
                logical_block_token_count,
                model.intermediate_size,
            )

    def block_boundary_state():
        return (
            previous,
            previous_rank_frontiers,
            residual_source,
            residual_rank_frontiers,
            previous_pool_name,
            previous_pipeline_stage,
        )

    def record_materialized_block(
        signature,
        block_id,
        node_start,
        edge_start,
        state_before,
    ):
        nonlocal active_block_run
        materialized_nodes = tuple(nodes[node_start:])
        materialized_edges = tuple(edges[edge_start:])
        internal_node_ids = {
            node.node_id for node in materialized_nodes
        }
        if not internal_node_ids:
            raise RuntimeError(
                "materialized decoder block emitted no workload nodes")
        state_after = block_boundary_state()
        boundary_pairs = []
        recurrence_supported = True

        def append_boundary_pair(before, after):
            nonlocal recurrence_supported
            if before is None:
                if after is not None:
                    recurrence_supported = False
                return
            if after is None:
                recurrence_supported = False
                return
            boundary_pairs.append((before, after))

        append_boundary_pair(state_before[0], state_after[0])
        append_boundary_pair(state_before[2], state_after[2])
        for before_frontiers, after_frontiers, label in (
                (state_before[1], state_after[1], "activation"),
                (state_before[3], state_after[3], "residual")):
            if len(before_frontiers) != len(after_frontiers):
                recurrence_supported = False
                continue
            for before_frontier, after_frontier in zip(
                    before_frontiers, after_frontiers, strict=True):
                if len(before_frontier) != len(after_frontier):
                    recurrence_supported = False
                    continue
                for before, after in zip(
                        before_frontier, after_frontier, strict=True):
                    append_boundary_pair(before, after)

        feedback_source_by_external_id = {}
        if recurrence_supported:
            for before, after in boundary_pairs:
                if after not in internal_node_ids:
                    raise RuntimeError(
                        "decoder block recurrence output is outside its template")
                if before in feedback_source_by_external_id:
                    if feedback_source_by_external_id[before] != after:
                        raise RuntimeError(
                            "decoder block recurrence state is ambiguous")
                else:
                    feedback_source_by_external_id[before] = after
        feedback_edges = []
        if recurrence_supported:
            for edge in materialized_edges:
                if (edge.destination not in internal_node_ids
                        or edge.source in internal_node_ids):
                    continue
                if edge.source not in feedback_source_by_external_id:
                    raise RuntimeError(
                        "decoder block external dependency lacks recurrence "
                        f"state: {edge.source!r}")
                feedback_edges.append(WorkloadEdge(
                    feedback_source_by_external_id[edge.source],
                    edge.destination,
                    edge.kind,
                ))
        if recurrence_supported and not feedback_edges:
            raise RuntimeError(
                "decoder block template requires recurrence feedback edges")
        active_block_run = _BlockRunDraft(
            signature=signature,
            block_id=block_id,
            node_ids=tuple(node.node_id for node in materialized_nodes),
            feedback_edges=tuple(feedback_edges),
            state_after=state_after,
            trace_node_ids={
                node.node_id: [node.node_id]
                for node in materialized_nodes
                if node.kind is WorkloadKind.AF_ACTIVATION_TRANSFER
            },
        )
        block_run_drafts.append(active_block_run)

    def append_cached_decoder_layer(
        prefix,
        block_token_count,
        query_lengths,
        kv_lengths,
        *,
        uses_moe,
        attention_phase,
        layer_index,
        input_norm_has_residual,
    ):
        nonlocal previous, previous_rank_frontiers, previous_pool_name
        nonlocal residual_source, residual_rank_frontiers
        nonlocal previous_pipeline_stage, transfer_index, active_block_run
        if block_memo is None:
            append_decoder_layer(
                prefix,
                block_token_count,
                query_lengths,
                kv_lengths,
                uses_moe=uses_moe,
                attention_phase=attention_phase,
                layer_index=layer_index,
                input_norm_has_residual=input_norm_has_residual,
            )
            return
        if runtime.expert_routing.source_kind == "replay_observation":
            raise NotImplementedError(
                "per-layer expert routing replay requires uncompressed "
                "workload construction")
        def make_signature(routing_counts):
            return BlockExecutionSignature(
                model_identity=id(model),
                deployment_identity=id(deployment),
                launch_policy=launch_policy,
                phase=attention_phase,
                token_count=block_token_count,
                query_lengths=tuple(query_lengths),
                kv_lengths=tuple(kv_lengths),
                attention_kv_layout=attention_kv_layout,
                attention_kv_distribution=runtime.attention_kv_distribution,
                graph_runtime_mode=graph_shape.runtime_mode,
                uses_moe=uses_moe,
                routing_policy_sha256=(
                    runtime.expert_routing.policy_descriptor_sha256),
                routing_counts=routing_counts,
                input_norm_has_residual=input_norm_has_residual,
                pipeline_stage=active_pipeline_stage,
                previous_pipeline_stage=previous_pipeline_stage,
                previous_pool_name=previous_pool_name,
                previous_rank_frontier_widths=tuple(
                    len(frontier) for frontier in previous_rank_frontiers),
                has_previous=previous is not None,
                has_residual_source=residual_source is not None,
                residual_rank_frontier_widths=tuple(
                    len(frontier) for frontier in residual_rank_frontiers),
            )

        if uses_moe and active_block_run is not None:
            reused_signature = make_signature(
                active_block_run.signature.routing_counts)
        else:
            reused_signature = None
        if (reused_signature is not None
                and reused_signature == active_block_run.signature):
            signature = reused_signature
        else:
            if uses_moe:
                routing_counts = resolve_moe_routing(
                    sum(query_lengths),
                    layer_index=layer_index,
                    invocation_id=f"{step.step_id}:{prefix}:moe",
                ).tokens_per_expert
            else:
                routing_counts = None
            signature = make_signature(routing_counts)
        template = block_memo.template_for(signature)
        if (active_block_run is not None
                and active_block_run.signature == signature
                and active_block_run.feedback_edges
                and active_block_run.state_after == block_boundary_state()):
            if template is None:
                raise RuntimeError(
                    "active decoder block repeat lost its construction template")
            for template_node_id, logical_node_ids in (
                    active_block_run.trace_node_ids.items()):
                marker = f"{step.step_id}:af-transfer-"
                if not template_node_id.startswith(marker):
                    raise RuntimeError(
                        "decoder block trace node is not an activation transfer")
                suffix = template_node_id[len(marker):]
                index_text, remainder = suffix.split(":", 1)
                logical_index = (
                    int(index_text)
                    + active_block_run.repeat_count * template.transfer_count
                )
                logical_node_ids.append(
                    f"{marker}{logical_index}:{remainder}")
            active_block_run.repeat_count += 1
            transfer_index += template.transfer_count
            return
        if template is None:
            node_start = len(nodes)
            edge_start = len(edges)
            state_before = block_boundary_state()
            previous_before = previous
            previous_rank_frontiers_before = previous_rank_frontiers
            residual_source_before = residual_source
            residual_rank_frontiers_before = residual_rank_frontiers
            transfer_index_before = transfer_index
            append_decoder_layer(
                prefix,
                block_token_count,
                query_lengths,
                kv_lengths,
                uses_moe=uses_moe,
                attention_phase=attention_phase,
                layer_index=layer_index,
                input_norm_has_residual=input_norm_has_residual,
            )
            template = _BlockTemplate(
                step_id=step.step_id,
                prefix=prefix,
                nodes=tuple(nodes[node_start:]),
                edges=tuple(edges[edge_start:]),
                previous_before=previous_before,
                previous_after=previous,
                previous_rank_frontiers_before=(
                    previous_rank_frontiers_before),
                previous_rank_frontiers_after=previous_rank_frontiers,
                residual_source_before=residual_source_before,
                residual_source_after=residual_source,
                residual_rank_frontiers_before=(
                    residual_rank_frontiers_before),
                residual_rank_frontiers_after=residual_rank_frontiers,
                previous_pool_name_after=previous_pool_name,
                previous_pipeline_stage_after=previous_pipeline_stage,
                transfer_count=transfer_index - transfer_index_before,
            )
            block_memo.store(signature, template)
            record_materialized_block(
                signature,
                f"{step.step_id}:{prefix}",
                node_start,
                edge_start,
                state_before,
            )
            return

        node_start = len(nodes)
        edge_start = len(edges)
        state_before = block_boundary_state()
        external_id_map = {}

        def bind_external(template_id, current_id):
            if template_id is None:
                if current_id is not None:
                    raise RuntimeError(
                        "block template external state shape changed")
                return
            if current_id is None:
                raise RuntimeError(
                    "block template external state lost a required value")
            if (template_id in external_id_map
                    and external_id_map[template_id] != current_id):
                raise RuntimeError(
                    "block template external state mapping is inconsistent")
            external_id_map[template_id] = current_id

        bind_external(template.previous_before, previous)
        bind_external(template.residual_source_before, residual_source)
        if (len(template.previous_rank_frontiers_before)
                != len(previous_rank_frontiers)):
            raise RuntimeError(
                "block template rank frontier count changed")
        for old_frontier, current_frontier in zip(
                template.previous_rank_frontiers_before,
                previous_rank_frontiers,
                strict=True):
            if len(old_frontier) != len(current_frontier):
                raise RuntimeError(
                    "block template rank frontier width changed")
            for old_id, current_id in zip(
                    old_frontier, current_frontier, strict=True):
                bind_external(old_id, current_id)
        if (len(template.residual_rank_frontiers_before)
                != len(residual_rank_frontiers)):
            raise RuntimeError(
                "block template residual frontier count changed")
        for old_frontier, current_frontier in zip(
                template.residual_rank_frontiers_before,
                residual_rank_frontiers,
                strict=True):
            if len(old_frontier) != len(current_frontier):
                raise RuntimeError(
                    "block template residual frontier width changed")
            for old_id, current_id in zip(
                    old_frontier, current_frontier, strict=True):
                bind_external(old_id, current_id)

        old_transfer_indices = []
        for node in template.nodes:
            marker = f"{template.step_id}:af-transfer-"
            if not node.node_id.startswith(marker):
                continue
            suffix = node.node_id[len(marker):]
            index_text = suffix.split(":", 1)[0]
            index = int(index_text)
            if index not in old_transfer_indices:
                old_transfer_indices.append(index)
        if len(old_transfer_indices) != template.transfer_count:
            raise RuntimeError(
                "block template activation transfer count is inconsistent")
        transfer_index_map = {
            old_index: transfer_index + offset
            for offset, old_index in enumerate(old_transfer_indices)
        }

        def rewrite_internal_id(old_id):
            if old_id in external_id_map:
                return external_id_map[old_id]
            old_step_prefix = f"{template.step_id}:"
            if not old_id.startswith(old_step_prefix):
                raise RuntimeError(
                    f"block template ID is outside its step: {old_id!r}")
            rewritten = f"{step.step_id}:{old_id[len(old_step_prefix):]}"
            rewritten = rewritten.replace(
                f":{template.prefix}", f":{prefix}", 1)
            for old_index, new_index in transfer_index_map.items():
                rewritten = rewritten.replace(
                    f":af-transfer-{old_index}:",
                    f":af-transfer-{new_index}:",
                    1,
                )
            return rewritten

        internal_id_map = {
            node.node_id: rewrite_internal_id(node.node_id)
            for node in template.nodes
        }

        for template_node in template.nodes:
            attributes = dict(template_node.attributes)
            if "moe_invocation_id" in attributes:
                invocation_id = f"{step.step_id}:{prefix}:moe"
                attributes["moe_invocation_id"] = invocation_id
                if template_node.kind is WorkloadKind.FUSED_MOE:
                    resolution = resolve_moe_routing(
                        sum(query_lengths),
                        layer_index=layer_index,
                        invocation_id=invocation_id,
                    )
                    attributes["tokens_per_expert"] = (
                        resolution.tokens_per_expert)
                    attributes["routing_provenance"] = dict(
                        resolution.routing_provenance)
                    if "owned_experts" in attributes:
                        owned_experts = set(attributes["owned_experts"])
                        attributes["local_tokens_per_expert"] = tuple(
                            count if expert_id in owned_experts else 0
                            for expert_id, count in enumerate(
                                resolution.tokens_per_expert)
                        )
            nodes.append(WorkloadNode(
                internal_id_map[template_node.node_id],
                template_node.kind,
                template_node.pool_name,
                template_node.shape,
                dtype=template_node.dtype,
                variant=template_node.variant,
                rank=template_node.rank,
                attributes=attributes,
            ))
        for template_edge in template.edges:
            edges.append(WorkloadEdge(
                internal_id_map[template_edge.source]
                if template_edge.source in internal_id_map
                else rewrite_internal_id(template_edge.source),
                internal_id_map[template_edge.destination]
                if template_edge.destination in internal_id_map
                else rewrite_internal_id(template_edge.destination),
                template_edge.kind,
            ))

        def rewrite_state_id(value):
            if value is None:
                return None
            if value in internal_id_map:
                return internal_id_map[value]
            return rewrite_internal_id(value)

        previous = rewrite_state_id(template.previous_after)
        previous_rank_frontiers = tuple(
            tuple(rewrite_state_id(value) for value in frontier)
            for frontier in template.previous_rank_frontiers_after
        )
        residual_source = rewrite_state_id(template.residual_source_after)
        residual_rank_frontiers = tuple(
            tuple(rewrite_state_id(value) for value in frontier)
            for frontier in template.residual_rank_frontiers_after
        )
        previous_pool_name = template.previous_pool_name_after
        previous_pipeline_stage = template.previous_pipeline_stage_after
        transfer_index += template.transfer_count
        record_materialized_block(
            signature,
            f"{step.step_id}:{prefix}",
            node_start,
            edge_start,
            state_before,
        )

    if (sum(target_query_lengths) != graph_shape.attention_active_tokens
            or sum(target_query_lengths) != graph_shape.kv_write_tokens):
        raise RuntimeError(
            "graph shape domains disagree with logical Scheduler tokens")
    for layer in range(model.num_hidden_layers):
        active_pipeline_stage = plan.layer_to_pipeline_stage[layer]
        append_cached_decoder_layer(
            f"layer-{layer}",
            token_count,
            target_query_lengths,
            target_kv_lengths,
            uses_moe=model.layer_uses_moe(layer),
            attention_phase=phase,
            layer_index=layer,
            input_norm_has_residual=layer > 0,
        )
    active_pipeline_stage = pipeline_size - 1
    append(WorkloadKind.NORM, "final-norm", {
        "rows": token_count,
        "hidden": model.hidden_size,
    }, variant="add_rmsnorm",
           token_domain=(logical_token_count, token_count))
    # Pinned vLLM selects one target logit row per scheduled request plus one
    # row per verified draft token. Partial-prefill rows are still sampled on
    # device and discarded only during bookkeeping.
    target_logit_rows = (
        len(step.requests)
        + sum(request.mtp_verified_tokens for request in step.requests)
    )
    # Helper contract §2.3 (model output boundary): vLLM gathers exactly one
    # hidden-state row per target logit row before the lm_head projection
    # (sample_hidden_states = hidden_states[logits_indices]). The count is
    # the source predicate itself — one launch whenever a step computes
    # logits — with the row set fixed by the same pinned selection rule.
    append(WorkloadKind.GATHER, "logits-boundary-gather", {
        "elements": target_logit_rows * model.hidden_size,
        "rows": target_logit_rows,
        "hidden": model.hidden_size,
    }, variant="logits_indices", attributes={
        "gather_factor": 1,
        "index_count": target_logit_rows,
        "index_bytes": 8,
        "predicate": "per-step target logit row selection",
        "scattered": True,
        "source_symbol": boundary_logits_gather_source(
            pool_for("attention").hardware_id),
    })
    m, n, k = infer_gemm_dimensions(
        model, "lm_head", target_logit_rows, tp_size=attention_tp)
    append(WorkloadKind.LM_HEAD, "lm-head", {
        "b": 1, "m": m, "n": n, "k": k,
    }, variant="gemv" if m == 1 else "gemm", attributes={
        "has_bias": False,
        "position": "lm_head",
        "launch_policy": launch_policy,
        "tensor_parallel_size": attention_tp,
        "expert_parallel_size": attention_plan.config.expert_parallel_size,
    }, tp_collective_operation=(
       "all_gather" if tp_rank_local_emission else None),
       tp_collective_bytes=(
           tp_collective_payload_bytes(
               model,
               "all_gather",
               m,
               local_output_width=n,
           ) if tp_rank_local_emission else None))

    mtp_requests = tuple(
        request for request in step.requests
        if request.mtp_proposer_tokens
    )
    if mtp_requests:
        if model.mtp_architecture != "glm4_moe_mtp":
            raise NotImplementedError(
                "MTP schedule requires a supported model MTP architecture")
        if model.num_mtp_layers != 1:
            raise NotImplementedError(
                "pinned vLLM GLM MTP workload currently requires one MTP layer")
        proposal_depths = {
            request.mtp_proposer_tokens for request in mtp_requests}
        if len(proposal_depths) != 1:
            raise ValueError(
                "one vLLM MTP step must use one global proposal depth")
        proposal_depth = next(iter(proposal_depths))
        proposer_modes = {
            request.mtp_proposer_mode for request in mtp_requests}
        if proposer_modes != {"padded"}:
            raise NotImplementedError(
                "GLM MTP workload requires pinned padded proposer batches")
        initial_query_lengths = tuple(
            request.scheduled_tokens
            for request in mtp_requests)
        if any(length <= 0 for length in initial_query_lengths):
            raise ValueError("MTP proposer requires a positive accepted query")
        initial_kv_lengths = tuple(
            request.computed_tokens
            + request.scheduled_tokens
            for request in mtp_requests)
        for proposer_pass in range(proposal_depth):
            if proposer_pass == 0:
                proposer_query_lengths = initial_query_lengths
                proposer_kv_lengths = initial_kv_lengths
            else:
                proposer_query_lengths = (1,) * len(mtp_requests)
                proposer_kv_lengths = tuple(
                    length + proposer_pass
                    for length in initial_kv_lengths)
            proposer_token_count = sum(proposer_query_lengths)
            prefix = f"mtp-pass-{proposer_pass}"
            append(WorkloadKind.GATHER, f"{prefix}:token-embedding", {
                "elements": proposer_token_count * model.hidden_size,
                "rows": proposer_token_count,
                "hidden": model.hidden_size,
            }, variant="embedding", attributes={
                "gather_factor": 1,
                "index_count": proposer_token_count,
                "index_bytes": 8,
                "scattered": True,
            }, activation_token_count=proposer_token_count)
            append(WorkloadKind.ELEMENTWISE, f"{prefix}:position-zero-mask", {
                "elements": proposer_token_count * model.hidden_size,
            }, variant="masked-fill-zero", attributes={
                "reads": 1,
                "writes": 1,
                "operation": "copy",
            }, activation_token_count=proposer_token_count)
            append(WorkloadKind.NORM, f"{prefix}:embedding-norm", {
                "rows": proposer_token_count,
                "hidden": model.hidden_size,
            }, variant="rmsnorm",
                   activation_token_count=proposer_token_count)
            append(WorkloadKind.NORM, f"{prefix}:hidden-norm", {
                "rows": proposer_token_count,
                "hidden": model.hidden_size,
            }, variant="rmsnorm",
                   activation_token_count=proposer_token_count)
            append(WorkloadKind.LINEAR, f"{prefix}:merge-proj", {
                "b": 1,
                "m": proposer_token_count,
                "n": model.hidden_size,
                "k": 2 * model.hidden_size,
            }, variant="gemm", attributes={
                "has_bias": False,
                "position": "mtp_merge_proj",
                "launch_policy": launch_policy,
            }, activation_token_count=proposer_token_count)
            append_cached_decoder_layer(
                f"{prefix}:decoder",
                proposer_token_count,
                proposer_query_lengths,
                proposer_kv_lengths,
                uses_moe=model.mtp_layer_uses_moe(0),
                attention_phase="mtp",
                layer_index=None,
                input_norm_has_residual=False,
            )
            append(WorkloadKind.ELEMENTWISE, f"{prefix}:residual-add", {
                "elements": proposer_token_count * model.hidden_size,
            }, variant="add", attributes={
                "reads": 2, "writes": 1, "operation": "add"},
                   activation_token_count=proposer_token_count)
            append(WorkloadKind.NORM, f"{prefix}:head-norm", {
                "rows": len(mtp_requests),
                "hidden": model.hidden_size,
            }, variant="rmsnorm",
                   activation_token_count=len(mtp_requests))
            m, n, k = infer_gemm_dimensions(
                model,
                "lm_head",
                len(mtp_requests),
                tp_size=attention_tp,
            )
            append(WorkloadKind.LM_HEAD, f"{prefix}:head", {
                "b": 1, "m": m, "n": n, "k": k,
            }, variant="gemv" if m == 1 else "gemm", attributes={
                "has_bias": False,
                "position": "lm_head",
                "launch_policy": launch_policy,
                "tensor_parallel_size": attention_tp,
                "expert_parallel_size": (
                    attention_plan.config.expert_parallel_size),
            }, activation_token_count=len(mtp_requests))

    for ready_event, attributes in pending_evict_attributes:
        append(
            WorkloadKind.KV_OFFLOAD,
            f"kv-offload:{ready_event}",
            {},
            attributes=attributes,
            pool_override=scheduler_pool,
        )

    block_repeats = tuple(
        WorkloadBlockRepeat(
            draft.block_id,
            draft.node_ids,
            draft.repeat_count,
            draft.feedback_edges,
            tuple(
                (template_id, tuple(logical_ids))
                for template_id, logical_ids in draft.trace_node_ids.items()
            ),
        )
        for draft in block_run_drafts
        if draft.repeat_count > 1
    )
    return WorkloadGraph(nodes, edges, block_repeats)
