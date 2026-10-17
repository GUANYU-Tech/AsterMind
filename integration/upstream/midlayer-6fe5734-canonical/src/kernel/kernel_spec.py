"""Production contracts for single-kernel prediction.

These objects deliberately exclude profiler truth such as runtime kernel names,
measured duration, and counters. Those belong to ``evaluation.kernel``.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from .launch_config import (
    CannGemmLaunchProgram,
    CannGemmTilePipelineProgram,
    FiaCrossDomainLaunchProgram,
    GpuCtaRasterProgram,
    LaunchConfig,
)
from .model_contract import (
    PredictionDiagnostics,
    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER,
    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
    SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
    SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
    SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER,
    SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER,
    SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
    SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER,
    SOURCE_CANN_BF16_ELEMENTWISE_DAG_PARAMETER,
    SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
    SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER,
    SOURCE_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_PROGRAM_PARAMETER,
    SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER,
    SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER,
    SOURCE_METAX_ROPE_VLLM_0_9_1_ABI11_PROGRAM_PARAMETER,
    SOURCE_METAX_ROPE_VLLM_0_24_0_ABI13_PROGRAM_PARAMETER,
    SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER,
    SOURCE_QK_ROPE_LOOP_PARAMETER,
    SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER,
    SOURCE_PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_PARAMETER,
    SOURCE_PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_PARAMETER,
    SOURCE_METAX_BF16_GROUPED_STRIDED_COPY_PARAMETER,
    SOURCE_SCALAR_KV_LOOP_PARAMETER,
    SOURCE_VECTORIZED_KV_COPY_PARAMETER,
    SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER,
)


class _CachedVectorParameterSchemas(list):
    """Avoid recompiling an already validated closed vector schema family."""

    def __init__(self, schemas):
        super().__init__()
        self._schemas = schemas

    def __contains__(self, value):
        return frozenset(value) in self._schemas

    def __iter__(self):
        return iter(())

    def append(self, value):
        del value

    def extend(self, values):
        del values


class _PrunedVectorParameterSchemas:
    """Build only closed schemas that can still equal the input parameters.

    Vector launch validation expresses a closed schema as additive branches.
    A branch that has already introduced a parameter absent from the launch
    being validated cannot become a match later: no branch removes a
    parameter.  Retaining only subsets of the submitted parameter names is
    therefore equivalent to building the full family for membership while
    avoiding its combinatorial intermediate states.
    """

    def __init__(self, parameter_names):
        self._parameter_names = frozenset(parameter_names)
        self._schemas = {frozenset()}

    def __contains__(self, value):
        return frozenset(value) in self._schemas

    def __iter__(self):
        return iter(self._schemas)

    def append(self, value):
        schema = frozenset(value)
        if schema <= self._parameter_names:
            self._schemas.add(schema)

    def extend(self, values):
        for value in values:
            self.append(value)


_VECTOR_PARAMETER_SCHEMA_CACHE = {}


class KernelKind(str, Enum):
    GEMM = "GEMM"
    FLASH_ATTN = "FLASH_ATTN"
    EAGER_ATTN = "EAGER_ATTN"
    SPARSE_ATTN_INDEXER = "SPARSE_ATTN_INDEXER"
    SPARSE_ATTN = "SPARSE_ATTN"
    KV_CACHE = "KV_CACHE"
    FUSED_MOE = "FUSED_MOE"
    UNFUSED_MOE = "UNFUSED_MOE"
    VECTOR = "VECTOR"
    COMMUNICATION = "COMMUNICATION"
    MEMCPY = "MEMCPY"
    KV_OFFLOAD = "KV_OFFLOAD"

    @classmethod
    def parse(cls, value: "KernelKind | str") -> "KernelKind":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str) or not value.strip():
            raise TypeError("kernel kind must be a non-empty string or KernelKind")
        normalized = value.strip().upper().replace("-", "_")
        aliases = {
            "FLASHATTN": cls.FLASH_ATTN,
            "EAGERATTN": cls.EAGER_ATTN,
            "SPARSEATTNINDEXER": cls.SPARSE_ATTN_INDEXER,
            "SPARSEATTN": cls.SPARSE_ATTN,
            "KVCACHEUPDATE": cls.KV_CACHE,
            "KV_CACHE_UPDATE": cls.KV_CACHE,
            "FUSEDMOE": cls.FUSED_MOE,
            "UNFUSEDMOE": cls.UNFUSED_MOE,
            "NORM": cls.VECTOR,
            "COLLECTIVE": cls.COMMUNICATION,
            "CUSTOM_COMM": cls.COMMUNICATION,
            "ROTARY": cls.VECTOR,
            "ACTIVATION": cls.VECTOR,
            "REDUCE": cls.VECTOR,
            "GATHER": cls.VECTOR,
            "ELEMENTWISE": cls.VECTOR,
            "MOE_SORT": cls.VECTOR,
            "MOE_HISTOGRAM": cls.VECTOR,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(f"unknown kernel kind: {value!r}") from exc


def canonical_physical_algorithm(
        hardware_id: str, kind: KernelKind, variant: str,
        algorithm: str) -> str:
    """Return the source-neutral physical implementation identity.

    Dataset observations and Serving launch selection can have different
    provenance labels for the same physical launch.  Forward must not branch
    on that provenance, so every KernelSpec crosses this normalization boundary
    before reaching a predictor.
    """

    if hardware_id in ("nvidia_a800", "nvidia_h200") and kind is KernelKind.GEMM:
        if variant in ("gemm", "batched_gemm", "moe_router") and (
                algorithm == "observed_gemm:gemm"
                or algorithm.startswith("cublaslt_injected_gemm")):
            return "nvidia_cublaslt_gemm"
        if variant == "gemv" and algorithm in {
                "observed_gemm:gemv", "cublaslt_injected_gemv"}:
            return "nvidia_cublas_gemv"
    if hardware_id == "metax_c500" and kind is KernelKind.GEMM:
        if variant in ("gemm", "batched_gemm", "moe_router") and algorithm in {
                "mcblas_bf16_gemm", "mcblas_injected_gemm:gemm",
                "observed_mcblas_gemm:gemm"}:
            return "metax_mcblas_gemm"
        if variant == "gemv" and algorithm in {
                "observed_gemm:gemv", "mcblas_gemv", "mcblas_gemv_wave"}:
            return "metax_mcblas_gemv"
    if hardware_id == "ascend_910b" and kind is KernelKind.GEMM:
        if algorithm in {
                "cann_matmul", "cann_injected_matmul",
                "observed_gemm:gemm"}:
            return "ascend_cann_matmul"
        if algorithm == "observed_gemm:gemv":
            return "ascend_cann_gemv"

    aliases = {
        "metax_flash_attn_lut": "metax_flash_attention_generic",
    }
    if algorithm in aliases:
        return aliases[algorithm]
    if algorithm.startswith("observed_metax_"):
        return algorithm[len("observed_"):]
    if (hardware_id == "metax_c500"
            and kind is KernelKind.FLASH_ATTN
            and algorithm.startswith("observed_flash_attention:")):
        return "metax_flash_attention_generic"
    if algorithm.startswith("observed_flash_attention:"):
        return f"flash_attention:{algorithm.split(':', 1)[1]}"
    return algorithm


MOE_FILL_ROLES = (
    "sorted_token_ids",
    "expert_ids",
    "num_tokens_post_padded",
)

MOE_CAST_ROLES = (
    "topk_ids",
    "expert_tokens",
)

PAGED_KV_CACHE_ATTRIBUTES = MappingProxyType({
    "cache_layout": "paged",
    "cache_head_layout": "nhd",
    "kv_cache_dtype": "auto",
    "cache_storage_matches_input": True,
    "slot_mapping_all_valid": True,
    "vector_copy_base_alignment_bytes": 16,
})


@dataclass(frozen=True)
class PhysicalTransferConfig:
    """Complete physical inputs for communication and data movement."""

    algorithm: str
    bytes: int
    direction: str
    link: str
    topology: Optional[str] = None
    tier: Optional[str] = None
    participants: Tuple[int, ...] = ()
    operation: Optional[str] = None
    bytes_semantics: Optional[str] = None
    source_rank: Optional[int] = None
    destination_rank: Optional[int] = None
    ring_order: Tuple[int, ...] = ()
    reduction: Optional[str] = None
    backend: Optional[str] = None
    resource_id: Optional[str] = None
    backend_config_sha256: Optional[str] = None
    # Framework collective implementation when it differs from the default
    # (vLLM) custom all-reduce: SGLang's CustomAllReduceV2, or SGLang's
    # flashinfer all-reduce fused with the following residual RMSNorm.
    implementation: Optional[str] = None

    def __post_init__(self) -> None:
        if self.implementation not in (
                None, "sglang_custom_all_reduce_v2",
                "sglang_flashinfer_allreduce_residual_rmsnorm"):
            raise ValueError(
                f"unknown transfer implementation: {self.implementation!r}")
        if self.implementation is not None and (
                self.operation != "all_reduce" or self.link != "nvlink"):
            raise ValueError(
                "the SGLang custom all-reduce implementation applies only to "
                "NVLink all_reduce")
        for name in ("algorithm", "direction", "link"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"transfer {name} must be non-empty")
            object.__setattr__(self, name, value.strip().lower())
        for name in (
                "topology", "tier", "operation", "bytes_semantics",
                "reduction", "backend"):
            value = getattr(self, name)
            if value is not None:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"transfer {name} cannot be empty")
                object.__setattr__(self, name, value.strip().lower())
        if self.resource_id is not None:
            if (not isinstance(self.resource_id, str)
                    or not self.resource_id.strip()):
                raise ValueError("transfer resource_id cannot be empty")
            object.__setattr__(
                self, "resource_id", self.resource_id.strip())
        if self.backend_config_sha256 is not None:
            if (not isinstance(self.backend_config_sha256, str)
                    or not self.backend_config_sha256.strip()):
                raise ValueError(
                    "transfer backend_config_sha256 cannot be empty")
            object.__setattr__(
                self,
                "backend_config_sha256",
                self.backend_config_sha256.strip(),
            )
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int):
            raise TypeError("transfer bytes must be an integer")
        if self.bytes < 0:
            raise ValueError("transfer bytes cannot be negative")
        object.__setattr__(self, "bytes", self.bytes)
        if not isinstance(self.participants, (list, tuple)):
            raise TypeError("transfer participants must be a list or tuple")
        participants = tuple(self.participants)
        if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in participants):
            raise TypeError("transfer participant ranks must be integers")
        if any(rank < 0 for rank in participants):
            raise ValueError("transfer participant ranks cannot be negative")
        if len(participants) != len(set(participants)):
            raise ValueError("transfer participant ranks must be unique")
        object.__setattr__(self, "participants", participants)
        for name in ("source_rank", "destination_rank"):
            rank = getattr(self, name)
            if rank is not None:
                if isinstance(rank, bool) or not isinstance(rank, int):
                    raise TypeError(f"transfer {name} must be an integer or null")
                if rank < 0:
                    raise ValueError(f"transfer {name} cannot be negative")
        if not isinstance(self.ring_order, (list, tuple)):
            raise TypeError("transfer ring_order must be a list or tuple")
        ring_order = tuple(self.ring_order)
        if any(isinstance(rank, bool) or not isinstance(rank, int)
               for rank in ring_order):
            raise TypeError("transfer ring_order ranks must be integers")
        if any(rank < 0 for rank in ring_order):
            raise ValueError("transfer ring_order ranks cannot be negative")
        if len(ring_order) != len(set(ring_order)):
            raise ValueError("transfer ring_order ranks must be unique")
        object.__setattr__(self, "ring_order", ring_order)
        if self.backend_config_sha256 is not None:
            digest = self.backend_config_sha256
            if (len(digest) != 64
                    or any(character not in "0123456789abcdef"
                           for character in digest)):
                raise ValueError(
                    "transfer backend_config_sha256 must be a lowercase SHA256")


def _immutable_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise TypeError("expected a mapping")
    return MappingProxyType(dict(values))


# Storage width of every dtype the compute-shape contract accepts.  This is a
# physical constant of the dtype, so it is declared once and read by every
# consumer instead of being restated as a local map.
DTYPE_BYTES = {
    "fp8": 1,
    "int8": 1,
    "uint8": 1,
    "fp16": 2,
    "bf16": 2,
    "float16": 2,
    "float32": 4,
    "fp32": 4,
    "int32": 4,
    "int64": 8,
    "bool": 1,
}


def _normalize_compute_shape(kind: KernelKind, variant: str,
                             dtype: str,
                             shape: Mapping[str, int],
                             attributes: Mapping[str, Any]
                             ) -> Mapping[str, int]:
    normalized = dict(shape)
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"unsupported compute dtype: {dtype!r}")
    expected_dtype_bytes = DTYPE_BYTES[dtype]
    if ("dtype_bytes" in normalized
            and normalized["dtype_bytes"] != expected_dtype_bytes):
        raise ValueError(
            f"shape dtype_bytes={normalized['dtype_bytes']} disagrees with "
            f"dtype {dtype!r}")
    # dtype_bytes is derived from the required dtype contract.  A supplied
    # copy is only a consistency assertion; omission never selects behavior.
    normalized["dtype_bytes"] = expected_dtype_bytes
    if kind is KernelKind.KV_CACHE and variant == "paged_kv_cache":
        has_elements = "elements" in normalized
        has_element_count = "element_count" in normalized
        if has_elements and has_element_count:
            if normalized["elements"] != normalized["element_count"]:
                raise ValueError(
                    "KV_CACHE/paged_kv_cache elements and "
                    "element_count disagree")
        elif has_elements:
            normalized["element_count"] = normalized["elements"]
        elif has_element_count:
            normalized["elements"] = normalized["element_count"]
        structured_fields = ("seq_len", "num_kv_heads", "head_dim")
        present = {name for name in structured_fields if name in normalized}
        if present and present != set(structured_fields):
            missing = set(structured_fields) - present
            raise ValueError(
                "KV_CACHE/paged_kv_cache has a partial structured "
                f"shape; missing {sorted(missing)}")
        if present:
            structured_count = math.prod(
                normalized[name] for name in structured_fields)
            if ("element_count" in normalized
                    and normalized["element_count"] != structured_count):
                raise ValueError(
                    "KV_CACHE/paged_kv_cache element_count "
                    f"{normalized['element_count']} disagrees with "
                    f"structured shape count {structured_count}")
            normalized["element_count"] = structured_count
            normalized["elements"] = structured_count
    elif kind is KernelKind.VECTOR:
        if variant == "flash_mla_metadata":
            fields = {
                "batch", "kv_block_count", "scheduler_partitions",
                "scheduler_metadata_ints", "split_metadata_ints",
                "scheduler_segment_count",
                "scheduler_worker_run_count",
            }
            unknown = set(normalized) - fields - {"dtype_bytes"}
            if unknown:
                raise ValueError(
                    "VECTOR/flash_mla_metadata has unknown shape fields: "
                    f"{sorted(unknown)}")
            if fields.issubset(normalized):
                if (normalized["scheduler_metadata_ints"]
                        != 8 * normalized["scheduler_partitions"]):
                    raise ValueError(
                        "VECTOR/flash_mla_metadata scheduler_metadata_ints "
                        "must equal 8 * scheduler_partitions")
                if (normalized["split_metadata_ints"]
                        != normalized["batch"] + 1):
                    raise ValueError(
                        "VECTOR/flash_mla_metadata split_metadata_ints must "
                        "equal batch + 1")
                if normalized["kv_block_count"] < normalized["batch"]:
                    raise ValueError(
                        "VECTOR/flash_mla_metadata kv_block_count cannot be "
                        "smaller than batch")
                if (normalized["scheduler_segment_count"]
                        < normalized["batch"]):
                    raise ValueError(
                        "VECTOR/flash_mla_metadata scheduler_segment_count "
                        "cannot be smaller than batch")
                if (normalized["scheduler_worker_run_count"] > normalized[
                        "scheduler_segment_count"]
                        or normalized["scheduler_worker_run_count"]
                        > normalized["scheduler_partitions"]):
                    raise ValueError(
                        "VECTOR/flash_mla_metadata worker runs cannot exceed "
                        "the segment or partition count")
        if variant in ("rmsnorm", "add_rmsnorm"):
            has_rows = "rows" in normalized
            has_tokens = "tok" in normalized
            if has_rows and has_tokens:
                if normalized["rows"] != normalized["tok"]:
                    raise ValueError("VECTOR normalization rows and tok disagree")
            elif has_rows:
                normalized["tok"] = normalized["rows"]
            elif has_tokens:
                normalized["rows"] = normalized["tok"]
        if variant == "qkv_rmsnorm_rope":
            fields = {
                "seq_len", "num_query_heads", "num_kv_heads", "head_dim",
                "rotary_dim",
            }
            present = fields & set(normalized)
            if present and present != fields:
                raise ValueError(
                    "VECTOR/qkv_rmsnorm_rope has a partial structured shape; "
                    f"missing {sorted(fields - present)}")
            if present:
                if normalized["rotary_dim"] > normalized["head_dim"]:
                    raise ValueError(
                        "VECTOR/qkv_rmsnorm_rope rotary_dim cannot exceed "
                        "head_dim")
                if normalized["rotary_dim"] % 2:
                    raise ValueError(
                        "VECTOR/qkv_rmsnorm_rope rotary_dim must be even")
                total_heads = (
                    normalized["num_query_heads"]
                    + 2 * normalized["num_kv_heads"])
                structured_count = (
                    normalized["seq_len"] * total_heads
                    * normalized["head_dim"])
                if ("element_count" in normalized
                        and normalized["element_count"] != structured_count):
                    raise ValueError(
                        "VECTOR/qkv_rmsnorm_rope element_count disagrees "
                        "with the structured QKV extent")
                normalized["element_count"] = structured_count
                normalized["elements"] = structured_count
        has_elements = "elements" in normalized
        has_element_count = "element_count" in normalized
        if has_elements and has_element_count:
            if normalized["elements"] != normalized["element_count"]:
                raise ValueError("VECTOR elements and element_count disagree")
        elif has_elements:
            normalized["element_count"] = normalized["elements"]
        elif has_element_count:
            normalized["elements"] = normalized["element_count"]
        structured_fields = {
            "rotary": (
                "seq_len", "num_query_heads", "num_kv_heads", "head_dim"),
            "rope": (
                "seq_len", "num_query_heads", "num_kv_heads", "head_dim",
                "rotary_dim"),
            "activation": ("tok", "intermediate"),
            "silu": ("tok", "intermediate"),
        }
        if variant in structured_fields:
            fields = structured_fields[variant]
            present = {name for name in fields if name in normalized}
            if present and present != set(fields):
                missing = set(fields) - present
                raise ValueError(
                    f"VECTOR/{variant} has a partial structured shape; "
                    f"missing {sorted(missing)}")
            if present:
                if variant == "rotary":
                    structured_count = (
                        normalized["seq_len"]
                        * (normalized["num_query_heads"]
                           + normalized["num_kv_heads"])
                        * normalized["head_dim"]
                    )
                elif variant == "rope":
                    rotary_dim = normalized["rotary_dim"]
                    if rotary_dim > normalized["head_dim"]:
                        raise ValueError(
                            "VECTOR/rope rotary_dim cannot exceed head_dim")
                    if rotary_dim % 2:
                        raise ValueError(
                            "VECTOR/rope rotary_dim must be even")
                    key_heads = (
                        normalized["num_kv_heads"]
                        if attributes["key_present"] else 0)
                    structured_count = (
                        normalized["seq_len"]
                        * (normalized["num_query_heads"] + key_heads)
                        * rotary_dim
                    )
                elif variant in ("activation", "silu"):
                    structured_count = math.prod(
                        normalized[name] for name in fields)
                else:
                    raise ValueError(
                        f"unhandled structured VECTOR variant: {variant!r}")
                if ("element_count" in normalized
                        and normalized["element_count"] != structured_count):
                    raise ValueError(
                        f"VECTOR/{variant} element_count "
                        f"{normalized['element_count']} disagrees with structured "
                        f"shape count {structured_count}")
                normalized["element_count"] = structured_count
                normalized["elements"] = structured_count
    elif kind in (
            KernelKind.GEMM, KernelKind.FLASH_ATTN, KernelKind.EAGER_ATTN,
            KernelKind.SPARSE_ATTN_INDEXER, KernelKind.SPARSE_ATTN,
            KernelKind.KV_CACHE, KernelKind.FUSED_MOE,
            KernelKind.UNFUSED_MOE):
        pass
    else:
        raise ValueError(f"unhandled compute shape kind: {kind.value}")
    return normalized


def _require_shape_fields(kind: KernelKind, variant: str,
                          shape: Mapping[str, int]) -> None:
    variants = {
        KernelKind.GEMM: {"gemm", "gemv", "batched_gemm", "moe_router"},
        KernelKind.FLASH_ATTN: {
            "prefill", "decode", "combine", "split_combine",
            "mla_prefill", "mla_decode", "mla_absorbed_prefill"},
        KernelKind.EAGER_ATTN: {"decomposed"},
        KernelKind.SPARSE_ATTN_INDEXER: {
            "dsa_indexer_prefill", "dsa_indexer_decode",
            "dsa_indexer_mixed", "dsa_index_reuse"},
        KernelKind.SPARSE_ATTN: {
            "dsa_sparse_prefill", "dsa_sparse_decode"},
        KernelKind.KV_CACHE: {
            "paged_kv_cache", "mla_kv_cache", "dsa_fp8_index_cache"},
        KernelKind.FUSED_MOE: {"w1", "w2", "routed_experts"},
        KernelKind.UNFUSED_MOE: {"decomposed"},
        KernelKind.VECTOR: {
            "rmsnorm", "add_rmsnorm", "layernorm", "mla_rope", "elementwise",
            "elementwise_pytorch_vec4", "rotary", "rope",
            "activation", "silu", "reduce", "gather", "softmax",
                "qkv_rmsnorm_rope",
                "gemm_reduce", "gemm_elementwise", "flash_mla_metadata",
                "fp8_group_quant",
                "moe_swiglu",
                "moe_init_routing", "moe_init_routing_v2",
                "moe_init_routing_v3", "moe_finalize_routing",
                "moe_compute_expert_tokens", "moe_transpose", "moe_cast",
                "moe_range", "moe_topk", "moe_softmax", "moe_gating",
                "moe_gating_topk", "moe_abs", "moe_token_unpermute",
            "moe_act_and_mul", "moe_silu", "moe_mul", "moe_sum",
            "moe_align",
            "moe_align_opt_128", "moe_align_small_batch",
            "moe_count_and_sort", "moe_reduce", "moe_fill",
            "moe_routing_weight_norm", "moe_routing_weight_scale",
            "moe_routed_id_extent", "moe_expert_map_lookup",
            "moe_router_logit_cast", "moe_routed_id_widen",
            "moe_enable_if",
        },
    }
    if variant not in variants[kind]:
        raise ValueError(f"unknown {kind.value} variant: {variant!r}")

    required = {
        KernelKind.GEMM: {"b", "m", "n", "k"},
        KernelKind.FLASH_ATTN: {
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"},
        KernelKind.EAGER_ATTN: {
            "batch", "q_len", "kv_len", "head_dim",
            "num_query_heads", "num_kv_heads"},
        KernelKind.SPARSE_ATTN: {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "topk"},
        KernelKind.FUSED_MOE: {
            "tokens", "experts", "topk", "hidden", "intermediate"},
        KernelKind.UNFUSED_MOE: {
            "tokens", "experts", "topk", "hidden", "intermediate"},
    }
    if kind is KernelKind.FLASH_ATTN and variant in (
            "mla_prefill", "mla_decode", "mla_absorbed_prefill"):
        missing = {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "kv_lora_rank",
        } - set(shape)
    elif (kind is KernelKind.SPARSE_ATTN_INDEXER
          and variant == "dsa_index_reuse"):
        missing = {"element_count"} - set(shape)
    elif kind is KernelKind.SPARSE_ATTN_INDEXER:
        missing = {
            "batch", "q_len", "kv_len", "num_query_heads", "head_dim",
            "topk",
        } - set(shape)
    elif (kind is KernelKind.KV_CACHE
          and variant == "paged_kv_cache"):
        missing = {
            "element_count", "seq_len", "num_kv_heads", "head_dim",
        } - set(shape)
    elif kind is KernelKind.KV_CACHE and variant == "mla_kv_cache":
        missing = {"element_count", "seq_len", "compressed_dim"} - set(shape)
    elif (kind is KernelKind.KV_CACHE
          and variant == "dsa_fp8_index_cache"):
        missing = {
            "element_count", "seq_len", "key_dim",
            "scale_bytes_per_token",
        } - set(shape)
    elif kind in required:
        missing = required[kind] - set(shape)
    elif kind is KernelKind.VECTOR and variant in (
            "rmsnorm", "add_rmsnorm", "layernorm"):
        missing = {"hidden", "rows"} - set(shape)
    elif kind is KernelKind.VECTOR and variant == "mla_rope":
        missing = {
            "element_count", "seq_len", "num_query_heads",
            "num_key_heads", "head_dim",
        } - set(shape)
    elif kind is KernelKind.VECTOR and variant == "rope":
        missing = {
            "seq_len", "num_query_heads", "num_kv_heads", "head_dim",
            "rotary_dim", "element_count",
        } - set(shape)
    elif kind is KernelKind.VECTOR and variant == "qkv_rmsnorm_rope":
        missing = {
            "seq_len", "num_query_heads", "num_kv_heads", "head_dim",
            "rotary_dim", "element_count",
        } - set(shape)
    elif variant in (
            "elementwise", "elementwise_pytorch_vec4",
            "rotary", "activation", "silu",
            "reduce", "gather", "softmax",
            "gemm_reduce", "gemm_elementwise", "fp8_group_quant"):
        missing = {"element_count"} - set(shape)
    elif variant == "flash_mla_metadata":
        missing = {
            "batch", "kv_block_count", "scheduler_partitions",
            "scheduler_metadata_ints", "split_metadata_ints",
            "scheduler_segment_count",
            "scheduler_worker_run_count",
        } - set(shape)
    elif variant.startswith("moe_"):
        missing = {
            "tokens", "experts", "topk", "hidden", "intermediate"
        } - set(shape)
    else:
        raise ValueError(
            f"unhandled {kind.value} shape contract for {variant!r}")
    if missing:
        raise ValueError(
            f"{kind.value}/{variant} missing shape fields: {sorted(missing)}")
    exact_shapes = {
        (KernelKind.GEMM, "batched_gemm"):
            {"b", "m", "n", "k", "dtype_bytes"},
        (KernelKind.GEMM, "moe_router"):
            {"b", "m", "n", "k", "dtype_bytes"},
        (KernelKind.FLASH_ATTN, "mla_prefill"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "kv_lora_rank", "dtype_bytes",
        },
        (KernelKind.FLASH_ATTN, "mla_decode"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "kv_lora_rank", "dtype_bytes",
        },
        (KernelKind.FLASH_ATTN, "mla_absorbed_prefill"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "kv_lora_rank", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN_INDEXER, "dsa_indexer_prefill"): {
            "batch", "q_len", "kv_len", "num_query_heads", "head_dim",
            "topk", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN_INDEXER, "dsa_indexer_decode"): {
            "batch", "q_len", "kv_len", "num_query_heads", "head_dim",
            "topk", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN_INDEXER, "dsa_indexer_mixed"): {
            "batch", "q_len", "kv_len", "num_query_heads", "head_dim",
            "topk", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN_INDEXER, "dsa_index_reuse"):
            {"element_count", "dtype_bytes"},
        (KernelKind.SPARSE_ATTN, "dsa_sparse_prefill"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "topk", "dtype_bytes",
        },
        (KernelKind.SPARSE_ATTN, "dsa_sparse_decode"): {
            "batch", "q_len", "kv_len", "num_query_heads",
            "qk_head_dim", "v_head_dim", "topk", "dtype_bytes",
        },
        (KernelKind.KV_CACHE, "mla_kv_cache"): {
            "element_count", "seq_len", "compressed_dim", "dtype_bytes",
        },
        (KernelKind.KV_CACHE, "paged_kv_cache"): {
            "element_count", "elements", "seq_len", "num_kv_heads",
            "head_dim", "dtype_bytes",
        },
        (KernelKind.KV_CACHE, "dsa_fp8_index_cache"): {
            "element_count", "seq_len", "key_dim", "scale_bytes_per_token",
            "dtype_bytes",
        },
        (KernelKind.FUSED_MOE, "routed_experts"): {
            "tokens", "experts", "topk", "hidden", "intermediate",
            "dtype_bytes",
        },
        (KernelKind.VECTOR, "layernorm"):
            {"rows", "hidden", "dtype_bytes"},
        (KernelKind.VECTOR, "mla_rope"): {
            "element_count", "elements", "seq_len", "num_query_heads",
            "num_key_heads", "head_dim", "dtype_bytes",
        },
        (KernelKind.VECTOR, "qkv_rmsnorm_rope"): {
            "element_count", "elements", "seq_len", "num_query_heads",
            "num_kv_heads", "head_dim", "rotary_dim", "dtype_bytes",
        },
    }
    key = (kind, variant)
    if key in exact_shapes:
        unknown = set(shape) - exact_shapes[key]
        if unknown:
            raise ValueError(
                f"{kind.value}/{variant} unknown shape fields: "
                f"{sorted(unknown)}")


def _normalize_request_lengths(
    owner: str,
    normalized: dict[str, Any],
    shape: Mapping[str, int],
) -> None:
    model_input_names = {"model_input_q_lens", "model_input_kv_lens"}
    present_model_input_names = model_input_names & set(normalized)
    if present_model_input_names:
        if (present_model_input_names != model_input_names
                or "graph_runtime_mode" not in normalized):
            raise ValueError(
                f"{owner} graph attention domains must be complete: "
                f"present={sorted(present_model_input_names | ({'graph_runtime_mode'} & set(normalized)))}")
        graph_runtime_mode = normalized["graph_runtime_mode"]
        if graph_runtime_mode not in ("NONE", "FULL", "PIECEWISE"):
            raise ValueError(
                f"{owner} graph_runtime_mode is invalid: "
                f"{graph_runtime_mode!r}")
        for name in ("q_lens", "kv_lens"):
            raw_lengths = normalized[name]
            if not isinstance(raw_lengths, (list, tuple)):
                raise TypeError(f"{owner} {name} must be a list or tuple")
            lengths = tuple(raw_lengths)
            if not lengths:
                raise ValueError(f"{owner} {name} cannot be empty")
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in lengths):
                raise ValueError(
                    f"{owner} {name} must contain positive integers")
            maximum_field = "q_len" if name == "q_lens" else "kv_len"
            if max(lengths) > shape[maximum_field]:
                raise ValueError(
                    f"{owner} {name} exceeds model-input "
                    f"{maximum_field}")
            normalized[name] = lengths
        if len(normalized["q_lens"]) != len(normalized["kv_lens"]):
            raise ValueError(f"{owner} logical request lengths must align")
        if any(request_query > request_kv for request_query, request_kv in zip(
                normalized["q_lens"], normalized["kv_lens"])):
            raise ValueError(
                f"{owner} logical request requires q_lens <= kv_lens")
        for name, maximum_field in (
                ("model_input_q_lens", "q_len"),
                ("model_input_kv_lens", "kv_len")):
            raw_lengths = normalized[name]
            if not isinstance(raw_lengths, (list, tuple)):
                raise TypeError(f"{owner} {name} must be a list or tuple")
            lengths = tuple(raw_lengths)
            if len(lengths) != shape["batch"]:
                raise ValueError(
                    f"{owner} {name} must contain one value per model-input "
                    f"request: expected={shape['batch']}, actual={len(lengths)}")
            if any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in lengths):
                raise ValueError(
                    f"{owner} {name} must contain positive integers")
            if max(lengths) != shape[maximum_field]:
                raise ValueError(
                    f"{owner} {name} maximum {max(lengths)} disagrees with "
                    f"shape {maximum_field}={shape[maximum_field]}")
            normalized[name] = lengths
        if any(request_query > request_kv for request_query, request_kv in zip(
                normalized["model_input_q_lens"],
                normalized["model_input_kv_lens"])):
            raise ValueError(
                f"{owner} model-input request requires q_lens <= kv_lens")
        if sum(normalized["q_lens"]) > sum(
                normalized["model_input_q_lens"]):
            raise ValueError(
                f"{owner} logical attention rows exceed model-input rows")
        if graph_runtime_mode == "NONE" and (
                normalized["q_lens"] != normalized["model_input_q_lens"]
                or normalized["kv_lens"]
                != normalized["model_input_kv_lens"]):
            raise ValueError(
                f"{owner} eager execution cannot carry graph padding")
        return
    for name, maximum_field in (
            ("q_lens", "q_len"), ("kv_lens", "kv_len")):
        raw_lengths = normalized[name]
        if not isinstance(raw_lengths, (list, tuple)):
            raise TypeError(f"{owner} {name} must be a list or tuple")
        lengths = tuple(raw_lengths)
        if len(lengths) != shape["batch"]:
            raise ValueError(
                f"{owner} {name} must contain one value per request: "
                f"expected={shape['batch']}, actual={len(lengths)}")
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in lengths):
            raise TypeError(f"{owner} {name} must contain integers")
        if any(value <= 0 for value in lengths):
            raise ValueError(f"{owner} {name} must contain positive values")
        expected_maximum = shape[maximum_field]
        if max(lengths) != expected_maximum:
            raise ValueError(
                f"{owner} {name} maximum {max(lengths)} disagrees with "
                f"shape {maximum_field}={expected_maximum}")
        normalized[name] = lengths
    if any(request_query > request_kv for request_query, request_kv in zip(
            normalized["q_lens"], normalized["kv_lens"])):
        raise ValueError(
            f"{owner} each request requires q_lens <= kv_lens: "
            f"q_lens={normalized['q_lens']}, "
            f"kv_lens={normalized['kv_lens']}")


def _normalize_compute_attributes(kind: KernelKind, variant: str,
                                  attributes: Mapping[str, Any],
                                  shape: Mapping[str, int]):
    normalized = dict(attributes)
    optional = set()
    graph_token_domain_fields = {
        "graph_logical_tokens", "graph_model_input_tokens",
        "graph_runtime_mode", "model_input_shape",
    }
    if (kind is KernelKind.FLASH_ATTN
            and variant in (
                "mla_prefill", "mla_decode", "mla_absorbed_prefill")):
        # The two deployment CUDA-graph facts are part of the MLA contract:
        # vLLM's MLA metadata builders read them (FlashAttnMLA split bound).
        required = {
            "causal", "kv_layout", "kv_dist", "q_lens", "kv_lens",
            "full_cuda_graphs", "max_cudagraph_capture_size",
        }
        # A graph-replayed MLA decode declares its step token domain; the
        # padded requests (query 0, seq_len 0) carry no attention work.
        optional = {
            "graph_logical_tokens", "graph_model_input_tokens",
            "graph_runtime_mode", "attention_backend",
        }
    elif kind is KernelKind.FLASH_ATTN:
        required = {
            "causal", "kv_layout", "kv_dist", "fa_ksteps",
            "fa_eff_mfrac", "q_lens", "kv_lens",
        }
        optional = {
            "model_input_q_lens", "model_input_kv_lens",
            "graph_runtime_mode",
        }
    elif (kind is KernelKind.FUSED_MOE
          and variant == "routed_experts"):
        required = {
            "tokens_per_expert", "local_tokens_per_expert",
            "expert_parallel_size", "expert_rank", "owned_experts",
            "expert_weight_tile_rows", "routed_scaling_factor",
            "routed_weight_dtype", "weight_group_size",
        }
    elif kind is KernelKind.FUSED_MOE:
        required = {
            "tokens_per_expert",
            "local_tokens_per_expert",
            "expert_parallel_size",
            "expert_rank",
            "owned_experts",
            "expert_weight_tile_rows",
        }
        optional = {
            "routed_scaling_factor", "routed_weight_dtype",
            "weight_group_size",
            # SGLang shared-experts fusion: the shared expert runs inside the
            # routed kernel as expert E with every token routed to it.
            "fused_shared_experts",
        }
    elif kind is KernelKind.UNFUSED_MOE:
        required = {"tokens_per_expert", "expert_weight_tile_rows"}
    elif (kind is KernelKind.GEMM and variant == "batched_gemm"):
        required = {
            "batch_layout", "a_batch_stride_elements",
            "b_batch_stride_elements", "c_batch_stride_elements",
            "weight_dtype", "weight_group_size",
        }
    elif kind is KernelKind.GEMM and variant == "moe_router":
        required = {
            "score_function", "topk_method", "expert_group_count",
            "topk_group_count", "normalize_topk_probability",
            "weight_dtype", "weight_group_size",
        }
    elif kind is KernelKind.SPARSE_ATTN_INDEXER:
        if variant == "dsa_index_reuse":
            required = {
                "reuse_semantics", "topk_capacity", "topk_index_dtype",
            }
        elif variant in (
                "dsa_indexer_prefill", "dsa_indexer_decode",
                "dsa_indexer_mixed"):
            required = {
                "causal", "score_accumulator_dtype", "score_function",
                "cache_layout", "cache_quantization", "q_lens", "kv_lens",
                "request_phases", "topk_capacity", "topk_index_dtype",
                "topk_sentinel",
            }
        else:
            raise AssertionError(
                f"unhandled SPARSE_ATTN_INDEXER variant: {variant!r}")
    elif kind is KernelKind.SPARSE_ATTN:
        required = {
            "causal", "indexer_type", "kv_layout", "kv_dist", "q_lens",
            "kv_lens", "selected_kv_lens", "topk_capacity",
            "topk_index_dtype",
        }
    elif kind is KernelKind.KV_CACHE:
        if variant == "paged_kv_cache":
            required = {
                "cache_layout", "cache_head_layout", "kv_cache_dtype",
                "cache_storage_matches_input", "slot_mapping_all_valid",
                "vector_copy_base_alignment_bytes",
            }
        elif variant == "mla_kv_cache":
            required = {"cache_layout", "cache_write_semantics"}
        elif variant == "dsa_fp8_index_cache":
            required = {
                "cache_layout", "cache_write_semantics", "quantization",
            }
        else:
            raise AssertionError(
                f"unhandled KV cache update variant: {variant!r}")
    elif kind is KernelKind.VECTOR and variant in (
            "elementwise", "elementwise_pytorch_vec4"):
        required = {"reads", "writes", "operation"}
        optional = {"helper_launch"}
        materialization_fields = {
            "input_materialization",
            "input_outer_groups",
            "input_rows_per_group",
            "input_group_stride_elements",
            "input_row_stride_elements",
        }
        present_materialization_fields = (
            materialization_fields & set(normalized))
        if present_materialization_fields:
            if ("operation" not in normalized
                    or normalized["operation"] != "copy"):
                raise ValueError(
                    "VECTOR grouped input materialization requires copy")
            required |= materialization_fields
    elif kind is KernelKind.VECTOR and variant == "layernorm":
        required = {"affine", "epsilon"}
    elif kind is KernelKind.VECTOR and variant == "mla_rope":
        required = {
            "rope_layout", "rope_type", "rope_theta", "rope_factor",
            "rope_original_max_position_embeddings", "rope_beta_fast",
            "rope_beta_slow", "rope_mscale", "rope_mscale_all_dim",
        }
    elif kind is KernelKind.VECTOR and variant == "rope":
        required = {"is_neox", "key_present"}
    elif kind is KernelKind.VECTOR and variant == "qkv_rmsnorm_rope":
        required = {"bias", "is_neox"}
    elif kind is KernelKind.VECTOR and variant == "moe_fill":
        required = {"fill_role"}
    elif kind is KernelKind.VECTOR and variant == "moe_cast":
        required = {"cast_role"}
    elif kind is KernelKind.VECTOR and variant == "moe_swiglu":
        required = {"routed_rows"}
    elif (kind is KernelKind.VECTOR
          and variant == "moe_compute_expert_tokens"):
        required = {"tokens_per_expert"}
    elif kind is KernelKind.VECTOR and variant in (
            "moe_silu", "moe_mul", "moe_sum", "moe_reduce"):
        required = {"work_fraction"}
    elif kind is KernelKind.VECTOR and variant == "reduce":
        required = {"reduction_factor"}
    elif kind is KernelKind.VECTOR and variant == "gather":
        required = {
            "gather_factor", "index_count", "index_bytes", "scattered"}
    elif kind is KernelKind.VECTOR and variant == "gemm_reduce":
        required = {"split_k"}
    elif kind is KernelKind.VECTOR and variant == "fp8_group_quant":
        required = {"group_size"}
    elif kind is KernelKind.VECTOR and variant == "gemm_elementwise":
        required = set()
    elif kind in (KernelKind.GEMM, KernelKind.EAGER_ATTN):
        required = set()
        optional = {"helper_launch"}
    elif kind is KernelKind.VECTOR and variant in {
            "rmsnorm", "add_rmsnorm", "rotary", "activation",
            "silu", "softmax", "flash_mla_metadata",
            "moe_init_routing", "moe_init_routing_v2",
            "moe_init_routing_v3",
            "moe_finalize_routing",
            "moe_transpose", "moe_cast", "moe_range", "moe_topk",
            "moe_softmax", "moe_gating", "moe_gating_topk", "moe_abs",
            "moe_token_unpermute", "moe_act_and_mul",
            "moe_align", "moe_align_opt_128", "moe_align_small_batch",
            "moe_count_and_sort", "moe_routing_weight_norm", "moe_routing_weight_scale",
            "moe_routed_id_extent", "moe_expert_map_lookup",
            "moe_router_logit_cast", "moe_routed_id_widen", "moe_enable_if",
    }:
        required = set()
    else:
        raise ValueError(
            f"unhandled {kind.value}/{variant} attribute contract")
    if kind is not KernelKind.FLASH_ATTN:
        present_graph_domain_fields = graph_token_domain_fields & set(
            normalized)
        if present_graph_domain_fields:
            if present_graph_domain_fields != graph_token_domain_fields:
                raise ValueError(
                    f"{kind.value}/{variant} graph token domains must be "
                    f"complete: present={sorted(present_graph_domain_fields)}")
            logical_tokens = normalized["graph_logical_tokens"]
            model_input_tokens = normalized["graph_model_input_tokens"]
            runtime_mode = normalized["graph_runtime_mode"]
            for name, value in (
                    ("graph_logical_tokens", logical_tokens),
                    ("graph_model_input_tokens", model_input_tokens)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"{kind.value}/{variant} {name} must be an integer")
                if value <= 0:
                    raise ValueError(
                        f"{kind.value}/{variant} {name} must be positive")
            if model_input_tokens < logical_tokens:
                raise ValueError(
                    f"{kind.value}/{variant} model-input tokens cannot "
                    "shrink logical work")
            if runtime_mode not in ("NONE", "FULL", "PIECEWISE"):
                raise ValueError(
                    f"{kind.value}/{variant} graph_runtime_mode is invalid")
            if runtime_mode == "NONE" and logical_tokens != model_input_tokens:
                raise ValueError(
                    f"{kind.value}/{variant} eager execution cannot carry "
                    "graph padding")
            model_input_shape = normalized["model_input_shape"]
            if not isinstance(model_input_shape, Mapping):
                raise TypeError(
                    f"{kind.value}/{variant} model_input_shape must be a mapping")
            # ``elements`` is a normalized alias for VECTOR element_count and
            # ``tok`` is synthesized from ``rows`` only for normalization
            # variants.  model-input shape records the pre-normalization
            # launch ABI, so it retains a source ``tok`` for activation/Silu
            # but does not need the normalization-only alias.
            derived_shape_fields = {"dtype_bytes", "elements"}
            if (kind is KernelKind.VECTOR
                    and variant in ("rmsnorm", "add_rmsnorm")):
                derived_shape_fields.add("tok")
            expected_shape_fields = set(shape) - derived_shape_fields
            if set(model_input_shape) != expected_shape_fields:
                raise ValueError(
                    f"{kind.value}/{variant} model_input_shape fields must "
                    "match the effective shape")
            normalized_model_input_shape = {}
            for name, value in model_input_shape.items():
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"{kind.value}/{variant} model_input_shape values "
                        "must be integers")
                if value <= 0:
                    raise ValueError(
                        f"{kind.value}/{variant} model_input_shape values "
                        "must be positive")
                if value < shape[name]:
                    raise ValueError(
                        f"{kind.value}/{variant} model_input_shape cannot "
                        "shrink effective dimensions")
                normalized_model_input_shape[name] = value
            normalized["model_input_shape"] = _immutable_mapping(
                normalized_model_input_shape)
            optional |= graph_token_domain_fields
    missing = required - set(normalized)
    if missing:
        raise ValueError(
            f"{kind.value}/{variant} missing attributes: {sorted(missing)}")
    unknown = set(normalized) - required - optional
    if unknown:
        raise ValueError(
            f"unknown {kind.value}/{variant} attributes: {sorted(unknown)}")
    if "helper_launch" in normalized:
        # Helper/glue emission contract attribute: a workload-node may carry
        # exactly one emission class (docs/serving_helper_launch_contract.md
        # §2).  The value is a mapping with one non-empty "class" string; the
        # harness and lowering read it structurally, never by kernel name.
        helper = normalized["helper_launch"]
        if (not isinstance(helper, Mapping)
                or set(helper) != {"class"}
                or not isinstance(helper["class"], str)
                or not helper["class"].strip()):
            raise ValueError(
                f"{kind.value}/{variant} helper_launch must be a mapping "
                "with exactly one non-empty 'class' string")
    if (kind is KernelKind.VECTOR
            and variant in ("elementwise", "elementwise_pytorch_vec4")
            and "input_materialization" in normalized):
        if normalized["input_materialization"] != (
                "grouped_strided_to_contiguous"):
            raise ValueError(
                "VECTOR copy input_materialization must be "
                "grouped_strided_to_contiguous")
        dimensions = {
            name: normalized[name]
            for name in (
                "input_outer_groups", "input_rows_per_group",
                "input_group_stride_elements", "input_row_stride_elements",
            )
        }
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in dimensions.values()):
            raise TypeError(
                "VECTOR grouped input materialization dimensions must be "
                "integers")
        if any(value <= 0 for value in dimensions.values()):
            raise ValueError(
                "VECTOR grouped input materialization dimensions must be "
                "positive")
        copied_elements = (
            dimensions["input_outer_groups"]
            * dimensions["input_rows_per_group"]
            * dimensions["input_row_stride_elements"])
        if copied_elements != shape["element_count"]:
            raise ValueError(
                "VECTOR grouped input materialization dimensions disagree "
                "with element_count")
        if dimensions["input_group_stride_elements"] <= (
                dimensions["input_rows_per_group"]
                * dimensions["input_row_stride_elements"]):
            raise ValueError(
                "VECTOR grouped input materialization requires an "
                "inter-group gap")
    if kind is KernelKind.VECTOR and variant == "moe_swiglu":
        routed_rows = normalized["routed_rows"]
        if isinstance(routed_rows, bool) or not isinstance(routed_rows, int):
            raise TypeError("VECTOR/moe_swiglu routed_rows must be an integer")
        if routed_rows <= 0:
            raise ValueError("VECTOR/moe_swiglu routed_rows must be positive")
        if routed_rows > shape["tokens"] * shape["topk"]:
            raise ValueError(
                "VECTOR/moe_swiglu routed_rows cannot exceed tokens * topk")
    if (kind is KernelKind.VECTOR
            and variant == "moe_compute_expert_tokens"):
        raw_counts = normalized["tokens_per_expert"]
        if (not isinstance(raw_counts, (list, tuple))
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in raw_counts)):
            raise TypeError(
                "VECTOR/moe_compute_expert_tokens tokens_per_expert must "
                "contain integers")
        counts = tuple(raw_counts)
        if (len(counts) != shape["experts"]
                or any(value < 0 for value in counts)):
            raise ValueError(
                "VECTOR/moe_compute_expert_tokens tokens_per_expert must "
                "contain one non-negative count per expert")
        expected = shape["tokens"] * shape["topk"]
        if sum(counts) != expected:
            raise ValueError(
                "VECTOR/moe_compute_expert_tokens tokens_per_expert sum "
                f"{sum(counts)} != tokens*topk {expected}")
        normalized["tokens_per_expert"] = counts
    if kind is KernelKind.FLASH_ATTN:
        if not isinstance(normalized["causal"], bool):
            raise ValueError("FLASH_ATTN causal must be boolean")
        if variant in ("prefill", "decode", "combine", "split_combine"):
            if (isinstance(normalized["fa_eff_mfrac"], bool)
                    or not isinstance(
                        normalized["fa_eff_mfrac"], (int, float))):
                raise TypeError("FLASH_ATTN fa_eff_mfrac must be numeric")
            fraction = float(normalized["fa_eff_mfrac"])
            if not math.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError("FLASH_ATTN fa_eff_mfrac must be in (0, 1]")
            normalized["fa_eff_mfrac"] = fraction
            if normalized["fa_ksteps"] is not None:
                if (isinstance(normalized["fa_ksteps"], bool)
                        or not isinstance(normalized["fa_ksteps"], int)):
                    raise TypeError("FLASH_ATTN fa_ksteps must be an integer")
                steps = normalized["fa_ksteps"]
                if steps <= 0:
                    raise ValueError("FLASH_ATTN fa_ksteps must be positive")
                normalized["fa_ksteps"] = steps
        elif variant in ("mla_prefill", "mla_decode", "mla_absorbed_prefill"):
            if ("attention_backend" in normalized
                    and normalized["attention_backend"] != "flashinfer"):
                raise ValueError(
                    "MLA attention_backend must be 'flashinfer' when given")
            if normalized["kv_layout"] != "mla":
                raise ValueError("MLA attention requires kv_layout='mla'")
            if normalized["kv_dist"] != "uniform":
                raise ValueError("MLA attention requires kv_dist='uniform'")
            if not isinstance(normalized["full_cuda_graphs"], bool):
                raise TypeError("MLA full_cuda_graphs must be boolean")
            capture = normalized["max_cudagraph_capture_size"]
            if (isinstance(capture, bool) or not isinstance(capture, int)
                    or capture < 0):
                raise ValueError(
                    "MLA max_cudagraph_capture_size must be a non-negative "
                    "integer (0 when CUDA graphs are disabled)")
            if normalized["full_cuda_graphs"] and capture == 0:
                raise ValueError(
                    "MLA full CUDA graphs require a positive capture size")
            graph_fields = {
                "graph_logical_tokens", "graph_model_input_tokens",
                "graph_runtime_mode"}
            present = graph_fields & set(normalized)
            if present:
                if present != graph_fields:
                    raise ValueError(
                        "MLA graph token domain must be complete: "
                        f"present={sorted(present)}")
                if variant != "mla_decode":
                    raise ValueError(
                        "only MLA decode carries a graph token domain")
                logical = normalized["graph_logical_tokens"]
                physical = normalized["graph_model_input_tokens"]
                if (any(isinstance(value, bool) or not isinstance(value, int)
                        or value <= 0 for value in (logical, physical))
                        or physical < logical):
                    raise ValueError(
                        "MLA graph token domain must be positive and "
                        "non-shrinking")
                if normalized["graph_runtime_mode"] != "FULL":
                    raise ValueError(
                        "MLA decode padding is a FULL CUDA-graph domain")
                if logical != sum(normalized["q_lens"]):
                    raise ValueError(
                        "MLA graph logical tokens disagree with q_lens")
        else:
            raise AssertionError(
                f"unhandled FLASH_ATTN variant: {variant!r}")
        _normalize_request_lengths("FLASH_ATTN", normalized, shape)
    elif kind in (KernelKind.FUSED_MOE, KernelKind.UNFUSED_MOE):
        raw_counts = normalized["tokens_per_expert"]
        if (not isinstance(raw_counts, (list, tuple))
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in raw_counts)):
            raise TypeError("tokens_per_expert must contain integers")
        counts = tuple(raw_counts)
        if len(counts) != shape["experts"] or any(value < 0 for value in counts):
            raise ValueError(
                "tokens_per_expert must contain one non-negative count per "
                "expert")
        expected = shape["tokens"] * shape["topk"]
        if sum(counts) != expected:
            raise ValueError(
                f"tokens_per_expert sum {sum(counts)} != tokens*topk "
                f"{expected}")
        normalized["tokens_per_expert"] = counts
        if kind is KernelKind.FUSED_MOE:
            ep_size = normalized["expert_parallel_size"]
            expert_rank = normalized["expert_rank"]
            if (isinstance(ep_size, bool)
                    or not isinstance(ep_size, int)):
                raise TypeError("expert_parallel_size must be an integer")
            if ep_size <= 0 or ep_size > shape["experts"]:
                raise ValueError(
                    "expert_parallel_size must be within [1, experts]")
            if shape["experts"] % ep_size:
                raise ValueError(
                    "expert_parallel_size must divide experts; uneven "
                    "expert placement is not supported")
            if (isinstance(expert_rank, bool)
                    or not isinstance(expert_rank, int)):
                raise TypeError("expert_rank must be an integer")
            if not 0 <= expert_rank < ep_size:
                raise ValueError(
                    "expert_rank must be within expert_parallel_size")

            raw_owned = normalized["owned_experts"]
            if (not isinstance(raw_owned, (list, tuple))
                    or any(isinstance(value, bool)
                           or not isinstance(value, int)
                           for value in raw_owned)):
                raise TypeError("owned_experts must contain integers")
            owned = tuple(raw_owned)
            owned_count = shape["experts"] // ep_size
            first_owned = expert_rank * owned_count
            expected_owned = tuple(
                range(first_owned, first_owned + owned_count))
            if owned != expected_owned:
                raise ValueError(
                    "owned_experts disagrees with the canonical contiguous "
                    "expert-parallel placement")
            normalized["owned_experts"] = owned

            raw_local_counts = normalized["local_tokens_per_expert"]
            if (not isinstance(raw_local_counts, (list, tuple))
                    or any(isinstance(value, bool)
                           or not isinstance(value, int)
                           for value in raw_local_counts)):
                raise TypeError(
                    "local_tokens_per_expert must contain integers")
            local_counts = tuple(raw_local_counts)
            if (len(local_counts) != shape["experts"]
                    or any(value < 0 for value in local_counts)):
                raise ValueError(
                    "local_tokens_per_expert must contain one non-negative "
                    "count per expert")
            expected_local_counts = tuple(
                count if expert_id in owned else 0
                for expert_id, count in enumerate(counts)
            )
            if local_counts != expected_local_counts:
                raise ValueError(
                    "local_tokens_per_expert disagrees with global routing "
                    "and owned_experts")
            normalized["local_tokens_per_expert"] = local_counts
            normalized["expert_parallel_size"] = ep_size
            normalized["expert_rank"] = expert_rank
        elif kind is KernelKind.UNFUSED_MOE:
            local_counts = counts
            owned = tuple(range(shape["experts"]))
        else:
            raise AssertionError(f"unexpected MoE kernel kind: {kind!r}")
        windows = normalized["expert_weight_tile_rows"]
        if windows is not None:
            if (not isinstance(windows, (list, tuple))
                    or any(isinstance(value, bool)
                           or not isinstance(value, (int, float))
                           for value in windows)):
                raise TypeError(
                    "expert_weight_tile_rows must contain numeric values")
            windows = tuple(float(value) for value in windows)
            if len(windows) != shape["experts"]:
                raise ValueError(
                    "expert_weight_tile_rows must contain one value per expert")
            invalid_windows = [
                (count, window)
                for count, window in zip(local_counts, windows)
                if (not math.isfinite(window) or window < 0
                    or (count > 0 and window <= 0))
            ]
            if invalid_windows:
                raise ValueError(
                    "expert_weight_tile_rows must be finite and non-negative, "
                    "and positive for every active expert")
            if (kind is KernelKind.FUSED_MOE
                    and any(windows[expert_id] != 0.0
                            for expert_id in range(shape["experts"])
                            if expert_id not in owned)):
                raise ValueError(
                    "expert_weight_tile_rows must be zero for non-owned "
                    "experts")
            normalized["expert_weight_tile_rows"] = windows
        if kind is KernelKind.FUSED_MOE and "routed_weight_dtype" in normalized:
            routed_dtype = normalized["routed_weight_dtype"]
            if routed_dtype not in ("bf16", "fp8", "int4"):
                raise ValueError(
                    "FUSED_MOE routed_weight_dtype must be bf16, fp8, or int4")
            group_size = normalized["weight_group_size"]
            if isinstance(group_size, bool) or not isinstance(group_size, int):
                raise TypeError("FUSED_MOE weight_group_size must be an integer")
            if ((routed_dtype in ("bf16", "fp8") and group_size != 0)
                    or (routed_dtype == "int4" and group_size <= 0)):
                raise ValueError(
                    "FUSED_MOE weight_group_size disagrees with routed weight "
                    "dtype")
            if "fused_shared_experts" in normalized:
                fused = normalized["fused_shared_experts"]
                if isinstance(fused, bool) or fused != 1:
                    raise ValueError(
                        "FUSED_MOE fused_shared_experts must be 1 (one "
                        "shared expert fused into the routed kernel)")
                if routed_dtype != "fp8":
                    raise ValueError(
                        "FUSED_MOE fused shared experts are source-backed "
                        "only for the FP8 block kernel")
            scaling = normalized["routed_scaling_factor"]
            if (isinstance(scaling, bool)
                    or not isinstance(scaling, (int, float))
                    or not math.isfinite(scaling)
                    or scaling <= 0):
                raise ValueError(
                    "FUSED_MOE routed_scaling_factor must be finite and "
                    "positive")
            normalized["routed_scaling_factor"] = float(scaling)
    elif kind is KernelKind.GEMM and variant == "batched_gemm":
        if normalized["batch_layout"] != "strided":
            raise ValueError("GEMM/batched_gemm requires strided batch layout")
        expected_strides = {
            "a_batch_stride_elements": shape["m"] * shape["k"],
            "b_batch_stride_elements": shape["k"] * shape["n"],
            "c_batch_stride_elements": shape["m"] * shape["n"],
        }
        for name, expected in expected_strides.items():
            value = normalized[name]
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"GEMM/batched_gemm {name} must be an integer")
            if value != expected:
                raise ValueError(
                    f"GEMM/batched_gemm {name} must equal {expected}, got "
                    f"{value}")
        if (normalized["weight_dtype"] != "bf16"
                or normalized["weight_group_size"] != 0):
            raise ValueError(
                "GEMM/batched_gemm requires unquantized BF16 weights")
    elif kind is KernelKind.GEMM and variant == "moe_router":
        if normalized["score_function"] != "sigmoid":
            raise ValueError("GEMM/moe_router requires sigmoid scoring")
        if normalized["topk_method"] != "noaux_tc":
            raise ValueError("GEMM/moe_router requires noaux_tc top-k")
        for name in ("expert_group_count", "topk_group_count"):
            value = normalized[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"GEMM/moe_router {name} must be positive")
        if normalized["topk_group_count"] > normalized["expert_group_count"]:
            raise ValueError(
                "GEMM/moe_router top-k groups cannot exceed expert groups")
        if not isinstance(normalized["normalize_topk_probability"], bool):
            raise TypeError(
                "GEMM/moe_router normalize_topk_probability must be boolean")
        if (normalized["weight_dtype"] != "bf16"
                or normalized["weight_group_size"] != 0):
            raise ValueError("GEMM/moe_router requires unquantized BF16 weights")
    elif kind is KernelKind.SPARSE_ATTN_INDEXER:
        if variant == "dsa_index_reuse":
            if normalized["reuse_semantics"] != "alias":
                raise ValueError(
                    "SPARSE_ATTN_INDEXER reuse requires alias semantics")
        elif variant in (
                "dsa_indexer_prefill", "dsa_indexer_decode",
                "dsa_indexer_mixed"):
            if normalized["causal"] is not True:
                raise ValueError("DSA indexer requires causal=True")
            if normalized["score_accumulator_dtype"] != "fp32":
                raise ValueError("DSA indexer requires FP32 score accumulation")
            if normalized["score_function"] != "weighted_dot_product_topk":
                raise ValueError(
                    "DSA indexer requires weighted_dot_product_topk scoring")
            if normalized["cache_layout"] != "paged_mla_indexer":
                raise ValueError("DSA indexer cache layout is invalid")
            if normalized["cache_quantization"] != (
                    "fp8_e4m3_with_ue8m0_scale"):
                raise ValueError("DSA indexer cache quantization is invalid")
            phases = tuple(normalized["request_phases"])
            if (len(phases) != shape["batch"]
                    or any(phase not in ("prefill", "decode")
                           for phase in phases)):
                raise ValueError(
                    "DSA indexer request_phases must contain one valid phase "
                    "per request")
            expected_phases = {
                "dsa_indexer_prefill": {"prefill"},
                "dsa_indexer_decode": {"decode"},
                "dsa_indexer_mixed": {"prefill", "decode"},
            }[variant]
            if set(phases) != expected_phases:
                raise ValueError(
                    f"{variant} request phases disagree with its variant")
            normalized["request_phases"] = phases
            _normalize_request_lengths(
                "SPARSE_ATTN_INDEXER", normalized, shape)
            # The phase split carries no q_len constraint: the decode group
            # absorbs plain single-token decodes and MTP/EAGLE verify windows
            # (q_len > 1 through the absorbed decode path), while the prefill
            # group covers chunked-prefill rows and single-token prefill tail
            # chunks.  Per-request q_lens <= kv_lens is already enforced by
            # _normalize_request_lengths.
        else:
            raise AssertionError(
                f"unhandled SPARSE_ATTN_INDEXER variant: {variant!r}")
        for name in ("topk_capacity",):
            value = normalized[name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(
                    f"SPARSE_ATTN_INDEXER {name} must be positive")
        if normalized["topk_index_dtype"] != "int32":
            raise ValueError("DSA top-k index dtype must be int32")
        if variant != "dsa_index_reuse" and normalized["topk_sentinel"] != -1:
            raise ValueError("DSA indexer top-k sentinel must be -1")
        if (variant != "dsa_index_reuse"
                and shape["topk"] != min(
                    normalized["topk_capacity"], shape["kv_len"])):
            raise ValueError(
                "DSA indexer topk must equal min(topk_capacity, kv_len)")
    elif kind is KernelKind.SPARSE_ATTN:
        if normalized["causal"] is not True:
            raise ValueError("SPARSE_ATTN requires causal=True")
        if normalized["indexer_type"] not in ("full", "shared"):
            raise ValueError("SPARSE_ATTN indexer_type must be full or shared")
        if normalized["kv_layout"] != "mla":
            raise ValueError("SPARSE_ATTN requires kv_layout='mla'")
        if normalized["kv_dist"] != "uniform":
            raise ValueError("SPARSE_ATTN requires kv_dist='uniform'")
        if normalized["topk_index_dtype"] != "int32":
            raise ValueError("SPARSE_ATTN top-k indices must be int32")
        _normalize_request_lengths("SPARSE_ATTN", normalized, shape)
        selected = tuple(normalized["selected_kv_lens"])
        if (len(selected) != shape["batch"]
                or any(isinstance(value, bool) or not isinstance(value, int)
                       or value <= 0 for value in selected)):
            raise ValueError(
                "SPARSE_ATTN selected_kv_lens must contain one positive "
                "integer per request")
        if max(selected) != shape["topk"]:
            raise ValueError(
                "SPARSE_ATTN selected_kv_lens maximum disagrees with topk")
        normalized["selected_kv_lens"] = selected
        capacity = normalized["topk_capacity"]
        if (isinstance(capacity, bool) or not isinstance(capacity, int)
                or capacity <= 0 or shape["topk"] > capacity):
            raise ValueError(
                "SPARSE_ATTN topk_capacity must be positive and cover topk")
        expected_selected = tuple(
            min(capacity, kv_len) for kv_len in normalized["kv_lens"])
        if selected != expected_selected:
            raise ValueError(
                "SPARSE_ATTN selected_kv_lens disagree with KV lengths and "
                "top-k capacity")
    elif kind is KernelKind.KV_CACHE:
        if variant == "paged_kv_cache":
            layout = normalized["cache_layout"]
            if not isinstance(layout, str) or not layout.strip():
                raise TypeError(
                    "KV_CACHE/paged_kv_cache cache_layout must be a "
                    "non-empty string")
            if layout.strip().lower() != "paged":
                raise NotImplementedError(
                    "KV_CACHE/paged_kv_cache supports only paged "
                    "layout")
            normalized["cache_layout"] = "paged"
            head_layout = normalized["cache_head_layout"]
            if not isinstance(head_layout, str) or not head_layout.strip():
                raise TypeError(
                    "KV_CACHE/paged_kv_cache cache_head_layout must "
                    "be a non-empty string")
            head_layout = head_layout.strip().lower()
            if head_layout not in ("hnd", "nhd"):
                raise NotImplementedError(
                    "KV_CACHE/paged_kv_cache requires hnd or nhd "
                    "head layout")
            normalized["cache_head_layout"] = head_layout
            kv_cache_dtype = normalized["kv_cache_dtype"]
            if (not isinstance(kv_cache_dtype, str)
                    or not kv_cache_dtype.strip()):
                raise TypeError(
                    "KV_CACHE/paged_kv_cache kv_cache_dtype must be "
                    "a non-empty string")
            if kv_cache_dtype.strip().lower() != "auto":
                raise NotImplementedError(
                    "KV_CACHE/paged_kv_cache currently requires "
                    "kv_cache_dtype='auto'")
            normalized["kv_cache_dtype"] = "auto"
            for name in (
                    "cache_storage_matches_input", "slot_mapping_all_valid"):
                if not isinstance(normalized[name], bool):
                    raise TypeError(
                        f"KV_CACHE/paged_kv_cache {name} must be "
                        "boolean")
                if not normalized[name]:
                    raise NotImplementedError(
                        f"KV_CACHE/paged_kv_cache requires "
                        f"{name}=True")
            alignment = normalized["vector_copy_base_alignment_bytes"]
            if (isinstance(alignment, bool)
                    or not isinstance(alignment, int)):
                raise TypeError(
                    "KV_CACHE/paged_kv_cache alignment must be an "
                    "integer")
            if alignment <= 0 or alignment & (alignment - 1):
                raise ValueError(
                    "KV_CACHE/paged_kv_cache alignment must be a "
                    "positive power of two")
            expected_elements = (
                shape["seq_len"] * shape["num_kv_heads"]
                * shape["head_dim"])
        elif variant == "mla_kv_cache":
            if normalized["cache_write_semantics"] != "append":
                raise ValueError("KV_CACHE requires append semantics")
            if normalized["cache_layout"] != "paged_mla":
                raise ValueError(
                    "compressed MLA cache requires paged_mla layout")
            expected_elements = shape["seq_len"] * shape["compressed_dim"]
        elif variant == "dsa_fp8_index_cache":
            if normalized["cache_write_semantics"] != "append":
                raise ValueError("KV_CACHE requires append semantics")
            if normalized["cache_layout"] != "paged_mla_indexer":
                raise ValueError("DSA index cache layout is invalid")
            if normalized["quantization"] != (
                    "fp8_e4m3_with_ue8m0_scale"):
                raise ValueError("DSA index cache quantization is invalid")
            expected_elements = shape["seq_len"] * (
                shape["key_dim"] + shape["scale_bytes_per_token"])
        else:
            raise AssertionError(
                f"unhandled KV_CACHE variant: {variant!r}")
        if shape["element_count"] != expected_elements:
            raise ValueError(
                "KV_CACHE element_count disagrees with its typed "
                "cache representation")
    elif kind is KernelKind.VECTOR and variant == "layernorm":
        if not isinstance(normalized["affine"], bool):
            raise TypeError("VECTOR/layernorm affine must be boolean")
        epsilon = normalized["epsilon"]
        if (isinstance(epsilon, bool)
                or not isinstance(epsilon, (int, float))
                or not math.isfinite(epsilon)
                or epsilon <= 0):
            raise ValueError("VECTOR/layernorm epsilon must be positive")
        normalized["epsilon"] = float(epsilon)
    elif kind is KernelKind.VECTOR and variant == "mla_rope":
        if normalized["rope_layout"] != "interleaved":
            raise ValueError("VECTOR/mla_rope requires interleaved layout")
        if normalized["rope_type"] not in ("default", "deepseek_yarn"):
            raise ValueError("VECTOR/mla_rope has unsupported rope_type")
        theta = normalized["rope_theta"]
        if (isinstance(theta, bool)
                or not isinstance(theta, (int, float))
                or not math.isfinite(theta) or theta <= 0):
            raise ValueError("VECTOR/mla_rope rope_theta must be positive")
        normalized["rope_theta"] = float(theta)
        expected_elements = (
            shape["seq_len"]
            * (shape["num_query_heads"] + shape["num_key_heads"])
            * shape["head_dim"]
        )
        if shape["element_count"] != expected_elements:
            raise ValueError(
                "VECTOR/mla_rope element_count disagrees with head geometry")
        yarn_names = (
            "rope_factor", "rope_original_max_position_embeddings",
            "rope_beta_fast", "rope_beta_slow", "rope_mscale",
            "rope_mscale_all_dim",
        )
        if normalized["rope_type"] == "default":
            if any(normalized[name] is not None for name in yarn_names):
                raise ValueError("default MLA RoPE cannot carry YaRN fields")
        elif normalized["rope_type"] == "deepseek_yarn":
            if any(normalized[name] is None for name in yarn_names):
                raise ValueError("DeepSeek YaRN requires every scaling field")
            original_max = normalized[
                "rope_original_max_position_embeddings"]
            if (isinstance(original_max, bool)
                    or not isinstance(original_max, int)
                    or original_max <= 0):
                raise ValueError(
                    "DeepSeek YaRN original max position must be positive")
            for name in (
                    "rope_factor", "rope_beta_fast", "rope_beta_slow",
                    "rope_mscale", "rope_mscale_all_dim"):
                value = normalized[name]
                if (isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0):
                    raise ValueError(
                        f"DeepSeek YaRN {name} must be finite and positive")
                normalized[name] = float(value)
            if normalized["rope_beta_fast"] <= normalized["rope_beta_slow"]:
                raise ValueError(
                    "DeepSeek YaRN beta_fast must exceed beta_slow")
        else:
            raise AssertionError(
                f"unhandled MLA RoPE type: {normalized['rope_type']!r}")
    elif kind is KernelKind.VECTOR and variant == "rope":
        for name in ("is_neox", "key_present"):
            if not isinstance(normalized[name], bool):
                raise ValueError(f"VECTOR/rope {name} must be boolean")
    elif kind is KernelKind.VECTOR and variant in (
            "elementwise", "elementwise_pytorch_vec4"):
        for name in ("reads", "writes"):
            if isinstance(normalized[name], bool) or not isinstance(
                    normalized[name], int):
                raise TypeError(f"VECTOR {name} must be an integer")
        if normalized["reads"] < 0 or normalized["writes"] < 0:
            raise ValueError("VECTOR reads/writes cannot be negative")
        operation = normalized["operation"]
        if not isinstance(operation, str) or not operation.strip():
            raise TypeError(
                "VECTOR/elementwise operation must be a non-empty string")
        operation = operation.strip().lower()
        expected_io = {
            "add": (2, 1),
            "mul": (2, 1),
            # One tensor operand and a Python scalar (x + c, x * c, cumsum).
            "unary": (1, 1),
            # BF16 logits materialized as FP32 (sglang logits_processor).
            "cast_bf16_float32": (1, 1),
            "copy": (1, 1),
            "fill": (0, 1),
            "read": (1, 0),
            "notequal_int32_bool": (1, 1),
            "cast_bool_bf16": (1, 1),
        }
        if operation not in expected_io:
            raise ValueError(
                "VECTOR/elementwise operation must be one of "
                f"{sorted(expected_io)}, got {operation!r}")
        expected_reads, expected_writes = expected_io[operation]
        if (normalized["reads"], normalized["writes"]) != (
                expected_reads, expected_writes):
            raise ValueError(
                "VECTOR/elementwise reads/writes disagree with operation "
                f"{operation!r}: expected="
                f"({expected_reads}, {expected_writes}), actual="
                f"({normalized['reads']}, {normalized['writes']})")
        normalized["operation"] = operation
    elif kind is KernelKind.VECTOR and variant in (
            "moe_silu", "moe_mul", "moe_sum", "moe_reduce"):
        work_fraction = normalized["work_fraction"]
        if (isinstance(work_fraction, bool)
                or not isinstance(work_fraction, (int, float))):
            raise TypeError(
                f"VECTOR/{variant} work_fraction must be numeric")
        work_fraction = float(work_fraction)
        if not math.isfinite(work_fraction) or not 0 < work_fraction <= 1:
            raise ValueError(
                f"VECTOR/{variant} work_fraction must be finite and in "
                "(0, 1]")
        normalized["work_fraction"] = work_fraction
    elif kind is KernelKind.VECTOR and variant == "moe_fill":
        fill_role = normalized["fill_role"]
        if not isinstance(fill_role, str) or not fill_role.strip():
            raise TypeError("VECTOR/moe_fill fill_role must be a non-empty string")
        fill_role = fill_role.strip().lower()
        valid_fill_roles = set(MOE_FILL_ROLES)
        if fill_role not in valid_fill_roles:
            raise ValueError(
                "VECTOR/moe_fill fill_role must be one of "
                f"{sorted(valid_fill_roles)}, got {fill_role!r}")
        normalized["fill_role"] = fill_role
    elif kind is KernelKind.VECTOR and variant == "moe_cast":
        cast_role = normalized["cast_role"]
        if not isinstance(cast_role, str) or not cast_role.strip():
            raise TypeError(
                "VECTOR/moe_cast cast_role must be a non-empty string")
        cast_role = cast_role.strip().lower()
        valid_cast_roles = set(MOE_CAST_ROLES)
        if cast_role not in valid_cast_roles:
            raise ValueError(
                "VECTOR/moe_cast cast_role must be one of "
                f"{sorted(valid_cast_roles)}, got {cast_role!r}")
        normalized["cast_role"] = cast_role
    elif kind is KernelKind.VECTOR and variant == "reduce":
        if (isinstance(normalized["reduction_factor"], bool)
                or not isinstance(normalized["reduction_factor"], int)):
            raise TypeError("VECTOR reduction_factor must be an integer")
        if normalized["reduction_factor"] <= 0:
            raise ValueError("VECTOR reduction_factor must be positive")
    elif kind is KernelKind.VECTOR and variant == "gather":
        for name in ("gather_factor", "index_count", "index_bytes"):
            if isinstance(normalized[name], bool) or not isinstance(
                    normalized[name], int):
                raise TypeError(f"VECTOR {name} must be an integer")
        for name in ("gather_factor", "index_bytes"):
            if normalized[name] <= 0:
                raise ValueError(f"VECTOR {name} must be positive")
        if normalized["index_count"] < 0:
            raise ValueError("VECTOR index_count cannot be negative")
        if not isinstance(normalized["scattered"], bool):
            raise ValueError("VECTOR scattered must be boolean")
    elif kind is KernelKind.VECTOR and variant == "fp8_group_quant":
        group_size = normalized["group_size"]
        if isinstance(group_size, bool) or not isinstance(group_size, int):
            raise TypeError("VECTOR/fp8_group_quant group_size must be an integer")
        if group_size <= 0 or shape["element_count"] % group_size:
            raise ValueError(
                "VECTOR/fp8_group_quant needs whole quantisation groups")
    elif kind is KernelKind.VECTOR and variant == "gemm_reduce":
        split_k = normalized["split_k"]
        if isinstance(split_k, bool) or not isinstance(split_k, int):
            raise TypeError("VECTOR/gemm_reduce split_k must be an integer")
        if split_k <= 1:
            raise ValueError("VECTOR/gemm_reduce split_k must exceed one")
    return normalized


def _validate_compute_dtype_contract(
    kind: KernelKind,
    variant: str,
    dtype: str,
    shape: Mapping[str, int],
    attributes: Mapping[str, Any],
) -> None:
    if kind is KernelKind.SPARSE_ATTN_INDEXER:
        if dtype != "int32":
            raise ValueError("SPARSE_ATTN_INDEXER requires int32 output dtype")
        if (variant != "dsa_index_reuse"
                and shape["topk"] > attributes["topk_capacity"]):
            raise ValueError(
                "SPARSE_ATTN_INDEXER topk cannot exceed topk_capacity")
    elif kind is KernelKind.SPARSE_ATTN:
        if dtype not in ("bf16", "fp16"):
            raise ValueError("SPARSE_ATTN requires BF16 or FP16 compute dtype")
    elif kind is KernelKind.KV_CACHE:
        if variant == "dsa_fp8_index_cache":
            expected_dtype = "uint8"
        elif variant == "mla_kv_cache":
            expected_dtype = "bf16"
        elif variant == "paged_kv_cache":
            if dtype not in ("bf16", "fp16"):
                raise ValueError(
                    "KV_CACHE/paged_kv_cache requires BF16 or FP16 "
                    "dtype")
            expected_dtype = dtype
        else:
            raise AssertionError(
                f"unhandled KV_CACHE variant: {variant!r}")
        if dtype != expected_dtype:
            raise ValueError(
                f"KV_CACHE/{variant} requires {expected_dtype} dtype")
    elif kind is KernelKind.FLASH_ATTN and variant in (
            "mla_prefill", "mla_decode", "mla_absorbed_prefill"):
        if dtype not in ("bf16", "fp16"):
            raise ValueError("MLA attention requires BF16 or FP16 dtype")
    elif kind is KernelKind.FUSED_MOE and variant == "routed_experts":
        if dtype not in ("bf16", "fp16"):
            raise ValueError("routed experts require BF16 or FP16 activations")
        routed_dtype = attributes["routed_weight_dtype"]
        if routed_dtype == "int4":
            group_size = attributes["weight_group_size"]
            for dimension in (shape["hidden"], shape["intermediate"]):
                if dimension % group_size:
                    raise ValueError(
                        "INT4 routed-expert group size must divide hidden and "
                        "intermediate dimensions")


def _normalize_launch_parameters(
        kind: KernelKind,
        shape: Mapping[str, int],
        launch: LaunchConfig,
        attributes: Mapping[str, Any] | None = None,
) -> LaunchConfig:
    graph_padding = (
        attributes is not None
        and "graph_logical_tokens" in attributes
        and attributes["graph_logical_tokens"]
        != attributes["graph_model_input_tokens"])
    parameters = dict(launch.parameters)
    if (launch.source_program is not None
            and kind not in (
                KernelKind.GEMM, KernelKind.FLASH_ATTN, KernelKind.VECTOR)):
        raise ValueError(
            "source_program is supported only for GEMM, FLASH_ATTN or "
            "VECTOR launches")
    if (kind is KernelKind.VECTOR
            and launch.source_program is not None):
        raise TypeError(
            "VECTOR launches carry no source program: the vLLM-Ascend "
            "QKNormRope fusion is compiled into one split_qkv_rmsnorm_rope "
            "kernel for both the replayed-graph and fallback steps")
    if kind is KernelKind.GEMM:
        if (launch.source_program is not None
                and not isinstance(
                    launch.source_program,
                    (CannGemmLaunchProgram, CannGemmTilePipelineProgram,
                     GpuCtaRasterProgram))):
            raise TypeError(
                "GEMM source_program must be a typed CANN tile program or "
                "GPU CTA raster program")
        required = {
            "compulsory_traffic", "weight_tile_rows", "mte2_tile_bytes",
            "aiv_mixed", "n_major_traffic",
            "split_k_partial_bytes_per_element",
            "split_k_partial_dram_bytes_per_element",
        }
        if (launch.execution_variant == "moe_router"
                and "split_k_count_observed" in launch.parameters):
            # An observed-grid launch whose split count the record omits.
            required.add("split_k_count_observed")
    elif kind is KernelKind.FLASH_ATTN:
        if (launch.source_program is not None
                and not isinstance(
                    launch.source_program, FiaCrossDomainLaunchProgram)):
            raise TypeError(
                "FLASH_ATTN source_program must be a typed FIA program")
        base_required = {
            "unpadded_flops", "mma_smem_bytes_per_flop",
            "split_scratch_dram_bytes_per_element",
            "page_table_bytes_per_kv_panel",
        }
        source_schedule_required = base_required | {
            "cann_source_m_base_size",
            "cann_source_s2_base_size",
            "cann_source_task_count",
            "cann_source_tensor_logical_operations",
            "cann_source_tensor_shape_efficiency",
            "cann_source_tensor_schedule_efficiency",
            "cann_source_vector_operations",
            "cann_source_vector_main_operations",
            "cann_source_vector_initialization_operations",
            "cann_source_vector_terminal_operations",
            "cann_source_vector_vec1_operations",
            "cann_source_vector_vec2_operations",
            "cann_source_vector_worker_count",
            "cann_source_vector_schedule_efficiency",
            "cann_source_vec1_read_bytes",
            "cann_source_mm2_score_read_bytes",
            "cann_source_vec2_mm2_read_bytes",
            "cann_source_vec2_state_read_bytes",
            "cann_source_flash_decode_terminal_read_bytes",
            "cann_source_vec1_write_bytes",
            "cann_source_nupdate_write_bytes",
            "cann_source_vec2_write_bytes",
            "cann_source_flash_decode_terminal_write_bytes",
            "cann_source_output_init_bytes",
            "cann_source_external_write_bytes",
            "cann_source_vec1_unique_write_bytes",
            "cann_source_vec2_unique_write_bytes",
            "cann_source_aiv_unique_write_bytes",
            "cann_source_mm1_unique_write_bytes",
            "cann_source_mm2_unique_write_bytes",
            "cann_source_aic_unique_write_bytes",
        }
        if set(parameters) == base_required:
            required = base_required
        elif "cann_source_vector_worker_count" in parameters:
            worker_count_value = parameters[
                "cann_source_vector_worker_count"]
            if (not worker_count_value.is_integer()
                    or worker_count_value <= 0):
                raise ValueError(
                    "FLASH_ATTN source Vector worker count must be a positive "
                    "integer")
            worker_count = int(worker_count_value)
            worker_fields = {
                "generic_operations", "exp_operations", "div_operations",
                "exp_invocations", "div_invocations",
            }
            worker_required = {
                f"cann_source_vector_worker_{worker}_{field}"
                for worker in range(worker_count)
                for field in worker_fields
            }
            required = source_schedule_required | worker_required
        else:
            required = source_schedule_required
        if set(parameters) != required:
            raise ValueError(
                "FLASH_ATTN launch parameter schema mismatch: "
                f"actual={sorted(parameters)}")
    elif kind is KernelKind.FUSED_MOE:
        tiling = launch.tiling
        if tiling is None or not {"m", "n"}.issubset(tiling.tile_shape):
            raise ValueError("FUSED_MOE launch requires M and N tiles")
        required = {
            "group_size_m", "padded_rows", "mte2_tile_bytes",
            "weight_tile_rows", "operand_replay_a", "operand_replay_b",
        }
        cann_grouped_k_program = {
            "cann_depth_a1", "cann_depth_b1",
            "cann_step_ka", "cann_step_kb",
            "cann_step_m", "cann_step_n",
            "cann_db_l0a", "cann_db_l0b", "cann_db_l0c",
        }
        present_cann_program = set(parameters) & cann_grouped_k_program
        if present_cann_program:
            missing_cann_program = (
                cann_grouped_k_program - present_cann_program)
            if missing_cann_program:
                raise ValueError(
                    "FUSED_MOE CANN GroupedMatmul K program is incomplete: "
                    f"missing={sorted(missing_cann_program)}")
            required.update(cann_grouped_k_program)
        if "expert_parallel_size" in parameters:
            launch_ep = parameters["expert_parallel_size"]
            if (not launch_ep.is_integer() or launch_ep <= 1):
                raise ValueError(
                    "FUSED_MOE launch expert_parallel_size must be an "
                    "integer greater than one")
            required.add("expert_parallel_size")
        if "dynamic_shared_memory_bytes" in parameters:
            dynamic_shared_memory = parameters["dynamic_shared_memory_bytes"]
            if (not dynamic_shared_memory.is_integer()
                    or dynamic_shared_memory <= 0):
                raise ValueError(
                    "FUSED_MOE dynamic_shared_memory_bytes must be a "
                    "positive integer")
            required.add("dynamic_shared_memory_bytes")
        compiled_program_parameters = {"num_warps", "num_stages"}
        present_compiled_program = (
            set(parameters) & compiled_program_parameters)
        if present_compiled_program:
            # Two deployed configurations can share tile shape, block threads
            # and dynamic shared memory yet compile to different programs, so
            # these two source/compiler facts are recorded together or not at
            # all.
            if present_compiled_program != compiled_program_parameters:
                raise ValueError(
                    "FUSED_MOE compiled-program launch facts are incomplete: "
                    f"missing={sorted(compiled_program_parameters - present_compiled_program)}")
            for name in sorted(compiled_program_parameters):
                value = parameters[name]
                if not value.is_integer() or value <= 0:
                    raise ValueError(
                        f"FUSED_MOE {name} must be a positive integer")
            required.update(compiled_program_parameters)
        routed_weight_program_parameter = (
            SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER)
        if routed_weight_program_parameter in parameters:
            routed_weight_program = parameters[routed_weight_program_parameter]
            if (not routed_weight_program.is_integer()
                    or int(routed_weight_program) not in (0, 1)):
                raise ValueError(
                    "FUSED_MOE source W2 routed-weight program must be "
                    "zero or one")
            required.add(routed_weight_program_parameter)
    elif kind in (
            KernelKind.EAGER_ATTN, KernelKind.UNFUSED_MOE):
        required = set()
    elif kind in (
            KernelKind.SPARSE_ATTN_INDEXER, KernelKind.SPARSE_ATTN):
        required = set()
    elif kind is KernelKind.KV_CACHE:
        if launch.execution_variant == "paged_kv_cache":
            allowed = {
                "kv_cache_slot_l1_bytes", "lowered_sfu_operations",
                SOURCE_VECTORIZED_KV_COPY_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
                "source_bound_resident_ctas_per_unit",
                SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
                SOURCE_SCALAR_KV_LOOP_PARAMETER,
                SOURCE_PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_PARAMETER,
                "cann_tiling_key",
            }
            unknown = set(parameters) - allowed
            if unknown:
                raise ValueError(
                    "KV_CACHE/paged_kv_cache launch parameter schema "
                    f"mismatch: unknown={sorted(unknown)}")
            required = set(parameters)
        elif launch.execution_variant in (
                "mla_kv_cache", "dsa_fp8_index_cache"):
            required = set()
        else:
            raise AssertionError(
                "unhandled KV_CACHE launch variant: "
                f"{launch.execution_variant!r}")
    elif kind is KernelKind.VECTOR:
        variant = launch.execution_variant
        vector_schema_cache_key = None
        if variant in ("rmsnorm", "add_rmsnorm"):
            vector_schema_cache_key = (
                variant,
                (shape["hidden"] == 128 or shape["hidden"] % 256 == 0)
                if variant == "rmsnorm" else None,
                frozenset(parameters),
            )
            if vector_schema_cache_key in _VECTOR_PARAMETER_SCHEMA_CACHE:
                allowed_schemas = _CachedVectorParameterSchemas(
                    _VECTOR_PARAMETER_SCHEMA_CACHE[
                        vector_schema_cache_key])
            else:
                allowed_schemas = _PrunedVectorParameterSchemas(parameters)
        else:
            allowed_schemas = [set()]
        if variant in ("rmsnorm", "add_rmsnorm"):
            allowed_schemas.extend((
                {"input_reread_count"},
                {"input_reread_count", "lowered_sfu_operations"},
            ))
            allowed_schemas.extend(
                schema | {
                    "norm_reduction_tree_flops",
                    "norm_epsilon_add_flops",
                }
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.extend(
                schema | {"rstd_output_bytes"}
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.extend(
                schema | {"cann_rmsnorm_rstd_transfer_bytes"}
                for schema in tuple(allowed_schemas)
                if variant == "rmsnorm"
            )
            allowed_schemas.extend(
                schema | {
                    "lowered_vector_input_operations",
                    "lowered_vector_output_operations",
                    "norm_weight_l2_copies",
                }
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.extend(
                schema | {"lowered_vector_reduce_operations"}
                for schema in tuple(allowed_schemas)
                if "lowered_vector_input_operations" in schema
            )
            allowed_schemas.extend(
                schema | {
                    "cann_tiling_key", "cann_tiling_payload_bytes",
                    "norm_block_factor",
                }
                for schema in tuple(allowed_schemas)
            )
            if variant in ("rmsnorm", "add_rmsnorm"):
                allowed_schemas.extend(
                    schema | {"source_bound_block_reduction"}
                    for schema in tuple(allowed_schemas)
                )
                allowed_schemas.extend(
                    schema | {SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER}
                    for schema in tuple(allowed_schemas)
                    if ("source_bound_block_reduction" in schema
                        or variant == "add_rmsnorm")
                )
            if variant == "add_rmsnorm":
                allowed_schemas.extend(
                    schema | {"source_bound_add_rmsnorm_single_n_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
                allowed_schemas.extend(
                    schema | {"source_bound_add_rmsnorm_row_group_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
                allowed_schemas.extend(
                    schema | {"source_bound_add_rmsnorm_split_d_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
            elif variant == "rmsnorm":
                allowed_schemas.extend(
                    schema | {"source_bound_rmsnorm_single_row_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
                allowed_schemas.extend(
                    schema | {"source_bound_rmsnorm_row_group_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
                allowed_schemas.extend(
                    schema | {"source_bound_rmsnorm_split_d_dag"}
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
                # CANN 8.5.1 MERGE_N (key 2) publishes its row-group extent
                # and the host tiler's is_performance flag alongside the
                # source identity; both are discrete launch facts the key-2
                # worker program needs and no other key defines.
                allowed_schemas.extend(
                    schema | {
                        "source_bound_rmsnorm_merge_n_dag",
                        "norm_merge_n_row_factor",
                        "norm_merge_n_performance_flag",
                    }
                    for schema in tuple(allowed_schemas)
                    if "cann_tiling_key" in schema
                )
            else:
                raise AssertionError(
                    f"unexpected normalization variant: {variant!r}")
            # GPU launch packages publish the reduction-tree and epsilon
            # work together, while the fixed CANN sources expose only the
            # scalar epsilon add here (ReduceSum is retained as its own
            # dependency service).  Extend this optional source fact after
            # all NORM schema variants have been composed so it remains
            # compatible with rstd/lowered/tiling/DAG fields without
            # weakening the closed schema boundary.
            allowed_schemas.extend(
                schema | {"norm_epsilon_add_flops"}
                for schema in tuple(allowed_schemas)
                if "norm_epsilon_add_flops" not in schema
            )
            allowed_schemas.extend(
                schema | {SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            if variant in ("rmsnorm", "add_rmsnorm"):
                allowed_schemas.extend(
                    schema | {
                        SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER}
                    for schema in tuple(allowed_schemas)
                )
            allowed_schemas.extend(
                schema | {
                    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
        if variant in ("rotary", "rope"):
            rotary_schemas = (
                {"rotary_cache_path_bytes"},
                {
                    "rotary_cache_path_bytes",
                    "rotary_position_path_bytes",
                },
                {"rotary_cache_path_bytes", "lowered_sfu_operations"},
            )
            rotary_schemas += tuple(
                schema | {"rotary_position_path_bytes"}
                for schema in rotary_schemas
                if "rotary_position_path_bytes" not in schema
            )
            rotary_schemas += tuple(
                schema | {
                    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER}
                for schema in rotary_schemas
            )
            rotary_schemas += tuple(
                schema | {SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER}
                for schema in rotary_schemas
                if SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
                in schema
            )
            allowed_schemas.extend(rotary_schemas)
            allowed_schemas.extend(
                schema | {SOURCE_QK_ROPE_LOOP_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            for program_parameter in (
                    SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER,
                    SOURCE_METAX_ROPE_VLLM_0_9_1_ABI11_PROGRAM_PARAMETER,
                    SOURCE_METAX_ROPE_VLLM_0_24_0_ABI13_PROGRAM_PARAMETER):
                allowed_schemas.extend(
                    schema | {program_parameter}
                    for schema in tuple(allowed_schemas)
                    if SOURCE_QK_ROPE_LOOP_PARAMETER in schema
                )
            allowed_schemas.extend(
                schema | {SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            atb_rope_schema = {
                "cann_tiling_key",
                "lowered_vector_operations",
                "source_bound_atb_bf16_rope_dag",
                "atb_rope_slice_elements",
                "atb_rope_rotary_coefficient",
                "atb_rope_cos_format",
            }
            atb_rope_schema.add(
                SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER)
            # The public torch_npu RoPE entry has two GatherV3 prefixes
            # before it dispatches the ATB body.  Keep their selected
            # versioned program and typed entry payload in the normalized
            # launch contract: Dataset and Serving must not infer either at
            # prediction time.
            atb_rope_schema.update({
                SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
                "cann_rope_gather_v3_real_core_num",
                "cann_rope_gather_v3_selected_row_bytes",
                "cann_rope_gather_v3_tiling_payload_bytes",
            })
            allowed_schemas.append(atb_rope_schema)
            allowed_schemas.append({"cann_tiling_key"})
        if variant == "qkv_rmsnorm_rope":
            allowed_schemas = [{
                SOURCE_VLLM_ASCEND_SPLIT_QKV_RMSNORM_ROPE_PROGRAM_PARAMETER,
                "qk_batch_rows_per_tile",
                "v_batch_rows_per_tile",
                "source_ub_bytes",
            }]
        if variant in ("rotary", "rope"):
            allowed_schemas.append({"lowered_sfu_operations"})
        if variant == "moe_compute_expert_tokens":
            allowed_schemas = [{
                "cann_tiling_key",
                "cann_moe_compute_expert_tokens_mix_aiv_1_0",
            }]
        if variant == "moe_finalize_routing":
            allowed_schemas = [{"cann_tiling_key"}]
        if variant in (
                "moe_init_routing", "moe_init_routing_v2",
                "moe_init_routing_v3"):
            # The CANN 8.5.1 MoeInitRouting manifest declares a zero-AIC,
            # one-AIV MIX_AIV task.  This is a launch topology fact, not a
            # tunable predictor parameter, and it must not be admitted for
            # other vector variants.
            allowed_schemas = [{"cann_moe_init_routing_mix_aiv_1_0"}]
        if variant in ("activation", "silu"):
            allowed_schemas.append({"warp_instructions_lower_bound"})
            allowed_schemas.append({"lowered_vector_operations"})
            allowed_schemas.append({
                "lowered_vector_operations",
                "vector_exp_serial_invocations",
                "vector_div_serial_invocations",
            })
            allowed_schemas.append({
                "lowered_vector_operations",
                "vector_exp_serial_invocations",
                "vector_div_serial_invocations",
                "swiglu_base_row_len",
                "swiglu_base_col_len",
                "swiglu_total_tiles",
                "swiglu_queue_depth",
            })
            allowed_schemas.extend(
                schema | {SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.extend(
                schema | {SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.extend(
                schema | {SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER}
                for schema in tuple(allowed_schemas)
                if "warp_instructions_lower_bound" in schema
            )
            allowed_schemas.extend(
                schema | {
                    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
            allowed_schemas.append({
                SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER,
            })
            allowed_schemas.append({
                SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER,
                "warp_instructions_lower_bound",
            })
        if (variant == "moe_swiglu"
                and launch.algorithm == "aclnn_moe_swiglu"):
            allowed_schemas.append({
                "lowered_vector_operations",
                "vector_exp_serial_invocations",
                "vector_div_serial_invocations",
                "swiglu_base_row_len",
                "swiglu_base_col_len",
                "swiglu_total_tiles",
                "swiglu_queue_depth",
            })
        if variant in ("elementwise", "elementwise_pytorch_vec4"):
            allowed_schemas.append({"lowered_vector_operations"})
            if variant == "elementwise_pytorch_vec4":
                allowed_schemas.append({"warp_instructions"})
            if variant == "elementwise":
                elementwise_tbe_tiling_schema = {
                    "cann_tiling_key",
                    "cann_tiling_payload_bytes",
                    "elementwise_block_factor",
                    "elementwise_ub_factor",
                    "elementwise_input_queue_depth",
                    "elementwise_input_ping_pong_slots",
                    "elementwise_output_queue_depth",
                    SOURCE_CANN_BF16_ELEMENTWISE_DAG_PARAMETER,
                }
                allowed_schemas.extend(
                    schema | elementwise_tbe_tiling_schema
                    for schema in tuple(allowed_schemas)
                )
                allowed_schemas.extend(
                    schema | {
                        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER}
                    for schema in tuple(allowed_schemas)
                )
                allowed_schemas.extend(
                    schema | {SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER}
                    for schema in tuple(allowed_schemas)
                )
                allowed_schemas.extend(
                    schema | {SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER}
                    for schema in tuple(allowed_schemas)
                )
                allowed_schemas.extend(
                    schema | {
                        SOURCE_PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_PARAMETER}
                    for schema in tuple(allowed_schemas)
                    if SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER in schema
                )
            allowed_schemas.extend(
                schema | {SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER}
                for schema in tuple(allowed_schemas)
                if SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
                in schema
            )
        if (variant == "elementwise"
                and launch.algorithm in (
                    "observed_flash_attention_copy:elementwise",
                    "observed_norm_input_materialization_copy:elementwise")):
            allowed_schemas.append({
                SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER})
        if (variant == "elementwise"
                and launch.algorithm == (
                    "pytorch_2_4_metax_3_7_2_bf16_grouped_strided_copy")):
            allowed_schemas.append({
                SOURCE_METAX_BF16_GROUPED_STRIDED_COPY_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
            })
        if variant == "gemm_reduce":
            reducer_schema = {
                "split_k_partial_bytes_per_element",
                "split_k_partial_dram_bytes_per_element",
            }
            allowed_schemas.append(reducer_schema)
            allowed_schemas.append(reducer_schema | {"split_k_count_observed"})
            allowed_schemas.append(reducer_schema | {
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
                "warp_instructions_lower_bound",
            })
        if variant == "rmsnorm":
            allowed_schemas.extend(
                schema | {
                    SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
        source_task_schemas = []
        for schema in tuple(allowed_schemas):
            eligible = (
                (variant == "rmsnorm"
                 and {
                     "input_reread_count",
                     "norm_reduction_tree_flops",
                     "norm_epsilon_add_flops",
                }.issubset(schema))
                or (variant == "add_rmsnorm"
                    and {
                        "input_reread_count",
                        "norm_reduction_tree_flops",
                        "norm_epsilon_add_flops",
                    }.issubset(schema))
                or (variant in ("rotary", "rope")
                    and ("rotary_position_path_bytes" in schema
                         or SOURCE_QK_ROPE_LOOP_PARAMETER in schema))
                or (variant in ("activation", "silu")
                    and (not schema
                         or SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER
                         in schema))
                or (variant == "elementwise_pytorch_vec4" and not schema)
                or (variant == "elementwise"
                    and SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER in schema)
                or (variant == "add_rmsnorm"
                    and "warp_instructions_lower_bound" in schema)
                or (variant == "rope"
                    and {
                        "warp_instructions",
                        "rotary_position_path_bytes",
                    }.issubset(schema))
                or (variant == "silu"
                    and "warp_instructions_lower_bound" in schema)
                or (variant == "elementwise_pytorch_vec4"
                    and "warp_instructions" in schema)
                or (variant == "elementwise"
                    and launch.algorithm
                    == "observed_flash_attention_copy:elementwise"
                    and "warp_instructions" in schema)
            )
            if eligible:
                source_task_schemas.append(
                    schema | {SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER})
        read_source_required = (
            (variant == "rmsnorm"
             and (shape["hidden"] == 128
                  or shape["hidden"] % 256 == 0))
            or variant == "silu"
            or (variant == "elementwise_pytorch_vec4"
                and shape["element_count"] % 1024 == 0)
            or (variant == "elementwise"
                and launch.algorithm
                == "observed_flash_attention_copy:elementwise")
        )
        allowed_schemas.extend(source_task_schemas)
        if read_source_required:
            read_source_schemas = tuple(
                schema | {
                    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER}
                for schema in source_task_schemas
            )
            allowed_schemas.extend(read_source_schemas)
            if variant == "rmsnorm":
                allowed_schemas.extend(
                    schema | {"norm_weight_l2_copies"}
                    for schema in (
                        *source_task_schemas, *read_source_schemas)
                )
            allowed_schemas.extend(
                schema | {SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER}
                for schema in read_source_schemas
            )
        if variant == "moe_act_and_mul":
            # The gated-activation program reads two operands per element and
            # publishes their per-wavefront request depth; it carries no other
            # launch facts.
            allowed_schemas.append({
                SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
            })
            allowed_schemas.append({
                SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER})
        if variant in (
                "rotary", "rope", "activation", "silu"):
            allowed_schemas.extend(
                schema | {
                    SOURCE_PYTORCH_CPP_EXTENSION_INDEPENDENT_CTA_MEMORY_TASK_PARAMETER}
                for schema in tuple(allowed_schemas)
            )
        if set(parameters) not in allowed_schemas:
            raise ValueError(
                f"VECTOR/{variant} launch parameter schema mismatch: "
                f"actual={sorted(parameters)}")
        if (vector_schema_cache_key is not None
                and vector_schema_cache_key
                not in _VECTOR_PARAMETER_SCHEMA_CACHE):
            _VECTOR_PARAMETER_SCHEMA_CACHE[vector_schema_cache_key] = (
                frozenset({frozenset(parameters)}))
        required = set(parameters)
    else:
        raise ValueError(
            f"unhandled {kind.value} launch parameter contract")
    missing = required - set(parameters)
    unknown = set(parameters) - required
    if missing or unknown:
        raise ValueError(
            f"{kind.value} launch parameter schema mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}")
    if any(value < 0 for value in parameters.values()):
        raise ValueError(
            f"{kind.value} launch parameters cannot be negative: "
            f"{parameters}")
    if kind is KernelKind.GEMM:
        for name in (
                "compulsory_traffic", "aiv_mixed", "n_major_traffic"):
            if parameters[name] not in (0.0, 1.0):
                raise ValueError(
                    f"GEMM launch parameter {name} must be 0 or 1")
        partial_bytes = parameters["split_k_partial_bytes_per_element"]
        partial_dram_bytes = parameters[
            "split_k_partial_dram_bytes_per_element"]
        if not partial_bytes.is_integer():
            raise ValueError(
                "GEMM split_k_partial_bytes_per_element must be integral")
        if not partial_dram_bytes.is_integer():
            raise ValueError(
                "GEMM split_k_partial_dram_bytes_per_element must be "
                "integral")
        if partial_dram_bytes not in (0.0, partial_bytes):
            raise ValueError(
                "GEMM split-K partial DRAM width must be zero or equal to "
                "the materialized partial width")
        split_k = None if launch.tiling is None else launch.tiling.split_k
        if split_k is not None:
            if (split_k == 1
                    and (partial_bytes != 0 or partial_dram_bytes != 0)):
                raise ValueError(
                    "non-split GEMM cannot declare a partial element width")
    elif kind is KernelKind.FLASH_ATTN:
        if parameters["unpadded_flops"] not in (0.0, 1.0):
            raise ValueError(
                "FLASH_ATTN launch parameter unpadded_flops must be 0 or 1")
        if parameters["mma_smem_bytes_per_flop"] < 0:
            raise ValueError(
                "FLASH_ATTN mma_smem_bytes_per_flop cannot be negative")
        scratch_dram_width = parameters[
            "split_scratch_dram_bytes_per_element"]
        if (not scratch_dram_width.is_integer()
                or scratch_dram_width not in (0.0, 4.0)):
            raise ValueError(
                "FLASH_ATTN split scratch DRAM width must be zero or the "
                "four-byte FP32 materialized width")
        splits = (
            None if launch.tiling is None
            else launch.tiling.num_kv_splits)
        if splits == 1 and scratch_dram_width != 0.0:
            raise ValueError(
                "non-split FLASH_ATTN cannot declare a scratch DRAM width")
        page_table_bytes = parameters["page_table_bytes_per_kv_panel"]
        if (not page_table_bytes.is_integer()
                or page_table_bytes not in (0.0, 4.0)):
            raise ValueError(
                "FLASH_ATTN page-table access must be zero or one int32 "
                "entry per KV panel")
        if splits == 1 and page_table_bytes != 0.0:
            raise ValueError(
                "non-split FLASH_ATTN cannot declare page-table panel work")
        if "cann_source_task_count" in parameters:
            if (launch.source_program is None
                    and parameters[
                        "cann_source_vector_terminal_operations"] != 0.0):
                raise ValueError(
                    "FLASH_ATTN FlashDecode terminal lifecycle requires a "
                    "typed FIA main program")
            for name in (
                    "cann_source_task_count",
                    "cann_source_tensor_logical_operations",
                    "cann_source_vector_operations",
                    "cann_source_vector_main_operations",
                    "cann_source_vector_initialization_operations",
                    "cann_source_vector_terminal_operations",
                    "cann_source_vector_vec1_operations",
                    "cann_source_vector_vec2_operations",
                    "cann_source_vector_worker_count",
                    "cann_source_vec1_read_bytes",
                    "cann_source_mm2_score_read_bytes",
                    "cann_source_vec2_mm2_read_bytes",
                    "cann_source_vec2_state_read_bytes",
                    "cann_source_flash_decode_terminal_read_bytes",
                    "cann_source_vec1_write_bytes",
                    "cann_source_nupdate_write_bytes",
                    "cann_source_vec2_write_bytes",
                    "cann_source_flash_decode_terminal_write_bytes",
                    "cann_source_output_init_bytes",
                    "cann_source_external_write_bytes",
                    "cann_source_vec1_unique_write_bytes",
                    "cann_source_vec2_unique_write_bytes",
                    "cann_source_aiv_unique_write_bytes",
                    "cann_source_mm1_unique_write_bytes",
                    "cann_source_mm2_unique_write_bytes",
                    "cann_source_aic_unique_write_bytes"):
                if not parameters[name].is_integer():
                    raise ValueError(
                        f"FLASH_ATTN source parameter {name} must be integral")
            for name in (
                    "cann_source_task_count",
                    "cann_source_tensor_logical_operations",
                    "cann_source_vector_operations",
                    "cann_source_vector_main_operations"):
                if parameters[name] <= 0:
                    raise ValueError(
                        f"FLASH_ATTN source parameter {name} must be positive")
            for name in (
                    "cann_source_tensor_shape_efficiency",
                    "cann_source_tensor_schedule_efficiency",
                    "cann_source_vector_schedule_efficiency"):
                if not 0 < parameters[name] <= 1:
                    raise ValueError(
                        f"FLASH_ATTN source parameter {name} must be in "
                        "(0, 1]")
            if not math.isclose(
                    parameters["cann_source_vector_operations"],
                    parameters["cann_source_vector_main_operations"]
                    + parameters[
                        "cann_source_vector_terminal_operations"],
                    rel_tol=0.0, abs_tol=0.0):
                raise ValueError(
                    "FLASH_ATTN source Vector phases do not conserve "
                    "operations")
            if not math.isclose(
                    parameters["cann_source_vector_main_operations"],
                    parameters[
                        "cann_source_vector_initialization_operations"]
                    + parameters["cann_source_vector_vec1_operations"]
                    + parameters["cann_source_vector_vec2_operations"],
                    rel_tol=0.0, abs_tol=0.0):
                raise ValueError(
                    "FLASH_ATTN source Vec1/Vec2 phases do not conserve main "
                    "Vector operations")
            worker_count = int(parameters[
                "cann_source_vector_worker_count"])
            worker_operations = []
            for worker in range(worker_count):
                for service in ("generic", "exp", "div"):
                    operation_name = (
                        f"cann_source_vector_worker_{worker}_{service}_operations")
                    operation_value = parameters[operation_name]
                    if not operation_value.is_integer():
                        raise ValueError(
                            f"FLASH_ATTN source parameter {operation_name} "
                            "must be integral")
                    worker_operations.append(operation_value)
                for service in ("exp", "div"):
                    operation_name = (
                        f"cann_source_vector_worker_{worker}_{service}_operations")
                    invocation_name = (
                        f"cann_source_vector_worker_{worker}_{service}_invocations")
                    invocation_value = parameters[invocation_name]
                    if not invocation_value.is_integer():
                        raise ValueError(
                            f"FLASH_ATTN source parameter {invocation_name} "
                            "must be integral")
                    if ((parameters[operation_name] > 0)
                            != (invocation_value > 0)):
                        raise ValueError(
                            "FLASH_ATTN source worker Exp/Div operations and "
                            "invocations must be positive together")
            if not math.isclose(
                    math.fsum(worker_operations),
                    parameters["cann_source_vector_operations"],
                    rel_tol=0.0, abs_tol=0.0):
                raise ValueError(
                    "FLASH_ATTN source Vector workers do not conserve "
                    "operations")
            if (parameters["cann_source_external_write_bytes"]
                    < parameters["cann_source_output_init_bytes"]):
                raise ValueError(
                    "FLASH_ATTN external writes cannot be smaller than "
                    "output initialization")
            # The distinct AIV-written extent cannot exceed the bytes MTE3
            # actually moves through the same path.
            if (parameters["cann_source_aiv_unique_write_bytes"]
                    > parameters["cann_source_vec1_write_bytes"]
                    + parameters["cann_source_nupdate_write_bytes"]
                    + parameters["cann_source_vec2_write_bytes"]
                    + parameters[
                        "cann_source_flash_decode_terminal_write_bytes"]):
                raise ValueError(
                    "FLASH_ATTN distinct AIV writes exceed the AIV write "
                    "stream")
            # Likewise for the Fixpipe ring against the bytes the AIC
            # actually stores, which the Vector side reads back in full.
            if (parameters["cann_source_mm1_unique_write_bytes"]
                    > parameters["cann_source_vec1_read_bytes"]
                    or parameters["cann_source_mm2_unique_write_bytes"]
                    > parameters["cann_source_vec2_mm2_read_bytes"]
                    or parameters["cann_source_aic_unique_write_bytes"]
                    != parameters["cann_source_mm1_unique_write_bytes"]
                    + parameters["cann_source_mm2_unique_write_bytes"]):
                raise ValueError(
                    "FLASH_ATTN distinct AIC writes exceed the Fixpipe write "
                    "stream")
    elif kind is KernelKind.FUSED_MOE:
        for name in (
                "group_size_m", "padded_rows",
                "operand_replay_a", "operand_replay_b"):
            if not parameters[name].is_integer():
                raise ValueError(
                    f"FUSED_MOE launch parameter {name} must be integral")
        for name in ("operand_replay_a", "operand_replay_b"):
            if parameters[name] <= 0:
                raise ValueError(
                    f"FUSED_MOE launch parameter {name} must be positive")
    elif kind in (KernelKind.VECTOR, KernelKind.KV_CACHE):
        for name in (
                "input_reread_count", "lowered_sfu_operations",
                "lowered_vector_operations", "lowered_vector_input_operations",
                "lowered_vector_reduce_operations",
                "lowered_vector_output_operations", "norm_weight_l2_copies",
                "norm_epsilon_add_flops", "norm_reduction_tree_flops",
                "rotary_cache_path_bytes", "rotary_position_path_bytes",
                "atb_rope_slice_elements",
                "atb_rope_rotary_coefficient",
                "cann_rope_gather_v3_real_core_num",
                "cann_rope_gather_v3_selected_row_bytes",
                "cann_rope_gather_v3_tiling_payload_bytes",
                "kv_cache_slot_l1_bytes",
                "rstd_output_bytes", "cann_rmsnorm_rstd_transfer_bytes",
                "norm_merge_n_row_factor",
                "cann_tiling_payload_bytes", "warp_instructions",
                "warp_instructions_lower_bound", "norm_block_factor",
                "source_bound_block_reduction",
                SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER,
                SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER,
                SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER,
                SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
                SOURCE_QK_ROPE_LOOP_PARAMETER,
                SOURCE_SCALAR_KV_LOOP_PARAMETER,
                SOURCE_VECTORIZED_KV_COPY_PARAMETER,
                SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
                SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER,
                "source_bound_resident_ctas_per_unit",
                SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
                SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
                SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER,
                SOURCE_RMSNORM_VLLM_0_8_5_MACA_3_7_2_PROGRAM_PARAMETER,
                SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER,
                SOURCE_METAX_ROPE_VLLM_0_9_1_ABI11_PROGRAM_PARAMETER,
                SOURCE_METAX_ROPE_VLLM_0_24_0_ABI13_PROGRAM_PARAMETER,
                SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER,
                SOURCE_VECTOR_ELEMENTS_PER_THREAD_PARAMETER):
            if name in parameters:
                value = parameters[name]
                if not value.is_integer() or value <= 0:
                    raise ValueError(
                        f"VECTOR {name} must be a positive integer")
        if (not graph_padding
                and "norm_epsilon_add_flops" in parameters
                and parameters["norm_epsilon_add_flops"]
                != float(shape["rows"])):
            raise ValueError(
                "VECTOR norm_epsilon_add_flops must equal one scalar add "
                "per normalization row")
        if ("source_bound_block_reduction" in parameters
                and parameters["source_bound_block_reduction"] != 1.0):
            raise ValueError(
                "VECTOR source_bound_block_reduction must equal one")
        if (SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER in parameters
                and parameters[
                    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER] != 1.0):
            raise ValueError(
                "VECTOR source_bound_gpu_norm_cta_pipeline must equal one")
        if (SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER in parameters
                and launch.execution_variant not in (
                    "rmsnorm", "add_rmsnorm")):
            raise ValueError(
                "VECTOR source_bound_gpu_norm_cta_pipeline requires a GPU "
                "normalization launch")
        if SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER in parameters:
            if parameters[
                    SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER
                    ] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_silu_nonrecurrent_cta_pipeline "
                    "must equal one")
            if (not graph_padding
                    and (launch.execution_variant != "silu"
                    or launch.grid != (shape["tok"], 1, 1)
                    or launch.grid_size <= 108
                    or launch.block != (1024, 1, 1))):
                raise ValueError(
                    "VECTOR source_bound_silu_nonrecurrent_cta_pipeline "
                    "requires its BF16 multi-wave block1024 source domain")
        for source_loop_parameter in (
                SOURCE_QK_ROPE_LOOP_PARAMETER,
                SOURCE_SCALAR_KV_LOOP_PARAMETER,
                SOURCE_VECTORIZED_KV_COPY_PARAMETER):
            if (source_loop_parameter in parameters
                    and parameters[source_loop_parameter] != 1.0):
                raise ValueError(
                    f"VECTOR {source_loop_parameter} must equal one")
        metax_rope_program_markers = tuple(
            parameter for parameter in (
                SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER,
                SOURCE_METAX_ROPE_VLLM_0_9_1_ABI11_PROGRAM_PARAMETER,
                SOURCE_METAX_ROPE_VLLM_0_24_0_ABI13_PROGRAM_PARAMETER,
            )
            if parameter in parameters)
        if len(metax_rope_program_markers) > 1:
            raise ValueError(
                "MetaX RoPE versioned program identities are mutually "
                "exclusive")
        if (SOURCE_QK_ROPE_LOOP_PARAMETER in parameters
                and len(metax_rope_program_markers) != 1):
            raise ValueError(
                "MetaX source Q/K RoPE loop requires exactly one "
                "versioned program identity")
        if metax_rope_program_markers:
            if SOURCE_QK_ROPE_LOOP_PARAMETER not in parameters:
                raise ValueError(
                    "MetaX RoPE program identity requires the C500 "
                    "source Q/K loop")
            expected_variant = (
                "rope"
                if metax_rope_program_markers[0]
                == SOURCE_METAX_ROPE_VLLM_0_8_5_ABI10_PROGRAM_PARAMETER
                else "rotary")
            if launch.execution_variant != expected_variant:
                raise ValueError(
                    "MetaX RoPE program identity disagrees with the "
                    f"versioned execution variant: expected "
                    f"{expected_variant!r}")
        if (SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER in parameters
                and parameters[
                    SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER] != 1.0):
            raise ValueError(
                "VECTOR source_bound_memory_dependency_task must equal one")
        if SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER in parameters:
            dimensions = parameters[
                SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER]
            if dimensions not in (3.0, 4.0):
                raise ValueError(
                    "VECTOR source_bound_tensor_iterator_copy_task must "
                    "equal three or four")
            if (variant != "elementwise"
                    or launch.algorithm not in (
                        "observed_flash_attention_copy:elementwise",
                        "observed_norm_input_materialization_copy:"
                        "elementwise")):
                raise ValueError(
                    "VECTOR TensorIterator copy task requires an observed "
                    "FlashAttention or Norm input-materialization "
                    "elementwise copy")
        if (SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER in parameters
                and launch.execution_variant not in (
                    "rmsnorm", "add_rmsnorm", "rotary", "rope", "elementwise",
                    "elementwise_pytorch_vec4", "paged_kv_cache",
                    "moe_act_and_mul", "gemm_reduce")):
            raise ValueError(
                "VECTOR source-bound L2 read warps require a source-audited "
                "RMSNorm, Rotary, Elementwise, gated-activation or KV-cache "
                "launch")
        if ("source_bound_resident_ctas_per_unit" in parameters
                and (launch.execution_variant != "paged_kv_cache"
                     or SOURCE_BOUND_L2_READ_WARPS_PER_CTA_PARAMETER
                     not in parameters
                     or SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER
                     not in parameters)):
            raise ValueError(
                "VECTOR source-bound resident CTAs require an exact "
                "paged KV-cache source launch")
        if (SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER in parameters
                and SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER
                not in parameters
                and not any(marker in parameters for marker in (
                    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER,
                    SOURCE_BF16_CONVERSION_OPERATIONS_PARAMETER,
                    SOURCE_METAX_BF16_GROUPED_STRIDED_COPY_PARAMETER,
                    SOURCE_QK_ROPE_LOOP_PARAMETER,
                    SOURCE_SCALAR_KV_LOOP_PARAMETER,
                    SOURCE_VECTORIZED_KV_COPY_PARAMETER,
                    SOURCE_VLLM_0_24_0_ACT_AND_MUL_PROGRAM_PARAMETER,
                    "warp_instructions_lower_bound",
                ))):
            raise ValueError(
                "VECTOR launch parameter schema mismatch: source-bound L2 "
                "reads require a source identity")
        if "source_bound_add_rmsnorm_single_n_dag" in parameters:
            if parameters[
                    "source_bound_add_rmsnorm_single_n_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_single_n_dag must "
                    "equal one")
            if (launch.execution_variant != "add_rmsnorm"
                    or parameters["cann_tiling_key"] != 33.0):
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_single_n_dag requires "
                    "AddRmsNorm CANN key 33")
        if "source_bound_add_rmsnorm_row_group_dag" in parameters:
            if parameters[
                    "source_bound_add_rmsnorm_row_group_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_row_group_dag must "
                    "equal one")
            if (launch.execution_variant != "add_rmsnorm"
                    or parameters["cann_tiling_key"] != 30.0):
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_row_group_dag requires "
                    "AddRmsNorm CANN key 30")
        if "source_bound_add_rmsnorm_split_d_dag" in parameters:
            if parameters[
                    "source_bound_add_rmsnorm_split_d_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_split_d_dag must "
                    "equal one")
            if (launch.execution_variant != "add_rmsnorm"
                    or parameters["cann_tiling_key"] != 31.0):
                raise ValueError(
                    "VECTOR source_bound_add_rmsnorm_split_d_dag requires "
                    "AddRmsNorm CANN key 31")
        if sum(
                name in parameters
                for name in (
                    "source_bound_add_rmsnorm_single_n_dag",
                    "source_bound_add_rmsnorm_row_group_dag",
                    "source_bound_add_rmsnorm_split_d_dag",
                )) > 1:
            raise ValueError(
                "VECTOR AddRMSNorm source DAG identities are mutually "
                "exclusive")
        if "source_bound_rmsnorm_single_row_dag" in parameters:
            if parameters["source_bound_rmsnorm_single_row_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_single_row_dag must equal "
                    "one")
            if (launch.execution_variant != "rmsnorm"
                    or parameters["cann_tiling_key"] != 0.0):
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_single_row_dag requires "
                    "RmsNorm CANN key 0")
        if "source_bound_rmsnorm_row_group_dag" in parameters:
            if parameters["source_bound_rmsnorm_row_group_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_row_group_dag must equal "
                    "one")
            if (launch.execution_variant != "rmsnorm"
                    or parameters["cann_tiling_key"] != 0.0):
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_row_group_dag requires "
                    "RmsNorm CANN key 0")
        if "source_bound_rmsnorm_split_d_dag" in parameters:
            if parameters["source_bound_rmsnorm_split_d_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_split_d_dag must equal "
                    "one")
            if (launch.execution_variant != "rmsnorm"
                    or parameters["cann_tiling_key"] != 1.0):
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_split_d_dag requires "
                    "RmsNorm CANN key 1")
        if "source_bound_rmsnorm_merge_n_dag" in parameters:
            if parameters["source_bound_rmsnorm_merge_n_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_merge_n_dag must equal one")
            if (launch.execution_variant != "rmsnorm"
                    or parameters["cann_tiling_key"] != 2.0):
                raise ValueError(
                    "VECTOR source_bound_rmsnorm_merge_n_dag requires "
                    "RmsNorm CANN key 2")
            if parameters["norm_merge_n_row_factor"] < 1.0:
                raise ValueError(
                    "VECTOR RMSNorm MERGE_N row factor must be positive")
            if parameters["norm_merge_n_performance_flag"] not in (0.0, 1.0):
                raise ValueError(
                    "VECTOR RMSNorm MERGE_N is_performance flag must be "
                    "zero or one")
        if sum(
                name in parameters
                for name in (
                    "source_bound_rmsnorm_single_row_dag",
                    "source_bound_rmsnorm_row_group_dag",
                    "source_bound_rmsnorm_split_d_dag",
                    "source_bound_rmsnorm_merge_n_dag",
                )) > 1:
            raise ValueError(
                "VECTOR RMSNorm source DAG identities are mutually "
                "exclusive")
        if (not graph_padding
                and "source_bound_atb_bf16_rope_dag" in parameters):
            if parameters["source_bound_atb_bf16_rope_dag"] != 1.0:
                raise ValueError(
                    "VECTOR source_bound_atb_bf16_rope_dag must equal one")
            if launch.execution_variant not in ("rotary", "rope"):
                raise ValueError(
                    "VECTOR ATB BF16 RoPE DAG requires a RoPE launch")
            if parameters["atb_rope_rotary_coefficient"] != 2.0:
                raise ValueError(
                    "VECTOR ATB BF16 RoPE rotary coefficient must equal two")
            if parameters["atb_rope_cos_format"] != 0.0:
                raise ValueError(
                    "VECTOR ATB BF16 RoPE cos format must equal zero")
            if shape["head_dim"] % 32:
                raise ValueError(
                    "VECTOR ATB BF16 RoPE DAG requires a 32-element aligned "
                    "head dimension")
            if launch.execution_variant == "rope":
                if shape["rotary_dim"] != shape["head_dim"]:
                    raise ValueError(
                        "VECTOR ATB BF16 RoPE DAG requires full-dimension "
                        "semantics")
            expected_key = 24 if shape["seq_len"] >= 64 else 22
            if parameters["cann_tiling_key"] != float(expected_key):
                raise ValueError(
                    "VECTOR ATB BF16 RoPE tiling key disagrees with the "
                    "source selector")
            nominal_slice = 2_048 if expected_key == 24 else 4_096
            expected_slice = (
                nominal_slice // shape["head_dim"] * shape["head_dim"])
            if (expected_slice <= 0
                    or parameters["atb_rope_slice_elements"]
                    != float(expected_slice)):
                raise ValueError(
                    "VECTOR ATB BF16 RoPE slice disagrees with the selected "
                    "source kernel")
            query_elements = shape["num_query_heads"] * shape["head_dim"]
            key_elements = shape["num_kv_heads"] * shape["head_dim"]
            if query_elements % 64 or key_elements % 64:
                raise ValueError(
                    "VECTOR ATB BF16 RoPE DAG requires complete 64-element "
                    "Vector repeats")
            expected_lowered = (
                5 * shape["element_count"]
                + 2 * shape["seq_len"] * min(query_elements, expected_slice))
            worker_count = launch.grid_size
            base_tokens_per_worker, worker_remainder = divmod(
                shape["seq_len"], worker_count)
            second_mask_workers = (
                worker_count
                if expected_key == 24 and base_tokens_per_worker > 1
                else (
                    worker_remainder
                    if expected_key == 24 and base_tokens_per_worker == 1
                    else 0))
            expected_lowered += shape["head_dim"] * (
                worker_count + second_mask_workers)
            query_heads_per_slice = min(
                query_elements, expected_slice) // shape["head_dim"]
            mask_copy_operations = (
                (worker_count + second_mask_workers)
                * (query_heads_per_slice - 1)
                * shape["head_dim"] * 4)
            cos_sin_copy_operations = (
                shape["seq_len"] * 2 * (query_heads_per_slice - 1)
                * shape["head_dim"] * 2)
            half_rotation_copy_operations = (
                2 * shape["seq_len"] * (query_elements + key_elements))
            expected_lowered += (
                mask_copy_operations
                + cos_sin_copy_operations
                + half_rotation_copy_operations)
            if parameters["lowered_vector_operations"] != float(
                    expected_lowered):
                raise ValueError(
                    "VECTOR ATB BF16 RoPE lowered work disagrees with the "
                    "source program")
            gather_fields = {
                SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER,
                SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER,
                "cann_rope_gather_v3_real_core_num",
                "cann_rope_gather_v3_selected_row_bytes",
                "cann_rope_gather_v3_tiling_payload_bytes",
            }
            missing_gather_fields = gather_fields - set(parameters)
            if missing_gather_fields:
                raise ValueError(
                    "VECTOR public ATB RoPE GatherV3 launch is incomplete: "
                    f"missing={sorted(missing_gather_fields)}")
            if parameters[SOURCE_CANN_ROPE_INDEX_SELECT_COUNT_PARAMETER] != 2.0:
                raise ValueError(
                    "VECTOR public ATB RoPE requires exactly two GatherV3 "
                    "prefixes")
            gather_program = parameters[
                SOURCE_CANN_ROPE_GATHER_V3_PROGRAM_PARAMETER]
            if (not gather_program.is_integer()
                    or int(gather_program) != 10_330):
                raise ValueError(
                    "VECTOR public ATB RoPE GatherV3 must select the CANN "
                    "8.5.1 direct-row key 10330")
            gather_workers = parameters["cann_rope_gather_v3_real_core_num"]
            if not gather_workers.is_integer() or gather_workers <= 0:
                raise ValueError(
                    "VECTOR public ATB RoPE GatherV3 realCoreNum must be a "
                    "positive integer")
            selected_row_bytes = parameters[
                "cann_rope_gather_v3_selected_row_bytes"]
            if (not selected_row_bytes.is_integer()
                    or selected_row_bytes != float(
                        shape["seq_len"] * shape["head_dim"] * 2)):
                raise ValueError(
                    "VECTOR public ATB RoPE GatherV3 selected-row bytes "
                    "disagree with its BF16 [T, D] output ABI")
            if parameters["cann_rope_gather_v3_tiling_payload_bytes"] != 192.0:
                raise ValueError(
                    "VECTOR public ATB RoPE GatherV3 requires its fixed "
                    "192-byte CANN entry payload")
        for name in (
                "vector_exp_serial_invocations",
                "vector_div_serial_invocations"):
            if name in parameters:
                value = parameters[name]
                if not value.is_integer() or value <= 0:
                    raise ValueError(
                        f"VECTOR {name} must be a positive integer")
        if "cann_tiling_key" in parameters:
            tiling_key = parameters["cann_tiling_key"]
            if launch.execution_variant == "rmsnorm":
                # rms_norm.cpp:31-34/49-58 binds key 0 to KernelRmsNorm, key 1
                # to KernelRmsNormSplitD and key 2 to KernelRmsNormMergeN.
                allowed_keys = (0, 1, 2)
                key_owner = "RmsNorm"
            elif launch.execution_variant == "add_rmsnorm":
                allowed_keys = (30, 31, 33)
                key_owner = "AddRmsNorm"
            elif launch.execution_variant == "paged_kv_cache":
                allowed_keys = (200_000_000, 202_000_000)
                key_owner = "ReshapeAndCache"
            elif launch.execution_variant in ("rotary", "rope"):
                allowed_keys = (22, 24)
                key_owner = "ATB BF16 RoPE"
            elif launch.execution_variant == "moe_compute_expert_tokens":
                allowed_keys = (1001, 1002, 1003)
                key_owner = "MoeComputeExpertTokens"
            elif launch.execution_variant == "moe_finalize_routing":
                allowed_keys = (20036,)
                key_owner = "MoeFinalizeRoutingV2 BF16 no-bias"
            elif launch.execution_variant == "elementwise":
                allowed_keys = (210_000_000, 210_010_000, 210_020_000)
                key_owner = "TBE elementwise"
            else:
                raise ValueError(
                    "VECTOR cann_tiling_key requires a supported CANN "
                    "operator variant")
            if (not tiling_key.is_integer()
                    or int(tiling_key) not in allowed_keys):
                raise ValueError(
                    f"VECTOR cann_tiling_key must be one of the supported "
                    f"CANN 8.5.1 {key_owner} keys {allowed_keys}")
            if launch.execution_variant in ("rmsnorm", "add_rmsnorm"):
                # ``GET_TILING_DATA`` deserializes the source entry's typed
                # payload before the key-specific program begins.  This is a
                # fail-closed normalized launch fact: 28-byte RmsNorm and
                # 80-byte AddRmsNorm payloads use different independently
                # measured materialization mechanisms.
                payload_bytes = parameters["cann_tiling_payload_bytes"]
                expected_payload_bytes = (
                    28.0 if launch.execution_variant == "rmsnorm" else 80.0)
                if payload_bytes != expected_payload_bytes:
                    raise ValueError(
                        "VECTOR CANN normalization tiling payload disagrees "
                        "with its fixed CANN 8.5.1 source entry: "
                        f"expected={expected_payload_bytes}, "
                        f"actual={payload_bytes}")
            if launch.execution_variant == "moe_compute_expert_tokens":
                from src.kernel.launch_config.ascend_910b.vector_launch import (
                    select_moe_compute_expert_tokens_tiling_key,
                )
                expected_key = select_moe_compute_expert_tokens_tiling_key(
                    shape["tokens"] * shape["topk"], shape["experts"])
                if int(tiling_key) != expected_key:
                    raise ValueError(
                        "VECTOR MoeComputeExpertTokens cann_tiling_key "
                        "disagrees with the source selector: "
                        f"expected={expected_key}, actual={int(tiling_key)}")
                task_topology = parameters[
                    "cann_moe_compute_expert_tokens_mix_aiv_1_0"]
                if task_topology != 1.0:
                    raise ValueError(
                        "VECTOR MoeComputeExpertTokens MIX_AIV 1:0 source "
                        "topology must be exactly one")
            if launch.execution_variant == "elementwise":
                # CANN 8.5.1 ``BroadcastSch<201>`` invokes
                # ``GET_TILING_DATA_WITH_STRUCT(BroadcastOneDimTilingData,
                # ...)``. The fixed source layout contains five int32, two
                # aligned int64, and 16 scalar bytes: 56 B in total.
                if parameters["cann_tiling_payload_bytes"] != 56.0:
                    raise ValueError(
                        "VECTOR CANN elementwise tiling payload disagrees "
                        "with its fixed CANN 8.5.1 BroadcastOneDim source "
                        "entry: expected=56.0, actual="
                        f"{parameters['cann_tiling_payload_bytes']}")
                block_factor = parameters["elementwise_block_factor"]
                ub_factor = parameters["elementwise_ub_factor"]
                input_depth = parameters[
                    "elementwise_input_queue_depth"]
                input_ping_pong_slots = parameters[
                    "elementwise_input_ping_pong_slots"]
                output_depth = parameters[
                    "elementwise_output_queue_depth"]
                for name, value in (
                        ("elementwise_block_factor", block_factor),
                        ("elementwise_ub_factor", ub_factor),
                        ("elementwise_input_queue_depth", input_depth),
                        ("elementwise_input_ping_pong_slots",
                         input_ping_pong_slots),
                        ("elementwise_output_queue_depth", output_depth)):
                    if not value.is_integer() or value <= 0:
                        raise ValueError(
                            f"VECTOR {name} must be a positive integer")
                elements = shape["element_count"]
                if int(block_factor) < int(ub_factor):
                    raise ValueError(
                        "VECTOR elementwise UB factor cannot exceed its "
                        "block factor")
                # Under graph padding the launch is selected on the captured
                # model-input extent while ``shape`` carries logical work, so
                # the two element counts legitimately differ; the same
                # relaxation guards every other shape-derived VECTOR
                # cross-check in this function.
                if (not graph_padding
                        and launch.grid_size != math.ceil(
                            elements / block_factor)):
                    raise ValueError(
                        "VECTOR elementwise block factor disagrees with its "
                        "launch grid")
                expected_depths = {
                    210_000_000: (1, 2, 1),
                    210_010_000: (2, 2, 1),
                    210_020_000: (4, 2, 1),
                }[int(tiling_key)]
                if (int(input_depth), int(input_ping_pong_slots),
                        int(output_depth)) != expected_depths:
                    raise ValueError(
                        "VECTOR elementwise queue/admission depths disagree "
                        "with its CANN tiling key")
        if "split_k_partial_bytes_per_element" in parameters:
            partial_bytes = parameters[
                "split_k_partial_bytes_per_element"]
            if not partial_bytes.is_integer() or partial_bytes <= 0:
                raise ValueError(
                    "VECTOR split_k_partial_bytes_per_element must be a "
                    "positive integer")
            partial_dram_bytes = parameters[
                "split_k_partial_dram_bytes_per_element"]
            if not partial_dram_bytes.is_integer():
                raise ValueError(
                    "VECTOR split_k_partial_dram_bytes_per_element must be "
                    "integral")
            if partial_dram_bytes not in (0.0, partial_bytes):
                raise ValueError(
                    "VECTOR split-K partial DRAM width must be zero or equal "
                    "to the materialized partial width")
    elif kind in (
            KernelKind.EAGER_ATTN, KernelKind.UNFUSED_MOE,
            KernelKind.SPARSE_ATTN_INDEXER, KernelKind.SPARSE_ATTN):
        pass
    else:
        raise ValueError(
            f"unhandled {kind.value} launch parameter validation")
    return launch


@dataclass(frozen=True)
class L2ExecutionContext:
    """Explicit cache history supplied by an execution owner.

    ``same_program_operand_replays`` counts immediately preceding launches
    that reuse this invocation's immutable operand allocations.  It does not
    assert that result buffers reuse an address: a predictor must conservatively
    account for every predecessor result allocation before declaring an operand
    resident.  Dataset capture and Serving lowering may supply different
    contexts, but they retain the same launch program and predictor path.
    """

    same_program_operand_replays: int = 0

    def __post_init__(self) -> None:
        value = self.same_program_operand_replays
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                "L2ExecutionContext same_program_operand_replays must be an "
                "integer")
        if value < 0:
            raise ValueError(
                "L2ExecutionContext same_program_operand_replays must be "
                "nonnegative")


@dataclass(frozen=True)
class KernelSpec:
    """A complete hardware modeling primitive consumed by predictors."""

    kind: KernelKind
    hardware_id: str
    shape: Mapping[str, int] = field(default_factory=dict)
    dtype: Optional[str] = None
    variant: Optional[str] = None
    launch: Optional[LaunchConfig] = None
    transfer: Optional[PhysicalTransferConfig] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    anonymous_l2_dirty_bytes: int = 0
    execution_support: str = "supported"
    unsupported_reason: Optional[str] = None
    l2_execution_context: L2ExecutionContext = field(
        default_factory=L2ExecutionContext)

    def __post_init__(self) -> None:
        kind = KernelKind.parse(self.kind)
        object.__setattr__(self, "kind", kind)
        from src.kernel.hardware_ids import canonical_hardware_id
        hardware_id = canonical_hardware_id(self.hardware_id)
        object.__setattr__(self, "hardware_id", hardware_id)
        if not isinstance(self.shape, Mapping):
            raise TypeError("KernelSpec.shape must be a mapping")
        if not isinstance(self.attributes, Mapping):
            raise TypeError("KernelSpec.attributes must be a mapping")
        if (isinstance(self.anonymous_l2_dirty_bytes, bool)
                or not isinstance(self.anonymous_l2_dirty_bytes, int)):
            raise TypeError(
                "KernelSpec anonymous_l2_dirty_bytes must be an integer")
        if self.anonymous_l2_dirty_bytes < 0:
            raise ValueError(
                "KernelSpec anonymous_l2_dirty_bytes must be nonnegative")
        if not isinstance(self.l2_execution_context, L2ExecutionContext):
            raise TypeError(
                "KernelSpec l2_execution_context must be an "
                "L2ExecutionContext")
        invalid_shape_keys = [
            name for name in self.shape if not isinstance(name, str) or not name
        ]
        if invalid_shape_keys:
            raise TypeError(
                f"shape keys must be non-empty strings: {invalid_shape_keys}")
        invalid_shape_types = {
            name: value for name, value in self.shape.items()
            if isinstance(value, bool) or not isinstance(value, int)
        }
        if invalid_shape_types:
            raise TypeError(
                f"shape dimensions must be integers: {invalid_shape_types}")
        normalized_shape = dict(self.shape)
        if any(v < 0 for v in normalized_shape.values()):
            raise ValueError(f"shape dimensions cannot be negative: {normalized_shape}")
        object.__setattr__(self, "shape", _immutable_mapping(normalized_shape))
        object.__setattr__(self, "attributes", _immutable_mapping(self.attributes))
        if self.execution_support not in ("supported", "unsupported"):
            raise ValueError(
                "KernelSpec.execution_support must be supported or unsupported")
        if self.execution_support == "supported":
            if self.unsupported_reason is not None:
                raise ValueError(
                    "supported KernelSpec cannot declare unsupported_reason")
        elif self.execution_support == "unsupported":
            if self.unsupported_reason != "launch_selector_unavailable":
                raise ValueError(
                    "unsupported KernelSpec requires unsupported_reason="
                    "'launch_selector_unavailable'")
        else:
            raise AssertionError(
                f"unhandled execution support: {self.execution_support!r}")

        compute_kinds = {
            KernelKind.GEMM,
            KernelKind.FLASH_ATTN,
            KernelKind.EAGER_ATTN,
            KernelKind.SPARSE_ATTN_INDEXER,
            KernelKind.SPARSE_ATTN,
            KernelKind.KV_CACHE,
            KernelKind.FUSED_MOE,
            KernelKind.UNFUSED_MOE,
            KernelKind.VECTOR,
        }
        transfer_kinds = {
            KernelKind.COMMUNICATION,
            KernelKind.MEMCPY,
            KernelKind.KV_OFFLOAD,
        }
        if kind in compute_kinds:
            if self.execution_support == "supported" and self.launch is None:
                raise ValueError(
                    f"supported {kind.value} requires a complete LaunchConfig")
            if self.execution_support == "unsupported" and self.launch is not None:
                raise ValueError(
                    f"unsupported {kind.value} cannot contain LaunchConfig")
            if self.transfer is not None:
                raise ValueError(f"{kind.value} cannot contain transfer configuration")
            if (self.dtype is None or not isinstance(self.dtype, str)
                    or not self.dtype.strip()):
                raise ValueError(f"{kind.value} requires dtype")
            if (self.variant is None or not isinstance(self.variant, str)
                    or not self.variant.strip()):
                raise ValueError(f"{kind.value} requires variant")
            normalized_variant = self.variant.strip().lower()
            if self.launch is not None:
                if self.launch.execution_variant is None:
                    raise ValueError(
                        f"{kind.value} launch requires execution_variant")
                if self.launch.execution_variant != normalized_variant:
                    raise ValueError(
                        f"{kind.value} variant {normalized_variant!r} does not "
                        "match launch execution_variant "
                        f"{self.launch.execution_variant!r}")
            object.__setattr__(self, "dtype", self.dtype.strip().lower())
            object.__setattr__(self, "variant", normalized_variant)
            normalized_attributes = None
            if kind is KernelKind.VECTOR and normalized_variant == "rope":
                # Typed RoPE element work depends on the validated key-presence
                # contract.  Validate that closed attribute schema before
                # deriving the canonical active-element count; all other
                # variants retain the established shape-first boundary.
                normalized_attributes = _normalize_compute_attributes(
                    kind, normalized_variant, self.attributes,
                    normalized_shape)
            normalized_shape = _normalize_compute_shape(
                kind, normalized_variant, self.dtype.strip().lower(),
                normalized_shape,
                (normalized_attributes
                 if normalized_attributes is not None else self.attributes))
            if any(value <= 0 for value in normalized_shape.values()):
                raise ValueError(
                    f"compute shape dimensions must be positive: "
                    f"{normalized_shape}")
            _require_shape_fields(kind, normalized_variant, normalized_shape)
            if (kind in (KernelKind.FLASH_ATTN, KernelKind.EAGER_ATTN)
                    and not (
                        kind is KernelKind.FLASH_ATTN
                        and normalized_variant in (
                            "mla_prefill", "mla_decode",
                            "mla_absorbed_prefill"))):
                query_heads = normalized_shape["num_query_heads"]
                kv_heads = normalized_shape["num_kv_heads"]
                if query_heads < kv_heads or query_heads % kv_heads:
                    raise ValueError(
                        f"{kind.value} num_query_heads must be an integer "
                        "multiple of num_kv_heads")
            if (kind in (KernelKind.FUSED_MOE, KernelKind.UNFUSED_MOE)
                    and normalized_shape["topk"] > normalized_shape["experts"]):
                raise ValueError(f"{kind.value} topk cannot exceed experts")
            if normalized_attributes is None:
                normalized_attributes = _normalize_compute_attributes(
                    kind, normalized_variant, self.attributes,
                    normalized_shape)
            _validate_compute_dtype_contract(
                kind,
                normalized_variant,
                self.dtype.strip().lower(),
                normalized_shape,
                normalized_attributes,
            )
            if self.launch is None:
                object.__setattr__(
                    self, "shape", _immutable_mapping(normalized_shape))
                object.__setattr__(
                    self, "attributes", _immutable_mapping(normalized_attributes))
                return
            normalized_launch = _normalize_launch_parameters(
                kind, normalized_shape, self.launch, normalized_attributes)
            if (kind is KernelKind.VECTOR
                    and normalized_variant == "moe_swiglu"):
                from src.kernel.launch_config.ascend_910b.vector_launch import (
                    validate_cann_moe_swiglu_launch,
                )
                validate_cann_moe_swiglu_launch(
                    hardware_id,
                    self.dtype.strip().lower(),
                    normalized_shape,
                    normalized_attributes,
                    normalized_launch,
                )
            if (kind is KernelKind.VECTOR
                    and "cann_rmsnorm_rstd_transfer_bytes"
                    in normalized_launch.parameters
                    and (hardware_id != "ascend_910b"
                         or self.dtype.strip().lower() != "bf16"
                         or normalized_variant != "rmsnorm")):
                raise ValueError(
                    "CANN RMSNorm rstd transfer geometry requires "
                    "ascend_910b BF16 RMSNorm")
            if (kind is KernelKind.VECTOR
                    and "source_bound_atb_bf16_rope_dag"
                    in normalized_launch.parameters):
                if hardware_id != "ascend_910b" or self.dtype != "bf16":
                    raise ValueError(
                        "VECTOR ATB BF16 RoPE DAG requires ascend_910b BF16")
                if (normalized_variant == "rope"
                        and (not normalized_attributes["is_neox"]
                             or not normalized_attributes["key_present"])):
                    raise ValueError(
                        "VECTOR ATB BF16 RoPE DAG requires NeoX Q+K "
                        "semantics")
            if (kind in (KernelKind.VECTOR, KernelKind.KV_CACHE)
                    and any(parameter in normalized_launch.parameters for parameter in (
                        SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
                        SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
                    ))):
                from src.kernel.launch_config.nvidia_a800.vector_launch import (
                    validate_source_bound_vector_launch,
                )
                source_launch_shape = normalized_shape
                if "model_input_shape" in normalized_attributes:
                    source_launch_shape = normalized_attributes[
                        "model_input_shape"]
                validate_source_bound_vector_launch(
                    hardware_id,
                    self.dtype.strip().lower(),
                    normalized_variant,
                    source_launch_shape,
                    normalized_attributes,
                    normalized_launch,
                )
            if (kind is KernelKind.VECTOR
                    and normalized_variant in (
                        "elementwise", "elementwise_pytorch_vec4")
                    and "lowered_vector_operations"
                    in normalized_launch.parameters
                    and normalized_attributes["operation"]
                    not in ("add", "mul")):
                raise ValueError(
                    "VECTOR/elementwise lowered_vector_operations requires "
                    "the source-closed add or mul operation")
            physical_algorithm = canonical_physical_algorithm(
                hardware_id,
                kind,
                normalized_variant,
                normalized_launch.algorithm,
            )
            if physical_algorithm != normalized_launch.algorithm:
                normalized_launch = replace(
                    normalized_launch, algorithm=physical_algorithm)
            object.__setattr__(self, "shape", _immutable_mapping(normalized_shape))
            object.__setattr__(
                self, "attributes", _immutable_mapping(normalized_attributes))
            object.__setattr__(self, "launch", normalized_launch)
        elif kind in transfer_kinds:
            if self.execution_support != "supported":
                raise ValueError(
                    f"{kind.value} transfer cannot be declared unsupported")
            if self.transfer is None:
                raise ValueError(f"{kind.value} requires physical transfer configuration")
            if self.launch is not None:
                raise ValueError(f"{kind.value} does not use GPU launch geometry")
            if (self.shape or self.dtype is not None or self.variant is not None
                    or self.attributes):
                raise ValueError(
                    f"{kind.value} accepts only PhysicalTransferConfig fields")
            if kind is KernelKind.COMMUNICATION:
                if self.transfer.tier is not None:
                    raise ValueError("COMMUNICATION does not accept tier")
                if not self.transfer.topology:
                    raise ValueError("COMMUNICATION requires topology")
                if not self.transfer.operation:
                    raise ValueError("COMMUNICATION requires operation")
                if not self.transfer.bytes_semantics:
                    raise ValueError("COMMUNICATION requires bytes_semantics")
                if not self.transfer.backend:
                    raise ValueError("COMMUNICATION requires backend")
                if not self.transfer.resource_id:
                    raise ValueError("COMMUNICATION requires resource_id")
                if self.transfer.bytes <= 0:
                    raise ValueError("COMMUNICATION bytes must be positive")
                if len(self.transfer.participants) < 2:
                    raise ValueError(
                        "COMMUNICATION requires at least two participants")
                operation = self.transfer.operation
                collective_operations = {
                    "all_reduce", "all_gather", "all_to_all"}
                if operation in collective_operations:
                    if self.transfer.direction != "collective":
                        raise ValueError(
                            f"{operation} COMMUNICATION requires collective "
                            "direction")
                    if self.transfer.bytes_semantics != "per_rank_input":
                        raise ValueError(
                            f"{operation} COMMUNICATION requires "
                            "per_rank_input bytes_semantics")
                    if (self.transfer.source_rank is not None
                            or self.transfer.destination_rank is not None):
                        raise ValueError(
                            f"{operation} COMMUNICATION does not accept "
                            "source/destination ranks")
                elif operation == "send_recv":
                    if self.transfer.direction != "point_to_point":
                        raise ValueError(
                            "send_recv COMMUNICATION requires point_to_point "
                            "direction")
                    if self.transfer.bytes_semantics != "payload":
                        raise ValueError(
                            "send_recv COMMUNICATION requires payload "
                            "bytes_semantics")
                    if len(self.transfer.participants) != 2:
                        raise ValueError(
                            "send_recv COMMUNICATION requires two participants")
                    source_rank = self.transfer.source_rank
                    destination_rank = self.transfer.destination_rank
                    if source_rank is None or destination_rank is None:
                        raise ValueError(
                            "send_recv COMMUNICATION requires source and "
                            "destination ranks")
                    if source_rank == destination_rank:
                        raise ValueError(
                            "send_recv COMMUNICATION endpoints must differ")
                    if {source_rank, destination_rank} != set(
                            self.transfer.participants):
                        raise ValueError(
                            "send_recv COMMUNICATION endpoints must exactly "
                            "match participants")
                else:
                    raise ValueError(
                        f"unknown COMMUNICATION operation: {operation!r}")
                if operation in ("all_reduce", "all_gather"):
                    if operation == "all_reduce":
                        collective_layouts = {
                            "fc2": "fully_connected",
                            "fullmesh": "fully_connected",
                            "mesh": "fully_connected",
                            "ring": "ring",
                            "tree": "tree",
                        }
                    elif operation == "all_gather":
                        collective_layouts = {
                            "fc2": "fully_connected",
                            "fullmesh": "fully_connected",
                            "mesh": "fully_connected",
                            "ring": "ring",
                        }
                    else:
                        raise AssertionError(
                            "validated collective operation disappeared")
                    if self.transfer.algorithm not in collective_layouts:
                        raise ValueError(
                            f"{operation} COMMUNICATION currently requires "
                            "the ring, tree, fc2, mesh, or fullmesh algorithm")
                    expected_topology = collective_layouts[
                        self.transfer.algorithm]
                    if self.transfer.topology != expected_topology:
                        raise ValueError(
                            f"{operation} {self.transfer.algorithm} "
                            f"COMMUNICATION requires {expected_topology} "
                            "topology")
                    if self.transfer.algorithm == "ring":
                        if (len(self.transfer.ring_order)
                                != len(self.transfer.participants)
                                or set(self.transfer.ring_order)
                                != set(self.transfer.participants)):
                            raise ValueError(
                                f"{operation} COMMUNICATION ring_order must "
                                "be a permutation of participants")
                    elif self.transfer.algorithm in (
                            "fc2", "mesh", "fullmesh", "tree"):
                        if self.transfer.ring_order:
                            raise ValueError(
                                f"{operation} {self.transfer.algorithm} "
                                "COMMUNICATION does not accept ring_order")
                    else:
                        raise AssertionError(
                            "validated collective algorithm disappeared")
                elif operation == "all_to_all":
                    if self.transfer.algorithm != "pairwise":
                        raise ValueError(
                            "all_to_all COMMUNICATION currently requires the "
                            "pairwise algorithm")
                    if self.transfer.topology != "fully_connected":
                        raise ValueError(
                            "all_to_all COMMUNICATION currently requires a "
                            "fully_connected topology")
                    if self.transfer.ring_order:
                        raise ValueError(
                            "all_to_all COMMUNICATION does not accept ring_order")
                    if self.transfer.bytes % len(
                            self.transfer.participants):
                        raise ValueError(
                            "all_to_all per-rank input bytes must divide the "
                            "participant count")
                elif operation == "send_recv":
                    if self.transfer.algorithm != "send_recv":
                        raise ValueError(
                            "send_recv COMMUNICATION requires send_recv "
                            "algorithm")
                    if self.transfer.topology != "point_to_point":
                        raise ValueError(
                            "send_recv COMMUNICATION requires point_to_point "
                            "topology")
                    if self.transfer.ring_order:
                        raise ValueError(
                            "send_recv COMMUNICATION does not accept ring_order")
                else:
                    raise ValueError(
                        f"unhandled COMMUNICATION operation: {operation!r}")
                if operation == "all_reduce":
                    if self.transfer.reduction != "sum":
                        raise ValueError(
                            "all_reduce COMMUNICATION currently requires sum "
                            "reduction")
                elif self.transfer.reduction is not None:
                    raise ValueError(
                        f"{operation} COMMUNICATION does not accept reduction")
                if self.transfer.backend not in (
                        "analytical_roofline_v1", "protobridge"):
                    raise ValueError(
                        "COMMUNICATION backend must be "
                        "analytical_roofline_v1 or protobridge")
                if self.transfer.backend == "analytical_roofline_v1":
                    if self.transfer.link not in ("nvlink", "infiniband"):
                        raise ValueError(
                            "analytical Roofline COMMUNICATION models only "
                            "NVLink or InfiniBand paths")
                    if (operation in ("all_reduce", "all_gather")
                            and self.transfer.algorithm != "ring"):
                        raise ValueError(
                            "analytical Roofline all_reduce/all_gather "
                            "requires the ring algorithm")
                    if self.transfer.backend_config_sha256 is not None:
                        raise ValueError(
                            "analytical Roofline COMMUNICATION does not accept "
                            "backend_config_sha256")
                elif self.transfer.backend == "protobridge":
                    if self.transfer.link not in (
                            "nvlink", "metaxlink", "hccs", "infiniband"):
                        raise ValueError(
                            "ProtoBridge COMMUNICATION link must be nvlink, "
                            "metaxlink, hccs, or infiniband")
                    if self.transfer.backend_config_sha256 is None:
                        raise ValueError(
                            "ProtoBridge COMMUNICATION requires a versioned "
                            "backend config SHA256")
                else:
                    raise ValueError(
                        f"unhandled COMMUNICATION backend: "
                        f"{self.transfer.backend!r}")
            elif kind is KernelKind.MEMCPY:
                if (self.transfer.topology is not None
                        or self.transfer.tier is not None
                        or self.transfer.participants
                        or self.transfer.operation is not None
                        or self.transfer.bytes_semantics is not None
                        or self.transfer.source_rank is not None
                        or self.transfer.destination_rank is not None
                        or self.transfer.ring_order
                        or self.transfer.reduction is not None
                        or self.transfer.backend is not None
                        or self.transfer.resource_id is not None
                        or self.transfer.backend_config_sha256 is not None):
                    raise ValueError(
                        "MEMCPY does not accept communication-specific fields")
                if self.transfer.algorithm not in (
                        "dma", "native_d2d_flash_attention_copy",
                        "stream_copy", "stream_copy_pinned",
                        "stream_copy_pageable"):
                    raise ValueError(
                        "MEMCPY currently models only DMA, the native "
                        "C500 FlashAttention D2D copy or an eager stream "
                        "copy")
                direction = self.transfer.direction
                if direction in ("d2d", "dtod", "device_to_device"):
                    if self.transfer.link != "dram":
                        raise ValueError("device MEMCPY requires the DRAM link")
                    if self.transfer.algorithm in (
                            "stream_copy_pinned", "stream_copy_pageable"):
                        raise ValueError(
                            "a device-to-device stream copy has no host "
                            "memory kind")
                    if (self.transfer.algorithm
                            == "native_d2d_flash_attention_copy"
                            and self.hardware_id != "metax_c500"):
                        raise ValueError(
                            "native FlashAttention D2D copy requires "
                            "metax_c500")
                elif direction in (
                        "h2d", "d2h", "host_to_device", "device_to_host"):
                    if self.transfer.link != "pcie":
                        raise ValueError("host MEMCPY requires the PCIe link")
                    if self.transfer.algorithm == "stream_copy":
                        raise ValueError(
                            "a host stream copy must name pinned or pageable "
                            "host memory")
                    if self.transfer.algorithm not in (
                            "dma", "stream_copy_pinned",
                            "stream_copy_pageable"):
                        raise ValueError(
                            "native FlashAttention D2D copy requires a "
                            "device-to-device transfer")
                else:
                    raise ValueError(
                        f"unknown MEMCPY direction: {self.transfer.direction!r}")
            elif kind is KernelKind.KV_OFFLOAD:
                if (self.transfer.topology is not None
                        or self.transfer.participants
                        or self.transfer.operation is not None
                        or self.transfer.bytes_semantics is not None
                        or self.transfer.source_rank is not None
                        or self.transfer.destination_rank is not None
                        or self.transfer.ring_order
                        or self.transfer.reduction is not None
                        or self.transfer.backend is not None
                        or self.transfer.resource_id is not None
                        or self.transfer.backend_config_sha256 is not None):
                    raise ValueError(
                        "KV_OFFLOAD does not accept communication-specific "
                        "fields")
                if not self.transfer.tier:
                    raise ValueError("KV_OFFLOAD requires tier")
                if self.transfer.algorithm != "dma":
                    raise ValueError("KV_OFFLOAD currently models only DMA")
                if self.transfer.direction not in ("load", "evict"):
                    raise ValueError(
                        "KV_OFFLOAD direction must be 'load' or 'evict'")
                if self.transfer.link not in ("pcie", "nvlink"):
                    raise ValueError(
                        "KV_OFFLOAD link must be 'pcie' or 'nvlink'")
            else:
                raise AssertionError(
                    f"unhandled validated transfer kind: {kind.value}")
        else:
            raise ValueError(f"unhandled kernel kind: {kind.value}")


@dataclass(frozen=True)
class KernelPrediction:
    """Predicted latency and physical breakdown for one complete KernelSpec."""

    t_total_s: float
    t_per_group: Mapping[str, float]
    t_launch_s: float
    bottleneck_group: str
    t_per_resource: Mapping[str, float]
    diagnostics: Optional[PredictionDiagnostics] = None

    def __post_init__(self) -> None:
        if not isinstance(self.t_per_group, Mapping):
            raise TypeError("t_per_group must be a mapping")
        if not isinstance(self.t_per_resource, Mapping):
            raise TypeError("t_per_resource must be a mapping")
        scalar_times = {
            "t_total_s": self.t_total_s,
            "t_launch_s": self.t_launch_s,
        }
        invalid_scalars = {
            name: value for name, value in scalar_times.items()
            if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0)
        }
        if invalid_scalars:
            raise ValueError(
                f"prediction times must be finite and non-negative: "
                f"{invalid_scalars}")
        invalid_breakdown_keys = [
            name
            for mapping in (self.t_per_group, self.t_per_resource)
            for name in mapping
            if not isinstance(name, str) or not name
        ]
        if invalid_breakdown_keys:
            raise TypeError(
                "prediction breakdown keys must be non-empty strings: "
                f"{invalid_breakdown_keys}")
        invalid_breakdown = {
            name: value
            for mapping in (self.t_per_group, self.t_per_resource)
            for name, value in mapping.items()
            if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0)
        }
        if invalid_breakdown:
            raise ValueError(
                "prediction breakdowns must be finite and non-negative: "
                f"{invalid_breakdown}")
        if not isinstance(self.bottleneck_group, str) or not self.bottleneck_group:
            raise ValueError("prediction bottleneck_group must be non-empty")
        if (self.diagnostics is not None
                and not isinstance(self.diagnostics, PredictionDiagnostics)):
            raise TypeError(
                "prediction diagnostics must be PredictionDiagnostics or None")
        object.__setattr__(self, "t_per_group", _immutable_mapping(self.t_per_group))
        object.__setattr__(self, "t_per_resource", _immutable_mapping(self.t_per_resource))

    @property
    def t_total_ns(self) -> float:
        return self.t_total_s * 1e9
