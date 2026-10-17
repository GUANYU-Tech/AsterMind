"""Normalize profiler observations into complete production ``KernelSpec`` objects.

This is the only single-kernel path allowed to inspect runtime implementation
names.  The resulting spec contains semantic variants and structured launch
facts, never the name itself or measured counters/duration.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import re
from typing import Mapping, Optional

from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    DTYPE_BYTES,
    L2ExecutionContext,
    MOE_CAST_ROLES,
    MOE_FILL_ROLES,
    PAGED_KV_CACHE_ATTRIBUTES,
)
from src.kernel.launch_config import GpuCtaRasterProgram, TilingSpec
from src.kernel.launch_config.ascend_910b.gemm_launch import (
    launch_from_cann_gemm_observation,
)
from src.kernel.launch_config.nvidia_a800.gemm_launch import (
    cublaslt_output_dtype_reducer_source_parameters,
    observed_gemm_cta_raster,
)
from src.kernel.launch_config.tiling_spec import (
    METAX_MLA_SCHEDULER_PARTITIONS,
    derive_mla_scheduler_work,
)
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.launch_config.nvidia_h200.flash_attn_launch import (
    FA3_COMBINE_ALGORITHM,
    FA3_DECODE_ALGORITHM,
)
from src.kernel.predictors.flash_attn import (
    FA3_UNMODELED_OBSERVATION_ALGORITHM,
    fa3_decode_dynamic_splits,
)
from src.kernel.launch_config.select_launch_config import (
    select_fused_moe_reduce_work_fractions,
)
from src.kernel.launch_config.tiling_spec import (
    find_gemm_swizzle_convention,
    parse_flash_attention_trait,
    parse_nvjet_gemm_symbol,
)
from evaluation.kernel.observation_parser import (
    is_non_tensor,
    observed_launch_config,
)
from evaluation.kernel.profiler_observation import extract_ncu_metrics
from src.kernel.model_contract import (
    SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER,
    SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
)

from evaluation.json_shape import require_positive_metric as _positive_int


# The vector variants whose Ascend observation is bound to the retained CANN
# source program.  A800 binds every vector observation and the MetaX backends
# bind the variants their typed flags mark; this constant holds only the
# Ascend side, which is a fixed set rather than a per-observation fact.
_ASCEND_SOURCE_BOUND_VECTOR_VARIANTS = frozenset({
    "rmsnorm", "add_rmsnorm", "activation", "silu", "rotary", "rope",
    "paged_kv_cache", "elementwise", "qkv_rmsnorm_rope",
})


def _is_source_bound_vector_observation(
        hardware_id: str, variant: str, *, c500_typed) -> bool:
    """Whether this vector observation is bound to a retained source program.

    A800 binds every observation; the MetaX backends bind the variants their
    typed flags mark (their symbols were checked against the retained
    artifact above); Ascend binds the fixed CANN program set.
    """

    if hardware_id == "nvidia_a800":
        return True
    if any(c500_typed):
        return True
    return (hardware_id == "ascend_910b"
            and variant in _ASCEND_SOURCE_BOUND_VECTOR_VARIANTS)


# The one hardware each source-derived MoE auxiliary handler normalizes.
# A variant whose handler is hardware-specific names its hardware here and the
# message its mismatch raises; the handler body is unchanged.  ``{variant}``
# and ``{hardware_id}`` are filled at the raise site.
_MOE_AUX_HARDWARE = {
    "moe_reduce": (
        "nvidia_a800",
        "source-derived moe_reduce observation normalization is "
        "defined only for nvidia_a800, got {hardware_id!r}"),
    "moe_swiglu": (
        "ascend_910b",
        "source-derived moe_swiglu normalization is defined only "
        "for ascend_910b, got {hardware_id!r}"),
    "moe_compute_expert_tokens": (
        "ascend_910b",
        "source-derived moe_compute_expert_tokens normalization is "
        "defined only for ascend_910b, got {hardware_id!r}"),
    "moe_cast": (
        "ascend_910b",
        "typed moe_cast normalization is defined only for "
        "ascend_910b, got {hardware_id!r}"),
    "moe_init_routing": (
        "ascend_910b",
        "source-derived {variant} normalization is defined only "
        "for ascend_910b, got {hardware_id!r}"),
    "moe_finalize_routing": (
        "ascend_910b",
        "source-derived {variant} normalization is defined only "
        "for ascend_910b, got {hardware_id!r}"),
    "moe_transpose": (
        "ascend_910b",
        "source-derived {variant} normalization is defined only "
        "for ascend_910b, got {hardware_id!r}"),
    "moe_range": (
        "ascend_910b",
        "source-derived {variant} normalization is defined only "
        "for ascend_910b, got {hardware_id!r}"),
    "moe_silu": (
        "metax_c500",
        "source-derived {variant} work partition is defined only "
        "for metax_c500, got {hardware_id!r}"),
    "moe_mul": (
        "metax_c500",
        "source-derived {variant} work partition is defined only "
        "for metax_c500, got {hardware_id!r}"),
}


def require_moe_aux_hardware(variant: str, hardware_id: str) -> None:
    """Fail loud when a hardware-specific MoE auxiliary sees another hardware."""

    if variant not in _MOE_AUX_HARDWARE:
        return
    expected, message = _MOE_AUX_HARDWARE[variant]
    if hardware_id != expected:
        raise ValueError(message.format(variant=variant, hardware_id=hardware_id))


# The runtime symbol each backend's fused SiLU (activation) kernel publishes.
# A hardware absent from this map has no bound SILU symbol, so its observation
# is not checked against one.
_SILU_RUNTIME_SYMBOLS = {
    "nvidia_a800": (
        "void vllm::act_and_mul_kernel<c10::BFloat16, "
        "&vllm::silu_kernel<c10::BFloat16>, (bool)1>"
        "(T1 *, const T1 *, int)"
    ),
    "metax_c500": (
        "void vllm::act_and_mul_kernel_bd_opt<c10::BFloat16, "
        "float4, 8, &(c10::BFloat16 vllm::silu_kernel<"
        "c10::BFloat16>(c10::BFloat16 const&)), true>"
        "(c10::BFloat16*, c10::BFloat16 const*, int, int)"
    ),
    "ascend_910b": "SwiGlu_3_high_performance_27",
}


_ASCEND_ELEMENTWISE_RUNTIME_SYMBOLS = {
        "add": (
            "Add_ce3d1ff8ba23a0bcb292da3577c1625d_"
            "high_performance_210000000",
            "Add_ce3d1ff8ba23a0bcb292da3577c1625d_"
            "high_performance_210010000",
        ),
        "mul": (
            "Mul_ce3d1ff8ba23a0bcb292da3577c1625d_"
            "high_performance_210020000",
        ),
    }


_METAX_VECTOR_RUNTIME_SYMBOLS = {
        "rmsnorm": (
            "void vllm::rms_norm_kernel<c10::BFloat16>"
            "(c10::BFloat16*, c10::BFloat16 const*, "
            "c10::BFloat16 const*, float, int, int)"),
        "add_rmsnorm": (
            "std::enable_if<((8)==(0))||"
            "(!_typeConvert<c10::BFloat16>::exists), void>::type "
            "vllm::fused_add_rms_norm_kernel<c10::BFloat16, 8>"
            "(c10::BFloat16*, c10::BFloat16*, "
            "c10::BFloat16 const*, float, int, int)"),
        "paged_kv_cache": (
            "void vllm::reshape_and_cache_flash_kernel<"
            "__maca_bfloat16, __maca_bfloat16, "
            "(vllm::Fp8KVCacheDataType)0>"
            "(__maca_bfloat16 const*, __maca_bfloat16 const*, "
            "__maca_bfloat16*, __maca_bfloat16*, long const*, "
            "long, long, long, long, long, int, int, int, "
            "float const*, float const*)"),
    }


# The typed PyTorch vec4 elementwise symbol the A800 capture publishes, keyed
# by the Attr operation.
_A800_ELEMENTWISE_RUNTIME_SYMBOLS = {
    "add": (
        "void at::native::vectorized_elementwise_kernel<(int)4, "
        "at::native::CUDAFunctor_add<c10::BFloat16>, "
        "std::array<char *, (unsigned long)3>>(int, T2, T3)"
    ),
    "mul": (
        "void at::native::vectorized_elementwise_kernel<(int)4, "
        "at::native::BinaryFunctor<c10::BFloat16, "
        "c10::BFloat16, c10::BFloat16, "
        "at::native::binary_internal::MulFunctor<float>>, "
        "std::array<char *, (unsigned long)3>>(int, T2, T3)"
    ),
}


# The vLLM 0.11.0 runtime symbol each A800 Vector kernel publishes, keyed by
# the canonical variant.  A variant absent here has no retained symbol, so its
# observation is not bound to one.
_A800_VECTOR_RUNTIME_SYMBOLS = {
    "rmsnorm": (
        "void vllm::rms_norm_kernel<c10::BFloat16>"
        "(T1 *, const T1 *, long, const T1 *, float, int, int)"),
    "add_rmsnorm": (
        "std::enable_if<T2>(int)0&&vllm::_typeConvert<T1>::exists, "
        "void>::type vllm::fused_add_rms_norm_kernel<"
        "c10::BFloat16, (int)8>(T1 *, long, T1 *, const T1 *, "
        "float, int, int)"),
    "rope": (
        "void vllm::rotary_embedding_kernel<c10::BFloat16, "
        "(bool)1>(const long *, T1 *, T1 *, const T1 *, int, "
        "long, long, long, int, int, int)"),
    "silu": (
        "void vllm::act_and_mul_kernel<c10::BFloat16, "
        "&vllm::silu_kernel<c10::BFloat16>, (bool)1>"
        "(T1 *, const T1 *, int)"),
    "paged_kv_cache": (
        "void vllm::reshape_and_cache_flash_kernel<"
        "__nv_bfloat16, __nv_bfloat16, "
        "(vllm::Fp8KVCacheDataType)0>(const T1 *, const T1 *, "
        "T2 *, T2 *, const long *, long, long, long, long, long, "
        "int, int, int, const float *, const float *)"),
}


_METAX_372_STATIC_FLASH_TRAITS = frozenset({
    (64, 16, 16, 1),
    (64, 64, 64, 4),
    (128, 16, 16, 1),
    (128, 64, 64, 4),
    (128, 128, 64, 4),
})
_A800_CUBLASLT_OUTPUT_DTYPE_REDUCER = (
    "void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, __half, "
    "__half, float, __half, (bool)0, __half, __half, __half, (bool)1, "
    "(bool)0, (bool)0>(cublasLt::cublasSplitKParams<T6>, const T4 *, "
    "const T10 *, T9 *, T5 *, const T6 *, const T6 *, const T11 *, "
    "const T4 *, T11 *, void *, long, T6 *, int *, T6 *, T6 *, const T6 "
    "*, const T6 *, const T6 *, const T6 *, const T6 *)"
)
_METAX_372_Q64_PREFILL_RUNTIME = (
    "void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 64, 64, "
    "4, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, "
    "64, 64, 4, mctlass::bfloat16_t> >, true, false, false, false, true, "
    "false, false, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, "
    "int, int)")
_METAX_372_D64_Q64_PREFILL_RUNTIME = (
    "void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<64, 64, 64, "
    "4, true, true, mctlass::bfloat16_t, 64, Flash_kernel_traits<64, "
    "64, 64, 4, mctlass::bfloat16_t> >, true, false, false, false, true, "
    "false, false, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, "
    "int, int)")
_METAX_372_Q64_PREFILL_ALGORITHM = (
    "observed_metax_flash_attention_trait_direct:prefill_q64_full_tile")
_METAX_372_Q64_PACKED_PREFILL_ALGORITHM = (
    "observed_metax_flash_attention_trait_packed:prefill_q64_full_tile")
_METAX_372_Q64_DECODE_RUNTIME = (
    "void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, "
    "true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, "
    "64, 64, 4, mctlass::bfloat16_t> >, false, false, false, false, "
    "false, false, true, false, false, 1, false, (Arch)1000>"
    "(mcFlashAttn::Flash_fwd_params, int, int)")
_METAX_372_Q64_DECODE_ALGORITHM = (
    "observed_metax_flash_attention_trait_direct:decode_q64_full_tile")
_METAX_372_Q128_PREFILL_RUNTIME = (
    "void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, "
    "true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, "
    "128, 64, 4, mctlass::bfloat16_t> >, false, true, false, false, "
    "false, false, true, false, false, 1, false, (Arch)1000>"
    "(mcFlashAttn::Flash_fwd_params, int, int)")
_METAX_372_Q128_SPLIT_PREFILL_RUNTIME = (
    "void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<128, 128, 64, "
    "4, true, true, mctlass::bfloat16_t, 128, Flash_kernel_traits<128, "
    "128, 64, 4, mctlass::bfloat16_t> >, true, false, false, false, true, "
    "false, false, false, true, (Arch)1000>(mcFlashAttn::Flash_fwd_params, "
    "int, int)")
_METAX_372_Q128_PREFILL_ALGORITHM = (
    "observed_metax_flash_attention_trait_direct:prefill_q128_full_tile")
_METAX_372_Q128_SPLIT_PREFILL_ALGORITHM = (
    "observed_metax_flash_attention_trait_direct:"
    "prefill_q128_split_full_tile")

_ASCEND_910B_MOE_COMPUTE_EXPERT_TOKENS_CODE_HASH = (
    "d1c627adc45619675efeff94a5330a80")
_ASCEND_910B_MOE_CAST_RUNTIME_BY_ROLE = {
    "topk_ids": (
        "Cast_9f288b80370c6545ffe8cef142d37f5c_"
        "high_performance_210000000"),
    "expert_tokens": (
        "Cast_e87590d11ccda8b259ab6b1ea7212319_"
        "high_performance_210000000"),
}
_A800_PYTORCH_2_8_BF16_REDUCE_SYMBOL = (
    "void at::native::reduce_kernel<(int)128, (int)4, "
    "at::native::ReduceOp<c10::BFloat16, "
    "at::native::func_wrapper_t<c10::BFloat16, "
    "at::native::sum_functor<c10::BFloat16, float, "
    "c10::BFloat16>::operator ()(at::TensorIterator &)::"
    "[lambda(float, float) (instance 1)]>, unsigned int, "
    "c10::BFloat16, (int)4, (int)4>>(T3)")


def _metax_372_static_flash_trait(runtime_name: str, metrics: Mapping,
                                  shape: Mapping[str, int], hardware):
    """Return a version-bound C500 ``(D, BlockM, BlockN, warps)`` trait.

    The MACA 3.7.2 flash-attn 2.6.3 candidate registry names these exact five
    BF16 tuples as ``hdim_qk/hdim_v/block_m/block_n/k_nwarps``.  Runtime
    dispatch, tail arithmetic and CTA phase order are separate facts and are
    deliberately not inferred here.
    """

    trait = parse_flash_attention_trait(runtime_name)
    if trait is None or "flash_fwd_splitkv_mla_kernel" in runtime_name.lower():
        return None
    if hardware.hardware_id != "metax_c500":
        raise ValueError(
            "MetaX FlashAttention trait metadata requires metax_c500")
    if trait not in _METAX_372_STATIC_FLASH_TRAITS:
        raise ValueError(
            "MetaX MACA 3.7.2 FlashAttention runtime exposed an unregistered "
            f"static trait: {trait}")
    head_dim, block_m, block_n, warps = trait
    if head_dim != shape["head_dim"]:
        raise ValueError(
            "MetaX FlashAttention trait head dimension disagrees with the "
            f"workload: trait={head_dim}, shape={shape['head_dim']}")
    expected_block_size = warps * 64
    if metrics["block_size"] != expected_block_size:
        raise ValueError(
            "MetaX FlashAttention trait warp count disagrees with the "
            f"observed block: trait={warps}, block={metrics['block_size']}")
    return head_dim, block_m, block_n, warps




def _optional_positive_int(mapping: Mapping, name: str, context: str):
    value = mapping[name]
    if value is None:
        return None
    return _positive_int(mapping, name, context)


def _non_empty_string(mapping: Mapping, name: str, context: str) -> str:
    value = mapping[name]
    if not isinstance(value, str):
        raise TypeError(
            f"{context}.{name} must be a string, got {type(value).__name__}"
        )
    if not value:
        raise ValueError(f"{context}.{name} must not be empty")
    return value


def _finite_number(mapping: Mapping, name: str, context: str):
    value = mapping[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{context}.{name} must be numeric, got {type(value).__name__}"
        )
    if not math.isfinite(value):
        raise ValueError(f"{context}.{name} must be finite, got {value!r}")
    return value


def _positive_int_sequence(mapping: Mapping, name: str, context: str,
                           expected_length: int, expected_maximum: int):
    value = mapping[name]
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{context}.{name} must be a list or tuple")
    result = tuple(value)
    if len(result) != expected_length:
        raise ValueError(
            f"{context}.{name} must contain {expected_length} values, got "
            f"{len(result)}")
    if any(isinstance(item, bool) or not isinstance(item, int)
           for item in result):
        raise TypeError(f"{context}.{name} must contain integers")
    if any(item <= 0 for item in result):
        raise ValueError(f"{context}.{name} must contain positive values")
    if max(result) != expected_maximum:
        raise ValueError(
            f"{context}.{name} maximum must equal {expected_maximum}, got "
            f"{max(result)}")
    return result


def validate_kernel_observation_sequence(
        kernel_records, hardware_id: str, *, op_info: Optional[Mapping] = None,
        hardware=None):
    """Reject per-invocation observations with ambiguous counter provenance.

    MetaX mcTracer and mcProfiler execute the runner separately.  A historical
    name-wise trace union could therefore attach one numbered layer-1 payload
    to two mutually exclusive kernel variants.  Two genuinely distinct kernel
    identities cannot share the entire nested per-invocation counter payload;
    seeing that exact duplication is evidence of the known invalid join, not a
    second kernel execution.
    """

    if not isinstance(hardware_id, str) or not hardware_id:
        raise TypeError("observation hardware_id must be a non-empty string")
    if not isinstance(kernel_records, (list, tuple)) or not kernel_records:
        raise TypeError(
            "profiler case must contain a non-empty kernel-record sequence")
    if (op_info is None) != (hardware is None):
        raise ValueError(
            "observation sequence validation requires op_info and hardware "
            "together")
    if hardware_id == "nvidia_a800" and op_info is not None:
        _validate_nvidia_a800_moe_reduce_sequence(
            op_info, kernel_records, hardware)
        return
    if hardware_id != "metax_c500":
        return

    observations = []
    for ordinal, record in enumerate(kernel_records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"MetaX kernel record {ordinal} must be a mapping")
        payload = record["MetaX"]
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"MetaX kernel record {ordinal} payload must be a mapping")
        name = extract_ncu_metrics(record)["kernel_name"]
        for previous_ordinal, previous_name, previous_payload in observations:
            if name != previous_name and payload == previous_payload:
                raise ValueError(
                    "distinct MetaX kernel identities share an identical "
                    "per-invocation counter payload; trace/layer-1 provenance "
                    "is ambiguous: "
                    f"ordinals=({previous_ordinal}, {ordinal}), "
                    f"kernels=({previous_name!r}, {name!r})")
        observations.append((ordinal, name, payload))


def _validate_nvidia_a800_moe_reduce_sequence(
        op_info: Mapping, kernel_records, hardware) -> None:
    """Validate the exact source-derived reduction suffix and partition."""

    op_type = _non_empty_string(op_info, "Type", "op_info").upper()
    if op_type != "FUSEDMOE":
        return
    attributes = op_info["Attr"]
    shape = {
        "tokens": _positive_int(attributes, "TOK", "FUSED_MOE Attr"),
        "experts": _positive_int(attributes, "E", "FUSED_MOE Attr"),
        "topk": _positive_int(attributes, "TOPK", "FUSED_MOE Attr"),
        "hidden": _positive_int(attributes, "H", "FUSED_MOE Attr"),
        "intermediate": _positive_int(
            attributes, "INT", "FUSED_MOE Attr"),
    }
    runtime_names = [
        extract_ncu_metrics(record)["kernel_name"]
        for record in kernel_records
    ]
    topk = shape["topk"]
    if shape["topk"] in (2, 3, 4):
        expected_symbols = (_a800_moe_sum_symbol(topk),)
        expected_launch = select_launch_config(
            "nvidia_a800", KernelKind.VECTOR, shape,
            hardware.kernel_dtype, "moe_sum",
            {"work_fraction": 1.0}, hardware)
        expected_launches = ((expected_launch.grid, expected_launch.block),)
    else:
        if hardware.kernel_dtype != "bf16":
            raise NotImplementedError(
                "A800 PyTorch 2.8 moe_reduce symbol identity is only bound "
                "for the retained BF16 binary")
        fractions = select_fused_moe_reduce_work_fractions(
            "nvidia_a800", shape, hardware)
        expected_symbols = tuple(
            _A800_PYTORCH_2_8_BF16_REDUCE_SYMBOL for _ in fractions)
        expected_launches = []
        for fraction in fractions:
            launch = select_launch_config(
                "nvidia_a800", KernelKind.VECTOR, shape,
                hardware.kernel_dtype, "moe_reduce",
                {"work_fraction": fraction}, hardware)
            expected_launches.append((launch.grid, launch.block))
        expected_launches = tuple(expected_launches)
    if len(runtime_names) < len(expected_symbols):
        raise ValueError(
            "A800 FUSEDMOE reduction suffix is incomplete: "
            f"expected={len(expected_symbols)}, actual_records={len(runtime_names)}")
    reduction_like = tuple(
        index for index, name in enumerate(runtime_names)
        if "reduce_kernel" in name.lower() or "moe_sum" in name.lower()
    )
    suffix_start = len(runtime_names) - len(expected_symbols)
    if (tuple(runtime_names[suffix_start:]) != expected_symbols
            or reduction_like != tuple(range(suffix_start, len(runtime_names)))):
        raise ValueError(
            "A800 FUSEDMOE reduction suffix disagrees with the exact "
            "vLLM 0.11.0/PyTorch 2.8 source recipe")
    observed_launches = [
        (
            (metrics["grid_x"], metrics["grid_y"], metrics["grid_z"]),
            (metrics["block_x"], metrics["block_y"], metrics["block_z"]),
        )
        for metrics in (
            extract_ncu_metrics(record)
            for record in kernel_records[suffix_start:]
        )
    ]
    if tuple(observed_launches) != expected_launches:
        raise ValueError(
            "A800 FUSEDMOE reduction physical sequence disagrees with the "
            "source-derived recipe/32-bit indexing partition: "
            f"expected={expected_launches}, observed={observed_launches}")


def _a800_moe_sum_symbol(topk: int) -> str:
    if topk not in (2, 3, 4):
        raise ValueError("A800 specialized moe_sum requires top-k 2, 3, or 4")
    return (
        "void vllm::moe::moe_sum_kernel<c10::BFloat16, "
        f"(int){topk}>(T1 *, const T1 *, int)")


def observed_flash_attention_splits(
        op_attributes: Mapping, kernel_records, hardware_id: str) -> int:
    """Derive the executed FA split count from one profiler case.

    Canonical ``SP`` is the executed count. NVIDIA exposes it in the split main
    kernel grid, while MetaX records the public API request or FlashMLA metadata
    because its grid axes are implementation-specific.  H200 FA3 runs a
    persistent varlen scheduler whose grid carries no split axis; its ``SP`` is
    the largest per-request ``num_splits_dynamic`` of the recorded FA3
    scheduler metadata (``SP_PER_REQUEST``), and ``FA3_SPLIT_COMBINE`` (static
    split bound > 1) must match the presence of the combine launch.
    """

    recorded_splits = _positive_int(
        op_attributes, "SP", "FLASH_ATTN Attr")
    batch = _positive_int(op_attributes, "B", "FLASH_ATTN Attr")
    kv_heads = _positive_int(op_attributes, "HKV", "FLASH_ATTN Attr")
    query_length = _positive_int(
        op_attributes, "Q_LEN", "FLASH_ATTN Attr")
    kv_layout = _non_empty_string(
        op_attributes, "kv_layout", "FLASH_ATTN Attr")
    if not isinstance(kernel_records, (list, tuple)) or not kernel_records:
        raise TypeError(
            "FLASH_ATTN profiler case must contain a non-empty kernel sequence")
    if hardware_id == "nvidia_h200":
        return _observed_h200_fa3_splits(
            op_attributes, kernel_records, recorded_splits, batch)
    if query_length > 1:
        if recorded_splits != 1:
            raise ValueError("FLASH_ATTN prefill observation requires SP=1")
        return 1
    if hardware_id == "metax_c500":
        # MetaX uses different grid axes across implementations. The C500
        # collector records the scalar passed to the public API, or for MLA the
        # actual device metadata value, so the grid is not a second SP source.
        return recorded_splits
    if hardware_id != "nvidia_a800":
        raise ValueError(
            "observed GPU FlashAttention split validation supports only "
            f"nvidia_a800 and metax_c500, got {hardware_id!r}")
    candidates = set()
    stacked_kv_heads = batch * kv_heads
    for record in kernel_records:
        metrics = extract_ncu_metrics(record)
        lowered = metrics["kernel_name"].lower()
        if metrics["op_type"] is not None:
            raise ValueError(
                "NVIDIA A800 FLASH_ATTN record cannot contain Ascend Op Type")
        if "flash_fwd_splitkv_kernel" in lowered:
            grid_y = metrics["grid_y"]
            grid_z = metrics["grid_z"]
            if grid_y == batch and grid_z == kv_heads:
                candidates.add(1)
            elif grid_z == stacked_kv_heads:
                candidates.add(grid_y)
            elif grid_y == stacked_kv_heads:
                candidates.add(grid_z)
            elif metrics["grid_x"] == stacked_kv_heads and grid_z == 1:
                candidates.add(grid_y)
            else:
                raise ValueError(
                    "FLASH_ATTN split-kv grid does not contain B*HKV: "
                    f"grid=({metrics['grid_x']}, {grid_y}, {grid_z}), "
                    f"B*HKV={stacked_kv_heads}")
        elif "flash_fwd_kernel" in lowered and "splitkv" not in lowered:
            candidates.add(1)
        elif (kv_layout == "mla"
              and lowered == "_fwd_grouped_kernel_stage1"):
            # vLLM Triton MLA launches (batch, query-head groups, KV splits).
            # grid_y=ceil(HQ/BLOCK_H); only grid_z represents executed SP.
            candidates.add(metrics["grid_z"])

    if len(candidates) > 1:
        raise ValueError(
            "FLASH_ATTN profiler case contains conflicting observed split "
            f"counts: {sorted(candidates)}")
    if candidates:
        observed_splits = next(iter(candidates))
        if observed_splits <= 0:
            raise ValueError(
                "FLASH_ATTN observed split count must be positive")
        if observed_splits != recorded_splits:
            raise ValueError(
                "FLASH_ATTN recorded SP disagrees with profiler launch: "
                f"recorded={recorded_splits}, observed={observed_splits}")
        return recorded_splits
    raise ValueError(
        "FLASH_ATTN GPU profiler case has no recognized main-kernel launch")


def _observed_h200_fa3_splits(
        op_attributes, kernel_records, recorded_splits, batch) -> int:
    per_request = op_attributes["SP_PER_REQUEST"]
    if (not isinstance(per_request, list) or len(per_request) != batch
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in per_request)):
        raise ValueError(
            "H200 FLASH_ATTN Attr.SP_PER_REQUEST must hold one positive "
            f"split count per request: {per_request!r}")
    if max(per_request) != recorded_splits:
        raise ValueError(
            "H200 FLASH_ATTN SP must be the largest per-request split: "
            f"SP={recorded_splits}, per_request_max={max(per_request)}")
    combine_expected = op_attributes["FA3_SPLIT_COMBINE"]
    if not isinstance(combine_expected, bool):
        raise TypeError("H200 FLASH_ATTN Attr.FA3_SPLIT_COMBINE must be bool")
    names = [extract_ncu_metrics(record)["kernel_name"]
             for record in kernel_records]
    if not any("FlashAttnFwdSm90" in name for name in names):
        raise ValueError(
            "H200 FLASH_ATTN case has no FA3 FlashAttnFwdSm90 main launch")
    combine_launched = any("FlashAttnFwdCombine" in name for name in names)
    if combine_launched != combine_expected:
        raise ValueError(
            "H200 FLASH_ATTN combine launch disagrees with the FA3 scheduler "
            f"metadata: launched={combine_launched}, "
            f"FA3_SPLIT_COMBINE={combine_expected}")
    if recorded_splits > 1 and not combine_launched:
        raise ValueError("H200 FLASH_ATTN SP > 1 requires a combine launch")
    return recorded_splits


def observed_nvjet_gemm_split_k(
        op_attributes: Mapping, metrics: Mapping) -> int:
    """Return the K-split count of one cuBLASLt ``nvjet`` launch.

    Hopper ``nvjet`` kernels keep ``grid.z == 1`` and launch the output tile
    grid in thread-block clusters; the symbol states the tile, the cluster
    shape and whether the program is split-K (``_splitK``).  A split launch
    enumerates every cluster-padded output tile once per K slice, so the
    slice count is the launched CTA count over the padded tile count.  A
    non-split launch is either one CTA per padded tile or a persistent grid
    with fewer CTAs than tiles.  Only launch geometry and the symbol enter
    this derivation; no duration or counter is read.
    """

    symbol = parse_nvjet_gemm_symbol(metrics["kernel_name"])
    if symbol is None:
        raise ValueError("nvjet split-K derivation requires an nvjet symbol")
    if not isinstance(op_attributes, Mapping):
        raise ValueError(
            "nvjet split-K derivation requires the GEMM Attr shape")
    rows = (_positive_int(op_attributes, "B", "GEMM Attr")
            * _positive_int(op_attributes, "M", "GEMM Attr"))
    columns = _positive_int(op_attributes, "N", "GEMM Attr")
    n_tiles = math.ceil(columns / symbol.tile_n)
    m_tiles = math.ceil(rows / symbol.tile_m)
    padded_tiles = (
        math.ceil(n_tiles / symbol.cluster_n_tiles) * symbol.cluster_n_tiles
        * math.ceil(m_tiles / symbol.cluster_m_tiles)
        * symbol.cluster_m_tiles)
    if metrics["grid_z"] != 1:
        raise ValueError("nvjet GEMM launch must keep grid.z == 1")
    grid_size = metrics["grid_x"] * metrics["grid_y"]
    if symbol.split_k:
        splits, remainder = divmod(grid_size, padded_tiles)
        if remainder or splits < 2:
            raise NotImplementedError(
                "nvjet split-K launch grid is not an integer multiple (>= 2) "
                "of its cluster-padded output tiles: "
                f"grid={grid_size}, padded_tiles={padded_tiles}")
        return splits
    if grid_size > padded_tiles:
        raise ValueError(
            "non-split nvjet launch has more CTAs than padded output tiles: "
            f"grid={grid_size}, padded_tiles={padded_tiles}")
    return 1


def _gemm_tensor_split(metrics: Mapping, op_attributes) -> int:
    if parse_nvjet_gemm_symbol(metrics["kernel_name"]) is not None:
        return observed_nvjet_gemm_split_k(op_attributes, metrics)
    return metrics["grid_z"]


def observed_gemm_split_k(kernel_records, op_attributes=None) -> int:
    """Return the split count carried by a case's tensor GEMM launch.

    A split-K reducer's own grid does not encode how many partial matrices it
    consumes.  The immediately preceding tensor launch does, so evaluation
    normalizes that launch fact into the same production ``split_k`` contract
    used by Serving.  No duration or counter is consumed.  Hopper ``nvjet``
    launches state the split through their symbol and cluster-padded grid
    (``observed_nvjet_gemm_split_k``) and require ``op_attributes``.
    """

    if not isinstance(kernel_records, (list, tuple)) or not kernel_records:
        raise TypeError(
            "GEMM split-K derivation requires a non-empty kernel sequence")
    tensor_splits = []
    for ordinal, record in enumerate(kernel_records):
        if not isinstance(record, Mapping):
            raise TypeError(
                f"GEMM kernel record {ordinal} must be a mapping")
        metrics = extract_ncu_metrics(record)
        lowered = metrics["kernel_name"].lower()
        non_tensor = is_non_tensor(
            "GEMM", metrics["kernel_name"],
            ascend_op_type=metrics["op_type"])
        # MetaX GEMV-family kernels are implemented on ordinary vector lanes,
        # so the observation classifier correctly reports them as non-tensor.
        # They are nevertheless the case's primary GEMM launch (and are
        # normalized to KernelKind.GEMM below), not an auxiliary reducer.
        if not non_tensor or "gemv" in lowered:
            tensor_splits.append(_gemm_tensor_split(metrics, op_attributes))
    if not tensor_splits:
        raise ValueError("GEMM profiler case has no tensor launch")
    unique = set(tensor_splits)
    if len(unique) != 1:
        raise ValueError(
            "GEMM tensor launches disagree on split_k: "
            f"{sorted(unique)}")
    return tensor_splits[0]


def observed_gemm_split_k_partial_bytes(
        kernel_records, op_attributes=None) -> int:
    """Return a source-visible split-K partial width, or zero if unknown.

    ``grid_z`` proves that the tensor launch is split, but not whether the
    implementation uses a materialized partial matrix.  A directly following
    cuBLAS ``splitKreduce`` exposes its partial type as the fourth runtime
    template argument; mcBLAS ``DevDstElementwiseAdd`` exposes an FP32 partial
    pointer.  No timing or profiler counter participates in this derivation.
    """

    split_k = observed_gemm_split_k(kernel_records, op_attributes)
    if split_k == 1:
        return 0
    tensor_ordinals = []
    reducer_ordinals = []
    reducer_names = []
    for ordinal, record in enumerate(kernel_records):
        metrics = extract_ncu_metrics(record)
        name = metrics["kernel_name"]
        lowered = name.lower()
        non_tensor = is_non_tensor(
            "GEMM", name, ascend_op_type=metrics["op_type"])
        if ((not non_tensor or "gemv" in lowered)
                and _gemm_tensor_split(metrics, op_attributes) > 1):
            tensor_ordinals.append(ordinal)
        if "splitkreduce" in lowered or "elementwiseadd" in lowered:
            reducer_ordinals.append(ordinal)
            reducer_names.append(name)
    if not reducer_ordinals:
        return 0
    if len(tensor_ordinals) != 1 or len(reducer_ordinals) != 1:
        raise ValueError(
            "GEMM split-K partial ABI requires one tensor main followed by "
            "one reducer")
    if reducer_ordinals[0] != tensor_ordinals[0] + 1:
        raise ValueError(
            "GEMM split-K reducer must immediately follow its tensor main")
    reducer_name = reducer_names[0]
    lowered_reducer = reducer_name.lower()
    if "splitkreduce" in lowered_reducer:
        marker = "splitKreduce_kernel<"
        if marker not in reducer_name or ">(" not in reducer_name:
            raise ValueError(
                "cuBLAS split-K reducer name does not expose its template ABI")
        template_text = reducer_name.split(marker, 1)[1].split(">(", 1)[0]
        template_arguments = tuple(
            value.strip() for value in template_text.split(","))
        if len(template_arguments) < 4:
            raise ValueError(
                "cuBLAS split-K reducer template ABI is incomplete")
        partial_type = template_arguments[3]
        if partial_type == "__half":
            return 2
        elif partial_type == "float":
            return 4
        else:
            raise ValueError(
                "cuBLAS split-K reducer has an unsupported partial type: "
                f"{partial_type!r}")
    elif "elementwiseadd" in lowered_reducer:
        if "float const*" not in reducer_name:
            raise ValueError(
                "mcBLAS split-K reducer ABI does not expose an FP32 partial")
        return 4
    else:
        raise AssertionError("unhandled recognized split-K reducer")


def observed_gemm_split_k_partial_dram_bytes(
        kernel_records, op_attributes=None) -> int:
    """Return the frozen complete-partial DRAM width for a split launch.

    The retained mcBLAS ``DevDstElementwiseAdd`` family has 70/70 exact
    main/reducer observations whose non-duration physical global write/read
    counters cover a complete FP32 partial buffer.  The retained cuBLAS
    ``splitKreduce`` family does not: all 26 main launches write less than the
    complete partial buffer to DRAM.  Map those versioned launch families to
    the source-visible width and 0 bytes/element respectively; individual
    counter magnitudes do not enter Forward.
    """

    partial_bytes = observed_gemm_split_k_partial_bytes(
        kernel_records, op_attributes)
    if partial_bytes == 0:
        return 0
    reducer_names = []
    for record in kernel_records:
        metrics = extract_ncu_metrics(record)
        lowered = metrics["kernel_name"].lower()
        if "splitkreduce" in lowered or "elementwiseadd" in lowered:
            reducer_names.append(lowered)
    if len(reducer_names) != 1:
        raise ValueError(
            "GEMM split-K DRAM partial contract requires one reducer")
    reducer_name = reducer_names[0]
    if "elementwiseadd" in reducer_name:
        return partial_bytes
    if "splitkreduce" in reducer_name:
        return 0
    raise AssertionError("unhandled recognized split-K reducer")


_GEMM_ATTR_DTYPES = frozenset({"bf16", "fp16", "float16", "fp8", "int8"})


def _normalize_gemm(op_attributes: Mapping, kdata: Mapping,
                    hardware_id: str, hardware,
                    case_split_k: Optional[int],
                    case_partial_bytes: Optional[int],
                    case_partial_dram_bytes: Optional[int]) -> KernelSpec:
    metrics = extract_ncu_metrics(kdata)
    runtime_name = metrics["kernel_name"]
    lowered = runtime_name.lower()
    dtype = _non_empty_string(
        op_attributes, "dtype", "GEMM Attr").lower()
    if dtype not in _GEMM_ATTR_DTYPES:
        raise ValueError(f"unsupported GEMM Attr.dtype: {dtype!r}")
    dtype_bytes = DTYPE_BYTES[dtype]
    shape = {
        "b": _positive_int(op_attributes, "B", "GEMM Attr"),
        "m": _positive_int(op_attributes, "M", "GEMM Attr"),
        "n": _positive_int(op_attributes, "N", "GEMM Attr"),
        "k": _positive_int(op_attributes, "K", "GEMM Attr"),
    }
    non_tensor = is_non_tensor(
        "GEMM", runtime_name, ascend_op_type=metrics["op_type"])
    if non_tensor and "gemv" not in lowered:
        is_split_reducer = (
            "splitkreduce" in lowered or "elementwiseadd" in lowered)
        if is_split_reducer:
            if case_split_k is None:
                raise ValueError(
                    "GEMM split reducer requires the sibling tensor "
                    "launch split_k")
            if case_split_k <= 1:
                raise ValueError(
                    "GEMM split reducer requires tensor split_k > 1")
            variant = "gemm_reduce"
            attributes = {"split_k": case_split_k}
            observed_aux_algorithm = (
                "observed_mcblas_gemm_aux"
                if "elementwiseadd" in lowered
                else "observed_gemm_aux")
        elif "reduce" in lowered:
            raise ValueError(
                "GEMM reduction identity lacks a source-backed split count")
        elif any(token in lowered for token in (
                "elementwise", "vectorized", "permute")):
            variant = "gemm_elementwise"
            attributes = {}
        else:
            raise ValueError(
                f"unknown GEMM auxiliary implementation: {runtime_name}")
        launch = observed_launch_config(
            metrics, KernelKind.VECTOR,
            {"element_count": shape["b"] * shape["m"] * shape["n"],
             "dtype_bytes": dtype_bytes},
            algorithm=(
                observed_aux_algorithm
                if is_split_reducer else "observed_gemm_aux"),
            variant=variant)
        if is_split_reducer:
            # Both retained cuBLAS splitKreduce and mcBLAS
            # DevDstElementwiseAdd ABIs consume FP32 accumulator partials
            # before storing the requested output dtype.  The width is a
            # discrete runtime-template fact, not a duration-derived value.
            if case_partial_bytes is None or case_partial_bytes <= 0:
                raise ValueError(
                    "GEMM split reducer requires a source-backed partial "
                    "element width")
            if (case_partial_dram_bytes is None
                    or case_partial_dram_bytes not in (
                        0, case_partial_bytes)):
                raise ValueError(
                    "GEMM split reducer requires an explicit zero or full "
                    "partial DRAM width")
            launch = dataclasses.replace(
                launch,
                parameters={
                    "split_k_partial_bytes_per_element": float(
                        case_partial_bytes),
                    "split_k_partial_dram_bytes_per_element": float(
                        case_partial_dram_bytes),
                },
            )
            if (hardware_id == "nvidia_a800"
                    and runtime_name == _A800_CUBLASLT_OUTPUT_DTYPE_REDUCER):
                source_parameters = (
                    cublaslt_output_dtype_reducer_source_parameters(
                        element_count=shape["b"] * shape["m"] * shape["n"],
                        split_k=case_split_k,
                        grid=launch.grid,
                        block=launch.block,
                    ))
                launch = dataclasses.replace(
                    launch,
                    parameters={**launch.parameters, **source_parameters},
                )
        return KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape={
                "element_count": shape["b"] * shape["m"] * shape["n"],
                "dtype_bytes": dtype_bytes,
            },
            dtype=dtype,
            variant=variant,
            launch=launch,
            attributes=attributes,
        )

    variant = "gemv" if non_tensor else "gemm"
    if lowered.startswith("mcblas__"):
        observed_algorithm = "observed_mcblas_gemm"
    else:
        observed_algorithm = "observed_gemm"
    launch = observed_launch_config(
        metrics, KernelKind.GEMM, shape,
        algorithm=observed_algorithm, variant=variant)
    nvjet = parse_nvjet_gemm_symbol(runtime_name)
    if nvjet is None:
        launch_split_k = metrics["grid_z"]
    else:
        # Hopper nvjet keeps grid.z == 1; its K split is the cluster-padded
        # tile multiplicity of the launched grid (a launch-geometry fact).
        launch_split_k = observed_nvjet_gemm_split_k(op_attributes, metrics)
        launch = dataclasses.replace(
            launch,
            tiling=dataclasses.replace(
                launch.tiling, split_k=launch_split_k))
    if case_split_k is not None and launch_split_k != case_split_k:
        raise ValueError(
            "GEMM tensor launch disagrees with case split_k: "
            f"launch={launch_split_k}, case={case_split_k}")
    parameters = {
        "compulsory_traffic": 0.0,
        "weight_tile_rows": 0.0,
        "mte2_tile_bytes": 0.0,
        "aiv_mixed": 0.0,
        "n_major_traffic": 0.0,
        "split_k_partial_bytes_per_element": (
            float(case_partial_bytes)
            if launch_split_k > 1 and case_partial_bytes is not None
            else 0.0),
        "split_k_partial_dram_bytes_per_element": (
            float(case_partial_dram_bytes)
            if (launch_split_k > 1
                and case_partial_dram_bytes is not None)
            else 0.0),
    }
    tiling = launch.tiling
    if not non_tensor and hardware_id == "ascend_910b":
        # The Dataset owns the exact duration-free CANN payload emitted by its
        # measured rank-two operation.  Decode that observation directly;
        # Serving independently selects a payload-derived program from
        # launch_data, and both paths meet at CannGemmLaunchProgram.
        selection_shape = {
            "b": 1,
            "m": shape["b"] * shape["m"],
            "n": shape["n"],
            "k": shape["k"],
        }
        observation = metrics["ascend"]["gemm_launch"]
        expected_observation_fields = {
            "schema_version", "software_revision", "input_rank",
            "kernel_family", "kernel_symbol", "block_dim",
            "tiling_payload_hex", "tiling_payload_sha256",
            "capture_baseline_commit", "capture_artifact_sha256",
            "capture_sha256",
            "duration_consumed", "device_kernel_submitted",
        }
        if (not isinstance(observation, Mapping)
                or set(observation) != expected_observation_fields):
            raise ValueError(
                "Ascend 910B GEMM Dataset launch observation fields differ")
        if (observation["schema_version"] != 1
                or observation["software_revision"] != "cann_8.5.1"
                or observation["input_rank"] != 2
                or observation["duration_consumed"] is not False
                or observation["device_kernel_submitted"] is not False):
            raise ValueError(
                "Ascend 910B GEMM Dataset launch observation contract differs")
        if runtime_name not in (
                observation["kernel_symbol"],
                f'{observation["kernel_symbol"]}_mix_aic'):
            raise ValueError(
                "Ascend 910B GEMM Dataset payload symbol disagrees with its "
                "profiler record")
        if observation["block_dim"] != launch.grid_size:
            raise ValueError(
                "Ascend 910B GEMM Dataset payload blockDim disagrees with its "
                "profiler record")
        for field in (
                "tiling_payload_sha256", "capture_artifact_sha256",
                "capture_sha256"):
            value = observation[field]
            if (not isinstance(value, str)
                    or re.fullmatch(r"[0-9a-f]{64}", value) is None):
                raise ValueError(
                    f"Ascend 910B GEMM Dataset {field} is not SHA256")
        baseline = observation["capture_baseline_commit"]
        if (not isinstance(baseline, str)
                or re.fullmatch(r"[0-9a-f]{40}", baseline) is None):
            raise ValueError(
                "Ascend 910B GEMM Dataset capture baseline is not a commit")
        payload_hex = observation["tiling_payload_hex"]
        if not isinstance(payload_hex, str):
            raise TypeError(
                "Ascend 910B GEMM Dataset tiling payload must be hexadecimal")
        try:
            payload = bytes.fromhex(payload_hex)
        except ValueError as error:
            raise ValueError(
                "Ascend 910B GEMM Dataset tiling payload is not hexadecimal") \
                from error
        if hashlib.sha256(payload).hexdigest() != (
                observation["tiling_payload_sha256"]):
            raise ValueError(
                "Ascend 910B GEMM Dataset tiling payload SHA256 changed")
        projected_launch = launch_from_cann_gemm_observation(
            payload, selection_shape, observation["block_dim"],
            observation["kernel_symbol"], observation["kernel_family"],
            variant, hardware)
        tiling = projected_launch.tiling
        parameters = dict(projected_launch.parameters)
        if tiling is None:
            raise ValueError(
                "Ascend 910B GEMM launch projection requires a macro tile")
        source_program = projected_launch.source_program
        launch = dataclasses.replace(
            launch,
            algorithm=projected_launch.algorithm,
            grid=projected_launch.grid,
            tiling=tiling,
            parameters=parameters,
            execution_variant=projected_launch.execution_variant,
            source_program=source_program,
        )
    elif not non_tensor and hardware.aic_aiv_separated:
        raise NotImplementedError(
            f"{hardware_id} tensor GEMM requires an explicit macro tile")
    elif not non_tensor and tiling is None:
        if hardware.kernel_memory_model not in (
                "pim_bank", "systolic_scratchpad"):
            raise ValueError(
                "tensor GEMM observation has no parseable tile on a tiled "
                f"architecture: {runtime_name!r}")
        # Cacheless architectures do not encode a GPU tile in their runtime
        # symbol. A concrete structural tile satisfies the common launch
        # contract; their architecture path prices traffic independently of it.
        tiling = TilingSpec(
            tile_shape={"m": 16, "n": 16, "k": 16},
            split_k=metrics["grid_z"],
        )

    if not non_tensor and observed_algorithm == "observed_mcblas_gemm":
        if tiling is None:
            raise ValueError("observed mcBLAS GEMM requires a concrete tile")
        m_tiles = math.ceil(
            (shape["b"] * shape["m"]) / tiling.tile_shape["m"])
        n_tiles = math.ceil(shape["n"] / tiling.tile_shape["n"])
        swizzle = find_gemm_swizzle_convention(
            m_tiles, n_tiles, launch.grid[0], launch.grid[1])
        parameters["n_major_traffic"] = float(swizzle is None)

    # One observed XMMa launch has no valid swizzled LRU schedule in the
    # historical model. Stamp compulsory traffic as an explicit fact of this
    # exact observed implementation and launch geometry.
    if ("xmma" in lowered
            and shape == {"b": 512, "m": 1, "n": 5120, "k": 5120}
            and tiling is not None
            and tuple(tiling.tile_shape[name] for name in ("m", "n", "k"))
            == (128, 192, 32)):
        parameters["compulsory_traffic"] = 1.0
    cta_raster = None
    # No verified CTA raster word layout exists for nvjet (H200 WP0 launch
    # report): its launch keeps the schedule-free traffic envelope instead of
    # decoding a CUTLASS swizzle from a cluster-shaped grid by coincidence.
    if (nvjet is None and not non_tensor and tiling is not None
            and not hardware.aic_aiv_separated):
        # A launch whose selection retains the vendor CTA map uses it.  A
        # launch that does not can still decode one when its grid shape is
        # itself the grouped encoding: a grid of ``tiles * group`` by
        # ``ceil(other / group)`` states the group, and a grid that is one CTA
        # per tile states a direct raster.  A grid that is neither mounts no
        # map and keeps the schedule-free envelope.
        if observed_algorithm == "observed_gemm":
            cta_raster = observed_gemm_cta_raster(
                shape, dtype, variant,
                tuple(tiling.tile_shape[name] for name in ("m", "n", "k")),
                tiling.split_k)
        if cta_raster is None:
            cta_raster = _decoded_grid_cta_raster(
                shape, tiling, launch.grid, metrics["grid_z"])
    launch = dataclasses.replace(launch, tiling=tiling, parameters=parameters)
    if cta_raster is not None:
        launch = dataclasses.replace(launch, source_program=cta_raster)
    return KernelSpec(
        kind=KernelKind.GEMM,
        hardware_id=hardware_id,
        shape=shape,
        dtype=dtype,
        variant=variant,
        launch=launch,
    )


def _decoded_grid_cta_raster(shape, tiling, grid, grid_z):
    """Decode the CTA map a tensor GEMM grid shape states, when it states one."""

    m_tiles = math.ceil((shape["b"] * shape["m"]) / tiling.tile_shape["m"])
    n_tiles = math.ceil(shape["n"] / tiling.tile_shape["n"])
    swizzle = find_gemm_swizzle_convention(m_tiles, n_tiles, grid[0], grid[1])
    if swizzle is None or grid[2] != grid_z:
        return None
    axis, width = swizzle
    if width & (width - 1):
        return None
    convention = "cutlass_n_group" if axis == "M" else "cutlass_m_group"
    return GpuCtaRasterProgram(
        convention, width.bit_length() - 1, False, grid_z)


def _normalize_flash_attention(op_attributes: Mapping, kdata: Mapping,
                               hardware_id: str, hardware) -> KernelSpec:
    metrics = extract_ncu_metrics(kdata)
    runtime_name = metrics["kernel_name"]
    query_length = _positive_int(
        op_attributes, "Q_LEN", "FLASH_ATTN Attr")
    kv_length = _positive_int(
        op_attributes, "KV_LEN", "FLASH_ATTN Attr")
    causal = op_attributes["causal"]
    kv_layout = _non_empty_string(
        op_attributes, "kv_layout", "FLASH_ATTN Attr")
    if hardware_id == "metax_c500" and kv_layout == "mla":
        # The pinned runner passes causal=True to flash_mla_with_kvcache even
        # though all 16 historical case labels recorded C0/False.  Decode Q=1
        # has the same attended-element count under bottom-right masking, but
        # the normalized execution contract must carry the fact actually
        # supplied to the host ABI.  Reject a future incompatible recording
        # instead of silently accepting two meanings.
        if causal is not False:
            raise ValueError(
                "canonical MetaX MLA observation must retain its historical "
                "causal=False label before runner-fact normalization")
        causal = True
    kv_distribution = _non_empty_string(
        op_attributes, "kv_dist", "FLASH_ATTN Attr")
    batch = _positive_int(op_attributes, "B", "FLASH_ATTN Attr")
    query_lengths = _positive_int_sequence(
        op_attributes, "Q_LENS", "FLASH_ATTN Attr", batch,
        query_length)
    kv_lengths = _positive_int_sequence(
        op_attributes, "KV_LENS", "FLASH_ATTN Attr", batch,
        kv_length)
    if any(request_query > request_kv
           for request_query, request_kv in zip(query_lengths, kv_lengths)):
        raise ValueError(
            "FLASH_ATTN each request requires Q_LENS <= KV_LENS: "
            f"Q_LENS={query_lengths}, KV_LENS={kv_lengths}")
    requested_query_length = _positive_int(
        op_attributes, "Q_REQUESTED", "FLASH_ATTN Attr")
    requested_kv_length = _positive_int(
        op_attributes, "KV_REQUESTED", "FLASH_ATTN Attr")
    if requested_kv_length != _positive_int(
            op_attributes, "S", "FLASH_ATTN Attr"):
        raise ValueError(
            "FLASH_ATTN KV_REQUESTED must equal the case S bucket")
    if requested_kv_length < kv_length:
        raise ValueError(
            "FLASH_ATTN KV_REQUESTED cannot be smaller than KV_LEN")
    if kv_layout == "chunked_prefill":
        if sum(query_lengths) > requested_query_length:
            raise ValueError(
                "FLASH_ATTN chunked query lengths exceed Q_REQUESTED")
    elif requested_query_length != query_length:
        raise ValueError(
            "non-chunked FLASH_ATTN Q_REQUESTED must equal Q_LEN")
    split_policy = _non_empty_string(
        op_attributes, "split_policy", "FLASH_ATTN Attr")
    if split_policy not in ("forced", "serving"):
        raise ValueError(
            f"unsupported FLASH_ATTN split_policy: {split_policy!r}")
    shape = {
        "batch": batch,
        "q_len": query_length,
        "kv_len": kv_length,
        "head_dim": _positive_int(op_attributes, "D", "FLASH_ATTN Attr"),
        "num_query_heads": _positive_int(
            op_attributes, "HQ", "FLASH_ATTN Attr"),
        "num_kv_heads": _positive_int(
            op_attributes, "HKV", "FLASH_ATTN Attr"),
    }
    copy_dtype = None
    copy_vector_width = None
    if "direct_copy_kernel_cuda" in runtime_name:
        upstream_bf16 = (
            runtime_name.startswith("void at::native::elementwise_kernel<")
            and "lambda(c10::BFloat16)" in runtime_name
            and "elementwise_kernel<(int)128, (int)4" in runtime_name
        )
        upstream_float32 = (
            runtime_name.startswith("void at::native::elementwise_kernel<")
            and "lambda(float)" in runtime_name
            and "elementwise_kernel<(int)128, (int)2" in runtime_name
        )
        metax_bf16 = (
            hardware_id == "metax_c500"
            and runtime_name.startswith((
                "void at::native::elementwise_kernel_3_1_"
                "transpose12_half_copy_64<",
                "void at::native::elementwise_kernel_4_1_"
                "transpose12_copy_64<",
            ))
            and "c10::BFloat16" in runtime_name
            and "lambda(c10::BFloat16)" in runtime_name
        )
        if upstream_bf16 or metax_bf16:
            copy_dtype = "bf16"
            copy_vector_width = 4
            copy_elements = (
                sum(query_lengths) * shape["num_query_heads"]
                * shape["head_dim"])
        elif upstream_float32:
            copy_dtype = "float32"
            copy_vector_width = 2
            copy_elements = sum(query_lengths) * shape["num_query_heads"]
        else:
            raise ValueError(
                "FLASH_ATTN direct-copy kernel has an unknown or unsupported "
                "dtype/vector "
                f"template: {runtime_name!r}")
        if hardware_id == "metax_c500":
            allowed_copy_layouts = (
                "contiguous", "paged", "paged_varlen")
        elif hardware_id == "nvidia_a800":
            allowed_copy_layouts = ("paged", "paged_varlen")
        else:
            raise ValueError(
                "FLASH_ATTN direct-copy observation is unsupported on "
                f"hardware {hardware_id!r}")
        if query_length != 1 or kv_layout not in allowed_copy_layouts:
            raise ValueError(
                "FLASH_ATTN direct-copy observation has an unsupported "
                f"decode layout: hardware={hardware_id!r}, "
                f"kv_layout={kv_layout!r}")
        if metax_bf16:
            # MetaX's transpose12 copy assigns one workgroup to each 64-value
            # slice of the outer (Q,D) plane and loops over heads inside the
            # workgroup.  It is therefore not the upstream flat elementwise
            # launch even though both kernels copy the same semantic tensor.
            expected_grid_x = math.ceil(
                sum(query_lengths) * shape["head_dim"] / 64)
        else:
            expected_grid_x = math.ceil(
                copy_elements / (metrics["block_size"] * copy_vector_width))
        if ((metrics["grid_x"], metrics["grid_y"], metrics["grid_z"])
                != (expected_grid_x, 1, 1)):
            raise ValueError(
                "FLASH_ATTN direct-copy grid disagrees with semantic element "
                f"count: grid={(metrics['grid_x'], metrics['grid_y'], metrics['grid_z'])}, "
                f"expected={(expected_grid_x, 1, 1)}")
        copy_shape = {"element_count": copy_elements}
        copy_launch = observed_launch_config(
            metrics, KernelKind.VECTOR, copy_shape,
            algorithm="observed_flash_attention_copy", variant="elementwise")
        if hardware_id == "nvidia_a800":
            if copy_dtype == "bf16":
                tensor_iterator_dimensions = 4
            elif copy_dtype == "float32":
                tensor_iterator_dimensions = 3
            else:
                raise ValueError(
                    "A800 FLASH_ATTN direct-copy dtype is unsupported: "
                    f"{copy_dtype!r}")
            copy_launch = dataclasses.replace(
                copy_launch,
                parameters={
                    SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER: (
                        float(tensor_iterator_dimensions)),
                },
            )
        return KernelSpec(
            kind=KernelKind.VECTOR,
            hardware_id=hardware_id,
            shape=copy_shape,
            dtype=copy_dtype,
            variant="elementwise",
            launch=copy_launch,
            attributes={
                "reads": 1, "writes": 1, "operation": "copy"},
        )

    if hardware_id == "metax_c500" and kv_layout == "mla":
        lowered_runtime_name = runtime_name.lower()
        if "get_mla_metadata_kernel" in lowered_runtime_name:
            if query_length != 1:
                raise ValueError(
                    "MetaX FlashMLA metadata setup supports decode Q=1 only")
            scheduler_partitions = METAX_MLA_SCHEDULER_PARTITIONS
            scheduler_metadata_ints = scheduler_partitions * 8
            split_metadata_ints = batch + 1
            scheduler_work = derive_mla_scheduler_work(kv_lengths)
            expected_dynamic_smem = 4 * (5 * batch + 1)
            launch_header = kdata["Sections"]["Launch Statistics"]["Header"]
            dynamic_smem = launch_header[
                "Dynamic Shared Memory Per Block [byte/block]"]
            if isinstance(dynamic_smem, bool) or not isinstance(
                    dynamic_smem, int):
                raise TypeError(
                    "MetaX FlashMLA metadata dynamic shared memory must be "
                    "an integer")
            actual_grid = (
                metrics["grid_x"], metrics["grid_y"], metrics["grid_z"])
            if actual_grid != (1, 1, 1):
                raise ValueError(
                    "MetaX FlashMLA metadata grid must be (1, 1, 1), got "
                    f"{actual_grid}")
            if metrics["block_size"] != 32:
                raise ValueError(
                    "MetaX FlashMLA metadata block size must be 32, got "
                    f"{metrics['block_size']}")
            if dynamic_smem != expected_dynamic_smem:
                raise ValueError(
                    "MetaX FlashMLA metadata dynamic shared memory disagrees "
                    f"with 4 * (5 * B + 1): actual={dynamic_smem}, "
                    f"expected={expected_dynamic_smem}")
            metadata_shape = {
                "batch": batch,
                "kv_block_count": scheduler_work.kv_block_count,
                "scheduler_partitions": scheduler_partitions,
                "scheduler_metadata_ints": scheduler_metadata_ints,
                "split_metadata_ints": split_metadata_ints,
                "scheduler_segment_count": scheduler_work.segment_count,
                "scheduler_worker_run_count": (
                    scheduler_work.segment_count
                    - scheduler_work.same_worker_transitions),
            }
            metadata_launch = observed_launch_config(
                metrics, KernelKind.VECTOR, metadata_shape,
                algorithm="observed_flash_mla_metadata",
                variant="flash_mla_metadata")
            return KernelSpec(
                kind=KernelKind.VECTOR,
                hardware_id=hardware_id,
                shape=metadata_shape,
                dtype="int32",
                variant="flash_mla_metadata",
                launch=metadata_launch,
            )
        if ("fillfunctor<bool>" in lowered_runtime_name
                or "fill_functor<bool>" in lowered_runtime_name):
            if query_length != 1:
                raise ValueError(
                    "MetaX FlashMLA bool fill supports decode Q=1 only")
            fill_shape = {"element_count": sum(query_lengths)}
            fill_launch = observed_launch_config(
                metrics, KernelKind.VECTOR, fill_shape,
                algorithm="observed_flash_mla_bool_fill",
                variant="elementwise")
            return KernelSpec(
                kind=KernelKind.VECTOR,
                hardware_id=hardware_id,
                shape=fill_shape,
                dtype="uint8",
                variant="elementwise",
                launch=fill_launch,
                attributes={
                    "reads": 0, "writes": 1, "operation": "fill"},
            )

    if hardware_id == "nvidia_h200":
        return _normalize_h200_flash_attention(
            op_attributes, metrics, shape, query_lengths, kv_lengths,
            causal, kv_layout, kv_distribution, split_policy, hardware)
    non_tensor = is_non_tensor(
        "FLASHATTN", runtime_name, ascend_op_type=metrics["op_type"])
    if non_tensor:
        variant = "combine"
    elif query_length == 1:
        variant = "decode"
    elif query_length > 1:
        variant = "prefill"
    else:
        raise ValueError("FLASH_ATTN query length must be positive")
    launch = observed_launch_config(
        metrics, KernelKind.FLASH_ATTN, shape,
        algorithm="observed_flash_attention", variant=variant)
    metax_static_trait = None
    metax_padded_tensor_work = False
    metax_tile16_decode_counter_program = False
    if (hardware_id == "metax_c500" and not non_tensor
            and kv_layout != "mla"):
        metax_static_trait = _metax_372_static_flash_trait(
            runtime_name, metrics, shape, hardware)
    if hardware_id == "metax_c500" and kv_layout == "mla" and non_tensor:
        # The pinned FlashMLA invocation emits a dedicated combine launch.
        # Give that observed implementation fact its own canonical identity so
        # the predictor can preserve the unresolved MACA partial layout
        # without branching on hardware ID or changing ordinary/A800 combine
        # semantics.
        launch = dataclasses.replace(
            launch, algorithm="observed_metax_flash_mla_combine")
    if hardware_id == "metax_c500" and not non_tensor:
        lowered_runtime_name = runtime_name.lower()
        runtime_core_name = (
            lowered_runtime_name[len("void "):]
            if lowered_runtime_name.startswith("void ")
            else lowered_runtime_name
        )
        # Three MACA 3.7.2 layer-1 observations bind this non-causal direct
        # specialization across GQA ratios 5/8 and non-GQA.  Every fully
        # covered workgroup executes 4,096 MMA instructions; the package
        # registry's BlockM/BlockN meaning therefore closes the complete
        # 128x64 QK/PV program for each full KV tile, including the Q=1 row
        # tail.  The 32-workgroup non-GQA capture covered 31 workgroups, so
        # its raw total is not treated as a full-grid counter: grid scaling
        # follows the identical static CTA program.  A partial final KV tile
        # remains unproved and keeps the semantic lower bound.
        metax_padded_tensor_work = (
            metax_static_trait == (128, 128, 64, 4)
            and variant == "decode"
            and causal is False
            and all(length % 64 == 0 for length in kv_lengths)
            and runtime_core_name.startswith("flash_fwd_kernel<")
        )
        observed_tile = (
            None if launch.tiling is None
            else launch.tiling.tile_shape)
        tile16_worker = (
            runtime_name.startswith((
                "void flash_fwd_kernel<",
                "void flash_fwd_splitkv_kernel<",
            ))
            and metrics["block_size"] == 64
            and observed_tile is not None
            and {"q", "kv"}.issubset(observed_tile)
            and (observed_tile["q"], observed_tile["kv"]) == (16, 16)
        )
        if tile16_worker:
            logical_q_tile = (
                32 if variant == "prefill" and batch == 1 else 16)
            tile_shape = dict(observed_tile)
            tile_shape.update({
                "q": logical_q_tile,
                "query_heads_per_workgroup": 4,
            })
            launch = dataclasses.replace(
                launch,
                tiling=dataclasses.replace(
                    launch.tiling, tile_shape=tile_shape),
            )
            metax_tile16_decode_counter_program = (
                metax_static_trait in {
                    (64, 16, 16, 1),
                    (128, 16, 16, 1),
                }
                and variant == "decode"
                and causal is False)
            if metax_tile16_decode_counter_program:
                # Exact layer-1 counters close eight (D64) or sixteen (D128)
                # MMA instructions for every scheduled BlockN step, including
                # ragged final steps.
                metax_padded_tensor_work = True
            if variant == "prefill" and kv_layout == "paged":
                algorithm = (
                    "observed_metax_flash_attention_grouped_direct"
                    if batch == 1 else
                    "observed_metax_flash_attention_persistent_queue")
                launch = dataclasses.replace(launch, algorithm=algorithm)
            elif (variant == "prefill"
                  and kv_layout == "chunked_prefill"):
                launch = dataclasses.replace(
                    launch,
                    algorithm=(
                        "observed_metax_flash_attention_packed_varlen_queue"))
            elif (variant == "decode"
                  and kv_layout == "paged_varlen"):
                recorded_splits = _positive_int(
                    op_attributes, "SP", "FLASH_ATTN Attr")
                request_major = (
                    metrics["grid_x"] == shape["num_kv_heads"]
                    and metrics["grid_y"] >= batch
                    and metrics["grid_z"] == recorded_splits)
                if not request_major:
                    launch = dataclasses.replace(
                        launch,
                        algorithm=(
                            "observed_metax_flash_attention_"
                            "packed_varlen_queue"))
        elif "flash_fwd_splitkv_mla_kernel" in lowered_runtime_name:
            launch = dataclasses.replace(
                launch,
                algorithm=(
                    "observed_metax_flash_attention_persistent_queue"))
        elif (variant == "prefill"
              and kv_layout == "chunked_prefill"
              and runtime_core_name.startswith(
                  "flash_fwd_splitkv_kernel<")
              and observed_tile is not None
              and {"q", "kv"}.issubset(observed_tile)
              and (observed_tile["q"], observed_tile["kv"]) == (64, 64)
              and metrics["block_size"] == 256):
            tile_shape = dict(observed_tile)
            tile_shape["query_heads_per_workgroup"] = 1
            launch = dataclasses.replace(
                launch,
                algorithm=(
                    "observed_metax_flash_attention_packed_varlen_queue"),
                tiling=dataclasses.replace(
                    launch.tiling, tile_shape=tile_shape),
            )
    if hardware_id == "ascend_910b":
        if op_attributes["SP"] is not None:
            raise ValueError(
                "Ascend 910B FLASH_ATTN observation must record SP=None")
        if op_attributes["SP_REQUESTED"] is not None:
            raise ValueError(
                "Ascend 910B FLASH_ATTN observation must record "
                "SP_REQUESTED=None")
        # CANN exposes one fused attention operator rather than a public
        # split-KV + combine launch contract. The internal TilingSpec uses one
        # logical partition so the shared predictor contract remains complete;
        # this is not a dataset SP field or a user-controlled CANN input.
        splits = 1
        production_launch = select_launch_config(
            hardware_id,
            KernelKind.FLASH_ATTN,
            shape,
            hardware.kernel_dtype,
            variant,
            {
                "is_decode": variant == "decode",
                "causal": causal,
                "kv_layout": kv_layout,
                "kv_dist": kv_distribution,
                "q_lens": query_lengths,
                "kv_lens": kv_lengths,
                "launch_policy": "injected_launch",
            },
            hardware=hardware,
        )
        if (launch.grid != production_launch.grid
                or launch.block != production_launch.block):
            raise ValueError(
                "Ascend 910B FLASH_ATTN observed launch disagrees with "
                "the exact production launch fact: "
                f"observed={launch.grid}/{launch.block}, "
                f"production={production_launch.grid}/"
                f"{production_launch.block}")
        algorithm_prefix = "cann_flash_attention_score:"
        if not production_launch.algorithm.startswith(algorithm_prefix):
            raise ValueError(
                "Ascend 910B FLASH_ATTN production algorithm is invalid")
        production_variant = production_launch.algorithm[
            len(algorithm_prefix):]
        expected_symbol_suffix = f"_{production_variant}_mix_aic"
        if not runtime_name.endswith(expected_symbol_suffix):
            raise ValueError(
                "Ascend 910B FLASH_ATTN observed runtime variant disagrees "
                "with the exact production launch fact")
        ascend_tiling = production_launch.tiling
    elif hardware_id in ("metax_c500", "nvidia_a800"):
        splits = _positive_int(op_attributes, "SP", "FLASH_ATTN Attr")
        requested_splits = _optional_positive_int(
            op_attributes, "SP_REQUESTED", "FLASH_ATTN Attr")
        metadata_selected_mla = (
            hardware_id == "metax_c500"
            and kv_layout == "mla"
            and split_policy == "serving")
        if requested_splits is None and not metadata_selected_mla:
            raise ValueError(
                f"{hardware_id} FLASH_ATTN requires SP_REQUESTED")
        if (split_policy == "forced"
                and requested_splits is not None
                and requested_splits != splits):
            raise ValueError(
                "forced FLASH_ATTN SP must equal SP_REQUESTED")
        if (hardware_id == "nvidia_a800" and not non_tensor
                and kv_layout != "mla"):
            # The executed split count is a versioned discrete launch fact,
            # not a duration-derived parameter.  Re-run the canonical selector
            # in forced-SP mode to bind the observation to the pinned FA2
            # source family and its exact semantic tile.  Keep the profiler's
            # physical grid and resource fields below: packed/varlen launch
            # geometry is not reconstructed by the generic selector.
            production_launch = select_launch_config(
                hardware_id,
                KernelKind.FLASH_ATTN,
                shape,
                hardware.kernel_dtype,
                variant,
                {
                    "is_decode": variant == "decode",
                    "causal": causal,
                    "kv_layout": kv_layout,
                    "kv_dist": kv_distribution,
                    "q_lens": query_lengths,
                    "kv_lens": kv_lengths,
                    "launch_policy": "injected_launch",
                    "split_policy": "forced",
                    "num_kv_splits": splits,
                },
                hardware=hardware,
            )
            production_tiling = production_launch.tiling
            observed_tiling = launch.tiling
            if production_tiling is None or observed_tiling is None:
                raise ValueError(
                    "A800 FA2 main launch requires source and observed tiles")
            semantic_axes = ("q", "kv", "head")
            production_tile = tuple(
                production_tiling.tile_shape[name]
                for name in semantic_axes)
            observed_tile = tuple(
                observed_tiling.tile_shape[name]
                for name in semantic_axes)
            if observed_tile != production_tile:
                raise ValueError(
                    "A800 FA2 observed tile disagrees with the pinned source "
                    f"launch: observed={observed_tile}, "
                    f"production={production_tile}")
            if launch.block != production_launch.block:
                raise ValueError(
                    "A800 FA2 observed block disagrees with the pinned source "
                    f"launch: observed={launch.block}, "
                    f"production={production_launch.block}")
            lowered_runtime_name = runtime_name.lower()
            if production_launch.algorithm == "flash_attn_dense":
                expected_symbol = (
                    "flash_fwd_kernel<" in lowered_runtime_name
                    and "flash_fwd_splitkv_kernel<" not in lowered_runtime_name)
            elif production_launch.algorithm in (
                    "flash_attn_splitkv", "vllm_flash_attn_splitkv"):
                expected_symbol = (
                    "flash_fwd_splitkv_kernel<" in lowered_runtime_name)
            else:
                raise ValueError(
                    "A800 FA2 selector returned an unsupported source family: "
                    f"{production_launch.algorithm!r}")
            if not expected_symbol:
                raise ValueError(
                    "A800 FA2 runtime symbol disagrees with the pinned source "
                    f"family {production_launch.algorithm!r}: "
                    f"{runtime_name!r}")
            launch = dataclasses.replace(
                launch,
                algorithm=production_launch.algorithm,
                tiling=production_tiling,
                parameters=production_launch.parameters,
            )
        elif (hardware_id == "nvidia_a800" and non_tensor
              and kv_layout != "mla"):
            production_launch = select_launch_config(
                hardware_id,
                KernelKind.FLASH_ATTN,
                shape,
                hardware.kernel_dtype,
                "combine",
                {
                    "launch_policy": "injected_launch",
                    "num_kv_splits": splits,
                    "kv_layout": kv_layout,
                },
                hardware=hardware,
            )
            if (launch.grid != production_launch.grid
                    or launch.block != production_launch.block):
                raise ValueError(
                    "A800 FA2 combine observation disagrees with the pinned "
                    "source launch: "
                    f"observed={launch.grid}/{launch.block}, "
                    f"production={production_launch.grid}/"
                    f"{production_launch.block}")
            if "flash_fwd_splitkv_combine_kernel<" not in runtime_name:
                raise ValueError(
                    "A800 FA2 combine runtime symbol disagrees with the "
                    f"pinned source family: {runtime_name!r}")
            template = re.search(
                r">>,\s*\(int\)(\d+),\s*\(int\)(\d+),\s*\(bool\)",
                runtime_name,
            )
            if template is None:
                raise ValueError(
                    "A800 FA2 combine runtime symbol does not expose "
                    "kBlockM/Log_max_splits")
            observed_block_m = int(template.group(1))
            observed_log_max_splits = int(template.group(2))
            expected_block_m = (
                production_launch.block_size * 4 // shape["head_dim"])
            expected_log_max_splits = (splits - 1).bit_length()
            if (observed_block_m, observed_log_max_splits) != (
                    expected_block_m, expected_log_max_splits):
                raise ValueError(
                    "A800 FA2 combine template disagrees with the pinned "
                    "source dispatch: "
                    f"observed={(observed_block_m, observed_log_max_splits)}, "
                    f"expected={(expected_block_m, expected_log_max_splits)}")
            launch = dataclasses.replace(
                launch,
                algorithm=production_launch.algorithm,
                tiling=production_launch.tiling,
                parameters=production_launch.parameters,
            )
    else:
        raise ValueError(
            f"unsupported FLASH_ATTN observation hardware: {hardware_id!r}")
    grouped_mla_stage1 = runtime_name == "_fwd_grouped_kernel_stage1"
    grouped_mla_stage2 = (
        runtime_name == "_fwd_kernel_stage2" and kv_layout == "mla")
    if grouped_mla_stage1:
        if (kv_layout != "mla" or query_length != 1
                or shape["head_dim"] != 512):
            raise ValueError(
                "_fwd_grouped_kernel_stage1 requires MLA decode with D=512")
        head_tile = 16
        kv_tile = 32
        expected_grid = (
            batch, math.ceil(shape["num_query_heads"] / head_tile), splits)
        actual_grid = (
            metrics["grid_x"], metrics["grid_y"], metrics["grid_z"])
        if actual_grid != expected_grid:
            raise ValueError(
                "grouped MLA launch grid disagrees with B/head-groups/SP: "
                f"actual={actual_grid}, expected={expected_grid}")
        # vLLM v0.11's Ampere fallback fixes BLOCK_H=16, BLOCK_N=32 and
        # qk_rope_dim=64.  These are structured launch facts; BLOCK_H is a
        # query-head tile, never a Q-row tile.
        tiling = TilingSpec(
            tile_shape={
                "query_heads": head_tile,
                "kv": kv_tile,
                "head": shape["head_dim"],
                "qk_head": shape["head_dim"] + 64,
                "value_head": shape["head_dim"],
            },
            num_kv_splits=splits,
        )
        launch = dataclasses.replace(
            launch,
            algorithm="triton_grouped_mla_stage1",
            tiling=tiling,
            parameters={
                "unpadded_flops": 0.0,
                # Grouped MLA is a Triton kernel, not FA2's SM80
                # m16n8k16 + gemm_rs implementation.
                "mma_smem_bytes_per_flop": 0.0,
                "split_scratch_dram_bytes_per_element": 0.0,
                "page_table_bytes_per_kv_panel": 0.0,
            },
        )
    elif grouped_mla_stage2:
        if (variant != "combine" or query_length != 1
                or shape["head_dim"] != 512
                or shape["num_query_heads"] != 64
                or shape["num_kv_heads"] != 1
                or splits <= 1):
            raise ValueError(
                "_fwd_kernel_stage2 requires split grouped-MLA decode")
        expected_grid = (batch, shape["num_query_heads"], 1)
        if launch.grid != expected_grid or launch.block != (128, 1, 1):
            raise ValueError(
                "grouped MLA stage2 launch disagrees with its pinned source: "
                f"actual={launch.grid}/{launch.block}, "
                f"expected={expected_grid}/(128, 1, 1)")
        launch = dataclasses.replace(
            launch,
            algorithm="triton_grouped_mla_stage2",
            tiling=TilingSpec(tile_shape={}, num_kv_splits=splits),
            parameters={
                "unpadded_flops": 1.0,
                "mma_smem_bytes_per_flop": 0.0,
                # The pinned stage-2 program reads FP32 Mid_O scratch from a
                # preceding launch.  Its NCU L2/DRAM counters close the full
                # four-byte scratch read for every canonical batch size,
                # including footprints smaller than L2, so this is an
                # inter-kernel DRAM handoff rather than an L2-resident input.
                "split_scratch_dram_bytes_per_element": 4.0,
                "page_table_bytes_per_kv_panel": 0.0,
            },
        )
    # MACA 3.7.2's versioned candidate registry binds the five standard trait
    # tuples to BlockM/BlockN.  Only direct launches may publish that static
    # tile here: queue classifiers still lack a semantic-job-to-worker map and
    # therefore retain an empty production tile.  A tile alone does not prove
    # tail arithmetic, traffic padding or a timed CTA worker schedule.
    metax_direct_trait_tile = (
        hardware_id == "metax_c500"
        and metax_static_trait is not None
        and launch.algorithm.startswith("observed_flash_attention:")
    )
    if metax_direct_trait_tile:
        launch = dataclasses.replace(
            launch,
            algorithm=(
                "observed_metax_flash_attention_trait_direct:"
                f"{variant}"),
        )
    metax_q64_prefill_runtime = (
        (metax_static_trait == (128, 64, 64, 4)
         and runtime_name == _METAX_372_Q64_PREFILL_RUNTIME)
        or (metax_static_trait == (64, 64, 64, 4)
            and runtime_name == _METAX_372_D64_Q64_PREFILL_RUNTIME)
    )
    metax_q64_prefill_source_program = (
        metax_direct_trait_tile
        and metax_q64_prefill_runtime
        and variant == "prefill"
        and causal is True
        and kv_layout in ("paged", "paged_varlen")
        and splits == 1
        and query_length % 64 == 0
        and all(
            query == key_value and query % 64 == 0
            for query, key_value in zip(query_lengths, kv_lengths))
        and launch.grid == (
            shape["num_query_heads"], query_length // 64, batch)
        and launch.block == (256, 1, 1)
    )
    if metax_q64_prefill_source_program:
        launch = dataclasses.replace(
            launch, algorithm=_METAX_372_Q64_PREFILL_ALGORITHM)
        metax_padded_tensor_work = True
    metax_q64_packed_counter_program = (
        metax_q64_prefill_runtime
        and variant == "prefill"
        and causal is True
        and kv_layout == "chunked_prefill"
        and splits == 1
        and launch.block == (256, 1, 1)
        and launch.algorithm
        == "observed_metax_flash_attention_packed_varlen_queue")
    if metax_q64_packed_counter_program:
        launch = dataclasses.replace(
            launch, algorithm=_METAX_372_Q64_PACKED_PREFILL_ALGORITHM)
        metax_padded_tensor_work = True
    metax_q64_decode_source_program = (
        metax_direct_trait_tile
        and runtime_name == _METAX_372_Q64_DECODE_RUNTIME
        and metax_static_trait == (128, 64, 64, 4)
        and variant == "decode"
        and causal is False
        and kv_layout == "contiguous"
        and splits == 1
        and all(query == 1 for query in query_lengths)
        and all(length % 64 == 0 for length in kv_lengths)
        and launch.grid == (shape["num_kv_heads"], batch, 1)
        and launch.block == (256, 1, 1)
    )
    if metax_q64_decode_source_program:
        launch = dataclasses.replace(
            launch, algorithm=_METAX_372_Q64_DECODE_ALGORITHM)
        # The exact-symbol layer-1 counter reports 2,048 MMA instructions per
        # CTA for the eight-panel S512 program, independent of GQA ratio.  The
        # compiler therefore executes the complete BlockM64 Tensor program;
        # only global Q/O accesses are predicated to live GQA rows.
        metax_padded_tensor_work = True
    metax_q128_prefill_source_program = (
        metax_direct_trait_tile
        and runtime_name == _METAX_372_Q128_PREFILL_RUNTIME
        and metax_static_trait == (128, 128, 64, 4)
        and variant == "prefill"
        and causal is True
        and kv_layout == "contiguous"
        and splits == 1
        and query_length % 128 == 0
        and all(
            query == key_value and query % 128 == 0
            for query, key_value in zip(query_lengths, kv_lengths))
        and launch.grid == (
            shape["num_query_heads"], batch, query_length // 128)
        and launch.block == (256, 1, 1)
    )
    if metax_q128_prefill_source_program:
        launch = dataclasses.replace(
            launch, algorithm=_METAX_372_Q128_PREFILL_ALGORITHM)
        metax_padded_tensor_work = True
    metax_q128_split_prefill_source_program = (
        metax_direct_trait_tile
        and runtime_name == _METAX_372_Q128_SPLIT_PREFILL_RUNTIME
        and metax_static_trait == (128, 128, 64, 4)
        and variant == "prefill"
        and causal is True
        and kv_layout in ("paged", "paged_varlen", "chunked_prefill")
        and splits == 1
        and query_length % 128 == 0
        and all(
            query == query_length
            and query % 128 == 0
            and key_value >= query
            and key_value % 64 == 0
            for query, key_value in zip(query_lengths, kv_lengths))
        and launch.grid == (
            shape["num_query_heads"], query_length // 128, batch)
        and launch.block == (256, 1, 1)
    )
    if metax_q128_split_prefill_source_program:
        launch = dataclasses.replace(
            launch, algorithm=_METAX_372_Q128_SPLIT_PREFILL_ALGORITHM)
        metax_padded_tensor_work = True
    has_attention_tile = (
        launch.tiling is not None
        and {"q", "kv"}.issubset(launch.tiling.tile_shape)
        and (hardware_id != "metax_c500"
             or metax_direct_trait_tile
             or metax_tile16_decode_counter_program
             or metax_q64_packed_counter_program)
    )
    if metax_direct_trait_tile:
        expected_tile = metax_static_trait[1:3]
        observed_tile = tuple(
            int(launch.tiling.tile_shape[name]) for name in ("q", "kv"))
        if observed_tile != expected_tile:
            raise ValueError(
                "MetaX MACA 3.7.2 static trait disagrees with the normalized "
                f"tile: trait={expected_tile}, tile={observed_tile}")
    if grouped_mla_stage1 or grouped_mla_stage2:
        pass
    elif hardware_id == "ascend_910b":
        if ascend_tiling is None:
            raise ValueError(
                "Ascend 910B FLASH_ATTN production launch requires tiling")
        tiling = dataclasses.replace(
            ascend_tiling, num_kv_splits=splits)
        launch = dataclasses.replace(
            launch, algorithm=production_launch.algorithm)
    elif has_attention_tile:
        tiling = dataclasses.replace(
            launch.tiling, num_kv_splits=splits)
    else:
        tiling = TilingSpec(
            tile_shape={}, num_kv_splits=splits)
    if not grouped_mla_stage1 and hardware_id == "ascend_910b":
        # The exact CANN launch package owns both the discrete variant and any
        # source-reflected worker schedule.  The profiler launch above is used
        # only to prove that its physical grid/block and runtime symbol agree
        # with that package.  Replacing these parameters with the generic
        # three-field observation schema would silently discard the canonical
        # source schedule from single-kernel evaluation.
        launch = dataclasses.replace(
            launch,
            tiling=tiling,
            parameters=production_launch.parameters,
            source_program=production_launch.source_program,
        )
    elif not (grouped_mla_stage1 or grouped_mla_stage2):
        if hardware_id == "nvidia_a800" and has_attention_tile:
            mma_smem_bytes_per_flop = 0.125
        elif hardware_id in ("nvidia_a800", "metax_c500"):
            mma_smem_bytes_per_flop = 0.0
        else:
            raise AssertionError(
                "FLASH_ATTN observation hardware was not normalized")
        launch = dataclasses.replace(
            launch,
            tiling=tiling,
            parameters={
                "unpadded_flops": (
                    0.0
                    if (hardware_id == "nvidia_a800" and has_attention_tile)
                    or metax_padded_tensor_work
                    else 1.0),
                "mma_smem_bytes_per_flop": mma_smem_bytes_per_flop,
                # Standard paged split main/combine families materialize an
                # FP32 scratch round trip.  A800's source-bound paged path and
                # C500's retained counters both expose that global-memory
                # traffic; MLA does not close uniformly.
                "split_scratch_dram_bytes_per_element": (
                    4.0
                    if (splits > 1 and (
                        (hardware_id == "metax_c500" and kv_layout != "mla")
                        or hardware_id == "nvidia_a800"))
                    else 0.0),
                "page_table_bytes_per_kv_panel": (
                    4.0 if (hardware_id == "nvidia_a800" and splits > 1
                            and kv_layout in ("paged", "paged_varlen")
                            and variant not in ("combine", "split_combine"))
                    else 0.0),
            },
        )
    if not isinstance(causal, bool):
        raise ValueError("FLASH_ATTN causal must be boolean")
    attributes = {
        "causal": causal,
        "kv_layout": kv_layout,
        "kv_dist": kv_distribution,
        "q_lens": query_lengths,
        "kv_lens": kv_lengths,
        "fa_ksteps": None,
        "fa_eff_mfrac": 1.0,
    }
    if "FA_KSTEPS" in op_attributes:
        attributes["fa_ksteps"] = _positive_int(
            op_attributes, "FA_KSTEPS", "FLASH_ATTN Attr")
    if "FA_EFF_MFRAC" in op_attributes:
        attributes["fa_eff_mfrac"] = _finite_number(
            op_attributes, "FA_EFF_MFRAC", "FLASH_ATTN Attr")
    return KernelSpec(
        kind=KernelKind.FLASH_ATTN,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant=variant,
        launch=launch,
        attributes=attributes,
    )


# H200 kernel-dataset runner (scripts/kernel_dataset/nvidia_h200/
# flash_attn_runner.py, MLA_ROPE_DIM): the absorbed MLA call passes q_pe /
# k_pe with the DeepSeek rope head 64 and q_nope / kv_c with the latent D.
_H200_MLA_ROPE_DIM = 64
_H200_FA3_MAIN_SYMBOL = "FlashAttnFwdSm90"
_H200_FA3_COMBINE_SYMBOL = "FlashAttnFwdCombine"


def _normalize_h200_flash_attention(
        op_attributes: Mapping, metrics: Mapping, shape: Mapping,
        query_lengths, kv_lengths, causal, kv_layout: str,
        kv_distribution: str, split_policy: str, hardware) -> KernelSpec:
    """Bind one H200 FA3 launch to the source-derived FA3 package.

    Only the absorbed MLA decode (FA3 ``hdim 64 / 512`` with its split
    combine) has a derived H200 template.  Its launch is re-selected from
    the pinned source (the runner calls FA3 with ``num_splits=0``, i.e. the
    heuristic, eager: no full-CUDA-graph split bound) and must agree with the
    observed grid/block, the observed combine launch and the recorded
    per-request dynamic splits; no duration is read.  Every other FA3 launch
    is normalized under an explicit unmodeled identity that every pricing
    method rejects.
    """

    runtime_name = metrics["kernel_name"]
    batch = shape["batch"]
    splits = _positive_int(op_attributes, "SP", "FLASH_ATTN Attr")
    observed = observed_launch_config(
        metrics, KernelKind.FLASH_ATTN, shape,
        algorithm="observed_flash_attention")
    is_main = _H200_FA3_MAIN_SYMBOL in runtime_name
    is_combine = _H200_FA3_COMBINE_SYMBOL in runtime_name
    common_attributes = {
        "causal": causal,
        "kv_layout": kv_layout,
        "kv_dist": kv_distribution,
        "q_lens": query_lengths,
        "kv_lens": kv_lengths,
    }
    if kv_layout != "mla":
        if is_combine:
            variant = "combine"
        elif shape["q_len"] == 1:
            variant = "decode"
        else:
            variant = "prefill"
        return KernelSpec(
            kind=KernelKind.FLASH_ATTN,
            hardware_id="nvidia_h200",
            shape=shape,
            dtype=hardware.kernel_dtype,
            variant=variant,
            launch=dataclasses.replace(
                observed,
                algorithm=FA3_UNMODELED_OBSERVATION_ALGORITHM,
                tiling=TilingSpec(tile_shape={}, num_kv_splits=splits),
                parameters={
                    "unpadded_flops": 1.0,
                    "mma_smem_bytes_per_flop": 0.0,
                    "split_scratch_dram_bytes_per_element": 0.0,
                    "page_table_bytes_per_kv_panel": 0.0,
                },
                execution_variant=variant,
            ),
            attributes={
                **common_attributes, "fa_ksteps": None, "fa_eff_mfrac": 1.0},
        )
    if not (is_main or is_combine):
        raise ValueError(
            "H200 FA3 MLA case contains a launch outside the FA3 main/"
            f"combine pair: {runtime_name!r}")
    # The runner calls FA3 MLA with causal=True regardless of the case label
    # (A800 canonical MLA cases are labelled C0); a one-token query executes
    # non-causally in FA3 either way, so only Q_LEN > 1 must carry True.
    if causal is not True and not (causal is False and shape["q_len"] == 1):
        raise ValueError(
            "H200 FA3 MLA multi-token observation must be causal")
    causal = True
    common_attributes["causal"] = True
    # The runner rejects non-uniform MLA lengths, so the executed KV
    # distribution is uniform whatever label the A800 case list carried
    # (its MLA cases are labelled ``skew``); bind the executed fact.
    if len(set(kv_lengths)) != 1 or len(set(query_lengths)) != 1:
        raise ValueError("H200 FA3 MLA observation requires uniform lengths")
    common_attributes["kv_dist"] = "uniform"
    if split_policy != "serving" or shape["num_kv_heads"] != 1:
        raise ValueError(
            "H200 FA3 MLA observation must be the one-latent-head serving "
            "call with FA3's own split heuristic")
    latent = shape["head_dim"]
    mla_shape = {
        "batch": batch,
        "q_len": shape["q_len"],
        "kv_len": shape["kv_len"],
        "num_query_heads": shape["num_query_heads"],
        "qk_head_dim": latent + _H200_MLA_ROPE_DIM,
        "v_head_dim": latent,
        "kv_lora_rank": latent,
    }
    mla_attributes = {
        **common_attributes,
        "full_cuda_graphs": False,
        "max_cudagraph_capture_size": 0,
    }
    main_launch = select_launch_config(
        "nvidia_h200", KernelKind.FLASH_ATTN, mla_shape,
        hardware.kernel_dtype, "mla_decode", mla_attributes, hardware)
    if main_launch.algorithm != FA3_DECODE_ALGORITHM:
        raise NotImplementedError(
            "H200 MLA observation is outside the derived FA3 decode domain: "
            f"selector returned {main_launch.algorithm!r}")
    static_splits = main_launch.tiling.num_kv_splits
    dynamic_splits = fa3_decode_dynamic_splits(
        query_lengths, kv_lengths, shape["num_query_heads"], static_splits,
        int(hardware.num_compute_units))
    if (list(dynamic_splits) != list(op_attributes["SP_PER_REQUEST"])
            or (static_splits > 1) != op_attributes["FA3_SPLIT_COMBINE"]):
        raise ValueError(
            "H200 FA3 MLA scheduler metadata disagrees with the source "
            f"split derivation: static={static_splits}, derived="
            f"{list(dynamic_splits)}, recorded="
            f"{op_attributes['SP_PER_REQUEST']}")
    if is_main:
        selected = main_launch
        spec_shape = mla_shape
        variant = "mla_decode"
        attributes = mla_attributes
    else:
        spec_shape = {
            "batch": batch,
            "q_len": shape["q_len"],
            "kv_len": shape["kv_len"],
            "head_dim": latent,
            "num_query_heads": shape["num_query_heads"],
            "num_kv_heads": 1,
        }
        attributes = {
            **common_attributes, "fa_ksteps": None, "fa_eff_mfrac": 1.0}
        selected = select_launch_config(
            "nvidia_h200", KernelKind.FLASH_ATTN, spec_shape,
            hardware.kernel_dtype, "combine",
            {**attributes, "num_kv_splits": static_splits}, hardware)
        if selected.algorithm != FA3_COMBINE_ALGORITHM:
            raise ValueError(
                "H200 FA3 MLA combine was not selected by the FA3 package")
        variant = "combine"
    if (observed.grid != selected.grid
            or observed.block != selected.block):
        raise ValueError(
            "H200 FA3 MLA observed launch disagrees with the pinned source: "
            f"observed={observed.grid}/{observed.block}, "
            f"source={selected.grid}/{selected.block}")
    return KernelSpec(
        kind=KernelKind.FLASH_ATTN,
        hardware_id="nvidia_h200",
        shape=spec_shape,
        dtype=hardware.kernel_dtype,
        variant=variant,
        launch=selected,
        attributes=attributes,
    )


def _normalize_eager_attn(op_attributes: Mapping, kdata: Mapping,
                          hardware_id: str, hardware) -> KernelSpec:
    """Normalize one decomposed QK-softmax-PV observation.

    Eager attention deliberately has no FlashAttention tile, split, layout, or
    fused-pipeline attributes in its production contract. Dataset-only fields
    are still validated here so malformed observations fail at the boundary.
    """

    metrics = extract_ncu_metrics(kdata)
    runtime_name = metrics["kernel_name"]
    if is_non_tensor(
            "EAGERATTN", runtime_name,
            ascend_op_type=metrics["op_type"]):
        raise ValueError(
            "EAGERATTN dataset entry must describe its decomposed tensor path")

    _positive_int(op_attributes, "S", "EAGERATTN Attr")
    _positive_int(op_attributes, "SP", "EAGERATTN Attr")
    _non_empty_string(op_attributes, "kv_layout", "EAGERATTN Attr")
    _non_empty_string(op_attributes, "kv_dist", "EAGERATTN Attr")
    causal = op_attributes["causal"]
    if not isinstance(causal, bool):
        raise ValueError("EAGERATTN causal must be boolean")

    shape = {
        "batch": _positive_int(op_attributes, "B", "EAGERATTN Attr"),
        "q_len": _positive_int(
            op_attributes, "Q_LEN", "EAGERATTN Attr"),
        "kv_len": _positive_int(
            op_attributes, "KV_LEN", "EAGERATTN Attr"),
        "head_dim": _positive_int(
            op_attributes, "D", "EAGERATTN Attr"),
        "num_query_heads": _positive_int(
            op_attributes, "HQ", "EAGERATTN Attr"),
        "num_kv_heads": _positive_int(
            op_attributes, "HKV", "EAGERATTN Attr"),
    }
    launch = observed_launch_config(
        metrics, KernelKind.EAGER_ATTN, shape,
        algorithm="observed_eager_attn", variant="decomposed")
    return KernelSpec(
        kind=KernelKind.EAGER_ATTN,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant="decomposed",
        launch=launch,
        attributes={},
    )


def _normalize_unfused_moe(op_attributes: Mapping, kdata: Mapping,
                           hardware_id: str, hardware) -> KernelSpec:
    """Normalize one aggregate router/expert/reduction decomposition."""

    metrics = extract_ncu_metrics(kdata)
    if is_non_tensor(
            "UNFUSEDMOE", metrics["kernel_name"],
            ascend_op_type=metrics["op_type"]):
        raise ValueError(
            "UNFUSEDMOE dataset entry must describe its aggregate tensor path")
    if hardware.kernel_memory_model not in (
            "pim_bank", "systolic_scratchpad"):
        raise ValueError(
            "UNFUSEDMOE observations require a cacheless execution model")
    shape = {
        "tokens": _positive_int(op_attributes, "TOK", "UNFUSED_MOE Attr"),
        "experts": _positive_int(op_attributes, "E", "UNFUSED_MOE Attr"),
        "topk": _positive_int(op_attributes, "TOPK", "UNFUSED_MOE Attr"),
        "hidden": _positive_int(op_attributes, "H", "UNFUSED_MOE Attr"),
        "intermediate": _positive_int(
            op_attributes, "INT", "UNFUSED_MOE Attr"),
    }
    raw_counts = op_attributes["tok_per_expert"]
    if not isinstance(raw_counts, list):
        raise TypeError("UNFUSED_MOE Attr.tok_per_expert must be a list")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in raw_counts):
        raise TypeError(
            "UNFUSED_MOE Attr.tok_per_expert must contain integers")
    launch = observed_launch_config(
        metrics, KernelKind.UNFUSED_MOE, shape,
        algorithm="observed_unfused_moe", variant="decomposed")
    return KernelSpec(
        kind=KernelKind.UNFUSED_MOE,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant="decomposed",
        launch=launch,
        attributes={
            "tokens_per_expert": tuple(raw_counts),
            "expert_weight_tile_rows": None,
        },
    )


def _fused_moe_projection(tiling, runtime_name: str, grid_size: int,
                          tensor_ordinal: Optional[int]) -> str:
    lowered = runtime_name.lower()
    if lowered.endswith("_w1") or "_w1" in lowered:
        return "w1"
    if lowered.endswith("_w2") or "_w2" in lowered:
        return "w2"
    if tiling is not None:
        matches = [
            name for name in ("w1", "w2")
            if _positive_int(
                tiling[name], "grid", f"FUSED_MOE {name} tiling")
            == grid_size]
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 2 and tensor_ordinal in (0, 1):
            return ("w1", "w2")[tensor_ordinal]
    raise ValueError(
        "cannot determine fused-MoE projection from structured observation; "
        "pass tensor_ordinal when the two observed grids collide")


def _normalize_fused_moe(op_attributes: Mapping, kdata: Mapping,
                         hardware_id: str, hardware,
                         tensor_ordinal: Optional[int],
                         kernel_ordinal: Optional[int],
                         moe_reduce_ordinal: Optional[int], *,
                         profiler_metrics: Optional[Mapping] = None
                         ) -> KernelSpec:
    metrics = (
        extract_ncu_metrics(kdata)
        if profiler_metrics is None else profiler_metrics)
    if not isinstance(metrics, Mapping):
        raise TypeError("FUSEDMOE profiler_metrics must be a mapping")
    runtime_name = metrics["kernel_name"]
    if is_non_tensor(
            "FUSEDMOE", runtime_name,
            ascend_op_type=metrics["op_type"]):
        return _normalize_moe_aux(
            op_attributes, kdata, hardware_id, hardware, kernel_ordinal,
            moe_reduce_ordinal, profiler_metrics=metrics)
    memory_model = hardware.kernel_memory_model
    if memory_model == "npu_tile":
        projection = _fused_moe_projection(
            None, runtime_name, metrics["grid_size"], tensor_ordinal)
    elif memory_model == "gpu_mshr":
        projection = _fused_moe_projection(
            op_attributes["moe_tiling"], runtime_name,
            metrics["grid_size"], tensor_ordinal)
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")
    shape = {
        "tokens": _positive_int(op_attributes, "TOK", "FUSED_MOE Attr"),
        "experts": _positive_int(op_attributes, "E", "FUSED_MOE Attr"),
        "topk": _positive_int(op_attributes, "TOPK", "FUSED_MOE Attr"),
        "hidden": _positive_int(op_attributes, "H", "FUSED_MOE Attr"),
        "intermediate": _positive_int(
            op_attributes, "INT", "FUSED_MOE Attr"),
    }
    op_type = metrics["op_type"]
    if op_type == "cube":
        algorithm = "cann_grouped_matmul"
    elif op_type is None:
        algorithm = "triton_grouped_gemm"
    else:
        raise ValueError(
            f"unsupported FUSED_MOE tensor profiler op_type: {op_type!r}")
    launch = observed_launch_config(
        metrics, KernelKind.FUSED_MOE, shape,
        algorithm=algorithm,
        variant=projection)
    parameters = {
        "group_size_m": 0.0,
        "padded_rows": 0.0,
        "mte2_tile_bytes": 0.0,
        "weight_tile_rows": 0.0,
        "operand_replay_a": 1.0,
        "operand_replay_b": 1.0,
    }
    raw_counts = op_attributes["tok_per_expert"]
    if not isinstance(raw_counts, list):
        raise TypeError("FUSED_MOE Attr.tok_per_expert must be a list")
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in raw_counts):
        raise TypeError(
            "FUSED_MOE Attr.tok_per_expert must contain integers")
    attributes = {
        "tokens_per_expert": tuple(raw_counts),
        "local_tokens_per_expert": tuple(raw_counts),
        "expert_parallel_size": 1,
        "expert_rank": 0,
        "owned_experts": tuple(range(shape["experts"])),
        "expert_weight_tile_rows": None,
    }
    if op_type == "cube":
        active_counts = [
            count for count in attributes["tokens_per_expert"] if count > 0]
        if not active_counts:
            raise ValueError("FUSED_MOE routing must contain an active expert")
        canonical_launch = select_launch_config(
            hardware_id,
            KernelKind.FUSED_MOE,
            shape,
            hardware.kernel_dtype,
            projection,
            {
                "launch_policy": "analytical_launch",
                "expert_parallel_size": 1,
            },
            hardware,
        )
        tiling = canonical_launch.tiling
        parameters = dict(canonical_launch.parameters)
    elif op_type is None:
        if projection in ("w1", "w2"):
            tiling_projection = projection
        else:
            raise ValueError(
                f"unknown FUSED_MOE projection: {projection!r}")
        tiling_data = op_attributes["moe_tiling"][tiling_projection]
        tiling = TilingSpec(tile_shape={
            "m": _positive_int(tiling_data, "BM", "FUSED_MOE tiling"),
            "n": _positive_int(tiling_data, "BN", "FUSED_MOE tiling"),
            "k": _positive_int(tiling_data, "BK", "FUSED_MOE tiling"),
        })
        parameters.update({
            "group_size_m": _positive_int(
                tiling_data, "GSM", "FUSED_MOE tiling"),
            "padded_rows": _positive_int(
                tiling_data, "EM", "FUSED_MOE tiling"),
        })
        if hardware_id == "metax_c500":
            from src.kernel.launch_config.metax_c500.fused_moe_launch import (
                select_fused_moe_operand_replay,
                select_fused_moe_w2_routed_weight_program,
            )
            from src.kernel.model_contract import (
                SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER,
            )
            operand_replay = select_fused_moe_operand_replay(
                int(tiling.tile_shape["m"]))
            parameters["operand_replay_a"] = operand_replay
            parameters["operand_replay_b"] = operand_replay
            parameters[
                SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER
            ] = select_fused_moe_w2_routed_weight_program(projection)
    else:
        raise ValueError(
            f"unhandled FUSED_MOE tensor profiler op_type: {op_type!r}")
    launch = dataclasses.replace(
        launch, tiling=tiling, parameters=parameters)
    return KernelSpec(
        kind=KernelKind.FUSED_MOE,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant=projection,
        launch=launch,
        attributes=attributes,
    )


def _moe_aux_variant(runtime_name: str, ascend_op_type: Optional[str]) -> str:
    lowered = runtime_name.lower()
    if ascend_op_type:
        if "swiglu" in lowered:
            return "moe_swiglu"
        if "initrouting" in lowered:
            return "moe_init_routing"
        if "finalizerouting" in lowered:
            return "moe_finalize_routing"
        if "computeexperttokens" in lowered:
            return "moe_compute_expert_tokens"
        if "transpose" in lowered:
            return "moe_transpose"
        if "cast" in lowered:
            return "moe_cast"
        if "range" in lowered or "arange" in lowered:
            return "moe_range"
        if "softmax" in lowered:
            return "moe_softmax"
        if "topk" in lowered:
            return "moe_topk"
        if "gating" in lowered:
            return "moe_gating"
        raise ValueError(f"unknown Huawei MoE auxiliary implementation: {runtime_name}")
    if "act_and_mul" in lowered:
        return "moe_act_and_mul"
    if "silu_kernel" in lowered:
        return "moe_silu"
    if "mulfunctor" in lowered:
        return "moe_mul"
    # PyTorch's CUDA reduction symbol also embeds ``sum_functor``.  Match the
    # implementation entry point first so default top-k reductions do not get
    # confused with vLLM's dedicated ``moe_sum`` kernel or MetaX's distinct
    # TensorIterator implementation.
    if "reduce_kernel" in lowered:
        return "moe_reduce"
    if "moe_sum" in lowered or "sum_functor" in lowered:
        return "moe_sum"
    if "moe_align" in lowered:
        if "opt_128" in lowered:
            return "moe_align_opt_128"
        if "small_batch" in lowered:
            return "moe_align_small_batch"
        return "moe_align"
    if "count_and_sort" in lowered:
        return "moe_count_and_sort"
    if "vectorized_elementwise" in lowered:
        return "moe_fill"
    if "enable_if" in lowered:
        return "moe_enable_if"
    if "softmax" in lowered:
        return "moe_softmax"
    if "topk" in lowered:
        return "moe_topk"
    if "gating" in lowered:
        return "moe_gating"
    raise ValueError(f"unknown MoE auxiliary implementation: {runtime_name}")


def _normalize_a800_moe_reduce_attributes(
        shape: Mapping[str, int], runtime_name: str, launch, hardware,
        moe_reduce_ordinal: Optional[int]) -> Mapping[str, float]:
    """Bind one observed launch to a PyTorch 2.8 output-slice fraction."""

    if (hardware.kernel_dtype != "bf16"
            or runtime_name != _A800_PYTORCH_2_8_BF16_REDUCE_SYMBOL):
        raise ValueError(
            "A800 PyTorch moe_reduce observation does not match the retained "
            "BF16 PyTorch 2.8 reduction symbol")
    fractions = select_fused_moe_reduce_work_fractions(
        "nvidia_a800", shape, hardware)
    if moe_reduce_ordinal is None:
        raise ValueError(
            "A800 PyTorch moe_reduce requires its physical reduce ordinal")
    if not 0 <= moe_reduce_ordinal < len(fractions):
        raise ValueError(
            "A800 PyTorch moe_reduce ordinal is outside the source-derived "
            f"partition: ordinal={moe_reduce_ordinal}, count={len(fractions)}")
    fraction = fractions[moe_reduce_ordinal]
    expected = select_launch_config(
        "nvidia_a800", KernelKind.VECTOR, shape,
        hardware.kernel_dtype, "moe_reduce",
        {"work_fraction": fraction}, hardware)
    if launch.grid != expected.grid or launch.block != expected.block:
        raise ValueError(
            "A800 PyTorch moe_reduce launch disagrees with its ordered "
            "source-derived output slice: "
            f"ordinal={moe_reduce_ordinal}, "
            f"grid={launch.grid}, block={launch.block}, "
            f"expected_grid={expected.grid}, "
            f"expected_block={expected.block}")
    return {"work_fraction": fraction}


def _normalize_ascend_moe_compute_expert_tokens_launch(
        shape: Mapping[str, int], metrics: Mapping, observed_launch,
        hardware):
    """Bind the observed helper to CANN's source-selected program."""

    canonical_launch = select_launch_config(
        "ascend_910b", KernelKind.VECTOR, shape,
        hardware.kernel_dtype, "moe_compute_expert_tokens",
        {"launch_policy": "injected_launch"}, hardware)
    tiling_key = int(canonical_launch.parameters["cann_tiling_key"])
    expected_runtime_name = (
        "MoeComputeExpertTokens_"
        f"{_ASCEND_910B_MOE_COMPUTE_EXPERT_TOKENS_CODE_HASH}_"
        f"{tiling_key}_mix_aiv")
    _require_source_bound_vector_observation(
        "ascend_910b", "moe_compute_expert_tokens",
        metrics["kernel_name"], observed_launch, canonical_launch,
        expected_runtime_name)
    return dataclasses.replace(
        observed_launch,
        algorithm=canonical_launch.algorithm,
        parameters=canonical_launch.parameters,
        execution_variant=canonical_launch.execution_variant,
        source_program=canonical_launch.source_program,
    )


def _normalize_ascend_moe_source_launch(
        shape: Mapping[str, int], metrics: Mapping, observed_launch,
        variant: str, attributes: Mapping[str, object], hardware):
    """Require one retained CANN MoE helper to match its source tiler."""

    selection_attributes = {
        "launch_policy": "analytical_launch",
        **attributes,
    }
    canonical_launch = select_launch_config(
        "ascend_910b", KernelKind.VECTOR, shape,
        hardware.kernel_dtype, variant, selection_attributes, hardware)
    if (observed_launch.grid != canonical_launch.grid
            or observed_launch.block != canonical_launch.block):
        raise ValueError(
            f"Ascend {variant} profiler geometry disagrees with the CANN "
            f"source selector: expected grid={canonical_launch.grid}, "
            f"block={canonical_launch.block}, actual "
            f"grid={observed_launch.grid}, block={observed_launch.block}")
    return dataclasses.replace(
        observed_launch,
        algorithm=canonical_launch.algorithm,
        parameters=canonical_launch.parameters,
        execution_variant=canonical_launch.execution_variant,
        source_program=canonical_launch.source_program,
    )


def _normalize_metax_moe_fill_attributes(
        shape: Mapping[str, int], metrics: Mapping, launch,
        hardware_id: str, kernel_ordinal: Optional[int]) -> Mapping[str, str]:
    """Bind each native int32 Fill launch to its source-level output tensor."""

    if hardware_id != "metax_c500":
        raise ValueError(
            "VECTOR/moe_fill observation semantics are version-bound to "
            f"metax_c500, got {hardware_id!r}")
    if kernel_ordinal is None:
        raise ValueError(
            "metax_c500 VECTOR/moe_fill requires the physical kernel ordinal")
    if kernel_ordinal >= len(MOE_FILL_ROLES):
        raise ValueError(
            "metax_c500 VECTOR/moe_fill must be one of the first three "
            f"physical kernels, got ordinal {kernel_ordinal}")
    fill_role = MOE_FILL_ROLES[kernel_ordinal]

    block_m = 16 if shape["tokens"] <= shape["experts"] else 64
    max_padded = (
        shape["tokens"] * shape["topk"]
        + shape["experts"] * (block_m - 1)
    )
    elements_by_role = {
        "sorted_token_ids": max_padded,
        "expert_ids": (max_padded + block_m - 1) // block_m,
        "num_tokens_post_padded": 1,
    }
    elements = elements_by_role[fill_role]

    runtime_name = metrics["kernel_name"]
    standard_template = (
        "vectorized_elementwise_kernel<4, at::native::FillFunctor<int>")
    optimized_template = (
        "vectorized_elementwise_kernel_nullary_opt<1024, 4, int, "
        "at::native::FillFunctor<int>")
    if standard_template in runtime_name:
        expected_block_x = 256
    elif optimized_template in runtime_name:
        expected_block_x = 1024
    else:
        raise ValueError(
            "metax_c500 VECTOR/moe_fill runtime symbol does not name a "
            "source-supported int32 vec4 Fill template")
    if launch.block != (expected_block_x, 1, 1):
        raise ValueError(
            "metax_c500 VECTOR/moe_fill block disagrees with its runtime "
            f"template: expected={(expected_block_x, 1, 1)}, "
            f"actual={launch.block}")
    expected_grid = (
        (elements + expected_block_x * 4 - 1)
        // (expected_block_x * 4),
        1,
        1,
    )
    if launch.grid != expected_grid:
        raise ValueError(
            "metax_c500 VECTOR/moe_fill grid disagrees with the source "
            f"tensor length for {fill_role}: expected={expected_grid}, "
            f"actual={launch.grid}")
    return {"fill_role": fill_role}


# vLLM 0.30.0 ``_prepare_expert_assignment`` (vllm/model_executor/layers/
# fused_moe/fused_moe.py): when ``num_tokens * top_k * 4 <= num_experts`` and
# no expert map is set, ``moe_align_block_size`` is skipped and the only
# helper is ``torch.full((1,), topk_ids.numel() * BLOCK_SIZE_M, int32)`` --
# the ``num_tokens_post_padded`` scalar.  PyTorch's vectorized FillFunctor
# launch uses ``num_threads() = C10_WARP_SIZE * 4`` = 128 threads and one CTA
# per 512 elements.
_H200_VLLM030_NAIVE_ASSIGNMENT_SPARSITY = 4
_H200_TORCH_FILL_SYMBOL_PREFIX = (
    "void at::native::vectorized_elementwise_kernel<(int)4, "
    "at::native::FillFunctor<int>, ")
_H200_TORCH_VECTORIZED_THREADS = 128
_H200_TORCH_VECTORIZED_ELEMENTS_PER_THREAD = 4


def _normalize_h200_moe_fill_attributes(
        shape: Mapping[str, int], metrics: Mapping, launch,
        kernel_ordinal: Optional[int]) -> Mapping[str, str]:
    """Bind the H200 vLLM 0.30 naive-assignment Fill to its source tensor."""

    naive = (
        shape["tokens"] * shape["topk"]
        * _H200_VLLM030_NAIVE_ASSIGNMENT_SPARSITY <= shape["experts"])
    if not naive:
        raise ValueError(
            "nvidia_h200 VECTOR/moe_fill appears only on the vLLM 0.30 "
            "naive block-assignment path (TOK*TOPK*4 <= E)")
    if kernel_ordinal != 0:
        raise ValueError(
            "nvidia_h200 VECTOR/moe_fill must be the first physical kernel, "
            f"got ordinal {kernel_ordinal}")
    if not metrics["kernel_name"].startswith(_H200_TORCH_FILL_SYMBOL_PREFIX):
        raise ValueError(
            "nvidia_h200 VECTOR/moe_fill runtime symbol is not the int32 "
            "vec4 FillFunctor template")
    elements = 1
    per_cta = (_H200_TORCH_VECTORIZED_THREADS
               * _H200_TORCH_VECTORIZED_ELEMENTS_PER_THREAD)
    expected_grid = ((elements + per_cta - 1) // per_cta, 1, 1)
    expected_block = (_H200_TORCH_VECTORIZED_THREADS, 1, 1)
    if launch.grid != expected_grid or launch.block != expected_block:
        raise ValueError(
            "nvidia_h200 VECTOR/moe_fill geometry disagrees with the "
            f"one-element torch.full launch: expected {expected_grid}/"
            f"{expected_block}, actual {launch.grid}/{launch.block}")
    return {"fill_role": "num_tokens_post_padded"}


# vLLM 0.30.0 csrc/libtorch_stable/moe/moe_align_sum_kernels.cu ``moe_sum``
# (non-pad-aware path, BF16, contiguous hidden): 16-byte vectors
# (VEC = 8 BF16), block 256, grid min(ceil(TOK * H / 8 / 256), 65535);
# ``moe_sum_vec_kernel<..., TOPK, false>`` for TOPK in {1, 2, 4, 6, 8, 9},
# otherwise ``moe_sum_vec_dynamic_kernel``.
_H200_MOE_SUM_STATIC_TOPK = (1, 2, 4, 6, 8, 9)
_H200_MOE_SUM_BLOCK = 256
_H200_MOE_SUM_VEC_BF16 = 8
_H200_MOE_SUM_MAX_GRID = 65535


def _validate_h200_moe_sum_launch(
        shape: Mapping[str, int], metrics: Mapping, launch, hardware) -> None:
    if hardware.kernel_dtype != "bf16":
        raise ValueError("nvidia_h200 moe_sum source binding is BF16 only")
    hidden = shape["hidden"]
    if hidden % _H200_MOE_SUM_VEC_BF16:
        raise NotImplementedError(
            "nvidia_h200 moe_sum scalar (non-vectorized) path is not bound")
    topk = shape["topk"]
    if topk in _H200_MOE_SUM_STATIC_TOPK:
        expected_prefix = (
            "void vllm::moe::moe_sum_vec_kernel<c10::BFloat16, int, "
            f"(int){topk}, (bool)0>(")
    else:
        expected_prefix = (
            "void vllm::moe::moe_sum_vec_dynamic_kernel<c10::BFloat16, int, "
            "(bool)0>(")
    if not metrics["kernel_name"].startswith(expected_prefix):
        raise ValueError(
            "nvidia_h200 moe_sum runtime symbol disagrees with the vLLM 0.30 "
            f"TOPK={topk} dispatch")
    total = shape["tokens"] * (hidden // _H200_MOE_SUM_VEC_BF16)
    expected_grid = (
        min((total + _H200_MOE_SUM_BLOCK - 1) // _H200_MOE_SUM_BLOCK,
            _H200_MOE_SUM_MAX_GRID), 1, 1)
    if (launch.grid != expected_grid
            or launch.block != (_H200_MOE_SUM_BLOCK, 1, 1)):
        raise ValueError(
            "nvidia_h200 moe_sum geometry disagrees with the vLLM 0.30 "
            f"source: expected {expected_grid}/({_H200_MOE_SUM_BLOCK}, 1, 1),"
            f" actual {launch.grid}/{launch.block}")


def _normalize_moe_aux(op_attributes: Mapping, kdata: Mapping,
                       hardware_id: str, hardware,
                       kernel_ordinal: Optional[int],
                       moe_reduce_ordinal: Optional[int], *,
                       profiler_metrics: Optional[Mapping] = None
                       ) -> KernelSpec:
    metrics = (
        extract_ncu_metrics(kdata)
        if profiler_metrics is None else profiler_metrics)
    if not isinstance(metrics, Mapping):
        raise TypeError("MoE auxiliary profiler_metrics must be a mapping")
    variant = _moe_aux_variant(metrics["kernel_name"], metrics["op_type"])
    shape = {
        "tokens": _positive_int(op_attributes, "TOK", "FUSED_MOE Attr"),
        "experts": _positive_int(op_attributes, "E", "FUSED_MOE Attr"),
        "topk": _positive_int(op_attributes, "TOPK", "FUSED_MOE Attr"),
        "hidden": _positive_int(op_attributes, "H", "FUSED_MOE Attr"),
        "intermediate": _positive_int(
            op_attributes, "INT", "FUSED_MOE Attr"),
        "dtype_bytes": 2,
    }
    launch = observed_launch_config(
        metrics, KernelKind.VECTOR, shape,
        algorithm="observed_moe_aux", variant=variant)
    attributes = {}
    if variant == "moe_fill" and hardware_id == "nvidia_h200":
        attributes = _normalize_h200_moe_fill_attributes(
            shape, metrics, launch, kernel_ordinal)
    elif variant == "moe_fill":
        attributes = _normalize_metax_moe_fill_attributes(
            shape, metrics, launch, hardware_id, kernel_ordinal)
    elif variant == "moe_sum":
        if hardware_id == "nvidia_h200":
            _validate_h200_moe_sum_launch(shape, metrics, launch, hardware)
        elif hardware_id == "nvidia_a800":
            if (hardware.kernel_dtype != "bf16"
                    or metrics["kernel_name"]
                    != _a800_moe_sum_symbol(shape["topk"])):
                raise ValueError(
                    "A800 moe_sum observation does not match the exact "
                    "retained vLLM 0.11.0 BF16 symbol")
        elif hardware_id == "metax_c500":
            pass
        else:
            raise ValueError(
                "moe_sum observation normalization is defined only for "
                "nvidia_a800 and metax_c500, got "
                f"{hardware_id!r}")
        # MetaX TensorIterator splits reductions once the logical input crosses
        # its signed 30-bit indexing window.  Both observed invocations retain
        # the same launch grid, but the independent per-invocation counters
        # show that each reads and writes exactly half of the logical tensor.
        logical_input_elements = (
            shape["tokens"] * shape["topk"] * shape["hidden"])
        if hardware_id == "metax_c500" and logical_input_elements > 2 ** 30:
            work_fraction = 0.5
        else:
            work_fraction = 1.0
        attributes = {"work_fraction": work_fraction}
    elif variant == "moe_reduce":
        require_moe_aux_hardware(variant, hardware_id)
        attributes = _normalize_a800_moe_reduce_attributes(
            shape, metrics["kernel_name"], launch, hardware,
            moe_reduce_ordinal)
    elif variant == "moe_swiglu":
        require_moe_aux_hardware(variant, hardware_id)
        if metrics["kernel_name"] != "SwiGlu_3_high_performance_27":
            raise ValueError(
                "Ascend moe_swiglu observation does not match the exact "
                "SwiGlu_3_high_performance_27 source symbol")
        routed_rows = shape["tokens"] * shape["topk"]
        attributes = {"routed_rows": routed_rows}
        source_launch = select_launch_config(
            hardware_id,
            KernelKind.VECTOR,
            shape,
            hardware.kernel_dtype,
            variant,
            {
                "launch_policy": "analytical_launch",
                "routed_rows": routed_rows,
            },
            hardware,
        )
        if (launch.grid != source_launch.grid
                or launch.block != source_launch.block):
            raise ValueError(
                "Ascend moe_swiglu profiler geometry disagrees with the "
                "CANN source selector: "
                f"expected grid={source_launch.grid}, "
                f"block={source_launch.block}, actual grid={launch.grid}, "
                f"block={launch.block}")
        launch = source_launch
    elif variant == "moe_compute_expert_tokens":
        require_moe_aux_hardware(variant, hardware_id)
        raw_counts = op_attributes["tok_per_expert"]
        if (not isinstance(raw_counts, list)
                or any(isinstance(value, bool)
                       or not isinstance(value, int)
                       for value in raw_counts)):
            raise TypeError(
                "FUSED_MOE Attr.tok_per_expert must contain integers")
        attributes = {"tokens_per_expert": tuple(raw_counts)}
        launch = _normalize_ascend_moe_compute_expert_tokens_launch(
            shape, metrics, launch, hardware)
    elif variant == "moe_cast":
        require_moe_aux_hardware(variant, hardware_id)
        roles = [
            role for role in MOE_CAST_ROLES
            if metrics["kernel_name"]
            == _ASCEND_910B_MOE_CAST_RUNTIME_BY_ROLE[role]
        ]
        if len(roles) != 1:
            raise ValueError(
                "Ascend moe_cast runtime symbol does not identify exactly "
                f"one source role: {metrics['kernel_name']!r}")
        attributes = {"cast_role": roles[0]}
        launch = _normalize_ascend_moe_source_launch(
            shape, metrics, launch, variant, attributes, hardware)
    elif variant in {
            "moe_init_routing", "moe_finalize_routing",
            "moe_transpose", "moe_range"}:
        require_moe_aux_hardware(variant, hardware_id)
        launch = _normalize_ascend_moe_source_launch(
            shape, metrics, launch, variant, attributes, hardware)
    elif variant in {"moe_silu", "moe_mul"}:
        require_moe_aux_hardware(variant, hardware_id)
        logical_elements = (
            shape["tokens"] * shape["topk"] * shape["intermediate"])
        work_fraction = 0.5 if logical_elements > 2 ** 29 else 1.0
        attributes = {"work_fraction": work_fraction}
        launch_attributes = {
            "launch_policy": "injected_launch",
            **attributes,
        }
        source_launch = select_launch_config(
            hardware_id, KernelKind.VECTOR, shape,
            hardware.kernel_dtype, variant, launch_attributes, hardware)
        if (launch.grid != source_launch.grid
                or launch.block != source_launch.block):
            raise ValueError(
                f"MetaX {variant} observation disagrees with its "
                "source-derived TensorIterator partition")
        launch = dataclasses.replace(
            launch,
            algorithm=source_launch.algorithm,
            execution_variant=source_launch.execution_variant,
        )
    elif variant in {
            "moe_softmax",
            "moe_topk", "moe_gating", "moe_act_and_mul",
            "moe_align", "moe_align_opt_128",
            "moe_align_small_batch", "moe_count_and_sort",
            "moe_enable_if"}:
        pass
    else:
        raise ValueError(
            f"unhandled normalized MoE auxiliary variant: {variant!r}")
    return KernelSpec(
        kind=KernelKind.VECTOR,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant=variant,
        launch=launch,
        attributes=attributes,
    )


def _validate_observed_vector_element_count(
        op_attributes: Mapping, label: str, canonical_count: int,
        legacy_count: int) -> None:
    """Accept only the documented legacy or canonical vector count schema.

    Existing protected datasets stored ``tok * hidden`` for SILU and KV-cache,
    while their structured fields describe the actual launched shape.  Resolve
    that known schema difference here at the observation boundary so production
    predictors never select a shape based on whichever keys happen to exist.
    """

    observed = _positive_int(
        op_attributes, "element_count", f"{label} Attr")
    allowed = {canonical_count, legacy_count}
    if observed not in allowed:
        raise ValueError(
            f"{label} Attr.element_count {observed} matches neither canonical "
            f"structured count {canonical_count} nor documented legacy count "
            f"{legacy_count}")


def _validate_canonical_vector_element_count(
        op_attributes: Mapping, label: str, canonical_count: int) -> None:
    """Require the observation count to match its structured shape exactly."""

    observed = _positive_int(
        op_attributes, "element_count", f"{label} Attr")
    if observed != canonical_count:
        raise ValueError(
            f"{label} Attr.element_count {observed} disagrees with canonical "
            f"structured count {canonical_count}")


def _require_source_bound_vector_observation(
        hardware_id: str, variant: str, runtime_name: str,
        observed_launch, canonical_launch, expected_runtime_name: str) -> None:
    """Prove a runtime launch before admitting source-bound parameters."""

    if runtime_name != expected_runtime_name:
        raise ValueError(
            f"{hardware_id} {variant} runtime symbol disagrees with the "
            "source-bound implementation: "
            f"expected={expected_runtime_name!r}, observed={runtime_name!r}")
    if (observed_launch.grid != canonical_launch.grid
            or observed_launch.block != canonical_launch.block):
        raise ValueError(
            f"{hardware_id} {variant} profiler geometry disagrees with the "
            "source-bound launch: "
            f"expected={canonical_launch.grid}/{canonical_launch.block}, "
            f"observed={observed_launch.grid}/{observed_launch.block}")


def _normalize_vector(op_type: str, op_attributes: Mapping, kdata: Mapping,
                      hardware_id: str, hardware) -> KernelSpec:
    metrics = extract_ncu_metrics(kdata)
    typed_a800_rope = False
    typed_c500_rope = False
    typed_ascend_qkv_rmsnorm_rope = False
    if op_type == "NORM":
        kind = KernelKind.VECTOR
        variant = _non_empty_string(
            op_attributes, "op", "NORM Attr").lower()
    elif op_type == "KV_CACHE":
        kind = KernelKind.KV_CACHE
        variant = "paged_kv_cache"
    elif op_type in (
            "ROTARY", "ACTIVATION", "ELEMENTWISE",
            "REDUCE", "GATHER"):
        kind = KernelKind.VECTOR
        variants = {
            "ROTARY": "rotary",
            "ACTIVATION": "activation",
            "ELEMENTWISE": "elementwise",
            "REDUCE": "reduce",
            "GATHER": "gather",
        }
        variant = variants[op_type]
        if op_type == "ACTIVATION":
            variant = _non_empty_string(
                op_attributes, "op", "ACTIVATION Attr").lower()
            if variant not in ("activation", "silu"):
                raise ValueError(
                    f"unsupported ACTIVATION Attr.op: {variant!r}")
    else:
        raise ValueError(f"unknown VECTOR observation type: {op_type!r}")
    if op_type == "NORM":
        shape = {
            "rows": _positive_int(op_attributes, "tok", "NORM Attr"),
            "hidden": _positive_int(
                op_attributes, "hidden", "NORM Attr"),
        }
    elif op_type == "ROTARY":
        operator = _non_empty_string(
            op_attributes, "op", "ROTARY Attr").lower()
        if operator not in ("rope", "rotary", "qkv_rmsnorm_rope"):
            raise ValueError(
                f"unsupported ROTARY Attr.op: {operator!r}")
        seq_len = _positive_int(op_attributes, "tok", "ROTARY Attr")
        num_query_heads = _positive_int(
            op_attributes, "num_heads", "ROTARY Attr")
        num_kv_heads = _positive_int(
            op_attributes, "num_kv_heads", "ROTARY Attr")
        head_dim = _positive_int(
            op_attributes, "head_dim", "ROTARY Attr")
        typed_ascend_qkv_rmsnorm_rope = (
            operator == "qkv_rmsnorm_rope")
        if typed_ascend_qkv_rmsnorm_rope:
            if (hardware_id != "ascend_910b"
                    or hardware.kernel_dtype != "bf16"):
                raise ValueError(
                    "qkv_rmsnorm_rope observation requires the pinned "
                    "Ascend 910B BF16 program")
            rotary_dim = _positive_int(
                op_attributes, "rotary_dim", "ROTARY Attr")
            canonical_count = (
                seq_len * (num_query_heads + 2 * num_kv_heads) * head_dim)
            _validate_canonical_vector_element_count(
                op_attributes, "QKV_RMSNORM_ROPE", canonical_count)
            shape = {
                "seq_len": seq_len,
                "num_query_heads": num_query_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
                "rotary_dim": rotary_dim,
            }
            variant = operator
        else:
            # Dataset element_count historically denotes the Q tensor. K is
            # represented independently by num_kv_heads and is included when
            # the production KernelSpec derives its total rotary work.
            canonical_count = seq_len * num_query_heads * head_dim
            _validate_canonical_vector_element_count(
                op_attributes, "ROTARY", canonical_count)
            shape = {
                "seq_len": seq_len,
                "num_query_heads": num_query_heads,
                "num_kv_heads": num_kv_heads,
                "head_dim": head_dim,
            }
        # The A800 dataset runner is source-pinned to vLLM's full-dimension
        # NeoX call with both Q and K.  Only the matching BF16 template identity
        # is enriched into the typed ``rope`` contract; all other ROTARY
        # observations retain the cross-hardware ``rotary`` semantics and do
        # not borrow its SASS lowering.
        source_identity = (
            "vllm::rotary_embedding_kernel<c10::BFloat16, (bool)1>")
        typed_a800_rope = (
            hardware_id == "nvidia_a800"
            and hardware.kernel_dtype == "bf16"
            and operator == "rope"
            and metrics["kernel_name"].count(source_identity) == 1
        )
        c500_source_identity = (
            "void vllm::rotary_embedding_kernel<c10::BFloat16, true>"
            "(long const*, c10::BFloat16*, c10::BFloat16*, "
            "c10::BFloat16 const*, int, long, long, int, int, int)")
        typed_c500_rope = (
            hardware_id == "metax_c500"
            and hardware.kernel_dtype == "bf16"
            and operator == "rope"
            and metrics["kernel_name"] == c500_source_identity
        )
        if typed_a800_rope or typed_c500_rope:
            variant = "rope"
            shape["rotary_dim"] = head_dim
    elif op_type == "ACTIVATION":
        tokens = _positive_int(op_attributes, "tok", "ACTIVATION Attr")
        intermediate = _positive_int(
            op_attributes, "intermediate", "ACTIVATION Attr")
        canonical_count = tokens * intermediate
        legacy_count = tokens * _positive_int(
            op_attributes, "hidden", "ACTIVATION Attr")
        _validate_observed_vector_element_count(
            op_attributes, "ACTIVATION", canonical_count, legacy_count)
        shape = {
            "tok": tokens,
            "intermediate": intermediate,
            "element_count": canonical_count,
        }
    elif op_type == "KV_CACHE":
        batch = _positive_int(op_attributes, "batch", "KV_CACHE Attr")
        if batch != 1:
            raise ValueError(
                "KV_CACHE Attr.batch is a legacy label, not a physical "
                f"dimension; expected 1, got {batch}")
        seq_len = _positive_int(op_attributes, "tok", "KV_CACHE Attr")
        num_kv_heads = _positive_int(
            op_attributes, "num_kv_heads", "KV_CACHE Attr")
        head_dim = _positive_int(
            op_attributes, "head_dim", "KV_CACHE Attr")
        canonical_count = seq_len * num_kv_heads * head_dim
        legacy_count = seq_len * _positive_int(
            op_attributes, "hidden", "KV_CACHE Attr")
        _validate_observed_vector_element_count(
            op_attributes, "KV_CACHE", canonical_count, legacy_count)
        shape = {
            "seq_len": seq_len,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        }
    elif op_type in ("ELEMENTWISE", "REDUCE", "GATHER"):
        shape = {"element_count": _positive_int(
            op_attributes, "element_count", f"{op_type} Attr")}
    else:
        raise ValueError(f"unknown VECTOR observation type: {op_type!r}")
    launch = observed_launch_config(
        metrics, kind, shape, algorithm="observed_vector", variant=variant)
    attributes = {
        name: value for name, value in op_attributes.items()
        if name in (
            "reads", "writes", "reduction_factor",
            "gather_factor", "index_count", "index_bytes", "scattered")
    }
    if typed_ascend_qkv_rmsnorm_rope:
        bias = op_attributes["bias"]
        is_neox = op_attributes["is_neox"]
        if not isinstance(bias, bool) or not isinstance(is_neox, bool):
            raise TypeError(
                "QKV_RMSNORM_ROPE bias and is_neox must be boolean")
        attributes = {"bias": bias, "is_neox": is_neox}
    elif typed_a800_rope or typed_c500_rope:
        attributes = {"is_neox": True, "key_present": True}
    elif variant == "elementwise":
        # The Vector dataset runner executes a binary add/mul: two input reads
        # and one output write. Keep the semantic traffic contract independent
        # of profiler counters.
        if "op2" in op_attributes:
            operation = _non_empty_string(
                op_attributes, "op2", "ELEMENTWISE Attr").lower()
        elif (hardware_id == "samsung_hbm_pim"
              and op_attributes["op"] == "elementwise"):
            # The protected historical UniNDP dataset predates the add/mul
            # label, but its ``elewise`` primitive explicitly contains one
            # ordinary arithmetic stage.  Both add and mul map to the same
            # single-operation demand, so normalize that legacy simulator
            # contract without changing target-hardware behavior.
            operation = "add"
        else:
            raise ValueError(
                "ELEMENTWISE Attr requires explicit op2 operation identity")
        if operation not in ("add", "mul"):
            raise ValueError(
                f"unsupported ELEMENTWISE Attr.op2: {operation!r}")
        attributes = {
            "reads": 2, "writes": 1, "operation": operation}
        if hardware_id == "nvidia_a800":
            expected_symbol = _A800_ELEMENTWISE_RUNTIME_SYMBOLS[operation]
            if metrics["kernel_name"] != expected_symbol:
                raise ValueError(
                    "A800 ELEMENTWISE runtime symbol disagrees with the "
                    "typed PyTorch vec4 implementation: "
                    f"expected={expected_symbol!r}, "
                    f"observed={metrics['kernel_name']!r}")
            variant = "elementwise_pytorch_vec4"
            launch = dataclasses.replace(
                launch, execution_variant=variant)
    elif variant == "reduce":
        attributes = {"reduction_factor": op_attributes["reduction_factor"]}
    elif variant == "gather":
        attributes = {
            name: op_attributes[name]
            for name in (
                "gather_factor", "index_count", "index_bytes", "scattered")
        }
    elif variant in (
            "rmsnorm", "add_rmsnorm", "rotary", "rope", "activation",
            "silu", "paged_kv_cache", "qkv_rmsnorm_rope"):
        if variant == "paged_kv_cache":
            # All three pinned runners use a paged cache with the same storage
            # dtype as K/V and an arange slot mapping.  These are runner/source
            # semantics, not profiler-duration observations.
            attributes = PAGED_KV_CACHE_ATTRIBUTES
    else:
        raise ValueError(f"unknown normalized VECTOR variant: {variant!r}")
    if variant == "silu":
        if hardware_id in _SILU_RUNTIME_SYMBOLS:
            expected_symbol = _SILU_RUNTIME_SYMBOLS[hardware_id]
            if metrics["kernel_name"] != expected_symbol:
                raise ValueError(
                    "SILU observation runtime symbol disagrees with its "
                    f"operator identity: hardware={hardware_id!r}, "
                    f"expected={expected_symbol!r}, "
                    f"observed={metrics['kernel_name']!r}")
    if hardware_id == "nvidia_a800" and variant == "rmsnorm":
        expected_symbol = _A800_VECTOR_RUNTIME_SYMBOLS["rmsnorm"]
        if metrics["kernel_name"] != expected_symbol:
            raise ValueError(
                "A800 RMSNorm observation runtime symbol disagrees with "
                "the retained vLLM BF16 implementation: "
                f"expected={expected_symbol!r}, "
                f"observed={metrics['kernel_name']!r}")
    if hardware_id == "metax_c500" and variant in _METAX_VECTOR_RUNTIME_SYMBOLS:
        expected_symbol = _METAX_VECTOR_RUNTIME_SYMBOLS[variant]
        if metrics["kernel_name"] != expected_symbol:
            raise ValueError(
                "MetaX C500 MACA 3.7.2 Vector observation runtime symbol "
                "disagrees with the retained plugin artifact: "
                f"variant={variant!r}, expected={expected_symbol!r}, "
                f"observed={metrics['kernel_name']!r}")
    if hardware_id == "ascend_910b" and variant == "elementwise":
        operation = attributes["operation"]
        expected_symbols = _ASCEND_ELEMENTWISE_RUNTIME_SYMBOLS[operation]
        if metrics["kernel_name"] not in expected_symbols:
            raise ValueError(
                "Ascend 910B ELEMENTWISE runtime symbol disagrees with the "
                "CANN 8.5.1 BF16 BroadcastSch implementation: "
                f"operation={operation!r}, "
                f"expected_one_of={expected_symbols!r}, "
                f"observed={metrics['kernel_name']!r}")
    # C500 NORM, SiLU, and typed RoPE observations have exact retained source
    # symbols; their observed geometry is checked against the MACA 3.7.2
    # selector before source-semantic parameters are admitted. Other C500
    # observations remain self-contained. A800 and 910B observations below are
    # likewise bound to the same source revision as their production selectors.
    typed_c500_norm = (
        hardware_id == "metax_c500"
        and variant in ("rmsnorm", "add_rmsnorm"))
    typed_c500_silu = (
        hardware_id == "metax_c500" and variant == "silu")
    typed_c500_kv_cache = (
        hardware_id == "metax_c500" and variant == "paged_kv_cache")
    if _is_source_bound_vector_observation(
            hardware_id, variant,
            c500_typed=(typed_c500_norm, typed_c500_silu, typed_c500_rope,
                        typed_c500_kv_cache)):
        selection_attributes = dict(attributes)
        selection_attributes["launch_policy"] = "injected_launch"
        if hardware_id == "ascend_910b" and variant == "silu":
            selection_attributes["injected_grid_size"] = launch.grid_size
        canonical_launch = select_launch_config(
            hardware_id, kind, shape, hardware.kernel_dtype, variant,
            selection_attributes, hardware)
        if hardware_id == "nvidia_a800":
            if variant == "elementwise_pytorch_vec4":
                expected_runtime_name = expected_symbol
            elif variant in _A800_VECTOR_RUNTIME_SYMBOLS:
                expected_runtime_name = _A800_VECTOR_RUNTIME_SYMBOLS[variant]
            else:
                expected_runtime_name = None
            if expected_runtime_name is not None:
                _require_source_bound_vector_observation(
                    hardware_id, variant, metrics["kernel_name"], launch,
                    canonical_launch, expected_runtime_name)
        if hardware_id == "metax_c500" and typed_c500_rope:
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, c500_source_identity)
        if typed_c500_norm:
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, _METAX_VECTOR_RUNTIME_SYMBOLS[variant])
        if typed_c500_silu:
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, _SILU_RUNTIME_SYMBOLS[hardware_id])
        if typed_c500_kv_cache:
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, _METAX_VECTOR_RUNTIME_SYMBOLS[variant])
        if (hardware_id == "ascend_910b"
                and variant == "paged_kv_cache"):
            tiling_key = int(
                canonical_launch.parameters["cann_tiling_key"])
            expected_kernel_name = f"reshape_and_cache_{tiling_key}"
            if metrics["kernel_name"] != expected_kernel_name:
                raise ValueError(
                    "Ascend 910B KV_CACHE source selector disagrees with "
                    "profiler kernel identity: "
                    f"expected={expected_kernel_name!r}, "
                    f"observed={metrics['kernel_name']!r}")
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, expected_kernel_name)
        if hardware_id == "ascend_910b" and variant == "silu":
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, _SILU_RUNTIME_SYMBOLS[hardware_id])
        if typed_ascend_qkv_rmsnorm_rope:
            _require_source_bound_vector_observation(
                hardware_id, variant, metrics["kernel_name"], launch,
                canonical_launch, "split_qkv_rmsnorm_rope_kernel")
        launch_replacements = {
            "parameters": canonical_launch.parameters,
        }
        if typed_c500_norm or typed_ascend_qkv_rmsnorm_rope:
            # Runtime symbol and launch geometry were proved against the
            # retained source program above. The observation label is an
            # evaluator ingestion detail, not a second physical algorithm:
            # publish exactly the launch identity selected for Serving.
            launch_replacements["algorithm"] = canonical_launch.algorithm
            launch_replacements["execution_variant"] = (
                canonical_launch.execution_variant)
            launch_replacements["source_program"] = (
                canonical_launch.source_program)
        elif (hardware_id == "ascend_910b"
              and variant in ("rmsnorm", "add_rmsnorm")):
            tiling_key = int(
                canonical_launch.parameters["cann_tiling_key"])
            expected_kernel_names = {
                ("rmsnorm", 0): (
                    "RmsNorm_normal_all_bf16_high_performance_0"),
                ("rmsnorm", 1): (
                    "RmsNorm_normal_all_bf16_high_performance_1"),
                ("add_rmsnorm", 30): (
                    "AddRmsNorm_3_high_performance_30"),
                ("add_rmsnorm", 31): (
                    "AddRmsNorm_3_high_performance_31"),
                ("add_rmsnorm", 33): (
                    "AddRmsNorm_3_high_performance_33"),
            }
            selector_key = (variant, tiling_key)
            if selector_key not in expected_kernel_names:
                raise ValueError(
                    "Ascend 910B NORM source selector produced an unknown "
                    f"variant/key pair: {selector_key!r}")
            expected_kernel_name = expected_kernel_names[selector_key]
            if (metrics["kernel_name"] != expected_kernel_name
                    or launch.grid != canonical_launch.grid
                    or launch.block != canonical_launch.block):
                raise ValueError(
                    "Ascend 910B NORM source selector disagrees with "
                    "profiler launch identity: "
                    f"expected_kernel={expected_kernel_name!r}, "
                    f"observed_kernel={metrics['kernel_name']!r}, "
                    f"expected_grid={canonical_launch.grid}, "
                    f"observed_grid={launch.grid}, "
                    f"expected_block={canonical_launch.block}, "
                    f"observed_block={launch.block}")
            # After the runtime identity and geometry agree with the fixed
            # CANN selector, normalize the evaluator-only observed label to
            # the canonical production algorithm.  This lets the production
            # predictor enforce that source-bound DAGs cannot be attached to
            # an unverified generic observation.
            launch_replacements["algorithm"] = canonical_launch.algorithm
            launch_replacements["execution_variant"] = (
                canonical_launch.execution_variant)
            launch_replacements["source_program"] = (
                canonical_launch.source_program)
        launch = dataclasses.replace(launch, **launch_replacements)
        if (hardware_id == "nvidia_a800"
                and variant == "add_rmsnorm"):
            expected_symbol = (
                "std::enable_if<T2>(int)0&&vllm::_typeConvert<T1>::exists, "
                "void>::type vllm::fused_add_rms_norm_kernel<"
                "c10::BFloat16, (int)8>(T1 *, long, T1 *, const T1 *, "
                "float, int, int)"
            )
            if metrics["kernel_name"] == expected_symbol:
                if (launch.grid != canonical_launch.grid
                        or launch.block != canonical_launch.block):
                    raise ValueError(
                        "A800 typed Add-RMS vec8 observation disagrees with "
                        "the source-defined launch geometry")
                parameters = dict(launch.parameters)
                parameters[
                    SOURCE_BOUND_MEMORY_DEPENDENCY_TASK_PARAMETER] = 1.0
                launch = dataclasses.replace(
                    launch, parameters=parameters)
    return KernelSpec(
        kind=kind,
        hardware_id=hardware_id,
        shape=shape,
        dtype=hardware.kernel_dtype,
        variant=variant,
        launch=launch,
        attributes=attributes,
    )


def normalize_kernel_spec(op_info: Mapping, kdata: Mapping,
                          hardware_id: str, hardware,
                          *, tensor_ordinal: Optional[int] = None,
                          kernel_ordinal: Optional[int] = None,
                          moe_reduce_ordinal: Optional[int] = None,
                          gemm_split_k: Optional[int] = None,
                          gemm_split_k_partial_bytes: Optional[int] = None,
                          gemm_split_k_partial_dram_bytes: Optional[int] = None,
                          profiler_metrics: Optional[Mapping] = None,
                          l2_execution_context: Optional[
                              L2ExecutionContext] = None,
                          ) -> KernelSpec:
    """Return a complete production spec for one profiler kernel observation."""

    if kernel_ordinal is not None:
        if isinstance(kernel_ordinal, bool) or not isinstance(
                kernel_ordinal, int):
            raise TypeError("kernel_ordinal must be an integer or None")
        if kernel_ordinal < 0:
            raise ValueError("kernel_ordinal cannot be negative")
    if moe_reduce_ordinal is not None:
        if (isinstance(moe_reduce_ordinal, bool)
                or not isinstance(moe_reduce_ordinal, int)):
            raise TypeError("moe_reduce_ordinal must be an integer or None")
        if moe_reduce_ordinal < 0:
            raise ValueError("moe_reduce_ordinal cannot be negative")
    profiler_metrics = (
        extract_ncu_metrics(kdata)
        if profiler_metrics is None else profiler_metrics)
    if not isinstance(profiler_metrics, Mapping):
        raise TypeError("normalize_kernel_spec profiler_metrics must be a mapping")
    if l2_execution_context is None:
        l2_execution_context = L2ExecutionContext()
    if not isinstance(l2_execution_context, L2ExecutionContext):
        raise TypeError(
            "normalize_kernel_spec l2_execution_context must be an "
            "L2ExecutionContext or None")
    if (hardware.kernel_memory_model == "npu_tile"
            and profiler_metrics["op_type"] is None):
        raise ValueError(
            "npu_tile observation requires an explicit profiler Op Type")
    op_type = _non_empty_string(op_info, "Type", "op_info").upper()
    attributes = op_info["Attr"]
    if op_type == "GEMM":
        spec = _normalize_gemm(
            attributes, kdata, hardware_id, hardware, gemm_split_k,
            gemm_split_k_partial_bytes,
            gemm_split_k_partial_dram_bytes)
        return dataclasses.replace(
            spec, l2_execution_context=l2_execution_context)
    elif op_type == "FLASHATTN":
        return _normalize_flash_attention(
            attributes, kdata, hardware_id, hardware)
    elif op_type == "EAGERATTN":
        return _normalize_eager_attn(
            attributes, kdata, hardware_id, hardware)
    elif op_type == "FUSEDMOE":
        return _normalize_fused_moe(
            attributes, kdata, hardware_id, hardware, tensor_ordinal,
            kernel_ordinal, moe_reduce_ordinal,
            profiler_metrics=profiler_metrics)
    elif op_type == "UNFUSEDMOE":
        return _normalize_unfused_moe(
            attributes, kdata, hardware_id, hardware)
    elif op_type in (
            "NORM", "ROTARY", "ACTIVATION", "KV_CACHE", "ELEMENTWISE",
            "REDUCE", "GATHER"):
        return _normalize_vector(
            op_type, attributes, kdata, hardware_id, hardware)
    else:
        raise NotImplementedError(
            f"observation normalization is not implemented for {op_type}")
