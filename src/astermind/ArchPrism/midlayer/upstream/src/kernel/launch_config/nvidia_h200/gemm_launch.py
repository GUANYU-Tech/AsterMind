"""NVIDIA H200 launch selection for FP8 block-scaled dense GEMMs.

vLLM 0.24.0 prices a DeepSeek-V3 FP8 linear (128x128 e4m3 weight blocks,
dynamic 1x128 activation groups, BF16 output) through
``FlashInferFp8DeepGEMMDynamicBlockScaledKernel``
(``vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py``), which
dispatches on the token count M with ``torch.cond(M < 32, ...)``:

* ``M < 32``: FlashInfer ``fp8_blockscale_gemm`` (TensorRT-LLM
  ``CutlassFp8BlockScaleGemmRunner::gemm`` with BF16 input): one
  ``scale_1x128_kernel<<<num_sms * 8, 256>>>`` activation quantisation
  followed by ``deep_gemm::fp8_gemm_kernel_swapAB`` whose configuration comes
  from ``deep_gemm::jit::get_best_gemm_config(N, M, K, 1, num_sms, false,
  true)`` (``flashinfer/data/csrc/nv_internal/tensorrt_llm/deep_gemm/
  jit_utils.cuh``);
* ``M >= 32``: ``per_token_group_quant_fp8`` (a separate kernel, outside this
  selector) and DeepGEMM ``fp8_gemm_nt`` -> ``sm90_fp8_gemm_1d2d_impl`` whose
  configuration comes from ``get_best_config<SM90ArchSpec>`` (DeepGEMM
  ``891d57b4`` ``csrc/jit_kernels/heuristics/sm90.hpp``, the commit pinned by
  vLLM 0.24.0 ``cmake/external_projects/deepgemm.cmake``).

Both heuristics are reproduced here as static source facts; nothing is
fitted.  ``owns_select_gemm_launch`` declares this module's request domain
(a GEMM whose lowering declares block-FP8 ``weight_quantization``); the
dispatcher sends every other GEMM to the projected package, so no projected
rule is restated here.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Optional

import numpy as np

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.block_fp8_gemm import (
    BLOCK_FP8_WEIGHT_QUANTIZATION,
    DEEPGEMM_1D2D_ALGORITHM,
    DEEPGEMM_FP8_EINSUM_BATCHED_ALGORITHM,
    TRTLLM_SWAP_AB_ALGORITHM,
)
from src.kernel.launch_config.gemm_launch_selection import (
    ObservedGridLaunchSelection,
    load_gemm_launch_selection,
)
from src.kernel.launch_config.gemm_workload import (
    analytical_cublas_tiling,
    cublaslt_main_block_threads,
)
from src.kernel.launch_config.launch_program import GpuCtaRasterProgram
from src.kernel.launch_config.selection_primitives import (
    ceil_div,
    launch_policy,
    require_dtype,
    require_shape,
    require_variant,
)

# vLLM ``_dynamic_flashinfer_deepgemm_blockscale_gemm_impl``: M < 32 -> swapAB.
_SWAP_AB_M_THRESHOLD = 32
# FlashInfer eligibility (``should_use_flashinfer_for_blockscale_fp8_gemm``).
_N_MULTIPLE = 64
_K_MULTIPLE = 128
_BLOCK_K = 128
# Shared-memory capacity constant hard-coded by both heuristics.
_SM90_SMEM_CAPACITY = 232448
_TMA_THREADS = 128


@dataclass(frozen=True)
class Fp8BlockGemmConfig:
    """One source-derived FP8 block-scaled GEMM launch (problem coordinates).

    ``tile_m``/``tile_n`` are the token/output-channel extents of one tile in
    GEMM coordinates (for swap-AB the kernel's BLOCK_N covers tokens and its
    BLOCK_M covers output channels).  ``multicast_on_a`` is the DeepGEMM
    ``kIsTMAMulticastOnA`` flag (activation multicast across the cluster).
    """

    algorithm: str
    tile_m: int
    tile_n: int
    stages: int
    cluster: int
    multicast_on_a: bool
    math_threads: int
    smem_bytes: int
    swizzle_cd: int


def _trtllm_swap_ab_smem(stages: int, k: int, block_m: int, block_n: int) -> int:
    smem_d = block_n * block_m * 2
    smem_a_per_stage = block_m * _BLOCK_K
    smem_scales_a = ceil_div(k, _BLOCK_K) * 4
    smem_b_per_stage = block_n * _BLOCK_K
    smem_scales_b = ceil_div(block_n * 4, 128) * 128
    smem_barrier = stages * 8 * 2
    return (smem_d + stages * smem_a_per_stage + stages * smem_scales_b
            + stages * smem_b_per_stage + ceil_div(smem_scales_a, 8) * 8
            + smem_barrier)


def trtllm_swap_ab_config(m: int, n: int, k: int,
                          num_sms: int) -> Fp8BlockGemmConfig:
    """``get_best_gemm_config(N, M, K, 1, num_sms, false, swap_ab=true)``."""

    shape_m, shape_n = n, m  # the heuristic is called with swapped M/N
    block_m = 64 if shape_m <= 64 else 128

    def waves(bm, bn):
        return ceil_div(ceil_div(shape_m, bm) * ceil_div(shape_n, bn), num_sms)

    def last_wave_util(bm, bn):
        util = (ceil_div(shape_m, bm) * ceil_div(shape_n, bn)) % num_sms
        return num_sms if util == 0 else util

    best_m = best_n = 0
    for block_n in range(16, 129, 8):
        if best_m == 0:
            success = True
        elif waves(block_m, block_n) < waves(best_m, best_n):
            success = True
        elif waves(block_m, block_n) == waves(best_m, best_n):
            util = last_wave_util(block_m, block_n)
            best_util = last_wave_util(best_m, best_n)
            success = util > best_util or (
                util == best_util and (
                    block_m > best_m
                    or (block_m == best_m and block_n < best_n)))
        else:
            success = False
        if success:
            best_m, best_n = block_m, block_n
    candidates = (6, 5, 4) if 128 % best_n else (8, 7, 6, 5, 4)
    stages = smem = None
    for candidate in candidates:
        size = _trtllm_swap_ab_smem(candidate, k, best_m, best_n)
        if size <= _SM90_SMEM_CAPACITY:
            stages, smem = candidate, size
            break
    if stages is None:
        raise ValueError(
            f"TRT-LLM swap-AB FP8 GEMM has no legal stage count for K={k}")
    multicast = 1
    if (shape_n >= 1024 and shape_m % (best_m * 2) == 0
            and num_sms % 2 == 0):
        multicast = 2
    return Fp8BlockGemmConfig(
        algorithm=TRTLLM_SWAP_AB_ALGORITHM,
        tile_m=best_n, tile_n=best_m, stages=stages, cluster=multicast,
        multicast_on_a=False,
        math_threads=(best_m // 64) * 128, smem_bytes=smem,
        swizzle_cd=0)


def _swizzle_mode(block_bytes: int) -> int:
    for mode in (128, 64, 32, 16):
        if block_bytes % mode == 0:
            return mode
    raise AssertionError("DeepGEMM swizzle mode is unreachable")


def _align(value: int, alignment: int) -> int:
    return ceil_div(value, alignment) * alignment


def _deepgemm_1d2d_pipeline(k: int, block_m: int, block_n: int):
    smem_cd = _align(block_m * block_n * 2, 1024)
    smem_barriers = 16 * 8 * 2
    smem_a_per_stage = block_m * _BLOCK_K
    smem_b_per_stage = block_n * _BLOCK_K
    smem_sfa_per_stage = _align(block_m * 4, 128)
    use_uniform_sfb = 1 if _BLOCK_K % block_n == 0 else 2
    smem_extra_sfb = _align(ceil_div(k, _BLOCK_K) * 4 * use_uniform_sfb, 8)
    smem_extra = smem_cd + smem_barriers + smem_extra_sfb
    smem_per_stage = smem_a_per_stage + smem_b_per_stage + smem_sfa_per_stage
    stages = min((_SM90_SMEM_CAPACITY - smem_extra) // smem_per_stage, 16)
    return smem_extra + stages * smem_per_stage, stages


def deepgemm_1d2d_config(m: int, n: int, k: int,
                         num_sms: int) -> Fp8BlockGemmConfig:
    """``get_best_config<SM90ArchSpec>`` for a Normal K-major 1D2D GEMM."""

    block_m_candidates = [64, 128]
    if m <= 16:
        block_m_candidates.append(16)
    if m <= 32:
        block_m_candidates.append(32)
    block_m_candidates.append(256)  # BF16 output supports 256
    block_n_candidates = list(range(16, 193, 16))
    candidates = []
    for cluster_m in (1, 2):
        for cluster_n in (1, 2):
            if cluster_m * cluster_n > 2:
                continue
            if num_sms % (cluster_m * cluster_n):
                continue
            for block_m in block_m_candidates:
                for block_n in block_n_candidates:
                    if (block_n > _BLOCK_K
                            and block_n % (block_n - _BLOCK_K) != 0
                            and _BLOCK_K % (block_n - _BLOCK_K) != 0):
                        continue
                    if block_m > 128 and block_n > 128:
                        continue
                    # K-major FP8 A/B swizzle is 128 B for every candidate.
                    _, stages = _deepgemm_1d2d_pipeline(k, block_m, block_n)
                    if stages < 3 or (
                            block_m * block_n < 128 * 192 and stages < 4):
                        continue
                    candidates.append((block_m, block_n, cluster_m, cluster_n))
    if not candidates:
        raise ValueError("DeepGEMM SM90 1D2D has no layout candidate")

    l2_bandwidth_per_cycle = int(min(64.0 * num_sms, 8e6 / 1.3e3))
    l1_bandwidth_per_cycle = 128 * num_sms

    def num_cycles(layout):
        block_m, block_n, cluster_m, cluster_n = layout
        num_blocks = ceil_div(m, block_m) * ceil_div(n, block_n)
        num_waves = ceil_div(num_blocks, num_sms)
        bytes_l2_ab = k * (block_m // cluster_n + block_n // cluster_m)
        bytes_l1_ab = k * (block_m + block_n)
        bytes_l1_tc = k * (max(64, block_m) + block_n) + block_m * block_n * 2
        bytes_cd = block_m * block_n * 2
        l2_cycles = ((bytes_l2_ab + bytes_cd) * num_blocks
                     // l2_bandwidth_per_cycle)
        l1_cycles = ((bytes_l1_ab + bytes_l1_tc + bytes_cd) * num_blocks
                     // l1_bandwidth_per_cycle)
        # C++: int64 / float, float wave efficiency, truncated to int64.
        efficiency = np.float32(num_blocks) / np.float32(num_waves * num_sms)
        cycles = int(np.float32(max(l1_cycles, l2_cycles)) / efficiency)
        if cluster_m * cluster_n > 1 and num_waves <= 1:
            return np.iinfo(np.int64).max
        return cycles

    best = candidates[0]
    best_cycles = num_cycles(best)
    for layout in candidates[1:]:
        cycles = num_cycles(layout)
        if cycles < best_cycles:
            best, best_cycles = layout, cycles
    block_m, block_n, cluster_m, cluster_n = best
    smem, stages = _deepgemm_1d2d_pipeline(k, block_m, block_n)
    return Fp8BlockGemmConfig(
        algorithm=DEEPGEMM_1D2D_ALGORITHM,
        tile_m=block_m, tile_n=block_n, stages=stages,
        cluster=cluster_m * cluster_n, multicast_on_a=cluster_n > 1,
        math_threads=128 if block_m <= 64 else 256,
        smem_bytes=smem, swizzle_cd=_swizzle_mode(block_n * 2))


def select_block_fp8_gemm_config(
        m: int, n: int, k: int, num_sms: int) -> Fp8BlockGemmConfig:
    """Return the vLLM 0.24 kernel configuration for one block-FP8 linear."""

    if n % _N_MULTIPLE or k % _K_MULTIPLE:
        raise NotImplementedError(
            "vLLM 0.24 FlashInfer/DeepGEMM block-FP8 linear requires "
            f"N % {_N_MULTIPLE} == 0 and K % {_K_MULTIPLE} == 0, got "
            f"N={n}, K={k}")
    if m < _SWAP_AB_M_THRESHOLD:
        return trtllm_swap_ab_config(m, n, k, num_sms)
    return deepgemm_1d2d_config(m, n, k, num_sms)


def owns_select_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> bool:
    """Whether this package selects the GEMM: a declared block-FP8 linear."""

    if "weight_quantization" not in attributes:
        return False
    if attributes["weight_quantization"] != BLOCK_FP8_WEIGHT_QUANTIZATION:
        raise ValueError(
            "unsupported weight_quantization "
            f"{attributes['weight_quantization']!r}")
    return True


def _require_block_fp8_output_storage(attributes):
    if ("output_dtype" in attributes
            and attributes["output_dtype"] not in (None, "bf16")):
        raise ValueError("known packed block-FP8 GEMM launch requires BF16 output storage")


def select_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Select the H200 vLLM 0.24 launch of one block-FP8 dense linear.

    Outside ``owns_select_gemm_launch`` this selector has no rule and fails.
    """

    if not owns_select_gemm_launch(
            shape, dtype, variant, attributes, hardware):
        raise ValueError(
            "H200 GEMM selector only owns declared block-FP8 linears")
    _require_block_fp8_output_storage(attributes)
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    require_variant(variant, ("gemm",), "H200 block-FP8 GEMM")
    require_dtype(dtype, ("bf16",))
    if b != 1:
        raise ValueError("H200 block-FP8 linear is a single (b=1) GEMM")
    if hardware.max_smem_per_block_bytes < _SM90_SMEM_CAPACITY:
        raise ValueError(
            "H200 block-FP8 heuristics assume 232448 B opt-in shared memory")
    num_sms = int(hardware.num_compute_units)
    if "fp8_linear_runner" in attributes:
        if attributes["fp8_linear_runner"] != "sglang_deepgemm":
            raise ValueError(
                f"unknown fp8_linear_runner {attributes['fp8_linear_runner']!r}")
        # SGLang deepgemm_w8a8_block_fp8_linear_with_fallback: DeepGEMM
        # 1d2d at every M (no FlashInfer swap-AB branch).
        if n % _N_MULTIPLE or k % _K_MULTIPLE:
            raise NotImplementedError(
                "SGLang DeepGEMM block-FP8 linear requires N % 64 == 0 and "
                "K % 128 == 0 (else Triton fallback, not derived)")
        config = deepgemm_1d2d_config(m, n, k, num_sms)
    else:
        config = select_block_fp8_gemm_config(m, n, k, num_sms)
    return LaunchConfig(
        algorithm=config.algorithm,
        grid=(num_sms, 1, 1),
        block=(_TMA_THREADS + config.math_threads, 1, 1),
        cluster=(config.cluster, 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": config.tile_m, "n": config.tile_n,
                        "k": _BLOCK_K},
            pipeline_depth=config.stages,
            split_k=1,
        ),
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 0.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
        },
        execution_variant="gemm",
    )


def owns_select_target_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> bool:
    """Whether this package selects the target-model GEMM.

    The MLA absorbed projections are strided-batched GEMMs (``torch.bmm``
    over the local heads, ``vllm/model_executor/layers/attention/
    mla_attention.py`` ``forward_mqa`` / ``_v_up_proj``); the MoE router is
    owned where vLLM dispatches its dedicated DSV3 router kernel.
    """

    if variant == "batched_gemm":
        weight_dtype = attributes["weight_dtype"]
        if weight_dtype == "bf16":
            if attributes["weight_group_size"] != 0:
                raise ValueError(
                    "H200 BF16 batched GEMM requires unquantized weights")
            return True
        # DeepSeek-V4 wo_a FP8 einsum (deepseek_v4/nvidia/ops/o_proj.py).
        if weight_dtype == "fp8":
            _require_block_fp8_output_storage(attributes)
            if attributes["weight_group_size"] != 128:
                raise ValueError(
                    "H200 FP8 batched GEMM requires 128x128 block scales")
            return True
        raise ValueError(
            "H200 batched GEMM owns only BF16 and block-scaled FP8 "
            f"weights, got {weight_dtype!r}")
    if variant == "moe_router":
        return (_uses_dsv3_router_gemm(shape, attributes)
                or _observed_sglang_router_launch(shape, attributes)
                is not None)
    return False


# vLLM 0.24 ``GateLinear`` (vllm/model_executor/layers/fused_moe/router/
# gate_linear.py): tier 1 is ``ops.dsv3_router_gemm`` on Hopper when
# (hidden, experts) is an instantiated DSV3 shape, the router weight is BF16
# without bias, and the token count is at most ``_dsv3_max_batch`` (16 on
# Hopper).  Larger token counts take tier 3 (cuBLAS ``torch.mm`` with an FP32
# output), which this package does not own.
DSV3_ROUTER_SHAPES = frozenset({(7168, 256), (6144, 256), (7168, 384)})
DSV3_ROUTER_MAX_TOKENS_HOPPER = 16
# csrc/libtorch_stable/moe/dsv3_router_gemm_float_out.cu
# ``invokeRouterGemmFloatOutput``: gridDim = kNumExperts, blockDim = 128;
# each CTA reduces one expert column over the hidden dimension with SIMT FP32
# FMAs (VPT = 16 B / sizeof(bf16) = 8 elements per thread per K iteration)
# for every token of the batch, then a warp shuffle + shared-memory reduction.
DSV3_ROUTER_GEMM_ALGORITHM = "vllm_dsv3_router_gemm_float_out"
_DSV3_ROUTER_BLOCK_THREADS = 128


def _uses_dsv3_router_gemm(
        shape: Mapping[str, int], attributes: Mapping[str, object]) -> bool:
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    return ((k, n) in DSV3_ROUTER_SHAPES
            and b * m <= DSV3_ROUTER_MAX_TOKENS_HOPPER
            and attributes["weight_dtype"] == "bf16"
            and attributes["weight_group_size"] == 0)


# SGLang v0.5.19 ``MoEGate.forward`` (python/sglang/srt/models/
# deepseek_v2.py:540-563) runs the DSV3 router kernel only for <= 16 tokens
# on SM >= 90; every larger CUDA batch calls ``linear_bf16_fp32``
# (python/sglang/kernels/ops/attention/dsv4/gemm.py:144), whose default
# ``SGLANG_OPT_BF16_FP32_GEMM_ALGO="cublas"`` (srt/environ.py:1392) is
# ``torch.mm(x, W.t(), out_dtype=torch.float32)`` -- the call the H200
# grid-block-only launch selection observed.  The workload builder declares
# this runner on SGLang router nodes only, so no vLLM request can reach it.
SGLANG_ROUTER_LINEAR_RUNNER = "sglang_linear_bf16_fp32_cublas"
OBSERVED_GRID_ROUTER_ALGORITHM = "cublaslt_observed_grid:moe_router"
OBSERVED_GRID_REDUCER_ALGORITHM = "observed_gemm_aux:gemm_reduce"
# The observation's cuBLAS/CUDA revision (manifest ``software_revision``).
_OBSERVED_ROUTER_SOFTWARE_REVISION = "cuda_12.9"


@lru_cache(maxsize=1)
def _observed_launch_lut():
    return load_gemm_launch_selection(
        "nvidia_h200", software_revision=_OBSERVED_ROUTER_SOFTWARE_REVISION)


def _observed_sglang_router_launch(
        shape: Mapping[str, int], attributes: Mapping[str, object],
        ) -> ObservedGridLaunchSelection | None:
    """Return the observed launch of an SGLang cuBLAS router GEMM, if any.

    A request outside the observed rows keeps the projected analytical
    launch (``cublas_analytical_gemm:moe_router``), whose name states that no
    launch was observed.
    """

    if "router_linear_runner" not in attributes:
        return None
    if attributes["router_linear_runner"] != SGLANG_ROUTER_LINEAR_RUNNER:
        raise ValueError(
            "unknown router_linear_runner "
            f"{attributes['router_linear_runner']!r}")
    if _uses_dsv3_router_gemm(shape, attributes):
        return None
    if (attributes["weight_dtype"] != "bf16"
            or attributes["weight_group_size"] != 0):
        raise ValueError("SGLang cuBLAS router requires unquantized BF16")
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    key = ("bf16", "gemm", 1, b * m, n, k, "row_major", "row_major",
           "row_major", False, True, "none", False)
    lut = _observed_launch_lut()
    if key not in lut:
        return None
    selection = lut[key].observed_grid
    if selection is None or selection.output_dtype != "fp32":
        raise ValueError(
            "H200 router launch row must be an FP32-output observed grid")
    return selection


def _select_observed_router_launch(
        selection: ObservedGridLaunchSelection) -> LaunchConfig:
    return LaunchConfig(
        algorithm=OBSERVED_GRID_ROUTER_ALGORITHM,
        grid=selection.grid,
        block=(selection.block_threads, 1, 1),
        # Neither the tile nor the split count is observed.
        tiling=None,
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 4.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
            "split_k_count_observed": 0.0,
        },
        execution_variant="moe_router",
    )


def owns_select_gemm_reducer_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> bool:
    """Whether the reducer belongs to an observed SGLang router launch."""

    return (variant == "moe_router"
            and _observed_sglang_router_launch(shape, attributes)
            is not None)


def select_gemm_reducer_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Return the observed ``cublasLt::splitKreduce_kernel`` launch."""

    selection = (
        _observed_sglang_router_launch(shape, attributes)
        if variant == "moe_router" else None)
    if selection is None:
        raise ValueError(
            "H200 reducer selector only owns observed SGLang router launches")
    return LaunchConfig(
        algorithm=OBSERVED_GRID_REDUCER_ALGORITHM,
        grid=(*selection.reducer_grid, 1),
        block=(*selection.reducer_block, 1),
        tiling=None,
        parameters={
            "split_k_partial_bytes_per_element": 4.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
            "split_k_count_observed": 0.0,
        },
        execution_variant="gemm_reduce",
    )


def _select_dsv3_router_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str]) -> LaunchConfig:
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    require_dtype(dtype, ("bf16", "fp32"))
    if k % (8 * _DSV3_ROUTER_BLOCK_THREADS):
        raise ValueError("dsv3_router_gemm needs whole K iterations")
    return LaunchConfig(
        algorithm=DSV3_ROUTER_GEMM_ALGORITHM,
        grid=(n, 1, 1),
        block=(_DSV3_ROUTER_BLOCK_THREADS, 1, 1),
        tiling=None,
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 0.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
        },
        execution_variant="moe_router",
    )


def select_target_gemm_launch(
        shape: Mapping[str, int], dtype: Optional[str],
        variant: Optional[str], attributes: Mapping[str, object],
        hardware) -> LaunchConfig:
    """Select one strided-batched BF16 GEMM launch.

    A strided-batched GEMM is ``b`` independent problems, each with its own A
    and B operand: an output tile can never span two batch problems, so the
    launch holds the tiles of one (m, n) problem once per batch (cuBLAS puts
    the batch index on a grid axis).  The per-problem tile is the analytical
    cuBLAS tiler the projected package uses for one GEMM, applied to the
    per-batch rows ``m`` instead of the folded ``b * m``.
    """

    if not owns_select_target_gemm_launch(
            shape, dtype, variant, attributes, hardware):
        raise ValueError(
            "H200 target GEMM selector only owns strided-batched GEMMs and "
            "the vLLM DSV3 router GEMM")
    if variant == "moe_router":
        observed = _observed_sglang_router_launch(shape, attributes)
        if observed is not None:
            require_dtype(dtype, ("bf16", "fp32"))
            return _select_observed_router_launch(observed)
        return _select_dsv3_router_gemm_launch(shape, dtype)
    b, m, n, k = require_shape(shape, ("b", "m", "n", "k"))
    require_variant(variant, ("batched_gemm",), "H200 batched GEMM")
    if launch_policy(attributes) != "analytical_launch":
        raise NotImplementedError(
            "H200 batched GEMM has no captured launch selection")
    if attributes["weight_dtype"] == "fp8":
        # DeepSeek-V4 wo_a einsum (deepseek_v4/nvidia/ops/o_proj.py
        # deep_gemm_fp8_o_proj, SM90 recipe (1, 128, 128)): one DeepGEMM
        # SM90 block-scaled problem per o_group, configured by the same
        # source heuristic as the dense block-FP8 linear; the activation
        # quantisation is the separate fused_inv_rope_fp8_quant kernel.
        require_dtype(dtype, ("bf16",))
        if (attributes["batch_layout"] != "einsum_bhr_hdr_bhd"
                or attributes["c_batch_stride_elements"] != n):
            raise ValueError("H200 FP8 einsum requires token-major bhr,hdr->bhd output with group stride n")
        if hardware.max_smem_per_block_bytes < _SM90_SMEM_CAPACITY:
            raise ValueError(
                "H200 FP8 einsum heuristics assume 232448 B opt-in shared "
                "memory")
        config = select_block_fp8_gemm_config(
            m, n, k, int(hardware.num_compute_units))
        return LaunchConfig(
            algorithm=DEEPGEMM_FP8_EINSUM_BATCHED_ALGORITHM,
            grid=(int(hardware.num_compute_units), 1, 1),
            block=(_TMA_THREADS + config.math_threads, 1, 1),
            cluster=(config.cluster, 1, 1),
            tiling=TilingSpec(
                tile_shape={"m": config.tile_m, "n": config.tile_n,
                            "k": _BLOCK_K},
                pipeline_depth=config.stages,
                split_k=1,
            ),
            parameters={
                "compulsory_traffic": 0.0,
                "weight_tile_rows": 0.0,
                "mte2_tile_bytes": 0.0,
                "aiv_mixed": 0.0,
                "n_major_traffic": 0.0,
                "split_k_partial_bytes_per_element": 0.0,
                "split_k_partial_dram_bytes_per_element": 0.0,
            },
            execution_variant="batched_gemm",
            source_program=GpuCtaRasterProgram("none", 0, False, 1),
        )
    require_dtype(dtype, ("bf16",))
    tile_m, tile_n, tile_k, depth = analytical_cublas_tiling(m, n, k)
    return LaunchConfig(
        algorithm="cublas_analytical_gemm:batched_gemm",
        grid=(ceil_div(m, tile_m) * ceil_div(n, tile_n), b, 1),
        block=(cublaslt_main_block_threads(tile_m, tile_n), 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": tile_m, "n": tile_n, "k": tile_k},
            pipeline_depth=depth,
            split_k=1,
        ),
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 0.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
        },
        execution_variant="batched_gemm",
        # No vendor raster map was captured: the traffic model rasters each
        # batch problem's tile grid directly.
        source_program=GpuCtaRasterProgram("none", 0, False, 1),
    )
