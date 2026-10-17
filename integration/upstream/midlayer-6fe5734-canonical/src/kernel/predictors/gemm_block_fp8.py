"""Hopper FP8 block-scaled dense GEMM template (vLLM 0.24 stack).

Prices the two launches a hardware block-FP8 selector emits (the Hopper
package reproduces the vLLM 0.24 FlashInfer/DeepGEMM heuristics):

``trtllm_fp8_blockscale_swapab_gemm`` (M < 32)
    FlashInfer/TensorRT-LLM ``fp8_gemm_run``: ``scale_1x128_kernel`` over
    ``num_sms * 8`` CTAs, then the persistent ``fp8_gemm_kernel_swapAB``.
``deepgemm_sm90_fp8_gemm_1d2d`` (M >= 32)
    DeepGEMM persistent ``sm90_fp8_gemm_1d2d_impl`` (its separate
    ``per_token_group_quant`` activation quantiser is not part of this call).

Execution structure (source): a grid of ``num_sms`` persistent CTAs, one per
SM; CTA ``i`` owns tiles ``i, i + num_sms, ...``.  Per tile an elected TMA
thread streams ``ceil(K / 128)`` k-blocks of the weight box and the
activation box through a ``stages``-deep full/empty mbarrier ring; the math
warp-groups wait on each stage, issue the FP8 wgmma k-block and release the
stage; the epilogue stores the BF16 tile with TMA.

Timing template:

* one persistent wave (``tiles <= num_sms``, every decode launch): each
  working CTA runs exactly one tile, so the launch lasts one TMA-ring pass:
  a fill (launch shell, ring priming, BF16 tile store) plus ``ceil(K/128)``
  stage intervals.  The interval composes the per-SM ring stage service with
  the chip DRAM stream of the ``ceil(N / tile_n)`` distinct weight boxes per
  stage round; both, the composition exponent and the fill come from the
  hardware's MMA-free TMA-ring probe (``fp8_tma_operand_stream.cu`` under its
  ``microbench/<hardware>/operators/gemm``).
  The k-block tensor time at the public FP8 rate bounds the interval below.
* several waves (prefill): the busiest CTA runs ``ceil(tiles / num_sms)``
  tiles of ``ceil(K/128)`` k-blocks each at the public FP8 tensor rate of one
  SM (never below the per-SM ring stage service), bounded below by the
  compulsory A/B reads at the measured TMA DRAM stream rate plus the BF16
  output writes at the measured full-chip tile-store rate; the measured
  launch shell and the final tile-store drain (one BF16 tile per CTA at the
  tile-store rate) are added.

Every duration comes from the HWSpec ``block_fp8_gemm_tma_ring_service``
section (mechanism probes) or the HWSpec public FP8 tensor rate; the
launch span floor composes by maximum.  No production duration is read.
"""

from src.kernel.formula import small_kernel_span_floor_s
from src.kernel.kernel_spec import KernelPrediction, KernelSpec
from src.kernel.launch_config.block_fp8_gemm import (
    DEEPGEMM_1D2D_ALGORITHM,
    TRTLLM_SCALE_1X128_CTAS_PER_SM,
    TRTLLM_SCALE_1X128_THREADS,
    TRTLLM_SWAP_AB_ALGORITHM,
)
from src.kernel.predictors.common import (
    graph_replayed_launch,
    prediction_from_ns,
)

BLOCK_FP8_GEMM_ALGORITHMS = frozenset({
    TRTLLM_SWAP_AB_ALGORITHM, DEEPGEMM_1D2D_ALGORITHM,
})
_BLOCK_K = 128


def _ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def ring_stage_interval_ns(ring, weight_box_bytes, streaming_weight_blocks):
    """Per-stage TMA-ring interval of one CTA (measured service form).

    Each CTA pays the per-SM ring stage service ``stage_floor_ns``; the chip
    streams ``streaming_weight_blocks * weight_box_bytes`` distinct DRAM bytes
    per stage round at ``dram_stream_bytes_per_ns``; the two compose by the
    measured p-norm.
    """

    stream_ns = (streaming_weight_blocks * weight_box_bytes
                 / ring["dram_stream_bytes_per_ns"])
    power = ring["combination_power"]
    return (ring["stage_floor_ns"] ** power
            + stream_ns ** power) ** (1.0 / power)


def _fp8_tensor_flops_per_sm_s(hardware) -> float:
    tensor = hardware.resource_by_role("compute_tensor_fp8")
    return float(tensor.capacity) / int(hardware.num_compute_units)


def predict_block_fp8_gemm(kernel: KernelSpec, hardware) -> KernelPrediction:
    """Price one block-FP8 dense GEMM call."""

    launch = kernel.launch
    if launch.algorithm not in BLOCK_FP8_GEMM_ALGORITHMS:
        raise ValueError(
            f"not a block-FP8 GEMM launch: {launch.algorithm!r}")
    service = hardware.block_fp8_gemm_tma_ring_service
    if service is None:
        raise ValueError(
            f"{hardware.hardware_id} publishes no block-FP8 GEMM TMA-ring "
            "service")
    b, m, n, k = (int(kernel.shape[name]) for name in ("b", "m", "n", "k"))
    if b != 1:
        raise ValueError("block-FP8 GEMM is a single (b=1) GEMM")
    tiling = launch.tiling
    tile_m = int(tiling.tile_shape["m"])
    tile_n = int(tiling.tile_shape["n"])
    if int(tiling.tile_shape["k"]) != _BLOCK_K or tiling.split_k != 1:
        raise ValueError("block-FP8 GEMM uses BLOCK_K=128 without split-K")
    grid = launch.grid_size
    if grid != int(hardware.num_compute_units):
        raise ValueError("block-FP8 GEMM launches one persistent CTA per SM")
    tiles = _ceil_div(m, tile_m) * _ceil_div(n, tile_n)
    kblocks = _ceil_div(k, _BLOCK_K)
    tensor_kblock_ns = (
        2.0 * tile_m * tile_n * _BLOCK_K
        / _fp8_tensor_flops_per_sm_s(hardware) * 1e9)
    groups = {}
    ring = service["ring"]
    if tiles <= grid:
        # One persistent wave: every working CTA streams its tile's weight
        # rows while the ceil(N / tile_n) distinct weight blocks stream from
        # DRAM concurrently (CTAs sharing a weight block hit L2).
        interval_ns = ring_stage_interval_ns(
            ring, tile_n * _BLOCK_K, _ceil_div(n, tile_n))
        fill_ns = (ring["fill_base_ns"]
                   + ring["fill_per_store_byte_ns"] * 2 * tile_m * tile_n)
        groups["tma_ring_fill"] = fill_ns
        groups["tma_ring_stages"] = kblocks * interval_ns
        groups["fp8_tensor"] = kblocks * tensor_kblock_ns
        gemm_ns = fill_ns + kblocks * max(interval_ns, tensor_kblock_ns)
    else:
        waves = _ceil_div(tiles, grid)
        shell_ns = service["launch_shell_ns"]
        # The last tile of every CTA stores after the mainloop ends.
        store_drain_ns = (grid * 2 * tile_m * tile_n
                          / service["full_chip_tile_store_bytes_per_ns"])
        # Compulsory operand/scale reads and BF16 output writes share the
        # HBM interface, so their measured stream times add.
        read_bytes = (
            m * k + n * k
            + 4 * m * kblocks + 4 * _ceil_div(n, _BLOCK_K) * kblocks)
        write_bytes = 2 * m * n
        groups["fp8_tensor"] = waves * kblocks * max(
            tensor_kblock_ns, ring["stage_floor_ns"])
        groups["dram_stream"] = (
            read_bytes / ring["dram_stream_bytes_per_ns"]
            + write_bytes / service["full_chip_tile_store_bytes_per_ns"])
        groups["launch_shell"] = shell_ns
        groups["tile_store_drain"] = store_drain_ns
        gemm_ns = (shell_ns + max(groups["fp8_tensor"], groups["dram_stream"])
                   + store_drain_ns)
    bottleneck = max(groups, key=lambda name: groups[name])
    replayed = graph_replayed_launch(kernel)
    gemm_floor_ns = small_kernel_span_floor_s(
        hardware, grid, kernel.variant, replayed) * 1e9
    groups["launch_span_floor"] = gemm_floor_ns
    if gemm_floor_ns > gemm_ns:
        gemm_ns = gemm_floor_ns
        bottleneck = "launch_span_floor"
    scale_ns = 0.0
    if launch.algorithm == TRTLLM_SWAP_AB_ALGORITHM:
        if TRTLLM_SCALE_1X128_THREADS != 256:
            raise AssertionError("scale_1x128 geometry drifted")
        scale_grid = TRTLLM_SCALE_1X128_CTAS_PER_SM * grid
        # BF16 input read, FP8 output and FP32 1x128 scales written.
        scale_bytes = 2 * m * k + m * k + 4 * m * kblocks
        scale_ns = max(
            small_kernel_span_floor_s(
                hardware, scale_grid, kernel.variant, replayed) * 1e9,
            scale_bytes / ring["dram_stream_bytes_per_ns"])
        groups["trtllm_scale_1x128"] = scale_ns
    total_ns = scale_ns + gemm_ns
    per_group_s = {name: value * 1e-9 for name, value in groups.items()}
    return KernelPrediction(
        t_total_s=total_ns * 1e-9,
        t_per_group=per_group_s,
        t_launch_s=0.0,
        bottleneck_group=bottleneck,
        t_per_resource=dict(per_group_s),
    )


def roofline_block_fp8_gemm(kernel: KernelSpec, hardware) -> KernelPrediction:
    """Roofline of one block-FP8 call: FP8 tensor peak vs compulsory DRAM."""

    if kernel.launch.algorithm not in BLOCK_FP8_GEMM_ALGORITHMS:
        raise ValueError(
            f"not a block-FP8 GEMM launch: {kernel.launch.algorithm!r}")
    m, n, k = (int(kernel.shape[name]) for name in ("m", "n", "k"))
    kblocks = _ceil_div(k, _BLOCK_K)
    compulsory_bytes = (
        m * k + n * k + 2 * m * n
        + 4 * m * kblocks + 4 * _ceil_div(n, _BLOCK_K) * kblocks)
    tensor_s = (2.0 * m * n * k
                / float(hardware.resource_by_role("compute_tensor_fp8")
                        .capacity))
    dram_s = compulsory_bytes / float(
        hardware.resource_by_role("dram").capacity)
    return prediction_from_ns(max(tensor_s, dram_s) * 1e9, "roofline")


def is_block_fp8_gemm(kernel: KernelSpec) -> bool:
    return kernel.launch.algorithm in BLOCK_FP8_GEMM_ALGORITHMS
