"""NVIDIA H200 fused-MoE grouped-GEMM launch selection.

The Hopper vLLM 0.24.0 stack runs DeepSeek-V3's routed experts through the
Triton ``fused_moe_kernel`` with ``use_fp8_w8a8`` and a [128, 128] weight
block shape.  ``try_get_optimal_moe_config`` resolves the bundled
``E=256,N=<2048/TP>,device_name=NVIDIA_H200,dtype=fp8_w8a8,
block_shape=[128,128].json`` file by the nearest tuned batch key; the
resolved tile/warp/stage configuration and the Triton 3.6 compile facts of
every range are recorded in
``microbench/nvidia_h200/references/fused_moe_tuned_config_vllm_0_24_0_h200.json``
(``deepseek_v3_selection`` / ``compile_metadata``).  The tables below restate
that record; nothing here is a duration.

Launch derivation follows the installed source
(``vllm/model_executor/layers/fused_moe/fused_moe.py`` sha256 57a3f7ef...):

* ``_prepare_expert_assignment`` skips ``moe_align_block_size`` when
  ``num_tokens * top_k * 4 <= global_num_experts`` (no expert map, no
  INT8/INT4 W8A16 block path).  The kernel is then compiled with
  ``naive_block_assignment=True``: every routed token owns one M tile whose
  row 0 is its only valid row, and ``EM = num_tokens * top_k * BLOCK_SIZE_M``.
* otherwise ``moe_align_block_size`` sizes ``sorted_token_ids`` as
  ``numel + E * (BLOCK_SIZE_M - 1)``, capped at ``numel * BLOCK_SIZE_M`` when
  ``numel < E``, and ``invoke_fused_moe_triton_kernel`` further caps ``EM`` at
  ``A.size(0) * top_k * BLOCK_SIZE_M`` when ``A.size(0) < BLOCK_SIZE_M``
  (W1: A = tokens, top_k = topk; W2: A = tokens * topk, top_k = 1).
* ``grid = cdiv(EM, BLOCK_SIZE_M) * cdiv(B.size(1), BLOCK_SIZE_N)`` with
  ``B.size(1) = 2 * intermediate`` (W1) or ``hidden`` (W2); the block is
  ``num_warps * 32`` threads; ``BLOCK_SIZE_K`` is clamped to the 128 block.
* W2 multiplies the router weight (``MUL_ROUTED_WEIGHT=True``), W1 does not.

``owns_select_fused_moe_launch`` declares that enumerated domain; the
dispatcher routes every other request (other dtypes, experts, widths, expert
parallelism and the INT4 ``routed_experts`` aggregate) to the projected A800
package, so this module restates no A800 rule.
"""

from typing import Mapping, Optional

from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.selection_primitives import (
    ceil_div, require_shape, require_variant)
from src.kernel.model_contract import (
    SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER,
)


H200_TUNED_CONFIG_EVIDENCE_REFERENCE = (
    "microbench/nvidia_h200/references/"
    "fused_moe_tuned_config_vllm_0_24_0_h200.json")

FP8_BLOCK_ALGORITHM_PREFIX = "vllm_triton_fused_moe_fp8_block128"
FP8_BLOCK_SIZE = 128

# DeepSeek-V3 routed-expert domain of the bundled H200 FP8 block configs.
_FP8_GLOBAL_EXPERTS = 256
_FP8_TOPK = 8
_FP8_HIDDEN = 7168

# (m_min, m_max, config_id) per bundled file, keyed by N = intermediate per
# rank.  The m ranges are the nearest-key partition of the tuned batch keys.
_FP8_SELECTION_RANGES = {
    512: (
        (1, 1, "cb47dc9924056440"),
        (2, 3, "32e1b6206e8cd459"),
        (4, 6, "0e0b351e3086258f"),
        (7, 12, "96f97a9b394c141b"),
        (13, 20, "cb47dc9924056440"),
        (21, 28, "c707ac95391bd157"),
        (29, 80, "cb47dc9924056440"),
        (81, 112, "96f97a9b394c141b"),
        (113, 192, "0e0b351e3086258f"),
        (193, 384, "cb47dc9924056440"),
        (385, 768, "a2b5ff407c7333d6"),
        (769, 1792, "57bc196faf4fc400"),
        (1793, 32768, "d26bcbdc1237b406"),
    ),
    256: (
        (1, 3, "12d9d4bdf00fb2ff"),
        (4, 6, "35591cb2c794b91b"),
        (7, 12, "12d9d4bdf00fb2ff"),
        (13, 20, "63d65789df08a862"),
        (21, 28, "12d9d4bdf00fb2ff"),
        (29, 40, "b336fde8d25c71c0"),
        (41, 80, "50e8c1da81dc66d8"),
        (81, 112, "63d65789df08a862"),
        (113, 384, "65c0c8e5705f60f2"),
        (385, 768, "63d65789df08a862"),
        (769, 3584, "bc3c0c641b7ec1b3"),
        (3585, 32768, "12974297821ec13c"),
    ),
}

# config_id -> (BM, BN, BK, GROUP_SIZE_M, num_warps, num_stages,
#               W1 registers, W2 registers, dynamic shared bytes,
#               resident CTAs/SM) from the reference's compile_metadata
# (Triton 3.6, sm_90, aligned-assignment specialization).
_FP8_KERNEL_CONFIGS = {
    "cb47dc9924056440": (16, 128, 128, 1, 4, 3, 100, 100, 37376, 4),
    "32e1b6206e8cd459": (16, 128, 128, 16, 8, 4, 80, 76, 55296, 3),
    "0e0b351e3086258f": (16, 128, 128, 16, 4, 3, 100, 100, 37376, 4),
    "96f97a9b394c141b": (16, 128, 128, 32, 4, 3, 100, 100, 37376, 4),
    "c707ac95391bd157": (16, 128, 128, 16, 4, 4, 100, 100, 55808, 4),
    "a2b5ff407c7333d6": (32, 128, 128, 32, 4, 3, 160, 159, 41472, 3),
    "57bc196faf4fc400": (64, 128, 128, 32, 4, 4, 236, 237, 98816, 2),
    "d26bcbdc1237b406": (64, 128, 128, 16, 4, 4, 236, 237, 98816, 2),
    "12d9d4bdf00fb2ff": (16, 128, 128, 16, 4, 4, 100, 100, 55808, 4),
    "35591cb2c794b91b": (16, 128, 128, 64, 4, 4, 100, 100, 55808, 4),
    "63d65789df08a862": (16, 128, 128, 16, 4, 3, 100, 100, 37376, 4),
    "b336fde8d25c71c0": (16, 128, 128, 16, 4, 5, 100, 100, 74240, 3),
    "50e8c1da81dc66d8": (16, 128, 128, 64, 4, 3, 100, 100, 37376, 4),
    "65c0c8e5705f60f2": (16, 128, 128, 32, 4, 3, 100, 100, 37376, 4),
    "bc3c0c641b7ec1b3": (64, 128, 128, 32, 4, 4, 236, 237, 98816, 2),
    "12974297821ec13c": (64, 128, 128, 16, 4, 3, 236, 237, 74240, 2),
}

# SGLang 0.5.12 runs its own ``fused_moe_kernel``
# (moe_runner/triton_utils/fused_moe_triton_kernels.py) with the shared
# expert fused in (E = 257, top-k 9) and the H200 configs
# ``configs/triton_3_5_1/E=257,N=256,device_name=NVIDIA_H200,dtype=fp8_w8a8,
# block_shape=[128, 128].json`` (up) and ``..._down.json`` (down, USE_TMA);
# SGLang logs the triton_3_6_0 -> triton_3_5_1 fallback.  Key -> (BM, BN, BK,
# GROUP_SIZE_M, num_warps, num_stages); ``try_get_optimal_moe_config`` takes
# ``configs[min(keys, key=lambda x: abs(x - M))]`` (ties to the smaller key).
SGLANG_ALGORITHM_PREFIX = "sglang_triton_fused_moe_fp8_block128"
_SGLANG_INTERMEDIATE = 256
_SGLANG_UP_CONFIGS = {
    1: (16, 64, 128, 1, 4, 5), 2: (16, 64, 128, 1, 4, 5),
    4: (16, 64, 128, 32, 4, 5), 8: (16, 64, 128, 1, 4, 5),
    16: (16, 64, 128, 1, 4, 3), 24: (16, 128, 128, 16, 4, 3),
    32: (16, 64, 128, 16, 4, 5), 48: (16, 64, 128, 1, 4, 5),
    64: (16, 64, 128, 16, 4, 4), 96: (16, 64, 128, 16, 4, 5),
    128: (16, 64, 128, 16, 4, 3), 256: (16, 128, 128, 32, 4, 4),
    512: (32, 128, 128, 1, 4, 5), 1024: (64, 128, 128, 1, 4, 4),
}
_SGLANG_DOWN_CONFIGS = {
    1: (16, 128, 128, 16, 4, 2), 2: (16, 128, 64, 64, 4, 3),
    4: (16, 128, 128, 1, 4, 2), 8: (16, 128, 128, 1, 4, 2),
    16: (16, 128, 128, 1, 4, 2), 24: (16, 128, 64, 64, 4, 3),
    32: (16, 256, 64, 1, 4, 3), 48: (16, 128, 64, 1, 4, 3),
    64: (16, 128, 128, 64, 4, 2), 96: (16, 128, 128, 64, 4, 2),
    128: (16, 128, 128, 64, 4, 2), 256: (16, 128, 128, 64, 4, 2),
    512: (32, 128, 128, 16, 4, 2), 1024: (64, 128, 128, 64, 4, 3),
}


def _sglang_nearest(configs, tokens):
    return configs[min(configs, key=lambda key: abs(key - tokens))]


def sglang_fp8_block_program(tile_m, tile_n, tile_k, warps, stages,
                             use_tma):
    """Name of the measured SGLang tile-program row (HWSpec family)."""

    return (f"sgl_{'tma_' if use_tma else ''}bm{tile_m}_bn{tile_n}"
            f"_bk{tile_k}_w{warps}_s{stages}")


def sglang_expanded_rows(tokens: int, topk: int, experts: int,
                         tile_m: int) -> int:
    """SGLang ``moe_align_block_size`` ``sorted_token_ids`` length."""

    numel = tokens * topk
    if numel < experts + 1:
        return numel * tile_m
    return numel + (experts + 1) * (tile_m - 1)


def _sglang_domain(shape, dtype, variant, attributes) -> bool:
    if "fused_shared_experts" not in attributes:
        return False
    if not _fp8_block_domain(shape, dtype, variant, attributes):
        raise ValueError(
            "SGLang fused shared-expert FUSED_MOE is source-backed only for "
            "the DeepSeek-V3 FP8 block domain")
    if shape["intermediate"] != _SGLANG_INTERMEDIATE:
        raise NotImplementedError(
            "SGLang H200 fused-MoE configs are enumerated for N=256 (TP8)")
    return True


def _select_sglang_fp8_block_launch(shape, variant, attributes, hardware):
    tokens, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    projection = require_variant(variant, ("w1", "w2"), "FUSED_MOE")
    fused = attributes["fused_shared_experts"]
    kernel_experts, kernel_topk = experts + fused, topk + fused
    up = _sglang_nearest(_SGLANG_UP_CONFIGS, tokens)
    config = up if projection == "w1" else _sglang_nearest(
        _SGLANG_DOWN_CONFIGS, tokens)
    tile_m, tile_n, tile_k, group_size_m, warps, stages = config
    use_tma = projection == "w2"
    # One sorted_token_ids (sized with the up config's BLOCK_SIZE_M) feeds
    # both projections; the grid divides it by each config's BLOCK_SIZE_M.
    expanded_rows = sglang_expanded_rows(
        tokens, kernel_topk, kernel_experts, up[0])
    projection_n = 2 * intermediate if projection == "w1" else hidden
    grid = ceil_div(expanded_rows, tile_m) * ceil_div(projection_n, tile_n)
    program = sglang_fp8_block_program(
        tile_m, tile_n, tile_k, warps, stages, use_tma)
    family = hardware.grouped_gemm_fp8_block_tile_pipelines
    if family is None or program not in family:
        raise NotImplementedError(
            f"SGLang fused-MoE tile program {program} is not measured")
    return LaunchConfig(
        algorithm=(f"{SGLANG_ALGORITHM_PREFIX}_aligned_"
                   f"{'tma_' if use_tma else ''}{projection}"),
        grid=(grid, 1, 1),
        block=(32 * warps, 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": tile_m, "n": tile_n, "k": tile_k},
            pipeline_depth=stages),
        parameters={
            "group_size_m": float(group_size_m),
            "padded_rows": float(expanded_rows),
            "mte2_tile_bytes": 0.0,
            "weight_tile_rows": 0.0,
            "operand_replay_a": 1.0,
            "operand_replay_b": 1.0,
            "dynamic_shared_memory_bytes": float(
                family[program]["dynamic_shared_memory_bytes_per_cta"]),
            "num_warps": float(warps),
            "num_stages": float(stages),
            SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER: (
                1.0 if projection == "w2" else 0.0),
        },
        execution_variant=projection,
    )




def fp8_block_naive_assignment(tokens: int, topk: int, experts: int) -> bool:
    """vLLM 0.24 ``_prepare_expert_assignment`` naive-block predicate."""

    return tokens * topk * 4 <= experts


def _fp8_block_domain(shape: Mapping[str, int], dtype: Optional[str],
                      variant: Optional[str],
                      attributes: Mapping[str, object]) -> bool:
    if variant is None or variant.strip().lower() not in ("w1", "w2"):
        return False
    if dtype not in ("bf16",):
        return False
    # Both quantization attributes are schema-optional: a legacy spec that
    # declares no routed quantization contract is a BF16 launch.
    if ("routed_weight_dtype" not in attributes
            or attributes["routed_weight_dtype"] != "fp8"):
        return False
    # The Serving FP8 contract records scale granularity as group size 0 for
    # both per-tensor and 128x128 block scales; within the enumerated
    # (E, topk, hidden) domain the model contract admits only DeepSeek-V3,
    # whose checkpoint schema requires [128, 128] weight blocks.
    if attributes["weight_group_size"] != 0:
        return False
    if attributes["expert_parallel_size"] != 1:
        return False
    _, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    return (
        (experts, topk, hidden) == (_FP8_GLOBAL_EXPERTS, _FP8_TOPK, _FP8_HIDDEN)
        and intermediate in _FP8_SELECTION_RANGES)


def fp8_block_kernel_config(tokens: int, intermediate: int) -> dict:
    """Return the bundled H200 FP8 block config vLLM resolves for M tokens."""

    if intermediate not in _FP8_SELECTION_RANGES:
        raise NotImplementedError(
            "H200 FP8 block fused-MoE tuned config is enumerated only for "
            f"N in {sorted(_FP8_SELECTION_RANGES)}, got {intermediate}")
    matches = tuple(
        config_id
        for low, high, config_id in _FP8_SELECTION_RANGES[intermediate]
        if low <= tokens <= high)
    if len(matches) != 1:
        raise NotImplementedError(
            "H200 FP8 block fused-MoE tuned config is enumerated for "
            f"M in [1, 32768], got {tokens}")
    config_id, = matches
    (tile_m, tile_n, tile_k, group_size_m, warps, stages, registers_w1,
     registers_w2, shared_bytes, resident) = _FP8_KERNEL_CONFIGS[config_id]
    return {
        "config_id": config_id,
        "tile_m": tile_m, "tile_n": tile_n, "tile_k": tile_k,
        "group_size_m": group_size_m,
        "num_warps": warps, "num_stages": stages,
        "registers_per_thread": {"w1": registers_w1, "w2": registers_w2},
        "dynamic_shared_memory_bytes": shared_bytes,
        "resident_ctas_per_sm": resident,
    }


def fp8_block_expanded_rows(
        tokens: int, topk: int, experts: int, tile_m: int,
        projection: str) -> int:
    """Return the source ``EM`` passed to ``fused_moe_kernel``."""

    if fp8_block_naive_assignment(tokens, topk, experts):
        return tokens * topk * tile_m
    numel = tokens * topk
    padded = numel + experts * (tile_m - 1)
    if numel < experts:
        padded = min(numel * tile_m, padded)
    if projection == "w1":
        a_rows, kernel_topk = tokens, topk
    elif projection == "w2":
        a_rows, kernel_topk = tokens * topk, 1
    else:
        raise ValueError(f"FUSED_MOE projection must be w1 or w2: {projection!r}")
    if a_rows < tile_m:
        padded = min(padded, a_rows * kernel_topk * tile_m)
    return padded


def _select_fp8_block_launch(
        shape: Mapping[str, int], variant: str) -> LaunchConfig:
    tokens, experts, topk, hidden, intermediate = require_shape(
        shape, ("tokens", "experts", "topk", "hidden", "intermediate"))
    projection = require_variant(variant, ("w1", "w2"), "FUSED_MOE")
    config = fp8_block_kernel_config(tokens, intermediate)
    tile_m, tile_n, tile_k = (
        config["tile_m"], config["tile_n"], config["tile_k"])
    if tile_k != FP8_BLOCK_SIZE:
        raise AssertionError("FP8 block BLOCK_SIZE_K must equal the 128 block")
    projection_n = 2 * intermediate if projection == "w1" else hidden
    expanded_rows = fp8_block_expanded_rows(
        tokens, topk, experts, tile_m, projection)
    grid = ceil_div(expanded_rows, tile_m) * ceil_div(projection_n, tile_n)
    assignment = (
        "naive" if fp8_block_naive_assignment(tokens, topk, experts)
        else "aligned")
    return LaunchConfig(
        algorithm=f"{FP8_BLOCK_ALGORITHM_PREFIX}_{assignment}_{projection}",
        grid=(grid, 1, 1),
        block=(32 * config["num_warps"], 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": tile_m, "n": tile_n, "k": tile_k},
            pipeline_depth=config["num_stages"]),
        parameters={
            "group_size_m": float(config["group_size_m"]),
            "padded_rows": float(expanded_rows),
            "mte2_tile_bytes": 0.0,
            "weight_tile_rows": 0.0,
            "operand_replay_a": 1.0,
            "operand_replay_b": 1.0,
            "dynamic_shared_memory_bytes": float(
                config["dynamic_shared_memory_bytes"]),
            "num_warps": float(config["num_warps"]),
            "num_stages": float(config["num_stages"]),
            SOURCE_FUSED_MOE_W2_ROUTED_WEIGHT_PROGRAM_PARAMETER: (
                1.0 if projection == "w2" else 0.0),
        },
        execution_variant=projection,
    )


def owns_select_fused_moe_launch(shape: Mapping[str, int],
                                 dtype: Optional[str],
                                 variant: Optional[str],
                                 attributes: Mapping[str, object],
                                 hardware) -> bool:
    """Whether the request is in the vLLM 0.24 FP8 block Triton domain."""

    del hardware
    return _fp8_block_domain(shape, dtype, variant, attributes)


def select_fused_moe_launch(shape: Mapping[str, int], dtype: Optional[str],
                            variant: Optional[str],
                            attributes: Mapping[str, object],
                            hardware) -> LaunchConfig:
    if _sglang_domain(shape, dtype, variant, attributes):
        return _select_sglang_fp8_block_launch(
            shape, variant, attributes, hardware)
    del hardware
    if not _fp8_block_domain(shape, dtype, variant, attributes):
        raise ValueError(
            "H200 FUSED_MOE selector owns only the vLLM 0.24 FP8 block "
            "domain; the dispatcher routes other requests to the projected "
            "package")
    return _select_fp8_block_launch(shape, variant)
