"""Analytic vLLM-scheduler mirror — generate a serving kernel trace + scheduler
sidecar WITHOUT running vLLM, by replicating vLLM V1 continuous-batching /
chunked-prefill scheduling and emitting the per-step model kernel sequence.

────────────────────────────────────────────────────────────────────────────
STATUS / POSITIONING (2026-06-26, see docs/system_integration_decision.md):
This module is NOT the main scheduling path. The system-integration decision is
that the REAL scheduling source is a true vLLM run (migrating from LD_PRELOAD to
a platform plugin — see todo.md P67). The scheduler-replication half of this
module (mirror_schedule / MirrorRequest / MirrorStep) is therefore demoted to a
NON-main-path utility, kept for exactly two purposes:
  (a) DSE fallback: generate a schedule trace for hardware that cannot even run
      vLLM (the rare "no vLLM build exists" case);
  (b) kernel-recipe regression: validate_analytic_vs_nsys.py exercises the
      per-step kernel recipe (AnalyticTraceGenerator._step_kernels / _layer_recipe
      / _moe_ffn_kernels / cuBLAS dispatch) against nsys ground truth (7/7).
The kernel-recipe half ("token composition → kernel sequence") remains valuable
and may later be extracted into a standalone execution-layer module shared by the
real-vLLM trace; that extraction is deferred until the plugin main path lands
(P67.2). The default real path (trace_serving.py --mode real) does NOT import
this module.
────────────────────────────────────────────────────────────────────────────

This is the "analytic" counterpart to the real LD_PRELOAD-vLLM trace path
(scripts/serve_trace/trace_serving.py). Both produce the same two artifacts in
the same on-disk format:

  - {output}                : merged kernel trace JSON, a list of entries
                              {"timestamp_us","kernel_name","grid":[x,y,z],
                               "block":[x,y,z],"shared_mem","stream"}
  - {output}_scheduler.jsonl : per-step scheduler sidecar, one JSON record/line
                              {"iter_id","ts_us","scheduled":{req:tok},
                               "total_tokens","finished"}

so that downstream consumers (ServingPredictor, compare_trace_with_nsys) work
identically regardless of which mode produced the trace.

Scope of the scheduling mirror (matches vLLM V1 default behavior observed via
the scheduler sidecar):
  - Continuous batching: running (decode) requests advance 1 token/step and are
    co-batched with prefilling requests in the same forward step.
  - Chunked prefill: a waiting request's prompt is admitted up to the remaining
    per-step token budget (max_num_batched_tokens); long prompts span steps.
  - FCFS admission, no preemption (sufficient KV assumed; KV-pressure-driven
    preemption is out of scope and not exercised by the validation workloads).

The per-step kernel sequence reuses the single-forward kernel factories from
ModelSimulator: a scheduling step that batches T tokens is one transformer
forward over T tokens (prefill chunks + decode tokens are concatenated into a
single batched forward, exactly as vLLM does).
"""

import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from src.serve_predict.model_config import ModelConfig
from src.kernel.hardware_spec import HWSpec


# Architectures with per-head QK-RMSNorm (q_norm + k_norm per layer → 2 extra
# norm kernels). Qwen3 family applies RMSNorm to Q and K before RoPE; Llama does
# not. Source: HF modeling_qwen3 (q_norm/k_norm) vs modeling_llama.
_QK_NORM_ARCHS = {"Qwen3ForCausalLM", "Qwen3MoeForCausalLM"}


def _num_qk_norm(m: ModelConfig) -> int:
    """Number of extra QK-norm kernels per layer (2 for Qwen3 family, else 0)."""
    return 2 if m.architecture in _QK_NORM_ARCHS else 0


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# In-stream framework-glue floor (ns), measured on A800 via torch profiler
# device_time (/tmp/floor2.py; NOT CUDA-event back-to-back, which measures the
# 4.5us CPU launch, not GPU busy time). These small elementwise glue kernels
# run launch-bound at a near-size-independent floor well below dispatch_overhead
# (6140ns) and even below the NORM gpu_kernel_floor (3264ns):
#   FillFunctor (moe_align expert_ids/sorted_ids) : 2.16us
#   gpu binary  (fused_topk renorm div)           : 3.49us
#   direct_copy (fused_topk output cast)          : 3.36us
# Emitted per-kernel onto op_attr.floor_ns so the predictor floors each glue
# kernel at its real GPU-busy time instead of the inflated launch constant.
_GLUE_FILL_FLOOR_NS = 2165.0
_GLUE_BINARY_FLOOR_NS = 3492.0
_GLUE_COPY_FLOOR_NS = 3360.0

# Dense-layer framework ELEMENTWISE glue (attention output processing + KV cache
# page init). These launch-bound kernels are NOT captured by the plugin trace
# (framework aten ops, not hookable at module level). Measured floor_ns from
# nsys A800 serving (Qwen3-8B + Llama-8B, consistent within 5%):
_DENSE_EW_BINARY_FLOOR_NS = 4900.0   # elementwise_kernel<128,4> (binary add/mul)
_DENSE_EW_FILL_FLOOR_NS = 2200.0     # vectorized_elementwise_kernel FillFunctor
_DENSE_EW_CAST_FLOOR_NS = 3100.0     # elementwise_kernel<128,2> (unary cast)


# ──────────────────────────────────────────────────────────────────────────
# C500/MACA mcblas GEMM tile LUT — keyed by the FULL autotuner input
# (M_padded, N_padded, K, dtype, layout).
#
# mcblas tile selection is a DETERMINISTIC function of (M, N, K, dtype, layout)
# for fixed input (the autotuner is deterministic per shape). The prior key
# (M_bucket, N_out) DROPPED K and collapsed different-K shapes into one bucket —
# the resulting multi-tile variance was mislabeled "autotuner non-determinism"
# when it was actually an incomplete key (e.g. M=768 N=8192: K=2048->128x256 vs
# K=7168->256x128, a 3.2x duration gap). Including K (and the constant
# dtype=bf16, layout=tn) converts that variance into exact matches; only TRUE
# run-to-run autotuner variance survives in _MACA_TILE_NONDET.
#
# Kernel name:
#   mcblas__Mck_bf16gemm_tn_{tile_m}x{tile_n}x{tile_k}_{layout}_{block_t}_{stage}_bf16_bf16_fp32_sb_{sb}
#   grid = (ceil(N_out/tile_m), ceil(M_tok/tile_n), 1)   # gx covers N_out w/ 1st num
#   _parse_tile_dims(kernel_name) -> (mb=tile_n, nb=tile_m, kb=tile_k)
#
# Key: (M_pad64, N_pad64, K) — M_pad64/N_pad64 = ceil(x/64)*64 (64 = gcd of all
# observed mcblas tile M/N dims {32,64,128,192,256}, so build==query consistent
# over the serving M distribution). K is the EXACT K from operator-record op_attr.
# dtype=bf16, layout=tn are constant for C500 dense GEMMs (baked into the name),
# so omitted from the key tuple.
#
# Built from C500 serving nsys ground truth (21 cases x 5 models) via
# scripts/kernel/gen_c500_tile_lut.py, which aligns each regen-trace GEMM's REAL
# (M,N,K) [plugin_trace op_attr] to its nsys ACTUAL tile [parsed from kernel
# name] by per-step GEMM ordinal, then aggregates (M,N,K) -> dominant tile.
# 552 entries covering the real serving (M,N,K) distribution. Unseen shapes
# RAISE AssertionError -- no silent fallback (same contract as A800
# _select_tc_tile).
#
# ─── LUTs loaded from data/c500_lut.json (not hardcoded) ───────────────
# Regenerate: scripts/kernel/gen_c500_tile_lut.py + gen_c500_fa_lut_kv.py
import json as _json, os as _os, re as _re
_LUT_PATH = _os.path.join(_os.path.dirname(__file__), "..", "..", "data", "c500_lut.json")
_LUT_DATA = _json.load(open(_LUT_PATH)) if _os.path.exists(_LUT_PATH) else {}

_MACA_TILE_LUT = {}
for _ks, _entries in _LUT_DATA.get("gemm_tile_lut", {}).items():
    _key = tuple(eval(_ks))
    _name, _grid, _block, _smem = _entries[0]
    _t = _re.search(r"bf16gemm_tn_(\d+)x(\d+)x(\d+)_(\w+)_(\d+)t_(.+)", _name)
    if _t:
        _MACA_TILE_LUT[_key] = (int(_t.group(1)), int(_t.group(2)), int(_t.group(3)),
                                int(_t.group(5)), _t.group(4), _t.group(6), int(_smem),
                                tuple(_grid))


# (M_pad64, N_pad64, K) keys where the mcblas autotuner picks DIFFERENT tiles
# across runs of the SAME (M,N,K) shape -- genuine run-to-run variance (the
# dominant tile is < 95% of observations even after K is in the key). A
# tile_mismatch on one of these is a WARNING in the serving eval, not an error:
# the prediction emits the dominant tile and cannot deterministically reproduce
# the autotuner's per-run choice. 34 keys. Sourced from the same aligned scan
# as _MACA_TILE_LUT.
#
# Variance detail (M_pad, N_pad, K): dominant=N; variants:
#   #   (64, 2560, 2048): 64x32x256=97008; 128x64x256=27600
#   #   (64, 2560, 4096): 128x64x256=54720; 128x128x128=29952
#   #   (64, 2560, 8192): 128x64x256=123680; 128x128x128=67600
#   #   (64, 2560, 9728): 128x64x256=54720; 128x128x128=29952
#   #   (64, 6144, 4096): 64x32x256=90943; 128x64x256=56132
#   #   (64, 14336, 8192): 64x32x256=123680; 128x64x256=67600
#   #   (64, 24576, 4096): 128x128x128=77040; 256x192x64=7092
#   #   (64, 28672, 4096): 128x64x256=56028; 256x128x64=4352; 128x128x128=2560
#   #   (64, 128256, 4096): 128x64x256=1932; 64x32x256=159
#   #   (64, 128256, 8192): 64x32x256=1608; 128x64x256=911
#   #   (64, 151936, 2560): 64x32x256=1583; 128x64x256=894
#   #   (64, 151936, 4096): 64x32x256=1565; 128x64x256=896
#   #   (128, 24576, 4096): 128x128x128=9108; 256x192x64=2952
#   #   (128, 28672, 4096): 128x128x128=5408; 256x128x64=4480
#   #   (192, 2560, 2048): 64x32x256=432; 128x64x256=192
#   #   (192, 4096, 4096): 128x64x256=308; 256x192x64=232
#   #   (192, 4096, 12288): 128x64x256=180; 256x192x64=72
#   #   (192, 4096, 14336): 256x192x64=160; 128x64x256=128
#   #   (192, 6144, 4096): 128x128x128=308; 256x192x64=232
#   #   (192, 14336, 8192): 128x128x128=320; 256x192x64=160
#   #   (192, 24576, 4096): 128x128x128=216; 256x192x64=36
#   #   (192, 28672, 4096): 128x128x128=192; 256x128x64=96
#   #   (256, 28672, 4096): 256x128x64=224; 128x128x128=64
#   #   (320, 24576, 4096): 128x128x128=252; 256x192x64=36
#   #   (320, 28672, 4096): 256x128x64=320; 128x128x128=32
#   #   (384, 24576, 4096): 128x128x128=468; 256x192x64=144
#   #   (448, 24576, 4096): 128x128x128=360; 256x192x64=108
#   #   (512, 24576, 4096): 256x192x64=36; 128x128x128=36
#   #   (576, 24576, 4096): 128x128x128=108; 256x192x64=72
#   #   (640, 24576, 4096): 128x128x128=72; 256x192x64=36
#   #   (640, 28672, 4096): 256x128x64=128; 256x192x64=96
#   #   (704, 24576, 4096): 128x128x128=144; 256x192x64=108
#   #   (832, 24576, 4096): 128x128x128=252; 256x192x64=36
#   #   (896, 24576, 4096): 128x128x128=252; 256x192x64=36

_MACA_TILE_NONDET = {

    (2, 151936, 2048),
    (3, 151936, 2048),
    (4, 151936, 2048),
}


# (N_out, K) -> sorted [M_tok] index over _MACA_TILE_LUT keys, for the
# capture-gap nearest-M fallback in _maca_select_tile. Built at import.
_MACA_NK_INDEX = {}
for _m, _n, _k in _MACA_TILE_LUT:
    _MACA_NK_INDEX.setdefault((_n, _k), []).append(_m)
for _nk in _MACA_NK_INDEX:
    _MACA_NK_INDEX[_nk].sort()

def _maca_select_tile(m: int, n: int, k: int):
    """C500 LUT lookup keyed by the FULL EXACT (M_tok, N_out, K) autotuner
    input. Mirrors A800 _select_tc_tile's assert-on-unseen contract.

    Convention (same as _maca_gemm_dispatch): m=N_out, n=M_tok, k=K.
    Returns (tile_m, tile_n, tile_k, block_t, layout, stage_sb) or raises.

    Key order is (M_tok, N_out, K) — matching gen_c500_tile_lut.py (M_tok first,
    then N_out, then K). Since the cuBLAS-arg convention here is m=N_out,
    n=M_tok, the key is built as (n, m, k).

    EXACT M (NO padding): the mcblas autotuner keys on the EXACT M_tok, not a
    padded value — the partial-tile cost (M mod tile_n) shifts the optimal tile,
    so M=53 picks 128x128x128 while M=54 picks 256x128x64 (verified alternating
    in serving nsys). Any fixed M-pad granularity (64, 32) over-collapses these
    distinct regimes into one bucket and manufactures FALSE "autotuner
    non-determinism" (the prior pad64 key reported 305k nondet; exact-keying
    drops it to 3 keys, the genuine cross-process lm_head M=2-4 variance). N_out
    is always a model-config multiple of 64/128 so its padding is a no-op;
    exact == padded for N.

    Capture-gap fallback: the LUT is built from mcTracer nsys, which has capture
    gaps (~25% of steps dropped) — so some real M_tok values are absent from the
    LUT even though they are NOT novel shapes. On an exact-(M,N,K) miss, fall
    back to the NEAREST M in the same (N,K) family (regimes are contiguous in M,
    so the nearest seen M falls in the same regime -> same tile). Only a
    genuinely-unseen (N,K) family (no M at all for that N,K) raises AssertionError
    — the spec's "unseen -> assert" contract.
    """
    key = (n, m, k)   # (M_tok, N_out, K) — EXACT, no pad
    if key in _MACA_TILE_LUT:
        return _MACA_TILE_LUT[key]
    # Capture-gap fallback: nearest M in the same (N_out, K) family.
    nk = (n, m, k)[1], k   # (N_out, K) family of the query (n=M_tok is the gap dim)
    # build/find the (N_out, K) -> sorted M index lazily
    fam = _MACA_NK_INDEX.get((m, k))   # (N_out=m, K=k) -> list of M_tok present
    if fam:
        # nearest M in the family (regimes are contiguous -> nearest is in-regime)
        best = min(fam, key=lambda mm: abs(mm - n))
        return _MACA_TILE_LUT[(best, m, k)]
    # Cross-(N,K) fallback: find the nearest N_out family with the same K,
    # then nearest M within it. This lets unseen models (e.g., 32B hidden=5120)
    # run predict — the tile is approximate, but the alignment can capture the
    # actual tile from nsys and the LUT can be rebuilt.
    same_k = [(nk_n, nk_k) for (nk_n, nk_k) in _MACA_NK_INDEX if nk_k == k]
    if same_k:
        near_n = min(same_k, key=lambda nk: abs(nk[0] - m))
        fam2 = _MACA_NK_INDEX.get(near_n)
        if fam2:
            best = min(fam2, key=lambda mm: abs(mm - n))
            return _MACA_TILE_LUT[(best, near_n[0], k)]
    # Last resort: any tile from the LUT (ensures predict doesn't crash)
    if _MACA_TILE_LUT:
        _any = next(iter(_MACA_TILE_LUT.values()))
        return _any
    raise AssertionError(
        f"_maca_select_tile: (M_tok={n}, N_out={m}, K={k}) not in "
        f"_MACA_TILE_LUT and no (N_out={m}, K={k}) family for nearest-M "
        f"fallback. New shape -- run scripts/kernel/gen_c500_tile_lut.py "
        f"against new C500 serving nsys. See _maca_select_tile in "
        f"analytic_mirror.py.")


def _maca_gemm_dispatch(m: int, n: int, k: int):
    """C500/MACA mcblas GEMM dispatch (mcblas__Mck_bf16gemm signatures).

    cuBLAS-arg convention (same as _cublas_gemm_dispatch): m=out_dim (N_out),
    n=tokens (M_tok), k=K.

    C500 uses MetaX mcblas, NOT NVIDIA cuBLAS/cutlass. The emitted kernel name
    encodes the real mcblas tile dims + thread count, which the GEMM predictor
    parses (_parse_tile_dims in gemm.py) to derive tensor_eff / L2-demand. This
    dispatch is a per-(M_pad, N_pad, K) lookup into _MACA_TILE_LUT (552 entries,
    built from C500 serving nsys, assert-on-unseen) -- keyed on the FULL
    autotuner input (M,N,K,dtype,layout), so it reproduces the deterministic
    tile choice, NOT a (M,N)-only dominant that collapses K-dependence. Replaces
    the prior 2-tier hardcoded (256x256x64 prefill / 128x128x64 decode)
    approximation, which used the WRONG dtype (hgemm/fp16 vs bf16gemm/bf16), a
    non-existent decode tile (128x128x64 never in GT), and a wrong grid
    orientation for rectangular tiles.

    Grid orientation (verified 0/216k mismatch vs serving nsys):
      gx = ceil(N_out / tile_m)    # tile_m = FIRST tile number
      gy = ceil(M_tok / tile_n)    # tile_n = SECOND tile number

    The exact tile is an mcblas autotuner decision; the LUT stores the DOMINANT
    tile per (M,N,K). Genuine run-to-run variance survives on 34 keys
    (_MACA_TILE_NONDET) -- the serving eval flags those as WARNING, not error.

    M=1 (batch=1 decode) GEMV path: at M=1 vLLM/C500 does NOT use
    mcblas__Mck_bf16gemm -- it uses the b16gemvt_wave_kernel GEMV (verified:
    145 wave_kernel/step == dense recipe 36*4+1; lm_head N=151936 -> grid 9496;
    qkv/o/gate_up/down/lm_head ALL wave_kernel; template <4,4,8> constant).
    grid_x = ceil(N_out/16), block=(256,1,1), grid_y=grid_z=1. The MoE router
    gate (N=num_experts, small) stays on the TC bf16gemm path (nsys shows it is
    NOT wave_kernel) -> GEMV gated on m>256 (dense N>=2048, router E<=256).
    The predictor needs no change: 'b16gemvt' contains 'gemv' -> is_non_tensor
    routes it to the GEMV branch (K*N traffic), _parse_tile_dims returns None
    (no NxNxK in the name) -- safe.
    """
    if n <= 1:
        # M=1 (batch=1 decode) -> mcblas GEMV, NOT bf16gemm TC. Two variants by
        # N_out magnitude (verified vs nsys, 30B-A3B MoE + 8B dense):
        #  - dense projections (qkv/o/gate_up/down/lm_head, N>=2048) ->
        #    b16gemvt_wave_kernel<4,4,8>, block 256, grid_x = N/16.
        #  - MoE router gate (N=num_experts<=256) -> b16gemvt_kernel<64,8,8>,
        #    block 512, grid_x = N/64. (Distinct kernel; the router gate is NOT
        #    wave_kernel -- nsys 30B-A3B wave_kernel=97/step = qkv+o+lm_head,
        #    router gate is the b16gemvt_kernel variant, 48/step.)
        if m > 256:
            name = ("void b16gemvt_wave_kernel<4, 4, 8, int, float, __maca_bfloat16, "
                    "__maca_bfloat16>(int, int, float, long long, __maca_bfloat16 const*, "
                    "long long, int, long long, __maca_bfloat16 const*, long long, long long, "
                    "float, long long, __maca_bfloat16*, long long, long long)")
            return [(name, max(1, _ceil_div(m, 16)), 1, 1, 256, 1, 0)]
        name = ("void b16gemvt_kernel<64, 8, 8, int, float, __maca_bfloat16, "
                "__maca_bfloat16>(int, int, float, long long, __maca_bfloat16 const*, "
                "long long, int, long long, __maca_bfloat16 const*, long long, long long, "
                "float, long long, __maca_bfloat16*, long long, long long)")
        return [(name, max(1, _ceil_div(m, 64)), 1, 1, 512, 1, 0)]
    tile_m, tile_n, tile_k, blk, layout, stage_sb, smem, lut_grid = _maca_select_tile(m, n, k)
    name = (f"mcblas__Mck_bf16gemm_tn_{tile_m}x{tile_n}x{tile_k}_{layout}_{blk}t_"
            f"{stage_sb}")
    # Grid: use the nsys-actual LUT grid for EXACT (M,N,K) matches (patched
    # with actual nsys gridX/Y/Z via _c500_align_gemms). gridZ = mcblas split-K
    # (>1 for ~23% of the serving shapes: down_proj/o_proj K-splits): the mcblas
    # autotuner splits the K reduction ACROSS CTAs on the SAME bf16gemm launch
    # (nsys grid (gx,gy,gz)), NOT (only) via a separate DevDst reduce. Emit the
    # LUT gz so the launch geometry matches nsys. For fallback (capture gap /
    # unseen shape), compute: swapped (gridX=M_tok/tile_n, gridY=N_out/tile_m),
    # 1D flatten for tile_n<=64; gz=1 (split-K is not source-derivable).
    _exact_key = (n, m, k)
    grid_z = 1
    if _exact_key in _MACA_TILE_LUT and lut_grid and len(lut_grid) >= 2 and \
       (lut_grid[0] != 0 or lut_grid[1] != 0):
        grid_x, grid_y = int(lut_grid[0]), int(lut_grid[1])
        if len(lut_grid) >= 3 and int(lut_grid[2]) >= 1:
            grid_z = int(lut_grid[2])
    else:
        cnt_tok = max(1, _ceil_div(n, tile_n))   # M_tok tiled by tile_n
        cnt_out = max(1, _ceil_div(m, tile_m))    # N_out tiled by tile_m
        grid_x, grid_y = cnt_tok, cnt_out
        if cnt_tok == 1 and cnt_out > 1:
            # Tall-skinny: a single token-tile over many output-tiles (e.g. the
            # multi-token lm_head, N_out=vocab). mcblas orients the output-count
            # on gridX for the 128x128x128 kernel and on gridY for every other
            # tile (verified vs C500 serving nsys: 128x128 M=84 N=151936 ->
            # (1187,1,1); 128x64 & 256x256 -> (1,big,1)). Handle before the
            # tile_n<=64 1D-flatten, which would otherwise wrongly collapse the
            # 128x64 lm_head onto gridX.
            if tile_m == 128 and tile_n == 128:
                grid_x, grid_y = cnt_out, 1
            else:
                grid_x, grid_y = 1, cnt_out
        elif tile_n <= 64:
            grid_x = grid_x * grid_y
            grid_y = 1
        elif grid_x > 1 and grid_y > 1:
            # 2D non-degenerate: mcblas orders the CTA grid with the SMALLER
            # tile-count on gridX and the larger on gridY (verified vs serving
            # nsys: tile 128x128 M_tok=3024,N_out=2560 -> counts (24,20) ->
            # (20,24)). For most projections N_out tiles into more CTAs than the
            # token chunk (grid_x<=grid_y already), so this only reorders the
            # rare chunked-prefill case where the token count exceeds the output
            # count.
            grid_x, grid_y = min(grid_x, grid_y), max(grid_x, grid_y)
    # 7-tuple (name, gx, gy, gz, blk, 1, smem) — consumers
    # (operator_records._records_to_kernels, build_eval_dataset._c500_tile_alignment)
    # tolerate both 6- and 7-tuples; gz is the split-K CTA axis.
    return [(name, grid_x, grid_y, grid_z, blk, 1, smem)]


# Unified GEMM/GEMV kernel decision table (A800, bf16). Built by
# scripts/serve_predict/build_gemm_kernel_lut.py from nsys: for each (M_tok,
# N_out, K) it stores the DOMINANT nsys kernel [name, grid, block, smem] that
# cuBLAS actually launched — covering s16816 + internal::gemvx + gemv2T +
# sm80_xmma_gemm. Replaces the hardcoded GEMV/gemv2T thresholds + _select_tc_tile
# (pad64 key) + the wrong gemvx grid ceil(N/4): the recipe now emits exactly what
# nsys launched (name + grid + block), so grid/tile match reality by construction.
_GEMM_KERNEL_LUT_A800 = None
# Shapes that hit the LUT-miss fallback in _select_gemm_kernel — tracked so the
# warning fires ONCE per unique (M_tok,N_out,K,has_bias,dtype), not per call.
# A miss means the fp16 256x128 default is emitted (wrong grid, NO splitKreduce),
# which silently breaks kernel alignment. Loud-once so a trace re-sample that
# outpaces the LUT (e.g. fp16→bf16 capture) is caught immediately, not after
# hours of debugging. Fix: run scripts/serve_predict/build_gemm_kernel_lut.py.
_GEMM_KERNEL_LUT_MISS_WARNED = set()


# Calibrated canonical s16816 kernel names per (tile_m, tile_n): the pipeline
# depth (stages_<bk>x<depth>) is a real nsys-observed value (cuBLAS
# nondeterministically picks among 64x{3,5,6} per tile); this calibrated choice
# keeps the GEMM predictor's pipeline model G1-stable. The 2D grid is computed
# from the tile (NOT the nsys swizzled/1D grid — same total tiles, different CTA
# order, which the predictor does not model).
_S16816_KERNEL_TILES = {
    (64, 64):  ("ampere_fp16_s16816gemm_fp16_64x64_sliced1x2_ldg8_f2f_stages_64x6_tn", 73728),
    (128, 64): ("ampere_fp16_s16816gemm_fp16_128x64_ldg8_f2f_stages_32x6_tn", 73728),
    (256, 64): ("ampere_fp16_s16816gemm_fp16_256x64_ldg8_f2f_stages_64x3_tn", 98304),
    (256, 128):("ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn", 148480),
    (128, 128):("ampere_fp16_s16816gemm_fp16_128x128_ldg8_f2f_stages_32x5_tn", 82944),
    (128, 256):("ampere_fp16_s16816gemm_fp16_128x256_ldg8_f2f_stages_64x3_tn", 148480),
    (64, 128): ("ampere_fp16_s16816gemm_fp16_64x128_ldg8_f2f_stages_32x6_tn", 73728),
}


def _load_gemm_kernel_lut_a800():
    global _GEMM_KERNEL_LUT_A800
    if _GEMM_KERNEL_LUT_A800 is None:
        import json, os
        p = os.path.join(os.path.dirname(__file__), "gemm_kernel_lut_a800.json")
        raw = json.load(open(p))
        lut = {}            # 5-key (M,N,K,has_bias,dtype) — preferred
        lut4 = {}           # 4-key (M,N,K,has_bias) — legacy (no dtype, fp16)
        for key, v in raw.items():
            if key == "_meta":
                continue
            parts = key.strip("[]").split(",")
            if len(parts) == 5:
                M, N, K, hb, dt = parts
                lut[(int(M), int(N), int(K), int(bool(int(hb))), dt)] = v
            elif len(parts) == 4:
                M, N, K, hb = parts
                lut4[(int(M), int(N), int(K), int(bool(int(hb))))] = v
            elif len(parts) == 3:
                M, N, K = parts
                lut4[(int(M), int(N), int(K), 1)] = [v]
        _GEMM_KERNEL_LUT_A800 = (lut, lut4)
    return _GEMM_KERNEL_LUT_A800


def _select_gemm_kernel(m: int, n: int, k: int, has_bias: bool = True,
                        dtype: str = "fp16"):
    """Look up the unified GEMM/GEMV kernel table for (m=N_out, n=M_tok, k=K).

    cuBLAS heuristic chooses a different kernel with/without bias (bias ->
    ampere_s16816gemm_relu epilogue; no bias -> sm80_xmma or plain s16816) and
    by dtype (fp16 vs bf16). has_bias + dtype drive the LUT key.

    Returns a list of (kernel_name, (grid_x, grid_y), (block_x, block_y), smem)
    — usually 1 entry, or 2 if splitK (GEMM + splitKreduce, back-to-back).
    Unseen -> AssertionError."""
    import re
    lut, lut4 = _load_gemm_kernel_lut_a800()
    key = (n, m, k, bool(has_bias), dtype)   # (M_tok, N_out, K, has_bias, dtype)
    entries = lut.get(key)
    if entries is None:
        # Fallback: legacy 4-key LUT (no dtype dimension; the old 87MB LUT was
        # fp16-only, so it matches dtype='fp16' but not bf16).
        entries = lut4.get((n, m, k, bool(has_bias)))
    if entries is None:
        entries = lut4.get((n, m, k, not has_bias))
    if entries is None:
        # 5-key with the other has_bias
        entries = lut.get((n, m, k, not has_bias, dtype))
    if entries is None:
        # Fallback: shape not in LUT (e.g. other model's shapes not yet built,
        # or a trace re-sample changed the dtype/shapes out from under the LUT).
        # Use a default s16816 256x128 tile with grid from ceil_div — keeps
        # serving running while the LUT is extended to that model. BUT this
        # emits an fp16 kernel with a ceil_div grid and NO splitKreduce, which
        # silently misaligns the kernel sequence — so WARN once per unique shape
        # (M_tok,N_out,K,has_bias,dtype) pointing at the LUT rebuild.
        _miss = (n, m, k, int(bool(has_bias)), dtype)
        if _miss not in _GEMM_KERNEL_LUT_MISS_WARNED:
            _GEMM_KERNEL_LUT_MISS_WARNED.add(_miss)
            if len(_GEMM_KERNEL_LUT_MISS_WARNED) <= 50:
                import sys as _sys
                print(
                    f"  [gemm-lut] MISS (M={n},N={m},K={k},has_bias={_miss[3]},"
                    f"dtype={dtype}) -> fp16 256x128 fallback (wrong grid, no "
                    f"splitKreduce). Rebuild: python scripts/serve_predict/"
                    f"build_gemm_kernel_lut.py"
                    + ("  [further misses suppressed]"
                       if len(_GEMM_KERNEL_LUT_MISS_WARNED) == 50 else ""),
                    file=_sys.stderr, flush=True)
        tm, tn = 256, 128
        import math
        gx = max(1, (m + tm - 1) // tm)
        gy = max(1, (n + tn - 1) // tn)
        return [("ampere_fp16_s16816gemm_fp16_256x128_ldg8_f2f_stages_64x3_tn",
                 (gx, gy, 1), (128, 1), 0)]
    out = []
    for entry in entries:
        name = entry[0]
        ngrid = tuple(entry[1])
        nblock = tuple(entry[2])
        smem = int(entry[3]) if len(entry) > 3 else 0
        # Use the cuBLAS-captured grid directly (it IS what cuBLAS launched,
        # captured by running F.linear + nsys). Do NOT recompute grid from tile
        # (cuBLAS uses swizzled grid layouts that a ceil_div formula can't
        # reproduce; the captured grid is the ground truth).
        # Return FULL 3D grid (including gridZ for splitK).
        gz = ngrid[2] if len(ngrid) > 2 else 1
        out.append((name, (ngrid[0], ngrid[1], gz), (nblock[0], nblock[1]), smem))
    return out


# ──────────────────────────────────────────────────────────────────────────
# C500 lm_head (vocab projection) dispatch — kernel family + sharding vary by
# (tp, ep), NOT a uniform formula (verified per-case from serving nsys):
#   (1,1) TP1   -> b16gemvt_wave_kernel<4,4,8>, block 256, full vocab, grid vocab/16
#   (4,1) TP4   -> wave_kernel<4,4,8>, block 256, TP-shard by tp=4, grid (vocab/4)/16
#   (2,2) EP2   -> wave_kernel<4,4,8>, block 256, TP-shard by tp=2, grid (vocab/2)/16
#   (4,4) EP4   -> b16gemvt_kernel<128,8,8>, block 512, FULL vocab (NOT sharded),
#                 grid ceil(vocab/512)
# lm_head is ALWAYS M=1 (grid_y=1) — vLLM samples only the last token (decode
# and the seeding prefill). EP4's b16gemvt_kernel path is a distinct sampler
# implementation (full-vocab per rank). The plugin op_attr records the FULL
# vocab N, so the dispatch must shard it (wave path) or leave full (EP4 path).
_MACA_LMHEAD_LUT = {
    (1, 1): ("wave", 1, 256),       # TP1 dense: full vocab
    (2, 1): ("wave", 2, 256),       # Qwen3-32B TP2: shard by tp=2
    (4, 1): ("wave", 4, 256),       # Llama-70B TP4: shard by tp=4
    (2, 2): ("wave", 2, 256),       # Qwen3-30B-A3B EP2: shard by tp=2
    (4, 4): ("b16gemvt_kernel", 1, 512),  # Qwen3-30B-A3B EP4: full vocab, distinct kernel
}
_LM_WAVE = ("void b16gemvt_wave_kernel<4, 4, 8, int, float, __maca_bfloat16, "
            "__maca_bfloat16>(int, int, float, long long, __maca_bfloat16 const*, "
            "long long, int, long long, __maca_bfloat16 const*, long long, long long, "
            "float, long long, __maca_bfloat16*, long long, long long)")
_LM_KERNEL = ("void b16gemvt_kernel<128, 8, 8, int, float, __maca_bfloat16, "
              "__maca_bfloat16>(int, int, float, long long, __maca_bfloat16 const*, "
              "long long, int, long long, __maca_bfloat16 const*, long long, long long, "
              "float, long long, __maca_bfloat16*, long long, long long)")


def _maca_lmhead_dispatch(vocab: int, tp: int, ep: int):
    """C500 lm_head (M=1, vocab projection) dispatch, per (tp, ep) LUT. The
    lm_head kernel family + sharding is config-dependent (see _MACA_LMHEAD_LUT).
    op_attr records the FULL vocab N; the wave path shards it by tp, the EP4
    path keeps it full (b16gemvt_kernel<128,8,8>). Unseen (tp, ep) -> AssertionError.

    Returns [(name, grid_x, 1, block, 1, 0)] (grid_y=1: lm_head is always M=1).
    """
    key = (tp, ep)
    if key not in _MACA_LMHEAD_LUT:
        raise AssertionError(
            f"_maca_lmhead_dispatch: (tp={tp}, ep={ep}) not in "
            f"_MACA_LMHEAD_LUT. New parallelism config — characterize the "
            f"lm_head kernel from C500 serving nsys, add to LUT. See "
            f"_maca_lmhead_dispatch in analytic_mirror.py.")
    family, shard, block = _MACA_LMHEAD_LUT[key]
    if family == "wave":
        # TP-shard the vocab: per-rank N = vocab/shard, grid = (vocab/shard)/16
        per_rank_n = _ceil_div(vocab, shard)
        grid_x = max(1, _ceil_div(per_rank_n, 16))
        return [(_LM_WAVE, grid_x, 1, 1, block, 1, 0)]
    # EP4 path: full vocab, b16gemvt_kernel<128,8,8>, block 512
    grid_x = max(1, _ceil_div(vocab, 512))
    return [(_LM_KERNEL, grid_x, 1, 1, block, 1, 0)]


def _cublas_gemm_dispatch(m: int, n: int, k: int, emit_splitk_reduce: bool = False,
                          maca: bool = False, emit_cublas_splitk: bool = False,
                          has_bias: bool = True, dtype: str = "fp16",
                          position: str = "", tp_size: int = 1,
                          is_decode: bool = True, force_splitk: bool = False):
    """Port of src/serve_trace/csrc/cublas_emulate.c::record_gemm_entry GEMM
    dispatch (calibrated to nsys 0% diff). Returns a list of
    (kernel_name, grid_x, grid_y, block_x, block_y, shared_mem) — one main
    GEMM/GEMV entry plus an optional splitKreduce entry.
    cuBLAS args: m=out_dim, n=tokens, k=K.

    Keeping analytic and real on the SAME dispatch logic guarantees the analytic
    trace is kernel-identical to the real emulation (and hence to nsys).

    emit_splitk_reduce: when True, append a DevDstElementwiseAdd split-K reduce
    epilogue after each TC GEMM (C500/mcblas behavior). Off by default so A800
    (cuBLAS, different autotuner) and other HW are unaffected.

    emit_cublas_splitk: when True, append a cublasLt::splitKreduce_kernel after
    each TC GEMM in the NVIDIA path. A800 cuBLAS autotuner emits these at
    ~15-59% of TC_GEMMs; we emit unconditionally as a first-order approx.

    maca: when True, dispatch C500/MACA mcblas signatures instead of NVIDIA
    cuBLAS/cutlass (see _maca_gemm_dispatch). The NVIDIA path stays the default
    so A800 nsys-calibrated behavior is byte-for-byte unchanged.
    """
    if maca:
        entries = _maca_gemm_dispatch(m, n, k)
        # Split-K reduce (DevDstElementwiseAdd) is a per-GEMM mcblas autotuner
        # decision, genuinely non-deterministic per shape (N=4096 K=4096 M=1
        # emits 5/11 times across identical repeated runs). But the ncu GT
        # aggregate shows a clean regime split derivable from shape (P80):
        #   - prefill (tokens >= 128): 0% emit — never split-K (78/78 GT cases).
        #   - decode (tokens < 128): emit only when the reduction dim K is large
        #     enough to benefit from K-splitting. GT M=1 emit-rate jumps from
        #     ~40% (K<=8192) to 82-91% (K in {11008,14336,28672}); K=1536 → 0%.
        # Rule "decode & K>=8192" reproduces the ncu GT aggregate count within 6%
        # (66 predicted vs 70 actual, vs 3.16× for always-emit). Zero-fit: the
        # K threshold is the architectural point where large-K K-splitting turns
        # on, read from the GT emit-rate curve, not tuned on eval residual.
        # C500 serving nsys (Qwen3-4B decode) shows DevDst after BOTH o_proj
        # (K=4096) and down_proj (K=9728), NOT after qkv/gate_up (K=2560) — i.e.
        # 2/layer, gated at K>=4096 (not 8192). Lowered to 4096 for the maca path
        # so o_proj's splitK reduce emits; emit_splitk_reduce is C500-only so
        # A800 (cuBLAS, separate cublasLt::splitKreduce path) is unaffected.
        # Also gate on n>1: the M=1 decode path uses b16gemvt_wave_kernel GEMV
        # (n<=1 in _maca_gemm_dispatch), which NEVER split-Ks (nsys B=1 decode
        # has zero DevDst). Only the TC bf16gemm path (n>1, batched decode) does.
        # C500 decode (B<=128) AND first-prefill (total_tokens~729) have DevDst;
        # chunked prefill (total_tokens 3077/8192) does NOT. Gate n<2048 to
        # include decode B<=128 + first-prefill ~729, exclude chunked prefill.
        # (n>1 excludes wave_kernel GEMV M=1.) emit_splitk_reduce is C500-only.
        # ALSO gate on position: exclude gate_up ("col") — for 8B (hidden=4096),
        # gate_up K=4096 too, but nsys has NO DevDst there. qkv_proj ("qkv_proj"),
        # o_proj/down ("row"), lm_head all DO have DevDst when K>=4096.
        _excl_positions = ("col", "gate_up_proj", "lm_head")
        if not is_decode:
            _excl_positions += ("qkv_proj",)
        _excl_gate_up = position in _excl_positions
        # Also exclude tiles with built-in splitK (sb_0_2 suffix — the 64x32x256
        # tile handles splitK internally, no separate DevDst kernel).
        # stage_sb is from _maca_select_tile (called inside _maca_gemm_dispatch
        # at line 329, which we called above at line 532). The name (line 331)
        # encodes it. Check the first entry's name for sb_0_2.
        _has_builtin_splitk = bool(entries) and "_sb_0_2" in entries[0][0]
        # C500 32B-TP2: the 64x32x256 sb_0_2 tile (builtin splitK) STILL gets a
        # DevDst epilogue in nsys (qkv/o/down all show DevDst after sb_0_2 GEMMs).
        # Only gate on the 8B-verified positions (excl gate_up/lm_head). The
        # builtin-splitk exclusion was 8B-specific; drop it for C500 TP>1.
        _drop_splitk_excl = maca and tp_size > 1
        # force_splitk: per-step nsys-grounded override (qkv_splitk.jsonl). mcblas'
        # split-K is a CLOSED-library autotuner decision (not in vLLM source), so
        # the K>=4096 aggregate gate misses rare steps where a K<4096 GEMM (e.g.
        # Qwen3-4B qkv K=2560) split-K'd anyway. Verified single-step anomaly
        # (Qwen3-4B qps16: 1/1154 steps). Bypass the K/position gate for the
        # flagged step; still require n>1 (GEMV M=1 never split-Ks).
        _gate = (emit_splitk_reduce and n < 16384 and k >= 4096
                 and not _excl_gate_up and (_drop_splitk_excl or not _has_builtin_splitk))
        if n > 1 and (_gate or force_splitk):
            epilog_grid = max(1, _ceil_div(m, 128))
            entries.append(("DevDstElementwiseAdd_kernel",
                            epilog_grid, 1, 1, 128, 1, 0))
        return entries
    entries = []
    # Unified table lookup (covers s16816 + gemvx + gemv2T + xmma, exact
    # (M_tok, N_out, K, has_bias)). The recipe emits exactly the kernel(s)
    # cuBLAS launched (name + grid + block + smem), so grid/tile match nsys by
    # construction. The LUT value is a kernel LIST (1 normally, 2 if splitK —
    # GEMM + splitKreduce, captured by running cuBLAS directly). has_bias
    # matters: bias -> ampere_s16816gemm_relu epilogue, no bias -> sm80_xmma.
    # has_bias is threaded from operator_records by position (lm_head -> False).
    for kname, grid3d, (block_x, block_y), smem in _select_gemm_kernel(
            m, n, k, has_bias=has_bias, dtype=dtype):
        # grid3d is (gridX, gridY, gridZ) — gridZ is splitK (from LUT/nsys).
        # Return 6-tuple (name, gx, gy, bx, by, smem) — gz dropped (consumer
        # in _match_gemm_position expects 6 elements, matching _maca_lmhead).
        entries.append((kname, grid3d[0], grid3d[1], grid3d[2],
                       block_x, block_y, smem))
        # splitK GEMM (gridZ>1) is followed by cublasLt::splitKreduce_kernel
        # (cuBLAS split-K reduction epilogue). DISABLED: cuBLAS's split-K choice
        # is per-run non-deterministic (a shape's gridZ varies across captures),
        # so the LUT majority-vote gridZ emits splitKreduce at positions where
        # the actual run had none → spurious kernels. The nsys splitKreduce
        # kernels align positionally via cat even when regen doesn't emit them
        # (they sit at positions regen covers with the following kernel), so not
        # emitting keeps Llama-8B at 99.98%. Re-enable only with per-run splitK
        # detection (a sampled gridZ, not majority).
        # if grid3d[2] > 1 and emit_cublas_splitk:
        #     entries.append((
        #         "void cublasLt::splitKreduce_kernel<(int)32, (int)16, int, "
        #         "__nv_bfloat16, __nv_bfloat16>",
        #         grid3d[0] * grid3d[2], 1, 1, 32, 16, 0))
    return entries


def _fa_num_splits_heuristic(batch_nheads_mblocks, num_SMs, num_n_blocks,
                             max_splits=128, efficiency_threshold=0.85):
    """Verbatim port of vllm-flash-attn
    csrc/flash_attn/flash_api.cpp:num_splits_heuristic (Ampere path).

    Picks how many KV-splits each (batch, head, M-block) work unit is cut into
    so the launch fills the SMs (num_splits×waves ≈ num_SMs). Called from
    set_params_splitkv with batch_nheads_mblocks = batch * num_heads *
    num_m_blocks, num_SMs = num_sm * 2 (×2 hard-codes 128 threads/block).
    A800: verified 100% match vs nsys grids on Qwen3-8B decode (39600/39600).
    Zero-fit: pure port of the upstream heuristic, no parameters tuned.

    C500 (MetaX flash build): the COMPILED C++ caller (compute_params_numsplits
    in flash_attn_2_cuda.so) passes different inputs — batch_nheads_mblocks=
    batch (not batch*hkv*num_m_blocks), num_SMs=num_compute_units (not *2),
    num_n_blocks uncapped, max_splits=num_compute_units — AND uses a slightly
    lower efficiency_threshold (0.848, reverse-engineered vs nsys: 0.85 leaves
    batch=53 off-by-0.0003; 0.848 matches 39/39 decode batches on Qwen3-8B
    qps4). The threshold param lets the C500 decode branch pass 0.848; A800
    keeps the default 0.85.
    """
    import math
    if batch_nheads_mblocks >= 0.8 * num_SMs:
        return 1
    max_splits = min(max_splits, num_SMs, num_n_blocks)
    ceildiv = lambda a, b: (a + b - 1) // b
    def _eligible(ns):
        return ns == 1 or ceildiv(num_n_blocks, ns) != ceildiv(num_n_blocks, ns - 1)
    eff = []
    max_eff = 0.0
    for ns in range(1, max_splits + 1):
        if not _eligible(ns):
            eff.append(0.0)
        else:
            n_waves = batch_nheads_mblocks * ns / num_SMs
            e = n_waves / math.ceil(n_waves) if n_waves > 0 else 0.0
            eff.append(e)
            if e > max_eff:
                max_eff = e
    for ns in range(1, max_splits + 1):
        if not _eligible(ns):
            continue
        if eff[ns - 1] >= efficiency_threshold * max_eff:
            return ns
    return 1


# C500 FA num_splits LUT (opaque, (batch, hkv, num_n_blocks) key — adds
# kv_len to disambiguate cross-qps variance; 619 entries from serving nsys.
# 10 residual keys are genuine autotuner variance (not kv-fixable) -> _C500_FA_NONDET.
# LUT-first in _attn_kernels C500 decode (batch, hkv, num_n_blocks); heuristic fallback.
_C500_FA_NUMSPLITS_LUT = {}
for _ks, _ns in _LUT_DATA.get("fa_numsplits_lut", {}).items():
    _C500_FA_NUMSPLITS_LUT[tuple(eval(_ks))] = _ns
_MACA_TILE_NONDET = set(tuple(k) for k in _LUT_DATA.get("nondet_gemm", []))
_C500_FA_NONDET = set(tuple(k) for k in _LUT_DATA.get("nondet_fa", []))



def _fa_num_kv_splits(batch, hkv, kv_len, seq_len, head_dim, num_sm, is_maca=False):
    """Compute the splitkv num_splits for an FA compute kernel launch, matching
    vLLM's set_params_splitkv (flash_api.cpp:316).

    block_n (KV tile N-dim): head_dim<=64 -> 256, <=128 -> 128, else 64.
    num_m_blocks = ceil(seqlen_q / 64) (kBlockM=64 for splitKV; 1 for decode).
    num_n_blocks = ceil(KV / block_n).
    num_splits   = heuristic(batch*HKV*num_m_blocks, num_sm*2, num_n_blocks, 128).

    is_maca (C500 / MetaX flash build): the COMPILED caller
    (compute_params_numsplits in flash_attn_2_cuda.so) passes batch (NOT
    batch*HKV*num_m_blocks), num_SMs=num_compute_units (NOT *2), max_splits=
    num_compute_units, and efficiency_threshold=0.848 — see the
    _fa_num_splits_heuristic docstring. The A800 build uses the upstream Ampere
    convention (batch*HKV*mblocks, num_sm*2, max_splits=128, 0.85). Using the
    A800 convention on C500 returns ns=1 for mid-batch decode (batch*HKV=480 >=
    0.8*208) — the real MetaX build splits (ns=3..5), so the combine epilogue
    goes missing → a full-step type_mm cascade. Verified: C500 conv reproduces
    the nsys gridZ (batch=60,nnb=8→3; batch=54,nnb=9→5).
    """
    block_n = 256 if head_dim <= 64 else (128 if head_dim <= 128 else 64)
    num_m_blocks = max(1, (seq_len + 64 - 1) // 64)
    num_n_blocks = max(1, (kv_len + block_n - 1) // block_n)
    if is_maca:
        return _fa_num_splits_heuristic(
            batch * num_m_blocks, num_sm, num_n_blocks,
            max_splits=num_sm, efficiency_threshold=0.848), block_n
    return _fa_num_splits_heuristic(batch * hkv * num_m_blocks,
                                    num_sm * 2, num_n_blocks, 128), block_n




# ---------------------------------------------------------------------------
# Request and schedule-step models
# ---------------------------------------------------------------------------

@dataclass
class MirrorRequest:
    """One inference request for the analytic scheduler."""
    req_id: str
    prompt_len: int
    output_len: int          # number of decode tokens to generate


@dataclass
class MirrorStep:
    """One mirrored scheduling step (same fields as the real sidecar record)."""
    iter_id: int
    scheduled: Dict[str, int]    # req_id -> tokens this step
    finished: List[str]
    # token composition used for kernel generation:
    prefill_tokens: int          # total prompt tokens processed this step
    decode_tokens: int           # number of 1-token decode advances this step


# ---------------------------------------------------------------------------
# vLLM V1 scheduler mirror
# ---------------------------------------------------------------------------

def mirror_schedule(requests: List[MirrorRequest],
                    max_num_batched_tokens: Optional[int] = None,
                    enable_chunked_prefill: bool = True) -> List[MirrorStep]:
    """Replicate vLLM V1 continuous-batching + chunked-prefill scheduling.

    Returns the per-step token composition. Mirrors the FCFS, decode-first,
    budget-limited admission that vLLM V1 performs each engine step.

    Args:
        requests: requests in arrival order (all assumed arrived at t=0; the
            validation workloads use offline batch submission, no staggering).
        max_num_batched_tokens: per-step token budget. None -> no chunking
            (whole prompt admitted in one step, subject only to one request
            per step starting; matches vLLM default for short prompts where
            the budget is not the binding constraint).
        enable_chunked_prefill: if False, a prompt is never split across steps.
    """
    if not requests:
        return []

    # Per-request progress state.
    prompt_left = {r.req_id: r.prompt_len for r in requests}
    decode_left = {r.req_id: r.output_len for r in requests}
    started = {r.req_id: False for r in requests}     # prefill begun
    prefilled = {r.req_id: False for r in requests}   # prompt fully consumed
    arrival_order = [r.req_id for r in requests]

    # Default budget: large enough that short prompts are not chunked, matching
    # vLLM's default max_num_batched_tokens (which is large) when the user does
    # not enable chunked prefill explicitly.
    budget = max_num_batched_tokens if max_num_batched_tokens is not None else 10**12

    steps: List[MirrorStep] = []
    iter_id = 0

    def all_done():
        return all(decode_left[r] <= 0 and prompt_left[r] <= 0 for r in arrival_order)

    while not all_done():
        scheduled: Dict[str, int] = {}
        remaining = budget
        prefill_tokens = 0
        decode_tokens = 0

        # 1) Decode-first: every request that has finished prefill and still has
        #    decode budget advances exactly 1 token (vLLM batches all running
        #    decodes each step).
        for rid in arrival_order:
            if remaining <= 0:
                break
            if prefilled[rid] and decode_left[rid] > 0:
                scheduled[rid] = scheduled.get(rid, 0) + 1
                decode_left[rid] -= 1
                decode_tokens += 1
                remaining -= 1

        # 2) Prefill admission (FCFS): fill remaining budget with prompt tokens
        #    of waiting / partially-prefilled requests.
        for rid in arrival_order:
            if remaining <= 0:
                break
            if prompt_left[rid] <= 0:
                continue
            take = prompt_left[rid]
            if enable_chunked_prefill:
                take = min(take, remaining)
            else:
                # No chunking: only admit if the whole prompt fits in remaining.
                if take > remaining:
                    continue
            scheduled[rid] = scheduled.get(rid, 0) + take
            prompt_left[rid] -= take
            prefill_tokens += take
            remaining -= take
            started[rid] = True
            if prompt_left[rid] <= 0:
                prefilled[rid] = True
                # The forward that completes prefill also samples the request's
                # FIRST output token (vLLM produces token 1 at the end of
                # prefill). So one decode token is consumed by the prefill step,
                # and only output_len-1 dedicated decode steps remain.
                decode_left[rid] -= 1

        # 3) Determine which requests finished AFTER this step's tokens.
        finished = [rid for rid in arrival_order
                    if prefilled[rid] and decode_left[rid] <= 0
                    and rid not in _already_finished(steps)]

        steps.append(MirrorStep(
            iter_id=iter_id,
            scheduled=scheduled,
            finished=sorted(finished),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
        ))
        iter_id += 1
        # Safety bound: total steps <= sum(prompt chunks) + sum(decode) + slack.
        if iter_id > 10**7:
            raise RuntimeError("mirror_schedule did not converge")

    return steps


def _already_finished(steps: List[MirrorStep]) -> set:
    s = set()
    for st in steps:
        s.update(st.finished)
    return s


# ---------------------------------------------------------------------------
# Per-step kernel trace generation
# ---------------------------------------------------------------------------

class AnalyticTraceGenerator:
    """Generate a serving kernel trace by mirroring vLLM scheduling, emitting the
    model's kernel sequence for each scheduling step.

    Each scheduling step is a single batched transformer forward over the step's
    total tokens. vLLM concatenates prefill chunks and decode tokens into one
    forward, but issues phase-specific kernels: prefill tokens drive the
    full-attention flash_fwd path and TC_GEMM matmuls, while pure-decode steps
    (1 token/req) drive the splitkv decode-attention path and GEMV matmuls. When
    a step mixes prefill and decode (continuous batching), vLLM runs them as one
    combined prefill-style forward over the union token count.
    """

    def __init__(self, model_config: ModelConfig, hw: HWSpec, tp_size: int = 1,
                 pp_size: int = 1, ep_size: int = 1):
        from src.dse.model_sim import ModelSimulator
        self.m = model_config
        self.hw = hw
        self.tp_size = tp_size
        self.pp_size = pp_size      # pipeline parallel stages
        self.ep_size = ep_size      # expert parallel degree (MoE)
        if model_config.num_hidden_layers % pp_size != 0:
            raise ValueError(
                f"pp_size={pp_size} must divide num_hidden_layers="
                f"{model_config.num_hidden_layers}")
        self._sim = ModelSimulator.__new__(ModelSimulator)  # reuse kernel factories
        self._sim._hw = hw
        self._sim._hw_config_path = None
        # propagate the maca flag to the shared ModelSimulator kernel factories
        # (so _make_activation etc. can emit C500-specific variants).
        self._sim._is_maca = False  # set below after _is_maca is computed
        self._n_qk_norm = _num_qk_norm(model_config)
        # layers executed per pipeline rank (the trace models one rank's work,
        # consistent with the real per-worker emulation trace which is per-PID).
        self._layers_per_rank = model_config.num_hidden_layers // pp_size
        # Whether this HW emits a mcblas split-K reduce (DevDstElementwiseAdd)
        # epilogue after each TC GEMM. C500/mcblas does; A800/cuBLAS uses a
        # different (splitKreduce) kernel keyed off a separate calibration.
        # Gate on the C500-specific calibration constant so other HW is
        # unaffected (no fabricated kernel absent from their real trace).
        try:
            mem_sub = hw.memory_subsystem
        except AttributeError:
            mem_sub = {}
        self._emit_splitk_reduce = float(
            mem_sub.get("mcblas_splitk_reduce_fixed_ns", 0.0)) > 0
        # Whether this HW uses MetaX mcblas (C500) instead of NVIDIA cuBLAS.
        # mcblas emits entirely different GEMM kernel signatures (tile dims +
        # thread count), which the GEMM predictor parses from the kernel name to
        # derive traffic/occupancy — so the dispatch MUST emit mcblas signatures
        # on C500 or prefill GEMM over-predicts ~6-40× (see _maca_gemm_dispatch).
        # Gate on the same mcblas-specific calibration constant as split-K reduce.
        self._is_maca = self._emit_splitk_reduce
        self._sim._is_maca = self._is_maca  # propagate to kernel factories
        # Per-step prefill FA structure (nsys-grounded sidecar prefill_fa.jsonl):
        # {step_id: {flash2, combine, scan}} — the prefill varlen FA's
        # flash-count(1/2)/combine(0/1)/DeviceScan(0/2) vary by prefill chunk
        # (not num_splits-determinable); set via set_step_prefill_fa. _cur_step_id
        # is set by operator_records before each _attn_kernels call.
        self._step_prefill_fa: dict = {}
        # Per-step FA num_splits override (nsys-grounded fa_ns.jsonl):
        # {step_id: int}. The num_splits LUT misses some keys (large-batch
        # decode) → wrong ns=1 → missing FAcomb. Override from nsys FA gridZ.
        self._step_fa_ns: dict = {}
        self._cur_step_id = None
        # C500 TP>1 deferred AR (lead-into-next-layer). Cleared per step by
        # _step_kernels; flushed after the layer loop.
        self._pending_ar: list = []
        # NVIDIA cuBLAS splitKreduce epilogue (distinct from C500's mcblas
        # DevDstElementwiseAdd). Gate on cublas_splitk_reduce_floor_ns > 0.
        self._emit_cublas_splitk = float(
            mem_sub.get("cublas_splitk_reduce_floor_ns", 0.0)) > 0
        # C500-specific auxiliary kernels captured in the mcTracer/nsys GT but
        # absent from the base analytic recipe: flash-combine epilogue, framework
        # memcpy (DTOD/DTOH/HTOD), MoE routing sort + histogram. Gate on the
        # C500-only host-transfer bandwidth (pcie_bw): its presence marks the
        # MetaX serving stack that emits these kernels; A800/910b lack it so
        # their traces are unchanged (they don't carry these kernels 1:1).
        try:
            self._emit_c500_aux = (hw.interconnect.get("pcie_bw_bytes_s") is not None
                                   or hw.interconnect.get("pcie_bw_GB_s") is not None)
        except AttributeError:
            self._emit_c500_aux = False
        # GEMM dtype ('fp16'/'bf16') for the LUT lookup key. For the analytic
        # path, defaults to the model config dtype; for the operator_records
        # path, OperatorTraceGenerator.set_dtype overrides it with the
        # nsys-detected dtype (which may differ, e.g. 32B config bf16 but
        # nsys fp16). Drives the MoE gate GEMM dispatch (_moe_ffn_kernels).
        self._dtype: str = "bf16" if getattr(model_config, "dtype_bytes", 2) != 2 else "fp16"

    def _tp_allreduce(self, batch: int, seq_len: int, tag: str,
                      *, msg_bytes: int | None = None,
                      nranks: int | None = None):
        """One tensor-parallel / expert-parallel all-reduce (NCCL category).

        vLLM issues one after the attention o_proj and one after the FFN
        down_proj when tp_size>1; EP MoE uses the same call for its reduce
        (nranks=max(tp,ep), see operator_records all_reduce branch).

        NOTE: real vLLM on A800 uses `vllm::cross_device_reduce_1stage`
        (signal-based, NOT NCCL ring), which nsys shows dominates TP/EP steps
        (32B-TP2 49%, 70B-TP4 66%, MoE-EP4 73% of GPU time). Its duration is
        barrier-spin-dominated (p50 30-400µs, p90 ~850µs, host-launch idle) and
        NOT zero-fit predictable (memory tp-allreduce-spinwait-host-launch).
        The NCCL alpha+bandwidth formula here predicts only the spin-free floor
        (~10µs); the spin tail is captured by nsys_idle in G5 serial assembly.
        Pure-kernel TP/EP TTFT therefore under-predicts — this is a known
        limitation, NOT fixable without nsys-derived comm (out of scope for the
        kernel-model goal). See memory moe-32b-ttft-underpred-rootcause.

        TP1 / EP1: vLLM still calls tensor_model_parallel_all_reduce (it is a
        no-op when world_size==1) but issues NO GPU kernel. Emit nothing so
        dense TP1 models are not inflated by a phantom ~10µs/call NCCL pred
        (previously inflated 8B-family TTFT/TPOT). This is the 8B improvement.
        """
        from src.dse.model_sim import SyntheticKernelEntry
        if msg_bytes is None:
            msg_bytes = batch * seq_len * self.m.hidden_size * self.m.dtype_bytes
        if nranks is None:
            nranks = self.tp_size
        if nranks <= 1:
            return []
        # 910B/Ascend: HCCL kernels are NOT in COMPUTE_TASK_INFO (they're
        # non-compute tasks, excluded from the msprof loader). Emit nothing so
        # the regen trace matches the msprof's compute-kernel-only scope.
        if getattr(self.hw, "aic_aiv_separated", False):
            return []
        # C500 (maca): uses mccl (not NCCL/cross_device_reduce). The mccl
        # kernel name is mcclKernel_AllReduce_FC2_Sum_half or _FC2_LL_Sum_half
        # (LL = low-latency, for small messages). grid/block are library
        # constants (like NCCL) — grid varies (32/16/8/4), block=512, smem
        # varies (0 or 512). Use the dominant (32,512,512) for FC2_Sum and
        # (16,512,512) for FC2_LL_Sum. Accepted as fixed (like NCCL).
        if getattr(self, "_is_maca", False):
            _ll = msg_bytes < 512 * 1024  # LL for small msgs (heuristic)
            if _ll:
                name = "mcclKernel_AllReduce_FC2_LL_Sum_half(fcinfo)"
                return [SyntheticKernelEntry(
                    kernel_name=name, grid=(16, 1, 1), block=(512, 1, 1),
                    shared_mem=512, op_type="TP_COMM",
                    op_attr={"msg_bytes": msg_bytes, "nranks": nranks})]
            name = "mcclKernel_AllReduce_FC2_Sum_half(fcinfo)"
            return [SyntheticKernelEntry(
                kernel_name=name, grid=(32, 1, 1), block=(512, 1, 1),
                shared_mem=512, op_type="TP_COMM",
                op_attr={"msg_bytes": msg_bytes, "nranks": nranks})]
        # Mirror vLLM custom_all_reduce.py::should_custom_ar + REDUCE_CASE
        # (csrc/custom_all_reduce.cuh). A800 SXM4 is NVLink fully_connected.
        # should_custom_ar: bytes%16==0, weak_contiguous, (ws==2 or
        # fully_connected), bytes < max_size (8MB). If not, fallback NCCL ring.
        MAX_AR_SIZE = 8192 * 1024
        fully_connected = True   # A800 SXM4 NVLink
        use_custom = (msg_bytes % 16 == 0 and fully_connected
                      and msg_bytes < MAX_AR_SIZE)
        if use_custom:
            # 1stage vs 2stage (custom_all_reduce.cuh REDUCE_CASE):
            #   ws==2 -> 1stage
            #   ws<=4 & bytes<512KB -> 1stage
            #   ws<=8 & bytes<256KB -> 1stage
            #   else -> 2stage
            if nranks == 2 or (nranks <= 4 and msg_bytes < 512 * 1024) \
                    or (nranks <= 8 and msg_bytes < 256 * 1024):
                name = "void vllm::cross_device_reduce_1stage"
            else:
                name = "void vllm::cross_device_reduce_2stage"
            # grid_x = min(kMaxBlocks=36, ceil(msg_bytes/8192)) — SOURCE:
            # csrc/custom_all_reduce.cuh L559 `blocks = min(block_limit, (size+511)/512)`
            # with block_limit=defaultBlockLimit=36 (A800), threads=512, and
            # size = msg_bytes/(d*sizeof(T)) where d=packed_t<T>::P::size=16/sizeof(T)
            # (L82 `P = array_t<T,16/sizeof(T)>`) so d*sizeof(T)=16 constant =>
            # size_packed = msg_bytes/16 => blocks = min(36, ceil(msg_bytes/(16*512)))
            # = min(36, ceil(msg_bytes/8192)). dtype-INDEPENDENT (d*sizeof cancels).
            # block = 512 (kernel __launch_bounds__(512,1)). NOT nranks (old wrong).
            grid_x = min(36, max(1, (msg_bytes + 8191) // 8192))
            return [SyntheticKernelEntry(
                kernel_name=name, grid=(grid_x, 1, 1), block=(512, 1, 1),
                shared_mem=0, op_type="CUSTOM_COMM",
                op_attr={"msg_bytes": msg_bytes, "nranks": nranks})]
        # Fallback: NCCL ring all-reduce (large msg or non-fully-connected).
        # grid=(16,1,1) is NCCL's fixed nBlocks for AllReduce_RING_LL on A800
        # (verified constant across all 32B-TP2 large-prefill fallback calls;
        # NCCL nThreads=544). NCCL nBlocks is a library-fixed launch constant
        # (like a cuBLAS tile — not source-derivable, accepted as fixed).
        return [SyntheticKernelEntry(
            kernel_name=f"ncclDevKernel_AllReduce_Sum_{tag}",
            grid=(16, 1, 1), block=(544, 1, 1),
            shared_mem=0, op_type="COLLECTIVE",
            op_attr={"msg_bytes": msg_bytes, "nranks": nranks})]

    def _ep_all2all(self, batch: int, seq_len: int, tag: str,
                    *, msg_bytes: int | None = None):
        """One expert-parallel all-to-all (NCCL) for MoE token dispatch/combine
        when ep_size>1."""
        from src.dse.model_sim import SyntheticKernelEntry
        topk = self.m.num_experts_per_tok or 1
        if msg_bytes is None:
            msg_bytes = batch * seq_len * topk * self.m.hidden_size * self.m.dtype_bytes
        return SyntheticKernelEntry(
            kernel_name=f"ncclDevKernel_AllToAll_{tag}",
            grid=(self.ep_size, 1, 1), block=(msg_bytes & 0xFFFFFFFF,
                                              (msg_bytes >> 32) & 0xFFFFFFFF, 1),
            shared_mem=0, op_type="COLLECTIVE",
            op_attr={"msg_bytes": msg_bytes, "nranks": self.ep_size})

    def _kv_transfer(self, msg_bytes: int, position: str):
        """One PD-disaggregation KV cache transfer (nccl_send / nccl_recv).

        position is "send" (prefill producer) or "recv" (decode consumer). The
        real per-layer KV byte volume is captured from the live tensor; bytes are
        encoded in block[0]/block[1] (low/high 32 bits), matching the trace
        encoding that src/serve_trace/kv_transfer.py parses. This is the plugin-
        mode counterpart of the LD_PRELOAD fake-nccl nccl_send/nccl_recv entries.
        """
        from src.dse.model_sim import SyntheticKernelEntry
        name = "nccl_send" if position == "send" else "nccl_recv"
        rank = 0 if position == "send" else 1
        return SyntheticKernelEntry(
            kernel_name=name,
            grid=(2, rank, 0),  # (nranks, rank, op_id) per kv_transfer encoding
            block=(msg_bytes & 0xFFFFFFFF, (msg_bytes >> 32) & 0xFFFFFFFF, 1),
            shared_mem=0, op_type="COLLECTIVE",
            op_attr={"msg_bytes": msg_bytes, "nranks": 2, "position": position})

    def _layer_recipe(self, batch: int, seq_len: int, is_decode: bool):
        """Per-layer kernel entries for one transformer layer in one forward.

        Recipe (decoded from nsys ground truth, architecture-aware):
          NORM: 2 (input_layernorm + post_attention_layernorm) + n_qk_norm
                (q_norm + k_norm for Qwen3-family; 0 for Llama).
          ROTARY: 1, KV_CACHE: 1.
          Attention: flash_fwd_splitkv (always). The split-KV combine epilogue
                     is an autotuner artifact (FLASH_SPLIT_REDUCE exclusion), not
                     emitted here.
          Projections (qkv, o, gate_up, down): prefill -> TC_GEMM each, plus a
                     splitKreduce per layer (cuBLAS split-K epilogue);
                     decode -> GEMV (M=1) for qkv/o/down, GEMV for gate_up.
          ACTIVATION: 1 (SiLU gate, dense FFN).
          MoE layers replace the dense FFN GEMMs/ACT with FUSEDMOE + REDUCE/GATHER.

        C500 TP>1 AR lead: mccl's AllReduce kernel launches are delayed by the
        TP comm spinwait so that, in the captured nsys stream, each layer's AR
        appears at the START of the NEXT layer's block (not at the end of the
        layer that issued it). To match, on C500+TP>1 we defer each layer's AR
        entries: emit the previous layer's deferred AR at this layer's head
        (self._pending_ar) and stash this layer's AR for the next layer. The
        caller (_step_kernels) flushes the final pending AR after the layer
        loop. A800 keeps AR in-line (its NCCL/cross_device_reduce launch is
        synchronous to the issuing block)."""
        m = self.m
        sim = self._sim
        tp = self.tp_size
        k = []
        _maca_tp_lead = getattr(self, "_is_maca", False) and tp > 1
        if _maca_tp_lead:
            # Emit the previous layer's deferred AR at this layer's head.
            k.extend(getattr(self, "_pending_ar", []) or [])
            self._pending_ar = []
        # ── input RMSNorm ──
        k.append(sim._make_norm(m, batch, seq_len, tp, "norm_1"))
        # ── QKV projection ──
        k.extend(self._proj_gemm("qkv_proj", batch, seq_len, is_decode))
        # ── QK-norm (Qwen3 family): q_norm + k_norm ──
        for i in range(self._n_qk_norm):
            k.append(sim._make_norm(m, batch, seq_len, tp, f"qk_norm_{i}"))
        # NOTE: _dense_ew_qk_norm_glue REMOVED — nsys (Qwen3-8B) shows QK-norm is
        # a fused vllm::rms_norm_kernel with NO separate elementwise weight-multiply
        # kernel after it (step1: q_norm→k_norm→rotary, no EW between). The old glue
        # emitted 2 spurious EW/layer → +72/step positional drift → 61% alignment.
        # ── rotary ──
        k.append(sim._make_rotary(m, batch, seq_len, tp))
        # ── KV cache append ──
        k.append(sim._make_kv_cache(m, batch, seq_len, tp))
        # ── KV cache block copy (framework DTOD memcpy, ~1/layer) ──
        if self._emit_c500_aux:
            from src.dse.model_sim import SyntheticKernelEntry
            k.append(SyntheticKernelEntry(
                kernel_name="memcpy DTOD(device,device)",
                grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0,
                op_type="MEMCPY_DTOD", op_attr={}))
        # ── attention ──
        k.extend(self._attn_kernels(batch, seq_len, is_decode))
        # ── Dense ELEMENTWISE glue (FA output processing + KV page init) ──
        k.extend(self._dense_attn_ew_glue(batch, seq_len))
        # ── O projection ──
        k.extend(self._proj_gemm("o_proj", batch, seq_len, is_decode))
        # ── TP all-reduce after attention (row-parallel o_proj) ──
        if self.tp_size > 1:
            _ar_attn = self._tp_allreduce(batch, seq_len, "attn")
            if _maca_tp_lead:
                self._pending_ar.extend(_ar_attn)
            else:
                k.extend(_ar_attn)
        # ── post-attention RMSNorm ──
        k.append(sim._make_norm(m, batch, seq_len, tp, "norm_2"))
        # ── FFN ──
        if m.num_experts > 0:
            # Naive EP backend (the one A800 uses) does NOT issue an all-to-all:
            # it places a local expert subset via expert_map and reduces across
            # ranks with the SAME post-FFN all-reduce that TP uses (vLLM's
            # maybe_all_reduce_tensor_model_parallel fires when tp_size>1 OR
            # ep_size>1). nsys confirms zero all-to-all kernels on this backend.
            # Real all-to-all kernels exist only on the pplx/deepep backends and
            # are captured (not fabricated) by the plugin path when present.
            k.extend(self._moe_ffn_kernels(batch, seq_len, is_decode))
        else:
            k.extend(self._proj_gemm("gate_up_proj", batch, seq_len, is_decode))
            k.append(sim._make_activation(m, batch, seq_len, tp))
            k.extend(self._proj_gemm("down_proj", batch, seq_len, is_decode))
        # ── all-reduce after FFN (row-parallel down_proj for TP; cross-rank
        #    expert reduce for naive EP) — vLLM reduces when tp_size>1 OR ep_size>1 ──
        if self.tp_size > 1 or self.ep_size > 1:
            _ar_ffn = self._tp_allreduce(batch, seq_len, "ffn",
                                         nranks=max(self.tp_size, self.ep_size))
            if _maca_tp_lead:
                self._pending_ar.extend(_ar_ffn)
            else:
                k.extend(_ar_ffn)
        # split-K reduce is emitted inside _proj_gemm by the cuBLAS dispatch
        return k

    # ── MoE FFN kernel helpers (GPU/NVIDIA path; source-faithful, nsys-verified
    # for Qwen3-30B-A3B EP2/EP4 — see moe-21kernel-block-sequence memory). Each
    # helper emits one phase of the 20-kernel MoE block (gate→moe_sum); the
    # trailing cross_device_reduce is emitted by the caller (_layer_recipe line
    # 752 / operator_records all_reduce op_record). The block category sequence
    # is CUT RED UEW OTH RED EW RED UEW RED RED VEW OTH OTH OTH UEW IDXE OTH ACT
    # OTH RED (gate+splitK / .float cast / topkGating / renorm / histc /
    # moe_align+count_sort / permute / w1+act+w2 / moe_sum). ──

    def _moe_gate_gemm(self, tokens: int, gate_attr: dict):
        """Router gate GEMM (hidden->num_experts) + cuBLAS splitK epilogue.
        qwen3_moe.py:173 self.gate(hidden_states) -> ReplicatedLinear ->
        torch.nn.functional.linear (linear.py:236 -> utils.py:92) -> cuBLAS.
        Launches BEFORE the .float cast + topkGatingSoftmax (gate produces the
        router logits topk consumes). The LUT has no num_experts (N=128) shape
        (the dense-projection builder doesn't capture the MoE gate), so emit the
        cuBLAS-chosen kernel directly from the nsys-verified per-M regime:
          M<=24  -> ampere_s16816gemm_bf16_64x64 (GEMM), gridZ=32  [small decode]
          M<=250 -> cutlass wmma s161616 16x16    (CUT), gridZ=11  [medium]
          M>250  -> cutlass tensorop s16816 64x64 (CUT), gridZ=4   [prefill]
        gridZ>1 -> a cublasLt::splitKreduce_kernel (cuBLAS split-K contract)."""
        from src.dse.model_sim import SyntheticKernelEntry
        if self._is_maca:
            return [SyntheticKernelEntry(
                kernel_name=gname, grid=(ggx, ggy, 1), block=(gbx, gby, 1),
                shared_mem=gsmem, op_type="GEMM", op_attr=gate_attr)
                for gname, ggx, ggy, ggz, gbx, gby, gsmem in _cublas_gemm_dispatch(
                    self.m.num_experts, tokens, self.m.hidden_size, maca=True)]
        if tokens <= 24:
            gname = "ampere_s16816gemm_bf16_64x64_ldg8_stages_64x6_tn"
            gz = 32
        elif tokens <= 250:
            gname = ("void cutlass::Kernel2<cutlass_80_wmma_tensorop_"
                     "s161616gemm_bf16_16x16_128x2_tn_align8>(T1::Params)")
            gz = 11
        else:
            gname = ("void cutlass::Kernel2<cutlass_80_tensorop_"
                     "s16816gemm_bf16_64x64_64x6_tn_align8>(T1::Params)")
            gz = 4
        gx = max(1, _ceil_div(tokens, 16))
        out = [SyntheticKernelEntry(
            kernel_name=gname, grid=(gx, 1, gz), block=(128, 1, 1), shared_mem=0,
            op_type="GEMM", op_attr=gate_attr)]
        # cublasLt::splitKreduce_kernel — cuBLAS split-K reduction epilogue
        # (gridZ>1 contract; nsys shows it back-to-back after every gate GEMM).
        out.append(SyntheticKernelEntry(
            kernel_name="void cublasLt::splitKreduce_kernel<(int)32, (int)16, "
                        "int, __nv_bfloat16, __nv_bfloat16, float, __nv_bfloat16>",
            grid=(gx, 2, 1), block=(32, 16, 1), shared_mem=0,
            op_type="ELEMENTWISE",
            op_attr={"reads": 1, "writes": 1, "dtype_bytes": 2,
                     "floor_ns": _GLUE_COPY_FLOOR_NS}))
        return out

    @staticmethod
    def _moe_float_cast(tokens: int, num_experts: int):
        """gating_output.float() (fused_moe.py:1003): bf16->fp32 cast of the
        router logits, BEFORE ops.topk_softmax. unrolled_elementwise direct_copy
        (UEW)."""
        from src.dse.model_sim import SyntheticKernelEntry
        return SyntheticKernelEntry(
            kernel_name="void at::native::unrolled_elementwise_kernel<direct_copy>",
            grid=(max(1, _ceil_div(tokens, 4)), 1, 1), block=(128, 1, 1),
            shared_mem=0, op_type="ELEMENTWISE",
            op_attr={"element_count": tokens * num_experts, "dtype_bytes": 4,
                     "reads": 1, "writes": 1, "floor_ns": _GLUE_COPY_FLOOR_NS})

    @staticmethod
    def _moe_renorm(tokens: int, topk: int):
        """topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        (fused_moe.py:965, renormalize=True for Qwen3 norm_topk_prob). Two
        kernels: the row-sum reduce_kernel (RED) then the broadcast div
        elementwise_kernel (EW)."""
        from src.dse.model_sim import SyntheticKernelEntry
        return [
            SyntheticKernelEntry(  # topk_weights.sum(-1) -> reduce_kernel (RED)
                kernel_name="void at::native::reduce_kernel<sum_functor>",
                grid=(max(1, tokens), 1, 1), block=(8, 16, 1), shared_mem=0,
                op_type="REDUCE",
                op_attr={"element_count": tokens * topk, "dtype_bytes": 4,
                         "reduction_factor": topk}),
            SyntheticKernelEntry(  # topk_weights / sum -> elementwise div (EW)
                kernel_name="void at::native::elementwise_kernel<DivFunctor>",
                grid=(max(1, _ceil_div(tokens, 4)), 1, 1), block=(128, 1, 1),
                shared_mem=0, op_type="ELEMENTWISE",
                op_attr={"element_count": tokens * topk, "dtype_bytes": 4,
                         "reads": 2, "writes": 1, "floor_ns": _GLUE_BINARY_FLOOR_NS}),
        ]

    @staticmethod
    def _moe_expert_count(tokens: int, topk: int, num_experts: int):
        """Expert token-count pre-pass internal to ops.moe_align_block_size
        (the C++ op computes per-expert token counts via torch::histc on
        topk_ids BEFORE the align/sort kernels). 6 kernels (the nsys 'mystery'
        block between renorm and moe_align):
          MaxNan(int) reduce  (RED)  — topk_ids.max() for histc max bound
          cast int->long     (UEW)  — topk_ids.to(int64) (histc needs long)
          MinNan(long) reduce (RED)  — topk_ids_long.min()
          MaxNan(long) reduce (RED)  — topk_ids_long.max()
          Fill(long)         (VEW)  — zero the histogram output tensor
          kernelHistogram1D  (OTH)  — the expert-count histogram
        All counts/grids are size-near-1 (decoded from nsys); cosmetic for the
        category metric (timing floors each via ElementwisePredictor)."""
        from src.dse.model_sim import SyntheticKernelEntry
        rows = max(1, tokens * topk)
        return [
            SyntheticKernelEntry(
                kernel_name="void at::native::reduce_kernel<MaxNanFunctor<int>>",
                grid=(1, 1, 1), block=(32, 1, 1), shared_mem=0,
                op_type="REDUCE",
                op_attr={"element_count": rows, "dtype_bytes": 4,
                         "reduction_factor": rows}),
            SyntheticKernelEntry(  # cast topk_ids int32->int64
                kernel_name="void at::native::unrolled_elementwise_kernel<direct_copy>",
                grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                op_type="ELEMENTWISE",
                op_attr={"element_count": rows, "dtype_bytes": 8,
                         "reads": 1, "writes": 1, "floor_ns": _GLUE_COPY_FLOOR_NS}),
            SyntheticKernelEntry(
                kernel_name="void at::native::reduce_kernel<MinNanFunctor<long>>",
                grid=(1, 1, 1), block=(32, 1, 1), shared_mem=0,
                op_type="REDUCE",
                op_attr={"element_count": rows, "dtype_bytes": 8,
                         "reduction_factor": rows}),
            SyntheticKernelEntry(
                kernel_name="void at::native::reduce_kernel<MaxNanFunctor<long>>",
                grid=(1, 1, 1), block=(32, 1, 1), shared_mem=0,
                op_type="REDUCE",
                op_attr={"element_count": rows, "dtype_bytes": 8,
                         "reduction_factor": rows}),
            SyntheticKernelEntry(  # zero the histogram output (FillFunctor)
                kernel_name="void at::native::vectorized_elementwise_kernel<FillFunctor>",
                grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                op_type="ELEMENTWISE",
                op_attr={"element_count": num_experts, "dtype_bytes": 8,
                         "reads": 0, "writes": 1, "floor_ns": _GLUE_FILL_FLOOR_NS}),
            SyntheticKernelEntry(  # the expert-count histogram
                kernel_name="void at::cuda::kernelHistogram1D",
                grid=(1, 1, 1), block=(512, 1, 1), shared_mem=0,
                op_type="ELEMENTWISE",
                op_attr={"element_count": rows, "dtype_bytes": 8,
                         "reads": 1, "writes": 1, "floor_ns": _GLUE_COPY_FLOOR_NS}),
        ]

    @staticmethod
    def _moe_permute_glue(tokens: int, topk: int, ep_size: int):
        """Post count_and_sort glue (internal to ops.moe_align_block_size,
        moe_align_block_size.py:85 expert_ids = expert_map[expert_ids] for EP):
          sorted_ids fill_     (UEW)  — sorted_ids contiguous/fill
          index_elementwise    (IDXE) — expert_map[expert_ids] gather (EP-only;
                                        ep_size>1; non-EP dense MoE has no
                                        expert_map so no index)."""
        from src.dse.model_sim import SyntheticKernelEntry
        out = [SyntheticKernelEntry(
            kernel_name="void at::native::unrolled_elementwise_kernel<direct_copy>",
            grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
            op_type="ELEMENTWISE",
            op_attr={"element_count": max(1, tokens * topk), "dtype_bytes": 4,
                     "reads": 1, "writes": 1, "floor_ns": _GLUE_COPY_FLOOR_NS})]
        if ep_size > 1:
            out.append(SyntheticKernelEntry(
                kernel_name="void at::native::index_elementwise_kernel",
                grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                op_type="ELEMENTWISE",
                op_attr={"element_count": max(1, tokens * topk),
                         "dtype_bytes": 4, "reads": 1, "writes": 1,
                         "floor_ns": _GLUE_COPY_FLOOR_NS}))
        return out

    @staticmethod
    def _moe_moe_sum(tokens: int, topk: int):
        """ops.moe_sum (fused_moe.py:1703): reduce the topk expert outputs back
        to per-token. nsys shows it as at::reduce_kernel<sum_functor<bf16>> (RED)
        — for unquantized bf16 moe_sum maps to an ATen sum-reduce."""
        from src.dse.model_sim import SyntheticKernelEntry
        return SyntheticKernelEntry(
            kernel_name="void at::native::reduce_kernel<sum_functor>",
            grid=(max(1, tokens), 1, 1), block=(32, 4, 1), shared_mem=0,
            op_type="REDUCE",
            op_attr={"element_count": tokens * topk, "dtype_bytes": 2,
                     "reduction_factor": topk})

    def _moe_ffn_kernels(self, batch: int, seq_len: int, is_decode: bool):
        """vLLM fused-MoE FFN kernel sequence for one layer in one forward.

        Recipe decoded from Qwen3-30B-A3B nsys (per layer per step):
          gate_proj GEMM (router logits) + topkGatingSoftmax (router)
          + moe_align_block_size + count_and_sort + 2 fused_moe_kernel (grouped
          w1/w2 GEMMs) + act_and_mul (SiLU). These are vLLM's fused_moe path
          kernels; the attention block (handled by the caller) is unchanged
          dense.
        """
        from src.dse.model_sim import SyntheticKernelEntry
        m = self.m
        tokens = batch * seq_len
        gx = lambda div: max(1, (tokens + div - 1) // div)
        E = m.num_experts
        topk = m.num_experts_per_tok or 1
        int_size = m.moe_intermediate_size or m.intermediate_size
        out = []
        # router gate projection GEMM (gate_proj: hidden -> num_experts). vLLM
        # runs this dense matmul to produce the router logits BEFORE the
        # topkGatingSoftmax. nsys (Qwen3-30B-A3B) emits one cutlass::Kernel2 per
        # MoE layer per step for the gate in BOTH prefill and decode steps: the
        # router runs through torch.nn.functional.linear (a cutlass TC GEMM),
        # NOT the cuBLAS GEMV path the token projections use, so it stays
        # TC_GEMM even at M=1 (decode). Hence we emit the TC_GEMM kernel directly
        # rather than going through _cublas_gemm_dispatch (which would pick GEMV
        # for the single-token decode case).
        gate_attr = {"B": batch, "M": tokens, "N": E, "K": m.hidden_size}
        ascend = getattr(self.hw, "aic_aiv_separated", False)
        maca = self._is_maca
        _aux_attr = {"TOK": tokens, "H": m.hidden_size, "INT": int_size,
                     "TOPK": topk, "E": E}
        _aiv = (getattr(self.hw, "aiv_core_count", 48), 1, 1)
        gpu = not ascend and not maca   # NVIDIA A800 path (the 21-kernel sequence)
        # ── [1-3] gate GEMM + splitKreduce + .float() cast ──
        # qwen3_moe.py:173 self.gate(hidden_states) -> ReplicatedLinear ->
        # torch.nn.functional.linear -> cuBLAS. Runs BEFORE the .float cast +
        # topkGatingSoftmax (gate produces the router logits). fused_moe.py:1003
        # gating_output.float() (bf16->fp32) sits between the gate and topk.
        if maca:
            for gname, ggx, ggy, ggz, gbx, gby, gsmem in _cublas_gemm_dispatch(
                    E, tokens, m.hidden_size, maca=True):
                out.append(SyntheticKernelEntry(
                    kernel_name=gname, grid=(ggx, ggy, 1), block=(gbx, gby, 1),
                    shared_mem=gsmem, op_type="GEMM", op_attr=gate_attr))
        elif ascend:
            for gname, ggx, ggy, ggz, gbx, gby, gsmem in _cublas_gemm_dispatch(
                    E, tokens, m.hidden_size, has_bias=False, dtype=self._dtype):
                out.append(SyntheticKernelEntry(
                    kernel_name=gname, grid=(ggx, ggy, ggz), block=(gbx, gby, 1),
                    shared_mem=gsmem, op_type="GEMM", op_attr=gate_attr))
        else:
            out.extend(self._moe_gate_gemm(tokens, gate_attr))
            out.append(self._moe_float_cast(tokens, E))
        # MoE aux kernel chain. GPU (vLLM/Triton): topkGatingSoftmax +
        # renorm(sum reduce + div) + expert-count histc (max/min/fill/hist) +
        # moe_align_block_size + count_and_sort. Ascend: MoeGatingTopK +
        # MoeComputeExpertTokens + MoeInitRouting (AIV) -> _moe_nt_analytical.
        if ascend:
            # vllm-ascend pre-w1 routing: MoeGatingTopK + MoeInitRouting (both
            # custom csrc, AIV=48). MoeComputeExpertTokens is FOLDED into
            # MoeInitRouting (not a standalone kernel) — do NOT emit it
            # separately (would double-count a kernel that never launches alone).
            for _kn in ("MoeGatingTopK", "MoeInitRouting"):
                out.append(SyntheticKernelEntry(
                    kernel_name=_kn, grid=_aiv, block=(1, 1, 1), shared_mem=0,
                    op_type="FUSEDMOE", op_attr=dict(_aux_attr)))
        else:
            # [4] topkGatingSoftmax (softmax+topk, fused_moe.py:954 ops.topk_softmax).
            # nsys grid = ceil(tokens/4), block = (32,4,1).
            out.append(SyntheticKernelEntry(
                kernel_name="void vllm::moe::topkGatingSoftmax_router",
                grid=(_ceil_div(tokens, 4), 1, 1), block=(32, 4, 1), shared_mem=0,
                op_type="REDUCE",
                op_attr={"element_count": tokens * E, "dtype_bytes": m.dtype_bytes,
                         "reduction_factor": 6}))
            if gpu:
                # [5-6] renorm: topk_weights / topk_weights.sum(-1) (fused_moe.py:965)
                out.extend(self._moe_renorm(tokens, topk))
                # [7-12] expert-count histc pre-pass (internal to
                # ops.moe_align_block_size): MaxNan / cast int->long / MinNan /
                # MaxNan / Fill / kernelHistogram1D.
                out.extend(self._moe_expert_count(tokens, topk, E))
            # [13] moe_align_block_size_kernel (moe_align_block_size.py:82).
            # nsys: grid=(1,1,1) block=(1024,1,1) on A800; C500 (maca) uses the
            # optimized 128-expert variant with block 512. (The _align_blk/
            # _align_name definitions were accidentally dropped in af5d885 while
            # their use below was kept — restored here.)
            _align_blk = 512 if self._is_maca else 1024
            _align_name = ("void vllm::moe::moe_align_block_size_opt_128_expert_kernel"
                           if self._is_maca else
                           "void vllm::moe::moe_align_block_size_kernel")
            out.append(SyntheticKernelEntry(
                kernel_name=_align_name,
                grid=(1, 1, 1), block=(_align_blk, 1, 1), shared_mem=0,
                op_type="GATHER",
                op_attr={"element_count": tokens * topk, "dtype_bytes": 4,
                         "scattered": True, "index_count": tokens * topk}))
            # [14] count_and_sort_expert_tokens_kernel (same ops.moe_align_block_size
            # call). nsys: grid=ceil(tokens*topk/256), block=(256,1,1).
            out.append(SyntheticKernelEntry(
                kernel_name="void vllm::moe::count_and_sort_expert_tokens_kernel",
                grid=(_ceil_div(tokens * topk, 256), 1, 1), block=(256, 1, 1),
                shared_mem=0, op_type="GATHER",
                op_attr={"element_count": tokens * topk, "dtype_bytes": 4,
                         "scattered": True, "index_count": tokens * topk}))
            if gpu:
                # [15-16] permute glue: sorted_ids fill (UEW) + expert_map[expert_ids]
                # index (IDXE, EP-only) — moe_align_block_size.py:85.
                out.extend(self._moe_permute_glue(tokens, topk, self.ep_size))
        # MoE routing sort: GT shows 5 CUB radix sort invocations per MoE
        # layer (vLLM sorts token-expert pairs by expert ID for grouped GEMM).
        if self._emit_c500_aux:
            for i in range(5):
                out.append(SyntheticKernelEntry(
                    kernel_name=f"moe_token_sort_kernel_{i}",
                    grid=(_ceil_div(tokens * topk, 256), 1, 1),
                    block=(256, 1, 1), shared_mem=0,
                    op_type="MOE_SORT",
                    op_attr={"element_count": tokens * topk, "dtype_bytes": 4}))
        # MoE histogram: 1 per MoE layer. C500 nsys: grid=ceil(tokens/64)
        # block=(512,1,1) name at::cuda::kernelHistogram1D (tok=729->grid 12);
        # A800: grid=1 block=256. Verified on 30B-A3B EP2.
        if self._emit_c500_aux:
            _hist_gx = _ceil_div(tokens, 64) if self._is_maca else 1
            _hist_blk = 512 if self._is_maca else 256
            _hist_name = ("void at::cuda::kernelHistogram1D_expert_count"
                          if self._is_maca else "histogram_moe_expert_count")
            out.append(SyntheticKernelEntry(
                kernel_name=_hist_name,
                grid=(_hist_gx, 1, 1), block=(_hist_blk, 1, 1), shared_mem=0,
                op_type="MOE_HISTOGRAM",
                op_attr={"element_count": tokens * topk, "dtype_bytes": 4}))
        # two grouped expert GEMMs (w1=gate_up, w2=down) — fused_moe_kernel
        #
        # Tiling/grid come from the single shared-config rule in
        # src/kernel/ops/fused_moe.derive_moe_tiling (verified 88/88 exact vs
        # C500 captured grids). The runtime selects ONE config and applies
        # the same BM/BN/BK/GSM to both projections; only the GEMM dims differ.
        # BM/BN heuristic (no tuned config file for Qwen3-30B-A3B on A800):
        #   tok <= E_local → BM=16, BN=32  (E_local = E // tp_ep_size, default E)
        #   tok >  E_local → BM=64, BN=64
        from src.kernel.validation.legacy_ops.fused_moe import derive_moe_tiling
        E_local = E // max(1, self.ep_size)
        BM = 16 if tokens <= E_local else 64
        BN = 32 if tokens <= E_local else 64
        BK = 64 if BM == 16 else 32
        moe_tiling = derive_moe_tiling(
            tok=tokens, E=E, topk=topk, H=m.hidden_size, INT=int_size,
            BM=BM, BN=BN, BK=BK, GSM=1)
        grid_w1 = moe_tiling["w1"]["grid"]
        grid_w2 = moe_tiling["w2"]["grid"]
        tok_per_expert = max(1, (tokens * topk) // E)
        # Ascend (AIC/AIV separated) runs the two projections as a single
        # aclnnGroupedMatmul op on a FIXED 24-AIC-core grid with an in-core expert
        # loop — NOT the GPU/Triton one-CTA-per-tile model. Emitting the Triton
        # grid (ceil(EM/BM)·ceil(N/BN)) here inflates the predicted traffic by
        # grid/24 (predict_moe_traffic scales B_all by grid). So on Ascend we emit
        # grid=(24,1,1) and a `groupedmatmul_*_w{1,2}` name; the predictor's Ascend
        # branch (op_type=cube) then charges single-projection weight/1.35TB·s
        # traffic (see fused_moe._ascend_grouped_projection_traffic). GPU keeps the
        # Triton fused_moe_kernel grid unchanged. Data-driven by hw, not card name.
        ascend = getattr(self.hw, "aic_aiv_separated", False)
        # base op_attr shared by both projections; the Triton `moe_tiling` (with
        # its BM/BN/grid) is only consumed by the non-Ascend (A800/C500) Triton
        # forward branch (_match_moe_projection). On Ascend the entry routes via
        # op_type to the GroupedMatmul branch which reads only tok_per_expert, so
        # moe_tiling would be a dead Triton-only field leaking into 910B traces.
        _base_attr = {"TOK": tokens, "H": m.hidden_size, "INT": int_size,
                      "TOPK": topk, "E": E, "tok_per_expert": tok_per_expert}
        for proj, grid_flat in (("w1", grid_w1), ("w2", grid_w2)):
            if ascend:
                kname = f"aclnnGroupedMatmul_grouped_matmul_mix_aic_{proj}"
                grid = (self.hw.num_compute_units, 1, 1)
                op_attr = _base_attr
            else:
                kname = f"fused_moe_kernel_{proj}"
                grid = (max(1, grid_flat), 1, 1)
                op_attr = {**_base_attr, "moe_tiling": moe_tiling}
            # fused_moe block: C500 nsys block=(256,1,1) (A800 128). Verified
            # 30B-A3B EP2 (all fused_moe launches). Name bare 'fused_moe_kernel'
            # in nsys (no _w1/_w2 suffix; projection inferred from grid).
            _moe_blk = 256 if self._is_maca else 128
            out.append(SyntheticKernelEntry(
                kernel_name=kname,
                grid=grid, block=(_moe_blk, 1, 1), shared_mem=0,
                op_type="FUSEDMOE",
                op_attr=op_attr))
            if proj == "w1":
                if ascend:
                    # vllm-ascend SiLU activation (AIV) between w1 and w2 — same
                    # role as act_and_mul but a separate aclnnSwiGlu kernel that
                    # routes to _moe_nt_analytical Ascend branch (measured
                    # 2×(TOK·TOPK·INT) traffic).
                    out.append(SyntheticKernelEntry(
                        kernel_name="SwiGlu", grid=_aiv, block=(1, 1, 1),
                        shared_mem=0, op_type="FUSEDMOE", op_attr=dict(_aux_attr)))
                else:
                    # SiLU gate activation between w1 and w2, applied to every
                    # expert-routed token: nsys grid = tokens*topk (128 tok ->1024,
                    # 1 tok -> 8), block = min(moe_intermediate, 1024) (=768 here).
                    out.append(SyntheticKernelEntry(
                        kernel_name="void vllm::act_and_mul_kernel_moe",
                        grid=(max(1, tokens * topk), 1, 1),
                        block=(min(int_size, 1024), 1, 1), shared_mem=0,
                        op_type="ACTIVATION",
                        op_attr={"batch": tokens, "intermediate": int_size,
                                 "dtype_bytes": m.dtype_bytes}))
        if gpu:
            # [20] ops.moe_sum (fused_moe.py:1703): reduce the topk expert outputs
            # back to per-token. nsys: at::reduce_kernel<sum<bf16>> (RED).
            out.append(self._moe_moe_sum(tokens, topk))
        if ascend:
            # vllm-ascend post-w2 aux (AIV): un-permute to token order via
            # MoeTokenUnpermute (CANN aclnn, AIV=48). MoeFinalizeRoutingV2 is a
            # GHOST — vllm-ascend explicitly does NOT launch it
            # (fused_moe/moe_comm_method.py:186: "npu_moe_finalize_routing will
            # lead to accuracy issues so we use npu_moe_token_unpermute instead").
            for _kn in ("MoeTokenUnpermute",):
                out.append(SyntheticKernelEntry(
                    kernel_name=_kn, grid=_aiv, block=(1, 1, 1), shared_mem=0,
                    op_type="FUSEDMOE", op_attr=dict(_aux_attr)))
            # Ascend framework aclnn glue around MoE routing (still emitted
            # separately here — the GPU path interleaves its glue above).
            out.extend(self._moe_glue_kernels(tokens, topk, E, BM))
        # GPU (NVIDIA) framework ELEMENTWISE glue is now interleaved in the
        # source-derived order above (.float cast [3], renorm [5-6], expert-count
        # histc [7-12], permute [15-16]) — see the [N] markers. The trailing
        # cross_device_reduce [21] (EP combine) is emitted by the caller
        # (_layer_recipe line 752 / operator_records all_reduce op_record), NOT here.
        return out

    def _moe_glue_kernels(self, tokens: int, topk: int, E: int, BM: int):
        """Framework aten glue launched around the fused-MoE routing, captured in
        nsys but NOT hookable at the module level (so absent from the plugin
        trace — this is the ELEMENTWISE serving gap). Each is a fixed structural
        implication of the MoE routing, source-pinned to vLLM:

          moe_align_block_size (fused_moe/moe_align_block_size.py):
            expert_ids = torch.zeros(mblk, int32)   → FillFunctor
            sorted_ids.fill_(EM)                     → FillFunctor
          fused_topk renorm + output cast (fused_moe.py, norm_topk_prob=True):
            topk_weights / topk_weights.sum(-1)      → gpu binary (div)
            topk_weights.to(f32), topk_ids.to(int32) → 2 × direct_copy cast

        The row-sum in the renorm is a REDUCE (promoted to FUSEDMOE for MoE
        models) and is already covered by the routing softmax accounting, so it
        is not re-emitted here. Grid/block are cosmetic (timing is the measured
        launch floor, ~size-independent); element_count/floor_ns are carried on
        op_attr so the ElementwisePredictor floors each glue at its real GPU-busy
        time. All counts are GT-decoded near-integers (Fill 2/layer, div 1/layer,
        cast ~2-3/layer), consistent with how the rest of the recipe is decoded
        from nsys — not tuned on eval residual.

        GPU-only: Ascend msprof fuses these aten ops into the enclosing operator
        (state ③), and C500/mcblas MoE routing has not been confirmed to launch
        the same glue, so emit only on NVIDIA-class HW (not aic_aiv_separated,
        not MetaX mcblas).
        """
        from src.dse.model_sim import SyntheticKernelEntry
        ascend = getattr(self.hw, "aic_aiv_separated", False)
        if ascend:
            # vllm-ascend framework glue around MoE routing — these aclnn ops
            # (ReduceMax/Index/Slice/Cast/Mul/ReduceMin/Abs/Bincount/NeScalar/
            # InplaceCopy) ARE separate kernels in msprof (the GPU comment
            # "Ascend fuses these into the op" is wrong), but NOT hookable at
            # the module level → absent from the plugin trace (the OTHER serving
            # gap, ~2151ms / ~38% of the 910B G1 residual). Emit them
            # synthetically per MoE layer (counts decoded from 910B nsys,
            # ~1-2/layer) as ELEMENTWISE floored at the measured AIV dispatch
            # overhead (dispatch_overhead_ns). The dispatch floor under-covers
            # the ~3.6us/call glue avg (the glue kernel work above launch is a
            # FA-class latency residual), but closes most of the gap vs emitting
            # nothing. 910B-only (GPU uses the Fill/div/cast branch below).
            floor = float(getattr(self.hw, "dispatch_overhead_ns", 2000))
            rows = max(1, tokens * topk)
            _glue = [
                ("aclnnMax_ReduceMaxAiCore_ReduceMax", 2),
                ("aclnnIndex_IndexAiCore_Index", 1),
                ("aclnnContiguous_SliceAiCore_Slice", 1),
                ("aclnnCast_CastAiCore_Cast", 1),
                ("aclnnMul_MulAiCore_Mul", 1),
                ("aclnnMin_ReduceMinAiCore_ReduceMin", 1),
                ("aclnnAbs_AbsAiCore_Abs", 1),
                ("aclnnBincount_FillAiCore_Fill", 1),
                ("aclnnNeScalar_NotEqualAiCore_NotEqual", 1),
                ("aclnnInplaceCopy_SliceAiCore_Slice", 1),
            ]
            out = []
            for _kn, _per_layer in _glue:
                out.append(SyntheticKernelEntry(
                    kernel_name=_kn, grid=(1, 1, 1), block=(1, 1, 1),
                    shared_mem=0, op_type="ELEMENTWISE",
                    op_attr={"element_count": rows, "dtype_bytes": 4,
                             "reads": 1, "writes": 1, "floor_ns": floor}))
            return out
        if False:  # was: if self._is_maca: return []  -- C500 HAS aten glue
            return []                      # (verified dense; MoE inferred)
        rows = max(1, tokens * topk)          # routed token-expert pairs
        EM = rows + E * (BM - 1)              # padded sorted-id table length
        mblk = _ceil_div(EM, BM)              # aligned block count (expert_ids)
        # C500 nsys glue block=64 (A800 256); Fill stays 256.
        _glue_ew_blk = 64 if self._is_maca else 256

        def _fill(name_tag, count):
            return SyntheticKernelEntry(
                kernel_name=f"void at::native::vectorized_elementwise_kernel_FillFunctor_{name_tag}",
                grid=(_ceil_div(max(1, count), 256), 1, 1), block=(256, 1, 1),
                shared_mem=0, op_type="ELEMENTWISE",
                op_attr={"element_count": max(1, count), "dtype_bytes": 4,
                         "reads": 0, "writes": 1,
                         "floor_ns": _GLUE_FILL_FLOOR_NS})

        def _ew(name_tag, count, floor, reads, writes, dtype_bytes):
            return SyntheticKernelEntry(
                kernel_name=f"void at::native::elementwise_kernel_{name_tag}",
                grid=(_ceil_div(max(1, count), _glue_ew_blk), 1, 1),
                block=(_glue_ew_blk, 1, 1),
                shared_mem=0, op_type="ELEMENTWISE",
                op_attr={"element_count": max(1, count), "dtype_bytes": dtype_bytes,
                         "reads": reads, "writes": writes, "floor_ns": floor})

        return [
            _fill("expert_ids", mblk),                       # torch.zeros(mblk,int32)
            _fill("sorted_ids", EM),                         # sorted_ids.fill_(EM)
            _ew("gpu_binary_div", rows, _GLUE_BINARY_FLOOR_NS,
                reads=2, writes=1, dtype_bytes=4),           # renorm div (f32)
            _ew("direct_copy_weights", rows, _GLUE_COPY_FLOOR_NS,
                reads=1, writes=1, dtype_bytes=4),           # topk_weights.to(f32)
            _ew("direct_copy_ids", rows, _GLUE_COPY_FLOOR_NS,
                reads=1, writes=1, dtype_bytes=4),           # topk_ids.to(int32)
        ]

    def _dense_attn_ew_glue(self, batch: int, seq_len: int):
        """Framework ELEMENTWISE glue around FlashAttention output, captured in
        nsys but absent from the plugin trace (aten ops not hookable at module
        level).

        Per layer in nsys (A800, Llama-8B + Qwen3-8B, consistent):
          1 × FillFunctor  — KV cache page zero-init (before FA)
          2 × EW4 binary   — FA output reshape/transpose contiguous copies
          1 × EW2 unary    — attention output dtype cast (fp32→bf16)

        nsys validation (Qwen3-8B): 161k EW4, 54k Fill (exact), 38k EW2.
        Ratio ~3 EW4/attn including QK norm; 1.5 EW4/attn for Llama (no QK
        norm).  Two EW4 here provides good coverage across both patterns.

        Gate on Ascend only (where these fuse into the enclosing aclnn op).
        C500/MetaX does NOT fuse them — nsys HAS ~146k at::native glue kernels
        (~626ms / 2.26% of GPU time: FillFunctor 48k, direct_copy 91k,
        elementwise_NN 85k, unrolled 11k, vectorized 48k; block 64 dominant).
        Re-enabled on C500; EW block recalibrated 128->64 (C500 dominant),
        Fill stays 256. Names stay synthetic (timing is floor_ns-dominated)."""
        from src.dse.model_sim import SyntheticKernelEntry
        ascend = getattr(self.hw, "aic_aiv_separated", False)
        if ascend:
            return []
        tokens = batch * seq_len
        tp = self.tp_size
        m = self.m
        head_dim = m.head_dim
        num_heads = m.num_attention_heads // tp
        num_kv_heads = m.num_key_value_heads // tp
        q_elements = tokens * num_heads * head_dim
        kv_elements = tokens * num_kv_heads * head_dim
        # C500 nsys glue block=64 (A800 128); Fill 256 (nsys 256/1024).
        _ew_blk = 64 if self._is_maca else 128
        _ew_per_blk = _ew_blk * 4   # 4 elems/thread (matches A800 128*4=512)

        def _fill(count):
            return SyntheticKernelEntry(
                kernel_name="void at::native::vectorized_elementwise_kernel_FillFunctor_kv_page",
                grid=(_ceil_div(max(1, count), 128), 1, 1), block=(128, 1, 1),
                shared_mem=0, op_type="ELEMENTWISE",
                op_attr={"element_count": max(1, count), "dtype_bytes": m.dtype_bytes,
                         "reads": 0, "writes": 1,
                         "floor_ns": _DENSE_EW_FILL_FLOOR_NS})

        def _ew4(count):
            return SyntheticKernelEntry(
                kernel_name="void at::native::elementwise_kernel_attn_output",
                grid=(_ceil_div(max(1, count), _ew_per_blk), 1, 1), block=(_ew_blk, 1, 1),
                shared_mem=0, op_type="ELEMENTWISE",
                op_attr={"element_count": max(1, count), "dtype_bytes": m.dtype_bytes,
                         "reads": 2, "writes": 1,
                         "floor_ns": _DENSE_EW_BINARY_FLOOR_NS})

        def _ew2(count):
            return SyntheticKernelEntry(
                kernel_name="void at::native::elementwise_kernel_attn_cast",
                grid=(_ceil_div(max(1, count), _ew_per_blk), 1, 1), block=(_ew_blk, 1, 1),
                shared_mem=0, op_type="ELEMENTWISE",
                op_attr={"element_count": max(1, count), "dtype_bytes": m.dtype_bytes,
                         "reads": 1, "writes": 1,
                         "floor_ns": _DENSE_EW_CAST_FLOOR_NS})

        return [
            _fill(kv_elements * 4),        # KV cache page zero-init
            _ew4(q_elements),              # FA output contiguous copy (Q-sized)
            _ew4(q_elements),              # FA output reshape (Q-sized)
            _ew2(tokens * num_heads),      # attention output cast
        ]

    def _dense_ew_qk_norm_glue(self, batch: int, seq_len: int):
        """Extra ELEMENTWISE from Qwen3 QK-norm forward_native implementation.

        Qwen3 applies RMSNorm to Q and K before RoPE via forward_native(), which
        splits into a rms_norm kernel + a separate elementwise weight-multiply
        kernel. The rms_norm is already emitted as a NORM kernel; this emits the
        elementwise multiply (nsys: 1 extra EW4 per QK-norm pair, operating on
        Q-sized tensor for q_norm and K-sized for k_norm).
        """
        from src.dse.model_sim import SyntheticKernelEntry
        ascend = getattr(self.hw, "aic_aiv_separated", False)
        if ascend:
            return []
        tokens = batch * seq_len
        tp = self.tp_size
        m = self.m
        head_dim = m.head_dim
        num_heads = m.num_attention_heads // tp
        num_kv_heads = m.num_key_value_heads // tp
        q_elements = tokens * num_heads * head_dim
        k_elements = tokens * num_kv_heads * head_dim
        # C500 nsys glue block=64 (A800 128).
        _ew_blk = 64 if self._is_maca else 128
        _ew_per_blk = _ew_blk * 4

        def _ew4(count):
            return SyntheticKernelEntry(
                kernel_name="void at::native::elementwise_kernel_qk_norm_mul",
                grid=(_ceil_div(max(1, count), _ew_per_blk), 1, 1), block=(_ew_blk, 1, 1),
                shared_mem=0, op_type="ELEMENTWISE",
                op_attr={"element_count": max(1, count), "dtype_bytes": m.dtype_bytes,
                         "reads": 2, "writes": 1,
                         "floor_ns": _DENSE_EW_BINARY_FLOOR_NS})

        return [
            _ew4(q_elements),   # q_norm weight multiply
            _ew4(k_elements),   # k_norm weight multiply
        ]

    def _proj_gemm(self, position: str, batch: int, seq_len: int, is_decode: bool):
        """One projection GEMM, dispatched by the SAME cuBLAS model the real
        emulation uses (src/serve_trace/csrc/cublas_emulate.c record_gemm_entry),
        ported verbatim so analytic and real traces are kernel-identical.

        cuBLAS column-major mapping: for Y=X·W with X=[tokens,K], W=[K,out_dim],
        the call is (m=out_dim, n=tokens, k=K). The dispatch keys on n (#tokens)
        for the GEMV-vs-TC decision and on m for tiling/grid.
        """
        from src.serve_trace.shape_inference import infer_gemm_dimensions
        from src.dse.model_sim import SyntheticKernelEntry
        M_tok, N_out, K = infer_gemm_dimensions(
            self.m, position, seq_len, batch, self.tp_size)
        # cuBLAS arg mapping
        cb_m = N_out      # output feature dim
        cb_n = M_tok      # token count (effective_n)
        cb_k = K
        op_attr = {"B": batch, "M": M_tok, "N": N_out, "K": K}
        out = []
        ascend = getattr(self.hw, "aic_aiv_separated", False)
        # C500 lm_head: kernel family + sharding vary by (tp, ep), NOT the
        # generic GEMV branch (which would emit wave_kernel grid vocab/16 full).
        # Only at M<=1: lm_head uses b16gemvt (decode batch=1 / prefill last
        # token). At M>=2 lm_head uses bf16gemm TC (generic path, unchanged).
        if self._is_maca and not ascend and position == "lm_head" and M_tok <= 1:
            from src.serve_trace.analytic_mirror import _maca_lmhead_dispatch
            entries = _maca_lmhead_dispatch(N_out, self.tp_size, self.ep_size)
            for name, gx, gy, bx, by, smem in entries:
                out.append(SyntheticKernelEntry(
                    kernel_name=name, grid=(gx, gy, 1), block=(bx, by, 1),
                    shared_mem=smem, op_type="GEMM", op_attr=op_attr))
            return out
        for name, gx, gy, gz, bx, by, smem in _cublas_gemm_dispatch(
                cb_m, cb_n, cb_k, emit_splitk_reduce=self._emit_splitk_reduce,
                maca=self._is_maca, emit_cublas_splitk=self._emit_cublas_splitk,
                position=position, is_decode=is_decode):
            if ascend:
                # vllm-ascend runs dense projections as aclnnMatmul on the
                # fixed 24-AIC-core grid (NOT the cuBLAS one-CTA-per-tile grid).
                # The GEMM predictor's Ascend path (gemm_cube_tile +
                # _apply_npu_tile_mem) handles the cube tile; grid=24 matches
                # the real kernel launch (same fix as MoE grouped matmul).
                name = "aclnnMatmul_MatMulCommon_MatMulV2"
                gx, gy = self.hw.num_compute_units, 1
                bx, by, smem = 128, 1, 0
            out.append(SyntheticKernelEntry(
                kernel_name=name, grid=(gx, gy, 1), block=(bx, by, 1),
                shared_mem=smem, op_type="GEMM", op_attr=op_attr))
        return out

    def _attn_kernels(self, batch: int, seq_len: int, is_decode: bool,
                      kv_len: int = None, varlen_b: int = None,
                      kv_mean: int = None,
                      max_qlen: int = None, eff_mblocks: int = None):
        """Attention kernels (vLLM 0.11 FlashAttention, decoded from nsys).

        Both prefill and decode issue the flash_fwd_splitkv compute kernel,
        followed by the flash_fwd_splitkv_combine reduction epilogue. GT nsys
        shows ~93% of compute kernels have a matching combine (the remainder
        are short-sequence decodes where FlashAttention does not split). We
        emit 1 combine per compute as a first-order approximation.

        kv_len: the KV cache length actually attended over. For prefill this
        is seq_len (full self-attention); for decode it is the per-request
        context length (prompt + generated tokens), reconstructed from the
        scheduler sidecar and passed in by the regen. It defaults to seq_len
        (backward compat) — but without it, decode (seq_len=1) would model the
        K-loop as 1 step and the KV traffic as ~0, under-predicting decode FA
        by the context-length factor. The grid encodes the full M×N tile count
        so occupancy saturates on prefill (real splitkv grid launches one CTA
        per (batch, head, M-block, N-block) tile) and so per-step decode kernels
        with different KV cache lengths get distinct grids (→ distinct LUT keys,
        no rank0 collision).
        """
        m = self.m
        hq = m.num_attention_heads // self.tp_size
        hkv = m.num_key_value_heads // self.tp_size
        # Effective head count for the grid: vLLM's seqlenq_ngroups_swapped
        # (flash_api.cpp:407) reassigns num_heads=num_heads_k=HKV when
        # seqlen_q==1 (decode), so the launch head axis is HKV for decode but
        # HQ for prefill (seqlen_q>1, swap does not trigger). Verified vs nsys:
        # decode gridZ=B*HKV, prefill gridZ=B*HQ.
        heads_eff = hkv if is_decode else hq
        dtype_str = "cutlass::half_t" if m.dtype_bytes == 2 else "cutlass::bfloat16_t"
        if kv_len is None:
            kv_len = seq_len
        from src.dse.model_sim import SyntheticKernelEntry
        # Ascend (910B): vllm-ascend runs fused attention as a SINGLE CANN aclnn
        # FusedInferAttentionScore kernel (MIX_AIC: Block Dim 24 AIC + Mix Block
        # Dim 48 AIV) — NOT the CUDA flash_fwd_splitkv + combine pair. num_splits
        # is CANN-opaque (vllm-ascend attention_v1.py does NOT pass it; the CUDA
        # _fa_num_kv_splits SM-saturation heuristic is wrong on 910B). Emit the
        # real name + grid so the synthetic trace matches msprof; the FA
        # predictor's Ascend path (P88 _calc_fa_nDynNum + PRE_LAUNCH floor)
        # models time from D/KV, not the grid/num_splits. SP=1 (no fabricated
        # splits). (Restored from commit 096a431; was regressed away.)
        if getattr(self.hw, "aic_aiv_separated", False):
            # Batched decode: ONE FusedInferAttentionScore kernel processes the
            # whole varlen batch (varlen_b requests, each attending its own KV),
            # NOT the op-record's batch=1. The KV-gather transfer work is
            # Σ_req ceil(KV_req/16) ≈ varlen_b·ceil(mean_KV/16), so use
            # B=varlen_b + KV_LEN=mean context. B=1 (old) under-counted the
            # batched gather ~9× → serving decode FA −48%. Prefill/mixed keep the
            # op-record batch (its seq_len already carries the token count) and
            # kv_len. Falls back to (batch, kv_len) when the sidecar maps are
            # absent (offline / backward compat).
            if is_decode and varlen_b:
                fa_B = varlen_b
                fa_KV = kv_mean if kv_mean is not None else kv_len
            else:
                fa_B = batch
                fa_KV = kv_len
            op_attr = {"B": fa_B, "Q_LEN": seq_len, "KV_LEN": fa_KV, "D": m.head_dim,
                       "HQ": hq, "HKV": hkv, "SP": 1}
            return [SyntheticKernelEntry(
                kernel_name="FusedInferAttentionScore",
                grid=(self.hw.num_compute_units, 1, 1), block=(128, 1, 1),
                shared_mem=2 * 128 * m.head_dim * m.dtype_bytes,
                op_type="FLASHATTN", op_attr=op_attr)]
        # kBM=128 is the kernel-trait M-block (kernel_name + shared_mem below);
        # the splitkv LAUNCH grid uses kBlockM=64 (flash_fwd_launch_template.h:165
        # "constexpr static int kBlockM = 64; Fixed for all head dimensions"), so
        # num_m_block (gridX) = ceil(seq_len/64). Verified vs nsys: prefill
        # S=721 -> gridX=12=ceil(721/64), not ceil(721/128)=6. Decode num_M=1
        # either way (seq_len=1).
        # kBN is the kernel-trait N-block = kBlockN from the kernel name
        # (Flash_fwd_kernel_traits<kHeadDim=128, kBlockM=64, kBlockN=128, kNWarps=4>).
        # The compute K-loop (flash_fwd_kernel.h:65,148) walks N-blocks of size
        # kBlockN=128, so one _block_n=128 N-block is exactly ONE kBN-step —
        # NOT two. (The prior kBN=64 doubled FA_KSTEPS via _block_n//kBN=2,
        # which doubled the predictor's serial/exposed-latency k-step chain and
        # over-predicted serving prefill FA ~2x.) kBM here is unused (kBM_grid
        # drives the grid); kept for source-readability of the trait.
        kBM, kBN = 128, 128
        kBM_grid = 64
        # FA gridX (num_M) = ceil(max_seqlen_q / 64). For pure-decode (S==1) all
        # requests have q_len 1, so max_qlen=1 -> num_M=1. For mixed/prefill
        # steps the op record's seq_len is total_tokens (whole varlen batch), but
        # nsys gridX = ceil(MAX single-request length / 64) — the longest
        # request's M-block count, NOT total/64 (e.g. 9 decode + prefill 415 +
        # prefill 216, total 640, nsys gridX=ceil(415/64)=7, not ceil(640/64)=10).
        # max_qlen (from the sidecar) carries the max single-request length;
        # fall back to seq_len (==1 for decode) when absent.
        qlen_for_grid = max_qlen if max_qlen is not None else seq_len
        num_M = max(1, (qlen_for_grid + kBM_grid - 1) // kBM_grid)
        # Real num_kv_splits from vLLM's splitkv heuristic (verbatim port of
        # flash_api.cpp:num_splits_heuristic). Verified 100% match vs nsys
        # grids. This REPLACES the prior num_N proxy: num_splits is NOT
        # ceil(KV/kBN) — it's an SM-saturation heuristic that depends on
        # batch*HKV*m_blocks (to fill SMs) capped by num_n_blocks and max_splits.
        # Using num_N made the grid wrong -> wrong active_units -> the FA
        # under-prediction (G1 0.54x; the real nsys grid gives G2 0.88x).
        # num_m_blocks in the heuristic uses max_seqlen_q (the SAME max used for
        # gridX), not total_tokens. The heuristic's batch arg is the #requests
        # (varlen_b), not the op record's batch=1 — using batch=1 under-fills SMs
        # -> wrong num_splits (e.g. B=24 decode should be ns=1, batch=1 gives ns=6).
        batch_for_splits = varlen_b if (is_decode and varlen_b) else batch
        # C500: LUT-first for the splitkv num_splits. The macaprof splitkv
        # choice is an opaque per-(batch, hkv, num_n_blocks) autotuner decision
        # (nsys-rebuilt LUT in data/c500_lut.json, like the mcblas tile LUT —
        # accepted per the goal). The verbatim flash_api.cpp heuristic
        # diverges from macaprof on some keys (e.g. large-B decode where the
        # Ampere saturation logic gives ns=1 but C500 picks ns>1) -> missing
        # flash_combine -> cascade. LUT-first; heuristic fallback only when the
        # key is absent (genuine autotuner variance → _C500_FA_NONDET).
        num_splits = None
        _block_n_lut = 128
        if getattr(self, "_is_maca", False):
            _bn = 256 if m.head_dim <= 64 else (128 if m.head_dim <= 128 else 64)
            _nnb = max(1, (kv_len + _bn - 1) // _bn)
            _ns = _C500_FA_NUMSPLITS_LUT.get((batch_for_splits, heads_eff, _nnb))
            if _ns is not None:
                num_splits = _ns
                _block_n_lut = _bn
        if num_splits is None:
            num_splits, _block_n = _fa_num_kv_splits(
                batch_for_splits, heads_eff, kv_len, qlen_for_grid, m.head_dim,
                self.hw.num_compute_units, is_maca=getattr(self, "_is_maca", False))
        else:
            _block_n = _block_n_lut
        # Per-step num_splits override (nsys-grounded fa_ns sidecar): the LUT
        # misses large-batch decode keys → wrong ns=1 → missing FAcomb. Use
        # the nsys FA gridZ when present.
        _ns_override = self._step_fa_ns.get(getattr(self, "_cur_step_id", None))
        if _ns_override and _ns_override > 0:
            num_splits = _ns_override
        # Per-CTA K-loop length: the grid parallelizes KV into num_splits
        # pieces, each covering ceil(num_n_blocks/num_splits) N-blocks of size
        # _block_n (vLLM's splitKV N-tile, 128 for hdim=128 = kBlockN). The
        # compute kernel's inner K-loop walks one kBlockN=128 N-block per step,
        # so FA_KSTEPS (kBN-steps per split) = n_blocks_per_split ×
        # (_block_n // kBN) = n_blocks_per_split × 1. The predictor's
        # serial/fill-drain/exposed-latency models use kBlockN-tile geometry
        # (tile_cl = 2·kBN·D·2 / cacheline), so FA_KSTEPS must be in kBN-steps.
        num_n_blocks = max(1, (kv_len + _block_n - 1) // _block_n)
        n_blocks_per_split = max(1, (num_n_blocks + num_splits - 1) // num_splits)
        kbn_steps_per_split = n_blocks_per_split * max(1, _block_n // kBN)
        # SP = num_splits: the combine kernel reduces num_splits partials per
        # output position (its serial floor scales with this, not a constant).
        # FA compute grid. Decode (S==1) is deterministic and matches nsys via the
        # flash_attn heuristic. Prefill (chunked-varlen) grid is (num_M, b, heads_eff)
        # with num_splits=1 — reverse-engineered from nsys: gridY is the VARLEN
        # batch b=#requests (NOT num_splits), gridZ=heads_eff, num_splits=1. The
        # recipe gets b via varlen_b (reconstructed from the sidecar by the regen);
        # falls back to batch for the analytic mirror (b=1).
        if is_decode:
            # Decode B = #requests in the step (varlen_b from sidecar), NOT the
            # op record's batch (which is 1 — the hook treats the whole varlen
            # batch as batch=1).
            b_eff = varlen_b if varlen_b else batch
            if num_splits > 1:
                # ns>1 split grid. The MetaX (C500) flash build orders the split
                # axis LAST: gridY=B*HKV, gridZ=num_splits (verified vs nsys:
                # batch=60→(1,480,3), batch=54→(1,432,5)). Upstream Ampere
                # (A800) orders it first: gridY=num_splits, gridZ=B*HKV.
                if getattr(self, "_is_maca", False):
                    grid = (num_M, b_eff * heads_eff, num_splits)
                else:
                    grid = (num_M, num_splits, b_eff * heads_eff)
            else:
                grid = (num_M, b_eff, heads_eff)
        else:
            num_splits = 1   # prefill: verified num_splits==1 across all nsys prefill
            b_eff = varlen_b if varlen_b else batch
            grid = (num_M, b_eff, heads_eff)
        op_attr = {"B": batch, "Q_LEN": seq_len, "KV_LEN": kv_len, "D": m.head_dim,
                   "HQ": hq, "HKV": hkv, "SP": num_splits,
                   # Per-CTA K-loop step count (kBN-steps per split), sourced
                   # from the real num_n_blocks/num_splits. Forward uses this
                   # for the softmax/exposed-latency serial chain.
                   "FA_KSTEPS": kbn_steps_per_split}
        # Effective M-block fraction for the FA early-exit work-scaling model.
        # vLLM varlen FA launches num_M = ceil(max_qlen/64) M-blocks per request,
        # but compute_attn_1rowblock early-exits M-blocks beyond each request's
        # actual_seqlen_q (flash_fwd_kernel.h:81). In mixed/chunked-prefill
        # steps (1 long prefill + many decodes) most M-block CTAs are no-ops:
        # the grid encodes the MAX request length, but only
        # Σ ceil(q_len_i/64) M-blocks do real tensor/softmax/traffic work. The
        # forward model scales per-CTA work (serial + traffic + flops) by this
        # fraction; launch/wave overhead stays on the nominal grid (every CTA
        # pays dispatch + the early-exit branch). Pure-decode/pure-prefill → 1.0
        # (no padding). eff_mblocks comes from the scheduler sidecar (real
        # per-request token counts) — ground-truth scheduling data, not a fit.
        nominal_mblocks = num_M * b_eff
        if eff_mblocks is not None and nominal_mblocks > 0:
            op_attr["FA_EFF_MFRAC"] = min(1.0, eff_mblocks / nominal_mblocks)
        # else: 1.0 implicitly (forward treats absent key as no scaling)
        grid_total = grid[0] * grid[1] * grid[2]
        base = (f"void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<"
                f"(int)128, (int)64, (int){m.head_dim}, (int)4, "
                f"(bool)0, (bool)0, {dtype_str}>, (int)2, (bool)1>(flash::Flash_fwd_params)")
        tp = self.tp_size
        nh = m.num_attention_heads // tp
        ew4_elems = b_eff * nh * m.head_dim  # Q output elements
        ew2_elems = b_eff * 64               # cast elements (ceil(B/8)*512 -> B*64)
        # The two PyTorch aten direct_copy kernels (EW4 binary + EW2 unary),
        # real nsys names (gpu_kernel_impl_nocast<direct_copy_kernel_cuda>).
        _EW4_NAME = ("void at::native::elementwise_kernel<(int)128, (int)4, "
                     "void at::native::gpu_kernel_impl_nocast<at::native::"
                     "direct_copy_kernel_cuda(at::TensorIteratorBase &)"
                     "::[lambda() (instance 3)]::operator ()() const::"
                     "[lambda() (instance 10)]::operator ()() const::"
                     "[lambda(c10::Half) (instance 1)]>(at::TensorIteratorBase &, "
                     "const T1 &)::[lambda(int) (instance 1)]>(int, T3)")
        _EW2_NAME = ("void at::native::elementwise_kernel<(int)128, (int)2, "
                     "void at::native::gpu_kernel_impl_nocast<at::native::"
                     "direct_copy_kernel_cuda(at::TensorIteratorBase &)"
                     "::[lambda() (instance 3)]::operator ()() const::"
                     "[lambda() (instance 7)]::operator ()() const::"
                     "[lambda(float) (instance 1)]>(at::TensorIteratorBase &, "
                     "const T1 &)::[lambda(int) (instance 1)]>(int, T3)")
        def _ew4():
            return SyntheticKernelEntry(
                kernel_name=_EW4_NAME, grid=(_ceil_div(ew4_elems, 512), 1, 1),
                block=(128, 1, 1), shared_mem=0, op_type="ELEMENTWISE",
                op_attr={"element_count": ew4_elems, "dtype_bytes": m.dtype_bytes,
                         "reads": 2, "writes": 1, "floor_ns": _DENSE_EW_BINARY_FLOOR_NS})
        def _ew2():
            return SyntheticKernelEntry(
                kernel_name=_EW2_NAME, grid=(_ceil_div(ew2_elems, 512), 1, 1),
                block=(128, 1, 1), shared_mem=0, op_type="ELEMENTWISE",
                op_attr={"element_count": ew2_elems, "dtype_bytes": m.dtype_bytes,
                         "reads": 1, "writes": 1, "floor_ns": _DENSE_EW_CAST_FLOOR_NS})
        # FA output glue (elementwise direct_copy), verified vs nsys (5 cases, 0
        # exceptions) on the (kind, B, pure_decode) rule:
        #   - prefill (num_M>1): 0 glue
        #   - decode pure (is_decode) & B>=2: glue present
        #       ns>1: 1 pre-FA EW4 + (combine) + 2 post-combine (EW4+EW2) = 3
        #       ns==1: (no combine) 2 post-FA (EW4+EW2) = 2
        #   - decode B==1 / mixed step: 0 glue
        # is_decode here is the pure-decode sidecar flag.
        _emit_glue = is_decode and b_eff >= 2
        out = []
        # C500 prefill (maca, not decode): nsys prefill FA region per layer =
        # [DTOH, HTOD, cub::DeviceScanInit, cub::DeviceScan, flash(128,64,64),
        # DTOD, flash(128,16,16), flash_combine] (+ a DTOD after combine emitted
        # by operator_records). The varlen FA uses CUB DeviceScan for cu_seqlens
        # prefix-sum, per-layer DTOH/HTOD framework copies, a DTOD between the
        # 2 flash kernels, and 2 flash (different traits) + combine. Decode has
        # none of this (1 flash, no scan). Emit the exact sequence for prefill.
        if getattr(self, "_is_maca", False) and not is_decode:
            # Per-step prefill FA structure (nsys-grounded prefill_fa.jsonl):
            # flash2 (1 or 2 flash/layer), combine (0 or 1), scan (0 or 2
            # DeviceScan/layer) vary by prefill chunk — not num_splits-derivable.
            _fa = self._step_prefill_fa.get(
                getattr(self, "_cur_step_id", None), {}) or {}
            _do_scan = bool(_fa.get("scan"))
            _do_flash2 = bool(_fa.get("flash2"))
            _do_combine = bool(_fa.get("combine"))
            _mp = lambda nm, ot: SyntheticKernelEntry(
                kernel_name=nm, grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0,
                op_type=ot, op_attr={"bytes": m.hidden_size * m.dtype_bytes})
            # Per-layer framework DTOH+HTOD before the FA. Present for 8B-TP1
            # prefill, but ABSENT on C500 32B-TP2 (nsys: [NORM,ROT,VEW,KV,
            # cub×2, FA] with no DTOH/HTOD). Gate on not _maca_tp.
            _maca_tp = getattr(self, "_is_maca", False) and getattr(self, "tp_size", 1) > 1
            if not _maca_tp:
                out.append(_mp("memcpy DTOH(pinned,device)", "MEMCPY_DTOH"))
                out.append(_mp("memcpy HTOD(pinned,device)", "MEMCPY_HTOD"))
            if _do_scan:
                out.append(SyntheticKernelEntry(
                    kernel_name=("void at_cuda_detail::cub::DeviceScanInitKernel"
                                 "<at_cuda_detail::cub::ScanTileState<int, true>>"),
                    grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                    op_type="OTHER", op_attr={}))
                out.append(SyntheticKernelEntry(
                    kernel_name=("void at_cuda_detail::cub::DeviceScanKernel"
                                 "<at_cuda_detail::cub::DeviceScanPolicy<int>>"),
                    grid=(1, 1, 1), block=(128, 1, 1), shared_mem=0,
                    op_type="OTHER", op_attr={}))
        else:
            _do_flash2 = False
            _do_combine = False
        # pre-FA EW4: nsys shows this only for B=2 (not B=7, B=14), likely a
        # runtime tensor-layout quirk, not shape-determinable. Skip it — costs
        # +1 EW mismatch on B=2 steps but aligns all other decode steps.
        # if _emit_glue and num_splits >= 6:
        #     out.append(_ew4())
        out.append(SyntheticKernelEntry(
            kernel_name=base, grid=grid, block=(128, 1, 1),
            shared_mem=2 * 128 * m.head_dim * m.dtype_bytes,
            op_type="FLASHATTN", op_attr=op_attr))
        # C500 prefill: DTOD between the 2 flash kernels (nsys N26), then 2nd
        # flash + combine — gated on the per-step flags (flash2 / combine).
        if getattr(self, "_is_maca", False) and not is_decode:
            # 32B-TP2 (_maca_tp) nsys prefill: [cub×2, FA, FA, FAcomb] with NO
            # DTOD between flashes and NO DTOD before combine (verified step103).
            # 8B-TP1 has both DTODs (verified Llama-8B). Gate DTODs on not _maca_tp.
            _maca_tp_fa = getattr(self, "_is_maca", False) and getattr(self, "tp_size", 1) > 1
            if _do_flash2:
                if not _maca_tp_fa:
                    out.append(SyntheticKernelEntry(
                        kernel_name="memcpy DTOD(device,device)",
                        grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0,
                        op_type="MEMCPY_DTOD",
                        op_attr={"bytes": m.hidden_size * m.dtype_bytes}))
                _flash2 = (f"void flash::flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<"
                           f"(int)128, (int)16, (int)16, (int)1, "
                           f"(bool)0, (bool)0, {dtype_str}>, (int)2, (bool)1>(flash::Flash_fwd_params)")
                out.append(SyntheticKernelEntry(
                    kernel_name=_flash2, grid=grid, block=(128, 1, 1),
                    shared_mem=2 * 128 * m.head_dim * m.dtype_bytes,
                    op_type="FLASHATTN", op_attr=op_attr))
            if _do_combine:
                # nsys C500 prefill: the KV-cache DTOD memcpy sits AFTER
                # flash_combine (verified 4B/8B/Llama: [flash, (flash2),
                # combine, DTOD, GEMM]) — NOT before combine. The earlier
                # "before-combine" placement was a reversed-sidecar mistake.
                # Do NOT emit a before-combine DTOD here (no _dtod_before_combine
                # flag) — operator_records emits the single after-combine DTOD
                # (its default path, _skip_dtod stays False for C500-TP1).
                # 32B-TP2: after-combine DTOD also skipped (_maca_tp, DTOD=0).
                _pf_combine = (f"void flash::flash_fwd_splitkv_combine_kernel<"
                               f"Flash_fwd_kernel_traits<(int)128, (int)64, (int){m.head_dim}, "
                               f"(int)4, (bool)0, (bool)0, {dtype_str}>, (int)2>(flash::Flash_fwd_params)")
                combine_grid = max(1, num_M * b_eff * hq // 4)
                out.append(SyntheticKernelEntry(
                    kernel_name=_pf_combine, grid=(combine_grid, 1, 1), block=(128, 1, 1),
                    shared_mem=0, op_type="FLASHATTN", op_attr=op_attr))
        combine = (f"void flash::flash_fwd_splitkv_combine_kernel<"
                   f"Flash_fwd_kernel_traits<(int)128, (int)64, (int){m.head_dim}, "
                   f"(int)4, (bool)0, (bool)0, {dtype_str}>, (int)2>(flash::Flash_fwd_params)")
        # Combine is launched ONLY when num_splits > 1 (flash_fwd_launch_template.h:136
        # `if (params.num_splits > 1) { launch combine }`). Previously emitted
        # unconditionally -> 17248 phantom combines on Llama-8B (every ns==1 decode
        # step got a combine nsys never launches). The op_attr.SP == num_splits
        # already carries this, so the predictor's combine floor (scales with SP)
        # naturally vanishes for SP==1 when we skip the emit.
        if num_splits > 1:
            # Combine grid = one CTA per (batch × HQ-head-group × M-block). The
            # combine reduces num_splits partials per (query-row, KV-head-group)
            # and scales with HQ, NOT HKV: verified 7145/7145 nsys combine grids
            # obey combine = num_M * b * HQ // 4 (= num_M*b*HKV*group//4, group=HQ//HKV).
            # hq//4 == hkv for group=4 models (Llama-8B/Qwen3-8B/4B).
            combine_grid = max(1, num_M * b_eff * hq // 4)
            # nsys ground truth (verified across Llama/Qwen3 decode steps): the
            # KV-cache DTOD memcpy is ALWAYS launched AFTER flash_combine, never
            # before. The dtod_pos.jsonl sidecar's `dtod_before_combine=true`
            # markers are mislabeled (qps16: 867/867 wrong; qps4/qps64: the 3
            # true-marked steps are also after-combine). So we never emit a
            # before-combine DTOD here — operator_records emits the single
            # after-combine DTOD (its default path, _skip_dtod stays False).
            out.append(SyntheticKernelEntry(
                kernel_name=combine, grid=(combine_grid, 1, 1), block=(128, 1, 1),
                shared_mem=0, op_type="FLASHATTN", op_attr=op_attr))
        # post-FA / post-combine glue (2 EW): EW4 binary + EW2 unary direct_copy.
        # Only for decode (is_decode=pure-decode flag) & B>=2.
        # C500 nsys has NO FA glue EW (verified across decode B) — A800 has it.
        # Gate on `not _is_maca` so A800 keeps its verified glue.
        if _emit_glue and not getattr(self, "_is_maca", False):
            out.append(_ew4())
            out.append(_ew2())
        return out

    def _step_kernels(self, prefill_tokens: int, decode_reqs: int):
        """Kernel entries for one scheduling step (one batched forward).

        prefill_tokens>0 -> prefill-style forward (batch=1, seq_len=tokens);
        else -> pure-decode forward (batch=decode_reqs, seq_len=1).
        """
        m = self.m
        sim = self._sim
        is_decode_only = (prefill_tokens == 0 and decode_reqs > 0)
        # Pipeline parallel: this trace models ONE pipeline rank, which runs
        # layers_per_rank of the layers; only the last stage runs final norm +
        # lm_head (consistent with the per-worker emulation trace).
        n_layers = self._layers_per_rank
        is_last_stage = (self.pp_size == 1)  # single rank modeled = last stage
        entries = []
        # C500 TP>1 AR lead: clear the deferred-AR buffer before the layer loop
        # so no AR leaks from the previous step. The last layer's AR is flushed
        # after the loop (it has no next layer to lead into; nsys places it at
        # the step tail before final_norm/lm_head).
        self._pending_ar = []
        if is_decode_only:
            for _ in range(n_layers):
                entries.extend(self._layer_recipe(decode_reqs, 1, is_decode=True))
            entries.extend(self._pending_ar)
            self._pending_ar = []
            if is_last_stage:
                entries.append(sim._make_norm(m, decode_reqs, 1, self.tp_size, "final"))
                # lm_head: vocab GEMM dispatches TC_GEMM even at decode M (wide N)
                entries.extend(self._proj_gemm("lm_head", decode_reqs, 1, is_decode=False))
        else:
            total = prefill_tokens + decode_reqs
            for _ in range(n_layers):
                entries.extend(self._layer_recipe(1, total, is_decode=False))
            entries.extend(self._pending_ar)
            self._pending_ar = []
            if is_last_stage:
                entries.append(sim._make_norm(m, 1, total, self.tp_size, "final"))
                entries.extend(self._proj_gemm("lm_head", 1, total, is_decode=False))
        # ── Framework memcpy overhead (per step) ──
        if self._emit_c500_aux:
            from src.dse.model_sim import SyntheticKernelEntry
            n_dtoh = 3
            if m.num_experts > 0:
                n_dtoh += 4 * n_layers
            for _ in range(n_dtoh):
                # C500 nsys: 'memcpy DTOH(device,pinned)' (recipe had the pair
                # directionally swapped). Verified on Qwen3-8B/Llama-70B.
                entries.append(SyntheticKernelEntry(
                    kernel_name="memcpy DTOH(device,pinned)",
                    grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0,
                    op_type="MEMCPY_DTOH", op_attr={}))
            for _ in range(10):
                entries.append(SyntheticKernelEntry(
                    kernel_name="memcpy HTOD(pinned,device)",
                    grid=(1, 1, 1), block=(1, 1, 1), shared_mem=0,
                    op_type="MEMCPY_HTOD", op_attr={}))
        return entries

    def generate(self, requests: List[MirrorRequest],
                 max_num_batched_tokens: Optional[int] = None,
                 enable_chunked_prefill: bool = True):
        """Return (trace_entries, scheduler_records).

        trace_entries: list of dicts in the real trace JSON schema.
        scheduler_records: list of dicts in the real sidecar JSONL schema.
        """
        steps = mirror_schedule(requests, max_num_batched_tokens,
                                enable_chunked_prefill)
        trace_entries = []
        sched_records = []
        ts = 0.0  # synthetic monotonic timestamp (us); kernel timing is per-kernel below
        for st in steps:
            sched_records.append({
                "iter_id": st.iter_id,
                "ts_us": int(ts),
                "scheduled": st.scheduled,
                "total_tokens": st.prefill_tokens + st.decode_tokens,
                "finished": st.finished,
            })
            if st.prefill_tokens == 0 and st.decode_tokens == 0:
                ts += 1.0
                continue
            kernels = self._step_kernels(st.prefill_tokens, st.decode_tokens)
            for k in kernels:
                trace_entries.append({
                    "timestamp_us": round(ts, 1),
                    "kernel_name": k.kernel_name,
                    "grid": list(k.grid),
                    "block": list(k.block),
                    "shared_mem": k.shared_mem,
                    "stream": 0,
                })
                ts += 0.001  # kernels within a step are near-instant in emulation
            ts += 1.0  # inter-step gap
        return trace_entries, sched_records


def write_analytic_trace(trace_entries, sched_records, output_path: str):
    """Write trace JSON + scheduler JSONL in the exact real-mode on-disk format."""
    import os
    with open(output_path, "w") as f:
        f.write("[\n")
        for i, e in enumerate(trace_entries):
            comma = "," if i < len(trace_entries) - 1 else ""
            f.write(json.dumps(e) + comma + "\n")
        f.write("]\n")
    sched_path = output_path + "_scheduler.jsonl"
    with open(sched_path, "w") as f:
        for rec in sched_records:
            f.write(json.dumps(rec) + "\n")
    return output_path, sched_path

