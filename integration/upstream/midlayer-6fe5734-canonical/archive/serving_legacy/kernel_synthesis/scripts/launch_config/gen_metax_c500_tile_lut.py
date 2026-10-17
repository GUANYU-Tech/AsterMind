"""Build the C500/MACA GEMM tile LUT keyed by (M_tok, N_out, K, dtype,
layout) — the FULL cuBLAS-equivalent input — from C500 serving nsys ground
truth aligned via operator-record (M,N,K).

WHY THE KEY CARRIES K (per goal spec):
  mcblas tile selection is a DETERMINISTIC function of (M, N, K, dtype, layout)
  for fixed input — the autotuner is deterministic per shape. The prior
  (M_bucket, N_out)-only key COLLAPSED different K into one bucket (e.g.
  M=768 N=8192: K=2048 -> 128x256, K=7168 -> 256x128, a 3.2x duration gap), and
  the resulting multi-tile variance was mislabeled "autotuner non-determinism"
  when it was actually an incomplete key. Including K (and the constant
  dtype=bf16, layout=tn) converts that variance into exact matches; only TRUE
  run-to-run autotuner variance survives in _MACA_TILE_NONDET (warning, error).

DATA SOURCE:
  scripts.serve_predict.build_eval_dataset._c500_align_gemms aligns each regen-
  trace GEMM's REAL (M,N,K) [plugin_trace op_attr] to its nsys ACTUAL tile [parsed
  from kernel name] by per-step GEMM ordinal. This script aggregates that aligned
  (M,N,K,actual_tile) across all 21 C500 serving cases into:
  _MACA_TILE_LUT      : {(M_tok, N_out, K): (tile_m, tile_n, tile_k, block, layout, stage_sb)}  # EXACT M, no pad
  _MACA_TILE_NONDET   : {(M_tok, N_out, K)} where dominant < 95% of obs = TRUE cross-process variance
  (dtype=bf16, layout=tn are constant for C500 dense GEMMs, so omitted from the
  key tuple — they are baked into the emitted kernel name.)

Run:  python -m scripts.kernel.gen_c500_tile_lut
"""
import os
import sys
import multiprocessing as mp
from collections import defaultdict, Counter

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

C500_DIR = os.path.join(REPO, "serve_exp", "eval_serving_sharegpt_c500")

# tile_family metadata: (block_threads, layout, stage_suffix) per observed C500
# mcblas kernel family. Sourced from kernel_dataset/metax_c500/gemm.json.
TILE_META = {
    "64x32x256":  (256, "2m1n8k", "bf16_bf16_fp32_sb_0_2"),
    "128x64x256": (256, "2m1n8k", "bf16_bf16_fp32_sb_0_0"),
    "128x128x128":(256, "4m1n8k", "4stage_bf16_bf16_fp32_sb_0_0"),
    "256x256x64": (512, "2m1n8k", "4stage_bf16_bf16_fp32_sb_0_0"),
    "256x192x64": (512, "2m1n8k", "4stage_bf16_bf16_fp32_sb_0_0"),
    "256x128x64": (512, "2m1n8k", "4stage_bf16_bf16_fp32_sb_0_0"),
}


def _align_one(case_dir):
    """Run _c500_align_gemms on one case; return list of (M,N,K,actual_tile)."""
    from pathlib import Path
    from scripts.serve_predict.build_eval_dataset import _c500_align_gemms
    recs = _c500_align_gemms(Path(case_dir))
    if not recs:
        return []
    return [(r["M"], r["N"], r["K"], r.get("actual_tile"))
            for r in recs if r.get("actual_tile")]


def scan():
    cases = [os.path.join(C500_DIR, c)
             for c in sorted(os.listdir(C500_DIR))
             if os.path.isdir(os.path.join(C500_DIR, c))]
    agg = defaultdict(Counter)   # (M_pad, N_pad, K) -> Counter(tile)
    print(f"[gen_c500_lut] aligning {len(cases)} cases (parallel, 6 workers)...")
    with mp.Pool(min(6, len(cases))) as pool:
        for recs in pool.imap_unordered(_align_one, cases):
            for m_tok, n_out, k, tile in recs:
                if tile not in TILE_META:
                    continue
                agg[(m_tok, n_out, k)][tile] += 1  # EXACT M,N (no pad): autotuner sees exact M (partial-tile cost)
    return agg


def emit(agg):
    print("_MACA_TILE_LUT = {")
    n = 0
    nondet = []
    for key in sorted(agg):
        c = agg[key]
        dom, cnt = c.most_common(1)[0]
        total = sum(c.values())
        tm, tn, tk = (int(x) for x in dom.split("x"))
        blk, lay, stage_sb = TILE_META[dom]
        print(f"    ({key[0]}, {key[1]}, {key[2]}): "
              f"({tm}, {tn}, {tk}, {blk}, \"{lay}\", \"{stage_sb}\"),")
        n += 1
        if len(c) > 1 and cnt / total < 0.95:
            nondet.append((key, c, dom, cnt / total))
    print(f"}}  # {n} entries")
    print("\n_MACA_TILE_NONDET = {")
    for key, c, dom, frac in sorted(nondet, key=lambda x: x[0]):
        print(f"    ({key[0]}, {key[1]}, {key[2]}),")
    print(f"}}  # {len(nondet)} nondeterministic keys")
    print("\n# --- variance detail (M_pad, N_pad, K): dominant frac, variants ---")
    for key, c, dom, frac in sorted(nondet, key=lambda x: x[0]):
        var = "; ".join(f"{t}={cnt}" for t, cnt in c.most_common())
        print(f"#   ({key[0]}, {key[1]}, {key[2]}): {var}")


if __name__ == "__main__":
    emit(scan())
