#!/usr/bin/env python3
"""Per-kernel positional alignment of regen trace vs nsys (A800 TP1).

Reports offset=0 positional category match rate per step (regen vs nsys at the
same position). This is the true per-kernel alignment metric; the family-filter
grid zip in validate_trace_alignment.py only checks 4 families and hides global
order drift. Steps partitioned by cudaStreamSynchronize.end.

NOTE: this positional-alignment diagnostic predates the Phase 2 route-③ redesign
that deprecated byte-exact kernel-trace alignment; the step-level synthesis path
(OperatorTraceGenerator.records_to_trace) it was written against was deleted in
T4-4b. Kept as a historical diagnostic; the live trace is priced per-op via
records_to_trace_direct (no sampler remap).

Usage:
  python scripts/serving/validation/check_kernel_alignment.py --case <name>
  python scripts/serving/validation/check_kernel_alignment.py   # all A800 TP1 cases
"""
import sys, json, argparse
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.serving.validation.validate_trace_alignment import (
    load_nsys_all, load_regen_trace, _partition)

def _cat(name):
    n = name or ""
    for k, r in [("s16816gemm", "GEMM"), ("s1688gemm", "GEMM"),
                 ("xmma_gemm", "GEMM"),
                 ("flash_fwd_splitkv_kernel", "FA"), ("splitkv_combine", "FAcomb"),
                 ("rms_norm", "NORM"), ("fused_add_rms_norm", "NORM"),
                 ("rotary", "ROT"), ("reshape_and_cache", "KV"),
                 ("act_and_mul", "ACT"), ("vectorized_gather", "GATH"),
                 ("unrolled_elementwise", "UEW"), ("index_elementwise", "IDXE"),
                 ("vectorized_elementwise", "VEW"), ("elementwise_kernel", "EW"),
                 ("typeConvert", "TC"), ("reduce_kernel", "RED"), ("cutlass", "CUT")]:
        if k in n:
            return r
    return "OTH"

def check(case_dir, rank=0):
    regen = load_regen_trace(case_dir, rank=rank)
    nsys_kernels, iters, bounds = load_nsys_all(case_dir, device_id=rank)
    if regen is None or nsys_kernels is None:
        return None
    regen_by = defaultdict(list)
    for sid, nm, grid, op, *_ in regen:
        regen_by[sid].append(nm)
    nsys_by = _partition(nsys_kernels, bounds, iters)
    tm = tp_ = 0
    for idx, it in enumerate(iters):
        rs = [_cat(n) for n in regen_by.get(it, [])]
        ns = [_cat(r[1]) for r in nsys_by.get(it, [])]
        m = min(len(rs), len(ns)); tp_ += m
        for i in range(m):
            if rs[i] == ns[i]:
                tm += 1
    return tm, tp_, len(regen)

def _n_ranks(case_dir):
    import sqlite3
    p = case_dir / "nsys_profile.sqlite"
    if not p.exists():
        return 1
    conn = sqlite3.connect(str(p))
    n = conn.execute("SELECT COUNT(DISTINCT deviceId) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    conn.close()
    return max(1, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    a = ap.parse_args()
    EXP = ROOT / "serve_exp" / "eval_serving_sharegpt"
    cases = [EXP / a.case] if a.case else sorted(EXP.iterdir())
    for cd in cases:
        if not (cd / "meta.json").exists():
            continue
        if "A800" not in json.load(open(cd / "meta.json")).get("hw_config", ""):
            continue
        nr = _n_ranks(cd)
        any_ok = False
        for rank in range(nr):
            r = check(cd, rank=rank)
            if r is None:
                continue
            any_ok = True
            tm, tp_, n = r
            tag = f"[rank{rank}]" if nr > 1 else ""
            print(f"[align]{tag} {cd.name}: {tm}/{tp_} = {100*tm/tp_:.2f}%  regen={n}")
        if not any_ok:
            print(f"[align] {cd.name}: SKIP (no regen/nsys)")

if __name__ == "__main__":
    main()
