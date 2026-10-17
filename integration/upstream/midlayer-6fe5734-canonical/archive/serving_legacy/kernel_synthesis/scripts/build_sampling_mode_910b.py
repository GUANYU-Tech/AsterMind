#!/usr/bin/env python3
"""Build per-step sampling_mode.jsonl for 910B from msprof.

A800 derives greedy/multinomial from nsys kernel names (indexSelectSmallIndex
= greedy, vectorized_gather = multinomial). 910B's signal: the multinomial
sampler scatters the sampled token via aclnnInplaceScatter (Index +
InplaceScatter_LinearIndex + InplaceScatter_ScatterElements) — greedy uses
argmax's IndexPut only (no InplaceScatter). So a step is 'multinomial' iff its
msprof kernels contain any aclnnInplaceScatter.

Writes <case_dir>/sampling_mode.jsonl with {iter_id, sampling_mode} per step,
consumed by build_eval_dataset's gen.set_step_sampling_mode.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.serve_predict.validate_trace_alignment import (
    load_msprof_all, _partition_910b_by_fa)


def build(case_dir, device_id=0):
    case_dir = Path(case_dir)
    nk, iters, bounds = load_msprof_all(case_dir, device_id=device_id)
    if nk is None or not iters:
        return 0
    nby = _partition_910b_by_fa(nk, iters)
    out_path = case_dir / "sampling_mode.jsonl"
    n = 0
    with open(out_path, "w") as f:
        for it in iters:
            entries = nby.get(it, [])
            has_scatter = any(
                "InplaceScatter" in str(e[1]) for e in entries if len(e) > 1)
            mode = "multinomial" if has_scatter else "greedy"
            f.write(json.dumps({"iter_id": it, "sampling_mode": mode}) + "\n")
            n += 1
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--device-id", type=int, default=0)
    a = ap.parse_args()
    n = build(a.case_dir, a.device_id)
    print(f"[sampling_mode] wrote {n} steps -> {a.case_dir}/sampling_mode.jsonl")


if __name__ == "__main__":
    main()
