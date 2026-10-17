#!/usr/bin/env python3
"""Build per-step fa_tm.jsonl for 910B from msprof.

CANN emits 2× aclnnInplaceCopy_TensorMoveAiCore_TensorMove (FA-output reshape)
per layer for B>aiv_core_count decode (and prefill), but SUPPRESSES the 2nd TM
of the FIRST layer for some steps (batch-changed / prev-not-pure-decode). The
suppress gate is NOT a clean sidecar signal (~7-8% of steps mis-classified by
batch_changed alone), and a 1-kernel mis-count cascades into ~1000 positional
mismatches per wrong step. So record the EXACT nsys body-TensorMove count per
step (ground truth) — operator_records uses it to set the first-layer suppress
flag (suppress iff tm_count == 2*num_hidden_layers - 1).

Body TM count = aclnnInplaceCopy_TensorMove kernels BEFORE the sampler Fill
(the sampler's own Fill/Embedding/etc. are not FA-output TMs).

Writes <case_dir>/fa_tm.jsonl: {iter_id, tm_count} per non-idle step. Absent
step -> regen falls back to its batch_changed/prev_not_pd heuristic.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.serving.validation.validate_trace_alignment import (
    load_msprof_all, _partition_910b_by_fa)

FILL_NAME = "aclnnInplaceFillScalar_FillAiCore_Fill"
TM_SUBSTR = "InplaceCopy_TensorMove"


def build(case_dir, device_id=0):
    case_dir = Path(case_dir)
    nk, iters, bounds = load_msprof_all(case_dir, device_id=device_id)
    if nk is None or not iters:
        return 0
    nby = _partition_910b_by_fa(nk, iters)
    out_path = case_dir / "fa_tm.jsonl"
    n = 0
    with open(out_path, "w") as f:
        for it in iters:
            entries = nby.get(it, [])
            # body TMs = InplaceCopy_TensorMove before the first sampler Fill.
            fill_i = None
            for i, e in enumerate(entries):
                if len(e) > 1 and e[1] == FILL_NAME:
                    fill_i = i
                    break
            end = fill_i if fill_i is not None else len(entries)
            tm = sum(1 for i in range(end)
                     if len(entries[i]) > 1 and TM_SUBSTR in entries[i][1])
            f.write(json.dumps({"iter_id": it, "tm_count": tm}) + "\n")
            n += 1
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--device-id", type=int, default=0)
    a = ap.parse_args()
    n = build(a.case_dir, a.device_id)
    print(f"[fa_tm] wrote {n} steps -> {a.case_dir}/fa_tm.jsonl")


if __name__ == "__main__":
    main()
