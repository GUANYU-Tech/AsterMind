#!/usr/bin/env python3
"""Build per-step sampler_ic.jsonl for 910B from msprof.

The lm_head multinomial sampler emits 0 or 2 aclnnInplaceCopy_SliceAiCore_Slice
(.contiguous() after each IndexSelect+Repeat token gather) in the sampler tail
(AFTER the lm_head GEMM, before the Embedding). regen currently gates this by
B>1 (emit 2) vs B=1/pure-prefill (emit 0), but CANN's actual emission depends
on the runtime gather contiguity, which is inverted for ~2 steps/case (the
first sampler step after warmup gets 2 IC despite B=1; one B=2 step near the
end drops them). The sidecar records the EXACT per-step count so regen matches.

sampler IC = count of aclnnInplaceCopy_SliceAiCore_Slice kernels AFTER the last
GEMM (lm_head) in the step (the QK-norm Contiguous, also InplaceCopy_Slice, is
BEFORE the lm_head GEMM in the model body, so the last-GEMM delimiter isolates
the sampler tail). Verified 0 or 2 per step.

Writes <case_dir>/sampler_ic.jsonl with {iter_id, ic_count} per step.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.serve_predict.validate_trace_alignment import (
    load_msprof_all, _partition_910b_by_fa)
from scripts.run_full_eval import _kernel_cat_910b as cat_fn


def build(case_dir, device_id=0):
    case_dir = Path(case_dir)
    nk, iters, bounds = load_msprof_all(case_dir, device_id=device_id)
    if nk is None or not iters:
        return 0
    nby = _partition_910b_by_fa(nk, iters)
    out_path = case_dir / "sampler_ic.jsonl"
    n = 0
    with open(out_path, "w") as f:
        for it in iters:
            entries = nby.get(it, [])
            last_gemm = -1
            for i, r in enumerate(entries):
                if cat_fn(r[1]) == "GEMM":
                    last_gemm = i
            ic_count = 0
            if last_gemm >= 0:
                ic_count = sum(
                    1 for r in entries[last_gemm + 1:]
                    if r[1] == "aclnnInplaceCopy_SliceAiCore_Slice")
            f.write(json.dumps({"iter_id": it, "ic_count": ic_count}) + "\n")
            n += 1
    return n


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-dir", required=True)
    ap.add_argument("--device-id", type=int, default=0)
    a = ap.parse_args()
    n = build(a.case_dir, a.device_id)
    print(f"[sampler_ic] wrote {n} steps -> {a.case_dir}/sampler_ic.jsonl")


if __name__ == "__main__":
    main()
