#!/usr/bin/env python3
"""Build per-step lm_head_cast.jsonl from nsys (A800, multi-rank-safe).

The lm_head logits.to(float32) cast kernel (the FIRST post-lm_head kernel,
right after the lm_head GEMM, or after the TP AllGather for TP>1) is NOT a
fixed `unrolled_elementwise<direct_copy>` (UEW): for large logits (TP>1
AllGather doubles them; big prefill batches) torch picks the vectorized
`elementwise_kernel<(int)128,(int)4,...>` (EW category); for small (decode
B=1, single-rank) it picks `unrolled_elementwise<direct_copy>` (UEW).
regen hardcodes the unrolled variant, mismatching the EW cases (32B-TP2
prefill/mixed). This builder records, per step (attributed to the step's
iter — the cast sits at the END of the lm_head's own step, NOT remapped),
the cast kernel type ('unrolled'/'elementwise') + grid, so operator_records
emits the right one. TP1 dense models are all 'unrolled' (unchanged, 100%).

Uses the correct multi-rank partitioner (load_nsys_all + _partition).
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.serve_predict.validate_trace_alignment import load_nsys_all, _partition
from scripts.serve_predict.build_gemm_tile_lut_mnk import EXP_ROOT


def build(case_dir, device_id=0):
    nk, iters, bounds = load_nsys_all(case_dir, device_id=device_id)
    if not nk or not iters or bounds is None:
        return None
    by_step = _partition(nk, bounds, iters)
    out = []
    for idx, it in enumerate(iters):
        seq = by_step.get(it, [])
        if not seq:
            continue
        # lm_head GEMM = the LAST s16816/s1688/xmma in the step (largest gridX,
        # the vocab matmul). The cast(s) = the elementwise direct_copy kernel(s)
        # after it (skipping the TP AllGather that sits between the GEMM and the
        # cast). vLLM emits ONE cast for B=1 decode (logits.to(float32) → UEW)
        # but TWO casts for B>1 prefill/mixed (the tensor_model_parallel_all_gather
        # output reassembly → elementwise_kernel EW, then the .to(float32)/
        # vocab-slice materialization → unrolled_elementwise UEW), each grid=
        # ceil(B*V/512). Collect ALL consecutive EW/UEW after the GEMM (the
        # AllGather/CDR may sit between) until a non-cast kernel (rms_norm /
        # reduce / sampler / triton). The torch elementwise launch grid is a
        # heuristic (like cuBLAS) — recorded from nsys, not reverse-fit.
        lh_idx = None
        for i in range(len(seq) - 1, -1, -1):
            nm = seq[i][1]
            if "s16816gemm" in nm or "s1688gemm" in nm or "xmma_gemm" in nm:
                lh_idx = i; break
        if lh_idx is None:
            continue
        casts = []
        allgather_grid = None
        for j in range(lh_idx + 1, len(seq)):
            nm = seq[j][1]
            if "AllGather" in nm:
                # the lm_head TP AllGather (ncclDevKernel_AllGather_RING_LL).
                # gridX = NCCL nBlocks (15 or 16) — NCCL's internal launch
                # heuristic (nThreads 512/544, nBlocks), NOT source-derivable
                # (like cuBLAS). Recorded per-step so the regen emits the exact
                # nsys gridX (the lm_head AllGather is 1/step, TP>1 only).
                g = seq[j][2]
                allgather_grid = [g[0], g[1], g[2]]
                continue
            if "cross_device_reduce" in nm:
                continue  # skip CDR (not between lm_head GEMM and cast, but be safe)
            if "elementwise_kernel" in nm or "unrolled_elementwise" in nm:
                ct = "unrolled" if "unrolled_elementwise" in nm else "elementwise"
                g = seq[j][2]
                casts.append({"cast_type": ct, "cast_grid": [g[0], g[1], g[2]]})
                continue
            # hit a non-elementwise (rms_norm, sampler, etc.) -> cast run ends
            break
        if casts:
            rec = {"iter_id": it, "casts": casts}
            if allgather_grid is not None:
                rec["allgather_grid"] = allgather_grid
            out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    a = ap.parse_args()
    cases = [EXP_ROOT / a.case] if a.case else sorted(EXP_ROOT.iterdir())
    for cd in cases:
        if not (cd / "meta.json").exists():
            continue
        if "A800" not in json.load(open(cd / "meta.json")).get("hw_config", ""):
            continue
        out = build(cd)
        if out is None:
            print(f"[lm_head_cast] {cd.name}: SKIP")
            continue
        with open(cd / "lm_head_cast.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        from collections import Counter
        c = Counter(len(r.get("casts", [])) for r in out)
        print(f"[lm_head_cast] {cd.name}: {len(out)} steps, casts/step dist {dict(c)}")


if __name__ == "__main__":
    main()
