#!/usr/bin/env python3
"""Build per-step fill_grid.jsonl from nsys (A800).

For each step, find the sampler's FillFunctor kernel (the output-buffer fill
after argmax) and record (gridX, type) where type is 'unrolled' or 'vectorized'.
Steps partitioned by cudaStreamSynchronize.end, aligned to sidecar iter_id.

Usage:
  python scripts/serve_predict/build_fill_grid.py --case <name>
  python scripts/serve_predict/build_fill_grid.py   # all A800 TP1
"""
import sys, sqlite3, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.serve_predict.validate_trace_alignment import load_nsys_all, _partition
from scripts.serve_predict.build_gemm_tile_lut_mnk import EXP_ROOT

def build(case_dir):
    # Multi-rank-safe partitioner: load_nsys_all(device_id=0) + _partition
    # dedupe/sort the sync events (TP2 has 2x syncs interleaved non-monotonically;
    # the hand-rolled bounds=[nsys0]+sync_ends created 1938 empty windows for 32B-TP2).
    nk, iters, bounds = load_nsys_all(case_dir, device_id=0)
    if not nk or not iters or bounds is None:
        return None
    by_step = _partition(nk, bounds, iters)
    out = []
    for idx, it in enumerate(iters):
        seq = by_step.get(it, [])
        if not seq:
            continue
        # FillFunctor = the sampler output fill; pick the FillFunctor kernel
        # whose grid is small (16/8/15...) — the post-argmax fill, NOT the
        # large KV-fill (grid=2884). The sampler FillFunctor is the one with
        # 'FillFunctor<long>' and grid <= 64 (output token buffer).
        fg = None; ft = None
        for r in seq:
            nm = r[1]; gx = r[2][0] if r[2] else 0
            if "FillFunctor" in nm and "long" in nm and gx <= 64:
                ft = "unrolled" if "unrolled_elementwise" in nm else "vectorized"
                fg = gx
                break
        if fg is not None:
            # The sampler FillFunctor lands at the FRONT of sync window i but
            # belongs to the PREVIOUS step's lm_head (vLLM async scheduling
            # runs the lm_head sampler in the NEXT sync window). regen looks up
            # fill_grid[rec.step_id] where rec.step_id is the LM_HEAD's iter
            # (the producer), so record under iters[idx-1] (lm_head iter), NOT
            # iters[idx] (the window the FillFunctor sits in). Window 0 has no
            # lm_head predecessor (capture tail / warmup) → its own iter.
            owner = iters[idx - 1] if idx > 0 else it
            out.append({"iter_id": owner, "fill_grid": fg, "fill_type": ft})
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
            print(f"[fill_grid] {cd.name}: SKIP")
            continue
        with open(cd / "fill_grid.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"[fill_grid] {cd.name}: {len(out)} steps with FillFunctor")

if __name__ == "__main__":
    main()
