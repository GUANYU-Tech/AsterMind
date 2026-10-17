#!/usr/bin/env python3
"""Build per-step sampling_mode.jsonl from nsys (A800).

Detects each step's final gather kernel (post-lm_head sampler):
  indexSelectSmallIndex → greedy
  vectorized_gather     → multinomial
Steps partitioned by cudaStreamSynchronize.end (sync.end boundaries), aligned
to sidecar iter_id (drop capture-tail). Output: {iter_id, sampling_mode} per line.

Usage:
  python scripts/serve_predict/build_sampling_mode.py --case <name>
  python scripts/serve_predict/build_sampling_mode.py   # all A800 TP1
"""
import sys, sqlite3, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.serve_predict.validate_trace_alignment import load_nsys_all, _partition
from scripts.serve_predict.build_gemm_tile_lut_mnk import EXP_ROOT

def build(case_dir):
    # Multi-rank-safe partitioner (TP2 has 2x syncs interleaved non-monotonically;
    # the hand-rolled bounds=[nsys0]+sync_ends created 1938 empty windows for 32B-TP2).
    nk, iters, bounds = load_nsys_all(case_dir, device_id=0)
    if not nk or not iters or bounds is None:
        return None
    by_step = _partition(nk, bounds, iters)
    out = []
    for idx, it in enumerate(iters):
        names = [r[1] for r in by_step.get(it, [])]
        # final gather in the step (last occurrence of either marker)
        mode = "greedy"
        for nm in names:
            if "indexSelectSmallIndex" in nm:
                mode = "greedy"; break
            if "vectorized_gather" in nm:
                mode = "multinomial"; break
        # The sampler gather lands at the FRONT of sync window i but belongs to
        # the PREVIOUS step's lm_head (vLLM async scheduling runs the lm_head
        # sampler in the NEXT sync window — see operator_records sampler remap
        # and build_fill_grid's matching shift). regen looks up
        # sampling_mode[rec.step_id] where rec.step_id is the LM_HEAD's iter
        # (the producer), so record under iters[idx-1] (lm_head iter), NOT
        # iters[idx]. Window 0 has no lm_head predecessor → its own iter.
        owner = iters[idx - 1] if idx > 0 else it
        out.append({"iter_id": owner, "sampling_mode": mode})
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
            print(f"[sampling] {cd.name}: SKIP (no nsys/sidecar)")
            continue
        n_greedy = sum(1 for r in out if r["sampling_mode"] == "greedy")
        n_multi = len(out) - n_greedy
        with open(cd / "sampling_mode.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"[sampling] {cd.name}: {len(out)} steps, greedy={n_greedy} multinomial={n_multi}")

if __name__ == "__main__":
    main()
