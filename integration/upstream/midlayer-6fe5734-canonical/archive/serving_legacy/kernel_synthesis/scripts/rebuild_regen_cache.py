#!/usr/bin/env python3
"""Rebuild ONLY the regen cache (no full predict pipeline) for given cases.

Calls _regenerate_trace_with_step_id directly so check_kernel_alignment sees
the trace produced by the CURRENT operator_records.py / analytic_mirror.py.
"""
import sys, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.serve_predict.model_config import load_model_config
from scripts.serve_predict.build_eval_dataset import _regenerate_trace_with_step_id

def rebuild(cid):
    cd = ROOT / "serve_exp" / "eval_serving_sharegpt" / cid
    if not (cd / "meta.json").exists():
        print(f"[rebuild] {cid}: no meta.json"); return
    meta = json.load(open(cd / "meta.json"))
    mc = load_model_config(meta["model"])
    tp = meta.get("tensor_parallel_size", 1)
    ep = meta.get("expert_parallel_size", 1)
    hw = meta["hw_config"]
    # delete stale caches so a key collision never serves old data
    for p in cd.glob(".regen_cache_*.pkl.gz"): p.unlink()
    for p in cd.glob(".regen_cache_*.pkl"): p.unlink()
    for p in cd.glob(".regen_cache_*.json.gz"): p.unlink()
    _regenerate_trace_with_step_id(cd, mc, tp, ep, hw)
    print(f"[rebuild] {cid}: done")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", required=True)
    a = ap.parse_args()
    for cid in a.cases:
        rebuild(cid)

if __name__ == "__main__":
    main()
