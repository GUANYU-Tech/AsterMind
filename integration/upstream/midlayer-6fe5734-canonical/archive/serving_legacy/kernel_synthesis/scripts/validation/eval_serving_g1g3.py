#!/usr/bin/env python3
"""Legacy G1/G3 batch driver over synthesized serving traces.

Thin orchestration only — all metrics come from build_eval_dataset.run_comparison
(G1 = group1_synth_gpu, G3 = group3_real_e2e). GPU is predicted ONCE per
experiment inside run_prediction; run_comparison derives both G1 (pure GPU vs
nsys gpu_ns) and G3 (GPU + nsys real CPU vs wall_s) from that single prediction.
G2 (oracle grid) is available via run_g2_comparison but not required.

For each experiment dir under serve_exp/eval_serving_sharegpt/ (which already has
plugin_trace + nsys + serving_metrics), this runs stages predict+compare and
prints the resulting group1/group3 APEs. No re-capture of GPU/nsys.
"""
import sys, os, json, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.serve_predict.build_eval_dataset import (
    run_prediction, run_comparison)

EXP_ROOT = ROOT / "serve_exp" / "eval_serving_sharegpt"


def _case_from_meta(d: Path):
    m = json.load(open(d / "meta.json"))
    return {
        "id": m["case_id"],
        "model": m["model"],
        "tensor_parallel_size": m.get("tensor_parallel_size", 1),
        "expert_parallel_size": m.get("expert_parallel_size", 1),
    }, m.get("hw_config", "configs/hardware/nvidia_a800.yaml")


def eval_one(name: str, overlap_mode: str = "serial", validate_tiles: bool = False):
    d = EXP_ROOT / name
    case, hw_config = _case_from_meta(d)
    # Force current code path: drop the prediction LUT cache.
    lut = d / "prediction_lut.json"
    if lut.exists():
        lut.unlink()
    run_prediction(case, d, hw_config, overlap_mode=overlap_mode)
    cmp = run_comparison(case, d)  # populates group1_synth_gpu, group3_real_e2e
    g1 = (cmp or {}).get("group1_synth_gpu", {})
    g3 = (cmp or {}).get("group3_real_e2e", {})
    result = {
        "case": name,
        "tp": case["tensor_parallel_size"],
        "ep": case["expert_parallel_size"],
        "G1_ape": g1.get("ape"),                    # GPU-time-weighted per-rank
        "G3_ape": g3.get("ape"),
        "G1_per_rank_ape": g1.get("per_rank_ape"),
        "G1_n_ranks": g1.get("n_ranks"),
        "G3_pred_ms": g3.get("pred_ms"),
        "G3_real_ms": g3.get("real_ms"),
    }
    # Tile-LUT validation: for each nsys GEMM kernel, look up the K-aware LUT
    # with the op-record (M,N,K) and compare to the nsys actual tile. Prints a
    # per-case mismatch report (nondeterministic ties are expected; genuine
    # surprises indicate an LUT coverage gap or wrong dominant tile).
    if validate_tiles:
        from scripts.serve_predict.build_gemm_tile_lut_mnk import (
            validate_gemm_tile_lut)
        n, mism, nondet = validate_gemm_tile_lut(d)
        result["tile_validate_n"] = n
        result["tile_validate_mismatch"] = mism
        result["tile_validate_nondet"] = nondet
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None, help="single experiment dir name")
    ap.add_argument("--overlap-mode", default="serial", choices=["serial", "max"],
                    help="e2e CPU assembly: serial=GPU+CPU (default), max=max(GPU,CPU)")
    ap.add_argument("--out", default=None, help="write result json")
    ap.add_argument("--validate-tiles", action="store_true",
                    help="also validate the K-aware GEMM tile LUT vs nsys tiles")
    a = ap.parse_args()
    if a.case:
        r = eval_one(a.case, overlap_mode=a.overlap_mode,
                     validate_tiles=a.validate_tiles)
        print(json.dumps(r))
        if a.out:
            json.dump(r, open(a.out, "w"))
    else:
        print("specify --case (run one per process; parallelize with xargs)")
