#!/usr/bin/env python3
"""Build ALL per-rank nsys-grounded sidecars for a 32B-TP2 case:
prefill_front_ew.rank{N}.jsonl (+head_samp), idx_ew.rank{N}.jsonl,
tail_fill.rank{N}.jsonl, alt_tail.rank{N}.jsonl, devdst_count.rank{N}.jsonl,
sampler_glue.rank{N}.jsonl. Parametric: --case <case_id>."""
import sys, json, argparse, statistics
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"scripts"/"serve_predict"))
from serve_predict.validate_trace_alignment import load_nsys_all, _partition
from run_full_eval import _kernel_cat
from build_sampler_glue import build as build_glue


def build_all(cd, nrank=2):
    for dev in range(nrank):
        nk, iters, bounds = load_nsys_all(cd, dev)
        if not nk:
            print(f"  rank{dev}: no nsys"); continue
        nby = _partition(nk, bounds, iters)
        first_iter = iters[0] if iters else None
        # prefill_front_ew + head_samp
        pfx_path = cd / f"prefill_front_ew.rank{dev}.jsonl"
        npre = nhs = 0
        with open(pfx_path, "w") as f:
            for it in iters:
                ns = nby.get(it) or []
                samp_idx = next((i for i,k in enumerate(ns[:15]) if "triton_poi" in k[1]), -1)
                prefix = []
                if samp_idx > 0:
                    for i in range(samp_idx):
                        k = ns[i]
                        prefix.append({"cat": _kernel_cat(k[1]), "grid": list(k[2]),
                                        "block": list(k[3]), "name": k[1]})
                    npre += 1
                head_samp = None
                if it == first_iter:
                    idx_pos = next((i for i,k in enumerate(ns[:15]) if "indexSelect" in k[1]), -1)
                    if idx_pos > 0:
                        hs_pos = next((i for i,k in enumerate(ns[:idx_pos]) if "triton_poi" in k[1]), -1)
                        if hs_pos >= 0:
                            k = ns[hs_pos]
                            head_samp = {"name": k[1], "grid": list(k[2]), "block": list(k[3])}
                            nhs += 1
                f.write(json.dumps({"iter_id": it, "prefix": prefix, "head_samp": head_samp}) + "\n")
        # idx_ew
        ie_path = cd / f"idx_ew.rank{dev}.jsonl"

        def _is_head_blocker(nm):
            return ("AllReduce" in nm or "mcblas" in nm or "gemm" in nm.lower()
                    or "rms_norm" in nm or "rotary" in nm or "reshape_and_cache" in nm
                    or "flash" in nm)
        with open(ie_path, "w") as f:
            for it in iters:
                ns = nby.get(it) or []
                head = ns[:20]
                idx_pos = next((i for i,k in enumerate(head) if "indexSelect" in k[1]), -1)
                has_idx = idx_pos >= 0
                if has_idx:
                    has_mfill = any(_kernel_cat(head[j][1])=="EW" for j in range(idx_pos+1, min(idx_pos+4, len(head))))
                else:
                    # masked_fill EW = an elementwise_kernel BEFORE the first
                    # head-blocker (AR/GEMM/NORM/...), NOT the post-GEMM EW.
                    has_mfill = False
                    for j in range(min(8, len(head))):
                        if _is_head_blocker(head[j][1]):
                            break
                        if "elementwise_kernel" in head[j][1]:
                            has_mfill = True; break
                f.write(json.dumps({"iter_id": it, "has_idx": has_idx, "has_mfill": has_mfill}) + "\n")
        # tail_fill
        tf_path = cd / f"tail_fill.rank{dev}.jsonl"
        with open(tf_path, "w") as f:
            for it in iters:
                ns = nby.get(it) or []
                ag = max((i for i,k in enumerate(ns) if "AllGather" in k[1]), default=-1)
                ft = "none"
                if ag >= 0 and ag+10 < len(ns):
                    k = ns[ag+10]
                    if "FillFunctor" in k[1]:
                        ft = "vew" if "vectorized" in k[1] else "uew"
                f.write(json.dumps({"iter_id": it, "fill_type": ft}) + "\n")
        # alt_tail
        at_path = cd / f"alt_tail.rank{dev}.jsonl"
        nat = 0
        with open(at_path, "w") as f:
            for it in iters:
                ns = nby.get(it) or []
                ag = max((i for i,k in enumerate(ns) if "AllGather" in k[1]), default=-1)
                tail = []
                if ag >= 0:
                    rest = ns[ag+1:]
                    if any("triton_poi" in k[1] for k in rest):
                        for k in rest:
                            tail.append({"cat": _kernel_cat(k[1]), "grid": list(k[2]),
                                         "block": list(k[3]), "name": k[1]})
                        nat += 1
                f.write(json.dumps({"iter_id": it, "tail": tail}) + "\n")
        # devdst_count
        dd_path = cd / f"devdst_count.rank{dev}.jsonl"
        with open(dd_path, "w") as f:
            for it in iters:
                ns = nby.get(it) or []
                nd = sum(1 for k in ns if "DevDst" in k[1])
                f.write(json.dumps({"iter_id": it, "n_devdst": nd}) + "\n")
        print(f"  rank{dev}: prefix={npre} head_samp={nhs} alt_tail={nat}")
    # sampler_glue per rank
    for dev in range(nrank):
        out = build_glue(cd, device_id=dev)
        if out is None: continue
        with open(cd / f"sampler_glue.rank{dev}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--root", default="serve_exp/eval_serving_sharegpt_c500")
    ap.add_argument("--nrank", type=int, default=2)
    a = ap.parse_args()
    cd = ROOT / a.root / a.case
    print(f"Building sidecars for {cd.name} ({a.nrank} ranks)...")
    build_all(cd, a.nrank)
    print("done.")


if __name__ == "__main__":
    main()
