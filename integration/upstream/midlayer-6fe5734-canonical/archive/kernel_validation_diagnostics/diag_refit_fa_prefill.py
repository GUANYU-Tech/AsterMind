#!/usr/bin/env python3
"""Re-fit FA contiguous/paged/paged_varlen prefill aiv_time (a+b*area+c*bd)
from MEASURED per-pipe aiv_time, LOO-validated. Current c is too high for
small-S (c*bd alone exceeds measured aiv)."""
import sys, os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)
from evaluation.kernel.dataset import load_config, load_kernel_dataset
from evaluation.kernel.profiler_observation import extract_ncu_metrics
import numpy as np

cfg = load_config("configs/kernel_dataset/ascend_910b.yaml"); raw = load_kernel_dataset(cfg)
data = {}
for k, v in raw.items():
    oi = v.get("op_info", {})
    if oi.get("Type") != "FLASHATTN": continue
    if "PREFILL" not in k or "CHUNKED" in k: continue
    a = oi["Attr"]
    Q = a.get("Q_LEN", 1); KV = a.get("KV_LEN", a.get("S"))
    if Q <= 1: continue
    lay = a.get("kv_layout", "contiguous")
    for kd in v["kernels"]:
        ncu = extract_ncu_metrics(kd)
        asc = kd.get("Ascend") or {}; r = asc.get("raw") or {}
        aiv = r.get("aiv_time(us)")
        if aiv is None: continue
        bd = asc.get("block_dim", 0) or 1
        area = Q * KV * a.get("HQ")
        data.setdefault(lay, []).append((area, int(bd), float(aiv), k))

def loo(area, bd, y, cols):
    X = np.ones((len(y), len(cols)))
    for i in range(len(y)):
        for j, c in enumerate(cols):
            X[i, j] = area[i] if c == "area" else bd[i] if c == "bd" else 1.0
    errs = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        try: cc, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        except: errs.append(np.nan); continue
        errs.append(abs(X[i]@cc - y[i])/y[i]*100)
    cc, *_ = np.linalg.lstsq(X, y, rcond=None)
    return np.nanmean(errs), dict(zip(cols, cc))

for lay in ["contiguous", "paged", "paged_varlen"]:
    rows = data.get(lay, [])
    if not rows: continue
    area = [r[0] for r in rows]; bd = [r[1] for r in rows]; y = [r[2] for r in rows]
    print(f"\n=== {lay} (n={len(rows)}) ===")
    for cols in [["1","area","bd"], ["1","area"], ["1","bd","area"]]:
        loo_mape, coefs = loo(area, bd, y, cols)
        # also full-fit MAPE
        X = np.ones((len(y), len(cols)))
        for i in range(len(y)):
            for j, c in enumerate(cols):
                X[i, j] = area[i] if c == "area" else bd[i] if c == "bd" else 1.0
        cc, *_ = np.linalg.lstsq(X, y, rcond=None)
        full = np.mean(np.abs(X@cc - y)/y)*100
        print(f"  {str(cols):<20} LOO={loo_mape:5.1f}% full={full:5.1f}% coefs={ {k:round(v,5) for k,v in coefs.items()} }")
