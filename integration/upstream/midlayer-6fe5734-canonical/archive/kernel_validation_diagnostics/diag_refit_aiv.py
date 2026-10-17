#!/usr/bin/env python3
"""Re-fit 910B vector-op aiv_time/dur models with physically-structured
regressions, validated by LOO MAPE on dur (the eval target). Uses measured
per-pipe aiv_time + per-op shape. NOT overfitting: structures are physical
(linear in work, linear in cores, quadratic for cross-core reduce, additive
launch overhead) and LOO is reported. Picks the best structure per op."""
import sys, os, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)
from evaluation.kernel.dataset import load_config, load_kernel_dataset
import numpy as np

cfg = load_config("configs/kernel_dataset/ascend_910b.yaml")
raw = load_kernel_dataset(cfg)

data = {}
for k, v in raw.items():
    oi = v.get("op_info", {})
    t = oi.get("Type")
    if t not in ("NORM","ROTARY","ACTIVATION","KV_CACHE","ELEMENTWISE"): continue
    a = oi["Attr"]; op = a.get("op", t.lower())
    ks = v.get("kernels")
    if isinstance(ks, dict): ks = list(ks.values())
    for kd in ks:
        if not isinstance(kd, dict): continue
        asc = kd.get("Ascend") or {}; r = asc.get("raw") or {}
        aiv = r.get("aiv_time(us)"); dur = kd.get("Sections",{}).get("GPU Speed Of Light Throughput",{}).get("Header",{}).get("Duration [nsecond]")
        if aiv is None or dur is None: continue
        try: aiv=float(aiv); dur=float(dur)/1000.0
        except: continue
        bd = asc.get("block_dim") or r.get("Block Dim") or 1
        try: bd=float(bd)
        except: bd=1
        rec = dict(tok=a.get("tok",1), hidden=a.get("hidden",0), inter=a.get("intermediate",0),
                   bd=int(bd), aiv=aiv, dur=dur,
                   mte2=float(r.get("aiv_mte2_time(us)",0) or 0),
                   num_heads=a.get("num_heads",0), head_dim=a.get("head_dim",0),
                   num_kv=a.get("num_kv_heads",0), seq=a.get("seq_len",a.get("tok",1)))
        data.setdefault(op, []).append(rec)

def mape(p, t): return np.mean(np.abs(p-t)/t)*100

# feature builders per op family
def feats_rmsnorm(r):
    tot = r["tok"]*r["hidden"]
    return dict(tot=tot, hid=r["hidden"], tok=r["tok"], bd=r["bd"])
def feats_silu(r):
    tot = r["tok"]*r["inter"]
    return dict(tot=tot, hid=r["inter"], tok=r["tok"], bd=r["bd"])
def feats_elem(r):
    # element_count not directly here; use tok*hidden as work proxy (matches dataset stamp)
    tot = r["tok"]*r["hidden"]
    return dict(tot=tot, hid=r["hidden"], tok=r["tok"], bd=r["bd"])
def feats_kvcache(r):
    tot = r["tok"]*r["num_kv"]*r["head_dim"]
    return dict(tot=tot, hid=r["num_kv"]*r["head_dim"], tok=r["tok"], bd=r["bd"])
def feats_rope(r):
    tot = r["seq"]*r["num_heads"]*r["head_dim"]
    return dict(tot=tot, hid=r["num_heads"]*r["head_dim"], tok=r["seq"], bd=r["bd"])

FEATS = {"rmsnorm":feats_rmsnorm,"add_rmsnorm":feats_rmsnorm,"silu":feats_silu,
         "elementwise":feats_elem,"kvcache":feats_kvcache,"rope":feats_rope}

# Model structures: name -> (design matrix builder, predict fn)
# All predict dur (us) directly. f = feature dict.
def design(rows, feats_fn, cols):
    X = np.ones((len(rows), len(cols)))
    for i, r in enumerate(rows):
        ff = feats_fn(r)
        for j, c in enumerate(cols):
            X[i, j] = eval(c, {}, ff) if isinstance(c,str) else c
    return X

MODELS = {
    "lin_tot_bd":      ["1", "tot", "bd"],
    "lin_tot_bd_sq":   ["1", "tot", "tot*tot*1e-9", "bd"],
    "lin_hid_tok_bd":  ["1", "hid", "tot", "bd"],
    "lin_hid_tok_bd_sq":["1", "hid", "tot", "tot*tot*1e-9", "bd"],
}

for op in ["rmsnorm","add_rmsnorm","silu","elementwise","kvcache","rope"]:
    rows = data.get(op, [])
    if not rows: continue
    f = FEATS[op]
    dur = np.array([r["dur"] for r in rows])
    print(f"\n===== {op} (n={len(rows)}) — LOO MAPE on dur =====")
    best = None
    for mname, cols in MODELS.items():
        X = design(rows, f, cols)
        # LOO
        errs = []
        coefs_all = None
        for i in range(len(rows)):
            mask = np.ones(len(rows), bool); mask[i]=False
            Xi, yi = X[mask], dur[mask]
            try:
                c, *_ = np.linalg.lstsq(Xi, yi, rcond=None)
            except: errs.append(np.nan); continue
            pred = X[i] @ c
            errs.append(abs(pred-dur[i])/dur[i]*100)
        loo = np.nanmean(errs)
        # full-fit coefs for inspection
        c_full, *_ = np.linalg.lstsq(X, dur, rcond=None)
        pred_full = X @ c_full
        mape_full = mape(pred_full, dur)
        flag = " <== BEST" if (best is None or loo < best[1]) else ""
        if best is None or loo < best[1]:
            best = (mname, loo, cols, c_full)
        print(f"  {mname:<22} LOO={loo:5.1f}%  full={mape_full:5.1f}%{flag}")
        print(f"       coefs={dict(zip(cols, [round(x,5) for x in c_full]))}")
