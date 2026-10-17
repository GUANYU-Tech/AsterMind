#!/usr/bin/env python3
"""Re-fit measured aiv_time (the physical per-pipe ground truth) with richer
physical features, LOO-validated. The model keeps aiv×disp; improving the aiv
regression directly helps. Prints LOO MAPE on aiv_time per structure per op."""
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
                   num_heads=a.get("num_heads",0), head_dim=a.get("head_dim",0),
                   num_kv=a.get("num_kv_heads",0), seq=a.get("seq_len",a.get("tok",1)))
        data.setdefault(op, []).append(rec)

def mape(p,t): return np.mean(np.abs(p-t)/t)*100

# physical feature: primary work dim + tok
def fdict(op, r):
    if op in ("rmsnorm","add_rmsnorm","elementwise"):
        tot=r["tok"]*r["hidden"]; prim=r["hidden"]; tok=r["tok"]
    elif op=="silu":
        tot=r["tok"]*r["inter"]; prim=r["inter"]; tok=r["tok"]
    elif op=="kvcache":
        tot=r["tok"]*r["num_kv"]*r["head_dim"]; prim=r["num_kv"]*r["head_dim"]; tok=r["tok"]
    elif op=="rope":
        tot=r["seq"]*r["num_heads"]*r["head_dim"]; prim=r["num_heads"]*r["head_dim"]; tok=r["seq"]
    return dict(tot=tot, prim=prim, tok=tok, bd=r["bd"], hid=r["hidden"],
               tot2=tot*tot*1e-9, prim2=prim*prim*1e-6, tok2=tok*tok*1e-3)

MODELS = {
    "a+b*tot+c*bd":         ["1","tot","bd"],
    "a+b*prim+b*tot+c*bd":  ["1","prim","tot","bd"],
    "a+b*prim+b*tot+d*tot2+e*bd":["1","prim","tot","tot2","bd"],
    "a+b*prim+b*tok2+b*tot+c*bd":["1","prim","tok2","tot","bd"],
    "a+b*prim+c*tok+e*bd":  ["1","prim","tok","bd"],
    "a+b*prim2+c*prim+d*tok+e*bd":["1","prim2","prim","tok","bd"],
}

for op in ["rmsnorm","add_rmsnorm","silu","elementwise","kvcache","rope"]:
    rows = data.get(op, [])
    if not rows: continue
    aiv = np.array([r["aiv"] for r in rows])
    print(f"\n===== {op} (n={len(rows)}) target=aiv_time — LOO MAPE =====")
    best=None
    for mname, cols in MODELS.items():
        X = np.ones((len(rows),len(cols)))
        for i,r in enumerate(rows):
            fd=fdict(op,r)
            for j,c in enumerate(cols): X[i,j]=eval(c,{},fd)
        errs=[]
        for i in range(len(rows)):
            m=np.ones(len(rows),bool); m[i]=False
            try: c,*_=np.linalg.lstsq(X[m],aiv[m],rcond=None)
            except: errs.append(np.nan); continue
            errs.append(abs(X[i]@c-aiv[i])/aiv[i]*100)
        loo=np.nanmean(errs)
        cfull,*_=np.linalg.lstsq(X,aiv,rcond=None)
        cf=dict(zip(cols,[float(x) for x in cfull]))
        flag=" <== BEST" if (best is None or loo<best[1]) else ""
        if best is None or loo<best[1]: best=(mname,loo,cols,cfull)
        print(f"  {mname:<34} LOO={loo:5.1f}%{flag}")
        # print coefs compactly
        print(f"       {cf}")
