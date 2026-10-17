#!/usr/bin/env python3
"""Extract measured per-pipe aiv_time(us) + dur + shape for all 910b vector ops,
to re-fit physically-structured aiv_time models. aiv_time is measured hardware
ground truth (NOT the target duration) so regressing it vs shape is physical."""
import sys, os, json
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)
from evaluation.kernel.dataset import load_config, load_kernel_dataset
import numpy as np

cfg = load_config("configs/kernel_dataset/ascend_910b.yaml")
raw = load_kernel_dataset(cfg)

# op -> list of dict(tot, hidden, intermediate, tok, bd, aiv_us, dur_us, mte2_us, vec_us, scalar_us)
data = {}
for k, v in raw.items():
    oi = v.get("op_info", {})
    t = oi.get("Type")
    if t not in ("NORM","ROTARY","ACTIVATION","KV_CACHE","ELEMENTWISE"):
        continue
    a = oi["Attr"]
    op = a.get("op", t.lower())
    ks = v.get("kernels")
    if isinstance(ks, dict): ks = list(ks.values())
    for kd in ks:
        if not isinstance(kd, dict): continue
        asc = kd.get("Ascend") or {}
        r = asc.get("raw") or {}
        aiv = r.get("aiv_time(us)")
        dur = kd.get("Sections",{}).get("GPU Speed Of Light Throughput",{}).get("Header",{}).get("Duration [nsecond]")
        if aiv is None or dur is None: continue
        try:
            aiv=float(aiv); dur=float(dur)/1000.0
        except: continue
        bd = asc.get("block_dim") or r.get("Block Dim") or 1
        try: bd=float(bd)
        except: bd=1
        rec = dict(
            tok=a.get("tok",1), hidden=a.get("hidden",0), inter=a.get("intermediate",0),
            bd=int(bd), aiv_us=aiv, dur_us=dur,
            mte2=float(r.get("aiv_mte2_time(us)",0) or 0),
            vec=float(r.get("aiv_vec_time(us)",0) or 0),
            scal=float(r.get("aiv_scalar_time(us)",0) or 0),
            seq=a.get("seq_len",a.get("tok",1)), num_heads=a.get("num_heads",0),
            head_dim=a.get("head_dim",0), num_kv=a.get("num_kv_heads",0),
            elem=a.get("element_count",0), layer=a.get("layer",""),
        )
        data.setdefault(op, []).append(rec)

for op in sorted(data):
    rows = data[op]
    print(f"\n===== {op} (n={len(rows)}) =====")
    print(f"{'tok':>5} {'hid':>6} {'inter':>6} {'bd':>4} {'aiv':>8} {'dur':>8} {'d/a':>5} {'vec':>7} {'scal':>7} {'mte2':>7}")
    for r in sorted(rows, key=lambda x:(x["tok"],x["hidden"])):
        print(f"{r['tok']:>5} {r['hidden']:>6} {r['inter']:>6} {r['bd']:>4} {r['aiv_us']:>8.3f} {r['dur_us']:>8.3f} {r['dur_us']/r['aiv_us']:>5.2f} {r['vec']:>7.3f} {r['scal']:>7.3f} {r['mte2']:>7.3f}")
