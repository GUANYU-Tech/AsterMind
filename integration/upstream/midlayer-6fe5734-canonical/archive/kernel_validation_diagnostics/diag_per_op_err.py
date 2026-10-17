#!/usr/bin/env python3
"""Dump per-op forward errors sorted worst-first, per bucket.
Reuses eval_kernel infra. For finding outliers to attack."""
import sys, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)

import scripts.kernel.validation.eval_kernel as E

cfg = E.load_config("configs/kernel_dataset/ascend_910b.yaml")
raw = E.load_kernel_dataset(cfg)
import os
hw_config_path = os.path.join(cfg["modeling"]["hw_config"], f"{cfg['modeling']['hw_name']}.yaml")

packed = [(k, v, hw_config_path, False, True, False, 0.0, False, False, None, None)
          for k, v in sorted(raw.items())]
with ThreadPoolExecutor(max_workers=64) as pool:
    results = list(pool.map(E._process_op_star, packed))

rows = []  # (task_key, op_type, attr, eqm, pred, actual, err%)
for r in results:
    if r is None:
        continue
    task_key, meta, _, fp, _, _, krs = r
    if fp is None:
        continue
    pred, actual = fp
    err = abs(pred-actual)/actual*100 if actual>0 else 0
    rows.append((task_key, meta["op_type"], meta["attr"], meta["eqm"], pred, actual, err, meta["range"]))

from collections import defaultdict
by_op = defaultdict(list)
for r in rows:
    by_op[r[1]].append(r)

print(f"{'op_type':<14} {'err%':>7} {'pred_us':>9} {'actual_us':>9} {'ratio':>6}  tok/shape")
for ot in ("GEMM","FUSEDMOE","FLASHATTN","NORM","ROTARY","ACTIVATION","ELEMENTWISE","KV_CACHE"):
    lst = sorted(by_op.get(ot,[]), key=lambda r:-r[6])
    print(f"\n===== {ot} (n={len(lst)}) worst 12 =====")
    for tk,otp,attr,eqm,pred,actual,err,rng in lst[:12]:
        a = attr
        def g(*ks):
            for k in ks:
                if k in a: return f"{k}={a[k]}"
            return ""
        shp = " ".join(s for s in [g("tok"),g("B"),g("M"),g("K"),g("N"),g("S"),g("D"),g("HQ"),
                                  g("H"),g("INT"),g("hidden"),g("intermediate"),g("element_count"),
                                  g("batch"),g("seq_len"),g("Q_LEN"),g("KV_LEN")] if s)
        print(f"{err:7.1f} {pred/1e3:9.1f} {actual/1e3:9.1f} {pred/actual:6.2f}  {shp}  [{rng}]")
