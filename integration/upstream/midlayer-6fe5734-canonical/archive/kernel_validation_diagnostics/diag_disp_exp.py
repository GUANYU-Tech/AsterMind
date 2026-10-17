#!/usr/bin/env python3
"""Experiment: try different disp / overhead configs for vector ops,
measure per-op forward MAPE on the real eval (no commit)."""
import sys, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT)); os.chdir(PROJECT_ROOT)

import tools.kernel.validation.reference_models.norm as N
import scripts.kernel.validation.eval_kernel as E
import numpy as np

def run(label):
    cfg = E.load_config("configs/kernel_dataset/ascend_910b.yaml")
    raw = E.load_kernel_dataset(cfg)
    hw_config_path = os.path.join(cfg["modeling"]["hw_config"], f"{cfg['modeling']['hw_name']}.yaml")
    packed = [(k, v, hw_config_path, False, True, False, 0.0, False, False, None, None)
              for k, v in sorted(raw.items())]
    with ThreadPoolExecutor(max_workers=64) as pool:
        results = list(pool.map(E._process_op_star, packed))
    from collections import defaultdict
    by = defaultdict(list)
    for r in results:
        if r is None: continue
        tk, meta, _, fp, _, _, _ = r
        if fp is None: continue
        p, a = fp
        by[meta["op_type"]].append(abs(p-a)/a*100 if a>0 else 0)
    line = f"[{label}] "
    for ot in ("NORM","ROTARY","ACTIVATION","ELEMENTWISE","KV_CACHE"):
        v = by.get(ot)
        if v: line += f"{ot}={np.mean(v):.1f} "
    # ALL vector
    allv = sum(by.values(), [])
    line += f"| VEC={np.mean(allv):.1f}"
    print(line)

# baseline
N._VECTOR_AIVTIME_ORIG = dict(N._VECTOR_AIVTIME)
run("baseline")

# disp=1.0 everywhere
N._VECTOR_AIVTIME = {k: (v[0], v[1], v[2], 1.0) if len(v)==4 else (v[0],v[1],v[2],v[3],1.0)
                     for k,v in N._VECTOR_AIVTIME_ORIG.items()}
run("disp=1.0 all")
