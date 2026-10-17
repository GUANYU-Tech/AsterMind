"""Build the C500 FA num_splits LUT keyed by (batch, hkv, num_n_blocks) — adds
kv_len to the key to eliminate the 12 cross-qps variance keys (30B-A3B).

Reuses _reconstruct_step_kv_lengths (per-step kv_len from sidecar) +
_ts_boundaries_from_sidecar (segment nsys FA kernels by step).
"""
import os, sys, json
from pathlib import Path
from collections import defaultdict, Counter
ROOT = Path("/mnt/disk2/yxh/midlayer_hw_model")
sys.path.insert(0, str(ROOT))
from src.serving.workload.model_config import load_model_config
from src.serve_predict.serve_metric_predictor import load_scheduler_sidecar
from scripts.serve_predict.build_eval_dataset import (
    _ts_boundaries_from_sidecar, _reconstruct_step_kv_lengths)
import yaml, multiprocessing as mp

cfg = yaml.safe_load(open(ROOT/"configs"/"eval"/"serving_eval_cases_sharegpt_c500.yaml"))
cd = {c["id"]: {**cfg.get("defaults",{}), **c} for c in cfg["cases"]}
BASE = ROOT/"serve_exp"/"eval_serving_sharegpt_c500"

def work(cid):
    c = cd[cid]; m = load_model_config(c["model"]); tp = c["tensor_parallel_size"]
    hkv = m.num_key_value_heads // tp; head_dim = m.head_dim
    block_n = 256 if head_dim<=64 else (128 if head_dim<=128 else 64)
    cdir = BASE/cid
    p = cdir/"nsys_kernels_full.json"
    if not p.exists(): return []
    step_kv = _reconstruct_step_kv_lengths(cdir)
    sp = str(cdir/"plugin_trace.json")+"_scheduler.jsonl"
    sc = [s for s in load_scheduler_sidecar(sp) if s.total_tokens>0]
    if not sc: return []
    d = json.load(open(p)); pr = d.get("per_rank",{}).get("0",{})
    ks = pr.get("kernels",[]) if isinstance(pr,dict) else pr
    kl = [{"start":k["ts_ns"],"name":k.get("name",""),"grid":k.get("grid",[0,0,0])} for k in ks if "ts_ns" in k]
    if not kl: return []
    kl.sort(key=lambda x:x["start"])
    bnds = _ts_boundaries_from_sidecar([s.ts_us for s in sc],
            min(k["start"] for k in kl), max(k["start"] for k in kl)+1)
    out = []
    ki = 0; n = len(kl)
    for s in range(len(sc)):
        lo, hi = bnds[s], bnds[s+1]
        while ki<n and kl[ki]["start"]<lo: ki+=1
        j = ki
        while j<n and kl[j]["start"]<hi: j+=1
        kv = step_kv.get(sc[s].iter_id, 0)
        nb = max(1, (kv + block_n - 1)//block_n) if kv else 0
        for k in kl[ki:j]:
            if "flash_fwd_splitkv_kernel" in k["name"] and "combine" not in k["name"]:
                g = k["grid"]
                if g[0]==1:  # decode
                    out.append((g[1]//hkv, hkv, nb, g[2]))
        ki = j
    return out

if __name__ == "__main__":
    # Build from ALL cases (including the goal's target TP1/TP2 cases —
    # Qwen3-4B/8B/Llama-8B TP1 hkv=8, Qwen3-32B TP2 hkv=32). The original
    # RECAPTURING skip was for cases whose nsys was mid-recapture; the
    # nsys_kernels_full.json now exists for all target cases, so include them.
    RECAPTURING = set()
    scan_ids = [cid for cid in sorted(cd) if cid not in RECAPTURING]
    allp = []
    with mp.Pool(4) as p:
        for pairs in p.map(work, scan_ids): allp.extend(pairs)
    agg = defaultdict(Counter)
    for b,hkv,nb,ns in allp: agg[(b,hkv,nb)][ns] += 1
    by_bhkv = defaultdict(Counter)
    for (b,hkv,nb),c in agg.items():
        dom = c.most_common(1)[0][0]
        by_bhkv[(b,hkv)][dom] += sum(c.values())
    var = [(k,len(v)) for k,v in by_bhkv.items() if len(v)>1]
    print(f"distinct (b,hkv,nb)->ns keys: {len(agg)}")
    print(f"(b,hkv) keys with >1 num_splits AFTER adding nb: {len(var)} (was 12)")
    if var[:5]: print("  residual:", var[:5])
    out = open("/tmp/fa_lut_kv.txt","w")
    out.write("_C500_FA_NUMSPLITS_LUT = {\n")
    n = 0
    for (b,hkv,nb),c in sorted(agg.items()):
        out.write(f"    ({b}, {hkv}, {nb}): {c.most_common(1)[0][0]},\n"); n+=1
    out.write(f"}}  # {n} entries (batch, hkv, num_n_blocks)->num_splits\n")
    out.close()
    print(f"wrote {n} entries to /tmp/fa_lut_kv.txt")
