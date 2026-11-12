"""Assign pending kcampaign plans to GPUs so every GPU gets a similar amount of
remaining work and no plan is on two GPUs (runs on h200-tmp, stdlib only).

usage: balance_chains.py --gpus 0,1,...  --plans p1 p2 ...  [--priority p ...] [--suffix tag]
Remaining work = 3*cases - existing window_0[123].launch.json. Priority plans are
placed first (still one GPU each); the rest by longest-processing-time greedy.
Prints one `chain.sh` command per GPU (REPO/PY for vLLM 0.30 are set by the caller).
"""
import argparse, glob, json, os
K = os.path.join(
    os.environ["MIDLAYER_H200_PUBLIC_ROOT"], "h200_microbench/kcampaign")
ap = argparse.ArgumentParser()
ap.add_argument("--gpus", required=True)
ap.add_argument("--plans", nargs="+", required=True)
ap.add_argument("--priority", nargs="*", default=[])
ap.add_argument("--suffix", required=True)
a = ap.parse_args()
gpus = [int(g) for g in a.gpus.split(",")]
def remaining(p):
    n = len(json.load(open(f"{K}/plans/{p}.json"))["plans"])
    return 3 * n - len(glob.glob(f"{K}/runs/{p}/cases/*/window_0[123].launch.json"))
work = {p: remaining(p) for p in a.plans}
for p in a.priority:
    if p not in work:
        raise SystemExit(f"priority plan {p} not in --plans")
load = {g: 0 for g in gpus}; queue = {g: [] for g in gpus}
order = [p for p in a.priority] + sorted((p for p in a.plans if p not in a.priority), key=lambda p: -work[p])
for p in order:
    if work[p] == 0:
        continue
    g = min(gpus, key=lambda x: load[x])
    queue[g].append(p); load[g] += work[p]
for g in gpus:
    print(f"# gpu{g}: {load[g]} windows")
    if queue[g]:
        print(f"setsid nohup ./chain.sh {g} {' '.join(queue[g])} > logs/chain_gpu{g}_{a.suffix}.log 2>&1 < /dev/null &")
