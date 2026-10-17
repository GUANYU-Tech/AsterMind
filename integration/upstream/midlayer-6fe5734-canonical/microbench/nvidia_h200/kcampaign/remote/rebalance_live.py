"""Re-plan the pending kcampaign plans over all GPUs while the current plans keep running
(runs on h200-tmp, stdlib only; used when GPUs 4-7 join on 2026-09-27).

usage: rebalance_live.py --gpus 0,1,...,7 --order p1 p2 ... --split-windows N --suffix TAG [--write]

- Running plans (the `flock -x .../gpu<N>.proc.lock ... --plans plans/<p>.json` processes) stay
  where they are; their remaining windows are that GPU's starting load (a retry run, which has
  --case-ids, counts as 0).
- `--order` lists the pending plans in priority order (a plan that is running is rejected).
  Remaining work = 3 * cases - existing window_0[123].launch.json.
- A pending plan with more than N remaining windows is split into k = ceil(remaining / N)
  disjoint part plans `<base>_p<i>of<k>__v030` holding its not-yet-complete cases (cases that
  already have 3 windows in the original run directory stay there); each part is a separate plan
  file and run directory, so one plan still never runs on two GPUs. Part files carry
  `split_of` (original plan name and sha256, part index, excluded complete cases).
- Items are assigned in priority order, each to the GPU with the smallest load, so every GPU's
  queue keeps the priority order.
- Prints the processes that currently queue work (chain.sh parents and waiter shells; they hold
  no GPU context) for review, and one launch line per GPU: a waiter on the running plan's flock
  pid that then execs chain.sh, or chain.sh directly on an idle GPU. Nothing is started or killed.
  Part plan files are written only with --write.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import subprocess

K = "/mnt/public/yangxinhao/h200_microbench/kcampaign"
ap = argparse.ArgumentParser()
ap.add_argument("--gpus", required=True)
ap.add_argument("--order", nargs="+", required=True)
ap.add_argument("--split-windows", type=int, required=True)
ap.add_argument("--suffix", required=True)
ap.add_argument("--write", action="store_true")
ap.add_argument("--skip-running", action="store_true",
                help="drop --order plans that are running instead of failing")
ap.add_argument("--json", help="also write the assignment to this JSON file")
ap.add_argument("--base-load", nargs="*", default=[],
                help="gpu:windows of other work queued ahead on that GPU (e.g. a serving slot)")
a = ap.parse_args()
gpus = [int(g) for g in a.gpus.split(",")]


def plan_path(p):
    return f"{K}/plans/{p}.json"


def complete_cases(p):
    """Cases of plan p that already have their three regular windows."""
    cases = [f.split("/")[-2]
             for f in glob.glob(f"{K}/runs/{p}/cases/*/window_0[123].launch.json")]
    return {case for case in set(cases) if cases.count(case) == 3}


def measured_windows(p):
    return len(glob.glob(f"{K}/runs/{p}/cases/*/window_0[123].launch.json"))


def remaining(p):
    n = len(json.load(open(plan_path(p)))["plans"])
    return 3 * n - measured_windows(p)


ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True,
                    check=True).stdout.splitlines()[1:]
running = {}
queuers = []
for line in ps:
    pid, args = line.strip().split(" ", 1)
    m = re.match(r"flock -x \S*/gpu(\d+)\.proc\.lock .* --plans \S*/plans/(\S+)\.json", args)
    if m:
        gpu, plan = int(m.group(1)), m.group(2)
        if gpu in running:
            raise SystemExit(f"two running plans on gpu{gpu}: {running[gpu]} and {plan}")
        running[gpu] = (int(pid), plan, "--case-ids" in args)
        continue
    if re.search(r"chain\.sh \d+ ", args) or (
            args.startswith("bash -c ") and "chain.sh" in args):
        queuers.append((int(pid), args))
for gpu in running:
    if gpu not in gpus:
        raise SystemExit(f"gpu{gpu} runs {running[gpu][1]} but is not in --gpus")

load = {g: 0 for g in gpus}
for gpu, (pid, plan, is_retry) in running.items():
    load[gpu] = 0 if is_retry else remaining(plan)
running_plans = {plan for _, plan, _ in running.values()}
for spec in a.base_load:
    gpu, windows = (int(x) for x in spec.split(":"))
    if gpu not in load:
        raise SystemExit(f"--base-load gpu{gpu} is not in --gpus")
    load[gpu] += windows

items = []
for p in a.order:
    if p in running_plans:
        if not a.skip_running:
            raise SystemExit(f"{p} is running; it cannot be re-queued")
        print(f"# {p} is running; left where it is")
        continue
    r = remaining(p)
    if r == 0:
        continue
    if r <= a.split_windows:
        items.append((p, r))
        continue
    doc = json.load(open(plan_path(p)))
    done = complete_cases(p)
    todo = [c for c in doc["plans"] if c["case_id"] not in done]
    complete = [c["case_id"] for c in doc["plans"] if c["case_id"] in done]
    k = math.ceil(3 * len(todo) / a.split_windows)
    if not p.endswith("__v030"):
        raise SystemExit(f"{p}: only __v030 plans are split")
    base = p[:-len("__v030")]
    sha = hashlib.sha256(open(plan_path(p), "rb").read()).hexdigest()
    for i in range(k):
        part_cases = todo[i::k]
        name = f"{base}_p{i + 1}of{k}__v030"
        part = {key: value for key, value in doc.items() if key != "plans"}
        part["case_count"] = len(part_cases)
        part["split_of"] = {"plan": p, "plan_sha256": sha, "part": i + 1, "parts": k,
                            "complete_cases_left_in_original": complete}
        part["plans"] = part_cases
        if os.path.exists(plan_path(name)):
            existing = json.load(open(plan_path(name)))
            if existing != part:
                raise SystemExit(f"{plan_path(name)} exists with different content")
        elif a.write:
            with open(plan_path(name), "w") as f:
                json.dump(part, f, indent=1)
                f.write("\n")
        items.append((name, 3 * len(part_cases)))

queue = {g: [] for g in gpus}
for name, r in items:
    g = min(gpus, key=lambda x: load[x])
    queue[g].append(name)
    load[g] += r

print("# processes that queue work now (review, then kill these PIDs only):")
for pid, args in queuers:
    print(f"#   {pid}  {args[:150]}")
for g in gpus:
    run = f"running {running[g][1]} (flock pid {running[g][0]})" if g in running else "idle"
    print(f"# gpu{g}: {run}; total {load[g]} windows; queue {queue[g]}")
    if not queue[g]:
        continue
    chain = f"./chain.sh {g} {' '.join(queue[g])}"
    log = f"logs/chain_gpu{g}_{a.suffix}.log"
    if g in running:
        pid = running[g][0]
        print(f'setsid nohup bash -c "while kill -0 {pid} 2>/dev/null; do sleep 20; done; '
              f'exec {chain}" < /dev/null >> {log} 2>&1 &')
    else:
        print(f"setsid nohup {chain} < /dev/null >> {log} 2>&1 &")
if a.json:
    with open(a.json, "w") as f:
        json.dump({"gpus": gpus,
                   "running": {str(g): {"flock_pid": running[g][0], "plan": running[g][1]}
                               for g in running},
                   "queue": {str(g): queue[g] for g in gpus},
                   "load_windows": {str(g): load[g] for g in gpus},
                   "queuers": [{"pid": pid, "args": args} for pid, args in queuers]},
                  f, indent=1)
        f.write("\n")
if not a.write and any("_p" in name and "of" in name for name, _ in items):
    print("# dry run: part plan files not written (add --write)")
