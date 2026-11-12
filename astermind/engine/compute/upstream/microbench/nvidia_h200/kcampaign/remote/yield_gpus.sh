#!/bin/bash
# yield_gpus.sh <gpu>...: hand GPUs to another owner at a safe window boundary (2026-09-27 GPU-sharing rule).
# Per GPU: record the running kernel plan + remaining chain queue (logs/yield_gpu<N>.json), wait until the
# current window's launch.json is written (max 180 s), stop chain parent / waiter shell / flock / runner,
# wait until the GPU has no compute process (3 checks), then hold runs/gpu<N>.proc.lock with a GPU-free
# `sleep` so none of our jobs start there while it is lent. Serving lanes are NOT stopped (reported instead).
set -u
K=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/kcampaign; cd $K
say() { echo "$(date '+%F %T') $*"; }
if pgrep -f "^(timeout \S+ )?bash \S*with_gpu_locks\.sh 0,1,2,3,4,5,6,7 " > /dev/null; then  # argv must start with it: a shell whose command line merely mentions the string does not count
  say "a whole-machine job (with_gpu_locks.sh on all 8 GPUs) is running; nothing yielded"; exit 3
fi
if [ $# -gt 1 ]; then  # several GPUs: handle each in parallel (each waits for its own window boundary)
  for g in "$@"; do bash "$0" "$g" & done; wait; exit 0
fi
for g in "$@"; do
  info=$(python3 - "$g" <<'PY'
import json, re, subprocess, sys
g = sys.argv[1]
rows = [l.strip().split(None, 2) for l in subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True,
        text=True, check=True).stdout.splitlines()[1:]]
rows = [(int(p), int(pp), a) for p, pp, a in rows]
lock = f"/gpu{g}.proc.lock"
holders = [(p, pp, a) for p, pp, a in rows if a.startswith("flock -x ") and lock in a.split(" ")[2]]
serving = [p for p, _, a in rows if a.startswith("flock -x ") and lock in a and "--plans" not in a]
for p, _, a in rows:
    m = re.search(r"run_nvidia_h200_sharegpt_capture\.py run .*--gpus (\S+)", a)
    if a.startswith("/usr/bin/python3 ") and m and g in m.group(1).split(","):
        serving.append(p)
if serving:
    print(json.dumps({"gpu": g, "kind": "serving_or_other", "pids": serving})); sys.exit()
if not holders:
    print(json.dumps({"gpu": g, "kind": "idle"})); sys.exit()
f, fpp, fa = holders[0]
plan = re.search(r"plans/(\S+)\.json", fa).group(1)
runner = [p for p, pp, _ in rows if pp == f]
# chain parent: the bash running chain.sh <g> (possibly an exec'd waiter), found by walking up from flock
parents, cur = [], fpp
by_pid = {p: (pp, a) for p, pp, a in rows}
while cur in by_pid and cur > 1:
    pp, a = by_pid[cur]
    if "chain.sh" in a or "retry.sh" in a or a.startswith("bash -c"):
        parents.append((cur, a))
    cur = pp
queue = []
for _, a in parents:
    m = re.search(r"chain\.sh " + g + r" ((?:\S+__v030 ?)+)", a)
    if m:
        plans = m.group(1).split()
        queue = plans[plans.index(plan):] if plan in plans else [plan] + plans
        break
repo = "repo_v030_mlaq" if "repo_v030_mlaq" in open(f"/proc/{f}/environ", "rb").read().decode(errors="ignore") else "repo_v030"
print(json.dumps({"gpu": g, "kind": "kernel", "plan": plan, "flock": f, "runner": runner,
                  "parents": parents, "resume_queue": queue or [plan], "repo": repo,
                  "retry_or_waiter": any("retry.sh" in a or "kill -0" in a for _, a in parents)}))
PY
)
  [ "${DRY:-0}" = 1 ] && { echo "$info" | cut -c1-400; continue; }
  echo "$info" > logs/yield_gpu$g.json
  kind=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['kind'])" "$info")
  if [ "$kind" = serving_or_other ]; then say "gpu$g: serving lane / other job holds the lock; NOT stopped: $info"; continue; fi
  if [ "$kind" = kernel ]; then
    plan=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['plan'])" "$info")
    n0=$(ls runs/$plan/cases/*/window_*.launch.json 2>/dev/null | wc -l)
    for i in $(seq 1 36); do [ "$(ls runs/$plan/cases/*/window_*.launch.json 2>/dev/null | wc -l)" -gt "$n0" ] && break; sleep 5; done
    pids=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(' '.join(str(p) for p,_ in d['parents'])+' '+str(d['flock'])+' '+' '.join(map(str,d['runner'])))" "$info")
    kill $pids 2>/dev/null
    say "gpu$g: stopped $plan at a window boundary (killed $pids)"
  fi
  n=0; for i in $(seq 1 60); do
    if [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; then n=$((n+1)); else n=0; fi
    [ $n -ge 3 ] && break; sleep 5; done
  [ $n -ge 3 ] || { say "gpu$g: still has a compute process after 5 min; NOT reserved"; continue; }
  setsid nohup flock -x runs/gpu$g.proc.lock sleep 86400 < /dev/null > /dev/null 2>&1 &
  sleep 1; rp=$(pgrep -f "^flock -x runs/gpu$g.proc.lock sleep 86400")
  echo "$rp" > logs/yield_gpu$g.reservation
  say "gpu$g: empty; our lock held by reservation pid $rp (resume_gpus.sh $g to take it back)"
done
