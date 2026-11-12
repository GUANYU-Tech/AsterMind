#!/bin/bash
# resume_gpus.sh <gpu>...: release the reservation from yield_gpus.sh and restart the recorded kernel queue.
set -u
K=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/kcampaign; cd $K
say() { echo "$(date '+%F %T') $*"; }
release() {  # release <gpu>: the reservation is `flock -x ... sleep 86400`; the sleep inherits the lock,
  # so stop the sleep first, then the flock, and verify the lock is really free
  local g=$1 r; r=$(cat logs/yield_gpu$g.reservation)
  kill $(ps -o pid= --ppid $r) $r 2>/dev/null
  for i in $(seq 1 10); do flock -n runs/gpu$g.proc.lock true && break; sleep 1; done
  flock -n runs/gpu$g.proc.lock true || { say "gpu$g: reservation lock still held; NOT resumed"; return 1; }
  mv logs/yield_gpu$g.reservation logs/yield_gpu$g.reservation.done
}
for g in "$@"; do
  [ -f logs/yield_gpu$g.reservation ] || { say "gpu$g: no reservation recorded"; continue; }
  if [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" != 0 ]; then
    say "gpu$g: the borrower still has a compute process; NOT resumed"; continue; fi
  release $g || continue
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('logs/yield_gpu$g.json')).get('retry_or_waiter') else 1)"; then
    say "gpu$g: released; it was running a retry/waiter sequence -> resume MANUALLY (see logs/yield_gpu$g.json)"; continue; fi
  read -r repo queue < <(python3 -c "import json; d=json.load(open('logs/yield_gpu$g.json')); print(d.get('repo','repo_v030'), ' '.join(d.get('resume_queue', [])))")
  if [ -z "$queue" ]; then say "gpu$g: released; nothing recorded to resume (was idle/serving)"; continue; fi
  first=${queue%% *}; rest=${queue#"$first"}; rest=${rest# }
  if [ "$first" = dsr1_mtpon_verify_mla_m2__v030 ]; then  # own snapshot for this plan only
    body="REPO=$K/repo_v030_mlaq ./launch.sh $g $first --windows 3 --concurrency 1 >> logs/${first}_gpu$g.log 2>&1"
    [ -n "$rest" ] && body="$body; exec ./chain.sh $g $rest"
  else
    body="exec ./chain.sh $g $queue"
  fi
  REPO=$K/repo_v030 PY=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/envs/vllm030/bin/python setsid nohup bash -c "$body" < /dev/null >> logs/chain_gpu${g}_resume.log 2>&1 &
  say "gpu$g: released and resumed: $body"
done
