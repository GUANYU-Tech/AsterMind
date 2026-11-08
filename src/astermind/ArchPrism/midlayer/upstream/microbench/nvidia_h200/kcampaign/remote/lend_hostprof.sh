#!/bin/bash
# H200 host profiles for the ShareGPT serving evaluation (host launch, scheduler prefill,
# runtime/collective prepare, request path, NCCL 2-rank sweep). usage: lend_hostprof.sh <snapshot> <vllm env> <gpuA> <gpuB>
# (22:18 run: a132507d8 + vllm011; 23:30 run: e6a79951a + vllm030, the preferred H200 truth version). The host-side
# steps are CPU-timing sensitive, so ALL 8 GPUs' kernel queues are stopped at a window boundary
# (same live-queue record/resume as lend_fusion2048.sh), the profile script runs on <gpuA>+<gpuB> with our
# reservations holding the locks (NO_FLOCK=1), then every recorded queue is re-run unchanged.
K=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/kcampaign; cd $K
SNAPSHOT=$1; VLLM_ENV=$2; GA=$3; GB=$4; TAG=hostprof_${VLLM_ENV}_$(date +%m%d_%H%M)  # fresh state dir per run
case "$GA$GB" in ([0-7][0-7]) ;; (*) echo "usage: lend_hostprof.sh <snapshot> <vllm env> <gpuA> <gpuB>"; exit 2;; esac
[ -f $SNAPSHOT/SOURCE_COMMIT ] || { echo "no snapshot $SNAPSHOT"; exit 2; }
say() { echo "$(date '+%F %T') $*"; }
tree() { local p=$1; echo $p; for c in $(ps -o pid= --ppid $p); do tree $c; done; }
declare -A TOP
for p in $(ps -eo pid,ppid | awk '$2==1 {print $1}'); do
  a=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null)
  g=$(echo "$a" | grep -oE '(chain|launch|retry)\.sh [0-7] |resume_gpu[0-7]\.sh' | head -1 | grep -oE '[0-7]')
  [ -n "$g" ] || continue
  TOP[$g]="${TOP[$g]:-} $p"  # several (a queue plus a waiter queued after it): resumed in pid order
done
for g in 0 1 2 3 4 5 6 7; do [ -n "${TOP[$g]:-}" ] || { say "gpu$g: no top-level queue found; nothing lent"; exit 2; }; done
mkdir -p logs/$TAG
for g in 0 1 2 3 4 5 6 7; do
  { echo "cd $K"
    for p in $(echo ${TOP[$g]} | tr ' ' '\n' | sort -n); do
      echo "( cd $(readlink /proc/$p/cwd)"
      tr '\0' '\n' < /proc/$p/environ | grep -E '^(REPO|PY)=' | sed 's/^/  export /'
      printf ' '; while IFS= read -r -d '' x; do printf ' %q' "$x"; done < /proc/$p/cmdline; echo
      echo ")"
    done
  } > logs/$TAG/resume_gpu$g.sh
  # the queues now run one after another, so drop their "wait for pid N" prefixes (N may be reused)
  sed -i -E 's/while\\ kill\\ -0\\ [0-9]+\\ 2\\>\/dev\/null\\;\\ do\\ sleep\\ 20\\;\\ done\\;\\ //' logs/$TAG/resume_gpu$g.sh
done
say "recorded queues: $(for g in 0 1 2 3 4 5 6 7; do echo -n "gpu$g=${TOP[$g]} "; done)"
stop() {
  local g=$1 top F plan n0
  F=$(ps -eo pid,args | awk -v l="gpu$g.proc.lock" '$2=="flock" && $4 ~ l"$" {print $1}')
  if [ -n "$F" ]; then
    plan=$(ps -o args= -p $F | grep -oE "plans/[a-z0-9_]+\.json" | sed 's#plans/##; s#\.json##')
    n0=$(ls runs/$plan/cases/*/window_*.launch.json 2>/dev/null | wc -l)
    for i in $(seq 1 60); do [ "$(ls runs/$plan/cases/*/window_*.launch.json 2>/dev/null | wc -l)" -gt "$n0" ] && break; sleep 3; done
  fi
  for top in ${TOP[$g]}; do kill $(tree $top) 2>/dev/null; done
  local n=0; for i in $(seq 1 60); do
    if [ "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)" = 0 ]; then n=$((n+1)); else n=0; fi
    [ $n -ge 3 ] && break; sleep 3; done
  [ $n -ge 3 ] || { say "gpu$g not empty"; return 1; }
  setsid nohup flock -x runs/gpu$g.proc.lock sleep 86400 < /dev/null > /dev/null 2>&1 &
  sleep 1; pgrep -f "^flock -x runs/gpu$g.proc.lock sleep 86400" > logs/$TAG/reservation_gpu$g
  say "gpu$g: stopped (${plan:-idle}) at a window boundary; reserved by $(cat logs/$TAG/reservation_gpu$g)"
}
for g in 0 1 2 3 4 5 6 7; do stop $g & done; wait
for g in 0 1 2 3 4 5 6 7; do [ -s logs/$TAG/reservation_gpu$g ] || { say "gpu$g NOT reserved"; FAIL=1; }; done
if [ -z "${FAIL:-}" ]; then
  say "all 8 reserved; host profiles start"
  NO_FLOCK=1 VLLM_ENV=$VLLM_ENV timeout 2700 bash $SNAPSHOT/microbench/nvidia_h200/runtime/host_profiles/run_nvidia_h200_host_profiles.sh $GA $GB >> logs/$TAG/run.log 2>&1
  say "HOST PROFILES rc=$? (logs/$TAG/run.log)"
fi
left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')
if [ -n "$left" ]; then
  for p in $left; do say "borrower pid $p still on a GPU: $(ps -o args= -p $p | cut -c1-150)"; done
  kill $left 2>/dev/null; sleep 15
  left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' '); [ -n "$left" ] && kill -9 $left 2>/dev/null
fi
for i in $(seq 1 30); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 2; done
for g in 0 1 2 3 4 5 6 7; do
  if [ -s logs/$TAG/reservation_gpu$g ]; then
    r=$(cat logs/$TAG/reservation_gpu$g); kill $(ps -o pid= --ppid $r) $r 2>/dev/null
  fi
  for i in $(seq 1 10); do flock -n runs/gpu$g.proc.lock true && break; sleep 1; done
  flock -n runs/gpu$g.proc.lock true || { say "gpu$g: lock still held; NOT resumed"; continue; }
  setsid nohup bash logs/$TAG/resume_gpu$g.sh < /dev/null >> logs/chain_gpu${g}_$TAG.log 2>&1 &
  say "gpu$g resumed: $(grep -c "^(" logs/$TAG/resume_gpu$g.sh) queue(s)"
done
say "HOSTPROF RESUME DONE"
