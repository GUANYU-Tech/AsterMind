#!/usr/bin/env bash
# T5: fresh-regen honest baseline for the 5 remaining A800 MoE cases.
# Sequential (one at a time) to avoid memory thrash. Waits for the qps4 job
# to exit first. Cases already parked+cache-cleared (fresh regen forced).
set -u
cd /share_data/yangxinhao/chip_modeling/midlayer

# 1) wait for the running qps4 eval to finish
while pgrep -f "eval_serving_g1g3.py --case ShareGPT-Qwen3-30B-A3B_EP2_n128_qps4" >/dev/null 2>&1; do
  sleep 20
done
echo "[T5-runner] qps4 done, starting 5 remaining MoE cases"

CASES="EP2_n128_qps1 EP2_n128_qps16 EP4_n128_qps1 EP4_n128_qps4 EP4_n128_qps16"
for c in $CASES; do
  CASE="ShareGPT-Qwen3-30B-A3B_$c"
  echo "[T5-runner] === $CASE START ==="
  timeout 3600 python3 scripts/serve_predict/eval_serving_g1g3.py --case "$CASE" \
    > "/tmp/t5_${c}.log" 2>&1
  rc=$?
  # extract the result json line (last line of eval)
  res=$(grep -E '"G1_ape"' "/tmp/t5_${c}.log" | tail -1)
  echo "[T5-runner] === $CASE DONE rc=$rc :: ${res:-NO_RESULT} ==="
done
echo "[T5-runner] ALL 5 DONE"
