#!/usr/bin/env bash
# T5 EP4: fresh-regen honest baseline for the 3 EP4 MoE cases (4-rank, heavy).
# EP4 regen is ~4x EP2 → prior 3600s cap timed out (rc=124). Bump to 4h + -u
# (unbuffered) so regen progress is visible. Cases already parked+cache-cleared.
set -u
cd /share_data/yangxinhao/chip_modeling/midlayer

CASES="EP4_n128_qps1 EP4_n128_qps4 EP4_n128_qps16"
for c in $CASES; do
  CASE="ShareGPT-Qwen3-30B-A3B_$c"
  echo "[T5-EP4] === $CASE START ($(date +%H:%M:%S)) ==="
  timeout 14400 python3 -u scripts/serve_predict/eval_serving_g1g3.py --case "$CASE" \
    > "/tmp/t5_${c}.log" 2>&1
  rc=$?
  res=$(grep -E '"G1_ape"' "/tmp/t5_${c}.log" | tail -1)
  echo "[T5-EP4] === $CASE DONE rc=$rc ($(date +%H:%M:%S)) :: ${res:-NO_RESULT} ==="
done
echo "[T5-EP4] ALL 3 DONE"
