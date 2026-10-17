#!/bin/bash
# Historical Phase 1 capture for the archived full-framework Serving path.
# Wraps the serving driver; --task-time=on gives the per-kernel timeline
# (ground truth, analogous to nsys on NV). Output parsed in a later step.
set -e
export ASCEND_RT_VISIBLE_DEVICES=0,1
cd /mnt/gpfs/midlayer_yxh
OUT=/mnt/gpfs/midlayer_yxh/serve_exp_910b_yxh/msprof_30ba3b_tp2_n4
rm -rf "$OUT"; mkdir -p "$OUT"

msprof --output="$OUT" \
  --task-time=on \
  --ascendcl=on \
  --model-execution=on \
  --application="python3 archive/serving_legacy/full_framework_capture/scripts/serve_driver_910b.py --model /mnt/gpfs/models/Qwen3-30B-A3B --tp 2 --num-requests 4 --max-model-len 8192 --meta-out $OUT/meta.json" \
  2>&1 | tail -25
echo "=MSPROF DONE="
echo "--- output tree ---"
find "$OUT" -maxdepth 3 -type d | head -20
echo "--- timeline/summary csvs ---"
find "$OUT" -name "*.csv" -o -name "*.json" 2>/dev/null | grep -iE "task|op_summary|timeline|kernel" | head
