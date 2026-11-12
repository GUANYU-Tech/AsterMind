#!/usr/bin/env bash
set -euo pipefail

task_dir=${MIDLAYER_ASCEND_TASK_ROOT:?set MIDLAYER_ASCEND_TASK_ROOT}
container_dir=${MIDLAYER_ASCEND_CONTAINER_TASK_ROOT:?set MIDLAYER_ASCEND_CONTAINER_TASK_ROOT}
wake_runner=${MIDLAYER_ASCEND_ASYNC_WAKE_RUNNER:?set MIDLAYER_ASCEND_ASYNC_WAKE_RUNNER}
tokenizer=${MIDLAYER_ASCEND_TOKENIZER:?set MIDLAYER_ASCEND_TOKENIZER}
timeout 240 docker exec \
  -e ASCEND_RT_VISIBLE_DEVICES=5 \
  -e MIDLAYER_ASCEND_CONTAINER_TASK_ROOT="$container_dir" \
  -e MIDLAYER_ASCEND_TOKENIZER="$tokenizer" \
  -e MID_WORKER_OVERLAP_GATE=1 \
  -e MID_WORKER_OVERLAP_GATE_NODES=512 \
  -e MID_WORKER_OVERLAP_GATE_MODE=graph \
  -e MID_WORKER_OVERLAP_OUTPUT_DIR=/tmp/midlayer_async_overlap_gate3 \
  -w "$container_dir" \
  yxh_midlayer_serving_910b1 \
  bash -lc "PYTHONPATH=\"$container_dir/source:\$PYTHONPATH\" python $wake_runner" \
  > "$task_dir/run_gate3.log" 2>&1
