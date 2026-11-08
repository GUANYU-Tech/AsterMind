#!/usr/bin/env bash
set -euo pipefail
task_dir=${MIDLAYER_ASCEND_TASK_ROOT:?set MIDLAYER_ASCEND_TASK_ROOT}
container_dir=${MIDLAYER_ASCEND_CONTAINER_TASK_ROOT:?set MIDLAYER_ASCEND_CONTAINER_TASK_ROOT}
tokenizer=${MIDLAYER_ASCEND_TOKENIZER:?set MIDLAYER_ASCEND_TOKENIZER}
timeout 180 docker exec \
  -e ASCEND_RT_VISIBLE_DEVICES=4 \
  -e MIDLAYER_ASCEND_CONTAINER_TASK_ROOT="$container_dir" \
  -e MIDLAYER_ASCEND_TOKENIZER="$tokenizer" \
  -w "$container_dir" \
  yxh_midlayer_serving_910b1 \
  sh -c 'PYTHONPATH="$PWD:$PYTHONPATH" python run_tiny_async.py' \
  > "$task_dir/run.log" 2>&1
