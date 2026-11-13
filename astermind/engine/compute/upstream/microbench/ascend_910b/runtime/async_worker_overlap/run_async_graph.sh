#!/usr/bin/env bash
set -euo pipefail

task_dir=${MIDLAYER_ASCEND_TASK_ROOT:?set MIDLAYER_ASCEND_TASK_ROOT}
container_dir=${MIDLAYER_ASCEND_CONTAINER_TASK_ROOT:?set MIDLAYER_ASCEND_CONTAINER_TASK_ROOT}
model_work_root=${MIDLAYER_ASCEND_MODEL_WORK_ROOT:?set MIDLAYER_ASCEND_MODEL_WORK_ROOT}
tokenizer=${MIDLAYER_ASCEND_TOKENIZER:?set MIDLAYER_ASCEND_TOKENIZER}
timeout 360 docker exec \
  -e ASCEND_RT_VISIBLE_DEVICES=5 \
  -e MIDLAYER_ASCEND_MODEL_WORK_ROOT="$model_work_root" \
  -e MIDLAYER_ASCEND_TOKENIZER="$tokenizer" \
  -e MID_WORKER_OVERLAP_GATE=1 \
  -e MID_WORKER_OVERLAP_GATE_NODES=256 \
  -e MID_WORKER_OVERLAP_GATE_MODE=eager_add \
  -e MID_WORKER_OVERLAP_OUTPUT_DIR=/tmp/midlayer_async_overlap_graph_eager \
  -w "$container_dir" \
  yxh_midlayer_serving_910b1 \
  bash -lc "PYTHONPATH=\"$container_dir/source:\$PYTHONPATH\" python $container_dir/source/run_async_graph.py" \
  > "$task_dir/run_async_graph_eager.log" 2>&1
