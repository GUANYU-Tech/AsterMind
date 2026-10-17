#!/bin/bash
# Phase 2+3 batch: for each case in the 910B serving config, run msprof capture
# then the end-to-end eval, and collect comparison.json per case.
# msprof full-app capture is slow (~15min/case); run serially.
set -uo pipefail
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
cd /mnt/gpfs/midlayer_yxh
ROOT=/mnt/gpfs/midlayer_yxh
EXP=$ROOT/serve_exp_910b_yxh/eval_serving_sharegpt
HW=configs/hardware/huawei_910b.yaml
DS=/mnt/gpfs/dataset/ShareGPT_V3_unfiltered_cleaned_split.json
N=${N:-16}

# case: id model tp ep gpu_ids
run_case () {
  local id=$1 model=$2 tp=$3 ep=$4 gpus=$5
  local mdir=$ROOT/serve_exp_910b_yxh/msprof_${id}
  echo "======== CASE $id (tp=$tp ep=$ep gpus=$gpus N=$N) ========"
  if [ -n "$(find $mdir -name op_summary_*.csv 2>/dev/null | head -1)" ]; then
    echo "  [skip capture] op_summary exists"
  else
    rm -rf "$mdir"; mkdir -p "$mdir"
    ASCEND_RT_VISIBLE_DEVICES=$gpus msprof --output="$mdir" --task-time=on --aic-metrics=PipeUtilization --ascendcl=on \
      --application="python3 scripts/serve_trace/serve_driver_910b.py --model $model --tp $tp --ep $ep --num-requests $N --dataset $DS --max-model-len 8192 --meta-out $mdir/meta.json" \
      > $mdir/capture.log 2>&1
    echo "  [capture] $(grep -c DRIVER_OK $mdir/meta.json 2>/dev/null; echo done)"
  fi
  python3 scripts/serve_predict/e2e_one_case_910b.py \
    --case-id $id --model $model --tp $tp --ep $ep --hw-config $HW \
    --msprof-dir "$mdir" --out-dir $EXP/$id > $EXP/${id}_eval.log 2>&1
  grep -E "G1-|comparison" $EXP/${id}_eval.log | tail -3
}

# gpfs-resident models (config serving_eval_cases_sharegpt_910b.yaml).
run_case "Qwen3-8B_TP1"       /mnt/gpfs/models/Qwen3-8B      1 1 "0"
run_case "Qwen3-32B_TP2"      /mnt/gpfs/models/Qwen3-32B    2 1 "2,3"
run_case "Qwen3-30B-A3B_EP2"  /mnt/gpfs/models/Qwen3-30B-A3B 2 2 "0,1"
run_case "Qwen3-30B-A3B_EP4"  /mnt/gpfs/models/Qwen3-30B-A3B 4 4 "4,5,6,7"
echo "=BATCH DONE="
