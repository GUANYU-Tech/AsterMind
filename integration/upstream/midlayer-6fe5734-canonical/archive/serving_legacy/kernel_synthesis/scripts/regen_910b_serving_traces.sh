#!/usr/bin/env bash
# (Re)capture the 910B ShareGPT serving eval traces in PLUGIN mode — real
# vLLM-Ascend on device with forward-hook operator capture. Analytic mode is
# NOT used (per-rank operator records are required for correct EP expert
# sharding, and only a real capture reflects the true operator sequence).
#
# Each case is captured against its msprof ground-truth workload (num_requests/
# seed read from the msprof meta so prediction and GT share one ShareGPT sample),
# then e2e_one_case_910b.py runs parse_msprof_timeline (GT) + predict + compare
# → comparison.json. run_full_eval --machine-config configs/machine/huawei_910b.yaml
# PART 2 then consumes these dirs.
#
# Requires: NPUs free + vllm/vllm_ascend importable. EP cases MUST use
# VLLM_WORKER_MULTIPROC_METHOD=spawn (fork trips a PyTorch OpenMP
# "Invalid thread pool" abort under enable_expert_parallel).
set -euo pipefail
ROOT=/mnt/gpfs/midlayer_yxh; cd "$ROOT"
PY=${PY:-/usr/local/python3.11.14/bin/python3}
EVAL=serve_exp/serve_exp_910b/eval_serving_sharegpt

# case  model  tp  ep  msprof_gt_dir  npu_ids
CASES=(
  "Qwen3-8B_TP1        Qwen3-8B       1 1 msprof_Qwen3-8B_TP1_aic   0"
  "Qwen3-32B_TP2       Qwen3-32B      2 1 msprof_Qwen3-32B_TP2      2,3"
  "Qwen3-30B-A3B_EP2   Qwen3-30B-A3B  2 2 msprof_Qwen3-30B-A3B_EP2  0,1"
  "Qwen3-30B-A3B_EP4   Qwen3-30B-A3B  4 4 msprof_Qwen3-30B-A3B_EP4  4,5,6,7"
)
for row in "${CASES[@]}"; do
  read -r case model tp ep gt npus <<< "$row"
  d="$EVAL/$case"
  echo "=== plugin capture $case (tp=$tp ep=$ep, NPU $npus) ==="
  rm -f "$d/plugin_trace.json" "$d/prediction.json" "$d/prediction_lut.json" \
        "$d"/.regen_cache_* "$d"/*operators* 2>/dev/null || true
  spawn=""; [ "$ep" -gt 1 ] && spawn="VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=8"
  env $spawn "$PY" scripts/serve_predict/e2e_one_case_910b.py \
    --case-id "$case" --model /mnt/gpfs/models/$model --tp "$tp" --ep "$ep" \
    --msprof-dir "serve_exp/serve_exp_910b/$gt" \
    --out-dir "$d" --gpu-ids "$npus" --max-tokens 1024
done
echo "Done. Now: $PY scripts/run_full_eval.py --machine-config configs/machine/huawei_910b.yaml"
