#!/bin/bash
# usage: launch.sh <gpu> <plan_name> [extra nsys_campaign_runner args...]
# Canonical A800 protocol (nsys_nvtx_kernel_median, 3 windows, 1980 MHz gate);
# one flock per physical GPU so campaigns on different GPUs never share a lock.
set -e
W=/mnt/public/yangxinhao/h200_microbench/wp9_kdataset
K=/mnt/public/yangxinhao/h200_microbench/kcampaign
export HOME=/mnt/public/yangxinhao/h200_microbench/home
export XDG_CACHE_HOME=/mnt/public/yangxinhao/cache
export TRITON_CACHE_DIR=/mnt/public/yangxinhao/cache/triton_kc_gpu$1
export VLLM_CACHE_ROOT=/mnt/public/yangxinhao/cache/vllm_kc_gpu$1
export TORCHINDUCTOR_CACHE_DIR=/mnt/public/yangxinhao/cache/inductor_kc_gpu$1
export PYTHONPATH=$W/pydeps
export PATH=/mnt/public/yangxinhao/tools/cuda-12.9/bin:$PATH
export CUDA_HOME=/mnt/public/yangxinhao/tools/cuda-12.9
GPU=$1; PLAN=$2; shift 2
R=${REPO:-$W/repo}
python3 $K/write_provenance.py $R $K/plans/$PLAN.json $K/runs/$PLAN $GPU
cd $R
exec /mnt/public/yangxinhao/envs/vllm024/bin/python scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py \
  --plans $K/plans/$PLAN.json --output-dir $K/runs/$PLAN \
  --physical-gpu $GPU --nsys /mnt/public/yangxinhao/tools/nsys_2025.6.1/bin/nsys \
  --python /mnt/public/yangxinhao/envs/vllm024/bin/python --gpu-lock $K/runs/gpu$GPU.lock "$@"
