#!/bin/bash
# usage: launch.sh <gpu> <plan_name> [extra nsys_campaign_runner args...]
# Canonical A800 protocol (nsys_nvtx_kernel_median, 3 windows, 1980 MHz gate);
# one flock per physical GPU so campaigns on different GPUs never share a lock.
set -e
W=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/wp9_kdataset
K=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/kcampaign
export HOME=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/h200_microbench/home
export XDG_CACHE_HOME=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache
export TRITON_CACHE_DIR=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/triton_kc_gpu$1
export VLLM_CACHE_ROOT=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/vllm_kc_gpu$1
export TORCHINDUCTOR_CACHE_DIR=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/cache/inductor_kc_gpu$1
export PYTHONPATH=$W/pydeps
export PATH=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/tools/cuda-12.9/bin:$PATH
export CUDA_HOME=${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/tools/cuda-12.9
GPU=$1; PLAN=$2; shift 2
R=${REPO:-$W/repo}
python3 $K/write_provenance.py $R $K/plans/$PLAN.json $K/runs/$PLAN $GPU
cd $R
exec ${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/envs/vllm024/bin/python scripts/kernel_dataset/nvidia_h200/nsys_campaign_runner.py \
  --plans $K/plans/$PLAN.json --output-dir $K/runs/$PLAN \
  --physical-gpu $GPU --nsys ${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/tools/nsys_2025.6.1/bin/nsys \
  --python ${MIDLAYER_H200_PUBLIC_ROOT:?set MIDLAYER_H200_PUBLIC_ROOT}/envs/vllm024/bin/python --gpu-lock $K/runs/gpu$GPU.lock "$@"
