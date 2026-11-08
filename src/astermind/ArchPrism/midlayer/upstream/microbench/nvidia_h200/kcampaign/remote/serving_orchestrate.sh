#!/bin/bash
# 2026-09-26 serving slot on GPUs 1-3: per lane, vLLM 0.30 then 0.11 ShareGPT capture, then the
# first retry batch (lane 1-2 only) and the lane's kernel chains resume. GPU 0 is not touched.
# Every GPU stage in the serving runner and in launch.sh holds gpu<N>.proc.lock (one process per GPU).
K=/mnt/public/yangxinhao/h200_microbench/kcampaign
C=/mnt/public/yangxinhao/src/midlayer_h200_serving_capture/current/scripts/experiments/run_nvidia_h200_sharegpt_capture.py
L=/mnt/public/yangxinhao/serving_capture
TP2="--case qwen3-32b/tp2_ep1_qps1 --case qwen3-30b-a3b/tp2_ep2_qps1 --case qwen3-32b/tp2_ep1_qps4 --case qwen3-30b-a3b/tp2_ep2_qps4 --case qwen3-32b/tp2_ep1_qps16 --case qwen3-30b-a3b/tp2_ep2_qps16"
TP1="--case qwen3-4b/tp1_qps4 --case qwen3-4b/tp1_qps16 --case qwen3-4b/tp1_qps64 --case llama-3.1-8b/tp1_qps4 --case llama-3.1-8b/tp1_qps16 --case llama-3.1-8b/tp1_qps64 --case qwen3-8b/tp1_qps4 --case qwen3-8b/tp1_qps16 --case qwen3-8b/tp1_qps64"
say() { echo "$(date '+%F %T') $*"; }
lane12() {
  for v in 0.30.0 0.11.0; do
    mkdir -p $L/$v/lanes; say "lane gpu1-2 START serving $v"
    /usr/bin/python3 $C run --runtime-version $v --gpus 1,2 $TP2 > $L/$v/lanes/gpu1-2.log 2>&1 < /dev/null
    say "lane gpu1-2 END serving $v rc=$?"
  done
  export REPO=$K/repo_v030 PY=/mnt/public/yangxinhao/envs/vllm030/bin/python
  say "lane gpu1-2 START retry r1"
  $K/retry.sh 1 4,5,6 kc_kda_models__v030:$K/plans/retry_kc_kda_models__v030_r1.txt kc_kda_k3_native__v030:$K/plans/retry_kc_kda_k3_native__v030_r1.txt &
  $K/retry.sh 2 4,5,6 kc_v4attn_models__v030:$K/plans/retry_kc_v4attn_models__v030_r1.txt kc_kda_k3_gated_mla__v030:$K/plans/retry_kc_kda_k3_gated_mla__v030_r1.txt &
  wait; say "lane gpu1-2 END retry r1; resuming kernel chains 1 and 2"
  cd $K
  ./chain.sh 1 kc_dsa_cache_write__v030 kc_dsa_fused_norm_rope__v030 kc_gemm_bf16_dense_decode_b__v030 > logs/chain_gpu1_v030d.log 2>&1 < /dev/null &
  ./chain.sh 2 kc_dsa_indexer__v030 wp9_vector__v030 kc_vector_models__v030 kc_moe_bf16_models__v030 kc_gemm_bf16_dense_decode_a__v030 > logs/chain_gpu2_v030d.log 2>&1 < /dev/null &
  wait
}
lane3() {
  for v in 0.30.0 0.11.0; do
    mkdir -p $L/$v/lanes; say "lane gpu3 START serving $v"
    /usr/bin/python3 $C run --runtime-version $v --gpus 3 $TP1 > $L/$v/lanes/gpu3.log 2>&1 < /dev/null
    say "lane gpu3 END serving $v rc=$?"
  done
  export REPO=$K/repo_v030 PY=/mnt/public/yangxinhao/envs/vllm030/bin/python
  say "lane gpu3 resuming kernel chain 3"
  cd $K
  ./chain.sh 3 kc_dsa_sparse_mla_decode__v030 wp9_flash_attn__v030 kc_fa_paged_serving_models__v030 kc_gemm_bf16_mla_models__v030 dsr1_decode_phase1__v030 dsr1_mtpoff_fa3__v030 > logs/chain_gpu3_v030d.log 2>&1 < /dev/null
}
lane12 > $K/logs/serving_lane12.log 2>&1 &
lane3 > $K/logs/serving_lane3.log 2>&1 &
wait
