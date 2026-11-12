#!/usr/bin/env bash
# One InferenceX-style GLM-5.2 1K/128 H200 simulation smoke run (concurrency 4).
set -euo pipefail

cd "$(dirname "$0")/../../.."
PYTHONHASHSEED=0 "${MIDLAYER_PYTHON:-python}" -c 'from evaluation.serving.simulation.simulation_runner import main; raise SystemExit(main())' \
  --canonical-requests serving_dataset/synthetic/inferencex_style/glm_5_2_h200_1k128_c4/requests.csv \
  --canonical-requests-sha256 a22431264a5e76189a1b5111e82174fcc1c16d54e413f715273b083240855e2f \
  --model configs/model/glm-5.2-fp8.json --model-config-sha256 fb2555ddea2a16f6aa9915a6d7a5acca1cd7dd8406255e66b65ab48805ca4f3d \
  --hardware-config configs/hardware/nvidia_h200.yaml --hardware-config-sha256 cb5f33773e1332d666267b4b10243331ba6a4acfaebabc638d62fbd35436291e \
  --deployment colocated --backend vllm --prediction-method roofline --collective-backend analytical_roofline_v1 \
  --launch-policy analytical_launch --attention-kv-layout mla --attention-kv-distribution uniform \
  --prefill-tp-size 8 --prefill-pp-size 1 --pipeline-ranks-per-node 8 --scheduler-pipeline-depth 1 \
  --max-batched-tokens 8192 --runtime-memory-policy configs/runtime_memory_policy/inferencex/glm_5_2_nvidia_h200_aggregate_tp8_runtime_memory.json \
  --kv-block-size-tokens 16 --max-model-len 9216 --expert-routing-policy seeded_assumption \
  --expert-routing-source-id inferencex-style-smoke-routing-v1 --expert-routing-seed 0 --max-requests 4 \
  --no-enable-prefix-caching --output experiment_logs/inferencex_style_1k128_c4/result.json > /dev/null
