#!/usr/bin/env bash
# Sweep H200 PD max-requests for the GLM models (thin wrapper of run_nvidia_h200_pd_sweep.py).
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
exec /home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python \
  scripts/evaluation/serving/run_nvidia_h200_pd_sweep.py --models glm "$@"
