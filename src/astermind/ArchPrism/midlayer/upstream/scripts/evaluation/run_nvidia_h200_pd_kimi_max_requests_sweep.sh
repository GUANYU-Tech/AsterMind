#!/usr/bin/env bash
# Sweep H200 PD max-requests for the Kimi models (thin wrapper of run_nvidia_h200_pd_sweep.py).
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$root"
exec "${MIDLAYER_PYTHON:-python}" \
  scripts/evaluation/serving/run_nvidia_h200_pd_sweep.py --models kimi "$@"
