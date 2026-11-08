#!/usr/bin/env python3
'Closed-loop colocated vs PD GLM-5.2 8K/1K sweeps with MTP off/on.\n\nSame zero-think-time closed-loop client contract and ProtoBridge scale-out\nPD handoff as the 1K/128 sweep; only the request-length targets change\n(8192/1024 with random_range_ratio 0.8, i.e. uniform [6553, 8192] input and\n[819, 1024] output tokens).\n\nUsage:\n  python .../run_inferencex_glm_8k1024_mtp5_closedloop_comparison.py       --prediction-method {roofline,midlayer} --tag <results subdir> [group:cN ...]\n\nWith no arguments all 32 points run.  SWEEP_WORKERS sets the subprocess\nparallelism (default 4); points whose outputs already exist are skipped, so\na memory probe phase (big points first) can be followed by a wide rerun.'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.inferencex_glm_8k1024_mtp5_closedloop import main

if __name__ == "__main__":
    main()
