#!/usr/bin/env python3
'Compare GLM-5.2 1K/128 concurrency with MTP disabled/enabled.\n\nFor each C, simulate 2C warmup requests separately (discarded) and 10C\nmeasured requests. The serving model is stateless across simulation invocations;\nall requests in each canonical CSV arrive at t=0, with scheduler capacity C.\nThis preserves InferenceX request counts and concurrency while documenting the\nstatic-arrival approximation used by this simulator.'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation.serving.experiments.inferencex_style_glm_1k128_mtp_sweep import main

if __name__ == "__main__":
    main()
