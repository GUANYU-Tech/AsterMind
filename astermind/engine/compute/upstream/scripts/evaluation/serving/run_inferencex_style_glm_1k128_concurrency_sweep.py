#!/usr/bin/env python3
'Sweep first-wave client concurrency for the 1K/128 GLM-5.2 H200 smoke setup.\n\nEach point submits C requests at t=0.  It deliberately does not emulate later\nclosed-loop replenishment; use it to validate and visualize scheduler behavior.'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation.serving.experiments.inferencex_style_glm_1k128_concurrency_sweep import main

if __name__ == "__main__":
    main()
