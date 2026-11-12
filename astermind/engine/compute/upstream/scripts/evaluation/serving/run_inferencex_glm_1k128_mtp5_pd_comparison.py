#!/usr/bin/env python3
'Compare colocated and PD GLM-5.2 1K/128 sweeps with MTP off/on.'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.experiments.inferencex_glm_1k128_mtp5_pd import main

if __name__ == "__main__":
    main()
