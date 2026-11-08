#!/usr/bin/env python3
"""Run the published DSV3/H200 closed-loop input suite without rewriting it."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset_pipeline.serving_dataset.compact_case import load_compact_case
from evaluation.serving.experiments.inferencex_deepseek_v3 import main


if __name__ == "__main__":
    main(case_loader=load_compact_case)
