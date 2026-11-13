#!/usr/bin/env python3
"""Run one explicitly selected revision diagnostic."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.metrics.diagnostics.paper_revision import run_study
from dataset_pipeline.serving_dataset import load_compact_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--kind", required=True, choices=(
        "baseline", "kernel_scale", "l2_scale", "dram_bandwidth_scale",
        "l2_bandwidth_scale", "reserve", "ladder", "padding"))
    parser.add_argument("--factor", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run_study(
        ROOT, args.case, args.kind, args.factor, args.output.resolve(),
        load_compact_case)


if __name__ == "__main__":
    main()
