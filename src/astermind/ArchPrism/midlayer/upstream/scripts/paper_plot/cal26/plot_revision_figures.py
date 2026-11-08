#!/usr/bin/env python3
"""Render approved revision figures without replacing submitted figures."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from evaluation.serving.metrics.diagnostics.paper_revision_report import render


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-dir", type=Path, required=True)
    args = parser.parse_args()
    render(ROOT, args.figure_dir)
