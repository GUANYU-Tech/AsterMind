#!/usr/bin/env python3
"""Run a bounded H200 proxy sweep and render its metric-contract plots."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from evaluation.serving.experiments.nvidia_h200_pd_sweep import main

if __name__ == '__main__':
    raise SystemExit(main())
