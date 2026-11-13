#!/usr/bin/env python3
"""Render the explicit revision ablation figure."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.metrics.diagnostics.paper_revision_ablation import main

if __name__ == "__main__":
    main()
