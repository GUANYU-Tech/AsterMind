#!/usr/bin/env python3
"""Run the approved cache, latency and capacity sensitivity matrix."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from evaluation.serving.metrics.diagnostics.paper_revision import run_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    results = run_campaign(ROOT, args.jobs)
    for result in results:
        print(result, flush=True)
    return int(any(result["returncode"] != 0 for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
