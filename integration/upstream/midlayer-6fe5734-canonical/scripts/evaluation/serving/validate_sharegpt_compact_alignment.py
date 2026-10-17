#!/usr/bin/env python3
"""Validate all compact ShareGPT cases and run structural alignment."""

import argparse
import json
import os
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from dataset_pipeline.serving_dataset import load_compact_case  # noqa: E402
from evaluation.serving.metrics.compact_alignment_suite import (  # noqa: E402
    run_compact_alignment_suite,
)
from evaluation.serving.metrics.sharegpt_compact_evaluation import (  # noqa: E402
    repository_source_commit,
)


MAX_JOBS = 64


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=int,
        choices=range(1, MAX_JOBS + 1),
        default=min(MAX_JOBS, os.cpu_count() or 1),
        help="case-level worker processes (default: up to 64)",
    )
    args = parser.parse_args()
    result = run_compact_alignment_suite(
        REPOSITORY_ROOT,
        load_compact_case,
        source_commit=repository_source_commit(REPOSITORY_ROOT),
        jobs=args.jobs,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if (
        result["alignment_checked"] is True
        and result["formal_conclusion_allowed"] is True
        and result["mismatched_count"] == 0
        and result["unsupported_count"] == 0
    ):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
