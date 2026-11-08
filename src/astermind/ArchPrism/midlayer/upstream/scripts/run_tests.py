#!/usr/bin/env python3
"""Run repository tests with the canonical local Python environment."""

import os
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_TEST_PYTHON = Path(
    "/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python")


def main():
    if not CANONICAL_TEST_PYTHON.is_file():
        raise SystemExit(
            "canonical test Python is unavailable: "
            f"{CANONICAL_TEST_PYTHON}")
    arguments = sys.argv[1:] if len(sys.argv) > 1 else ["-q"]
    os.chdir(REPOSITORY_ROOT)
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = "0"
    os.execv(
        CANONICAL_TEST_PYTHON,
        [str(CANONICAL_TEST_PYTHON), "-m", "pytest", *arguments],
    )


if __name__ == "__main__":
    main()
