#!/usr/bin/env python3
"""Run repository tests with the canonical local Python environment."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_TEST_ENVIRONMENT = "r2r_qwen3_next"


def main():
    conda_executable = (
        os.environ["CONDA_EXE"]
        if "CONDA_EXE" in os.environ
        else shutil.which("conda")
    )
    if conda_executable is None or not Path(conda_executable).is_file():
        raise SystemExit(
            "conda is unavailable; set CONDA_EXE or add conda to PATH "
            f"to run tests in {CANONICAL_TEST_ENVIRONMENT}"
        )
    environment_info = subprocess.run(
        [conda_executable, "info", "--envs", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    environment_prefixes = [
        Path(prefix)
        for prefix in json.loads(environment_info.stdout)["envs"]
        if Path(prefix).name == CANONICAL_TEST_ENVIRONMENT
    ]
    if len(environment_prefixes) != 1:
        raise SystemExit(
            f"expected one {CANONICAL_TEST_ENVIRONMENT} Conda environment, "
            f"found {len(environment_prefixes)}"
        )
    canonical_test_python = environment_prefixes[0] / "bin/python"
    if not canonical_test_python.is_file():
        raise SystemExit(f"canonical test Python is unavailable: {canonical_test_python}")
    arguments = sys.argv[1:] if len(sys.argv) > 1 else ["-q"]
    os.chdir(REPOSITORY_ROOT)
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = "0"
    # Pytest already provides process-level parallelism. Keep native math
    # libraries in each worker/child from multiplying that concurrency.
    # Explicit caller settings always win; the parent shell is untouched.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if name not in os.environ:
            os.environ[name] = "1"
    os.execv(
        canonical_test_python,
        [str(canonical_test_python), "-m", "pytest", *arguments],
    )


if __name__ == "__main__":
    main()
