"""Repository-wide pytest bootstrap for native test dependencies."""

from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LRU_SOURCE = REPOSITORY_ROOT / "src/kernel/csrc/lru_sim.c"
LRU_LIBRARY = REPOSITORY_ROOT / "build/lru_sim.so"
CANONICAL_TEST_PYTHON = Path(
    "/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python").resolve()


def pytest_configure(config):
    """Build the local LRU helper before test-module collection.

    The output is ignored and worktree-local.  The controller builds it before
    xdist starts workers, so workers never race or borrow another checkout's
    binary.
    """

    if Path(sys.executable).resolve() != CANONICAL_TEST_PYTHON:
        raise pytest.UsageError(
            "Midlayer tests require the r2r_qwen3_next environment; run "
            "`python scripts/run_tests.py` instead of invoking pytest "
            f"with {sys.executable}")
    if hasattr(config, "workerinput"):
        return
    if (LRU_LIBRARY.is_file()
            and LRU_LIBRARY.stat().st_mtime_ns >= LRU_SOURCE.stat().st_mtime_ns):
        return
    LRU_LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "cc", "-O3", "-shared", "-fPIC", "-o", str(LRU_LIBRARY),
            str(LRU_SOURCE),
        ],
        check=True,
    )
