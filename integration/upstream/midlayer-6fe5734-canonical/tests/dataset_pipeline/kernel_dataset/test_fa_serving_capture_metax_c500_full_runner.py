"""Focused contract for C500 layer-1 mcProfiler admission."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/kernel_dataset/fa_serving_capture_metax_c500_full_runner.py"
SPEC = importlib.util.spec_from_file_location("fa_serving_capture_c500_full", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_c500_layer1_check_rejects_whole_shape_and_period_rollups(tmp_path):
    report = tmp_path / "report.txt.json"
    period = tmp_path / "3_period0.txt.json"
    report.write_text("{}", encoding="utf-8")
    period.write_text("{}", encoding="utf-8")

    command = MODULE._perkernel_output_check_command(str(tmp_path))
    assert subprocess.run(["bash", "-lc", command], check=False).returncode == 1

    (tmp_path / "2_flash_fwd_splitkv_kernel.txt.json").write_text(
        "{}", encoding="utf-8")
    assert subprocess.run(["bash", "-lc", command], check=False).returncode == 0


def test_c500_layer1_attempt_count_has_a_strict_positive_contract():
    assert MODULE._positive_layer1_attempts(1) == 1
    assert MODULE._positive_layer1_attempts(5) == 5
    with pytest.raises(ValueError, match="positive"):
        MODULE._positive_layer1_attempts(0)


def test_c500_mcprofiler_timeout_has_a_strict_positive_contract():
    assert MODULE._positive_mcprofiler_timeout_seconds(600) == 600
    with pytest.raises(ValueError, match="positive"):
        MODULE._positive_mcprofiler_timeout_seconds(0)
    with pytest.raises(ValueError, match="positive"):
        MODULE._positive_mcprofiler_timeout_seconds(False)


def test_c500_profiler_attach_wait_has_a_strict_positive_contract():
    assert MODULE._positive_profiler_wait_seconds(15) == 15
    assert MODULE._positive_profiler_wait_seconds(60) == 60
    with pytest.raises(ValueError, match="positive"):
        MODULE._positive_profiler_wait_seconds(0)
    with pytest.raises(ValueError, match="positive"):
        MODULE._positive_profiler_wait_seconds(False)


def test_c500_explicit_profiler_headers_keep_all_candidate_evidence():
    required = sorted(MODULE._REQUIRED_MCPROFILER_HEADNAMES)

    assert set(MODULE._validated_mcprofiler_headnames(required)) == set(required)

    with pytest.raises(ValueError, match="required candidate evidence"):
        MODULE._validated_mcprofiler_headnames(required[:-1])
    with pytest.raises(ValueError, match="unknown"):
        MODULE._validated_mcprofiler_headnames([*required, "not a profiler header"])
    with pytest.raises(ValueError, match="unique"):
        MODULE._validated_mcprofiler_headnames([*required, required[0]])
