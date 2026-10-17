"""The CLI wires the dataset validator into a read-only experiment plan."""

import subprocess
import sys


def test_published_suite_resolves_without_running_or_generating_inputs():
    result = subprocess.run([
        sys.executable,
        "scripts/evaluation/serving/run_inferencex_deepseek_v3_8k1024_closedloop_comparison.py",
        "--validate-only",
    ], check=True, capture_output=True, text=True)
    assert "6 settings, 5 shared inputs, 30 selected runs" in result.stdout


def test_suite_rejects_an_unpublished_concurrency():
    result = subprocess.run([
        sys.executable,
        "scripts/evaluation/serving/run_inferencex_deepseek_v3_8k1024_closedloop_comparison.py",
        "--validate-only", "--concurrencies", "128",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "unsupported --concurrencies" in result.stderr


def test_paper_collection_validates_its_vllm_suite_selection(tmp_path):
    output = tmp_path / "unused"
    result = subprocess.run([
        sys.executable, "scripts/paper_plot/cal26/collect_data.py",
        "--validate-only", "--output-dir", str(output),
    ], check=True, capture_output=True, text=True)
    assert "20 selected runs" in result.stdout
    assert not output.exists()
