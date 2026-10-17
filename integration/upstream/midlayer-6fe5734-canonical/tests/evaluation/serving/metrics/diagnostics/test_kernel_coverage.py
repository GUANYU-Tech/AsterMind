from pathlib import Path

import pytest

from evaluation.serving.metrics.diagnostics import metax_c500_kernel_coverage as coverage


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
CONFIG = "configs/evaluation/experiments/sharegpt_phase1.json"


@pytest.mark.parametrize(
    ("hardware_id", "cell_id"),
    (
        ("nvidia_a800", "nvidia_a800/qwen3-8b/tp1_qps64"),
        ("metax_c500", "metax_c500/qwen3-8b/tp1_qps64"),
        ("ascend_910b", "ascend_910b/qwen3-8b/tp1_qps64"),
    ),
)
def test_coverage_selection_supports_each_hardware(hardware_id, cell_id):
    arguments = coverage.parse_coverage_arguments((
        "--case", cell_id,
        "--run-id", "unit-test",
    ))

    selected = coverage.select_coverage_cells(
        REPOSITORY_ROOT, arguments, (CONFIG,), hardware_id)

    assert len(selected) == 1
    assert selected[0][1].cell_id == cell_id


@pytest.mark.parametrize(
    "hardware_id",
    ("nvidia_a800", "metax_c500", "ascend_910b"),
)
def test_canonical_timing_index_covers_four_families(hardware_id):
    index = coverage.load_canonical_timing_index(
        REPOSITORY_ROOT, hardware_id)

    assert set(index) == {"gemm", "flash_attn", "fused_moe", "vector"}
    assert len(index["gemm"]) == 221
    assert len(index["flash_attn"]) == 334
    assert len(index["fused_moe"]) == 176
    assert len(index["vector"]) == 296
