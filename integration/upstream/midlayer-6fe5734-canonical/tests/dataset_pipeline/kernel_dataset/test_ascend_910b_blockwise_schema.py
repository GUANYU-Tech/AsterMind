"""Contracts for the 910B converter's ②③ schema retention: per-block tables
(`Ascend.invocations[i].blocks`) and per-invocation retention
(`Ascend.invocations`), on top of the pre-existing ① aggregate scalars.

Fixtures under fixtures/ are real msprof CSV captures (CSV-only, no dump/*.bin)
copied from a verified-matching-canonical raw case for each op_type shape:
gemm_sample (cube, 3 invocations), flashattn_sample (mix, 3 invocations),
groupedmatmul_sample (mix, 12 invocations split w1/w2 by GM_to_L1 volume).
"""

from __future__ import annotations

import os

import pytest

from dataset_pipeline.kernel_dataset.ascend_910b import (
    _aggregate_parsed,
    _kernel_to_ncu_format,
    _load_hw_clocks,
    _parse_kernel_dir,
    _parse_kernel_invocation,
    _parse_timeline_kernel,
    _read_csv_rows,
    _rows_to_block_table,
)
from dataset_pipeline.kernel_dataset.config import load_dataset_config

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
GEMM_DIR = os.path.join(FIXTURES, "gemm_sample")
FLASHATTN_DIR = os.path.join(FIXTURES, "flashattn_sample")
GROUPEDMATMUL_DIR = os.path.join(FIXTURES, "groupedmatmul_sample")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


@pytest.fixture(scope="module")
def hw_clocks():
    cwd = os.getcwd()
    os.chdir(_REPO_ROOT)
    try:
        cfg = load_dataset_config("configs/dataset/ascend_910b.yaml")
        return _load_hw_clocks(cfg)
    finally:
        os.chdir(cwd)


def test_rows_to_block_table_preserves_literal_cells():
    rows = [
        {"block_id": "0", "sub_block_id": "0", "aic_cube_ratio": "12.5"},
        {"block_id": "0", "sub_block_id": "1", "aic_cube_ratio": "NA"},
    ]
    table = _rows_to_block_table(rows)
    assert table["columns"] == ["block_id", "sub_block_id", "aic_cube_ratio"]
    assert table["rows"] == [["0", "0", "12.5"], ["0", "1", "NA"]]


def test_rows_to_block_table_rejects_inconsistent_columns():
    rows = [
        {"block_id": "0", "sub_block_id": "0"},
        {"block_id": "0", "other": "1"},
    ]
    with pytest.raises(ValueError):
        _rows_to_block_table(rows)


@pytest.mark.parametrize("kernel_dir,expected_op_type", [
    (GEMM_DIR, "cube"),
    (FLASHATTN_DIR, "mix"),
])
def test_parse_kernel_invocation_blocks_present_for_all_four_csvs(
        kernel_dir, expected_op_type):
    inv0 = _parse_kernel_invocation(os.path.join(kernel_dir, "0"))
    assert inv0["op_type"] == expected_op_type
    blocks = inv0["blocks"]
    assert set(blocks) == {
        "ArithmeticUtilization", "L2Cache", "Memory", "PipeUtilization"}
    for csv_name, table in blocks.items():
        assert len(table["rows"]) > 0, csv_name
        assert all(len(row) == len(table["columns"]) for row in table["rows"])


@pytest.mark.parametrize("kernel_dir", [GEMM_DIR, FLASHATTN_DIR])
def test_parse_kernel_invocation_blocks_round_trip_raw_csv(kernel_dir):
    inv_dir = os.path.join(kernel_dir, "0")
    import glob
    memory_csv = glob.glob(os.path.join(inv_dir, "Memory_*.csv"))[0]
    fresh_rows = _read_csv_rows(memory_csv)

    parsed = _parse_kernel_invocation(inv_dir)
    stored = parsed["blocks"]["Memory"]

    assert stored["columns"] == list(fresh_rows[0].keys())
    assert len(stored["rows"]) == len(fresh_rows)
    for stored_row, fresh_row in zip(stored["rows"], fresh_rows):
        assert stored_row == [fresh_row[c] for c in stored["columns"]]


def test_aggregate_parsed_carries_forward_every_invocation():
    inv_dirs = sorted(
        os.path.join(GEMM_DIR, d) for d in os.listdir(GEMM_DIR) if d.isdigit())
    parsed = [_parse_kernel_invocation(d) for d in inv_dirs]
    agg = _aggregate_parsed(parsed, "test_op", "cube")

    assert agg["invocations"] == parsed
    assert len(agg["invocations"]) == len(inv_dirs) == 3
    # Each retained invocation still carries its own blocks layer.
    for inv in agg["invocations"]:
        assert "blocks" in inv
        assert set(inv["blocks"]) == {
            "ArithmeticUtilization", "L2Cache", "Memory", "PipeUtilization"}


def test_kernel_to_ncu_format_exposes_ascend_invocations(hw_clocks):
    dram_freq_hz, dram_peak_bw = hw_clocks
    aggregates = _parse_kernel_dir(GEMM_DIR)
    assert len(aggregates) == 1
    ncu = _kernel_to_ncu_format(aggregates[0], dram_freq_hz, dram_peak_bw)

    invocations = ncu["Ascend"]["invocations"]
    assert len(invocations) == 3
    durations = [inv["duration_us"] for inv in invocations]
    assert len(set(durations)) > 1, (
        "3 invocations should be independently retained, not a 3x copy of "
        f"the aggregate mean: {durations}")


def test_kernel_to_ncu_format_timeline_mode_has_empty_invocations(hw_clocks, tmp_path):
    dram_freq_hz, dram_peak_bw = hw_clocks
    output_dir = tmp_path / "TIMELINE_case"
    summary_dir = output_dir / "PROF_000" / "mindstudio_profiler_output"
    summary_dir.mkdir(parents=True)
    header = (
        "Op Name,Task Duration(us),Current Freq,Block Dim,Mix Block Dim,"
        "aic_scalar_time(us),aic_cube_time(us),aic_mte2_time(us),"
        "aiv_vec_time(us),aiv_scalar_time(us)\n")
    row = "AtbRopeKernel_0,12.5,1800,20,0,0.1,0.2,0.3,0.4,0.5\n"
    (summary_dir / "op_summary_0.csv").write_text(header + row)

    aggregate = _parse_timeline_kernel(str(output_dir), "AtbRopeKernel")
    assert aggregate["invocations"] == []

    ncu = _kernel_to_ncu_format(aggregate, dram_freq_hz, dram_peak_bw)
    assert ncu["Ascend"]["invocations"] == []


def test_groupedmatmul_w1_w2_split_does_not_cross_contaminate_invocations(hw_clocks):
    dram_freq_hz, dram_peak_bw = hw_clocks
    aggregates = _parse_kernel_dir(GROUPEDMATMUL_DIR)
    assert len(aggregates) == 2
    names = {a["op_name"] for a in aggregates}
    assert any(name.endswith("_w1") for name in names)
    assert any(name.endswith("_w2") for name in names)

    w1 = next(a for a in aggregates if a["op_name"].endswith("_w1"))
    w2 = next(a for a in aggregates if a["op_name"].endswith("_w2"))

    w1_vols = {inv["gm_to_l1_kb"] for inv in w1["invocations"]}
    w2_vols = {inv["gm_to_l1_kb"] for inv in w2["invocations"]}
    assert len(w1_vols) == 1 and len(w2_vols) == 1
    (w1_vol,), (w2_vol,) = w1_vols, w2_vols
    assert w1_vol > w2_vol
    assert w1_vol / w2_vol == pytest.approx(2.0, rel=0.05)

    ncu_w1 = _kernel_to_ncu_format(w1, dram_freq_hz, dram_peak_bw)
    ncu_w2 = _kernel_to_ncu_format(w2, dram_freq_hz, dram_peak_bw)
    assert len(ncu_w1["Ascend"]["invocations"]) + len(ncu_w2["Ascend"]["invocations"]) == 12
