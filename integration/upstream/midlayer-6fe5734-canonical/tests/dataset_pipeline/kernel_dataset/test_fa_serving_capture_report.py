"""Focused fail-closed contracts for FA candidate publication reporting."""

from __future__ import annotations

import json

from dataset_pipeline.kernel_dataset.fa_serving_capture_report import (
    _add_replace_rows,
    _counter_records,
)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_counter_selection_ignores_stale_snapshot_but_not_accepted_ambiguity(tmp_path):
    expected = {"case": {"workload_digest": "a" * 64}}
    timing_path = tmp_path / "timing.json"
    _write_json(timing_path, {})
    timings = {"case": (timing_path, {"source_snapshot": {"sha256": "b" * 64}})}
    stale = tmp_path / "stale.counter.json"
    accepted = tmp_path / "accepted.counter.json"
    for path, snapshot in ((stale, "c" * 64), (accepted, "b" * 64)):
        _write_json(path, {
            "hardware_id": "nvidia_a800", "case_id": "case",
            "workload_digest": "a" * 64, "source_snapshot_sha256": snapshot,
            "counter_contract": {"duration_consumed": False},
        })

    selected = _counter_records(tmp_path, "nvidia_a800", expected, timings)

    assert selected == {"case": (accepted, json.loads(accepted.read_text()))}


def test_add_replace_marks_nonmatching_candidate_as_add_without_inferring_dtype(tmp_path):
    canonical_path = tmp_path / "flash_attn.json.zst"
    canonical_path.write_bytes(b"canonical bytes")
    workload = {
        "B": 1, "Q_LEN": 1, "KV_LEN": 512, "HQ": 32, "HKV": 8, "D": 128,
        "Q_LENS": [1], "KV_LENS": [512], "split_policy": "serving",
        "kv_layout": "paged_varlen", "causal": False,
    }
    cases = {"case": {"capture_metadata": {
        "logical_identity_sha256": "d" * 64, "workload": workload,
    }}}
    canonical = {"old": {"op_info": {"Attr": {
        **workload, "B": 2,
    }}}}

    rows = _add_replace_rows(
        hardware_id="nvidia_a800", cases=cases, canonical=canonical,
        canonical_path=canonical_path)

    assert rows[0]["action"] == "add"
    assert rows[0]["replaced_canonical_id"] == ""


def test_add_replace_preserves_dtype_unknown_canonical_partial_match(tmp_path):
    canonical_path = tmp_path / "flash_attn.json.zst"
    canonical_path.write_bytes(b"canonical bytes")
    workload = {
        "B": 1, "Q_LEN": 1, "KV_LEN": 512, "HQ": 32, "HKV": 8, "D": 128,
        "Q_LENS": [1], "KV_LENS": [512], "split_policy": "serving",
        "kv_layout": "paged_varlen", "causal": False,
    }
    cases = {"case": {"capture_metadata": {
        "logical_identity_sha256": "d" * 64, "workload": workload,
    }}}
    canonical = {"dtype_unknown": {"op_info": {"Attr": workload}}}

    rows = _add_replace_rows(
        hardware_id="nvidia_a800", cases=cases, canonical=canonical,
        canonical_path=canonical_path)

    assert rows[0]["action"] == "add"
    assert rows[0]["replaced_canonical_id"] == ""
    assert "no dtype evidence" in rows[0]["explanation"]
