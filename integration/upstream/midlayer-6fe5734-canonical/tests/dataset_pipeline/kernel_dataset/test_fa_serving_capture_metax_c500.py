"""Focused fail-closed contracts for C500 FA serving candidate evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_pipeline.kernel_dataset import fa_serving_capture_metax_c500 as c500


CASE_ID = "FA_SERVING_HQ32_HKV8_D128_DECODE_B14_QSUM14_QMAX1_KVMAX1275_3cd41ba9deb5"
DIGEST = "a" * 64
IDENTITY = "b" * 64
SNAPSHOT = "c" * 64


def _event_payload():
    return {
        "hardware_id": "metax_c500",
        "family": "flash_attn",
        "case_id": CASE_ID,
        "source_snapshot_sha256": SNAPSHOT,
        "canonical_attributes_sha256": IDENTITY,
        "status": "measured",
        "workload": {"capture_identity": {
            "case_id": CASE_ID,
            "hardware_id": "metax_c500",
            "logical_identity_sha256": IDENTITY,
            "workload_digest": DIGEST,
            "requested_num_splits": 1,
            "kv_page_size_tokens": 16,
        }},
        "measurement": {
            "mode": "external_events_around_cuda_graph_replay",
            "repetition_runs": [
                {"repetition_count": 50, "sample_count": 20, "statistics": {
                    "median_ns": 1000.0, "coefficient_of_variation": 0.01}},
                {"repetition_count": 200, "sample_count": 20, "statistics": {
                    "median_ns": 1002.0, "coefficient_of_variation": 0.01}},
            ],
            "statistics": {
                "median_ns": 1002.0,
                "coefficient_of_variation": 0.01,
            },
        },
    }


def _launch():
    return {
        "name": "flash_fwd_splitkv_kernel_final",
        "grid_x": 1,
        "grid_y": 2,
        "grid_z": 3,
        "block_size": 256,
        "regs_per_thread": 32,
        "dynamic_smem": 0,
        "static_smem": 0,
        "dur_ns": 21.0,
    }


def _counters():
    return {
        "total_cycles": 1,
        "ap_active_cycles": 2,
        "ap_mma_duty_pct": 3,
        "ap_mte_duty_pct": 4,
        "ap_ste_duty_pct": 5,
        "vls_duty_pct": 6,
        "l2c_duty_pct": 7,
        "vl1_hit_rate_pct": 8,
        "l2c_hit_rate_pct": 9,
        "global_memory_read_bytes": 10,
        "global_memory_write_bytes": 11,
        "achieved_waves": 12,
        "dispatched_waves": 13,
    }


def test_c500_one_split_evidence_requires_no_fabricated_combine(monkeypatch, tmp_path):
    case_root = tmp_path / "case"
    trace_path = case_root / "traces" / CASE_ID / "tracer_out-1.json"
    counter_path = case_root / "mcprofiler_out" / "_allk" / "1_flash_fwd_splitkv_kernel.txt.json"
    event_path = case_root / "golden" / "event.json"
    for path, value in ((trace_path, "{}"), (counter_path, "{}")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")
    event_path.parent.mkdir(parents=True)
    event_path.write_text(json.dumps(_event_payload()), encoding="utf-8")

    monkeypatch.setattr(c500, "_parse_trace_json", lambda path: [_launch()])
    monkeypatch.setattr(c500, "_allk_real_files", lambda directory: [str(counter_path)])
    monkeypatch.setattr(
        c500, "_join_layer1_observations",
        lambda directory, trace, case_id: ([(_launch(), _counters())], []))

    timing, candidate = c500.build_c500_case_evidence(
        case_id=CASE_ID, workload_digest=DIGEST,
        logical_identity_sha256=IDENTITY, source_snapshot_sha256=SNAPSHOT,
        case_root=case_root, event_path=event_path, trace_paths=[trace_path],
        allk_dir=counter_path.parent)

    assert timing["observed_execution"] == {
        "requested_num_splits": 1,
        "separate_combine_kernel_required": False,
        "reason": "one split produces one final result and no reduction stage",
    }
    assert candidate["counter_contract"]["duration_consumed"] is False
    timing_path = tmp_path / "timing.json"
    candidate_path = tmp_path / "candidate.json"
    timing_path.write_text(json.dumps(timing), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    closure = c500.build_c500_canary_closure(
        case_id=CASE_ID, workload_digest=DIGEST,
        timing_audit_path=timing_path, counter_candidate_path=candidate_path)
    assert closure["status"] == "pass"


def test_c500_evidence_rejects_raw_from_another_attempt(tmp_path):
    case_root = tmp_path / "case"
    case_root.mkdir()
    event_path = tmp_path / "other_attempt" / "event.json"

    with pytest.raises(ValueError, match="outside the C500 case root"):
        c500.build_c500_case_evidence(
            case_id=CASE_ID, workload_digest=DIGEST,
            logical_identity_sha256=IDENTITY, source_snapshot_sha256=SNAPSHOT,
            case_root=case_root, event_path=event_path,
            trace_paths=[case_root / "traces" / "missing.json"],
            allk_dir=case_root / "mcprofiler_out" / "_allk")


def test_c500_timing_without_counter_preserves_explicit_unavailable_state(
        monkeypatch, tmp_path):
    event_path = tmp_path / "golden" / "event.json"
    trace_path = tmp_path / "traces" / CASE_ID / "tracer_out-1.json"
    event_path.parent.mkdir(parents=True)
    trace_path.parent.mkdir(parents=True)
    event_path.write_text(json.dumps(_event_payload()), encoding="utf-8")
    trace_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(c500, "_parse_trace_json", lambda path: [_launch()])

    timing = c500.build_c500_timing_without_counter(
        case_id=CASE_ID, workload_digest=DIGEST,
        logical_identity_sha256=IDENTITY, source_snapshot_sha256=SNAPSHOT,
        event_path=event_path, trace_paths=[trace_path])

    assert timing["counter_status"] == "unavailable"
    assert timing["mctracer"]["kernels"][0]["kernel_wall_time_ns"] == 21.0
    assert "aggregate report" in timing["counter_unavailable_reason"]
