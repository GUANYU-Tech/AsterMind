import json

import pytest

from evaluation.serving.traces.trace_validation import (
    load_scheduler_records,
    validate_case,
)
from evaluation.serving.traces.trace_alignment_io import nsys_time_bounds


def _write_valid_case(case_dir):
    (case_dir / "arrival_schedule.json").write_text(json.dumps({
        "qps": 1.0,
        "arrival_times_s": [0.0, 1.0, 2.0],
    }))
    (case_dir / "plugin_trace.json_scheduler.jsonl").write_text(
        json.dumps({
            "iter_id": 0,
            "ts_us": 1,
            "total_tokens": 1,
            "scheduled": {"warmup_0": 1},
        }) + "\n" + json.dumps({
            "iter_id": 1,
            "ts_us": 1_000_001,
            "total_tokens": 1,
            "scheduled": {"req_0": 1},
        }) + "\n" + json.dumps({
            "iter_id": 2,
            "ts_us": 2_000_001,
            "total_tokens": 1,
            "scheduled": {"req_1": 1},
        }) + "\n")
    (case_dir / "serving_metrics.json").write_text(json.dumps({
        "per_request": [
            {"n_output_tokens": 1, "send_time_s": 0.0},
            {"n_output_tokens": 2, "send_time_s": 1.0},
            {"n_output_tokens": 3, "send_time_s": 2.0},
        ],
    }))


def test_valid_serving_trace_passes(tmp_path):
    _write_valid_case(tmp_path)

    assert validate_case(tmp_path) == []


def test_missing_trace_artifact_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="arrival schedule"):
        validate_case(tmp_path)


def test_zero_span_arrival_schedule_is_rejected(tmp_path):
    _write_valid_case(tmp_path)
    (tmp_path / "arrival_schedule.json").write_text(json.dumps({
        "qps": 0.0,
        "arrival_times_s": [0.0, 0.0],
    }))

    with pytest.raises(ValueError, match="positive time span"):
        validate_case(tmp_path)


def test_scheduler_sidecar_rejects_duplicate_iterations(tmp_path):
    path = tmp_path / "plugin_trace.json_scheduler.jsonl"
    path.write_text(
        '{"iter_id": 0, "ts_us": 1, "total_tokens": 1, '
        '"scheduled": {"req_0": 1}}\n'
        '{"iter_id": 0, "ts_us": 2, "total_tokens": 1, '
        '"scheduled": {"req_1": 1}}\n')

    with pytest.raises(ValueError, match="duplicate iter_id"):
        load_scheduler_records(path)


def test_scheduler_sidecar_rejects_blank_lines(tmp_path):
    path = tmp_path / "plugin_trace.json_scheduler.jsonl"
    path.write_text(
        '{"iter_id": 0, "ts_us": 1, "total_tokens": 1, '
        '"scheduled": {"req_0": 1}}\n\n')

    with pytest.raises(ValueError, match="blank line"):
        load_scheduler_records(path)


def _write_nsys_kernel_trace(case_dir, names):
    kernels = [
        {
            "ts_ns": index + 1,
            "name": name,
            "grid": [1, 1, 1],
            "block": [1, 1, 1],
            "smem": 0,
        }
        for index, name in enumerate(names)
    ]
    (case_dir / "nsys_kernels_full.json").write_text(json.dumps({
        "kernels": kernels,
    }))


def test_nsys_time_bounds_rejects_unidentified_model_start(tmp_path):
    _write_nsys_kernel_trace(tmp_path, ["framework_setup", "framework_copy"])

    with pytest.raises(ValueError, match="first model kernel"):
        nsys_time_bounds(tmp_path)


def test_nsys_time_bounds_uses_explicit_model_start_marker(tmp_path):
    _write_nsys_kernel_trace(
        tmp_path, ["framework_setup", "rms_norm_kernel", "model_gemm"])

    assert nsys_time_bounds(tmp_path) == (2, 3)
