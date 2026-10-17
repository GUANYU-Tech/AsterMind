"""Focused contracts for Ascend FA Timeline selection and stability."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/kernel_dataset/fa_serving_capture_ascend_910b_full_runner.py"
SPEC = importlib.util.spec_from_file_location("fa_serving_capture_ascend_full", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_op_summary(path, *, durations):
    fields = [
        "Device_id", "Task ID", "Stream ID", "Op Name", "OP Type",
        "Task Type", "Task Duration(us)", "Block Dim", "Mix Block Dim",
    ]
    rows = [
        {
            "Device_id": "0", "Task ID": "3", "Stream ID": "2",
            "Op Name": "FusedInferAttentionScore", "OP Type": "FusedInferAttentionScore",
            "Task Type": "MIX_AIC", "Task Duration(us)": str(durations[0]),
            "Block Dim": "24", "Mix Block Dim": "48",
        },
        {
            "Device_id": "0", "Task ID": "37", "Stream ID": "2",
            "Op Name": "FusedInferAttentionScore", "OP Type": "FusedInferAttentionScore",
            "Task Type": "MIX_AIC", "Task Duration(us)": str(durations[1]),
            "Block Dim": "24", "Mix Block Dim": "48",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_ascend_timeline_uses_final_public_task_and_median_stability(tmp_path):
    paths = []
    for index, duration in enumerate((100.0, 102.0, 101.0), start=1):
        path = tmp_path / f"window_{index}.csv"
        _write_op_summary(path, durations=(1.0, duration))
        paths.append(path)

    audit = MODULE._build_timing_audit(
        case_id="case", workload_digest="a" * 64,
        source_snapshot_sha256="b" * 64, op_summary_paths=paths,
        profile_warmup_iterations=20)

    assert audit["status"] == "pass"
    assert [window["task_id"] for window in audit["windows"]] == [37, 37, 37]
    assert audit["aggregation"]["median_duration_ns"] == 101_000.0
    assert audit["aggregation"]["max_relative_spread"] < 0.05
    assert audit["measurement_conditions"]["profile_warmup_iterations"] == 20


def test_ascend_timeline_stability_rejects_spread_above_gate(tmp_path):
    paths = []
    for index, duration in enumerate((100.0, 106.0, 102.0), start=1):
        path = tmp_path / f"window_{index}.csv"
        _write_op_summary(path, durations=(1.0, duration))
        paths.append(path)

    audit = MODULE._build_timing_audit(
        case_id="case", workload_digest="a" * 64,
        source_snapshot_sha256="b" * 64, op_summary_paths=paths,
        profile_warmup_iterations=20)

    assert audit["status"] == "fail"
    assert audit["aggregation"]["pass"] is False


def test_ascend_timing_failure_is_not_a_completion_record(tmp_path):
    timing_path = tmp_path / "timing.json"
    record = MODULE._timing_failure_record(
        case_id="case", workload_digest="a" * 64, timing_audit_path=timing_path)

    assert record == {
        "schema_version": 1,
        "kind": "fa_serving_coverage_candidate_ascend_910b_timing_failure",
        "case_id": "case",
        "workload_digest": "a" * 64,
        "status": "timing_stability_failed",
        "timing_audit": str(timing_path),
    }
    assert "counter_candidate" not in record


def test_ascend_remote_timeline_summary_groups_are_complete_and_scoped():
    root = "/gpfs/capture/case"
    paths = [
        f"{root}/golden/window_{window}/msprof/profile/op_summary_{copy}.csv"
        for window in range(1, 4)
        for copy in ("automatic", "explicit")
    ]

    groups = MODULE._group_remote_timeline_summaries(
        remote_host_case_root=root, paths=paths)

    assert list(groups) == ["window_1", "window_2", "window_3"]
    assert all(len(values) == 2 for values in groups.values())

    incomplete = paths[:-1]
    try:
        MODULE._group_remote_timeline_summaries(
            remote_host_case_root=root, paths=incomplete)
    except ValueError as error:
        assert "exactly two" in str(error)
    else:
        raise AssertionError("missing Timeline export must fail closed")


def test_ascend_timeline_requires_identical_automatic_and_explicit_exports(tmp_path):
    summary_root = tmp_path / "msprof" / "profile" / "mindstudio_profiler_output"
    summary_root.mkdir(parents=True)
    first = summary_root / "op_summary_001.csv"
    second = summary_root / "op_summary_002.csv"
    _write_op_summary(first, durations=(1.0, 100.0))
    _write_op_summary(second, durations=(1.0, 100.0))

    assert MODULE._timeline_summary_path(tmp_path) == second

    _write_op_summary(second, durations=(1.0, 101.0))
    try:
        MODULE._timeline_summary_path(tmp_path)
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("divergent Timeline exports must fail closed")


def test_ascend_counter_output_can_use_real_path_while_application_uses_short_root():
    args = SimpleNamespace(
        remote_output_root="/gpfs/midlayer/.short/raw",
        remote_container_root="/mnt/gpfs/midlayer/.short/raw",
        counter_output_container_root=None,
        counter_output_host_root=None,
    )
    assert MODULE._counter_container_case_root(
        args=args, case_id="case") == "/mnt/gpfs/midlayer/.short/raw/case"
    assert MODULE._counter_host_case_root(
        args=args, case_id="case") == "/gpfs/midlayer/.short/raw/case"

    args.counter_output_container_root = "/mnt/gpfs/midlayer/real/raw"
    try:
        MODULE._counter_host_case_root(args=args, case_id="case")
    except ValueError as error:
        assert "--counter-output-host-root" in str(error)
    else:
        raise AssertionError("separate physical counter root must not fall back")

    args.counter_output_host_root = "/gpfs/midlayer/real/raw"
    assert MODULE._counter_container_case_root(
        args=args, case_id="case") == "/mnt/gpfs/midlayer/real/raw/case"
    assert MODULE._counter_host_case_root(
        args=args, case_id="case") == "/gpfs/midlayer/real/raw/case"


def test_ascend_resume_existing_counter_requires_explicit_physical_root():
    args = SimpleNamespace(
        resume_existing_counter_host_root="/gpfs/midlayer/counter/raw")
    assert MODULE._existing_counter_host_case_root(
        args=args, case_id="case") == "/gpfs/midlayer/counter/raw/case"

    args.resume_existing_counter_host_root = "relative/counter"
    try:
        MODULE._existing_counter_host_case_root(args=args, case_id="case")
    except ValueError as error:
        assert "absolute" in str(error)
    else:
        raise AssertionError("relative counter root must fail")

    args.resume_existing_counter_host_root = "/gpfs/midlayer/../counter"
    try:
        MODULE._existing_counter_host_case_root(args=args, case_id="case")
    except ValueError as error:
        assert "normalized" in str(error)
    else:
        raise AssertionError("unnormalized counter root must fail")


def test_ascend_copy_counter_uses_physical_counter_host_root(tmp_path, monkeypatch):
    commands = []

    def record(command, *, log_path):
        commands.append((command, log_path))

    monkeypatch.setattr(MODULE, "_run", record)
    MODULE._copy_counter_raw(
        args=SimpleNamespace(host="910b-1"),
        counter_host_case_root="/gpfs/midlayer/physical/raw/case",
        local_case_root=tmp_path / "case", local_log_root=tmp_path / "logs",
        log_tag="resume")

    assert [command[1] for command, _ in commands] == ["-r"]
    assert [command[-2] for command, _ in commands] == [
        "910b-1:/gpfs/midlayer/physical/raw/case/counter"]


def test_ascend_remote_program_hashes_are_complete_and_scoped():
    root = "/gpfs/capture/case"
    paths = [
        f"{root}/golden/window_{window}/msprof/profile/"
        f"unaging.additional.hash_dic.slice_{window}"
        for window in range(1, 4)
    ]

    groups = MODULE._group_remote_program_hashes(
        remote_host_case_root=root, paths=paths)

    assert list(groups) == ["window_1", "window_2", "window_3"]
    assert all(len(values) == 1 for values in groups.values())

    try:
        MODULE._group_remote_program_hashes(
            remote_host_case_root=root, paths=paths[:-1])
    except ValueError as error:
        assert "has no program-hash" in str(error)
    else:
        raise AssertionError("missing Timeline program hash must fail closed")
