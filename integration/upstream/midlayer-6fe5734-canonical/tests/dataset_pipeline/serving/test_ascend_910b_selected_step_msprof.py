import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from dataset_pipeline.serving_dataset.ascend_910b_selected_step_msprof import (
    _model_runner_outer_timing,
    _source_sequence_classification,
    _stack_sample_interval_attribution,
    analyze_ascend_910b_selected_step_dispatches,
    analyze_ascend_910b_selected_step_msprof,
    summarize_ascend_910b_model_runner_outer_timing,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def test_model_runner_outer_timing_partitions_same_run_source_scopes(tmp_path):
    def record(inclusive_wall_ns, exclusive_wall_ns):
        return {
            "call_count": 1,
            "exclusive_thread_cpu_ns": exclusive_wall_ns,
            "exclusive_wall_ns": exclusive_wall_ns,
            "inclusive_thread_cpu_ns": inclusive_wall_ns,
            "inclusive_wall_ns": inclusive_wall_ns,
            "maximum_wall_ns": inclusive_wall_ns,
            "minimum_wall_ns": inclusive_wall_ns,
        }

    profile_path = tmp_path / "framework-timing.json"
    profile_path.write_text(json.dumps({
        "schema_version": 2,
        "source": "selected_function_aggregate_timing",
        "process_pid": 101,
        "worker_rank": 1,
        "trace_step_id": "vllm:29",
        "thread_cpu_clock": "time.thread_time_ns",
        "wall_clock": "time.perf_counter_ns",
        "nesting_semantics": "exclusive subtracts only nested instrumented functions",
        "torch_dispatch_count_by_source": {},
        "records": {
            "model_runner_execute": record(180, 10),
            "model_forward": record(130, 3),
            "model_runner_input_prepare": record(12, 4),
            "model_runner_compute_logits": record(9, 2),
            "model_runner_attention_metadata": record(3, 3),
            "model_runner_update_cos_sin": record(4, 4),
            "model_runner_state_update": record(5, 5),
            "model_runner_forward_context_enter": record(2, 2),
            "model_runner_forward_context_exit": record(1, 1),
            "model_runner_batch_descriptor": record(1, 1),
            "model_runner_preprocess": record(1, 1),
            "model_runner_ubatch_slices": record(2, 2),
        },
    }))

    result = _model_runner_outer_timing(
        profile_path,
        101,
        1,
        "vllm:29",
        {
            "status": "observed",
            "wall_span_ns": 190,
            "thread_cpu_span_ns": 180,
        },
    )

    assert result["model_forward_wall_ns"] == 130
    assert result["model_runner_outer_wall_ns"] == 50
    assert result["model_runner_outer_components_ns"] == {
        "input_prepare_ns": 12,
        "logits_prepare_ns": 9,
        "forward_metadata_ns": 19,
        "model_runner_uninstrumented_control_ns": 10,
    }
    assert result["executor_outer_wall_ns"] == 10
    assert result["executor_outer_thread_cpu_ns"] == 0
    assert result["executor_outer_non_cpu_wait_ns"] == 10
    assert result["production_forward_eligible"] is False


def test_model_runner_outer_summary_requires_exact_boundary_identity(tmp_path):
    def record(inclusive_wall_ns, exclusive_wall_ns):
        return {
            "call_count": 1,
            "exclusive_thread_cpu_ns": exclusive_wall_ns,
            "exclusive_wall_ns": exclusive_wall_ns,
            "inclusive_thread_cpu_ns": inclusive_wall_ns,
            "inclusive_wall_ns": inclusive_wall_ns,
            "maximum_wall_ns": inclusive_wall_ns,
            "minimum_wall_ns": inclusive_wall_ns,
        }

    framework_path = tmp_path / "framework-timing.json"
    framework_path.write_text(json.dumps({
        "schema_version": 2,
        "source": "selected_function_aggregate_timing",
        "process_pid": 101,
        "worker_rank": 1,
        "trace_step_id": "vllm:29",
        "thread_cpu_clock": "time.thread_time_ns",
        "wall_clock": "time.perf_counter_ns",
        "nesting_semantics": "exclusive subtracts only nested instrumented functions",
        "torch_dispatch_count_by_source": {},
        "records": {
            "model_runner_execute": record(180, 10),
            "model_forward": record(130, 3),
            "model_runner_input_prepare": record(12, 4),
            "model_runner_compute_logits": record(9, 2),
            "model_runner_attention_metadata": record(3, 3),
            "model_runner_update_cos_sin": record(4, 4),
            "model_runner_state_update": record(5, 5),
            "model_runner_forward_context_enter": record(2, 2),
            "model_runner_forward_context_exit": record(1, 1),
            "model_runner_batch_descriptor": record(1, 1),
            "model_runner_preprocess": record(1, 1),
            "model_runner_ubatch_slices": record(2, 2),
        },
    }))
    boundary_path = tmp_path / "event-boundary.jsonl"
    boundary_path.write_text(json.dumps({
        "schema_version": 2,
        "source": "ascend_aclrt_event_record_boundary",
        "process_pid": 101,
        "worker_rank": 1,
        "marker_ordinal": 0,
        "trace_step_id": "vllm:29",
        "stream_handle": 1,
        "event_handle": 2,
        "event_issue_start_ns": 300,
        "event_issue_end_ns": 301,
        "event_issue_start_wall_ns": 300,
        "event_issue_end_wall_ns": 301,
        "worker_execution_start_ns": 100,
        "worker_execution_end_ns": 290,
        "worker_execution_clock": "time.perf_counter_ns",
        "worker_thread_cpu_start_ns": 30,
        "worker_thread_cpu_end_ns": 210,
        "worker_thread_cpu_clock": "time.thread_time_ns",
    }) + "\n")

    result = summarize_ascend_910b_model_runner_outer_timing(
        framework_path, boundary_path)

    assert result["trace_step_id"] == "vllm:29"
    assert result["worker_rank"] == 1
    assert result["worker_execution_wall_ns"] == 190
    assert result["worker_execution_thread_cpu_ns"] == 180
    assert result["executor_outer_wall_ns"] == 10
    assert result["executor_outer_thread_cpu_ns"] == 0
    assert result["executor_outer_non_cpu_wait_ns"] == 10


def test_stack_sample_attribution_distributes_exact_exposed_time(tmp_path):
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "schema_version": 1,
        "source": "python_current_frames_statistical_sampling",
        "process_pid": 101,
        "worker_rank": 1,
        "worker_thread_native_id": 77,
        "trace_step_id": "vllm:29",
        "sample_interval_s": 0.001,
        "worker_call_start_wall_ns": 100,
        "worker_call_end_wall_ns": 140,
        "samples": [
            {"wall_ns": 105, "stack": [{
                "name": "first", "file": "/vllm-workspace/vllm/a.py", "line": 1,
            }]},
            {"wall_ns": 115, "stack": [{
                "name": "second", "file": "/vllm-workspace/vllm/b.py", "line": 2,
            }]},
            {"wall_ns": 135, "stack": [{
                "name": "third", "file": "/vllm-workspace/vllm/c.py", "line": 3,
            }]},
        ],
    }))

    result = _stack_sample_interval_attribution(
        profile_path, 101, 1, "vllm:29", ((110, 130),)
    )

    assert result["target_interval_ns"] == 20
    assert result["matching_sample_count"] == 1
    assert result["total_sample_count"] == 3
    assert result["top_vllm_callsites"] == [{
        "name": "second",
        "file": "/vllm-workspace/vllm/b.py",
        "line": 2,
        "sampled_ns": 20,
        "sample_count": 1,
    }]


def test_selected_step_analyzer_cli_is_runnable_from_repository_root():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluation/serving/diagnostics/"
            "analyze_ascend_910b_selected_step_msprof.py",
            "--help",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--replay" in result.stdout


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trace_path = run_dir / "scheduler_steps.raw.jsonl"
    trace_rows = [
        {
            "step_id": "vllm:28", "schedule_end_ns": 10,
            "update_start_ns": 90, "step_start_ns": 0, "step_end_ns": 95,
            "device_execution_start_ns": 15, "device_execution_end_ns": 85,
        },
        {
            "step_id": "vllm:29", "schedule_end_ns": 105,
            "update_start_ns": 190, "step_start_ns": 100, "step_end_ns": 195,
            "device_execution_start_ns": 110, "device_execution_end_ns": 180,
        },
    ]
    trace_path.write_text(
        "".join(json.dumps(row) + "\n" for row in trace_rows), encoding="utf-8"
    )
    Path(f"{trace_path}.clock.json").write_text(json.dumps({
        "clock_domain": "shared_host_time.perf_counter_ns",
        "trace_origin_monotonic_ns": 0,
    }))
    validation_path = run_dir / "validation.json"
    validation_path.write_text(json.dumps({
        "canonical_step_id": "vllm:105",
        "injected_step_id": "vllm:29",
        "structural_fingerprint_match": True,
    }))
    for rank in range(2):
        pid = 100 + rank
        device_dir = run_dir / "msprof" / f"PROF_{rank}" / f"device_{rank}"
        sqlite_dir = device_dir / "sqlite"
        sqlite_dir.mkdir(parents=True)
        (device_dir / f"info.json.{rank}").write_text(json.dumps({
            "pid": str(pid), "DeviceInfo": [{"id": rank}],
        }))
        marker_path = run_dir / (
            f"{trace_path.name}.ascend-step-marker.rank-{rank}.pid-{pid}.jsonl"
        )
        marker_path.write_text("".join(json.dumps({
            "schema_version": 3,
            "source": "ascend_mstx_device_end_mark",
            "process_pid": pid,
            "worker_rank": rank,
            "marker_ordinal": ordinal,
            "trace_step_id": step_id,
            "stream_handle": 19,
            "mark_issue_start_ns": boundary,
            "mark_issue_end_ns": boundary + 1,
        }) + "\n" for ordinal, (step_id, boundary) in enumerate(
            (("vllm:28", 100), ("vllm:29", 200))
        )))
        with sqlite3.connect(sqlite_dir / "step_trace.db") as connection:
            connection.execute(
                "CREATE TABLE DeviceTaskFlip(stream_id INTEGER, timestamp NUMERIC, "
                "task_id INTEGER, flip_num INTEGER)"
            )
            connection.executemany(
                "INSERT INTO DeviceTaskFlip VALUES (?, ?, ?, ?)",
                ((2, 100, 65535, 65535), (2, 200, 65535, 65535)),
            )
        with sqlite3.connect(sqlite_dir / "time.db") as connection:
            connection.execute(
                "CREATE TABLE Time(device_id INTEGER, dev_mon INTEGER, "
                "host_mon INTEGER)"
            )
            connection.execute("INSERT INTO Time VALUES (?, 0, 0)", (rank,))
        with sqlite3.connect(sqlite_dir / "ascend_task.db") as connection:
            connection.execute(
                "CREATE TABLE AscendTask(start_time INTEGER, duration INTEGER, "
                "host_task_type TEXT, device_task_type TEXT, stream_id INTEGER, "
                "task_id INTEGER)"
            )
            rows = (
                ((110, 20, "GEMM", "AI_CORE", 2, 1),
                 (140, 20, "VECTOR", "AI_CORE", 2, 2))
                if rank == 0
                else ((115, 30, "GEMM", "AI_CORE", 2, 3),
                      (160, 10, "HCCL", "COMM", 3, 4))
            )
            connection.executemany("INSERT INTO AscendTask VALUES (?, ?, ?, ?, ?, ?)", rows)
    return run_dir, trace_path, validation_path


def test_selected_step_uses_adjacent_device_marks_and_cross_rank_union(tmp_path):
    run_dir, trace_path, validation_path = _write_fixture(tmp_path)

    result = analyze_ascend_910b_selected_step_msprof(
        run_dir, trace_path, validation_path, "vllm:29", 2
    )

    assert result["comparison_status"] == "valid_same_run_device_timeline"
    assert result["device_task_union_ms"] == pytest.approx(0.000060)
    assert result["exposed_device_task_union_ms"] == pytest.approx(0.000060)
    assert result["serialized_service_ms"] == pytest.approx(0.000085)
    assert result["uncovered_service_ms"] == pytest.approx(0.000025)
    assert [profile["rank"] for profile in result["profiles"]] == [0, 1]


def test_selected_step_rejects_nonmatching_fingerprint(tmp_path):
    run_dir, trace_path, validation_path = _write_fixture(tmp_path)
    validation = json.loads(validation_path.read_text())
    validation["structural_fingerprint_match"] = False
    validation_path.write_text(json.dumps(validation))

    with pytest.raises(ValueError, match="INVALID EXPERIMENT"):
        analyze_ascend_910b_selected_step_msprof(
            run_dir, trace_path, validation_path, "vllm:29", 2
        )


def test_selected_step_dispatches_expand_collectives_and_count_physical_launches(
    tmp_path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    trace_path = run_dir / "scheduler_steps.raw.jsonl"
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in (
        {
            "step_id": "vllm:28", "step_start_ns": 0,
            "schedule_end_ns": 10, "update_start_ns": 125,
            "step_end_ns": 130,
        },
        {
            "step_id": "vllm:29", "step_start_ns": 100,
            "schedule_end_ns": 105, "update_start_ns": 190,
            "step_end_ns": 195,
        },
    )))
    Path(f"{trace_path}.clock.json").write_text(json.dumps({
        "clock_domain": "shared_host_time.perf_counter_ns",
        "trace_origin_monotonic_ns": 0,
    }))
    validation_path = run_dir / "validation.json"
    validation_path.write_text(json.dumps({
        "canonical_step_id": "vllm:105",
        "injected_step_id": "vllm:29",
        "structural_fingerprint_match": True,
    }))
    replay_path = run_dir / "replay.json"
    replay_path.write_text(json.dumps({
        "step_comparison": [{
            "step_id": "vllm:29",
            "step_fingerprint": {"graph_execution": {"runtime_mode": "NONE"}},
            "predicted_kernel_contributions": [
                {
                    "rank": 0,
                    "occurrence_count": 2,
                    "kernel_spec": {"kind": "VECTOR", "transfer": None},
                },
                {
                    "rank": 1,
                    "occurrence_count": 2,
                    "kernel_spec": {"kind": "VECTOR", "transfer": None},
                },
                {
                    "rank": 0,
                    "occurrence_count": 1,
                    "kernel_spec": {
                        "kind": "COMMUNICATION",
                        "transfer": {"participants": [0, 1]},
                    },
                },
            ],
        }],
    }))
    for rank in range(2):
        pid = 100 + rank
        global_tid = (pid << 32) | 7
        sidecar = run_dir / (
            "scheduler_steps.raw.jsonl.ascend-event-boundary."
            f"rank-{rank}.pid-{pid}.jsonl"
        )
        sidecar.write_text(json.dumps({
            "schema_version": 2,
            "source": "ascend_aclrt_event_record_boundary",
            "process_pid": pid,
            "worker_rank": rank,
            "marker_ordinal": 0,
            "trace_step_id": "vllm:29",
            "stream_handle": 11,
            "event_handle": 12,
            "event_issue_start_ns": 200,
            "event_issue_end_ns": 201,
            "event_issue_start_wall_ns": 200,
            "event_issue_end_wall_ns": 201,
            "worker_execution_start_ns": 90,
            "worker_execution_end_ns": 190,
            "worker_execution_clock": "time.perf_counter_ns",
            "worker_thread_cpu_start_ns": 30,
            "worker_thread_cpu_end_ns": 90,
            "worker_thread_cpu_clock": "time.thread_time_ns",
        }) + "\n")
        with sqlite3.connect(run_dir / f"msprof_rank{rank}.db") as connection:
            connection.execute(
                "CREATE TABLE STRING_IDS(id INTEGER PRIMARY KEY, value TEXT)"
            )
            connection.executemany(
                "INSERT INTO STRING_IDS VALUES (?, ?)",
                (
                    (1, "aclrtRecordEvent"),
                    (2, "KERNEL_AIVEC"),
                    (3, "AI_CORE"),
                    (4, "primary_kernel"),
                    (5, "helper_kernel"),
                    (6, "EVENT_RECORD"),
                ),
            )
            connection.execute(
                "CREATE TABLE CANN_API(startNs INTEGER, endNs INTEGER, "
                "type INTEGER, globalTid INTEGER, connectionId INTEGER, name INTEGER)"
            )
            connection.executemany(
                "INSERT INTO CANN_API VALUES (?, ?, ?, ?, ?, ?)",
                (
                    (100, 110, 10000, global_tid, 10, 4),
                    (120, 130, 10000, global_tid, 11, 4),
                    (140, 150, 10000, global_tid, 12, 4),
                    (160, 170, 10000, global_tid, 13, 5),
                    (200, 210, 20000, global_tid, 99, 1),
                ),
            )
            connection.execute(
                "CREATE TABLE TASK(startNs INTEGER, endNs INTEGER, "
                "connectionId INTEGER, globalTaskId INTEGER, taskType INTEGER)"
            )
            connection.executemany(
                "INSERT INTO TASK VALUES (?, ?, ?, ?, ?)",
                (
                    (111, 112, 10, 100, 2),
                    (131, 132, 11, 101, 2),
                    (151, 152, 12, 102, 3),
                    (171, 172, 13, 103, 2),
                    (220, 221, 99, 104, 6),
                ),
            )
            connection.execute(
                "CREATE TABLE COMPUTE_TASK_INFO(globalTaskId INTEGER, name INTEGER)"
            )
            connection.executemany(
                "INSERT INTO COMPUTE_TASK_INFO VALUES (?, ?)",
                ((100, 4), (101, 4), (103, 5)),
            )

    result = analyze_ascend_910b_selected_step_dispatches(
        run_dir, validation_path, replay_path, "vllm:29", 2
    )

    assert result["schema_version"] == 7
    assert result["comparison_status"] == "valid_same_run_dispatch_correlation"
    assert result["production_dispatch_mapping_complete"] is False
    assert result["production_host_timing_eligibility"] == (
        "ineligible_incomplete_physical_dispatch_mapping"
    )
    assert [profile["planned_timeline_occurrences"] for profile in result["profiles"]] == [3, 2]
    assert [profile["planned_participant_expanded_occurrences"] for profile in result["profiles"]] == [3, 3]
    assert [profile["observed_physical_launches"] for profile in result["profiles"]] == [4, 4]
    assert [profile["unmapped_excess_launches"] for profile in result["profiles"]] == [1, 1]
    assert [
        profile["source_sequence_classification"]
        for profile in result["profiles"]
    ] == [
        {
            "method": "ordered_kernel_adjacency",
            "attention_occurrences": 0,
            "attention_first_output_copies": 0,
            "attention_second_output_copies": 0,
            "attention_missing_launches": 0,
            "tensor_move_launches": 0,
            "non_attention_tensor_move_launches": 0,
            "moe_routing_sequences": 0,
            "moe_missing_launches": 0,
            "layer_regular_missing_launches": 0,
            "residual_unmapped_launches": 1,
        },
        {
            "method": "ordered_kernel_adjacency",
            "attention_occurrences": 0,
            "attention_first_output_copies": 0,
            "attention_second_output_copies": 0,
            "attention_missing_launches": 0,
            "tensor_move_launches": 0,
            "non_attention_tensor_move_launches": 0,
            "moe_routing_sequences": 0,
            "moe_missing_launches": 0,
            "layer_regular_missing_launches": 0,
            "residual_unmapped_launches": 1,
        },
    ]
    assert [
        profile["planned_to_observed_delta_launches"]
        for profile in result["profiles"]
    ] == [1, 1]
    assert [
        profile["dispatch_mapping_status"] for profile in result["profiles"]
    ] == ["incomplete_observed_excess", "incomplete_observed_excess"]
    assert [profile["host_program_span_ns"] for profile in result["profiles"]] == [70, 70]
    assert [profile["host_launch_union_ns"] for profile in result["profiles"]] == [40, 40]
    assert [profile["host_outside_profiler_api_ns"] for profile in result["profiles"]] == [30, 30]
    assert [profile["device_task_union_ns"] for profile in result["profiles"]] == [3, 3]
    assert [profile["device_no_task_gap_ns"] for profile in result["profiles"]] == [67, 67]
    assert [profile["cann_api_in_device_gap_ns"] for profile in result["profiles"]] == [40, 40]
    assert [
        profile["outside_cann_api_in_device_gap_ns"]
        for profile in result["profiles"]
    ] == [27, 27]
    assert result["scheduler_pipeline_breakdown"] == {
        "clock_domain": "shared_host_time.perf_counter_ns",
        "selected_step_full_ns": 95,
        "selected_scheduler_plan_ns": 5,
        "overlapped_by_previous_step_ns": 25,
        "serialized_service_ns": 65,
        "selected_scheduler_update_ns": 5,
    }
    for profile in result["profiles"]:
        assert profile["stack_sample_attribution"] == {"status": "not_collected"}
        assert profile["host_overlap_breakdown"]["total"] == {
            "total_ns": 70,
            "overlapped_by_previous_step_ns": 25,
            "overlapped_by_selected_device_ns": 2,
            "exposed_device_idle_ns": 43,
        }
        assert profile["host_overlap_breakdown"]["categories"] == {
            "launch_api": {
                "total_ns": 40,
                "overlapped_by_previous_step_ns": 15,
                "overlapped_by_selected_device_ns": 0,
                "exposed_device_idle_ns": 25,
            },
            "non_launch_cann_api": {
                "total_ns": 0,
                "overlapped_by_previous_step_ns": 0,
                "overlapped_by_selected_device_ns": 0,
                "exposed_device_idle_ns": 0,
            },
            "outside_cann_api_unknown": {
                "total_ns": 30,
                "overlapped_by_previous_step_ns": 10,
                "overlapped_by_selected_device_ns": 2,
                "exposed_device_idle_ns": 18,
            },
        }
    for profile in result["profiles"]:
        observed = profile["observed_timeline"]
        assert observed["clock_domain"] == "msprof_host_wall_ns"
        assert len(observed["physical_dispatch_events"]) == 4
        assert observed["device_task_union_intervals"] == (
            {"start_ns": 111, "end_ns": 112},
            {"start_ns": 131, "end_ns": 132},
            {"start_ns": 151, "end_ns": 152},
        )
    assert [profile["worker_execution"] for profile in result["profiles"]] == [
        {
            "status": "observed",
            "clock_domain": "time.perf_counter_ns",
            "thread_cpu_clock": "time.thread_time_ns",
            "wall_span_ns": 100,
            "thread_cpu_ns": 60,
            "non_cpu_wall_ns": 40,
        },
        {
            "status": "observed",
            "clock_domain": "time.perf_counter_ns",
            "thread_cpu_clock": "time.thread_time_ns",
            "wall_span_ns": 100,
            "thread_cpu_ns": 60,
            "non_cpu_wall_ns": 40,
        },
    ]


def test_source_sequence_classification_uses_order_not_name_totals():
    names = (
        "aclnnInplaceCopy_TensorMoveAiCore_TensorMove",
        "FusedInferAttentionScore",
        "aclnnInplaceCopy_TensorMoveAiCore_TensorMove",
        "other",
        "FusedInferAttentionScore",
        "aclnnInplaceCopy_TensorMoveAiCore_TensorMove",
        "aclnnInplaceCopy_TensorMoveAiCore_TensorMove",
        "MoeGatingTopK",
        "aclnnIndex_IndexAiCore_Index",
        "aclnnNeScalar_NotEqualAiCore_NotEqual",
        "aclnnMul_CastAiCore_Cast",
        "aclnnMul_MulAiCore_Mul",
        "aclnnMoeInitRoutingCustom_MoeInitRoutingCustom_MoeInitRoutingCustom",
    )
    launches = [
        (index, index + 1, index, 7, "compute", name)
        for index, name in enumerate(names)
    ]

    assert _source_sequence_classification(launches, 7) == {
        "method": "ordered_kernel_adjacency",
        "attention_occurrences": 2,
        "attention_first_output_copies": 2,
        "attention_second_output_copies": 1,
        "attention_missing_launches": 3,
        "tensor_move_launches": 4,
        "non_attention_tensor_move_launches": 1,
        "moe_routing_sequences": 1,
        "moe_missing_launches": 3,
        "layer_regular_missing_launches": 6,
        "residual_unmapped_launches": 1,
    }
