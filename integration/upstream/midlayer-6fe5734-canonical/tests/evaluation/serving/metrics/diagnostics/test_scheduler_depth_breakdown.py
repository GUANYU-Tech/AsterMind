"""Contracts for Scheduler-depth diagnostic trace projection."""

import json
from types import SimpleNamespace

import pytest
import zstandard

from evaluation.serving.metrics.diagnostics import scheduler_depth_breakdown
from evaluation.serving.metrics.diagnostics import scheduling_trajectory_replay_report
from evaluation.serving.traces import scheduling_trajectory_replay as replay


def test_final_serving_cell_persists_native_pipeline_depth(
    tmp_path, monkeypatch,
):
    case = SimpleNamespace(manifest={
        "runtime": {
            "scheduler_pipeline_depth_evidence": {
                "source": "test",
            },
        },
    })
    cell = SimpleNamespace(cell_id="metax_c500/qwen3-4b/tp1_qps16")
    monkeypatch.setattr(
        scheduler_depth_breakdown,
        "_prepare_final_serving_quantile_case",
        lambda *_args: (object(), cell, case, {}, ["runner"]),
    )
    monkeypatch.setattr(
        scheduler_depth_breakdown.compact,
        "scheduler_pipeline_depth_from_validated_case",
        lambda value: 2 if value is case else 0,
    )
    monkeypatch.setattr(
        scheduler_depth_breakdown.compact,
        "evaluate_validated_case",
        lambda *_args, **_kwargs: {"cell_id": cell.cell_id},
    )

    cache_path = tmp_path / "cell.json"
    result = scheduler_depth_breakdown._run_one_final_serving_quantile_case(
        "/repo",
        "/repo/config.json",
        "/repo/case",
        cell.cell_id,
        "/repo/host.json",
        "/repo/scheduler_host.json",
        "a" * 40,
        lambda _path: case,
        str(cache_path),
    )

    expected_depth = {
        "configured_depth": 2,
        "resolved_depth": 2,
        "mode": "dataset_native",
        "evidence": {"source": "test"},
    }
    assert result["depth_identity"] == expected_depth
    persisted = json.loads(cache_path.read_text())
    assert persisted["schema_version"] == 3
    assert persisted["cell"]["depth_identity"] == expected_depth


def test_full_structure_reads_current_nested_host_timing_schema():
    aggregate = {
        "step_count": 3,
        "predicted_device_s": 1.0,
        "launch_cpu_s": 2.0,
        "scheduler_cpu_s": 3.0,
        "schedule_makespan_s": 4.0,
    }

    assert scheduler_depth_breakdown._full_structure(
        {"host_timing": {"aggregate": aggregate}}
    ) == aggregate


def test_full_structure_missing_host_timing_fails_closed():
    with pytest.raises(KeyError, match="host_timing"):
        scheduler_depth_breakdown._full_structure({})


def test_full_metric_comparison_reads_current_nested_schema():
    full = {
        "metrics": {
            "throughput": {
                "request_throughput_per_s": {"observed": 3.0, "predicted": 4.0}
            },
            "latency": {
                "ttft_ms": {"mean": {"observed": 5.0, "predicted": 6.0}},
                "tpot_ms": {"mean": {"observed": 7.0, "predicted": 8.0}},
            },
        }
    }

    assert scheduling_trajectory_replay_report._full_metric_comparison(
        full, "request_throughput_per_s"
    ) == {"observed": 3.0, "predicted": 4.0}
    assert scheduling_trajectory_replay_report._full_metric_comparison(
        full, "tpot_ms"
    ) == {"observed": 7.0, "predicted": 8.0}


def test_full_metric_comparison_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unsupported full Serving plot metric"):
        scheduling_trajectory_replay_report._full_metric_comparison(
            {"metrics": {}}, "e2e_latency_ms"
        )


def test_projects_schema_v4_rank_observation_trace_without_duration(tmp_path):
    source = tmp_path / "scheduler_steps.jsonl"
    projected = tmp_path / "projected.jsonl"
    row = {
        "schema_version": 4,
        "step_index": 0,
        "step_id": "vllm:1",
        "source_iteration_id": 1,
        "step_start_ns": 10,
        "schedule_end_ns": 20,
        "update_start_ns": 30,
        "step_end_ns": 40,
        "scheduled_requests": [{
            "request_idx": 0,
            "prefill_tokens": 0,
            "decode_tokens": 1,
            "scheduled_tokens": 1,
        }],
        "scheduled_request_count": 1,
        "prefill_request_count": 0,
        "decode_request_count": 1,
        "prefill_tokens": 0,
        "decode_tokens": 1,
        "scheduled_tokens": 1,
        "finished_request_indexes": [],
        "cumulative_scheduled_tokens": 1,
        "cumulative_completed_requests": 0,
        "schedule_start_ns": 11,
        "scheduler_update_start_ns": 30,
        "scheduler_update_end_ns": 31,
        "device_execution_start_ns": 21,
        "device_execution_end_ns": 29,
        "device_execution_clock": "host_monotonic",
        "device_execution_source": "executor_rpc",
        "in_flight_step_count": 1,
        "rank_observations": [{
            "schema_version": 1,
            "rank": 0,
            "trace_step_id": "vllm:1",
            "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
            "runtime_cudagraph_mode": "FULL",
            "dispatch_input_key": {
                "num_tokens": 1,
                "uniform_decode": True,
            },
            "selected_graph_key": {
                "num_tokens": 1,
                "uniform_decode": True,
            },
            "scheduled_tokens": 1,
            "num_unpadded_tokens": 1,
            "padded_total_tokens": 1,
            "padding_tokens": 0,
            "graph_fallback_reason": None,
            "worker_execution_clock": "time.perf_counter_ns",
            "worker_execution_start_ns": 21,
            "worker_execution_end_ns": 29,
            "worker_execution_source": "vllm_v1_worker_execute_model_envelope",
            "source": "vllm_v1_worker_cudagraph_dispatch",
        }],
    }
    source.write_text(json.dumps(row) + "\n")

    with projected.open("w", encoding="utf-8") as destination:
        summary = scheduler_depth_breakdown._project_v5_scheduler_trace(
            source,
            destination,
            [{"output_tokens": 2}],
        )

    projected_row = json.loads(projected.read_text())
    assert projected_row == row
    assert projected_row["step_id"] == "vllm:1"
    assert summary["scheduler_schema_version"] == 4
    assert summary["decode_work_token_offset"] == 0
    assert summary["phase_step_count"] == {
        "prefill": 0,
        "mixed": 0,
        "decode": 1,
    }


def test_projects_compressed_schema_v4_trace(tmp_path):
    source = tmp_path / "scheduler_steps.jsonl.zst"
    projected = tmp_path / "projected.jsonl"
    row = {
        "schema_version": 4,
        "step_index": 0,
        "step_id": "vllm:1",
        "source_iteration_id": 1,
        "step_start_ns": 10,
        "schedule_end_ns": 20,
        "update_start_ns": 30,
        "step_end_ns": 40,
        "scheduled_requests": [{
            "request_idx": 0,
            "prefill_tokens": 0,
            "decode_tokens": 1,
            "scheduled_tokens": 1,
        }],
        "scheduled_request_count": 1,
        "prefill_request_count": 0,
        "decode_request_count": 1,
        "prefill_tokens": 0,
        "decode_tokens": 1,
        "scheduled_tokens": 1,
        "finished_request_indexes": [],
        "cumulative_scheduled_tokens": 1,
        "cumulative_completed_requests": 0,
        "schedule_start_ns": 11,
        "scheduler_update_start_ns": 30,
        "scheduler_update_end_ns": 31,
        "device_execution_start_ns": 21,
        "device_execution_end_ns": 29,
        "device_execution_clock": "host_monotonic",
        "device_execution_source": "executor_rpc",
        "in_flight_step_count": 1,
        "rank_observations": [{
            "schema_version": 1,
            "rank": 0,
            "trace_step_id": "vllm:1",
            "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
            "runtime_cudagraph_mode": "FULL",
            "dispatch_input_key": {
                "num_tokens": 1,
                "uniform_decode": True,
            },
            "selected_graph_key": {
                "num_tokens": 1,
                "uniform_decode": True,
            },
            "scheduled_tokens": 1,
            "num_unpadded_tokens": 1,
            "padded_total_tokens": 1,
            "padding_tokens": 0,
            "graph_fallback_reason": None,
            "worker_execution_clock": "time.perf_counter_ns",
            "worker_execution_start_ns": 21,
            "worker_execution_end_ns": 29,
            "worker_execution_source": "vllm_v1_worker_execute_model_envelope",
            "source": "vllm_v1_worker_cudagraph_dispatch",
        }],
    }
    source.write_bytes(zstandard.ZstdCompressor().compress(
        (json.dumps(row) + "\n").encode("utf-8")
    ))

    with projected.open("w", encoding="utf-8") as destination:
        summary = scheduler_depth_breakdown._project_v5_scheduler_trace(
            source,
            destination,
            [{"output_tokens": 2}],
        )

    assert json.loads(projected.read_text()) == row
    assert summary["scheduler_schema_version"] == 4
    assert summary["decode_work_token_offset"] == 0
