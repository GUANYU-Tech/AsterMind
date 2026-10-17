"""Focused schema-v2 tests for the compact ShareGPT candidate boundary."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest

from dataset_pipeline.serving_dataset.compact_candidate_writer import (
    AvailabilityPayload,
    CompactCasePayload,
    write_compact_case_candidate,
)
from dataset_pipeline.serving_dataset.compact_case import (
    ALIGNMENT_FILENAME,
    CANONICAL_REQUEST_COLUMNS,
    CanonicalRequestRow,
    _validate_launch_signature,
    load_compact_case,
    scheduler_pipeline_depth_from_compact_case,
    scheduler_pipeline_depths_from_compact_case,
    signature_id,
)
from dataset_pipeline.serving_dataset.compressed_artifact import (
    compress_canonical_bytes,
    read_compressed_canonical_bytes,
)
from dataset_pipeline.serving_dataset.scheduler_capture_candidate import (
    build_scheduler_capture_candidate,
)
from dataset_pipeline.serving_dataset.scheduler_depth_migration import (
    migrate_scheduler_depth_metadata_case,
)
from dataset_pipeline.serving_dataset.compact_case import (
    _SCHEDULER_FIELD_PROVENANCE,
)


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_alignment(case_dir: Path) -> dict[str, object]:
    manifest = json.loads((case_dir / "manifest.json").read_text())
    entry = manifest["alignment_steps"]
    content = read_compressed_canonical_bytes(
        case_dir / entry["path"],
        compressed_sha256=entry["compressed_sha256"],
        content_sha256=entry["sha256"],
        label=ALIGNMENT_FILENAME,
    )
    return json.loads(content)


def _write_alignment(case_dir: Path, alignment: object) -> None:
    content = json.dumps(alignment).encode("utf-8")
    compressed = compress_canonical_bytes(content)
    (case_dir / ALIGNMENT_FILENAME).write_bytes(compressed)
    manifest_path = case_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["alignment_steps"]["sha256"] = hashlib.sha256(content).hexdigest()
    manifest["alignment_steps"]["compressed_sha256"] = hashlib.sha256(
        compressed).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def _raw_source() -> dict[str, object]:
    return {
        "archive_id": "raw-a",
        "role": "serving-runtime",
        "relative_path": "capture/metrics.json",
        "bytes": 17,
        "sha256": "a" * 64,
        "tool": "vllm",
        "tool_version": "0.1",
    }


def _payload() -> CompactCasePayload:
    operator = {"operation": "flash_attn", "attrs": {"head_dim": 128}}
    launch = {
        "kernel_name": "flash_fwd",
        "geometry": {
            "kind": "cuda_dim3",
            "grid": [2, 1, 1],
            "block": [256, 1, 1],
            "shared_memory_bytes": 0,
            "registers_per_thread": 64,
        },
        "stream": 7,
        "device": 0,
        "op_family": "attention",
    }
    operator_id = signature_id(operator)
    launch_id = signature_id(launch)
    operator_events = [{"order": 0, "signature_id": operator_id, "layer": 0}]
    launch_events = [{"order": 0, "signature_id": launch_id}]
    source_set_sha = _sha([_raw_source()])
    sample = {
        "source_step_id": "iteration-9",
        "source_execution_order": 9,
        "scheduler": asdict(ScheduleStep(
            step_id="step-9",
            pool_name="main",
            requests=(ScheduledRequest("request-000000000000", "prefill", 2, 0),),
        )),
        "scheduler_field_provenance": _SCHEDULER_FIELD_PROVENANCE,
        "rank_ids": [0],
        "operators": [{"rank": 0, "events": operator_events}],
        "launches": [{"rank": 0, "events": launch_events}],
        "association": "step_order_only",
        "coverage": "complete",
        "provenance": {
            "source_archive_ids": ["raw-a"],
            "raw_source_set_sha256": source_set_sha,
        },
        "sequence_sha256": {
            "operators": {"0": _sha(operator_events)},
            "launches": {"0": _sha(launch_events)},
        },
    }
    request_provenance = {
        field: {"kind": "observed_actual", "source_archive_id": "raw-a", "reason": None}
        for field in CANONICAL_REQUEST_COLUMNS
    }
    return CompactCasePayload(
        identity={
            "source_kind": "sharegpt",
            "case_id": "tp1_qps4",
            "hardware_id": "nvidia_a800",
            "model_id": "qwen3-4b",
            "raw_model_id": "Qwen3-4B",
            "model_revision": None,
            "tokenizer_revision": None,
        },
        runtime={
            "dtype": "bfloat16",
            "deployment": "vllm",
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "target_qps": 4,
            "seed": 1,
            "scheduler_config_sha256": None,
            "runtime_version": "0.10",
            "source_commit": "0123456",
            "capture_enable_prefix_caching": False,
        },
        requests=(
            CanonicalRequestRow(0.0, 2, 1, 0),
            CanonicalRequestRow(1.0, 3, 2, 0),
        ),
        request_field_provenance=request_provenance,
        cached_hit_value_evidence=None,
        metrics=AvailabilityPayload(
            "available",
            None,
            {
                "metric_provenance": {
                    "source_kind": "serving_runtime",
                    "source_archive_id": "raw-a",
                    "source_sha256": "a" * 64,
                    "profiler_attached": False,
                },
                "per_request": [
                    {
                        "request_idx": 0,
                        "actual_output_tokens": 1,
                        "ttft_ms": 3.0,
                        "tpot_ms": 0.0,
                        "e2e_latency_ms": 3.0,
                    },
                    {
                        "request_idx": 1,
                        "actual_output_tokens": 2,
                        "ttft_ms": 2.0,
                        "tpot_ms": 1.0,
                        "e2e_latency_ms": 3.0,
                    },
                ],
                "summary": {
                    "request_count": 2,
                    "request_throughput_per_s": 1.0,
                    "output_token_throughput_per_s": 1.0,
                    "total_token_throughput_per_s": 2.0,
                },
            },
        ),
        alignment_steps={
            "operator_signatures": {operator_id: operator},
            "launch_signatures": {launch_id: launch},
            "phases": {
                "prefill": {"status": "available", "sample": sample},
                "decode": {"status": "not_observed", "sample": None},
                "mixed": {"status": "not_observed", "sample": None},
            },
        },
        raw_sources=(_raw_source(),),
        provenance_gaps=(),
        converter={
            "implementation_id": "midlayer.sharegpt_compactor.v2",
            "source_commit": "a" * 40,
            "selection_rule": "middle_complete_work_bearing_step_v1",
        },
    )


def _v3_payload() -> CompactCasePayload:
    return replace(
        _payload(),
        schema_version=3,
        scheduler_steps=(
            {
                "step_index": 0,
                "step_id": "iteration-9",
                "source_iteration_id": 9,
                "step_start_ns": 0,
                "schedule_start_ns": 0,
                "schedule_end_ns": 100,
                "device_execution_start_ns": 200,
                "device_execution_end_ns": 800,
                "update_start_ns": 900,
                "scheduler_update_start_ns": 900,
                "scheduler_update_end_ns": 1000,
                "step_end_ns": 1000,
                "scheduled_requests": [
                    {"request_idx": 0, "prefill_tokens": 2, "decode_tokens": 0, "scheduled_tokens": 2},
                    {"request_idx": 1, "prefill_tokens": 3, "decode_tokens": 0, "scheduled_tokens": 3},
                ],
                "scheduled_request_count": 2,
                "prefill_request_count": 2,
                "decode_request_count": 0,
                "prefill_tokens": 5,
                "decode_tokens": 0,
                "scheduled_tokens": 5,
                "num_unpadded_tokens": 5,
                "capture_bucket_tokens": 8,
                "capture_limit_tokens": 512,
                "padded_total_tokens": 8,
                "padding_tokens": 3,
                "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
                "runtime_cudagraph_mode": "FULL",
                "graph_fallback_reason": None,
                "device_execution_clock": "time.perf_counter_ns",
                "device_execution_source": "host_monotonic_executor_rpc_envelope",
                "in_flight_step_count": 1,
                "finished_request_indexes": [0],
                "cumulative_scheduled_tokens": 5,
                "cumulative_completed_requests": 1,
            },
            {
                "step_index": 1,
                "step_id": "iteration-10",
                "source_iteration_id": 10,
                "step_start_ns": 500,
                "schedule_start_ns": 500,
                "schedule_end_ns": 600,
                "device_execution_start_ns": 700,
                "device_execution_end_ns": 1800,
                "update_start_ns": 1900,
                "scheduler_update_start_ns": 1900,
                "scheduler_update_end_ns": 2000,
                "step_end_ns": 2000,
                "scheduled_requests": [
                    {"request_idx": 1, "prefill_tokens": 0, "decode_tokens": 1, "scheduled_tokens": 1},
                ],
                "scheduled_request_count": 1,
                "prefill_request_count": 0,
                "decode_request_count": 1,
                "prefill_tokens": 0,
                "decode_tokens": 1,
                "scheduled_tokens": 1,
                "num_unpadded_tokens": 1,
                "capture_bucket_tokens": 1,
                "capture_limit_tokens": 512,
                "padded_total_tokens": 1,
                "padding_tokens": 0,
                "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
                "runtime_cudagraph_mode": "PIECEWISE",
                "graph_fallback_reason": None,
                "device_execution_clock": "time.perf_counter_ns",
                "device_execution_source": "host_monotonic_executor_rpc_envelope",
                "in_flight_step_count": 2,
                "finished_request_indexes": [1],
                "cumulative_scheduled_tokens": 6,
                "cumulative_completed_requests": 2,
            },
        ),
        scheduler_timing_boundary={
            "clock": "time.perf_counter_ns",
            "origin": "first_scheduler_step_start",
            "step_start": "immediately_before_scheduler_schedule",
            "schedule_end": "immediately_after_scheduler_schedule",
            "update_start": "immediately_before_scheduler_output_update",
            "step_end": "immediately_after_scheduler_output_update",
        },
        scheduler_provenance={
            "source_archive_id": "raw-a",
            "source_sha256": "a" * 64,
            "capture_run_id": "trace-on-run",
            "instrumentation_id": "midlayer-vllm-scheduler-trace",
            "instrumentation_version": 1,
            "buffering": "in_memory_append_then_single_write",
            "profiler_attached": False,
        },
        capture={
            "run_id": "trace-on-run",
            "requests_source_archive_ids": ["raw-a"],
            "metrics_source_archive_id": "raw-a",
            "scheduler_source_archive_id": "raw-a",
            "scheduler_trace_enabled": True,
            "profiler_attached": False,
            "observer_effect_validation": {
                "status": "measured_for_this_case",
                "validation_case_id": "nvidia_a800/qwen3-4b/tp1_qps4",
                "baseline_run_id": "trace-off-run",
                "instrumented_run_id": "trace-on-run",
                "threshold_percent": 3.0,
                "metric_delta_percent": {
                    "request_throughput": 0.1,
                    "ttft": 0.2,
                    "tpot": 0.3,
                    "e2e": 0.4,
                },
                "passed": True,
            },
        },
    )


def _v4_payload() -> CompactCasePayload:
    flat_runtime_fields = {
        "configured_cudagraph_mode", "runtime_cudagraph_mode",
        "num_unpadded_tokens", "capture_bucket_tokens",
        "capture_limit_tokens", "padded_total_tokens", "padding_tokens",
        "graph_fallback_reason", "schedule_start_ns",
        "device_execution_start_ns", "device_execution_end_ns",
        "device_execution_clock", "device_execution_source",
        "scheduler_update_start_ns", "scheduler_update_end_ns",
        "in_flight_step_count",
    }
    legacy_steps = [
        {
            key: value
            for key, value in step.items()
            if key not in flat_runtime_fields
        }
        for step in _v3_payload().scheduler_steps
    ]
    legacy_steps[0] = {
        **legacy_steps[0],
        "finished_request_indexes": [],
        "cumulative_completed_requests": 0,
    }
    legacy_steps[1] = {
        **legacy_steps[1],
        "scheduled_requests": [
            {"request_idx": 0, "prefill_tokens": 0, "decode_tokens": 1, "scheduled_tokens": 1},
            {"request_idx": 1, "prefill_tokens": 0, "decode_tokens": 1, "scheduled_tokens": 1},
        ],
        "scheduled_request_count": 2,
        "decode_request_count": 2,
        "decode_tokens": 2,
        "scheduled_tokens": 2,
        "finished_request_indexes": [0],
        "cumulative_scheduled_tokens": 7,
        "cumulative_completed_requests": 1,
    }
    legacy_steps.append({
        "step_index": 2,
        "step_id": "iteration-11",
        "source_iteration_id": 11,
        "step_start_ns": 2000,
        "schedule_end_ns": 2100,
        "update_start_ns": 2900,
        "step_end_ns": 3000,
        "scheduled_requests": [
            {"request_idx": 1, "prefill_tokens": 0, "decode_tokens": 1, "scheduled_tokens": 1},
        ],
        "scheduled_request_count": 1,
        "prefill_request_count": 0,
        "decode_request_count": 1,
        "prefill_tokens": 0,
        "decode_tokens": 1,
        "scheduled_tokens": 1,
        "finished_request_indexes": [1],
        "cumulative_scheduled_tokens": 8,
        "cumulative_completed_requests": 2,
    })
    runtime_steps = []
    for step in legacy_steps:
        runtime_step = dict(step)
        runtime_step.update({
            "schedule_start_ns": step["step_start_ns"],
            "scheduler_update_start_ns": step["update_start_ns"],
            "scheduler_update_end_ns": step["step_end_ns"],
            "in_flight_step_count": min(step["step_index"] + 1, 2),
            "device_execution_clock": "time.perf_counter_ns",
            "device_execution_start_ns": step["schedule_end_ns"],
            "device_execution_end_ns": step["update_start_ns"],
            "device_execution_source": "host_monotonic_executor_rpc_envelope",
        })
        if step["step_index"] == 0:
            mode = "NONE"
            selected_graph_key = None
            fallback = "runtime_dispatch_no_matching_graph"
            dispatch_key = {"num_tokens": 5, "uniform_decode": False}
        else:
            mode = "FULL"
            selected_graph_key = {"num_tokens": 1, "uniform_decode": True}
            fallback = None
            dispatch_key = {"num_tokens": 1, "uniform_decode": True}
        runtime_step["rank_observations"] = [{
            "schema_version": 1,
            "rank": 0,
            "trace_step_id": step["step_id"],
            "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
            "runtime_cudagraph_mode": mode,
            "dispatch_input_key": dispatch_key,
            "selected_graph_key": selected_graph_key,
            "scheduled_tokens": step["scheduled_tokens"],
            "num_unpadded_tokens": step["scheduled_tokens"],
            "padded_total_tokens": step["scheduled_tokens"],
            "padding_tokens": 0,
            "graph_fallback_reason": fallback,
            "worker_execution_clock": "time.perf_counter_ns",
            "worker_execution_start_ns": step["schedule_end_ns"],
            "worker_execution_end_ns": step["update_start_ns"],
            "worker_execution_source": "vllm_v1_worker_execute_model_envelope",
            "source": "vllm_v1_worker_cudagraph_dispatch",
        }]
        runtime_steps.append(runtime_step)
    return replace(_v3_payload(), schema_version=4, scheduler_steps=tuple(runtime_steps))


def _v5_payload(depth: int = 3) -> CompactCasePayload:
    runtime = {
        **_v4_payload().runtime,
        "scheduler_pipeline_depth": depth,
        "scheduler_pipeline_depth_evidence": {
            "source_kind": "runtime_executor_property",
            "source_field": "max_concurrent_batches",
            "executor_class": "vllm.v1.executor.multiproc_executor.MultiprocExecutor",
            "pipeline_parallel_size": 1,
            "runtime_version": "0.10",
            "source_identity": "0123456",
        },
    }
    return replace(
        _v4_payload(),
        schema_version=5,
        runtime=runtime,
    )


def test_v2_writer_round_trip_restricts_case_files(monkeypatch, tmp_path):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    written = write_compact_case_candidate(destination, _payload())
    case = load_compact_case(written)
    assert case.manifest["schema_version"] == 2
    assert case.alignment_steps["prefill"] is not None
    assert case.alignment_steps["decode"] is None
    assert sorted(path.name for path in written.iterdir()) == [
        "alignment_steps.json.zst",
        "manifest.json",
        "metrics.json",
        "requests.csv",
    ]


def test_v3_writer_round_trip_requires_same_run_scheduler_steps(
    monkeypatch, tmp_path
):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    written = write_compact_case_candidate(destination, _v3_payload())
    case = load_compact_case(written)

    assert case.manifest["schema_version"] == 3
    assert len(case.scheduler_steps) == 2
    assert sorted(path.name for path in written.iterdir()) == [
        "alignment_steps.json.zst",
        "manifest.json",
        "metrics.json",
        "requests.csv",
        "scheduler_steps.jsonl.zst",
    ]


def test_v4_writer_round_trip_preserves_runtime_graph_observations(
    monkeypatch, tmp_path
):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    written = write_compact_case_candidate(destination, _v4_payload())
    case = load_compact_case(written)

    assert case.manifest["schema_version"] == 4
    assert case.scheduler_steps[0]["rank_observations"][0][
        "runtime_cudagraph_mode"
    ] == "NONE"


def test_v5_configured_depth_is_independent_from_unsaturated_observation(
        monkeypatch, tmp_path):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    written = write_compact_case_candidate(destination, _v5_payload(3))
    case = load_compact_case(written)

    assert scheduler_pipeline_depth_from_compact_case(case) == 3
    assert max(row["in_flight_step_count"] for row in case.scheduler_steps) == 2


@pytest.mark.parametrize(
    ("runtime_mutation", "message"),
    (
        ({"scheduler_pipeline_depth": True}, "scheduler_pipeline_depth"),
        ({"scheduler_pipeline_depth": 0}, "scheduler_pipeline_depth"),
        ({"scheduler_pipeline_depth_evidence": {
            "source_kind": "runtime_executor_property",
            "source_field": "max_concurrent_batches",
            "executor_class": "vllm.v1.executor.multiproc_executor.MultiprocExecutor",
            "pipeline_parallel_size": 2,
            "runtime_version": "0.10",
            "source_identity": "0123456",
        }}, "PP disagrees"),
    ),
)
def test_v5_writer_rejects_invalid_or_conflicting_depth_evidence(
        monkeypatch, tmp_path, runtime_mutation, message):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    payload = _v5_payload()
    runtime = {**payload.runtime, **runtime_mutation}

    with pytest.raises((TypeError, ValueError), match=message):
        write_compact_case_candidate(destination, replace(payload, runtime=runtime))


def test_old_compact_case_rejects_depth_resolution(monkeypatch, tmp_path):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    written = write_compact_case_candidate(destination, _v4_payload())

    with pytest.raises(ValueError, match="predates configured Scheduler depth"):
        scheduler_pipeline_depth_from_compact_case(load_compact_case(written))


def test_v5_pd_depths_resolve_independently_by_pool(monkeypatch, tmp_path):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    base = _v5_payload()
    evidence = base.runtime["scheduler_pipeline_depth_evidence"]
    runtime = {
        **base.runtime,
        "deployment": "pd_disagg",
        "instances": {
            "prefill": {
                "scheduler_pipeline_depth": 1,
                "scheduler_pipeline_depth_evidence": evidence,
            },
            "decode": {
                "scheduler_pipeline_depth": 3,
                "scheduler_pipeline_depth_evidence": evidence,
            },
        },
    }
    del runtime["scheduler_pipeline_depth"]
    del runtime["scheduler_pipeline_depth_evidence"]
    written = write_compact_case_candidate(
        destination, replace(base, runtime=runtime))
    case = load_compact_case(written)

    assert scheduler_pipeline_depths_from_compact_case(case, {
        "prefill-0": "prefill",
        "decode-0": "decode",
    }) == {"prefill-0": 1, "decode-0": 3}
    with pytest.raises(ValueError, match="per-pool"):
        scheduler_pipeline_depth_from_compact_case(case)


@pytest.mark.parametrize(
    ("source_payload", "scheduler_step_schema_version"),
    ((_v3_payload, 3), (_v4_payload, 4)),
)
def test_metadata_migration_reads_raw_executor_capacity(
    monkeypatch, tmp_path, source_payload, scheduler_step_schema_version,
):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    canonical_root = tmp_path / "canonical/profiler_output/serving_dataset_compaction"
    canonical_destination = canonical_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    canonical_destination.parent.mkdir(parents=True)
    candidate_root.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", canonical_root)
    source_case = write_compact_case_candidate(
        canonical_destination, source_payload())
    raw_path = tmp_path / "ground_truth.json"
    raw_path.write_text(json.dumps({
        "server": {
            "client_vllm_version": "0.10",
            "engine": {
                "max_concurrent_batches": 2,
                "pipeline_parallel_size": 1,
            },
        },
    }))
    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)

    result = migrate_scheduler_depth_metadata_case(
        canonical_case_dir=source_case,
        raw_runtime_path=raw_path,
        executor_class="vllm.v1.executor.multiproc_executor.MultiprocExecutor",
        source_identity="a800-vllm-source-identity",
        candidate_root=candidate_root,
        converter_source_commit="b" * 40,
    )

    migrated = load_compact_case(result.candidate_dir)
    assert scheduler_pipeline_depth_from_compact_case(migrated) == 2
    assert result.candidate_dir == (
        candidate_root
        / "sharegpt_scheduler_depth_metadata/nvidia_a800/qwen3-4b/tp1_qps4"
    )
    assert result.source_file_sha256["requests.csv"] == (
        result.candidate_file_sha256["requests.csv"])
    assert migrated.scheduler_steps[0]["schema_version"] == (
        scheduler_step_schema_version)


def test_v3_writer_rejects_scheduler_token_nonconservation(monkeypatch, tmp_path):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    payload = _v3_payload()
    steps = [dict(step) for step in payload.scheduler_steps]
    steps[1]["scheduled_requests"] = [
        {"request_idx": 1, "prefill_tokens": 0, "decode_tokens": 2, "scheduled_tokens": 2}
    ]
    steps[1]["decode_tokens"] = 2
    steps[1]["scheduled_tokens"] = 2
    steps[1]["num_unpadded_tokens"] = 2
    steps[1]["capture_bucket_tokens"] = 2
    steps[1]["padded_total_tokens"] = 2
    steps[1]["cumulative_scheduled_tokens"] = 7

    with pytest.raises(ValueError, match="token conservation failed"):
        write_compact_case_candidate(
            destination, replace(payload, scheduler_steps=tuple(steps))
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("num_unpadded_tokens", 4, "num_unpadded_tokens"),
        ("padded_total_tokens", 7, "padded_total_tokens"),
        ("runtime_cudagraph_mode", "FULL_AND_PIECEWISE", "runtime_cudagraph_mode"),
        ("device_execution_end_ns", 950, "execution timing boundaries"),
        ("in_flight_step_count", 3, "in-flight depth"),
    ),
)
def test_v3_writer_rejects_invalid_runtime_scheduler_evidence(
    monkeypatch, tmp_path, field, value, message
):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    payload = _v3_payload()
    steps = [dict(step) for step in payload.scheduler_steps]
    steps[0][field] = value

    with pytest.raises(ValueError, match=message):
        write_compact_case_candidate(
            destination, replace(payload, scheduler_steps=tuple(steps))
        )


def test_scheduler_capture_candidate_uses_trace_on_requests_metrics_and_steps(
    monkeypatch, tmp_path
):
    source_root = tmp_path / "source/profiler_output/serving_dataset_compaction"
    source_case = source_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    source_case.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", source_root)
    write_compact_case_candidate(source_case, _payload())
    capture_dir = tmp_path / "capture" / "trace-on"
    capture_dir.mkdir(parents=True)
    scheduler_path = capture_dir / "scheduler_steps.raw.jsonl"
    scheduler_path.write_text("\n".join(
        json.dumps(step, sort_keys=True, separators=(",", ":"))
        for step in _v4_payload().scheduler_steps
    ) + "\n")
    runtime_metadata_path = Path(f"{scheduler_path}.runtime.json")
    runtime_metadata_path.write_text(json.dumps({
        "schema_version": 1,
        "scheduler_pipeline_depth": 2,
        "scheduler_pipeline_depth_evidence": {
            "source_kind": "runtime_executor_property",
            "source_field": "max_concurrent_batches",
            "executor_class": "vllm.v1.executor.multiproc_executor.MultiprocExecutor",
            "pipeline_parallel_size": 1,
            "runtime_version": "0.11.0",
            "source_identity": "0123456",
        },
    }) + "\n")
    trace_on_path = capture_dir / "ground_truth.json"
    trace_on = {
        "schema_version": 1,
        "status": "success",
        "workload": {
            "identity": {
                "hardware_id": "nvidia_a800",
                "model_id": "qwen3-4b",
                "case_id": "tp1_qps4",
            },
        },
        "server": {
            "client_vllm_version": "0.11.0",
            "profiler": {"requested": False},
            "engine": {
                "dtype": "bfloat16",
                "seed": 1,
                "enable_prefix_caching": True,
            },
            "scheduler_trace": {
                "instrumentation_id": "midlayer-vllm-scheduler-trace",
                "instrumentation_version": 2,
            },
        },
        "metrics": {
            "duration_s": 2.0,
            "request_throughput_per_s": 1.0,
            "output_token_throughput_per_s": 1.5,
            "per_request": [
                {
                    "request_id": "request-000000000000",
                    "scheduled_arrival_s": 0.0,
                    "submit_offset_s": 0.001,
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "ttft_ms": 3.0,
                    "tpot_ms": 0.0,
                    "e2e_ms": 3.0,
                },
                {
                    "request_id": "request-000000000001",
                    "scheduled_arrival_s": 0.001,
                    "submit_offset_s": 0.0,
                    "input_tokens": 3,
                    "output_tokens": 2,
                    "ttft_ms": 2.0,
                    "tpot_ms": 1.0,
                    "e2e_ms": 3.0,
                },
            ],
        },
        "scheduler_trace": {
            "path": str(scheduler_path.resolve()),
            "sha256": hashlib.sha256(scheduler_path.read_bytes()).hexdigest(),
            "row_count": 2,
        },
    }
    trace_on_path.write_text(json.dumps(trace_on) + "\n")
    observer_path = tmp_path / "capture" / "observer_effect.json"
    observer_path.write_text(json.dumps({
        "kind": "scheduler_trace_observer_effect",
        "validation_case_id": "nvidia_a800/qwen3-4b/tp1_qps4",
        "trace_off": {"run_id": "trace-off/ground_truth.json"},
        "trace_on": {
            "run_id": "trace-on/ground_truth.json",
            "sha256": hashlib.sha256(trace_on_path.read_bytes()).hexdigest(),
        },
        "threshold_percent": 3.0,
        "metric_delta_percent": {
            "request_throughput": 0.1,
            "output_token_throughput": 0.1,
            "ttft": 0.2,
            "tpot": 0.3,
            "e2e": 0.4,
        },
        "passed": True,
    }) + "\n")
    candidate_root = (
        tmp_path / "candidate/profiler_output/serving_dataset_compaction"
    )
    candidate_root.mkdir(parents=True)

    written = build_scheduler_capture_candidate(
        canonical_case_dir=source_case,
        trace_on_result_path=trace_on_path,
        scheduler_trace_path=scheduler_path,
        observer_effect_path=observer_path,
        candidate_root=candidate_root,
        converter_source_commit="b" * 40,
    )
    case = load_compact_case(written)

    assert (written / "requests.csv").read_bytes() == (
        source_case / "requests.csv"
    ).read_bytes()
    assert case.requests[0].input_tokens == 2
    assert case.requests[0].arrival_time_ms == 0.0
    assert scheduler_pipeline_depth_from_compact_case(case) == 2
    assert case.scheduler_steps[0]["scheduled_requests"][0]["request_idx"] == 0
    assert case.manifest["runtime"]["dtype"] == "bfloat16"
    assert case.manifest["runtime"]["seed"] == 1
    assert case.manifest["runtime"]["capture_enable_prefix_caching"] is True
    assert case.manifest["scheduler_steps"]["provenance"][
        "instrumentation_version"
    ] == 2
    assert case.manifest["capture"]["run_id"] == "trace-on/ground_truth.json"
    assert case.manifest["capture"]["observer_effect_validation"]["status"] == (
        "measured_for_this_case"
    )

    embedded_scheduler_path = capture_dir / "scheduler_steps.embedded.jsonl"
    embedded_depth_evidence = {
        "source_kind": "runtime_vllm_config_field",
        "source_field": "vllm_config.max_concurrent_batches",
        "configured_depth": 2,
        "executor_class": "vllm.v1.executor.uniproc_executor.UniProcExecutor",
        "pipeline_parallel_size": 1,
        "runtime_version": "0.11.0",
        "source_identity": {
            "module_file": "/opt/vllm/config/vllm.py",
            "module_name": "vllm.v1.executor.uniproc_executor",
        },
    }
    embedded_scheduler_path.write_text("\n".join(
        json.dumps(
            {**step, "scheduler_pipeline_depth_evidence": embedded_depth_evidence},
            sort_keys=True,
            separators=(",", ":"),
        )
        for step in _v4_payload().scheduler_steps
    ) + "\n")
    trace_on["scheduler_trace"]["path"] = str(embedded_scheduler_path.resolve())
    trace_on["scheduler_trace"]["sha256"] = hashlib.sha256(
        embedded_scheduler_path.read_bytes()
    ).hexdigest()
    trace_on_path.write_text(json.dumps(trace_on) + "\n")
    observer = json.loads(observer_path.read_text())
    observer["trace_on"]["sha256"] = hashlib.sha256(
        trace_on_path.read_bytes()
    ).hexdigest()
    observer_path.write_text(json.dumps(observer) + "\n")
    embedded_root = (
        tmp_path / "candidate-embedded/profiler_output/serving_dataset_compaction"
    )
    embedded_root.mkdir(parents=True)
    embedded_written = build_scheduler_capture_candidate(
        canonical_case_dir=source_case,
        trace_on_result_path=trace_on_path,
        scheduler_trace_path=embedded_scheduler_path,
        observer_effect_path=observer_path,
        candidate_root=embedded_root,
        converter_source_commit="b" * 40,
    )
    embedded_case = load_compact_case(embedded_written)
    assert scheduler_pipeline_depth_from_compact_case(embedded_case) == 2
    assert embedded_case.manifest["runtime"][
        "scheduler_pipeline_depth_evidence"
    ] == embedded_depth_evidence
    assert {
        source["archive_id"] for source in embedded_case.manifest["raw_sources"]
    } >= {"accepted-ground-truth-v3", "accepted-scheduler-trace-v3"}
    assert "accepted-scheduler-runtime-depth-v1" not in {
        source["archive_id"] for source in embedded_case.manifest["raw_sources"]
    }

    trace_on["scheduler_trace"]["path"] = "/remote/capture/scheduler_steps.jsonl"
    trace_on["scheduler_trace"]["sha256"] = hashlib.sha256(
        scheduler_path.read_bytes()
    ).hexdigest()
    trace_on_path.write_text(json.dumps(trace_on) + "\n")
    observer = json.loads(observer_path.read_text())
    observer["trace_on"]["sha256"] = hashlib.sha256(
        trace_on_path.read_bytes()
    ).hexdigest()
    observer_path.write_text(json.dumps(observer) + "\n")
    relocated_root = (
        tmp_path / "candidate-relocated/profiler_output/serving_dataset_compaction"
    )
    relocated_root.mkdir(parents=True)
    assert build_scheduler_capture_candidate(
        canonical_case_dir=source_case,
        trace_on_result_path=trace_on_path,
        scheduler_trace_path=scheduler_path,
        observer_effect_path=observer_path,
        candidate_root=relocated_root,
        converter_source_commit="b" * 40,
    ).is_dir()

    inherited_observer_path = tmp_path / "capture" / "representative_observer.json"
    inherited_observer_path.write_text(json.dumps({
        "baseline_run_id": "representative-off/ground_truth.json",
        "instrumented_run_id": "representative-on/ground_truth.json",
        "validation_case_id": "nvidia_a800/qwen3-4b/tp1_qps16",
        "threshold_percent": 3.0,
        "metric_delta_percent": {
            "request_throughput": 0.1,
            "output_token_throughput": 0.1,
            "ttft": 0.2,
            "tpot": 0.3,
            "e2e": 0.4,
        },
        "passed": True,
        "status": "measured_for_this_case",
    }) + "\n")
    inherited_root = (
        tmp_path / "candidate-inherited/profiler_output/serving_dataset_compaction"
    )
    inherited_root.mkdir(parents=True)
    inherited_written = build_scheduler_capture_candidate(
        canonical_case_dir=written,
        trace_on_result_path=trace_on_path,
        scheduler_trace_path=scheduler_path,
        observer_effect_path=inherited_observer_path,
        candidate_root=inherited_root,
        converter_source_commit="b" * 40,
        observer_validation_status="inherited_from_representative_case",
    )
    inherited_case = load_compact_case(inherited_written)
    inherited_validation = inherited_case.manifest["capture"][
        "observer_effect_validation"
    ]
    assert inherited_case.manifest["capture"]["run_id"] == (
        "trace-on/ground_truth.json"
    )
    assert inherited_validation["status"] == (
        "inherited_from_representative_case"
    )
    assert inherited_validation["instrumented_run_id"] == (
        "representative-on/ground_truth.json"
    )


@pytest.mark.parametrize(
    "invalid_commit",
    (None, "", "a" * 39, "a" * 41, "A" * 40, "g" * 40, " " + "a" * 40),
)
def test_v2_loader_rejects_noncanonical_converter_commit(
    monkeypatch, tmp_path, invalid_commit
):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    written = write_compact_case_candidate(destination, _payload())
    manifest_path = written / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["converter"]["source_commit"] = invalid_commit
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        load_compact_case(written)


def test_v2_schema_limits_unavailable_geometry_to_c500():
    signature = {
        "kernel_name": "projected_kernel",
        "geometry": {"kind": "unavailable"},
        "stream": None,
        "device": None,
        "op_family": "metax_legacy_launch_projection",
    }
    _validate_launch_signature(signature, "metax_c500")
    with pytest.raises(ValueError, match="reserved for C500"):
        _validate_launch_signature(signature, "nvidia_a800")


def test_v2_writer_rejects_arbitrary_candidate_root(tmp_path):
    candidate_root = tmp_path / "scratch" / "profiler_output" / "other"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    candidate_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="exactly"):
        write_compact_case_candidate(
            destination, _payload(), candidate_root=candidate_root
        )


def test_v2_loader_rejects_timing_payload(monkeypatch, tmp_path):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    written = write_compact_case_candidate(destination, _payload())
    alignment = _read_alignment(written)
    alignment["phases"]["prefill"]["sample"]["scheduler"]["duration_us"] = 1
    _write_alignment(written, alignment)
    with pytest.raises(ValueError, match="timing field"):
        load_compact_case(written)


@pytest.mark.parametrize(
    "field",
    (
        "startNs", "endNs", "tsUs", "elapsedTime", "kernel_time_ns",
        "wall_time_ms", "device_time", "finish_time_ns", "clock_rate_hz",
    ),
)
def test_v2_loader_rejects_camel_case_timing_payload(monkeypatch, tmp_path, field):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    written = write_compact_case_candidate(destination, _payload())
    alignment = _read_alignment(written)
    alignment["phases"]["prefill"]["sample"][field] = 1
    _write_alignment(written, alignment)
    with pytest.raises(ValueError, match="timing field"):
        load_compact_case(written)


@pytest.mark.parametrize("field", ("tile_shape", "runtime_variant", "clockwise_order"))
def test_v2_loader_allows_explicit_non_timing_words(monkeypatch, tmp_path, field):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    written = write_compact_case_candidate(destination, _payload())
    alignment = _read_alignment(written)
    alignment["operator_signatures"][
        next(iter(alignment["operator_signatures"]))
    ]["attrs"][field] = 1
    signature = next(iter(alignment["operator_signatures"].values()))
    new_identifier = _sha(signature)
    alignment["operator_signatures"] = {new_identifier: signature}
    alignment["phases"]["prefill"]["sample"]["operators"][0]["events"][0][
        "signature_id"
    ] = new_identifier
    alignment["phases"]["prefill"]["sample"]["sequence_sha256"]["operators"][
        "0"
    ] = _sha(
        alignment["phases"]["prefill"]["sample"]["operators"][0]["events"]
    )
    _write_alignment(written, alignment)
    load_compact_case(written)


def test_v2_writer_omits_unavailable_metrics(monkeypatch, tmp_path):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    payload = replace(
        _payload(),
        metrics=AvailabilityPayload(
            "unavailable", "missing_non_profiler_serving_ground_truth", None
        ),
    )
    written = write_compact_case_candidate(destination, payload)
    case = load_compact_case(written)
    assert case.metrics is None
    assert not (written / "metrics.json").exists()


def test_v2_writer_rejects_unproven_cached_hit_representative(monkeypatch, tmp_path):
    candidate_root = tmp_path / "profiler_output/serving_dataset_compaction"
    destination = candidate_root / "nvidia_a800/qwen3-4b/tp1_qps4"
    destination.parent.mkdir(parents=True)
    import dataset_pipeline.serving_dataset.compact_candidate_writer as writer

    monkeypatch.setattr(writer, "_SERVING_CANDIDATE_ROOT", candidate_root)
    payload = _payload()
    field_provenance = dict(payload.request_field_provenance)
    field_provenance["Cached hit [tokens]"] = {
        "kind": "scheduler_equivalent_representative",
        "source_archive_id": None,
        "reason": None,
    }
    invalid = replace(
        payload,
        request_field_provenance=field_provenance,
        cached_hit_value_evidence=None,
    )
    with pytest.raises(TypeError, match="cached-hit value evidence"):
        write_compact_case_candidate(destination, invalid)
