import hashlib
import json
import sqlite3

import pytest

from dataset_pipeline.serving_dataset.sharegpt_raw_adapter import (
    AvailabilityPayload,
    _a800_rank_devices,
    _a800_sampler_tail_anchor_ends,
    _ascend_capture_runtime,
    _compact_alignment_steps,
    _c500_projection_sync_windows,
    _scan_c500_projection_events,
    _normalize_a800_profiler,
    _normalize_c500_projection,
    _normalize_operators,
    _normalize_profiler,
    _normalize_scheduler,
    _ordinal_request_contract,
    _provenance_gaps,
    _runtime_contract,
    _second_phase_a800_contract,
    _validate_second_phase_case_identity,
    _validate_meta,
    build_second_phase_a800_sharegpt_compact_payload,
    build_sharegpt_case_compact_payload,
    build_sharegpt_compact_payload,
)
from dataset_pipeline.serving_dataset.compact_case import raw_source_set_sha256


def _scheduler_record(iteration, timestamp, scheduled, finished=()):
    return {
        "iter_id": iteration,
        "ts_us": timestamp,
        "total_tokens": sum(scheduled.values()),
        "scheduled": scheduled,
        "finished": list(finished),
    }


def _normalized_scheduler(raw, mapping, inputs):
    return _normalize_scheduler(raw, mapping, inputs)


def _available_test_launches(scheduler):
    signature_id = "a" * 64
    signature = {
        "schema_version": 2,
        "record_type": "signature",
        "signature_id": signature_id,
        "kernel_name": "test_kernel",
        "kernel_family": "test_kernel",
        "geometry_kind": "ascend_block_dim",
        "grid": None,
        "block": None,
        "ascend_block_dim": 1,
        "ascend_mix_block_dim": 0,
        "shared_memory_bytes": None,
        "registers_per_thread": None,
        "stream": None,
        "device": None,
    }
    events = tuple(
        {
            "schema_version": 2,
            "record_type": "event",
            "launch_id": f"launch-{record['step_id']}",
            "signature_id": signature_id,
            "rank": 0,
            "step_id": record["step_id"],
            "sequence_idx": 0,
            "operator_ids": [],
            "link_status": "step_order_only",
        }
        for record in scheduler
        if record["scheduled_requests"]
    )
    return AvailabilityPayload(
        "available", "complete", None, (signature, *events), ("device-profiler",)
    )


def _alignment_scheduler_record(index):
    return {
        "step_id": f"step-{index:012d}",
        "scheduled_requests": [{
            "request_idx": 0,
            "phase": "prefill",
            "input_tokens": 8,
            "output_tokens": 1,
            "computed_tokens": None,
            "recomputed_tokens": None,
            "prefix_replay_tokens": None,
        }],
        "finished_request_idxs": (),
    }


def _alignment_operator_records(shapes):
    signatures = []
    events = []
    for index, (operation, shape) in enumerate(shapes):
        identifier = f"raw-operator-{index}"
        signatures.append({
            "record_type": "signature",
            "signature_id": identifier,
            "operator_kind": operation,
            "shape": shape,
            "dtype": "bfloat16",
            "variant": "test",
            "attrs": {},
        })
        events.append({
            "record_type": "event",
            "signature_id": identifier,
            "step_id": f"step-{index:012d}",
            "rank": 0,
            "sequence_idx": 0,
            "layer_idx": 0,
        })
    return (*signatures, *events)


def _alignment_launches(count):
    signatures = []
    events = []
    for index in range(count):
        identifier = f"raw-launch-{index}"
        signatures.append({
            "record_type": "signature",
            "signature_id": identifier,
            "kernel_name": f"kernel-{index}",
            "kernel_family": "test",
            "geometry_kind": "ascend_block_dim",
            "grid": None,
            "block": None,
            "ascend_block_dim": index + 1,
            "ascend_mix_block_dim": 0,
            "shared_memory_bytes": None,
            "registers_per_thread": None,
            "stream": None,
            "device": 0,
        })
        events.append({
            "record_type": "event",
            "signature_id": identifier,
            "step_id": f"step-{index:012d}",
            "rank": 0,
            "sequence_idx": 0,
        })
    return AvailabilityPayload(
        "available", "complete", None, (*signatures, *events), ("device-profiler",)
    )


def test_sharegpt_build_requires_raw_evidence_not_user_cache_assertion(tmp_path):
    case_dir = (
        tmp_path
        / "legacy-raw-layout"
        / "eval_serving_sharegpt"
        / "ShareGPT-Qwen3-4B_TP1_n128_qps4"
    )
    case_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="missing required"):
        build_sharegpt_compact_payload(
            case_dir,
            hardware_id="nvidia_a800",
            model_id="Qwen3-4B",
            target_qps=4,
            converter_source_commit="a" * 40,
        )


def test_sharegpt_public_builders_reject_invalid_commit_before_raw_access(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        build_sharegpt_compact_payload(
            missing,
            hardware_id="nvidia_a800",
            model_id="Qwen3-4B",
            target_qps=4,
            converter_source_commit="deadbeef",
        )
    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        build_sharegpt_case_compact_payload(
            missing,
            hardware_id="metax_c500",
            raw_model_id="Qwen3-4B",
            target_qps=4,
            tensor_parallel_size=1,
            expert_parallel_size=1,
            case_id="tp1_qps4",
            publish_metrics=True,
            converter_source_commit="A" * 40,
        )
    with pytest.raises(ValueError, match="40 lowercase hexadecimal"):
        build_second_phase_a800_sharegpt_compact_payload(
            missing,
            model_id="Qwen3-32B",
            target_qps=1,
            converter_source_commit="a" * 39,
        )


def test_ordinal_mapping_uses_first_scheduler_appearance_and_token_closure():
    metrics = (
        {"request_id": "metrics-a", "n_output_tokens": 2},
        {"request_id": "metrics-b", "n_output_tokens": 1},
    )
    scheduler = (
        _scheduler_record(0, 100, {"warmup-0": 8}),
        _scheduler_record(1, 200, {"opaque-b": 5}),
        _scheduler_record(2, 300, {"opaque-a": 3, "opaque-b": 1}),
        _scheduler_record(3, 400, {"opaque-a": 1}, ("opaque-a", "opaque-b")),
        _scheduler_record(4, 500, {}, ("warmup-0",)),
    )
    meta = {
        "model": "/models/Qwen3-4B",
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
    }
    provenance = {
        "model": "/models/Qwen3-4B",
        "qps": 4,
        "num_requests": 2,
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prompt_lengths": [5, 4],
        "output_lengths": [2, 1],
        "engine_kwargs": {"dtype": "bfloat16"},
    }

    mapping, inputs, runtime = _ordinal_request_contract(
        metrics, scheduler, provenance, meta, "Qwen3-4B", 4
    )

    assert mapping == {"opaque-b": 0, "opaque-a": 1}
    assert inputs == (5, 4)
    assert scheduler[2]["scheduled"]["opaque-a"] < inputs[1]
    assert runtime["dtype"] == "bfloat16"
    assert runtime["capture_enable_prefix_caching"] is None


def test_ordinal_mapping_rejects_independent_prompt_length_mismatch():
    metrics = ({"request_id": "metrics-a", "n_output_tokens": 2},)
    scheduler = (
        _scheduler_record(0, 100, {"opaque-a": 5}),
        _scheduler_record(1, 200, {"opaque-a": 1}, ("opaque-a",)),
    )
    provenance = {
        "model": "/models/Qwen3-4B",
        "qps": 4,
        "num_requests": 1,
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "prompt_lengths": [4],
        "output_lengths": [2],
        "engine_kwargs": {"dtype": "bfloat16"},
    }
    meta = {
        "model": "/models/Qwen3-4B",
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
    }

    with pytest.raises(ValueError, match="full Scheduler token conservation failed"):
        _ordinal_request_contract(
            metrics, scheduler, provenance, meta, "Qwen3-4B", 4
        )


def test_runtime_leaves_unobserved_cache_configuration_null_and_in_gap_inventory():
    runtime = _runtime_contract(
        {
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
        },
        {
            "model": "/models/Qwen3-4B",
            "qps": 4,
            "num_requests": 1,
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "engine_kwargs": {"dtype": "bfloat16"},
        },
        "Qwen3-4B",
        4,
        1,
    )

    assert runtime["capture_enable_prefix_caching"] is None
    assert "runtime.capture_enable_prefix_caching" in {
        gap["field"] for gap in _provenance_gaps(runtime, ())
    }


def test_native_c500_mctracer_stream_gap_is_explicit_and_not_duplicated():
    native_sources = ({
        "archive_id": "device-profiler",
        "tool": "mcTracer",
        "tool_version": None,
    },)
    native_gaps = _provenance_gaps({}, native_sources)
    stream_gaps = [
        gap for gap in native_gaps
        if gap["field"] == "alignment.launch_signatures.stream"
    ]
    assert len(stream_gaps) == 1
    assert "native mcTracer" in stream_gaps[0]["reason"]

    projection_sources = ({
        "archive_id": "device-launch-projection",
        "tool": "midlayer.c500_projection",
        "tool_version": "git:" + "a" * 40,
    },)
    projection_gaps = _provenance_gaps({}, projection_sources)
    assert sum(
        gap["field"] == "alignment.launch_signatures.stream"
        for gap in projection_gaps
    ) == 1


def test_llama_case_uses_explicit_instruct_raw_model_identity():
    meta = {
        "model": "/models/Llama-3.1-8B-Instruct",
        "num_requests": 128,
        "qps": 4,
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
    }
    _validate_meta(meta, "Llama-3.1-8B", 4)

    meta["model"] = "/models/Llama-3.1-8B-Instruct-copy"
    with pytest.raises(ValueError, match="meta model path disagrees"):
        _validate_meta(meta, "Llama-3.1-8B", 4)


@pytest.mark.parametrize(
    ("model_id", "target_qps", "dtype", "raw_name", "expert_parallel_size"),
    (
        ("Qwen3-32B", 1, "float16", "ShareGPT-Qwen3-32B_TP2_n128_qps1", 1),
        ("Qwen3-32B", 4, "bfloat16", "ShareGPT-Qwen3-32B_TP2_n128_qps4", 1),
        ("Qwen3-32B", 16, "float16", "ShareGPT-Qwen3-32B_TP2_n128_qps16", 1),
        (
            "Qwen3-30B-A3B",
            1,
            "bfloat16",
            "ShareGPT-Qwen3-30B-A3B_EP2_n128_qps1",
            2,
        ),
        (
            "Qwen3-30B-A3B",
            4,
            "bfloat16",
            "ShareGPT-Qwen3-30B-A3B_EP2_n128_qps4",
            2,
        ),
        (
            "Qwen3-30B-A3B",
            16,
            "bfloat16",
            "ShareGPT-Qwen3-30B-A3B_EP2_n128_qps16",
            2,
        ),
    ),
)
def test_second_phase_identity_is_proven_by_meta_and_trace_provenance(
    tmp_path,
    model_id,
    target_qps,
    dtype,
    raw_name,
    expert_parallel_size,
):
    case_dir = tmp_path / raw_name
    case_dir.mkdir()
    (case_dir / "meta.json").write_text(
        json.dumps(
            {
                "case_id": raw_name,
                "workload": "sharegpt",
                "model": f"/models/{model_id}",
                "num_requests": 128,
                "qps": target_qps,
                "tensor_parallel_size": 2,
                "expert_parallel_size": expert_parallel_size,
            }
        )
    )
    engine_kwargs = {
        "dtype": dtype,
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
    }
    if expert_parallel_size == 2:
        engine_kwargs["enable_expert_parallel"] = True
    (case_dir / "plugin_trace.json_provenance.json").write_text(
        json.dumps(
            {
                "model": f"/models/{model_id}",
                "qps": target_qps,
                "num_requests": 128,
                "tensor_parallel_size": 2,
                "expert_parallel_size": expert_parallel_size,
                "pipeline_parallel_size": 1,
                "engine_kwargs": engine_kwargs,
            }
        )
    )

    _validate_second_phase_case_identity(
        case_dir,
        _second_phase_a800_contract(model_id, target_qps),
    )


@pytest.mark.parametrize(
    ("field", "wrong_value", "message"),
    (
        ("model", "/models/Qwen3-4B", "model path disagrees"),
        ("qps", 16, "qps disagrees"),
        ("tensor_parallel_size", 1, "TP disagrees"),
        ("expert_parallel_size", 1, "EP disagrees"),
    ),
)
def test_second_phase_provenance_mismatch_fails_loud(
    tmp_path,
    field,
    wrong_value,
    message,
):
    case_dir = tmp_path / "ShareGPT-Qwen3-30B-A3B_EP2_n128_qps4"
    case_dir.mkdir()
    meta = {
        "case_id": "ShareGPT-Qwen3-30B-A3B_EP2_n128_qps4",
        "workload": "sharegpt",
        "model": "/models/Qwen3-30B-A3B",
        "num_requests": 128,
        "qps": 4,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 2,
    }
    provenance = {
        "model": "/models/Qwen3-30B-A3B",
        "qps": 4,
        "num_requests": 128,
        "tensor_parallel_size": 2,
        "expert_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "engine_kwargs": {
            "dtype": "bfloat16",
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "enable_expert_parallel": True,
        },
    }
    provenance[field] = wrong_value
    (case_dir / "meta.json").write_text(json.dumps(meta))
    (case_dir / "plugin_trace.json_provenance.json").write_text(
        json.dumps(provenance)
    )

    with pytest.raises(ValueError, match=message):
        _validate_second_phase_case_identity(
            case_dir,
            _second_phase_a800_contract("Qwen3-30B-A3B", 4),
        )


def test_second_phase_dtype_mismatch_fails_loud(tmp_path):
    case_dir = tmp_path / "ShareGPT-Qwen3-32B_TP2_n128_qps1"
    case_dir.mkdir()
    (case_dir / "meta.json").write_text(
        json.dumps(
            {
                "case_id": "ShareGPT-Qwen3-32B_TP2_n128_qps1",
                "workload": "sharegpt",
                "model": "/models/Qwen3-32B",
                "num_requests": 128,
                "qps": 1,
                "tensor_parallel_size": 2,
                "expert_parallel_size": 1,
            }
        )
    )
    (case_dir / "plugin_trace.json_provenance.json").write_text(
        json.dumps(
            {
                "model": "/models/Qwen3-32B",
                "qps": 1,
                "num_requests": 128,
                "tensor_parallel_size": 2,
                "expert_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "engine_kwargs": {
                    "dtype": "bfloat16",
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="dtype disagrees"):
        _validate_second_phase_case_identity(
            case_dir,
            _second_phase_a800_contract("Qwen3-32B", 1),
        )


@pytest.mark.parametrize("engine_pipeline_parallel_size", (1, 2))
def test_second_phase_rejects_pp2_in_provenance_or_engine(
    tmp_path,
    engine_pipeline_parallel_size,
):
    case_dir = tmp_path / "ShareGPT-Qwen3-32B_TP2_n128_qps4"
    case_dir.mkdir()
    (case_dir / "meta.json").write_text(
        json.dumps(
            {
                "case_id": "ShareGPT-Qwen3-32B_TP2_n128_qps4",
                "workload": "sharegpt",
                "model": "/models/Qwen3-32B",
                "num_requests": 128,
                "qps": 4,
                "tensor_parallel_size": 2,
                "expert_parallel_size": 1,
            }
        )
    )
    provenance_pipeline_parallel_size = (
        2 if engine_pipeline_parallel_size == 1 else 1
    )
    (case_dir / "plugin_trace.json_provenance.json").write_text(
        json.dumps(
            {
                "model": "/models/Qwen3-32B",
                "qps": 4,
                "num_requests": 128,
                "tensor_parallel_size": 2,
                "expert_parallel_size": 1,
                "pipeline_parallel_size": provenance_pipeline_parallel_size,
                "engine_kwargs": {
                    "dtype": "bfloat16",
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": engine_pipeline_parallel_size,
                },
            }
        )
    )

    with pytest.raises(ValueError, match="PP disagrees"):
        _validate_second_phase_case_identity(
            case_dir,
            _second_phase_a800_contract("Qwen3-32B", 4),
        )


def test_ordinal_runtime_uses_independent_trace_provenance_lengths_and_dtype(
    tmp_path, monkeypatch
):
    case_dir = (
        tmp_path
        / "ascend_910b"
        / "sharegpt"
        / "ShareGPT-Qwen3-4B_TP1_n128_qps4"
    )
    case_dir.mkdir(parents=True)
    arrivals = [index / 4 for index in range(128)]
    metrics = {
        "num_requests": 128,
        "wall_s": 32.0,
        "throughput_rps": 4.0,
        "per_request": [
            {
                "arrival_s": arrival,
                "n_output_tokens": 1,
                "ttft_ms": 1.0,
                "tpot_ms": None,
                "e2e_ms": 1.0,
            }
            for arrival in arrivals
        ],
    }
    scheduler = [
        _scheduler_record(
            index,
            index + 1,
            {f"opaque-{index}": 1},
            (f"opaque-{index}",),
        )
        for index in range(128)
    ]
    (case_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": "/models/Qwen3-4B",
                "num_requests": 128,
                "qps": 4,
                "tensor_parallel_size": 1,
                "expert_parallel_size": 1,
            }
        )
    )
    (case_dir / "serving_metrics.json").write_text(json.dumps(metrics))
    (case_dir / "plugin_trace.json_scheduler.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in scheduler)
    )
    (case_dir / "plugin_trace.json_provenance.json").write_text(
        json.dumps(
            {
                "model": "/models/Qwen3-4B",
                "qps": 4,
                "num_requests": 128,
                "tensor_parallel_size": 1,
                "expert_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "prompt_lengths": [1] * 128,
                "output_lengths": [1] * 128,
                "engine_kwargs": {"dtype": "bfloat16"},
            }
        )
    )
    (case_dir / "capture.log").write_text(
        "Initializing a V1 LLM engine (v0.18.0) with config: "
        "dtype=torch.bfloat16, enable_prefix_caching=False, "
        "enable_chunked_prefill=False\n"
        "INFO 07-16 14:15:14 [utils.py:1188] Block size is set to 128 if "
        "prefix cache or chunked prefill is enabled.\n"
    )
    (case_dir / "plugin_trace.json_operators.jsonl.rank0").write_text(
        "".join(
            json.dumps(
                {
                    "step_id": index,
                    "op_idx": 0,
                    "op_type": "linear",
                    "position": "qkv_proj",
                    "layer_idx": 0,
                    "attrs": {"M": 1, "K": 8, "N": 16},
                }
            )
            + "\n"
            for index in range(128)
        )
    )
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.sharegpt_raw_adapter._normalize_profiler",
        lambda *_args, **_kwargs: (
            (),
            _available_test_launches(_args[-1]),
            AvailabilityPayload(
                "unavailable", None, "test timing unavailable", None, None
            ),
        ),
    )

    payload = build_sharegpt_compact_payload(
        case_dir,
        hardware_id="ascend_910b",
        model_id="Qwen3-4B",
        target_qps=4,
        converter_source_commit="a" * 40,
    )

    assert payload.runtime["dtype"] == "bfloat16"
    assert payload.runtime["runtime_version"] == "0.18.0"
    assert payload.runtime["capture_enable_prefix_caching"] is False
    assert payload.request_field_provenance["Query Time Stamp [ms]"] == {
        "kind": "requested",
        "source_archive_id": "serving-metrics",
        "reason": (
            "the retained 910B raw has no arrival_schedule.json; serving metrics "
            "retain the requested arrival sequence"
        ),
    }
    assert payload.request_field_provenance["In Token [tokens]"] == {
        "kind": "requested",
        "source_archive_id": "trace-provenance",
        "reason": None,
    }
    assert "arrival-schedule" not in {
        source["archive_id"] for source in payload.raw_sources
    }
    assert next(
        source for source in payload.raw_sources if source["archive_id"] == "runtime-log"
    )["tool_version"] == "0.18.0"
    signatures = tuple(payload.alignment_steps["operator_signatures"].values())
    assert signatures
    assert all(record["attrs"]["dtype"] == "bfloat16" for record in signatures)
    assert payload.request_field_provenance["Cached hit [tokens]"] == {
        "kind": "scheduler_equivalent_representative",
        "source_archive_id": None,
        "reason": None,
    }
    assert payload.cached_hit_value_evidence == {
        "kind": "scheduler_equivalent_representative",
        "logical_prefix_hit_observability": "unavailable",
        "representative_value": 0,
        "net_prefix_work_reduction_tokens": 0,
        "derivation": "full_scheduler_token_conservation",
        "source_archive_ids": (
            "scheduler",
            "trace-provenance",
            "serving-metrics",
            "runtime-log",
        ),
        "raw_source_set_sha256": raw_source_set_sha256(payload.raw_sources),
    }


def test_ordinal_capture_records_enabled_cache_without_claiming_logical_hit_zero(
    tmp_path, monkeypatch
):
    case_dir = (
        tmp_path
        / "ascend_910b"
        / "sharegpt"
        / "ShareGPT-Qwen3-4B_TP1_n128_qps4"
    )
    case_dir.mkdir(parents=True)
    arrivals = [index / 4 for index in range(128)]
    (case_dir / "meta.json").write_text(
        json.dumps(
            {
                "model": "/models/Qwen3-4B",
                "num_requests": 128,
                "qps": 4,
                "tensor_parallel_size": 1,
                "expert_parallel_size": 1,
            }
        )
    )
    (case_dir / "serving_metrics.json").write_text(
        json.dumps(
            {
                "num_requests": 128,
                "wall_s": 32.0,
                "throughput_rps": 4.0,
                "per_request": [
                    {
                        "arrival_s": arrival,
                        "n_output_tokens": 1,
                        "ttft_ms": 1.0,
                        "tpot_ms": None,
                        "e2e_ms": 1.0,
                    }
                    for arrival in arrivals
                ],
            }
        )
    )
    (case_dir / "plugin_trace.json_scheduler.jsonl").write_text(
        "".join(
            json.dumps(
                _scheduler_record(
                    index,
                    index + 1,
                    {f"opaque-{index}": 1},
                    (f"opaque-{index}",),
                )
            )
            + "\n"
            for index in range(128)
        )
    )
    (case_dir / "plugin_trace.json_provenance.json").write_text(
        json.dumps(
            {
                "model": "/models/Qwen3-4B",
                "qps": 4,
                "num_requests": 128,
                "tensor_parallel_size": 1,
                "expert_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "prompt_lengths": [1] * 128,
                "output_lengths": [1] * 128,
                "engine_kwargs": {"dtype": "bfloat16"},
            }
        )
    )
    (case_dir / "plugin_trace.json_operators.jsonl.rank0").write_text(
        "".join(
            json.dumps(
                {
                    "step_id": index,
                    "op_idx": 0,
                    "op_type": "linear",
                    "position": "qkv_proj",
                    "layer_idx": 0,
                    "attrs": {"M": 1, "K": 8, "N": 16},
                }
            )
            + "\n"
            for index in range(128)
        )
    )
    (case_dir / "capture.log").write_text(
        "Initializing a V1 LLM engine (v0.18.0) with config: "
        "dtype=torch.bfloat16, enable_prefix_caching=True, "
        "enable_chunked_prefill=False\n"
    )
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.sharegpt_raw_adapter._normalize_profiler",
        lambda *_args, **_kwargs: (
            (),
            _available_test_launches(_args[-1]),
            AvailabilityPayload(
                "unavailable", None, "test timing unavailable", None, None
            ),
        ),
    )

    payload = build_sharegpt_compact_payload(
        case_dir,
        hardware_id="ascend_910b",
        model_id="Qwen3-4B",
        target_qps=4,
        converter_source_commit="a" * 40,
    )

    assert payload.runtime["capture_enable_prefix_caching"] is True
    assert payload.cached_hit_value_evidence["logical_prefix_hit_observability"] == (
        "unavailable"
    )
    assert payload.cached_hit_value_evidence["net_prefix_work_reduction_tokens"] == 0
    assert payload.cached_hit_value_evidence["representative_value"] == 0


def test_ascend_kv_block_size_requires_one_positive_capture_hint(tmp_path):
    capture_path = tmp_path / "capture.log"
    capture_path.write_text(
        "Initializing a V1 LLM engine (v0.18.0) with config: "
        "enable_prefix_caching=True, enable_chunked_prefill=False\n"
        "Block size is set to 128 if prefix cache or chunked prefill is enabled.\n"
        "Block size is set to 128 if prefix cache or chunked prefill is enabled.\n"
    )

    version, prefix_enabled, kv_block_size_tokens = _ascend_capture_runtime(
        capture_path
    )

    assert version == "0.18.0"
    assert prefix_enabled is True
    assert kv_block_size_tokens == 128

    capture_path.write_text(
        "Initializing a V1 LLM engine (v0.18.0) with config: "
        "enable_prefix_caching=True, enable_chunked_prefill=False\n"
        "Block size is set to 128 if prefix cache or chunked prefill is enabled.\n"
        "Block size is set to 64 if prefix cache or chunked prefill is enabled.\n"
    )
    with pytest.raises(ValueError, match="conflicting Block size hints"):
        _ascend_capture_runtime(capture_path)


@pytest.mark.parametrize("raw_block_size", ("0", "-1", "invalid"))
def test_ascend_kv_block_size_rejects_malformed_matched_hint(
    tmp_path, raw_block_size
):
    capture_path = tmp_path / "capture.log"
    capture_path.write_text(
        "Initializing a V1 LLM engine (v0.18.0) with config: "
        "enable_prefix_caching=False, enable_chunked_prefill=False\n"
        f"Block size is set to {raw_block_size} if prefix cache or chunked "
        "prefill is enabled.\n"
    )

    with pytest.raises(ValueError, match="Block size hint"):
        _ascend_capture_runtime(capture_path)


def test_scheduler_excludes_warmup_but_retains_idle_record():
    raw = (
        _scheduler_record(0, 100, {"warmup-0": 8}),
        _scheduler_record(1, 200, {"req_0": 4}),
        _scheduler_record(2, 300, {"req_0": 1}, ("req_0",)),
        _scheduler_record(3, 400, {}),
    )

    records, source_steps, phases = _normalized_scheduler(raw, {"req_0": 0}, (4,))

    assert len(records) == 4
    assert records[0]["scheduled_requests"] == []
    assert records[0]["total_scheduled_tokens"] == 0
    assert records[-1]["scheduled_requests"] == []
    assert phases[source_steps[1]] == "prefill"
    assert phases[source_steps[2]] == "decode"


def test_operator_records_use_raw_step_and_never_claim_request_exactness(tmp_path):
    raw_scheduler = (
        _scheduler_record(1, 100, {"req_0": 4}),
        _scheduler_record(2, 200, {"req_0": 1}, ("req_0",)),
    )
    scheduler, source_steps, phases = _normalized_scheduler(
        raw_scheduler, {"req_0": 0}, (4,)
    )
    path = tmp_path / "operators.jsonl"
    rows = (
        {
            "step_id": 1,
            "op_idx": 0,
            "op_type": "linear",
            "position": "qkv_proj",
            "layer_idx": 0,
            "attrs": {"M": 4, "K": 8, "N": 16},
        },
        {
            "step_id": 2,
            "op_idx": 0,
            "op_type": "attention",
            "position": "",
            "layer_idx": 0,
            "attrs": {"batch": 1, "seq_len": 4},
        },
    )
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    records, coverage = _normalize_operators(
        path, source_steps, phases, "bfloat16", scheduler
    )
    events = [record for record in records if record["record_type"] == "event"]

    assert coverage == "complete"
    assert [record["link_status"] for record in events] == ["step_only", "step_only"]
    assert [record["request_idxs"] for record in events] == [[], []]


def test_moe_operator_preserves_histogram_for_later_step_selection(tmp_path):
    raw_scheduler = (_scheduler_record(7, 100, {"req_0": 3}),)
    scheduler, source_steps, phases = _normalized_scheduler(
        raw_scheduler, {"req_0": 0}, (3,)
    )
    path = tmp_path / "plugin_trace.json_operators.jsonl.rank0"
    histogram = [0] * 18
    histogram[3] = 5
    histogram[17] = 7
    path.write_text(
        json.dumps(
            {
                "step_id": 7,
                "op_idx": 0,
                "op_type": "moe",
                "position": "",
                "layer_idx": 0,
                "attrs": {
                    "batch": 1,
                    "seq_len": 3,
                    "tok_per_expert": histogram,
                },
            }
        )
        + "\n"
    )

    operators, coverage = _normalize_operators(
        path,
        source_steps,
        phases,
        "bfloat16",
        scheduler,
    )

    assert coverage == "complete"
    signature = next(row for row in operators if row["record_type"] == "signature")
    assert signature["shape"] == {
        "batch": 1,
        "seq_len": 3,
        "tok_per_expert": histogram,
    }


def test_moe_operator_defers_unselected_histogram_validation(tmp_path):
    raw_scheduler = (_scheduler_record(7, 100, {"req_0": 3}),)
    scheduler, source_steps, phases = _normalized_scheduler(
        raw_scheduler, {"req_0": 0}, (3,)
    )
    path = tmp_path / "operators.jsonl"
    path.write_text(json.dumps({
        "step_id": 7,
        "op_idx": 0,
        "op_type": "moe",
        "position": "",
        "layer_idx": 47,
        "attrs": {
            "batch": 1,
            "seq_len": 3,
            "tok_per_expert": "discarded-source-payload",
        },
    }) + "\n")

    operators, coverage = _normalize_operators(
        path, source_steps, phases, "bfloat16", scheduler
    )

    assert coverage == "complete"
    signature = next(row for row in operators if row["record_type"] == "signature")
    assert signature["shape"] == {
        "batch": 1,
        "seq_len": 3,
        "tok_per_expert": "discarded-source-payload",
    }


def test_alignment_keeps_only_signatures_referenced_by_selected_step():
    scheduler = tuple(_alignment_scheduler_record(index) for index in range(3))
    operators = _alignment_operator_records((
        ("linear", {"M": 1, "K": 8, "N": 16}),
        ("moe", {"tokens": 7, "tok_per_expert": [0, 2, 5]}),
        ("linear", {"M": 3, "K": 8, "N": 16}),
    ))

    alignment = _compact_alignment_steps(
        scheduler,
        operators,
        _alignment_launches(3),
        (0,),
        "a" * 64,
        moe_num_experts=4,
        moe_topk=1,
        moe_num_layers=1,
    )

    sample = alignment["phases"]["prefill"]["sample"]
    operator_ids = {
        event["signature_id"]
        for group in sample["operators"] for event in group["events"]
    }
    launch_ids = {
        event["signature_id"]
        for group in sample["launches"] for event in group["events"]
    }
    assert sample["source_step_id"] == "step-000000000001"
    assert set(alignment["operator_signatures"]) == operator_ids
    assert set(alignment["launch_signatures"]) == launch_ids
    assert len(alignment["operator_signatures"]) == 1
    assert len(alignment["launch_signatures"]) == 1
    selected = next(iter(alignment["operator_signatures"].values()))
    assert selected["attrs"]["shape"]["tok_per_expert"] == [0, 2, 5, 0]


def test_alignment_skips_step_with_finished_request_not_scheduled_in_same_step():
    scheduler = tuple(_alignment_scheduler_record(index) for index in range(3))
    scheduler[1]["finished_request_idxs"] = (1,)
    operators = _alignment_operator_records((
        ("linear", {"M": 1, "K": 8, "N": 16}),
        ("linear", {"M": 2, "K": 8, "N": 16}),
        ("linear", {"M": 3, "K": 8, "N": 16}),
    ))

    alignment = _compact_alignment_steps(
        scheduler,
        operators,
        _alignment_launches(3),
        (0,),
        "a" * 64,
    )

    assert alignment["phases"]["prefill"]["sample"]["source_step_id"] == (
        "step-000000000000"
    )


@pytest.mark.parametrize(
    ("shape", "message"),
    (
        ({"tokens": 7}, "lacks tok_per_expert"),
        ({"tokens": 7, "tok_per_expert": []}, "cannot be empty"),
        ({"tokens": 7, "tok_per_expert": "invalid"}, "must be an array"),
        ({"tokens": 7, "tok_per_expert": [1, -1]}, "must be >= 0"),
        ({"tokens": 3, "tok_per_expert": [1, 1, 1, 0]}, "exceeds the model expert count"),
        ({"tokens": 3, "tok_per_expert": [1, 1]}, r"does not conserve tokens \* topk"),
    ),
)
def test_selected_moe_alignment_step_requires_valid_histogram(shape, message):
    scheduler = (_alignment_scheduler_record(0),)
    operators = _alignment_operator_records((("moe", shape),))

    with pytest.raises((TypeError, ValueError), match=message):
        _compact_alignment_steps(
            scheduler,
            operators,
            _alignment_launches(1),
            (0,),
            "a" * 64,
            moe_num_experts=3,
            moe_topk=1,
            moe_num_layers=1,
        )


def test_invalid_unselected_moe_histogram_does_not_block_compaction():
    scheduler = tuple(_alignment_scheduler_record(index) for index in range(3))
    operators = _alignment_operator_records((
        ("moe", {"tokens": 1, "tok_per_expert": "unselected-invalid"}),
        ("moe", {"tokens": 2, "tok_per_expert": [1, 1]}),
        ("moe", {"tokens": 3, "tok_per_expert": []}),
    ))

    alignment = _compact_alignment_steps(
        scheduler,
        operators,
        _alignment_launches(3),
        (0,),
        "a" * 64,
        moe_num_experts=2,
        moe_topk=1,
        moe_num_layers=1,
    )

    signature = next(iter(alignment["operator_signatures"].values()))
    assert signature["attrs"]["shape"]["tok_per_expert"] == [1, 1]


def test_selected_moe_alignment_rejects_cross_rank_histogram_conflict():
    scheduler = (_alignment_scheduler_record(0),)
    operators = (
        {
            "record_type": "signature",
            "signature_id": "raw-operator-0",
            "operator_kind": "moe",
            "shape": {"tokens": 1, "tok_per_expert": [1]},
            "dtype": "bfloat16",
            "variant": "test",
            "attrs": {},
        },
        {
            "record_type": "signature",
            "signature_id": "raw-operator-1",
            "operator_kind": "moe",
            "shape": {"tokens": 1, "tok_per_expert": [0, 1]},
            "dtype": "bfloat16",
            "variant": "test",
            "attrs": {},
        },
        {
            "record_type": "event",
            "signature_id": "raw-operator-0",
            "step_id": "step-000000000000",
            "rank": 0,
            "sequence_idx": 0,
            "layer_idx": 0,
        },
        {
            "record_type": "event",
            "signature_id": "raw-operator-1",
            "step_id": "step-000000000000",
            "rank": 1,
            "sequence_idx": 0,
            "layer_idx": 0,
        },
    )
    launches = _alignment_launches(1)
    launch_records = list(launches.payload)
    launch_records.append({
        **launch_records[-1],
        "rank": 1,
    })

    with pytest.raises(ValueError, match="histograms disagree across ranks"):
        _compact_alignment_steps(
            scheduler,
            operators,
            AvailabilityPayload(
                launches.status,
                launches.coverage,
                launches.reason,
                tuple(launch_records),
                launches.source_archive_ids,
            ),
            (0, 1),
            "a" * 64,
            moe_num_experts=2,
            moe_topk=1,
            moe_num_layers=1,
        )


def _c500_kernel_event(name, timestamp, duration, device_id=0):
    return {
        "name": name,
        "ph": "X",
        "cat": "0",
        "ts": timestamp,
        "dur": duration,
        "args": {
            "device_id": device_id,
            "grid": {"x": 3, "y": 2, "z": 1},
            "block": {"x": 64, "y": 2, "z": 1},
            "mem": {
                "static_shared": 128,
                "dynamic_shared": 256,
                "registers_per_thread": 37,
            },
        },
    }


def _c500_memory_event(event_kind, timestamp, duration, device_id=0):
    if event_kind == "memcpy":
        native = {
            "kind": 1,
            "copy_kind": 8,
            "src_kind": 3,
            "dst_kind": 2,
        }
    elif event_kind == "memset":
        native = {"kind": 2, "mem_kind": 3, "value": 0}
    else:
        raise AssertionError(f"unsupported test event kind: {event_kind}")
    return {
        "name": f"typed-{event_kind}",
        "ph": "X",
        "cat": "0",
        "ts": timestamp,
        "dur": duration,
        "args": {"device_id": device_id, "bytes": 4096, **native},
    }


def _write_c500_trace(case_dir, events, filename="mctracer-1.json"):
    trace_dir = case_dir / "mctracer_out"
    trace_dir.mkdir(parents=True, exist_ok=True)
    path = trace_dir / filename
    path.write_text(json.dumps({"displayTimeUnit": "ns", "traceEvents": events}))
    return path


def _c500_projected_event(name, timestamp, duration=None):
    event = {
        "name": name,
        "ts_ns": timestamp,
        "grid": [3, 2, 1],
        "block": [64, 2, 1],
        "smem": 384,
    }
    if duration is not None:
        event["dur_ns"] = duration
    return event


def _c500_projection_operators(scheduler, ranks):
    return tuple(
        {
            "record_type": "event",
            "step_id": record["step_id"],
            "rank": rank,
        }
        for record in scheduler
        if record["scheduled_requests"]
        for rank in ranks
    )


def _c500_sampler_step(timestamp, label):
    return [
        _c500_projected_event(
            "std::enable_if<void>::type "
            "at::native::InputPerOutputContinuousReduceKernel<float>("
            f"float const*, at::native::ArgMaxOps<float>, {label})",
            timestamp,
            10,
        ),
        _c500_projected_event(
            "void at::native::unrolled_elementwise_kernel<"
            "direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() "
            "const::{lambda()#3}::operator()() const::{lambda(int)#1}",
            timestamp + 20,
            10,
        ),
        _c500_projected_event(
            "void at::native::unrolled_elementwise_kernel<"
            "direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() "
            "const::{lambda()#4}::operator()() const::{lambda(long)#1}",
            timestamp + 40,
            10,
        ),
        _c500_projected_event("memcpy DTOH(device,pinned)", timestamp + 60, 10),
    ]


def _write_c500_projection_capture_log(case_dir, pids=(100, 101), *, with_ep=False):
    lines = []
    for rank, pid in enumerate(pids):
        ep = f"_EP{rank}" if with_ep else ""
        lines.append(f"(Worker_TP{rank}{ep} pid={pid}) worker ready")
        lines.append(f"Output file: /capture/mctracer-{pid}.json")
    (case_dir / "mctracer_stdout.log").write_text("\n".join(lines) + "\n")
    (case_dir / "meta.json").write_text(json.dumps({
        "gpu_ids": list(range(len(pids))),
    }))


def test_c500_legacy_projection_uses_compound_sampler_terminal_for_all_ranks(tmp_path):
    _write_c500_projection_capture_log(tmp_path)
    projection = tmp_path / "nsys_kernels_full.json"
    events = [
        *_c500_sampler_step(1_100, "prefill"),
        *_c500_sampler_step(2_100, "decode-a"),
        *_c500_sampler_step(3_100, "decode-b"),
    ]
    projection.write_text(json.dumps({
        "kernels": events,
        "per_rank": {
            "0": {"kernels": events},
            "1": {"kernels": events},
        },
        "bottleneck_rank": "1",
    }))
    raw_scheduler = (
        _scheduler_record(0, 1, {"req_0": 2}),
        _scheduler_record(1, 2, {"req_0": 1}),
        _scheduler_record(2, 3, {"req_0": 1}, ("req_0",)),
        _scheduler_record(3, 4, {}),
    )
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (2,))

    launches, timing = _normalize_c500_projection(
        tmp_path,
        projection,
        (),
        raw_scheduler,
        scheduler,
        (0, 1),
        _c500_projection_operators(scheduler, (0, 1)),
    )

    assert launches.status == "available"
    assert launches.source_archive_ids == (
        "device-launch-projection",
        "mctracer-capture-log",
    )
    launch_events = [
        record for record in launches.payload if record["record_type"] == "event"
    ]
    assert {record["rank"] for record in launch_events} == {0, 1}
    signatures = [
        record for record in launches.payload if record["record_type"] == "signature"
    ]
    assert all(record["registers_per_thread"] is None for record in signatures)
    assert all(record["stream"] is None for record in signatures)
    assert all(record["geometry_kind"] == "unavailable" for record in signatures)
    assert all(record["grid"] is None for record in signatures)
    assert all(record["block"] is None for record in signatures)
    assert all(record["shared_memory_bytes"] is None for record in signatures)
    assert all(record["device"] is None for record in signatures)
    assert timing.status == "unavailable"


def test_c500_legacy_projection_maps_rank_syncs_without_fixed_drop(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.sharegpt_raw_adapter."
        "_validate_c500_projection_rank_provenance",
        lambda *_args: None,
    )
    projection = tmp_path / "nsys_kernels_full.json"
    rank_zero_events = [
        _c500_sampler_step(1_500, "warmup-a")[0],
        _c500_sampler_step(1_600, "warmup-b")[0],
        _c500_sampler_step(3_500, "warmup-c")[0],
        _c500_sampler_step(3_600, "warmup-d")[0],
        _c500_sampler_step(3_700, "warmup-e")[0],
        _c500_sampler_step(10_500_000, "prefill")[0],
        _c500_sampler_step(20_500_000, "decode-a")[0],
        _c500_sampler_step(30_500_000, "decode-b")[0],
    ]
    rank_one_events = [
        {**event, "ts_ns": event["ts_ns"] + 200_000}
        for event in rank_zero_events
    ]
    projection.write_text(json.dumps({
        "kernels": [],
        "per_rank": {"1": rank_one_events, "0": rank_zero_events},
    }))
    raw_scheduler = (
        _scheduler_record(0, 10_000, {"req_0": 2}),
        _scheduler_record(1, 20_000, {"req_0": 1}),
        _scheduler_record(2, 30_000, {"req_0": 1}, ("req_0",)),
        _scheduler_record(3, 40_000, {}),
    )
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (2,))
    syncs = [
        [100, 1_000], [1_100, 2_000], [2_100, 3_000], [3_100, 9_000_000],
        [10_900_000, 11_000_000],
        [18_900_000, 19_000_000], [19_900_000, 20_000_000],
        [20_900_000, 21_000_000],
        [30_900_000, 31_000_000],
    ]
    sync_paths = []
    for rank, offset in ((0, 0), (1, 200_000)):
        path = tmp_path / f".mctracer_sync_ends_{rank}.json"
        path.write_text(json.dumps([
            [start + offset, end + offset] for start, end in syncs
        ]))
        sync_paths.append(path)

    launches, timing = _normalize_c500_projection(
        tmp_path,
        projection,
        tuple(sync_paths),
        raw_scheduler,
        scheduler,
        (0, 1),
        _c500_projection_operators(scheduler, (0, 1)),
    )

    events = [
        record for record in launches.payload if record["record_type"] == "event"
    ]
    assert {record["step_id"] for record in events} == {
        "step-000000000000", "step-000000000001"
    }
    assert timing.status == "unavailable"
    assert timing.source_archive_ids == (
        "device-launch-projection",
        "mctracer-capture-log",
        "profiler-step-boundaries-rank0",
        "profiler-step-boundaries-rank1",
    )


def test_c500_legacy_projection_rejects_missing_rank_and_boundary_crossing(tmp_path):
    _write_c500_projection_capture_log(tmp_path)
    projection = tmp_path / "nsys_kernels_full.json"
    projection.write_text(json.dumps({
        "kernels": [],
        "per_rank": {
            "0": [_c500_projected_event("rank-zero", 1_100, 100)],
        },
    }))
    raw_scheduler = (
        _scheduler_record(0, 1, {"req_0": 2}),
        _scheduler_record(1, 2, {"req_0": 1}, ("req_0",)),
        _scheduler_record(2, 3, {}),
    )
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (2,))
    operators = _c500_projection_operators(scheduler, (0, 1))

    with pytest.raises(ValueError, match="exactly match"):
        _normalize_c500_projection(
            tmp_path,
            projection,
            (),
            raw_scheduler,
            scheduler,
            (0, 1),
            operators,
        )


def test_c500_projection_stream_parser_checks_complete_document_and_digest(tmp_path):
    projection = tmp_path / "nsys_kernels_full.json"
    event = _c500_projected_event("real-kernel", 100, 0)
    projection.write_text(json.dumps({
        "kernels": [event],
        "per_rank": {"0": {"kernels": [event]}},
        "bottleneck_rank": "0",
    }))

    scan = _scan_c500_projection_events(
        projection, (0,), lambda *_args: None, chunk_bytes=7
    )

    assert scan.top_level_count == 1
    assert scan.entry_count_by_rank == {0: 1}
    assert scan.top_level_digest == scan.digest_by_rank[0]
    assert scan.duration_present is True

    projection.write_text(
        '{"kernels":[],"kernels":[],"per_rank":{"0":[]}}'
    )
    with pytest.raises(ValueError, match="duplicate top-level key"):
        _scan_c500_projection_events(projection, (0,), lambda *_args: None)

    projection.write_text(
        '{"kernels":[{"name":"broken"}],"per_rank":{"0":[]}} trailing'
    )
    with pytest.raises(ValueError):
        _scan_c500_projection_events(projection, (0,), lambda *_args: None)


def test_c500_projection_rejects_redundant_bottleneck_mismatch(tmp_path):
    projection = tmp_path / "nsys_kernels_full.json"
    projection.write_text(json.dumps({
        "kernels": [_c500_projected_event("top", 100)],
        "per_rank": {"0": [_c500_projected_event("rank", 100)]},
        "bottleneck_rank": "0",
    }))

    with pytest.raises(ValueError, match="redundant kernels disagree"):
        _scan_c500_projection_events(projection, (0,), lambda *_args: None)


def test_c500_projection_excludes_legacy_memory_rows_from_launches(tmp_path):
    _write_c500_projection_capture_log(tmp_path, (100,))
    projection = tmp_path / "nsys_kernels_full.json"
    events = [
        _c500_projected_event("memset", 100, 0),
        _c500_projected_event("memcpy DTOH(device,pinned)", 200, 10),
        *_c500_sampler_step(300, "sample"),
    ]
    projection.write_text(json.dumps({
        "kernels": events,
        "per_rank": {"0": events},
        "bottleneck_rank": "0",
    }))
    raw_scheduler = (
        _scheduler_record(0, 99_999, {"req_0": 1}, ("req_0",)),
    )
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (1,))

    launches, timing = _normalize_c500_projection(
        tmp_path,
        projection,
        (),
        raw_scheduler,
        scheduler,
        (0,),
        _c500_projection_operators(scheduler, (0,)),
    )

    signatures = [
        record for record in launches.payload if record["record_type"] == "signature"
    ]
    assert all("memcpy" not in record["kernel_name"] for record in signatures)
    assert all(record["kernel_name"] != "memset" for record in signatures)
    assert len(signatures) == 3
    assert timing.status == "unavailable"
    assert timing.payload is None


def test_c500_legacy_projection_rejects_interleaved_sampler_trailer(tmp_path):
    _write_c500_projection_capture_log(tmp_path, (100,))
    events = _c500_sampler_step(300, "sample")
    events.insert(1, _c500_projected_event("interleaved-real-kernel", 310, 10))
    projection = tmp_path / "nsys_kernels_full.json"
    projection.write_text(json.dumps({
        "kernels": events,
        "per_rank": {"0": events},
        "bottleneck_rank": "0",
    }))
    raw_scheduler = (_scheduler_record(0, 1, {"req_0": 1}, ("req_0",)),)
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (1,))

    with pytest.raises(ValueError, match="not consecutive"):
        _normalize_c500_projection(
            tmp_path,
            projection,
            (),
            raw_scheduler,
            scheduler,
            (0,),
            _c500_projection_operators(scheduler, (0,)),
        )


def test_c500_projection_rejects_unknown_memory_name(tmp_path):
    _write_c500_projection_capture_log(tmp_path, (100,))
    events = [
        _c500_projected_event("memcpy UNKNOWN(device,device)", 100, 10),
        *_c500_sampler_step(300, "sample"),
    ]
    projection = tmp_path / "nsys_kernels_full.json"
    projection.write_text(json.dumps({
        "kernels": events,
        "per_rank": {"0": events},
        "bottleneck_rank": "0",
    }))
    raw_scheduler = (_scheduler_record(0, 1, {"req_0": 1}, ("req_0",)),)
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (1,))

    with pytest.raises(ValueError, match="unsupported C500 legacy memory"):
        _normalize_c500_projection(
            tmp_path,
            projection,
            (),
            raw_scheduler,
            scheduler,
            (0,),
            _c500_projection_operators(scheduler, (0,)),
        )


def test_c500_projection_rejects_duplicate_worker_pid_and_gpu_ids(tmp_path):
    events = _c500_sampler_step(300, "sample")
    projection = tmp_path / "nsys_kernels_full.json"
    projection.write_text(json.dumps({
        "kernels": events,
        "per_rank": {"0": events, "1": events},
        "bottleneck_rank": "0",
    }))
    raw_scheduler = (_scheduler_record(0, 1, {"req_0": 1}, ("req_0",)),)
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (1,))
    operators = _c500_projection_operators(scheduler, (0, 1))

    _write_c500_projection_capture_log(tmp_path, (100, 100))
    with pytest.raises(ValueError, match="reuses one worker PID"):
        _normalize_c500_projection(
            tmp_path, projection, (), raw_scheduler, scheduler, (0, 1), operators
        )

    _write_c500_projection_capture_log(tmp_path, (100, 101))
    (tmp_path / "meta.json").write_text(json.dumps({"gpu_ids": [0, 0]}))
    with pytest.raises(ValueError, match="one unique device"):
        _normalize_c500_projection(
            tmp_path, projection, (), raw_scheduler, scheduler, (0, 1), operators
        )



def test_c500_native_typed_trace_preserves_prefix_order_and_provenance(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_c500_trace(
        case_dir,
        [
            _c500_kernel_event("warmup", 50, 10, device_id=1),
            _c500_kernel_event("prefill", 120, 10, device_id=1),
            _c500_memory_event("memcpy", 140, 5, device_id=1),
            _c500_memory_event("memset", 150, 3, device_id=1),
            _c500_kernel_event("decode", 220, 11, device_id=1),
            _c500_kernel_event("unproven-tail", 350, 7, device_id=1),
        ],
    )
    (case_dir / "nsys_kernels_full.json").write_text("not canonical evidence")
    (case_dir / ".mctracer_sync_ends_0.json").write_text(
        json.dumps([[90, 100], [190, 200], [290, 300]])
    )
    raw_scheduler = (
        _scheduler_record(0, 1, {"warmup-0": 1}),
        _scheduler_record(1, 2, {"req_0": 1}),
        _scheduler_record(2, 3, {"req_0": 1}, ("req_0",)),
        _scheduler_record(3, 4, {}),
    )
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (1,))

    sources, launches, timing = _normalize_profiler(
        case_dir, "metax_c500", raw_scheduler, scheduler
    )

    assert [source["relative_path"] for source in sources] == [
        "mctracer_out/mctracer-1.json",
        ".mctracer_sync_ends_0.json",
    ]
    assert launches.status == "available"
    assert launches.coverage == "partial"
    assert timing.status == "available"
    assert timing.coverage == "partial"
    signatures = [
        record for record in launches.payload if record["record_type"] == "signature"
    ]
    launch_events = [
        record for record in launches.payload if record["record_type"] == "event"
    ]
    signatures_by_name = {record["kernel_name"]: record for record in signatures}
    assert set(signatures_by_name) == {"decode", "prefill"}
    assert [record["rank"] for record in launch_events] == [1, 1]
    assert [record["link_status"] for record in launch_events] == [
        "step_order_only",
        "step_order_only",
    ]
    assert signatures_by_name["prefill"]["grid"] == [3, 2, 1]
    assert signatures_by_name["prefill"]["block"] == [64, 2, 1]
    assert signatures_by_name["prefill"]["shared_memory_bytes"] == 384
    assert signatures_by_name["prefill"]["registers_per_thread"] == 37
    assert [record["interval_kind"] for record in timing.payload] == [
        "device_kernel",
        "device_copy",
        "device_copy",
        "device_kernel",
    ]
    assert [record["correlation"]["source_id"] for record in timing.payload] == [
        "mctracer-kernel-1",
        "mctracer-memcpy-2",
        "mctracer-memset-3",
        "mctracer-kernel-4",
    ]
    assert all(record["rank"] == 1 for record in timing.payload)
    assert all(
        record["source_archive_id"] == "device-profiler"
        for record in timing.payload
    )


def test_c500_native_tracer_out_filename_is_normalized(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    trace_path = _write_c500_trace(
        case_dir,
        [
            _c500_kernel_event("warmup", 50, 10),
            _c500_kernel_event("decode", 150, 10),
        ],
        filename="tracer_out-42.json",
    )
    (case_dir / ".mctracer_sync_ends_0.json").write_text(
        json.dumps([[40, 70], [140, 170]])
    )
    raw_scheduler = (
        _scheduler_record(0, 1, {"warmup-0": 1}),
        _scheduler_record(1, 2, {"req_0": 1}, ("req_0",)),
    )
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (1,))

    sources, launches, timing = _normalize_profiler(
        case_dir, "metax_c500", raw_scheduler, scheduler
    )

    assert sources[0]["relative_path"] == trace_path.relative_to(case_dir).as_posix()
    assert launches.status == "available"
    assert timing.status == "available"


def test_c500_profiler_requires_exactly_one_device_event_trace(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / ".mctracer_sync_ends_0.json").write_text(
        json.dumps([[90, 100], [190, 200]])
    )
    raw_scheduler = (
        _scheduler_record(0, 1, {"warmup-0": 1}),
        _scheduler_record(1, 2, {"req_0": 1}, ("req_0",)),
    )
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (1,))

    with pytest.raises(FileNotFoundError, match="native mcTracer directory"):
        _normalize_profiler(case_dir, "metax_c500", raw_scheduler, scheduler)

    device_trace = _write_c500_trace(
        case_dir,
        [_c500_kernel_event("first", 195, 4)],
    )
    _write_c500_trace(
        case_dir,
        [
            {
                "name": "mcInit",
                "ph": "X",
                "cat": "host",
                "ts": 1,
                "dur": 1,
                "args": {"co_id": 1},
            }
        ],
        filename="mctracer-2.json",
    )
    sources, launches, timing = _normalize_profiler(
        case_dir, "metax_c500", raw_scheduler, scheduler
    )
    assert sources[0]["relative_path"] == device_trace.relative_to(case_dir).as_posix()
    assert launches.status == "available"
    assert timing.status == "available"

    _write_c500_trace(
        case_dir,
        [_c500_kernel_event("second", 196, 3)],
        filename="mctracer-3.json",
    )
    with pytest.raises(ValueError, match="exactly one mcTracer process file"):
        _normalize_profiler(case_dir, "metax_c500", raw_scheduler, scheduler)


def test_c500_profiler_rejects_event_crossing_sync_boundary(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_c500_trace(case_dir, [_c500_kernel_event("crossing", 195, 10)])
    (case_dir / ".mctracer_sync_ends_0.json").write_text(
        json.dumps([[90, 100], [190, 200]])
    )
    raw_scheduler = (
        _scheduler_record(0, 1, {"warmup-0": 1}),
        _scheduler_record(1, 2, {"req_0": 1}, ("req_0",)),
    )
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (1,))

    with pytest.raises(ValueError, match="crosses a sync boundary"):
        _normalize_profiler(case_dir, "metax_c500", raw_scheduler, scheduler)


def test_c500_profiler_orders_concurrent_events_by_device_timestamp(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    _write_c500_trace(
        case_dir,
        [
            _c500_kernel_event("later", 196, 3),
            _c500_memory_event("memcpy", 195, 4),
        ],
    )
    (case_dir / ".mctracer_sync_ends_0.json").write_text(
        json.dumps([[90, 100], [190, 200]])
    )
    raw_scheduler = (
        _scheduler_record(0, 1, {"warmup-0": 1}),
        _scheduler_record(1, 2, {"req_0": 1}, ("req_0",)),
    )
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (1,))

    _, launches, timing = _normalize_profiler(
        case_dir, "metax_c500", raw_scheduler, scheduler
    )

    launch_events = [
        record for record in launches.payload if record["record_type"] == "event"
    ]
    assert [record["sequence_idx"] for record in launch_events] == [0]
    assert [record["start_ns"] for record in timing.payload] == [195, 196]
    assert [record["correlation"]["source_id"] for record in timing.payload] == [
        "mctracer-memcpy-1",
        "mctracer-kernel-0",
    ]


def test_a800_profiler_uses_correlated_rank_and_sampler_tail_boundaries(tmp_path):
    database = tmp_path / "nsys_profile.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE StringIds(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(
                start INTEGER, end INTEGER, nameId INTEGER,
                globalTid INTEGER, correlationId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(
                start INTEGER, end INTEGER, deviceId INTEGER, correlationId INTEGER,
                demangledName INTEGER,
                gridX INTEGER, gridY INTEGER, gridZ INTEGER,
                blockX INTEGER, blockY INTEGER, blockZ INTEGER,
                staticSharedMemory INTEGER, dynamicSharedMemory INTEGER,
                registersPerThread INTEGER, streamId INTEGER
            );
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(
                start INTEGER, end INTEGER, deviceId INTEGER, copyKind INTEGER
            );
            INSERT INTO StringIds VALUES(1, 'cudaLaunchKernel');
            INSERT INTO StringIds VALUES(2, 'rms_norm_kernel');
            INSERT INTO StringIds VALUES(3, 'gemm_kernel');
            INSERT INTO StringIds VALUES(4, 'sampler_direct_copy_kernel_cuda');
            INSERT INTO StringIds VALUES(5, 'cross_device_reduce_kernel');
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(90, 91, 1, 101, 10);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(100, 101, 1, 101, 11);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(160, 161, 1, 101, 12);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(290, 291, 1, 101, 13);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(360, 361, 1, 101, 14);
            INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(250, 251, 1, 101, 15);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(100, 150, 0, 10, 2, 1, 1, 1, 32, 1, 1, 0, 0, 16, 7);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(110, 140, 0, 11, 3, 2, 1, 1, 64, 1, 1, 8, 4, 32, 7);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(170, 180, 0, 12, 4, 1, 1, 1, 32, 1, 1, 0, 0, 8, 7);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(300, 350, 0, 13, 3, 2, 1, 1, 64, 1, 1, 8, 4, 32, 7);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(370, 380, 0, 14, 4, 1, 1, 1, 32, 1, 1, 0, 0, 8, 7);
            INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(250, 260, 0, 15, 5, 1, 1, 1, 32, 1, 1, 0, 0, 8, 7);
            INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(160, 170, 0, 1);
            """
        )
    scheduler = (
        _compact_scheduler_record("step-0", "prefill"),
        _compact_scheduler_record("step-1", "decode"),
    )

    launches, timing = _normalize_a800_profiler(database, scheduler)

    launch_events = [
        record for record in launches.payload if record["record_type"] == "event"
    ]
    assert launches.status == "available"
    assert launches.coverage == "complete"
    assert timing.coverage == "partial"
    assert len(launch_events) == 6
    assert [record["step_id"] for record in launch_events] == [
        "step-0", "step-0", "step-0", "step-1", "step-1", "step-1"
    ]
    assert all(record["interval_kind"] == "device_kernel" for record in timing.payload)


def _ascend_operator_events(scheduler, ranks=(0,)):
    return tuple(
        {
            "record_type": "event",
            "rank": rank,
            "step_id": record["step_id"],
        }
        for rank in ranks
        for record in scheduler
        if record["scheduled_requests"]
    )


def _write_ascend_intermediate_profile(case_dir, *, duplicate_device_match=False):
    profile_dir = case_dir / "msprof" / "PROF_1"
    device_dir = profile_dir / "device_0"
    host_dir = profile_dir / "host" / "sqlite"
    (device_dir / "sqlite").mkdir(parents=True)
    host_dir.mkdir(parents=True)
    with sqlite3.connect(profile_dir / "msprof_1.db") as connection:
        connection.executescript(
            """
            CREATE TABLE TASK(startNs INTEGER, endNs INTEGER, deviceId INTEGER, globalTaskId INTEGER, streamId INTEGER, taskId INTEGER);
            CREATE TABLE COMPUTE_TASK_INFO(name INTEGER, globalTaskId INTEGER PRIMARY KEY, blockDim INTEGER, mixBlockDim INTEGER, opType INTEGER, taskType INTEGER);
            INSERT INTO COMPUTE_TASK_INFO VALUES(1, 1, 8, 0, 2, 3);
            INSERT INTO COMPUTE_TASK_INFO VALUES(1, 2, 0, 0, 2, 4);
            INSERT INTO COMPUTE_TASK_INFO VALUES(1, 3, 8, 0, 2, 3);
            INSERT INTO COMPUTE_TASK_INFO VALUES(5, 4, 8, 0, 2, 3);
            INSERT INTO COMPUTE_TASK_INFO VALUES(1, 5, 8, 0, 2, 3);
            """
        )
    with sqlite3.connect(device_dir / "sqlite" / "ascend_task.db") as connection:
        connection.executescript(
            """
            CREATE TABLE AscendTask(stream_id INTEGER, task_id INTEGER, start_time INTEGER, duration INTEGER, device_task_type TEXT, connection_id INTEGER);
            INSERT INTO AscendTask VALUES(2, 1, 49000, 100, 'AI_CORE', 1);
            INSERT INTO AscendTask VALUES(2, 2, 109000, 100, 'AI_CPU', 2);
            INSERT INTO AscendTask VALUES(2, 3, 149000, 100, 'AI_CORE', 3);
            INSERT INTO AscendTask VALUES(2, 4, 209000, 100, 'AI_CORE', 4);
            INSERT INTO AscendTask VALUES(2, 5, 249000, 100, 'AI_CORE', 5);
            """
        )
        if duplicate_device_match:
            connection.execute(
                "INSERT INTO AscendTask VALUES(2, 4, 209001, 100, 'AI_CORE', 4)"
            )
    with sqlite3.connect(host_dir / "ge_info.db") as connection:
        connection.executescript(
            """
            CREATE TABLE TaskInfo(op_name TEXT, stream_id INTEGER, task_id INTEGER, block_dim INTEGER, mix_block_dim INTEGER, task_type TEXT, op_type TEXT, thread_id INTEGER, timestamp INTEGER);
            INSERT INTO TaskInfo VALUES('aclnnEmbedding_GatherV2AiCore_GatherV2', 2, 1, 8, 0, 'AI_VECTOR_CORE', 'GatherV2', 101, 1000);
            INSERT INTO TaskInfo VALUES('aclnnBincount_BincountAiCpu_Bincount', 2, 2, 0, 0, 'AI_CPU', 'Bincount', 101, 2000);
            INSERT INTO TaskInfo VALUES('aclnnEmbedding_GatherV2AiCore_GatherV2', 2, 3, 8, 0, 'AI_VECTOR_CORE', 'GatherV2', 101, 3000);
            INSERT INTO TaskInfo VALUES('kernel-with-geometry', 2, 4, 8, 0, 'AI_VECTOR_CORE', 'MatMul', 101, 4000);
            INSERT INTO TaskInfo VALUES('aclnnEmbedding_GatherV2AiCore_GatherV2', 2, 5, 8, 0, 'AI_VECTOR_CORE', 'GatherV2', 101, 5000);
            """
        )
    with sqlite3.connect(host_dir / "runtime.db") as connection:
        connection.executescript(
            """
            CREATE TABLE HostTask(stream_id INTEGER, task_id INTEGER, device_id INTEGER, timestamp INTEGER, connection_id INTEGER, thread_id INTEGER);
            INSERT INTO HostTask VALUES(2, 1, 0, 1000, 1, 101);
            INSERT INTO HostTask VALUES(2, 2, 0, 2000, 2, 101);
            INSERT INTO HostTask VALUES(2, 3, 0, 3000, 3, 101);
            INSERT INTO HostTask VALUES(2, 4, 0, 4000, 4, 101);
            INSERT INTO HostTask VALUES(2, 5, 0, 5000, 5, 101);
            """
        )
    (device_dir / "info.json.0").write_text(json.dumps({
        "version": "1.0", "pid": "101", "devices": "0",
        "DeviceInfo": [{"id": 0}],
    }))
    (device_dir / "start_info.0").write_text(json.dumps({
        "clockMonotonicRaw": "1000", "collectionTimeBegin": "2",
    }))
    case_dir.mkdir(exist_ok=True)
    (case_dir / "capture.log").write_text(
        "(Worker pid=101) INFO world_size=1 rank=0 local_rank=0 backend=hccl\n"
    )


def test_ascend_profiler_recovers_supported_intermediate_task_sources(tmp_path):
    case_dir = tmp_path / "case"
    _write_ascend_intermediate_profile(case_dir)
    raw_scheduler = (
        _scheduler_record(0, 100, {"req_0": 4}),
        _scheduler_record(1, 200, {"req_0": 1}, ("req_0",)),
        _scheduler_record(2, 300, {}),
    )
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (4,))

    sources, launches, timing = _normalize_profiler(
        case_dir,
        "ascend_910b",
        raw_scheduler,
        scheduler,
        expected_ranks=(0,),
        operator_records=_ascend_operator_events(scheduler),
    )

    assert [source["archive_id"] for source in sources] == [
        "device-profiler-rank0",
        "profiler-metadata-rank0",
        "intermediate-device-tasks-rank0",
        "intermediate-ge-task-info-rank0",
        "intermediate-host-runtime-rank0",
        "intermediate-device-clock-rank0",
    ]
    assert launches.status == "available"
    assert timing.status == "available"
    assert [record["start_ns"] for record in timing.payload] == [110000, 150000, 210000, 250000]


def test_ascend_intermediate_recovery_rejects_ambiguous_device_join(tmp_path):
    case_dir = tmp_path / "case"
    _write_ascend_intermediate_profile(case_dir, duplicate_device_match=True)
    raw_scheduler = (_scheduler_record(0, 100, {"req_0": 4}, ("req_0",)),)
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (4,))

    with pytest.raises(ValueError, match="complete 1:1 join"):
        _normalize_profiler(
            case_dir,
            "ascend_910b",
            raw_scheduler,
            scheduler,
            expected_ranks=(0,),
            operator_records=_ascend_operator_events(scheduler),
        )


def test_ascend_profiler_retains_aicpu_task_without_launch_geometry(tmp_path):
    case_dir = tmp_path / "case"
    profile_dir = case_dir / "msprof" / "PROF_1"
    device_dir = profile_dir / "device_0"
    device_dir.mkdir(parents=True)
    database = profile_dir / "msprof_1.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE STRING_IDS(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE TASK(startNs INTEGER, endNs INTEGER, deviceId INTEGER, globalTaskId INTEGER, streamId INTEGER, taskId INTEGER);
            CREATE TABLE COMPUTE_TASK_INFO(
                name INTEGER, globalTaskId INTEGER PRIMARY KEY, blockDim INTEGER,
                mixBlockDim INTEGER, opType INTEGER, taskType INTEGER
            );
            INSERT INTO STRING_IDS VALUES(1, 'aclnnBincount_BincountAiCpu_Bincount');
            INSERT INTO STRING_IDS VALUES(2, 'Bincount');
            INSERT INTO STRING_IDS VALUES(3, 'AI_CPU');
            INSERT INTO STRING_IDS VALUES(4, 'aclnnEmbedding_GatherV2AiCore_GatherV2');
            INSERT INTO STRING_IDS VALUES(5, 'GatherV2');
            INSERT INTO STRING_IDS VALUES(6, 'AI_CORE');
            INSERT INTO STRING_IDS VALUES(7, 'kernel-with-geometry');
            INSERT INTO TASK VALUES(50000, 50100, 0, 9, 2, 9);
            INSERT INTO TASK VALUES(110000, 110100, 0, 10, 2, 10);
            INSERT INTO TASK VALUES(150000, 150100, 0, 11, 2, 11);
            INSERT INTO TASK VALUES(210000, 210100, 0, 12, 2, 12);
            INSERT INTO TASK VALUES(250000, 250100, 0, 13, 2, 13);
            INSERT INTO COMPUTE_TASK_INFO VALUES(4, 9, 8, 0, 5, 6);
            INSERT INTO COMPUTE_TASK_INFO VALUES(1, 10, 0, 0, 2, 3);
            INSERT INTO COMPUTE_TASK_INFO VALUES(4, 11, 8, 0, 5, 6);
            INSERT INTO COMPUTE_TASK_INFO VALUES(7, 12, 8, 0, 5, 6);
            INSERT INTO COMPUTE_TASK_INFO VALUES(4, 13, 8, 0, 5, 6);
            """
        )
    (device_dir / "info.json.0").write_text(json.dumps({
        "version": "1.0",
        "pid": "101",
        "devices": "0",
        "DeviceInfo": [{"id": 0}],
    }))
    (case_dir / "capture.log").write_text(
        "(Worker pid=101) INFO world_size=1 rank=0 local_rank=0 backend=hccl\n"
    )
    raw_scheduler = (
        _scheduler_record(0, 100, {"req_0": 4}),
        _scheduler_record(1, 200, {"req_0": 1}, ("req_0",)),
        _scheduler_record(2, 300, {}),
    )
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (4,))

    sources, launches, timing = _normalize_profiler(
        case_dir,
        "ascend_910b",
        raw_scheduler,
        scheduler,
        expected_ranks=(0,),
        operator_records=_ascend_operator_events(scheduler),
    )

    launch_events = tuple(
        record for record in launches.payload if record["record_type"] == "event"
    )
    assert [source["archive_id"] for source in sources] == [
        "device-profiler-rank0",
        "profiler-metadata-rank0",
    ]
    assert launches.status == "available"
    assert launches.coverage == "complete"
    assert launches.reason is None
    assert [record["step_id"] for record in launch_events] == [
        "step-000000000000",
        "step-000000000000",
        "step-000000000001",
        "step-000000000001",
    ]
    assert [record["interval_kind"] for record in timing.payload] == [
        "device_aicpu_task",
        "device_kernel",
        "device_kernel",
        "device_kernel",
    ]
    assert timing.payload[0]["launch_id"] == launch_events[0]["launch_id"]


def test_ascend_profiler_binds_each_database_to_rank_by_worker_pid(tmp_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "capture.log").write_text(
        "(Worker pid=202) INFO world_size=2 rank=1 local_rank=1 backend=hccl\n"
        "(Worker pid=101) INFO world_size=2 rank=0 local_rank=0 backend=hccl\n"
    )
    for profile_name, rank, pid in (("PROF_A", 1, 202), ("PROF_B", 0, 101)):
        profile_dir = case_dir / "msprof" / profile_name
        device_dir = profile_dir / f"device_{rank}"
        device_dir.mkdir(parents=True)
        (device_dir / f"info.json.{rank}").write_text(json.dumps({
            "version": "1.0",
            "pid": str(pid),
            "devices": str(rank),
            "DeviceInfo": [{"id": rank}],
        }))
        with sqlite3.connect(profile_dir / f"msprof_{rank}.db") as connection:
            connection.executescript(
                f"""
                CREATE TABLE STRING_IDS(id INTEGER PRIMARY KEY, value TEXT);
                    CREATE TABLE TASK(startNs INTEGER, endNs INTEGER, deviceId INTEGER, globalTaskId INTEGER, streamId INTEGER, taskId INTEGER);
                    CREATE TABLE COMPUTE_TASK_INFO(
                        name INTEGER, globalTaskId INTEGER PRIMARY KEY, blockDim INTEGER,
                        mixBlockDim INTEGER, opType INTEGER, taskType INTEGER
                    );
                    INSERT INTO STRING_IDS VALUES(1, 'rank-{rank}-kernel');
                    INSERT INTO STRING_IDS VALUES(2, 'MatMul');
                    INSERT INTO STRING_IDS VALUES(3, 'AI_CORE');
                    INSERT INTO STRING_IDS VALUES(4, 'aclnnEmbedding_GatherV2AiCore_GatherV2');
                    INSERT INTO STRING_IDS VALUES(5, 'GatherV2');
                    INSERT INTO TASK VALUES(50000, 50100, {rank}, 9, 2, 9);
                    INSERT INTO TASK VALUES(110000, 110100, {rank}, 10, 2, 10);
                    INSERT INTO TASK VALUES(150000, 150100, {rank}, 11, 2, 11);
                    INSERT INTO COMPUTE_TASK_INFO VALUES(4, 9, 8, 0, 5, 3);
                    INSERT INTO COMPUTE_TASK_INFO VALUES(1, 10, 8, 0, 2, 3);
                    INSERT INTO COMPUTE_TASK_INFO VALUES(4, 11, 8, 0, 5, 3);
                """
            )
    raw_scheduler = (_scheduler_record(0, 100, {"req_0": 4}, ("req_0",)),)
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (4,))

    sources, launches, timing = _normalize_profiler(
        case_dir,
        "ascend_910b",
        raw_scheduler,
        scheduler,
        expected_ranks=(0, 1),
        operator_records=_ascend_operator_events(scheduler, (0, 1)),
    )

    assert [source["archive_id"] for source in sources] == [
        "device-profiler-rank0",
        "profiler-metadata-rank0",
        "device-profiler-rank1",
        "profiler-metadata-rank1",
    ]
    launch_events = tuple(
        record for record in launches.payload if record["record_type"] == "event"
    )
    assert [(record["rank"], record["step_id"]) for record in launch_events] == [
        (0, "step-000000000000"),
        (0, "step-000000000000"),
        (1, "step-000000000000"),
        (1, "step-000000000000"),
    ]
    signature_by_id = {
        record["signature_id"]: record
        for record in launches.payload if record["record_type"] == "signature"
    }
    assert [
        signature_by_id[event["signature_id"]]["device"]
        for event in launch_events
    ] == [0, 0, 1, 1]
    assert {record["source_archive_id"] for record in timing.payload} == {
        "device-profiler-rank0",
        "device-profiler-rank1",
    }


def test_ascend_profiler_allows_unambiguous_single_rank_without_worker_evidence(tmp_path):
    case_dir = tmp_path / "case"
    profile_dir = case_dir / "msprof" / "PROF_A"
    device_dir = profile_dir / "device_0"
    device_dir.mkdir(parents=True)
    (device_dir / "info.json.0").write_text(json.dumps({
        "version": "1.0",
        "pid": "101",
        "devices": "0",
        "DeviceInfo": [{"id": 0}],
    }))
    database = profile_dir / "msprof_0.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE STRING_IDS(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE TASK(startNs INTEGER, endNs INTEGER, deviceId INTEGER, globalTaskId INTEGER, streamId INTEGER, taskId INTEGER);
            CREATE TABLE COMPUTE_TASK_INFO(
                name INTEGER, globalTaskId INTEGER PRIMARY KEY, blockDim INTEGER,
                mixBlockDim INTEGER, opType INTEGER, taskType INTEGER
            );
            INSERT INTO STRING_IDS VALUES(1, 'kernel');
            INSERT INTO STRING_IDS VALUES(2, 'MatMul');
            INSERT INTO STRING_IDS VALUES(3, 'AI_CORE');
            INSERT INTO STRING_IDS VALUES(4, 'aclnnEmbedding_GatherV2AiCore_GatherV2');
            INSERT INTO STRING_IDS VALUES(5, 'GatherV2');
            INSERT INTO TASK VALUES(50000, 50100, 0, 9, 2, 9);
            INSERT INTO TASK VALUES(110000, 110100, 0, 10, 2, 10);
            INSERT INTO TASK VALUES(150000, 150100, 0, 11, 2, 11);
            INSERT INTO COMPUTE_TASK_INFO VALUES(4, 9, 8, 0, 5, 3);
            INSERT INTO COMPUTE_TASK_INFO VALUES(1, 10, 8, 0, 2, 3);
            INSERT INTO COMPUTE_TASK_INFO VALUES(4, 11, 8, 0, 5, 3);
            """
        )
    (case_dir / "capture.log").write_text("no worker rank mapping\n")
    raw_scheduler = (_scheduler_record(0, 100, {"req_0": 4}, ("req_0",)),)
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (4,))

    sources, launches, _ = _normalize_profiler(
        case_dir,
        "ascend_910b",
        raw_scheduler,
        scheduler,
        expected_ranks=(0,),
        operator_records=_ascend_operator_events(scheduler),
    )

    assert [source["archive_id"] for source in sources] == [
        "device-profiler-rank0",
        "profiler-metadata-rank0",
    ]
    assert launches.status == "available"


def test_ascend_unknown_op_type_id_fails_loud(tmp_path):
    case_dir = tmp_path / "case"
    profile_dir = case_dir / "msprof" / "PROF_1"
    device_dir = profile_dir / "device_0"
    device_dir.mkdir(parents=True)
    database = profile_dir / "msprof_1.db"
    with sqlite3.connect(database) as connection:
        connection.executescript(
            """
            CREATE TABLE STRING_IDS(id INTEGER PRIMARY KEY, value TEXT);
            CREATE TABLE TASK(startNs INTEGER, endNs INTEGER, deviceId INTEGER, globalTaskId INTEGER, streamId INTEGER, taskId INTEGER);
            CREATE TABLE COMPUTE_TASK_INFO(
                name INTEGER, globalTaskId INTEGER PRIMARY KEY, blockDim INTEGER,
                mixBlockDim INTEGER, opType INTEGER, taskType INTEGER
            );
            INSERT INTO STRING_IDS VALUES(1, 'kernel');
            INSERT INTO STRING_IDS VALUES(2, 'AI_CORE');
            INSERT INTO STRING_IDS VALUES(3, 'aclnnEmbedding_GatherV2AiCore_GatherV2');
            INSERT INTO STRING_IDS VALUES(4, 'GatherV2');
            INSERT INTO TASK VALUES(50000, 50100, 0, 9, 2, 9);
            INSERT INTO TASK VALUES(110000, 110100, 0, 10, 2, 10);
            INSERT INTO TASK VALUES(150000, 150100, 0, 11, 2, 11);
            INSERT INTO COMPUTE_TASK_INFO VALUES(3, 9, 8, 0, 4, 2);
            INSERT INTO COMPUTE_TASK_INFO VALUES(1, 10, 8, 0, 999, 2);
            INSERT INTO COMPUTE_TASK_INFO VALUES(3, 11, 8, 0, 4, 2);
            """
        )
    (device_dir / "info.json.0").write_text(json.dumps({
        "version": "1.0",
        "pid": "101",
        "devices": "0",
        "DeviceInfo": [{"id": 0}],
    }))
    (case_dir / "capture.log").write_text(
        "(Worker pid=101) INFO world_size=1 rank=0 local_rank=0 backend=hccl\n"
    )
    raw_scheduler = (_scheduler_record(0, 100, {"req_0": 4}, ("req_0",)),)
    scheduler, _, _ = _normalized_scheduler(raw_scheduler, {"req_0": 0}, (4,))

    with pytest.raises(ValueError, match="opType references unknown STRING_IDS ID"):
        _normalize_profiler(
            case_dir,
            "ascend_910b",
            raw_scheduler,
            scheduler,
            expected_ranks=(0,),
            operator_records=_ascend_operator_events(scheduler),
        )


def _compact_scheduler_record(step_id, phase):
    return {
        "step_id": step_id,
        "scheduled_requests": [{"request_idx": 0, "phase": phase}],
    }


def test_a800_sampler_tail_uses_unique_latest_direct_copy_and_not_sync_count():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE StringIds(id INTEGER, value TEXT)")
    connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(deviceId INTEGER, correlationId INTEGER, demangledName INTEGER, start INTEGER, end INTEGER)")
    connection.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INTEGER, correlationId INTEGER, nameId INTEGER)")
    connection.executemany("INSERT INTO StringIds VALUES (?, ?)", ((1, "cudaLaunchKernel"), (2, "direct_copy_kernel_cuda sampler input"), (3, "direct_copy_kernel_cuda sampler tail")))
    connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES (?, ?, ?)", ((200, 20, 1), (100, 10, 1)))
    connection.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?)", ((1, 20, 2, 10, 11), (0, 10, 2, 10, 11), (0, 10, 2, 20, 21), (0, 10, 3, 30, 31), (0, 10, 3, 40, 41)))
    assert _a800_rank_devices(connection) == (0, 1)
    assert _a800_sampler_tail_anchor_ends(connection, 0, 2) == (31, 41)
