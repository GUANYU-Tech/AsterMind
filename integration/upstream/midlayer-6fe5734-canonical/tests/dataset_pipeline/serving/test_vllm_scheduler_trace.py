"""Tests for the Serving Scheduler trace dataset adapter."""

import json
import os
from pathlib import Path
import pstats
import sys
import time
from types import ModuleType
from types import SimpleNamespace

import pytest

from dataset_pipeline.serving_dataset import vllm_scheduler_trace
from dataset_pipeline.serving_dataset.vllm_scheduler_trace import (
    _PDHandoffRecorder,
    _TraceState,
    _base_request_id,
    _install_p2p_capture_bootstrap_for,
    _install_p2p_handoff_wrappers_for,
    _request_index,
)


def test_request_index_accepts_vllm_completion_prefix():
    assert _request_index("request-000000000123") == 123
    assert _request_index("cmpl-request-000000000123") == 123
    assert _request_index("cmpl-request-000000000123-0") == 123
    assert _request_index("cmpl-request-000000000123-0-943ccd3e") == 123
    assert _request_index("sharegpt-warmup") is None
    assert _request_index("cmpl-sharegpt-warmup") is None


def test_explicit_in_process_trace_finalize(monkeypatch, tmp_path):
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    state = _TraceState(output_path, 2)
    state.finished_request_indexes = {7, 11}

    def write_complete_trace():
        output_path.write_text("sealed\n", encoding="utf-8")
        state.written = True

    monkeypatch.setattr(state, "_write_complete_trace", write_complete_trace)
    monkeypatch.setattr(vllm_scheduler_trace, "_ACTIVE_TRACE_STATE", state)
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_OUTPUT", str(output_path))

    assert vllm_scheduler_trace.finalize_scheduler_trace_from_environment() is True
    assert output_path.read_text(encoding="utf-8") == "sealed\n"


def test_explicit_in_process_trace_finalize_fails_before_completion(
    monkeypatch, tmp_path
):
    state = _TraceState(tmp_path / "scheduler_steps.raw.jsonl", 2)
    state.finished_request_indexes = {7}
    monkeypatch.setattr(vllm_scheduler_trace, "_ACTIVE_TRACE_STATE", state)
    monkeypatch.setenv(
        "MIDLAYER_SCHEDULER_TRACE_OUTPUT",
        str(tmp_path / "scheduler_steps.raw.jsonl"),
    )

    with pytest.raises(RuntimeError, match="1 != 2"):
        vllm_scheduler_trace.finalize_scheduler_trace_from_environment()


def test_in_process_trace_reports_pending_depth(monkeypatch, tmp_path):
    state = _TraceState(tmp_path / "scheduler_steps.raw.jsonl", 1)
    state.pending[7] = {"trace_step_id": "vllm:7"}
    monkeypatch.setattr(vllm_scheduler_trace, "_ACTIVE_TRACE_STATE", state)
    monkeypatch.setenv(
        "MIDLAYER_SCHEDULER_TRACE_OUTPUT",
        str(tmp_path / "scheduler_steps.raw.jsonl"),
    )

    assert (
        vllm_scheduler_trace.scheduler_trace_pending_step_count_from_environment()
        == 1
    )


def test_request_index_accepts_pd_role_suffixes():
    assert _request_index(
        "cmpl-request-000000000123___decode_addr_10.0.0.2:25000-0-943ccd3e"
    ) == 123
    assert _request_index(
        "cmpl-request-000000000123___prefill_addr_10.0.0.1:24000___-0-a4be912c"
    ) == 123


def test_base_request_id_normalizes_completion_and_pd_role_suffixes():
    assert _base_request_id("cmpl-request-000000000123-0-943ccd3e") == (
        "request-000000000123"
    )
    assert _base_request_id(
        "cmpl-request-000000000123___decode_addr_10.0.0.2:25000-0-943ccd3e"
    ) == "request-000000000123"
    assert _base_request_id(
        "request-000000000123___prefill_addr_10.0.0.1:24000___-0-a4be912c"
    ) == "request-000000000123"


def test_p2p_capture_bootstrap_skips_only_metadata_free_graph_dummy_runs():
    calls = []

    class FakeConnector:
        def __init__(self):
            self._connector_metadata = None
            self.is_producer = True
            self.chunked_prefill = {}

        def save_kv_layer(self, value):
            calls.append(value)
            if self._connector_metadata is None:
                raise AssertionError("metadata is required")
            return value

        def build_connector_meta(self, scheduler_output):
            for request_id in scheduler_output.scheduled_cached_reqs.req_ids:
                assert request_id in self.chunked_prefill
            return tuple(scheduler_output.scheduled_cached_reqs.req_ids)

        def request_finished(self, request, block_ids):
            return request.request_id, block_ids

        def start_load_kv(self):
            return None

    connector = FakeConnector()

    class FakeModelRunner:
        def capture_model(self):
            connector.save_kv_layer("dummy")
            return 17

    _install_p2p_capture_bootstrap_for(FakeConnector, FakeModelRunner)

    assert FakeModelRunner().capture_model() == 17
    assert calls == []
    with pytest.raises(AssertionError, match="metadata is required"):
        connector.save_kv_layer("missing")
    connector._connector_metadata = object()
    assert connector.save_kv_layer("real") == "real"
    assert calls == ["missing", "real"]

    complete = SimpleNamespace(
        req_id="request-0",
        prompt_token_ids=[1, 2],
        num_computed_tokens=0,
    )
    empty_cached = SimpleNamespace(
        req_ids=[],
        resumed_from_preemption=[],
        new_token_ids=[],
        new_block_ids=[],
        num_computed_tokens=[],
    )
    first = SimpleNamespace(
        scheduled_new_reqs=[complete],
        scheduled_cached_reqs=empty_cached,
        num_scheduled_tokens={"request-0": 2},
    )
    assert connector.build_connector_meta(first) == ()
    speculative = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["request-0"],
            resumed_from_preemption=[False],
            new_token_ids=[[]],
            new_block_ids=[([],)],
            num_computed_tokens=[2],
        ),
        num_scheduled_tokens={"request-0": 1},
    )
    assert connector.build_connector_meta(speculative) == ()
    assert speculative.scheduled_cached_reqs.req_ids == ["request-0"]
    assert connector.request_finished(SimpleNamespace(request_id="request-0"), []) == (
        "request-0",
        [],
    )

    unexpected = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["request-1"],
            resumed_from_preemption=[False],
            new_token_ids=[[]],
            new_block_ids=[([],)],
            num_computed_tokens=[2],
        ),
        num_scheduled_tokens={"request-1": 1},
    )
    with pytest.raises(AssertionError):
        connector.build_connector_meta(unexpected)


def test_runtime_depth_evidence_accepts_vllm_config_field(monkeypatch, tmp_path):
    class FakeVllmConfig:
        max_concurrent_batches = 2
        parallel_config = SimpleNamespace(pipeline_parallel_size=1)

    class FakeExecutor:
        vllm_config = FakeVllmConfig()

    vllm_module = ModuleType("vllm")
    vllm_module.__version__ = "0.24.0"
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    state = _TraceState(
        tmp_path / "scheduler.jsonl",
        1,
        configured_cudagraph_mode="FULL_AND_PIECEWISE",
        max_in_flight=2,
        hardware_id="metax_c500",
        worker_rank_count=None,
    )

    evidence = state._runtime_depth_evidence(FakeExecutor())

    assert evidence["configured_depth"] == 2
    assert evidence["source_kind"] == "runtime_vllm_config_field"
    assert evidence["source_field"] == "vllm_config.max_concurrent_batches"
    assert evidence["executor_class"].endswith("FakeExecutor")


def test_runtime_metadata_preserves_scheduler_clock_origin(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "MIDLAYER_SCHEDULER_TRACE_SOURCE_IDENTITY", "unit-test-source"
    )
    output_path = tmp_path / "scheduler.jsonl"
    state = _TraceState(
        output_path,
        1,
        configured_cudagraph_mode="FULL_AND_PIECEWISE",
        max_in_flight=2,
        hardware_id="nvidia_a800",
        worker_rank_count=1,
    )
    state.configured_depth_evidence = {
        "source_kind": "runtime_executor_property",
        "source_field": "max_concurrent_batches",
        "configured_depth": 2,
        "executor_class": "unit.test.Executor",
        "pipeline_parallel_size": 1,
        "runtime_version": "unit-test",
        "source_identity": {
            "module_file": "/unit/test.py",
            "module_name": "unit.test",
        },
    }

    state._write_runtime_metadata(123456789)

    runtime = json.loads(Path(f"{output_path}.runtime.json").read_text())
    clock = json.loads(Path(f"{output_path}.clock.json").read_text())
    assert runtime["scheduler_pipeline_depth"] == 2
    assert runtime["scheduler_pipeline_depth_evidence"]["source_identity"] == (
        "unit-test-source"
    )
    assert clock["clock_domain"] == "shared_host_time.perf_counter_ns"
    assert clock["trace_origin_monotonic_ns"] == 123456789


def test_pd_handoff_recorder_buffers_until_complete(monkeypatch, tmp_path):
    monkeypatch.setenv("MIDLAYER_PD_HANDOFF_EVENTS_DIR", str(tmp_path))
    monkeypatch.setenv("MIDLAYER_PD_HANDOFF_CONNECTOR", "P2pMcclConnector")
    recorder = _PDHandoffRecorder({
        "run_id": "run-0",
        "run_origin_monotonic_ns": 100,
        "host_boot_id": "boot-0",
        "instance_role": "prefill",
        "instance_id": "prefill-0",
    })
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.vllm_scheduler_trace.time.perf_counter_ns",
        lambda: 200,
    )

    recorder.record(
        "transfer_start",
        request_id="request-0",
        kv_tokens=16,
        kv_blocks=1,
        byte_count=1024,
    )

    assert not recorder.path.exists()
    recorder.write_complete()
    record = json.loads(recorder.path.read_text())
    assert isinstance(record.pop("process_pid"), int)
    assert record == {
        "base_request_id": "request-0",
        "byte_count": 1024,
        "clock_domain": "shared_host_time.perf_counter_ns",
        "connector": "P2pMcclConnector",
        "event": "transfer_start",
        "event_kind": "pd_connector_handoff",
        "host_boot_id": "boot-0",
        "instance_id": "prefill-0",
        "instance_role": "prefill",
        "kv_blocks": 1,
        "kv_tokens": 16,
        "request_id": "request-0",
        "run_id": "run-0",
        "run_origin_monotonic_ns": 100,
        "schema_version": 1,
        "timestamp_ns": 100,
    }


def test_pd_handoff_recorder_write_through_persists_each_observation(
        monkeypatch, tmp_path):
    monkeypatch.setenv("MIDLAYER_PD_HANDOFF_EVENTS_DIR", str(tmp_path))
    monkeypatch.setenv("MIDLAYER_PD_HANDOFF_CONNECTOR", "P2pNcclConnector")
    recorder = _PDHandoffRecorder({
        "run_id": "run-0",
        "run_origin_monotonic_ns": 100,
        "host_boot_id": "boot-0",
        "instance_role": "decode",
        "instance_id": "decode-0",
    }, write_through=True)
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.vllm_scheduler_trace.time.perf_counter_ns",
        lambda: 200,
    )

    recorder.record(
        "consumer_load_start",
        request_id="request-0",
        kv_tokens=None,
        kv_blocks=None,
        byte_count=None,
    )
    recorder.write_complete()
    recorder.record(
        "consumer_load_end",
        request_id="request-0",
        kv_tokens=None,
        kv_blocks=None,
        byte_count=None,
    )
    recorder.write_complete()

    records = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "consumer_load_start", "consumer_load_end"
    ]
    assert records[0]["timestamp_ns"] == 100


def test_pd_handoff_recorder_buffers_runtime_writes_until_flush(
        monkeypatch, tmp_path):
    monkeypatch.setenv("MIDLAYER_PD_HANDOFF_EVENTS_DIR", str(tmp_path))
    monkeypatch.setenv("MIDLAYER_PD_HANDOFF_CONNECTOR", "P2pNcclConnector")
    recorder = _PDHandoffRecorder({
        "run_id": "run-0",
        "run_origin_monotonic_ns": 100,
        "host_boot_id": "boot-0",
        "instance_role": "decode",
        "instance_id": "decode-0",
    }, write_through=True)

    class Handle:
        def __init__(self):
            self.payload = ""
            self.flush_count = 0

        def write(self, value):
            self.payload += value

        def flush(self):
            self.flush_count += 1

    handle = Handle()
    recorder.write_through_handle = handle
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.vllm_scheduler_trace.time.perf_counter_ns",
        lambda: 200,
    )

    recorder.record(
        "consumer_load_start",
        request_id="request-0",
        kv_tokens=None,
        kv_blocks=None,
        byte_count=None,
    )

    assert handle.flush_count == 0
    recorder.write_complete()
    assert handle.flush_count == 1
    assert json.loads(handle.payload)["event"] == "consumer_load_start"


def test_p2p_capture_bootstrap_uses_aggregated_worker_receive_evidence():
    class Connector:
        _midlayer_p2p_capture_bootstrap_installed = True

        def build_connector_meta(self):
            return None

    class Engine:
        def send_sync(self):
            return None

        def recv_tensor(self):
            return None

    original_recv_tensor = Engine.recv_tensor

    _install_p2p_handoff_wrappers_for(object(), Connector, Engine)

    assert Engine.recv_tensor is original_recv_tensor


def test_p2p_capture_bootstrap_keeps_unknown_receive_bytes_null(
        monkeypatch, tmp_path):
    for name, value in {
        "MIDLAYER_PD_HANDOFF_EVENTS_DIR": str(tmp_path),
        "MIDLAYER_PD_HANDOFF_CONNECTOR": "P2pNcclConnector",
        "MIDLAYER_PD_RUN_ID": "run-0",
        "MIDLAYER_PD_RUN_ORIGIN_MONOTONIC_NS": "100",
        "MIDLAYER_PD_HOST_BOOT_ID": "boot-0",
        "MIDLAYER_PD_INSTANCE_ROLE": "decode",
        "MIDLAYER_PD_INSTANCE_ID": "decode-0",
    }.items():
        monkeypatch.setenv(name, value)

    class Engine:
        def recv_tensor(self, tensor_id, remote_address=None):
            return None

    class Connector:
        def __init__(self):
            self.is_producer = False
            self.p2p_nccl_engine = Engine()

        def save_kv_layer(self):
            return None

        def build_connector_meta(self, scheduler_output):
            return scheduler_output

        def request_finished(self, request):
            return request

        def start_load_kv(self):
            return self.p2p_nccl_engine.recv_tensor("request-0#kv")

    class ModelRunner:
        def capture_model(self):
            return None

    _install_p2p_capture_bootstrap_for(Connector, ModelRunner)
    connector = Connector()

    assert connector.start_load_kv() is None
    recorder = connector._midlayer_p2p_runtime_handoff_recorder
    recorder.write_complete()
    rows = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    assert [row["event"] for row in rows] == [
        "transfer_start", "consumer_load_start", "transfer_end", "consumer_load_end",
    ]
    assert rows[-1]["byte_count"] is None


def test_trace_state_splits_prefill_decode_and_writes_once(monkeypatch, tmp_path):
    timestamps = iter((1000, 1100))
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.vllm_scheduler_trace.time.perf_counter_ns",
        lambda: next(timestamps),
    )
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    request_add_path = tmp_path / "scheduler_request_add.json"
    state = _TraceState(
        output_path, 1, request_add_output_path=request_add_path
    )
    state.after_add_request(
        SimpleNamespace(request_id="request-000000000000"), 900
    )
    scheduler = SimpleNamespace(requests={
        "request-000000000000": SimpleNamespace(
            num_computed_tokens=6,
            num_prompt_tokens=5,
        ),
    })
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request-000000000000": 3}
    )

    iteration, started_ns = state.before_schedule()
    state.after_schedule(
        scheduler, scheduler_output, iteration, started_ns
    )
    engine_outputs = {
        0: SimpleNamespace(
            outputs=[SimpleNamespace(
                request_id="request-000000000000",
                finished=True,
            )],
            finished_requests=None,
        ),
    }
    state.after_update(scheduler_output, 1200, 1300, engine_outputs)

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert records == [{
        "step_index": 0,
        "step_id": "vllm:0",
        "source_iteration_id": 0,
        "step_start_ns": 0,
        "schedule_end_ns": 100,
        "update_start_ns": 200,
        "step_end_ns": 300,
        "scheduled_requests": [{
            "request_idx": 0,
            "prefill_tokens": 2,
            "decode_tokens": 1,
            "scheduled_tokens": 3,
        }],
        "scheduled_request_count": 1,
        "prefill_request_count": 1,
        "decode_request_count": 1,
        "prefill_tokens": 2,
        "decode_tokens": 1,
        "scheduled_tokens": 3,
        "finished_request_indexes": [0],
        "cumulative_scheduled_tokens": 3,
        "cumulative_completed_requests": 1,
    }]
    assert json.loads(request_add_path.read_text()) == {
        "schema_version": 1,
        "clock_domain": "shared_host_time.perf_counter_ns",
        "boundary": "Scheduler.add_request return",
        "request_count": 1,
        "records": [{
            "request_idx": 0,
            "scheduler_add_end_monotonic_ns": 900,
        }],
    }


def test_trace_state_ignores_warmup_request(monkeypatch, tmp_path):
    timestamps = iter((1000, 1100))
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.vllm_scheduler_trace.time.perf_counter_ns",
        lambda: next(timestamps),
    )
    state = _TraceState(tmp_path / "scheduler_steps.raw.jsonl", 1)
    scheduler = SimpleNamespace(requests={
        "sharegpt-warmup": SimpleNamespace(
            num_computed_tokens=1,
            num_prompt_tokens=1,
        ),
    })
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"sharegpt-warmup": 1}
    )

    iteration, started_ns = state.before_schedule()
    state.after_schedule(
        scheduler, scheduler_output, iteration, started_ns
    )

    assert state.pending == {}
    assert not state.output_path.exists()


def test_trace_state_drains_async_pending_step_before_writing(tmp_path):
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    state = _TraceState(output_path, 1, max_in_flight=2)
    scheduler = SimpleNamespace(requests={
        "request-000000000000": SimpleNamespace(
            num_computed_tokens=2,
            num_prompt_tokens=2,
        ),
    })
    first = SimpleNamespace(
        num_scheduled_tokens={"request-000000000000": 1}
    )
    second = SimpleNamespace(
        num_scheduled_tokens={"request-000000000000": 1}
    )

    first_iteration, first_started = state.before_schedule()
    state.after_schedule(scheduler, first, first_iteration, first_started)
    second_iteration, second_started = state.before_schedule()
    state.after_schedule(scheduler, second, second_iteration, second_started)
    completed = {
        0: SimpleNamespace(
            outputs=[SimpleNamespace(
                request_id="request-000000000000",
                finished=True,
            )],
            finished_requests=None,
        ),
    }

    state.after_update(first, 10, 11, completed)

    assert len(state.pending) == 1
    assert not output_path.exists()
    state.after_update(second, 12, 13, {})
    assert state.pending == {}
    assert state.written is True
    assert len(output_path.read_text().splitlines()) == 2


def test_rich_trace_records_runtime_graph_and_async_boundaries(
    monkeypatch, tmp_path
):
    timestamps = iter((1000, 1100))
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.vllm_scheduler_trace.time.perf_counter_ns",
        lambda: next(timestamps),
    )
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    state = _TraceState(
        output_path,
        1,
        configured_cudagraph_mode="FULL_AND_PIECEWISE",
        max_in_flight=2,
    )
    scheduler = SimpleNamespace(
        requests={
            "request-000000000000": SimpleNamespace(
                num_computed_tokens=3,
                num_prompt_tokens=5,
            ),
        },
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(max_cudagraph_capture_size=512)
        ),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request-000000000000": 3}
    )
    iteration, started_ns = state.before_schedule()
    state.after_schedule(scheduler, scheduler_output, iteration, started_ns)
    state.executor_started(scheduler_output, 1200)
    state.executor_finished(scheduler_output, 1300)
    engine_outputs = {
        0: SimpleNamespace(
            outputs=[SimpleNamespace(
                request_id="request-000000000000",
                finished=True,
            )],
            finished_requests=None,
        ),
    }
    model_output = SimpleNamespace(cudagraph_stats=SimpleNamespace(
        num_unpadded_tokens=3,
        num_padded_tokens=4,
        num_paddings=1,
        runtime_mode="CUDAGraphMode.FULL",
    ))

    state.after_update(
        scheduler_output,
        1400,
        1500,
        engine_outputs,
        model_output,
    )

    record = json.loads(output_path.read_text())
    assert record["configured_cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert record["runtime_cudagraph_mode"] == "FULL"
    assert record["num_unpadded_tokens"] == record["scheduled_tokens"] == 3
    assert record["capture_bucket_tokens"] == 4
    assert record["padded_total_tokens"] == 4
    assert record["padding_tokens"] == 1
    assert record["graph_fallback_reason"] is None
    assert record["capture_limit_tokens"] == 512
    assert record["device_execution_start_ns"] == 200
    assert record["device_execution_end_ns"] == 300
    assert record["scheduler_update_start_ns"] == 400
    assert record["scheduler_update_end_ns"] == 500
    assert record["in_flight_step_count"] == 1


def test_eager_trace_records_no_capture_limit(monkeypatch, tmp_path):
    timestamps = iter((1000, 1100))
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.vllm_scheduler_trace.time.perf_counter_ns",
        lambda: next(timestamps),
    )
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    state = _TraceState(
        output_path,
        1,
        configured_cudagraph_mode="NONE",
        max_in_flight=2,
    )
    scheduler = SimpleNamespace(
        requests={
            "request-000000000000": SimpleNamespace(
                num_computed_tokens=3,
                num_prompt_tokens=5,
            ),
        },
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(max_cudagraph_capture_size=0)
        ),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request-000000000000": 1}
    )

    iteration, started_ns = state.before_schedule()
    state.after_schedule(scheduler, scheduler_output, iteration, started_ns)

    assert state.pending[id(scheduler_output)]["capture_limit_tokens"] is None
    state.executor_started(scheduler_output, 1200)
    state.executor_finished(scheduler_output, 1300)
    state.after_update(
        scheduler_output,
        1400,
        1500,
        {
            0: SimpleNamespace(
                outputs=[SimpleNamespace(
                    request_id="request-000000000000",
                    finished=True,
                )],
                finished_requests=None,
            ),
        },
        SimpleNamespace(cudagraph_stats=SimpleNamespace(
            num_unpadded_tokens=1,
            num_padded_tokens=1,
            num_paddings=0,
            runtime_mode="CUDAGraphMode.NONE",
        )),
    )
    record = json.loads(output_path.read_text())
    assert record["configured_cudagraph_mode"] == "NONE"
    assert record["runtime_cudagraph_mode"] == "NONE"
    assert record["capture_limit_tokens"] is None
    assert record["graph_fallback_reason"] is None


def test_scheduler_update_observation_closes_callback_race(monkeypatch, tmp_path):
    timestamps = iter((1000, 1100))
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.vllm_scheduler_trace.time.perf_counter_ns",
        lambda: next(timestamps),
    )
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    state = _TraceState(
        output_path,
        1,
        configured_cudagraph_mode="FULL_AND_PIECEWISE",
        max_in_flight=2,
    )
    scheduler = SimpleNamespace(
        requests={
            "request-000000000000": SimpleNamespace(
                num_computed_tokens=3,
                num_prompt_tokens=5,
            ),
        },
        vllm_config=SimpleNamespace(
            compilation_config=SimpleNamespace(max_cudagraph_capture_size=512)
        ),
    )
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request-000000000000": 3}
    )
    iteration, started_ns = state.before_schedule()
    state.after_schedule(scheduler, scheduler_output, iteration, started_ns)
    state.executor_started(scheduler_output, 1200)
    state.executor_completion_observed_by_scheduler(scheduler_output, 1300)
    model_output = SimpleNamespace(cudagraph_stats=SimpleNamespace(
        num_unpadded_tokens=3,
        num_padded_tokens=4,
        num_paddings=1,
        runtime_mode="CUDAGraphMode.FULL",
    ))
    completed = {
        0: SimpleNamespace(
            outputs=[SimpleNamespace(
                request_id="request-000000000000",
                finished=True,
            )],
            finished_requests=None,
        ),
    }

    state.after_update(scheduler_output, 1400, 1500, completed, model_output)

    record = json.loads(output_path.read_text())
    assert record["device_execution_end_ns"] == 300


def test_a800_rich_trace_merges_complete_per_rank_dispatch_evidence(
    monkeypatch, tmp_path
):
    timestamps = iter((1000, 1100))
    monkeypatch.setattr(
        "dataset_pipeline.serving_dataset.vllm_scheduler_trace.time.perf_counter_ns",
        lambda: next(timestamps),
    )
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    state = _TraceState(
        output_path,
        1,
        configured_cudagraph_mode="FULL_AND_PIECEWISE",
        max_in_flight=2,
        hardware_id="nvidia_a800",
        worker_rank_count=2,
    )
    scheduler = SimpleNamespace(requests={
        "request-000000000000": SimpleNamespace(
            num_computed_tokens=3,
            num_prompt_tokens=5,
        ),
    })
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request-000000000000": 3},
        total_num_scheduled_tokens=3,
    )
    iteration, started_ns = state.before_schedule()
    state.after_schedule(scheduler, scheduler_output, iteration, started_ns)
    state.executor_started(scheduler_output, 1150)

    def observation(rank, mode, selected, fallback):
        return {
            "schema_version": 1,
            "source": "vllm_v1_worker_cudagraph_dispatch",
            "rank": rank,
            "trace_step_id": "vllm:0",
            "scheduled_tokens": 3,
            "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
            "runtime_cudagraph_mode": mode,
            "dispatch_input_key": {
                "num_tokens": 4,
                "uniform_decode": False,
            },
            "selected_graph_key": selected,
            "graph_fallback_reason": fallback,
            "num_unpadded_tokens": 3,
            "padded_total_tokens": 4,
            "padding_tokens": 1,
            "worker_execution_start_ns": 1200,
            "worker_execution_end_ns": 1300,
            "worker_execution_clock": "time.perf_counter_ns",
            "worker_execution_source": "vllm_v1_worker_execute_model_envelope",
        }

    (tmp_path / "scheduler_steps.raw.jsonl.rank-0.jsonl").write_text(
        json.dumps(observation(
            0,
            "FULL",
            {"num_tokens": 4, "uniform_decode": False},
            None,
        )) + "\n"
    )
    (tmp_path / "scheduler_steps.raw.jsonl.rank-1.jsonl").write_text(
        json.dumps(observation(
            1,
            "NONE",
            None,
            "runtime_dispatch_no_matching_graph",
        )) + "\n"
    )
    state.executor_finished(scheduler_output, 1400)
    engine_outputs = {
        0: SimpleNamespace(
            outputs=[SimpleNamespace(
                request_id="request-000000000000",
                finished=True,
            )],
            finished_requests=None,
        ),
    }
    state.after_update(scheduler_output, 1450, 1500, engine_outputs)

    record = json.loads(output_path.read_text())
    assert record["in_flight_step_count"] == 1
    assert record["rank_observations"][0]["dispatch_input_key"] == {
        "num_tokens": 4,
        "uniform_decode": False,
    }
    assert record["rank_observations"][0]["selected_graph_key"] == {
        "num_tokens": 4,
        "uniform_decode": False,
    }
    assert record["rank_observations"][1]["selected_graph_key"] is None
    assert record["rank_observations"][1]["graph_fallback_reason"] == (
        "runtime_dispatch_no_matching_graph"
    )


def test_a800_rank_observations_are_read_incrementally(tmp_path):
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    state = _TraceState(
        output_path,
        1,
        configured_cudagraph_mode="FULL_AND_PIECEWISE",
        hardware_id="nvidia_a800",
        worker_rank_count=1,
    )
    observation_path = tmp_path / "scheduler_steps.raw.jsonl.rank-0.jsonl"
    state._read_a800_rank_observations(0)
    assert state.a800_rank_observation_cache == {0: {}}
    observation_path.write_text(
        '{"trace_step_id":"vllm:0"}\n'
        '{"trace_step_id":"vllm:1"}\n'
        '{"trace_step_id":"vl',
        encoding="utf-8",
    )
    state._read_a800_rank_observations(0)
    # A row still being appended is left on disk and re-read whole later.
    assert list(state.a800_rank_observation_cache[0]) == ["vllm:0", "vllm:1"]
    with observation_path.open("a", encoding="utf-8") as stream:
        stream.write('lm:2"}\n')
    state._read_a800_rank_observations(0)
    assert list(state.a800_rank_observation_cache[0]) == [
        "vllm:0", "vllm:1", "vllm:2"
    ]
    assert state.a800_rank_observation_cache[0]["vllm:2"] == {
        "trace_step_id": "vllm:2"
    }


def _a800_worker_observation(rank, step, scheduled_tokens, start_ns):
    return {
        "schema_version": 1,
        "source": "vllm_v1_worker_cudagraph_dispatch",
        "rank": rank,
        "trace_step_id": step,
        "scheduled_tokens": scheduled_tokens,
        "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
        "runtime_cudagraph_mode": "FULL",
        "dispatch_input_key": {"num_tokens": 4, "uniform_decode": False},
        "selected_graph_key": {"num_tokens": 4, "uniform_decode": False},
        "graph_fallback_reason": None,
        "num_unpadded_tokens": scheduled_tokens,
        "padded_total_tokens": 4,
        "padding_tokens": 4 - scheduled_tokens,
        "worker_execution_start_ns": start_ns + rank,
        "worker_execution_end_ns": start_ns + 50 + rank,
        "worker_execution_clock": "time.perf_counter_ns",
        "worker_execution_source": "vllm_v1_worker_execute_model_envelope",
    }


def _run_a800_tp2_batch_queue_trace(monkeypatch, tmp_path, late_rank_1):
    """Drive two depth-2 formal steps through a TP2 vLLM 0.11 trace.

    ``late_rank_1`` reproduces the H200 ordering: MultiprocExecutor waits
    only for the output rank, so rank 1's row for step N lands after the
    Scheduler's ``update_from_output`` for step N (its file does not even
    exist at the first update, and a row may be half-appended).
    """

    clock = iter(range(1000, 100000, 10))
    monkeypatch.setattr(
        vllm_scheduler_trace.time, "perf_counter_ns", lambda: next(clock)
    )
    tmp_path.mkdir()
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    state = _TraceState(
        output_path,
        2,
        configured_cudagraph_mode="FULL_AND_PIECEWISE",
        max_in_flight=2,
        hardware_id="nvidia_a800",
        worker_rank_count=2,
    )
    rows = {
        (rank, step): json.dumps(
            _a800_worker_observation(rank, f"vllm:{step}", 3, 2000 + 100 * step),
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        for rank in (0, 1)
        for step in (0, 1)
    }

    def append(rank, text):
        path = Path(f"{output_path}.rank-{rank}.jsonl")
        with path.open("a", encoding="utf-8") as stream:
            stream.write(text)

    def finished(request_id):
        return {0: SimpleNamespace(
            outputs=[SimpleNamespace(request_id=request_id, finished=True)],
            finished_requests=None,
        )}

    scheduler = SimpleNamespace(requests={
        "request-000000000000": SimpleNamespace(
            num_computed_tokens=3, num_prompt_tokens=3
        ),
        "request-000000000001": SimpleNamespace(
            num_computed_tokens=3, num_prompt_tokens=3
        ),
    })
    outputs = [
        SimpleNamespace(
            num_scheduled_tokens={f"request-00000000000{step}": 3},
            total_num_scheduled_tokens=3,
        )
        for step in (0, 1)
    ]
    for step in (0, 1):
        iteration, started_ns = state.before_schedule()
        state.after_schedule(scheduler, outputs[step], iteration, started_ns)
        state.executor_started(outputs[step], 1900 + 100 * step)
    state.executor_finished(outputs[0], 2060)
    append(0, rows[(0, 0)])
    if late_rank_1:
        append(1, rows[(1, 0)][:40])
    else:
        append(1, rows[(1, 0)])
    state.after_update(outputs[0], 2070, 2080, finished("request-000000000000"))
    if late_rank_1:
        assert [entry["trace_step_id"] for entry in state.a800_unresolved_steps] == [
            "vllm:0"
        ]
        append(1, rows[(1, 0)][40:])
    state.executor_finished(outputs[1], 2160)
    append(0, rows[(0, 1)])
    sleeps = []
    if late_rank_1:
        # Rank 1's final row arrives only while the seal is waiting.
        def late_sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) == 1:
                append(1, rows[(1, 1)])

        monkeypatch.setattr(vllm_scheduler_trace.time, "sleep", late_sleep)
    else:
        append(1, rows[(1, 1)])
    state.after_update(outputs[1], 2170, 2180, finished("request-000000000001"))
    assert state.a800_unresolved_steps == []
    assert len(sleeps) == (1 if late_rank_1 else 0)
    return output_path.read_bytes()


def test_a800_late_rank_observation_trace_matches_in_order_trace(
    monkeypatch, tmp_path
):
    in_order = _run_a800_tp2_batch_queue_trace(
        monkeypatch, tmp_path / "in_order", late_rank_1=False
    )
    late = _run_a800_tp2_batch_queue_trace(
        monkeypatch, tmp_path / "late", late_rank_1=True
    )
    assert late == in_order
    records = [json.loads(line) for line in in_order.decode().splitlines()]
    assert [
        [observation["rank"] for observation in record["rank_observations"]]
        for record in records
    ] == [[0, 1], [0, 1]]
    assert [
        record["rank_observations"][1]["trace_step_id"] for record in records
    ] == ["vllm:0", "vllm:1"]


def test_a800_missing_rank_observation_fails_loud_at_seal(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vllm_scheduler_trace, "_WORKER_OBSERVATION_ARRIVAL_TIMEOUT_S", 0.02
    )
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    state = _TraceState(
        output_path,
        1,
        configured_cudagraph_mode="FULL_AND_PIECEWISE",
        hardware_id="nvidia_a800",
        worker_rank_count=2,
    )
    scheduler = SimpleNamespace(requests={
        "request-000000000000": SimpleNamespace(
            num_computed_tokens=3, num_prompt_tokens=3
        ),
    })
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"request-000000000000": 3},
        total_num_scheduled_tokens=3,
    )
    iteration, started_ns = state.before_schedule()
    state.after_schedule(scheduler, scheduler_output, iteration, started_ns)
    state.executor_started(scheduler_output, 1900)
    state.executor_finished(scheduler_output, 2000)
    Path(f"{output_path}.rank-0.jsonl").write_text(
        json.dumps(_a800_worker_observation(0, "vllm:0", 3, 1950)) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        RuntimeError,
        match=r"did not arrive within 0\.02 s .*\('vllm:0', \[1\]\)",
    ):
        state.after_update(scheduler_output, 2010, 2020, {0: SimpleNamespace(
            outputs=[SimpleNamespace(
                request_id="request-000000000000", finished=True
            )],
            finished_requests=None,
        )})
    assert not output_path.exists()


def test_install_forwards_scheduler_arguments(monkeypatch, tmp_path):
    class FakeScheduler:
        def schedule(self, throttle_prefills=False):
            self.throttle_prefills = throttle_prefills
            return SimpleNamespace(num_scheduled_tokens={})

        def update_from_output(self, scheduler_output, model_runner_output):
            return {}

    scheduler_module = ModuleType("vllm.v1.core.sched.scheduler")
    scheduler_module.Scheduler = FakeScheduler
    monkeypatch.setitem(
        sys.modules, "vllm.v1.core.sched.scheduler", scheduler_module
    )
    monkeypatch.setenv(
        "MIDLAYER_SCHEDULER_TRACE_OUTPUT", str(tmp_path / "scheduler.jsonl")
    )
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_REQUEST_COUNT", "1")
    wrapper = tmp_path / "spawn.sh"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.setenv(
        "MIDLAYER_SCHEDULER_TRACE_SPAWN_WRAPPER", str(wrapper))
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_SPAWN_PYTHON", sys.executable)
    monkeypatch.setenv(
        "MIDLAYER_SCHEDULER_TRACE_SPAWN_PYTHONPATH", "/trace-bootstrap:/repo")
    monkeypatch.setattr(vllm_scheduler_trace, "_INSTALLED", False)

    assert vllm_scheduler_trace.install_from_environment() is True
    scheduler = FakeScheduler()
    scheduler.requests = {}

    scheduler.schedule(True)

    assert scheduler.throttle_prefills is True


def test_ascend_step_timestamp_marks_only_formal_scheduler_steps(
    monkeypatch, tmp_path
):
    calls = []

    class FakeFunction:
        def __init__(self, name):
            self.name = name
            self.restype = None
            self.argtypes = None

        def __call__(self, *args):
            calls.append((self.name, args))
            if self.name == "create":
                return 91
            return 0

    class FakeProfiler:
        aclprofCreateStepInfo = FakeFunction("create")
        aclprofGetStepTimestamp = FakeFunction("timestamp")
        aclprofDestroyStepInfo = FakeFunction("destroy")

    class FakeWorker:
        def __init__(self, rank):
            self.rank = rank

        def execute_model(self, scheduler_output):
            calls.append(("execute", scheduler_output))
            return "model-output"

    ascend_module = ModuleType("vllm_ascend")
    worker_package = ModuleType("vllm_ascend.worker")
    worker_module = ModuleType("vllm_ascend.worker.worker")
    worker_module.NPUWorker = FakeWorker
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker", worker_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", worker_module)
    torch_npu_module = ModuleType("torch_npu")
    torch_npu_module.npu = SimpleNamespace(
        current_stream=lambda: SimpleNamespace(npu_stream=19)
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_module)
    monkeypatch.setattr(
        vllm_scheduler_trace.ctypes, "CDLL", lambda _name: FakeProfiler()
    )
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_OUTPUT", str(output_path))
    monkeypatch.setenv("MIDLAYER_910B_MSPROF_STEP_TIMESTAMP", "1")

    vllm_scheduler_trace._install_ascend_worker_wrappers()

    formal = SimpleNamespace(_midlayer_scheduler_trace_step_id="vllm:17")
    following = SimpleNamespace(_midlayer_scheduler_trace_step_id="vllm:18")
    assert FakeWorker(1).execute_model(formal) == "model-output"
    assert FakeWorker(0).execute_model(formal) == "model-output"
    assert FakeWorker(1).execute_model(following) == "model-output"
    warmup = SimpleNamespace()
    assert FakeWorker(0).execute_model(warmup) == "model-output"

    marker_paths = tuple(tmp_path.glob(
        "scheduler_steps.raw.jsonl.ascend-step-marker.*.jsonl"
    ))
    assert len(marker_paths) == 2
    markers = {}
    for marker_path in marker_paths:
        rows = [json.loads(line) for line in marker_path.read_text().splitlines()]
        markers[rows[0]["worker_rank"]] = (marker_path, rows)
    assert set(markers) == {0, 1}
    for rank, (marker_path, rows) in markers.items():
        assert marker_path.name.endswith(
            f".ascend-step-marker.rank-{rank}.pid-{os.getpid()}.jsonl"
        )
        assert all(row["schema_version"] == 2 for row in rows)
        assert all(row["source"] == "ascend_aclprof_step_timestamp" for row in rows)
        assert all(row["process_pid"] == os.getpid() for row in rows)
        assert all(row["worker_rank"] == rank for row in rows)
    assert [row["marker_ordinal"] for row in markers[0][1]] == [0]
    assert [row["trace_step_id"] for row in markers[0][1]] == ["vllm:17"]
    assert [row["marker_ordinal"] for row in markers[1][1]] == [0, 1]
    assert [row["trace_step_id"] for row in markers[1][1]] == [
        "vllm:17", "vllm:18",
    ]
    assert [entry[0] for entry in calls] == [
        "create", "timestamp", "execute", "timestamp", "destroy",
        "create", "timestamp", "execute", "timestamp", "destroy",
        "create", "timestamp", "execute", "timestamp", "destroy", "execute",
    ]


def test_ascend_mstx_marks_formal_step_ends_on_current_stream(
    monkeypatch, tmp_path
):
    calls = []

    class FakeWorker:
        def __init__(self, rank):
            self.rank = rank

        def execute_model(self, scheduler_output):
            calls.append(("execute", scheduler_output))
            return "model-output"

    ascend_module = ModuleType("vllm_ascend")
    worker_package = ModuleType("vllm_ascend.worker")
    worker_module = ModuleType("vllm_ascend.worker.worker")
    worker_module.NPUWorker = FakeWorker
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker", worker_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", worker_module)
    stream = SimpleNamespace(npu_stream=19)
    class FakeEvent:
        def __init__(self):
            self.npu_event = 23

        def record(self, recorded_stream):
            calls.append(("event-record", recorded_stream))

    mstx = SimpleNamespace(mark=lambda *args: calls.append(("mark", args)))
    torch_npu_module = ModuleType("torch_npu")
    torch_npu_module.npu = SimpleNamespace(
        current_stream=lambda: stream,
        mstx=mstx,
        Event=FakeEvent,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_module)
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_OUTPUT", str(output_path))
    monkeypatch.setenv("MIDLAYER_910B_MSPROFTX_STEP_MARK", "1")
    monkeypatch.setenv("MIDLAYER_910B_THREAD_PROFILE_STEP_ID", "vllm:17")

    vllm_scheduler_trace._install_ascend_worker_wrappers()

    formal = SimpleNamespace(_midlayer_scheduler_trace_step_id="vllm:17")
    assert FakeWorker(1).execute_model(formal) == "model-output"
    assert FakeWorker(0).execute_model(SimpleNamespace()) == "model-output"

    marker_path = next(tmp_path.glob(
        "scheduler_steps.raw.jsonl.ascend-step-marker.*.jsonl"
    ))
    row = json.loads(marker_path.read_text())
    assert row["schema_version"] == 3
    assert row["source"] == "ascend_mstx_device_end_mark"
    assert row["process_pid"] == os.getpid()
    assert row["worker_rank"] == 1
    assert row["marker_ordinal"] == 0
    assert row["trace_step_id"] == "vllm:17"
    assert row["stream_handle"] == 19
    assert row["mark_issue_start_ns"] <= row["mark_issue_end_ns"]
    assert calls[0] == ("execute", formal)
    assert calls[1] == (
        "mark",
        ("midlayer-step-end:vllm:17", stream, "midlayer-selected-step"),
    )
    assert calls[2] == ("event-record", stream)
    event_path = next(tmp_path.glob(
        "scheduler_steps.raw.jsonl.ascend-event-boundary.*.jsonl"
    ))
    event_row = json.loads(event_path.read_text())
    assert event_row["schema_version"] == 2
    assert event_row["source"] == "ascend_aclrt_event_record_boundary"
    assert event_row["process_pid"] == os.getpid()
    assert event_row["worker_rank"] == 1
    assert event_row["marker_ordinal"] == 0
    assert event_row["trace_step_id"] == "vllm:17"
    assert event_row["stream_handle"] == 19
    assert event_row["event_handle"] == 23
    assert event_row["event_issue_start_ns"] <= event_row["event_issue_end_ns"]
    assert event_row["event_issue_start_wall_ns"] <= event_row["event_issue_end_wall_ns"]
    assert event_row["worker_execution_clock"] == "time.perf_counter_ns"
    assert event_row["worker_thread_cpu_clock"] == "time.thread_time_ns"
    assert event_row["worker_execution_start_ns"] <= event_row["worker_execution_end_ns"]
    assert event_row["worker_thread_cpu_start_ns"] <= event_row["worker_thread_cpu_end_ns"]
    assert (
        event_row["worker_thread_cpu_end_ns"]
        - event_row["worker_thread_cpu_start_ns"]
        <= event_row["worker_execution_end_ns"]
        - event_row["worker_execution_start_ns"]
    )
    profile_path = next(tmp_path.glob(
        "scheduler_steps.raw.jsonl.thread-cpu-profile.*.pstats"
    ))
    assert pstats.Stats(str(profile_path)).total_calls > 0


def test_ascend_mstx_samples_selected_worker_python_stack(monkeypatch, tmp_path):
    class FakeWorker:
        rank = 1

        def execute_model(self, scheduler_output):
            time.sleep(0.2)
            return "model-output"

    ascend_module = ModuleType("vllm_ascend")
    worker_package = ModuleType("vllm_ascend.worker")
    worker_module = ModuleType("vllm_ascend.worker.worker")
    worker_module.NPUWorker = FakeWorker
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker", worker_package)
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", worker_module)
    stream = SimpleNamespace(npu_stream=19)

    class FakeEvent:
        npu_event = 23

        def record(self, recorded_stream):
            assert recorded_stream is stream

    torch_npu_module = ModuleType("torch_npu")
    torch_npu_module.npu = SimpleNamespace(
        current_stream=lambda: stream,
        mstx=SimpleNamespace(mark=lambda *_args: None),
        Event=FakeEvent,
    )
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_module)
    output_path = tmp_path / "scheduler_steps.raw.jsonl"
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_OUTPUT", str(output_path))
    monkeypatch.setenv("MIDLAYER_910B_MSPROFTX_STEP_MARK", "1")
    monkeypatch.setenv("MIDLAYER_910B_STACK_SAMPLE_STEP_ID", "vllm:17")

    vllm_scheduler_trace._install_ascend_worker_wrappers()

    formal = SimpleNamespace(_midlayer_scheduler_trace_step_id="vllm:17")
    assert FakeWorker().execute_model(formal) == "model-output"
    profile_path = next(tmp_path.glob(
        "scheduler_steps.raw.jsonl.stack-sample-cpu-profile.*.json"
    ))
    profile = json.loads(profile_path.read_text())
    assert profile["schema_version"] == 1
    assert profile["source"] == "python_current_frames_statistical_sampling"
    assert profile["worker_rank"] == 1
    assert profile["trace_step_id"] == "vllm:17"
    assert profile["worker_call_start_wall_ns"] < profile["worker_call_end_wall_ns"]
    assert profile["samples"]
    assert all(sample["stack"] for sample in profile["samples"])


def test_c500_graph_launch_context_wraps_only_formal_scheduler_steps(
    monkeypatch, tmp_path
):
    calls = []
    context_base = tmp_path / "graph-context"
    interposer_path = tmp_path / "launch.so"
    interposer_path.write_bytes(b"test interposer")

    class FakeFunction:
        argtypes = None
        restype = object()

        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            calls.append((self.name, args))

    fake_interposer = SimpleNamespace(
        midlayer_c500_set_step_context=FakeFunction("set"),
        midlayer_c500_clear_step_context=FakeFunction("clear"),
    )

    class FakeWorker:
        rank = 1

        def execute_model(self, scheduler_output):
            calls.append(("execute", (scheduler_output,)))
            return "model-output"

    vllm_module = ModuleType("vllm")
    v1_module = ModuleType("vllm.v1")
    worker_package = ModuleType("vllm.v1.worker")
    worker_module = ModuleType("vllm.v1.worker.gpu_worker")
    worker_module.Worker = FakeWorker
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", worker_package)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", worker_module)
    monkeypatch.setattr(vllm_scheduler_trace.ctypes, "CDLL", lambda _path: fake_interposer)
    monkeypatch.setenv("MIDLAYER_C500_GRAPH_LAUNCH_CONTEXT", str(context_base))
    monkeypatch.setenv(
        "MIDLAYER_C500_GRAPH_LAUNCH_LOG", str(tmp_path / "launch-log")
    )
    monkeypatch.setenv(
        "MIDLAYER_C500_KERNEL_LAUNCH_LOG", str(tmp_path / "kernel-launch-log")
    )
    monkeypatch.setenv("MIDLAYER_C500_GRAPH_LAUNCH_INTERPOSER", str(interposer_path))

    vllm_scheduler_trace._install_c500_worker_wrappers()

    formal = SimpleNamespace(_midlayer_scheduler_trace_step_id="vllm:17")
    assert FakeWorker().execute_model(formal) == "model-output"
    warmup = SimpleNamespace()
    assert FakeWorker().execute_model(warmup) == "model-output"
    assert calls == [
        ("set", (b"vllm:17", 1, 0)),
        ("execute", (formal,)),
        ("clear", ()),
        ("execute", (warmup,)),
    ]


def test_trace_spawn_bootstrap_requires_executable_contract(monkeypatch, tmp_path):
    wrapper = tmp_path / "spawn.sh"
    wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
    wrapper.chmod(0o755)
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    monkeypatch.setenv(
        "MIDLAYER_SCHEDULER_TRACE_SPAWN_WRAPPER", str(wrapper))
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_SPAWN_PYTHON", str(python))
    monkeypatch.setenv(
        "MIDLAYER_SCHEDULER_TRACE_SPAWN_PYTHONPATH", "/trace-bootstrap:/repo")
    configured = []
    monkeypatch.setattr(
        vllm_scheduler_trace.multiprocessing,
        "set_executable",
        configured.append,
    )

    vllm_scheduler_trace._configure_spawn_bootstrap()

    assert configured == [str(wrapper.resolve())]


def test_engine_core_spawn_target_bootstraps_trace_in_child(monkeypatch):
    calls = []

    class FakeEngineCoreProc:
        @staticmethod
        def run_engine_core(*args, **kwargs):
            calls.append((args, kwargs))
            return "engine-result"

    engine_module = ModuleType("vllm.v1.engine.core")
    engine_module.EngineCoreProc = FakeEngineCoreProc
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.core", engine_module)
    monkeypatch.setattr(
        vllm_scheduler_trace, "_ENGINE_CORE_ORIGINAL_TARGET", None
    )
    monkeypatch.setattr(
        vllm_scheduler_trace, "install_from_environment", lambda: True
    )

    vllm_scheduler_trace._install_engine_core_spawn_target()

    assert FakeEngineCoreProc.run_engine_core is (
        vllm_scheduler_trace._run_engine_core_with_scheduler_trace
    )
    assert FakeEngineCoreProc.run_engine_core("arg", option="value") == (
        "engine-result"
    )
    assert calls == [(('arg',), {"option": "value"})]
