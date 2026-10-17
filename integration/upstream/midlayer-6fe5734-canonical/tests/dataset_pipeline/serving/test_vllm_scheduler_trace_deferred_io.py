"""Contract tests: H200 vLLM 0.30 trace I/O leaves the Scheduler thread.

The golden files under ``fixtures/nvidia_h200_vllm030_scheduler_trace`` were written
by the synchronous tracer (commit d3d503e89, before deferred I/O) from the
synthetic step sequence in ``drive_synthetic_trace``.  The deferred tracer
must publish byte-identical content.
"""

import itertools
import json
from pathlib import Path
import sys
import threading
from types import ModuleType
from types import SimpleNamespace

import pytest

from dataset_pipeline.serving_dataset import vllm_scheduler_trace


GOLDEN = Path(__file__).parent / "fixtures" / "nvidia_h200_vllm030_scheduler_trace"
TRACE_NAME = "scheduler_steps.raw.jsonl"
SIDECARS = (".runtime.json", ".clock.json", ".warmup.json")
WARMUP_ID = "cmpl-warmup-batch-00-0-8a3b088c"
R0 = "cmpl-request-000000000000-0-0a1b2c3d"
R1 = "cmpl-request-000000000001-0-4e5f6a7b"


class FakeVllmConfig:
    max_concurrent_batches = 2
    parallel_config = SimpleNamespace(pipeline_parallel_size=1)
    compilation_config = SimpleNamespace(max_cudagraph_capture_size=512)


class FakeExecutor:
    vllm_config = FakeVllmConfig()


def _engine_outputs(finished_ids):
    return {0: SimpleNamespace(
        outputs=[
            SimpleNamespace(request_id=request_id, finished=True)
            for request_id in finished_ids
        ],
        finished_requests=None,
    )}


def _model_output(tokens, padded, mode):
    return SimpleNamespace(cudagraph_stats=SimpleNamespace(
        num_unpadded_tokens=tokens,
        num_padded_tokens=padded,
        num_paddings=padded - tokens,
        runtime_mode=mode,
    ))


def drive_synthetic_trace(module, output_path, set_attr):
    """Drive warmup plus an async depth-2 formal run through the hooks.

    ``set_attr(target, name, value)`` patches for the duration of the run.
    """

    vllm_module = ModuleType("vllm")
    vllm_module.__version__ = "0.30.0"
    set_attr(sys.modules, "vllm", vllm_module)
    set_attr(module, "_installed_vllm_release", lambda: "0.30.0")
    clock = itertools.count(1_000_000, 1_000)
    set_attr(module.time, "perf_counter_ns", lambda: next(clock))
    state = module._TraceState(
        output_path,
        2,
        configured_cudagraph_mode="FULL_AND_PIECEWISE",
        max_in_flight=2,
        hardware_id="nvidia_h200",
        worker_rank_count=1,
    )
    executor = FakeExecutor()
    requests = {}
    scheduler = SimpleNamespace(
        requests=requests, vllm_config=FakeVllmConfig())

    def schedule(tokens, computed_after):
        for request_id, value in computed_after.items():
            requests[request_id].num_computed_tokens = value
        output = SimpleNamespace(num_scheduled_tokens=dict(tokens))
        iteration, started = state.before_schedule()
        state.after_schedule(scheduler, output, iteration, started)
        state.executor_started(
            executor, output, module.time.perf_counter_ns())
        state.executor_finished(output, module.time.perf_counter_ns())
        return output

    def update(output, finished, model_output):
        started = module.time.perf_counter_ns()
        state.executor_completion_observed_by_scheduler(output, started)
        state.after_update(
            output,
            started,
            module.time.perf_counter_ns(),
            _engine_outputs(finished),
            model_output,
        )

    requests[WARMUP_ID] = SimpleNamespace(
        num_computed_tokens=0, num_prompt_tokens=4)
    warm = schedule({WARMUP_ID: 4}, {WARMUP_ID: 4})
    update(warm, [WARMUP_ID], _model_output(4, 4, "CUDAGraphMode.FULL"))
    del requests[WARMUP_ID]

    requests[R0] = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=600)
    requests[R1] = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=3)
    s0 = schedule({R0: 600}, {R0: 600})
    s1 = schedule({R0: 1, R1: 3}, {R0: 601, R1: 3})
    update(s0, [], _model_output(600, 600, "CUDAGraphMode.NONE"))
    s2 = schedule({R0: 1, R1: 1}, {R0: 602, R1: 4})
    update(s1, [R1], _model_output(4, 4, "CUDAGraphMode.FULL"))
    s3 = schedule({R0: 1}, {R0: 603})
    update(s2, [], _model_output(2, 2, "CUDAGraphMode.FULL"))
    update(s3, [R0], _model_output(1, 1, "CUDAGraphMode.FULL"))
    return state


def _normalized(path, suffix):
    text = Path(f"{path}{suffix}").read_text()
    if suffix == ".runtime.json":
        value = json.loads(text)
        _normalize_evidence(value["scheduler_pipeline_depth_evidence"])
        return value
    if suffix == ".clock.json":
        value = json.loads(text)
        value["host_boot_id"] = "<boot_id>"
        return value
    if suffix == ".bootstrap.jsonl":
        return [
            json.loads(line)["event"] for line in text.splitlines()
        ]
    return [
        _normalize_row(json.loads(line)) for line in text.splitlines()
    ]


def _normalize_evidence(evidence):
    # The fake executor's module name and file depend on how the test module
    # was imported; every other evidence field is compared exactly.
    evidence["executor_class"] = evidence["executor_class"].rsplit(".", 1)[-1]
    evidence["source_identity"] = {
        key: "<import-location>" for key in evidence["source_identity"]
    }


def _normalize_row(row):
    if "scheduler_pipeline_depth_evidence" in row:
        _normalize_evidence(row["scheduler_pipeline_depth_evidence"])
    return row


def _run(monkeypatch, tmp_path, *, force_synchronous=False):
    marker = tmp_path / f"{TRACE_NAME}.bootstrap.jsonl"
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_BOOTSTRAP_MARKER", str(marker))
    output_path = tmp_path / TRACE_NAME
    if force_synchronous:
        monkeypatch.setattr(
            vllm_scheduler_trace, "_uses_deferred_trace_io", lambda _hw: False)
    state = drive_synthetic_trace(
        vllm_scheduler_trace, output_path, _patcher(monkeypatch))
    return state, output_path


def _patcher(monkeypatch):
    def set_attr(target, name, value):
        if isinstance(target, dict):
            monkeypatch.setitem(target, name, value)
        else:
            monkeypatch.setattr(target, name, value)
    return set_attr


@pytest.mark.parametrize("force_synchronous", (False, True))
def test_h200_030_trace_content_matches_synchronous_golden(
    monkeypatch, tmp_path, force_synchronous
):
    state, output_path = _run(
        monkeypatch, tmp_path, force_synchronous=force_synchronous)
    assert state.deferred_io is (not force_synchronous)
    state.wait_for_deferred_io()
    golden = GOLDEN / TRACE_NAME
    assert _normalized(output_path, "") == _normalized(golden, "")
    for suffix in (*SIDECARS, ".bootstrap.jsonl"):
        assert _normalized(output_path, suffix) == _normalized(golden, suffix)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        path.name for path in GOLDEN.iterdir())


def test_h200_030_final_publication_leaves_scheduler_thread(
    monkeypatch, tmp_path
):
    gate = threading.Event()
    real_write = vllm_scheduler_trace._TraceState._write_complete_trace

    def gated_write(state):
        assert threading.current_thread() is not threading.main_thread()
        assert gate.wait(10)
        real_write(state)

    monkeypatch.setattr(
        vllm_scheduler_trace._TraceState, "_write_complete_trace", gated_write)
    state, output_path = _run(monkeypatch, tmp_path)
    # The last update has returned while its publication is still blocked.
    assert not output_path.exists()
    gate.set()
    state.wait_for_deferred_io()
    assert output_path.is_file()
    assert state.written is True
    for suffix in SIDECARS:
        assert Path(f"{output_path}{suffix}").is_file()


def test_h200_030_depth_evidence_is_resolved_before_formal_steps(
    monkeypatch, tmp_path
):
    set_attr = _patcher(monkeypatch)
    vllm_module = ModuleType("vllm")
    vllm_module.__version__ = "0.30.0"
    set_attr(sys.modules, "vllm", vllm_module)
    set_attr(vllm_scheduler_trace, "_installed_vllm_release", lambda: "0.30.0")
    state = vllm_scheduler_trace._TraceState(
        tmp_path / TRACE_NAME,
        1,
        configured_cudagraph_mode="FULL_AND_PIECEWISE",
        max_in_flight=2,
        hardware_id="nvidia_h200",
        worker_rank_count=1,
    )
    warmup_output = SimpleNamespace(num_scheduled_tokens={WARMUP_ID: 1})
    state.executor_started(FakeExecutor(), warmup_output, 1)
    assert state.configured_depth_evidence["configured_depth"] == 2
    assert state.pending == {}


def test_h200_030_marker_write_failure_fails_the_next_update(
    monkeypatch, tmp_path
):
    def failing_append(_path, _value):
        raise OSError("marker filesystem unavailable")

    monkeypatch.setattr(vllm_scheduler_trace, "_append_marker", failing_append)
    with pytest.raises(RuntimeError, match="deferred file I/O failed"):
        _run(monkeypatch, tmp_path)


@pytest.mark.parametrize(
    ("hardware_id", "release", "expected"),
    (
        ("nvidia_h200", "0.30.0", True),
        ("nvidia_h200", "0.11.0", False),
        ("nvidia_a800", "0.11.0", False),
        ("metax_c500", "0.24.0", False),
        ("ascend_910b", "0.18.0", False),
        (None, "0.30.0", False),
    ),
)
def test_only_h200_030_defers_trace_io(monkeypatch, hardware_id, release, expected):
    monkeypatch.setattr(
        vllm_scheduler_trace, "_installed_vllm_release", lambda: release)
    assert vllm_scheduler_trace._uses_deferred_trace_io(hardware_id) is expected


def test_unknown_h200_release_fails_closed(monkeypatch):
    monkeypatch.setattr(
        vllm_scheduler_trace, "_installed_vllm_release", lambda: "0.31.0")
    with pytest.raises(ValueError, match="does not support vLLM"):
        vllm_scheduler_trace._uses_deferred_trace_io("nvidia_h200")
