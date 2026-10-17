import hashlib
import json

import pytest

from evaluation.serving.metrics.diagnostics.selected_step_injection import (
    build_selected_step_injection_replay,
    build_step_rankings,
    materialize_selected_step,
    materialize_selected_step_token_prefixes,
    validate_selected_step_injection_trace,
)
from scripts.evaluation.serving.diagnostics import selected_step_runner


def _fingerprint(step_id, request_idx, *, runtime_mode="FULL"):
    return {
        "schema_version": 1,
        "cell_id": "nvidia_a800/model/tp2_ep1_qps16",
        "hardware_id": "nvidia_a800",
        "model_id": "model",
        "step_id": step_id,
        "request_workload": [{
            "request_idx": request_idx,
            "request_id": f"request-{request_idx:012d}",
            "computed_tokens_before_step": 1,
            "kv_context_tokens_after_step": 2,
            "prefill_tokens": 0,
            "decode_tokens": 1,
            "scheduled_tokens": 1,
        }],
        "token_totals": {
            "prefill_tokens": 0,
            "decode_tokens": 1,
            "scheduled_tokens": 1,
        },
        "parallelism": {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
        },
        "graph_execution": {
            "capture_bucket_tokens": 1 if runtime_mode == "FULL" else None,
            "capture_limit_tokens": 512,
            "configured_mode": "FULL_AND_PIECEWISE",
            "fallback_reason": None if runtime_mode == "FULL" else "above_capture_limit",
            "logical_tokens": 1,
            "model_input_tokens": 1,
            "padding_tokens": 0,
            "rank_ids": [0, 1],
            "runtime_mode": runtime_mode,
            "uniform_decode": True,
        },
        "normalized_launch_program_sha256": "a" * 64,
        "normalized_launch_call_count": 3,
    }


def _replay():
    return {
        "schema_version": 5,
        "cell_id": "nvidia_a800/model/tp2_ep1_qps16",
        "forward_duration_observation_consumed": False,
        "source_sha256": {"candidate_scheduler_trace": "b" * 64},
        "step_comparison": [
            {
                "step_index": 0,
                "step_id": "vllm:0",
                "phase": "prefill",
                "step_fingerprint": _fingerprint("vllm:0", 0),
                "observed_serialized_service_s": 0.020,
                "predicted_device_s": 0.005,
            },
            {
                "step_index": 1,
                "step_id": "vllm:1",
                "phase": "decode",
                "step_fingerprint": _fingerprint("vllm:1", 0),
                "observed_serialized_service_s": 0.010,
                "predicted_device_s": 0.004,
            },
        ],
    }


def _trace_rows():
    common = {
        "schema_version": 3,
        "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
        "runtime_cudagraph_mode": "FULL",
        "capture_bucket_tokens": 1,
        "capture_limit_tokens": 512,
        "num_unpadded_tokens": 1,
        "padded_total_tokens": 1,
        "padding_tokens": 0,
        "graph_fallback_reason": None,
        "scheduled_tokens": 1,
        "prefill_tokens": 0,
        "decode_tokens": 1,
        "scheduled_request_count": 1,
        "scheduled_requests": [{
            "request_idx": 0,
            "scheduled_tokens": 1,
            "prefill_tokens": 0,
            "decode_tokens": 1,
        }],
        "schedule_start_ns": 0,
        "schedule_end_ns": 1_000_000,
        "device_execution_start_ns": 1_000_000,
        "device_execution_end_ns": 4_000_000,
        "scheduler_update_start_ns": 5_000_000,
        "scheduler_update_end_ns": 6_000_000,
        "update_start_ns": 5_000_000,
        "step_start_ns": 0,
        "step_end_ns": 6_000_000,
    }
    first = dict(common, step_id="vllm:0", step_index=0)
    second = json.loads(json.dumps(common))
    second.update({"step_id": "vllm:1", "step_index": 1})
    second.update({
        "schedule_start_ns": 6_000_000,
        "schedule_end_ns": 7_000_000,
        "device_execution_start_ns": 7_000_000,
        "device_execution_end_ns": 9_000_000,
        "scheduler_update_start_ns": 10_000_000,
        "scheduler_update_end_ns": 11_000_000,
        "update_start_ns": 10_000_000,
        "step_start_ns": 6_000_000,
        "step_end_ns": 11_000_000,
    })
    return first, second


def _injection_replay(trace, fingerprint, *, step_id="vllm:1"):
    return {
        "schema_version": 5,
        "cell_id": "nvidia_a800/model/tp2_ep1_qps16",
        "forward_duration_observation_consumed": False,
        "source_sha256": {
            "scheduler_trace": hashlib.sha256(trace.read_bytes()).hexdigest(),
        },
        "step_id_mapping": {
            "canonical_workload_step_id": "vllm:353",
            "injected_trace_step_id": step_id,
        },
        "step_comparison": [{
            "step_id": step_id,
            "step_fingerprint": fingerprint,
        }],
    }


def test_ranking_selects_tpot_phase_not_prefill():
    ranking = build_step_rankings(_replay())

    assert ranking["overall_ranking"][0]["step_id"] == "vllm:0"
    assert ranking["selected_phase"] == "decode"
    assert ranking["selected_step_id"] == "vllm:1"
    assert ranking["phase_rankings"]["decode"][0]["c_step_ms"] == 6.0


def test_materialization_checks_canonical_context(tmp_path):
    trace = tmp_path / "scheduler_steps.jsonl"
    first, second = _trace_rows()
    trace.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")

    manifest = materialize_selected_step(_replay(), trace, "vllm:1")

    assert manifest["expected_fingerprint"]["request_workload"][0][
        "computed_tokens_before_step"
    ] == 1
    assert manifest["canonical_boundaries"]["serialized_service_ms"] == 3.0
    assert manifest["injection_contract"]["full_request_stream_replay_allowed"] is False


def test_materialization_fails_closed_on_graph_mismatch(tmp_path):
    trace = tmp_path / "scheduler_steps.jsonl"
    first, second = _trace_rows()
    second["runtime_cudagraph_mode"] = "NONE"
    second["capture_bucket_tokens"] = None
    second["graph_fallback_reason"] = "above_capture_limit"
    trace.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")

    with pytest.raises(
        ValueError, match="INVALID EXPERIMENT — fingerprint.graph_execution"
    ):
        materialize_selected_step(_replay(), trace, "vllm:1")


def test_materialization_accepts_rank_graph_observations(tmp_path):
    trace = tmp_path / "scheduler_steps.jsonl"
    first, second = _trace_rows()
    for row in (first, second):
        row["schema_version"] = 4
        row["rank_observations"] = [{
            "rank": rank,
            "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
            "runtime_cudagraph_mode": "FULL",
            "selected_graph_key": {"num_tokens": 1, "uniform_decode": True},
            "graph_fallback_reason": None,
            "num_unpadded_tokens": 1,
            "padded_total_tokens": 1,
            "padding_tokens": 0,
            "worker_execution_start_ns": 2_000_000,
            "worker_execution_end_ns": 3_000_000,
        } for rank in (0, 1)]
    trace.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")

    manifest = materialize_selected_step(_replay(), trace, "vllm:1")

    assert manifest["expected_fingerprint"]["graph_execution"]["rank_ids"] == [0, 1]
    assert manifest["canonical_boundaries"]["worker_execute_envelopes"] == [
        {"rank": 0, "worker_execute_ms": 1.0},
        {"rank": 1, "worker_execute_ms": 1.0},
    ]


def test_materialization_accepts_flat_schema_four_graph_observation(tmp_path):
    trace = tmp_path / "scheduler_steps.jsonl"
    first, second = _trace_rows()
    first["schema_version"] = 4
    second["schema_version"] = 4
    trace.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")

    manifest = materialize_selected_step(_replay(), trace, "vllm:1")

    assert manifest["expected_fingerprint"]["graph_execution"] == (
        _fingerprint("vllm:1", 0)["graph_execution"]
    )
    assert manifest["canonical_boundaries"]["worker_execute_envelopes"] == []


def test_materialization_accepts_request_first_seen_in_selected_step(tmp_path):
    trace = tmp_path / "scheduler_steps.jsonl"
    first, second = _trace_rows()
    second["scheduled_requests"][0]["request_idx"] = 1
    trace.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    replay = _replay()
    replay["step_comparison"][1]["step_fingerprint"] = _fingerprint(
        "vllm:1", 1
    )
    replay["step_comparison"][1]["step_fingerprint"]["request_workload"][0][
        "computed_tokens_before_step"
    ] = 0
    replay["step_comparison"][1]["step_fingerprint"]["request_workload"][0][
        "kv_context_tokens_after_step"
    ] = 1

    manifest = materialize_selected_step(replay, trace, "vllm:1")

    assert manifest["expected_fingerprint"]["request_workload"][0][
        "computed_tokens_before_step"
    ] == 0


def test_materialize_selected_step_token_prefixes_uses_real_token_values():
    selected = {
        "kind": "selected_step_injection_input",
        "cell_id": "nvidia_a800/model/tp2_ep1_qps16",
        "step_id": "vllm:1",
        "expected_fingerprint": _fingerprint("vllm:1", 7),
    }
    selected["expected_fingerprint"]["request_workload"][0][
        "computed_tokens_before_step"
    ] = 3
    selected["expected_fingerprint"]["request_workload"][0][
        "kv_context_tokens_after_step"
    ] = 4

    content = materialize_selected_step_token_prefixes(
        selected,
        {7: [11, 12]},
        {7: [13, 14]},
        {"method": "test"},
    )

    assert content["total_computed_tokens_before_step"] == 3
    assert content["requests"][0]["prefix_token_ids"] == [11, 12, 13]
    assert content["content_source"] == {"method": "test"}


def test_materialize_selected_step_token_prefixes_fails_on_short_output():
    selected = {
        "kind": "selected_step_injection_input",
        "cell_id": "nvidia_a800/model/tp2_ep1_qps16",
        "step_id": "vllm:1",
        "expected_fingerprint": _fingerprint("vllm:1", 7),
    }
    selected["expected_fingerprint"]["request_workload"][0][
        "computed_tokens_before_step"
    ] = 4

    with pytest.raises(ValueError, match="generated-token prefix is incomplete"):
        materialize_selected_step_token_prefixes(
            selected,
            {7: [11, 12]},
            {7: [13]},
            {"method": "test"},
        )


def test_materialize_selected_step_token_prefixes_supports_fresh_mixed_prefill():
    selected = {
        "kind": "selected_step_injection_input",
        "cell_id": "ascend_910b/model/tp2_ep2_qps16",
        "step_id": "vllm:105",
        "expected_fingerprint": _fingerprint(
            "vllm:105", 7, runtime_mode="NONE"
        ),
    }
    decode = selected["expected_fingerprint"]["request_workload"][0]
    decode["computed_tokens_before_step"] = 2
    decode["kv_context_tokens_after_step"] = 3
    prefill = {
        "request_idx": 8,
        "request_id": "request-000000000008",
        "computed_tokens_before_step": 0,
        "kv_context_tokens_after_step": 3,
        "prefill_tokens": 3,
        "decode_tokens": 0,
        "scheduled_tokens": 3,
    }
    selected["expected_fingerprint"]["request_workload"].append(prefill)

    content = materialize_selected_step_token_prefixes(
        selected,
        {7: [11, 12], 8: [21, 22, 23]},
        {7: []},
        {"method": "test"},
    )

    assert content["schema_version"] == 2
    assert content["requests"][0]["role"] == "decode_state_prefix"
    assert content["requests"][0]["prefix_token_ids"] == [11, 12]
    assert content["requests"][1]["role"] == "fresh_prefill_prompt"
    assert content["requests"][1]["prefix_token_ids"] == []
    assert content["requests"][1]["injection_prompt_token_ids"] == [21, 22, 23]


def test_selected_step_runner_drains_reconstruction_pipeline():
    class Scheduler:
        @staticmethod
        def has_requests():
            return False

    class Core:
        def __init__(self):
            self.batch_queue = [object(), object()]
            self.scheduler = Scheduler()

    class Client:
        def __init__(self):
            self.engine_core = Core()

        def get_output(self):
            self.engine_core.batch_queue.pop()

    class Engine:
        def __init__(self):
            self.engine_core = Client()

    class LLM:
        def __init__(self):
            self.llm_engine = Engine()

    llm = LLM()

    assert selected_step_runner._drain_inprocess_async_pipeline(llm) == 2
    assert llm.llm_engine.engine_core.engine_core.batch_queue == []


def test_selected_step_runner_warms_c500_full_decode_bucket():
    class Scheduler:
        def __init__(self):
            self.requests = {}

        def has_requests(self):
            return bool(self.requests)

    class Core:
        def __init__(self):
            self.batch_queue = []
            self.scheduler = Scheduler()

    class Client:
        def __init__(self):
            self.engine_core = Core()

        def get_output(self):
            raise AssertionError("warmup must not leave async results pending")

    class Prompt:
        def __init__(self, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    class Output:
        def __init__(self, request_id):
            self.request_id = request_id
            self.finished = True

    class Engine:
        def __init__(self):
            self.engine_core = Client()

        def add_request(self, request_id, prompt, params):
            del params
            self.engine_core.engine_core.scheduler.requests[request_id] = prompt

        def has_unfinished_requests(self):
            return bool(self.engine_core.engine_core.scheduler.requests)

        def step(self):
            requests = self.engine_core.engine_core.scheduler.requests
            outputs = [Output(request_id) for request_id in requests]
            requests.clear()
            return outputs

    class LLM:
        def __init__(self):
            self.llm_engine = Engine()
            self.prefix_cache_resets = 0

        def reset_prefix_cache(self):
            self.prefix_cache_resets += 1

    llm = LLM()

    assert selected_step_runner._warm_c500_full_decode_graph(
        llm, 40, object(), Prompt
    ) == 0
    assert llm.prefix_cache_resets == 1
    assert llm.llm_engine.engine_core.engine_core.scheduler.requests == {}


def test_selected_step_runner_requires_stable_request_ids(monkeypatch):
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_REQUEST_COUNT", "1")
    monkeypatch.setenv(
        "MIDLAYER_SCHEDULER_TRACE_CONFIGURED_CUDAGRAPH_MODE",
        "FULL_AND_PIECEWISE",
    )
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_MAX_IN_FLIGHT", "2")
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_OUTPUT", "/tmp/trace.jsonl")
    monkeypatch.setenv("MIDLAYER_SCHEDULER_TRACE_SOURCE_IDENTITY", "test")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DISABLE_REQUEST_ID_RANDOMIZATION", "0")

    with pytest.raises(
        RuntimeError,
        match="VLLM_DISABLE_REQUEST_ID_RANDOMIZATION='0', expected='1'",
    ):
        selected_step_runner._require_environment(
            1, "FULL_AND_PIECEWISE", "nvidia_a800", 2
        )


def test_selected_step_runner_accepts_ascend_mstx_boundary(monkeypatch, tmp_path):
    expected = {
        "MIDLAYER_SCHEDULER_TRACE_REQUEST_COUNT": "27",
        "MIDLAYER_SCHEDULER_TRACE_CONFIGURED_CUDAGRAPH_MODE": "FULL",
        "MIDLAYER_SCHEDULER_TRACE_MAX_IN_FLIGHT": "2",
        "MIDLAYER_SCHEDULER_TRACE_OUTPUT": str(tmp_path / "trace.jsonl"),
        "MIDLAYER_SCHEDULER_TRACE_SOURCE_IDENTITY": "test",
        "MIDLAYER_SCHEDULER_TRACE_HARDWARE_ID": "ascend_910b",
        "MIDLAYER_SCHEDULER_TRACE_WORKER_RANK_COUNT": "2",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1",
        "MIDLAYER_910B_MSPROFTX_STEP_MARK": "1",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    selected_step_runner._require_environment(27, "FULL", "ascend_910b", 2)


@pytest.mark.parametrize("hardware_id", ["ascend_910b", "metax_c500"])
def test_selected_step_runner_accepts_supported_mixed_eager_targets(hardware_id):
    selected_step_runner._validate_target_graph_execution(
        hardware_id,
        "NONE",
        "above_capture_limit",
        mixed_target=True,
    )


def test_selected_step_runner_rejects_mixed_target_without_capture_fallback():
    with pytest.raises(ValueError, match="above-capture-limit eager target"):
        selected_step_runner._validate_target_graph_execution(
            "metax_c500", "FULL", None, mixed_target=True
        )


@pytest.mark.parametrize("hardware_id", ["ascend_910b", "metax_c500"])
def test_selected_step_runner_accepts_supported_standalone_prefill_targets(hardware_id):
    selected_step_runner._validate_target_graph_execution(
        hardware_id,
        "NONE",
        "above_capture_limit",
        mixed_target=False,
        standalone_prefill_target=True,
    )


def test_selected_step_runner_rejects_standalone_prefill_without_capture_fallback():
    with pytest.raises(ValueError, match="standalone prefill selected-step runner"):
        selected_step_runner._validate_target_graph_execution(
            "metax_c500",
            "FULL",
            None,
            mixed_target=False,
            standalone_prefill_target=True,
        )


def test_selected_step_runner_separates_dataset_and_runtime_seeds():
    assert selected_step_runner._seed_inputs(
        {"seed": None}, {"seed": 42}
    ) == (42, {})
    assert selected_step_runner._seed_inputs(
        {"seed": 7}, {"seed": 42}
    ) == (42, {"seed": 7})


def test_selected_step_runner_reserves_prefill_token_before_graph_anchor():
    assert selected_step_runner._target_max_tokens(0) == 2
    assert selected_step_runner._target_max_tokens(1) == 3


def test_selected_step_runner_warms_standalone_prefill_outside_formal_step():
    class Output:
        request_id = "warmup-c500-standalone-prefill-0"
        finished = True

    class Scheduler:
        def has_requests(self):
            return False

    class Core:
        batch_queue = []
        scheduler = Scheduler()

    class Engine:
        def __init__(self):
            self.engine_core = type("Client", (), {"engine_core": Core()})()
            self.unfinished = True
            self.requests = []

        def add_request(self, request_id, prompt, params):
            self.requests.append((request_id, prompt, params))

        def has_unfinished_requests(self):
            return self.unfinished

        def step(self):
            self.unfinished = False
            return [Output()]

    class LLM:
        def __init__(self):
            self.llm_engine = Engine()
            self.prefix_cache_resets = 0

        def reset_prefix_cache(self):
            self.prefix_cache_resets += 1

    class Prompt:
        def __init__(self, *, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    llm = LLM()
    assert selected_step_runner._warm_c500_standalone_prefill(
        llm, 729, object(), Prompt, "warmup-c500-standalone-prefill-0"
    ) == 0
    request_id, prompt, _ = llm.llm_engine.requests[0]
    assert request_id == "warmup-c500-standalone-prefill-0"
    assert prompt.prompt_token_ids == [0] * 729
    assert llm.prefix_cache_resets == 1


@pytest.mark.parametrize("value", [False, -1, 1.5])
def test_selected_step_runner_rejects_invalid_graph_anchor_count(value):
    with pytest.raises((TypeError, ValueError)):
        selected_step_runner._target_max_tokens(value)


@pytest.mark.parametrize(
    ("runtime_seed", "dataset_seed", "message"),
    [
        (None, None, "canonical workload seed must be an integer"),
        (False, 42, "runtime engine seed must be null or an integer"),
    ],
)
def test_selected_step_runner_rejects_invalid_seed_contract(
    runtime_seed, dataset_seed, message
):
    with pytest.raises(ValueError, match=message):
        selected_step_runner._seed_inputs(
            {"seed": runtime_seed}, {"seed": dataset_seed}
        )


def test_selected_step_runner_normalizes_explicit_request_status_forms():
    class Status:
        name = "RUNNING"

    assert selected_step_runner._request_status_name("WAITING") == "WAITING"
    assert selected_step_runner._request_status_name(Status()) == "RUNNING"


@pytest.mark.parametrize("status", [object(), "", type("Status", (), {"name": 1})()])
def test_selected_step_runner_rejects_invalid_request_status(status):
    with pytest.raises(TypeError, match="selected-step request status"):
        selected_step_runner._request_status_name(status)


def test_selected_step_runner_materializes_decode_then_admits_prefill():
    class Status:
        def __init__(self, name):
            self.name = name

    class Request:
        def __init__(self, prompt_tokens):
            self.prompt_tokens = prompt_tokens
            self.num_computed_tokens = 0
            self.status = Status("WAITING")

    class Scheduler:
        def __init__(self):
            self.requests = {}

    class Core:
        def __init__(self):
            self.batch_queue = []
            self.batch_queue_size = 2
            self.scheduler = Scheduler()
            self.paused = False

        def pause_scheduler(self, mode, clear_cache):
            assert mode == "keep"
            assert clear_cache is False
            self.paused = True

        def resume_scheduler(self):
            self.paused = False

        def is_scheduler_paused(self):
            return self.paused

    class Prompt:
        def __init__(self, prompt_token_ids):
            self.prompt_token_ids = prompt_token_ids

    class Client:
        def __init__(self):
            self.engine_core = Core()

    class Engine:
        def __init__(self):
            self.engine_core = Client()

        def add_request(self, request_id, prompt, params):
            del params
            self.engine_core.engine_core.scheduler.requests[request_id] = Request(
                prompt.prompt_token_ids
            )

        def step(self):
            core = self.engine_core.engine_core
            assert core.batch_queue_size == 1
            for request in core.scheduler.requests.values():
                request.num_computed_tokens = len(request.prompt_tokens)
                request.status = Status("RUNNING")
            return []

    class LLM:
        def __init__(self):
            self.llm_engine = Engine()

    llm = LLM()
    decode = [{
        "request_id": "request-000000000000",
        "computed_tokens_before_step": 3,
        "injection_prompt_token_ids": [1, 2, 3],
    }]
    prefill = [{
        "request_id": "request-000000000001",
        "computed_tokens_before_step": 0,
        "injection_prompt_token_ids": [4, 5],
    }]

    selected_step_runner._prepare_mixed_decode_state(
        llm, decode, object(), Prompt
    )
    core = llm.llm_engine.engine_core.engine_core
    assert core.paused is True
    assert core.scheduler.requests[decode[0]["request_id"]].num_computed_tokens == 3

    selected_step_runner._admit_mixed_prefill_requests(
        llm, prefill, object(), Prompt
    )
    assert core.paused is False
    assert core.scheduler.requests[prefill[0]["request_id"]].status.name == "WAITING"


def _injection_inputs(tmp_path):
    selected = {
        "kind": "selected_step_injection_input",
        "cell_id": "nvidia_a800/model/tp2_ep1_qps16",
        "step_id": "vllm:353",
        "expected_fingerprint": _fingerprint("vllm:353", 0),
    }
    content = materialize_selected_step_token_prefixes(
        selected,
        {0: [11]},
        {0: [12]},
        {
            "method": "test",
            "historical_output_token_ids_observed": False,
        },
    )
    first, second = _trace_rows()
    for row in (first, second):
        row["schema_version"] = 4
        row["rank_observations"] = [{
            "rank": rank,
            "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
            "runtime_cudagraph_mode": "FULL",
            "selected_graph_key": {"num_tokens": 1, "uniform_decode": True},
            "graph_fallback_reason": None,
            "num_unpadded_tokens": 1,
            "padded_total_tokens": 1,
            "padding_tokens": 0,
            "worker_execution_start_ns": 2_000_000,
            "worker_execution_end_ns": 3_000_000,
        } for rank in (0, 1)]
    trace = tmp_path / "injection.jsonl"
    trace.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    replay = _injection_replay(trace, _fingerprint("vllm:1", 0))
    return selected, content, trace, replay


def test_build_selected_step_injection_replay_uses_actual_raw_step(
    tmp_path, monkeypatch
):
    selected = {
        "kind": "selected_step_injection_input",
        "cell_id": "nvidia_a800/model/tp2_ep1_qps16",
        "step_id": "vllm:353",
        "expected_fingerprint": _fingerprint("vllm:353", 0),
    }
    content = materialize_selected_step_token_prefixes(
        selected,
        {0: [11]},
        {0: [12]},
        {
            "method": "test",
            "historical_output_token_ids_observed": False,
            "model_config_sha256": "c" * 64,
        },
    )
    first, second = _trace_rows()
    second["finished_request_indexes"] = []
    trace = tmp_path / "injection.jsonl"
    trace.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    config = tmp_path / "config.json"
    config.write_text("{}\n")
    calls = []

    def fake_predict(repository_root, config_path, cell_id, rows, requests,
                     model_config_sha256, *, host_launch_profile,
                     host_runtime_prepare_profile,
                     runtime_prepare_event_builder,
                     include_step_kernel_contributions):
        calls.append({
            "repository_root": repository_root,
            "config_path": config_path,
            "cell_id": cell_id,
            "rows": rows,
            "requests": requests,
            "model_config_sha256": model_config_sha256,
            "host_launch_profile": host_launch_profile,
            "host_runtime_prepare_profile": host_runtime_prepare_profile,
            "runtime_prepare_event_builder": runtime_prepare_event_builder,
            "include_step_kernel_contributions": include_step_kernel_contributions,
        })
        fingerprint = _fingerprint("vllm:353", 0)
        return ([{
            "step_fingerprint": fingerprint,
            "predicted_timeline_ns": 4_000_000,
            "predicted_kernel_contributions": [{"call_id": "test"}],
        }], {}, {"source": "test"})

    monkeypatch.setattr(
        "evaluation.serving.metrics.diagnostics.selected_step_injection."
        "predict_scheduling_trajectory_steps",
        fake_predict,
    )

    result = build_selected_step_injection_replay(
        tmp_path,
        config,
        selected,
        content,
        trace,
    )

    assert len(calls) == 1
    assert calls[0]["rows"][0]["step_id"] == "vllm:353"
    assert calls[0]["rows"][0]["requests"][0]["computed_tokens"] == 1
    assert calls[0]["model_config_sha256"] == "c" * 64
    assert calls[0]["host_launch_profile"] is None
    assert calls[0]["include_step_kernel_contributions"] is True
    assert result["schema_version"] == 5
    assert result["forward_duration_observation_consumed"] is False
    assert result["step_id_mapping"] == {
        "canonical_workload_step_id": "vllm:353",
        "injected_trace_step_id": "vllm:1",
    }
    assert result["step_comparison"][0]["step_id"] == "vllm:1"
    assert result["step_comparison"][0]["predicted_device_s"] == 0.004
    assert result["step_comparison"][0]["predicted_kernel_contributions"] == [
        {"call_id": "test"}
    ]


def test_validate_selected_step_injection_trace_marks_reconstruction(tmp_path):
    selected, content, trace, replay = _injection_inputs(tmp_path)

    result = validate_selected_step_injection_trace(
        selected, content, trace, replay
    )

    assert result["injected_step_id"] == "vllm:1"
    assert result["step_id_mapping"] == {
        "canonical_step_id": "vllm:353",
        "injected_step_id": "vllm:1",
    }
    assert result["canonical_step_fingerprint_sha256"] != result[
        "injected_step_fingerprint_sha256"
    ]
    assert result["mapped_canonical_step_fingerprint_sha256"] == result[
        "injected_step_fingerprint_sha256"
    ]
    assert result["structural_fingerprint_match"] is True
    assert result["comparison_status"] == "independent_reconstruction"
    assert result["historical_output_token_identity_confirmed"] is False


def test_validate_independent_eager_matches_workload_but_not_graph(tmp_path):
    selected, content, trace, _ = _injection_inputs(tmp_path)
    first, second = _trace_rows()
    first["prefill_tokens"] = 1
    first["decode_tokens"] = 0
    first["scheduled_requests"][0]["prefill_tokens"] = 1
    first["scheduled_requests"][0]["decode_tokens"] = 0
    for row in (first, second):
        row["configured_cudagraph_mode"] = "NONE"
        row["runtime_cudagraph_mode"] = "NONE"
        row["capture_bucket_tokens"] = None
        row["capture_limit_tokens"] = None
        row["graph_fallback_reason"] = None
    trace.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    eager_fingerprint = _fingerprint("vllm:1", 0)
    eager_fingerprint["graph_execution"] = {
        "capture_bucket_tokens": None,
        "capture_limit_tokens": None,
        "configured_mode": "NONE",
        "fallback_reason": None,
        "logical_tokens": 1,
        "model_input_tokens": 1,
        "padding_tokens": 0,
        "rank_ids": [0, 1],
        "runtime_mode": "NONE",
        "uniform_decode": True,
    }
    eager_fingerprint["normalized_launch_program_sha256"] = "e" * 64
    eager_fingerprint["normalized_launch_call_count"] = 9
    replay = _injection_replay(trace, eager_fingerprint)

    result = validate_selected_step_injection_trace(
        selected, content, trace, replay, independent_eager=True
    )

    assert result["injected_step_id"] == "vllm:1"
    assert result["structural_fingerprint_match"] is True
    assert result["execution_boundary"] == "independent_eager"
    assert result["comparison_status"] == "independent_eager_reconstruction"


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        (
            "normalized_launch_program_sha256",
            "c" * 64,
            "fingerprint.normalized_launch_program_sha256",
        ),
        (
            "normalized_launch_call_count",
            4,
            "fingerprint.normalized_launch_call_count",
        ),
    ),
)
def test_validate_selected_step_injection_trace_rejects_launch_mismatch(
    tmp_path, field, value, error
):
    selected, content, trace, replay = _injection_inputs(tmp_path)
    replay["step_comparison"][0]["step_fingerprint"][field] = value

    with pytest.raises(ValueError, match=f"INVALID EXPERIMENT — {error}"):
        validate_selected_step_injection_trace(selected, content, trace, replay)


@pytest.mark.parametrize(
    ("mutation", "error"),
    (
        (
            lambda fingerprint: fingerprint.update(hardware_id="metax_c500"),
            "fingerprint.hardware_id",
        ),
        (
            lambda fingerprint: fingerprint.update(model_id="other-model"),
            "fingerprint.model_id",
        ),
        (
            lambda fingerprint: fingerprint["parallelism"].update(
                tensor_parallel_size=4
            ),
            "fingerprint.parallelism.tensor_parallel_size",
        ),
    ),
)
def test_validate_selected_step_injection_trace_rejects_identity_mismatch(
    tmp_path, mutation, error
):
    selected, content, trace, replay = _injection_inputs(tmp_path)
    mutation(replay["step_comparison"][0]["step_fingerprint"])

    with pytest.raises(ValueError, match=f"INVALID EXPERIMENT — {error}"):
        validate_selected_step_injection_trace(selected, content, trace, replay)


def test_validate_selected_step_injection_trace_rejects_source_sha(tmp_path):
    selected, content, trace, replay = _injection_inputs(tmp_path)
    replay["source_sha256"]["scheduler_trace"] = "0" * 64

    with pytest.raises(
        ValueError,
        match=(
            "INVALID EXPERIMENT — "
            "injection_replay.source_sha256.scheduler_trace"
        ),
    ):
        validate_selected_step_injection_trace(selected, content, trace, replay)


def test_validate_selected_step_injection_trace_rejects_step_id_mapping(tmp_path):
    selected, content, trace, replay = _injection_inputs(tmp_path)
    replay["step_id_mapping"]["canonical_workload_step_id"] = "vllm:352"

    with pytest.raises(
        ValueError,
        match="INVALID EXPERIMENT — injection_replay.step_id_mapping",
    ):
        validate_selected_step_injection_trace(selected, content, trace, replay)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema_version", 4),
        ("forward_duration_observation_consumed", True),
    ),
)
def test_validate_injection_requires_observation_free_schema_five(
    tmp_path, field, value
):
    selected, content, trace, replay = _injection_inputs(tmp_path)
    replay[field] = value

    with pytest.raises(
        ValueError, match=f"INVALID EXPERIMENT — injection_replay.{field}"
    ):
        validate_selected_step_injection_trace(selected, content, trace, replay)


@pytest.mark.parametrize("count", (0, 2))
def test_validate_injection_rejects_missing_or_duplicate_replay_step(
    tmp_path, count
):
    selected, content, trace, replay = _injection_inputs(tmp_path)
    replay["step_comparison"] = replay["step_comparison"] * count

    with pytest.raises(
        ValueError, match="INVALID EXPERIMENT — exact injected replay step count"
    ):
        validate_selected_step_injection_trace(selected, content, trace, replay)
