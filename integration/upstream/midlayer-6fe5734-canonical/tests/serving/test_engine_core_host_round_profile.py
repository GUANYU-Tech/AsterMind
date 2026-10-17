"""Contract tests for deterministic EngineCore host-round pacing."""

import hashlib
import json
import types

import pytest

from src.serving.host_timing import (
    EngineCoreHostRoundProfile,
    load_engine_core_host_round_profile,
)
from src.serving.backend.vllm import VLLMSchedulerConfig
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.single_pool_engine import run_single_pool_schedule
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduleStep,
    ScheduledRequest,
)


def _profile(**overrides):
    values = {
        "hardware_id": "ascend_910b",
        "vllm_version": "0.18.0",
        "source_identity": "vllm=0.18.0;source=test",
        "max_scheduled_request_count": 4,
        "graph_submit_intercept_s": 1e-3,
        "graph_wake_intercept_s": 2e-5,
        "graph_wake_per_request_s": 3e-6,
        "eager_submit_intercept_s": 4e-4,
        "eager_wake_intercept_s": 5e-6,
        "eager_wake_per_request_s": 6e-7,
        "profile_path": "synthetic",
        "profile_sha256": "a" * 64,
    }
    values.update(overrides)
    return EngineCoreHostRoundProfile(**values)


def _schedule(request_count, phase="decode"):
    if phase == "mixed":
        phases = ("prefill", "decode") * request_count
    else:
        phases = (phase,) * request_count
    return ScheduleStep(
        "main:vllm:0",
        "main",
        tuple(
            ScheduledRequest(f"request-{index}", item, 0, 1)
            for index, item in enumerate(phases)
        ),
    )


def test_engine_core_host_round_profile_dispatches_lanes_and_fails_closed():
    profile = _profile()

    # Pure decode steps dispatch to the graph lane; submit is intercept-only.
    assert profile.service_times_s(_schedule(2)) == pytest.approx(
        (1e-3, 2e-5 + 3e-6 * 2))
    # Prefill and mixed steps dispatch to the eager lane.
    assert profile.service_times_s(_schedule(3, "prefill")) == pytest.approx(
        (4e-4, 5e-6 + 6e-7 * 3))
    with pytest.raises(ValueError, match="no coverage"):
        profile.service_times_s(_schedule(5))
    empty = types.SimpleNamespace(requests=())
    with pytest.raises(ValueError, match="requires scheduled requests"):
        profile.service_times_s(empty)


def test_engine_core_host_round_profile_rejects_invalid_values():
    with pytest.raises(ValueError, match="finite positive"):
        _profile(graph_submit_intercept_s=0.0)
    with pytest.raises(ValueError, match="finite non-negative"):
        _profile(graph_wake_per_request_s=-1e-9)
    with pytest.raises(ValueError, match="maximum request count"):
        _profile(max_scheduled_request_count=0)
    with pytest.raises(ValueError, match="SHA256"):
        _profile(profile_sha256="z" * 64)


def test_engine_core_host_round_loader_verifies_artifacts(tmp_path):
    artifact = tmp_path / "probe.jsonl"
    artifact.write_bytes(b"synthetic probe reference")
    payload = {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "scheduler": {
            "vllm_version": "0.18.0",
            "source_identity": "vllm=0.18.0;source=test",
        },
        "model": {
            "kind": "lane_dispatch_scheduled_request_count_affine",
            "state_dimensions": ["scheduled_request_count"],
            "lane_dispatch": (
                "decode_steps_to_graph_replay_lane"
                "_prefill_and_mixed_steps_to_eager_dispatch_lane"
            ),
            "statistic": "median_of_count_medians_then_least_squares_slope",
        },
        "max_scheduled_request_count": 4,
        "service_time_s": {
            "graph_lane": {
                "submit": {"intercept": 1e-3},
                "wake": {"intercept": 2e-5, "per_request": 3e-6},
            },
            "eager_lane": {
                "submit": {"intercept": 4e-4},
                "wake": {"intercept": 5e-6, "per_request": 6e-7},
            },
        },
        "measurement": {
            "boundary": "vllm_engine_core_host_round_mechanism_probe",
            "time_unit": "seconds",
            "chain_definition": "synthetic chain definition",
            "training_selection": {
                "eligible_round_rule": "non_blocked_engine_owned_rounds",
                "function_form_rule": "synthetic function form rule",
                "parameter_run": "synthetic parameter run",
                "retest_runs": ["synthetic retest run"],
            },
            "artifacts": [{
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }],
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_engine_core_host_round_profile(profile_path)

    assert profile.service_times_s(_schedule(1)) == pytest.approx(
        (1e-3, 2.3e-5))
    artifact.write_bytes(b"tampered probe reference")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_engine_core_host_round_profile(profile_path)


def test_engine_core_host_round_loader_rejects_unknown_fields(tmp_path):
    payload = json.loads(json.dumps(_synthetic_payload()))
    payload["unexpected_field"] = 1
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields differ"):
        load_engine_core_host_round_profile(profile_path)


def _synthetic_payload():
    return {
        "schema_version": 1,
        "hardware_id": "ascend_910b",
        "scheduler": {
            "vllm_version": "0.18.0",
            "source_identity": "vllm=0.18.0;source=test",
        },
        "model": {
            "kind": "lane_dispatch_scheduled_request_count_affine",
            "state_dimensions": ["scheduled_request_count"],
            "lane_dispatch": (
                "decode_steps_to_graph_replay_lane"
                "_prefill_and_mixed_steps_to_eager_dispatch_lane"
            ),
            "statistic": "median_of_count_medians_then_least_squares_slope",
        },
        "max_scheduled_request_count": 4,
        "service_time_s": {
            "graph_lane": {
                "submit": {"intercept": 1e-3},
                "wake": {"intercept": 2e-5, "per_request": 3e-6},
            },
            "eager_lane": {
                "submit": {"intercept": 4e-4},
                "wake": {"intercept": 5e-6, "per_request": 6e-7},
            },
        },
        "measurement": {
            "boundary": "vllm_engine_core_host_round_mechanism_probe",
            "time_unit": "seconds",
            "chain_definition": "synthetic chain definition",
            "training_selection": {
                "eligible_round_rule": "non_blocked_engine_owned_rounds",
                "function_form_rule": "synthetic function form rule",
                "parameter_run": "synthetic parameter run",
                "retest_runs": ["synthetic retest run"],
            },
            "artifacts": [],
        },
    }


def test_published_engine_core_host_round_profile_covers_both_lanes():
    profile = load_engine_core_host_round_profile(
        "configs/host_timing/engine_core_host_round_profiles/"
        "ascend_910b.json")

    for count in (1, 120, 128):
        submit, wake = profile.service_times_s(_schedule(count))
        assert submit > 0 and wake > 0
    for count in (1, 64):
        submit, wake = profile.service_times_s(_schedule(count, "mixed"))
        assert submit > 0 and wake > 0


def test_vllm_scheduler_config_validates_engine_core_profile():
    profile = _profile()
    config = VLLMSchedulerConfig(
        "test-model", engine_core_host_round_profile=profile)
    assert config.engine_core_host_round_profile is profile
    with pytest.raises(TypeError, match="EngineCore host-round profile"):
        VLLMSchedulerConfig(
            "test-model", engine_core_host_round_profile="not-a-profile")


class _EngineRoundBackend:
    """Fixed-service fixture: plan 0.2, submit 0.15, wake 0.1, update 0.3."""

    config = None

    def __init__(self, engine_fields=True):
        self.requests = []
        self.engine_fields = engine_fields

    def add_request(self, request):
        self.requests.append(request)

    def has_work(self):
        return bool(self.requests)

    def plan_step(self, virtual_time_s):
        request = self.requests[0]
        scheduled = ScheduledRequest(
            request.request_id, "decode", request.input_tokens, 1,
            raw_model_output_tokens=0)
        fields = {}
        if self.engine_fields:
            fields = {
                "scheduler_plan_cpu_s": 0.2,
                "engine_submit_host_s": 0.15,
                "engine_wake_host_s": 0.1,
            }
        return BackendStep(
            ScheduleStep(request.request_id, "main", (scheduled,)),
            generated_tokens={request.request_id: 0},
            **fields,
        )

    def complete_step(self, step, virtual_time_s):
        self.requests.pop(0)
        return 0.3 if self.engine_fields else 0.0


def test_engine_core_host_round_chains_submit_and_wake_in_full_mode():
    requests = RequestStream((RequestSpec(1.0, "a", 1, 1),))
    trace = run_single_pool_schedule(
        _EngineRoundBackend(),
        requests,
        lambda step: 0.5,
        1,
        timing_mode="device_plus_launch_plus_scheduler", async_output_release="self")

    entry = trace.entries[0]
    # plan (0.2) + submit (0.15) precede device execution; wake (0.1)
    # precedes the update (0.3).
    assert entry.start_time_s == pytest.approx(1.0)
    assert entry.execution_wall_s == pytest.approx(0.5)
    assert entry.device_queue_wait_s == pytest.approx(0.15)
    assert entry.completion_queue_wait_s == pytest.approx(0.1)
    assert entry.end_time_s == pytest.approx(1.0 + 0.2 + 0.15 + 0.5 + 0.1 + 0.3)


def test_engine_core_host_round_paces_consecutive_rounds():
    requests = RequestStream((
        RequestSpec(0.0, "a", 1, 1), RequestSpec(0.0, "b", 1, 1)))
    trace = run_single_pool_schedule(
        _EngineRoundBackend(),
        requests,
        lambda step: 0.5,
        1,
        timing_mode="device_plus_launch_plus_scheduler", async_output_release="self")

    # Round one: plan 0.0, submit ends 0.35, device 0.35-0.85, wake ends
    # 0.95, update ends 1.25.  Round two cannot be planned before the whole
    # host round completes.
    assert [entry.start_time_s for entry in trace.entries] == pytest.approx(
        [0.0, 1.25])
    assert [entry.end_time_s for entry in trace.entries] == pytest.approx(
        [1.25, 2.5])


def test_engine_core_host_round_fields_are_ignored_below_full_mode():
    requests = RequestStream((
        RequestSpec(0.0, "a", 1, 1), RequestSpec(0.0, "b", 1, 1)))

    with_fields = run_single_pool_schedule(
        _EngineRoundBackend(engine_fields=True),
        requests,
        lambda step: 0.5,
        1,
        timing_mode="device_only", async_output_release="self")
    without_fields = run_single_pool_schedule(
        _EngineRoundBackend(engine_fields=False),
        requests,
        lambda step: 0.5,
        1,
        timing_mode="device_only", async_output_release="self")

    assert with_fields.entries == without_fields.entries


def test_backend_step_rejects_invalid_engine_host_fields():
    scheduled = ScheduledRequest("a", "decode", 1, 1, raw_model_output_tokens=0)
    schedule = ScheduleStep("main:vllm:0", "main", (scheduled,))
    with pytest.raises(ValueError, match="engine_submit_host_s"):
        BackendStep(
            schedule, generated_tokens={"a": 0}, engine_submit_host_s=-1.0)
    with pytest.raises(ValueError, match="engine_wake_host_s"):
        BackendStep(
            schedule, generated_tokens={"a": 0}, engine_wake_host_s=float("nan"))
