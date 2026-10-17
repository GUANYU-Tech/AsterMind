"""Contract tests for deterministic Scheduler Host service profiles."""

import hashlib
import json

import pytest

from evaluation.serving.metrics.scheduler_progress import scheduler_progress_manifest
from src.serving.host_timing import (
    SchedulerHostProfile,
    load_scheduler_host_profile,
)
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.single_pool_engine import run_single_pool_schedule
from src.serving.workload.model_config import load_model_config
from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest


def _schedule(request_count, phase="decode"):
    return ScheduleStep(
        "main:vllm:0",
        "main",
        tuple(
            ScheduledRequest(f"request-{index}", phase, 0, 1)
            for index in range(request_count)
        ),
    )


def test_scheduler_host_profile_is_deterministic_and_fails_closed():
    profile = SchedulerHostProfile(
        hardware_id="nvidia_a800",
        vllm_version="0.11.0",
        source_identity="vllm=0.11.0;source=test",
        max_scheduled_request_count=2,
        supported_phases=("decode",),
        plan_intercept_s=1e-6,
        plan_per_request_s=2e-6,
        update_intercept_s=3e-6,
        update_per_request_s=4e-6,
        profile_path="synthetic",
        profile_sha256="a" * 64,
    )

    assert profile.service_time_s(_schedule(2), "plan") == pytest.approx(5e-6)
    assert profile.service_time_s(_schedule(2), "update") == pytest.approx(11e-6)
    with pytest.raises(ValueError, match="no coverage"):
        profile.service_time_s(_schedule(3), "plan")
    with pytest.raises(ValueError, match="phase='prefill'"):
        profile.service_time_s(_schedule(1, "prefill"), "plan")


def test_scheduler_host_profile_charges_measured_fallback_dispatch_range():
    profile = SchedulerHostProfile(
        hardware_id="ascend_910b",
        vllm_version="0.18.0",
        source_identity="vllm=0.18.0;source=test",
        max_scheduled_request_count=120,
        supported_phases=("mixed", "decode"),
        plan_intercept_s=1e-6,
        plan_per_request_s=2e-6,
        update_intercept_s=3e-6,
        update_per_request_s=4e-6,
        profile_path="synthetic",
        profile_sha256="a" * 64,
        fallback_dispatch_min_scheduled_request_count=121,
        fallback_dispatch_max_scheduled_request_count=128,
        fallback_dispatch_supported_phases=("mixed", "decode"),
        fallback_dispatch_plan_s=17e-6,
        fallback_dispatch_update_s=19e-6,
        fallback_dispatch_occurrence_contract="synthetic plan and update",
    )
    assert profile.service_time_s(_schedule(121), "plan") == pytest.approx(17e-6)
    assert profile.service_time_s(_schedule(128), "update") == pytest.approx(19e-6)
    with pytest.raises(ValueError, match="no coverage"):
        profile.service_time_s(_schedule(129), "plan")
    with pytest.raises(ValueError, match="no coverage"):
        profile.service_time_s(_schedule(121, "prefill"), "plan")


def _prefill_capable_profile(**overrides):
    arguments = {
        "hardware_id": "nvidia_a800",
        "vllm_version": "0.11.0",
        "source_identity": "vllm=0.11.0;source=test",
        "max_scheduled_request_count": 4,
        "supported_phases": ("prefill", "mixed", "decode"),
        "plan_intercept_s": 1e-6,
        "plan_per_request_s": 2e-6,
        "update_intercept_s": 3e-6,
        "update_per_request_s": 4e-6,
        "prefill_plan_intercept_s": 5e-6,
        "prefill_plan_per_request_s": 6e-6,
        "prefill_update_intercept_s": 7e-6,
        "prefill_update_per_request_s": 8e-6,
        "profile_path": "synthetic",
        "profile_sha256": "a" * 64,
    }
    arguments.update(overrides)
    return SchedulerHostProfile(**arguments)


def test_prefill_steps_are_priced_by_their_own_measured_parameters():
    profile = _prefill_capable_profile()
    mixed = ScheduleStep(
        "main:vllm:0",
        "main",
        (
            ScheduledRequest("request-0", "prefill", 0, 1),
            ScheduledRequest("request-1", "decode", 0, 1),
        ),
    )

    assert profile.service_time_s(
        _schedule(2, "prefill"), "plan") == pytest.approx(17e-6)
    assert profile.service_time_s(
        _schedule(2, "prefill"), "update") == pytest.approx(23e-6)
    # Mixed and decode keep the shared line.
    assert profile.service_time_s(_schedule(2), "plan") == pytest.approx(5e-6)
    assert profile.service_time_s(mixed, "update") == pytest.approx(11e-6)
    # Coverage still fails closed past the measured request count.
    with pytest.raises(ValueError, match="no coverage"):
        profile.service_time_s(_schedule(5, "prefill"), "plan")


def test_prefill_coverage_and_parameters_must_agree():
    with pytest.raises(ValueError, match="prefill_plan_intercept_s"):
        _prefill_capable_profile(prefill_plan_intercept_s=None)
    with pytest.raises(ValueError, match="without prefill coverage"):
        _prefill_capable_profile(supported_phases=("mixed", "decode"))


def test_scheduler_host_profile_builder_requires_prefill_sample_threshold():
    from dataset_pipeline.serving_dataset.scheduler_host_profile import (
        _MIN_TRAINING_SAMPLES_PER_PHASE,
        _supported_phases,
    )

    below = (
        [(1, 100, 100, "prefill")] * (_MIN_TRAINING_SAMPLES_PER_PHASE - 1)
        + [(1, 100, 100, "decode")] * _MIN_TRAINING_SAMPLES_PER_PHASE
    )
    at_threshold = (
        [(1, 100, 100, "prefill")] * _MIN_TRAINING_SAMPLES_PER_PHASE
        + [(1, 100, 100, "decode")] * _MIN_TRAINING_SAMPLES_PER_PHASE
    )

    assert _supported_phases(below) == ("decode",)
    assert _supported_phases(at_threshold) == ("prefill", "decode")


def test_scheduler_host_profile_loader_verifies_artifact(tmp_path):
    artifact = tmp_path / "trace.jsonl.zst"
    artifact.write_bytes(b"synthetic trace")
    payload = {
        "schema_version": 1,
        "hardware_id": "nvidia_a800",
        "scheduler": {
            "vllm_version": "0.11.0",
            "source_identity": "vllm=0.11.0;source=test",
        },
        "model": {
            "kind": "scheduled_request_count_affine",
            "state_dimensions": ["scheduled_request_count"],
            "statistic": "median_of_count_medians_theil_sen",
        },
        "max_scheduled_request_count": 4,
        "supported_phases": ["decode"],
        "service_time_s": {
            "plan": {"intercept": 1e-6, "per_request": 2e-6},
            "update": {"intercept": 3e-6, "per_request": 4e-6},
        },
        "measurement": {
            "boundary": "vllm_scheduler_trace_host_monotonic_plan_and_update",
            "time_unit": "seconds",
            "training_selection": {
                "excluded_target_qps": 16,
                "minimum_samples_per_phase": 32,
                "eligible_step_rule": (
                    "nonzero_scheduled_request_count_with_positive_plan_and_update"
                ),
            },
            "artifacts": [{
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }],
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_scheduler_host_profile(profile_path)

    assert profile.service_time_s(_schedule(1), "plan") == pytest.approx(3e-6)
    artifact.write_bytes(b"tampered trace")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_scheduler_host_profile(profile_path)


def test_scheduler_host_profile_loader_requires_prefill_block(tmp_path):
    artifact = tmp_path / "trace.jsonl.zst"
    artifact.write_bytes(b"synthetic trace")
    payload = {
        "schema_version": 1,
        "hardware_id": "nvidia_a800",
        "scheduler": {
            "vllm_version": "0.11.0",
            "source_identity": "vllm=0.11.0;source=test",
        },
        "model": {
            "kind": "scheduled_request_count_affine",
            "state_dimensions": ["scheduled_request_count"],
            "statistic": "median_of_count_medians_theil_sen",
        },
        "max_scheduled_request_count": 4,
        "supported_phases": ["prefill", "mixed", "decode"],
        "service_time_s": {
            "plan": {"intercept": 1e-6, "per_request": 2e-6},
            "update": {"intercept": 3e-6, "per_request": 4e-6},
        },
        "measurement": {
            "boundary": "vllm_scheduler_trace_host_monotonic_plan_and_update",
            "time_unit": "seconds",
            "training_selection": {
                "excluded_target_qps": 16,
                "minimum_samples_per_phase": 32,
                "eligible_step_rule": (
                    "nonzero_scheduled_request_count_with_positive_plan_and_update"
                ),
            },
            "artifacts": [{
                "path": str(artifact),
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            }],
        },
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    # Declaring prefill coverage without measured prefill parameters must fail.
    with pytest.raises(ValueError, match="Scheduler Host services"):
        load_scheduler_host_profile(profile_path)

    payload["service_time_s"]["prefill"] = {
        "plan": {"intercept": 5e-6, "per_request": 6e-6},
        "update": {"intercept": 7e-6, "per_request": 8e-6},
    }
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    profile = load_scheduler_host_profile(profile_path)

    assert profile.service_time_s(
        _schedule(1, "prefill"), "plan") == pytest.approx(11e-6)
    assert profile.service_time_s(_schedule(1), "plan") == pytest.approx(3e-6)


def test_published_scheduler_host_profiles_cover_prefill():
    for hardware_id in ("nvidia_a800", "metax_c500", "ascend_910b"):
        profile = load_scheduler_host_profile(
            "configs/host_timing/scheduler_host_profiles/"
            f"{hardware_id}.json")
        assert "prefill" in profile.supported_phases
        # Every production pure-prefill step observed in the ShareGPT captures
        # schedules exactly one newly admitted request.
        assert profile.supports(_schedule(1, "prefill"), "plan")
        assert profile.service_time_s(_schedule(1, "prefill"), "plan") > 0


@pytest.mark.scheduler
def test_pinned_scheduler_uses_profile_not_live_elapsed_time():
    profile = _prefill_capable_profile()
    model = load_model_config("configs/model/llama-3.1-8b-instruct.json")
    backend = VLLMBackend.from_model(
        BackendConfig(
            "main", "colocated", 128, 4, 64,
            kv_bytes_per_token=model.kv_bytes_per_token_per_parallel_rank(1, 1),
            scheduler_pipeline_depth=1),
        VLLMSchedulerConfig(
            "configs/model/llama-3.1-8b-instruct.json",
            max_model_len=256,
            scheduler_host_profile=profile,
        ),
    )
    try:
        backend.add_request(RequestSpec(0.0, "profile-smoke", 32, 2))
        step = backend.plan_step(0.0)
        assert step is not None
        assert step.scheduler_plan_cpu_s == pytest.approx(
            profile.service_time_s(step.schedule, "plan"))
        assert backend.complete_step(step, 0.0) == pytest.approx(
            profile.service_time_s(step.schedule, "update"))
    finally:
        backend.close()


def _profiled_scheduler_progress_sha256():
    profile = _prefill_capable_profile()
    model = load_model_config("configs/model/llama-3.1-8b-instruct.json")
    backend = VLLMBackend.from_model(
        BackendConfig(
            "main", "colocated", 128, 4, 64,
            kv_bytes_per_token=model.kv_bytes_per_token_per_parallel_rank(1, 1),
            scheduler_pipeline_depth=1),
        VLLMSchedulerConfig(
            "configs/model/llama-3.1-8b-instruct.json",
            max_model_len=256,
            scheduler_host_profile=profile,
        ),
    )
    requests = RequestStream((
        RequestSpec(0.0, "profile-repeat-a", 32, 2),
        RequestSpec(0.0001, "profile-repeat-b", 32, 2),
    ))
    try:
        trace = run_single_pool_schedule(
            backend,
            requests,
            lambda _step: 1e-4,
            scheduler_pipeline_depth=1,
            timing_mode="device_plus_launch_plus_scheduler", async_output_release="self")
    finally:
        backend.close()
    return scheduler_progress_manifest(requests, trace)["sha256"]


@pytest.mark.scheduler
def test_profiled_scheduler_progress_is_bit_repeatable():
    assert _profiled_scheduler_progress_sha256() == (
        _profiled_scheduler_progress_sha256())
