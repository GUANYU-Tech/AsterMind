"""Deterministic virtual-time contract for the native vLLM adapter."""

import hashlib

from evaluation.serving.metrics.scheduler_progress import (
    scheduler_progress_manifest,
)
from src.serving.backend.vllm import _VIRTUAL_SCHEDULER_CPU_S
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduleStep,
    ScheduledRequest,
    schedule_step_sha256,
)
from src.serving.schedule.single_pool_engine import run_single_pool_schedule


class _MinimalSchedulerBackend:
    config = BackendConfig(
        "main", "colocated", 16, 2, 16, scheduler_pipeline_depth=1)

    def __init__(self):
        self._waiting = []

    def add_request(self, request):
        self._waiting.append(request)

    def has_work(self):
        return bool(self._waiting)

    def plan_step(self, virtual_time_s):
        request = self._waiting[0]
        scheduled = ScheduledRequest(
            request.request_id,
            "prefill",
            request.input_tokens,
            0,
        )
        return BackendStep(
            ScheduleStep(
                f"main:{request.request_id}",
                "main",
                (scheduled,),
                finished_request_ids=(request.request_id,),
            ),
            generated_tokens={request.request_id: 0},
            scheduler_plan_cpu_s=_VIRTUAL_SCHEDULER_CPU_S,
        )

    def complete_step(self, step, virtual_time_s):
        self._waiting.pop(0)
        return _VIRTUAL_SCHEDULER_CPU_S


def _run_minimal_case():
    requests = RequestStream((
        RequestSpec(0.0, "request-a", 2, 0),
        RequestSpec(0.1, "request-b", 3, 0),
    ))
    trace = run_single_pool_schedule(
        _MinimalSchedulerBackend(),
        requests,
        lambda _step: 0.1,
        scheduler_pipeline_depth=1,
        timing_mode="device_plus_launch_plus_scheduler", async_output_release="self")
    artifact_sha256 = hashlib.sha256(
        "".join(
            schedule_step_sha256(entry.backend_step.schedule)
            for entry in trace.entries
        ).encode("utf-8")
    ).hexdigest()
    return scheduler_progress_manifest(requests, trace)["sha256"], artifact_sha256


def test_native_scheduler_virtual_time_is_repeatable():
    first_progress, first_artifact = _run_minimal_case()
    second_progress, second_artifact = _run_minimal_case()

    assert (first_progress, first_artifact) == (second_progress, second_artifact)
