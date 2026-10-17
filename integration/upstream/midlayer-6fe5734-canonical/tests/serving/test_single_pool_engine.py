import math

import pytest

from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.single_pool_engine import run_single_pool_schedule
from src.serving.schedule.schedule_step import BackendStep, ScheduleStep, ScheduledRequest
from src.serving.schedule.schedule_trace import ScheduleTrace, ScheduleTraceEntry


class _Backend:
    config = BackendConfig("main", "colocated", 1024, 4, 128, scheduler_pipeline_depth=1)

    def __init__(self):
        self.requests = []

    def add_request(self, request):
        self.requests.append(request)

    def has_work(self):
        return bool(self.requests)

    def plan_step(self, virtual_time_s):
        request = self.requests[0]
        scheduled = ScheduledRequest(
            request.request_id, "prefill", request.input_tokens, 1,
            raw_model_output_tokens=0)
        return BackendStep(
            ScheduleStep(request.request_id, "main", (scheduled,)),
            generated_tokens={request.request_id: 0},
        )

    def complete_step(self, step, virtual_time_s):
        self.requests.pop(0)


class _BulkBackend(_Backend):
    def __init__(self):
        super().__init__()
        self.admission_batches = []

    def add_requests(self, requests):
        self.admission_batches.append(requests)
        self.requests.extend(requests)


class _TimedBackend(_Backend):
    def plan_step(self, virtual_time_s):
        original = super().plan_step(virtual_time_s)
        return BackendStep(
            original.schedule,
            original.kv_cache_events,
            original.generated_tokens,
            scheduler_plan_cpu_s=0.2,
        )

    def complete_step(self, step, virtual_time_s):
        super().complete_step(step, virtual_time_s)
        return 0.3


class _PipelinedBackend:
    """Source-derived queue fixture: plan/submit precedes FIFO update."""

    config = BackendConfig("main", "colocated", 1024, 8, 128, scheduler_pipeline_depth=1)

    def __init__(self):
        self.waiting = []
        self.pending = []
        self.completed = []

    def add_request(self, request):
        self.waiting.append(request)

    def has_work(self):
        return bool(self.waiting or self.pending)

    def plan_step(self, virtual_time_s):
        if not self.waiting:
            return None
        request = self.waiting.pop(0)
        scheduled = ScheduledRequest(
            request.request_id, "prefill", request.input_tokens, 1,
            raw_model_output_tokens=0)
        step = BackendStep(
            ScheduleStep(request.request_id, "main", (scheduled,)),
            generated_tokens={request.request_id: 0},
        )
        self.pending.append(step)
        return step

    def complete_step(self, step, virtual_time_s):
        if not self.pending or self.pending[0] is not step:
            raise ValueError("fixture completion must release the oldest slot")
        self.pending.pop(0)
        self.completed.append(step.schedule.step_id)
        return 0.0


class _TimedPipelinedBackend(_PipelinedBackend):
    def plan_step(self, virtual_time_s):
        step = super().plan_step(virtual_time_s)
        if step is None:
            return None
        timed = BackendStep(
            step.schedule,
            step.kv_cache_events,
            step.generated_tokens,
            scheduler_plan_cpu_s=0.2,
        )
        self.pending[-1] = timed
        return timed

    def complete_step(self, step, virtual_time_s):
        super().complete_step(step, virtual_time_s)
        return 0.3


def test_single_pool_engine_advances_to_arrivals_and_predicted_completions():
    requests = RequestStream((
        RequestSpec(1.0, "a", 1, 1), RequestSpec(3.0, "b", 1, 1)))
    trace = run_single_pool_schedule(
        _Backend(), requests, lambda step: 0.25, 1, async_output_release="self")
    assert [(entry.start_time_s, entry.end_time_s) for entry in trace.entries] == [
        (1.0, 1.25), (3.0, 3.25)]


def test_single_pool_engine_reports_completed_steps():
    requests = RequestStream((RequestSpec(1.0, "a", 1, 1),))
    completed = []
    trace = run_single_pool_schedule(
        _Backend(),
        requests,
        lambda step: 0.25,
        1,
        on_step_complete=lambda step, now: completed.append((step, now)), async_output_release="self")
    assert completed == [(trace.entries[0].backend_step, 1.25)]


def test_single_pool_engine_batches_arrivals_without_changing_trace():
    requests = RequestStream((
        RequestSpec(0.0, "a", 1, 1),
        RequestSpec(0.0, "b", 1, 1),
        RequestSpec(1.0, "c", 1, 1),
    ))
    sequential_trace = run_single_pool_schedule(
        _Backend(), requests, lambda step: 0.25, 1, async_output_release="self")
    bulk_backend = _BulkBackend()
    bulk_trace = run_single_pool_schedule(
        bulk_backend, requests, lambda step: 0.25, 1, async_output_release="self")

    assert bulk_trace == sequential_trace
    assert tuple(
        tuple(request.request_id for request in batch)
        for batch in bulk_backend.admission_batches
    ) == (("a", "b"), ("c",))


def test_scheduler_cpu_is_applied_linearly_at_step_boundaries():
    requests = RequestStream((RequestSpec(1.0, "a", 1, 1),))
    trace = run_single_pool_schedule(
        _TimedBackend(),
        requests,
        lambda step: 0.5,
        1,
        timing_mode="device_plus_launch_plus_scheduler", async_output_release="self")

    entry = trace.entries[0]
    assert entry.start_time_s == 1.0
    assert entry.end_time_s == 2.0
    assert entry.scheduler_plan_cpu_s == 0.2
    assert entry.execution_wall_s == 0.5
    assert entry.scheduler_update_cpu_s == 0.3


@pytest.mark.parametrize(
    ("depth", "expected_starts", "expected_device_queue_waits"),
    [
        (1, (0.0, 1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 0.0)),
        (2, (0.0, 0.0, 1.0, 2.0), (0.0, 1.0, 1.0, 1.0)),
        (3, (0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 2.0, 2.0)),
        (4, (0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 2.0, 3.0)),
    ],
)
def test_pipeline_depth_controls_submit_slots_not_device_parallelism(
        depth, expected_starts, expected_device_queue_waits):
    requests = RequestStream(tuple(
        RequestSpec(0.0, request_id, 1, 1)
        for request_id in ("a", "b", "c", "d")
    ))
    backend = _PipelinedBackend()

    trace = run_single_pool_schedule(
        backend,
        requests,
        lambda step: 1.0,
        depth, async_output_release="self")

    assert tuple(entry.start_time_s for entry in trace.entries) == expected_starts
    assert tuple(entry.end_time_s for entry in trace.entries) == (1.0, 2.0, 3.0, 4.0)
    assert tuple(entry.execution_wall_s for entry in trace.entries) == (1.0,) * 4
    assert tuple(entry.device_queue_wait_s for entry in trace.entries) == (
        expected_device_queue_waits)
    assert backend.completed == ["a", "b", "c", "d"]


def test_pipeline_depth_waits_for_empty_source_decision_with_full_timing():
    requests = RequestStream((
        RequestSpec(0.0, "a", 1, 1),
        RequestSpec(0.0, "b", 1, 1),
    ))
    backend = _TimedPipelinedBackend()

    trace = run_single_pool_schedule(
        backend,
        requests,
        lambda step: 1.0,
        2,
        timing_mode="device_plus_launch_plus_scheduler", async_output_release="self")

    assert tuple(entry.start_time_s for entry in trace.entries) == (0.0, 0.2)
    assert tuple(entry.end_time_s for entry in trace.entries) == (1.5, 2.5)
    assert tuple(entry.device_queue_wait_s for entry in trace.entries) == pytest.approx(
        (0.0, 0.8))
    assert tuple(entry.completion_queue_wait_s for entry in trace.entries) == (
        0.0, 0.0)
    assert backend.completed == ["a", "b"]


@pytest.mark.parametrize("depth", [True, 0, -1, 1.0, "2", None])
def test_single_pool_requires_explicit_positive_non_bool_depth(depth):
    requests = RequestStream((RequestSpec(0.0, "a", 1, 1),))
    with pytest.raises((TypeError, ValueError), match="depth"):
        run_single_pool_schedule(_Backend(), requests, lambda step: 1.0, depth, async_output_release="self")


@pytest.mark.parametrize("value", [True, "0.25", None])
def test_single_pool_engine_rejects_non_numeric_duration(value):
    requests = RequestStream((RequestSpec(0.0, "a", 1, 1),))
    with pytest.raises(TypeError, match="duration must be numeric"):
        run_single_pool_schedule(_Backend(), requests, lambda step: value, 1, async_output_release="self")


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_schedule_trace_rejects_non_finite_intervals(value):
    request = ScheduledRequest("request", "prefill", 1, 0)
    step = BackendStep(
        ScheduleStep("step", "main", (request,)),
        generated_tokens={"request": 0},
    )
    with pytest.raises(ValueError, match="schedule trace interval"):
        ScheduleTraceEntry(step, 0.0, value)


def test_schedule_trace_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one entry"):
        ScheduleTrace(())


@pytest.mark.parametrize("value", ["", "oldest", "newest", None, 0])
def test_single_pool_rejects_unknown_async_output_release(value):
    requests = RequestStream((RequestSpec(0.0, "a", 1, 1),))
    with pytest.raises(ValueError, match="async output release"):
        run_single_pool_schedule(
            _Backend(), requests, lambda step: 1.0, 1,
            async_output_release=value)


def test_successor_release_holds_the_output_until_the_next_step_ends():
    """The measured Ascend semantics: a step's output is taken only after the
    successor's device work ends, so a long successor delays the update while
    "self" release closes the step at its own device end."""

    class _DepthBackend(_Backend):
        """Pops on plan so a depth-two queue holds two distinct steps."""

        def __init__(self):
            super().__init__()
            self.planned = []

        def has_work(self):
            return bool(self.requests)

        def plan_step(self, virtual_time_s):
            step = super().plan_step(virtual_time_s)
            self.planned.append(self.requests.pop(0))
            return step

        def complete_step(self, step, virtual_time_s):
            return None

    requests = RequestStream(tuple(
        RequestSpec(0.0, f"r{index}", 1, 1) for index in range(3)))

    def duration(step):
        return 4.0 if step.schedule.step_id == "r1" else 1.0

    ends = {}
    for release in ("self", "successor"):
        trace = run_single_pool_schedule(
            _DepthBackend(), requests, duration, 2,
            async_output_release=release)
        ends[release] = [entry.end_time_s for entry in trace.entries]

    # r0 runs 1.0 and its successor r1 runs 4.0: self release closes r0 at its
    # own device end, successor release holds it until r1's device end.
    assert ends["self"][0] == pytest.approx(1.0)
    assert ends["successor"][0] == pytest.approx(5.0)
    assert ends["successor"][0] > ends["self"][0]
