"""Drive one scheduler backend on a virtual clock."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Callable, Iterable

from src.serving.request_gen.closed_loop import ClosedLoopClients
from src.serving.request_gen.request_stream import RequestStream

from .backend_protocol import BulkRequestSchedulerBackend, SchedulerBackend
from .pipeline_depth import validate_scheduler_pipeline_depth
from .schedule_step import BackendStep
from .schedule_trace import ScheduleTrace, ScheduleTraceEntry
from src.serving.host_timing import validate_serving_timing_mode


StepDuration = Callable[[BackendStep], float]
StepComplete = Callable[[BackendStep, float], None]
TailDrain = Callable[
    [SchedulerBackend], Iterable[tuple[BackendStep, float]]]


@dataclass
class _InFlightStep:
    step: BackendStep
    plan_start_s: float
    scheduler_plan_cpu_s: float
    engine_submit_host_s: float
    engine_wake_host_s: float
    execution_start_s: float
    execution_end_s: float
    execution_wall_s: float


def run_single_pool_schedule(
    backend: SchedulerBackend,
    requests: RequestStream,
    step_duration: StepDuration,
    scheduler_pipeline_depth: int,
    *,
    async_output_release: str,
    on_step_complete: StepComplete | None = None,
    tail_drain: TailDrain | None = None,
    timing_mode: str = "device_only",
    closed_loop: ClosedLoopClients | None = None,
) -> ScheduleTrace:
    if on_step_complete is not None and not callable(on_step_complete):
        raise TypeError("single-pool step-complete callback must be callable")
    if tail_drain is not None and not callable(tail_drain):
        raise TypeError("single-pool tail drain must be callable")
    if closed_loop is not None and not isinstance(
            closed_loop, ClosedLoopClients):
        raise TypeError(
            "single-pool closed_loop must be a ClosedLoopClients")
    depth = validate_scheduler_pipeline_depth(
        scheduler_pipeline_depth, "single-pool scheduler pipeline depth")
    # Which in-flight step's device completion releases the oldest output.
    # "self" is the upstream semantics (vLLM 0.18 core.py:492-506 blocks on the
    # oldest future, whose copy event is recorded behind that step's own work,
    # gpu_model_runner.py:241-253/261).  "successor" is the measured Ascend
    # semantics; both are declared per hardware in HWSpec features.
    if async_output_release == "self":
        release_index = 0
    elif async_output_release == "successor":
        release_index = -1
    else:
        raise ValueError(
            "single-pool async output release must be 'self' or 'successor', "
            f"got {async_output_release!r}")
    validate_serving_timing_mode(timing_mode)
    if tail_drain is not None and (
            depth != 1 or timing_mode != "device_only"):
        raise NotImplementedError(
            "single-pool tail drain requires depth-one device-only timing")
    pending = iter(requests.requests)
    next_request = next(pending, None)
    pending_injected: list = []
    virtual_time = 0.0
    cpu_ready_s = 0.0
    device_ready_s = 0.0
    in_flight: list[_InFlightStep] = []
    pending_update: tuple[_InFlightStep, float] | None = None
    entries: list[ScheduleTraceEntry] = []

    def inject_successors(step: BackendStep, completion_time_s: float) -> None:
        if closed_loop is None:
            return
        for request_id in step.finished_request_ids:
            successor = closed_loop.notify_finished(
                request_id, completion_time_s)
            if successor is not None:
                pending_injected.append(successor)

    while (next_request is not None or pending_injected or backend.has_work()
           or in_flight or pending_update is not None):
        if pending_update is not None and cpu_ready_s <= virtual_time:
            completed, completion_time_s = pending_update
            if not in_flight or in_flight[0] is not completed:
                raise RuntimeError("single-pool Scheduler slot order is inconsistent")
            in_flight.pop(0)
            if on_step_complete is not None:
                on_step_complete(completed.step, completion_time_s)
            inject_successors(completed.step, completion_time_s)
            pending_update = None

        arrived_requests = []
        while next_request is not None and next_request.arrival_time_s <= virtual_time:
            arrived_requests.append(next_request)
            next_request = next(pending, None)
        while (pending_injected
               and pending_injected[0].arrival_time_s <= virtual_time):
            arrived_requests.append(pending_injected.pop(0))
        if arrived_requests:
            if isinstance(backend, BulkRequestSchedulerBackend):
                backend.add_requests(tuple(arrived_requests))
            else:
                for arrived_request in arrived_requests:
                    backend.add_request(arrived_request)

        if (next_request is None and not pending_injected
                and tail_drain is not None
                and not in_flight and pending_update is None
                and backend.has_work()):
            for step, duration_s in tail_drain(backend):
                if isinstance(duration_s, bool) or not isinstance(duration_s, Real):
                    raise TypeError("predicted step duration must be numeric")
                if not math.isfinite(duration_s) or duration_s < 0:
                    raise ValueError(
                        "predicted step duration must be finite and non-negative")
                completion_time_s = virtual_time + float(duration_s)
                entries.append(ScheduleTraceEntry(
                    step,
                    virtual_time,
                    completion_time_s,
                    timing_mode,
                    0.0,
                    0.0,
                    float(duration_s),
                ))
                if on_step_complete is not None:
                    on_step_complete(step, completion_time_s)
                inject_successors(step, completion_time_s)
                virtual_time = completion_time_s
            if backend.has_work():
                raise RuntimeError(
                    "single-pool tail drain returned with pending work")
            if pending_injected:
                # A finished client's successor arrived during the drain;
                # return to the main loop so it is scheduled causally.
                continue
            break

        if (pending_update is None and len(in_flight) < depth
                and cpu_ready_s <= virtual_time and backend.has_work()):
            step = backend.plan_step(virtual_time)
            if step is None:
                # A source-derived backend can have only submitted steps left;
                # EngineCore then waits for the oldest future rather than
                # inventing a new batch.
                if not in_flight:
                    raise RuntimeError(
                        "backend has work but plan_step returned no step")
            else:
                duration_s = step_duration(step)
                if isinstance(duration_s, bool) or not isinstance(duration_s, Real):
                    raise TypeError("predicted step duration must be numeric")
                if not math.isfinite(duration_s) or duration_s < 0:
                    raise ValueError(
                        "predicted step duration must be finite and non-negative")
                scheduler_plan_cpu_s = {
                    "device_only": 0.0,
                    "device_plus_launch": 0.0,
                    "device_plus_launch_plus_scheduler": step.scheduler_plan_cpu_s,
                }[timing_mode]
                engine_submit_host_s = {
                    "device_only": 0.0,
                    "device_plus_launch": 0.0,
                    "device_plus_launch_plus_scheduler": step.engine_submit_host_s,
                }[timing_mode]
                engine_wake_host_s = {
                    "device_only": 0.0,
                    "device_plus_launch": 0.0,
                    "device_plus_launch_plus_scheduler": step.engine_wake_host_s,
                }[timing_mode]
                # The EngineCore busy loop pays its submit-lane host chain
                # (input-prep sync, pinned H2D, pre-replay sync wall cost,
                # sampler/thread submits, queue poll) after planning; the
                # device start still waits for both the host and the device.
                execution_start_s = max(
                    virtual_time + scheduler_plan_cpu_s + engine_submit_host_s,
                    device_ready_s)
                execution_end_s = execution_start_s + float(duration_s)
                in_flight.append(_InFlightStep(
                    step,
                    virtual_time,
                    scheduler_plan_cpu_s,
                    engine_submit_host_s,
                    engine_wake_host_s,
                    execution_start_s,
                    execution_end_s,
                    float(duration_s),
                ))
                cpu_ready_s = (
                    virtual_time + scheduler_plan_cpu_s + engine_submit_host_s)
                device_ready_s = execution_end_s
                if (len(in_flight) < depth
                        and in_flight[release_index].execution_end_s
                        > virtual_time):
                    # vLLM 0.18 core.py:476-482: while the batch queue still
                    # has room and the releasing future is not done, the round
                    # ends without taking an output -- "fulfilling the batch
                    # queue has a higher priority than getting model outputs".
                    continue

        if (pending_update is None and in_flight
                and in_flight[release_index].execution_end_s <= virtual_time
                and cpu_ready_s <= virtual_time):
            completed = in_flight[0]
            update_value = backend.complete_step(completed.step, virtual_time)
            if timing_mode == "device_plus_launch_plus_scheduler":
                if (isinstance(update_value, bool)
                        or not isinstance(update_value, Real)
                        or not math.isfinite(update_value)
                        or update_value < 0):
                    raise ValueError(
                        "Scheduler update CPU time must be finite and non-negative")
                scheduler_update_cpu_s = float(update_value)
            elif timing_mode in ("device_only", "device_plus_launch"):
                scheduler_update_cpu_s = 0.0
            else:
                raise ValueError(f"unsupported Serving timing mode: {timing_mode!r}")
            # The worker thread wakes (D2H event sync, tolist, output put)
            # after the releasing step's device work finishes; the Scheduler
            # update consumes the woken output, so the completion queue wait
            # includes the wake host chain.
            update_start_s = max(
                virtual_time,
                in_flight[release_index].execution_end_s
                + completed.engine_wake_host_s)
            completion_time_s = update_start_s + scheduler_update_cpu_s
            entries.append(ScheduleTraceEntry(
                completed.step,
                completed.plan_start_s,
                completion_time_s,
                timing_mode,
                completed.scheduler_plan_cpu_s,
                scheduler_update_cpu_s,
                completed.execution_wall_s,
                completed.execution_start_s - (
                    completed.plan_start_s + completed.scheduler_plan_cpu_s),
                update_start_s - completed.execution_end_s,
            ))
            pending_update = (completed, completion_time_s)
            cpu_ready_s = completion_time_s
            continue

        next_times = []
        if next_request is not None:
            next_times.append(next_request.arrival_time_s)
        if pending_update is not None and cpu_ready_s > virtual_time:
            next_times.append(cpu_ready_s)
        elif cpu_ready_s > virtual_time:
            next_times.append(cpu_ready_s)
        if (in_flight
                and in_flight[release_index].execution_end_s > virtual_time):
            next_times.append(in_flight[release_index].execution_end_s)
        if not next_times:
            if backend.has_work():
                raise RuntimeError(
                    "single-pool Scheduler has work but no causal event")
            break
        next_time = min(next_times)
        if next_time <= virtual_time:
            raise RuntimeError("single-pool Scheduler made no causal progress")
        virtual_time = next_time

    return ScheduleTrace(sorted(entries, key=lambda entry: entry.start_time_s))
