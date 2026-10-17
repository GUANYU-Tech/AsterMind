from dataclasses import replace

import pytest

from src.serving.request_gen.closed_loop import ClosedLoopClients
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.multi_pool_engine import (
    HandoffDispatch,
    HandoffTransfer,
    LeastQueuedTokensRouter,
    run_multi_pool_schedule,
)
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduledRequest,
    ScheduleStep,
)
from src.serving.schedule.single_pool_engine import run_single_pool_schedule


def _stream(count, arrival=0.0):
    return RequestStream(tuple(
        RequestSpec(arrival, f"r{index}", 4, 2) for index in range(count)))


def _arrivals(stream):
    return {
        request.request_id: request.arrival_time_s
        for request in stream.requests
    }


class TestClosedLoopClients:
    def test_rejects_non_multiple_stream(self):
        with pytest.raises(ValueError, match="multiple of the client count"):
            ClosedLoopClients(_stream(3), 2)

    def test_rejects_empty_stream(self):
        with pytest.raises(ValueError, match="at least one request"):
            ClosedLoopClients(RequestStream(()), 1)

    def test_rejects_invalid_client_count(self):
        for bad in (0, -1, 1.5, True, "2"):
            with pytest.raises((TypeError, ValueError)):
                ClosedLoopClients(_stream(2), bad)

    def test_initial_requests_cover_each_client_once(self):
        clients = ClosedLoopClients(_stream(6), 2)
        initial = clients.initial_requests()
        assert tuple(request.request_id for request in initial) == ("r0", "r1")
        assert clients.requests_per_client == 3

    def test_successor_injected_at_finish_time_until_chain_exhausted(self):
        clients = ClosedLoopClients(_stream(4), 2)
        clients.initial_requests()

        successor = clients.notify_finished("r0", 1.25)
        assert successor.request_id == "r2"
        assert successor.arrival_time_s == 1.25
        successor = clients.notify_finished("r1", 2.5)
        assert successor.request_id == "r3"
        assert successor.arrival_time_s == 2.5

        assert clients.notify_finished("r2", 3.0) is None
        assert clients.notify_finished("r3", 4.0) is None

        effective = clients.effective_request_stream()
        assert _arrivals(effective) == {
            "r0": 0.0, "r1": 0.0, "r2": 1.25, "r3": 2.5,
        }

    def test_finish_before_arrival_fails_closed(self):
        clients = ClosedLoopClients(_stream(2, arrival=1.0), 1)
        clients.initial_requests()
        with pytest.raises(RuntimeError, match="before its request arrived"):
            clients.notify_finished("r0", 0.5)

    def test_unknown_request_fails_closed(self):
        clients = ClosedLoopClients(_stream(2), 1)
        clients.initial_requests()
        with pytest.raises(ValueError, match="unknown closed-loop request"):
            clients.notify_finished("ghost", 1.0)

    def test_double_finish_fails_closed(self):
        clients = ClosedLoopClients(_stream(2), 1)
        clients.initial_requests()
        clients.notify_finished("r0", 1.0)
        with pytest.raises(RuntimeError, match="more than once"):
            clients.notify_finished("r0", 2.0)

    def test_out_of_order_finish_fails_closed(self):
        clients = ClosedLoopClients(_stream(4), 1)
        clients.initial_requests()
        clients.notify_finished("r0", 1.0)
        with pytest.raises(RuntimeError, match="was outstanding"):
            clients.notify_finished("r2", 1.5)

    def test_effective_stream_fails_closed_when_run_is_truncated(self):
        clients = ClosedLoopClients(_stream(4), 2)
        clients.initial_requests()
        clients.notify_finished("r0", 1.0)
        with pytest.raises(RuntimeError, match="did not inject every"):
            clients.effective_request_stream()

    def test_arrival_transform_applies_only_to_scheduler_visible_requests(self):
        def shift(request):
            return replace(
                request, arrival_time_s=request.arrival_time_s + 0.5)

        clients = ClosedLoopClients(
            _stream(2), 1, arrival_transform=shift)
        initial = clients.initial_requests()
        assert initial[0].arrival_time_s == 0.5

        successor = clients.notify_finished("r0", 2.0)
        assert successor.arrival_time_s == 2.5

        effective = clients.effective_request_stream()
        assert _arrivals(effective) == {"r0": 0.0, "r1": 2.0}


class _OneShotBackend:
    config = BackendConfig(
        "main", "colocated", 1024, 4, 128, scheduler_pipeline_depth=1)

    def __init__(self):
        self.requests = []

    def add_request(self, request):
        self.requests.append(request)

    def has_work(self):
        return bool(self.requests)

    def plan_step(self, virtual_time_s):
        request = self.requests[0]
        scheduled = ScheduledRequest(
            request.request_id, "decode", 0, 1, request.input_tokens,
            raw_model_output_tokens=1)
        return BackendStep(
            ScheduleStep(
                request.request_id, "main", (scheduled,),
                (request.request_id,)),
            generated_tokens={request.request_id: 1},
        )

    def complete_step(self, step, virtual_time_s):
        self.requests.pop(0)


def test_single_pool_engine_injects_successors_at_completion():
    clients = ClosedLoopClients(_stream(6), 2)
    engine_requests = RequestStream(clients.initial_requests())

    trace = run_single_pool_schedule(
        _OneShotBackend(),
        engine_requests,
        lambda step: 1.0,
        1,
        async_output_release="self",
        closed_loop=clients,
    )

    assert tuple(
        entry.backend_step.schedule.step_id for entry in trace.entries
    ) == ("r0", "r1", "r2", "r3", "r4", "r5")
    assert tuple(entry.start_time_s for entry in trace.entries) == (
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

    effective = clients.effective_request_stream()
    assert _arrivals(effective) == {
        "r0": 0.0, "r1": 0.0,
        "r2": 1.0, "r3": 2.0, "r4": 3.0, "r5": 4.0,
    }


class _ColocatedPoolBackend:
    def __init__(self, pool_name):
        self.config = BackendConfig(
            pool_name, "colocated", 128, 8, 128, scheduler_pipeline_depth=1)
        self.requests = []
        self._step = 0

    def add_request(self, request):
        self.requests.append(request)

    def has_work(self):
        return bool(self.requests)

    def plan_step(self, virtual_time_s):
        request = self.requests[0]
        scheduled = ScheduledRequest(
            request.request_id, "decode", 0, 1, request.input_tokens,
            raw_model_output_tokens=1)
        step = BackendStep(
            ScheduleStep(
                f"{self.config.pool_name}:{self._step}",
                self.config.pool_name, (scheduled,),
                (request.request_id,)),
            generated_tokens={request.request_id: 1},
        )
        self._step += 1
        return step

    def complete_step(self, step, virtual_time_s):
        self.requests.pop(0)


def test_multi_pool_engine_injects_successors_on_colocated_pools():
    backends = {
        "c0": _ColocatedPoolBackend("c0"),
        "c1": _ColocatedPoolBackend("c1"),
    }
    router = LeastQueuedTokensRouter("agg", backends)
    clients = ClosedLoopClients(_stream(4), 2)
    engine_requests = RequestStream(clients.initial_requests())

    def route_request(request, now):
        return router.route(
            request.request_id,
            "ingress_to_colocated",
            ("c0", "c1"),
            request.input_tokens,
            now,
        )

    def on_completion(pool_name, step, now):
        request_id = step.schedule.requests[0].request_id
        router.set_request_tokens(
            request_id, "ingress_to_colocated", 0, pool_name)
        return ()

    trace = run_multi_pool_schedule(
        backends,
        engine_requests,
        route_request,
        lambda pool_name, step, now: 1.0,
        {pool_name: 1 for pool_name in backends},
        on_completion,
        closed_loop=clients,
    )

    assert [(
        entry.backend_step.schedule.pool_name,
        entry.start_time_s,
        entry.end_time_s,
    ) for entry in trace.entries] == [
        ("c0", 0.0, 1.0),
        ("c1", 0.0, 1.0),
        ("c0", 1.0, 2.0),
        ("c1", 1.0, 2.0),
    ]
    effective = clients.effective_request_stream()
    assert _arrivals(effective) == {
        "r0": 0.0, "r1": 0.0, "r2": 1.0, "r3": 1.0,
    }
    router.assert_drained()


class _OneStepPoolBackend:
    def __init__(self, pool_name, role):
        self.config = BackendConfig(
            pool_name, role, 128, 8, 128, scheduler_pipeline_depth=1)
        self.requests = []
        self._step = 0

    def add_request(self, request):
        self.requests.append(request)

    def has_work(self):
        return bool(self.requests)

    def plan_step(self, virtual_time_s):
        request = self.requests[0]
        if self.config.role == "prefill":
            scheduled = ScheduledRequest(
                request.request_id, "prefill", request.input_tokens, 0, 0,
                raw_model_output_tokens=1)
        elif self.config.role == "decode":
            scheduled = ScheduledRequest(
                request.request_id, "decode", 0, 1, request.input_tokens,
                raw_model_output_tokens=1)
        else:
            raise ValueError(f"unsupported test role: {self.config.role!r}")
        step = BackendStep(
            ScheduleStep(
                f"{self.config.pool_name}:{self._step}",
                self.config.pool_name, (scheduled,),
                (request.request_id,)),
            generated_tokens={request.request_id: 1},
        )
        self._step += 1
        return step

    def complete_step(self, step, virtual_time_s):
        self.requests.pop(0)


def test_multi_pool_engine_injects_only_on_terminal_decode_completion():
    backends = {
        "p": _OneStepPoolBackend("p", "prefill"),
        "d": _OneStepPoolBackend("d", "decode"),
    }
    router = LeastQueuedTokensRouter("pd", backends)
    clients = ClosedLoopClients(_stream(2), 1)
    engine_requests = RequestStream(clients.initial_requests())

    def route_request(request, now):
        return router.route(
            request.request_id,
            "ingress_to_prefill",
            ("p",),
            request.input_tokens,
            now,
        )

    def on_completion(pool_name, step, now):
        request_id = step.schedule.requests[0].request_id
        if pool_name == "p":
            router.set_request_tokens(
                request_id, "ingress_to_prefill", 0, "p")
            decision = router.route(
                request_id,
                "prefill_to_decode",
                ("d",),
                1,
                now,
                source_engine_id="p",
            )
            return (HandoffDispatch(
                decision,
                RequestSpec(now, request_id, 4, 2),
                (HandoffTransfer(
                    f"{request_id}:pd-kv-transfer:p-to-d",
                    transfer_duration_s=3.0,
                    transfer_resource_id="shared-link",
                    transfer_link="nvlink",
                    transfer_reserved_resource_ids=(
                        "communication:endpoint:fixture:p:nvlink",
                        "communication:endpoint:fixture:d:nvlink",
                        "communication:fabric:shared-link",),
                    transfer_byte_count=16,
                ),),
            ),)
        if pool_name == "d":
            router.set_request_tokens(
                request_id, "prefill_to_decode", 0, "d")
            return ()
        raise ValueError(f"unknown test pool: {pool_name!r}")

    def step_duration(pool_name, step, now):
        if pool_name == "p":
            return 2.0
        if pool_name == "d":
            return 1.0
        raise ValueError(f"unknown test pool: {pool_name!r}")

    trace = run_multi_pool_schedule(
        backends,
        engine_requests,
        route_request,
        step_duration,
        {pool_name: 1 for pool_name in backends},
        on_completion,
        closed_loop=clients,
    )

    # r0: prefill [0,2], transfer [2,5], decode [5,6].  The successor is
    # injected only when the decode pool completes r0, never at the
    # prefill handoff, so r1's chain starts at 6.0.
    assert [(
        entry.backend_step.schedule.pool_name,
        entry.start_time_s,
        entry.end_time_s,
    ) for entry in trace.entries] == [
        ("p", 0.0, 2.0),
        ("d", 5.0, 6.0),
        ("p", 6.0, 8.0),
        ("d", 11.0, 12.0),
    ]
    effective = clients.effective_request_stream()
    assert _arrivals(effective) == {"r0": 0.0, "r1": 6.0}
    router.assert_drained()
