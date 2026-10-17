import pytest

from src.serving.simulation import _is_discarded_async_terminal_step
from src.serving.deployment.deployment_config import RouterConfig
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.backend_protocol import BackendConfig, backend_phase
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


class _OneStepBackend:
    def __init__(self, pool_name, role):
        self.config = BackendConfig(pool_name, role, 128, 8, 128, scheduler_pipeline_depth=1)
        self.requests = []
        self._pending = None
        self._step = 0

    def add_request(self, request):
        self.requests.append(request)

    def has_work(self):
        return bool(self.requests)

    def plan_step(self, virtual_time_s):
        if self._pending is not None:
            raise RuntimeError("complete pending test step")
        request = self.requests[0]
        phase = backend_phase(self.config.role)
        if phase == "prefill":
            scheduled = ScheduledRequest(
                request.request_id,
                "prefill",
                request.input_tokens,
                0,
                0,
                raw_model_output_tokens=1,
            )
        elif phase == "decode":
            scheduled = ScheduledRequest(
                request.request_id,
                "decode",
                0,
                1,
                request.input_tokens,
                raw_model_output_tokens=1,
            )
        else:
            raise ValueError(f"unsupported test backend phase: {phase!r}")
        step = BackendStep(
            ScheduleStep(
                f"{self.config.pool_name}:{self._step}",
                self.config.pool_name,
                (scheduled,),
                (request.request_id,),
            ),
            generated_tokens={request.request_id: 1},
        )
        self._pending = step
        self._step += 1
        return step

    def complete_step(self, step, virtual_time_s):
        if step is not self._pending:
            raise ValueError("test completion does not match pending step")
        self.requests.pop(0)
        self._pending = None

    def close(self):
        return None


class _PipelinedBackend:
    """FIFO plan/submit fixture with no device parallelism."""

    def __init__(self, pool_name, role):
        self.config = BackendConfig(
            pool_name, role, 128, 8, 128, scheduler_pipeline_depth=1)
        self.waiting = []
        self.pending = []
        self.step_index = 0

    def add_request(self, request):
        self.waiting.append(request)

    def has_work(self):
        return bool(self.waiting or self.pending)

    def plan_step(self, virtual_time_s):
        if not self.waiting:
            return None
        request = self.waiting.pop(0)
        scheduled = ScheduledRequest(
            request.request_id,
            backend_phase(self.config.role),
            request.input_tokens,
            0,
            raw_model_output_tokens=0,
        )
        step = BackendStep(
            ScheduleStep(
                f"{self.config.pool_name}:{self.step_index}",
                self.config.pool_name,
                (scheduled,),
            ),
            generated_tokens={request.request_id: 0},
        )
        self.step_index += 1
        self.pending.append(step)
        return step

    def complete_step(self, step, virtual_time_s):
        if not self.pending or self.pending[0] is not step:
            raise ValueError("fixture update must release the oldest pool slot")
        self.pending.pop(0)
        return 0.0

    def close(self):
        return None


def test_least_queued_tokens_reserves_each_same_time_decision():
    router = LeastQueuedTokensRouter("deployment", ("p1", "p0"))

    first = router.route(
        "r0", "ingress_to_prefill", ("p1", "p0"), 4, 0.0)
    second = router.route(
        "r1", "ingress_to_prefill", ("p1", "p0"), 4, 0.0)

    assert first.selected_engine_id == "p0"
    assert first.candidate_queued_tokens == (("p0", 0), ("p1", 0))
    assert second.selected_engine_id == "p1"
    assert second.candidate_queued_tokens == (("p0", 4), ("p1", 0))
    assert dict(router.queued_tokens) == {"p0": 4, "p1": 4}


def test_router_serializes_configured_dispatch_service():
    router = LeastQueuedTokensRouter(
        "deployment",
        ("p0", "p1"),
        RouterConfig("ingress-cpu", 0.5),
    )

    first = router.route(
        "r0", "ingress_to_prefill", ("p0", "p1"), 4, 0.0)
    second = router.route(
        "r1", "ingress_to_prefill", ("p0", "p1"), 4, 0.0)
    after_idle = router.route(
        "r2", "ingress_to_prefill", ("p0", "p1"), 1, 2.0)

    assert (
        first.queue_enter_time_s,
        first.decision_time_s,
        first.queue_exit_time_s,
        first.dispatch_time_s,
    ) == (0.0, 0.0, 0.0, 0.5)
    assert (
        second.queue_enter_time_s,
        second.decision_time_s,
        second.queue_exit_time_s,
        second.dispatch_time_s,
    ) == (0.0, 0.0, 0.5, 1.0)
    assert (
        after_idle.queue_enter_time_s,
        after_idle.decision_time_s,
        after_idle.queue_exit_time_s,
        after_idle.dispatch_time_s,
    ) == (2.0, 2.0, 2.0, 2.5)
    assert [
        first.selected_engine_id,
        second.selected_engine_id,
        after_idle.selected_engine_id,
    ] == ["p0", "p1", "p0"]


def test_multi_pool_admits_only_after_serial_router_dispatch():
    backends = {
        "p1": _OneStepBackend("p1", "prefill"),
        "p0": _OneStepBackend("p0", "prefill"),
    }
    router = LeastQueuedTokensRouter(
        "deployment",
        backends,
        RouterConfig("ingress-cpu", 1.0),
    )
    requests = RequestStream((
        RequestSpec(0.0, "r0", 2, 1),
        RequestSpec(0.0, "r1", 2, 1),
    ))

    def route_request(request, now):
        return router.route(
            request.request_id,
            "ingress_to_prefill",
            ("p1", "p0"),
            request.input_tokens,
            now,
        )

    def on_completion(pool_name, step, now):
        request_id = step.schedule.requests[0].request_id
        router.set_request_tokens(
            request_id, "ingress_to_prefill", 0, pool_name)
        return ()

    completed = []
    trace = run_multi_pool_schedule(
        backends,
        requests,
        route_request,
        lambda pool_name, step, now: 2.0,
        {pool_name: 1 for pool_name in backends},
        on_completion,
        on_step_complete=lambda step, now: completed.append(
            (step.schedule.pool_name, now)),
    )

    assert [(
        decision.request_id,
        decision.selected_engine_id,
        decision.queue_exit_time_s,
        decision.dispatch_time_s,
    ) for decision in trace.route_decisions] == [
        ("r0", "p0", 0.0, 1.0),
        ("r1", "p1", 1.0, 2.0),
    ]
    assert [(
        entry.backend_step.schedule.pool_name,
        entry.start_time_s,
        entry.end_time_s,
    ) for entry in trace.entries] == [
        ("p0", 1.0, 3.0),
        ("p1", 2.0, 4.0),
    ]
    assert completed == [("p0", 3.0), ("p1", 4.0)]


def test_router_updates_visible_remaining_work_without_future_output_length():
    router = LeastQueuedTokensRouter("deployment", ("p0", "p1"))
    router.route("long-output", "ingress_to_prefill", ("p0", "p1"), 3, 0.0)
    router.route("short-output", "ingress_to_prefill", ("p0", "p1"), 2, 0.0)

    router.set_request_tokens(
        "long-output", "ingress_to_prefill", 1, "p0")
    decision = router.route(
        "new", "ingress_to_prefill", ("p0", "p1"), 1, 1.0)

    assert decision.candidate_queued_tokens == (("p0", 1), ("p1", 2))
    assert decision.selected_engine_id == "p0"
    router.set_request_tokens(
        "long-output", "ingress_to_prefill", 0, "p0")
    router.set_request_tokens("new", "ingress_to_prefill", 0, "p0")
    router.set_request_tokens(
        "short-output", "ingress_to_prefill", 0, "p1")
    assert dict(router.queued_tokens) == {"p0": 0, "p1": 0}


def test_prefill_to_decode_records_source_and_one_visible_decode_token():
    router = LeastQueuedTokensRouter("deployment", ("p0", "d1", "d0"))

    decision = router.route(
        "request",
        "prefill_to_decode",
        ("d1", "d0"),
        1,
        2.0,
        source_engine_id="p0",
        queue_enter_time_s=1.5,
    )

    assert decision.selected_engine_id == "d0"
    assert decision.source_engine_id == "p0"
    assert decision.queue_enter_time_s == 1.5
    assert decision.decision_time_s == 2.0
    assert decision.queue_exit_time_s == 2.0
    assert decision.dispatch_time_s == 2.0
    assert decision.reason == "minimum_visible_queued_tokens_then_engine_id"
    assert dict(router.queued_tokens) == {"d0": 1, "d1": 0, "p0": 0}


@pytest.mark.parametrize(
    "candidates,token_count,message",
    [
        ((), 1, "at least one"),
        (("p0", "p0"), 1, "unique"),
        (("unknown",), 1, "unknown engines"),
        (("p0",), 0, "positive"),
    ],
)
def test_router_rejects_invalid_or_unsupported_decisions(
        candidates, token_count, message):
    router = LeastQueuedTokensRouter("deployment", ("p0",))

    with pytest.raises((TypeError, ValueError), match=message):
        router.route(
            "request", "ingress_to_prefill", candidates,
            token_count, 0.0)


def test_router_rejects_duplicate_assignment_and_unknown_release():
    router = LeastQueuedTokensRouter("deployment", ("p0",))
    router.route("request", "ingress_to_prefill", ("p0",), 1, 0.0)

    with pytest.raises(ValueError, match="already has"):
        router.route("request", "ingress_to_prefill", ("p0",), 1, 0.0)
    with pytest.raises(ValueError, match="assigned to 'p0', not 'other'"):
        router.set_request_tokens(
            "request", "ingress_to_prefill", 0, "other")
    with pytest.raises(ValueError, match="no 'prefill_to_decode'"):
        router.set_request_tokens(
            "request", "prefill_to_decode", 0, "p0")


def test_router_requires_every_reservation_to_be_released():
    router = LeastQueuedTokensRouter("deployment", ("p0",))
    router.route("request", "ingress_to_prefill", ("p0",), 4, 0.0)

    with pytest.raises(RuntimeError, match="not fully released"):
        router.assert_drained()

    router.set_request_tokens("request", "ingress_to_prefill", 0, "p0")
    router.assert_drained()


def test_async_terminal_tail_is_not_second_logical_route_work():
    scheduled = ScheduledRequest(
        "request", "decode", 0, 1, 128, raw_model_output_tokens=1)
    discarded = BackendStep(
        ScheduleStep("pool:tail", "pool", (scheduled,)),
        generated_tokens={"request": 0},
    )
    emitted = BackendStep(
        ScheduleStep("pool:decode", "pool", (scheduled,)),
        generated_tokens={"request": 1},
    )

    assert _is_discarded_async_terminal_step(discarded, scheduled, 2)
    assert not _is_discarded_async_terminal_step(discarded, scheduled, 1)
    assert not _is_discarded_async_terminal_step(emitted, scheduled, 2)


@pytest.mark.parametrize(
    ("depths", "expected_prefill_starts", "expected_decode_starts"),
    [
        ({"p": 1, "d": 3}, (0.0, 1.0, 2.0), (0.0, 0.0, 0.0)),
        ({"p": 3, "d": 2}, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ],
)
def test_multi_pool_uses_independent_pipeline_depth_ledgers(
        depths, expected_prefill_starts, expected_decode_starts):
    backends = {
        "p": _PipelinedBackend("p", "prefill"),
        "d": _PipelinedBackend("d", "decode"),
    }
    router = LeastQueuedTokensRouter("pd", backends)
    requests = RequestStream(tuple(
        RequestSpec(0.0, request_id, 1, 1)
        for request_id in ("p0", "p1", "p2", "d0", "d1", "d2")
    ))

    def route_request(request, now):
        destination = "p" if request.request_id.startswith("p") else "d"
        return router.route(
            request.request_id,
            "ingress_to_prefill",
            (destination,),
            1,
            now,
        )

    def on_completion(pool_name, step, now):
        request_id = step.schedule.requests[0].request_id
        router.set_request_tokens(
            request_id, "ingress_to_prefill", 0, pool_name)
        return ()

    trace = run_multi_pool_schedule(
        backends,
        requests,
        route_request,
        lambda pool_name, step, now: 1.0,
        depths,
        on_completion,
    )

    starts = {pool_name: tuple(
        entry.start_time_s
        for entry in trace.entries
        if entry.backend_step.schedule.pool_name == pool_name
    ) for pool_name in backends}
    assert starts == {
        "p": expected_prefill_starts,
        "d": expected_decode_starts,
    }
    assert all(
        current.end_time_s <= following.end_time_s
        for pool_name in backends
        for current, following in zip(
            [entry for entry in trace.entries
             if entry.backend_step.schedule.pool_name == pool_name],
            [entry for entry in trace.entries
             if entry.backend_step.schedule.pool_name == pool_name][1:],
            strict=False,
        )
    )
    router.assert_drained()


def test_multi_pool_waits_for_request_scoped_transfer_before_decode():
    backends = {
        "p0": _OneStepBackend("p0", "prefill"),
        "d0": _OneStepBackend("d0", "decode"),
    }
    router = LeastQueuedTokensRouter(
        "deployment", backends, RouterConfig("router-cpu", 1.0))
    requests = RequestStream((RequestSpec(0.0, "r0", 2, 2),))

    def route_request(request, now):
        return router.route(
            request.request_id,
            "ingress_to_prefill",
            ("p0",),
            request.input_tokens,
            now,
        )

    def on_completion(pool_name, step, now):
        request_id = step.schedule.requests[0].request_id
        if pool_name == "p0":
            router.set_request_tokens(
                request_id, "ingress_to_prefill", 0, "p0")
            decision = router.route(
                request_id,
                "prefill_to_decode",
                ("d0",),
                1,
                now,
                source_engine_id="p0",
            )
            return (HandoffDispatch(
                decision,
                RequestSpec(now, request_id, 2, 2),
                (HandoffTransfer(
                    f"{request_id}:pd-kv-transfer:p0-to-d0",
                    transfer_duration_s=3.0,
                    transfer_resource_id="shared-link",
                    transfer_link="nvlink",
                    transfer_reserved_resource_ids=(
                        "communication:endpoint:fixture:p0:nvlink",
                        "communication:endpoint:fixture:d0:nvlink",
                        "communication:fabric:shared-link",),
                    transfer_byte_count=16,
                ),),
            ),)
        elif pool_name == "d0":
            router.set_request_tokens(
                request_id, "prefill_to_decode", 0, "d0")
            return ()
        else:
            raise ValueError(f"unknown test pool: {pool_name!r}")

    def step_duration(pool_name, step, now):
        assert now >= 0
        if pool_name == "p0":
            return 2.0
        elif pool_name == "d0":
            return 1.0
        else:
            raise ValueError(f"unknown test pool: {pool_name!r}")

    trace = run_multi_pool_schedule(
        backends,
        requests,
        route_request,
        step_duration,
        {pool_name: 1 for pool_name in backends},
        on_completion,
    )

    assert [(
        entry.backend_step.schedule.pool_name,
        entry.start_time_s,
        entry.end_time_s,
    ) for entry in trace.entries] == [
        ("p0", 1.0, 3.0),
        ("d0", 7.0, 8.0),
    ]
    assert [(
        transfer.start_time_s,
        transfer.end_time_s,
        transfer.byte_count,
    ) for transfer in trace.transfers] == [(4.0, 7.0, 16)]
    assert [decision.stage for decision in trace.route_decisions] == [
        "ingress_to_prefill", "prefill_to_decode",
    ]
    assert [
        decision.dispatch_time_s
        for decision in trace.route_decisions
    ] == [1.0, 4.0]


def test_shared_transfer_resource_serializes_simultaneous_prefill_handoffs():
    backends = {
        "p1": _OneStepBackend("p1", "prefill"),
        "d0": _OneStepBackend("d0", "decode"),
        "p0": _OneStepBackend("p0", "prefill"),
    }
    router = LeastQueuedTokensRouter("deployment", backends)
    requests = RequestStream((
        RequestSpec(0.0, "r0", 2, 2),
        RequestSpec(0.0, "r1", 2, 2),
    ))

    def route_request(request, now):
        return router.route(
            request.request_id,
            "ingress_to_prefill",
            ("p1", "p0"),
            request.input_tokens,
            now,
        )

    def on_completion(pool_name, step, now):
        request_id = step.schedule.requests[0].request_id
        phase = backend_phase(backends[pool_name].config.role)
        if phase == "prefill":
            router.set_request_tokens(
                request_id, "ingress_to_prefill", 0, pool_name)
            decision = router.route(
                request_id,
                "prefill_to_decode",
                ("d0",),
                1,
                now,
                source_engine_id=pool_name,
            )
            return (HandoffDispatch(
                decision,
                RequestSpec(now, request_id, 2, 2),
                (HandoffTransfer(
                    f"{request_id}:pd-kv-transfer:{pool_name}-to-d0",
                    transfer_duration_s=1.0,
                    transfer_resource_id="shared-link",
                    transfer_link="nvlink",
                    transfer_reserved_resource_ids=(
                        f"communication:endpoint:fixture:{pool_name}:nvlink",
                        "communication:endpoint:fixture:d0:nvlink",
                        "communication:fabric:shared-link",),
                    transfer_byte_count=16,
                ),),
            ),)
        elif phase == "decode":
            router.set_request_tokens(
                request_id, "prefill_to_decode", 0, pool_name)
            return ()
        else:
            raise ValueError(f"unknown test phase: {phase!r}")

    trace = run_multi_pool_schedule(
        backends,
        requests,
        route_request,
        lambda pool_name, step, now: (
            2.0 if pool_name.startswith("p") else 1.0),
        {pool_name: 1 for pool_name in backends},
        on_completion,
    )

    assert [decision.selected_engine_id for decision in trace.route_decisions[:2]] == [
        "p0", "p1",
    ]
    assert [(
        transfer.request_ids,
        transfer.start_time_s,
        transfer.end_time_s,
    ) for transfer in trace.transfers] == [
        (("r0",), 2.0, 3.0),
        (("r1",), 3.0, 4.0),
    ]
    decode_entries = [
        entry for entry in trace.entries
        if entry.backend_step.schedule.pool_name == "d0"
    ]
    assert [(entry.start_time_s, entry.end_time_s)
            for entry in decode_entries] == [(3.0, 4.0), (4.0, 5.0)]


def test_distinct_transfer_resources_allow_simultaneous_prefill_handoffs():
    backends = {
        "p1": _OneStepBackend("p1", "prefill"),
        "d1": _OneStepBackend("d1", "decode"),
        "p0": _OneStepBackend("p0", "prefill"),
        "d0": _OneStepBackend("d0", "decode"),
    }
    router = LeastQueuedTokensRouter("deployment", backends)
    requests = RequestStream((
        RequestSpec(0.0, "r0", 2, 2),
        RequestSpec(0.0, "r1", 2, 2),
    ))

    def route_request(request, now):
        return router.route(
            request.request_id,
            "ingress_to_prefill",
            ("p1", "p0"),
            request.input_tokens,
            now,
        )

    def on_completion(pool_name, step, now):
        request_id = step.schedule.requests[0].request_id
        phase = backend_phase(backends[pool_name].config.role)
        if phase == "prefill":
            router.set_request_tokens(
                request_id, "ingress_to_prefill", 0, pool_name)
            decision = router.route(
                request_id,
                "prefill_to_decode",
                ("d1", "d0"),
                1,
                now,
                source_engine_id=pool_name,
            )
            return (HandoffDispatch(
                decision,
                RequestSpec(now, request_id, 2, 2),
                (HandoffTransfer(
                    f"{request_id}:pd-kv-transfer:{pool_name}-to-"
                    f"{decision.selected_engine_id}",
                    transfer_duration_s=1.0,
                    transfer_resource_id=f"{pool_name}-link",
                    transfer_link="nvlink",
                    transfer_reserved_resource_ids=(
                        f"communication:endpoint:fixture:{pool_name}:nvlink",
                        f"communication:endpoint:fixture:"
                        f"{decision.selected_engine_id}:nvlink",
                        f"communication:fabric:{pool_name}-link",),
                    transfer_byte_count=16,
                ),),
            ),)
        elif phase == "decode":
            router.set_request_tokens(
                request_id, "prefill_to_decode", 0, pool_name)
            return ()
        else:
            raise ValueError(f"unknown test phase: {phase!r}")

    trace = run_multi_pool_schedule(
        backends,
        requests,
        route_request,
        lambda pool_name, step, now: (
            2.0 if pool_name.startswith("p") else 1.0),
        {pool_name: 1 for pool_name in backends},
        on_completion,
    )

    assert [(
        transfer.request_ids,
        transfer.resource_id,
        transfer.start_time_s,
        transfer.end_time_s,
    ) for transfer in trace.transfers] == [
        (("r0",), "p0-link", 2.0, 3.0),
        (("r1",), "p1-link", 2.0, 3.0),
    ]
    decode_entries = [
        entry for entry in trace.entries
        if entry.backend_step.schedule.pool_name.startswith("d")
    ]
    assert [(entry.start_time_s, entry.end_time_s)
            for entry in decode_entries] == [(3.0, 4.0), (3.0, 4.0)]


def test_same_time_completion_updates_load_before_new_arrival_routes():
    backends = {
        "p1": _OneStepBackend("p1", "prefill"),
        "p0": _OneStepBackend("p0", "prefill"),
    }
    router = LeastQueuedTokensRouter("deployment", backends)
    requests = RequestStream((
        RequestSpec(0.0, "r0", 1, 1),
        RequestSpec(2.0, "r1", 1, 1),
    ))

    def route_request(request, now):
        return router.route(
            request.request_id,
            "ingress_to_prefill",
            ("p1", "p0"),
            request.input_tokens,
            now,
        )

    def on_completion(pool_name, step, now):
        request_id = step.schedule.requests[0].request_id
        router.set_request_tokens(
            request_id, "ingress_to_prefill", 0, pool_name)
        return ()

    trace = run_multi_pool_schedule(
        backends,
        requests,
        route_request,
        lambda pool_name, step, now: 2.0,
        {pool_name: 1 for pool_name in backends},
        on_completion,
    )

    assert [decision.selected_engine_id for decision in trace.route_decisions] == [
        "p0", "p0",
    ]


def test_same_time_prefill_completions_reserve_router_before_new_arrival():
    backends = {
        "p1": _OneStepBackend("p1", "prefill"),
        "d1": _OneStepBackend("d1", "decode"),
        "p0": _OneStepBackend("p0", "prefill"),
        "d0": _OneStepBackend("d0", "decode"),
    }
    router = LeastQueuedTokensRouter(
        "deployment", backends, RouterConfig("router-cpu", 1.0))
    requests = RequestStream((
        RequestSpec(0.0, "r0", 2, 2),
        RequestSpec(0.0, "r1", 2, 2),
        RequestSpec(3.0, "r2", 1, 1),
    ))

    def route_request(request, now):
        return router.route(
            request.request_id,
            "ingress_to_prefill",
            ("p1", "p0"),
            request.input_tokens,
            now,
        )

    def on_completion(pool_name, step, now):
        request_id = step.schedule.requests[0].request_id
        phase = backend_phase(backends[pool_name].config.role)
        if phase == "prefill":
            router.set_request_tokens(
                request_id, "ingress_to_prefill", 0, pool_name)
            if request_id == "r2":
                return ()
            decision = router.route(
                request_id,
                "prefill_to_decode",
                ("d1", "d0"),
                1,
                now,
                source_engine_id=pool_name,
            )
            return (HandoffDispatch(
                decision,
                RequestSpec(now, request_id, 2, 2),
                (HandoffTransfer(
                    f"{request_id}:pd-kv-transfer:{pool_name}-to-"
                    f"{decision.selected_engine_id}",
                    transfer_duration_s=1.0,
                    transfer_resource_id=f"{pool_name}-link",
                    transfer_link="nvlink",
                    transfer_reserved_resource_ids=(
                        f"communication:endpoint:fixture:{pool_name}:nvlink",
                        f"communication:endpoint:fixture:"
                        f"{decision.selected_engine_id}:nvlink",
                        f"communication:fabric:{pool_name}-link",),
                    transfer_byte_count=16,
                ),),
            ),)
        elif phase == "decode":
            router.set_request_tokens(
                request_id, "prefill_to_decode", 0, pool_name)
            return ()
        else:
            raise ValueError(f"unknown test phase: {phase!r}")

    def step_duration(pool_name, step, now):
        request_id = step.schedule.requests[0].request_id
        if pool_name.startswith("d"):
            return 1.0
        elif request_id == "r0":
            return 2.0
        elif request_id == "r1":
            return 1.0
        elif request_id == "r2":
            return 1.0
        else:
            raise ValueError(f"unknown test request: {request_id!r}")

    trace = run_multi_pool_schedule(
        backends,
        requests,
        route_request,
        step_duration,
        {pool_name: 1 for pool_name in backends},
        on_completion,
    )

    assert [(
        decision.request_id,
        decision.stage,
        decision.selected_engine_id,
        decision.queue_exit_time_s,
        decision.dispatch_time_s,
    ) for decision in trace.route_decisions] == [
        ("r0", "ingress_to_prefill", "p0", 0.0, 1.0),
        ("r1", "ingress_to_prefill", "p1", 1.0, 2.0),
        ("r0", "prefill_to_decode", "d0", 3.0, 4.0),
        ("r1", "prefill_to_decode", "d1", 4.0, 5.0),
        ("r2", "ingress_to_prefill", "p0", 5.0, 6.0),
    ]
    assert [(
        transfer.request_ids,
        transfer.start_time_s,
        transfer.end_time_s,
    ) for transfer in trace.transfers] == [
        (("r0",), 4.0, 5.0),
        (("r1",), 5.0, 6.0),
    ]
    entries = {
        (entry.backend_step.schedule.pool_name,
         entry.backend_step.schedule.requests[0].request_id): (
             entry.start_time_s, entry.end_time_s)
        for entry in trace.entries
    }
    assert entries[("d0", "r0")] == (5.0, 6.0)
    assert entries[("d1", "r1")] == (6.0, 7.0)
    assert entries[("p0", "r2")] == (6.0, 7.0)


def test_tail_drain_never_starts_before_the_pool_latest_admission():
    backends = {
        "p": _OneStepBackend("p", "prefill"),
        "d": _OneStepBackend("d", "decode"),
    }
    router = LeastQueuedTokensRouter("pd", backends)
    requests = RequestStream((RequestSpec(0.0, "r0", 2, 2),))

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
                RequestSpec(now, request_id, 2, 2),
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

    def tail_drain(pool_name, backend, start_time_s):
        # The last trace entry is the prefill step ending at 2.0, but the
        # decode pool admitted r0 only at transfer completion (5.0); the
        # drain must not start before that admission.
        assert start_time_s == 5.0
        step = backend.plan_step(start_time_s)
        backend.complete_step(step, start_time_s)
        return ((step, 1.0),)

    trace = run_multi_pool_schedule(
        backends,
        requests,
        route_request,
        lambda pool_name, step, now: 2.0,
        {pool_name: 1 for pool_name in backends},
        on_completion,
        tail_drain=tail_drain,
    )

    assert [(
        entry.backend_step.schedule.pool_name,
        entry.start_time_s,
        entry.end_time_s,
    ) for entry in trace.entries] == [
        ("p", 0.0, 2.0),
        ("d", 5.0, 6.0),
    ]
    assert [(
        transfer.start_time_s,
        transfer.end_time_s,
    ) for transfer in trace.transfers] == [(2.0, 5.0)]
    router.assert_drained()
