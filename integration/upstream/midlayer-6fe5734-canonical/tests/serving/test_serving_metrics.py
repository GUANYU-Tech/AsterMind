import pytest

from src.serving.metrics.serving_metrics import (
    compute_serving_metrics,
    engine_clock_first_schedule_start_s,
)
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduledRequest,
    ScheduleStep,
)
from src.serving.schedule.schedule_trace import (
    ScheduleTrace,
    ScheduleTraceEntry,
)


def _trace_entry(
    step_id,
    pool_name,
    request_id,
    phase,
    input_tokens,
    generated_tokens,
    start_time_s,
    end_time_s,
    *,
    finished,
):
    if phase == "prefill":
        scheduled_output_tokens = 0
    elif phase == "decode":
        scheduled_output_tokens = 1
    else:
        raise ValueError(f"unknown test phase: {phase!r}")
    scheduled = ScheduledRequest(
        request_id,
        phase,
        input_tokens,
        scheduled_output_tokens,
        raw_model_output_tokens=1,
    )
    backend_step = BackendStep(
        ScheduleStep(
            step_id,
            pool_name,
            (scheduled,),
            finished_request_ids=(request_id,) if finished else (),
        ),
        generated_tokens={request_id: generated_tokens},
    )
    return ScheduleTraceEntry(backend_step, start_time_s, end_time_s)


def test_serving_metrics_derive_latency_and_throughput_from_unified_trace():
    requests = RequestStream((
        RequestSpec(1.0, "multi", 4, 3),
        RequestSpec(0.5, "single", 2, 1),
        RequestSpec(0.0, "zero", 3, 0),
    ))
    trace = ScheduleTrace((
        _trace_entry(
            "zero-prefill", "main", "zero", "prefill", 3, 0,
            0.0, 0.5, finished=True),
        _trace_entry(
            "single-prefill", "main", "single", "prefill", 2, 1,
            0.5, 2.0, finished=True),
        _trace_entry(
            "multi-prefill", "main", "multi", "prefill", 4, 1,
            1.0, 2.0, finished=False),
        _trace_entry(
            "multi-decode-1", "main", "multi", "decode", 0, 1,
            2.0, 3.0, finished=False),
        _trace_entry(
            "multi-decode-2", "main", "multi", "decode", 0, 1,
            3.0, 5.0, finished=True),
    ))

    metrics = compute_serving_metrics(requests, trace)

    assert metrics.requests["multi"].ttft_s == pytest.approx(1.0)
    assert metrics.requests["multi"].tpot_s == pytest.approx(1.5)
    assert metrics.requests["multi"].latency_s == pytest.approx(4.0)
    assert metrics.requests["single"].ttft_s == pytest.approx(1.5)
    assert metrics.requests["single"].tpot_s == 0.0
    assert metrics.requests["single"].latency_s == pytest.approx(1.5)
    assert metrics.requests["zero"].ttft_s is None
    assert metrics.requests["zero"].tpot_s is None
    assert metrics.requests["zero"].latency_s == pytest.approx(0.5)
    assert metrics.request_throughput_per_s == pytest.approx(3 / 5)
    assert metrics.output_token_throughput_per_s == pytest.approx(4 / 5)
    assert metrics.token_throughput_per_s == pytest.approx(13 / 5)


def test_serving_metrics_use_last_pd_finish_as_global_completion():
    requests = RequestStream((RequestSpec(0.0, "request", 4, 2),))
    trace = ScheduleTrace((
        _trace_entry(
            "prefill", "prefill", "request", "prefill", 4, 1,
            0.0, 1.0, finished=True),
        _trace_entry(
            "decode", "decode", "request", "decode", 0, 1,
            2.0, 3.0, finished=True),
    ))

    metrics = compute_serving_metrics(requests, trace)

    # The decode pool is the terminal pool: its first token defines TTFT, so
    # the post-handoff slot wait (1.0 -> 2.0) is TTFT, not an inter-token
    # gap.  The decode pool produced a single token, hence TPOT is zero.
    assert metrics.requests["request"].ttft_s == pytest.approx(3.0)
    assert metrics.requests["request"].tpot_s == 0.0
    assert metrics.requests["request"].latency_s == pytest.approx(3.0)


def test_serving_metrics_pd_tpot_uses_terminal_pool_production_window():
    requests = RequestStream((RequestSpec(0.0, "request", 4, 4),))
    trace = ScheduleTrace((
        _trace_entry(
            "prefill", "prefill", "request", "prefill", 4, 1,
            0.0, 1.0, finished=True),
        _trace_entry(
            "decode-1", "decode", "request", "decode", 0, 1,
            3.0, 4.0, finished=False),
        _trace_entry(
            "decode-2", "decode", "request", "decode", 0, 1,
            4.0, 5.0, finished=False),
        _trace_entry(
            "decode-3", "decode", "request", "decode", 0, 1,
            5.0, 7.0, finished=True),
    ))

    metrics = compute_serving_metrics(requests, trace)

    # Three decode-pool tokens span [4.0, 7.0]; the pre-decode slot wait
    # (1.0 -> 3.0) belongs to TTFT.
    assert metrics.requests["request"].ttft_s == pytest.approx(4.0)
    assert metrics.requests["request"].tpot_s == pytest.approx(1.5)
    assert metrics.requests["request"].latency_s == pytest.approx(7.0)


def _multi_request_entry(step_id, entries, start_time_s, end_time_s,
                         *, finished=()):
    """One step carrying several requests, each with its own token count."""

    scheduled = tuple(
        ScheduledRequest(
            request_id,
            phase,
            input_tokens,
            0 if phase == "prefill" else 1,
            raw_model_output_tokens=1,
        )
        for request_id, phase, input_tokens, _ in entries
    )
    backend_step = BackendStep(
        ScheduleStep(step_id, "main", scheduled,
                     finished_request_ids=tuple(finished)),
        generated_tokens={
            request_id: generated for request_id, _, _, generated in entries
        },
    )
    return ScheduleTraceEntry(backend_step, start_time_s, end_time_s)


def test_first_token_delivery_is_charged_per_output_of_its_own_step():
    # Both requests prefill in a step that carries two outputs, so each pays
    # the delivery path once for itself and once for the other output.
    requests = RequestStream((
        RequestSpec(0.0, "a", 4, 1),
        RequestSpec(0.0, "b", 4, 1),
    ))
    trace = ScheduleTrace((
        _multi_request_entry(
            "shared-prefill",
            (("a", "prefill", 4, 1), ("b", "prefill", 4, 1)),
            0.0, 1.0, finished=("a", "b")),
    ))
    charged = []

    def delivery_of(step_output_count):
        charged.append(step_output_count)
        return 0.1 * step_output_count

    metrics = compute_serving_metrics(requests, trace, delivery_of)
    assert charged == [2, 2]
    for request_id in ("a", "b"):
        assert metrics.requests[request_id].ttft_s == pytest.approx(1.2)


def test_a_step_output_count_ignores_requests_that_produced_no_token():
    requests = RequestStream((
        RequestSpec(0.0, "a", 4, 1),
        RequestSpec(0.0, "b", 4, 1),
    ))
    trace = ScheduleTrace((
        # "b" is scheduled here but its prefill chunk produces no token yet.
        _multi_request_entry(
            "chunk", (("a", "prefill", 4, 1), ("b", "prefill", 4, 0)),
            0.0, 1.0, finished=("a",)),
        _multi_request_entry(
            "tail", (("b", "prefill", 4, 1),), 1.0, 2.0, finished=("b",)),
    ))
    charged = []

    def delivery_of(step_output_count):
        charged.append(step_output_count)
        return 0.0

    compute_serving_metrics(requests, trace, delivery_of)
    assert charged == [1, 1]


def test_a_plain_delivery_number_keeps_the_step_fan_out_out_of_the_model():
    requests = RequestStream((RequestSpec(0.0, "a", 4, 1),))
    trace = ScheduleTrace((
        _multi_request_entry(
            "prefill", (("a", "prefill", 4, 1),), 0.0, 1.0, finished=("a",)),
    ))
    metrics = compute_serving_metrics(requests, trace, 0.25)
    assert metrics.requests["a"].ttft_s == pytest.approx(1.25)


def test_engine_clock_puts_the_first_schedule_at_the_run_origin():
    requests = RequestStream((
        RequestSpec(0.0, "a", 4, 1),
        RequestSpec(1.0, "b", 4, 1),
    ))
    # The simulated run starts late because "a" pays the Host admission path
    # before the Scheduler sees it; the observed trace's origin absorbs the
    # very same path, so on the engine clock "a" waits nothing.
    trace = ScheduleTrace((
        _multi_request_entry(
            "first", (("a", "prefill", 4, 1),), 0.3, 1.0, finished=("a",)),
        _multi_request_entry(
            "second", (("b", "prefill", 4, 1),), 1.4, 2.0, finished=("b",)),
    ))
    starts = engine_clock_first_schedule_start_s(trace, requests)
    assert starts["a"] == pytest.approx(0.0)
    assert starts["b"] == pytest.approx(1.1)


def test_engine_clock_never_reports_a_schedule_before_its_own_arrival():
    requests = RequestStream((
        RequestSpec(0.0, "a", 4, 1),
        RequestSpec(1.0, "b", 4, 1),
    ))
    # "b" is admitted faster than "a" was, so shifting by the run origin alone
    # would place it before it was ever sent.
    trace = ScheduleTrace((
        _multi_request_entry(
            "first", (("a", "prefill", 4, 1),), 0.9, 1.0, finished=("a",)),
        _multi_request_entry(
            "second", (("b", "prefill", 4, 1),), 1.2, 2.0, finished=("b",)),
    ))
    starts = engine_clock_first_schedule_start_s(trace, requests)
    assert starts["b"] == pytest.approx(1.0)


def test_engine_clock_rejects_a_request_missing_from_the_stream():
    requests = RequestStream((RequestSpec(0.0, "a", 4, 1),))
    trace = ScheduleTrace((
        _multi_request_entry(
            "first", (("ghost", "prefill", 4, 1),), 0.0, 1.0),
    ))
    with pytest.raises(ValueError):
        engine_clock_first_schedule_start_s(trace, requests)


def test_engine_clock_origin_is_the_earliest_step_of_an_interleaved_trace():
    requests = RequestStream((
        RequestSpec(0.0, "a", 4, 1),
        RequestSpec(0.0, "b", 4, 1),
    ))
    # A unified two-pool trace need not list its steps in time order.
    trace = ScheduleTrace((
        _multi_request_entry(
            "later", (("b", "prefill", 4, 1),), 0.8, 1.5, finished=("b",)),
        _multi_request_entry(
            "earlier", (("a", "prefill", 4, 1),), 0.3, 1.0, finished=("a",)),
    ))
    starts = engine_clock_first_schedule_start_s(trace, requests)
    assert starts["a"] == pytest.approx(0.0)
    assert starts["b"] == pytest.approx(0.5)
