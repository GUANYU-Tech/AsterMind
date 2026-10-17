import pytest

from evaluation.serving.metrics.scheduler_progress import (
    scheduler_progress_manifest,
    validate_scheduler_progress_manifest,
)
from evaluation.serving.metrics.sharegpt_compact_evaluation import (
    _timing_components,
)
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduledRequest,
    ScheduleStep,
)
from src.serving.schedule.schedule_trace import ScheduleTrace, ScheduleTraceEntry


def _step(
    step_id: str,
    requests: tuple[ScheduledRequest, ...],
    *,
    finished: tuple[str, ...] = (),
    preempted: tuple[str, ...] = (),
    generated_tokens: dict[str, int] | None = None,
    pool: str = "main",
) -> BackendStep:
    if ":" not in step_id:
        step_id = f"default:{step_id}"
    schedule = ScheduleStep(
        step_id,
        pool,
        requests,
        finished_request_ids=finished,
        preempted_request_ids=preempted,
    )
    return BackendStep(
        schedule,
        generated_tokens=(
            {request.request_id: 0 for request in requests}
            if generated_tokens is None
            else generated_tokens
        ),
    )


def test_scheduler_progress_tracks_work_and_cumulative_completion():
    requests = RequestStream((
        RequestSpec(0.0, "request-0", 3, 2),
        RequestSpec(0.0, "request-1", 2, 1),
    ))
    first = _step(
        "step-0",
        (ScheduledRequest("request-0", "prefill", 3, 0),),
    )
    second = _step(
        "step-1",
        (ScheduledRequest("request-0", "decode", 0, 1),),
        finished=("request-0",),
    )
    third = _step(
        "step-2",
        (ScheduledRequest("request-1", "prefill", 2, 0),),
        finished=("request-1",),
    )
    trace = ScheduleTrace((
        ScheduleTraceEntry(first, 0.0, 0.1),
        ScheduleTraceEntry(second, 0.1, 0.2),
        ScheduleTraceEntry(third, 0.2, 0.3),
    ))

    manifest = scheduler_progress_manifest(requests, trace)

    assert manifest["summary"] == {
        "request_count": 2,
        "step_count": 3,
        "scheduled_request_rows": 3,
        "scheduled_tokens": 6,
        "prefill_tokens": 5,
        "decode_tokens": 1,
        "completed_requests": 2,
        "end_time_s": 0.3,
    }
    assert manifest["series"][1]["cumulative_scheduled_tokens"] == 4
    assert manifest["series"][1]["cumulative_completed_requests"] == 1
    assert validate_scheduler_progress_manifest(manifest) == manifest


def test_scheduler_progress_allows_fifo_pipelined_step_intervals():
    requests = RequestStream((RequestSpec(0.0, "request-0", 1, 2),))
    first = _step(
        "step-0",
        (ScheduledRequest("request-0", "prefill", 1, 0),),
    )
    second = _step(
        "step-1",
        (ScheduledRequest("request-0", "decode", 0, 1),),
        finished=("request-0",),
    )
    trace = ScheduleTrace((
        ScheduleTraceEntry(first, 0.0, 2.0),
        ScheduleTraceEntry(second, 0.1, 3.0),
    ))

    assert validate_scheduler_progress_manifest(
        scheduler_progress_manifest(requests, trace))[
            "summary"]["step_count"] == 2


def test_scheduler_progress_rejects_nonconserved_work():
    requests = RequestStream((RequestSpec(0.0, "request-0", 3, 2),))
    step = _step(
        "step-0",
        (ScheduledRequest("request-0", "prefill", 3, 0),),
        finished=("request-0",),
    )

    with pytest.raises(ValueError, match="does not conserve"):
        scheduler_progress_manifest(
            requests,
            ScheduleTrace((ScheduleTraceEntry(step, 0.0, 0.1),)),
        )


def test_scheduler_progress_accounts_for_scheduler_prefix_reuse():
    requests = RequestStream((RequestSpec(0.0, "request-0", 33, 2, 20),))
    first = _step(
        "step-0",
        (ScheduledRequest(
            "request-0",
            "prefill",
            17,
            0,
            computed_tokens=16,
            trace_prefix_hit_tokens=20,
            scheduler_prefix_hit_tokens=16,
        ),),
    )
    second = _step(
        "step-1",
        (ScheduledRequest("request-0", "decode", 0, 1),),
        finished=("request-0",),
    )

    manifest = scheduler_progress_manifest(
        requests,
        ScheduleTrace((
            ScheduleTraceEntry(first, 0.0, 0.1),
            ScheduleTraceEntry(second, 0.1, 0.2),
        )),
    )

    assert manifest["summary"]["scheduled_tokens"] == 18


def test_scheduler_progress_accounts_for_mtp_rejection_overhead():
    requests = RequestStream((RequestSpec(0.0, "request-0", 3, 3),))
    prefill = _step(
        "step-0",
        (ScheduledRequest(
            "request-0",
            "prefill",
            3,
            0,
            raw_model_output_tokens=1,
            mtp_proposer_tokens=2,
            mtp_next_proposed_tokens=2,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="test-mtp-v1",
        ),),
        generated_tokens={"request-0": 1},
    )
    decode = _step(
        "step-1",
        (ScheduledRequest(
            "request-0",
            "decode",
            0,
            3,
            raw_model_output_tokens=2,
            mtp_proposed_tokens=2,
            mtp_verified_tokens=2,
            mtp_accepted_tokens=1,
            mtp_proposer_tokens=2,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="test-mtp-v1",
        ),),
        finished=("request-0",),
        generated_tokens={"request-0": 2},
    )

    manifest = scheduler_progress_manifest(
        requests,
        ScheduleTrace((
            ScheduleTraceEntry(prefill, 0.0, 0.1),
            ScheduleTraceEntry(decode, 0.1, 0.2),
        )),
    )

    assert manifest["summary"]["scheduled_tokens"] == 6
    assert manifest["token_accounting"] == (
        "input_plus_output_minus_terminal_plus_mtp_overhead_v1")
    assert validate_scheduler_progress_manifest(manifest) == manifest


def test_scheduler_progress_mtp_rollback_recompute_is_not_extra_work():
    # MTP rejection rolls num_computed_tokens back, so the next verify step
    # re-schedules the rolled-back position and trips the high-watermark
    # recompute counter.  That position emits its canonical token in this
    # step; the only true extra work is the wasted speculative pass, already
    # counted as rejection overhead.  The recomputed term must NOT be added
    # again on the expected side.
    requests = RequestStream((RequestSpec(0.0, "request-0", 3, 4),))
    prefill = _step(
        "step-0",
        (ScheduledRequest(
            "request-0", "prefill", 3, 0,
            raw_model_output_tokens=1,
            mtp_proposer_tokens=1,
            mtp_next_proposed_tokens=1,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="test-mtp-v1",
        ),),
        generated_tokens={"request-0": 1},
    )
    rejection = _step(
        "step-1",
        (ScheduledRequest(
            "request-0", "decode", 0, 2,
            computed_tokens=4,
            raw_model_output_tokens=1,
            mtp_proposed_tokens=1,
            mtp_verified_tokens=1,
            mtp_accepted_tokens=0,
            mtp_proposer_tokens=1,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="test-mtp-v1",
        ),),
        generated_tokens={"request-0": 1},
    )
    rollback_replay = _step(
        "step-2",
        (ScheduledRequest(
            "request-0", "decode", 0, 2,
            computed_tokens=5,
            recomputed_tokens=1,
            raw_model_output_tokens=2,
            mtp_proposed_tokens=1,
            mtp_verified_tokens=1,
            mtp_accepted_tokens=1,
            mtp_proposer_tokens=1,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="test-mtp-v1",
        ),),
        finished=("request-0",),
        generated_tokens={"request-0": 2},
    )

    manifest = scheduler_progress_manifest(
        requests,
        ScheduleTrace((
            ScheduleTraceEntry(prefill, 0.0, 0.1),
            ScheduleTraceEntry(rejection, 0.1, 0.2),
            ScheduleTraceEntry(rollback_replay, 0.2, 0.3),
        )),
    )

    # scheduled = 3 + 2 + 2 = 7 = canonical(3 + 4 - 1) + rejection overhead 1
    assert manifest["summary"]["scheduled_tokens"] == 7
    assert validate_scheduler_progress_manifest(manifest) == manifest


def test_scheduler_progress_conserves_tokens_across_preemption_recompute():
    # Physically complete preempt->resume trace: a partial prefill computes
    # 2 tokens, the request is preempted, and the resume step replays the
    # full 3-token prefix (2 of them recomputed).  The replay is real extra
    # forward work, so the expected side must add the recomputed term.
    # A second request carries the step that reports the preemption, since a
    # step cannot schedule and preempt the same request.
    requests = RequestStream((
        RequestSpec(0.0, "request-0", 3, 2),
        RequestSpec(0.0, "request-1", 2, 2),
    ))
    partial_prefill = _step(
        "step-0",
        (
            ScheduledRequest("request-0", "prefill", 2, 0),
            ScheduledRequest("request-1", "prefill", 2, 0),
        ),
    )
    preempt = _step(
        "step-1",
        (ScheduledRequest("request-1", "decode", 0, 1),),
        finished=("request-1",),
        preempted=("request-0",),
    )
    resume = _step(
        "step-2",
        (ScheduledRequest(
            "request-0", "prefill", 3, 0, recomputed_tokens=2,
        ),),
    )
    decode = _step(
        "step-3",
        (ScheduledRequest("request-0", "decode", 0, 1),),
        finished=("request-0",),
    )

    manifest = scheduler_progress_manifest(
        requests,
        ScheduleTrace((
            ScheduleTraceEntry(partial_prefill, 0.0, 0.1),
            ScheduleTraceEntry(preempt, 0.1, 0.2),
            ScheduleTraceEntry(resume, 0.2, 0.3),
            ScheduleTraceEntry(decode, 0.3, 0.4),
        )),
    )

    assert manifest["summary"]["scheduled_tokens"] == 9
    assert validate_scheduler_progress_manifest(manifest) == manifest


def test_scheduler_progress_counts_resume_interrupted_by_second_preemption():
    # A request preempted mid-resume has a computed position below its
    # emitted frontier: the first preemption hit after a 4-token partial
    # prefill, the first resume replays 3 tokens, and a second preemption
    # interrupts the resume.  The second resume replays the full 6-token
    # prompt, recomputing 4 positions (the high-water mark).  All 4 are
    # real extra work even though the computed position at the second
    # preemption is only 3, because the replay liability is the emitted
    # frontier, not the interrupted computed position.
    requests = RequestStream((
        RequestSpec(0.0, "request-0", 6, 2),
        RequestSpec(0.0, "request-1", 2, 2),
        RequestSpec(0.0, "request-2", 2, 2),
    ))
    trace = ScheduleTrace((
        ScheduleTraceEntry(_step(
            "step-0",
            (
                ScheduledRequest("request-0", "prefill", 4, 0),
                ScheduledRequest("request-1", "prefill", 2, 0),
            ),
        ), 0.0, 0.1),
        ScheduleTraceEntry(_step(
            "step-1",
            (ScheduledRequest("request-1", "decode", 0, 1),),
            finished=("request-1",),
            preempted=("request-0",),
        ), 0.1, 0.2),
        ScheduleTraceEntry(_step(
            "step-2",
            (ScheduledRequest(
                "request-0", "prefill", 3, 0, recomputed_tokens=3,
            ),),
        ), 0.2, 0.3),
        ScheduleTraceEntry(_step(
            "step-3",
            (ScheduledRequest("request-2", "prefill", 2, 0),),
            preempted=("request-0",),
        ), 0.3, 0.4),
        ScheduleTraceEntry(_step(
            "step-4",
            (ScheduledRequest(
                "request-0", "prefill", 6, 0, recomputed_tokens=4,
            ),),
        ), 0.4, 0.5),
        ScheduleTraceEntry(_step(
            "step-5",
            (ScheduledRequest("request-2", "decode", 0, 1),),
            finished=("request-2",),
        ), 0.5, 0.6),
        ScheduleTraceEntry(_step(
            "step-6",
            (ScheduledRequest("request-0", "decode", 0, 1),),
            finished=("request-0",),
        ), 0.6, 0.7),
    ))

    manifest = scheduler_progress_manifest(requests, trace)

    assert manifest["summary"]["scheduled_tokens"] == 20
    assert validate_scheduler_progress_manifest(manifest) == manifest


def test_scheduler_progress_preemption_replay_tail_is_canonical_refeed():
    # A decode-phase preemption whose high-water mark has reached the
    # emitted frontier: position frontier - 1 was forwarded speculatively
    # and its candidate rejected.  The resume replay's tail re-forward of
    # that position is the canonical forward emitting the frontier token
    # (the role the free MTP verify refeed plays without preemption), so
    # the replay liability is min(frontier, high-water) - 1.  Charging the
    # full recomputed count would double-count one token.
    # Request-0 (input 3, output 4): prefill forwards 0..3 and emits the
    # token at 3; the decode window [3, 5) re-feeds 3, emits the token at
    # 4, and rejects the speculative candidate at 5.  Preemption hits at
    # frontier 5 with high-water 5.  The resume replays [0, 5), emitting
    # the token at 5 from the tail re-forward of 4; only positions 0..3
    # are replay waste.  Per-position accounting for request-0:
    #   0,1,2: prefill canonical + resume replay (waste 3)
    #   3:     prefill spec (overhead) + decode re-feed canonical + replay
    #   4:     decode spec (overhead) + resume tail canonical -- no waste
    #   5:     final window canonical; 6: final window tail (overhead)
    # = canonical 6 + replay 4 + overhead 3 = 13 scheduled.
    requests = RequestStream((
        RequestSpec(0.0, "request-0", 3, 4),
        RequestSpec(0.0, "request-1", 2, 1),
    ))
    prefill = _step(
        "step-0",
        (ScheduledRequest(
            "request-0", "prefill", 3, 1,
            mtp_proposed_tokens=1, mtp_verified_tokens=1,
            mtp_accepted_tokens=0,
            raw_model_output_tokens=1,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="test-mtp-v1",
        ),),
        generated_tokens={"request-0": 1},
    )
    rejection = _step(
        "step-1",
        (ScheduledRequest(
            "request-0", "decode", 0, 2, computed_tokens=3,
            recomputed_tokens=1,
            mtp_proposed_tokens=1, mtp_verified_tokens=1,
            mtp_accepted_tokens=0,
            raw_model_output_tokens=1,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="test-mtp-v1",
        ),),
        generated_tokens={"request-0": 1},
    )
    preempt = _step(
        "step-2",
        (ScheduledRequest(
            "request-1", "prefill", 2, 0,
            raw_model_output_tokens=1,
        ),),
        finished=("request-1",),
        preempted=("request-0",),
        generated_tokens={"request-1": 1},
    )
    resume = _step(
        "step-3",
        (ScheduledRequest(
            "request-0", "prefill", 3, 2, computed_tokens=0,
            recomputed_tokens=5, resumed_from_preemption=True,
            raw_model_output_tokens=1,
            mtp_proposer_tokens=1, mtp_next_proposed_tokens=1,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="test-mtp-v1",
        ),),
        generated_tokens={"request-0": 1},
    )
    finish = _step(
        "step-4",
        (ScheduledRequest(
            "request-0", "decode", 0, 2, computed_tokens=5,
            mtp_proposed_tokens=1, mtp_verified_tokens=1,
            mtp_accepted_tokens=0,
            raw_model_output_tokens=1,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="test-mtp-v1",
        ),),
        finished=("request-0",),
        generated_tokens={"request-0": 1},
    )

    manifest = scheduler_progress_manifest(
        requests,
        ScheduleTrace((
            ScheduleTraceEntry(prefill, 0.0, 0.1),
            ScheduleTraceEntry(rejection, 0.1, 0.2),
            ScheduleTraceEntry(preempt, 0.2, 0.3),
            ScheduleTraceEntry(resume, 0.3, 0.4),
            ScheduleTraceEntry(finish, 0.4, 0.5),
        )),
    )

    # scheduled = 4 + 2 + 2 + 5 + 2 = 15 = canonical(6 + 2) + replay 4
    # + overhead 3
    assert manifest["summary"]["scheduled_tokens"] == 15
    assert validate_scheduler_progress_manifest(manifest) == manifest


def test_scheduler_progress_mtp_rollback_below_emitted_frontier_is_free():
    # MTP verify re-feeds the last emitted position, so a rejection rollback
    # can trip the recompute counter at or below the emitted frontier even
    # when the request was never preempted.  That recompute is part of
    # normal verify mechanics, not extra work: only the rejected draft is
    # overhead.  A preemption-replay liability must not be invented from the
    # position alone.
    requests = RequestStream((RequestSpec(0.0, "request-0", 3, 3),))
    trace = ScheduleTrace((
        ScheduleTraceEntry(_step(
            "step-0",
            (ScheduledRequest(
                "request-0", "prefill", 3, 1,
                mtp_proposed_tokens=1, mtp_verified_tokens=1,
                mtp_accepted_tokens=0,
                raw_model_output_tokens=1,
                mtp_proposer_mode="padded",
                mtp_acceptance_source_kind="explicit_assumption",
                mtp_acceptance_source_id="test-mtp-v1",
            ),),
            generated_tokens={"request-0": 1},
        ), 0.0, 0.1),
        ScheduleTraceEntry(_step(
            "step-1",
            (ScheduledRequest(
                "request-0", "decode", 0, 2, computed_tokens=3,
                recomputed_tokens=1,
                mtp_proposed_tokens=1, mtp_verified_tokens=1,
                mtp_accepted_tokens=1,
                raw_model_output_tokens=2,
                mtp_proposer_mode="padded",
                mtp_acceptance_source_kind="explicit_assumption",
                mtp_acceptance_source_id="test-mtp-v1",
            ),),
            finished=("request-0",),
            generated_tokens={"request-0": 2},
        ), 0.1, 0.2),
    ))

    manifest = scheduler_progress_manifest(requests, trace)

    assert manifest["summary"]["scheduled_tokens"] == 6
    assert validate_scheduler_progress_manifest(manifest) == manifest


def test_scheduler_progress_charges_full_replay_for_stp_pd_decode_preemption():
    # PD decode pool, MTP disabled: the request arrives seeded with the
    # transferred prompt KV (computed = input) and the prefill pool already
    # emitted the first token, so every decode forward advances the
    # high-water mark in lockstep with this pool's emitted frontier.  At the
    # preemption the watermark equals the frontier WITHOUT any speculative
    # rejection, so the resume replay below the watermark is entirely extra
    # work: the canonical tail emission comes from forwarding the position
    # AT the watermark, which lies above the recomputed window.  The
    # MTP-only tail-refeed exclusion must not shave one token off the
    # replay liability here; if it did, one scheduled token per decode-pool
    # preemption would leak out of the conservation check.
    # Request-0 (input 3, output 3): the prefill pool forwards 0..2 and
    # emits the first token; the decode pool forwards position 3 (the
    # transferred first token) and emits the second; preemption hits at
    # frontier 4 with high-water 4; the resume replays 0..3 (recomputed 4,
    # all waste) and forwards position 4, emitting the terminal token.
    # Request-1 (input 2, output 2) carries the preempting step and finishes
    # with its single decode-pool forward.
    requests = RequestStream((
        RequestSpec(0.0, "request-0", 3, 3),
        RequestSpec(0.0, "request-1", 2, 2),
    ))
    trace = ScheduleTrace((
        ScheduleTraceEntry(_step(
            "prefill-0:step-0",
            (
                ScheduledRequest("request-0", "prefill", 3, 0,
                                 raw_model_output_tokens=1),
                ScheduledRequest("request-1", "prefill", 2, 0,
                                 raw_model_output_tokens=1),
            ),
            generated_tokens={"request-0": 1, "request-1": 1},
            pool="prefill-0",
        ), 0.0, 0.1),
        ScheduleTraceEntry(_step(
            "decode-0:step-1",
            (ScheduledRequest(
                "request-0", "decode", 0, 1, computed_tokens=3,
                raw_model_output_tokens=1,
            ),),
            generated_tokens={"request-0": 1},
            pool="decode-0",
        ), 0.1, 0.2),
        ScheduleTraceEntry(_step(
            "decode-0:step-2",
            (ScheduledRequest(
                "request-1", "decode", 0, 1, computed_tokens=2,
                raw_model_output_tokens=1,
            ),),
            finished=("request-1",),
            preempted=("request-0",),
            generated_tokens={"request-1": 1},
            pool="decode-0",
        ), 0.2, 0.3),
        ScheduleTraceEntry(_step(
            "decode-0:step-3",
            (ScheduledRequest(
                "request-0", "prefill", 3, 2, computed_tokens=0,
                recomputed_tokens=4, resumed_from_preemption=True,
                raw_model_output_tokens=1,
            ),),
            finished=("request-0",),
            generated_tokens={"request-0": 1},
            pool="decode-0",
        ), 0.3, 0.4),
    ))

    manifest = scheduler_progress_manifest(requests, trace)

    # scheduled = (3 + 2) + 1 + 1 + 5 = 12
    # = canonical (5 + 3) + preemption replay 4
    assert manifest["summary"]["scheduled_tokens"] == 12
    assert validate_scheduler_progress_manifest(manifest) == manifest


def test_scheduler_progress_rejects_stp_recompute_without_preemption():
    # Without MTP there is no verify rollback, so any recomputed position
    # not covered by an open preemption-replay liability is unexplained
    # scheduled work and must fail closed.
    requests = RequestStream((RequestSpec(0.0, "request-0", 3, 2),))
    trace = ScheduleTrace((
        ScheduleTraceEntry(_step(
            "step-0",
            (ScheduledRequest("request-0", "prefill", 3, 0),),
        ), 0.0, 0.1),
        ScheduleTraceEntry(_step(
            "step-1",
            (ScheduledRequest(
                "request-0", "decode", 0, 2, computed_tokens=2,
                recomputed_tokens=1, raw_model_output_tokens=1,
            ),),
            finished=("request-0",),
            generated_tokens={"request-0": 1},
        ), 0.1, 0.2),
    ))

    with pytest.raises(ValueError, match="does not conserve"):
        scheduler_progress_manifest(requests, trace)


def test_scheduler_progress_accounts_for_discarded_async_decode():
    requests = RequestStream((RequestSpec(0.0, "request-0", 3, 2),))
    prefill = _step(
        "step-0",
        (ScheduledRequest(
            "request-0", "prefill", 3, 0,
            raw_model_output_tokens=1,
        ),),
        generated_tokens={"request-0": 1},
    )
    terminal = _step(
        "step-1",
        (ScheduledRequest(
            "request-0", "decode", 0, 1,
            raw_model_output_tokens=1,
        ),),
        finished=("request-0",),
        generated_tokens={"request-0": 1},
    )
    discarded = _step(
        "step-2",
        (ScheduledRequest(
            "request-0", "decode", 0, 1,
            raw_model_output_tokens=1,
        ),),
        generated_tokens={"request-0": 0},
    )

    manifest = scheduler_progress_manifest(
        requests,
        ScheduleTrace((
            ScheduleTraceEntry(prefill, 0.0, 0.1),
            ScheduleTraceEntry(terminal, 0.1, 0.2),
            ScheduleTraceEntry(discarded, 0.2, 0.3),
        )),
    )

    assert manifest["summary"]["scheduled_tokens"] == 5
    assert manifest["token_accounting"] == (
        "input_plus_output_minus_terminal_plus_async_discard_v1")
    assert validate_scheduler_progress_manifest(manifest) == manifest


def test_scheduler_progress_validation_rejects_tampering():
    requests = RequestStream((RequestSpec(0.0, "request-0", 3, 1),))
    step = _step(
        "step-0",
        (ScheduledRequest("request-0", "prefill", 3, 0),),
        finished=("request-0",),
    )
    manifest = scheduler_progress_manifest(
        requests,
        ScheduleTrace((ScheduleTraceEntry(step, 0.0, 0.1),)),
    )
    manifest["summary"]["scheduled_tokens"] = 4

    with pytest.raises(ValueError, match="summary disagrees"):
        validate_scheduler_progress_manifest(manifest)


def test_timing_components_do_not_add_simulator_wall_time_to_latency():
    components = _timing_components(True, 12.5)

    assert components["device_only"] == {
        "availability": "available",
        "metrics_field": "metrics",
        "time_axis": "predicted_device_only_virtual_time",
    }
    assert components["cpu_plus_device"]["availability"] == "unavailable"
    assert components["cpu_plus_device"]["cpu_source"] is None
    assert components["cpu_plus_device"]["overlap_rule"] is None
    assert components["simulator_wall_time_seconds"] == 12.5
