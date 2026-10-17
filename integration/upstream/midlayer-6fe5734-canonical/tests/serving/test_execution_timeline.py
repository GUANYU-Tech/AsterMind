from pathlib import Path

import pytest

from src.kernel.kernel_spec import (
    KernelKind,
    KernelPrediction,
    KernelSpec,
    PhysicalTransferConfig,
)
from src.serving.execution import _runtime_graph_mode
from src.serving.schedule.schedule_step import ScheduledRequest, ScheduleStep
from src.serving.workload.graph_padding import GraphExecutionConfig
from src.serving.workload.workload_graph import (
    WorkloadGraph,
    WorkloadKind,
    WorkloadNode,
)
from src.kernel.launch_config import LaunchConfig
from src.serving.kernel_lowering.execution_timeline import (
    ExecutionResourceCalendar,
    build_execution_timeline,
    build_runtime_graph_segments,
)
from src.serving.kernel_lowering.kernel_plan import (
    KernelBlockRepeat,
    KernelCall,
    KernelPlan,
)
from src.serving.host_timing import (
    HostLaunchProfile,
    HostRuntimePrepareProfile,
    RuntimePrepareEvent,
    RuntimeGraphSegmentDescriptor,
    load_host_launch_profile,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_resource_calendar_inserts_into_earliest_non_overlapping_gap():
    calendar = ExecutionResourceCalendar()

    assert calendar.reserve("link", 10.0, 5.0) == (10.0, 15.0)
    assert calendar.reserve("link", 0.0, 4.0) == (0.0, 4.0)
    assert calendar.reserve("link", 2.0, 4.0) == (4.0, 8.0)
    assert calendar.reserve("other-link", 2.0, 4.0) == (2.0, 6.0)
    assert [
        (entry.resource_id, entry.start_time_s, entry.end_time_s)
        for entry in calendar.reservations
    ] == [
        ("link", 0.0, 4.0),
        ("other-link", 2.0, 6.0),
        ("link", 4.0, 8.0),
        ("link", 10.0, 15.0),
    ]


def test_resource_calendar_atomically_finds_gap_shared_by_every_resource():
    calendar = ExecutionResourceCalendar()
    assert calendar.reserve("endpoint-a", 0.0, 2.0) == (0.0, 2.0)
    assert calendar.reserve("endpoint-b", 2.0, 2.0) == (2.0, 4.0)

    assert calendar.reserve_many(
        ("endpoint-a", "endpoint-b"), 0.0, 1.0) == (4.0, 5.0)

    assert sum(
        entry.start_time_s == 4.0 and entry.end_time_s == 5.0
        for entry in calendar.reservations
    ) == 2


def test_resource_calendar_invalid_multi_reservation_does_not_mutate_state():
    calendar = ExecutionResourceCalendar()
    calendar.reserve("endpoint-a", 0.0, 2.0)
    before = calendar.reservations

    with pytest.raises(ValueError, match="unique"):
        calendar.reserve_many(
            ("endpoint-a", "endpoint-a"), 0.0, 1.0)

    assert calendar.reservations == before


@pytest.mark.parametrize(
    "resource,start,duration,exception",
    (
        ("", 0.0, 1.0, ValueError),
        ("link", -1.0, 1.0, ValueError),
        ("link", 0.0, -1.0, ValueError),
        ("link", float("inf"), 1.0, ValueError),
        ("link", 0.0, float("nan"), ValueError),
    ),
)
def test_resource_calendar_rejects_invalid_reservations(
    resource, start, duration, exception,
):
    with pytest.raises(exception):
        ExecutionResourceCalendar().reserve(resource, start, duration)


def _compute_spec():
    return KernelSpec(
        KernelKind.VECTOR,
        "nvidia_a800",
        {"element_count": 1},
        dtype="bf16",
        variant="elementwise",
        launch=LaunchConfig(
            "timeline-fixture",
            (1, 1, 1),
            (1, 1, 1),
            execution_variant="elementwise",
        ),
        attributes={"reads": 1, "writes": 1, "operation": "copy"},
    )


def _all_reduce_spec(resource_id="node0-nvlink", participants=(0, 1)):
    return KernelSpec(
        KernelKind.COMMUNICATION,
        "nvidia_a800",
        transfer=PhysicalTransferConfig(
            algorithm="ring",
            bytes=128,
            direction="collective",
            link="nvlink",
            topology="ring",
            participants=participants,
            operation="all_reduce",
            bytes_semantics="per_rank_input",
            ring_order=participants,
            reduction="sum",
            backend="analytical_roofline_v1",
            resource_id=resource_id,
        ),
    )


def _prediction(seconds):
    return KernelPrediction(
        seconds,
        {"fixture": seconds},
        0.0,
        "fixture",
        {"fixture": seconds},
    )


def _host_profile():
    return HostLaunchProfile(
        hardware_id="nvidia_a800",
        host_cpu="fixture-cpu",
        operating_system="fixture-os",
        driver_version="fixture-driver",
        runtime_name="cuda",
        runtime_version="fixture-runtime",
        framework_name="pytorch",
        framework_version="fixture-framework",
        eager_enqueue_s=2.0,
        graph_replay_enqueue_s=2.0,
        measurement_artifact="fixture.json",
        measurement_sha256="0" * 64,
        source_commit="0" * 40,
        source_artifact="fixture.py",
        source_sha256="0" * 64,
        device_launch_gaps=None,
    )


@pytest.mark.parametrize(
    "hardware_id",
    ("nvidia_a800", "metax_c500", "ascend_910b"),
)
def test_host_launch_profiles_use_runtime_mode_schema(hardware_id):
    profile = load_host_launch_profile(
        REPOSITORY_ROOT
        / "configs/host_timing/host_launch_profiles"
        / f"{hardware_id}.json"
    )

    assert profile.hardware_id == hardware_id
    assert profile.manifest()["schema_version"] == 2
    assert "phase_launch_kind" not in profile.manifest()


def test_host_launch_pipeline_schedules_eager_dispatches_before_device_calls():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))
    timeline = build_execution_timeline(
        plan,
        {"first": _prediction(10.0), "second": _prediction(10.0)},
        host_launch_profiles={
            "nvidia_a800": _host_profile(),
        },
        runtime_graph_mode="NONE",
    )

    assert timeline.predicted_device_time_s == 20.0
    assert timeline.launch_cpu_time_s == 4.0
    assert timeline.launch_critical_path_s == 4.0
    assert timeline.host_device_overlap_s == 2.0
    assert timeline.host_exposed_critical_path_s == 2.0
    assert timeline.duration_s == 22.0
    assert timeline.eager_dispatch_count == 2
    assert timeline.graph_replay_count == 0
    assert timeline.runtime_graph_mode == "NONE"
    assert tuple(event.event_kind for event in timeline.host_intervals) == (
        "eager_submit", "eager_submit"
    )
    assert sum(
        event.exposed_time_s for event in timeline.host_intervals
    ) == 2.0
    assert sum(
        event.overlapped_time_s for event in timeline.host_intervals
    ) == 2.0
    assert all(
        event.runtime_graph_mode == "NONE"
        for event in timeline.host_intervals
    )
    first, second = timeline.host_intervals
    assert first.call_occurrence_index == -1
    assert first.device_predecessor_occurrences == ()
    assert first.host_predecessor_event_id is None
    assert second.call_occurrence_index == -1
    assert second.device_predecessor_occurrences == (("first", -1),)
    assert second.host_predecessor_event_id == first.event_id


def test_runtime_prepare_is_inserted_before_its_eager_submit():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))
    timeline = build_execution_timeline(
        plan,
        {"first": _prediction(10.0), "second": _prediction(10.0)},
        host_launch_profiles={"nvidia_a800": _host_profile()},
        host_runtime_prepare_profiles={
            "nvidia_a800": HostRuntimePrepareProfile(
                "nvidia_a800", (("synthetic_prepare", 3.0),)),
        },
        runtime_prepare_events={
            ("main", 0, "second", -1): (
                RuntimePrepareEvent("synthetic_prepare", count=2),
            ),
        },
        runtime_graph_mode="NONE",
    )

    assert timeline.duration_s == 22.0
    assert timeline.launch_cpu_time_s == 10.0
    assert timeline.launch_critical_path_s == 10.0
    assert timeline.host_exposed_critical_path_s == 2.0
    assert [event.event_kind for event in timeline.host_intervals] == [
        "eager_submit", "runtime_prepare", "eager_submit",
    ]
    assert timeline.host_intervals[1].call_id == "second"
    assert timeline.host_intervals[1].mechanism == "synthetic_prepare"
    assert timeline.host_intervals[1].event_count == 2
    assert timeline.host_intervals[1].end_time_s == 8.0


def test_runtime_prepare_uses_exact_lane_service_without_generic_fallback():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("rank-zero", compute, "main", 0, "compute", ("rank-zero",)),
        KernelCall("rank-one", compute, "main", 1, "compute", ("rank-one",)),
    ))
    timeline = build_execution_timeline(
        plan,
        {"rank-zero": _prediction(1.0), "rank-one": _prediction(1.0)},
        host_launch_profiles={"nvidia_a800": _host_profile()},
        host_runtime_prepare_profiles={
            "nvidia_a800": HostRuntimePrepareProfile(
                "nvidia_a800",
                (("synthetic_prepare", 99.0),),
                (("main", 0, "synthetic_prepare", 3.0),
                 ("main", 1, "synthetic_prepare", 7.0)),
            ),
        },
        runtime_prepare_events={
            ("main", 0, "rank-zero", -1): (
                RuntimePrepareEvent("synthetic_prepare"),),
            ("main", 1, "rank-one", -1): (
                RuntimePrepareEvent("synthetic_prepare"),),
        },
        runtime_graph_mode="NONE",
    )

    prepares = [
        event for event in timeline.host_intervals
        if event.event_kind == "runtime_prepare"
    ]
    assert [(event.rank, event.end_time_s - event.start_time_s)
            for event in prepares] == [(0, 3.0), (1, 7.0)]


def test_runtime_prepare_rejects_missing_lane_specific_service():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("rank-one", compute, "main", 1, "compute", ("rank-one",)),
    ))

    with pytest.raises(ValueError, match="does not cover Host lane"):
        build_execution_timeline(
            plan,
            {"rank-one": _prediction(1.0)},
            host_launch_profiles={"nvidia_a800": _host_profile()},
            host_runtime_prepare_profiles={
                "nvidia_a800": HostRuntimePrepareProfile(
                    "nvidia_a800",
                    (("synthetic_prepare", 3.0),),
                    (("main", 0, "synthetic_prepare", 3.0),),
                ),
            },
            runtime_prepare_events={
                ("main", 1, "rank-one", -1): (
                    RuntimePrepareEvent("synthetic_prepare"),),
            },
            runtime_graph_mode="NONE",
        )


def test_piecewise_charges_runtime_prepare_only_on_eager_occurrences():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))

    timeline = build_execution_timeline(
        plan,
        {"first": _prediction(10.0), "second": _prediction(10.0)},
        host_launch_profiles={"nvidia_a800": _host_profile()},
        host_runtime_prepare_profiles={
            "nvidia_a800": HostRuntimePrepareProfile(
                "nvidia_a800", (("synthetic_prepare", 3.0),)),
        },
        runtime_prepare_events={
            ("main", 0, "first", -1): (
                RuntimePrepareEvent("synthetic_prepare"),),
            ("main", 0, "second", -1): (
                RuntimePrepareEvent("synthetic_prepare"),),
        },
        runtime_graph_mode="PIECEWISE",
        runtime_graph_segments=(RuntimeGraphSegmentDescriptor(
            "captured-prefix", "main", 0, (("first", -1),)),),
    )

    assert timeline.duration_s == 22.0
    assert timeline.launch_cpu_time_s == 7.0
    assert timeline.eager_dispatch_count == 1
    assert timeline.graph_replay_count == 1
    assert timeline.launch_kinds == ("eager", "graph_replay", "runtime_prepare")
    assert [event.event_kind for event in timeline.host_intervals] == [
        "graph_replay", "runtime_prepare", "eager_submit",
    ]
    prepare = timeline.host_intervals[1]
    assert prepare.call_id == "second"
    assert prepare.mechanism == "synthetic_prepare"
    assert prepare.start_time_s == 2.0
    assert prepare.end_time_s == 5.0


def test_full_graph_replay_discards_captured_runtime_prepare_events():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))

    timeline = build_execution_timeline(
        plan,
        {"first": _prediction(10.0), "second": _prediction(10.0)},
        host_launch_profiles={"nvidia_a800": _host_profile()},
        host_runtime_prepare_profiles={
            "nvidia_a800": HostRuntimePrepareProfile(
                "nvidia_a800", (("synthetic_prepare", 3.0),)),
        },
        runtime_prepare_events={
            ("main", 0, "first", -1): (
                RuntimePrepareEvent("synthetic_prepare"),),
            ("main", 0, "second", -1): (
                RuntimePrepareEvent("synthetic_prepare"),),
        },
        runtime_graph_mode="FULL",
        runtime_graph_segments=(RuntimeGraphSegmentDescriptor(
            "whole-step", "main", 0, (("first", -1), ("second", -1))),),
    )

    assert timeline.launch_cpu_time_s == 2.0
    assert timeline.eager_dispatch_count == 0
    assert timeline.graph_replay_count == 1
    assert timeline.launch_kinds == ("graph_replay",)
    assert [event.event_kind for event in timeline.host_intervals] == [
        "graph_replay",
    ]


def test_full_graph_replay_keeps_graph_external_runtime_prepare_events():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))

    timeline = build_execution_timeline(
        plan,
        {"first": _prediction(10.0), "second": _prediction(10.0)},
        host_launch_profiles={"nvidia_a800": _host_profile()},
        host_runtime_prepare_profiles={
            "nvidia_a800": HostRuntimePrepareProfile(
                "nvidia_a800", (("synthetic_prepare", 3.0),)),
        },
        runtime_prepare_events={
            ("main", 0, "first", -1): (
                RuntimePrepareEvent(
                    "synthetic_prepare", graph_capturable=False),),
            ("main", 0, "second", -1): (
                RuntimePrepareEvent(
                    "synthetic_prepare", graph_capturable=False),),
        },
        runtime_graph_mode="FULL",
        runtime_graph_segments=(RuntimeGraphSegmentDescriptor(
            "whole-step", "main", 0, (("first", -1), ("second", -1))),),
    )

    assert timeline.graph_replay_count == 1
    assert timeline.launch_cpu_time_s == 8.0
    assert [event.event_kind for event in timeline.host_intervals] == [
        "runtime_prepare", "graph_replay", "runtime_prepare",
    ]


def test_piecewise_runtime_prepare_summary_matches_details():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))
    arguments = {
        "plan": plan,
        "predictions": {"first": _prediction(10.0), "second": _prediction(10.0)},
        "host_launch_profiles": {"nvidia_a800": _host_profile()},
        "host_runtime_prepare_profiles": {
            "nvidia_a800": HostRuntimePrepareProfile(
                "nvidia_a800", (("synthetic_prepare", 3.0),)),
        },
        "runtime_prepare_events": {
            ("main", 0, "first", -1): (
                RuntimePrepareEvent("synthetic_prepare"),),
            ("main", 0, "second", -1): (
                RuntimePrepareEvent("synthetic_prepare"),),
        },
        "runtime_graph_mode": "PIECEWISE",
        "runtime_graph_segments": (RuntimeGraphSegmentDescriptor(
            "captured-prefix", "main", 0, (("first", -1),)),),
    }

    detailed = build_execution_timeline(
        **arguments, materialize_timeline_details=True)
    summary = build_execution_timeline(
        **arguments, materialize_timeline_details=False)

    assert summary.duration_s == detailed.duration_s
    assert summary.launch_cpu_time_s == detailed.launch_cpu_time_s
    assert summary.launch_critical_path_s == detailed.launch_critical_path_s
    assert summary.host_exposed_critical_path_s == (
        detailed.host_exposed_critical_path_s)
    assert summary.launch_kinds == detailed.launch_kinds


def test_runtime_graph_mode_overrides_decode_phase_for_eager_fallback():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))

    timeline = build_execution_timeline(
        plan,
        {"first": _prediction(10.0), "second": _prediction(10.0)},
        host_launch_profiles={
            "nvidia_a800": _host_profile(),
        },
        runtime_graph_mode="NONE",
    )

    assert timeline.launch_cpu_time_s == 4.0
    assert timeline.eager_dispatch_count == 2
    assert timeline.graph_replay_count == 0


def test_host_timing_summary_matches_detailed_timeline():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))
    arguments = {
        "plan": plan,
        "predictions": {
            "first": _prediction(10.0),
            "second": _prediction(10.0),
        },
        "host_launch_profiles": {
            "nvidia_a800": _host_profile(),
        },
        "runtime_graph_mode": "NONE",
    }

    detailed = build_execution_timeline(
        **arguments, materialize_timeline_details=True)
    summary = build_execution_timeline(
        **arguments, materialize_timeline_details=False)

    for field in (
            "duration_s", "predicted_device_time_s", "launch_cpu_time_s",
            "launch_critical_path_s", "host_device_overlap_s",
            "eager_dispatch_count", "graph_replay_count", "launch_kinds",
            "call_occurrence_counts"):
        assert getattr(summary, field) == getattr(detailed, field)
    assert detailed.host_intervals
    assert summary.host_intervals == ()


def test_runtime_graph_mode_uses_actual_capture_fallback():
    schedule = ScheduleStep(
        "target",
        "main",
        (
            ScheduledRequest("prefill", "prefill", 723, 0),
            ScheduledRequest("decode", "decode", 0, 24),
        ),
    )

    assert _runtime_graph_mode(
        schedule,
        GraphExecutionConfig(
            "FULL_AND_PIECEWISE", (512,), ("flash_attn", "kv_update")),
    ) == "NONE"


@pytest.mark.parametrize("runtime_graph_mode", ("FULL", "PIECEWISE"))
def test_graph_host_timing_fails_without_segment_descriptor(
        runtime_graph_mode):
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("only", compute, "main", 0, "compute", ("only",)),
    ))

    with pytest.raises(
            NotImplementedError,
            match=(
                rf"{runtime_graph_mode} Host timing requires an explicit "
                "runtime Graph segment descriptor")):
        build_execution_timeline(
            plan,
            {"only": _prediction(1.0)},
            host_launch_profiles={
                "nvidia_a800": _host_profile(),
            },
            runtime_graph_mode=runtime_graph_mode,
        )


def test_full_graph_replay_uses_explicit_segment_coverage():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))
    segment = RuntimeGraphSegmentDescriptor(
        "decoder", "main", 0, (("first", -1), ("second", -1)))

    detailed = build_execution_timeline(
        plan,
        {"first": _prediction(10.0), "second": _prediction(10.0)},
        host_launch_profiles={"nvidia_a800": _host_profile()},
        runtime_graph_mode="FULL",
        runtime_graph_segments=(segment,),
    )
    summary = build_execution_timeline(
        plan,
        {"first": _prediction(10.0), "second": _prediction(10.0)},
        host_launch_profiles={"nvidia_a800": _host_profile()},
        runtime_graph_mode="FULL",
        runtime_graph_segments=(segment,),
        materialize_timeline_details=False,
    )

    assert detailed.duration_s == summary.duration_s == 22.0
    assert detailed.launch_cpu_time_s == summary.launch_cpu_time_s == 2.0
    assert detailed.eager_dispatch_count == 0
    assert detailed.graph_replay_count == 1
    assert detailed.launch_kinds == ("graph_replay",)
    assert detailed.runtime_graph_mode == summary.runtime_graph_mode == "FULL"
    assert len(detailed.host_intervals) == 1
    event = detailed.host_intervals[0]
    assert event.graph_segment_id == "decoder"
    assert event.covered_call_occurrences == (
        ("first", -1), ("second", -1))


def test_full_graph_segment_producer_covers_every_host_lane():
    compute = _compute_spec()
    collective = _all_reduce_spec()
    workload = WorkloadGraph((
        WorkloadNode(
            "compute", WorkloadKind.LINEAR, "main", {"rows": 1}, rank=0),
        WorkloadNode(
            "collective", WorkloadKind.COLLECTIVE, "main", {"bytes": 128},
            rank=0),
    ))
    plan = KernelPlan((
        KernelCall(
            "compute", compute, "main", 0, "compute", ("compute",)),
        KernelCall(
            "collective", collective, "main", 0, "node0-nvlink",
            ("collective",), ("compute",)),
    ))

    segments = build_runtime_graph_segments(workload, plan, "FULL", ())

    assert tuple(
        (segment.rank, segment.call_occurrences) for segment in segments
    ) == (
        (0, (("compute", -1), ("collective", -1))),
        (1, (("collective", -1),)),
    )
    timeline = build_execution_timeline(
        plan,
        {"compute": _prediction(1.0), "collective": _prediction(1.0)},
        host_launch_profiles={"nvidia_a800": _host_profile()},
        runtime_graph_mode="FULL",
        runtime_graph_segments=segments,
    )
    assert timeline.graph_replay_count == 2
    assert timeline.eager_dispatch_count == 0


def test_piecewise_graph_segment_producer_expands_repeats_at_split_ops():
    compute = _compute_spec()
    workload = WorkloadGraph((
        WorkloadNode(
            "seed", WorkloadKind.LINEAR, "main", {"rows": 1}, rank=0),
        WorkloadNode(
            "before", WorkloadKind.LINEAR, "main", {"rows": 1}, rank=0),
        WorkloadNode(
            "attention", WorkloadKind.FLASH_ATTN, "main", {"tokens": 1},
            rank=0),
        WorkloadNode(
            "after", WorkloadKind.NORM, "main", {"rows": 1}, rank=0),
    ))
    plan = KernelPlan(
        (
            KernelCall(
                "seed", compute, "main", 0, "compute", ("seed",)),
            KernelCall(
                "before", compute, "main", 0, "compute", ("before",),
                ("seed",)),
            KernelCall(
                "attention", compute, "main", 0, "compute",
                ("attention",), ("before",)),
            KernelCall(
                "after", compute, "main", 0, "compute", ("after",),
                ("attention",)),
        ),
        (KernelBlockRepeat(
            "decoder",
            ("before", "attention", "after"),
            2,
            (("before", ("after",)),),
        ),),
    )

    segments = build_runtime_graph_segments(
        workload,
        plan,
        "PIECEWISE",
        ("flash_attn", "kv_update"),
    )

    assert tuple(
        segment.call_occurrences for segment in segments
    ) == (
        (("seed", -1), ("before", 0)),
        (("after", 0), ("before", 1)),
        (("after", 1),),
    )
    timeline = build_execution_timeline(
        plan,
        {call.call_id: _prediction(1.0) for call in plan.calls},
        host_launch_profiles={"nvidia_a800": _host_profile()},
        runtime_graph_mode="PIECEWISE",
        runtime_graph_segments=segments,
    )
    assert timeline.graph_replay_count == 3
    assert timeline.eager_dispatch_count == 2


def test_piecewise_graph_interleaves_explicit_replay_and_eager_submit():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))

    timeline = build_execution_timeline(
        plan,
        {"first": _prediction(10.0), "second": _prediction(10.0)},
        host_launch_profiles={"nvidia_a800": _host_profile()},
        runtime_graph_mode="PIECEWISE",
        runtime_graph_segments=(RuntimeGraphSegmentDescriptor(
            "prefix", "main", 0, (("first", -1),)),),
    )

    assert timeline.duration_s == 22.0
    assert timeline.launch_cpu_time_s == 4.0
    assert timeline.eager_dispatch_count == 1
    assert timeline.graph_replay_count == 1
    assert timeline.launch_kinds == ("eager", "graph_replay")
    assert timeline.runtime_graph_mode == "PIECEWISE"
    assert tuple(
        event.event_kind for event in timeline.host_intervals
    ) == ("graph_replay", "eager_submit")
    assert timeline.host_intervals[1].host_predecessor_event_id == (
        timeline.host_intervals[0].event_id)


def test_piecewise_repeated_collective_summary_matches_multi_rank_details():
    transfer = _all_reduce_spec("piecewise-repeat-fabric")
    plan = KernelPlan(
        (
            KernelCall(
                "seed", _compute_spec(), "main", 0, "compute", ("seed",)),
            KernelCall(
                "transfer", transfer, "main", 0,
                "piecewise-repeat-fabric", ("transfer",), ("seed",)),
        ),
        (KernelBlockRepeat(
            "piecewise-repeat",
            ("transfer",),
            3,
            (("transfer", ("transfer",)),),
        ),),
    )
    segments = tuple(
        RuntimeGraphSegmentDescriptor(
            "captured-prefix",
            "main",
            rank,
            (("transfer", 0), ("transfer", 1)),
        )
        for rank in (0, 1)
    )
    arguments = {
        "plan": plan,
        "predictions": {
            "seed": _prediction(0.5),
            "transfer": _prediction(1.0),
        },
        "host_launch_profiles": {"nvidia_a800": _host_profile()},
        "runtime_graph_mode": "PIECEWISE",
        "runtime_graph_segments": segments,
    }

    detailed = build_execution_timeline(
        **arguments, materialize_timeline_details=True)
    summary = build_execution_timeline(
        **arguments, materialize_timeline_details=False)

    for field in (
            "duration_s", "predicted_device_time_s", "launch_cpu_time_s",
            "launch_critical_path_s", "host_device_overlap_s",
            "eager_dispatch_count", "graph_replay_count", "launch_kinds",
            "call_occurrence_counts"):
        assert getattr(summary, field) == getattr(detailed, field)
    assert detailed.graph_replay_count == 2
    assert detailed.eager_dispatch_count == 3
    assert tuple(
        (event.event_kind, event.rank, event.call_occurrence_index)
        for event in detailed.host_intervals
    ) == (
        ("eager_submit", 0, -1),
        ("graph_replay", 0, 0),
        ("graph_replay", 1, 0),
        ("eager_submit", 0, 2),
        ("eager_submit", 1, 2),
    )
    assert summary.host_intervals == ()


def test_full_graph_descriptor_fails_when_one_occurrence_is_uncovered():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))

    with pytest.raises(ValueError, match="must cover every call occurrence"):
        build_execution_timeline(
            plan,
            {"first": _prediction(10.0), "second": _prediction(10.0)},
            host_launch_profiles={"nvidia_a800": _host_profile()},
            runtime_graph_mode="FULL",
            runtime_graph_segments=(RuntimeGraphSegmentDescriptor(
                "incomplete", "main", 0, (("first", -1),)),),
        )


def test_piecewise_graph_descriptor_rejects_noncontiguous_members():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "middle", compute, "main", 0, "compute", ("middle",),
            ("first",)),
        KernelCall(
            "last", compute, "main", 0, "compute", ("last",),
            ("middle",)),
    ))

    with pytest.raises(ValueError, match="must be contiguous"):
        build_execution_timeline(
            plan,
            {
                "first": _prediction(1.0),
                "middle": _prediction(1.0),
                "last": _prediction(1.0),
            },
            host_launch_profiles={"nvidia_a800": _host_profile()},
            runtime_graph_mode="PIECEWISE",
            runtime_graph_segments=(RuntimeGraphSegmentDescriptor(
                "split", "main", 0,
                (("first", -1), ("last", -1))),),
        )


def test_host_launch_critical_path_replaces_overtaken_device_prefix():
    compute = _compute_spec()
    plan = KernelPlan((
        KernelCall("first", compute, "main", 0, "compute", ("first",)),
        KernelCall(
            "second", compute, "main", 0, "compute", ("second",),
            ("first",)),
    ))

    timeline = build_execution_timeline(
        plan,
        {"first": _prediction(1.0), "second": _prediction(1.0)},
        host_launch_profiles={
            "nvidia_a800": _host_profile(),
        },
        runtime_graph_mode="NONE",
    )

    assert timeline.duration_s == 5.0
    assert dict(timeline.critical_path_contributions) == {
        "host:launch": 4.0,
        "kernel:fixture": 1.0,
    }


def test_kernel_plan_topological_order_preserves_insertion_order_by_level():
    compute = _compute_spec()
    calls = (
        KernelCall("root-a", compute, "main", 0, "compute", ("node-a",)),
        KernelCall(
            "level-one-a", compute, "main", 0, "compute", ("node-b",),
            ("root-a",)),
        KernelCall("root-b", compute, "main", 0, "compute", ("node-c",)),
        KernelCall(
            "level-one-b", compute, "main", 0, "compute", ("node-d",),
            ("root-b",)),
        KernelCall(
            "level-one-c", compute, "main", 0, "compute", ("node-e",),
            ("root-a", "root-b")),
        KernelCall(
            "level-two", compute, "main", 0, "compute", ("node-f",),
            ("level-one-a", "level-one-b")),
    )

    plan = KernelPlan(calls)

    assert tuple(call.call_id for call in plan.topological_order()) == (
        "root-a",
        "root-b",
        "level-one-a",
        "level-one-b",
        "level-one-c",
        "level-two",
    )


def test_collective_waits_for_all_producers_and_releases_all_consumers():
    compute = _compute_spec()
    collective = _all_reduce_spec()
    plan = KernelPlan((
        KernelCall("producer-0", compute, "main", 0, "compute", ("p0",)),
        KernelCall("producer-1", compute, "main", 1, "compute", ("p1",)),
        KernelCall(
            "all-reduce",
            collective,
            "main",
            0,
            "node0-nvlink",
            ("collective",),
            ("producer-0", "producer-1"),
        ),
        KernelCall(
            "consumer-0", compute, "main", 0, "compute", ("c0",),
            ("all-reduce",),
        ),
        KernelCall(
            "consumer-1", compute, "main", 1, "compute", ("c1",),
            ("all-reduce",),
        ),
    ))
    predictions = {
        "producer-0": _prediction(2.0),
        "producer-1": _prediction(3.0),
        "all-reduce": _prediction(5.0),
        "consumer-0": _prediction(1.0),
        "consumer-1": _prediction(1.0),
    }

    timeline = build_execution_timeline(plan, predictions)
    intervals = {
        interval.call_id: interval for interval in timeline.intervals}

    assert intervals["producer-0"].start_time_s == 0.0
    assert intervals["producer-1"].start_time_s == 0.0
    assert intervals["all-reduce"].start_time_s == 3.0
    assert intervals["all-reduce"].end_time_s == 8.0
    assert intervals["all-reduce"].reserved_resource_ids == (
        "communication:endpoint:nvidia_a800:0:nvlink",
        "communication:endpoint:nvidia_a800:1:nvlink",
        "communication:fabric:node0-nvlink",
    )
    assert intervals["consumer-0"].start_time_s == 8.0
    assert intervals["consumer-1"].start_time_s == 8.0
    assert timeline.duration_s == 9.0


def test_collective_host_submit_runs_once_on_each_participant_lane():
    collective = _all_reduce_spec()
    plan = KernelPlan((
        KernelCall(
            "all-reduce",
            collective,
            "main",
            0,
            "node0-nvlink",
            ("collective",),
        ),
    ))

    timeline = build_execution_timeline(
        plan,
        {"all-reduce": _prediction(5.0)},
        host_launch_profiles={
            "nvidia_a800": _host_profile(),
        },
        runtime_graph_mode="NONE",
    )

    assert timeline.duration_s == 7.0
    assert timeline.predicted_device_time_s == 5.0
    assert timeline.launch_cpu_time_s == 4.0
    assert timeline.launch_critical_path_s == 2.0
    assert timeline.eager_dispatch_count == 2
    assert tuple(
        (event.rank, event.exposed_time_s)
        for event in timeline.host_intervals
    ) == ((0, 2.0), (1, 2.0))


def test_collectives_sharing_one_participant_endpoint_do_not_overlap():
    first = _all_reduce_spec("fabric-a")
    second = _all_reduce_spec("fabric-b")
    plan = KernelPlan((
        KernelCall(
            "first", first, "main", 0, "fabric-a", ("first-node",)),
        KernelCall(
            "second", second, "main", 0, "fabric-b", ("second-node",)),
    ))
    predictions = {
        "first": _prediction(2.0),
        "second": _prediction(3.0),
    }

    timeline = build_execution_timeline(plan, predictions)
    intervals = {
        interval.call_id: interval for interval in timeline.intervals}

    assert intervals["first"].start_time_s == 0.0
    assert intervals["first"].end_time_s == 2.0
    assert intervals["second"].start_time_s == 2.0
    assert intervals["second"].end_time_s == 5.0


def test_shared_calendar_serializes_distinct_fabrics_at_shared_endpoint():
    calendar = ExecutionResourceCalendar()
    first_plan = KernelPlan((
        KernelCall(
            "first",
            _all_reduce_spec("fabric-a", (0, 1)),
            "main",
            0,
            "fabric-a",
            ("first-node",),
        ),
    ))
    second_plan = KernelPlan((
        KernelCall(
            "second",
            _all_reduce_spec("fabric-b", (1, 2)),
            "main",
            1,
            "fabric-b",
            ("second-node",),
        ),
    ))

    first = build_execution_timeline(
        first_plan,
        {"first": _prediction(2.0)},
        shared_resource_calendar=calendar,
    )
    second = build_execution_timeline(
        second_plan,
        {"second": _prediction(3.0)},
        shared_resource_calendar=calendar,
    )

    assert first.intervals[0].start_time_s == 0.0
    assert first.intervals[0].end_time_s == 2.0
    assert second.intervals[0].start_time_s == 2.0
    assert second.intervals[0].end_time_s == 5.0


def test_shared_calendar_allows_disjoint_endpoints_and_fabrics_to_overlap():
    calendar = ExecutionResourceCalendar()
    first_plan = KernelPlan((
        KernelCall(
            "first",
            _all_reduce_spec("fabric-a", (0, 1)),
            "main-a",
            0,
            "fabric-a",
            ("first-node",),
        ),
    ))
    second_plan = KernelPlan((
        KernelCall(
            "second",
            _all_reduce_spec("fabric-b", (2, 3)),
            "main-b",
            2,
            "fabric-b",
            ("second-node",),
        ),
    ))

    first = build_execution_timeline(
        first_plan,
        {"first": _prediction(2.0)},
        shared_resource_calendar=calendar,
    )
    second = build_execution_timeline(
        second_plan,
        {"second": _prediction(3.0)},
        shared_resource_calendar=calendar,
    )

    assert first.intervals[0].start_time_s == 0.0
    assert second.intervals[0].start_time_s == 0.0


def test_compact_repeat_uses_the_same_future_calendar_gap_as_expanded_calls():
    transfer = _all_reduce_spec("repeat-fabric")
    compact_plan = KernelPlan(
        (
            KernelCall(
                "seed", _compute_spec(), "main", 0, "compute", ("seed",)),
            KernelCall(
                "transfer", transfer, "main", 0, "repeat-fabric",
                ("transfer",), ("seed",)),
        ),
        (KernelBlockRepeat(
            "repeat-block",
            ("transfer",),
            3,
            (("transfer", ("transfer",)),),
        ),),
    )
    expanded_plan = KernelPlan((
        KernelCall(
            "seed", _compute_spec(), "main", 0, "compute", ("seed",)),
        KernelCall(
            "transfer-0", transfer, "main", 0, "repeat-fabric",
            ("transfer-0",), ("seed",)),
        KernelCall(
            "transfer-1", transfer, "main", 0, "repeat-fabric",
            ("transfer-1",), ("transfer-0",)),
        KernelCall(
            "transfer-2", transfer, "main", 0, "repeat-fabric",
            ("transfer-2",), ("transfer-1",)),
    ))
    reserved_resources = (
        "communication:endpoint:nvidia_a800:0:nvlink",
        "communication:endpoint:nvidia_a800:1:nvlink",
        "communication:fabric:repeat-fabric",
    )
    compact_calendar = ExecutionResourceCalendar()
    expanded_calendar = ExecutionResourceCalendar()
    for calendar in (compact_calendar, expanded_calendar):
        assert calendar.reserve_many(
            reserved_resources, 1.75, 1.0) == (1.75, 2.75)

    compact = build_execution_timeline(
        compact_plan,
        {"seed": _prediction(0.5), "transfer": _prediction(1.0)},
        shared_resource_calendar=compact_calendar,
    )
    expanded = build_execution_timeline(
        expanded_plan,
        {
            "seed": _prediction(0.5),
            "transfer-0": _prediction(1.0),
            "transfer-1": _prediction(1.0),
            "transfer-2": _prediction(1.0),
        },
        shared_resource_calendar=expanded_calendar,
    )

    assert compact.duration_s == expanded.duration_s == 4.75
    assert dict(compact.block_repeats[0].call_intervals)["transfer"] == (
        (0.5, 1.5),
        (2.75, 3.75),
        (3.75, 4.75),
    )
    assert compact.reserved_resource_activity == (
        expanded.reserved_resource_activity)


def test_repeated_host_events_retain_exact_device_and_lane_predecessors():
    transfer = _all_reduce_spec("repeat-fabric")
    plan = KernelPlan(
        (
            KernelCall(
                "seed", _compute_spec(), "main", 0, "compute", ("seed",)),
            KernelCall(
                "transfer", transfer, "main", 0, "repeat-fabric",
                ("transfer",), ("seed",)),
        ),
        (KernelBlockRepeat(
            "repeat-block",
            ("transfer",),
            3,
            (("transfer", ("transfer",)),),
        ),),
    )

    timeline = build_execution_timeline(
        plan,
        {"seed": _prediction(0.5), "transfer": _prediction(1.0)},
        host_launch_profiles={"nvidia_a800": _host_profile()},
        runtime_graph_mode="NONE",
    )

    rank_zero = tuple(
        event for event in timeline.host_intervals
        if event.call_id == "transfer" and event.rank == 0
    )
    rank_one = tuple(
        event for event in timeline.host_intervals
        if event.call_id == "transfer" and event.rank == 1
    )
    assert tuple(
        event.call_occurrence_index for event in rank_zero
    ) == (0, 1, 2)
    assert tuple(
        event.device_predecessor_occurrences for event in rank_zero
    ) == (
        (("seed", -1),),
        (("transfer", 0),),
        (("transfer", 1),),
    )
    assert rank_zero[0].host_predecessor_event_id == (
        timeline.host_intervals[0].event_id)
    assert tuple(
        event.host_predecessor_event_id for event in rank_zero[1:]
    ) == (rank_zero[0].event_id, rank_zero[1].event_id)
    assert rank_one[0].host_predecessor_event_id is None
    assert tuple(
        event.host_predecessor_event_id for event in rank_one[1:]
    ) == (rank_one[0].event_id, rank_one[1].event_id)
