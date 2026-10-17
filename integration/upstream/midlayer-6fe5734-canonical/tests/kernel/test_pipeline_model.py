import dataclasses

import pytest

import src.kernel.pipeline_model as pipeline_model
from src.kernel.model_contract import (
    AttentionPipelineClass,
    AttentionPipelineEnvelope,
    AttentionPipelinePhase,
    PipelineBuffer,
    PipelinePlan,
    PipelineResource,
    PipelineStage,
    PipelineTask,
    PipelineTaskProgram,
    PipelineTaskSequence,
)
from src.kernel.pipeline_model import (
    attention_round_robin_signatures,
    evaluate_attention_pipeline_envelope,
    evaluate_pipeline,
)


def _task(name, worker, *, load=2.0, compute=3.0, store=1.0):
    return PipelineTask(
        name=name,
        worker=worker,
        stages=(
            PipelineStage("load", "load", load),
            PipelineStage("compute", "compute", compute),
            PipelineStage("store", "store", store),
        ),
    )


def _resources(scope="shared"):
    return tuple(
        PipelineResource(name, scope, 1)
        for name in ("load", "compute", "store")
    )


def test_completion_latency_does_not_hold_resource_past_occupancy():
    plan = PipelinePlan(
        worker_count=1,
        resources=(PipelineResource("pipe", "shared", 1),),
        tasks=(
            PipelineTask("first", 0, stages=(
                PipelineStage(
                    "request", "pipe", 10.0, predecessors=(),
                    occupancy_s=2.0),
            )),
            PipelineTask("second", 0, stages=(
                PipelineStage(
                    "request", "pipe", 10.0, predecessors=(),
                    occupancy_s=2.0),
            )),
        ),
    )

    result = evaluate_pipeline(plan)

    assert result.execution_time_s == pytest.approx(12.0)


def test_pipeline_stage_default_occupancy_preserves_serial_service():
    plan = PipelinePlan(
        worker_count=1,
        resources=(PipelineResource("pipe", "shared", 1),),
        tasks=(
            PipelineTask("first", 0, stages=(
                PipelineStage("request", "pipe", 10.0, predecessors=()),
            )),
            PipelineTask("second", 0, stages=(
                PipelineStage("request", "pipe", 10.0, predecessors=()),
            )),
        ),
    )

    result = evaluate_pipeline(plan)

    assert result.execution_time_s == pytest.approx(20.0)


def test_pipeline_buffer_can_release_at_consumer_issue():
    """A source-issued DMA credit is not a completion-lifetime credit."""

    plan = PipelinePlan(
        worker_count=1,
        resources=(
            PipelineResource("load", "worker", 1),
            PipelineResource("compute", "worker", 1),
            PipelineResource("store", "worker", 1),
        ),
        tasks=(PipelineTask(
            "row", 0, repeat_count=2,
            stages=(
                PipelineStage("load", "load", 1.0, predecessors=()),
                PipelineStage(
                    "compute", "compute", 8.0, predecessors=("load",)),
                PipelineStage(
                    "store", "store", 10.0, predecessors=("compute",)),
            )),),
        buffers=(PipelineBuffer(
            "issued_store", "load", "store", 1, "worker", "start"),),
    )

    result = evaluate_pipeline(plan)

    # Row two may issue its load when row one's MTE3 store is accepted (t=9),
    # while its output still waits for that MTE3 completion (t=19).
    assert result.execution_time_s == pytest.approx(29.0)


def _attention_envelope(class_counts):
    classes = []
    durations = ((2.0, 3.0, 1.0), (1.0, 4.0, 2.0), (3.0, 2.0, 1.5))
    for class_index, count in enumerate(class_counts):
        load, compute, store = durations[class_index]
        classes.append(AttentionPipelineClass(
            name=f"class{class_index}",
            count=count,
            query_rows=64,
            panel_count=class_index + 1,
            phases=(AttentionPipelinePhase(
                name="complete_cta",
                stages=(
                    PipelineStage("begin", "load", load, predecessors=()),
                    PipelineStage(
                        "compute", "compute", compute,
                        predecessors=("begin",)),
                    PipelineStage(
                        "end", "store", store,
                        predecessors=("compute",)),
                ),
            ),),
        ))
    return AttentionPipelineEnvelope(
        active_lane_count=7,
        classes=tuple(classes),
        buffers=(PipelineBuffer(
            "resident_cta", "begin", "end", 2, "worker"),),
    )


def _explicit_attention_envelope(envelope):
    tasks = []
    ordinal = 0
    for work_class in envelope.classes:
        phase, = work_class.phases
        for class_ordinal in range(work_class.count):
            tasks.append(PipelineTask(
                name=f"{work_class.name}_{class_ordinal}",
                worker=ordinal % envelope.active_lane_count,
                stages=phase.stages,
            ))
            ordinal += 1
    return PipelinePlan(
        worker_count=envelope.active_lane_count,
        resources=_resources(scope="worker"),
        tasks=tuple(tasks),
        buffers=envelope.buffers,
    )


def test_attention_envelope_matches_explicit_class_major_round_robin():
    envelope = _attention_envelope((17, 23, 5))

    analytical = evaluate_attention_pipeline_envelope(envelope)
    explicit = evaluate_pipeline(_explicit_attention_envelope(envelope))

    assert analytical.execution_time_s == pytest.approx(
        explicit.execution_time_s, rel=1e-12)
    assert analytical.stage_service_s == pytest.approx(
        explicit.stage_service_s, rel=1e-12)
    assert analytical.tasks == ()


def test_attention_source_ordinal_marker_shifts_round_robin_without_work():
    def work_class(name, duration):
        return AttentionPipelineClass(
            name=name,
            count=1,
            query_rows=64,
            panel_count=1,
            phases=(AttentionPipelinePhase(
                name="complete_cta",
                stages=(
                    PipelineStage(
                        "begin", "load", duration - 1.0,
                        predecessors=()),
                    PipelineStage(
                        "end", "store", 1.0,
                        predecessors=("begin",)),
                ),
            ),),
        )

    ordinal = AttentionPipelineClass(
        name="early_exit", count=1, query_rows=0, panel_count=0,
        phases=())
    envelope = AttentionPipelineEnvelope(
        active_lane_count=2,
        classes=(work_class("long", 10.0), ordinal,
                 work_class("short", 8.0)),
        buffers=(PipelineBuffer(
            "resident_cta", "begin", "end", 2, "worker"),),
    )

    result = evaluate_attention_pipeline_envelope(envelope)
    compressed = evaluate_attention_pipeline_envelope(dataclasses.replace(
        envelope, classes=(envelope.classes[0], envelope.classes[2])))

    assert result.execution_time_s == pytest.approx(17.0)
    assert compressed.execution_time_s == pytest.approx(10.0)
    assert result.stage_service_s == compressed.stage_service_s


def test_attention_source_ordinal_marker_advances_resident_lifecycle():
    def work_class(name):
        return AttentionPipelineClass(
            name=name, count=1, query_rows=64, panel_count=1,
            phases=(AttentionPipelinePhase(
                name="complete_cta",
                stages=(
                    PipelineStage(
                        "begin", "load", 9.0, predecessors=()),
                    PipelineStage(
                        "end", "store", 1.0,
                        predecessors=("begin",)),
                ),
            ),),
        )

    ordinal = AttentionPipelineClass(
        name="early_exit", count=1, query_rows=0, panel_count=0,
        phases=())
    envelope = AttentionPipelineEnvelope(
        active_lane_count=1,
        classes=(work_class("first"), ordinal, work_class("second")),
        buffers=(PipelineBuffer(
            "resident_cta", "begin", "end", 2, "worker"),),
    )

    result = evaluate_attention_pipeline_envelope(envelope)
    compressed = evaluate_attention_pipeline_envelope(dataclasses.replace(
        envelope, classes=(envelope.classes[0], envelope.classes[2])))

    assert result.execution_time_s == pytest.approx(20.0)
    assert compressed.execution_time_s == pytest.approx(19.0)
    assert result.stage_service_s == compressed.stage_service_s


def test_attention_paired_resident_fifo_uses_earliest_free_slot():
    durations = (10.0, 2.0, 3.0, 3.0, 3.0, 3.0)
    classes = tuple(
        AttentionPipelineClass(
            name=f"source{index}",
            count=1,
            query_rows=64,
            panel_count=1,
            phases=(AttentionPipelinePhase(
                name="complete_cta",
                stages=(
                    PipelineStage(
                        "begin", "compute", duration - 1.0,
                        predecessors=()),
                    PipelineStage(
                        "end", "store", 1.0,
                        predecessors=("begin",)),
                ),
            ),),
        )
        for index, duration in enumerate(durations)
    )
    envelope = AttentionPipelineEnvelope(
        active_lane_count=2,
        classes=classes,
        buffers=(PipelineBuffer(
            "resident_cta", "begin", "end", 2, "worker"),),
        cta_admission_policy="paired_resident_fifo",
    )

    result = evaluate_attention_pipeline_envelope(envelope)
    round_robin = evaluate_attention_pipeline_envelope(dataclasses.replace(
        envelope, cta_admission_policy="round_robin"))

    assert result.execution_time_s == pytest.approx(11.0)
    assert round_robin.execution_time_s == pytest.approx(14.0)
    assert result.stage_service_s == round_robin.stage_service_s


def test_attention_bounded_resident_suffix_fifo_preserves_prefix_balance():
    durations = (10.0, 2.0, 3.0, 3.0, 3.0, 3.0)
    classes = tuple(
        AttentionPipelineClass(
            name=f"source{index}", count=1, query_rows=64, panel_count=1,
            phases=(AttentionPipelinePhase(
                name="complete_cta",
                stages=(
                    PipelineStage(
                        "begin", "compute", duration - 1.0,
                        predecessors=()),
                    PipelineStage(
                        "end", "store", 1.0, predecessors=("begin",)),
                ),
            ),),
        )
        for index, duration in enumerate(durations)
    )
    envelope = AttentionPipelineEnvelope(
        active_lane_count=2,
        classes=classes,
        buffers=(PipelineBuffer(
            "resident_cta", "begin", "end", 2, "worker"),),
        cta_admission_policy="bounded_resident_suffix_fifo",
    )

    result = evaluate_attention_pipeline_envelope(envelope)

    assert result.execution_time_s == pytest.approx(12.0)
    assert result.tasks == ()


def _attention_tail_envelope(
        class_counts, panel_counts, tail_latency, *,
        group_interval=0, group_latency=0.0):
    classes = tuple(
        AttentionPipelineClass(
            name=f"class{class_index}",
            count=count,
            query_rows=16,
            panel_count=panel_count,
            phases=(AttentionPipelinePhase(
                name="complete_cta",
                stages=(
                    PipelineStage(
                        "begin", "load", 1.0, predecessors=()),
                    PipelineStage(
                        "end", "compute", 1.0,
                        predecessors=("begin",)),
                ),
            ),),
        )
        for class_index, (count, panel_count) in enumerate(
            zip(class_counts, panel_counts))
    )
    return AttentionPipelineEnvelope(
        active_lane_count=1,
        classes=classes,
        buffers=(PipelineBuffer(
            "resident_cta", "begin", "end", 2, "worker"),),
        resident_tail_latency_s_per_panel=tail_latency,
        resident_tail_group_interval_panels=group_interval,
        resident_tail_latency_s_per_group=group_latency,
    )


def test_attention_resident_tail_charges_all_panels_for_single_cta():
    baseline = evaluate_attention_pipeline_envelope(
        _attention_tail_envelope((1,), (5,), 0.0))
    with_tail = evaluate_attention_pipeline_envelope(
        _attention_tail_envelope((1,), (5,), 2.0))

    assert with_tail.execution_time_s == pytest.approx(
        baseline.execution_time_s + 10.0)


def test_attention_resident_tail_is_zero_for_equal_ctas():
    baseline = evaluate_attention_pipeline_envelope(
        _attention_tail_envelope((2,), (5,), 0.0))
    with_tail = evaluate_attention_pipeline_envelope(
        _attention_tail_envelope((2,), (5,), 2.0))

    assert with_tail.execution_time_s == pytest.approx(
        baseline.execution_time_s)


def test_attention_resident_tail_charges_only_long_short_panel_difference():
    baseline = evaluate_attention_pipeline_envelope(
        _attention_tail_envelope((1, 1), (3, 8), 0.0))
    with_tail = evaluate_attention_pipeline_envelope(
        _attention_tail_envelope((1, 1), (3, 8), 2.0))

    assert with_tail.execution_time_s == pytest.approx(
        baseline.execution_time_s + 10.0)


def test_attention_resident_tail_charges_group_boundaries_exactly():
    baseline = evaluate_attention_pipeline_envelope(
        _attention_tail_envelope((1, 1), (7, 17), 0.0))
    with_tail = evaluate_attention_pipeline_envelope(
        _attention_tail_envelope(
            (1, 1), (7, 17), 0.0,
            group_interval=8, group_latency=3.0))

    assert with_tail.execution_time_s == pytest.approx(
        baseline.execution_time_s + 6.0)


def test_attention_round_robin_signatures_do_not_enumerate_lanes():
    counts = (17, 23, 5)
    lane_count = 7
    signatures = attention_round_robin_signatures(counts, lane_count)

    counts_by_lane = [[0] * len(counts) for _ in range(lane_count)]
    ordinal = 0
    for class_index, count in enumerate(counts):
        for _ in range(count):
            counts_by_lane[ordinal % lane_count][class_index] += 1
            ordinal += 1
    explicit = {}
    for values in counts_by_lane:
        signature = tuple(values)
        explicit[signature] = explicit[signature] + 1 \
            if signature in explicit else 1

    assert dict(signatures) == explicit
    assert len(signatures) <= 2 * len(counts)
    huge = attention_round_robin_signatures(
        (262_144, 131_073, 65_537), 108)
    assert len(huge) <= 6


@pytest.mark.parametrize(("counts", "lane_count"), (
    ((1, 1, 1), 7),
    ((2, 9, 3), 7),
    ((14, 1, 8), 7),
    ((100, 37, 65), 11),
))
def test_attention_round_robin_sparse_signatures_preserve_lane_assignment(
        counts, lane_count):
    signatures = pipeline_model._attention_round_robin_sparse_signatures(
        counts, lane_count)
    actual = {}
    for entries, multiplicity in signatures:
        dense = [0] * len(counts)
        for class_index, count in entries:
            dense[class_index] = count
        key = tuple(dense)
        actual[key] = actual[key] + multiplicity if key in actual \
            else multiplicity

    expected_by_lane = [[0] * len(counts) for _ in range(lane_count)]
    ordinal = 0
    for class_index, count in enumerate(counts):
        for _ in range(count):
            expected_by_lane[ordinal % lane_count][class_index] += 1
            ordinal += 1
    expected = {}
    for dense in expected_by_lane:
        if not any(dense):
            continue
        key = tuple(dense)
        expected[key] = expected[key] + 1 if key in expected else 1

    assert actual == expected


def test_attention_envelope_cost_is_independent_of_cta_count(monkeypatch):
    original_apply = pipeline_model._maxplus_apply
    calls = []

    def counted_apply(matrix, vector):
        calls.append(None)
        return original_apply(matrix, vector)

    monkeypatch.setattr(pipeline_model, "_maxplus_apply", counted_apply)
    evaluate_attention_pipeline_envelope(_attention_envelope((10_000, 1, 1)))
    small_calls = len(calls)
    calls.clear()
    evaluate_attention_pipeline_envelope(
        _attention_envelope((10_000_000, 1, 1)))
    large_calls = len(calls)

    assert large_calls == small_calls


def test_attention_envelope_compiles_shared_source_program_once(monkeypatch):
    envelope = _attention_envelope((17, 23, 5))
    shared_phases = envelope.classes[0].phases
    envelope = dataclasses.replace(
        envelope,
        classes=tuple(
            dataclasses.replace(work_class, phases=shared_phases)
            for work_class in envelope.classes),
    )
    original_compile = pipeline_model._attention_class_transition_matrix
    calls = []

    def counted_compile(*args, **kwargs):
        calls.append(None)
        return original_compile(*args, **kwargs)

    monkeypatch.setattr(
        pipeline_model, "_attention_class_transition_matrix",
        counted_compile)
    pipeline_model._attention_envelope_cache.clear()

    evaluate_attention_pipeline_envelope(envelope)

    assert len(calls) == 1


def test_unified_topology_compiler_reuses_dag_across_attention_and_tasks(
        monkeypatch):
    stages = (
        PipelineStage("load", "load", 2.0, predecessors=()),
        PipelineStage("compute", "compute", 3.0, predecessors=("load",)),
        PipelineStage("store", "store", 1.0, predecessors=("compute",)),
    )
    buffers = (PipelineBuffer("tile", "load", "store", 2, "worker"),)
    signature = pipeline_model._pipeline_stage_topology_signature(stages)
    original_compile = pipeline_model._compile_pipeline_topology
    calls = []

    def counted_compile(*args, **kwargs):
        calls.append(args)
        return original_compile(*args, **kwargs)

    original_compile.cache_clear()
    monkeypatch.setattr(
        pipeline_model, "_compile_pipeline_topology", counted_compile)

    evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTask("tile", 0, stages=stages, repeat_count=2),),
        buffers=buffers,
    ))
    attention = AttentionPipelineEnvelope(
        active_lane_count=1,
        classes=(AttentionPipelineClass(
            name="class0", count=2, query_rows=64, panel_count=1,
            phases=(AttentionPipelinePhase("body", stages),),
        ),),
        buffers=buffers,
    )
    evaluate_attention_pipeline_envelope(attention)

    matching = [
        args for args in calls
        if args[0] == signature
        and args[1] == pipeline_model._pipeline_buffer_topology_signature(
            buffers)
        and args[2] is False
    ]
    # Both solver frontends bind the template; the cached compiler performs
    # the static DAG compilation once and serves the second binding as a hit.
    assert len(matching) == 2
    assert original_compile.cache_info().misses == 1
    original_compile.cache_clear()


def test_unified_topology_binding_rejects_changed_static_dag():
    stages = (
        PipelineStage("load", "load", 2.0, predecessors=()),
        PipelineStage("store", "store", 1.0, predecessors=("load",)),
    )
    topology = pipeline_model._compile_pipeline_topology(
        pipeline_model._pipeline_stage_topology_signature(stages), (), False)

    with pytest.raises(ValueError, match="stage DAG does not match"):
        pipeline_model._bind_pipeline_topology(
            topology,
            (
                PipelineStage("load", "load", 2.0, predecessors=()),
                PipelineStage("store", "store", 1.0, predecessors=()),
            ),
            {"load": 0, "store": 1}, {}, 2)


def test_unified_topology_rebinds_dynamic_durations_without_recompiling():
    first = (
        PipelineStage("load", "load", 2.0, predecessors=()),
        PipelineStage("store", "store", 1.0, predecessors=("load",)),
    )
    second = (
        PipelineStage("load", "load", 5.0, predecessors=()),
        PipelineStage("store", "store", 1.0, predecessors=("load",)),
    )
    resource_indexes = {"load": 0, "store": 1}
    pipeline_model._compile_pipeline_topology.cache_clear()

    first_transition = pipeline_model._compile_pipeline_transition(
        first, resource_indexes, {}, (), 2)
    second_transition = pipeline_model._compile_pipeline_transition(
        second, resource_indexes, {}, (), 2)

    assert pipeline_model._maxplus_apply(
        first_transition, (0.0, 0.0, 0.0)) == pytest.approx((2.0, 3.0, 0.0))
    assert pipeline_model._maxplus_apply(
        second_transition, (0.0, 0.0, 0.0)) == pytest.approx((5.0, 6.0, 0.0))
    assert pipeline_model._compile_pipeline_topology.cache_info().misses == 1
    pipeline_model._compile_pipeline_topology.cache_clear()


def test_unified_topology_reuses_binding_skeleton_for_dynamic_durations():
    first = (
        PipelineStage("load", "load", 2.0, predecessors=()),
        PipelineStage("store", "store", 1.0, predecessors=("load",)),
    )
    second = (
        PipelineStage("load", "load", 5.0, predecessors=()),
        PipelineStage("store", "store", 1.0, predecessors=("load",)),
    )
    resource_indexes = {"load": 0, "store": 1}
    pipeline_model._compile_pipeline_topology.cache_clear()
    pipeline_model._compile_pipeline_binding_skeleton.cache_clear()

    first_transition = pipeline_model._compile_pipeline_transition(
        first, resource_indexes, {}, (), 2)
    second_transition = pipeline_model._compile_pipeline_transition(
        second, resource_indexes, {}, (), 2)

    assert pipeline_model._maxplus_apply(
        first_transition, (0.0, 0.0, 0.0)) == pytest.approx((2.0, 3.0, 0.0))
    assert pipeline_model._maxplus_apply(
        second_transition, (0.0, 0.0, 0.0)) == pytest.approx((5.0, 6.0, 0.0))
    assert pipeline_model._compile_pipeline_binding_skeleton.cache_info().misses == 1
    pipeline_model._compile_pipeline_topology.cache_clear()
    pipeline_model._compile_pipeline_binding_skeleton.cache_clear()


def test_repeated_transition_reuses_maxplus_power_ladder(monkeypatch):
    transition = ((1.0,),)
    product_count = 0
    original_product = pipeline_model._maxplus_product

    def counted_product(left, right):
        nonlocal product_count
        product_count += 1
        return original_product(left, right)

    pipeline_model._maxplus_powers.cache_clear()
    monkeypatch.setattr(pipeline_model, "_maxplus_product", counted_product)
    first = pipeline_model.apply_repeated_maxplus_transition(
        transition, 37, (0.0,))
    first_product_count = product_count
    second = pipeline_model.apply_repeated_maxplus_transition(
        tuple(tuple(value for value in row) for row in transition),
        37,
        (0.0,),
    )

    assert first == (37.0,)
    assert second == first
    assert first_product_count > 0
    assert product_count == first_product_count
    assert pipeline_model._maxplus_powers.cache_info().hits == 1
    pipeline_model._maxplus_powers.cache_clear()


def test_vectorized_maxplus_apply_preserves_dense_and_unreachable_states():
    negative = float("-inf")
    transition = (
        (1.0, negative, 4.0),
        (negative, 2.0, 3.0),
        (negative, negative, negative),
    )

    actual = pipeline_model._maxplus_apply(
        transition, (2.0, negative, 5.0))

    assert actual == (9.0, 8.0, negative)


def test_batched_maxplus_programs_match_independent_recurrences():
    first = ((1.0, 0.0), (negative := float("-inf"), 2.0))
    second = ((0.0, 3.0), (1.0, 0.0))
    programs = (
        ((first, 3), (second, 1)),
        ((second, 2),),
    )
    states = ((0.0, 1.0), (2.0, 0.0))
    expected = []
    for program, state in zip(programs, states, strict=True):
        current = state
        for transition, repeat_count in program:
            current = pipeline_model.apply_repeated_maxplus_transition(
                transition, repeat_count, current)
        expected.append(current)

    actual = pipeline_model.apply_maxplus_transition_programs(
        programs, states)

    assert actual == tuple(expected)


def test_single_tile_is_pipeline_fill_and_drain():
    result = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(),
        tasks=(_task("tile0", 0),),
    ))

    assert result.execution_time_s == 6.0
    assert result.initiation_interval_s == 0.0
    assert result.worker_makespan_s["0"] == 6.0
    assert result.critical_path == (
        "tile0/load", "tile0/compute", "tile0/store")


def test_two_buffer_pipeline_reduces_to_fill_plus_steady_interval():
    result = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(),
        tasks=(_task("tile0", 0), _task("tile1", 0)),
        buffers=(PipelineBuffer(
            "input", "load", "compute", 2, "worker"),),
    ))

    assert result.execution_time_s == 9.0
    assert result.initiation_interval_s == 2.0
    assert result.tasks[1].stage_start_s["load"] == 2.0


def test_one_buffer_prevents_load_compute_overlap():
    result = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(),
        tasks=(_task("tile0", 0), _task("tile1", 0)),
        buffers=(PipelineBuffer(
            "input", "load", "compute", 1, "worker"),),
    ))

    assert result.tasks[1].stage_start_s["load"] == 5.0
    assert result.execution_time_s == 11.0


def test_worker_makespan_owns_partial_tail_wave():
    tasks = tuple(
        _task(f"tile{index}", index % 2, load=1.0, compute=1.0, store=1.0)
        for index in range(5)
    )
    result = evaluate_pipeline(PipelinePlan(
        worker_count=2,
        resources=_resources(scope="worker"),
        tasks=tasks,
    ))

    assert result.worker_makespan_s["0"] == 5.0
    assert result.worker_makespan_s["1"] == 4.0
    assert result.execution_time_s == 5.0


def test_qk_and_pv_serialize_on_one_shared_matrix_resource():
    task = PipelineTask(
        name="kv0",
        worker=0,
        stages=(
            PipelineStage("load", "load", 1.0),
            PipelineStage("qk", "matrix", 2.0),
            PipelineStage("softmax", "vector", 3.0),
            PipelineStage("pv", "matrix", 2.0),
            PipelineStage("store", "store", 1.0),
        ),
    )
    result = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=(
            PipelineResource("load", "worker", 1),
            PipelineResource("matrix", "shared", 1),
            PipelineResource("vector", "worker", 1),
            PipelineResource("store", "worker", 1),
        ),
        tasks=(task,),
    ))

    assert result.execution_time_s == 9.0
    assert result.stage_service_s["qk"] + result.stage_service_s["pv"] == 4.0


def _fork_join_task(name, *, repeat_count=1):
    return PipelineTask(
        name=name,
        worker=0,
        stages=(
            PipelineStage(
                "qk", "matrix", 3.0, predecessors=()),
            PipelineStage(
                "load_v", "load", 5.0, predecessors=()),
            PipelineStage(
                "load_k_next", "load", 7.0,
                predecessors=("qk", "load_v")),
            PipelineStage(
                "softmax", "vector", 2.0,
                predecessors=("qk", "load_v"),
                start_predecessors=("load_k_next",)),
            PipelineStage(
                "pv", "matrix", 4.0, predecessors=("softmax",)),
            PipelineStage(
                "store", "store", 1.0,
                predecessors=("load_k_next", "pv")),
        ),
        repeat_count=repeat_count,
    )


def _fork_join_resources():
    return tuple(
        PipelineResource(name, "worker", 1)
        for name in ("load", "matrix", "vector", "store")
    )


def test_explicit_stage_dag_forks_and_joins_source_services():
    result = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_fork_join_resources(),
        tasks=(_fork_join_task("tile0"),),
    ))

    task = result.tasks[0]
    assert task.stage_start_s["qk"] == 0.0
    assert task.stage_start_s["load_v"] == 0.0
    assert task.stage_start_s["load_k_next"] == 5.0
    assert task.stage_start_s["softmax"] == 5.0
    assert task.stage_start_s["pv"] == 7.0
    assert task.stage_start_s["store"] == 12.0
    assert result.execution_time_s == 13.0


def test_repeated_stage_dag_matches_explicit_tasks():
    compact = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_fork_join_resources(),
        tasks=(_fork_join_task("tiles", repeat_count=3),),
    ))
    explicit = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_fork_join_resources(),
        tasks=tuple(_fork_join_task(f"tile{index}") for index in range(3)),
    ))

    assert compact.execution_time_s == pytest.approx(explicit.execution_time_s)
    assert compact.worker_makespan_s == pytest.approx(
        explicit.worker_makespan_s)
    assert compact.stage_service_s == pytest.approx(explicit.stage_service_s)


def test_stage_dag_rejects_forward_or_unknown_predecessors():
    with pytest.raises(ValueError, match="must name earlier stages"):
        PipelineTask(
            name="invalid",
            worker=0,
            stages=(
                PipelineStage(
                    "load", "load", 1.0, predecessors=("compute",)),
                PipelineStage("compute", "compute", 1.0),
            ),
        )

    with pytest.raises(ValueError, match="must name earlier stages"):
        PipelineTask(
            name="invalid_start",
            worker=0,
            stages=(
                PipelineStage(
                    "load", "load", 1.0,
                    start_predecessors=("compute",)),
                PipelineStage("compute", "compute", 1.0),
            ),
        )


def test_stage_start_dependency_waits_for_delayed_issue_not_completion():
    task = PipelineTask(
        name="issue_order",
        worker=0,
        stages=(
            PipelineStage(
                "occupy_load", "load", 10.0, predecessors=()),
            PipelineStage("qk", "matrix", 3.0, predecessors=()),
            PipelineStage("load_v", "other_load", 5.0, predecessors=()),
            PipelineStage(
                "load_k_next", "load", 7.0,
                predecessors=("qk", "load_v")),
            PipelineStage(
                "softmax", "vector", 2.0,
                predecessors=("qk", "load_v"),
                start_predecessors=("load_k_next",)),
        ),
    )
    resources = tuple(
        PipelineResource(name, "worker", 1)
        for name in ("load", "matrix", "other_load", "vector"))

    result = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=resources,
        tasks=(task,),
    ))
    diagnostic = result.tasks[0]

    assert diagnostic.stage_start_s["load_k_next"] == 10.0
    assert diagnostic.stage_start_s["softmax"] == 10.0
    assert diagnostic.stage_end_s["load_k_next"] == 17.0


@pytest.mark.parametrize("buffer_slots", [1, 2, 4])
@pytest.mark.parametrize("repeat_count", [2, 3, 17, 1000])
def test_repeated_worker_task_matches_explicit_schedule(
        buffer_slots, repeat_count):
    repeated_task = PipelineTask(
        name="tiles",
        worker=0,
        stages=_task("template", 0).stages,
        repeat_count=repeat_count,
    )
    buffers = (PipelineBuffer(
        "input", "load", "compute", buffer_slots, "worker"),)
    repeated = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(repeated_task,),
        buffers=buffers,
    ))
    explicit = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=tuple(_task(f"tile{index}", 0)
                    for index in range(repeat_count)),
        buffers=buffers,
    ))

    assert repeated.execution_time_s == pytest.approx(
        explicit.execution_time_s)
    assert repeated.initiation_interval_s == pytest.approx(
        explicit.initiation_interval_s)
    assert repeated.worker_makespan_s == pytest.approx(
        explicit.worker_makespan_s)
    assert repeated.stage_service_s == pytest.approx(
        explicit.stage_service_s)
    assert repeated.tasks[0].repeat_count == repeat_count


def test_repeated_tasks_preserve_state_across_different_runs():
    first = PipelineTask(
        name="first",
        worker=0,
        stages=_task("template0", 0, load=2.0, compute=3.0, store=1.0).stages,
        repeat_count=7,
    )
    second = PipelineTask(
        name="second",
        worker=0,
        stages=_task("template1", 0, load=1.0, compute=4.0, store=2.0).stages,
        repeat_count=11,
    )
    buffers = (PipelineBuffer(
        "input", "load", "compute", 2, "worker"),)
    repeated = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(first, second),
        buffers=buffers,
    ))
    explicit_tasks = tuple(
        _task(f"first{index}", 0, load=2.0, compute=3.0, store=1.0)
        for index in range(7)) + tuple(
            _task(f"second{index}", 0, load=1.0, compute=4.0, store=2.0)
            for index in range(11))
    explicit = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=explicit_tasks,
        buffers=buffers,
    ))

    assert repeated.execution_time_s == pytest.approx(
        explicit.execution_time_s)
    assert repeated.initiation_interval_s == pytest.approx(
        explicit.initiation_interval_s)
    assert repeated.stage_service_s == pytest.approx(
        explicit.stage_service_s)


def test_repeated_task_rejects_shared_resource_until_order_is_explicit():
    with pytest.raises(ValueError, match="worker-local resources"):
        evaluate_pipeline(PipelinePlan(
            worker_count=1,
            resources=_resources(scope="shared"),
            tasks=(PipelineTask(
                name="tiles",
                worker=0,
                stages=_task("template", 0).stages,
                repeat_count=2,
            ),),
        ))


@pytest.mark.parametrize("outer_repeats", [1, 2, 7, 1000])
def test_repeated_task_sequence_matches_explicit_body_and_final(
        outer_repeats):
    body = PipelineTask(
        name="body",
        worker=0,
        stages=(
            PipelineStage("load", "load", 2.0),
            PipelineStage("compute", "compute", 3.0),
        ),
        repeat_count=4,
    )
    final = _task("final", 0, load=2.0, compute=3.0, store=1.0)
    buffers = (PipelineBuffer(
        "input", "load", "compute", 2, "worker"),)
    compact = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskSequence(
            name="cta_program",
            worker=0,
            tasks=(body, final),
            repeat_count=outer_repeats,
        ),),
        buffers=buffers,
    ))
    explicit_tasks = []
    for outer in range(outer_repeats):
        for inner in range(4):
            explicit_tasks.append(PipelineTask(
                name=f"cta{outer}_body{inner}",
                worker=0,
                stages=body.stages,
            ))
        explicit_tasks.append(_task(f"cta{outer}_final", 0))
    explicit = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=tuple(explicit_tasks),
        buffers=buffers,
    ))

    assert compact.execution_time_s == pytest.approx(
        explicit.execution_time_s)
    assert compact.initiation_interval_s == pytest.approx(
        explicit.initiation_interval_s)
    assert compact.worker_makespan_s == pytest.approx(
        explicit.worker_makespan_s)
    assert compact.stage_service_s == pytest.approx(
        explicit.stage_service_s)


@pytest.mark.parametrize("body_repeats", [1, 2, 8, 64])
def test_repeated_task_carries_named_stage_dependencies(body_repeats):
    prologue = PipelineTask(
        name="prologue",
        worker=0,
        stages=(PipelineStage("ready", "load", 2.0),),
    )
    body = PipelineTask(
        name="body",
        worker=0,
        stages=(
            PipelineStage(
                "compute", "compute", 3.0,
                predecessors=(), prior_resources=("load",)),
            PipelineStage(
                "ready", "load", 2.0, predecessors=("compute",)),
        ),
        repeat_count=body_repeats,
    )
    epilogue = PipelineTask(
        name="epilogue",
        worker=0,
        stages=(PipelineStage(
            "store", "store", 1.0,
            predecessors=(), prior_resources=("compute",)),),
    )
    result = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskSequence(
            name="compact",
            worker=0,
            tasks=(prologue, body, epilogue),
        ),),
    ))

    assert result.execution_time_s == pytest.approx(
        2.0 + (body_repeats - 1) * 5.0 + 3.0 + 1.0)


def test_buffer_lifetime_can_span_one_repeated_task_sequence():
    sequence = PipelineTaskSequence(
        name="cta",
        worker=0,
        tasks=(
            PipelineTask(
                name="prologue", worker=0,
                stages=(PipelineStage("begin", "load", 1.0),)),
            PipelineTask(
                name="body", worker=0,
                stages=(PipelineStage(
                    "compute", "compute", 3.0,
                    predecessors=(), prior_resources=("load",)),)),
            PipelineTask(
                name="epilogue", worker=0,
                stages=(PipelineStage(
                    "end", "store", 1.0,
                    predecessors=(), prior_resources=("compute",)),)),
        ),
        repeat_count=4,
    )
    result = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(sequence,),
        buffers=(PipelineBuffer(
            "resident", "begin", "end", 2, "worker"),),
    ))

    assert result.execution_time_s == pytest.approx(14.0)
    assert result.tasks[-1].stage_end_s["end"] == pytest.approx(14.0)


def test_worker_program_pattern_order_matches_explicit_schedule():
    def pattern(name, body_count):
        body = PipelineTask(
            name="body",
            worker=0,
            stages=(
                PipelineStage("load", "load", 2.0),
                PipelineStage("compute", "compute", 3.0),
            ),
            repeat_count=body_count,
        )
        return PipelineTaskSequence(
            name=name,
            worker=0,
            tasks=(body, _task("final", 0)),
        )

    patterns = (pattern("short", 1), pattern("long", 4))
    order = (0, 1, 0, 0, 1, 1, 0)
    buffers = (PipelineBuffer(
        "input", "load", "compute", 2, "worker"),)
    compact = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskProgram(
            name="worker_program",
            worker=0,
            patterns=patterns,
            order=order,
        ),),
        buffers=buffers,
    ))
    explicit_tasks = []
    for outer, pattern_index in enumerate(order):
        selected = patterns[pattern_index]
        body, final = selected.tasks
        for inner in range(body.repeat_count):
            explicit_tasks.append(PipelineTask(
                name=f"program{outer}_body{inner}",
                worker=0,
                stages=body.stages,
            ))
        explicit_tasks.append(_task(f"program{outer}_final", 0))
    explicit = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=tuple(explicit_tasks),
        buffers=buffers,
    ))

    assert compact.execution_time_s == pytest.approx(
        explicit.execution_time_s)
    assert compact.initiation_interval_s == pytest.approx(
        explicit.initiation_interval_s)
    assert compact.worker_makespan_s == pytest.approx(
        explicit.worker_makespan_s)
    assert compact.stage_service_s == pytest.approx(
        explicit.stage_service_s)


def test_repeated_worker_program_matches_explicit_schedule():
    patterns = tuple(
        PipelineTaskSequence(
            name=f"pattern{pattern_index}",
            worker=0,
            tasks=(_task(
                f"task{pattern_index}", 0,
                load=1.0 + pattern_index,
                compute=2.0 + pattern_index,
                store=0.5 + pattern_index,
            ),),
        )
        for pattern_index in range(3)
    )
    order = (0, 1, 2, 1)
    repeat_count = 37
    tail_order = (2, 0)
    buffers = (PipelineBuffer(
        "input", "load", "compute", 2, "worker"),)
    compact = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskProgram(
            name="worker_program",
            worker=0,
            patterns=patterns,
            order=order,
            repeat_count=repeat_count,
            tail_order=tail_order,
        ),),
        buffers=buffers,
    ))
    explicit = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=tuple(
            _task(
                f"occurrence{repeat_index}_{order_index}", 0,
                load=1.0 + pattern_index,
                compute=2.0 + pattern_index,
                store=0.5 + pattern_index,
            )
            for repeat_index in range(repeat_count)
            for order_index, pattern_index in enumerate(order)
        ) + tuple(
            _task(
                f"tail{order_index}", 0,
                load=1.0 + pattern_index,
                compute=2.0 + pattern_index,
                store=0.5 + pattern_index,
            )
            for order_index, pattern_index in enumerate(tail_order)
        ),
        buffers=buffers,
    ))

    assert compact.execution_time_s == pytest.approx(
        explicit.execution_time_s, rel=1e-12)
    assert compact.initiation_interval_s == pytest.approx(
        explicit.initiation_interval_s, rel=1e-12)
    assert compact.worker_makespan_s == pytest.approx(
        explicit.worker_makespan_s, rel=1e-12)
    assert compact.stage_service_s == pytest.approx(
        explicit.stage_service_s, rel=1e-12)


def test_equivalent_worker_sequences_preserve_source_diagnostics_exactly():
    worker_count = 4
    entries = []
    for worker in range(worker_count):
        first_repeats = 5 if worker < 2 else 4
        entries.append(PipelineTaskSequence(
            name=f"worker{worker}_first",
            worker=worker,
            tasks=(_task(f"first_body_w{worker}", worker),),
            repeat_count=first_repeats,
        ))
        entries.append(PipelineTaskSequence(
            name=f"worker{worker}_last",
            worker=worker,
            tasks=(_task(f"last_body_w{worker}", worker),),
            repeat_count=2,
        ))
    plan = PipelinePlan(
        worker_count=worker_count,
        resources=_resources(scope="worker"),
        tasks=tuple(entries),
        buffers=(PipelineBuffer(
            "input", "load", "compute", 2, "worker"),),
    )

    compact = pipeline_model._equivalent_worker_sequence_plan(plan)
    reference = pipeline_model._evaluate_repeated_pipeline(plan)
    result = evaluate_pipeline(plan)

    assert compact is not None
    assert compact[0].worker_count == 2
    assert result == reference


@pytest.mark.parametrize("tail_order", [(), (4, 2)])
def test_repeated_program_uses_sparse_path_when_matrix_does_not_amortize(
        monkeypatch, tail_order):
    patterns = tuple(
        PipelineTaskSequence(
            name=f"pattern{pattern_index}",
            worker=0,
            tasks=(_task(
                f"task{pattern_index}", 0,
                load=1.0 + pattern_index,
                compute=2.0 + pattern_index,
                store=0.5 + pattern_index,
            ),),
        )
        for pattern_index in range(5)
    )
    order = tuple(index % len(patterns) for index in range(4096))
    buffers = (PipelineBuffer(
        "input", "load", "compute", 2, "worker"),)
    product_count = 0
    original_product = pipeline_model._maxplus_left_sparse_product

    def counted_product(left, right):
        nonlocal product_count
        product_count += 1
        return original_product(left, right)

    monkeypatch.setattr(
        pipeline_model, "_maxplus_left_sparse_product", counted_product)
    compact = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskProgram(
            name="compact",
            worker=0,
            patterns=patterns,
            order=order,
            repeat_count=2,
            tail_order=tail_order,
        ),),
        buffers=buffers,
    ))
    sparse_reference = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskProgram(
            name="expanded_order",
            worker=0,
            patterns=patterns,
            order=order * 2 + tail_order,
        ),),
        buffers=buffers,
    ))

    assert product_count == 0
    assert compact.execution_time_s == pytest.approx(
        sparse_reference.execution_time_s, rel=1e-12)
    assert compact.initiation_interval_s == pytest.approx(
        sparse_reference.initiation_interval_s, rel=1e-12)
    assert compact.worker_makespan_s == pytest.approx(
        sparse_reference.worker_makespan_s, rel=1e-12)
    assert compact.stage_service_s == pytest.approx(
        sparse_reference.stage_service_s, rel=1e-12)


def test_repeated_program_amortizes_matrix_path_across_workers(monkeypatch):
    worker_count = 8
    order = tuple(index % 3 for index in range(128))
    buffers = (PipelineBuffer(
        "input", "load", "compute", 2, "worker"),)

    def program(worker, *, expanded):
        patterns = tuple(
            PipelineTaskSequence(
                name=f"pattern{pattern_index}",
                worker=worker,
                tasks=(_task(
                    f"task{pattern_index}", worker,
                    load=1.0 + pattern_index,
                    compute=2.0 + pattern_index,
                    store=0.5 + pattern_index,
                ),),
            )
            for pattern_index in range(3)
        )
        return PipelineTaskProgram(
            name=f"worker{worker}",
            worker=worker,
            patterns=patterns,
            order=order * 2 if expanded else order,
            repeat_count=1 if expanded else 2,
        )

    product_count = 0
    original_product = pipeline_model._maxplus_left_sparse_product

    def counted_product(left, right):
        nonlocal product_count
        product_count += 1
        return original_product(left, right)

    monkeypatch.setattr(
        pipeline_model, "_maxplus_left_sparse_product", counted_product)
    compact = evaluate_pipeline(PipelinePlan(
        worker_count=worker_count,
        resources=_resources(scope="worker"),
        tasks=tuple(program(worker, expanded=False)
                    for worker in range(worker_count)),
        buffers=buffers,
    ))
    sparse_reference = evaluate_pipeline(PipelinePlan(
        worker_count=worker_count,
        resources=_resources(scope="worker"),
        tasks=tuple(program(worker, expanded=True)
                    for worker in range(worker_count)),
        buffers=buffers,
    ))

    assert product_count > 0
    assert compact.execution_time_s == pytest.approx(
        sparse_reference.execution_time_s, rel=1e-12)
    assert compact.initiation_interval_s == pytest.approx(
        sparse_reference.initiation_interval_s, rel=1e-12)
    assert compact.worker_makespan_s == pytest.approx(
        sparse_reference.worker_makespan_s, rel=1e-12)
    assert compact.stage_service_s == pytest.approx(
        sparse_reference.stage_service_s, rel=1e-12)


def test_repeated_long_single_run_keeps_sparse_power_path(monkeypatch):
    pattern = PipelineTaskSequence(
        name="uniform",
        worker=0,
        tasks=(_task("template", 0),),
    )
    order = (0,) * 4096
    repeat_count = 10
    buffers = (PipelineBuffer(
        "input", "load", "compute", 2, "worker"),)
    product_count = 0
    original_product = pipeline_model._maxplus_left_sparse_product

    def counted_product(left, right):
        nonlocal product_count
        product_count += 1
        return original_product(left, right)

    monkeypatch.setattr(
        pipeline_model, "_maxplus_left_sparse_product", counted_product)
    compact = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskProgram(
            name="compact",
            worker=0,
            patterns=(pattern,),
            order=order,
            repeat_count=repeat_count,
        ),),
        buffers=buffers,
    ))
    sparse_reference = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskProgram(
            name="expanded_order",
            worker=0,
            patterns=(pattern,),
            order=order * repeat_count,
        ),),
        buffers=buffers,
    ))

    assert product_count == 0
    assert compact.execution_time_s == pytest.approx(
        sparse_reference.execution_time_s, rel=1e-12)
    assert compact.initiation_interval_s == pytest.approx(
        sparse_reference.initiation_interval_s, rel=1e-12)
    assert compact.worker_makespan_s == pytest.approx(
        sparse_reference.worker_makespan_s, rel=1e-12)
    assert compact.stage_service_s == pytest.approx(
        sparse_reference.stage_service_s, rel=1e-12)


def test_worker_program_long_pattern_run_uses_logarithmic_state_updates(
        monkeypatch):
    pattern = PipelineTaskSequence(
        name="uniform",
        worker=0,
        tasks=(_task("template", 0),),
    )
    order = (0,) * 2048
    buffers = (PipelineBuffer(
        "input", "load", "compute", 2, "worker"),)
    apply_count = 0
    original_apply = pipeline_model._maxplus_apply

    def counted_apply(matrix, vector):
        nonlocal apply_count
        apply_count += 1
        return original_apply(matrix, vector)

    monkeypatch.setattr(pipeline_model, "_maxplus_apply", counted_apply)
    compact = pipeline_model.evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskProgram(
            name="worker_program",
            worker=0,
            patterns=(pattern,),
            order=order,
        ),),
        buffers=buffers,
    ))
    explicit = pipeline_model.evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=tuple(_task(f"tile{index}", 0) for index in range(len(order))),
        buffers=buffers,
    ))

    assert compact.execution_time_s == pytest.approx(
        explicit.execution_time_s)
    assert compact.initiation_interval_s == pytest.approx(
        explicit.initiation_interval_s)
    assert compact.worker_makespan_s == pytest.approx(
        explicit.worker_makespan_s)
    assert compact.stage_service_s == pytest.approx(
        explicit.stage_service_s)
    assert apply_count < 32


@pytest.mark.parametrize(
    "order",
    [
        (0,),
        (0, 0),
        (0, 0, 0),
        (0, 1, 1, 1, 2),
        (0, 1, 1, 2, 1, 1, 1, 1, 0),
    ],
)
def test_worker_program_multi_run_matches_explicit_schedule(order):
    patterns = tuple(
        PipelineTaskSequence(
            name=f"pattern{pattern_index}",
            worker=0,
            tasks=(_task(
                f"task{pattern_index}", 0,
                load=1.25 + pattern_index,
                compute=2.5 + pattern_index / 2,
                store=0.75 + pattern_index / 4,
            ),),
        )
        for pattern_index in range(3)
    )
    buffers = (
        PipelineBuffer("input", "load", "compute", 2, "worker"),
        PipelineBuffer("output", "compute", "store", 4, "worker"),
    )
    compact = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskProgram(
            name="worker_program",
            worker=0,
            patterns=patterns,
            order=order,
        ),),
        buffers=buffers,
    ))
    explicit = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=tuple(
            _task(
                f"occurrence{occurrence}", 0,
                load=1.25 + pattern_index,
                compute=2.5 + pattern_index / 2,
                store=0.75 + pattern_index / 4,
            )
            for occurrence, pattern_index in enumerate(order)
        ),
        buffers=buffers,
    ))

    assert compact.execution_time_s == pytest.approx(
        explicit.execution_time_s, rel=1e-12)
    assert compact.initiation_interval_s == pytest.approx(
        explicit.initiation_interval_s, rel=1e-12)
    assert compact.worker_makespan_s == pytest.approx(
        explicit.worker_makespan_s, rel=1e-12)
    assert compact.stage_service_s == pytest.approx(
        explicit.stage_service_s, rel=1e-12)


@pytest.mark.parametrize("sequence_repeats", [2, 7])
@pytest.mark.parametrize("position", ["first", "middle", "last"])
def test_worker_program_sequence_repeat_matches_explicit_schedule(
        sequence_repeats, position):
    repeated_index = {"first": 0, "middle": 1, "last": 2}[position]
    patterns = tuple(
        PipelineTaskSequence(
            name=f"pattern{pattern_index}",
            worker=0,
            tasks=(_task(
                f"task{pattern_index}", 0,
                load=1.0 + pattern_index,
                compute=2.0 + pattern_index,
                store=0.5 + pattern_index,
            ),),
            repeat_count=(
                sequence_repeats if pattern_index == repeated_index else 1),
        )
        for pattern_index in range(3)
    )
    order = (0, 1, 2)
    buffers = (PipelineBuffer(
        "input", "load", "compute", 2, "worker"),)
    compact = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskProgram(
            name="worker_program",
            worker=0,
            patterns=patterns,
            order=order,
        ),),
        buffers=buffers,
    ))
    explicit_tasks = []
    for pattern_index in order:
        for repeat in range(patterns[pattern_index].repeat_count):
            explicit_tasks.append(_task(
                f"pattern{pattern_index}_repeat{repeat}", 0,
                load=1.0 + pattern_index,
                compute=2.0 + pattern_index,
                store=0.5 + pattern_index,
            ))
    explicit = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=tuple(explicit_tasks),
        buffers=buffers,
    ))

    assert compact.execution_time_s == pytest.approx(
        explicit.execution_time_s, rel=1e-12)
    assert compact.initiation_interval_s == pytest.approx(
        explicit.initiation_interval_s, rel=1e-12)
    assert compact.worker_makespan_s == pytest.approx(
        explicit.worker_makespan_s, rel=1e-12)
    assert compact.stage_service_s == pytest.approx(
        explicit.stage_service_s, rel=1e-12)
    assert "repeat=" not in compact.critical_path


def test_worker_program_run_of_nested_repeated_sequence_matches_explicit():
    nested = PipelineTaskSequence(
        name="nested",
        worker=0,
        tasks=(
            PipelineTask(
                name="body",
                worker=0,
                stages=_task(
                    "body_template", 0,
                    load=1.0, compute=2.0, store=3.0,
                ).stages,
                repeat_count=2,
            ),
            _task("tail", 0, load=2.0, compute=3.0, store=1.0),
        ),
        repeat_count=2,
    )
    order = (0, 0, 0, 0)
    buffers = (
        PipelineBuffer("input", "load", "compute", 1, "worker"),
        PipelineBuffer("output", "compute", "store", 1, "worker"),
    )
    compact = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=(PipelineTaskProgram(
            name="worker_program",
            worker=0,
            patterns=(nested,),
            order=order,
        ),),
        buffers=buffers,
    ))

    explicit_tasks = []
    for occurrence in range(len(order)):
        for sequence_repeat in range(nested.repeat_count):
            for task_repeat in range(nested.tasks[0].repeat_count):
                explicit_tasks.append(PipelineTask(
                    name=(
                        f"body_{occurrence}_{sequence_repeat}_{task_repeat}"),
                    worker=0,
                    stages=nested.tasks[0].stages,
                ))
            explicit_tasks.append(PipelineTask(
                name=f"tail_{occurrence}_{sequence_repeat}",
                worker=0,
                stages=nested.tasks[1].stages,
            ))
    explicit = evaluate_pipeline(PipelinePlan(
        worker_count=1,
        resources=_resources(scope="worker"),
        tasks=tuple(explicit_tasks),
        buffers=buffers,
    ))

    assert compact.execution_time_s == pytest.approx(
        explicit.execution_time_s, rel=1e-12)
    assert compact.initiation_interval_s == pytest.approx(
        explicit.initiation_interval_s, rel=1e-12)
    assert compact.worker_makespan_s == pytest.approx(
        explicit.worker_makespan_s, rel=1e-12)
    assert compact.stage_service_s == pytest.approx(
        explicit.stage_service_s, rel=1e-12)
