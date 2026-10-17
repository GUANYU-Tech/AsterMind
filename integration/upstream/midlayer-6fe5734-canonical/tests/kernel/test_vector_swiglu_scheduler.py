from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.model_contract import (
    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
    AIV_WORKER_START_FILL,
    PipelineTaskDemand,
    PipelineTaskProgramDemand,
)


HARDWARE = HWSpec("configs/hardware/ascend_910b.yaml")


def _swiglu_input(tokens: int, intermediate: int):
    shape = {
        "tok": tokens,
        "intermediate": intermediate,
        "element_count": tokens * intermediate,
    }
    launch = select_launch_config(
        HARDWARE.hardware_id, KernelKind.VECTOR, shape, "bf16", "silu",
        {"launch_policy": "injected_launch"},
        hardware=HARDWARE)
    kernel = KernelSpec(
        KernelKind.VECTOR, HARDWARE.hardware_id, shape, dtype="bf16",
        variant=launch.execution_variant, launch=launch)
    return create_predictor(kernel.kind).resource_input(kernel, HARDWARE)[0]


def test_all_source_workers_use_one_entry_fill_and_empty_task_shell():
    shallow = _swiglu_input(tokens=64, intermediate=14336)
    steady = _swiglu_input(tokens=256, intermediate=14336)

    for prediction_input in (shallow, steady):
        assert prediction_input.task_duration_floor_contract == (
            AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM)
        assert prediction_input.aiv_worker_start_contract == (
            AIV_WORKER_START_FILL)
        assert prediction_input.aiv_source_entry_fill_contract is None


def _input_pair_ingress_group_sizes(prediction_input):
    """Return source-declared grouping for the ordered A/B CopyIn stage."""

    result = []
    for entry in prediction_input.pipeline_plan.tasks:
        sequences = (
            entry.patterns
            if isinstance(entry, PipelineTaskProgramDemand) else (entry,))
        for sequence in sequences:
            tasks = (
                sequence.tasks
                if hasattr(sequence, "tasks") else (sequence,))
            for task in tasks:
                result.extend(
                    stage.aiv_ingress_group_size
                    for stage in task.stages
                    if stage.name == "load_input_pair")
    return tuple(result)


def _input_pair_completion_by_source_position(prediction_input):
    """Return exact A/B completions keyed by source-loop position."""

    result = []
    for entry in prediction_input.pipeline_plan.tasks:
        if isinstance(entry, PipelineTaskProgramDemand):
            sequences = tuple(
                (pattern.name, task, entry.worker_multiplicity)
                for pattern_index in entry.order
                for pattern in (entry.patterns[pattern_index],)
                for task in pattern.tasks)
        elif isinstance(entry, PipelineTaskDemand):
            sequences = (("first", entry, 1),)
        else:
            raise AssertionError(f"unexpected SwiGLU pipeline task: {entry!r}")
        for source_position, task, multiplicity in sequences:
            stages = {stage.name: stage for stage in task.stages}
            result.append((
                source_position,
                multiplicity,
                stages["load_input_pair"].memory_exact_request_demands,
            ))
    return tuple(result)


def _tiling_control_prior_resources(prediction_input):
    """Return the source loop dependency for every compressed tile class."""

    result = []
    for entry in prediction_input.pipeline_plan.tasks:
        sequences = (
            entry.patterns
            if isinstance(entry, PipelineTaskProgramDemand) else (entry,))
        for sequence in sequences:
            tasks = (
                sequence.tasks
                if hasattr(sequence, "tasks") else (sequence,))
            for task in tasks:
                stages = {stage.name: stage for stage in task.stages}
                result.append(stages["tiling_index_control"].prior_resources)
    return tuple(result)


def _output_queue_lifetimes(prediction_input):
    """Return the source VECOUT credit boundaries for a SwiGLU program."""

    return tuple(
        (buffer.producer_stage, buffer.consumer_stage, buffer.slots)
        for buffer in prediction_input.pipeline_plan.buffers
        if buffer.name == "output_c_queue")


def test_one_tile_q2_swiglu_uses_exact_entry_or_grouped_ingress():
    """A finite source program selects one source-derived CopyIn mechanism."""

    cases = (
        _swiglu_input(tokens=1, intermediate=9728),
        _swiglu_input(tokens=1, intermediate=12288),
        _swiglu_input(tokens=2, intermediate=9728),
        _swiglu_input(tokens=4, intermediate=53248),
    )
    for prediction_input in cases:
        assert all(
            isinstance(entry, PipelineTaskDemand)
            for entry in prediction_input.pipeline_plan.tasks)
        assert len(prediction_input.pipeline_plan.tasks) == (
            prediction_input.active_units)
        entry_completions = _input_pair_completion_by_source_position(
            prediction_input)
        has_exact_entry = any(
            completion for _position, _multiplicity, completion
            in entry_completions)
        ingress_groups = _input_pair_ingress_group_sizes(prediction_input)
        if has_exact_entry:
            assert set(ingress_groups) == {None}
        else:
            assert set(ingress_groups) == {2}


def test_swiglu_pair_completion_is_distinct_from_grouped_ingress():
    """Published pair cells replace only the ordinary source ingress."""

    exact_pair = _swiglu_input(tokens=32, intermediate=9728)
    active20_strided = _swiglu_input(tokens=4, intermediate=25600)

    assert all(
        isinstance(entry, PipelineTaskProgramDemand)
        for entry in exact_pair.pipeline_plan.tasks)
    assert set(_input_pair_ingress_group_sizes(exact_pair)) == {None}
    assert any(
        completion for _position, _multiplicity, completion
        in _input_pair_completion_by_source_position(exact_pair))
    assert set(_input_pair_ingress_group_sizes(active20_strided)) == {2}
    assert not any(
        completion for _position, _multiplicity, completion
        in _input_pair_completion_by_source_position(active20_strided))


def test_swiglu_pair_completion_applies_only_to_each_worker_first_tile():
    """Repeated Key-27 programs retain entry cost but never replay it."""

    repeated = _swiglu_input(tokens=33, intermediate=9728)
    completions = _input_pair_completion_by_source_position(repeated)
    first_count = sum(
        multiplicity
        for source_position, multiplicity, completion in completions
        if source_position == "first" and completion)
    later = tuple(
        completion
        for source_position, _multiplicity, completion in completions
        if source_position != "first")

    assert repeated.active_units == 48
    # Every first source tile has the ordered pair stage. Only the 32 full
    # tiles match the published 18,432-byte completion cell; the remaining
    # first tiles deliberately retain the generic pair service rather than
    # borrowing a neighbouring exact cell.
    assert first_count == 32
    assert sum(
        multiplicity
        for source_position, multiplicity, _completion in completions
        if source_position == "first") == repeated.active_units
    assert later
    assert all(completion == () for completion in later)


def test_swiglu_source_loop_carries_copyout_before_next_tile_control():
    """Key-27 cannot begin its next tile before the prior CopyOut path."""

    for prediction_input in (
            _swiglu_input(tokens=1, intermediate=9728),
            _swiglu_input(tokens=33, intermediate=9728),
            _swiglu_input(tokens=1024, intermediate=53248)):
        dependencies = _tiling_control_prior_resources(prediction_input)
        assert dependencies
        assert set(dependencies) == {("memory:mte3_write",)}


def test_swiglu_output_queue_is_reserved_at_compute_entry():
    """Key-27 allocates VECOUT before Cast(A), then frees it in CopyOut."""

    for prediction_input in (
            _swiglu_input(tokens=33, intermediate=9728),
            _swiglu_input(tokens=1024, intermediate=53248)):
        assert _output_queue_lifetimes(prediction_input) == (
            ("cast_a", "store", 2),)


def test_finite_q2_grouped_ingress_uses_the_common_entry_fill():
    """Finite group ingress has the same AIV admission envelope as loops."""

    finite_q2 = (
        _swiglu_input(tokens=1, intermediate=12288),
        _swiglu_input(tokens=4, intermediate=25600),
        _swiglu_input(tokens=4, intermediate=53248),
    )
    for prediction_input in finite_q2:
        assert prediction_input.task_duration_floor_contract == (
            AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM)
        assert prediction_input.aiv_worker_start_contract == (
            AIV_WORKER_START_FILL)
        assert prediction_input.aiv_source_entry_fill_contract is None
