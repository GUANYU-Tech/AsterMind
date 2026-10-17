"""Deterministic worker-aware tile pipeline evaluator."""

from collections import OrderedDict
import dataclasses
from functools import lru_cache
import math
import threading

import numpy as np

from src.kernel.model_contract import (
    AttentionPipelineEnvelope,
    BarrierPipelineEnvelope,
    PipelineDiagnostics,
    PipelinePlan,
    PipelineTaskDiagnostic,
    PipelineTaskProgram,
    PipelineTaskSequence,
)


def _stage_predecessors(task, stage_index):
    """Resolve the legacy linear default or one explicit source DAG edge set."""

    stage = task.stages[stage_index]
    if stage.predecessors is None:
        return () if stage_index == 0 else (task.stages[stage_index - 1].name,)
    return stage.predecessors


def _stage_start_predecessors(task, stage_index):
    """Return explicit source issue-order dependencies for one stage."""

    return task.stages[stage_index].start_predecessors


def _stage_prior_resources(task, stage_index):
    """Return resource completions captured before the logical task starts."""

    return task.stages[stage_index].prior_resources


def _evaluate_explicit_pipeline(plan: PipelinePlan) -> PipelineDiagnostics:
    """Schedule a structured pipeline without wave-efficiency correction.

    Stages obey intra-tile order, finite buffer reuse, and explicit shared or
    worker-local service slots.  Task order is launch order and therefore also
    defines deterministic arbitration when multiple stages become ready at the
    same time.
    """

    if not isinstance(plan, PipelinePlan):
        raise TypeError("evaluate_pipeline requires PipelinePlan")
    resources = {resource.name: resource for resource in plan.resources}
    shared_availability = {}
    shared_predecessors = {}
    worker_availability = {}
    worker_predecessors = {}
    for resource in plan.resources:
        if resource.scope == "shared":
            shared_availability[resource.name] = [0.0] * resource.slots
            shared_predecessors[resource.name] = [None] * resource.slots
        elif resource.scope == "worker":
            worker_availability[resource.name] = {
                worker: [0.0] * resource.slots
                for worker in range(plan.worker_count)
            }
            worker_predecessors[resource.name] = {
                worker: [None] * resource.slots
                for worker in range(plan.worker_count)
            }
        else:
            raise AssertionError(
                f"unhandled pipeline resource scope: {resource.scope!r}")
    initial_availability = dict(plan.initial_resource_availability_s)
    for resource in plan.resources:
        if resource.name not in initial_availability:
            continue
        duration_s = initial_availability[resource.name]
        if resource.scope == "shared":
            shared_availability[resource.name] = (
                [duration_s] * resource.slots)
        else:
            worker_availability[resource.name] = {
                worker: [duration_s] * resource.slots
                for worker in range(plan.worker_count)
            }

    worker_tasks = {worker: [] for worker in range(plan.worker_count)}
    global_tasks = []
    stage_start = {}
    stage_end = {}
    predecessor = {}
    task_diagnostics = []
    worker_makespan = {
        str(worker): 0.0 for worker in range(plan.worker_count)
    }
    stage_service = {}
    first_stage_starts = {
        worker: [] for worker in range(plan.worker_count)
    }

    for task in plan.tasks:
        per_task_start = {}
        per_task_end = {}
        worker_index = len(worker_tasks[task.worker])
        global_index = len(global_tasks)
        for stage_index, stage in enumerate(task.stages):
            stage_key = (task.name, stage.name)
            ready_candidates = [(0.0, None)]
            for predecessor_name in _stage_predecessors(task, stage_index):
                predecessor_key = (task.name, predecessor_name)
                ready_candidates.append((
                    stage_end[predecessor_key], predecessor_key))
            for predecessor_name in _stage_start_predecessors(
                    task, stage_index):
                predecessor_key = (task.name, predecessor_name)
                ready_candidates.append((
                    stage_start[predecessor_key], predecessor_key))
            for buffer in plan.buffers:
                if buffer.producer_stage != stage.name:
                    continue
                if buffer.scope == "worker":
                    history = worker_tasks[task.worker]
                    history_index = worker_index - buffer.slots
                elif buffer.scope == "shared":
                    history = global_tasks
                    history_index = global_index - buffer.slots
                else:
                    raise AssertionError(
                        f"unhandled pipeline buffer scope: {buffer.scope!r}")
                if history_index >= 0:
                    prior_task = history[history_index]
                    prior_key = (prior_task.name, buffer.consumer_stage)
                    release_time = (
                        stage_start[prior_key]
                        if buffer.release == "start" else stage_end[prior_key])
                    ready_candidates.append((release_time, prior_key))

            resource = resources[stage.resource]
            if resource.scope == "shared":
                availability = shared_availability[resource.name]
                resource_predecessors = shared_predecessors[resource.name]
            elif resource.scope == "worker":
                availability = worker_availability[resource.name][task.worker]
                resource_predecessors = worker_predecessors[
                    resource.name][task.worker]
            elif envelope.cta_admission_policy == (
                    "bounded_resident_suffix_fifo"):
                raise AssertionError(
                    f"unhandled pipeline resource scope: {resource.scope!r}")
            slot = min(range(resource.slots), key=lambda value: availability[value])
            ready_candidates.append((
                availability[slot], resource_predecessors[slot]))
            start, critical_predecessor = max(
                ready_candidates,
                key=lambda item: (item[0], item[1] is not None))
            end = start + stage.duration_s
            per_task_start[stage.name] = start
            per_task_end[stage.name] = end
            stage_start[stage_key] = start
            stage_end[stage_key] = end
            predecessor[stage_key] = critical_predecessor
            availability[slot] = start + stage.occupancy_s
            resource_predecessors[slot] = stage_key
            stage_service[stage.name] = (
                stage_service[stage.name] + stage.duration_s
                if stage.name in stage_service else stage.duration_s)
            if stage_index == 0:
                first_stage_starts[task.worker].append(start)
        worker_tasks[task.worker].append(task)
        global_tasks.append(task)
        task_end = max(per_task_end.values())
        worker_makespan[str(task.worker)] = max(
            worker_makespan[str(task.worker)], task_end)
        task_diagnostics.append(PipelineTaskDiagnostic(
            task_name=task.name,
            worker=task.worker,
            stage_start_s=per_task_start,
            stage_end_s=per_task_end,
        ))

    execution_time = max(worker_makespan.values())
    intervals = []
    for starts in first_stage_starts.values():
        if len(starts) < 2:
            continue
        intervals.append(
            (starts[-1] - starts[0]) / (len(starts) - 1))
    initiation_interval = max(intervals) if intervals else 0.0

    terminal_key = max(stage_end, key=lambda key: stage_end[key])
    reverse_path = []
    cursor = terminal_key
    while cursor is not None:
        reverse_path.append(f"{cursor[0]}/{cursor[1]}")
        cursor = predecessor[cursor]
    critical_path = tuple(reversed(reverse_path))
    return PipelineDiagnostics(
        execution_time_s=execution_time,
        initiation_interval_s=initiation_interval,
        worker_makespan_s=worker_makespan,
        stage_service_s=stage_service,
        critical_path=critical_path,
        tasks=tuple(task_diagnostics),
    )


_NEGATIVE_INFINITY = float("-inf")
_ATTENTION_ENVELOPE_CACHE_SIZE = 8
_attention_envelope_cache = OrderedDict()
_attention_envelope_cache_lock = threading.Lock()


class _AttentionEnvelopeCacheKey:
    """Exact immutable payload with its recursive hash computed once."""

    __slots__ = ("payload", "cached_hash")

    def __init__(self, payload):
        self.payload = payload
        self.cached_hash = hash(payload)

    def __hash__(self):
        return self.cached_hash

    def __eq__(self, other):
        return (
            isinstance(other, _AttentionEnvelopeCacheKey)
            and self.payload == other.payload)


_ATTENTION_PHASE_TRANSITION_CACHE_SIZE = 512
_attention_phase_transition_cache = OrderedDict()
_attention_phase_transition_cache_lock = threading.Lock()
_ATTENTION_SOURCE_ORDINAL_RESOURCE = "__source_ordinal__"


def attention_round_robin_signatures(class_counts, lane_count):
    """Return class-count signatures without enumerating physical lanes.

    A class-major CTA stream assigns ordinal ``i`` to lane ``i % W``.  Each
    class therefore contributes a base quotient to every lane and one extra
    CTA on a circular interval.  Interval endpoints partition the lane ring
    into at most ``2 * class_count`` distinct signatures.
    """

    if (isinstance(lane_count, bool) or not isinstance(lane_count, int)
            or lane_count <= 0):
        raise ValueError("attention pipeline lane count must be positive")
    if (not isinstance(class_counts, tuple) or not class_counts
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in class_counts)):
        raise ValueError(
            "attention pipeline class counts must be positive integers")
    sparse = _attention_round_robin_sparse_signatures(
        class_counts, lane_count)
    grouped = []
    for entries, multiplicity in sparse:
        counts = [0] * len(class_counts)
        for class_index, count in entries:
            counts[class_index] = count
        grouped.append((tuple(counts), multiplicity))
    return tuple(grouped)


@lru_cache(maxsize=8)
def _attention_round_robin_sparse_signatures(class_counts, lane_count):
    """Return only source classes assigned to each distinct physical lane."""

    if (isinstance(lane_count, bool) or not isinstance(lane_count, int)
            or lane_count <= 0):
        raise ValueError("attention pipeline lane count must be positive")
    if (not isinstance(class_counts, tuple) or not class_counts
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or value <= 0 for value in class_counts)):
        raise ValueError(
            "attention pipeline class counts must be positive integers")
    entries_by_lane = [[] for _ in range(lane_count)]
    start = 0
    for class_index, count in enumerate(class_counts):
        base, remainder = divmod(count, lane_count)
        if base:
            for lane, entries in enumerate(entries_by_lane):
                extra = int((lane - start) % lane_count < remainder)
                entries.append((class_index, base + extra))
        else:
            for offset in range(remainder):
                lane = (start + offset) % lane_count
                entries_by_lane[lane].append((class_index, 1))
        start = (start + remainder) % lane_count
    grouped = {}
    for entries in entries_by_lane:
        if not entries:
            continue
        signature = tuple(entries)
        grouped[signature] = (
            grouped[signature] + 1 if signature in grouped else 1)
    if sum(grouped.values()) != min(lane_count, sum(class_counts)):
        raise RuntimeError(
            "attention round-robin signatures lost active lanes")
    return tuple(grouped.items())


def _shortest_exact_period(tokens):
    """Return the shortest period only when it tiles the complete sequence."""

    prefix = [0] * len(tokens)
    for index in range(1, len(tokens)):
        length = prefix[index - 1]
        while length and tokens[index] != tokens[length]:
            length = prefix[length - 1]
        if tokens[index] == tokens[length]:
            length += 1
        prefix[index] = length
    period = len(tokens) - prefix[-1]
    return period if len(tokens) % period == 0 else len(tokens)


@lru_cache(maxsize=512)
def _attention_source_blocks(entries, class_period):
    """Partition one immutable lane stream at exact source-period edges."""

    blocks = []
    current_block = None
    current_entries = []
    for class_index, count in entries:
        block = class_index // class_period
        if current_block is not None and block != current_block:
            blocks.append(tuple(current_entries))
            current_entries = []
        current_block = block
        current_entries.append((class_index, count))
    blocks.append(tuple(current_entries))
    return tuple(blocks)


def _maxplus_product(left, right):
    """Compose two max-plus linear state transitions."""

    size = len(left)
    if size >= 12:
        # Controller transitions are dense once their pipeline credits have
        # propagated.  NumPy evaluates the same max-plus inner reduction in
        # C, avoiding one Python traversal for every (row, inner, column).
        left_array = _maxplus_matrix_array(left)
        right_array = _maxplus_matrix_array(right)
        product_array = np.max(
            left_array[:, :, np.newaxis]
            + right_array[np.newaxis, :, :], axis=1)
        return tuple(tuple(row) for row in product_array.tolist())
    right_sparse = _maxplus_sparse_rows(right)
    product = []
    for left_row in left:
        output_row = [_NEGATIVE_INFINITY] * size
        for inner, lhs in enumerate(left_row):
            if lhs == _NEGATIVE_INFINITY:
                continue
            for column, rhs in right_sparse[inner]:
                candidate = lhs + rhs
                if candidate > output_row[column]:
                    output_row[column] = candidate
        product.append(tuple(output_row))
    return tuple(product)


def _maxplus_sparse_rows(matrix):
    """Pre-index finite coefficients of a max-plus matrix."""

    return tuple(
        tuple(
            (inner, coefficient)
            for inner, coefficient in enumerate(row)
            if coefficient != _NEGATIVE_INFINITY)
        for row in matrix)


def _maxplus_sparse_apply(matrix, vector):
    """Apply pre-indexed max-plus rows to one numeric state vector."""

    return tuple(
        max(coefficient + vector[column] for column, coefficient in row)
        for row in matrix)


def _maxplus_left_sparse_product(left_sparse, right):
    """Compose a sparse left transition with a dense right transition."""

    size = len(left_sparse)
    product = []
    for row in range(size):
        output_row = []
        for column in range(size):
            best = _NEGATIVE_INFINITY
            for inner, lhs in left_sparse[row]:
                rhs = right[inner][column]
                if rhs == _NEGATIVE_INFINITY:
                    continue
                best = max(best, lhs + rhs)
            output_row.append(best)
        product.append(tuple(output_row))
    return tuple(product)


@lru_cache(maxsize=4096)
def _maxplus_matrix_array(matrix):
    """Materialize one immutable square transition for vectorized application."""

    size = len(matrix)
    if any(not isinstance(row, tuple) or len(row) != size for row in matrix):
        raise ValueError("max-plus vectorized transition must be square tuples")
    array = np.ascontiguousarray(matrix, dtype=np.float64)
    array.setflags(write=False)
    return array


def _maxplus_apply(matrix, vector):
    """Apply one max-plus matrix to one numeric state vector."""

    size = len(matrix)
    if (isinstance(matrix, tuple) and size > 0 and len(vector) == size
            and all(isinstance(row, tuple) and len(row) == size
                    for row in matrix)):
        matrix_array = _maxplus_matrix_array(matrix)
        vector_array = np.ascontiguousarray(vector, dtype=np.float64)
        return tuple(np.max(matrix_array + vector_array, axis=1).tolist())
    output = []
    for row in matrix:
        best = _NEGATIVE_INFINITY
        for coefficient, value in zip(row, vector):
            if (coefficient == _NEGATIVE_INFINITY
                    or value == _NEGATIVE_INFINITY):
                continue
            best = max(best, coefficient + value)
        output.append(best)
    return tuple(output)


@lru_cache(maxsize=4096)
def _maxplus_square(matrix):
    """Reuse one exact max-plus square across different repeat counts."""

    return _maxplus_product(matrix, matrix)


@lru_cache(maxsize=4096)
def _maxplus_powers(matrix, exponent):
    powers = []
    current = matrix
    remaining = exponent
    while remaining > 0:
        powers.append(current)
        remaining >>= 1
        if remaining > 0:
            current = _maxplus_square(current)
    return tuple(powers)


def _maxplus_identity(size):
    return tuple(
        tuple(0.0 if row == column else _NEGATIVE_INFINITY
              for column in range(size))
        for row in range(size))


def _maxplus_matrix_power(matrix, exponent):
    """Return one exact binary max-plus power."""

    if exponent < 0:
        raise ValueError("max-plus exponent must be non-negative")
    return _maxplus_power_matrix(
        _maxplus_powers(matrix, exponent), exponent, len(matrix))


@dataclasses.dataclass(frozen=True)
class _CompiledPipelineTopology:
    """Static, duration-free source DAG compiled for max-plus binding.

    A topology deliberately retains source names instead of concrete state
    indexes.  That lets the same finite source DAG bind to a different
    hardware resource layout (or a different resolved duration vector)
    without caching a prediction or relaxing the source-program contract.
    """

    stage_signature: tuple
    buffer_signature: tuple
    uses_source_ordinal: bool
    predecessor_indexes: tuple
    start_predecessor_indexes: tuple
    producer_buffer_indexes: tuple
    consumer_stage_indexes: tuple
    source_ordinal_producer_stages: tuple


@dataclasses.dataclass(frozen=True)
class _CompiledPipelineBindingSkeleton:
    """Static state layout for one duration-free source topology.

    ``_CompiledPipelineTopology`` owns source-DAG facts.  This companion
    records how that DAG is embedded in one max-plus state vector.  It is
    deliberately independent of the current stage duration and occupancy:
    those remain dynamic prediction inputs to ``_bind_pipeline_topology``.
    Keeping this split means a source program is compiled once while a
    fill/steady/drain envelope can price every shape afresh.
    """

    stage_resource_indexes: tuple
    stage_prior_resource_indexes: tuple
    buffer_indexes: tuple
    constant_index: int
    source_ordinal_index: object


def _pipeline_stage_topology_signature(stages):
    """Normalize one ordered source DAG without its dynamic scalar values."""

    if not isinstance(stages, tuple) or not stages:
        raise TypeError("pipeline topology stages must be a non-empty tuple")
    names = []
    prior_names = set()
    signature = []
    for stage_index, stage in enumerate(stages):
        if stage.name in prior_names:
            raise ValueError("pipeline topology stage names must be unique")
        predecessors = (
            (() if stage_index == 0 else (names[-1],))
            if stage.predecessors is None else stage.predecessors)
        unknown_predecessors = set(predecessors) - prior_names
        unknown_start_predecessors = (
            set(stage.start_predecessors) - prior_names)
        if unknown_predecessors or unknown_start_predecessors:
            raise ValueError(
                "pipeline topology stage predecessors must name earlier "
                "stages: "
                f"end={sorted(unknown_predecessors)}, "
                f"start={sorted(unknown_start_predecessors)}")
        signature.append((
            stage.name,
            stage.resource,
            predecessors,
            stage.start_predecessors,
            stage.prior_resources,
        ))
        names.append(stage.name)
        prior_names.add(stage.name)
    return tuple(signature)


def _pipeline_buffer_topology_signature(buffers):
    """Return source-visible buffer lifetimes without state indexes."""

    if not isinstance(buffers, tuple):
        raise TypeError("pipeline topology buffers must be a tuple")
    return tuple(
        (
            buffer.name,
            buffer.producer_stage,
            buffer.consumer_stage,
            buffer.slots,
            buffer.release,
        )
        for buffer in buffers)


@lru_cache(maxsize=4096)
def _compile_pipeline_topology(
        stage_signature, buffer_signature, uses_source_ordinal):
    """Compile a finite source DAG once, independent of duration binding.

    The output contains only immutable topology: stage dependencies, producer
    credits, consumer releases, and the optional source-order gate.  Dynamic
    stage duration and occupancy are bound by ``_bind_pipeline_topology`` for
    every prediction, so this cache is neither a duration model nor a lookup
    table.
    """

    if not isinstance(stage_signature, tuple) or not stage_signature:
        raise TypeError("pipeline topology signature must be non-empty tuple")
    if not isinstance(buffer_signature, tuple):
        raise TypeError("pipeline buffer signature must be a tuple")
    if not isinstance(uses_source_ordinal, bool):
        raise TypeError("pipeline source-ordinal policy must be boolean")
    stage_names = tuple(item[0] for item in stage_signature)
    if len(stage_names) != len(set(stage_names)):
        raise ValueError("pipeline topology stage names must be unique")
    stage_indexes = {
        name: index for index, name in enumerate(stage_names)}
    predecessor_indexes = []
    start_predecessor_indexes = []
    for stage_index, item in enumerate(stage_signature):
        if not isinstance(item, tuple) or len(item) != 5:
            raise TypeError("pipeline topology stage signature is malformed")
        _, _, predecessors, start_predecessors, _ = item
        unknown_predecessors = set(predecessors) - set(stage_names[:stage_index])
        unknown_start_predecessors = (
            set(start_predecessors) - set(stage_names[:stage_index]))
        if unknown_predecessors or unknown_start_predecessors:
            raise ValueError(
                "pipeline topology dependencies must name earlier stages: "
                f"end={sorted(unknown_predecessors)}, "
                f"start={sorted(unknown_start_predecessors)}")
        predecessor_indexes.append(
            tuple(stage_indexes[name] for name in predecessors))
        start_predecessor_indexes.append(
            tuple(stage_indexes[name] for name in start_predecessors))

    buffer_names = []
    producer_buffer_indexes = [[] for _ in stage_signature]
    consumer_stage_indexes = []
    source_ordinal_producer_stages = [False] * len(stage_signature)
    for buffer_index, item in enumerate(buffer_signature):
        if not isinstance(item, tuple) or len(item) != 5:
            raise TypeError("pipeline topology buffer signature is malformed")
        name, producer_stage, consumer_stage, slots, release = item
        if name in buffer_names:
            raise ValueError("pipeline topology buffer names must be unique")
        if (isinstance(slots, bool) or not isinstance(slots, int)
                or slots <= 0):
            raise ValueError("pipeline topology buffer slots must be positive")
        if release not in ("start", "completion"):
            raise ValueError("pipeline topology buffer release is unknown")
        buffer_names.append(name)
        if producer_stage in stage_indexes:
            producer_index = stage_indexes[producer_stage]
            producer_buffer_indexes[producer_index].append(buffer_index)
            if uses_source_ordinal:
                source_ordinal_producer_stages[producer_index] = True
        consumer_stage_indexes.append(
            stage_indexes[consumer_stage]
            if consumer_stage in stage_indexes else None)
    return _CompiledPipelineTopology(
        stage_signature=stage_signature,
        buffer_signature=buffer_signature,
        uses_source_ordinal=uses_source_ordinal,
        predecessor_indexes=tuple(predecessor_indexes),
        start_predecessor_indexes=tuple(start_predecessor_indexes),
        producer_buffer_indexes=tuple(
            tuple(indexes) for indexes in producer_buffer_indexes),
        consumer_stage_indexes=tuple(consumer_stage_indexes),
        source_ordinal_producer_stages=tuple(source_ordinal_producer_stages),
    )


@lru_cache(maxsize=4096)
def _compile_pipeline_binding_skeleton(
        topology, resource_layout, buffer_layout, constant_index,
        source_ordinal_index):
    """Compile one checked source-DAG/state-vector embedding.

    The cache key contains topology and state *layout* only.  In particular,
    it contains neither a prediction duration nor a workload shape, so this
    is not a result or duration cache.
    """

    if not isinstance(topology, _CompiledPipelineTopology):
        raise TypeError("pipeline binding skeleton requires compiled topology")
    if (not isinstance(resource_layout, tuple)
            or any(not isinstance(item, tuple) or len(item) != 2
                   for item in resource_layout)):
        raise TypeError("pipeline binding resource layout is malformed")
    if (not isinstance(buffer_layout, tuple)
            or len(buffer_layout) != len(topology.buffer_signature)):
        raise TypeError("pipeline binding buffer layout is malformed")
    if (isinstance(constant_index, bool) or not isinstance(constant_index, int)
            or constant_index < 0):
        raise ValueError("pipeline topology constant index is invalid")
    if topology.uses_source_ordinal != (source_ordinal_index is not None):
        raise ValueError(
            "pipeline topology source-ordinal policy does not match binding")
    if source_ordinal_index is not None:
        if (isinstance(source_ordinal_index, bool)
                or not isinstance(source_ordinal_index, int)
                or source_ordinal_index < 0
                or source_ordinal_index >= constant_index):
            raise ValueError("pipeline source-ordinal index is invalid")

    resource_indexes = dict(resource_layout)
    if len(resource_indexes) != len(resource_layout):
        raise ValueError("pipeline topology binding resources cannot repeat")
    required_resources = set()
    for _, service, _, _, prior_resources in topology.stage_signature:
        required_resources.add(service)
        required_resources.update(prior_resources)
    missing_resources = required_resources - set(resource_indexes)
    if missing_resources:
        raise ValueError(
            "pipeline topology binding is missing resources: "
            f"{sorted(missing_resources)}")

    stage_resource_indexes = []
    stage_prior_resource_indexes = []
    for _, resource, _, _, prior_resources in topology.stage_signature:
        resource_index = resource_indexes[resource]
        if (isinstance(resource_index, bool)
                or not isinstance(resource_index, int)
                or resource_index < 0 or resource_index >= constant_index):
            raise ValueError(
                "pipeline topology binding resource index is invalid: "
                f"{resource!r}")
        stage_resource_indexes.append(resource_index)
        prior_indexes = []
        for prior_resource in prior_resources:
            prior_index = resource_indexes[prior_resource]
            if (isinstance(prior_index, bool)
                    or not isinstance(prior_index, int)
                    or prior_index < 0 or prior_index >= constant_index):
                raise ValueError(
                    "pipeline topology binding resource index is invalid: "
                    f"{prior_resource!r}")
            prior_indexes.append(prior_index)
        stage_prior_resource_indexes.append(tuple(prior_indexes))

    buffer_indexes = []
    for (name, _, _, slots, _), indexes in zip(
            topology.buffer_signature, buffer_layout):
        if (not isinstance(indexes, tuple) or len(indexes) != slots
                or any(isinstance(index, bool) or not isinstance(index, int)
                       or index < 0 or index >= constant_index
                       for index in indexes)):
            raise ValueError(
                "pipeline topology binding buffer state indexes are invalid: "
                f"{name!r}")
        buffer_indexes.append(indexes)
    return _CompiledPipelineBindingSkeleton(
        stage_resource_indexes=tuple(stage_resource_indexes),
        stage_prior_resource_indexes=tuple(stage_prior_resource_indexes),
        buffer_indexes=tuple(buffer_indexes),
        constant_index=constant_index,
        source_ordinal_index=source_ordinal_index,
    )


def _bind_pipeline_topology(
        topology, stages, resource_indexes, buffer_state_indexes,
        constant_index, *, source_ordinal_index=None, _stage_signature=None):
    """Bind resolved stage scalars to one compiled source topology exactly."""

    if not isinstance(topology, _CompiledPipelineTopology):
        raise TypeError("pipeline topology binding requires compiled topology")
    stage_signature = (
        _pipeline_stage_topology_signature(stages)
        if _stage_signature is None else _stage_signature)
    if stage_signature != topology.stage_signature:
        raise ValueError(
            "pipeline topology binding stage DAG does not match template")
    if not isinstance(resource_indexes, dict):
        raise TypeError("pipeline topology resource indexes must be a dict")
    if not isinstance(buffer_state_indexes, dict):
        raise TypeError("pipeline topology buffer indexes must be a dict")
    required_buffers = tuple(item[0] for item in topology.buffer_signature)
    unexpected_buffers = set(buffer_state_indexes) - set(required_buffers)
    missing_buffers = set(required_buffers) - set(buffer_state_indexes)
    if unexpected_buffers or missing_buffers:
        raise ValueError(
            "pipeline topology binding buffer names disagree: "
            f"missing={sorted(missing_buffers)}, "
            f"unexpected={sorted(unexpected_buffers)}")
    buffer_layout = []
    for name, _, _, slots, _ in topology.buffer_signature:
        indexes = buffer_state_indexes[name]
        buffer_layout.append(indexes)
    skeleton = _compile_pipeline_binding_skeleton(
        topology,
        tuple(sorted(resource_indexes.items())),
        tuple(buffer_layout),
        constant_index,
        source_ordinal_index,
    )

    # This is the numerical binding half of a finite source program.  The
    # previous implementation rebuilt a tuple expression graph for every
    # prediction.  Here the cached skeleton supplies all static dependencies
    # and the current prediction contributes only duration/occupancy scalars.
    # It produces the same exact max-plus transition, without enumerating
    # CTAs, panels, or waves.
    size = skeleton.constant_index + 1
    state = np.full((size, size), _NEGATIVE_INFINITY, dtype=np.float64)
    np.fill_diagonal(state, 0.0)
    prior_resource_state = state.copy()
    stage_start = [None] * len(stages)
    stage_end = [None] * len(stages)
    for stage_index, stage in enumerate(stages):
        start = state[skeleton.stage_resource_indexes[stage_index]].copy()
        predecessors = topology.predecessor_indexes[stage_index]
        if not predecessors:
            np.maximum(start, state[skeleton.constant_index], out=start)
        else:
            for index in predecessors:
                np.maximum(start, stage_end[index], out=start)
        for index in topology.start_predecessor_indexes[stage_index]:
            np.maximum(start, stage_start[index], out=start)
        for index in skeleton.stage_prior_resource_indexes[stage_index]:
            np.maximum(start, prior_resource_state[index], out=start)
        for buffer_index in topology.producer_buffer_indexes[stage_index]:
            np.maximum(start, state[skeleton.buffer_indexes[buffer_index][0]],
                       out=start)
        if topology.source_ordinal_producer_stages[stage_index]:
            np.maximum(start, state[skeleton.source_ordinal_index], out=start)
        end = start + stage.duration_s
        stage_start[stage_index] = start
        stage_end[stage_index] = end
        state[skeleton.stage_resource_indexes[stage_index]] = (
            start + stage.occupancy_s)
        if topology.source_ordinal_producer_stages[stage_index]:
            state[skeleton.source_ordinal_index] = start
    for buffer_index, (_, _, _, _, release) in enumerate(
            topology.buffer_signature):
        consumer_index = topology.consumer_stage_indexes[buffer_index]
        if consumer_index is None:
            continue
        indexes = skeleton.buffer_indexes[buffer_index]
        old_values = state[np.asarray(indexes)].copy()
        release_value = (
            stage_start[consumer_index]
            if release == "start" else stage_end[consumer_index])
        state[np.asarray(indexes)] = np.vstack((old_values[1:], release_value))
    state[skeleton.constant_index] = _NEGATIVE_INFINITY
    state[skeleton.constant_index, skeleton.constant_index] = 0.0
    return tuple(tuple(row.tolist()) for row in state)


def _compile_pipeline_transition(
        stages, resource_indexes, buffer_state_indexes, buffer_signature,
        constant_index, *, source_ordinal_index=None):
    """Compile then dynamically bind one exact finite max-plus transition."""

    stage_signature = _pipeline_stage_topology_signature(stages)
    topology = _compile_pipeline_topology(
        stage_signature,
        buffer_signature,
        source_ordinal_index is not None,
    )
    return _bind_pipeline_topology(
        topology, stages, resource_indexes, buffer_state_indexes,
        constant_index, source_ordinal_index=source_ordinal_index,
        _stage_signature=stage_signature)


def _attention_phase_transition_matrix(
        phase, resources, buffer_indexes, constant_index,
        base_phase_transitions=None):
    """Compile one finite source phase without constructing a task object."""

    stages_identity = id(phase.stages)
    if (base_phase_transitions is not None
            and stages_identity in base_phase_transitions):
        matrix = base_phase_transitions[stages_identity]
    else:
        cache_key = _AttentionEnvelopeCacheKey((
            phase.stages,
            tuple(sorted(resources.items())),
            buffer_indexes,
            constant_index,
        ))
        with _attention_phase_transition_cache_lock:
            if cache_key in _attention_phase_transition_cache:
                matrix = _attention_phase_transition_cache[cache_key]
                _attention_phase_transition_cache.move_to_end(cache_key)
            else:
                matrix = None
        if matrix is not None:
            if base_phase_transitions is not None:
                base_phase_transitions[stages_identity] = matrix
            if phase.repeat_count == 1:
                return matrix
            return _maxplus_matrix_power(matrix, phase.repeat_count)
        source_ordinal_index = None
        if _ATTENTION_SOURCE_ORDINAL_RESOURCE in resources:
            source_ordinal_index = resources[
                _ATTENTION_SOURCE_ORDINAL_RESOURCE]
        attention_buffer_signature = tuple(
            (name, producer_stage, consumer_stage, len(indexes),
             "completion")
            for name, indexes, producer_stage, consumer_stage in buffer_indexes)
        attention_buffer_indexes = {
            name: indexes
            for name, indexes, _, _ in buffer_indexes}
        matrix = _compile_pipeline_transition(
            phase.stages, resources, attention_buffer_indexes,
            attention_buffer_signature, constant_index,
            source_ordinal_index=source_ordinal_index)
        with _attention_phase_transition_cache_lock:
            if cache_key in _attention_phase_transition_cache:
                matrix = _attention_phase_transition_cache[cache_key]
                _attention_phase_transition_cache.move_to_end(cache_key)
            else:
                _attention_phase_transition_cache[cache_key] = matrix
                if (len(_attention_phase_transition_cache)
                        > _ATTENTION_PHASE_TRANSITION_CACHE_SIZE):
                    _attention_phase_transition_cache.popitem(last=False)
        if base_phase_transitions is not None:
            base_phase_transitions[stages_identity] = matrix
    if phase.repeat_count == 1:
        return matrix
    return _maxplus_matrix_power(matrix, phase.repeat_count)


def _attention_class_transition_matrix(
        work_class, resources, buffer_indexes, constant_index,
        phase_transitions=None, base_phase_transitions=None):
    """Compile one complete CTA-class source program into one transition."""

    if not work_class.phases:
        if work_class.query_rows != 0 or work_class.panel_count != 0:
            raise ValueError(
                "phase-free attention class must be a source-ordinal marker")
        if len(buffer_indexes) != 1:
            raise ValueError(
                "attention source-ordinal marker requires one lifecycle "
                "buffer")
        transition = list(_maxplus_identity(constant_index + 1))
        source_ordinal_index = resources[_ATTENTION_SOURCE_ORDINAL_RESOURCE]
        _, indexes, _, _ = buffer_indexes[0]
        admission = _expression_max((
            _expression_basis(constant_index + 1, source_ordinal_index),
            _expression_basis(constant_index + 1, indexes[0]),
            _expression_basis(constant_index + 1, constant_index),
        ))
        for current, following in zip(indexes, indexes[1:]):
            transition[current] = _expression_basis(
                constant_index + 1, following)
        transition[indexes[-1]] = admission
        transition[source_ordinal_index] = admission
        return tuple(transition)
    result = None
    for phase in work_class.phases:
        phase_key = (id(phase.stages), phase.repeat_count)
        if (phase_transitions is not None
                and phase_key in phase_transitions):
            matrix = phase_transitions[phase_key]
        else:
            matrix = _attention_phase_transition_matrix(
                phase, resources, buffer_indexes, constant_index,
                base_phase_transitions)
            if phase_transitions is not None:
                phase_transitions[phase_key] = matrix
        result = (
            matrix if result is None else _maxplus_product(matrix, result))
    if result is None:
        raise RuntimeError("attention pipeline class has no source phases")
    return result


def _ulp_close(left, right, *, ulps=64):
    if left == right:
        return True
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    tolerance = ulps * max(math.ulp(left), math.ulp(right))
    return abs(left - right) <= tolerance


def _attention_affine_shift(first, second, constant_index):
    """Return the common physical-state shift, or ``None`` if non-affine."""

    if first[constant_index] != 0.0 or second[constant_index] != 0.0:
        raise RuntimeError("attention affine coordinate must remain zero")
    if any(value == _NEGATIVE_INFINITY
           for value in first[:constant_index]):
        return None
    delta = second[0] - first[0]
    if delta < 0.0:
        return None
    if all(_ulp_close(
            second[index], first[index] + delta)
           for index in range(constant_index)):
        return delta
    return None


def _attention_dynamic_terms_dominate(
        transition, state, constant_index):
    """Prove an affine shift cannot return to a startup-only branch."""

    for row in range(constant_index):
        dynamic_terms = tuple(
            transition[row][column] + state[column]
            for column in range(constant_index)
            if transition[row][column] != _NEGATIVE_INFINITY
            and state[column] != _NEGATIVE_INFINITY)
        constant = transition[row][constant_index]
        if constant == _NEGATIVE_INFINITY:
            continue
        if not dynamic_terms:
            return False
        dynamic = max(dynamic_terms)
        tolerance = 64 * max(math.ulp(dynamic), math.ulp(constant))
        if dynamic + tolerance < constant:
            return False
    return True


def _apply_attention_closed_form(
        transition, repeat_count, initial_state, constant_index):
    """Apply a repeated CTA class as startup + steady state + residue.

    The affine regime is accepted only after a complete physical state repeats
    with one common shift and the corresponding transition's dynamic terms
    dominate its constant startup column.  Those two conditions make the
    extrapolation inductive; no observed duration or heuristic convergence
    threshold enters the result.
    """

    if repeat_count < 0:
        raise ValueError("attention class repeat count cannot be negative")
    if repeat_count == 0:
        return initial_state, 0.0
    state_size = constant_index + 1
    maximum_period = max(1, constant_index)
    # The state dimension is fixed by the finite source phase graph.  Four
    # state widths cover the transient of the supported SM80 envelopes while
    # keeping the proof work independent of the launch grid.
    transient_limit = 4 * state_size
    states = [initial_state]
    for _ in range(min(repeat_count, transient_limit + maximum_period)):
        states.append(_maxplus_apply(transition, states[-1]))
    if repeat_count < len(states):
        return states[repeat_count], 0.0

    period_transitions = {}
    current = _maxplus_identity(state_size)
    for period in range(1, maximum_period + 1):
        current = _maxplus_product(transition, current)
        period_transitions[period] = current

    for transient in range(1, transient_limit + 1):
        for period in range(1, maximum_period + 1):
            later = transient + period
            if later >= len(states):
                continue
            delta = _attention_affine_shift(
                states[transient], states[later], constant_index)
            if delta is None:
                continue
            period_transition = period_transitions[period]
            if not _attention_dynamic_terms_dominate(
                    period_transition, states[transient], constant_index):
                continue
            quotient, residue = divmod(repeat_count - transient, period)
            shifted = tuple(
                value + quotient * delta
                if index < constant_index else value
                for index, value in enumerate(states[transient]))
            for _ in range(residue):
                shifted = _maxplus_apply(transition, shifted)
            return shifted, delta / period
    raise RuntimeError(
        "attention pipeline did not prove a finite startup/steady/tail "
        "envelope")


def _attention_envelope_cache_key(envelope):
    program_indexes = {}
    programs = []
    classes = []
    for work_class in envelope.classes:
        identity = id(work_class.phases)
        if identity not in program_indexes:
            program_indexes[identity] = len(programs)
            programs.append(work_class.phases)
        classes.append((
            work_class.count,
            work_class.query_rows,
            work_class.panel_count,
            program_indexes[identity],
        ))
    return _AttentionEnvelopeCacheKey((
        envelope.active_lane_count,
        tuple(programs),
        tuple(classes),
        envelope.source_repeat_count,
        envelope.cta_admission_policy,
        envelope.buffers,
        envelope.resident_tail_latency_s_per_panel,
        envelope.resident_tail_group_interval_panels,
        envelope.resident_tail_latency_s_per_group,
    ))


def evaluate_attention_pipeline_envelope(
        envelope: AttentionPipelineEnvelope) -> PipelineDiagnostics:
    """Evaluate and reuse an identical immutable attention source envelope."""

    if not isinstance(envelope, AttentionPipelineEnvelope):
        raise TypeError(
            "evaluate_attention_pipeline_envelope requires "
            "AttentionPipelineEnvelope")
    key = _attention_envelope_cache_key(envelope)
    with _attention_envelope_cache_lock:
        if key in _attention_envelope_cache:
            result = _attention_envelope_cache[key]
            _attention_envelope_cache.move_to_end(key)
            return result
    result = _evaluate_attention_pipeline_envelope_uncached(envelope)
    with _attention_envelope_cache_lock:
        if key in _attention_envelope_cache:
            result = _attention_envelope_cache[key]
            _attention_envelope_cache.move_to_end(key)
            return result
        _attention_envelope_cache[key] = result
        if len(_attention_envelope_cache) > _ATTENTION_ENVELOPE_CACHE_SIZE:
            _attention_envelope_cache.popitem(last=False)
    return result


def _evaluate_attention_pipeline_envelope_uncached(
        envelope: AttentionPipelineEnvelope) -> PipelineDiagnostics:
    """Evaluate finite attention classes without CTA or worker expansion."""

    if not isinstance(envelope, AttentionPipelineEnvelope):
        raise TypeError(
            "evaluate_attention_pipeline_envelope requires "
            "AttentionPipelineEnvelope")
    phase_by_identity = {}
    multiplier_by_phase_identity = {}
    for work_class in envelope.classes:
        for phase in work_class.phases:
            identity = id(phase)
            phase_by_identity[identity] = phase
            multiplier = (
                envelope.source_repeat_count
                * work_class.count
                * phase.repeat_count)
            multiplier_by_phase_identity[identity] = (
                multiplier_by_phase_identity[identity] + multiplier
                if identity in multiplier_by_phase_identity
                else multiplier)
    service_names = tuple(sorted({
        stage.resource
        for phase in phase_by_identity.values()
        for stage in phase.stages
    }))
    if any(not work_class.phases for work_class in envelope.classes):
        if _ATTENTION_SOURCE_ORDINAL_RESOURCE in service_names:
            raise ValueError(
                "attention source ordinal resource name is reserved")
        service_names += (_ATTENTION_SOURCE_ORDINAL_RESOURCE,)
    resources = {
        name: index for index, name in enumerate(service_names)}
    next_index = len(resources)
    buffer_indexes = []
    for buffer in envelope.buffers:
        indexes = tuple(range(next_index, next_index + buffer.slots))
        next_index += buffer.slots
        buffer_indexes.append((
            buffer.name, indexes,
            buffer.producer_stage, buffer.consumer_stage))
    constant_index = next_index
    state_size = constant_index + 1
    initial = [0.0] * state_size
    for _, indexes, _, _ in buffer_indexes:
        for index in indexes:
            initial[index] = _NEGATIVE_INFINITY
    initial[constant_index] = 0.0
    initial = tuple(initial)

    transition_by_program = {}
    sparse_transition_by_program = {}
    phase_transitions = {}
    base_phase_transitions = {}
    class_transitions = []
    class_sparse_transitions = []
    class_programs = []
    for work_class in envelope.classes:
        program = tuple(id(phase) for phase in work_class.phases)
        if program not in transition_by_program:
            transition_by_program[program] = (
                _attention_class_transition_matrix(
                    work_class, resources, tuple(buffer_indexes),
                    constant_index, phase_transitions,
                    base_phase_transitions))
            sparse_transition_by_program[program] = _maxplus_sparse_rows(
                transition_by_program[program])
        class_transitions.append(transition_by_program[program])
        class_sparse_transitions.append(
            sparse_transition_by_program[program])
        class_programs.append(program)
    class_transitions = tuple(class_transitions)
    class_sparse_transitions = tuple(class_sparse_transitions)
    base_class_count = len(envelope.classes)
    if envelope.cta_admission_policy in (
            "paired_resident_fifo", "bounded_resident_suffix_fifo"):
        if (len(buffer_indexes) != 1
                or len(buffer_indexes[0][1]) != 2):
            raise ValueError(
                "resident FIFO admission requires exactly one "
                "two-slot lifecycle buffer")
        source_class_count = envelope.source_repeat_count * sum(
            work_class.count for work_class in envelope.classes)
        if source_class_count > 512:
            raise ValueError(
                "paired resident FIFO admission exceeds its 512-CTA "
                "mechanism bound")
        if source_class_count <= 2 * envelope.active_lane_count:
            raise ValueError(
                "resident FIFO admission requires a queued CTA tail")
        if (envelope.resident_tail_latency_s_per_panel > 0
                or envelope.resident_tail_latency_s_per_group > 0):
            raise ValueError(
                "resident FIFO admission already resolves the finite "
                "resident tail")
    logical_class_counts = (
        tuple(work_class.count for work_class in envelope.classes)
        * envelope.source_repeat_count)
    signatures = _attention_round_robin_sparse_signatures(
        logical_class_counts, envelope.active_lane_count)
    class_period = (
        base_class_count
        if envelope.source_repeat_count > 1
        else _shortest_exact_period(tuple(
            (class_programs[index], work_class.count,
             work_class.query_rows, work_class.panel_count)
            for index, work_class in enumerate(envelope.classes))))
    compact_source_order = class_period < len(envelope.classes)
    if envelope.source_repeat_count > 1:
        compact_source_order = True
    composed_blocks = {}
    makespans = {}
    entries_by_key = {}
    maximum_initiation_interval = 0.0

    def _apply_entries_exact(entries, state):
        nonlocal maximum_initiation_interval
        for class_index, count in entries:
            class_index %= base_class_count
            if count == 1:
                state = _maxplus_sparse_apply(
                    class_sparse_transitions[class_index], state)
                initiation_interval = 0.0
            else:
                transition = class_transitions[class_index]
                state, initiation_interval = _apply_attention_closed_form(
                    transition, count, state, constant_index)
            maximum_initiation_interval = max(
                maximum_initiation_interval, initiation_interval)
        return state

    def _source_block_matrix(block_entries):
        if any(count != 1 for _, count in block_entries):
            return None
        signature = tuple(
            class_programs[class_index % base_class_count]
            for class_index, _ in block_entries)
        if signature not in composed_blocks:
            combined = None
            for class_index, _ in block_entries:
                class_index %= base_class_count
                transition = np.asarray(
                    class_transitions[class_index], dtype=np.float64)
                combined = (
                    transition if combined is None else
                    np.max(
                        transition[:, :, None]
                        + combined[None, :, :],
                        axis=1))
            composed_blocks[signature] = combined
        return composed_blocks[signature]

    def _apply_repeated_source_blocks(entries, state):
        for block_entries in _attention_source_blocks(
                entries, class_period):
            matrix = _source_block_matrix(block_entries)
            if matrix is None:
                state = _apply_entries_exact(block_entries, state)
                continue
            state = tuple(np.max(
                matrix + np.asarray(state, dtype=np.float64)[None, :],
                axis=1))
        return state

    def _add_resident_tail(makespan, entries):
        if (envelope.resident_tail_latency_s_per_panel <= 0
                and envelope.resident_tail_latency_s_per_group <= 0):
            return makespan
        largest_panel_count = 0
        second_panel_count = 0
        for class_index, count in entries:
            work_class = envelope.classes[class_index % base_class_count]
            panel_count = work_class.panel_count
            if panel_count >= largest_panel_count:
                if count >= 2:
                    second_panel_count = panel_count
                else:
                    second_panel_count = largest_panel_count
                largest_panel_count = panel_count
            elif panel_count > second_panel_count:
                second_panel_count = panel_count
        makespan += (
            largest_panel_count - second_panel_count
        ) * envelope.resident_tail_latency_s_per_panel
        if envelope.resident_tail_group_interval_panels > 0:
            interval = envelope.resident_tail_group_interval_panels
            makespan += (
                math.ceil(largest_panel_count / interval)
                - math.ceil(second_panel_count / interval)
            ) * envelope.resident_tail_latency_s_per_group
        return makespan

    if envelope.cta_admission_policy in (
            "paired_resident_fifo", "bounded_resident_suffix_fifo"):
        source_classes = tuple(
            class_index
            for _ in range(envelope.source_repeat_count)
            for class_index, work_class in enumerate(envelope.classes)
            for _ in range(work_class.count))
        lane_states = [initial] * envelope.active_lane_count
        resident_prefix = 2 * envelope.active_lane_count
        oldest_slot_index = buffer_indexes[0][1][0]
        for ordinal, class_index in enumerate(source_classes):
            if envelope.cta_admission_policy == "paired_resident_fifo":
                # The CUDA probe fixes adjacent source pairs in the two
                # resident slots of every SM, then admits the global FIFO
                # head wherever the oldest resident slot completes first.
                if ordinal < resident_prefix:
                    lane = ordinal // 2
                else:
                    lane = min(
                        range(envelope.active_lane_count),
                        key=lambda index: (
                            lane_states[index][oldest_slot_index], index))
            elif envelope.cta_admission_policy == (
                    "bounded_resident_suffix_fifo"):
                # Preserve the constant-size balanced prefix and resolve only
                # the final hardware-resident cohort.  This is the bounded
                # drain topology observed by the dense Q128/KV64 probe.
                balanced_prefix_count = len(source_classes) - resident_prefix
                if ordinal < balanced_prefix_count:
                    lane = ordinal % envelope.active_lane_count
                else:
                    lane = min(
                        range(envelope.active_lane_count),
                        key=lambda index: (
                            lane_states[index][oldest_slot_index], index))
            else:
                raise ValueError(
                    "unsupported attention CTA admission policy")
            lane_states[lane] = _maxplus_sparse_apply(
                class_sparse_transitions[class_index], lane_states[lane])
        makespans = {
            f"lane{lane}": max(state[:constant_index])
            for lane, state in enumerate(lane_states)
        }
        execution_time = max(makespans.values())
        compact_source_order = False
        signatures = ()

    batched_states = None
    if compact_source_order:
        block_matrices = []
        for entries, _ in signatures:
            matrices = tuple(
                _source_block_matrix(block)
                for block in _attention_source_blocks(
                    entries, class_period))
            if any(matrix is None for matrix in matrices):
                block_matrices = []
                break
            block_matrices.append(matrices)
        if (block_matrices
                and len({len(matrices) for matrices in block_matrices}) == 1):
            states = np.tile(
                np.asarray(initial, dtype=np.float64),
                (len(signatures), 1))
            for block_index in range(len(block_matrices[0])):
                matrices = np.stack(tuple(
                    lane[block_index] for lane in block_matrices))
                states = np.max(
                    matrices + states[:, None, :], axis=2)
            batched_states = states

    for signature_index, (entries, multiplicity) in enumerate(signatures):
        if batched_states is not None:
            state = tuple(batched_states[signature_index])
        else:
            state = (
                _apply_repeated_source_blocks(entries, initial)
                if compact_source_order
                else _apply_entries_exact(entries, initial))
        key = f"signature{signature_index}_x{multiplicity}"
        makespans[key] = _add_resident_tail(
            max(state[:constant_index]), entries)
        entries_by_key[key] = entries

    if compact_source_order:
        approximate_maximum = max(makespans.values())
        tolerance = (
            1024 * max(len(entries) for entries, _ in signatures)
            * math.ulp(max(1.0, abs(approximate_maximum))))
        for key, makespan in tuple(makespans.items()):
            if approximate_maximum - makespan > tolerance:
                continue
            entries = entries_by_key[key]
            exact_state = _apply_entries_exact(entries, initial)
            makespans[key] = _add_resident_tail(
                max(exact_state[:constant_index]), entries)
    execution_time = max(makespans.values())

    stage_service = {}
    for identity, phase in phase_by_identity.items():
        multiplier = multiplier_by_phase_identity[identity]
        for stage in phase.stages:
            stage_service[stage.name] = (
                stage_service[stage.name]
                + multiplier * stage.duration_s
                if stage.name in stage_service
                else multiplier * stage.duration_s)
    return PipelineDiagnostics(
        execution_time_s=execution_time,
        initiation_interval_s=maximum_initiation_interval,
        worker_makespan_s=makespans,
        stage_service_s=stage_service,
        critical_path=(
            "attention_envelope/startup",
            "attention_envelope/steady_state",
            "attention_envelope/tail",
        ),
        tasks=(),
    )


def evaluate_barrier_pipeline_envelope(
        envelope: BarrierPipelineEnvelope) -> PipelineDiagnostics:
    """Evaluate independent tile classes without materializing CTAs.

    Every class has a source-proven full resource barrier at its entry.  Its
    duration can therefore be evaluated once, then summed exactly according to
    the physical class histogram of every worker.
    """

    if not isinstance(envelope, BarrierPipelineEnvelope):
        raise TypeError(
            "evaluate_barrier_pipeline_envelope requires "
            "BarrierPipelineEnvelope")
    class_count = len(envelope.classes)
    if not class_count:
        raise ValueError("barrier pipeline envelope has no classes")
    class_durations = (
        envelope.class_durations_s
        if envelope.class_durations_s else tuple(
            _evaluate_barrier_pipeline_sequence(sequence, envelope.buffers)
            for sequence in envelope.classes))
    class_totals = tuple(
        sum(counts[index] for counts in envelope.counts_by_worker)
        for index in range(class_count))
    worker_makespan = {
        str(worker): math.fsum(
            count * class_durations[index]
            for index, count in enumerate(counts))
        for worker, counts in enumerate(envelope.counts_by_worker)
    }
    execution_time = max(worker_makespan.values())
    stage_service = {}
    if envelope.class_stage_service_s:
        for services, total in zip(
                envelope.class_stage_service_s, class_totals):
            for name, duration in services:
                stage_service[name] = (
                    stage_service[name] + total * duration
                    if name in stage_service else total * duration)
    else:
        for sequence, total in zip(envelope.classes, class_totals):
            for task in sequence.tasks:
                multiplier = total * task.repeat_count * sequence.repeat_count
                for stage in task.stages:
                    stage_service[stage.name] = (
                        stage_service[stage.name]
                        + multiplier * stage.duration_s
                        if stage.name in stage_service
                        else multiplier * stage.duration_s)
    critical_class = max(
        range(class_count), key=lambda index: class_durations[index])
    return PipelineDiagnostics(
        execution_time_s=execution_time,
        initiation_interval_s=0.0,
        worker_makespan_s=worker_makespan,
        stage_service_s=stage_service,
        critical_path=(
            "barrier_pipeline_envelope/"
            f"{envelope.classes[critical_class].name}",),
        tasks=(),
    )


def _evaluate_barrier_pipeline_sequence(sequence, buffers):
    """Return one full-resource-barrier tile class makespan exactly.

    Barrier envelopes run classes independently and then sum their complete
    worker-local spans.  A direct local replay avoids constructing a dense
    max-plus plan for every tile class while retaining each source stage,
    dependency, buffer credit, and compressed K-loop repetition.
    """

    resource_completion = {}
    buffer_completion = {
        buffer.name: [_NEGATIVE_INFINITY] * buffer.slots
        for buffer in buffers}
    for task in sequence.tasks:
        for _ in range(task.repeat_count):
            prior_resource_completion = dict(resource_completion)
            stage_start = {}
            stage_end = {}
            old_buffer_completion = {
                name: tuple(values)
                for name, values in buffer_completion.items()}
            for stage_index, stage in enumerate(task.stages):
                if stage.resource in resource_completion:
                    resource_start = resource_completion[stage.resource]
                else:
                    resource_start = 0.0
                candidates = [resource_start]
                predecessors = _stage_predecessors(task, stage_index)
                if not predecessors:
                    candidates.append(0.0)
                else:
                    candidates.extend(
                        stage_end[name] for name in predecessors)
                candidates.extend(
                    stage_start[name]
                    for name in _stage_start_predecessors(task, stage_index))
                for name in _stage_prior_resources(task, stage_index):
                    if name in prior_resource_completion:
                        candidates.append(prior_resource_completion[name])
                    else:
                        candidates.append(0.0)
                for buffer in buffers:
                    if buffer.producer_stage == stage.name:
                        candidates.append(old_buffer_completion[
                            buffer.name][0])
                start = max(candidates)
                end = start + stage.duration_s
                stage_start[stage.name] = start
                stage_end[stage.name] = end
                resource_completion[stage.resource] = (
                    start + stage.occupancy_s)
            for buffer in buffers:
                if buffer.consumer_stage not in stage_end:
                    continue
                values = old_buffer_completion[buffer.name]
                buffer_completion[buffer.name] = list(values[1:]) + [
                    (stage_start[buffer.consumer_stage]
                     if buffer.release == "start"
                     else stage_end[buffer.consumer_stage])]
    if not resource_completion:
        raise RuntimeError("barrier pipeline class has no resource stages")
    return max(resource_completion.values())


def _maxplus_power_matrix(powers, exponent, size):
    result = None
    bit = 0
    remaining = exponent
    while remaining > 0:
        if remaining & 1:
            result = (
                powers[bit] if result is None
                else _maxplus_product(powers[bit], result))
        bit += 1
        remaining >>= 1
    return _maxplus_identity(size) if result is None else result


def _maxplus_power_apply(powers, exponent, vector):
    output = tuple(vector)
    bit = 0
    remaining = exponent
    while remaining > 0:
        if remaining & 1:
            output = _maxplus_apply(powers[bit], output)
        bit += 1
        remaining >>= 1
    return output


def apply_repeated_maxplus_transition(transition, repeat_count, state):
    """Advance a fixed-size analytical recurrence without loop expansion."""

    if (isinstance(repeat_count, bool)
            or not isinstance(repeat_count, int)
            or repeat_count < 0):
        raise ValueError("max-plus repeat count must be a non-negative integer")
    if repeat_count == 0:
        return tuple(state)
    if not isinstance(transition, tuple) or not transition:
        raise TypeError("max-plus transition must be a non-empty tuple")
    size = len(transition)
    if len(state) != size or any(
            not isinstance(row, tuple) or len(row) != size
            for row in transition):
        raise ValueError("max-plus transition and state dimensions disagree")
    return _maxplus_power_apply(
        _maxplus_powers(transition, repeat_count), repeat_count, tuple(state))


def apply_maxplus_transition_programs(programs, states):
    """Advance independent same-size max-plus programs in one array batch.

    Each program is an ordered tuple of ``(transition, repeat_count)`` pairs.
    Binary powers retain the exact logarithmic recurrence used by
    :func:`apply_repeated_maxplus_transition`; only independent worker states
    share one vectorized reduction.
    """

    if not isinstance(programs, tuple) or not isinstance(states, tuple):
        raise TypeError("max-plus programs and states must be tuples")
    if len(programs) != len(states):
        raise ValueError("max-plus program and state counts differ")
    if not programs:
        return ()
    state_size = None
    power_programs = []
    validated_transition_ids = set()
    for program_index, (program, state) in enumerate(
            zip(programs, states, strict=True)):
        if not isinstance(program, tuple) or not isinstance(state, tuple):
            raise TypeError("max-plus programs and states must contain tuples")
        if state_size is None:
            state_size = len(state)
            if state_size <= 0:
                raise ValueError("max-plus states must be non-empty")
        elif len(state) != state_size:
            raise ValueError("max-plus program state dimensions disagree")
        power_sequence = []
        for transition, repeat_count in program:
            if (isinstance(repeat_count, bool)
                    or not isinstance(repeat_count, int)
                    or repeat_count < 0):
                raise ValueError(
                    "max-plus program repeat counts must be non-negative integers")
            if repeat_count == 0:
                continue
            transition_id = id(transition)
            if transition_id not in validated_transition_ids:
                if (not isinstance(transition, tuple)
                        or len(transition) != state_size
                        or any(not isinstance(row, tuple)
                               or len(row) != state_size
                               for row in transition)):
                    raise ValueError(
                        "max-plus program transition and state dimensions "
                        "disagree")
                validated_transition_ids.add(transition_id)
            powers = _maxplus_powers(transition, repeat_count)
            bit = 0
            remaining = repeat_count
            while remaining > 0:
                if remaining & 1:
                    power_sequence.append(powers[bit])
                bit += 1
                remaining >>= 1
        power_programs.append(tuple(power_sequence))
    state_array = np.ascontiguousarray(states, dtype=np.float64)
    positions = [0] * len(power_programs)
    while True:
        active = tuple(
            index for index, program in enumerate(power_programs)
            if positions[index] < len(program))
        if not active:
            break
        matrices = np.stack(tuple(
            _maxplus_matrix_array(power_programs[index][positions[index]])
            for index in active))
        vectors = state_array[np.asarray(active)]
        state_array[np.asarray(active)] = np.max(
            matrices + vectors[:, np.newaxis, :], axis=2)
        for index in active:
            positions[index] += 1
    return tuple(tuple(row.tolist()) for row in state_array)


def compose_maxplus_transitions(transitions):
    """Compose an ordered finite program of fixed-size max-plus steps."""

    if not isinstance(transitions, tuple) or not transitions:
        raise TypeError(
            "max-plus transition program must be a non-empty tuple")
    size = len(transitions[0])
    if size <= 0:
        raise ValueError("max-plus transition dimension must be positive")
    for transition in transitions:
        if (not isinstance(transition, tuple)
                or len(transition) != size
                or any(not isinstance(row, tuple) or len(row) != size
                       for row in transition)):
            raise ValueError(
                "max-plus transition program dimensions disagree")
    result = transitions[0]
    for transition in transitions[1:]:
        result = _maxplus_product(transition, result)
    return result


def _expression_basis(size, index):
    values = [_NEGATIVE_INFINITY] * size
    values[index] = 0.0
    return tuple(values)


def _expression_max(expressions):
    return tuple(max(values) for values in zip(*expressions))




def _state_layout(plan):
    resource_index = {
        resource.name: index
        for index, resource in enumerate(plan.resources)
    }
    next_index = len(resource_index)
    buffer_indexes = {}
    for buffer in plan.buffers:
        indexes = tuple(range(next_index, next_index + buffer.slots))
        buffer_indexes[buffer.name] = indexes
        next_index += buffer.slots
    constant_index = next_index
    return resource_index, buffer_indexes, constant_index


def _task_transition_matrix(
        task, plan, resource_index, buffer_indexes, constant_index):
    """Build the exact max-plus recurrence for one logical task."""

    return _compile_pipeline_transition(
        task.stages,
        resource_index,
        buffer_indexes,
        _pipeline_buffer_topology_signature(plan.buffers),
        constant_index,
    )


def _advance_one_task(
        task, state, plan, resource_index, buffer_indexes, constant_index):
    """Advance one logical task and retain its concrete stage timestamps."""

    output = list(state)
    prior_resource_state = tuple(output)
    stage_start = {}
    stage_end = {}
    old_buffer_values = {
        buffer.name: tuple(output[index] for index in buffer_indexes[buffer.name])
        for buffer in plan.buffers
    }
    for stage_index, stage in enumerate(task.stages):
        candidates = [output[resource_index[stage.resource]]]
        predecessors = _stage_predecessors(task, stage_index)
        if not predecessors:
            candidates.append(output[constant_index])
        else:
            candidates.extend(stage_end[name] for name in predecessors)
        candidates.extend(
            stage_start[name]
            for name in _stage_start_predecessors(task, stage_index))
        candidates.extend(
            prior_resource_state[resource_index[name]]
            for name in _stage_prior_resources(task, stage_index))
        for buffer in plan.buffers:
            if buffer.producer_stage == stage.name:
                candidates.append(old_buffer_values[buffer.name][0])
        start = max(candidates)
        end = start + stage.duration_s
        stage_start[stage.name] = start
        stage_end[stage.name] = end
        output[resource_index[stage.resource]] = start + stage.occupancy_s
    for buffer in plan.buffers:
        if buffer.consumer_stage not in stage_end:
            continue
        indexes = buffer_indexes[buffer.name]
        shifted = list(old_buffer_values[buffer.name][1:])
        shifted.append(
            stage_start[buffer.consumer_stage]
            if buffer.release == "start"
            else stage_end[buffer.consumer_stage])
        for index, value in zip(indexes, shifted):
            output[index] = value
    output[constant_index] = 0.0
    return tuple(output), stage_start, stage_end


def _evaluate_repeated_pipeline(plan: PipelinePlan) -> PipelineDiagnostics:
    """Evaluate repeated worker-local tasks without expanding logical tiles."""

    non_worker_resources = {
        resource.name for resource in plan.resources
        if resource.scope != "worker" or resource.slots != 1
    }
    non_worker_buffers = {
        buffer.name for buffer in plan.buffers
        if buffer.scope != "worker"
    }
    if non_worker_resources or non_worker_buffers:
        raise ValueError(
            "repeated pipeline tasks require one-slot worker-local resources "
            "and worker-local buffers: "
            f"resources={sorted(non_worker_resources)}, "
            f"buffers={sorted(non_worker_buffers)}")

    resource_index, buffer_indexes, constant_index = _state_layout(plan)
    initial_state = [0.0] * (constant_index + 1)
    for name, duration_s in plan.initial_resource_availability_s:
        initial_state[resource_index[name]] = duration_s
    for indexes in buffer_indexes.values():
        for index in indexes:
            initial_state[index] = _NEGATIVE_INFINITY
    initial_state[constant_index] = 0.0
    worker_state = {
        worker: tuple(initial_state) for worker in range(plan.worker_count)
    }
    worker_makespan = {
        str(worker): 0.0 for worker in range(plan.worker_count)
    }
    first_stage_first_start = {}
    first_stage_last_start = {}
    worker_logical_tasks = {worker: 0 for worker in range(plan.worker_count)}
    stage_service = {}
    task_diagnostics = []
    terminal_task = None
    terminal_end = -1.0
    transition_cache = {}
    power_cache = {}
    task_matrix_cache = {}
    pattern_matrix_cache = {}
    sequence_matrix_cache = {}
    sequence_sparse_rows_cache = {}
    sequence_power_cache = {}
    program_trie_roots = {}
    program_trie_edges = {}
    next_program_node_id = 0
    program_count_cache = {}

    task_key_cache = {}

    def task_key(task):
        cache_key = id(task.stages)
        if cache_key in task_key_cache:
            source_stages, key = task_key_cache[cache_key]
            if source_stages is not task.stages:
                raise RuntimeError("pipeline task-key identity cache collision")
            return key
        key = tuple(
            (
                stage.name, stage.resource, stage.duration_s,
                stage.occupancy_s,
                _stage_predecessors(task, stage_index),
                _stage_start_predecessors(task, stage_index),
                _stage_prior_resources(task, stage_index),
            )
            for stage_index, stage in enumerate(task.stages))
        task_key_cache[cache_key] = (task.stages, key)
        return key

    def task_powers(task, exponent):
        key = task_key(task)
        if key not in transition_cache:
            transition_cache[key] = _task_transition_matrix(
                task, plan, resource_index, buffer_indexes, constant_index)
        if (key not in power_cache
                or len(power_cache[key]) < exponent.bit_length()):
            power_cache[key] = _maxplus_powers(
                transition_cache[key], exponent)
        return power_cache[key]

    def task_run_matrix(task):
        key = (task_key(task), task.repeat_count)
        if key in task_matrix_cache:
            return task_matrix_cache[key]
        transition_key = key[0]
        if transition_key not in transition_cache:
            transition_cache[transition_key] = _task_transition_matrix(
                task, plan, resource_index, buffer_indexes, constant_index)
        if task.repeat_count == 1:
            matrix = transition_cache[transition_key]
        else:
            matrix = _maxplus_power_matrix(
                task_powers(task, task.repeat_count),
                task.repeat_count,
                constant_index + 1,
            )
        task_matrix_cache[key] = matrix
        return matrix

    def pattern_key(pattern):
        return tuple(
            (task_key(task), task.repeat_count) for task in pattern)

    def sequence_key(sequence):
        return (pattern_key(sequence.tasks), sequence.repeat_count)

    def cached_sequence_powers(key, matrix, exponent):
        required = exponent.bit_length()
        if key not in sequence_power_cache:
            sequence_power_cache[key] = (matrix,)
        if len(sequence_power_cache[key]) < required:
            powers = list(sequence_power_cache[key])
            current = powers[-1]
            while len(powers) < required:
                current = _maxplus_product(current, current)
                powers.append(current)
            sequence_power_cache[key] = tuple(powers)
        return sequence_power_cache[key]

    def advance_task_run(task, state):
        state_after_first, first_start, first_end = _advance_one_task(
            task, state, plan, resource_index, buffer_indexes,
            constant_index)
        if task.repeat_count == 1:
            return state_after_first, first_start, first_start, first_end
        powers = task_powers(task, task.repeat_count - 1)
        state_before_last = _maxplus_power_apply(
            powers, task.repeat_count - 1, state)
        state_after, last_start, last_end = _advance_one_task(
            task, state_before_last, plan, resource_index, buffer_indexes,
            constant_index)
        return state_after, first_start, last_start, last_end

    def pattern_matrix(pattern):
        key = pattern_key(pattern)
        if key in pattern_matrix_cache:
            return pattern_matrix_cache[key]
        result = None
        for task in pattern:
            task_matrix = task_run_matrix(task)
            result = (
                task_matrix if result is None
                else _maxplus_product(task_matrix, result))
        if result is None:
            raise RuntimeError("pipeline pattern has no logical task")
        pattern_matrix_cache[key] = result
        return result

    def sequence_matrix(sequence):
        key = sequence_key(sequence)
        if key in sequence_matrix_cache:
            return sequence_matrix_cache[key]
        matrix = pattern_matrix(sequence.tasks)
        if sequence.repeat_count > 1:
            powers = cached_sequence_powers(
                ("sequence_body", key), matrix, sequence.repeat_count)
            matrix = _maxplus_power_matrix(
                powers, sequence.repeat_count, constant_index + 1)
        sequence_matrix_cache[key] = matrix
        return matrix

    def sequence_sparse_rows(sequence, matrix):
        key = sequence_key(sequence)
        if key not in sequence_sparse_rows_cache:
            sequence_sparse_rows_cache[key] = _maxplus_sparse_rows(matrix)
        return sequence_sparse_rows_cache[key]

    def advance_pattern(pattern, state):
        output = state
        diagnostics = []
        first_logical_start = None
        last_logical_start = None
        for task in pattern:
            output, first_start, last_start, last_end = advance_task_run(
                task, output)
            first_stage = task.stages[0].name
            if first_logical_start is None:
                first_logical_start = first_start[first_stage]
            last_logical_start = last_start[first_stage]
            diagnostics.append((task, first_start, last_end))
        if first_logical_start is None or last_logical_start is None:
            raise RuntimeError("pipeline sequence has no logical task")
        return (
            output, tuple(diagnostics),
            first_logical_start, last_logical_start)

    def advance_sequence(sequence, state):
        """Advance one complete sequence occurrence with exact endpoints."""

        if sequence.repeat_count == 1:
            output, diagnostics, first_start, last_start = advance_pattern(
                sequence.tasks, state)
            return (
                output, diagnostics, diagnostics, first_start, last_start)
        matrix = pattern_matrix(sequence.tasks)
        powers = cached_sequence_powers(
            ("sequence_endpoint", pattern_key(sequence.tasks)),
            matrix,
            sequence.repeat_count - 1,
        )
        _, first_diagnostics, first_start, _ = advance_pattern(
            sequence.tasks, state)
        state_before_last_pattern = _maxplus_power_apply(
            powers, sequence.repeat_count - 1, state)
        output, last_diagnostics, _, last_start = advance_pattern(
            sequence.tasks, state_before_last_pattern)
        return (
            output, first_diagnostics, last_diagnostics,
            first_start, last_start)

    def ordered_program_matrix(
            pattern_keys, order, pattern_matrices, pattern_sparse_rows):
        nonlocal next_program_node_id
        if pattern_keys not in program_trie_roots:
            next_program_node_id += 1
            program_trie_roots[pattern_keys] = next_program_node_id
        node_id = program_trie_roots[pattern_keys]
        matrix = None
        for pattern_index in order:
            edge = (node_id, pattern_index)
            if edge in program_trie_edges:
                node_id, matrix = program_trie_edges[edge]
                continue
            next_matrix = pattern_matrices[pattern_index]
            if matrix is None:
                matrix = next_matrix
            else:
                matrix = _maxplus_left_sparse_product(
                    pattern_sparse_rows[pattern_index], matrix)
            next_program_node_id += 1
            node_id = next_program_node_id
            program_trie_edges[edge] = (node_id, matrix)
        if matrix is None:
            raise RuntimeError("pipeline program has no sequence occurrence")
        return (pattern_keys, node_id), matrix

    def apply_program_power(matrix_key, matrix, exponent, state):
        if exponent == 0:
            return state
        powers = cached_sequence_powers(
            ("program_repeat", matrix_key), matrix, exponent)
        return _maxplus_power_apply(powers, exponent, state)

    def apply_program_prefix(
            pattern_keys, order, pattern_matrices, pattern_sparse_rows, state):
        if not order:
            return state
        _, matrix = ordered_program_matrix(
            pattern_keys, order, pattern_matrices, pattern_sparse_rows)
        return _maxplus_apply(matrix, state)

    def advance_program_order(
            entry, order, pattern_matrices, pattern_sparse_rows, state):
        """Advance an order without retaining endpoint diagnostics."""

        run_start = 0
        while run_start < len(order):
            pattern_index = order[run_start]
            run_end = run_start + 1
            while (run_end < len(order)
                   and order[run_end] == pattern_index):
                run_end += 1
            run_length = run_end - run_start
            if run_length == 1:
                state = _maxplus_sparse_apply(
                    pattern_sparse_rows[pattern_index], state)
            else:
                powers = cached_sequence_powers(
                    (
                        "program_run",
                        sequence_key(entry.patterns[pattern_index]),
                    ),
                    pattern_matrices[pattern_index],
                    run_length,
                )
                state = _maxplus_power_apply(powers, run_length, state)
            run_start = run_end
        return state

    def advance_program_once(
            entry, order, pattern_matrices, pattern_sparse_rows, state):
        first_sequence = entry.patterns[order[0]]
        (
            state, first_diagnostics, first_last_diagnostics,
            first_start, first_last_start,
        ) = advance_sequence(first_sequence, state)
        if len(order) == 1:
            return (
                state, first_sequence, first_diagnostics,
                first_sequence, first_last_diagnostics,
                first_start, first_last_start)
        state = advance_program_order(
            entry, order[1:-1],
            pattern_matrices, pattern_sparse_rows, state)
        last_sequence = entry.patterns[order[-1]]
        state, _, last_diagnostics, _, last_start = advance_sequence(
            last_sequence, state)
        return (
            state, first_sequence, first_diagnostics,
            last_sequence, last_diagnostics, first_start, last_start)

    def advance_program_sparse(
            entry, pattern_matrices, pattern_sparse_rows, state):
        """Advance body repetitions and tail with only global endpoints."""

        first_sequence = entry.patterns[entry.order[0]]
        (
            state, first_diagnostics, _, first_start, _,
        ) = advance_sequence(first_sequence, state)
        state = advance_program_order(
            entry, entry.order[1:],
            pattern_matrices, pattern_sparse_rows, state)
        for repeat_index in range(1, entry.repeat_count):
            if (not entry.tail_order
                    and repeat_index == entry.repeat_count - 1):
                state = advance_program_order(
                    entry, entry.order[:-1],
                    pattern_matrices, pattern_sparse_rows, state)
                last_sequence = entry.patterns[entry.order[-1]]
                state, _, last_diagnostics, _, last_start = advance_sequence(
                    last_sequence, state)
                return (
                    state, first_sequence, first_diagnostics,
                    last_sequence, last_diagnostics, first_start, last_start)
            state = advance_program_order(
                entry, entry.order,
                pattern_matrices, pattern_sparse_rows, state)
        if entry.tail_order:
            state = advance_program_order(
                entry, entry.tail_order[:-1],
                pattern_matrices, pattern_sparse_rows, state)
            last_sequence = entry.patterns[entry.tail_order[-1]]
        else:
            last_sequence = entry.patterns[entry.order[-1]]
        state, _, last_diagnostics, _, last_start = advance_sequence(
            last_sequence, state)
        return (
            state, first_sequence, first_diagnostics,
            last_sequence, last_diagnostics, first_start, last_start)

    def dense_program_groups():
        """Select matrix execution only when shared structural cost is lower."""

        groups = {}
        for entry in plan.tasks:
            if (not isinstance(entry, PipelineTaskProgram)
                    or (entry.repeat_count == 1 and not entry.tail_order)):
                continue
            pattern_keys = tuple(
                sequence_key(sequence) for sequence in entry.patterns)
            if pattern_keys not in groups:
                groups[pattern_keys] = []
            groups[pattern_keys].append(entry)

        selected = set()
        state_size = constant_index + 1
        dense_apply_cost = state_size * state_size
        dense_product_cost = dense_apply_cost * state_size
        for pattern_keys, entries in groups.items():
            representative = entries[0]
            pattern_matrices = tuple(
                sequence_matrix(sequence)
                for sequence in representative.patterns)
            pattern_sparse_rows = tuple(
                sequence_sparse_rows(sequence, matrix)
                for sequence, matrix in zip(
                    representative.patterns, pattern_matrices))
            nonzero_counts = tuple(
                sum(len(row) for row in rows)
                for rows in pattern_sparse_rows)

            edges = {}
            next_node = 0
            dense_cost = 0
            dense_power_bits = {}
            sparse_cost = 0
            sparse_power_bits = {}

            def visit(order):
                nonlocal dense_cost, next_node
                node = 0
                for pattern_index in order:
                    edge = (node, pattern_index)
                    if edge not in edges:
                        next_node += 1
                        edges[edge] = next_node
                        if node != 0:
                            dense_cost += (
                                state_size * nonzero_counts[pattern_index])
                    node = edges[edge]
                return node

            def add_sparse_order_cost(entry, order, multiplicity):
                nonlocal sparse_cost
                if multiplicity <= 0 or not order:
                    return
                run_start = 0
                while run_start < len(order):
                    pattern_index = order[run_start]
                    run_end = run_start + 1
                    while (run_end < len(order)
                           and order[run_end] == pattern_index):
                        run_end += 1
                    run_length = run_end - run_start
                    if run_length == 1:
                        run_cost = nonzero_counts[pattern_index]
                    else:
                        run_cost = (
                            run_length.bit_count() * dense_apply_cost)
                        power_key = sequence_key(
                            entry.patterns[pattern_index])
                        current_power_bits = 0
                        if power_key in sparse_power_bits:
                            current_power_bits = sparse_power_bits[power_key]
                        sparse_power_bits[power_key] = max(
                            current_power_bits,
                            run_length.bit_length(),
                        )
                    sparse_cost += multiplicity * run_cost
                    run_start = run_end

            for entry in entries:
                full_node = visit(entry.order)
                if entry.tail_order:
                    prefix = entry.tail_order[:-1]
                    exponent = entry.repeat_count
                else:
                    prefix = entry.order[:-1]
                    exponent = entry.repeat_count - 1
                visit(prefix)
                if exponent > 0:
                    current_power_bits = 0
                    if full_node in dense_power_bits:
                        current_power_bits = dense_power_bits[full_node]
                    dense_power_bits[full_node] = max(
                        current_power_bits,
                        exponent.bit_length(),
                    )
                    dense_cost += exponent.bit_count() * dense_apply_cost
                if prefix:
                    dense_cost += dense_apply_cost

                add_sparse_order_cost(entry, entry.order[1:], 1)
                if entry.tail_order:
                    add_sparse_order_cost(
                        entry, entry.order, entry.repeat_count - 1)
                    add_sparse_order_cost(
                        entry, entry.tail_order[:-1], 1)
                else:
                    add_sparse_order_cost(
                        entry, entry.order,
                        max(entry.repeat_count - 2, 0))
                    add_sparse_order_cost(entry, entry.order[:-1], 1)

            dense_cost += sum(
                max(0, bit_length - 1) * dense_product_cost
                for bit_length in dense_power_bits.values())
            sparse_cost += sum(
                max(0, bit_length - 1) * dense_product_cost
                for bit_length in sparse_power_bits.values())
            if dense_cost < sparse_cost:
                selected.add(pattern_keys)
        return selected

    matrix_program_groups = dense_program_groups()

    for entry in plan.tasks:
        if isinstance(entry, PipelineTaskProgram):
            pattern_matrices = tuple(
                sequence_matrix(sequence) for sequence in entry.patterns)
            pattern_sparse_rows = tuple(
                sequence_sparse_rows(sequence, matrix)
                for sequence, matrix in zip(
                    entry.patterns, pattern_matrices))
            pattern_keys = tuple(
                sequence_key(sequence) for sequence in entry.patterns)
            state_before = worker_state[entry.worker]
            if entry.repeat_count == 1 and not entry.tail_order:
                (
                    state, first_sequence, first_diagnostics,
                    last_sequence, last_diagnostics, first_start, last_start,
                ) = advance_program_once(
                    entry, entry.order,
                    pattern_matrices, pattern_sparse_rows, state_before)
            elif pattern_keys in matrix_program_groups:
                first_sequence = entry.patterns[entry.order[0]]
                _, first_diagnostics, _, first_start, _ = advance_sequence(
                    first_sequence, state_before)
                matrix_key, matrix = ordered_program_matrix(
                    pattern_keys, entry.order,
                    pattern_matrices, pattern_sparse_rows)
                if entry.tail_order:
                    state_before_tail = apply_program_power(
                        matrix_key, matrix, entry.repeat_count, state_before)
                    last_sequence = entry.patterns[entry.tail_order[-1]]
                    state_before_last = apply_program_prefix(
                        pattern_keys, entry.tail_order[:-1],
                        pattern_matrices, pattern_sparse_rows,
                        state_before_tail)
                else:
                    state_before_last_program = apply_program_power(
                        matrix_key, matrix,
                        entry.repeat_count - 1, state_before)
                    last_sequence = entry.patterns[entry.order[-1]]
                    state_before_last = apply_program_prefix(
                        pattern_keys, entry.order[:-1],
                        pattern_matrices, pattern_sparse_rows,
                        state_before_last_program)
                state, _, last_diagnostics, _, last_start = advance_sequence(
                    last_sequence, state_before_last)
            else:
                (
                    state, first_sequence, first_diagnostics,
                    last_sequence, last_diagnostics, first_start, last_start,
                ) = advance_program_sparse(
                    entry, pattern_matrices, pattern_sparse_rows, state_before)
            worker_state[entry.worker] = state

            count_key = (
                len(entry.patterns), entry.order,
                entry.repeat_count, entry.tail_order)
            if count_key not in program_count_cache:
                pattern_counts = [0] * len(entry.patterns)
                for pattern_index in entry.order:
                    pattern_counts[pattern_index] += entry.repeat_count
                for pattern_index in entry.tail_order:
                    pattern_counts[pattern_index] += 1
                program_count_cache[count_key] = tuple(pattern_counts)
            logical_task_count = 0
            for pattern, count in zip(
                    entry.patterns, program_count_cache[count_key]):
                if count == 0:
                    continue
                for task in pattern.tasks:
                    multiplier = (
                        task.repeat_count * pattern.repeat_count * count)
                    logical_task_count += multiplier
                    aggregate_multiplier = (
                        multiplier * entry.worker_multiplicity)
                    for stage in task.stages:
                        stage_service[stage.name] = (
                            stage_service[stage.name]
                            + stage.duration_s * aggregate_multiplier
                            if stage.name in stage_service
                            else stage.duration_s * aggregate_multiplier)
            if entry.worker not in first_stage_first_start:
                first_stage_first_start[entry.worker] = first_start
            first_stage_last_start[entry.worker] = last_start
            worker_logical_tasks[entry.worker] += logical_task_count

            final_task, _, final_end = last_diagnostics[-1]
            task_end = max(final_end.values())
            worker_makespan[str(entry.worker)] = max(
                worker_makespan[str(entry.worker)], task_end)
            if task_end > terminal_end:
                terminal_end = task_end
                terminal_task = (
                    f"{entry.name}/{last_sequence.name}", final_task,
                    last_sequence.repeat_count)

            endpoint_diagnostics = [(
                first_sequence, first_diagnostics, "first")]
            if (len(entry.order) * entry.repeat_count
                    + len(entry.tail_order) > 1):
                endpoint_diagnostics.append((
                    last_sequence, last_diagnostics, "last"))
            for sequence, diagnostics, endpoint in endpoint_diagnostics:
                for task, task_start, task_end_map in diagnostics:
                    task_diagnostics.append(PipelineTaskDiagnostic(
                        task_name=(
                            f"{entry.name}/{endpoint}/"
                            f"{sequence.name}/{task.name}"),
                        worker=entry.worker,
                        stage_start_s=task_start,
                        stage_end_s=task_end_map,
                        repeat_count=(task.repeat_count * sequence.repeat_count),
                    ))
            continue
        if isinstance(entry, PipelineTaskSequence):
            pattern = entry.tasks
            outer_repeats = entry.repeat_count
            entry_name = entry.name
            worker = entry.worker
        else:
            pattern = (entry,)
            outer_repeats = 1
            entry_name = None
            worker = entry.worker
        state_before = worker_state[worker]
        if isinstance(entry, PipelineTaskSequence):
            (
                state_after, first_diagnostics, last_diagnostics,
                first_start, last_start,
            ) = advance_sequence(entry, state_before)
        else:
            state_after, first_diagnostics, first_start, last_start = (
                advance_pattern(pattern, state_before))
            last_diagnostics = first_diagnostics
        worker_state[worker] = state_after

        logical_tasks_per_pattern = sum(
            task.repeat_count for task in pattern)
        logical_task_count = logical_tasks_per_pattern * outer_repeats
        if worker not in first_stage_first_start:
            first_stage_first_start[worker] = first_start
        first_stage_last_start[worker] = last_start
        worker_logical_tasks[worker] += logical_task_count

        final_task, _, final_end = last_diagnostics[-1]
        task_end = max(final_end.values())
        worker_makespan[str(worker)] = max(
            worker_makespan[str(worker)], task_end)
        if task_end > terminal_end:
            terminal_end = task_end
            terminal_task = (entry_name, final_task, outer_repeats)

        for task in pattern:
            multiplier = task.repeat_count * outer_repeats
            for stage in task.stages:
                stage_service[stage.name] = (
                    stage_service[stage.name]
                    + stage.duration_s * multiplier
                    if stage.name in stage_service
                    else stage.duration_s * multiplier)
        for first_item, last_item in zip(
                first_diagnostics, last_diagnostics):
            task, first_task_start, _ = first_item
            last_task, _, last_task_end = last_item
            if task.name != last_task.name:
                raise RuntimeError(
                    "pipeline sequence diagnostic order changed")
            diagnostic_name = (
                task.name if entry_name is None
                else f"{entry_name}/{task.name}")
            task_diagnostics.append(PipelineTaskDiagnostic(
                task_name=diagnostic_name,
                worker=worker,
                stage_start_s=first_task_start,
                stage_end_s=last_task_end,
                repeat_count=task.repeat_count * outer_repeats,
            ))

    execution_time = max(worker_makespan.values())
    intervals = []
    for worker, task_count in worker_logical_tasks.items():
        if task_count < 2:
            continue
        intervals.append((
            first_stage_last_start[worker]
            - first_stage_first_start[worker]
        ) / (task_count - 1))
    initiation_interval = max(intervals) if intervals else 0.0
    if terminal_task is None:
        raise RuntimeError("repeated pipeline produced no terminal task")
    terminal_entry_name, terminal_logical_task, terminal_outer_repeats = (
        terminal_task)
    total_terminal_repeats = (
        terminal_logical_task.repeat_count * terminal_outer_repeats)
    base_label = (
        terminal_logical_task.name
        if terminal_entry_name is None
        else f"{terminal_entry_name}/{terminal_logical_task.name}")
    repeat_label = (
        base_label if total_terminal_repeats == 1
        else f"{base_label}[repeat={total_terminal_repeats}]")
    critical_path = tuple(
        f"{repeat_label}/{stage.name}"
        for stage in terminal_logical_task.stages)
    if not math.isfinite(execution_time):
        raise RuntimeError("repeated pipeline produced a non-finite makespan")
    return PipelineDiagnostics(
        execution_time_s=execution_time,
        initiation_interval_s=initiation_interval,
        worker_makespan_s=worker_makespan,
        stage_service_s=stage_service,
        critical_path=critical_path,
        tasks=tuple(task_diagnostics),
    )


def evaluate_pipeline(plan: PipelinePlan) -> PipelineDiagnostics:
    """Schedule explicit or compact repeated tile tasks exactly."""

    if not isinstance(plan, PipelinePlan):
        raise TypeError("evaluate_pipeline requires PipelinePlan")
    equivalent_program_plan = _equivalent_worker_program_plan(plan)
    if equivalent_program_plan is not None:
        compact_plan, physical_to_compact = equivalent_program_plan
        diagnostics = _evaluate_repeated_pipeline(compact_plan)
        diagnostics = dataclasses.replace(
            diagnostics,
            worker_makespan_s={
                str(worker): diagnostics.worker_makespan_s[
                    str(physical_to_compact[worker])]
                for worker in range(plan.worker_count)
            },
        )
        return _expand_physical_worker_diagnostics(diagnostics, plan)
    equivalent_sequence_plan = _equivalent_worker_sequence_plan(plan)
    if equivalent_sequence_plan is not None:
        compact_plan, physical_to_compact, entries_by_worker = (
            equivalent_sequence_plan)
        diagnostics = _evaluate_repeated_pipeline(compact_plan)
        return _expand_equivalent_worker_sequence_diagnostics(
            diagnostics, plan, physical_to_compact, entries_by_worker)
    if any(
            isinstance(task, (PipelineTaskSequence, PipelineTaskProgram))
            or task.repeat_count > 1
            for task in plan.tasks):
        return _expand_physical_worker_diagnostics(
            _evaluate_repeated_pipeline(plan), plan)
    return _evaluate_explicit_pipeline(plan)


def _expand_physical_worker_diagnostics(diagnostics, plan):
    """Restore launch worker order after source-proven class compression.

    A source program may materialize one physical prefix per worker (for
    example a launch-shared fill) and then compact only its repeatable suffix
    into equivalent worker classes.  Preserve diagnostics for the uncovered
    prefix-only workers while replacing every class-covered worker with its
    completed compact suffix makespan.
    """

    programs = tuple(
        entry for entry in plan.tasks
        if isinstance(entry, PipelineTaskProgram) and entry.physical_workers)
    if not programs:
        return diagnostics
    mapping = {}
    for program in programs:
        value = diagnostics.worker_makespan_s[str(program.worker)]
        for worker in program.physical_workers:
            if worker in mapping:
                raise ValueError(
                    "compressed pipeline worker classes overlap: "
                    f"worker={worker}")
            mapping[worker] = value
    uncovered_workers = set(range(plan.worker_count)) - set(mapping)
    missing_diagnostics = {
        worker for worker in uncovered_workers
        if str(worker) not in diagnostics.worker_makespan_s
    }
    if missing_diagnostics:
        raise ValueError(
            "compressed pipeline worker classes leave workers without "
            f"diagnostics: {sorted(missing_diagnostics)}")
    for worker in uncovered_workers:
        mapping[worker] = diagnostics.worker_makespan_s[str(worker)]
    return dataclasses.replace(
        diagnostics,
        worker_makespan_s={
            str(worker): mapping[worker]
            for worker in range(plan.worker_count)},
    )


def _equivalent_worker_program_plan(plan):
    """Collapse equivalent, independent worker programs exactly."""

    if len(plan.tasks) != plan.worker_count:
        return None
    if any(not isinstance(entry, PipelineTaskProgram) for entry in plan.tasks):
        return None
    workers = tuple(entry.worker for entry in plan.tasks)
    if set(workers) != set(range(plan.worker_count)):
        return None
    def signature(entry):
        return (
            tuple(
                (tuple(id(task.stages) for task in pattern.tasks),
                 pattern.repeat_count)
                for pattern in entry.patterns),
            entry.order,
            entry.repeat_count,
            entry.tail_order,
            entry.worker_multiplicity,
        )

    groups = {}
    for entry in plan.tasks:
        key = signature(entry)
        if key in groups:
            groups[key].append(entry)
        else:
            groups[key] = [entry]
    if len(groups) == len(plan.tasks):
        return None
    compact_entries = []
    physical_to_compact = {}
    for compact_worker, entries in enumerate(groups.values()):
        representative = entries[0]
        template_patterns = tuple(
            dataclasses.replace(
                pattern,
                worker=compact_worker,
                tasks=tuple(dataclasses.replace(task, worker=compact_worker)
                            for task in pattern.tasks),
            )
            for pattern in representative.patterns)
        compact_entries.append(PipelineTaskProgram(
            name=representative.name,
            worker=compact_worker,
            patterns=template_patterns,
            order=representative.order,
            repeat_count=representative.repeat_count,
            tail_order=representative.tail_order,
            worker_multiplicity=len(entries),
        ))
        for entry in entries:
            physical_to_compact[entry.worker] = compact_worker
    return PipelinePlan(
        worker_count=plan.worker_count,
        resources=plan.resources,
        tasks=tuple(compact_entries),
        buffers=plan.buffers,
    ), physical_to_compact


def _pipeline_task_signature(task):
    """Return schedule-relevant task fields, excluding source labels."""

    return (
        tuple(
            (
                stage.name,
                stage.resource,
                stage.duration_s,
                stage.occupancy_s,
                _stage_predecessors(task, stage_index),
                _stage_start_predecessors(task, stage_index),
                _stage_prior_resources(task, stage_index),
            )
            for stage_index, stage in enumerate(task.stages)),
        task.repeat_count,
    )


def _pipeline_sequence_signature(sequence):
    """Return the worker-local recurrence identity of one source sequence."""

    return (
        tuple(_pipeline_task_signature(task) for task in sequence.tasks),
        sequence.repeat_count,
    )


def _equivalent_worker_sequence_plan(plan):
    """Collapse matching source sequence schedules while retaining their labels.

    A source program may be represented as several ``PipelineTaskSequence``
    entries instead of one ``PipelineTaskProgram``.  When every worker has an
    identical sequence recurrence, the state evolution is independent across
    workers.  Evaluate one representative per class, then reconstruct the
    original source-labelled diagnostics below.
    """

    if not plan.tasks:
        return None
    if any(not isinstance(entry, PipelineTaskSequence) for entry in plan.tasks):
        return None
    entries_by_worker = {
        worker: [] for worker in range(plan.worker_count)
    }
    for entry in plan.tasks:
        entries_by_worker[entry.worker].append(entry)
    if any(not entries for entries in entries_by_worker.values()):
        return None

    groups = {}
    for worker, entries in entries_by_worker.items():
        signature = tuple(
            _pipeline_sequence_signature(entry) for entry in entries)
        if signature in groups:
            groups[signature].append(worker)
        else:
            groups[signature] = [worker]
    if len(groups) == plan.worker_count:
        return None

    compact_entries = []
    physical_to_compact = {}
    for compact_worker, workers in enumerate(groups.values()):
        representative_entries = entries_by_worker[workers[0]]
        compact_entries.extend(
            dataclasses.replace(
                entry,
                worker=compact_worker,
                tasks=tuple(
                    dataclasses.replace(task, worker=compact_worker)
                    for task in entry.tasks),
            )
            for entry in representative_entries)
        for worker in workers:
            physical_to_compact[worker] = compact_worker
    return (
        PipelinePlan(
            worker_count=len(groups),
            resources=plan.resources,
            tasks=tuple(compact_entries),
            buffers=plan.buffers,
        ),
        physical_to_compact,
        entries_by_worker,
    )


def _expand_equivalent_worker_sequence_diagnostics(
        diagnostics, plan, physical_to_compact, entries_by_worker):
    """Restore exact source-labelled diagnostics after worker compaction."""

    def relabel_task_diagnostic(diagnostic, *, task_name, worker):
        """Copy only source identity from an already immutable diagnostic.

        The compact evaluator constructed and validated ``diagnostic``.  Its
        timestamp maps are immutable contract mappings, so preserving them is
        equivalent to the former ``dataclasses.replace`` revalidation while
        avoiding a fresh mapping copy for every physical worker label.
        """

        relabeled = object.__new__(PipelineTaskDiagnostic)
        object.__setattr__(relabeled, "task_name", task_name)
        object.__setattr__(relabeled, "worker", worker)
        object.__setattr__(
            relabeled, "stage_start_s", diagnostic.stage_start_s)
        object.__setattr__(relabeled, "stage_end_s", diagnostic.stage_end_s)
        object.__setattr__(relabeled, "repeat_count", diagnostic.repeat_count)
        return relabeled

    compact_diagnostics_by_worker = {
        worker: []
        for worker in range(len(set(physical_to_compact.values())))
    }
    for diagnostic in diagnostics.tasks:
        compact_diagnostics_by_worker[diagnostic.worker].append(diagnostic)

    compact_entries_by_worker = {
        worker: [] for worker in compact_diagnostics_by_worker
    }
    representative_worker_by_compact = {}
    for worker, compact_worker in physical_to_compact.items():
        if compact_worker not in representative_worker_by_compact:
            representative_worker_by_compact[compact_worker] = worker
    for compact_worker in compact_entries_by_worker:
        compact_entries_by_worker[compact_worker] = entries_by_worker[
            representative_worker_by_compact[compact_worker]]

    entry_index_by_identity = {
        id(entry): index
        for entries in entries_by_worker.values()
        for index, entry in enumerate(entries)
    }

    diagnostics_by_entry = {}
    for compact_worker, compact_entries in compact_entries_by_worker.items():
        compact_diagnostics = compact_diagnostics_by_worker[compact_worker]
        offset = 0
        for entry in compact_entries:
            count = len(entry.tasks)
            entry_diagnostics = tuple(
                compact_diagnostics[offset:offset + count])
            if len(entry_diagnostics) != count:
                raise RuntimeError(
                    "equivalent worker sequence diagnostics are incomplete")
            diagnostics_by_entry[(compact_worker, entry.name)] = (
                entry_diagnostics)
            offset += count
        if offset != len(compact_diagnostics):
            raise RuntimeError(
                "equivalent worker sequence diagnostics have extra entries")

    source_diagnostics = []
    worker_makespan_s = {}
    stage_service_s = {}
    terminal_end = -1.0
    terminal_entry = None
    terminal_task = None
    for entry in plan.tasks:
        compact_worker = physical_to_compact[entry.worker]
        representative_worker = representative_worker_by_compact[
            compact_worker]
        representative_entries = entries_by_worker[representative_worker]
        entry_index = entry_index_by_identity[id(entry)]
        representative_entry = representative_entries[entry_index]
        entry_diagnostics = diagnostics_by_entry[(
            compact_worker, representative_entry.name)]
        for task, representative_task, diagnostic in zip(
                entry.tasks, representative_entry.tasks, entry_diagnostics):
            expected_name = (
                f"{representative_entry.name}/{representative_task.name}")
            if diagnostic.task_name != expected_name:
                raise RuntimeError(
                    "equivalent worker sequence diagnostic order changed")
            source_diagnostics.append(relabel_task_diagnostic(
                diagnostic,
                task_name=f"{entry.name}/{task.name}",
                worker=entry.worker,
            ))
            for stage in task.stages:
                multiplier = task.repeat_count * entry.repeat_count
                stage_service_s[stage.name] = (
                    stage_service_s[stage.name]
                    + stage.duration_s * multiplier
                    if stage.name in stage_service_s
                    else stage.duration_s * multiplier)
        final_diagnostic = entry_diagnostics[-1]
        entry_end = max(final_diagnostic.stage_end_s.values())
        if entry_end > terminal_end:
            terminal_end = entry_end
            terminal_entry = entry
            terminal_task = entry.tasks[-1]

    for worker in range(plan.worker_count):
        compact_worker = physical_to_compact[worker]
        worker_makespan_s[str(worker)] = diagnostics.worker_makespan_s[
            str(compact_worker)]
    if terminal_entry is None or terminal_task is None:
        raise RuntimeError("equivalent worker sequence pipeline has no terminal")
    total_terminal_repeats = (
        terminal_task.repeat_count * terminal_entry.repeat_count)
    base_label = f"{terminal_entry.name}/{terminal_task.name}"
    repeat_label = (
        base_label if total_terminal_repeats == 1
        else f"{base_label}[repeat={total_terminal_repeats}]")
    return PipelineDiagnostics(
        execution_time_s=max(worker_makespan_s.values()),
        initiation_interval_s=diagnostics.initiation_interval_s,
        worker_makespan_s=worker_makespan_s,
        stage_service_s=stage_service_s,
        critical_path=tuple(
            f"{repeat_label}/{stage.name}"
            for stage in terminal_task.stages),
        tasks=tuple(source_diagnostics),
    )
