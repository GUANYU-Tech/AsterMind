"""Per-Cell algorithm choices over one shared communication runtime.

The resolver changes plans, never resource IDs or runtime ownership. A task's
ordered participants and data layout remain immutable when selecting an algorithm.
"""
from dataclasses import replace
from astermind.builder.config.communication import (
    Scope, ALGORITHMS, DEFAULT_ALGORITHMS, CellCommPolicy, LAYERED_COLLECTIVES,
)


def resolve_task_policy(policy, task, routes):
    """Return (task, effective_policy, selection_metadata).

Cell tasks use policy/default algorithms exclusively. Legacy source-independent
tasks with no cell_id retain their explicit algorithm for backwards compatibility.
Without explicit su groups, single_su is only the same-satellite policy bucket:
it does NOT certify a nonblocking su fabric. Model eligibility must still validate
the real path before using whole-operation prediction.
"""
    if task.cell_id is not None and (not isinstance(task.cell_id, str) or not task.cell_id.strip()):
        raise ValueError('cell_id must be a nonempty string')
    if task.task_id is not None and (not isinstance(task.task_id, str) or not task.task_id.strip()):
        raise ValueError('task_id must be a nonempty string')
    cell_policy = policy.cell_policies.get(task.cell_id) if task.cell_id is not None else None
    effective = policy
    if cell_policy is not None:
        changes = {}
        if cell_policy.intra_mode is not None:
            changes['intra_mode'] = cell_policy.intra_mode
        effective = policy.model_copy(update=changes)
    from astermind.engine.comm.intra_hierarchy import classify_participants
    scope, hierarchy, _, domain_source = classify_participants(routes.frame, task.participants)
    effective = effective.model_copy()
    effective._intra_hierarchy = hierarchy
    metadata = dict(cell_id=task.cell_id, task_id=task.task_id, scope=scope,
                    domain_source=domain_source, intra_mode=effective.intra_mode)
    if scope != 'inter_satellite':
        hardware_default = (cell_policy.hardware_id if cell_policy else None) or policy.default_hardware_id
        metadata['hardware_by_gpu'] = {gpu: policy.gpu_hardware.get(gpu, hardware_default)
                                      for gpu in task.participants}
    if task.cell_id is None:
        algorithm = task.algorithm
        if scope != 'inter_satellite' and algorithm is None:
            algorithm = DEFAULT_ALGORITHMS.get(task.operation)
        metadata.update(algorithm=algorithm, algorithm_source='standalone_task')
        return replace(task, algorithm=algorithm), effective, metadata
    if task.operation not in DEFAULT_ALGORITHMS or (scope == 'inter_satellite' and task.operation not in ALGORITHMS):
        raise NotImplementedError(f'unsupported Comm operation: {task.operation}')
    algorithm = DEFAULT_ALGORITHMS[task.operation]
    source = 'comm_default'
    if cell_policy is not None and task.operation in cell_policy.algorithms.get(scope, {}):
        algorithm = cell_policy.algorithms[scope][task.operation]
        source = 'cell_policy'
    if (effective.intra_mode == 'surrogate' and scope != 'inter_satellite'
            and task.operation in LAYERED_COLLECTIVES):
        choices = {}
        for layer in ('single_su', 'intra_cross_su'):
            configured = cell_policy.algorithms.get(layer, {}).get(task.operation) if cell_policy else None
            explicit = configured is not None and configured != 'hierarchical'
            choices[layer] = dict(algorithm=configured if explicit else 'ring',
                                  source='cell_policy' if explicit else 'comm_default')
        metadata['layer_choices'] = choices
        chosen = choices[scope]
        algorithm, source = chosen['algorithm'], chosen['source']
    if algorithm == 'hierarchical' and task.traffic_matrix_bytes:
        raise NotImplementedError('hierarchical policy does not accept traffic matrices')
    # Source ring_order is an algorithm plan, not the semantic participant order.
    # The first version has no per-task override; user control is per Cell only.
    normalized = replace(task, algorithm=algorithm, ring_order=())
    metadata.update(algorithm=algorithm, algorithm_source=source)
    return normalized, effective, metadata
