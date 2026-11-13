"""Per-Cell algorithm choices over one shared communication runtime.

The resolver changes plans, never resource IDs or runtime ownership. A task's
ordered participants and data layout remain immutable when selecting an algorithm.
"""
from dataclasses import replace
from astermind.builder.config.communication import Scope, ALGORITHMS, DEFAULT_ALGORITHMS, CellCommPolicy


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
    scope, hierarchy, full_groups, domain_source = classify_participants(routes.frame, task.participants)
    effective = effective.model_copy()
    effective._intra_hierarchy = hierarchy
    metadata = dict(cell_id=task.cell_id, task_id=task.task_id, scope=scope,
                    domain_source=domain_source, intra_mode=effective.intra_mode)
    if task.cell_id is None:
        metadata.update(algorithm=task.algorithm, algorithm_source='standalone_task')
        return task, effective, metadata
    if task.operation not in DEFAULT_ALGORITHMS:
        raise NotImplementedError(f'unsupported Comm operation: {task.operation}')
    algorithm = DEFAULT_ALGORITHMS[task.operation]
    source = 'comm_default'
    if (effective.intra_mode == 'surrogate' and scope == 'intra_cross_su' and full_groups and not task.partition_bytes
            and not task.traffic_matrix_bytes
            and task.operation in ('all_reduce', 'all_gather', 'all_to_all')):
        algorithm = 'hierarchical'
    if cell_policy is not None and task.operation in cell_policy.algorithms.get(scope, {}):
        algorithm = cell_policy.algorithms[scope][task.operation]
        source = 'cell_policy'
    if algorithm == 'hierarchical' and (task.partition_bytes or task.traffic_matrix_bytes):
        raise NotImplementedError('hierarchical policy does not accept variable partitions or traffic matrices')
    # Source ring_order is an algorithm plan, not the semantic participant order.
    # The first version has no per-task override; user control is per Cell only.
    normalized = replace(task, algorithm=algorithm, ring_order=())
    metadata.update(algorithm=algorithm, algorithm_source=source)
    return normalized, effective, metadata
