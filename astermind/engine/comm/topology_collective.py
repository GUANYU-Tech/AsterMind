"""Lay out each layer's chosen collective algorithm on the physical routes."""
from collections import defaultdict
from itertools import permutations

from .frame_surrogate import NoRouteError
from .registered_profiles import route_scope


FULLMESH_ALIASES = frozenset(('fullmesh', 'fc2', 'mesh', 'direct', 'pairwise'))


def partitions(task):
    """Return each participant's original contribution or destination slice."""
    gather = task.operation in ('all_gather', 'all_gatherv')
    variable = task.operation.endswith('v')
    semantics = 'aggregate_input' if task.operation == 'all_gatherv' else 'per_rank_input'
    if (task.bytes_semantics != semantics or task.reduction != (None if gather else 'sum')
            or task.traffic_matrix_bytes or (task.partition_bytes and not variable)):
        raise ValueError(f'invalid byte or reduction contract for {task.operation}')
    if variable:
        sizes = task.partition_bytes
        if (len(sizes) != len(task.participants) or any(type(n) is not int or n < 0 for n in sizes)
                or sum(sizes) != task.bytes):
            raise ValueError(f'invalid partitions for {task.operation}')
    elif gather:
        sizes = (task.bytes,) * len(task.participants)
    else:
        chunk, remainder = divmod(task.bytes, len(task.participants))
        if remainder and task.operation == 'reduce_scatter':
            raise ValueError('reduce_scatter requires equal partitions')
        sizes = tuple(chunk + (i < remainder) for i in range(len(task.participants)))
    return dict(zip(task.participants, sizes))


def _walk_ring(neighbors, root):
    if any(len(peers) != 2 for peers in neighbors.values()):
        return ()
    order = [root, neighbors[root][0]]
    while len(order) < len(neighbors):
        next_rank = next(peer for peer in neighbors[order[-1]] if peer != order[-2])
        if next_rank in order:
            return ()
        order.append(next_rank)
    return tuple(order) if root in neighbors[order[-1]] else ()


def _spanning_tree(neighbors, root, scope):
    seen = {root}
    pending = [(root, 0)]
    levels = defaultdict(list)
    for parent, depth in pending:
        for child in neighbors[parent]:
            if child not in seen:
                seen.add(child)
                pending.append((child, depth + 1))
                levels[depth].append((parent, child))
    if len(seen) != len(neighbors):
        raise NoRouteError(f'{scope} collective participants have no connected bidirectional schedule')
    return tuple(levels.values())


def _schedule(ranks, scope, algorithm, get_route):
    if len(ranks) < 2:
        return 'local', ranks
    if algorithm in FULLMESH_ALIASES:
        return 'fullmesh', ranks
    if algorithm == 'ring' and all(get_route(a, b, scope) and get_route(b, a, scope)
                                    for a, b in zip(ranks, (*ranks[1:], ranks[0]))):
        return 'ring', ranks
    paths = {(a, b): get_route(a, b, scope) for a, b in permutations(ranks, 2)}
    neighbors = {a: [b for b in ranks if a != b and paths[a, b] and paths[b, a]]
                 for a in ranks}
    if algorithm == 'tree':
        return 'tree', _spanning_tree(neighbors, ranks[0], scope)
    ring = _walk_ring(neighbors, ranks[0])
    if not ring:
        raise NoRouteError(f'{scope} ring requires a reachable cycle through {ranks}')
    return 'ring', ring


def _gather_waves(algorithm, layout, sizes, direction=1):
    if algorithm == 'local':
        return []
    if algorithm == 'fullmesh':
        return [tuple((a, b, sizes[a]) for a, b in permutations(layout, 2))]
    if algorithm == 'ring':
        width = len(layout)
        return [tuple((a, layout[(i + direction) % width],
                       sizes[layout[(i - step * direction) % width]])
                      for i, a in enumerate(layout)) for step in range(width - 1)]
    totals = sizes.copy()
    up = []
    for level in reversed(layout):
        up.append(tuple((child, parent, totals[child]) for parent, child in level))
        for parent, child in level:
            totals[parent] += totals[child]
    total = sum(sizes.values())
    return up + [tuple((parent, child, total - totals[child]) for parent, child in level)
                 for level in layout]


def _stage_waves(plan, sizes, operation, direction):
    algorithm, layout = plan
    gather_direction = -direction if operation == 'reduce_scatter' else direction
    waves = _gather_waves(algorithm, layout, sizes, gather_direction)
    if operation == 'reduce_scatter':
        return [tuple((b, a, n) for a, b, n in wave) for wave in reversed(waves)]
    return waves


def layered_plan(task, groups, routes, hierarchy, choices):
    """Return required transfers and per-layer algorithm metadata."""
    chunks = partitions(task)
    cache = {}

    def get_route(a, b, scope, size=task.bytes):
        key = (a, b, size)
        if key not in cache:
            # Use the graph's actual routes. An absent GPU edge is not a relay.
            try:
                cache[key] = routes.route(a, b, size)
            except NoRouteError:
                cache[key] = None
        route = cache[key]
        if route and route_scope(routes, route, hierarchy) == scope:
            return route
        return None

    su_scope = 'single_su'
    so_scope = 'intra_cross_su'
    # Collapse rank-owned chunks to one chunk per source SU for gather, or
    # one chunk of all same-lane destinations for the first scatter phase.
    local_gather = [(group, {g: chunks[g] for g in group}) for group in groups]
    gather = task.operation in ('all_gather', 'all_gatherv')
    reduce = task.operation in ('reduce_scatter', 'reduce_scatterv')
    if len(groups) == 1:
        if gather:
            stages = [(su_scope, 'all_gather', local_gather, 1)]
        elif reduce:
            stages = [(su_scope, 'reduce_scatter', local_gather, 1)]
        else:
            stages = [(su_scope, 'reduce_scatter', local_gather, 1),
                      (su_scope, 'all_gather', local_gather, -1)]
    else:
        # A smaller SU can assign more than one lane to the same GPU.
        # Missing destination slices have zero bytes; its GPU still relays
        # that SU's input contribution for the other destinations in the lane.
        lanes = [tuple(group[i % len(group)] for group in groups)
                 for i in range(max(map(len, groups)))]
        if gather:
            totals = [sum(chunks[g] for g in group) for group in groups]
            cross_gather = [(lane, dict(zip(lane, totals))) for lane in lanes]
            stages = [(su_scope, 'all_gather', local_gather, 1),
                      (so_scope, 'all_gather', cross_gather, 1)]
        else:
            cross_reduce = [(lane, {g: chunks[g] if i < len(group) else 0
                                    for g, group in zip(lane, groups)})
                            for i, lane in enumerate(lanes)]
            lane_bytes = [sum(sizes.values()) for _, sizes in cross_reduce]
            local_reduce = []
            for group in groups:
                sizes = dict.fromkeys(group, 0)
                for i, size in enumerate(lane_bytes):
                    sizes[group[i % len(group)]] += size
                local_reduce.append((group, sizes))
            stages = [(su_scope, 'reduce_scatter', local_reduce, 1),
                      (so_scope, 'reduce_scatter', cross_reduce, 1)]
            if not reduce:
                stages += [(so_scope, 'all_gather', cross_reduce, -1),
                           (su_scope, 'all_gather', local_reduce, -1)]

    waves, metadata, schedules = [], [], {}
    for scope, operation, inputs, direction in stages:
        choice = choices[scope]
        phase, algorithms = [], []
        for ranks, sizes in inputs:
            key = (ranks, scope)
            if key not in schedules:
                schedules[key] = _schedule(ranks, scope, choice['algorithm'], get_route)
            schedule = schedules[key]
            algorithms.append(schedule[0])
            for index, wave in enumerate(_stage_waves(schedule, sizes, operation, direction)):
                if index == len(phase):
                    phase.append([])
                for a, b, size in wave:
                    if size:
                        route = get_route(a, b, scope, size)
                        if route is None:
                            raise NoRouteError(f'no {scope} path from {a} to {b} for {operation}')
                        phase[index].append((route, size))
        waves.extend(tuple(wave) for wave in phase)
        metadata.append(dict(scope=scope, operation=operation, algorithm=choice['algorithm'],
                             algorithm_source=choice['source'], algorithms=algorithms, rounds=len(phase)))
    return tuple(waves), metadata, schedules
