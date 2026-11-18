"""Uniform hierarchical collective admission and its explicit alternative."""
from itertools import permutations
from math import ceil

from .estimator import Estimate
from .frame_surrogate import NoRouteError


class HierarchyNotApplicable(ValueError):
    """Valid hierarchy that the uniform whole-model shortcut cannot price."""


def participating_groups(task, config):
    if config is None:
        raise ValueError('hierarchical requires SU membership')
    p = set(task.participants)
    groups = tuple(tuple(gpu for gpu in group if gpu in p) for group in config.su_groups)
    groups = tuple(group for group in groups if group)
    if len(groups) < 2 or len(set(map(len, groups))) != 1 or len(groups[0]) < 2:
        raise ValueError('hierarchical requires at least two equal participating SU groups')
    if set().union(*map(set, groups)) != p:
        raise ValueError('hierarchical participants must belong to a configured SU')
    return groups


def validate_task(task):
    if task.operation not in ('all_reduce', 'all_gather', 'all_to_all'):
        raise NotImplementedError('hierarchical supports uniform all_reduce/all_gather/all_to_all')
    if task.bytes_semantics != 'per_rank_input' or task.partition_bytes or task.traffic_matrix_bytes or task.ring_order:
        raise ValueError('hierarchical requires per_rank_input without a global ring order')
    if task.reduction != ('sum' if task.operation == 'all_reduce' else None):
        raise ValueError('invalid hierarchical reduction')


def _ring(groups, chunk, direction):
    return [
        tuple((gpu, g[(index + direction) % len(g)], chunk)
              for g in groups for index, gpu in enumerate(g))
        for _ in range(len(groups[0]) - 1)
    ]


def expansion(task, config):
    """Expand the same SU/SO stages as the whole model."""
    validate_task(task)
    groups = participating_groups(task, config)
    width = len(groups[0])
    stages = tuple(zip(*groups))
    size = task.bytes
    if task.operation == 'all_gather':
        return _ring(groups, size, 1) + _ring(stages, size * width, 1)
    chunk, extra = divmod(size, len(task.participants))
    if extra:
        raise ValueError('hierarchical equal partitions require divisible per-rank input')
    if task.operation == 'all_reduce':
        return (_ring(groups, size // width, 1) + _ring(stages, chunk, 1)
                + _ring(stages, chunk, -1) + _ring(groups, size // width, -1))
    return [
        tuple((a, b, chunk) for group in groups for a in group for b in task.participants
              if b != a and (b in group) == local)
        for local in (True, False)
    ]


def estimate_hierarchical(task, routes, config, profile=None, *, policy=None, selection=None):
    if task.algorithm != 'hierarchical':
        return None
    validate_task(task)
    groups = participating_groups(task, config)
    if len({routes.topology.devices[gpu].satellite for gpu in task.participants}) != 1:
        raise HierarchyNotApplicable('whole hierarchy cannot span satellites')
    if task.operation != 'all_gather' and task.bytes % len(task.participants):
        raise ValueError('hierarchical equal partitions require divisible per-rank input')
    if policy is None:
        from astermind.builder.config.communication import CommPolicy
        policy = CommPolicy(intra_profile=profile)
    from .registered_profiles import LINKS, gpu_port_bandwidth, p2p_timing, route_scope, upstream_theory
    theory = upstream_theory()
    cards = set(selection['hardware_by_gpu'].values()) if selection else {policy.default_hardware_id}
    if len(cards) != 1 or next(iter(cards)) not in LINKS:
        raise HierarchyNotApplicable('whole hierarchy requires homogeneous GPUs with a known SU profile')
    hardware = cards.pop()
    membership = {gpu: order for order, group in enumerate(groups) for gpu in group}
    paths_by_scope = {'single_su': [], 'intra_cross_su': []}
    try:
        for a, b in permutations(task.participants, 2):
            route = routes.route(a, b, task.bytes)
            scope = route_scope(routes, route, config)
            expected = 'single_su' if membership[a] == membership[b] else 'intra_cross_su'
            if scope != expected:
                raise HierarchyNotApplicable('peer path leaves its declared fabric')
            if scope == 'single_su' and any(edge['attributes'].get('comm_technology', LINKS[hardware]) != LINKS[hardware]
                                           for edge in routes.route_resources(route)):
                raise HierarchyNotApplicable('SU path technology differs from reference profile')
            if scope == 'intra_cross_su':
                for edge in routes.route_resources(route):
                    technology = edge['attributes'].get('comm_technology')
                    if technology is not None and technology not in (LINKS[hardware], 'roce', 'rocev2', 'infiniband'):
                        raise HierarchyNotApplicable('SO path technology differs from RoCE reference')
            paths_by_scope[scope].append(route)
    except NoRouteError as error:
        raise HierarchyNotApplicable(str(error)) from error

    width, domains = len(groups[0]), len(groups)
    data = task.bytes * len(task.participants) if task.operation == 'all_gather' else task.bytes
    so_size = task.bytes // width if task.operation == 'all_reduce' else task.bytes
    parameters = {}
    models = []
    for scope, size in (('single_su', task.bytes), ('intra_cross_su', so_size)):
        paths = paths_by_scope[scope]
        signatures = {(r['bandwidth_gbps'], r['latency_ns'], len(r['resources'])) for r in paths}
        if len(signatures) != 1:
            raise HierarchyNotApplicable(f'{scope} paths are not uniform')
        bandwidth, latency, _ = signatures.pop()
        timing, = p2p_timing(size, scope, (hardware,), policy)
        if timing['placeholder']:
            raise HierarchyNotApplicable('no registered whole SO profile for this GPU; use P2P placeholder')
        capacity = bandwidth * 125
        ports = [gpu_port_bandwidth(gpu, hardware, policy) for gpu in task.participants]
        rate = min(capacity, timing['rate'] or capacity, *(port or capacity for port in ports))
        fixed = max(latency / 1000, timing['phase_fixed_us'])
        models.append(theory.TheoryDerivedSurrogate(**dict(timing['parameters'], ring_bw_bytes_per_us=rate,
                                                    packet_bw_bytes_per_us=rate, link_latency_us=fixed,
                                                    wire_efficiency=timing['wire_efficiency'])))
        parameters[scope] = dict(protocol=timing['protocol'], profile_sha=timing['profile_sha'],
                                 startup_us=timing['startup_us'], phase_fixed_us=fixed,
                                 ring_bw_bytes_per_us=rate, wire_efficiency=timing['wire_efficiency'])
    startup = max(item['startup_us'] for item in parameters.values()) if data else 0.
    model = theory.HierarchicalSurrogate(*models, width, domains)
    duration = model.predict(data, 'ring', timing['credits'], startup_us=0.,
                             collective=task.operation.replace('_', '')) if data else 0.
    paths = tuple(path for options in paths_by_scope.values() for path in options)
    return Estimate(ceil((duration + startup) * 1000), 'intra_hierarchical', 0, 0, 0, 0, paths,
                    dict(hardware_id=hardware, domains=domains, gpus_per_domain=width,
                         upstream_data_bytes=data, startup_us=startup, effective_parameters=parameters))
