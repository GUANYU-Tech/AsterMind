"""Use Uni3Sim's pinned profile catalog with Comm's physical routes."""
from dataclasses import replace
from functools import lru_cache
from itertools import permutations
from math import ceil, isfinite
from types import SimpleNamespace

from .collective_plan import collective_waves
from .estimator import Estimate
from .frame_surrogate import NoRouteError

LINKS = dict(nvidia_a800='nvlink', nvidia_h200='nvlink',
             metax_c500='metaxlink', ascend_910b='hccs')
FULLMESH = {'metax_c500': 'fc2', 'ascend_910b': 'mesh'}


class ProfileMiss(Exception):
    """The task is valid but the whole-operation model does not cover it."""


def elapsed_ns(duration_us):
    if not isfinite(duration_us) or duration_us < 0:
        raise ValueError('invalid ProtoBridge duration')
    return ceil(duration_us * 1000)


@lru_cache(maxsize=1)
def _adapter():
    from astermind.common.bootstrap import activate
    activate()
    from src.kernel.predictors.protobridge import adapter
    return adapter


@lru_cache(maxsize=1)
def upstream_theory():
    _adapter()
    from src.kernel.predictors.protobridge.upstream import load_upstream_module_and_verify_license
    return load_upstream_module_and_verify_license()


@lru_cache(maxsize=128)
def registered(hardware, operation, algorithm, link):
    adapter = _adapter()
    digest = adapter.development_profile_config_sha256(hardware, operation, algorithm, link)
    return adapter.resolve_registered_profile(digest)


def route_scope(routes, route, hierarchy):
    path = route['path']
    if len({routes.topology.devices[node].satellite for node in path}) != 1:
        return 'inter_satellite'
    if hierarchy is None:
        nodes = getattr(getattr(routes, 'frame', None), 'node_states', {})
        if path[0] in nodes:
            hierarchy = nodes[path[0]].attributes.get('intra_hierarchy')
    groups = hierarchy.get('su_groups', ()) if isinstance(hierarchy, dict) else (
        hierarchy.su_groups if hierarchy else ())
    if groups and not any(path[0] in group and path[-1] in group for group in groups):
        return 'intra_cross_su'
    return 'single_su'


def _p2p_profile(hardware, scope):
    if scope == 'intra_cross_su':
        return registered('nvidia_h200', 'send_recv', 'send_recv', 'infiniband'), hardware != 'nvidia_h200'
    reference = hardware if hardware in LINKS else 'nvidia_h200'
    return registered(reference, 'send_recv', 'send_recv', LINKS[reference]), reference != hardware


def p2p_timing(size, scope, hardware, policy):
    """Resolve per-message timing on the actual layer, including placeholders."""
    cards = tuple(dict.fromkeys(hardware))
    timings = []
    for card in cards:
        profile, placeholder = _p2p_profile(card, scope)
        protocol, startup = profile.protocol_policy.select(max(1, size))
        fixed = profile.protocol_policy.phase_fixed_for_protocol(protocol)
        if profile.hardware_id == 'nvidia_h200' and profile.link == 'nvlink':
            # Estimate the H200 P2P phase cost using A800's ratio for this protocol.
            reference = registered('nvidia_a800', 'send_recv', 'send_recv', 'nvlink').protocol_policy
            fixed = startup * (reference.phase_fixed_for_protocol(protocol)
                               / reference.startup_for_protocol(protocol))
        parameters = dict(profile.model_parameters)
        overrides = (policy.resolved_intra_profile.so if scope == 'intra_cross_su'
                     else policy.resolved_intra_profile.su).overrides
        rate = overrides.ring_bw_bytes_per_us
        timings.append(dict(hardware=card, profile_hardware=profile.hardware_id,
                            link=profile.link, profile_sha=profile.config_sha256,
                            placeholder=placeholder, protocol=protocol,
                            startup_us=startup if overrides.startup_us is None else overrides.startup_us,
                            phase_fixed_us=fixed if overrides.phase_fixed_us is None else overrides.phase_fixed_us,
                            rate=rate, wire_efficiency=(parameters['wire_efficiency']
                                                        if overrides.wire_efficiency is None else overrides.wire_efficiency),
                            parameters=parameters, credits=profile.credits))
    return timings


@lru_cache(maxsize=128)
def nominal_gpu_bandwidth(hardware):
    if hardware not in LINKS:
        return None
    _adapter()
    from src.kernel.launch_config.communication_profile import development_protobridge_inputs_for
    inputs = development_protobridge_inputs_for(hardware, link=LINKS[hardware])
    return inputs['per_link_bytes_per_us'] * inputs['num_lanes']


def gpu_port_bandwidth(gpu, hardware, policy):
    override = policy.gpu_port_gbps.get(gpu)
    return override * 125 if override is not None else nominal_gpu_bandwidth(hardware)


def p2p_duration(routes, route, size, scope, hardware, policy, *, include_startup, wave_rate_limit=None):
    """Price a fabric SendRecv, with an optional standalone launch."""
    timings = p2p_timing(size, scope, hardware, policy)
    cap = route['bandwidth_gbps'] * 125
    if cap <= 0:
        raise NoRouteError('no positive capacity on intra-satellite path')
    # A heterogeneous transfer is constrained by both endpoints.
    limits = [cap, *(r['rate'] or cap for r in timings)]
    for gpu, card in zip((route['path'][0], route['path'][-1]), hardware):
        port = gpu_port_bandwidth(gpu, card, policy)
        if port is not None:
            limits.append(port)
    if wave_rate_limit is not None:
        limits.append(wave_rate_limit)
    rate = min(limits)
    fixed = max(route['latency_ns'] / 1000, *(r['phase_fixed_us'] for r in timings))
    startup = max(r['startup_us'] for r in timings) if include_startup else 0.
    wire_efficiency = min(r['wire_efficiency'] for r in timings)
    parameters = dict(timings[0]['parameters'], ring_bw_bytes_per_us=rate,
                      packet_bw_bytes_per_us=rate, link_latency_us=fixed,
                      wire_efficiency=wire_efficiency)
    model = upstream_theory().TheoryDerivedSurrogate(**parameters)
    duration = model.predict(size, 2, 'ring', timings[0]['credits'], startup_us=startup,
                             collective='sendrecv')
    details = dict(domain=scope, startup_us=startup, phase_fixed_us=fixed,
                   ring_bw_bytes_per_us=rate, wire_efficiency=wire_efficiency,
                   endpoint_profiles=[{k: v for k, v in timing.items() if k != 'parameters'}
                                      for timing in timings])
    return elapsed_ns(duration), details


def _required_pairs(task):
    ranks = task.ring_order or task.participants
    if task.algorithm == 'ring':
        if len(ranks) != len(task.participants) or set(ranks) != set(task.participants):
            raise ValueError('invalid ring order')
        pairs = tuple(zip(ranks, (*ranks[1:], ranks[0])))
        return pairs + tuple((b, a) for a, b in pairs) if task.operation == 'all_reduce' else pairs
    if task.algorithm == 'tree':
        waves = collective_waves(task.operation, 'tree', ranks, task.bytes)
        return tuple(dict.fromkeys((a, b) for wave in waves for a, b, _ in wave))
    return tuple(permutations(ranks, 2))


def _whole_parameters(task, profile, protocol, algorithm, capacity, latency, policy):
    phase_policy = profile.protocol_policy
    fixed = max(latency / 1000, phase_policy.phase_fixed_for_protocol(protocol))
    rate = phase_policy.ring_bandwidth_for(protocol, task.operation, len(task.participants), capacity)
    parameters = dict(profile.model_parameters, ring_bw_bytes_per_us=rate,
                      packet_bw_bytes_per_us=rate)
    if _whole_large_message(task, phase_policy):
        parameters['wire_efficiency'] = phase_policy.large_message_wire_efficiency
    overrides = policy.resolved_intra_profile.su.overrides.model_dump(exclude_none=True)
    startup = overrides.pop('startup_us', phase_policy.startup_for_protocol(protocol))
    fixed_override = overrides.pop('phase_fixed_us', None)
    fixed = max(latency / 1000, fixed if fixed_override is None else fixed_override)
    parameters.update(overrides, link_latency_us=fixed)
    parameters['packet_bw_bytes_per_us'] = parameters['ring_bw_bytes_per_us']
    if algorithm == 'tree':
        parameters['t_min_switch_us'] = max(latency / 1000, parameters['t_min_switch_us']
                                            if fixed_override is None else fixed_override)
        fixed = parameters['t_min_switch_us']
    for key in ('ring_bw_bytes_per_us', 'packet_bw_bytes_per_us', 'tree_bw_per_level_bytes_per_us',
                'fullmesh_bw_bytes_per_us'):
        parameters[key] = min(parameters[key], capacity)
    return parameters, startup, fixed


def _whole_large_message(task, phase_policy):
    cutoff = phase_policy.large_message_min_bytes
    return (cutoff is not None and task.operation == 'all_reduce'
            and len(task.participants) == 2 and task.bytes >= cutoff)


def bind_registered_ring(task, policy, card, bandwidth_gbps, latency_ns, port):
    """Price the registered ring after Fast has certified its routes and policy."""
    profile = registered(card, task.operation, 'ring', LINKS[card])
    prototype = replace(task, algorithm='ring')
    width = len(task.participants)
    capacity = min(bandwidth_gbps * 125, port)
    preset = profile.surrogate_topology
    topology = None if preset is None else upstream_theory().Topology(preset.bisection_bw_bytes_per_us,
                                                                     preset.hop_count)
    entries = {}

    def evaluate(size, _):
        if task.operation == 'all_reduce' and size % width:
            return None
        actual = replace(prototype, bytes=size)
        try:
            protocol = _adapter()._selected_vendor_protocol(actual, profile)
        except (ValueError, NotImplementedError):
            return None
        key = protocol, _whole_large_message(actual, profile.protocol_policy)
        if key not in entries:
            parameters, startup, _ = _whole_parameters(
                actual, profile, protocol, 'ring', capacity, latency_ns, policy)
            entries[key] = upstream_theory().TheoryDerivedSurrogate(**parameters), startup
        model, startup = entries[key]
        data_bytes = size * width if task.operation == 'all_gather' else size
        value = model.predict(data_bytes, width, 'ring', profile.credits, startup_us=startup,
                              topology=topology, collective=task.operation.replace('_', ''))
        return elapsed_ns(value) * 10**9

    return evaluate


def whole_registered(task, routes, policy, selection):
    if task.operation not in ('all_reduce', 'all_gather'):
        raise ProfileMiss('no registered whole profile for this operation')
    if selection['scope'] != 'single_su':
        raise ProfileMiss('registered whole profile requires one SU')
    if task.partition_bytes or task.traffic_matrix_bytes:
        raise ProfileMiss('registered whole profile requires uniform data')
    if task.bytes_semantics != 'per_rank_input' or task.reduction != (
            'sum' if task.operation == 'all_reduce' else None):
        raise ValueError('invalid uniform collective byte/reduction contract')
    if task.ring_order and task.algorithm != 'ring':
        raise ValueError('ring_order requires ring')
    if task.operation == 'all_reduce' and task.algorithm == 'ring' and task.bytes % len(task.participants):
        raise ProfileMiss('ring profile requires equal partitions')
    cards = set(selection['hardware_by_gpu'].values())
    if len(cards) != 1:
        raise ProfileMiss('registered whole profile requires homogeneous GPUs')
    hardware = cards.pop()
    algorithm = FULLMESH.get(hardware, 'fullmesh') if task.algorithm == 'fullmesh' else task.algorithm
    link = LINKS.get(hardware)
    if link is None:
        raise ProfileMiss(f'no registered SU profile for {hardware}')
    try:
        profile = registered(hardware, task.operation, algorithm, link)
        protocol = _adapter()._selected_vendor_protocol(task, profile)
    except (ValueError, NotImplementedError) as error:
        raise ProfileMiss(str(error)) from error

    # Required peer paths must describe one consistent fabric. A legacy switch
    # vertex may remain in the input graph, but there is no switch model.
    try:
        paths = tuple(routes.route(a, b, task.bytes) for a, b in _required_pairs(task))
    except NoRouteError as error:
        raise ProfileMiss(f'whole profile route unavailable: {error}') from error
    def uniform_path(route):
        if route_scope(routes, route, policy.intra_hierarchy) != 'single_su':
            return False
        intermediate = route['path'][1:-1]
        return (not intermediate or len(intermediate) == 1
                and routes.topology.devices[intermediate[0]].kind == 'switch')
    if not all(uniform_path(path) for path in paths):
        raise ProfileMiss('peer paths do not match one local fabric')
    signatures = {(r['bandwidth_gbps'], r['latency_ns'], tuple(r['path'][1:-1])) for r in paths}
    if len(signatures) != 1:
        raise ProfileMiss('peer paths are not uniform')
    # An explicitly annotated link can disambiguate a different local fabric.
    for path in paths:
        for edge in routes.route_resources(path):
            technology = edge['attributes'].get('comm_technology')
            if technology is not None and technology != link:
                raise ProfileMiss(f'path technology {technology} differs from profile {link}')
    cap, latency, _ = signatures.pop()
    capacity = cap * 125
    ports = [gpu_port_bandwidth(gpu, hardware, policy) for gpu in task.participants]
    capacity = min(capacity, *(port or capacity for port in ports))
    parameters, startup, fixed = _whole_parameters(task, profile, protocol, algorithm, capacity, latency, policy)
    if algorithm in ('fc2', 'mesh'):
        from collections import Counter
        peers = len(task.participants) - 1
        peer_rate = parameters['fullmesh_bw_bytes_per_us']
        if any(port is not None and peer_rate * peers > port for port in ports):
            raise ProfileMiss('whole FullMesh needs enough GPU bandwidth for its concurrent peers')
        # Use separate directions for full-duplex edges. A shared resource is
        # counted once per route and may be used by more than one peer flow.
        demand = Counter(edge['resource_key'] for route in paths for edge in routes.route_resources(route))
        capacities = {edge['resource_key']: edge['bandwidth_gbps'] * 125 for edge in routes.edges.values()}
        if any(count * peer_rate > capacities[resource] for resource, count in demand.items()):
            raise ProfileMiss('whole FullMesh peer paths share a physical link')
        if any(edge['shared_pools'] for route in paths for edge in routes.route_resources(route)):
            raise ProfileMiss('whole FullMesh peer paths use a shared capacity pool')
    model = upstream_theory().TheoryDerivedSurrogate(**parameters)
    preset = profile.surrogate_topology
    topology = None if preset is None else upstream_theory().Topology(preset.bisection_bw_bytes_per_us, preset.hop_count)
    data_bytes = task.bytes * len(task.participants) if task.operation == 'all_gather' else task.bytes
    upstream_algo = 'fullmesh' if algorithm in ('fc2', 'mesh') else algorithm
    duration = model.predict(data_bytes, len(task.participants), upstream_algo, profile.credits,
                             startup_us=startup, topology=topology,
                             collective=task.operation.replace('_', ''))
    return Estimate(elapsed_ns(duration), 'intra_' + upstream_algo, 0, 0, 0, 0, paths,
                    dict(hardware_id=hardware, registered_algorithm=algorithm, link=link,
                         protocol=protocol, profile_sha=profile.config_sha256, startup_us=startup,
                         phase_fixed_us=fixed, physical_path_capacity_gbps=cap,
                         upstream_data_bytes=data_bytes, effective_parameters=parameters))


def collective_startup(task, waves, routes, policy, selection):
    """One launch, priced at the slowest setup of the fabrics the plan uses."""
    if not waves or task.operation == 'send_recv' or not task.bytes:
        return 0, {}
    size = max(task.partition_bytes) if task.operation == 'all_gatherv' else task.bytes
    details = {}
    cards = selection['hardware_by_gpu']
    for wave in waves:
        for route, amount in wave:
            if not amount:
                continue
            scope = route_scope(routes, route, policy.intra_hierarchy)
            if policy.intra_mode != 'surrogate':
                continue
            a, b = route['path'][0], route['path'][-1]
            key = (scope, cards[a], cards[b])
            if key not in details:
                details[key] = max(p['startup_us'] for p in p2p_timing(size, scope, (cards[a], cards[b]), policy))
    breakdown = {}
    for (scope, _, _), startup in details.items():
        breakdown[scope] = max(breakdown.get(scope, 0.), startup)
    return elapsed_ns(max(breakdown.values(), default=0.)), breakdown
