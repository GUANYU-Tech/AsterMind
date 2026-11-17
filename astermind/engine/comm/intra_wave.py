"""Bandwidth sharing among local P2P transfers submitted in one collective wave."""
from collections import Counter

from .frame_surrogate import NoRouteError
from .registered_profiles import gpu_port_bandwidth, p2p_timing, route_scope


def local_wave_limits(routes, wave, policy, hardware, capacities):
    """Return per-transfer wire-rate limits in B/µs for a synchronized wave.

    Each shared resource budgets its total wire bytes once. Its service time
    is apportioned by bytes, so a common port cannot transmit at its full
    rate to every peer. Startup and fixed phase latency remain in SendRecv.
    """
    if len(wave) < 2 or policy.intra_mode != 'surrogate':
        return {}
    supply = {}
    loads = Counter()
    users = Counter()
    work = {}
    efficiencies = {}
    for index, (route, size) in enumerate(wave):
        if not size or route.get('unavailable'):
            continue
        scope = route_scope(routes, route, policy.intra_hierarchy)
        if scope == 'inter_satellite':
            continue
        source, target = route['path'][0], route['path'][-1]
        cards = (hardware[source], hardware[target])
        profile_key = (scope, cards)
        if profile_key not in efficiencies:
            efficiencies[profile_key] = min(t['wire_efficiency'] for t in
                                            p2p_timing(1, scope, cards, policy))
        wire_bytes = size / efficiencies[profile_key]
        resources = Counter()
        for endpoint, card, direction in zip((source, target), cards, ('tx', 'rx')):
            port = gpu_port_bandwidth(endpoint, card, policy)
            if port is not None:
                resource = ('gpu_port', endpoint, direction)
                resources[resource] = 1
                supply[resource] = port
        for edge in routes.route_resources(route):
            for resource in (edge['resource_key'], *(('pool', pool) for pool in edge['shared_pools'])):
                if resource in capacities:
                    resources[resource] += 1
                    supply[resource] = capacities[resource] * 1000
        for resource, traversals in resources.items():
            loads[resource] += traversals * wire_bytes
            users[resource] += 1
        work[index] = (route, wire_bytes, resources)
    result = {}
    for index, (route, wire_bytes, resources) in work.items():
        shared = [resource for resource, traversals in resources.items()
                  if users[resource] > 1 or traversals > 1]
        if shared:
            if any(supply[r] <= 0 for r in shared):
                raise NoRouteError('local shared capacity has no bandwidth in this wave')
            rate_limit = min(supply[r] * wire_bytes / loads[r] for r in shared)
            result[index] = (tuple(route['resources']), rate_limit)
    return result
