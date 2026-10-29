"""Deterministic ground/satellite admission with shared and per-service quotas.

Deferred acceptance uses ground proposals and responsive satellite preferences
under a total quota and nested user/gateway quotas. It is a per-boundary model;
it does not claim optimal throughput or load-aware scheduling.
"""

from collections import deque
import math


def ground_endpoints(nodes, edge):
    a, b = edge['endpoint_a'], edge['endpoint_b']
    return (a, b) if nodes[a].role == 'ground' else (b, a)


def association_key(state, edge, policy, time_s):
    ground, satellite = ground_endpoints(state.nodes, edge)
    score = edge['distance_km']
    if policy == 'longest_available':
        score = -(edge['available_until_s'] - time_s)
    elif policy == 'highest_capacity':
        score = -min(state.directional_attributes(edge, ground, satellite, time_s)['bandwidth_bps'],
                     state.directional_attributes(edge, satellite, ground, time_s)['bandwidth_bps'])
    elif policy == 'highest_elevation':
        origin = state.position(ground, time_s)
        target = state.position(satellite, time_s)
        sight = tuple(b - a for a, b in zip(origin, target))
        # Ellipsoidal surface normal (WGS84); enough for a ranking, not visibility.
        normal = (origin[0], origin[1], origin[2] / (1 - 1 / 298.257223563)**2)
        norm = math.sqrt(sum(x*x for x in normal) * sum(x*x for x in sight))
        score = -sum(a*b for a, b in zip(normal, sight)) / norm if norm else 0.
    return score, edge['distance_km'], edge['link_id']


def select_ground_links(state, edges, time_s, retained=()):
    nodes, spec = state.nodes, state.spec
    reserved_ground, reserved_satellite, reserved_service = {}, {}, {}
    for edge in retained:
        ground, satellite = ground_endpoints(nodes, edge)
        service = nodes[ground].ground_kind
        reserved_ground[ground] = reserved_ground.get(ground, 0) + 1
        reserved_satellite[satellite] = reserved_satellite.get(satellite, 0) + 1
        key = satellite, service
        reserved_service[key] = reserved_service.get(key, 0) + 1

    def ground_capacity(ground):
        return nodes[ground].sgl_terminals - reserved_ground.get(ground, 0)

    def choose(satellite, proposals, policy):
        node = nodes[satellite]
        accepted, counts = [], {'user': 0, 'gateway': 0}
        for edge in sorted(proposals, key=lambda e: association_key(state, e, policy, time_s)):
            ground, _ = ground_endpoints(nodes, edge)
            service = nodes[ground].ground_kind
            quota = getattr(node, 'max_' + service + '_links')
            if (len(accepted) < node.sgl_terminals - reserved_satellite.get(satellite, 0)
                and (quota is None or counts[service] + reserved_service.get((satellite,service), 0) < quota)):
                accepted.append(edge)
                counts[service] += 1
        return accepted

    if spec.ground_matching == 'greedy':
        accepted, ground_counts, satellite_edges = [], {}, {}
        for edge in sorted(edges, key=lambda e: association_key(state, e, spec.ground_access_policy, time_s)):
            ground, satellite = ground_endpoints(nodes, edge)
            if ground_counts.get(ground, 0) >= ground_capacity(ground):
                continue
            previous = satellite_edges.get(satellite, [])
            chosen = choose(satellite, [*previous, edge], spec.ground_access_policy)
            if len(chosen) > len(previous):
                satellite_edges[satellite] = chosen
                ground_counts[ground] = ground_counts.get(ground, 0) + 1
                accepted.append(edge)
        return {e['link_id'] for e in accepted}

    preferences = {}
    for edge in edges:
        ground, _ = ground_endpoints(nodes, edge)
        preferences.setdefault(ground, []).append(edge)
    for rows in preferences.values():
        rows.sort(key=lambda e: association_key(state, e, spec.ground_access_policy, time_s))
    pending = deque(sorted(preferences))
    next_choice = dict.fromkeys(preferences, 0)
    counts = dict.fromkeys(preferences, 0)
    held = {}
    while pending:
        ground = pending.popleft()
        while counts[ground] < ground_capacity(ground) and next_choice[ground] < len(preferences[ground]):
            edge = preferences[ground][next_choice[ground]]
            next_choice[ground] += 1
            _, satellite = ground_endpoints(nodes, edge)
            proposals = [*held.get(satellite, ()), edge]
            chosen = choose(satellite, proposals, spec.satellite_access_policy)
            chosen_ids = {e['link_id'] for e in chosen}
            counts[ground] += 1
            for rejected in proposals:
                if rejected['link_id'] not in chosen_ids:
                    peer, _ = ground_endpoints(nodes, rejected)
                    counts[peer] -= 1
                    if peer != ground:
                        pending.append(peer)
            held[satellite] = chosen
    return {e['link_id'] for rows in held.values() for e in rows}
