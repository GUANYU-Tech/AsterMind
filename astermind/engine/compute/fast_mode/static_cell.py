"""Deployment checks for rank paths that outlive the global Network frame."""

from itertools import chain
from math import isclose

from astermind.engine.clock import Time
from astermind.engine.comm.target_network import TargetNetworkFrames


def static_rank_path_window(frames, devices, program):
    """Return when all internal paths are invariant until the replay horizon."""
    if type(frames) is not TargetNetworkFrames:
        return None
    host = frames.host
    net = host.net
    physical = net.physical
    if (physical is None or physical.physical_attributes is None
            or physical.spec.faults or net.active_faults
            or host.spec.routing_policy not in ('min_hops', 'min_propagation')
            or (host.controller is not None
                and not getattr(host.controller, 'automatic', False))):
        return None

    if any(program.experts):
        return None
    rank_owners = tuple(net.topology.devices[device].satellite for device in devices)
    transfers = chain(program.input_transfers, chain.from_iterable(program.stages),
                      chain.from_iterable(chain.from_iterable(program.boundaries)))
    pp_links = set()
    for transfer in transfers:
        participants = frozenset(rank_owners[rank] for rank in transfer.ranks)
        if len(participants) > 1:
            if transfer.operation != 'send_recv' or len(participants) != 2:
                return None
            pp_links.add(participants)

    owners = set(rank_owners)
    boundary_nodes = set()
    attachments = {}
    ring_edges = {}
    ready = Time()
    for edge in net.edges:
        a, b = edge['endpoint_a'], edge['endpoint_b']
        inside_a, inside_b = net.owner[a] in owners, net.owner[b] in owners
        if inside_a != inside_b:
            # A simple route between two local ranks cannot leave and return
            # through the same vertex. Two distinct exits need the normal check.
            boundary_nodes.add(a if inside_a else b)
            if len(boundary_nodes) > 1:
                return None
        if not (inside_a and inside_b):
            continue
        if any((net.metadata(edge, src, dst) or {}).get('schedule')
               for src, dst in ((a, b), (b, a))):
            return None

        link = edge['link_id']
        physical_id = net.metadata(edge).get('physical_link_id')
        owner_a, owner_b = net.owner[a], net.owner[b]
        if owner_a != owner_b:
            if physical_id is None:
                return None
            if attachments.setdefault(owner_a, a) != a or attachments.setdefault(owner_b, b) != b:
                return None
        history = net.link_frames[link]
        if not history:
            return None
        last = history[-1]
        if last['state'] != 'UP' or last['end_s'] < host.end:
            return None
        edge_ready = max(last['time_s'], last['ready_s'])
        if physical_id is not None:
            boundaries = physical.attribute_boundaries.get(physical_id, ())
            if (physical_id in physical.added_edge_ids or not boundaries
                    or boundaries[0] > edge_ready or boundaries[-1] < host.end
                    or any(edge_ready < at < host.end for at in boundaries)):
                return None
            physical_edge = physical.edge_by_id[physical_id]
            samples = (edge_ready, *(at for at in physical.environment_times if edge_ready < at < host.end))
            distances = [physical.distance(physical_edge, at) for at in samples]
            if not isclose(min(distances), max(distances), rel_tol=0., abs_tol=1e-9):
                return None
            for src, dst in ((a, b), (b, a)):
                reference = net.directional_attributes(edge, src, dst, edge_ready)
                if reference['bandwidth_bps'] <= 0 or reference.get('direction_blocked') or any(
                        net.directional_attributes(edge, src, dst, at) != reference for at in samples[1:]):
                    return None
            owners_pair = frozenset((owner_a, owner_b))
            if owners_pair in ring_edges:
                return None
            ring_edges[owners_pair] = min(distances), max(distances)
        ready = max(ready, Time(edge_ready))

    if pp_links:
        if not pp_links <= ring_edges.keys():
            return None
        # Adjacent stages use the direct link even under rounding of sampled
        # distances: any alternative has at least two physical hops. The
        # endpoint attachment within each satellite is the same on both paths.
        min_hop = min(low for low, _ in ring_edges.values())
        if any(ring_edges[pair][1] >= 2 * min_hop for pair in pp_links):
            return None
    return ready, host.end, net.revision
