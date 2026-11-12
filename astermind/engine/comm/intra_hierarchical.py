"""Single-satellite hierarchical adapter for equal complete su groups.

Supports two or more su stars with uniform access links and a symmetric,
uniform direct so switch fullmesh. This physical contract is narrower than
upstream's abstract topology catalogue.
"""
from itertools import permutations
from math import ceil, isfinite
from .estimator import Estimate
from ._vendor.intra_theory.hierarchical_upstream import make_h200_hierarchical_theory, make_h200_ring_theory

class HierarchyNotApplicable(ValueError):
    """Valid task, but physical fabric does not fit the whole H-S model."""


from astermind.engine.comm.intra_hierarchy import IntraHierarchy


def estimate_hierarchical(task, routes, config):
    """Return None for ordinary algorithms; never silently replace their plan."""
    if task.algorithm != 'hierarchical':
        return None
    if config is None:
        raise ValueError('algorithm=hierarchical requires intra_hierarchy configuration')
    if task.operation not in ('all_reduce', 'all_gather', 'all_to_all'):
        raise NotImplementedError('intra hierarchical supports uniform all_reduce/all_gather/all_to_all')
    if task.bytes_semantics != 'per_rank_input' or task.partition_bytes or task.traffic_matrix_bytes or task.ring_order:
        raise ValueError('hierarchical requires per_rank_input, no variable partitions/matrix/global ring_order')
    if type(task.bytes) is not int or task.bytes < 0:
        raise ValueError('bytes must be a nonnegative integer')
    if task.operation == 'all_reduce' and task.reduction != 'sum':
        raise ValueError('hierarchical all_reduce requires reduction=sum')
    if task.operation != 'all_reduce' and task.reduction is not None:
        raise ValueError('non-reducing operation does not accept reduction')
    selected=[(g,s) for g,s in zip(config.su_groups,config.su_switches) if set(g)&set(task.participants)]
    groups=tuple(g for g,s in selected);switches=tuple(s for g,s in selected)
    if len(groups)<2:raise ValueError('hierarchical needs at least two complete su groups')
    gpus = sum(groups,())
    if len(task.participants) != len(gpus) or set(task.participants) != set(gpus):
        raise ValueError('hierarchical task must cover complete configured su groups')
    devices = routes.topology.devices
    ids = gpus + switches
    if any(x not in devices for x in ids):
        raise ValueError('configured su device missing from Network Frame')
    if len({devices[x].satellite for x in ids}) != 1:
        raise ValueError('intra hierarchical cannot span satellites')
    if any(devices[x].kind != 'gpu' for x in gpus) or any(devices[x].kind != 'switch' for x in switches):
        raise ValueError('su groups must contain GPUs and su_switches must be switches')
    from .intra_topology import nonblocking_switches
    if not nonblocking_switches(routes,switches):
        raise HierarchyNotApplicable('declared switch aggregate capacity cannot sustain all fabric ports; whole H-S does not model this bottleneck')
    # Check the actual route, not just the supplied grouping labels.
    paths = []
    group_of = {g: i for i, group in enumerate(groups) for g in group}
    for a, b in permutations(gpus, 2):
        i, j = group_of[a], group_of[b]
        expected = [a, switches[i], b] if i == j else [a, switches[i], switches[j], b]
        from .frame_surrogate import NoRouteError
        try:route = routes.route(a, b, task.bytes)
        except NoRouteError as error:raise HierarchyNotApplicable(str(error)) from error
        if route['path'] != expected:
            raise HierarchyNotApplicable('hierarchical requires su stars and a direct so switch fullmesh')
        paths.append(route)
    trunk = [e for e in routes.edges.values() if e['source'] in switches and e['target'] in switches]
    if len(trunk) != len(groups)*(len(groups)-1) or len({(e['bandwidth_gbps'], e['latency_ns']) for e in trunk}) != 1:
        raise HierarchyNotApplicable('so switch fullmesh must be symmetric and uniform')
    access = [e for e in routes.edges.values() if e['source'] in gpus or e['target'] in gpus]
    if len(access)!=2*len(gpus) or len({(e['bandwidth_gbps'], e['latency_ns']) for e in access}) != 1:
        raise HierarchyNotApplicable('hierarchical requires uniform su access links')
    # Upstream factory argument named gbps is actually GB/s (50 == 400GbE).
    # One shared trunk per su domain, not one independently duplicated per GPU.
    rate, latency = trunk[0]['bandwidth_gbps'], trunk[0]['latency_ns']
    model = make_h200_hierarchical_theory(len(groups[0]), len(groups),
        so_per_link_gbps=rate / 8, so_links_per_gpu=len(groups)-1,
        link_latency_us=latency / 1000)
    # Bound the calibrated su schedule by the configured GPU access rate.
    # This remains a uniform analytical fabric, not switch queue simulation.
    su_rate = access[0]['bandwidth_gbps'] * 125  # B/us
    model.su = make_h200_ring_theory(ring_bw_bytes_per_us=min(177000, su_rate),
        packet_bw_bytes_per_us=min(177000, su_rate),
        link_latency_us=2 * access[0]['latency_ns'] / 1000)
    data = task.bytes * len(gpus) if task.operation == 'all_gather' else task.bytes
    if task.operation in ('all_reduce','all_to_all') and task.bytes % len(gpus):
        raise ValueError('equal hierarchical partitions require bytes divisible by participant count')
    us = model.predict(data, 'ring', 128, collective=task.operation.replace('_', '')) if data else 0
    if not isfinite(us) or us < 0:
        raise ValueError('invalid hierarchical duration')
    inputs=dict(su_groups=groups,su_switches=switches,gpus_per_domain=len(groups[0]),domains=len(groups),
        operation=task.operation,application_bytes=task.bytes,upstream_data_bytes=data,
        su_access_gbps=access[0]['bandwidth_gbps'],su_path_latency_ns=2*access[0]['latency_ns'],
        so_link_gbps=rate,so_links_per_domain=len(groups)-1,so_link_latency_ns=latency,
        credits=128,su_profile='h200_ring',so_profile='h200_roce',upstream_commit='c0e4bdc434bd1f18bd3f8e80d80ca2263369ff1f')
    return Estimate(ceil(us * 1000), 'intra_hierarchical', 0, 0, 0, 0, tuple(paths),inputs)
