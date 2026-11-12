"""Read physical paths before admitting whole prediction; adapt intra P2P.

Without explicit su groups the legacy H200 profile declares one NVLink fabric.
Only uniform direct-fullmesh or single-switch fabrics get the whole shortcut.
"""
from itertools import permutations
from math import ceil,isfinite
from .estimator import Estimate
from ._vendor.intra_theory import hierarchical_upstream as theory

def _ns(us):
    if not isfinite(us) or us<0:raise ValueError('invalid intra surrogate duration')
    return ceil(us*1000)


def nonblocking_switches(routes, switches):
    """Whole models require enough aggregate fabric capacity for all ports.

    This admission check does not implement time-varying intra contention.
    """
    for switch in switches:
        capacity=routes.frame.node_states[switch].attributes.get('capacity_gbps')
        if capacity is None:continue
        outgoing=sum(e['bandwidth_gbps'] for e in routes.edges.values() if e['source']==switch)
        if capacity < outgoing:
            return False
    return True

def p2p_ns(routes,route,size,config=None):
    if not size:return 0
    if len({routes.topology.devices[n].satellite for n in route['path']})!=1:
        raise ValueError('intra path leaves satellite')
    cross=False
    if config:
        membership={g:i for i,group in enumerate(config.su_groups) for g in group}
        a,b=route['path'][0],route['path'][-1]
        if a not in membership or b not in membership:raise ValueError('GPU absent from configured su domains')
        cross=membership[a]!=membership[b]
    cap=route['bandwidth_gbps']*125
    if cross:
        model=theory.make_h200_roce_theory(num_lanes=1,ring_bw_bytes_per_us=cap*.4,
            packet_bw_bytes_per_us=cap*.4,link_latency_us=route['latency_ns']/1000)
    else:
        model=theory.make_h200_sendrecv_theory(ring_bw_bytes_per_us=min(177000,cap),
            packet_bw_bytes_per_us=min(177000,cap),link_latency_us=route['latency_ns']/1000)
    return _ns(model.predict(size,2,'ring',128,collective='sendrecv'))

def whole_estimate(task,routes,config=None):
    p=task.participants;op=task.operation;algo=task.algorithm
    if op not in ('all_reduce','all_gather','all_to_all') or task.partition_bytes or task.traffic_matrix_bytes:return None
    if len({routes.topology.devices[g].satellite for g in p})!=1:return None
    if config and not any(set(p)<=set(g) for g in config.su_groups):return None
    from .frame_surrogate import NoRouteError
    try:paths=tuple(routes.route(a,b,task.bytes) for a,b in permutations(p,2))
    except NoRouteError:return None
    mids={tuple(r['path'][1:-1]) for r in paths}
    switched=len(mids)==1 and len(next(iter(mids)))==1
    direct=all(len(r['path'])==2 for r in paths)
    if not(switched or direct):return None
    if switched and routes.topology.devices[next(iter(mids))[0]].kind!='switch':return None
    if switched and not nonblocking_switches(routes,(next(iter(mids))[0],)):return None
    values={(r['bandwidth_gbps'],r['latency_ns']) for r in paths}
    if len(values)!=1:return None
    cap,lat=next(iter(values));cap*=125
    factories={'ring':theory.make_h200_ring_theory,'tree':theory.make_h200_tree_theory,
        'fullmesh':theory.make_h200_fullmesh_theory,'pairwise':theory.make_h200_alltoall_theory}
    if algo not in factories:return None
    model=factories[algo]()
    for attr in ('ring_bw','BW_link','tree_bw_per_level','fullmesh_bw'):
        if hasattr(model,attr):setattr(model,attr,min(getattr(model,attr),cap))
    model.lat=lat/1000
    data=task.bytes*len(p) if op=='all_gather' else task.bytes
    us=model.predict(data,len(p),'ring' if algo=='pairwise' else algo,128,collective=op.replace('_','')) if data else 0
    return Estimate(_ns(us),'intra_'+algo,0,0,0,0,paths,dict(profile='h200_'+algo,
        physical_path_capacity_gbps=cap/125,path_latency_ns=lat,upstream_data_bytes=data,credits=128))
