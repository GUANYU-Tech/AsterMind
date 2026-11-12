"""Whole-path fluid budget using Wang's exact frame accounting (no ARQ).

Native FlowSim accepts flow/resource coefficients, so last-frame padding is
represented exactly in work. This does not model packet pipeline transients.
"""
from math import ceil, isfinite
from ._vendor.inter_mean.mean import mean_frame_budget


def _capacity_contract(attrs, default_basis):
    basis = attrs.get('capacity_basis', default_basis)
    if basis not in ('raw_line', 'net_after_existing_fec', 'application_effective'):
        raise ValueError(f'unknown capacity basis: {basis}')
    applied = set(attrs.get('applied_costs', ()))
    if applied - {'framing', 'mapping', 'fec'}:
        raise ValueError(f'unknown applied costs: {applied}')
    if basis == 'net_after_existing_fec':
        applied.add('fec')
    elif basis == 'application_effective':
        applied.update(('framing', 'mapping', 'fec'))
    return basis, tuple(sorted(applied))


def resource_capacities(routes, policy):
    """Read physical supply once, in application-work accounting units B/ns."""
    capacities = {}
    pool_contracts = {}
    for edge in routes.edges.values():
        key = edge['resource_key']
        capacity = edge['bandwidth_gbps'] / 8
        if key in capacities and capacities[key] != capacity:
            raise ValueError('inconsistent directional capacities for shared duplex resource')
        capacities[key] = capacity
        attrs = {**edge['attributes'], **policy.resource_profiles.get(key[0], {})}
        contract = _capacity_contract(attrs, policy.runtime_capacity_basis)
        for pool in edge['shared_pools']:
            if pool in edge.get('shared_pool_coefficients',{}):
                continue  # Explicit payload-gate coefficients are independent of link encoding.
            if pool in pool_contracts and pool_contracts[pool] != contract:
                raise ValueError('shared pool requires uniform capacity basis and applied costs')
            pool_contracts[pool] = contract
    pools = [node.attributes.get('shared_capacity_pools_bps', {})
             for node in routes.frame.node_states.values()]
    pools.append(getattr(routes.frame, 'shared_capacity_pools_bps', {}))
    for pool_map in pools:
        for name, bps in pool_map.items():
            key, value = ('pool', name), bps / 8e9
            if not isfinite(value) or value < 0:
                raise ValueError('invalid shared pool capacity')
            if key in capacities and capacities[key] != value:
                raise ValueError('inconsistent pool capacity')
            capacities[key] = value
    return capacities


def isolated_path_ns(routes, route, size, policy):
    """The runtime whole-path model with exactly one flow and a frozen Frame."""
    if not size:
        return 0
    budget = path_budget(routes, route, size, policy)
    capacities = resource_capacities(routes, policy)
    limits = []
    for key, coefficient in budget['demand'].items():
        if key not in capacities:
            raise ValueError(f'missing shared capacity pool: {key}')
        limits.append(capacities[key] / coefficient)
    if budget['local_cap'] is not None:
        limits.append(budget['local_cap'])
    if not limits or min(limits) <= 0:
        raise ValueError('path has no positive effective capacity')
    return ceil(budget['startup'] + size / min(limits) + budget['tail'])


def path_budget(routes,route,size,policy):
    edges=routes.route_resources(route);settings=policy.inter_parameters
    frame=mean_frame_budget(size,settings)
    demand={};local_caps=[];tail=0.;has_isl=False;records=[]
    for edge in edges:
        same=edge['source_owner']==edge['target_owner']
        ground=not same and bool({edge['source_owner'],edge['target_owner']} & routes.ground_owners)
        attrs={**edge['attributes'],**policy.resource_profiles.get(edge['resource_key'][0],{})}
        if same:
            continue
        tail+=edge['latency_ns']
        cost=1.;processing=policy.ground_processing_ns if ground else 0.
        basis, applied = _capacity_contract(attrs, policy.runtime_capacity_basis)
        applied = set(applied)
        if not ground:
            has_isl=True
            if policy.runtime_protocol=='inter_reference':
                if 'framing' not in applied:cost*=frame.ethernet_time_bytes/size
                if 'mapping' not in applied:cost/=settings.mapping_efficiency
                if settings.fec.enabled and 'fec' not in applied:cost*=settings.fec.n/settings.fec.k
                processing=(settings.hop_processing_us+settings.routing_processing_us)*1000
                if settings.fec.enabled:processing+=settings.fec.encode_ns+settings.fec.decode_ns
        processing=attrs.get('processing_ns',processing)
        if not isfinite(processing) or processing<0:raise ValueError('invalid processing_ns')
        tail+=processing;key=edge['resource_key']
        if key in demand:raise ValueError('route repeats a capacity resource')
        demand[key]=cost
        for pool in edge['shared_pools']:
            # Adapter-owned native terminal gates forward payload exactly once.
            coefficient=edge.get('shared_pool_coefficients',{}).get(pool,cost)
            demand[('pool',pool)]=demand.get(('pool',pool),0)+coefficient
        records.append(dict(resource=str(key),capacity_basis=basis,applied_costs=sorted(applied),
            application_bytes=size,equivalent_resource_bytes=size*cost,coefficient=cost,
            kind='ground' if ground else 'isl',processing_ns=processing))
    return dict(demand=demand,local_cap=min(local_caps,default=None),tail=tail,
        startup=settings.startup_us*1000 if has_isl else 0.,path_keys=tuple(route['resources']),
        path=route['path'],budgets=records)
