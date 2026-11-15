"""Project cached topology paths into Comm's physical resource descriptions.

Path selection never forecasts traffic. Comm owns capacity competition,
protocol budgets, progress, retries and completion events.
"""
from types import SimpleNamespace

from astermind.engine.clock import Time
from astermind.engine.state.network import NetworkUnavailable
from .frame_surrogate import FrameRoutes, NoRouteError


class TargetNetworkFrames:
    def __init__(self, network):
        self.network = network
        self.host = network.host
        self.horizon = self.host.end

    def frame_at(self, time_s):
        host, topology = self.host, self.network.topology
        if time_s != host.now:
            raise ValueError("Comm may only read the current physical state")
        net = host.net
        nodes = {}
        pools = dict(net.pool_capacities)
        for identity, device in topology.devices.items():
            attrs = dict(device_kind=device.kind, capacity_gbps=device.capacity_gbps)
            hierarchy = getattr(topology, 'intra_hierarchies', {}).get(device.satellite)
            if hierarchy is not None:
                attrs['intra_hierarchy'] = hierarchy
            nodes[identity] = SimpleNamespace(satellite_id=device.satellite,
                role='ground' if device.satellite in net.ground_owners else 'compute',
                attributes=attrs)
            if device.kind == 'terminal' and device.capacity_gbps is not None:
                pools[f'device:{identity}'] = device.capacity_gbps * 1e9
        established = []
        for edge in net.edges:
            state = net.link_at(edge['link_id'], host.now)
            if state['state'] != 'UP':
                continue
            a, b = edge['endpoint_a'], edge['endpoint_b']
            for source, target in ((a, b), (b, a)):
                metadata = net.metadata(edge, source, target)
                if metadata is None or metadata.get('gate'):
                    continue
                attrs = net.directional_attributes(edge, source, target, host.now)
                if attrs['bandwidth_bps'] <= 0:
                    continue
                src, dst = topology.node_device[source], topology.node_device[target]
                shared = list(net.pools_for_hop(source, target))
                if topology.devices[src].satellite != topology.devices[dst].satellite:
                    shared += [f'device:{n}' for n in (src, dst) if f'device:{n}' in pools]
                established.append(SimpleNamespace(link_id=edge['link_id'],
                    source_node_id=src, target_node_id=dst,
                    bandwidth_bps=attrs['bandwidth_bps'],
                    propagation_delay_s=state['propagation_delay_s'],
                    shared_capacity_pools=tuple(dict.fromkeys(shared)),
                    attributes={**attrs, 'bidirectional': False,
                        'target_native_direction': (source, target)}))
        # All physical updates are pushed by the host. The expiry only names
        # actual known supply boundaries, never an artificial polling interval.
        boundaries = [Time(t) for t in net.environment_times if t > host.now]
        boundaries += [Time(t) for rows in net.link_times.values() for t in rows if t > host.now]
        return SimpleNamespace(start_time_s=host.now,
            valid_until_s=min(boundaries, default=None), revision=net.revision,
            node_states=nodes, established_topology_graph=tuple(established),
            shared_capacity_pools_bps=pools)

    def routes(self, frame):
        return TargetRoutes(self.network, frame)

    def candidate_path(self, source, target):
        return self.network.candidate_path(source, target)


class TargetRoutes(FrameRoutes):
    def __init__(self, network, frame):
        super().__init__(frame)
        self.network = network
        self.routing_policy = network.host.spec.routing_policy
        for edge in self.edges.values():
            edge['shared_pool_coefficients'] = {
                pool: 1.0 for pool in edge['shared_pools'] if pool.startswith('device:')
            }

    def route(self, source, target, size, *, first_hop=None):
        if type(size) is not int or size < 0:
            raise ValueError('payload must be nonnegative integer bytes')
        a, b = self.network.resolve_pair(source, target)
        try:
            selected = self.network.host.net.select_route(a, b, self.network.host.now, self.routing_policy)
        except NetworkUnavailable as error:
            raise NoRouteError(str(error)) from error
        topology = self.network.topology
        path = tuple(topology.physical_path(selected.nodes))
        if first_hop is not None and (len(path) < 2 or path[1] != first_hop):
            raise ValueError('first_hop would change the retained routing policy')
        keys = []
        for link_id, first, second in selected.hops:
            src = topology.node_device[first]
            dst = topology.node_device[second]
            if src == dst:
                continue
            key = (link_id, src, dst)
            if key not in self.edges:
                raise NoRouteError('selected path is not simultaneously available')
            keys.append(key)
        result = self._result(path, tuple(keys), size)
        result['target_route_legs'] = selected.hops
        result['routing_cost'] = selected.cost
        result['routing_distance_km'] = selected.distance_km
        return result

    def route_resources(self, route):
        # Native device routing crosses a terminal ingress/egress gate once.
        # The public projection puts that gate on both adjoining external
        # edges; charge it once, while retaining additive physical RF pools.
        seen = set()
        edges = super().route_resources(route)
        for edge in edges:
            pools = []
            for pool in edge['shared_pools']:
                if pool.startswith('device:'):
                    if pool in seen:
                        continue
                    seen.add(pool)
                pools.append(pool)
            edge['shared_pools'] = tuple(pools)
            edge['shared_pool_coefficients'] = {p: 1.0 for p in pools if p.startswith('device:')}
        return edges
