"""Expose the retained device network to Comm without selecting another route.

The original transfer(commit=False) query remains authoritative, including its
cache and future-contact lookup. Comm can only admit that path when every edge
is currently available. Gate nodes are collapsed to their public device IDs;
terminal capacity is a shared pool, not a second bandwidth supply.
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
        self.routing_policy = network.host.comm_policy.routing_policy
        for edge in self.edges.values():
            edge['shared_pool_coefficients'] = {
                pool: 1.0 for pool in edge['shared_pools'] if pool.startswith('device:')
            }

    def _publish_forecast(self):
        # Retain the route solver's observed-contender inputs. Whole-path Comm
        # replaces the active transport, but does not replace route selection.
        net = self.network.host.net
        net.shared_load, net.shared_pool_load = {}, {}
        active = self.network.communication_policy.active
        topology = self.network.topology
        for flow in active.values():
            for resource in flow['budget']['demand']:
                if resource[0] == 'pool':
                    pool = resource[1]
                    if pool.startswith('device:'):
                        identity = pool[len('device:'):]
                        a, b = topology.ingress[identity], topology.egress[identity]
                        edge = net._pairs[tuple(sorted((a, b)))]
                        key = (edge['link_id'], a, b)
                        net.shared_load[key] = net.shared_load.get(key, 0) + 1
                    continue
                edge = self.edges.get(resource)
                if edge is None:
                    continue
                a, b = edge['attributes']['target_native_direction']
                key = (resource[0], a, b)
                net.shared_load[key] = net.shared_load.get(key, 0) + 1
                for pool in net.pools_for_hop(a, b):
                    net.shared_pool_load[pool] = net.shared_pool_load.get(pool, 0) + 1

    def route(self, source, target, size, *, first_hop=None):
        if type(size) is not int or size < 0:
            raise ValueError('payload must be nonnegative integer bytes')
        self._publish_forecast()
        a, b = self.network.resolve_pair(source, target)
        try:
            _, legs = self.network.host.net.transfer(
                a, b, self.network.host.now, size, policy=self.routing_policy, commit=False)
        except NetworkUnavailable as error:
            raise NoRouteError(str(error)) from error
        topology = self.network.topology
        path = tuple(topology.physical_path((a, *(leg['target_node_id'] for leg in legs))))
        if first_hop is not None and (len(path) < 2 or path[1] != first_hop):
            raise ValueError('first_hop would change the retained routing policy')
        keys = []
        for leg in legs:
            src = topology.node_device[leg['source_node_id']]
            dst = topology.node_device[leg['target_node_id']]
            if src == dst:
                continue
            key = (leg['link_id'], src, dst)
            if key not in self.edges:
                raise NoRouteError('selected path is not simultaneously available')
            keys.append(key)
        result = self._result(path, tuple(keys), size)
        result['target_route_legs'] = tuple((leg['link_id'], leg['source_node_id'],
                                          leg['target_node_id']) for leg in legs)
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
