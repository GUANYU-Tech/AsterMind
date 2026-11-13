"""Existing route search and replay policy over Engine-owned network state."""
import heapq
import math

def _flow_leg_budget(net, leg, query_time):
    """Snapshot equal-share estimate; never admits or reserves a flow."""
    from astermind.engine.state.network import C_KM_PER_S
    source, target = leg['source_node_id'], leg['target_node_id']
    edge = net.edge_by_id[leg['link_id']]
    at = query_time
    rate = net.directional_attributes(edge, source, target, at)['bandwidth_bps']
    # Retain future-contact feasibility, using its first service capacity
    # when the current snapshot has no capacity.
    if rate <= 0:
        at = leg['segments'][0]['start_s']
        rate = net.directional_attributes(edge, source, target, at)['bandwidth_bps']
    rate /= 1 + net.shared_load.get((leg['link_id'], source, target), 0)
    for pool in net.pools_for_hop(source, target):
        rate = min(rate, net.pool_capacities[pool] /
                   (1 + net.shared_pool_load.get(pool, 0)))
    return net.distance(edge, at) / C_KM_PER_S, rate


def _flow_forecast(net, path, query_time, payload_bytes):
    delay, rate = 0.0, math.inf
    for leg in path:
        propagation, capacity = _flow_leg_budget(net, leg, query_time)
        delay += propagation + leg['queue_wait_s']
        rate = min(rate, capacity)
    return query_time + delay + (8 * payload_bytes / rate if rate > 0 else math.inf)


def route_transfer(
    self,
    source: str,
    target: str,
    time_s: float,
    payload_bytes: float,
    policy: str = "earliest_arrival",
    *,
    commit: bool = False,
    allow_wait: bool = True,
    kinds: set[str] | None = None,
    avoid_nodes: set[str] | None = None,
    recompute: bool = False,
) -> tuple[float, tuple[dict, ...]]:
    """Reuse a selected path; recompute current leg timing on every call.

    Policies choose the initial path. Traffic alone does not change it.
    Physical changes invalidate only cached paths using changed links.
    ``recompute=True`` bypasses persistence for an explicit fresh search.
    A path that cannot carry this payload also falls back to a fresh search.
    """
    from astermind.engine.state.network import NetworkUnavailable

    if policy not in {"min_hops", "min_propagation", "earliest_arrival"}:
        raise ValueError("unsupported routing policy")
    if not math.isfinite(payload_bytes) or payload_bytes < 0:
        raise ValueError("payload_bytes must be finite and nonnegative")
    if (
        source not in self.nodes
        or target not in self.nodes
        or not self.start_s <= time_s < self.end_s
    ):
        raise NetworkUnavailable("route_endpoint_or_time_unavailable")
    from astermind.engine.state.route_cache import PersistentRouteCache

    if not hasattr(self, '_route_cache'):
        self._route_cache = PersistentRouteCache()
    route_cache = self._route_cache
    route_key = route_cache.key(source, target, policy, allow_wait, kinds, avoid_nodes)
    selected = None if recompute else route_cache.get(self, route_key, time_s)
    if selected is not None:
        _, hops = selected
        arrival, replay = time_s, []
        for link_id, first, second in hops:
            leg = self._edge_transfer(
                link_id, first, second, arrival, payload_bytes, allow_wait
            )
            if leg is None:
                break
            replay.append(leg)
            arrival = leg['arrival_s']
        else:
            route_cache.hits += 1
            if commit:
                for leg in replay:
                    self.reserve(leg)
            estimate = (_flow_forecast(self, replay, time_s, payload_bytes)
                        if policy == 'earliest_arrival' and not commit else arrival)
            return estimate, tuple(replay)
        # A long transfer may no longer fit its contact window. Search for
        # a feasible alternative without caching failed timing predictions.
        route_cache.replay_failures += 1
    route_cache.searches += 1
    adjacency = self._routing_adjacency(kinds)
    # Device ports share their satellite/ground owner. Orbital graph nodes
    # already represent those owners. Only crossing owners counts as a hop.
    owners = getattr(self, 'owner', {})
    queue = [(0.0, time_s, (source,), 0, (), 0.0, math.inf)]
    labels = {}
    serial = 0
    while queue:
        cost, arrival, nodes, _, path, flow_delay, bottleneck = heapq.heappop(queue)
        node = nodes[-1]
        if node == target:
            if not recompute:
                route_cache.remember(self, route_key, path, time_s)
            if commit:
                for leg in path:
                    self.reserve(leg)
            estimate = time_s + cost if policy == 'earliest_arrival' and not commit else arrival
            return estimate, path
        prior = labels.setdefault(node, [])
        # Prefix scores alone cannot dominate a wider bottleneck: a later
        # narrow link can reverse the ranking of two prefixes.
        label = ((flow_delay, -bottleneck, arrival) if policy == 'earliest_arrival'
                 else (cost, arrival))
        if any(all(a <= b for a, b in zip(old, label)) for old in prior):
            continue
        prior[:] = [old for old in prior if not all(a <= b for a, b in zip(label, old))]
        prior.append(label)
        for neighbor, link_id in adjacency.get(node, ()):
            if (
                neighbor in nodes
                or neighbor in (avoid_nodes or ())
                or not self.route_node_allowed(neighbor, source, target)
            ):
                continue
            leg = self._edge_transfer(
                link_id, node, neighbor, arrival, payload_bytes, allow_wait
            )
            if leg is None:
                continue
            extra = (
                float(owners.get(node, node) != owners.get(neighbor, neighbor))
                if policy == "min_hops"
                else leg["propagation_delay_s"]
                if policy == "min_propagation"
                else leg["arrival_s"] - arrival
            )
            next_delay, next_bottleneck = flow_delay, bottleneck
            next_cost = cost + extra
            if policy == 'earliest_arrival':
                propagation, rate = _flow_leg_budget(self, leg, time_s)
                next_delay += propagation + leg['queue_wait_s']
                next_bottleneck = min(bottleneck, rate)
                if next_bottleneck <= 0:
                    continue
                next_cost = next_delay + 8 * payload_bytes / next_bottleneck
            serial += 1
            heapq.heappush(
                queue,
                (
                    next_cost,
                    leg["arrival_s"],
                    (*nodes, neighbor),
                    serial,
                    (*path, leg),
                    next_delay,
                    next_bottleneck,
                ),
            )
    raise NetworkUnavailable("no_feasible_dynamic_route")

