"""Existing route search and replay policy over Engine-owned network state."""
import heapq
import math

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
            return arrival, tuple(replay)
        # A long transfer may no longer fit its contact window. Search for
        # a feasible alternative without caching failed timing predictions.
        route_cache.replay_failures += 1
    route_cache.searches += 1
    adjacency = self._routing_adjacency(kinds)
    queue = [(0.0, time_s, (source,), 0, ())]
    labels: dict[str, list[tuple[float, float]]] = {}
    serial = 0
    while queue:
        cost, arrival, nodes, _, path = heapq.heappop(queue)
        node = nodes[-1]
        if node == target:
            if not recompute:
                route_cache.remember(self, route_key, path, time_s)
            if commit:
                for leg in path:
                    self.reserve(leg)
            return arrival, path
        prior = labels.setdefault(node, [])
        if any(c <= cost and t <= arrival for c, t in prior):
            continue
        prior[:] = [(c, t) for c, t in prior if not (cost <= c and arrival <= t)]
        prior.append((cost, arrival))
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
                1.0
                if policy == "min_hops"
                else leg["propagation_delay_s"]
                if policy == "min_propagation"
                else leg["arrival_s"] - arrival
            )
            serial += 1
            heapq.heappush(
                queue,
                (
                    cost + extra,
                    leg["arrival_s"],
                    (*nodes, neighbor),
                    serial,
                    (*path, leg),
                ),
            )
    raise NetworkUnavailable("no_feasible_dynamic_route")

