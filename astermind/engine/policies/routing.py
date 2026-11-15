"""Snapshot path selection, plus the separate legacy offline transfer forecast."""
import heapq
import math
import re
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class RoutingEdge:
    link_id: str
    source: str
    target: str
    distance_km: float

    @property
    def key(self):
        return self.link_id, self.source, self.target


@dataclass(frozen=True)
class RoutingGraph:
    """Read-only, currently usable directed links; no traffic or payload inputs."""
    adjacency: object
    time_s: object


@dataclass(frozen=True)
class SelectedRoute:
    nodes: tuple
    hops: tuple
    distance_km: float
    policy: str

    @property
    def cost(self):
        return self.distance_km if self.policy == "min_propagation" else len(self.hops)


_ROUTING_POLICIES = {}


def register_routing_policy(name, selector):
    """Register trusted application code before starting a Simulation.

    selector(graph, source, target) returns node IDs, or None if unreachable.
    Paths are validated before use. JSON never imports or executes plugin code.
    """
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,127}", name):
        raise ValueError("invalid routing policy name")
    if not callable(selector) or name in _ROUTING_POLICIES:
        raise ValueError(f"routing policy must be callable and registered once: {name}")
    _ROUTING_POLICIES[name] = selector


def get_routing_policy(name):
    try:
        return _ROUTING_POLICIES[name]
    except KeyError:
        raise ValueError(f"Unregistered routing policy {name!r}; register_routing_policy() before starting the runtime") from None


def _unwind(parent, source, target):
    if target not in parent:
        return None
    path = [target]
    while path[-1] != source:
        path.append(parent[path[-1]])
    return tuple(reversed(path))


def min_hops(graph, source, target):
    parent, queue = {source: None}, deque([source])
    while queue:
        node = queue.popleft()
        if node == target:
            return _unwind(parent, source, target)
        for edge in graph.adjacency.get(node, ()):
            if edge.target not in parent:
                parent[edge.target] = node
                queue.append(edge.target)
    return None


def min_propagation(graph, source, target):
    parent, costs, queue = {source: None}, {source: 0.0}, [(0.0, source)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != costs[node]:
            continue
        if node == target:
            return _unwind(parent, source, target)
        for edge in graph.adjacency.get(node, ()):
            candidate = cost + edge.distance_km
            if candidate < costs.get(edge.target, math.inf):
                parent[edge.target], costs[edge.target] = node, candidate
                heapq.heappush(queue, (candidate, edge.target))
    return None


register_routing_policy("min_hops", min_hops)
register_routing_policy("min_propagation", min_propagation)


def select_route(net, source, target, time_s, policy="min_hops", *, kinds=None, avoid_nodes=()):
    """Look up a path; search only on a miss or engine-controlled invalidation."""
    from astermind.engine.state.network import NetworkUnavailable
    from astermind.engine.state.route_cache import PathRouteCache

    selector = get_routing_policy(policy)
    if source not in net.nodes or target not in net.nodes or not net.start_s <= time_s < net.end_s:
        raise NetworkUnavailable("route_endpoint_or_time_unavailable")
    cache = getattr(net, "_path_route_cache", None)
    if cache is None:
        cache = net._path_route_cache = PathRouteCache()
    cache.synchronize(net, time_s)
    kinds = None if kinds is None else frozenset(kinds)
    avoid_nodes = frozenset(avoid_nodes or ())
    key = (source, target, policy, kinds, avoid_nodes)
    found, result = cache.get(key)
    if not found:
        # Filtering is done only for a new search, never for a cache hit.
        allowed = {n for n in net.nodes if n not in avoid_nodes and
                   (n == source or net.route_node_allowed(n, source, target))}
        if len(allowed) == len(net.nodes) and kinds is None:
            adjacency = cache.adjacency
        else:
            adjacency = MappingProxyType({n: tuple(e for e in edges if e.target in allowed and
                (kinds is None or net.edge_by_id[e.link_id]['kind'] in kinds))
                for n, edges in cache.adjacency.items() if n in allowed})
        graph = RoutingGraph(adjacency, time_s)
        cache.searches += 1
        nodes = selector(graph, source, target)
        result = None
        if nodes is not None:
            nodes = tuple(nodes)
            if not nodes or nodes[0] != source or nodes[-1] != target or len(set(nodes)) != len(nodes):
                raise ValueError(f"routing policy {policy!r} returned invalid endpoints or a cycle")
            hops, distance = [], 0.0
            for a, b in zip(nodes, nodes[1:]):
                options = [e for e in adjacency.get(a, ()) if e.target == b]
                if not options:
                    raise ValueError(f"routing policy {policy!r} returned an unavailable/forbidden hop: {a}->{b}")
                edge = min(options, key=lambda e: (e.distance_km, e.link_id)) if policy == 'min_propagation' else options[0]
                hops.append(edge.key)
                distance += edge.distance_km
            result = SelectedRoute(nodes, tuple(hops), distance, policy)
        cache.remember(key, result)
    if result is None:
        raise NetworkUnavailable("no_current_route")
    return result

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

