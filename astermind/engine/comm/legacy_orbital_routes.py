"""Case Study III request routes on the original satellite-level graph."""

from bisect import bisect_right
import gzip
import json
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

from astermind.engine.clock import Time
from astermind.engine.policies.routing import RoutingEdge
from astermind.engine.state.network import NetworkUnavailable
from astermind.engine.state.route_cache import PathRouteCache

from .frame_surrogate import NoRouteError


class OrbitalRouteCache(PathRouteCache):
    """Read the legacy gzip frames and retain its directed-hop invalidation."""

    def __init__(self, net, folder):
        super().__init__()
        self.folder = Path(folder) if folder is not None else None
        if self.folder is not None:
            index = json.loads((self.folder / "index-24h.json").read_bytes())
            self.frames, self.end = index["times"], index["end"]
            if self.frames != sorted(set(self.frames)) or not self.frames or self.end <= self.frames[-1]:
                raise ValueError("Invalid legacy orbital frame index")
        else:
            times = set(net.environment_times)
            times.update(t for values in net.link_times.values() for t in values)
            times.update(t for values in net.attribute_boundaries.values() for t in values)
            self.frames = sorted(t for t in times if net.start_s <= t < net.end_s)
            self.end = net.end_s
        # The archived loader used sample_times, not the prepared snapshot's
        # ephemeris index. Preserve when that loader invalidated a route.
        self.geometry_times = tuple(sorted({t for values in getattr(net, "sample_times", {}).values()
                                            for t in values}))

    def records(self, net, start):
        if self.folder is not None:
            path = self.folder / f"h{int(start // 3600):02d}" / f"{start}.gz"
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                return json.load(stream).items()
        return physical_route_rows(net, start)

    def synchronize(self, net, time_s):
        marker = (net.revision, id(net.edges), id(net.link_frames))
        if marker == self.marker and self.start <= time_s < self.deadline:
            return
        i = bisect_right(self.frames, time_s) - 1
        if i < 0 or time_s >= self.end:
            raise NetworkUnavailable("outside legacy orbital frames")
        start = self.frames[i]
        deadline = self.frames[i + 1] if i + 1 < len(self.frames) else self.end
        period = bisect_right(self.geometry_times, time_s) - 1
        started = perf_counter()
        adjacency, edges = {}, {}
        for link, (a, b, distance, bits) in self.records(net, start):
            for direction, (source, target) in enumerate(((a, b), (b, a))):
                if bits & (1 << direction):
                    edge = RoutingEdge(link, source, target, distance)
                    edges[edge.key] = edge
                    adjacency.setdefault(source, []).append(edge)
        adjacency = MappingProxyType({source: tuple(sorted(rows, key=lambda e: (e.target, e.link_id)))
                                      for source, rows in adjacency.items()})
        self._install(marker, time_s, deadline, period, adjacency, edges, {}, {}, started)


def physical_route_rows(net, at):
    """The legacy offline export rule, evaluated on physical link records."""
    for edge in net.edges:
        link = edge["link_id"]
        state = net.link_at(link, at)
        if state["state"] != "UP" or state.get("ready_s", net.start_s) > at:
            continue
        a, b = edge["endpoint_a"], edge["endpoint_b"]
        bits = 0
        for direction, (source, target) in enumerate(((a, b), (b, a))):
            attrs = net.directional_attributes(edge, source, target, at)
            if attrs["bandwidth_bps"] > 0 and not attrs.get("direction_blocked"):
                bits |= 1 << direction
        if bits:
            yield link, (a, b, net.distance(edge, at), bits)


def export_orbital_frames(net, folder, horizon):
    cache = OrbitalRouteCache(net, None)
    times = [float(at) for at in cache.frames if at < horizon]
    for at in times:
        hour = folder / f"h{int(at // 3600):02d}"
        hour.mkdir(parents=True, exist_ok=True)
        with gzip.open(hour / f"{at}.gz", "wt", encoding="utf-8", compresslevel=1) as stream:
            json.dump(dict(physical_route_rows(net, at)), stream, separators=(",", ":"), allow_nan=False)
    (folder / "index-24h.json").write_text(json.dumps({"times": times, "end": float(horizon)}),
                                          encoding="utf-8")


class OrbitalRequestRouting:
    def __init__(self, host, topology):
        self.host, self.topology = host, topology
        self.net = host.net.physical
        self.net._path_route_cache = OrbitalRouteCache(self.net, host.orbital_frames)
        self.terminals = {device.satellite: identifier for identifier, device in topology.devices.items()
                          if device.kind == "terminal"}

    def quote(self, source, target, size, *, legs=None):
        devices = self.topology.devices
        a, b = devices[source].satellite, devices[target].satellite
        cursor = float(self.host.now)
        try:
            selected = self.net.select_route(a, b, cursor, self.host.spec.routing_policy)
            for link, first, second in selected.hops:
                kind = self.net.edge_by_id[link]['kind']
                scale = self.host.comm_policy.link_bandwidth_factors.get(kind, 1.0)
                leg = self.net._edge_transfer(link, first, second, cursor, size, False,
                                              bandwidth_fraction=scale)
                if leg is None:
                    raise NetworkUnavailable("selected path unavailable at hop arrival")
                if legs is not None:
                    legs.append(leg)
                cursor = leg["arrival_s"]
        except NetworkUnavailable as error:
            raise NoRouteError(str(error)) from error
        intermediate = tuple(self.terminals[owner] for owner in selected.nodes[1:-1])
        path = (source, *intermediate, target) if source != target else (source,)
        return Time(cursor), path, selected
