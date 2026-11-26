"""Persistent path identities, separate from per-transfer timing and traffic.

A route is selected once and reused until one of its directed links changes at
a physical boundary. Changes elsewhere do not evict it, even if a new, better
alternative appears. This is a routing policy, not an exact shortest-path memo.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass, replace
from types import MappingProxyType
from copy import copy
import math
from time import perf_counter


class PathRouteCache:
    """Path table refreshed at physical samples and observed topology events.

    No message sizes, reservations, shared-load scans or transfer forecasts.
    Snapshot construction is amortized across every query in that snapshot.
    """
    def __init__(self, max_entries=16384):
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.hits = self.searches = self.invalidations = self.snapshot_builds = 0
        self.negative_hits = 0
        self.marker = None
        self.start = self.deadline = None
        self.period = None
        self.adjacency = MappingProxyType({})
        self.edges = {}
        self.link_states = {}
        self.directional_attributes = {}
        self.boundaries = self.geometry_times = ()
        self.snapshot_wall_s = self.search_wall_s = 0.

    def fork(self):
        result = copy(self)
        result.entries = self.entries.copy()
        result.hits = result.searches = result.invalidations = result.snapshot_builds = 0
        result.negative_hits = 0
        result.snapshot_wall_s = result.search_wall_s = 0.
        return result

    def synchronize(self, net, time_s):
        from astermind.engine.policies.routing import RoutingEdge

        marker = (net.revision, id(net.edges), id(net.link_frames))
        if marker == self.marker and self.start <= time_s < self.deadline:
            return
        started = perf_counter()
        if marker != self.marker:
            # Construct the supply-boundary index once per graph version.
            times = set(net.environment_times)
            times.update(t for rows in net.link_times.values() for t in rows)
            times.update(t for rows in getattr(net, 'attribute_boundaries', {}).values() for t in rows)
            self.boundaries = tuple(sorted(times))
            physical = getattr(net, 'physical', None) or net
            self.geometry_times = tuple(sorted({t for rows in getattr(physical, 'sample_times', {}).values() for t in rows}))
        period = bisect.bisect_right(self.geometry_times, time_s) - 1
        index = bisect.bisect_right(self.boundaries, time_s)
        deadline = self.boundaries[index] if index < len(self.boundaries) else net.end_s
        provider = getattr(net, "prepared_route_frames", None)
        prepared = provider.at(net, time_s) if provider is not None else None
        if prepared is not None:
            offline_deadline, (adjacency, edges, states, directions) = prepared
            if offline_deadline != deadline:
                raise ValueError("Prepared route frame misses a live topology boundary")
            self._install(marker, time_s, deadline, period, adjacency, edges, states, directions, started)
            return
        adjacency, edges, states, directions = {}, {}, {}, {}
        allows = getattr(net, 'allows_hop', lambda a, b: True)
        for edge in net.edges:
            row = net.link_at(edge['link_id'], time_s)
            states[edge['link_id']] = row
            if row['state'] != 'UP':
                continue
            distance = net.distance(edge, time_s)
            if not math.isfinite(distance) or distance < 0:
                raise ValueError('routing distance must be finite and nonnegative')
            a, b = edge['endpoint_a'], edge['endpoint_b']
            for source, target in ((a, b), (b, a)):
                if not allows(source, target):
                    continue
                attrs = net.directional_attributes(edge, source, target, time_s)
                directions[edge['link_id'], source, target] = attrs
                if (row.get('ready_s', net.start_s) > time_s
                        or attrs['bandwidth_bps'] <= 0 or attrs.get('direction_blocked')):
                    continue
                item = RoutingEdge(edge['link_id'], source, target, distance)
                edges[item.key] = item
                adjacency.setdefault(source, []).append(item)
        adjacency = MappingProxyType({node: tuple(sorted(rows, key=lambda e: (e.target, e.link_id)))
                                           for node, rows in adjacency.items()})
        self._install(marker, time_s, deadline, period, adjacency, edges, states, directions, started)

    def _install(self, marker, time_s, deadline, period, adjacency, edges, states, directions, started):
        rewind = self.start is not None and time_s < self.start
        periodic = self.period is not None and period != self.period
        changed = edges != self.edges
        for key, route in tuple(self.entries.items()):
            # Periodic geometry refresh may choose a new optimum. Between
            # samples, only affected paths are removed; restored links wake
            # negative entries. Custom policies may depend on the whole graph.
            invalid = rewind or periodic or (route is None and changed)
            if route is not None:
                invalid |= any(hop not in edges for hop in route.hops)
                invalid |= key[2] not in {'min_hops', 'min_propagation'} and changed
            if invalid:
                del self.entries[key]
                self.invalidations += 1
        self.adjacency = adjacency
        self.edges = edges
        # Communication and routing read one physical snapshot. Records are
        # replaced at boundaries; readers never mutate a retained snapshot.
        self.link_states, self.directional_attributes = states, directions
        self.marker, self.start, self.deadline, self.period = marker, time_s, deadline, period
        self.snapshot_builds += 1
        self.snapshot_wall_s += perf_counter() - started

    def get(self, key):
        if key not in self.entries:
            return False, None
        result = self.entries[key]
        self.entries.move_to_end(key)
        self.hits += 1
        self.negative_hits += result is None
        return True, result

    def remember(self, key, route):
        self.entries[key] = route
        self.entries.move_to_end(key)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

    def info(self):
        return dict(entries=len(self.entries), hits=self.hits, searches=self.searches,
                    invalidations=self.invalidations, negative_hits=self.negative_hits,
                    snapshot_builds=self.snapshot_builds,
                    snapshot_wall_s=self.snapshot_wall_s, search_wall_s=self.search_wall_s)


@dataclass(frozen=True)
class CachedPath:
    hops: tuple
    signatures: tuple
    epoch: tuple


class PersistentRouteCache:
    """Bounded, fork-local cache; validate only paths actually used again."""

    def __init__(self, max_entries=16384):
        self.max_entries = max_entries
        self.entries = OrderedDict()
        self.hits = 0
        self.searches = 0
        self.invalidations = 0
        self.replay_failures = 0

    def fork(self):
        result = PersistentRouteCache(self.max_entries)
        result.entries = self.entries.copy()
        return result

    @staticmethod
    def key(source, target, policy, allow_wait, kinds, avoid_nodes=()):
        return (
            source, target, policy, allow_wait,
            None if kinds is None else frozenset(kinds),
            frozenset(avoid_nodes or ()),
        )

    @staticmethod
    def epoch(net, time_s):
        return (
            net.revision,
            id(net.edges),
            id(net.link_frames),
            bisect.bisect_right(net.environment_times, time_s) - 1,
            bisect.bisect_right(net.frame_times, time_s) - 1,
        )

    @staticmethod
    def signatures(net, hops, time_s):
        signatures = []
        for link_id, source, target in hops:
            edge = net.edge_by_id.get(link_id)
            if edge is None or {source, target} != {
                edge["endpoint_a"], edge["endpoint_b"]
            }:
                return None
            row = net.link_at(link_id, time_s)
            attrs = net.directional_attributes(row, source, target, time_s)
            signatures.append((
                row["state"], net.faulted(edge), row.get("setup_start_s"),
                row.get("ready_s"), row.get("available_until_s"),
                attrs["bandwidth_bps"], net.distance(edge, time_s),
            ))
        return tuple(signatures)

    def get(self, net, key, time_s):
        # A suffix installed by a full-path search has no avoid-list key. It
        # remains usable when the caller's visited prefix does not intersect it.
        candidates = (key, (*key[:-1], frozenset())) if key[-1] else (key,)
        epoch = self.epoch(net, time_s)
        for candidate in candidates:
            entry = self.entries.get(candidate)
            if entry is None:
                continue
            if any(
                target in key[-1]
                or not net.route_node_allowed(target, key[0], key[1])
                for _, _, target in entry.hops
            ):
                continue
            if entry.epoch != epoch:
                signatures = self.signatures(net, entry.hops, time_s)
                if signatures != entry.signatures:
                    del self.entries[candidate]
                    self.invalidations += 1
                    continue
                entry = replace(entry, epoch=epoch)
                self.entries[candidate] = entry
            self.entries.move_to_end(candidate)
            return candidate, entry.hops
        return None

    def remember(self, net, key, path, time_s):
        hops = tuple(
            (leg["link_id"], leg["source_node_id"], leg["target_node_id"])
            for leg in path
        )
        signatures = self.signatures(net, hops, time_s)
        epoch = self.epoch(net, time_s)
        self.entries[key] = CachedPath(hops, signatures, epoch)
        self.entries.move_to_end(key)
        # Seed forwarding suffixes so hop arrivals do not each run a new search.
        # Existing choices for other endpoint pairs remain persistent.
        for index in range(1, len(hops)):
            suffix_key = (hops[index][1], *key[1:-1], frozenset())
            if suffix_key not in self.entries:
                self.entries[suffix_key] = CachedPath(
                    hops[index:], signatures[index:], epoch
                )
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)

    def info(self):
        return {
            "entries": len(self.entries), "hits": self.hits,
            "searches": self.searches, "invalidations": self.invalidations,
            "replay_failures": self.replay_failures,
        }
