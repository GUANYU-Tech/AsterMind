"""Persistent path identities, separate from per-transfer timing and traffic.

A route is selected once and reused until one of its directed links changes at
a physical boundary. Changes elsewhere do not evict it, even if a new, better
alternative appears. This is a routing policy, not an exact shortest-path memo.
"""

from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass, replace


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
