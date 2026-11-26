"""Load prepared directed routing snapshots for one fixed Fast L3 deployment."""

from bisect import bisect_right
from collections import OrderedDict
import gzip
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from astermind.engine.clock import Time
from astermind.engine.policies.routing import RoutingEdge


def deployment_key(net):
    physical = net.physical
    payload = (physical.dataset_identity, net.edges, sorted(net._directions.items()))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class PreparedRouteFrames:
    def __init__(self, path, net):
        self.path = Path(path)
        document = json.loads((self.path/"index.json").read_bytes())
        if document["format"] != 1 or document["deployment"] != deployment_key(net):
            raise ValueError("Offline route frames do not match the deployed physical graph")
        self.frames = document["frames"]
        self.starts = tuple(start for start, _ in self.frames)
        self.revision = net.revision
        self.loaded = OrderedDict()

    def at(self, net, at):
        if net.revision != self.revision or net.active_faults or net.disabled_device_links or net.failed_device_links:
            return None
        ticks = Time(at).ticks
        index = bisect_right(self.starts, ticks)-1
        if index < 0 or ticks >= self.frames[index][1]:
            return None
        start, end = self.frames[index]
        if start not in self.loaded:
            with gzip.open(self.path/f"{start}.json.gz", "rt", encoding="utf-8") as stream:
                frame = json.load(stream)
            edges, adjacency = {}, {}
            for identity, source, target, distance in frame["edges"]:
                edge = RoutingEdge(identity, source, target, distance)
                edges[edge.key] = edge
                adjacency.setdefault(source, []).append(edge)
            adjacency = MappingProxyType({source: tuple(rows) for source, rows in adjacency.items()})
            states = {identifier: {"state": state, "propagation_delay_s": delay}
                      for identifier, state, delay in frame["states"]}
            directions = {(identity, source, target): row
                          for identity, source, target, row in frame["directions"]}
            self.loaded[start] = adjacency, edges, states, directions
            if len(self.loaded) > 2:
                self.loaded.popitem(last=False)
        self.loaded.move_to_end(start)
        return min(Time.from_ticks(end), net.end_s), self.loaded[start]
