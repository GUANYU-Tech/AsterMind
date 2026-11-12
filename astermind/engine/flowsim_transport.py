"""FlowSim service for directed physical hops on the OrbitServe event clock."""

from __future__ import annotations

import importlib
import math

from astermind.engine.setup.radio_capacity import capacity_pools, hop_pools
from astermind.engine.clock import CLOCK, Time

# Preserve fractional logical bytes while the FlowSim API uses integer bytes.
BYTE_SCALE = 1_000_000


def require_flowsim():
    try:
        module = importlib.import_module("_flowsim")
    except ImportError as error:
        raise RuntimeError(
            "FlowSim mode requires the compatible AsterMind _flowsim extension"
        ) from error
    if not all(hasattr(module.EventSimulator, name) for name in (
        "advance_until_ticks", "start_hop", "set_bandwidth", "remaining", "cancel"
    )) or not hasattr(module.Topology, "attach_pool"):
        raise RuntimeError("FlowSim extension lacks the integer attosecond clock API; rebuild with astermind/engine/adapters/flowsim_clock.patch before running")
    return module


class FlowSimTransport:
    def __init__(self, nodes, edges, profiles, *, shared_pools=None, link_pools=None):
        del profiles
        module = require_flowsim()
        self.nodes = nodes
        self.edges = tuple(edges)
        self.active = {}
        self.now = Time()
        self.completed = []
        self._configured_capacities = {}
        pools = capacity_pools(nodes) if shared_pools is None else dict(shared_pools)
        link_pools = link_pools or {}
        topology = module.Topology()
        for node in sorted(nodes):
            topology.add_node(node)
        for pool, capacity in pools.items():
            topology.set_pool_capacity(pool, capacity * BYTE_SCALE / 1e9)
        for edge in self.edges:
            a, b = edge["endpoint_a"], edge["endpoint_b"]
            for source, target in ((a, b), (b, a)):
                topology.add_link(source, target, 1.0, 0, False)
                for pool in link_pools.get((edge['link_id'], source, target), hop_pools(nodes, source, target)):
                    topology.attach_pool(source, target, pool)
        self.simulator = module.EventSimulator(topology)
        self.simulator.advance_until_ticks('0')
        self.provenance = {
            "backend": "flowsim",
            "sharing_scope": "each_directed_physical_hop",
            "completion_source": "flowsim_event_simulator",
            "propagation_accounting": "orbitserve_after_serialization",
            "packetization": False,
            "aggregate_sharing": "max_min_directed_link_and_satellite_rf_pools",
            "payload_quantization_bytes": 1 / BYTE_SCALE,
            "clock": CLOCK,
            "bandwidth_evidence": "native_allocations_and_shared_resources",
        }

    def resource(self, leg):
        return leg["link_id"], leg["source_node_id"], leg["target_node_id"]

    def register_edge(self, edge):
        raise NotImplementedError(
            "FlowSim currently requires candidate links to be declared before the run"
        )

    def configure(self, capacities):
        for (_, source, target), value in capacities.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError("FlowSim capacities must be finite nonnegative bits/s")
            key = source, target
            if self._configured_capacities.get(key) != value:
                self.simulator.set_bandwidth(source, target, value * BYTE_SCALE / 1e9)
                self._configured_capacities[key] = value

    def start(self, identity, leg, payload_bytes):
        if identity in self.active or isinstance(payload_bytes, bool) or not math.isfinite(payload_bytes) or payload_bytes <= 0:
            raise ValueError("FlowSim requires a unique transfer and positive finite bytes")
        amount = max(1, round(payload_bytes * BYTE_SCALE))
        if amount >= 2**64:
            raise ValueError("FlowSim payload exceeds its scaled size limit")
        self.simulator.start_hop(
            identity, leg["source_node_id"], leg["target_node_id"], amount,
            lambda: self.completed.append(identity),
        )
        self.active[identity] = {"key": self.resource(leg), "leg": leg}

    def advance_until(self, time_s):
        if isinstance(time_s, bool) or not math.isfinite(time_s) or time_s < self.now:
            raise ValueError("FlowSim time must be finite and monotonic")
        start = self.now
        native = self.simulator.advance_until_ticks(str(Time(time_s).ticks))
        self.now = Time.from_ticks(int(native['end_ticks']))
        sent = {row['id']: row['bytes_sent'] / BYTE_SCALE for row in native['services']}
        rates = {row['id']: row['rate_bytes_per_ns'] * 8e9 / BYTE_SCALE for row in native['services']}
        resources = [dict(resource_id=row['id'], allocated_bandwidth_bps=row['allocated_bytes_per_ns'] * 8e9 / BYTE_SCALE,
                          capacity_bps=row['capacity_bytes_per_ns'] * 8e9 / BYTE_SCALE) for row in native['resources']]
        done, self.completed = sorted(self.completed), []
        if time_s > start and self.now <= start and not done:
            raise RuntimeError(
                f"FlowSim made no observable progress at {start}s toward {time_s}s"
            )
        for identity in done:
            self.active.pop(identity)
        return {"time_s": self.now, "interval_start_s": start,
                "bytes_sent": sent, "completed": done,
                'allocated_rates_bps': rates, 'resources': resources,
                'interval_start_ticks': native['start_ticks'], 'interval_end_ticks': native['end_ticks']}

    def cancel(self, identities):
        for identity in identities:
            if identity in self.active:
                self.simulator.cancel(identity)
                del self.active[identity]

    def close(self, force=False):
        del force
        self.cancel(tuple(self.active))
