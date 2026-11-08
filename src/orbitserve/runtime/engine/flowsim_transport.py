"""FlowSim service for directed physical hops on the OrbitServe event clock."""

from __future__ import annotations

import importlib
import math

from ..setup.radio_capacity import capacity_pools, hop_pools

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
        "advance_until", "start_hop", "set_bandwidth", "remaining", "cancel"
    )) or not hasattr(module.Topology, "attach_pool"):
        raise RuntimeError("FlowSim extension lacks the shared-runtime API")
    return module


class FlowSimTransport:
    def __init__(self, nodes, edges, profiles, *, shared_pools=None, link_pools=None):
        del profiles
        module = require_flowsim()
        self.nodes = nodes
        self.edges = tuple(edges)
        self.active = {}
        self.now = 0.0
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
        self.provenance = {
            "backend": "flowsim",
            "sharing_scope": "each_directed_physical_hop",
            "completion_source": "flowsim_event_simulator",
            "propagation_accounting": "orbitserve_after_serialization",
            "packetization": False,
            "aggregate_sharing": "max_min_directed_link_and_satellite_rf_pools",
            "payload_quantization_bytes": 1 / BYTE_SCALE,
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
        before = self.simulator.remaining()
        start = self.now
        target_ns = time_s * 1e9
        self.simulator.advance_until(target_ns)
        native_now = self.simulator.now_ns
        # A round trip through nanoseconds can round one ULP below the
        # original seconds value. Snap only after the native target was reached.
        self.now = time_s if native_now >= target_ns else native_now / 1e9
        after = self.simulator.remaining()
        sent = {identity: max(0.0, amount - after.get(identity, 0.0)) / BYTE_SCALE
                for identity, amount in before.items()}
        done, self.completed = sorted(self.completed), []
        if time_s > start and self.now <= start and not done:
            raise RuntimeError(
                f"FlowSim made no observable progress at {start}s toward {time_s}s"
            )
        for identity in done:
            self.active.pop(identity)
        return {"time_s": self.now, "interval_start_s": start,
                "bytes_sent": sent, "completed": done}

    def cancel(self, identities):
        for identity in identities:
            if identity in self.active:
                self.simulator.cancel(identity)
                del self.active[identity]

    def close(self, force=False):
        del force
        self.cancel(tuple(self.active))
