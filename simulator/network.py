"""One Runtime clock/transport for local devices and orbital links."""

from dataclasses import replace
from decimal import ROUND_CEILING, Decimal
from types import SimpleNamespace

from orbitserve.runtime import RuntimeStrategySpec
from orbitserve.runtime.engine.cell_bridge import RuntimeCellNetwork
from orbitserve.runtime.engine.events import _EventEngine
from orbitserve.runtime.state.device_network import DeviceNetworkState
from orbitserve.runtime.state.runtime_state import RuntimeState
from orbitserve.runtime.timeline import GlobalTimeline


class DeviceRuntimeHost(_EventEngine):
    def __init__(self, topology, config, physical=None):
        self.record_packets = getattr(config, "record_packets", False)
        network = DeviceNetworkState(
            topology,
            config.horizon_ns / 1e9,
            config.backend,
            physical=physical,
            bindings=config.physical_link_bindings,
            ground_owners=getattr(config, "ground_owners", ()),
            failure_policy=config.runtime_network.get("failure_policy", "fail"),
            max_transfer_retries=config.runtime_network.get("max_transfer_retries", 0),
        )
        spec = SimpleNamespace(
            network=network.spec,
            serving_profile=None,
            strategy=RuntimeStrategySpec(),
            time_horizon_s=config.horizon_ns / 1e9,
        )
        environment = SimpleNamespace(
            network=network, time={"start_s": 0.0, "end_s": network.end_s}
        )
        super().__init__(spec, environment, None, RuntimeState(), ())
        for time_s in network.environment_times:
            self.push(time_s, 0, "topology_sample")
        for event in network.events:
            self.push(event["time_s"], 0, "link_state", payload=event)
        if physical is not None:
            batches = {}
            for time_s, index, restored, fault in physical.fault_observations():
                batches.setdefault(time_s, []).append((index, fault, restored))
            for time_s, observations in batches.items():
                self.push(time_s, -2, "fault_batch", payload=observations)
        self.timeline = GlobalTimeline(host=self)

    def debit(self, job, node_id, *, prompt=0, output=0, size=0, stage):
        # Native kernel energy is not inferred from byte traffic or fake SOC.
        return None


class DeviceCommunicationRuntime(RuntimeCellNetwork):
    def __init__(self, host, topology):
        super().__init__(host)
        self.topology = topology

    def subscribe(self, callback):
        def delivered(event):
            # The native graph uses integer ns. Physical service remains exact;
            # notify at the next native tick, never before physical delivery.
            at_ns = int(
                (Decimal(str(self.host.now)) * 1_000_000_000).to_integral_value(
                    rounding=ROUND_CEILING
                )
            )
            event = replace(
                event,
                **(
                    {"completion_time_ns": at_ns}
                    if event.kind == "completed"
                    else {"failure_time_ns": at_ns}
                ),
            )
            self.host.push(
                at_ns / 1e9, 32, "external_callback", payload=lambda: callback(event)
            )

        super().subscribe(delivered)

    def resolve_pair(self, source, target):
        if source == target:
            if source not in self.topology.devices:
                raise ValueError("unknown device endpoint")
            node = self.topology.egress[source]
            return node, node
        return self.topology.endpoints(source, target)

    def map_path(self, path):
        return self.topology.physical_path(path)

    def route(self, source, target, size_bytes=0):
        a, b = self.resolve_pair(source, target)
        arrival, path = self.host.net.transfer(a, b, self.host.now, size_bytes)
        nodes = (a, *(leg["target_node_id"] for leg in path))
        return {
            "path": self.map_path(nodes),
            "isolated_estimate_ns": (arrival - self.host.now) * 1e9,
        }

    def snapshot(self):
        return {
            "backend": self.host.net.spec.transport_model,
            "clock_s": self.host.now,
            "transport": self.host.transport_provenance,
            "flows": list(self.events),
            "device_topology": self.topology.view("devices"),
            "physical_dataset_identity": getattr(
                self.host.net.physical, "dataset_identity", None
            ),
            "physical_dataset_root": getattr(
                self.host.net.physical, "dataset_root", None
            ),
            "physical_link_bindings": self.host.net._bindings,
            "energy_model": "not_evaluated",
            "event_log": self.host.log,
        }
