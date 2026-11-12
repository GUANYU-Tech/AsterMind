"""One Runtime clock/transport for local devices and orbital links."""

from types import SimpleNamespace

from orbitserve.runtime import RuntimeStrategySpec
from orbitserve.runtime.engine.cell_bridge import RuntimeCellNetwork, CellTransferExecutor as CellTransferExecutor
from orbitserve.runtime.engine.events import _EventEngine
from orbitserve.runtime.state.device_network import DeviceNetworkState
from orbitserve.runtime.state.runtime_state import RuntimeState
from orbitserve.runtime.timeline import GlobalTimeline
from orbitserve.runtime.clock import Time


class DeviceRuntimeHost(_EventEngine):
    def __init__(self, topology, config, physical=None):
        timeline_mode=getattr(config,'timeline_mode','simgrid' if config.backend=='simgrid' else 'orbitserve')
        if timeline_mode not in {'orbitserve','simgrid'} or (timeline_mode=='simgrid') != (config.backend=='simgrid'):
            raise ValueError('timeline mode and communication backend are inconsistent')
        self.record_packets = getattr(config, "record_packets", False)
        network = DeviceNetworkState(
            topology,
            Time.from_ns(config.horizon_ns),
            config.backend,
            physical=physical,
            bindings=config.physical_link_bindings,
            ground_owners=getattr(config, "ground_owners", ()),
            failure_policy=config.runtime_network.get("failure_policy", "fail"),
            max_transfer_retries=config.runtime_network.get("max_transfer_retries", 0),
        )
        network.timeline_mode = timeline_mode
        spec = SimpleNamespace(
            network=network.spec,
            serving_profile=None,
            strategy=RuntimeStrategySpec(),
            time_horizon_s=Time.from_ns(config.horizon_ns),
        )
        environment = SimpleNamespace(
            network=network, time={"start_s": 0.0, "end_s": network.end_s}
        )
        super().__init__(spec, environment, None, RuntimeState(), ())
        for time_s in network.environment_times:
            self.push(time_s, 0, "topology_sample")
            if self.controller is not None and time_s < self.end:
                self.push(time_s, 5, "control_boundary")
        for event in network.events:
            self.push(event["time_s"], 0, "link_state", payload=event)
        if physical is not None:
            batches = {}
            for time_s, index, restored, fault in physical.fault_observations():
                batches.setdefault(time_s, []).append((index, fault, restored))
            for time_s, observations in batches.items():
                self.push(time_s, -2, "fault_batch", payload=observations)
        self.timeline = GlobalTimeline(host=self)
        self.communication_mode = getattr(config, "communication_mode", "flowsim")
        self.protobridge_startup_us = getattr(config, "protobridge_startup_us", 0.0)

    def debit(self, job, node_id, *, prompt=0, output=0, size=0, stage):
        # Native kernel energy is not inferred from byte traffic or fake SOC.
        return None

    def schedule_reallocation(self):
        super().schedule_reallocation()
        policy = getattr(self, "communication_policy", None)
        if policy is not None:
            policy.network_changed()


class DeviceCommunicationRuntime(RuntimeCellNetwork):
    def __init__(self, host, topology):
        super().__init__(host)
        self.topology = topology
        self.communication_policy = None
        if host.communication_mode != "flowsim":
            from .communication import ScopedProtoBridge

            self.communication_policy = ScopedProtoBridge(self, host.communication_mode)
        host.communication_policy = self.communication_policy

    def submit(self, request):
        if self.communication_policy is None:
            return super().submit(request)
        return self.communication_policy.submit(request, super().submit)

    def try_collective(self, cell_id, transfer, devices, record, on_complete):
        if self.communication_policy is None:
            return False
        return self.communication_policy.collective(
            cell_id, transfer, devices, record, on_complete
        )

    def cancel(self, identifiers, reason="cell_transfer_cancelled"):
        identifiers = tuple(identifiers)
        if self.communication_policy is not None:
            self.communication_policy.cancel(identifiers, reason)
        return super().cancel(identifiers, reason)

    def close(self):
        self.closed = True
        if self.communication_policy is not None:
            self.communication_policy.cancel(
                tuple(self.communication_policy.requests), "cell_network_closed"
            )
            self.communication_policy.pending.clear()
            self.communication_policy.active.clear()
        return super().close()

    def subscribe(self, callback):
        def delivered(event):
            if self.closed:
                return
            # Queue the dependency notification at this exact physical tick.
            self.host.push(
                self.host.now, 32, "external_callback", payload=lambda: callback(event)
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
            "communication_mode": self.host.communication_mode,
            "timeline_mode": getattr(self.host.net, 'timeline_mode', 'orbitserve'),
            "timeline_authority": 'simgrid_engine' if self.host.global_scheduler else 'orbitserve_runtime',
            "scheduler": getattr(self.host.transport, 'scheduler_stats', {}),
            "surrogate_operations": self.communication_policy.records
            if self.communication_policy
            else [],
            "clock_s": self.host.now,
            "clock": self.host.transport_provenance.get('clock'),
            "clock_ticks": str(self.host.now.ticks),
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
