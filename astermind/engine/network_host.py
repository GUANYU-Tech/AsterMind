"""One Runtime clock/transport for local devices and orbital links."""

from types import SimpleNamespace

from astermind.engine.cell_bridge import RuntimeCellNetwork, CellTransferExecutor as CellTransferExecutor
from astermind.engine.events import _EventEngine
from astermind.engine.state.device_network import DeviceNetworkState
from astermind.engine.timeline import GlobalTimeline
from astermind.engine.clock import Time, CLOCK
from astermind.engine.timeline import nanoseconds
from astermind.engine.cell_bridge import CellTransferEvent


class DeviceRuntimeHost(_EventEngine):
    def __init__(self, topology, config, physical=None):
        timeline_mode="orbitserve"
        network = DeviceNetworkState(
            topology,
            Time.from_ns(config.horizon_ns),
            "runtime_comm",
            physical=physical,
            bindings=config.physical_link_bindings,
            ground_owners=getattr(config, "ground_owners", ()),
        )
        network.timeline_mode = timeline_mode
        spec = SimpleNamespace(
            network=network.spec,
            routing_policy="earliest_arrival",
            time_horizon_s=Time.from_ns(config.horizon_ns),
        )
        environment = SimpleNamespace(
            network=network, time={"start_s": 0.0, "end_s": network.end_s}
        )
        super().__init__(spec, environment)
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
        self.comm_policy = config.communication
        self.communication_policy = None

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
        from astermind.engine.comm.target_network import TargetNetworkFrames
        from astermind.engine.comm.event_runtime import RuntimeCommunication
        self.communication_policy = RuntimeCommunication(TargetNetworkFrames(self), host.timeline, host.comm_policy)
        host.communication_policy = self.communication_policy
        self.pending = {}

    def submit(self, request):
        if self.closed:
            raise RuntimeError("Cell network is closed")
        if request.id in self.pending or request.id in self.communication_policy.used_ids:
            raise ValueError("duplicate Cell transfer")
        if type(request.size_bytes) is not int or request.size_bytes < 0:
            raise ValueError("Cell transfer requires nonnegative integer bytes")
        ready = self.ready_time(request)
        self.resolve_pair(request.source, request.target)
        self.pending[request.id] = request

        def finish(failure=None):
            if self.pending.pop(request.id, None) is None:
                return
            now = self.host.now
            event = CellTransferEvent(
                "completed" if failure is None else "failed", request.id,
                request.source, request.target, request.size_bytes,
                request.origin, request.operation, request.correlation_id,
                tuple(self.communication_policy.delivery_paths.get(request.id, (request.source, request.target))),
                completion_time_ns=nanoseconds(now) if failure is None else None,
                failure_time_ns=nanoseconds(now) if failure is not None else None,
                reason=None if failure is None else failure.reason, time_ticks=str(now.ticks))
            self.events.append(event.model_dump())
            for listener in tuple(self._listeners):
                listener(event)

        def begin():
            if request.id not in self.pending:
                return
            self.communication_policy.submit_transfer(request.id, request.source, request.target,
                request.size_bytes, finish, finish)
        self.host.timeline.schedule(ready, 1, "comm:submit_transfer", begin)
        return request.id

    def cancel(self, identifiers, reason="cell_transfer_cancelled"):
        for identifier in tuple(identifiers):
            if identifier in self.communication_policy.tasks:
                self.communication_policy.cancel(identifier, reason)
            elif identifier in self.pending:
                request = self.pending.pop(identifier)
                event = CellTransferEvent("failed", identifier, request.source, request.target,
                    request.size_bytes, request.origin, request.operation, request.correlation_id, (),
                    failure_time_ns=nanoseconds(self.host.now), reason=reason,
                    time_ticks=str(self.host.now.ticks))
                self.events.append(event.model_dump())
                for listener in tuple(self._listeners):
                    listener(event)

    def close(self):
        if self.closed:
            return
        self.cancel(tuple(self.pending), "cell_network_closed")
        for identifier in tuple(self.communication_policy.tasks):
            self.communication_policy.cancel(identifier, "cell_network_closed")
        self.communication_policy.close()
        self.closed = True

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
        self.communication_policy._refresh()._publish_forecast()
        a, b = self.resolve_pair(source, target)
        arrival, path = self.host.net.transfer(a, b, self.host.now, size_bytes)
        nodes = (a, *(leg["target_node_id"] for leg in path))
        return {
            "path": self.map_path(nodes),
            "isolated_estimate_ns": (arrival - self.host.now) * 1e9,
        }

    def snapshot(self):
        return {
            "backend": "runtime_comm",
            "timeline_authority": "integer_ticks",
            "clock_s": self.host.now, "clock_ticks": str(self.host.now.ticks), "clock": CLOCK,
            "transport": {"allocator": self.communication_policy.allocator.provenance,
                          "semantics": "whole_path_external_resources",
                          "intra_competition": False},
            "flows": list(self.events),
            "device_topology": self.topology.view("devices"),
            "physical_dataset_identity": getattr(self.host.net.physical, "dataset_identity", None),
            "physical_dataset_root": getattr(self.host.net.physical, "dataset_root", None),
            "physical_link_bindings": self.host.net._bindings,
            "energy_model": "not_evaluated", "event_log": self.host.log,
        }
