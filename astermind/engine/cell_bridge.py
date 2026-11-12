"""Host Native Cell contracts on the dynamic Runtime network.

No native model is imported here. A Cell factory supplies its scheduler and
kernel executor; all callbacks, collectives and KV copies use the host clock.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

from astermind.engine.clock import Time


@dataclass(frozen=True)
class CellTransfer:
    id: str
    source: str
    target: str
    size_bytes: float
    ready_time_ns: float
    origin: str = "midlayer"
    operation: str = "transfer"
    correlation_id: str | None = None
    ready_time_ticks: str | None = None

    def model_dump(self):
        return dict(id=self.id, source=self.source, target=self.target,
                    size_bytes=self.size_bytes, ready_time_ns=self.ready_time_ns,
                    origin=self.origin, operation=self.operation,
                    correlation_id=self.correlation_id, ready_time_ticks=self.ready_time_ticks)


@dataclass(frozen=True)
class CellTransferEvent:
    kind: str
    id: str
    source: str
    target: str
    size_bytes: float
    origin: str
    operation: str
    correlation_id: str | None
    path: tuple[str, ...]
    completion_time_ns: float | None = None
    failure_time_ns: float | None = None
    reason: str | None = None
    time_ticks: str | None = None

    def model_dump(self):
        # Contract fields are immutable scalar values and a tuple of node IDs.
        # Avoid recursively deepcopying every field for each completion listener.
        result = dict(kind=self.kind, id=self.id, source=self.source, target=self.target,
                      size_bytes=self.size_bytes, origin=self.origin,
                      operation=self.operation, correlation_id=self.correlation_id,
                      path=self.path, completion_time_ns=self.completion_time_ns,
                      failure_time_ns=self.failure_time_ns, reason=self.reason, time_ticks=self.time_ticks)
        if self.kind == "completed":
            result.pop("failure_time_ns")
            result.pop("reason")
        else:
            result.pop("completion_time_ns")
            result["remaining_bytes"] = self.size_bytes
        return result


class RuntimeCellNetwork:
    """Callback-driven transfers on the host's existing directed resources.

    DeviceCommunicationRuntime supplies the Comm task lifecycle.
    """

    def __init__(self, host, *, endpoints=None, cell_members=None):
        self.host = host
        self.endpoints = dict(endpoints or {})
        if any(node not in host.net.nodes for node in self.endpoints.values()):
            raise ValueError("Cell endpoint maps to an unknown physical node")
        members = dict(cell_members or {})
        self.topology = SimpleNamespace(
            devices={
                key: SimpleNamespace(kind="gpu", satellite=node)
                for key, node in self.endpoints.items()
            },
            satellite_to_cell=members,
        )
        self._listeners = []
        self.boundary_listeners = []
        self._jobs = {}
        self.events = []
        self.closed = False

    @property
    def now_ns(self):
        return self.host.now * 1e9

    def resolve(self, endpoint):
        node = self.endpoints.get(str(endpoint), str(endpoint))
        if node not in self.host.net.nodes:
            raise ValueError(f"unknown physical endpoint: {endpoint}")
        return node

    def resolve_pair(self, source, target):
        return self.resolve(source), self.resolve(target)

    def bind_endpoint(self, endpoint, node, cell_id):
        """Bind a native GPU ID to its declared host physical member."""
        if self.topology.satellite_to_cell.get(node) != cell_id:
            raise ValueError("native endpoint does not belong to this physical Cell")
        if endpoint in self.host.net.nodes and endpoint != node:
            raise ValueError("native endpoint shadows another physical node")
        if endpoint in self.endpoints and self.endpoints[endpoint] != node:
            raise ValueError(
                "native endpoint already belongs to another physical member"
            )
        self.endpoints[endpoint] = node
        self.topology.devices[endpoint] = SimpleNamespace(kind="gpu", satellite=node)

    def subscribe(self, callback):
        self._listeners.append(callback)

    def candidate_path(self, source, target):
        """Validate structure only; do not assume a future contact is active."""
        source, target = self.resolve_pair(source, target)
        adjacency = {}
        for edge in self.host.net.edges:
            a, b = edge["endpoint_a"], edge["endpoint_b"]
            allowed = getattr(self.host.net, "allows_hop", lambda source, target: True)
            if allowed(a, b):
                adjacency.setdefault(a, []).append(b)
            if allowed(b, a):
                adjacency.setdefault(b, []).append(a)
        queue = deque([(source, (source,))])
        visited = {source}
        while queue:
            node, path = queue.popleft()
            if node == target:
                return path
            for neighbor in sorted(adjacency.get(node, ())):
                if neighbor not in visited and self.host.net.route_node_allowed(
                    neighbor, source, target
                ):
                    visited.add(neighbor)
                    queue.append((neighbor, (*path, neighbor)))
        raise ValueError(f"no candidate path: {source}->{target}")

    def ready_time(self, request):
        exact=getattr(request,'ready_time_ticks',None)
        ready=Time.from_ticks(int(exact)) if exact is not None else Time.from_ns(request.ready_time_ns)
        now=self.host.now if isinstance(self.host.now,Time) else Time(self.host.now)
        if ready < now and exact is None and isinstance(request.ready_time_ns,float):
            # Legacy callbacks can only return the rounded public ns value.
            # Align that one representation of "now"; real past events and
            # every explicit integer-tick deadline remain strictly rejected.
            budget=math.ceil(math.ulp(request.ready_time_ns)*1e9)
            if request.ready_time_ns==self.now_ns and now.ticks-ready.ticks<=budget:
                ready=now
        if ready < now:
            raise ValueError("Cell transfer cannot be submitted in the past")
        return ready


    def map_path(self, path):
        return path




class CellTransferExecutor:
    """The submit_transport seam used by per-rank KV migration."""

    def __init__(self, loop, network):
        self.loop, self.network = loop, network
        self._callbacks = {}
        self.boundary_listeners = network.boundary_listeners
        self.transfer_events = []
        network.subscribe(self._deliver)

    def _deliver(self, event):
        pair = self._callbacks.pop(event.id, None)
        if pair is None:
            return
        self.transfer_events.append(event.model_dump())
        complete, failure = pair
        if event.kind == "failed":
            if failure is None:
                raise RuntimeError(f"Cell communication failed: {event.reason}")
            failure(event)
        else:
            complete()

    def submit_transport(
        self,
        identifier,
        source,
        target,
        size_bytes,
        operation,
        correlation_id,
        on_complete,
        on_failure=None,
    ):
        if identifier in self._callbacks:
            raise ValueError("duplicate active Cell transfer")
        self._callbacks[identifier] = on_complete, on_failure
        try:
            self.network.submit(
                CellTransfer(
                    identifier,
                    source,
                    target,
                    size_bytes,
                    self.network.now_ns,
                    operation=operation,
                    correlation_id=correlation_id,
                    ready_time_ticks=str(self.loop.now_s.ticks),
                )
            )
        except BaseException:
            self._callbacks.pop(identifier, None)
            raise

    def notify_batch_boundary(self):
        for notify in tuple(self.boundary_listeners):
            notify()
