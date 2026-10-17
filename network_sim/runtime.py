"""Externally clocked native FlowSim with terminal acquisition and failure events."""
from __future__ import annotations

import ctypes
import json
import math
from pathlib import Path
import sys

from .link_policy import LinkController
from .model import CompletionEvent, FailureEvent, FlowStatus, TransferRequest
from .topology import PhysicalTopology


class CommunicationRuntime:
    """Single-threaded. One runtime owns all transfers sharing these resources."""

    def __init__(self, topology: PhysicalTopology, library: str | Path | None = None,
                 link_selector=None):
        self.topology = topology
        self.controller = LinkController(topology, link_selector) if topology.scenario.link_policy else None
        if link_selector is not None and self.controller is None:
            raise ValueError("custom link selector requires link_policy")
        default = Path(__file__).parent / "build" / (
            "network_native.dll" if sys.platform == "win32" else "libnetwork_native.so")
        path = Path(library) if library else default
        if not path.exists():
            raise RuntimeError("native library is missing; run python -m network_sim.build_native")
        self._lib = ctypes.CDLL(str(path.resolve()))
        self._bind()
        self._handle = self._lib.net_create()
        self._requests, self._pending = {}, {}
        self._delivered = set()
        self._edge_enabled = {}
        self.now_ns = 0
        try:
            for node in topology.native_nodes:
                self._call("net_add_node", node.encode())
            for edge in topology.native_edges:
                self._call("net_add_link", edge.source.encode(), edge.target.encode(),
                           edge.bandwidth_gbps, edge.latency_ns)
                self._edge_enabled[(edge.source, edge.target)] = True
            self._apply_links()
            self._call("net_initialize")
        except Exception:
            self.close()
            raise

    def _bind(self):
        pointer, text, number = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64
        definitions = {
            "net_create": ([], pointer), "net_destroy": ([pointer], None),
            "net_error": ([pointer], text), "net_output": ([pointer], text),
            "net_add_node": ([pointer, text], ctypes.c_int),
            "net_add_link": ([pointer, text, text, ctypes.c_double, number], ctypes.c_int),
            "net_set_link_enabled": ([pointer, text, text, ctypes.c_int], ctypes.c_int),
            "net_initialize": ([pointer], ctypes.c_int),
            "net_route": ([pointer, text, text], ctypes.c_int),
            "net_candidate_route": ([pointer, text, text], ctypes.c_int),
            "net_submit": ([pointer, text, text, text, number, number], ctypes.c_int),
            "net_advance": ([pointer, number], ctypes.c_int),
            "net_next_event": ([pointer], ctypes.c_int),
            "net_snapshot": ([pointer], ctypes.c_int),
        }
        for name, (arguments, result) in definitions.items():
            function = getattr(self._lib, name)
            function.argtypes, function.restype = arguments, result

    def _call(self, name, *arguments):
        if self._handle is None:
            raise RuntimeError("runtime is closed")
        if getattr(self._lib, name)(self._handle, *arguments) != 0:
            raise ValueError(self._lib.net_error(self._handle).decode("utf-8"))

    def _output(self):
        return json.loads(self._lib.net_output(self._handle).decode("utf-8"))

    def _apply_links(self):
        if self.controller is None:
            return
        # Apply individual transitions after initialization, including down/up
        # at the same timestamp (a zero-delay reacquisition still loses a flow).
        if hasattr(self, "_history_cursor"):
            for change in self.controller.history[self._history_cursor:]:
                link = self.controller.links[change["link_id"]]
                pairs = [(link.source, link.target)]
                if link.bidirectional:
                    pairs.append((link.target, link.source))
                for source, target in pairs:
                    a, b = self.topology.egress[source], self.topology.ingress[target]
                    flag = change["state"] == "up"
                    if self._edge_enabled[(a, b)] != flag:
                        self._call("net_set_link_enabled", a.encode(), b.encode(), int(flag))
                        self._edge_enabled[(a, b)] = flag
            self._history_cursor = len(self.controller.history)
            return
        enabled = {}
        for identifier, state in self.controller.states.items():
            link = self.controller.links[identifier]
            enabled[f"link:{identifier}:{link.source}->{link.target}"] = state.state == "up"
            enabled[f"link:{identifier}:{link.target}->{link.source}"] = state.state == "up"
        # Disable first, so old flows fail before new paths are admitted.
        changes = [(edge, enabled[edge.resource]) for edge in self.topology.native_edges
                   if edge.resource in enabled and
                   self._edge_enabled[(edge.source, edge.target)] != enabled[edge.resource]]
        for edge, flag in sorted(changes, key=lambda item: item[1]):
            self._call("net_set_link_enabled", edge.source.encode(), edge.target.encode(), int(flag))
            self._edge_enabled[(edge.source, edge.target)] = flag
        self._history_cursor = len(self.controller.history)

    def route(self, source: str, target: str, size_bytes: int = 0):
        """Current active route. The estimate excludes queueing and acquisition."""
        if type(size_bytes) is not int or size_bytes < 0:
            raise ValueError("size_bytes must be a nonnegative integer")
        start, end = self.topology.endpoints(source, target)
        self._call("net_route", start.encode(), end.encode())
        result = self._output()
        native_path = result.pop("path")
        result["path"] = self.topology.physical_path(native_path)
        result["resources"] = self.topology.resources(native_path)
        result["isolated_estimate_ns"] = math.ceil(size_bytes * 8 / result["bandwidth_gbps"]) + result["latency_ns"]
        return result

    def candidate_path(self, source, target):
        """Structural path ignoring acquisition/visibility; not permission to transmit."""
        start, end = self.topology.endpoints(source, target)
        self._call("net_candidate_route", start.encode(), end.encode())
        return self.topology.physical_path(self._output())

    def submit(self, request: TransferRequest) -> str:
        if not isinstance(request, TransferRequest):
            raise TypeError("submit expects a validated TransferRequest")
        if request.id in self._requests:
            raise ValueError(f"duplicate transfer ID {request.id}")
        if request.ready_time_ns < self.now_ns:
            raise ValueError("cannot submit a transfer in the past")
        if self.controller is None:
            self.route(request.source, request.target)
        else:
            self.candidate_path(request.source, request.target)
        self._requests[request.id] = request
        self._pending[request.id] = request
        return request.id

    def _dispatch_ready(self):
        started = False
        for identifier, request in list(self._pending.items()):
            if request.ready_time_ns > self.now_ns:
                continue
            try:
                self.route(request.source, request.target)
            except ValueError as error:
                if self.controller is not None and str(error).startswith("no route from "):
                    continue
                raise
            start, end = self.topology.endpoints(request.source, request.target)
            self._call("net_submit", identifier.encode(), start.encode(), end.encode(),
                       request.size_bytes, self.now_ns)
            del self._pending[identifier]
            started = True
        if started:
            self._call("net_advance", self.now_ns)

    def next_event_time_ns(self) -> int | None:
        self._call("net_next_event")
        times = [self._output()]
        if self.controller:
            times.append(self.controller.next_event_time_ns())
        for request in self._pending.values():
            if request.ready_time_ns > self.now_ns:
                times.append(request.ready_time_ns)
            else:
                try:
                    self.route(request.source, request.target)
                    times.append(self.now_ns)
                except ValueError as error:
                    if not str(error).startswith("no route from "):
                        raise
        return min((t for t in times if t is not None), default=None)

    def _events(self):
        report, events = self.snapshot(), []
        for key, event_type in (("flows", CompletionEvent), ("failures", FailureEvent)):
            for flow in report[key]:
                if flow["id"] in self._delivered:
                    continue
                request = self._requests[flow["id"]]
                fields = dict(id=request.id, origin=request.origin, operation=request.operation,
                              correlation_id=request.correlation_id, source=request.source,
                              target=request.target, size_bytes=request.size_bytes, path=tuple(flow["path"]))
                if key == "flows":
                    fields["completion_time_ns"] = flow["completion_time_ns"]
                else:
                    fields.update(failure_time_ns=flow["failure_time_ns"], reason=flow["reason"],
                                  remaining_bytes=flow["remaining_bytes"])
                events.append(event_type(**fields))
                self._delivered.add(request.id)
        return sorted(events, key=lambda e: (e.completion_time_ns if e.kind == "completed" else e.failure_time_ns, e.id))

    def advance_to(self, time_ns: int) -> tuple[CompletionEvent | FailureEvent, ...]:
        if type(time_ns) is not int or not self.now_ns <= time_ns <= 2**64 - 1:
            raise ValueError("time_ns must be an integer not before the current clock")
        events = []
        while True:
            next_time = self.next_event_time_ns()
            boundary = time_ns if next_time is None or next_time > time_ns else next_time
            self._call("net_advance", boundary)
            self.now_ns = boundary
            if self.controller:
                self.controller.advance_to(boundary)
                self._apply_links()
            self._dispatch_ready()
            events.extend(self._events())
            if boundary == time_ns:
                break
        return tuple(events)

    def run(self, until_ns: int | None = None) -> tuple[CompletionEvent | FailureEvent, ...]:
        """Dynamic runs require a finite horizon; integrated callers drive the clock."""
        if self.controller and until_ns is None:
            raise ValueError("dynamic link policy requires run(until_ns=...) or external clock advancement")
        if until_ns is not None:
            return self.advance_to(until_ns)
        events = []
        while (time_ns := self.next_event_time_ns()) is not None:
            events.extend(self.advance_to(time_ns))
        return tuple(events)

    def snapshot(self):
        self._call("net_snapshot")
        result = self._output()
        completed = {flow["id"]: flow for flow in result["flows"]}
        for flow in result["flows"] + result["failures"]:
            flow["path"] = self.topology.physical_path(flow["path"])
            request = self._requests[flow["id"]]
            flow.update(source=request.source, target=request.target, origin=request.origin,
                        operation=request.operation, correlation_id=request.correlation_id)
        progress = {p["id"]: p for p in result.pop("progress")}
        result["statuses"] = []
        for identifier, request in self._requests.items():
            p = progress.get(identifier)
            if p is not None:
                state = "failed" if p["failed"] else "completed" if p["completed"] else "transmitting" if p["transmitting"] else "propagating"
            else:
                state = "scheduled" if request.ready_time_ns > self.now_ns else "waiting_link"
            flow = completed.get(identifier, {})
            result["statuses"].append(FlowStatus(id=identifier, state=state,
                remaining_bytes=request.size_bytes if p is None else p["remaining_bytes"],
                rate_gbps=0 if p is None else p["rate_gbps"],
                transmission_end_ns=flow.get("transmission_end_ns"),
                completion_ns=flow.get("completion_time_ns")).model_dump())
        for resource in result["resources"]:
            resource["id"] = self.topology.edge_resource[(resource.pop("source"), resource.pop("target"))]
        result["model"] = "AsterMind FlowSim / pipelined fluid / max-min sharing"
        result["link_policy"] = self.controller.snapshot() if self.controller else None
        result["byte_counter_scope"] = "finished_transmissions_only; use per-flow remaining_bytes for in-flight diagnostics"
        return result

    def close(self):
        if getattr(self, "_handle", None) is not None:
            self._lib.net_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
