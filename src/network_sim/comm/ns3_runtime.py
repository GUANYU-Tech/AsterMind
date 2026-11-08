"""Externally clocked real ns-3 packets over the same compiled resource graph.

Point-to-point queues and PPP serialization, deterministic physical routes.
No TCP, NCCL protocol or optical BER/retransmission model is claimed.
"""
import ctypes as c
from pathlib import Path

from .model import CompletionEvent, FailureEvent, TransferRequest
from .runtime import CommunicationRuntime


DEFAULT_LIBRARY = Path(__file__).resolve().parents[1] / "build/libserving_ns3.so"


class NS3Runtime:
    def __init__(self, topology, *, packet_bytes=1400, library=None):
        if topology.scenario.link_policy is not None:
            raise NotImplementedError("first ns-3 adapter is static; use FlowSim for acquisition/visibility windows")
        if type(packet_bytes) is not int or not 1 <= packet_bytes <= 65000:
            raise ValueError("packet_bytes must be in [1,65000]")
        self.topology, self.now_ns = topology, 0
        path = Path(library) if library else DEFAULT_LIBRARY
        if not path.is_file():
            raise RuntimeError("ns-3 bridge missing; run python -m simulator.build_ns3 in Linux/WSL")
        self._lib = c.CDLL(str(path.resolve()))
        definitions = dict(n3_version=([], c.c_char_p), n3_create=([c.c_uint32], c.c_void_p), n3_destroy=([c.c_void_p], None),
            n3_error=([c.c_void_p], c.c_char_p), n3_add_edge=([c.c_void_p, c.c_double, c.c_uint64], c.c_int),
            n3_submit=([c.c_void_p, c.c_uint64, c.c_uint64, c.c_uint64, c.POINTER(c.c_int), c.c_int], c.c_int),
            n3_advance=([c.c_void_p, c.c_uint64], c.c_int), n3_next=([c.c_void_p], c.c_uint64),
            n3_poll=([c.c_void_p, c.POINTER(c.c_uint64), c.POINTER(c.c_uint64), c.POINTER(c.c_uint64), c.POINTER(c.c_int)], c.c_int),
            n3_packets=([c.c_void_p], c.c_uint64))
        for name, (args, result) in definitions.items():
            method = getattr(self._lib, name)
            method.argtypes, method.restype = args, result
        self._handle = self._lib.n3_create(packet_bytes)
        if not self._handle:
            raise RuntimeError("ns-3 runtime already active or initialization failed")
        self.packet_bytes = packet_bytes
        self._requests, self._routes, self._events = {}, {}, []
        self._route_engine = None
        try:
            # Only route queries are used. No FlowSim flow or timing is added.
            self._route_engine = CommunicationRuntime(topology)
            self._edges = {(e.source, e.target): i for i, e in enumerate(topology.native_edges)}
            for index, edge in enumerate(topology.native_edges):
                if edge.latency_ns > 2**63 - 1 or edge.bandwidth_gbps * 1e9 > 2**64 - 1:
                    raise ValueError("link timing/rate exceeds ns-3 integer range")
                if self._lib.n3_add_edge(self._handle, edge.bandwidth_gbps, edge.latency_ns) != index:
                    raise ValueError(self._lib.n3_error(self._handle).decode())
        except BaseException:
            self.close()
            raise

    def route(self, source, target, size_bytes=0):
        self._require_open()
        return self._route_engine.route(source, target, size_bytes)

    def candidate_path(self, source, target):
        self._require_open()
        return self._route_engine.candidate_path(source, target)

    def submit(self, request):
        self._require_open()
        if not isinstance(request, TransferRequest) or request.id in {r.id for r in self._requests.values()}:
            raise ValueError("validated, unique transfer required")
        if not self.now_ns <= request.ready_time_ns <= 2**63 - 1:
            raise ValueError("cannot submit in the past")
        source, target = self.topology.endpoints(request.source, request.target)
        self._route_engine._call("net_route", source.encode(), target.encode())
        route = self._route_engine._output()["path"]
        indices = [self._edges[a, b] for a, b in zip(route, route[1:])]
        identifier = len(self._requests)
        array = (c.c_int * len(indices))(*indices)
        self._check(self._lib.n3_submit(self._handle, identifier, request.size_bytes, request.ready_time_ns, array, len(indices)))
        self._requests[identifier] = request
        self._routes[identifier] = tuple(self.topology.physical_path(route))
        return request.id

    def _check(self, code):
        if code:
            raise ValueError(self._lib.n3_error(self._handle).decode())

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("ns-3 runtime is closed")

    def next_event_time_ns(self):
        self._require_open()
        value = self._lib.n3_next(self._handle)
        return None if value == 2**64 - 1 else value

    def advance_to(self, time_ns):
        self._require_open()
        if type(time_ns) is not int or not self.now_ns <= time_ns <= 2**63 - 1:
            raise ValueError("invalid ns-3 clock")
        self._check(self._lib.n3_advance(self._handle, time_ns))
        self.now_ns = time_ns
        result = []
        identifier, at, remaining, failed = c.c_uint64(), c.c_uint64(), c.c_uint64(), c.c_int()
        while self._lib.n3_poll(self._handle, c.byref(identifier), c.byref(at), c.byref(remaining), c.byref(failed)):
            request = self._requests[identifier.value]
            fields = dict(id=request.id, source=request.source, target=request.target,
                origin=request.origin, operation=request.operation, correlation_id=request.correlation_id,
                size_bytes=request.size_bytes, path=self._routes[identifier.value])
            event = (FailureEvent(**fields, failure_time_ns=at.value, reason="ns3_queue_drop", remaining_bytes=remaining.value)
                     if failed.value else CompletionEvent(**fields, completion_time_ns=at.value))
            result.append(event)
            self._events.append(event.model_dump())
        return tuple(result)

    def run(self, until_ns=None):
        if until_ns is not None:
            return self.advance_to(until_ns)
        result = []
        while (at := self.next_event_time_ns()) is not None:
            result.extend(self.advance_to(at))
        return tuple(result)

    def snapshot(self):
        self._require_open()
        return dict(model=f"ns-3 {self._lib.n3_version().decode()} / PointToPointNetDevice / DropTail / PPP / explicit forwarding",
            now_ns=self.now_ns, packet_payload_bytes=self.packet_bytes,
            queue_limit_packets=1000000,
            packet_hop_transmissions=self._lib.n3_packets(self._handle),
            flows=[event for event in self._events if event["kind"] == "completed"],
            failures=[event for event in self._events if event["kind"] == "failed"],
            limitations=["static links", "no TCP/IP or retransmission", "one active ns-3 runtime per process"])

    def close(self):
        if getattr(self, "_route_engine", None) is not None:
            self._route_engine.close()
            self._route_engine = None
        if getattr(self, "_handle", None):
            self._lib.n3_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
