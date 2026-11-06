"""Physical communication execution over one current Network State Frame."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Any, Callable, Literal, Mapping, Protocol

from .network import EstablishedLink, NetworkFrameSource, NetworkStateFrame
from .timeline import GlobalTimeline


BackendMode = Literal["flow", "packet"]


@dataclass(frozen=True)
class PhysicalTransfer:
    """Minimal post-deployment transfer understood by the platform network."""

    transfer_id: str
    source_satellite_id: str
    target_satellite_id: str
    size_bytes: float
    source_endpoint: str | None = None
    target_endpoint: str | None = None
    operation: str = "send_recv"
    priority: int = 20
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.transfer_id:
            raise ValueError("transfer ID must be nonempty")
        if not self.source_satellite_id or not self.target_satellite_id:
            raise ValueError("source and target satellites must be nonempty")
        if not math.isfinite(self.size_bytes) or self.size_bytes < 0:
            raise ValueError("transfer size must be finite and nonnegative")
        if type(self.priority) is not int:
            raise TypeError("transfer priority must be an integer")


@dataclass(frozen=True)
class RouteHop:
    link_id: str
    source_node_id: str
    target_node_id: str
    bandwidth_bps: float
    propagation_delay_s: float
    shared_capacity_pools: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PhysicalRoute:
    source_satellite_id: str
    target_satellite_id: str
    hops: tuple[RouteHop, ...]
    frame_revision: int


@dataclass(frozen=True)
class CommunicationCompleted:
    transfer_id: str
    started_at_s: float
    completed_at_s: float
    backend: str
    route: PhysicalRoute | None


@dataclass(frozen=True)
class CommunicationFailed:
    transfer_id: str
    failed_at_s: float
    reason: str


CommunicationResult = CommunicationCompleted | CommunicationFailed
CommunicationCallback = Callable[[CommunicationResult], None]


class CommunicationBackend(Protocol):
    name: str

    def submit(
        self,
        transfer: PhysicalTransfer,
        route: PhysicalRoute | None,
        on_result: CommunicationCallback,
    ) -> None: ...


class FrameRouter:
    """Route satellite-to-satellite traffic over established physical links."""

    def route(
        self,
        frame: NetworkStateFrame,
        source_satellite_id: str,
        target_satellite_id: str,
    ) -> PhysicalRoute:
        if source_satellite_id == target_satellite_id:
            return PhysicalRoute(
                source_satellite_id, target_satellite_id, (), frame.revision
            )
        satellite_nodes: dict[str, list[str]] = {}
        for node_id, state in frame.node_states.items():
            if state.state == "up":
                satellite_nodes.setdefault(state.satellite_id, []).append(node_id)
        starts = tuple(sorted(satellite_nodes.get(source_satellite_id, ())))
        targets = set(satellite_nodes.get(target_satellite_id, ()))
        if not starts or not targets:
            raise LookupError("route endpoint satellite is unavailable")
        adjacency: dict[str, list[tuple[float, str, EstablishedLink]]] = {}
        for link in frame.established_topology_graph:
            weight = link.propagation_delay_s + 1.0 / link.bandwidth_bps
            adjacency.setdefault(link.source_node_id, []).append(
                (weight, link.target_node_id, link)
            )
            adjacency.setdefault(link.target_node_id, []).append(
                (weight, link.source_node_id, link)
            )
        queue: list[
            tuple[float, int, str, tuple[tuple[str, str, EstablishedLink], ...]]
        ] = []
        sequence = 0
        for start in starts:
            heapq.heappush(queue, (0.0, sequence, start, ()))
            sequence += 1
        best: dict[str, float] = {}
        while queue:
            cost, _, node, path = heapq.heappop(queue)
            if node in targets:
                hops = tuple(
                    RouteHop(
                        link.link_id,
                        source,
                        target,
                        link.bandwidth_bps,
                        link.propagation_delay_s,
                        link.shared_capacity_pools,
                        link.attributes,
                    )
                    for source, target, link in path
                )
                return PhysicalRoute(
                    source_satellite_id,
                    target_satellite_id,
                    hops,
                    frame.revision,
                )
            if cost >= best.get(node, math.inf):
                continue
            best[node] = cost
            for weight, neighbor, link in sorted(
                adjacency.get(node, ()), key=lambda item: (item[0], item[1])
            ):
                heapq.heappush(
                    queue,
                    (cost + weight, sequence, neighbor, (*path, (node, neighbor, link))),
                )
                sequence += 1
        raise LookupError(
            f"no established route: {source_satellite_id}->{target_satellite_id}"
        )


class CommunicationExecutor:
    """Select routing and exactly one of the three supported backends."""

    def __init__(
        self,
        timeline: GlobalTimeline,
        network_frames: NetworkFrameSource,
        *,
        intra_satellite: CommunicationBackend,
        inter_satellite_flow: CommunicationBackend,
        packet_ns3: CommunicationBackend,
        mode: BackendMode = "flow",
        router: FrameRouter | None = None,
    ):
        if mode not in {"flow", "packet"}:
            raise ValueError("communication mode must be flow or packet")
        names = {
            intra_satellite.name,
            inter_satellite_flow.name,
            packet_ns3.name,
        }
        if len(names) != 3:
            raise ValueError("the three communication backends require unique names")
        self.timeline = timeline
        self.network_frames = network_frames
        self.intra_satellite = intra_satellite
        self.inter_satellite_flow = inter_satellite_flow
        self.packet_ns3 = packet_ns3
        self.mode = mode
        self.router = router or FrameRouter()

    def submit(
        self,
        transfer: PhysicalTransfer,
        on_result: CommunicationCallback,
    ) -> None:
        if not isinstance(transfer, PhysicalTransfer):
            raise TypeError("CommunicationExecutor requires PhysicalTransfer")
        if not callable(on_result):
            raise TypeError("communication completion callback must be callable")
        frame = self.network_frames.frame_at(self.timeline.now_s)
        if transfer.source_satellite_id == transfer.target_satellite_id:
            self.intra_satellite.submit(transfer, None, on_result)
            return
        try:
            route = self.router.route(
                frame,
                transfer.source_satellite_id,
                transfer.target_satellite_id,
            )
        except LookupError as exc:
            on_result(
                CommunicationFailed(
                    transfer.transfer_id, self.timeline.now_s, str(exc)
                )
            )
            return
        backend = (
            self.inter_satellite_flow if self.mode == "flow" else self.packet_ns3
        )
        if transfer.size_bytes == 0:
            started = self.timeline.now_s
            self.timeline.schedule(
                started,
                transfer.priority,
                "communication_completed",
                lambda: on_result(
                    CommunicationCompleted(
                        transfer.transfer_id,
                        started,
                        self.timeline.now_s,
                        backend.name,
                        route,
                    )
                ),
                owner_id=transfer.transfer_id,
                metadata={"backend": backend.name, "zero_byte": True},
            )
            return
        backend.submit(transfer, route, on_result)


class IntraSatelliteSurrogateBackend:
    """Whole-operation star-internal surrogate; no inter-satellite routing."""

    name = "intra_satellite_surrogate"

    def __init__(
        self,
        timeline: GlobalTimeline,
        predict_duration_s: Callable[[PhysicalTransfer], float],
    ):
        self.timeline = timeline
        self.predict_duration_s = predict_duration_s

    def submit(
        self,
        transfer: PhysicalTransfer,
        route: PhysicalRoute | None,
        on_result: CommunicationCallback,
    ) -> None:
        if route is not None:
            raise ValueError("intra-satellite backend does not accept a physical route")
        duration = float(self.predict_duration_s(transfer))
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("intra-satellite surrogate returned an invalid duration")
        started = self.timeline.now_s
        self.timeline.schedule_after(
            duration,
            transfer.priority,
            "communication_completed",
            lambda: on_result(
                CommunicationCompleted(
                    transfer.transfer_id,
                    started,
                    self.timeline.now_s,
                    self.name,
                    None,
                )
            ),
            owner_id=transfer.transfer_id,
            metadata={"backend": self.name},
        )


class FlowSimDriver(Protocol):
    def start_transfer(
        self,
        transfer_id: str,
        hops: tuple[RouteHop, ...],
        size_bytes: float,
        on_complete: Callable[[float], None],
        on_failure: Callable[[str], None],
    ) -> None: ...


class InterSatelliteFlowBackend:
    """One backend: inter-satellite surrogate parameters consumed by FlowSim."""

    name = "inter_satellite_surrogate_flowsim"

    def __init__(self, timeline: GlobalTimeline, driver: FlowSimDriver):
        if not hasattr(driver, "start_transfer"):
            raise TypeError("FlowSim driver must provide start_transfer()")
        self.timeline = timeline
        self.driver = driver

    def submit(
        self,
        transfer: PhysicalTransfer,
        route: PhysicalRoute | None,
        on_result: CommunicationCallback,
    ) -> None:
        if route is None or not route.hops:
            raise ValueError("inter-satellite FlowSim backend requires a route")
        started = self.timeline.now_s

        def completed(completed_at_s: float) -> None:
            if completed_at_s < self.timeline.now_s:
                raise ValueError("FlowSim completion cannot precede global time")
            self.timeline.schedule(
                completed_at_s,
                transfer.priority,
                "communication_completed",
                lambda: on_result(
                    CommunicationCompleted(
                        transfer.transfer_id,
                        started,
                        self.timeline.now_s,
                        self.name,
                        route,
                    )
                ),
                owner_id=transfer.transfer_id,
                metadata={"backend": self.name},
            )

        def failed(reason: str) -> None:
            on_result(
                CommunicationFailed(transfer.transfer_id, self.timeline.now_s, reason)
            )

        self.driver.start_transfer(
            transfer.transfer_id,
            route.hops,
            transfer.size_bytes,
            completed,
            failed,
        )


class PacketDriver(Protocol):
    def start_transfer(
        self,
        transfer_id: str,
        hops: tuple[RouteHop, ...],
        size_bytes: float,
        start_s: float,
    ) -> float: ...


class Ns3PacketBackend:
    """Packet-level ns-3 backend selected explicitly; never falls back."""

    name = "packet_ns3"

    def __init__(self, timeline: GlobalTimeline, driver: PacketDriver):
        if not hasattr(driver, "start_transfer"):
            raise TypeError("ns-3 driver must provide start_transfer()")
        self.timeline = timeline
        self.driver = driver

    def submit(
        self,
        transfer: PhysicalTransfer,
        route: PhysicalRoute | None,
        on_result: CommunicationCallback,
    ) -> None:
        if route is None or not route.hops:
            raise ValueError("ns-3 backend requires an inter-satellite route")
        started = self.timeline.now_s
        try:
            completed_at = float(
                self.driver.start_transfer(
                    transfer.transfer_id,
                    route.hops,
                    transfer.size_bytes,
                    started,
                )
            )
        except Exception as exc:
            on_result(
                CommunicationFailed(transfer.transfer_id, self.timeline.now_s, str(exc))
            )
            return
        if not math.isfinite(completed_at) or completed_at < started:
            raise ValueError("ns-3 returned an invalid completion time")
        self.timeline.schedule(
            completed_at,
            transfer.priority,
            "communication_completed",
            lambda: on_result(
                CommunicationCompleted(
                    transfer.transfer_id,
                    started,
                    self.timeline.now_s,
                    self.name,
                    route,
                )
            ),
            owner_id=transfer.transfer_id,
            metadata={"backend": self.name},
        )


__all__ = [
    "CommunicationCompleted",
    "CommunicationExecutor",
    "CommunicationFailed",
    "FrameRouter",
    "InterSatelliteFlowBackend",
    "IntraSatelliteSurrogateBackend",
    "Ns3PacketBackend",
    "PhysicalRoute",
    "PhysicalTransfer",
    "RouteHop",
]
