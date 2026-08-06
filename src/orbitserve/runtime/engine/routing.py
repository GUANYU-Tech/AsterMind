"""Stateless routing over Orbit geometry; communication contention is not modeled."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

from ...common import to_jsonable


C_KM_PER_S = 299_792.458
EARTH_RADIUS_KM = 6_378.137


class RouteUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class AccessCandidate:
    access_type: str
    ingress_id: str
    satellite_id: str | None
    start_s: float
    end_s: float
    uplink_bandwidth_bps: float
    downlink_bandwidth_bps: float
    propagation_delay_s: float
    anchor_unit_vector: tuple[float, float, float]

    @property
    def remaining_visibility_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass(frozen=True)
class RouteLeg:
    link_id: str
    kind: str
    source_node_id: str
    target_node_id: str
    payload_bytes: float
    bandwidth_bps: float
    propagation_delay_s: float
    valid_until_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class RoutePlan:
    access_type: str
    access_satellite_id: str | None
    compute_satellite_id: str
    inbound_legs: tuple[RouteLeg, ...]
    outbound_legs: tuple[RouteLeg, ...]
    arc_distance_rad: float
    route_satellite_ids: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def path_id(self) -> str:
        access = self.access_satellite_id or "communication_satellite"
        return f"{access}->{self.compute_satellite_id}"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def communication_access_candidate(profile: Any, request: dict[str, Any], horizon_s: float) -> AccessCandidate:
    if not getattr(profile, "configured", False):
        raise RouteUnavailable("communication_ingress_not_configured")
    lat = math.radians(float(request.get("lat_deg", 0.0)))
    lon = math.radians(float(request.get("lon_deg", 0.0)))
    anchor = (
        math.cos(lat) * math.cos(lon),
        math.cos(lat) * math.sin(lon),
        math.sin(lat),
    )
    return AccessCandidate(
        access_type="communication_satellite",
        ingress_id=str(request.get("ingress_id") or "communication_ingress"),
        satellite_id=None,
        start_s=float(request["arrival_time_s"]),
        end_s=float(horizon_s),
        uplink_bandwidth_bps=float(profile.uplink_bandwidth_bps),
        downlink_bandwidth_bps=float(profile.downlink_bandwidth_bps),
        propagation_delay_s=float(profile.altitude_km) / C_KM_PER_S,
        anchor_unit_vector=anchor,
    )


def compute_access_candidates(
    trace_bundle: Any,
    *,
    ingress_id: str,
    time_s: float,
    uplink_bandwidth_bps: float,
    downlink_bandwidth_bps: float,
) -> tuple[AccessCandidate, ...]:
    candidates: list[AccessCandidate] = []
    for window in trace_bundle.sgl_windows:
        if str(window.get("ingress_id")) != ingress_id or not _active(window, time_s):
            continue
        satellite_id = str(window["satellite_id"])
        vector = satellite_position_unit(trace_bundle, satellite_id, time_s)
        candidates.append(
            AccessCandidate(
                access_type="compute_satellite",
                ingress_id=ingress_id,
                satellite_id=satellite_id,
                start_s=float(window["start_s"]),
                end_s=float(window["end_s"]),
                uplink_bandwidth_bps=uplink_bandwidth_bps,
                downlink_bandwidth_bps=downlink_bandwidth_bps,
                propagation_delay_s=_window_propagation(window),
                anchor_unit_vector=vector,
            )
        )
    return tuple(sorted(candidates, key=lambda item: (item.propagation_delay_s, item.satellite_id or "")))


def build_route_plan(
    trace_bundle: Any,
    *,
    access: AccessCandidate,
    compute_satellite_id: str,
    time_s: float,
    input_bytes: float,
    output_bytes: float,
    isl_bandwidth_bps: float,
) -> RoutePlan | None:
    target_vector = satellite_position_unit(trace_bundle, compute_satellite_id, time_s)
    angle = central_angle(access.anchor_unit_vector, target_vector)
    if access.access_type == "communication_satellite":
        network_propagation = angle * EARTH_RADIUS_KM / C_KM_PER_S
        inbound = (
            RouteLeg(
                link_id=f"communication:{access.ingress_id}:{compute_satellite_id}",
                kind="communication_access",
                source_node_id=access.ingress_id,
                target_node_id=compute_satellite_id,
                payload_bytes=input_bytes,
                bandwidth_bps=access.uplink_bandwidth_bps,
                propagation_delay_s=access.propagation_delay_s + network_propagation,
            ),
        )
        outbound = (
            RouteLeg(
                link_id=f"communication:{compute_satellite_id}:{access.ingress_id}",
                kind="communication_access",
                source_node_id=compute_satellite_id,
                target_node_id=access.ingress_id,
                payload_bytes=output_bytes,
                bandwidth_bps=access.downlink_bandwidth_bps,
                propagation_delay_s=access.propagation_delay_s + network_propagation,
            ),
        )
        return RoutePlan(
            access_type=access.access_type,
            access_satellite_id=None,
            compute_satellite_id=compute_satellite_id,
            inbound_legs=inbound,
            outbound_legs=outbound,
            arc_distance_rad=angle,
            route_satellite_ids=(compute_satellite_id,),
        )

    source = str(access.satellite_id)
    route = shortest_active_isl_path(trace_bundle, source, compute_satellite_id, time_s)
    if route is None:
        return None
    satellite_ids, windows = route
    inbound: list[RouteLeg] = [
        RouteLeg(
            link_id=f"sgl:{access.ingress_id}:{source}",
            kind="sgl",
            source_node_id=access.ingress_id,
            target_node_id=source,
            payload_bytes=input_bytes,
            bandwidth_bps=access.uplink_bandwidth_bps,
            propagation_delay_s=access.propagation_delay_s,
            valid_until_s=access.end_s,
        )
    ]
    for window, first, second in zip(windows, satellite_ids, satellite_ids[1:]):
        inbound.append(
            RouteLeg(
                link_id=str(window.get("link_id") or f"isl:{first}:{second}"),
                kind="isl",
                source_node_id=first,
                target_node_id=second,
                payload_bytes=input_bytes,
                bandwidth_bps=isl_bandwidth_bps,
                propagation_delay_s=_window_propagation(window),
                valid_until_s=float(window["end_s"]),
            )
        )
    outbound: list[RouteLeg] = []
    for leg in reversed(inbound[1:]):
        outbound.append(
            RouteLeg(
                link_id=leg.link_id,
                kind=leg.kind,
                source_node_id=leg.target_node_id,
                target_node_id=leg.source_node_id,
                payload_bytes=output_bytes,
                bandwidth_bps=leg.bandwidth_bps,
                propagation_delay_s=leg.propagation_delay_s,
                valid_until_s=leg.valid_until_s,
            )
        )
    outbound.append(
        RouteLeg(
            link_id=f"sgl:{access.ingress_id}:{source}",
            kind="sgl",
            source_node_id=source,
            target_node_id=access.ingress_id,
            payload_bytes=output_bytes,
            bandwidth_bps=access.downlink_bandwidth_bps,
            propagation_delay_s=access.propagation_delay_s,
            valid_until_s=access.end_s,
        )
    )
    return RoutePlan(
        access_type=access.access_type,
        access_satellite_id=source,
        compute_satellite_id=compute_satellite_id,
        inbound_legs=tuple(inbound),
        outbound_legs=tuple(outbound),
        arc_distance_rad=angle,
        route_satellite_ids=tuple(satellite_ids),
    )


def execute_route(legs: tuple[RouteLeg, ...], start_s: float) -> tuple[float, tuple[dict[str, Any], ...]]:
    current = float(start_s)
    records: list[dict[str, Any]] = []
    for leg in legs:
        if leg.bandwidth_bps <= 0.0:
            raise RouteUnavailable("link_bandwidth_unavailable")
        leg_start_s = current
        transmission_s = max(0.0, leg.payload_bytes) * 8.0 / leg.bandwidth_bps
        finish = current + transmission_s
        if leg.valid_until_s is not None and finish > leg.valid_until_s + 1e-9:
            raise RouteUnavailable("link_window_closed_during_transmission")
        arrival = finish + leg.propagation_delay_s
        record = leg.to_dict()
        record.update(
            {
                "transmission_s": transmission_s,
                "start_s": leg_start_s,
                "arrival_s": arrival,
                "queue_wait_s": 0.0,
            }
        )
        records.append(record)
        current = arrival
    return current, tuple(records)


def shortest_active_isl_path(
    trace_bundle: Any,
    source: str,
    target: str,
    time_s: float,
) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]] | None:
    if source == target:
        return (source,), ()
    graph: dict[str, list[tuple[float, str, dict[str, Any]]]] = {}
    for window in trace_bundle.isl_windows:
        if not _active(window, time_s):
            continue
        first = str(window["endpoint_a"])
        second = str(window["endpoint_b"])
        weight = _window_distance(window)
        graph.setdefault(first, []).append((weight, second, dict(window)))
        graph.setdefault(second, []).append((weight, first, dict(window)))
    queue: list[tuple[float, str, tuple[str, ...], tuple[dict[str, Any], ...]]] = [(0.0, source, (source,), ())]
    best: dict[str, float] = {}
    while queue:
        cost, node, nodes, windows = heapq.heappop(queue)
        if node == target:
            return nodes, windows
        if cost >= best.get(node, math.inf):
            continue
        best[node] = cost
        for weight, neighbor, window in sorted(graph.get(node, ()), key=lambda item: (item[0], item[1])):
            heapq.heappush(queue, (cost + weight, neighbor, (*nodes, neighbor), (*windows, window)))
    return None


def build_isl_transfer_legs(
    trace_bundle: Any,
    *,
    source_satellite_id: str,
    target_satellite_id: str,
    time_s: float,
    payload_bytes: float,
    bandwidth_bps: float,
) -> tuple[RouteLeg, ...] | None:
    route = shortest_active_isl_path(
        trace_bundle, source_satellite_id, target_satellite_id, time_s
    )
    if route is None:
        return None
    nodes, windows = route
    return tuple(
        RouteLeg(
            link_id=str(window.get("link_id") or f"isl:{first}:{second}"),
            kind="kv_copy",
            source_node_id=first,
            target_node_id=second,
            payload_bytes=payload_bytes,
            bandwidth_bps=bandwidth_bps,
            propagation_delay_s=_window_propagation(window),
            valid_until_s=float(window["end_s"]),
        )
        for window, first, second in zip(windows, nodes, nodes[1:])
    )


def satellite_position_unit(trace_bundle: Any, satellite_id: str, time_s: float) -> tuple[float, float, float]:
    rows = [
        row for row in trace_bundle.ephemeris_samples
        if str(row.get("satellite_id")) == satellite_id
    ]
    if not rows:
        raise RouteUnavailable(f"ephemeris_missing:{satellite_id}")
    row = min(rows, key=lambda item: (abs(float(item["time_s"]) - time_s), float(item["time_s"])))
    vector = tuple(float(item) for item in row["position_itrs_km"])
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 0.0:
        raise RouteUnavailable(f"ephemeris_invalid:{satellite_id}")
    return tuple(item / norm for item in vector)


def central_angle(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(first, second))))
    return math.acos(dot)


def _active(window: dict[str, Any], time_s: float) -> bool:
    return float(window["start_s"]) <= time_s < float(window["end_s"])


def _window_propagation(window: dict[str, Any]) -> float:
    if window.get("propagation_delay_s") is not None:
        return float(window["propagation_delay_s"])
    low = float(window.get("propagation_delay_min_s", 0.0))
    high = float(window.get("propagation_delay_max_s", low))
    return (low + high) / 2.0


def _window_distance(window: dict[str, Any]) -> float:
    if window.get("distance_km") is not None:
        return float(window["distance_km"])
    low = float(window.get("distance_min_km", 0.0))
    high = float(window.get("distance_max_km", low))
    return (low + high) / 2.0


__all__ = [
    "AccessCandidate",
    "RouteLeg",
    "RoutePlan",
    "RouteUnavailable",
    "build_route_plan",
    "build_isl_transfer_legs",
    "central_angle",
    "communication_access_candidate",
    "compute_access_candidates",
    "execute_route",
    "satellite_position_unit",
    "shortest_active_isl_path",
]
