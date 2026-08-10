"""Temporal SGL/ISL routing helpers for preliminary runtime evaluation."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace
from typing import Any

from ...orbit import active_sgl_access_window, has_sgl_access_windows, sgl_window_link_sample
from ...common import to_jsonable


@dataclass(frozen=True)
class TraceQueryResult:
    available: bool
    reason: str | None = None
    sample_time_s: float | None = None
    stale: bool = False
    value: Any = None
    warnings: tuple[str, ...] = ()


class RouteSamplingError(ValueError):
    def __init__(
        self,
        reason: str,
        *,
        sample_time_s: float | None = None,
        warnings: tuple[str, ...] = (),
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.sample_time_s = sample_time_s
        self.warnings = warnings


@dataclass(frozen=True)
class AccessCandidate:
    ingress_id: str
    satellite_id: str
    window_id: str
    start_s: float
    end_s: float
    sgl_link_id: str
    propagation_delay_s: float
    bandwidth_bps: float
    sample_time_s: float | None = None
    stale: bool = False
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_window(
        cls,
        window: dict[str, Any],
        *,
        sample_time_s: float | None = None,
        stale: bool = False,
        warnings: tuple[str, ...] = (),
    ) -> "AccessCandidate":
        return cls(
            ingress_id=str(window["ingress_id"]),
            satellite_id=str(window["satellite_id"]),
            window_id=str(window.get("window_id", "")),
            start_s=float(window.get("start_s", 0.0)),
            end_s=float(window.get("end_s", 0.0)),
            sgl_link_id=str(window["sgl_link_id"]),
            propagation_delay_s=float(
                _first_present(window.get("propagation_delay_s"), window.get("propagation_delay_min_s"), 0.0)
            ),
            bandwidth_bps=float(window.get("bandwidth_bps", 0.0)),
            sample_time_s=sample_time_s,
            stale=stale,
            warnings=warnings,
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class RouteLeg:
    link_id: str
    kind: str
    endpoint_a: str
    endpoint_b: str
    payload_bytes: float
    source_node_id: str | None = None
    target_node_id: str | None = None
    queue_wait_s: float | None = None
    transmission_s: float | None = None
    propagation_delay_s: float | None = None
    bandwidth_bps: float | None = None
    setup_delay_s: float | None = None
    sample_time_s: float | None = None
    active_graph_time_s: float | None = None
    edge_available_external: bool | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class RoutePlan:
    access_satellite_id: str
    compute_satellite_id: str
    inbound_legs: tuple[RouteLeg, ...]
    outbound_legs: tuple[RouteLeg, ...]
    hop_count: int
    predicted_delay_s: float | None = None
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def path_id(self) -> str:
        return f"{self.access_satellite_id}->{self.compute_satellite_id}:{self.hop_count}"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def find_access_candidates(trace_bundle: Any, ingress_id: str, time_s: float) -> tuple[AccessCandidate, ...]:
    result = query_access_at(trace_bundle, ingress_id, time_s)
    if not result.available or result.stale:
        return ()
    candidates: list[AccessCandidate] = []
    for item in _access_items(result.value):
        candidate = item if isinstance(item, AccessCandidate) else AccessCandidate.from_window(
            _dict(item),
            sample_time_s=result.sample_time_s,
            stale=result.stale,
            warnings=result.warnings,
        )
        if (
            str(candidate.ingress_id) == str(ingress_id)
            and candidate.start_s <= time_s < candidate.end_s
            and _sgl_available(trace_bundle, candidate.sgl_link_id, time_s)
        ):
            candidates.append(candidate)
    return tuple(sorted(candidates, key=lambda item: (item.propagation_delay_s, item.satellite_id, item.window_id)))


def find_next_access_candidate(trace_bundle: Any, ingress_id: str, after_s: float) -> AccessCandidate | None:
    method = getattr(trace_bundle, "next_access_after", None)
    if callable(method):
        result = _as_query_result(method(ingress_id, after_s), unavailable_reason="no_next_access_window")
        if not result.available or result.stale:
            return None
        candidates = [
            item if isinstance(item, AccessCandidate) else AccessCandidate.from_window(
                _dict(item),
                sample_time_s=result.sample_time_s,
                stale=result.stale,
                warnings=result.warnings,
            )
            for item in _access_items(result.value)
        ]
    else:
        candidates = [
            AccessCandidate.from_window(window)
            for window in _access_windows(trace_bundle)
            if str(window.get("ingress_id")) == str(ingress_id)
            and float(window.get("end_s", 0.0)) > after_s
            and float(window.get("start_s", 0.0)) >= after_s
        ]
    candidates = [
        candidate
        for candidate in candidates
        if candidate.start_s >= after_s and _sgl_available(trace_bundle, candidate.sgl_link_id, candidate.start_s)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item.start_s, item.propagation_delay_s, item.satellite_id, item.window_id))


def build_ring_path(
    links: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    source_satellite_id: str,
    target_satellite_id: str,
    max_hop_count: int | None = None,
) -> tuple[str, ...] | None:
    if source_satellite_id == target_satellite_id:
        return ()
    return _build_ring_path_from_graph(
        _ring_graph_from_links(links),
        source_satellite_id,
        target_satellite_id,
        max_hop_count=max_hop_count,
    )


def _ring_graph_from_links(
    links: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> dict[str, tuple[tuple[float, str, str], ...]]:
    graph: dict[str, list[tuple[float, str, str]]] = {}
    for link in links:
        if _runtime_kind(link) != "isl":
            continue
        endpoint_a = str(link.get("endpoint_a"))
        endpoint_b = str(link.get("endpoint_b"))
        delay = float(_first_present(link.get("propagation_delay_s"), link.get("tau_prop_s"), 0.0))
        link_id = str(link.get("link_id"))
        graph.setdefault(endpoint_a, []).append((delay, endpoint_b, link_id))
        if bool(link.get("bidirectional", True)):
            graph.setdefault(endpoint_b, []).append((delay, endpoint_a, link_id))
    return {
        node: tuple(sorted(edges, key=lambda item: (item[0], item[1], item[2])))
        for node, edges in graph.items()
    }


def _build_ring_path_from_graph(
    graph: dict[str, tuple[tuple[float, str, str], ...]],
    source_satellite_id: str,
    target_satellite_id: str,
    max_hop_count: int | None = None,
) -> tuple[str, ...] | None:
    if source_satellite_id == target_satellite_id:
        return ()
    queue: list[tuple[float, int, str, tuple[str, ...]]] = [(0.0, 0, source_satellite_id, ())]
    seen: dict[str, tuple[float, int]] = {}
    while queue:
        cost, hops, node, path = heapq.heappop(queue)
        if node == target_satellite_id:
            return path
        if node in seen and seen[node] <= (cost, hops):
            continue
        seen[node] = (cost, hops)
        if max_hop_count is not None and hops >= max_hop_count:
            continue
        for delay, neighbor, link_id in graph.get(node, ()):
            heapq.heappush(queue, (cost + delay, hops + 1, neighbor, (*path, link_id)))
    return None


def enumerate_compute_candidates(
    trace_bundle: Any,
    access_candidates: tuple[AccessCandidate, ...],
    policy: str,
    max_hop_count: int | None,
    *,
    time_s: float | None = None,
    failures: list[str] | None = None,
    warnings: list[str] | None = None,
    target_satellite_ids: tuple[str, ...] | None = None,
) -> tuple[tuple[AccessCandidate, str, tuple[str, ...]], ...]:
    if max_hop_count is None:
        max_hop_count = 0 if policy in {"local_only", "nearest_visible"} else 2
    satellite_ids = tuple(str(item.get("satellite_id")) for item in getattr(trace_bundle, "satellite_catalog", ()) or ())
    target_ids = None if target_satellite_ids is None else {str(item) for item in target_satellite_ids}
    if target_ids is not None:
        satellite_ids = tuple(satellite_id for satellite_id in satellite_ids if satellite_id in target_ids)
    rows: list[tuple[AccessCandidate, str, tuple[str, ...]]] = []
    for access in access_candidates:
        if policy in {"local_only", "nearest_visible"} or max_hop_count == 0:
            if target_ids is not None and access.satellite_id not in target_ids:
                continue
            rows.append((access, access.satellite_id, ()))
            continue
        active = query_active_links_at(trace_bundle, time_s if time_s is not None else access.start_s)
        if not active.available or active.stale:
            reason = _stale_reason("active_graph", active.reason) if active.stale else active.reason or "active_graph_unavailable"
            if failures is not None:
                failures.append(reason)
            if warnings is not None:
                warnings.extend(active.warnings or (reason,))
            continue
        links = _active_links_from_result(trace_bundle, active.value)
        graph = _cached_ring_graph_for_active(trace_bundle, active, links)
        for satellite_id in satellite_ids:
            path = _build_ring_path_from_graph(
                graph,
                access.satellite_id,
                satellite_id,
                max_hop_count=max_hop_count,
            )
            if path is None:
                continue
            rows.append((access, satellite_id, path))
    return tuple(sorted(rows, key=lambda item: (item[0].satellite_id, item[1], item[2])))


def _cached_ring_graph_for_active(
    trace_bundle: Any,
    active: TraceQueryResult,
    links: tuple[dict[str, Any], ...],
) -> dict[str, tuple[tuple[float, str, str], ...]]:
    cache = getattr(trace_bundle, "__dict__", {}).get("_routing_ring_graph_cache")
    if not isinstance(cache, dict):
        cache = {}
        try:
            object.__setattr__(trace_bundle, "_routing_ring_graph_cache", cache)
        except Exception:
            pass
    value = _dict(active.value)
    snapshot = _dict(value.get("snapshot"))
    if snapshot.get("activation_policy") == "static_compat_links":
        key = ("static_compat_links", len(links))
    else:
        active_ids = tuple(str(item) for item in value.get("active_link_ids", ()) or ())
        key = ("active", active.sample_time_s, active_ids or len(links))
    graph = cache.get(key)
    if isinstance(graph, dict):
        return graph
    graph = _ring_graph_from_links(links)
    cache[key] = graph
    return graph


def make_route_plan(
    *,
    trace_bundle: Any,
    access_candidate: AccessCandidate,
    compute_satellite_id: str,
    isl_link_ids: tuple[str, ...],
    input_bytes: float,
    output_bytes: float,
    time_s: float | None = None,
) -> RoutePlan:
    links = _links_by_id(trace_bundle)
    inbound: list[RouteLeg] = []
    outbound: list[RouteLeg] = []
    active_graph_time_s = None
    if time_s is not None:
        active = query_active_links_at(trace_bundle, time_s)
        if active.available and not active.stale:
            active_graph_time_s = active.sample_time_s
    sgl = _require_link(links, access_candidate.sgl_link_id)
    inbound.append(
        _leg_from_link(
            trace_bundle,
            sgl,
            payload_bytes=input_bytes,
            time_s=time_s,
            active_graph_time_s=active_graph_time_s,
            source_node_id=str(access_candidate.ingress_id),
            target_node_id=str(access_candidate.satellite_id),
        )
    )
    current_node = str(access_candidate.satellite_id)
    for link_id in isl_link_ids:
        link = _require_link(links, link_id)
        next_node = _other_endpoint(link, current_node)
        inbound.append(
            _leg_from_link(
                trace_bundle,
                link,
                payload_bytes=input_bytes,
                time_s=time_s,
                active_graph_time_s=active_graph_time_s,
                source_node_id=current_node,
                target_node_id=next_node,
            )
        )
        current_node = next_node
    current_node = str(compute_satellite_id)
    for link_id in reversed(isl_link_ids):
        link = _require_link(links, link_id)
        next_node = _other_endpoint(link, current_node)
        outbound.append(
            _leg_from_link(
                trace_bundle,
                link,
                payload_bytes=output_bytes,
                time_s=time_s,
                active_graph_time_s=active_graph_time_s,
                source_node_id=current_node,
                target_node_id=next_node,
            )
        )
        current_node = next_node
    outbound.append(
        _leg_from_link(
            trace_bundle,
            sgl,
            payload_bytes=output_bytes,
            time_s=time_s,
            active_graph_time_s=active_graph_time_s,
            source_node_id=str(access_candidate.satellite_id),
            target_node_id=str(access_candidate.ingress_id),
        )
    )
    delay = sum(
        float(leg.propagation_delay_s or 0.0) + float(leg.setup_delay_s or 0.0)
        for leg in (*inbound, *outbound)
    )
    return RoutePlan(
        access_satellite_id=access_candidate.satellite_id,
        compute_satellite_id=compute_satellite_id,
        inbound_legs=tuple(inbound),
        outbound_legs=tuple(outbound),
        hop_count=len(isl_link_ids),
        predicted_delay_s=delay,
        warnings=tuple(dict.fromkeys((*access_candidate.warnings, *(warning for leg in (*inbound, *outbound) for warning in leg.warnings)))),
        provenance={
            "routing_model": "temporal_trace_route_plan",
            "decision_time_s": time_s,
            "active_graph_time_s": active_graph_time_s,
        },
    )


def sample_route_leg(trace_bundle: Any, leg: RouteLeg, time_s: float) -> RouteLeg:
    active_sample_time_s = leg.active_graph_time_s
    active_warnings: tuple[str, ...] = ()
    if leg.kind != "sgl":
        active = query_active_links_at(trace_bundle, time_s)
        if not active.available:
            raise RouteSamplingError(
                active.reason or "active_graph_unavailable",
                sample_time_s=active.sample_time_s,
                warnings=active.warnings,
            )
        if active.stale:
            raise RouteSamplingError(
                _stale_reason("active_graph", active.reason),
                sample_time_s=active.sample_time_s,
                warnings=active.warnings,
            )
        active_ids = _active_link_id_set_from_result(active.value)
        if active_ids and leg.link_id not in active_ids:
            raise RouteSamplingError(
                "link_inactive_at_reservation",
                sample_time_s=active.sample_time_s,
                warnings=active.warnings,
            )
        active_sample_time_s = active.sample_time_s
        active_warnings = active.warnings
    sample = query_link_sample_at(trace_bundle, leg.link_id, time_s)
    if not sample.available:
        raise RouteSamplingError(
            sample.reason or "link_sample_unavailable",
            sample_time_s=sample.sample_time_s,
            warnings=sample.warnings,
        )
    if sample.stale:
        raise RouteSamplingError(
            _stale_reason("link", sample.reason),
            sample_time_s=sample.sample_time_s,
            warnings=sample.warnings,
        )
    link = _links_by_id(trace_bundle).get(leg.link_id, {})
    value = _dict(sample.value)
    bandwidth_bps = _sample_bandwidth_bps(value, link)
    propagation_delay_s = _sample_propagation_delay_s(value, link)
    setup_delay_s = _sample_setup_delay_s(value, link)
    edge_available = bool(_first_present(value.get("edge_available_external"), link.get("edge_available_external"), True))
    if not edge_available:
        raise RouteSamplingError(
            str(value.get("unavailable_reason") or "link_unavailable_external"),
            sample_time_s=sample.sample_time_s,
            warnings=sample.warnings,
        )
    return replace(
        leg,
        bandwidth_bps=bandwidth_bps,
        propagation_delay_s=propagation_delay_s,
        setup_delay_s=setup_delay_s,
        sample_time_s=sample.sample_time_s,
        active_graph_time_s=active_sample_time_s,
        edge_available_external=edge_available,
        warnings=tuple(dict.fromkeys((*leg.warnings, *active_warnings, *sample.warnings))),
    )


def query_access_at(trace_bundle: Any, ingress_id: str, time_s: float) -> TraceQueryResult:
    method = getattr(trace_bundle, "access_at", None)
    if callable(method):
        return _as_query_result(method(ingress_id, time_s), unavailable_reason="no_access_window")
    matches = [
        window
        for window in _access_windows(trace_bundle)
        if str(window.get("ingress_id")) == str(ingress_id)
        and float(window.get("start_s", 0.0)) <= time_s < float(window.get("end_s", 0.0))
    ]
    if not matches:
        return TraceQueryResult(available=False, reason="no_access_window", value=())
    return TraceQueryResult(available=True, value=tuple(matches))


def query_active_links_at(trace_bundle: Any, time_s: float) -> TraceQueryResult:
    method = getattr(trace_bundle, "active_links_at", None)
    if callable(method):
        result = _as_query_result(method(time_s), unavailable_reason="active_graph_unavailable")
        return _runtime_terminal_selection(trace_bundle, result)
    snapshots = _active_graph_snapshots(trace_bundle)
    if snapshots:
        selected = _sample_row_at(snapshots, time_s, kind="active_graph", trace_bundle=trace_bundle)
        if not selected.available:
            return selected
        active_ids = tuple(str(item) for item in _dict(selected.value).get("active_link_ids", ()) or ())
        result = TraceQueryResult(
            available=True,
            sample_time_s=selected.sample_time_s,
            stale=selected.stale,
            value={"active_link_ids": active_ids, "links": tuple(_links_by_id(trace_bundle)[item] for item in active_ids if item in _links_by_id(trace_bundle))},
            warnings=selected.warnings,
        )
        return _runtime_terminal_selection(trace_bundle, result)
    if _has_dynamic_link_samples(trace_bundle):
        return TraceQueryResult(available=False, reason="active_graph_unavailable")
    return TraceQueryResult(available=True, value=tuple(_links_by_id(trace_bundle).values()))


def _runtime_terminal_selection(trace_bundle: Any, result: TraceQueryResult) -> TraceQueryResult:
    """Select visible ISL candidates using per-satellite static terminal caps."""

    if not result.available:
        return result
    links_by_id = _links_by_id(trace_bundle)
    value = _dict(result.value)
    raw_ids = value.get("active_link_ids")
    if raw_ids is None:
        raw_ids = [item.get("link_id") for item in value.get("links", ()) if isinstance(item, dict)]
    candidates = [links_by_id[str(item)] for item in raw_ids or () if str(item) in links_by_id]
    candidates.sort(key=lambda link: (int(link.get("activation_priority", 100)), str(link.get("link_id"))))
    selected: list[dict[str, Any]] = []
    used: dict[str, int] = {}
    blocked = 0
    for link in candidates:
        if str(link.get("kind")) == "sgl":
            selected.append(link)
            continue
        cap = _optional_int(
            _first_present(
                link.get("terminal_pool_capacity_per_satellite"),
                link.get("terminal_count_per_satellite"),
            )
        )
        if cap is None:
            # A standalone runtime trace may already contain a materialized active graph.
            selected.append(link)
            continue
        if cap <= 0:
            blocked += 1
            continue
        endpoint_a = str(link.get("endpoint_a"))
        endpoint_b = str(link.get("endpoint_b"))
        if used.get(endpoint_a, 0) >= cap or used.get(endpoint_b, 0) >= cap:
            blocked += 1
            continue
        used[endpoint_a] = used.get(endpoint_a, 0) + 1
        used[endpoint_b] = used.get(endpoint_b, 0) + 1
        selected.append(link)
    selected_ids = tuple(str(link["link_id"]) for link in selected)
    warnings = result.warnings
    if blocked:
        warnings = tuple(dict.fromkeys((*warnings, f"runtime terminal selection blocked {blocked} visible ISL candidates")))
    return TraceQueryResult(
        available=True,
        sample_time_s=result.sample_time_s,
        stale=result.stale,
        value={
            "active_link_ids": selected_ids,
            "links": tuple(selected),
            "visible_candidate_count": len(candidates),
            "terminal_blocked_count": blocked,
            "selection_policy": "runtime_per_satellite_terminal_cap",
        },
        warnings=warnings,
    )


def query_link_sample_at(trace_bundle: Any, link_id: str, time_s: float) -> TraceQueryResult:
    method = getattr(trace_bundle, "link_sample_at", None)
    if callable(method):
        return _as_query_result(method(link_id, time_s), unavailable_reason="link_sample_unavailable")
    sgl_window_sample = sgl_window_link_sample(
        _access_windows(trace_bundle),
        link_id,
        time_s,
        link=_links_by_id(trace_bundle).get(str(link_id)),
    )
    if sgl_window_sample is not None:
        if not bool(sgl_window_sample.get("edge_available_external", False)):
            return TraceQueryResult(
                available=False,
                reason=str(sgl_window_sample.get("unavailable_reason") or "invalid_sgl_capacity"),
                sample_time_s=time_s,
                value=sgl_window_sample,
            )
        return TraceQueryResult(available=True, sample_time_s=time_s, value=sgl_window_sample)
    if has_sgl_access_windows(_access_windows(trace_bundle), link_id):
        return TraceQueryResult(available=False, reason="no_sgl_access_window", sample_time_s=time_s)
    samples = [row for row in _link_samples(trace_bundle) if str(row.get("link_id")) == str(link_id)]
    if samples:
        selected = _sample_row_at(samples, time_s, kind="link", trace_bundle=trace_bundle)
        if not selected.available:
            return selected
        value = _dict(selected.value)
        if not bool(value.get("edge_available_external", False)):
            return TraceQueryResult(
                available=False,
                reason=str(value.get("unavailable_reason") or "link_unavailable_external"),
                sample_time_s=selected.sample_time_s,
                stale=selected.stale,
                value=value,
                warnings=selected.warnings,
            )
        return selected
    link = _links_by_id(trace_bundle).get(str(link_id))
    if link is None:
        return TraceQueryResult(available=False, reason="link_missing")
    if _has_dynamic_link_samples(trace_bundle):
        return TraceQueryResult(available=False, reason="link_sample_unavailable")
    return TraceQueryResult(available=True, value=link)


def _require_link(links: dict[str, dict[str, Any]], link_id: str) -> dict[str, Any]:
    if link_id not in links:
        raise ValueError(f"missing route link: {link_id}")
    return links[link_id]


def _leg_from_link(
    trace_bundle: Any,
    link: dict[str, Any],
    *,
    payload_bytes: float,
    time_s: float | None,
    active_graph_time_s: float | None,
    source_node_id: str,
    target_node_id: str,
) -> RouteLeg:
    leg = RouteLeg(
        link_id=str(link["link_id"]),
        kind=_runtime_kind(link),
        endpoint_a=str(link["endpoint_a"]),
        endpoint_b=str(link["endpoint_b"]),
        payload_bytes=payload_bytes,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        propagation_delay_s=float(_first_present(link.get("propagation_delay_s"), link.get("tau_prop_s"), 0.0)),
        bandwidth_bps=float(_first_present(link.get("bandwidth_bps"), link.get("bandwidth_bps_nominal"), link.get("B_raw_bps"), 0.0)),
        setup_delay_s=float(_first_present(link.get("setup_delay_s"), link.get("tau_setup_s"), 0.0)),
        active_graph_time_s=active_graph_time_s,
        edge_available_external=bool(link.get("edge_available_external", True)),
    )
    if time_s is None:
        return leg
    return sample_route_leg(trace_bundle, leg, time_s)


def _other_endpoint(link: dict[str, Any], current_node: str) -> str:
    endpoint_a = str(link.get("endpoint_a"))
    endpoint_b = str(link.get("endpoint_b"))
    if current_node == endpoint_a:
        return endpoint_b
    if current_node == endpoint_b:
        return endpoint_a
    raise ValueError(
        f"route link {link.get('link_id')} is not incident to current node {current_node}"
    )


def _sgl_available(trace_bundle: Any, link_id: str, time_s: float) -> bool:
    windows = _access_windows(trace_bundle)
    if active_sgl_access_window(windows, link_id, time_s) is not None:
        sample = sgl_window_link_sample(
            windows,
            link_id,
            time_s,
            link=_links_by_id(trace_bundle).get(str(link_id)),
        )
        return bool(sample and sample.get("edge_available_external", False))
    if has_sgl_access_windows(windows, link_id):
        return False
    if not _has_link_sample_query(trace_bundle):
        return True
    result = query_link_sample_at(trace_bundle, link_id, time_s)
    return result.available and not result.stale


def _access_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, AccessCandidate):
        return (value,)
    if isinstance(value, dict):
        for key in ("access_windows", "candidates", "windows"):
            if key in value:
                return tuple(value.get(key) or ())
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return ()


def _active_links_from_result(trace_bundle: Any, value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        links = value.get("links")
        if links is not None:
            return tuple(_dict(item) for item in links)
        ids = tuple(str(item) for item in value.get("active_link_ids", ()) or ())
        by_id = _links_by_id(trace_bundle)
        return tuple(by_id[item] for item in ids if item in by_id)
    if isinstance(value, (tuple, list)):
        by_id = _links_by_id(trace_bundle)
        rows: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                if item in by_id:
                    rows.append(by_id[item])
            else:
                rows.append(_dict(item))
        return tuple(rows)
    return ()


def _active_link_id_set_from_result(value: Any) -> frozenset[str]:
    if isinstance(value, dict):
        cached_ids = value.get("active_link_id_set")
        if isinstance(cached_ids, frozenset):
            return cached_ids
        if isinstance(cached_ids, set):
            return frozenset(str(item) for item in cached_ids)
        if value.get("active_link_ids") is not None:
            raw_ids = value.get("active_link_ids") or ()
            return frozenset(str(item) for item in raw_ids)
        if value.get("links") is not None:
            return frozenset(str(_dict(item).get("link_id")) for item in value.get("links") or ())
    if isinstance(value, (tuple, list)):
        ids = []
        for item in value:
            ids.append(str(item if isinstance(item, str) else _dict(item).get("link_id")))
        return frozenset(ids)
    return frozenset()


def _as_query_result(value: Any, *, unavailable_reason: str) -> TraceQueryResult:
    if isinstance(value, TraceQueryResult):
        return value
    if isinstance(value, dict) and "available" in value:
        return TraceQueryResult(
            available=bool(value.get("available")),
            reason=_optional_str(value.get("reason")),
            sample_time_s=_optional_float(value.get("sample_time_s")),
            stale=bool(value.get("stale", False)),
            value=value.get("value"),
            warnings=_string_tuple(value.get("warnings")),
        )
    if hasattr(value, "available"):
        return TraceQueryResult(
            available=bool(getattr(value, "available")),
            reason=_optional_str(getattr(value, "reason", None)),
            sample_time_s=_optional_float(getattr(value, "sample_time_s", None)),
            stale=bool(getattr(value, "stale", False)),
            value=getattr(value, "value", None),
            warnings=_string_tuple(getattr(value, "warnings", ())),
        )
    if value is None:
        return TraceQueryResult(available=False, reason=unavailable_reason)
    return TraceQueryResult(available=True, value=value)


def _sample_row_at(rows: list[dict[str, Any]], time_s: float, *, kind: str, trace_bundle: Any) -> TraceQueryResult:
    ordered = sorted(rows, key=lambda item: float(item.get("time_s", 0.0)))
    selected: dict[str, Any] | None = None
    for row in ordered:
        row_time = float(row.get("time_s", 0.0))
        if row_time <= time_s:
            selected = row
        else:
            break
    if selected is None:
        return TraceQueryResult(available=False, reason=f"{kind}_sample_unavailable")
    sample_time_s = float(selected.get("time_s", 0.0))
    step_s = _trace_step_s(trace_bundle, ordered)
    stale = False
    warnings: tuple[str, ...] = ()
    if step_s is not None and time_s - sample_time_s > step_s + _boundary_tolerance_s(trace_bundle):
        stale = True
        warnings = (f"stale_trace_sample:{kind}",)
    return TraceQueryResult(
        available=not stale,
        reason=f"stale_trace_sample:{kind}" if stale else None,
        sample_time_s=sample_time_s,
        stale=stale,
        value=selected,
        warnings=warnings,
    )


def _access_windows(trace_bundle: Any) -> tuple[dict[str, Any], ...]:
    windows = getattr(trace_bundle, "access_windows", None)
    if windows is None:
        windows = _dict(getattr(trace_bundle, "access_trace", None)).get("access_windows")
    return tuple(_dict(item) for item in windows or ())


def _active_graph_snapshots(trace_bundle: Any) -> list[dict[str, Any]]:
    snapshots = getattr(trace_bundle, "active_graph_snapshots", None)
    if snapshots is None:
        snapshots = _dict(getattr(trace_bundle, "link_trace", None)).get("active_graph_snapshots")
    return [_dict(item) for item in snapshots or ()]


def _link_samples(trace_bundle: Any) -> list[dict[str, Any]]:
    samples = getattr(trace_bundle, "link_samples", None)
    if samples is None:
        samples = _dict(getattr(trace_bundle, "link_trace", None)).get("link_samples")
    return [_dict(item) for item in samples or ()]


def _links_by_id(trace_bundle: Any) -> dict[str, dict[str, Any]]:
    cached = getattr(trace_bundle, "__dict__", {}).get("_links_by_id_cache")
    if isinstance(cached, dict):
        return cached
    rows: list[dict[str, Any]] = []
    for attr in ("physical_links", "links"):
        rows.extend(_dict(item) for item in (getattr(trace_bundle, attr, None) or ()))
    link_trace = _dict(getattr(trace_bundle, "link_trace", None))
    rows.extend(_dict(item) for item in (link_trace.get("physical_links") or ()))
    rows.extend(_dict(item) for item in (link_trace.get("links") or ()))
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        link_id = _optional_str(row.get("link_id"))
        if link_id is not None:
            by_id[link_id] = row
    try:
        object.__setattr__(trace_bundle, "_links_by_id_cache", by_id)
    except Exception:
        pass
    return by_id


def _has_link_sample_query(trace_bundle: Any) -> bool:
    return callable(getattr(trace_bundle, "link_sample_at", None)) or bool(_link_samples(trace_bundle))


def _has_dynamic_link_samples(trace_bundle: Any) -> bool:
    return bool(_active_graph_snapshots(trace_bundle) or _link_samples(trace_bundle))


def _trace_step_s(trace_bundle: Any, rows: list[dict[str, Any]] | None = None) -> float | None:
    attr_step_s = _optional_float(getattr(trace_bundle, "step_s", None))
    if attr_step_s is not None and attr_step_s > 0:
        return attr_step_s
    for source in (
        getattr(trace_bundle, "time", None),
        getattr(trace_bundle, "trace_time", None),
        _dict(getattr(trace_bundle, "trace_manifest", None)).get("time"),
        _dict(getattr(trace_bundle, "manifest", None)).get("time"),
    ):
        value = _optional_float(_dict(source).get("step_s") if isinstance(source, dict) else getattr(source, "step_s", None))
        if value and value > 0:
            return value
    if rows:
        times = sorted({float(row.get("time_s", 0.0)) for row in rows})
        deltas = [b - a for a, b in zip(times, times[1:]) if b > a]
        if deltas:
            return min(deltas)
    return None


def _boundary_tolerance_s(trace_bundle: Any) -> float:
    attr_tolerance_s = _optional_float(getattr(trace_bundle, "boundary_tolerance_s", None))
    if attr_tolerance_s is not None:
        return attr_tolerance_s
    for source in (
        getattr(trace_bundle, "time", None),
        getattr(trace_bundle, "trace_time", None),
        _dict(getattr(trace_bundle, "trace_manifest", None)).get("time"),
        _dict(getattr(trace_bundle, "manifest", None)).get("time"),
    ):
        value = _optional_float(_dict(source).get("boundary_tolerance_s") if isinstance(source, dict) else getattr(source, "boundary_tolerance_s", None))
        if value is not None:
            return value
    return 1e-9


def _sample_bandwidth_bps(sample: dict[str, Any], link: dict[str, Any]) -> float:
    return float(_first_present(sample.get("B_raw_bps"), sample.get("bandwidth_bps"), link.get("bandwidth_bps_nominal"), link.get("bandwidth_bps"), 0.0))


def _sample_propagation_delay_s(sample: dict[str, Any], link: dict[str, Any]) -> float:
    return float(_first_present(sample.get("tau_prop_s"), sample.get("propagation_delay_s"), link.get("propagation_delay_s"), 0.0))


def _sample_setup_delay_s(sample: dict[str, Any], link: dict[str, Any]) -> float:
    return float(_first_present(sample.get("tau_setup_s"), sample.get("setup_delay_s"), link.get("setup_delay_s"), 0.0))


def _runtime_kind(link: dict[str, Any]) -> str:
    kind = str(_first_present(link.get("runtime_kind"), link.get("kind"), ""))
    return "isl" if kind.startswith("isl") else kind


def _stale_reason(kind: str, reason: str | None) -> str:
    return reason if reason and reason.startswith("stale_trace_sample") else f"stale_trace_sample:{kind}"


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)






__all__ = [
    "AccessCandidate",
    "RouteLeg",
    "RoutePlan",
    "RouteSamplingError",
    "TraceQueryResult",
    "build_ring_path",
    "enumerate_compute_candidates",
    "find_access_candidates",
    "find_next_access_candidate",
    "make_route_plan",
    "query_access_at",
    "query_active_links_at",
    "query_link_sample_at",
    "sample_route_leg",
]
