"""Deterministic event simulation for one assembled Runtime input."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from .contracts import RuntimeEngineResult
from .decision import SchedulerCandidate, build_candidate_selector
from .eligibility import build_compute_eligibility_set
from .routing import (
    AccessCandidate,
    RoutePlan,
    RouteUnavailable,
    build_isl_transfer_legs,
    build_route_plan,
    communication_access_candidate,
    compute_access_candidates,
    execute_route,
)
from ..metrics.records import RuntimeRequestRecord
from ..state.energy import (
    EnergyWindowError,
    advance_satellite_energy,
    apply_request_energy,
    energy_window_at,
)
from ..state.kv_cache import commit_session_kv, local_kv_reuse_tokens
from ..state.runtime_state import RuntimeState


def run_runtime_engine(simulation_spec: Any) -> RuntimeEngineResult:
    trace_bundle = simulation_spec.orbit_trace_bundle
    if trace_bundle is None:
        return _all_uncompleted(simulation_spec, "orbit_trace_bundle_missing")
    serving = simulation_spec.serving_profile
    if serving is None or not serving.has_compute_capacity:
        return _all_uncompleted(simulation_spec, "serving_capacity_unavailable")
    try:
        energy_trace, initial_soc = _build_energy_trace(trace_bundle, simulation_spec.energy_profile)
        sgl_bandwidth_bps = _architecture_bandwidth(
            simulation_spec.constellation,
            ("sgl_bandwidth_bps", "sgl_total_bandwidth_bps", "sgl_bandwidth_gbps"),
            required=simulation_spec.strategy.ingress_selection != "communication_satellite",
        )
        isl_bandwidth_bps = _architecture_bandwidth(
            simulation_spec.constellation,
            ("isl_bandwidth_bps", "isl_total_bandwidth_bps", "isl_bandwidth_gbps"),
            required=False,
        )
    except ValueError as exc:
        return _all_uncompleted(simulation_spec, str(exc))

    state = RuntimeState()
    for row in trace_bundle.satellite_catalog:
        state.satellite_for(str(row["satellite_id"]), initial_soc=initial_soc)
    eligibility = build_compute_eligibility_set(
        satellite_catalog=tuple(trace_bundle.satellite_catalog)
    )
    if eligibility.empty:
        return _all_uncompleted(simulation_spec, "eligible_compute_set_empty")

    records: list[RuntimeRequestRecord] = []
    decisions: list[dict[str, Any]] = []
    for raw in sorted(
        simulation_spec.requests,
        key=lambda row: (float(row.get("arrival_time_s", 0.0)), str(row.get("request_id", ""))),
    ):
        request = normalize_runtime_request(raw)
        record, decision = _run_request(
            state=state,
            request=request,
            simulation_spec=simulation_spec,
            trace_bundle=trace_bundle,
            energy_trace=energy_trace,
            eligible_satellite_ids=eligibility.eligible_compute_satellite_ids,
            sgl_bandwidth_bps=sgl_bandwidth_bps,
            isl_bandwidth_bps=isl_bandwidth_bps,
        )
        records.append(record)
        decisions.append(decision)
    return RuntimeEngineResult(
        records=tuple(records),
        decision_traces=tuple(decisions),
        state_summary=state.to_dict(),
        provenance={
            "runtime_engine": "deterministic_event_runtime",
            "communication_queue_model": "not_modeled",
            "compute_queue_model": "per_satellite_prefill_decode_fifo",
        },
    )


def normalize_runtime_request(raw_request: dict[str, Any]) -> dict[str, Any]:
    if "prompt_tokens" not in raw_request or "output_tokens" not in raw_request:
        raise ValueError("canonical request requires prompt_tokens and output_tokens")
    request_id = str(raw_request.get("request_id") or "")
    if not request_id:
        raise ValueError("canonical request requires request_id")
    prompt_tokens = int(raw_request["prompt_tokens"])
    output_tokens = int(raw_request["output_tokens"])
    if prompt_tokens < 0 or output_tokens <= 0:
        raise ValueError("prompt_tokens must be non-negative and output_tokens must be positive")
    return {
        **dict(raw_request),
        "request_id": request_id,
        "arrival_time_s": float(raw_request.get("arrival_time_s", 0.0)),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "input_bytes": max(0.0, float(raw_request.get("input_bytes", 0.0))),
        "output_bytes": max(0.0, float(raw_request.get("output_bytes", 0.0))),
    }


def _run_request(
    *,
    state: RuntimeState,
    request: dict[str, Any],
    simulation_spec: Any,
    trace_bundle: Any,
    energy_trace: Any,
    eligible_satellite_ids: tuple[str, ...],
    sgl_bandwidth_bps: float,
    isl_bandwidth_bps: float,
) -> tuple[RuntimeRequestRecord, dict[str, Any]]:
    arrival_s = float(request["arrival_time_s"])
    horizon_s = float(trace_bundle.time["end_s"])
    if arrival_s < float(trace_bundle.time["start_s"]) or arrival_s >= horizon_s:
        return _uncompleted(request, "request_time_outside_trace_horizon"), _decision(request, None, ())
    try:
        access = _select_access(
            state=state,
            request=request,
            simulation_spec=simulation_spec,
            trace_bundle=trace_bundle,
            energy_trace=energy_trace,
            eligible_satellite_ids=eligible_satellite_ids,
            sgl_bandwidth_bps=sgl_bandwidth_bps,
            horizon_s=horizon_s,
        )
    except (RouteUnavailable, EnergyWindowError) as exc:
        return _uncompleted(request, str(exc)), _decision(request, None, ())

    candidates: list[SchedulerCandidate] = []
    candidate_payloads: dict[str, dict[str, Any]] = {}
    for satellite_id in eligible_satellite_ids:
        try:
            payload = _preview_candidate(
                state=state,
                request=request,
                strategy=simulation_spec.strategy,
                trace_bundle=trace_bundle,
                energy_trace=energy_trace,
                serving=simulation_spec.serving_profile,
                access=access,
                compute_satellite_id=satellite_id,
                isl_bandwidth_bps=isl_bandwidth_bps,
            )
        except (RouteUnavailable, EnergyWindowError, ValueError):
            continue
        candidate = payload["candidate"]
        candidates.append(candidate)
        candidate_payloads[candidate.candidate_id] = payload
    selector = build_candidate_selector(simulation_spec.strategy)
    selected = selector.select_action(
        {"candidates": tuple(candidate.to_dict() for candidate in candidates)},
        request,
    )
    if not selected.evaluated or selected.selected_candidate_id not in candidate_payloads:
        return _uncompleted(request, "no_feasible_compute_candidate"), _decision(request, None, candidates)
    payload = candidate_payloads[selected.selected_candidate_id]
    try:
        record = _commit_candidate(
            state=state,
            request=request,
            strategy=simulation_spec.strategy,
            trace_bundle=trace_bundle,
            energy_trace=energy_trace,
            serving=simulation_spec.serving_profile,
            payload=payload,
            selection=selected,
        )
    except (RouteUnavailable, EnergyWindowError, ValueError) as exc:
        return _uncompleted(request, str(exc)), _decision(request, selected, candidates)
    return record, _decision(request, selected, candidates)


def _select_access(
    *,
    state: RuntimeState,
    request: dict[str, Any],
    simulation_spec: Any,
    trace_bundle: Any,
    energy_trace: Any,
    eligible_satellite_ids: tuple[str, ...],
    sgl_bandwidth_bps: float,
    horizon_s: float,
) -> AccessCandidate:
    mode = simulation_spec.strategy.ingress_selection
    if mode == "communication_satellite":
        return communication_access_candidate(
            simulation_spec.communication_ingress, request, horizon_s
        )
    ingress_id = str(request.get("ingress_id") or "")
    if not ingress_id:
        raise RouteUnavailable("request_ingress_id_missing")
    rows = tuple(
        candidate
        for candidate in compute_access_candidates(
            trace_bundle,
            ingress_id=ingress_id,
            time_s=float(request["arrival_time_s"]),
            uplink_bandwidth_bps=sgl_bandwidth_bps,
            downlink_bandwidth_bps=sgl_bandwidth_bps,
        )
        if candidate.satellite_id in set(eligible_satellite_ids)
    )
    if not rows:
        raise RouteUnavailable("no_visible_compute_ingress")
    if mode == "nearest_compute_satellite":
        return min(rows, key=lambda item: (item.propagation_delay_s, item.satellite_id or ""))
    if mode == "longest_visible_compute_satellite":
        return min(rows, key=lambda item: (-item.end_s, item.satellite_id or ""))
    if mode == "highest_energy_compute_satellite":
        scored: list[tuple[float, str, AccessCandidate]] = []
        for candidate in rows:
            satellite = state.satellite_for(str(candidate.satellite_id))
            advance_satellite_energy(satellite, energy_trace, float(request["arrival_time_s"]))
            scored.append((-satellite.soc, str(candidate.satellite_id), candidate))
        return min(scored)[2]
    raise RouteUnavailable("unsupported_ingress_selection")


def _preview_candidate(
    *,
    state: RuntimeState,
    request: dict[str, Any],
    strategy: Any,
    trace_bundle: Any,
    energy_trace: Any,
    serving: Any,
    access: AccessCandidate,
    compute_satellite_id: str,
    isl_bandwidth_bps: float,
) -> dict[str, Any]:
    route = build_route_plan(
        trace_bundle,
        access=access,
        compute_satellite_id=compute_satellite_id,
        time_s=float(request["arrival_time_s"]),
        input_bytes=float(request["input_bytes"]),
        output_bytes=float(request["output_bytes"]),
        isl_bandwidth_bps=isl_bandwidth_bps,
    )
    if route is None:
        raise RouteUnavailable("no_route_to_compute_satellite")
    inbound_done_s, inbound_path = execute_route(route.inbound_legs, float(request["arrival_time_s"]))
    session_id = request.get("session_id")
    required_context = int(request["prompt_tokens"])
    local_hit_tokens = local_kv_reuse_tokens(
        state,
        satellite_id=compute_satellite_id,
        session_id=str(session_id) if session_id else None,
        required_context_tokens=required_context,
    )
    copy_plan = _preview_kv_copy(
        state=state,
        session_id=str(session_id) if session_id else None,
        target_satellite_id=compute_satellite_id,
        required_context_tokens=required_context,
        local_hit_tokens=local_hit_tokens,
        copy_ratio=float(strategy.kv_copy_ratio),
        kv_bytes_per_token=float(serving.kv_cache_bytes_per_token or 0.0),
        trace_bundle=trace_bundle,
        start_s=inbound_done_s,
        isl_bandwidth_bps=isl_bandwidth_bps,
    )
    copy_path = copy_plan["path"]
    copied_from = copy_plan["source_satellite_id"]
    copied_tokens = copy_plan["copied_tokens"]
    copy_done_s = copy_plan["done_s"]
    effective_reuse_tokens = min(
        required_context,
        local_hit_tokens + copied_tokens,
    )
    recompute_tokens = max(0, required_context - effective_reuse_tokens)
    satellite = state.satellite_for(compute_satellite_id)
    compute_ready_s = copy_done_s
    raw_window = energy_window_at(energy_trace, compute_satellite_id, max(compute_ready_s, satellite.compute_queue.prefill_available_s))
    dvfs_scale = (
        float(strategy.shadow_dvfs_scale)
        if strategy.shadow_dvfs_enabled and not bool(raw_window.get("sunlit", True))
        else 1.0
    )
    prefill_service_s = _service_time(recompute_tokens, serving.prefill_tokens_per_s) / dvfs_scale
    decode_service_s = _service_time(request["output_tokens"], serving.decode_tokens_per_s) / dvfs_scale
    prefill_start_s, prefill_done_s, prefill_wait_s = satellite.compute_queue.preview_prefill(compute_ready_s, prefill_service_s)
    decode_start_s, decode_done_s, decode_wait_s = satellite.compute_queue.preview_decode(prefill_done_s, decode_service_s)
    first_token_service_s = _service_time(1, serving.decode_tokens_per_s) / dvfs_scale
    first_token_bytes = _first_token_bytes(request)
    first_token_legs = tuple(replace(leg, payload_bytes=first_token_bytes) for leg in route.outbound_legs)
    first_done_s, first_path = execute_route(first_token_legs, decode_start_s + first_token_service_s)
    remaining_output_bytes = max(0.0, float(request["output_bytes"]) - first_token_bytes)
    remaining_legs = tuple(replace(leg, payload_bytes=remaining_output_bytes) for leg in route.outbound_legs)
    remaining_done_s, output_path = execute_route(remaining_legs, decode_done_s)
    output_done_s = max(first_done_s, remaining_done_s)

    preview = replace(satellite, compute_queue=replace(satellite.compute_queue))
    advance_satellite_energy(preview, energy_trace, prefill_start_s)
    energy_window = _energy_window_with_dvfs(
        energy_window_at(energy_trace, compute_satellite_id, prefill_start_s),
        dvfs_scale,
    )
    physical_paths = (inbound_path, first_path, output_path, copy_path)
    link_bytes_by_satellite = _link_bytes_by_satellite(
        physical_paths,
        {
            str(row["satellite_id"])
            for row in trace_bundle.satellite_catalog
        },
    )
    compute_link_bytes = link_bytes_by_satellite.get(compute_satellite_id, 0.0)
    link_update = apply_request_energy(
        preview,
        prompt_tokens=0,
        output_tokens=0,
        link_bytes=compute_link_bytes,
        energy_window=energy_window,
        event_time_s=prefill_start_s,
        metadata={"request_id": request["request_id"], "preview": True, "energy_stage": "communication"},
    )
    compute_update = apply_request_energy(
        preview,
        prompt_tokens=recompute_tokens,
        output_tokens=request["output_tokens"],
        link_bytes=0.0,
        energy_window=energy_window,
        event_time_s=prefill_start_s,
        metadata={"request_id": request["request_id"], "preview": True, "energy_stage": "compute"},
    )
    advance_satellite_energy(preview, energy_trace, decode_done_s)
    energy_score = max(0.0, min(1.0, float(preview.soc)))
    service_time_s = prefill_service_s + decode_service_s
    queue_wait_s = prefill_wait_s + decode_wait_s
    compute_load_score = 1.0 / (1.0 + queue_wait_s / max(service_time_s, 1e-12))
    kv_score = effective_reuse_tokens / required_context if required_context > 0 else 0.0
    latency_score = 1.0 - max(0.0, min(math.pi, route.arc_distance_rad)) / math.pi
    candidate_id = f"{access.access_type}:{compute_satellite_id}:{route.path_id}"
    violations = (
        ("energy_unavailable",)
        if link_update["energy_violation"] or compute_update["energy_violation"]
        else ()
    )
    candidate = SchedulerCandidate(
        candidate_id=candidate_id,
        compute_satellite_id=compute_satellite_id,
        feasible=not violations,
        route_plan=route.to_dict(),
        latency_score=latency_score,
        kv_reuse_score=kv_score,
        energy_score=energy_score,
        compute_load_score=compute_load_score,
        arc_distance_rad=route.arc_distance_rad,
        compute_queue_wait_s=queue_wait_s,
        hard_gate_violations=violations,
    )
    return {
        "candidate": candidate,
        "route": route,
        "inbound_path": inbound_path,
        "first_path": first_path,
        "output_path": output_path,
        "copy_path": copy_path,
        "copied_from": copied_from,
        "copy_done_s": copy_done_s,
        "prefill_start_s": prefill_start_s,
        "prefill_done_s": prefill_done_s,
        "decode_start_s": decode_start_s,
        "decode_done_s": decode_done_s,
        "first_done_s": first_done_s,
        "output_done_s": output_done_s,
        "prefill_service_s": prefill_service_s,
        "decode_service_s": decode_service_s,
        "prefill_wait_s": prefill_wait_s,
        "decode_wait_s": decode_wait_s,
        "dvfs_scale": dvfs_scale,
        "projected_soc_after": preview.soc,
        "local_hit_tokens": local_hit_tokens,
        "effective_reuse_tokens": effective_reuse_tokens,
        "recompute_tokens": recompute_tokens,
        "energy_window": energy_window,
        "link_bytes_by_satellite": link_bytes_by_satellite,
    }


def _commit_candidate(
    *,
    state: RuntimeState,
    request: dict[str, Any],
    strategy: Any,
    trace_bundle: Any,
    energy_trace: Any,
    serving: Any,
    payload: dict[str, Any],
    selection: Any,
) -> RuntimeRequestRecord:
    route: RoutePlan = payload["route"]
    satellite = state.satellite_for(route.compute_satellite_id)
    affected_ids = set(payload["link_bytes_by_satellite"])
    affected_ids.add(route.compute_satellite_id)
    ledger_starts = {
        satellite_id: len(state.satellite_for(satellite_id).energy_ledger)
        for satellite_id in affected_ids
    }
    prefill_start_s, prefill_done_s, prefill_wait_s = satellite.compute_queue.reserve_prefill(
        payload["copy_done_s"], payload["prefill_service_s"]
    )
    decode_start_s, decode_done_s, decode_wait_s = satellite.compute_queue.reserve_decode(
        prefill_done_s, payload["decode_service_s"]
    )
    if not math.isclose(prefill_start_s, payload["prefill_start_s"], abs_tol=1e-9):
        raise ValueError("candidate_preview_state_changed")
    link_violation = False
    for satellite_id, link_bytes in sorted(payload["link_bytes_by_satellite"].items()):
        endpoint = state.satellite_for(satellite_id)
        advance_satellite_energy(endpoint, energy_trace, prefill_start_s)
        endpoint_window = energy_window_at(energy_trace, satellite_id, prefill_start_s)
        update = apply_request_energy(
            endpoint,
            prompt_tokens=0,
            output_tokens=0,
            link_bytes=link_bytes,
            energy_window=endpoint_window,
            event_time_s=prefill_start_s,
            metadata={"request_id": request["request_id"], "energy_stage": "communication"},
        )
        link_violation = link_violation or bool(update["energy_violation"])
    advance_satellite_energy(satellite, energy_trace, prefill_start_s)
    compute_update = apply_request_energy(
        satellite,
        prompt_tokens=payload["recompute_tokens"],
        output_tokens=request["output_tokens"],
        link_bytes=0.0,
        energy_window=payload["energy_window"],
        event_time_s=prefill_start_s,
        metadata={"request_id": request["request_id"], "preview": False},
    )
    advance_satellite_energy(satellite, energy_trace, decode_done_s)
    context_tokens = int(request["prompt_tokens"]) + int(request["output_tokens"])
    kv_event = commit_session_kv(
        state,
        session_id=str(request.get("session_id")) if request.get("session_id") else None,
        satellite_id=route.compute_satellite_id,
        context_tokens=context_tokens,
        capacity_tokens=serving.kv_cache_capacity_tokens,
        kv_bytes_per_token=float(serving.kv_cache_bytes_per_token or 0.0),
        now_s=decode_done_s,
        copied_from=payload["copied_from"],
    )
    selected_path = tuple(
        (*payload["inbound_path"], *payload["copy_path"], *payload["output_path"])
    )
    ttft_s = payload["first_done_s"] - float(request["arrival_time_s"])
    e2e_s = payload["output_done_s"] - float(request["arrival_time_s"])
    energy_ledger = tuple(
        row
        for satellite_id in sorted(affected_ids)
        for row in state.satellite_for(satellite_id).energy_ledger[ledger_starts[satellite_id]:]
    )
    return RuntimeRequestRecord(
        request_id=request["request_id"],
        arrival_time_s=request["arrival_time_s"],
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        session_id=request.get("session_id"),
        region_id=request.get("region_id"),
        slo_class=request.get("slo_class"),
        action_type="compute",
        access_type=route.access_type,
        access_satellite_id=route.access_satellite_id,
        serving_satellite_id=route.compute_satellite_id,
        compute_owner_satellite_id=route.compute_satellite_id,
        compute_owner_satellite_ids=(route.compute_satellite_id,),
        dvfs_scale=payload["dvfs_scale"],
        completed=True,
        prefill_queue_wait_s=prefill_wait_s,
        decode_queue_wait_s=decode_wait_s,
        prefill_service_s=payload["prefill_service_s"],
        decode_service_s=payload["decode_service_s"],
        link_transmission_s=sum(float(leg.get("transmission_s", 0.0)) for leg in selected_path),
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        selected_path=selected_path,
        soc_before=compute_update["soc_before"],
        soc_after=satellite.soc,
        energy_violation=link_violation or compute_update["energy_violation"],
        energy_ledger=energy_ledger,
        compute_sunlit=bool(payload["energy_window"].get("sunlit", True)),
        kv_serve_action=kv_event.get("action"),
        kv_hit_tokens=payload["effective_reuse_tokens"],
        kv_miss_tokens=payload["recompute_tokens"],
        context_recompute_tokens=payload["recompute_tokens"],
        kv_evicted_bytes=float(kv_event.get("evicted_bytes", 0.0) or 0.0),
        kv_evicted_entry_count=len(tuple(kv_event.get("evicted_session_ids", ()) or ())),
        hbm_occupancy_after=kv_event.get("occupancy_bytes_after"),
        bottleneck_reason="compute_queue" if prefill_wait_s + decode_wait_s > 0.0 else "none",
        provenance={
            "score": selection.score,
            "score_terms": selection.score_terms,
            "communication_queue_model": "not_modeled",
            "trace_id": trace_bundle.trace_id,
        },
    )


def _preview_kv_copy(
    *,
    state: RuntimeState,
    session_id: str | None,
    target_satellite_id: str,
    required_context_tokens: int,
    local_hit_tokens: int,
    copy_ratio: float,
    kv_bytes_per_token: float,
    trace_bundle: Any,
    start_s: float,
    isl_bandwidth_bps: float,
) -> dict[str, Any]:
    """Choose one deterministic, routable KV source without mutating state."""

    empty = {
        "source_satellite_id": None,
        "copied_tokens": 0,
        "done_s": float(start_s),
        "path": (),
    }
    if not session_id or copy_ratio <= 0.0 or required_context_tokens <= local_hit_tokens:
        return empty
    session = state.sessions.get(session_id)
    if session is None:
        return empty
    options: list[tuple[int, float, str, int, tuple[dict[str, Any], ...]]] = []
    for source_satellite_id in sorted(set(session.satellite_ids)):
        if source_satellite_id == target_satellite_id:
            continue
        source_reusable_tokens = local_kv_reuse_tokens(
            state,
            satellite_id=source_satellite_id,
            session_id=session_id,
            required_context_tokens=required_context_tokens,
        )
        remote_missing_tokens = max(0, source_reusable_tokens - local_hit_tokens)
        copied_tokens = int(math.floor(copy_ratio * remote_missing_tokens))
        if copied_tokens <= 0:
            continue
        legs = build_isl_transfer_legs(
            trace_bundle,
            source_satellite_id=source_satellite_id,
            target_satellite_id=target_satellite_id,
            time_s=start_s,
            payload_bytes=copied_tokens * kv_bytes_per_token,
            bandwidth_bps=isl_bandwidth_bps,
        )
        if legs is None:
            continue
        try:
            done_s, path = execute_route(legs, start_s)
        except RouteUnavailable:
            continue
        options.append(
            (-source_reusable_tokens, done_s, source_satellite_id, copied_tokens, path)
        )
    if not options:
        return empty
    _negative_reusable, done_s, source_satellite_id, copied_tokens, path = min(options)
    return {
        "source_satellite_id": source_satellite_id,
        "copied_tokens": copied_tokens,
        "done_s": done_s,
        "path": path,
    }


def _energy_window_with_dvfs(window: dict[str, Any], dvfs_scale: float) -> dict[str, Any]:
    """Apply Runtime's equal compute-capacity/power DVFS assumption."""

    base = float(window["compute_energy_j_per_token"])
    return {
        **dict(window),
        "compute_energy_j_per_token": base * dvfs_scale,
        "base_compute_energy_j_per_token": base,
        "runtime_compute_energy_scale": dvfs_scale,
    }


def _build_energy_trace(trace_bundle: Any, profile: dict[str, Any]) -> tuple[Any, float]:
    required = (
        "battery_capacity_j",
        "charge_rate_w",
        "idle_power_w",
        "compute_energy_j_per_token",
        "link_energy_j_per_byte",
        "soc_min",
        "initial_soc",
    )
    missing = [field for field in required if profile.get(field) is None]
    if missing:
        raise ValueError("runtime_energy_profile_missing:" + ",".join(missing))
    static = {
        "battery_capacity_j": float(profile["battery_capacity_j"]),
        "idle_power_w": float(profile["idle_power_w"]),
        "compute_energy_j_per_token": float(profile["compute_energy_j_per_token"]),
        "link_energy_j_per_byte": float(profile["link_energy_j_per_byte"]),
        "soc_min": float(profile["soc_min"]),
        "battery_charge_efficiency": float(profile.get("battery_charge_efficiency", 1.0)),
        "battery_discharge_efficiency": float(profile.get("battery_discharge_efficiency", 1.0)),
        "energy_accounting_mode": str(profile.get("energy_accounting_mode", "event_incremental_v1")),
    }
    windows = tuple(
        {
            **dict(window),
            **static,
            "charge_rate_w": float(profile["charge_rate_w"]) if bool(window.get("sunlit", not window.get("eclipsed", False))) else 0.0,
            "sunlit": bool(window.get("sunlit", not window.get("eclipsed", False))),
        }
        for window in trace_bundle.sunlight_windows
    )
    return SimpleNamespace(energy_windows=windows), float(profile["initial_soc"])


def _architecture_bandwidth(
    constellation: dict[str, Any],
    keys: tuple[str, ...],
    *,
    required: bool,
) -> float:
    architecture = dict(constellation.get("architecture") or {})
    value = _recursive_first(architecture, keys)
    if value is None:
        if required:
            raise ValueError(f"runtime_bandwidth_missing:{keys[0]}")
        return 0.0
    numeric = float(value)
    if numeric <= 0.0:
        raise ValueError(f"runtime_bandwidth_invalid:{keys[0]}")
    return numeric


def _recursive_first(value: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        for key in keys:
            if value.get(key) is not None:
                numeric = value[key]
                return float(numeric) * 1e9 if key.endswith("_gbps") else numeric
        for nested in value.values():
            found = _recursive_first(nested, keys)
            if found is not None:
                return found
    return None


def _link_bytes_by_satellite(
    paths: tuple[tuple[dict[str, Any], ...], ...],
    satellite_ids: set[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for path in paths:
        for leg in path:
            kind = str(leg.get("kind") or "")
            payload_bytes = float(leg.get("payload_bytes", 0.0) or 0.0)
            endpoints = (str(leg.get("source_node_id") or ""), str(leg.get("target_node_id") or ""))
            for endpoint in endpoints:
                if endpoint not in satellite_ids:
                    continue
                share = payload_bytes / 2.0 if kind in {"isl", "kv_copy"} else payload_bytes
                result[endpoint] = result.get(endpoint, 0.0) + share
    return result


def _service_time(tokens: int, rate: Any) -> float:
    value = float(rate or 0.0)
    if value <= 0.0:
        raise ValueError("serving_rate_unavailable")
    return max(0, int(tokens)) / value


def _first_token_bytes(request: dict[str, Any]) -> float:
    return float(request["output_bytes"]) / max(1, int(request["output_tokens"]))


def _uncompleted(request: dict[str, Any], reason: str) -> RuntimeRequestRecord:
    return RuntimeRequestRecord(
        request_id=str(request.get("request_id") or "unknown"),
        arrival_time_s=float(request.get("arrival_time_s", 0.0)),
        prompt_tokens=int(request.get("prompt_tokens", 0)),
        output_tokens=int(request.get("output_tokens", 0)),
        session_id=request.get("session_id"),
        region_id=request.get("region_id"),
        slo_class=request.get("slo_class"),
        completed=False,
        uncompleted_reason=reason,
        bottleneck_reason=reason,
    )


def _decision(request: dict[str, Any], selection: Any, candidates: Any) -> dict[str, Any]:
    return {
        "request_id": request.get("request_id"),
        "selected_candidate_id": getattr(selection, "selected_candidate_id", None),
        "score": getattr(selection, "score", None),
        "score_terms": getattr(selection, "score_terms", {}),
        "candidate_count": len(tuple(candidates)),
    }


def _all_uncompleted(simulation_spec: Any, reason: str) -> RuntimeEngineResult:
    records = tuple(_uncompleted(normalize_runtime_request(row), reason) for row in simulation_spec.requests)
    return RuntimeEngineResult(
        records=records,
        decision_traces=tuple(_decision(row, None, ()) for row in simulation_spec.requests),
        state_summary=RuntimeState().to_dict(),
        warnings=(reason,),
        provenance={"runtime_engine": "deterministic_event_runtime"},
    )


__all__ = ["normalize_runtime_request", "run_runtime_engine"]
