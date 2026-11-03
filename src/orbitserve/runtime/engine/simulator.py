"""Deterministic event simulation for one assembled Runtime input."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

from .contracts import RuntimeEngineResult
from .request_input import normalize_runtime_request
from .decision import SchedulerCandidate
from .routing import AccessCandidate, RoutePlan, RouteUnavailable
from ..state.network import NetworkState
from ..metrics.records import RuntimeRequestRecord
from ..setup.serving_time import decode_time_s, prefill_suffix_time_s
from ..state.energy import (
    advance_satellite_energy,
    apply_request_energy,
    energy_window_at,
)
from ..state.kv_cache import (
    kv_locations_for_session,
    local_kv_reuse_tokens,
)
from ..state.runtime_state import RuntimeState


def run_runtime_engine(simulation_spec: Any) -> RuntimeEngineResult:
    from .events import run_event_engine

    return run_event_engine(simulation_spec)


def _select_access(
    *,
    state: RuntimeState,
    request: dict[str, Any],
    simulation_spec: Any,
    environment_state: Any,
    energy_trace: Any,
    eligible_satellite_ids: tuple[str, ...],
    sgl_bandwidth_bps: float,
    horizon_s: float,
) -> AccessCandidate:
    mode = simulation_spec.strategy.ingress_selection
    ingress_id = str(request.get("ingress_id") or "")
    network = environment_state.network
    time_s = float(request["arrival_time_s"])
    rows = []
    for edge in network.edges:
        if edge["kind"] != "sgl" or ingress_id not in (
            edge["endpoint_a"],
            edge["endpoint_b"],
        ):
            continue
        node = (
            edge["endpoint_b"]
            if edge["endpoint_a"] == ingress_id
            else edge["endpoint_a"]
        )
        if mode != "nearest_satellite" and network.nodes[node].role != "compute":
            continue
        leg = network._edge_transfer(
            edge["link_id"], ingress_id, node, time_s, 0.0, True
        )
        if leg is not None:
            rows.append(
                AccessCandidate(
                    network.nodes[node].role,
                    ingress_id,
                    node,
                    leg["start_s"],
                    leg["geometry_end_s"],
                    leg["propagation_delay_s"],
                )
            )
    if not rows:
        raise RouteUnavailable("no_physical_ingress")
    if mode == "longest_visible_compute_satellite":
        return min(rows, key=lambda item: (-item.end_s, item.satellite_id))
    if mode == "highest_energy_compute_satellite":

        def energy(item):
            satellite = state.satellite_for(item.satellite_id)
            preview = replace(satellite)
            advance_satellite_energy(preview, energy_trace, time_s)
            return (-preview.soc, item.satellite_id)

        return min(rows, key=energy)
    return min(
        rows,
        key=lambda item: (item.start_s + item.propagation_delay_s, item.satellite_id),
    )


def _preview_candidate(
    *,
    state: RuntimeState,
    request: dict[str, Any],
    strategy: Any,
    environment_state: Any,
    energy_trace: Any,
    serving: Any,
    access: AccessCandidate,
    compute_satellite_id: str,
    isl_bandwidth_bps: float,
) -> dict[str, Any]:
    network = environment_state.network.fork()
    access_done, access_path = network.transfer(
        access.ingress_id,
        access.satellite_id,
        float(request["arrival_time_s"]),
        float(request["input_bytes"]),
        strategy.routing_policy,
        commit=True,
        kinds={"sgl"},
    )
    inbound_done_s, relay_path = network.transfer(
        access.satellite_id,
        compute_satellite_id,
        access_done,
        float(request["input_bytes"]),
        strategy.routing_policy,
        commit=True,
        kinds={
            "communication_isl",
            "cross_layer",
            "cell_isl",
            "inter_cell",
        },
    )
    inbound_path = (*access_path, *relay_path)
    route = RoutePlan(
        access.access_type, access.satellite_id, compute_satellite_id, inbound_path
    )
    session_id = request.get("session_id")
    prompt_length = int(request["prompt_tokens"])
    reusable_context = min(
        prompt_length,
        int(request.get("context_tokens_before_turn", 0) or 0),
    )
    local_hit_tokens = local_kv_reuse_tokens(
        state,
        satellite_id=compute_satellite_id,
        session_id=str(session_id) if session_id else None,
        required_context_tokens=reusable_context,
    )
    satellite = state.satellite_for(compute_satellite_id)
    comparison_concurrency = satellite.compute_queue.comparison_concurrency_at(
        inbound_done_s
    )
    copy_plan = _preview_kv_copy(
        state=state,
        session_id=str(session_id) if session_id else None,
        target_satellite_id=compute_satellite_id,
        required_context_tokens=reusable_context,
        prompt_length=prompt_length,
        local_hit_tokens=local_hit_tokens,
        copy_enabled=bool(strategy.kv_cache_copy_enabled),
        kv_bytes_per_token=float(serving.kv_cache_bytes_per_token or 0.0),
        environment_state=environment_state,
        start_s=inbound_done_s,
        isl_bandwidth_bps=isl_bandwidth_bps,
        prefill_rate_table=serving.prefill_rate_table,
        concurrency=comparison_concurrency,
        network=network,
        routing_policy=strategy.routing_policy,
    )
    copy_path = copy_plan["path"]
    copied_from = copy_plan["source_satellite_id"]
    copied_tokens = copy_plan["copied_tokens"]
    copy_done_s = copy_plan["done_s"]
    effective_reuse_tokens = min(
        reusable_context,
        local_hit_tokens + copied_tokens,
    )
    recompute_tokens = max(0, prompt_length - effective_reuse_tokens)
    compute_ready_s = copy_done_s

    def service_profile(concurrency: int, start_s: float) -> dict[str, Any]:
        raw_window = energy_window_at(
            energy_trace,
            compute_satellite_id,
            start_s,
        )
        dvfs = (
            float(strategy.shadow_dvfs_scale)
            if strategy.shadow_dvfs_enabled and not bool(raw_window.get("sunlit", True))
            else 1.0
        )
        prefill_s = (
            prefill_suffix_time_s(
                serving.prefill_rate_table,
                prompt_length=prompt_length,
                cached_prefix_tokens=effective_reuse_tokens,
                concurrency=concurrency,
            )
            / dvfs
        )
        decode_s = (
            decode_time_s(
                serving.decode_rate_table,
                context_length=prompt_length,
                output_tokens=request["output_tokens"],
                concurrency=concurrency,
            )
            / dvfs
        )
        return {
            "dvfs_scale": dvfs,
            "energy_window": raw_window,
            "prefill_service_s": prefill_s,
            "decode_service_s": decode_s,
        }

    reservation = satellite.compute_queue.preview(
        request_id=request["request_id"],
        ready_s=compute_ready_s,
        duration_for=lambda concurrency, start_s: sum(
            service_profile(concurrency, start_s)[key]
            for key in ("prefill_service_s", "decode_service_s")
        ),
    )
    concurrency = reservation.concurrency_at_admission
    service = service_profile(concurrency, reservation.start_s)
    dvfs_scale = service["dvfs_scale"]
    prefill_service_s = service["prefill_service_s"]
    decode_service_s = service["decode_service_s"]
    prefill_start_s = reservation.start_s
    prefill_done_s = prefill_start_s + prefill_service_s
    prefill_wait_s = max(0.0, prefill_start_s - compute_ready_s)
    decode_start_s = prefill_done_s
    decode_done_s = reservation.finish_s
    decode_wait_s = 0.0
    first_token_service_s = (
        decode_time_s(
            serving.decode_rate_table,
            context_length=prompt_length,
            output_tokens=1,
            concurrency=concurrency,
        )
        / dvfs_scale
    )
    first_token_bytes = _first_token_bytes(request)
    first_done_s, first_path = network.transfer(
        compute_satellite_id,
        access.ingress_id,
        decode_start_s + first_token_service_s,
        first_token_bytes,
        strategy.routing_policy,
        commit=True,
    )
    remaining_output_bytes = max(
        0.0, float(request["output_bytes"]) - first_token_bytes
    )
    remaining_done_s, output_path = (first_done_s, ())
    if remaining_output_bytes:
        remaining_done_s, output_path = network.transfer(
            compute_satellite_id,
            access.ingress_id,
            decode_done_s,
            remaining_output_bytes,
            strategy.routing_policy,
            commit=True,
        )
    route = replace(route, outbound_legs=(*first_path, *output_path))
    output_done_s = max(first_done_s, remaining_done_s)

    energy_window = service["energy_window"]
    physical_paths = (inbound_path, first_path, output_path, copy_path)
    link_bytes_by_satellite = _link_bytes_by_satellite(
        physical_paths,
        {str(row["satellite_id"]) for row in environment_state.satellite_catalog},
    )
    projected_soc_after, energy_violation = _preview_candidate_energy(
        state=state,
        network=network,
        energy_trace=energy_trace,
        physical_paths=physical_paths,
        compute_satellite_id=compute_satellite_id,
        compute_start_s=prefill_start_s,
        compute_end_s=decode_done_s,
        prompt_tokens=recompute_tokens,
        output_tokens=request["output_tokens"],
    )
    energy_score = max(0.0, min(1.0, projected_soc_after))
    service_time_s = prefill_service_s + decode_service_s
    queue_wait_s = prefill_wait_s + decode_wait_s
    compute_load_score = 1.0 / (1.0 + queue_wait_s / max(service_time_s, 1e-12))
    kv_score = (
        effective_reuse_tokens / reusable_context if reusable_context > 0 else 0.0
    )
    predicted_network_s = sum(
        float(leg["arrival_s"]) - float(leg["start_s"]) + float(leg["queue_wait_s"])
        for path in physical_paths
        for leg in path
    )
    latency_score = 1.0 / (1.0 + predicted_network_s)
    candidate_id = f"{access.access_type}:{compute_satellite_id}:{route.path_id}"
    violations = ("energy_unavailable",) if energy_violation else ()
    candidate = SchedulerCandidate(
        candidate_id=candidate_id,
        compute_satellite_id=compute_satellite_id,
        feasible=not violations,
        route_plan=route.to_dict(),
        latency_score=latency_score,
        kv_reuse_score=kv_score,
        energy_score=energy_score,
        compute_load_score=compute_load_score,
        predicted_network_s=predicted_network_s,
        compute_queue_wait_s=queue_wait_s,
        hard_gate_violations=violations,
    )
    return {
        "network": network,
        "candidate": candidate,
        "route": route,
        "inbound_path": inbound_path,
        "first_path": first_path,
        "output_path": output_path,
        "copy_path": copy_path,
        "copied_from": copied_from,
        "copied_tokens": copied_tokens,
        "copy_done_s": copy_done_s,
        "reservation": reservation,
        "concurrency_at_admission": concurrency,
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
        "projected_soc_after": projected_soc_after,
        "local_hit_tokens": local_hit_tokens,
        "effective_reuse_tokens": effective_reuse_tokens,
        "recompute_tokens": recompute_tokens,
        "energy_window": energy_window,
        "link_bytes_by_satellite": link_bytes_by_satellite,
    }


def _compute_members(network: NetworkState, node_id: str) -> list[str]:
    cell = network.nodes[node_id].cell_id
    if cell:
        return sorted(
            node.node_id
            for node in network.nodes.values()
            if node.role == "compute" and node.cell_id == cell
        )
    return [node_id]


def _preview_candidate_energy(
    *,
    state: RuntimeState,
    network: NetworkState,
    energy_trace: Any,
    physical_paths,
    compute_satellite_id: str,
    compute_start_s: float,
    compute_end_s: float,
    prompt_tokens: int,
    output_tokens: int,
) -> tuple[float, bool]:
    """Check every physical participant at its predicted debit time, in isolation.

    Match execution: charge each hop's endpoints when serialization starts and
    share compute work across all cell members. A healthy compute target cannot
    make an energy-starved relay or cell member into a feasible candidate.
    """
    events = []
    for path in physical_paths:
        for leg in path:
            size = float(leg["payload_bytes"])
            if leg.get("link_kind", leg["kind"]) != "sgl":
                size /= 2
            for endpoint in (leg["source_node_id"], leg["target_node_id"]):
                if network.nodes[endpoint].role != "ground":
                    events.append((float(leg["start_s"]), endpoint, 0.0, 0.0, size))
    members = _compute_members(network, compute_satellite_id)
    for member in members:
        events.append(
            (
                compute_start_s,
                member,
                prompt_tokens / len(members),
                output_tokens / len(members),
                0.0,
            )
        )
    previews = {}
    violation = False
    for time_s, node_id, prompt, output, size in sorted(events, key=lambda row: row[0]):
        if node_id not in previews:
            # Energy histories are immutable tuples; all new events and snapshots
            # belong to this copy, including rejected/unselected candidates.
            previews[node_id] = replace(state.compute_nodes[node_id])
        preview = previews[node_id]
        advance_satellite_energy(preview, energy_trace, time_s)
        update = apply_request_energy(
            preview,
            prompt_tokens=prompt,
            output_tokens=output,
            link_bytes=size,
            energy_window=energy_window_at(energy_trace, node_id, time_s),
            event_time_s=time_s,
            metadata={"preview": True},
        )
        violation |= update["energy_violation"]
    preview = previews[compute_satellite_id]
    advance_satellite_energy(
        preview, energy_trace, max(compute_end_s, preview.last_energy_update_s)
    )
    return float(preview.soc), violation


def _preview_kv_copy(
    *,
    state: RuntimeState,
    session_id: str | None,
    target_satellite_id: str,
    required_context_tokens: int,
    prompt_length: int,
    local_hit_tokens: int,
    copy_enabled: bool,
    kv_bytes_per_token: float,
    environment_state: Any,
    start_s: float,
    isl_bandwidth_bps: float,
    prefill_rate_table: tuple[dict[str, Any], ...],
    concurrency: int,
    network: NetworkState,
    routing_policy: str = "earliest_arrival",
) -> dict[str, Any]:
    """Choose a same-session source only when copy beats local recompute."""

    empty = {
        "source_satellite_id": None,
        "copied_tokens": 0,
        "done_s": float(start_s),
        "path": (),
    }
    if (
        not session_id
        or not copy_enabled
        or required_context_tokens <= local_hit_tokens
    ):
        return empty
    local_prefill_time_s = prefill_suffix_time_s(
        prefill_rate_table,
        prompt_length=prompt_length,
        cached_prefix_tokens=local_hit_tokens,
        concurrency=concurrency,
    )
    options: list[dict[str, Any]] = []
    for source_satellite_id in kv_locations_for_session(state, session_id):
        if source_satellite_id == target_satellite_id:
            continue
        source_reusable_tokens = local_kv_reuse_tokens(
            state,
            satellite_id=source_satellite_id,
            session_id=session_id,
            required_context_tokens=required_context_tokens,
        )
        copied_tokens = max(
            0,
            min(source_reusable_tokens, required_context_tokens) - local_hit_tokens,
        )
        if copied_tokens <= 0:
            continue
        trial = network.fork()
        try:
            done_s, path = trial.transfer(
                source_satellite_id,
                target_satellite_id,
                start_s,
                copied_tokens * kv_bytes_per_token,
                routing_policy,
                commit=True,
                kinds={
                    "communication_isl",
                    "cross_layer",
                    "cell_isl",
                    "inter_cell",
                },
            )
            path = tuple(
                {**leg, "link_kind": leg["kind"], "kind": "kv_copy"} for leg in path
            )
        except RouteUnavailable:
            continue
        migration_time_s = done_s - start_s
        copied_prefill_time_s = prefill_suffix_time_s(
            prefill_rate_table,
            prompt_length=prompt_length,
            cached_prefix_tokens=source_reusable_tokens,
            concurrency=concurrency,
        )
        total_time_s = migration_time_s + copied_prefill_time_s
        if total_time_s >= local_prefill_time_s:
            continue
        options.append(
            {
                "network": trial,
                "source_satellite_id": source_satellite_id,
                "source_reusable_tokens": source_reusable_tokens,
                "copied_tokens": copied_tokens,
                "done_s": done_s,
                "migration_time_s": migration_time_s,
                "total_time_s": total_time_s,
                "path": path,
            }
        )
    if not options:
        return empty
    selected = min(
        options,
        key=lambda option: (
            option["total_time_s"],
            option["migration_time_s"],
            option["source_satellite_id"],
        ),
    )
    network.calendars = selected["network"].calendars
    return {
        "source_satellite_id": selected["source_satellite_id"],
        "copied_tokens": selected["copied_tokens"],
        "done_s": selected["done_s"],
        "path": selected["path"],
    }


def _build_energy_trace(
    environment_state: Any, profile: dict[str, Any]
) -> tuple[Any, float]:
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
        "battery_charge_efficiency": float(
            profile.get("battery_charge_efficiency", 1.0)
        ),
        "battery_discharge_efficiency": float(
            profile.get("battery_discharge_efficiency", 1.0)
        ),
        "energy_accounting_mode": str(
            profile.get("energy_accounting_mode", "event_incremental_v1")
        ),
    }
    windows = tuple(
        {
            **dict(window),
            **static,
            **{
                k: v
                for k, v in profile.get("node_profiles", {})
                .get(str(window["satellite_id"]), {})
                .items()
                if k != "provenance"
            },
            "charge_rate_w": float(
                profile.get("node_profiles", {}).get(
                    str(window["satellite_id"]), profile
                )["charge_rate_w"]
            )
            if bool(window.get("sunlit", not window.get("eclipsed", False)))
            else 0.0,
            "sunlit": bool(window.get("sunlit", not window.get("eclipsed", False))),
        }
        for window in environment_state.sunlight_windows
    )
    return SimpleNamespace(energy_windows=windows), float(profile["initial_soc"])


def _link_bytes_by_satellite(
    paths: tuple[tuple[dict[str, Any], ...], ...],
    satellite_ids: set[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for path in paths:
        for leg in path:
            kind = str(leg.get("kind") or "")
            payload_bytes = float(leg.get("payload_bytes", 0.0) or 0.0)
            endpoints = (
                str(leg.get("source_node_id") or ""),
                str(leg.get("target_node_id") or ""),
            )
            for endpoint in endpoints:
                if endpoint not in satellite_ids:
                    continue
                share = (
                    payload_bytes / 2.0
                    if str(leg.get("link_kind", kind)) != "sgl"
                    else payload_bytes
                )
                result[endpoint] = result.get(endpoint, 0.0) + share
    return result


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
        completed=False,
        uncompleted_reason=reason,
        bottleneck_reason=reason,
    )


def _decision(
    request: dict[str, Any], selection: Any, candidates: Any
) -> dict[str, Any]:
    return {
        "request_id": request.get("request_id"),
        "selected_candidate_id": getattr(selection, "selected_candidate_id", None),
        "score": getattr(selection, "score", None),
        "score_terms": getattr(selection, "score_terms", {}),
        "candidate_count": len(tuple(candidates)),
    }


def _all_uncompleted(simulation_spec: Any, reason: str) -> RuntimeEngineResult:
    records = tuple(
        _uncompleted(normalize_runtime_request(row), reason)
        for row in simulation_spec.requests
    )
    return RuntimeEngineResult(
        records=records,
        decision_traces=tuple(
            _decision(row, None, ()) for row in simulation_spec.requests
        ),
        state_summary=RuntimeState().to_dict(),
        warnings=(reason,),
        provenance={"runtime_engine": "deterministic_event_runtime"},
    )


__all__ = ["normalize_runtime_request", "run_runtime_engine"]
