"""Event-engine implementations for OrbitServe runtime evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from ...common import to_jsonable
from .contracts import RuntimeEngineResult
from ..state.energy import EnergyWindowError, advance_satellite_energy, apply_request_energy, energy_window_at
from .routing import (
    RouteLeg,
    RouteSamplingError,
    _build_ring_path_from_graph,
    _ring_graph_from_links,
    enumerate_compute_candidates,
    find_access_candidates,
    find_next_access_candidate,
    make_route_plan,
    query_active_links_at,
    query_access_at,
    query_link_sample_at,
    sample_route_leg,
)
from ..contracts import ServingCapacityProfile
from .runtime_eligibility import ComputeEligibilitySet, build_compute_eligibility_set
from ..metrics.records import RuntimeRequestRecord
from ..state.runtime_state import RuntimeState
from .runtime_timing import TTFT_DEFINITION, ttft_breakdown
from ..state.kv_cache import (
    _admit_session_kv,
    _remove_session_kv,
)
from ..strategy.schedulers import SchedulerCandidate, get_scheduler
from ...orbit import TRACE_BACKED_RUNTIME_SURROGATE_SCOPE, normalize_trace_scope




def run_trace_backed_runtime(
    *,
    requests: tuple[dict[str, Any], ...],
    serving_profile: ServingCapacityProfile,
    runtime_strategy: Any,
    trace_bundle: Any,
    time_horizon_s: float | None = None,
    scope: str = TRACE_BACKED_RUNTIME_SURROGATE_SCOPE,
) -> RuntimeEngineResult:
    """Run a deterministic surrogate access/link/energy runtime model."""

    state = RuntimeState()
    scope = normalize_trace_scope(scope)
    warnings = list(getattr(trace_bundle, "warnings", ()))
    warnings.extend(serving_profile.warnings)
    trace_model = str(getattr(trace_bundle, "trace_model", "surrogate_orbit_trace"))
    policy = str(getattr(runtime_strategy, "policy", "local_only"))
    deployment = serving_profile
    satellite_catalog = tuple(getattr(trace_bundle, "satellite_catalog", ()) or ())
    if len(satellite_catalog) <= 0:
        return RuntimeEngineResult(
            records=tuple(
                _reject_record(
                    normalize_runtime_request(request),
                    "satellite_catalog_empty",
                    scope=scope,
                    trace_model=trace_model,
                )
                for request in requests
            ),
            decision_traces=(),
            state_summary=state.to_dict(),
            warnings=("surrogate trace satellite_catalog is empty",),
            scope=scope,
            trace_model=trace_model,
            provenance={"runtime_engine": "trace_backed_runtime_surrogate"},
            valid_for_paper_main_results=False,
        )
    if deployment is None or not deployment.has_compute_capacity:
        return RuntimeEngineResult(
            records=tuple(
                _reject_record(
                    normalize_runtime_request(request),
                    "compute_capacity_unavailable",
                    scope=scope,
                    trace_model=trace_model,
                )
                for request in requests
            ),
            decision_traces=(),
            state_summary=state.to_dict(),
            warnings=("runtime serving profile has no usable compute capacity",),
            scope=scope,
            trace_model=trace_model,
            provenance={"runtime_engine": "trace_backed_runtime_surrogate"},
            valid_for_paper_main_results=False,
        )

    for satellite in satellite_catalog:
        satellite_id = str(satellite.get("satellite_id"))
        initial_soc = float(getattr(trace_bundle, "satellite_initial_soc", {}).get(satellite_id, 0.8))
        state.satellite_for(satellite_id, initial_soc=initial_soc)
    for link in getattr(trace_bundle, "links", ()) or ():
        state.link_for(str(link.get("link_id")))
    for link in getattr(trace_bundle, "physical_links", ()) or ():
        state.link_for(str(link.get("link_id")))
    for link in _dict(getattr(trace_bundle, "link_trace", None)).get("physical_links", ()) or ():
        state.link_for(str(link.get("link_id")))

    ordered_requests = sorted(
        ((normalize_runtime_request(request), request) for request in requests),
        key=lambda item: (item[0]["arrival_time_s"], item[0]["request_id"]),
    )
    records: list[RuntimeRequestRecord] = []
    decision_traces: list[dict[str, Any]] = []
    session_action_counts: dict[str, int] = {}
    max_hop_count = getattr(runtime_strategy, "max_hop_count", None)
    scheduler = get_scheduler(runtime_strategy)
    trace_horizon_s, horizon_warnings = _trace_horizon_s(trace_bundle)
    if trace_horizon_s is None:
        warnings.append("trace_horizon_missing")
    else:
        configured_horizon_s = _optional_float(time_horizon_s)
        if configured_horizon_s is not None and abs(configured_horizon_s - trace_horizon_s) > 1e-9:
            warnings.append("trace_horizon_mismatch")
    warnings.extend(horizon_warnings)

    for normalized, raw_request in ordered_requests:
        if normalized["output_tokens"] <= 0:
            record = _reject_record(
                normalized,
                "output_tokens_must_be_positive_for_ttft",
                scope=scope,
                trace_model=trace_model,
                provenance={"ttft_definition": TTFT_DEFINITION},
            )
            trace = _decision_trace(
                normalized,
                policy,
                (),
                None,
                "output_tokens_must_be_positive_for_ttft",
            )
        else:
            record, trace = _evaluate_trace_request(
                state=state,
                request=normalized,
                raw_request=raw_request,
                trace_bundle=trace_bundle,
                deployment=deployment,
                policy=policy,
                scheduler=scheduler,
                runtime_strategy=runtime_strategy,
                max_hop_count=max_hop_count,
                scope=scope,
                trace_model=trace_model,
                trace_horizon_s=trace_horizon_s,
            )
        records.append(record)
        decision_traces.append(trace)
        if record.session_action:
            session_action_counts[record.session_action] = session_action_counts.get(record.session_action, 0) + 1

    warnings.append("trace-backed runtime is a preliminary surrogate model, not paper-main STK-calibrated simulation")
    return RuntimeEngineResult(
        records=tuple(records),
        decision_traces=tuple(decision_traces),
        state_summary=state.to_dict(),
        warnings=tuple(dict.fromkeys(warnings)),
        scope=scope,
        trace_model=trace_model,
        session_metrics={
            "evaluated": True,
            "action_counts": session_action_counts,
            "warnings": ("session/KV model records sticky owner or recompute only",),
        },
        provenance={
            "runtime_engine": "trace_backed_runtime_surrogate",
            "capacity_allocation_assumption": "full_single_satellite_serving_capacity_per_member",
            "configured_time_horizon_s": time_horizon_s,
            "trace_horizon_s": trace_horizon_s,
            "effective_time_horizon_s": trace_horizon_s,
        },
        valid_for_paper_main_results=False,
    )




def normalize_runtime_request(raw_request: dict[str, Any]) -> dict[str, Any]:
    declared_prompt_tokens = raw_request.get("prompt_tokens")
    declared_new_user_tokens = raw_request.get("new_user_tokens")
    request_id = _optional_str(raw_request.get("request_id"))
    if request_id is None:
        raise ValueError("canonical Runtime request requires request_id")
    if declared_prompt_tokens is None or raw_request.get("output_tokens") is None:
        raise ValueError(
            "canonical Runtime request requires prompt_tokens and output_tokens"
        )
    prompt_tokens = max(0, _int(declared_prompt_tokens, 0))
    output_tokens = max(0, _int(raw_request.get("output_tokens"), 0))
    return {
        "request_id": request_id,
        "arrival_time_s": _float(raw_request.get("arrival_time_s"), 0.0),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "input_bytes": _float(raw_request.get("input_bytes"), prompt_tokens * 4.0),
        "output_bytes": _float(raw_request.get("output_bytes"), output_tokens * 4.0),
        "session_id": _optional_str(raw_request.get("session_id")),
        "ingress_id": _optional_str(raw_request.get("ingress_id")),
        "prompt_tokens_declared": declared_prompt_tokens is not None,
        "new_user_tokens": _optional_int(declared_new_user_tokens),
        "context_tokens_before_turn": _optional_int(raw_request.get("context_tokens_before_turn")),
        "context_window_tokens": _optional_int(raw_request.get("context_window_tokens")),
    }


def _evaluate_trace_request(
    *,
    state: RuntimeState,
    request: dict[str, Any],
    raw_request: dict[str, Any],
    trace_bundle: Any,
    deployment: ServingCapacityProfile,
    policy: str,
    scheduler: Any,
    runtime_strategy: Any,
    max_hop_count: int | None,
    scope: str,
    trace_model: str,
    trace_horizon_s: float | None,
) -> tuple[RuntimeRequestRecord, dict[str, Any]]:
    del raw_request
    ingress_id = request.get("ingress_id") or _default_ingress(trace_bundle)
    arrival_s = request["arrival_time_s"]
    access_wait_s = 0.0
    deferred = False
    defer_until_s = None
    if trace_horizon_s is None:
        record = _reject_record(
            request,
            "trace_horizon_missing",
            scope=scope,
            trace_model=trace_model,
            warnings=("trace_horizon_missing",),
        )
        return record, _decision_trace(request, policy, (), None, "trace_horizon_missing", warnings=("trace_horizon_missing",))
    if arrival_s < 0 or arrival_s >= trace_horizon_s:
        record = _reject_record(
            request,
            "request_time_outside_trace_horizon",
            scope=scope,
            trace_model=trace_model,
            provenance={"trace_horizon_s": trace_horizon_s},
        )
        return record, _decision_trace(request, policy, (), None, "request_time_outside_trace_horizon")
    if ingress_id is None:
        return _fallback_or_reject(request, runtime_strategy, "no_ingress", scope, trace_model), _decision_trace(request, policy, (), None, "no_ingress")
    access_query = query_access_at(trace_bundle, str(ingress_id), arrival_s)
    access_candidates = find_access_candidates(trace_bundle, str(ingress_id), arrival_s)
    if not access_candidates:
        if access_query.stale:
            reason = _query_reject_reason(access_query, "access")
            record = _reject_record(
                request,
                reason,
                scope=scope,
                trace_model=trace_model,
                warnings=access_query.warnings or (reason,),
                provenance={"access_query": _query_payload(access_query)},
            )
            return record, _decision_trace(request, policy, (), None, reason, warnings=access_query.warnings or (reason,))
        sgl_failure = _access_sgl_failure(trace_bundle, access_query, routing_time_s=arrival_s)
        if sgl_failure is not None:
            reason, failure_warnings, provenance = sgl_failure
            record = _fallback_or_reject(
                request,
                runtime_strategy,
                reason,
                scope,
                trace_model,
                warnings=failure_warnings,
                provenance=provenance,
            )
            return record, _decision_trace(request, policy, (), None, reason, warnings=failure_warnings)
        if bool(getattr(runtime_strategy, "defer_enabled", False)):
            next_access = find_next_access_candidate(trace_bundle, str(ingress_id), arrival_s)
            max_defer_s = _optional_float(_dict(getattr(runtime_strategy, "hard_gate_overrides", {})).get("max_defer_s"))
            if next_access is not None:
                wait_s = max(0.0, next_access.start_s - arrival_s)
                max_defer_ok = max_defer_s is None or wait_s <= max_defer_s
                if max_defer_ok:
                    access_candidates = (next_access,)
                    access_wait_s = wait_s
                    deferred = True
                    defer_until_s = next_access.start_s
        if not access_candidates:
            record = _fallback_or_reject(request, runtime_strategy, "no_access_window", scope, trace_model)
            return record, _decision_trace(request, policy, (), None, record.uncompleted_reason or "fallback")

    routing_time_s = arrival_s + access_wait_s
    eligibility = _compute_eligibility_at(
        deployment=deployment,
        trace_bundle=trace_bundle,
        state=state,
        time_s=routing_time_s,
    )
    if eligibility.empty or eligibility.capacity_member_count <= 0:
        exclusion_reasons = tuple(
            reason
            for reasons in eligibility.excluded_reasons.values()
            for reason in reasons
        )
        failure_reason = exclusion_reasons[0] if exclusion_reasons else "eligible_compute_set_empty"
        record = _fallback_or_reject(
            request,
            runtime_strategy,
            failure_reason,
            scope,
            trace_model,
            warnings=(failure_reason,),
            provenance={"compute_eligibility": eligibility.to_dict()},
        )
        return record, _decision_trace(
            request,
            policy,
            access_candidates,
            None,
            record.uncompleted_reason or failure_reason,
            warnings=(failure_reason,),
        )
    prefill_rate_per_sat = float(deployment.prefill_tokens_per_s or 0.0)
    decode_rate_per_sat = float(deployment.decode_tokens_per_s or 0.0)
    route_failures: list[str] = []
    route_warnings: list[str] = []
    route_rows = enumerate_compute_candidates(
        trace_bundle,
        access_candidates,
        policy,
        max_hop_count=max_hop_count,
        time_s=routing_time_s,
        failures=route_failures,
        warnings=route_warnings,
        target_satellite_ids=eligibility.eligible_compute_satellite_ids,
    )
    if not route_rows:
        reason = _first_stale_or_default(route_failures, "no_route_candidate")
        warnings = tuple(dict.fromkeys(route_warnings or route_failures))
        record = _fallback_or_reject(
            request,
            runtime_strategy,
            reason,
            scope,
            trace_model,
            warnings=warnings,
            provenance={"route_failures": tuple(route_failures)} if route_failures else None,
        )
        return record, _decision_trace(
            request,
            policy,
            access_candidates,
            None,
            record.uncompleted_reason or "fallback",
            warnings=warnings,
        )

    candidate_failures: list[str] = []
    candidate_warnings: list[str] = []
    selected = _select_candidate(
        state=state,
        trace_bundle=trace_bundle,
        route_rows=route_rows,
        request=request,
        scheduler=scheduler,
        runtime_strategy=runtime_strategy,
        prefill_rate_per_sat=prefill_rate_per_sat,
        decode_rate_per_sat=decode_rate_per_sat,
        access_wait_s=access_wait_s,
        arrival_s=arrival_s,
        candidate_failures=candidate_failures,
        candidate_warnings=candidate_warnings,
    )
    if selected is None:
        reason = _first_stale_or_default(candidate_failures, "candidate_prediction_failed")
        record = _fallback_or_reject(
            request,
            runtime_strategy,
            reason,
            scope,
            trace_model,
            warnings=tuple(dict.fromkeys(candidate_warnings or candidate_failures)),
        )
        return record, _decision_trace(
            request,
            policy,
            access_candidates,
            None,
            record.uncompleted_reason or reason or "fallback",
            warnings=tuple(dict.fromkeys(candidate_warnings or candidate_failures)),
        )

    if selected.get("scheduler_action") == "reject_drop":
        scheduler_decision = selected.get("scheduler_decision")
        provenance = {
            "scheduler_decision": scheduler_decision,
            "score_terms": selected.get("score_terms"),
            "selected_candidate_id": selected.get("selected_candidate_id"),
            "candidate_count": selected.get("candidate_count"),
            "feasible_candidate_count": selected.get("feasible_candidate_count"),
            "gate_results": selected.get("gate_results"),
            "rejected_candidates": selected.get("rejected_candidates"),
            "hard_gate_violations": selected.get("hard_gate_violations"),
            "hard_gate_results": selected.get("hard_gate_results"),
        }
        record = _fallback_or_reject(
            request,
            runtime_strategy,
            "no_feasible_scheduler_candidate",
            scope,
            trace_model,
            provenance=provenance,
        )
        return record, _decision_trace(
            request,
            policy,
            access_candidates,
            None,
            record.uncompleted_reason or "fallback",
            scheduler_decision=scheduler_decision,
        )

    route_plan = selected["route_plan"]
    satellite = state.satellite_for(route_plan.compute_satellite_id)
    try:
        inbound_done_s, inbound_path, sgl_wait_in, isl_wait_in, tx_in, sampled_inbound_legs = _reserve_route(
            state, trace_bundle, route_plan.inbound_legs, routing_time_s,
            request_id=request["request_id"], energy_stage="input_route",
        )
    except RouteSamplingError as exc:
        record = _fallback_or_reject(
            request,
            runtime_strategy,
            exc.reason,
            scope,
            trace_model,
            warnings=exc.warnings or (exc.reason,),
        )
        return record, _decision_trace(request, policy, access_candidates, route_plan, exc.reason, warnings=exc.warnings or (exc.reason,))
    route_plan = replace(route_plan, inbound_legs=sampled_inbound_legs)
    prefill_ready_s = max(inbound_done_s, satellite.compute_queue.prefill_available_s)
    try:
        advance_satellite_energy(satellite, trace_bundle, prefill_ready_s)
        raw_request_energy_window = energy_window_at(trace_bundle, satellite.satellite_id, prefill_ready_s)
    except EnergyWindowError as exc:
        warnings = _energy_error_warnings(exc)
        record = _fallback_or_reject(
            request,
            runtime_strategy,
            exc.reason,
            scope,
            trace_model,
            warnings=warnings,
            provenance={"energy_query": _energy_error_payload(exc)},
        )
        return record, _decision_trace(request, policy, access_candidates, route_plan, exc.reason, warnings=warnings)
    action_type = str(selected.get("action_type") or ("local_compute" if route_plan.hop_count == 0 else "cross_satellite_forward"))
    request_energy_window = _energy_window_for_runtime_action(
        raw_request_energy_window,
        runtime_strategy=runtime_strategy,
        action_type=action_type,
    )
    dvfs_scale = float(request_energy_window.get("dvfs_scale", 1.0))
    satellite.dvfs_scale = dvfs_scale
    prefill_service_s = _safe_service_time(request["prompt_tokens"], prefill_rate_per_sat) / dvfs_scale
    decode_service_s = _safe_service_time(request["output_tokens"], decode_rate_per_sat) / dvfs_scale
    if not math.isfinite(prefill_service_s) or not math.isfinite(decode_service_s):
        record = _reject_record(request, "non_finite_service_time", scope=scope, trace_model=trace_model)
        return record, _decision_trace(request, policy, access_candidates, route_plan, "non_finite_service_time")
    prefill_start_s, prefill_done_s, prefill_wait_s = satellite.compute_queue.reserve_prefill(inbound_done_s, prefill_service_s)
    decode_start_s, decode_done_s, decode_wait_s = satellite.compute_queue.reserve_decode(prefill_done_s, decode_service_s)
    first_token_service_s = min(
        decode_service_s,
        _safe_service_time(1, decode_rate_per_sat) / dvfs_scale,
    )
    first_token_ready_s = decode_start_s + first_token_service_s
    first_token_bytes = _first_token_bytes(request)
    remaining_output_bytes = max(0.0, request["output_bytes"] - first_token_bytes)
    first_plan, outbound_reason, outbound_warnings = _build_post_compute_outbound_plan(
        trace_bundle=trace_bundle,
        ingress_id=str(ingress_id),
        compute_satellite_id=route_plan.compute_satellite_id,
        policy=policy,
        max_hop_count=max_hop_count,
        output_bytes=first_token_bytes,
        decode_done_s=first_token_ready_s,
    )
    remaining_plan = None
    if first_plan is None:
        advance_satellite_energy(satellite, trace_bundle, prefill_start_s)
        energy_update = apply_request_energy(
            satellite,
            prompt_tokens=request["prompt_tokens"],
            output_tokens=request["output_tokens"],
            link_bytes=0.0,
            energy_window=request_energy_window,
            event_time_s=prefill_start_s,
            metadata={"request_id": request["request_id"], "energy_stage": "colocated_compute"},
        )
        energy_update = _energy_summary_for_events(
            state,
            trace_bundle,
            [energy_update["ledger_entry"], *_path_energy_entries(inbound_path)],
        )
        session_action = _session_action(
            state, request.get("session_id"), route_plan.compute_satellite_id, arrival_s,
            deployment=deployment, runtime_strategy=runtime_strategy, request=request,
        )
        record = _post_compute_route_failure_record(
            request=request,
            action_type=action_type,
            route_plan=route_plan,
            inbound_path=inbound_path,
            arrival_s=arrival_s,
            access_wait_s=access_wait_s,
            prefill_wait_s=prefill_wait_s,
            decode_wait_s=decode_wait_s,
            sgl_wait_in=sgl_wait_in,
            isl_wait_in=isl_wait_in,
            tx_in=tx_in,
            prefill_service_s=prefill_service_s,
            decode_service_s=decode_service_s,
            prefill_done_s=prefill_done_s,
            compute_satellite_id=route_plan.compute_satellite_id,
            access_satellite_id=route_plan.access_satellite_id,
            energy_update=energy_update,
            deferred=deferred,
            defer_until_s=defer_until_s,
            session_action=session_action,
            scope=scope,
            trace_model=trace_model,
            failure_reason=outbound_reason,
            failure_warnings=outbound_warnings,
            scheduler_decision=selected.get("scheduler_decision"),
            score_terms=selected.get("score_terms"),
            selected_candidate_id=selected.get("selected_candidate_id"),
            candidate_count=selected.get("candidate_count"),
            feasible_candidate_count=selected.get("feasible_candidate_count"),
        )
        return record, _decision_trace(
            request,
            policy,
            access_candidates,
            route_plan,
            "route_became_unavailable_after_compute",
            warnings=outbound_warnings or (outbound_reason,),
            scheduler_decision=selected.get("scheduler_decision"),
        )
    try:
        first_done_s, first_path, sgl_wait_first, isl_wait_first, tx_first, sampled_first_legs = _reserve_route(
            state, trace_bundle, first_plan.outbound_legs, first_token_ready_s,
            request_id=request["request_id"], energy_stage="first_token_route",
        )
    except RouteSamplingError as exc:
        advance_satellite_energy(satellite, trace_bundle, prefill_start_s)
        energy_update = apply_request_energy(
            satellite,
            prompt_tokens=request["prompt_tokens"],
            output_tokens=request["output_tokens"],
            link_bytes=0.0,
            energy_window=request_energy_window,
            event_time_s=prefill_start_s,
            metadata={"request_id": request["request_id"], "energy_stage": "colocated_compute"},
        )
        energy_update = _energy_summary_for_events(
            state,
            trace_bundle,
            [energy_update["ledger_entry"], *_path_energy_entries(inbound_path)],
        )
        session_action = _session_action(
            state, request.get("session_id"), route_plan.compute_satellite_id, arrival_s,
            deployment=deployment, runtime_strategy=runtime_strategy, request=request,
        )
        record = _post_compute_route_failure_record(
            request=request,
            action_type=action_type,
            route_plan=route_plan,
            inbound_path=inbound_path,
            arrival_s=arrival_s,
            access_wait_s=access_wait_s,
            prefill_wait_s=prefill_wait_s,
            decode_wait_s=decode_wait_s,
            sgl_wait_in=sgl_wait_in,
            isl_wait_in=isl_wait_in,
            tx_in=tx_in,
            prefill_service_s=prefill_service_s,
            decode_service_s=decode_service_s,
            prefill_done_s=prefill_done_s,
            compute_satellite_id=route_plan.compute_satellite_id,
            access_satellite_id=route_plan.access_satellite_id,
            energy_update=energy_update,
            deferred=deferred,
            defer_until_s=defer_until_s,
            session_action=session_action,
            scope=scope,
            trace_model=trace_model,
            failure_reason=exc.reason,
            failure_warnings=exc.warnings or (exc.reason,),
            scheduler_decision=selected.get("scheduler_decision"),
            score_terms=selected.get("score_terms"),
            selected_candidate_id=selected.get("selected_candidate_id"),
            candidate_count=selected.get("candidate_count"),
            feasible_candidate_count=selected.get("feasible_candidate_count"),
        )
        return record, _decision_trace(
            request,
            policy,
            access_candidates,
            route_plan,
            "route_became_unavailable_after_compute",
            warnings=exc.warnings or (exc.reason,),
            scheduler_decision=selected.get("scheduler_decision"),
        )
    remaining_done_s = first_done_s
    remaining_path: tuple[dict[str, Any], ...] = ()
    sgl_wait_remaining = 0.0
    isl_wait_remaining = 0.0
    tx_remaining = 0.0
    sampled_remaining_legs: tuple[Any, ...] = ()
    if remaining_output_bytes > 0:
        try:
            remaining_plan, outbound_reason, outbound_warnings = _build_post_compute_outbound_plan(
                trace_bundle=trace_bundle,
                ingress_id=str(ingress_id),
                compute_satellite_id=route_plan.compute_satellite_id,
                policy=policy,
                max_hop_count=max_hop_count,
                output_bytes=remaining_output_bytes,
                decode_done_s=decode_done_s,
            )
            if remaining_plan is None:
                raise RouteSamplingError(outbound_reason, warnings=outbound_warnings)
            remaining_done_s, remaining_path, sgl_wait_remaining, isl_wait_remaining, tx_remaining, sampled_remaining_legs = _reserve_route(
                state, trace_bundle, remaining_plan.outbound_legs, decode_done_s,
                request_id=request["request_id"], energy_stage="remaining_output_route",
            )
        except RouteSamplingError as exc:
            advance_satellite_energy(satellite, trace_bundle, prefill_start_s)
            energy_update = apply_request_energy(
                satellite,
                prompt_tokens=request["prompt_tokens"],
                output_tokens=request["output_tokens"],
                link_bytes=0.0,
                energy_window=request_energy_window,
                event_time_s=prefill_start_s,
                metadata={"request_id": request["request_id"], "energy_stage": "colocated_compute"},
            )
            energy_update = _energy_summary_for_events(
                state,
                trace_bundle,
                [
                    energy_update["ledger_entry"],
                    *_path_energy_entries(inbound_path, first_path),
                ],
            )
            session_action = _session_action(
            state, request.get("session_id"), route_plan.compute_satellite_id, arrival_s,
            deployment=deployment, runtime_strategy=runtime_strategy, request=request,
        )
            record = _post_compute_route_failure_record(
                request=request,
                action_type=action_type,
                route_plan=replace(route_plan, outbound_legs=sampled_first_legs),
                inbound_path=tuple((*inbound_path, *first_path)),
                arrival_s=arrival_s,
                access_wait_s=access_wait_s,
                prefill_wait_s=prefill_wait_s,
                decode_wait_s=decode_wait_s,
                sgl_wait_in=sgl_wait_in + sgl_wait_first,
                isl_wait_in=isl_wait_in + isl_wait_first,
                tx_in=tx_in + tx_first,
                prefill_service_s=prefill_service_s,
                decode_service_s=decode_service_s,
                prefill_done_s=prefill_done_s,
                compute_satellite_id=route_plan.compute_satellite_id,
                access_satellite_id=route_plan.access_satellite_id,
                energy_update=energy_update,
                deferred=deferred,
                defer_until_s=defer_until_s,
                session_action=session_action,
                scope=scope,
                trace_model=trace_model,
                failure_reason=exc.reason,
                failure_warnings=exc.warnings or (exc.reason,),
                scheduler_decision=selected.get("scheduler_decision"),
                score_terms=selected.get("score_terms"),
                selected_candidate_id=selected.get("selected_candidate_id"),
                candidate_count=selected.get("candidate_count"),
                feasible_candidate_count=selected.get("feasible_candidate_count"),
            )
            return record, _decision_trace(request, policy, access_candidates, route_plan, "route_became_unavailable_after_compute", warnings=exc.warnings or (exc.reason,), scheduler_decision=selected.get("scheduler_decision"))
    route_plan = replace(route_plan, outbound_legs=tuple((*sampled_first_legs, *sampled_remaining_legs)))
    advance_satellite_energy(satellite, trace_bundle, prefill_start_s)
    energy_update = apply_request_energy(
        satellite,
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        link_bytes=0.0,
        energy_window=request_energy_window,
        event_time_s=prefill_start_s,
        metadata={"request_id": request["request_id"], "energy_stage": "colocated_compute"},
    )
    energy_update = _energy_summary_for_events(
        state,
        trace_bundle,
        [
            energy_update["ledger_entry"],
            *_path_energy_entries(inbound_path, first_path, remaining_path),
        ],
    )
    uplink_s = max(0.0, inbound_done_s - (arrival_s + access_wait_s))
    first_token_return_s = max(0.0, first_done_s - first_token_ready_s)
    timing = ttft_breakdown(
        access_wait_s=access_wait_s,
        uplink_s=uplink_s,
        prefill_queue_s=prefill_wait_s,
        prefill_compute_s=prefill_service_s,
        kv_transfer_s=0.0,
        decode_queue_s=decode_wait_s,
        first_token_decode_s=first_token_service_s,
        return_path_s=first_token_return_s,
    )
    ttft_s = float(timing["ttft_s"])
    e2e_s = remaining_done_s - arrival_s
    session_action = _session_action(
            state, request.get("session_id"), route_plan.compute_satellite_id, arrival_s,
            deployment=deployment, runtime_strategy=runtime_strategy, request=request,
        )
    session = state.sessions.get(str(request.get("session_id"))) if request.get("session_id") else None
    kv_event = dict(getattr(session, "last_kv_event", {}) or {})
    total_sgl_wait = sgl_wait_in + sgl_wait_first + sgl_wait_remaining
    total_isl_wait = isl_wait_in + isl_wait_first + isl_wait_remaining
    bottleneck = _bottleneck(energy_update["energy_violation"], prefill_wait_s, decode_wait_s, total_sgl_wait + total_isl_wait)
    selected_path = tuple((*inbound_path, *first_path, *remaining_path))
    record = RuntimeRequestRecord(
        request_id=request["request_id"],
        session_id=request.get("session_id"),
        arrival_time_s=arrival_s,
        action_type=action_type,
        completed=True,
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        prefill_queue_wait_s=prefill_wait_s,
        decode_queue_wait_s=decode_wait_s,
        sgl_queue_wait_s=total_sgl_wait,
        isl_queue_wait_s=total_isl_wait,
        link_queue_wait_s=total_sgl_wait + total_isl_wait,
        access_wait_s=access_wait_s,
        link_transmission_s=tx_in + tx_first + tx_remaining,
        prefill_service_s=prefill_service_s,
        decode_service_s=decode_service_s,
        uplink_s=uplink_s,
        kv_transfer_s=0.0,
        first_token_decode_s=first_token_service_s,
        return_path_s=first_token_return_s,
        ttft_breakdown_s=timing,
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        serving_satellite_id=route_plan.compute_satellite_id,
        compute_owner_satellite_id=route_plan.compute_satellite_id,
        compute_owner_satellite_ids=(route_plan.compute_satellite_id,),
        access_satellite_id=route_plan.access_satellite_id,
        first_token_return_s=first_token_return_s,
        response_streaming_model="first_token_plus_bulk_remaining",
        selected_path=selected_path,
        route_node_ids=_route_node_ids(selected_path),
        soc_before=energy_update["soc_before"],
        soc_after=energy_update["soc_after"],
        energy_violation=energy_update["energy_violation"],
        energy_ledger=tuple(energy_update.get("energy_ledger", ())),
        compute_sunlit=_energy_window_sunlit(raw_request_energy_window),
        access_sunlit=selected.get("access_sunlit"),
        dvfs_scale=float(request_energy_window.get("dvfs_scale", 1.0)),
        shadow_strategy=selected.get("shadow_strategy"),
        fallback_used=False,
        deferred=deferred,
        defer_until_s=defer_until_s,
        session_action=session_action,
        kv_capacity_admission=kv_event.get("admission"),
        kv_serve_action=session_action,
        kv_hit_tokens=kv_event.get("hit_tokens"),
        kv_miss_tokens=kv_event.get("miss_tokens"),
        context_recompute_tokens=kv_event.get("recompute_tokens"),
        kv_evicted_bytes=kv_event.get("evicted_bytes"),
        kv_evicted_entry_count=len(tuple(kv_event.get("evicted_session_ids", ()) or ())),
        hbm_occupancy_before=kv_event.get("occupancy_bytes_before"),
        hbm_occupancy_after=kv_event.get("occupancy_bytes_after"),
        bottleneck_reason=bottleneck,
        scope=scope,
        trace_model=trace_model,
        valid_for_paper_main_results=False,
        provenance={
            "policy": policy,
            "route_plan": route_plan.to_dict(),
            "first_token_route_plan": first_plan.to_dict(),
            "remaining_route_plan": remaining_plan.to_dict() if remaining_plan is not None else None,
            "compute_eligibility": eligibility.to_dict(),
            "ttft_definition": TTFT_DEFINITION,
            "selected_candidate_id": selected.get("selected_candidate_id"),
            "selection_score": selected.get("selection_score"),
            "selection_weights": selected.get("selection_weights"),
            "selection_score_terms": selected.get("selection_score_terms"),
            "scheduler_decision": selected.get("scheduler_decision"),
            "score_terms": selected.get("score_terms"),
            "candidate_count": selected.get("candidate_count"),
            "feasible_candidate_count": selected.get("feasible_candidate_count"),
            "gate_results": selected.get("gate_results"),
            "rejected_candidates": selected.get("rejected_candidates"),
            "hard_gate_violations": selected.get("hard_gate_violations"),
            "hard_gate_results": selected.get("hard_gate_results"),
            "predicted_e2e_s": selected.get("predicted_e2e_s"),
            "predicted_ttft_s": selected.get("predicted_ttft_s"),
            "predicted_nominal_latency_s": selected.get("predicted_nominal_latency_s"),
            "predicted_queue_wait_s": selected.get("predicted_queue_wait_s"),
            "predicted_link_wait_s": selected.get("predicted_link_wait_s"),
            "predicted_soc_after": selected.get("predicted_soc_after"),
            "predicted_energy_risk": selected.get("predicted_energy_risk"),
            "access_sunlit": selected.get("access_sunlit"),
            "compute_sunlit": _energy_window_sunlit(raw_request_energy_window),
            "configured_shadow_dvfs_scale": runtime_strategy.shadow_dvfs_scale,
            "effective_dvfs_scale": request_energy_window.get("dvfs_scale"),
            "shadow_strategy": selected.get("shadow_strategy"),
            "shadow_triggered": _shadow_triggered(action_type, selected.get("access_energy_window"), raw_request_energy_window),
            "shadow_runtime_action": action_type,
            "capacity_allocation_assumption": "full_single_satellite_serving_capacity_per_member",
            "energy_model": "event_incremental_per_node_per_physical_leg_v1",
            "energy_ledger": tuple(energy_update.get("energy_ledger", ())),
            "preliminary_surrogate": True,
        },
    )
    return record, _decision_trace(
        request,
        policy,
        access_candidates,
        route_plan,
        bottleneck,
        scheduler_decision=selected.get("scheduler_decision"),
    )



def _compact_scheduler_decision(decision: Any) -> dict[str, Any]:
    selected_action = _dict(getattr(decision, "selected_action", {}) or {})
    return {
        "action": getattr(decision, "action", None),
        "evaluated": getattr(decision, "evaluated", None),
        "policy_name": getattr(decision, "policy_name", None),
        "rationale": getattr(decision, "rationale", None),
        "selected_candidate_id": getattr(decision, "selected_candidate_id", None),
        "selected_action": {
            key: to_jsonable(selected_action[key])
            for key in (
                "candidate_id",
                "action_type",
                "route_plan",
                "hard_gate_violations",
                "hard_gate_results",
                "score",
                "score_terms",
            )
            if key in selected_action
        },
        "score": getattr(decision, "score", None),
        "score_terms": to_jsonable(getattr(decision, "score_terms", {}) or {}),
        "candidate_count": getattr(decision, "candidate_count", None),
        "feasible_candidate_count": getattr(decision, "feasible_candidate_count", None),
        "rejected_candidates": to_jsonable(getattr(decision, "rejected_candidates", ()) or ()),
        "gate_results": to_jsonable(getattr(decision, "gate_results", {}) or {}),
    }



def _satellite_route_legs(
    trace_bundle: Any,
    source_satellite_id: str,
    target_satellite_id: str,
    payload_bytes: float,
    *,
    time_s: float,
    max_hop_count: int | None,
) -> tuple[RouteLeg, ...]:
    if source_satellite_id == target_satellite_id:
        return ()
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
    links = _active_link_rows(trace_bundle, active.value)
    hop_limit = max_hop_count if max_hop_count is not None else 2
    link_ids = _build_ring_path_from_graph(
        _ring_graph_from_active_links(trace_bundle, active, links),
        source_satellite_id,
        target_satellite_id,
        max_hop_count=hop_limit,
    )
    if link_ids is None:
        raise RouteSamplingError(
            "kv_route_unavailable",
            sample_time_s=active.sample_time_s,
            warnings=active.warnings or ("kv_route_unavailable",),
        )
    by_id = _runtime_links_by_id(trace_bundle)
    legs: list[RouteLeg] = []
    current_node = str(source_satellite_id)
    for link_id in link_ids:
        link = by_id.get(str(link_id))
        if link is None:
            raise RouteSamplingError("link_missing", sample_time_s=active.sample_time_s)
        endpoint_a = str(link.get("endpoint_a"))
        endpoint_b = str(link.get("endpoint_b"))
        if current_node == endpoint_a:
            next_node = endpoint_b
        elif current_node == endpoint_b:
            next_node = endpoint_a
        else:
            raise RouteSamplingError(
                "kv_route_direction_mismatch",
                sample_time_s=active.sample_time_s,
            )
        legs.append(
            RouteLeg(
                link_id=str(link_id),
                kind=_link_runtime_kind(link),
                endpoint_a=str(link.get("endpoint_a")),
                endpoint_b=str(link.get("endpoint_b")),
                payload_bytes=payload_bytes,
                source_node_id=current_node,
                target_node_id=next_node,
                propagation_delay_s=float(_first_present(link.get("propagation_delay_s"), link.get("tau_prop_s"), 0.0)),
                bandwidth_bps=float(
                    _first_present(
                        link.get("bandwidth_bps"),
                        link.get("bandwidth_bps_nominal"),
                        link.get("B_raw_bps"),
                        0.0,
                    )
                ),
                setup_delay_s=float(_first_present(link.get("setup_delay_s"), link.get("tau_setup_s"), 0.0)),
                active_graph_time_s=active.sample_time_s,
                edge_available_external=bool(link.get("edge_available_external", True)),
                warnings=active.warnings,
            )
        )
        current_node = next_node
    if current_node != str(target_satellite_id):
        raise RouteSamplingError(
            "kv_route_direction_mismatch",
            sample_time_s=active.sample_time_s,
        )
    return tuple(legs)


def _active_link_rows(trace_bundle: Any, value: Any) -> tuple[dict[str, Any], ...]:
    by_id = _runtime_links_by_id(trace_bundle)
    if value is None:
        return tuple(by_id.values())
    if isinstance(value, dict):
        if value.get("links") is not None:
            return tuple(_dict(item) for item in value.get("links") or ())
        ids = tuple(str(item) for item in value.get("active_link_ids", ()) or ())
        return tuple(by_id[item] for item in ids if item in by_id)
    if isinstance(value, (tuple, list)):
        rows: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, str):
                if item in by_id:
                    rows.append(by_id[item])
            else:
                rows.append(_dict(item))
        return tuple(rows)
    return tuple(by_id.values())


def _ring_graph_from_active_links(
    trace_bundle: Any,
    active: Any,
    links: tuple[dict[str, Any], ...],
) -> dict[str, tuple[tuple[float, str, str], ...]]:
    cache = getattr(trace_bundle, "__dict__", {}).get("_ring_graph_cache")
    if not isinstance(cache, dict):
        cache = {}
        try:
            object.__setattr__(trace_bundle, "_ring_graph_cache", cache)
        except Exception:
            pass
    value = _dict(getattr(active, "value", None))
    snapshot = _dict(value.get("snapshot"))
    if snapshot.get("activation_policy") == "static_compat_links":
        key = ("static_compat_links", len(links))
    else:
        active_ids = tuple(str(item) for item in value.get("active_link_ids", ()) or ())
        key = ("active", getattr(active, "sample_time_s", None), active_ids or len(links))
    graph = cache.get(key)
    if isinstance(graph, dict):
        return graph
    graph = _ring_graph_from_links(links)
    cache[key] = graph
    return graph


def _runtime_links_by_id(trace_bundle: Any) -> dict[str, dict[str, Any]]:
    cached = getattr(trace_bundle, "__dict__", {}).get("_runtime_links_by_id_cache")
    if isinstance(cached, dict):
        return cached
    rows: list[dict[str, Any]] = []
    for attr in ("physical_links", "links"):
        rows.extend(_dict(item) for item in getattr(trace_bundle, attr, ()) or ())
    link_trace = _dict(getattr(trace_bundle, "link_trace", None))
    for key in ("physical_links", "links"):
        rows.extend(_dict(item) for item in link_trace.get(key, ()) or ())
    by_id = {
        str(row.get("link_id")): row
        for row in rows
        if row.get("link_id") is not None
    }
    try:
        object.__setattr__(trace_bundle, "_runtime_links_by_id_cache", by_id)
    except Exception:
        pass
    return by_id


def _link_runtime_kind(link: dict[str, Any]) -> str:
    kind = str(_first_present(link.get("runtime_kind"), link.get("kind"), ""))
    return "isl" if kind.startswith("isl") else kind


def _first_token_bytes(request: dict[str, Any]) -> float:
    output_tokens = max(1, int(request.get("output_tokens") or 1))
    return max(0.0, float(request.get("output_bytes", 0.0))) / output_tokens



def _stale_reason(kind: str, reason: str | None) -> str:
    return reason if reason and reason.startswith("stale_trace_sample") else f"stale_trace_sample:{kind}"


def _select_candidate(
    *,
    state: RuntimeState,
    trace_bundle: Any,
    route_rows: tuple[Any, ...],
    request: dict[str, Any],
    scheduler: Any,
    runtime_strategy: Any,
    prefill_rate_per_sat: float,
    decode_rate_per_sat: float,
    access_wait_s: float,
    arrival_s: float,
    candidate_failures: list[str],
    candidate_warnings: list[str],
) -> dict[str, Any] | None:
    candidates, payload_by_id = _build_scheduler_candidates(
        state=state,
        trace_bundle=trace_bundle,
        route_rows=route_rows,
        request=request,
        prefill_rate_per_sat=prefill_rate_per_sat,
        decode_rate_per_sat=decode_rate_per_sat,
        runtime_strategy=runtime_strategy,
        access_wait_s=access_wait_s,
        arrival_s=arrival_s,
        candidate_failures=candidate_failures,
        candidate_warnings=candidate_warnings,
    )
    if not candidates:
        return None
    decision = scheduler.select_action(
        state={
            "runtime_strategy": _jsonable_strategy(runtime_strategy),
            "candidates": tuple(candidate.to_dict() for candidate in candidates),
        },
        request=request,
    )
    selected_id = getattr(decision, "selected_candidate_id", None)
    if not getattr(decision, "evaluated", False) or selected_id not in payload_by_id:
        candidate_failures.append("scheduler_candidate_selection_failed")
        return None
    selected = dict(payload_by_id[str(selected_id)])
    score_terms = _dict(getattr(decision, "score_terms", {}) or {})
    selected.update(
        {
            "selected_candidate_id": selected_id,
            "selection_score": getattr(decision, "score", None),
            "selection_score_terms": score_terms,
            "selection_weights": _dict(score_terms.get("weights")),
            "scheduler_decision": _compact_scheduler_decision(decision),
            "score_terms": score_terms,
            "candidate_count": getattr(decision, "candidate_count", None),
            "feasible_candidate_count": getattr(decision, "feasible_candidate_count", None),
            "gate_results": _dict(getattr(decision, "gate_results", {}) or {}),
            "rejected_candidates": tuple(getattr(decision, "rejected_candidates", ()) or ()),
            "scheduler_action": getattr(decision, "action", None),
        }
    )
    return selected


def _build_scheduler_candidates(
    *,
    state: RuntimeState,
    trace_bundle: Any,
    route_rows: tuple[Any, ...],
    request: dict[str, Any],
    prefill_rate_per_sat: float,
    decode_rate_per_sat: float,
    runtime_strategy: Any,
    access_wait_s: float,
    arrival_s: float,
    candidate_failures: list[str],
    candidate_warnings: list[str],
) -> tuple[tuple[SchedulerCandidate, ...], dict[str, dict[str, Any]]]:
    candidates: list[SchedulerCandidate] = []
    payload_by_id: dict[str, dict[str, Any]] = {}
    for index, (access_candidate, compute_satellite_id, isl_link_ids) in enumerate(route_rows):
        try:
            route_plan = make_route_plan(
                trace_bundle=trace_bundle,
                access_candidate=access_candidate,
                compute_satellite_id=compute_satellite_id,
                isl_link_ids=isl_link_ids,
                input_bytes=request["input_bytes"],
                output_bytes=request["output_bytes"],
                time_s=arrival_s + access_wait_s,
            )
        except RouteSamplingError as exc:
            candidate_failures.append(exc.reason)
            candidate_warnings.extend(exc.warnings or (exc.reason,))
            continue
        except ValueError as exc:
            candidate_failures.append(str(exc))
            continue
        try:
            inbound_done, link_wait, _tx, sampled_inbound_legs = _preview_route(
                state, trace_bundle, route_plan.inbound_legs, arrival_s + access_wait_s
            )
        except RouteSamplingError as exc:
            candidate_failures.append(exc.reason)
            candidate_warnings.extend(exc.warnings or (exc.reason,))
            continue
        route_plan = replace(route_plan, inbound_legs=sampled_inbound_legs)
        satellite = state.satellite_for(route_plan.compute_satellite_id)
        prefill_ready_s = max(inbound_done, satellite.compute_queue.prefill_available_s)
        access_energy_window = None
        try:
            access_energy_window = energy_window_at(
                trace_bundle,
                route_plan.access_satellite_id,
                arrival_s + access_wait_s,
            )
        except EnergyWindowError:
            access_energy_window = None
        try:
            raw_request_energy_window = energy_window_at(trace_bundle, route_plan.compute_satellite_id, prefill_ready_s)
        except EnergyWindowError as exc:
            candidate_failures.append(exc.reason)
            candidate_warnings.extend(_energy_error_warnings(exc))
            continue
        action_type = _shadow_candidate_action_type(
            route_plan=route_plan,
            access_energy_window=access_energy_window,
            compute_energy_window=raw_request_energy_window,
            runtime_strategy=runtime_strategy,
        )
        if not _runtime_action_allowed(action_type, route_plan=route_plan, runtime_strategy=runtime_strategy):
            continue
        request_energy_window = _energy_window_for_runtime_action(
            raw_request_energy_window,
            runtime_strategy=runtime_strategy,
            action_type=action_type,
        )
        dvfs_scale = float(request_energy_window.get("dvfs_scale", 1.0))
        prefill_service_s = _safe_service_time(request["prompt_tokens"], prefill_rate_per_sat) / dvfs_scale
        decode_service_s = _safe_service_time(request["output_tokens"], decode_rate_per_sat) / dvfs_scale
        if not math.isfinite(prefill_service_s) or not math.isfinite(decode_service_s):
            continue
        prefill_start, prefill_done, prefill_wait = satellite.compute_queue.preview_prefill(inbound_done, prefill_service_s)
        decode_start, decode_done, decode_wait = satellite.compute_queue.preview_decode(prefill_done, decode_service_s)
        first_token_service_s = min(
            decode_service_s,
            _safe_service_time(1, decode_rate_per_sat) / dvfs_scale,
        )
        first_token_route_plan = replace(
            route_plan,
            inbound_legs=(),
            outbound_legs=tuple(
                replace(leg, payload_bytes=_first_token_bytes(request))
                for leg in route_plan.outbound_legs
            ),
        )
        estimated_outbound_s = _estimate_route_duration_s(route_plan.outbound_legs)
        try:
            preview_state = _clone_runtime_state_for_preview(state)
            inbound_energy_states, inbound_energy_by_leg = _preview_route_energy_updates(
                preview_state,
                trace_bundle,
                sampled_inbound_legs,
                arrival_s + access_wait_s,
                request_id=f"candidate-preview-{index}",
                energy_stage="input_route",
            )
            _commit_energy_preview_states(preview_state, inbound_energy_states)
            preview_compute = preview_state.satellite_for(route_plan.compute_satellite_id)
            preview_prefill_start = max(
                inbound_done,
                preview_compute.compute_queue.prefill_available_s,
            )
            advance_satellite_energy(preview_compute, trace_bundle, preview_prefill_start)
            preview_compute_update = apply_request_energy(
                preview_compute,
                prompt_tokens=request["prompt_tokens"],
                output_tokens=request["output_tokens"],
                link_bytes=0.0,
                energy_window=request_energy_window,
                event_time_s=preview_prefill_start,
                metadata={"request_id": f"candidate-preview-{index}", "energy_stage": "colocated_compute"},
            )
            outbound_energy_states, outbound_energy_by_leg = _preview_route_energy_updates(
                preview_state,
                trace_bundle,
                route_plan.outbound_legs,
                decode_done,
                request_id=f"candidate-preview-{index}",
                energy_stage="output_route",
            )
            _commit_energy_preview_states(preview_state, outbound_energy_states)
            preview_energy = _energy_summary_for_events(
                preview_state,
                trace_bundle,
                [
                    preview_compute_update["ledger_entry"],
                    *(
                        row
                        for per_leg in (inbound_energy_by_leg, outbound_energy_by_leg)
                        for rows in per_leg.values()
                        for row in rows
                    ),
                ],
            )
        except (EnergyWindowError, RouteSamplingError) as exc:
            reason = getattr(exc, "reason", "candidate_energy_preview_failed")
            candidate_failures.append(str(reason))
            candidate_warnings.extend(tuple(getattr(exc, "warnings", ()) or (str(reason),)))
            continue
        predicted_soc_after = float(preview_energy["soc_after"])
        energy_rows = tuple(preview_energy.get("energy_ledger", ()))
        request_energy_rows = [row for row in energy_rows if row.get("energy_kind") == "request_event"]
        min_energy_slack = min(
            (
                float(row.get("soc_after", 0.0)) - float(row.get("soc_min", 0.0))
                for row in request_energy_rows
            ),
            default=0.0,
        )
        predicted_energy_risk = max(0.0, -min_energy_slack)
        soc_min = 0.0
        predicted_e2e = (decode_done + estimated_outbound_s) - arrival_s
        predicted_ttft = (
            decode_start
            + first_token_service_s
            + _estimate_route_duration_s(first_token_route_plan.outbound_legs)
            - arrival_s
        )
        predicted_queue_wait = prefill_wait + decode_wait
        predicted_link_wait = link_wait
        predicted_nominal_latency_s = (
            _estimate_route_duration_s(route_plan.inbound_legs)
            + prefill_service_s
            + decode_service_s
            + estimated_outbound_s
        )
        candidate_id = f"candidate-{index}:{route_plan.path_id}"
        hard_gate_results = _candidate_hard_gate_results(
            request=request,
            predicted_ttft_s=predicted_ttft,
            predicted_e2e_s=predicted_e2e,
            predicted_soc_after=predicted_soc_after,
            soc_min=soc_min,
        )
        hard_gate_results["energy_soc"] = {
            "passed": not bool(preview_energy["energy_violation"]),
            "reason": "energy_soc" if preview_energy["energy_violation"] else None,
            "predicted_soc_after": predicted_soc_after,
            "soc_min": None,
            "soc_slack": min_energy_slack,
            "per_node_ledger": request_energy_rows,
        }
        hard_gate_violations = _candidate_hard_gate_violations(hard_gate_results)
        candidate = SchedulerCandidate(
            candidate_id=candidate_id,
            action_type=action_type,
            feasible=True,
            route_plan=route_plan.to_dict(),
            predicted_ttft_s=predicted_ttft,
            predicted_e2e_s=predicted_e2e,
            predicted_queue_wait_s=predicted_queue_wait,
            predicted_link_wait_s=predicted_link_wait,
            predicted_energy_risk=predicted_energy_risk,
            predicted_soc_after=predicted_soc_after,
            hop_count=route_plan.hop_count,
            hard_gate_violations=hard_gate_violations,
            hard_gate_results=hard_gate_results,
            predicted_nominal_latency_s=predicted_nominal_latency_s,
            provenance={
                "access_satellite_id": route_plan.access_satellite_id,
                "compute_satellite_id": route_plan.compute_satellite_id,
                "compute_owner_satellite_id": route_plan.compute_satellite_id,
                "path_id": route_plan.path_id,
                "access_sunlit": _energy_window_sunlit(access_energy_window),
                "compute_sunlit": _energy_window_sunlit(raw_request_energy_window),
                "configured_shadow_dvfs_scale": runtime_strategy.shadow_dvfs_scale,
                "effective_dvfs_scale": request_energy_window.get("dvfs_scale"),
                "shadow_strategy": _shadow_strategy_label(action_type, raw_request_energy_window),
                "shadow_triggered": _shadow_triggered(action_type, access_energy_window, raw_request_energy_window),
            },
        )
        payload = {
            "candidate_id": candidate_id,
            "action_type": action_type,
            "route_plan": route_plan,
            "first_token_route_plan": first_token_route_plan,
            "hard_gate_violations": hard_gate_violations,
            "hard_gate_results": hard_gate_results,
            "predicted_ttft_s": predicted_ttft,
            "predicted_e2e_s": predicted_e2e,
            "predicted_nominal_latency_s": predicted_nominal_latency_s,
            "predicted_queue_wait_s": predicted_queue_wait,
            "predicted_link_wait_s": predicted_link_wait,
            "predicted_soc_after": predicted_soc_after,
            "predicted_energy_risk": predicted_energy_risk,
            "request_energy_window": request_energy_window,
            "raw_request_energy_window": raw_request_energy_window,
            "access_energy_window": access_energy_window,
            "access_sunlit": _energy_window_sunlit(access_energy_window),
            "compute_sunlit": _energy_window_sunlit(raw_request_energy_window),
            "dvfs_scale": request_energy_window.get("dvfs_scale"),
            "shadow_strategy": _shadow_strategy_label(action_type, raw_request_energy_window),
            "scheduler_candidate": candidate.to_dict(),
        }
        candidates.append(candidate)
        payload_by_id[candidate_id] = payload
    return tuple(candidates), payload_by_id


def _shadow_candidate_action_type(
    *,
    route_plan: Any,
    access_energy_window: dict[str, Any] | None,
    compute_energy_window: dict[str, Any],
    runtime_strategy: Any,
) -> str:
    compute_sunlit = _energy_window_sunlit(compute_energy_window)
    access_sunlit = _energy_window_sunlit(access_energy_window)
    if int(getattr(route_plan, "hop_count", 0) or 0) == 0:
        if compute_sunlit is False and _runtime_dvfs_enabled(runtime_strategy, "shadow_dvfs"):
            return "shadow_dvfs"
        if compute_sunlit is False:
            return "shadow_local_compute"
        return "local_compute"
    if access_sunlit is False and compute_sunlit is not False:
        return "shadow_offload"
    return "cross_satellite_forward"


def _runtime_dvfs_enabled(runtime_strategy: Any, mode: str) -> bool:
    del mode
    return bool(getattr(runtime_strategy, "shadow_dvfs_enabled", False))



def _runtime_action_allowed(action_type: str, *, route_plan: Any, runtime_strategy: Any) -> bool:
    enabled = {str(item) for item in tuple(getattr(runtime_strategy, "enabled_actions", ()) or ())}
    if not enabled:
        return True
    aliases = {action_type}
    if action_type in {"shadow_dvfs", "shadow_local_compute"}:
        aliases.add("local_compute")
    if action_type == "shadow_offload":
        aliases.update(("cross_satellite_forward", "same_plane_forward", "cross_plane_forward"))
    if action_type == "cross_satellite_forward":
        aliases.update(("same_plane_forward", "cross_plane_forward"))
    if int(getattr(route_plan, "hop_count", 0) or 0) == 0:
        aliases.add("local_compute")
    return bool(enabled.intersection(aliases))


def _energy_window_for_runtime_action(
    energy_window: dict[str, Any],
    *,
    runtime_strategy: Any,
    action_type: str,
) -> dict[str, Any]:
    adjusted = dict(energy_window)
    adjusted["base_compute_energy_j_per_token"] = float(
        energy_window["compute_energy_j_per_token"]
    )
    adjusted["base_link_energy_j_per_byte"] = float(
        energy_window["link_energy_j_per_byte"]
    )
    adjusted["runtime_compute_energy_scale"] = 1.0
    adjusted["runtime_link_energy_scale"] = 1.0
    adjusted["dvfs_scale"] = 1.0
    if _energy_window_sunlit(energy_window) is not False:
        return adjusted
    if action_type == "shadow_dvfs":
        effective_scale = float(getattr(runtime_strategy, "shadow_dvfs_scale", 0.5))
        if not 0.0 < effective_scale <= 1.0:
            raise ValueError("shadow_dvfs_scale must be in (0, 1]")
        adjusted["dvfs_scale"] = effective_scale
        adjusted["shadow_strategy"] = "shadow_dvfs"
        # Compute power and capacity scale together.  Their effects cancel in
        # the per-token dynamic-energy coefficient, so that coefficient must
        # not be scaled a second time here.
        return adjusted
    adjusted["shadow_strategy"] = "shadow_local_compute" if action_type == "shadow_local_compute" else action_type
    return adjusted


def _energy_window_sunlit(energy_window: dict[str, Any] | None) -> bool | None:
    if not energy_window:
        return None
    if "sunlit" not in energy_window:
        return None
    return bool(energy_window.get("sunlit"))


def _shadow_strategy_label(action_type: str, compute_energy_window: dict[str, Any]) -> str | None:
    if _energy_window_sunlit(compute_energy_window) is not False and action_type != "shadow_offload":
        return None
    if action_type in {"shadow_dvfs", "shadow_offload", "shadow_local_compute"}:
        return action_type
    return "shadow_affected"


def _shadow_triggered(
    action_type: str,
    access_energy_window: dict[str, Any] | None,
    compute_energy_window: dict[str, Any],
) -> bool:
    return (
        action_type in {"shadow_dvfs", "shadow_offload", "shadow_local_compute"}
        or _energy_window_sunlit(access_energy_window) is False
        or _energy_window_sunlit(compute_energy_window) is False
    )


def _candidate_hard_gate_results(
    *,
    request: dict[str, Any],
    predicted_ttft_s: float,
    predicted_e2e_s: float,
    predicted_soc_after: float,
    soc_min: float,
) -> dict[str, Any]:
    del request, predicted_ttft_s, predicted_e2e_s
    energy_miss = (
        math.isfinite(predicted_soc_after)
        and math.isfinite(soc_min)
        and predicted_soc_after < soc_min
    )
    return {
        "energy_soc": {
            "passed": not energy_miss,
            "reason": "energy_soc" if energy_miss else None,
            "predicted_soc_after": predicted_soc_after,
            "soc_min": soc_min,
            "soc_slack": predicted_soc_after - soc_min,
        },
    }


def _candidate_hard_gate_violations(hard_gate_results: dict[str, Any]) -> tuple[str, ...]:
    violations: list[str] = []
    for result in hard_gate_results.values():
        payload = _dict(result)
        if payload.get("passed") is False:
            reason = _optional_str(payload.get("reason"))
            if reason:
                violations.append(reason)
    return tuple(violations)


def _preview_route(
    state: RuntimeState,
    trace_bundle: Any,
    legs: tuple[Any, ...],
    start_s: float,
) -> tuple[float, float, float, tuple[Any, ...]]:
    current_s = start_s
    total_wait_s = 0.0
    total_tx_s = 0.0
    local_available: dict[str, float] = {}
    sampled_legs: list[Any] = []
    for leg in legs:
        link = state.link_for(leg.link_id)
        available_s = local_available.get(leg.link_id, link.available_s)
        sampled_leg = sample_route_leg(trace_bundle, leg, current_s)
        bandwidth_bps = float(sampled_leg.bandwidth_bps or 0.0)
        propagation_delay_s = float(sampled_leg.propagation_delay_s or 0.0)
        setup_delay_s = float(sampled_leg.setup_delay_s or 0.0)
        if bandwidth_bps <= 0:
            transmission_s = float("inf")
        else:
            transmission_s = max(0.0, leg.payload_bytes) * 8.0 / bandwidth_bps
        begin_s = max(current_s, available_s)
        wait_s = max(0.0, begin_s - current_s)
        finish_s = begin_s + setup_delay_s + transmission_s
        current_s = finish_s + propagation_delay_s
        local_available[leg.link_id] = finish_s
        total_wait_s += wait_s
        total_tx_s += transmission_s
        sampled_legs.append(
            replace(
                sampled_leg,
                queue_wait_s=wait_s,
                transmission_s=transmission_s,
                propagation_delay_s=propagation_delay_s,
                setup_delay_s=setup_delay_s,
            )
        )
    return current_s, total_wait_s, total_tx_s, tuple(sampled_legs)


def _clone_runtime_state_for_preview(state: RuntimeState) -> RuntimeState:
    """Clone only mutable scheduling/energy state needed by a candidate preview."""

    return RuntimeState(
        satellites={
            key: replace(satellite, compute_queue=replace(satellite.compute_queue))
            for key, satellite in state.satellites.items()
        },
        links={key: replace(link) for key, link in state.links.items()},
        sessions={},
        kv_caches={},
    )


def _reserve_route(
    state: RuntimeState,
    trace_bundle: Any,
    legs: tuple[Any, ...],
    start_s: float,
    *,
    request_id: str | None = None,
    energy_stage: str = "route",
) -> tuple[float, tuple[dict[str, Any], ...], float, float, float, tuple[Any, ...]]:
    _, _, _, sampled_legs = _preview_route(state, trace_bundle, legs, start_s)
    return _reserve_sampled_route(
        state,
        trace_bundle,
        sampled_legs,
        start_s,
        request_id=request_id,
        energy_stage=energy_stage,
    )


def _reserve_sampled_route(
    state: RuntimeState,
    trace_bundle: Any,
    sampled_legs: tuple[Any, ...],
    start_s: float,
    *,
    request_id: str | None = None,
    energy_stage: str = "route",
) -> tuple[float, tuple[dict[str, Any], ...], float, float, float, tuple[Any, ...]]:
    """Reserve already sampled legs without querying mutable trace sources again."""

    try:
        energy_states, route_energy_rows = _preview_route_energy_updates(
            state,
            trace_bundle,
            sampled_legs,
            start_s,
            request_id=request_id,
            energy_stage=energy_stage,
        )
    except EnergyWindowError as exc:
        raise RouteSamplingError(
            exc.reason,
            sample_time_s=exc.sample_time_s or exc.time_s,
            warnings=exc.warnings or (exc.reason,),
        ) from exc
    current_s = start_s
    path: list[dict[str, Any]] = []
    sgl_wait_s = 0.0
    isl_wait_s = 0.0
    tx_s = 0.0
    for leg_index, leg in enumerate(sampled_legs):
        link = state.link_for(leg.link_id)
        result = link.reserve(
            current_s,
            leg.payload_bytes,
            bandwidth_bps=float(leg.bandwidth_bps or 0.0),
            propagation_delay_s=float(leg.propagation_delay_s or 0.0),
            setup_delay_s=float(leg.setup_delay_s or 0.0),
        )
        current_s = result["arrival_s"]
        wait_s = result["queue_wait_s"]
        if leg.kind == "sgl":
            sgl_wait_s += wait_s
        else:
            isl_wait_s += wait_s
        tx_s += result["transmission_s"]
        leg_payload = leg.to_dict()
        leg_payload.update(
            {
                "queue_wait_s": wait_s,
                "transmission_s": result["transmission_s"],
                "setup_delay_s": result["setup_delay_s"],
                "propagation_delay_s": result["propagation_delay_s"],
                "arrival_s": result["arrival_s"],
                "energy_ledger": tuple(route_energy_rows.get(leg_index, ())),
            }
        )
        path.append(leg_payload)
    _commit_energy_preview_states(state, energy_states)
    return current_s, tuple(path), sgl_wait_s, isl_wait_s, tx_s, sampled_legs


def _preview_route_energy_updates(
    state: RuntimeState,
    trace_bundle: Any,
    sampled_legs: tuple[Any, ...],
    start_s: float,
    *,
    request_id: str | None,
    energy_stage: str,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    """Preview per-physical-leg endpoint energy without mutating runtime state."""

    satellite_ids = {
        str(row.get("satellite_id"))
        for row in tuple(getattr(trace_bundle, "satellite_catalog", ()) or ())
    }
    previews: dict[str, Any] = {}
    rows: dict[int, list[dict[str, Any]]] = {}
    local_available: dict[str, float] = {}
    current_s = float(start_s)

    def preview_for(satellite_id: str) -> Any:
        if satellite_id not in previews:
            previews[satellite_id] = replace(state.satellite_for(satellite_id))
        return previews[satellite_id]

    for leg_index, leg in enumerate(sampled_legs):
        link = state.link_for(leg.link_id)
        available_s = local_available.get(leg.link_id, link.available_s)
        bandwidth_bps = float(getattr(leg, "bandwidth_bps", None) or 0.0)
        transmission_s = (
            float(leg.transmission_s)
            if getattr(leg, "transmission_s", None) is not None
            else (
                max(0.0, float(leg.payload_bytes)) * 8.0 / bandwidth_bps
                if bandwidth_bps > 0
                else math.inf
            )
        )
        setup_delay_s = float(leg.setup_delay_s or 0.0)
        propagation_delay_s = float(leg.propagation_delay_s or 0.0)
        begin_s = max(current_s, available_s)
        finish_transmit_s = begin_s + setup_delay_s + transmission_s
        arrival_s = finish_transmit_s + propagation_delay_s
        local_available[leg.link_id] = finish_transmit_s
        current_s = arrival_s
        source = str(getattr(leg, "source_node_id", None) or leg.endpoint_a)
        target = str(getattr(leg, "target_node_id", None) or leg.endpoint_b)
        endpoint_bytes = float(leg.payload_bytes)
        if str(leg.kind) != "sgl":
            endpoint_bytes *= 0.5
        endpoint_events = (
            (source, finish_transmit_s, "link_tx"),
            (target, arrival_s, "link_rx"),
        )
        for satellite_id, event_time_s, endpoint_role in endpoint_events:
            if satellite_id not in satellite_ids:
                continue
            preview = preview_for(satellite_id)
            advance_satellite_energy(preview, trace_bundle, event_time_s)
            window = energy_window_at(trace_bundle, satellite_id, event_time_s)
            update = apply_request_energy(
                preview,
                prompt_tokens=0,
                output_tokens=0,
                link_bytes=endpoint_bytes,
                energy_window=window,
                event_time_s=event_time_s,
                metadata={
                    "request_id": request_id,
                    "energy_stage": energy_stage,
                    "endpoint_role": endpoint_role,
                    "link_id": str(leg.link_id),
                    "link_kind": str(leg.kind),
                    "payload_bytes": float(leg.payload_bytes),
                    "link_energy_semantics": "per_physical_link_total_split_across_isl_endpoints",
                },
            )
            rows.setdefault(leg_index, []).append(update["ledger_entry"])
    return previews, rows


def _commit_energy_preview_states(state: RuntimeState, previews: dict[str, Any]) -> None:
    for satellite_id, preview in previews.items():
        target = state.satellite_for(satellite_id)
        target.soc = preview.soc
        target.initial_soc = preview.initial_soc
        target.last_energy_update_s = preview.last_energy_update_s
        target.sunlit = preview.sunlit
        target.dvfs_scale = preview.dvfs_scale
        target.energy_events = preview.energy_events
        target.energy_ledger = preview.energy_ledger
        target.materialized_energy_event_ids = preview.materialized_energy_event_ids


def _path_energy_entries(*paths: tuple[dict[str, Any], ...]) -> list[dict[str, Any]]:
    return [
        dict(entry)
        for path in paths
        for leg in path
        for entry in tuple(leg.get("energy_ledger", ()) or ())
    ]


def _energy_summary_for_events(
    state: RuntimeState,
    trace_bundle: Any,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    identities = {
        (str(entry.get("satellite_id")), str(entry.get("event_id")))
        for entry in entries
        if entry.get("satellite_id") is not None and entry.get("event_id") is not None
    }
    max_time_by_satellite: dict[str, float] = {}
    for entry in entries:
        satellite_id = str(entry.get("satellite_id"))
        if (satellite_id, str(entry.get("event_id"))) not in identities:
            continue
        max_time_by_satellite[satellite_id] = max(
            max_time_by_satellite.get(satellite_id, 0.0),
            float(entry.get("event_time_s", 0.0)),
        )
    materialized: list[dict[str, Any]] = []
    for satellite_id, event_time_s in max_time_by_satellite.items():
        satellite = state.satellite_for(satellite_id)
        advance_satellite_energy(satellite, trace_bundle, event_time_s)
        materialized.extend(
            dict(row)
            for row in tuple(getattr(satellite, "energy_ledger", ()) or ())
            if (satellite_id, str(row.get("event_id"))) in identities
        )
    request_rows = [row for row in materialized if row.get("energy_kind") == "request_event"]
    if not request_rows:
        return {
            "soc_before": None,
            "soc_after": None,
            "energy_violation": False,
            "compute_j": 0.0,
            "link_j": 0.0,
            "energy_ledger": tuple(materialized),
        }
    return {
        "soc_before": min(float(row["soc_before"]) for row in request_rows),
        "soc_after": min(float(row["soc_after"]) for row in request_rows),
        "energy_violation": any(
            float(row["soc_after"]) < float(row.get("soc_min", 0.0))
            or float(row.get("unserved_energy_j", 0.0)) > 0
            for row in request_rows
        ),
        "compute_j": sum(float(row.get("compute_bus_j", 0.0)) for row in request_rows),
        "link_j": sum(float(row.get("link_bus_j", 0.0)) for row in request_rows),
        "energy_ledger": tuple(materialized),
    }


def _energy_summary_for_request(
    state: RuntimeState,
    trace_bundle: Any,
    request_id: str,
) -> dict[str, Any]:
    identities: list[dict[str, Any]] = []
    for satellite_id, satellite in state.satellites.items():
        for event in tuple(getattr(satellite, "energy_events", ()) or ()):
            if str(_dict(event.get("metadata")).get("request_id")) != str(request_id):
                continue
            identities.append(
                {
                    "satellite_id": satellite_id,
                    "event_id": event.get("event_id"),
                    "event_time_s": event.get("event_time_s", 0.0),
                }
            )
    return _energy_summary_for_events(state, trace_bundle, identities)


def _build_post_compute_outbound_plan(
    *,
    trace_bundle: Any,
    ingress_id: str,
    compute_satellite_id: str,
    policy: str,
    max_hop_count: int | None,
    output_bytes: float,
    decode_done_s: float,
) -> tuple[Any | None, str, tuple[str, ...]]:
    access_query = query_access_at(trace_bundle, ingress_id, decode_done_s)
    access_candidates = find_access_candidates(trace_bundle, ingress_id, decode_done_s)
    if not access_candidates:
        sgl_failure = _access_sgl_failure(trace_bundle, access_query, routing_time_s=decode_done_s)
        if sgl_failure is not None:
            reason, failure_warnings, _provenance = sgl_failure
            return None, reason, failure_warnings
        reason = _query_reject_reason(access_query, "outbound_access")
        return None, reason if reason != "stale_trace_sample:outbound_access" else "stale_trace_sample:access", access_query.warnings or (reason,)
    failures: list[str] = []
    warnings: list[str] = []
    route_rows = enumerate_compute_candidates(
        trace_bundle,
        access_candidates,
        policy,
        max_hop_count=max_hop_count,
        time_s=decode_done_s,
        failures=failures,
        warnings=warnings,
        target_satellite_ids=(compute_satellite_id,),
    )
    for access_candidate, candidate_compute_satellite_id, isl_link_ids in route_rows:
        if candidate_compute_satellite_id != compute_satellite_id:
            continue
        try:
            return (
                make_route_plan(
                    trace_bundle=trace_bundle,
                    access_candidate=access_candidate,
                    compute_satellite_id=compute_satellite_id,
                    isl_link_ids=isl_link_ids,
                    input_bytes=0.0,
                    output_bytes=output_bytes,
                    time_s=decode_done_s,
                ),
                "",
                (),
            )
        except RouteSamplingError as exc:
            failures.append(exc.reason)
            warnings.extend(exc.warnings or (exc.reason,))
        except ValueError as exc:
            failures.append(str(exc))
    reason = _first_stale_or_default(failures, "outbound_route_unavailable")
    return None, reason, tuple(dict.fromkeys(warnings or failures or (reason,)))


def _compute_eligibility_at(
    *,
    deployment: ServingCapacityProfile,
    trace_bundle: Any,
    state: RuntimeState,
    time_s: float,
) -> ComputeEligibilitySet:
    runtime_exclusions: dict[str, str] = {}
    for satellite_id in tuple(
        str(item.get("satellite_id"))
        for item in (getattr(trace_bundle, "satellite_catalog", ()) or ())
        if isinstance(item, dict) and item.get("satellite_id") is not None
    ):
        satellite_id = str(satellite_id)
        try:
            preview = replace(state.satellite_for(satellite_id))
            advance_satellite_energy(preview, trace_bundle, time_s)
            window = energy_window_at(trace_bundle, satellite_id, time_s)
        except EnergyWindowError as exc:
            runtime_exclusions[satellite_id] = exc.reason
            continue
        if window.get("compute_enabled") is False or window.get("energy_available") is False:
            runtime_exclusions[satellite_id] = "energy_state_compute_ineligible"
            continue
    return build_compute_eligibility_set(
        satellite_catalog=tuple(getattr(trace_bundle, "satellite_catalog", ()) or ()),
        runtime_exclusions=runtime_exclusions,
        evaluated_at_s=time_s,
    )


def _route_node_ids(path: tuple[dict[str, Any], ...]) -> tuple[str, ...]:
    nodes: list[str] = []
    for leg in path:
        for key in ("endpoint_a", "endpoint_b"):
            node = leg.get(key)
            if node is not None and str(node) not in nodes:
                nodes.append(str(node))
    return tuple(nodes)


def _post_compute_route_failure_record(
    *,
    request: dict[str, Any],
    action_type: str,
    route_plan: Any,
    inbound_path: tuple[dict[str, Any], ...],
    arrival_s: float,
    access_wait_s: float,
    prefill_wait_s: float,
    decode_wait_s: float,
    sgl_wait_in: float,
    isl_wait_in: float,
    tx_in: float,
    prefill_service_s: float,
    decode_service_s: float,
    prefill_done_s: float,
    compute_satellite_id: str,
    access_satellite_id: str,
    energy_update: dict[str, Any],
    deferred: bool,
    defer_until_s: float | None,
    session_action: str,
    scope: str,
    trace_model: str,
    failure_reason: str,
    failure_warnings: tuple[str, ...],
    scheduler_decision: dict[str, Any] | None = None,
    score_terms: dict[str, Any] | None = None,
    selected_candidate_id: str | None = None,
    candidate_count: int | None = None,
    feasible_candidate_count: int | None = None,
) -> RuntimeRequestRecord:
    del prefill_done_s
    return RuntimeRequestRecord(
        request_id=request["request_id"],
        session_id=request.get("session_id"),
        arrival_time_s=arrival_s,
        action_type=action_type,
        completed=False,
        uncompleted_stage="delivery",
        uncompleted_reason="link_interrupted",
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        prefill_queue_wait_s=prefill_wait_s,
        decode_queue_wait_s=decode_wait_s,
        sgl_queue_wait_s=sgl_wait_in,
        isl_queue_wait_s=isl_wait_in,
        link_queue_wait_s=sgl_wait_in + isl_wait_in,
        access_wait_s=access_wait_s,
        link_transmission_s=tx_in,
        prefill_service_s=prefill_service_s,
        decode_service_s=decode_service_s,
        ttft_s=None,
        e2e_s=None,
        serving_satellite_id=compute_satellite_id,
        compute_owner_satellite_id=compute_satellite_id,
        compute_owner_satellite_ids=(compute_satellite_id,),
        access_satellite_id=access_satellite_id,
        selected_path=inbound_path,
        route_node_ids=_route_node_ids(inbound_path),
        soc_before=energy_update["soc_before"],
        soc_after=energy_update["soc_after"],
        energy_violation=energy_update["energy_violation"],
        fallback_used=False,
        deferred=deferred,
        defer_until_s=defer_until_s,
        session_action=session_action,
        bottleneck_reason="route_became_unavailable_after_compute",
        scope=scope,
        trace_model=trace_model,
        valid_for_paper_main_results=False,
        provenance={
            "policy": "post_compute_outbound_revalidation",
            "route_plan": route_plan.to_dict() if hasattr(route_plan, "to_dict") else route_plan,
            "outbound_failure_reason": failure_reason,
            "warnings": failure_warnings,
            "selected_candidate_id": selected_candidate_id,
            "scheduler_decision": scheduler_decision,
            "score_terms": score_terms or {},
            "candidate_count": candidate_count,
            "feasible_candidate_count": feasible_candidate_count,
            "partial_failure_contract": "route_became_unavailable_after_compute",
            "ttft_definition": TTFT_DEFINITION,
            "energy_charged_for": ("compute", "completed_link_bytes_before_failure"),
            "energy_ledger": tuple(energy_update.get("energy_ledger", ())),
            "preliminary_surrogate": True,
        },
    )


def _estimate_route_duration_s(legs: tuple[Any, ...]) -> float:
    duration_s = 0.0
    for leg in legs:
        bandwidth_bps = float(getattr(leg, "bandwidth_bps", None) or 0.0)
        if bandwidth_bps <= 0:
            return math.inf
        duration_s += max(0.0, float(getattr(leg, "payload_bytes", 0.0))) * 8.0 / bandwidth_bps
        duration_s += float(getattr(leg, "setup_delay_s", None) or 0.0)
        duration_s += float(getattr(leg, "propagation_delay_s", None) or 0.0)
    return duration_s


def _fallback_or_reject(
    request: dict[str, Any],
    runtime_strategy: Any,
    reason: str,
    scope: str,
    trace_model: str,
    *,
    warnings: tuple[str, ...] = (),
    provenance: dict[str, Any] | None = None,
) -> RuntimeRequestRecord:
    fail_closed = _is_fail_closed_reason(reason)
    if bool(getattr(runtime_strategy, "fallback_enabled", False)) and not fail_closed:
        latency = _optional_float(_dict(getattr(runtime_strategy, "hard_gate_overrides", {})).get("ground_fallback_latency_s"))
        return _fallback_record(
            request,
            reason,
            latency,
            scope=scope,
            trace_model=trace_model,
            warnings=warnings,
            provenance=provenance,
        )
    return _reject_record(
        request,
        reason,
        scope=scope,
        trace_model=trace_model,
        warnings=warnings,
        provenance=provenance,
    )


def _is_fail_closed_reason(reason: str) -> bool:
    reason_text = str(reason)
    fail_closed_prefixes = (
        "stale_trace_sample",
        "trace_sample_missing",
        "link_sample",
        "link_unavailable",
        "energy_",
    )
    return reason_text.startswith(fail_closed_prefixes) or reason_text in {
        "active_graph_unavailable",
        "inactive_in_active_graph",
        "link_inactive_at_reservation",
        "no_energy_window",
        "trace_horizon_missing",
    }


def _fallback_record(
    request: dict[str, Any],
    reason: str,
    latency_s: float | None,
    *,
    scope: str,
    trace_model: str,
    warnings: tuple[str, ...] = (),
    provenance: dict[str, Any] | None = None,
) -> RuntimeRequestRecord:
    completed = latency_s is not None
    ttft_s = None
    e2e_s = None
    if latency_s is not None:
        ttft_s = latency_s
        e2e_s = latency_s
    return RuntimeRequestRecord(
        request_id=request["request_id"],
        session_id=request.get("session_id"),
        arrival_time_s=request["arrival_time_s"],
        action_type="ground_fallback",
        completed=completed,
        uncompleted_stage=None if completed else _failure_stage(reason),
        uncompleted_reason=None if completed else _failure_reason(reason),
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        bottleneck_reason="ground_fallback" if completed else "fallback_latency_missing",
        scope=scope,
        trace_model=trace_model,
        fallback_used=True,
        valid_for_paper_main_results=False,
        provenance={
            "fallback_reason": reason,
            "ground_fallback_latency_s": latency_s,
            "warnings": warnings,
            **(provenance or {}),
        },
    )


def _reject_record(
    request: dict[str, Any],
    reason: str,
    *,
    scope: str,
    trace_model: str = "surrogate_orbit_trace",
    warnings: tuple[str, ...] = (),
    provenance: dict[str, Any] | None = None,
) -> RuntimeRequestRecord:
    return RuntimeRequestRecord(
        request_id=request["request_id"],
        session_id=request.get("session_id"),
        arrival_time_s=request["arrival_time_s"],
        action_type="reject_drop",
        completed=False,
        uncompleted_stage=_failure_stage(reason),
        uncompleted_reason=_failure_reason(reason),
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        bottleneck_reason=reason,
        scope=scope,
        trace_model=trace_model,
        valid_for_paper_main_results=False,
        provenance={"failure_reason": reason, "warnings": warnings, **(provenance or {})},
    )


def _failure_stage(reason: str) -> str:
    text = str(reason)
    if "invalid" in text or "must_be_positive" in text:
        return "validation"
    if "trace_horizon" in text or "outside_trace" in text:
        return "validation"
    if "access" in text or "ingress" in text or "sgl" in text:
        return "access"
    if "outbound" in text or "after_compute" in text:
        return "delivery"
    if "route" in text or "link" in text or "graph" in text:
        return "routing"
    if "energy" in text:
        return "compute"
    if "kv" in text or "hbm" in text:
        return "compute"
    return "compute"


def _failure_reason(reason: str) -> str:
    text = str(reason)
    if "invalid" in text or "must_be_positive" in text:
        return "invalid_request"
    if "trace_horizon" in text or "outside_trace" in text:
        return "trace_ended"
    if "access" in text or "ingress" in text or "sgl" in text:
        return "no_access"
    if "energy" in text:
        return "energy_unavailable"
    if "kv" in text or "hbm" in text:
        return "kv_capacity_unavailable"
    if "interrupted" in text or "after_compute" in text:
        return "link_interrupted"
    if "route" in text or "link" in text or "graph" in text:
        return "no_route"
    return "no_compute_capacity"


def _session_action(
    state: RuntimeState,
    session_id: str | None,
    satellite_id: str,
    now_s: float,
    *,
    deployment: Any,
    runtime_strategy: Any,
    request: dict[str, Any],
) -> str:
    if not session_id:
        return "none"
    existed = session_id in state.sessions
    session = state.session_for(session_id)
    previous_owner = session.owner_satellite_id
    cache = state.kv_cache_for(satellite_id)
    cache_hit = existed and session_id in cache.entries
    action = "sticky_kv_owner" if cache_hit else "recompute_context"
    tokens = max(0, int(request.get("prompt_tokens", 0))) + max(
        0, int(request.get("output_tokens", 0))
    )
    admission = _admit_session_kv(
        state,
        capacity_tokens=getattr(deployment, "kv_cache_capacity_tokens", None),
        capacity_bytes=None,
        kv_policy=dict(getattr(runtime_strategy, "kv_policy", {}) or {}),
        session_id=str(session_id),
        satellite_id=str(satellite_id),
        tokens=tokens,
        kv_bytes_per_token=float(getattr(deployment, "kv_cache_bytes_per_token", 0.0) or 0.0),
        now_s=float(now_s),
    )
    if not admission["admitted"]:
        session.owner_satellite_id = None
        session.last_session_action = "recompute_context"
        session.last_kv_event = dict(admission)
        session.last_seen_s = now_s
        return "recompute_context"
    if previous_owner and previous_owner != satellite_id:
        _remove_session_kv(state, previous_owner, str(session_id))
    entry = admission["entry"]
    session.owner_satellite_id = satellite_id
    session.context_tokens = tokens
    session.last_seen_s = now_s
    session.last_session_action = action
    session.last_kv_event = {
        **dict(admission),
        "hit_tokens": tokens if action == "sticky_kv_owner" else 0,
        "miss_tokens": tokens if action == "recompute_context" else 0,
        "recompute_tokens": tokens if action == "recompute_context" else 0,
    }
    return action


def _bottleneck(
    energy_violation: bool,
    prefill_wait_s: float,
    decode_wait_s: float,
    link_wait_s: float,
) -> str:
    if energy_violation:
        return "energy_violation"
    if link_wait_s > 0:
        return "link_queue"
    if prefill_wait_s > 0 or decode_wait_s > 0:
        return "compute_queue"
    return "service"


def _decision_trace(
    request: dict[str, Any],
    policy: str,
    access_candidates: tuple[Any, ...],
    route_plan: Any,
    bottleneck: str,
    *,
    warnings: tuple[str, ...] = (),
    scheduler_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scheduler_payload = scheduler_decision or {}
    selected_action = route_plan.to_dict() if hasattr(route_plan, "to_dict") else None
    return {
        "request_id": request["request_id"],
        "arrival_time_s": request["arrival_time_s"],
        "decision_time_s": request["arrival_time_s"],
        "policy_name": policy,
        "candidate_action_count": scheduler_payload.get("candidate_count", len(access_candidates)),
        "feasible_action_count": scheduler_payload.get("feasible_candidate_count", 1 if route_plan is not None else 0),
        "selected_candidate_id": scheduler_payload.get("selected_candidate_id"),
        "selected_action": selected_action,
        "selected_candidate": scheduler_payload.get("selected_action"),
        "rejected_actions": scheduler_payload.get("rejected_candidates", ()),
        "gate_results": scheduler_payload.get("gate_results", {}),
        "score_components": scheduler_payload.get("score_terms", {}),
        "scheduler_decision": scheduler_payload,
        "bottleneck_reason": bottleneck,
        "trace_schema_version": "orbitserve.runtime_decision_trace.v1",
        "warnings": warnings,
    }


def _access_sgl_failure(
    trace_bundle: Any,
    access_query: Any,
    *,
    routing_time_s: float,
) -> tuple[str, tuple[str, ...], dict[str, Any]] | None:
    if not getattr(access_query, "available", False) or getattr(access_query, "stale", False):
        return None
    windows = _access_query_items(getattr(access_query, "value", None))
    if not windows:
        return None
    failures: list[str] = []
    warnings: list[str] = []
    failure_details: list[dict[str, Any]] = []
    checked_count = 0
    for window in windows:
        sgl_link_id = _access_sgl_link_id(window)
        if sgl_link_id is None:
            continue
        checked_count += 1
        sample = query_link_sample_at(trace_bundle, sgl_link_id, routing_time_s)
        if sample.available and not sample.stale:
            continue
        reason = _query_reject_reason(sample, "link")
        failures.append(reason)
        warnings.extend(tuple(getattr(sample, "warnings", ()) or (reason,)))
        failure_details.append(
            {
                "sgl_link_id": sgl_link_id,
                "link_query": _query_payload(sample),
            }
        )
    if checked_count <= 0 or len(failures) != checked_count:
        return None
    reason = _first_stale_or_default(failures, failures[0])
    return (
        reason,
        tuple(dict.fromkeys(warnings or failures)),
        {"sgl_link_failures": tuple(failure_details)},
    )


def _access_query_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        for key in ("access_windows", "candidates", "windows"):
            if key in value:
                return tuple(value.get(key) or ())
        return (value,)
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _access_sgl_link_id(value: Any) -> str | None:
    payload = _dict(value)
    if payload:
        return _optional_str(payload.get("sgl_link_id"))
    return _optional_str(getattr(value, "sgl_link_id", None))


def _default_ingress(trace_bundle: Any) -> str | None:
    for window in getattr(trace_bundle, "access_windows", ()) or ():
        if window.get("ingress_id") is not None:
            return str(window["ingress_id"])
    return None


def _trace_horizon_s(trace_bundle: Any) -> tuple[float | None, tuple[str, ...]]:
    for source in _trace_time_sources(trace_bundle):
        horizon_s = _optional_float(
            _first_present(
                source.get("duration_s"),
                source.get("time_horizon_s"),
                source.get("final_boundary_s"),
            )
        )
        if horizon_s is not None and horizon_s > 0:
            return horizon_s, ()
    provenance_horizon = _optional_float(_dict(getattr(trace_bundle, "provenance", None)).get("effective_duration_s"))
    if provenance_horizon is not None and provenance_horizon > 0:
        return provenance_horizon, ()
    window_ends: list[float] = []
    for attr, trace_attr, key in (
        ("access_windows", "access_trace", "access_windows"),
        ("energy_windows", "energy_trace", "energy_windows"),
        ("shadow_windows", "energy_trace", "shadow_windows"),
    ):
        for row in getattr(trace_bundle, attr, ()) or ():
            end_s = _optional_float(_dict(row).get("end_s"))
            if end_s is not None:
                window_ends.append(end_s)
        for row in _dict(getattr(trace_bundle, trace_attr, None)).get(key, ()) or ():
            end_s = _optional_float(_dict(row).get("end_s"))
            if end_s is not None:
                window_ends.append(end_s)
    if window_ends:
        return max(window_ends), ()
    sample_times = _trace_sample_times(trace_bundle)
    if sample_times:
        step_s = _trace_step_s(trace_bundle, sample_times)
        if step_s is not None:
            return max(sample_times) + step_s, ()
    return None, ("trace_horizon_missing",)


def _trace_time_sources(trace_bundle: Any) -> tuple[dict[str, Any], ...]:
    sources: list[dict[str, Any]] = []
    attr_horizon_s = _optional_float(getattr(trace_bundle, "horizon_s", None))
    attr_step_s = _optional_float(getattr(trace_bundle, "step_s", None))
    attr_tolerance_s = _optional_float(getattr(trace_bundle, "boundary_tolerance_s", None))
    attr_source = {
        key: value
        for key, value in (
            ("duration_s", attr_horizon_s),
            ("step_s", attr_step_s),
            ("boundary_tolerance_s", attr_tolerance_s),
        )
        if value is not None
    }
    if attr_source:
        sources.append(attr_source)
    for value in (
        getattr(trace_bundle, "time", None),
        getattr(trace_bundle, "trace_time", None),
        _dict(getattr(trace_bundle, "trace_manifest", None)).get("time"),
        _dict(getattr(trace_bundle, "manifest", None)).get("time"),
        _dict(getattr(trace_bundle, "access_trace", None)).get("time"),
        _dict(getattr(trace_bundle, "link_trace", None)).get("time"),
        _dict(getattr(trace_bundle, "energy_trace", None)).get("time"),
    ):
        payload = _dict(value)
        if payload:
            sources.append(payload)
    return tuple(sources)


def _trace_sample_times(trace_bundle: Any) -> tuple[float, ...]:
    times: list[float] = []
    for trace_attr, key in (
        ("access_trace", "visible_count_samples"),
        ("link_trace", "link_samples"),
        ("link_trace", "active_graph_snapshots"),
        ("energy_trace", "solar_samples"),
    ):
        for row in _dict(getattr(trace_bundle, trace_attr, None)).get(key, ()) or ():
            value = _optional_float(_dict(row).get("time_s"))
            if value is not None:
                times.append(value)
    for attr in ("link_samples", "active_graph_snapshots", "visible_count_samples", "solar_samples"):
        for row in getattr(trace_bundle, attr, ()) or ():
            value = _optional_float(_dict(row).get("time_s"))
            if value is not None:
                times.append(value)
    return tuple(sorted(set(times)))


def _trace_step_s(trace_bundle: Any, sample_times: tuple[float, ...]) -> float | None:
    for source in _trace_time_sources(trace_bundle):
        step_s = _optional_float(_first_present(source.get("step_s"), source.get("time_step_s"), source.get("time_bin_s")))
        if step_s is not None and step_s > 0:
            return step_s
    deltas = [b - a for a, b in zip(sample_times, sample_times[1:]) if b > a]
    return min(deltas) if deltas else None


def _query_reject_reason(query: Any, kind: str) -> str:
    if getattr(query, "stale", False):
        reason = _optional_str(getattr(query, "reason", None))
        return reason if reason and reason.startswith("stale_trace_sample") else f"stale_trace_sample:{kind}"
    return _optional_str(getattr(query, "reason", None)) or f"no_{kind}"


def _query_payload(query: Any) -> dict[str, Any]:
    return {
        "available": bool(getattr(query, "available", False)),
        "reason": getattr(query, "reason", None),
        "sample_time_s": getattr(query, "sample_time_s", None),
        "stale": bool(getattr(query, "stale", False)),
        "warnings": tuple(getattr(query, "warnings", ()) or ()),
    }


def _energy_error_warnings(exc: EnergyWindowError) -> tuple[str, ...]:
    return tuple(dict.fromkeys((*exc.warnings, exc.reason)))


def _energy_error_payload(exc: EnergyWindowError) -> dict[str, Any]:
    return {
        "available": False,
        "reason": exc.reason,
        "satellite_id": exc.satellite_id,
        "time_s": exc.time_s,
        "sample_time_s": exc.sample_time_s,
        "warnings": exc.warnings,
    }


def _first_stale_or_default(reasons: list[str], default: str) -> str:
    for reason in reasons:
        if str(reason).startswith("stale_trace_sample"):
            return str(reason)
    return str(reasons[0]) if reasons else default


def _jsonable_strategy(runtime_strategy: Any) -> dict[str, Any]:
    if hasattr(runtime_strategy, "to_dict") and callable(runtime_strategy.to_dict):
        payload = runtime_strategy.to_dict()
        return payload if isinstance(payload, dict) else {}
    if isinstance(runtime_strategy, dict):
        return dict(runtime_strategy)
    return {
        key: getattr(runtime_strategy, key)
        for key in (
            "policy",
            "weights",
            "normalization_refs",
            "max_hop_count",
            "fallback_enabled",
            "defer_enabled",
        )
        if hasattr(runtime_strategy, key)
    }


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}






def _safe_service_time(tokens: int, tokens_per_s: float | None) -> float:
    if tokens <= 0:
        return 0.0
    if tokens_per_s is None or tokens_per_s <= 0:
        return math.inf
    return tokens / tokens_per_s


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
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
    if isinstance(value, (tuple, list, set)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)


def _int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


from dataclasses import replace
from typing import Any

from ..contracts import RuntimePerformanceResult, RuntimeStrategySpec
from ..result import build_runtime_performance_result
from ...orbit import (
    CIRCULAR_GEOMETRY_TRACE,
    SURROGATE_ORBIT_TRACE,
    RuntimeTraceManifest,
    load_surrogate_trace_bundle,
)


def try_run_runtime_simulation(
    scenario: Any,
) -> RuntimePerformanceResult | None:
    runtime_strategy = RuntimeStrategySpec.from_value(scenario.strategy)
    if runtime_strategy.policy.startswith("ground_") or runtime_strategy.policy == "skywalker_like_ground":
        return None
    requests = tuple(getattr(scenario, "requests", ()) or ())
    if not requests:
        return None
    trace_manifest = RuntimeTraceManifest.from_value(
        getattr(scenario, "simulation_trace_ref", None)
    )
    if not trace_manifest.evaluated:
        return None
    if trace_manifest.trace_model not in {SURROGATE_ORBIT_TRACE, CIRCULAR_GEOMETRY_TRACE}:
        return None
    try:
        trace_bundle = load_surrogate_trace_bundle(trace_manifest)
    except ValueError as exc:
        return RuntimePerformanceResult(
            evaluated=False,
            scenario={
                "schema_version": getattr(scenario, "schema_version", "orbitserve.runtime_scenario.v1"),
                "scenario_id": getattr(scenario, "scenario_id", None),
                "design_id": getattr(scenario, "design_id", None),
                "time_horizon_s": getattr(scenario, "time_horizon_s", None),
                "simulation_evaluated": True,
                "trace_model": trace_manifest.trace_model,
                "scope": trace_manifest.scope,
                "valid_for_paper_main_results": False,
            },
            strategy=runtime_strategy.to_dict(),
            workload=getattr(scenario, "workload_bundle", {}),
            constellation=getattr(scenario, "constellation", {}),
            scheduler=runtime_strategy.to_dict(),
            trace_model=trace_manifest.trace_model,
            scope=trace_manifest.scope,
            valid_for_paper_main_results=False,
            module_status={"simulation": "surrogate_trace", "runtime_strategy": runtime_strategy.policy},
            details={"valid_for_paper_main_results": False, "trace_model": trace_manifest.trace_model},
            warnings=(f"runtime trace loader failed: {exc}",),
        )
    trace_bundle = _trace_bundle_with_scenario_energy(trace_bundle, scenario)
    serving_profile = _serving_profile_for_trace(scenario, trace_bundle)
    if serving_profile is None or not serving_profile.has_compute_capacity:
        return None
    engine_result = run_trace_backed_runtime(
        requests=requests,
        serving_profile=serving_profile,
        runtime_strategy=runtime_strategy,
        trace_bundle=trace_bundle,
        time_horizon_s=getattr(scenario, "time_horizon_s", None),
        scope=trace_manifest.scope,
    )
    circular = trace_manifest.trace_model == CIRCULAR_GEOMETRY_TRACE
    return build_runtime_performance_result(
        scenario=scenario,
        runtime_strategy=runtime_strategy,
        engine_result=engine_result,
        trace_manifest=trace_manifest,
        effective_trace_model=trace_manifest.trace_model,
        effective_scope=trace_manifest.scope,
        serving_profile=serving_profile,
        evaluator_name="circular_geometry_trace_backed_runtime" if circular else "surrogate_trace_backed_runtime",
        path_model="circular_geometry_trace_sgl_isl" if circular else "surrogate_trace_sgl_isl",
        energy_model="circular_geometry_shadow_dvfs" if circular else "surrogate_soc_shadow_dvfs",
        queue_model="per_satellite_fifo",
        link_model_status="surrogate_link_queue",
        energy_model_status="surrogate_soc",
    )


def _serving_profile_for_trace(scenario: Any, trace_bundle: Any) -> Any:
    del trace_bundle
    return getattr(scenario, "serving_profile", None)


def _trace_bundle_with_scenario_energy(trace_bundle: Any, scenario: Any) -> Any:
    """Overlay static scenario energy parameters onto Orbit sunlight windows."""

    profile = getattr(scenario, "energy_profile", {}) or {}
    if not isinstance(profile, dict) or profile.get("evaluated") is False:
        return trace_bundle
    fields = {
        key: value
        for key, value in profile.items()
        if key not in {"evaluated", "provenance", "warnings"}
    }
    if not fields:
        return trace_bundle
    windows = tuple(
        {**dict(window), **fields}
        for window in (getattr(trace_bundle, "energy_windows", ()) or ())
    )
    energy_trace = dict(getattr(trace_bundle, "energy_trace", {}) or {})
    if windows:
        energy_trace["energy_windows"] = list(windows)
    satellite_ids = (
        str(item.get("satellite_id"))
        for item in (getattr(trace_bundle, "satellite_catalog", ()) or ())
        if isinstance(item, dict) and item.get("satellite_id") is not None
    )
    initial_soc = float(fields.get("initial_soc", 0.8))
    return replace(
        trace_bundle,
        energy_windows=windows,
        energy_trace=energy_trace,
        satellite_initial_soc={satellite_id: initial_soc for satellite_id in satellite_ids},
    )


__all__ = [
    "RuntimeEngineResult",
    "normalize_runtime_request",
    "run_trace_backed_runtime",
    "try_run_runtime_simulation",
]
