"""Event-engine implementations for OrbitServe runtime evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

from ...common import to_jsonable
from .contracts import RuntimeEngineResult
from ..state.energy_state import EnergyWindowError, advance_satellite_energy, apply_request_energy, energy_window_at
from ..state.energy_accounting import EnergyIntensity, compute_dynamic_energy_j
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
from ..scenario.deployment_contracts import RuntimeDeployment, RuntimeDeploymentCatalog
from .runtime_eligibility import ComputeEligibilitySet, build_compute_eligibility_set
from ..metrics.runtime_records import RuntimeRequestRecord
from ..state.runtime_state import RuntimeState
from .runtime_timing import TTFT_DEFINITION, ttft_breakdown
from ..state.kv_runtime import (
    _pd_prefill_context,
    _pd_prefill_context_for_action,
    _pd_context_tokens_after,
    _pd_context_recompute_tokens,
    _pd_session_action_preview,
    _pd_kv_serve_actions_for_decode,
    _pd_session_kv_available_on_decode,
    _pd_kv_post_turn_candidates,
    _pd_replica_target_deployment,
    _pd_session_action_compat,
    _pd_migration_start_s,
    _kv_policy,
    _pd_max_kv_hop_count,
    _kv_transfer_chunk_bytes,
    _kv_page_size_tokens,
    _kv_required_pages,
    _request_kv_deadline_s,
    _session_expires_at,
    _session_ttl_deadline,
    _pd_kv_capacity_result,
    _pd_existing_kv_tokens_for_decode,
    _pd_existing_kv_bytes_for_decode,
    _deployment_hbm_capacity_bytes,
    _kv_accounted_bytes,
    _card_level_hbm_enabled,
    _sync_kv_cache_state_for_deployment,
    _update_pd_session_state,
    _expire_session_kv,
    _session_ttl_s,
    _token_context_contract_error,
    _pd_catalog_externalizes_kv,
    _pd_catalog_has_kv_bytes,
    _pd_internal_comm_accounting_values,
)
from ..strategy.schedulers import SchedulerCandidate, get_scheduler
from ...orbit import TRACE_BACKED_RUNTIME_SURROGATE_SCOPE, normalize_trace_scope




@dataclass(frozen=True)
class KVTransferResult:
    done_s: float
    path: tuple[dict[str, Any], ...] = ()
    sgl_wait_s: float = 0.0
    isl_wait_s: float = 0.0
    transmission_s: float = 0.0
    attempted_bytes: float = 0.0
    delivered_bytes: float = 0.0
    committed_page_bytes: float = 0.0
    useful_bytes: float = 0.0
    interrupted: bool = False
    expired: bool = False
    interruption_reason: str | None = None
    expiry_reason: str | None = None
    sampled_legs: tuple[Any, ...] = ()


class KVTransferRuntimeError(RuntimeError):
    def __init__(
        self,
        reason: str,
        *,
        transfer: KVTransferResult,
        kv_serve_action: str | None,
        kv_post_turn_action: str | None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.transfer = transfer
        self.kv_serve_action = kv_serve_action
        self.kv_post_turn_action = kv_post_turn_action




def run_trace_backed_runtime(
    *,
    requests: tuple[dict[str, Any], ...],
    deployment_catalog: RuntimeDeploymentCatalog,
    runtime_strategy: Any,
    trace_bundle: Any,
    time_horizon_s: float | None = None,
    scope: str = TRACE_BACKED_RUNTIME_SURROGATE_SCOPE,
) -> RuntimeEngineResult:
    """Run a deterministic surrogate access/link/energy runtime model."""

    state = RuntimeState()
    scope = normalize_trace_scope(scope)
    explicit_pd = _has_explicit_pd_deployments(deployment_catalog)
    state.module_status.update(
        {
            "queue_model": "deployment_fifo" if explicit_pd else "per_satellite_fifo",
            "link_model": "surrogate_link_queue",
            "energy_model": "surrogate_soc_shadow_dvfs",
            "kv_state": (
                "pd_externalized_kv_handoff_session_actions"
                if explicit_pd
                else "sticky_or_recompute_only"
            ),
            "kv_transfer_model": "page_chunk_store_and_forward" if explicit_pd else "not_implemented",
            "kv_hbm_model": "deployment_token_capacity_surrogate" if explicit_pd else "not_implemented",
            "paged_attention_model": "session_level_page_accounting" if explicit_pd else "not_implemented",
        }
    )
    warnings = list(getattr(trace_bundle, "warnings", ()))
    warnings.extend(deployment_catalog.warnings)
    if explicit_pd:
        warnings.append("pd_compute_energy_uses_explicit_member_weights_and_link_energy_uses_executed_legs")
        warnings.append("pd_session_kv_migration_replication_use_trace_backed_surrogate")
    trace_model = str(getattr(trace_bundle, "trace_model", "surrogate_orbit_trace"))
    policy = str(getattr(runtime_strategy, "policy", "local_only"))
    deployments = deployment_catalog.deployments
    deployment = deployment_catalog.default_deployment
    satellite_catalog = tuple(getattr(trace_bundle, "satellite_catalog", ()) or ())
    if len(satellite_catalog) <= 0:
        return RuntimeEngineResult(
            records=tuple(
                _reject_record(
                    normalize_runtime_request(request, index),
                    "satellite_catalog_empty",
                    scope=scope,
                    trace_model=trace_model,
                )
                for index, request in enumerate(requests)
            ),
            decision_traces=(),
            state_summary=state.to_dict(),
            warnings=("surrogate trace satellite_catalog is empty",),
            scope=scope,
            trace_model=trace_model,
            provenance={"runtime_engine": "trace_backed_runtime_surrogate"},
            valid_for_paper_main_results=False,
        )
    if (deployment is None or not deployment.has_compute_capacity) and not explicit_pd:
        return RuntimeEngineResult(
            records=tuple(
                _reject_record(
                    normalize_runtime_request(request, index),
                    "compute_capacity_unavailable",
                    scope=scope,
                    trace_model=trace_model,
                )
                for index, request in enumerate(requests)
            ),
            decision_traces=(),
            state_summary=state.to_dict(),
            warnings=("runtime deployment catalog has no usable compute capacity",),
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
        ((normalize_runtime_request(request, index), request) for index, request in enumerate(requests)),
        key=lambda item: (item[0]["arrival_time_s"], item[0]["request_id"]),
    )
    records: list[RuntimeRequestRecord] = []
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
        _expire_session_kv(state, normalized["arrival_time_s"])
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
        elif explicit_pd:
            record, trace = _evaluate_trace_pd_request(
                state=state,
                request=normalized,
                raw_request=raw_request,
                trace_bundle=trace_bundle,
                deployment_catalog=deployment_catalog,
                deployment_id=_pd_logical_deployment_id(deployment_catalog),
                policy=policy,
                scheduler=scheduler,
                runtime_strategy=runtime_strategy,
                max_hop_count=max_hop_count,
                scope=scope,
                trace_model=trace_model,
                trace_horizon_s=trace_horizon_s,
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
        state.request_records.append(record)
        state.decision_traces.append(trace)
        if record.session_action:
            session_action_counts[record.session_action] = session_action_counts.get(record.session_action, 0) + 1

    warnings.append("trace-backed runtime is a preliminary surrogate model, not paper-main STK-calibrated simulation")
    return RuntimeEngineResult(
        records=tuple(records),
        decision_traces=tuple(state.decision_traces),
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
            "capacity_allocation_assumption": (
                "explicit_pd_deployment_group_per_group_throughput"
                if explicit_pd
                else "nominal_equal_share_per_deployed_compute_member"
            ),
            "configured_time_horizon_s": time_horizon_s,
            "trace_horizon_s": trace_horizon_s,
            "effective_time_horizon_s": trace_horizon_s,
        },
        valid_for_paper_main_results=False,
    )




def normalize_runtime_request(raw_request: dict[str, Any], index: int = 0) -> dict[str, Any]:
    declared_prompt_tokens = _first_present(
        raw_request.get("prompt_tokens"),
        raw_request.get("input_tokens"),
    )
    declared_new_user_tokens = raw_request.get("new_user_tokens")
    prompt_tokens = max(
        0,
        _int(
            _first_present(
                declared_prompt_tokens,
                raw_request.get("new_user_tokens"),
            ),
            0,
        ),
    )
    output_tokens = max(
        0,
        _int(
            _first_present(
                raw_request.get("output_tokens"),
                raw_request.get("max_output_tokens"),
                raw_request.get("decode_tokens"),
            ),
            0,
        ),
    )
    return {
        "request_id": str(raw_request.get("request_id") or f"req-{index:06d}"),
        "arrival_time_s": _float(_first_present(raw_request.get("arrival_time_s"), raw_request.get("time_s")), 0.0),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "input_bytes": _float(raw_request.get("input_bytes"), prompt_tokens * 4.0),
        "output_bytes": _float(raw_request.get("output_bytes"), output_tokens * 4.0),
        "ttft_deadline_s": _optional_float(_first_present(raw_request.get("ttft_deadline_s"), raw_request.get("deadline_ttft_s"))),
        "e2e_deadline_s": _optional_float(
            _first_present(raw_request.get("e2e_deadline_s"), raw_request.get("deadline_s"), raw_request.get("deadline_e2e_s"))
        ),
        "session_id": _optional_str(raw_request.get("session_id")),
        "ingress_id": _optional_str(_first_present(raw_request.get("ingress_id"), raw_request.get("gateway_id"))),
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
    deployment: RuntimeDeployment,
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
                deadline_ok = request["e2e_deadline_s"] is None or wait_s < request["e2e_deadline_s"]
                max_defer_ok = max_defer_s is None or wait_s <= max_defer_s
                if deadline_ok and max_defer_ok:
                    access_candidates = (next_access,)
                    access_wait_s = wait_s
                    deferred = True
                    defer_until_s = next_access.start_s
        if not access_candidates:
            record = _fallback_or_reject(request, runtime_strategy, "no_access_window", scope, trace_model)
            return record, _decision_trace(request, policy, (), None, record.reject_reason or "fallback")

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
            record.reject_reason or failure_reason,
            warnings=(failure_reason,),
        )
    prefill_rate_per_sat = float(deployment.prefill_tokens_per_s or 0.0) / eligibility.capacity_member_count
    decode_rate_per_sat = float(deployment.decode_tokens_per_s or 0.0) / eligibility.capacity_member_count
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
            record.reject_reason or "fallback",
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
            record.reject_reason or reason or "fallback",
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
            record.reject_reason or "fallback",
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
    dvfs = float(request_energy_window.get("dvfs_multiplier", 1.0))
    prefill_service_s = _safe_service_time(request["prompt_tokens"], prefill_rate_per_sat) * dvfs
    decode_service_s = _safe_service_time(request["output_tokens"], decode_rate_per_sat) * dvfs
    if not math.isfinite(prefill_service_s) or not math.isfinite(decode_service_s):
        record = _reject_record(request, "non_finite_service_time", scope=scope, trace_model=trace_model)
        return record, _decision_trace(request, policy, access_candidates, route_plan, "non_finite_service_time")
    prefill_start_s, prefill_done_s, prefill_wait_s = satellite.compute_queue.reserve_prefill(inbound_done_s, prefill_service_s)
    decode_start_s, decode_done_s, decode_wait_s = satellite.compute_queue.reserve_decode(prefill_done_s, decode_service_s)
    first_token_service_s = min(decode_service_s, _safe_service_time(1, decode_rate_per_sat) * dvfs)
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
        session_action = _session_action(state, request.get("session_id"), route_plan.compute_satellite_id, arrival_s)
        record = _post_compute_route_failure_record(
            request=request,
            deployment_id=deployment.deployment_id,
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
        session_action = _session_action(state, request.get("session_id"), route_plan.compute_satellite_id, arrival_s)
        record = _post_compute_route_failure_record(
            request=request,
            deployment_id=deployment.deployment_id,
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
            session_action = _session_action(state, request.get("session_id"), route_plan.compute_satellite_id, arrival_s)
            record = _post_compute_route_failure_record(
                request=request,
                deployment_id=deployment.deployment_id,
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
            partial_timing = ttft_breakdown(
                access_wait_s=access_wait_s,
                uplink_s=max(0.0, inbound_done_s - (arrival_s + access_wait_s)),
                prefill_queue_s=prefill_wait_s,
                prefill_compute_s=prefill_service_s,
                kv_transfer_s=0.0,
                decode_queue_s=decode_wait_s,
                first_token_decode_s=first_token_service_s,
                return_path_s=max(0.0, first_done_s - first_token_ready_s),
            )
            record = replace(
                record,
                ttft_s=float(partial_timing["ttft_s"]),
                uplink_s=float(partial_timing["uplink_s"]),
                kv_transfer_s=0.0,
                first_token_decode_s=first_token_service_s,
                return_path_s=float(partial_timing["return_path_s"]),
                ttft_breakdown_s=partial_timing,
                first_token_return_s=float(partial_timing["return_path_s"]),
                response_streaming_model="first_token_plus_bulk_remaining",
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
    timeout, timeout_reason = _timeout_status(ttft_s, e2e_s, request["ttft_deadline_s"], request["e2e_deadline_s"])
    session_action = _session_action(state, request.get("session_id"), route_plan.compute_satellite_id, arrival_s)
    served = not timeout and not energy_update["energy_violation"]
    total_sgl_wait = sgl_wait_in + sgl_wait_first + sgl_wait_remaining
    total_isl_wait = isl_wait_in + isl_wait_first + isl_wait_remaining
    bottleneck = _bottleneck(timeout_reason, energy_update["energy_violation"], prefill_wait_s, decode_wait_s, total_sgl_wait + total_isl_wait)
    selected_path = tuple((*inbound_path, *first_path, *remaining_path))
    record = RuntimeRequestRecord(
        request_id=request["request_id"],
        arrival_time_s=arrival_s,
        deployment_id=deployment.deployment_id,
        action_type=action_type,
        accepted=True,
        completed=True,
        served=served,
        rejected=False,
        timeout=timeout,
        timeout_reason=timeout_reason,
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        ttft_deadline_s=request["ttft_deadline_s"],
        e2e_deadline_s=request["e2e_deadline_s"],
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
        assigned_satellite_id=route_plan.compute_satellite_id,
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
        dvfs_multiplier=float(request_energy_window.get("dvfs_multiplier", 1.0)),
        shadow_strategy=selected.get("shadow_strategy"),
        fallback_used=False,
        deferred=deferred,
        defer_until_s=defer_until_s,
        session_action=session_action,
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
            "raw_dvfs_multiplier": raw_request_energy_window.get("dvfs_multiplier"),
            "effective_dvfs_multiplier": request_energy_window.get("dvfs_multiplier"),
            "shadow_strategy": selected.get("shadow_strategy"),
            "shadow_triggered": _shadow_triggered(action_type, selected.get("access_energy_window"), raw_request_energy_window),
            "shadow_runtime_action": action_type,
            "capacity_allocation_assumption": "nominal_equal_share_per_deployed_compute_member",
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


def _evaluate_trace_pd_request(
    *,
    state: RuntimeState,
    request: dict[str, Any],
    raw_request: dict[str, Any],
    trace_bundle: Any,
    deployment_catalog: RuntimeDeploymentCatalog,
    deployment_id: str,
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
    token_contract_error = _token_context_contract_error(request)
    if token_contract_error is not None:
        record = _reject_record(
            request,
            token_contract_error,
            scope=scope,
            trace_model=trace_model,
        )
        return record, _decision_trace(request, policy, (), None, token_contract_error)
    if not _pd_catalog_externalizes_kv(deployment_catalog):
        reason = "not_implemented:no_external_kv_route"
        record = _reject_record(
            request,
            reason,
            scope=scope,
            trace_model=trace_model,
            provenance={"internal_comm_accounting": _pd_internal_comm_accounting_values(deployment_catalog)},
        )
        return record, _decision_trace(request, policy, (), None, reason)
    if not _pd_catalog_has_kv_bytes(deployment_catalog):
        reason = "pd_kv_cache_bytes_per_token_missing"
        record = _reject_record(
            request,
            reason,
            scope=scope,
            trace_model=trace_model,
        )
        return record, _decision_trace(request, policy, (), None, reason)

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
                deadline_ok = request["e2e_deadline_s"] is None or wait_s < request["e2e_deadline_s"]
                max_defer_ok = max_defer_s is None or wait_s <= max_defer_s
                if deadline_ok and max_defer_ok:
                    access_candidates = (next_access,)
                    access_wait_s = wait_s
                    deferred = True
                    defer_until_s = next_access.start_s
        if not access_candidates:
            record = _fallback_or_reject(request, runtime_strategy, "no_access_window", scope, trace_model)
            return record, _decision_trace(request, policy, (), None, record.reject_reason or "fallback")

    selected = _select_pd_candidate(
        state=state,
        trace_bundle=trace_bundle,
        deployment_catalog=deployment_catalog,
        request=request,
        ingress_id=str(ingress_id),
        access_candidates=access_candidates,
        policy=policy,
        scheduler=scheduler,
        runtime_strategy=runtime_strategy,
        max_hop_count=max_hop_count,
        access_wait_s=access_wait_s,
        arrival_s=arrival_s,
    )
    if selected is None:
        record = _fallback_or_reject(
            request,
            runtime_strategy,
            "candidate_prediction_failed",
            scope,
            trace_model,
            warnings=("pd_candidate_prediction_failed",),
        )
        return record, _decision_trace(request, policy, access_candidates, None, record.reject_reason or "candidate_prediction_failed")
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
        reject_reason = (
            "kv_capacity"
            if "kv_capacity" in tuple(selected.get("hard_gate_violations", ()) or ())
            else "no_feasible_scheduler_candidate"
        )
        record = _fallback_or_reject(
            request,
            runtime_strategy,
            reject_reason,
            scope,
            trace_model,
            provenance=provenance,
        )
        if reject_reason == "kv_capacity":
            record = replace(record, kv_capacity_admission="fail")
        return record, _decision_trace(
            request,
            policy,
            access_candidates,
            None,
            record.reject_reason or "fallback",
            scheduler_decision=scheduler_decision,
        )

    try:
        return _reserve_selected_pd_candidate(
            state=state,
            request=request,
            selected=selected,
            trace_bundle=trace_bundle,
            deployment_id=deployment_id,
            policy=policy,
            runtime_strategy=runtime_strategy,
            scope=scope,
            trace_model=trace_model,
            access_wait_s=access_wait_s,
            deferred=deferred,
            defer_until_s=defer_until_s,
        )
    except KVTransferRuntimeError as exc:
        record = _fallback_or_reject(
            request,
            runtime_strategy,
            exc.reason,
            scope,
            trace_model,
            warnings=(exc.reason,),
        )
        transfer = exc.transfer
        partial_energy = _energy_summary_for_request(state, trace_bundle, request["request_id"])
        record = replace(
            record,
            kv_transfer_attempted_bytes=transfer.attempted_bytes,
            kv_transfer_delivered_bytes=transfer.delivered_bytes,
            kv_transfer_committed_page_bytes=transfer.committed_page_bytes,
            kv_transfer_useful_bytes=transfer.useful_bytes,
            kv_transfer_interrupted=transfer.interrupted,
            kv_transfer_expired=transfer.expired,
            kv_transfer_interruption_reason=transfer.interruption_reason,
            kv_transfer_expiry_reason=transfer.expiry_reason,
            kv_serve_action=exc.kv_serve_action,
            kv_post_turn_action=exc.kv_post_turn_action,
            soc_before=partial_energy["soc_before"],
            soc_after=partial_energy["soc_after"],
            energy_violation=partial_energy["energy_violation"],
            energy_ledger=tuple(partial_energy.get("energy_ledger", ())),
            provenance={
                **record.provenance,
                "energy_ledger": tuple(partial_energy.get("energy_ledger", ())),
                "energy_charged_for": "completed_events_before_kv_failure",
            },
        )
        return record, _decision_trace(request, policy, access_candidates, selected.get("route_in_plan"), exc.reason, warnings=(exc.reason,))
    except RouteSamplingError as exc:
        record = _fallback_or_reject(
            request,
            runtime_strategy,
            exc.reason,
            scope,
            trace_model,
            warnings=exc.warnings or (exc.reason,),
        )
        if str(exc.reason).startswith(("pd_handoff:", "session_migration:", "replication:", "kv_")):
            record = replace(
                record,
                kv_transfer_attempted_bytes=0.0,
                kv_transfer_delivered_bytes=0.0,
                kv_transfer_committed_page_bytes=0.0,
                kv_transfer_useful_bytes=0.0,
                kv_transfer_interrupted=not str(exc.reason).endswith("expired"),
                kv_transfer_expired=str(exc.reason).endswith("expired"),
                kv_transfer_interruption_reason=exc.reason,
                kv_serve_action=selected.get("kv_serve_action"),
                kv_post_turn_action=selected.get("kv_post_turn_action"),
            )
        partial_energy = _energy_summary_for_request(state, trace_bundle, request["request_id"])
        record = replace(
            record,
            soc_before=partial_energy["soc_before"],
            soc_after=partial_energy["soc_after"],
            energy_violation=partial_energy["energy_violation"],
            energy_ledger=tuple(partial_energy.get("energy_ledger", ())),
            provenance={
                **record.provenance,
                "energy_ledger": tuple(partial_energy.get("energy_ledger", ())),
                "energy_charged_for": "completed_events_before_route_failure",
            },
        )
        return record, _decision_trace(request, policy, access_candidates, selected.get("route_in_plan"), exc.reason, warnings=exc.warnings or (exc.reason,))


def _select_pd_candidate(
    *,
    state: RuntimeState,
    trace_bundle: Any,
    deployment_catalog: RuntimeDeploymentCatalog,
    request: dict[str, Any],
    ingress_id: str,
    access_candidates: tuple[Any, ...],
    policy: str,
    scheduler: Any,
    runtime_strategy: Any,
    max_hop_count: int | None,
    access_wait_s: float,
    arrival_s: float,
) -> dict[str, Any] | None:
    candidates, payload_by_id = _build_pd_scheduler_candidates(
        state=state,
        trace_bundle=trace_bundle,
        deployment_catalog=deployment_catalog,
        request=request,
        ingress_id=ingress_id,
        access_candidates=access_candidates,
        policy=policy,
        runtime_strategy=runtime_strategy,
        max_hop_count=max_hop_count,
        access_wait_s=access_wait_s,
        arrival_s=arrival_s,
    )
    if not candidates:
        return None
    decision = scheduler.select_action(
        state={
            "runtime_strategy": _jsonable_strategy(runtime_strategy),
            "candidates": tuple(candidate.to_dict() for candidate in candidates),
            "pd_mode": "disaggregated",
        },
        request=request,
    )
    selected_id = getattr(decision, "selected_candidate_id", None)
    if not getattr(decision, "evaluated", False) or selected_id not in payload_by_id:
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


def _build_pd_scheduler_candidates(
    *,
    state: RuntimeState,
    trace_bundle: Any,
    deployment_catalog: RuntimeDeploymentCatalog,
    request: dict[str, Any],
    ingress_id: str,
    access_candidates: tuple[Any, ...],
    policy: str,
    runtime_strategy: Any,
    max_hop_count: int | None,
    access_wait_s: float,
    arrival_s: float,
) -> tuple[tuple[SchedulerCandidate, ...], dict[str, dict[str, Any]]]:
    eligibility_time_s = arrival_s + access_wait_s
    prefill_deployments = tuple(getattr(deployment_catalog, "prefill_deployments", ()) or ())
    decode_candidates = tuple(getattr(deployment_catalog, "decode_deployments", ()) or ())
    prefill_eligibility = {
        deployment.deployment_id: _compute_eligibility_at(
            deployment=deployment,
            trace_bundle=trace_bundle,
            state=state,
            time_s=eligibility_time_s,
        )
        for deployment in prefill_deployments
    }
    decode_eligibility = {
        deployment.deployment_id: _compute_eligibility_at(
            deployment=deployment,
            trace_bundle=trace_bundle,
            state=state,
            time_s=eligibility_time_s,
        )
        for deployment in decode_candidates
    }
    prefill_by_entry = _deployments_by_entry(
        tuple(
            deployment
            for deployment in prefill_deployments
            if _deployment_entry_satellite(deployment)
            in prefill_eligibility[deployment.deployment_id].eligible_compute_satellite_ids
        )
    )
    decode_deployments = tuple(
        deployment
        for deployment in decode_candidates
        if _deployment_entry_satellite(deployment)
        in decode_eligibility[deployment.deployment_id].eligible_compute_satellite_ids
    )
    if not prefill_by_entry or not decode_deployments:
        return (), {}
    decode_by_id = {str(deployment.deployment_id): deployment for deployment in decode_deployments}
    route_failures: list[str] = []
    route_warnings: list[str] = []
    route_rows = enumerate_compute_candidates(
        trace_bundle,
        access_candidates,
        policy,
        max_hop_count=max_hop_count,
        time_s=arrival_s + access_wait_s,
        failures=route_failures,
        warnings=route_warnings,
        target_satellite_ids=tuple(prefill_by_entry),
    )
    candidates: list[SchedulerCandidate] = []
    payload_by_id: dict[str, dict[str, Any]] = {}
    for index, (access_candidate, compute_satellite_id, isl_link_ids) in enumerate(route_rows):
        for prefill in prefill_by_entry.get(str(compute_satellite_id), ()):
            try:
                route_in_plan = make_route_plan(
                    trace_bundle=trace_bundle,
                    access_candidate=access_candidate,
                    compute_satellite_id=str(compute_satellite_id),
                    isl_link_ids=isl_link_ids,
                    input_bytes=request["input_bytes"],
                    output_bytes=0.0,
                    time_s=arrival_s + access_wait_s,
                )
                inbound_done, inbound_wait, inbound_tx, sampled_inbound_legs = _preview_route(
                    state,
                    trace_bundle,
                    route_in_plan.inbound_legs,
                    arrival_s + access_wait_s,
                )
                route_in_plan = replace(route_in_plan, inbound_legs=sampled_inbound_legs)
            except (RouteSamplingError, ValueError):
                continue
            prefill_entry = _deployment_entry_satellite(prefill)
            prefill_queue = state.queue_for(prefill.deployment_id)
            paired_decode_ids = tuple(str(item) for item in getattr(prefill, "paired_deployment_ids", ()) or ())
            paired_decodes = tuple(
                decode_by_id[decode_id]
                for decode_id in paired_decode_ids
                if decode_id in decode_by_id
            )
            for decode in paired_decodes or decode_deployments:
                pair_scope = _pd_runtime_pair_scope(prefill, decode)
                if pair_scope is None:
                    continue
                pair_rule = _pd_runtime_pair_rule(prefill, decode) or pair_scope
                decode_entry = _deployment_entry_satellite(decode)
                try:
                    _deployment_member_energy_weights(prefill)
                    _deployment_member_energy_weights(decode)
                except ValueError as exc:
                    route_failures.append(str(exc))
                    continue
                for kv_serve_action in _pd_kv_serve_actions_for_decode(
                    state,
                    request,
                    decode,
                    runtime_strategy,
                ):
                    prefill_context = _pd_prefill_context_for_action(
                        state,
                        request,
                        decode,
                        runtime_strategy,
                        kv_serve_action=kv_serve_action,
                    )
                    prefill_tokens_effective = prefill_context["prefill_tokens_effective"]
                    kv_bytes_per_token = float(
                        _first_present(
                            getattr(decode, "kv_cache_bytes_per_token", None),
                            getattr(prefill, "kv_cache_bytes_per_token", None),
                            0.0,
                        )
                    )
                    kv_bytes = float(prefill_context["kv_generated_bytes"])
                    migration_bytes = float(prefill_context["kv_migration_bytes"])
                    migration_legs: tuple[Any, ...] = ()
                    sampled_migration_legs: tuple[Any, ...] = ()
                    migration_done_s = arrival_s
                    migration_wait = 0.0
                    migration_tx = 0.0
                    migration_start_s = arrival_s
                    migration_source = _optional_str(prefill_context.get("kv_source_satellite_id"))
                    if migration_bytes > 0:
                        if migration_source is None:
                            continue
                        migration_start_s = _pd_migration_start_s(state, request, arrival_s)
                        try:
                            migration_legs = _satellite_route_legs(
                                trace_bundle,
                                migration_source,
                                decode_entry,
                                migration_bytes,
                                time_s=migration_start_s,
                                max_hop_count=_pd_max_kv_hop_count(runtime_strategy, max_hop_count),
                            )
                            migration_done_s, migration_wait, migration_tx, sampled_migration_legs = _preview_route(
                                state,
                                trace_bundle,
                                migration_legs,
                                migration_start_s,
                            )
                        except RouteSamplingError:
                            continue
                    prefill_ready_for_action_s = max(inbound_done, migration_done_s)
                    prefill_start_s = max(prefill_ready_for_action_s, prefill_queue.prefill_available_s)
                    try:
                        raw_prefill_energy_window = energy_window_at(
                            trace_bundle,
                            prefill_entry,
                            prefill_start_s,
                        )
                        access_energy_window = energy_window_at(
                            trace_bundle,
                            route_in_plan.access_satellite_id,
                            arrival_s + access_wait_s,
                        )
                    except EnergyWindowError:
                        continue
                    prefill_profile = _pd_group_phase_profile(
                        trace_bundle=trace_bundle,
                        deployment=prefill,
                        start_s=prefill_start_s,
                        runtime_strategy=runtime_strategy,
                    )
                    prefill_action_type = str(prefill_profile["group_action_type"])
                    access_action_type = _pd_access_action_type(
                        route_in_plan=route_in_plan,
                        access_energy_window=access_energy_window,
                        prefill_energy_window=raw_prefill_energy_window,
                    )
                    if not _pd_phase_actions_allowed(
                        (access_action_type, *prefill_profile["member_actions"].values()),
                        runtime_strategy=runtime_strategy,
                    ):
                        continue
                    prefill_energy_window = prefill_profile["member_windows"][prefill_entry]["effective"]
                    prefill_dvfs = float(prefill_profile["dvfs_multiplier"])
                    prefill_service_s = _safe_service_time(
                        prefill_tokens_effective,
                        getattr(prefill, "per_group_prefill_tokens_per_s", None)
                        or getattr(prefill, "prefill_tokens_per_s", None),
                    ) * prefill_dvfs
                    if not math.isfinite(prefill_service_s):
                        continue
                    prefill_start_s, prefill_done_s, prefill_wait_s = prefill_queue.preview_prefill(
                        prefill_ready_for_action_s,
                        prefill_service_s,
                    )
                    try:
                        kv_legs = _satellite_route_legs(
                            trace_bundle,
                            prefill_entry,
                            decode_entry,
                            kv_bytes,
                            time_s=prefill_done_s,
                            max_hop_count=_pd_max_kv_hop_count(runtime_strategy, max_hop_count),
                        )
                        kv_done, kv_wait, kv_tx, sampled_kv_legs = _preview_route(
                            state,
                            trace_bundle,
                            kv_legs,
                            prefill_done_s,
                        )
                    except RouteSamplingError:
                        continue
                    decode_queue = state.queue_for(decode.deployment_id)
                    decode_start_s = max(kv_done, decode_queue.decode_available_s)
                    try:
                        raw_decode_energy_window = energy_window_at(
                            trace_bundle,
                            decode_entry,
                            decode_start_s,
                        )
                    except EnergyWindowError:
                        continue
                    decode_profile = _pd_group_phase_profile(
                        trace_bundle=trace_bundle,
                        deployment=decode,
                        start_s=decode_start_s,
                        runtime_strategy=runtime_strategy,
                    )
                    decode_action_type = str(decode_profile["group_action_type"])
                    if not _pd_phase_actions_allowed(
                        tuple(decode_profile["member_actions"].values()),
                        runtime_strategy=runtime_strategy,
                    ):
                        continue
                    decode_energy_window = decode_profile["member_windows"][decode_entry]["effective"]
                    decode_dvfs = float(decode_profile["dvfs_multiplier"])
                    decode_rate = getattr(decode, "per_group_decode_tokens_per_s", None) or getattr(decode, "decode_tokens_per_s", None)
                    full_decode_service_s = _safe_service_time(request["output_tokens"], decode_rate) * decode_dvfs
                    if not math.isfinite(full_decode_service_s):
                        continue
                    first_token_service_s = min(
                        full_decode_service_s,
                        _safe_service_time(1, decode_rate) * decode_dvfs,
                    )
                    decode_start_s, decode_done_s, decode_wait_s = decode_queue.preview_decode(kv_done, full_decode_service_s)
                    first_token_ready_s = decode_start_s + first_token_service_s
                    first_token_bytes = _first_token_bytes(request)
                    remaining_output_bytes = max(0.0, request["output_bytes"] - first_token_bytes)
                    first_plan, first_reason, _first_warnings = _build_post_compute_outbound_plan(
                        trace_bundle=trace_bundle,
                        ingress_id=ingress_id,
                        compute_satellite_id=decode_entry,
                        policy=policy,
                        max_hop_count=max_hop_count,
                        output_bytes=first_token_bytes,
                        decode_done_s=first_token_ready_s,
                    )
                    if first_plan is None:
                        continue
                    try:
                        first_done, first_wait, first_tx, sampled_first_legs = _preview_route(
                            state,
                            trace_bundle,
                            first_plan.outbound_legs,
                            first_token_ready_s,
                        )
                    except RouteSamplingError:
                        continue
                    first_plan = replace(first_plan, outbound_legs=sampled_first_legs)
                    remaining_plan = None
                    remaining_done = first_done
                    remaining_wait = 0.0
                    remaining_tx = 0.0
                    if remaining_output_bytes > 0:
                        remaining_plan, _remaining_reason, _remaining_warnings = _build_post_compute_outbound_plan(
                            trace_bundle=trace_bundle,
                            ingress_id=ingress_id,
                            compute_satellite_id=decode_entry,
                            policy=policy,
                            max_hop_count=max_hop_count,
                            output_bytes=remaining_output_bytes,
                            decode_done_s=decode_done_s,
                        )
                        if remaining_plan is None:
                            continue
                        try:
                            remaining_done, remaining_wait, remaining_tx, sampled_remaining_legs = _preview_route(
                                state,
                                trace_bundle,
                                remaining_plan.outbound_legs,
                                decode_done_s,
                            )
                        except RouteSamplingError:
                            continue
                        remaining_plan = replace(remaining_plan, outbound_legs=sampled_remaining_legs)
                    context_tokens_after = _pd_context_tokens_after(
                        request,
                        prefill_context=prefill_context,
                    )
                    for post_turn in _pd_kv_post_turn_candidates(
                        state,
                        request,
                        decode,
                        deployment_catalog,
                        runtime_strategy,
                    ):
                        kv_post_turn_action = str(post_turn["action"])
                        replica_target = post_turn.get("target_deployment")
                        replication_bytes = (
                            context_tokens_after * kv_bytes_per_token
                            if kv_post_turn_action == "replicate_kv" and replica_target is not None
                            else 0.0
                        )
                        replication_legs: tuple[Any, ...] = ()
                        sampled_replication_legs: tuple[Any, ...] = ()
                        replication_wait = 0.0
                        replication_tx = 0.0
                        if replication_bytes > 0 and replica_target is not None:
                            try:
                                replication_target_entry = _deployment_entry_satellite(replica_target)
                                replication_legs = _satellite_route_legs(
                                    trace_bundle,
                                    decode_entry,
                                    replication_target_entry,
                                    replication_bytes,
                                    time_s=decode_done_s,
                                    max_hop_count=_pd_max_kv_hop_count(runtime_strategy, max_hop_count),
                                )
                                _rep_done, replication_wait, replication_tx, sampled_replication_legs = _preview_route(
                                    state,
                                    trace_bundle,
                                    replication_legs,
                                    decode_done_s,
                                )
                            except RouteSamplingError:
                                continue
                        kv_capacity_result = _pd_kv_capacity_result(
                            state,
                            request,
                            decode,
                            context_tokens_after,
                            runtime_strategy=runtime_strategy,
                            kv_bytes_per_token=kv_bytes_per_token,
                            kv_post_turn_action=kv_post_turn_action,
                            replica_target_deployment=replica_target,
                        )
                        predicted_ttft = first_done - arrival_s
                        predicted_e2e = remaining_done - arrival_s
                        predicted_queue_wait = (
                            prefill_wait_s
                            + decode_wait_s
                            + inbound_wait
                            + migration_wait
                            + kv_wait
                            + first_wait
                            + remaining_wait
                            + replication_wait
                        )
                        predicted_link_wait = inbound_wait + migration_wait + kv_wait + first_wait + remaining_wait + replication_wait
                        predicted_energy_risk, predicted_soc_after, pd_energy_gate = _pd_energy_prediction(
                            state=state,
                            trace_bundle=trace_bundle,
                            prefill_entry=prefill_entry,
                            decode_entry=decode_entry,
                            prefill_deployment=prefill,
                            decode_deployment=decode,
                            prefill_energy_window=prefill_energy_window,
                            decode_energy_window=decode_energy_window,
                            prefill_start_s=prefill_start_s,
                            decode_start_s=decode_start_s,
                            arrival_s=arrival_s,
                            prefill_ready_floor_s=arrival_s + access_wait_s,
                            prefill_queue_available_s=prefill_queue.prefill_available_s,
                            decode_queue_available_s=decode_queue.decode_available_s,
                            prefill_base_service_s=_safe_service_time(
                                prefill_tokens_effective,
                                getattr(prefill, "per_group_prefill_tokens_per_s", None)
                                or getattr(prefill, "prefill_tokens_per_s", None),
                            ),
                            decode_base_service_s=_safe_service_time(request["output_tokens"], decode_rate),
                            first_token_base_service_s=_safe_service_time(1, decode_rate),
                            output_tokens=request["output_tokens"],
                            prefill_tokens_effective=prefill_tokens_effective,
                            runtime_strategy=runtime_strategy,
                            prefill_action_type=prefill_action_type,
                            decode_action_type=decode_action_type,
                            pre_prefill_routes=(
                                (sampled_migration_legs, migration_start_s, "session_migration"),
                                (route_in_plan.inbound_legs, arrival_s + access_wait_s, "input_route"),
                            ),
                            between_phase_routes=(
                                (sampled_kv_legs, prefill_done_s, "pd_handoff"),
                            ),
                            post_decode_routes=(
                                (first_plan.outbound_legs, first_token_ready_s, "first_token_route"),
                                *(
                                    ((remaining_plan.outbound_legs, decode_done_s, "remaining_output_route"),)
                                    if remaining_plan is not None
                                    else ()
                                ),
                                (sampled_replication_legs, decode_done_s, "replication"),
                            ),
                        )
                        prefill_profile = pd_energy_gate["prefill_profile"]
                        decode_profile = pd_energy_gate["decode_profile"]
                        prefill_action_type = str(prefill_profile["group_action_type"])
                        decode_action_type = str(decode_profile["group_action_type"])
                        raw_prefill_energy_window = prefill_profile["member_windows"][prefill_entry]["raw"]
                        prefill_energy_window = prefill_profile["member_windows"][prefill_entry]["effective"]
                        raw_decode_energy_window = decode_profile["member_windows"][decode_entry]["raw"]
                        decode_energy_window = decode_profile["member_windows"][decode_entry]["effective"]
                        if not _pd_phase_actions_allowed(
                            (
                                access_action_type,
                                *prefill_profile["member_actions"].values(),
                                *decode_profile["member_actions"].values(),
                            ),
                            runtime_strategy=runtime_strategy,
                        ):
                            continue
                        prefill_start_s = float(pd_energy_gate["prefill_start_s"])
                        decode_start_s = float(pd_energy_gate["decode_start_s"])
                        prefill_done_s = float(pd_energy_gate["prefill_done_s"])
                        decode_done_s = float(pd_energy_gate["decode_done_s"])
                        prefill_service_s = float(pd_energy_gate["prefill_service_s"])
                        full_decode_service_s = float(pd_energy_gate["decode_service_s"])
                        first_token_service_s = float(pd_energy_gate["first_token_service_s"])
                        first_token_ready_s = float(pd_energy_gate["first_token_ready_s"])
                        predicted_ttft = float(pd_energy_gate["predicted_ttft_s"])
                        predicted_e2e = float(pd_energy_gate["predicted_e2e_s"])
                        predicted_link_wait = float(pd_energy_gate["route_wait_s"])
                        predicted_queue_wait = (
                            max(0.0, prefill_start_s - max(arrival_s + access_wait_s, pd_energy_gate.get("route_results", {}).get("input_route", {}).get("done_s", arrival_s)))
                            + max(0.0, decode_start_s - max(prefill_done_s, pd_energy_gate.get("route_results", {}).get("pd_handoff", {}).get("done_s", prefill_done_s)))
                            + predicted_link_wait
                        )
                        session_action = _pd_session_action_compat(kv_serve_action, kv_post_turn_action)
                        hard_gate_results = _candidate_hard_gate_results(
                            request=request,
                            predicted_ttft_s=predicted_ttft,
                            predicted_e2e_s=predicted_e2e,
                            predicted_soc_after=predicted_soc_after,
                            soc_min=0.0,
                        )
                        hard_gate_results["energy_soc"] = pd_energy_gate
                        hard_gate_results["kv_capacity"] = kv_capacity_result
                        hard_gate_violations = _candidate_hard_gate_violations(hard_gate_results)
                        candidate_id = (
                            f"pd-candidate-{index}:{prefill.deployment_id}->{decode.deployment_id}:"
                            f"{kv_serve_action}+{kv_post_turn_action}"
                        )
                        route_dict = route_in_plan.to_dict()
                        route_dict["pd"] = {
                            "prefill_deployment_id": prefill.deployment_id,
                            "decode_deployment_id": decode.deployment_id,
                            "migration_route": tuple(leg.to_dict() for leg in sampled_migration_legs),
                            "kv_route": tuple(leg.to_dict() for leg in sampled_kv_legs),
                            "replication_route": tuple(leg.to_dict() for leg in sampled_replication_legs),
                            "first_token_route": first_plan.to_dict(),
                            "remaining_route": remaining_plan.to_dict() if remaining_plan is not None else None,
                        }
                        provenance = {
                            "access_satellite_id": route_in_plan.access_satellite_id,
                            "prefill_entry_satellite_id": prefill_entry,
                            "decode_entry_satellite_id": decode_entry,
                            "prefill_deployment_id": prefill.deployment_id,
                            "decode_deployment_id": decode.deployment_id,
                            "pd_placement_scope": pair_scope,
                            "resolved_pd_pair_scope": pair_scope,
                            "pd_pairing_rule": pair_rule,
                            "pd_placement_candidate_scope": getattr(prefill, "placement_scope", None),
                            "kv_serve_action": kv_serve_action,
                            "kv_post_turn_action": kv_post_turn_action,
                            "kv_handoff_bytes": kv_bytes,
                            "kv_migration_bytes": migration_bytes,
                            "kv_replication_bytes": replication_bytes,
                            "kv_hit_tokens": prefill_context["kv_hit_tokens"],
                            "kv_miss_tokens": prefill_context["kv_miss_tokens"],
                            "context_recompute_tokens": prefill_context["historical_recompute_tokens"],
                            "kv_migration_visible_s": max(0.0, migration_done_s - arrival_s) if migration_bytes > 0 else 0.0,
                            "hbm_occupancy_after": kv_capacity_result.get("occupancy_after"),
                            "session_action": session_action,
                            "prefill_tokens_effective": prefill_tokens_effective,
                            "token_context_contract": prefill_context["token_context_contract"],
                            "kv_capacity_admission": kv_capacity_result["admission"],
                            "compute_eligibility": {
                                "prefill": prefill_eligibility[prefill.deployment_id].to_dict(),
                                "decode": decode_eligibility[decode.deployment_id].to_dict(),
                            },
                            "access_action_type": access_action_type,
                            "prefill_action_type": prefill_action_type,
                            "decode_action_type": decode_action_type,
                            "access_sunlit": _energy_window_sunlit(access_energy_window),
                            "compute_sunlit": bool(prefill_profile["all_sunlit"] and decode_profile["all_sunlit"]),
                            "shadow_strategies": tuple(
                                dict.fromkeys(
                                    action
                                    for action in (
                                        access_action_type,
                                        *prefill_profile["member_actions"].values(),
                                        *decode_profile["member_actions"].values(),
                                    )
                                    if action in {"shadow_offload", "shadow_dvfs", "shadow_local_compute"}
                                )
                            ),
                            "pd_energy_gate": pd_energy_gate,
                            "shadow_triggered": bool(
                                _energy_window_sunlit(access_energy_window) is False
                                or not prefill_profile["all_sunlit"]
                                or not decode_profile["all_sunlit"]
                            ),
                            "prefill_member_actions": dict(prefill_profile["member_actions"]),
                            "decode_member_actions": dict(decode_profile["member_actions"]),
                            "predicted_prefill_start_s": prefill_start_s,
                            "predicted_decode_start_s": decode_start_s,
                        }
                        candidate = SchedulerCandidate(
                            candidate_id=candidate_id,
                            action_type="pd_disaggregated",
                            feasible=True,
                            route_plan=route_dict,
                            predicted_ttft_s=predicted_ttft,
                            predicted_e2e_s=predicted_e2e,
                            predicted_queue_wait_s=predicted_queue_wait,
                            predicted_link_wait_s=predicted_link_wait,
                            predicted_energy_risk=predicted_energy_risk,
                            predicted_soc_after=predicted_soc_after,
                            hop_count=route_in_plan.hop_count + len(sampled_migration_legs) + len(sampled_kv_legs) + len(sampled_replication_legs),
                            hard_gate_violations=hard_gate_violations,
                            hard_gate_results=hard_gate_results,
                            predicted_nominal_latency_s=predicted_e2e,
                            provenance=provenance,
                        )
                        payload_by_id[candidate_id] = {
                            "candidate_id": candidate_id,
                            "action_type": "pd_disaggregated",
                            "route_in_plan": route_in_plan,
                            "migration_legs": sampled_migration_legs,
                            "migration_start_s": migration_start_s,
                            "kv_legs": sampled_kv_legs,
                            "replication_legs": sampled_replication_legs,
                            "replication_target_deployment": replica_target,
                            "first_plan": first_plan,
                            "remaining_plan": remaining_plan,
                            "prefill": prefill,
                            "decode": decode,
                            "resolved_pd_pair_scope": pair_scope,
                            "pd_pairing_rule": pair_rule,
                            "prefill_entry": prefill_entry,
                            "decode_entry": decode_entry,
                            "prefill_service_s": prefill_service_s,
                            "full_decode_service_s": full_decode_service_s,
                            "first_token_service_s": first_token_service_s,
                            "prefill_tokens_effective": prefill_tokens_effective,
                            "prefill_context": prefill_context,
                            "session_action": session_action,
                            "kv_serve_action": kv_serve_action,
                            "kv_post_turn_action": kv_post_turn_action,
                            "context_tokens_after": context_tokens_after,
                            "kv_handoff_bytes": kv_bytes,
                            "kv_migration_bytes": migration_bytes,
                            "kv_replication_bytes": replication_bytes,
                            "kv_capacity_result": kv_capacity_result,
                            "kv_capacity_admission": kv_capacity_result["admission"],
                            "predicted_ttft_s": predicted_ttft,
                            "predicted_e2e_s": predicted_e2e,
                            "predicted_nominal_latency_s": predicted_e2e,
                            "predicted_queue_wait_s": predicted_queue_wait,
                            "predicted_link_wait_s": predicted_link_wait,
                            "predicted_soc_after": predicted_soc_after,
                            "predicted_energy_risk": predicted_energy_risk,
                            "prefill_energy_window": prefill_energy_window,
                            "decode_energy_window": decode_energy_window,
                            "raw_prefill_energy_window": raw_prefill_energy_window,
                            "raw_decode_energy_window": raw_decode_energy_window,
                            "access_energy_window": access_energy_window,
                            "access_action_type": access_action_type,
                            "prefill_action_type": prefill_action_type,
                            "decode_action_type": decode_action_type,
                            "prefill_start_s": prefill_start_s,
                            "decode_start_s": decode_start_s,
                            "shadow_strategies": provenance["shadow_strategies"],
                            "pd_energy_gate": pd_energy_gate,
                            "scheduler_candidate": candidate.to_dict(),
                            "hard_gate_violations": hard_gate_violations,
                            "hard_gate_results": hard_gate_results,
                            "compute_eligibility": provenance["compute_eligibility"],
                        }
                        candidates.append(candidate)
    return tuple(candidates), payload_by_id


def _reserve_selected_pd_candidate(
    *,
    state: RuntimeState,
    request: dict[str, Any],
    selected: dict[str, Any],
    trace_bundle: Any,
    deployment_id: str,
    policy: str,
    runtime_strategy: Any,
    scope: str,
    trace_model: str,
    access_wait_s: float,
    deferred: bool,
    defer_until_s: float | None,
) -> tuple[RuntimeRequestRecord, dict[str, Any]]:
    arrival_s = request["arrival_time_s"]
    route_in_plan = selected["route_in_plan"]
    prefill = selected["prefill"]
    decode = selected["decode"]
    prefill_entry = selected["prefill_entry"]
    decode_entry = selected["decode_entry"]
    kv_policy = _kv_policy(runtime_strategy)
    migration_result = KVTransferResult(done_s=arrival_s)
    if float(selected.get("kv_migration_bytes") or 0.0) > 0:
        migration_source = _optional_str(selected["prefill_context"].get("kv_source_satellite_id"))
        if migration_source is None:
            raise RouteSamplingError("kv_migration_source_missing", sample_time_s=arrival_s)
        migration_result = _reserve_kv_transfer(
            state=state,
            trace_bundle=trace_bundle,
            source_satellite_id=migration_source,
            target_satellite_id=decode_entry,
            payload_bytes=float(selected["kv_migration_bytes"]),
            start_s=float(selected.get("migration_start_s") or arrival_s),
            max_hop_count=_pd_max_kv_hop_count(runtime_strategy, getattr(runtime_strategy, "max_hop_count", None)),
            chunk_bytes=_kv_transfer_chunk_bytes(kv_policy),
            flow_kind="session_migration",
            deadline_s=_request_kv_deadline_s(request, arrival_s),
            expires_at_s=_session_expires_at(state, request.get("session_id")),
            request_id=request["request_id"],
        )
        if migration_result.interrupted or migration_result.expired:
            raise KVTransferRuntimeError(
                migration_result.interruption_reason or migration_result.expiry_reason or "kv_migration_failed",
                transfer=migration_result,
                kv_serve_action=selected.get("kv_serve_action"),
                kv_post_turn_action=selected.get("kv_post_turn_action"),
            )
    inbound_done, inbound_path, sgl_wait_in, isl_wait_in, tx_in, sampled_inbound = _reserve_route(
        state,
        trace_bundle,
        route_in_plan.inbound_legs,
        arrival_s + access_wait_s,
        request_id=request["request_id"], energy_stage="input_route",
    )
    route_in_plan = replace(route_in_plan, inbound_legs=sampled_inbound)
    prefill_queue = state.queue_for(prefill.deployment_id)
    prefill_ready_s = max(inbound_done, migration_result.done_s)
    actual_prefill_start_s = max(prefill_ready_s, prefill_queue.prefill_available_s)
    prefill_profile = _pd_group_phase_profile(
        trace_bundle=trace_bundle,
        deployment=prefill,
        start_s=actual_prefill_start_s,
        runtime_strategy=runtime_strategy,
    )
    prefill_action_type = str(prefill_profile["group_action_type"])
    raw_prefill_energy_window = prefill_profile["member_windows"][prefill_entry]["raw"]
    prefill_energy_window = prefill_profile["member_windows"][prefill_entry]["effective"]
    prefill_service_s = _safe_service_time(
        selected["prefill_tokens_effective"],
        getattr(prefill, "per_group_prefill_tokens_per_s", None)
        or getattr(prefill, "prefill_tokens_per_s", None),
    ) * float(prefill_profile["dvfs_multiplier"])
    prefill_start_s, prefill_done_s, prefill_wait_s = prefill_queue.reserve_prefill(
        prefill_ready_s,
        prefill_service_s,
    )
    prefill_phase_energy = _apply_pd_compute_phase_energy(
        state=state,
        trace_bundle=trace_bundle,
        deployment=prefill,
        start_s=prefill_start_s,
        tokens=selected["prefill_tokens_effective"],
        phase="prefill",
        runtime_strategy=runtime_strategy,
        request_id=request["request_id"],
    )
    kv_result = _reserve_kv_transfer(
        state=state,
        trace_bundle=trace_bundle,
        source_satellite_id=prefill_entry,
        target_satellite_id=decode_entry,
        payload_bytes=float(selected["kv_handoff_bytes"] or 0.0),
        start_s=prefill_done_s,
        max_hop_count=_pd_max_kv_hop_count(runtime_strategy, getattr(runtime_strategy, "max_hop_count", None)),
        chunk_bytes=_kv_transfer_chunk_bytes(kv_policy),
        flow_kind="pd_handoff",
        deadline_s=_request_kv_deadline_s(request, arrival_s),
        expires_at_s=None,
        request_id=request["request_id"],
    )
    if kv_result.interrupted or kv_result.expired:
        raise KVTransferRuntimeError(
            kv_result.interruption_reason or kv_result.expiry_reason or "kv_handoff_failed",
            transfer=kv_result,
            kv_serve_action=selected.get("kv_serve_action"),
            kv_post_turn_action=selected.get("kv_post_turn_action"),
        )
    decode_queue = state.queue_for(decode.deployment_id)
    actual_decode_start_s = max(kv_result.done_s, decode_queue.decode_available_s)
    decode_profile = _pd_group_phase_profile(
        trace_bundle=trace_bundle,
        deployment=decode,
        start_s=actual_decode_start_s,
        runtime_strategy=runtime_strategy,
    )
    decode_action_type = str(decode_profile["group_action_type"])
    raw_decode_energy_window = decode_profile["member_windows"][decode_entry]["raw"]
    decode_energy_window = decode_profile["member_windows"][decode_entry]["effective"]
    decode_rate = getattr(decode, "per_group_decode_tokens_per_s", None) or getattr(
        decode,
        "decode_tokens_per_s",
        None,
    )
    full_decode_service_s = _safe_service_time(request["output_tokens"], decode_rate) * float(
        decode_profile["dvfs_multiplier"]
    )
    first_token_service_s = min(
        full_decode_service_s,
        _safe_service_time(1, decode_rate) * float(decode_profile["dvfs_multiplier"]),
    )
    decode_start_s, decode_done_s, decode_wait_s = decode_queue.reserve_decode(
        kv_result.done_s,
        full_decode_service_s,
    )
    decode_phase_energy = _apply_pd_compute_phase_energy(
        state=state,
        trace_bundle=trace_bundle,
        deployment=decode,
        start_s=decode_start_s,
        tokens=request["output_tokens"],
        phase="decode",
        runtime_strategy=runtime_strategy,
        request_id=request["request_id"],
    )
    first_token_ready_s = decode_start_s + first_token_service_s
    first_done, first_path, sgl_wait_first, isl_wait_first, tx_first, _sampled_first = _reserve_route(
        state,
        trace_bundle,
        selected["first_plan"].outbound_legs,
        first_token_ready_s,
        request_id=request["request_id"], energy_stage="first_token_route",
    )
    remaining_plan = selected.get("remaining_plan")
    remaining_done = first_done
    remaining_path: tuple[dict[str, Any], ...] = ()
    sgl_wait_remaining = 0.0
    isl_wait_remaining = 0.0
    tx_remaining = 0.0
    if remaining_plan is not None:
        try:
            remaining_done, remaining_path, sgl_wait_remaining, isl_wait_remaining, tx_remaining, _sampled_remaining = _reserve_route(
                state,
                trace_bundle,
                remaining_plan.outbound_legs,
                decode_done_s,
                request_id=request["request_id"], energy_stage="remaining_output_route",
            )
        except RouteSamplingError as exc:
            partial_path = tuple((*migration_result.path, *inbound_path, *kv_result.path, *first_path))
            partial_energy_update = _energy_summary_for_events(
                state,
                trace_bundle,
                [
                    *prefill_phase_energy["ledger_entries"],
                    *decode_phase_energy["ledger_entries"],
                    *_path_energy_entries(partial_path),
                ],
            )
            partial_timing = ttft_breakdown(
                access_wait_s=access_wait_s,
                uplink_s=max(0.0, inbound_done - (arrival_s + access_wait_s)),
                prefill_queue_s=prefill_wait_s,
                prefill_compute_s=prefill_service_s,
                kv_transfer_s=max(0.0, migration_result.done_s - inbound_done) + max(0.0, kv_result.done_s - prefill_done_s),
                decode_queue_s=decode_wait_s,
                first_token_decode_s=first_token_service_s,
                return_path_s=max(0.0, first_done - first_token_ready_s),
            )
            record = _post_compute_route_failure_record(
                request=request,
                deployment_id=deployment_id,
                action_type="pd_disaggregated",
                route_plan=route_in_plan,
                inbound_path=partial_path,
                arrival_s=arrival_s,
                access_wait_s=access_wait_s,
                prefill_wait_s=prefill_wait_s,
                decode_wait_s=decode_wait_s,
                sgl_wait_in=sgl_wait_in + migration_result.sgl_wait_s + kv_result.sgl_wait_s + sgl_wait_first,
                isl_wait_in=isl_wait_in + migration_result.isl_wait_s + kv_result.isl_wait_s + isl_wait_first,
                tx_in=tx_in + migration_result.transmission_s + kv_result.transmission_s + tx_first,
                prefill_service_s=prefill_service_s,
                decode_service_s=full_decode_service_s,
                prefill_done_s=prefill_done_s,
                compute_satellite_id=decode_entry,
                access_satellite_id=route_in_plan.access_satellite_id,
                energy_update=partial_energy_update,
                deferred=deferred,
                defer_until_s=defer_until_s,
                session_action=selected.get("session_action") or "none",
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
            record = replace(
                record,
                ttft_s=float(partial_timing["ttft_s"]),
                uplink_s=float(partial_timing["uplink_s"]),
                kv_transfer_s=float(partial_timing["kv_transfer_s"]),
                first_token_decode_s=first_token_service_s,
                return_path_s=float(partial_timing["return_path_s"]),
                ttft_breakdown_s=partial_timing,
                first_token_return_s=float(partial_timing["return_path_s"]),
                assigned_satellite_id=decode_entry,
                compute_owner_satellite_id=decode_entry,
                compute_owner_satellite_ids=tuple(
                    dict.fromkeys(
                        (*tuple(prefill.member_satellites), *tuple(decode.member_satellites))
                    )
                ),
                prefill_deployment_id=prefill.deployment_id,
                decode_deployment_id=decode.deployment_id,
                prefill_entry_satellite_id=prefill_entry,
                decode_entry_satellite_id=decode_entry,
                response_streaming_model="first_token_plus_bulk_remaining",
                route_node_ids=_route_node_ids(partial_path),
                provenance={
                    **record.provenance,
                    "compute_eligibility": selected.get("compute_eligibility"),
                    "ttft_definition": TTFT_DEFINITION,
                    "partial_response_stage": "remaining_output_return",
                },
            )
            return record, _decision_trace(
                request,
                policy,
                (),
                route_in_plan,
                "route_became_unavailable_after_compute",
                warnings=exc.warnings or (exc.reason,),
                scheduler_decision=selected.get("scheduler_decision"),
            )
    replication_result = KVTransferResult(done_s=decode_done_s)
    replication_target = selected.get("replication_target_deployment")
    if float(selected.get("kv_replication_bytes") or 0.0) > 0 and replication_target is not None:
        replication_result = _reserve_kv_transfer(
            state=state,
            trace_bundle=trace_bundle,
            source_satellite_id=decode_entry,
            target_satellite_id=_deployment_entry_satellite(replication_target),
            payload_bytes=float(selected["kv_replication_bytes"]),
            start_s=decode_done_s,
            max_hop_count=_pd_max_kv_hop_count(runtime_strategy, getattr(runtime_strategy, "max_hop_count", None)),
            chunk_bytes=_kv_transfer_chunk_bytes(kv_policy),
            flow_kind="replication",
            deadline_s=None,
            expires_at_s=_session_ttl_deadline(runtime_strategy, decode_done_s),
            request_id=request["request_id"],
        )
    replication_succeeded = bool(
        selected.get("kv_post_turn_action") == "replicate_kv"
        and replication_target is not None
        and not replication_result.interrupted
        and not replication_result.expired
    )
    state_post_turn_action = (
        "replicate_kv"
        if replication_succeeded
        else ("evict_session" if selected.get("kv_post_turn_action") == "evict_session" else "none")
    )
    selected_path = tuple((*migration_result.path, *inbound_path, *kv_result.path, *first_path, *remaining_path, *replication_result.path))
    pd_energy_update = _energy_summary_for_events(
        state,
        trace_bundle,
        [
            *prefill_phase_energy["ledger_entries"],
            *decode_phase_energy["ledger_entries"],
            *_path_energy_entries(selected_path),
        ],
    )
    ttft_s = first_done - arrival_s
    e2e_s = remaining_done - arrival_s
    timeout, timeout_reason = _timeout_status(ttft_s, e2e_s, request["ttft_deadline_s"], request["e2e_deadline_s"])
    energy_violation = bool(pd_energy_update["energy_violation"])
    session_action = selected.get("session_action") or _pd_session_action_preview(
        state,
        request.get("session_id"),
        decode,
    )
    _update_pd_session_state(
        state=state,
        request=request,
        runtime_strategy=runtime_strategy,
        prefill=prefill,
        decode=decode,
        session_action=session_action,
        prefill_context=selected["prefill_context"],
        context_tokens_after=selected["context_tokens_after"],
        kv_post_turn_action=state_post_turn_action,
        replica_target_deployment=replication_target if replication_succeeded else None,
        now_s=decode_done_s,
    )
    _sync_kv_cache_state_for_deployment(state, decode, runtime_strategy)
    if replication_target is not None:
        _sync_kv_cache_state_for_deployment(state, replication_target, runtime_strategy)
    served = not timeout and not energy_violation
    total_sgl_wait = sgl_wait_in + migration_result.sgl_wait_s + kv_result.sgl_wait_s + sgl_wait_first + sgl_wait_remaining + replication_result.sgl_wait_s
    total_isl_wait = isl_wait_in + migration_result.isl_wait_s + kv_result.isl_wait_s + isl_wait_first + isl_wait_remaining + replication_result.isl_wait_s
    bottleneck = _bottleneck(timeout_reason, energy_violation, prefill_wait_s, decode_wait_s, total_sgl_wait + total_isl_wait)
    kv_visible_s = _route_visible_s(kv_result.path)
    migration_visible_s = max(0.0, migration_result.done_s - arrival_s) if selected.get("kv_serve_action") == "migrate_kv" else 0.0
    migration_total_s = max(0.0, migration_result.done_s - float(selected.get("migration_start_s") or arrival_s))
    migration_hidden_s = max(0.0, migration_total_s - migration_visible_s)
    first_token_return_s = first_done - first_token_ready_s
    uplink_s = max(0.0, inbound_done - (arrival_s + access_wait_s))
    blocking_migration_s = max(0.0, migration_result.done_s - inbound_done)
    ttft_kv_transfer_s = blocking_migration_s + max(0.0, kv_result.done_s - prefill_done_s)
    timing = ttft_breakdown(
        access_wait_s=access_wait_s,
        uplink_s=uplink_s,
        prefill_queue_s=prefill_wait_s,
        prefill_compute_s=prefill_service_s,
        kv_transfer_s=ttft_kv_transfer_s,
        decode_queue_s=decode_wait_s,
        first_token_decode_s=first_token_service_s,
        return_path_s=first_token_return_s,
    )
    ttft_s = float(timing["ttft_s"])
    timeout, timeout_reason = _timeout_status(
        ttft_s,
        e2e_s,
        request["ttft_deadline_s"],
        request["e2e_deadline_s"],
    )
    served = not timeout and not energy_violation
    bottleneck = _bottleneck(timeout_reason, energy_violation, prefill_wait_s, decode_wait_s, total_sgl_wait + total_isl_wait)
    kv_capacity_result = _dict(selected.get("kv_capacity_result"))
    transfer_attempted = migration_result.attempted_bytes + kv_result.attempted_bytes + replication_result.attempted_bytes
    transfer_delivered = migration_result.delivered_bytes + kv_result.delivered_bytes + replication_result.delivered_bytes
    transfer_committed = migration_result.committed_page_bytes + kv_result.committed_page_bytes + replication_result.committed_page_bytes
    transfer_useful = migration_result.useful_bytes + kv_result.useful_bytes + replication_result.useful_bytes
    transfer_interrupted = bool(migration_result.interrupted or kv_result.interrupted or replication_result.interrupted)
    transfer_expired = bool(migration_result.expired or kv_result.expired or replication_result.expired)
    access_action_type = str(selected.get("access_action_type") or "local_compute")
    member_shadow_actions = (
        *tuple(prefill_phase_energy["shadow_strategies"]),
        *tuple(decode_phase_energy["shadow_strategies"]),
    )
    shadow_strategies = tuple(
        dict.fromkeys(
            action
            for action in (access_action_type, *member_shadow_actions)
            if action in {"shadow_offload", "shadow_dvfs", "shadow_local_compute"}
        )
    )
    action_components = (
        "pd_disaggregated",
        f"access:{access_action_type}",
        *(
            f"prefill:{satellite_id}:{action}"
            for satellite_id, action in prefill_phase_energy["member_actions"].items()
        ),
        *(
            f"decode:{satellite_id}:{action}"
            for satellite_id, action in decode_phase_energy["member_actions"].items()
        ),
        *(("kv:cross_satellite_forward",) if kv_result.path else ()),
    )
    record = RuntimeRequestRecord(
        request_id=request["request_id"],
        arrival_time_s=arrival_s,
        deployment_id=deployment_id,
        action_type="pd_disaggregated",
        action_components=action_components,
        accepted=True,
        completed=True,
        served=served,
        rejected=False,
        timeout=timeout,
        timeout_reason=timeout_reason,
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        ttft_deadline_s=request["ttft_deadline_s"],
        e2e_deadline_s=request["e2e_deadline_s"],
        prefill_queue_wait_s=prefill_wait_s,
        decode_queue_wait_s=decode_wait_s,
        sgl_queue_wait_s=total_sgl_wait,
        isl_queue_wait_s=total_isl_wait,
        link_queue_wait_s=total_sgl_wait + total_isl_wait,
        access_wait_s=access_wait_s,
        link_transmission_s=tx_in + migration_result.transmission_s + kv_result.transmission_s + tx_first + tx_remaining + replication_result.transmission_s,
        prefill_service_s=prefill_service_s,
        decode_service_s=full_decode_service_s,
        uplink_s=uplink_s,
        kv_transfer_s=ttft_kv_transfer_s,
        first_token_decode_s=first_token_service_s,
        return_path_s=first_token_return_s,
        ttft_breakdown_s=timing,
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        assigned_satellite_id=decode_entry,
        compute_owner_satellite_id=decode_entry,
        compute_owner_satellite_ids=tuple(
            dict.fromkeys(
                (*tuple(prefill.member_satellites), *tuple(decode.member_satellites))
            )
        ),
        access_satellite_id=route_in_plan.access_satellite_id,
        prefill_deployment_id=prefill.deployment_id,
        decode_deployment_id=decode.deployment_id,
        prefill_entry_satellite_id=prefill_entry,
        decode_entry_satellite_id=decode_entry,
        pd_placement_scope=selected.get("resolved_pd_pair_scope") or getattr(prefill, "placement_scope", None),
        resolved_pd_pair_scope=selected.get("resolved_pd_pair_scope"),
        pd_pairing_rule=selected.get("pd_pairing_rule"),
        kv_handoff_bytes=selected["kv_handoff_bytes"],
        kv_handoff_queue_wait_s=kv_result.isl_wait_s,
        kv_handoff_transmission_s=kv_result.transmission_s,
        kv_handoff_propagation_s=_route_component_s(kv_result.path, "propagation_delay_s"),
        kv_handoff_setup_s=_route_component_s(kv_result.path, "setup_delay_s"),
        kv_handoff_visible_s=kv_visible_s,
        first_token_return_s=first_token_return_s,
        session_kv_migration_bytes=selected.get("kv_migration_bytes"),
        session_kv_visible_s=migration_visible_s,
        kv_capacity_admission=selected.get("kv_capacity_admission"),
        kv_serve_action=selected.get("kv_serve_action"),
        kv_post_turn_action=selected.get("kv_post_turn_action"),
        kv_background_event_count=int(bool(selected.get("kv_replication_bytes"))),
        kv_hit_tokens=int(selected["prefill_context"].get("kv_hit_tokens") or 0),
        kv_miss_tokens=int(selected["prefill_context"].get("kv_miss_tokens") or 0),
        context_recompute_tokens=int(selected["prefill_context"].get("historical_recompute_tokens") or 0),
        kv_reused_bytes=float(selected["prefill_context"].get("kv_reused_bytes") or 0.0),
        kv_generated_bytes=float(selected["prefill_context"].get("kv_generated_bytes") or 0.0),
        kv_migration_bytes=selected.get("kv_migration_bytes"),
        kv_replication_bytes=selected.get("kv_replication_bytes"),
        kv_transfer_attempted_bytes=transfer_attempted,
        kv_transfer_delivered_bytes=transfer_delivered,
        kv_transfer_committed_page_bytes=transfer_committed,
        kv_transfer_useful_bytes=transfer_useful,
        kv_transfer_interrupted=transfer_interrupted,
        kv_transfer_expired=transfer_expired,
        kv_transfer_interruption_reason=(
            migration_result.interruption_reason or kv_result.interruption_reason or replication_result.interruption_reason
        ),
        kv_transfer_expiry_reason=(
            migration_result.expiry_reason or kv_result.expiry_reason or replication_result.expiry_reason
        ),
        kv_pages_required=int(selected["prefill_context"].get("kv_pages_required") or 0),
        kv_pages_hit=int(selected["prefill_context"].get("kv_pages_hit") or 0),
        kv_pages_migrated=int(selected["prefill_context"].get("kv_pages_migrated") or 0),
        kv_pages_replicated=(
            _kv_required_pages(selected["context_tokens_after"], _kv_page_size_tokens(kv_policy))
            if replication_succeeded
            else 0
        ),
        kv_pages_evicted=0,
        hbm_admission_result=selected.get("kv_capacity_admission"),
        hbm_occupancy_before=_optional_float(kv_capacity_result.get("occupancy_before")),
        hbm_occupancy_after=_optional_float(kv_capacity_result.get("occupancy_after")),
        visible_kv_delay_s=kv_visible_s + migration_visible_s,
        hidden_kv_delay_s=migration_hidden_s,
        response_streaming_model="first_token_plus_bulk_remaining",
        selected_path=selected_path,
        route_node_ids=_route_node_ids(selected_path),
        soc_before=pd_energy_update["soc_before"],
        soc_after=pd_energy_update["soc_after"],
        energy_violation=energy_violation,
        compute_sunlit=bool(prefill_phase_energy["all_sunlit"] and decode_phase_energy["all_sunlit"]),
        access_sunlit=_energy_window_sunlit(selected.get("access_energy_window")),
        dvfs_multiplier=max(
            float(prefill_phase_energy["dvfs_multiplier"]),
            float(decode_phase_energy["dvfs_multiplier"]),
        ),
        shadow_strategy=shadow_strategies[0] if shadow_strategies else None,
        shadow_strategies=shadow_strategies,
        access_action_type=access_action_type,
        prefill_action_type=prefill_action_type,
        decode_action_type=decode_action_type,
        energy_ledger=tuple(pd_energy_update.get("energy_ledger", ())),
        fallback_used=False,
        deferred=deferred,
        defer_until_s=defer_until_s,
        session_action=session_action,
        bottleneck_reason=bottleneck,
        scope=scope,
        trace_model=trace_model,
        valid_for_paper_main_results=False,
        provenance={
            "policy": policy,
            "route_plan": route_in_plan.to_dict(),
            "migration_route": tuple(leg for leg in migration_result.path),
            "kv_route": tuple(leg for leg in kv_result.path),
            "replication_route": tuple(leg for leg in replication_result.path),
            "first_token_route_plan": selected["first_plan"].to_dict(),
            "remaining_route_plan": remaining_plan.to_dict() if remaining_plan is not None else None,
            "compute_eligibility": selected.get("compute_eligibility"),
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
            "internal_comm_accounting": getattr(prefill, "internal_comm_accounting", None),
            "capacity_allocation_assumption": "explicit_pd_deployment_group_per_group_throughput",
            "pd_placement_candidate_scope": getattr(prefill, "placement_scope", None),
            "resolved_pd_pair_scope": selected.get("resolved_pd_pair_scope"),
            "pd_pairing_rule": selected.get("pd_pairing_rule"),
            "prefill_tokens_effective": selected["prefill_tokens_effective"],
            "token_context_contract": selected["prefill_context"].get("token_context_contract"),
            "context_tokens_before_turn": selected["prefill_context"].get("context_tokens_before_turn"),
            "context_tokens_after_turn": selected["context_tokens_after"],
            "historical_recompute_tokens": selected["prefill_context"].get("historical_recompute_tokens"),
            "new_input_tokens_this_turn": selected["prefill_context"].get("new_input_tokens_this_turn"),
            "context_recompute_tokens": selected["prefill_context"].get("historical_recompute_tokens"),
            "sticky_prefill_kv_hit": bool(selected["prefill_context"].get("kv_hit_tokens")),
            "kv_capacity_result": kv_capacity_result,
            "energy_model": "event_incremental_per_node_per_physical_leg_v1",
            "dvfs_sample_model": "actual_prefill_decode_start_time_strategy_controlled",
            "action_components": action_components,
            "shadow_strategies": shadow_strategies,
            "access_action_type": access_action_type,
            "prefill_action_type": prefill_action_type,
            "decode_action_type": decode_action_type,
            "prefill_raw_dvfs_multiplier": raw_prefill_energy_window.get("dvfs_multiplier"),
            "prefill_effective_dvfs_multiplier": prefill_energy_window.get("dvfs_multiplier"),
            "decode_raw_dvfs_multiplier": raw_decode_energy_window.get("dvfs_multiplier"),
            "decode_effective_dvfs_multiplier": decode_energy_window.get("dvfs_multiplier"),
            "prefill_member_energy_weights": _deployment_member_energy_weights(prefill),
            "decode_member_energy_weights": _deployment_member_energy_weights(decode),
            "prefill_member_actions": prefill_phase_energy["member_actions"],
            "decode_member_actions": decode_phase_energy["member_actions"],
            "compute_energy_distribution": "explicit_member_weights",
            "actual_prefill_start_s": prefill_start_s,
            "actual_decode_start_s": decode_start_s,
            "energy_ledger": tuple(pd_energy_update.get("energy_ledger", ())),
            "preliminary_surrogate": True,
        },
    )
    return record, _decision_trace(
        request,
        policy,
        (route_in_plan,),
        route_in_plan,
        bottleneck,
        scheduler_decision=selected.get("scheduler_decision"),
    )


def _has_explicit_pd_deployments(deployment_catalog: RuntimeDeploymentCatalog) -> bool:
    return bool(
        tuple(getattr(deployment_catalog, "prefill_deployments", ()) or ())
        and tuple(getattr(deployment_catalog, "decode_deployments", ()) or ())
    )


def _pd_logical_deployment_id(deployment_catalog: RuntimeDeploymentCatalog) -> str:
    default = getattr(deployment_catalog, "default_deployment", None)
    if default is not None and getattr(default, "role", None) == "colocated":
        return str(default.deployment_id)
    for deployment in tuple(getattr(deployment_catalog, "prefill_deployments", ()) or ()) + tuple(
        getattr(deployment_catalog, "decode_deployments", ()) or ()
    ):
        deployment_id = str(getattr(deployment, "deployment_id", "pd_disaggregated"))
        for marker in (":prefill:", ":decode:"):
            if marker in deployment_id:
                return deployment_id.split(marker, 1)[0]
        return deployment_id
    return "pd_disaggregated"


def _deployments_by_entry(deployments: tuple[Any, ...]) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = {}
    for deployment in deployments:
        entry = _deployment_entry_satellite(deployment)
        if entry is None:
            continue
        grouped.setdefault(entry, []).append(deployment)
    return {key: tuple(value) for key, value in grouped.items()}


def _deployment_entry_satellite(deployment: Any) -> str:
    entries = tuple(getattr(deployment, "entry_satellite_ids", ()) or ())
    if entries:
        return str(entries[0])
    members = tuple(getattr(deployment, "member_satellites", ()) or ())
    return str(members[0]) if members else str(getattr(deployment, "deployment_id", ""))


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
    links = _active_link_rows_for_pd(trace_bundle, active.value)
    hop_limit = max_hop_count if max_hop_count is not None else 2
    link_ids = _build_ring_path_from_graph(
        _pd_ring_graph_from_active_links(trace_bundle, active, links),
        source_satellite_id,
        target_satellite_id,
        max_hop_count=hop_limit,
    )
    if link_ids is None:
        raise RouteSamplingError(
            "pd_kv_route_unavailable",
            sample_time_s=active.sample_time_s,
            warnings=active.warnings or ("pd_kv_route_unavailable",),
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
                "pd_kv_route_direction_mismatch",
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
            "pd_kv_route_direction_mismatch",
            sample_time_s=active.sample_time_s,
        )
    return tuple(legs)


def _active_link_rows_for_pd(trace_bundle: Any, value: Any) -> tuple[dict[str, Any], ...]:
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


def _pd_ring_graph_from_active_links(
    trace_bundle: Any,
    active: Any,
    links: tuple[dict[str, Any], ...],
) -> dict[str, tuple[tuple[float, str, str], ...]]:
    cache = getattr(trace_bundle, "__dict__", {}).get("_pd_ring_graph_cache")
    if not isinstance(cache, dict):
        cache = {}
        try:
            object.__setattr__(trace_bundle, "_pd_ring_graph_cache", cache)
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


def _pd_runtime_pair_allowed(prefill: Any, decode: Any) -> bool:
    return _pd_runtime_pair_scope(prefill, decode) is not None


def _pd_runtime_pair_rule(prefill: Any, decode: Any) -> str | None:
    decode_id = str(getattr(decode, "deployment_id", ""))
    paired_rules = getattr(prefill, "paired_deployment_rules", None) or {}
    if isinstance(paired_rules, dict) and decode_id in paired_rules:
        return str(paired_rules[decode_id])
    scope = _optional_str(getattr(prefill, "placement_scope", None))
    return scope


def _pd_runtime_pair_scope(prefill: Any, decode: Any) -> str | None:
    paired = tuple(str(item) for item in getattr(prefill, "paired_deployment_ids", ()) or ())
    decode_id = str(getattr(decode, "deployment_id", ""))
    if paired and decode_id not in paired:
        return None
    paired_scopes = getattr(prefill, "paired_deployment_scopes", None) or {}
    if isinstance(paired_scopes, dict) and decode_id in paired_scopes:
        return str(paired_scopes[decode_id])
    scope = _optional_str(getattr(prefill, "placement_scope", None)) or _optional_str(
        getattr(decode, "placement_scope", None)
    )
    if scope in {None, "hybrid", "unscoped_disaggregated", "aggregate_logical_pool"}:
        return "unscoped_disaggregated" if scope != "hybrid" else "hybrid"
    if scope == "intra_cluster":
        return "intra_cluster" if _pd_same_cluster(prefill, decode) else None
    if scope == "same_plane_cross_cluster":
        return (
            "same_plane_cross_cluster"
            if _pd_same_plane(prefill, decode) and _pd_different_cluster_or_cell(prefill, decode)
            else None
        )
    if scope == "cross_plane":
        prefill_plane = getattr(prefill, "plane_index", None)
        decode_plane = getattr(decode, "plane_index", None)
        return "cross_plane" if prefill_plane is not None and decode_plane is not None and prefill_plane != decode_plane else None
    return None


def _pd_same_cluster(prefill: Any, decode: Any) -> bool:
    prefill_cluster = getattr(prefill, "cluster_id", None)
    decode_cluster = getattr(decode, "cluster_id", None)
    if prefill_cluster is not None and decode_cluster is not None:
        return prefill_cluster == decode_cluster
    prefill_cell = getattr(prefill, "compute_cell_id", None)
    decode_cell = getattr(decode, "compute_cell_id", None)
    if prefill_cell is not None and decode_cell is not None:
        return prefill_cell == decode_cell
    return (
        _pd_same_plane(prefill, decode)
        and getattr(prefill, "cell_index", None) is not None
        and getattr(prefill, "cell_index", None) == getattr(decode, "cell_index", None)
    )


def _pd_same_plane(prefill: Any, decode: Any) -> bool:
    prefill_plane = getattr(prefill, "plane_index", None)
    decode_plane = getattr(decode, "plane_index", None)
    return prefill_plane is not None and decode_plane is not None and prefill_plane == decode_plane


def _pd_different_cluster_or_cell(prefill: Any, decode: Any) -> bool:
    prefill_cluster = getattr(prefill, "cluster_id", None)
    decode_cluster = getattr(decode, "cluster_id", None)
    if prefill_cluster is not None and decode_cluster is not None:
        return prefill_cluster != decode_cluster
    prefill_cell = getattr(prefill, "compute_cell_id", None)
    decode_cell = getattr(decode, "compute_cell_id", None)
    if prefill_cell is not None and decode_cell is not None:
        return prefill_cell != decode_cell
    prefill_cell_index = getattr(prefill, "cell_index", None)
    decode_cell_index = getattr(decode, "cell_index", None)
    return prefill_cell_index is not None and decode_cell_index is not None and prefill_cell_index != decode_cell_index




































































def _deployment_member_energy_weights(deployment: Any) -> dict[str, float]:
    members = tuple(str(item) for item in tuple(getattr(deployment, "member_satellites", ()) or ()))
    if not members:
        raise ValueError("pd_deployment_energy_members_missing")
    if len(members) == 1:
        return {members[0]: 1.0}
    raw = {
        str(key): float(value)
        for key, value in _dict(getattr(deployment, "member_energy_weights", {})).items()
    }
    if set(raw) != set(members):
        raise ValueError("pd_deployment_member_energy_weights_missing")
    if any(not math.isfinite(value) or value <= 0 for value in raw.values()):
        raise ValueError("pd_deployment_member_energy_weights_invalid")
    if abs(sum(raw.values()) - 1.0) > 1e-9:
        raise ValueError("pd_deployment_member_energy_weights_not_normalized")
    return raw


def _pd_group_phase_profile(
    *,
    trace_bundle: Any,
    deployment: Any,
    start_s: float,
    runtime_strategy: Any,
) -> dict[str, Any]:
    """Resolve a synchronous PD group's action from every physical member.

    Group throughput is a critical-path contract: the slowest member's effective
    DVFS multiplier determines phase latency. Energy remains explicitly weighted
    per member, so this helper is intentionally shared by prediction and commit.
    """

    member_actions: dict[str, str] = {}
    member_windows: dict[str, dict[str, Any]] = {}
    for satellite_id in _deployment_member_energy_weights(deployment):
        raw_window = energy_window_at(trace_bundle, satellite_id, start_s)
        action_type = _pd_compute_phase_action_type(raw_window, runtime_strategy)
        effective_window = _energy_window_for_runtime_action(
            raw_window,
            runtime_strategy=runtime_strategy,
            action_type=action_type,
        )
        member_actions[satellite_id] = action_type
        member_windows[satellite_id] = {"raw": raw_window, "effective": effective_window}
    multiplier = max(
        float(row["effective"].get("dvfs_multiplier", 1.0))
        for row in member_windows.values()
    )
    shadow_strategies = tuple(
        dict.fromkeys(
            action
            for action in member_actions.values()
            if action in {"shadow_offload", "shadow_dvfs", "shadow_local_compute"}
        )
    )
    group_action_type = (
        "shadow_dvfs"
        if "shadow_dvfs" in shadow_strategies
        else (
            "shadow_offload"
            if "shadow_offload" in shadow_strategies
            else (shadow_strategies[0] if shadow_strategies else "local_compute")
        )
    )
    return {
        "member_actions": member_actions,
        "member_windows": member_windows,
        "dvfs_multiplier": multiplier,
        "shadow_strategies": shadow_strategies,
        "group_action_type": group_action_type,
        "all_sunlit": all(
            _energy_window_sunlit(row["raw"]) for row in member_windows.values()
        ),
    }


def _apply_pd_compute_phase_energy(
    *,
    state: RuntimeState,
    trace_bundle: Any,
    deployment: Any,
    start_s: float,
    tokens: float,
    phase: str,
    runtime_strategy: Any,
    request_id: str,
) -> dict[str, Any]:
    updates: list[dict[str, Any]] = []
    profile = _pd_group_phase_profile(
        trace_bundle=trace_bundle,
        deployment=deployment,
        start_s=start_s,
        runtime_strategy=runtime_strategy,
    )
    for satellite_id, weight in _deployment_member_energy_weights(deployment).items():
        satellite = state.satellite_for(satellite_id)
        advance_satellite_energy(satellite, trace_bundle, start_s)
        action_type = profile["member_actions"][satellite_id]
        effective_window = profile["member_windows"][satellite_id]["effective"]
        update = apply_request_energy(
            satellite,
            prompt_tokens=tokens * weight if phase == "prefill" else 0,
            output_tokens=tokens * weight if phase == "decode" else 0,
            link_bytes=0.0,
            energy_window=effective_window,
            event_time_s=start_s,
            metadata={
                "request_id": request_id,
                "energy_stage": phase,
                "endpoint_role": "compute_member",
                "deployment_id": str(getattr(deployment, "deployment_id", "")),
                "member_energy_weight": weight,
                "compute_energy_distribution": "explicit_member_weights",
            },
        )
        updates.append(update)
    return {
        "updates": updates,
        "ledger_entries": [update["ledger_entry"] for update in updates],
        **profile,
        "soc_before": min(float(update["soc_before"]) for update in updates),
        "soc_after": min(float(update["soc_after"]) for update in updates),
        "energy_violation": any(bool(update["energy_violation"]) for update in updates),
    }


def _pd_energy_prediction(
    *,
    state: RuntimeState,
    trace_bundle: Any,
    prefill_entry: str,
    decode_entry: str,
    prefill_deployment: Any,
    decode_deployment: Any,
    prefill_energy_window: dict[str, Any],
    decode_energy_window: dict[str, Any],
    prefill_start_s: float,
    decode_start_s: float,
    arrival_s: float,
    prefill_ready_floor_s: float,
    prefill_queue_available_s: float,
    decode_queue_available_s: float,
    prefill_base_service_s: float,
    decode_base_service_s: float,
    first_token_base_service_s: float,
    output_tokens: int,
    prefill_tokens_effective: int,
    runtime_strategy: Any,
    prefill_action_type: str,
    decode_action_type: str,
    pre_prefill_routes: tuple[tuple[tuple[Any, ...], float, str], ...],
    between_phase_routes: tuple[tuple[tuple[Any, ...], float, str], ...],
    post_decode_routes: tuple[tuple[tuple[Any, ...], float, str], ...],
) -> tuple[float, float, dict[str, Any]]:
    preview_state = _clone_runtime_state_for_preview(state)
    event_entries: list[dict[str, Any]] = []

    route_results: dict[str, dict[str, Any]] = {}

    def apply_routes(specs: tuple[tuple[tuple[Any, ...], float, str], ...]) -> float:
        latest_done_s = 0.0
        for legs, start_s, stage in specs:
            if not legs:
                route_results[stage] = {"done_s": start_s, "path": (), "wait_s": 0.0, "tx_s": 0.0, "legs": ()}
                latest_done_s = max(latest_done_s, start_s)
                continue
            original_sample_s = _optional_float(getattr(legs[0], "sample_time_s", None))
            reserve = (
                _reserve_route
                if original_sample_s is None or abs(original_sample_s - start_s) > 1e-9
                else _reserve_sampled_route
            )
            done_s, path, sgl_wait, isl_wait, tx_s, sampled_legs = reserve(
                preview_state,
                trace_bundle,
                legs,
                start_s,
                request_id="pd-candidate-preview",
                energy_stage=stage,
            )
            event_entries.extend(_path_energy_entries(path))
            route_results[stage] = {
                "done_s": done_s,
                "path": path,
                "wait_s": sgl_wait + isl_wait,
                "tx_s": tx_s,
                "legs": sampled_legs,
            }
            latest_done_s = max(latest_done_s, done_s)
        return latest_done_s

    pre_routes_done_s = apply_routes(pre_prefill_routes)
    prefill_start_s = max(prefill_ready_floor_s, prefill_queue_available_s, pre_routes_done_s)
    prefill_profile = _pd_group_phase_profile(
        trace_bundle=trace_bundle,
        deployment=prefill_deployment,
        start_s=prefill_start_s,
        runtime_strategy=runtime_strategy,
    )
    prefill_service_s = prefill_base_service_s * float(prefill_profile["dvfs_multiplier"])
    prefill_done_s = prefill_start_s + prefill_service_s
    prefill_phase_energy = _apply_pd_compute_phase_energy(
        state=preview_state,
        trace_bundle=trace_bundle,
        deployment=prefill_deployment,
        start_s=prefill_start_s,
        tokens=prefill_tokens_effective,
        phase="prefill",
        runtime_strategy=runtime_strategy,
        request_id="pd-candidate-preview",
    )
    event_entries.extend(prefill_phase_energy["ledger_entries"])
    between_routes_done_s = apply_routes(
        tuple((legs, prefill_done_s, stage) for legs, _old_start, stage in between_phase_routes)
    )
    decode_ready_s = max(prefill_done_s, between_routes_done_s)
    decode_start_s = max(decode_queue_available_s, decode_ready_s)
    decode_profile = _pd_group_phase_profile(
        trace_bundle=trace_bundle,
        deployment=decode_deployment,
        start_s=decode_start_s,
        runtime_strategy=runtime_strategy,
    )
    decode_service_s = decode_base_service_s * float(decode_profile["dvfs_multiplier"])
    first_token_service_s = min(
        decode_service_s,
        first_token_base_service_s * float(decode_profile["dvfs_multiplier"]),
    )
    first_token_ready_s = decode_start_s + first_token_service_s
    decode_done_s = decode_start_s + decode_service_s
    decode_phase_energy = _apply_pd_compute_phase_energy(
        state=preview_state,
        trace_bundle=trace_bundle,
        deployment=decode_deployment,
        start_s=decode_start_s,
        tokens=output_tokens,
        phase="decode",
        runtime_strategy=runtime_strategy,
        request_id="pd-candidate-preview",
    )
    event_entries.extend(decode_phase_energy["ledger_entries"])
    rewritten_post_routes = tuple(
        (
            legs,
            first_token_ready_s if stage == "first_token_route" else decode_done_s,
            stage,
        )
        for legs, _old_start, stage in post_decode_routes
    )
    apply_routes(rewritten_post_routes)
    summary = _energy_summary_for_events(
        preview_state,
        trace_bundle,
        event_entries,
    )
    def phase_gate(phase_energy: dict[str, Any]) -> dict[str, Any]:
        members = {
            str(row["satellite_id"]): {
                "satellite_id": str(row["satellite_id"]),
                "soc_before": float(row["soc_before"]),
                "soc_after": float(row["soc_after"]),
                "soc_min": float(row.get("soc_min", 0.0)),
                "passed": bool(
                    float(row["soc_after"]) >= float(row.get("soc_min", 0.0))
                    and float(row.get("unserved_energy_j", 0.0)) <= 0
                ),
            }
            for row in phase_energy["ledger_entries"]
        }
        return {
            "passed": all(member["passed"] for member in members.values()),
            "members": members,
            "soc_before": min(member["soc_before"] for member in members.values()),
            "soc_after": min(member["soc_after"] for member in members.values()),
            "soc_min": None,
        }

    prefill_gate = phase_gate(prefill_phase_energy)
    decode_gate = phase_gate(decode_phase_energy)
    prefill_passed = bool(prefill_gate["passed"])
    decode_passed = bool(decode_gate["passed"])
    prefill_risk = max(
        (max(0.0, member["soc_min"] - member["soc_after"]) for member in prefill_gate["members"].values()),
        default=0.0,
    )
    decode_risk = max(
        (max(0.0, member["soc_min"] - member["soc_after"]) for member in decode_gate["members"].values()),
        default=0.0,
    )
    final_by_satellite = {
        str(row["satellite_id"]): float(row["soc_after"])
        for row in tuple(summary.get("energy_ledger", ()))
        if row.get("energy_kind") == "request_event"
    }
    first_token_done_s = route_results.get("first_token_route", {}).get("done_s", first_token_ready_s)
    remaining_done_s = route_results.get("remaining_output_route", {}).get("done_s", first_token_done_s)
    gate = {
        "passed": prefill_passed and decode_passed and not summary["energy_violation"],
        "reason": None if prefill_passed and decode_passed and not summary["energy_violation"] else "energy_soc",
        "predicted_soc_after": float(summary["soc_after"]),
        "soc_min": None,
        "soc_slack": min(
            *(member["soc_after"] - member["soc_min"] for member in prefill_gate["members"].values()),
            *(member["soc_after"] - member["soc_min"] for member in decode_gate["members"].values()),
        ),
        "prefill": {**prefill_gate, "satellite_id": prefill_entry},
        "decode": {**decode_gate, "satellite_id": decode_entry},
        "final_soc_by_satellite": final_by_satellite,
        "per_node_ledger": tuple(summary.get("energy_ledger", ())),
        "prefill_start_s": prefill_start_s,
        "decode_start_s": decode_start_s,
        "prefill_done_s": prefill_done_s,
        "decode_done_s": decode_done_s,
        "prefill_service_s": prefill_service_s,
        "decode_service_s": decode_service_s,
        "first_token_service_s": first_token_service_s,
        "first_token_ready_s": first_token_ready_s,
        "first_token_done_s": first_token_done_s,
        "remaining_done_s": remaining_done_s,
        "predicted_ttft_s": first_token_done_s - arrival_s,
        "predicted_e2e_s": remaining_done_s - arrival_s,
        "prefill_profile": prefill_profile,
        "decode_profile": decode_profile,
        "route_results": route_results,
        "route_wait_s": sum(float(row["wait_s"]) for row in route_results.values()),
    }
    ledger_risk = max(
        (
            max(0.0, float(row.get("soc_min", 0.0)) - float(row.get("soc_after", 0.0)))
            for row in tuple(summary.get("energy_ledger", ()))
            if row.get("energy_kind") == "request_event"
        ),
        default=0.0,
    )
    return max(prefill_risk, decode_risk, ledger_risk), float(summary["soc_after"]), gate


def _predicted_soc_after(
    satellite: Any,
    energy_window: dict[str, Any],
    *,
    compute_tokens: float,
    link_bytes: float,
) -> float:
    capacity_j = max(1.0, float(energy_window["battery_capacity_j"]))
    discharge_efficiency = float(energy_window.get("battery_discharge_efficiency", 1.0))
    if (
        not math.isfinite(discharge_efficiency)
        or discharge_efficiency <= 0.0
        or discharge_efficiency > 1.0
    ):
        return 0.0
    dynamic = compute_dynamic_energy_j(
        prompt_tokens=max(0.0, compute_tokens),
        output_tokens=0.0,
        link_bytes=max(0.0, link_bytes),
        intensity=EnergyIntensity(
            compute_energy_j_per_token=float(energy_window["compute_energy_j_per_token"]),
            link_energy_j_per_byte=float(energy_window["link_energy_j_per_byte"]),
        ),
    )
    chemical_energy_j = dynamic.total_j / discharge_efficiency
    return max(
        0.0,
        float(getattr(satellite, "soc", 0.8)) - chemical_energy_j / capacity_j,
    )


def _route_visible_s(path: tuple[dict[str, Any], ...]) -> float:
    return sum(
        float(row.get("queue_wait_s", 0.0) or 0.0)
        + float(row.get("transmission_s", 0.0) or 0.0)
        + float(row.get("setup_delay_s", 0.0) or 0.0)
        + float(row.get("propagation_delay_s", 0.0) or 0.0)
        for row in path
    )


def _route_component_s(path: tuple[dict[str, Any], ...], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in path)


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
        dvfs = float(request_energy_window.get("dvfs_multiplier", 1.0))
        prefill_service_s = _safe_service_time(request["prompt_tokens"], prefill_rate_per_sat) * dvfs
        decode_service_s = _safe_service_time(request["output_tokens"], decode_rate_per_sat) * dvfs
        if not math.isfinite(prefill_service_s) or not math.isfinite(decode_service_s):
            continue
        prefill_start, prefill_done, prefill_wait = satellite.compute_queue.preview_prefill(inbound_done, prefill_service_s)
        decode_start, decode_done, decode_wait = satellite.compute_queue.preview_decode(prefill_done, decode_service_s)
        first_token_service_s = min(decode_service_s, _safe_service_time(1, decode_rate_per_sat) * dvfs)
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
                "raw_dvfs_multiplier": raw_request_energy_window.get("dvfs_multiplier"),
                "effective_dvfs_multiplier": request_energy_window.get("dvfs_multiplier"),
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
            "dvfs_multiplier": request_energy_window.get("dvfs_multiplier"),
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
    return mode in {str(item) for item in tuple(getattr(runtime_strategy, "dvfs_modes_enabled", ()) or ())}


def _pd_compute_phase_action_type(
    energy_window: dict[str, Any],
    runtime_strategy: Any,
) -> str:
    if _energy_window_sunlit(energy_window) is False:
        if _runtime_dvfs_enabled(runtime_strategy, "shadow_dvfs"):
            return "shadow_dvfs"
        return "shadow_local_compute"
    return "local_compute"


def _pd_access_action_type(
    *,
    route_in_plan: Any,
    access_energy_window: dict[str, Any],
    prefill_energy_window: dict[str, Any],
) -> str:
    if int(getattr(route_in_plan, "hop_count", 0) or 0) <= 0:
        return "local_compute"
    if (
        _energy_window_sunlit(access_energy_window) is False
        and _energy_window_sunlit(prefill_energy_window) is not False
    ):
        return "shadow_offload"
    return "cross_satellite_forward"


def _pd_phase_actions_allowed(
    actions: tuple[str, ...],
    *,
    runtime_strategy: Any,
) -> bool:
    enabled = {str(item) for item in tuple(getattr(runtime_strategy, "enabled_actions", ()) or ())}
    if not enabled:
        return True
    for action in actions:
        aliases = {action}
        if action in {"shadow_dvfs", "shadow_local_compute"}:
            aliases.add("local_compute")
        if action == "shadow_offload":
            aliases.update(("cross_satellite_forward", "same_plane_forward", "cross_plane_forward"))
        if action == "cross_satellite_forward":
            aliases.update(("same_plane_forward", "cross_plane_forward"))
        if not enabled.intersection(aliases):
            return False
    return True


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
    if _energy_window_sunlit(energy_window) is not False:
        return adjusted
    overrides = _dict(getattr(runtime_strategy, "hard_gate_overrides", {}))
    if action_type == "shadow_dvfs":
        compute_scale = _optional_float(overrides.get("shadow_dvfs_compute_energy_scale"))
        link_scale = _optional_float(overrides.get("shadow_dvfs_link_energy_scale"))
        adjusted["compute_energy_j_per_token"] = float(adjusted["compute_energy_j_per_token"]) * (
            compute_scale if compute_scale is not None else 0.72
        )
        adjusted["link_energy_j_per_byte"] = float(adjusted["link_energy_j_per_byte"]) * (
            link_scale if link_scale is not None else 1.0
        )
        adjusted["shadow_strategy"] = "shadow_dvfs"
        adjusted["shadow_dvfs_compute_energy_scale"] = compute_scale if compute_scale is not None else 0.72
        adjusted["shadow_dvfs_link_energy_scale"] = link_scale if link_scale is not None else 1.0
        adjusted["runtime_compute_energy_scale"] = adjusted["shadow_dvfs_compute_energy_scale"]
        adjusted["runtime_link_energy_scale"] = adjusted["shadow_dvfs_link_energy_scale"]
        return adjusted
    if not _runtime_dvfs_enabled(runtime_strategy, "shadow_dvfs"):
        adjusted["dvfs_multiplier"] = 1.0
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
    ttft_deadline_s = _optional_float(request.get("ttft_deadline_s"))
    e2e_deadline_s = _optional_float(request.get("e2e_deadline_s"))
    ttft_miss = (
        ttft_deadline_s is not None
        and math.isfinite(predicted_ttft_s)
        and predicted_ttft_s > ttft_deadline_s
    )
    e2e_miss = (
        e2e_deadline_s is not None
        and math.isfinite(predicted_e2e_s)
        and predicted_e2e_s > e2e_deadline_s
    )
    energy_miss = (
        math.isfinite(predicted_soc_after)
        and math.isfinite(soc_min)
        and predicted_soc_after < soc_min
    )
    return {
        "slo_ttft": {
            "passed": None if ttft_deadline_s is None else not ttft_miss,
            "reason": "predicted_ttft_deadline_miss" if ttft_miss else None,
            "predicted_s": predicted_ttft_s,
            "deadline_s": ttft_deadline_s,
        },
        "slo_e2e": {
            "passed": None if e2e_deadline_s is None else not e2e_miss,
            "reason": "predicted_e2e_deadline_miss" if e2e_miss else None,
            "predicted_s": predicted_e2e_s,
            "deadline_s": e2e_deadline_s,
        },
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
        compute_queues={key: replace(queue) for key, queue in state.compute_queues.items()},
        satellites={
            key: replace(satellite, compute_queue=replace(satellite.compute_queue))
            for key, satellite in state.satellites.items()
        },
        links={key: replace(link) for key, link in state.links.items()},
        sessions={},
        kv_cache_deployments={},
        request_records=[],
        decision_traces=[],
        module_status=dict(state.module_status),
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
        target.dvfs_multiplier = preview.dvfs_multiplier
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


def _reserve_kv_transfer(
    *,
    state: RuntimeState,
    trace_bundle: Any,
    source_satellite_id: str,
    target_satellite_id: str,
    payload_bytes: float,
    start_s: float,
    max_hop_count: int | None,
    chunk_bytes: float | None,
    flow_kind: str,
    deadline_s: float | None,
    expires_at_s: float | None,
    request_id: str | None = None,
) -> KVTransferResult:
    payload_bytes = max(0.0, float(payload_bytes or 0.0))
    if payload_bytes <= 0:
        return KVTransferResult(done_s=start_s)
    if source_satellite_id == target_satellite_id:
        return KVTransferResult(
            done_s=start_s,
            committed_page_bytes=0.0,
            useful_bytes=0.0,
        )
    chunk_size = payload_bytes if chunk_bytes is None else max(1.0, min(payload_bytes, float(chunk_bytes)))
    remaining = payload_bytes
    current_s = start_s
    path: list[dict[str, Any]] = []
    sampled: list[Any] = []
    sgl_wait = 0.0
    isl_wait = 0.0
    tx = 0.0
    attempted = 0.0
    delivered = 0.0
    committed = 0.0
    useful = 0.0
    while remaining > 1e-9:
        chunk = min(chunk_size, remaining)
        try:
            legs = _satellite_route_legs(
                trace_bundle,
                source_satellite_id,
                target_satellite_id,
                chunk,
                time_s=current_s,
                max_hop_count=max_hop_count,
            )
            attempted += chunk
            done_s, chunk_path, chunk_sgl_wait, chunk_isl_wait, chunk_tx, chunk_sampled = _reserve_route(
                state,
                trace_bundle,
                legs,
                current_s,
                request_id=request_id,
                energy_stage=flow_kind,
            )
        except RouteSamplingError as exc:
            return KVTransferResult(
                done_s=current_s,
                path=tuple(path),
                sgl_wait_s=sgl_wait,
                isl_wait_s=isl_wait,
                transmission_s=tx,
                attempted_bytes=attempted,
                delivered_bytes=delivered,
                committed_page_bytes=committed,
                useful_bytes=useful,
                interrupted=True,
                interruption_reason=f"{flow_kind}:{exc.reason}",
                sampled_legs=tuple(sampled),
            )
        path.extend(chunk_path)
        sampled.extend(chunk_sampled)
        sgl_wait += chunk_sgl_wait
        isl_wait += chunk_isl_wait
        tx += chunk_tx
        delivered += chunk
        current_s = done_s
        if deadline_s is not None and current_s > deadline_s:
            return KVTransferResult(
                done_s=current_s,
                path=tuple(path),
                sgl_wait_s=sgl_wait,
                isl_wait_s=isl_wait,
                transmission_s=tx,
                attempted_bytes=attempted,
                delivered_bytes=delivered,
                committed_page_bytes=committed,
                useful_bytes=useful,
                expired=True,
                expiry_reason=f"{flow_kind}:deadline_expired",
                sampled_legs=tuple(sampled),
            )
        if expires_at_s is not None and current_s > expires_at_s:
            return KVTransferResult(
                done_s=current_s,
                path=tuple(path),
                sgl_wait_s=sgl_wait,
                isl_wait_s=isl_wait,
                transmission_s=tx,
                attempted_bytes=attempted,
                delivered_bytes=delivered,
                committed_page_bytes=committed,
                useful_bytes=useful,
                expired=True,
                expiry_reason=f"{flow_kind}:ttl_expired",
                sampled_legs=tuple(sampled),
            )
        remaining -= chunk
    committed = payload_bytes
    useful = payload_bytes
    return KVTransferResult(
        done_s=current_s,
        path=tuple(path),
        sgl_wait_s=sgl_wait,
        isl_wait_s=isl_wait,
        transmission_s=tx,
        attempted_bytes=attempted,
        delivered_bytes=delivered,
        committed_page_bytes=committed,
        useful_bytes=useful,
        sampled_legs=tuple(sampled),
    )


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
    deployment: RuntimeDeployment,
    trace_bundle: Any,
    state: RuntimeState,
    time_s: float,
) -> ComputeEligibilitySet:
    runtime_exclusions: dict[str, str] = {}
    for satellite_id in tuple(getattr(deployment, "member_satellites", ()) or ()):
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
        deployment=deployment,
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
    deployment_id: str,
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
        arrival_time_s=arrival_s,
        deployment_id=deployment_id,
        action_type=action_type,
        accepted=True,
        completed=False,
        served=False,
        rejected=False,
        timeout=True,
        timeout_reason="route_became_unavailable_after_compute",
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        ttft_deadline_s=request["ttft_deadline_s"],
        e2e_deadline_s=request["e2e_deadline_s"],
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
        assigned_satellite_id=compute_satellite_id,
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
    timeout = False
    timeout_reason = None
    served = False
    if latency_s is not None:
        ttft_s = latency_s
        e2e_s = latency_s
        timeout, timeout_reason = _timeout_status(ttft_s, e2e_s, request["ttft_deadline_s"], request["e2e_deadline_s"])
        served = not timeout
    return RuntimeRequestRecord(
        request_id=request["request_id"],
        arrival_time_s=request["arrival_time_s"],
        deployment_id=None,
        action_type="ground_fallback",
        accepted=True,
        completed=completed,
        served=served,
        rejected=False,
        timeout=timeout,
        reject_reason=None if completed else reason,
        timeout_reason=timeout_reason,
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        ttft_deadline_s=request["ttft_deadline_s"],
        e2e_deadline_s=request["e2e_deadline_s"],
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
        arrival_time_s=request["arrival_time_s"],
        deployment_id=None,
        action_type="reject_drop",
        accepted=False,
        completed=False,
        served=False,
        rejected=True,
        timeout=False,
        reject_reason=reason,
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        ttft_deadline_s=request.get("ttft_deadline_s"),
        e2e_deadline_s=request.get("e2e_deadline_s"),
        bottleneck_reason=reason,
        scope=scope,
        trace_model=trace_model,
        valid_for_paper_main_results=False,
        provenance={"reject_reason": reason, "warnings": warnings, **(provenance or {})},
    )


def _timeout_status(
    ttft_s: float | None,
    e2e_s: float | None,
    ttft_deadline_s: float | None,
    e2e_deadline_s: float | None,
) -> tuple[bool, str | None]:
    ttft_miss = ttft_s is not None and ttft_deadline_s is not None and ttft_s > ttft_deadline_s
    e2e_miss = e2e_s is not None and e2e_deadline_s is not None and e2e_s > e2e_deadline_s
    if ttft_miss and e2e_miss:
        return True, "ttft_and_e2e_deadline_miss"
    if ttft_miss:
        return True, "ttft_deadline_miss"
    if e2e_miss:
        return True, "e2e_deadline_miss"
    return False, None


def _session_action(state: RuntimeState, session_id: str | None, satellite_id: str, now_s: float) -> str:
    if not session_id:
        return "none"
    session = state.session_for(session_id)
    if session.owner_satellite_id is None or session.owner_satellite_id == satellite_id:
        session.owner_satellite_id = satellite_id
        session.last_seen_s = now_s
        return "sticky_kv_owner"
    session.owner_satellite_id = satellite_id
    session.last_seen_s = now_s
    return "recompute_context"


def _bottleneck(
    timeout_reason: str | None,
    energy_violation: bool,
    prefill_wait_s: float,
    decode_wait_s: float,
    link_wait_s: float,
) -> str:
    if timeout_reason:
        return timeout_reason
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


__all__ = [
    "RuntimeEngineResult",
    "normalize_runtime_request",
    "run_trace_backed_runtime",
]
