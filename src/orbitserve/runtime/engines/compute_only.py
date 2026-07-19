"""Ideal full-visibility compute-only FIFO engine."""

from __future__ import annotations

import math
from typing import Any

from .contracts import RuntimeEngineResult
from ..deployment_contracts import RuntimeDeploymentCatalog
from ..runtime_records import RuntimeRequestRecord
from ..runtime_state import RuntimeState
from ..runtime_timing import TTFT_DEFINITION, ttft_breakdown

def run_compute_only_fifo(
    *,
    requests: tuple[dict[str, Any], ...],
    deployment_catalog: RuntimeDeploymentCatalog,
    policy: Any,
    trace_model: str,
    scope: str = "idealized_compute_only",
) -> RuntimeEngineResult:
    """Run a Phase-1a compute-only FIFO work-seconds model."""

    state = RuntimeState()
    records: list[RuntimeRequestRecord] = []
    warnings = list(deployment_catalog.warnings)
    deployments = deployment_catalog.deployments
    ordered_requests = sorted(
        requests,
        key=lambda item: (float(item.get("arrival_time_s", 0.0)), str(item.get("request_id", ""))),
    )

    for request in ordered_requests:
        action = policy.select_action(state, request, deployments)
        deployment_id = action.serving_deployment
        deployment = _deployment_by_id(deployments, deployment_id)
        record = _evaluate_request(
            state,
            request,
            action=action,
            deployment=deployment,
            trace_model=trace_model,
            scope=scope,
        )
        records.append(record)
        state.request_records.append(record)
        state.decision_traces.append(
            {
                "request_id": record.request_id,
                "arrival_time_s": record.arrival_time_s,
                "decision_time_s": record.arrival_time_s,
                "policy_name": getattr(policy, "name", None),
                "candidate_action_count": len(deployments),
                "feasible_action_count": 1 if deployment and deployment.has_compute_capacity else 0,
                "selected_action": action.to_dict() if hasattr(action, "to_dict") else action,
                "bottleneck_reason": record.bottleneck_reason,
                "trace_schema_version": "orbitserve.runtime_decision_trace.v1",
                "warnings": (),
            }
        )

    warnings.append("compute-only FIFO runtime model uses idealized full visibility and no link/energy events")
    return RuntimeEngineResult(
        records=tuple(records),
        decision_traces=tuple(state.decision_traces),
        state_summary=state.to_dict(),
        warnings=tuple(dict.fromkeys(warnings)),
    )

def _evaluate_request(
    state: RuntimeState,
    request: dict[str, Any],
    *,
    action: Any,
    deployment: Any,
    trace_model: str,
    scope: str,
) -> RuntimeRequestRecord:
    request_id = str(request.get("request_id") or f"request-{len(state.request_records):06d}")
    arrival_s = _float(request.get("arrival_time_s"), 0.0)
    prompt_tokens = max(0, _int(_first_present(request.get("prompt_tokens"), request.get("new_user_tokens")), 0))
    output_tokens = max(0, _int(request.get("output_tokens"), 0))
    ttft_deadline_s = _optional_float(request.get("ttft_deadline_s"))
    e2e_deadline_s = _optional_float(request.get("e2e_deadline_s"))

    if output_tokens <= 0:
        return RuntimeRequestRecord(
            request_id=request_id,
            arrival_time_s=arrival_s,
            deployment_id=getattr(deployment, "deployment_id", None),
            action_type="reject_drop",
            accepted=False,
            completed=False,
            served=False,
            rejected=True,
            timeout=False,
            reject_reason="output_tokens_must_be_positive_for_ttft",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_deadline_s=ttft_deadline_s,
            e2e_deadline_s=e2e_deadline_s,
            bottleneck_reason="invalid_request",
            scope=scope,
            trace_model=trace_model,
            provenance={"ttft_definition": TTFT_DEFINITION},
        )

    if action.action_type != "local_compute":
        return RuntimeRequestRecord(
            request_id=request_id,
            arrival_time_s=arrival_s,
            deployment_id=None,
            action_type=action.action_type,
            accepted=False,
            completed=False,
            served=False,
            rejected=True,
            timeout=False,
            reject_reason=action.action_type,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_deadline_s=ttft_deadline_s,
            e2e_deadline_s=e2e_deadline_s,
            bottleneck_reason="reject",
            scope=scope,
            trace_model=trace_model,
            provenance={"policy_action": action.to_dict() if hasattr(action, "to_dict") else action},
        )

    if deployment is None or not deployment.has_compute_capacity:
        return RuntimeRequestRecord(
            request_id=request_id,
            arrival_time_s=arrival_s,
            deployment_id=getattr(deployment, "deployment_id", None),
            action_type=action.action_type,
            accepted=False,
            completed=False,
            served=False,
            rejected=True,
            timeout=False,
            reject_reason="compute_capacity_unavailable",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_deadline_s=ttft_deadline_s,
            e2e_deadline_s=e2e_deadline_s,
            bottleneck_reason="compute_capacity",
            scope=scope,
            trace_model=trace_model,
            provenance={"policy_action": action.to_dict() if hasattr(action, "to_dict") else action},
        )

    prefill_service_s = _safe_service_time(prompt_tokens, deployment.prefill_tokens_per_s)
    decode_service_s = _safe_service_time(output_tokens, deployment.decode_tokens_per_s)
    if not math.isfinite(prefill_service_s) or not math.isfinite(decode_service_s):
        return RuntimeRequestRecord(
            request_id=request_id,
            arrival_time_s=arrival_s,
            deployment_id=deployment.deployment_id,
            action_type=action.action_type,
            accepted=False,
            completed=False,
            served=False,
            rejected=True,
            timeout=False,
            reject_reason="non_finite_service_time",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_deadline_s=ttft_deadline_s,
            e2e_deadline_s=e2e_deadline_s,
            bottleneck_reason="service_time",
            scope=scope,
            trace_model=trace_model,
            provenance={"policy_action": action.to_dict() if hasattr(action, "to_dict") else action},
        )

    queue = state.queue_for(deployment.deployment_id)
    prefill_start_s, prefill_done_s, prefill_wait_s = queue.reserve_prefill(arrival_s, prefill_service_s)
    decode_start_s, decode_done_s, decode_wait_s = queue.reserve_decode(prefill_done_s, decode_service_s)
    del prefill_start_s
    first_token_service_s = min(decode_service_s, _safe_service_time(1, deployment.decode_tokens_per_s))
    timing = ttft_breakdown(
        access_wait_s=0.0,
        uplink_s=0.0,
        prefill_queue_s=prefill_wait_s,
        prefill_compute_s=prefill_service_s,
        kv_transfer_s=0.0,
        decode_queue_s=decode_wait_s,
        first_token_decode_s=first_token_service_s,
        return_path_s=0.0,
    )
    ttft_s = float(timing["ttft_s"]) if output_tokens > 0 else None
    e2e_s = decode_done_s - arrival_s
    ttft_miss = ttft_deadline_s is not None and ttft_s > ttft_deadline_s
    e2e_miss = e2e_deadline_s is not None and e2e_s > e2e_deadline_s
    timeout = ttft_miss or e2e_miss
    timeout_reason = None
    if ttft_miss and e2e_miss:
        timeout_reason = "ttft_and_e2e_deadline_miss"
    elif ttft_miss:
        timeout_reason = "ttft_deadline_miss"
    elif e2e_miss:
        timeout_reason = "e2e_deadline_miss"

    bottleneck = "compute_queue" if prefill_wait_s > 0 or decode_wait_s > 0 else "compute_service"
    if timeout:
        bottleneck = timeout_reason or bottleneck

    return RuntimeRequestRecord(
        request_id=request_id,
        arrival_time_s=arrival_s,
        deployment_id=deployment.deployment_id,
        action_type=action.action_type,
        accepted=True,
        completed=True,
        served=not timeout,
        rejected=False,
        timeout=timeout,
        timeout_reason=timeout_reason,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_deadline_s=ttft_deadline_s,
        e2e_deadline_s=e2e_deadline_s,
        prefill_queue_wait_s=prefill_wait_s,
        decode_queue_wait_s=decode_wait_s,
        link_queue_wait_s=None,
        access_wait_s=0.0,
        prefill_service_s=prefill_service_s,
        decode_service_s=decode_service_s,
        uplink_s=0.0,
        kv_transfer_s=0.0,
        first_token_decode_s=first_token_service_s if output_tokens > 0 else None,
        return_path_s=0.0 if output_tokens > 0 else None,
        ttft_breakdown_s=timing if output_tokens > 0 else {},
        ttft_s=ttft_s,
        e2e_s=e2e_s,
        bottleneck_reason=bottleneck,
        scope=scope,
        trace_model=trace_model,
        valid_for_paper_main_results=False,
        provenance={
            "policy_action": action.to_dict() if hasattr(action, "to_dict") else action,
            "service_time_model": "tokens / aggregate_serving_throughput",
            "batching_model": "conservative_non_batched_work_seconds",
            "deployment_selection": getattr(deployment, "selection_policy", None),
            "placement_model": getattr(deployment, "placement_model", None),
            "ttft_definition": TTFT_DEFINITION,
        },
    )

def _deployment_by_id(deployments: tuple[Any, ...], deployment_id: str | None) -> Any:
    for deployment in deployments:
        if deployment.deployment_id == deployment_id:
            return deployment
    return deployments[0] if deployments else None

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

def _int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default
