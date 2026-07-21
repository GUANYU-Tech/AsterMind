"""Assemble engine records into the stable runtime result schema."""

from __future__ import annotations

from typing import Any

from ..contracts import RuntimePerformanceResult
from .runtime_metrics import aggregate_runtime_records
from .runtime_records import records_to_jsonable
from ...orbit import CIRCULAR_GEOMETRY_TRACE, IDEAL_FULL_VISIBILITY, SURROGATE_ORBIT_TRACE

def build_runtime_performance_result(
    *,
    scenario: Any,
    runtime_strategy: Any,
    engine_result: Any,
    trace_manifest: Any,
    effective_trace_model: str,
    effective_scope: str,
    deployment_catalog: Any,
    evaluator_name: str,
    path_model: str,
    energy_model: str,
    queue_model: str,
    link_model_status: str,
    energy_model_status: str,
    terrestrial_baseline: bool = False,
    centralized_ground_baseline: bool = False,
    terrestrial_costing: Any = None,
) -> RuntimePerformanceResult:
    aggregate_horizon_s = _aggregate_horizon_s(
        engine_result=engine_result,
        trace_model=effective_trace_model,
        scenario_horizon_s=getattr(scenario, "time_horizon_s", None),
    )
    aggregates = aggregate_runtime_records(
        engine_result.records,
        time_horizon_s=aggregate_horizon_s,
        trace_model=effective_trace_model,
        scope=effective_scope,
    )
    runtime_metrics = {
        **aggregates.runtime_metrics,
        **(getattr(engine_result, "runtime_metrics", None) or {}),
        **(terrestrial_costing.runtime_metrics() if terrestrial_costing is not None else {}),
    }
    module_status = {
        "workload": "loaded",
        "simulation": "not_required_terrestrial_baseline"
        if terrestrial_baseline
        else "idealized_trace"
        if trace_manifest.trace_model == IDEAL_FULL_VISIBILITY
        else "geometry_trace"
        if trace_manifest.trace_model == CIRCULAR_GEOMETRY_TRACE
        else "surrogate_trace",
        "serving_throughput": "ground_service_model_evaluated"
        if terrestrial_baseline
        else "evaluated",
        "runtime_strategy": runtime_strategy.policy,
        "queue_model": queue_model,
        "link_model": link_model_status,
        "energy_model": energy_model_status,
        "cost_model": (
            "terrestrial_cost_evaluated"
            if terrestrial_costing is not None and terrestrial_costing.evaluated
            else "terrestrial_cost_not_configured"
        )
        if terrestrial_baseline
        else "static_cost_only",
        "scope": effective_scope,
        "trace_model": effective_trace_model,
        "valid_for_paper_main_results": "false",
    }
    scenario_warnings = (
        tuple(getattr(scenario, "workload_bundle", {}).get("warnings", ()) or ())
        if terrestrial_baseline
        else tuple(getattr(scenario, "warnings", ()) or ())
    )
    trace_warnings = () if terrestrial_baseline else trace_manifest.warnings
    warnings = tuple(
        dict.fromkeys(
            (
                *scenario_warnings,
                *trace_warnings,
                *(deployment_catalog.warnings if deployment_catalog is not None else ()),
                *engine_result.warnings,
                *aggregates.warnings,
                *(terrestrial_costing.warnings if terrestrial_costing is not None else ()),
                (
                    "centralized/GKE-like terrestrial comparison baseline uses explicit topology and dispatcher surrogates"
                    if centralized_ground_baseline
                    else "SkyWalker ground runtime reproduces published control logic with explicit surrogate parameters"
                    if terrestrial_baseline
                    else "Phase 1a runtime metrics are idealized compute-only outputs"
                    if trace_manifest.trace_model == IDEAL_FULL_VISIBILITY
                    else "circular geometry trace-backed runtime metrics are preliminary surrogate outputs"
                    if trace_manifest.trace_model == CIRCULAR_GEOMETRY_TRACE
                    else "trace-backed runtime metrics are preliminary surrogate outputs"
                ),
            )
        )
    )
    request_records = tuple(records_to_jsonable(engine_result.records))
    effective_ground_config = (
        (engine_result.provenance or {}).get("effective_ground_config")
        if terrestrial_baseline
        else None
    )
    deployment_payload = (
        deployment_catalog.to_dict()
        if deployment_catalog is not None
        else {
            "model": "geo_distributed_terrestrial_llm_serving",
            "policy": runtime_strategy.policy,
            "ground_config": effective_ground_config or runtime_strategy.ground_config,
            "provenance": engine_result.provenance or {},
        }
    )
    details = {
        "scope": effective_scope,
        "trace_model": effective_trace_model,
        "valid_for_paper_main_results": False,
        "request_records": request_records,
        "state_summary": engine_result.state_summary,
        "deployment_catalog": deployment_payload,
        "engine_provenance": engine_result.provenance or {},
        "metrics_contract": {
            "served_ratio": "completed and TTFT/E2E deadlines both satisfied",
            "rejected_ratio": "arrival-time admission rejection",
            "timeout_ratio": "accepted request completed after TTFT or E2E deadline",
            "cost_per_served": (
                "explicit terrestrial cloud instance/optional-egress contract"
                if terrestrial_costing is not None and terrestrial_costing.evaluated
                else "terrestrial TCO not configured"
                if terrestrial_baseline
                else "not evaluated in Phase 1a"
            ),
        },
        **(
            {"terrestrial_costing": terrestrial_costing.to_dict()}
            if terrestrial_costing is not None
            else {}
        ),
    }
    return RuntimePerformanceResult(
        evaluated=True,
        scenario={
            "schema_version": getattr(scenario, "schema_version", "orbitserve.runtime_scenario.v1"),
            "scenario_id": getattr(scenario, "scenario_id", None),
            "design_id": None
            if terrestrial_baseline
            else getattr(scenario, "design_id", None),
            "time_horizon_s": getattr(scenario, "time_horizon_s", None),
            "simulation_evaluated": False if terrestrial_baseline else True,
            "simulation_required": not terrestrial_baseline,
            "trace_model": effective_trace_model,
            "scope": effective_scope,
            "valid_for_paper_main_results": False,
        },
        strategy=runtime_strategy.to_dict(),
        workload=getattr(scenario, "workload_bundle", {}),
        constellation={} if terrestrial_baseline else getattr(scenario, "constellation", {}),
        serving=deployment_payload,
        scheduler=runtime_strategy.to_dict(),
        config={
            "time_mode": getattr(scenario, "time_mode", "event_driven"),
            "time_step_s": getattr(scenario, "time_step_s", None),
            "queue_model": queue_model,
            "path_model": path_model,
            "energy_model": energy_model,
            "fallback_accounting": "not_applicable_terrestrial_baseline"
            if terrestrial_baseline
            else "not_enabled",
            "internal_comm_accounting": "explicit_ground_network_and_queue"
            if terrestrial_baseline
            else "embedded_in_serving_throughput",
            "ground_config": (
                effective_ground_config or runtime_strategy.ground_config
                if terrestrial_baseline
                else {}
            ),
            "scope": effective_scope,
            "trace_model": effective_trace_model,
            "valid_for_paper_main_results": False,
        },
        performance=aggregates.performance.to_dict(),
        slo=aggregates.slo.to_dict(),
        runtime_metrics=runtime_metrics,
        bottleneck_breakdown=aggregates.bottleneck_breakdown,
        session_metrics=(
            getattr(engine_result, "session_metrics", None)
            if getattr(engine_result, "session_metrics", None) is not None
            else {"evaluated": False, "value": None, "warnings": ("KV/session state is not implemented",)}
        ),
        communication_metrics=(
            getattr(engine_result, "communication_metrics", None)
            or aggregates.communication_metrics
        ),
        energy_metrics=(getattr(engine_result, "energy_metrics", None) or aggregates.energy_metrics),
        region_metrics=tuple(getattr(engine_result, "region_metrics", ()) or ()),
        prediction_error={
            "evaluated": False,
            "value": None,
            "warnings": ("observed-vs-predicted error requires a prediction model",),
        },
        non_separability={
            "evaluated": False,
            "value": None,
            "warnings": ("idealized compute-only metrics are not valid for paper main non-separability",),
        },
        decision_trace={
            "trace_schema_version": "orbitserve.runtime_decision_trace.v1",
            "evaluated": True,
            "trace_level": runtime_strategy.trace_level,
            "records": engine_result.decision_traces,
            "request_records_path": None,
            "warnings": (),
        },
        request_records_path=None,
        request_records=request_records,
        trace_model=effective_trace_model,
        scope=effective_scope,
        valid_for_paper_main_results=False,
        module_status=module_status,
        details=details,
        provenance={
            **getattr(scenario, "provenance", {}),
            "performance_evaluator": evaluator_name,
            "runtime_interface_schema": "orbitserve.runtime_scenario.v1",
            "runtime_performance_schema": "orbitserve.runtime_performance.v1",
            "trace_manifest": (
                {
                    "evaluated": True,
                    "trace_model": effective_trace_model,
                    "scope": effective_scope,
                    "source": "ground_config_and_workload_regions",
                }
                if terrestrial_baseline
                else trace_manifest.to_dict()
            ),
            "runtime_engine": engine_result.provenance or {},
            "valid_for_paper_main_results": False,
        },
        warnings=warnings,
    )

def _aggregate_horizon_s(
    *,
    engine_result: Any,
    trace_model: str,
    scenario_horizon_s: Any,
) -> float | None:
    if trace_model == "terrestrial_network_surrogate":
        return _optional_float((engine_result.provenance or {}).get("effective_time_horizon_s"))
    if trace_model in {SURROGATE_ORBIT_TRACE, CIRCULAR_GEOMETRY_TRACE}:
        return _optional_float((engine_result.provenance or {}).get("trace_horizon_s"))
    return _optional_float(scenario_horizon_s)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


__all__ = ["build_runtime_performance_result"]
