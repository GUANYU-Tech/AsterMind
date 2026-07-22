"""Runtime evaluation entry point and conservative fallback."""

from __future__ import annotations

from typing import Any

from .contracts import (
    RUNTIME_INTERFACE_DOC,
    RUNTIME_PERFORMANCE_DOC,
    RUNTIME_RESULT_FIELDS,
    RuntimePerformanceResult,
    RuntimeStrategySpec,
)
from .scenario.runtime_scenario import RuntimeScenario


def evaluate_runtime_performance(
    scenario: RuntimeScenario,
    strategy: RuntimeStrategySpec | dict[str, Any] | Any | None = None,
) -> RuntimePerformanceResult:
    runtime_strategy = RuntimeStrategySpec.from_value(strategy or scenario.runtime_config)
    leakage = _runtime_leakage_records(scenario.requests, scenario.sessions)
    if leakage:
        leaked_fields = sorted({field for record in leakage for field in record["fields"]})
        raise ValueError(
            "workload runtime leakage detected in canonical bundle: "
            + ", ".join(leaked_fields)
        )
    from .engine.runtime_evaluator import try_evaluate_runtime_performance

    evaluated_result = try_evaluate_runtime_performance(scenario, runtime_strategy)
    if evaluated_result is not None:
        return evaluated_result
    serving_profile = scenario.serving_profile
    serving_evaluated = bool(serving_profile and serving_profile.evaluated)
    serving_feasible = bool(serving_profile and serving_profile.feasible is not False)
    simulation_evaluated = bool(scenario.simulation_trace_ref.get("evaluated", False))
    workload_status = _workload_status(scenario, leakage)
    simulation_status = "evaluated" if simulation_evaluated else "skeleton"
    serving_status = (
        "missing"
        if serving_profile is None
        else "infeasible"
        if not serving_feasible
        else "evaluated"
        if serving_evaluated
        else "missing"
    )
    cost_status = "annualized" if scenario.cost_power.get("annualized_cost_usd_per_year") is not None else "static_cost_only"
    warnings = list(scenario.warnings)
    if not serving_evaluated:
        warnings.append("serving_throughput.evaluated=false; cannot compute service time")
    if serving_evaluated and not serving_feasible:
        warnings.append("serving_throughput.feasible=false; runtime cannot admit LEO actions")
    if not simulation_evaluated:
        warnings.append("simulation trace is not implemented; runtime observed metrics are null")
    warnings.append("runtime event simulator is not implemented; completion_ratio and observed latency metrics are null")
    total_request_count = len(scenario.requests) or scenario.workload_bundle.get("request_count")
    bottleneck_breakdown: dict[str, Any] = {}
    bottleneck_breakdown["not_implemented"] = total_request_count
    module_status = {
        "workload": workload_status,
        "simulation": simulation_status,
        "serving_throughput": serving_status,
        "runtime_strategy": "provided",
        "queue_model": "not_implemented",
        "link_model": "not_implemented",
        "energy_model": "not_implemented",
        "cost_model": cost_status,
    }
    config = {
        "time_mode": scenario.time_mode,
        "time_step_s": scenario.time_step_s,
        "warmup_s": 0,
        "fallback_accounting": "separate_from_leo_only",
        "internal_comm_accounting": (
            serving_profile.internal_comm_accounting if serving_profile else "embedded"
        ),
        "kv_accounting": "active_plus_idle_session",
        "recompute_input_bytes_policy": "new_user_tokens_only",
        "path_model": "store_and_forward",
        "queue_model": "work_seconds_fifo",
        "energy_lookahead": {
            "h_lookahead_s": runtime_strategy.h_lookahead_s,
            "lookahead_orbits": runtime_strategy.lookahead_orbits,
            "forecast_step_s": None,
            "soc_min_policy": "per_satellite",
        },
    }
    runtime_metrics = {
        "total_request_count": total_request_count,
        "completion_ratio": None,
        "uncompleted_ratio": None,
        "uncompleted_stage_breakdown": {},
        "uncompleted_reason_breakdown": {},
        "fallback_request_fraction": None,
        "ttft_p50_s": None,
        "ttft_p95_s": None,
        "ttft_p99_s": None,
        "tpot_p50_s": None,
        "tpot_p95_s": None,
        "tpot_p99_s": None,
        "e2e_p50_s": None,
        "e2e_p95_s": None,
        "e2e_p99_s": None,
        "useful_request_throughput_per_s": None,
        "useful_token_throughput_per_s": None,
        "prefill_queue_wait_p95_s": None,
        "decode_queue_wait_p95_s": None,
        "link_queue_wait_p95_s": None,
        "access_wait_p95_s": None,
        "eta_serving": None,
        "eta_prefill_capacity": None,
        "eta_decode_capacity": None,
        "eta_compute_useful": None,
    }
    serving_summary = serving_profile.to_dict() if serving_profile else {}
    return RuntimePerformanceResult(
        evaluated=False,
        scenario={
            "schema_version": scenario.schema_version,
            "scenario_id": scenario.scenario_id,
            "design_id": scenario.design_id,
            "time_horizon_s": scenario.time_horizon_s,
            "simulation_evaluated": simulation_evaluated,
        },
        strategy=runtime_strategy.to_dict(),
        workload=scenario.workload_bundle,
        constellation=scenario.constellation,
        serving=serving_summary,
        scheduler=runtime_strategy.to_dict(),
        config=config,
        runtime_metrics=runtime_metrics,
        bottleneck_breakdown=bottleneck_breakdown,
        session_metrics=_unevaluated_metric_group(
            "session metrics require runtime event simulation and KV state tracking"
        ),
        communication_metrics=_unevaluated_metric_group(
            "communication metrics require SGL/ISL path and queue traces"
        ),
        energy_metrics=_unevaluated_metric_group("energy metrics require SOC trace integration"),
        region_metrics=(),
        prediction_error=_unevaluated_metric_group("observed metrics unavailable; prediction_error is not evaluated"),
        non_separability=_unevaluated_metric_group(
            "non-separability metrics require a design x scheduler runtime matrix"
        ),
        decision_trace={
            "trace_schema_version": "orbitserve.runtime_decision_trace.v1",
            "evaluated": False,
            "trace_level": runtime_strategy.trace_level,
            "request_records_path": None,
            "warnings": ["request-level runtime decision trace is not implemented"],
        },
        module_status=module_status,
        provenance={
            **scenario.provenance,
            "runtime_interface_schema": "orbitserve.runtime_scenario.v1",
            "performance_evaluator": "conservative_skeleton",
            "runtime_performance_schema": "orbitserve.runtime_performance.v1",
            "runtime_interface_doc": RUNTIME_INTERFACE_DOC,
            "runtime_performance_model_doc": RUNTIME_PERFORMANCE_DOC,
        },
        warnings=tuple(dict.fromkeys(warnings)),
    )

def _runtime_leakage_records(
    requests: tuple[dict[str, Any], ...],
    sessions: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    leaks: list[dict[str, Any]] = []
    for collection_name, rows in (("requests", requests), ("sessions", sessions)):
        for index, row in enumerate(rows):
            leaked = sorted(set(row) & RUNTIME_RESULT_FIELDS)
            if leaked:
                leaks.append(
                    {
                        "collection": collection_name,
                        "index": index,
                        "request_id": row.get("request_id"),
                        "session_id": row.get("session_id"),
                        "fields": leaked,
                    }
                )
    return tuple(leaks)

def _workload_status(scenario: RuntimeScenario, leakage: tuple[dict[str, Any], ...]) -> str:
    if leakage:
        return "invalid"
    status = str(scenario.workload_bundle.get("status") or "")
    if status in {"loaded", "generated"}:
        return status
    if scenario.requests:
        return "loaded"
    if status:
        return status
    return "not_configured"

def _unevaluated_metric_group(reason: str) -> dict[str, Any]:
    return {
        "evaluated": False,
        "value": None,
        "warnings": (reason,),
    }
