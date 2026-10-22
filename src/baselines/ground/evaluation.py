"""Evaluate terrestrial baseline performance, metrics and cost."""

from types import SimpleNamespace
from orbitserve.common import to_jsonable
from .contracts import GroundScenario
from .global_runtime import (
    is_centralized_ground_policy,
    run_centralized_ground_runtime,
)
from .runtime import (
    run_ground_runtime,
    _ground_runtime_metrics,
    _ground_completed_throughput_metrics,
    _effective_horizon_s,
)
from .costing import estimate_terrestrial_ground_cost


class GroundEvaluation(SimpleNamespace):
    def to_dict(self):
        return to_jsonable(vars(self))


def aggregate_ground_records(records, *, time_horizon_s=None, **unused):
    rows = tuple(records)
    metrics = _ground_runtime_metrics(rows)
    metrics.update(
        _ground_completed_throughput_metrics(
            rows, effective_horizon_s=time_horizon_s or _effective_horizon_s(rows)
        )
    )
    served = sum(bool(r.served) for r in rows)
    metrics.update(
        leo_served_ratio=0.0,
        terrestrial_served_ratio=served / len(rows) if rows else 0.0,
        fallback_request_fraction=0.0,
    )
    return GroundEvaluation(
        runtime_metrics=metrics,
        communication_metrics={
            "evaluated": True,
            "model": "terrestrial_internet_rtt_surrogate",
        },
        energy_metrics={"evaluated": False, "not_applicable": True},
    )


def evaluate_ground_scenario(scenario, strategy=None):
    if not isinstance(scenario, GroundScenario):
        raise TypeError("expected GroundScenario")
    strategy = strategy or scenario.runtime_config
    runner = (
        run_centralized_ground_runtime
        if is_centralized_ground_policy(strategy.policy)
        else run_ground_runtime
    )
    result = runner(
        requests=scenario.requests,
        runtime_strategy=strategy,
        regions=scenario.regions,
        gateways=scenario.gateways,
        policy=strategy.policy,
    )
    metrics = dict(result.runtime_metrics or {})
    horizon = scenario.time_horizon_s or _effective_horizon_s(result.records)
    metrics.update(
        _ground_completed_throughput_metrics(
            result.records, effective_horizon_s=horizon
        )
    )
    metrics["useful_request_throughput_per_s"] = (
        sum(bool(r.served) for r in result.records) / horizon if horizon else 0.0
    )
    metrics["useful_token_throughput_per_s"] = (
        sum((r.prompt_tokens + r.output_tokens) for r in result.records if r.served)
        / horizon
        if horizon
        else 0.0
    )
    metrics.update(
        leo_served_ratio=0.0,
        terrestrial_served_ratio=sum(bool(r.served) for r in result.records)
        / len(result.records)
        if result.records
        else 0.0,
        fallback_request_fraction=0.0,
    )
    cost = estimate_terrestrial_ground_cost(
        records=result.records,
        ground_config=strategy.ground_config,
        runtime_observation_window_s=scenario.time_horizon_s
        or strategy.ground_config.get("terrestrial_costing", {}).get(
            "workload_observation_period_s"
        ),
    )
    costing = cost.to_dict()
    if cost.evaluated:
        metrics.update(cost.runtime_metrics())
        metrics.update(
            terrestrial_total_cost_usd=costing.get("total_cost_usd"),
            terrestrial_annualized_cost_usd_per_year=costing.get(
                "annualized_cost_usd_per_year"
            ),
            cost_per_served_request_usd=costing.get("cost_per_served_request_usd"),
            cost_per_served_mtoken_usd=costing.get("cost_per_served_mtoken_usd"),
        )
    return GroundEvaluation(
        evaluated=True,
        scenario={
            "scenario_id": scenario.scenario_id,
            "simulation_evaluated": False,
            "simulation_required": False,
        },
        constellation={},
        runtime_metrics=metrics,
        communication_metrics=result.communication_metrics,
        energy_metrics=result.energy_metrics,
        request_records=tuple(r.to_dict() for r in result.records),
        decision_trace=result.decision_traces,
        state_summary=result.state_summary,
        trace_model=result.trace_model,
        scope=result.scope,
        valid_for_paper_main_results=False,
        module_status={
            "simulation": "not_required_terrestrial_baseline",
            "queue_model": "centralized_blind_push_replica_queue"
            if is_centralized_ground_policy(strategy.policy)
            else "regional_lb_fcfs_pending_selective_pushing",
            "link_model": "terrestrial_internet_rtt_surrogate",
            "energy_model": "not_applicable",
            "cost_model": "terrestrial_cost_evaluated"
            if cost.evaluated
            else "terrestrial_cost_not_configured",
        },
        details={"terrestrial_costing": costing},
        provenance={
            **(result.provenance or {}),
            "terrestrial_costing": costing,
            "runtime_engine": {
                "leo_trace_consumed": False,
                "leo_cost_consumed": False,
                "policy": strategy.policy,
            },
        },
        warnings=result.warnings,
    )
