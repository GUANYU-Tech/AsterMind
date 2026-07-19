"""Ground-baseline evaluation adapter outside the core runtime engine."""

from __future__ import annotations

from typing import Any

from orbitserve.runtime import RuntimeStrategySpec
from orbitserve.runtime import build_runtime_performance_result
from orbitserve.orbit import RuntimeTraceManifest
from .costing import estimate_terrestrial_ground_cost
from .global_runtime import is_centralized_ground_policy, run_centralized_ground_runtime
from .runtime import (
    TERRESTRIAL_GROUND_SCOPE,
    TERRESTRIAL_NETWORK_TRACE,
    run_ground_runtime,
)


def evaluate_ground_runtime_performance(scenario: Any, strategy: Any) -> Any:
    runtime_strategy = RuntimeStrategySpec.from_value(
        strategy or getattr(scenario, "runtime_config", None)
    )
    centralized = is_centralized_ground_policy(runtime_strategy.policy)
    if runtime_strategy.policy not in {"ground_regional", "skywalker_like_ground"} and not centralized:
        raise ValueError(f"not a terrestrial baseline policy: {runtime_strategy.policy}")
    runner = run_centralized_ground_runtime if centralized else run_ground_runtime
    engine_result = runner(
        requests=tuple(getattr(scenario, "requests", ()) or ()),
        runtime_strategy=runtime_strategy,
        regions=tuple(getattr(scenario, "regions", ()) or ()),
        gateways=tuple(getattr(scenario, "gateways", ()) or ()),
        policy=runtime_strategy.policy,
    )
    costing = estimate_terrestrial_ground_cost(
        records=engine_result.records,
        ground_config=runtime_strategy.ground_config,
    )
    trace_manifest = RuntimeTraceManifest.from_value(
        getattr(scenario, "simulation_trace_ref", None)
    )
    return build_runtime_performance_result(
        scenario=scenario,
        runtime_strategy=runtime_strategy,
        engine_result=engine_result,
        trace_manifest=trace_manifest,
        effective_trace_model=TERRESTRIAL_NETWORK_TRACE,
        effective_scope=TERRESTRIAL_GROUND_SCOPE,
        deployment_catalog=None,
        evaluator_name=(
            "skywalker_centralized_ground_discrete_event_surrogate"
            if centralized
            else "skywalker_ground_discrete_event_reproduction"
        ),
        path_model=(
            "centralized_or_gke_style_terrestrial_baseline"
            if centralized
            else "geo_distributed_two_layer_terrestrial_routing"
        ),
        energy_model="not_applicable_terrestrial_baseline",
        queue_model=(
            "centralized_blind_push_replica_queue"
            if centralized
            else "regional_lb_fcfs_pending_selective_pushing"
        ),
        link_model_status="terrestrial_internet_rtt_surrogate",
        energy_model_status="not_applicable",
        terrestrial_baseline=True,
        centralized_ground_baseline=centralized,
        terrestrial_costing=costing,
    )


__all__ = ["evaluate_ground_runtime_performance"]
