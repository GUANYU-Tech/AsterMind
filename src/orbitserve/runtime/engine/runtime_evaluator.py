"""Orbit runtime evaluator."""

from __future__ import annotations

from typing import Any

from ..contracts import RuntimePerformanceResult, RuntimeStrategySpec
from .compute_only import run_compute_only_fifo
from .trace_backed import run_trace_backed_runtime
from ..metrics.result_builder import build_runtime_performance_result
from ..scenario.runtime_deployments import build_runtime_deployment_catalog
from ..strategy.runtime_policies import get_runtime_policy
from ...orbit import (
    CIRCULAR_GEOMETRY_TRACE,
    IDEAL_FULL_VISIBILITY,
    SURROGATE_ORBIT_TRACE,
    RuntimeTraceManifest,
    load_surrogate_trace_bundle,
)


def try_evaluate_runtime_performance(
    scenario: Any,
    strategy: RuntimeStrategySpec | dict[str, Any] | Any | None = None,
) -> RuntimePerformanceResult | None:
    runtime_strategy = RuntimeStrategySpec.from_value(
        strategy or getattr(scenario, "runtime_config", None)
    )
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
    deployment_catalog = build_runtime_deployment_catalog(scenario)
    policy = get_runtime_policy(runtime_strategy)

    if trace_manifest.trace_model == IDEAL_FULL_VISIBILITY:
        if policy is None or runtime_strategy.policy != "local_only" or not deployment_catalog.default_deployment:
            return None
        engine_result = run_compute_only_fifo(
            requests=requests,
            deployment_catalog=deployment_catalog,
            policy=policy,
            trace_model=trace_manifest.trace_model,
            scope=trace_manifest.scope,
        )
        return build_runtime_performance_result(
            scenario=scenario,
            runtime_strategy=runtime_strategy,
            engine_result=engine_result,
            trace_manifest=trace_manifest,
            effective_trace_model=trace_manifest.trace_model,
            effective_scope=trace_manifest.scope,
            deployment_catalog=deployment_catalog,
            evaluator_name="compute_only_fifo",
            path_model="ideal_full_visibility_no_external_link",
            energy_model="not_implemented",
            queue_model="fifo_work_seconds",
            link_model_status="not_implemented",
            energy_model_status="not_implemented",
        )

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
    engine_result = run_trace_backed_runtime(
    deployment_catalog = build_runtime_deployment_catalog(
        scenario,
        satellite_catalog=tuple(getattr(trace_bundle, "satellite_catalog", ()) or ()),
    )
    if not deployment_catalog.default_deployment:
        return None
        requests=requests,
        deployment_catalog=deployment_catalog,
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
        deployment_catalog=deployment_catalog,
        evaluator_name="circular_geometry_trace_backed_runtime" if circular else "surrogate_trace_backed_runtime",
        path_model="circular_geometry_trace_sgl_isl" if circular else "surrogate_trace_sgl_isl",
        energy_model="circular_geometry_shadow_dvfs" if circular else "surrogate_soc_shadow_dvfs",
        queue_model="per_satellite_fifo",
        link_model_status="surrogate_link_queue",
        energy_model_status="surrogate_soc",
    )


__all__ = ["try_evaluate_runtime_performance"]
