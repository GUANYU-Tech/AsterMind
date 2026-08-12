"""Experiment runner that wires framework modules together."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from orbitserve.common import to_jsonable
from orbitserve.experiments.design_evaluation import evaluate_materialized_design
from orbitserve.dse.search import DSEStudyResult, run_dse
from orbitserve.dse.satellite_budget import analyze_communication_window_satellite_budget

try:
    from orbitserve.workload import materialize_workload_bundle
except ImportError:  # pragma: no cover - import failure is reported at execution.
    materialize_workload_bundle = None

from .models import ExperimentResult, ExperimentSpec


def run_experiment(spec: ExperimentSpec) -> ExperimentResult:
    """Run the currently implemented portions of an OrbitServe experiment."""

    base_dir = Path(spec.base_dir) if spec.base_dir else None
    workload_generation, serving_workload, workload_generation_warnings = _materialize_workload_for_runner(
        spec.workload,
        base_dir=base_dir,
    )
    evaluation = evaluate_materialized_design(
        spec,
        design=spec.constellation,
        workload_generation=workload_generation,
        serving_workload=serving_workload,
        evaluation_id=spec.experiment_id,
        input_warnings=workload_generation_warnings,
    )
    constellation_architecture = evaluation.constellation_architecture
    resolved_hardware = evaluation.resolved_hardware
    cost = evaluation.cost
    simulation = evaluation.simulation
    serving_throughput = evaluation.serving_throughput
    communication_window_budget_config = _communication_window_budget_config(spec)
    communication_window_budget_analysis = (
        analyze_communication_window_satellite_budget(
            simulation_result=simulation,
            workload_generation=workload_generation,
            serving_throughput=serving_throughput,
            config=communication_window_budget_config,
            base_dir=base_dir,
        )
        if communication_window_budget_config
        else None
    )
    runtime_performance = evaluation.runtime_performance
    performance = serving_throughput.to_performance_metrics()
    performance = _performance_with_runtime_metrics(performance, runtime_performance)
    performance = replace(
        performance,
        details={
            **performance.details,
            "runtime": runtime_performance.to_dict(),
            **(
                {"communication_window_budget_analysis": communication_window_budget_analysis}
                if communication_window_budget_analysis is not None
                else {}
            ),
        },
        warnings=tuple(
            dict.fromkeys(
                (
                    *performance.warnings,
                    *runtime_performance.warnings,
                    *(
                        communication_window_budget_analysis.get("warnings", ())
                        if communication_window_budget_analysis is not None
                        else ()
                    ),
                )
            )
        ),
    )
    dse = _run_dse_for_experiment(
        spec,
        workload_generation=workload_generation,
        serving_workload=serving_workload,
        workload_generation_warnings=workload_generation_warnings,
    )
    slo = simulation.slo
    module_status = {
        "constellation_architecture": "evaluated"
        if constellation_architecture.get("evaluated")
        else "not_configured",
        "costing": "evaluated" if cost.evaluated else "not_configured",
        "hardware": "resolved"
        if resolved_hardware is not None
        else "not_configured",
        "workload_generation": str(workload_generation.get("status", "unknown")),
        "simulation": _simulation_status(simulation),
        "serving_throughput": "evaluated"
        if serving_throughput.evaluated
        else "not_configured",
        "runtime_performance": "evaluated"
        if runtime_performance.evaluated
        else "skeleton",
        "communication_window_budget_analysis": "evaluated"
        if communication_window_budget_analysis is not None
        and communication_window_budget_analysis.get("evaluated")
        else "not_configured"
        if communication_window_budget_analysis is None
        else "unavailable",
        "dse": "skeleton" if spec.dse.enabled else "disabled",
    }
    warnings = _dedupe((
        *evaluation.warnings,
        *runtime_performance.warnings,
        *(
            communication_window_budget_analysis.get("warnings", ())
            if communication_window_budget_analysis is not None
            else ()
        ),
        *dse.warnings,
    ))
    return ExperimentResult(
        experiment_id=spec.experiment_id,
        manifest=_manifest_with_constellation_architecture(
            spec,
            constellation_architecture,
        ),
        constellation_architecture=constellation_architecture,
        cost=cost,
        performance=performance,
        slo=slo,
        dse=dse,
        module_status=module_status,
        warnings=warnings,
        workload_generation=workload_generation,
    )


def _manifest_with_constellation_architecture(
    spec: ExperimentSpec,
    constellation_architecture: dict[str, Any],
) -> dict[str, Any]:
    manifest = spec.to_dict()
    constellation = dict(manifest.get("constellation") or {})
    constellation["architecture_analysis"] = to_jsonable(constellation_architecture)
    manifest["constellation"] = constellation
    return manifest


def _run_dse_for_experiment(
    spec: ExperimentSpec,
    *,
    workload_generation: dict[str, Any],
    serving_workload: Any,
    workload_generation_warnings: tuple[str, ...],
) -> DSEStudyResult:
    if not spec.dse.enabled:
        return DSEStudyResult()
    return run_dse(
        spec,
        workload_generation=workload_generation,
        serving_workload=serving_workload,
        workload_warnings=workload_generation_warnings,
    )


def _performance_with_runtime_metrics(performance: Any, runtime_performance: Any) -> Any:
    if not getattr(runtime_performance, "evaluated", False):
        return performance
    inference_metrics = dict(getattr(runtime_performance, "inference_metrics", {}) or {})
    request_summary = dict(getattr(runtime_performance, "request_summary", {}) or {})
    provenance = dict(getattr(runtime_performance, "provenance", {}) or {})
    warnings = tuple(
        dict.fromkeys(
            (
                *getattr(performance, "warnings", ()),
                *tuple(getattr(runtime_performance, "warnings", ())),
                "top-level performance contains deterministic Runtime metrics",
            )
        )
    )
    return replace(
        performance,
        evaluated=True,
        inference_metrics=inference_metrics,
        request_summary=request_summary,
        bottleneck=performance.bottleneck,
        details={
            **performance.details,
            "runtime_scope": provenance.get("trace_scope"),
            "runtime_trace_model": provenance.get("trace_model"),
            "valid_for_paper_main_results": provenance.get("valid_for_paper_main_results", False),
        },
        warnings=warnings,
    )


def _simulation_status(simulation: Any) -> str:
    if not getattr(simulation, "trace_evaluated", False):
        return "stub"
    if getattr(simulation, "trace_model", None) == "surrogate_orbit_trace":
        return "surrogate_trace"
    if getattr(simulation, "trace_model", None) == "circular_geometry_trace":
        return "geometry_trace"
    return "idealized_trace"


_WORKLOAD_CONFIGURATION_FIELDS = (
    "bundle_dir",
)


def _materialize_workload_for_runner(
    workload_ref: Any,
    *,
    base_dir: Path | None,
) -> tuple[dict[str, Any], Any, tuple[str, ...]]:
    if not _workload_generation_configured(workload_ref):
        status = "not_configured"
        return (
            {
                "status": status,
                "configured": False,
                "applied_to_serving": False,
                "representative_token_distribution": None,
                "warnings": (),
            },
            None,
            (f"workload_generation status: {status}",),
        )
    if materialize_workload_bundle is None:
        raise RuntimeError("orbitserve.workload.materialize_workload_bundle is unavailable")
    result = materialize_workload_bundle(workload_ref, base_dir=base_dir)
    payload = _mapping_from_result(result)
    status = str(payload.get("status") or "evaluated")
    result_warnings = _warnings_from_payload(payload)
    summary = _workload_summary_from_result(result, payload, base_dir=base_dir)
    token_values = _representative_token_distribution(summary)
    token_distribution = None
    if token_values is not None:
        from orbitserve.workload import TokenDistribution
        token_distribution = TokenDistribution.from_dict(token_values)
        token_warning = "workload representative token distribution applied to serving throughput"
    else:
        token_warning = "workload summary lacks complete prompt/output p50/p95; serving uses defaults"
    warnings = [f"workload_generation status: {status}", *result_warnings, token_warning]
    payload = dict(payload)
    payload.setdefault("status", status)
    payload["configured"] = True
    payload["applied_to_serving"] = token_distribution is not None
    payload["representative_token_distribution"] = token_values
    payload["warnings"] = tuple(dict.fromkeys(warnings[1:]))
    return to_jsonable(payload), token_distribution, tuple(dict.fromkeys(warnings))


def _workload_generation_configured(workload_ref: Any) -> bool:
    return _configured_value(getattr(workload_ref, "bundle_dir", None))


def _communication_window_budget_config(spec: ExperimentSpec) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    options = getattr(spec.simulation, "trace_options", {}) or {}
    if isinstance(options, dict):
        for key in (
            "communication_window_budget",
            "communication_window_budget_analysis",
            "satellite_budget_analysis",
        ):
            value = options.get(key)
            if isinstance(value, dict):
                merged.update(value)
    return merged if bool(merged.get("enabled", False)) else {}


def _configured_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return bool(value)


def _mapping_from_result(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return to_jsonable(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = to_jsonable(value.to_dict())
        return converted if isinstance(converted, dict) else {}
    converted = to_jsonable(value)
    return converted if isinstance(converted, dict) else {}


def _warnings_from_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    raw_warnings = payload.get("warnings", ())
    if isinstance(raw_warnings, str):
        return (raw_warnings,)
    if raw_warnings is None:
        return ()
    return tuple(str(warning) for warning in raw_warnings)


def _dedupe(values: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if value is not None))


def _workload_summary_from_result(
    result: Any,
    payload: dict[str, Any],
    *,
    base_dir: Path | None,
) -> dict[str, Any]:
    for candidate in (
        getattr(result, "summary", None),
        payload.get("summary"),
        payload.get("workload_summary"),
        payload.get("summary_stats"),
    ):
        mapping = _mapping_from_result(candidate)
        if mapping:
            return mapping

    for path_value in (
        getattr(result, "summary_path", None),
        payload.get("summary_path"),
        payload.get("workload_summary_path"),
        *_summary_files_from_payload(payload),
    ):
        summary = _read_summary_file(path_value, base_dir=base_dir)
        if summary:
            return summary
    return {}


def _summary_files_from_payload(payload: dict[str, Any]) -> tuple[Any, ...]:
    files = payload.get("files")
    if not isinstance(files, dict):
        return ()
    summary = files.get("summary") or files.get("workload_summary")
    if not _configured_value(summary):
        return ()
    candidates = [summary]
    bundle_dir = payload.get("bundle_dir")
    if _configured_value(bundle_dir):
        summary_path = Path(str(summary))
        if not summary_path.is_absolute():
            candidates.append(Path(str(bundle_dir)) / summary_path)
    return tuple(candidates)


def _read_summary_file(path_value: Any, *, base_dir: Path | None) -> dict[str, Any]:
    if not _configured_value(path_value):
        return {}
    summary_path = Path(str(path_value))
    if not summary_path.is_absolute() and base_dir is not None:
        summary_path = base_dir / summary_path
    if not summary_path.exists() or not summary_path.is_file():
        return {}
    with summary_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def _representative_token_distribution(summary: dict[str, Any]) -> dict[str, int] | None:
    if not summary:
        return None

    containers = _token_summary_containers(summary)
    prompt_p50 = _first_number(containers, "prompt", "p50")
    prompt_p95 = _first_number(containers, "prompt", "p95")
    output_p50 = _first_number(containers, "output", "p50")
    output_p95 = _first_number(containers, "output", "p95")
    if None in (prompt_p50, prompt_p95, output_p50, output_p95):
        return None
    return {
        "prompt_tokens_p50": int(prompt_p50),
        "prompt_tokens_p95": int(prompt_p95),
        "output_tokens_p50": int(output_p50),
        "output_tokens_p95": int(output_p95),
    }


def _token_summary_containers(summary: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    containers: list[dict[str, Any]] = [summary]
    for key in (
        "token_stats",
        "token_distribution",
        "tokens",
        "token_summary",
        "request_tokens",
        "summary",
    ):
        value = summary.get(key)
        if isinstance(value, dict):
            containers.append(value)
    return tuple(containers)


def _first_number(containers: tuple[dict[str, Any], ...], side: str, quantile: str) -> int | None:
    direct_keys = (
        f"{side}_tokens_{quantile}",
        f"{side}_{quantile}",
    )
    nested_keys = {
        "prompt": ("prompt_tokens", "prompt_distribution", "prompt", "input_tokens", "input_distribution", "input"),
        "output": (
            "output_tokens",
            "output_distribution",
            "output",
            "completion_tokens",
            "completion_distribution",
            "completion",
        ),
    }[side]
    for container in containers:
        for key in direct_keys:
            value = _positive_int(container.get(key))
            if value is not None:
                return value
        for key in nested_keys:
            nested = container.get(key)
            if isinstance(nested, dict):
                value = _positive_int(nested.get(quantile))
                if value is not None:
                    return value
    return None


def _positive_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None
