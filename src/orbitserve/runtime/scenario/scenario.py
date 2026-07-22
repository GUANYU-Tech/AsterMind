"""Materialize runtime scenarios from evaluated subsystem outputs."""

from __future__ import annotations

import json
from typing import Any

from ...orbit import RuntimeTraceManifest, resolve_trace_path
from ...common import to_jsonable
from ..contracts import (
    RUNTIME_INTERFACE_DOC,
    RUNTIME_PERFORMANCE_DOC,
    RUNTIME_OPTIMIZATION_DOC,
    RuntimeStrategySpec,
    ServingDeploymentView,
)
from .runtime_scenario import RuntimeScenario


def build_runtime_scenario(
    *,
    scenario_id: str | None = None,
    design_id: str | None = None,
    workload_generation: Any = None,
    workload_spec: Any = None,
    constellation: Any = None,
    constellation_architecture: Any = None,
    serving_spec: Any = None,
    serving_throughput: Any = None,
    simulation_result: Any = None,
    simulation_config: Any = None,
    cost_result: Any = None,
    scheduler_spec: Any = None,
    hardware: Any = None,
    resolved_hardware: Any = None,
    time_horizon_s: float | None = None,
    warnings: tuple[str, ...] = (),
) -> RuntimeScenario:
    workload_payload = _mapping_from_result(workload_generation)
    workload_spec_payload = _mapping_from_result(workload_spec)
    simulation_payload = _mapping_from_result(simulation_result)
    simulation_config_payload = _mapping_from_result(simulation_config)
    cost_payload = _mapping_from_result(cost_result)
    constellation_payload = _mapping_from_result(constellation)
    architecture_payload = _mapping_from_result(constellation_architecture)
    runtime_strategy = RuntimeStrategySpec.from_value(scheduler_spec)
    resolved_hardware_payload = _mapping_from_result(resolved_hardware)
    hardware_payload = _mapping_from_result(hardware)
    scenario_name = scenario_id or str(_first_present(workload_spec_payload.get("name"), "runtime_scenario"))
    design_name = design_id or str(
        _first_present(
            constellation_payload.get("name"),
            architecture_payload.get("name"),
            scenario_name,
        )
    )
    requests = tuple(_dict(item) for item in workload_payload.get("requests", ()) if isinstance(item, dict))
    sessions = tuple(_dict(item) for item in workload_payload.get("sessions", ()) if isinstance(item, dict))
    regions = tuple(_dict(item) for item in workload_payload.get("regions", ()) if isinstance(item, dict))
    gateways = tuple(_dict(item) for item in workload_payload.get("gateways", ()) if isinstance(item, dict))
    manifest = _dict(workload_payload.get("manifest"))
    summary = _dict(workload_payload.get("summary"))
    request_count = _optional_int(workload_payload.get("request_count")) or len(requests) or None
    source = _dict(manifest.get("source"))
    source_mode = _first_present(
        source.get("source_mode"),
        manifest.get("source_mode"),
        summary.get("source_mode"),
        workload_payload.get("source_mode"),
    )
    if time_horizon_s is None:
        time_horizon_s = _optional_float(
            _first_present(
                simulation_config_payload.get("duration_s"),
                workload_spec_payload.get("duration_s"),
                _dict(manifest.get("time")).get("time_horizon_s"),
            )
        )
    deployment = ServingDeploymentView.from_inputs(
        deployment_id="deployment_0",
        serving_spec=serving_spec,
        throughput_estimate=serving_throughput,
    )
    workload_configured = _workload_bundle_configured(workload_payload, requests, manifest)
    workload_bundle = {
        "status": workload_payload.get("status", "materialized_spec" if workload_spec_payload else "not_configured"),
        "configured": workload_configured,
        "bundle_dir": workload_payload.get("bundle_dir"),
        "schema_version": manifest.get("schema_version"),
        "source_mode": source_mode,
        "files": _dict(manifest.get("files")) or _dict(workload_payload.get("files")),
        "manifest": manifest,
        "summary": summary,
        "request_count": request_count,
        "session_count": _optional_int(workload_payload.get("session_count")) or len(sessions) or None,
        "workload_spec": workload_spec_payload,
        "warnings": _warnings_from_payload(workload_payload),
    }
    trace_manifest_obj = RuntimeTraceManifest.from_value(simulation_payload)
    trace_manifest = trace_manifest_obj.to_dict()
    satellite_catalog = _satellite_catalog_from_trace_manifest(trace_manifest_obj)
    trace_model = _first_present(
        simulation_payload.get("trace_model"),
        trace_manifest.get("trace_model"),
        "skeleton",
    )
    simulation_evaluated = bool(
        simulation_payload.get(
            "trace_evaluated",
            simulation_payload.get("evaluated", trace_manifest.get("evaluated", False)),
        )
    )
    simulation_trace_ref = {
        "evaluated": simulation_evaluated,
        "status": "evaluated" if simulation_evaluated else "skeleton",
        "config": simulation_config_payload,
        "result": simulation_payload,
        "trace_manifest": trace_manifest,
        "trace_model": trace_model,
        "trace_id": trace_manifest.get("trace_id"),
        "access_trace_sha256": trace_manifest.get("access_trace_sha256"),
        "link_trace_sha256": trace_manifest.get("link_trace_sha256"),
        "energy_trace_sha256": trace_manifest.get("energy_trace_sha256"),
        "files": _dict(trace_manifest.get("files")),
        "sha256": _dict(trace_manifest.get("sha256")),
        "time": _dict(trace_manifest.get("time")),
        "calibration": _dict(trace_manifest.get("calibration")),
        "scope": trace_manifest.get("scope"),
        "valid_for_paper_main_results": bool(trace_manifest.get("valid_for_paper_main_results", False)),
    }
    cost_power = {
        "evaluated": bool(cost_payload.get("evaluated", False)),
        "satellite_count": cost_payload.get("satellite_count"),
        "total_cost_usd": cost_payload.get("total_cost_usd"),
        "annualized_cost_usd_per_year": cost_payload.get("annualized_cost_usd_per_year"),
        "total_mass_kg": cost_payload.get("total_mass_kg"),
        "peak_power_w": cost_payload.get("peak_power_w"),
        "single_satellite": cost_payload.get("single_satellite"),
        "provenance": cost_payload.get("provenance", {}),
    }
    scenario_warnings = tuple(
        dict.fromkeys(
            (
                *warnings,
                *_warnings_from_payload(workload_payload),
                *_warnings_from_payload(simulation_payload),
                *_warnings_from_payload(cost_payload),
                *deployment.warnings,
            )
        )
    )
    return RuntimeScenario(
        scenario_id=scenario_name,
        design_id=design_name,
        time_horizon_s=time_horizon_s,
        time_step_s=_optional_float(simulation_config_payload.get("step_s")),
        workload_bundle=workload_bundle,
        requests=requests,
        sessions=sessions,
        regions=regions,
        gateways=gateways,
        simulation_trace_ref=simulation_trace_ref,
        constellation={
            "design": constellation_payload,
            "architecture": architecture_payload,
            "satellite_count": constellation_payload.get("satellite_count"),
            **({"satellite_catalog": list(satellite_catalog)} if satellite_catalog else {}),
        },
        serving_deployments=(deployment,),
        cost_power=cost_power,
        runtime_config=runtime_strategy,
        provenance={
            "workload_bundle": {
                "bundle_dir": workload_bundle["bundle_dir"],
                "request_count": request_count,
                "source_mode": source_mode,
            },
            "simulation_trace": {
                "trace_id": simulation_trace_ref["trace_id"],
                "trace_model": trace_model,
                "evaluated": simulation_evaluated,
                "valid_for_paper_main_results": simulation_trace_ref["valid_for_paper_main_results"],
            },
            "serving_model": {
                "modeling_doc": "docs/research/parameters/llm_serving_capacity_model.md",
                "estimate_schema": "ServingThroughputEstimate",
            },
            "hardware_config": resolved_hardware_payload or hardware_payload,
            "cost_model": cost_power["provenance"],
            "runtime_interface_doc": RUNTIME_INTERFACE_DOC,
            "runtime_performance_model_doc": RUNTIME_PERFORMANCE_DOC,
            "runtime_optimization_model_doc": RUNTIME_OPTIMIZATION_DOC,
        },
        warnings=scenario_warnings,
    )

def _workload_bundle_configured(
    payload: dict[str, Any],
    requests: tuple[dict[str, Any], ...],
    manifest: dict[str, Any],
) -> bool:
    if "configured" in payload:
        return bool(payload.get("configured"))
    return bool(payload.get("bundle_dir") or requests or manifest)

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

def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}

def _warnings_from_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    raw = payload.get("warnings", ())
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw)

def _satellite_catalog_from_trace_manifest(trace_manifest: RuntimeTraceManifest) -> tuple[dict[str, Any], ...]:
    trace_dir = _optional_str(trace_manifest.provenance.get("trace_dir"))
    satellite_catalog = _read_satellite_catalog_json(
        resolve_trace_path(
            trace_manifest.satellite_catalog_path,
            manifest_dir=trace_dir,
        )
    )
    if satellite_catalog:
        return satellite_catalog
    access_trace_path = resolve_trace_path(
        trace_manifest.access_trace_path,
        manifest_dir=trace_dir,
    )
    if access_trace_path is None or not access_trace_path.exists():
        return ()
    try:
        payload = json.loads(access_trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    return _satellite_catalog_from_value(payload.get("satellite_catalog"))

def _read_satellite_catalog_json(path: Any) -> tuple[dict[str, Any], ...]:
    if path is None or not path.exists():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    return _satellite_catalog_from_value(payload)

def _satellite_catalog_from_value(payload: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(payload, list):
        return ()
    return tuple(dict(item) for item in payload if isinstance(item, dict))

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
        return int(value)
    except (TypeError, ValueError):
        return None

def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None

def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None
