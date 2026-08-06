"""Assemble the immutable input consumed by Runtime for one experiment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from orbitserve.common import to_jsonable
from orbitserve.orbit import OrbitTraceBundle, OrbitTraceManifest
from orbitserve.runtime import RuntimeSimulationSpec, RuntimeStrategySpec


def build_simulation_spec(
    *,
    scenario_id: str | None = None,
    design_id: str | None = None,
    workload_generation: Any = None,
    workload_spec: Any = None,
    constellation: Any = None,
    constellation_architecture: Any = None,
    serving_profile: Any = None,
    simulation_result: Any = None,
    simulation_config: Any = None,
    energy_profile: Any = None,
    strategy: Any = None,
    time_horizon_s: float | None = None,
    warnings: tuple[str, ...] = (),
) -> RuntimeSimulationSpec:
    workload_payload = _mapping(workload_generation)
    workload_spec_payload = _mapping(workload_spec)
    constellation_payload = _mapping(constellation)
    architecture_payload = _mapping(constellation_architecture)
    simulation_config_payload = _mapping(simulation_config)
    energy_payload = _mapping(energy_profile)
    environment_state = simulation_result if isinstance(simulation_result, OrbitTraceBundle) else None
    trace_manifest = _trace_manifest(simulation_result, environment_state)
    trace_payload = trace_manifest.to_dict() if trace_manifest is not None else {}

    requests = _rows(workload_payload.get("requests"))
    sessions = _rows(workload_payload.get("sessions"))
    regions = _rows(workload_payload.get("regions"))
    gateways = _rows(workload_payload.get("gateways"))
    manifest = _mapping(workload_payload.get("manifest"))
    summary = _mapping(workload_payload.get("summary"))
    communication_payload = _communication_ingress_payload(
        trace_payload,
        simulation_config_payload,
        constellation_payload,
    )
    communication_ingress = {
        "altitude_km": _positive_optional(
            _first(communication_payload.get("altitude_km"), communication_payload.get("orbit_altitude_km"))
        ),
        "uplink_bandwidth_bps": _positive_optional(
            _first(communication_payload.get("uplink_bandwidth_bps"), communication_payload.get("bandwidth_bps"))
        ),
        "downlink_bandwidth_bps": _positive_optional(
            _first(communication_payload.get("downlink_bandwidth_bps"), communication_payload.get("bandwidth_bps"))
        ),
    }
    if time_horizon_s is None:
        time_horizon_s = _positive_optional(
            _first(
                _mapping(trace_payload.get("time")).get("duration_s"),
                simulation_config_payload.get("duration_s"),
                workload_spec_payload.get("duration_s"),
                _mapping(manifest.get("time")).get("time_horizon_s"),
            )
        )
    workload_bundle = {
        "status": workload_payload.get("status", "loaded" if requests else "not_configured"),
        "configured": bool(requests or manifest or workload_payload.get("bundle_dir")),
        "bundle_dir": workload_payload.get("bundle_dir"),
        "schema_version": manifest.get("schema_version"),
        "manifest": manifest,
        "summary": summary,
        "request_count": len(requests),
        "session_count": len(sessions),
        "warnings": list(_warnings(workload_payload)),
    }
    scenario_warnings = tuple(
        dict.fromkeys(
            (
                *warnings,
                *_warnings(workload_payload),
                *_warnings(trace_payload),
                *_warnings(energy_payload),
            )
        )
    )
    return RuntimeSimulationSpec(
        strategy=RuntimeStrategySpec.from_value(strategy),
        scenario_id=scenario_id or str(workload_spec_payload.get("name") or "runtime_simulation"),
        design_id=design_id or str(constellation_payload.get("name") or "runtime_design"),
        time_horizon_s=time_horizon_s,
        time_step_s=_positive_optional(simulation_config_payload.get("step_s")),
        workload_bundle=workload_bundle,
        requests=requests,
        sessions=sessions,
        regions=regions,
        gateways=gateways,
        simulation_trace_ref=trace_payload,
        environment_state=environment_state,
        constellation={
            "design": constellation_payload,
            "architecture": architecture_payload,
            "satellite_count": len(environment_state.satellite_catalog) if environment_state else constellation_payload.get("satellite_count"),
        },
        serving_profile=serving_profile,
        energy_profile=energy_payload,
        communication_ingress=communication_ingress,
        provenance={
            "orbit_trace_id": trace_payload.get("trace_id"),
            "trace_model": trace_payload.get("trace_model"),
            "trace_scope": trace_payload.get("scope"),
            "valid_for_paper_main_results": bool(
                trace_payload.get("valid_for_paper_main_results", False)
            ),
            "workload_request_count": len(requests),
            "energy_profile": _mapping(energy_payload.get("provenance")),
        },
        warnings=scenario_warnings,
    )


def _trace_manifest(value: Any, bundle: OrbitTraceBundle | None) -> OrbitTraceManifest | None:
    if bundle is not None:
        return bundle.trace_manifest
    if isinstance(value, OrbitTraceManifest):
        return value
    payload = _mapping(value)
    nested = _mapping(payload.get("trace_manifest"))
    candidate = nested or payload
    if not candidate or not candidate.get("schema_version"):
        return None
    return OrbitTraceManifest.from_value(candidate)


def _communication_ingress_payload(*payloads: dict[str, Any]) -> dict[str, Any]:
    for payload in payloads:
        for container in (
            payload,
            _mapping(payload.get("orbit")),
            _mapping(payload.get("trace_options")),
            _mapping(payload.get("provenance")),
        ):
            for key in ("communication_ingress", "communication_satellite"):
                candidate = _mapping(container.get(key))
                if candidate:
                    return candidate
    return {}


def _mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(to_jsonable(value))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        result = value.to_dict()
        return dict(to_jsonable(result)) if isinstance(result, Mapping) else {}
    return {}


def _rows(value: Any) -> tuple[dict[str, Any], ...]:
    return tuple(dict(item) for item in (value or ()) if isinstance(item, Mapping))


def _warnings(payload: dict[str, Any]) -> tuple[str, ...]:
    value = payload.get("warnings", ())
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _positive_optional(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if parsed > 0.0 else None


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


__all__ = ["build_simulation_spec"]
