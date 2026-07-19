"""Satellite motion/access/link/shadow trace materialization for OrbitServe.

This module turns the static constellation architecture into deterministic
time-series traces consumed by the runtime evaluator.  This implementation is
a lightweight circular-geometry surrogate, not an STK-calibrated simulator.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..common import to_jsonable
from .constellation import (
    C_LIGHT_KM_PER_S,
    EARTH_MU_KM3_S2,
    EARTH_RADIUS_KM,
    EARTH_ROTATION_RAD_PER_S,
    ConstellationDesign,
    line_of_sight_ok,
)
from .simulation_traces import (
    CIRCULAR_GEOMETRY_TRACE,
    SATELLITE_MOTION_TRACE_SCHEMA,
    OrbitTraceManifest,
    trace_file_sha256,
)


MOTION_TRACE_DOC = "docs/research/parameters/satellite_motion_trace_model.md"


@dataclass(frozen=True)
class MotionTraceBuildResult:
    trace_dir: str
    trace_manifest: OrbitTraceManifest
    manifest: dict[str, Any]
    satellite_catalog: tuple[dict[str, Any], ...]
    access_trace: dict[str, Any]
    link_trace: dict[str, Any]
    energy_trace: dict[str, Any]
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_trace_manifest(self) -> dict[str, Any]:
        return to_jsonable(self.manifest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_dir": self.trace_dir,
            "trace_manifest": to_jsonable(self.manifest),
            "satellite_catalog": list(self.satellite_catalog),
            "access_trace": to_jsonable(self.access_trace),
            "link_trace": to_jsonable(self.link_trace),
            "energy_trace": to_jsonable(self.energy_trace),
            "warnings": list(self.warnings),
            "provenance": to_jsonable(self.provenance),
        }


@dataclass(frozen=True)
class _SatelliteState:
    time_s: float
    satellite_id: str
    r_eci_km: tuple[float, float, float]
    v_eci_km_s: tuple[float, float, float]
    r_ecef_km: tuple[float, float, float]
    lat_deg: float
    lon_deg: float
    alt_km: float
    orbital_phase_deg: float
    sunlit: bool


def build_satellite_motion_trace(
    *args: Any,
    config: Any = None,
    constellation: ConstellationDesign | Any = None,
    workload_generation: Any = None,
    base_dir: str | Path | None = None,
    experiment_id: str | None = None,
) -> MotionTraceBuildResult:
    """Build a circular-geometry motion/access/link/shadow trace bundle."""

    config, constellation, workload_generation = _resolve_builder_args(
        args,
        config=config,
        constellation=constellation,
        workload_generation=workload_generation,
    )
    options = _trace_options(config)
    warnings: list[str] = [
        "circular_geometry_trace is not STK-calibrated",
        "valid_for_paper_main_results=false",
    ]
    if options.get("propagation_model", "circular_two_body") != "circular_two_body":
        warnings.append("only propagation_model='circular_two_body' is implemented in v1")
    if options.get("eci_ecef_model", "simplified_gmst") != "simplified_gmst":
        warnings.append("only eci_ecef_model='simplified_gmst' is implemented in v1")

    configured_duration_s = _positive_float(getattr(config, "duration_s", None), 21600.0, "duration_s")
    step_s = _positive_float(getattr(config, "step_s", None), 60.0, "step_s")
    duration_meta, duration_warnings = _effective_trace_duration(
        configured_duration_s=configured_duration_s,
        step_s=step_s,
        options=options,
        workload_generation=workload_generation,
    )
    warnings.extend(duration_warnings)
    effective_duration_s = float(duration_meta["effective_duration_s"])
    epoch = str(getattr(config, "epoch", "2026-06-16T00:00:00Z"))
    time_grid = {
        **_build_time_grid(effective_duration_s, step_s, epoch),
        **duration_meta,
    }
    trace_id = _trace_id(experiment_id, epoch, effective_duration_s)
    trace_dir = _trace_dir(config, base_dir=base_dir, experiment_id=experiment_id)
    trace_dir.mkdir(parents=True, exist_ok=True)

    constellation_payload = _mapping(constellation)
    architecture = _architecture_analysis(constellation, constellation_payload)
    gateway_profile = _gateway_profile(constellation, constellation_payload)
    if _float(gateway_profile.get("bandwidth_bps"), 0.0) <= 0.0:
        warnings.append(
            "constellation.gateway_profile.bandwidth_bps is missing or non-positive; "
            "SGL capacity fails closed"
        )
    if _optional_int(gateway_profile.get("terminal_count_per_satellite")) in (None, 0):
        warnings.append(
            "constellation.gateway_profile.terminal_count_per_satellite is missing or non-positive; "
            "SGL terminal capacity fails closed"
        )
    catalog, catalog_warnings = _build_satellite_catalog(constellation)
    warnings.extend(catalog_warnings)
    if not catalog:
        raise ValueError("circular_geometry_trace requires constellation satellite_count > 0")

    ingress_catalog, ingress_warnings = _build_ingress_catalog(workload_generation, options)
    warnings.extend(ingress_warnings)
    if not ingress_catalog:
        warnings.append("no gateway or direct-user ingress points were found; access trace will be empty")

    gmst0_deg = _optional_float(options.get("gmst0_deg"))
    if gmst0_deg is None:
        gmst0_deg = 0.0
        warnings.append("simplified GMST uses gmst0_deg=0; STK calibration is required for paper main results")
    sun_vector = _sun_vector(options, warnings)

    ephemeris_by_time, ephemeris_rows = _build_ephemeris(
        catalog=catalog,
        times=time_grid["regular_times"],
        epoch=epoch,
        gmst0_deg=gmst0_deg,
        sun_vector=sun_vector,
        shadow_options=_dict(options.get("shadow_model")),
    )
    satellite_by_id = {row["satellite_id"]: row for row in catalog}
    access_trace, access_warnings = _build_access_trace(
        trace_id=trace_id,
        time_grid=time_grid,
        ingress_catalog=ingress_catalog,
        catalog=catalog,
        satellite_by_id=satellite_by_id,
        ephemeris_by_time=ephemeris_by_time,
        gateway_profile=gateway_profile,
        options=options,
    )
    warnings.extend(access_warnings)
    link_trace, link_warnings = _build_link_trace(
        trace_id=trace_id,
        time_grid=time_grid,
        ingress_catalog=ingress_catalog,
        catalog=catalog,
        ephemeris_by_time=ephemeris_by_time,
        access_trace=access_trace,
        constellation=constellation,
        architecture=architecture,
        gateway_profile=gateway_profile,
        options=options,
    )
    warnings.extend(link_warnings)
    energy_trace, energy_warnings = _build_energy_trace(
        trace_id=trace_id,
        time_grid=time_grid,
        catalog=catalog,
        ephemeris_by_time=ephemeris_by_time,
        options=options,
        sun_vector=sun_vector,
    )
    warnings.extend(energy_warnings)
    warnings = list(dict.fromkeys(warnings))

    provenance = {
        "trace_dir": str(trace_dir),
        "builder": "circular_geometry_trace",
        "source": "orbitserve.orbit_traces.build_satellite_motion_trace",
        "schema_version": SATELLITE_MOTION_TRACE_SCHEMA,
        "modeling_doc": MOTION_TRACE_DOC,
        "trace_options": to_jsonable(options),
        "configured_duration_s": configured_duration_s,
        "effective_duration_s": effective_duration_s,
        "trace_horizon_margin_s": duration_meta.get("trace_horizon_margin_s"),
        "workload_max_request_time_s": duration_meta.get("workload_max_request_time_s"),
        "workload_max_request_arrival_s": duration_meta.get("workload_max_request_arrival_s"),
        "workload_request_time_field_counts": duration_meta.get("workload_request_time_field_counts"),
        "constellation_name": constellation_payload.get("name"),
        "architecture_source": architecture.get("model"),
        "workload_provenance": _workload_provenance(workload_generation),
    }
    for payload in (access_trace, link_trace, energy_trace):
        payload["warnings"] = list(dict.fromkeys((*payload.get("warnings", ()), *warnings)))
        payload["provenance"] = {**_dict(payload.get("provenance")), **provenance}

    satellite_catalog_path = trace_dir / "satellite_catalog.json"
    ephemeris_path = trace_dir / "ephemeris_trace.jsonl"
    access_path = trace_dir / "access_trace.json"
    link_path = trace_dir / "link_trace.json"
    energy_path = trace_dir / "energy_trace.json"
    manifest_path = trace_dir / "trace_manifest.json"

    _write_json(satellite_catalog_path, list(catalog))
    _write_jsonl(ephemeris_path, ephemeris_rows)
    _write_json(access_path, access_trace)
    _write_json(link_path, link_trace)
    _write_json(energy_path, energy_trace)

    sha256 = {
        "satellite_catalog_sha256": trace_file_sha256(satellite_catalog_path),
        "ephemeris_trace_sha256": trace_file_sha256(ephemeris_path),
        "access_trace_sha256": trace_file_sha256(access_path),
        "link_trace_sha256": trace_file_sha256(link_path),
        "energy_trace_sha256": trace_file_sha256(energy_path),
        "calibration_report_sha256": None,
    }
    manifest = {
        "schema_version": SATELLITE_MOTION_TRACE_SCHEMA,
        "trace_model": CIRCULAR_GEOMETRY_TRACE,
        "trace_id": trace_id,
        "evaluated": True,
        "access_trace_path": access_path.name,
        "link_trace_path": link_path.name,
        "energy_trace_path": energy_path.name,
        "access_trace_sha256": sha256["access_trace_sha256"],
        "link_trace_sha256": sha256["link_trace_sha256"],
        "energy_trace_sha256": sha256["energy_trace_sha256"],
        "scope": "geometry_trace_surrogate",
        "valid_for_paper_main_results": False,
        "time": {key: value for key, value in time_grid.items() if key != "regular_times"},
        "files": {
            "satellite_catalog_path": satellite_catalog_path.name,
            "ephemeris_trace_path": ephemeris_path.name,
            "access_trace_path": access_path.name,
            "link_trace_path": link_path.name,
            "energy_trace_path": energy_path.name,
            "calibration_report_path": None,
        },
        "sha256": sha256,
        "calibration": {
            "stk_calibrated": False,
            "calibration_report_evaluated": False,
            "valid_for_paper_main_results": False,
        },
        "provenance": provenance,
        "warnings": warnings,
    }
    _write_json(manifest_path, manifest)
    manifest = {**manifest, "sha256": {**sha256, "trace_manifest_sha256": trace_file_sha256(manifest_path)}}
    _write_json(manifest_path, manifest)

    trace_manifest = OrbitTraceManifest.from_value(manifest)
    return MotionTraceBuildResult(
        trace_dir=str(trace_dir),
        trace_manifest=trace_manifest,
        manifest=manifest,
        satellite_catalog=catalog,
        access_trace=access_trace,
        link_trace=link_trace,
        energy_trace=energy_trace,
        warnings=tuple(warnings),
        provenance=provenance,
    )


def _trace_options(config: Any) -> dict[str, Any]:
    options = {
        "propagation_model": "circular_two_body",
        "eci_ecef_model": "simplified_gmst",
        "ground_ingress_source": "workload_bundle_gateways",
        "time_grid": {
            "boundary_tolerance_s": 1.0,
            "min_window_duration_s": 5.0,
            "merge_gap_s": 2.0,
        },
        "shadow_model": {
            "model": "cylindrical_eclipse",
            "sun_vector_model": "simple_epoch_fixed",
            "atmosphere_margin_km": 0.0,
        },
        "stk_calibration": {"enabled": False, "import_dir": None},
        "initial_soc": 0.8,
        "soc_min": 0.2,
        "battery_capacity_j": 10_000_000.0,
        "charge_rate_w": 200.0,
        "idle_power_w": 50.0,
        "compute_energy_j_per_token": 0.5,
        "link_energy_j_per_byte": 1e-6,
        "battery_charge_efficiency": 1.0,
        "battery_discharge_efficiency": 1.0,
        "dvfs_multiplier_sunlit": 1.0,
        "dvfs_multiplier_shadow": 1.5,
    }
    configured = getattr(config, "trace_options", None)
    if isinstance(configured, dict):
        options = _deep_merge(options, configured)
    return options


def _resolve_builder_args(
    args: tuple[Any, ...],
    *,
    config: Any,
    constellation: Any,
    workload_generation: Any,
) -> tuple[Any, Any, Any]:
    positional = list(args)
    if config is not None and positional:
        if len(positional) == 1 and constellation is not None and workload_generation is not None:
            positional = []
        else:
            raise TypeError("build_satellite_motion_trace received both positional and keyword config")
    if config is None and positional:
        config = positional.pop(0)
    if constellation is None and positional:
        constellation = positional.pop(0)
    if workload_generation is None and positional:
        workload_generation = positional.pop(0)
    if positional:
        raise TypeError("build_satellite_motion_trace received too many positional arguments")
    if config is None or constellation is None or workload_generation is None:
        raise TypeError("config, constellation, and workload_generation are required")
    return config, constellation, workload_generation


def _build_time_grid(duration_s: float, step_s: float, epoch: str) -> dict[str, Any]:
    regular_sample_count = int(math.floor(duration_s / step_s)) + 1
    regular_times = tuple(round(index * step_s, 9) for index in range(regular_sample_count))
    tail_duration_s = duration_s - regular_times[-1]
    return {
        "epoch": epoch,
        "duration_s": duration_s,
        "step_s": step_s,
        "sample_count": regular_sample_count,
        "regular_sample_count": regular_sample_count,
        "sample_count_definition": "floor(duration_s / step_s) + 1",
        "final_boundary_s": duration_s,
        "tail_duration_s": tail_duration_s,
        "include_final_boundary_sample": False,
        "regular_times": regular_times,
    }


def _effective_trace_duration(
    *,
    configured_duration_s: float,
    step_s: float,
    options: dict[str, Any],
    workload_generation: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    request_times, field_counts = _workload_request_times(workload_generation)
    max_request_time_s = max(request_times) if request_times else None
    margin_s, margin_warnings = _trace_horizon_margin_s(options, step_s)
    workload_required_s = None
    if max_request_time_s is not None:
        workload_required_s = max(0.0, max_request_time_s + margin_s)
    effective_duration_s = max(configured_duration_s, workload_required_s or configured_duration_s)
    warnings = list(margin_warnings)
    if effective_duration_s > configured_duration_s:
        warnings.append(
            "circular_geometry_trace horizon extended from "
            f"configured_duration_s={configured_duration_s:g} to effective_duration_s={effective_duration_s:g} "
            f"to cover workload max request time {max_request_time_s:g}s with margin_s={margin_s:g}"
        )
    return (
        {
            "duration_s": effective_duration_s,
            "configured_duration_s": configured_duration_s,
            "effective_duration_s": effective_duration_s,
            "duration_source": "workload_arrival_extended" if effective_duration_s > configured_duration_s else "configured",
            "workload_max_request_time_s": max_request_time_s,
            "workload_max_request_arrival_s": max_request_time_s,
            "trace_horizon_margin_s": margin_s,
            "workload_request_time_field_counts": field_counts,
        },
        tuple(dict.fromkeys(warnings)),
    )


def _workload_request_times(workload_generation: Any) -> tuple[tuple[float, ...], dict[str, int]]:
    payload = _mapping(workload_generation)
    times: list[float] = []
    field_counts = {"arrival_time_s": 0, "time_s": 0, "missing": 0, "invalid": 0}
    for request in payload.get("requests", ()) or ():
        if not isinstance(request, dict):
            continue
        raw_value = request.get("arrival_time_s")
        field_name = "arrival_time_s"
        if raw_value is None:
            raw_value = request.get("time_s")
            field_name = "time_s"
        if raw_value is None:
            field_counts["missing"] += 1
            continue
        parsed = _optional_float(raw_value)
        if parsed is None:
            field_counts["invalid"] += 1
            continue
        if parsed < 0.0:
            field_counts["invalid"] += 1
            continue
        field_counts[field_name] += 1
        times.append(parsed)
    return tuple(times), field_counts


def _trace_horizon_margin_s(options: dict[str, Any], step_s: float) -> tuple[float, tuple[str, ...]]:
    time_options = _dict(options.get("time_grid"))
    default_margin_s = max(0.0, step_s)
    raw_margin = _first_present(
        options.get("trace_horizon_margin_s"),
        options.get("workload_trace_horizon_margin_s"),
        time_options.get("trace_horizon_margin_s"),
        time_options.get("workload_trace_horizon_margin_s"),
    )
    if raw_margin is None:
        return default_margin_s, ()
    parsed = _optional_float(raw_margin)
    if parsed is None:
        return default_margin_s, (
            f"invalid trace_horizon_margin_s={raw_margin!r}; using default margin_s={default_margin_s:g}",
        )
    if parsed <= 0.0:
        return default_margin_s, (
            f"non-positive trace_horizon_margin_s={parsed:g}; using default margin_s={default_margin_s:g}",
        )
    return parsed, ()


def _trace_id(experiment_id: str | None, epoch: str, duration_s: float) -> str:
    return f"{CIRCULAR_GEOMETRY_TRACE}:{experiment_id or 'runtime_scenario'}:{epoch}:{duration_s:g}s"


def _trace_dir(config: Any, *, base_dir: str | Path | None, experiment_id: str | None) -> Path:
    configured = getattr(config, "trace_output_dir", None)
    if configured:
        path = Path(str(configured)).expanduser()
        if path.is_absolute():
            return path
        root = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()
        return (root / path).resolve()
    root = Path(base_dir).expanduser().resolve() if base_dir is not None else Path.cwd()
    return (root / "results" / "simulation_traces" / (experiment_id or "runtime_scenario")).resolve()


def _build_satellite_catalog(constellation: Any) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    warnings: list[str] = []
    satellites: list[dict[str, Any]] = []
    ordinal = 0
    effective_sets = tuple(getattr(constellation, "effective_orbital_sets", ()) or ())
    for set_order, orbital_set in enumerate(effective_sets):
        resolved_planes = _resolved_planes(orbital_set)
        if getattr(orbital_set, "custom_planes", ()):
            warnings.append(
                f"orbital_set {getattr(orbital_set, 'name', set_order)} uses circular per-plane custom overrides; "
                "per-cell arbitrary anomaly is not implemented"
            )
        for plane in resolved_planes:
            plane_index = int(getattr(plane, "plane_index"))
            cell_indices = tuple(getattr(orbital_set, "effective_cell_indices")(plane_index))
            cluster = orbital_set.cluster
            offsets = tuple(getattr(cluster, "offsets_lvlh_m", ((0.0, 0.0, 0.0),)))
            for cell_index in cell_indices:
                u0_deg = float(getattr(orbital_set, "cell_argument_latitude_deg")(plane_index, cell_index))
                for member_index, offset in enumerate(offsets):
                    semantic_id = (
                        f"sat:{getattr(orbital_set, 'name', 'set')}:{plane_index}:{cell_index}:{member_index}"
                    )
                    compute_cell_id = f"cell:{getattr(orbital_set, 'name', 'set')}:{plane_index}:{cell_index}"
                    altitude_km = float(getattr(plane, "altitude_km"))
                    a_km = EARTH_RADIUS_KM + altitude_km
                    n_rad_s = math.sqrt(EARTH_MU_KM3_S2 / (a_km**3))
                    satellites.append(
                        {
                            "satellite_id": f"sat-{ordinal:05d}",
                            "semantic_satellite_id": semantic_id,
                            "ordinal": ordinal,
                            "deployment_id": "deployment_0",
                            "orbital_set_order": set_order,
                            "orbital_set_name": getattr(orbital_set, "name", None),
                            "plane_index": plane_index,
                            "cell_index": int(cell_index),
                            "cluster_member_index": member_index,
                            "compute_cell_id": compute_cell_id,
                            "is_compute_cell_centroid": member_index == 0,
                            "cluster_offset_lvlh_m": [float(value) for value in offset],
                            "orbit": {
                                "altitude_km": altitude_km,
                                "semimajor_axis_km": a_km,
                                "inclination_deg": float(getattr(plane, "inclination_deg")),
                                "raan_deg": float(getattr(plane, "raan_deg")),
                                "u0_deg": u0_deg,
                                "mean_motion_rad_s": n_rad_s,
                                "orbital_period_s": 2.0 * math.pi / n_rad_s,
                                "min_elevation_deg": float(getattr(plane, "min_elevation_deg", 25.0)),
                                "orbit_type": getattr(plane, "orbit_type", None),
                                "orbital_set_type": getattr(plane, "orbital_set_type", None),
                                "walker_type": getattr(plane, "walker_type", None),
                                "ltan": getattr(plane, "ltan", None),
                                "raan_pattern_or_ltan": getattr(plane, "raan_pattern_or_ltan", None),
                                "epoch": getattr(plane, "epoch", None),
                                "custom_override": bool(getattr(plane, "custom_override", False)),
                            },
                            "link_terminal_profile": {},
                            "provenance": {
                                "source": "ConstellationDesign.effective_orbital_sets",
                                "lvlh_offset_convention": "x=along-track,y=cross-track,z=radial",
                            },
                        }
                    )
                    ordinal += 1
    return tuple(satellites), tuple(dict.fromkeys(warnings))


def _resolved_planes(orbital_set: Any) -> tuple[Any, ...]:
    helper = getattr(orbital_set, "iter_resolved_planes", None)
    if callable(helper):
        return tuple(helper())
    return tuple()


def _build_ephemeris(
    *,
    catalog: tuple[dict[str, Any], ...],
    times: tuple[float, ...],
    epoch: str,
    gmst0_deg: float,
    sun_vector: tuple[float, float, float],
    shadow_options: dict[str, Any],
) -> tuple[dict[float, dict[str, _SatelliteState]], tuple[dict[str, Any], ...]]:
    by_time: dict[float, dict[str, _SatelliteState]] = {}
    rows: list[dict[str, Any]] = []
    atmosphere_margin_km = _float(shadow_options.get("atmosphere_margin_km"), 0.0)
    for time_s in times:
        by_time[time_s] = {}
        theta_rad = math.radians(gmst0_deg) + EARTH_ROTATION_RAD_PER_S * time_s
        for sat in catalog:
            orbit = _dict(sat.get("orbit"))
            centroid_r, centroid_v, phase_deg = _propagate_circular(
                altitude_km=float(orbit["altitude_km"]),
                inclination_deg=float(orbit["inclination_deg"]),
                raan_deg=float(orbit["raan_deg"]),
                u0_deg=float(orbit["u0_deg"]),
                time_s=time_s,
            )
            r_eci = _apply_lvlh_offset(
                centroid_r,
                centroid_v,
                tuple(float(value) for value in sat.get("cluster_offset_lvlh_m", (0.0, 0.0, 0.0))),
            )
            r_ecef = _rz(r_eci, -theta_rad)
            lat_deg, lon_deg, alt_km = _ecef_to_lla_spherical(r_ecef)
            sunlit = _sunlit_cylindrical(r_eci, sun_vector, atmosphere_margin_km=atmosphere_margin_km)
            state = _SatelliteState(
                time_s=time_s,
                satellite_id=str(sat["satellite_id"]),
                r_eci_km=r_eci,
                v_eci_km_s=centroid_v,
                r_ecef_km=r_ecef,
                lat_deg=lat_deg,
                lon_deg=lon_deg,
                alt_km=alt_km,
                orbital_phase_deg=phase_deg,
                sunlit=sunlit,
            )
            by_time[time_s][state.satellite_id] = state
            rows.append(
                {
                    "time_s": time_s,
                    "time_iso": _time_iso_from_epoch(epoch, time_s),
                    "satellite_id": state.satellite_id,
                    "r_eci_km": list(state.r_eci_km),
                    "v_eci_km_s": list(state.v_eci_km_s),
                    "r_ecef_km": list(state.r_ecef_km),
                    "lat_deg": state.lat_deg,
                    "lon_deg": state.lon_deg,
                    "alt_km": state.alt_km,
                    "orbital_phase_deg": state.orbital_phase_deg,
                    "sunlit": state.sunlit,
                    "provenance": {"model": "circular_two_body+simplified_gmst"},
                }
            )
    return by_time, tuple(rows)


def _build_ingress_catalog(workload_generation: Any, options: dict[str, Any]) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    payload = _mapping(workload_generation)
    gateways = tuple(_dict(item) for item in payload.get("gateways", ()) if isinstance(item, dict))
    regions = tuple(_dict(item) for item in payload.get("regions", ()) if isinstance(item, dict))
    requests = tuple(_dict(item) for item in payload.get("requests", ()) if isinstance(item, dict))
    source = str(options.get("ground_ingress_source", "workload_bundle_gateways"))
    include_gateways = source in {"workload_bundle_gateways", "workload_bundle_gateways_and_direct_requests", "all"}
    include_direct = source in {"direct_requests", "workload_bundle_gateways_and_direct_requests", "all"}
    include_regions = source in {"regions", "region_samples", "all"}
    rows: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    if include_gateways:
        for gateway in gateways:
            gateway_id = gateway.get("gateway_id")
            if gateway_id is None or gateway.get("lat_deg") is None or gateway.get("lon_deg") is None:
                warnings.append("gateway ingress missing gateway_id/lat_deg/lon_deg and was skipped")
                continue
            rows[str(gateway_id)] = {
                "ingress_id": str(gateway_id),
                "ingress_type": "gateway",
                "lat_deg": float(gateway["lat_deg"]),
                "lon_deg": float(gateway["lon_deg"]),
                "served_region_ids": list(gateway.get("served_region_ids", ()) or ()),
                "bandwidth_bps": _optional_float(gateway.get("gateway_bandwidth_bps")),
                "provenance": {"source": "workload_bundle.gateways"},
            }
    if include_direct:
        for request in requests:
            direct = str(request.get("access_mode") or "") == "direct_user_sgl"
            lat = _first_present(request.get("ingress_lat_deg"), request.get("lat_deg"))
            lon = _first_present(request.get("ingress_lon_deg"), request.get("lon_deg"))
            if not direct and (lat is None or lon is None):
                continue
            if lat is None or lon is None:
                warnings.append(f"direct request {request.get('request_id')} missing ingress lat/lon")
                continue
            ingress_id = _direct_ingress_id(request, float(lat), float(lon))
            rows.setdefault(
                ingress_id,
                {
                    "ingress_id": ingress_id,
                    "ingress_type": "direct_user",
                    "lat_deg": float(lat),
                    "lon_deg": float(lon),
                    "served_region_ids": [str(request.get("region_id"))] if request.get("region_id") else [],
                    "bandwidth_bps": None,
                    "provenance": {"source": "workload_bundle.requests", "access_mode": "direct_user_sgl"},
                },
            )
    if include_regions:
        for region in regions:
            if region.get("region_id") is None or region.get("lat_deg") is None or region.get("lon_deg") is None:
                continue
            ingress_id = f"region:{region['region_id']}"
            rows.setdefault(
                ingress_id,
                {
                    "ingress_id": ingress_id,
                    "ingress_type": "region_sample",
                    "lat_deg": float(region["lat_deg"]),
                    "lon_deg": float(region["lon_deg"]),
                    "served_region_ids": [str(region["region_id"])],
                    "bandwidth_bps": None,
                    "provenance": {"source": "workload_bundle.regions"},
                },
            )
    ordered = tuple(rows[key] for key in sorted(rows))
    return ordered, tuple(dict.fromkeys(warnings))


def _build_access_trace(
    *,
    trace_id: str,
    time_grid: dict[str, Any],
    ingress_catalog: tuple[dict[str, Any], ...],
    catalog: tuple[dict[str, Any], ...],
    satellite_by_id: dict[str, dict[str, Any]],
    ephemeris_by_time: dict[float, dict[str, _SatelliteState]],
    gateway_profile: dict[str, Any],
    options: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    warnings: list[str] = []
    profile = gateway_profile
    if _float(profile.get("bandwidth_bps"), 0.0) <= 0.0:
        warnings.append("missing gateway-profile SGL bandwidth; access links fail closed for runtime capacity")
    if (_optional_int(profile.get("terminal_count_per_satellite")) or 0) <= 0:
        warnings.append("missing gateway-profile SGL terminal count; access links fail closed for runtime capacity")
    time_opts = _dict(options.get("time_grid"))
    min_window_duration_s = _float(time_opts.get("min_window_duration_s"), 5.0)
    merge_gap_s = _float(time_opts.get("merge_gap_s"), 2.0)
    times = tuple(time_grid["regular_times"])
    windows: list[dict[str, Any]] = []
    visible_count_samples: list[dict[str, Any]] = []
    handover_events: list[dict[str, Any]] = []

    for ingress in ingress_catalog:
        ground_ecef = _ground_ecef(float(ingress["lat_deg"]), float(ingress["lon_deg"]))
        previous_preferred: str | None = None
        pair_windows: list[dict[str, Any]] = []
        per_time_visible: dict[float, list[tuple[str, float]]] = {}
        for sat in catalog:
            sat_id = str(sat["satellite_id"])
            orbit = _dict(satellite_by_id[sat_id].get("orbit"))
            min_elevation_deg = _float(
                _first_present(
                    ingress.get("min_elevation_deg"),
                    profile.get("min_elevation_deg") if profile else None,
                    orbit.get("min_elevation_deg"),
                ),
                25.0,
            )
            samples: list[dict[str, Any]] = []
            max_range_km = _optional_float(profile.get("max_range_km") if profile else None)
            for time_s in times:
                state = ephemeris_by_time[time_s][sat_id]
                elevation_deg, range_km = _elevation_range(ground_ecef, state.r_ecef_km)
                visible = elevation_deg >= min_elevation_deg and (
                    max_range_km is None or range_km <= max_range_km
                )
                if visible:
                    per_time_visible.setdefault(time_s, []).append((sat_id, elevation_deg))
                samples.append(
                    {
                        "time_s": time_s,
                        "visible": visible,
                        "score": elevation_deg - min_elevation_deg,
                        "elevation_deg": elevation_deg,
                        "range_km": range_km,
                        "propagation_delay_s": range_km / C_LIGHT_KM_PER_S,
                    }
                )
            pair_windows.extend(
                _access_windows_from_samples(
                    trace_id=trace_id,
                    ingress=ingress,
                    satellite_id=sat_id,
                    samples=samples,
                    duration_s=float(time_grid["duration_s"]),
                    min_window_duration_s=min_window_duration_s,
                    merge_gap_s=merge_gap_s,
                    min_elevation_deg=min_elevation_deg,
                    bandwidth_bps=_float(profile.get("bandwidth_bps") if profile else None, 0.0),
                    terminal_count_per_satellite=_nonnegative_int(
                        profile.get("terminal_count_per_satellite") if profile else None
                    ),
                    setup_delay_s=_float(profile.get("setup_delay_s") if profile else None, 0.0),
                    epoch=str(time_grid.get("epoch")) if time_grid.get("epoch") is not None else None,
                )
            )
        for time_s in times:
            visible = sorted(per_time_visible.get(time_s, ()), key=lambda item: (-item[1], item[0]))
            preferred = visible[0][0] if visible else None
            visible_count_samples.append(
                {
                    "ingress_id": ingress["ingress_id"],
                    "time_s": time_s,
                    "geometry_visible_count": len(visible),
                    "externally_available_count": len(visible),
                    "preferred_satellite_id": preferred,
                }
            )
            if preferred != previous_preferred:
                handover_events.append(
                    {
                        "event_id": f"handover-{len(handover_events):06d}",
                        "ingress_id": ingress["ingress_id"],
                        "time_s": time_s,
                        "from_satellite_id": previous_preferred,
                        "to_satellite_id": preferred,
                        "reason": "preferred_satellite_changed",
                    }
                )
                previous_preferred = preferred
        windows.extend(pair_windows)
    windows.sort(key=lambda item: (item["start_s"], item["ingress_id"], item["satellite_id"]))
    for index, window in enumerate(windows):
        window["window_id"] = f"access-{index:06d}"
    return (
        {
            "schema_version": SATELLITE_MOTION_TRACE_SCHEMA,
            "trace_id": trace_id,
            "ingress_catalog": list(ingress_catalog),
            "satellite_catalog": list(catalog),
            "access_windows": windows,
            "visible_count_samples": visible_count_samples,
            "handover_events": handover_events,
            "warnings": warnings,
            "provenance": {"model": "circular_geometry_sgl_elevation"},
        },
        tuple(dict.fromkeys(warnings)),
    )


def _build_link_trace(
    *,
    trace_id: str,
    time_grid: dict[str, Any],
    ingress_catalog: tuple[dict[str, Any], ...],
    catalog: tuple[dict[str, Any], ...],
    ephemeris_by_time: dict[float, dict[str, _SatelliteState]],
    access_trace: dict[str, Any],
    constellation: Any,
    architecture: dict[str, Any],
    gateway_profile: dict[str, Any],
    options: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    warnings: list[str] = [
        "optical FOR/pointing/acquisition/sun-exclusion are placeholders, not high-fidelity terminal control"
    ]
    physical: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    times = tuple(time_grid["regular_times"])
    sgl_profile = gateway_profile
    sgl_profile_id = str(
        sgl_profile.get("profile_id")
        or sgl_profile.get("profile")
        or "gateway_profile"
    )
    for ingress in ingress_catalog:
        for sat in catalog:
            link_id = f"sgl:{ingress['ingress_id']}:{sat['satellite_id']}"
            physical[link_id] = _physical_link(
                link_id=link_id,
                kind="sgl",
                endpoint_a=str(ingress["ingress_id"]),
                endpoint_b=str(sat["satellite_id"]),
                relation="ground_to_space",
                profile_id=sgl_profile_id,
                profile=sgl_profile,
                policy="sgl_visibility",
                priority=10,
                terminal_pool_id=str(sgl_profile.get("terminal_pool") or "gateway_sgl"),
                terminal_pool_kind="sgl",
                terminal_count_per_satellite=_optional_int(
                    sgl_profile.get("terminal_count_per_satellite")
                ),
                setup_delay_s=_optional_float(sgl_profile.get("setup_delay_s")),
            )
    access_by_link_time = _access_membership(access_trace)
    for link in list(physical.values()):
        if link["kind"] != "sgl":
            continue
        ingress = next((row for row in ingress_catalog if row["ingress_id"] == link["endpoint_a"]), None)
        if ingress is None:
            continue
        ground_ecef = _ground_ecef(float(ingress["lat_deg"]), float(ingress["lon_deg"]))
        for time_s in times:
            state = ephemeris_by_time[time_s][str(link["endpoint_b"])]
            _, range_km = _elevation_range(ground_ecef, state.r_ecef_km)
            visible = access_by_link_time.get((str(link["link_id"]), time_s), False)
            available = bool(link.get("profile_resolved", False)) and visible
            if available:
                reason = None
            elif not link.get("profile_resolved", False):
                reason = "missing_profile"
            else:
                reason = "not_visible"
            samples.append(_link_sample(link, time_s, range_km, True, available, reason))

    isl_links, isl_warnings = _visibility_isl_candidates(
        catalog=catalog,
        options=options,
    )
    warnings.extend(isl_warnings)
    physical.update({link["link_id"]: link for link in isl_links})
    for link in isl_links:
        for time_s in times:
            first = ephemeris_by_time[time_s][str(link["endpoint_a"])]
            second = ephemeris_by_time[time_s][str(link["endpoint_b"])]
            distance_km = _norm(_sub(second.r_eci_km, first.r_eci_km))
            los_ok = line_of_sight_ok(first.r_eci_km, second.r_eci_km)
            max_range_km = _optional_float(link.get("max_range_km"))
            distance_ok = max_range_km is None or distance_km <= max_range_km
            available = bool(link.get("profile_resolved", False)) and los_ok and distance_ok
            reason = None
            if not link.get("profile_resolved", False):
                reason = "missing_profile"
            elif not los_ok:
                reason = "earth_blocked"
            elif not distance_ok:
                reason = "range"
            samples.append(_link_sample(link, time_s, distance_km, los_ok, available, reason))

    active_snapshots = _active_graph_snapshots(times, physical, samples)
    runtime_links = _runtime_compat_links(physical, samples)
    ordered_physical = tuple(physical[key] for key in sorted(physical))
    return (
        {
            "schema_version": SATELLITE_MOTION_TRACE_SCHEMA,
            "trace_id": trace_id,
            "physical_links": list(ordered_physical),
            "links": runtime_links,
            "link_samples": samples,
            "active_graph_snapshots": active_snapshots,
            "terminal_catalog": _terminal_catalog(catalog, physical),
            "warnings": warnings,
            "provenance": {"model": "circular_geometry_physical_graph"},
        },
        tuple(dict.fromkeys(warnings)),
    )


def _build_energy_trace(
    *,
    trace_id: str,
    time_grid: dict[str, Any],
    catalog: tuple[dict[str, Any], ...],
    ephemeris_by_time: dict[float, dict[str, _SatelliteState]],
    options: dict[str, Any],
    sun_vector: tuple[float, float, float],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    warnings = [
        "energy_trace contains exogenous sun/shadow inputs; scheduler-dependent SOC is evaluated downstream",
        "sun vector uses simplified placeholder; shadow intervals require STK calibration for paper main results",
    ]
    if _runtime_energy_envelope_unavailable(options):
        warnings.append("runtime energy envelope unavailable; runtime SOC fields omitted to fail closed")
    times = tuple(time_grid["regular_times"])
    duration_s = float(time_grid["duration_s"])
    shadow_windows: list[dict[str, Any]] = []
    energy_windows: list[dict[str, Any]] = []
    solar_samples: list[dict[str, Any]] = []
    initial_soc = {str(sat["satellite_id"]): _float(options.get("initial_soc"), 0.8) for sat in catalog}
    sunlight_contract_summary = _geometry_orbit_average_sunlight(
        catalog=catalog,
        sun_vector=sun_vector,
        shadow_options=_dict(options.get("shadow_model")),
    )
    for sat in catalog:
        sat_id = str(sat["satellite_id"])
        states = [ephemeris_by_time[time_s][sat_id] for time_s in times]
        state_windows = _boolean_windows(
            [{"time_s": state.time_s, "flag": state.sunlit} for state in states],
            duration_s=duration_s,
        )
        for start_s, end_s, sunlit in state_windows:
            row = {
                "satellite_id": sat_id,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": max(0.0, end_s - start_s),
                "sunlit": bool(sunlit),
                "eclipsed": not bool(sunlit),
                "beta_angle_deg": None,
                "sun_vector_eci": list(sun_vector),
                "model": "cylindrical_eclipse",
            }
            shadow_windows.append(row)
            energy_window = {
                **row,
                "runtime_energy_envelope": _dict(options.get("runtime_energy_envelope")),
            }
            if _runtime_energy_envelope_unavailable(options):
                energy_window["energy_unavailable_reason"] = "runtime_energy_envelope_unavailable"
            else:
                energy_window.update(
                    {
                        "dvfs_multiplier": _float(
                            options.get("dvfs_multiplier_sunlit" if sunlit else "dvfs_multiplier_shadow"),
                            1.0 if sunlit else 1.5,
                        ),
                        "charge_rate_w": _float(options.get("charge_rate_w"), 200.0) if sunlit else 0.0,
                        "idle_power_w": _energy_idle_power_w(options, bool(sunlit)),
                        "compute_energy_j_per_token": _float(options.get("compute_energy_j_per_token"), 0.5),
                        "link_energy_j_per_byte": _float(options.get("link_energy_j_per_byte"), 1e-6),
                        "battery_capacity_j": _float(options.get("battery_capacity_j"), 10_000_000.0),
                        "soc_min": _float(options.get("soc_min"), 0.2),
                        "battery_charge_efficiency": _float(options.get("battery_charge_efficiency"), 1.0),
                        "battery_discharge_efficiency": _float(options.get("battery_discharge_efficiency"), 1.0),
                        "energy_accounting_mode": str(options.get("energy_accounting_mode", "event_incremental_v1")),
                    }
                )
            energy_windows.append(energy_window)
        for state in states:
            solar_samples.append(
                {
                    "satellite_id": sat_id,
                    "time_s": state.time_s,
                    "sunlit": state.sunlit,
                    "P_solar_potential_w": _float(options.get("charge_rate_w"), 200.0) if state.sunlit else 0.0,
                    "sun_vector_eci": list(sun_vector),
                }
            )
    shadow_windows.sort(key=lambda item: (item["satellite_id"], item["start_s"], item["end_s"]))
    energy_windows.sort(key=lambda item: (item["satellite_id"], item["start_s"], item["end_s"]))
    soc_inputs = {
        "runtime_energy_envelope": _dict(options.get("runtime_energy_envelope")),
    }
    if not _runtime_energy_envelope_unavailable(options):
        soc_inputs.update(
            {
                "battery_capacity_j": _float(options.get("battery_capacity_j"), 10_000_000.0),
                "soc_min": _float(options.get("soc_min"), 0.2),
                "charge_rate_w": _float(options.get("charge_rate_w"), 200.0),
                "idle_power_w_sunlit": _energy_idle_power_w(options, True),
                "idle_power_w_shadow": _energy_idle_power_w(options, False),
                "compute_energy_j_per_token": _float(
                    options.get("compute_energy_j_per_token"), 0.5
                ),
                "link_energy_j_per_byte": _float(
                    options.get("link_energy_j_per_byte"), 1e-6
                ),
                "battery_charge_efficiency": _float(options.get("battery_charge_efficiency"), 1.0),
                "battery_discharge_efficiency": _float(options.get("battery_discharge_efficiency"), 1.0),
                "energy_accounting_mode": str(options.get("energy_accounting_mode", "event_incremental_v1")),
            }
        )
    return (
        {
            "schema_version": SATELLITE_MOTION_TRACE_SCHEMA,
            "trace_id": trace_id,
            "satellite_initial_soc": initial_soc,
            "shadow_windows": shadow_windows,
            "energy_windows": energy_windows,
            "solar_samples": solar_samples,
            "sunlight_contract_summary": sunlight_contract_summary,
            "soc_inputs": soc_inputs,
            "warnings": warnings,
            "provenance": {
                "model": "cylindrical_eclipse",
                "runtime_energy_envelope": _dict(options.get("runtime_energy_envelope")),
            },
        },
        tuple(warnings),
    )


def _geometry_orbit_average_sunlight(
    *,
    catalog: tuple[dict[str, Any], ...],
    sun_vector: tuple[float, float, float],
    shadow_options: dict[str, Any],
) -> dict[str, Any]:
    """Sample one complete orbit per satellite for a horizon-independent contract."""

    sample_count_per_satellite = 360
    atmosphere_margin_km = _float(shadow_options.get("atmosphere_margin_km"), 0.0)
    sunlit_samples = 0
    total_samples = 0
    per_satellite: dict[str, float] = {}
    per_satellite_maximum_eclipse_s: dict[str, float] = {}
    maximum_eclipse_s = 0.0
    for sat in catalog:
        orbit = _dict(sat.get("orbit"))
        period_s = _float(orbit.get("orbital_period_s"), 0.0)
        if period_s <= 0.0:
            continue
        satellite_sunlit = 0
        sunlight_flags: list[bool] = []
        for sample_index in range(sample_count_per_satellite):
            time_s = period_s * sample_index / sample_count_per_satellite
            centroid_r, centroid_v, _ = _propagate_circular(
                altitude_km=float(orbit["altitude_km"]),
                inclination_deg=float(orbit["inclination_deg"]),
                raan_deg=float(orbit["raan_deg"]),
                u0_deg=float(orbit["u0_deg"]),
                time_s=time_s,
            )
            r_eci = _apply_lvlh_offset(
                centroid_r,
                centroid_v,
                tuple(
                    float(value)
                    for value in sat.get("cluster_offset_lvlh_m", (0.0, 0.0, 0.0))
                ),
            )
            sunlit = _sunlit_cylindrical(
                r_eci,
                sun_vector,
                atmosphere_margin_km=atmosphere_margin_km,
            )
            sunlight_flags.append(sunlit)
            if sunlit:
                satellite_sunlit += 1
        sat_id = str(sat.get("satellite_id"))
        per_satellite[sat_id] = satellite_sunlit / sample_count_per_satellite
        longest_eclipse_samples = _longest_circular_false_run(sunlight_flags)
        satellite_eclipse_s = (
            longest_eclipse_samples * period_s / sample_count_per_satellite
        )
        per_satellite_maximum_eclipse_s[sat_id] = satellite_eclipse_s
        maximum_eclipse_s = max(maximum_eclipse_s, satellite_eclipse_s)
        sunlit_samples += satellite_sunlit
        total_samples += sample_count_per_satellite
    return {
        "evaluated": total_samples > 0,
        "actual_sunlit_fraction": (
            sunlit_samples / total_samples if total_samples > 0 else None
        ),
        "sample_count_per_satellite": sample_count_per_satellite,
        "satellite_count": len(per_satellite),
        "model": "cylindrical_eclipse_full_orbit_sampling",
        "per_satellite_sunlit_fraction": per_satellite,
        "maximum_shadow_window_s": maximum_eclipse_s,
        "per_satellite_maximum_shadow_window_s": per_satellite_maximum_eclipse_s,
    }


def _longest_circular_false_run(flags: list[bool]) -> int:
    if not flags:
        return 0
    if not any(flags):
        return len(flags)
    longest = 0
    current = 0
    limit = len(flags)
    for flag in (*flags, *flags):
        if flag:
            current = 0
            continue
        current = min(limit, current + 1)
        longest = max(longest, current)
    return longest


def _visibility_isl_candidates(
    *,
    catalog: tuple[dict[str, Any], ...],
    options: dict[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    """Return all satellite pairs as candidates; geometry decides visibility."""

    hardware = _dict(options.get("satellite_link_hardware"))
    terminal_count = _optional_int(hardware.get("terminal_count_per_satellite")) or 0
    terminal_bandwidth_gbps = _float(hardware.get("terminal_bandwidth_gbps"), 0.0)
    profile = {
        "bandwidth_bps": terminal_bandwidth_gbps * 1_000_000_000.0,
        "terminal_count_per_satellite": terminal_count,
        "terminal_pool": "satellite_isl_terminals",
    }
    warnings: list[str] = []
    if terminal_count <= 0 or terminal_bandwidth_gbps <= 0.0:
        warnings.append(
            "single-satellite ISL terminal hardware is missing or non-positive; visibility candidates fail closed"
        )
    links: list[dict[str, Any]] = []
    ordered = sorted(catalog, key=lambda item: str(item["satellite_id"]))
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            endpoint_a = str(first["satellite_id"])
            endpoint_b = str(second["satellite_id"])
            link_id = f"isl:{endpoint_a}:{endpoint_b}"
            links.append(
                _physical_link(
                    link_id=link_id,
                    kind="isl",
                    endpoint_a=endpoint_a,
                    endpoint_b=endpoint_b,
                    relation="geometric_visibility_candidate",
                    profile_id="single_satellite_isl_terminal",
                    profile=profile,
                    policy="runtime_visibility_selection",
                    priority=20,
                    terminal_pool_id="satellite_isl_terminals",
                    terminal_pool_kind="isl",
                    terminal_count_per_satellite=terminal_count,
                    setup_delay_s=0.0,
                )
            )
    return tuple(links), tuple(warnings)


def _access_windows_from_samples(
    *,
    trace_id: str,
    ingress: dict[str, Any],
    satellite_id: str,
    samples: list[dict[str, Any]],
    duration_s: float,
    min_window_duration_s: float,
    merge_gap_s: float,
    min_elevation_deg: float,
    bandwidth_bps: float,
    terminal_count_per_satellite: int,
    setup_delay_s: float,
    epoch: str | None,
) -> list[dict[str, Any]]:
    windows: list[tuple[float, float, list[dict[str, Any]]]] = []
    active_start: float | None = None
    active_samples: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        visible = bool(sample["visible"])
        if visible and active_start is None:
            if index == 0:
                active_start = 0.0
            else:
                active_start = _linear_boundary(samples[index - 1], sample)
            active_samples = []
        if visible:
            active_samples.append(sample)
        if not visible and active_start is not None:
            end_s = _linear_boundary(samples[index - 1], sample)
            windows.append((active_start, end_s, list(active_samples)))
            active_start = None
            active_samples = []
    if active_start is not None:
        windows.append((active_start, duration_s, list(active_samples)))
    merged = _merge_windows(windows, merge_gap_s=merge_gap_s)
    rows: list[dict[str, Any]] = []
    for start_s, end_s, items in merged:
        if end_s - start_s < min_window_duration_s:
            continue
        max_item = max(items, key=lambda item: (item["elevation_deg"], -item["range_km"])) if items else None
        min_range = min((item["range_km"] for item in items), default=None)
        max_range = max((item["range_km"] for item in items), default=None)
        min_delay = (min_range / C_LIGHT_KM_PER_S) if min_range is not None else None
        max_delay = (max_range / C_LIGHT_KM_PER_S) if max_range is not None else None
        rows.append(
            {
                "window_id": "",
                "ingress_id": str(ingress["ingress_id"]),
                "ingress_type": str(ingress["ingress_type"]),
                "satellite_id": satellite_id,
                "sgl_link_id": f"sgl:{ingress['ingress_id']}:{satellite_id}",
                "start_s": max(0.0, start_s),
                "end_s": min(duration_s, end_s),
                "duration_s": max(0.0, min(duration_s, end_s) - max(0.0, start_s)),
                "start_time_iso": _time_iso_from_epoch(epoch, max(0.0, start_s)),
                "end_time_iso": _time_iso_from_epoch(epoch, min(duration_s, end_s)),
                "min_elevation_deg": min_elevation_deg,
                "max_elevation_deg": None if max_item is None else max_item["elevation_deg"],
                "tca_s": None if max_item is None else max_item["time_s"],
                "range_min_km": min_range,
                "range_max_km": max_range,
                "propagation_delay_min_s": min_delay,
                "propagation_delay_max_s": max_delay,
                "propagation_delay_s": min_delay or 0.0,
                "bandwidth_bps": bandwidth_bps,
                "terminal_count_per_satellite": terminal_count_per_satellite,
                "tau_setup_s": setup_delay_s,
                "gateway_available_external": None,
                "provenance": {
                    "trace_id": trace_id,
                    "model": "elevation_scan_linear_boundary",
                    "physical_parameter_source": "constellation.gateway_profile",
                },
            }
        )
    return rows


def _access_membership(access_trace: dict[str, Any]) -> dict[tuple[str, float], bool]:
    membership: dict[tuple[str, float], bool] = {}
    sample_times = sorted({float(row["time_s"]) for row in access_trace.get("visible_count_samples", ())})
    for window in access_trace.get("access_windows", ()):
        for time_s in sample_times:
            if float(window["start_s"]) <= time_s < float(window["end_s"]):
                membership[(str(window["sgl_link_id"]), time_s)] = True
    return membership


def _physical_link(
    *,
    link_id: str,
    kind: str,
    endpoint_a: str,
    endpoint_b: str,
    relation: str,
    profile_id: str,
    profile: dict[str, Any],
    policy: str,
    priority: int,
    terminal_pool_id: str | None = None,
    terminal_pool_kind: str | None = None,
    terminal_count_per_satellite: int | None = None,
    setup_delay_s: float | None = None,
) -> dict[str, Any]:
    bandwidth_bps = _float(profile.get("bandwidth_bps"), 0.0) if profile else 0.0
    max_range_km = _optional_float(profile.get("max_range_km")) if profile else None
    terminal_count = terminal_count_per_satellite
    if terminal_count is None:
        terminal_count = _optional_int(profile.get("terminal_count_per_satellite")) if profile else None
    resolved_setup_delay_s = setup_delay_s
    if resolved_setup_delay_s is None and profile:
        resolved_setup_delay_s = _optional_float(profile.get("setup_delay_s"))
    if resolved_setup_delay_s is None and profile:
        resolved_setup_delay_s = _optional_float(profile.get("tau_setup_s"))
    profile_resolved = (
        bool(profile)
        and bandwidth_bps > 0.0
        and terminal_count is not None
        and terminal_count > 0
    )
    pool_id = terminal_pool_id or str(profile.get("terminal_pool") if profile else "") or kind
    return {
        "link_id": link_id,
        "kind": kind,
        "runtime_kind": "sgl" if kind == "sgl" else "isl",
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "orbital_relation": relation,
        "bidirectional": True,
        "profile_id": profile_id,
        "terminal_pool_id": pool_id,
        "terminal_pool_kind": terminal_pool_kind or kind,
        "terminal_pool_capacity_per_satellite": terminal_count,
        "terminal_required_per_endpoint": 1,
        "installed_by_policy": policy,
        "activation_priority": priority,
        "bandwidth_bps_nominal": bandwidth_bps,
        "bandwidth_bps": bandwidth_bps,
        "propagation_delay_s": 0.0,
        "setup_delay_s": resolved_setup_delay_s or 0.0,
        "max_range_km": max_range_km,
        "terminal_count_per_satellite": terminal_count,
        "profile_resolved": profile_resolved,
        "edge_available_external": profile_resolved,
        "unavailable_reason": (
            None
            if profile_resolved
            else "missing_profile_bandwidth_or_terminal_count"
        ),
    }


def _link_sample(
    link: dict[str, Any],
    time_s: float,
    distance_km: float,
    los_ok: bool,
    available: bool,
    unavailable_reason: str | None,
) -> dict[str, Any]:
    max_range_km = _optional_float(link.get("max_range_km"))
    distance_ok = max_range_km is None or distance_km <= max_range_km
    return {
        "time_s": time_s,
        "link_id": str(link["link_id"]),
        "distance_km": distance_km,
        "tau_prop_s": distance_km / C_LIGHT_KM_PER_S,
        "line_of_sight_ok": bool(los_ok),
        "distance_ok": bool(distance_ok),
        "field_of_regard_ok": None,
        "pointing_ok": None,
        "acquisition_tracking_ok": None,
        "sun_exclusion_ok": None,
        "edge_available_external": bool(available),
        "unavailable_reason": unavailable_reason,
        "B_raw_bps": float(link.get("bandwidth_bps_nominal", 0.0)),
        "tau_setup_s": _float(link.get("setup_delay_s"), 0.0),
    }


def _active_graph_snapshots(
    times: tuple[float, ...],
    physical: dict[str, dict[str, Any]],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_time: dict[float, list[str]] = {time_s: [] for time_s in times}
    for sample in samples:
        if sample.get("edge_available_external"):
            by_time.setdefault(float(sample["time_s"]), []).append(str(sample["link_id"]))
    return [
        {
            "time_s": time_s,
            "active_link_ids": sorted(set(by_time.get(time_s, ()))),
            "candidate_link_count": len(set(by_time.get(time_s, ()))),
            "activation_policy": "orbit_visibility_only_runtime_selects_terminals",
        }
        for time_s in times
    ]


def _runtime_compat_links(physical: dict[str, dict[str, Any]], samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available_by_link: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        if sample.get("edge_available_external"):
            available_by_link.setdefault(str(sample["link_id"]), []).append(sample)
    rows: list[dict[str, Any]] = []
    for link_id in sorted(physical):
        link = physical[link_id]
        if link_id not in available_by_link and link.get("kind") != "sgl":
            continue
        link_samples = available_by_link.get(link_id, [])
        delay = min((float(sample["tau_prop_s"]) for sample in link_samples), default=0.0)
        rows.append(
            {
                "link_id": link_id,
                "kind": link.get("runtime_kind", link.get("kind")),
                "physical_kind": link.get("kind"),
                "endpoint_a": link["endpoint_a"],
                "endpoint_b": link["endpoint_b"],
                "bandwidth_bps": link.get("bandwidth_bps_nominal", 0.0),
                "propagation_delay_s": delay,
                "setup_delay_s": link.get("setup_delay_s", 0.0),
                "bidirectional": bool(link.get("bidirectional", True)),
                "terminal_pool_id": link.get("terminal_pool_id"),
                "terminal_pool_kind": link.get("terminal_pool_kind"),
                "edge_available_external": bool(link_samples) or bool(
                    link.get("kind") == "sgl" and link.get("profile_resolved", False)
                ),
                "provenance": {"source": "circular_geometry_trace.compat_links"},
            }
        )
    return rows


def _terminal_catalog(catalog: tuple[dict[str, Any], ...], physical: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sat in catalog:
        satellite_id = str(sat["satellite_id"])
        incident = [link for link in physical.values() if satellite_id in {str(link["endpoint_a"]), str(link["endpoint_b"])}]
        pools: dict[str, dict[str, Any]] = {}
        for link in incident:
            pool = str(link.get("terminal_pool_id") or link.get("kind"))
            count = _optional_int(link.get("terminal_pool_capacity_per_satellite"))
            if count is None:
                count = _optional_int(link.get("terminal_count_per_satellite")) or 0
            item = pools.setdefault(
                pool,
                {
                    "terminal_count": 0,
                    "profile_id": str(link.get("profile_id") or ""),
                    "terminal_pool_kind": str(link.get("terminal_pool_kind") or link.get("kind")),
                    "link_kinds": set(),
                },
            )
            item["terminal_count"] = max(int(item["terminal_count"]), count)
            item["link_kinds"].add(str(link.get("kind")))
        for pool, item in sorted(pools.items()):
            rows.append(
                {
                    "satellite_id": satellite_id,
                    "terminal_pool_id": pool,
                    "terminal_count": int(item["terminal_count"]),
                    "profile_id": item["profile_id"] or None,
                    "terminal_pool_kind": item["terminal_pool_kind"],
                    "link_kinds": sorted(item["link_kinds"]),
                    "shared_with": [],
                }
            )
    return rows


def _propagate_circular(
    *,
    altitude_km: float,
    inclination_deg: float,
    raan_deg: float,
    u0_deg: float,
    time_s: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    a_km = EARTH_RADIUS_KM + altitude_km
    n = math.sqrt(EARTH_MU_KM3_S2 / (a_km**3))
    u_rad = math.radians(u0_deg) + n * time_s
    r_pqw = (a_km * math.cos(u_rad), a_km * math.sin(u_rad), 0.0)
    v_pqw = (-a_km * n * math.sin(u_rad), a_km * n * math.cos(u_rad), 0.0)
    inc = math.radians(inclination_deg)
    raan = math.radians(raan_deg)
    r_eci = _rz(_rx(r_pqw, inc), raan)
    v_eci = _rz(_rx(v_pqw, inc), raan)
    return r_eci, v_eci, math.degrees(u_rad) % 360.0


def _apply_lvlh_offset(
    r_eci: tuple[float, float, float],
    v_eci: tuple[float, float, float],
    offset_lvlh_m: tuple[float, float, float],
) -> tuple[float, float, float]:
    if not any(abs(value) > 0.0 for value in offset_lvlh_m):
        return r_eci
    e_r = _unit(r_eci)
    e_h = _unit(_cross(r_eci, v_eci))
    e_t = _cross(e_h, e_r)
    x_m, y_m, z_m = offset_lvlh_m
    offset_km = _scale(_add(_add(_scale(e_t, x_m), _scale(e_h, y_m)), _scale(e_r, z_m)), 1.0 / 1000.0)
    return _add(r_eci, offset_km)


def _elevation_range(
    ground_ecef: tuple[float, float, float],
    sat_ecef: tuple[float, float, float],
) -> tuple[float, float]:
    rho = _sub(sat_ecef, ground_ecef)
    range_km = _norm(rho)
    if range_km <= 0.0:
        return -90.0, 0.0
    elevation = math.asin(max(-1.0, min(1.0, _dot(_scale(rho, 1.0 / range_km), _unit(ground_ecef)))))
    return math.degrees(elevation), range_km


def _boolean_windows(rows: list[dict[str, Any]], *, duration_s: float) -> list[tuple[float, float, bool]]:
    if not rows:
        return []
    windows: list[tuple[float, float, bool]] = []
    current_flag = bool(rows[0]["flag"])
    current_start = 0.0
    previous = rows[0]
    for row in rows[1:]:
        flag = bool(row["flag"])
        if flag != current_flag:
            boundary = (float(previous["time_s"]) + float(row["time_s"])) / 2.0
            windows.append((current_start, boundary, current_flag))
            current_start = boundary
            current_flag = flag
        previous = row
    windows.append((current_start, duration_s, current_flag))
    return [(start, end, flag) for start, end, flag in windows if end > start]


def _linear_boundary(previous: dict[str, Any], current: dict[str, Any]) -> float:
    t0 = float(previous["time_s"])
    t1 = float(current["time_s"])
    f0 = float(previous["score"])
    f1 = float(current["score"])
    if f1 == f0:
        return (t0 + t1) / 2.0
    ratio = max(0.0, min(1.0, (0.0 - f0) / (f1 - f0)))
    return t0 + ratio * (t1 - t0)


def _merge_windows(
    windows: list[tuple[float, float, list[dict[str, Any]]]],
    *,
    merge_gap_s: float,
) -> list[tuple[float, float, list[dict[str, Any]]]]:
    if not windows:
        return []
    merged = [windows[0]]
    for start_s, end_s, samples in windows[1:]:
        prev_start, prev_end, prev_samples = merged[-1]
        if start_s - prev_end <= merge_gap_s:
            merged[-1] = (prev_start, end_s, [*prev_samples, *samples])
        else:
            merged.append((start_s, end_s, samples))
    return merged


def _sunlit_cylindrical(
    r_eci: tuple[float, float, float],
    sun_vector: tuple[float, float, float],
    *,
    atmosphere_margin_km: float,
) -> bool:
    projection = _dot(r_eci, sun_vector)
    behind_earth = projection < 0.0
    closest = _sub(r_eci, _scale(sun_vector, projection))
    shadow_distance = _norm(closest)
    return not (behind_earth and shadow_distance < EARTH_RADIUS_KM + atmosphere_margin_km)


def _sun_vector(options: dict[str, Any], warnings: list[str]) -> tuple[float, float, float]:
    shadow = _dict(options.get("shadow_model"))
    configured = shadow.get("sun_vector_eci") or options.get("sun_vector_eci")
    if configured:
        try:
            vector = tuple(float(value) for value in configured)
            if len(vector) == 3:
                return _unit(vector)
        except (TypeError, ValueError):
            warnings.append("invalid configured sun_vector_eci; using [1,0,0]")
    warnings.append("sun vector uses simplified placeholder [1,0,0]")
    return (1.0, 0.0, 0.0)


def _ground_ecef(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    return (
        EARTH_RADIUS_KM * math.cos(lat) * math.cos(lon),
        EARTH_RADIUS_KM * math.cos(lat) * math.sin(lon),
        EARTH_RADIUS_KM * math.sin(lat),
    )


def _ecef_to_lla_spherical(r_ecef: tuple[float, float, float]) -> tuple[float, float, float]:
    radius = _norm(r_ecef)
    if radius <= 0:
        return 0.0, 0.0, -EARTH_RADIUS_KM
    lat = math.degrees(math.asin(max(-1.0, min(1.0, r_ecef[2] / radius))))
    lon = math.degrees(math.atan2(r_ecef[1], r_ecef[0]))
    return lat, lon, radius - EARTH_RADIUS_KM


def _rx(vec: tuple[float, float, float], angle_rad: float) -> tuple[float, float, float]:
    x, y, z = vec
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (x, y * c - z * s, y * s + z * c)


def _rz(vec: tuple[float, float, float], angle_rad: float) -> tuple[float, float, float]:
    x, y, z = vec
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (x * c - y * s, x * s + y * c, z)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(a, a))


def _unit(a: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = _norm(a)
    if norm <= 0:
        return (1.0, 0.0, 0.0)
    return (a[0] / norm, a[1] / norm, a[2] / norm)


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a: tuple[float, float, float], scalar: float) -> tuple[float, float, float]:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def _architecture_analysis(
    constellation: Any,
    payload: dict[str, Any],
) -> dict[str, Any]:
    method = getattr(constellation, "architecture_analysis", None)
    if callable(method):
        try:
            return _mapping(method())
        except Exception:
            return _dict(payload.get("architecture_analysis"))
    return _dict(payload.get("architecture_analysis"))


def _gateway_profile(constellation: Any, payload: dict[str, Any]) -> dict[str, Any]:
    configured = getattr(constellation, "gateway_profile", None)
    if configured is None:
        configured = payload.get("gateway_profile")
    profile = _mapping(configured)
    return {
        **profile,
        "bandwidth_bps": _nonnegative_finite_float(profile.get("bandwidth_bps")),
        "setup_delay_s": _nonnegative_finite_float(profile.get("setup_delay_s")),
        "terminal_count_per_satellite": _nonnegative_int(
            profile.get("terminal_count_per_satellite")
        ),
    }


def _workload_provenance(workload_generation: Any) -> dict[str, Any]:
    payload = _mapping(workload_generation)
    manifest = _dict(payload.get("manifest"))
    return {
        "bundle_dir": payload.get("bundle_dir"),
        "request_count": payload.get("request_count") or len(payload.get("requests", ()) or ()),
        "manifest_schema_version": manifest.get("schema_version"),
        "manifest_provenance": _dict(manifest.get("provenance")),
    }


def _direct_ingress_id(request: dict[str, Any], lat: float, lon: float) -> str:
    if request.get("ingress_id") is not None:
        return str(request["ingress_id"])
    if request.get("region_id") is not None:
        return f"user_region:{request['region_id']}"
    return f"direct:{lat:.4f}:{lon:.4f}"


def _time_iso_from_epoch(epoch: str | None, time_s: float) -> str:
    if epoch:
        try:
            base = datetime.fromisoformat(epoch.replace("Z", "+00:00"))
            return (base + timedelta(seconds=time_s)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    return f"epoch+{time_s:.3f}s"


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _mapping(value: Any) -> dict[str, Any]:
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


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


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
    except (TypeError, ValueError, OverflowError):
        return None


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _nonnegative_finite_float(value: Any) -> float:
    parsed = _optional_float(value)
    if parsed is None or not math.isfinite(parsed):
        return 0.0
    return max(0.0, parsed)


def _nonnegative_int(value: Any) -> int:
    parsed = _optional_float(value)
    if parsed is None or not math.isfinite(parsed):
        return 0
    return max(0, int(parsed))


def _energy_idle_power_w(options: dict[str, Any], sunlit: bool) -> float:
    key = "idle_power_w_sunlit" if sunlit else "idle_power_w_shadow"
    if options.get(key) is not None:
        return _float(options.get(key), 50.0)
    return _float(options.get("idle_power_w"), 50.0)


def _runtime_energy_envelope_unavailable(options: dict[str, Any]) -> bool:
    envelope = _dict(options.get("runtime_energy_envelope"))
    return bool(
        options.get("runtime_energy_envelope_required")
        and (options.get("runtime_energy_envelope_unavailable") or not envelope.get("evaluated", False))
    )


def _positive_float(value: Any, default: float, field_name: str) -> float:
    result = _float(value, default)
    if result <= 0:
        raise ValueError(f"{field_name} must be positive")
    return result


def _normalise(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


__all__ = [
    "MotionTraceBuildResult",
    "build_satellite_motion_trace",
]
