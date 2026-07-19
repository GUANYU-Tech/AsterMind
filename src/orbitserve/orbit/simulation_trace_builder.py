"""Deterministic surrogate orbit-trace builder."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import to_jsonable
from .simulation_traces import (
    SURROGATE_ORBIT_TRACE,
    SURROGATE_TRACE_SCHEMA,
    TRACE_BACKED_RUNTIME_SURROGATE_SCOPE,
    OrbitTraceManifest,
    trace_file_sha256,
)


@dataclass(frozen=True)
class OrbitTraceBuildResult:
    trace_dir: str
    trace_manifest: OrbitTraceManifest
    access_trace: dict[str, Any]
    link_trace: dict[str, Any]
    energy_trace: dict[str, Any]
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_trace_manifest(self) -> dict[str, Any]:
        return self.trace_manifest.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def stable_int_hash(*parts: Any) -> int:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def build_surrogate_orbit_trace(
    *,
    config: Any,
    constellation: Any,
    workload_generation: Any,
    base_dir: str | Path | None = None,
    experiment_id: str | None = None,
) -> OrbitTraceBuildResult:
    workload = _mapping(workload_generation)
    requests = tuple(_dict(item) for item in workload.get("requests", ()) if isinstance(item, dict))
    gateways = tuple(_dict(item) for item in workload.get("gateways", ()) if isinstance(item, dict))
    regions = tuple(_dict(item) for item in workload.get("regions", ()) if isinstance(item, dict))
    options = _trace_options(config)
    configured_duration_s = _float(getattr(config, "duration_s", None), 21600.0)
    access_duration_s = _float(options.get("access_duration_s"), 240.0)
    gap_next_access_delay_s = _float(options.get("gap_next_access_delay_s"), 120.0)
    max_arrival_s = max((_float(item.get("arrival_time_s"), 0.0) for item in requests), default=0.0)
    effective_duration_s = max(
        configured_duration_s,
        max_arrival_s + access_duration_s + gap_next_access_delay_s,
    )
    trace_id = _trace_id(config, experiment_id, effective_duration_s)
    trace_dir = _trace_dir(config, base_dir=base_dir, experiment_id=experiment_id)
    trace_dir.mkdir(parents=True, exist_ok=True)

    constellation_payload = _mapping(constellation)
    satellite_count = _satellite_count(constellation, constellation_payload)
    if satellite_count <= 0:
        raise ValueError("surrogate trace requires constellation satellite_count > 0")
    architecture = _architecture_analysis(constellation, constellation_payload)
    gateway_profile = _gateway_profile(constellation, constellation_payload)
    warnings: list[str] = [
        "surrogate_orbit_trace is deterministic and not STK-calibrated",
        "valid_for_paper_main_results=false",
    ]
    if _float(gateway_profile.get("bandwidth_bps"), 0.0) <= 0.0:
        warnings.append(
            "constellation.gateway_profile.bandwidth_bps is missing or non-positive; "
            "surrogate SGL capacity fails closed"
        )
    if _int(gateway_profile.get("terminal_count_per_satellite"), 0) <= 0:
        warnings.append(
            "constellation.gateway_profile.terminal_count_per_satellite is missing or non-positive; "
            "surrogate SGL terminal capacity fails closed"
        )
    satellite_catalog, catalog_warnings = _satellite_catalog(constellation, constellation_payload, satellite_count)
    warnings.extend(catalog_warnings)

    ingress_catalog = _ingress_catalog(gateways)
    access_trace, access_warnings = _build_access_trace(
        trace_id=trace_id,
        requests=requests,
        gateways=gateways,
        regions=regions,
        ingress_catalog=ingress_catalog,
        satellite_catalog=satellite_catalog,
        gateway_profile=gateway_profile,
        options=options,
        effective_duration_s=effective_duration_s,
    )
    warnings.extend(access_warnings)
    link_trace, link_warnings = _build_link_trace(
        trace_id=trace_id,
        access_windows=access_trace["access_windows"],
        satellite_catalog=satellite_catalog,
        constellation=constellation,
        constellation_payload=constellation_payload,
        architecture=architecture,
        gateway_profile=gateway_profile,
        options=options,
    )
    warnings.extend(link_warnings)
    energy_trace, energy_warnings = _build_energy_trace(
        trace_id=trace_id,
        satellite_catalog=satellite_catalog,
        architecture=architecture,
        options=options,
        effective_duration_s=effective_duration_s,
    )
    warnings.extend(energy_warnings)

    provenance = {
        "trace_dir": str(trace_dir),
        "builder": "surrogate_orbit_trace",
        "schema_version": SURROGATE_TRACE_SCHEMA,
        "source": "simulation_trace_builder.build_surrogate_orbit_trace",
        "trace_options": to_jsonable(options),
        "configured_duration_s": configured_duration_s,
        "effective_duration_s": effective_duration_s,
    }
    for payload in (access_trace, link_trace, energy_trace):
        payload["warnings"] = list(dict.fromkeys((*payload.get("warnings", ()), *warnings)))
        payload["provenance"] = {**_dict(payload.get("provenance")), **provenance}

    access_path = trace_dir / "access_trace.json"
    link_path = trace_dir / "link_trace.json"
    energy_path = trace_dir / "energy_trace.json"
    _write_json_deterministic(access_path, access_trace)
    _write_json_deterministic(link_path, link_trace)
    _write_json_deterministic(energy_path, energy_trace)
    manifest = OrbitTraceManifest(
        trace_model=SURROGATE_ORBIT_TRACE,
        evaluated=True,
        trace_id=trace_id,
        access_trace_path=access_path.name,
        link_trace_path=link_path.name,
        energy_trace_path=energy_path.name,
        access_trace_sha256=trace_file_sha256(access_path),
        link_trace_sha256=trace_file_sha256(link_path),
        energy_trace_sha256=trace_file_sha256(energy_path),
        valid_for_paper_main_results=False,
        scope=TRACE_BACKED_RUNTIME_SURROGATE_SCOPE,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=provenance,
    )
    return OrbitTraceBuildResult(
        trace_dir=str(trace_dir),
        trace_manifest=manifest,
        access_trace=access_trace,
        link_trace=link_trace,
        energy_trace=energy_trace,
        warnings=tuple(dict.fromkeys(warnings)),
        provenance=provenance,
    )


def _trace_options(config: Any) -> dict[str, Any]:
    options = {
        "access_period_s": 600.0,
        "access_duration_s": 240.0,
        "visible_satellite_count": 2,
        "sgl_propagation_delay_s": 0.02,
        "isl_propagation_delay_s": 0.005,
        "orbit_period_s": 5700.0,
        "shadow_fraction": 0.35,
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
        "access_gap_request_ids": [],
        "force_access_gaps": [],
        "gap_next_access_delay_s": 120.0,
        "generate_defer_windows_for_gap_requests": True,
    }
    configured = getattr(config, "trace_options", None)
    if isinstance(configured, dict):
        options.update(configured)
    return options


def _trace_id(config: Any, experiment_id: str | None, duration_s: float) -> str:
    epoch = getattr(config, "epoch", "unknown_epoch")
    name = experiment_id or "runtime_scenario"
    return f"{SURROGATE_ORBIT_TRACE}:{name}:{epoch}:{duration_s:g}s"


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


def _satellite_count(constellation: Any, payload: dict[str, Any]) -> int:
    value = getattr(constellation, "satellite_count", None)
    if value is None:
        value = payload.get("satellite_count")
    if value is None:
        value = _dict(payload.get("architecture_analysis")).get("satellite_count")
    return _int(value, 0)


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
    # The gateway profile is the only SGL physical input.  Keep the runtime
    # representation canonical without introducing a second profile catalog.
    return {
        **profile,
        "bandwidth_bps": _nonnegative_finite_float(profile.get("bandwidth_bps")),
        "setup_delay_s": _nonnegative_finite_float(profile.get("setup_delay_s")),
        "terminal_count_per_satellite": _nonnegative_int(
            profile.get("terminal_count_per_satellite")
        ),
    }


def _satellite_catalog(
    constellation: Any,
    payload: dict[str, Any],
    satellite_count: int,
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    del constellation
    orbital_sets = payload.get("orbital_sets") or _dict(payload.get("architecture_analysis")).get("orbital_sets") or ()
    satellites: list[dict[str, Any]] = []
    ordinal = 0
    for orbital_set in orbital_sets if isinstance(orbital_sets, list) else ():
        item = _dict(orbital_set)
        planes = max(1, _int(item.get("planes") or _dict(item.get("placement")).get("effective_plane_count"), 1))
        cells = max(1, _int(item.get("clusters_per_plane") or _dict(item.get("placement")).get("clusters_per_plane"), 1))
        cluster = _dict(item.get("cluster"))
        sats_per_cluster = max(
            1,
            _int(
                cluster.get("satellites_per_cluster")
                or _dict(item.get("placement")).get("satellites_per_cluster"),
                1,
            ),
        )
        for plane_index in range(planes):
            for cell_index in range(cells):
                for member_index in range(sats_per_cluster):
                    if ordinal >= satellite_count:
                        break
                    satellites.append(
                        {
                            "satellite_id": f"sat-{ordinal:05d}",
                            "ordinal": ordinal,
                            "deployment_id": "deployment_0",
                            "orbital_set_name": item.get("name"),
                            "plane_index": plane_index,
                            "cell_index": cell_index,
                            "cluster_member_index": member_index,
                        }
                    )
                    ordinal += 1
    if satellites:
        while len(satellites) < satellite_count:
            ordinal = len(satellites)
            satellites.append({"satellite_id": f"sat-{ordinal:05d}", "ordinal": ordinal, "deployment_id": "deployment_0"})
        return tuple(satellites), ()
    return (
        tuple(
            {"satellite_id": f"sat-{index:05d}", "ordinal": index, "deployment_id": "deployment_0"}
            for index in range(satellite_count)
        ),
        ("count-only satellite catalog used",),
    )


def _ingress_catalog(gateways: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    return {
        str(gateway["gateway_id"]): gateway
        for gateway in gateways
        if gateway.get("gateway_id") is not None
    }


def _build_access_trace(
    *,
    trace_id: str,
    requests: tuple[dict[str, Any], ...],
    gateways: tuple[dict[str, Any], ...],
    regions: tuple[dict[str, Any], ...],
    ingress_catalog: dict[str, dict[str, Any]],
    satellite_catalog: tuple[dict[str, Any], ...],
    gateway_profile: dict[str, Any],
    options: dict[str, Any],
    effective_duration_s: float,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    del regions
    warnings: list[str] = []
    windows: dict[tuple[str, str, float, float], dict[str, Any]] = {}
    gap_ids = {str(item) for item in options.get("access_gap_request_ids", ()) or ()}
    gap_ids.update(str(item) for item in options.get("force_access_gaps", ()) or ())
    for request in requests:
        request_id = str(request.get("request_id", ""))
        ingress_id = _resolve_ingress_id(request, gateways, ingress_catalog)
        arrival_s = _float(request.get("arrival_time_s"), 0.0)
        if ingress_id is None:
            warnings.append(f"request {request_id or '<unknown>'} has no resolvable ingress")
            continue
        if request_id in gap_ids:
            continue
        for satellite_id in _visible_satellite_ids(trace_id, ingress_id, arrival_s, satellite_catalog, options):
            start_s = max(0.0, arrival_s - _float(options.get("access_duration_s"), 240.0) / 2.0)
            end_s = min(effective_duration_s, arrival_s + _float(options.get("access_duration_s"), 240.0) / 2.0)
            _add_access_window(
                windows,
                trace_id,
                ingress_id,
                satellite_id,
                start_s,
                end_s,
                gateway_profile,
                options,
            )

    for request in requests:
        request_id = str(request.get("request_id", ""))
        if request_id not in gap_ids:
            continue
        ingress_id = _resolve_ingress_id(request, gateways, ingress_catalog)
        arrival_s = _float(request.get("arrival_time_s"), 0.0)
        if ingress_id is None:
            continue
        future_start = arrival_s + _float(options.get("gap_next_access_delay_s"), 120.0)
        for key, window in list(windows.items()):
            overlaps_forced_gap = (
                window["start_s"] < future_start
                and window["end_s"] >= arrival_s
            )
            if window["ingress_id"] == ingress_id and overlaps_forced_gap:
                del windows[key]
        if bool(options.get("generate_defer_windows_for_gap_requests", True)):
            future_end = min(
                effective_duration_s,
                future_start + _float(options.get("access_duration_s"), 240.0),
            )
            for satellite_id in _visible_satellite_ids(trace_id, ingress_id, future_start, satellite_catalog, options):
                _add_access_window(
                    windows,
                    trace_id,
                    ingress_id,
                    satellite_id,
                    future_start,
                    future_end,
                    gateway_profile,
                    options,
                )

    ordered_windows = sorted(windows.values(), key=lambda item: (item["start_s"], item["ingress_id"], item["satellite_id"]))
    for index, window in enumerate(ordered_windows):
        window["window_id"] = f"access-{index:06d}"
    return (
        {
            "schema_version": SURROGATE_TRACE_SCHEMA,
            "trace_id": trace_id,
            "satellite_catalog": list(satellite_catalog),
            "access_windows": ordered_windows,
            "warnings": warnings,
            "provenance": {"model": "request-aware deterministic coverage surrogate"},
        },
        tuple(warnings),
    )


def _add_access_window(
    windows: dict[tuple[str, str, float, float], dict[str, Any]],
    trace_id: str,
    ingress_id: str,
    satellite_id: str,
    start_s: float,
    end_s: float,
    gateway_profile: dict[str, Any],
    options: dict[str, Any],
) -> None:
    key = (ingress_id, satellite_id, round(start_s, 3), round(end_s, 3))
    if key in windows:
        return
    terminal_count = _int(gateway_profile.get("terminal_count_per_satellite"), 0)
    bandwidth_bps = _float(gateway_profile.get("bandwidth_bps"), 0.0) if terminal_count > 0 else 0.0
    windows[key] = {
        "window_id": "",
        "ingress_id": ingress_id,
        "satellite_id": satellite_id,
        "start_s": start_s,
        "end_s": end_s,
        "sgl_link_id": f"sgl:{ingress_id}:{satellite_id}",
        "propagation_delay_s": _float(options.get("sgl_propagation_delay_s"), 0.02),
        "setup_delay_s": _float(gateway_profile.get("setup_delay_s"), 0.0),
        "bandwidth_bps": bandwidth_bps,
        "terminal_count_per_satellite": terminal_count,
        "provenance": {
            "trace_id": trace_id,
            "model": "request_centered_access_window",
            "physical_parameter_source": "constellation.gateway_profile",
        },
    }


def _visible_satellite_ids(
    trace_id: str,
    ingress_id: str,
    arrival_s: float,
    satellite_catalog: tuple[dict[str, Any], ...],
    options: dict[str, Any],
) -> tuple[str, ...]:
    satellite_ids = [str(item["satellite_id"]) for item in satellite_catalog]
    if not satellite_ids:
        return ()
    period = max(1.0, _float(options.get("access_period_s"), 600.0))
    bin_index = math.floor(arrival_s / period)
    count = min(max(1, _int(options.get("visible_satellite_count"), 2)), len(satellite_ids))
    start = stable_int_hash(trace_id, ingress_id, bin_index, "visible") % len(satellite_ids)
    return tuple(sorted(satellite_ids[(start + offset) % len(satellite_ids)] for offset in range(count)))


def _resolve_ingress_id(
    request: dict[str, Any],
    gateways: tuple[dict[str, Any], ...],
    ingress_catalog: dict[str, dict[str, Any]],
) -> str | None:
    value = request.get("ingress_id")
    if value is not None:
        return str(value)
    value = request.get("gateway_id")
    if value is not None:
        return str(value)
    region_id = request.get("region_id")
    if region_id is not None:
        for gateway in gateways:
            if str(region_id) in {str(item) for item in gateway.get("served_region_ids", ()) or ()}:
                return str(gateway.get("gateway_id"))
    if gateways:
        return str(gateways[0].get("gateway_id"))
    return None


def _build_link_trace(
    *,
    trace_id: str,
    access_windows: list[dict[str, Any]],
    satellite_catalog: tuple[dict[str, Any], ...],
    constellation: Any,
    constellation_payload: dict[str, Any],
    architecture: dict[str, Any],
    gateway_profile: dict[str, Any],
    options: dict[str, Any],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    warnings: list[str] = []
    links: dict[str, dict[str, Any]] = {}
    for window in access_windows:
        link_id = str(window["sgl_link_id"])
        links[link_id] = {
            "link_id": link_id,
            "kind": "sgl",
            "endpoint_a": window["ingress_id"],
            "endpoint_b": window["satellite_id"],
            "bandwidth_bps": window["bandwidth_bps"],
            "propagation_delay_s": window["propagation_delay_s"],
            "setup_delay_s": window.get("setup_delay_s", 0.0),
            "max_range_km": gateway_profile.get("max_range_km"),
            "terminal_pool_id": str(gateway_profile.get("terminal_pool") or "gateway_sgl"),
            "terminal_pool_capacity_per_satellite": window.get("terminal_count_per_satellite", 0),
            "bidirectional": True,
            "provenance": {
                "source": "access_window",
                "physical_parameter_source": "constellation.gateway_profile",
            },
        }
    satellite_ids = [str(item["satellite_id"]) for item in satellite_catalog]
    isl_bandwidth_bps, terminal_count, bandwidth_warning = _isl_hardware_capability(options)
    if bandwidth_warning:
        warnings.append(bandwidth_warning)
    if len(satellite_ids) > 1:
        for index, source in enumerate(sorted(satellite_ids)):
            for target in sorted(satellite_ids)[index + 1:]:
                link_id = f"isl:{source}:{target}"
                links[link_id] = {
                    "link_id": link_id,
                    "kind": "isl",
                    "endpoint_a": source,
                    "endpoint_b": target,
                    "bandwidth_bps": isl_bandwidth_bps,
                    "bandwidth_bps_nominal": isl_bandwidth_bps,
                    "propagation_delay_s": _float(options.get("isl_propagation_delay_s"), 0.005),
                    "terminal_pool_id": "satellite_isl_terminals",
                    "terminal_pool_capacity_per_satellite": terminal_count,
                    "terminal_count_per_satellite": terminal_count,
                    "edge_available_external": bool(isl_bandwidth_bps > 0 and terminal_count > 0),
                    "bidirectional": True,
                    "provenance": {"source": "surrogate_visibility_candidate"},
                }
    ordered = sorted(links.values(), key=lambda item: (item["kind"], item["link_id"]))
    return (
        {
            "schema_version": SURROGATE_TRACE_SCHEMA,
            "trace_id": trace_id,
            "links": ordered,
            "warnings": warnings,
            "provenance": {"model": "sgl_access_links_plus_visibility_candidates"},
        },
        tuple(warnings),
    )


def _isl_hardware_capability(options: dict[str, Any]) -> tuple[float, int, str | None]:
    hardware = _dict(options.get("satellite_link_hardware"))
    terminal_count = int(_float(hardware.get("terminal_count_per_satellite"), 0.0))
    terminal_bandwidth_gbps = _float(hardware.get("terminal_bandwidth_gbps"), 0.0)
    if terminal_count > 0 and terminal_bandwidth_gbps > 0.0:
        return terminal_bandwidth_gbps * 1_000_000_000.0, terminal_count, None
    return 0.0, 0, "single-satellite ISL terminal capability is missing; surrogate ISL links fail closed"


def _build_energy_trace(
    *,
    trace_id: str,
    satellite_catalog: tuple[dict[str, Any], ...],
    architecture: dict[str, Any],
    options: dict[str, Any],
    effective_duration_s: float,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    warnings = ["periodic shadow surrogate, not solar geometry"]
    if _runtime_energy_envelope_unavailable(options):
        warnings.append("runtime energy envelope unavailable; runtime SOC fields omitted to fail closed")
    orbit_period_s = _orbit_period_s(architecture, options)
    shadow_fraction = min(max(_float(options.get("shadow_fraction"), 0.35), 0.0), 1.0)
    shadow_duration_s = orbit_period_s * shadow_fraction
    energy_windows: list[dict[str, Any]] = []
    initial_soc = {
        str(item["satellite_id"]): _float(options.get("initial_soc"), 0.8)
        for item in satellite_catalog
    }
    for satellite in satellite_catalog:
        satellite_id = str(satellite["satellite_id"])
        phase_s = stable_int_hash(trace_id, satellite_id, "shadow_phase") % max(1, int(round(orbit_period_s)))
        first_cycle = math.floor((-phase_s) / orbit_period_s) - 1
        last_cycle = math.ceil((effective_duration_s - phase_s) / orbit_period_s) + 1
        for cycle in range(first_cycle, last_cycle + 1):
            cycle_start = phase_s + cycle * orbit_period_s
            shadow_start = cycle_start
            shadow_end = cycle_start + shadow_duration_s
            sunlit_start = shadow_end
            sunlit_end = cycle_start + orbit_period_s
            _append_energy_window(energy_windows, satellite_id, shadow_start, shadow_end, False, options, effective_duration_s)
            _append_energy_window(energy_windows, satellite_id, sunlit_start, sunlit_end, True, options, effective_duration_s)
    energy_windows.sort(key=lambda item: (item["satellite_id"], item["start_s"], item["end_s"], item["sunlit"]))
    soc_inputs = {
        "runtime_energy_envelope": _dict(options.get("runtime_energy_envelope")),
    }
    if not _runtime_energy_envelope_unavailable(options):
        soc_inputs.update(
            {
                "battery_capacity_j": _float(options.get("battery_capacity_j"), 10_000_000.0),
                "soc_min": _float(options.get("soc_min"), 0.2),
                "battery_charge_efficiency": _float(options.get("battery_charge_efficiency"), 1.0),
                "battery_discharge_efficiency": _float(options.get("battery_discharge_efficiency"), 1.0),
                "energy_accounting_mode": str(options.get("energy_accounting_mode", "event_incremental_v1")),
                "charge_rate_w": _float(options.get("charge_rate_w"), 200.0),
                "idle_power_w_sunlit": _energy_idle_power_w(options, True),
                "idle_power_w_shadow": _energy_idle_power_w(options, False),
                "compute_energy_j_per_token": _float(
                    options.get("compute_energy_j_per_token"), 0.5
                ),
                "link_energy_j_per_byte": _float(
                    options.get("link_energy_j_per_byte"), 1e-6
                ),
            }
        )
    return (
        {
            "schema_version": SURROGATE_TRACE_SCHEMA,
            "trace_id": trace_id,
            "satellite_initial_soc": initial_soc,
            "energy_windows": energy_windows,
            "warnings": warnings,
            "soc_inputs": soc_inputs,
            "provenance": {
                "model": "periodic_shadow_soc_surrogate",
                "orbit_period_s": orbit_period_s,
                "runtime_energy_envelope": _dict(options.get("runtime_energy_envelope")),
            },
        },
        tuple(warnings),
    )


def _append_energy_window(
    energy_windows: list[dict[str, Any]],
    satellite_id: str,
    start_s: float,
    end_s: float,
    sunlit: bool,
    options: dict[str, Any],
    effective_duration_s: float,
) -> None:
    clipped_start = max(0.0, start_s)
    clipped_end = min(effective_duration_s, end_s)
    if clipped_end <= clipped_start:
        return
    window = {
        "satellite_id": satellite_id,
        "start_s": clipped_start,
        "end_s": clipped_end,
        "sunlit": sunlit,
        "runtime_energy_envelope": _dict(options.get("runtime_energy_envelope")),
    }
    if _runtime_energy_envelope_unavailable(options):
        window["energy_unavailable_reason"] = "runtime_energy_envelope_unavailable"
    else:
        window.update(
            {
                "dvfs_multiplier": _float(
                    options.get("dvfs_multiplier_sunlit" if sunlit else "dvfs_multiplier_shadow"),
                    1.0 if sunlit else 1.5,
                ),
                "charge_rate_w": _float(options.get("charge_rate_w"), 200.0) if sunlit else 0.0,
                "idle_power_w": _energy_idle_power_w(options, sunlit),
                "compute_energy_j_per_token": _float(options.get("compute_energy_j_per_token"), 0.5),
                "link_energy_j_per_byte": _float(options.get("link_energy_j_per_byte"), 1e-6),
                "battery_capacity_j": _float(options.get("battery_capacity_j"), 10_000_000.0),
                "soc_min": _float(options.get("soc_min"), 0.2),
                "battery_charge_efficiency": _float(
                    options.get("battery_charge_efficiency"),
                    1.0,
                ),
                "battery_discharge_efficiency": _float(
                    options.get("battery_discharge_efficiency"),
                    1.0,
                ),
                "energy_accounting_mode": str(
                    options.get("energy_accounting_mode", "event_incremental_v1")
                ),
            }
        )
    energy_windows.append(window)


def _orbit_period_s(architecture: dict[str, Any], options: dict[str, Any]) -> float:
    if options.get("orbit_period_s") is not None:
        return _float(options.get("orbit_period_s"), 5700.0)
    for item in architecture.get("orbital_sets", ()) or ():
        period = _dict(_dict(item).get("orbit")).get("orbital_period_s")
        if period is not None:
            return _float(period, 5700.0)
    return 5700.0


def _write_json_deterministic(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


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


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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


def _int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _nonnegative_finite_float(value: Any) -> float:
    parsed = _float(value, float("nan"))
    if not math.isfinite(parsed):
        return 0.0
    return max(0.0, parsed)


def _nonnegative_int(value: Any) -> int:
    parsed = _float(value, float("nan"))
    if not math.isfinite(parsed):
        return 0
    return max(0, int(parsed))


__all__ = [
    "OrbitTraceBuildResult",
    "build_surrogate_orbit_trace",
    "stable_int_hash",
]
