"""STK 11.6 geometry-event capture for the isolated Orbit validation lane."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence

from astermind.builder.orbit.sgp4.elements import Sgp4ElementBundle, load_sgp4_element_bundle
from astermind.builder.orbit.sgp4.geometry import Sgp4GroundIngress
from astermind.builder.orbit.validation.stk import STK_11_BASELINE_BUILD, STK_11_BASELINE_VERSION, STK_CONNECT_HOST, STK_CONNECT_PORT, STK_CONNECT_TRANSPORT, STK_PROPAGATOR_MODULE, STK_UPDATE_MODE, STK_VALIDATION_CONTRACT_ID, StkConnectClient, StkConnectError, StkEnvironmentError, StkValidationError, _format_number, _parse_finite_number, _parse_stk_utcg, _prepare_sample_grid, _recover_temporary_scenario, _representable_timedelta, _scenario_is_open, _utc_datetime, build_stk_set_state_command, format_stk_utcg


STK_GEOMETRY_CAPTURE_SCHEMA = 'orbitserve.orbit.stk-geometry-capture'
STK_AER_REPORT_STYLE = "AER"
STK_LIGHTING_REPORT_STYLE = "Lighting Times"
STK_ACCESS_TIME_CONVERGENCE_S = 0.001

_ACCESS_HEADER = re.compile(r"^(?P<from>\S+)\s+(?P<to>\S+)\s+(?P<count>\d+)\s*")
_NO_ACCESSES = re.compile(r"^\S+\s+\S+\s+NoAccesses$")
_UTCG_TOKEN = re.compile(
    r"\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"\d{4} \d{2}:\d{2}:\d{2}\.\d{1,6}"
)
_LIGHTING_SECTION = {
    "Sunlight Times": "sunlight",
    "Penumbra Times": "penumbra",
    "Umbra Times": "umbra",
}
_LIGHTING_ROW = re.compile(
    rf"^\s*(?P<start>{_UTCG_TOKEN.pattern})\s+"
    rf"(?P<stop>{_UTCG_TOKEN.pattern})\s+"
    r"(?P<duration>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)
_AER_HEADER = (
    "Time (UTCG)",
    "Azimuth (deg)",
    "Elevation (deg)",
    "Range (km)",
)


@dataclass(frozen=True, slots=True)
class StkSglWindow:
    ingress_id: str
    satellite_id: str
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_s": self.end_s,
            "ingress_id": self.ingress_id,
            "satellite_id": self.satellite_id,
            "start_s": self.start_s,
        }


@dataclass(frozen=True, slots=True)
class StkIslWindow:
    endpoint_a: str
    endpoint_b: str
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_s": self.end_s,
            "endpoint_a": self.endpoint_a,
            "endpoint_b": self.endpoint_b,
            "start_s": self.start_s,
        }


@dataclass(frozen=True, slots=True)
class StkLightingWindow:
    satellite_id: str
    lighting: Literal["sunlight", "penumbra", "umbra"]
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "end_s": self.end_s,
            "lighting": self.lighting,
            "satellite_id": self.satellite_id,
            "start_s": self.start_s,
        }


@dataclass(frozen=True, slots=True)
class StkAerSample:
    ingress_id: str
    satellite_id: str
    time_s: float
    azimuth_deg: float
    elevation_deg: float
    range_km: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "azimuth_deg": self.azimuth_deg,
            "elevation_deg": self.elevation_deg,
            "ingress_id": self.ingress_id,
            "range_km": self.range_km,
            "satellite_id": self.satellite_id,
            "time_s": self.time_s,
        }


@dataclass(frozen=True, slots=True)
class StkGeometryCapture:
    """Canonical SGL, ISL, lighting, and AER evidence from STK."""

    element_set_id: str
    simulation_epoch_utc: str
    sample_times_s: tuple[float, ...]
    stk_version: str
    installed_build: str | None
    propagator_defaults_configured: bool
    access_defaults_configured: bool
    sgl_windows: tuple[StkSglWindow, ...]
    isl_windows: tuple[StkIslWindow, ...]
    lighting_windows: tuple[StkLightingWindow, ...]
    aer_samples: tuple[StkAerSample, ...]
    warnings: tuple[str, ...]
    schema: str = STK_GEOMETRY_CAPTURE_SCHEMA
    contract_id: str = STK_VALIDATION_CONTRACT_ID
    transport: str = STK_CONNECT_TRANSPORT
    propagator_module: str = STK_PROPAGATOR_MODULE
    update_mode: str = STK_UPDATE_MODE
    access_time_convergence_s: float = STK_ACCESS_TIME_CONVERGENCE_S
    time_scale: str = "UTC"
    distance_unit: str = "km"
    angle_unit: str = "deg"
    interval_semantics: str = "[start_s,end_s)"
    formal_thresholds_frozen: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_defaults_configured": self.access_defaults_configured,
            "access_time_convergence_s": self.access_time_convergence_s,
            "aer_samples": [sample.to_dict() for sample in self.aer_samples],
            "angle_unit": self.angle_unit,
            "contract_id": self.contract_id,
            "distance_unit": self.distance_unit,
            "element_set_id": self.element_set_id,
            "formal_thresholds_frozen": self.formal_thresholds_frozen,
            "installed_build": self.installed_build,
            "interval_semantics": self.interval_semantics,
            "isl_windows": [window.to_dict() for window in self.isl_windows],
            "lighting_windows": [
                window.to_dict() for window in self.lighting_windows
            ],
            "propagator_defaults_configured": (
                self.propagator_defaults_configured
            ),
            "propagator_module": self.propagator_module,
            "sample_times_s": list(self.sample_times_s),
            "schema": self.schema,
            "sgl_windows": [window.to_dict() for window in self.sgl_windows],
            "simulation_epoch_utc": self.simulation_epoch_utc,
            "stk_version": self.stk_version,
            "time_scale": self.time_scale,
            "transport": self.transport,
            "update_mode": self.update_mode,
            "warnings": list(self.warnings),
        }

    def canonical_bytes(self) -> bytes:
        return (
            json.dumps(
                _normalise_json(self.to_dict()),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )


def parse_stk_access_intervals(
    payload: str,
    *,
    simulation_epoch_utc: str,
    interval_start_s: float,
    interval_end_s: float,
) -> tuple[tuple[float, float], ...]:
    """Parse an STK Access response and normalize it to clipped half-open offsets."""

    if not isinstance(payload, str):
        raise TypeError("STK Access payload must be text")
    text = payload.strip()
    if text == "NoAccesses" or _NO_ACCESSES.fullmatch(text):
        return ()
    match = _ACCESS_HEADER.match(text)
    if match is None:
        raise StkValidationError(
            f"unexpected STK Access response header: {text[:240]!r}"
        )
    count = int(match.group("count"))
    timestamps = _UTCG_TOKEN.findall(text[match.end() :])
    if len(timestamps) != count * 2:
        raise StkValidationError("STK Access interval count does not match its payload")
    epoch = _utc_datetime(simulation_epoch_utc)
    intervals = tuple(
        (
            _offset_seconds(_parse_stk_utcg(timestamps[index]), epoch),
            _offset_seconds(_parse_stk_utcg(timestamps[index + 1]), epoch),
        )
        for index in range(0, len(timestamps), 2)
    )
    return _clip_intervals(intervals, interval_start_s, interval_end_s)


def parse_stk_lighting_report(
    rows: Iterable[str],
    *,
    satellite_id: str,
    simulation_epoch_utc: str,
    interval_start_s: float,
    interval_end_s: float,
) -> tuple[StkLightingWindow, ...]:
    """Parse the named Sunlight/Penumbra/Umbra sections of Lighting Times."""

    epoch = _utc_datetime(simulation_epoch_utc)
    current: Literal["sunlight", "penumbra", "umbra"] | None = None
    found: set[str] = set()
    raw: dict[str, list[tuple[float, float]]] = {
        "sunlight": [],
        "penumbra": [],
        "umbra": [],
    }
    for line in rows:
        if not isinstance(line, str):
            raise TypeError("STK lighting report rows must be strings")
        stripped = line.strip()
        if stripped in _LIGHTING_SECTION:
            current = _LIGHTING_SECTION[stripped]  # type: ignore[assignment]
            found.add(current)
            continue
        match = _LIGHTING_ROW.match(line)
        if match is None:
            continue
        if current is None:
            raise StkValidationError("STK lighting row appeared before a named section")
        start = _offset_seconds(_parse_stk_utcg(match.group("start")), epoch)
        stop = _offset_seconds(_parse_stk_utcg(match.group("stop")), epoch)
        duration = _parse_finite_number(
            match.group("duration"), "STK lighting duration"
        )
        if abs((stop - start) - duration) > 0.002:
            raise StkValidationError("STK lighting duration is inconsistent")
        raw[current].append((start, stop))
    if found != set(_LIGHTING_SECTION.values()):
        raise StkValidationError("STK Lighting Times report is missing named sections")
    windows: list[StkLightingWindow] = []
    for lighting in ("sunlight", "penumbra", "umbra"):
        for start, stop in _clip_intervals(
            tuple(raw[lighting]), interval_start_s, interval_end_s
        ):
            windows.append(
                StkLightingWindow(
                    satellite_id=satellite_id,
                    lighting=lighting,  # type: ignore[arg-type]
                    start_s=start,
                    end_s=stop,
                )
            )
    return tuple(windows)


def parse_stk_aer_report(
    rows: Iterable[str],
    *,
    ingress_id: str,
    satellite_id: str,
    simulation_epoch_utc: str,
    interval_start_s: float,
    interval_end_s: float,
) -> tuple[StkAerSample, ...]:
    """Parse Facility-side AER samples and retain only the requested interval."""

    parsed: list[list[str]] = []
    for line in rows:
        if not isinstance(line, str):
            raise TypeError("STK AER report rows must be strings")
        if line.strip():
            parsed.append(
                [field.strip() for field in next(csv.reader(io.StringIO(line)))]
            )
    if not parsed:
        return ()
    if tuple(parsed[0]) != _AER_HEADER:
        raise StkValidationError("unexpected STK AER report header")
    epoch = _utc_datetime(simulation_epoch_utc)
    samples: list[StkAerSample] = []
    for index, row in enumerate(parsed[1:]):
        if len(row) != len(_AER_HEADER):
            raise StkValidationError(f"STK AER row {index} has the wrong field count")
        time_s = _offset_seconds(_parse_stk_utcg(row[0]), epoch)
        if time_s < interval_start_s - 0.001 or time_s > interval_end_s + 0.001:
            raise StkValidationError("STK AER sample is outside the requested interval")
        samples.append(
            StkAerSample(
                ingress_id=ingress_id,
                satellite_id=satellite_id,
                time_s=_q(min(max(time_s, interval_start_s), interval_end_s)),
                azimuth_deg=_parse_finite_number(row[1], "STK AER azimuth"),
                elevation_deg=_parse_finite_number(row[2], "STK AER elevation"),
                range_km=_parse_finite_number(row[3], "STK AER range"),
            )
        )
    keys = [sample.time_s for sample in samples]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise StkValidationError("STK AER samples must be unique and ordered")
    return tuple(samples)


def capture_stk_geometry(
    elements: Sgp4ElementBundle | Mapping[str, Any] | str | Path,
    *,
    simulation_epoch_utc: str,
    sample_times_s: Sequence[int | float],
    ground_ingresses: Sequence[Mapping[str, Any] | Sgp4GroundIngress],
    isl_pairs: Sequence[Sequence[str]],
    host: str = STK_CONNECT_HOST,
    port: int = STK_CONNECT_PORT,
    installed_build: str | None = None,
    configure_application_defaults: bool = False,
    connect_timeout_s: float = 10.0,
    read_timeout_s: float = 120.0,
    client_factory: Callable[..., StkConnectClient] = StkConnectClient,
) -> StkGeometryCapture:
    """Capture independent STK geometry evidence without saving a scenario."""

    bundle = load_sgp4_element_bundle(elements)
    if not isinstance(configure_application_defaults, bool):
        raise TypeError("configure_application_defaults must be a boolean")
    sample_times, step_s = _prepare_sample_grid(sample_times_s)
    epoch = _utc_datetime(simulation_epoch_utc)
    start_s, end_s = sample_times[0], sample_times[-1]
    start = epoch + _representable_timedelta(start_s)
    stop = epoch + _representable_timedelta(end_s)
    ingresses = _prepare_ingresses(ground_ingresses)
    satellite_ids = {element.satellite_id for element in bundle.satellites}
    pairs = _prepare_pairs(isl_pairs, satellite_ids)
    satellite_names = {
        element.satellite_id: f"OrbitSat{index:06d}"
        for index, element in enumerate(bundle.satellites, start=1)
    }
    facility_names = {
        ingress.ingress_id: f"OrbitFac{index:06d}"
        for index, ingress in enumerate(ingresses, start=1)
    }
    scenario_name = _geometry_scenario_name(
        bundle, simulation_epoch_utc, sample_times, ingresses, pairs
    )
    warnings: list[str] = []
    if installed_build is None:
        warnings.append("STK build was not verified by the Connect adapter")
    elif installed_build != STK_11_BASELINE_BUILD:
        raise StkEnvironmentError(
            f"STK build {installed_build!r} does not match "
            f"{STK_11_BASELINE_BUILD!r}"
        )
    if not configure_application_defaults:
        warnings.append(
            "STK propagator and Access application defaults were not changed; "
            "geometry evidence is qualified only in a separately configured instance"
        )

    def new_client() -> StkConnectClient:
        return client_factory(
            host=host,
            port=port,
            connect_timeout_s=connect_timeout_s,
            read_timeout_s=read_timeout_s,
        )

    with new_client() as client:
        created_scenario = False
        creation_outcome_unknown = False
        transport_failed = False
        active_error: BaseException | None = None
        try:
            version_response = client.execute(
                "GetSTKVersion /", response_kind="single"
            )
            stk_version = (version_response.payload or "").strip()
            if stk_version != STK_11_BASELINE_VERSION:
                raise StkEnvironmentError(
                    f"STK version {stk_version!r} does not match "
                    f"{STK_11_BASELINE_VERSION!r}"
                )
            if _scenario_is_open(client):
                raise StkEnvironmentError(
                    "an STK scenario is already open; refusing to replace it"
                )
            if configure_application_defaults:
                _configure_geometry_defaults(client, step_s)
            try:
                client.execute(f"New / Scenario {scenario_name}")
            except StkConnectError:
                creation_outcome_unknown = True
                raise
            created_scenario = True
            client.execute(
                f'SetTimePeriod * "{format_stk_utcg(start)}" '
                f'"{format_stk_utcg(stop)}"'
            )
            for element in bundle.satellites:
                object_name = satellite_names[element.satellite_id]
                client.execute(f"New / */Satellite {object_name}")
                client.execute(
                    build_stk_set_state_command(
                        element, stk_object_name=object_name, step_s=step_s
                    )
                )
            for ingress in ingresses:
                object_name = facility_names[ingress.ingress_id]
                client.execute(f"New / */Facility {object_name}")
                client.execute(
                    f"SetPosition */Facility/{object_name} Geodetic "
                    f"{_format_number(ingress.lat_deg)} "
                    f"{_format_number(ingress.lon_deg)} "
                    f"{_format_number(ingress.altitude_km)}"
                )
                client.execute(
                    f"SetConstraint */Facility/{object_name} "
                    f"ElevationAngle Min {_format_number(ingress.min_elevation_deg)}"
                )

            sgl_windows: list[StkSglWindow] = []
            aer_samples: list[StkAerSample] = []
            for ingress in ingresses:
                facility_name = facility_names[ingress.ingress_id]
                for element in bundle.satellites:
                    satellite_name = satellite_names[element.satellite_id]
                    response = client.execute(
                        _access_command(
                            f"*/Facility/{facility_name}",
                            f"*/Satellite/{satellite_name}",
                            step_s,
                        ),
                        response_kind="single",
                    )
                    access_intervals = parse_stk_access_intervals(
                        response.payload or "",
                        simulation_epoch_utc=simulation_epoch_utc,
                        interval_start_s=start_s,
                        interval_end_s=end_s,
                    )
                    for window_start, window_end in access_intervals:
                        sgl_windows.append(
                            StkSglWindow(
                                ingress.ingress_id,
                                element.satellite_id,
                                window_start,
                                window_end,
                            )
                        )
                    aer_response = client.execute(
                        _aer_report_command(
                            facility_name, satellite_name, step_s
                        ),
                        response_kind="multiple",
                    )
                    pair_aer = parse_stk_aer_report(
                        aer_response.rows,
                        ingress_id=ingress.ingress_id,
                        satellite_id=element.satellite_id,
                        simulation_epoch_utc=simulation_epoch_utc,
                        interval_start_s=start_s,
                        interval_end_s=end_s,
                    )
                    if access_intervals and not pair_aer:
                        raise StkValidationError(
                            "STK AER report is empty for a non-empty Access"
                        )
                    aer_samples.extend(pair_aer)

            isl_windows: list[StkIslWindow] = []
            for endpoint_a, endpoint_b in pairs:
                response = client.execute(
                    _access_command(
                        f"*/Satellite/{satellite_names[endpoint_a]}",
                        f"*/Satellite/{satellite_names[endpoint_b]}",
                        step_s,
                    ),
                    response_kind="single",
                )
                for window_start, window_end in parse_stk_access_intervals(
                    response.payload or "",
                    simulation_epoch_utc=simulation_epoch_utc,
                    interval_start_s=start_s,
                    interval_end_s=end_s,
                ):
                    isl_windows.append(
                        StkIslWindow(
                            endpoint_a, endpoint_b, window_start, window_end
                        )
                    )

            lighting_windows: list[StkLightingWindow] = []
            for element in bundle.satellites:
                response = client.execute(
                    _lighting_report_command(
                        satellite_names[element.satellite_id], step_s
                    ),
                    response_kind="multiple",
                )
                lighting_windows.extend(
                    parse_stk_lighting_report(
                        response.rows,
                        satellite_id=element.satellite_id,
                        simulation_epoch_utc=simulation_epoch_utc,
                        interval_start_s=start_s,
                        interval_end_s=end_s,
                    )
                )
            return StkGeometryCapture(
                element_set_id=bundle.element_set_id,
                simulation_epoch_utc=simulation_epoch_utc,
                sample_times_s=sample_times,
                stk_version=stk_version,
                installed_build=installed_build,
                propagator_defaults_configured=configure_application_defaults,
                access_defaults_configured=configure_application_defaults,
                sgl_windows=tuple(
                    sorted(
                        sgl_windows,
                        key=lambda value: (
                            value.ingress_id,
                            value.satellite_id,
                            value.start_s,
                        ),
                    )
                ),
                isl_windows=tuple(
                    sorted(
                        isl_windows,
                        key=lambda value: (
                            value.endpoint_a,
                            value.endpoint_b,
                            value.start_s,
                        ),
                    )
                ),
                lighting_windows=tuple(
                    sorted(
                        lighting_windows,
                        key=lambda value: (
                            value.satellite_id,
                            value.lighting,
                            value.start_s,
                        ),
                    )
                ),
                aer_samples=tuple(
                    sorted(
                        aer_samples,
                        key=lambda value: (
                            value.time_s,
                            value.ingress_id,
                            value.satellite_id,
                        ),
                    )
                ),
                warnings=tuple(sorted(warnings)),
            )
        except BaseException as exc:
            active_error = exc
            transport_failed = isinstance(exc, StkConnectError)
            raise
        finally:
            if created_scenario and not transport_failed:
                try:
                    client.execute("Unload / *")
                except Exception as cleanup_error:
                    if active_error is None:
                        raise StkValidationError(
                            "STK temporary geometry scenario cleanup failed"
                        ) from cleanup_error
                    active_error.add_note(
                        "STK temporary geometry scenario cleanup also failed: "
                        f"{cleanup_error}"
                    )
            elif created_scenario or creation_outcome_unknown:
                client.close()
                try:
                    _recover_temporary_scenario(new_client)
                except Exception as cleanup_error:
                    if active_error is None:
                        raise StkValidationError(
                            "STK temporary geometry scenario recovery failed"
                        ) from cleanup_error
                    active_error.add_note(
                        "STK temporary geometry scenario recovery also failed: "
                        f"{cleanup_error}"
                    )


def _configure_geometry_defaults(client: StkConnectClient, step_s: float) -> None:
    client.execute(
        "SetPropagatorDefaults / SGP4 Module SGP4 UpdateMode None "
        "ValidateEphemeris On Use1PtInterpolation Off Output1PtWarning On"
    )
    client.execute("AccessConfig / UseLightTimeDelay Off")
    client.execute("AccessConfig / AberrationType None")
    client.execute(
        f"AccessConfig / TimeConvergence "
        f"{_format_number(STK_ACCESS_TIME_CONVERGENCE_S)}"
    )
    client.execute(
        f"AccessConfig / MaxSamplingStepSize {_format_number(step_s)}"
    )


def _access_command(from_path: str, to_path: str, step_s: float) -> str:
    return (
        f"Access {from_path} {to_path} TimePeriod UseScenarioInterval "
        f"MaxStepSize {_format_number(step_s)} "
        f"TimeConvergence {_format_number(STK_ACCESS_TIME_CONVERGENCE_S)} "
        "AberrationType None"
    )


def _aer_report_command(
    facility_name: str, satellite_name: str, step_s: float
) -> str:
    return (
        f'Report_RM */Facility/{facility_name} Style "{STK_AER_REPORT_STYLE}" '
        f"AccessObject Satellite/{satellite_name} TimePeriod UseAccessTimes "
        f"TimeStep Bound {_format_number(step_s)}"
    )


def _lighting_report_command(satellite_name: str, step_s: float) -> str:
    return (
        f'Report_RM */Satellite/{satellite_name} '
        f'Style "{STK_LIGHTING_REPORT_STYLE}" '
        f"TimePeriod UseScenarioInterval TimeStep {_format_number(step_s)} "
        "AllLines On"
    )


def _prepare_ingresses(
    values: Sequence[Mapping[str, Any] | Sgp4GroundIngress],
) -> tuple[Sgp4GroundIngress, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("ground_ingresses must be a sequence")
    result = tuple(
        sorted(
            (Sgp4GroundIngress.from_value(value) for value in values),
            key=lambda value: value.ingress_id,
        )
    )
    identifiers = [value.ingress_id for value in result]
    if not result or len(identifiers) != len(set(identifiers)):
        raise ValueError("ground_ingresses must be non-empty and unique")
    return result


def _prepare_pairs(
    values: Sequence[Sequence[str]], satellite_ids: set[str]
) -> tuple[tuple[str, str], ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise TypeError("isl_pairs must be a sequence")
    result: list[tuple[str, str]] = []
    for value in values:
        if (
            isinstance(value, (str, bytes, bytearray))
            or not isinstance(value, Sequence)
            or len(value) != 2
        ):
            raise ValueError("each ISL pair must contain two identifiers")
        first, second = value
        if (
            not isinstance(first, str)
            or not isinstance(second, str)
            or first == second
            or first not in satellite_ids
            or second not in satellite_ids
        ):
            raise ValueError("each ISL pair must name two distinct satellites")
        result.append(tuple(sorted((first, second))))
    if not result or len(result) != len(set(result)):
        raise ValueError("isl_pairs must be non-empty and unique")
    return tuple(sorted(result))


def _geometry_scenario_name(
    bundle: Sgp4ElementBundle,
    simulation_epoch_utc: str,
    sample_times_s: tuple[float, ...],
    ingresses: tuple[Sgp4GroundIngress, ...],
    pairs: tuple[tuple[str, str], ...],
) -> str:
    source = json.dumps(
        {
            "element_set_id": bundle.element_set_id,
            "ground_ingresses": [value.to_dict() for value in ingresses],
            "isl_pairs": pairs,
            "sample_times_s": sample_times_s,
            "simulation_epoch_utc": simulation_epoch_utc,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "OrbitGeo" + hashlib.sha256(source).hexdigest()[:16]


def _clip_intervals(
    values: tuple[tuple[float, float], ...],
    interval_start_s: float,
    interval_end_s: float,
) -> tuple[tuple[float, float], ...]:
    start_limit = _finite(interval_start_s, "interval_start_s")
    end_limit = _finite(interval_end_s, "interval_end_s")
    if end_limit <= start_limit:
        raise ValueError("interval_end_s must be greater than interval_start_s")
    result: list[tuple[float, float]] = []
    for start, stop in values:
        first = max(_finite(start, "interval start"), start_limit)
        second = min(_finite(stop, "interval stop"), end_limit)
        if second <= first:
            continue
        if result and first < result[-1][1] - 0.001:
            raise StkValidationError("STK event intervals overlap or are unordered")
        result.append((_q(first), _q(second)))
    return tuple(result)


def _offset_seconds(instant: datetime, epoch: datetime) -> float:
    return _q((instant - epoch).total_seconds())


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _q(value: float) -> float:
    result = round(float(value), 9)
    return 0.0 if result == 0.0 else result


def _normalise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical STK geometry evidence contains non-finite data")
        return 0.0 if value == 0.0 else value
    return value


__all__ = [
    "STK_ACCESS_TIME_CONVERGENCE_S",
    "STK_AER_REPORT_STYLE",
    "STK_GEOMETRY_CAPTURE_SCHEMA",
    "STK_LIGHTING_REPORT_STYLE",
    "StkAerSample",
    "StkGeometryCapture",
    "StkIslWindow",
    "StkLightingWindow",
    "StkSglWindow",
    "capture_stk_geometry",
    "parse_stk_access_intervals",
    "parse_stk_aer_report",
    "parse_stk_lighting_report",
]
