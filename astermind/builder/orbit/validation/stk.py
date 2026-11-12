"""Optional STK 11.6 validation support for the SGP4 Orbit implementation.

This module is deliberately isolated from the production Orbit Trace path. It
uses only the Python standard library and an already running, qualified STK
Desktop or Engine instance with STK Connect enabled.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import socket
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Mapping, Protocol, Sequence

from astermind.builder.orbit.sgp4.elements import Sgp4ElementBundle, Sgp4SatelliteElements, load_sgp4_element_bundle

STK_VALIDATION_CONTRACT_ID = 'orbitserve.orbit.stk_validation'
STK_11_BASELINE_VERSION = "STK v11.6.0"
STK_11_BASELINE_BUILD = "11.6.0.453"
STK_CONNECT_HOST = "127.0.0.1"
STK_CONNECT_PORT = 5001
STK_CONNECT_TRANSPORT = "stk-connect-tcp"
STK_PROPAGATOR_MODULE = "SGP4"
STK_UPDATE_MODE = "None"
STK_TEME_REPORT_STYLE = "TEMEofDate Position Velocity"
STK_FIXED_REPORT_STYLE = "Fixed Position Velocity"

_ACK = "ACK"
_NACK = "NACK"
_HEADER_LENGTH = 40
_MAX_PAYLOAD_BYTES = 64 * 1024 * 1024
_MAX_REPORT_ROWS = 10_000_000
_STK_OBJECT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_STATE_HEADER = (
    "Time (UTCG)",
    "x (km)",
    "y (km)",
    "z (km)",
    "vx (km/sec)",
    "vy (km/sec)",
    "vz (km/sec)",
)
_MONTH_ABBREVIATIONS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
_MONTH_NUMBERS = {
    abbreviation: number
    for number, abbreviation in enumerate(_MONTH_ABBREVIATIONS, start=1)
}
_STK_UTCG = re.compile(
    r"^(?P<day>\d{1,2}) "
    r"(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) "
    r"(?P<year>\d{4}) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})"
    r"\.(?P<fraction>\d{1,6})$"
)


class StkValidationError(RuntimeError):
    """Base error for STK validation failures."""


class StkConnectError(StkValidationError):
    """STK Connect transport or framing failure."""


class StkCommandError(StkValidationError):
    """STK rejected a Connect command."""

    def __init__(self, command: str) -> None:
        super().__init__(f"STK returned NACK for command: {command}")
        self.command = command


class StkEnvironmentError(StkValidationError):
    """The active STK environment does not match the frozen baseline."""


class StkMappingError(ValueError):
    """An Orbit value cannot be represented by the STK 11.6 direct lane."""


@dataclass(frozen=True, slots=True)
class StkConnectResponse:
    """One framed STK Connect response."""

    ack: Literal["ACK", "NACK"]
    return_type: str | None = None
    payload: str | None = None
    rows: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StkEnvironmentProbe:
    """Read-only qualification information returned by STK Connect."""

    stk_version: str
    expected_version: str
    installed_build: str | None
    expected_build: str
    transport: str
    scenario_open: bool

    @property
    def version_matches(self) -> bool:
        return self.stk_version == self.expected_version

    @property
    def build_verified(self) -> bool:
        return self.installed_build is not None

    @property
    def build_matches(self) -> bool:
        return self.installed_build == self.expected_build

    @property
    def baseline_matches(self) -> bool:
        return self.version_matches and self.build_matches

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_matches": self.baseline_matches,
            "build_matches": self.build_matches,
            "build_verified": self.build_verified,
            "expected_build": self.expected_build,
            "expected_version": self.expected_version,
            "installed_build": self.installed_build,
            "scenario_open": self.scenario_open,
            "stk_version": self.stk_version,
            "transport": self.transport,
            "version_matches": self.version_matches,
        }


@dataclass(frozen=True, slots=True)
class StkStateSample:
    """One STK state-vector sample associated with an Orbit satellite."""

    satellite_id: str
    time_s: float
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not isinstance(self.satellite_id, str) or not self.satellite_id:
            raise ValueError("satellite_id must be a non-empty string")
        object.__setattr__(self, "time_s", _finite_float(self.time_s, "time_s"))
        object.__setattr__(
            self,
            "position_km",
            _finite_vector(self.position_km, "position_km"),
        )
        object.__setattr__(
            self,
            "velocity_km_s",
            _finite_vector(self.velocity_km_s, "velocity_km_s"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_km": list(self.position_km),
            "satellite_id": self.satellite_id,
            "time_s": self.time_s,
            "velocity_km_s": list(self.velocity_km_s),
        }


@dataclass(frozen=True, slots=True)
class StkDirectStateCapture:
    """Canonical diagnostic evidence from the STK direct-element lane."""

    element_set_id: str
    simulation_epoch_utc: str
    sample_times_s: tuple[float, ...]
    stk_version: str
    installed_build: str | None
    propagator_defaults_configured: bool
    teme_samples: tuple[StkStateSample, ...]
    fixed_samples: tuple[StkStateSample, ...]
    warnings: tuple[str, ...]
    contract_id: str = STK_VALIDATION_CONTRACT_ID
    transport: str = STK_CONNECT_TRANSPORT
    propagator_module: str = STK_PROPAGATOR_MODULE
    update_mode: str = STK_UPDATE_MODE
    teme_report_style: str = STK_TEME_REPORT_STYLE
    fixed_report_style: str = STK_FIXED_REPORT_STYLE
    time_scale: str = "UTC"
    position_unit: str = "km"
    velocity_unit: str = "km/s"
    formal_thresholds_frozen: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "element_set_id": self.element_set_id,
            "fixed_report_style": self.fixed_report_style,
            "fixed_samples": [sample.to_dict() for sample in self.fixed_samples],
            "formal_thresholds_frozen": self.formal_thresholds_frozen,
            "installed_build": self.installed_build,
            "position_unit": self.position_unit,
            "propagator_module": self.propagator_module,
            "propagator_defaults_configured": self.propagator_defaults_configured,
            "sample_times_s": list(self.sample_times_s),
            "simulation_epoch_utc": self.simulation_epoch_utc,
            "stk_version": self.stk_version,
            "teme_report_style": self.teme_report_style,
            "teme_samples": [sample.to_dict() for sample in self.teme_samples],
            "time_scale": self.time_scale,
            "transport": self.transport,
            "update_mode": self.update_mode,
            "velocity_unit": self.velocity_unit,
            "warnings": list(self.warnings),
        }

    def canonical_bytes(self) -> bytes:
        normalized = _normalise_json(self.to_dict())
        return (
            json.dumps(
                normalized,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )


class _SocketLike(Protocol):
    def recv(self, count: int) -> bytes: ...

    def sendall(self, data: bytes) -> None: ...

    def settimeout(self, value: float | None) -> None: ...

    def close(self) -> None: ...


_ConnectionFactory = Callable[[tuple[str, int], float], _SocketLike]


class StkConnectClient:
    """Small synchronous client for the STK 11.6 Connect framing protocol."""

    def __init__(
        self,
        *,
        host: str = STK_CONNECT_HOST,
        port: int = STK_CONNECT_PORT,
        connect_timeout_s: float = 10.0,
        read_timeout_s: float = 60.0,
        connection_factory: _ConnectionFactory | None = None,
    ) -> None:
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("port must be an integer in [1, 65535]")
        self.host = host
        self.port = port
        self.connect_timeout_s = _positive_float(connect_timeout_s, "connect_timeout_s")
        self.read_timeout_s = _positive_float(read_timeout_s, "read_timeout_s")
        self._connection_factory = connection_factory or socket.create_connection
        self._socket: _SocketLike | None = None

    def __enter__(self) -> StkConnectClient:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        if self.connected:
            raise StkConnectError("STK Connect client is already connected")
        try:
            connection = self._connection_factory(
                (self.host, self.port),
                self.connect_timeout_s,
            )
            connection.settimeout(self.read_timeout_s)
            self._socket = connection
            self._write_line("ConControl / AckOn")
            if self._read_ack() != _ACK:
                raise StkConnectError("STK rejected acknowledgement mode")
            self._write_line("ConControl / AsyncOff")
            if self._read_ack() != _ACK:
                raise StkConnectError("STK rejected synchronous mode")
        except StkValidationError:
            self.close()
            raise
        except (OSError, UnicodeError) as exc:
            self.close()
            raise StkConnectError(
                f"unable to connect to STK at {self.host}:{self.port}"
            ) from exc

    def close(self) -> None:
        connection = self._socket
        self._socket = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass

    def execute(
        self,
        command: str,
        *,
        response_kind: Literal["none", "single", "multiple"] = "none",
        allow_nack: bool = False,
    ) -> StkConnectResponse:
        if response_kind not in {"none", "single", "multiple"}:
            raise ValueError("response_kind must be none, single, or multiple")
        self._write_line(command)
        ack = self._read_ack()
        if ack == _NACK:
            if allow_nack:
                return StkConnectResponse(ack=_NACK)
            raise StkCommandError(command)
        if response_kind == "none":
            return StkConnectResponse(ack=_ACK)
        if response_kind == "single":
            return_type, length = self._read_return_header()
            payload = self._read_exact(length).decode("ascii")
            return StkConnectResponse(
                ack=_ACK,
                return_type=return_type,
                payload=payload,
            )

        return_type, count_length = self._read_return_header()
        count_text = self._read_exact(count_length).decode("ascii").strip()
        try:
            row_count = int(count_text)
        except ValueError as exc:
            raise StkConnectError(
                f"invalid STK multiple-response row count: {count_text!r}"
            ) from exc
        if not 0 <= row_count <= _MAX_REPORT_ROWS:
            raise StkConnectError(f"STK row count is out of bounds: {row_count}")
        rows: list[str] = []
        for _ in range(row_count):
            _, row_length = self._read_return_header()
            rows.append(self._read_exact(row_length).decode("ascii"))
        return StkConnectResponse(
            ack=_ACK,
            return_type=return_type,
            rows=tuple(rows),
        )

    def _write_line(self, command: str) -> None:
        connection = self._require_connection()
        if not isinstance(command, str) or not command:
            raise ValueError("STK command must be a non-empty string")
        if "\r" in command or "\n" in command:
            raise ValueError("STK command must contain exactly one logical line")
        try:
            connection.sendall(command.encode("ascii") + b"\n")
        except (OSError, UnicodeError) as exc:
            raise StkConnectError("unable to write STK Connect command") from exc

    def _read_ack(self) -> Literal["ACK", "NACK"]:
        first = self._read_exact(1)
        remaining = 3 if first == b"N" else 2
        value = (first + self._read_exact(remaining)).decode("ascii")
        if value not in {_ACK, _NACK}:
            raise StkConnectError(f"invalid STK acknowledgement: {value!r}")
        return value  # type: ignore[return-value]

    def _read_return_header(self) -> tuple[str, int]:
        header = self._read_exact(_HEADER_LENGTH).decode("ascii")
        match = re.match(r"^(\S+)\s+(\d+)", header)
        if match is None:
            raise StkConnectError(f"invalid STK return header: {header!r}")
        length = int(match.group(2))
        if not 0 <= length <= _MAX_PAYLOAD_BYTES:
            raise StkConnectError(f"STK payload length is out of bounds: {length}")
        return match.group(1), length

    def _read_exact(self, count: int) -> bytes:
        connection = self._require_connection()
        output = bytearray()
        try:
            while len(output) < count:
                chunk = connection.recv(count - len(output))
                if not chunk:
                    raise StkConnectError(
                        f"STK closed the connection after {len(output)} of {count} bytes"
                    )
                output.extend(chunk)
        except StkValidationError:
            raise
        except OSError as exc:
            raise StkConnectError(
                f"unable to read {count} bytes from STK Connect"
            ) from exc
        return bytes(output)

    def _require_connection(self) -> _SocketLike:
        if self._socket is None:
            raise StkConnectError("STK Connect client is not connected")
        return self._socket


def probe_stk_11_connect(
    *,
    host: str = STK_CONNECT_HOST,
    port: int = STK_CONNECT_PORT,
    installed_build: str | None = None,
    connect_timeout_s: float = 10.0,
    read_timeout_s: float = 60.0,
    client_factory: Callable[..., StkConnectClient] = StkConnectClient,
) -> StkEnvironmentProbe:
    """Read STK version and scenario state without modifying the application."""

    with client_factory(
        host=host,
        port=port,
        connect_timeout_s=connect_timeout_s,
        read_timeout_s=read_timeout_s,
    ) as client:
        version_response = client.execute(
            "GetSTKVersion /",
            response_kind="single",
        )
        scenario_open = _scenario_is_open(client)
    version = (version_response.payload or "").strip()
    return StkEnvironmentProbe(
        stk_version=version,
        expected_version=STK_11_BASELINE_VERSION,
        installed_build=installed_build,
        expected_build=STK_11_BASELINE_BUILD,
        transport=STK_CONNECT_TRANSPORT,
        scenario_open=scenario_open,
    )


def build_stk_set_state_command(
    element: Sgp4SatelliteElements,
    *,
    stk_object_name: str,
    step_s: float,
) -> str:
    """Map one validated Orbit element record to STK 11.6 direct elements."""

    if not isinstance(element, Sgp4SatelliteElements):
        raise TypeError("element must be Sgp4SatelliteElements")
    if not _STK_OBJECT_NAME.fullmatch(stk_object_name):
        raise StkMappingError(
            "stk_object_name must match [A-Za-z][A-Za-z0-9_]{0,63}"
        )
    if element.satnum > 99999:
        raise StkMappingError(
            "STK 11.6 direct SSCNumber cannot represent satnum above 99999"
        )
    step = _positive_float(step_s, "step_s")
    values = (
        element.mean_motion_rev_per_day,
        element.eccentricity,
        element.inclination_deg,
        element.argument_of_perigee_deg,
        element.raan_deg,
        element.mean_anomaly_deg,
        element.bstar_inv_earth_radii,
    )
    parts = [
        f"SetState */Satellite/{stk_object_name}",
        "SGP4",
        "UseScenarioInterval",
        _format_number(step),
        str(element.satnum),
        "TLESource DefineElements Source Elements",
        f'OrbitEpoch "{format_stk_utcg(element.epoch_utc)}"',
        "ElementSet " + " ".join(_format_number(value) for value in values),
        f"RevNumber {element.revolution_number_at_epoch}",
        f"Classification {element.classification}",
    ]
    international_designator = _stk_international_designator(element.object_id)
    if international_designator is not None:
        parts.append(f"IntlDesignator {international_designator}")
    parts.extend(
        (
            "MeanMotionDot "
            + _format_number(element.mean_motion_dot_rev_per_day2),
            "MeanMotionDotDot "
            + _format_number(element.mean_motion_ddot_rev_per_day3),
        )
    )
    return " ".join(parts)


def format_stk_utcg(value: str | datetime) -> str:
    """Format a canonical UTC instant for locale-independent STK Connect use."""

    instant = _utc_datetime(value)
    return (
        f"{instant.day} {_MONTH_ABBREVIATIONS[instant.month - 1]} "
        f"{instant.year:04d} "
        f"{instant:%H:%M:%S}.{instant.microsecond:06d}"
    )


def parse_stk_state_report(
    rows: Iterable[str],
    *,
    satellite_id: str,
    simulation_epoch_utc: str,
    expected_sample_times_s: Sequence[int | float],
) -> tuple[StkStateSample, ...]:
    """Parse and strictly align one STK position/velocity report."""

    expected_times, _ = _prepare_sample_grid(expected_sample_times_s)
    epoch = _utc_datetime(simulation_epoch_utc)
    parsed_rows: list[list[str]] = []
    for line in rows:
        if not isinstance(line, str):
            raise TypeError("STK report rows must be strings")
        if not line.strip():
            continue
        parsed = next(csv.reader(io.StringIO(line)))
        parsed_rows.append([field.strip() for field in parsed])
    if not parsed_rows:
        raise StkValidationError("STK state report is empty")
    if tuple(parsed_rows[0]) != _STATE_HEADER:
        raise StkValidationError(
            f"unexpected STK state report header: {tuple(parsed_rows[0])!r}"
        )
    data_rows = parsed_rows[1:]
    if len(data_rows) != len(expected_times):
        raise StkValidationError(
            "STK state report sample count does not match the Orbit grid"
        )

    samples: list[StkStateSample] = []
    for index, (row, expected_time) in enumerate(zip(data_rows, expected_times)):
        if len(row) != len(_STATE_HEADER):
            raise StkValidationError(
                f"STK state report row {index} has {len(row)} fields"
            )
        reported_time = _parse_stk_utcg(row[0])
        expected_instant = epoch + _representable_timedelta(expected_time)
        if abs((reported_time - expected_instant).total_seconds()) > 0.000500001:
            raise StkValidationError(
                f"STK state report time does not match sample {expected_time}"
            )
        numeric = tuple(
            _parse_finite_number(value, f"STK state row {index}") for value in row[1:]
        )
        samples.append(
            StkStateSample(
                satellite_id=satellite_id,
                time_s=expected_time,
                position_km=numeric[:3],
                velocity_km_s=numeric[3:],
            )
        )
    return tuple(samples)


def capture_stk_direct_states(
    elements: Sgp4ElementBundle | Mapping[str, Any] | str | Path,
    *,
    simulation_epoch_utc: str,
    sample_times_s: Sequence[int | float],
    host: str = STK_CONNECT_HOST,
    port: int = STK_CONNECT_PORT,
    installed_build: str | None = None,
    configure_application_defaults: bool = False,
    connect_timeout_s: float = 10.0,
    read_timeout_s: float = 120.0,
    client_factory: Callable[..., StkConnectClient] = StkConnectClient,
) -> StkDirectStateCapture:
    """Capture TEME and Fixed state evidence from the STK direct-element lane.

    The function refuses to replace an existing scenario and always attempts to
    unload the temporary scenario it created. Setting application defaults is
    intended only for a dedicated STK validation instance.
    """

    bundle = load_sgp4_element_bundle(elements)
    if not isinstance(configure_application_defaults, bool):
        raise TypeError("configure_application_defaults must be a boolean")
    connect_timeout = _positive_float(connect_timeout_s, "connect_timeout_s")
    read_timeout = _positive_float(read_timeout_s, "read_timeout_s")
    sample_times, step_s = _prepare_sample_grid(sample_times_s)
    epoch = _utc_datetime(simulation_epoch_utc)
    start = epoch + _representable_timedelta(sample_times[0])
    stop = epoch + _representable_timedelta(sample_times[-1])
    object_names = {
        element.satellite_id: f"OrbitSat{index:06d}"
        for index, element in enumerate(bundle.satellites, start=1)
    }
    scenario_name = _scenario_name(bundle, simulation_epoch_utc, sample_times)
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
            "STK application defaults were not changed; run only in a separately "
            "qualified instance"
        )

    def new_client() -> StkConnectClient:
        return client_factory(
            host=host,
            port=port,
            connect_timeout_s=connect_timeout,
            read_timeout_s=read_timeout,
        )

    with new_client() as client:
        created_scenario = False
        creation_outcome_unknown = False
        transport_failed = False
        active_error: BaseException | None = None
        try:
            version_response = client.execute(
                "GetSTKVersion /",
                response_kind="single",
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
                client.execute(
                    "SetPropagatorDefaults / SGP4 Module SGP4 UpdateMode None "
                    "ValidateEphemeris On Use1PtInterpolation Off "
                    "Output1PtWarning On"
                )
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
                object_name = object_names[element.satellite_id]
                client.execute(f"New / */Satellite {object_name}")
                client.execute(
                    build_stk_set_state_command(
                        element,
                        stk_object_name=object_name,
                        step_s=step_s,
                    )
                )

            teme_samples: list[StkStateSample] = []
            fixed_samples: list[StkStateSample] = []
            for element in bundle.satellites:
                object_name = object_names[element.satellite_id]
                teme_response = client.execute(
                    _state_report_command(
                        object_name,
                        STK_TEME_REPORT_STYLE,
                        step_s,
                    ),
                    response_kind="multiple",
                )
                fixed_response = client.execute(
                    _state_report_command(
                        object_name,
                        STK_FIXED_REPORT_STYLE,
                        step_s,
                    ),
                    response_kind="multiple",
                )
                teme_samples.extend(
                    parse_stk_state_report(
                        teme_response.rows,
                        satellite_id=element.satellite_id,
                        simulation_epoch_utc=simulation_epoch_utc,
                        expected_sample_times_s=sample_times,
                    )
                )
                fixed_samples.extend(
                    parse_stk_state_report(
                        fixed_response.rows,
                        satellite_id=element.satellite_id,
                        simulation_epoch_utc=simulation_epoch_utc,
                        expected_sample_times_s=sample_times,
                    )
                )

            ordered_teme = tuple(
                sorted(teme_samples, key=lambda value: (value.time_s, value.satellite_id))
            )
            ordered_fixed = tuple(
                sorted(fixed_samples, key=lambda value: (value.time_s, value.satellite_id))
            )
            return StkDirectStateCapture(
                element_set_id=bundle.element_set_id,
                simulation_epoch_utc=simulation_epoch_utc,
                sample_times_s=sample_times,
                stk_version=stk_version,
                installed_build=installed_build,
                propagator_defaults_configured=configure_application_defaults,
                teme_samples=ordered_teme,
                fixed_samples=ordered_fixed,
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
                            "STK temporary scenario cleanup failed"
                        ) from cleanup_error
                    active_error.add_note(
                        "STK temporary scenario cleanup also failed: "
                        f"{cleanup_error}"
                    )
            elif created_scenario or creation_outcome_unknown:
                client.close()
                try:
                    _recover_temporary_scenario(new_client)
                except Exception as cleanup_error:
                    if active_error is None:
                        raise StkValidationError(
                            "STK temporary scenario recovery cleanup failed"
                        ) from cleanup_error
                    active_error.add_note(
                        "STK temporary scenario recovery cleanup also failed: "
                        f"{cleanup_error}"
                    )


def _scenario_is_open(client: StkConnectClient) -> bool:
    response = client.execute(
        "CheckScenario /",
        response_kind="single",
    )
    value = (response.payload or "").strip()
    if value not in {"0", "1"}:
        raise StkConnectError(
            f"unexpected CheckScenario response: {response.payload!r}"
        )
    return value == "1"


def _recover_temporary_scenario(
    client_factory: Callable[[], StkConnectClient],
) -> None:
    with client_factory() as recovery_client:
        if _scenario_is_open(recovery_client):
            recovery_client.execute("Unload / *")


def _state_report_command(object_name: str, style: str, step_s: float) -> str:
    return (
        f'Report_RM */Satellite/{object_name} Style "{style}" '
        f"TimePeriod UseScenarioInterval TimeStep {_format_number(step_s)}"
    )


def _prepare_sample_grid(
    values: Sequence[int | float],
) -> tuple[tuple[float, ...], float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("sample_times_s must be a sequence")
    times = tuple(_finite_float(value, "sample_times_s") for value in values)
    if len(times) < 2:
        raise ValueError("STK validation requires at least two sample times")
    if times[0] < 0.0:
        raise ValueError("sample_times_s must be non-negative")
    deltas = tuple(second - first for first, second in zip(times, times[1:]))
    if any(delta <= 0.0 for delta in deltas):
        raise ValueError("sample_times_s must be strictly increasing")
    step = deltas[0]
    tolerance = max(1e-12 * abs(step), 1e-9)
    if any(abs(delta - step) > tolerance for delta in deltas[1:]):
        raise ValueError("sample_times_s must use a uniform step")
    for value in times:
        _representable_timedelta(value)
    return times, step


def _utc_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        instant = value
        if instant.tzinfo is None or instant.utcoffset() != timedelta(0):
            raise ValueError("datetime must be timezone-aware UTC")
        return instant.astimezone(timezone.utc)
    if not isinstance(value, str):
        raise TypeError("UTC instant must be canonical text or datetime")
    try:
        instant = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise ValueError(
            "UTC instant must use YYYY-MM-DDTHH:MM:SS.ffffffZ"
        ) from exc
    return instant.replace(tzinfo=timezone.utc)


def _parse_stk_utcg(value: str) -> datetime:
    match = _STK_UTCG.fullmatch(value)
    if match is None:
        raise StkValidationError(f"invalid STK UTCG timestamp: {value!r}")
    try:
        fraction = match.group("fraction").ljust(6, "0")
        instant = datetime(
            int(match.group("year")),
            _MONTH_NUMBERS[match.group("month")],
            int(match.group("day")),
            int(match.group("hour")),
            int(match.group("minute")),
            int(match.group("second")),
            int(fraction),
            tzinfo=timezone.utc,
        )
    except ValueError as exc:
        raise StkValidationError(f"invalid STK UTCG timestamp: {value!r}") from exc
    return instant


def _representable_timedelta(value: float) -> timedelta:
    microseconds = value * 1_000_000.0
    rounded = round(microseconds)
    if abs(microseconds - rounded) > 1e-6:
        raise StkMappingError(
            f"sample time {value!r} is not exactly representable in STK UTC input"
        )
    return timedelta(microseconds=rounded)


def _scenario_name(
    bundle: Sgp4ElementBundle,
    simulation_epoch_utc: str,
    sample_times_s: tuple[float, ...],
) -> str:
    identity = (
        bundle.element_set_id
        + "\n"
        + simulation_epoch_utc
        + "\n"
        + ",".join(_format_number(value) for value in sample_times_s)
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"OrbitValidation_{digest}"


def _stk_international_designator(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.fullmatch(r"(\d{4})-(\d{3})([A-Z]{1,3})", value)
    if match is None:
        raise StkMappingError(f"object_id is not representable by STK: {value!r}")
    return match.group(1)[-2:] + match.group(2) + match.group(3)


def _format_number(value: int | float) -> str:
    number = _finite_float(value, "STK numeric value")
    if number == 0.0:
        return "0"
    return format(number, ".17g")


def _parse_finite_number(value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise StkValidationError(f"{label} contains a non-numeric value") from exc
    if not math.isfinite(number):
        raise StkValidationError(f"{label} contains a non-finite value")
    return 0.0 if number == 0.0 else number


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return 0.0 if number == 0.0 else number


def _positive_float(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number <= 0.0:
        raise ValueError(f"{label} must be strictly positive")
    return number


def _finite_vector(value: Any, label: str) -> tuple[float, float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be a three-value sequence")
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly three values")
    return tuple(_finite_float(item, label) for item in value)  # type: ignore[return-value]


def _normalise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalise_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalise_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical STK evidence contains a non-finite float")
        if value == 0.0:
            return 0.0
        return float(format(value, ".15g"))
    return value


__all__ = [
    "STK_11_BASELINE_BUILD",
    "STK_11_BASELINE_VERSION",
    "STK_CONNECT_HOST",
    "STK_CONNECT_PORT",
    "STK_CONNECT_TRANSPORT",
    "STK_FIXED_REPORT_STYLE",
    "STK_PROPAGATOR_MODULE",
    "STK_TEME_REPORT_STYLE",
    "STK_UPDATE_MODE",
    "STK_VALIDATION_CONTRACT_ID",
    "StkCommandError",
    "StkConnectClient",
    "StkConnectError",
    "StkConnectResponse",
    "StkDirectStateCapture",
    "StkEnvironmentError",
    "StkEnvironmentProbe",
    "StkMappingError",
    "StkStateSample",
    "StkValidationError",
    "build_stk_set_state_command",
    "capture_stk_direct_states",
    "format_stk_utcg",
    "parse_stk_state_report",
    "probe_stk_11_connect",
]
