"""Deterministic state-vector comparison metrics for STK validation."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from ..sgp4.frames import Sgp4EarthFixedResult, Sgp4ItrsSample
from ..sgp4.propagation import (
    Sgp4PropagationResult,
    Sgp4TemeSample,
)
from .stk import (
    STK_VALIDATION_CONTRACT_ID,
    StkDirectStateCapture,
    StkStateSample,
    StkValidationError,
)

STK_STATE_COMPARISON_SCHEMA = "orbitserve.orbit.stk-state-comparison"
_P95_METHOD = "type-7-linear"
_DIFFERENCE_SIGN = "orbit_minus_stk"
_RIC_BASIS = "stk_reference_state"


@dataclass(frozen=True, slots=True)
class StkStateDifference:
    """One Orbit-minus-STK state difference on the common sample grid."""

    satellite_id: str
    time_s: float
    position_difference_m: tuple[float, float, float]
    velocity_difference_m_s: tuple[float, float, float]
    position_error_m: float
    velocity_error_m_s: float
    radial_position_error_m: float
    in_track_position_error_m: float
    cross_track_position_error_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cross_track_position_error_m": self.cross_track_position_error_m,
            "in_track_position_error_m": self.in_track_position_error_m,
            "position_difference_m": list(self.position_difference_m),
            "position_error_m": self.position_error_m,
            "radial_position_error_m": self.radial_position_error_m,
            "satellite_id": self.satellite_id,
            "time_s": self.time_s,
            "velocity_difference_m_s": list(self.velocity_difference_m_s),
            "velocity_error_m_s": self.velocity_error_m_s,
        }


@dataclass(frozen=True, slots=True)
class StkMetricSummary:
    """Deterministic distribution summary for one non-negative error metric."""

    metric: str
    unit: str
    rms: float
    p95: float
    maximum: float
    worst_satellite_id: str
    worst_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum": self.maximum,
            "metric": self.metric,
            "p95": self.p95,
            "rms": self.rms,
            "unit": self.unit,
            "worst_satellite_id": self.worst_satellite_id,
            "worst_time_s": self.worst_time_s,
        }


@dataclass(frozen=True, slots=True)
class StkFrameComparison:
    """Per-sample and aggregate evidence for one explicit frame pairing."""

    stk_reference_frame: str
    orbit_reference_frame: str
    samples: tuple[StkStateDifference, ...]
    metrics: tuple[StkMetricSummary, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": [metric.to_dict() for metric in self.metrics],
            "orbit_reference_frame": self.orbit_reference_frame,
            "samples": [sample.to_dict() for sample in self.samples],
            "stk_reference_frame": self.stk_reference_frame,
        }


@dataclass(frozen=True, slots=True)
class StkStateComparisonReport:
    """Canonical diagnostic metrics for TEME and Fixed/ITRS comparisons."""

    element_set_id: str
    simulation_epoch_utc: str
    sample_times_s: tuple[float, ...]
    teme: StkFrameComparison
    fixed: StkFrameComparison
    schema: str = STK_STATE_COMPARISON_SCHEMA
    contract_id: str = STK_VALIDATION_CONTRACT_ID
    difference_sign: str = _DIFFERENCE_SIGN
    ric_basis: str = _RIC_BASIS
    p95_method: str = _P95_METHOD
    formal_thresholds_frozen: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "difference_sign": self.difference_sign,
            "element_set_id": self.element_set_id,
            "fixed": self.fixed.to_dict(),
            "formal_thresholds_frozen": self.formal_thresholds_frozen,
            "p95_method": self.p95_method,
            "ric_basis": self.ric_basis,
            "sample_times_s": list(self.sample_times_s),
            "schema": self.schema,
            "simulation_epoch_utc": self.simulation_epoch_utc,
            "teme": self.teme.to_dict(),
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


def compare_stk_state_capture(
    capture: StkDirectStateCapture,
    orbit_teme: Sgp4PropagationResult,
    orbit_fixed: Sgp4EarthFixedResult,
) -> StkStateComparisonReport:
    """Compare one STK capture with matching Orbit TEME and ITRS results.

    Inputs must already share the same element identity, epoch, canonical sample
    order, satellite identities, and times. No interpolation or reordering is
    performed.
    """

    if not isinstance(capture, StkDirectStateCapture):
        raise TypeError("capture must be StkDirectStateCapture")
    if not isinstance(orbit_teme, Sgp4PropagationResult):
        raise TypeError("orbit_teme must be Sgp4PropagationResult")
    if not isinstance(orbit_fixed, Sgp4EarthFixedResult):
        raise TypeError("orbit_fixed must be Sgp4EarthFixedResult")
    _require_equal(
        "element_set_id",
        capture.element_set_id,
        orbit_teme.element_set_id,
        orbit_fixed.element_set_id,
    )
    _require_equal(
        "simulation_epoch_utc",
        capture.simulation_epoch_utc,
        orbit_teme.simulation_epoch_utc,
        orbit_fixed.simulation_epoch_utc,
    )
    if orbit_teme.reference_frame != "TEME":
        raise StkValidationError("Orbit propagation result must use TEME")
    if orbit_fixed.source_reference_frame != "TEME":
        raise StkValidationError("Orbit Earth-fixed source frame must use TEME")
    if orbit_fixed.reference_frame != "ITRS":
        raise StkValidationError("Orbit Earth-fixed result must use ITRS")

    stk_teme = _stk_samples(capture.teme_samples, "STK TEME")
    stk_fixed = _stk_samples(capture.fixed_samples, "STK Fixed")
    teme = _orbit_teme_samples(orbit_teme.samples)
    fixed = _orbit_fixed_samples(orbit_fixed.samples)
    _require_sample_keys(stk_teme, stk_fixed, "STK TEME and Fixed")
    _require_sample_keys(teme, fixed, "Orbit TEME and ITRS")
    _require_sample_keys(stk_teme, teme, "STK and Orbit TEME")

    expected_times = tuple(float(value) for value in capture.sample_times_s)
    actual_times = tuple(dict.fromkeys(key[0] for key, _, _ in stk_teme))
    if actual_times != expected_times:
        raise StkValidationError(
            "state comparison sample times do not match the capture grid"
        )

    return StkStateComparisonReport(
        element_set_id=capture.element_set_id,
        simulation_epoch_utc=capture.simulation_epoch_utc,
        sample_times_s=expected_times,
        teme=_compare_frame(
            stk_teme,
            teme,
            stk_reference_frame="TEME",
            orbit_reference_frame="TEME",
        ),
        fixed=_compare_frame(
            stk_fixed,
            fixed,
            stk_reference_frame="Fixed",
            orbit_reference_frame="ITRS",
        ),
    )


_ComparableSample = tuple[
    tuple[float, str],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _stk_samples(
    samples: Sequence[StkStateSample],
    label: str,
) -> tuple[_ComparableSample, ...]:
    return _validated_samples(
        (
            (
                (sample.time_s, sample.satellite_id),
                sample.position_km,
                sample.velocity_km_s,
            )
            for sample in samples
        ),
        label,
    )


def _orbit_teme_samples(
    samples: Sequence[Sgp4TemeSample],
) -> tuple[_ComparableSample, ...]:
    return _validated_samples(
        (
            (
                (sample.time_s, sample.satellite_id),
                sample.position_teme_km,
                sample.velocity_teme_km_s,
            )
            for sample in samples
        ),
        "Orbit TEME",
    )


def _orbit_fixed_samples(
    samples: Sequence[Sgp4ItrsSample],
) -> tuple[_ComparableSample, ...]:
    return _validated_samples(
        (
            (
                (sample.time_s, sample.satellite_id),
                sample.position_itrs_km,
                sample.velocity_itrs_km_s,
            )
            for sample in samples
        ),
        "Orbit ITRS",
    )


def _validated_samples(
    values: Iterable[
        tuple[
            tuple[float, str],
            Sequence[float],
            Sequence[float],
        ]
    ],
    label: str,
) -> tuple[_ComparableSample, ...]:
    output: list[_ComparableSample] = []
    for key, position, velocity in values:
        time_s = _finite_float(key[0], f"{label} time_s")
        satellite_id = key[1]
        if not isinstance(satellite_id, str) or not satellite_id:
            raise StkValidationError(f"{label} satellite_id must be non-empty")
        output.append(
            (
                (time_s, satellite_id),
                _vector(position, f"{label} position"),
                _vector(velocity, f"{label} velocity"),
            )
        )
    if not output:
        raise StkValidationError(f"{label} samples must not be empty")
    keys = tuple(item[0] for item in output)
    if len(set(keys)) != len(keys):
        raise StkValidationError(f"{label} contains duplicate sample keys")
    if keys != tuple(sorted(keys)):
        raise StkValidationError(
            f"{label} samples must be ordered by time_s then satellite_id"
        )
    return tuple(output)


def _require_sample_keys(
    first: tuple[_ComparableSample, ...],
    second: tuple[_ComparableSample, ...],
    label: str,
) -> None:
    if tuple(item[0] for item in first) != tuple(item[0] for item in second):
        raise StkValidationError(f"{label} sample keys do not match exactly")


def _compare_frame(
    reference: tuple[_ComparableSample, ...],
    actual: tuple[_ComparableSample, ...],
    *,
    stk_reference_frame: str,
    orbit_reference_frame: str,
) -> StkFrameComparison:
    samples: list[StkStateDifference] = []
    for (key, reference_position, reference_velocity), (
        _,
        actual_position,
        actual_velocity,
    ) in zip(reference, actual):
        position_difference = _scale(
            _subtract(actual_position, reference_position),
            1000.0,
        )
        velocity_difference = _scale(
            _subtract(actual_velocity, reference_velocity),
            1000.0,
        )
        radial, in_track, cross_track = _ric_axes(
            reference_position,
            reference_velocity,
            satellite_id=key[1],
            time_s=key[0],
        )
        samples.append(
            StkStateDifference(
                satellite_id=key[1],
                time_s=key[0],
                position_difference_m=position_difference,
                velocity_difference_m_s=velocity_difference,
                position_error_m=_norm(position_difference),
                velocity_error_m_s=_norm(velocity_difference),
                radial_position_error_m=_dot(position_difference, radial),
                in_track_position_error_m=_dot(position_difference, in_track),
                cross_track_position_error_m=_dot(position_difference, cross_track),
            )
        )
    canonical_samples = tuple(samples)
    metric_specs: tuple[
        tuple[str, str, Callable[[StkStateDifference], float]],
        ...,
    ] = (
        ("position_error", "m", lambda sample: sample.position_error_m),
        ("velocity_error", "m/s", lambda sample: sample.velocity_error_m_s),
        (
            "radial_position_error",
            "m",
            lambda sample: abs(sample.radial_position_error_m),
        ),
        (
            "in_track_position_error",
            "m",
            lambda sample: abs(sample.in_track_position_error_m),
        ),
        (
            "cross_track_position_error",
            "m",
            lambda sample: abs(sample.cross_track_position_error_m),
        ),
    )
    return StkFrameComparison(
        stk_reference_frame=stk_reference_frame,
        orbit_reference_frame=orbit_reference_frame,
        samples=canonical_samples,
        metrics=tuple(
            _summarise_metric(canonical_samples, name, unit, accessor)
            for name, unit, accessor in metric_specs
        ),
    )


def _summarise_metric(
    samples: tuple[StkStateDifference, ...],
    metric: str,
    unit: str,
    accessor: Callable[[StkStateDifference], float],
) -> StkMetricSummary:
    values = tuple(_finite_float(accessor(sample), metric) for sample in samples)
    maximum = max(values)
    worst_index = values.index(maximum)
    worst = samples[worst_index]
    return StkMetricSummary(
        metric=metric,
        unit=unit,
        rms=math.sqrt(math.fsum(value * value for value in values) / len(values)),
        p95=_percentile(values, 0.95),
        maximum=maximum,
        worst_satellite_id=worst.satellite_id,
        worst_time_s=worst.time_s,
    )


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    rank = (len(ordered) - 1) * probability
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _ric_axes(
    position: tuple[float, float, float],
    velocity: tuple[float, float, float],
    *,
    satellite_id: str,
    time_s: float,
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    radial = _unit(
        position,
        f"RIC radial axis for satellite_id={satellite_id!r} at time_s={time_s}",
    )
    cross_track = _unit(
        _cross(position, velocity),
        f"RIC cross-track axis for satellite_id={satellite_id!r} at time_s={time_s}",
    )
    in_track = _unit(
        _cross(cross_track, radial),
        f"RIC in-track axis for satellite_id={satellite_id!r} at time_s={time_s}",
    )
    return radial, in_track, cross_track


def _require_equal(label: str, *values: str) -> None:
    if len(set(values)) != 1:
        raise StkValidationError(f"{label} values do not match exactly")


def _vector(value: Sequence[float], label: str) -> tuple[float, float, float]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise StkValidationError(f"{label} must contain exactly three values")
    return tuple(_finite_float(item, label) for item in value)  # type: ignore[return-value]


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StkValidationError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise StkValidationError(f"{label} must be a finite number")
    return 0.0 if number == 0.0 else number


def _subtract(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(first, second))  # type: ignore[return-value]


def _scale(
    value: tuple[float, float, float],
    factor: float,
) -> tuple[float, float, float]:
    return tuple(component * factor for component in value)  # type: ignore[return-value]


def _dot(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.fsum(a * b for a, b in zip(first, second))


def _cross(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _norm(value: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(value, value))


def _unit(
    value: tuple[float, float, float],
    label: str,
) -> tuple[float, float, float]:
    length = _norm(value)
    if length == 0.0:
        raise StkValidationError(f"{label} is undefined")
    return _scale(value, 1.0 / length)


def _normalise_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalise_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalise_json(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise StkValidationError(
                "canonical STK comparison contains a non-finite float"
            )
        return 0.0 if value == 0.0 else value
    return value


__all__ = [
    "STK_STATE_COMPARISON_SCHEMA",
    "StkFrameComparison",
    "StkMetricSummary",
    "StkStateComparisonReport",
    "StkStateDifference",
    "compare_stk_state_capture",
]
