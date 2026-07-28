"""Deterministic comparisons between Orbit geometry and STK event evidence."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from ..sgp4.geometry import Sgp4GeometryResult
from .stk import StkValidationError
from .stk_geometry import StkGeometryCapture


STK_GEOMETRY_COMPARISON_SCHEMA = "orbitserve.orbit.stk-geometry-comparison"


@dataclass(frozen=True, slots=True)
class StkWindowMetric:
    channel: str
    identity: tuple[str, ...]
    orbit_window_count: int
    stk_window_count: int
    true_positive_samples: int
    true_negative_samples: int
    false_positive_samples: int
    false_negative_samples: int
    sample_agreement_ratio: float
    intersection_duration_s: float
    union_duration_s: float
    interval_iou: float
    max_start_error_s: float | None
    max_end_error_s: float | None
    max_duration_error_s: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "false_negative_samples": self.false_negative_samples,
            "false_positive_samples": self.false_positive_samples,
            "identity": list(self.identity),
            "intersection_duration_s": self.intersection_duration_s,
            "interval_iou": self.interval_iou,
            "max_duration_error_s": self.max_duration_error_s,
            "max_end_error_s": self.max_end_error_s,
            "max_start_error_s": self.max_start_error_s,
            "orbit_window_count": self.orbit_window_count,
            "sample_agreement_ratio": self.sample_agreement_ratio,
            "stk_window_count": self.stk_window_count,
            "true_negative_samples": self.true_negative_samples,
            "true_positive_samples": self.true_positive_samples,
            "union_duration_s": self.union_duration_s,
        }


@dataclass(frozen=True, slots=True)
class StkAerMetric:
    ingress_id: str
    satellite_id: str
    matched_sample_count: int
    elevation_error_rms_deg: float
    elevation_error_max_abs_deg: float
    range_error_rms_km: float
    range_error_max_abs_km: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "elevation_error_max_abs_deg": self.elevation_error_max_abs_deg,
            "elevation_error_rms_deg": self.elevation_error_rms_deg,
            "ingress_id": self.ingress_id,
            "matched_sample_count": self.matched_sample_count,
            "range_error_max_abs_km": self.range_error_max_abs_km,
            "range_error_rms_km": self.range_error_rms_km,
            "satellite_id": self.satellite_id,
        }


@dataclass(frozen=True, slots=True)
class StkGeometryComparisonReport:
    element_set_id: str
    simulation_epoch_utc: str
    window_metrics: tuple[StkWindowMetric, ...]
    aer_metrics: tuple[StkAerMetric, ...]
    schema: str = STK_GEOMETRY_COMPARISON_SCHEMA
    sunlight_primary_reference: str = "stk_umbra"
    penumbra_retained_as_diagnostic: bool = True
    formal_thresholds_frozen: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "aer_metrics": [metric.to_dict() for metric in self.aer_metrics],
            "element_set_id": self.element_set_id,
            "formal_thresholds_frozen": self.formal_thresholds_frozen,
            "penumbra_retained_as_diagnostic": (
                self.penumbra_retained_as_diagnostic
            ),
            "schema": self.schema,
            "simulation_epoch_utc": self.simulation_epoch_utc,
            "sunlight_primary_reference": self.sunlight_primary_reference,
            "window_metrics": [
                metric.to_dict() for metric in self.window_metrics
            ],
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


def compare_stk_geometry_capture(
    capture: StkGeometryCapture,
    orbit: Sgp4GeometryResult,
) -> StkGeometryComparisonReport:
    """Compare SGL, ISL, and Orbit eclipse intervals against STK evidence."""

    if not isinstance(capture, StkGeometryCapture):
        raise TypeError("capture must be StkGeometryCapture")
    if not isinstance(orbit, Sgp4GeometryResult):
        raise TypeError("orbit must be Sgp4GeometryResult")
    if capture.element_set_id != orbit.element_set_id:
        raise StkValidationError("STK and Orbit element_set_id values differ")
    if capture.simulation_epoch_utc != orbit.simulation_epoch_utc:
        raise StkValidationError("STK and Orbit simulation epochs differ")
    if (
        capture.sample_times_s[0] != orbit.interval_start_s
        or capture.sample_times_s[-1] != orbit.interval_end_s
    ):
        raise StkValidationError("STK and Orbit geometry intervals differ")
    sample_times = capture.sample_times_s[:-1]

    metrics: list[StkWindowMetric] = []
    sgl_identities = sorted(
        {(sample.ingress_id, sample.satellite_id) for sample in orbit.sgl_samples}
    )
    for identity in sgl_identities:
        orbit_windows = tuple(
            (window.start_s, window.end_s)
            for window in orbit.sgl_windows
            if (window.ingress_id, window.satellite_id) == identity
        )
        stk_windows = tuple(
            (window.start_s, window.end_s)
            for window in capture.sgl_windows
            if (window.ingress_id, window.satellite_id) == identity
        )
        metrics.append(
            _window_metric("sgl", identity, orbit_windows, stk_windows, sample_times)
        )

    isl_identities = sorted(
        {(sample.endpoint_a, sample.endpoint_b) for sample in orbit.isl_samples}
    )
    for identity in isl_identities:
        orbit_windows = tuple(
            (window.start_s, window.end_s)
            for window in orbit.isl_windows
            if (window.endpoint_a, window.endpoint_b) == identity
        )
        stk_windows = tuple(
            (window.start_s, window.end_s)
            for window in capture.isl_windows
            if (window.endpoint_a, window.endpoint_b) == identity
        )
        metrics.append(
            _window_metric("isl", identity, orbit_windows, stk_windows, sample_times)
        )

    sunlight_identities = sorted(
        {sample.satellite_id for sample in orbit.sunlight_samples}
    )
    for satellite_id in sunlight_identities:
        orbit_windows = tuple(
            (window.start_s, window.end_s)
            for window in orbit.sunlight_windows
            if window.satellite_id == satellite_id and not window.sunlit
        )
        stk_windows = tuple(
            (window.start_s, window.end_s)
            for window in capture.lighting_windows
            if window.satellite_id == satellite_id and window.lighting == "umbra"
        )
        metrics.append(
            _window_metric(
                "umbra",
                (satellite_id,),
                orbit_windows,
                stk_windows,
                sample_times,
            )
        )
    return StkGeometryComparisonReport(
        element_set_id=orbit.element_set_id,
        simulation_epoch_utc=orbit.simulation_epoch_utc,
        window_metrics=tuple(metrics),
        aer_metrics=_aer_metrics(capture, orbit),
    )


def _window_metric(
    channel: str,
    identity: tuple[str, ...],
    orbit_windows: tuple[tuple[float, float], ...],
    stk_windows: tuple[tuple[float, float], ...],
    sample_times: tuple[float, ...],
) -> StkWindowMetric:
    confusion = [0, 0, 0, 0]  # tp, tn, fp, fn
    for time_s in sample_times:
        orbit_active = _contains(orbit_windows, time_s)
        stk_active = _contains(stk_windows, time_s)
        if orbit_active and stk_active:
            confusion[0] += 1
        elif not orbit_active and not stk_active:
            confusion[1] += 1
        elif orbit_active:
            confusion[2] += 1
        else:
            confusion[3] += 1
    # Rename the directional disagreements relative to Orbit.
    tp, tn, false_negative, false_positive = confusion
    intersection = _intersection_duration(orbit_windows, stk_windows)
    orbit_duration = _duration(orbit_windows)
    stk_duration = _duration(stk_windows)
    union = orbit_duration + stk_duration - intersection
    agreement = (tp + tn) / len(sample_times) if sample_times else 1.0
    iou = intersection / union if union > 0.0 else 1.0
    if len(orbit_windows) == len(stk_windows):
        start_errors = [
            abs(first[0] - second[0])
            for first, second in zip(orbit_windows, stk_windows)
        ]
        end_errors = [
            abs(first[1] - second[1])
            for first, second in zip(orbit_windows, stk_windows)
        ]
        duration_errors = [
            abs((first[1] - first[0]) - (second[1] - second[0]))
            for first, second in zip(orbit_windows, stk_windows)
        ]
        max_start = max(start_errors, default=0.0)
        max_end = max(end_errors, default=0.0)
        max_duration = max(duration_errors, default=0.0)
    else:
        max_start = max_end = max_duration = None
    return StkWindowMetric(
        channel=channel,
        identity=identity,
        orbit_window_count=len(orbit_windows),
        stk_window_count=len(stk_windows),
        true_positive_samples=tp,
        true_negative_samples=tn,
        false_positive_samples=false_positive,
        false_negative_samples=false_negative,
        sample_agreement_ratio=_q(agreement),
        intersection_duration_s=_q(intersection),
        union_duration_s=_q(union),
        interval_iou=_q(iou),
        max_start_error_s=None if max_start is None else _q(max_start),
        max_end_error_s=None if max_end is None else _q(max_end),
        max_duration_error_s=None if max_duration is None else _q(max_duration),
    )


def _aer_metrics(
    capture: StkGeometryCapture,
    orbit: Sgp4GeometryResult,
) -> tuple[StkAerMetric, ...]:
    orbit_by_key = {
        (sample.ingress_id, sample.satellite_id, sample.time_s): sample
        for sample in orbit.sgl_samples
    }
    grid = set(capture.sample_times_s[:-1])
    errors: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for sample in capture.aer_samples:
        time_s = _nearest_grid_time(sample.time_s, grid)
        if time_s is None:
            continue
        orbit_sample = orbit_by_key.get(
            (sample.ingress_id, sample.satellite_id, time_s)
        )
        if orbit_sample is None or not orbit_sample.visible:
            continue
        errors[(sample.ingress_id, sample.satellite_id)].append(
            (
                orbit_sample.elevation_deg - sample.elevation_deg,
                orbit_sample.slant_range_km - sample.range_km,
            )
        )
    result: list[StkAerMetric] = []
    for (ingress_id, satellite_id), values in sorted(errors.items()):
        elevation = [value[0] for value in values]
        distance = [value[1] for value in values]
        result.append(
            StkAerMetric(
                ingress_id=ingress_id,
                satellite_id=satellite_id,
                matched_sample_count=len(values),
                elevation_error_rms_deg=_q(_rms(elevation)),
                elevation_error_max_abs_deg=_q(max(map(abs, elevation))),
                range_error_rms_km=_q(_rms(distance)),
                range_error_max_abs_km=_q(max(map(abs, distance))),
            )
        )
    return tuple(result)


def _nearest_grid_time(value: float, grid: set[float]) -> float | None:
    rounded = round(value, 9)
    if rounded in grid:
        return rounded
    candidate = min(grid, key=lambda item: abs(item - value), default=None)
    if candidate is not None and abs(candidate - value) <= 0.001:
        return candidate
    return None


def _contains(windows: Sequence[tuple[float, float]], time_s: float) -> bool:
    return any(start <= time_s < stop for start, stop in windows)


def _duration(windows: Iterable[tuple[float, float]]) -> float:
    return sum(stop - start for start, stop in windows)


def _intersection_duration(
    first: Sequence[tuple[float, float]],
    second: Sequence[tuple[float, float]],
) -> float:
    left = right = 0
    total = 0.0
    while left < len(first) and right < len(second):
        start = max(first[left][0], second[right][0])
        stop = min(first[left][1], second[right][1])
        if stop > start:
            total += stop - start
        if first[left][1] <= second[right][1]:
            left += 1
        else:
            right += 1
    return total


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


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
            raise ValueError("geometry comparison contains non-finite data")
        return 0.0 if value == 0.0 else value
    return value


__all__ = [
    "STK_GEOMETRY_COMPARISON_SCHEMA",
    "StkAerMetric",
    "StkGeometryComparisonReport",
    "StkWindowMetric",
    "compare_stk_geometry_capture",
]
