"""Shared token/byte energy accounting for cost sizing and runtime.

This module deliberately depends only on plain Python data.  It is the common
arithmetic boundary between offered-workload sizing and runtime event debits;
it must not read runtime results.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class EnergyIntensity:
    """Baseline-exclusive bus-side energy coefficients."""

    compute_standby_power_w: float = 0.0
    communication_standby_power_w: float = 0.0
    compute_energy_j_per_token: float = 0.5
    link_energy_j_per_byte: float = 1e-6

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_nonnegative_finite(value, name)

    @property
    def standby_power_w(self) -> float:
        return self.compute_standby_power_w + self.communication_standby_power_w

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicEnergyBreakdown:
    compute_j: float
    link_j: float

    @property
    def total_j(self) -> float:
        return self.compute_j + self.link_j

    def to_dict(self) -> dict[str, float]:
        return {
            "compute_j": self.compute_j,
            "link_j": self.link_j,
            "total_j": self.total_j,
        }


@dataclass(frozen=True)
class PlannedEnergyDemand:
    """Offered-workload token/byte demand evenly normalized per satellite.

    ``planned_*`` totals are allowed to be fractional because
    ``target_served_ratio`` scales the offered workload statistically.  No
    field in this object is derived from runtime accepted/served records.
    """

    horizon_s: float
    satellite_count: int
    offered_request_count: int
    target_served_ratio: float
    planned_request_count: float
    nominal_isl_hops_per_request: float
    total_compute_tokens: float
    total_ground_link_bytes: float
    total_isl_link_bytes: float
    total_physical_link_bytes: float
    total_compute_energy_j: float
    total_link_energy_j: float
    per_satellite_compute_tokens: float
    per_satellite_physical_link_bytes: float
    per_satellite_compute_energy_j: float
    per_satellite_link_energy_j: float
    per_satellite_dynamic_power_w: float
    maximum_request_dynamic_energy_j: float
    compute_energy_j_per_token: float
    link_energy_j_per_byte: float
    sizing_mode: str = "offered_workload_energy_audit"
    uses_runtime_served_requests: bool = False

    def __post_init__(self) -> None:
        if self.sizing_mode != "offered_workload_energy_audit":
            raise ValueError("planned energy demand sizing_mode must be 'offered_workload_energy_audit'")
        _require_positive_finite(self.horizon_s, "horizon_s")
        if isinstance(self.satellite_count, bool) or int(self.satellite_count) != self.satellite_count:
            raise ValueError("satellite_count must be a positive integer")
        if self.satellite_count <= 0:
            raise ValueError("satellite_count must be a positive integer")
        if isinstance(self.offered_request_count, bool) or int(self.offered_request_count) != self.offered_request_count:
            raise ValueError("offered_request_count must be a non-negative integer")
        if self.offered_request_count < 0:
            raise ValueError("offered_request_count must be a non-negative integer")
        _require_fraction(self.target_served_ratio, "target_served_ratio")
        for name, value in asdict(self).items():
            if name in {
                "sizing_mode",
                "uses_runtime_served_requests",
                "horizon_s",
                "satellite_count",
                "offered_request_count",
                "target_served_ratio",
            }:
                continue
            _require_nonnegative_finite(value, name)
        if self.uses_runtime_served_requests:
            raise ValueError("planned energy demand must not use runtime served requests")

    @property
    def total_dynamic_energy_j(self) -> float:
        return self.total_compute_energy_j + self.total_link_energy_j

    @property
    def per_satellite_dynamic_energy_j(self) -> float:
        return self.per_satellite_compute_energy_j + self.per_satellite_link_energy_j

    @property
    def per_satellite_dynamic_buffer_energy_j(self) -> float:
        """Dynamic burst energy one identically sized satellite must support."""

        return max(
            self.per_satellite_dynamic_energy_j,
            self.maximum_request_dynamic_energy_j,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_dynamic_energy_j"] = self.total_dynamic_energy_j
        payload["per_satellite_dynamic_energy_j"] = self.per_satellite_dynamic_energy_j
        payload[
            "per_satellite_dynamic_buffer_energy_j"
        ] = self.per_satellite_dynamic_buffer_energy_j
        return payload


def compute_dynamic_energy_j(
    *,
    prompt_tokens: float,
    output_tokens: float,
    link_bytes: float,
    intensity: EnergyIntensity,
) -> DynamicEnergyBreakdown:
    """Return baseline-exclusive compute and physical-link bus energy."""

    prompt = _require_nonnegative_finite(prompt_tokens, "prompt_tokens")
    output = _require_nonnegative_finite(output_tokens, "output_tokens")
    physical_bytes = _require_nonnegative_finite(link_bytes, "link_bytes")
    return DynamicEnergyBreakdown(
        compute_j=(prompt + output) * intensity.compute_energy_j_per_token,
        link_j=physical_bytes * intensity.link_energy_j_per_byte,
    )


def build_planned_energy_demand(
    workload_generation: Any,
    *,
    satellite_count: int,
    intensity: EnergyIntensity,
    target_served_ratio: float = 1.0,
    nominal_isl_hops_per_request: float = 0.0,
    simulation_duration_s: float | None = None,
) -> PlannedEnergyDemand:
    """Build per-satellite demand from offered requests, never runtime output.

    Horizon authority is ``workload manifest.time.time_horizon_s`` first and
    ``simulation_duration_s`` second.  Missing or malformed values fail closed.
    Each request crosses one ground link in each direction using its explicit
    input/output byte counts; nominal ISL hops multiply the same payload bytes.
    """

    count = _require_positive_integer(satellite_count, "satellite_count")
    ratio = _require_fraction(target_served_ratio, "target_served_ratio")
    nominal_hops = _require_nonnegative_finite(
        nominal_isl_hops_per_request,
        "nominal_isl_hops_per_request",
    )
    manifest = _mapping(_field(workload_generation, "manifest", {}))
    manifest_time = _mapping(manifest.get("time"))
    horizon_value = manifest_time.get("time_horizon_s")
    if horizon_value is None:
        horizon_value = manifest.get("time_horizon_s")
    if horizon_value is None:
        horizon_value = simulation_duration_s
    horizon_s = _require_positive_finite(horizon_value, "energy sizing horizon_s")

    requests = tuple(_request_rows(_field(workload_generation, "requests", ())))
    total_prompt_tokens = 0.0
    total_output_tokens = 0.0
    total_payload_bytes = 0.0
    maximum_request_dynamic_energy_j = 0.0
    for index, request in enumerate(requests):
        prompt_tokens = _require_nonnegative_finite(
            request.get("prompt_tokens"),
            f"requests[{index}].prompt_tokens",
        )
        output_tokens = _require_nonnegative_finite(
            request.get("output_tokens"),
            f"requests[{index}].output_tokens",
        )
        input_bytes = _require_nonnegative_finite(
            request.get("input_bytes"),
            f"requests[{index}].input_bytes",
        )
        output_bytes = _require_nonnegative_finite(
            request.get("output_bytes"),
            f"requests[{index}].output_bytes",
        )
        total_prompt_tokens += prompt_tokens
        total_output_tokens += output_tokens
        request_payload_bytes = input_bytes + output_bytes
        total_payload_bytes += request_payload_bytes
        if ratio > 0.0:
            request_energy = compute_dynamic_energy_j(
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                link_bytes=request_payload_bytes * (1.0 + nominal_hops),
                intensity=intensity,
            )
            maximum_request_dynamic_energy_j = max(
                maximum_request_dynamic_energy_j,
                request_energy.total_j,
            )

    planned_prompt_tokens = total_prompt_tokens * ratio
    planned_output_tokens = total_output_tokens * ratio
    planned_tokens = planned_prompt_tokens + planned_output_tokens
    ground_link_bytes = total_payload_bytes * ratio
    isl_link_bytes = total_payload_bytes * nominal_hops * ratio
    physical_link_bytes = ground_link_bytes + isl_link_bytes
    energy = compute_dynamic_energy_j(
        prompt_tokens=planned_prompt_tokens,
        output_tokens=planned_output_tokens,
        link_bytes=physical_link_bytes,
        intensity=intensity,
    )
    per_sat_compute_tokens = planned_tokens / count
    per_sat_link_bytes = physical_link_bytes / count
    per_sat_compute_j = energy.compute_j / count
    per_sat_link_j = energy.link_j / count
    return PlannedEnergyDemand(
        horizon_s=horizon_s,
        satellite_count=count,
        offered_request_count=len(requests),
        target_served_ratio=ratio,
        planned_request_count=len(requests) * ratio,
        nominal_isl_hops_per_request=nominal_hops,
        total_compute_tokens=planned_tokens,
        total_ground_link_bytes=ground_link_bytes,
        total_isl_link_bytes=isl_link_bytes,
        total_physical_link_bytes=physical_link_bytes,
        total_compute_energy_j=energy.compute_j,
        total_link_energy_j=energy.link_j,
        per_satellite_compute_tokens=per_sat_compute_tokens,
        per_satellite_physical_link_bytes=per_sat_link_bytes,
        per_satellite_compute_energy_j=per_sat_compute_j,
        per_satellite_link_energy_j=per_sat_link_j,
        per_satellite_dynamic_power_w=(per_sat_compute_j + per_sat_link_j) / horizon_s,
        maximum_request_dynamic_energy_j=maximum_request_dynamic_energy_j,
        compute_energy_j_per_token=intensity.compute_energy_j_per_token,
        link_energy_j_per_byte=intensity.link_energy_j_per_byte,
    )


def _field(value: Any, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _request_rows(value: Any) -> Iterable[dict[str, Any]]:
    if value is None or isinstance(value, (str, bytes, Mapping)):
        if isinstance(value, Mapping):
            raise ValueError("workload requests must be a sequence of objects")
        return ()
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise ValueError("workload requests must be a sequence of objects") from exc
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"requests[{index}] must be an object")
    return tuple(dict(row) for row in rows)


def _require_positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be a positive integer") from None
    if not math.isfinite(numeric) or numeric != parsed or parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _require_positive_finite(value: Any, name: str) -> float:
    parsed = _number(value, name)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _require_nonnegative_finite(value: Any, name: str) -> float:
    parsed = _number(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _require_fraction(value: Any, name: str) -> float:
    parsed = _require_nonnegative_finite(value, name)
    if parsed > 1:
        raise ValueError(f"{name} must be <= 1")
    return parsed


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


__all__ = [
    "DynamicEnergyBreakdown",
    "EnergyIntensity",
    "PlannedEnergyDemand",
    "build_planned_energy_demand",
    "compute_dynamic_energy_j",
]
