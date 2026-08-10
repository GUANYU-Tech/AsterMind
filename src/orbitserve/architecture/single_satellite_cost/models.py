"""Static catalog and cost-result models used by Architecture."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SatelliteDesignLimits:
    """Optional launch-time feasibility limits for one satellite."""

    payload_power_limit_w: float | None = None
    satellite_mass_limit_kg: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SatelliteDesignLimits":
        data = data or {}
        return cls(
            payload_power_limit_w=_optional_float(data.get("payload_power_limit_w")),
            satellite_mass_limit_kg=_optional_float(data.get("satellite_mass_limit_kg")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceleratorProfile:
    name: str
    peak_compute_tflops: float
    fast_mem_gb: float
    fast_mem_bw_gbs: float
    peak_power_w: float
    mass_kg: float
    intra_bw_gbs: float
    mass_source: str = ""
    source_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AcceleratorProfile":
        return cls(
            name=str(data["name"]),
            peak_compute_tflops=float(data["peak_compute_tflops"]),
            fast_mem_gb=float(data["fast_mem_gb"]),
            fast_mem_bw_gbs=float(data["fast_mem_bw_gbs"]),
            peak_power_w=float(data["peak_power_w"]),
            mass_kg=float(data["mass_kg"]),
            intra_bw_gbs=float(data.get("intra_bw_gbs", 0.0)),
            mass_source=str(data.get("mass_source", "custom")),
            source_ids=tuple(str(item) for item in data.get("source_ids", ())),
        )


@dataclass(frozen=True)
class StorageDevice:
    name: str
    capacity_tb: float
    bandwidth_gbs: float
    power_w: float
    mass_kg: float
    cost_per_gb_usd: float
    source_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StorageDevice":
        return cls(
            name=str(data["name"]),
            capacity_tb=float(data["capacity_tb"]),
            bandwidth_gbs=float(data["bandwidth_gbs"]),
            power_w=float(data["power_w"]),
            mass_kg=float(data["mass_kg"]),
            cost_per_gb_usd=float(data["cost_per_gb_usd"]),
            source_ids=tuple(str(item) for item in data.get("source_ids", ())),
        )


@dataclass(frozen=True)
class SatelliteCostProfile:
    """Cost, mass, power, and static feasibility of one satellite."""

    total_cost_usd: float
    cost_breakdown_usd: dict[str, float]
    total_mass_kg: float
    mass_breakdown_kg: dict[str, float]
    peak_power_w: float
    battery_capacity_wh: float
    solar_array_power_w: float
    radiator_area_m2: float
    annualized_cost_usd_per_year: float
    feasible: bool
    violations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


__all__ = [
    "AcceleratorProfile",
    "SatelliteCostProfile",
    "SatelliteDesignLimits",
    "StorageDevice",
]
