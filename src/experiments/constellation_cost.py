"""Cost adapters that lift single-satellite estimates to constellation results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import to_jsonable
from ..orbit import ConstellationDesign
from ..architecture import CostResult, estimate_single_satellite_cost
from ..architecture import SatelliteDesignLimits, SingleSatelliteConfig


@dataclass(frozen=True)
class CostingSpec:
    single_satellite_case_path: str | None = None
    scale_by: str = "satellite_count"
    energy_accounting_mode: str = "offered_workload"
    target_served_ratio: float = 1.0
    nominal_isl_hops_per_request: float = 0.0
    energy_alignment_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.energy_accounting_mode != "offered_workload":
            raise ValueError("costing.energy_accounting_mode must be 'offered_workload'")
        if not math.isfinite(self.target_served_ratio) or not 0.0 <= self.target_served_ratio <= 1.0:
            raise ValueError("costing.target_served_ratio must be between 0 and 1")
        if not math.isfinite(self.nominal_isl_hops_per_request) or self.nominal_isl_hops_per_request < 0:
            raise ValueError("costing.nominal_isl_hops_per_request must be non-negative")
        if not math.isfinite(self.energy_alignment_tolerance) or self.energy_alignment_tolerance < 0:
            raise ValueError("costing.energy_alignment_tolerance must be non-negative")

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CostingSpec":
        data = data or {}
        result = cls(
            single_satellite_case_path=data.get("single_satellite_case_path"),
            scale_by=str(data.get("scale_by", "satellite_count")),
            energy_accounting_mode=str(data.get("energy_accounting_mode", "offered_workload")),
            target_served_ratio=float(data.get("target_served_ratio", 1.0)),
            nominal_isl_hops_per_request=float(data.get("nominal_isl_hops_per_request", 0.0)),
            energy_alignment_tolerance=float(data.get("energy_alignment_tolerance", 0.05)),
        )
        return result

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class ConstellationCostResult:
    evaluated: bool
    satellite_count: int
    single_satellite_case_path: str | None
    single_satellite: dict[str, Any] | None
    total_cost_usd: float | None
    annualized_cost_usd_per_year: float | None
    total_mass_kg: float | None
    peak_power_w: float | None
    warnings: tuple[str, ...]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def estimate_constellation_cost(
    design: ConstellationDesign,
    costing: CostingSpec,
    base_dir: Path | None = None,
    cost_limits: SatelliteDesignLimits | None = None,
    cost_config: SingleSatelliteConfig | None = None,
    single_satellite_case_path: str | None = None,
) -> ConstellationCostResult:
    """Estimate constellation cost by scaling an existing single-satellite case."""

    if cost_limits is None and cost_config is None and costing.single_satellite_case_path is None:
        return ConstellationCostResult(
            evaluated=False,
            satellite_count=design.satellite_count,
            single_satellite_case_path=None,
            single_satellite=None,
            total_cost_usd=None,
            annualized_cost_usd_per_year=None,
            total_mass_kg=None,
            peak_power_w=None,
            warnings=("costing.single_satellite_case_path is not configured",),
            provenance={"scale_by": costing.scale_by},
        )
    if costing.scale_by != "satellite_count":
        raise ValueError("only costing.scale_by='satellite_count' is implemented in the skeleton")

    if cost_limits is None or cost_config is None:
        raise ValueError(
            "constellation cost requires resolved immutable single-satellite hardware"
        )
    single_result = estimate_single_satellite_cost(
        cost_limits,
        cost_config,
        orbit_energy_condition=design.energy_design_condition(),
    )
    case_path_text = single_satellite_case_path
    adapter_warnings: tuple[str, ...] = ()

    n_sat = design.satellite_count
    warnings = tuple((*single_result.warnings, *adapter_warnings))
    return ConstellationCostResult(
        evaluated=True,
        satellite_count=n_sat,
        single_satellite_case_path=case_path_text,
        single_satellite=single_result.to_dict(),
        total_cost_usd=single_result.C_sat_usd * n_sat,
        annualized_cost_usd_per_year=single_result.annualized_cost_usd_per_year * n_sat,
        total_mass_kg=single_result.M_sat_kg * n_sat,
        peak_power_w=single_result.P_peak_w * n_sat,
        warnings=warnings,
        provenance={
            "scale_by": costing.scale_by,
            "single_satellite_input": "resolved_hardware" if cost_limits is not None else "case_path",
            "single_satellite_provenance": single_result.provenance,
        },
    )






