"""Experiment-level cost aggregation for logical L2 compute nodes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..architecture import HardwareSpec, ResolvedArchitecture, resolve_architecture
from ..common import to_jsonable


@dataclass(frozen=True)
class CostingSpec:
    """Cost aggregation policy; Architecture remains the single-member authority."""

    scale_by: str = "compute_node_count"
    energy_accounting_mode: str = "offered_workload"
    target_served_ratio: float = 1.0
    nominal_isl_hops_per_request: float = 0.0
    energy_alignment_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.scale_by not in {"compute_node_count", "satellite_count"}:
            raise ValueError("costing.scale_by must be 'compute_node_count'")
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
        unsupported = sorted(set(data) - set(cls.__dataclass_fields__) - {"single_satellite_case_path"})
        if unsupported:
            raise ValueError("Unsupported CostingSpec fields: " + ", ".join(unsupported))
        if data.get("single_satellite_case_path"):
            raise ValueError("costing.single_satellite_case_path is obsolete; configure hardware")
        return cls(
            scale_by=str(data.get("scale_by", "compute_node_count")),
            energy_accounting_mode=str(data.get("energy_accounting_mode", "offered_workload")),
            target_served_ratio=float(data.get("target_served_ratio", 1.0)),
            nominal_isl_hops_per_request=float(data.get("nominal_isl_hops_per_request", 0.0)),
            energy_alignment_tolerance=float(data.get("energy_alignment_tolerance", 0.05)),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class ConstellationCostResult:
    evaluated: bool
    compute_node_count: int
    formation_member_count: int
    physical_satellite_count: int
    single_satellite: dict[str, Any] | None
    total_cost_usd: float | None
    annualized_cost_usd_per_year: float | None
    total_mass_kg: float | None
    peak_power_w: float | None
    warnings: tuple[str, ...]
    provenance: dict[str, Any]

    @property
    def satellite_count(self) -> int:
        """Compatibility alias; canonical outputs use physical_satellite_count."""

        return self.physical_satellite_count

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def estimate_constellation_cost(
    design: Any,
    costing: CostingSpec,
    *,
    hardware: HardwareSpec | None = None,
    resolved_architecture: ResolvedArchitecture | None = None,
    formation_member_count: int = 1,
    **_: Any,
) -> ConstellationCostResult:
    """Scale closed single-satellite cost by logical nodes and L2 members.

    The single-satellite profile already includes its configured communication
    terminals.  No topology-specific terminal surcharge is added here.
    """

    compute_node_count = _compute_node_count(design)
    if formation_member_count < 1:
        raise ValueError("formation_member_count must be at least 1")
    physical_satellite_count = compute_node_count * formation_member_count
    resolved = resolved_architecture or (resolve_architecture(hardware) if hardware else None)
    if resolved is None:
        return ConstellationCostResult(
            evaluated=False,
            compute_node_count=compute_node_count,
            formation_member_count=formation_member_count,
            physical_satellite_count=physical_satellite_count,
            single_satellite=None,
            total_cost_usd=None,
            annualized_cost_usd_per_year=None,
            total_mass_kg=None,
            peak_power_w=None,
            warnings=("hardware is not configured",),
            provenance={"scale_by": "compute_node_count_x_formation_member_count"},
        )
    single = resolved.cost_profile
    return ConstellationCostResult(
        evaluated=True,
        compute_node_count=compute_node_count,
        formation_member_count=formation_member_count,
        physical_satellite_count=physical_satellite_count,
        single_satellite=single.to_dict(),
        total_cost_usd=single.total_cost_usd * physical_satellite_count,
        annualized_cost_usd_per_year=(
            single.annualized_cost_usd_per_year * physical_satellite_count
        ),
        total_mass_kg=single.total_mass_kg * physical_satellite_count,
        peak_power_w=single.peak_power_w * physical_satellite_count,
        warnings=single.warnings,
        provenance={
            "scale_by": "compute_node_count_x_formation_member_count",
            "single_satellite_includes_configured_laser_terminals": True,
            "topology_specific_cost_added": False,
            "single_satellite_provenance": single.provenance,
        },
    )


def _compute_node_count(design: Any) -> int:
    value = getattr(design, "compute_node_count", None)
    if value is None:
        value = getattr(design, "satellite_count", None)
    if value is None and isinstance(design, dict):
        value = design.get("compute_node_count", design.get("satellite_count"))
    if value is None or int(value) < 1:
        raise ValueError("design must define a positive compute_node_count")
    return int(value)
