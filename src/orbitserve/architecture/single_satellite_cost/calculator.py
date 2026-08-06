"""Static single-satellite cost closure from one HardwareSpec."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from .defaults import PARAMETER_DOC
from .models import SatelliteCostProfile

if TYPE_CHECKING:
    from ..hardware import HardwareSpec


SIGMA_SB_W_M2_K4 = 5.670374419e-8


def estimate_single_satellite_cost(spec: "HardwareSpec") -> SatelliteCostProfile:
    """Calculate one satellite using only its immutable Architecture input."""

    from ..hardware import HardwareSpec

    if not isinstance(spec, HardwareSpec):
        raise TypeError("estimate_single_satellite_cost requires HardwareSpec")

    violations: list[str] = []
    warnings: list[str] = []
    _validate_nonnegative(spec, "hardware", violations)
    _validate_fractions(spec, violations)

    accelerator = spec.accelerator.resolve()
    storage = spec.storage.resolve()
    compute = spec.compute_payload
    interconnect = spec.intra_interconnect
    communication = spec.communication
    power = spec.power

    accelerator_count = compute.accelerator_count
    storage_count = spec.storage.device_count
    switch_count = interconnect.switch_count
    if accelerator_count <= 0:
        violations.append("hardware.compute_payload.accelerator_count must be positive")
        accelerator_count = 0
    if storage_count < 0:
        violations.append("hardware.storage.device_count must be non-negative")
        storage_count = 0
    if switch_count < 0:
        violations.append("hardware.intra_interconnect.switch_count must be non-negative")
        switch_count = 0
    if compute.purchase_cost_per_unit_usd is None:
        violations.append("hardware.compute_payload.purchase_cost_per_unit_usd is required")

    accelerator_cost_usd = accelerator_count * float(compute.purchase_cost_per_unit_usd or 0.0)
    accelerator_power_w = accelerator_count * accelerator.peak_power_w
    accelerator_mass_kg = accelerator_count * accelerator.mass_kg

    storage_cost_usd = storage_count * storage.capacity_tb * 1000.0 * storage.cost_per_gb_usd
    storage_power_w = storage_count * storage.power_w
    storage_mass_kg = storage_count * storage.mass_kg

    intra_bandwidth_gbs = switch_count * interconnect.switch_bandwidth_gbs
    interconnect_cost_usd = (
        interconnect.cost_fraction * (accelerator_cost_usd + storage_cost_usd)
        + switch_count * interconnect.switch_cost_usd
    )
    interconnect_power_w = (
        intra_bandwidth_gbs * interconnect.power_per_gbs_w
        + switch_count * interconnect.switch_power_w
    )
    interconnect_mass_kg = (
        intra_bandwidth_gbs * interconnect.mass_per_gbs_kg
        + switch_count * interconnect.switch_mass_kg
    )

    compute_base_cost_usd = accelerator_cost_usd + storage_cost_usd + interconnect_cost_usd
    compute_base_power_w = accelerator_power_w + storage_power_w + interconnect_power_w
    compute_base_mass_kg = accelerator_mass_kg + storage_mass_kg + interconnect_mass_kg
    compute_payload_cost_usd = (
        compute_base_cost_usd * (1.0 + compute.packaging_cost_fraction)
        + compute.space_adaptation_cost_usd
    )
    compute_payload_power_w = compute_base_power_w * (1.0 + compute.power_overhead_fraction)
    compute_payload_mass_kg = compute_base_mass_kg * (1.0 + compute.mass_overhead_fraction)

    communication_mass_kg = (
        communication.ground_link_mass_kg
        + communication.isl_terminal_count * communication.isl_terminal_mass_kg
    )
    communication_power_w = (
        communication.ground_link_power_w
        + communication.isl_terminal_count * communication.isl_terminal_power_w
    )
    communication_cost_usd = communication.allocated_cost_usd
    if communication.ground_cost_per_gbps_usd is not None:
        communication_cost_usd += (
            communication.ground_cost_per_gbps_usd
            * communication.ground_link_bandwidth_gbps
        )
    if communication.isl_cost_per_gbps_usd is not None:
        communication_cost_usd += (
            communication.isl_cost_per_gbps_usd
            * communication.isl_total_bandwidth_gbps
        )
    if communication.ground_cost_per_gbps_usd is None and communication.isl_cost_per_gbps_usd is None:
        communication_cost_usd += communication_mass_kg * spec.cost.build_cost_per_kg_usd

    if communication.ground_link_bandwidth_gbps > 0.0 and (
        communication.ground_link_mass_kg <= 0.0 or communication.ground_link_power_w <= 0.0
    ):
        violations.append("positive ground-link bandwidth requires positive terminal mass and power")
    if communication.isl_terminal_count > 0 and (
        communication.isl_terminal_bandwidth_gbps <= 0.0
        or communication.isl_terminal_mass_kg <= 0.0
        or communication.isl_terminal_power_w <= 0.0
    ):
        violations.append("positive ISL terminal count requires positive bandwidth, mass, and power")

    solar_mass_kg = _safe_ratio(
        power.solar_array_power_w,
        power.solar_specific_power_w_per_kg,
        "hardware.power.solar_specific_power_w_per_kg",
        violations,
    )
    battery_mass_kg = _safe_ratio(
        power.battery_capacity_wh,
        power.battery_specific_energy_wh_per_kg,
        "hardware.power.battery_specific_energy_wh_per_kg",
        violations,
    )
    solar_cost_usd = (
        power.solar_array_power_w * power.solar_cost_per_w_usd
        if power.solar_cost_per_w_usd is not None
        else solar_mass_kg * spec.cost.build_cost_per_kg_usd
    )
    battery_cost_usd = (
        power.battery_capacity_wh * power.battery_cost_per_wh_usd
        if power.battery_cost_per_wh_usd is not None
        else battery_mass_kg * spec.cost.build_cost_per_kg_usd
    )
    power_mass_kg = solar_mass_kg + battery_mass_kg
    power_cost_usd = solar_cost_usd + battery_cost_usd

    peak_power_w = spec.platform.idle_power_w + compute_payload_power_w + communication_power_w
    radiator_denominator = (
        spec.thermal.emissivity
        * spec.thermal.view_factor
        * SIGMA_SB_W_M2_K4
        * spec.thermal.radiator_temperature_k**4
        - spec.thermal.environment_heat_flux_w_m2
    )
    if radiator_denominator <= 0.0:
        violations.append("radiator heat-rejection denominator must be positive")
        radiator_area_m2 = 0.0
    else:
        radiator_area_m2 = peak_power_w / radiator_denominator
    thermal_mass_kg = spec.thermal.radiator_areal_density_kg_m2 * radiator_area_m2
    thermal_cost_usd = (
        peak_power_w * spec.thermal.thermal_cost_per_w_usd
        if spec.thermal.thermal_cost_per_w_usd is not None
        else thermal_mass_kg * spec.cost.build_cost_per_kg_usd
    )

    dry_mass_kg = (
        spec.platform.mass_kg
        + compute_payload_mass_kg
        + communication_mass_kg
        + power_mass_kg
        + thermal_mass_kg
    )
    total_mass_kg = dry_mass_kg * (1.0 + spec.cost.mass_margin_fraction)
    mass_margin_kg = total_mass_kg - dry_mass_kg
    platform_cost_usd = spec.platform.mass_kg * spec.cost.build_cost_per_kg_usd
    launch_cost_usd = total_mass_kg * spec.cost.launch_cost_per_kg_usd

    cost_breakdown = {
        "platform": platform_cost_usd,
        "compute_payload": compute_payload_cost_usd,
        "communication_payload": communication_cost_usd,
        "power_system": power_cost_usd,
        "thermal_system": thermal_cost_usd,
        "launch": launch_cost_usd,
    }
    mass_breakdown = {
        "platform": spec.platform.mass_kg,
        "compute_payload": compute_payload_mass_kg,
        "communication_payload": communication_mass_kg,
        "power_system": power_mass_kg,
        "thermal_system": thermal_mass_kg,
        "mass_margin": mass_margin_kg,
    }

    limits = spec.design_limits
    payload_peak_w = compute_payload_power_w + communication_power_w
    if limits.payload_power_limit_w is not None and payload_peak_w > limits.payload_power_limit_w:
        violations.append(
            f"payload peak power {payload_peak_w:.3f} W exceeds limit {limits.payload_power_limit_w:.3f} W"
        )
    if limits.satellite_mass_limit_kg is not None and total_mass_kg > limits.satellite_mass_limit_kg:
        violations.append(
            f"satellite mass {total_mass_kg:.3f} kg exceeds limit {limits.satellite_mass_limit_kg:.3f} kg"
        )
    if any(token in accelerator.name.lower() for token in ("h100", "h200", "b200")):
        warnings.append(
            f"{accelerator.name} is a datacenter-class sensitivity profile, not flight-proven hardware"
        )

    total_cost_usd = sum(cost_breakdown.values())
    return SatelliteCostProfile(
        total_cost_usd=total_cost_usd,
        cost_breakdown_usd=cost_breakdown,
        total_mass_kg=total_mass_kg,
        mass_breakdown_kg=mass_breakdown,
        peak_power_w=peak_power_w,
        battery_capacity_wh=power.battery_capacity_wh,
        solar_array_power_w=power.solar_array_power_w,
        radiator_area_m2=radiator_area_m2,
        annualized_cost_usd_per_year=total_cost_usd * spec.cost.capital_recovery_factor,
        feasible=not violations,
        violations=tuple(dict.fromkeys(violations)),
        warnings=tuple(dict.fromkeys(warnings)),
        provenance={
            "source": "HardwareSpec",
            "parameter_doc": PARAMETER_DOC,
            "accelerator_profile": accelerator.name,
            "storage_profile": storage.name,
            "cost_dependency": "architecture_static_configuration_only",
        },
    )


def _safe_ratio(numerator: float, denominator: float, label: str, violations: list[str]) -> float:
    if denominator <= 0.0:
        violations.append(f"{label} must be positive")
        return 0.0
    return numerator / denominator


def _validate_nonnegative(value: Any, prefix: str, violations: list[str]) -> None:
    if not is_dataclass(value):
        return
    for item in fields(value):
        current = getattr(value, item.name)
        name = f"{prefix}.{item.name}"
        if is_dataclass(current):
            _validate_nonnegative(current, name, violations)
        elif isinstance(current, (int, float)) and not isinstance(current, bool):
            numeric = float(current)
            if not math.isfinite(numeric) or numeric < 0.0:
                violations.append(f"{name} must be finite and non-negative")


def _validate_fractions(spec: "HardwareSpec", violations: list[str]) -> None:
    fractions = {
        "hardware.compute_payload.packaging_cost_fraction": spec.compute_payload.packaging_cost_fraction,
        "hardware.compute_payload.power_overhead_fraction": spec.compute_payload.power_overhead_fraction,
        "hardware.compute_payload.mass_overhead_fraction": spec.compute_payload.mass_overhead_fraction,
        "hardware.intra_interconnect.cost_fraction": spec.intra_interconnect.cost_fraction,
        "hardware.power.initial_soc": spec.power.initial_soc,
        "hardware.power.maximum_depth_of_discharge": spec.power.maximum_depth_of_discharge,
        "hardware.power.battery_eol_fraction": spec.power.battery_eol_fraction,
        "hardware.power.battery_charge_efficiency": spec.power.battery_charge_efficiency,
        "hardware.power.battery_discharge_efficiency": spec.power.battery_discharge_efficiency,
        "hardware.power.pv_eol_fraction": spec.power.pv_eol_fraction,
        "hardware.power.power_management_efficiency": spec.power.power_management_efficiency,
        "hardware.thermal.emissivity": spec.thermal.emissivity,
        "hardware.thermal.view_factor": spec.thermal.view_factor,
        "hardware.cost.mass_margin_fraction": spec.cost.mass_margin_fraction,
        "hardware.cost.capital_recovery_factor": spec.cost.capital_recovery_factor,
    }
    for name, value in fractions.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            violations.append(f"{name} must be in [0, 1]")
    if spec.power.initial_soc < spec.power.minimum_soc:
        violations.append("hardware.power.initial_soc must be greater than or equal to soc_min")


__all__ = ["estimate_single_satellite_cost"]
