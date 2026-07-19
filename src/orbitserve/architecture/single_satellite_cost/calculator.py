"""Single-satellite cost calculator."""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

from .defaults import PARAMETER_DOC, accelerator_profile, storage_profile
from .models import (
    AcceleratorProfile,
    CostResult,
    SatelliteDesignLimits,
    SingleSatelliteConfig,
    StorageDevice,
)


SIGMA_SB_W_M2_K4 = 5.670374419e-8


def estimate_single_satellite_cost(
    limits: SatelliteDesignLimits,
    config: SingleSatelliteConfig,
    *,
    orbit_energy_condition: dict[str, Any] | None = None,
) -> CostResult:
    """Estimate single-satellite cost, mass, power, and feasibility."""

    violations: list[str] = []
    warnings: list[str] = []
    _validate_non_negative(limits, "design_limits", violations)
    _validate_non_negative(config, "config", violations)

    profile = _resolve_accelerator_profile(config, violations)
    storage = _resolve_storage_device(config, violations)

    if config.compute.mode != "cots":
        violations.append("only compute.mode='cots' is implemented")
    if config.compute.purchase_cost_per_unit_usd is None:
        violations.append("compute.purchase_cost_per_unit_usd is required for COTS cost closure")

    n_acc = config.compute.accelerator_count
    if n_acc <= 0:
        violations.append("compute.accelerator_count must be positive")
        n_acc = 0
    acc_drivers = {"configured": n_acc}
    purchase_cost = config.compute.purchase_cost_per_unit_usd or 0.0
    C_compute_module_usd = n_acc * purchase_cost
    P_compute_module_w = n_acc * profile.peak_power_w
    M_compute_module_kg = n_acc * profile.mass_kg

    n_storage = config.storage.device_count
    if n_storage < 0:
        violations.append("storage.device_count must be non-negative")
        n_storage = 0
    storage_drivers = {"configured": n_storage}
    C_memory_storage_usd = n_storage * storage.capacity_tb * 1000.0 * storage.cost_per_gb_usd
    P_memory_storage_w = n_storage * storage.power_w
    M_memory_storage_kg = n_storage * storage.mass_kg

    n_switch = config.interconnect.switch_count
    if n_switch < 0:
        violations.append("interconnect.switch_count must be non-negative")
        n_switch = 0
    provisioned_intra_bw_gbs = n_switch * config.interconnect.switch_bw_gbs
    C_intra_sat_interconnect_usd = (
        config.interconnect.alpha_intra * (C_compute_module_usd + C_memory_storage_usd)
        + n_switch * config.interconnect.switch_cost_usd
    )
    P_intra_sat_interconnect_w = (
        config.interconnect.p_intra_per_gbs_w * provisioned_intra_bw_gbs
        + n_switch * config.interconnect.switch_power_w
    )
    M_intra_sat_interconnect_kg = (
        config.interconnect.m_intra_per_gbs_kg * provisioned_intra_bw_gbs
        + n_switch * config.interconnect.switch_mass_kg
    )

    C_board_package_usd = config.compute.alpha_pkg * (
        C_compute_module_usd + C_memory_storage_usd + C_intra_sat_interconnect_usd
    )
    P_board_package_w = config.compute.alpha_power_overhead * (
        P_compute_module_w + P_memory_storage_w + P_intra_sat_interconnect_w
    )
    M_board_package_kg = config.compute.alpha_mass_overhead * (
        M_compute_module_kg + M_memory_storage_kg + M_intra_sat_interconnect_kg
    )

    C_compute_payload_usd = (
        C_compute_module_usd
        + C_memory_storage_usd
        + C_intra_sat_interconnect_usd
        + C_board_package_usd
        + config.compute.space_adaptation_cost_usd
    )
    P_compute_payload_w = (
        P_compute_module_w
        + P_memory_storage_w
        + P_intra_sat_interconnect_w
        + P_board_package_w
    )
    M_compute_payload_kg = (
        M_compute_module_kg
        + M_memory_storage_kg
        + M_intra_sat_interconnect_kg
        + M_board_package_kg
    )

    M_comm_payload_kg = (
        config.communication.ground_link_mass_kg
        + config.communication.isl_terminal_count * config.communication.isl_terminal_mass_kg
    )
    P_comm_payload_w = (
        config.communication.ground_link_power_w
        + config.communication.isl_terminal_count * config.communication.isl_terminal_power_w
    )
    isl_total_bandwidth_gbps = config.communication.isl_terminal_count * config.communication.isl_terminal_bandwidth_gbps
    if config.communication.ground_link_bandwidth_gbps > 0 and (
        config.communication.ground_link_mass_kg <= 0
        or config.communication.ground_link_power_w <= 0
    ):
        violations.append("positive ground bandwidth requires positive ground link bandwidth, mass, and power")
    if config.communication.isl_terminal_count > 0 and (
        config.communication.isl_terminal_bandwidth_gbps <= 0
        or config.communication.isl_terminal_mass_kg <= 0
        or config.communication.isl_terminal_power_w <= 0
    ):
        violations.append("positive ISL bandwidth requires positive ISL terminal count, bandwidth, mass, and power")
    if config.communication.use_direct_link_costs:
        warnings.append("direct USD/Gbps link costs are enabled; verify source before paper use")
        C_comm_payload_usd = config.communication.clink_allocated_usd
        if config.communication.ground_link_bandwidth_gbps > 0 and config.communication.ground_cost_per_gbps_usd is None:
            violations.append("direct communication cost mode requires ground_cost_per_gbps_usd for positive ground bandwidth")
        elif config.communication.ground_cost_per_gbps_usd is not None:
            C_comm_payload_usd += config.communication.ground_cost_per_gbps_usd * config.communication.ground_link_bandwidth_gbps
        if config.communication.isl_terminal_count > 0 and config.communication.isl_cost_per_gbps_usd is None:
            violations.append("direct communication cost mode requires isl_cost_per_gbps_usd for positive ISL bandwidth")
        elif config.communication.isl_cost_per_gbps_usd is not None:
            C_comm_payload_usd += config.communication.isl_cost_per_gbps_usd * isl_total_bandwidth_gbps
    else:
        C_comm_payload_usd = (
            M_comm_payload_kg * config.cost.build_cost_per_kg_usd
            + config.communication.clink_allocated_usd
        )

    P_peak_w = config.platform.power_w + P_compute_payload_w + P_comm_payload_w
    standby_power_w = (
        config.power.compute_standby_power_w
        + config.power.communication_standby_power_w
    )
    baseline_power_w = config.platform.power_w + standby_power_w
    power_sizing_mode = "conservative_peak_load"
    P_avg_w = P_peak_w
    P_shadow_avg_w = P_peak_w
    condition = dict(orbit_energy_condition or {})
    eclipse_duration_min = float(condition.get("eclipse_duration_min", 35.0))
    sunlit_fraction = float(condition.get("sunlit_fraction", 0.64))
    if not math.isfinite(eclipse_duration_min) or eclipse_duration_min < 0.0:
        violations.append("orbit_energy_condition.eclipse_duration_min must be finite and non-negative")
        eclipse_duration_min = 0.0
    if not math.isfinite(sunlit_fraction) or not 0.0 < sunlit_fraction <= 1.0:
        violations.append("orbit_energy_condition.sunlit_fraction must be in (0, 1]")
        # Keep the remaining accounting finite while reporting the invalid input.
        sunlit_fraction = 1.0
    if not orbit_energy_condition:
        warnings.append("orbit energy condition is not supplied; conservative default eclipse/sunlit geometry is used")

    soc_min = 1.0 - config.power.dod_max
    usable_initial_soc_fraction = 1.0 - soc_min
    battery_denominator = (
        usable_initial_soc_fraction
        * config.power.battery_eol_fraction
        * config.power.battery_discharge_efficiency
    )
    if battery_denominator <= 0:
        violations.append(
            "(initial_soc - soc_min) * battery_eol_fraction * "
            "battery_discharge_efficiency must be positive"
        )
        E_battery_req_wh = 0.0
        eclipse_battery_req_wh = 0.0
    else:
        eclipse_battery_req_wh = (
            P_shadow_avg_w
            * (eclipse_duration_min / 60.0)
            / battery_denominator
        )
        E_battery_req_wh = max(
            eclipse_battery_req_wh,
            config.power.minimum_battery_capacity_wh,
        )
        if E_battery_req_wh == config.power.minimum_battery_capacity_wh:
            warnings.append(
                "battery sizing uses minimum_battery_capacity_wh because configured eclipse demand is smaller"
            )

    solar_denominator = config.power.eta_pmad * config.power.pv_eol_fraction
    shadow_fraction = max(0.0, 1.0 - sunlit_fraction)
    storage_denominator = (
        sunlit_fraction
        * config.power.battery_charge_efficiency
        * config.power.battery_discharge_efficiency
    )
    if solar_denominator <= 0 or storage_denominator <= 0:
        violations.append(
            "eta_pmad * pv_eol_fraction and sunlit_fraction * battery_charge_efficiency * "
            "battery_discharge_efficiency must be positive"
        )
        P_solar_req_w = 0.0
    else:
        P_sunlit_avg_w = (
            P_avg_w - shadow_fraction * P_shadow_avg_w
        ) / sunlit_fraction
        if P_sunlit_avg_w < 0:
            violations.append(
                "P_avg is inconsistent with sunlit_fraction and P_shadow_avg"
            )
            P_sunlit_avg_w = 0.0
        shadow_recharge_bus_w = (
            P_shadow_avg_w * shadow_fraction / storage_denominator
        )
        P_solar_req_w = (
            (P_sunlit_avg_w + shadow_recharge_bus_w)
            * (1.0 + config.power.power_margin_fraction)
            / solar_denominator
        )

    installed_battery_wh = (
        E_battery_req_wh
        if config.power.battery_capacity_wh is None
        else config.power.battery_capacity_wh
    )
    installed_solar_w = (
        P_solar_req_w
        if config.power.solar_array_power_w is None
        else config.power.solar_array_power_w
    )
    if installed_battery_wh < E_battery_req_wh:
        violations.append(
            "power.battery_capacity_wh is smaller than the required eclipse capacity"
        )
    if installed_solar_w < P_solar_req_w:
        violations.append(
            "power.solar_array_power_w is smaller than the required solar capacity"
        )

    M_solar_kg = _safe_ratio(installed_solar_w, config.power.solar_specific_power_w_per_kg, "solar_specific_power_w_per_kg", violations)
    M_battery_kg = _safe_ratio(installed_battery_wh, config.power.battery_specific_energy_wh_per_kg, "battery_specific_energy_wh_per_kg", violations)
    M_power_kg = M_solar_kg + M_battery_kg

    if config.power.solar_cost_per_w_usd is None:
        C_solar_usd = M_solar_kg * config.cost.build_cost_per_kg_usd
    else:
        C_solar_usd = config.power.solar_cost_per_w_usd * installed_solar_w
    if config.power.battery_cost_per_wh_usd is None:
        C_battery_usd = M_battery_kg * config.cost.build_cost_per_kg_usd
    else:
        C_battery_usd = config.power.battery_cost_per_wh_usd * installed_battery_wh
    C_power_usd = C_solar_usd + C_battery_usd

    radiator_denominator = (
        config.thermal.emissivity
        * config.thermal.view_factor
        * SIGMA_SB_W_M2_K4
        * config.thermal.radiator_temperature_k**4
        - config.thermal.environment_heat_flux_w_m2
    )
    if radiator_denominator <= 0:
        violations.append("radiator heat rejection denominator must be positive")
        A_rad_m2 = 0.0
        M_thermal_kg = 0.0
    else:
        A_rad_m2 = P_peak_w / radiator_denominator
        M_thermal_kg = config.thermal.radiator_areal_density_kg_m2 * A_rad_m2

    if config.thermal.thermal_cost_per_w_usd is None:
        C_thermal_usd = M_thermal_kg * config.cost.build_cost_per_kg_usd
    else:
        C_thermal_usd = config.thermal.thermal_cost_per_w_usd * P_peak_w

    M_platform_kg = config.platform.mass_kg
    C_platform_usd = M_platform_kg * config.cost.build_cost_per_kg_usd
    M_sat_dry_before_margin_kg = (
        M_platform_kg
        + M_compute_payload_kg
        + M_comm_payload_kg
        + M_power_kg
        + M_thermal_kg
    )
    M_sat_kg = M_sat_dry_before_margin_kg * (1.0 + config.cost.mass_margin_fraction)
    C_launch_usd = config.cost.launch_cost_per_kg_usd * M_sat_kg

    C_sat_usd = (
        C_platform_usd
        + C_compute_payload_usd
        + C_comm_payload_usd
        + C_power_usd
        + C_thermal_usd
        + C_launch_usd
    )

    payload_peak_w = P_compute_payload_w + P_comm_payload_w
    if limits.payload_power_limit_w is not None and payload_peak_w > limits.payload_power_limit_w:
        violations.append(
            f"payload peak power {payload_peak_w:.3f} W exceeds limit {limits.payload_power_limit_w:.3f} W"
        )
    if limits.satellite_mass_limit_kg is not None and M_sat_kg > limits.satellite_mass_limit_kg:
        violations.append(f"satellite mass {M_sat_kg:.3f} kg exceeds limit {limits.satellite_mass_limit_kg:.3f} kg")
    if any(token in profile.name.lower() for token in ("h100", "h200", "b200")):
        warnings.append(f"{profile.name} is a datacenter-class upper-bound/sensitivity profile, not flight-proven hardware")
    if "proxy" in profile.mass_source.lower():
        warnings.append(f"accelerator mass source is proxy/sensitivity: {profile.mass_source}")

    C_breakdown_usd = {
        "platform": C_platform_usd,
        "compute_payload": C_compute_payload_usd,
        "communication_payload": C_comm_payload_usd,
        "power": C_power_usd,
        "thermal": C_thermal_usd,
        "launch": C_launch_usd,
    }
    M_breakdown_kg = {
        "platform_before_margin": M_platform_kg,
        "compute_payload_before_margin": M_compute_payload_kg,
        "communication_payload_before_margin": M_comm_payload_kg,
        "power_before_margin": M_power_kg,
        "thermal_before_margin": M_thermal_kg,
        "mass_margin": M_sat_kg - M_sat_dry_before_margin_kg,
    }
    kg_per_kw = M_sat_kg / (P_peak_w / 1000.0) if P_peak_w > 0 else math.inf

    provenance = {
        "parameter_doc": PARAMETER_DOC,
        "mode": config.compute.mode,
        "accelerator_profile": profile.name,
        "accelerator_source_ids": list(profile.source_ids),
        "accelerator_mass_source": profile.mass_source,
        "storage_profile": storage.name,
        "storage_source_ids": list(storage.source_ids),
        "N_acc": n_acc,
        "N_acc_constraint_counts": acc_drivers,
        "N_acc_dominant_constraints": _dominant_constraints(acc_drivers),
        "N_storage": n_storage,
        "N_storage_constraint_counts": storage_drivers,
        "N_switch": n_switch,
        "ground_link_bandwidth_gbps": config.communication.ground_link_bandwidth_gbps,
        "isl_total_bandwidth_gbps": isl_total_bandwidth_gbps,
        "communication_cost_mode": "direct" if config.communication.use_direct_link_costs else "mass_proxy",
        "thermal_radiator_denominator_w_m2": radiator_denominator,
        "P_sunlit_avg_w": P_sunlit_avg_w if solar_denominator > 0 and storage_denominator > 0 else None,
        "power_sizing_mode": power_sizing_mode,
        "energy_parameters": {
            "installed_battery_capacity_wh": installed_battery_wh,
            "installed_solar_array_power_w": installed_solar_w,
            "compute_standby_power_w": config.power.compute_standby_power_w,
            "communication_standby_power_w": config.power.communication_standby_power_w,
            "compute_energy_j_per_token": config.power.compute_energy_j_per_token,
            "link_energy_j_per_byte": config.power.link_energy_j_per_byte,
        },
        "solar_sizing_energy_semantics": "conservative_peak_load_with_orbit_condition",
        "orbit_energy_condition": {"eclipse_duration_min": eclipse_duration_min, "sunlit_fraction": sunlit_fraction},
        "battery_sizing": {
            "eclipse_requirement_wh": eclipse_battery_req_wh,
            "minimum_battery_wh": config.power.minimum_battery_capacity_wh,
            "selected_battery_wh": E_battery_req_wh,
            "installed_battery_wh": installed_battery_wh,
            "soc_min": soc_min,
            "usable_initial_soc_fraction": usable_initial_soc_fraction,
        },
    }

    return CostResult(
        C_sat_usd=C_sat_usd,
        C_breakdown_usd=C_breakdown_usd,
        M_sat_kg=M_sat_kg,
        M_breakdown_kg=M_breakdown_kg,
        P_peak_w=P_peak_w,
        P_avg_w=P_avg_w,
        P_shadow_avg_w=P_shadow_avg_w,
        E_battery_req_wh=E_battery_req_wh,
        P_solar_req_w=P_solar_req_w,
        A_rad_m2=A_rad_m2,
        kg_per_kw=kg_per_kw,
        annualized_cost_usd_per_year=C_sat_usd * config.cost.capital_recovery_factor,
        feasible=len(violations) == 0,
        violations=violations,
        warnings=warnings,
        provenance=provenance,
    )


def _resolve_accelerator_profile(config: SingleSatelliteConfig, violations: list[str]) -> AcceleratorProfile:
    if config.compute.custom_profile is not None:
        return config.compute.custom_profile
    try:
        return accelerator_profile(config.compute.profile_name)
    except KeyError as exc:
        violations.append(str(exc))
        return AcceleratorProfile("invalid", 1.0, 1.0, 1.0, 1.0, 0.0)


def _resolve_storage_device(config: SingleSatelliteConfig, violations: list[str]) -> StorageDevice:
    if config.storage.custom_device is not None:
        return config.storage.custom_device
    try:
        return storage_profile(config.storage.profile_name)
    except KeyError as exc:
        violations.append(str(exc))
        return StorageDevice("invalid", 0.0, 0.0, 0.0, 0.0, 0.0)








def _safe_ratio(numerator: float, denominator: float, label: str, violations: list[str]) -> float:
    if numerator == 0:
        return 0.0
    if denominator <= 0:
        violations.append(f"{label} must be positive")
        return 0.0
    return numerator / denominator


def _dominant_constraints(counts: dict[str, int]) -> list[str]:
    if not counts:
        return []
    max_count = max(counts.values())
    return [name for name, value in counts.items() if value == max_count and value > 0]


def _validate_non_negative(obj: Any, prefix: str, violations: list[str]) -> None:
    if hasattr(obj, "__dataclass_fields__"):
        for item in fields(obj):
            value = getattr(obj, item.name)
            _validate_non_negative(value, f"{prefix}.{item.name}", violations)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _validate_non_negative(value, f"{prefix}.{key}", violations)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _validate_non_negative(value, f"{prefix}[{index}]", violations)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if not math.isfinite(float(obj)):
            violations.append(f"{prefix} must be finite")
        elif obj < 0:
            violations.append(f"{prefix} must be non-negative")
    _validate_fraction_bounds(obj, prefix, violations)


def _validate_fraction_bounds(obj: Any, prefix: str, violations: list[str]) -> None:
    if not isinstance(obj, (int, float)) or isinstance(obj, bool):
        return
    fraction_suffixes = (
        "_fraction",
        "dod_max",
        "eta_pmad",
        "battery_charge_efficiency",
        "battery_discharge_efficiency",
        "emissivity",
        "view_factor",
    )
    if prefix.endswith(fraction_suffixes) and obj > 1:
        violations.append(f"{prefix} must be <= 1")
