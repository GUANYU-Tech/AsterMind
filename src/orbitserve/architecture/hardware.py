"""Static single-satellite hardware configuration and downstream projections."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import to_jsonable
from .single_satellite_cost import estimate_single_satellite_cost
from .single_satellite_cost.defaults import accelerator_profile, storage_profile
from .single_satellite_cost.models import (
    AcceleratorProfile as CostAcceleratorProfile,
    SatelliteDesignLimits,
    SingleSatelliteConfig,
    StorageDevice,
    load_case,
)

def _pick(data: dict[str, Any], key: str, default: Any) -> Any:
    return data[key] if key in data else default

@dataclass(frozen=True)
class AcceleratorHardwareProfile:
    name: str
    peak_compute_tflops: float
    fast_mem_gb: float
    fast_mem_bw_gbs: float
    peak_power_w: float = 0.0
    mass_kg: float = 0.0
    intra_bw_gbs: float = 0.0
    source_ids: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AcceleratorHardwareProfile":
        return cls(
            name=str(data["name"]),
            peak_compute_tflops=float(data["peak_compute_tflops"]),
            fast_mem_gb=float(data["fast_mem_gb"]),
            fast_mem_bw_gbs=float(data["fast_mem_bw_gbs"]),
            peak_power_w=float(_pick(data, "peak_power_w", 0.0)),
            mass_kg=float(_pick(data, "mass_kg", 0.0)),
            intra_bw_gbs=float(_pick(data, "intra_bw_gbs", 0.0)),
            source_ids=tuple(data.get("source_ids", ())),
            notes=str(_pick(data, "notes", "")),
        )

    @classmethod
    def from_cost_profile(cls, profile: CostAcceleratorProfile) -> "AcceleratorHardwareProfile":
        return cls(
            name=profile.name,
            peak_compute_tflops=profile.peak_compute_tflops,
            fast_mem_gb=profile.fast_mem_gb,
            fast_mem_bw_gbs=profile.fast_mem_bw_gbs,
            peak_power_w=profile.peak_power_w,
            mass_kg=profile.mass_kg,
            intra_bw_gbs=profile.intra_bw_gbs,
            source_ids=profile.source_ids,
            notes=f"derived from single_satellite_cost profile; mass_source={profile.mass_source}",
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class SatelliteHardwareProfile:
    name: str = "generic_single_satellite"
    accelerator: AcceleratorHardwareProfile = field(
        default_factory=lambda: AcceleratorHardwareProfile(
            name="generic_accelerator",
            peak_compute_tflops=1.0,
            fast_mem_gb=1.0,
            fast_mem_bw_gbs=1.0,
        )
    )
    accelerator_count: int = 1
    storage_capacity_tb: float = 0.0
    storage_bandwidth_gbs: float = 0.0
    intra_satellite_bandwidth_gbs: float | None = None
    inter_satellite_bandwidth_gbps: float = 0.0
    ground_link_bandwidth_gbps: float = 0.0
    laser_terminal_count: int = 1
    laser_terminal_bandwidth_gbps: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SatelliteHardwareProfile":
        data = data or {}
        accelerator_data = data.get("accelerator")
        if isinstance(accelerator_data, str):
            accelerator = AcceleratorHardwareProfile.from_cost_profile(accelerator_profile(accelerator_data))
        elif isinstance(accelerator_data, dict):
            accelerator = AcceleratorHardwareProfile.from_dict(accelerator_data)
        else:
            accelerator = cls().accelerator
        return cls(
            name=str(_pick(data, "name", "configured_single_satellite")),
            accelerator=accelerator,
            accelerator_count=int(_pick(data, "accelerator_count", 1)),
            storage_capacity_tb=float(_pick(data, "storage_capacity_tb", 0.0)),
            storage_bandwidth_gbs=float(_pick(data, "storage_bandwidth_gbs", 0.0)),
            intra_satellite_bandwidth_gbs=_optional_float(data.get("intra_satellite_bandwidth_gbs")),
            inter_satellite_bandwidth_gbps=float(_pick(data, "inter_satellite_bandwidth_gbps", 0.0)),
            ground_link_bandwidth_gbps=float(_pick(data, "ground_link_bandwidth_gbps", 0.0)),
            laser_terminal_count=max(0, int(_pick(data, "laser_terminal_count", 1))),
            laser_terminal_bandwidth_gbps=_optional_float(data.get("laser_terminal_bandwidth_gbps")),
            provenance=dict(data.get("provenance", {})),
        )

    @property
    def total_peak_compute_tflops(self) -> float:
        return self.accelerator_count * self.accelerator.peak_compute_tflops

    @property
    def total_fast_mem_gb(self) -> float:
        return self.accelerator_count * self.accelerator.fast_mem_gb

    @property
    def total_fast_mem_bw_gbs(self) -> float:
        return self.accelerator_count * self.accelerator.fast_mem_bw_gbs

    @property
    def total_intra_satellite_bandwidth_gbs(self) -> float:
        if self.intra_satellite_bandwidth_gbs is not None:
            return self.intra_satellite_bandwidth_gbs
        return self.accelerator_count * self.accelerator.intra_bw_gbs

    @property
    def resolved_laser_terminal_bandwidth_gbps(self) -> float:
        if self.laser_terminal_bandwidth_gbps is not None:
            return self.laser_terminal_bandwidth_gbps
        if self.laser_terminal_count <= 0:
            return 0.0
        return self.inter_satellite_bandwidth_gbps / self.laser_terminal_count


    def to_dict(self) -> dict[str, Any]:
        payload = to_jsonable(self)
        payload.update(
            {
                "total_peak_compute_tflops": self.total_peak_compute_tflops,
                "total_fast_mem_gb": self.total_fast_mem_gb,
                "resolved_laser_terminal_bandwidth_gbps": self.resolved_laser_terminal_bandwidth_gbps,
            }
        )
        return payload


@dataclass(frozen=True)
class SatelliteEnergyProfile:
    """Static, pre-launch energy configuration for one satellite."""

    battery_capacity_wh: float
    solar_array_power_w: float
    dod_max: float
    battery_eol_fraction: float
    battery_charge_efficiency: float
    battery_discharge_efficiency: float
    pv_eol_fraction: float
    eta_pmad: float
    platform_power_w: float
    compute_standby_power_w: float
    communication_standby_power_w: float
    compute_energy_j_per_token: float
    link_energy_j_per_byte: float

    def __post_init__(self) -> None:
        positive = {
            "battery_capacity_wh": self.battery_capacity_wh,
            "solar_array_power_w": self.solar_array_power_w,
            "battery_eol_fraction": self.battery_eol_fraction,
            "pv_eol_fraction": self.pv_eol_fraction,
            "eta_pmad": self.eta_pmad,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"energy.{name} must be positive")
        for name, value in {
            "dod_max": self.dod_max,
            "battery_eol_fraction": self.battery_eol_fraction,
            "battery_charge_efficiency": self.battery_charge_efficiency,
            "battery_discharge_efficiency": self.battery_discharge_efficiency,
            "pv_eol_fraction": self.pv_eol_fraction,
            "eta_pmad": self.eta_pmad,
        }.items():
            if not 0 < value <= 1:
                raise ValueError(f"energy.{name} must be in (0, 1]")
        for name, value in {
            "platform_power_w": self.platform_power_w,
            "compute_standby_power_w": self.compute_standby_power_w,
            "communication_standby_power_w": self.communication_standby_power_w,
            "compute_energy_j_per_token": self.compute_energy_j_per_token,
            "link_energy_j_per_byte": self.link_energy_j_per_byte,
        }.items():
            if value < 0:
                raise ValueError(f"energy.{name} must be non-negative")

    @classmethod
    def from_cost_config(cls, config: SingleSatelliteConfig) -> "SatelliteEnergyProfile":
        power = config.power
        if power.battery_capacity_wh is None:
            raise ValueError("power.battery_capacity_wh is required by the static architecture contract")
        if power.solar_array_power_w is None:
            raise ValueError("power.solar_array_power_w is required by the static architecture contract")
        return cls(
            battery_capacity_wh=float(power.battery_capacity_wh),
            solar_array_power_w=float(power.solar_array_power_w),
            dod_max=float(power.dod_max),
            battery_eol_fraction=float(power.battery_eol_fraction),
            battery_charge_efficiency=float(power.battery_charge_efficiency),
            battery_discharge_efficiency=float(power.battery_discharge_efficiency),
            pv_eol_fraction=float(power.pv_eol_fraction),
            eta_pmad=float(power.eta_pmad),
            platform_power_w=float(config.platform.power_w),
            compute_standby_power_w=float(power.compute_standby_power_w),
            communication_standby_power_w=float(power.communication_standby_power_w),
            compute_energy_j_per_token=float(power.compute_energy_j_per_token),
            link_energy_j_per_byte=float(power.link_energy_j_per_byte),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class SatelliteArchitectureProfile:
    """Frozen static architecture exported to Serving and Runtime."""

    architecture_id: str
    hardware: SatelliteHardwareProfile
    energy: SatelliteEnergyProfile
    _cost_limits: SatelliteDesignLimits = field(repr=False, compare=False)
    _cost_config: SingleSatelliteConfig = field(repr=False, compare=False)
    _source_case_path: str | None = field(default=None, repr=False, compare=False)

    def cost_inputs(self) -> tuple[SatelliteDesignLimits, SingleSatelliteConfig, str | None]:
        return self._cost_limits, self._cost_config, self._source_case_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "hardware": self.hardware.to_dict(),
            "energy": self.energy.to_dict(),
        }

def hardware_from_single_satellite_case(
    case_path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> SatelliteHardwareProfile:
    """Derive a shared hardware profile from an existing cost-model case."""

    path, design_limits, config = _load_single_satellite_case(case_path, base_dir)
    return hardware_from_cost_inputs(
        design_limits,
        config,
        name=f"{path.stem}_hardware",
        source="single_satellite_cost_case",
        case_path=str(path),
    )

def hardware_from_cost_inputs(
    design_limits: SatelliteDesignLimits,
    config: SingleSatelliteConfig,
    *,
    name: str,
    source: str,
    case_path: str | None = None,
) -> SatelliteHardwareProfile:
    accelerator = _resolve_cost_accelerator(config)
    storage = _resolve_cost_storage(config)
    accelerator_count = int(config.compute.accelerator_count)
    inter_satellite_bandwidth_gbps = (
        config.communication.isl_terminal_count * config.communication.isl_terminal_bandwidth_gbps
    )
    hardware = SatelliteHardwareProfile(
        name=name,
        accelerator=AcceleratorHardwareProfile.from_cost_profile(accelerator),
        accelerator_count=accelerator_count,
        storage_capacity_tb=config.storage.device_count * storage.capacity_tb,
        storage_bandwidth_gbs=config.storage.device_count * storage.bandwidth_gbs,
        intra_satellite_bandwidth_gbs=max(
            config.interconnect.switch_count * config.interconnect.switch_bw_gbs,
            accelerator_count * accelerator.intra_bw_gbs,
        ),
        inter_satellite_bandwidth_gbps=inter_satellite_bandwidth_gbps,
        ground_link_bandwidth_gbps=config.communication.ground_link_bandwidth_gbps,
        laser_terminal_count=config.communication.isl_terminal_count,
        laser_terminal_bandwidth_gbps=config.communication.isl_terminal_bandwidth_gbps,
        provenance={
            "source": source,
            "case_path": case_path,
            "design_limits": design_limits,
        },
    )
    return hardware

@dataclass(frozen=True)
class AcceleratorSpec:
    """Accelerator parameters shared by cost and performance adapters."""

    profile: str | dict[str, Any] = "h200_sxm"

    @classmethod
    def from_value(cls, value: Any) -> "AcceleratorSpec":
        if isinstance(value, AcceleratorSpec):
            return value
        if value is None:
            return cls()
        if isinstance(value, dict) and "profile" in value and len(value) == 1:
            return cls(profile=value["profile"])
        if isinstance(value, (str, dict)):
            return cls(profile=value)
        raise TypeError("hardware.accelerator must be a profile name or custom accelerator dict")

    def to_cost_profile_input(self) -> str | dict[str, Any]:
        return self.profile

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class ComputePayloadSpec:
    """Compute payload closure: quantity, utilization, and packaging/adaptation cost terms."""

    accelerator_count: int | None = None
    mode: str = "cots"
    purchase_cost_per_unit_usd: float | None = None
    space_adaptation_cost_usd: float = 0.0
    alpha_pkg: float = 0.10
    alpha_power_overhead: float = 0.10
    alpha_mass_overhead: float = 0.20

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None, *, accelerator_count: int | None = None) -> "ComputePayloadSpec":
        data = data or {}
        return cls(
            accelerator_count=_optional_int(_pick(data, "accelerator_count", accelerator_count)),
            mode=str(_pick(data, "mode", "cots")),
            purchase_cost_per_unit_usd=_optional_float(data.get("purchase_cost_per_unit_usd")),
            space_adaptation_cost_usd=float(_pick(data, "space_adaptation_cost_usd", 0.0)),
            alpha_pkg=float(_pick(data, "alpha_pkg", 0.10)),
            alpha_power_overhead=float(_pick(data, "alpha_power_overhead", 0.10)),
            alpha_mass_overhead=float(_pick(data, "alpha_mass_overhead", 0.20)),
        )

    def to_cost_dict(self, accelerator: AcceleratorSpec) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "profile": accelerator.to_cost_profile_input(),
            "accelerator_count": self.accelerator_count,
            "purchase_cost_per_unit_usd": self.purchase_cost_per_unit_usd,
            "space_adaptation_cost_usd": self.space_adaptation_cost_usd,
            "alpha_pkg": self.alpha_pkg,
            "alpha_power_overhead": self.alpha_power_overhead,
            "alpha_mass_overhead": self.alpha_mass_overhead,
        }

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class StorageSpec:
    """Persistent storage source parameters for cost and optional performance metadata."""

    profile: str | dict[str, Any] = "nvme_8tb"
    device_count: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StorageSpec":
        data = data or {}
        return cls(profile=_pick(data, "profile", "nvme_8tb"), device_count=int(_pick(data, "device_count", 1)))

    def to_cost_dict(self) -> dict[str, Any]:
        return {"profile": self.profile, "device_count": self.device_count}

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class IntraInterconnectSpec:
    """Intra-satellite interconnect and switch cost/power/mass terms."""

    alpha_intra: float = 0.10
    switch_count: int = 1
    switch_bw_gbs: float = 900.0
    switch_cost_usd: float = 0.0
    switch_power_w: float = 0.0
    switch_mass_kg: float = 0.0
    p_intra_per_gbs_w: float = 0.0
    m_intra_per_gbs_kg: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "IntraInterconnectSpec":
        data = data or {}
        return cls(
            alpha_intra=float(_pick(data, "alpha_intra", 0.10)),
            switch_count=int(_pick(data, "switch_count", 1)),
            switch_bw_gbs=float(_pick(data, "switch_bw_gbs", 900.0)),
            switch_cost_usd=float(_pick(data, "switch_cost_usd", 0.0)),
            switch_power_w=float(_pick(data, "switch_power_w", 0.0)),
            switch_mass_kg=float(_pick(data, "switch_mass_kg", 0.0)),
            p_intra_per_gbs_w=float(_pick(data, "p_intra_per_gbs_w", 0.0)),
            m_intra_per_gbs_kg=float(_pick(data, "m_intra_per_gbs_kg", 0.0)),
        )

    def to_cost_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class CommunicationSpec:
    """Ground/ISL bandwidth plus terminal cost, power, and mass terms."""

    ground_link_bandwidth_gbps: float = 1.0
    ground_link_mass_kg: float = 5.0
    ground_link_power_w: float = 20.0
    isl_terminal_count: int = 0
    isl_terminal_bandwidth_gbps: float = 10.0
    isl_terminal_mass_kg: float = 10.0
    isl_terminal_power_w: float = 100.0
    clink_allocated_usd: float = 0.0
    use_direct_link_costs: bool = False
    ground_cost_per_gbps_usd: float | None = None
    isl_cost_per_gbps_usd: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CommunicationSpec":
        data = data or {}
        return cls(
            ground_link_bandwidth_gbps=float(
                _pick(data, "ground_link_bandwidth_gbps", 1.0)
            ),
            ground_link_mass_kg=float(_pick(data, "ground_link_mass_kg", 5.0)),
            ground_link_power_w=float(_pick(data, "ground_link_power_w", 20.0)),
            isl_terminal_count=int(_pick(data, "isl_terminal_count", 0)),
            isl_terminal_bandwidth_gbps=float(
                _pick(data, "isl_terminal_bandwidth_gbps", 10.0)
            ),
            isl_terminal_mass_kg=float(_pick(data, "isl_terminal_mass_kg", 10.0)),
            isl_terminal_power_w=float(_pick(data, "isl_terminal_power_w", 100.0)),
            clink_allocated_usd=float(_pick(data, "clink_allocated_usd", 0.0)),
            use_direct_link_costs=bool(_pick(data, "use_direct_link_costs", False)),
            ground_cost_per_gbps_usd=_optional_float(data.get("ground_cost_per_gbps_usd")),
            isl_cost_per_gbps_usd=_optional_float(data.get("isl_cost_per_gbps_usd")),
        )

    @property
    def isl_total_bandwidth_gbps(self) -> float:
        return self.isl_terminal_count * self.isl_terminal_bandwidth_gbps

    def to_cost_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class PlatformSpec:
    """Satellite platform mass and power."""

    mass_kg: float = 100.0
    power_w: float = 50.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlatformSpec":
        data = data or {}
        return cls(
            mass_kg=float(_pick(data, "mass_kg", 100.0)),
            power_w=float(_pick(data, "power_w", 50.0)),
        )

    def to_cost_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class PowerSpec:
    battery_capacity_wh: float | None = None
    solar_array_power_w: float | None = None
    compute_standby_power_w: float = 0.0
    communication_standby_power_w: float = 0.0
    compute_energy_j_per_token: float = 0.5
    link_energy_j_per_byte: float = 1e-6
    dod_max: float = 0.30
    battery_eol_fraction: float = 0.80
    minimum_battery_capacity_wh: float = 1.0
    solar_specific_power_w_per_kg: float = 100.0
    battery_specific_energy_wh_per_kg: float = 200.0
    eta_pmad: float = 0.92
    battery_charge_efficiency: float = 1.0
    battery_discharge_efficiency: float = 1.0
    pv_eol_fraction: float = 0.85
    power_margin_fraction: float = 0.20
    solar_cost_per_w_usd: float | None = None
    battery_cost_per_wh_usd: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PowerSpec":
        data = data or {}
        return cls(
            battery_capacity_wh=_optional_float(data.get("battery_capacity_wh")),
            solar_array_power_w=_optional_float(data.get("solar_array_power_w")),
            compute_standby_power_w=float(_pick(data, "compute_standby_power_w", 0.0)),
            communication_standby_power_w=float(
                _pick(data, "communication_standby_power_w", 0.0)
            ),
            compute_energy_j_per_token=float(
                _pick(data, "compute_energy_j_per_token", 0.5)
            ),
            link_energy_j_per_byte=float(_pick(data, "link_energy_j_per_byte", 1e-6)),
            dod_max=float(_pick(data, "dod_max", 0.30)),
            battery_eol_fraction=float(_pick(data, "battery_eol_fraction", 0.80)),
            minimum_battery_capacity_wh=float(
                _pick(data, "minimum_battery_capacity_wh", 1.0)
            ),
            solar_specific_power_w_per_kg=float(_pick(data, "solar_specific_power_w_per_kg", 100.0)),
            battery_specific_energy_wh_per_kg=float(_pick(data, "battery_specific_energy_wh_per_kg", 200.0)),
            eta_pmad=float(_pick(data, "eta_pmad", 0.92)),
            battery_charge_efficiency=float(_pick(data, "battery_charge_efficiency", 1.0)),
            battery_discharge_efficiency=float(_pick(data, "battery_discharge_efficiency", 1.0)),
            pv_eol_fraction=float(_pick(data, "pv_eol_fraction", 0.85)),
            power_margin_fraction=float(_pick(data, "power_margin_fraction", 0.20)),
            solar_cost_per_w_usd=_optional_float(data.get("solar_cost_per_w_usd")),
            battery_cost_per_wh_usd=_optional_float(data.get("battery_cost_per_wh_usd")),
        )

    def to_cost_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class ThermalSpec:
    radiator_areal_density_kg_m2: float = 5.0
    radiator_temperature_k: float = 350.0
    emissivity: float = 0.90
    view_factor: float = 0.85
    environment_heat_flux_w_m2: float = 150.0
    thermal_cost_per_w_usd: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ThermalSpec":
        data = data or {}
        return cls(
            radiator_areal_density_kg_m2=float(_pick(data, "radiator_areal_density_kg_m2", 5.0)),
            radiator_temperature_k=float(_pick(data, "radiator_temperature_k", 350.0)),
            emissivity=float(_pick(data, "emissivity", 0.90)),
            view_factor=float(_pick(data, "view_factor", 0.85)),
            environment_heat_flux_w_m2=float(_pick(data, "environment_heat_flux_w_m2", 150.0)),
            thermal_cost_per_w_usd=_optional_float(data.get("thermal_cost_per_w_usd")),
        )

    def to_cost_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class CostClosureSpec:
    build_cost_per_kg_usd: float = 500.0
    launch_cost_per_kg_usd: float = 3360.0
    mass_margin_fraction: float = 0.20
    capital_recovery_factor: float = 0.10

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CostClosureSpec":
        data = data or {}
        return cls(
            build_cost_per_kg_usd=float(_pick(data, "build_cost_per_kg_usd", 500.0)),
            launch_cost_per_kg_usd=float(_pick(data, "launch_cost_per_kg_usd", 3360.0)),
            mass_margin_fraction=float(_pick(data, "mass_margin_fraction", 0.20)),
            capital_recovery_factor=float(_pick(data, "capital_recovery_factor", 0.10)),
        )

    def to_cost_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class HardwareSpec:
    """Canonical hardware source-of-truth for an experiment manifest."""

    mode: str = "from_cost_case"
    name: str = "configured_hardware"
    source_case_path: str | None = None
    accelerator: AcceleratorSpec = field(default_factory=AcceleratorSpec)
    compute_payload: ComputePayloadSpec = field(default_factory=ComputePayloadSpec)
    design_limits: dict[str, Any] = field(default_factory=dict)
    storage: StorageSpec = field(default_factory=StorageSpec)
    intra_interconnect: IntraInterconnectSpec = field(default_factory=IntraInterconnectSpec)
    communication: CommunicationSpec = field(default_factory=CommunicationSpec)
    platform: PlatformSpec = field(default_factory=PlatformSpec)
    power: PowerSpec = field(default_factory=PowerSpec)
    thermal: ThermalSpec = field(default_factory=ThermalSpec)
    cost: CostClosureSpec = field(default_factory=CostClosureSpec)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HardwareSpec":
        data = data or {}
        compute_payload_data = dict(data.get("compute_payload") or {})

        mode = data.get("mode")
        if mode is None:
            mode = "fixed" if _optional_int(compute_payload_data.get("accelerator_count")) is not None else "from_cost_case"

        accelerator_source = data.get("accelerator")
        if accelerator_source is None:
            accelerator_source = compute_payload_data.get("profile")

        return cls(
            mode=str(mode),
            name=str(_pick(data, "name", "configured_hardware")),
            source_case_path=data.get("source_case_path"),
            accelerator=AcceleratorSpec.from_value(accelerator_source),
            compute_payload=ComputePayloadSpec.from_dict(compute_payload_data),
            design_limits=dict(data.get("design_limits", {})),
            storage=StorageSpec.from_dict(data.get("storage")),
            intra_interconnect=IntraInterconnectSpec.from_dict(
                data.get("intra_interconnect")
            ),
            communication=CommunicationSpec.from_dict(data.get("communication")),
            platform=PlatformSpec.from_dict(data.get("platform")),
            power=PowerSpec.from_dict(data.get("power")),
            thermal=ThermalSpec.from_dict(data.get("thermal")),
            cost=CostClosureSpec.from_dict(data.get("cost")),
        )

    def to_cost_inputs(
        self,
    ) -> tuple[SatelliteDesignLimits, SingleSatelliteConfig, tuple[str, ...]]:
        """Project explicit static hardware and optional feasibility limits to cost inputs."""

        if self.mode == "fixed":
            if self.compute_payload.accelerator_count is None:
                raise ValueError("hardware.compute_payload.accelerator_count is required for fixed mode")
            if self.compute_payload.accelerator_count <= 0:
                raise ValueError("hardware.compute_payload.accelerator_count must be positive")
        config = SingleSatelliteConfig.from_dict(
            {
                "platform": self.platform.to_cost_dict(),
                "compute": self.compute_payload.to_cost_dict(self.accelerator),
                "storage": self.storage.to_cost_dict(),
                "interconnect": self.intra_interconnect.to_cost_dict(),
                "communication": self.communication.to_cost_dict(),
                "power": self.power.to_cost_dict(),
                "thermal": self.thermal.to_cost_dict(),
                "cost": self.cost.to_cost_dict(),
            }
        )
        return SatelliteDesignLimits.from_dict(self.design_limits), config, ()

    def to_performance_profile(
        self,
        *,
        design_limits: SatelliteDesignLimits | None = None,
        config: SingleSatelliteConfig | None = None,
    ) -> SatelliteHardwareProfile:
        """Project this hardware source into the existing performance-facing profile."""

        if design_limits is None or config is None:
            design_limits, config, _warnings = self.to_cost_inputs()
        return hardware_from_cost_inputs(
            design_limits,
            config,
            name=f"{self.name}_hardware",
            source=f"top_level_hardware.{self.mode}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "name": self.name,
            "source_case_path": self.source_case_path,
            "accelerator": self.accelerator.to_dict(),
            "compute_payload": self.compute_payload.to_dict(),
            "design_limits": to_jsonable(self.design_limits),
            "storage": self.storage.to_dict(),
            "intra_interconnect": self.intra_interconnect.to_dict(),
            "communication": self.communication.to_dict(),
            "platform": self.platform.to_dict(),
            "power": self.power.to_dict(),
            "thermal": self.thermal.to_dict(),
            "cost": self.cost.to_dict(),
        }

def resolve_satellite_architecture(
    hardware: HardwareSpec | None,
    *,
    base_dir: str | Path | None = None,
) -> SatelliteArchitectureProfile | None:
    """Resolve one immutable pre-launch satellite architecture."""

    if hardware is None:
        return None
    if hardware.mode == "from_cost_case":
        if hardware.source_case_path is None:
            raise ValueError("hardware.source_case_path is required in from_cost_case mode")
        path, design_limits, config = _load_single_satellite_case(
            hardware.source_case_path,
            base_dir,
        )
        profile = hardware_from_cost_inputs(
            design_limits,
            config,
            name=f"{path.stem}_hardware",
            source="single_satellite_cost_case",
            case_path=str(path),
        )
        return SatelliteArchitectureProfile(
            architecture_id=hardware.name,
            hardware=profile,
            energy=SatelliteEnergyProfile.from_cost_config(config),
            _cost_limits=design_limits,
            _cost_config=config,
            _source_case_path=str(path),
        )
    if hardware.mode != "fixed":
        raise ValueError("hardware.mode must be 'fixed' or 'from_cost_case'")
    design_limits, config, _warnings = hardware.to_cost_inputs()
    profile = hardware.to_performance_profile(
        design_limits=design_limits,
        config=config,
    )
    return SatelliteArchitectureProfile(
        architecture_id=hardware.name,
        hardware=profile,
        energy=SatelliteEnergyProfile.from_cost_config(config),
        _cost_limits=design_limits,
        _cost_config=config,
    )

def _resolve_cost_accelerator(config: SingleSatelliteConfig) -> CostAcceleratorProfile:
    if config.compute.custom_profile is not None:
        return config.compute.custom_profile
    return accelerator_profile(config.compute.profile_name)

def _resolve_cost_storage(config: SingleSatelliteConfig) -> StorageDevice:
    if config.storage.custom_device is not None:
        return config.storage.custom_device
    return storage_profile(config.storage.profile_name)

def _load_single_satellite_case(
    case_path: str | Path,
    base_dir: str | Path | None,
) -> tuple[Path, SatelliteDesignLimits, SingleSatelliteConfig]:
    path = _resolve_path(case_path, base_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    design_limits, config = load_case(data)
    return path, design_limits, config

def _resolve_path(path_text: str | Path, base_dir: str | Path | None) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    if base_dir is not None:
        candidate = Path(base_dir) / path
        if candidate.exists():
            return candidate
    return path

def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)

def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
