"""Canonical static configuration and output profiles for one satellite."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..common import to_jsonable
from .single_satellite_cost.defaults import accelerator_profile, storage_profile
from .single_satellite_cost.models import (
    AcceleratorProfile,
    SatelliteCostProfile,
    SatelliteDesignLimits,
    StorageDevice,
)


def _pick(data: dict[str, Any], key: str, default: Any) -> Any:
    return data[key] if key in data else default


@dataclass(frozen=True)
class AcceleratorSpec:
    profile: str | dict[str, Any] = "h200_sxm"

    @classmethod
    def from_value(cls, value: Any) -> "AcceleratorSpec":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, dict) and set(value) == {"profile"}:
            return cls(profile=value["profile"])
        if isinstance(value, (str, dict)):
            return cls(profile=value)
        raise TypeError("hardware.accelerator must be a profile name or a complete custom profile")

    def resolve(self) -> AcceleratorProfile:
        if isinstance(self.profile, str):
            return accelerator_profile(self.profile)
        return AcceleratorProfile.from_dict(self.profile)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class ComputePayloadSpec:
    accelerator_count: int = 1
    purchase_cost_per_unit_usd: float | None = None
    space_adaptation_cost_usd: float = 0.0
    packaging_cost_fraction: float = 0.10
    power_overhead_fraction: float = 0.10
    mass_overhead_fraction: float = 0.20

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ComputePayloadSpec":
        data = data or {}
        _reject_unknown(data, cls, "hardware.compute_payload")
        return cls(
            accelerator_count=int(_pick(data, "accelerator_count", 1)),
            purchase_cost_per_unit_usd=_optional_float(data.get("purchase_cost_per_unit_usd")),
            space_adaptation_cost_usd=float(_pick(data, "space_adaptation_cost_usd", 0.0)),
            packaging_cost_fraction=float(_pick(data, "packaging_cost_fraction", 0.10)),
            power_overhead_fraction=float(_pick(data, "power_overhead_fraction", 0.10)),
            mass_overhead_fraction=float(_pick(data, "mass_overhead_fraction", 0.20)),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class StorageSpec:
    profile: str | dict[str, Any] = "nvme_8tb"
    device_count: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StorageSpec":
        data = data or {}
        _reject_unknown(data, cls, "hardware.storage")
        return cls(
            profile=_pick(data, "profile", "nvme_8tb"),
            device_count=int(_pick(data, "device_count", 1)),
        )

    def resolve(self) -> StorageDevice:
        if isinstance(self.profile, str):
            return storage_profile(self.profile)
        return StorageDevice.from_dict(self.profile)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class IntraInterconnectSpec:
    cost_fraction: float = 0.10
    switch_count: int = 1
    switch_bandwidth_gbs: float = 900.0
    switch_cost_usd: float = 0.0
    switch_power_w: float = 0.0
    switch_mass_kg: float = 0.0
    power_per_gbs_w: float = 0.0
    mass_per_gbs_kg: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "IntraInterconnectSpec":
        data = data or {}
        _reject_unknown(data, cls, "hardware.intra_interconnect")
        return cls(
            cost_fraction=float(_pick(data, "cost_fraction", 0.10)),
            switch_count=int(_pick(data, "switch_count", 1)),
            switch_bandwidth_gbs=float(_pick(data, "switch_bandwidth_gbs", 900.0)),
            switch_cost_usd=float(_pick(data, "switch_cost_usd", 0.0)),
            switch_power_w=float(_pick(data, "switch_power_w", 0.0)),
            switch_mass_kg=float(_pick(data, "switch_mass_kg", 0.0)),
            power_per_gbs_w=float(_pick(data, "power_per_gbs_w", 0.0)),
            mass_per_gbs_kg=float(_pick(data, "mass_per_gbs_kg", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class CommunicationSpec:
    ground_link_bandwidth_gbps: float = 1.0
    ground_link_mass_kg: float = 5.0
    ground_link_power_w: float = 20.0
    isl_terminal_count: int = 0
    isl_terminal_bandwidth_gbps: float = 10.0
    isl_terminal_mass_kg: float = 10.0
    isl_terminal_power_w: float = 100.0
    allocated_cost_usd: float = 0.0
    ground_cost_per_gbps_usd: float | None = None
    isl_cost_per_gbps_usd: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CommunicationSpec":
        data = data or {}
        _reject_unknown(data, cls, "hardware.communication")
        return cls(
            ground_link_bandwidth_gbps=float(_pick(data, "ground_link_bandwidth_gbps", 1.0)),
            ground_link_mass_kg=float(_pick(data, "ground_link_mass_kg", 5.0)),
            ground_link_power_w=float(_pick(data, "ground_link_power_w", 20.0)),
            isl_terminal_count=int(_pick(data, "isl_terminal_count", 0)),
            isl_terminal_bandwidth_gbps=float(_pick(data, "isl_terminal_bandwidth_gbps", 10.0)),
            isl_terminal_mass_kg=float(_pick(data, "isl_terminal_mass_kg", 10.0)),
            isl_terminal_power_w=float(_pick(data, "isl_terminal_power_w", 100.0)),
            allocated_cost_usd=float(_pick(data, "allocated_cost_usd", 0.0)),
            ground_cost_per_gbps_usd=_optional_float(data.get("ground_cost_per_gbps_usd")),
            isl_cost_per_gbps_usd=_optional_float(data.get("isl_cost_per_gbps_usd")),
        )

    @property
    def isl_total_bandwidth_gbps(self) -> float:
        return self.isl_terminal_count * self.isl_terminal_bandwidth_gbps

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class PlatformSpec:
    mass_kg: float = 100.0
    idle_power_w: float = 50.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlatformSpec":
        data = data or {}
        _reject_unknown(data, cls, "hardware.platform")
        return cls(
            mass_kg=float(_pick(data, "mass_kg", 100.0)),
            idle_power_w=float(_pick(data, "idle_power_w", 50.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class PowerSpec:
    """Static power-system configuration fixed before launch."""

    battery_capacity_wh: float = 1000.0
    solar_array_power_w: float = 1000.0
    initial_soc: float = 0.80
    maximum_depth_of_discharge: float = 0.30
    battery_eol_fraction: float = 0.80
    battery_charge_efficiency: float = 1.0
    battery_discharge_efficiency: float = 1.0
    pv_eol_fraction: float = 0.85
    power_management_efficiency: float = 0.92
    compute_standby_power_w: float = 0.0
    communication_standby_power_w: float = 0.0
    compute_energy_j_per_token: float = 0.5
    link_energy_j_per_byte: float = 1e-6
    solar_specific_power_w_per_kg: float = 100.0
    battery_specific_energy_wh_per_kg: float = 200.0
    solar_cost_per_w_usd: float | None = None
    battery_cost_per_wh_usd: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PowerSpec":
        data = data or {}
        _reject_unknown(data, cls, "hardware.power")
        return cls(
            battery_capacity_wh=float(_pick(data, "battery_capacity_wh", 1000.0)),
            solar_array_power_w=float(_pick(data, "solar_array_power_w", 1000.0)),
            initial_soc=float(_pick(data, "initial_soc", 0.80)),
            maximum_depth_of_discharge=float(_pick(data, "maximum_depth_of_discharge", 0.30)),
            battery_eol_fraction=float(_pick(data, "battery_eol_fraction", 0.80)),
            battery_charge_efficiency=float(_pick(data, "battery_charge_efficiency", 1.0)),
            battery_discharge_efficiency=float(_pick(data, "battery_discharge_efficiency", 1.0)),
            pv_eol_fraction=float(_pick(data, "pv_eol_fraction", 0.85)),
            power_management_efficiency=float(_pick(data, "power_management_efficiency", 0.92)),
            compute_standby_power_w=float(_pick(data, "compute_standby_power_w", 0.0)),
            communication_standby_power_w=float(_pick(data, "communication_standby_power_w", 0.0)),
            compute_energy_j_per_token=float(_pick(data, "compute_energy_j_per_token", 0.5)),
            link_energy_j_per_byte=float(_pick(data, "link_energy_j_per_byte", 1e-6)),
            solar_specific_power_w_per_kg=float(_pick(data, "solar_specific_power_w_per_kg", 100.0)),
            battery_specific_energy_wh_per_kg=float(_pick(data, "battery_specific_energy_wh_per_kg", 200.0)),
            solar_cost_per_w_usd=_optional_float(data.get("solar_cost_per_w_usd")),
            battery_cost_per_wh_usd=_optional_float(data.get("battery_cost_per_wh_usd")),
        )

    @property
    def minimum_soc(self) -> float:
        return 1.0 - self.maximum_depth_of_discharge

    @property
    def charge_rate_w(self) -> float:
        return self.solar_array_power_w * self.pv_eol_fraction * self.power_management_efficiency

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
        _reject_unknown(data, cls, "hardware.thermal")
        return cls(**{name: float(_pick(data, name, default)) if default is not None else _optional_float(data.get(name)) for name, default in {
            "radiator_areal_density_kg_m2": 5.0,
            "radiator_temperature_k": 350.0,
            "emissivity": 0.90,
            "view_factor": 0.85,
            "environment_heat_flux_w_m2": 150.0,
            "thermal_cost_per_w_usd": None,
        }.items()})

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class CostSpec:
    build_cost_per_kg_usd: float = 500.0
    launch_cost_per_kg_usd: float = 3360.0
    mass_margin_fraction: float = 0.20
    capital_recovery_factor: float = 0.10

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CostSpec":
        data = data or {}
        _reject_unknown(data, cls, "hardware.cost")
        return cls(
            build_cost_per_kg_usd=float(_pick(data, "build_cost_per_kg_usd", 500.0)),
            launch_cost_per_kg_usd=float(_pick(data, "launch_cost_per_kg_usd", 3360.0)),
            mass_margin_fraction=float(_pick(data, "mass_margin_fraction", 0.20)),
            capital_recovery_factor=float(_pick(data, "capital_recovery_factor", 0.10)),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class HardwareSpec:
    """The only configurable Architecture input for one identical satellite."""

    name: str = "configured_satellite"
    accelerator: AcceleratorSpec = field(default_factory=AcceleratorSpec)
    compute_payload: ComputePayloadSpec = field(default_factory=ComputePayloadSpec)
    storage: StorageSpec = field(default_factory=StorageSpec)
    intra_interconnect: IntraInterconnectSpec = field(default_factory=IntraInterconnectSpec)
    communication: CommunicationSpec = field(default_factory=CommunicationSpec)
    platform: PlatformSpec = field(default_factory=PlatformSpec)
    power: PowerSpec = field(default_factory=PowerSpec)
    thermal: ThermalSpec = field(default_factory=ThermalSpec)
    cost: CostSpec = field(default_factory=CostSpec)
    design_limits: SatelliteDesignLimits = field(default_factory=SatelliteDesignLimits)
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "HardwareSpec":
        data = data or {}
        unknown = sorted(set(data) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unsupported HardwareSpec fields: " + ", ".join(unknown))
        return cls(
            name=str(_pick(data, "name", "configured_satellite")),
            accelerator=AcceleratorSpec.from_value(data.get("accelerator")),
            compute_payload=ComputePayloadSpec.from_dict(data.get("compute_payload")),
            storage=StorageSpec.from_dict(data.get("storage")),
            intra_interconnect=IntraInterconnectSpec.from_dict(data.get("intra_interconnect")),
            communication=CommunicationSpec.from_dict(data.get("communication")),
            platform=PlatformSpec.from_dict(data.get("platform")),
            power=PowerSpec.from_dict(data.get("power")),
            thermal=ThermalSpec.from_dict(data.get("thermal")),
            cost=CostSpec.from_dict(data.get("cost")),
            design_limits=SatelliteDesignLimits.from_dict(data.get("design_limits")),
            provenance=dict(data.get("provenance", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class AcceleratorHardwareProfile:
    name: str
    peak_compute_tflops: float
    fast_mem_gb: float
    fast_mem_bw_gbs: float
    peak_power_w: float
    mass_kg: float
    intra_bw_gbs: float
    source_ids: tuple[str, ...] = ()
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AcceleratorHardwareProfile":
        return cls(
            name=str(data["name"]),
            peak_compute_tflops=float(data["peak_compute_tflops"]),
            fast_mem_gb=float(data["fast_mem_gb"]),
            fast_mem_bw_gbs=float(data["fast_mem_bw_gbs"]),
            peak_power_w=float(data.get("peak_power_w", 0.0)),
            mass_kg=float(data.get("mass_kg", 0.0)),
            intra_bw_gbs=float(data.get("intra_bw_gbs", 0.0)),
            source_ids=tuple(str(item) for item in data.get("source_ids", ())),
            notes=str(data.get("notes", "")),
        )

    @classmethod
    def from_catalog(cls, profile: AcceleratorProfile) -> "AcceleratorHardwareProfile":
        return cls(
            name=profile.name,
            peak_compute_tflops=profile.peak_compute_tflops,
            fast_mem_gb=profile.fast_mem_gb,
            fast_mem_bw_gbs=profile.fast_mem_bw_gbs,
            peak_power_w=profile.peak_power_w,
            mass_kg=profile.mass_kg,
            intra_bw_gbs=profile.intra_bw_gbs,
            source_ids=profile.source_ids,
            notes=f"mass_source={profile.mass_source}",
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class SatelliteHardwareProfile:
    """Stable compute, storage, and communication view consumed by Serving."""

    name: str
    accelerator: AcceleratorHardwareProfile
    accelerator_count: int
    storage_capacity_tb: float
    storage_bandwidth_gbs: float
    intra_satellite_bandwidth_gbs: float
    inter_satellite_bandwidth_gbps: float
    ground_link_bandwidth_gbps: float
    laser_terminal_count: int
    laser_terminal_bandwidth_gbps: float
    provenance: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SatelliteHardwareProfile":
        accelerator = data.get("accelerator")
        if isinstance(accelerator, str):
            resolved = AcceleratorHardwareProfile.from_catalog(accelerator_profile(accelerator))
        elif isinstance(accelerator, dict):
            resolved = AcceleratorHardwareProfile.from_dict(accelerator)
        else:
            raise ValueError("SatelliteHardwareProfile.accelerator is required")
        return cls(
            name=str(data.get("name", "configured_satellite")),
            accelerator=resolved,
            accelerator_count=int(data["accelerator_count"]),
            storage_capacity_tb=float(data.get("storage_capacity_tb", 0.0)),
            storage_bandwidth_gbs=float(data.get("storage_bandwidth_gbs", 0.0)),
            intra_satellite_bandwidth_gbs=float(data.get("intra_satellite_bandwidth_gbs", 0.0)),
            inter_satellite_bandwidth_gbps=float(data.get("inter_satellite_bandwidth_gbps", 0.0)),
            ground_link_bandwidth_gbps=float(data.get("ground_link_bandwidth_gbps", 0.0)),
            laser_terminal_count=int(data.get("laser_terminal_count", 0)),
            laser_terminal_bandwidth_gbps=float(data.get("laser_terminal_bandwidth_gbps", 0.0)),
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
        return self.intra_satellite_bandwidth_gbs

    @property
    def resolved_laser_terminal_bandwidth_gbps(self) -> float:
        return self.laser_terminal_bandwidth_gbps

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class SatelliteEnergyProfile:
    battery_capacity_wh: float
    battery_capacity_j: float
    battery_eol_fraction: float
    initial_soc: float
    soc_min: float
    battery_charge_efficiency: float
    battery_discharge_efficiency: float
    solar_array_power_w: float
    charge_rate_w: float
    platform_idle_power_w: float
    compute_standby_power_w: float
    communication_standby_power_w: float
    idle_power_w: float
    compute_energy_j_per_token: float
    link_energy_j_per_byte: float
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class ResolvedArchitecture:
    hardware_profile: SatelliteHardwareProfile
    energy_profile: SatelliteEnergyProfile
    cost_profile: SatelliteCostProfile

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def build_satellite_hardware_profile(spec: HardwareSpec) -> SatelliteHardwareProfile:
    accelerator = spec.accelerator.resolve()
    storage = spec.storage.resolve()
    accelerator_count = spec.compute_payload.accelerator_count
    return SatelliteHardwareProfile(
        name=spec.name,
        accelerator=AcceleratorHardwareProfile.from_catalog(accelerator),
        accelerator_count=accelerator_count,
        intra_satellite_bandwidth_gbs=max(
            spec.intra_interconnect.switch_count * spec.intra_interconnect.switch_bandwidth_gbs,
            accelerator_count * accelerator.intra_bw_gbs,
        ),
        storage_capacity_tb=spec.storage.device_count * storage.capacity_tb,
        storage_bandwidth_gbs=spec.storage.device_count * storage.bandwidth_gbs,
        ground_link_bandwidth_gbps=spec.communication.ground_link_bandwidth_gbps,
        inter_satellite_bandwidth_gbps=spec.communication.isl_total_bandwidth_gbps,
        laser_terminal_count=spec.communication.isl_terminal_count,
        laser_terminal_bandwidth_gbps=spec.communication.isl_terminal_bandwidth_gbps,
        provenance={"source": "HardwareSpec", **spec.provenance},
    )


def build_satellite_energy_profile(spec: HardwareSpec) -> SatelliteEnergyProfile:
    power = spec.power
    idle_power_w = (
        spec.platform.idle_power_w
        + power.compute_standby_power_w
        + power.communication_standby_power_w
    )
    return SatelliteEnergyProfile(
        battery_capacity_wh=power.battery_capacity_wh,
        battery_capacity_j=power.battery_capacity_wh * 3600.0,
        battery_eol_fraction=power.battery_eol_fraction,
        initial_soc=power.initial_soc,
        soc_min=power.minimum_soc,
        battery_charge_efficiency=power.battery_charge_efficiency,
        battery_discharge_efficiency=power.battery_discharge_efficiency,
        solar_array_power_w=power.solar_array_power_w,
        charge_rate_w=power.charge_rate_w,
        platform_idle_power_w=spec.platform.idle_power_w,
        compute_standby_power_w=power.compute_standby_power_w,
        communication_standby_power_w=power.communication_standby_power_w,
        idle_power_w=idle_power_w,
        compute_energy_j_per_token=power.compute_energy_j_per_token,
        link_energy_j_per_byte=power.link_energy_j_per_byte,
        provenance={"source": "HardwareSpec.power", **spec.provenance},
    )


def resolve_architecture(spec: HardwareSpec | dict[str, Any]) -> ResolvedArchitecture:
    if isinstance(spec, dict):
        spec = HardwareSpec.from_dict(spec)
    if not isinstance(spec, HardwareSpec):
        raise TypeError("resolve_architecture requires HardwareSpec")
    from .single_satellite_cost.calculator import estimate_single_satellite_cost

    return ResolvedArchitecture(
        hardware_profile=build_satellite_hardware_profile(spec),
        energy_profile=build_satellite_energy_profile(spec),
        cost_profile=estimate_single_satellite_cost(spec),
    )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _reject_unknown(data: dict[str, Any], cls: type[Any], prefix: str) -> None:
    unknown = sorted(set(data) - set(cls.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unsupported {prefix} fields: " + ", ".join(unknown))


__all__ = [
    "AcceleratorHardwareProfile",
    "AcceleratorSpec",
    "CommunicationSpec",
    "ComputePayloadSpec",
    "CostSpec",
    "HardwareSpec",
    "IntraInterconnectSpec",
    "PlatformSpec",
    "PowerSpec",
    "ResolvedArchitecture",
    "SatelliteEnergyProfile",
    "SatelliteHardwareProfile",
    "StorageSpec",
    "ThermalSpec",
    "build_satellite_energy_profile",
    "build_satellite_hardware_profile",
    "resolve_architecture",
]
