"""Data models for the OrbitServe single-satellite cost model.

All internal numeric fields use explicit unit suffixes. CostResult.to_dict()
uses one canonical, unit-explicit output schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _pick(data: dict[str, Any], key: str, default: Any) -> Any:
    return data[key] if key in data else default






@dataclass(frozen=True)
class SatelliteDesignLimits:
    """Physical pre-launch limits; never workload or cloud-service demand."""

    payload_power_limit_w: float | None = None
    satellite_mass_limit_kg: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SatelliteDesignLimits":
        data = data or {}
        return cls(
            payload_power_limit_w=_optional_float(data.get("payload_power_limit_w")),
            satellite_mass_limit_kg=_optional_float(data.get("satellite_mass_limit_kg")),
        )


@dataclass(frozen=True)
class AcceleratorProfile:
    name: str
    peak_compute_tflops: float
    fast_mem_gb: float
    fast_mem_bw_gbs: float
    peak_power_w: float
    mass_kg: float
    intra_bw_gbs: float = 0.0
    mass_source: str = "proxy"
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
            intra_bw_gbs=float(_pick(data, "intra_bw_gbs", 0.0)),
            mass_source=str(_pick(data, "mass_source", "proxy")),
            source_ids=tuple(data.get("source_ids", ())),
        )


@dataclass(frozen=True)
class PlatformConfig:
    mass_kg: float = 100.0
    power_w: float = 50.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "PlatformConfig":
        data = data or {}
        return cls(
            mass_kg=float(_pick(data, "mass_kg", 100.0)),
            power_w=float(_pick(data, "power_w", 50.0)),
        )


@dataclass(frozen=True)
class ComputeConfig:
    mode: str = "cots"
    profile_name: str = "h200_sxm"
    accelerator_count: int = 1
    custom_profile: AcceleratorProfile | None = None
    purchase_cost_per_unit_usd: float | None = None
    space_adaptation_cost_usd: float = 0.0
    alpha_pkg: float = 0.10
    alpha_power_overhead: float = 0.10
    alpha_mass_overhead: float = 0.20

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ComputeConfig":
        data = data or {}
        profile = data.get("profile", "h200_sxm")
        custom_profile = None
        profile_name = str(profile)
        if isinstance(profile, dict):
            custom_profile = AcceleratorProfile.from_dict(profile)
            profile_name = custom_profile.name
        return cls(
            mode=str(_pick(data, "mode", "cots")),
            profile_name=profile_name,
            accelerator_count=int(_pick(data, "accelerator_count", 1)),
            custom_profile=custom_profile,
            purchase_cost_per_unit_usd=_optional_float(data.get("purchase_cost_per_unit_usd")),
            space_adaptation_cost_usd=float(_pick(data, "space_adaptation_cost_usd", 0.0)),
            alpha_pkg=float(_pick(data, "alpha_pkg", 0.10)),
            alpha_power_overhead=float(_pick(data, "alpha_power_overhead", 0.10)),
            alpha_mass_overhead=float(_pick(data, "alpha_mass_overhead", 0.20)),
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
            source_ids=tuple(data.get("source_ids", ())),
        )


@dataclass(frozen=True)
class StorageConfig:
    profile_name: str = "nvme_8tb"
    device_count: int = 1
    custom_device: StorageDevice | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "StorageConfig":
        data = data or {}
        profile = data.get("profile", "nvme_8tb")
        custom_device = None
        profile_name = str(profile)
        if isinstance(profile, dict):
            custom_device = StorageDevice.from_dict(profile)
            profile_name = custom_device.name
        return cls(profile_name=profile_name, device_count=int(_pick(data, "device_count", 1)), custom_device=custom_device)


@dataclass(frozen=True)
class InterconnectConfig:
    alpha_intra: float = 0.10
    switch_count: int = 1
    switch_bw_gbs: float = 900.0
    switch_cost_usd: float = 0.0
    switch_power_w: float = 0.0
    switch_mass_kg: float = 0.0
    p_intra_per_gbs_w: float = 0.0
    m_intra_per_gbs_kg: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "InterconnectConfig":
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


@dataclass(frozen=True)
class CommunicationConfig:
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
    def from_dict(cls, data: dict[str, Any] | None) -> "CommunicationConfig":
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

@dataclass(frozen=True)
class PowerConfig:
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
    def from_dict(cls, data: dict[str, Any] | None) -> "PowerConfig":
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


@dataclass(frozen=True)
class ThermalConfig:
    radiator_areal_density_kg_m2: float = 5.0
    radiator_temperature_k: float = 350.0
    emissivity: float = 0.90
    view_factor: float = 0.85
    environment_heat_flux_w_m2: float = 150.0
    thermal_cost_per_w_usd: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ThermalConfig":
        data = data or {}
        return cls(
            radiator_areal_density_kg_m2=float(_pick(data, "radiator_areal_density_kg_m2", 5.0)),
            radiator_temperature_k=float(_pick(data, "radiator_temperature_k", 350.0)),
            emissivity=float(_pick(data, "emissivity", 0.90)),
            view_factor=float(_pick(data, "view_factor", 0.85)),
            environment_heat_flux_w_m2=float(_pick(data, "environment_heat_flux_w_m2", 150.0)),
            thermal_cost_per_w_usd=_optional_float(data.get("thermal_cost_per_w_usd")),
        )


@dataclass(frozen=True)
class CostConfig:
    build_cost_per_kg_usd: float = 500.0
    launch_cost_per_kg_usd: float = 3360.0
    mass_margin_fraction: float = 0.20
    capital_recovery_factor: float = 0.10

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CostConfig":
        data = data or {}
        return cls(
            build_cost_per_kg_usd=float(_pick(data, "build_cost_per_kg_usd", 500.0)),
            launch_cost_per_kg_usd=float(_pick(data, "launch_cost_per_kg_usd", 3360.0)),
            mass_margin_fraction=float(_pick(data, "mass_margin_fraction", 0.20)),
            capital_recovery_factor=float(_pick(data, "capital_recovery_factor", 0.10)),
        )


@dataclass(frozen=True)
class SingleSatelliteConfig:
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    interconnect: InterconnectConfig = field(default_factory=InterconnectConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    power: PowerConfig = field(default_factory=PowerConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    cost: CostConfig = field(default_factory=CostConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SingleSatelliteConfig":
        data = data or {}
        return cls(
            platform=PlatformConfig.from_dict(data.get("platform")),
            compute=ComputeConfig.from_dict(data.get("compute")),
            storage=StorageConfig.from_dict(data.get("storage")),
            interconnect=InterconnectConfig.from_dict(data.get("interconnect")),
            communication=CommunicationConfig.from_dict(data.get("communication")),
            power=PowerConfig.from_dict(data.get("power")),
            thermal=ThermalConfig.from_dict(data.get("thermal")),
            cost=CostConfig.from_dict(data.get("cost")),
        )


@dataclass(frozen=True)
class CostResult:
    C_sat_usd: float
    C_breakdown_usd: dict[str, float]
    M_sat_kg: float
    M_breakdown_kg: dict[str, float]
    P_peak_w: float
    P_avg_w: float
    P_shadow_avg_w: float
    E_battery_req_wh: float
    P_solar_req_w: float
    A_rad_m2: float
    kg_per_kw: float
    annualized_cost_usd_per_year: float
    feasible: bool
    violations: list[str]
    warnings: list[str]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_case(data: dict[str, Any]) -> tuple[SatelliteDesignLimits, SingleSatelliteConfig]:
    """Load the public JSON case shape into API objects."""

    return SatelliteDesignLimits.from_dict(data.get("design_limits")), SingleSatelliteConfig.from_dict(data)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
