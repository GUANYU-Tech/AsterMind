"""Static configuration and resolved profiles for one identical satellite."""

from .hardware import (
    AcceleratorHardwareProfile,
    AcceleratorSpec,
    CommunicationSpec,
    ComputePayloadSpec,
    CostSpec,
    HardwareSpec,
    IntraInterconnectSpec,
    PlatformSpec,
    PowerSpec,
    ResolvedArchitecture,
    SatelliteEnergyProfile,
    SatelliteHardwareProfile,
    StorageSpec,
    ThermalSpec,
    build_satellite_energy_profile,
    build_satellite_hardware_profile,
    resolve_architecture,
)
from .single_satellite_cost import (
    SatelliteCostProfile,
    SatelliteDesignLimits,
    estimate_single_satellite_cost,
)

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
    "SatelliteCostProfile",
    "SatelliteDesignLimits",
    "SatelliteEnergyProfile",
    "SatelliteHardwareProfile",
    "StorageSpec",
    "ThermalSpec",
    "build_satellite_energy_profile",
    "build_satellite_hardware_profile",
    "estimate_single_satellite_cost",
    "resolve_architecture",
]
