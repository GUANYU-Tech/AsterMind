"""Static, identical per-satellite hardware, energy-sizing, and cost models."""

from .hardware import (
    HardwareSpec,
    SatelliteArchitectureProfile,
    SatelliteEnergyProfile,
    SatelliteHardwareProfile,
    resolve_satellite_architecture,
)
from .single_satellite_cost import (
    CostResult,
    SatelliteDesignLimits,
    SingleSatelliteConfig,
    estimate_single_satellite_cost,
)

__all__ = [
    "HardwareSpec",
    "CostResult",
    "SatelliteArchitectureProfile",
    "SatelliteEnergyProfile",
    "SatelliteDesignLimits",
    "SatelliteHardwareProfile",
    "SingleSatelliteConfig",
    "estimate_single_satellite_cost",
    "resolve_satellite_architecture",
]
