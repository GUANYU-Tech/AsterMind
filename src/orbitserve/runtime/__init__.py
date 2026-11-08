"""Public API for deterministic OrbitServe Runtime simulation."""

from .api import check_network_backend, run_simulation
from .communication import (
    CommunicationExecutor,
    InterSatelliteFlowBackend,
    IntraSatelliteSurrogateBackend,
    Ns3PacketBackend,
    PhysicalTransfer,
)
from .network import NetworkStateFrame
from .timeline import CellManager, GlobalScheduling, GlobalTimeline
from .astermind import export_cell_network
from .contracts import RuntimeStrategySpec
from .engine.topology import candidate_links
from .engine.control import runtime_control
from .result import RuntimePerformanceResult
from .setup.network_spec import LinkProfile, NetworkNodeSpec, NetworkSpec
from .setup.radio_capacity import STARLINK_RADIO_PROFILES
from .setup.simulation_spec import RuntimeSimulationSpec
from .setup.faults import RUNTIME_NETWORK_FIELDS, physical_network_settings
from .setup.islsim import freeze_profiles as freeze_network_profiles, resolve_config as resolve_islsim_config
from .setup.islsim_config import prepare_config as prepare_islsim_config
from .setup.sun_geometry import SUN_GEOMETRY_VERSION
from .setup.network_geometry import sun_positions_from_trace
from .state.network import NetworkState, NetworkUnavailable

__all__ = [
    "CellManager", "CommunicationExecutor", "GlobalScheduling", "GlobalTimeline",
    "InterSatelliteFlowBackend", "IntraSatelliteSurrogateBackend", "NetworkStateFrame",
    "Ns3PacketBackend", "PhysicalTransfer",
    "freeze_network_profiles",
    "prepare_islsim_config",
    "resolve_islsim_config",
    "SUN_GEOMETRY_VERSION",
    "sun_positions_from_trace",
    "RUNTIME_NETWORK_FIELDS",
    "physical_network_settings",
    "runtime_control",
    "STARLINK_RADIO_PROFILES",
    "LinkProfile",
    "NetworkNodeSpec",
    "NetworkSpec",
    "NetworkState",
    "NetworkUnavailable",
    "RuntimePerformanceResult",
    "RuntimeSimulationSpec",
    "RuntimeStrategySpec",
    "candidate_links",
    "check_network_backend",
    "export_cell_network",
    "run_simulation",
]
