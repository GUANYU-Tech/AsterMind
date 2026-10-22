"""Public API for deterministic OrbitServe Runtime simulation."""

from .api import run_simulation, check_network_backend
from .contracts import RuntimeStrategySpec
from .result import RuntimePerformanceResult
from .setup.simulation_spec import RuntimeSimulationSpec

from .setup.network_spec import LinkProfile, NetworkNodeSpec, NetworkSpec
from .setup.radio_capacity import STARLINK_RADIO_PROFILES
from .state.network import NetworkState, NetworkUnavailable
from .engine.topology import candidate_links
from .astermind import export_cell_network

__all__ = [
    "LinkProfile", "NetworkNodeSpec", "NetworkSpec", "NetworkState",
    "NetworkUnavailable", "candidate_links",
    "RuntimePerformanceResult",
    "RuntimeSimulationSpec",
    "RuntimeStrategySpec",
    "run_simulation",
    "check_network_backend",
    "export_cell_network",
    "STARLINK_RADIO_PROFILES",
]
