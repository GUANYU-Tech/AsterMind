"""Static Runtime scenario and deployment preparation."""

from .model import RuntimeSimulationSpec
from .builder import build_runtime_simulation_spec

__all__ = [
    "RuntimeSimulationSpec",
    "build_runtime_simulation_spec",
]
