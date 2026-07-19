"""Orbit geometry, visibility, sunlight, and trace contracts."""

from .constellation import ConstellationDesign, FormationCluster, OrbitalSet
from .sgl_access_windows import (
    active_sgl_access_window,
    has_sgl_access_windows,
    sgl_window_link_sample,
)
from .simulation import SimulationConfig, SimulationResult, run_simulation
from .simulation_traces import (
    CIRCULAR_GEOMETRY_TRACE,
    IDEAL_FULL_VISIBILITY,
    SURROGATE_ORBIT_TRACE,
    TRACE_BACKED_RUNTIME_SURROGATE_SCOPE,
    RuntimeTraceBundle,
    RuntimeTraceManifest,
    load_surrogate_trace_bundle,
    normalize_trace_scope,
    resolve_trace_path,
)

__all__ = [
    "CIRCULAR_GEOMETRY_TRACE",
    "ConstellationDesign",
    "FormationCluster",
    "IDEAL_FULL_VISIBILITY",
    "OrbitalSet",
    "RuntimeTraceBundle",
    "RuntimeTraceManifest",
    "SURROGATE_ORBIT_TRACE",
    "SimulationConfig",
    "SimulationResult",
    "TRACE_BACKED_RUNTIME_SURROGATE_SCOPE",
    "active_sgl_access_window",
    "has_sgl_access_windows",
    "load_surrogate_trace_bundle",
    "normalize_trace_scope",
    "resolve_trace_path",
    "run_simulation",
    "sgl_window_link_sample",
]
