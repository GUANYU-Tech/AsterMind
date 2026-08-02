"""Single public entry point for Runtime simulation."""

from __future__ import annotations

from .engine.simulator import run_runtime_engine
from .result import RuntimePerformanceResult, build_runtime_performance_result
from .setup.simulation_spec import RuntimeSimulationSpec


def run_runtime_simulation(simulation_spec: RuntimeSimulationSpec) -> RuntimePerformanceResult:
    if not isinstance(simulation_spec, RuntimeSimulationSpec):
        raise TypeError("run_runtime_simulation requires RuntimeSimulationSpec")
    if (
        simulation_spec.strategy.ingress_selection == "communication_satellite"
        and not simulation_spec.communication_ingress.configured
    ):
        raise ValueError("communication_satellite ingress requires altitude and uplink/downlink bandwidth")
    engine_result = run_runtime_engine(simulation_spec)
    return build_runtime_performance_result(
        simulation_spec=simulation_spec,
        engine_result=engine_result,
    )


__all__ = ["run_runtime_simulation"]
