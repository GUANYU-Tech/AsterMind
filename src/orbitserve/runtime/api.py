"""Single public entry point for Runtime simulation."""

from __future__ import annotations

from .engine.simulator import run_runtime_engine
from .result import RuntimePerformanceResult, build_runtime_performance_result
from .setup.simulation_spec import RuntimeSimulationSpec


def check_network_backend(transport_model: str) -> None:
    """Validate an explicitly selected backend before expensive precomputation."""
    if transport_model not in {"queue", "fluid", "simgrid", "ns3", "flowsim"}:
        raise ValueError("transport_model must be queue, fluid, simgrid, ns3 or flowsim")
    if transport_model == "simgrid":
        from .engine.simgrid_transport import require_simgrid

        require_simgrid()
    if transport_model == 'ns3':
        from .engine.ns3_transport import require_ns3
        require_ns3()
    if transport_model == 'flowsim':
        from .engine.flowsim_transport import require_flowsim
        require_flowsim()


def run_simulation(simulation_spec: RuntimeSimulationSpec, *, compute_runtime=None) -> RuntimePerformanceResult:
    if not isinstance(simulation_spec, RuntimeSimulationSpec):
        raise TypeError("run_simulation requires RuntimeSimulationSpec")
    if (
        simulation_spec.network is None
        and simulation_spec.environment_state is not None
    ):
        raise ValueError("physical network configuration is required")
    if (
        simulation_spec.network is not None
        and simulation_spec.network.transport_model != "queue"
    ):
        check_network_backend(simulation_spec.network.transport_model)
    from .engine.control import compute_runtime_scope

    with compute_runtime_scope(compute_runtime):
        engine_result = run_runtime_engine(simulation_spec)
    return build_runtime_performance_result(
        simulation_spec=simulation_spec,
        engine_result=engine_result,
    )


__all__ = ["run_simulation"]
