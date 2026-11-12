"""Validation of explicitly selected network backends."""

from __future__ import annotations

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


__all__ = ["check_network_backend"]
