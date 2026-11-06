"""Explicit ns-3 packet driver for the Communication packet backend.

This adapter reuses the native point-to-point worker from dynamic-links.  It
never falls back to an analytical or SimGrid model when ns-3 is unavailable.
The first integration executes one routed transfer at a time; shared concurrent
packet arbitration remains inside the native worker interface and will be
enabled when the global communication pump submits multiple active transfers.
"""

from __future__ import annotations

from importlib.util import find_spec
import math

from ..communication import RouteHop


class Ns3Unavailable(RuntimeError):
    pass


def require_ns3() -> None:
    if find_spec("ns") is None:
        raise Ns3Unavailable(
            "packet mode requires the optional ns3 Python package and a C++ compiler"
        )


class Ns3PointToPointDriver:
    """Run a routed transfer through native ns-3 point-to-point devices."""

    def __init__(self) -> None:
        require_ns3()

    def start_transfer(
        self,
        transfer_id: str,
        hops: tuple[RouteHop, ...],
        size_bytes: float,
        start_s: float,
    ) -> float:
        if not transfer_id or not hops:
            raise ValueError("ns-3 transfer requires an ID and a nonempty route")
        if not math.isfinite(size_bytes) or size_bytes < 0:
            raise ValueError("ns-3 transfer size must be finite and nonnegative")
        if not math.isfinite(start_s) or start_s < 0:
            raise ValueError("ns-3 start time must be finite and nonnegative")
        from .ns3_worker import NativeEngine

        engine = NativeEngine()
        try:
            if start_s:
                engine.call("advance", start_s)
            known_pools: set[str] = set()
            for hop_index, hop in enumerate(hops):
                link_key = f"{transfer_id}/hop-{hop_index}/{hop.link_id}"
                engine.call("link", link_key)
                for pool in hop.shared_capacity_pools:
                    if pool not in known_pools:
                        engine.call("addPool", pool, hop.bandwidth_bps)
                        known_pools.add(pool)
                    engine.call("attachPool", link_key, pool)
                engine.call("configure", link_key, hop.bandwidth_bps)
                flow_id = f"{transfer_id}/hop-{hop_index}"
                engine.call("start", flow_id, link_key, size_bytes)
                # The deadline is only a guard.  NativeEngine stops at PhyTxEnd,
                # so its reported time is the actual packetized completion.
                packets = math.ceil(size_bytes / 1400.0) if size_bytes else 0
                wire_bytes = size_bytes + packets * 2
                guard_s = max(1e-9, wire_bytes * 8.0 / hop.bandwidth_bps * 2 + 1.0)
                engine.call("advance", engine.value("now") + guard_s)
                if flow_id not in engine.completed():
                    raise RuntimeError(f"ns-3 transfer did not complete: {flow_id}")
                engine.call("clearCompleted")
                if hop.propagation_delay_s:
                    engine.call(
                        "advance", engine.value("now") + hop.propagation_delay_s
                    )
            return engine.value("now")
        finally:
            engine.close()


__all__ = ["Ns3PointToPointDriver", "Ns3Unavailable", "require_ns3"]
