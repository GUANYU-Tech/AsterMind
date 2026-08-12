"""Mutable compute and per-satellite resource state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common import to_jsonable


@dataclass
class ComputeQueueState:
    prefill_available_s: float = 0.0
    decode_available_s: float = 0.0

    def preview_prefill(self, arrival_s: float, service_s: float) -> tuple[float, float, float]:
        start_s = max(arrival_s, self.prefill_available_s)
        finish_s = start_s + service_s
        wait_s = max(0.0, start_s - arrival_s)
        return start_s, finish_s, wait_s

    def preview_decode(self, ready_s: float, service_s: float) -> tuple[float, float, float]:
        start_s = max(ready_s, self.decode_available_s)
        finish_s = start_s + service_s
        wait_s = max(0.0, start_s - ready_s)
        return start_s, finish_s, wait_s

    def reserve_prefill(self, arrival_s: float, service_s: float) -> tuple[float, float, float]:
        start_s, finish_s, wait_s = self.preview_prefill(arrival_s, service_s)
        self.prefill_available_s = finish_s
        return start_s, finish_s, wait_s

    def reserve_decode(self, ready_s: float, service_s: float) -> tuple[float, float, float]:
        start_s, finish_s, wait_s = self.preview_decode(ready_s, service_s)
        self.decode_available_s = finish_s
        return start_s, finish_s, wait_s

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class ComputeNodeState:
    compute_node_id: str
    compute_queue: ComputeQueueState = field(default_factory=ComputeQueueState)
    soc: float = 0.8
    initial_soc: float = 0.8
    last_energy_update_s: float = 0.0
    sunlit: bool = True
    dvfs_scale: float = 1.0
    energy_events: tuple[dict[str, Any], ...] = ()
    energy_ledger: tuple[dict[str, Any], ...] = ()
    materialized_energy_event_ids: tuple[str, ...] = ()

    @property
    def satellite_id(self) -> str:
        """Geometry-layer compatibility alias for ``compute_node_id``."""

        return self.compute_node_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_node_id": self.compute_node_id,
            "compute_queue": self.compute_queue.to_dict(),
            "soc": self.soc,
            "initial_soc": self.initial_soc,
            "last_energy_update_s": self.last_energy_update_s,
            "sunlit": self.sunlit,
            "dvfs_scale": self.dvfs_scale,
            "energy_events": to_jsonable(self.energy_events),
            "energy_ledger": to_jsonable(self.energy_ledger),
            "materialized_energy_event_ids": self.materialized_energy_event_ids,
        }


SatelliteState = ComputeNodeState


__all__ = ["ComputeNodeState", "ComputeQueueState", "SatelliteState"]
