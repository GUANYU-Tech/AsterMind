"""Mutable compute, link, and per-satellite resource state."""

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
class LinkState:
    link_id: str
    available_s: float = 0.0

    def preview(
        self,
        start_s: float,
        payload_bytes: float,
        *,
        bandwidth_bps: float,
        propagation_delay_s: float,
        setup_delay_s: float = 0.0,
    ) -> dict[str, float]:
        if bandwidth_bps <= 0:
            transmission_s = float("inf")
        else:
            transmission_s = max(0.0, payload_bytes) * 8.0 / bandwidth_bps
        begin_s = max(start_s, self.available_s)
        finish_transmit_s = begin_s + max(0.0, setup_delay_s) + transmission_s
        return {
            "queue_wait_s": max(0.0, begin_s - start_s),
            "transmission_s": transmission_s,
            "setup_delay_s": max(0.0, setup_delay_s),
            "propagation_delay_s": propagation_delay_s,
            "arrival_s": finish_transmit_s + propagation_delay_s,
            "finish_transmit_s": finish_transmit_s,
        }

    def reserve(
        self,
        start_s: float,
        payload_bytes: float,
        *,
        bandwidth_bps: float,
        propagation_delay_s: float,
        setup_delay_s: float = 0.0,
    ) -> dict[str, float]:
        result = self.preview(
            start_s,
            payload_bytes,
            bandwidth_bps=bandwidth_bps,
            propagation_delay_s=propagation_delay_s,
            setup_delay_s=setup_delay_s,
        )
        self.available_s = result["finish_transmit_s"]
        return result

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class SatelliteState:
    satellite_id: str
    compute_queue: ComputeQueueState = field(default_factory=ComputeQueueState)
    soc: float = 0.8
    initial_soc: float = 0.8
    last_energy_update_s: float = 0.0
    sunlit: bool = True
    dvfs_scale: float = 1.0
    energy_events: tuple[dict[str, Any], ...] = ()
    energy_ledger: tuple[dict[str, Any], ...] = ()
    materialized_energy_event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "satellite_id": self.satellite_id,
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


__all__ = ["ComputeQueueState", "LinkState", "SatelliteState"]
