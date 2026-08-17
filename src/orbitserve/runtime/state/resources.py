"""Mutable compute-concurrency and per-node resource state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from ...common import to_jsonable


@dataclass(frozen=True)
class ComputeReservation:
    request_id: str
    ready_s: float
    start_s: float
    finish_s: float
    concurrency_at_admission: int


@dataclass
class ComputeConcurrencyState:
    max_concurrency: int = 1
    reservations: list[ComputeReservation] = field(default_factory=list)

    def __post_init__(self) -> None:
        if type(self.max_concurrency) is not int or self.max_concurrency < 1:
            raise ValueError("max_concurrency must be a positive integer")

    def active_count_at(self, time_s: float) -> int:
        return sum(
            1
            for row in self.reservations
            if row.start_s <= time_s < row.finish_s
        )

    def comparison_concurrency_at(self, time_s: float) -> int:
        return min(self.max_concurrency, self.active_count_at(time_s) + 1)

    def preview(
        self,
        *,
        request_id: str,
        ready_s: float,
        duration_for: Callable[[int, float], float],
    ) -> ComputeReservation:
        """Find the first FIFO-compatible interval with one free request slot."""
        candidate_s = float(ready_s)
        if not math.isfinite(candidate_s) or candidate_s < 0.0:
            raise ValueError("compute ready time must be finite and non-negative")
        for _ in range(len(self.reservations) * 3 + 10):
            active = self._active_at(candidate_s)
            if len(active) >= self.max_concurrency:
                candidate_s = min(row.finish_s for row in active)
                continue
            concurrency = len(active) + 1
            duration_s = float(duration_for(concurrency, candidate_s))
            if not math.isfinite(duration_s) or duration_s < 0.0:
                raise ValueError("compute service time must be finite and non-negative")
            finish_s = candidate_s + duration_s
            conflict_s = self._first_capacity_conflict(candidate_s, finish_s)
            if conflict_s is None:
                return ComputeReservation(
                    request_id=str(request_id),
                    ready_s=float(ready_s),
                    start_s=candidate_s,
                    finish_s=finish_s,
                    concurrency_at_admission=concurrency,
                )
            conflicting = self._active_at(conflict_s)
            candidate_s = min(row.finish_s for row in conflicting)
        raise ValueError("compute_reservation_search_did_not_converge")

    def reserve(self, reservation: ComputeReservation) -> ComputeReservation:
        expected = self.preview(
            request_id=reservation.request_id,
            ready_s=reservation.ready_s,
            duration_for=lambda concurrency, start_s: reservation.finish_s - reservation.start_s,
        )
        if (
            not math.isclose(expected.start_s, reservation.start_s, abs_tol=1e-9)
            or not math.isclose(expected.finish_s, reservation.finish_s, abs_tol=1e-9)
            or expected.concurrency_at_admission != reservation.concurrency_at_admission
        ):
            raise ValueError("candidate_preview_state_changed")
        self.reservations.append(reservation)
        self.reservations.sort(key=lambda row: (row.start_s, row.finish_s, row.request_id))
        return reservation

    def _active_at(self, time_s: float) -> tuple[ComputeReservation, ...]:
        return tuple(
            row
            for row in self.reservations
            if row.start_s <= time_s < row.finish_s
        )

    def _first_capacity_conflict(self, start_s: float, finish_s: float) -> float | None:
        if finish_s <= start_s:
            return None
        boundaries = sorted(
            {
                row.start_s
                for row in self.reservations
                if start_s < row.start_s < finish_s
            }
        )
        return next(
            (
                boundary
                for boundary in boundaries
                if self.active_count_at(boundary) >= self.max_concurrency
            ),
            None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_concurrency": self.max_concurrency,
            "reservations": to_jsonable(tuple(self.reservations)),
        }


@dataclass
class ComputeNodeState:
    compute_node_id: str
    compute_queue: ComputeConcurrencyState = field(default_factory=ComputeConcurrencyState)
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


ComputeQueueState = ComputeConcurrencyState
SatelliteState = ComputeNodeState


__all__ = [
    "ComputeConcurrencyState",
    "ComputeNodeState",
    "ComputeQueueState",
    "ComputeReservation",
    "SatelliteState",
]
