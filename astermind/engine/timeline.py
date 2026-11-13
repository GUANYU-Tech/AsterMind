"""One simulation clock shared by OrbitServe, AsterMind and communication.

The timeline is deliberately small at its public interface: every subsystem
registers an absolute-time event and the timeline executes events in
``(time, priority, sequence)`` order.  Cells may keep dependency state, but
they do not own another clock or event queue.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal
from typing import Any, Protocol
from astermind.engine.clock import CLOCK, Time, ClockLog

EventCallback = Callable[[], None]


def nanoseconds(seconds):
    """Map host seconds to the next native tick without float multiplication."""
    if isinstance(seconds, Time):
        return (seconds.ticks + 10**9 - 1) // 10**9
    return int(
        (Decimal(str(seconds)) * 1_000_000_000).to_integral_value(
            rounding=ROUND_CEILING
        )
    )


@dataclass(frozen=True)
class TimelineEvent:
    time_s: float
    priority: int
    sequence: int
    kind: str
    owner_id: str | None = None
    metadata: dict[str, Any] | None = None


class GlobalTimeline:
    """The single timing authority for one simulation run."""

    def __init__(self, *, start_s: float = 0.0, host=None):
        if not math.isfinite(start_s) or start_s < 0:
            raise ValueError("timeline start must be finite and nonnegative")
        self._now_s = Time(start_s)
        self._host = host
        self._pending = 0
        self._sequence = itertools.count()
        self._queue: list[tuple[float, int, int, TimelineEvent, EventCallback]] = []
        self.events: list[dict[str, Any]] = ClockLog()
        self._stopped = False
        self.current_event: TimelineEvent | None = None

    @property
    def now_s(self) -> float:
        return self._now_s if self._host is None else self._host.now

    @property
    def now_ns(self) -> float:
        return self.now_s * 1e9

    @property
    def pending_count(self) -> int:
        return len(self._queue) if self._host is None else self._pending

    def schedule(
        self,
        at_s: float,
        priority: int,
        kind: str,
        callback: EventCallback,
        *,
        owner_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TimelineEvent:
        at_s = at_s if isinstance(at_s, Time) else Time(at_s)
        if at_s < self.now_s:
            raise ValueError("cannot schedule a past or nonfinite event")
        if type(priority) is not int:
            raise TypeError("event priority must be an integer")
        if not isinstance(kind, str) or not kind:
            raise ValueError("event kind must be nonempty")
        if not callable(callback):
            raise TypeError("event callback must be callable")
        sequence = next(self._sequence)
        event = TimelineEvent(at_s, priority, sequence, kind, owner_id, metadata)
        if self._host is not None:
            self._pending += 1

            def dispatch():
                self._pending -= 1
                self.events.append(
                    {
                        "time_s": self._host.now,
                        "priority": event.priority,
                        "sequence": event.sequence,
                        "event": event.kind,
                        "owner_id": event.owner_id,
                        "metadata": event.metadata,
                    }
                )
                self.current_event = event
                try:
                    callback()
                finally:
                    self.current_event = None

            # Physical updates and hop completions precede native Cell work.
            self._host.push(at_s, 30 + priority, "external_callback", payload=dispatch)
            return event
        heapq.heappush(
            self._queue,
            (event.time_s, event.priority, event.sequence, event, callback),
        )
        return event

    def schedule_after(
        self,
        delay_s: float,
        priority: int,
        kind: str,
        callback: EventCallback,
        **event_fields: Any,
    ) -> TimelineEvent:
        if not math.isfinite(delay_s) or delay_s < 0:
            raise ValueError("event delay must be finite and nonnegative")
        return self.schedule(
            self.now_s + delay_s,
            priority,
            kind,
            callback,
            **event_fields,
        )

    def stop(self) -> None:
        if self._host is not None:
            raise RuntimeError("the host owns the Runtime event loop")
        self._stopped = True

    def run(
        self,
        *,
        until_s: float | None = None,
        max_events: int = 1_000_000,
    ) -> dict[str, Any]:
        if self._host is not None:
            raise RuntimeError("advance the host Runtime, not a second Cell event loop")
        if until_s is not None and (not math.isfinite(until_s) or until_s < self.now_s):
            raise ValueError("invalid event horizon")
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        handled = 0
        self._stopped = False
        while self._queue and not self._stopped:
            if until_s is not None and self._queue[0][0] > until_s:
                break
            if handled >= max_events:
                raise RuntimeError("event limit reached; simulation did not drain")
            _, _, _, event, callback = heapq.heappop(self._queue)
            self._now_s = event.time_s
            self.events.append(
                {
                    "time_s": event.time_s,
                    "priority": event.priority,
                    "sequence": event.sequence,
                    "event": event.kind,
                    "owner_id": event.owner_id,
                    "metadata": event.metadata,
                }
            )
            self.current_event = event
            try:
                callback()
            finally:
                self.current_event = None
            handled += 1
        if until_s is not None and not self._stopped:
            self._now_s = Time(until_s)
        return self.report()

    def report(self) -> dict[str, Any]:
        return {
            'clock': CLOCK,
            'time_ticks': str(self.now_s.ticks),
            "time_s": self.now_s,
            "handled_events": len(self.events),
            "pending_events": self.pending_count,
            "stopped": self._stopped,
        }


class CellRuntimeProtocol(Protocol):
    cell_id: str

    def submit_ready_request(self, request: Any) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


class CellManager:
    """Own Cell registrations and dispatch onto the shared timeline."""

    def __init__(self, timeline: GlobalTimeline):
        self.timeline = timeline
        self._cells: dict[str, CellRuntimeProtocol] = {}

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._cells))

    def register(self, cell: CellRuntimeProtocol) -> None:
        cell_id = getattr(cell, "cell_id", None)
        if not isinstance(cell_id, str) or not cell_id:
            raise ValueError("Cell runtime requires a nonempty cell_id")
        if cell_id in self._cells:
            raise ValueError(f"duplicate Cell runtime: {cell_id}")
        self._cells[cell_id] = cell

    def get(self, cell_id: str) -> CellRuntimeProtocol:
        try:
            return self._cells[cell_id]
        except KeyError as exc:
            raise KeyError(f"unknown Compute Cell: {cell_id}") from exc

    def submit(self, cell_id: str, request: Any, *, at_s: float) -> TimelineEvent:
        cell = self.get(cell_id)
        return self.timeline.schedule(
            at_s,
            0,
            "request_dispatched_to_cell",
            lambda: cell.submit_ready_request(request),
            owner_id=cell_id,
            metadata={"request_id": getattr(request, "request_id", None)},
        )

    def snapshots(self) -> dict[str, dict[str, Any]]:
        return {cell_id: self._cells[cell_id].snapshot() for cell_id in self.cell_ids}


class SchedulingPolicy(Protocol):
    def select_cell(
        self,
        request: Any,
        cell_snapshots: dict[str, dict[str, Any]],
    ) -> str: ...


class GlobalScheduling:
    """Select a Cell from one consistent global-state snapshot."""

    def __init__(self, cells: CellManager, policy: SchedulingPolicy):
        if not hasattr(policy, "select_cell"):
            raise TypeError("global scheduling policy must provide select_cell()")
        self.cells = cells
        self.policy = policy

    def dispatch(self, request: Any, *, at_s: float | None = None) -> str:
        selected = self.policy.select_cell(request, self.cells.snapshots())
        ready_at = self.cells.timeline.now_s if at_s is None else Time(at_s)
        self.cells.submit(selected, request, at_s=ready_at)
        return selected


class FixedCellScheduling:
    """Standalone AsterMind policy: always use its only configured Cell."""

    def __init__(self, cell_id: str):
        if not cell_id:
            raise ValueError("fixed Cell ID must be nonempty")
        self.cell_id = cell_id

    def select_cell(
        self,
        request: Any,
        cell_snapshots: dict[str, dict[str, Any]],
    ) -> str:
        if self.cell_id not in cell_snapshots:
            raise KeyError(f"fixed Compute Cell is unavailable: {self.cell_id}")
        return self.cell_id


__all__ = [
    "CellManager",
    "FixedCellScheduling",
    "GlobalScheduling",
    "GlobalTimeline",
    "SchedulingPolicy",
    "TimelineEvent",
]
