"""SimGrid-owned deadlines; Python retains only event payloads, not a time heap."""

import math

from astermind.engine.clock import Time, ticks


SIMGRID_CLOCK = {
    "schema": "orbitserve.simgrid_clock.v1",
    "authority": "simgrid_engine",
    "representation": "float64_seconds",
    "timing_tolerance_s": 1e-15,
    "record_unit": "attosecond",
}


def native_time(value):
    """Round a requested deadline upward to a representable native second."""
    if hasattr(value, "native_seconds"):
        return value.native_seconds
    requested = Time(value)
    result = float(requested)
    if not math.isfinite(result) or result < 0:
        raise ValueError("SimGrid event time must be finite and nonnegative")
    if Time(result) < requested:
        result = math.nextafter(result, math.inf)
    return result


def simgrid_time(value):
    """Keep Engine.clock's exact double; integer stamps are record projections."""
    result = float.__new__(Time, float(value))
    result.ticks = ticks(value)
    result.native_seconds = float(value)
    return result


class SimGridEventQueue:
    """Payload registry. SimGrid native waits determine the next causal boundary."""

    def __init__(self, transport):
        self.transport = transport
        self.events = {}

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return iter(self.events.values())

    def append(self, event):
        at, priority, sequence, kind, owner, payload = event
        at = self.transport.schedule_event(sequence, at, priority)
        value = at, priority, sequence, kind, owner, payload
        self.events[sequence] = value
        return at

    def pop_due(self):
        identity = self.transport.pop_event()
        if identity is None:
            return None
        event = self.events.pop(identity)
        # A native sleep_until may wake one floating ULP beyond its requested
        # deadline. Domain callbacks execute at Engine.clock, never a past label.
        return (self.transport.now, *event[1:])

    def retain(self, predicate):
        discarded = [key for key, event in self.events.items() if not predicate(event)]
        self.transport.cancel_events(discarded)
        for key in discarded:
            del self.events[key]


class NativeGlobalEvents:
    """Worker-only scheduler using the controller's native timed waits.

    Ready events use domain priority/sequence at the same native time. This is
    necessary because actor wake order must not decide fault/dependency order.
    Deadlines are multiplexed onto one native wait instead of creating one OS
    thread per future event. SimGrid still advances its clock and interrupts the
    wait when a competing communication completes. No outer Python time heap
    advances the clock.
    """

    def __init__(self, sg, host, active):
        self.sg, self.host, self.active = sg, host, active
        self.pending = {}
        self.stats = dict(scheduled=0, dispatched=0, cancelled=0, advances=0)

    def schedule(self, identity, at, priority, *, restore=False):
        if (
            identity in self.pending
            or type(identity) is not int
            or type(priority) is not int
        ):
            raise ValueError("global event needs unique integer identity and priority")
        if (
            not math.isfinite(at)
            or at < 0
            or (at < self.sg.Engine.clock and not restore)
        ):
            raise ValueError("global event cannot be scheduled in the past")
        row = dict(id=identity, time_s=at, priority=priority)
        self.pending[identity] = row
        self.stats["scheduled"] += 1

    def cancel(self, identities):
        for identity in identities:
            row = self.pending.pop(identity, None)
            if row is not None:
                self.stats["cancelled"] += 1

    def due(self):
        return [
            row
            for row in self.pending.values()
            if row["time_s"] <= self.sg.Engine.clock
        ]

    def pop(self):
        rows = self.due()
        if not rows:
            return None
        row = min(rows, key=lambda r: (r["time_s"], r["priority"], r["id"]))
        del self.pending[row["id"]]
        self.stats["dispatched"] += 1
        return row["id"]

    def advance(self, horizon):
        self.stats["advances"] += 1
        while not self.due() and self.sg.Engine.clock < horizon:
            if any(row["comm"].test() for row in self.active.values()):
                break
            deadline = min(
                (row["time_s"] for row in self.pending.values()), default=horizon
            )
            deadline = min(deadline, horizon)
            if self.active:
                activities = self.sg.ActivitySet(
                    [row["comm"] for row in self.active.values()]
                )
                try:
                    activities.wait_any_for(deadline - self.sg.Engine.clock)
                except self.sg.TimeoutException:
                    pass
            else:
                self.sg.this_actor.sleep_until(deadline)

    def close(self):
        self.cancel(tuple(self.pending))

    def report(self):
        return {
            **self.stats,
            "pending": len(self.pending),
            "authority": "simgrid_engine",
        }
