"""Terminal reservations and acquisition events, independent of model placement."""

from dataclasses import dataclass
import math

from ._vendor.orbitserve_link_selection import OrbitServeSelector

MAX_TIME = 2**64 - 1


@dataclass
class LinkState:
    state: str = "down"
    ready_ns: int | None = None


class LinkController:
    """Known visibility losses are exact events; partner selection is periodic.

    A custom selector receives (ordered_candidates, reserved_link_ids, time_ns)
    and returns candidate link IDs. The controller still enforces terminal slots.
    """

    def __init__(self, topology, selector=None):
        self.topology = topology
        self.config = topology.scenario.link_policy
        if self.config is None:
            raise ValueError("LinkController needs link_policy configuration")
        self.selector = selector
        self.links = {l.id: l for l in topology.links if l.enabled and
                      topology.devices[l.source].satellite != topology.devices[l.target].satellite}
        self.states = {identifier: LinkState() for identifier in self.links}
        self.used_terminals = set()
        self.history = []
        self.now_ns = 0
        self.next_check_ns = 0
        self.advance_to(0)

    def _window(self, link, time_ns):
        if link.availability_windows is None:
            return MAX_TIME + 1
        return next((w.end_ns for w in link.availability_windows
                     if w.start_ns <= time_ns < w.end_ns), None)

    def _key(self, link):
        a, b = (self.topology.devices[x].satellite for x in (link.source, link.target))
        pa, pb = (self.topology.satellites[x].ecef_position_km for x in (a, b))
        distance = link.distance_km
        if distance is None:
            distance = math.dist(pa, pb) if pa is not None and pb is not None else math.inf
        return distance, min(a, b), max(a, b), link.id

    def _change(self, identifier, state, ready_ns=None, reason=""):
        record = self.states[identifier]
        if record.state == state and record.ready_ns == ready_ns:
            return
        self.history.append(dict(time_ns=self.now_ns, link_id=identifier,
                                 previous=record.state, state=state, reason=reason,
                                 ready_ns=ready_ns))
        record.state, record.ready_ns = state, ready_ns
        if state == "up":
            link = self.links[identifier]
            self.used_terminals.update((link.source, link.target))

    def _select(self):
        candidates = sorted((l for l in self.links.values()
                             if self._window(l, self.now_ns) is not None), key=self._key)
        reserved = {i for i, s in self.states.items() if s.state != "down"}
        if self.selector is not None:
            chosen = list(self.selector(tuple(candidates), frozenset(reserved), self.now_ns))
        elif self.config.mode == "os_reference":
            rows = []
            for l in candidates:
                a, b = sorted(self.topology.devices[x].satellite for x in (l.source, l.target))
                rows.append((a, b, {"link_id": l.id}, self._key(l)[0]))
            chosen = [e[2]["link_id"] for e in OrbitServeSelector(
                self.config.os_layout, self.config.os_terminal_count)._select_terminal_edges(tuple(rows))]
        else:
            chosen = sorted(reserved, key=lambda i: self._key(self.links[i])) + [l.id for l in candidates]
        valid = {l.id for l in candidates}
        selected, slots = set(), {}
        for identifier in chosen:
            if identifier in selected:
                continue
            if identifier not in valid:
                raise ValueError("selector returned an unavailable or unknown link")
            link = self.links[identifier]
            ends = (link.source, link.target)
            if any(slots.get(x, 0) >= self.topology.devices[x].max_external_links for x in ends):
                continue
            selected.add(identifier)
            for x in ends:
                slots[x] = slots.get(x, 0) + 1
        for identifier in sorted(reserved - selected):
            self._change(identifier, "down", reason="policy_reselection")
        for identifier in sorted(selected - reserved):
            link = self.links[identifier]
            reacquire = any(x in self.used_terminals for x in (link.source, link.target))
            delay = self.config.reacquisition_ns if reacquire else self.config.initial_acquisition_ns
            ready = self.now_ns + delay
            if ready > MAX_TIME:
                raise OverflowError("acquisition time overflow")
            self._change(identifier, "acquiring", ready, "reacquisition" if reacquire else "initial")

    def next_event_time_ns(self):
        times = [] if self.next_check_ns is None else [self.next_check_ns]
        for identifier, state in self.states.items():
            if state.state == "down":
                continue
            if state.ready_ns is not None:
                times.append(state.ready_ns)
            end = self._window(self.links[identifier], self.now_ns)
            if end is not None and end <= MAX_TIME:
                times.append(end)
        return min(times) if times else None

    def advance_to(self, time_ns):
        if type(time_ns) is not int or not self.now_ns <= time_ns <= MAX_TIME:
            raise ValueError("controller time must be an integer, monotonic nanoseconds")
        while (event := self.next_event_time_ns()) is not None and event <= time_ns:
            self.now_ns = event
            for identifier, state in self.states.items():
                if state.state != "down" and self._window(self.links[identifier], event) is None:
                    self._change(identifier, "down", reason="visibility_loss")
            if self.next_check_ns == event:
                self._select()
                next_check = event + self.config.check_interval_ns
                self.next_check_ns = next_check if next_check <= MAX_TIME else None
            for identifier, state in self.states.items():
                if state.state == "acquiring" and state.ready_ns == event:
                    self._change(identifier, "up", reason="acquisition_complete")
        self.now_ns = time_ns

    def snapshot(self):
        return dict(mode=self.config.mode, time_ns=self.now_ns,
                    next_check_ns=self.next_check_ns,
                    links={i: dict(state=s.state, ready_ns=s.ready_ns) for i, s in self.states.items()},
                    changes=list(self.history))
