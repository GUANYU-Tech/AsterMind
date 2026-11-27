"""Physical events and callbacks on the authoritative integer clock.

Communication progress, retry and completion belong solely to Comm.
"""
import heapq
from astermind.engine.clock import Time, ClockLog


class _EventEngine:
    def __init__(self, spec, environment):
        from astermind.engine.control import current_controller
        self.controller = current_controller()
        self.spec, self.env, self.net = spec, environment, environment.network
        end = environment.time['end_s']
        self.end = Time(end) if end is not None else None
        if spec.time_horizon_s is not None:
            limit = Time(spec.time_horizon_s)
            self.end = limit if self.end is None else min(self.end, limit)
        self.heap, self.log = [], ClockLog()
        self.sequence, self.now = 0, Time()

    def push(self, time, priority, kind, request_id=None, payload=None):
        time = time if isinstance(time, Time) else Time(time)
        if time < self.now:
            raise RuntimeError("event scheduled in the past")
        self.sequence += 1
        heapq.heappush(self.heap, (time, priority, self.sequence, kind, request_id, payload))

    def run(self, *, stop_when=None):
        while self.heap:
            # Drain callbacks at the last request's tick before ending the run.
            if stop_when is not None and self.heap[0][0] > self.now and stop_when():
                break
            if self.controller and hasattr(self.controller, 'poll'):
                self.controller.poll(self)
            time, priority, sequence, kind, request_id, payload = heapq.heappop(self.heap)
            if self.end is not None and time > self.end:
                break
            self.now = time
            if kind in {'link_state', 'network_update'} and (payload or {}).get('revision', 0) != self.net.revision:
                continue
            self.log.append(dict(time_s=time, kind=kind, request_id=request_id))
            if kind in {'link_state', 'topology_sample', 'network_update'}:
                policy = getattr(self, 'communication_policy', None)
                if policy is not None:
                    policy.network_changed()
                continue
            getattr(self, 'on_' + kind)(None, payload)
        if self.end is not None:
            self.now = self.end
        if self.controller:
            self.controller.finished(self)

    def close(self):
        self.heap.clear()

    def on_control_boundary(self, unused, payload):
        self.controller.checkpoint(self)

    def on_external_callback(self, unused, callback):
        callback()

    def on_fault_start(self, unused, payload):
        index, fault = payload
        self.on_fault_batch(unused, ((index, fault, False),))

    def on_fault_end(self, unused, payload):
        index, fault = payload
        self.on_fault_batch(unused, ((index, fault, True),))

    def on_fault_batch(self, unused, observations):
        from astermind.engine.setup.faults import fault_step_details
        self.net.observe_faults(observations, self.now)
        self.schedule_reallocation()
        for index, fault, restored in observations:
            self.log.append(dict(time_s=self.now, kind='fault_restored' if restored else 'fault_injected',
                                 **fault, **fault_step_details(fault, self.net.environment_times)))

    def schedule_reallocation(self):
        # Replaced graph versions must not accumulate future events or wake
        # route waiters after they have become obsolete.
        def keep(e):
            return e[3] not in {'link_state', 'network_update'} or (e[5] or {}).get('revision', 0) == self.net.revision
        self.heap[:] = [e for e in self.heap if keep(e)]
        heapq.heapify(self.heap)
        for event in self.net.events:
            if event['time_s'] >= self.now:
                self.push(event['time_s'], 0, 'link_state', None,
                          {**event, 'revision': self.net.revision})
        for frame in self.net.frames:
            if frame['time_s'] > self.now:
                self.push(frame['time_s'], 0, 'network_update', None, {'revision': self.net.revision})
