"""Drive a shared communication backend from the global timeline."""

from decimal import Decimal, ROUND_CEILING

from network_sim.comm.model import TransferRequest

from .cell_events import DISPATCH


def nanoseconds(seconds):
    return int((Decimal(str(seconds)) * 1_000_000_000).to_integral_value(rounding=ROUND_CEILING))


class CommunicationFailed(RuntimeError):
    def __init__(self, event):
        self.event = event.model_dump()
        super().__init__(f"network transfer failed: {event.id}: {event.reason}")


class NetworkEventDriver:
    """Own the single network-event registration and transfer callback table."""

    def __init__(self, loop, network):
        self.loop, self.network = loop, network
        self._generation = 0
        self._callbacks = {}
        self.transfer_events = []

    def _arm_network(self):
        self._generation += 1
        generation = self._generation
        next_ns = self.network.next_event_time_ns()
        if next_ns is not None:
            at_s = next_ns / 1e9
            if at_s < self.loop.now_s:
                raise RuntimeError("network clock missed an event")
            self.loop.schedule(at_s, DISPATCH, "network:advance",
                               lambda: self._pump(generation, next_ns))

    def _pump(self, generation, next_ns):
        if generation != self._generation:
            return
        for event in self.network.advance_to(next_ns):
            self.transfer_events.append(event.model_dump())
            callback, failure = self._callbacks.pop(event.id)
            if event.kind == "failed":
                if failure is None:
                    raise CommunicationFailed(event)
                failure(event)
            else:
                if event.completion_time_ns != next_ns:
                    raise RuntimeError("network completion was not delivered at its event time")
                callback()
        self._arm_network()

    def submit_transport(self, identifier, source, target, size_bytes, operation,
                         correlation_id, on_complete, on_failure=None):
        if identifier in self._callbacks:
            raise ValueError("duplicate active transport")
        if source == target:
            on_complete()
            return
        request = TransferRequest(id=identifier, source=source, target=target,
            size_bytes=size_bytes, ready_time_ns=nanoseconds(self.loop.now_s),
            origin="orbitserve", operation=operation, correlation_id=correlation_id)
        self._submit(request, on_complete, on_failure)
        self._arm_network()

    def submit_all(self, requests, on_complete):
        """Submit a set of ready flows and arm the clock once for the group."""
        for request in requests:
            self._submit(request, lambda identifier=request.id: on_complete(identifier))
        self._arm_network()

    def _submit(self, request, on_complete, on_failure=None):
        if request.id in self._callbacks:
            raise ValueError("duplicate active transport")
        self.network.submit(request)
        self._callbacks[request.id] = (on_complete, on_failure)
