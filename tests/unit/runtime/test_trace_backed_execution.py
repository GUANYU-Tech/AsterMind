import unittest

from orbitserve.orbit import RuntimeTraceBundle
from orbitserve.runtime.engine.simulator import _reserve_route
from orbitserve.runtime.state.runtime_state import RuntimeState


class TraceBackedExecutionTests(unittest.TestCase):
    def test_empty_route_reservation_preserves_time_and_state(self):
        state = RuntimeState()
        trace = RuntimeTraceBundle(
            trace_id="unit-empty-route",
            horizon_s=10.0,
        )

        result = _reserve_route(
            state,
            trace,
            (),
            3.0,
            request_id="request-0",
            energy_stage="unit_route",
        )

        done_s, path, sgl_wait_s, isl_wait_s, transmission_s, sampled = result
        self.assertEqual(done_s, 3.0)
        self.assertEqual(path, ())
        self.assertEqual(sgl_wait_s, 0.0)
        self.assertEqual(isl_wait_s, 0.0)
        self.assertEqual(transmission_s, 0.0)
        self.assertEqual(sampled, ())
