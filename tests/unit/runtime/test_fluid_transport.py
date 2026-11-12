import unittest
from orbitserve.runtime.engine.fluid_transport import FluidTransport


class FluidTransportTests(unittest.TestCase):
    def test_tiny_payload_requires_actual_service_and_conserves_bytes(self):
        engine = FluidTransport({}, (), {})
        leg = dict(link_id='ab', source_node_id='a', target_node_id='b')
        engine.configure({engine.resource(leg): 0.})
        engine.start('tiny', leg, 1e-8)
        for time_s in (0., 1.):
            result = engine.advance_until(time_s)
            self.assertEqual(result['completed'], [])
            self.assertEqual(result['bytes_sent']['tiny'], 0.)
        engine.configure({engine.resource(leg): 8.})
        result = engine.advance_until(2.)
        self.assertEqual(result['completed'], ['tiny'])
        self.assertEqual(result['bytes_sent']['tiny'], 1e-8)
    def test_complete_runtime_uses_selected_fluid_backend(self):
        from dataclasses import replace
        from tests.network_case import NetworkCase, run_traffic, traffic
        scenario = NetworkCase(traffic=(traffic(),))
        result = run_traffic(replace(scenario, network=replace(scenario.network, transport_model="fluid")))
        self.assertEqual(result.completed, 1)
        self.assertEqual(result.transport_provenance["backend"], "fluid")

    def test_late_join_capacity_change_reverse_direction_and_cancellation(self):
        engine = FluidTransport({}, (), {})
        forward = {'link_id': 'ab', 'source_node_id': 'a', 'target_node_id': 'b'}
        reverse = {**forward, 'source_node_id': 'b', 'target_node_id': 'a'}
        engine.configure({engine.resource(forward): 8000, engine.resource(reverse): 8000})
        engine.start('a', forward, 1000)
        self.assertEqual(engine.advance_until(.5)['bytes_sent']['a'], 500)
        engine.start('b', forward, 500)
        engine.start('reverse', reverse, 500)
        result = engine.advance_until(1)
        self.assertEqual(result['completed'], ['reverse'])
        self.assertEqual(result['bytes_sent']['a'], 250)
        engine.cancel(['b'])
        engine.configure({engine.resource(forward): 4000})
        self.assertEqual(engine.advance_until(2)['time_s'], 1.5)
        self.assertFalse(engine.active)
