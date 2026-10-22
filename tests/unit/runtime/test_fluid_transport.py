import unittest
from orbitserve.runtime.engine.fluid_transport import FluidTransport


class FluidTransportTests(unittest.TestCase):
    def test_complete_runtime_uses_selected_fluid_backend(self):
        from dataclasses import replace
        from orbitserve.runtime import run_simulation
        from tests.unit.runtime.test_runtime_engine import _scenario
        from tests.unit.runtime.test_runtime_event_order import strategy
        scenario=_scenario(strategy())
        scenario=replace(scenario,network=replace(scenario.network,transport_model='fluid'))
        result=run_simulation(scenario)
        self.assertEqual(result.request_summary['completed_request_count'],1)
        self.assertEqual(result.provenance['network_transport']['backend'],'fluid')

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
