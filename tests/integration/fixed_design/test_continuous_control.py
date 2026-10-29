"""Mid-sample failures and repairs update the graph and transport immediately."""

import importlib.util
import unittest
from dataclasses import replace

from orbitserve.runtime.state.interventions import apply_topology_command
from tests.integration.fixed_design.test_boundary_control import scenario, job


class ContinuousControlTests(unittest.TestCase):
    def modes(self):
        return [mode for mode, module in [('queue', None), ('fluid', None), ('simgrid', 'simgrid'), ('ns3', 'ns')]
                if module is None or importlib.util.find_spec(module)]

    def test_mid_sample_failure_reroutes_at_its_actual_time(self):
        for mode in self.modes():
            with self.subTest(mode=mode):
                engine = scenario(mode)
                engine.net.spec = replace(engine.net.spec, fault_timing='continuous')
                request = job(engine, 'continuous')
                engine.send(request, 'a', 'c', 80000., None, 'input', 'delivered')
                fault = dict(target_type='link', target_id='communication_isl:a:b', start_s=30.125, end_s=45.5)
                engine.push(fault['start_s'], -2, 'fault_start', None, (0, fault))
                engine.push(fault['end_s'], -2, 'fault_end', None, (0, fault))
                engine.run()
                recovery = next(e for e in engine.log if e['kind'] == 'route_recovery')
                self.assertEqual(recovery['time_s'], 30.125)
                self.assertNotIn('failure', request)
                self.assertEqual(request['path'][1]['source_node_id'], 'a')
                self.assertEqual(request['path'][1]['target_node_id'], 'd')
                self.assertLess(request['path'][1]['start_s'], 60.)

    def test_mid_sample_repair_wakes_route_waiter_before_next_sample(self):
        for mode in self.modes():
            with self.subTest(mode=mode):
                engine = scenario(mode, edges=(('a', 'b', 0., 240.), ('b', 'c', 0., 240.)))
                engine.net.spec = replace(engine.net.spec, fault_timing='continuous')
                request = job(engine, 'repair')
                engine.send(request, 'a', 'c', 80000., None, 'input', 'delivered')
                fault = dict(target_type='link', target_id='communication_isl:a:b', start_s=30.125, end_s=45.5)
                engine.push(30.125, -2, 'fault_start', None, (0, fault))
                engine.push(45.5, -2, 'fault_end', None, (0, fault))
                engine.run()
                self.assertNotIn('failure', request)
                self.assertEqual(request['path'][1]['start_s'], 45.5)

    def test_failed_buffer_owner_cannot_recover_as_a_link_failure(self):
        engine = scenario()
        engine.net.spec = replace(engine.net.spec, fault_timing='continuous')
        request = job(engine, 'node-loss')
        engine.send(request, 'a', 'c', 80000., None, 'input', 'delivered')
        fault = dict(target_type='node', target_id='a', start_s=30.125, end_s=45.5)
        engine.push(30.125, -2, 'fault_start', None, (0, fault))
        engine.run()
        self.assertEqual(request['failure'], 'forwarding_buffer_lost_on_node_failure')

    def test_repair_waits_for_acquisition_without_rounding_to_a_sample(self):
        engine = scenario(setup=3.)
        engine.net.spec = replace(engine.net.spec, fault_timing='continuous')
        fault = dict(target_type='link', target_id='communication_isl:a:b', start_s=30.125, end_s=45.5)
        engine.now = 30.125
        engine.on_fault_start(None, (0, fault))
        engine.now = 45.5
        engine.on_fault_end(None, (0, fault))
        link = engine.net.link_at(fault['target_id'], 45.5)
        self.assertEqual(link['state'], 'SETTING_UP')
        self.assertEqual(link['ready_s'], 48.5)
        self.assertEqual(engine.net.link_at(fault['target_id'], 48.5)['state'], 'UP')

    def test_manual_changes_mid_sample_and_independent_fault_causes(self):
        engine = scenario()
        engine.net.spec = replace(engine.net.spec, fault_timing='continuous')
        engine.now = 30.125
        link = 'communication_isl:a:b'
        natural = dict(target_type='link', target_id=link, source='random', start_s=30.125, end_s=90.)
        engine.on_fault_start(None, ('random', natural))
        apply_topology_command(engine, dict(action='link_fail', target_id=link))
        apply_topology_command(engine, dict(action='link_restore', target_id=link))
        self.assertEqual(engine.net.link_at(link, engine.now)['state'], 'DOWN')
        with self.assertRaises(ValueError):
            apply_topology_command(engine, dict(action='connect', endpoint_a='a', endpoint_b='b'))
        engine.now = 45.5
        engine.on_fault_end(None, ('random', natural))
        self.assertEqual(engine.net.link_at(link, engine.now)['state'], 'UP')
        apply_topology_command(engine, dict(action='disconnect', target_id=link))
        self.assertEqual(engine.net.link_at(link, 60.)['state'], 'DOWN')
        reply = apply_topology_command(engine, dict(action='connect', endpoint_a='a', endpoint_b='b'))
        self.assertEqual(reply['message'], '连接成功')


if __name__ == '__main__':
    unittest.main()
