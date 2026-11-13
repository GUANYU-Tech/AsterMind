"""Mid-sample failures and repairs update the graph and transport immediately."""

import importlib.util
import unittest
from dataclasses import replace

from astermind.engine.state.interventions import apply_topology_command
from tests.integration.fixed_design.test_boundary_control import scenario


class ContinuousControlTests(unittest.TestCase):
    def modes(self):
        return [mode for mode, module in [('queue', None), ('fluid', None), ('simgrid', 'simgrid'), ('ns3', 'ns')]
                if module is None or importlib.util.find_spec(module)]





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
