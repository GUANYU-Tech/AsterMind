import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from astermind.engine.ns3_transport import Ns3Transport, Ns3Unavailable


@unittest.skipUnless(importlib.util.find_spec('ns'), 'Install OrbitServe[ns3] for native ns-3 checks')
class NativeNs3Tests(unittest.TestCase):
    def setUp(self):
        self.engine = Ns3Transport(('a','b'), [{'endpoint_a':'a','endpoint_b':'b','link_id':'ab','kind':'isl'}], {'isl':SimpleNamespace(bandwidth_bps=8_000_000)}, record_packets=True)
        self.addCleanup(self.engine.close)
        self.leg = {'source_node_id':'a','target_node_id':'b','link_id':'ab'}
        self.key = self.engine.resource(self.leg)
        self.engine.configure({self.key:8_000_000})

    def test_native_packet_round_robin_and_wire_overhead(self):
        self.engine.start('one',self.leg,2800)
        self.engine.start('two',self.leg,2800)
        first=self.engine.advance_until(.1)
        self.assertAlmostEqual(first['time_s'],.004206,places=9)
        self.assertEqual(first['completed'],['one'])
        self.assertAlmostEqual(first['bytes_sent']['two'],1400)
        second=self.engine.advance_until(.1)
        self.assertAlmostEqual(second['time_s'],.005608,places=9)
        self.assertEqual(second['completed'],['two'])
        packets=self.engine.packet_trace
        self.assertEqual([p['transfer_id'] for p in packets],['one','two','one','two'])
        self.assertEqual([p['offset_bytes'] for p in packets],[0,0,1400,1400])
        self.assertEqual(sum(p['payload_bytes'] for p in packets),5600)
        self.assertEqual(sum(p['wire_bytes'] for p in packets),5608)
        self.assertEqual(self.engine.provenance['native_counters']['packets_started'],4)
        self.assertTrue(all(p['start_s'] < p['finish_s'] for p in packets))
        self.assertEqual(self.engine.provenance['completion_source'],'native_ns3_point_to_point_phy_tx_end')

    def test_partial_progress_cancel_and_independent_reverse_direction(self):
        reverse={**self.leg,'source_node_id':'b','target_node_id':'a'}
        self.engine.configure({self.engine.resource(reverse):8_000_000})
        self.engine.start('forward',self.leg,2800)
        self.engine.start('reverse',reverse,1400)
        partial=self.engine.advance_until(.0007)
        self.assertGreater(partial['bytes_sent']['forward'],0)
        self.assertLess(partial['bytes_sent']['forward'],1400)
        self.engine.cancel(['forward'])
        done=self.engine.advance_until(.1)
        self.assertEqual(done['completed'],['reverse'])
        self.assertAlmostEqual(done['time_s'],.001402,places=9)

    def test_disabling_packet_trace_retains_native_counters(self):
        self.engine._request({'op': 'trace', 'enabled': False})
        self.engine.start('untraced', self.leg, 2800)
        done = self.engine.advance_until(.1)
        self.assertEqual(done['completed'], ['untraced'])
        self.assertEqual(self.engine.packet_trace, [])
        self.assertEqual(self.engine.provenance['native_counters']['packets_started'], 2)
        self.assertEqual(self.engine.provenance['native_counters']['wire_bytes_started'], 2804)

    def test_fractional_runtime_boundary_advances_without_zero_time_rpc_loop(self):
        boundary=.1234567891234
        result=self.engine.advance_until(boundary)
        self.assertGreaterEqual(result['time_s'],boundary)
        self.assertLess(result['time_s']-boundary,2e-12)
        self.engine.start('tiny',self.leg,1)
        done=self.engine.advance_until(boundary+.01)
        self.assertEqual(done['completed'],['tiny'])

    def test_fractional_logical_bytes_are_conserved_with_wire_padding(self):
        self.engine.start('fraction', self.leg, 10.25)
        done = self.engine.advance_until(.01)
        self.assertEqual(done['completed'], ['fraction'])
        self.assertAlmostEqual(done['bytes_sent']['fraction'], 10.25)
        self.assertAlmostEqual(done['time_s'], 13e-6, places=10)  # ceil(10.25) + 2 PPP

    def test_native_exception_preserves_worker_and_unicode_flow_identity(self):
        with self.assertRaisesRegex(Ns3Unavailable, 'map::at'):
            self.engine._request({'op': 'cancel', 'ids': ['missing']})
        with self.assertRaisesRegex(Ns3Unavailable, 'must not contain NUL'):
            self.engine.start('invalid\0identity', self.leg, 10.25)
        self.engine.start('数据流', self.leg, 10.25)
        done = self.engine.advance_until(.01)
        self.assertEqual(done['completed'], ['数据流'])
        self.assertAlmostEqual(done['bytes_sent']['数据流'], 10.25)

    def test_missing_compiler_reports_required_dependency(self):
        with patch.dict(os.environ, {'CXX': '/nonexistent/orbitserve-cxx'}):
            with self.assertRaisesRegex(Ns3Unavailable, 'requires a C\\+\\+ compiler'):
                Ns3Transport(('a', 'b'), [], {})
