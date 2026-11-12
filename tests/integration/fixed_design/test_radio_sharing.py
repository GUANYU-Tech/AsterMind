"""Real backends must obey the same satellite RF aggregate, in both directions."""
import importlib.util
import unittest

from astermind.engine import NetworkNodeSpec, LinkProfile
from astermind.engine.fluid_transport import FluidTransport


def radio_fixture():
    nodes = {n.node_id: n for n in (
        NetworkNodeSpec('s', 'communication', sgl_terminals=2,
                        radio_capacity_bps={'user_uplink_bps': 8000, 'user_downlink_bps': 16000},
                        capacity_assumptions=('Scaled capacities for deterministic byte accounting.',)),
        NetworkNodeSpec('u', 'ground', ground_kind='user'),
        NetworkNodeSpec('v', 'ground', ground_kind='user'))}
    edges = tuple(dict(link_id='sgl:' + n, endpoint_a=n, endpoint_b='s', kind='sgl') for n in ('u', 'v'))
    legs = [dict(link_id=e['link_id'], source_node_id=e['endpoint_a'], target_node_id='s') for e in edges]
    return nodes, edges, {'sgl': LinkProfile(32000, 0)}, legs


class RadioSharingTests(unittest.TestCase):
    def verify(self, backend, packet=False):
        nodes, edges, profiles, legs = radio_fixture()
        engine = backend(nodes, edges, profiles)
        self.addCleanup(engine.close)
        reverse = dict(link_id='sgl:u', source_node_id='s', target_node_id='u')
        for leg in [*legs, reverse]:
            engine.configure({engine.resource(leg): 32000})
        for identity, leg in zip(('first', 'second', 'reverse'), [*legs, reverse]):
            engine.start(identity, leg, 10000)
        sample = engine.advance_until(1.)
        sent = sample['bytes_sent']
        # Two distinct GSLs share one receiver, not 8 kbps each. The independent
        # transmitter can simultaneously deliver 16 kbps in the other direction.
        self.assertLessEqual(sent['first'] + sent['second'], 1000 + 1e-5)
        self.assertLessEqual(sent['reverse'], 2000 + 1e-5)
        self.assertGreater(sent['first'] + sent['second'], 990)
        self.assertGreater(sent['reverse'], 1990)
        totals = dict(sent)
        while engine.active:
            row = engine.advance_until(30.)
            self.assertLess(row['time_s'], 30.)
            for identity, amount in row['bytes_sent'].items():
                totals[identity] += amount
        for amount in totals.values():
            self.assertAlmostEqual(amount, 10000, places=5)
        self.assertGreaterEqual(engine.now, 20.)
        self.assertLess(engine.now, 20.1 if packet else 20.00001)

    def test_fluid_satellite_aggregate_and_direction_independence(self):
        self.verify(FluidTransport)

    @unittest.skipUnless(importlib.util.find_spec('simgrid'), 'requires native SimGrid')
    def test_simgrid_native_shared_rf_resources(self):
        from astermind.engine.simgrid_transport import SimGridTransport
        self.verify(SimGridTransport)

    @unittest.skipUnless(importlib.util.find_spec('ns'), 'requires native ns-3')
    def test_ns3_packet_rf_arbitration(self):
        from astermind.engine.ns3_transport import Ns3Transport
        self.verify(Ns3Transport, packet=True)

