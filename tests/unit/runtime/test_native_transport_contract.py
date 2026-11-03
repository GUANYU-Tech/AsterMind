"""Real native backends reject corrupting operations without losing progress."""
import importlib.util
from types import SimpleNamespace
import unittest

from orbitserve.runtime.engine.ns3_transport import Ns3Transport
from orbitserve.runtime.engine.simgrid_transport import SimGridTransport


@unittest.skipUnless(importlib.util.find_spec('ns') and importlib.util.find_spec('simgrid'),
                     'Install OrbitServe[simgrid,ns3] for native contract checks')
class NativeTransportContractTests(unittest.TestCase):
    def test_duplicate_id_bad_time_and_oversized_payload_do_not_corrupt_flow(self):
        for backend in (SimGridTransport, Ns3Transport):
            with self.subTest(backend=backend.__name__):
                edges = (dict(endpoint_a='a', endpoint_b='b', link_id='ab', kind='isl') for _ in range(1))
                engine = backend(('a', 'b'), edges,
                                 {'isl': SimpleNamespace(bandwidth_bps=8_000_000)})
                self.addCleanup(engine.close)
                leg = dict(source_node_id='a', target_node_id='b', link_id='ab')
                engine.configure({engine.resource(leg): 8_000_000})
                engine.start('one', leg, 1000)
                with self.assertRaisesRegex(ValueError, 'unique'):
                    engine.start('one', leg, 1)
                with self.assertRaisesRegex(ValueError, 'size limit'):
                    engine.start('huge', leg, 1e300)
                first = engine.advance_until(.00025)
                self.assertGreater(first['bytes_sent']['one'], 0.)
                for time_s in (float('nan'), float('inf'), -1., 0., True):
                    with self.subTest(time_s=time_s), self.assertRaises(ValueError):
                        engine.advance_until(time_s)
                second = engine.advance_until(.1)
                self.assertEqual(second['completed'], ['one'])
                self.assertAlmostEqual(first['bytes_sent']['one'] + second['bytes_sent']['one'], 1000.)
