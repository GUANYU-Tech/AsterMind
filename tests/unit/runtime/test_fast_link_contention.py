"""Exercise Fast Mode phases against the real allocator on a fixed shared ISL."""
import unittest
from types import SimpleNamespace

from astermind.common.bootstrap import activate
activate()
from astermind.engine.comm.event_runtime import RuntimeCommunication
from astermind.engine.comm.frame_surrogate import CommPolicy
from astermind.engine.comm.operator import CommunicationOperator
from astermind.engine.compute.fast_mode.executor import FastModeExecutor
from astermind.engine.timeline import GlobalTimeline
from tests.comm_fixture import Frames


class FastLinkContentionTests(unittest.TestCase):
    def durations(self, pairs):
        clock = GlobalTimeline()
        network = RuntimeCommunication(Frames(), clock, CommPolicy(execution='runtime'))
        cells = []
        for i, pair in enumerate(pairs):
            cell = FastModeExecutor(f'cell-{i}', clock, network, ())
            cell.link_contention = True
            transfer = CommunicationOperator('send_recv', 'send_recv', pair, 2_000_000, 'payload')
            cell.pricing = SimpleNamespace(runtime_phases=lambda batch, op=transfer: [((op,),)])
            batch = SimpleNamespace(replica_tokens=(1,), stage_compute_s=(0.,))
            cell.start_batch(cell.cell_id, 'step', batch, None, lambda now: None,
                             token_count=1, replica_tokens=(1,))
            cells.append(cell)
        clock.run()
        records = [cell.records[0] for cell in cells]
        for record in records:
            self.assertEqual(record['status'], 'completed')
            self.assertEqual(record['runtime_reason'], 'link_bandwidth_contention')
        return [record['communication_s'] for record in records]

    def test_simultaneous_batches_share_the_bottleneck(self):
        pairs = [('g0', 'g1'), ('g2', 'g3')]
        single, = self.durations(pairs[:1])
        concurrent = self.durations(pairs)
        self.assertGreater(single, 0)
        self.assertTrue(all(duration > single for duration in concurrent))
        self.assertAlmostEqual(*concurrent)
