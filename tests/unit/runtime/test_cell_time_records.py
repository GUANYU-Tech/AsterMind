"""Native tick boundaries must agree between distribution and batch records."""
import math
import unittest
from types import SimpleNamespace

from astermind.engine.cell_bridge import bind_batch_executor
from astermind.engine.timeline import GlobalTimeline, nanoseconds
from astermind.engine.audit import audit


class CellTimeRecordTests(unittest.TestCase):
    def test_distribution_and_batch_share_the_same_nanosecond(self):
        at_s, expected_ns = 64.2947864, 64294786400
        self.assertEqual(math.ceil(at_s * 1e9), expected_ns + 1)
        loop = GlobalTimeline(start_s=at_s)
        starts = []

        class NativeExecutor:
            def __init__(self, *args):
                pass

            def _phase(self, cell, pool, operation, correlation, sends, done, record):
                done()

            def start_batch(self, cell, step, components, execution, done, **kwargs):
                starts.append(nanoseconds(loop.now_s))
                done(loop.now_s)

        network = SimpleNamespace(boundary_listeners=[], subscribe=lambda fn: None)
        executor = bind_batch_executor(NativeExecutor, loop, network, ())
        pool = SimpleNamespace(physical_ranks_for_pipeline_stage=lambda stage: (0, 1))
        execution = SimpleNamespace(
            deployment=SimpleNamespace(worker_pools={'main': pool}),
            input_distribution_in_graph=False, last_step_token_count=64,
        )
        executor.start_batch('cell', 'step', None, execution, lambda at: None)
        record = executor.input_distributions[0]
        self.assertEqual(record['ready_ns'], expected_ns)
        self.assertEqual(record['completed_ns'], expected_ns)
        self.assertEqual(starts, [expected_ns])

    def test_audit_still_rejects_distribution_one_nanosecond_after_compute(self):
        start = 64294786400
        distribution = dict(cell_id='cell', step_id='step', completed_ns=start, phases=[])
        value = dict(requests={}, transfers=[], input_distributions=[distribution], batches=[
            dict(cell_id='cell', step_id='step', start_time_ns=start, end_time_ns=start,
                 native_communication_duration_charged_s=0, calls=[
                     dict(call_id='compute', occurrence=0, communication=False,
                          start_time_ns=start, end_time_ns=start, host_ready_time_ns=start,
                          dependencies=[], native_prediction_s=0)
                 ])
        ])
        self.assertTrue(audit(value)['passed'])
        distribution['completed_ns'] += 1
        with self.assertRaisesRegex(AssertionError, 'after batch start'):
            audit(value)


if __name__ == '__main__':
    unittest.main()
