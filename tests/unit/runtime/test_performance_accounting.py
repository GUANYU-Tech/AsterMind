"""Nested diagnostic timers must conserve time even when callbacks fail."""

from dataclasses import asdict
import unittest
from unittest.mock import patch

from orbitserve.runtime.metrics.performance import WallProfiler
from orbitserve.runtime.engine.cell_bridge import CellTransfer, CellTransferEvent


class PerformanceAccountingTests(unittest.TestCase):
    def test_nested_exclusive_times_close_and_ignore_setup(self):
        profiler = WallProfiler(root_label='loop')
        inner = profiler.wrap('child', lambda: 7)
        outer = profiler.wrap('loop', inner)
        self.assertEqual(inner(), 7)
        self.assertEqual(profiler.report(), {})
        with patch('orbitserve.runtime.metrics.performance.perf_counter', side_effect=(0., 1., 3., 5.)):
            self.assertEqual(outer(), 7)
        report = profiler.report()
        self.assertEqual(report['loop']['inclusive_wall_s'], 5.)
        self.assertEqual(sum(row['exclusive_wall_s'] for row in report.values()), 5.)

    def test_failed_callback_does_not_leave_active_scope(self):
        profiler = WallProfiler()
        def failed():
            raise ValueError('callback failed')
        with self.assertRaisesRegex(ValueError, 'callback failed'):
            profiler.wrap('failed', failed)()
        self.assertEqual(profiler.report()['failed']['calls'], 1)

    def test_transfer_records_preserve_the_existing_schema(self):
        request = CellTransfer('x', 'a', 'b', 42., 3., correlation_id='r')
        self.assertEqual(request.model_dump(), asdict(request))
        for kind in ('completed', 'failed'):
            event = CellTransferEvent(kind, 'x', 'a', 'b', 42., 'native', 'tp',
                                      'r', ('a', 'gate', 'b'), 5., 7., 'failure')
            expected = asdict(event)
            if kind == 'completed':
                expected.pop('failure_time_ns')
                expected.pop('reason')
            else:
                expected.pop('completion_time_ns')
                expected['remaining_bytes'] = event.size_bytes
            actual = event.model_dump()
            self.assertEqual(actual, expected)
            actual['source'] = 'changed'
            self.assertEqual(event.source, 'a')


if __name__ == '__main__':
    unittest.main()
