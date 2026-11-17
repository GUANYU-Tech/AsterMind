"""Focused storage checks; no model loading, simulation or server required."""
from pathlib import Path
import json
import tempfile
import unittest

from astermind.common.result_storage import load_result, restore_result, write_result, result_digest


class NativeResultStorageTest(unittest.TestCase):
    def test_snapshot_references_restore_all_events_and_share_tables(self):
        capacity = {f'pool-{i}': i+1.5 for i in range(64)}
        report = dict(schema_version=2, status='completed', communication_events=[
            dict(kind='allocation', time_ticks=str(i), capacities=capacity,
                 flows={'flow':dict(rate_bytes_per_ns=.5,demand={'pool-0':1})})
            for i in range(100)
        ])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'result.json'
            digest = write_result(path, report)
            stored = json.loads(path.read_text())
            self.assertEqual(len(stored['_storage']['capacity_snapshots']), 1)
            self.assertNotIn('capacities', stored['communication_events'][0])
            restored = load_result(path)
            self.assertEqual(restored, report)
            self.assertIs(restored['communication_events'][0]['capacities'], restored['communication_events'][-1]['capacities'])
            self.assertEqual(result_digest(path), digest['sha256'])
            self.assertIn('capacities', report['communication_events'][0])

    def test_capacity_changes_remain_distinct_and_corruption_is_rejected(self):
        report = dict(communication_events=[dict(kind='allocation', capacities={'pool':value}, flows={}) for value in (1,2,1)])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'result.json'
            write_result(path, report)
            self.assertEqual(load_result(path), report)
            stored = json.loads(path.read_text())
            self.assertEqual(len(stored['_storage']['capacity_snapshots']), 2)
            next(iter(stored['_storage']['capacity_snapshots'].values()))['pool'] = 3
            with self.assertRaisesRegex(ValueError, 'checksum'):
                restore_result(stored)

    def test_legacy_and_unrelated_json_load_unchanged(self):
        report = dict(communication_events=[dict(kind='allocation',capacities={'pool':1},flows={})])
        self.assertIs(restore_result(report), report)
        self.assertEqual(restore_result([1,2]), [1,2])


if __name__ == '__main__':
    unittest.main()
