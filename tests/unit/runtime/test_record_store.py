import io
import json
import unittest

from simulator.record_store import CompressedRecords, RecordView


class RecordStoreTests(unittest.TestCase):
    def test_chunk_boundaries_slices_and_json_keep_every_record(self):
        expected = [dict(time_s=i/1000, kind='event', nested=[i, {'x': i}]) for i in range(10003)]
        records = CompressedRecords(chunk_size=37)
        records.extend(expected)
        self.assertEqual(list(records), expected)
        self.assertEqual(records, expected)
        self.assertNotEqual(records, [])
        self.assertIn(expected[999], records)
        self.assertEqual(list(records[-150:13:-7]), expected[-150:13:-7])
        for i in (0, 36, 37, 74, 9999, -1):
            self.assertEqual(records[i], expected[i])
        with self.assertRaises(IndexError):
            records[len(expected)]
        self.assertLessEqual(len(records.tail), 37)
        self.assertLess(sum(map(len, records.chunks)), len(json.dumps(expected)))
        target = io.StringIO()
        json.dump({'events': records}, target)
        self.assertEqual(json.loads(target.getvalue())['events'], expected)
        self.assertEqual(json.loads(json.dumps(records)), expected)

    def test_active_graph_is_not_compressed_before_completion(self):
        records = CompressedRecords(chunk_size=1, sealed=lambda row: row['end'] is not None)
        active = dict(end=None, calls=[dict(end=None)])
        records.append(active)
        records.append(dict(end=2, calls=[]))
        self.assertEqual(records.saved, 0)
        active['end'] = active['calls'][0]['end'] = 3
        records.append(dict(end=4, calls=[]))
        self.assertEqual(records.saved, 2)
        self.assertEqual(records[0], dict(end=3, calls=[dict(end=3)]))

    def test_projection_does_not_retain_or_mutate_full_history(self):
        records = CompressedRecords(chunk_size=3)
        records.extend(dict(i=i, endpoint='gpu') for i in range(50))
        view = RecordView(records, range(len(records)), lambda r: dict(r, endpoint='satellite'))
        self.assertEqual(view[-1], dict(i=49, endpoint='satellite'))
        self.assertEqual(list(view[20:23]), [dict(i=i, endpoint='satellite') for i in range(20,23)])
        self.assertEqual(records[-1]['endpoint'], 'gpu')


if __name__ == '__main__':
    unittest.main()
