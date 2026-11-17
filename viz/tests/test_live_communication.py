"""V1 streaming uses real whole-path attempts without changing solver timing."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from live_communication import CommunicationPublisher

TEST_TMP = Path(__file__).resolve().parents[1] / '.runtime/tmp'
TEST_TMP.mkdir(parents=True, exist_ok=True)


class NativeCommunicationTests(unittest.TestCase):
    def test_native_attempts_supply_continuous_busy_intervals_and_prefix_state(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as folder:
            output = Path(folder)
            self.assertTrue(output.resolve().is_relative_to(TEST_TMP.resolve()))
            publisher = CommunicationPublisher(output)

            def append(*events):
                with gzip.open(output / 'runtime_events.jsonl.gz', 'at', encoding='utf8') as stream:
                    for event in events:
                        stream.write(json.dumps(event) + '\n')
                publisher.poll()
                return json.loads((output / 'native-replay/live-communication.json').read_text(encoding='utf8'))

            segment = dict(transfer_id='a/segment/0', source_node_id='sat-a', target_node_id='sat-b')
            index = append(dict(kind='v1_admit', id='attempt-a', task='request-a', time_s=1,
                                path_legs=[segment], source_node_id='sat-a', target_node_id='sat-b'),
                           dict(kind='v1_service', id='attempt-a', time_s=1.1))
            self.assertEqual(index['busy_intervals'], [[1, 1.1]])
            index = append(dict(kind='v1_p2p_complete', id='attempt-a', time_s=1.2))
            self.assertEqual(index['busy_intervals'], [[1, 1.2]])
            page = json.loads((output / 'native-replay' / index['pages'][-1]['file']).read_text(encoding='utf8'))
            self.assertEqual(page['initial']['active_transfers'][0]['v1_flow_id'], 'attempt-a')
            self.assertEqual(index['through_order'], 3)
            self.assertEqual(publisher.v1_attempts, {})

    def test_failure_by_task_closes_only_its_own_native_attempt(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as folder:
            output = Path(folder)
            self.assertTrue(output.resolve().is_relative_to(TEST_TMP.resolve()))
            publisher = CommunicationPublisher(output)
            rows = [dict(kind='v1_admit', id='attempt-a', task='request-a', time_s=1),
                    dict(kind='v1_admit', id='attempt-b', task='request-b', time_s=1.1),
                    dict(kind='v1_failed', id='request-a', time_s=1.2)]
            with gzip.open(output / 'runtime_events.jsonl.gz', 'wt', encoding='utf8') as stream:
                for row in rows:
                    stream.write(json.dumps(row) + '\n')
            publisher.poll()
            self.assertEqual(set(publisher.v1_attempts), {'attempt-b'})
            self.assertEqual(publisher.intervals, [[1, 1.2]])


if __name__ == '__main__':
    unittest.main()
