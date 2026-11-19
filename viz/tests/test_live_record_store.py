from pathlib import Path
import gzip
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

VIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIZ))
from event_catalog import annotate, EventWindow
from live_evidence import CapacitySnapshots, expand_event


class DisplayEvidenceTests(unittest.TestCase):
    def test_archived_tables_are_saved_once_and_events_expand_without_data_loss(self):
        temporary = VIZ / '.runtime/tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            output = Path(folder)
            self.assertTrue(output.resolve().is_relative_to(temporary.resolve()))
            archive = CapacitySnapshots(output)
            for i in range(5):
                row = dict(kind='v1_allocation',time_ticks=str(i),capacities={'a':3,'b':7},flows={'f':{'rate':i}})
                encoded = json.loads(json.dumps(archive.encode(row)))
                self.assertNotIn('capacities', encoded)
                self.assertEqual(expand_event(encoded, output), row)
                self.assertEqual(annotate(encoded)['capacity_pool_count'], 2)
            self.assertEqual(len(list((output/'runtime-capacities').glob('*.gz'))), 1)
            row['capacities']={'a':2,'b':7}
            self.assertEqual(expand_event(archive.encode(row),output),row)
            self.assertEqual(len(list((output/'runtime-capacities').glob('*.gz'))), 2)

    def test_http_summaries_leave_full_allocation_evidence_untouched(self):
        row = dict(kind='v1_allocation', time_s=1, time_ticks='1000000000000000001',
                   capacities={f'pool-{i}':100 for i in range(14324)}, flows={'f':{'rate':2}})
        displayed = annotate(row)
        self.assertNotIn('capacities', displayed)
        self.assertNotIn('flows', displayed)
        self.assertEqual(displayed['capacity_pool_count'], 14324)
        self.assertEqual(displayed['flow_count'], 1)
        self.assertEqual(displayed['time_ticks'], row['time_ticks'])
        self.assertEqual(annotate(displayed), displayed)
        self.assertLess(len(json.dumps(displayed)), 400)
        self.assertEqual(len(row['capacities']), 14324)
        window = EventWindow()
        result = window.capture([row])
        self.assertNotIn('capacities', result['recent']['diagnostic'][0])


class CompactEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from runtime_environment import activate
        activate()
        from live_record_store import LiveRecords, compact_communication_trace
        cls.Records = LiveRecords
        cls.install = staticmethod(compact_communication_trace)

    def test_all_records_and_exact_ticks_survive_chunking_and_supply_changes(self):
        rows = []
        records = self.Records(chunk_size=3)
        for i in range(20):
            row = dict(kind='allocation', time_ticks=str(10**18+i),
                       capacities={'a':100 if i < 10 else 50}, flows={str(i):{'rate':i}})
            rows.append(row)
            records.append(row)
        self.assertEqual(list(records), rows)
        self.assertEqual(records[-1], rows[-1])
        self.assertEqual(list(records[2:17]), rows[2:17])
        self.assertEqual(json.loads(json.dumps(records)), rows)
        self.assertLessEqual(len(records.tail), 3)

    def test_identical_capacity_tables_share_storage_without_merging_events(self):
        records = self.Records(chunk_size=100)
        for i in range(30):
            records.append(dict(kind='v1_allocation', time_ticks=str(i), capacities={'a':2}, flows={}))
        self.assertEqual(len(records), 30)
        self.assertTrue(all(row['capacities'] is records[0]['capacities'] for row in records))

    def test_old_compressed_chunks_leave_memory_and_remain_randomly_readable(self):
        temporary = VIZ / '.runtime/tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            self.assertTrue(Path(folder).resolve().is_relative_to(temporary.resolve()))
            records = self.Records(chunk_size=3, output=folder)
            rows = [dict(kind='allocation',time_ticks=str(i),capacities={'a':3},flows={str(i):i}) for i in range(40)]
            records.extend(rows)
            self.assertFalse(isinstance(records.chunks, list))
            self.assertGreater(records.chunks.stored_bytes, 0)
            self.assertEqual(list(records), rows)
            self.assertEqual(records[2], rows[2])
            self.assertEqual(records[-1], rows[-1])
            self.assertEqual(json.loads(json.dumps(records)), rows)
            self.assertLessEqual(len(records.tail), 3)

    def test_native_append_stamps_current_tick_and_existing_evidence_is_preserved(self):
        loop = SimpleNamespace(now_s=SimpleNamespace(ticks=12))
        host = SimpleNamespace(timeline=loop, communication_policy=SimpleNamespace(trace=[dict(kind='allocation',time_ticks='4',capacities={})]))
        self.install(host)
        host.communication_policy.trace.append(dict(kind='service', application_bytes=7))
        self.assertEqual([r['time_ticks'] for r in host.communication_policy.trace], ['4','12'])
        records = host.communication_policy.trace
        self.install(host)
        self.assertIs(host.communication_policy.trace, records)

    def test_live_projection_merges_streams_lazily_and_keeps_equal_tick_order(self):
        from native_scene_live import DeviceLiveSession
        session = DeviceLiveSession('projection-test', None, SimpleNamespace(physical={'devices':[]},requests=[]), False)
        physical = SimpleNamespace(trace=SimpleNamespace(time={}))
        engine = SimpleNamespace(net=SimpleNamespace(physical=physical, owner={}, edge_by_id={}),
                                 now=0, end=10,
                                 log=[dict(kind='external_callback',time_ticks='4')],
                                 communication_policy=SimpleNamespace(trace=[dict(kind='service',time_ticks='2')]))
        view = session.record_view(engine)
        self.assertEqual([row['time_ticks'] for row in view.log], ['2','4'])
        engine.log.append(dict(kind='control_boundary',time_ticks='5'))
        engine.communication_policy.trace.append(dict(kind='service',time_ticks='5'))
        view = session.record_view(engine)
        self.assertEqual([row['kind'] for row in view.log], ['v1_service','external_callback','control_boundary','v1_service'])
        self.assertEqual(len(session.record_view(engine).log), 4)

    def test_recording_export_uses_existing_configuration_without_loading_large_result(self):
        from scene_recording import build_recording
        temporary = VIZ / '.runtime/tmp'
        temporary.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary) as folder:
            output = Path(folder)
            self.assertTrue(output.resolve().is_relative_to(temporary.resolve()))
            event = dict(kind='v1_allocation',time_s=0,time_ticks='0',capacities={'a':3},flows={})
            with gzip.open(output/'runtime_events.jsonl.gz','wt') as stream:
                stream.write(json.dumps(event)+'\n')
            result = build_recording(output,configuration={'physical':{'devices':[]}})
            self.assertEqual(result['end_s'], 0)
            self.assertTrue((output/'native-replay/scene-recording.json').is_file())


if __name__ == '__main__':
    unittest.main()
