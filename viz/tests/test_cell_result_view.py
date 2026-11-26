import json
from pathlib import Path
import tempfile
import unittest

from viz.cell_result_view import build_view, report


def fixture():
    native = dict(status='completed', configuration=dict(cells=[dict(id='cell', placements=[
        dict(rank=0, device='edge-a.gpu0', pool='main')])], communication={}),
        batches=[dict(cell_id='cell', step_id='main:0', pool_name='main', calls=[
            dict(rank=0, kind='GEMM', call_id='a', start_time_ns=0, end_time_ns=10, communication=False),
            dict(rank=0, kind='VECTOR', call_id='b', start_time_ns=5, end_time_ns=15, communication=False),
            dict(rank=0, kind='COMMUNICATION', call_id='c', start_time_ns=10, end_time_ns=20, communication=True)])],
        cells={'cell': dict(steps=[], scheduler_runtime={})},
        requests={'request-0': dict(status='completed', arrival_ns=0, completed_ns=25, received_tokens=2,
            metrics=dict(ttft_ns=20, tpot_ns=5, e2e_ns=25))},
        measurement=dict(simulated_time_ns=60_000_000_000), communication_events=[])
    return native, dict(model='qwen3-4b', hardware=dict(satellite_count=1), deployment={})


class CellResultViewTests(unittest.TestCase):
    def test_result_mode_uses_saved_effective_configuration(self):
        native, submitted = fixture()
        submitted['compute_mode'] = 'batch_roofline'
        self.assertEqual(build_view(native, submitted)['summary']['compute_mode'], 'batch_roofline')
        native['configuration']['compute_mode'] = 'fast_mode'
        self.assertEqual(build_view(native, submitted)['summary']['compute_mode'], 'fast_mode')

    def test_overlapping_intervals_do_not_inflate_rank_wall_time(self):
        view = build_view(*fixture())
        row = view['summary']['ranks'][0]
        self.assertEqual((row['compute_ns'], row['communication_ns'], row['overlap_ns']), (15, 10, 5))
        self.assertEqual(row['compute_ns']+row['exposed_communication_ns']+row['waiting_ns'], row['wall_ns'])
        self.assertEqual(row['satellite'], 'edge-a')

    def test_request_completion_is_distinct_from_simulation_horizon(self):
        summary = build_view(*fixture())['summary']
        self.assertEqual(summary['completion_ns'], 25)
        self.assertEqual(summary['simulation_ns'], 60_000_000_000)
        self.assertEqual(summary['completed'], 1)
        self.assertEqual(summary['failed'], 0)

    def test_transfer_bytes_are_service_bytes_not_repeated_per_hop(self):
        native, submitted = fixture()
        native['communication_events'] = [
            dict(kind='admit', id='f', attempt_id=1, time_ns=0, path=['a', 'switch', 'b']),
            dict(kind='service', id='f', attempt_id=1, start_ns=0, end_ns=2, application_bytes=100),
            dict(kind='service', id='f', attempt_id=1, start_ns=2, end_ns=5, application_bytes=50),
            dict(kind='p2p_complete', id='f', attempt_id=1, time_ns=10)]
        network = build_view(native, submitted)['summary']['network']
        self.assertEqual(network['application_bytes'], 150)
        self.assertEqual(network['attempts'], 1)
        self.assertEqual(network['paths'][0]['service_gbps'], 240)
        self.assertEqual(network['concurrency'][-1][1], 0)

    def test_kv_curve_applies_allocation_and_release_not_cache_misses(self):
        native, submitted = fixture()
        native['cells']['cell']['steps'] = [dict(end_time_s=1, step=dict(kv_cache_events=[
            dict(action='allocate', tier='hbm', bytes=100), dict(action='miss', tier='hbm', bytes=100)])),
            dict(end_time_s=2, step=dict(kv_cache_events=[dict(action='release', tier='hbm', bytes=100)]))]
        memory = build_view(native, submitted)['summary']['memory'][0]
        self.assertEqual(memory['samples'], [[0, 0], [1e9, 100], [2e9, 0]])
        self.assertEqual(memory['peak_kv_bytes'], 100)

    def test_simultaneous_flow_replacement_does_not_create_false_concurrency_spike(self):
        native, submitted = fixture()
        native['communication_events'] = [
            dict(kind='admit', id='a', attempt_id=1, time_ns=0, path=['a','b']),
            dict(kind='admit', id='b', attempt_id=1, time_ns=5, path=['a','b']),
            dict(kind='p2p_complete', id='a', attempt_id=1, time_ns=5),
            dict(kind='p2p_complete', id='b', attempt_id=1, time_ns=10)]
        points = build_view(native, submitted)['summary']['network']['concurrency']
        self.assertEqual(max(value for _, value in points), 1)

    def test_operator_filters_pagination_and_missing_historical_fields(self):
        native, submitted = fixture()
        temporary_root = Path(__file__).resolve().parents[1]/'.runtime/verification/result-restoration'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(temporary_root.resolve()))
            root = Path(directory)
            (root/'native_result.json').write_text(json.dumps(native))
            (root/'configuration.json').write_text(json.dumps(submitted))
            result = report(root, dict(section=['operators'], kind=['GEMM'], rank=['cell/0'], limit=['1']))
            self.assertEqual(result['total'], 1)
            self.assertEqual(result['items'][0]['name'], 'a')
            summary = report(root, {})
            self.assertIsNone(summary['ranks'][0]['weight_bytes'])
            with self.assertRaisesRegex(ValueError, 'Rank'):
                report(root, dict(section=['timeline'], rank=['missing']))


if __name__ == '__main__':
    unittest.main()
