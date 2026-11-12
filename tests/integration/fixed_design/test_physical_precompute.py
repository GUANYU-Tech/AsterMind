"""Parallel physical attributes retain serial semantics and bounded payloads."""
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from astermind.engine import LinkProfile
from astermind.engine.state.physical_dataset import _chunks
from tests.unit.runtime.test_dynamic_network import network


class PhysicalPrecomputeTests(unittest.TestCase):
    def test_parallel_prepared_numeric_columns_match_direct_time_varying_geometry(self):
        from astermind.engine import NetworkState

        original = self.state()
        original.trace.ephemeris_samples.extend(
            {'satellite_id': node, 'time_s': 10., 'position_itrs_km': (7000., i*100., 0.)}
            for i, node in enumerate(sorted(original.nodes)))
        direct = NetworkState(original.trace, original.spec)
        cached = NetworkState(original.trace, original.spec, _defer_compile=True)
        with tempfile.TemporaryDirectory() as folder:
            cached.save_physical_state(Path(folder)/'varying.sqlite', workers=2)
            for t in (0., 5., 9.99, 10., 14.99, 15., 20.):
                self.assertEqual(json.loads(json.dumps(cached.snapshot(t))),
                                 json.loads(json.dumps(direct.snapshot(t))))

    def test_parallel_worker_error_releases_encode_barrier_without_publishing(self):
        state = self.state()
        state.edges = ({**state.edges[0], 'profile_id': 'missing'}, *state.edges[1:])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'failed-parallel.sqlite'
            with self.assertRaises(KeyError):
                state.save_physical_state(path, workers=2)
            self.assertFalse(path.exists())

    def test_symmetric_budget_is_computed_once_per_visible_link_sample(self):
        from astermind.engine import NetworkState

        original = self.state()
        state = NetworkState(original.trace, original.spec, _defer_compile=True)
        calls = []
        calculate = LinkProfile.attributes

        def counted(profile, distance, time_s):
            calls.append((distance, time_s))
            return calculate(profile, distance, time_s)

        with tempfile.TemporaryDirectory() as folder, patch.object(LinkProfile, 'attributes', counted):
            state.save_physical_state(Path(folder)/'once.sqlite')
        # Only the 5 s contact sample is visible; both directions have the same
        # profile as the base geometry budget and must reuse its result.
        self.assertGreater(len(calls), 0)
        self.assertLessEqual(len(calls), len(state.edges))
        self.assertEqual({t for _, t in calls}, {5.})

    def test_cached_allocation_matches_direct_for_both_handover_and_score_modes(self):
        from astermind.engine import NetworkState
        from tests.unit.runtime.test_radio_capacity import access_network

        for handover in ('retain_valid', 'quality_first'):
            for ranking in ('nearest_distance', 'highest_capacity'):
                with self.subTest(handover=handover, ranking=ranking):
                    direct = access_network(handover_policy=handover, ground_access_policy=ranking)
                    direct.trace.trace_id = 'policy-cache-equivalence'
                    cached = NetworkState(direct.trace, direct.spec,
                        [{'ingress_id': 'g', 'lat_deg': 0., 'lon_deg': 0.}], _defer_compile=True)
                    with tempfile.TemporaryDirectory() as folder:
                        cached.save_physical_state(Path(folder)/'policy.sqlite')
                        for time_s in (0., .4, 1., 5., 9.5, 10.):
                            self.assertEqual(json.loads(json.dumps(cached.snapshot(time_s))),
                                             json.loads(json.dumps(direct.snapshot(time_s))))

    def test_deferred_allocation_and_snapshots_never_recalculate_saved_budgets(self):
        from astermind.engine import NetworkState

        original = self.state()
        expected = [json.loads(json.dumps(original.snapshot(t))) for t in (0., 5., 10., 15., 20.)]
        deferred = NetworkState(original.trace, original.spec, _defer_compile=True)
        compile_timeline = NetworkState._compile

        def checked_compile(state, *args, **kwargs):
            self.assertIsNotNone(state.physical_attributes)
            with patch.object(LinkProfile, 'attributes', side_effect=AssertionError('budget recalculated during allocation')):
                return compile_timeline(state, *args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            with patch.object(NetworkState, '_compile', checked_compile):
                deferred.save_physical_state(Path(folder)/'prepared.sqlite')
            with patch.object(LinkProfile, 'attributes', side_effect=AssertionError('budget recalculated after preparation')):
                actual = [json.loads(json.dumps(deferred.snapshot(t))) for t in (0., 5., 10., 15., 20.)]
            self.assertEqual(actual, expected)
            self.assertGreater(deferred.precompute_timings['attributes_compute_s'], 0.)
            self.assertFalse(deferred._compile_pending)

    def test_compact_cache_preserves_values_and_obeys_zero_byte_budget(self):
        from astermind.engine.state.attribute_cache import AttributeBlock
        from astermind.engine.state.physical_dataset import PhysicalAttributes, _edge_blocks

        state = self.state()
        rows = [row for _, _, block in _edge_blocks(state, state.edges[0], encoded=False) for row in block]
        compact = AttributeBlock(rows)
        self.assertEqual([compact.at(i) for i in range(len(rows))], json.loads(json.dumps([row[1] for row in rows])))
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'cache.sqlite'
            state.save_physical_state(target)
            reader = PhysicalAttributes(target)
            with patch('astermind.engine.state.physical_dataset.DECODED_READ_CACHE_BYTES', 0):
                for t, value in rows:
                    self.assertEqual(reader.geometry_at(state.edges[0]['link_id'], t), json.loads(json.dumps(value['geometry'])))
                    self.assertEqual(reader.decoded_bytes, 0)
            self.assertGreater(reader.block_loads, 0)

    def test_runtime_cache_uses_physical_boundaries_and_reads_small_data_once(self):
        from astermind.engine import NetworkState
        from astermind.engine.state.physical_dataset import PhysicalAttributes
        state=network(edges=(('a','b',5.,15.),),schedule=((6.25,.5),))
        state.trace.trace_id='cached-test'
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'cached.sqlite'
            state.save_physical_state(path)
            saved=NetworkState(state.trace,state.spec,physical_state=path)
            edge=saved.edges[0]
            reader=saved.physical_attributes
            self.assertIsNotNone(reader._packed)
            with patch.object(reader,'connect',side_effect=AssertionError('reopened database')):
                for i in range(5000):
                    saved.distance(edge,5.+i/10000.)
                self.assertEqual(reader.at.cache_info().misses,1)
                for t in (0.,4.99,5.,6.249,6.25,6.251,14.99,15.,19.9,20.):
                    self.assertEqual(json.loads(json.dumps(saved.snapshot(t))),json.loads(json.dumps(state.snapshot(t))))
            with patch('astermind.engine.state.physical_dataset.COMPRESSED_READ_CACHE_BYTES',0):
                streamed=PhysicalAttributes(path)
            self.assertIsNone(streamed._packed)
            self.assertEqual(streamed.at(edge['link_id'],6.25),reader.at(edge['link_id'],6.25))

    def state(self):
        state = network(edges=tuple((a,b,5.,15.) for a,b in itertools.combinations('abcdefg',2)))
        state.trace.trace_id = 'parallel-test'
        return state

    def test_spawn_parallel_matches_serial_database_bytes_and_progress(self):
        state = self.state()
        with tempfile.TemporaryDirectory() as folder:
            first, second = (Path(folder)/name for name in ('serial.sqlite','parallel.sqlite'))
            state.save_physical_state(first)
            progress=[]
            state.save_physical_state(second,workers=2,progress=progress.append)
            self.assertEqual(first.read_bytes(),second.read_bytes())
            self.assertIn('0/21 links; 2 workers',progress[0])
            self.assertIn('21/21 links; 2 workers',progress[-1])
            for context,edges in _chunks(state):
                self.assertLessEqual(len(edges),8)
                self.assertEqual(set(context['nodes']),{e[k] for e in edges for k in ('endpoint_a','endpoint_b')})
                self.assertNotIn('trace',context)
                self.assertNotIn('link_frames',context)

    def test_blocked_times_skip_budget_calculation_and_failure_is_not_published(self):
        state=self.state()
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/'failed.sqlite'
            with patch.object(LinkProfile,'attributes',side_effect=RuntimeError('budget failed')):
                # Endpoints are geometrically blocked, so only the visible
                # boundary at 5 s reaches the patched budget calculation.
                self.assertEqual(state.snapshot(0)['links'][0]['bandwidth_bps'],0)
                with self.assertRaisesRegex(RuntimeError,'budget failed'):
                    state.save_physical_state(target)
            self.assertFalse(target.exists())
            self.assertTrue(target.with_name(target.name+'.tmp').exists())

    def test_invisible_adaptive_pairs_are_removed_without_dropping_future_contacts(self):
        from types import SimpleNamespace

        from astermind.engine import NetworkNodeSpec, NetworkSpec, NetworkState
        spec=NetworkSpec(
            (NetworkNodeSpec('a','communication',4,2),NetworkNodeSpec('b','communication',4,2),NetworkNodeSpec('g','ground',sgl_terminals=1)),
            {'sgl':LinkProfile(100,0),'communication_isl':LinkProfile(100,0)},
            communication_topology='explicit',explicit_pairs=(('a','b'),))
        trace=SimpleNamespace(time={'start_s':0,'end_s':20},satellite_catalog=[{'satellite_id':i} for i in 'ab'],
            ephemeris_samples=[{'satellite_id':i,'time_s':t,'position_itrs_km':(7000,0,0)} for i in 'ab' for t in (0,20)],
            isl_windows=[{'endpoint_a': 'a','endpoint_b': 'b','start_s': 0,'end_s': 20}],
            sgl_windows=[{'ingress_id': 'g','satellite_id': 'b','start_s': 5,'end_s': 15}])
        state=NetworkState(trace,spec,[{'ingress_id': 'g','lat_deg': 0,'lon_deg': 0}])
        self.assertEqual({e['link_id'] for e in state.edges},{'communication_isl:a:b','sgl:b:g'})
        self.assertTrue(any(e['kind']=='sgl' and e['state']=='UP' for e in state.snapshot(5)['links']))
        self.assertFalse(any(e['kind']=='sgl' and e['state']=='UP' for e in state.snapshot(0)['links']))
