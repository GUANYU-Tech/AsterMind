"""Structural comparison must reject every simulation-value change."""
import unittest
from copy import deepcopy
from pathlib import Path
from tests.scripts.compare_structure_results import config_digest, differences, host_observation, normalize


class StructureComparisonTests(unittest.TestCase):
    def test_one_tick_change_and_event_reordering_are_not_tolerated(self):
        before = {'events': [{'time_ticks': '100', 'kind': 'ready'}, {'time_ticks': '100', 'kind': 'done'}]}
        for after in ({'events': [{'time_ticks': '101', 'kind': 'ready'}, before['events'][1]]},
                      {'events': list(reversed(before['events']))}):
            changes = list(differences(before, after))
            self.assertTrue(changes)
            self.assertTrue(all(not host_observation(*change) for change in changes))

    def test_only_exact_named_host_observation_paths_are_allowed(self):
        self.assertTrue(host_observation(('cells', 'c', 'pools', 'prefill', 'scheduler_runtime', 'worker_pid'), 1, 2))
        for path in (
            ('cells', 'c', 'execution_performance', 'phase_wall_time_s', 'kernel_prediction'),
            ('communication_runtime_stats', 'allocation_wall_s')):
            self.assertTrue(host_observation(path, 1., 2.))
        for path in (('measurement', 'unreviewed'), ('requests', 'r', 'completed_ticks'),
                     ('communication_runtime_stats', 'rate_allocations'),
                     ('cells', 'c', 'execution_performance', 'phase_wall_time_s', 'new_phase')):
            self.assertFalse(host_observation(path, 1., 2.))

    def test_ignored_measurements_still_require_valid_values(self):
        for value in (float('nan'), -1, True, '0'):
            with self.assertRaises(AssertionError):
                host_observation(('communication_runtime_stats', 'route_wall_s'), value, 1.)

    def test_added_fields_and_changed_numeric_types_are_differences(self):
        self.assertTrue(list(differences({'x': 1}, {'x': 1, 'y': 2})))
        self.assertTrue(list(differences({'x': 1}, {'x': 1.})))

    def test_replica_pid_exception_does_not_cover_other_replica_fields(self):
        path = ('cells','c','scheduler_runtime','replicas',0,'worker_pid')
        self.assertTrue(host_observation(path, 1, 2))
        self.assertFalse(host_observation((*path[:-1],'num_kv_blocks'), 1, 2))
        self.assertFalse(host_observation(('requests','r','replicas',0,'worker_pid'), 1, 2))

    def test_source_config_receipts_are_verified_and_semantic_changes_rejected(self):
        def report(root, width=8):
            config = {'model': str(root/'model.json'), 'tensor_parallel_size': width}
            return {'cells': {'c': {'scheduler_runtime': {
                'source_scheduler_config': config, 'source_scheduler_config_sha256': config_digest(config)}}}}
        a, b = Path('/base'), Path('/candidate')
        first = normalize(report(a), Path('/out-a'), a)
        second = normalize(report(b), Path('/out-b'), b)
        self.assertEqual(first, second)
        self.assertNotEqual(first, normalize(report(b,4), Path('/out-b'), b))
        tampered = report(a)
        tampered['cells']['c']['scheduler_runtime']['source_scheduler_config_sha256'] = '0'*64
        with self.assertRaises(AssertionError):
            normalize(tampered, Path('/out-a'), a)

    def test_traceback_normalizes_only_file_prefix_not_error_message(self):
        def error(root, line=42):
            return f'failure\nTraceback (most recent call last):\n  File "{root}/module.py", line {line}, in run\nRuntimeError: bad\n'
        a,b = Path('/base'),Path('/candidate')
        def normalized(root, line=42):
            return normalize({'cleanup_error':error(root,line)},Path('/out'),root)
        first = normalized(a)
        self.assertEqual(first,normalized(b))
        self.assertNotEqual(first,normalized(b,43))
        self.assertNotEqual(normalize({'prompt':error(a)},Path('/oa'),a),
                            normalize({'prompt':error(b)},Path('/ob'),b))
        for root in (a,b):
            message = {'cleanup_error':f'Traceback (most recent call last):\n  File "{root}/message only"\n'}
            self.assertEqual(normalize(message,Path('/out'),root),message)
        self.assertNotEqual(normalize('bad /base/model',Path('/oa'),a),
                            normalize('bad /candidate/model',Path('/ob'),b))

    def test_host_startup_completion_grouping_is_validated(self):
        initial = dict(phase='cells',completed=0,total=2,workers=2,active=['a','b'],wall_s=1.)
        middle = dict(phase='cells',completed=1,total=2,workers=2,active=['a'],wall_s=1.)
        final = dict(phase='cells',completed=2,total=2,workers=2,active=[],wall_s=1.)
        def report(rows):
            return {'cells': {'a': {}, 'b': {}}, 'measurement': {'startup_stages': rows}}
        def normalized(rows):
            return normalize(report(rows),Path('/out'),Path('/repo'))
        self.assertEqual(normalized([initial,middle,final]),normalized([initial,final]))
        for changes in ({'completed':0},{'active':['unknown']},{'total':3},
                        {'workers':1},{'total':2.},{'workers':2.},
                        {'wall_s':-1.},{'new_field':True}):
            wrong = deepcopy(middle);wrong.update(changes)
            with self.assertRaises(AssertionError):
                normalized([initial,wrong,final])
        with self.assertRaises(AssertionError):
            normalized([initial,middle])
        self.assertNotEqual(normalized([initial,final]),normalized([]))
        # With one worker, bool True would otherwise compare equal to 1.
        one = [dict(initial,workers=1,active=['a']),dict(middle,workers=True),dict(final,workers=1)]
        with self.assertRaises(AssertionError):
            normalized(one)


if __name__ == '__main__':
    unittest.main()
