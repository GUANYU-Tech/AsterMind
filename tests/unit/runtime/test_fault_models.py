"""Probability models, independent scopes and uncensored repair samples."""

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

from astermind.engine.setup.faults import realize_runtime_faults, fault_step_details, validate_random_faults
from tests.unit.runtime.test_dynamic_network import network


class FaultModelTests(unittest.TestCase):
    def setUp(self):
        net = network()
        self.spec, self.edges = replace(net.spec, fault_timing='boundary'), net.edges
        self.boundaries = tuple(float(i) for i in range(101))
        self.common = dict(enabled=True, seed=42, start_step=0, end_step=90)

    def realize(self, settings, boundaries=None):
        return realize_runtime_faults(replace(self.spec, random_faults=settings),
                                      boundaries or self.boundaries, self.edges).faults

    def test_node_and_link_processes_are_independent_and_legacy_input_is_preserved(self):
        nodes = dict(self.common, model='per_step', failure_probability=.2, duration_steps=2)
        links = dict(self.common, model='exponential', seed=71, mean_up_steps=6, mean_down_steps=2)
        both = self.realize(dict(nodes=nodes, links=links))
        self.assertTrue(any(f['target_type'] == 'node' for f in both))
        self.assertTrue(any(f['target_type'] == 'link' for f in both))
        for kind, config in (('node', nodes), ('link', links)):
            self.assertEqual(tuple(f for f in both if f['target_type'] == kind),
                             self.realize(dict(config, target_type=kind)))
        self.assertEqual(tuple(f for f in both if f['target_type'] == 'link'),
                         self.realize(dict(nodes=dict(enabled=False), links=links)))

    def test_all_models_reproduce_and_have_nonoverlapping_intervals(self):
        for model, parameters in (
            ('fixed_ratio', dict(ratio=.5)),
            ('per_step', dict(failure_probability=.3, duration=dict(distribution='uniform', min_steps=2, max_steps=5))),
            ('exponential', dict(mean_up_steps=6, mean_down_steps=2)),
            ('weibull', dict(up_scale_steps=6, up_shape=2, duration=dict(distribution='lognormal', mean_steps=3, sigma=.6))),
        ):
            with self.subTest(model=model):
                settings=dict(self.common, target_type='node', model=model, **parameters)
                faults=self.realize(settings)
                self.assertTrue(faults)
                self.assertEqual(faults, self.realize(settings))
                previous={}
                for fault in sorted(faults, key=lambda f:f['start_s']):
                    self.assertGreater(fault['end_s'], fault['start_s'])
                    self.assertEqual(fault['end_s'] % 1, 0)
                    self.assertGreater(fault['start_s'], previous.get(fault['target_id'], -1))
                    previous[fault['target_id']]=fault['end_s']

    def test_uniform_duration_has_inclusive_bounds_and_varies(self):
        faults=self.realize(dict(self.common, target_type='link', model='per_step', failure_probability=1,
                                 duration=dict(distribution='uniform', min_steps=2, max_steps=5)))
        durations={f['end_s']-f['start_s'] for f in faults}
        self.assertEqual(durations, {2., 3., 4., 5.})

    def test_exponential_samples_match_rounded_distribution_mean(self):
        boundaries=tuple(float(i) for i in range(20001))
        faults=self.realize(dict(self.common, target_type='node', model='exponential', end_step=20000,
                                 mean_up_steps=5, mean_down_steps=3), boundaries)
        durations=[f['end_s']-f['start_s'] for f in faults]
        expected=1 / -math.expm1(-1/3)
        self.assertGreater(len(durations), 1000)
        self.assertAlmostEqual(sum(durations)/len(durations), expected, delta=.15)

    def test_out_of_horizon_duration_is_preserved_and_marked_censored(self):
        settings=dict(self.common, target_type='node', model='per_step', end_step=1, failure_probability=1,
                      duration=dict(distribution='exponential', mean_steps=3))
        with patch('astermind.engine.setup.faults.random.Random.expovariate', return_value=8.25):
            faults=self.realize(settings, (0.,1.,2.,3.,4.,5.))
        self.assertTrue(faults)
        for fault in faults:
            details=fault_step_details(fault, (0.,1.,2.,3.,4.,5.))
            self.assertEqual(details['end_step'], 9)
            self.assertEqual(details['duration_steps'], 9)
            self.assertTrue(details['right_censored'])

    def test_invalid_distribution_parameters_and_mixed_scopes_are_rejected(self):
        base=dict(self.common, target_type='node', model='weibull', up_scale_steps=5, up_shape=2,
                  duration=dict(distribution='fixed', steps=2))
        for change in ({'up_shape':0}, {'up_scale_steps':math.inf},
                       {'duration':dict(distribution='uniform',min_steps=5,max_steps=2)},
                       {'duration':dict(distribution='lognormal',mean_steps=2,sigma=0)},
                       {'duration':dict(distribution='exponential',mean_steps=-1)}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_random_faults({**base, **change})
        for settings in ({'nodes':base,'enabled':True}, {'nodes':dict(base,target_type='link')}, {'links':[]}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_random_faults(settings)

    def test_network_spec_owns_nested_fault_settings(self):
        settings=dict(nodes=dict(self.common, model='per_step', failure_probability=.2,
                                 duration=dict(distribution='fixed',steps=2)))
        spec=replace(self.spec, random_faults=settings)
        settings['nodes']['duration']['steps']=99
        self.assertEqual(spec.random_faults['nodes']['duration']['steps'], 2)

    def test_simulation_end_is_not_treated_as_an_observed_repair(self):
        net=network()
        fault=dict(target_type='link',target_id=net.edges[0]['link_id'],start_s=0.,end_s=40.,source='random')
        net.observe_fault(0,fault,0.)
        self.assertTrue(net.faulted(net.edges[0],net.end_s))
        self.assertEqual(net._geometry_attributes(net.edges[0],net.end_s)['reason'],'injected_fault')
