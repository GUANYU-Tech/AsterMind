"""Continuous random processes are reproducible independently of time slices."""

import unittest
from dataclasses import replace

from orbitserve.runtime.setup.faults import realize_runtime_faults, validate_random_faults
from tests.unit.runtime.test_dynamic_network import network


class ContinuousFaultModelTests(unittest.TestCase):
    def setUp(self):
        self.net = network()
        self.settings = dict(enabled=True, target_type='node', seed=43, time_basis='seconds',
                             model='exponential', start_s=.125, end_s=180., mean_up_s=20.,
                             duration=dict(distribution='fixed', seconds=7.25))

    def realize(self, settings, boundaries):
        spec = replace(self.net.spec, random_faults=settings, fault_timing='continuous')
        return realize_runtime_faults(spec, boundaries, self.net.edges).faults

    def test_no_rounding_and_invariance_to_physical_cadence(self):
        a = self.realize(self.settings, (0., 60., 120., 180., 240.))
        b = self.realize(self.settings, tuple(float(i) for i in range(241)))
        self.assertTrue(a)
        self.assertEqual(a, b)
        self.assertTrue(any(f['start_s'] % 60 != 0 for f in a))
        self.assertTrue(all(abs(f['end_s']-f['start_s']-7.25)<1e-10 for f in a))

    def test_weibull_and_duration_distributions_are_continuous(self):
        for duration in (dict(distribution='uniform', min_s=.5, max_s=1.5),
                         dict(distribution='exponential', mean_s=2.),
                         dict(distribution='lognormal', mean_s=2., sigma=.5)):
            config = dict(self.settings, model='weibull', up_scale_s=20., up_shape=1.5, duration=duration)
            faults = self.realize(config, (0., 60., 120., 180., 240.))
            self.assertTrue(faults)
            self.assertTrue(all(f['end_s']>f['start_s'] for f in faults))
            self.assertEqual(faults, self.realize(config, (0., 60., 120., 180., 240.)))

    def test_invalid_seconds_inputs_are_rejected(self):
        for change in ({'mean_up_s':0}, {'start_step':0}, {'end_s':False},
                       {'duration':dict(distribution='fixed', seconds=-1)}, {'duration':[]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_random_faults(dict(self.settings, **change))
