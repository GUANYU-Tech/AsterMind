"""Observe real backend capacity updates against independent per-time RF budgets."""
import importlib.util
import unittest
from dataclasses import replace
from unittest.mock import patch

from tests.network_case import run_traffic
from orbitserve.runtime.engine.fluid_transport import FluidTransport
from orbitserve.runtime.engine.simgrid_transport import SimGridTransport
from orbitserve.runtime.engine.ns3_transport import Ns3Transport
from tests.integration.fixed_design.test_network_acceptance import acceptance_spec
from tests.unit.runtime.test_network_time_consistency import temporal_state, expected_rate


class TimeVaryingTransportTests(unittest.TestCase):
    def verify(self, mode, backend):
        base = acceptance_spec()
        physical = temporal_state()
        scenario = replace(base, network=replace(physical.spec, transport_model=mode),
                           traffic=({**base.traffic[0], 'arrival_time_s': 1.25, 'input_bytes': 16384},))
        calls = []
        original = backend.configure

        def observe(engine, capacities):
            for key, rate in capacities.items():
                direction = next((k for k, v in engine.resources.items() if v == key), None) if hasattr(engine, 'resources') else key
                calls.append((engine.now, direction, rate))
            return original(engine, capacities)

        with patch.object(backend, 'configure', observe):
            result = run_traffic(scenario)
        self.assertEqual(result.completed, 1)
        leg = result.transfer_jobs[0]['path'][0]
        self.assertGreater(leg['finish_s'], 5.75)
        direction = (leg['link_id'], 'ground-00', 'sat-a')
        updates = [(t, v) for t, k, v in calls if k == direction]
        self.assertTrue({2., 2.25, 3., 4., 5., 5.75} <= {t for t, _ in updates}, updates)
        for t, actual in updates:
            distance = physical.distance(leg, t)
            expected = expected_rate(physical.spec.link_profiles['sgl'], distance, t)
            self.assertAlmostEqual(actual / expected, 1., places=10)
        self.assertAlmostEqual(sum(s['bytes'] for s in leg['segments']), 16384, places=5)
        self.assertAlmostEqual(leg['distance_km'] / leg['propagation_delay_s'], 299792.458, places=7)
        self.assertEqual(leg['distance_time_s'], leg['finish_s'])

    def test_fluid_updates_on_geometry_and_exact_schedule_boundaries(self): self.verify('fluid', FluidTransport)

    @unittest.skipUnless(importlib.util.find_spec('simgrid'), 'requires native SimGrid')
    def test_simgrid_updates_on_geometry_and_exact_schedule_boundaries(self): self.verify('simgrid', SimGridTransport)

    @unittest.skipUnless(importlib.util.find_spec('ns'), 'requires native ns-3')
    def test_ns3_updates_on_geometry_and_exact_schedule_boundaries(self): self.verify('ns3', Ns3Transport)
