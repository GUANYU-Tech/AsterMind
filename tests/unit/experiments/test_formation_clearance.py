"""Long-horizon multi-row formations stay separated under independent SGP4."""

import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from astermind.builder.physical.formation_validation import validate_formation_input
from astermind.builder.physical.network import run_network_experiment
from astermind.builder.physical.network_inputs import materialize_network_input

ROOT = Path(__file__).resolve().parents[3]


def formation_input(*, phase=0., inclination=53., shape='5x2', count=10, duration=3600., step=60.):
    value = json.loads((ROOT/'tests/fixtures/network/multi_shell.json').read_text())
    value['constellations']['communication_shells'] = []
    value['constellations']['compute_formations'] = value['constellations']['compute_formations'][:1]
    value['scenario_type'] = 'compute_only'
    value['time'].update(duration_s=duration, step_s=step)
    formation = value['constellations']['compute_formations'][0]
    formation['source'].update(type='walker_delta', altitude_km=800., inclination_deg=inclination,
                               planes=1, clusters_per_plane=2, walker_phase=0, raan0_deg=0., u0_deg=phase)
    formation['cluster'].update(satellites_per_cluster=count, grid_shape=shape, formation_pitch_m=500.,
                               d_keepout_m=50., sigma_control_m=0., sigma_navigation_m=0.)
    value['route_probes'] = []
    return value


def legacy_crossing_input():
    """Reconstruct v1's inclination-only offsets for the reported 5x2 input."""
    value = materialize_network_input(formation_input())
    for row in value['elements']['satellites']:
        member = int(row['satellite_id'].rsplit('-m', 1)[1])
        across = ((member // 5) - .5) * .5
        n = row['mean_motion_rev_per_day'] * 2 * math.pi / 86400
        radius = (398600.8 / n**2) ** (1/3)
        row['mean_anomaly_deg'] = (row['mean_anomaly_deg'] + math.degrees(2*across/radius)
                                   + row['argument_of_perigee_deg']) % 360
        row.update(eccentricity=0., argument_of_perigee_deg=0.)
    return value


class FormationClearanceTests(unittest.TestCase):
    def test_reported_one_hour_multirow_configuration_has_safe_clearance(self):
        concrete = materialize_network_input(formation_input())
        checks = validate_formation_input(concrete, tuple(float(t) for t in range(0, 3601, 60)))
        self.assertEqual(len(checks), 2)
        for check in checks.values():
            self.assertGreater(check['min_separation_m'], 450.)
            self.assertEqual(check['required_m'], 50.)
            self.assertEqual(check['sample_count'], 61)
        provenance = concrete['initialization_provenance'][0]
        self.assertEqual(provenance['formation_motion'], 'free_sgp4_ei_separated_mean_element_offsets_v2')

    def test_dense_two_hour_checks_cover_different_phases_and_inclinations(self):
        for phase, inclination in ((0., 53.), (45., 53.), (90., 97.), (180., 90.)):
            with self.subTest(phase=phase, inclination=inclination):
                concrete = materialize_network_input(formation_input(phase=phase, inclination=inclination))
                checks = validate_formation_input(concrete, tuple(float(t) for t in range(0, 7201, 5)))
                self.assertTrue(all(check['min_separation_m'] > 450. for check in checks.values()))

    def test_three_rows_stay_separated_and_along_only_members_remain_circular(self):
        for shape, count in (('3x3', 9), ('4x1', 4)):
            with self.subTest(shape=shape):
                concrete = materialize_network_input(formation_input(shape=shape, count=count))
                checks = validate_formation_input(concrete, tuple(float(t) for t in range(0, 7201, 10)))
                self.assertTrue(all(check['min_separation_m'] > 450. for check in checks.values()))
                if shape == '4x1':
                    self.assertTrue(all(row['eccentricity'] == 0. for row in concrete['elements']['satellites']))

    def test_legacy_crossing_has_actionable_cell_pair_and_time_diagnostics(self):
        with self.assertRaisesRegex(ValueError, r'compute-mid-p000-s0000.*t=.+between.+-m'):
            validate_formation_input(legacy_crossing_input(), tuple(float(t) for t in range(0, 3601, 60)))

    def test_unsafe_formation_rejected_before_orbit_export_or_link_budgets(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch('astermind.builder.physical.network.generate_orbit_trace') as generate:
            with self.assertRaisesRegex(ValueError, 'formation keepout preflight failed'):
                run_network_experiment(legacy_crossing_input(), folder, _workers=1)
            generate.assert_not_called()
            self.assertFalse((Path(folder)/'physical_state.sqlite').exists())

    def test_nonformation_input_and_explicit_keepout_are_preserved(self):
        value = materialize_network_input(formation_input())
        original = deepcopy(value)
        validate_formation_input(value, (0., 60.))
        self.assertEqual(value, original)
        value['initialization_provenance'] = []
        self.assertEqual(validate_formation_input(value, (0., 60.)), {})
