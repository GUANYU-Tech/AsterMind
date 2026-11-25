import unittest

from astermind.builder.physical.tle_inspection import inspect_tle_catalog, inspect_tle_constellation
from astermind.builder.orbit import walker_delta_elements
from astermind.builder.orbit.sgp4.propagation import _initialize_satellite, _load_sgp4_api


def observed_text(planes=4,slots=8,missing=()):
    from sgp4.exporter import export_tle
    bundle=walker_delta_elements(name='sample',altitude_km=570,inclination_deg=70,
        planes=planes,satellites_per_plane=slots,epoch_utc='2026-10-20T00:00:00Z')
    return '\n'.join('\n'.join(export_tle(_initialize_satellite(_load_sgp4_api(),s)))
        for i,s in enumerate(bundle.satellites) if i not in missing)+'\n'


class TleInspectionTests(unittest.TestCase):
    def test_sparse_observations_infer_grid_without_compressing_missing_positions(self):
        result=inspect_tle_constellation(observed_text(missing=(1,2,8,9,10,11,12,13,14,15)))
        self.assertEqual(result['satellite_count'],22)
        self.assertEqual(result['observed_plane_groups'],3)
        self.assertEqual(result['inferred_grid_shape'],[4,8])
        self.assertEqual(result['grid_placed_count'],22)
        self.assertEqual(result['nominal_grid_empty_positions'],10)
        self.assertIn('not_authoritative',result['grid_inference'])

    def test_checksum_error_is_not_accepted_as_an_import(self):
        text=observed_text()
        first=text.splitlines()[0]
        damaged=first[:-1]+str((int(first[-1])+1)%10)
        with self.assertRaisesRegex(ValueError,'checksum'):
            inspect_tle_constellation(text.replace(first,damaged,1))

    def test_catalog_over_ten_thousand_preserves_every_observation(self):
        text = observed_text(planes=96, slots=108)
        result = inspect_tle_catalog(text)
        self.assertEqual(result['satellite_count'], 10368)
        self.assertEqual(result['inferred_grid_shape'], [96, 108])
        self.assertEqual(result['grid_placed_count'], 10368)
        self.assertEqual(result['suggested_source']['tle_text'], text)

    def test_catalog_groups_by_orbit_and_retains_sparse_records_and_names(self):
        lines = observed_text().splitlines()
        # Move one observation to another inclination and repair its TLE checksum.
        changed = lines[1][:8] + ' 30.0000' + lines[1][16:68]
        lines[1] = changed + str(sum(int(c) if c.isdigit() else c=='-' for c in changed) % 10)
        text = 'SAT-A\n'+'\n'.join(lines)
        result = inspect_tle_catalog(text)
        self.assertEqual(result['satellite_count'], 32)
        sparse, regular = result['shells']
        self.assertEqual(sparse['satellite_count'], 1)
        self.assertIsNone(sparse['inferred_grid_shape'])
        self.assertIn('SAT-A\n', sparse['suggested_source']['tle_text'])
        self.assertEqual(regular['satellite_count'], 31)
        self.assertEqual(regular['grid_placed_count'], 31)

    def test_phase_matches_circular_distance_minimum(self):
        from astermind.builder.physical.tle_layout import _phase
        for angles in ([359.9, 0.1, 0, 10], [0, 90, 180, 270], [1, 15, 16, 93, 170, 220, 314]):
            cost = lambda p: sum(min((v-p)%360, (p-v)%360) for v in angles)
            self.assertAlmostEqual(cost(_phase(angles, 360)), min(map(cost, angles)))
