import unittest

from orbitserve.experiments.tle_inspection import inspect_tle_constellation
from orbitserve.orbit import walker_delta_elements
from orbitserve.orbit.sgp4.propagation import _initialize_satellite, _load_sgp4_api


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

    def test_whole_catalog_is_rejected_before_orbit_propagation(self):
        with self.assertRaisesRegex(ValueError,'one shell'):
            inspect_tle_constellation('1 not-a-valid-tle\n'*2501)
