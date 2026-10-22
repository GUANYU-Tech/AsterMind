import math
import unittest

from orbitserve.orbit import walker_delta_elements, tle_elements, propagate_sgp4_teme, dawn_dusk_elements
from orbitserve.orbit.sgp4.propagation import _initialize_satellite, _load_sgp4_api


class Sgp4InitializationTests(unittest.TestCase):
    def test_dawn_dusk_has_retrograde_sso_inclination_and_distinct_coplanar_members(self):
        params = dict(name='dawn', altitude_km=550., satellites_per_plane=4,
                      planes=2, epoch_utc='2026-03-20T00:00:00Z')
        dawn = dawn_dusk_elements(**params, ltan_hours=6.)
        dusk = dawn_dusk_elements(**params, ltan_hours=18.)
        self.assertTrue(97 < dawn.satellites[0].inclination_deg < 99)
        self.assertEqual(len({(s.raan_deg,s.mean_anomaly_deg) for s in dawn.satellites}), 8)
        self.assertAlmostEqual((dusk.satellites[0].raan_deg-dawn.satellites[0].raan_deg)%360,180.)
        self.assertFalse(dawn.provenance.observationally_fitted)
        later = dawn_dusk_elements(**{**params, 'epoch_utc':'2026-06-20T00:00:00Z'})
        change = (later.satellites[0].raan_deg-dawn.satellites[0].raan_deg)%360
        self.assertTrue(89 < change < 93)
        for change in ({'ltan_hours':12}, {'plane_spread_deg':90}, {'altitude_km':40000}):
            with self.assertRaises(ValueError):
                dawn_dusk_elements(**{**params, **change})

    def test_72_by_22_walker_has_1584_unique_independent_mean_elements(self):
        bundle = walker_delta_elements(
            name="starlink",
            altitude_km=550.0,
            inclination_deg=53.0,
            planes=72,
            satellites_per_plane=22,
            epoch_utc="2026-07-20T00:00:00Z",
            walker_phase=1,
        )
        self.assertEqual(len(bundle.satellites), 1584)
        self.assertEqual(len({s.satellite_id for s in bundle.satellites}), 1584)
        self.assertEqual(len({s.raan_deg for s in bundle.satellites}), 72)
        self.assertFalse(bundle.provenance.observationally_fitted)
        first = bundle.satellites[0]
        n = first.mean_motion_rev_per_day * 2 * math.pi / 86400
        self.assertAlmostEqual((398600.8 / n**2) ** (1 / 3) - 6378.135, 550.0)

    def test_tle_adapter_preserves_engine_predictions_and_rejects_bad_checksum(self):
        from sgp4.exporter import export_tle

        bundle = walker_delta_elements(
            name="sample",
            altitude_km=550.0,
            inclination_deg=53.0,
            planes=1,
            satellites_per_plane=1,
            epoch_utc="2026-07-20T00:00:00Z",
        )
        sat = _initialize_satellite(_load_sgp4_api(), bundle.satellites[0])
        first, second = export_tle(sat)
        observed = tle_elements(first + "\n" + second, name="observed")
        states = propagate_sgp4_teme(
            observed,
            simulation_epoch_utc=observed.satellites[0].epoch_utc,
            sample_times_s=(0.0, 60.0),
        )
        from sgp4.api import Satrec, WGS72

        reference = Satrec.twoline2rv(first, second, WGS72)
        for state in states.samples:
            code, position, _ = reference.sgp4(
                reference.jdsatepoch, reference.jdsatepochF + state.time_s / 86400
            )
            self.assertEqual(code, 0)
            self.assertLess(math.dist(position, state.position_teme_km), 1e-6)
        with self.assertRaisesRegex(ValueError, "checksum"):
            tle_elements(
                first[:-1] + str((int(first[-1]) + 1) % 10) + "\n" + second,
                name="invalid",
            )

    def test_invalid_walker_grid_is_rejected(self):
        for planes, slots, phase in ((0, 22, 0), (72, 0, 0), (72, 22, 72)):
            with self.assertRaises(ValueError):
                walker_delta_elements(
                    name="bad",
                    altitude_km=550,
                    inclination_deg=53,
                    planes=planes,
                    satellites_per_plane=slots,
                    walker_phase=phase,
                    epoch_utc="2026-07-20T00:00:00Z",
                )
