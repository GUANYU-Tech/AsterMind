import hashlib
import importlib.util
import math
import unittest
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version

from astermind.builder.orbit import ASTROPY_ENGINE_VERSION, SGP4_ENGINE_VERSION, SGP4_GEOMETRY_SCHEMA, WGS84_SEMI_MAJOR_AXIS_KM, Sgp4EarthFixedResult, Sgp4GroundIngress, Sgp4ItrsSample, Sgp4SunPositionResult, Sgp4SunPositionSample, compute_sgp4_sun_positions_itrs, evaluate_sgp4_geometry, propagate_sgp4_teme, transform_sgp4_teme_to_itrs

from tests.unit.orbit.test_sgp4_elements import _bundle


def _has_version(package, expected):
    if importlib.util.find_spec(package.replace("-", "_")) is None:
        return False
    try:
        return version(package) == expected
    except PackageNotFoundError:
        return False


def _earth_fixed():
    radius = WGS84_SEMI_MAJOR_AXIS_KM + 500.0
    samples = []
    for time_s in (0.0, 60.0):
        samples.extend(
            (
                Sgp4ItrsSample(
                    time_s=time_s,
                    satellite_id="sat-a",
                    position_itrs_km=(radius, 0.0, 0.0),
                    velocity_itrs_km_s=(0.0, 7.5, 0.0),
                ),
                Sgp4ItrsSample(
                    time_s=time_s,
                    satellite_id="sat-b",
                    position_itrs_km=(-radius, 0.0, 0.0),
                    velocity_itrs_km_s=(0.0, -7.5, 0.0),
                ),
            )
        )
    return Sgp4EarthFixedResult(
        element_set_id="sgp4-elements-v1-geometry-test",
        simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
        samples=tuple(samples),
        iers_a_sha256="iers-test",
        iers_readme_sha256="readme-test",
        leap_seconds_sha256="leap-test",
        earth_orientation_quality="predictive",
        iers_data_release="test-release",
        astropy_iers_data_version="test-version",
    )


def _sun_positions(earth_fixed=None):
    earth_fixed = earth_fixed or _earth_fixed()
    return Sgp4SunPositionResult(
        element_set_id=earth_fixed.element_set_id,
        simulation_epoch_utc=earth_fixed.simulation_epoch_utc,
        samples=tuple(
            Sgp4SunPositionSample(
                time_s=time_s,
                position_itrs_km=(149_600_000.0, 0.0, 0.0),
            )
            for time_s in (0.0, 60.0)
        ),
        iers_a_sha256=earth_fixed.iers_a_sha256,
        iers_readme_sha256=earth_fixed.iers_readme_sha256,
        leap_seconds_sha256=earth_fixed.leap_seconds_sha256,
        earth_orientation_quality=earth_fixed.earth_orientation_quality,
        iers_data_release=earth_fixed.iers_data_release,
        astropy_iers_data_version=earth_fixed.astropy_iers_data_version,
    )


class Sgp4GeometryTests(unittest.TestCase):
    def test_wgs84_sgl_isl_and_conical_shadow_facts(self):
        result = evaluate_sgp4_geometry(
            _earth_fixed(),
            ground_ingresses=(
                {
                    "ingress_id": "ground-0",
                    "lat_deg": 0.0,
                    "lon_deg": 0.0,
                    "min_elevation_deg": 10.0,
                },
            ),
            isl_pairs=(("sat-b", "sat-a"),),
            sun_positions=_sun_positions(),
        )

        self.assertEqual(result.schema_version, SGP4_GEOMETRY_SCHEMA)
        self.assertEqual(result.isl_pairs, (("sat-a", "sat-b"),))
        sgl = {(row.time_s, row.satellite_id): row for row in result.sgl_samples}
        self.assertTrue(sgl[(0.0, "sat-a")].visible)
        self.assertAlmostEqual(sgl[(0.0, "sat-a")].elevation_deg, 90.0)
        self.assertFalse(sgl[(0.0, "sat-b")].visible)
        self.assertEqual(
            [row.to_dict() for row in result.sgl_windows],
            [
                {
                    "ingress_id": "ground-0",
                    "satellite_id": "sat-a",
                    "start_s": 0.0,
                    "end_s": 60.0,
                }
            ],
        )

        self.assertTrue(all(not row.visible for row in result.isl_samples))
        self.assertEqual(result.isl_windows, ())
        sunlight = {(row.time_s, row.satellite_id): row for row in result.sunlight_samples}
        self.assertTrue(sunlight[(0.0, "sat-a")].sunlit)
        self.assertFalse(sunlight[(0.0, "sat-b")].sunlit)
        self.assertEqual(
            [(row.satellite_id, row.sunlit, row.start_s, row.end_s) for row in result.sunlight_windows],
            [("sat-a", True, 0.0, 60.0), ("sat-b", False, 0.0, 60.0)],
        )

    def test_input_order_does_not_change_canonical_bytes(self):
        earth_fixed = _earth_fixed()
        sun = _sun_positions(earth_fixed)
        first = evaluate_sgp4_geometry(
            earth_fixed,
            ground_ingresses=(
                Sgp4GroundIngress("z-ground", 20.0, 30.0, min_elevation_deg=5.0),
                Sgp4GroundIngress("a-ground", 0.0, 0.0),
            ),
            isl_pairs=(("sat-b", "sat-a"),),
            sun_positions=sun,
        )
        second = evaluate_sgp4_geometry(
            earth_fixed,
            ground_ingresses=(
                Sgp4GroundIngress("a-ground", -0.0, -0.0),
                Sgp4GroundIngress("z-ground", 20.0, 30.0, min_elevation_deg=5.0),
            ),
            isl_pairs=(("sat-a", "sat-b"),),
            sun_positions=sun,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertTrue(first.canonical_bytes().endswith(b"\n"))
        self.assertNotIn(b"NaN", first.canonical_bytes())
        self.assertNotIn(b"-0.0", first.canonical_bytes())
        self.assertEqual(
            hashlib.sha256(first.canonical_bytes()).hexdigest(),
            "cc05767e932652dc9f9394b1332745ad83d4108af438401d9afe983f7bc42ac3",
        )

    def test_same_side_isl_is_visible_and_margin_can_block_it(self):
        earth_fixed = _earth_fixed()
        radius = WGS84_SEMI_MAJOR_AXIS_KM + 500.0
        changed = tuple(
            replace(
                sample,
                position_itrs_km=(radius, 1000.0, 0.0),
            )
            if sample.satellite_id == "sat-b"
            else sample
            for sample in earth_fixed.samples
        )
        earth_fixed = replace(earth_fixed, samples=changed)
        result = evaluate_sgp4_geometry(
            earth_fixed,
            isl_pairs=(("sat-a", "sat-b"),),
            sun_positions=_sun_positions(earth_fixed),
        )
        self.assertTrue(all(row.visible for row in result.isl_samples))
        self.assertEqual(len(result.isl_windows), 1)
        self.assertGreater(result.isl_samples[0].earth_clearance_km, 0.0)

    def test_invalid_or_misaligned_inputs_fail_closed(self):
        earth_fixed = _earth_fixed()
        with self.assertRaisesRegex(ValueError, "unique after canonical"):
            evaluate_sgp4_geometry(
                earth_fixed,
                isl_pairs=(("sat-a", "sat-b"), ("sat-b", "sat-a")),
                sun_positions=_sun_positions(earth_fixed),
            )
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            evaluate_sgp4_geometry(
                earth_fixed,
                ground_ingresses=(
                    {"ingress_id": "x", "lat_deg": 0.0, "lon_deg": 0.0, "bandwidth_bps": 1},
                ),
                sun_positions=_sun_positions(earth_fixed),
            )
        wrong_sun = replace(_sun_positions(earth_fixed), element_set_id="wrong")
        with self.assertRaisesRegex(ValueError, "element_set_id"):
            evaluate_sgp4_geometry(earth_fixed, sun_positions=wrong_sun)
        incomplete = replace(
            earth_fixed,
            samples=tuple(sample for sample in earth_fixed.samples if not (sample.time_s == 60.0 and sample.satellite_id == "sat-b")),
        )
        with self.assertRaisesRegex(ValueError, "rectangular"):
            evaluate_sgp4_geometry(incomplete, sun_positions=_sun_positions(incomplete))

    @unittest.skipUnless(
        _has_version("sgp4", SGP4_ENGINE_VERSION)
        and _has_version("astropy", ASTROPY_ENGINE_VERSION),
        "requires the frozen SGP4 and Astropy engines",
    )
    def test_public_sgp4_to_geometry_pipeline_is_deterministic(self):
        propagation = propagate_sgp4_teme(
            _bundle(),
            simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
            sample_times_s=(0.0, 60.0),
        )
        earth_fixed = transform_sgp4_teme_to_itrs(propagation)
        sun = compute_sgp4_sun_positions_itrs(earth_fixed)
        kwargs = {
            "ground_ingresses": (
                {"ingress_id": "equator", "lat_deg": 0.0, "lon_deg": 0.0},
            ),
            "isl_pairs": (("sat-00002", "sat-00001"),),
            "sun_positions": sun,
        }
        first = evaluate_sgp4_geometry(earth_fixed, **kwargs)
        second = evaluate_sgp4_geometry(earth_fixed, **kwargs)

        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertEqual(len(first.sgl_samples), 4)
        self.assertEqual(len(first.isl_samples), 2)
        self.assertEqual(len(first.sunlight_samples), 4)
        self.assertTrue(
            all(math.isfinite(value) for row in first.sunlight_samples for value in (row.shadow_margin_km, row.sun_distance_km))
        )


if __name__ == "__main__":
    unittest.main()
