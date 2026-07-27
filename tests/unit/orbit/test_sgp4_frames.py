import importlib.util
import math
import shutil
import tempfile
import unittest
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from unittest.mock import patch

from orbitserve.orbit import (
    ASTROPY_ENGINE_VERSION,
    PINNED_IERS_A_SHA256,
    PINNED_IERS_DATA_RELEASE,
    PINNED_IERS_README_SHA256,
    PINNED_LEAP_SECONDS_SHA256,
    SGP4_EARTH_FIXED_REFERENCE_FRAME,
    Sgp4FrameTransformError,
    Sgp4PropagationResult,
    Sgp4TemeSample,
    propagate_sgp4_teme,
    transform_sgp4_teme_to_itrs,
)
from orbitserve.orbit.sgp4 import frames as frame_module

from .test_sgp4_propagation import _has_tested_sgp4, _iss_bundle
from .test_sgp4_elements import _bundle


def _has_tested_astropy():
    if importlib.util.find_spec("astropy") is None:
        return False
    try:
        return version("astropy") == ASTROPY_ENGINE_VERSION
    except PackageNotFoundError:
        return False


def _propagation(epoch="2019-12-09T20:42:00.000000Z"):
    return Sgp4PropagationResult(
        element_set_id="sgp4-elements-v1-frame-test",
        simulation_epoch_utc=epoch,
        samples=(
            Sgp4TemeSample(
                time_s=0.0,
                satellite_id="iss",
                position_teme_km=(-6088.92, -936.13, -2866.44),
                velocity_teme_km_s=(-1.525, -5.538, 5.068),
            ),
        ),
    )


@unittest.skipUnless(_has_tested_astropy(), "requires the frozen astropy 6.1.3 engine")
class Sgp4FrameTransformTests(unittest.TestCase):
    @unittest.skipUnless(_has_tested_sgp4(), "requires the frozen python-sgp4 2.27 engine")
    def test_public_sgp4_to_itrs_pipeline_composes_deterministically(self):
        propagation = propagate_sgp4_teme(
            _iss_bundle(),
            simulation_epoch_utc="2019-12-09T20:42:00.000000Z",
            sample_times_s=(60.0, 0.0),
        )
        first = transform_sgp4_teme_to_itrs(propagation)
        second = transform_sgp4_teme_to_itrs(propagation)

        self.assertEqual(first, second)
        self.assertEqual([sample.time_s for sample in first.samples], [0.0, 60.0])
        self.assertTrue(
            all(math.isfinite(value) for sample in first.samples for value in sample.position_itrs_km)
        )

    def test_teme_to_itrs_matches_frozen_reference_and_records_data(self):
        import erfa
        import numpy as np
        from astropy.time import core as time_core

        saved_leap_seconds = erfa.leap_seconds.get().copy()
        saved_leap_check = time_core._LEAP_SECONDS_CHECK
        result = transform_sgp4_teme_to_itrs(_propagation())
        sample = result.samples[0]

        self.assertEqual(result.reference_frame, SGP4_EARTH_FIXED_REFERENCE_FRAME)
        self.assertFalse(result.network_access)
        self.assertEqual(result.iers_data_release, PINNED_IERS_DATA_RELEASE)
        self.assertEqual(result.iers_a_sha256, PINNED_IERS_A_SHA256)
        self.assertEqual(result.iers_readme_sha256, PINNED_IERS_README_SHA256)
        self.assertEqual(result.leap_seconds_sha256, PINNED_LEAP_SECONDS_SHA256)
        self.assertIn(result.earth_orientation_quality, {"definitive", "predictive"})
        self.assertEqual(sample.satellite_id, "iss")
        expected_position = (-5786.6977008, 2113.1608607, -2866.43425063)
        expected_velocity = (-3.85031878, -3.69625304, 5.06799712)
        for actual, expected in zip(sample.position_itrs_km, expected_position):
            self.assertAlmostEqual(actual, expected, delta=1e-7)
        for actual, expected in zip(sample.velocity_itrs_km_s, expected_velocity):
            self.assertAlmostEqual(actual, expected, delta=1e-8)
        self.assertTrue(np.array_equal(erfa.leap_seconds.get(), saved_leap_seconds))
        self.assertIs(time_core._LEAP_SECONDS_CHECK, saved_leap_check)

    def test_identical_data_at_different_paths_produces_identical_result(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_iers = Path(tmpdir) / "first-finals.all"
            second_iers = Path(tmpdir) / "second-finals.all"
            first_leap = Path(tmpdir) / "first-leap.dat"
            second_leap = Path(tmpdir) / "second-leap.dat"
            shutil.copyfile(frame_module._PINNED_IERS_A_PATH, first_iers)
            shutil.copyfile(frame_module._PINNED_IERS_A_PATH, second_iers)
            shutil.copyfile(frame_module._PINNED_LEAP_SECONDS_PATH, first_leap)
            shutil.copyfile(frame_module._PINNED_LEAP_SECONDS_PATH, second_leap)

            first = transform_sgp4_teme_to_itrs(
                _propagation(), iers_a_path=first_iers, leap_seconds_path=first_leap
            )
            second = transform_sgp4_teme_to_itrs(
                _propagation(), iers_a_path=second_iers, leap_seconds_path=second_leap
            )
        self.assertEqual(first, second)
        serialized = str(first.to_dict())
        self.assertNotIn(str(first_iers), serialized)
        self.assertNotIn(str(second_iers), serialized)

    @unittest.skipUnless(_has_tested_sgp4(), "requires the frozen python-sgp4 2.27 engine")
    def test_default_pinned_data_covers_full_project_2026_pipeline(self):
        propagation = propagate_sgp4_teme(
            _bundle(),
            simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
            sample_times_s=(0.0, 60.0),
        )
        result = transform_sgp4_teme_to_itrs(propagation)
        self.assertEqual(result.iers_data_release, PINNED_IERS_DATA_RELEASE)
        self.assertEqual(result.earth_orientation_quality, "predictive")
        self.assertEqual(len(result.samples), 4)
        self.assertEqual([sample.time_s for sample in result.samples], [0.0, 0.0, 60.0, 60.0])

    def test_default_data_hash_mismatch_fails_before_parsing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tampered = Path(tmpdir) / "tampered-finals.all"
            shutil.copyfile(frame_module._PINNED_IERS_A_PATH, tampered)
            tampered.write_bytes(tampered.read_bytes() + b" ")
            with patch.object(frame_module, "_PINNED_IERS_A_PATH", tampered):
                with self.assertRaisesRegex(Sgp4FrameTransformError, "SHA-256 mismatch"):
                    transform_sgp4_teme_to_itrs(_propagation())

    def test_far_future_sample_fails_closed_with_time(self):
        with self.assertRaises(Sgp4FrameTransformError) as caught:
            transform_sgp4_teme_to_itrs(_propagation("2200-07-20T00:00:00.000000Z"))
        self.assertEqual(caught.exception.time_s, 0.0)
        self.assertRegex(str(caught.exception), "outside the pinned")

    def test_rejects_noncanonical_or_nonfinite_propagation_samples(self):
        duplicate = Sgp4PropagationResult(
            element_set_id="duplicate",
            simulation_epoch_utc="2019-12-09T20:42:00.000000Z",
            samples=(_propagation().samples[0], _propagation().samples[0]),
        )
        with self.assertRaisesRegex(ValueError, "unique and ordered"):
            transform_sgp4_teme_to_itrs(duplicate)

        nonfinite = Sgp4PropagationResult(
            element_set_id="nonfinite",
            simulation_epoch_utc="2019-12-09T20:42:00.000000Z",
            samples=(
                Sgp4TemeSample(
                    time_s=0.0,
                    satellite_id="bad",
                    position_teme_km=(math.nan, 0.0, 0.0),
                    velocity_teme_km_s=(0.0, 0.0, 0.0),
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "three finite numbers"):
            transform_sgp4_teme_to_itrs(nonfinite)


if __name__ == "__main__":
    unittest.main()
