import math
import unittest
from dataclasses import replace

from astermind.builder.orbit import Sgp4EarthFixedResult, Sgp4ItrsSample, Sgp4PropagationResult, Sgp4TemeSample
from astermind.builder.orbit.validation import STK_11_BASELINE_BUILD, STK_11_BASELINE_VERSION, StkDirectStateCapture, StkStateSample, StkValidationError, compare_stk_state_capture


_ELEMENT_SET_ID = "sgp4-elements-v1-stk-metric-test"
_EPOCH = "2026-07-20T00:00:00.000000Z"


def _stk_sample(time_s, position=(7000.0, 0.0, 0.0), velocity=(0.0, 7.5, 0.0)):
    return StkStateSample(
        satellite_id="sat-00001",
        time_s=time_s,
        position_km=position,
        velocity_km_s=velocity,
    )


def _capture():
    reference = (_stk_sample(0.0), _stk_sample(60.0))
    return StkDirectStateCapture(
        element_set_id=_ELEMENT_SET_ID,
        simulation_epoch_utc=_EPOCH,
        sample_times_s=(0.0, 60.0),
        stk_version=STK_11_BASELINE_VERSION,
        installed_build=STK_11_BASELINE_BUILD,
        propagator_defaults_configured=True,
        teme_samples=reference,
        fixed_samples=reference,
        warnings=(),
    )


def _teme_result():
    return Sgp4PropagationResult(
        element_set_id=_ELEMENT_SET_ID,
        simulation_epoch_utc=_EPOCH,
        samples=(
            Sgp4TemeSample(
                time_s=0.0,
                satellite_id="sat-00001",
                position_teme_km=(7000.001, 0.002, 0.003),
                velocity_teme_km_s=(0.0001, 7.5002, 0.0003),
            ),
            Sgp4TemeSample(
                time_s=60.0,
                satellite_id="sat-00001",
                position_teme_km=(7000.004, 0.005, 0.006),
                velocity_teme_km_s=(0.0004, 7.5005, 0.0006),
            ),
        ),
    )


def _fixed_result():
    return Sgp4EarthFixedResult(
        element_set_id=_ELEMENT_SET_ID,
        simulation_epoch_utc=_EPOCH,
        samples=(
            Sgp4ItrsSample(
                time_s=0.0,
                satellite_id="sat-00001",
                position_itrs_km=(7000.0, 0.0, 0.0),
                velocity_itrs_km_s=(0.0, 7.5, 0.0),
            ),
            Sgp4ItrsSample(
                time_s=60.0,
                satellite_id="sat-00001",
                position_itrs_km=(7000.0, 0.0, 0.0),
                velocity_itrs_km_s=(0.0, 7.5, 0.0),
            ),
        ),
        iers_a_sha256="a" * 64,
        iers_readme_sha256="b" * 64,
        leap_seconds_sha256="c" * 64,
        earth_orientation_quality="final",
        iers_data_release="unit",
        astropy_iers_data_version="unit",
    )


class StkValidationMetricTests(unittest.TestCase):
    def test_computes_orbit_minus_stk_vectors_ric_and_summaries(self):
        report = compare_stk_state_capture(
            _capture(),
            _teme_result(),
            _fixed_result(),
        )

        first = report.teme.samples[0]
        for actual, expected in zip(
            first.position_difference_m,
            (1.0, 2.0, 3.0),
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(first.velocity_difference_m_s[0], 0.1)
        self.assertAlmostEqual(first.velocity_difference_m_s[1], 0.2)
        self.assertAlmostEqual(first.velocity_difference_m_s[2], 0.3)
        self.assertAlmostEqual(first.position_error_m, math.sqrt(14.0))
        self.assertAlmostEqual(first.velocity_error_m_s, math.sqrt(0.14))
        self.assertAlmostEqual(first.radial_position_error_m, 1.0)
        self.assertAlmostEqual(first.in_track_position_error_m, 2.0)
        self.assertAlmostEqual(first.cross_track_position_error_m, 3.0)

        summaries = {metric.metric: metric for metric in report.teme.metrics}
        position = summaries["position_error"]
        self.assertAlmostEqual(position.rms, math.sqrt((14.0 + 77.0) / 2.0))
        self.assertAlmostEqual(
            position.p95,
            math.sqrt(14.0) + 0.95 * (math.sqrt(77.0) - math.sqrt(14.0)),
        )
        self.assertAlmostEqual(position.maximum, math.sqrt(77.0))
        self.assertEqual(position.worst_satellite_id, "sat-00001")
        self.assertEqual(position.worst_time_s, 60.0)

        radial = summaries["radial_position_error"]
        self.assertAlmostEqual(radial.rms, math.sqrt((1.0 + 16.0) / 2.0))
        self.assertAlmostEqual(radial.p95, 3.85)
        self.assertAlmostEqual(radial.maximum, 4.0)

    def test_keeps_teme_and_fixed_metrics_separate(self):
        report = compare_stk_state_capture(
            _capture(),
            _teme_result(),
            _fixed_result(),
        )

        self.assertGreater(report.teme.metrics[0].maximum, 0.0)
        self.assertEqual(report.fixed.metrics[0].maximum, 0.0)
        self.assertEqual(report.teme.stk_reference_frame, "TEME")
        self.assertEqual(report.teme.orbit_reference_frame, "TEME")
        self.assertEqual(report.fixed.stk_reference_frame, "Fixed")
        self.assertEqual(report.fixed.orbit_reference_frame, "ITRS")

    def test_canonical_report_records_conventions_and_is_repeatable(self):
        first = compare_stk_state_capture(
            _capture(),
            _teme_result(),
            _fixed_result(),
        )
        second = compare_stk_state_capture(
            _capture(),
            _teme_result(),
            _fixed_result(),
        )

        self.assertEqual(first.canonical_bytes(), second.canonical_bytes())
        self.assertTrue(first.canonical_bytes().endswith(b"\n"))
        self.assertNotIn(b"\r\n", first.canonical_bytes())
        self.assertEqual(first.difference_sign, "orbit_minus_stk")
        self.assertEqual(first.ric_basis, "stk_reference_state")
        self.assertEqual(first.p95_method, "type-7-linear")
        self.assertFalse(first.formal_thresholds_frozen)

    def test_rejects_identity_frame_and_sample_mismatches(self):
        cases = (
            (
                _capture(),
                replace(_teme_result(), element_set_id="wrong"),
                _fixed_result(),
                "element_set_id",
            ),
            (
                _capture(),
                replace(_teme_result(), reference_frame="GCRS"),
                _fixed_result(),
                "must use TEME",
            ),
            (
                _capture(),
                replace(_teme_result(), samples=tuple(reversed(_teme_result().samples))),
                _fixed_result(),
                "ordered",
            ),
            (
                replace(_capture(), fixed_samples=(_stk_sample(0.0),)),
                _teme_result(),
                _fixed_result(),
                "sample keys",
            ),
        )
        for capture, teme, fixed, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                StkValidationError,
                message,
            ):
                compare_stk_state_capture(capture, teme, fixed)

    def test_rejects_duplicate_nonfinite_and_degenerate_reference_states(self):
        duplicate = replace(
            _capture(),
            teme_samples=(_stk_sample(0.0), _stk_sample(0.0)),
        )
        with self.assertRaisesRegex(StkValidationError, "duplicate"):
            compare_stk_state_capture(
                duplicate,
                _teme_result(),
                _fixed_result(),
            )

        nonfinite_teme = replace(
            _teme_result(),
            samples=(
                replace(
                    _teme_result().samples[0],
                    position_teme_km=(math.nan, 0.0, 0.0),
                ),
                _teme_result().samples[1],
            ),
        )
        with self.assertRaisesRegex(StkValidationError, "finite"):
            compare_stk_state_capture(
                _capture(),
                nonfinite_teme,
                _fixed_result(),
            )

        degenerate_samples = (
            _stk_sample(0.0, position=(0.0, 0.0, 0.0)),
            _stk_sample(60.0, position=(0.0, 0.0, 0.0)),
        )
        degenerate = replace(
            _capture(),
            teme_samples=degenerate_samples,
            fixed_samples=degenerate_samples,
        )
        with self.assertRaisesRegex(StkValidationError, "RIC radial axis"):
            compare_stk_state_capture(
                degenerate,
                _teme_result(),
                _fixed_result(),
            )


if __name__ == "__main__":
    unittest.main()
