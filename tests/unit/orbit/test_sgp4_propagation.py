import importlib.util
import math
import unittest
from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version
from unittest.mock import patch

from astermind.builder.orbit import SGP4_ENGINE_VERSION, SGP4_REFERENCE_FRAME, Sgp4PropagationError, propagate_sgp4_teme
from astermind.builder.orbit.sgp4.propagation import _Sgp4Api

from tests.unit.orbit.test_sgp4_elements import _bundle, _satellite


def _has_tested_sgp4():
    if importlib.util.find_spec("sgp4") is None:
        return False
    try:
        return version("sgp4") == SGP4_ENGINE_VERSION
    except PackageNotFoundError:
        return False


class _FakeSatrec:
    instances = []
    sample_error = 0
    return_nonfinite = False

    def __init__(self):
        self.init_args = None
        self.error = 0
        self.__class__.instances.append(self)

    def sgp4init(self, *args):
        self.init_args = args

    def sgp4(self, jd, fr):
        if self.sample_error:
            return self.sample_error, (math.nan,) * 3, (math.nan,) * 3
        if self.return_nonfinite:
            return 0, (math.nan,) * 3, (1.0, 2.0, 3.0)
        return 0, (jd, fr, jd + fr), (1.0, 2.0, 3.0)


def _fake_api():
    _FakeSatrec.instances = []
    _FakeSatrec.sample_error = 0
    _FakeSatrec.return_nonfinite = False
    return _Sgp4Api(
        Satrec=_FakeSatrec,
        WGS72="fake-wgs72",
        errors={6: "satellite has decayed"},
        check_satrec=lambda satrec: None,
    )


def _iss_bundle():
    payload = _bundle()
    payload["satellites"] = [
        {
            **_satellite("iss", 25544),
            "object_id": "1998-067A",
            "epoch_utc": "2019-12-09T16:38:29.363424Z",
            "inclination_deg": 51.6439,
            "raan_deg": 211.2001,
            "eccentricity": 0.0007417,
            "argument_of_perigee_deg": 17.6667,
            "mean_anomaly_deg": 85.6398,
            "mean_motion_rev_per_day": 15.50103472,
            "mean_motion_dot_rev_per_day2": 0.00001764,
            "bstar_inv_earth_radii": 0.000038792,
            "element_set_number": 999,
            "revolution_number_at_epoch": 20248,
        }
    ]
    payload["provenance"] = {
        "source_kind": "observationally_fitted_general_perturbations",
        "source_name": "python-sgp4-official-iss-example",
        "source_version": "2019-343",
        "observationally_fitted": True,
    }
    return payload


class Sgp4PropagationTests(unittest.TestCase):
    def test_fake_engine_receives_contract_units_and_output_is_canonical(self):
        payload = _bundle()
        payload["satellites"] = [payload["satellites"][0]]
        element = payload["satellites"][0]
        with patch('astermind.builder.orbit.sgp4.propagation._load_sgp4_api', return_value=_fake_api()):
            result = propagate_sgp4_teme(
                payload,
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                sample_times_s=(60.0, -0.0),
            )

        satrec = _FakeSatrec.instances[0]
        args = satrec.init_args
        self.assertEqual(args[0:3], ("fake-wgs72", "i", element["satnum"]))
        self.assertAlmostEqual(args[4], element["bstar_inv_earth_radii"], places=15)
        self.assertAlmostEqual(
            args[5], element["mean_motion_dot_rev_per_day2"] * 2.0 * math.pi / 1440.0**2
        )
        self.assertAlmostEqual(
            args[6], element["mean_motion_ddot_rev_per_day3"] * 2.0 * math.pi / 1440.0**3
        )
        self.assertAlmostEqual(args[8], math.radians(element["argument_of_perigee_deg"]))
        self.assertAlmostEqual(args[9], math.radians(element["inclination_deg"]))
        self.assertAlmostEqual(args[10], math.radians(element["mean_anomaly_deg"]))
        self.assertAlmostEqual(
            args[11], element["mean_motion_rev_per_day"] * 2.0 * math.pi / 1440.0
        )
        self.assertAlmostEqual(args[12], math.radians(element["raan_deg"]))
        self.assertEqual([sample.time_s for sample in result.samples], [0.0, 60.0])
        self.assertEqual(result.reference_frame, SGP4_REFERENCE_FRAME)
        self.assertEqual(result.to_dict()["position_unit"], "km")

    def test_sample_grid_validation_fails_before_engine_loading(self):
        invalid = (
            ((), "non-empty"),
            ((-1.0,), "non-negative"),
            ((0.0, -0.0), "unique"),
            ((float("nan"),), "finite"),
            ((True,), "finite numbers"),
        )
        for sample_times, message in invalid:
            with self.subTest(sample_times=sample_times):
                with patch('astermind.builder.orbit.sgp4.propagation._load_sgp4_api') as loader:
                    with self.assertRaisesRegex((TypeError, ValueError), message):
                        propagate_sgp4_teme(
                            _bundle(),
                            simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                            sample_times_s=sample_times,
                        )
                    loader.assert_not_called()

    def test_any_sample_error_fails_complete_call_with_context(self):
        api = _fake_api()
        _FakeSatrec.sample_error = 6
        with patch('astermind.builder.orbit.sgp4.propagation._load_sgp4_api', return_value=api):
            with self.assertRaises(Sgp4PropagationError) as caught:
                propagate_sgp4_teme(
                    _bundle(),
                    simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                    sample_times_s=(0.0,),
                )
        self.assertEqual(caught.exception.satellite_id, "sat-00001")
        self.assertEqual(caught.exception.time_s, 0.0)
        self.assertEqual(caught.exception.error_code, 6)
        self.assertEqual(caught.exception.error_message, "satellite has decayed")

    def test_nonfinite_success_payload_fails_with_sample_context(self):
        api = _fake_api()
        _FakeSatrec.return_nonfinite = True
        with patch('astermind.builder.orbit.sgp4.propagation._load_sgp4_api', return_value=api):
            with self.assertRaises(Sgp4PropagationError) as caught:
                propagate_sgp4_teme(
                    _bundle(),
                    simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                    sample_times_s=(0.0,),
                )
        self.assertEqual(caught.exception.satellite_id, "sat-00001")
        self.assertEqual(caught.exception.time_s, 0.0)
        self.assertEqual(caught.exception.error_code, -2)
        self.assertIn("non-finite position_teme_km", caught.exception.error_message)

    def test_result_order_is_independent_of_bundle_and_grid_order(self):
        api = _fake_api()
        first = _bundle()
        second = deepcopy(first)
        second["satellites"].reverse()
        with patch('astermind.builder.orbit.sgp4.propagation._load_sgp4_api', return_value=api):
            left = propagate_sgp4_teme(
                first,
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                sample_times_s=(60.0, 0.0),
            )
        with patch('astermind.builder.orbit.sgp4.propagation._load_sgp4_api', return_value=_fake_api()):
            right = propagate_sgp4_teme(
                second,
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                sample_times_s=(0.0, 60.0),
            )
        self.assertEqual(left, right)
        self.assertEqual(
            [(sample.time_s, sample.satellite_id) for sample in left.samples],
            [
                (0.0, "sat-00001"),
                (0.0, "sat-00002"),
                (60.0, "sat-00001"),
                (60.0, "sat-00002"),
            ],
        )

    @unittest.skipUnless(_has_tested_sgp4(), "requires the frozen python-sgp4 2.27 engine")
    def test_official_iss_example_agrees_with_tle_reference(self):
        from sgp4.api import Satrec, jday

        result = propagate_sgp4_teme(
            _iss_bundle(),
            simulation_epoch_utc="2019-12-09T20:42:00.000000Z",
            sample_times_s=(0.0,),
        )

        line1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0  9991"
        line2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
        reference = Satrec.twoline2rv(line1, line2)
        jd, fr = jday(2019, 12, 9, 20, 42, 0.0)
        error, expected_position, expected_velocity = reference.sgp4(jd, fr)
        self.assertEqual(error, 0)
        sample = result.samples[0]
        for actual, expected in zip(sample.position_teme_km, expected_position):
            self.assertAlmostEqual(actual, expected, delta=0.01)
        for actual, expected in zip(sample.velocity_teme_km_s, expected_velocity):
            self.assertAlmostEqual(actual, expected, delta=0.00002)


if __name__ == "__main__":
    unittest.main()
