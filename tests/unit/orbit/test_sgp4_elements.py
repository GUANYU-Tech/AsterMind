import json
import math
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from astermind.builder.orbit import SGP4_ELEMENT_BUNDLE_SCHEMA, Sgp4ElementBundle, Sgp4ElementProvenance, Sgp4SatelliteElements, load_sgp4_element_bundle


def _satellite(satellite_id="sat-00001", satnum=10001):
    return {
        "satellite_id": satellite_id,
        "satnum": satnum,
        "classification": "U",
        "object_id": "2026-001A",
        "epoch_utc": "2026-07-20T00:00:00.000000Z",
        "inclination_deg": 53.0,
        "raan_deg": -0.0,
        "eccentricity": 0.001,
        "argument_of_perigee_deg": 0.0,
        "mean_anomaly_deg": 0.0,
        "mean_motion_rev_per_day": 15.25,
        "mean_motion_dot_rev_per_day2": 0.0,
        "mean_motion_ddot_rev_per_day3": 0.0,
        "bstar_inv_earth_radii": 0.00001,
        "ephemeris_type": 0,
        "element_set_number": 1,
        "revolution_number_at_epoch": 1,
    }


def _bundle():
    return {
        "schema_version": SGP4_ELEMENT_BUNDLE_SCHEMA,
        "element_semantics": "sgp4_mean_elements",
        "gravity_model": "wgs72",
        "ops_mode": "improved",
        "time_scale": "utc",
        "satellites": [
            _satellite("sat-00002", 10002),
            _satellite("sat-00001", 10001),
        ],
        "provenance": {
            "source_kind": "synthetic_sgp4_mean_elements",
            "source_name": "unit-scenario-builder",
            "source_version": "unit-v1",
            "observationally_fitted": False,
        },
    }


class Sgp4ElementBundleTests(unittest.TestCase):
    def test_public_models_normalize_order_numbers_and_roundtrip(self):
        bundle = Sgp4ElementBundle.from_value(_bundle())

        self.assertIsInstance(bundle.satellites[0], Sgp4SatelliteElements)
        self.assertIsInstance(bundle.provenance, Sgp4ElementProvenance)
        self.assertEqual([item.satellite_id for item in bundle.satellites], ["sat-00001", "sat-00002"])
        self.assertEqual(bundle.satellites[0].raan_deg, 0.0)
        self.assertEqual(math.copysign(1.0, bundle.satellites[0].raan_deg), 1.0)
        self.assertIs(Sgp4ElementBundle.from_value(bundle), bundle)
        self.assertEqual(Sgp4ElementBundle.from_value(bundle.to_dict()), bundle)
        self.assertTrue(bundle.canonical_bytes().endswith(b"\n"))
        self.assertNotIn(b"\r\n", bundle.canonical_bytes())
        self.assertNotIn(b"element_set_id", bundle.canonical_bytes())

    def test_element_set_id_ignores_input_order_and_provenance(self):
        first = _bundle()
        second = deepcopy(first)
        second["satellites"].reverse()
        second["provenance"]["source_name"] = "independent-audit-source"

        first_bundle = Sgp4ElementBundle.from_value(first)
        second_bundle = Sgp4ElementBundle.from_value(second)

        self.assertEqual(first_bundle.element_set_id, second_bundle.element_set_id)
        self.assertEqual(
            first_bundle.element_set_id,
            "sgp4-elements-v1-bcec1599cf44d9352b48a2dbeb2901a44e5bdaf9023abf5464c89f16860446b0",
        )
        self.assertNotEqual(first_bundle.canonical_bytes(), second_bundle.canonical_bytes())
        self.assertRegex(first_bundle.element_set_id, r"^sgp4-elements-v1-[0-9a-f]{64}$")

    def test_rejects_missing_unknown_duplicate_and_wrong_constant_fields(self):
        cases = []
        missing = _bundle()
        missing.pop("time_scale")
        cases.append((missing, "missing required fields"))
        unknown = _bundle()
        unknown["tle_line1"] = "not-authoritative"
        cases.append((unknown, "unknown fields"))
        bad_constant = _bundle()
        bad_constant["gravity_model"] = "wgs84"
        cases.append((bad_constant, "gravity_model"))
        duplicate_id = _bundle()
        duplicate_id["satellites"][1]["satellite_id"] = "sat-00002"
        cases.append((duplicate_id, "satellite_id values must be unique"))
        duplicate_satnum = _bundle()
        duplicate_satnum["satellites"][1]["satnum"] = 10002
        cases.append((duplicate_satnum, "satnum values must be unique"))

        for payload, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex((TypeError, ValueError), message):
                Sgp4ElementBundle.from_value(payload)

    def test_rejects_invalid_element_types_ranges_epoch_and_object_id(self):
        cases = (
            ("satnum", True, "integer"),
            ("satnum", 0, r"\[1, 339999\]"),
            ("raan_deg", 360.0, r"\[0.0, 360.0\)"),
            ("eccentricity", 1.0, r"\[0.0, 1.0\)"),
            ("mean_motion_rev_per_day", 0.0, "strictly positive"),
            ("mean_motion_dot_rev_per_day2", float("nan"), "finite"),
            ("epoch_utc", "2026-07-20T00:00:00Z", "canonical form"),
            ("epoch_utc", "2026-02-30T00:00:00.000000Z", "valid canonical"),
            ("object_id", "26-001A", "CCSDS form"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                payload = _bundle()
                payload["satellites"][0][field] = value
                with self.assertRaisesRegex((TypeError, ValueError), message):
                    Sgp4ElementBundle.from_value(payload)

    def test_rejects_inconsistent_or_extended_provenance(self):
        inconsistent = _bundle()
        inconsistent["provenance"]["observationally_fitted"] = True
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            Sgp4ElementBundle.from_value(inconsistent)

        extended = _bundle()
        extended["provenance"]["generated_at"] = "host-clock-value"
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            Sgp4ElementBundle.from_value(extended)

    def test_direct_public_constructors_cannot_bypass_validation(self):
        valid = Sgp4ElementBundle.from_value(_bundle())
        invalid_satellite = valid.satellites[0].to_dict()
        invalid_satellite["eccentricity"] = 1.0
        with self.assertRaisesRegex(ValueError, r"\[0.0, 1.0\)"):
            Sgp4SatelliteElements(**invalid_satellite)

        with self.assertRaisesRegex(ValueError, "inconsistent"):
            Sgp4ElementProvenance(
                source_kind="synthetic_sgp4_mean_elements",
                source_name="direct-constructor",
                source_version="v1",
                observationally_fitted=True,
            )

        with self.assertRaisesRegex(ValueError, "satellite_id values must be unique"):
            Sgp4ElementBundle(
                satellites=(valid.satellites[0], valid.satellites[0]),
                provenance=valid.provenance,
            )

    def test_loader_rejects_duplicate_keys_and_nonfinite_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            valid_path = Path(tmpdir) / "valid.json"
            valid_path.write_text(json.dumps(_bundle()), encoding="utf-8")
            loaded = load_sgp4_element_bundle(valid_path)
            self.assertIs(load_sgp4_element_bundle(loaded), loaded)
            self.assertEqual(load_sgp4_element_bundle(_bundle()), loaded)

            duplicate_path = Path(tmpdir) / "duplicate.json"
            duplicate_path.write_text('{"schema_version":"a","schema_version":"b"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                load_sgp4_element_bundle(duplicate_path)

            nonfinite_path = Path(tmpdir) / "nonfinite.json"
            nonfinite_path.write_text('{"value":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON number"):
                load_sgp4_element_bundle(nonfinite_path)


if __name__ == "__main__":
    unittest.main()
