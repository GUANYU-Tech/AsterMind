import hashlib
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from orbitserve.orbit import (
    ASTROPY_ENGINE_VERSION,
    ORBIT_TRACE_V2_SCOPE,
    SGP4_ENGINE_VERSION,
    SGP4_ORBIT_TRACE_MODEL,
    WGS84_SEMI_MAJOR_AXIS_KM,
    Sgp4EarthFixedResult,
    Sgp4ElementBundle,
    Sgp4ItrsSample,
    Sgp4PropagationResult,
    Sgp4SunPositionResult,
    Sgp4SunPositionSample,
    Sgp4TemeSample,
    compute_sgp4_sun_positions_itrs,
    evaluate_sgp4_geometry,
    load_orbit_trace_v2,
    propagate_sgp4_teme,
    transform_sgp4_teme_to_itrs,
    write_sgp4_orbit_trace_v2,
)

from .test_sgp4_elements import _bundle
from .test_sgp4_geometry import _has_version


def _pipeline_inputs(times=(0.0, 60.0)):
    elements = Sgp4ElementBundle.from_value(_bundle())
    radius = WGS84_SEMI_MAJOR_AXIS_KM + 500.0
    positions = {
        "sat-00001": (radius, 0.0, 0.0),
        "sat-00002": (radius, 1000.0, 0.0),
    }
    teme_samples = []
    itrs_samples = []
    for time_s in times:
        for satellite_id in ("sat-00001", "sat-00002"):
            position = positions[satellite_id]
            velocity = (0.0, 7.5, 0.0)
            teme_samples.append(
                Sgp4TemeSample(time_s, satellite_id, position, velocity)
            )
            itrs_samples.append(
                Sgp4ItrsSample(time_s, satellite_id, position, velocity)
            )
    propagation = Sgp4PropagationResult(
        element_set_id=elements.element_set_id,
        simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
        samples=tuple(teme_samples),
    )
    earth_fixed = Sgp4EarthFixedResult(
        element_set_id=elements.element_set_id,
        simulation_epoch_utc=propagation.simulation_epoch_utc,
        samples=tuple(itrs_samples),
        iers_a_sha256="unit-iers-a",
        iers_readme_sha256="unit-iers-readme",
        leap_seconds_sha256="unit-leap-seconds",
        earth_orientation_quality="predictive",
        iers_data_release="unit-iers-release",
        astropy_iers_data_version="unit-iers-package",
    )
    sun_positions = Sgp4SunPositionResult(
        element_set_id=elements.element_set_id,
        simulation_epoch_utc=propagation.simulation_epoch_utc,
        samples=tuple(
            Sgp4SunPositionSample(time_s, (149_600_000.0, 0.0, 0.0))
            for time_s in times
        ),
        iers_a_sha256=earth_fixed.iers_a_sha256,
        iers_readme_sha256=earth_fixed.iers_readme_sha256,
        leap_seconds_sha256=earth_fixed.leap_seconds_sha256,
        earth_orientation_quality=earth_fixed.earth_orientation_quality,
        iers_data_release=earth_fixed.iers_data_release,
        astropy_iers_data_version=earth_fixed.astropy_iers_data_version,
    )
    geometry = evaluate_sgp4_geometry(
        earth_fixed,
        ground_ingresses=(
            {
                "ingress_id": "equator",
                "lat_deg": 0.0,
                "lon_deg": 0.0,
                "min_elevation_deg": 5.0,
            },
        ),
        isl_pairs=(("sat-00002", "sat-00001"),),
        sun_positions=sun_positions,
    )
    return elements, propagation, earth_fixed, geometry


def _trace_files(path):
    return {
        item.name: item.read_bytes()
        for item in sorted(Path(path).iterdir(), key=lambda value: value.name)
        if item.is_file()
    }


class Sgp4OrbitTraceV2Tests(unittest.TestCase):
    def test_adapter_writes_explicit_frames_geometry_windows_and_provenance(self):
        elements, propagation, earth_fixed, geometry = _pipeline_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            manifest = write_sgp4_orbit_trace_v2(
                trace_dir,
                elements=elements,
                propagation=propagation,
                earth_fixed=earth_fixed,
                geometry=geometry,
            )
            bundle = load_orbit_trace_v2(trace_dir / "trace_manifest.json")
            files = _trace_files(trace_dir)

        self.assertEqual(manifest.trace_model, SGP4_ORBIT_TRACE_MODEL)
        self.assertEqual(manifest.scope, ORBIT_TRACE_V2_SCOPE)
        self.assertFalse(manifest.valid_for_paper_main_results)
        self.assertEqual(bundle.trace_id, manifest.trace_id)
        self.assertEqual(bundle.horizon_s, 60.0)
        self.assertEqual(bundle.step_s, 60.0)
        self.assertIn("mean_motion_rev_per_day", bundle.satellite_catalog[0])
        ephemeris = bundle.ephemeris_samples[0]
        self.assertEqual(ephemeris["source_reference_frame"], "TEME")
        self.assertEqual(ephemeris["earth_fixed_reference_frame"], "ITRS")
        self.assertIn("position_teme_km", ephemeris)
        self.assertIn("velocity_teme_km_s", ephemeris)
        self.assertIn("position_itrs_km", ephemeris)
        self.assertIn("velocity_itrs_km_s", ephemeris)
        self.assertNotIn("r_eci_km", ephemeris)
        self.assertGreater(len(bundle.access_windows), 0)
        self.assertIn("propagation_delay_min_s", bundle.access_windows[0])
        self.assertGreater(len(bundle.link_trace["isl_windows"]), 0)
        self.assertIn("distance_max_km", bundle.link_trace["isl_windows"][0])
        provenance = bundle.provenance
        self.assertEqual(
            provenance["element_input"]["element_set_id"], elements.element_set_id
        )
        self.assertEqual(
            provenance["earth_fixed_transform"]["iers_a_sha256"],
            earth_fixed.iers_a_sha256,
        )
        all_text = b"".join(files.values())
        self.assertNotIn(b"bandwidth_bps", all_text)
        self.assertNotIn(b"terminal_catalog", all_text)
        self.assertNotIn(b"satellite_initial_soc", all_text)

    def test_canonical_trace_bytes_ignore_element_input_order_and_output_path(self):
        elements, propagation, earth_fixed, geometry = _pipeline_inputs()
        reversed_elements = deepcopy(_bundle())
        reversed_elements["satellites"].reverse()
        with tempfile.TemporaryDirectory() as tmpdir:
            first_dir = Path(tmpdir) / "first"
            second_dir = Path(tmpdir) / "second"
            first = write_sgp4_orbit_trace_v2(
                first_dir,
                elements=elements,
                propagation=propagation,
                earth_fixed=earth_fixed,
                geometry=geometry,
                warnings=("audit warning", "audit warning"),
            )
            second = write_sgp4_orbit_trace_v2(
                second_dir,
                elements=reversed_elements,
                propagation=propagation,
                earth_fixed=earth_fixed,
                geometry=geometry,
                warnings=("audit warning",),
            )
            first_files = _trace_files(first_dir)
            second_files = _trace_files(second_dir)

        self.assertEqual(first.trace_id, second.trace_id)
        self.assertEqual(first_files, second_files)
        self.assertEqual(
            hashlib.sha256(first_files["trace_manifest.json"]).hexdigest(),
            "e3112b1645ceebd1bb333b3985378648c4b862df4ea9b08fef9804e444a46f59",
        )

    def test_synthetic_and_predictive_status_are_explicit_warnings(self):
        elements, propagation, earth_fixed, geometry = _pipeline_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_sgp4_orbit_trace_v2(
                Path(tmpdir) / "trace",
                elements=elements,
                propagation=propagation,
                earth_fixed=earth_fixed,
                geometry=geometry,
                valid_for_paper_main_results=True,
            )
        warning_text = "\n".join(manifest.warnings)
        self.assertTrue(manifest.valid_for_paper_main_results)
        self.assertIn("synthetic", warning_text)
        self.assertIn("predictive", warning_text)
        self.assertIn("opt-in", warning_text)

    def test_alignment_and_uniform_grid_fail_before_writing(self):
        elements, propagation, earth_fixed, geometry = _pipeline_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            with self.assertRaisesRegex(ValueError, "element_set_id"):
                write_sgp4_orbit_trace_v2(
                    trace_dir,
                    elements=elements,
                    propagation=propagation,
                    earth_fixed=earth_fixed,
                    geometry=replace(geometry, element_set_id="wrong"),
                )
            self.assertFalse(trace_dir.exists())

        irregular = _pipeline_inputs((0.0, 50.0, 120.0))
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            with self.assertRaisesRegex(ValueError, "uniform"):
                write_sgp4_orbit_trace_v2(
                    trace_dir,
                    elements=irregular[0],
                    propagation=irregular[1],
                    earth_fixed=irregular[2],
                    geometry=irregular[3],
                )
            self.assertFalse(trace_dir.exists())

        missing_itrs = replace(earth_fixed, samples=earth_fixed.samples[:-1])
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            with self.assertRaisesRegex(ValueError, "identical ordered keys"):
                write_sgp4_orbit_trace_v2(
                    trace_dir,
                    elements=elements,
                    propagation=propagation,
                    earth_fixed=missing_itrs,
                    geometry=geometry,
                )
            self.assertFalse(trace_dir.exists())

    def test_rejects_non_boolean_policy_and_invalid_warning_collection(self):
        elements, propagation, earth_fixed, geometry = _pipeline_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(TypeError, "boolean"):
                write_sgp4_orbit_trace_v2(
                    Path(tmpdir) / "trace",
                    elements=elements,
                    propagation=propagation,
                    earth_fixed=earth_fixed,
                    geometry=geometry,
                    valid_for_paper_main_results=1,
                )
            with self.assertRaisesRegex(TypeError, "sequence"):
                write_sgp4_orbit_trace_v2(
                    Path(tmpdir) / "trace",
                    elements=elements,
                    propagation=propagation,
                    earth_fixed=earth_fixed,
                    geometry=geometry,
                    warnings="not-a-sequence",
                )

    @unittest.skipUnless(
        _has_version("sgp4", SGP4_ENGINE_VERSION)
        and _has_version("astropy", ASTROPY_ENGINE_VERSION),
        "requires the frozen SGP4 and Astropy engines",
    )
    def test_complete_public_pipeline_writes_loadable_trace(self):
        elements = Sgp4ElementBundle.from_value(_bundle())
        propagation = propagate_sgp4_teme(
            elements,
            simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
            sample_times_s=(0.0, 60.0),
        )
        earth_fixed = transform_sgp4_teme_to_itrs(propagation)
        sun_positions = compute_sgp4_sun_positions_itrs(earth_fixed)
        geometry = evaluate_sgp4_geometry(
            earth_fixed,
            ground_ingresses=(
                {"ingress_id": "equator", "lat_deg": 0.0, "lon_deg": 0.0},
            ),
            isl_pairs=(("sat-00001", "sat-00002"),),
            sun_positions=sun_positions,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            manifest = write_sgp4_orbit_trace_v2(
                trace_dir,
                elements=elements,
                propagation=propagation,
                earth_fixed=earth_fixed,
                geometry=geometry,
            )
            loaded = load_orbit_trace_v2(trace_dir / "trace_manifest.json")

        self.assertEqual(loaded.trace_id, manifest.trace_id)
        self.assertEqual(len(loaded.ephemeris_samples), 4)
        self.assertEqual(loaded.trace_model, SGP4_ORBIT_TRACE_MODEL)


if __name__ == "__main__":
    unittest.main()
