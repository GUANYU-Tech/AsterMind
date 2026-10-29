import hashlib
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orbitserve.orbit import (
    ASTROPY_ENGINE_VERSION,
    ORBIT_TRACE_SCOPE,
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
    evaluate_sgp4_geometry,
    generate_orbit_trace,
    load_orbit_trace,
    write_sgp4_orbit_trace,
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
            Sgp4SunPositionSample(time_s, (149_600_000.0, 0.0, 0.0)) for time_s in times
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


class Sgp4OrbitTraceTests(unittest.TestCase):
    def test_adapter_writes_explicit_frames_geometry_windows_and_provenance(self):
        elements, propagation, earth_fixed, geometry = _pipeline_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            manifest = write_sgp4_orbit_trace(
                trace_dir,
                elements=elements,
                propagation=propagation,
                earth_fixed=earth_fixed,
                geometry=geometry,
            )
            bundle = load_orbit_trace(trace_dir / "trace_manifest.json")
            files = _trace_files(trace_dir)

        self.assertEqual(manifest.trace_model, SGP4_ORBIT_TRACE_MODEL)
        self.assertEqual(manifest.scope, ORBIT_TRACE_SCOPE)
        self.assertFalse(manifest.valid_for_paper_main_results)
        self.assertEqual(bundle.trace_id, manifest.trace_id)
        self.assertEqual(bundle.time["end_s"], 60.0)
        self.assertEqual(bundle.time["step_s"], 60.0)
        self.assertIn("mean_motion_rev_per_day", bundle.satellite_catalog[0])
        ephemeris = bundle.ephemeris_samples[0]
        self.assertEqual(ephemeris["source_reference_frame"], "TEME")
        self.assertEqual(ephemeris["earth_fixed_reference_frame"], "ITRS")
        self.assertIn("position_teme_km", ephemeris)
        self.assertIn("velocity_teme_km_s", ephemeris)
        self.assertIn("position_itrs_km", ephemeris)
        self.assertIn("velocity_itrs_km_s", ephemeris)
        self.assertNotIn("r_eci_km", ephemeris)
        self.assertGreater(len(bundle.sgl_windows), 0)
        self.assertIn("propagation_delay_min_s", bundle.sgl_windows[0])
        self.assertGreater(len(bundle.isl_windows), 0)
        self.assertIn("distance_max_km", bundle.isl_windows[0])
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
            first = write_sgp4_orbit_trace(
                first_dir,
                elements=elements,
                propagation=propagation,
                earth_fixed=earth_fixed,
                geometry=geometry,
                warnings=("audit warning", "audit warning"),
            )
            second = write_sgp4_orbit_trace(
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
            "8518bc2af62efb317a3fd5656ef45b689e1b09b9b817db4083207338bb47733f",
        )

    def test_synthetic_and_predictive_status_are_explicit_warnings(self):
        elements, propagation, earth_fixed, geometry = _pipeline_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = write_sgp4_orbit_trace(
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
        self.assertNotIn("not yet consumed", warning_text)

    def test_alignment_and_uniform_grid_fail_before_writing(self):
        elements, propagation, earth_fixed, geometry = _pipeline_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            with self.assertRaisesRegex(ValueError, "element_set_id"):
                write_sgp4_orbit_trace(
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
                write_sgp4_orbit_trace(
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
                write_sgp4_orbit_trace(
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
                write_sgp4_orbit_trace(
                    Path(tmpdir) / "trace",
                    elements=elements,
                    propagation=propagation,
                    earth_fixed=earth_fixed,
                    geometry=geometry,
                    valid_for_paper_main_results=1,
                )
            with self.assertRaisesRegex(TypeError, "sequence"):
                write_sgp4_orbit_trace(
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
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_dir = Path(tmpdir) / "trace"
            manifest = generate_orbit_trace(
                trace_dir,
                elements=_bundle(),
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                sample_times_s=(0.0, 60.0),
                ground_ingresses=(
                    {"ingress_id": "equator", "lat_deg": 0.0, "lon_deg": 0.0},
                ),
                isl_pairs=(("sat-00001", "sat-00002"),),
            )
            loaded = load_orbit_trace(trace_dir / "trace_manifest.json")

        self.assertEqual(loaded.trace_id, manifest.trace_id)
        self.assertEqual(len(loaded.ephemeris_samples), 4)
        self.assertEqual(loaded.trace_model, SGP4_ORBIT_TRACE_MODEL)

    def test_generate_orbit_trace_rejects_invalid_gateway_contract(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            self.assertRaisesRegex(ValueError, "missing required fields"),
        ):
            generate_orbit_trace(
                Path(tmpdir) / "trace",
                elements=_bundle(),
                simulation_epoch_utc="2026-07-20T00:00:00.000000Z",
                sample_times_s=(0.0, 60.0),
                ground_ingresses=(
                    {
                        "ingress_id": "equator",
                        "lat_deg": 0.0,
                    },
                ),
                isl_pairs=(),
            )


if __name__ == "__main__":
    unittest.main()


class SampleHoldWindowsTests(unittest.TestCase):
    def test_sampled_window_transition_uses_observed_boundary_without_interpolation(
        self,
    ):
        from orbitserve.orbit.sgp4.geometry import _state_windows

        samples = (
            (0.0, False, -1.0),
            (1.0, True, 3.0),
            (2.0, True, 1.0),
            (3.0, False, -1.0),
        )
        self.assertEqual(
            _state_windows(samples, sample_hold=True),
            ((0.0, 1.0, False), (1.0, 3.0, True)),
        )


class BatchedTraceTests(unittest.TestCase):
    def test_parallel_memory_handoff_matches_verified_files_without_batch_reads(self):
        times = tuple(float(t) for t in range(122))
        with tempfile.TemporaryDirectory() as folder:
            handoff, timings = [], {}
            with patch('orbitserve.orbit.trace.load_orbit_trace', side_effect=AssertionError('intermediate trace read')):
                generate_orbit_trace(Path(folder)/'parallel', elements=_bundle(),
                    simulation_epoch_utc='2026-07-20T00:00:00.000000Z',
                    sample_times_s=times, ground_ingresses=(),
                    isl_pairs=(('sat-00001', 'sat-00002'),), workers=2,
                    _bundle_sink=handoff, _timings=timings)
            loaded = load_orbit_trace(Path(folder)/'parallel/trace_manifest.json')
            self.assertEqual(handoff, [loaded])
            self.assertGreater(timings['position_s'], 0.)
            self.assertGreater(timings['candidate_geometry_s'], 0.)
            generate_orbit_trace(Path(folder)/'serial', elements=_bundle(),
                simulation_epoch_utc='2026-07-20T00:00:00.000000Z',
                sample_times_s=times, ground_ingresses=(),
                isl_pairs=(('sat-00001', 'sat-00002'),), workers=1)
            serial = load_orbit_trace(Path(folder)/'serial/trace_manifest.json')
            self.assertEqual(loaded.trace_id, serial.trace_id)
            self.assertEqual(loaded.ephemeris_samples, serial.ephemeris_samples)
            self.assertEqual(loaded.isl_windows, serial.isl_windows)
            self.assertEqual(loaded.sunlight_windows, serial.sunlight_windows)

    def test_time_batches_preserve_states_and_merge_windows_without_boundary_duplicates(
        self,
    ):
        from orbitserve.orbit.sgp4.elements import load_sgp4_element_bundle
        from orbitserve.orbit.sgp4.trace import _state_partition

        bundle = load_sgp4_element_bundle(_bundle())
        times = tuple(float(t) for t in range(122))
        epoch = "2026-07-20T00:00:00.000000Z"
        with tempfile.TemporaryDirectory() as folder:
            chunked = generate_orbit_trace(
                Path(folder) / "chunked",
                elements=bundle,
                simulation_epoch_utc=epoch,
                sample_times_s=times,
                ground_ingresses=(),
                isl_pairs=(
                    (
                        bundle.satellites[0].satellite_id,
                        bundle.satellites[1].satellite_id,
                    ),
                ),
                workers=1,
            )
            propagation, fixed = _state_partition((bundle, epoch, times))
            from orbitserve.orbit import evaluate_sgp4_geometry

            geometry = evaluate_sgp4_geometry(
                fixed,
                ground_ingresses=(),
                isl_pairs=(
                    (
                        bundle.satellites[0].satellite_id,
                        bundle.satellites[1].satellite_id,
                    ),
                ),
                sample_hold=True,
            )
            whole = write_sgp4_orbit_trace(
                Path(folder) / "whole",
                elements=bundle,
                propagation=propagation,
                earth_fixed=fixed,
                geometry=geometry,
                sample_hold=True,
            )
            self.assertEqual(chunked.trace_id, whole.trace_id)
            self.assertEqual(chunked.provenance["precompute"]["time_batches"], 2)
