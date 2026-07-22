import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.experiments import load_experiment_spec, run_experiment


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/fixtures/manifests/smoke_circular_geometry_trace.json"
MATRIX_MANIFEST = ROOT / "tests/fixtures/manifests/smoke_circular_geometry_runtime_matrix.json"
CIRCULAR_TRACE_MODEL = "circular_geometry_trace"


class CircularGeometryTraceManifestTests(unittest.TestCase):
    def test_manifest_loads_orbital_set_trace_options(self):
        spec = load_experiment_spec(MANIFEST)

        self.assertEqual(spec.experiment_id, "smoke_circular_geometry_trace")
        self.assertEqual(spec.simulation.trace_model, CIRCULAR_TRACE_MODEL)
        self.assertEqual(spec.constellation.compute_cell_count, 4)
        self.assertEqual(spec.constellation.satellite_count, 8)
        self.assertEqual(len(spec.constellation.effective_orbital_sets), 1)
        options = spec.simulation.trace_options
        self.assertEqual(options["propagation_model"], "circular_two_body")
        self.assertEqual(options["eci_ecef_model"], "simplified_gmst")
        self.assertNotIn("link_profiles", options)
        self.assertNotIn("isl_bandwidth_bps", options)
        self.assertNotIn("sgl_bandwidth_bps", options)
        self.assertFalse(options["stk_calibration"]["enabled"])
        gateway_profile = spec.constellation.gateway_profile
        self.assertEqual(gateway_profile["bandwidth_bps"], 1_000_000_000)
        self.assertEqual(gateway_profile["terminal_count_per_satellite"], 1)
        self.assertNotIn("architecture_isl_profile", spec.constellation.architecture_analysis())

    def test_removed_simulation_link_inputs_are_rejected_instead_of_silently_ignored(self):
        base_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        removed_inputs = {
            "link_profiles": {"same_plane_100g_optical": {"bandwidth_bps": 1}},
            "isl_bandwidth_bps": 1,
            "sgl_bandwidth_bps": 1,
        }
        for key, value in removed_inputs.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmpdir:
                payload = json.loads(json.dumps(base_payload))
                payload["simulation"]["trace_options"][key] = value
                path = Path(tmpdir) / f"removed-{key}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, key):
                    load_experiment_spec(path)

    def test_removed_satellite_link_catalog_is_rejected(self):
        base_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        base_payload["constellation"]["orbital_sets"][0]["satellite_profile"] = {
            "link_profiles": {
                "same_plane_1g_conflict": {
                    "bandwidth_bps": 1_000_000_000,
                    "terminal_count_per_satellite": 1,
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "removed-satellite-link-catalog.json"
            path.write_text(json.dumps(base_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"satellite_profile\.link_profiles"):
                load_experiment_spec(path)

    def test_orbital_sets_reject_removed_isl_topology_field(self):
        base_payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        base_payload["constellation"]["isl_topology"] = {
            "mode": "ring",
            "terminal_count_per_sat": 99,
            "bandwidth_gbps": 1,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "mixed-isl-authorities.json"
            path.write_text(json.dumps(base_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"isl_topology was removed"):
                load_experiment_spec(path)

    def test_runner_smoke_does_not_promote_geometry_trace_to_paper_results(self):
        spec = load_experiment_spec(MANIFEST)
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = run_experiment(
                replace(spec, simulation=replace(spec.simulation, trace_output_dir=tmpdir))
            ).to_dict()
        runtime = payload["performance"]["details"]["runtime"]
        self.assertTrue(payload["cost"]["evaluated"])
        self.assertTrue(payload["performance"]["evaluated"])
        self.assertEqual(runtime["provenance"]["simulation_trace"]["trace_model"], CIRCULAR_TRACE_MODEL)
        self.assertFalse(runtime["valid_for_paper_main_results"])
        self.assertNotIn("runtime_optimization", payload["performance"]["details"])
        self.assertEqual(payload["module_status"]["runtime_performance"], "evaluated")
        self.assertIn("valid_for_paper_main_results=false", " ".join(payload["warnings"]))

    def test_cli_smoke_path_writes_json_without_runtime_promotion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "smoke_circular_geometry_trace.json"
            completed = subprocess.run(
                [sys.executable, "-m", "orbitserve.experiments.cli", str(MANIFEST), "--out", str(out_path)],
                cwd=ROOT, env=dict(os.environ), check=False, capture_output=True, text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(data["schema_version"], "orbitserve.experiment_summary.v1")
        self.assertEqual(data["output_mode"], "summary")
        self.assertTrue(data["artifact_manifest"]["enabled"])
        self.assertFalse(data["performance"]["details"]["runtime"]["valid_for_paper_main_results"])
        self.assertIn(data["module_status"]["simulation"], {"stub", "geometry_trace", "surrogate_trace"})

    def test_dse_row_receives_circular_geometry_trace_when_enabled(self):
        spec = load_experiment_spec(MANIFEST)
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = replace(
                spec,
                simulation=replace(spec.simulation, trace_output_dir=tmpdir),
                dse=replace(spec.dse, enabled=True, sweep={"altitude_km": (550,)}),
            )
            payload = run_experiment(spec).to_dict()

        self.assertTrue(payload["dse"]["evaluated"])
        row_runtime = payload["dse"]["rows"][0]["performance"]["details"]["runtime"]
        self.assertTrue(row_runtime["evaluated"])
        self.assertEqual(row_runtime["trace_model"], CIRCULAR_TRACE_MODEL)
        self.assertEqual(row_runtime["scope"], "geometry_trace_surrogate")
        self.assertFalse(row_runtime["valid_for_paper_main_results"])
        self.assertFalse(payload["dse"]["runtime_matrix"]["evaluated"])

    def test_runtime_matrix_smoke_scores_preliminary_rows_without_paper_promotion(self):
        spec = load_experiment_spec(MATRIX_MANIFEST)
        with tempfile.TemporaryDirectory() as tmpdir:
            payload = run_experiment(
                replace(spec, simulation=replace(spec.simulation, trace_output_dir=tmpdir))
            ).to_dict()
        runtime = payload["performance"]["details"]["runtime"]
        matrix = payload["dse"]["runtime_matrix"]
        frontier = payload["dse"]["frontier"]
        self.assertTrue(runtime["evaluated"])
        self.assertFalse(runtime["valid_for_paper_main_results"])
        self.assertNotIn("architecture_isl_profile", payload["constellation_architecture"])
        self.assertTrue(matrix["evaluated"])
        self.assertEqual(matrix["analysis_scope"], "preliminary_model")
        self.assertGreater(matrix["row_count"], 0)
        self.assertEqual(len(matrix["selected_rows"]), 2)
        self.assertTrue(frontier["evaluated"])
        self.assertEqual(frontier["frontier_count"], 2)
        self.assertTrue(all(not row["valid_for_paper_main_results"] for row in matrix["selected_rows"]))


if __name__ == "__main__":
    unittest.main()
