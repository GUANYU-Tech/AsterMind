import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


import orbitserve.dse.search as dse_module
from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.architecture.hardware import SatelliteHardwareProfile
from orbitserve.workload import compile_workload_description


ROOT = Path(__file__).resolve().parents[2]


class ExperimentManifestAndCliTests(unittest.TestCase):
    def test_manifest_loader_round_trips_to_json_safe_dict(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_single_design.json")
        self.assertEqual(spec.experiment_id, "smoke_single_design")
        self.assertEqual(spec.constellation.satellite_count, 6)
        payload = spec.to_dict()
        encoded = json.dumps(payload, ensure_ascii=False)
        self.assertIn("smoke_walker_550km", encoded)
        self.assertIn("base_dir", payload)

    def test_runner_outputs_throughput_metrics_without_fake_slo_values(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_single_design.json")
        result = run_experiment(spec)
        payload = result.to_dict()
        self.assertTrue(payload["cost"]["evaluated"])
        self.assertTrue(payload["performance"]["evaluated"])
        self.assertIsNone(payload["performance"]["request_throughput_per_s"])
        self.assertGreater(payload["performance"]["prefill_tokens_per_s"], 0)
        self.assertGreater(payload["performance"]["decode_tokens_per_s"], 0)
        self.assertFalse(payload["slo"]["evaluated"])
        self.assertIsNone(payload["slo"]["served_ratio"])
        self.assertEqual(payload["module_status"]["simulation"], "stub")
        self.assertEqual(payload["module_status"]["serving_throughput"], "evaluated")
        self.assertEqual(payload["module_status"]["dse"], "disabled")
        self.assertEqual(payload["cost"]["provenance"]["single_satellite_input"], "satellite_architecture")

    def test_runner_uses_top_level_hardware_interface_for_cost_and_performance(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_hardware_interface.json")
        self.assertIsNotNone(spec.hardware)
        self.assertIsNone(spec.costing.single_satellite_case_path)
        result = run_experiment(spec)
        payload = result.to_dict()
        self.assertTrue(payload["cost"]["evaluated"])
        self.assertTrue(payload["performance"]["evaluated"])
        self.assertEqual(payload["module_status"]["hardware"], "resolved")
        self.assertEqual(payload["cost"]["provenance"]["single_satellite_input"], "satellite_architecture")
        self.assertEqual(payload["cost"]["single_satellite"]["provenance"]["N_acc"], 3)
        self.assertEqual(payload["performance"]["details"]["hardware"]["accelerator_count"], 3)
        self.assertIsNone(payload["performance"]["request_throughput_per_s"])




    def test_moe_ep_parallel_manifest_reports_expert_parallel_metrics(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_moe_ep_parallel.json")
        result = run_experiment(spec)
        payload = result.to_dict()
        details = payload["performance"]["details"]
        self.assertTrue(payload["performance"]["evaluated"])
        self.assertTrue(payload["performance"]["feasible"])
        self.assertEqual(details["deployment_mode"], "expert_parallel")
        self.assertEqual(details["parallelism"]["expert_parallel_degree"], 4)
        self.assertGreater(details["prefill_interconnect_time_s"], 0)
        self.assertGreater(
            details["provenance"]["weight_memory_breakdown_gb_per_satellite"]["routed_expert"],
            details["provenance"]["weight_memory_breakdown_gb_per_satellite"]["non_routed"],
        )

    def test_tp_pp_ep_parallel_manifest_uses_lcm_stage_rank_pool(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_tp_pp_ep_parallel.json")
        result = run_experiment(spec)
        payload = result.to_dict()
        details = payload["performance"]["details"]
        self.assertTrue(payload["performance"]["evaluated"])
        self.assertTrue(payload["performance"]["feasible"])
        self.assertEqual(details["deployment_mode"], "tensor_pipeline_expert_parallel")
        self.assertEqual(details["parallelism"]["stage_parallel_degree"], 4)
        self.assertEqual(details["satellites_per_deployment"], 8)
        self.assertGreater(details["parallelism"]["bubble_multiplier"], 1.0)

    def test_serving_owns_runtime_utilization_calibration(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_hardware_interface.json")
        spec = replace(spec, serving=replace(spec.serving, effective_utilization=0.2))
        payload = run_experiment(spec).to_dict()
        self.assertEqual(
            payload["performance"]["details"]["provenance"]["serving_calibration"]["effective_compute_utilization"],
            0.2,
        )

    def test_runner_materializes_workload_description_and_feeds_token_summary_to_serving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workload_description = {
                "schema_version": "orbitserve.workload_description",
                "workload_id": "runner_workload_description_smoke",
                "source_mode": "synthetic_generation",
                "output_bundle_schema": "orbitserve.workload",
                "seed": 20260624,
                "time": {"epoch": "synthetic_day_0", "time_horizon_s": 120, "time_bin_s": 60},
                "geography": {
                    "mode": "named_gateway_service_regions",
                    "demand_weight_normalization": "require_sum_one",
                    "regions": [
                        {
                            "region_id": "na_east",
                            "region_name": "North America East",
                            "lat_deg": 39.0,
                            "lon_deg": -77.0,
                            "timezone_utc_offset_h": -5,
                            "demand_weight": 1.0,
                            "weight_source": "unit_test",
                        }
                    ],
                },
                "gateways": {"access_mode": "direct_user_sgl", "mapping_policy": "region_center"},
                "arrival": {"process": "nhpp_synthetic", "lambda_global": 0.5, "diurnal_profile": "flat"},
                "tokens": {
                    "model_family": "llama_70b_like",
                    "task_mix": [
                        {
                            "task_type": "conversation",
                            "weight": 1.0,
                            "prompt_distribution": {
                                "type": "quantile_buckets",
                                "p50": 128,
                                "p95": 512,
                                "p99": 768,
                                "max": 1024,
                            },
                            "output_distribution": {
                                "type": "quantile_buckets",
                                "p50": 64,
                                "p95": 256,
                                "p99": 384,
                                "max": 512,
                            },
                        }
                    ],
                },
                "slo": {
                    "mix": "balanced",
                    "classes": [
                        {
                            "slo_class": "interactive_5s",
                            "weight": 1.0,
                            "ttft_deadline_s": 5,
                            "e2e_deadline_s": 30,
                        }
                    ],
                },
                "sessions": {
                    "session_mode": "open_loop_turn_arrivals",
                    "session_ratio": 0.0,
                    "context_window_tokens": 8192,
                    "turn_count_distribution": "single_turn_only",
                },
                "validation": {"normalize_region_weights": False, "require_unique_request_id": True},
                "output": {"bundle_dir": str(tmp / "bundle"), "overwrite": True},
                "provenance": {"sources": ["unit test workload description"]},
            }
            description_path = tmp / "workload_description.json"
            description_path.write_text(json.dumps(workload_description, indent=2), encoding="utf-8")

            compiled = compile_workload_description(description_path)

            manifest = json.loads((ROOT / "tests/fixtures/manifests/smoke_single_design.json").read_text(encoding="utf-8"))
            manifest["costing"]["single_satellite_case_path"] = str(
                ROOT / "examples/single_satellite/h200_gpu_derived.json"
            )
            manifest["workload"] = {"bundle_dir": compiled.bundle_dir}
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            result = run_experiment(load_experiment_spec(manifest_path))
            payload = result.to_dict()
            workload_generation = payload["workload_generation"]
            representative = workload_generation["representative_token_distribution"]

            self.assertEqual(payload["module_status"]["workload_generation"], "loaded")
            self.assertFalse(workload_generation["applied_to_serving"])
            self.assertGreater(workload_generation["request_count"], 0)
            self.assertTrue((tmp / "bundle" / "requests.jsonl").exists())
            self.assertNotIn("token_distribution", payload["performance"]["details"]["provenance"])
            self.assertGreater(representative["prompt_tokens_p50"], 0)
            self.assertFalse(payload["slo"]["evaluated"])
            self.assertIsNone(payload["slo"]["served_ratio"])

    def test_runner_reuses_materialized_workload_for_embedded_dse(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workload_description = {
                "schema_version": "orbitserve.workload_description",
                "workload_id": "runner_embedded_dse_workload_smoke",
                "source_mode": "synthetic_generation",
                "output_bundle_schema": "orbitserve.workload",
                "seed": 20260624,
                "time": {"epoch": "synthetic_day_0", "time_horizon_s": 120, "time_bin_s": 60},
                "geography": {
                    "mode": "named_gateway_service_regions",
                    "demand_weight_normalization": "require_sum_one",
                    "regions": [
                        {
                            "region_id": "na_east",
                            "region_name": "North America East",
                            "lat_deg": 39.0,
                            "lon_deg": -77.0,
                            "timezone_utc_offset_h": -5,
                            "demand_weight": 1.0,
                            "weight_source": "unit_test",
                        }
                    ],
                },
                "gateways": {"access_mode": "direct_user_sgl", "mapping_policy": "region_center"},
                "arrival": {"process": "nhpp_synthetic", "lambda_global": 0.4, "diurnal_profile": "flat"},
                "tokens": {
                    "model_family": "llama_70b_like",
                    "task_mix": [
                        {
                            "task_type": "conversation",
                            "weight": 1.0,
                            "prompt_distribution": {
                                "type": "quantile_buckets",
                                "p50": 128,
                                "p95": 512,
                                "p99": 768,
                                "max": 1024,
                            },
                            "output_distribution": {
                                "type": "quantile_buckets",
                                "p50": 64,
                                "p95": 256,
                                "p99": 384,
                                "max": 512,
                            },
                        }
                    ],
                },
                "slo": {
                    "mix": "balanced",
                    "classes": [
                        {
                            "slo_class": "interactive_5s",
                            "weight": 1.0,
                            "ttft_deadline_s": 5,
                            "e2e_deadline_s": 30,
                        }
                    ],
                },
                "sessions": {
                    "session_mode": "open_loop_turn_arrivals",
                    "session_ratio": 0.0,
                    "context_window_tokens": 8192,
                    "turn_count_distribution": "single_turn_only",
                },
                "validation": {"normalize_region_weights": False, "require_unique_request_id": True},
                "output": {"bundle_dir": str(tmp / "bundle"), "overwrite": False},
                "provenance": {"sources": ["unit test workload description"]},
            }
            description_path = tmp / "workload_description.json"
            description_path.write_text(json.dumps(workload_description, indent=2), encoding="utf-8")

            compiled = compile_workload_description(description_path)

            manifest = json.loads((ROOT / "tests/fixtures/manifests/smoke_single_design.json").read_text(encoding="utf-8"))
            manifest["costing"]["single_satellite_case_path"] = str(
                ROOT / "examples/single_satellite/h200_gpu_derived.json"
            )
            manifest["workload"] = {"bundle_dir": compiled.bundle_dir}
            manifest["dse"] = {"enabled": True, "sweep": {"plane_count": [2]}}
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            original_materializer = dse_module.materialize_workload_bundle

            def fail_if_dse_materializes(*args, **kwargs):
                raise AssertionError("embedded DSE attempted to rematerialize workload")

            dse_module.materialize_workload_bundle = fail_if_dse_materializes
            try:
                result = run_experiment(load_experiment_spec(manifest_path))
            finally:
                dse_module.materialize_workload_bundle = original_materializer

            payload = result.to_dict()
            row = payload["dse"]["rows"][0]
            runtime = row["performance"]["details"]["runtime"]

            self.assertEqual(payload["dse"]["design_count"], 1)
            self.assertEqual(runtime["workload"]["source_mode"], "synthetic_generation")
            self.assertGreater(runtime["workload"]["request_count"], 0)
            self.assertTrue((tmp / "bundle" / "requests.jsonl").exists())
            self.assertTrue(
                any("DSE did not rematerialize workload" in warning for warning in payload["dse"]["warnings"])
            )

    def test_experiment_cli_writes_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "smoke_single_design.json"
            env = dict(os.environ)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orbitserve.experiments.cli",
                    str(ROOT / "tests/fixtures/manifests/smoke_single_design.json"),
                    "--out",
                    str(out_path),
                ],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], "orbitserve.experiment_summary.v1")
            self.assertEqual(data["output_mode"], "summary")
            self.assertIn("artifact_manifest", data)
            for key in ("cost", "performance", "slo", "dse", "module_status", "warnings"):
                self.assertIn(key, data)
            self.assertTrue(data["cost"]["evaluated"])
            self.assertTrue(data["performance"]["evaluated"])
            self.assertIsNone(data["performance"]["request_throughput_per_s"])

    def test_experiment_cli_full_mode_preserves_monolithic_json_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "smoke_single_design_full.json"
            env = dict(os.environ)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orbitserve.experiments.cli",
                    str(ROOT / "tests/fixtures/manifests/smoke_single_design.json"),
                    "--out",
                    str(out_path),
                    "--output-mode",
                    "full",
                ],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertNotEqual(data.get("schema_version"), "orbitserve.experiment_summary.v1")
            self.assertNotIn("artifact_manifest", data)
            for key in ("cost", "performance", "slo", "dse", "module_status", "warnings"):
                self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
