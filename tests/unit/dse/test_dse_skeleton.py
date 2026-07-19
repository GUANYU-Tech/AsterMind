import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.dse.search import run_dse
from orbitserve.dse.search import DSESpec
from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.architecture.hardware import SatelliteHardwareProfile
from orbitserve.workload import compile_workload_description


ROOT = Path(__file__).resolve().parents[3]


class DSESkeletonTests(unittest.TestCase):
    def test_dse_skeleton_enumerates_design_rows(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_dse.json")
        result = run_dse(spec)
        payload = result.to_dict()
        self.assertTrue(payload["evaluated"])
        self.assertEqual(payload["design_count"], 4)
        self.assertEqual(len(payload["rows"]), 4)
        self.assertFalse(payload["frontier"]["evaluated"])
        self.assertFalse(payload["metrics"]["pareto_frontier_evaluated"])
        for row in payload["rows"]:
            self.assertTrue(row["cost"]["evaluated"])
            self.assertTrue(row["performance"]["evaluated"])
            self.assertIsNone(row["performance"]["request_throughput_per_s"])
            self.assertGreater(row["performance"]["prefill_tokens_per_s"], 0)
            self.assertGreater(row["performance"]["decode_tokens_per_s"], 0)
            self.assertFalse(row["slo"]["evaluated"])
            self.assertIsNone(row["slo"]["served_ratio"])

    def test_experiment_runner_embeds_dse_result(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_dse.json")
        result = run_experiment(spec)
        payload = result.to_dict()
        self.assertEqual(payload["module_status"]["dse"], "skeleton")
        self.assertEqual(payload["dse"]["design_count"], 4)
        json.dumps(payload, ensure_ascii=False)


    def test_dse_rows_propagate_serving_throughput_warnings(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_hardware_interface.json")
        spec = replace(
            spec,
            serving=replace(
                spec.serving,
                parallelism=replace(
                    spec.serving.parallelism,
                    strategy="tensor_expert_parallel",
                    expert_parallel_degree=2,
                    max_satellites=2,
                ),
            ),
            dse=DSESpec(enabled=True, sweep={"plane_count": (2,)}),
        )
        result = run_dse(spec)
        payload = result.to_dict()
        self.assertTrue(
            any(
                "dense/non-MoE model ignores expert_parallel_degree" in warning
                for warning in payload["rows"][0]["warnings"]
            )
        )




    def test_direct_dse_loads_prebuilt_canonical_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            workload_description = {
                "schema_version": "orbitserve.workload_description",
                "workload_id": "direct_dse_workload_smoke",
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
            manifest["dse"] = {"enabled": True, "sweep": {"plane_count": [2]}}
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            result = run_dse(load_experiment_spec(manifest_path))
            payload = result.to_dict()
            runtime = payload["rows"][0]["performance"]["details"]["runtime"]
            self.assertNotIn(
                "token_distribution",
                payload["rows"][0]["performance"]["details"]["provenance"],
            )
            self.assertEqual(runtime["workload"]["source_mode"], "synthetic_generation")
            self.assertGreater(runtime["workload"]["request_count"], 0)
            self.assertGreater(runtime["provenance"]["workload_bundle"]["request_count"], 0)
            self.assertTrue((tmp / "bundle" / "requests.jsonl").exists())
            self.assertTrue(any("workload_generation status: loaded" in warning for warning in payload["warnings"]))


if __name__ == "__main__":
    unittest.main()
