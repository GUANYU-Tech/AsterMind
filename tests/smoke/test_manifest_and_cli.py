import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from orbitserve.architecture import HardwareSpec
from orbitserve.experiments import ExperimentSpec, run_experiment
from orbitserve.workload import materialize_workload_bundle
from tests.system_case import physical_case, ROOT


class ManifestAndCliTests(unittest.TestCase):
    def test_manifest_json_round_trip_and_workload_input_contract(self):
        spec = physical_case()
        restored = ExperimentSpec.from_dict(
            json.loads(json.dumps(spec.to_dict())), base_dir=spec.base_dir
        )
        self.assertEqual(restored.to_dict(), spec.to_dict())

    def test_top_level_architecture_drives_serving_and_cost(self):
        spec = physical_case()
        with tempfile.TemporaryDirectory() as folder:
            first = run_experiment(spec, output_dir=folder, _workers=1)
            hardware = HardwareSpec.from_dict(
                {
                    **spec.hardware.to_dict(),
                    "compute_payload": {
                        "accelerator_count": 8,
                        "purchase_cost_per_unit_usd": 50000,
                    },
                }
            )
            second = run_experiment(
                replace(spec, hardware=hardware), output_dir=folder, _workers=1
            )
            self.assertEqual(
                second.architecture["compute"]["hardware_profile"]["accelerator_count"],
                8,
            )
            self.assertGreater(
                second.cost["total_cost_usd"], first.cost["total_cost_usd"]
            )
            self.assertGreater(
                second.serving["prefill_rate_table"][0]["prefill_tokens_per_s"],
                first.serving["prefill_rate_table"][0]["prefill_tokens_per_s"],
            )

    def test_workload_is_materialized_once_and_shared_with_dse(self):
        spec = replace(physical_case(), dse={"enabled": True})
        with (
            tempfile.TemporaryDirectory() as folder,
            patch(
                "orbitserve.experiments.runner.materialize_workload_bundle",
                wraps=materialize_workload_bundle,
            ) as loader,
        ):
            result = run_experiment(spec, output_dir=folder, _workers=1)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual(result.runtime.request_summary["total_request_count"], 3)
            self.assertEqual(
                result.dse["rows"][0]["runtime"]["request_summary"][
                    "total_request_count"
                ],
                3,
            )

    def test_experiment_cli_summary_and_full_outputs(self):
        manifest = ROOT / "tests/fixtures/manifests/coupled_physical.json"
        for mode in ("summary", "full"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / "result.json"
                process = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "orbitserve.experiments.cli",
                        str(manifest),
                        "--out",
                        str(out),
                        "--output-mode",
                        mode,
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    env={**os.environ, "CONDA_DEFAULT_ENV": "portable-runtime"},
                )
                self.assertEqual(process.returncode, 0, process.stderr)
                payload = json.loads(out.read_text())
                self.assertEqual(
                    payload["runtime"]["request_summary"]["completed_request_count"], 3
                )
                self.assertNotIn("slo", payload["runtime"])
                if mode == "summary":
                    self.assertIn("artifact_manifest", payload)
                else:
                    self.assertEqual(len(payload["runtime"]["request_records"]), 3)

    def test_moe_fixture_names_match_the_deployment_actually_executed(self):
        from orbitserve.experiments import load_experiment_spec

        for name in ("smoke_moe_ep_parallel", "smoke_tp_pp_ep_parallel"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                spec = load_experiment_spec(
                    ROOT / "tests/fixtures/manifests" / (name + ".json")
                )
                self.assertEqual(spec.serving.model.name, "deepseek-v3")
                self.assertGreater(spec.serving.parallelism.expert_parallel_degree, 1)
                result = run_experiment(spec, output_dir=folder, _workers=1)
                self.assertEqual(
                    result.runtime.request_summary["completed_request_count"], 3
                )
