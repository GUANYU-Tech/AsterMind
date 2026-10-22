"""Use all real module implementations, not mock module return values."""

import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from orbitserve.architecture import HardwareSpec
from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.serving import ServingSpec

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "tests/fixtures/manifests/coupled_physical.json"


class CoupledPhysicalSystemTests(unittest.TestCase):
    def test_directional_rf_budget_cannot_bypass_hardware_terminal_limit(self):
        spec = load_experiment_spec(MANIFEST)
        physical = deepcopy(spec.physical_network)
        physical['network']['link_profiles']['sgl']['uplink_profile'] = 'uplink'
        physical['network']['link_profiles']['uplink'] = {'bandwidth_bps': 2e9, 'setup_time_s': 0.}
        with tempfile.TemporaryDirectory() as folder, self.assertRaisesRegex(ValueError, 'rate exceeds Architecture'):
            run_experiment(replace(spec, physical_network=physical), output_dir=folder, _workers=1)

    def test_workload_architecture_serving_network_runtime_end_to_end(self):
        spec = load_experiment_spec(MANIFEST)
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(spec, output_dir=folder, _workers=1)
            self.assertTrue(
                all(v == "evaluated" for v in result.module_status.values())
            )
            self.assertEqual(result.runtime.request_summary["total_request_count"], 3)
            self.assertEqual(
                result.runtime.request_summary["completed_request_count"], 3
            )
            self.assertEqual(
                result.runtime.provenance["runtime_engine"],
                "chronological_discrete_event_runtime",
            )
            arrivals = [
                e["time_s"]
                for e in result.runtime.state_summary["event_log"]
                if e["kind"] == "request_arrival"
            ]
            self.assertEqual(arrivals, [7.25, 7.3, 30.0])
            self.assertTrue(
                all(
                    r["routing"]["selected_path"]
                    for r in result.runtime.request_records
                )
            )
            self.assertGreater(
                result.cost["total_cost_usd"], result.cost["compute"]["total_cost_usd"]
            )
            self.assertTrue((Path(folder) / "runtime_events.jsonl.gz").exists())
            import gzip

            with gzip.open(Path(folder) / "runtime_events.jsonl.gz", "rt") as stream:
                events = [json.loads(line) for line in stream]
            times = [event["time_s"] for event in events]
            self.assertEqual(times, sorted(times))
            transmissions = [e for e in events if e["kind"] == "transmit_start"]
            queued = [e for e in events if e["kind"] == "transfer_queued"]
            completed = [e for e in events if e["kind"] == "transmit_complete"]
            self.assertEqual(len(queued), len(transmissions))
            self.assertEqual(len(transmissions), len(completed))
            self.assertEqual(
                {e["transfer_id"] for e in queued},
                {e["transfer_id"] for e in transmissions},
            )
            self.assertTrue(
                all(e["link_id"] and e["source_node_id"] for e in transmissions)
            )
            self.assertFalse((Path(folder) / "runtime_playback").exists())
            self.assertFalse((Path(folder) / "runtime_viewer.json").exists())
            self.assertEqual(result.network["checks"]["snapshot_count"], 120)

            self.assertTrue(
                (Path(folder) / "network/frames/frame-000002.json.gz").exists()
            )
            saved = json.loads((Path(folder) / "experiment_result.json").read_text())
            self.assertEqual(
                saved["runtime"]["request_summary"]["total_request_count"], 3
            )

    def test_hardware_terminal_limit_is_enforced_before_orbit_execution(self):
        spec = load_experiment_spec(MANIFEST)
        hardware = HardwareSpec.from_dict(
            {**spec.hardware.to_dict(), "communication": {"isl_terminal_count": 0}}
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(ValueError, "terminals exceed Architecture"),
        ):
            run_experiment(
                replace(spec, hardware=hardware), output_dir=folder, _workers=1
            )

    def test_serving_cell_member_count_mismatch_fails(self):
        spec = load_experiment_spec(MANIFEST)
        serving = ServingSpec.from_dict(
            {"formation": {"member_count": 1}, "parallelism": {"tp": 1}}
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(ValueError, "member_count"),
        ):
            run_experiment(
                replace(spec, serving=serving), output_dir=folder, _workers=1
            )

    def test_dse_uses_the_same_physical_pipeline(self):
        spec = load_experiment_spec(MANIFEST)
        dse = {
            "enabled": True,
            "runtime_matrix": {
                "enabled": True,
                "include_current_policy": False,
                "candidate_schedulers": [spec.runtime_strategy.to_dict()],
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(
                replace(spec, dse=dse), output_dir=folder, _workers=1
            )
            self.assertTrue(result.dse["evaluated"])
            self.assertEqual(result.dse["row_count"], 1)
            self.assertEqual(
                result.dse["rows"][0]["runtime"]["request_summary"][
                    "completed_request_count"
                ],
                3,
            )

    def test_moe_serving_profiles_execute_on_the_same_physical_network(self):
        import shutil
        from orbitserve.workload import WorkloadBundleRef
        from orbitserve.workload.contract import bundle_hash

        for model in ("deepseek-v3", "kimi-k3"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as folder:
                spec = load_experiment_spec(MANIFEST)
                bundle = Path(folder) / "workload"
                shutil.copytree(
                    ROOT / "tests/fixtures/workloads/coupled_physical", bundle
                )
                rows = [
                    json.loads(line)
                    for line in (bundle / "session_requests.jsonl")
                    .read_text()
                    .splitlines()
                ]
                for row in rows:
                    row["model_family"] = model
                (bundle / "session_requests.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows)
                )
                manifest = json.loads((bundle / "workload_manifest.json").read_text())
                manifest["bundle_sha256"] = bundle_hash(manifest["workload_id"], bundle)
                (bundle / "workload_manifest.json").write_text(json.dumps(manifest))
                serving = ServingSpec.from_dict(
                    {
                        **spec.serving.to_dict(),
                        "model": {"name": model},
                        "parallelism": {"tp": 2, "pp": 2, "ep": 2},
                    }
                )
                # Sufficient physical memory for the frozen MoE model weights.
                hardware = HardwareSpec.from_dict(
                    {
                        **spec.hardware.to_dict(),
                        "compute_payload": {"accelerator_count": 16},
                    }
                )
                result = run_experiment(
                    replace(
                        spec,
                        serving=serving,
                        hardware=hardware,
                        workload=WorkloadBundleRef(bundle_dir=str(bundle)),
                    ),
                    output_dir=Path(folder) / "result",
                    _workers=1,
                )
                self.assertEqual(
                    result.runtime.request_summary["completed_request_count"], 3
                )
                self.assertEqual(
                    result.runtime.request_records[0]["provenance"]["trace_model"],
                    "sgp4-wgs72-teme-astropy-itrs-wgs84",
                )
                self.assertEqual(
                    result.runtime.request_records[0]["timing"][
                        "response_streaming_model"
                    ],
                    "first_token_then_remaining_output",
                )

    def test_workload_model_mismatch_is_rejected(self):
        spec = load_experiment_spec(MANIFEST)
        serving = ServingSpec.from_dict(
            {**spec.serving.to_dict(), "model": {"name": "deepseek-v3"}}
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(ValueError, "model_family"),
        ):
            run_experiment(
                replace(spec, serving=serving), output_dir=folder, _workers=1
            )
