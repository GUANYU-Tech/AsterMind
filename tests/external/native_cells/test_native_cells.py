"""Explicit local suite: requires real pinned M-L, vLLM and native transports.

Run separately from dependency-light CI; missing native dependencies fail here.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from orbitserve.experiments.network import run_network_experiment

from simulator.audit import audit
from simulator.bootstrap import DEFAULT_ML
from simulator.config import SimulationConfig
from simulator.engine import Simulation

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "simulator/examples"


class NativeCellIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.midlayer = Path(os.environ.get("ORBITSERVE_MIDLAYER_ROOT", DEFAULT_ML))
        if not (cls.midlayer / "src/serving/execution.py").is_file():
            raise RuntimeError(
                "explicit native suite requires the complete external M-L snapshot"
            )
        cls.temporary = tempfile.TemporaryDirectory(prefix="orbitserve-native-test-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.dataset = Path(cls.temporary.name) / "physical"
        run_network_experiment(
            json.loads((EXAMPLES / "dynamic_physics.json").read_text()),
            cls.dataset,
            _workers=1,
            export_mode="core",
        )
        cls.reports = Path(
            os.environ.get(
                "ORBITSERVE_NATIVE_REPORT_DIR", ROOT / "results/native-cell-tests"
            )
        )
        cls.reports.mkdir(parents=True, exist_ok=True)

    def config(self, name, **updates):
        value = SimulationConfig.load(EXAMPLES / name).model_dump(mode="json")
        if value.get("physical_network"):
            value.update(physical_network=None, physical_dataset=str(self.dataset))
        value.update(updates)
        return SimulationConfig.model_validate(value)

    def run_scene(self, config):
        with Simulation(config, midlayer_root=self.midlayer) as simulation:
            result = simulation.run()
        from simulator.__main__ import json_default

        result["audit"] = (
            audit(result) if result["status"] == "completed" else {"passed": False}
        )
        (self.reports / (self._testMethodName + ".json")).write_text(
            json.dumps(result, indent=2, default=json_default), encoding="utf8"
        )
        for cell in result["cells"].values():
            runtime = cell["scheduler_runtime"]
            self.assertEqual(runtime["version"], "0.31.0")
            self.assertEqual(runtime["package_python_file_count"], 2771)
            self.assertEqual(
                runtime["package_python_manifest_sha256"],
                "1ddbe0ebcca48d6ed773a0ab92d55bbaaeffb883ddad2e97ac08d6fec5d446d5",
            )
        return result

    def test_cross_satellite_tp2_pp2_waits_for_recovery(self):
        result = self.run_scene(self.config("dynamic_tp2_pp2.json"))
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["audit"]["passed"])
        self.assertGreater(result["audit"]["communications"], 0)
        self.assertTrue(
            any(e["operation"] == "pp_send_recv" for e in result["transfers"])
        )
        events = result["network"]["event_log"]
        self.assertTrue(any(e["kind"] == "route_wait" for e in events))
        self.assertTrue(any(e["kind"] == "route_recovery" for e in events))
        self.assertTrue(
            any(
                e["kind"] == "bandwidth_update" and e.get("link_capacity_bps") == 40e6
                for e in events
            )
        )
        for transfer in result["transfers"]:
            if transfer["source"].startswith("gpu") and transfer["target"].startswith(
                "gpu"
            ):
                self.assertNotIn("user", transfer["path"])

    def check_migration(self, backend):
        result = self.run_scene(
            self.config("dynamic_migration_tp2.json", backend=backend, communication_mode="flowsim")
        )
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["audit"]["passed"])
        record = result["migrations"]["copy0"]
        self.assertEqual(record["bytes"], 4718592)
        self.assertEqual(len(record["partitions"]), 2)
        self.assertGreater(record["network_complete_time_s"], 0.505)
        self.assertGreater(
            record["completed_time_s"], record["network_complete_time_s"]
        )
        self.assertEqual(record["source_release"]["removed_blocks"], 2)
        source = result["cells"]["cellA"]["final_load"]["storage"]
        target = result["cells"]["cellB"]["final_load"]["storage"]
        self.assertEqual(source["ready_cached_blocks"], 0)
        self.assertEqual(target["migration_reserved_blocks"], 0)
        self.assertTrue(
            any(r.get("effective_prefix_hit_tokens") == 32 for r in target["records"])
        )
        for part in record["partitions"]:
            transfer = next(
                e for e in result["transfers"] if e["id"] == part["transfer_id"]
            )
            self.assertIn("isl0", transfer["path"])
            self.assertNotIn("user", transfer["path"])
        return result

    def test_native_tp2_move_and_prefix_reuse_flowsim(self):
        self.check_migration("flowsim")

    def test_native_tp2_move_and_prefix_reuse_ns3(self):
        result = self.check_migration("ns3")
        self.assertGreater(
            result["network"]["transport"]["native_counters"]["packets_started"], 0
        )

    def test_native_migration_failure_rolls_back_allocator(self):
        config = self.config("dynamic_migration_tp2.json")
        value = config.model_dump(mode="json")
        value["requests"] = value["requests"][:1]
        value["runtime_network"]["failure_policy"] = "fail"
        result = self.run_scene(SimulationConfig.model_validate(value))
        self.assertEqual(result["status"], "completed_with_failures")
        self.assertEqual(result["migrations"]["copy0"]["state"], "failed")
        source = result["cells"]["cellA"]["final_load"]["storage"]
        target = result["cells"]["cellB"]["final_load"]["storage"]
        self.assertEqual(source["ready_cached_blocks"], 2)
        self.assertEqual(target["migration_reserved_blocks"], 0)
        self.assertEqual(target["ready_cached_blocks"], 0)

    def test_heterogeneous_cells_and_hardware_derived_hbm_capacity(self):
        import hashlib

        from simulator.bootstrap import activate

        activate(self.midlayer)
        from src.serving.workload.model_config import load_model_config

        value = self.config("heterogeneous_cells.json").model_dump(mode="json")
        value["cells"][0].update(
            max_model_len=8192,
            max_batched_tokens=8192,
            max_requests=256,
            memory_policy_file="configs/runtime_memory_policy/nvidia_a800/qwen3-4b.json",
        )
        # The snapshot ships a policy bound to another hardware-file revision.
        # Reject it; do not relabel its measured reserves as current evidence.
        with self.assertRaisesRegex(ValueError, "hardware fingerprint"):
            Simulation(
                SimulationConfig.model_validate(value), midlayer_root=self.midlayer
            )
        policy = json.loads(
            (self.midlayer / value["cells"][0]["memory_policy_file"]).read_text()
        )
        pool = policy["scheduler_pools"][0]
        pool["binding"]["hardware_config_sha256"] = hashlib.sha256(
            (self.midlayer / value["cells"][0]["hardware"]).read_bytes()
        ).hexdigest()
        pool["binding"]["model_config_sha256"] = load_model_config(
            self.midlayer / value["cells"][0]["model"]
        ).config_sha256
        evidence = {
            "source_kind": "runtime_configuration",
            "source_sha256": None,
            "source_reference": "Synthetic accounting test: 80% HBM, 1 MiB per reserve category; no hardware calibration",
        }
        pool["gpu_memory_utilization"]["provenance"] = evidence
        for reserve in pool["non_kv_reserves"]:
            reserve.update(bytes=1024**2, provenance=evidence)
        path = Path(self.temporary.name) / "synthetic-memory-policy.json"
        path.write_text(json.dumps(policy), encoding="utf8")
        value["cells"][0]["memory_policy_file"] = str(path)
        result = self.run_scene(SimulationConfig.model_validate(value))
        self.assertEqual(result["status"], "completed")
        capacity = result["cells"]["cellA"]["scheduler_runtime"]["capacity"]
        self.assertEqual(capacity["source_mode"], "analytical_current_vllm_allocator")
        self.assertLess(capacity["available_kv_bytes"], capacity["physical_hbm_bytes"])
        self.assertGreater(capacity["num_kv_blocks"], 64)
        self.assertGreater(capacity["weight_resident_bytes"], 0)
        self.assertEqual(capacity["non_kv_reserve_bytes"], 6 * 1024**2)
        self.assertEqual(
            capacity["available_kv_bytes"],
            capacity["utilized_hbm_bytes"]
            - capacity["weight_resident_bytes"]
            - capacity["non_kv_reserve_bytes"],
        )

    def test_native_moe_tp2_ep2(self):
        result = self.run_scene(self.config("native_moe_tp2_ep2.json"))
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["audit"]["passed"])
        self.assertGreater(result["audit"]["communications"], 0)


if __name__ == "__main__":
    unittest.main()
