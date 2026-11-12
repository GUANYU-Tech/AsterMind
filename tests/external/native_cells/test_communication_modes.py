"""Real Scheduler, native kernels and ProtoBridge on the shared FlowSim clock."""

import json
import unittest
from tests.external.native_cells import test_native_cells as fixtures


class NativeCommunicationModesTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.NativeCellIntegrationTests.setUpClass.__func__)
    config = fixtures.NativeCellIntegrationTests.config
    run_scene = fixtures.NativeCellIntegrationTests.run_scene

    def test_three_modes_preserve_native_dependencies(self):
        results = {}
        for mode in ("flowsim", "protobridge_satellite", "protobridge_cell"):
            with self.subTest(mode=mode):
                result = self.run_scene(
                    self.config("dynamic_tp2_pp2.json", communication_mode=mode)
                )
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["audit"]["passed"])
                self.assertTrue(all(c["steps"] for c in result["cells"].values()))
                self.assertEqual(result["provenance"]["communication_mode"], mode)
                results[mode] = result
                from simulator.replay import audit_checks

                self.assertTrue(
                    all(not group["violations"] for group in audit_checks(result)), mode
                )
        self.assertFalse(results["flowsim"]["network"]["surrogate_operations"])
        for mode in ("protobridge_satellite", "protobridge_cell"):
            self.assertTrue(results[mode]["network"]["surrogate_operations"])
        counts = {
            m: sum(e["kind"] == "transmit_start" for e in r["network"]["event_log"])
            for m, r in results.items()
        }
        self.assertGreater(counts["flowsim"], counts["protobridge_satellite"])
        self.assertGreater(counts["protobridge_satellite"], counts["protobridge_cell"])
        (self.reports / "communication-mode-comparison.json").write_text(
            json.dumps(
                {
                    mode: dict(
                        **result["measurement"],
                        flows=counts[mode],
                        surrogate_operations=len(
                            result["network"]["surrogate_operations"]
                        ),
                        audit=result["audit"]["passed"],
                    )
                    for mode, result in results.items()
                },
                indent=2,
            )
        )

    def test_scene_builder_preserves_gpu_and_physical_resources(self):
        from simulator.scene import build_scene, validate_scene_config
        from pydantic import ValidationError

        config = build_scene(self.dataset, dict(tp=2, pp=1, gpus_per_satellite=2))
        self.assertEqual(config.communication_mode, "protobridge_cell")
        ground = next(
            d["id"]
            for d in config.physical["devices"]
            if d["satellite"] in config.ground_owners
        )
        data = config.model_dump(mode="json")
        data["requests"] = [
            dict(id="from-editor", user_device=ground, input_tokens=16, output_tokens=2)
        ]
        checked = validate_scene_config(data)
        self.assertEqual(
            sum(len(c.placements) for c in checked.cells), len(checked.cells) * 2
        )
        with self.assertRaises(ValidationError):
            build_scene(self.dataset, dict(gpus_per_satellite=100))
        data["requests"][0]["user_device"] = "missing"
        with self.assertRaisesRegex(ValueError, "endpoint"):
            validate_scene_config(data)

    def test_cell_surrogate_keeps_migration_on_flowsim(self):
        result = self.run_scene(
            self.config(
                "dynamic_migration_tp2.json", communication_mode="protobridge_cell"
            )
        )
        self.assertTrue(result["audit"]["passed"])
        migration = result["migrations"]["copy0"]
        self.assertEqual(migration["state"], "completed")
        for partition in migration["partitions"]:
            event = next(
                e for e in result["transfers"] if e["id"] == partition["transfer_id"]
            )
            self.assertTrue(event["path"])
            self.assertNotIn(
                event["id"],
                {
                    r["correlation_id"]
                    for r in result["network"]["surrogate_operations"]
                },
            )
