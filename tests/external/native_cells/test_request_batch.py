"""Imported and generated input reaches the real native Cell execution path."""

import unittest

from tests.external.native_cells import test_native_cells as fixtures


class NativeRequestBatchTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.NativeCellIntegrationTests.setUpClass.__func__)
    config = fixtures.NativeCellIntegrationTests.config
    run_scene = fixtures.NativeCellIntegrationTests.run_scene

    def test_batch_workload_preserves_requests_through_native_execution(self):
        from simulator.config import SimulationConfig
        from simulator.request_batch import prepare_batch
        from simulator.replay import audit_checks

        value = self.config("dynamic_tp2_pp2.json").model_dump(mode="json")
        value["communication_mode"] = "protobridge_cell"
        generated = prepare_batch(
            dict(
                experiment=value,
                kind="generate",
                mode="replace",
                options=dict(count=3, window_s=0.1, input_tokens=16, output_tokens=1),
            )
        )
        value["requests"] = generated["requests"]
        device = value["requests"][0]["user_device"]
        imported = prepare_batch(
            dict(
                experiment=value,
                mode="append",
                format="csv",
                content=f"id,arrival_s,user_device,input_tokens,output_tokens\nfrom-file,0.2,{device},16,1\n",
            )
        )
        value["requests"] = imported["requests"]
        result = self.run_scene(SimulationConfig.model_validate(value))
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["requests"]), 4)
        self.assertIn("from-file", result["requests"])
        self.assertEqual(result["audit"]["requests"], 4)
        self.assertTrue(all(row["state"] == "pass" for row in audit_checks(result)))
