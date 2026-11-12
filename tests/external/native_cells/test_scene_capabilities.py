"""Actual model, native Scheduler, GPU graph and network for scene choices."""

import unittest
from tests.external.native_cells import test_native_cells as fixtures


class NativeSceneCapabilitiesTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.NativeCellIntegrationTests.setUpClass.__func__)
    config = fixtures.NativeCellIntegrationTests.config
    run_scene = fixtures.NativeCellIntegrationTests.run_scene

    def run_choice(
        self, hardware, model, tp=1, pp=1, *, experts=False, scope="protobridge_cell"
    ):
        from simulator.config import SimulationConfig
        from simulator.replay import audit_checks

        value = self.config("dynamic_tp2_pp2.json").model_dump(mode="json")
        cell = value["cells"][0]
        cell.update(
            hardware=f"configs/hardware/{hardware}.yaml",
            hardware_id=hardware,
            model=f"configs/model/{model}.json",
            parallel=dict(tp=tp, pp=pp),
            native_expert_parallel=experts,
            expert_routing_seed=0 if experts else None,
        )
        cell["placements"] = cell["placements"][: tp * pp]
        cell["output_device"] = cell["placements"][tp * (pp - 1)]["device"]
        value["communication_mode"] = scope
        result = self.run_scene(SimulationConfig.model_validate(value))
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["audit"]["passed"])
        self.assertTrue(all(row["state"] == "pass" for row in audit_checks(result)))
        self.assertEqual(result["provenance"]["communication_mode"], scope)
        return result

    def test_h200_dense_inference(self):
        self.run_choice("nvidia_h200", "qwen3-8b")

    def test_ascend_dense_inference(self):
        self.run_choice("ascend_910b", "qwen3-4b")

    def test_metax_dense_inference(self):
        self.run_choice("metax_c500", "qwen3-4b")

    def test_llama_dense_inference(self):
        self.run_choice("nvidia_a800", "llama-3.1-8b-instruct")

    def test_qwen_32b_dense_inference(self):
        self.run_choice("nvidia_a800", "qwen3-32b", tp=2)

    def test_pipeline_two_with_real_flowsim(self):
        self.run_choice("nvidia_a800", "qwen3-8b", pp=2, scope="flowsim")

    def test_pipeline_four_with_cell_protobridge(self):
        self.run_choice("nvidia_a800", "qwen3-8b", pp=4)

    def test_moe_native_expert_parallel(self):
        self.run_choice("nvidia_a800", "qwen3-30b-a3b", tp=2, experts=True)
