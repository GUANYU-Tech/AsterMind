import unittest
from dataclasses import replace
from pathlib import Path

from orbitserve.architecture.hardware import hardware_from_single_satellite_case
from orbitserve.serving import (
    LLMModelProfile,
    ServingReferenceShape,
    ServingSpec,
    estimate_serving_capacity,
)


ROOT = Path(__file__).resolve().parents[3]


def h200_hardware():
    return hardware_from_single_satellite_case(
        ROOT / "examples/single_satellite/h200_gpu_derived.json"
    )


class ServingCapacityTests(unittest.TestCase):
    def test_model_presets_resolve_weight_and_kv_formulas(self):
        model = LLMModelProfile.from_dict({"preset": "llama3-8b", "precision": "bf16"})
        self.assertAlmostEqual(model.weight_memory_gb, 16.0)
        self.assertEqual(model.kv_cache_bytes_per_token, 131072.0)
        self.assertEqual(model.architecture_source, "preset_llama3")

    def test_moe_preset_resolves_active_and_routed_parameters(self):
        model = LLMModelProfile.from_dict({"preset": "deepseek-moe-671b"})
        self.assertTrue(model.is_moe)
        self.assertAlmostEqual(model.active_parameters_b, 37.0)
        self.assertGreater(model.routed_expert_parameters_b, model.active_routed_parameters_b)

    def test_reference_shape_is_serving_configuration_not_workload_input(self):
        shape = ServingReferenceShape.from_dict(
            {"prompt_tokens": 1024, "decode_context_tokens": 1536, "kv_context_tokens": 2048}
        )
        spec = ServingSpec.from_dict({"model": {"preset": "llama3-8b"}, "reference_shape": shape.to_dict()})
        result = estimate_serving_capacity(spec, hardware=h200_hardware())
        self.assertEqual(result.provenance["reference_shape"]["prompt_tokens"], 1024)
        self.assertNotIn("token_distribution", result.provenance)

    def test_single_deployment_reports_static_prefill_decode_capacity(self):
        result = estimate_serving_capacity(
            ServingSpec.from_dict({"model": {"preset": "llama3-8b"}}),
            hardware=h200_hardware(),
        )
        payload = result.to_dict()
        self.assertTrue(result.evaluated)
        self.assertTrue(result.feasible)
        self.assertEqual(result.satellites_per_deployment, 1)
        self.assertEqual(result.prefill_satellites_per_group, 1)
        self.assertEqual(result.decode_satellites_per_group, 1)
        self.assertGreater(result.prefill_tokens_per_s or 0, 0)
        self.assertGreater(result.decode_tokens_per_s or 0, 0)
        for forbidden in (
            "request_throughput_per_s",
            "ttft_s",
            "e2e_s",
            "constellation_satellite_count",
            "deployment_group_count",
        ):
            self.assertNotIn(forbidden, payload)

    def test_explicit_hardware_is_the_capacity_authority(self):
        one = replace(h200_hardware(), accelerator_count=1)
        three = replace(h200_hardware(), accelerator_count=3)
        spec = ServingSpec.from_dict({"model": {"preset": "llama3-8b"}})
        one_result = estimate_serving_capacity(spec, hardware=one)
        three_result = estimate_serving_capacity(spec, hardware=three)
        self.assertGreater(three_result.prefill_tokens_per_s, one_result.prefill_tokens_per_s)

    def test_tensor_parallel_layout_is_reported_per_deployment(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "qwen2.5-72b"},
                "parallelism": {"strategy": "tensor_parallel", "tp": 4, "max_satellites": 4},
            }
        )
        result = estimate_serving_capacity(spec, hardware=h200_hardware())
        self.assertEqual(result.satellites_per_deployment, 4)
        self.assertEqual(result.parallelism["tp"], 4)

    def test_pipeline_parallel_includes_static_communication_cost(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "llama3-8b"},
                "parallelism": {"strategy": "pipeline_parallel", "pp": 2, "max_satellites": 2},
            }
        )
        result = estimate_serving_capacity(spec, hardware=h200_hardware())
        self.assertEqual(result.satellites_per_deployment, 2)
        self.assertGreaterEqual(result.pp_communication_time_s or 0, 0)

    def test_pd_output_describes_one_prefill_and_one_decode_group(self):
        spec = ServingSpec.from_dict(
            {"model": {"preset": "llama3-8b"}, "pd_mode": "disaggregated"}
        )
        result = estimate_serving_capacity(spec, hardware=h200_hardware())
        self.assertEqual(result.prefill_satellites_per_group, result.satellites_per_deployment)
        self.assertEqual(result.decode_satellites_per_group, result.satellites_per_deployment)
        self.assertNotIn("prefill_deployment_group_count", result.to_dict())

    def test_missing_hardware_fails_closed(self):
        result = estimate_serving_capacity(ServingSpec.from_dict({"model": {"preset": "llama3-8b"}}))
        self.assertFalse(result.evaluated)
        self.assertFalse(result.feasible)
        self.assertIn("not configured", result.warnings[0])


if __name__ == "__main__":
    unittest.main()
