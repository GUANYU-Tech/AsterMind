import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.architecture.hardware import hardware_from_single_satellite_case
from orbitserve.serving.capacity import LLMModelProfile, ServingParallelism, ServingSpec, estimate_serving_throughput
from orbitserve.workload import TokenDistribution


ROOT = Path(__file__).resolve().parents[3]


def h200_hardware():
    return hardware_from_single_satellite_case(ROOT / "examples/single_satellite/h200_gpu_derived.json")


def chat_workload():
    return TokenDistribution(
            prompt_tokens_p50=512,
            prompt_tokens_p95=2048,
            output_tokens_p50=128,
            output_tokens_p95=512,
    )


class ServingCapacityTests(unittest.TestCase):
    def test_llama3_preset_has_weight_and_kv_formulas(self):
        model = LLMModelProfile.from_dict({"preset": "llama3-8b", "precision": "bf16"})
        self.assertEqual(model.family, "llama3")
        self.assertAlmostEqual(model.weight_memory_gb, 16.0)
        self.assertEqual(model.kv_cache_bytes_per_token, 131072.0)
        self.assertEqual(model.kv_head_count, 8)

    def test_qwen_and_305b_presets_are_available_without_heuristics(self):
        qwen = LLMModelProfile.from_dict({"preset": "qwen2.5-72b"})
        dense_305b = LLMModelProfile.from_dict({"preset": "generic-305b-dense"})
        moe_305b = LLMModelProfile.from_dict({"preset": "generic-305b-moe-a32b"})
        self.assertEqual(qwen.family, "qwen")
        self.assertEqual(qwen.architecture_source, "preset_qwen_review_required")
        self.assertAlmostEqual(dense_305b.parameter_count_b, 305.0)
        self.assertAlmostEqual(moe_305b.active_parameters_b, 32.0)

    def test_moe_split_derives_total_non_routed_and_routed_expert_parameters(self):
        model = LLMModelProfile.from_dict({"preset": "deepseek-moe-671b"})
        expected_per_expert = (671.0 - 37.0) / (256 - 8)
        self.assertAlmostEqual(model.routed_expert_parameters_b, expected_per_expert * 256)
        self.assertAlmostEqual(model.non_routed_parameters_b, 671.0 - expected_per_expert * 256)
        self.assertAlmostEqual(model.active_routed_parameters_b, 37.0 - model.non_routed_parameters_b)
        self.assertEqual(model.resolved_moe_layer_count, model.layer_count)

    def test_configurable_qwen_like_architecture_does_not_require_preset(self):
        model = LLMModelProfile.from_dict(
            {
                "name": "qwen-custom-7b",
                "family": "qwen",
                "parameter_count_b": 7.6,
                "layer_count": 28,
                "hidden_size": 3584,
                "attention_head_count": 28,
                "kv_head_count": 4,
                "intermediate_size": 18944,
                "max_context_tokens": 32768,
                "precision": {"name": "bf16", "weight_bits": 16, "activation_bits": 16, "kv_cache_bits": 16},
            }
        )
        self.assertEqual(model.architecture_source, "configured")
        self.assertAlmostEqual(model.weight_memory_gb, 15.2)
        self.assertGreater(model.kv_cache_bytes_per_token, 0)

    def test_single_satellite_throughput_outputs_prefill_decode_and_kv_capacity(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "llama3-8b"},
                "prefill": {"batch_size": 1},
                "decode": {"batch_size": 8},
                "effective_utilization": 0.5,
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=6,
        )
        self.assertTrue(result.evaluated)
        self.assertTrue(result.feasible)
        self.assertEqual(result.deployment_mode, "single_satellite")
        self.assertEqual(result.satellite_count_used, 1)
        self.assertGreater(result.prefill_tokens_per_s or 0, 0)
        self.assertGreater(result.decode_tokens_per_s or 0, 0)
        self.assertGreater(result.kv_cache_capacity_tokens or 0, 2048)
        metrics = result.to_performance_metrics().to_dict()
        self.assertTrue(metrics["evaluated"])
        self.assertGreater(metrics["request_throughput_per_s"], 0)
        self.assertEqual(metrics["satellite_count_used"], 1)
        self.assertEqual(result.deployment_group_count, 6)
        self.assertEqual(result.data_parallel_replicas_per_group, 1)
        self.assertEqual(
            result.provenance["single_satellite_multi_card_parallelism"],
            "not_modeled_todo",
        )

    def test_explicit_hardware_argument_overrides_serving_embedded_hardware(self):
        embedded = replace(h200_hardware(), name="embedded_one_acc", accelerator_count=1)
        explicit = replace(h200_hardware(), name="explicit_three_acc", accelerator_count=3)
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "llama3-8b"},
                "hardware": embedded.to_dict(),
            }
        )

        result = estimate_serving_throughput(
            spec,
            hardware=explicit,
            token_distribution=chat_workload(),
            constellation_satellite_count=6,
        )

        self.assertEqual(result.hardware["name"], "explicit_three_acc")
        self.assertEqual(result.hardware["accelerator_count"], 3)

    def test_embedded_serving_hardware_still_supports_performance_only_calls(self):
        embedded = replace(h200_hardware(), name="embedded_one_acc", accelerator_count=1)
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "llama3-8b"},
                "hardware": embedded.to_dict(),
                "effective_utilization": 0.2,
            }
        )

        result = estimate_serving_throughput(
            spec,
            token_distribution=chat_workload(),
            constellation_satellite_count=6,
        )

        self.assertTrue(result.evaluated)
        self.assertEqual(result.hardware["name"], "embedded_one_acc")
        self.assertEqual(result.hardware["accelerator_count"], 1)
        self.assertAlmostEqual(result.provenance["serving_calibration"]["effective_compute_utilization"], 0.2)

    def test_single_satellite_multi_card_parallelism_is_not_a_modeled_dimension(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "llama3-8b"},
                "prefill": {"batch_size": 1},
                "decode": {"batch_size": 1},
            }
        )
        one_acc = replace(h200_hardware(), accelerator_count=1)
        three_acc = h200_hardware()
        one = estimate_serving_throughput(spec, hardware=one_acc, token_distribution=chat_workload(), constellation_satellite_count=1)
        three = estimate_serving_throughput(spec, hardware=three_acc, token_distribution=chat_workload(), constellation_satellite_count=1)
        self.assertTrue(one.feasible)
        self.assertTrue(three.feasible)
        self.assertEqual(one.local_tensor_parallel_degree, 1)
        self.assertEqual(three.local_tensor_parallel_degree, 1)
        self.assertEqual(one.data_parallel_replicas_per_group, 1)
        self.assertEqual(three.data_parallel_replicas_per_group, 1)
        self.assertGreater((three.prefill_tokens_per_s or 0), (one.prefill_tokens_per_s or 0))

    def test_large_model_is_infeasible_without_cross_satellite_parallelism(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {"strategy": "single_satellite"},
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=6,
        )
        self.assertFalse(result.feasible)
        self.assertGreater(result.min_satellites_for_weights, 1)
        self.assertTrue(any("model weights need" in warning for warning in result.warnings))

    def test_auto_expert_parallel_fits_large_moe_model_across_satellites(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {"strategy": "auto", "auto_select": True, "max_satellites": 6},
                "decode": {"batch_size": 4},
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=6,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.deployment_mode, "expert_parallel")
        self.assertGreaterEqual(result.satellite_count_used, 4)
        self.assertEqual(result.parallelism["tensor_parallel_degree"], 1)
        self.assertGreater(result.parallelism["expert_parallel_degree"], 1)
        self.assertGreater(result.prefill_interconnect_time_s or 0, 0)
        self.assertTrue(any("auto parallelism selected" in warning for warning in result.warnings))

    def test_cross_satellite_ep_without_isl_is_infeasible_and_json_safe(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {
                    "strategy": "expert_parallel",
                    "expert_parallel_degree": 4,
                    "pipeline_parallel_degree": 1,
                    "max_satellites": 4,
                },
            }
        )
        no_isl_hardware = replace(h200_hardware(), inter_satellite_bandwidth_gbps=0.0)
        result = estimate_serving_throughput(
            spec,
            hardware=no_isl_hardware,
            token_distribution=chat_workload(),
            constellation_satellite_count=4,
        )
        self.assertFalse(result.feasible)
        self.assertIsNone(result.prefill_tokens_per_s)
        self.assertTrue(any("positive ISL bandwidth" in warning for warning in result.warnings))
        json.dumps(result.to_dict(), allow_nan=False)

    def test_manual_throughput_override_still_requires_cross_satellite_isl(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {
                    "strategy": "expert_parallel",
                    "expert_parallel_degree": 4,
                    "max_satellites": 4,
                },
                "prefill": {"token_throughput_per_s_per_accelerator": 1000},
                "decode": {"token_throughput_per_s_per_accelerator": 200},
            }
        )
        no_isl_hardware = replace(h200_hardware(), inter_satellite_bandwidth_gbps=0.0)
        result = estimate_serving_throughput(
            spec,
            hardware=no_isl_hardware,
            token_distribution=chat_workload(),
            constellation_satellite_count=4,
        )
        self.assertFalse(result.feasible)
        self.assertIsNone(result.prefill_tokens_per_s)
        self.assertTrue(any("positive ISL bandwidth" in warning for warning in result.warnings))

    def test_pipeline_parallel_reports_cross_satellite_activation_cost(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {
                    "strategy": "pipeline_parallel",
                    "tensor_parallel_degree": 1,
                    "pipeline_parallel_degree": 4,
                    "microbatch_count": 4,
                    "max_satellites": 4,
                },
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=6,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.deployment_mode, "pipeline_parallel")
        self.assertEqual(result.satellite_count_used, 4)
        self.assertGreater(result.prefill_interconnect_time_s or 0, 0)
        self.assertGreater(result.parallelism["bubble_multiplier"], 1.0)

    def test_auto_hybrid_never_exceeds_max_satellites(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {"strategy": "auto_hybrid", "auto_select": True, "max_satellites": 6},
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=6,
        )
        self.assertLessEqual(result.satellite_count_used, 6)
        self.assertTrue(result.feasible)

    def test_tp_ep_hybrid_splits_non_routed_and_expert_memory_separately(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {
                    "strategy": "tensor_expert_parallel",
                    "tensor_parallel_degree": 2,
                    "expert_parallel_degree": 4,
                    "max_satellites": 4,
                },
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=4,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.deployment_mode, "tensor_expert_parallel")
        self.assertEqual(result.satellite_count_used, 4)
        breakdown = result.provenance["weight_memory_breakdown_gb_per_satellite"]
        model = LLMModelProfile.from_dict({"preset": "deepseek-moe-671b"})
        self.assertAlmostEqual(breakdown["non_routed"], model.non_routed_memory_gb / 2)
        self.assertAlmostEqual(breakdown["routed_expert"], model.routed_expert_memory_gb / 4)

    def test_kv_capacity_is_not_sharded_by_expert_parallel_degree(self):
        base = {
            "model": {"preset": "deepseek-moe-671b"},
            "parallelism": {
                "strategy": "expert_parallel",
                "expert_parallel_degree": 1,
                "max_satellites": 1,
            },
        }
        ep1 = estimate_serving_throughput(
            ServingSpec.from_dict(base),
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=8,
        )
        base["parallelism"]["expert_parallel_degree"] = 4
        base["parallelism"]["max_satellites"] = 4
        ep4 = estimate_serving_throughput(
            ServingSpec.from_dict(base),
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=8,
        )
        model = LLMModelProfile.from_dict({"preset": "deepseek-moe-671b"})
        usable = h200_hardware().total_fast_mem_gb * 0.9
        expected_ep4 = int(
            (usable - ep4.weight_memory_per_satellite_gb)
            * 1e9
            / model.kv_cache_bytes_per_token
        )
        self.assertEqual(ep4.kv_cache_capacity_tokens, expected_ep4)
        self.assertLess(ep1.kv_cache_capacity_tokens or 0, ep4.kv_cache_capacity_tokens or 0)

    def test_lcm_stage_rank_pool_for_non_divisible_tp_ep(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {
                    "strategy": "tensor_expert_parallel",
                    "tensor_parallel_degree": 3,
                    "expert_parallel_degree": 4,
                    "pipeline_parallel_degree": 1,
                    "max_satellites": 12,
                },
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=12,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.parallelism["stage_parallel_degree"], 12)
        self.assertEqual(result.satellite_count_used, 12)

    def test_expert_parallel_interconnect_tracks_isl_bandwidth(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {
                    "strategy": "expert_parallel",
                    "expert_parallel_degree": 4,
                    "max_satellites": 4,
                },
                "decode": {"batch_size": 4},
            }
        )
        fast = estimate_serving_throughput(
            spec,
            hardware=replace(h200_hardware(), inter_satellite_bandwidth_gbps=800.0),
            token_distribution=chat_workload(),
            constellation_satellite_count=4,
        )
        slow = estimate_serving_throughput(
            spec,
            hardware=replace(h200_hardware(), inter_satellite_bandwidth_gbps=50.0),
            token_distribution=chat_workload(),
            constellation_satellite_count=4,
        )
        self.assertTrue(fast.feasible)
        self.assertTrue(slow.feasible)
        self.assertGreater(slow.prefill_interconnect_time_s or 0, (fast.prefill_interconnect_time_s or 0) * 10)
        self.assertLess(slow.prefill_tokens_per_s or 0, fast.prefill_tokens_per_s or 0)

    def test_no_stage_dp_tp_ep_uses_path_specific_throughput_parallelism_not_lcm(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {
                    "strategy": "tensor_expert_parallel",
                    "tp": 2,
                    "ep": 8,
                    "pp": 1,
                    "dp": 1,
                    "stage_dp_mode": "none",
                    "max_satellites": 8,
                },
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=8,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.stage_satellite_count, 8)
        self.assertEqual(result.satellites_per_deployment, 8)
        self.assertEqual(result.throughput_parallelism_always_path, 2)
        self.assertEqual(result.throughput_parallelism_routed_path, 8)
        self.assertEqual(result.provenance["prefill_stage"]["non_routed_throughput_parallelism"], 2.0)
        self.assertNotEqual(
            result.provenance["prefill_stage"]["non_routed_throughput_parallelism"],
            result.stage_satellite_count,
        )

    def test_uniform_stage_dp_increases_throughput_parallelism_and_satellite_placement(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {
                    "strategy": "tensor_expert_parallel",
                    "tp": 2,
                    "ep": 8,
                    "pp": 1,
                    "dp": 2,
                    "stage_dp_mode": "uniform",
                    "max_satellites": 16,
                },
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=16,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.stage_satellite_count, 16)
        self.assertEqual(result.satellites_per_deployment, 16)
        self.assertEqual(result.satellite_count_used, 16)
        self.assertEqual(result.throughput_parallelism_always_path, 4)
        self.assertEqual(result.throughput_parallelism_routed_path, 16)
        self.assertEqual(result.data_parallel_replicas_per_group, 2)

    def test_path_specific_stage_dp_reports_per_path_replicas_explicitly(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {
                    "strategy": "tensor_expert_parallel",
                    "tp": 2,
                    "ep": 8,
                    "stage_dp_mode": "path_specific",
                    "dp_attention": 2,
                    "dp_expert": 3,
                    "max_satellites": 24,
                },
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=24,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.stage_satellite_count, 24)
        self.assertEqual(result.throughput_parallelism_always_path, 4)
        self.assertEqual(result.throughput_parallelism_routed_path, 24)
        self.assertEqual(result.dp_attention, 2)
        self.assertEqual(result.dp_expert, 3)
        self.assertEqual(result.dp_replica_factor, 3)
        self.assertEqual(result.data_parallel_replicas_per_group, 3)
        self.assertEqual(
            result.provenance["data_parallel_replica_semantics"]["dp_replica_model"],
            "path_specific",
        )

    def test_stage_dp_does_not_reduce_per_satellite_weight_or_kv_memory(self):
        base = {
            "model": {"preset": "deepseek-moe-671b"},
            "parallelism": {
                "strategy": "tensor_expert_parallel",
                "tp": 2,
                "ep": 8,
                "stage_dp_mode": "none",
                "max_satellites": 8,
            },
        }
        no_dp = estimate_serving_throughput(
            ServingSpec.from_dict(base),
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=32,
        )
        base["parallelism"]["stage_dp_mode"] = "uniform"
        base["parallelism"]["dp"] = 2
        base["parallelism"]["max_satellites"] = 16
        uniform_dp = estimate_serving_throughput(
            ServingSpec.from_dict(base),
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=32,
        )
        self.assertAlmostEqual(
            uniform_dp.weight_memory_per_satellite_gb,
            no_dp.weight_memory_per_satellite_gb,
        )
        self.assertEqual(uniform_dp.kv_cache_capacity_tokens, no_dp.kv_cache_capacity_tokens)
        self.assertLess(uniform_dp.deployment_group_count, no_dp.deployment_group_count)

    def test_pd_colocated_uses_shared_pool_utilization_request_formula(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "llama3-8b"},
                "pd_mode": "colocated",
                "prefill": {"batch_size": 1},
                "decode": {"batch_size": 8},
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=6,
        )
        expected = 1.0 / (
            512 / (result.prefill_tokens_per_s or 1)
            + 128 / (result.decode_tokens_per_s or 1)
        )
        old_min_formula = min(
            (result.prefill_tokens_per_s or 0) / 512,
            (result.decode_tokens_per_s or 0) / 128,
        )
        self.assertAlmostEqual(result.request_throughput_per_s or 0, expected)
        self.assertLess(result.request_throughput_per_s or 0, old_min_formula)
        self.assertEqual(result.kv_handoff_time_s, 0.0)

    def test_pd_colocated_ignores_fraction_corner_cases(self):
        base = {
            "model": {"preset": "llama3-8b"},
            "pd_mode": "colocated",
            "prefill": {"batch_size": 1},
            "decode": {"batch_size": 8},
        }
        corner_cases = {
            "overallocated_fractions_are_not_pool_allocations": {
                "prefill_satellite_fraction": 0.8,
                "decode_satellite_fraction": 0.8,
            },
            "zero_decode_fraction_is_irrelevant_for_colocated": {
                "prefill_satellite_fraction": 1.0,
                "decode_satellite_fraction": 0.0,
            },
            "underallocated_fractions_are_not_pool_allocations": {
                "prefill_satellite_fraction": 0.25,
                "decode_satellite_fraction": 0.25,
            },
        }

        for name, fractions in corner_cases.items():
            with self.subTest(name=name):
                result = estimate_serving_throughput(
                    ServingSpec.from_dict({**base, **fractions}),
                    hardware=h200_hardware(),
                    token_distribution=chat_workload(),
                    constellation_satellite_count=6,
                )

                self.assertTrue(result.feasible)
                self.assertEqual(result.deployment_group_count, 6)
                self.assertEqual(result.prefill_deployment_group_count, 6)
                self.assertEqual(result.decode_deployment_group_count, 6)
                self.assertEqual(result.satellite_count_used, 1)
                self.assertEqual(result.kv_handoff_time_s, 0.0)
                self.assertNotIn("prefill_satellite_fraction", result.provenance["pd"])
                self.assertFalse(any("sum to <= 1.0" in warning for warning in result.warnings))
                self.assertFalse(any("positive prefill/decode" in warning for warning in result.warnings))

    def test_pd_comparison_uses_same_active_satellite_budget(self):
        budget = {
            "mode": "fixed_total",
            "active_satellite_count": 4,
            "comparison_group_id": "pd-fair-four",
        }
        base = {
            "model": {"preset": "llama3-8b"},
            "prefill": {"batch_size": 1},
            "decode": {"batch_size": 8},
            "satellite_budget": budget,
        }

        colocated = estimate_serving_throughput(
            ServingSpec.from_dict({**base, "pd_mode": "colocated"}),
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=8,
        )
        disaggregated = estimate_serving_throughput(
            ServingSpec.from_dict(
                {
                    **base,
                    "pd_mode": "disaggregated",
                    "prefill_satellite_fraction": 0.5,
                    "decode_satellite_fraction": 0.5,
                }
            ),
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=8,
        )

        self.assertTrue(colocated.feasible)
        self.assertTrue(disaggregated.feasible)
        self.assertEqual(colocated.deployment_group_count, 4)
        self.assertEqual(colocated.prefill_deployment_group_count, 4)
        self.assertEqual(colocated.decode_deployment_group_count, 4)
        self.assertEqual(colocated.satellite_count_used, 1)
        self.assertEqual(colocated.serving_satellite_budget_count, 4)
        self.assertEqual(disaggregated.serving_satellite_budget_count, 4)
        self.assertEqual(disaggregated.prefill_deployment_group_count, 2)
        self.assertEqual(disaggregated.decode_deployment_group_count, 2)
        self.assertEqual(disaggregated.satellite_count_used, 4)
        for result in (colocated, disaggregated):
            self.assertEqual(result.satellite_budget_mode, "fixed_total")
            self.assertEqual(result.comparison_group_id, "pd-fair-four")
            self.assertEqual(result.provenance["pd"]["serving_satellite_budget_count"], 4)
            self.assertEqual(result.provenance["pd"]["satellite_budget_unused_count"], 4)
            self.assertEqual(result.provenance["pd"]["comparison_group_id"], "pd-fair-four")

    def test_pd_budget_allows_asymmetric_prefill_and_decode_satellite_counts(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "llama3-8b"},
                "pd_mode": "disaggregated",
                "prefill": {"batch_size": 1},
                "decode": {"batch_size": 8},
                "satellite_budget": {
                    "mode": "fixed_total",
                    "active_satellite_count": 6,
                    "prefill_satellites": 4,
                    "decode_satellites": 2,
                },
            }
        )

        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=8,
        )

        self.assertTrue(result.feasible)
        self.assertEqual(result.serving_satellite_budget_count, 6)
        self.assertEqual(result.prefill_deployment_group_count, 4)
        self.assertEqual(result.decode_deployment_group_count, 2)
        self.assertEqual(result.satellite_count_used, 6)
        self.assertAlmostEqual(result.provenance["pd"]["prefill_satellite_fraction"], 4 / 6)
        self.assertAlmostEqual(result.provenance["pd"]["decode_satellite_fraction"], 2 / 6)
        self.assertEqual(result.provenance["pd"]["pd_role_budget_source"], "prefill_satellites+decode_satellites")

    def test_pd_disaggregated_splits_pools_and_limits_on_kv_handoff(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "llama3-8b"},
                "pd_mode": "disaggregated",
                "prefill_satellite_fraction": 0.5,
                "decode_satellite_fraction": 0.5,
                "prefill": {"batch_size": 1},
                "decode": {"batch_size": 8},
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=4,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.prefill_deployment_group_count, 2)
        self.assertEqual(result.decode_deployment_group_count, 2)
        self.assertGreater(result.kv_handoff_time_s or 0, 0)
        self.assertGreater(result.provenance["kv_handoff"]["kv_handoff_bytes_per_request_latency"], 0)
        expected = min(
            (result.prefill_tokens_per_s or 0) / 512,
            (result.decode_tokens_per_s or 0) / 128,
            result.provenance["kv_handoff"]["kv_handoff_request_limit_per_s"],
        )
        self.assertAlmostEqual(result.request_throughput_per_s or 0, expected)
        self.assertEqual(
            result.provenance["pd"]["weight_replication"],
            "prefill and decode pools each place a full deployment",
        )

        no_isl = replace(
            h200_hardware(),
            inter_satellite_bandwidth_gbps=0.0,
            laser_terminal_count=0,
            laser_terminal_bandwidth_gbps=0.0,
        )
        blocked = estimate_serving_throughput(
            spec,
            hardware=no_isl,
            token_distribution=chat_workload(),
            constellation_satellite_count=4,
        )
        self.assertFalse(blocked.feasible)
        self.assertTrue(any("KV handoff" in warning for warning in blocked.warnings))
        self.assertEqual(
            result.provenance["isl"]["flows_by_kind"]["prefill"]["kv_handoff"],
            0,
        )
        self.assertEqual(
            result.provenance["isl"]["flows_by_kind"]["kv_handoff"]["kv_handoff"],
            1,
        )

    def test_pd_disaggregated_fraction_corner_cases(self):
        base = {
            "model": {"preset": "llama3-8b"},
            "pd_mode": "disaggregated",
            "prefill": {"batch_size": 1},
            "decode": {"batch_size": 8},
        }
        corner_cases = (
            {
                "name": "default_even_split",
                "fractions": {},
                "constellation": 8,
                "prefill_groups": 4,
                "decode_groups": 4,
                "satellites_used": 8,
                "prefill_fraction": 0.5,
                "decode_fraction": 0.5,
                "feasible": True,
            },
            {
                "name": "prefill_only_fraction_complements_decode",
                "fractions": {"prefill_satellite_fraction": 0.75},
                "constellation": 8,
                "prefill_groups": 6,
                "decode_groups": 2,
                "satellites_used": 8,
                "prefill_fraction": 0.75,
                "decode_fraction": 0.25,
                "feasible": True,
            },
            {
                "name": "decode_only_fraction_complements_prefill",
                "fractions": {"decode_satellite_fraction": 0.25},
                "constellation": 8,
                "prefill_groups": 6,
                "decode_groups": 2,
                "satellites_used": 8,
                "prefill_fraction": 0.75,
                "decode_fraction": 0.25,
                "feasible": True,
            },
            {
                "name": "sum_below_one_leaves_unallocated_satellites",
                "fractions": {"prefill_satellite_fraction": 0.25, "decode_satellite_fraction": 0.25},
                "constellation": 8,
                "prefill_groups": 2,
                "decode_groups": 2,
                "satellites_used": 4,
                "prefill_fraction": 0.25,
                "decode_fraction": 0.25,
                "feasible": True,
            },
            {
                "name": "tiny_positive_fraction_can_floor_to_no_prefill_pool",
                "fractions": {"prefill_satellite_fraction": 0.09, "decode_satellite_fraction": 0.91},
                "constellation": 10,
                "prefill_groups": 0,
                "decode_groups": 9,
                "satellites_used": 9,
                "prefill_fraction": 0.09,
                "decode_fraction": 0.91,
                "feasible": False,
                "warning": "at least one prefill and one decode",
            },
        )

        for case in corner_cases:
            with self.subTest(name=case["name"]):
                result = estimate_serving_throughput(
                    ServingSpec.from_dict({**base, **case["fractions"]}),
                    hardware=h200_hardware(),
                    token_distribution=chat_workload(),
                    constellation_satellite_count=case["constellation"],
                )

                self.assertEqual(result.feasible, case["feasible"])
                self.assertEqual(result.prefill_deployment_group_count, case["prefill_groups"])
                self.assertEqual(result.decode_deployment_group_count, case["decode_groups"])
                self.assertEqual(result.satellite_count_used, case["satellites_used"])
                self.assertAlmostEqual(result.provenance["pd"]["prefill_satellite_fraction"], case["prefill_fraction"])
                self.assertAlmostEqual(result.provenance["pd"]["decode_satellite_fraction"], case["decode_fraction"])
                if case["feasible"]:
                    self.assertGreater(result.request_throughput_per_s or 0, 0)
                else:
                    self.assertIsNone(result.request_throughput_per_s)
                    self.assertTrue(any(case["warning"] in warning for warning in result.warnings))

    def test_pd_disaggregated_rejects_fraction_overallocation(self):
        with self.assertRaisesRegex(ValueError, "sum to <= 1.0"):
            ServingSpec.from_dict(
                {
                    "model": {"preset": "llama3-8b"},
                    "pd_mode": "disaggregated",
                    "prefill_satellite_fraction": 0.8,
                    "decode_satellite_fraction": 0.8,
                }
            )

        bypassed_parser = ServingSpec(
            model=LLMModelProfile.from_dict({"preset": "llama3-8b"}),
            pd_mode="disaggregated",
            prefill_satellite_fraction=0.8,
            decode_satellite_fraction=0.8,
        )
        result = estimate_serving_throughput(
            bypassed_parser,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=10,
        )

        self.assertFalse(result.feasible)
        self.assertIsNone(result.request_throughput_per_s)
        self.assertEqual(result.prefill_deployment_group_count, 0)
        self.assertEqual(result.decode_deployment_group_count, 0)
        self.assertEqual(result.satellite_count_used, 0)
        self.assertTrue(any("sum to <= 1.0" in warning for warning in result.warnings))
        self.assertEqual(result.provenance["pd"]["prefill_pool_satellites"], 0)
        self.assertEqual(result.provenance["pd"]["decode_pool_satellites"], 0)

    def test_pd_disaggregated_rejects_invalid_fraction_corner_cases_at_parse_time(self):
        base = {"model": {"preset": "llama3-8b"}, "pd_mode": "disaggregated"}
        corner_cases = {
            "one_sided_fraction_of_one_leaves_no_decode_pool": (
                {"prefill_satellite_fraction": 1.0},
                "positive",
            ),
            "negative_fraction_is_invalid": (
                {"prefill_satellite_fraction": -0.1, "decode_satellite_fraction": 0.5},
                "positive",
            ),
            "nonfinite_fraction_is_invalid": (
                {"prefill_satellite_fraction": float("inf"), "decode_satellite_fraction": 0.5},
                "finite",
            ),
        }

        for name, (fractions, message) in corner_cases.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    ServingSpec.from_dict({**base, **fractions})


    def test_dense_model_has_no_routed_path_throughput_parallelism_or_ep_speedup(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "generic-305b-dense"},
                "parallelism": {
                    "strategy": "tensor_expert_parallel",
                    "tp": 2,
                    "ep": 4,
                    "max_satellites": 4,
                },
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=4,
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.throughput_parallelism_always_path, 2)
        self.assertEqual(result.throughput_parallelism_routed_path, 0)
        self.assertEqual(result.provenance["prefill_stage"]["routed_expert_path_throughput_processing_time_s"], 0.0)
        self.assertTrue(any("dense/non-MoE model ignores expert_parallel_degree" in warning for warning in result.warnings))

    def test_dense_ep_does_not_increase_throughput_over_same_tp_without_ep(self):
        base = {
            "model": {"preset": "generic-305b-dense"},
            "parallelism": {
                "strategy": "tensor_parallel",
                "tp": 2,
                "ep": 1,
                "max_satellites": 2,
            },
        }
        ep1 = estimate_serving_throughput(
            ServingSpec.from_dict(base),
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
        )
        base["parallelism"]["strategy"] = "tensor_expert_parallel"
        base["parallelism"]["ep"] = 4
        base["parallelism"]["max_satellites"] = 4
        ep4 = estimate_serving_throughput(
            ServingSpec.from_dict(base),
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
        )
        self.assertAlmostEqual(ep4.prefill_tokens_per_s or 0, ep1.prefill_tokens_per_s or 0)
        self.assertAlmostEqual(ep4.decode_tokens_per_s or 0, ep1.decode_tokens_per_s or 0)

    def test_manual_throughput_override_rejects_cross_satellite_parallelism(self):
        spec = ServingSpec.from_dict(
            {
                "model": {"preset": "deepseek-moe-671b"},
                "parallelism": {
                    "strategy": "expert_parallel",
                    "ep": 4,
                    "max_satellites": 4,
                },
                "prefill": {"token_throughput_per_s_per_accelerator": 1000},
                "decode": {"token_throughput_per_s_per_accelerator": 200},
            }
        )
        result = estimate_serving_throughput(
            spec,
            hardware=h200_hardware(),
            token_distribution=chat_workload(),
            constellation_satellite_count=4,
        )
        self.assertFalse(result.feasible)
        self.assertIsNone(result.prefill_tokens_per_s)
        self.assertTrue(any("manual throughput override is only supported" in warning for warning in result.warnings))


if __name__ == "__main__":
    unittest.main()
