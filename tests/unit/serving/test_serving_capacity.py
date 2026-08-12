import unittest

from orbitserve.architecture import HardwareSpec, resolve_architecture
from orbitserve.serving import (
    MODEL_SERVING_PROFILES,
    ServingSpec,
    estimate_serving_throughput,
    prefill_flops,
)


def _hardware(accelerator_count=4):
    return resolve_architecture(
        HardwareSpec.from_dict({
            "accelerator": "h200_sxm",
            "compute_payload": {
                "accelerator_count": accelerator_count,
                "purchase_cost_per_unit_usd": 50000,
            },
        })
    ).hardware_profile


class ServingCapacityTests(unittest.TestCase):
    def test_only_three_frozen_models_are_supported(self):
        self.assertEqual(
            set(MODEL_SERVING_PROFILES),
            {"llama3.3-70b", "deepseek-v3", "kimi-k3"},
        )
        with self.assertRaisesRegex(ValueError, "unsupported serving model"):
            ServingSpec.from_dict({"model": {"name": "llama3-8b"}})

    def test_frozen_prefill_coefficients_and_kv_bytes(self):
        expected = {
            "llama3.3-70b": (136_902_082_560.0, 2_621_440.0, 327_680.0),
            "deepseek-v3": (71_182_974_976.0, 4_997_120.0, 70_272.0),
            "kimi-k3": (208_000_000_000.0, 1_474_560.0, 27_648.0),
        }
        for name, values in expected.items():
            profile = MODEL_SERVING_PROFILES[name]
            self.assertEqual(
                (
                    profile.prefill_linear_flops_per_token,
                    profile.prefill_quadratic_flops_coefficient,
                    profile.kv_cache_bytes_per_token,
                ),
                values,
            )

    def test_prefill_flops_use_A_times_P_plus_B_times_P_squared(self):
        profile = MODEL_SERVING_PROFILES["llama3.3-70b"]
        prompt = 4096
        self.assertEqual(
            prefill_flops(profile.name, prompt),
            profile.prefill_linear_flops_per_token * prompt
            + profile.prefill_quadratic_flops_coefficient * prompt**2,
        )

    def test_tables_have_exact_two_field_rows_and_cover_context(self):
        spec = ServingSpec.from_dict({
            "model": {"name": "deepseek-v3"},
            "formation": {
                "member_count": 4,
                "topology": "ring",
                "link_bandwidth_gbps": 100.0,
                "link_propagation_delay_s": 1e-5,
            },
            "parallelism": {"tp": 4, "ep": 1, "pp": 1},
        })
        result = estimate_serving_throughput(spec, hardware=_hardware())
        self.assertTrue(result.evaluated)
        self.assertEqual(result.prefill_time_table[-1]["prompt_length"], 128_000)
        self.assertEqual(result.decode_time_table[-1]["prompt_length"], 128_000)
        self.assertEqual(
            set(result.prefill_time_table[0]),
            {"prompt_length", "prefill_tokens_per_s"},
        )
        self.assertEqual(
            set(result.decode_time_table[0]),
            {"prompt_length", "decode_tokens_per_s"},
        )
        self.assertEqual(result.kv_cache_bytes_per_token, 70_272.0)
        self.assertFalse(hasattr(result, "prefill_tokens_per_s"))
        self.assertFalse(hasattr(result, "decode_tokens_per_s"))

    def test_legacy_batch_pd_and_budget_fields_are_rejected(self):
        for field, value in (
            ("prefill", {"batch_size": 1}),
            ("decode", {"batch_size": 1}),
            ("pd_mode", "colocated"),
            ("satellite_budget", {}),
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "Unsupported ServingSpec"):
                    ServingSpec.from_dict({field: value})
        with self.assertRaisesRegex(ValueError, "Unsupported serving.parallelism"):
            ServingSpec.from_dict({"parallelism": {"microbatch_count": 1}})

    def test_parallel_node_count_must_equal_formation_members(self):
        with self.assertRaisesRegex(ValueError, "parallel_node_count"):
            ServingSpec.from_dict({
                "formation": {
                    "member_count": 4,
                    "link_bandwidth_gbps": 100.0,
                },
                "parallelism": {"tp": 2},
            })

    def test_lower_bandwidth_and_higher_delay_reduce_rates(self):
        base = {
            "model": {"name": "llama3.3-70b"},
            "parallelism": {"tp": 4},
        }
        fast = estimate_serving_throughput(
            ServingSpec.from_dict({
                **base,
                "formation": {
                    "member_count": 4,
                    "topology": "switch",
                    "link_bandwidth_gbps": 100.0,
                    "link_propagation_delay_s": 1e-6,
                },
            }),
            hardware=_hardware(),
        )
        slow = estimate_serving_throughput(
            ServingSpec.from_dict({
                **base,
                "formation": {
                    "member_count": 4,
                    "topology": "switch",
                    "link_bandwidth_gbps": 10.0,
                    "link_propagation_delay_s": 1e-3,
                },
            }),
            hardware=_hardware(),
        )
        self.assertGreater(
            fast.prefill_time_table[5]["prefill_tokens_per_s"],
            slow.prefill_time_table[5]["prefill_tokens_per_s"],
        )
        self.assertGreater(
            fast.decode_time_table[5]["decode_tokens_per_s"],
            slow.decode_time_table[5]["decode_tokens_per_s"],
        )


if __name__ == "__main__":
    unittest.main()
