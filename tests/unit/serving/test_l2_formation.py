import unittest

from orbitserve.architecture import HardwareSpec, resolve_architecture
from orbitserve.serving import L2FormationSpec, ServingSpec, estimate_serving_throughput
from orbitserve.workload import TokenDistribution


def _hardware():
    return resolve_architecture(
        HardwareSpec.from_dict(
            {
                "accelerator": "h200_sxm",
                "compute_payload": {
                    "accelerator_count": 3,
                    "purchase_cost_per_unit_usd": 50000,
                },
            }
        )
    ).hardware_profile


def _workload():
    return TokenDistribution(
        prompt_tokens_p50=512,
        prompt_tokens_p95=1024,
        output_tokens_p50=128,
        output_tokens_p95=256,
    )


def _estimate(formation):
    return estimate_serving_throughput(
        ServingSpec.from_dict(
            {
                "model": {"preset": "llama3-8b"},
                "prefill": {"batch_size": 1},
                "decode": {"batch_size": 8},
                "formation": formation,
            }
        ),
        hardware=_hardware(),
        token_distribution=_workload(),
        constellation_satellite_count=1,
    )


class L2FormationTests(unittest.TestCase):
    def test_single_member_is_identity(self):
        result = _estimate({"member_count": 1})
        self.assertEqual(result.formation_member_count, 1)
        self.assertEqual(result.physical_satellite_count, 1)
        self.assertEqual(result.formation_prefill_communication_time_s, 0.0)
        self.assertEqual(result.prefill_scaling_efficiency, 1.0)

    def test_multi_member_embeds_communication_and_aggregates_kv(self):
        single = _estimate({"member_count": 1})
        formation = _estimate(
            {
                "member_count": 4,
                "topology": "ring",
                "link_bandwidth_gbps": 100.0,
                "link_propagation_delay_s": 0.001,
            }
        )
        self.assertEqual(formation.internal_communication_accounting, "embedded")
        self.assertGreater(formation.formation_prefill_communication_time_s, 0.0)
        self.assertEqual(
            formation.kv_cache_capacity_tokens,
            single.kv_cache_capacity_tokens * 4,
        )
        self.assertGreater(formation.prefill_tokens_per_s, single.prefill_tokens_per_s)
        self.assertLessEqual(formation.prefill_scaling_efficiency, 1.0)

    def test_lower_bandwidth_and_higher_delay_do_not_reduce_communication(self):
        fast = _estimate(
            {
                "member_count": 4,
                "topology": "switch",
                "link_bandwidth_gbps": 100.0,
                "link_propagation_delay_s": 0.0001,
            }
        )
        slow = _estimate(
            {
                "member_count": 4,
                "topology": "switch",
                "link_bandwidth_gbps": 10.0,
                "link_propagation_delay_s": 0.001,
            }
        )
        self.assertGreaterEqual(
            slow.formation_prefill_communication_time_s,
            fast.formation_prefill_communication_time_s,
        )

    def test_contract_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "member_count"):
            L2FormationSpec(member_count=0)
        with self.assertRaisesRegex(ValueError, "topology"):
            L2FormationSpec(topology="mesh")
        with self.assertRaisesRegex(ValueError, "link_bandwidth_gbps"):
            L2FormationSpec(member_count=2, link_bandwidth_gbps=0.0)


if __name__ == "__main__":
    unittest.main()
