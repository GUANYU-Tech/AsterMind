import sys
import unittest
from pathlib import Path


from orbitserve.baselines.ground.protocol import (
    build_skywalker_paper_strategy,
    even_replica_allocations,
    run_skywalker_paper_protocol,
)


def _base_config():
    return {
        "regions": [
            {"region_id": "us", "replica_count": 1, "local_rtt_s": 0.02},
            {"region_id": "eu", "replica_count": 1, "local_rtt_s": 0.02},
        ],
        "wan_rtt_s": {"us": {"eu": 0.2}, "eu": {"us": 0.2}},
        "baseline_topology": {"central_region_id": "us"},
        "replica_token_service": {
            "prefill_tokens_per_s": 100.0,
            "decode_tokens_per_s": 100.0,
            "continuous_batch_capacity_tokens": 4096,
            "kv_cache_capacity_tokens": 4096,
        },
        "max_batch_size": 1,
        "admission_output_reservation_tokens": 32,
        "hash_seed": "protocol-test",
    }


def _requests():
    return (
        {
            "request_id": "us-0",
            "arrival_time_s": 0.0,
            "origin_region_id": "us",
            "user_id": "u0",
            "session_id": "s0",
            "prompt_tokens": 20,
            "output_tokens": 10,
            "max_output_tokens": 10,
            "ttft_deadline_s": 100.0,
            "e2e_deadline_s": 100.0,
        },
        {
            "request_id": "eu-0",
            "arrival_time_s": 0.1,
            "origin_region_id": "eu",
            "user_id": "u1",
            "session_id": "s1",
            "prompt_tokens": 20,
            "output_tokens": 10,
            "max_output_tokens": 10,
            "ttft_deadline_s": 100.0,
            "e2e_deadline_s": 100.0,
        },
    )


class GroundPaperProtocolTests(unittest.TestCase):
    def test_even_capacity_sweep_matches_three_region_paper_shape(self):
        allocations = even_replica_allocations(("us", "eu", "asia"), (3, 9, 18))
        self.assertEqual(
            allocations,
            (
                {"us": 1, "eu": 1, "asia": 1},
                {"us": 3, "eu": 3, "asia": 3},
                {"us": 6, "eu": 6, "asia": 6},
            ),
        )

    def test_strategy_builder_separates_topology_from_routing_policy(self):
        allocation = {"us": 1, "eu": 1}
        sky = build_skywalker_paper_strategy("skywalker_ch", _base_config(), allocation)
        central = build_skywalker_paper_strategy("central_ch", _base_config(), allocation)
        local = build_skywalker_paper_strategy("region_local", _base_config(), allocation)

        self.assertEqual(sky.policy, "skywalker_like_ground")
        self.assertEqual(sky.ground_config["routing_policy"], "consistent_hash")
        self.assertEqual(central.policy, "ground_central_ch")
        self.assertEqual(local.policy, "ground_regional")

    def test_protocol_runs_same_corpus_against_multiple_topologies_and_reports_window_metrics(self):
        result = run_skywalker_paper_protocol(
            requests=_requests(),
            base_ground_config=_base_config(),
            policy_ids=("central_rr", "skywalker_prefix", "region_local"),
            replica_allocations=({"us": 1, "eu": 1},),
            warmup_s=0.0,
        )

        self.assertEqual(len(result["rows"]), 3)
        self.assertFalse(result["valid_for_paper_main_results"])
        for row in result["rows"]:
            self.assertEqual(row["replica_count"], 2)
            self.assertGreater(row["measurement"]["completed_request_count"], 0)
            self.assertIsNotNone(row["measurement"]["ttft_p90_s"])
            self.assertFalse(row["terrestrial_costing"]["evaluated"])
        central = next(row for row in result["rows"] if row["policy_id"] == "central_rr")
        self.assertEqual(central["runtime_policy"], "ground_central_rr")


if __name__ == "__main__":
    unittest.main()
