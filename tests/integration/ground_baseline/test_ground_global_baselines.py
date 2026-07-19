import sys
import unittest
from pathlib import Path


from orbitserve.baselines.ground.global_runtime import (
    CENTRALIZED_GROUND_POLICIES,
    is_centralized_ground_policy,
    run_centralized_ground_runtime,
)
from orbitserve.runtime.contracts import RuntimeStrategySpec
from orbitserve.runtime.contracts import RuntimeScenario
from orbitserve.experiments.runtime_dispatch import evaluate_runtime_performance


def _request(request_id, arrival_time_s, *, region_id="us", prefix_key=None):
    return {
        "request_id": request_id,
        "arrival_time_s": arrival_time_s,
        "origin_region_id": region_id,
        "session_id": prefix_key,
        "prompt_tokens": 20,
        "output_tokens": 10,
        "max_output_tokens": 10,
        "ttft_deadline_s": 100.0,
        "e2e_deadline_s": 100.0,
    }


def _strategy(policy, *, replicas=(('us', 1), ('eu', 1))):
    region_ids = [region_id for region_id, _count in replicas]
    return RuntimeStrategySpec.from_dict(
        {
            "policy": policy,
            "ground_config": {
                "baseline_topology": {
                    "central_region_id": "us",
                    "gke_backend_policy": "round_robin",
                },
                "regions": [
                    {"region_id": region_id, "replica_count": count, "local_rtt_s": 0.02}
                    for region_id, count in replicas
                ],
                "wan_rtt_s": {
                    source: {
                        target: 0.20
                        for target in region_ids
                        if target != source
                    }
                    for source in region_ids
                },
                "replica_token_service": {
                    "prefill_tokens_per_s": 20.0,
                    "decode_tokens_per_s": 20.0,
                    "continuous_batch_capacity_tokens": 256,
                    "kv_cache_capacity_tokens": 1000,
                },
                "max_batch_size": 1,
                "admission_output_reservation_tokens": 10,
                "prefix_tree_max_records": 64,
                "hash_seed": "global-baseline-test",
                "hash_virtual_nodes_per_target": 32,
            },
        }
    )


class GroundGlobalBaselineTests(unittest.TestCase):
    def test_policy_set_is_first_class_and_distinct_from_skywalker(self):
        self.assertEqual(
            CENTRALIZED_GROUND_POLICIES,
            {
                "ground_central_rr",
                "ground_central_ll",
                "ground_central_ch",
                "ground_central_sgl_like",
                "ground_gke_gateway_like",
            },
        )
        self.assertTrue(is_centralized_ground_policy("ground_central_rr"))
        self.assertFalse(is_centralized_ground_policy("skywalker_like_ground"))

    def test_central_rr_uses_single_us_ingress_then_global_replica_set(self):
        result = run_centralized_ground_runtime(
            requests=(_request("eu-client", 0.0, region_id="eu"),),
            runtime_strategy=_strategy("ground_central_rr"),
        )

        record = result.records[0]
        self.assertEqual(record.action_type, "ground_central_rr")
        self.assertEqual(record.provenance["topology"], "single_us_global_coordinator_blind_push")
        self.assertEqual(record.provenance["central_region_id"], "us")
        self.assertFalse(record.provenance["pending_gate_used"])
        self.assertFalse(record.provenance["remote_tau_gate_used"])
        self.assertEqual(record.selected_path[0]["target"], "lb:us")
        self.assertEqual(record.selected_path[1]["relation"], "baseline_ingress_to_replica")
        self.assertGreater(record.ground_network_delay_s, 0.20)

    def test_central_rr_blind_pushes_to_replica_queue_instead_of_waiting_at_skywalker_lb(self):
        result = run_centralized_ground_runtime(
            requests=(
                _request("first", 0.0),
                _request("second", 0.0),
            ),
            runtime_strategy=_strategy("ground_central_rr", replicas=(("us", 1),)),
        )

        first, second = result.records
        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertEqual(second.ground_load_balancer_queue_wait_s, 0.0)
        self.assertGreater(second.ground_replica_pending_wait_s, 0.0)
        self.assertEqual(result.state_summary["module_status"]["queue_model"], "blind_push_replica_admission_queue")

    def test_global_least_outstanding_observes_all_replicas(self):
        result = run_centralized_ground_runtime(
            requests=(
                _request("first", 0.0),
                _request("second", 0.0),
            ),
            runtime_strategy=_strategy("ground_central_ll"),
        )

        self.assertNotEqual(result.records[0].ground_replica_id, result.records[1].ground_replica_id)
        self.assertEqual(
            result.records[1].provenance["dispatcher_semantics"],
            "global_replica_least_outstanding_blind_push",
        )

    def test_global_sgl_like_uses_global_prefix_metadata_but_gke_like_does_not(self):
        requests = (
            _request("warm", 0.0, prefix_key="shared"),
            _request("reuse", 2.0, prefix_key="shared"),
        )
        sgl = run_centralized_ground_runtime(
            requests=requests,
            runtime_strategy=_strategy("ground_central_sgl_like"),
        )
        gke = run_centralized_ground_runtime(
            requests=requests,
            runtime_strategy=_strategy("ground_gke_gateway_like"),
        )

        self.assertEqual(sgl.records[0].ground_replica_id, sgl.records[1].ground_replica_id)
        self.assertGreater(sgl.records[1].kv_hit_tokens, 0)
        self.assertNotEqual(gke.records[0].ground_replica_id, gke.records[1].ground_replica_id)
        self.assertEqual(
            gke.records[0].provenance["dispatcher_semantics"],
            "regional_gateway_global_round_robin_surrogate",
        )
        self.assertFalse(gke.records[0].provenance["gke_gateway_exact_reproduction"])

    def test_gke_like_keeps_nearest_regional_ingress_not_us_central_ingress(self):
        result = run_centralized_ground_runtime(
            requests=(_request("eu-client", 0.0, region_id="eu"),),
            runtime_strategy=_strategy("ground_gke_gateway_like"),
        )

        record = result.records[0]
        self.assertIsNone(record.provenance["central_region_id"])
        self.assertEqual(record.provenance["ingress_region_id"], "eu")
        self.assertEqual(record.selected_path[0]["target"], "lb:eu")

    def test_gke_like_uses_explicit_geodns_ingress_without_double_counting_dns(self):
        payload = _strategy("ground_gke_gateway_like").to_dict()
        payload["ground_config"]["regions"] = [
            {"region_id": "us", "replica_count": 1, "local_rtt_s": 0.02, "lat_deg": 0.0, "lon_deg": 0.0},
            {"region_id": "eu", "replica_count": 1, "local_rtt_s": 0.02, "lat_deg": 45.0, "lon_deg": 20.0},
        ]
        payload["ground_config"]["dns_ingress"] = {
            "mode": "nearest_healthy_region",
            "healthy_region_ids": ["us"],
            "resolution_delay_s": 0.03,
        }
        request = _request("geodns", 0.0, region_id="eu")
        request.update({"client_lat_deg": 0.0, "client_lon_deg": 0.0})

        result = run_centralized_ground_runtime(
            requests=(request,), runtime_strategy=RuntimeStrategySpec.from_dict(payload)
        )

        record = result.records[0]
        self.assertEqual(record.provenance["ingress_region_id"], "us")
        self.assertEqual(record.selected_path[0]["relation"], "dns_resolution")
        path_delay = sum(leg["one_way_delay_s"] for leg in record.selected_path)
        self.assertAlmostEqual(path_delay, record.ground_network_delay_s)
        self.assertAlmostEqual(
            record.e2e_s,
            path_delay + record.ground_replica_pending_wait_s
            + record.prefill_service_s + record.decode_service_s,
        )

    def test_central_baseline_honors_one_program_per_client_closed_loop_contract(self):
        payload = _strategy("ground_central_rr", replicas=(("us", 1),)).to_dict()
        payload["ground_config"]["client_harness"] = {
            "mode": "one_program_per_client",
            "program_execution_model": "parallel_bag",
        }
        requests = [
            _request("program-0-a", 0.0),
            _request("program-0-b", 0.0),
            _request("program-1", 0.0),
        ]
        for request, program_id, sequence_index in (
            (requests[0], "program-0", 0),
            (requests[1], "program-0", 0),
            (requests[2], "program-1", 1),
        ):
            request.update(
                {
                    "client_id": "client-a",
                    "program_id": program_id,
                    "program_sequence_index": sequence_index,
                }
            )

        result = run_centralized_ground_runtime(
            requests=tuple(requests), runtime_strategy=RuntimeStrategySpec.from_dict(payload)
        )
        records = {item.request_id: item for item in result.records}
        first_program_end = max(
            records[request_id].arrival_time_s + records[request_id].e2e_s
            for request_id in ("program-0-a", "program-0-b")
        )
        self.assertGreaterEqual(records["program-1"].arrival_time_s, first_program_end)
        harness = result.state_summary["client_harness"]
        self.assertEqual(harness["completed_program_count"], 2)
        self.assertEqual(harness["max_inflight_programs_per_client"], 1)
        self.assertIn("client_harness", records["program-1"].provenance)

    def test_public_evaluator_treats_central_baseline_as_terrestrial_not_leo(self):
        strategy = _strategy("ground_central_ch")
        scenario = RuntimeScenario(
            scenario_id="central-ch-public-evaluator",
            requests=(_request("request", 0.0, region_id="eu", prefix_key="session"),),
            regions=({"region_id": "us"}, {"region_id": "eu"}),
            runtime_config=strategy,
        )

        result = evaluate_runtime_performance(scenario)

        self.assertTrue(result.evaluated)
        self.assertEqual(result.trace_model, "terrestrial_network_surrogate")
        self.assertEqual(
            result.module_status["queue_model"],
            "centralized_blind_push_replica_queue",
        )
        self.assertEqual(result.request_records[0]["action_type"], "ground_central_ch")
        self.assertFalse(result.details["terrestrial_costing"]["evaluated"])

    def test_central_replica_queue_uses_causal_target_arrival_not_ingress_order(self):
        strategy = _strategy("ground_central_ch", replicas=(("us", 1), ("eu", 1)))
        # EU starts first, but reaches the US coordinator/replica after the
        # later US request.  The US request must enter the replica first.
        result = run_centralized_ground_runtime(
            requests=(
                _request("early-long-path", 0.0, region_id="eu", prefix_key="same-session"),
                _request("late-short-path", 0.05, region_id="us", prefix_key="same-session"),
            ),
            runtime_strategy=strategy,
        )
        records = {record.request_id: record for record in result.records}
        traces = {trace["request_id"]: trace for trace in result.decision_traces}

        self.assertLess(
            traces["late-short-path"]["audit"]["replica_arrival_s"],
            traces["early-long-path"]["audit"]["replica_arrival_s"],
        )
        self.assertEqual(records["late-short-path"].ground_replica_pending_wait_s, 0.0)
        self.assertGreater(records["early-long-path"].ground_replica_pending_wait_s, 0.0)

    def test_central_latency_equals_all_path_legs_queue_and_service_including_local_return(self):
        result = run_centralized_ground_runtime(
            requests=(_request("us-local", 0.0, region_id="us"),),
            runtime_strategy=_strategy("ground_central_rr", replicas=(("us", 1),)),
        )

        record = result.records[0]
        path_delay = sum(leg["one_way_delay_s"] for leg in record.selected_path)
        self.assertAlmostEqual(record.ground_network_delay_s, path_delay)
        self.assertAlmostEqual(
            record.e2e_s,
            path_delay
            + record.ground_replica_pending_wait_s
            + record.prefill_service_s
            + record.decode_service_s,
        )
        # local RTT is 20ms in the fixture, so input and streaming return
        # each contribute 10ms even without a WAN leg.
        self.assertAlmostEqual(path_delay, 0.02)

    def test_central_client_harness_supports_program_dependency_dag(self):
        payload = _strategy("ground_central_rr", replicas=(("us", 1),)).to_dict()
        payload["ground_config"]["client_harness"] = {
            "mode": "one_program_per_client",
            "program_execution_model": "dependency_dag",
        }
        root = _request("root", 0.0)
        sibling = _request("sibling", 0.0)
        child = _request("child", 0.0)
        for request in (root, sibling, child):
            request.update(
                {
                    "client_id": "client-a",
                    "program_id": "tree-0",
                    "program_sequence_index": 0,
                }
            )
        child["program_request_dependencies"] = ["root"]

        result = run_centralized_ground_runtime(
            requests=(root, sibling, child),
            runtime_strategy=RuntimeStrategySpec.from_dict(payload),
        )
        records = {record.request_id: record for record in result.records}
        self.assertEqual(records["root"].arrival_time_s, 0.0)
        self.assertEqual(records["sibling"].arrival_time_s, 0.0)
        self.assertGreaterEqual(
            records["child"].arrival_time_s,
            records["root"].arrival_time_s + records["root"].e2e_s,
        )
        self.assertEqual(
            result.state_summary["client_harness"]["program_execution_model"],
            "dependency_dag",
        )

    def test_central_evaluator_costs_the_ingress_to_replica_leg_from_selected_path(self):
        payload = _strategy("ground_central_rr").to_dict()
        payload["ground_config"]["terrestrial_costing"] = {
            "service_period_s": 3600,
            "region_replica_hourly_price_usd": {"us": 0.5, "eu": 1.0},
            "region_egress_cost_per_gb_usd": {"us": 0.1, "eu": 0.2},
            "prompt_bytes_per_token": 4,
            "output_bytes_per_token": 4,
        }
        scenario = RuntimeScenario(
            scenario_id="central-cost-path",
            requests=(_request("eu-client", 0.0, region_id="eu"),),
            regions=({"region_id": "us"}, {"region_id": "eu"}),
            runtime_config=RuntimeStrategySpec.from_dict(payload),
        )

        result = evaluate_runtime_performance(scenario)

        costing = result.details["terrestrial_costing"]
        self.assertTrue(costing["evaluated"])
        self.assertGreater(costing["egress_cost_usd"], 0.0)
        us_row = next(row for row in costing["region_costs"] if row["region_id"] == "us")
        self.assertGreater(us_row["prompt_egress_bytes"], 0.0)


if __name__ == "__main__":
    unittest.main()
