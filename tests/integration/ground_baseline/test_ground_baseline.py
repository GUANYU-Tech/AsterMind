import sys
import unittest
from collections import OrderedDict
from pathlib import Path


from orbitserve.baselines.ground.runtime import (
    _CacheEntry,
    _ReplicaJob,
    _ReplicaState,
    _consistent_hash_target,
    _resolve_config,
    _ready_cache_entries,
    _remote_probe_candidates,
    _select_remote_region,
    _snapshot_prefix_score,
    run_ground_runtime,
)
from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.runtime import RuntimeScenario, RuntimeStrategySpec
from orbitserve.experiments.runtime_dispatch import evaluate_runtime_performance
from orbitserve.runtime.engine.trace_backed import run_trace_backed_runtime
from orbitserve.runtime.metrics.runtime_metrics import aggregate_runtime_records
from orbitserve.runtime.metrics.runtime_records import RuntimeRequestRecord
from orbitserve.runtime import RuntimeStrategySpec, get_scheduler


class TraceBundleWithoutSatelliteCatalog:
    trace_model = "surrogate_orbit_trace"
    warnings = ()


def _request(policy):
    return {
        "request_id": f"{policy}-req-0",
        "arrival_time_s": 0.0,
        "ingress_id": "ground_ingress",
        "prompt_tokens": 16,
        "output_tokens": 8,
        "input_bytes": 1024,
        "output_bytes": 512,
        "ttft_deadline_s": 10.0,
        "e2e_deadline_s": 20.0,
    }


def _ground_request(
    request_id,
    arrival_time_s,
    *,
    region_id="us",
    prompt_tokens=100,
    output_tokens=0,
    session_id=None,
    context_tokens_before_turn=None,
    prefix_key=None,
    prefix_tokens=None,
    prompt_token_ids=None,
):
    request = {
        "request_id": request_id,
        "arrival_time_s": arrival_time_s,
        "user_region_id": region_id,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_deadline_s": 1000.0,
        "e2e_deadline_s": 1000.0,
    }
    if session_id is not None:
        request["session_id"] = session_id
    if context_tokens_before_turn is not None:
        request["context_tokens_before_turn"] = context_tokens_before_turn
    if prefix_key is not None:
        request["prefix_key"] = prefix_key
    if prefix_tokens is not None:
        request["prefix_tokens"] = prefix_tokens
    if prompt_token_ids is not None:
        request["prompt_token_ids"] = list(prompt_token_ids)
    return request


def _ground_strategy(
    *,
    policy="skywalker_like_ground",
    selective_pushing="pending_requests",
    routing_policy="prefix_tree",
    region_replica_counts=(("us", 1), ("eu", 1)),
    local_rtt_s=0.02,
    wan_rtt_s=0.2,
    prefill_tokens_per_s=10.0,
    decode_tokens_per_s=10.0,
    max_batch_size=1,
    continuous_batch_capacity_tokens=1000,
    batch_contention_alpha=1.0,
    admission_output_reservation_tokens=256,
    max_replica_queue_requests=16,
    kv_cache_capacity_tokens=2000,
    remote_queue_buffer_requests=1,
):
    region_ids = [region_id for region_id, _ in region_replica_counts]
    return RuntimeStrategySpec.from_dict(
        {
            "policy": policy,
            "ground_config": {
                "probe_interval_s": 0.1,
                "remote_queue_buffer_requests": remote_queue_buffer_requests,
                "routing_policy": routing_policy,
                "selective_pushing": selective_pushing,
                "regions": [
                    {
                        "region_id": region_id,
                        "replica_count": replica_count,
                        "local_rtt_s": local_rtt_s,
                    }
                    for region_id, replica_count in region_replica_counts
                ],
                "wan_rtt_s": {
                    source: {
                        target: wan_rtt_s
                        for target in region_ids
                        if target != source
                    }
                    for source in region_ids
                },
                "replica_token_service": {
                    "prefill_tokens_per_s": prefill_tokens_per_s,
                    "decode_tokens_per_s": decode_tokens_per_s,
                    "continuous_batch_capacity_tokens": continuous_batch_capacity_tokens,
                    "kv_cache_capacity_tokens": kv_cache_capacity_tokens,
                },
                "max_batch_size": max_batch_size,
                "batch_contention_alpha": batch_contention_alpha,
                "admission_output_reservation_tokens": admission_output_reservation_tokens,
                "max_replica_queue_requests": max_replica_queue_requests,
                "max_lb_queue_wait_s": 100.0,
                "prefix_tree_max_records": 128,
            },
        }
    )


def _pending_pressure_requests():
    return (
        _ground_request("req-1", 0.0),
        _ground_request("req-2", 0.1),
        _ground_request("req-3", 0.2),
    )


class GroundBaselineRuntimeTests(unittest.TestCase):
    def test_skywalker_like_ground_defaults_are_standalone_terrestrial_actions(self):
        strategy = RuntimeStrategySpec.from_dict({"policy": "skywalker_like_ground"})

        self.assertFalse(strategy.fallback_enabled)
        self.assertIn("ground_local_compute", strategy.enabled_actions)
        self.assertIn("ground_cross_region_forward", strategy.enabled_actions)
        self.assertIn("ground_reject_drop", strategy.enabled_actions)
        self.assertNotIn("ground_fallback", strategy.enabled_actions)
        self.assertEqual(strategy.path_policy, "skywalker_two_layer_prefix_snapshot")
        self.assertEqual(strategy.queue_policy, "skywalker_pending_selective_fcfs")
        self.assertEqual(strategy.max_hop_count, 1)
        self.assertNotIn("ground_fallback_latency_s", strategy.hard_gate_overrides)

    def test_ground_config_round_trips_through_scheduler_and_runtime_strategy(self):
        ground_config = {
            "probe_interval_s": 0.1,
            "remote_queue_buffer_requests": 1,
            "routing_policy": "prefix_tree",
            "selective_pushing": "pending_requests",
            "regions": [
                {
                    "region_id": "us",
                    "replica_count": 2,
                    "local_rtt_s": 0.01,
                },
                {
                    "region_id": "eu",
                    "replica_count": 1,
                    "local_rtt_s": 0.01,
                },
            ],
            "wan_rtt_s": {"us": {"eu": 0.12}, "eu": {"us": 0.12}},
        }
        runtime_strategy = RuntimeStrategySpec.from_dict(
            {
                "policy": "skywalker_like_ground",
                "ground_config": ground_config,
            }
        )

        self.assertEqual(scheduler_spec.ground_config, ground_config)
        self.assertEqual(scheduler_spec.to_dict()["ground_config"], ground_config)
        self.assertEqual(runtime_strategy.ground_config, ground_config)
        self.assertEqual(runtime_strategy.to_dict()["ground_config"], ground_config)

    def test_calibrated_service_profiles_apply_by_region_and_replica_with_auditable_provenance(self):
        strategy = RuntimeStrategySpec.from_dict(
            {
                "policy": "ground_regional",
                "ground_config": {
                    "selective_pushing": "blind",
                    "routing_policy": "round_robin",
                    "regions": [
                        {"region_id": "us", "replica_count": 2, "local_rtt_s": 0.0}
                    ],
                    "service_profiles": {
                        "default": {
                            "profile_id": "l4-default-measured",
                            "calibration_source": "L4/SGLang microbenchmark: 2026-07-11",
                            "is_surrogate": False,
                            "prefill_tokens_per_s": 100.0,
                            "decode_tokens_per_s": 100.0,
                            "first_token_overhead_s": 0.05,
                            "continuous_batch_capacity_tokens": 1000,
                            "max_batch_size": 2,
                            "batch_slowdown": {
                                "per_active_request_multiplier": 0.25
                            },
                        },
                        "regions": {
                            "us": {
                                "profile_id": "us-l4-measured",
                                "prefill_tokens_per_s": 200.0,
                            }
                        },
                        "replicas": {
                            "ground:us:replica-1": {
                                "profile_id": "us-replica-1-measured",
                                "decode_tokens_per_s": 50.0,
                            }
                        },
                    },
                    "max_replica_queue_requests": 16,
                    "max_lb_queue_wait_s": 10.0,
                },
            }
        )
        result = run_ground_runtime(
            requests=(
                _ground_request("profile-0", 0.0, prompt_tokens=100, output_tokens=10),
                _ground_request("profile-1", 0.0, prompt_tokens=100, output_tokens=10),
                _ground_request("profile-2", 0.0, prompt_tokens=100, output_tokens=10),
            ),
            runtime_strategy=strategy,
        )
        records = {record.request_id: record for record in result.records}

        self.assertEqual(records["profile-0"].ground_replica_id, "ground:us:replica-0")
        self.assertEqual(records["profile-1"].ground_replica_id, "ground:us:replica-1")
        self.assertAlmostEqual(records["profile-0"].prefill_service_s, 0.55)
        self.assertAlmostEqual(records["profile-1"].decode_service_s, 0.2)
        self.assertAlmostEqual(records["profile-2"].prefill_service_s, 0.675)
        self.assertEqual(
            records["profile-1"].provenance["service_profile"]["profile_id"],
            "us-replica-1-measured",
        )
        self.assertEqual(
            records["profile-2"].provenance["service_profile"]["batch_slowdown_source"],
            "batch_slowdown.per_active_request_multiplier",
        )
        self.assertFalse(records["profile-0"].provenance["service_profile"]["is_surrogate"])
        calibration = result.provenance["execution_calibration"]
        self.assertTrue(calibration["execution_model_is_surrogate"])
        self.assertEqual(
            calibration["service_profiles_by_replica"]["ground:us:replica-1"]["calibration_source"],
            "L4/SGLang microbenchmark: 2026-07-11",
        )

    def test_wan_trace_samples_are_deterministic_and_emitted_in_cross_region_provenance(self):
        strategy = RuntimeStrategySpec.from_dict(
            {
                "policy": "skywalker_like_ground",
                "ground_config": {
                    "probe_interval_s": 0.1,
                    "selective_pushing": "pending_requests",
                    "routing_policy": "prefix_tree",
                    "regions": [
                        {"region_id": "us", "replica_count": 1, "local_rtt_s": 0.0},
                        {"region_id": "eu", "replica_count": 1, "local_rtt_s": 0.0},
                    ],
                    "replica_token_service": {
                        "prefill_tokens_per_s": 1.0,
                        "decode_tokens_per_s": 100.0,
                        "continuous_batch_capacity_tokens": 1000,
                    },
                    "max_batch_size": 1,
                    "max_replica_queue_requests": 16,
                    "max_lb_queue_wait_s": 100.0,
                    "wan_latency_profiles": {
                        "us": {
                            "eu": {
                                "profile_id": "us-eu-ping-trace",
                                "trace_samples_s": [0.10, 0.25, 0.40],
                                "calibration_source": "AWS us-east to eu-west RTT capture",
                                "is_surrogate": False,
                            }
                        }
                    },
                },
            }
        )
        requests = (
            _ground_request("wan-active", 0.0, prompt_tokens=100),
            _ground_request("wan-pending", 0.01, prompt_tokens=100),
            _ground_request("wan-cross", 0.11, prompt_tokens=10),
        )
        first = run_ground_runtime(requests=requests, runtime_strategy=strategy)
        second = run_ground_runtime(
            requests=tuple(reversed(requests)), runtime_strategy=strategy
        )
        first_cross = next(record for record in first.records if record.request_id == "wan-cross")
        second_cross = next(record for record in second.records if record.request_id == "wan-cross")
        profile = first_cross.provenance["wan_latency_profile"]

        self.assertTrue(first_cross.ground_cross_region)
        self.assertEqual(profile["profile_id"], "us-eu-ping-trace")
        self.assertEqual(profile["mode"], "trace_samples")
        self.assertEqual(profile["calibration_source"], "AWS us-east to eu-west RTT capture")
        self.assertFalse(profile["is_surrogate"])
        self.assertIn(profile["rtt_s"], (0.10, 0.25, 0.40))
        self.assertEqual(profile, second_cross.provenance["wan_latency_profile"])
        self.assertAlmostEqual(first_cross.ground_network_delay_s, profile["rtt_s"])
        applied = first.communication_metrics["applied_wan_latency_profiles"]
        self.assertEqual(applied[0]["profile_id"], "us-eu-ping-trace")
        self.assertTrue(
            any("deterministic request-key selection" in warning for warning in first.warnings)
        )

    def test_runtime_request_record_serializes_ground_audit_fields(self):
        record = RuntimeRequestRecord(
            request_id="ground-record-0",
            arrival_time_s=0.0,
            deployment_id="ground:eu:replica-0",
            action_type="ground_cross_region",
            accepted=True,
            completed=True,
            served=True,
            rejected=False,
            timeout=False,
            ground_origin_region_id="us",
            ground_serving_region_id="eu",
            ground_replica_id="eu-replica-0",
            ground_cross_region=True,
            ground_network_delay_s=0.12,
            ground_load_balancer_queue_wait_s=0.03,
            ground_replica_pending_wait_s=0.01,
            ground_prefix_cache_hit_ratio=0.75,
        )

        payload = record.to_dict()
        self.assertEqual(payload["ground_origin_region_id"], "us")
        self.assertEqual(payload["ground_serving_region_id"], "eu")
        self.assertEqual(payload["ground_replica_id"], "eu-replica-0")
        self.assertTrue(payload["ground_cross_region"])
        self.assertEqual(payload["ground_network_delay_s"], 0.12)
        self.assertEqual(payload["ground_load_balancer_queue_wait_s"], 0.03)
        self.assertEqual(payload["ground_replica_pending_wait_s"], 0.01)
        self.assertEqual(payload["ground_prefix_cache_hit_ratio"], 0.75)

    def test_ground_records_aggregate_as_terrestrial_not_leo_or_fallback(self):
        records = (
            RuntimeRequestRecord(
                request_id="ground-local",
                arrival_time_s=0.0,
                deployment_id="ground:us:replica-0",
                action_type="ground_local",
                accepted=True,
                completed=True,
                served=True,
                rejected=False,
                timeout=False,
                prompt_tokens=100,
                output_tokens=20,
                ttft_s=0.4,
                e2e_s=1.0,
                ground_origin_region_id="us",
                ground_serving_region_id="us",
                ground_replica_id="us-replica-0",
                ground_cross_region=False,
                ground_network_delay_s=0.01,
                fallback_used=False,
            ),
            RuntimeRequestRecord(
                request_id="ground-remote",
                arrival_time_s=1.0,
                deployment_id="ground:eu:replica-0",
                action_type="ground_cross_region",
                accepted=True,
                completed=True,
                served=True,
                rejected=False,
                timeout=False,
                prompt_tokens=100,
                output_tokens=20,
                ttft_s=0.55,
                e2e_s=1.15,
                ground_origin_region_id="us",
                ground_serving_region_id="eu",
                ground_replica_id="eu-replica-0",
                ground_cross_region=True,
                ground_network_delay_s=0.16,
                fallback_used=False,
            ),
        )

        aggregates = aggregate_runtime_records(
            records,
            time_horizon_s=10.0,
            trace_model="terrestrial_network_surrogate",
            scope="skywalker_ground_reproduction_surrogate",
        )

        metrics = aggregates.runtime_metrics
        self.assertEqual(metrics["leo_served_ratio"], 0.0)
        self.assertEqual(metrics["terrestrial_served_ratio"], 1.0)
        self.assertEqual(metrics["fallback_request_fraction"], 0.0)
        self.assertEqual(metrics["ground_cross_region_request_fraction"], 0.5)
        self.assertGreater(metrics["ground_network_delay_p95_s"], 0.01)
        self.assertTrue(aggregates.communication_metrics["evaluated"])
        self.assertEqual(
            aggregates.communication_metrics["model"],
            "terrestrial_internet_rtt_surrogate",
        )
        self.assertFalse(aggregates.energy_metrics["evaluated"])
        self.assertTrue(aggregates.energy_metrics["not_applicable"])

    def test_skywalker_local_first_ignores_remote_prefix_hit_while_local_is_available(self):
        strategy = _ground_strategy(
            prefill_tokens_per_s=100.0,
            decode_tokens_per_s=100.0,
        )
        result = run_ground_runtime(
            requests=(
                _ground_request(
                    "warm-remote-prefix",
                    0.0,
                    region_id="eu",
                    prefix_key="shared-prefix",
                    prefix_tokens=100,
                ),
                _ground_request(
                    "local-request",
                    2.0,
                    region_id="us",
                    prefix_key="shared-prefix",
                    prefix_tokens=100,
                ),
            ),
            runtime_strategy=strategy,
            policy="skywalker_like_ground",
        )

        local = result.records[1]
        self.assertEqual(local.action_type, "ground_local_compute")
        self.assertEqual(local.ground_origin_region_id, "us")
        self.assertEqual(local.ground_serving_region_id, "us")
        self.assertFalse(local.ground_cross_region)
        self.assertEqual(local.kv_hit_tokens, 0)
        self.assertEqual(local.prefill_service_s, 1.0)
        self.assertEqual(local.provenance["selection_reason"], "local_available")
        self.assertEqual(result.decision_traces[1]["local_candidate_count"], 1)

    def test_raw_origin_region_id_is_accepted_as_ground_ingress(self):
        request = _ground_request("origin-field", 0.0, region_id="us")
        request.pop("user_region_id")
        request["origin_region_id"] = "eu"

        record = run_ground_runtime(
            requests=(request,),
            runtime_strategy=_ground_strategy(),
        ).records[0]

        self.assertEqual(record.ground_origin_region_id, "eu")
        self.assertEqual(record.ground_serving_region_id, "eu")

    def test_local_pending_triggers_one_cross_region_forward_and_wan_rtt_enters_ttft(self):
        strategy = _ground_strategy()
        result = run_ground_runtime(
            requests=_pending_pressure_requests(),
            runtime_strategy=strategy,
            policy="skywalker_like_ground",
        )
        remote = result.records[2]
        control = run_ground_runtime(
            requests=(_ground_request("control", 0.2),),
            runtime_strategy=strategy,
            policy="skywalker_like_ground",
        ).records[0]

        self.assertEqual(remote.action_type, "ground_cross_region_forward")
        self.assertTrue(remote.ground_cross_region)
        self.assertEqual(remote.ground_origin_region_id, "us")
        self.assertEqual(remote.ground_serving_region_id, "eu")
        self.assertAlmostEqual(remote.ground_network_delay_s, 0.22)
        self.assertAlmostEqual(remote.ttft_s - control.ttft_s, 0.2)
        self.assertAlmostEqual(
            remote.ttft_s,
            remote.ground_network_delay_s
            + remote.ground_load_balancer_queue_wait_s
            + remote.ground_replica_pending_wait_s
            + remote.prefill_service_s,
        )
        self.assertEqual(
            [leg["relation"] for leg in remote.selected_path],
            [
                "client_to_regional_lb",
                "cross_region_lb_forward",
                "regional_lb_to_replica",
                "replica_to_client_stream",
            ],
        )
        self.assertEqual(result.decision_traces[2]["local_candidate_count"], 0)
        self.assertEqual(result.decision_traces[2]["remote_candidate_count"], 1)
        self.assertEqual(
            result.decision_traces[2]["selection_reason"],
            "local_unavailable_remote_available",
        )

    def test_ground_regional_never_crosses_region_and_queues_under_local_pressure(self):
        regional_strategy = _ground_strategy(policy="ground_regional")
        regional = run_ground_runtime(
            requests=_pending_pressure_requests(),
            runtime_strategy=regional_strategy,
            policy="ground_regional",
        )
        skywalker = run_ground_runtime(
            requests=_pending_pressure_requests(),
            runtime_strategy=_ground_strategy(),
            policy="skywalker_like_ground",
        )

        third = regional.records[2]
        self.assertTrue(all(not record.ground_cross_region for record in regional.records))
        self.assertEqual(third.action_type, "ground_local_compute")
        self.assertEqual(third.ground_origin_region_id, "us")
        self.assertEqual(third.ground_serving_region_id, "us")
        self.assertGreater(third.ground_load_balancer_queue_wait_s, 0.0)
        self.assertGreater(third.e2e_s, skywalker.records[2].e2e_s)
        self.assertEqual(regional.runtime_metrics["ground_cross_region_request_count"], 0)
        self.assertEqual(regional.runtime_metrics["ground_cross_region_request_fraction"], 0.0)
        self.assertEqual(
            regional.state_summary["module_status"]["routing_model"],
            "ground_region_local_only",
        )

    def test_same_session_cache_hit_reduces_effective_prefill_service(self):
        strategy = _ground_strategy(
            region_replica_counts=(("us", 1),),
            local_rtt_s=0.0,
            prefill_tokens_per_s=100.0,
            decode_tokens_per_s=100.0,
        )
        result = run_ground_runtime(
            requests=(
                _ground_request(
                    "turn-1",
                    0.0,
                    session_id="session-a",
                    prompt_tokens=100,
                    output_tokens=20,
                    context_tokens_before_turn=0,
                ),
                _ground_request(
                    "turn-2",
                    2.0,
                    session_id="session-a",
                    prompt_tokens=100,
                    output_tokens=20,
                    context_tokens_before_turn=80,
                ),
            ),
            runtime_strategy=strategy,
            policy="skywalker_like_ground",
        )

        first, second = result.records
        self.assertEqual(first.prefill_service_s, 1.0)
        self.assertEqual(first.kv_hit_tokens, 0)
        self.assertEqual(second.kv_hit_tokens, 80)
        self.assertEqual(second.kv_miss_tokens, 20)
        self.assertAlmostEqual(second.ground_prefix_cache_hit_ratio, 0.8)
        self.assertAlmostEqual(second.prefill_service_s, 0.2)
        self.assertLess(second.prefill_service_s, first.prefill_service_s)
        self.assertEqual(second.session_action, "sticky_kv_owner")
        self.assertAlmostEqual(result.session_metrics["kv_hit_rate_tokens"], 80 / 200)

    def test_pending_selective_pushing_avoids_blind_local_queue_buildup(self):
        pending = run_ground_runtime(
            requests=_pending_pressure_requests(),
            runtime_strategy=_ground_strategy(selective_pushing="pending_requests"),
            policy="skywalker_like_ground",
        )
        blind = run_ground_runtime(
            requests=_pending_pressure_requests(),
            runtime_strategy=_ground_strategy(selective_pushing="blind"),
            policy="skywalker_like_ground",
        )

        pending_third = pending.records[2]
        blind_third = blind.records[2]
        self.assertTrue(pending_third.ground_cross_region)
        self.assertFalse(blind_third.ground_cross_region)
        self.assertEqual(blind_third.ground_serving_region_id, "us")
        self.assertGreater(blind_third.ground_replica_pending_wait_s, 0.0)
        self.assertEqual(pending_third.ground_replica_pending_wait_s, 0.0)
        self.assertLess(pending_third.e2e_s, blind_third.e2e_s)
        self.assertGreater(
            blind.runtime_metrics["ground_replica_pending_wait_p95_s"],
            pending.runtime_metrics["ground_replica_pending_wait_p95_s"],
        )

    def test_public_runtime_evaluator_needs_no_leo_trace_or_deployment_for_ground(self):
        strategy = _ground_strategy(
            region_replica_counts=(("us", 1),),
            prefill_tokens_per_s=100.0,
            decode_tokens_per_s=100.0,
        )
        scenario = RuntimeScenario(
            scenario_id="ground-without-leo",
            time_horizon_s=10.0,
            requests=(
                _ground_request(
                    "ground-only-request",
                    0.0,
                    prompt_tokens=100,
                    output_tokens=10,
                ),
            ),
            regions=({"region_id": "us", "lat_deg": 39.0, "lon_deg": -77.0},),
            simulation_trace_ref={},
            serving_profile=None,
            runtime_config=strategy,
        )

        result = evaluate_runtime_performance(scenario)

        self.assertTrue(result.evaluated)
        self.assertFalse(result.scenario["simulation_evaluated"])
        self.assertFalse(result.scenario["simulation_required"])
        self.assertEqual(result.constellation, {})
        self.assertEqual(
            result.module_status["simulation"],
            "not_required_terrestrial_baseline",
        )
        self.assertEqual(
            result.module_status["queue_model"],
            "regional_lb_fcfs_pending_selective_pushing",
        )
        self.assertEqual(
            result.module_status["link_model"],
            "terrestrial_internet_rtt_surrogate",
        )
        self.assertEqual(result.module_status["energy_model"], "not_applicable")
        self.assertEqual(result.module_status["cost_model"], "terrestrial_cost_not_configured")
        self.assertEqual(result.trace_model, "terrestrial_network_surrogate")
        self.assertEqual(result.scope, "skywalker_ground_reproduction_surrogate")
        self.assertFalse(result.valid_for_paper_main_results)
        self.assertEqual(result.request_records[0]["action_type"], "ground_local_compute")
        self.assertFalse(result.request_records[0]["fallback_used"])
        self.assertFalse(result.provenance["runtime_engine"]["leo_trace_consumed"])
        self.assertFalse(result.provenance["runtime_engine"]["leo_cost_consumed"])

    def test_ground_runtime_is_deterministic_and_input_permutation_invariant(self):
        strategy = _ground_strategy(
            routing_policy="consistent_hash",
            region_replica_counts=(("us", 2), ("eu", 2)),
        )
        requests = (
            _ground_request("req-c", 0.2, session_id="session-c"),
            _ground_request("req-a", 0.0, session_id="session-a"),
            _ground_request("req-b", 0.1, session_id="session-b"),
            _ground_request("req-d", 0.2, region_id="eu", session_id="session-d"),
        )

        first = run_ground_runtime(
            requests=requests,
            runtime_strategy=strategy,
            policy="skywalker_like_ground",
        )
        second = run_ground_runtime(
            requests=tuple(reversed(requests)),
            runtime_strategy=strategy,
            policy="skywalker_like_ground",
        )

        self.assertEqual(
            [record.to_dict() for record in first.records],
            [record.to_dict() for record in second.records],
        )
        self.assertEqual(first.decision_traces, second.decision_traces)
        self.assertEqual(first.state_summary, second.state_summary)
        self.assertEqual(first.runtime_metrics, second.runtime_metrics)
        self.assertEqual(first.provenance, second.provenance)


    def test_global_event_queue_preserves_target_lb_arrival_causality(self):
        strategy = _ground_strategy(
            wan_rtt_s=1.0,
            prefill_tokens_per_s=1.0,
            decode_tokens_per_s=100.0,
            max_batch_size=1,
        )
        result = run_ground_runtime(
            requests=(
                _ground_request("us-active", 0.0, prompt_tokens=10),
                _ground_request("us-pending", 0.05, prompt_tokens=10),
                _ground_request("us-forwarded", 0.10, prompt_tokens=10),
                _ground_request("eu-arrives-first", 0.20, region_id="eu", prompt_tokens=10),
            ),
            runtime_strategy=strategy,
            policy="skywalker_like_ground",
        )

        records = {record.request_id: record for record in result.records}
        forwarded = records["us-forwarded"]
        target_local = records["eu-arrives-first"]
        self.assertTrue(forwarded.ground_cross_region)
        self.assertEqual(forwarded.ground_serving_region_id, "eu")
        self.assertEqual(target_local.ground_serving_region_id, "eu")
        self.assertEqual(target_local.ground_replica_pending_wait_s, 0.0)
        self.assertGreater(forwarded.ground_replica_pending_wait_s, 9.0)
        forwarded_trace = next(
            trace
            for trace in result.decision_traces
            if trace["request_id"] == "us-forwarded"
        )
        self.assertEqual(
            forwarded_trace["audit"]["event_ordering"],
            "global_arrival_before_dispatch",
        )

    def test_admission_uses_declared_reservation_not_realized_output_oracle(self):
        strategy = _ground_strategy(
            region_replica_counts=(("us", 1),),
            continuous_batch_capacity_tokens=400,
            admission_output_reservation_tokens=256,
            prefill_tokens_per_s=100.0,
            decode_tokens_per_s=1000.0,
        )
        short = run_ground_runtime(
            requests=(_ground_request("short", 0.0, prompt_tokens=100, output_tokens=0),),
            runtime_strategy=strategy,
        ).records[0]
        long = run_ground_runtime(
            requests=(_ground_request("long", 0.0, prompt_tokens=100, output_tokens=10000),),
            runtime_strategy=strategy,
        ).records[0]

        self.assertTrue(short.accepted)
        self.assertTrue(long.accepted)
        self.assertEqual(short.provenance["admission_tokens"], 356)
        self.assertEqual(long.provenance["admission_tokens"], 356)
        self.assertFalse(long.provenance["realized_output_tokens_used_for_admission"])
        self.assertGreater(long.decode_service_s, short.decode_service_s)

    def test_oversized_admission_is_rejected_instead_of_silently_capped(self):
        strategy = _ground_strategy(
            region_replica_counts=(("us", 1),),
            continuous_batch_capacity_tokens=400,
            admission_output_reservation_tokens=256,
        )
        result = run_ground_runtime(
            requests=(_ground_request("oversized", 0.0, prompt_tokens=200),),
            runtime_strategy=strategy,
        )

        record = result.records[0]
        self.assertTrue(record.rejected)
        self.assertFalse(record.accepted)
        self.assertEqual(
            record.reject_reason,
            "ground_request_exceeds_continuous_batch_capacity",
        )

    def test_concurrent_sequences_pay_explicit_contention_factor(self):
        strategy = _ground_strategy(
            region_replica_counts=(("us", 1),),
            selective_pushing="blind",
            prefill_tokens_per_s=100.0,
            decode_tokens_per_s=100.0,
            max_batch_size=2,
            continuous_batch_capacity_tokens=1000,
            batch_contention_alpha=1.0,
        )
        result = run_ground_runtime(
            requests=(
                _ground_request("first", 0.0, prompt_tokens=100),
                _ground_request("second", 0.0, prompt_tokens=100),
            ),
            runtime_strategy=strategy,
        )

        first, second = result.records
        self.assertEqual(first.ground_replica_pending_wait_s, 0.0)
        self.assertEqual(second.ground_replica_pending_wait_s, 0.0)
        self.assertAlmostEqual(first.prefill_service_s, 1.0)
        self.assertAlmostEqual(second.prefill_service_s, 2.0)
        self.assertEqual(second.provenance["batch_contention_factor"], 2.0)
        self.assertEqual(
            result.runtime_metrics["ground_replica_peak_active_request_count"],
            2,
        )

    def test_consistent_hash_uses_fixed_ring_when_an_unrelated_target_disappears(self):
        strategy = _ground_strategy(
            routing_policy="consistent_hash",
            region_replica_counts=(("us", 1), ("eu", 1), ("asia", 1)),
        )
        config = _resolve_config(strategy, regions=(), requests=())
        targets = ["us", "eu", "asia"]
        selected = _consistent_hash_target(
            "session-fixed-ring",
            all_target_ids=targets,
            available_target_ids=set(targets),
            config=config,
        )
        unrelated = next(target for target in targets if target != selected)
        selected_after = _consistent_hash_target(
            "session-fixed-ring",
            all_target_ids=targets,
            available_target_ids=set(targets) - {unrelated},
            config=config,
        )

        self.assertEqual(selected_after, selected)

    def test_routing_metadata_is_visible_before_kv_completion(self):
        strategy = _ground_strategy(
            region_replica_counts=(("us", 2),),
            prefill_tokens_per_s=100.0,
            decode_tokens_per_s=100.0,
            max_batch_size=1,
        )
        result = run_ground_runtime(
            requests=(
                _ground_request(
                    "turn-1",
                    0.0,
                    session_id="shared-session",
                    prompt_tokens=100,
                    output_tokens=20,
                    context_tokens_before_turn=0,
                ),
                _ground_request(
                    "turn-2",
                    0.01,
                    session_id="shared-session",
                    prompt_tokens=100,
                    output_tokens=20,
                    context_tokens_before_turn=80,
                ),
            ),
            runtime_strategy=strategy,
        )

        first, second = result.records
        self.assertEqual(first.ground_replica_id, second.ground_replica_id)
        self.assertGreater(second.ground_replica_pending_wait_s, 0.0)
        self.assertEqual(second.kv_hit_tokens, 80)
        self.assertTrue(second.provenance["routing_metadata_visible_at_dispatch"])
        self.assertTrue(second.provenance["kv_reuse_visible_at_completion"])

    def test_probe_snapshot_is_stale_within_a_100ms_tick(self):
        result = run_ground_runtime(
            requests=(
                _ground_request("tick-active", 0.0),
                _ground_request("tick-pending-1", 0.01),
                _ground_request("tick-pending-2", 0.02),
                _ground_request("next-tick-offload", 0.11),
            ),
            runtime_strategy=_ground_strategy(max_replica_queue_requests=16),
        )

        by_id = {record.request_id: record for record in result.records}
        self.assertFalse(by_id["tick-pending-1"].ground_cross_region)
        self.assertFalse(by_id["tick-pending-2"].ground_cross_region)
        self.assertTrue(by_id["next-tick-offload"].ground_cross_region)
        trace = next(
            item
            for item in result.decision_traces
            if item["request_id"] == "next-tick-offload"
        )
        self.assertAlmostEqual(trace["audit"]["observed_probe_time_s"], 0.1)

    def test_replica_pending_queue_hard_cap_survives_probe_staleness(self):
        result = run_ground_runtime(
            requests=(
                _ground_request("active", 0.0),
                _ground_request("one-pending", 0.01),
                _ground_request("must-not-overfill", 0.02),
            ),
            runtime_strategy=_ground_strategy(max_replica_queue_requests=1),
        )

        by_id = {record.request_id: record for record in result.records}
        self.assertFalse(by_id["one-pending"].ground_cross_region)
        self.assertTrue(by_id["must-not-overfill"].ground_cross_region)
        us_replica = result.state_summary["replicas"]["us"][0]
        self.assertLessEqual(us_replica["peak_pending_request_count"], 1)

    def test_future_cache_version_does_not_hide_an_older_ready_prefix(self):
        strategy = _ground_strategy(
            region_replica_counts=(("us", 1),),
            selective_pushing="blind",
            prefill_tokens_per_s=100.0,
            decode_tokens_per_s=100.0,
            max_batch_size=3,
            continuous_batch_capacity_tokens=2000,
            kv_cache_capacity_tokens=100,
        )
        result = run_ground_runtime(
            requests=(
                _ground_request(
                    "ready-first",
                    0.0,
                    session_id="temporal-cache",
                    prompt_tokens=100,
                    output_tokens=0,
                    context_tokens_before_turn=0,
                ),
                _ground_request(
                    "future-long",
                    0.1,
                    session_id="temporal-cache",
                    prompt_tokens=100,
                    output_tokens=1000,
                    context_tokens_before_turn=100,
                ),
                _ground_request(
                    "reuse-ready-version",
                    1.1,
                    session_id="temporal-cache",
                    prompt_tokens=100,
                    output_tokens=0,
                    context_tokens_before_turn=100,
                ),
            ),
            runtime_strategy=strategy,
        )

        by_id = {record.request_id: record for record in result.records}
        first = by_id["ready-first"]
        long_request = by_id["future-long"]
        third = by_id["reuse-ready-version"]
        self.assertLess(first.e2e_s, 1.1)
        self.assertGreater(long_request.e2e_s, third.arrival_time_s)
        self.assertEqual(third.kv_hit_tokens, 100)
        self.assertEqual(third.kv_miss_tokens, 0)

    def test_source_remote_tie_break_does_not_peek_at_hidden_replica_load(self):
        strategy = _ground_strategy(
            region_replica_counts=(("us", 1), ("asia", 1), ("eu", 1)),
        )
        config = _resolve_config(strategy, regions=(), requests=())
        asia = _ReplicaState("ground:asia:replica-0", "asia")
        europe = _ReplicaState("ground:eu:replica-0", "eu")
        snapshots = {
            "us": {
                "asia": OrderedDict(),
                "eu": OrderedDict(),
            }
        }
        request = {
            "prefix_key": "no-snapshot-hit",
            "prefix_signature": ("no-snapshot-hit",),
        }

        europe.jobs.append(
            _ReplicaJob("hidden-eu-load", 0.0, 0.0, 10.0, 100)
        )
        first = _select_remote_region(
            [("asia", [asia]), ("eu", [europe])],
            request,
            origin_region="us",
            remote_snapshots=snapshots,
            time_s=1.0,
            config=config,
            remote_lb_queue_depths={"asia": 0, "eu": 0},
        )
        europe.jobs.clear()
        asia.jobs.append(
            _ReplicaJob("hidden-asia-load", 0.0, 0.0, 10.0, 100)
        )
        second = _select_remote_region(
            [("asia", [asia]), ("eu", [europe])],
            request,
            origin_region="us",
            remote_snapshots=snapshots,
            time_s=1.0,
            config=config,
            remote_lb_queue_depths={"asia": 0, "eu": 0},
        )

        self.assertEqual(first, "asia")
        self.assertEqual(second, "asia")

    def test_cache_capacity_evicts_earliest_dispatch_not_earliest_completion(self):
        long_first = _CacheEntry(100, 10.0, ("long",), "prompt_token_ids")
        short_second = _CacheEntry(100, 1.0, ("short",), "prompt_token_ids")
        cache = OrderedDict(
            (
                ("long-dispatched-first", long_first),
                ("short-dispatched-second", short_second),
            )
        )

        retained = _ready_cache_entries(
            cache,
            at_s=11.0,
            max_entries=1,
            max_tokens=1000,
        )

        self.assertEqual(retained, (short_second,))

    def test_remote_tau_filters_a_queued_lb_despite_available_replicas(self):
        strategy = _ground_strategy(
            region_replica_counts=(("us", 1), ("asia", 1), ("eu", 1)),
            remote_queue_buffer_requests=0,
        )
        config = _resolve_config(strategy, regions=(), requests=())
        replicas = {
            region: [_ReplicaState(f"ground:{region}:replica-0", region)]
            for region in ("us", "asia", "eu")
        }

        eligible, depths, counts, available, filtered = _remote_probe_candidates(
            source_region_id="us",
            replicas_by_region=replicas,
            lb_queue_intervals={
                "us": [],
                "asia": [],
                "eu": [[0.0, 2.0]],
            },
            observed_s=1.0,
            config=config,
        )

        self.assertEqual([region for region, _ in eligible], ["asia"])
        self.assertEqual(depths, {"asia": 0, "eu": 1})
        self.assertEqual(counts, {"asia": 1, "eu": 1})
        self.assertEqual(available, ["asia", "eu"])
        self.assertEqual(filtered, ["eu"])

    def test_local_prefix_routing_chooses_the_longest_available_token_prefix(self):
        strategy = _ground_strategy(
            region_replica_counts=(("us", 2),),
            prefill_tokens_per_s=100.0,
            decode_tokens_per_s=100.0,
            max_batch_size=1,
        )
        result = run_ground_runtime(
            requests=(
                _ground_request(
                    "seed-long",
                    0.0,
                    prompt_token_ids=("root", "a", "deep"),
                ),
                _ground_request(
                    "make-long-pending",
                    0.01,
                    prompt_token_ids=("root", "a", "deep"),
                ),
                _ground_request(
                    "seed-short-other-replica",
                    0.11,
                    prompt_token_ids=("root", "b"),
                ),
                _ground_request(
                    "query-longest",
                    3.0,
                    prompt_token_ids=("root", "a", "deep", "more"),
                ),
            ),
            runtime_strategy=strategy,
        )

        by_id = {record.request_id: record for record in result.records}
        self.assertNotEqual(
            by_id["seed-long"].ground_replica_id,
            by_id["seed-short-other-replica"].ground_replica_id,
        )
        self.assertEqual(
            by_id["query-longest"].ground_replica_id,
            by_id["seed-long"].ground_replica_id,
        )

    def test_remote_prefix_snapshot_is_isolated_by_source_lb(self):
        strategy = _ground_strategy(
            region_replica_counts=(
                ("source-a", 1),
                ("source-x", 1),
                ("target-b", 1),
                ("target-c", 1),
            ),
        )
        config = _resolve_config(strategy, regions=(), requests=())
        target_b = _ReplicaState("ground:target-b:replica-0", "target-b")
        target_c = _ReplicaState("ground:target-c:replica-0", "target-c")
        shared_entry = _CacheEntry(
            100,
            0.0,
            ("shared", "prefix"),
            "prompt_token_ids",
        )
        snapshots = {
            "source-a": {
                "target-b": OrderedDict(),
                "target-c": OrderedDict((("shared", shared_entry),)),
            },
            "source-x": {
                "target-b": OrderedDict(),
                "target-c": OrderedDict(),
            },
        }
        request = {
            "prefix_key": "shared",
            "prefix_signature": ("shared", "prefix", "continuation"),
        }
        candidates = [("target-b", [target_b]), ("target-c", [target_c])]
        queue_depths = {"target-b": 0, "target-c": 0}

        from_a = _select_remote_region(
            candidates,
            request,
            origin_region="source-a",
            remote_snapshots=snapshots,
            time_s=1.0,
            config=config,
            remote_lb_queue_depths=queue_depths,
        )
        from_x = _select_remote_region(
            candidates,
            request,
            origin_region="source-x",
            remote_snapshots=snapshots,
            time_s=1.0,
            config=config,
            remote_lb_queue_depths=queue_depths,
        )

        self.assertEqual(from_a, "target-c")
        self.assertEqual(from_x, "target-b")

    def test_prefix_similarity_uses_paper_common_over_shorter_length(self):
        snapshot = OrderedDict(
            (
                (
                    "stored-short-prefix",
                    _CacheEntry(
                        2,
                        0.0,
                        ("root", "shared"),
                        "prompt_token_ids",
                    ),
                ),
            )
        )
        common, similarity = _snapshot_prefix_score(
            snapshot,
            {
                "prefix_key": "query",
                "prefix_signature": ("root", "shared", "long", "tail"),
            },
            at_s=1.0,
        )

        self.assertEqual(common, 2)
        self.assertEqual(similarity, 1.0)

    def test_rejected_late_arrival_still_extends_throughput_horizon(self):
        strategy = _ground_strategy(
            region_replica_counts=(("us", 1),),
            continuous_batch_capacity_tokens=400,
            admission_output_reservation_tokens=256,
        )
        requests = (
            _ground_request("early-served", 0.0, prompt_tokens=100),
            _ground_request("late-rejected", 100.0, prompt_tokens=200),
        )
        direct = run_ground_runtime(requests=requests, runtime_strategy=strategy)
        scenario = RuntimeScenario(
            scenario_id="late-reject-horizon",
            requests=requests,
            regions=({"region_id": "us"},),
            runtime_config=strategy,
        )
        evaluated = evaluate_runtime_performance(scenario)

        self.assertTrue(direct.records[0].served)
        self.assertTrue(direct.records[1].rejected)
        self.assertAlmostEqual(
            direct.provenance["effective_time_horizon_s"], 100.0
        )
        self.assertAlmostEqual(
            evaluated.runtime_metrics["useful_request_throughput_per_s"],
            0.01,
        )

    def test_reproduction_manifest_promotes_ground_scope_and_tail_metrics(self):
        root = Path(__file__).resolve().parents[3]
        spec = load_experiment_spec(
            root / "examples/experiments/skywalker_ground_reproduction.json"
        )
        result = run_experiment(spec)
        payload = result.to_dict()
        runtime = payload["performance"]["details"]["runtime"]

        self.assertEqual(
            payload["module_status"]["simulation"],
            "not_required_terrestrial_baseline",
        )
        self.assertEqual(
            payload["module_status"]["serving_throughput"],
            "ground_service_model_evaluated",
        )
        self.assertIsNotNone(payload["performance"]["ttft_p99_s"])
        self.assertIsNotNone(payload["performance"]["e2e_p99_s"])
        self.assertEqual(
            payload["performance"]["link_delay_p95_s"],
            runtime["runtime_metrics"]["ground_network_delay_p95_s"],
        )
        self.assertEqual(
            runtime["runtime_metrics"]["ground_output_reservation_exceeded_count"],
            0,
        )
        self.assertGreater(
            runtime["runtime_metrics"]["completed_request_throughput_per_s"],
            runtime["runtime_metrics"]["useful_request_throughput_per_s"],
        )
        self.assertGreater(
            runtime["runtime_metrics"]["completed_token_throughput_per_s"],
            runtime["runtime_metrics"]["useful_token_throughput_per_s"],
        )
        self.assertGreater(
            runtime["runtime_metrics"]["completed_output_token_throughput_per_s"],
            0.0,
        )
        self.assertEqual(
            runtime["runtime_metrics"]["completed_throughput_token_accounting"],
            "prompt_plus_output_and_output_only_both_reported",
        )
        self.assertTrue(
            any(
                "SkyWalker terrestrial reproduction surrogate" in warning
                for warning in payload["performance"]["warnings"]
            )
        )
        self.assertFalse(
            any(
                "idealized compute-only" in warning
                for warning in payload["performance"]["warnings"]
            )
        )

    def test_client_harness_serializes_programs_but_not_requests_inside_a_program(self):
        payload = _ground_strategy(
            region_replica_counts=(("us", 1),),
            local_rtt_s=0.0,
            max_batch_size=1,
            prefill_tokens_per_s=10.0,
            decode_tokens_per_s=10.0,
        ).to_dict()
        payload["ground_config"]["client_harness"] = {
            "mode": "one_program_per_client",
            "program_execution_model": "parallel_bag",
            "inter_program_think_time_s": 0.0,
        }
        strategy = RuntimeStrategySpec.from_dict(payload)
        requests = [
            _ground_request("program-0-a", 0.0),
            _ground_request("program-0-b", 0.0),
            _ground_request("program-1", 0.0),
        ]
        for request, program_id, sequence_index in (
            (requests[0], "tree-0", 0),
            (requests[1], "tree-0", 0),
            (requests[2], "tree-1", 1),
        ):
            request.update(
                {
                    "client_id": "client-0",
                    "program_id": program_id,
                    "program_sequence_index": sequence_index,
                }
            )

        result = run_ground_runtime(requests=tuple(requests), runtime_strategy=strategy)
        records = {record.request_id: record for record in result.records}
        first_program_end = max(
            records[request_id].arrival_time_s + records[request_id].e2e_s
            for request_id in ("program-0-a", "program-0-b")
        )

        self.assertEqual(records["program-0-a"].arrival_time_s, 0.0)
        self.assertEqual(records["program-0-b"].arrival_time_s, 0.0)
        self.assertGreaterEqual(records["program-1"].arrival_time_s, first_program_end)
        harness = result.state_summary["client_harness"]
        self.assertTrue(harness["inflight_program_constraint_satisfied"])
        self.assertEqual(harness["max_inflight_programs_per_client"], 1)
        self.assertEqual(harness["completed_program_count"], 2)
        self.assertIn("client_harness", records["program-1"].provenance)

    def test_client_harness_wait_limit_reject_releases_next_program_after_return_leg(self):
        """A rejected program remains client-visible until the reject returns."""

        payload = _ground_strategy(
            region_replica_counts=(("us", 1),),
            local_rtt_s=0.02,
            selective_pushing="max_outstanding",
            prefill_tokens_per_s=1.0,
            decode_tokens_per_s=1.0,
            max_batch_size=1,
        ).to_dict()
        payload["ground_config"].update(
            {
                "max_outstanding_requests": 1,
                "max_lb_queue_wait_s": 0.05,
                "client_harness": {
                    "mode": "one_program_per_client",
                    "program_execution_model": "parallel_bag",
                },
            }
        )
        blocker = _ground_request("a-blocker", 0.0, prompt_tokens=100, output_tokens=100)
        rejected = _ground_request("b-rejected", 0.10, prompt_tokens=10, output_tokens=0)
        next_program = _ground_request("c-next", 0.0, prompt_tokens=1, output_tokens=0)
        blocker.update(
            {"client_id": "blocker", "program_id": "p0", "program_sequence_index": 0}
        )
        rejected.update(
            {"client_id": "target", "program_id": "p0", "program_sequence_index": 0}
        )
        next_program.update(
            {"client_id": "target", "program_id": "p1", "program_sequence_index": 1}
        )

        result = run_ground_runtime(
            requests=(blocker, rejected, next_program),
            runtime_strategy=RuntimeStrategySpec.from_dict(payload),
        )

        records = {record.request_id: record for record in result.records}
        self.assertEqual(records["b-rejected"].reject_reason, "ground_lb_queue_wait_limit")
        reject_terminal_s = records["b-rejected"].provenance["client_harness"]["terminal_time_s"]
        # The request arrives at 100ms, then pays 10ms client->LB, 50ms LB
        # wait limit, and 10ms reject return.
        self.assertAlmostEqual(reject_terminal_s, 0.17)
        self.assertGreaterEqual(records["c-next"].arrival_time_s, reject_terminal_s)

    def test_client_harness_is_fail_closed_without_explicit_program_metadata(self):
        payload = _ground_strategy(region_replica_counts=(("us", 1),)).to_dict()
        payload["ground_config"]["client_harness"] = {
            "mode": "one_program_per_client",
            "program_execution_model": "parallel_bag",
        }
        with self.assertRaisesRegex(ValueError, "explicit client_id, program_id"):
            run_ground_runtime(
                requests=(_ground_request("missing-metadata", 0.0),),
                runtime_strategy=RuntimeStrategySpec.from_dict(payload),
            )

    def test_client_harness_rejects_time_compressed_trace(self):
        payload = _ground_strategy(region_replica_counts=(("us", 1),)).to_dict()
        payload["ground_config"]["arrival_time_scale"] = 0.1
        payload["ground_config"]["client_harness"] = {
            "mode": "one_program_per_client",
            "program_execution_model": "parallel_bag",
        }
        request = _ground_request("compressed", 0.0)
        request.update(
            {
                "client_id": "client-0",
                "program_id": "program-0",
                "program_sequence_index": 0,
            }
        )
        with self.assertRaisesRegex(ValueError, "incompatible with arrival_time_scale"):
            run_ground_runtime(
                requests=(request,), runtime_strategy=RuntimeStrategySpec.from_dict(payload)
            )

    def test_low_prefix_affinity_fallback_is_explicit_and_avoids_busy_replica(self):
        common = {
            "region_replica_counts": (("us", 2),),
            "local_rtt_s": 0.0,
            "prefill_tokens_per_s": 10.0,
            "decode_tokens_per_s": 10.0,
            "max_batch_size": 1,
        }
        requests = (
            _ground_request(
                "warm", 0.0, prompt_tokens=30, output_tokens=0,
                prompt_token_ids=("a", "b", "c"),
            ),
            _ground_request(
                "busy", 4.0, prompt_tokens=30, output_tokens=100,
                prompt_token_ids=("a", "b", "c"),
            ),
            _ground_request(
                "low-affinity", 4.1, prompt_tokens=30, output_tokens=0,
                prompt_token_ids=("a", "x", "y"),
            ),
        )
        strict = run_ground_runtime(
            requests=requests,
            runtime_strategy=_ground_strategy(**common),
        )
        fallback_payload = _ground_strategy(**common).to_dict()
        fallback_payload["ground_config"].update(
            {
                "prefix_affinity_min_similarity": 0.5,
                "prefix_low_affinity_fallback": "least_outstanding",
            }
        )
        fallback = run_ground_runtime(
            requests=requests,
            runtime_strategy=RuntimeStrategySpec.from_dict(fallback_payload),
        )
        strict_low = next(record for record in strict.records if record.request_id == "low-affinity")
        fallback_low = next(record for record in fallback.records if record.request_id == "low-affinity")

        self.assertNotEqual(strict_low.ground_replica_id, fallback_low.ground_replica_id)
        self.assertFalse(strict_low.provenance["prefix_affinity"]["fallback_used"])
        self.assertTrue(fallback_low.provenance["prefix_affinity"]["fallback_used"])
        self.assertAlmostEqual(
            fallback_low.provenance["prefix_affinity"]["threshold"], 0.5
        )
        self.assertEqual(
            fallback_low.provenance["prefix_affinity"]["fallback_policy"],
            "least_outstanding",
        )

    def test_client_harness_dependency_dag_releases_child_after_parent_and_keeps_sibling_parallel(self):
        payload = _ground_strategy(
            region_replica_counts=(("us", 1),),
            local_rtt_s=0.0,
            max_batch_size=1,
            prefill_tokens_per_s=10.0,
            decode_tokens_per_s=10.0,
        ).to_dict()
        payload["ground_config"]["client_harness"] = {
            "mode": "one_program_per_client",
            "program_execution_model": "dependency_dag",
        }
        requests = [
            _ground_request("root", 0.0),
            _ground_request("sibling", 0.0),
            _ground_request("child", 0.0),
        ]
        for request in requests:
            request.update(
                {
                    "client_id": "client-0",
                    "program_id": "tree-0",
                    "program_sequence_index": 0,
                }
            )
        requests[2]["program_request_dependencies"] = ["root"]

        result = run_ground_runtime(
            requests=tuple(requests), runtime_strategy=RuntimeStrategySpec.from_dict(payload)
        )
        records = {record.request_id: record for record in result.records}

        self.assertEqual(records["root"].arrival_time_s, 0.0)
        self.assertEqual(records["sibling"].arrival_time_s, 0.0)
        self.assertGreaterEqual(
            records["child"].arrival_time_s,
            records["root"].arrival_time_s + records["root"].e2e_s,
        )
        self.assertEqual(
            records["child"].provenance["client_harness"]["program_execution_model"],
            "dependency_dag",
        )

    def test_client_harness_requires_explicit_program_execution_model(self):
        payload = _ground_strategy(region_replica_counts=(("us", 1),)).to_dict()
        payload["ground_config"]["client_harness"] = {
            "mode": "one_program_per_client",
        }
        request = _ground_request("request", 0.0)
        request.update(
            {
                "client_id": "client-0",
                "program_id": "program-0",
                "program_sequence_index": 0,
            }
        )
        with self.assertRaisesRegex(ValueError, "program_execution_model"):
            run_ground_runtime(
                requests=(request,), runtime_strategy=RuntimeStrategySpec.from_dict(payload)
            )

    def test_low_prefix_affinity_fallback_applies_to_remote_snapshot_selection(self):
        strict_config = _resolve_config(
            _ground_strategy(region_replica_counts=(("source", 1), ("b", 1), ("c", 1))),
            regions=(),
            requests=(),
        )
        fallback_payload = _ground_strategy(
            region_replica_counts=(("source", 1), ("b", 1), ("c", 1))
        ).to_dict()
        fallback_payload["ground_config"].update(
            {
                "prefix_affinity_min_similarity": 0.5,
                "prefix_low_affinity_fallback": "least_outstanding",
            }
        )
        fallback_config = _resolve_config(
            RuntimeStrategySpec.from_dict(fallback_payload), regions=(), requests=()
        )
        target_b = _ReplicaState("ground:b:replica-0", "b")
        target_c = _ReplicaState("ground:c:replica-0", "c")
        snapshots = {
            "source": {
                "b": OrderedDict(
                    {
                        "partial": _CacheEntry(
                            10, 0.0, ("a", "b", "c"), "prompt_token_ids"
                        )
                    }
                ),
                "c": OrderedDict(),
            }
        }
        request = {
            "prefix_key": "request",
            "hash_key": "request",
            "prefix_signature": ("a", "x", "y"),
        }
        candidates = [("b", [target_b]), ("c", [target_c])]
        queue_depths = {"b": 1, "c": 0}

        strict = _select_remote_region(
            candidates,
            request,
            origin_region="source",
            remote_snapshots=snapshots,
            time_s=1.0,
            config=strict_config,
            remote_lb_queue_depths=queue_depths,
        )
        fallback = _select_remote_region(
            candidates,
            request,
            origin_region="source",
            remote_snapshots=snapshots,
            time_s=1.0,
            config=fallback_config,
            remote_lb_queue_depths=queue_depths,
        )

        self.assertEqual(strict, "b")
        self.assertEqual(fallback, "c")

    def test_geodns_nonlocal_ingress_has_one_input_dns_leg_and_one_return_leg(self):
        """Nonlocal GeoDNS must not omit ingress WAN or double-count DNS."""

        payload = _ground_strategy(
            policy="ground_regional",
            selective_pushing="blind",
            routing_policy="round_robin",
            region_replica_counts=(("us", 1), ("eu", 1)),
            local_rtt_s=0.02,
            wan_rtt_s=0.20,
            prefill_tokens_per_s=1000.0,
            decode_tokens_per_s=1000.0,
        ).to_dict()
        payload["ground_config"]["regions"] = [
            {"region_id": "us", "replica_count": 1, "local_rtt_s": 0.02, "lat_deg": 0.0, "lon_deg": 0.0},
            {"region_id": "eu", "replica_count": 1, "local_rtt_s": 0.02, "lat_deg": 40.0, "lon_deg": 20.0},
        ]
        payload["ground_config"]["dns_ingress"] = {
            "mode": "nearest_healthy_region",
            "healthy_region_ids": ["us"],
            "resolution_delay_s": 0.03,
        }
        request = _ground_request("geodns", 0.0, region_id="eu", prompt_tokens=10, output_tokens=10)
        request.update({"client_lat_deg": 0.0, "client_lon_deg": 0.0})

        result = run_ground_runtime(
            requests=(request,), runtime_strategy=RuntimeStrategySpec.from_dict(payload)
        )

        record = result.records[0]
        self.assertEqual(record.ground_serving_region_id, "us")
        self.assertEqual(record.selected_path[0]["relation"], "dns_resolution")
        self.assertEqual(record.selected_path[1]["target"], "lb:us")
        path_delay_s = sum(item["one_way_delay_s"] for item in record.selected_path)
        self.assertAlmostEqual(path_delay_s, 0.25)
        self.assertAlmostEqual(record.ground_network_delay_s, path_delay_s)
        self.assertAlmostEqual(
            record.e2e_s,
            path_delay_s + record.prefill_service_s + record.decode_service_s,
        )


if __name__ == "__main__":
    unittest.main()
