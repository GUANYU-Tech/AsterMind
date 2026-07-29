import importlib
from copy import deepcopy
import unittest

from orbitserve.runtime.contracts import ServingCapacityProfile
from orbitserve.runtime.engine.decision import CandidateSelector, SchedulerCandidate, build_candidate_selector


def _optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise


def _runtime_scenario(runtime, *, requests=None, sessions=None):
    performance = {
        "evaluated": True,
        "feasible": True,
        "prefill_tokens_per_s": 100.0,
        "decode_tokens_per_s": 50.0,
        "kv_cache_capacity_tokens": 100000,
        "request_throughput_per_s": 1.0,
    }
    strategy = runtime.RuntimeStrategySpec.from_dict(
        {
            "policy": "orbitserve",
            "lookahead_orbits": 1.0,
            "h_lookahead_s": 5700.0,
            "enabled_actions": ["local_compute", "ground_fallback", "reject_drop"],
            "fallback_enabled": True,
            "weights": {"latency": 1.0, "energy": 0.5, "queue": 1.0, "fallback": 10.0},
        }
    )
    deployment = ServingCapacityProfile.from_value(performance)
    scenario = runtime.RuntimeSimulationSpec(
        strategy=strategy,
        scenario_id="runtime-unit",
        time_horizon_s=60.0,
        workload_bundle={"status": "loaded", "request_count": len(requests or [])},
        requests=tuple(requests or [_request()]),
        sessions=tuple(sessions or ()),
        simulation_trace_ref={"evaluated": False, "warnings": ["simulation is a skeleton"]},
        serving_profile=deployment,
        warnings=("simulation trace is not implemented; runtime observed metrics are null",),
    )
    return scenario, strategy, performance


def _request(**overrides):
    row = {
        "request_id": "req-0",
        "arrival_time_s": 0.0,
        "region_id": "na_east",
        "lat_deg": 39.0,
        "lon_deg": -77.0,
        "access_mode": "gateway_sgl",
        "ingress_id": "gw_na_east",
        "prompt_tokens": 512,
        "output_tokens": 128,
        "input_bytes": 1024,
        "output_bytes": 256,
        "session_id": "session-0",
        "turn_index": 0,
        "session_turn_count": 1,
    }
    row.update(overrides)
    return row


@unittest.skipIf(_optional_import("orbitserve.runtime") is None, "orbitserve.runtime.core is not available yet")
class RuntimeSkeletonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = importlib.import_module("orbitserve.runtime")

    def test_strategy_spec_keeps_extended_scheduler_fields(self):
        strategy = self.runtime.RuntimeStrategySpec.from_dict(
            {
                "policy": "orbitserve",
                "lookahead_orbits": 1.5,
                "h_lookahead_s": 8550.0,
                "enabled_actions": ["local_compute", "cross_plane_forward", "ground_fallback"],
                "fallback_enabled": True,
                "defer_enabled": True,
                "weights": {"latency": 1.0, "energy": 0.25, "queue": 1.0, "fallback": 10.0},
            }
        )

        payload = strategy.to_dict()
        self.assertEqual(payload["policy"], "orbitserve")
        self.assertEqual(payload["lookahead_orbits"], 1.5)
        self.assertEqual(payload["h_lookahead_s"], 8550.0)
        self.assertEqual(payload["enabled_actions"], ["local_compute", "cross_plane_forward", "ground_fallback"])
        self.assertTrue(payload["fallback_enabled"])
        self.assertTrue(payload["defer_enabled"])
        self.assertEqual(payload["weights"]["fallback"], 10.0)

    def test_top_level_scheduler_exposes_fixed_runtime_strategy(self):
        scheduler = build_candidate_selector("orbitserve")
        decision = scheduler.select_action(state=None, request=None).to_dict()

        self.assertIsInstance(scheduler, CandidateSelector)
        self.assertEqual(decision["action"], "no_candidate")
        self.assertFalse(decision["evaluated"])
        self.assertEqual(decision["runtime_strategy"]["policy"], "orbitserve")
        self.assertIn("local_compute", decision["runtime_strategy"]["enabled_actions"])
        self.assertFalse(decision["optimization_support"]["strategy_mutable_during_run"])
        self.assertEqual(
            decision["optimization_support"]["selection_mode"],
            "deterministic_request_candidate_ranking",
        )
        self.assertEqual(decision["module_status"]["runtime_strategy"], "fixed_for_run")

    def test_top_level_scheduler_selects_request_level_candidate(self):
        scheduler = build_candidate_selector("energy_aware")
        local_candidate = SchedulerCandidate(
            candidate_id="local",
            action_type="local_compute",
            feasible=True,
            route_plan={"compute_satellite_id": "sat-a"},
            predicted_ttft_s=5.0,
            predicted_e2e_s=12.0,
            predicted_queue_wait_s=0.0,
            predicted_link_wait_s=0.0,
            predicted_energy_risk=0.5,
            predicted_soc_after=0.1,
            hop_count=0,
            predicted_nominal_latency_s=12.0,
            provenance={"compute_satellite_id": "sat-a", "access_satellite_id": "sat-a", "path_id": "local"},
        )
        offload_candidate = SchedulerCandidate(
            candidate_id="offload",
            action_type="cross_satellite_forward",
            feasible=True,
            route_plan={"compute_satellite_id": "sat-b"},
            predicted_ttft_s=6.0,
            predicted_e2e_s=14.0,
            predicted_queue_wait_s=0.0,
            predicted_link_wait_s=1.0,
            predicted_energy_risk=0.0,
            predicted_soc_after=0.8,
            hop_count=1,
            predicted_nominal_latency_s=14.0,
            provenance={"compute_satellite_id": "sat-b", "access_satellite_id": "sat-a", "path_id": "offload"},
        )
        decision = scheduler.select_action(
            state={"candidates": (local_candidate.to_dict(), offload_candidate.to_dict())},
            request={"request_id": "req-0"},
        ).to_dict()

        self.assertTrue(decision["evaluated"])
        self.assertEqual(decision["action"], "cross_satellite_forward")
        self.assertEqual(decision["selected_candidate_id"], "offload")
        self.assertEqual(decision["candidate_count"], 2)
        self.assertEqual(decision["feasible_candidate_count"], 2)
        self.assertEqual(
            decision["score_terms"]["score_model"],
            "request_level_resource_aware_surrogate",
        )
        self.assertEqual(
            decision["optimization_support"]["selection_mode"],
            "deterministic_request_candidate_ranking",
        )

    def test_hard_gate_mode_controls_candidate_admission(self):
        violating_fast = SchedulerCandidate(
            candidate_id="fast-low-soc",
            action_type="local_compute",
            feasible=True,
            route_plan={"compute_satellite_id": "sat-a"},
            predicted_ttft_s=1.0,
            predicted_e2e_s=2.0,
            predicted_queue_wait_s=0.0,
            predicted_link_wait_s=0.0,
            predicted_energy_risk=0.5,
            predicted_soc_after=0.1,
            hop_count=0,
            hard_gate_violations=("energy_soc",),
            hard_gate_results={"energy_soc": {"passed": False, "reason": "energy_soc"}},
            predicted_nominal_latency_s=2.0,
            provenance={"compute_satellite_id": "sat-a", "access_satellite_id": "sat-a", "path_id": "fast"},
        )
        safe_slow = SchedulerCandidate(
            candidate_id="safe-slow",
            action_type="cross_satellite_forward",
            feasible=True,
            route_plan={"compute_satellite_id": "sat-b"},
            predicted_ttft_s=10.0,
            predicted_e2e_s=20.0,
            predicted_queue_wait_s=0.0,
            predicted_link_wait_s=1.0,
            predicted_energy_risk=0.0,
            predicted_soc_after=0.8,
            hop_count=1,
            hard_gate_results={"energy_soc": {"passed": True, "reason": None}},
            predicted_nominal_latency_s=20.0,
            provenance={"compute_satellite_id": "sat-b", "access_satellite_id": "sat-a", "path_id": "safe"},
        )

        gate_mode_test_weights = {
            "latency": 1.0,
            "slack": 0.0,
            "queue": 0.0,
            "link": 0.0,
            "energy": 0.0,
            "hop": 0.0,
        }
        strict = build_candidate_selector(
            {"policy": "orbitserve", "hard_gate_mode": "strict", "weights": gate_mode_test_weights}
        ).select_action(
            state={"candidates": (violating_fast.to_dict(), safe_slow.to_dict())},
            request={"request_id": "req-0"},
        ).to_dict()
        audit = build_candidate_selector(
            {"policy": "orbitserve", "hard_gate_mode": "audit_only", "weights": gate_mode_test_weights}
        ).select_action(
            state={"candidates": (violating_fast.to_dict(), safe_slow.to_dict())},
            request={"request_id": "req-0"},
        ).to_dict()

        self.assertEqual(strict["selected_candidate_id"], "safe-slow")
        self.assertEqual(strict["feasible_candidate_count"], 1)
        self.assertEqual(strict["gate_results"]["violations_by_gate"]["energy_soc"], 1)
        self.assertEqual(strict["rejected_candidates"][0]["rejection_reason"], "hard_gate_violation")
        self.assertEqual(audit["selected_candidate_id"], "fast-low-soc")
        self.assertEqual(audit["feasible_candidate_count"], 2)
        self.assertFalse(audit["gate_results"]["hard_gate_blocking"])

    def test_scheduler_uses_one_normalization_reference_for_the_candidate_set(self):
        def candidate(candidate_id, satellite_id, predicted_e2e_s):
            return SchedulerCandidate(
                candidate_id=candidate_id,
                action_type="cross_satellite_forward",
                feasible=True,
                route_plan={"compute_satellite_id": satellite_id},
                predicted_ttft_s=predicted_e2e_s,
                predicted_e2e_s=predicted_e2e_s,
                predicted_queue_wait_s=0.0,
                predicted_link_wait_s=0.0,
                predicted_energy_risk=0.0,
                predicted_soc_after=0.8,
                hop_count=1,
                predicted_nominal_latency_s=predicted_e2e_s,
                provenance={
                    "compute_satellite_id": satellite_id,
                    "access_satellite_id": "access",
                    "path_id": candidate_id,
                },
            )

        decision = build_candidate_selector(
            {"policy": "shortest_path", "weights": {"hop": 0.0}}
        ).select_action(
            state={
                "candidates": (
                    candidate("slow", "sat-a", 20.0).to_dict(),
                    candidate("fast", "sat-b", 10.0).to_dict(),
                )
            },
            request={"request_id": "req-shared-normalization"},
        ).to_dict()

        self.assertEqual(decision["selected_candidate_id"], "fast")
        self.assertEqual(decision["score_terms"]["normalization_scope"], "request_candidate_set")
        self.assertEqual(decision["score_terms"]["normalization_refs"]["latency_s"], 20.0)
        self.assertEqual(decision["score_terms"]["normalized_e2e"], 0.5)
        latency_component = decision["score_terms"]["score_components"]["latency"]
        self.assertEqual(latency_component["raw_metric"], 10.0)
        self.assertEqual(latency_component["shared_reference"], 20.0)
        self.assertEqual(latency_component["normalized_metric"], 0.5)
        self.assertEqual(latency_component["weight"], 1.0)
        rejected_slow = decision["rejected_candidates"][0]
        self.assertEqual(rejected_slow["candidate_id"], "slow")
        self.assertEqual(rejected_slow["score_terms"]["normalization_refs"]["latency_s"], 20.0)
        self.assertEqual(rejected_slow["score_terms"]["normalized_e2e"], 1.0)


    def test_scheduler_hard_rejects_physical_ineligibility(self):
        ineligible = SchedulerCandidate(
            candidate_id="ineligible-fast",
            action_type="cross_satellite_forward",
            feasible=False,
            route_plan={"compute_satellite_id": "sat-a"},
            predicted_ttft_s=1.0,
            predicted_e2e_s=2.0,
            predicted_queue_wait_s=0.0,
            predicted_link_wait_s=0.0,
            predicted_energy_risk=0.0,
            predicted_soc_after=0.8,
            hop_count=1,
            predicted_nominal_latency_s=2.0,
            provenance={"compute_satellite_id": "sat-a", "access_satellite_id": "access", "path_id": "fast"},
        )
        eligible = SchedulerCandidate(
            candidate_id="eligible-slow",
            action_type="cross_satellite_forward",
            feasible=True,
            route_plan={"compute_satellite_id": "sat-b"},
            predicted_ttft_s=5.0,
            predicted_e2e_s=10.0,
            predicted_queue_wait_s=0.0,
            predicted_link_wait_s=0.0,
            predicted_energy_risk=0.0,
            predicted_soc_after=0.8,
            hop_count=1,
            predicted_nominal_latency_s=10.0,
            provenance={"compute_satellite_id": "sat-b", "access_satellite_id": "access", "path_id": "slow"},
        )

        decision = build_candidate_selector("shortest_path").select_action(
            state={"candidates": (ineligible.to_dict(), eligible.to_dict())},
            request={"request_id": "req-physical-gate"},
        ).to_dict()

        self.assertEqual(decision["selected_candidate_id"], "eligible-slow")
        rejected = decision["rejected_candidates"][0]
        self.assertEqual(rejected["rejection_reason"], "candidate_marked_infeasible")
        self.assertFalse(rejected["hard_gate_results"]["physical_eligibility"]["passed"])


    def test_scheduler_recomputes_soc_gate_when_threshold_is_available(self):
        candidate = SchedulerCandidate(
            candidate_id="stale-soc-pass",
            action_type="local_compute",
            feasible=True,
            route_plan={"compute_satellite_id": "sat-a"},
            predicted_ttft_s=1.0,
            predicted_e2e_s=2.0,
            predicted_queue_wait_s=0.0,
            predicted_link_wait_s=0.0,
            predicted_energy_risk=0.0,
            predicted_soc_after=0.1,
            hop_count=0,
            hard_gate_results={"energy_soc": {"passed": True, "reason": None, "soc_min": 0.2}},
            predicted_nominal_latency_s=2.0,
        )
        decision = build_candidate_selector("shortest_path").select_action(
            state={"candidates": (candidate.to_dict(),)},
            request={},
        ).to_dict()

        self.assertEqual(decision["action"], "reject_drop")
        self.assertEqual(decision["gate_results"]["violations_by_gate"]["energy_soc"], 1)

    def test_scheduler_blocks_failed_gate_without_caller_supplied_reason(self):
        candidate = SchedulerCandidate(
            candidate_id="reasonless-failure",
            action_type="local_compute",
            feasible=True,
            route_plan={"compute_satellite_id": "sat-a"},
            predicted_ttft_s=1.0,
            predicted_e2e_s=2.0,
            predicted_queue_wait_s=0.0,
            predicted_link_wait_s=0.0,
            predicted_energy_risk=0.0,
            predicted_soc_after=0.8,
            hop_count=0,
            hard_gate_results={"kv_capacity": {"passed": False, "reason": None}},
            predicted_nominal_latency_s=2.0,
        )
        decision = build_candidate_selector("shortest_path").select_action(
            state={"candidates": (candidate.to_dict(),)},
            request={},
        ).to_dict()

        self.assertEqual(decision["action"], "reject_drop")
        self.assertEqual(decision["gate_results"]["violations_by_gate"]["kv_capacity_failed"], 1)

    def test_scheduler_uses_authoritative_soc_threshold_precedence(self):
        candidate = SchedulerCandidate(
            candidate_id="stale-low-threshold",
            action_type="local_compute",
            feasible=True,
            route_plan={"compute_satellite_id": "sat-a"},
            predicted_ttft_s=1.0,
            predicted_e2e_s=2.0,
            predicted_queue_wait_s=0.0,
            predicted_link_wait_s=0.0,
            predicted_energy_risk=0.0,
            predicted_soc_after=0.1,
            hop_count=0,
            hard_gate_results={"energy_soc": {"passed": True, "reason": None, "soc_min": 0.0}},
            predicted_nominal_latency_s=2.0,
            provenance={"soc_min": 0.2},
        )
        decision = build_candidate_selector(
            {"policy": "shortest_path", "hard_gate_overrides": {"soc_min": 0.4}}
        ).select_action(
            state={"candidates": (candidate.to_dict(),), "soc_min": 0.3},
            request={},
        ).to_dict()

        self.assertEqual(decision["action"], "reject_drop")
        energy_gate = decision["score_terms"]["hard_gate_results"]["energy_soc"]
        self.assertEqual(energy_gate["soc_min"], 0.4)
        self.assertFalse(energy_gate["passed"])

    def test_scheduler_invalid_object_handling(self):
        with self.assertRaisesRegex(TypeError, "scheduler value must be"):
            build_candidate_selector(object())
        with self.assertRaisesRegex(TypeError, "scheduler value must be"):
            build_candidate_selector(type("NameOnly", (), {"name": "invalid"})())

    def test_candidate_selection_does_not_mutate_engine_state_view(self):
        candidate = SchedulerCandidate(
            candidate_id="candidate",
            action_type="local_compute",
            feasible=True,
            route_plan={"compute_satellite_id": "sat-a"},
            predicted_ttft_s=1.0,
            predicted_e2e_s=2.0,
            predicted_queue_wait_s=0.0,
            predicted_link_wait_s=0.0,
            predicted_energy_risk=0.0,
            predicted_soc_after=0.8,
            hop_count=0,
        )
        state_view = {"candidates": (candidate.to_dict(),), "soc_min": 0.2}
        before = deepcopy(state_view)

        build_candidate_selector("shortest_path").select_action(
            state=state_view,
            request={"request_id": "request"},
        )

        self.assertEqual(state_view, before)

    def test_performance_skeleton_keeps_observed_metrics_null_without_trace(self):
        scenario, strategy, serving_performance = _runtime_scenario(self.runtime)
        result = self.runtime.run_runtime_simulation(scenario).to_dict()

        self.assertFalse(result["evaluated"])
        self.assertEqual(result["module_status"]["simulation"], "skeleton")
        self.assertIsNone(result["runtime_metrics"]["completion_ratio"])
        self.assertIsNone(result["runtime_metrics"]["ttft_p95_s"])
        self.assertIsNone(result["runtime_metrics"]["e2e_p95_s"])
        self.assertNotEqual(
            result["runtime_metrics"]["completion_ratio"],
            serving_performance["request_throughput_per_s"],
        )
        self.assertTrue(result["warnings"])
        for group_name in (
            "session_metrics",
            "communication_metrics",
            "energy_metrics",
            "prediction_error",
            "non_separability",
        ):
            self.assertFalse(result[group_name]["evaluated"], group_name)
            self.assertIsNone(result[group_name]["value"], group_name)
        self.assertFalse(result["decision_trace"]["evaluated"])

    def test_workload_runtime_leakage_is_rejected(self):
        scenario, strategy, _performance = _runtime_scenario(
            self.runtime,
            requests=[
                _request(
                    assigned_satellite="sat-0",
                    completed=True,
                    completion_ratio=1.0,
                    observed_e2e_s=3.0,
                    cost_per_served_mtoken_usd=1.2,
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "assigned_satellite|completion_ratio|observed_e2e_s|runtime leakage"):
            self.runtime.run_runtime_simulation(scenario)

    def test_session_runtime_leakage_is_rejected(self):
        scenario, strategy, _performance = _runtime_scenario(
            self.runtime,
            sessions=[{"session_id": "session-0", "session_action": "sticky_kv_owner", "soc_before": 0.5}],
        )

        with self.assertRaisesRegex(ValueError, "session_action|soc_before|runtime leakage"):
            self.runtime.run_runtime_simulation(scenario)



if __name__ == "__main__":
    unittest.main()
