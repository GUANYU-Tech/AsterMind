import importlib
import sys
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.runtime.runtime_optimization import optimize_runtime_strategy
from orbitserve.runtime.schedulers import RuntimeScheduler, SchedulerCandidate, get_scheduler


ROOT = Path(__file__).resolve().parents[3]


def _optional_import(module_name):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None
        raise


def _to_dict(value):
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _runtime_scenario(runtime, *, requests=None, sessions=None):
    spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_single_design.json")
    experiment = run_experiment(spec)
    performance = experiment.performance.to_dict()
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
    deployment = runtime.ServingDeploymentView.from_value(performance)
    scenario = runtime.RuntimeScenario(
        scenario_id=spec.experiment_id,
        time_horizon_s=spec.simulation.duration_s,
        workload_bundle={"status": "loaded", "request_count": len(requests or [])},
        requests=tuple(requests or [_request()]),
        sessions=tuple(sessions or ()),
        simulation_trace_ref={"evaluated": False, "warnings": ["simulation is a skeleton"]},
        serving_deployments=(deployment,),
        runtime_config=strategy,
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
        "slo_class": "interactive_5s",
        "ttft_deadline_s": 5.0,
        "e2e_deadline_s": 30.0,
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

    def test_top_level_scheduler_exposes_runtime_bridge(self):
        scheduler = get_scheduler("orbitserve")
        decision = scheduler.select_action(state=None, request=None).to_dict()

        self.assertIsInstance(scheduler, RuntimeScheduler)
        self.assertEqual(decision["action"], "scheduler_bridge/runtime_strategy_preview")
        self.assertFalse(decision["evaluated"])
        self.assertEqual(decision["runtime_strategy"]["policy"], "orbitserve")
        self.assertIn("local_compute", decision["runtime_strategy"]["enabled_actions"])
        self.assertTrue(decision["optimization_support"]["runtime_strategy_available"])
        self.assertGreaterEqual(decision["optimization_support"]["candidate_strategy_count"], 5)
        self.assertIn("orbitserve", decision["optimization_support"]["candidate_policies"])
        self.assertTrue(decision["candidate_strategies"])
        self.assertEqual(
            decision["optimization_support"]["runtime_optimization_entrypoint"],
            "orbitserve.runtime_optimization.optimize_runtime_strategy",
        )

    def test_top_level_scheduler_selects_request_level_candidate(self):
        scheduler = get_scheduler("energy_aware")
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
            request={"request_id": "req-0", "ttft_deadline_s": 60.0, "e2e_deadline_s": 300.0},
        ).to_dict()

        self.assertTrue(decision["evaluated"])
        self.assertEqual(decision["action"], "cross_satellite_forward")
        self.assertEqual(decision["selected_candidate_id"], "offload")
        self.assertEqual(decision["candidate_count"], 2)
        self.assertEqual(decision["feasible_candidate_count"], 2)
        self.assertEqual(
            decision["score_terms"]["score_model"],
            "request_level_slo_aware_surrogate_not_full_shadow_kv_marginal_rule",
        )
        self.assertEqual(
            decision["optimization_support"]["paper_rule_status"],
            "request_level_slo_aware_surrogate_not_full_shadow_kv_marginal_rule",
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
        strict = get_scheduler(
            {"policy": "orbitserve", "hard_gate_mode": "strict", "weights": gate_mode_test_weights}
        ).select_action(
            state={"candidates": (violating_fast.to_dict(), safe_slow.to_dict())},
            request={"request_id": "req-0", "ttft_deadline_s": 60.0, "e2e_deadline_s": 300.0},
        ).to_dict()
        audit = get_scheduler(
            {"policy": "orbitserve", "hard_gate_mode": "audit_only", "weights": gate_mode_test_weights}
        ).select_action(
            state={"candidates": (violating_fast.to_dict(), safe_slow.to_dict())},
            request={"request_id": "req-0", "ttft_deadline_s": 60.0, "e2e_deadline_s": 300.0},
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

        decision = get_scheduler(
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

    def test_scheduler_hard_rejects_all_candidates_that_miss_deadline(self):
        def candidate(candidate_id, satellite_id, predicted_e2e_s):
            return SchedulerCandidate(
                candidate_id=candidate_id,
                action_type="cross_satellite_forward",
                feasible=True,
                route_plan={"compute_satellite_id": satellite_id},
                predicted_ttft_s=1.0,
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

        decision = get_scheduler(
            {"policy": "shortest_path", "weights": {"hop": 0.0}}
        ).select_action(
            state={
                "candidates": (
                    candidate("slow", "sat-a", 20.0).to_dict(),
                    candidate("fast", "sat-b", 10.0).to_dict(),
                )
            },
            request={"request_id": "req-deadline-gate", "e2e_deadline_s": 5.0},
        ).to_dict()

        self.assertEqual(decision["action"], "reject_drop")
        self.assertEqual(decision["feasible_candidate_count"], 0)
        self.assertEqual(
            decision["gate_results"]["violations_by_gate"]["predicted_e2e_deadline_miss"],
            2,
        )
        self.assertTrue(
            all(item["rejection_reason"] == "hard_gate_violation" for item in decision["rejected_candidates"])
        )
        self.assertTrue(
            all(item["score_terms"]["hard_gate_penalty"] == 0.0 for item in decision["rejected_candidates"])
        )

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

        decision = get_scheduler("shortest_path").select_action(
            state={"candidates": (ineligible.to_dict(), eligible.to_dict())},
            request={"request_id": "req-physical-gate"},
        ).to_dict()

        self.assertEqual(decision["selected_candidate_id"], "eligible-slow")
        rejected = decision["rejected_candidates"][0]
        self.assertEqual(rejected["rejection_reason"], "candidate_marked_infeasible")
        self.assertFalse(rejected["hard_gate_results"]["physical_eligibility"]["passed"])

    def test_scheduler_recomputes_stale_deadline_gate_results(self):
        stale = SchedulerCandidate(
            candidate_id="stale-pass",
            action_type="local_compute",
            feasible=True,
            route_plan={"compute_satellite_id": "sat-a"},
            predicted_ttft_s=1.0,
            predicted_e2e_s=20.0,
            predicted_queue_wait_s=0.0,
            predicted_link_wait_s=0.0,
            predicted_energy_risk=0.0,
            predicted_soc_after=0.8,
            hop_count=0,
            hard_gate_results={"slo_e2e": {"passed": True, "reason": None}},
            predicted_nominal_latency_s=20.0,
        )
        valid = SchedulerCandidate(
            candidate_id="valid",
            action_type="local_compute",
            feasible=True,
            route_plan={"compute_satellite_id": "sat-b"},
            predicted_ttft_s=1.0,
            predicted_e2e_s=4.0,
            predicted_queue_wait_s=2_000_000.0,
            predicted_link_wait_s=0.0,
            predicted_energy_risk=0.0,
            predicted_soc_after=0.8,
            hop_count=0,
            predicted_nominal_latency_s=4.0,
        )
        decision = get_scheduler(
            {
                "policy": "load_aware",
                "weights": {"latency": 0.0, "slack": 0.0, "queue": 1.0, "link": 0.0, "energy": 0.0, "hop": 0.0},
                "normalization_refs": {"queue_s": 1.0},
            }
        ).select_action(
            state={"candidates": (stale.to_dict(), valid.to_dict())},
            request={"e2e_deadline_s": 5.0},
        ).to_dict()

        self.assertEqual(decision["selected_candidate_id"], "valid")
        self.assertEqual(decision["feasible_candidate_count"], 1)
        rejected = decision["rejected_candidates"][0]
        self.assertIn("predicted_e2e_deadline_miss", rejected["hard_gate_violations"])
        self.assertFalse(rejected["hard_gate_results"]["slo_e2e"]["passed"])

    def test_scheduler_fail_closes_missing_and_nonfinite_deadline_metrics(self):
        candidates = tuple(
            SchedulerCandidate(
                candidate_id=candidate_id,
                action_type="local_compute",
                feasible=True,
                route_plan={"compute_satellite_id": candidate_id},
                predicted_ttft_s=1.0,
                predicted_e2e_s=predicted_e2e_s,
                predicted_queue_wait_s=0.0,
                predicted_link_wait_s=0.0,
                predicted_energy_risk=0.0,
                predicted_soc_after=0.8,
                hop_count=0,
                predicted_nominal_latency_s=predicted_e2e_s,
            ).to_dict()
            for candidate_id, predicted_e2e_s in (("missing", None), ("nan", float("nan")))
        )
        decision = get_scheduler("shortest_path").select_action(
            state={"candidates": candidates},
            request={"e2e_deadline_s": 5.0},
        ).to_dict()

        self.assertEqual(decision["action"], "reject_drop")
        self.assertEqual(decision["feasible_candidate_count"], 0)
        self.assertEqual(
            decision["gate_results"]["violations_by_gate"]["predicted_e2e_unavailable"],
            2,
        )

    def test_scheduler_resolves_symbolic_shared_normalization_refs(self):
        candidates = tuple(
            SchedulerCandidate(
                candidate_id=candidate_id,
                action_type="local_compute",
                feasible=True,
                route_plan={"compute_satellite_id": candidate_id},
                predicted_ttft_s=1.0,
                predicted_e2e_s=e2e,
                predicted_queue_wait_s=queue,
                predicted_link_wait_s=0.0,
                predicted_energy_risk=0.0,
                predicted_soc_after=0.8,
                hop_count=0,
                predicted_nominal_latency_s=e2e,
            ).to_dict()
            for candidate_id, e2e, queue in (("slow", 20.0, 2.0), ("fast", 10.0, 1.0))
        )
        decision = get_scheduler(
            {
                "policy": "shortest_path",
                "hard_gate_mode": "audit_only",
                "normalization_refs": {"latency_s": "e2e_deadline_s", "queue_s": "candidate_max"},
            }
        ).select_action(
            state={"candidates": candidates},
            request={"e2e_deadline_s": 5.0},
        ).to_dict()

        self.assertEqual(decision["selected_candidate_id"], "fast")
        self.assertEqual(decision["score_terms"]["normalization_refs"]["latency_s"], 5.0)
        self.assertEqual(decision["score_terms"]["normalization_refs"]["queue_s"], 2.0)
        self.assertEqual(decision["score_terms"]["normalized_e2e"], 2.0)

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
        decision = get_scheduler("shortest_path").select_action(
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
        decision = get_scheduler("shortest_path").select_action(
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
        decision = get_scheduler(
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
            get_scheduler(object())
        with self.assertRaisesRegex(TypeError, "policy field"):
            get_scheduler(type("NameOnly", (), {"name": "invalid"})())

    def test_scheduler_can_disable_closed_loop_runtime_optimization(self):
        scheduler = get_scheduler({"policy": "nearest_visible", "runtime_optimization_enabled": False})
        search_space = scheduler.optimization_search_space()

        self.assertFalse(search_space["enabled"])
        self.assertEqual(search_space["candidate_schedulers"], ())
        result = optimize_runtime_strategy(object(), search_space=search_space).to_dict()
        self.assertFalse(result["evaluated"])
        self.assertEqual(result["candidate_count"], 0)
        self.assertFalse(result["scheduler_search_space"]["enabled"])

    def test_performance_skeleton_keeps_observed_metrics_null_without_trace(self):
        scenario, strategy, serving_performance = _runtime_scenario(self.runtime)
        result = self.runtime.evaluate_runtime_performance(scenario, strategy).to_dict()

        self.assertFalse(result["evaluated"])
        self.assertEqual(result["module_status"]["simulation"], "skeleton")
        self.assertIsNone(result["runtime_metrics"]["served_ratio"])
        self.assertIsNone(result["runtime_metrics"]["ttft_p95_s"])
        self.assertIsNone(result["runtime_metrics"]["e2e_p95_s"])
        self.assertIsNone(serving_performance["request_throughput_per_s"])
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
                    served=True,
                    served_ratio=1.0,
                    observed_e2e_s=3.0,
                    cost_per_served_mtoken_usd=1.2,
                )
            ],
        )

        with self.assertRaisesRegex(ValueError, "assigned_satellite|served_ratio|observed_e2e_s|runtime leakage"):
            self.runtime.evaluate_runtime_performance(scenario, strategy)

    def test_session_runtime_leakage_is_rejected(self):
        scenario, strategy, _performance = _runtime_scenario(
            self.runtime,
            sessions=[{"session_id": "session-0", "session_action": "sticky_kv_owner", "soc_before": 0.5}],
        )

        with self.assertRaisesRegex(ValueError, "session_action|soc_before|runtime leakage"):
            self.runtime.evaluate_runtime_performance(scenario, strategy)

    def test_runner_runtime_details_use_materialized_workload_when_integration_exists(self):
        payload = run_experiment(
            load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_single_design.json")
        ).to_dict()
        details = payload["performance"].get("details", {})
        if "runtime" not in details or "runtime_optimization" not in details:
            self.skipTest("runner runtime integration is not available yet")

        self.assertFalse(details["runtime"]["evaluated"])
        self.assertTrue(details["runtime"]["workload"]["configured"])
        self.assertIsNone(payload["performance"]["ttft_p95_s"])
        self.assertIsNone(payload["performance"]["e2e_p95_s"])
        self.assertFalse(details["runtime_optimization"]["evaluated"])
        optimizer = details["runtime_optimization"]
        self.assertIsNone(optimizer["selected_strategy"])
        self.assertGreater(len(optimizer["score_table"]), 0)
        for row in optimizer["score_table"]:
            self.assertIsNone(row["objective_score"])
            self.assertFalse(row["evaluator_result"]["evaluated"])
        row_by_policy = {row["scheduler"]["policy"]: row["scheduler"] for row in optimizer["score_table"]}
        self.assertIn("local_compute", row_by_policy["orbitserve"]["enabled_actions"])
        self.assertIn("defer", row_by_policy["orbitserve"]["enabled_actions"])
        self.assertEqual(row_by_policy["orbitserve"]["path_policy"], "k_shortest_score")
        self.assertEqual(row_by_policy["local_only"]["enabled_actions"], ["local_compute", "reject_drop"])
        self.assertEqual(row_by_policy["local_only"]["path_policy"], "nearest")
        self.assertEqual(row_by_policy["shortest_path"]["path_policy"], "shortest_path")

    def test_runner_preserves_extended_scheduler_fields_in_runtime_strategy(self):
        base = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_single_design.json")
        spec = replace(
            base,
            scheduler=replace(
                base.scheduler,
                policy="orbitserve",
                lookahead_orbits=1.5,
                h_lookahead_s=8550.0,
                enabled_actions=("local_compute", "ground_fallback", "reject_drop"),
                fallback_enabled=True,
                defer_enabled=True,
                weights={"latency": 1.0, "energy": 0.25, "queue": 1.0, "fallback": 10.0},
                normalization_refs={"latency_s": "e2e_deadline_s"},
            ),
        )
        payload = run_experiment(spec).to_dict()
        strategy = payload["performance"]["details"]["runtime"]["strategy"]

        self.assertEqual(strategy["policy"], "orbitserve")
        self.assertEqual(strategy["lookahead_orbits"], 1.5)
        self.assertEqual(strategy["h_lookahead_s"], 8550.0)
        self.assertEqual(strategy["enabled_actions"], ["local_compute", "ground_fallback", "reject_drop"])
        self.assertTrue(strategy["fallback_enabled"])
        self.assertTrue(strategy["defer_enabled"])
        self.assertEqual(strategy["weights"]["fallback"], 10.0)
        self.assertEqual(strategy["normalization_refs"]["latency_s"], "e2e_deadline_s")
        self.assertEqual(payload["module_status"]["scheduling"], "runtime_bridge")

    def test_plain_runner_scheduler_uses_runtime_policy_defaults(self):
        payload = run_experiment(
            load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_single_design.json")
        ).to_dict()
        strategy = payload["performance"]["details"]["runtime"]["strategy"]
        optimizer = payload["performance"]["details"]["runtime_optimization"]

        self.assertEqual(strategy["policy"], "orbitserve")
        self.assertIn("local_compute", strategy["enabled_actions"])
        self.assertIn("ground_fallback", strategy["enabled_actions"])
        self.assertIn("sticky_kv_owner", strategy["session_actions_enabled"])
        self.assertEqual(strategy["path_policy"], "k_shortest_score")
        self.assertEqual(payload["module_status"]["scheduling"], "runtime_bridge")
        self.assertEqual(
            optimizer["scheduler_search_space"]["provenance"]["source"],
            "scheduler_runtime_bridge",
        )
        self.assertEqual(
            optimizer["scheduler_search_space"]["current_scheduler"]["policy"],
            "orbitserve",
        )
        self.assertGreaterEqual(
            optimizer["scheduler_search_space"]["provenance"]["candidate_strategy_count"],
            5,
        )
        self.assertTrue(optimizer["scheduler_search_space"]["candidate_schedulers"])


class RuntimeOptimizationSkeletonTests(unittest.TestCase):
    def test_optimizer_keeps_scores_null_for_unevaluated_evaluator_results(self):
        runtime = _optional_import("orbitserve.runtime")
        optimization = _optional_import("orbitserve.runtime.runtime_optimization")
        if runtime is None or optimization is None:
            self.skipTest("runtime optimization modules are not available yet")

        scenario, _strategy, _performance = _runtime_scenario(runtime)
        strategies = [
            runtime.RuntimeStrategySpec.from_dict({"policy": policy})
            for policy in ("local_only", "shortest_path", "energy_aware", "orbitserve")
        ]

        class UnevaluatedEvaluator:
            def evaluate(self, scenario, strategy):
                return {
                    "evaluated": False,
                    "runtime_metrics": {"served_ratio": None, "ttft_p95_s": None, "e2e_p95_s": None},
                    "warnings": ["simulation trace is not implemented"],
                }

        result = optimization.optimize_runtime_strategy(
            scenario=scenario,
            search_space=strategies,
            evaluator=UnevaluatedEvaluator(),
        )
        payload = _to_dict(result)

        self.assertFalse(payload["evaluated"])
        self.assertEqual(len(payload["score_table"]), len(strategies))
        self.assertEqual(
            [row["scheduler"]["policy"] for row in payload["score_table"]],
            ["local_only", "shortest_path", "energy_aware", "orbitserve"],
        )
        for row in payload["score_table"]:
            self.assertIsNone(row["objective_score"])
            self.assertFalse(row["evaluator_result"]["evaluated"])
            self.assertIsNone(row["evaluator_result"]["slo"]["served_ratio"])
            self.assertIsNone(row["evaluator_result"]["performance"]["ttft_p95_s"])
            self.assertIsNone(row["evaluator_result"]["performance"]["e2e_p95_s"])
        self.assertIn(payload.get("selected_strategy"), (None, {}))


if __name__ == "__main__":
    unittest.main()
