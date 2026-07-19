import sys
import unittest
from pathlib import Path
from unittest.mock import patch


from orbitserve.runtime.routing import (
    TraceQueryResult,
    enumerate_compute_candidates,
    find_access_candidates,
    make_route_plan,
)
from orbitserve.runtime.contracts import RuntimeScenario, RuntimeStrategySpec, ServingDeploymentView
from orbitserve.runtime.evaluation import evaluate_runtime_performance
from orbitserve.runtime.deployment_contracts import RuntimeDeployment, RuntimeDeploymentCatalog
from orbitserve.runtime.runtime_eligibility import build_compute_eligibility_set
from orbitserve.runtime.engines.trace_backed import run_trace_backed_runtime
from orbitserve.runtime.runtime_metrics import aggregate_runtime_records
from orbitserve.orbit.simulation_traces import TRACE_BACKED_RUNTIME_SURROGATE_SCOPE


def _request(arrival_time_s=0.0, **overrides):
    row = {
        "request_id": "req-0",
        "arrival_time_s": arrival_time_s,
        "ingress_id": "ingress",
        "prompt_tokens": 1,
        "output_tokens": 1,
        "input_bytes": 1000.0,
        "output_bytes": 1000.0,
        "ttft_deadline_s": 100.0,
        "e2e_deadline_s": 200.0,
    }
    row.update(overrides)
    return row


def _deployment_catalog(prefill_tokens_per_s=300.0, decode_tokens_per_s=300.0, member_satellites=("sat-a", "sat-b", "sat-c")):
    return RuntimeDeploymentCatalog(
        deployments=(
            RuntimeDeployment(
                deployment_id="deployment_0",
                member_satellites=tuple(member_satellites),
                prefill_tokens_per_s=prefill_tokens_per_s,
                decode_tokens_per_s=decode_tokens_per_s,
            ),
        )
    )


class FakeTemporalTraceBundle:
    def __init__(
        self,
        *,
        access_end_s=100.0,
        horizon_s=100.0,
        step_s=10.0,
        active_by_time=None,
        stale_access=False,
        missing_link_samples=(),
        link_edge_available=None,
        dual_access=False,
        active_graph_unavailable=False,
        soc_min_by_sat=None,
    ):
        self.trace_model = "surrogate_orbit_trace"
        self.time = {"duration_s": horizon_s, "step_s": step_s}
        self.warnings = ()
        self.satellite_catalog = (
            {"satellite_id": "sat-a"},
            {"satellite_id": "sat-b"},
            {"satellite_id": "sat-c"},
        )
        self.access_windows = [
            {
                "ingress_id": "ingress",
                "satellite_id": "sat-a",
                "window_id": "access-a",
                "start_s": 0.0,
                "end_s": access_end_s,
                "sgl_link_id": "sgl:ingress:sat-a",
                "propagation_delay_s": 0.001,
                "bandwidth_bps": 1_000_000_000.0,
            },
        ]
        links = [
            _link("sgl:ingress:sat-a", "sgl", "ingress", "sat-a", 0.001),
            _link("isl:sat-a:sat-b", "isl", "sat-a", "sat-b", 0.010),
            _link("isl:sat-a:sat-c", "isl", "sat-a", "sat-c", 0.011),
            _link("isl:sat-c:sat-b", "isl", "sat-c", "sat-b", 0.012),
        ]
        if dual_access:
            self.access_windows.append(
                {
                    "ingress_id": "ingress",
                    "satellite_id": "sat-b",
                    "window_id": "access-b",
                    "start_s": 0.0,
                    "end_s": access_end_s,
                    "sgl_link_id": "sgl:ingress:sat-b",
                    "propagation_delay_s": 0.050,
                    "bandwidth_bps": 1_000_000_000.0,
                }
            )
            links.append(_link("sgl:ingress:sat-b", "sgl", "ingress", "sat-b", 0.050))
        self.access_windows = tuple(self.access_windows)
        self.links = tuple(links)
        self.physical_links = self.links
        self._links_by_id = {row["link_id"]: row for row in self.links}
        self._active_by_time = active_by_time or {
            0.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-b"),
            20.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-c", "isl:sat-c:sat-b"),
            40.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-c", "isl:sat-c:sat-b"),
        }
        if dual_access and active_by_time is None:
            self._active_by_time = {
                time_s: tuple(dict.fromkeys((*link_ids, "sgl:ingress:sat-b")))
                for time_s, link_ids in self._active_by_time.items()
            }
        self._stale_access = stale_access
        self._missing_link_samples = set(missing_link_samples)
        self._link_edge_available = dict(link_edge_available or {})
        self._active_graph_unavailable = active_graph_unavailable
        self.satellite_initial_soc = {"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.8}
        soc_min_by_sat = dict(soc_min_by_sat or {})
        self.energy_windows = tuple(
            {
                "satellite_id": sat["satellite_id"],
                "start_s": 0.0,
                "end_s": horizon_s,
                "sunlit": True,
                "dvfs_multiplier": 1.0,
                "charge_rate_w": 0.0,
                "idle_power_w": 0.0,
                "compute_energy_j_per_token": 0.0,
                "link_energy_j_per_byte": 0.0,
                "battery_capacity_j": 10_000_000.0,
                "soc_min": soc_min_by_sat.get(str(sat["satellite_id"]), 0.2),
                "energy_accounting_mode": "event_incremental_v1",
            }
            for sat in self.satellite_catalog
        )

    def access_at(self, ingress_id, time_s):
        if self._stale_access:
            return TraceQueryResult(
                available=False,
                reason="stale_trace_sample:access",
                sample_time_s=0.0,
                stale=True,
                warnings=("stale_trace_sample:access",),
            )
        windows = tuple(
            window
            for window in self.access_windows
            if window["ingress_id"] == ingress_id and window["start_s"] <= time_s < window["end_s"]
        )
        return TraceQueryResult(
            available=bool(windows),
            reason=None if windows else "no_access_window",
            sample_time_s=time_s,
            value=windows,
        )

    def next_access_after(self, ingress_id, time_s):
        windows = tuple(
            window
            for window in self.access_windows
            if window["ingress_id"] == ingress_id and window["start_s"] >= time_s
        )
        return TraceQueryResult(
            available=bool(windows),
            reason=None if windows else "no_next_access_window",
            sample_time_s=time_s,
            value=windows,
        )

    def active_links_at(self, time_s):
        if self._active_graph_unavailable:
            return TraceQueryResult(
                available=False,
                reason="active_graph_unavailable",
                sample_time_s=time_s,
                warnings=("active_graph_unavailable",),
            )
        sample_time_s = self._sample_time(time_s)
        if sample_time_s is None:
            return TraceQueryResult(available=False, reason="active_graph_unavailable")
        active_ids = self._active_by_time.get(sample_time_s, ())
        return TraceQueryResult(
            available=True,
            sample_time_s=sample_time_s,
            value={
                "active_link_ids": active_ids,
                "links": tuple(self._links_by_id[link_id] for link_id in active_ids),
            },
        )

    def link_sample_at(self, link_id, time_s):
        if link_id in self._missing_link_samples:
            return TraceQueryResult(
                available=False,
                reason="link_sample_missing",
                sample_time_s=time_s,
                warnings=("link_sample_missing",),
            )
        sample_time_s = self._sample_time(time_s)
        if sample_time_s is None:
            return TraceQueryResult(available=False, reason="link_sample_unavailable")
        link = self._links_by_id[link_id]
        edge_available = self._link_edge_available.get(
            link_id,
            link_id in self._active_by_time.get(sample_time_s, ()),
        )
        sample = {
            "time_s": sample_time_s,
            "link_id": link_id,
            "B_raw_bps": link["bandwidth_bps"],
            "tau_prop_s": link["propagation_delay_s"],
            "tau_setup_s": link["setup_delay_s"],
            "edge_available_external": edge_available,
            "unavailable_reason": None if edge_available else "inactive_in_active_graph",
        }
        return TraceQueryResult(
            available=edge_available,
            reason=None if edge_available else "inactive_in_active_graph",
            sample_time_s=sample_time_s,
            value=sample,
        )

    def _sample_time(self, time_s):
        candidates = [sample_time for sample_time in self._active_by_time if sample_time <= time_s]
        if not candidates:
            return None
        return max(candidates)


def _link(link_id, kind, endpoint_a, endpoint_b, propagation_delay_s):
    return {
        "link_id": link_id,
        "kind": kind,
        "endpoint_a": endpoint_a,
        "endpoint_b": endpoint_b,
        "bandwidth_bps": 1_000_000_000.0,
        "propagation_delay_s": propagation_delay_s,
        "setup_delay_s": 0.0,
        "bidirectional": True,
        "edge_available_external": True,
    }


class TemporalRoutingRuntimeTests(unittest.TestCase):
    def test_compute_eligibility_combines_deployment_hardware_fault_and_energy_without_capacity_boost(self):
        deployment = RuntimeDeployment(
            deployment_id="deployment_0",
            member_satellites=("sat-a", "sat-b", "sat-c"),
        )
        eligibility = build_compute_eligibility_set(
            deployment=deployment,
            satellite_catalog=(
                {"satellite_id": "sat-a", "hardware_role": "compute"},
                {"satellite_id": "sat-b", "hardware_role": "relay"},
                {"satellite_id": "sat-c", "hardware_role": "compute", "failed": True},
                {"satellite_id": "sat-d", "hardware_role": "relay"},
            ),
            runtime_exclusions={"sat-a": "energy_state_compute_ineligible"},
            evaluated_at_s=1.0,
        )

        self.assertEqual(eligibility.capacity_member_ids, ("sat-a", "sat-c"))
        self.assertEqual(eligibility.eligible_compute_satellite_ids, ())
        self.assertIn("hardware_role_not_compute_capable", eligibility.excluded_reasons["sat-b"])
        self.assertIn("satellite_fault_or_unavailable", eligibility.excluded_reasons["sat-c"])
        self.assertIn("sat-d", eligibility.relay_satellite_ids)

    def test_active_graph_sample_changes_path(self):
        bundle = FakeTemporalTraceBundle()

        access0 = find_access_candidates(bundle, "ingress", 0.0)
        rows0 = enumerate_compute_candidates(bundle, access0, "shortest_path", max_hop_count=2, time_s=0.0)
        path0 = _path_to(rows0, "sat-b")

        access20 = find_access_candidates(bundle, "ingress", 20.0)
        rows20 = enumerate_compute_candidates(bundle, access20, "shortest_path", max_hop_count=2, time_s=20.0)
        path20 = _path_to(rows20, "sat-b")

        self.assertEqual(path0, ("isl:sat-a:sat-b",))
        self.assertEqual(path20, ("isl:sat-a:sat-c", "isl:sat-c:sat-b"))

    def test_inactive_link_does_not_enter_selected_route(self):
        bundle = FakeTemporalTraceBundle()
        access20 = find_access_candidates(bundle, "ingress", 20.0)
        rows20 = enumerate_compute_candidates(bundle, access20, "shortest_path", max_hop_count=2, time_s=20.0)
        access, compute, path = next(row for row in rows20 if row[1] == "sat-b")
        plan = make_route_plan(
            trace_bundle=bundle,
            access_candidate=access,
            compute_satellite_id=compute,
            isl_link_ids=path,
            input_bytes=1000.0,
            output_bytes=1000.0,
            time_s=20.0,
        )

        selected_ids = [leg.link_id for leg in (*plan.inbound_legs, *plan.outbound_legs)]
        self.assertNotIn("isl:sat-a:sat-b", selected_ids)

    def test_request_outside_trace_horizon_rejects(self):
        bundle = FakeTemporalTraceBundle(horizon_s=100.0)
        result = run_trace_backed_runtime(
            requests=(_request(100.0),),
            deployment_catalog=_deployment_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "nearest_visible"}),
            trace_bundle=bundle,
        )

        self.assertEqual(result.records[0].reject_reason, "request_time_outside_trace_horizon")
        self.assertEqual(result.records[0].bottleneck_reason, "request_time_outside_trace_horizon")

    def test_scheduler_hard_gate_miss_rejects_without_feasible_candidate(self):
        bundle = FakeTemporalTraceBundle()
        result = run_trace_backed_runtime(
            requests=(_request(0.0, ttft_deadline_s=0.000001, e2e_deadline_s=0.000001),),
            deployment_catalog=_deployment_catalog(prefill_tokens_per_s=1.0, decode_tokens_per_s=1.0),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "orbitserve", "max_hop_count": 2}),
            trace_bundle=bundle,
        )

        record = result.records[0]
        trace = result.decision_traces[0]
        self.assertTrue(record.rejected)
        self.assertEqual(record.reject_reason, "no_feasible_scheduler_candidate")
        self.assertEqual(record.provenance["scheduler_decision"]["feasible_candidate_count"], 0)
        self.assertEqual(trace["feasible_action_count"], 0)
        self.assertIsNone(trace["selected_action"])
        self.assertIn("selected_candidate", trace)

    def test_soc_hard_gate_rejects_low_soc_candidate_in_strict_mode(self):
        bundle = FakeTemporalTraceBundle(soc_min_by_sat={"sat-a": 0.95})
        result = run_trace_backed_runtime(
            requests=(_request(0.0),),
            deployment_catalog=_deployment_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "nearest_visible"}),
            trace_bundle=bundle,
        )

        record = result.records[0]
        trace = result.decision_traces[0]
        self.assertTrue(record.rejected)
        self.assertEqual(record.reject_reason, "no_feasible_scheduler_candidate")
        self.assertEqual(record.provenance["gate_results"]["violations_by_gate"]["energy_soc"], 1)
        self.assertIn("energy_soc", record.provenance["rejected_candidates"][0]["hard_gate_violations"])
        self.assertEqual(trace["gate_results"]["violations_by_gate"]["energy_soc"], 1)
        self.assertEqual(trace["rejected_actions"][0]["rejection_reason"], "hard_gate_violation")

    def test_trace_horizon_drives_evaluator_throughput_when_config_mismatches(self):
        bundle = FakeTemporalTraceBundle(horizon_s=100.0)
        scenario = RuntimeScenario(
            scenario_id="trace-horizon-unit",
            time_horizon_s=10.0,
            workload_bundle={"status": "loaded", "request_count": 1},
            requests=(_request(0.0),),
            simulation_trace_ref={
                "evaluated": True,
                "trace_model": "surrogate_orbit_trace",
                "trace_manifest": {
                    "evaluated": True,
                    "trace_model": "surrogate_orbit_trace",
                    "scope": TRACE_BACKED_RUNTIME_SURROGATE_SCOPE,
                    "time": {"duration_s": 100.0, "step_s": 10.0},
                    "valid_for_paper_main_results": False,
                },
            },
            serving_deployments=(
                ServingDeploymentView.from_inputs(
                    deployment_id="deployment_0",
                    throughput_estimate={
                        "evaluated": True,
                        "feasible": True,
                        "prefill_tokens_per_s": 300.0,
                        "decode_tokens_per_s": 300.0,
                        "member_satellites": ("sat-a",),
                    },
                ),
            ),
            runtime_config=RuntimeStrategySpec.from_dict({"policy": "nearest_visible"}),
        )

        with patch("orbitserve.runtime.runtime_evaluator.load_surrogate_trace_bundle", return_value=bundle):
            payload = evaluate_runtime_performance(scenario).to_dict()

        self.assertTrue(payload["evaluated"])
        self.assertEqual(payload["runtime_metrics"]["served_request_count"], 1)
        self.assertAlmostEqual(payload["runtime_metrics"]["useful_request_throughput_per_s"], 1.0 / 100.0)
        self.assertIn("trace_horizon_mismatch", payload["warnings"])
        self.assertEqual(payload["provenance"]["runtime_engine"]["trace_horizon_s"], 100.0)

    def test_energy_gap_rejects_without_default_window(self):
        bundle = FakeTemporalTraceBundle()
        bundle.energy_windows = ()
        result = run_trace_backed_runtime(
            requests=(_request(0.0),),
            deployment_catalog=_deployment_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "nearest_visible",
                    "fallback_enabled": True,
                    "hard_gate_overrides": {"ground_fallback_latency_s": 1.0},
                }
            ),
            trace_bundle=bundle,
        )

        record = result.records[0]
        self.assertTrue(record.rejected)
        self.assertFalse(record.fallback_used)
        self.assertEqual(record.reject_reason, "no_energy_window")
        self.assertIn("no_energy_window", record.provenance["warnings"])
        self.assertIn("no_energy_window", result.decision_traces[0]["warnings"])

    def test_dynamic_sgl_missing_sample_rejects_without_static_fallback(self):
        bundle = FakeTemporalTraceBundle(missing_link_samples=("sgl:ingress:sat-a",))
        result = run_trace_backed_runtime(
            requests=(_request(0.0),),
            deployment_catalog=_deployment_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "nearest_visible",
                    "fallback_enabled": True,
                    "hard_gate_overrides": {"ground_fallback_latency_s": 1.0},
                }
            ),
            trace_bundle=bundle,
        )

        record = result.records[0]
        self.assertTrue(record.rejected)
        self.assertFalse(record.fallback_used)
        self.assertEqual(record.reject_reason, "link_sample_missing")
        self.assertIn("link_sample_missing", record.provenance["warnings"])
        self.assertIn("link_sample_missing", result.decision_traces[0]["warnings"])

    def test_active_graph_unavailable_rejects_without_ground_fallback(self):
        bundle = FakeTemporalTraceBundle(active_graph_unavailable=True)
        result = run_trace_backed_runtime(
            requests=(_request(0.0),),
            deployment_catalog=_deployment_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "shortest_path",
                    "max_hop_count": 2,
                    "fallback_enabled": True,
                    "hard_gate_overrides": {"ground_fallback_latency_s": 1.0},
                }
            ),
            trace_bundle=bundle,
        )

        record = result.records[0]
        self.assertTrue(record.rejected)
        self.assertFalse(record.fallback_used)
        self.assertEqual(record.reject_reason, "active_graph_unavailable")
        self.assertIn("active_graph_unavailable", record.provenance["warnings"])
        self.assertIn("active_graph_unavailable", result.decision_traces[0]["warnings"])

    def test_ground_fallback_ttft_uses_observed_latency_for_deadline_check(self):
        result = run_trace_backed_runtime(
            requests=(
                _request(
                    0.0,
                    ttft_deadline_s=1.0,
                    e2e_deadline_s=10.0,
                ),
            ),
            deployment_catalog=_deployment_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "nearest_visible",
                    "fallback_enabled": True,
                    "hard_gate_overrides": {"ground_fallback_latency_s": 5.0},
                }
            ),
            trace_bundle=FakeTemporalTraceBundle(access_end_s=0.0),
        )

        record = result.records[0]
        self.assertTrue(record.fallback_used)
        self.assertTrue(record.completed)
        self.assertFalse(record.served)
        self.assertTrue(record.timeout)
        self.assertEqual(record.timeout_reason, "ttft_deadline_miss")
        self.assertEqual(record.ttft_s, 5.0)
        self.assertEqual(record.e2e_s, 5.0)


    def test_ground_fallback_deadline_corner_cases(self):
        cases = (
            {
                "name": "exactly_on_both_deadlines_is_served",
                "latency_s": 5.0,
                "ttft_deadline_s": 5.0,
                "e2e_deadline_s": 5.0,
                "served": True,
                "timeout": False,
                "timeout_reason": None,
            },
            {
                "name": "e2e_only_miss",
                "latency_s": 5.0,
                "ttft_deadline_s": 10.0,
                "e2e_deadline_s": 4.0,
                "served": False,
                "timeout": True,
                "timeout_reason": "e2e_deadline_miss",
            },
            {
                "name": "both_deadlines_miss",
                "latency_s": 5.0,
                "ttft_deadline_s": 1.0,
                "e2e_deadline_s": 4.0,
                "served": False,
                "timeout": True,
                "timeout_reason": "ttft_and_e2e_deadline_miss",
            },
            {
                "name": "missing_ttft_deadline_uses_e2e_only",
                "latency_s": 5.0,
                "ttft_deadline_s": None,
                "e2e_deadline_s": 10.0,
                "served": True,
                "timeout": False,
                "timeout_reason": None,
            },
            {
                "name": "missing_e2e_deadline_still_checks_ttft",
                "latency_s": 5.0,
                "ttft_deadline_s": 1.0,
                "e2e_deadline_s": None,
                "served": False,
                "timeout": True,
                "timeout_reason": "ttft_deadline_miss",
            },
        )
        for case in cases:
            with self.subTest(case["name"]):
                result = run_trace_backed_runtime(
                    requests=(
                        _request(
                            0.0,
                            ttft_deadline_s=case["ttft_deadline_s"],
                            e2e_deadline_s=case["e2e_deadline_s"],
                        ),
                    ),
                    deployment_catalog=_deployment_catalog(),
                    runtime_strategy=RuntimeStrategySpec.from_dict(
                        {
                            "policy": "nearest_visible",
                            "fallback_enabled": True,
                            "hard_gate_overrides": {"ground_fallback_latency_s": case["latency_s"]},
                        }
                    ),
                    trace_bundle=FakeTemporalTraceBundle(access_end_s=0.0),
                )

                record = result.records[0]
                self.assertTrue(record.fallback_used)
                self.assertTrue(record.completed)
                self.assertEqual(record.ttft_s, case["latency_s"])
                self.assertEqual(record.e2e_s, case["latency_s"])
                self.assertEqual(record.served, case["served"])
                self.assertEqual(record.timeout, case["timeout"])
                self.assertEqual(record.timeout_reason, case["timeout_reason"])

    def test_ground_fallback_missing_latency_is_not_served(self):
        result = run_trace_backed_runtime(
            requests=(_request(0.0),),
            deployment_catalog=_deployment_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "nearest_visible",
                    "fallback_enabled": True,
                }
            ),
            trace_bundle=FakeTemporalTraceBundle(access_end_s=0.0),
        )

        record = result.records[0]
        self.assertTrue(record.fallback_used)
        self.assertFalse(record.completed)
        self.assertFalse(record.served)
        self.assertFalse(record.timeout)
        self.assertEqual(record.reject_reason, "no_access_window")
        self.assertIsNone(record.ttft_s)
        self.assertIsNone(record.e2e_s)

    def test_load_aware_policy_uses_queue_signal_unlike_shortest_path(self):
        requests = (
            _request(0.0, request_id="req-0", prompt_tokens=9, output_tokens=9),
            _request(0.0, request_id="req-1", prompt_tokens=9, output_tokens=9),
        )
        catalog = _deployment_catalog(prefill_tokens_per_s=3.0, decode_tokens_per_s=3.0)

        shortest = run_trace_backed_runtime(
            requests=requests,
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=FakeTemporalTraceBundle(dual_access=True),
        )
        load_aware = run_trace_backed_runtime(
            requests=requests,
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "load_aware", "max_hop_count": 2}),
            trace_bundle=FakeTemporalTraceBundle(dual_access=True),
        )

        self.assertEqual(shortest.records[0].assigned_satellite_id, "sat-a")
        self.assertEqual(shortest.records[1].assigned_satellite_id, "sat-a")
        self.assertEqual(load_aware.records[0].assigned_satellite_id, "sat-a")
        self.assertNotEqual(load_aware.records[1].assigned_satellite_id, "sat-a")
        self.assertEqual(shortest.records[1].provenance["selection_weights"]["queue"], 0.0)
        self.assertGreater(load_aware.records[1].provenance["selection_weights"]["queue"], 0.0)
        self.assertTrue(load_aware.records[1].provenance["scheduler_decision"]["evaluated"])
        self.assertEqual(
            load_aware.records[1].provenance["scheduler_decision"]["selected_candidate_id"],
            load_aware.records[1].provenance["scheduler_decision"]["selected_action"]["candidate_id"],
        )
        self.assertEqual(
            load_aware.records[1].provenance["score_terms"]["score_model"],
            "request_level_slo_aware_surrogate_not_full_shadow_kv_marginal_rule",
        )
        self.assertIn("compute_satellite_id", load_aware.decision_traces[1]["selected_action"])
        self.assertIn("route_plan", load_aware.decision_traces[1]["selected_candidate"])

    def test_compute_targets_and_capacity_are_limited_to_deployment_members(self):
        requests = (
            _request(0.0, request_id="req-0", prompt_tokens=9, output_tokens=1),
            _request(0.0, request_id="req-1", prompt_tokens=9, output_tokens=1),
        )
        result = run_trace_backed_runtime(
            requests=requests,
            deployment_catalog=_deployment_catalog(
                prefill_tokens_per_s=3.0,
                decode_tokens_per_s=3.0,
                member_satellites=("sat-a",),
            ),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "load_aware", "max_hop_count": 2}),
            trace_bundle=FakeTemporalTraceBundle(dual_access=True),
        )

        self.assertEqual([record.compute_owner_satellite_id for record in result.records], ["sat-a", "sat-a"])
        self.assertAlmostEqual(result.records[0].prefill_service_s or 0.0, 3.0)
        self.assertEqual(
            result.records[0].provenance["compute_eligibility"]["eligible_compute_satellite_ids"],
            ["sat-a"],
        )

    def test_relay_node_is_not_confused_with_compute_owner(self):
        bundle = FakeTemporalTraceBundle(
            active_by_time={
                0.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-c", "isl:sat-c:sat-b"),
                20.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-c", "isl:sat-c:sat-b"),
                40.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-c", "isl:sat-c:sat-b"),
            }
        )
        result = run_trace_backed_runtime(
            requests=(_request(0.0),),
            deployment_catalog=_deployment_catalog(member_satellites=("sat-b",)),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=bundle,
        )

        record = result.records[0]
        self.assertEqual(record.compute_owner_satellite_id, "sat-b")
        self.assertIn("sat-c", record.route_node_ids)
        self.assertNotIn("sat-c", record.compute_owner_satellite_ids)

    def test_multihop_energy_is_charged_to_access_relay_and_compute_endpoints(self):
        bundle = FakeTemporalTraceBundle(
            active_by_time={
                0.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-c", "isl:sat-c:sat-b"),
                20.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-c", "isl:sat-c:sat-b"),
                40.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-c", "isl:sat-c:sat-b"),
            }
        )
        _set_satellite_energy(
            bundle,
            initial_soc={"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.8},
            per_satellite={
                satellite_id: {
                    "battery_capacity_j": 1_000_000.0,
                    "compute_energy_j_per_token": 0.0,
                    "link_energy_j_per_byte": 1.0,
                }
                for satellite_id in ("sat-a", "sat-b", "sat-c")
            },
        )

        record = run_trace_backed_runtime(
            requests=(_request(0.0, input_bytes=10.0, output_bytes=10.0),),
            deployment_catalog=_deployment_catalog(member_satellites=("sat-b",)),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=bundle,
        ).records[0]

        link_j_by_satellite = {}
        link_bytes_by_satellite = {}
        for row in record.energy_ledger:
            link_j_by_satellite[row["satellite_id"]] = (
                link_j_by_satellite.get(row["satellite_id"], 0.0)
                + float(row.get("link_bus_j", 0.0))
            )
            link_bytes_by_satellite[row["satellite_id"]] = (
                link_bytes_by_satellite.get(row["satellite_id"], 0.0)
                + float(row.get("link_bytes_accounted", 0.0))
            )
            self.assertEqual(row["link_energy_j_per_byte"], 1.0)
            if float(row.get("link_bus_j", 0.0)) > 0:
                self.assertEqual(row["compute_tokens"], 0.0)
        self.assertAlmostEqual(link_j_by_satellite["sat-a"], 30.0)
        self.assertAlmostEqual(link_j_by_satellite["sat-c"], 20.0)
        self.assertAlmostEqual(link_j_by_satellite["sat-b"], 10.0)
        self.assertAlmostEqual(sum(link_j_by_satellite.values()), 60.0)
        self.assertEqual(link_bytes_by_satellite, link_j_by_satellite)
        self.assertEqual(record.compute_owner_satellite_ids, ("sat-b",))

    def test_candidate_hard_gate_includes_low_soc_relay(self):
        bundle = FakeTemporalTraceBundle(
            active_by_time={
                0.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-c", "isl:sat-c:sat-b"),
                20.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-c", "isl:sat-c:sat-b"),
                40.0: ("sgl:ingress:sat-a", "isl:sat-a:sat-c", "isl:sat-c:sat-b"),
            }
        )
        _set_satellite_energy(
            bundle,
            initial_soc={"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.21},
            per_satellite={
                "sat-a": {"battery_capacity_j": 100.0, "link_energy_j_per_byte": 1.0},
                "sat-b": {"battery_capacity_j": 100.0, "link_energy_j_per_byte": 1.0},
                "sat-c": {
                    "battery_capacity_j": 100.0,
                    "link_energy_j_per_byte": 1.0,
                    "soc_min": 0.2,
                },
            },
        )

        record = run_trace_backed_runtime(
            requests=(_request(0.0, input_bytes=10.0, output_bytes=10.0),),
            deployment_catalog=_deployment_catalog(member_satellites=("sat-b",)),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "energy_aware", "max_hop_count": 2}),
            trace_bundle=bundle,
        ).records[0]

        self.assertTrue(record.rejected)
        self.assertEqual(record.reject_reason, "no_feasible_scheduler_candidate")

    def test_colocated_ttft_breakdown_closes_at_first_token_return(self):
        record = run_trace_backed_runtime(
            requests=(_request(0.0, prompt_tokens=3, output_tokens=3),),
            deployment_catalog=_deployment_catalog(
                prefill_tokens_per_s=3.0,
                decode_tokens_per_s=2.0,
                member_satellites=("sat-b",),
            ),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=FakeTemporalTraceBundle(),
        ).records[0]

        self.assertEqual(record.response_streaming_model, "first_token_plus_bulk_remaining")
        self.assertGreater(record.first_token_decode_s or 0.0, 0.0)
        self.assertGreater(record.return_path_s or 0.0, 0.0)
        self.assertAlmostEqual(record.ttft_s or 0.0, sum(
            float(record.ttft_breakdown_s[key])
            for key in (
                "access_wait_s", "uplink_s", "prefill_queue_s", "prefill_compute_s",
                "kv_transfer_s", "decode_queue_s", "first_token_decode_s", "return_path_s",
            )
        ))
    def test_energy_aware_policy_uses_soc_risk_unlike_shortest_path(self):
        request = _request(0.0, prompt_tokens=9, output_tokens=9)
        catalog = _deployment_catalog(prefill_tokens_per_s=3.0, decode_tokens_per_s=3.0)
        bundle = FakeTemporalTraceBundle(
            dual_access=True,
            soc_min_by_sat={"sat-a": 0.95, "sat-b": 0.2, "sat-c": 0.2},
        )

        shortest = run_trace_backed_runtime(
            requests=(request,),
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {"policy": "shortest_path", "max_hop_count": 2, "hard_gate_mode": "audit_only"}
            ),
            trace_bundle=bundle,
        )
        energy_aware = run_trace_backed_runtime(
            requests=(request,),
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "energy_aware", "max_hop_count": 2}),
            trace_bundle=FakeTemporalTraceBundle(
                dual_access=True,
                soc_min_by_sat={"sat-a": 0.95, "sat-b": 0.2, "sat-c": 0.2},
            ),
        )

        self.assertEqual(shortest.records[0].assigned_satellite_id, "sat-a")
        self.assertEqual(energy_aware.records[0].assigned_satellite_id, "sat-b")
        self.assertEqual(shortest.records[0].provenance["selection_weights"]["energy"], 0.0)
        self.assertGreater(energy_aware.records[0].provenance["selection_weights"]["energy"], 0.0)
        self.assertGreater(shortest.records[0].provenance["predicted_energy_risk"], 0.0)
        self.assertAlmostEqual(energy_aware.records[0].provenance["predicted_energy_risk"], 0.0)
        self.assertTrue(energy_aware.records[0].provenance["scheduler_decision"]["evaluated"])
        self.assertEqual(
            energy_aware.records[0].provenance["scheduler_decision"]["score_terms"]["score_model"],
            "request_level_slo_aware_surrogate_not_full_shadow_kv_marginal_rule",
        )

    def test_energy_prediction_advances_soc_to_candidate_execution_time(self):
        bundle = FakeTemporalTraceBundle()
        bundle.satellite_initial_soc = {"sat-a": 0.2, "sat-b": 0.2, "sat-c": 0.2}
        bundle.energy_windows = tuple(
            {
                **window,
                "battery_capacity_j": 100.0,
                "charge_rate_w": 1.0,
                "idle_power_w": 0.0,
                "compute_energy_j_per_token": 0.0,
                "link_energy_j_per_byte": 0.0,
            }
            for window in bundle.energy_windows
        )

        result = run_trace_backed_runtime(
            requests=(
                _request(
                    50.0,
                    prompt_tokens=1,
                    output_tokens=1,
                    input_bytes=0.0,
                    output_bytes=0.0,
                ),
            ),
            deployment_catalog=_deployment_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "nearest_visible"}),
            trace_bundle=bundle,
        )

        record = result.records[0]
        self.assertEqual(record.assigned_satellite_id, "sat-a")
        self.assertGreater(record.provenance["predicted_soc_after"], 0.69)
        self.assertLess(record.provenance["predicted_soc_after"], 0.71)
        self.assertAlmostEqual(record.provenance["predicted_soc_after"], record.soc_before)
        self.assertAlmostEqual(record.soc_before, record.soc_after)

    def test_energy_aware_uses_discharged_soc_at_candidate_execution_time(self):
        request = _request(
            50.0,
            prompt_tokens=1,
            output_tokens=1,
            input_bytes=0.0,
            output_bytes=0.0,
        )
        catalog = _deployment_catalog()

        shortest_bundle = FakeTemporalTraceBundle(dual_access=True)
        _set_satellite_energy(
            shortest_bundle,
            initial_soc={"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.8},
            per_satellite={
                "sat-a": {"battery_capacity_j": 100.0, "idle_power_w": 1.0, "soc_min": 0.5},
                "sat-b": {"battery_capacity_j": 100.0, "idle_power_w": 0.0, "soc_min": 0.5},
                "sat-c": {"battery_capacity_j": 100.0, "idle_power_w": 0.0, "soc_min": 0.5},
            },
        )
        energy_bundle = FakeTemporalTraceBundle(dual_access=True)
        _set_satellite_energy(
            energy_bundle,
            initial_soc={"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.8},
            per_satellite={
                "sat-a": {"battery_capacity_j": 100.0, "idle_power_w": 1.0, "soc_min": 0.5},
                "sat-b": {"battery_capacity_j": 100.0, "idle_power_w": 0.0, "soc_min": 0.5},
                "sat-c": {"battery_capacity_j": 100.0, "idle_power_w": 0.0, "soc_min": 0.5},
            },
        )

        shortest = run_trace_backed_runtime(
            requests=(request,),
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {"policy": "shortest_path", "max_hop_count": 2, "hard_gate_mode": "audit_only"}
            ),
            trace_bundle=shortest_bundle,
        )
        energy_aware = run_trace_backed_runtime(
            requests=(request,),
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "energy_aware", "max_hop_count": 2}),
            trace_bundle=energy_bundle,
        )

        self.assertEqual(shortest.records[0].assigned_satellite_id, "sat-a")
        self.assertEqual(energy_aware.records[0].assigned_satellite_id, "sat-b")
        self.assertLess(shortest.records[0].provenance["predicted_soc_after"], 0.31)
        self.assertGreater(shortest.records[0].provenance["predicted_energy_risk"], 0.19)
        self.assertAlmostEqual(energy_aware.records[0].provenance["predicted_energy_risk"], 0.0)

    def test_shadow_dvfs_is_explicit_and_trades_latency_for_soc(self):
        request = _request(
            0.0,
            prompt_tokens=10,
            output_tokens=10,
            input_bytes=0.0,
            output_bytes=0.0,
        )
        catalog = _deployment_catalog(prefill_tokens_per_s=10.0, decode_tokens_per_s=10.0)

        no_dvfs_bundle = FakeTemporalTraceBundle()
        _set_satellite_energy(
            no_dvfs_bundle,
            initial_soc={"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.8},
            per_satellite={
                "sat-a": {
                    "sunlit": False,
                    "dvfs_multiplier": 2.0,
                    "battery_capacity_j": 100.0,
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                    "soc_min": 0.2,
                },
            },
        )
        dvfs_bundle = FakeTemporalTraceBundle()
        _set_satellite_energy(
            dvfs_bundle,
            initial_soc={"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.8},
            per_satellite={
                "sat-a": {
                    "sunlit": False,
                    "dvfs_multiplier": 2.0,
                    "battery_capacity_j": 100.0,
                    "compute_energy_j_per_token": 1.0,
                    "link_energy_j_per_byte": 0.0,
                    "soc_min": 0.2,
                },
            },
        )

        no_dvfs = run_trace_backed_runtime(
            requests=(request,),
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "orbitserve",
                    "max_hop_count": 0,
                    "dvfs_modes_enabled": [],
                    "enabled_actions": ["local_compute", "reject_drop"],
                }
            ),
            trace_bundle=no_dvfs_bundle,
        )
        dvfs = run_trace_backed_runtime(
            requests=(request,),
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "orbitserve",
                    "max_hop_count": 0,
                    "hard_gate_overrides": {"shadow_dvfs_compute_energy_scale": 0.5},
                }
            ),
            trace_bundle=dvfs_bundle,
        )

        no_dvfs_record = no_dvfs.records[0]
        dvfs_record = dvfs.records[0]
        self.assertEqual(no_dvfs_record.action_type, "shadow_local_compute")
        self.assertEqual(dvfs_record.action_type, "shadow_dvfs")
        self.assertFalse(dvfs_record.compute_sunlit)
        self.assertEqual(dvfs_record.shadow_strategy, "shadow_dvfs")
        self.assertAlmostEqual(dvfs_record.dvfs_multiplier, 2.0)
        self.assertGreater(dvfs_record.e2e_s, no_dvfs_record.e2e_s)
        self.assertGreater(dvfs_record.soc_after, no_dvfs_record.soc_after)
        compute_row = next(
            row
            for row in dvfs_record.energy_ledger
            if float(row.get("compute_bus_j", 0.0)) > 0
        )
        self.assertEqual(compute_row["base_compute_energy_j_per_token"], 1.0)
        self.assertEqual(compute_row["runtime_compute_energy_scale"], 0.5)
        self.assertEqual(compute_row["compute_energy_j_per_token"], 0.5)

    def test_shadow_offload_is_explicit_when_shadow_access_satellite_is_energy_risky(self):
        request = _request(
            0.0,
            prompt_tokens=10,
            output_tokens=10,
            input_bytes=0.0,
            output_bytes=0.0,
        )
        bundle = FakeTemporalTraceBundle()
        _set_satellite_energy(
            bundle,
            initial_soc={"sat-a": 0.8, "sat-b": 0.8, "sat-c": 0.8},
            per_satellite={
                "sat-a": {
                    "sunlit": False,
                    "dvfs_multiplier": 2.0,
                    "battery_capacity_j": 100.0,
                    "compute_energy_j_per_token": 4.0,
                    "link_energy_j_per_byte": 0.0,
                    "soc_min": 0.78,
                },
                "sat-b": {
                    "sunlit": True,
                    "battery_capacity_j": 100.0,
                    "compute_energy_j_per_token": 0.0,
                    "link_energy_j_per_byte": 0.0,
                    "soc_min": 0.2,
                },
            },
        )

        result = run_trace_backed_runtime(
            requests=(request,),
            deployment_catalog=_deployment_catalog(prefill_tokens_per_s=10.0, decode_tokens_per_s=10.0),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "energy_aware",
                    "max_hop_count": 2,
                    "hard_gate_mode": "audit_only",
                    "weights": {"shadow_offload_credit": 1.0},
                }
            ),
            trace_bundle=bundle,
        )
        record = result.records[0]
        self.assertEqual(record.assigned_satellite_id, "sat-b")
        self.assertEqual(record.action_type, "shadow_offload")
        self.assertFalse(record.access_sunlit)
        self.assertTrue(record.compute_sunlit)
        self.assertEqual(record.shadow_strategy, "shadow_offload")
        metrics = aggregate_runtime_records(
            result.records,
            time_horizon_s=100.0,
            trace_model=result.trace_model,
            scope=result.scope,
        ).runtime_metrics
        self.assertEqual(metrics["shadow_offload_request_count"], 1)
        self.assertEqual(metrics["action_type_counts"]["shadow_offload"], 1)
        self.assertEqual(metrics["shadow_affected_request_count"], 1)

    def test_energy_prediction_accounts_for_link_queue_wait_before_prefill(self):
        bundle = FakeTemporalTraceBundle()
        _set_satellite_energy(
            bundle,
            initial_soc={"sat-a": 0.2, "sat-b": 0.2, "sat-c": 0.2},
            per_satellite={
                "sat-a": {
                    "battery_capacity_j": 100.0,
                    "charge_rate_w": 1.0,
                    "idle_power_w": 0.0,
                    "compute_energy_j_per_token": 0.0,
                    "link_energy_j_per_byte": 0.0,
                },
                "sat-b": {"battery_capacity_j": 100.0, "charge_rate_w": 0.0, "idle_power_w": 0.0},
                "sat-c": {"battery_capacity_j": 100.0, "charge_rate_w": 0.0, "idle_power_w": 0.0},
            },
        )

        result = run_trace_backed_runtime(
            requests=(
                _request(0.0, request_id="req-0", prompt_tokens=50, output_tokens=1, input_bytes=0.0, output_bytes=0.0),
                _request(0.0, request_id="req-1", prompt_tokens=1, output_tokens=1, input_bytes=0.0, output_bytes=0.0),
            ),
            deployment_catalog=_deployment_catalog(prefill_tokens_per_s=3.0, decode_tokens_per_s=300.0),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "local_only"}),
            trace_bundle=bundle,
        )

        first, second = result.records
        self.assertEqual(first.assigned_satellite_id, "sat-a")
        self.assertEqual(second.assigned_satellite_id, "sat-a")
        self.assertGreater(second.link_queue_wait_s, 49.0)
        self.assertGreater(second.provenance["predicted_soc_after"], 0.69)
        self.assertLess(second.provenance["predicted_soc_after"], 0.71)
        self.assertAlmostEqual(second.provenance["predicted_soc_after"], second.soc_before)

    def test_candidate_soc_preview_does_not_mutate_unselected_satellite_state(self):
        bundle = FakeTemporalTraceBundle(dual_access=True)
        _set_satellite_energy(
            bundle,
            initial_soc={"sat-a": 0.8, "sat-b": 0.2, "sat-c": 0.8},
            per_satellite={
                "sat-a": {"battery_capacity_j": 100.0, "charge_rate_w": 0.0, "idle_power_w": 0.0},
                "sat-b": {"battery_capacity_j": 100.0, "charge_rate_w": 1.0, "idle_power_w": 0.0},
                "sat-c": {"battery_capacity_j": 100.0, "charge_rate_w": 0.0, "idle_power_w": 0.0},
            },
        )

        result = run_trace_backed_runtime(
            requests=(
                _request(
                    50.0,
                    prompt_tokens=1,
                    output_tokens=1,
                    input_bytes=0.0,
                    output_bytes=0.0,
                ),
            ),
            deployment_catalog=_deployment_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "shortest_path", "max_hop_count": 2}),
            trace_bundle=bundle,
        )

        self.assertEqual(result.records[0].assigned_satellite_id, "sat-a")
        self.assertAlmostEqual(result.state_summary["satellites"]["sat-b"]["soc"], 0.2)

    def test_energy_window_gap_at_candidate_execution_time_fails_closed(self):
        bundle = FakeTemporalTraceBundle()
        bundle.energy_windows = tuple(
            {
                **window,
                "end_s": 10.0,
            }
            for window in bundle.energy_windows
        )

        result = run_trace_backed_runtime(
            requests=(_request(50.0),),
            deployment_catalog=_deployment_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "nearest_visible",
                    "fallback_enabled": True,
                    "hard_gate_overrides": {"ground_fallback_latency_s": 1.0},
                }
            ),
            trace_bundle=bundle,
        )

        record = result.records[0]
        self.assertTrue(record.rejected)
        self.assertFalse(record.fallback_used)
        self.assertEqual(record.reject_reason, "no_energy_window")
        self.assertIn("no_energy_window", record.provenance["warnings"])

    def test_stale_access_sample_rejects_with_warning(self):
        bundle = FakeTemporalTraceBundle(stale_access=True)
        result = run_trace_backed_runtime(
            requests=(_request(0.0),),
            deployment_catalog=_deployment_catalog(),
            runtime_strategy=RuntimeStrategySpec.from_dict({"policy": "nearest_visible"}),
            trace_bundle=bundle,
        )

        record = result.records[0]
        self.assertTrue(record.rejected)
        self.assertEqual(record.reject_reason, "stale_trace_sample:access")
        self.assertIn("stale_trace_sample:access", record.provenance["warnings"])
        self.assertIn("stale_trace_sample:access", result.decision_traces[0]["warnings"])

    def test_outbound_route_loss_after_compute_is_partial_failure(self):
        bundle = FakeTemporalTraceBundle(
            access_end_s=5.0,
            active_by_time={
                0.0: ("sgl:ingress:sat-a",),
                20.0: (),
                40.0: (),
            },
        )
        result = run_trace_backed_runtime(
            requests=(_request(0.0),),
            deployment_catalog=_deployment_catalog(prefill_tokens_per_s=0.3, decode_tokens_per_s=0.3),
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {
                    "policy": "nearest_visible",
                    "fallback_enabled": True,
                    "hard_gate_overrides": {"ground_fallback_latency_s": 1.0},
                }
            ),
            trace_bundle=bundle,
        )

        record = result.records[0]
        self.assertTrue(record.accepted)
        self.assertFalse(record.completed)
        self.assertFalse(record.served)
        self.assertFalse(record.rejected)
        self.assertTrue(record.timeout)
        self.assertFalse(record.fallback_used)
        self.assertEqual(record.timeout_reason, "route_became_unavailable_after_compute")
        self.assertEqual(record.bottleneck_reason, "route_became_unavailable_after_compute")
        self.assertEqual(record.action_type, "local_compute")
        self.assertGreater(len(record.selected_path), 0)
        self.assertTrue(record.provenance["scheduler_decision"]["evaluated"])
        self.assertIn("score_terms", record.provenance)
        self.assertTrue(result.decision_traces[0]["scheduler_decision"]["evaluated"])


def _path_to(rows, satellite_id):
    for _access, compute, path in rows:
        if compute == satellite_id:
            return path
    raise AssertionError(f"missing path to {satellite_id}")


def _set_satellite_energy(bundle, *, initial_soc, per_satellite):
    bundle.satellite_initial_soc = dict(initial_soc)
    bundle.energy_windows = tuple(
        {
            **window,
            **dict(per_satellite.get(str(window["satellite_id"]), {})),
        }
        for window in bundle.energy_windows
    )


if __name__ == "__main__":
    unittest.main()
