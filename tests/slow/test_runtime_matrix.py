import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from orbitserve.dse.search import DSESpec, DSERuntimeMatrixSpec, run_dse
from orbitserve.dse.runtime_matrix import DSEDesignContext, evaluate_runtime_matrix
from orbitserve.experiments import load_experiment_spec
from orbitserve.runtime.runtime_optimization import score_runtime_evaluator_result


ROOT = Path(__file__).resolve().parents[2]


def _base_spec(runtime_matrix=None, *, plane_count=(2,)):
    base = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_single_design.json")
    return replace(
        base,
        workload=replace(
            base.workload,
            bundle_dir="tests/fixtures/workloads/smoke_flat_single_turn_direct",
        ),
        simulation=replace(base.simulation, trace_model="ideal_full_visibility"),
        scheduler=replace(base.scheduler, policy="local_only"),
        dse=DSESpec(
            enabled=True,
            sweep={"plane_count": plane_count},
            runtime_matrix=runtime_matrix or DSERuntimeMatrixSpec(),
        ),
    )


def _matrix_context(design_id, *, served, e2e, annualized_cost):
    return DSEDesignContext(
        design_id=design_id,
        design_row={
            "design_id": design_id,
            "cost": {
                "total_cost_usd": annualized_cost * 10.0,
                "annualized_cost_usd_per_year": annualized_cost,
            },
            "performance": {
                "ttft_p95_s": e2e / 4.0,
                "e2e_p95_s": e2e,
                "details": {
                    "static_slo_proxy": {
                        "provenance": {
                            "analysis_scope": "design_only_surrogate",
                            "valid_for_paper_main_results": False,
                        }
                    }
                },
            },
            "slo": {
                "evaluated": True,
                "served_ratio": served,
                "timeout_ratio": 1.0 - served,
            },
        },
        runtime_scenario=SimpleNamespace(
            scenario_id=design_id,
            serving_deployments=(),
            provenance={},
        ),
        current_scheduler={"policy": "local_only"},
        cost_result={
            "total_cost_usd": annualized_cost * 10.0,
            "annualized_cost_usd_per_year": annualized_cost,
        },
    )


def _runtime_result(*, served, e2e):
    return SimpleNamespace(
        to_dict=lambda: {
            "evaluated": True,
            "runtime_metrics": {
                "served_ratio": served,
                "leo_served_ratio": max(0.0, served - 0.1),
                "useful_request_throughput_per_s": served / 10.0,
                "useful_token_throughput_per_s": served * 1000.0,
                "ttft_p95_s": e2e / 4.0,
                "ttft_p99_s": e2e / 2.0,
                "e2e_p95_s": e2e,
                "e2e_p99_s": e2e * 1.5,
                "scope": "trace_backed_runtime_surrogate",
                "trace_model": "surrogate_orbit_trace",
                "valid_for_paper_main_results": False,
            },
            "slo": {"served_ratio": served, "timeout_ratio": 1.0 - served},
            "performance": {
                "ttft_p95_s": e2e / 4.0,
                "ttft_p99_s": e2e / 2.0,
                "e2e_p95_s": e2e,
                "e2e_p99_s": e2e * 1.5,
            },
            "warnings": (),
        }
    )


class DSERuntimeMatrixTests(unittest.TestCase):
    def test_nested_runtime_matrix_spec_round_trips(self):
        spec = DSESpec.from_dict(
            {
                "enabled": True,
                "sweep": {"plane_count": [2]},
                "runtime_matrix": {
                    "enabled": True,
                    "include_current_policy": False,
                    "policies": ["local_only", "shortest_path"],
                    "ground_config": {
                        "probe_interval_s": 0.1,
                        "remote_queue_buffer_requests": 1,
                    },
                    "kv_policy_candidates": [
                        {
                            "kv_policy_id": "recompute_pages_16",
                            "kv_policy": {
                                "kv_serve_actions_enabled": ["recompute_context"],
                                "kv_page_size_tokens": 16,
                            },
                        }
                    ],
                    "allow_idealized_runtime_metrics": True,
                    "ignore_optional_cost": True,
                    "objective": {
                        "name": "preliminary_cost_slo_weighted",
                        "weights": {"served": 1.0, "tail": 1.0, "cost": 0.0},
                    },
                },
            }
        )

        payload = spec.to_dict()
        matrix = payload["runtime_matrix"]
        self.assertTrue(matrix["enabled"])
        self.assertEqual(matrix["policies"], ["local_only", "shortest_path"])
        self.assertEqual(matrix["ground_config"]["probe_interval_s"], 0.1)
        self.assertEqual(matrix["kv_policy_candidates"][0]["kv_policy_id"], "recompute_pages_16")
        self.assertEqual(matrix["kv_policy_candidates"][0]["kv_policy"]["kv_page_size_tokens"], 16)
        self.assertTrue(matrix["allow_preliminary_runtime_metrics"])
        self.assertTrue(matrix["allow_idealized_runtime_metrics"])
        self.assertTrue(matrix["ignore_optional_cost"])
        self.assertEqual(matrix["objective"]["name"], "preliminary_cost_slo_weighted")

        spec = DSESpec.from_dict(
            {
                "enabled": True,
                "runtime_matrix": {
                    "enabled": True,
                    "static_screening": {
                        "enabled": True,
                        "top_k_designs": 2,
                        "include_static_frontier": True,
                    },
                    "closed_loop": {
                        "prefer_static_candidates": True,
                    },
                },
            }
        )
        self.assertTrue(spec.runtime_matrix.static_screening["enabled"])
        self.assertEqual(spec.runtime_matrix.static_screening["top_k_designs"], 2)
        self.assertTrue(spec.runtime_matrix.static_screening["include_static_frontier"])
        self.assertTrue(spec.runtime_matrix.closed_loop["prefer_static_candidates"])

    def test_runtime_matrix_scope_candidate_schema_accepts_documented_names(self):
        spec = DSESpec.from_dict(
            {
                "enabled": True,
                "runtime_matrix": {
                    "enabled": True,
                    "placement_scope_search_space": {
                        "scope_candidates": [
                            {
                                "placement_scope_strategy_id": "same_plane_overflow.v1",
                                "placement_scope": "hybrid_template_mix",
                                "hybrid_template": {
                                    "template_id": "same_plane_plus_cross_plane_overflow",
                                    "max_pairs_per_prefill_group": 2,
                                },
                            }
                        ]
                    },
                },
            }
        )

        candidate = spec.runtime_matrix.placement_scope_candidates[0]
        self.assertEqual(candidate["strategy_id"], "same_plane_overflow.v1")
        self.assertEqual(candidate["placement_scope_strategy_id"], "same_plane_overflow.v1")
        self.assertEqual(candidate["pd_placement"]["placement_scope"], "hybrid_template_mix")
        self.assertEqual(candidate["pd_placement"]["hybrid_template"]["template_id"], "same_plane_plus_cross_plane_overflow")

    def test_runtime_matrix_disabled_preserves_unevaluated_behavior(self):
        result = run_dse(_base_spec())
        payload = result.to_dict()

        self.assertTrue(payload["evaluated"])
        self.assertFalse(payload["runtime_matrix"]["evaluated"])
        self.assertEqual(payload["runtime_matrix"]["matrix_rows"], [])
        self.assertEqual(payload["runtime_matrix"]["selected_rows"], [])
        self.assertTrue(payload["frontier"]["evaluated"])
        self.assertEqual(payload["frontier"]["analysis_scope"], "design_only_surrogate")
        self.assertEqual(payload["frontier"]["provenance"]["source"], "static_slo_proxy")
        self.assertFalse(payload["frontier"]["valid_for_paper_main_results"])
        self.assertTrue(payload["metrics"]["pareto_frontier_evaluated"])
        self.assertIn("runtime_current_strategy", payload["rows"][0]["performance"]["details"])
        self.assertTrue(payload["rows"][0]["slo"]["evaluated"])
        self.assertIsNotNone(payload["rows"][0]["slo"]["served_ratio"])
        self.assertIsNotNone(payload["rows"][0]["performance"]["ttft_p95_s"])
        self.assertIsNotNone(payload["rows"][0]["performance"]["ttft_p99_s"])
        self.assertIsNotNone(payload["rows"][0]["performance"]["e2e_p95_s"])
        self.assertIsNotNone(payload["rows"][0]["performance"]["e2e_p99_s"])

    def test_static_slo_proxy_consumes_trace_access_and_energy_summary(self):
        base = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_trace_backed_runtime.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = replace(
                base,
                simulation=replace(base.simulation, trace_output_dir=tmpdir),
                dse=DSESpec(enabled=True, sweep={"plane_count": (2,)}),
            )
            payload = run_dse(spec).to_dict()

        row = payload["rows"][0]
        proxy = row["performance"]["details"]["static_slo_proxy"]
        self.assertTrue(row["slo"]["evaluated"])
        self.assertIsNotNone(row["slo"]["served_ratio"])
        self.assertTrue(proxy["timebin_summary"]["evaluated"])
        self.assertTrue(proxy["access_summary"]["evaluated"])
        self.assertTrue(proxy["energy_summary"]["evaluated"])
        self.assertEqual(proxy["access_summary"]["trace_model"], "surrogate_orbit_trace")
        self.assertIn("capacity_factor", proxy["energy_summary"])
        self.assertTrue(payload["frontier"]["evaluated"])
        self.assertEqual(payload["frontier"]["analysis_scope"], "design_only_surrogate")

    def test_trace_builder_exception_excludes_design_from_static_frontier(self):
        base = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_trace_backed_runtime.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = replace(
                base,
                simulation=replace(base.simulation, trace_output_dir=tmpdir),
                dse=DSESpec(enabled=True, sweep={"plane_count": (2,)}),
            )
            with patch(
                "orbitserve.orbit.simulation_trace_builder.build_surrogate_orbit_trace",
                side_effect=RuntimeError("forced trace builder failure"),
            ):
                payload = run_dse(spec).to_dict()

        row = payload["rows"][0]
        proxy = row["performance"]["details"]["static_slo_proxy"]
        self.assertFalse(proxy["evaluated"])
        self.assertFalse(proxy["slo"]["evaluated"])
        self.assertIsNone(proxy["slo"]["served_ratio"])
        self.assertIsNone(proxy["performance"]["e2e_p95_s"])
        self.assertFalse(row["slo"]["evaluated"])
        self.assertIsNone(row["slo"]["served_ratio"])
        self.assertFalse(payload["frontier"]["evaluated"])
        self.assertEqual(payload["frontier"]["frontier_rows"], [])
        self.assertFalse(payload["static_frontier"]["evaluated"])
        self.assertEqual(payload["static_frontier"]["candidate_count"], 1)
        self.assertEqual(payload["static_frontier"]["excluded_count"], 1)
        self.assertEqual(payload["static_frontier"]["frontier_rows"], [])
        self.assertEqual(
            payload["static_frontier"]["excluded_candidates"][0]["excluded_reason"],
            "static_slo_evaluated is false",
        )
        self.assertTrue(any("forced trace builder failure" in warning for warning in proxy["warnings"]))
        self.assertFalse(any("assuming full visibility" in warning for warning in proxy["warnings"]))

    def test_strict_runtime_matrix_has_rows_but_no_scores_or_selection(self):
        matrix_spec = DSERuntimeMatrixSpec(
            enabled=True,
            include_current_policy=False,
            policies=("local_only", "shortest_path"),
        )
        result = run_dse(_base_spec(matrix_spec, plane_count=(2, 3)))
        payload = result.to_dict()
        matrix = payload["runtime_matrix"]

        self.assertTrue(matrix["evaluated"])
        self.assertEqual(matrix["design_count"], 2)
        self.assertEqual(matrix["policy_count"], 2)
        self.assertEqual(matrix["row_count"], matrix["design_count"] * matrix["policy_count"])
        self.assertEqual(len(matrix["matrix_rows"]), matrix["row_count"])
        self.assertEqual(matrix["selected_rows"], [])
        for row in matrix["matrix_rows"]:
            self.assertIsNone(row["objective_score"])
            self.assertIsNone(row["objective_score_scope"])
        self.assertFalse(payload["frontier"]["evaluated"])
        self.assertEqual(payload["frontier"]["candidate_count"], matrix["row_count"])
        self.assertEqual(payload["frontier"]["excluded_count"], matrix["row_count"])
        self.assertFalse(payload["metrics"]["pareto_frontier_evaluated"])

    def test_allow_preliminary_runtime_matrix_selects_preliminary_rows(self):
        matrix_spec = DSERuntimeMatrixSpec(
            enabled=True,
            include_current_policy=False,
            policies=("local_only", "shortest_path"),
            allow_preliminary_runtime_metrics=True,
            ignore_optional_cost=True,
        )
        result = run_dse(_base_spec(matrix_spec, plane_count=(2, 3)))
        payload = result.to_dict()
        matrix = payload["runtime_matrix"]

        self.assertTrue(matrix["evaluated"])
        self.assertEqual(matrix["row_count"], matrix["design_count"] * matrix["policy_count"])
        self.assertEqual(len(matrix["selected_rows"]), matrix["design_count"])
        self.assertTrue(payload["frontier"]["evaluated"])
        self.assertEqual(payload["frontier"]["analysis_scope"], "preliminary_surrogate")
        self.assertEqual(payload["frontier"]["calibration"], "not_stk_calibrated")
        self.assertFalse(payload["frontier"]["valid_for_paper_main_results"])
        self.assertTrue(payload["metrics"]["pareto_frontier_evaluated"])
        self.assertIsNotNone(payload["metrics"]["dominated_design_count"])
        for row in matrix["selected_rows"]:
            self.assertEqual(row["objective_score_scope"], "preliminary_idealized")
            self.assertFalse(row["valid_for_paper_main_results"])
            self.assertIsNotNone(row["objective_score"])
            self.assertIsNotNone(row["static_rank"])
            self.assertTrue(row["static_candidate_selected"])
            self.assertIn("static_frontier_member", row)
        for row in payload["frontier"]["frontier_rows"]:
            self.assertFalse(row["valid_for_paper_main_results"])
            self.assertIsNotNone(row["served_ratio"])
            self.assertIsNotNone(row["e2e_p95_s"])
        selected_ids = {(row["design_id"], row["strategy_id"]) for row in matrix["selected_rows"]}
        selected_row_ids = {
            (row["design_id"], row["strategy_id"])
            for row in matrix["matrix_rows"]
            if row["selected"]
        }
        self.assertEqual(selected_ids, selected_row_ids)

    def test_runtime_matrix_searches_pd_scope_candidates_per_policy(self):
        matrix_spec = DSERuntimeMatrixSpec(
            enabled=True,
            include_current_policy=False,
            policies=("local_only", "shortest_path"),
            placement_scope_candidates=(
                {"strategy_id": "scope_intra", "pd_placement": {"placement_scope": "intra_cluster"}},
                {"strategy_id": "scope_cross", "pd_placement": {"placement_scope": "cross_plane"}},
            ),
            allow_preliminary_runtime_metrics=True,
            ignore_optional_cost=True,
        )
        deployment = SimpleNamespace(
            serving_spec={"pd_mode": "disaggregated"},
            throughput_estimate={
                "pd_mode": "disaggregated",
                "evaluated": True,
                "feasible": True,
                "prefill_tokens_per_s": 100.0,
                "decode_tokens_per_s": 50.0,
                "prefill_deployment_group_count": 1,
                "decode_deployment_group_count": 2,
                "satellites_per_deployment": 1,
                "stage_satellite_count": 1,
                "serving_satellite_budget_count": 3,
                "satellite_budget_mode": "fixed_total",
                "comparison_group_id": "pd-scope-budget",
                "prefill_pool_satellites": 1,
                "decode_pool_satellites": 2,
                "kv_cache_bytes_per_token": 16.0,
                "kv_cache_capacity_tokens": 1000,
            },
        )
        scenario = SimpleNamespace(
            scenario_id="runtime_scenario",
            constellation={
                "satellite_count": 3,
                "satellite_catalog": (
                    {"satellite_id": "sat-p0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 0, "cluster_id": "c0"},
                    {"satellite_id": "sat-d0", "orbital_set_name": "walker", "plane_index": 0, "cell_index": 0, "cluster_id": "c0"},
                    {"satellite_id": "sat-d1", "orbital_set_name": "walker", "plane_index": 1, "cell_index": 1, "cluster_id": "c1"},
                ),
            },
            serving_deployments=(deployment,),
            provenance={},
        )
        context = DSEDesignContext(
            design_id="design_0",
            design_row={
                "performance": {
                    "details": {
                        "serving_satellite_budget_count": 3,
                        "satellite_budget_mode": "fixed_total",
                        "comparison_group_id": "pd-scope-budget",
                        "prefill_pool_satellites": 1,
                        "decode_pool_satellites": 2,
                    }
                }
            },
            runtime_scenario=scenario,
            current_scheduler={"policy": "local_only"},
            cost_result={"total_cost_usd": 100.0, "annualized_cost_usd_per_year": 10.0},
        )
        observed_scopes = []

        def fake_evaluate_runtime_performance(scoped_scenario, strategy):
            scoped_deployment = scoped_scenario.serving_deployments[0]
            scope = scoped_deployment.throughput_estimate["pd_placement"]["placement_scope"]
            observed_scopes.append((scope, strategy.policy))
            return SimpleNamespace(
                to_dict=lambda: {
                    "evaluated": True,
                    "runtime_metrics": {
                        "served_ratio": 1.0 if scope == "intra_cluster" else 0.9,
                        "ttft_p95_s": 1.0,
                        "e2e_p95_s": 2.0 if scope == "intra_cluster" else 3.0,
                        "scope": "trace_backed_runtime_surrogate",
                        "trace_model": "surrogate_orbit_trace",
                        "valid_for_paper_main_results": False,
                    },
                    "slo": {"served_ratio": 1.0 if scope == "intra_cluster" else 0.9},
                    "performance": {"ttft_p95_s": 1.0, "e2e_p95_s": 2.0 if scope == "intra_cluster" else 3.0},
                    "serving": {"provenance": {"placement_scope": scope}},
                    "request_records": [{"pd_pairing_rule": scope, "pd_placement_scope": scope}],
                    "warnings": (),
                }
            )

        with patch("orbitserve.dse.runtime_matrix.evaluate_runtime_performance", fake_evaluate_runtime_performance):
            matrix = evaluate_runtime_matrix((context,), matrix_spec).to_dict()

        self.assertEqual(matrix["design_count"], 1)
        self.assertEqual(matrix["policy_count"], 2)
        self.assertEqual(matrix["placement_scope_candidate_count"], 2)
        self.assertEqual(matrix["row_count"], 4)
        self.assertEqual(
            {(scope, policy) for scope, policy in observed_scopes},
            {
                ("intra_cluster", "local_only"),
                ("intra_cluster", "shortest_path"),
                ("cross_plane", "local_only"),
                ("cross_plane", "shortest_path"),
            },
        )
        self.assertTrue(all(row["placement_scope_strategy_id"] in {"scope_intra", "scope_cross"} for row in matrix["matrix_rows"]))
        self.assertEqual({row["pd_placement_scope"] for row in matrix["matrix_rows"]}, {"intra_cluster", "cross_plane"})
        self.assertTrue(all(row["serving_satellite_budget_count"] == 3 for row in matrix["matrix_rows"]))
        self.assertTrue(all(row["satellite_budget_mode"] == "fixed_total" for row in matrix["matrix_rows"]))
        self.assertTrue(all(row["comparison_group_id"] == "pd-scope-budget" for row in matrix["matrix_rows"]))
        self.assertTrue(all(row["prefill_pool_satellites"] == 1 for row in matrix["matrix_rows"]))
        self.assertTrue(all(row["decode_pool_satellites"] == 2 for row in matrix["matrix_rows"]))
        self.assertTrue(all(row["catalog_rebuilt"] for row in matrix["matrix_rows"]))
        self.assertTrue(all(row["catalog_builder"] == "orbitserve.runtime_deployments.build_runtime_deployment_catalog" for row in matrix["matrix_rows"]))
        self.assertEqual(len(matrix["selected_rows"]), 1)
        self.assertEqual(matrix["selected_rows"][0]["pd_placement_scope"], "intra_cluster")
        self.assertEqual(
            matrix["selected_scope_by_design"],
            {
                "design_0": {
                    "placement_scope_strategy_id": "scope_intra",
                    "pd_placement_scope": "intra_cluster",
                    "pd_pairing_rule": "intra_cluster",
                }
            },
        )

    def test_runtime_matrix_expands_kv_policy_candidates(self):
        deployment = SimpleNamespace(
            serving_spec={"pd_mode": "disaggregated"},
            throughput_estimate={
                "pd_mode": "disaggregated",
                "evaluated": True,
                "feasible": True,
                "prefill_tokens_per_s": 100.0,
                "decode_tokens_per_s": 50.0,
                "prefill_deployment_group_count": 1,
                "decode_deployment_group_count": 1,
                "satellites_per_deployment": 1,
                "stage_satellite_count": 1,
                "kv_cache_bytes_per_token": 16.0,
                "kv_cache_capacity_tokens": 1000,
            },
        )
        scenario = SimpleNamespace(
            scenario_id="runtime_scenario",
            constellation={"satellite_count": 1, "satellite_catalog": ({"satellite_id": "sat-0"},)},
            serving_deployments=(deployment,),
            provenance={},
        )
        matrix_spec = SimpleNamespace(
            include_current_policy=False,
            policies=("orbitserve",),
            allow_preliminary_runtime_metrics=True,
            ignore_optional_cost=True,
            kv_policy_candidates=(
                {
                    "kv_policy_id": "recompute_pages_16",
                    "kv_policy": {
                        "kv_serve_actions_enabled": ("recompute_context",),
                        "kv_post_turn_actions_enabled": ("none",),
                        "kv_page_size_tokens": 16,
                        "kv_transfer_chunk_bytes": 1024,
                    },
                },
                {
                    "kv_policy_id": "replicate_pages_32",
                    "kv_policy": {
                        "kv_serve_actions_enabled": ("sticky_kv_owner", "migrate_kv", "recompute_context"),
                        "kv_post_turn_actions_enabled": ("replicate_kv",),
                        "replica_count_limit": 1,
                        "replica_target_deployment_id": "deployment_0:decode:far",
                        "kv_page_size_tokens": 32,
                        "eviction_policy": "replica_first",
                    },
                },
            ),
        )
        context = DSEDesignContext(
            design_id="design_0",
            design_row={},
            runtime_scenario=scenario,
            current_scheduler={"policy": "local_only"},
            cost_result={"total_cost_usd": 100.0},
        )
        observed = []

        def fake_evaluate_runtime_performance(scoped_scenario, strategy):
            del scoped_scenario
            observed.append((strategy.policy, strategy.kv_policy))
            return SimpleNamespace(
                to_dict=lambda: {
                    "evaluated": True,
                    "runtime_metrics": {
                        "served_ratio": 1.0,
                        "ttft_p95_s": 1.0,
                        "e2e_p95_s": 2.0,
                        "scope": "trace_backed_runtime_surrogate",
                        "trace_model": "surrogate_orbit_trace",
                        "valid_for_paper_main_results": False,
                    },
                    "slo": {"served_ratio": 1.0},
                    "performance": {"ttft_p95_s": 1.0, "e2e_p95_s": 2.0},
                    "warnings": (),
                }
            )

        with patch("orbitserve.dse.runtime_matrix.evaluate_runtime_performance", fake_evaluate_runtime_performance):
            matrix = evaluate_runtime_matrix((context,), matrix_spec).to_dict()

        self.assertEqual(matrix["row_count"], 2)
        self.assertEqual({row["kv_policy_id"] for row in matrix["matrix_rows"]}, {"recompute_pages_16", "replicate_pages_32"})
        by_policy = {row["kv_policy_id"]: row for row in matrix["matrix_rows"]}
        self.assertEqual(by_policy["recompute_pages_16"]["kv_page_size_tokens"], 16)
        self.assertEqual(by_policy["recompute_pages_16"]["kv_transfer_chunk_bytes"], 1024.0)
        self.assertEqual(by_policy["replicate_pages_32"]["replica_count_limit"], 1)
        self.assertEqual(by_policy["replicate_pages_32"]["replica_target_deployment_id"], "deployment_0:decode:far")
        self.assertEqual(by_policy["replicate_pages_32"]["eviction_policy"], "replica_first")
        self.assertEqual(by_policy["replicate_pages_32"]["kv_post_turn_actions_enabled"], ["replicate_kv"])
        self.assertEqual(len(observed), 2)

    def test_static_screening_top_k_limits_runtime_evaluation_and_records_skips(self):
        matrix_spec = DSERuntimeMatrixSpec(
            enabled=True,
            include_current_policy=False,
            policies=("local_only",),
            allow_preliminary_runtime_metrics=True,
            ignore_optional_cost=True,
            static_screening={"enabled": True, "top_k_designs": 1, "include_static_best": True},
        )
        contexts = (
            _matrix_context("design_slow", served=0.8, e2e=8.0, annualized_cost=10.0),
            _matrix_context("design_best", served=0.95, e2e=3.0, annualized_cost=12.0),
            _matrix_context("design_mid", served=0.9, e2e=5.0, annualized_cost=11.0),
        )
        evaluated_designs = []

        def fake_evaluate_runtime_performance(scenario, strategy):
            del strategy
            evaluated_designs.append(scenario.scenario_id)
            return _runtime_result(served=0.9, e2e=4.0)

        with patch("orbitserve.dse.runtime_matrix.evaluate_runtime_performance", fake_evaluate_runtime_performance):
            matrix = evaluate_runtime_matrix(contexts, matrix_spec).to_dict()

        self.assertEqual(evaluated_designs, ["design_best"])
        self.assertEqual(matrix["design_count"], 3)
        self.assertEqual(matrix["row_count"], 3)
        self.assertEqual(matrix["static_screening"]["selected_design_ids"], ["design_best"])
        self.assertEqual(matrix["static_screening"]["skipped_count"], 2)
        evaluated_rows = [row for row in matrix["matrix_rows"] if row["runtime_evaluated"]]
        skipped_rows = [row for row in matrix["matrix_rows"] if row["policy"] == "static_screen_skipped"]
        self.assertEqual(len(evaluated_rows), 1)
        self.assertEqual({row["design_id"] for row in skipped_rows}, {"design_slow", "design_mid"})
        self.assertEqual(evaluated_rows[0]["static_rank"], 1)
        self.assertTrue(evaluated_rows[0]["static_screen_selected"])
        self.assertTrue(evaluated_rows[0]["static_candidate_selected"])
        self.assertEqual(evaluated_rows[0]["static_screen_reason"], "static_top_1")
        self.assertTrue(all(not row["static_candidate_selected"] for row in skipped_rows))
        self.assertTrue(all(row["runtime_evaluated"] is False for row in skipped_rows))
        self.assertTrue(all(row["static_screen_reason"] == "not_selected_by_static_screening" for row in skipped_rows))

    def test_non_separability_metrics_compare_static_and_runtime_ranks(self):
        matrix_spec = DSERuntimeMatrixSpec(
            enabled=True,
            include_current_policy=False,
            policies=("local_only",),
            allow_preliminary_runtime_metrics=True,
            ignore_optional_cost=True,
            objective={
                "name": "rank_flip_probe",
                "weights": {"served": 100.0, "tail": 1.0, "cost": 0.0},
            },
        )
        contexts = (
            _matrix_context("static_best_runtime_worse", served=0.96, e2e=3.0, annualized_cost=10.0),
            _matrix_context("static_second_runtime_best", served=0.9, e2e=5.0, annualized_cost=10.0),
        )

        def fake_evaluate_runtime_performance(scenario, strategy):
            del strategy
            if scenario.scenario_id == "static_second_runtime_best":
                return _runtime_result(served=0.99, e2e=2.0)
            return _runtime_result(served=0.8, e2e=8.0)

        with patch("orbitserve.dse.runtime_matrix.evaluate_runtime_performance", fake_evaluate_runtime_performance):
            matrix = evaluate_runtime_matrix(contexts, matrix_spec).to_dict()

        self.assertTrue(matrix["non_separability"]["evaluated"])
        self.assertEqual(matrix["non_separability"]["static_best_design_id"], "static_best_runtime_worse")
        self.assertEqual(matrix["non_separability"]["runtime_best_design_id"], "static_second_runtime_best")
        self.assertEqual(matrix["non_separability"]["rank_disagreement_count"], 1)
        self.assertGreater(matrix["non_separability"]["regret_served_ratio"], 0.0)
        self.assertEqual(matrix["non_separability"]["dominated_design_count"], 1)
        self.assertFalse(matrix["non_separability"]["valid_for_paper_main_results"])
        self.assertIn("differ", matrix["non_separability"]["analysis"])
        self.assertEqual(matrix["metrics"]["rank_disagreement_count"], 1)
        self.assertGreater(matrix["metrics"]["non_separability_regret"], 0.0)
        self.assertEqual(matrix["metrics"]["dominated_design_count"], 1)

    def test_non_separability_marks_rank_comparison_incomplete_when_static_best_unscored(self):
        matrix_spec = DSERuntimeMatrixSpec(
            enabled=True,
            include_current_policy=False,
            policies=("local_only",),
            allow_preliminary_runtime_metrics=True,
            ignore_optional_cost=True,
            objective={
                "name": "static_best_unscored_probe",
                "weights": {"served": 100.0, "tail": 1.0, "cost": 0.0},
            },
        )
        contexts = (
            _matrix_context("static_best_unscored", served=0.99, e2e=2.0, annualized_cost=10.0),
            _matrix_context("runtime_best_scored", served=0.9, e2e=4.0, annualized_cost=10.0),
        )

        def fake_evaluate_runtime_performance(scenario, strategy):
            del strategy
            if scenario.scenario_id == "static_best_unscored":
                return SimpleNamespace(
                    to_dict=lambda: {
                        "evaluated": True,
                        "runtime_metrics": {
                            "served_ratio": 0.99,
                            "scope": "trace_backed_runtime_surrogate",
                            "trace_model": "surrogate_orbit_trace",
                            "valid_for_paper_main_results": False,
                        },
                        "slo": {"served_ratio": 0.99},
                        "performance": {"ttft_p95_s": None, "e2e_p95_s": None},
                        "warnings": ("missing tail latency",),
                    }
                )
            return _runtime_result(served=0.9, e2e=4.0)

        with patch("orbitserve.dse.runtime_matrix.evaluate_runtime_performance", fake_evaluate_runtime_performance):
            matrix = evaluate_runtime_matrix(contexts, matrix_spec).to_dict()

        nonsep = matrix["non_separability"]
        self.assertTrue(nonsep["evaluated"])
        self.assertEqual(nonsep["static_best_design_id"], "static_best_unscored")
        self.assertEqual(nonsep["runtime_best_design_id"], "runtime_best_scored")
        self.assertTrue(nonsep["static_runtime_best_differs"])
        self.assertFalse(nonsep["rank_comparison_complete"])
        self.assertIsNone(nonsep["rank_disagreement_count"])
        self.assertIn("static_best_unscored", nonsep["rank_comparison_missing_design_ids"])
        self.assertIsNone(matrix["metrics"]["rank_disagreement_count"])
        self.assertIn("static best design has no scored runtime row", "; ".join(nonsep["warnings"]))

    def test_shared_scorer_blocks_preliminary_by_default_and_missing_core_metrics(self):
        preliminary_result = {
            "evaluated": True,
            "runtime_metrics": {
                "served_ratio": 0.9,
                "e2e_p95_s": 4.0,
                "ttft_p95_s": 0.4,
                "scope": "idealized_compute_only",
                "trace_model": "ideal_full_visibility",
                "valid_for_paper_main_results": False,
            },
        }

        strict = score_runtime_evaluator_result(preliminary_result)
        self.assertIsNone(strict["objective_score"])
        self.assertIsNone(strict["objective_score_scope"])

        allowed = score_runtime_evaluator_result(
            preliminary_result,
            {
                "allow_preliminary_runtime_metrics": True,
                "ignore_optional_cost": True,
            },
        )
        self.assertIsNotNone(allowed["objective_score"])
        self.assertEqual(allowed["objective_score_scope"], "preliminary_idealized")
        self.assertFalse(allowed["valid_for_paper_main_results"])

        missing_core = score_runtime_evaluator_result(
            {
                "evaluated": True,
                "runtime_metrics": {
                    "served_ratio": 0.9,
                    "valid_for_paper_main_results": False,
                },
            },
            {
                "allow_preliminary_runtime_metrics": True,
                "ignore_optional_cost": True,
            },
        )
        self.assertIsNone(missing_core["objective_score"])
        self.assertTrue(
            any("served_ratio or e2e_p95_s is missing" in warning for warning in missing_core["warnings"])
        )

    def test_runtime_matrix_exposes_p99_leo_and_cost_efficiency_metrics(self):
        spec = DSERuntimeMatrixSpec(
            enabled=True,
            include_current_policy=False,
            policies=("orbitserve",),
            allow_preliminary_runtime_metrics=True,
            ignore_optional_cost=False,
            objective={
                "weights": {"served": 1.0, "tail": 1.0, "cost": 1.0e-6},
                "allow_preliminary_runtime_metrics": True,
            },
        )

        with patch(
            "orbitserve.dse.runtime_matrix.evaluate_runtime_performance",
            return_value=_runtime_result(served=0.8, e2e=4.0),
        ):
            result = evaluate_runtime_matrix(
                (_matrix_context("costed_design", served=0.7, e2e=5.0, annualized_cost=3_153_600.0),),
                spec,
            ).to_dict()

        row = result["matrix_rows"][0]
        self.assertEqual(row["ttft_p99_s"], 2.0)
        self.assertEqual(row["e2e_p99_s"], 6.0)
        self.assertAlmostEqual(row["leo_served_ratio"], 0.7)
        self.assertAlmostEqual(row["cost_per_served_request_usd"], 1.25)
        self.assertAlmostEqual(row["cost_per_served_mtoken_usd"], 125.0)
        self.assertIsNotNone(row["objective_score"])
        self.assertEqual(
            row["runtime_result"]["slo"]["cost_per_served_mtoken_usd"],
            row["cost_per_served_mtoken_usd"],
        )

    def test_runtime_matrix_leaves_ground_baseline_cost_per_served_null(self):
        spec = DSERuntimeMatrixSpec(
            enabled=True,
            include_current_policy=False,
            policies=("skywalker_like_ground",),
            allow_preliminary_runtime_metrics=True,
            ignore_optional_cost=True,
        )

        contaminated_payload = _runtime_result(served=1.0, e2e=1.8).to_dict()
        contaminated_payload["runtime_metrics"]["cost_per_served_request_usd"] = 123.0
        contaminated_payload["runtime_metrics"]["cost_per_served_mtoken_usd"] = 456.0
        contaminated_result = SimpleNamespace(to_dict=lambda: contaminated_payload)
        with patch(
            "orbitserve.dse.runtime_matrix.evaluate_runtime_performance",
            return_value=contaminated_result,
        ):
            result = evaluate_runtime_matrix(
                (_matrix_context("ground_context", served=0.7, e2e=5.0, annualized_cost=3_153_600.0),),
                spec,
            ).to_dict()

        row = result["matrix_rows"][0]
        self.assertEqual(row["policy"], "skywalker_like_ground")
        self.assertIsNone(row["cost_total_usd"])
        self.assertIsNone(row["annualized_cost_usd_per_year"])
        self.assertIsNone(row["cost_per_served_request_usd"])
        self.assertIsNone(row["cost_per_served_mtoken_usd"])
        self.assertIsNone(
            row["runtime_result"]["runtime_metrics"]["cost_per_served_request_usd"]
        )
        self.assertIsNone(
            row["runtime_result"]["runtime_metrics"]["cost_per_served_mtoken_usd"]
        )
        self.assertFalse(row["selected"])
        self.assertFalse(row["pareto_member"])
        self.assertTrue(
            any("terrestrial TCO" in warning for warning in row["warnings"])
        )

    def test_ground_reference_runs_once_outside_leo_design_and_scope_matrix(self):
        calls = []
        ground_config = {
            "probe_interval_s": 0.1,
            "remote_queue_buffer_requests": 1,
            "regions": [
                {"region_id": "us", "replica_count": 1, "local_rtt_s": 0.01},
                {"region_id": "eu", "replica_count": 1, "local_rtt_s": 0.01},
            ],
            "wan_rtt_s": {"us": {"eu": 0.12}, "eu": {"us": 0.12}},
        }
        spec = DSERuntimeMatrixSpec(
            enabled=True,
            include_current_policy=False,
            policies=("skywalker_like_ground",),
            ground_config=ground_config,
            placement_scope_candidates=(
                {
                    "strategy_id": "scope-a",
                    "pd_placement": {"placement_scope": "same_satellite"},
                },
                {
                    "strategy_id": "scope-b",
                    "pd_placement": {"placement_scope": "same_plane"},
                },
            ),
            static_screening={"enabled": True, "top_k_designs": 1},
            allow_preliminary_runtime_metrics=True,
            ignore_optional_cost=True,
        )

        def fake_evaluator(scenario, strategy):
            calls.append((scenario, strategy))
            return _runtime_result(served=1.0, e2e=1.8)

        contexts = (
            _matrix_context("leo-a", served=0.8, e2e=4.0, annualized_cost=100.0),
            _matrix_context("leo-b", served=0.7, e2e=5.0, annualized_cost=200.0),
        )
        with patch(
            "orbitserve.dse.runtime_matrix.evaluate_runtime_performance",
            side_effect=fake_evaluator,
        ):
            result = evaluate_runtime_matrix(contexts, spec).to_dict()

        ground_rows = [
            row
            for row in result["matrix_rows"]
            if row["policy"] == "skywalker_like_ground"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(ground_rows), 1)
        row = ground_rows[0]
        self.assertEqual(row["design_id"], "terrestrial_reference")
        self.assertFalse(row["static_screen_selected"])
        self.assertEqual(
            row["static_screen_reason"],
            "not_applicable_terrestrial_reference",
        )
        self.assertIsNone(row["placement_scope_strategy_id"])
        self.assertEqual(row["pd_scope_candidate_count"], 0)
        self.assertEqual(row["scheduler"]["ground_config"], ground_config)
        self.assertIsNone(getattr(calls[0][0], "design_id", None))
        self.assertFalse(row["selected"])
        self.assertFalse(row["pareto_member"])


if __name__ == "__main__":
    unittest.main()
