import sys
import unittest
from pathlib import Path
from math import inf, nan


from orbitserve.dse.frontier import evaluate_dse_frontier


def _row(
    design_id,
    strategy_id,
    *,
    annualized_cost=100.0,
    total_cost=None,
    served_ratio=0.9,
    e2e_p95_s=10.0,
    objective_score=1.0,
    runtime_evaluated=True,
    valid_for_paper=False,
):
    if total_cost is None and annualized_cost is not None:
        total_cost = annualized_cost * 5
    return {
        "design_id": design_id,
        "strategy_id": strategy_id,
        "policy": strategy_id,
        "runtime_evaluated": runtime_evaluated,
        "objective_score": objective_score,
        "objective_score_scope": "preliminary_surrogate",
        "valid_for_paper_main_results": valid_for_paper,
        "annualized_cost_usd_per_year": annualized_cost,
        "cost_total_usd": total_cost,
        "served_ratio": served_ratio,
        "e2e_p95_s": e2e_p95_s,
        "ttft_p95_s": 1.0,
        "trace_model": "circular_geometry_trace",
        "scope": "geometry_trace_surrogate",
        "warnings": (),
        "provenance": {"unit_test": True},
    }


class DSEFrontierTests(unittest.TestCase):
    def test_preliminary_frontier_uses_multidimensional_pareto_and_excludes_missing_fields(self):
        matrix = {
            "evaluated": True,
            "allow_preliminary_runtime_metrics": True,
            "matrix_rows": [
                _row("design_a", "orbitserve", annualized_cost=100.0, served_ratio=0.95, e2e_p95_s=10.0),
                _row("design_b", "orbitserve", annualized_cost=120.0, served_ratio=0.90, e2e_p95_s=12.0),
                _row("design_c", "orbitserve", annualized_cost=80.0, served_ratio=0.90, e2e_p95_s=14.0),
                _row("design_missing", "orbitserve", annualized_cost=70.0, served_ratio=0.99, e2e_p95_s=None),
                _row("design_reserved", "paper_main", annualized_cost=60.0, served_ratio=0.99, e2e_p95_s=8.0, valid_for_paper=True),
            ],
        }

        frontier = evaluate_dse_frontier(matrix).to_dict()

        self.assertTrue(frontier["evaluated"])
        self.assertEqual(frontier["analysis_scope"], "preliminary_surrogate")
        self.assertEqual(frontier["calibration"], "not_stk_calibrated")
        self.assertFalse(frontier["valid_for_paper_main_results"])
        self.assertEqual(frontier["eligible_count"], 3)
        self.assertEqual(frontier["excluded_count"], 2)
        self.assertEqual(frontier["dominated_candidate_count"], 1)
        frontier_ids = {row["design_strategy_id"] for row in frontier["frontier_rows"]}
        self.assertEqual(frontier_ids, {"design_a::orbitserve", "design_c::orbitserve"})

        excluded_by_id = {row["design_strategy_id"]: row for row in frontier["excluded_candidates"]}
        self.assertIn("e2e_p95_s", excluded_by_id["design_missing::orbitserve"]["missing_fields"])
        self.assertIn("reserved", excluded_by_id["design_reserved::paper_main"]["excluded_reason"])
        for row in frontier["frontier_rows"]:
            self.assertFalse(row["valid_for_paper_main_results"])
            self.assertRegex(row["candidate_id"], r"::row_\d+$")

    def test_strict_runtime_matrix_blocks_top_level_frontier(self):
        matrix = {
            "evaluated": True,
            "allow_preliminary_runtime_metrics": False,
            "matrix_rows": [
                _row("design_a", "orbitserve", annualized_cost=100.0, served_ratio=0.95, e2e_p95_s=10.0),
            ],
        }

        frontier = evaluate_dse_frontier(matrix).to_dict()

        self.assertFalse(frontier["evaluated"])
        self.assertEqual(frontier["candidate_count"], 1)
        self.assertEqual(frontier["excluded_count"], 1)
        self.assertEqual(frontier["frontier_rows"], [])

    def test_total_cost_without_annualized_cost_is_excluded(self):
        matrix = {
            "evaluated": True,
            "allow_preliminary_runtime_metrics": True,
            "matrix_rows": [
                _row("annualized", "orbitserve", annualized_cost=100.0, served_ratio=0.9, e2e_p95_s=10.0),
                _row("total_only", "orbitserve", annualized_cost=None, total_cost=50.0, served_ratio=0.99, e2e_p95_s=1.0),
            ],
        }

        frontier = evaluate_dse_frontier(matrix).to_dict()

        self.assertTrue(frontier["evaluated"])
        frontier_ids = {row["design_strategy_id"] for row in frontier["frontier_rows"]}
        self.assertEqual(frontier_ids, {"annualized::orbitserve"})
        excluded = {row["design_strategy_id"]: row for row in frontier["excluded_candidates"]}
        self.assertIn("annualized_cost_usd_per_year", excluded["total_only::orbitserve"]["missing_fields"])
        self.assertEqual(excluded["total_only::orbitserve"]["cost_total_usd"], 50.0)

    def test_nan_and_inf_metrics_are_excluded(self):
        matrix = {
            "evaluated": True,
            "allow_preliminary_runtime_metrics": True,
            "matrix_rows": [
                _row("finite", "orbitserve", annualized_cost=100.0, served_ratio=0.9, e2e_p95_s=10.0),
                _row("nan_served", "orbitserve", annualized_cost=80.0, served_ratio=nan, e2e_p95_s=8.0),
                _row("inf_latency", "orbitserve", annualized_cost=70.0, served_ratio=0.95, e2e_p95_s=inf),
            ],
        }

        frontier = evaluate_dse_frontier(matrix).to_dict()

        self.assertTrue(frontier["evaluated"])
        self.assertEqual({row["design_strategy_id"] for row in frontier["frontier_rows"]}, {"finite::orbitserve"})
        excluded = {row["design_strategy_id"]: row for row in frontier["excluded_candidates"]}
        self.assertIn("served_ratio", excluded["nan_served::orbitserve"]["missing_fields"])
        self.assertIn("e2e_p95_s", excluded["inf_latency::orbitserve"]["missing_fields"])

    def test_core_metric_domain_violations_are_excluded(self):
        matrix = {
            "evaluated": True,
            "allow_preliminary_runtime_metrics": True,
            "matrix_rows": [
                _row("finite", "orbitserve", annualized_cost=100.0, served_ratio=0.9, e2e_p95_s=10.0),
                _row("negative_cost", "orbitserve", annualized_cost=-1.0, served_ratio=0.9, e2e_p95_s=9.0),
                _row("served_above_one", "orbitserve", annualized_cost=80.0, served_ratio=1.01, e2e_p95_s=8.0),
                _row("served_below_zero", "orbitserve", annualized_cost=70.0, served_ratio=-0.01, e2e_p95_s=7.0),
                _row("negative_latency", "orbitserve", annualized_cost=60.0, served_ratio=0.95, e2e_p95_s=-0.1),
            ],
        }

        frontier = evaluate_dse_frontier(matrix).to_dict()

        self.assertTrue(frontier["evaluated"])
        self.assertEqual({row["design_strategy_id"] for row in frontier["frontier_rows"]}, {"finite::orbitserve"})
        excluded = {row["design_strategy_id"]: row for row in frontier["excluded_candidates"]}
        self.assertIn("annualized_cost_usd_per_year", excluded["negative_cost::orbitserve"]["missing_fields"])
        self.assertIn("served_ratio", excluded["served_above_one::orbitserve"]["missing_fields"])
        self.assertIn("served_ratio", excluded["served_below_zero::orbitserve"]["missing_fields"])
        self.assertIn("e2e_p95_s", excluded["negative_latency::orbitserve"]["missing_fields"])
        for row in excluded.values():
            self.assertIn("missing/invalid core Pareto fields", row["excluded_reason"])

    def test_duplicate_design_strategy_rows_use_row_level_candidate_ids(self):
        matrix = {
            "evaluated": True,
            "allow_preliminary_runtime_metrics": True,
            "matrix_rows": [
                _row("design_a", "orbitserve", annualized_cost=100.0, served_ratio=0.95, e2e_p95_s=10.0),
                _row("design_a", "orbitserve", annualized_cost=100.0, served_ratio=0.95, e2e_p95_s=10.0),
            ],
        }

        frontier = evaluate_dse_frontier(matrix).to_dict()

        self.assertTrue(frontier["evaluated"])
        self.assertEqual(frontier["frontier_count"], 2)
        candidate_ids = [row["candidate_id"] for row in frontier["frontier_rows"]]
        self.assertEqual(len(candidate_ids), len(set(candidate_ids)))
        self.assertEqual(
            {row["design_strategy_id"] for row in frontier["frontier_rows"]},
            {"design_a::orbitserve"},
        )
        self.assertEqual({row["row_index"] for row in frontier["frontier_rows"]}, {0, 1})

    def test_frontier_does_not_depend_on_weighted_objective_score(self):
        matrix = {
            "evaluated": True,
            "allow_preliminary_runtime_metrics": True,
            "matrix_rows": [
                _row("pareto_good", "orbitserve", annualized_cost=50.0, served_ratio=0.99, e2e_p95_s=5.0, objective_score=None),
                _row("dominated_better_score", "orbitserve", annualized_cost=100.0, served_ratio=0.90, e2e_p95_s=10.0, objective_score=-999.0),
            ],
        }

        frontier = evaluate_dse_frontier(matrix).to_dict()

        self.assertTrue(frontier["evaluated"])
        self.assertEqual(
            {row["design_strategy_id"] for row in frontier["frontier_rows"]},
            {"pareto_good::orbitserve"},
        )
        self.assertIsNone(frontier["frontier_rows"][0]["objective_score"])
        self.assertEqual(frontier["dominated_candidate_count"], 1)

    def test_paper_main_rows_are_fail_closed_for_preliminary_evaluator(self):
        matrix = {
            "evaluated": True,
            "allow_preliminary_runtime_metrics": True,
            "matrix_rows": [
                _row("paper_design", "orbitserve", annualized_cost=50.0, served_ratio=0.99, e2e_p95_s=5.0, valid_for_paper=True),
            ],
        }

        frontier = evaluate_dse_frontier(matrix).to_dict()

        self.assertFalse(frontier["evaluated"])
        self.assertEqual(frontier["eligible_count"], 0)
        self.assertEqual(frontier["excluded_count"], 1)
        self.assertEqual(frontier["frontier_rows"], [])
        self.assertTrue(frontier["excluded_candidates"][0]["valid_for_paper_main_results"])
        self.assertIn("reserved", frontier["excluded_candidates"][0]["excluded_reason"])


if __name__ == "__main__":
    unittest.main()
