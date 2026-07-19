import sys
import unittest
from pathlib import Path


from orbitserve.dse.search import DSESpec
from orbitserve.dse.satellite_budget import analyze_communication_window_satellite_budget_from_bundle
from orbitserve.orbit.simulation_traces import CIRCULAR_GEOMETRY_TRACE, RuntimeTraceBundle


def _bundle():
    return RuntimeTraceBundle(
        trace_id="unit-budget",
        trace_model=CIRCULAR_GEOMETRY_TRACE,
        satellite_catalog=tuple({"satellite_id": f"sat-{index}"} for index in range(4)),
        access_windows=(
            {
                "ingress_id": "gw-0",
                "satellite_id": "sat-0",
                "start_s": 0.0,
                "end_s": 60.0,
                "propagation_delay_s": 0.01,
                "tau_setup_s": 0.1,
            },
            {
                "ingress_id": "gw-0",
                "satellite_id": "sat-1",
                "start_s": 0.0,
                "end_s": 60.0,
                "propagation_delay_s": 0.02,
                "tau_setup_s": 0.2,
            },
            {
                "ingress_id": "gw-0",
                "satellite_id": "sat-2",
                "start_s": 60.0,
                "end_s": 120.0,
                "propagation_delay_s": 0.03,
                "tau_setup_s": 0.3,
            },
            {
                "ingress_id": "gw-1",
                "satellite_id": "sat-3",
                "start_s": 0.0,
                "end_s": 120.0,
                "propagation_delay_s": 0.04,
                "tau_setup_s": 0.4,
            },
        ),
        visible_count_samples=(
            {"time_s": 0.0, "ingress_id": "gw-0", "geometry_visible_count": 0},
            {"time_s": 30.0, "ingress_id": "gw-0", "geometry_visible_count": 1},
            {"time_s": 60.0, "ingress_id": "gw-0", "geometry_visible_count": 2},
            {"time_s": 90.0, "ingress_id": "gw-0", "geometry_visible_count": 4},
        ),
    )


class CommunicationWindowBudgetTests(unittest.TestCase):
    def test_budget_analysis_selects_min_budget_reaching_window_capacity_target(self):
        result = analyze_communication_window_satellite_budget_from_bundle(
            _bundle(),
            config={
                "enabled": True,
                "candidate_satellite_budgets": [1, 2, "visible_max"],
                "target_coverage_ratio": 0.95,
            },
        )

        self.assertTrue(result["evaluated"])
        self.assertFalse(result["valid_for_paper_main_results"])
        self.assertEqual(result["visible_satellite_count"], 4)
        self.assertEqual(result["visible_count_summary"]["zero_visible_sample_fraction"], 0.25)
        self.assertEqual([row["satellite_budget_used"] for row in result["candidate_budget_rows"]], [1, 2, 4])
        self.assertLess(
            result["candidate_budget_rows"][0]["window_limited_capacity_factor"],
            result["candidate_budget_rows"][-1]["window_limited_capacity_factor"],
        )
        self.assertEqual(result["selected_budget"]["satellite_budget_used"], 4)
        self.assertEqual(result["best_performance_budget"]["satellite_budget_used"], 4)

    def test_budget_analysis_uses_request_weighted_access_window_counts_when_available(self):
        result = analyze_communication_window_satellite_budget_from_bundle(
            _bundle(),
            workload_generation={
                "requests": (
                    {"request_id": "r0", "arrival_time_s": 10.0, "gateway_id": "gw-0"},
                    {"request_id": "r1", "arrival_time_s": 70.0, "gateway_id": "gw-0"},
                )
            },
            config={
                "enabled": True,
                "candidate_satellite_budgets": [1, 2],
                "target_coverage_ratio": 0.95,
            },
        )

        rows = result["candidate_budget_rows"]
        self.assertEqual(result["visible_count_summary"]["request_observation_count"], 2)
        self.assertEqual(result["visible_count_summary"]["request_weighted_visible_count_max"], 2)
        self.assertEqual([row["satellite_budget_used"] for row in rows], [1, 2])
        self.assertAlmostEqual(rows[0]["window_limited_capacity_factor"], 0.75)
        self.assertEqual(result["selected_budget"]["satellite_budget_used"], 2)

    def test_dse_spec_parses_communication_window_budget_config(self):
        spec = DSESpec.from_dict(
            {
                "communication_window_budget": {
                    "enabled": True,
                    "candidate_satellite_budgets": [1, 2, "visible_p95"],
                }
            }
        )

        self.assertTrue(spec.communication_window_budget["enabled"])
        self.assertEqual(spec.to_dict()["communication_window_budget"]["candidate_satellite_budgets"], [1, 2, "visible_p95"])


if __name__ == "__main__":
    unittest.main()
