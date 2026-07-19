import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace


from orbitserve.dse.search import run_dse
from orbitserve.dse.frontier import evaluate_dse_frontier
from orbitserve.runtime.energy_alignment import evaluate_energy_alignment
from orbitserve.experiments import load_experiment_spec


ROOT = Path(__file__).resolve().parents[3]


class EnergyAlignedDSETests(unittest.TestCase):
    def test_two_candidate_trace_dse_scores_preliminary_rows_and_keeps_energy_audits(self):
        base = load_experiment_spec(
            ROOT / "tests/fixtures/manifests/smoke_circular_geometry_runtime_matrix.json"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = replace(
                base,
                simulation=replace(base.simulation, trace_output_dir=str(Path(tmpdir) / "traces")),
            )
            payload = run_dse(spec).to_dict()

        self.assertTrue(payload["evaluated"])
        self.assertEqual(payload["design_count"], 2)
        self.assertEqual(len(payload["rows"]), 2)
        self.assertTrue(payload["runtime_matrix"]["evaluated"])
        self.assertGreater(payload["runtime_matrix"]["row_count"], 0)
        # This fixture explicitly enables preliminary surrogate scoring, so
        # each design contributes one selected runtime row and a frontier row.
        self.assertEqual(len(payload["runtime_matrix"]["selected_rows"]), 2)
        self.assertTrue(payload["frontier"]["evaluated"])
        self.assertEqual(payload["frontier"]["frontier_count"], 2)

        for row in payload["rows"]:
            self.assertTrue(row["cost"]["evaluated"])
            details = row["performance"]["details"]
            planned = details["planned_energy_demand"]
            self.assertEqual(planned["sizing_mode"], "offered_workload_energy_audit")
            self.assertEqual(planned["offered_request_count"], 79)
            self.assertFalse(planned["uses_runtime_served_requests"])
            alignment = details["energy_alignment"]
            self.assertIn("evaluated", alignment)
            self.assertFalse(alignment["provenance"]["runtime_records_used_for_cost_sizing"])

    def test_target_miss_forces_inconsistent_without_changing_planned_energy(self):
        planned = {
            "evaluated": True,
            "offered_request_count": 2,
            "target_served_ratio": 1.0,
            "total_compute_energy_j": 10.0,
            "total_link_energy_j": 2.0,
            "total_dynamic_energy_j": 12.0,
            "compute_energy_j_per_token": 0.5,
            "link_energy_j_per_byte": 1e-6,
            "uses_runtime_served_requests": False,
        }
        ledger = {
            "energy_kind": "request_event",
            "satellite_id": "sat-a",
            "event_id": "energy-event-0",
            "compute_bus_j": 10.0,
            "link_bus_j": 2.0,
            "compute_tokens": 20.0,
            "link_bytes_accounted": 2_000_000.0,
            "compute_energy_j_per_token": 0.5,
            "link_energy_j_per_byte": 1e-6,
            "base_compute_energy_j_per_token": 0.5,
            "base_link_energy_j_per_byte": 1e-6,
            "runtime_compute_energy_scale": 1.0,
            "runtime_link_energy_scale": 1.0,
        }
        runtime = SimpleNamespace(
            evaluated=True,
            runtime_metrics={"served_ratio": 0.5},
            request_records=(
                {"accepted": True, "served": True, "energy_ledger": (ledger,)},
                # The duplicate ledger row proves the audit does not double count.
                {"accepted": False, "served": False, "energy_ledger": (ledger,)},
            ),
            details={},
            to_dict=lambda: {
                "evaluated": True,
                "runtime_metrics": {"served_ratio": 0.5},
            },
        )
        envelope = {
            "trace_options": {
                "compute_energy_j_per_token": 0.5,
                "link_energy_j_per_byte": 1e-6,
            }
        }

        result = evaluate_energy_alignment(
            planned,
            runtime,
            runtime_energy_envelope=envelope,
        ).to_dict()

        self.assertTrue(result["evaluated"])
        self.assertEqual(result["difference_fraction"], 0.0)
        self.assertTrue(result["coefficients_match"])
        self.assertFalse(result["target_served_ratio_met"])
        self.assertFalse(result["consistent"])
        self.assertEqual(result["planned_energy"]["planned_dynamic_energy_j"], 12.0)
        self.assertEqual(result["runtime_energy"]["runtime_dynamic_energy_j"], 12.0)
        self.assertEqual(result["runtime_energy"]["request_event_count"], 1)

    def test_runtime_strategy_that_exceeds_energy_plan_is_excluded_from_frontier(self):
        result = evaluate_dse_frontier(
            {
                "evaluated": True,
                "allow_preliminary_runtime_metrics": True,
                "matrix_rows": (
                    {
                        "design_id": "design-a",
                        "strategy_id": "strategy-too-expensive",
                        "policy": "energy_aware",
                        "runtime_evaluated": True,
                        "valid_for_paper_main_results": False,
                        "annualized_cost_usd_per_year": 100.0,
                        "cost_total_usd": 1000.0,
                        "served_ratio": 0.9,
                        "e2e_p95_s": 1.0,
                        "provenance": {"cost_feasible": True},
                        "runtime_result": {
                            "details": {
                                "energy_alignment": {
                                    "evaluated": True,
                                    "requires_resizing": True,
                                }
                            }
                        },
                    },
                ),
            }
        ).to_dict()

        self.assertFalse(result["evaluated"])
        self.assertEqual(result["eligible_count"], 0)
        self.assertEqual(result["excluded_count"], 1)
        self.assertIn(
            "resize the design",
            result["excluded_candidates"][0]["excluded_reason"],
        )


if __name__ == "__main__":
    unittest.main()
