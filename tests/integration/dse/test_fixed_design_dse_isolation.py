import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from orbitserve.dse import physical_design_candidates
from orbitserve.experiments import run_experiment
from tests.system_case import physical_case


class PhysicalDesignIsolationTests(unittest.TestCase):
    def test_fixed_design_and_empty_dse_use_same_evaluator(self):
        spec = physical_case()
        before = deepcopy(spec.to_dict())
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(
                replace(spec, dse={"enabled": True}), output_dir=folder, _workers=1
            )
            candidate = result.dse["rows"][0]["runtime"]
            self.assertEqual(
                candidate["request_summary"], result.runtime.request_summary
            )
            self.assertEqual(
                candidate["inference_metrics"], result.runtime.inference_metrics
            )
            self.assertEqual(spec.to_dict(), before)

    def test_duplicate_candidate_ids_are_rejected_and_physical_paths_are_safe(self):
        spec = physical_case()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            physical_design_candidates(
                spec.physical_network,
                {"design_candidates": [{"candidate_id": "x"}, {"candidate_id": "x"}]},
            )
        rows = physical_design_candidates(
            spec.physical_network,
            {
                "design_candidates": [
                    {"candidate_id": "../../one"},
                    {"candidate_id": "../../two"},
                ]
            },
        )
        self.assertTrue(
            all(
                "/" not in row["design_id"] and "\\" not in row["design_id"]
                for row in rows
            )
        )
        self.assertNotEqual(rows[0]["design_id"], rows[1]["design_id"])

    def test_hardware_is_not_an_orbit_sweep_and_unknown_fields_fail(self):
        for sweep in [{"accelerator": ["h200_sxm"]}, {"altitude_km": [750]}]:
            with self.subTest(sweep=sweep), self.assertRaises((ValueError, KeyError)):
                physical_design_candidates(
                    physical_case().physical_network, {"sweep": sweep}
                )

    def test_current_route_matrix_and_diagnostics_are_executed(self):
        spec = physical_case()
        dse = {
            "enabled": True,
            "runtime_matrix": {
                "enabled": True,
                "include_current_policy": False,
                "policies": ["min_hops", "earliest_arrival"],
                "static_screening": {"enabled": True},
            },
            "communication_window_budget": {
                "enabled": True,
                "candidate_satellite_budgets": [1, 2],
            },
        }
        with tempfile.TemporaryDirectory() as folder:
            result = run_experiment(
                replace(spec, dse=dse), output_dir=folder, _workers=1
            )
            self.assertEqual(result.dse["row_count"], 2)
            self.assertEqual(
                {r["strategy"]["routing_policy"] for r in result.dse["rows"]},
                {"min_hops", "earliest_arrival"},
            )
            for row in result.dse["rows"]:
                self.assertTrue(row["diagnostics"]["geometry_screening"]["evaluated"])
                self.assertTrue(
                    row["diagnostics"]["communication_window_budget"]["evaluated"]
                )

    def test_legacy_controller_configuration_is_not_silently_ignored(self):
        from orbitserve.dse import run_dse

        spec = replace(
            physical_case(),
            dse={
                "enabled": True,
                "runtime_matrix": {"enabled": True, "closed_loop": {"enabled": True}},
            },
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(ValueError, "closed_loop"),
        ):
            run_dse(spec, output_dir=folder, _workers=1)
