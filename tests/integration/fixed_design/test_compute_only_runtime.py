import importlib
import sys
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.runtime.contracts import RuntimeScenario, RuntimeStrategySpec, ServingDeploymentView
from orbitserve.runtime.evaluation import evaluate_runtime_performance
from orbitserve.runtime.runtime_optimization import optimize_runtime_strategy


ROOT = Path(__file__).resolve().parents[3]


def _request(request_id, arrival_time_s, prompt_tokens=128, output_tokens=32, e2e_deadline_s=30.0):
    return {
        "request_id": request_id,
        "arrival_time_s": arrival_time_s,
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_deadline_s": 5.0,
        "e2e_deadline_s": e2e_deadline_s,
        "slo_class": "interactive_5s",
    }


class ComputeOnlyRuntimeTests(unittest.TestCase):
    def _serving_performance(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_single_design.json")
        return run_experiment(spec).performance.to_dict()

    def test_ideal_full_visibility_local_only_runtime_evaluates_request_records(self):
        performance = self._serving_performance()
        scenario = RuntimeScenario(
            scenario_id="compute_only_unit",
            time_horizon_s=60.0,
            workload_bundle={"status": "loaded", "request_count": 2},
            requests=(
                _request("req-0", 0.0),
                _request("req-1", 0.0),
            ),
            simulation_trace_ref={
                "evaluated": True,
                "trace_model": "ideal_full_visibility",
                "trace_manifest": {
                    "evaluated": True,
                    "trace_model": "ideal_full_visibility",
                    "scope": "idealized_compute_only",
                    "valid_for_paper_main_results": False,
                },
            },
            serving_deployments=(ServingDeploymentView.from_value(performance),),
            runtime_config=RuntimeStrategySpec.from_dict({"policy": "local_only"}),
        )

        result = evaluate_runtime_performance(scenario).to_dict()

        self.assertTrue(result["evaluated"])
        self.assertEqual(result["runtime_metrics"]["trace_model"], "ideal_full_visibility")
        self.assertEqual(result["runtime_metrics"]["scope"], "idealized_compute_only")
        self.assertFalse(result["runtime_metrics"]["valid_for_paper_main_results"])
        self.assertEqual(result["runtime_metrics"]["total_request_count"], 2)
        self.assertIn("request_records", result["details"])
        self.assertEqual(len(result["details"]["request_records"]), 2)
        self.assertFalse(result["communication_metrics"]["evaluated"])
        self.assertFalse(result["energy_metrics"]["evaluated"])

    def test_runner_backfills_top_level_slo_only_for_idealized_local_only_scope(self):
        base = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_single_design.json")
        spec = replace(
            base,
            workload=replace(
                base.workload,
                bundle_dir="tests/fixtures/workloads/smoke_flat_single_turn_direct",
            ),
            simulation=replace(base.simulation, trace_model="ideal_full_visibility"),
            scheduler=replace(base.scheduler, policy="local_only"),
        )

        payload = run_experiment(spec).to_dict()

        self.assertEqual(payload["module_status"]["simulation"], "idealized_trace")
        self.assertEqual(payload["module_status"]["runtime_performance"], "evaluated")
        self.assertTrue(payload["slo"]["evaluated"])
        self.assertIsNotNone(payload["slo"]["served_ratio"])
        self.assertIn("valid_for_paper_main_results=false", payload["slo"]["warnings"])
        runtime = payload["performance"]["details"]["runtime"]
        self.assertTrue(runtime["evaluated"])
        self.assertFalse(runtime["runtime_metrics"]["valid_for_paper_main_results"])

    def test_optimizer_refuses_to_score_idealized_runtime_by_default(self):
        base = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_single_design.json")
        experiment = run_experiment(
            replace(
                base,
                workload=replace(
                    base.workload,
                    bundle_dir="tests/fixtures/workloads/smoke_flat_single_turn_direct",
                ),
                simulation=replace(base.simulation, trace_model="ideal_full_visibility"),
                scheduler=replace(base.scheduler, policy="local_only"),
            )
        )
        runtime_payload = experiment.performance.to_dict()["details"]["runtime"]
        # Basic check that the evaluator produced metrics, then ensure optimizer
        # still treats them as audit-only unless explicitly allowed.
        self.assertTrue(runtime_payload["evaluated"])

        optimizer = experiment.performance.to_dict()["details"]["runtime_optimization"]
        self.assertFalse(optimizer["evaluated"])
        self.assertIsNone(optimizer["selected_strategy"])
        local_only_rows = [
            row for row in optimizer["score_table"]
            if row["scheduler"]["policy"] == "local_only"
        ]
        self.assertTrue(local_only_rows)
        self.assertIsNone(local_only_rows[0]["objective_score"])
        self.assertTrue(
            any("idealized" in warning for warning in local_only_rows[0]["warnings"])
        )


if __name__ == "__main__":
    unittest.main()
