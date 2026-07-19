import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.experiments import load_experiment_spec, run_experiment
from orbitserve.runtime.contracts import RuntimeScenario, RuntimeStrategySpec
from orbitserve.runtime.evaluation import evaluate_runtime_performance
from orbitserve.runtime.deployment_contracts import RuntimeDeployment, RuntimeDeploymentCatalog
from orbitserve.runtime.engines.trace_backed import run_trace_backed_runtime
from orbitserve.orbit.simulation_trace_builder import build_surrogate_orbit_trace
from orbitserve.orbit.simulation_traces import (
    OrbitTraceManifest,
    TRACE_BACKED_RUNTIME_SURROGATE_SCOPE,
    load_surrogate_trace_bundle,
)


ROOT = Path(__file__).resolve().parents[2]


def _request(request_id, arrival_time_s, **overrides):
    row = {
        "request_id": request_id,
        "arrival_time_s": arrival_time_s,
        "ingress_id": "test_ingress",
        "prompt_tokens": 100,
        "output_tokens": 20,
        "input_bytes": 10_000_000,
        "output_bytes": 1_000_000,
        "ttft_deadline_s": 60.0,
        "e2e_deadline_s": 300.0,
        "session_id": "sess-0",
    }
    row.update(overrides)
    return row


class TraceBackedRuntimeTests(unittest.TestCase):
    def _trace_bundle(self, requests, trace_options, *, gateway_bandwidth_bps=None):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_trace_backed_runtime.json")
        constellation = spec.constellation
        if gateway_bandwidth_bps is not None:
            constellation = replace(
                constellation,
                gateway_profile={
                    **constellation.gateway_profile,
                    "bandwidth_bps": gateway_bandwidth_bps,
                },
            )
        workload = {
            "requests": tuple(requests),
            "regions": (),
            "gateways": (),
            "manifest": {"schema_version": "orbitserve.workload"},
            "summary": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                spec.simulation,
                trace_output_dir=tmpdir,
                trace_options={**spec.simulation.trace_options, **trace_options},
            )
            result = build_surrogate_orbit_trace(
                config=config,
                constellation=constellation,
                workload_generation=workload,
                base_dir=ROOT,
                experiment_id="unit_runtime",
            )
            return load_surrogate_trace_bundle(OrbitTraceManifest.from_value(result.to_trace_manifest()))

    def test_engine_records_link_queue_and_energy_violation(self):
        requests = (
            _request("req-0", 0.0, ttft_deadline_s=300.0),
            _request("req-1", 0.0, ttft_deadline_s=300.0),
        )
        bundle = self._trace_bundle(
            requests,
            {
                "visible_satellite_count": 1,
                "initial_soc": 0.01,
                "soc_min": 0.2,
                "energy_accounting_mode": "event_incremental_v1",
                "compute_energy_j_per_token": 1000.0,
                "link_energy_j_per_byte": 1e-3,
            },
            gateway_bandwidth_bps=1_000_000.0,
        )
        catalog = RuntimeDeploymentCatalog(
            deployments=(
                RuntimeDeployment(
                    deployment_id="deployment_0",
                    prefill_tokens_per_s=1_000_000.0,
                    decode_tokens_per_s=1_000_000.0,
                ),
            )
        )
        result = run_trace_backed_runtime(
            requests=requests,
            deployment_catalog=catalog,
            runtime_strategy=RuntimeStrategySpec.from_dict(
                {"policy": "nearest_visible", "hard_gate_mode": "audit_only"}
            ),
            trace_bundle=bundle,
            scope="trace_backed_runtime_surrogate",
        )

        self.assertTrue(result.evaluated)
        self.assertEqual(result.scope, TRACE_BACKED_RUNTIME_SURROGATE_SCOPE)
        self.assertEqual(len(result.records), 2)
        self.assertGreater(result.records[1].sgl_queue_wait_s or 0.0, 0.0)
        self.assertTrue(any(record.energy_violation for record in result.records))
        self.assertEqual(result.records[0].session_action, "sticky_kv_owner")

    def test_runner_smoke_returns_surrogate_runtime_metrics(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_trace_backed_runtime.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = replace(spec, simulation=replace(spec.simulation, trace_output_dir=tmpdir))
            payload = run_experiment(spec).to_dict()

        runtime = payload["performance"]["details"]["runtime"]
        optimizer = payload["performance"]["details"]["runtime_optimization"]
        self.assertTrue(runtime["evaluated"])
        self.assertEqual(runtime["trace_model"], "surrogate_orbit_trace")
        self.assertEqual(runtime["scope"], TRACE_BACKED_RUNTIME_SURROGATE_SCOPE)
        self.assertFalse(runtime["valid_for_paper_main_results"])
        self.assertEqual(runtime["runtime_metrics"]["trace_model"], "surrogate_orbit_trace")
        self.assertEqual(runtime["runtime_metrics"]["scope"], TRACE_BACKED_RUNTIME_SURROGATE_SCOPE)
        self.assertFalse(runtime["runtime_metrics"]["valid_for_paper_main_results"])
        self.assertTrue(runtime["communication_metrics"]["evaluated"])
        self.assertTrue(runtime["energy_metrics"]["evaluated"])
        self.assertEqual(len(runtime["request_records"]), runtime["runtime_metrics"]["total_request_count"])
        self.assertIn("score_terms", runtime["request_records"][0]["provenance"])
        self.assertTrue(optimizer["evaluated"])
        self.assertTrue(optimizer["closed_loop_evaluated"])
        self.assertEqual(runtime["strategy"]["policy"], optimizer["selected_strategy"]["policy"])
        self.assertIn("runtime_initial", payload["performance"]["details"])
        self.assertFalse(optimizer["hardware_optimization"]["evaluated"])
        self.assertIn("preliminary surrogate", " ".join(payload["performance"]["warnings"]))

    def test_new_runtime_result_fields_are_rejected_in_workload_input(self):
        scenario = RuntimeScenario(
            scenario_id="leakage",
            requests=(
                {
                    "request_id": "req-leak",
                    "arrival_time_s": 0.0,
                    "prompt_tokens": 1,
                    "output_tokens": 1,
                    "assigned_satellite_id": "sat-00000",
                    "fallback_used": True,
                },
            ),
        )

        with self.assertRaisesRegex(ValueError, "assigned_satellite_id|fallback_used|runtime leakage"):
            evaluate_runtime_performance(scenario, RuntimeStrategySpec.from_dict({"policy": "local_only"}))


if __name__ == "__main__":
    unittest.main()
