import inspect
import unittest
from types import SimpleNamespace

import orbitserve.runtime as runtime
from orbitserve.runtime.engine.contracts import RuntimeEngineResult
from orbitserve.runtime.metrics.records import RuntimeRequestRecord
from orbitserve.runtime.result import build_runtime_performance_result


class RuntimeResultContractTests(unittest.TestCase):
    def test_public_api_is_frozen_to_simulation_boundary(self):
        self.assertEqual(
            set(runtime.__all__),
            {
                "RuntimePerformanceResult",
                "RuntimeSimulationSpec",
                "RuntimeStrategySpec",
                "build_runtime_simulation_spec",
                "run_simulation",
            },
        )
        self.assertEqual(tuple(inspect.signature(runtime.run_simulation).parameters), ("simulation_spec",))
        self.assertFalse(hasattr(runtime, "RuntimeScenario"))
        self.assertFalse(hasattr(runtime, "run_runtime_engine"))

    def test_request_metrics_keep_slo_uncomputed(self):
        incomplete = RuntimeRequestRecord(
            request_id="incomplete",
            completed=False,
            ttft_s=100.0,
            e2e_s=200.0,
            slo_class="interactive",
            uncompleted_reason="no_route",
        )
        self.assertIsNone(incomplete.ttft_s)
        self.assertIsNone(incomplete.e2e_s)
        self.assertFalse(incomplete.slo["evaluated"])
        self.assertIsNone(incomplete.slo["attained"])

    def test_result_aggregates_only_completed_latency_and_keeps_slo_empty(self):
        records = (
            RuntimeRequestRecord(request_id="a", completed=True, prompt_tokens=10, output_tokens=2, ttft_s=1.0, e2e_s=2.0),
            RuntimeRequestRecord(request_id="b", completed=False, uncompleted_reason="no_route"),
        )
        spec = SimpleNamespace(
            scenario_id="unit",
            design_id="design",
            time_horizon_s=10.0,
            schema_version="orbitserve.runtime_simulation_spec",
            strategy=runtime.RuntimeStrategySpec(),
            provenance={},
            warnings=(),
        )
        result = build_runtime_performance_result(
            simulation_spec=spec,
            engine_result=RuntimeEngineResult(records, (), {}, ()),
        )
        self.assertEqual(result.workload_metrics["completion_ratio"], 0.5)
        self.assertEqual(result.workload_metrics["ttft_p95_s"], 1.0)
        self.assertFalse(result.slo_metrics["evaluated"])


if __name__ == "__main__":
    unittest.main()
