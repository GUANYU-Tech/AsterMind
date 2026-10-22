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
                "run_simulation",
                "check_network_backend",
                "export_cell_network",
                "LinkProfile", "NetworkNodeSpec", "NetworkSpec", "NetworkState",
                "NetworkUnavailable", "candidate_links",
                "STARLINK_RADIO_PROFILES",
            },
        )
        self.assertEqual(tuple(inspect.signature(runtime.run_simulation).parameters), ("simulation_spec",))
        self.assertFalse(hasattr(runtime, "RuntimeScenario"))
        self.assertFalse(hasattr(runtime, "run_runtime_engine"))

    def test_incomplete_request_has_no_user_latency_or_slo_section(self):
        incomplete = RuntimeRequestRecord(
            request_id="incomplete",
            completed=False,
            output_tokens=2,
            ttft_s=100.0,
            e2e_s=200.0,
            uncompleted_reason="no_route",
        )
        self.assertIsNone(incomplete.ttft_s)
        self.assertIsNone(incomplete.tpot_s)
        self.assertIsNone(incomplete.e2e_s)
        self.assertNotIn("slo", incomplete.to_dict())

    def test_tpot_uses_user_visible_streaming_interval(self):
        record = RuntimeRequestRecord(
            request_id="a",
            completed=True,
            output_tokens=5,
            ttft_s=2.0,
            e2e_s=6.0,
        )
        self.assertEqual(record.tpot_s, 1.0)
        single_token = RuntimeRequestRecord(
            request_id="b",
            completed=True,
            output_tokens=1,
            ttft_s=2.0,
            e2e_s=2.0,
        )
        self.assertIsNone(single_token.tpot_s)

    def test_result_emits_only_three_inference_metric_distributions(self):
        records = (
            RuntimeRequestRecord(
                request_id="a",
                completed=True,
                prompt_tokens=10,
                output_tokens=2,
                ttft_s=1.0,
                e2e_s=2.0,
            ),
            RuntimeRequestRecord(
                request_id="b",
                completed=True,
                prompt_tokens=10,
                output_tokens=3,
                ttft_s=3.0,
                e2e_s=7.0,
            ),
            RuntimeRequestRecord(request_id="c", completed=False, uncompleted_reason="no_route"),
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
        self.assertEqual(set(result.inference_metrics), {"ttft_s", "tpot_s", "e2e_s"})
        for metric in result.inference_metrics.values():
            self.assertEqual(set(metric), {"average", "p50", "p95", "p99"})
        self.assertEqual(result.inference_metrics["ttft_s"]["average"], 2.0)
        self.assertEqual(result.inference_metrics["tpot_s"]["average"], 1.5)
        self.assertEqual(result.inference_metrics["e2e_s"]["average"], 4.5)
        self.assertEqual(result.request_summary["completed_request_count"], 2)
        self.assertEqual(result.request_summary["uncompleted_request_count"], 1)
        payload = result.to_dict()
        self.assertNotIn("workload_metrics", payload)
        self.assertNotIn("slo_metrics", payload)


if __name__ == "__main__":
    unittest.main()
