import unittest

from orbitserve.runtime import RuntimePerformanceResult
from orbitserve.runtime.engine.simulator import try_evaluate_runtime_performance
from orbitserve.runtime.engine.simulator import normalize_runtime_request
from orbitserve.runtime.metrics.runtime_metrics import aggregate_runtime_records
from orbitserve.runtime.metrics.runtime_records import RuntimeRequestRecord


class RuntimeResultContractTests(unittest.TestCase):
    def test_runtime_evaluator_entrypoint_is_importable(self):
        self.assertTrue(callable(try_evaluate_runtime_performance))

    def test_runtime_result_does_not_export_slo_group(self):
        result = RuntimePerformanceResult(
            evaluated=False,
            scenario={},
            strategy={},
        )

        self.assertNotIn("slo", result.to_dict())

    def test_latency_aggregates_only_include_completed_requests(self):
        completed = RuntimeRequestRecord(
            request_id="complete",
            arrival_time_s=0.0,
            prompt_tokens=10,
            output_tokens=2,
            completed=True,
            ttft_s=1.0,
            e2e_s=2.0,
        )
        incomplete = RuntimeRequestRecord(
            request_id="incomplete",
            arrival_time_s=0.0,
            prompt_tokens=10,
            output_tokens=2,
            completed=False,
            uncompleted_reason="no_route",
            ttft_s=100.0,
            e2e_s=200.0,
        )
        self.assertIsNone(incomplete.ttft_s)
        self.assertIsNone(incomplete.e2e_s)

        aggregates = aggregate_runtime_records(
            (completed, incomplete),
            time_horizon_s=10.0,
            trace_model="surrogate_orbit_trace",
            scope="trace_backed_runtime_surrogate",
        )

        self.assertEqual(aggregates.runtime_metrics["ttft_p95_s"], 1.0)
        self.assertEqual(aggregates.runtime_metrics["e2e_p95_s"], 2.0)
        self.assertEqual(aggregates.runtime_metrics["completion_ratio"], 0.5)

    def test_request_normalization_accepts_only_canonical_fields(self):
        canonical = normalize_runtime_request(
            {
                "request_id": "request-0",
                "arrival_time_s": 1.0,
                "prompt_tokens": 10,
                "output_tokens": 2,
                "ingress_id": "gateway-0",
            }
        )
        self.assertEqual(canonical["prompt_tokens"], 10)
        self.assertEqual(canonical["ingress_id"], "gateway-0")

        with self.assertRaisesRegex(ValueError, "prompt_tokens and output_tokens"):
            normalize_runtime_request(
                {
                    "request_id": "legacy-request",
                    "time_s": 1.0,
                    "input_tokens": 10,
                    "decode_tokens": 2,
                    "gateway_id": "gateway-0",
                }
            )


if __name__ == "__main__":
    unittest.main()
