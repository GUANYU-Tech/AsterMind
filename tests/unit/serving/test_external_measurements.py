from copy import deepcopy
import unittest
from orbitserve.serving import ServingSpec, estimate_serving_throughput
from orbitserve.architecture import resolve_architecture
from orbitserve.serving.external import from_astermind_measurements


class ExternalServingMeasurementsTests(unittest.TestCase):
    def setUp(self):
        from orbitserve.architecture import HardwareSpec

        self.base = estimate_serving_throughput(
            ServingSpec(),
            hardware=resolve_architecture(HardwareSpec()).hardware_profile,
        )
        self.doc = {
            "schema_version": "orbitserve.astermind_measurements.v1",
            "model_name": self.base.model_name,
            "formation": self.base.formation,
            "parallelism": self.base.parallelism,
            "hardware": self.base.hardware,
            "scope": "phase_latency_including_cell_communication",
            "samples": [
                {
                    "phase": "prefill",
                    "length": 32,
                    "concurrency": 1,
                    "result": {
                        "status": "success",
                        "versions": {"astermind": "test-fixture"},
                        "metrics": {"total_latency_s": 2},
                    },
                },
                {
                    "phase": "decode",
                    "length": 32,
                    "concurrency": 1,
                    "decode_tokens": 1,
                    "result": {
                        "status": "success",
                        "versions": {"astermind": "test-fixture"},
                        "metrics": {"total_latency_s": 0.1},
                    },
                },
            ],
        }

    def test_measured_latency_is_used_without_double_adding_internal_communication(
        self,
    ):
        actual = from_astermind_measurements(self.base, self.doc)
        self.assertEqual(actual.prefill_rate_table[0]["prefill_tokens_per_s"], 16)
        self.assertEqual(actual.decode_rate_table[0]["decode_tokens_per_s"], 10)
        self.assertEqual(actual.internal_communication_accounting, "embedded")
        self.assertEqual(
            actual.provenance["execution_model"], "astermind_measured_static_batch"
        )

    def test_failed_results_mismatched_hardware_and_missing_concurrency_are_rejected(
        self,
    ):
        for mode in ("status", "hardware", "coverage"):
            doc = deepcopy(self.doc)
            if mode == "status":
                doc["samples"][0]["result"]["status"] = "failed"
            if mode == "hardware":
                doc["hardware"] = {}
            if mode == "coverage":
                doc["samples"].pop()
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                from_astermind_measurements(self.base, doc)
