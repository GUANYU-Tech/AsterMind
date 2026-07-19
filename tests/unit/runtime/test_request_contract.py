import unittest

from orbitserve.runtime.request_contract import normalize_runtime_request


class RuntimeRequestContractTests(unittest.TestCase):
    def test_canonical_request_is_normalized_without_field_translation(self):
        request = normalize_runtime_request(
            {
                "request_id": "req-1",
                "arrival_time_s": 1.5,
                "prompt_tokens": 128,
                "output_tokens": 32,
                "ingress_id": "gw-1",
                "ttft_deadline_s": 2.0,
                "e2e_deadline_s": 10.0,
            }
        )
        self.assertEqual(request["request_id"], "req-1")
        self.assertEqual(request["prompt_tokens"], 128)
        self.assertEqual(request["ingress_id"], "gw-1")

    def test_old_request_aliases_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "canonical workload contract"):
            normalize_runtime_request(
                {
                    "request_id": "req-old",
                    "time_s": 1.5,
                    "input_tokens": 128,
                    "max_output_tokens": 32,
                    "gateway_id": "gw-1",
                }
            )


if __name__ == "__main__":
    unittest.main()
