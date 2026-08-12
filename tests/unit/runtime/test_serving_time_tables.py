import unittest

from orbitserve.runtime.setup import (
    decode_time_s,
    prefill_full_time_s,
    prefill_suffix_time_s,
)


PREFILL = (
    {"prompt_length": 1, "prefill_tokens_per_s": 10.0},
    {"prompt_length": 10, "prefill_tokens_per_s": 5.0},
)
DECODE = (
    {"prompt_length": 1, "decode_tokens_per_s": 10.0},
    {"prompt_length": 10, "decode_tokens_per_s": 5.0},
)


class ServingTimeTableTests(unittest.TestCase):
    def test_prefill_interpolates_cumulative_time(self):
        self.assertAlmostEqual(prefill_full_time_s(PREFILL, 1), 0.1)
        self.assertAlmostEqual(prefill_full_time_s(PREFILL, 10), 2.0)
        self.assertAlmostEqual(prefill_full_time_s(PREFILL, 5), 0.1 + 4.0 / 9.0 * 1.9)

    def test_cached_prefill_is_cumulative_difference(self):
        expected = prefill_full_time_s(PREFILL, 10) - prefill_full_time_s(PREFILL, 5)
        self.assertAlmostEqual(
            prefill_suffix_time_s(PREFILL, prompt_length=10, cached_prefix_tokens=5),
            expected,
        )

    def test_decode_integrates_context_dependent_per_token_time(self):
        # Per-token time rises linearly from 0.1 s at P=1 to 0.2 s at P=10.
        self.assertAlmostEqual(decode_time_s(DECODE, prompt_length=1, output_tokens=9), 1.35)

    def test_lookup_rejects_context_outside_table(self):
        with self.assertRaisesRegex(ValueError, "decode_context_exceeds_serving_table"):
            decode_time_s(DECODE, prompt_length=10, output_tokens=1)


if __name__ == "__main__":
    unittest.main()
