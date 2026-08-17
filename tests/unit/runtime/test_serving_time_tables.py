import unittest

from orbitserve.runtime.setup import (
    decode_time_s,
    prefill_full_time_s,
    prefill_suffix_time_s,
)


PREFILL = tuple(
    {"prompt_length": length, "concurrency": concurrency, "prefill_tokens_per_s": rate}
    for length, rates in ((1, (10.0, 5.0)), (10, (5.0, 2.5)))
    for concurrency, rate in enumerate(rates, 1)
)
DECODE = tuple(
    {"context_length": length, "concurrency": concurrency, "decode_tokens_per_s": rate}
    for length, rates in ((1, (10.0, 5.0)), (10, (5.0, 2.5)))
    for concurrency, rate in enumerate(rates, 1)
)


class ServingTimeTableTests(unittest.TestCase):
    def test_prefill_interpolates_cumulative_time_in_exact_slice(self):
        self.assertAlmostEqual(prefill_full_time_s(PREFILL, 1, 1), 0.1)
        self.assertAlmostEqual(prefill_full_time_s(PREFILL, 10, 1), 2.0)
        self.assertAlmostEqual(
            prefill_full_time_s(PREFILL, 5, 1),
            0.1 + 4.0 / 9.0 * 1.9,
        )
        self.assertAlmostEqual(prefill_full_time_s(PREFILL, 10, 2), 4.0)

    def test_cached_prefill_uses_one_concurrency_slice(self):
        expected = prefill_full_time_s(PREFILL, 10, 2) - prefill_full_time_s(PREFILL, 5, 2)
        self.assertAlmostEqual(
            prefill_suffix_time_s(
                PREFILL,
                prompt_length=10,
                cached_prefix_tokens=5,
                concurrency=2,
            ),
            expected,
        )

    def test_decode_integrates_context_time_at_fixed_concurrency(self):
        self.assertAlmostEqual(
            decode_time_s(DECODE, context_length=1, output_tokens=9, concurrency=1),
            1.35,
        )
        self.assertAlmostEqual(
            decode_time_s(DECODE, context_length=1, output_tokens=9, concurrency=2),
            2.7,
        )

    def test_concurrency_must_match_exact_table_slice(self):
        with self.assertRaisesRegex(ValueError, "serving_concurrency_not_in_table"):
            prefill_full_time_s(PREFILL, 1, 3)

    def test_lookup_rejects_context_outside_table(self):
        with self.assertRaisesRegex(ValueError, "decode_context_exceeds_serving_table"):
            decode_time_s(DECODE, context_length=10, output_tokens=1, concurrency=1)


if __name__ == "__main__":
    unittest.main()
