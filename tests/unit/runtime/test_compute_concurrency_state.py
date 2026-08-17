import unittest

from orbitserve.runtime.state.resources import ComputeConcurrencyState


class ComputeConcurrencyStateTests(unittest.TestCase):
    def test_admits_until_limit_then_waits_for_earliest_finish(self):
        state = ComputeConcurrencyState(max_concurrency=2)
        first = state.preview(request_id="a", ready_s=0.0, duration_for=lambda c, t: 10.0)
        state.reserve(first)
        second = state.preview(request_id="b", ready_s=1.0, duration_for=lambda c, t: 5.0)
        state.reserve(second)
        third = state.preview(request_id="c", ready_s=2.0, duration_for=lambda c, t: 1.0)
        self.assertEqual(first.concurrency_at_admission, 1)
        self.assertEqual(second.concurrency_at_admission, 2)
        self.assertEqual(third.start_s, 6.0)
        self.assertEqual(third.concurrency_at_admission, 2)

    def test_future_reservation_cannot_be_overbooked(self):
        state = ComputeConcurrencyState(max_concurrency=1)
        future = state.preview(request_id="future", ready_s=5.0, duration_for=lambda c, t: 5.0)
        state.reserve(future)
        row = state.preview(request_id="long", ready_s=0.0, duration_for=lambda c, t: 8.0)
        self.assertEqual(row.start_s, 10.0)

    def test_commit_rejects_stale_preview(self):
        state = ComputeConcurrencyState(max_concurrency=1)
        stale = state.preview(request_id="stale", ready_s=0.0, duration_for=lambda c, t: 1.0)
        other = state.preview(request_id="other", ready_s=0.0, duration_for=lambda c, t: 2.0)
        state.reserve(other)
        with self.assertRaisesRegex(ValueError, "candidate_preview_state_changed"):
            state.reserve(stale)


if __name__ == "__main__":
    unittest.main()
