import unittest

from orbitserve.runtime.state.kv_cache import (
    _admit_session_kv,
    commit_session_kv,
    local_kv_reuse_tokens,
)
from orbitserve.runtime.state.runtime_state import RuntimeState


class KVCapacityLRUTests(unittest.TestCase):
    def setUp(self):
        self.state = RuntimeState()

    def admit(self, session_id, tokens, now_s, satellite_id="sat-a"):
        self.state.session_for(session_id)
        return _admit_session_kv(
            self.state,
            capacity_tokens=10,
            capacity_bytes=None,
            kv_policy={},
            session_id=session_id,
            satellite_id=satellite_id,
            tokens=tokens,
            kv_bytes_per_token=1.0,
            now_s=now_s,
        )

    def test_capacity_pressure_evicts_lru_without_time_expiry(self):
        self.admit("a", 4, 1.0)
        self.admit("b", 4, 10_000.0)
        result = self.admit("c", 4, 20_000.0)
        self.assertEqual(result["evicted_session_ids"], ("a",))
        self.assertEqual(set(self.state.kv_cache_for("sat-a").entries), {"b", "c"})

    def test_local_reuse_is_bounded_by_required_context(self):
        self.admit("a", 8, 1.0)
        self.assertEqual(
            local_kv_reuse_tokens(
                self.state,
                satellite_id="sat-a",
                session_id="a",
                required_context_tokens=3,
            ),
            3,
        )

    def test_copy_retains_source_after_target_admission(self):
        commit_session_kv(
            self.state,
            session_id="a",
            satellite_id="sat-a",
            context_tokens=4,
            capacity_tokens=10,
            kv_bytes_per_token=1.0,
            now_s=1.0,
        )
        result = commit_session_kv(
            self.state,
            session_id="a",
            satellite_id="sat-b",
            context_tokens=4,
            capacity_tokens=10,
            kv_bytes_per_token=1.0,
            now_s=2.0,
            copied_from="sat-a",
        )
        self.assertEqual(result["action"], "copied")
        self.assertIn("a", self.state.kv_cache_for("sat-a").entries)
        self.assertIn("a", self.state.kv_cache_for("sat-b").entries)
        self.assertEqual(self.state.session_for("a").satellite_ids, ["sat-a", "sat-b"])

    def test_recompute_on_new_satellite_keeps_old_copy(self):
        commit_session_kv(
            self.state,
            session_id="a",
            satellite_id="sat-a",
            context_tokens=4,
            capacity_tokens=10,
            kv_bytes_per_token=1.0,
            now_s=1.0,
        )
        result = commit_session_kv(
            self.state,
            session_id="a",
            satellite_id="sat-b",
            context_tokens=4,
            capacity_tokens=10,
            kv_bytes_per_token=1.0,
            now_s=2.0,
        )
        self.assertEqual(result["action"], "recompute_context")
        self.assertIn("a", self.state.kv_cache_for("sat-a").entries)
        self.assertIn("a", self.state.kv_cache_for("sat-b").entries)

    def test_lru_eviction_removes_only_the_evicted_satellite(self):
        for session_id, now_s in (("a", 1.0), ("b", 2.0), ("c", 3.0)):
            commit_session_kv(
                self.state,
                session_id=session_id,
                satellite_id="sat-a",
                context_tokens=4,
                capacity_tokens=8,
                kv_bytes_per_token=1.0,
                now_s=now_s,
            )
        self.assertEqual(self.state.session_for("a").satellite_ids, [])
        self.assertEqual(self.state.session_for("b").satellite_ids, ["sat-a"])
        self.assertEqual(self.state.session_for("c").satellite_ids, ["sat-a"])


if __name__ == "__main__":
    unittest.main()
