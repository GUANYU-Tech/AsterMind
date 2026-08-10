import unittest

from orbitserve.runtime.state.kv_cache import (
    _admit_session_kv,
    commit_session_kv,
    kv_locations_for_session,
    local_kv_reuse_tokens,
    touch_kv_entry,
)
from orbitserve.runtime.state.runtime_state import RuntimeState


class KVCapacityLRUTests(unittest.TestCase):
    def setUp(self):
        self.state = RuntimeState()

    def admit(self, session_id, tokens, now_s, satellite_id="sat-a", capacity=10):
        self.state.session_for(session_id)
        self.state.kv_cache_for(satellite_id, capacity_tokens=capacity)
        return _admit_session_kv(
            self.state,
            session_id=session_id,
            satellite_id=satellite_id,
            cached_tokens=tokens,
            now_s=now_s,
        )

    def test_capacity_pressure_evicts_lru_without_time_expiry(self):
        self.admit("a", 4, 1.0)
        self.admit("b", 4, 10_000.0)
        result = self.admit("c", 4, 20_000.0)
        self.assertEqual(result["evicted_session_ids"], ("a",))
        self.assertEqual(result["evicted_tokens"], 4)
        self.assertEqual(set(self.state.kv_cache_for("sat-a").entries), {"b", "c"})
        self.assertEqual(self.state.kv_cache_for("sat-a").used_tokens, 8)

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
        self.state.kv_cache_for("sat-a", capacity_tokens=10)
        self.state.kv_cache_for("sat-b", capacity_tokens=10)
        commit_session_kv(
            self.state,
            session_id="a",
            satellite_id="sat-a",
            context_tokens=4,
            now_s=1.0,
        )
        result = commit_session_kv(
            self.state,
            session_id="a",
            satellite_id="sat-b",
            context_tokens=4,
            now_s=2.0,
            copied_from="sat-a",
        )
        self.assertEqual(result["action"], "copied")
        self.assertEqual(kv_locations_for_session(self.state, "a"), ("sat-a", "sat-b"))

    def test_recompute_on_new_satellite_keeps_old_copy(self):
        self.state.kv_cache_for("sat-a", capacity_tokens=10)
        self.state.kv_cache_for("sat-b", capacity_tokens=10)
        commit_session_kv(
            self.state,
            session_id="a",
            satellite_id="sat-a",
            context_tokens=4,
            now_s=1.0,
        )
        result = commit_session_kv(
            self.state,
            session_id="a",
            satellite_id="sat-b",
            context_tokens=4,
            now_s=2.0,
        )
        self.assertEqual(result["action"], "recompute_context")
        self.assertEqual(kv_locations_for_session(self.state, "a"), ("sat-a", "sat-b"))

    def test_lru_eviction_is_local_to_one_satellite(self):
        self.state.kv_cache_for("sat-a", capacity_tokens=8)
        self.state.kv_cache_for("sat-b", capacity_tokens=8)
        self.admit("a", 4, 1.0, "sat-a", 8)
        self.admit("a", 4, 1.0, "sat-b", 8)
        self.admit("b", 4, 2.0, "sat-a", 8)
        self.admit("c", 4, 3.0, "sat-a", 8)
        self.assertEqual(kv_locations_for_session(self.state, "a"), ("sat-b",))
        self.assertEqual(set(self.state.kv_cache_for("sat-a").entries), {"b", "c"})

    def test_copy_read_refreshes_source_lru(self):
        self.admit("a", 4, 1.0)
        touch_kv_entry(
            self.state,
            satellite_id="sat-a",
            session_id="a",
            now_s=9.0,
        )
        self.assertEqual(
            self.state.kv_cache_for("sat-a").entries["a"].last_access_s,
            9.0,
        )

    def test_entry_larger_than_capacity_does_not_replace_existing_copy(self):
        self.admit("a", 4, 1.0)
        result = commit_session_kv(
            self.state,
            session_id="a",
            satellite_id="sat-a",
            context_tokens=11,
            now_s=2.0,
        )
        self.assertFalse(result["admitted"])
        self.assertEqual(self.state.session_for("a").context_tokens, 11)
        self.assertEqual(
            self.state.kv_cache_for("sat-a").entries["a"].cached_tokens,
            4,
        )


if __name__ == "__main__":
    unittest.main()
