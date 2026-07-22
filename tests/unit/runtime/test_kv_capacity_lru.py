import unittest

from orbitserve.runtime.engine.simulator import _session_action
from orbitserve.runtime import ServingDeploymentView
from orbitserve.runtime.state.kv_cache import _admit_session_kv
from orbitserve.runtime.state.runtime_state import RuntimeState


def _deployment(capacity_tokens=10):
    return ServingDeploymentView(
        deployment_id="deployment-a",
        member_satellites=("sat-a",),
        kv_cache_capacity_tokens=capacity_tokens,
        kv_cache_bytes_per_token=1.0,
    )


class KVCapacityLRUTests(unittest.TestCase):
    def setUp(self):
        self.state = RuntimeState()
        self.deployment = _deployment()
        self.strategy = type("Strategy", (), {"kv_policy": {}, "hard_gate_overrides": {}})()

    def admit(self, session_id, tokens, now_s):
        self.state.session_for(session_id)
        return _admit_session_kv(
            self.state,
            self.deployment,
            self.strategy,
            session_id=session_id,
            satellite_id="sat-a",
            tokens=tokens,
            kv_bytes_per_token=1.0,
            now_s=now_s,
        )

    def test_capacity_sufficient_keeps_all_entries_without_time_expiry(self):
        self.assertTrue(self.admit("a", 4, 1.0)["admitted"])
        result = self.admit("b", 4, 10_000.0)

        self.assertTrue(result["admitted"])
        self.assertEqual(result["evicted_session_ids"], ())
        self.assertEqual(set(self.state.kv_cache_for("sat-a").entries), {"a", "b"})

    def test_capacity_pressure_evicts_least_recently_used_entry(self):
        self.admit("a", 4, 1.0)
        self.admit("b", 4, 2.0)

        result = self.admit("c", 4, 3.0)

        self.assertEqual(result["evicted_session_ids"], ("a",))
        self.assertEqual(set(self.state.kv_cache_for("sat-a").entries), {"b", "c"})

    def test_referenced_entry_is_not_evictable(self):
        self.admit("a", 4, 1.0)
        self.admit("b", 4, 2.0)
        self.state.kv_cache_for("sat-a").entries["a"].reference_count = 1

        result = self.admit("c", 6, 3.0)

        self.assertEqual(result["evicted_session_ids"], ("b",))
        self.assertEqual(set(self.state.kv_cache_for("sat-a").entries), {"a", "c"})

    def test_capacity_and_lru_are_scoped_to_each_satellite(self):
        self.admit("a", 8, 1.0)
        other = _admit_session_kv(
            self.state,
            self.deployment,
            self.strategy,
            session_id="b",
            satellite_id="sat-b",
            tokens=8,
            kv_bytes_per_token=1.0,
            now_s=2.0,
        )

        self.assertTrue(other["admitted"])
        self.assertEqual(set(self.state.kv_cache_for("sat-a").entries), {"a"})
        self.assertEqual(set(self.state.kv_cache_for("sat-b").entries), {"b"})

    def test_first_turn_recomputes_then_same_satellite_hits(self):
        deployment = _deployment(capacity_tokens=8)
        request = {"prompt_tokens": 3, "output_tokens": 1}
        first = _session_action(
            self.state,
            "a",
            "sat-a",
            1.0,
            deployment=deployment,
            runtime_strategy=self.strategy,
            request=request,
        )
        second = _session_action(
            self.state,
            "a",
            "sat-a",
            2.0,
            deployment=deployment,
            runtime_strategy=self.strategy,
            request=request,
        )

        self.assertEqual(first, "recompute_context")
        self.assertEqual(second, "sticky_kv_owner")

    def test_moving_session_invalidates_previous_satellite_entry(self):
        deployment = _deployment(capacity_tokens=8)
        request = {"prompt_tokens": 3, "output_tokens": 1}
        _session_action(
            self.state,
            "a",
            "sat-a",
            1.0,
            deployment=deployment,
            runtime_strategy=self.strategy,
            request=request,
        )
        moved = _session_action(
            self.state,
            "a",
            "sat-b",
            2.0,
            deployment=deployment,
            runtime_strategy=self.strategy,
            request=request,
        )

        self.assertEqual(moved, "recompute_context")
        self.assertNotIn("a", self.state.kv_cache_for("sat-a").entries)
        self.assertIn("a", self.state.kv_cache_for("sat-b").entries)

    def test_revisit_after_eviction_recomputes_context(self):
        deployment = _deployment(capacity_tokens=4)
        request = {"prompt_tokens": 3, "output_tokens": 1}

        first = _session_action(
            self.state,
            "a",
            "sat-a",
            1.0,
            deployment=deployment,
            runtime_strategy=self.strategy,
            request=request,
        )
        _session_action(
            self.state,
            "b",
            "sat-a",
            2.0,
            deployment=deployment,
            runtime_strategy=self.strategy,
            request=request,
        )
        revisited = _session_action(
            self.state,
            "a",
            "sat-a",
            3.0,
            deployment=deployment,
            runtime_strategy=self.strategy,
            request=request,
        )

        self.assertEqual(first, "recompute_context")
        self.assertEqual(revisited, "recompute_context")


if __name__ == "__main__":
    unittest.main()
