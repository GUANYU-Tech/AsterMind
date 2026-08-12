import unittest
from types import SimpleNamespace

from orbitserve.runtime.engine.simulator import _preview_kv_copy
from orbitserve.runtime.state.kv_cache import _admit_session_kv
from orbitserve.runtime.state.runtime_state import RuntimeState


def _window(source, target, propagation_delay_s):
    return {
        "link_id": f"isl:{source}:{target}",
        "endpoint_a": source,
        "endpoint_b": target,
        "start_s": 0.0,
        "end_s": 100.0,
        "distance_km": propagation_delay_s * 299_792.458,
        "propagation_delay_s": propagation_delay_s,
    }


class KVCopySelectionTests(unittest.TestCase):
    def setUp(self):
        self.state = RuntimeState()
        for satellite_id in ("sat-a", "sat-b", "sat-target"):
            self.state.kv_cache_for(satellite_id, capacity_tokens=1000)

    def store(self, satellite_id, tokens):
        _admit_session_kv(
            self.state,
            session_id="session-0",
            satellite_id=satellite_id,
            cached_tokens=tokens,
            now_s=0.0,
        )

    def preview(self, environment_state, *, local_hit_tokens=0):
        return _preview_kv_copy(
            state=self.state,
            session_id="session-0",
            target_satellite_id="sat-target",
            required_context_tokens=100,
            prompt_length=100,
            local_hit_tokens=local_hit_tokens,
            copy_enabled=True,
            kv_bytes_per_token=2.0,
            environment_state=environment_state,
            start_s=1.0,
            isl_bandwidth_bps=1e9,
            prefill_time_table=(
                {"prompt_length": 1, "prefill_tokens_per_s": 10.0},
                {"prompt_length": 100, "prefill_tokens_per_s": 10.0},
            ),
        )

    def test_source_with_shortest_copy_plus_recompute_time_is_selected(self):
        self.store("sat-a", 90)
        self.store("sat-b", 50)
        environment_state = SimpleNamespace(
            isl_windows=(
                _window("sat-a", "sat-target", 0.10),
                _window("sat-b", "sat-target", 0.01),
            )
        )

        plan = self.preview(environment_state)

        self.assertEqual(plan["source_satellite_id"], "sat-a")
        self.assertEqual(plan["copied_tokens"], 90)

    def test_nearer_smaller_source_can_win(self):
        self.store("sat-a", 90)
        self.store("sat-b", 50)
        environment_state = SimpleNamespace(
            isl_windows=(
                _window("sat-a", "sat-target", 10.0),
                _window("sat-b", "sat-target", 0.01),
            )
        )

        plan = self.preview(environment_state)

        self.assertEqual(plan["source_satellite_id"], "sat-b")
        self.assertEqual(plan["copied_tokens"], 50)

    def test_existing_target_cache_reduces_copy_to_the_missing_prefix(self):
        self.store("sat-a", 40)
        self.store("sat-target", 20)
        environment_state = SimpleNamespace(
            isl_windows=(_window("sat-a", "sat-target", 0.01),)
        )

        plan = self.preview(environment_state, local_hit_tokens=20)

        self.assertEqual(plan["copied_tokens"], 20)

    def test_no_copy_when_migration_is_slower_than_local_recompute(self):
        self.store("sat-a", 90)
        environment_state = SimpleNamespace(
            isl_windows=(_window("sat-a", "sat-target", 20.0),)
        )
        plan = self.preview(environment_state)
        self.assertIsNone(plan["source_satellite_id"])
        self.assertEqual(plan["copied_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
