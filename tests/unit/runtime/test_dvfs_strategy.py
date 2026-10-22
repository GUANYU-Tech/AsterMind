import unittest

from orbitserve.runtime import RuntimeStrategySpec
from orbitserve.runtime.engine.decision import SchedulerCandidate, build_candidate_selector


class RuntimeStrategyTests(unittest.TestCase):
    def test_contract_contains_only_resolved_strategy_parameters(self):
        strategy = RuntimeStrategySpec(
            ingress_selection="nearest_satellite",
            latency_weight=0.4,
            kv_reuse_weight=0.3,
            energy_weight=0.2,
            compute_load_weight=0.1,
            shadow_dvfs_enabled=True,
            shadow_dvfs_scale=0.5,
            kv_cache_copy_enabled=True,
        )
        self.assertAlmostEqual(sum(strategy.weights.values()), 1.0)
        self.assertNotIn("policy", strategy.to_dict())

    def test_weights_must_sum_to_one(self):
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            RuntimeStrategySpec(latency_weight=1.0)

    def test_kv_cache_copy_enabled_must_be_a_bool(self):
        with self.assertRaisesRegex(ValueError, "kv_cache_copy_enabled"):
            RuntimeStrategySpec(kv_cache_copy_enabled=1)

    def test_unknown_legacy_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            RuntimeStrategySpec.from_dict({"policy": "orbitserve"})

    def test_candidate_scores_are_weighted_and_ties_are_deterministic(self):
        strategy = RuntimeStrategySpec(
            latency_weight=1.0,
            kv_reuse_weight=0.0,
            energy_weight=0.0,
            compute_load_weight=0.0,
        )
        candidates = (
            SchedulerCandidate("b", "sat-b", True, {}, 0.8, 0.0, 0.5, 1.0, 0.2, 0.0),
            SchedulerCandidate("a", "sat-a", True, {}, 0.8, 0.0, 0.5, 1.0, 0.2, 0.0),
        )
        decision = build_candidate_selector(strategy).select_action(
            {"candidates": tuple(item.to_dict() for item in candidates)}
        )
        self.assertEqual(decision.selected_candidate_id, "a")
        self.assertEqual(decision.score_terms["score_model"], "normalized_compute_node_score_v1")


if __name__ == "__main__":
    unittest.main()
