import unittest

from orbitserve.runtime import RuntimeStrategySpec
from orbitserve.runtime.engine.decision import build_candidate_selector


def _candidate(candidate_id, satellite_id):
    return {
        "candidate_id": candidate_id,
        "action_type": "local_compute",
        "feasible": True,
        "route_plan": {},
        "predicted_ttft_s": 1.0,
        "predicted_e2e_s": 2.0,
        "predicted_queue_wait_s": 0.0,
        "predicted_link_wait_s": 0.0,
        "predicted_energy_risk": 0.0,
        "predicted_soc_after": 0.8,
        "hop_count": 0,
        "provenance": {
            "compute_satellite_id": satellite_id,
            "access_satellite_id": satellite_id,
            "path_id": satellite_id,
        },

    }
class RuntimeDVFSStrategyTests(unittest.TestCase):
    def test_strategy_owns_shadow_dvfs_configuration(self):
        strategy = RuntimeStrategySpec.from_dict(
            {
                "policy": "orbitserve",
                "shadow_dvfs_enabled": True,
                "shadow_dvfs_scale": 0.25,
            }
        )

        self.assertTrue(strategy.shadow_dvfs_enabled)
        self.assertEqual(strategy.shadow_dvfs_scale, 0.25)

    def test_shadow_dvfs_scale_must_not_exceed_nominal_capacity(self):
        with self.assertRaises(ValueError):
            RuntimeStrategySpec.from_dict({"shadow_dvfs_scale": 1.1})


    def test_fixed_strategy_selects_equal_candidates_deterministically(self):
        scheduler = build_candidate_selector(RuntimeStrategySpec.from_dict({"policy": "orbitserve"}))
        state = {"candidates": (_candidate("candidate-b", "sat-b"), _candidate("candidate-a", "sat-a"))}
        first = scheduler.select_action(state, {"request_id": "request-0"})
        second = scheduler.select_action(state, {"request_id": "request-0"})

        self.assertEqual(first.selected_candidate_id, "candidate-a")
        self.assertEqual(second.selected_candidate_id, first.selected_candidate_id)
        self.assertFalse(first.optimization_support["strategy_mutable_during_run"])

if __name__ == "__main__":
    unittest.main()
