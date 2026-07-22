import unittest

from orbitserve.runtime import RuntimeStrategySpec


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


if __name__ == "__main__":
    unittest.main()
