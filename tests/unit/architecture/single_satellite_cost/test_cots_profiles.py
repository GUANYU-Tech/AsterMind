import sys
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.architecture.single_satellite_cost import estimate_single_satellite_cost
from orbitserve.architecture.single_satellite_cost.models import (
    CommunicationConfig,
    ComputeConfig,
    CostConfig,
    SatelliteDesignLimits,
    PlatformConfig,
    PowerConfig,
    SingleSatelliteConfig,
    ThermalConfig,
)


def base_config():
    return SingleSatelliteConfig(
        platform=PlatformConfig(mass_kg=150, power_w=100),
        compute=ComputeConfig(profile_name="h200_sxm", purchase_cost_per_unit_usd=50000),
        communication=CommunicationConfig(ground_link_mass_kg=28, ground_link_power_w=100),
        power=PowerConfig(),
        thermal=ThermalConfig(),
        cost=CostConfig(),
    )


class CotsProfileTests(unittest.TestCase):
    def test_accelerator_count_is_explicit(self):
        configured = replace(
            base_config(),
            compute=replace(base_config().compute, accelerator_count=3),
        )
        result = estimate_single_satellite_cost(SatelliteDesignLimits(), configured)
        self.assertEqual(result.provenance["N_acc"], 3)
        self.assertTrue(result.feasible)


    def test_breakdowns_sum_to_totals(self):
        req = SatelliteDesignLimits()
        result = estimate_single_satellite_cost(req, base_config())
        self.assertAlmostEqual(sum(result.C_breakdown_usd.values()), result.C_sat_usd)
        self.assertAlmostEqual(sum(result.M_breakdown_kg.values()), result.M_sat_kg)


if __name__ == "__main__":
    unittest.main()
