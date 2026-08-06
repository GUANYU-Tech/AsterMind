from __future__ import annotations

import unittest
from dataclasses import replace

from orbitserve.architecture import HardwareSpec, SatelliteDesignLimits
from orbitserve.architecture.single_satellite_cost import estimate_single_satellite_cost


def base_spec() -> HardwareSpec:
    return HardwareSpec.from_dict(
        {
            "accelerator": "h200_sxm",
            "compute_payload": {
                "accelerator_count": 2,
                "purchase_cost_per_unit_usd": 50000,
            },
            "communication": {
                "ground_link_bandwidth_gbps": 2,
                "ground_link_mass_kg": 5,
                "ground_link_power_w": 20,
                "isl_terminal_count": 2,
                "isl_terminal_bandwidth_gbps": 10,
                "isl_terminal_mass_kg": 10,
                "isl_terminal_power_w": 100,
            },
            "power": {
                "battery_capacity_wh": 1000,
                "solar_array_power_w": 1200,
            },
        }
    )


class CostModelTests(unittest.TestCase):
    def test_breakdowns_sum_to_static_totals(self) -> None:
        result = estimate_single_satellite_cost(base_spec())
        self.assertAlmostEqual(sum(result.cost_breakdown_usd.values()), result.total_cost_usd)
        self.assertAlmostEqual(sum(result.mass_breakdown_kg.values()), result.total_mass_kg)
        self.assertEqual(result.battery_capacity_wh, 1000)
        self.assertEqual(result.solar_array_power_w, 1200)

    def test_cost_api_rejects_external_orbit_conditions(self) -> None:
        with self.assertRaises(TypeError):
            estimate_single_satellite_cost(base_spec(), orbit_energy_condition={})

    def test_configured_battery_and_solar_sizes_drive_cost(self) -> None:
        base = base_spec()
        larger = replace(
            base,
            power=replace(base.power, battery_capacity_wh=2000, solar_array_power_w=2400),
        )
        first = estimate_single_satellite_cost(base)
        second = estimate_single_satellite_cost(larger)
        self.assertGreater(second.mass_breakdown_kg["power_system"], first.mass_breakdown_kg["power_system"])
        self.assertGreater(second.total_cost_usd, first.total_cost_usd)

    def test_static_limits_mark_infeasible_designs(self) -> None:
        spec = replace(
            base_spec(),
            design_limits=SatelliteDesignLimits(payload_power_limit_w=1, satellite_mass_limit_kg=1),
        )
        result = estimate_single_satellite_cost(spec)
        self.assertFalse(result.feasible)
        self.assertTrue(any("payload peak power" in item for item in result.violations))
        self.assertTrue(any("satellite mass" in item for item in result.violations))

    def test_missing_accelerator_purchase_cost_is_infeasible(self) -> None:
        result = estimate_single_satellite_cost(HardwareSpec())
        self.assertFalse(result.feasible)
        self.assertTrue(any("purchase_cost_per_unit_usd" in item for item in result.violations))


if __name__ == "__main__":
    unittest.main()
