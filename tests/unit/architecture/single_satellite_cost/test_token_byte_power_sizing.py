from __future__ import annotations

import json
import math
import unittest
from dataclasses import replace
from pathlib import Path

from orbitserve.architecture.single_satellite_cost import estimate_single_satellite_cost
from orbitserve.architecture.single_satellite_cost.models import load_case


ROOT = Path(__file__).resolve().parents[4]
ORBIT_CONDITION = {"eclipse_duration_min": 35.0, "sunlit_fraction": 0.64}


class StaticPowerSizingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.requirements, cls.config = load_case(
            json.loads((ROOT / "examples/single_satellite/edge_orin_small_sat.json").read_text())
        )

    def test_static_power_uses_conservative_peak_load(self) -> None:
        result = estimate_single_satellite_cost(
            self.requirements,
            self.config,
            orbit_energy_condition=ORBIT_CONDITION,
        )
        self.assertEqual(result.P_avg_w, result.P_peak_w)
        self.assertEqual(result.P_shadow_avg_w, result.P_peak_w)
        self.assertEqual(result.provenance["power_sizing_mode"], "conservative_peak_load")

    def test_battery_uses_orbit_condition_and_static_dod(self) -> None:
        result = estimate_single_satellite_cost(
            self.requirements,
            self.config,
            orbit_energy_condition=ORBIT_CONDITION,
        )
        denominator = (
            self.config.power.dod_max
            * self.config.power.battery_eol_fraction
            * self.config.power.battery_discharge_efficiency
        )
        expected = result.P_shadow_avg_w * ORBIT_CONDITION["eclipse_duration_min"] / 60.0 / denominator
        self.assertTrue(
            math.isclose(
                result.E_battery_req_wh,
                max(expected, self.config.power.minimum_battery_capacity_wh),
            )
        )

    def test_cost_api_has_no_workload_derived_energy_argument(self) -> None:
        with self.assertRaises(TypeError):
            estimate_single_satellite_cost(
                self.requirements,
                self.config,
                planned_energy_demand=object(),
            )

    def test_explicit_static_component_counts_are_authoritative(self) -> None:
        changed = replace(
            self.config,
            compute=replace(self.config.compute, accelerator_count=4),
            storage=replace(self.config.storage, device_count=2),
            interconnect=replace(self.config.interconnect, switch_count=3),
        )
        result = estimate_single_satellite_cost(
            self.requirements,
            changed,
            orbit_energy_condition=ORBIT_CONDITION,
        )
        self.assertEqual(result.provenance["N_acc"], 4)
        self.assertEqual(result.provenance["N_storage"], 2)
        self.assertEqual(result.provenance["N_switch"], 3)


if __name__ == "__main__":
    unittest.main()
