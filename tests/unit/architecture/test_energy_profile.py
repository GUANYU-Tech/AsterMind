from __future__ import annotations

import unittest

from orbitserve.architecture import HardwareSpec, build_satellite_energy_profile


class EnergyProfileTests(unittest.TestCase):
    def test_energy_profile_is_derived_only_from_static_hardware(self) -> None:
        spec = HardwareSpec.from_dict(
            {
                "platform": {"idle_power_w": 40},
                "power": {
                    "battery_capacity_wh": 2000,
                    "solar_array_power_w": 1000,
                    "initial_soc": 0.85,
                    "maximum_depth_of_discharge": 0.25,
                    "battery_eol_fraction": 0.8,
                    "pv_eol_fraction": 0.9,
                    "power_management_efficiency": 0.8,
                    "compute_standby_power_w": 10,
                    "communication_standby_power_w": 5,
                    "compute_energy_j_per_token": 0.2,
                    "link_energy_j_per_byte": 3e-6,
                },
            }
        )
        profile = build_satellite_energy_profile(spec)
        self.assertEqual(profile.battery_capacity_j, 7_200_000)
        self.assertEqual(profile.soc_min, 0.75)
        self.assertEqual(profile.charge_rate_w, 720)
        self.assertEqual(profile.idle_power_w, 55)
        self.assertEqual(profile.compute_energy_j_per_token, 0.2)
        self.assertEqual(profile.link_energy_j_per_byte, 3e-6)


if __name__ == "__main__":
    unittest.main()
