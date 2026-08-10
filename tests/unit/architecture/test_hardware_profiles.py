from __future__ import annotations

import unittest

from orbitserve.architecture import (
    HardwareSpec,
    ResolvedArchitecture,
    SatelliteCostProfile,
    SatelliteEnergyProfile,
    SatelliteHardwareProfile,
    resolve_architecture,
)


def hardware_spec(**overrides):
    data = {
        "name": "test_satellite",
        "accelerator": "h200_sxm",
        "compute_payload": {
            "accelerator_count": 2,
            "purchase_cost_per_unit_usd": 30000,
        },
        "storage": {"profile": "nvme_8tb", "device_count": 2},
        "intra_interconnect": {"switch_count": 1, "switch_bandwidth_gbs": 900},
        "communication": {
            "ground_link_bandwidth_gbps": 2,
            "isl_terminal_count": 4,
            "isl_terminal_bandwidth_gbps": 20,
        },
        "power": {
            "battery_capacity_wh": 2000,
            "solar_array_power_w": 1500,
            "initial_soc": 0.8,
            "maximum_depth_of_discharge": 0.3,
            "compute_energy_j_per_token": 0.25,
            "link_energy_j_per_byte": 2e-6,
        },
    }
    data.update(overrides)
    return HardwareSpec.from_dict(data)


class HardwareProfileTests(unittest.TestCase):
    def test_one_spec_resolves_three_typed_profiles(self) -> None:
        resolved = resolve_architecture(hardware_spec())
        self.assertIsInstance(resolved, ResolvedArchitecture)
        self.assertIsInstance(resolved.hardware_profile, SatelliteHardwareProfile)
        self.assertIsInstance(resolved.energy_profile, SatelliteEnergyProfile)
        self.assertIsInstance(resolved.cost_profile, SatelliteCostProfile)

    def test_hardware_profile_is_the_serving_facing_view(self) -> None:
        profile = resolve_architecture(hardware_spec()).hardware_profile
        self.assertEqual(profile.accelerator_count, 2)
        self.assertEqual(profile.total_fast_mem_gb, 282)
        self.assertEqual(profile.storage_capacity_tb, 16)
        self.assertEqual(profile.laser_terminal_count, 4)
        self.assertEqual(profile.inter_satellite_bandwidth_gbps, 80)

    def test_hardware_spec_rejects_removed_configuration_layers(self) -> None:
        for removed in ("mode", "source_case_path", "compute", "interconnect"):
            with self.subTest(removed=removed), self.assertRaises(ValueError):
                HardwareSpec.from_dict({removed: {}})

        with self.assertRaises(ValueError):
            HardwareSpec.from_dict({"power": {"dod_max": 0.3}})

    def test_resolution_has_no_external_design_argument(self) -> None:
        with self.assertRaises(TypeError):
            resolve_architecture(hardware_spec(), constellation=object())


if __name__ == "__main__":
    unittest.main()
