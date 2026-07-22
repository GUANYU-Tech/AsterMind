import unittest

from orbitserve.runtime.setup.builder import build_runtime_simulation_spec


class RuntimeEnergyInputTests(unittest.TestCase):
    def test_builder_preserves_architecture_energy_profile(self):
        energy_profile = {
            "evaluated": True,
            "battery_capacity_j": 1000.0,
            "charge_rate_w": 40.0,
            "idle_power_w": 10.0,
            "battery_charge_efficiency": 0.9,
            "battery_discharge_efficiency": 0.9,
            "compute_energy_j_per_token": 0.5,
            "link_energy_j_per_byte": 1e-6,
            "soc_min": 0.2,
            "provenance": {"source": "architecture"},
            "warnings": [],
        }

        scenario = build_runtime_simulation_spec(energy_profile=energy_profile)

        self.assertEqual(scenario.energy_profile, energy_profile)
        self.assertEqual(
            scenario.provenance["energy_profile"],
            {"source": "architecture"},
        )

    def test_builder_does_not_derive_energy_from_cost(self):
        scenario = build_runtime_simulation_spec()

        self.assertEqual(scenario.energy_profile, {})


if __name__ == "__main__":
    unittest.main()
