import unittest

from orbitserve.experiments.simulation_builder import build_simulation_spec


class SimulationBuilderTests(unittest.TestCase):
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
            "initial_soc": 0.8,
            "provenance": {"source": "architecture"},
            "warnings": [],
        }

        scenario = build_simulation_spec(energy_profile=energy_profile)

        self.assertEqual(scenario.energy_profile, energy_profile)
        self.assertEqual(
            scenario.provenance["energy_profile"],
            {"source": "architecture"},
        )

    def test_builder_does_not_derive_energy_from_cost(self):
        scenario = build_simulation_spec()

        self.assertEqual(scenario.energy_profile, {})

    def test_builder_exposes_orbit_data_as_environment_state(self):
        marker = object()
        scenario = build_simulation_spec(simulation_result=marker)

        self.assertIsNone(scenario.environment_state)
        self.assertNotIn("orbit_trace_bundle", scenario.__dataclass_fields__)

    def test_builder_extracts_communication_ingress_without_putting_it_in_strategy(self):
        scenario = build_simulation_spec(
            simulation_config={
                "communication_satellite": {
                    "altitude_km": 550.0,
                    "uplink_bandwidth_bps": 1e9,
                    "downlink_bandwidth_bps": 2e9,
                }
            }
        )

        self.assertTrue(scenario.communication_ingress.configured)
        self.assertEqual(scenario.communication_ingress.altitude_km, 550.0)
        self.assertNotIn("altitude_km", scenario.strategy.to_dict())


if __name__ == "__main__":
    unittest.main()
