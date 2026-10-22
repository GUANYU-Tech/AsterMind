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

    def test_builder_preserves_physical_network(self):
        from orbitserve.runtime import NetworkSpec, NetworkNodeSpec
        network = NetworkSpec((NetworkNodeSpec("sat-a","compute"),), {}, communication_topology="explicit")
        scenario = build_simulation_spec(simulation_config={"network":network.to_dict()})
        self.assertEqual(scenario.network, network)
        self.assertNotIn("communication_ingress", scenario.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
