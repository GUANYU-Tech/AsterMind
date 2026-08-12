import unittest
from types import SimpleNamespace

from orbitserve.architecture import CommunicationSpec, HardwareSpec, resolve_architecture
from orbitserve.experiments.constellation_cost import CostingSpec, estimate_constellation_cost


class L2CostAggregationTests(unittest.TestCase):
    def test_cost_scales_single_satellite_by_nodes_and_members_once(self):
        hardware = HardwareSpec(
            communication=CommunicationSpec(
                isl_terminal_count=2,
                isl_terminal_bandwidth_gbps=10.0,
                isl_terminal_mass_kg=3.0,
                isl_terminal_power_w=4.0,
            )
        )
        resolved = resolve_architecture(hardware)

        result = estimate_constellation_cost(
            SimpleNamespace(compute_node_count=3),
            CostingSpec(),
            resolved_architecture=resolved,
            formation_member_count=4,
        )

        self.assertEqual(result.compute_node_count, 3)
        self.assertEqual(result.formation_member_count, 4)
        self.assertEqual(result.physical_satellite_count, 12)
        self.assertEqual(
            result.total_cost_usd,
            resolved.cost_profile.total_cost_usd * 12,
        )
        self.assertTrue(
            result.provenance["single_satellite_includes_configured_laser_terminals"]
        )
        self.assertFalse(result.provenance["topology_specific_cost_added"])

    def test_missing_hardware_keeps_cost_unevaluated(self):
        result = estimate_constellation_cost(
            {"compute_node_count": 2},
            CostingSpec(),
            formation_member_count=3,
        )
        self.assertFalse(result.evaluated)
        self.assertEqual(result.physical_satellite_count, 6)


if __name__ == "__main__":
    unittest.main()
