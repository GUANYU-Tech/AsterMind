import json
import unittest
from orbitserve.architecture import resolve_architecture
from orbitserve.experiments.constellation_cost import estimate_constellation_cost
from tests.system_case import physical_case


class ConstellationCostTests(unittest.TestCase):
    def test_cost_scales_architecture_per_member_profile(self):
        spec = physical_case()
        hardware = resolve_architecture(spec.hardware)
        result = estimate_constellation_cost(
            {"compute_node_count": 2},
            spec.costing,
            resolved_architecture=hardware,
            formation_member_count=4,
        )
        self.assertEqual(result.physical_satellite_count, 8)
        self.assertAlmostEqual(
            result.total_cost_usd, 8 * hardware.cost_profile.total_cost_usd
        )
        json.dumps(result.to_dict())

    def test_orbit_metadata_does_not_rewrite_immutable_hardware(self):
        spec = physical_case()
        hardware = resolve_architecture(spec.hardware)
        a = estimate_constellation_cost(
            {"compute_node_count": 2, "altitude_km": 500},
            spec.costing,
            resolved_architecture=hardware,
            formation_member_count=4,
        )
        b = estimate_constellation_cost(
            {"compute_node_count": 2, "altitude_km": 900},
            spec.costing,
            resolved_architecture=hardware,
            formation_member_count=4,
        )
        self.assertEqual(a.single_satellite, b.single_satellite)
