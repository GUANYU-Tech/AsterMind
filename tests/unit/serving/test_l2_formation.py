import unittest

from orbitserve.serving import L2FormationSpec, ServingSpec


class L2FormationTests(unittest.TestCase):
    def test_single_member_default_is_valid(self):
        spec = ServingSpec()
        self.assertEqual(spec.formation.member_count, 1)
        self.assertEqual(spec.parallelism.parallel_node_count, 1)

    def test_topology_contract_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "member_count"):
            L2FormationSpec(member_count=0)
        with self.assertRaisesRegex(ValueError, "topology"):
            L2FormationSpec(topology="mesh")
        with self.assertRaisesRegex(ValueError, "link_bandwidth_gbps"):
            L2FormationSpec(member_count=2, link_bandwidth_gbps=0.0)

    def test_parallelism_satellite_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "satellite_count"):
            ServingSpec.from_dict({"parallelism": {"satellite_count": 1}})


if __name__ == "__main__":
    unittest.main()
