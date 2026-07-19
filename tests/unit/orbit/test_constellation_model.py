import math
import unittest

from orbitserve.orbit.constellation import (
    EARTH_RADIUS_KM,
    ConstellationDesign,
    coverage_central_angle_deg,
    line_of_sight_ok,
)


def clustered_design(**cluster_overrides):
    cluster = {
        "grid_shape": "2x2",
        "satellites_per_cluster": 4,
        "formation_pitch_m": 500.0,
        "d_keepout_m": 10.0,
        **cluster_overrides,
    }
    return ConstellationDesign.from_dict({
        "name": "geometry_contract",
        "orbital_sets": [{
            "name": "walker",
            "altitude_km": 550,
            "inclination_deg": 53,
            "planes": 2,
            "clusters_per_plane": 4,
            "cluster": cluster,
        }],
    })


class ConstellationGeometryContractTests(unittest.TestCase):
    def test_cluster_contract_contains_geometry_without_link_topology(self):
        design = clustered_design()
        analysis = design.architecture_analysis()
        cluster = analysis["orbital_sets"][0]["cluster"]
        self.assertEqual(design.satellite_count, 32)
        self.assertEqual(cluster["grid_shape"], [2, 2])
        self.assertAlmostEqual(cluster["computed_cluster_radius_m"], math.sqrt(2) * 250.0)
        self.assertNotIn("topology", cluster)
        self.assertNotIn("short_isl_profile", cluster)
        self.assertNotIn("aggregate_topology", analysis)

    def test_orbit_rejects_link_topology_fields(self):
        for field, value in {
            "topology": "mesh",
            "short_isl_profile": {"bandwidth_gbps": 100},
            "same_plane_long_isl_policy": {"mode": "ring"},
            "cross_plane_isl_policy": {"mode": "nearest"},
        }.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "no longer accepts link-topology"):
                    clustered_design(**{field: value})

    def test_energy_design_condition_is_derived_from_orbit(self):
        condition = clustered_design().energy_design_condition()
        self.assertGreater(condition["eclipse_duration_min"], 0.0)
        self.assertGreater(condition["sunlit_fraction"], 0.0)
        self.assertLessEqual(condition["sunlit_fraction"], 1.0)

    def test_geometry_helpers_return_reasonable_values(self):
        theta = coverage_central_angle_deg(550, 25)
        self.assertGreater(theta, 8.0)
        self.assertLess(theta, 9.0)
        radius = EARTH_RADIUS_KM + 550
        near = math.radians(30)
        far = math.radians(60)
        r1 = (radius, 0.0, 0.0)
        self.assertTrue(line_of_sight_ok(r1, (radius * math.cos(near), radius * math.sin(near), 0.0)))
        self.assertFalse(line_of_sight_ok(r1, (radius * math.cos(far), radius * math.sin(far), 0.0)))

    def test_same_plane_and_shadow_geometry_are_reported(self):
        orbital_set = clustered_design().architecture_analysis()["orbital_sets"][0]
        self.assertIsNotNone(orbital_set["same_plane"]["adjacent_centroid_distance_km"])
        self.assertGreater(orbital_set["shadow"]["eclipse_duration_beta0_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
