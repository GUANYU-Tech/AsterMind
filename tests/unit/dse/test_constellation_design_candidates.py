import unittest
from copy import deepcopy
from orbitserve.dse import DSESpec,physical_design_candidates
from tests.system_case import physical_case

class ConstellationDesignCandidateDSETests(unittest.TestCase):
    def test_candidates_are_isolated_and_keep_hardware_outside_orbit_inputs(self):
        base=physical_case().physical_network
        before=deepcopy(base)
        candidates=physical_design_candidates(base,DSESpec.from_dict({"design_candidates":[
            {"candidate_id":"one","patches":{"constellations.compute_formations.0.source.altitude_km":750}},
            {"candidate_id":"two","patches":{"constellations.compute_formations.0.source.altitude_km":850}}]}))
        self.assertEqual(base,before)
        self.assertEqual(len(candidates),2)
        self.assertNotEqual(candidates[0]["design_id"],candidates[1]["design_id"])
        self.assertNotIn("hardware",candidates[0]["physical_network"])

    def test_fixed_physical_population_constraint_is_enforced(self):
        with self.assertRaisesRegex(ValueError,"same_physical_satellite_count"):
            physical_design_candidates(physical_case().physical_network,{"constraints":{"same_physical_satellite_count":True},
                "sweep":{"constellations.communication_shells.0.source.satellites_per_plane":[8]}})
