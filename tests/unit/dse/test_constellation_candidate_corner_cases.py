import unittest
from orbitserve.dse import DSESpec, physical_design_candidates
from tests.system_case import physical_case


class ConstellationCandidateCornerCaseTests(unittest.TestCase):
    def test_unknown_and_duplicate_candidates_fail_closed(self):
        with self.assertRaises(ValueError):
            DSESpec.from_dict({"unknown_search_space": {}})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            physical_design_candidates(
                physical_case().physical_network,
                {
                    "design_candidates": [
                        {"candidate_id": "same"},
                        {"candidate_id": "same"},
                    ]
                },
            )

    def test_empty_sweep_and_unknown_field_fail(self):
        for sweep in [{"time.step_s": []}, {"constellations.unknown": [1]}]:
            with self.subTest(sweep=sweep), self.assertRaises((ValueError, KeyError)):
                physical_design_candidates(
                    physical_case().physical_network, {"sweep": sweep}
                )
