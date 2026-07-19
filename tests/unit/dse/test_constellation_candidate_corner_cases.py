import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.dse.search import DSESpec, run_dse
from orbitserve.experiments import load_experiment_spec


ROOT = Path(__file__).resolve().parents[3]


def _hybrid_multi_set_candidate(spec):
    payload = copy.deepcopy(spec.constellation.to_dict())
    primary_set = payload["orbital_sets"][0]
    extra_set = copy.deepcopy(primary_set)
    primary_set["name"] = "hybrid_primary_walker"
    extra_set.update(
        {
            "name": "hybrid_dawn_dusk_extra",
            "orbit_type": "dawn_dusk_sso",
            "orbital_set_type": "walker_star",
            "walker_type": "star",
            "inclination_deg": 97.6,
            "planes": 1,
            "clusters_per_plane": 1,
            "raan_span_deg": 180,
            "ltan": "06:00/18:00",
        }
    )
    payload["name"] = "hybrid_multi_set_mismatched_count"
    payload["orbital_sets"] = [primary_set, extra_set]
    return {
        "candidate_id": "hybrid_multi_set_mismatched_count",
        "constellation_family": "hybrid",
        "constellation": payload,
    }


class ConstellationCandidateCornerCaseTests(unittest.TestCase):
    def test_from_dict_reads_nested_constellation_candidates_alias(self):
        dse = DSESpec.from_dict(
            {
                "enabled": True,
                "constellation_search_space": {
                    "constellation_candidates": [
                        {
                            "name": "alias_candidate",
                            "constellation": {"name": "alias_payload", "orbital_sets": []},
                        }
                    ]
                },
            }
        )

        self.assertTrue(dse.enabled)
        self.assertEqual(len(dse.design_candidates), 1)
        self.assertEqual(dse.design_candidates[0]["candidate_id"], "alias_candidate")
        self.assertEqual(dse.design_candidates[0]["constellation"]["name"], "alias_payload")

    def test_fixed_satellite_count_rejects_hybrid_multi_orbital_set_candidate(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_clustered_constellation.json")
        dse = DSESpec.from_dict(
            {
                "enabled": True,
                "constraints": {"same_satellite_count_as_base": True},
                "constellation_candidates": [_hybrid_multi_set_candidate(spec)],
            }
        )

        with self.assertRaisesRegex(ValueError, "violates fixed satellite_count"):
            run_dse(replace(spec, dse=dse))


if __name__ == "__main__":
    unittest.main()
