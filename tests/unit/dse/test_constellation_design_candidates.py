import copy
import sys
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.dse.search import DSESpec, run_dse
from orbitserve.experiments import load_experiment_spec


ROOT = Path(__file__).resolve().parents[3]


def _base_constellation_payload() -> dict:
    spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_clustered_constellation.json")
    return spec.constellation.to_dict()


def _candidate(candidate_id: str, *, orbit_type: str = "circular", clusters_by_plane=None) -> dict:
    payload = _base_constellation_payload()
    payload["name"] = candidate_id
    orbital_set = payload["orbital_sets"][0]
    orbital_set["name"] = f"{candidate_id}_set"
    orbital_set["orbit_type"] = orbit_type
    if orbit_type == "dawn_dusk_sso":
        orbital_set["inclination_deg"] = 97.6
        orbital_set["orbital_set_type"] = "walker_star"
        orbital_set["walker_type"] = "star"
        orbital_set["raan_span_deg"] = 180
        orbital_set["ltan"] = "06:00/18:00"
    if clusters_by_plane is not None:
        orbital_set["custom_planes"] = [
            {"index": index, "clusters_per_plane": count, "raan_deg": index * 37.5}
            for index, count in enumerate(clusters_by_plane)
        ]
    return {
        "candidate_id": candidate_id,
        "orbit_family": orbit_type,
        "isl_strategy": "candidate_default",
        "constellation": payload,
    }


class ConstellationDesignCandidateDSETests(unittest.TestCase):
    def test_constellation_candidates_keep_satellite_count_and_expose_metadata(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_clustered_constellation.json")
        base_count = spec.constellation.satellite_count
        dse = DSESpec.from_dict(
            {
                "enabled": True,
                "constraints": {
                    "same_satellite_count_as_base": True,
                    "require_single_satellite_config_fixed": True,
                },
                "constellation_candidates": [
                    _candidate("dawn_dusk_ring", orbit_type="dawn_dusk_sso"),
                    _candidate("non_uniform_walker_like", clusters_by_plane=[3, 1]),
                ],
            }
        )

        result = run_dse(replace(spec, dse=dse)).to_dict()

        self.assertTrue(result["evaluated"])
        self.assertEqual(result["design_count"], 2)
        self.assertEqual({row["design"]["satellite_count"] for row in result["rows"]}, {base_count})
        candidate_ids = {row["design"]["dse_candidate"]["candidate_id"] for row in result["rows"]}
        self.assertEqual(candidate_ids, {"dawn_dusk_ring", "non_uniform_walker_like"})
        orbit_types = {
            row["design"]["architecture_analysis"]["orbital_sets"][0]["orbit"]["orbit_type"]
            for row in result["rows"]
        }
        self.assertIn("dawn_dusk_sso", orbit_types)
        for row in result["rows"]:
            self.assertTrue(
                any("single-satellite hardware is fixed" in warning for warning in row["warnings"])
            )


    def test_fixed_satellite_count_constraint_rejects_mismatched_candidate(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_clustered_constellation.json")
        mismatched = copy.deepcopy(_candidate("bad_count", clusters_by_plane=[3, 3]))
        dse = DSESpec.from_dict(
            {
                "enabled": True,
                "constraints": {"same_satellite_count_as_base": True},
                "design_candidates": [mismatched],
            }
        )

        with self.assertRaisesRegex(ValueError, "violates fixed satellite_count"):
            run_dse(replace(spec, dse=dse))


if __name__ == "__main__":
    unittest.main()
