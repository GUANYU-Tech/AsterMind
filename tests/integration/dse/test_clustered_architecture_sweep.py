import unittest
from dataclasses import replace
from pathlib import Path

from orbitserve.dse.search import DSESpec, run_dse
from orbitserve.experiments import load_experiment_spec


ROOT = Path(__file__).resolve().parents[3]


class ClusteredOrbitDSEIntegrationTests(unittest.TestCase):
    def test_cluster_geometry_sweep_updates_satellite_count_and_pitch(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_clustered_constellation.json")
        spec = replace(spec, dse=DSESpec(enabled=True, sweep={
            "formation_pitch_m": (500, 1000),
            "satellites_per_cluster": (4, 16),
        }))
        payload = run_dse(spec).to_dict()
        self.assertTrue(payload["evaluated"])
        self.assertEqual(payload["design_count"], 4)
        rows = payload["rows"]
        self.assertEqual(
            {row["design"]["satellite_count"] for row in rows},
            {spec.constellation.compute_cell_count * 4, spec.constellation.compute_cell_count * 16},
        )
        self.assertEqual({
            row["design"]["architecture_analysis"]["orbital_sets"][0]["cluster"]["pitch_x_m"]
            for row in rows
        }, {500, 1000})
        for row in rows:
            cluster = row["design"]["architecture_analysis"]["orbital_sets"][0]["cluster"]
            self.assertNotIn("topology", cluster)
            self.assertNotIn("short_isl_profile", cluster)
            self.assertEqual(
                row["design"]["satellite_count"],
                row["design"]["architecture_analysis"]["compute_cell_count"] * cluster["satellites_per_cluster"],
            )
            self.assertEqual(row["cost"]["satellite_count"], row["design"]["satellite_count"])


if __name__ == "__main__":
    unittest.main()
