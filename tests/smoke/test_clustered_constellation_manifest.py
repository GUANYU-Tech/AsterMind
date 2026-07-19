import json
import unittest
from pathlib import Path

from orbitserve.experiments import load_experiment_spec, run_experiment


ROOT = Path(__file__).resolve().parents[2]


class ClusteredConstellationManifestTests(unittest.TestCase):
    def test_smoke_clustered_constellation_manifest_runs(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_clustered_constellation.json")
        self.assertEqual(spec.experiment_id, "smoke_clustered_constellation")
        self.assertEqual(spec.constellation.compute_cell_count, 4)
        self.assertEqual(spec.constellation.satellite_count, 64)

        analysis = spec.to_dict()["constellation"]["architecture_analysis"]
        cluster = analysis["orbital_sets"][0]["cluster"]
        self.assertEqual(cluster["grid_shape"], [4, 4])
        self.assertEqual(cluster["satellites_per_cluster"], 16)
        self.assertNotIn("topology", cluster)
        self.assertNotIn("short_isl_profile", cluster)
        self.assertNotIn("aggregate_topology", analysis)

        payload = run_experiment(spec).to_dict()
        self.assertTrue(payload["cost"]["evaluated"])
        self.assertTrue(payload["performance"]["evaluated"])
        self.assertFalse(payload["slo"]["evaluated"])
        self.assertEqual(payload["module_status"]["dse"], "disabled")
        self.assertEqual(payload["cost"]["satellite_count"], 64)
        self.assertEqual(payload["cost"]["single_satellite"]["provenance"]["isl_total_bandwidth_gbps"], 400.0)
        self.assertEqual(payload["performance"]["details"]["hardware"]["inter_satellite_bandwidth_gbps"], 400.0)
        self.assertEqual(payload["performance"]["details"]["hardware"]["laser_terminal_count"], 4)
        json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
