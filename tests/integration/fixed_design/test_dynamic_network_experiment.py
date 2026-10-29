import gzip
import json
import tempfile
import unittest
from pathlib import Path

from orbitserve.experiments.network import run_network_experiment
from tests.unit.orbit.test_sgp4_elements import _bundle


def manifest():
    elements = _bundle()
    elements["satellites"][0]["mean_anomaly_deg"] = 0.01
    return {
        "schema_version": "orbitserve.network_experiment.v1",
        "time": {
            "epoch_utc": "2026-07-20T00:00:00.000000Z",
            "duration_s": 120.0,
            "step_s": 60.0,
        },
        "elements": elements,
        "ground_ingresses": [
            {
                "ingress_id": "ground",
                "lat_deg": 0.0,
                "lon_deg": 0.0,
                "min_elevation_deg": 10.0,
            }
        ],
        "network": {
            "nodes": [
                {
                    "node_id": "sat-00001",
                    "role": "communication",
                    "isl_terminals": 1,
                    "sgl_terminals": 1,
                },
                {"node_id": "sat-00002", "role": "compute", "isl_terminals": 1},
                {"node_id": "ground", "role": "ground", "sgl_terminals": 1},
            ],
            "communication_topology": "explicit",
            "link_profiles": {
                "cross_layer": {"capacity_model": "fixed", "bandwidth_bps": 1e9, "setup_time_s": 2.0},
                "sgl": {
                    "bandwidth_bps": 1e9,
                    "setup_time_s": 2.0,
                    "sun_exclusion_angle_deg": 3.0,
                },
            },
        },
        "route_probes": [
            {"source": "sat-00001", "target": "sat-00002", "payload_bytes": 1000.0}
        ],
    }


class DynamicNetworkExperimentTests(unittest.TestCase):
    def test_fault_audit_reacquisition_and_online_probe_knowledge_are_consistent(self):
        value = manifest()
        value['network']['fault_timing'] = 'continuous'
        value["network"]["faults"] = [
            dict(
                target_type="link",
                target_id="cross_layer:sat-00001:sat-00002",
                start_s=0.5,
                end_s=30.5,
            )
        ]
        with tempfile.TemporaryDirectory() as folder:
            result = run_network_experiment(value, folder, _workers=1)
            self.assertTrue(result["checks"]["passed"])
            # At t=0 the probe does not know the future .5 s outage.
            self.assertTrue(result["routes"][0]["completed"])
            self.assertAlmostEqual(result["routes"][0]["path"][0]["start_s"], 2.0)
            restored = next(
                link
                for link in result["snapshots"][1]["links"]
                if link["kind"] == "cross_layer"
            )
            self.assertEqual(restored["state"], "UP")
            self.assertEqual(restored["setup_start_s"], 30.5)
            self.assertEqual(restored["ready_s"], 32.5)
            self.assertTrue(
                any(
                    e["time_s"] == 32.5 and e["state"] == "UP" for e in result["events"]
                )
            )

    def test_real_sgp4_snapshots_events_and_routes(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_network_experiment(manifest(), folder)
            self.assertTrue(result["checks"]["passed"])
            self.assertEqual(result["boundary_sample_times_s"], (0.0, 60.0, 120.0))
            self.assertEqual(len(result["snapshots"]), 2)
            self.assertEqual(result["checks"]["physical_satellites"], 2)
            self.assertEqual(len(result["snapshots"][0]["nodes"]), 3)
            self.assertNotEqual(
                result["snapshots"][0]["nodes"][1]["position_itrs_km"],
                result["snapshots"][1]["nodes"][1]["position_itrs_km"],
            )
            self.assertTrue(result["routes"][0]["completed"])
            self.assertTrue(
                any(e["state"] == "UP" and e["time_s"] == 2.0 for e in result["events"])
            )
            self.assertTrue((Path(folder) / "network_result.json").is_file())
            self.assertFalse((Path(folder) / "network_replay.html").exists())
            saved = json.loads((Path(folder) / "network_result.json").read_text())
            self.assertNotIn("nodes", saved["snapshots"][0])
            self.assertEqual(len(saved["artifacts"]["frames"]), 3)
            expected_ids = {edge["link_id"] for edge in result["candidate_links"]}
            for entry in saved["artifacts"]["frames"]:
                with gzip.open(Path(folder) / entry["artifact"], "rt") as stream:
                    frame = json.load(stream)
                self.assertEqual(frame["time_s"], entry["time_s"])
                self.assertEqual(len(frame["nodes"]), 3)
                self.assertEqual(
                    {edge["link_id"] for edge in frame["links"]}, expected_ids
                )
                self.assertTrue(
                    all(
                        edge["attributes_time_s"] == entry["time_s"]
                        for edge in frame["links"]
                    )
                )
                events = frame["events_until_next_boundary"]
                self.assertTrue(
                    all(
                        entry["time_s"] < event["time_s"] <= entry["time_s"] + 60
                        for event in events
                    )
                )
                if entry["time_s"] == 0:
                    self.assertTrue(
                        any(
                            event["time_s"] == 2 and event["state"] == "UP"
                            for event in events
                        )
                    )
                if entry["time_s"] == 120:
                    self.assertTrue(
                        all(edge["reason"] == "horizon_end" for edge in frame["links"])
                    )
            self.assertLess(
                (Path(folder) / "network_result.json").stat().st_size, 20000
            )
            with self.assertRaisesRegex(FileExistsError, "completed experiment"):
                run_network_experiment(manifest(), folder)

    def test_serial_parallel_geometry_bytes_and_network_results_match(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            serial = run_network_experiment(manifest(), root / "serial", _workers=1)
            parallel = run_network_experiment(manifest(), root / "parallel", _workers=2)
            self.assertEqual(serial["snapshots"], parallel["snapshots"])
            self.assertEqual(serial["events"], parallel["events"])
            self.assertEqual(serial["routes"], parallel["routes"])
            for path in (root / "serial" / "orbit_trace").glob("*.json"):
                self.assertEqual(
                    path.read_bytes(),
                    (root / "parallel" / "orbit_trace" / path.name).read_bytes(),
                    path.name,
                )

    def test_non_integral_time_grid_is_rejected(self):
        value = manifest()
        value["time"]["duration_s"] = 61.0
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(ValueError, "integer multiple"),
        ):
            run_network_experiment(value, folder)
