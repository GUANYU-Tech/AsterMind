import tempfile
import unittest
from pathlib import Path
from astermind.builder.physical.network import run_network_experiment
from astermind.builder.orbit import load_orbit_trace
from tests.system_case import physical_case


class PhysicalTraceContractTests(unittest.TestCase):
    def test_sgp4_trace_has_physical_members_separate_shells_and_geometry_only(self):
        with tempfile.TemporaryDirectory() as folder:
            result = run_network_experiment(
                physical_case()["physical_network"], folder, _workers=1
            )
            trace = load_orbit_trace(Path(folder) / "orbit_trace/trace_manifest.json")
            self.assertEqual(len(trace.satellite_catalog), 32)
            self.assertEqual(trace.time["step_s"], 1.0)
            self.assertEqual(len(result["snapshots"]), 120)
            self.assertEqual(len(trace.ephemeris_samples), 32 * 121)
            self.assertFalse(hasattr(trace, "network"))
            self.assertTrue(trace.sgl_windows)
            self.assertTrue(trace.sunlight_windows)
            self.assertNotEqual(
                trace.ephemeris_samples[0]["position_itrs_km"],
                trace.ephemeris_samples[32]["position_itrs_km"],
            )
            self.assertEqual(
                set(result["checks"]["communication_candidate_degrees"].values()), {4}
            )
