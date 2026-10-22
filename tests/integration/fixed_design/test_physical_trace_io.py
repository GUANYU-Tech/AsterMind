import tempfile
import unittest
from pathlib import Path
from orbitserve.experiments.network import run_network_experiment
from orbitserve.orbit import load_orbit_trace
from tests.integration.fixed_design.test_dynamic_network_experiment import manifest


class PhysicalTraceIOTests(unittest.TestCase):
    def test_altered_or_missing_trace_artifacts_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            run_network_experiment(manifest(), folder, _workers=1)
            root = Path(folder) / "orbit_trace"
            trace = load_orbit_trace(root / "trace_manifest.json")
            name = next(iter(trace.trace_manifest.files.values()))
            if isinstance(name, dict):
                name = name["path"]
            artifact = root / name
            artifact.write_bytes(artifact.read_bytes() + b" ")
            with self.assertRaises((ValueError, OSError)):
                load_orbit_trace(root / "trace_manifest.json")
            artifact.unlink()
            with self.assertRaises((ValueError, OSError)):
                load_orbit_trace(root / "trace_manifest.json")
