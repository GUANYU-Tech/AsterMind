import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class CliExampleTests(unittest.TestCase):
    def test_cli_reads_case_and_writes_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "h200.json"
            env = dict(os.environ)
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "orbitserve.architecture.single_satellite_cost.cli",
                    str(ROOT / "examples/single_satellite/h200_gpu_derived.json"),
                    "--out",
                    str(out_path),
                ],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            data = json.loads(out_path.read_text(encoding="utf-8"))
            for key in ("C_sat_usd", "M_sat_kg", "P_peak_w", "feasible", "provenance"):
                self.assertIn(key, data)



if __name__ == "__main__":
    unittest.main()
