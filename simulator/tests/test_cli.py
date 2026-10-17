import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class CLITests(unittest.TestCase):
    def test_failed_initialization_has_nonzero_exit_and_reviewable_report(self):
        root = Path(__file__).resolve().parents[2]
        temporary_root = root/"simulator/runs/test_tmp"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(temporary_root.resolve()))
            report = Path(directory)/"failed.json"
            result = subprocess.run([sys.executable, "-m", "simulator", "--scenario",
                str(root/"simulator/examples/two_cells.json"), "--midlayer-root", str(Path(directory)/"absent"),
                "--output", str(report)], cwd=root, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(report.is_file(), result.stderr)
            data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(data["status"], "failed")
            self.assertEqual(data["phase"], "initialization")


if __name__ == "__main__":
    unittest.main()
