import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from orbitserve.experiments import load_experiment_spec, run_experiment


ROOT = Path(__file__).resolve().parents[2]


class PublicExampleTests(unittest.TestCase):
    def _run_example(self, name: str):
        spec = load_experiment_spec(ROOT / "examples" / "experiments" / f"{name}.json")
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = replace(
                spec,
                simulation=replace(spec.simulation, trace_output_dir=str(Path(tmpdir) / "traces")),
            )
            return run_experiment(spec).to_dict()

    def test_fixed_design_geometry_example_runs_full_runtime_path(self):
        payload = self._run_example("fixed_design_geometry")
        self.assertEqual(payload["module_status"]["simulation"], "geometry_trace")
        self.assertEqual(payload["module_status"]["runtime_performance"], "evaluated")
        self.assertFalse(payload["performance"]["details"]["runtime"]["valid_for_paper_main_results"])
        self.assertEqual(payload["dse"]["design_count"], 0)

    def test_orbit_geometry_dse_example_evaluates_four_candidates(self):
        payload = self._run_example("orbit_geometry_dse")
        self.assertEqual(payload["module_status"]["runtime_performance"], "evaluated")
        self.assertFalse(payload["performance"]["details"]["runtime"]["valid_for_paper_main_results"])
        self.assertTrue(payload["dse"]["evaluated"])
        self.assertEqual(payload["dse"]["design_count"], 4)


if __name__ == "__main__":
    unittest.main()
