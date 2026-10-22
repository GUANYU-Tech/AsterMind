import tempfile
import unittest
from orbitserve.experiments import load_experiment_spec,run_experiment
from tests.system_case import ROOT

class PublicExampleTests(unittest.TestCase):
    def test_fixed_design_example_runs_all_current_modules(self):
        spec=load_experiment_spec(ROOT/"examples/experiments/fixed_design_geometry.json")
        with tempfile.TemporaryDirectory() as folder:
            result=run_experiment(spec,output_dir=folder,_workers=1)
            self.assertEqual(result.runtime.request_summary["completed_request_count"],3)
            self.assertTrue(all(value=="evaluated" for value in result.module_status.values()))
    def test_dse_example_evaluates_four_physical_candidates(self):
        spec=load_experiment_spec(ROOT/"examples/experiments/orbit_geometry_dse.json")
        with tempfile.TemporaryDirectory() as folder:
            result=run_experiment(spec,output_dir=folder,_workers=1)
            self.assertEqual(result.dse["design_count"],4)
            self.assertEqual(result.dse["row_count"],4)
            self.assertTrue(all(row["runtime"]["provenance"]["runtime_engine"]=="chronological_discrete_event_runtime" for row in result.dse["rows"]))
