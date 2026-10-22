import tempfile
import unittest
from dataclasses import replace
from orbitserve.experiments import run_experiment
from tests.system_case import physical_case

class PhysicalRuntimeMatrixTests(unittest.TestCase):
    def test_design_by_strategy_matrix_reuses_geometry_and_runs_every_request(self):
        spec=physical_case()
        a=spec.runtime_strategy.to_dict()
        b={**a,"routing_policy":"min_hops"}
        spec=replace(spec,dse={"enabled":True,
          "sweep":{"constellations.compute_formations.0.source.altitude_km":[750,850]},
          "runtime_matrix":{"enabled":True,"include_current_policy":False,"candidate_schedulers":[a,b]}})
        with tempfile.TemporaryDirectory() as folder:
            result=run_experiment(spec,output_dir=folder,_workers=1)
            self.assertEqual(result.dse["design_count"],2)
            self.assertEqual(result.dse["row_count"],4)
            for row in result.dse["rows"]:
                self.assertEqual(row["runtime"]["request_summary"]["total_request_count"],3)
                self.assertEqual(row["runtime"]["provenance"]["runtime_engine"],"chronological_discrete_event_runtime")
            self.assertTrue(result.dse["selected_rows"])
