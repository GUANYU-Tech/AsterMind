import tempfile
import unittest
from dataclasses import replace
from orbitserve.experiments import run_experiment
from tests.system_case import physical_case

class ClusteredPhysicalDSETests(unittest.TestCase):
    def test_formation_pitch_sweep_changes_sgp4_geometry_without_changing_hardware_cost(self):
        spec=replace(physical_case(),dse={"enabled":True,"sweep":{"constellations.compute_formations.0.cluster.formation_pitch_m":[500,1000]}})
        with tempfile.TemporaryDirectory() as folder:
            result=run_experiment(spec,output_dir=folder,_workers=1)
            self.assertEqual(result.dse["design_count"],2)
            a,b=result.dse["rows"]
            self.assertEqual(a["total_cost_usd"],b["total_cost_usd"])
            self.assertNotEqual(a["design_id"],b["design_id"])
            self.assertTrue(all(row["runtime"]["request_summary"]["total_request_count"]==3 for row in result.dse["rows"]))
