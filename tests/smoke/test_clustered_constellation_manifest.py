import tempfile
import unittest
from orbitserve.experiments import run_experiment
from tests.system_case import physical_case

class ClusteredConstellationManifestTests(unittest.TestCase):
    def test_physical_cell_and_logical_serving_target_are_consistent(self):
        with tempfile.TemporaryDirectory() as folder:
            result=run_experiment(physical_case(),output_dir=folder,_workers=1)
            self.assertEqual(result.serving["physical_satellite_count"],4)
            self.assertEqual(result.cost["compute"]["compute_node_count"],2)
            self.assertEqual(result.cost["compute"]["physical_satellite_count"],8)
            self.assertEqual(result.runtime.request_summary["completed_request_count"],3)
            self.assertTrue(all(len(row["placement"]["compute_owner_satellite_ids"])==4 for row in result.runtime.request_records))
