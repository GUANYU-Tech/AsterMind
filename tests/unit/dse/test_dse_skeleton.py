import unittest
import tempfile
from dataclasses import replace
from orbitserve.dse import physical_design_candidates,run_dse
from tests.system_case import physical_case

class PhysicalDSETests(unittest.TestCase):
    def test_sweep_enumerates_cartesian_product_deterministically(self):
        base=physical_case().physical_network
        config={"sweep":{"constellations.communication_shells.0.topology":["grid","honeycomb"],
                         "constellations.compute_formations.0.source.altitude_km":[750,850]}}
        rows=physical_design_candidates(base,config)
        self.assertEqual(len(rows),4)
        self.assertEqual(rows,physical_design_candidates(base,config))
    def test_direct_dse_reads_canonical_workload_and_runs_current_runtime(self):
        spec=replace(physical_case(),dse={"enabled":True})
        with tempfile.TemporaryDirectory() as folder:
            result=run_dse(spec,output_dir=folder,_workers=1)
            self.assertEqual(result["rows"][0]["runtime"]["request_summary"]["completed_request_count"],3)
            self.assertEqual(result["design_count"],1)
