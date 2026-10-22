import unittest
from dataclasses import replace
from orbitserve.dse import DSESpec
from orbitserve.dse.satellite_budget import analyze_communication_window_satellite_budget_from_bundle
from tests.unit.runtime.test_runtime_engine import _trace

def bundle():
    trace=_trace()
    return replace(trace,trace_manifest=replace(trace.trace_manifest,time={**trace.time,"end_s":120.,"duration_s":120.}),
        satellite_catalog=tuple({"satellite_id":f"sat-{i}"} for i in range(4)),
        ephemeris_samples=tuple({"satellite_id":f"sat-{i}","time_s":t,"position_itrs_km":[7000.,0.,0.]}
                                for i in range(4) for t in (0.,30.,60.,90.,120.)),
        sgl_windows=tuple({"ingress_id":g,"satellite_id":s,"start_s":a,"end_s":b} for g,s,a,b in
            (("gw-0","sat-0",0.,60.),("gw-0","sat-1",0.,60.),("gw-0","sat-2",60.,120.),("gw-1","sat-3",0.,120.))))

class CommunicationWindowBudgetTests(unittest.TestCase):
    def test_budget_uses_request_weighted_current_orbit_contacts(self):
        result=analyze_communication_window_satellite_budget_from_bundle(bundle(),
            workload_generation={"requests":({"arrival_time_s":10.,"gateway_id":"gw-0"},{"arrival_time_s":70.,"gateway_id":"gw-0"})},
            config={"enabled":True,"candidate_satellite_budgets":[1,2],"target_coverage_ratio":.95})
        self.assertTrue(result["evaluated"])
        self.assertAlmostEqual(result["candidate_budget_rows"][0]["window_limited_capacity_factor"],.75)
        self.assertEqual(result["selected_budget"]["satellite_budget_used"],2)
        self.assertFalse(result["valid_for_paper_main_results"])
    def test_budget_configuration_round_trips(self):
        spec=DSESpec.from_dict({"communication_window_budget":{"enabled":True,"candidate_satellite_budgets":[1,2]}})
        self.assertEqual(spec.to_dict()["communication_window_budget"]["candidate_satellite_budgets"],[1,2])
