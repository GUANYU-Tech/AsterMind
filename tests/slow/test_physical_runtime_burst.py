import unittest
from dataclasses import replace
from orbitserve.runtime import RuntimeStrategySpec,run_simulation
from orbitserve.workload.contract import reject_runtime_fields
from tests.unit.runtime.test_runtime_engine import _scenario,_request

class PhysicalRuntimeBurstTests(unittest.TestCase):
    def test_burst_events_are_chronological_and_link_directions_never_overlap(self):
        strategy=RuntimeStrategySpec(latency_weight=1,kv_reuse_weight=0,energy_weight=0,compute_load_weight=0)
        requests=tuple(_request(str(i),7.25,session_id=str(i)) for i in range(60))
        spec=_scenario(strategy,requests=requests)
        spec=replace(spec,energy_profile={**spec.energy_profile,"battery_capacity_j":1e12})
        result=run_simulation(spec)
        self.assertEqual(result.request_summary["completed_request_count"],60)
        times=[e["time_s"] for e in result.state_summary["event_log"]]
        self.assertEqual(times,sorted(times))
        for row in result.state_summary["network"]["directed_reservations"]:
            intervals=row["intervals"]
            self.assertTrue(all(a[1]<=b[0]+1e-10 for a,b in zip(intervals,intervals[1:])))
    def test_workload_never_accepts_runtime_result_fields(self):
        for key in ("selected_path","ttft_s","soc_after","energy_violation"):
            with self.subTest(key=key),self.assertRaises(ValueError):
                reject_runtime_fields({"request_id":"bad",key:1},context="test")
