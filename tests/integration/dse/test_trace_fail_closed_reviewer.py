import tempfile
import unittest
from dataclasses import replace
from orbitserve.runtime import run_simulation,RuntimeStrategySpec
from orbitserve.experiments import run_experiment
from tests.system_case import physical_case
from tests.unit.runtime.test_runtime_engine import _scenario,_trace

class TraceFailClosedTests(unittest.TestCase):
    def test_unknown_workload_ingress_fails_before_runtime(self):
        spec=physical_case()
        physical={**spec.physical_network,"ground_ingresses":[{**spec.physical_network["ground_ingresses"][0],"ingress_id":"other"}]}
        with tempfile.TemporaryDirectory() as folder,self.assertRaisesRegex(ValueError,"ingress"):
            run_experiment(replace(spec,physical_network=physical),output_dir=folder,_workers=1)
    def test_explicit_zero_coverage_is_evaluated_but_completes_no_requests(self):
        trace=replace(_trace(),sgl_windows=())
        result=run_simulation(_scenario(RuntimeStrategySpec(),trace=trace))
        self.assertTrue(result.evaluated)
        self.assertEqual(result.request_summary["completed_request_count"],0)
    def test_missing_energy_does_not_default_to_unlimited_compute(self):
        trace=replace(_trace(),sunlight_windows=())
        result=run_simulation(_scenario(RuntimeStrategySpec(),trace=trace))
        self.assertEqual(result.request_summary["completed_request_count"],0)
    def test_missing_boundary_position_fails_closed(self):
        trace=_trace()
        trace=replace(trace,ephemeris_samples=tuple(r for r in trace.ephemeris_samples if r["time_s"]==0.))
        with self.assertRaisesRegex(ValueError,"cover"):
            run_simulation(_scenario(RuntimeStrategySpec(),trace=trace))
