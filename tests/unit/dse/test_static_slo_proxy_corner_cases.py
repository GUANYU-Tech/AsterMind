import unittest
from dataclasses import replace
from orbitserve.dse.static_slo import evaluate_static_slo_proxy
from tests.unit.runtime.test_runtime_engine import _trace

class GeometryScreeningTests(unittest.TestCase):
    def test_geometry_screen_is_an_upper_bound_and_never_fabricates_runtime_metrics(self):
        result=evaluate_static_slo_proxy(simulation_result=_trace(),workload_generation={
            "session_requests":[{"ingress_id":"gateway-0","arrival_time_s":7.25},{"ingress_id":"missing","arrival_time_s":10.}]})
        self.assertTrue(result.evaluated)
        self.assertEqual(result.geometric_access_fraction,.5)
        self.assertEqual(result.runtime_metrics,{})
        self.assertFalse(result.provenance["runtime_metrics_inferred"])
    def test_missing_unvalidated_or_empty_inputs_fail_closed(self):
        for trace,requests in [(None,[{"ingress_id":"gateway-0","arrival_time_s":0.}]),
                               (_trace(),[]),
                               (replace(_trace(),trace_manifest=replace(_trace().trace_manifest,evaluated=False)),[{"ingress_id":"gateway-0","arrival_time_s":0.}])]:
            with self.subTest(trace=type(trace).__name__):
                result=evaluate_static_slo_proxy(simulation_result=trace,workload_generation={"requests":requests})
                self.assertFalse(result.evaluated)
    def test_explicit_empty_visibility_is_valid_zero_coverage(self):
        result=evaluate_static_slo_proxy(simulation_result=replace(_trace(),sgl_windows=()),workload_generation={"requests":[{"ingress_id":"gateway-0","arrival_time_s":0.}]})
        self.assertTrue(result.evaluated)
        self.assertEqual(result.geometric_access_fraction,0.)
