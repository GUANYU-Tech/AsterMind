import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


from orbitserve.experiments import load_experiment_spec
from orbitserve.orbit.simulation_trace_builder import build_surrogate_orbit_trace
from orbitserve.orbit.simulation_traces import (
    SURROGATE_ORBIT_TRACE,
    OrbitTraceManifest,
    load_surrogate_trace_bundle,
)
from orbitserve.workload import materialize_workload_bundle
from orbitserve.runtime.routing import find_access_candidates, find_next_access_candidate


ROOT = Path(__file__).resolve().parents[3]


class SurrogateTraceBuilderTests(unittest.TestCase):
    def _base_inputs(self):
        spec = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_trace_backed_runtime.json")
        workload = materialize_workload_bundle(spec.workload, base_dir=Path(spec.base_dir))
        return spec, workload

    def test_builder_writes_sha_checked_trace_bundle(self):
        spec, workload = self._base_inputs()
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(spec.simulation, trace_output_dir=tmpdir)

            result = build_surrogate_orbit_trace(
                config=config,
                constellation=spec.constellation,
                workload_generation=workload,
                base_dir=ROOT,
                experiment_id="unit_trace",
            )
            manifest = OrbitTraceManifest.from_value(result.to_trace_manifest())
            bundle = load_surrogate_trace_bundle(manifest)

            self.assertEqual(manifest.trace_model, SURROGATE_ORBIT_TRACE)
            self.assertTrue(manifest.evaluated)
            self.assertFalse(manifest.valid_for_paper_main_results)
            self.assertGreater(len(bundle.satellite_catalog), 0)
            self.assertGreater(len(bundle.access_windows), 0)
            self.assertGreater(len(bundle.links), 0)
            self.assertGreater(len(bundle.energy_windows), 0)

    def test_gap_request_has_no_current_access_but_has_defer_window(self):
        spec, workload = self._base_inputs()
        first_request = workload.requests[0]
        request_id = first_request["request_id"]
        ingress_id = first_request["ingress_id"]
        arrival_s = float(first_request["arrival_time_s"])
        with tempfile.TemporaryDirectory() as tmpdir:
            config = replace(
                spec.simulation,
                trace_output_dir=tmpdir,
                trace_options={
                    **spec.simulation.trace_options,
                    "access_gap_request_ids": [request_id],
                    "gap_next_access_delay_s": 60.0,
                    "generate_defer_windows_for_gap_requests": True,
                },
            )

            result = build_surrogate_orbit_trace(
                config=config,
                constellation=spec.constellation,
                workload_generation=workload,
                base_dir=ROOT,
                experiment_id="unit_gap",
            )
            bundle = load_surrogate_trace_bundle(OrbitTraceManifest.from_value(result.to_trace_manifest()))

            self.assertEqual(find_access_candidates(bundle, ingress_id, arrival_s), ())
            next_access = find_next_access_candidate(bundle, ingress_id, arrival_s)
            self.assertIsNotNone(next_access)
            self.assertGreaterEqual(next_access.start_s, arrival_s + 60.0)


if __name__ == "__main__":
    unittest.main()
