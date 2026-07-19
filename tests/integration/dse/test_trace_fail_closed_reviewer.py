import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


from orbitserve.orbit.simulation_traces import (
    IDEAL_FULL_VISIBILITY,
    SURROGATE_ORBIT_TRACE,
)
from orbitserve.dse.search import DSESpec, run_dse
from orbitserve.experiments import load_experiment_spec
from orbitserve.orbit.simulation_trace_builder import build_surrogate_orbit_trace as real_build_surrogate_orbit_trace
from orbitserve.dse.static_slo import evaluate_static_slo_proxy


ROOT = Path(__file__).resolve().parents[3]


def _workload():
    return {
        "requests": ({
            "request_id": "r0",
            "arrival_time_s": 0.0,
            "prompt_tokens": 1,
            "output_tokens": 1,
            "ttft_deadline_s": 10.0,
            "e2e_deadline_s": 20.0,
            "ingress_id": "ingress",
        },),
        "timebin_demand": ({
            "time_bin_start_s": 0.0,
            "time_bin_end_s": 60.0,
            "request_count": 1,
        },),
        "summary": {},
    }


def _serving():
    return {
        "evaluated": True,
        "feasible": True,
        "request_throughput_per_s": 1.0,
        "prefill_tokens_per_s": 10.0,
        "decode_tokens_per_s": 10.0,
    }


def _write_manifest(root: Path, *, access_payload=None, energy_payload=None):
    access = root / "access.json"
    link = root / "link.json"
    energy = root / "energy.json"
    access.write_text(json.dumps(access_payload if access_payload is not None else {
        "trace_id": "review",
        "access_windows": [{
            "ingress_id": "ingress",
            "satellite_id": "sat-0",
            "start_s": 0.0,
            "end_s": 60.0,
            "propagation_delay_s": 0.01,
        }],
    }), encoding="utf-8")
    link.write_text(json.dumps({"trace_id": "review", "links": []}), encoding="utf-8")
    energy.write_text(json.dumps(energy_payload if energy_payload is not None else {
        "trace_id": "review",
        "satellite_initial_soc": {"sat-0": 0.8},
        "energy_windows": [{
            "satellite_id": "sat-0",
            "start_s": 0.0,
            "end_s": 60.0,
            "sunlit": True,
            "dvfs_multiplier": 1.0,
        }],
    }), encoding="utf-8")
    return {
        "trace_model": SURROGATE_ORBIT_TRACE,
        "evaluated": True,
        "trace_id": "review",
        "access_trace_path": access.name,
        "link_trace_path": link.name,
        "energy_trace_path": energy.name,
        "provenance": {"trace_dir": str(root)},
    }


def _evaluate(simulation_result, *, base_dir=None):
    return evaluate_static_slo_proxy(
        workload_generation=_workload(),
        serving_throughput=_serving(),
        simulation_result=simulation_result,
        base_dir=base_dir,
    ).to_dict()


class TraceFailClosedReviewerTests(unittest.TestCase):
    def test_mixed_good_and_failed_dse_candidates_only_frontiers_good_candidate(self):
        base = load_experiment_spec(ROOT / "tests/fixtures/manifests/smoke_trace_backed_runtime.json")
        calls = 0

        def fail_first_builder(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("reviewer first candidate failure")
            return real_build_surrogate_orbit_trace(*args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            spec = replace(
                base,
                simulation=replace(base.simulation, trace_output_dir=tmpdir),
                dse=DSESpec(enabled=True, sweep={"plane_count": (2, 3)}),
            )
            with patch(
                "orbitserve.orbit.simulation_trace_builder.build_surrogate_orbit_trace",
                side_effect=fail_first_builder,
            ):
                payload = run_dse(spec).to_dict()

        self.assertEqual(len(payload["rows"]), 2)
        good_ids = {row["design_id"] for row in payload["rows"] if row["slo"]["evaluated"]}
        bad_ids = {row["design_id"] for row in payload["rows"] if not row["slo"]["evaluated"]}
        self.assertEqual(len(good_ids), 1)
        self.assertEqual(len(bad_ids), 1)
        self.assertTrue(payload["static_frontier"]["evaluated"])
        frontier_ids = {row["design_id"] for row in payload["static_frontier"]["frontier_rows"]}
        self.assertEqual(frontier_ids, good_ids)
        excluded_ids = {row["design_id"] for row in payload["static_frontier"]["excluded_candidates"]}
        self.assertEqual(excluded_ids, bad_ids)

    def test_top_level_trace_evaluated_false_overrides_stale_evaluated_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _write_manifest(Path(tmpdir))
            result = _evaluate({
                "trace_model": SURROGATE_ORBIT_TRACE,
                "trace_evaluated": False,
                "trace_manifest": manifest,
                "warnings": ("forced builder failure",),
            })
        self.assertFalse(result["evaluated"])

    def test_explicit_ideal_must_itself_be_evaluated(self):
        result = _evaluate({
            "trace_model": IDEAL_FULL_VISIBILITY,
            "trace_evaluated": False,
            "trace_manifest": {
                "trace_model": IDEAL_FULL_VISIBILITY,
                "evaluated": False,
            },
        })
        self.assertFalse(result["evaluated"])

    def test_unsupported_or_empty_configured_trace_fails_closed(self):
        for simulation_result in (
            {"trace_model": "unsupported_trace", "trace_evaluated": False},
            {"trace_model": "", "trace_evaluated": False},
            {"trace_manifest": {}},
        ):
            with self.subTest(simulation_result=simulation_result):
                self.assertFalse(_evaluate(simulation_result)["evaluated"])

    def test_loaded_bundle_without_access_windows_is_not_a_valid_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _write_manifest(Path(tmpdir), access_payload={"trace_id": "review"})
            result = _evaluate({"trace_evaluated": True, "trace_manifest": manifest})
        self.assertFalse(result["evaluated"])

    def test_loaded_bundle_with_explicit_empty_access_windows_is_valid_zero_coverage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _write_manifest(
                Path(tmpdir),
                access_payload={"trace_id": "review", "access_windows": []},
            )
            result = _evaluate({"trace_evaluated": True, "trace_manifest": manifest})
        self.assertTrue(result["evaluated"])
        self.assertTrue(result["slo"]["evaluated"])
        self.assertEqual(result["access_summary"]["request_access_coverage_ratio"], 0.0)
        self.assertEqual(result["slo"]["served_ratio"], 0.0)

    def test_loaded_bundle_without_energy_windows_is_not_a_valid_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _write_manifest(Path(tmpdir), energy_payload={"trace_id": "review"})
            result = _evaluate({"trace_evaluated": True, "trace_manifest": manifest})
        self.assertFalse(result["evaluated"])
        self.assertFalse(result["slo"]["evaluated"])
        self.assertIsNone(result["slo"]["served_ratio"])

    def test_access_window_missing_latency_does_not_default_to_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _write_manifest(Path(tmpdir), access_payload={
                "trace_id": "review",
                "access_windows": [{
                    "ingress_id": "ingress",
                    "satellite_id": "sat-0",
                    "start_s": 0.0,
                    "end_s": 60.0,
                }],
            })
            result = _evaluate({"trace_evaluated": True, "trace_manifest": manifest})
        self.assertFalse(result["evaluated"])

    def test_energy_window_missing_dvfs_does_not_default_to_no_derating(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _write_manifest(Path(tmpdir), energy_payload={
                "trace_id": "review",
                "energy_windows": [{
                    "satellite_id": "sat-0",
                    "start_s": 0.0,
                    "end_s": 60.0,
                    "sunlit": False,
                }],
            })
            result = _evaluate({"trace_evaluated": True, "trace_manifest": manifest})
        self.assertFalse(result["evaluated"])


if __name__ == "__main__":
    unittest.main()
