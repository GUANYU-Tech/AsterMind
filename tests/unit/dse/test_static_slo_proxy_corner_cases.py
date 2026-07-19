import json
import sys
import tempfile
import unittest
from pathlib import Path


from orbitserve.orbit.simulation_traces import (
    CIRCULAR_GEOMETRY_TRACE,
    IDEAL_FULL_VISIBILITY,
    SURROGATE_ORBIT_TRACE,
    OrbitTraceManifest,
)
from orbitserve.dse.static_slo import evaluate_static_slo_proxy


def _request(
    request_id,
    arrival_time_s,
    *,
    prompt_tokens=10,
    output_tokens=10,
    ttft_deadline_s=1.0,
    e2e_deadline_s=2.0,
    ingress_id="ingress",
):
    return {
        "request_id": request_id,
        "arrival_time_s": float(arrival_time_s),
        "prompt_tokens": prompt_tokens,
        "new_user_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "ttft_deadline_s": ttft_deadline_s,
        "e2e_deadline_s": e2e_deadline_s,
        "slo_class": "interactive_5s",
        "ingress_id": ingress_id,
        "region_id": ingress_id,
        "user_region_id": ingress_id,
    }


def _timebin(start_s, end_s, *, request_count, prompt_tokens_total=0, output_tokens_total=0):
    return {
        "time_bin_start_s": float(start_s),
        "time_bin_end_s": float(end_s),
        "region_id": "ingress",
        "request_count": int(request_count),
        "prompt_tokens_total": int(prompt_tokens_total),
        "new_user_tokens_total": int(prompt_tokens_total),
        "output_tokens_total": int(output_tokens_total),
        "interactive_request_count": int(request_count),
        "burst_request_count": 0,
    }


def _workload(requests, *, timebins=None):
    prompt_total = sum(int(row["prompt_tokens"]) for row in requests)
    output_total = sum(int(row["output_tokens"]) for row in requests)
    if timebins is None:
        timebins = (_timebin(0, 60, request_count=len(requests), prompt_tokens_total=prompt_total, output_tokens_total=output_total),)
    return {
        "status": "loaded",
        "requests": tuple(requests),
        "timebin_demand": tuple(timebins),
        "summary": {
            "request_count": len(requests),
            "token_stats": {
                "prompt_tokens": {"p50": 10, "p95": 10, "p99": 10},
                "output_tokens": {"p50": 10, "p95": 10, "p99": 10},
            },
        },
    }


def _serving(*, prefill_tokens_per_s=10.0, decode_tokens_per_s=10.0, request_throughput_per_s=5.0):
    return {
        "evaluated": True,
        "feasible": True,
        "request_throughput_per_s": request_throughput_per_s,
        "prefill_tokens_per_s": prefill_tokens_per_s,
        "decode_tokens_per_s": decode_tokens_per_s,
        "kv_cache_capacity_tokens": 1_000_000,
        "satellite_count_used": 1,
        "provenance": {
            "token_distribution": {
                "prompt_tokens_p50": 10,
                "prompt_tokens_p95": 10,
                "output_tokens_p50": 10,
                "output_tokens_p95": 10,
            }
        },
    }


def _trace_manifest(root, *, ingress_ids=("ingress",), energy_multiplier=1.0):
    root = Path(root)
    access_path = root / "access_trace.json"
    link_path = root / "link_trace.json"
    energy_path = root / "energy_trace.json"
    access_windows = [
        {
            "window_id": f"access-{index}",
            "ingress_id": ingress_id,
            "satellite_id": "sat-0",
            "start_s": 0.0,
            "end_s": 120.0,
            "sgl_link_id": f"sgl:{ingress_id}:sat-0",
            "propagation_delay_s": 0.0,
            "bandwidth_bps": 1_000_000_000.0,
        }
        for index, ingress_id in enumerate(ingress_ids)
    ]
    access_path.write_text(
        json.dumps(
            {
                "trace_id": "corner_trace",
                "satellite_catalog": [{"satellite_id": "sat-0"}],
                "access_windows": access_windows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    link_path.write_text(json.dumps({"trace_id": "corner_trace", "links": []}, indent=2), encoding="utf-8")
    energy_path.write_text(
        json.dumps(
            {
                "trace_id": "corner_trace",
                "satellite_initial_soc": {"sat-0": 0.8},
                "energy_windows": [
                    {
                        "satellite_id": "sat-0",
                        "start_s": 0.0,
                        "end_s": 120.0,
                        "sunlit": energy_multiplier <= 1.0,
                        "dvfs_multiplier": energy_multiplier,
                        "charge_rate_w": 100.0,
                        "idle_power_w": 10.0,
                        "compute_energy_j_per_token": 0.1,
                        "link_energy_j_per_byte": 0.0,
                        "battery_capacity_j": 100_000.0,
                        "soc_min": 0.2,
                    }
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "trace_model": SURROGATE_ORBIT_TRACE,
        "evaluated": True,
        "trace_id": "corner_trace",
        "access_trace_path": access_path.name,
        "link_trace_path": link_path.name,
        "energy_trace_path": energy_path.name,
        "valid_for_paper_main_results": False,
        "scope": "trace_backed_runtime_surrogate",
        "provenance": {"trace_dir": str(root)},
    }


class StaticSLOProxyCornerCaseTests(unittest.TestCase):
    def test_requested_nonideal_trace_without_evaluated_manifest_fails_closed(self):
        request = _request("trace-build-failed", 0.0)

        for trace_model in (SURROGATE_ORBIT_TRACE, CIRCULAR_GEOMETRY_TRACE):
            with self.subTest(trace_model=trace_model):
                build_warning = f"{trace_model} build failed: forced builder failure"
                result = evaluate_static_slo_proxy(
                    workload_generation=_workload((request,)),
                    serving_throughput=_serving(),
                    simulation_result={
                        "trace_model": trace_model,
                        "trace_evaluated": False,
                        "trace_manifest": OrbitTraceManifest(
                            trace_model=trace_model,
                            warnings=(build_warning,),
                        ).to_dict(),
                    },
                ).to_dict()

                self.assertFalse(result["evaluated"])
                self.assertFalse(result["performance"]["evaluated"])
                self.assertFalse(result["slo"]["evaluated"])
                self.assertIsNone(result["slo"]["served_ratio"])
                self.assertIsNone(result["performance"]["e2e_p95_s"])
                self.assertTrue(any("did not produce an evaluated trace manifest" in warning for warning in result["warnings"]))
                self.assertIn(build_warning, result["warnings"])
                self.assertFalse(any("assuming full visibility" in warning for warning in result["warnings"]))

    def test_evaluated_nonideal_manifest_with_missing_files_fails_closed(self):
        request = _request("missing-trace-files", 0.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = evaluate_static_slo_proxy(
                workload_generation=_workload((request,)),
                serving_throughput=_serving(),
                simulation_result={
                    "trace_manifest": {
                        "trace_model": SURROGATE_ORBIT_TRACE,
                        "evaluated": True,
                        "trace_id": "missing-files",
                        "access_trace_path": "missing-access.json",
                        "link_trace_path": "missing-link.json",
                        "energy_trace_path": "missing-energy.json",
                    }
                },
                base_dir=tmpdir,
            ).to_dict()

        self.assertFalse(result["evaluated"])
        self.assertFalse(result["slo"]["evaluated"])
        self.assertIsNone(result["slo"]["served_ratio"])
        self.assertTrue(any("manifest could not be loaded" in warning for warning in result["warnings"]))
        self.assertTrue(any("missing surrogate trace file" in warning for warning in result["warnings"]))
        self.assertFalse(any("assuming full visibility" in warning for warning in result["warnings"]))

    def test_explicit_ideal_full_visibility_remains_evaluated(self):
        request = _request("explicit-ideal", 0.0)
        result = evaluate_static_slo_proxy(
            workload_generation=_workload((request,)),
            serving_throughput=_serving(),
            simulation_result={"trace_manifest": OrbitTraceManifest.ideal_full_visibility().to_dict()},
        ).to_dict()

        self.assertTrue(result["evaluated"])
        self.assertTrue(result["slo"]["evaluated"])
        self.assertEqual(result["access_summary"]["trace_model"], IDEAL_FULL_VISIBILITY)
        self.assertEqual(result["access_summary"]["assumption"], "full_visibility_zero_access_wait")
        self.assertTrue(any("explicitly requested ideal_full_visibility" in warning for warning in result["warnings"]))

    def test_deadline_equality_and_independent_ttft_e2e_misses(self):
        requests = (
            _request("exact", 0.0, ttft_deadline_s=1.0, e2e_deadline_s=2.0),
            _request("ttft_only", 10.0, ttft_deadline_s=0.9, e2e_deadline_s=3.0),
            _request("e2e_only", 20.0, ttft_deadline_s=2.0, e2e_deadline_s=1.9),
            _request("both", 30.0, ttft_deadline_s=0.9, e2e_deadline_s=1.9),
            _request("missing_ttft", 40.0, ttft_deadline_s=None, e2e_deadline_s=2.0),
        )

        result = evaluate_static_slo_proxy(
            workload_generation=_workload(requests),
            serving_throughput=_serving(),
        ).to_dict()

        self.assertTrue(result["evaluated"])
        self.assertAlmostEqual(result["slo"]["served_ratio"], 2 / 5)
        self.assertAlmostEqual(result["slo"]["timeout_ratio"], 3 / 5)
        self.assertAlmostEqual(result["slo"]["rejected_ratio"], 0.0)
        self.assertIsNotNone(result["performance"]["ttft_p95_s"])
        self.assertIsNotNone(result["performance"]["ttft_p99_s"])
        self.assertIsNotNone(result["performance"]["e2e_p95_s"])
        self.assertIsNotNone(result["performance"]["e2e_p99_s"])

    def test_access_gap_rejects_uncovered_ingress_without_hiding_tail_metrics(self):
        requests = (
            _request("covered", 0.0, ingress_id="covered", e2e_deadline_s=2.0),
            _request("uncovered", 1.0, ingress_id="uncovered", e2e_deadline_s=2.0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _trace_manifest(tmpdir, ingress_ids=("covered",))
            result = evaluate_static_slo_proxy(
                workload_generation=_workload(requests),
                serving_throughput=_serving(),
                simulation_result={"trace_manifest": manifest},
            ).to_dict()

        self.assertTrue(result["evaluated"])
        self.assertTrue(result["access_summary"]["evaluated"])
        self.assertAlmostEqual(result["access_summary"]["request_access_coverage_ratio"], 0.5)
        self.assertAlmostEqual(result["slo"]["rejected_ratio"], 0.5)
        self.assertAlmostEqual(result["slo"]["served_ratio"], 0.5)
        self.assertIsNotNone(result["performance"]["e2e_p99_s"])

    def test_energy_dvfs_summary_changes_static_slo_result(self):
        request = _request("energy_sensitive", 0.0, e2e_deadline_s=3.0)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = _trace_manifest(tmpdir, energy_multiplier=2.0)
            result = evaluate_static_slo_proxy(
                workload_generation=_workload((request,)),
                serving_throughput=_serving(),
                simulation_result={"trace_manifest": manifest},
            ).to_dict()

        self.assertTrue(result["evaluated"])
        self.assertTrue(result["energy_summary"]["evaluated"])
        self.assertAlmostEqual(result["energy_summary"]["capacity_factor"], 0.5)
        self.assertGreater(result["runtime_metrics"]["e2e_p95_s"], 3.0)
        self.assertAlmostEqual(result["slo"]["timeout_ratio"], 1.0)
        self.assertAlmostEqual(result["slo"]["served_ratio"], 0.0)

    def test_timebin_burst_summary_survives_even_when_arrivals_are_sparse(self):
        requests = (
            _request("sparse-0", 0.0, e2e_deadline_s=2.0),
            _request("sparse-1", 50.0, e2e_deadline_s=2.0),
        )
        timebins = (
            _timebin(0, 10, request_count=100, prompt_tokens_total=1000, output_tokens_total=1000),
            _timebin(10, 20, request_count=0),
        )

        result = evaluate_static_slo_proxy(
            workload_generation=_workload(requests, timebins=timebins),
            serving_throughput=_serving(request_throughput_per_s=5.0),
        ).to_dict()

        self.assertTrue(result["evaluated"])
        self.assertTrue(result["timebin_summary"]["evaluated"])
        self.assertEqual(result["timebin_summary"]["request_count"], 100)
        self.assertAlmostEqual(result["timebin_summary"]["peak_request_rate_per_s"], 10.0)
        self.assertAlmostEqual(result["timebin_summary"]["peak_to_capacity_ratio"], 2.0)
        self.assertAlmostEqual(result["timebin_summary"]["overloaded_timebin_fraction"], 0.5)

    def test_missing_timebin_or_requests_fail_closed_instead_of_faking_slo(self):
        result = evaluate_static_slo_proxy(
            workload_generation={"requests": (), "timebin_demand": (), "summary": {}},
            serving_throughput=_serving(),
        ).to_dict()

        self.assertFalse(result["evaluated"])
        self.assertFalse(result["slo"]["evaluated"])
        self.assertIsNone(result["slo"]["served_ratio"])
        self.assertTrue(
            any("requires canonical workload requests and timebin_demand" in warning for warning in result["warnings"])
        )


if __name__ == "__main__":
    unittest.main()
