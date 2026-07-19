import csv
import copy
import json
import math
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path


import orbitserve.workload as workloads


REQUEST_REQUIRED_FIELDS = {
    "request_id",
    "arrival_time_s",
    "user_region_id",
    "region_id",
    "lat_deg",
    "lon_deg",
    "access_mode",
    "ingress_id",
    "ingress_lat_deg",
    "ingress_lon_deg",
    "gateway_id",
    "task_type",
    "model_family",
    "prompt_tokens",
    "new_user_tokens",
    "context_tokens_before_turn",
    "output_tokens",
    "input_bytes",
    "output_bytes",
    "slo_class",
    "ttft_deadline_s",
    "e2e_deadline_s",
    "priority",
    "session_id",
    "turn_index",
    "session_turn_count",
    "inter_turn_gap_s",
    "session_ttl_s",
    "context_window_tokens",
    "burst_id",
    "burst_multiplier",
    "seed",
}

RUNTIME_FIELDS = {
    "assigned_satellite",
    "assigned_deployment",
    "selected_path",
    "session_action",
    "served",
    "TTFT",
    "E2E",
    "ttft_s",
    "e2e_latency_s",
    "energy_violation",
    "cost_per_request",
    "soc_before",
    "soc_after",
}

TIMEBIN_REQUIRED_FIELDS = {
    "time_bin_start_s",
    "time_bin_end_s",
    "region_id",
    "request_count",
    "prompt_tokens_total",
    "new_user_tokens_total",
    "output_tokens_total",
    "interactive_request_count",
    "burst_request_count",
}


def _require_api(name):
    return getattr(workloads, name)


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _to_plain(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def _extract_loaded_manifest(loaded_bundle):
    data = _to_plain(loaded_bundle)
    if isinstance(data, dict):
        for key in ("manifest", "workload_manifest"):
            if key in data:
                return _to_plain(data[key])
        if data.get("schema_version") == "orbitserve.workload":
            return data
    if hasattr(loaded_bundle, "manifest"):
        return _to_plain(loaded_bundle.manifest)
    return {}


def _compile_description(description):
    compile_workload_description = _require_api("compile_workload_description")
    try:
        return compile_workload_description(description)
    except TypeError:
        description_path = Path(description["output"]["bundle_dir"]).parent / "workload_description.json"
        description_path.write_text(json.dumps(description, indent=2, sort_keys=True), encoding="utf-8")
        return compile_workload_description(description_path)


def _materialize_description(description):
    compiled = _compile_description(description)
    bundle = _require_api("generate_workload_bundle")(compiled)
    bundle_dir = Path(description["output"]["bundle_dir"])
    materialize_workload_bundle = _require_api("materialize_workload_bundle")
    try:
        materialized = materialize_workload_bundle(bundle, bundle_dir)
    except TypeError:
        materialized = materialize_workload_bundle(bundle)
    if materialized is None:
        return bundle_dir
    if isinstance(materialized, (str, Path)):
        return Path(materialized)
    if hasattr(materialized, "bundle_dir"):
        return Path(materialized.bundle_dir)
    return bundle_dir


def _load_bundle(bundle_dir):
    return _require_api("load_workload_bundle")(bundle_dir)


def _nearest_rank(values, quantile):
    ordered = sorted(values)
    index = max(0, math.ceil((quantile / 100.0) * len(ordered)) - 1)
    return ordered[index]


def _stats_block(summary, metric):
    if "token_stats" in summary:
        return summary["token_stats"][metric]
    if "tokens" in summary and metric in summary["tokens"]:
        return summary["tokens"][metric]
    raise AssertionError(f"missing token stats for {metric}")


def _slo_counts(summary):
    if "slo_mix" in summary and "class_counts" in summary["slo_mix"]:
        return summary["slo_mix"]["class_counts"]
    if "slo" in summary and "class_counts" in summary["slo"]:
        return summary["slo"]["class_counts"]
    if "slo_class_counts" in summary:
        return summary["slo_class_counts"]
    raise AssertionError("missing SLO class counts in workload summary")


def _synthetic_description(bundle_dir, seed=20260624):
    return {
        "schema_version": "orbitserve.workload_description",
        "workload_id": "unit_synthetic_gateway_sessions",
        "source_mode": "synthetic_generation",
        "output_bundle_schema": "orbitserve.workload",
        "seed": seed,
        "time": {
            "epoch": "synthetic_day_0",
            "time_horizon_s": 600,
            "time_bin_s": 60,
        },
        "geography": {
            "mode": "named_gateway_service_regions",
            "demand_weight_normalization": "require_sum_one",
            "regions": [
                {
                    "region_id": "na_east",
                    "region_name": "North America East",
                    "lat_deg": 39.0,
                    "lon_deg": -77.0,
                    "timezone_utc_offset_h": -5,
                    "demand_weight": 0.55,
                    "weight_source": "unit_test",
                    "market_class": "core_cloud",
                },
                {
                    "region_id": "europe_west",
                    "region_name": "Europe West",
                    "lat_deg": 50.1,
                    "lon_deg": 8.7,
                    "timezone_utc_offset_h": 1,
                    "demand_weight": 0.45,
                    "weight_source": "unit_test",
                    "market_class": "core_cloud",
                },
            ],
        },
        "gateways": {
            "access_mode": "gateway_sgl",
            "mapping_policy": "served_region_ids",
            "sites": [
                {
                    "gateway_id": "gw_us_east",
                    "lat_deg": 38.9,
                    "lon_deg": -77.0,
                    "served_region_ids": ["na_east"],
                    "gateway_bandwidth_bps": 100_000_000_000,
                    "availability_profile_id": "always_on",
                },
                {
                    "gateway_id": "gw_eu_west",
                    "lat_deg": 50.1,
                    "lon_deg": 8.7,
                    "served_region_ids": ["europe_west"],
                    "gateway_bandwidth_bps": 100_000_000_000,
                    "availability_profile_id": "always_on",
                },
            ],
        },
        "arrival": {
            "process": "nhpp_synthetic",
            "lambda_global": 0.35,
            "load_scale": 1.0,
            "diurnal_profile": "flat",
            "day_profile": "weekday",
            "seed": seed,
        },
        "tokens": {
            "model_family": "llama_70b_like",
            "bytes_per_input_token": 4,
            "bytes_per_output_token": 4,
            "request_metadata_bytes": 128,
            "response_metadata_bytes": 96,
            "task_mix": [
                {
                    "task_type": "conversation",
                    "weight": 0.7,
                    "prompt_distribution": {
                        "type": "quantile_buckets",
                        "p50": 256,
                        "p95": 1024,
                        "p99": 2048,
                        "max": 4096,
                    },
                    "output_distribution": {
                        "type": "quantile_buckets",
                        "p50": 128,
                        "p95": 512,
                        "p99": 1024,
                        "max": 2048,
                    },
                },
                {
                    "task_type": "summarization",
                    "weight": 0.3,
                    "prompt_distribution": {
                        "type": "quantile_buckets",
                        "p50": 1024,
                        "p95": 4096,
                        "p99": 8192,
                        "max": 8192,
                    },
                    "output_distribution": {
                        "type": "quantile_buckets",
                        "p50": 128,
                        "p95": 512,
                        "p99": 1024,
                        "max": 2048,
                    },
                },
            ],
        },
        "slo": {
            "mix": "balanced",
            "classes": [
                {
                    "slo_class": "interactive_5s",
                    "weight": 0.6,
                    "ttft_deadline_s": 5,
                    "e2e_deadline_s": 30,
                },
                {
                    "slo_class": "relaxed_60s",
                    "weight": 0.4,
                    "ttft_deadline_s": 10,
                    "e2e_deadline_s": 60,
                },
            ],
        },
        "sessions": {
            "session_mode": "open_loop_turn_arrivals",
            "session_ratio": 0.8,
            "context_window_tokens": 32768,
            "session_ttl_s": 300,
            "turn_count_distribution": {
                "type": "categorical",
                "values": {"1": 0.2, "2": 0.4, "4": 0.4},
            },
            "gap_distribution": {
                "type": "bucketed_uniform",
                "buckets": [
                    {"name": "short_gap", "weight": 0.7, "min_s": 5, "max_s": 30},
                    {"name": "medium_gap", "weight": 0.3, "min_s": 60, "max_s": 120},
                ],
            },
        },
        "bursts": [
            {
                "burst_id": "eu_burst",
                "region_ids": ["europe_west"],
                "start_time_s": 0,
                "duration_s": 600,
                "multiplier": 3.0,
                "shape": "step",
            }
        ],
        "validation": {
            "reject_runtime_fields": True,
            "sort_requests": True,
            "normalize_region_weights": False,
            "require_unique_request_id": True,
        },
        "output": {
            "bundle_dir": str(bundle_dir),
            "overwrite": True,
            "emit_timebin_demand": True,
        },
        "provenance": {
            "sources": ["unit test synthetic workload description"],
            "notes": ["Designed to exercise canonical bundle validation without runtime outputs."],
        },
    }


def _explicit_description(bundle_dir, requests_path, sessions_path=None, reject_runtime_fields=True):
    explicit = {
        "requests_path": str(requests_path),
        "format": "jsonl",
        "request_schema": "orbitserve.workload.request",
    }
    if sessions_path is not None:
        explicit["sessions_path"] = str(sessions_path)
    return {
        "schema_version": "orbitserve.workload_description",
        "workload_id": "unit_explicit_requests",
        "source_mode": "explicit_requests",
        "output_bundle_schema": "orbitserve.workload",
        "explicit_requests": explicit,
        "time": {
            "epoch": "synthetic_day_0",
            "time_horizon_s": 120,
            "time_bin_s": 60,
        },
        "geography": {
            "mode": "explicit_from_requests",
            "regions": [
                {
                    "region_id": "na_east",
                    "region_name": "North America East",
                    "lat_deg": 39.0,
                    "lon_deg": -77.0,
                    "timezone_utc_offset_h": -5,
                    "demand_weight": 1.0,
                    "weight_source": "explicit",
                    "market_class": "core_cloud",
                }
            ],
        },
        "gateways": {
            "access_mode": "gateway_sgl",
            "mapping_policy": "served_region_ids",
            "sites": [
                {
                    "gateway_id": "gw_us_east",
                    "lat_deg": 38.9,
                    "lon_deg": -77.0,
                    "served_region_ids": ["na_east"],
                    "gateway_bandwidth_bps": 100_000_000_000,
                    "availability_profile_id": "always_on",
                }
            ],
        },
        "arrival": {"process": "explicit_requests"},
        "tokens": {
            "validate_only": True,
            "model_family_fallback": "llama_70b_like",
            "bytes_per_input_token": 4,
            "bytes_per_output_token": 4,
            "request_metadata_bytes": 128,
            "response_metadata_bytes": 96,
        },
        "slo": {"validate_only": True},
        "sessions": {
            "validate_only": True,
            "fill_missing_sessions": "from_request_rows",
            "context_window_tokens": 4096,
            "session_ttl_s": 300,
        },
        "validation": {
            "reject_runtime_fields": reject_runtime_fields,
            "require_sorted_requests": False,
            "sort_requests": True,
            "require_unique_request_id": True,
        },
        "output": {
            "bundle_dir": str(bundle_dir),
            "overwrite": True,
            "emit_timebin_demand": True,
        },
        "provenance": {
            "sources": ["unit test explicit request JSONL"],
            "notes": ["Explicit rows intentionally omit canonical fillable fields."],
        },
    }


class SyntheticWorkloadBundleTests(unittest.TestCase):
    def test_synthetic_generation_materializes_complete_bundle_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = Path(tmpdir) / "bundle"
            materialized_dir = _materialize_description(_synthetic_description(bundle_dir))
            self.assertEqual(materialized_dir, bundle_dir)

            expected_files = {
                "workload_manifest.json",
                "regions.json",
                "gateways.json",
                "requests.jsonl",
                "sessions.jsonl",
                "timebin_demand.csv",
                "workload_summary.json",
            }
            self.assertEqual(expected_files, {path.name for path in bundle_dir.iterdir() if path.is_file()})

            manifest = _read_json(bundle_dir / "workload_manifest.json")
            self.assertEqual(manifest["schema_version"], "orbitserve.workload")
            self.assertEqual(manifest["source"]["mode"], "synthetic_generation")
            self.assertEqual(manifest["generator"]["seed"], 20260624)
            self.assertEqual(manifest["time"]["time_horizon_s"], 600)
            self.assertEqual(manifest["time"]["time_bin_s"], 60)
            self.assertEqual(manifest["files"]["requests"]["path"], "requests.jsonl")
            self.assertEqual(manifest["files"]["summary"]["path"], "workload_summary.json")

            loaded = _load_bundle(bundle_dir)
            loaded_manifest = _extract_loaded_manifest(loaded)
            self.assertEqual(loaded_manifest.get("schema_version"), "orbitserve.workload")

            self.assertEqual(len(_read_json(bundle_dir / "regions.json")), 2)
            self.assertEqual(len(_read_json(bundle_dir / "gateways.json")), 2)
            self.assertGreater(len(_read_jsonl(bundle_dir / "requests.jsonl")), 0)
            self.assertGreater(len(_read_jsonl(bundle_dir / "sessions.jsonl")), 0)

            timebin_rows = _read_csv(bundle_dir / "timebin_demand.csv")
            self.assertGreater(len(timebin_rows), 0)
            self.assertTrue(TIMEBIN_REQUIRED_FIELDS.issubset(timebin_rows[0]))

    def test_request_rows_are_sorted_canonical_and_gateway_mapped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_dir = _materialize_description(_synthetic_description(Path(tmpdir) / "bundle"))
            regions = {row["region_id"]: row for row in _read_json(bundle_dir / "regions.json")}
            gateways = {row["gateway_id"]: row for row in _read_json(bundle_dir / "gateways.json")}
            requests = _read_jsonl(bundle_dir / "requests.jsonl")

            self.assertEqual(
                requests,
                sorted(requests, key=lambda row: (float(row["arrival_time_s"]), str(row["request_id"]))),
            )
            self.assertEqual(len({row["request_id"] for row in requests}), len(requests))

            for row in requests:
                self.assertTrue(REQUEST_REQUIRED_FIELDS.issubset(row), row)
                self.assertFalse(RUNTIME_FIELDS.intersection(row), row)
                self.assertEqual(row["region_id"], row["user_region_id"])
                self.assertIn(row["user_region_id"], regions)
                self.assertEqual(row["access_mode"], "gateway_sgl")
                self.assertIn(row["gateway_id"], gateways)
                gateway = gateways[row["gateway_id"]]
                self.assertIn(row["user_region_id"], gateway["served_region_ids"])
                self.assertEqual(row["ingress_id"], row["gateway_id"])
                self.assertAlmostEqual(float(row["ingress_lat_deg"]), float(gateway["lat_deg"]))
                self.assertAlmostEqual(float(row["ingress_lon_deg"]), float(gateway["lon_deg"]))
                self.assertGreaterEqual(int(row["prompt_tokens"]), int(row["new_user_tokens"]))
                self.assertEqual(
                    int(row["prompt_tokens"]),
                    int(row["context_tokens_before_turn"]) + int(row["new_user_tokens"]),
                )
                self.assertGreater(int(row["input_bytes"]), 0)
                self.assertGreater(int(row["output_bytes"]), 0)

    def test_stats_slo_burst_session_context_and_seed_are_reproducible(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first_description = _synthetic_description(Path(tmpdir) / "bundle_a", seed=77)
            second_description = copy.deepcopy(first_description)
            second_description["output"]["bundle_dir"] = str(Path(tmpdir) / "bundle_b")

            first_dir = _materialize_description(first_description)
            second_dir = _materialize_description(second_description)
            first_requests = _read_jsonl(first_dir / "requests.jsonl")
            second_requests = _read_jsonl(second_dir / "requests.jsonl")
            self.assertEqual(first_requests, second_requests)

            summary = _read_json(first_dir / "workload_summary.json")
            prompt_values = [int(row["prompt_tokens"]) for row in first_requests]
            output_values = [int(row["output_tokens"]) for row in first_requests]
            self.assertEqual(summary["request_count"], len(first_requests))

            for metric, values in (("prompt_tokens", prompt_values), ("output_tokens", output_values)):
                stats = _stats_block(summary, metric)
                self.assertLessEqual(stats["p50"], stats["p95"])
                self.assertLessEqual(stats["p95"], stats["p99"])
                self.assertEqual(stats["max"], max(values))
                self.assertGreaterEqual(stats["p50"], _nearest_rank(values, 50) * 0.5)
                self.assertLessEqual(stats["p50"], _nearest_rank(values, 50) * 1.5)

            slo_counts = Counter(row["slo_class"] for row in first_requests)
            self.assertEqual(dict(slo_counts), _slo_counts(summary))
            self.assertIn("interactive_5s", slo_counts)
            self.assertIn("relaxed_60s", slo_counts)
            for row in first_requests:
                self.assertIsNotNone(row["slo_class"])
                self.assertIsNotNone(row["e2e_deadline_s"])

            burst_rows = [row for row in first_requests if row["burst_id"] == "eu_burst"]
            non_burst_rows = [row for row in first_requests if row["burst_id"] is None]
            self.assertGreater(len(burst_rows), 0)
            self.assertGreater(len(non_burst_rows), 0)
            for row in burst_rows:
                self.assertEqual(row["user_region_id"], "europe_west")
                self.assertGreater(float(row["burst_multiplier"]), 1.0)
            for row in non_burst_rows:
                self.assertEqual(float(row["burst_multiplier"]), 1.0)

            requests_by_session = defaultdict(list)
            for row in first_requests:
                if row["session_id"] is not None:
                    requests_by_session[row["session_id"]].append(row)
            multi_turn_sessions = [
                sorted(rows, key=lambda row: int(row["turn_index"]))
                for rows in requests_by_session.values()
                if len(rows) > 1
            ]
            self.assertGreater(len(multi_turn_sessions), 0)
            for rows in multi_turn_sessions:
                turn_indices = [int(row["turn_index"]) for row in rows]
                self.assertEqual(turn_indices, list(range(turn_indices[0], turn_indices[0] + len(rows))))
                self.assertEqual(int(rows[0]["context_tokens_before_turn"]), 0)
                previous_after = None
                for row in rows:
                    context_before = int(row["context_tokens_before_turn"])
                    new_user_tokens = int(row["new_user_tokens"])
                    output_tokens = int(row["output_tokens"])
                    context_window = int(row["context_window_tokens"])
                    if previous_after is not None:
                        self.assertEqual(context_before, min(context_window, previous_after))
                    self.assertEqual(int(row["prompt_tokens"]), context_before + new_user_tokens)
                    previous_after = context_before + new_user_tokens + output_tokens


class ExplicitRequestWorkloadTests(unittest.TestCase):
    def test_explicit_requests_jsonl_is_sorted_and_fillable_fields_are_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            request_rows = [
                {
                    "request_id": "req_b",
                    "arrival_time_s": 20.0,
                    "user_region_id": "na_east",
                    "gateway_id": "gw_us_east",
                    "task_type": "conversation",
                    "prompt_tokens": 40,
                    "output_tokens": 12,
                    "slo_class": "interactive_5s",
                    "ttft_deadline_s": 5,
                    "e2e_deadline_s": 30,
                    "session_id": "sess_a",
                    "turn_index": 1,
                    "session_turn_count": 2,
                },
                {
                    "request_id": "req_a",
                    "arrival_time_s": 10.0,
                    "user_region_id": "na_east",
                    "gateway_id": "gw_us_east",
                    "task_type": "conversation",
                    "model_family": "llama_70b_like",
                    "prompt_tokens": 20,
                    "new_user_tokens": 20,
                    "context_tokens_before_turn": 0,
                    "output_tokens": 8,
                    "slo_class": "interactive_5s",
                    "ttft_deadline_s": 5,
                    "e2e_deadline_s": 30,
                    "session_id": "sess_a",
                    "turn_index": 0,
                    "session_turn_count": 2,
                },
            ]
            sessions = [
                {
                    "session_id": "sess_a",
                    "user_region_id": "na_east",
                    "session_start_time_s": 10.0,
                    "session_turn_count": 2,
                    "session_slo_class": "interactive_5s",
                    "session_ttl_s": 300,
                    "context_window_tokens": 4096,
                }
            ]
            requests_path = tmp / "input_requests.jsonl"
            sessions_path = tmp / "input_sessions.jsonl"
            _write_jsonl(requests_path, request_rows)
            _write_jsonl(sessions_path, sessions)

            bundle_dir = _materialize_description(
                _explicit_description(tmp / "bundle", requests_path, sessions_path=sessions_path)
            )
            requests = _read_jsonl(bundle_dir / "requests.jsonl")
            self.assertEqual([row["request_id"] for row in requests], ["req_a", "req_b"])

            for row in requests:
                self.assertTrue(REQUEST_REQUIRED_FIELDS.issubset(row), row)
                self.assertFalse(RUNTIME_FIELDS.intersection(row), row)
                self.assertEqual(row["region_id"], "na_east")
                self.assertEqual(row["access_mode"], "gateway_sgl")
                self.assertEqual(row["ingress_id"], "gw_us_east")
                self.assertEqual(row["model_family"], "llama_70b_like")
                self.assertEqual(row["burst_multiplier"], 1.0)
                self.assertGreater(int(row["input_bytes"]), 0)
                self.assertGreater(int(row["output_bytes"]), 0)

            self.assertEqual(requests[1]["context_tokens_before_turn"], 28)
            self.assertEqual(requests[1]["new_user_tokens"], 12)
            manifest = _read_json(bundle_dir / "workload_manifest.json")
            self.assertEqual(manifest["source"]["mode"], "explicit_requests")
            self.assertTrue(manifest["source"]["inputs"])

    def test_explicit_requests_reject_runtime_fields_before_materialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            requests_path = tmp / "input_requests.jsonl"
            _write_jsonl(
                requests_path,
                [
                    {
                        "request_id": "req_runtime_leak",
                        "arrival_time_s": 1.0,
                        "user_region_id": "na_east",
                        "gateway_id": "gw_us_east",
                        "task_type": "conversation",
                        "model_family": "llama_70b_like",
                        "prompt_tokens": 20,
                        "output_tokens": 8,
                        "slo_class": "interactive_5s",
                        "ttft_deadline_s": 5,
                        "e2e_deadline_s": 30,
                        "served": False,
                    }
                ],
            )

            with self.assertRaises(ValueError):
                _materialize_description(_explicit_description(tmp / "bundle", requests_path))


class TraceReplayWorkloadTests(unittest.TestCase):
    def test_trace_replay_jsonl_uses_base_dir_and_normalizes_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            trace_path = tmp / "trace.jsonl"
            _write_jsonl(
                trace_path,
                [
                    {
                        "trace_id": "trace_b",
                        "timestamp_s": 12.0,
                        "region_id": "europe_west",
                        "prompt": 400,
                        "completion": 100,
                    },
                    {
                        "trace_id": "trace_a",
                        "timestamp_s": 4.0,
                        "region_id": "na_east",
                        "prompt": 200,
                        "completion": 50,
                    },
                ],
            )
            description = {
                "schema_version": "orbitserve.workload_description",
                "workload_id": "unit_trace_replay",
                "source_mode": "trace_replay",
                "output_bundle_schema": "orbitserve.workload",
                "seed": 123,
                "time": {"epoch": "trace_start", "time_horizon_s": 60, "time_bin_s": 30},
                "trace_replay": {
                    "path": "trace.jsonl",
                    "format": "jsonl",
                    "time_column": "timestamp_s",
                    "time_unit": "s",
                    "request_id_column": "trace_id",
                    "region_column": "region_id",
                    "column_mapping": {"prompt_tokens": "prompt", "output_tokens": "completion"},
                },
                "geography": {
                    "mode": "named_gateway_service_regions",
                    "demand_weight_normalization": "require_sum_one",
                    "regions": [
                        {
                            "region_id": "na_east",
                            "region_name": "North America East",
                            "lat_deg": 39.0,
                            "lon_deg": -77.0,
                            "timezone_utc_offset_h": -5,
                            "demand_weight": 0.5,
                            "weight_source": "unit_test",
                        },
                        {
                            "region_id": "europe_west",
                            "region_name": "Europe West",
                            "lat_deg": 50.1,
                            "lon_deg": 8.7,
                            "timezone_utc_offset_h": 1,
                            "demand_weight": 0.5,
                            "weight_source": "unit_test",
                        },
                    ],
                },
                "gateways": {"access_mode": "direct_user_sgl", "mapping_policy": "region_center"},
                "arrival": {"process": "trace_replay", "load_scale": 1.0},
                "tokens": {"model_family_fallback": "llama_70b_like"},
                "slo": {
                    "mix": "balanced",
                    "classes": [
                        {
                            "slo_class": "interactive_5s",
                            "weight": 1.0,
                            "ttft_deadline_s": 5,
                            "e2e_deadline_s": 30,
                        }
                    ],
                },
                "sessions": {"session_mode": "open_loop_turn_arrivals", "context_window_tokens": 4096},
                "validation": {"require_unique_request_id": True},
                "output": {"bundle_dir": "bundle", "overwrite": True},
                "provenance": {"sources": ["unit test trace replay"]},
            }
            description_path = tmp / "description.json"
            description_path.write_text(json.dumps(description, indent=2), encoding="utf-8")

            result = workloads.generate_workload_bundle("description.json", base_dir=tmp)
            requests = _read_jsonl(Path(result.bundle_dir) / "requests.jsonl")
            self.assertEqual([row["request_id"] for row in requests], ["trace_a", "trace_b"])
            self.assertEqual([row["prompt_tokens"] for row in requests], [200, 400])
            self.assertEqual(result.summary["source_mode"], "trace_replay")
            self.assertEqual(result.summary["generation_details"]["trace_rows_read"], 2)


class WorkloadBoundaryTests(unittest.TestCase):
    def test_closed_loop_session_templates_are_explicitly_rejected_until_runtime_support_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            description = _synthetic_description(Path(tmpdir) / "bundle")
            description["sessions"]["session_mode"] = "closed_loop_session_templates"
            with self.assertRaises(NotImplementedError):
                workloads.generate_workload_bundle(description)
