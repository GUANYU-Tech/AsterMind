import csv
import json
from pathlib import Path

import pytest

from evaluation.serving.experiments.experiment_plan import (
    vllm_bench_arrival_process,
)
from evaluation.serving.metrics.serving_metric_comparison import (
    ComparisonError,
    _component_notes,
    aggregate_run_directory,
    compare_cell,
    generate_final_report,
)
from src.serving.backend.vllm_scheduler_process import (
    VLLM_SCHEDULER_SOURCE_COMMIT,
)


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path


def _cell_manifest(tmp_path, *, cell_id="dense_tp2", **overrides):
    cell_root = tmp_path / "run" / cell_id
    manifest = {
        "version": 1,
        "cell_id": cell_id,
        "deployment": "colocated",
        "model": "/models/dense-7b",
        "model_type": "dense",
        "output_root": str(cell_root),
        "parallelism": {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
        },
        "workload": {
            "path": str(cell_root / "workload.json"),
            "hash": "workload-a",
            "num_requests": 2,
            "requested_num_requests": 2,
        },
        "token_controls": {
            "min_tokens": 4,
            "max_tokens": 4,
            "ignore_eos": True,
        },
        "benchmark": {
            "request_rate": 2.0,
            "metric_percentiles": [50, 90, 99],
        },
        "mtp_scenario": _disabled_mtp_scenario(),
        "commands": {
            "legacy_prediction": {"kind": "legacy_prediction"},
        },
        "outputs": {
            "status": str(cell_root / "status.json"),
            "ground_truth": str(cell_root / "ground_truth.json"),
            "prediction": str(cell_root / "prediction.json"),
            "comparison_report": str(cell_root / "comparison_report.json"),
            "comparison_markdown": str(cell_root / "comparison_report.md"),
            "logs": str(cell_root / "logs"),
        },
        "provenance": {
            "git_commit": "abc123",
            "hostname": "test-host",
            "cuda_driver_summary": {"nvidia_smi": "unavailable"},
            "python_executable": "/usr/bin/python",
            "vllm_version": "test",
            "midlayer_command": ["runner"],
            "environment_overrides": {"CUDA_VISIBLE_DEVICES": "0,1"},
            "gpu_ids": [0, 1],
            "timestamp_utc": "2026-06-07T00:00:00+00:00",
        },
    }
    manifest.update(overrides)
    path = cell_root / "command_manifest.json"
    _write_json(path, manifest)
    return path, manifest


def _metric(report, name, statistic):
    matches = [
        row for row in report["metrics"]
        if row["name"] == name and row["statistic"] == statistic
    ]
    assert len(matches) == 1
    return matches[0]


def _vllm_prediction_runtime():
    return {
        "version": "0.11.0",
        "source_commit": VLLM_SCHEDULER_SOURCE_COMMIT,
        "requested_chunked_prefill": True,
        "effective_chunked_prefill": True,
        "effective_prefix_caching": False,
        "prefix_oracle_mode": "trace_length_private_nonshared_v1",
        "max_num_batched_tokens": 6,
        "max_num_seqs": 2,
        "max_model_len": 8192,
        "kv_block_size_tokens": 16,
        "num_kv_blocks": 128,
        "scheduler_policy": "fcfs",
        "num_lookahead_slots": 0,
        "attention_kv_layout": "paged_varlen",
        "speculative_method": "disabled",
        "mtp_num_speculative_tokens": 0,
        "mtp_acceptance_policy": "disabled",
        "mtp_fixed_accepted_tokens": 0,
        "mtp_acceptance_source_id": "disabled",
        "mtp_padded_drafter_batch": False,
        "mtp_draft_model_type": "disabled",
        "mtp_num_mtp_layers": 0,
        "mtp_draft_max_model_len": 0,
    }


def _disabled_mtp_scenario():
    return {
        "num_speculative_tokens": 0,
        "acceptance_policy": "disabled",
        "fixed_accepted_tokens": 0,
        "acceptance_source_id": "disabled",
        "acceptance_replay": None,
        "validation_status": "disabled",
    }


def _vllm_arrival_contract():
    return vllm_bench_arrival_process(request_rate=2.0, seed=7)


def _canonical_vllm_commands():
    arrival = _vllm_arrival_contract()
    identity = {
        "workload_hash": "workload-a",
        "request_order_sha256": "request-order-hash",
        "arrival_contract_sha256": "arrival-contract-hash",
        "scheduler_contract_sha256": "scheduler-contract-hash",
    }
    return {
        "vllm_benchmark": {
            "kind": "real_vllm_benchmark",
            "arrival_process": dict(arrival),
            "validation_identity": identity,
            "vllm_consumed_request_ids": ["req-000000", "req-000001"],
            "mtp_scenario": _disabled_mtp_scenario(),
        },
        "colocated_midlayer_simulation": {
            "kind": "colocated_midlayer_simulation",
            "arrival_process": dict(arrival),
            "mtp_scenario": _disabled_mtp_scenario(),
        },
    }


def _canonical_vllm_prediction(runtime):
    return {
        "backend": "vllm",
        "version": 2,
        "deployment": "colocated",
        "workload_hash": "workload-a",
        "scheduler_runtime": {"main": runtime},
        "mtp_scenario": _disabled_mtp_scenario(),
        "arrival_process": _vllm_arrival_contract(),
        "request_arrivals_s": [
            {
                "request_id": "req-000000",
                "arrival_time_s": 0.04982432776885802,
            },
            {"request_id": "req-000001", "arrival_time_s": 1.0},
        ],
        "metrics": [],
    }


def test_compare_rejects_vllm_prediction_without_effective_runtime(tmp_path):
    manifest_path, manifest = _cell_manifest(
        tmp_path,
        parallelism={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 6,
            "block_size": 16,
            "num_gpu_blocks_override": 128,
        },
        commands=_canonical_vllm_commands(),
    )
    _write_json(manifest["outputs"]["ground_truth"], {})

    for prediction, message in (
        ({"backend": "vllm", "version": 1, "metrics": []}, "version"),
        ({"backend": "vllm", "version": 2,
          "deployment": "colocated", "workload_hash": "workload-a",
          "mtp_scenario": _disabled_mtp_scenario(),
          "metrics": []},
         "missing scheduler_runtime"),
        ({"backend": "vllm", "version": 2,
          "deployment": "colocated", "workload_hash": "workload-a",
          "mtp_scenario": _disabled_mtp_scenario(),
          "scheduler_runtime": {},
          "metrics": []}, "non-empty"),
    ):
        _write_json(manifest["outputs"]["prediction"], prediction)
        with pytest.raises(ComparisonError, match=message):
            compare_cell(str(manifest_path))


def test_compare_rejects_vllm_runtime_capacity_mismatch(tmp_path):
    manifest_path, manifest = _cell_manifest(
        tmp_path,
        parallelism={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 6,
            "block_size": 16,
            "num_gpu_blocks_override": 128,
        },
        commands=_canonical_vllm_commands(),
    )
    _write_json(manifest["outputs"]["ground_truth"], {})
    runtime = _vllm_prediction_runtime()
    runtime["max_num_seqs"] = 1
    _write_json(
        manifest["outputs"]["prediction"],
        _canonical_vllm_prediction(runtime),
    )

    with pytest.raises(ComparisonError, match="max_num_seqs"):
        compare_cell(str(manifest_path))


def test_compare_rejects_projected_mtp_against_real_benchmark(tmp_path):
    manifest_path, manifest = _cell_manifest(
        tmp_path,
        parallelism={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 6,
            "block_size": 16,
            "num_gpu_blocks_override": 128,
        },
        commands=_canonical_vllm_commands(),
    )
    _write_json(manifest["outputs"]["ground_truth"], {})
    prediction = _canonical_vllm_prediction(_vllm_prediction_runtime())
    prediction["mtp_scenario"] = {
        "num_speculative_tokens": 2,
        "acceptance_policy": "fixed_count",
        "fixed_accepted_tokens": 1,
        "acceptance_source_id": "projected-fixed-one",
        "acceptance_replay": None,
        "validation_status": "projected_explicit_assumption",
    }
    _write_json(manifest["outputs"]["prediction"], prediction)

    with pytest.raises(ComparisonError, match="cannot be compared"):
        compare_cell(str(manifest_path))


def test_canonical_comparison_rejects_unbound_real_ground_truth(tmp_path):
    manifest_path, manifest = _cell_manifest(
        tmp_path,
        parallelism={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 6,
            "block_size": 16,
            "num_gpu_blocks_override": 128,
        },
        commands=_canonical_vllm_commands(),
    )
    _write_json(
        manifest["outputs"]["prediction"],
        _canonical_vllm_prediction(_vllm_prediction_runtime()),
    )
    _write_json(manifest["outputs"]["ground_truth"], {})

    with pytest.raises(ComparisonError, match="declared invocation identity"):
        compare_cell(str(manifest_path))


@pytest.mark.parametrize(
    "prediction,message",
    [
        ([], "must be a JSON object"),
        ({"metrics": []}, "backend must be vllm"),
        ({"backend": "conformance", "metrics": []}, "backend must be vllm"),
    ],
)
def test_canonical_manifest_rejects_prediction_producer_bypass(
        tmp_path, prediction, message):
    manifest_path, manifest = _cell_manifest(
        tmp_path,
        parallelism={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 6,
            "block_size": 16,
            "num_gpu_blocks_override": 128,
        },
        commands=_canonical_vllm_commands(),
    )
    _write_json(manifest["outputs"]["ground_truth"], {})
    _write_json(manifest["outputs"]["prediction"], prediction)

    with pytest.raises(ComparisonError, match=message):
        compare_cell(str(manifest_path))


def test_component_notes_require_disaggregated_connector():
    with pytest.raises(KeyError, match="kv_connector"):
        _component_notes(
            {"deployment": "disaggregated", "model_type": "dense"},
            {"status": "success"},
        )


def test_component_notes_accept_executable_moe_settings():
    notes = _component_notes(
        {
            "deployment": "colocated",
            "model_type": "moe",
            "moe": {
                "all2all_backend": "deepep_low_latency",
                "eplb_enabled": False,
            },
        },
        {"status": "success"},
    )
    assert notes == [
        "MoE colocated serving is evaluated independently from dense colocated serving."
    ]


def test_compare_cell_writes_metric_errors_and_markdown(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path)
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "completed": 2,
        "total_input_tokens": 16,
        "total_output_tokens": 8,
        "request_throughput": 10.0,
        "output_throughput": 40.0,
        "total_token_throughput": 120.0,
        "mean_ttft_ms": 100.0,
        "median_ttft_ms": 95.0,
        "p90_ttft_ms": 110.0,
        "mean_tpot_ms": 10.0,
        "median_tpot_ms": 9.5,
        "p90_tpot_ms": 11.0,
    })
    _write_json(manifest["outputs"]["prediction"], [
        {
            "request_id": "req-000000",
            "ttft_ms": 110.0,
            "tpot_ms": 9.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
        {
            "request_id": "req-000001",
            "ttft_ms": 100.0,
            "tpot_ms": 11.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
    ])

    report = compare_cell(str(manifest_path))

    ttft = _metric(report, "ttft_ms", "mean")
    assert ttft["observed"] == 100.0
    assert ttft["predicted"] == 105.0
    assert ttft["signed_error"] == 5.0
    assert ttft["absolute_error"] == 5.0
    assert ttft["absolute_percentage_error"] == 5.0
    assert _metric(report, "request_throughput_rps", "value")["predicted"] == 10.0
    assert _metric(report, "output_token_throughput_tps", "value")["predicted"] == 40.0
    assert _metric(report, "total_token_throughput_tps", "value")["predicted"] == 120.0
    assert report["provenance"]["git_commit"] == "abc123"
    assert report["version"] == 2
    assert report["validation_scope"]["feature_status"] == "unscoped"

    unavailable = _metric(report, "ttft_ms", "p99")
    assert unavailable["observed_available"] is False
    assert unavailable["availability_reason"] == "observed percentile unavailable"
    assert Path(manifest["outputs"]["comparison_report"]).exists()
    assert Path(manifest["outputs"]["comparison_markdown"]).read_text().startswith("# Comparison Report")


def test_compare_cli_cell_subcommand_writes_report(tmp_path):
    from evaluation.serving.metrics.metric_evaluation import main

    manifest_path, manifest = _cell_manifest(tmp_path)
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "completed": 2,
        "total_input_tokens": 16,
        "total_output_tokens": 8,
        "request_throughput": 10.0,
        "output_throughput": 40.0,
        "total_token_throughput": 120.0,
        "mean_ttft_ms": 100.0,
        "mean_tpot_ms": 10.0,
    })
    _write_json(manifest["outputs"]["prediction"], [
        {
            "request_id": "req-000000",
            "ttft_ms": 100.0,
            "tpot_ms": 10.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
        {
            "request_id": "req-000001",
            "ttft_ms": 100.0,
            "tpot_ms": 10.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
    ])

    assert main(["cell", "--command-manifest", str(manifest_path)]) == 0
    assert Path(manifest["outputs"]["comparison_report"]).exists()


def test_compare_cell_converts_explicit_ns_latency_fields(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path)
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "completed": 2,
        "total_input_tokens": 16,
        "total_output_tokens": 8,
        "request_throughput": 10.0,
        "output_throughput": 40.0,
        "total_token_throughput": 120.0,
        "mean_ttft_ns": 100_000_000,
        "mean_tpot_ns": 10_000_000,
    })
    _write_json(manifest["outputs"]["prediction"], [
        {
            "request_id": "req-000000",
            "ttft_ms": 100.0,
            "tpot_ms": 10.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
        {
            "request_id": "req-000001",
            "ttft_ms": 100.0,
            "tpot_ms": 10.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
    ])

    report = compare_cell(str(manifest_path))

    assert _metric(report, "ttft_ms", "mean")["observed"] == 100.0
    assert _metric(report, "ttft_ms", "mean")["unit_conversions"]["observed"] == "ns_to_ms"


def test_compare_cell_rejects_identity_and_request_mismatches(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path)
    _write_json(manifest["outputs"]["prediction"], [])

    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "other",
        "completed": 2,
        "total_output_tokens": 8,
    })
    with pytest.raises(ComparisonError, match="workload hash"):
        compare_cell(str(manifest_path))

    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "completed": 1,
        "total_output_tokens": 4,
    })
    with pytest.raises(ComparisonError, match="completed requests"):
        compare_cell(str(manifest_path))


def test_compare_cell_rejects_metric_insufficient_predictions_without_writing_report(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path)
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "completed": 2,
        "total_input_tokens": 16,
        "total_output_tokens": 8,
        "request_throughput": 10.0,
        "output_throughput": 40.0,
        "total_token_throughput": 120.0,
        "mean_ttft_ms": 100.0,
        "mean_tpot_ms": 10.0,
    })

    cases = [
        ([], "prediction request count"),
        ([
            {"request_id": "req-000000", "e2e_ms": 200.0, "start_time_ms": 0.0, "end_time_ms": 200.0},
            {"request_id": "req-000001", "e2e_ms": 200.0, "start_time_ms": 0.0, "end_time_ms": 200.0},
        ], "prediction token counts are incomplete"),
        ([
            {"request_id": "req-000000", "ttft_ms": 100.0, "tpot_ms": 10.0, "e2e_ms": 200.0,
             "num_prefill_tokens": 8, "num_decode_tokens": 4},
            {"request_id": "req-000001", "ttft_ms": 100.0, "tpot_ms": 10.0, "e2e_ms": 200.0,
             "num_prefill_tokens": 8, "num_decode_tokens": 4},
        ], "missing predicted primary metrics"),
    ]
    for prediction, message in cases:
        _write_json(manifest["outputs"]["prediction"], prediction)
        with pytest.raises(ComparisonError, match=message):
            compare_cell(str(manifest_path))
        assert not Path(manifest["outputs"]["comparison_report"]).exists()


def test_compare_cell_rejects_prediction_row_count_mismatch(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path)
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "completed": 2,
        "total_input_tokens": 16,
        "total_output_tokens": 8,
        "request_throughput": 10.0,
        "output_throughput": 40.0,
        "total_token_throughput": 120.0,
        "mean_ttft_ms": 100.0,
        "mean_tpot_ms": 10.0,
    })
    _write_json(manifest["outputs"]["prediction"], [
        {
            "request_id": "req-000000",
            "ttft_ms": 100.0,
            "tpot_ms": 10.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
    ])

    with pytest.raises(ComparisonError, match="prediction request count"):
        compare_cell(str(manifest_path))


def test_compare_cell_rejects_implicit_or_malformed_prediction_schemas(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path)
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "completed": 2,
        "total_output_tokens": 8,
    })

    _write_json(manifest["outputs"]["prediction"], {
        "workload_hash": "workload-a",
        "request_throughput_rps": 10.0,
    })
    with pytest.raises(ComparisonError, match="explicit metrics"):
        compare_cell(str(manifest_path))

    _write_json(manifest["outputs"]["prediction"], {
        "workload_hash": "workload-a",
        "metrics": {
            "request_throughput_per_s": "10.0",
            "output_token_throughput_per_s": 40.0,
            "token_throughput_per_s": 120.0,
            "requests": {},
        },
    })
    with pytest.raises(ComparisonError, match="must be a number"):
        compare_cell(str(manifest_path))


def test_compare_cell_rejects_conflicting_identity_aliases(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path)
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "workload": {"hash": "workload-b"},
        "completed": 2,
        "total_output_tokens": 8,
    })
    _write_json(manifest["outputs"]["prediction"], [])

    with pytest.raises(ComparisonError, match="conflicting workload hashes"):
        compare_cell(str(manifest_path))


def test_compare_cell_rejects_unsupported_explicit_units(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path)
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "completed": 2,
        "total_output_tokens": 8,
        "metrics": {
            "ttft_ms": {"mean": {"value": 0.1, "unit": "s"}},
        },
    })
    _write_json(manifest["outputs"]["prediction"], [])

    with pytest.raises(ComparisonError, match="unsupported unit"):
        compare_cell(str(manifest_path))


def test_pd_comparison_marks_throughput_informational_and_keeps_kv_diagnostics(tmp_path):
    manifest_path, manifest = _cell_manifest(
        tmp_path,
        cell_id="dense_pd",
        deployment="disaggregated",
        parallelism={},
        prefill={
            "gpus": [0, 1],
            "tp_size": 2,
            "engine_args": {"tensor_parallel_size": 2},
        },
        decode={
            "gpus": [2],
            "tp_size": 1,
            "engine_args": {"tensor_parallel_size": 1},
        },
        comparison={
            "informational_metrics": ["throughput"],
            "throughput_pass_fail": False,
        },
    )
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "completed": 2,
        "total_input_tokens": 16,
        "total_output_tokens": 8,
        "request_throughput": 5.0,
        "output_throughput": 20.0,
        "total_token_throughput": 60.0,
        "mean_ttft_ms": 120.0,
        "mean_tpot_ms": 14.0,
    })
    _write_json(manifest["outputs"]["prediction"], {
        "kind": "disaggregated_prediction",
        "workload_hash": "workload-a",
        "kv_transfer": {"observed_kv_bytes": 4096},
        "metrics": [
            {
                "request_id": "req-000000",
                "ttft_disagg_ms": 130.0,
                "tpot_disagg_ms": 15.0,
                "e2e_disagg_ms": 400.0,
                "prefill_time_ms": 100.0,
                "kv_transfer_time_ms": 10.0,
                "decode_time_ms": 60.0,
                "num_prefill_tokens": 8,
                "num_decode_tokens": 4,
            },
            {
                "request_id": "req-000001",
                "ttft_disagg_ms": 110.0,
                "tpot_disagg_ms": 13.0,
                "e2e_disagg_ms": 400.0,
                "prefill_time_ms": 90.0,
                "kv_transfer_time_ms": 8.0,
                "decode_time_ms": 52.0,
                "num_prefill_tokens": 8,
                "num_decode_tokens": 4,
            },
        ],
    })

    report = compare_cell(str(manifest_path))

    assert report["diagnostic_mode"] == "pd"
    assert _metric(report, "request_throughput_rps", "value")["informational"] is True
    assert _metric(report, "request_throughput_rps", "value")["predicted_available"] is False
    kv = _metric(report, "kv_transfer_time_ms", "mean")
    assert kv["predicted"] == 9.0
    assert kv["diagnostic_family"] == "pd"


def test_aggregate_run_directory_writes_grouped_outputs_and_top_gaps(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path)
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "completed": 2,
        "total_input_tokens": 16,
        "total_output_tokens": 8,
        "request_throughput": 10.0,
        "output_throughput": 40.0,
        "total_token_throughput": 120.0,
        "mean_ttft_ms": 100.0,
        "mean_tpot_ms": 10.0,
    })
    _write_json(manifest["outputs"]["prediction"], [
        {
            "request_id": "req-000000",
            "ttft_ms": 120.0,
            "tpot_ms": 12.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
        {
            "request_id": "req-000001",
            "ttft_ms": 120.0,
            "tpot_ms": 12.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
    ])
    compare_cell(str(manifest_path))

    outputs = aggregate_run_directory(str(tmp_path / "run"))

    assert Path(outputs["csv"]).exists()
    assert Path(outputs["jsonl"]).exists()
    assert Path(outputs["markdown"]).exists()
    rows = list(csv.DictReader(Path(outputs["csv"]).open()))
    assert rows[0]["cell_id"] == "dense_tp2"
    assert rows[0]["group_key"].startswith("/models/dense-7b|colocated|")
    assert json.loads(Path(outputs["json"]).read_text())["top_modeling_gaps"][0]["name"] in {
        "ttft_ms",
        "tpot_ms",
    }


def test_aggregate_run_directory_rejects_reports_without_provenance(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path, provenance=None)
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-a",
        "completed": 2,
        "total_input_tokens": 16,
        "total_output_tokens": 8,
        "request_throughput": 10.0,
        "output_throughput": 40.0,
        "total_token_throughput": 120.0,
        "mean_ttft_ms": 100.0,
        "mean_tpot_ms": 10.0,
    })
    _write_json(manifest["outputs"]["prediction"], [
        {
            "request_id": "req-000000",
            "ttft_ms": 100.0,
            "tpot_ms": 10.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
        {
            "request_id": "req-000001",
            "ttft_ms": 100.0,
            "tpot_ms": 10.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
    ])
    compare_cell(str(manifest_path))

    with pytest.raises(ComparisonError, match="provenance"):
        aggregate_run_directory(str(tmp_path / "run"))


def test_aggregate_run_directory_groups_distinct_pd_parallelism(tmp_path):
    for cell_id, prefill_tp, decode_tp in (
        ("pd_a", 2, 1),
        ("pd_b", 1, 2),
    ):
        manifest_path, manifest = _cell_manifest(
            tmp_path,
            cell_id=cell_id,
            deployment="disaggregated",
            model="/models/dense-7b",
            parallelism={},
            prefill={
                "gpus": [0, 1][:prefill_tp],
                "tp_size": prefill_tp,
                "engine_args": {"tensor_parallel_size": prefill_tp},
            },
            decode={
                "gpus": [2, 3][:decode_tp],
                "tp_size": decode_tp,
                "engine_args": {"tensor_parallel_size": decode_tp},
            },
            comparison={
                "informational_metrics": ["throughput"],
                "throughput_pass_fail": False,
            },
        )
        _write_json(manifest["outputs"]["ground_truth"], {
            "workload_hash": "workload-a",
            "completed": 2,
            "total_input_tokens": 16,
            "total_output_tokens": 8,
            "request_throughput": 5.0,
            "output_throughput": 20.0,
            "total_token_throughput": 60.0,
            "mean_ttft_ms": 120.0,
            "mean_tpot_ms": 14.0,
        })
        _write_json(manifest["outputs"]["prediction"], {
            "kind": "disaggregated_prediction",
            "workload_hash": "workload-a",
            "metrics": [
                {
                    "request_id": "req-000000",
                    "ttft_disagg_ms": 130.0,
                    "tpot_disagg_ms": 15.0,
                    "e2e_disagg_ms": 400.0,
                    "prefill_time_ms": 100.0,
                    "kv_transfer_time_ms": 10.0,
                    "decode_time_ms": 60.0,
                    "num_prefill_tokens": 8,
                    "num_decode_tokens": 4,
                },
                {
                    "request_id": "req-000001",
                    "ttft_disagg_ms": 110.0,
                    "tpot_disagg_ms": 13.0,
                    "e2e_disagg_ms": 400.0,
                    "prefill_time_ms": 90.0,
                    "kv_transfer_time_ms": 8.0,
                    "decode_time_ms": 52.0,
                    "num_prefill_tokens": 8,
                    "num_decode_tokens": 4,
                },
            ],
        })
        compare_cell(str(manifest_path))

    outputs = aggregate_run_directory(str(tmp_path / "run"))
    aggregate = json.loads(Path(outputs["json"]).read_text())
    pd_groups = [
        group for group in aggregate["groups"]
        if group["deployment"] == "disaggregated"
    ]
    assert len(pd_groups) == 2
    assert len({group["parallelism_summary"] for group in pd_groups}) == 2


def test_final_report_rejects_conflicting_persisted_statuses(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path)
    manifest["status"] = "success"
    manifest["provenance"].update({
        "git_dirty": False,
        "git_status_short": "",
        "git_diff_hash": "clean",
    })
    _write_json(manifest_path, manifest)
    _write_json(manifest["outputs"]["status"], {
        "status": "failed",
    })
    _write_json(tmp_path / "run" / "run_summary.json", {
        "cells": [{
            "cell_id": manifest["cell_id"],
            "status": "success",
            "command_manifest": str(manifest_path),
        }],
    })

    with pytest.raises(ComparisonError, match="conflicting experiment statuses"):
        generate_final_report(tmp_path / "run")


def test_final_report_requires_persisted_status_artifact(tmp_path):
    manifest_path, manifest = _cell_manifest(tmp_path)
    manifest["status"] = "failed"
    manifest["provenance"].update({
        "git_dirty": False,
        "git_status_short": "",
        "git_diff_hash": "clean",
    })
    _write_json(manifest_path, manifest)
    _write_json(tmp_path / "run" / "run_summary.json", {
        "cells": [{
            "cell_id": manifest["cell_id"],
            "status": "failed",
            "command_manifest": str(manifest_path),
        }],
    })

    with pytest.raises(ComparisonError, match="missing status artifact"):
        generate_final_report(tmp_path / "run")
