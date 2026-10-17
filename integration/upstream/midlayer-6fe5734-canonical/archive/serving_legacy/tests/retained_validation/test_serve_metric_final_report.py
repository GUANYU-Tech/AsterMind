import json
from pathlib import Path

import pytest

from src.serve_predict.serve_metric_comparison import (
    ComparisonError,
    compare_cell,
    generate_final_report,
)


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return path


def _provenance():
    return {
        "git_commit": "abc123",
        "git_dirty": False,
        "git_status_short": [],
        "git_diff_hash": "empty",
        "hostname": "test-host",
        "cuda_driver_summary": {"nvidia_smi": "unavailable"},
        "python_executable": "/usr/bin/python",
        "vllm_version": "test",
        "midlayer_command": ["runner"],
        "environment_overrides": {"CUDA_VISIBLE_DEVICES": "0,1"},
        "gpu_ids": [0, 1],
        "timestamp_utc": "2026-06-07T00:00:00+00:00",
    }


def _cell_manifest(run_dir, cell_id, **overrides):
    cell_root = run_dir / cell_id
    manifest = {
        "version": 1,
        "cell_id": cell_id,
        "deployment": "colocated",
        "model": "/models/dense-7b",
        "model_type": "dense",
        "output_root": str(cell_root),
        "parallelism": {"tensor_parallel_size": 2},
        "workload": {
            "path": str(cell_root / "workload.json"),
            "hash": f"workload-{cell_id}",
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
        "outputs": {
            "ground_truth": str(cell_root / "ground_truth.json"),
            "prediction": str(cell_root / "prediction.json"),
            "comparison_report": str(cell_root / "comparison_report.json"),
            "comparison_markdown": str(cell_root / "comparison_report.md"),
            "logs": str(cell_root / "logs"),
        },
        "provenance": _provenance(),
    }
    manifest.update(overrides)
    path = cell_root / "command_manifest.json"
    _write_json(path, manifest)
    return path, manifest


def _write_success_cell(run_dir):
    manifest_path, manifest = _cell_manifest(run_dir, "dense_success")
    _write_json(manifest["outputs"]["ground_truth"], {
        "workload_hash": "workload-dense_success",
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
            "ttft_ms": 130.0,
            "tpot_ms": 12.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
        {
            "request_id": "req-000001",
            "ttft_ms": 130.0,
            "tpot_ms": 12.0,
            "e2e_ms": 200.0,
            "start_time_ms": 0.0,
            "end_time_ms": 200.0,
            "num_prefill_tokens": 8,
            "num_decode_tokens": 4,
        },
    ])
    compare_cell(str(manifest_path))
    _write_json(run_dir / "dense_success/status.json", {
        "status": "success",
        "cell_id": "dense_success",
        "logs": str(run_dir / "dense_success/logs"),
        "comparison_report": manifest["outputs"]["comparison_report"],
    })


def _write_blocked_cell(run_dir, cell_id, *, deployment, model_type, failed_stage):
    _, manifest = _cell_manifest(
        run_dir,
        cell_id,
        deployment=deployment,
        model_type=model_type,
        parallelism={"tensor_parallel_size": 2} if deployment == "colocated" else {},
        prefill={"gpus": [0], "tp_size": 1} if deployment == "disaggregated" else {},
        decode={"gpus": [1], "tp_size": 1} if deployment == "disaggregated" else {},
        moe={"dependency_status": "blocked-ready-to-check"} if model_type == "moe" else {},
    )
    _write_json(Path(manifest["output_root"]) / "status.json", {
        "status": "blocked",
        "cell_id": cell_id,
        "failed_stage": failed_stage,
        "dependency_category": "dependency",
        "stderr_summary": "dependency executable not found: vllm",
        "logs": str(Path(manifest["output_root"]) / "logs"),
        "unblock_checklist": ["Install vLLM and retry."],
        "reproducer_commands": {
            "serve": ["vllm", "serve", manifest["model"]],
        },
    })


def _write_run_summary(run_dir, cell_ids):
    _write_json(run_dir / "run_summary.json", {
        "version": 1,
        "mode": "real_run",
        "status": "blocked",
        "run_dir": str(run_dir),
        "cells": [
            {
                "cell_id": cell_id,
                "status": "success" if cell_id == "dense_success" else "blocked",
                "output_root": str(run_dir / cell_id),
                "command_manifest": str(run_dir / cell_id / "command_manifest.json"),
                "workload_hash": f"workload-{cell_id}",
            }
            for cell_id in cell_ids
        ],
    })


def test_final_report_summarizes_success_blockers_and_independent_conclusions(tmp_path):
    run_dir = tmp_path / "run"
    _write_success_cell(run_dir)
    _write_blocked_cell(
        run_dir,
        "moe_blocked",
        deployment="colocated",
        model_type="moe",
        failed_stage="vllm_serve",
    )
    _write_blocked_cell(
        run_dir,
        "pd_blocked",
        deployment="disaggregated",
        model_type="dense",
        failed_stage="pd_decode_serve",
    )
    _write_run_summary(run_dir, ["dense_success", "moe_blocked", "pd_blocked"])

    outputs = generate_final_report(run_dir)
    report = json.loads(Path(outputs["json"]).read_text())
    markdown = Path(outputs["markdown"]).read_text()

    assert report["kind"] == "real_vllm_final_report"
    assert report["measurement_policy"] == "metric errors are observations, not hard pass/fail thresholds"
    assert {cell["cell_id"]: cell["confidence"] for cell in report["cells"]} == {
        "dense_success": "measured",
        "moe_blocked": "blocked-runtime",
        "pd_blocked": "blocked-runtime",
    }
    dense = next(cell for cell in report["cells"] if cell["cell_id"] == "dense_success")
    assert dense["metrics"][0]["absolute_percentage_error"] is not None
    moe = next(cell for cell in report["cells"] if cell["cell_id"] == "moe_blocked")
    assert moe["known_blocker"]["dependency_category"] == "dependency"
    assert moe["known_blocker"]["reproducer_refs"] == ["serve"]
    assert report["conclusions"]["dense_colocated"]["cell_ids"] == ["dense_success"]
    assert report["conclusions"]["moe_colocated"]["cell_ids"] == ["moe_blocked"]
    assert report["conclusions"]["pd_disaggregated"]["cell_ids"] == ["pd_blocked"]
    assert "not generalized to MoE or PD" in report["conclusions"]["dense_colocated"]["summary"]
    assert report["top_modeling_gaps"]["evidence_backed"] is True
    assert report["top_modeling_gaps"]["items"][0]["cell_id"] == "dense_success"
    assert "# Real vLLM Final Modeling Report" in markdown
    assert "metric errors are observations" in markdown
    assert "MoE Colocated" in markdown
    assert "PD Disaggregated" in markdown


def test_final_report_marks_blocker_only_gap_ranking_as_not_evidence_backed(tmp_path):
    run_dir = tmp_path / "run"
    _write_blocked_cell(
        run_dir,
        "moe_blocked",
        deployment="colocated",
        model_type="moe",
        failed_stage="vllm_serve",
    )
    _write_blocked_cell(
        run_dir,
        "pd_blocked",
        deployment="disaggregated",
        model_type="dense",
        failed_stage="pd_decode_serve",
    )
    _write_run_summary(run_dir, ["moe_blocked", "pd_blocked"])

    outputs = generate_final_report(run_dir)
    report = json.loads(Path(outputs["json"]).read_text())

    assert report["top_modeling_gaps"]["evidence_backed"] is False
    assert report["top_modeling_gaps"]["items"] == []
    assert "not yet evidence-backed" in report["top_modeling_gaps"]["message"]
    assert report["top_modeling_gaps"]["current_unblock_items"][0]["cell_id"] == "moe_blocked"


def test_final_report_rejects_success_cell_missing_comparison_report(tmp_path):
    run_dir = tmp_path / "run"
    _, manifest = _cell_manifest(run_dir, "dense_success")
    _write_json(Path(manifest["output_root"]) / "status.json", {
        "status": "success",
        "cell_id": "dense_success",
        "logs": str(Path(manifest["output_root"]) / "logs"),
    })
    _write_run_summary(run_dir, ["dense_success"])

    with pytest.raises(ComparisonError, match="dense_success.*comparison_report.json"):
        generate_final_report(run_dir)


@pytest.mark.parametrize("comparison_payload", [
    {"version": 1, "kind": "real_vllm_comparison_report", "cell_id": "dense_success"},
    {"version": 1, "kind": "real_vllm_comparison_report", "cell_id": "dense_success", "metrics": []},
])
def test_final_report_rejects_success_cell_without_metric_evidence(tmp_path, comparison_payload):
    run_dir = tmp_path / "run"
    _, manifest = _cell_manifest(run_dir, "dense_success")
    _write_json(manifest["outputs"]["comparison_report"], comparison_payload)
    _write_json(Path(manifest["output_root"]) / "status.json", {
        "status": "success",
        "cell_id": "dense_success",
        "logs": str(Path(manifest["output_root"]) / "logs"),
        "comparison_report": manifest["outputs"]["comparison_report"],
    })
    _write_run_summary(run_dir, ["dense_success"])

    with pytest.raises(ComparisonError, match="dense_success.*metric evidence"):
        generate_final_report(run_dir)


def test_final_report_cli_generates_outputs(tmp_path):
    from scripts.serve_predict.eval_serve_metric import main

    run_dir = tmp_path / "run"
    _write_success_cell(run_dir)
    _write_run_summary(run_dir, ["dense_success"])

    assert main(["final", "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "final/final_report.json").exists()
    assert (run_dir / "final/final_report.md").exists()


def test_final_report_rejects_manifests_without_dirty_provenance(tmp_path):
    run_dir = tmp_path / "run"
    _, manifest = _cell_manifest(run_dir, "missing_dirty")
    manifest["provenance"].pop("git_dirty")
    _write_json(run_dir / "missing_dirty/command_manifest.json", manifest)
    _write_json(run_dir / "missing_dirty/status.json", {
        "status": "blocked",
        "cell_id": "missing_dirty",
        "dependency_category": "dependency",
    })
    _write_run_summary(run_dir, ["missing_dirty"])

    with pytest.raises(ComparisonError, match="git_dirty"):
        generate_final_report(run_dir)
