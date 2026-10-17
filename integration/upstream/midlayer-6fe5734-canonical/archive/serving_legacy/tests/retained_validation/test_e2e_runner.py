import json
from pathlib import Path

import pytest

from scripts.serve_predict.run_experiments import run_experiments
from src.serve_predict.experiment_manifest import (
    ManifestValidationError,
    load_experiment_manifest,
)


def _write_manifest(tmp_path, *cells):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"version": 1, "cells": list(cells)}))
    return path


def _argv_has_script(argv, script_name):
    """Check if any argv element ends with the given script filename."""
    return any(str(arg).endswith(script_name) for arg in argv)


def _base_cell(**overrides):
    cell = {
        "id": "dense_tp2",
        "model": "/models/dense-7b",
        "deployment": "colocated",
        "output_root": "serve_exp/dense_tp2",
        "gpus": [0, 1],
        "parallelism": {
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
        },
        "workload": {
            "name": "long-prefill",
            "num_requests": 4,
            "prompt_len": 8,
            "output_len": 8,
            "seed": 123,
        },
        "sampling": {
            "temperature": 0.0,
            "min_tokens": 8,
            "max_tokens": 8,
            "ignore_eos": True,
        },
        "benchmark": {
            "request_rate": 2.0,
            "metric_percentiles": [50, 90, 99],
        },
        "prediction": {
            "hw_config": "configs/hardware/nvidia_a800.yaml",
        },
        "metrics": ["throughput", "ttft", "tpot"],
    }
    cell.update(overrides)
    return cell


def _pd_cell(**overrides):
    cell = _base_cell(
        id="dense_pd",
        deployment="disaggregated",
        output_root="serve_exp/dense_pd",
        prediction={
            "prefill_hw_config": "configs/hardware/nvidia_a800.yaml",
            "decode_hw_config": "configs/hardware/nvidia_a800.yaml",
        },
        prefill={
            "gpus": [0, 1],
            "engine_args": {"tensor_parallel_size": 2},
            "tp_size": 2,
            "kv_port": 14579,
            "http_port": 15579,
        },
        decode={
            "gpus": [2],
            "engine_args": {"tensor_parallel_size": 1},
            "tp_size": 1,
            "kv_port": 14582,
            "http_port": 15580,
        },
        router={
            "host": "127.0.0.1",
            "http_port": 15590,
            "proxy_port": 30001,
        },
        kv_connector="P2pNcclConnector",
    )
    cell.pop("gpus")
    cell.pop("parallelism")
    cell.update(overrides)
    return cell


def _forbidden_run(*args, **kwargs):
    raise AssertionError("dry-run must not start subprocesses")


def _load(path):
    return json.loads(Path(path).read_text())


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True))


def test_minimal_real_vllm_e2e_manifest_dry_run_materializes_required_cells(tmp_path):
    manifest_path = Path("configs/serving/experiments/real_vllm_e2e_minimal.json")
    project_root = Path.cwd()

    manifest = load_experiment_manifest(manifest_path, project_root=project_root)

    assert [cell.id for cell in manifest.cells] == [
        "dense_colocated_tp2",
        "moe_colocated_tp2_ep_check",
        "dense_pd_prefill1_decode1",
    ]
    dense, moe, pd = manifest.cells
    assert dense.model == Path("/share_data/public_models/Qwen3-0.6B")
    assert dense.gpus == [0, 1]
    assert dense.parallelism["tensor_parallel_size"] == 2
    assert moe.model_type == "moe"
    assert moe.model == Path("/share_data/public_models/Qwen3-30B-A3B")
    assert moe.gpus == [2, 3]
    assert moe.parallelism["enable_expert_parallel"] is True
    assert moe.moe["dependency_status"] == "blocked-ready-to-check"
    assert pd.deployment == "disaggregated"
    assert pd.prefill["gpus"] == [0]
    assert pd.decode["gpus"] == [1]
    assert pd.prefill["kv_port"] == 29601
    assert pd.decode["kv_port"] == 29611

    run_dir = tmp_path / "real-vllm-e2e-minimal-dry-run"
    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(project_root),
            "--run-dir", str(run_dir),
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )

    assert summary["status"] == "dry_run"
    assert [cell["cell_id"] for cell in summary["cells"]] == [
        "dense_colocated_tp2",
        "moe_colocated_tp2_ep_check",
        "dense_pd_prefill1_decode1",
    ]

    dense_manifest = _load(run_dir / "dense_colocated_tp2/command_manifest.json")
    moe_manifest = _load(run_dir / "moe_colocated_tp2_ep_check/command_manifest.json")
    pd_manifest = _load(run_dir / "dense_pd_prefill1_decode1/command_manifest.json")
    for cell_manifest in (dense_manifest, moe_manifest, pd_manifest):
        _assert_real_manifest_dry_run_contract(cell_manifest, run_dir)

    dense_serve = dense_manifest["commands"]["vllm_serve"]["argv"]
    assert dense_manifest["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert dense_serve == [
        "vllm",
        "serve",
        "/share_data/public_models/Qwen3-0.6B",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "2",
        "--pipeline-parallel-size",
        "1",
        "--data-parallel-size",
        "1",
    ]
    dense_prediction = dense_manifest["commands"]["colocated_midlayer_prediction"]
    assert dense_prediction["outputs"]["prediction"] == dense_manifest["outputs"]["prediction"]
    assert dense_prediction["argv"][dense_prediction["argv"].index("--tp-size") + 1] == "2"

    moe_serve = moe_manifest["commands"]["vllm_serve"]["argv"]
    assert moe_manifest["model_type"] == "moe"
    assert moe_manifest["environment"]["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert moe_manifest["moe"]["dependency_status"] == "blocked-ready-to-check"
    assert "--enable-expert-parallel" in moe_serve
    assert moe_serve[moe_serve.index("--tensor-parallel-size") + 1] == "2"
    assert moe_serve[moe_serve.index("--data-parallel-size") + 1] == "1"

    pd_emulation = pd_manifest["commands"]["disaggregated_midlayer_emulation"]
    pd_prediction = pd_manifest["commands"]["disaggregated_midlayer_prediction"]
    assert pd_manifest["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert pd_manifest["prefill"]["kv_port"] == 29601
    assert pd_manifest["decode"]["kv_port"] == 29611
    assert pd_manifest["prefill"]["trace_output"] == pd_manifest["traces"]["prefill_trace"]
    assert pd_manifest["decode"]["trace_output"] == pd_manifest["traces"]["decode_trace"]
    assert pd_manifest["prefill"]["manifest_trace_output"].startswith(
        str(project_root / "serve_exp/real_vllm_e2e_minimal")
    )
    assert pd_manifest["decode"]["manifest_trace_output"].startswith(
        str(project_root / "serve_exp/real_vllm_e2e_minimal")
    )
    assert pd_emulation["argv"][pd_emulation["argv"].index("--prefill-kv-port") + 1] == "29601"
    assert pd_emulation["argv"][pd_emulation["argv"].index("--decode-kv-port") + 1] == "29611"
    assert pd_emulation["prefill_kv_port"] == 29601
    assert pd_emulation["decode_kv_port"] == 29611
    assert pd_prediction["argv"][pd_prediction["argv"].index("--output") + 1] == pd_manifest["outputs"]["prediction"]
    assert pd_prediction["outputs"]["prediction"] == pd_manifest["outputs"]["prediction"]
    assert pd_manifest["commands"]["pd_comparison"]["diagnostic_mode"] == "pd"
    assert pd_manifest["comparison"]["throughput_pass_fail"] is False


def _assert_real_manifest_dry_run_contract(
    cell_manifest: dict,
    run_dir: Path,
) -> None:
    cell_root = run_dir / cell_manifest["cell_id"]
    assert cell_manifest["status"] == "dry_run"
    assert cell_manifest["output_root"] == str(cell_root)
    assert cell_manifest["manifest_output_root"].startswith(str(Path.cwd() / "serve_exp/real_vllm_e2e_minimal"))
    assert cell_manifest["command_manifest_path"] == str(cell_root / "command_manifest.json")
    assert cell_manifest["outputs"]["command_manifest"] == str(cell_root / "command_manifest.json")
    assert cell_manifest["outputs"]["ground_truth"] == str(cell_root / "ground_truth.json")
    assert cell_manifest["outputs"]["prediction"] == str(cell_root / "prediction.json")
    assert cell_manifest["outputs"]["comparison_report"] == str(cell_root / "comparison_report.json")
    assert cell_manifest["outputs"]["comparison_markdown"] == str(cell_root / "comparison_report.md")
    assert cell_manifest["outputs"]["logs"] == str(cell_root / "logs")
    assert cell_manifest["workload"]["path"] == str(cell_root / "workload.json")
    assert Path(cell_manifest["workload"]["path"]).exists()
    assert cell_manifest["workload"]["hash"]
    assert cell_manifest["sampling"] == {
        "ignore_eos": True,
        "max_tokens": 32,
        "min_tokens": 32,
        "temperature": 0.0,
    }
    assert cell_manifest["token_controls"] == {
        "ignore_eos": True,
        "max_tokens": 32,
        "min_tokens": 32,
    }
    benchmark = cell_manifest["commands"]["vllm_benchmark"]
    assert benchmark["kind"] == "real_vllm_benchmark"
    assert benchmark["workload_path"] == cell_manifest["workload"]["path"]
    assert benchmark["workload_hash"] == cell_manifest["workload"]["hash"]
    assert "--base-url" in benchmark["argv"]
    assert benchmark["argv"][benchmark["argv"].index("--base-url") + 1] == benchmark["endpoint"]["base_url"]
    assert "--request-rate" in benchmark["argv"]
    assert "--dataset-path" in benchmark["argv"]
    assert str(cell_root / "workload.vllm-bench.jsonl") in benchmark["argv"]
    comparison = (
        cell_manifest["commands"]["pd_comparison"]
        if cell_manifest["deployment"] == "disaggregated"
        else cell_manifest["commands"]["comparison"]
    )
    assert comparison["kind"] == "real_vllm_comparison"
    assert comparison["inputs"]["command_manifest"] == cell_manifest["command_manifest_path"]
    assert comparison["inputs"]["ground_truth"] == cell_manifest["outputs"]["ground_truth"]
    assert comparison["inputs"]["prediction"] == cell_manifest["outputs"]["prediction"]
    assert comparison["outputs"]["comparison_report"] == cell_manifest["outputs"]["comparison_report"]
    assert comparison["outputs"]["comparison_markdown"] == cell_manifest["outputs"]["comparison_markdown"]
    provenance = cell_manifest["provenance"]
    assert provenance["git_commit"]
    assert isinstance(provenance["git_dirty"], bool)
    assert isinstance(provenance["git_status_short"], list)
    assert isinstance(provenance["git_diff_hash"], str)
    assert provenance["hostname"]
    assert provenance["python_executable"]
    assert provenance["vllm_version"]
    assert provenance["midlayer_command"][0] == "scripts/serve_predict/run_experiments.py"
    assert provenance["environment_overrides"] == cell_manifest["environment"]
    assert provenance["gpu_ids"]
    assert provenance["timestamp_utc"]


def test_dry_run_writes_dense_colocated_command_manifest_without_subprocess(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    run_dir = tmp_path / "run"

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )

    cell_root = run_dir / "dense_tp2"
    command_manifest = cell_root / "command_manifest.json"
    data = _load(command_manifest)
    assert summary["status"] == "dry_run"
    assert summary["cells"][0]["output_root"] == str(cell_root)
    assert summary["cells"][0]["command_manifest"] == str(command_manifest)
    assert data["status"] == "dry_run"
    assert data["cell_id"] == "dense_tp2"
    assert data["deployment"] == "colocated"
    assert data["output_root"] == str(cell_root)
    assert data["manifest_output_root"] == str(tmp_path / "serve_exp/dense_tp2")
    assert data["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    provenance = data["provenance"]
    assert provenance["git_commit"]
    assert provenance["hostname"]
    assert provenance["cuda_driver_summary"]
    assert provenance["python_executable"]
    assert provenance["vllm_version"]
    assert provenance["midlayer_command"] == [
        "scripts/serve_predict/run_experiments.py",
        "--manifest", str(manifest_path),
        "--project-root", str(tmp_path),
        "--run-dir", str(run_dir),
        "--dry-run",
    ]
    assert provenance["environment_overrides"] == data["environment"]
    assert provenance["gpu_ids"] == [0, 1]
    assert provenance["timestamp_utc"]
    assert data["workload"]["path"].endswith("workload.json")
    assert data["workload"]["path"].startswith(str(cell_root))
    assert data["workload"]["hash"]
    assert data["sampling"]["max_tokens"] == 8
    assert data["token_controls"] == {
        "min_tokens": 8,
        "max_tokens": 8,
        "ignore_eos": True,
    }
    assert command_manifest.exists()
    assert Path(data["workload"]["path"]).exists()
    assert (cell_root / "traces").is_dir()
    assert (run_dir / "run_summary.json").exists()
    assert not (tmp_path / "serve_exp/dense_tp2/command_manifest.json").exists()

    serve_argv = data["commands"]["vllm_serve"]["argv"]
    assert serve_argv[:3] == ["vllm", "serve", "/models/dense-7b"]
    assert serve_argv[serve_argv.index("--host") + 1] == "127.0.0.1"
    assert serve_argv[serve_argv.index("--port") + 1] == "8000"
    assert data["endpoint"]["base_url"] == "http://127.0.0.1:8000"
    assert "--tensor-parallel-size" in serve_argv
    assert "2" in serve_argv
    assert "--request-rate" in data["commands"]["vllm_benchmark"]["argv"]
    assert "2.0" in data["commands"]["vllm_benchmark"]["argv"]
    assert data["commands"]["vllm_benchmark"]["workload_hash"] == data["workload"]["hash"]
    emulation = data["commands"]["colocated_midlayer_emulation"]
    prediction = data["commands"]["colocated_midlayer_prediction"]
    assert emulation["kind"] == "colocated_midlayer_emulation"
    assert emulation["workload_path"] == data["workload"]["path"]
    assert "--tensor-parallel-size" in emulation["argv"]
    assert emulation["argv"][emulation["argv"].index("--tensor-parallel-size") + 1] == "2"
    assert emulation["tensor_parallel_size"] == 2
    emulation_env = emulation["environment"]
    assert "build/fake_cuda_libs/libcudart.so.12" in emulation_env["LD_PRELOAD"]
    assert "build/fake_cuda_libs/libcublas.so.12" in emulation_env["LD_PRELOAD"]
    assert "build/fake_cuda_libs/libnccl.so.2" in emulation_env["LD_PRELOAD"]
    assert emulation_env["LD_LIBRARY_PATH"].endswith("build/fake_cuda_libs")
    assert emulation_env["VLLM_NCCL_SO_PATH"].endswith("build/fake_cuda_libs/libnccl.so.2")
    assert emulation_env["VLLM_WORKER_MULTIPROC_METHOD"] == "fork"
    assert "LD_PRELOAD" not in data["commands"]["vllm_serve"]["environment"]
    assert prediction["kind"] == "colocated_midlayer_prediction_json"
    assert _argv_has_script(prediction["argv"], "scripts/serve_predict/predict_serve_metric.py")
    assert "--tp-size" in prediction["argv"]
    assert prediction["argv"][prediction["argv"].index("--tp-size") + 1] == "2"
    assert prediction["tp_size"] == 2
    assert "--json" in prediction["argv"]
    assert "--output" in prediction["argv"]
    assert data["outputs"]["prediction"] in prediction["argv"]
    assert prediction["outputs"]["prediction"] == data["outputs"]["prediction"]
    assert prediction["inputs"]["trace"] == data["traces"]["colocated_trace"]
    comparison = data["commands"]["comparison"]
    assert comparison["kind"] == "real_vllm_comparison"
    assert _argv_has_script(comparison["argv"], "scripts/serve_predict/eval_serve_metric.py")
    assert comparison["diagnostic_mode"] == "colocated"
    assert comparison["inputs"]["command_manifest"] == str(command_manifest)
    assert comparison["outputs"]["comparison_report"] == data["outputs"]["comparison_report"]


def test_dry_run_preserves_moe_parallel_args_and_dependency_status(tmp_path):
    cell = _base_cell(
        id="moe_ep",
        model="/models/moe",
        output_root="serve_exp/moe_ep",
        model_type="moe",
        gpus=[0, 1, 2, 3],
        parallelism={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 4,
            "enable_expert_parallel": True,
        },
        moe={
            "dependency_status": "ready",
            "all2all_backend": "deepep_low_latency",
        },
    )
    manifest_path = _write_manifest(tmp_path, cell)

    run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )

    data = _load(tmp_path / "run/moe_ep/command_manifest.json")
    serve_argv = data["commands"]["vllm_serve"]["argv"]
    assert "--data-parallel-size" in serve_argv
    assert "4" in serve_argv
    assert "--enable-expert-parallel" in serve_argv
    emulation = data["commands"]["colocated_midlayer_emulation"]
    expected_engine_args = json.dumps(data["parallelism"], sort_keys=True)
    assert "--engine-args-json" in emulation["argv"]
    assert emulation["argv"][emulation["argv"].index("--engine-args-json") + 1] == expected_engine_args
    assert emulation["engine_args"] == data["parallelism"]
    assert emulation["engine_args_json"] == expected_engine_args
    assert data["model_type"] == "moe"
    assert data["moe"]["dependency_status"] == "ready"


def test_only_filters_selected_cell(tmp_path):
    moe = _base_cell(id="moe_ep", output_root="serve_exp/moe_ep")
    manifest_path = _write_manifest(tmp_path, _base_cell(), moe)
    run_dir = tmp_path / "run"

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--only", "moe_ep",
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )

    assert [cell["cell_id"] for cell in summary["cells"]] == ["moe_ep"]
    assert (run_dir / "moe_ep/command_manifest.json").exists()
    assert not (run_dir / "dense_tp2").exists()


def test_runner_refuses_to_overwrite_existing_cell_without_resume_or_force(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    run_dir = tmp_path / "run"

    run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )

    with pytest.raises(ManifestValidationError, match="already exists"):
        run_experiments(
            [
                "--manifest", str(manifest_path),
                "--project-root", str(tmp_path),
                "--run-dir", str(run_dir),
                "--dry-run",
            ],
            subprocess_run=_forbidden_run,
        )


def test_force_allows_explicit_rerun_of_existing_cell(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    run_dir = tmp_path / "run"

    run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )
    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--force",
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )

    assert summary["status"] == "dry_run"
    assert _load(run_dir / "dense_tp2/command_manifest.json")["status"] == "dry_run"


def test_mock_run_success_writes_status_logs_and_placeholder_outputs(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _Completed(returncode=0, stdout="ok", stderr="")

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--mock-run",
        ],
        subprocess_run=fake_run,
    )

    cell_root = tmp_path / "run/dense_tp2"
    status = _load(cell_root / "status.json")
    assert summary["status"] == "success"
    assert status["status"] == "success"
    assert len(calls) == 4
    assert (cell_root / "ground_truth.json").exists()
    assert (cell_root / "prediction.json").exists()
    assert (cell_root / "logs/vllm_benchmark.stdout.log").exists()
    assert not (cell_root / "comparison_report.json").exists()


def test_resume_skips_successful_cell_without_subprocess(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    run_dir = tmp_path / "run"
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return _Completed(returncode=0, stdout="ok", stderr="")

    run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--mock-run",
        ],
        subprocess_run=fake_run,
    )
    calls.clear()

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--resume",
            "--mock-run",
        ],
        subprocess_run=_forbidden_run,
    )

    assert calls == []
    assert summary["status"] == "success"
    assert summary["cells"][0]["status"] == "skipped"
    assert summary["cells"][0]["resumed"] is True
    assert _load(run_dir / "dense_tp2/status.json")["status"] == "success"


def test_resume_reruns_failed_cell(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    run_dir = tmp_path / "run"
    calls = []

    def failing_run(argv, **kwargs):
        calls.append(argv)
        return _Completed(returncode=2, stdout="", stderr="benchmark failed")

    run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--mock-run",
        ],
        subprocess_run=failing_run,
    )
    assert _load(run_dir / "dense_tp2/status.json")["status"] == "failed"

    calls.clear()

    def successful_run(argv, **kwargs):
        calls.append(argv)
        return _Completed(returncode=0, stdout="ok", stderr="")

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--resume",
            "--mock-run",
        ],
        subprocess_run=successful_run,
    )

    assert len(calls) == 4
    assert summary["status"] == "success"
    assert _load(run_dir / "dense_tp2/status.json")["status"] == "success"


def test_resume_reruns_blocked_cell(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    run_dir = tmp_path / "run"

    run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )
    cell_root = run_dir / "dense_tp2"
    command_manifest = _load(cell_root / "command_manifest.json")
    command_manifest["status"] = "blocked"
    _write_json(cell_root / "command_manifest.json", command_manifest)
    _write_json(cell_root / "status.json", {
        "status": "blocked",
        "cell_id": "dense_tp2",
        "dependency_category": "cuda",
    })
    calls = []

    def successful_run(argv, **kwargs):
        calls.append(argv)
        return _Completed(returncode=0, stdout="ok", stderr="")

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--resume",
            "--mock-run",
        ],
        subprocess_run=successful_run,
    )

    assert len(calls) == 4
    assert summary["status"] == "success"
    assert _load(cell_root / "status.json")["status"] == "success"


@pytest.mark.parametrize(
    "manifest_status, status_payload",
    [
        ("running", None),
        ("unknown", {"status": "unknown"}),
        ("success", {"state": "success"}),
        (None, None),
    ],
)
def test_resume_rejects_unresumable_existing_cell_state(
    tmp_path,
    manifest_status,
    status_payload,
):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    run_dir = tmp_path / "run"

    run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )
    cell_root = run_dir / "dense_tp2"
    command_manifest_path = cell_root / "command_manifest.json"
    command_manifest = _load(command_manifest_path)
    if manifest_status is None:
        command_manifest.pop("status", None)
    else:
        command_manifest["status"] = manifest_status
    _write_json(command_manifest_path, command_manifest)
    (cell_root / "status.json").unlink(missing_ok=True)
    if status_payload is not None:
        _write_json(cell_root / "status.json", status_payload)

    with pytest.raises(ManifestValidationError, match="unresumable"):
        run_experiments(
            [
                "--manifest", str(manifest_path),
                "--project-root", str(tmp_path),
                "--run-dir", str(run_dir),
                "--resume",
                "--mock-run",
            ],
            subprocess_run=_forbidden_run,
        )


def test_force_allows_rerun_of_unresumable_existing_cell_state(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    run_dir = tmp_path / "run"

    run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )
    cell_root = run_dir / "dense_tp2"
    command_manifest = _load(cell_root / "command_manifest.json")
    command_manifest["status"] = "running"
    _write_json(cell_root / "command_manifest.json", command_manifest)
    calls = []

    def successful_run(argv, **kwargs):
        calls.append(argv)
        return _Completed(returncode=0, stdout="ok", stderr="")

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--force",
            "--mock-run",
        ],
        subprocess_run=successful_run,
    )

    assert len(calls) == 4
    assert summary["status"] == "success"


def test_mock_run_failure_writes_failed_status_without_success_report(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 2:
            return _Completed(returncode=2, stdout="", stderr="benchmark failed")
        return _Completed(returncode=0, stdout="ok", stderr="")

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--mock-run",
        ],
        subprocess_run=fake_run,
    )

    cell_root = tmp_path / "run/dense_tp2"
    status = _load(cell_root / "status.json")
    assert summary["status"] == "failed"
    assert status["status"] == "failed"
    assert status["failed_stage"] == "vllm_benchmark"
    assert status["stderr_summary"] == "benchmark failed"
    assert (cell_root / "logs/vllm_benchmark.stderr.log").exists()
    assert not (cell_root / "comparison_report.json").exists()


def test_real_run_colocated_runs_server_pipeline_and_writes_success_artifacts(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    run_dir = tmp_path / "run"
    calls = []
    servers = []
    readiness_urls = []
    sleeps = []

    def fake_popen(argv, **kwargs):
        server = _FakeServer()
        servers.append((server, argv, kwargs))
        return server

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ["vllm", "bench", "serve"]:
            return _Completed(returncode=0, stdout=json.dumps(_ground_truth_payload()), stderr="")
        if _argv_has_script(argv, "scripts/serve_trace/trace_serving.py"):
            env = kwargs["env"]
            assert "build/fake_cuda_libs/libcudart.so.12" in env["LD_PRELOAD"]
            assert "build/fake_cuda_libs/libnccl.so.2" in env["LD_PRELOAD"]
            assert env["LD_LIBRARY_PATH"].endswith("build/fake_cuda_libs")
            assert env["VLLM_NCCL_SO_PATH"].endswith("build/fake_cuda_libs/libnccl.so.2")
            assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
            return _Completed(returncode=0, stdout="emulation ok", stderr="")
        if _argv_has_script(argv, "scripts/serve_predict/predict_serve_metric.py"):
            _write_json(_argv_value(argv, "--output"), _prediction_payload())
            return _Completed(returncode=0, stdout="prediction ok", stderr="")
        if _argv_has_script(argv, "scripts/serve_predict/eval_serve_metric.py"):
            _write_json(_argv_value(argv, "--output"), {"status": "comparison-ok"})
            Path(_argv_value(argv, "--markdown")).write_text("# Comparison Report\n")
            return _Completed(returncode=0, stdout="comparison ok", stderr="")
        return _Completed(returncode=0, stdout="stage ok", stderr="")

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--real-run",
            "--server-wait-seconds", "2",
            "--readiness-poll-interval-seconds", "1",
        ],
        subprocess_run=fake_run,
        subprocess_popen=fake_popen,
        sleep=lambda seconds: sleeps.append(seconds),
        readiness_probe=lambda url: readiness_urls.append(url) or len(readiness_urls) > 1,
    )

    cell_root = run_dir / "dense_tp2"
    status = _load(cell_root / "status.json")
    command_manifest = _load(cell_root / "command_manifest.json")
    assert summary["mode"] == "real_run"
    assert summary["status"] == "success"
    assert status["status"] == "success"
    assert status["ground_truth"] == str(cell_root / "ground_truth.json")
    assert status["prediction"] == str(cell_root / "prediction.json")
    assert status["comparison_report"] == str(cell_root / "comparison_report.json")
    assert status["comparison_markdown"] == str(cell_root / "comparison_report.md")
    assert set(status["reproducer_commands"]) == set(command_manifest["commands"])
    assert _load(cell_root / "ground_truth.json")["completed"] == 4
    assert _load(cell_root / "prediction.json")[0]["request_id"] == "req-000000"
    assert _load(cell_root / "comparison_report.json")["status"] == "comparison-ok"
    assert (cell_root / "logs/vllm_serve.stdout.log").exists()
    assert (cell_root / "logs/vllm_benchmark.stdout.log").exists()
    assert (cell_root / "logs/comparison.stdout.log").exists()
    assert len(servers) == 1
    server, serve_argv, serve_kwargs = servers[0]
    assert serve_argv[:3] == ["vllm", "serve", "/models/dense-7b"]
    assert serve_argv[serve_argv.index("--host") + 1] == "127.0.0.1"
    assert serve_argv[serve_argv.index("--port") + 1] == "8000"
    assert serve_kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert "PATH" in serve_kwargs["env"]
    assert server.terminated is True
    assert readiness_urls == [
        "http://127.0.0.1:8000/v1/models",
        "http://127.0.0.1:8000/v1/models",
    ]
    assert sleeps == [1.0]
    assert command_manifest["runtime"]["readiness"]["vllm_serve"]["status"] == "ready"
    benchmark_argv = command_manifest["commands"]["vllm_benchmark"]["argv"]
    assert benchmark_argv[benchmark_argv.index("--base-url") + 1] == "http://127.0.0.1:8000"
    assert [call[1] if call[0] == "vllm" else Path(call[1]).name for call in calls] == [
        "bench",
        "trace_serving.py",
        "predict_serve_metric.py",
        "eval_serve_metric.py",
    ]


def test_real_run_colocated_server_dependency_failure_writes_blocked_status(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    calls = []

    def fake_popen(argv, **kwargs):
        kwargs["stderr"].write("CUDA runtime unavailable\n")
        kwargs["stderr"].flush()
        return _FakeServer(returncode=1)

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--real-run",
            "--server-wait-seconds", "0",
        ],
        subprocess_run=lambda argv, **kwargs: calls.append(argv),
        subprocess_popen=fake_popen,
        sleep=lambda seconds: None,
    )

    cell_root = tmp_path / "run/dense_tp2"
    status = _load(cell_root / "status.json")
    assert calls == []
    assert summary["status"] == "blocked"
    assert status["status"] == "blocked"
    assert status["cell_id"] == "dense_tp2"
    assert status["logs"] == str(cell_root / "logs")
    assert status["failed_stage"] == "vllm_serve"
    assert status["dependency_category"] == "cuda"
    assert "CUDA runtime unavailable" in status["stderr_summary"]
    assert status["reproducer_command"][:3] == ["vllm", "serve", "/models/dense-7b"]
    assert status["unblock_checklist"]
    assert not (cell_root / "comparison_report.json").exists()


def test_real_run_rejects_non_positive_readiness_poll_interval(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())

    with pytest.raises(ManifestValidationError, match="readiness-poll-interval"):
        run_experiments(
            [
                "--manifest", str(manifest_path),
                "--project-root", str(tmp_path),
                "--run-dir", str(tmp_path / "run"),
                "--real-run",
                "--readiness-poll-interval-seconds", "0",
            ],
            subprocess_run=_forbidden_run,
        )


def test_real_run_colocated_readiness_timeout_blocks_before_benchmark(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    calls = []

    def fake_popen(argv, **kwargs):
        return _FakeServer()

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--real-run",
            "--server-wait-seconds", "0",
        ],
        subprocess_run=lambda argv, **kwargs: calls.append(argv),
        subprocess_popen=fake_popen,
        sleep=lambda seconds: None,
        readiness_probe=lambda url: False,
    )

    cell_root = tmp_path / "run/dense_tp2"
    status = _load(cell_root / "status.json")
    command_manifest = _load(cell_root / "command_manifest.json")
    assert calls == []
    assert summary["status"] == "blocked"
    assert status["status"] == "blocked"
    assert status["failed_stage"] == "vllm_serve_readiness"
    assert status["dependency_category"] == "runtime"
    assert "readiness timeout" in status["stderr_summary"]
    assert command_manifest["runtime"]["readiness"]["vllm_serve"]["status"] == "timeout"
    assert (cell_root / "logs/vllm_serve_readiness.stderr.log").exists()


@pytest.mark.parametrize(
    "stderr, expected_category",
    [
        ("ModuleNotFoundError: No module named 'deepep'", "dependency"),
        ("unsupported all-to-all backend deepep_low_latency", "backend"),
        ("model path not found: /bad/model", "model"),
        ("Connection refused while polling /v1/models", "runtime"),
        ("address already in use on port 8000", "port"),
    ],
)
def test_real_run_server_failures_classify_common_blockers(
    tmp_path,
    stderr,
    expected_category,
):
    manifest_path = _write_manifest(tmp_path, _base_cell())

    def fake_popen(argv, **kwargs):
        kwargs["stderr"].write(stderr)
        kwargs["stderr"].flush()
        return _FakeServer(returncode=1)

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--real-run",
            "--server-wait-seconds", "0",
        ],
        subprocess_run=_forbidden_run,
        subprocess_popen=fake_popen,
        sleep=lambda seconds: None,
    )

    status = _load(tmp_path / "run/dense_tp2/status.json")
    assert summary["status"] == "blocked"
    assert status["dependency_category"] == expected_category


def test_real_run_pd_manages_servers_before_benchmark_and_blocks_on_dependency(tmp_path):
    manifest_path = _write_manifest(tmp_path, _pd_cell())
    calls = []
    servers = []

    def fake_popen(argv, **kwargs):
        server = _FakeServer()
        servers.append((server, argv, kwargs))
        return server

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ["vllm", "bench", "serve"]:
            return _Completed(returncode=0, stdout=json.dumps(_ground_truth_payload()), stderr="")
        if _argv_has_script(argv, "scripts/serve_trace/trace_disagg.py"):
            assert [server.terminated for server, _, _ in servers] == [True, True, True]
            return _Completed(
                returncode=1,
                stdout="",
                stderr="NCCL connector dependency missing",
            )
        raise AssertionError(f"unexpected stage after dependency failure: {argv}")

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--real-run",
            "--server-wait-seconds", "0",
        ],
        subprocess_run=fake_run,
        subprocess_popen=fake_popen,
        sleep=lambda seconds: None,
        readiness_probe=lambda url: True,
    )

    cell_root = tmp_path / "run/dense_pd"
    status = _load(cell_root / "status.json")
    command_manifest = _load(cell_root / "command_manifest.json")
    assert len(calls) == 2
    assert len(servers) == 3
    assert servers[0][1][:3] == ["vllm", "serve", "/models/dense-7b"]
    assert servers[0][1][servers[0][1].index("--port") + 1] == "15580"
    assert servers[1][1][servers[1][1].index("--port") + 1] == "15579"
    assert _argv_has_script(servers[2][1], "scripts/serve_trace/disagg_router.py")
    assert servers[2][1][servers[2][1].index("--port") + 1] == "15590"
    assert servers[2][1][servers[2][1].index("--proxy-port") + 1] == "30001"
    assert servers[0][2]["env"]["CUDA_VISIBLE_DEVICES"] == "2"
    assert servers[1][2]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert summary["status"] == "blocked"
    assert status["status"] == "blocked"
    assert status["failed_stage"] == "disaggregated_midlayer_emulation"
    assert status["dependency_category"] == "nccl"
    assert status["reproducer_command"] == calls[1]
    assert command_manifest["runtime"]["readiness"]["pd_decode_serve"]["status"] == "ready"
    assert command_manifest["runtime"]["readiness"]["pd_prefill_serve"]["status"] == "ready"
    assert command_manifest["runtime"]["readiness"]["pd_proxy_serve"]["status"] == "ready"
    assert calls[0][calls[0].index("--base-url") + 1] == "http://127.0.0.1:15590"
    assert (cell_root / "ground_truth.json").exists()
    assert (cell_root / "logs/disaggregated_midlayer_emulation.stderr.log").exists()
    assert not (cell_root / "prediction.json").exists()
    assert not (cell_root / "comparison_report.json").exists()


def test_real_run_pd_proxy_exit_blocks_before_benchmark(tmp_path):
    manifest_path = _write_manifest(tmp_path, _pd_cell())
    calls = []
    started = []

    def fake_popen(argv, **kwargs):
        started.append(argv)
        if _argv_has_script(argv, "scripts/serve_trace/disagg_router.py"):
            kwargs["stderr"].write("proxy dependency missing\n")
            kwargs["stderr"].flush()
            return _FakeServer(returncode=1)
        return _FakeServer()

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--real-run",
            "--server-wait-seconds", "0",
        ],
        subprocess_run=lambda argv, **kwargs: calls.append(argv),
        subprocess_popen=fake_popen,
        sleep=lambda seconds: None,
        readiness_probe=lambda url: True,
    )

    status = _load(tmp_path / "run/dense_pd/status.json")
    assert calls == []
    assert len(started) == 3
    assert summary["status"] == "blocked"
    assert status["status"] == "blocked"
    assert status["failed_stage"] == "pd_proxy_serve"
    assert status["dependency_category"] == "dependency"
    assert "proxy dependency missing" in status["stderr_summary"]
    assert set(status["reproducer_commands"]) == {
        "pd_decode_serve",
        "pd_prefill_serve",
        "pd_proxy_serve",
    }


def test_real_run_success_requires_persisted_comparison_artifacts(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())

    def fake_popen(argv, **kwargs):
        return _FakeServer()

    def fake_run(argv, **kwargs):
        if argv[:3] == ["vllm", "bench", "serve"]:
            return _Completed(returncode=0, stdout=json.dumps(_ground_truth_payload()), stderr="")
        if _argv_has_script(argv, "scripts/serve_predict/predict_serve_metric.py"):
            _write_json(_argv_value(argv, "--output"), _prediction_payload())
        return _Completed(returncode=0, stdout="ok", stderr="")

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--real-run",
            "--server-wait-seconds", "0",
        ],
        subprocess_run=fake_run,
        subprocess_popen=fake_popen,
        sleep=lambda seconds: None,
        readiness_probe=lambda url: True,
    )

    cell_root = tmp_path / "run/dense_tp2"
    status = _load(cell_root / "status.json")
    assert summary["status"] == "failed"
    assert status["status"] == "failed"
    assert status["failed_stage"] == "comparison"
    assert "missing real-run output artifacts" in status["stderr_summary"]
    assert not (cell_root / "comparison_report.json").exists()


def test_two_dry_runs_with_same_manifest_have_disjoint_cell_artifacts(tmp_path):
    manifest_path = _write_manifest(tmp_path, _base_cell())
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"

    summary_a = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_a),
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )
    summary_b = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_b),
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )

    manifest_a = Path(summary_a["cells"][0]["command_manifest"])
    manifest_b = Path(summary_b["cells"][0]["command_manifest"])
    data_a = _load(manifest_a)
    data_b = _load(manifest_b)

    assert manifest_a == run_a / "dense_tp2/command_manifest.json"
    assert manifest_b == run_b / "dense_tp2/command_manifest.json"
    assert manifest_a != manifest_b
    assert data_a["workload"]["path"].startswith(str(run_a))
    assert data_b["workload"]["path"].startswith(str(run_b))
    assert data_a["outputs"]["logs"].startswith(str(run_a))
    assert data_b["outputs"]["logs"].startswith(str(run_b))
    assert data_a["manifest_output_root"] == data_b["manifest_output_root"]


def test_pd_dry_run_writes_disaggregated_command_manifest(tmp_path):
    manifest_path = _write_manifest(tmp_path, _pd_cell())
    run_dir = tmp_path / "run"

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(run_dir),
            "--dry-run",
        ],
        subprocess_run=_forbidden_run,
    )

    cell_root = run_dir / "dense_pd"
    data = _load(cell_root / "command_manifest.json")
    assert summary["status"] == "dry_run"
    assert data["deployment"] == "disaggregated"
    assert data["output_root"] == str(cell_root)
    assert data["prefill"]["gpus"] == [0, 1]
    assert data["decode"]["gpus"] == [2]
    assert data["prefill"]["engine_args"] == {"tensor_parallel_size": 2}
    assert data["decode"]["engine_args"] == {"tensor_parallel_size": 1}
    assert data["prefill"]["kv_port"] == 14579
    assert data["decode"]["kv_port"] == 14582
    assert data["kv_connector"] == "P2pNcclConnector"
    assert data["workload"]["path"].startswith(str(cell_root))
    assert data["workload"]["hash"]
    assert data["traces"]["prefill_trace"].startswith(str(cell_root))
    assert data["traces"]["decode_trace"].startswith(str(cell_root))
    assert (cell_root / "traces").is_dir()
    assert data["sidecars"]["prefill_sidecar"].startswith(str(cell_root))
    assert data["sidecars"]["decode_sidecar"].startswith(str(cell_root))
    assert data["endpoints"]["prefill"]["base_url"] == "http://127.0.0.1:15579"
    assert data["endpoints"]["decode"]["base_url"] == "http://127.0.0.1:15580"
    assert data["endpoints"]["router"]["base_url"] == "http://127.0.0.1:15590"
    assert data["router"]["proxy_port"] == 30001
    assert data["commands"]["pd_prefill_serve"]["endpoint"] == data["endpoints"]["prefill"]
    assert data["commands"]["pd_decode_serve"]["endpoint"] == data["endpoints"]["decode"]
    assert data["commands"]["pd_proxy_serve"]["endpoint"] == data["endpoints"]["router"]
    prefill_serve = data["commands"]["pd_prefill_serve"]["argv"]
    decode_serve = data["commands"]["pd_decode_serve"]["argv"]
    assert prefill_serve[:3] == ["vllm", "serve", "/models/dense-7b"]
    assert decode_serve[:3] == ["vllm", "serve", "/models/dense-7b"]
    assert prefill_serve[prefill_serve.index("--port") + 1] == "15579"
    assert decode_serve[decode_serve.index("--port") + 1] == "15580"
    assert prefill_serve[prefill_serve.index("--tensor-parallel-size") + 1] == "2"
    assert decode_serve[decode_serve.index("--tensor-parallel-size") + 1] == "1"
    prefill_kv = json.loads(prefill_serve[prefill_serve.index("--kv-transfer-config") + 1])
    decode_kv = json.loads(decode_serve[decode_serve.index("--kv-transfer-config") + 1])
    assert prefill_kv["kv_connector"] == "P2pNcclConnector"
    assert prefill_kv["kv_role"] == "kv_producer"
    assert prefill_kv["kv_rank"] == 0
    assert prefill_kv["kv_parallel_size"] == 2
    assert prefill_kv["kv_ip"] == "127.0.0.1"
    assert prefill_kv["kv_port"] == 14579
    assert decode_kv["kv_role"] == "kv_consumer"
    assert decode_kv["kv_rank"] == 1
    assert decode_kv["kv_parallel_size"] == 2
    assert decode_kv["kv_ip"] == "127.0.0.1"
    assert decode_kv["kv_port"] == 14582
    assert prefill_kv["kv_connector_extra_config"] == {
        "http_ip": "127.0.0.1",
        "http_port": "15579",
        "proxy_ip": "127.0.0.1",
        "proxy_port": "30001",
        "send_type": "PUT_ASYNC",
    }
    assert decode_kv["kv_connector_extra_config"] == {
        "http_ip": "127.0.0.1",
        "http_port": "15580",
        "proxy_ip": "127.0.0.1",
        "proxy_port": "30001",
        "send_type": "PUT_ASYNC",
    }
    assert data["commands"]["pd_prefill_serve"]["environment"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert data["commands"]["pd_decode_serve"]["environment"]["CUDA_VISIBLE_DEVICES"] == "2"
    proxy_serve = data["commands"]["pd_proxy_serve"]["argv"]
    assert _argv_has_script(proxy_serve, "scripts/serve_trace/disagg_router.py")
    assert proxy_serve[proxy_serve.index("--prefill-base-url") + 1] == "http://127.0.0.1:15579"
    assert proxy_serve[proxy_serve.index("--decode-base-url") + 1] == "http://127.0.0.1:15580"
    assert "vllm_benchmark" in data["commands"]
    benchmark = data["commands"]["vllm_benchmark"]
    assert benchmark["argv"][benchmark["argv"].index("--base-url") + 1] == "http://127.0.0.1:15590"
    assert "disaggregated_midlayer_emulation" in data["commands"]
    assert "disaggregated_midlayer_prediction" in data["commands"]
    assert data["commands"]["disaggregated_midlayer_emulation"]["workload_path"] == data["workload"]["path"]
    assert data["commands"]["disaggregated_midlayer_prediction"]["workload_hash"] == data["workload"]["hash"]
    emulation = data["commands"]["disaggregated_midlayer_emulation"]
    prediction = data["commands"]["disaggregated_midlayer_prediction"]
    expected_prefill = json.dumps(data["prefill"]["engine_args"], sort_keys=True)
    expected_decode = json.dumps(data["decode"]["engine_args"], sort_keys=True)
    assert "--prefill-engine-args" in emulation["argv"]
    assert emulation["argv"][emulation["argv"].index("--prefill-engine-args") + 1] == expected_prefill
    assert "--decode-engine-args" in emulation["argv"]
    assert emulation["argv"][emulation["argv"].index("--decode-engine-args") + 1] == expected_decode
    assert emulation["prefill_engine_args"] == data["prefill"]["engine_args"]
    assert emulation["decode_engine_args"] == data["decode"]["engine_args"]
    assert emulation["prefill_engine_args_json"] == expected_prefill
    assert emulation["decode_engine_args_json"] == expected_decode
    assert "--prefill-kv-port" in emulation["argv"]
    assert emulation["argv"][emulation["argv"].index("--prefill-kv-port") + 1] == "14579"
    assert "--decode-kv-port" in emulation["argv"]
    assert emulation["argv"][emulation["argv"].index("--decode-kv-port") + 1] == "14582"
    assert emulation["prefill_kv_port"] == 14579
    assert emulation["decode_kv_port"] == 14582
    assert "--prefill-tp-size" in prediction["argv"]
    assert prediction["argv"][prediction["argv"].index("--prefill-tp-size") + 1] == "2"
    assert "--decode-tp-size" in prediction["argv"]
    assert prediction["argv"][prediction["argv"].index("--decode-tp-size") + 1] == "1"
    assert "--json" in prediction["argv"]
    assert "--output" in prediction["argv"]
    assert prediction["argv"][prediction["argv"].index("--output") + 1] == data["outputs"]["prediction"]
    assert prediction["prefill_tp_size"] == 2
    assert prediction["decode_tp_size"] == 1
    assert prediction["outputs"]["prediction"] == data["outputs"]["prediction"]
    assert data["comparison"]["output_path"] == str(cell_root / "comparison_report.json")
    assert data["comparison"]["primary_diagnostics"] == [
        "ttft",
        "tpot",
        "itl",
        "tail_latency",
        "kv_transfer_contribution",
        "prediction_error",
    ]
    assert data["comparison"]["informational_metrics"] == ["throughput"]
    assert data["comparison"]["throughput_pass_fail"] is False
    assert data["comparison"]["inputs"]["ground_truth"] == data["outputs"]["ground_truth"]
    assert data["comparison"]["inputs"]["prediction"] == data["outputs"]["prediction"]
    assert "pd_comparison" in data["commands"]
    assert data["commands"]["pd_comparison"]["kind"] == "real_vllm_comparison"
    assert _argv_has_script(data["commands"]["pd_comparison"]["argv"], "scripts/serve_predict/eval_serve_metric.py")
    assert data["commands"]["pd_comparison"]["diagnostic_mode"] == "pd"


def test_pd_runner_rejects_missing_http_routing_fields_before_materialization(tmp_path):
    cell = _pd_cell()
    cell["prefill"].pop("http_port")
    cell["decode"].pop("http_port")
    cell.pop("router")
    manifest_path = _write_manifest(tmp_path, cell)

    with pytest.raises(ManifestValidationError, match="prefill.http_port.*decode.http_port.*router.host"):
        run_experiments(
            [
                "--manifest", str(manifest_path),
                "--project-root", str(tmp_path),
                "--run-dir", str(tmp_path / "run"),
                "--dry-run",
            ],
            subprocess_run=_forbidden_run,
        )


def test_pd_mock_dependency_failure_writes_blocked_status(tmp_path):
    manifest_path = _write_manifest(tmp_path, _pd_cell())
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 2:
            return _Completed(
                returncode=1,
                stdout="",
                stderr="NCCL dependency missing for connector",
            )
        return _Completed(returncode=0, stdout="ok", stderr="")

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--mock-run",
        ],
        subprocess_run=fake_run,
    )

    cell_root = tmp_path / "run/dense_pd"
    status = _load(cell_root / "status.json")
    assert summary["status"] == "blocked"
    assert status["status"] == "blocked"
    assert status["dependency_category"] == "nccl"
    assert status["failed_stage"] == "disaggregated_midlayer_emulation"
    assert status["stderr_summary"] == "NCCL dependency missing for connector"
    assert status["unblock_checklist"]
    assert (cell_root / "logs/disaggregated_midlayer_emulation.stderr.log").exists()
    assert not (cell_root / "comparison_report.json").exists()


def test_pd_mock_success_writes_disaggregated_prediction_artifact(tmp_path):
    manifest_path = _write_manifest(tmp_path, _pd_cell())

    def fake_run(argv, **kwargs):
        return _Completed(returncode=0, stdout="ok", stderr="")

    summary = run_experiments(
        [
            "--manifest", str(manifest_path),
            "--project-root", str(tmp_path),
            "--run-dir", str(tmp_path / "run"),
            "--mock-run",
        ],
        subprocess_run=fake_run,
    )

    cell_root = tmp_path / "run/dense_pd"
    status = _load(cell_root / "status.json")
    prediction = _load(cell_root / "prediction.json")
    manifest = _load(cell_root / "command_manifest.json")
    assert summary["status"] == "success"
    assert status["status"] == "success"
    assert prediction["kind"] == "disaggregated_prediction"
    assert prediction["workload_hash"] == manifest["workload"]["hash"]
    assert prediction["inputs"]["prefill_trace"] == manifest["traces"]["prefill_trace"]
    assert prediction["inputs"]["decode_trace"] == manifest["traces"]["decode_trace"]
    assert prediction["inputs"]["prefill_sidecar"] == manifest["sidecars"]["prefill_sidecar"]
    assert prediction["inputs"]["decode_sidecar"] == manifest["sidecars"]["decode_sidecar"]
    assert prediction["kv_connector"] == "P2pNcclConnector"
    assert prediction["kv_transfer"]["status"] == "mock-placeholder"
    assert prediction["metrics"][0]["ttft_disagg_ms"] is None
    assert prediction["metrics"][0]["tpot_disagg_ms"] is None
    assert prediction["metrics"][0]["kv_transfer_time_ms"] is None


class _Completed:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _FakeServer:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.pid = 4242
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


def _argv_value(argv, flag):
    return Path(argv[argv.index(flag) + 1])


def _ground_truth_payload():
    return {
        "completed": 4,
        "total_input_tokens": 32,
        "total_output_tokens": 32,
        "request_throughput": 2.0,
        "output_throughput": 16.0,
        "mean_ttft_ms": 10.0,
        "mean_tpot_ms": 2.0,
    }


def _prediction_payload():
    return [
        {
            "request_id": "req-000000",
            "ttft_ms": 11.0,
            "tpot_ms": 2.2,
            "request_throughput": 2.0,
            "output_throughput": 16.0,
            "start_time_ms": 0.0,
            "end_time_ms": 100.0,
        }
    ]
