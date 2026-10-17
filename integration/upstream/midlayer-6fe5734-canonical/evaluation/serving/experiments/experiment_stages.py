"""Execute the process and server stages of one prepared Serving experiment."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import asdict, fields
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from evaluation.serving.experiments.experiment_manifest import ManifestValidationError
from evaluation.serving.metrics.production_trace_report import (
    render_production_trace_sweep_report,
)
from evaluation.serving.metrics.latency_distribution import (
    latency_distribution_manifest,
)
from evaluation.serving.metrics.interactivity_tgs import (
    arrival_rate_sweep_analysis_manifest,
    load_interactivity_tgs_contract,
    sweep_metric_point_from_manifest,
)
from evaluation.serving.metrics.slo_goodput import (
    load_slo_contract,
    slo_goodput_manifest,
)
from evaluation.serving.metrics.system_resource_analysis import (
    critical_path_manifest,
    system_resource_manifest,
)
from evaluation.serving.simulation.execution_artifact import (
    execution_artifact_manifest,
    execution_step_components_manifest,
    performance_timing_status,
    prefix_alignment_manifest,
)
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    PhysicalTransferConfig,
)
from src.serving.backend.vllm_scheduler_process import (
    VLLMSchedulerRuntimeInfo,
    vllm_scheduler_runtime_sha256,
    vllm_scheduler_source_identity,
)
from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.backend.mtp_acceptance_assumption import (
    MTPAcceptanceAssumption,
)
from src.serving.backend.mtp_acceptance_replay import (
    mtp_acceptance_step_from_backend_step,
    mtp_acceptance_steps_manifest,
)
from src.serving.deployment.af_disagg import build_af_disagg_plan
from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    PipelineP2PConfig,
    RouteConfig,
    RouterConfig,
    WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import (
    DeploymentPlan,
    deployment_plan_manifest,
)
from src.serving.deployment.pd_disagg import build_pd_disagg_plan
from src.serving.kv_cache.kv_cache_events import KVCacheEvent
from src.serving.kv_cache.kv_cache_tier_manager import KVCacheTierManager
from src.serving.kv_cache.kv_lifecycle_replay import (
    load_kv_lifecycle_replay,
)
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.request_gen.production_trace import (
    ProductionTraceReaderConfig,
    SuccessfulRequestFilter,
    build_production_trace_replay,
    load_production_trace,
    production_trace_manifest,
    production_trace_replay_manifest,
)
from src.serving.request_gen.request_stream import (
    RequestStream,
    request_stream_sha256,
)
from src.serving.schedule.backend_protocol import (
    BackendConfig,
    scheduler_effective_prefix_hit_tokens,
)
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduleStep,
    ScheduledRequest,
)
from src.serving.schedule.schedule_trace import (
    RouteDecision,
    ScheduleTrace,
    ScheduleTraceEntry,
)
from src.serving.schedule.single_pool_engine import run_single_pool_schedule
from src.serving.metrics.serving_metrics import compute_serving_metrics
from src.serving.execution import ServingExecutionContext
from src.serving.simulation import (
    MultiSimulation,
    Simulation,
    SimulationConfig,
    SimulationResult,
    SimulationStep,
)
from src.serving.workload.build_workload import (
    tp_collective_payload_bytes,
)
from src.serving.workload.graph_padding import GraphExecutionConfig
from src.serving.workload.model_config import ModelConfig, load_model_config
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    RuntimeInputs,
)
from src.serving.workload.shape_inference import infer_gemm_dimensions
from src.serving.workload.workload_graph import WorkloadKind

from evaluation.json_shape import write_json_file as _write_json

ROOT = Path(__file__).resolve().parents[3]


def execute_experiment_stages(
    command_manifest: dict[str, Any],
    *,
    mode: str,
    subprocess_run,
    subprocess_popen,
    server_wait_seconds: float,
    readiness_poll_interval_seconds: float,
    sleep,
    readiness_probe,
) -> dict[str, Any]:
    """Execute the explicit stages of one already materialized plan."""
    if mode == "dry_run":
        command_manifest["status"] = "dry_run"
        _write_cell_manifest(command_manifest)
        return command_manifest
    elif mode == "mock_run":
        return run_mock_stages(command_manifest, subprocess_run)
    elif mode == "real_run":
        return run_real_stages(
            command_manifest,
            subprocess_run,
            subprocess_popen,
            server_wait_seconds=server_wait_seconds,
            readiness_poll_interval_seconds=readiness_poll_interval_seconds,
            sleep=sleep,
            readiness_probe=readiness_probe,
        )
    else:
        raise ManifestValidationError(f"unsupported experiment mode: {mode}")


def run_mock_stages(
    command_manifest: dict[str, Any],
    subprocess_run,
) -> dict[str, Any]:
    if "experiment_type" in command_manifest:
        experiment_type = command_manifest["experiment_type"]
        if experiment_type == "production_trace":
            return _run_production_trace_stages(
                command_manifest, subprocess_run, mode="mock_run")
        raise ManifestValidationError(
            f"unknown experiment type: {experiment_type!r}")
    output_root = Path(command_manifest["output_root"])
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_manifest["status"] = "running"
    _write_cell_manifest(command_manifest)
    for stage in _mock_stages(command_manifest):
        command = command_manifest["commands"][stage]
        result = subprocess_run(
            command["argv"],
            cwd=str(ROOT),
            env=_stage_environment(command_manifest, command),
            capture_output=True,
            text=True,
        )
        _write_text(
            logs_dir / f"{stage}.stdout.log",
            _required_process_stream(result, "stdout"))
        _write_text(
            logs_dir / f"{stage}.stderr.log",
            _required_process_stream(result, "stderr"))
        if result.returncode != 0:
            status = _failure_status(stage, result, logs_dir, command)
            command_manifest["status"] = status["status"]
            command_manifest["failed_stage"] = stage
            _write_json(output_root / "status.json", status)
            _write_cell_manifest(command_manifest)
            return command_manifest

    _write_placeholder_outputs(command_manifest)
    status = {
        "status": "success",
        "cell_id": command_manifest["cell_id"],
        "logs": str(logs_dir),
        "ground_truth": command_manifest["outputs"]["ground_truth"],
        "prediction": command_manifest["outputs"]["prediction"],
    }
    command_manifest["status"] = "success"
    _write_json(output_root / "status.json", status)
    _write_cell_manifest(command_manifest)
    return command_manifest


def run_real_stages(
    command_manifest: dict[str, Any],
    subprocess_run,
    subprocess_popen,
    *,
    server_wait_seconds: float,
    readiness_poll_interval_seconds: float,
    sleep,
    readiness_probe,
) -> dict[str, Any]:
    if "experiment_type" in command_manifest:
        experiment_type = command_manifest["experiment_type"]
        if experiment_type == "production_trace":
            return _run_production_trace_stages(
                command_manifest, subprocess_run, mode="real_run")
        raise ManifestValidationError(
            f"unknown experiment type: {experiment_type!r}")
    if readiness_probe is None:
        readiness_probe = _http_readiness_probe
    if command_manifest["deployment"] == "disaggregated":
        return _run_real_disaggregated_cell(
            command_manifest,
            subprocess_run,
            subprocess_popen,
            server_wait_seconds=server_wait_seconds,
            readiness_poll_interval_seconds=readiness_poll_interval_seconds,
            sleep=sleep,
            readiness_probe=readiness_probe,
        )
    elif command_manifest["deployment"] == "colocated":
        return _run_real_colocated_cell(
            command_manifest,
            subprocess_run,
            subprocess_popen,
            server_wait_seconds=server_wait_seconds,
            readiness_poll_interval_seconds=readiness_poll_interval_seconds,
            sleep=sleep,
            readiness_probe=readiness_probe,
        )
    else:
        raise ManifestValidationError(
            "unknown experiment deployment: "
            f"{command_manifest['deployment']!r}")


def _run_production_trace_stages(
    command_manifest: dict[str, Any],
    subprocess_run,
    *,
    mode: str,
) -> dict[str, Any]:
    if mode not in {"mock_run", "real_run"}:
        raise ManifestValidationError(
            f"unsupported Production Trace execution mode: {mode!r}")
    output_root = Path(command_manifest["output_root"])
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_manifest["status"] = "running"
    command_manifest["runtime"] = {
        "mode": mode,
        "process_stages": [],
        "device_access": "forbidden",
    }
    _write_cell_manifest(command_manifest)
    for stage in (
        "production_trace_simulation",
        "production_trace_report",
    ):
        status = _run_production_trace_stage(
            command_manifest, stage, logs_dir, subprocess_run)
        if status is not None:
            return _finish_failed_cell(command_manifest, status)
        command_manifest["runtime"]["process_stages"].append(stage)
        _write_cell_manifest(command_manifest)
    try:
        validate_production_trace_outputs(command_manifest)
    except (OSError, TypeError, ValueError) as error:
        result = _SyntheticResult(
            returncode=1,
            stdout="",
            stderr=f"invalid Production Trace outputs: {error}",
        )
        status = _failure_status(
            "production_trace_report",
            result,
            logs_dir,
            command_manifest["commands"]["production_trace_report"],
        )
        return _finish_failed_cell(command_manifest, status)
    status = {
        "status": "success",
        "cell_id": command_manifest["cell_id"],
        "logs": str(logs_dir),
        "sweep": command_manifest["outputs"]["sweep"],
        "report_markdown": command_manifest["outputs"]["report_markdown"],
        "reproducer_commands": _production_trace_reproducer_commands(
            command_manifest),
    }
    command_manifest["status"] = "success"
    _write_json(output_root / "status.json", status)
    _write_cell_manifest(command_manifest)
    return command_manifest


def _run_production_trace_stage(
    command_manifest: dict[str, Any],
    stage: str,
    logs_dir: Path,
    subprocess_run,
) -> dict[str, Any] | None:
    allowed = {
        "production_trace_simulation",
        "production_trace_report",
    }
    if stage not in allowed:
        raise ManifestValidationError(
            f"unsupported Production Trace stage: {stage!r}")
    command = command_manifest["commands"][stage]
    if stage == "production_trace_simulation":
        required_path = Path(command_manifest["outputs"]["sweep"])
    elif stage == "production_trace_report":
        required_path = Path(command_manifest["outputs"]["report_markdown"])
    else:
        raise AssertionError("Production Trace stage allowlist is exhaustive")
    previous_identity = _artifact_write_identity(required_path)
    try:
        result = subprocess_run(
            command["argv"],
            cwd=str(ROOT),
            env=_stage_environment(command_manifest, command),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        result = _missing_executable_result(error)
    stdout = _required_process_stream(result, "stdout")
    stderr = _required_process_stream(result, "stderr")
    _write_text(logs_dir / f"{stage}.stdout.log", stdout)
    _write_text(logs_dir / f"{stage}.stderr.log", stderr)
    if result.returncode != 0:
        return _failure_status(stage, result, logs_dir, command)
    if not required_path.is_file():
        missing = _SyntheticResult(
            returncode=1,
            stdout=stdout,
            stderr=(
                f"Production Trace stage {stage!r} did not write required "
                f"artifact: {required_path}"),
        )
        return _failure_status(stage, missing, logs_dir, command)
    current_identity = _artifact_write_identity(required_path)
    if current_identity == previous_identity:
        stale = _SyntheticResult(
            returncode=1,
            stdout=stdout,
            stderr=(
                f"Production Trace stage {stage!r} did not replace its "
                f"pre-existing artifact: {required_path}"),
        )
        return _failure_status(stage, stale, logs_dir, command)
    return None


def _artifact_write_identity(path: Path) -> tuple[int, int, int, str] | None:
    if not path.is_file():
        return None
    payload = path.read_bytes()
    stat = path.stat()
    return (
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        len(payload),
        hashlib.sha256(payload).hexdigest(),
    )


def _run_real_colocated_cell(
    command_manifest: dict[str, Any],
    subprocess_run,
    subprocess_popen,
    *,
    server_wait_seconds: float,
    readiness_poll_interval_seconds: float,
    sleep,
    readiness_probe,
) -> dict[str, Any]:
    output_root = Path(command_manifest["output_root"])
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_manifest["status"] = "running"
    command_manifest["runtime"] = {
        "mode": "real_run",
        "servers": {},
        "readiness": {},
    }
    _write_cell_manifest(command_manifest)

    server_record = None
    try:
        server_record, status = _start_server_process(
            command_manifest,
            "vllm_serve",
            logs_dir,
            subprocess_popen,
        )
        if status is not None:
            return _finish_failed_cell(command_manifest, status)
        _write_cell_manifest(command_manifest)
        status = _wait_for_server_readiness(
            command_manifest,
            server_record,
            "vllm_serve",
            logs_dir,
            timeout_seconds=server_wait_seconds,
            poll_interval_seconds=readiness_poll_interval_seconds,
            sleep=sleep,
            readiness_probe=readiness_probe,
        )
        if status is not None:
            return _finish_failed_cell(command_manifest, status)

        for stage in (
            "vllm_benchmark",
            "colocated_midlayer_simulation",
            "comparison",
        ):
            status = _run_real_stage(command_manifest, stage, logs_dir, subprocess_run)
            if status is not None:
                return _finish_failed_cell(command_manifest, status)
        return _finish_successful_real_cell(command_manifest, logs_dir)
    finally:
        if server_record is not None:
            _terminate_server_process(server_record)


def _run_real_disaggregated_cell(
    command_manifest: dict[str, Any],
    subprocess_run,
    subprocess_popen,
    *,
    server_wait_seconds: float,
    readiness_poll_interval_seconds: float,
    sleep,
    readiness_probe,
) -> dict[str, Any]:
    output_root = Path(command_manifest["output_root"])
    logs_dir = output_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    command_manifest["status"] = "running"
    command_manifest["runtime"] = {
        "mode": "real_run",
        "servers": {},
        "readiness": {},
    }
    _write_cell_manifest(command_manifest)
    server_records = []
    try:
        for stage in ("pd_decode_serve", "pd_prefill_serve", "pd_proxy_serve"):
            server_record, status = _start_server_process(
                command_manifest,
                stage,
                logs_dir,
                subprocess_popen,
            )
            if status is not None:
                status["reproducer_commands"] = _pd_reproducer_commands(command_manifest)
                return _finish_failed_cell(command_manifest, status)
            server_records.append(server_record)
            _write_cell_manifest(command_manifest)

        for server_record in server_records:
            status = _wait_for_server_readiness(
                command_manifest,
                server_record,
                server_record["stage"],
                logs_dir,
                timeout_seconds=server_wait_seconds,
                poll_interval_seconds=readiness_poll_interval_seconds,
                sleep=sleep,
                readiness_probe=readiness_probe,
            )
            if status is not None:
                status["reproducer_commands"] = _pd_reproducer_commands(command_manifest)
                return _finish_failed_cell(command_manifest, status)
        for server_record in server_records:
            status = _server_exit_status(server_record, logs_dir)
            if status is not None:
                status["reproducer_commands"] = _pd_reproducer_commands(command_manifest)
                return _finish_failed_cell(command_manifest, status)

        status = _run_real_stage(command_manifest, "vllm_benchmark", logs_dir, subprocess_run)
        if status is not None:
            return _finish_failed_cell(command_manifest, status)
        _terminate_server_records(server_records)

        for stage in (
            "disaggregated_midlayer_simulation",
            "pd_comparison",
        ):
            status = _run_real_stage(command_manifest, stage, logs_dir, subprocess_run)
            if status is not None:
                return _finish_failed_cell(command_manifest, status)
        return _finish_successful_real_cell(command_manifest, logs_dir)
    finally:
        _terminate_server_records(server_records)


def _start_server_process(
    command_manifest: dict[str, Any],
    stage: str,
    logs_dir: Path,
    subprocess_popen,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    command = command_manifest["commands"][stage]
    stdout_path = logs_dir / f"{stage}.stdout.log"
    stderr_path = logs_dir / f"{stage}.stderr.log"
    stdout_handle = stdout_path.open("w")
    stderr_handle = stderr_path.open("w")
    try:
        try:
            process = subprocess_popen(
                command["argv"],
                cwd=str(ROOT),
                env=_stage_environment(command_manifest, command),
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
        except FileNotFoundError as exc:
            stdout_handle.close()
            stderr_handle.close()
            status = _failure_status(
                stage,
                _missing_executable_result(exc),
                logs_dir,
                command,
            )
            return None, status
        runtime_servers = command_manifest["runtime"]["servers"]
        runtime_servers[stage] = {
            "pid": process.pid,
            "endpoint": command["endpoint"],
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
        return {
            "stage": stage,
            "process": process,
            "stdout_handle": stdout_handle,
            "stderr_handle": stderr_handle,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "command": command,
        }, None
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise


def _wait_for_server_readiness(
    command_manifest: dict[str, Any],
    server_record: dict[str, Any],
    stage: str,
    logs_dir: Path,
    *,
    timeout_seconds: float,
    poll_interval_seconds: float,
    sleep,
    readiness_probe,
) -> dict[str, Any] | None:
    if timeout_seconds < 0:
        raise ValueError("server readiness timeout cannot be negative")
    if poll_interval_seconds <= 0:
        raise ValueError("server readiness poll interval must be positive")
    command = server_record["command"]
    endpoint = command["endpoint"]
    health_url = endpoint["health_url"]
    readiness_stage = f"{stage}_readiness"
    elapsed = 0.0
    attempts = 0
    readiness = command_manifest["runtime"]["readiness"]

    while True:
        attempts += 1
        returncode = server_record["process"].poll()
        if returncode is not None:
            _flush_server_logs(server_record)
            status = _failure_status(
                stage,
                _process_result(
                    returncode,
                    server_record["stdout_path"],
                    server_record["stderr_path"],
                ),
                logs_dir,
                command,
            )
            readiness[stage] = {
                "status": "process_exited",
                "attempts": attempts,
                "elapsed_seconds": elapsed,
                "health_url": health_url,
            }
            return status

        if readiness_probe(health_url):
            readiness[stage] = {
                "status": "ready",
                "attempts": attempts,
                "elapsed_seconds": elapsed,
                "health_url": health_url,
            }
            _write_cell_manifest(command_manifest)
            return None

        if elapsed >= timeout_seconds:
            stderr = f"readiness timeout for {health_url}"
            _write_text(logs_dir / f"{readiness_stage}.stdout.log", "")
            _write_text(logs_dir / f"{readiness_stage}.stderr.log", stderr)
            readiness[stage] = {
                "status": "timeout",
                "attempts": attempts,
                "elapsed_seconds": elapsed,
                "health_url": health_url,
            }
            status = _failure_status(
                readiness_stage,
                _SyntheticResult(returncode=1, stdout="", stderr=stderr),
                logs_dir,
                command,
            )
            return status

        sleep(poll_interval_seconds)
        elapsed += poll_interval_seconds


def _flush_server_logs(server_record: dict[str, Any]) -> None:
    server_record["stdout_handle"].flush()
    server_record["stderr_handle"].flush()


def _server_exit_status(
    server_record: dict[str, Any],
    logs_dir: Path,
) -> dict[str, Any] | None:
    returncode = server_record["process"].poll()
    if returncode is None:
        return None
    _flush_server_logs(server_record)
    return _failure_status(
        server_record["stage"],
        _process_result(
            returncode,
            server_record["stdout_path"],
            server_record["stderr_path"],
        ),
        logs_dir,
        server_record["command"],
    )


def _terminate_server_process(server_record: dict[str, Any]) -> None:
    process = server_record["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    server_record["stdout_handle"].close()
    server_record["stderr_handle"].close()


def _terminate_server_records(server_records: list[dict[str, Any]]) -> None:
    for server_record in reversed(server_records):
        _terminate_server_process(server_record)
    server_records.clear()


def _http_readiness_probe(health_url: str) -> bool:
    try:
        with urllib_request.urlopen(health_url, timeout=2.0) as response:
            return 200 <= response.status < 300
    except (OSError, urllib_error.URLError):
        return False


def _run_real_stage(
    command_manifest: dict[str, Any],
    stage: str,
    logs_dir: Path,
    subprocess_run,
) -> dict[str, Any] | None:
    allowed_stages = {
        "vllm_benchmark",
        "colocated_midlayer_simulation",
        "disaggregated_midlayer_simulation",
        "comparison",
        "pd_comparison",
    }
    if stage not in allowed_stages:
        raise ManifestValidationError(
            f"unsupported real execution stage: {stage!r}")
    command = command_manifest["commands"][stage]
    try:
        result = subprocess_run(
            command["argv"],
            cwd=str(ROOT),
            env=_stage_environment(command_manifest, command),
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        result = _missing_executable_result(exc)
    stdout = _required_process_stream(result, "stdout")
    stderr = _required_process_stream(result, "stderr")
    _write_text(logs_dir / f"{stage}.stdout.log", stdout)
    _write_text(logs_dir / f"{stage}.stderr.log", stderr)
    if result.returncode != 0:
        return _failure_status(stage, result, logs_dir, command)
    if stage == "vllm_benchmark":
        ground_truth_status = _ensure_ground_truth_output(
            command_manifest,
            stdout,
            logs_dir,
            command,
        )
        if ground_truth_status is not None:
            return ground_truth_status
        return None
    elif stage in {"comparison", "pd_comparison"}:
        return _ensure_real_success_outputs(command_manifest, stage, logs_dir, command)
    elif stage in {
        "colocated_midlayer_simulation",
        "disaggregated_midlayer_simulation",
    }:
        return None
    else:
        raise ManifestValidationError(
            f"unhandled real execution stage: {stage!r}")


def _ensure_ground_truth_output(
    command_manifest: dict[str, Any],
    stdout: str,
    logs_dir: Path,
    command: dict[str, Any],
) -> dict[str, Any] | None:
    output_path = Path(command_manifest["outputs"]["ground_truth"])
    if output_path.exists():
        return None
    text = stdout.strip()
    if text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if payload is not None:
            _write_json(output_path, payload)
            return None
    result = _SyntheticResult(
        returncode=1,
        stdout=stdout,
        stderr=(
            "vLLM benchmark completed but did not write ground_truth.json "
            "or emit JSON stdout"
        ),
    )
    return _failure_status("vllm_benchmark", result, logs_dir, command)


def _ensure_real_success_outputs(
    command_manifest: dict[str, Any],
    stage: str,
    logs_dir: Path,
    command: dict[str, Any],
) -> dict[str, Any] | None:
    required = [
        command_manifest["outputs"]["ground_truth"],
        command_manifest["outputs"]["prediction"],
        command_manifest["outputs"]["comparison_report"],
        command_manifest["outputs"]["comparison_markdown"],
    ]
    missing = [path for path in required if not Path(path).exists()]
    if not missing:
        return None
    result = _SyntheticResult(
        returncode=1,
        stdout="",
        stderr="missing real-run output artifacts: " + ", ".join(missing),
    )
    return _failure_status(stage, result, logs_dir, command)


def _finish_successful_real_cell(
    command_manifest: dict[str, Any],
    logs_dir: Path,
) -> dict[str, Any]:
    output_root = Path(command_manifest["output_root"])
    status = {
        "status": "success",
        "cell_id": command_manifest["cell_id"],
        "logs": str(logs_dir),
        "ground_truth": command_manifest["outputs"]["ground_truth"],
        "prediction": command_manifest["outputs"]["prediction"],
        "comparison_report": command_manifest["outputs"]["comparison_report"],
        "comparison_markdown": command_manifest["outputs"]["comparison_markdown"],
        "reproducer_commands": _reproducer_commands(command_manifest),
    }
    command_manifest["status"] = "success"
    _write_json(output_root / "status.json", status)
    _write_cell_manifest(command_manifest)
    return command_manifest


def _finish_failed_cell(
    command_manifest: dict[str, Any],
    status: dict[str, Any],
) -> dict[str, Any]:
    output_root = Path(command_manifest["output_root"])
    status["cell_id"] = command_manifest["cell_id"]
    status["logs"] = command_manifest["outputs"]["logs"]
    command_manifest["status"] = status["status"]
    command_manifest["failed_stage"] = status["failed_stage"]
    _write_json(output_root / "status.json", status)
    _write_cell_manifest(command_manifest)
    return command_manifest


def validate_production_trace_outputs(
    command_manifest: dict[str, Any],
) -> None:
    if command_manifest["experiment_type"] != "production_trace":
        raise ManifestValidationError(
            "Production Trace output validation requires "
            "experiment_type='production_trace'")
    sweep_path = Path(command_manifest["outputs"]["sweep"])
    report_path = Path(command_manifest["outputs"]["report_markdown"])
    if not sweep_path.is_file():
        raise ValueError(f"Production Trace sweep output is missing: {sweep_path}")
    if not report_path.is_file():
        raise ValueError(
            f"Production Trace report output is missing: {report_path}")
    try:
        payload = json.loads(
            sweep_path.read_bytes(),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid Production Trace sweep JSON: {sweep_path}") from error
    if not isinstance(payload, dict) or "schema_version" not in payload:
        raise ValueError("Production Trace sweep must be a versioned object")
    if payload["schema_version"] == 2:
        _validate_production_trace_v2_output_schema(payload)
        identity_validator = _validate_production_trace_output_identity
    elif payload["schema_version"] == 3:
        identity_validator = (
            _validate_production_trace_aggregate_output_identity)
    else:
        raise ValueError("unsupported Production Trace sweep schema_version")
    expected_report = render_production_trace_sweep_report(payload)
    plot_dir = report_path.with_name(f"{report_path.stem}_plots")
    plot_paths = {
        "throughput": plot_dir / "production_trace_throughput.png",
        "ttft_p0_p100": plot_dir / "production_trace_ttft_p0_p100.png",
        "tpot_p0_p100": plot_dir / "production_trace_tpot_p0_p100.png",
        "simulation_speed": plot_dir / "production_trace_simulation_speed.png",
    }
    expected_plot_report = expected_report + "## Plots\n\n"
    for name, path in sorted(plot_paths.items()):
        relative_path = os.path.relpath(path, report_path.parent)
        expected_plot_report += f"- {name}: ![{name}]({relative_path})\n"
    expected_plot_report += "\n"
    actual_report = report_path.read_text(encoding="utf-8")
    if actual_report not in (expected_report, expected_plot_report):
        raise ValueError(
            "Production Trace report bytes disagree with canonical renderer")
    identity_validator(command_manifest, payload)


def _require_exact_output_object(
    value: object,
    expected_fields: set[str],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    supplied_fields = set(value)
    if supplied_fields != expected_fields:
        raise ValueError(
            f"{context} fields disagree with schema: "
            f"missing={sorted(expected_fields - supplied_fields)}, "
            f"unexpected={sorted(supplied_fields - expected_fields)}")
    return value


def _validate_production_trace_v2_output_schema(payload: object) -> None:
    sweep = _require_exact_output_object(
        payload,
        {
            "schema_version",
            "kind",
            "interpretation",
            "interactivity_tgs",
            "peak_selection",
            "saturation_analysis",
            "scale_order",
            "arrival_transform",
            "point_count",
            "arrival_rate_scales",
            "source_trace",
            "execution_provenance",
            "points",
        },
        context="Production Trace sweep",
    )
    points = sweep["points"]
    if not isinstance(points, list):
        raise ValueError("Production Trace sweep points must be a list")
    for point_index, point_value in enumerate(points):
        point_context = f"Production Trace sweep point {point_index}"
        point = _require_exact_output_object(
            point_value,
            {
                "arrival_rate_scale",
                "request_stream_sha256",
                "validation_status",
                "metrics",
                "result",
            },
            context=point_context,
        )
        _require_exact_output_object(
            point["validation_status"],
            {
                "trace_replay",
                "scheduler_control_plane",
                "performance_timing",
            },
            context=f"{point_context} validation status",
        )
        _require_exact_output_object(
            point["metrics"],
            {
                "request_throughput_per_s",
                "output_token_throughput_per_s",
                "token_throughput_per_s",
                "latency_distributions",
                "slo_goodput",
            },
            context=f"{point_context} summary metrics",
        )
        result = _require_exact_output_object(
            point["result"],
            {
                "version",
                "backend",
                "deployment_id",
                "hardware_ids",
                "model_compute_dtype",
                "model_config_provenance",
                "deployment",
                "deployment_plan",
                "routing_policy",
                "route_decision_version",
                "transfer_trace_version",
                "kv_handoff_mode",
                "kv_capacity",
                "kv_lifecycle_replay",
                "launch_policy",
                "prediction_method",
                "execution_mode",
                "execution_cache_stats",
                "performance_timing_status",
                "execution_provenance",
                "execution_artifact",
                "scheduler_runtime",
                "mtp_scenario",
                "mtp_acceptance_outcomes",
                "mtp_pd_handoff_contract",
                "arrival_process",
                "request_arrivals_s",
                "workload_hash",
                "num_steps",
                "num_transfers",
                "num_kernel_calls",
                "schedule_end_time_s",
                "route_decisions",
                "transfers",
                "metrics",
                "request_source",
            },
            context=f"{point_context} result",
        )
        _require_exact_output_object(
            result["request_source"],
            {
                "schema_version",
                "kind",
                "trace",
                "replay",
                "zero_output_contract",
                "prefix_alignment",
            },
            context=f"{point_context} request source",
        )
        _require_exact_output_object(
            result["metrics"],
            {
                "request_throughput_per_s",
                "output_token_throughput_per_s",
                "token_throughput_per_s",
                "latency_distributions",
                "slo_goodput",
                "system_resources",
                "critical_path",
                "requests",
            },
            context=f"{point_context} result metrics",
        )
        artifact = _require_exact_output_object(
            result["execution_artifact"],
            {
                "schema_version",
                "hash_method",
                "performance_timing_status",
                "counts",
                "schedule_trace",
                "steps",
                "transfers",
                "modeled_kv_offload",
                "artifact_sha256",
            },
            context=f"{point_context} execution artifact",
        )
        _require_exact_output_object(
            artifact["counts"],
            {
                "schedule_steps",
                "simulation_steps",
                "simulation_transfers",
                "trace_transfers",
                "modeled_kv_offload_transfers",
                "kernel_calls",
            },
            context=f"{point_context} execution artifact counts",
        )
        schedule_trace = _require_exact_output_object(
            artifact["schedule_trace"],
            {
                "route_decision_version",
                "transfer_trace_version",
                "entries",
                "route_decisions",
                "transfers",
            },
            context=f"{point_context} execution artifact schedule trace",
        )
        entries = schedule_trace["entries"]
        steps = artifact["steps"]
        transfers = artifact["transfers"]
        if (not isinstance(entries, list)
                or not isinstance(steps, list)
                or not isinstance(transfers, list)):
            raise ValueError(
                f"{point_context} execution artifact collections must be lists")
        for entry_index, entry in enumerate(entries):
            _require_exact_output_object(
                entry,
                {"step_id", "pool_name", "start_time_s", "end_time_s"},
                context=(f"{point_context} execution artifact schedule "
                         f"entry {entry_index}"),
            )
        for step_index, step in enumerate(steps):
            validated_step = _require_exact_output_object(
                step,
                {
                    "step_id",
                    "pool_name",
                    "schedule_interval",
                    "backend_step",
                    "workload_graph",
                    "lowering",
                    "kernel_plan",
                    "predictions",
                    "execution_timeline",
                },
                context=(f"{point_context} execution artifact step "
                         f"{step_index}"),
            )
            try:
                _backend_step_from_output_manifest(
                    validated_step["backend_step"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{point_context} execution artifact step {step_index} "
                    "has an invalid BackendStep") from error
            _validate_execution_components_output_schema(
                validated_step,
                context=(f"{point_context} execution artifact step "
                         f"{step_index}"),
            )
        for transfer_index, transfer in enumerate(transfers):
            validated_transfer = _require_exact_output_object(
                transfer,
                {
                    "request_id",
                    "source_engine_id",
                    "destination_engine_id",
                    "trace_transfer",
                    "workload_graph",
                    "lowering",
                    "kernel_plan",
                    "predictions",
                    "execution_timeline",
                },
                context=(f"{point_context} execution artifact transfer "
                         f"{transfer_index}"),
            )
            _validate_execution_components_output_schema(
                validated_transfer,
                context=(f"{point_context} execution artifact transfer "
                         f"{transfer_index}"),
            )


def _require_output_list(value: object, *, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _validate_execution_components_output_schema(
    owner: dict[str, Any],
    *,
    context: str,
) -> None:
    workload = owner["workload_graph"]
    if not isinstance(workload, dict):
        raise ValueError(f"{context} workload graph must be an object")
    workload_fields = {
        "node_count", "edge_count", "topological_order", "nodes", "edges",
    }
    compact_workload = "representation" in workload
    if compact_workload:
        workload_fields.update({
            "representation", "logical_node_count", "logical_edge_count",
            "block_repeats",
        })
    workload = _require_exact_output_object(
        workload,
        workload_fields,
        context=f"{context} workload graph",
    )
    for node_index, node in enumerate(_require_output_list(
            workload["nodes"], context=f"{context} workload nodes")):
        _require_exact_output_object(
            node,
            {
                "node_id", "kind", "pool_name", "shape", "dtype",
                "variant", "rank", "attributes",
            },
            context=f"{context} workload node {node_index}",
        )
    for edge_index, edge in enumerate(_require_output_list(
            workload["edges"], context=f"{context} workload edges")):
        _require_exact_output_object(
            edge,
            {"source", "destination", "kind"},
            context=f"{context} workload edge {edge_index}",
        )
    if compact_workload:
        if workload["representation"] != (
                "materialized_templates_with_logical_repeats_v1"):
            raise ValueError(f"{context} workload representation is invalid")
        for repeat_index, repeat in enumerate(_require_output_list(
                workload["block_repeats"],
                context=f"{context} workload block repeats")):
            repeat = _require_exact_output_object(
                repeat,
                {
                    "block_id", "node_ids", "repeat_count",
                    "feedback_edges", "trace_node_ids",
                },
                context=f"{context} workload block repeat {repeat_index}",
            )
            for edge_index, edge in enumerate(_require_output_list(
                    repeat["feedback_edges"],
                    context=(f"{context} workload block repeat "
                             f"{repeat_index} feedback edges"))):
                _require_exact_output_object(
                    edge,
                    {"source", "destination", "kind"},
                    context=(f"{context} workload block repeat "
                             f"{repeat_index} feedback edge {edge_index}"),
                )
            for trace_index, trace in enumerate(_require_output_list(
                    repeat["trace_node_ids"],
                    context=(f"{context} workload block repeat "
                             f"{repeat_index} trace nodes"))):
                _require_exact_output_object(
                    trace,
                    {"template_node_id", "logical_node_ids"},
                    context=(f"{context} workload block repeat "
                             f"{repeat_index} trace node {trace_index}"),
                )

    lowering = _require_exact_output_object(
        owner["lowering"],
        {"relation", "node_to_call_ids"},
        context=f"{context} lowering",
    )
    for mapping_index, mapping in enumerate(_require_output_list(
            lowering["node_to_call_ids"],
            context=f"{context} lowering mappings")):
        _require_exact_output_object(
            mapping,
            {"node_id", "call_ids"},
            context=f"{context} lowering mapping {mapping_index}",
        )

    kernel_plan = owner["kernel_plan"]
    if not isinstance(kernel_plan, dict):
        raise ValueError(f"{context} kernel plan must be an object")
    kernel_plan_fields = {"call_count", "topological_order", "calls"}
    compact_kernel_plan = "representation" in kernel_plan
    if compact_kernel_plan:
        kernel_plan_fields.update({
            "representation", "logical_call_count", "block_repeats",
        })
    if compact_kernel_plan != compact_workload:
        raise ValueError(
            f"{context} workload and KernelPlan compact forms disagree")
    kernel_plan = _require_exact_output_object(
        kernel_plan,
        kernel_plan_fields,
        context=f"{context} kernel plan",
    )
    for call_index, call_value in enumerate(_require_output_list(
            kernel_plan["calls"], context=f"{context} kernel calls")):
        call = _require_exact_output_object(
            call_value,
            {
                "call_id", "pool_name", "rank", "resource",
                "source_node_ids", "dependencies", "kernel_spec",
            },
            context=f"{context} kernel call {call_index}",
        )
        spec = _require_exact_output_object(
            call["kernel_spec"],
            {
                "kind", "hardware_id", "shape", "dtype", "variant",
                "launch", "transfer", "attributes",
            },
            context=f"{context} kernel call {call_index} spec",
        )
        launch = spec["launch"]
        if launch is not None:
            launch = _require_exact_output_object(
                launch,
                {
                    "algorithm", "grid", "block", "tiling", "cluster",
                    "parameters", "execution_variant",
                },
                context=f"{context} kernel call {call_index} launch",
            )
            if launch["tiling"] is not None:
                _require_exact_output_object(
                    launch["tiling"],
                    {
                        "tile_shape", "pipeline_depth", "split_k",
                        "num_kv_splits",
                    },
                    context=(f"{context} kernel call {call_index} launch "
                             "tiling"),
                )
        transfer = spec["transfer"]
        if transfer is not None:
            _require_exact_output_object(
                transfer,
                {
                    "algorithm", "bytes", "direction", "link", "topology",
                    "tier", "participants", "operation", "bytes_semantics",
                    "source_rank", "destination_rank", "ring_order",
                    "reduction", "backend", "resource_id",
                    "backend_config_sha256",
                },
                context=f"{context} kernel call {call_index} transfer",
            )
    if compact_kernel_plan:
        if kernel_plan["representation"] != (
                "materialized_templates_with_logical_repeats_v1"):
            raise ValueError(
                f"{context} KernelPlan representation is invalid")
        for repeat_index, repeat in enumerate(_require_output_list(
                kernel_plan["block_repeats"],
                context=f"{context} kernel block repeats")):
            repeat = _require_exact_output_object(
                repeat,
                {
                    "block_id", "call_ids", "repeat_count",
                    "feedback_dependencies",
                },
                context=f"{context} kernel block repeat {repeat_index}",
            )
            for dependency_index, dependency in enumerate(
                    _require_output_list(
                        repeat["feedback_dependencies"],
                        context=(f"{context} kernel block repeat "
                                 f"{repeat_index} feedback dependencies"))):
                _require_exact_output_object(
                    dependency,
                    {"destination_call_id", "source_call_ids"},
                    context=(f"{context} kernel block repeat {repeat_index} "
                             f"feedback dependency {dependency_index}"),
                )

    for prediction_index, prediction in enumerate(_require_output_list(
            owner["predictions"], context=f"{context} predictions")):
        _require_exact_output_object(
            prediction,
            {
                "call_id", "t_total_s", "t_launch_s", "bottleneck_group",
                "t_per_group_s", "t_per_resource_s",
            },
            context=f"{context} prediction {prediction_index}",
        )
    timeline = owner["execution_timeline"]
    if not isinstance(timeline, dict):
        raise ValueError(f"{context} execution timeline must be an object")
    timeline_fields = {
        "time_basis", "origin_time_s", "duration_s", "intervals",
    }
    compact_timeline = "representation" in timeline
    if compact_timeline:
        timeline_fields.update({
            "representation", "logical_interval_count",
            "call_occurrence_counts", "pool_activity",
            "reserved_resource_activity", "block_repeats",
        })
    if compact_timeline != compact_kernel_plan:
        raise ValueError(
            f"{context} KernelPlan and timeline compact forms disagree")
    timeline = _require_exact_output_object(
        timeline,
        timeline_fields,
        context=f"{context} execution timeline",
    )
    for interval_index, interval in enumerate(_require_output_list(
            timeline["intervals"], context=f"{context} timeline intervals")):
        _require_exact_output_object(
            interval,
            {
                "call_id", "pool_name", "rank", "resource",
                "reserved_resource_ids", "start_time_s", "end_time_s",
                "absolute_start_time_s", "absolute_end_time_s",
            },
            context=f"{context} timeline interval {interval_index}",
        )
    if compact_timeline:
        if timeline["representation"] != (
                "materialized_templates_with_logical_repeats_v1"):
            raise ValueError(f"{context} timeline representation is invalid")
        for count_index, count in enumerate(_require_output_list(
                timeline["call_occurrence_counts"],
                context=f"{context} timeline occurrence counts")):
            _require_exact_output_object(
                count,
                {"call_id", "occurrence_count"},
                context=(f"{context} timeline occurrence count "
                         f"{count_index}"),
            )
        for activity_field, identity_field in (
                ("pool_activity", "pool_name"),
                ("reserved_resource_activity", "resource_id")):
            for activity_index, activity in enumerate(_require_output_list(
                    timeline[activity_field],
                    context=f"{context} timeline {activity_field}")):
                activity = _require_exact_output_object(
                    activity,
                    {identity_field, "interval_count", "intervals"},
                    context=(f"{context} timeline {activity_field} "
                             f"{activity_index}"),
                )
                for activity_interval_index, interval in enumerate(
                        _require_output_list(
                            activity["intervals"],
                            context=(f"{context} timeline {activity_field} "
                                     f"{activity_index} intervals"))):
                    _require_exact_output_object(
                        interval,
                        {
                            "start_time_s", "end_time_s",
                            "absolute_start_time_s", "absolute_end_time_s",
                        },
                        context=(f"{context} timeline {activity_field} "
                                 f"{activity_index} interval "
                                 f"{activity_interval_index}"),
                    )
        for repeat_index, repeat in enumerate(_require_output_list(
                timeline["block_repeats"],
                context=f"{context} timeline block repeats")):
            repeat = _require_exact_output_object(
                repeat,
                {
                    "block_id", "repeat_count", "first_iteration_start_time_s",
                    "last_iteration_end_time_s", "call_intervals",
                },
                context=f"{context} timeline block repeat {repeat_index}",
            )
            for call_index, call in enumerate(_require_output_list(
                    repeat["call_intervals"],
                    context=(f"{context} timeline block repeat "
                             f"{repeat_index} calls"))):
                call = _require_exact_output_object(
                    call,
                    {"call_id", "intervals"},
                    context=(f"{context} timeline block repeat {repeat_index} "
                             f"call {call_index}"),
                )
                for call_interval_index, interval in enumerate(
                        _require_output_list(
                            call["intervals"],
                            context=(f"{context} timeline block repeat "
                                     f"{repeat_index} call {call_index} "
                                     "intervals"))):
                    _require_exact_output_object(
                        interval,
                        {
                            "start_time_s", "end_time_s",
                            "absolute_start_time_s", "absolute_end_time_s",
                        },
                        context=(f"{context} timeline block repeat "
                                 f"{repeat_index} call {call_index} interval "
                                 f"{call_interval_index}"),
                    )


def _expected_performance_timing_status(
    simulation: dict[str, Any],
) -> str:
    hardware = simulation["hardware"]
    return performance_timing_status(
        simulation["launch_policy"],
        (hardware["hardware_id"],),
    )


def _load_expected_hardware(expected_simulation, hardware_id, context) -> HWSpec:
    """Load the hardware the simulation was recorded against, and check it.

    The recorded config path and the expected hardware id must agree; a
    mismatch means the artifact being validated was produced for another card.
    """

    hardware = HWSpec(str(expected_simulation["hardware"]["config_path"]))
    if hardware.hardware_id != hardware_id:
        raise ValueError(f"{context} hardware config identity changed")
    return hardware


def _validate_production_trace_output_identity(
    command_manifest: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    expected_input = command_manifest["input"]
    if payload["source_trace"] != expected_input["trace"]:
        raise ValueError(
            "Production Trace sweep source identity disagrees with command manifest")
    if tuple(payload["arrival_rate_scales"]) != tuple(
            expected_input["arrival_rate_scales"]):
        raise ValueError(
            "Production Trace sweep scales disagree with command manifest")
    expected_model_sha256 = command_manifest["model"]["config_sha256"]
    try:
        expected_model = load_model_config(command_manifest["model"]["path"])
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "Production Trace command manifest model cannot be loaded") \
            from error
    if expected_model.config_sha256 != expected_model_sha256:
        raise ValueError(
            "Production Trace command manifest model content changed")
    expected_simulation = command_manifest["simulation"]
    expected_hardware = expected_simulation["hardware"]
    expected_timing_status = _expected_performance_timing_status(
        expected_simulation)
    if command_manifest["validation_status"]["performance_timing"] != (
            expected_timing_status):
        raise ValueError(
            "Production Trace command manifest performance timing status "
            "disagrees with launch policy and hardware")
    provenance_hardware = payload["execution_provenance"]["hardware_configs"]
    repository = payload["execution_provenance"]["repository"]
    if repository["git_commit"] != command_manifest["provenance"]["git_commit"]:
        raise ValueError(
            "Production Trace sweep code commit disagrees with command manifest")
    if repository["git_diff_hash"] != command_manifest["provenance"][
            "git_diff_hash"]:
        raise ValueError(
            "Production Trace sweep code diff identity disagrees with "
            "command manifest")
    hardware_path = Path(expected_hardware["config_path"]).resolve()
    hardware_payload = hardware_path.read_bytes()
    expected_hardware_provenance = {
        "path": str(hardware_path),
        "sha256": expected_hardware["config_sha256"],
        "size_bytes": len(hardware_payload),
    }
    if hashlib.sha256(hardware_payload).hexdigest() != (
            expected_hardware["config_sha256"]):
        raise ValueError(
            "Production Trace command manifest hardware content changed")
    if provenance_hardware["prefill"] != expected_hardware_provenance:
        raise ValueError(
            "Production Trace sweep hardware identity disagrees with command manifest")
    if expected_simulation["deployment"] in ("colocated", "af_disagg"):
        if provenance_hardware["decode"] is not None:
            raise ValueError(
                "non-PD Production Trace sweep unexpectedly used decode "
                "hardware")
    elif expected_simulation["deployment"] in (
            "pd_disagg", "pd_af_disagg"):
        if provenance_hardware["decode"] != expected_hardware_provenance:
            raise ValueError(
                "PD Production Trace sweep decode hardware identity "
                "disagrees with command manifest")
    else:
        raise ValueError(
            "Production Trace command manifest has an unsupported deployment")
    if tuple(provenance_hardware["additional_colocated"]):
        raise ValueError(
            "single-colocated Production Trace sweep unexpectedly used "
            "additional hardware")
    if tuple(payload["execution_provenance"]["loaded_hardware_ids"]) != (
            expected_hardware["hardware_id"],):
        raise ValueError(
            "Production Trace sweep loaded hardware IDs disagree with "
            "command manifest")
    if payload["execution_provenance"]["performance_timing_status"] != (
            expected_timing_status):
        raise ValueError(
            "Production Trace sweep performance timing provenance disagrees "
            "with command manifest")
    expected_replays = _production_trace_replays_from_command_input(
        expected_input)
    points = payload["points"]
    if (not isinstance(points, list)
            or len(points) != len(expected_input["arrival_rate_scales"])
            or payload["point_count"] != len(points)):
        raise ValueError(
            "Production Trace sweep point count disagrees with command manifest")
    expected_slo = command_manifest["analysis"]["slo_contract"]
    if expected_slo is None:
        expected_slo_contract = None
    else:
        try:
            expected_slo_contract = load_slo_contract(expected_slo["path"])
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                "Production Trace command manifest SLO contract cannot be "
                "loaded") from error
        if expected_slo_contract.source_sha256 != expected_slo[
                "source_sha256"]:
            raise ValueError(
                "Production Trace command manifest SLO contract changed")
    expected_tgs = command_manifest["analysis"][
        "interactivity_tgs_contract"]
    if expected_tgs is None:
        expected_tgs_contract = None
    else:
        try:
            expected_tgs_contract = load_interactivity_tgs_contract(
                expected_tgs["path"])
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(
                "Production Trace command manifest Interactivity--TGS "
                "contract cannot be loaded") from error
        if expected_tgs_contract.source_sha256 != expected_tgs[
                "source_sha256"]:
            raise ValueError(
                "Production Trace command manifest Interactivity--TGS "
                "contract changed")
    for point, expected_scale, expected_replay in zip(
            points,
            expected_input["arrival_rate_scales"],
            expected_replays,
            strict=True):
        if point["arrival_rate_scale"] != expected_scale:
            raise ValueError(
                "Production Trace sweep point scale disagrees with "
                "command manifest")
        expected_point_status = {
            "trace_replay": "validated_conservation",
            "scheduler_control_plane": "native_pinned_vllm",
            "performance_timing": expected_timing_status,
        }
        if point["validation_status"] != expected_point_status:
            raise ValueError(
                "Production Trace sweep point validation status disagrees "
                "with command manifest")
        _validate_production_trace_replay_binding(
            expected_input["trace"],
            expected_replay,
            point,
        )
        result = point["result"]
        _validate_production_trace_point_simulation_identity(
            expected_simulation,
            expected_model,
            command_manifest["model"]["path"],
            payload["execution_provenance"],
            result,
            expected_replay.requests,
            expected_slo_contract,
        )
        if result["model_config_provenance"]["config_sha256"] != (
                expected_model_sha256):
            raise ValueError(
                "Production Trace sweep model identity disagrees with "
                "command manifest")
        actual_slo = result["metrics"]["slo_goodput"]["contract"]
        if expected_slo is None:
            if actual_slo is not None:
                raise ValueError(
                    "Production Trace sweep unexpectedly used an SLO contract")
        elif actual_slo != _json_round_trip(asdict(expected_slo_contract)):
            raise ValueError(
                "Production Trace sweep SLO identity disagrees with "
                "command manifest")
    if expected_tgs_contract is None:
        expected_analysis = arrival_rate_sweep_analysis_manifest(None, None)
    else:
        expected_analysis_points = tuple(
            sweep_metric_point_from_manifest(
                point["arrival_rate_scale"],
                point["result"]["metrics"],
                expected_tgs_contract,
            )
            for point in points
        )
        expected_analysis = arrival_rate_sweep_analysis_manifest(
            expected_analysis_points,
            expected_tgs_contract,
        )
    expected_analysis_json = _json_round_trip(expected_analysis)
    for analysis_field in (
            "interactivity_tgs", "peak_selection", "saturation_analysis"):
        if payload[analysis_field] != expected_analysis_json[analysis_field]:
            raise ValueError(
                "Production Trace sweep Interactivity--TGS analysis "
                "disagrees with the command manifest contract and canonical "
                "point metrics")


def _production_trace_replays_from_command_input(
    expected_input: dict[str, Any],
) -> tuple[Any, ...]:
    expected_trace = expected_input["trace"]
    field_mapping = expected_trace["field_mapping"]
    success_filter = expected_trace["success_filter"]
    try:
        trace = load_production_trace(
            expected_trace["source_path"],
            ProductionTraceReaderConfig(
                source_format=expected_trace["source_format"],
                arrival_time_field=field_mapping["arrival_time"],
                input_tokens_field=field_mapping["input_tokens"],
                output_tokens_field=field_mapping["output_tokens"],
                prefix_hit_tokens_field=field_mapping[
                    "prefix_hit_tokens"],
                request_id_field=field_mapping["request_id"],
                input_time_unit=expected_trace["input_time_unit"],
                successful_request_filter=SuccessfulRequestFilter(
                    mode=success_filter["mode"],
                    field_name=success_filter["field_name"],
                    accepted_value=success_filter["accepted_value"],
                    provenance=success_filter["provenance"],
                ),
            ),
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "Production Trace command input cannot be independently "
            "reloaded") from error
    if production_trace_manifest(trace) != expected_trace:
        raise ValueError(
            "Production Trace command input changed after materialization")
    return tuple(
        build_production_trace_replay(
            trace,
            arrival_rate_scale=arrival_rate_scale,
        )
        for arrival_rate_scale in expected_input["arrival_rate_scales"]
    )


def _validate_production_trace_aggregate_output_identity(
    command_manifest: dict[str, Any],
    payload: dict[str, Any],
) -> None:
    """Bind aggregate-only persisted output to command inputs."""

    expected_input = command_manifest["input"]
    if payload["source_trace"] != expected_input["trace"]:
        raise ValueError(
            "aggregate Production Trace source identity disagrees with the "
            "command manifest")
    expected_scales = tuple(expected_input["arrival_rate_scales"])
    if tuple(payload["arrival_rate_scales"]) != expected_scales:
        raise ValueError(
            "aggregate Production Trace scales disagree with the command "
            "manifest")
    expected_replays = _production_trace_replays_from_command_input(
        expected_input)
    points = payload["points"]
    if (not isinstance(points, list)
            or len(points) != len(expected_replays)
            or payload["point_count"] != len(points)):
        raise ValueError(
            "aggregate Production Trace point count disagrees with the "
            "command manifest")
    expected_simulation = command_manifest["simulation"]
    expected_mtp = expected_simulation["mtp"]
    expected_model_sha256 = command_manifest["model"]["config_sha256"]
    try:
        expected_model = load_model_config(command_manifest["model"]["path"])
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "aggregate Production Trace command model cannot be loaded") \
            from error
    if expected_model.config_sha256 != expected_model_sha256:
        raise ValueError(
            "aggregate Production Trace command model content changed")
    expected_timing = _expected_performance_timing_status(
        expected_simulation)
    repository = payload["execution_provenance"]["repository"]
    if repository["git_commit"] != command_manifest["provenance"][
            "git_commit"]:
        raise ValueError(
            "aggregate Production Trace code commit identity disagrees with "
            "the command manifest")
    if repository["git_diff_hash"] != command_manifest["provenance"][
            "git_diff_hash"]:
        raise ValueError(
            "aggregate Production Trace code diff identity disagrees with the "
            "command manifest")
    for point, expected_scale, replay in zip(
            points, expected_scales, expected_replays, strict=True):
        expected_replay = production_trace_replay_manifest(replay)
        result = point["result"]
        request_source = result["request_source"]
        if (point["arrival_rate_scale"] != expected_scale
                or point["request_stream_sha256"]
                != replay.request_stream_sha256
                or request_source["trace"] != expected_input["trace"]
                or request_source["replay"] != expected_replay):
            raise ValueError(
                "aggregate Production Trace replay identity disagrees with "
                "the command manifest")
        if (result["model_config_provenance"]["config_sha256"]
                != expected_model_sha256
                or result["deployment"] != expected_simulation["deployment"]
                or result["mtp_scenario"] != expected_mtp
                or result["performance_timing_status"] != expected_timing):
            raise ValueError(
                "aggregate Production Trace simulation identity disagrees "
                "with the command manifest")
        binding = result["workload_assumption_binding"]
        if binding["request_stream_sha256"] != replay.request_stream_sha256:
            raise ValueError(
                "aggregate Production Trace workload binding disagrees with "
                "the replay stream")
        _validate_production_trace_aggregate_scheduler_runtime(
            result["scheduler_runtime"],
            expected_simulation,
            expected_model,
        )
        prefix = request_source["prefix_alignment"]["aggregate"]
        statistics = expected_input["trace"]["statistics"]
        if (prefix["request_count"] != statistics["request_count"]
                or prefix["trace_prefix_hit_tokens"]
                != statistics["prefix_hit_tokens_total"]):
            raise ValueError(
                "aggregate Production Trace prefix totals disagree with the "
                "source Trace")
        artifact = result["execution_artifact"]
        if (artifact["counts"]["simulation_steps"] != result["num_steps"]
                or artifact["counts"]["kernel_calls"]
                != result["num_kernel_calls"]
                or artifact["counts"]["trace_transfers"]
                != result["num_transfers"]):
            raise ValueError(
                "aggregate Production Trace execution counts are inconsistent")
    expected_payload = _json_round_trip(
        _rerun_production_trace_aggregate(command_manifest))
    recorded_runtime = _production_trace_runtime_measurements(payload)
    replay_runtime = _production_trace_runtime_measurements(expected_payload)
    if (_production_trace_deterministic_projection(payload)
            != _production_trace_deterministic_projection(expected_payload)
            or (recorded_runtime == replay_runtime
                and payload != expected_payload)):
        raise ValueError(
            "aggregate Production Trace output disagrees with deterministic "
            "replay")


def _production_trace_runtime_measurements(
    payload: dict[str, Any],
) -> tuple[object, ...]:
    """Return run-local measurements excluded from deterministic replay."""

    measurements = []
    for point in payload["points"]:
        result = point["result"]
        parallelism = result["execution_parallelism"]
        measurements.append((
            result["simulation_wall_time_s"],
            result["evaluation_wall_time_breakdown_s"],
            result["process_peak_rss_bytes"],
            parallelism["parallel_execution_wall_time_s"],
            parallelism["worker_aggregate_phase_s"],
        ))
    return tuple(measurements)


def _production_trace_deterministic_projection(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Remove wall-clock and RSS observations before replay comparison."""

    projected = _json_round_trip(payload)
    for point in projected["points"]:
        result = point["result"]
        result["persistence"]["full_in_memory_result_sha256"] = None
        result["simulation_wall_time_s"] = None
        result["evaluation_wall_time_breakdown_s"] = None
        result["process_peak_rss_bytes"] = None
        parallelism = result["execution_parallelism"]
        parallelism["parallel_execution_wall_time_s"] = None
        parallelism["worker_aggregate_phase_s"] = None
    return projected


def _rerun_production_trace_aggregate(
    command_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Rebuild a request-safe aggregate under the recorded environment."""

    simulation_command = command_manifest["commands"][
        "production_trace_simulation"]
    simulation_argv = simulation_command["argv"]
    if (not isinstance(simulation_argv, list)
            or len(simulation_argv) < 3
            or any(not isinstance(value, str) or not value
                   for value in simulation_argv)
            or Path(simulation_argv[1]).resolve()
            != (ROOT / "scripts" / "run_serving_simulation.py").resolve()):
        raise ValueError(
            "aggregate Production Trace simulation command is not canonical")
    output_indices = tuple(
        index for index, value in enumerate(simulation_argv)
        if value == "--output"
    )
    if len(output_indices) != 1:
        raise ValueError(
            "aggregate Production Trace simulation command must contain one "
            "--output")
    output_index = output_indices[0]
    if output_index + 1 >= len(simulation_argv):
        raise ValueError(
            "aggregate Production Trace simulation --output has no path")
    with tempfile.TemporaryDirectory(
            prefix="midlayer-production-trace-validation-") as directory:
        validation_output = Path(directory) / "aggregate.json"
        validation_argv = list(simulation_argv)
        validation_argv[output_index + 1] = str(validation_output)
        completed = subprocess.run(
            validation_argv,
            cwd=str(ROOT),
            env=_stage_environment(command_manifest, simulation_command),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                "aggregate Production Trace deterministic replay failed")
        if not validation_output.is_file():
            raise ValueError(
                "aggregate Production Trace deterministic replay did not "
                "write its aggregate")
        try:
            payload = json.loads(
                validation_output.read_bytes(),
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                "aggregate Production Trace deterministic replay is invalid "
                "JSON") from error
    if (not isinstance(payload, dict)
            or payload["schema_version"] != 3
            or payload["kind"] != "production_trace_arrival_rate_sweep"):
        raise ValueError(
            "aggregate Production Trace deterministic replay is not schema v3")
    return payload


def _validate_production_trace_aggregate_scheduler_runtime(
    scheduler_runtime: dict[str, Any],
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
) -> None:
    """Validate retained Scheduler identities without request-level replay."""

    deployment = expected_simulation["deployment"]
    scheduler = expected_simulation["scheduler"]
    mtp = expected_simulation["mtp"]
    if deployment == "colocated":
        expected_pools = (("main", "colocated", "prefill_tp_size"),)
    elif deployment == "af_disagg":
        expected_pools = (("attention", "attention", "prefill_tp_size"),)
    elif deployment == "pd_disagg":
        pd_disagg = expected_simulation["pd_disagg"]
        expected_pools = (
            *tuple(
                (f"prefill-{index}", "prefill", "prefill_tp_size")
                for index in range(pd_disagg["num_prefill_instances"])
            ),
            *tuple(
                (f"decode-{index}", "decode", "decode_tp_size")
                for index in range(pd_disagg["num_decode_instances"])
            ),
        )
    elif deployment == "pd_af_disagg":
        pd_disagg = expected_simulation["pd_disagg"]
        if pd_disagg["num_prefill_instances"] == 1:
            prefill_names = ("prefill-attention",)
        elif pd_disagg["num_prefill_instances"] == 2:
            prefill_names = ("prefill-attention-0", "prefill-attention-1")
        else:
            raise ValueError(
                "aggregate Production Trace has an unsupported prefill "
                "instance count")
        if pd_disagg["num_decode_instances"] == 1:
            decode_names = ("decode-attention",)
        elif pd_disagg["num_decode_instances"] == 2:
            decode_names = ("decode-attention-0", "decode-attention-1")
        else:
            raise ValueError(
                "aggregate Production Trace has an unsupported decode "
                "instance count")
        expected_pools = (
            *tuple(
                (name, "prefill-attention", "prefill_tp_size")
                for name in prefill_names
            ),
            *tuple(
                (name, "decode-attention", "decode_tp_size")
                for name in decode_names
            ),
        )
    else:
        raise ValueError(
            "aggregate Production Trace has an unsupported deployment")
    if set(scheduler_runtime) != {
            pool_name for pool_name, _role, _tp_field in expected_pools}:
        raise ValueError(
            "aggregate Production Trace Scheduler pools disagree with the "
            "command manifest")
    for pool_name, role, tensor_parallel_field in expected_pools:
        _validate_production_trace_scheduler_runtime(
            scheduler_runtime[pool_name],
            scheduler,
            expected_model,
            mtp,
            role=role,
            tensor_parallel_field=tensor_parallel_field,
        )


def _validate_production_trace_replay_binding(
    expected_trace: dict[str, Any],
    expected_replay: Any,
    point: dict[str, Any],
) -> None:
    expected_replay_manifest = production_trace_replay_manifest(
        expected_replay)
    if (point["request_stream_sha256"]
            != expected_replay.request_stream_sha256):
        raise ValueError(
            "Production Trace sweep request stream disagrees with the "
            "command Trace replay")
    result = point["result"]
    request_source = result["request_source"]
    if set(request_source) != {
            "schema_version", "kind", "trace", "replay",
            "zero_output_contract", "prefix_alignment"}:
        raise ValueError(
            "Production Trace sweep request source fields disagree with "
            "the canonical contract")
    if (request_source["schema_version"] != 1
            or request_source["kind"] != "production_trace"
            or request_source["trace"] != expected_trace
            or request_source["replay"] != expected_replay_manifest
            or result["arrival_process"] != expected_replay_manifest):
        raise ValueError(
            "Production Trace sweep Scheduler input disagrees with the "
            "command Trace replay")
    expected_arrivals = [
        {
            "request_id": request.request_id,
            "arrival_time_s": request.arrival_time_s,
        }
        for request in expected_replay.requests
    ]
    if result["request_arrivals_s"] != expected_arrivals:
        raise ValueError(
            "Production Trace sweep request arrivals disagree with the "
            "command Trace replay")


def _validate_production_trace_point_simulation_identity(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    shared_execution_provenance: dict[str, Any],
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    for field in ("backend", "deployment", "prediction_method", "launch_policy"):
        if result[field] != expected_simulation[field]:
            raise ValueError(
                f"Production Trace sweep {field} disagrees with command manifest")
    expected_timing_status = _expected_performance_timing_status(
        expected_simulation)
    if result["performance_timing_status"] != expected_timing_status:
        raise ValueError(
            "Production Trace sweep performance timing status disagrees with "
            "command manifest")
    if result["execution_artifact"]["performance_timing_status"] != (
            expected_timing_status):
        raise ValueError(
            "Production Trace execution artifact performance timing status "
            "disagrees with command manifest")
    if result["mtp_scenario"] != expected_simulation["mtp"]:
        raise ValueError(
            "Production Trace sweep MTP scenario disagrees with command "
            "manifest")
    expert_routing = expected_simulation["expert_routing"]
    expected_execution_mode = (
        "reference"
        if (expert_routing is not None
            and expert_routing["source_kind"] == "replay_observation")
        else "accelerated"
    )
    if result["execution_mode"] != expected_execution_mode:
        raise ValueError(
            "Production Trace execution mode disagrees with command manifest")
    cache_stats = _require_exact_output_object(
        result["execution_cache_stats"],
        {
            "execution_mode", "block_construction", "expert_routing",
            "kernel_lowering", "kernel_prediction",
        },
        context="Production Trace execution cache stats",
    )
    if cache_stats["execution_mode"] != expected_execution_mode:
        raise ValueError(
            "Production Trace execution cache mode disagrees with result")
    for cache_name in (
            "block_construction", "expert_routing", "kernel_lowering",
            "kernel_prediction"):
        cache = cache_stats[cache_name]
        if expected_execution_mode == "reference" and cache is not None:
            raise ValueError(
                "reference Production Trace execution cannot carry cache "
                "statistics")
        if expected_execution_mode == "accelerated":
            cache = _require_exact_output_object(
                cache,
                {"entries", "hits", "misses"},
                context=f"Production Trace {cache_name} cache stats",
            )
            if any(
                    isinstance(cache[field], bool)
                    or not isinstance(cache[field], int)
                    or cache[field] < 0
                    for field in ("entries", "hits", "misses")):
                raise ValueError(
                    f"Production Trace {cache_name} cache stats are invalid")
    execution_provenance = result["execution_provenance"]
    if execution_provenance != shared_execution_provenance:
        raise ValueError(
            "Production Trace sweep point execution provenance disagrees with "
            "the shared sweep provenance")
    runtime_inputs = execution_provenance["runtime_inputs"]
    if (expected_simulation["deployment"] in ("af_disagg", "pd_af_disagg")
            and set(runtime_inputs) != {
                "expert_routing", "attention_kv_layout",
                "attention_kv_distribution", "graph_execution",
            }):
        raise ValueError(
            "AF-bearing Production Trace runtime input fields disagree with the "
            "canonical contract")
    for field in ("attention_kv_layout", "attention_kv_distribution"):
        if runtime_inputs[field] != expected_simulation[field]:
            raise ValueError(
                f"Production Trace sweep {field} disagrees with command manifest")
    if (runtime_inputs["graph_execution"]
            != GraphExecutionConfig.disabled().manifest()):
        raise ValueError(
            "Production Trace sweep unexpectedly enabled graph execution")
    if (expected_simulation["deployment"] in ("af_disagg", "pd_af_disagg")
            and runtime_inputs["expert_routing"]
            != _json_round_trip(ExpertRoutingInput.not_applicable().manifest())):
        raise ValueError(
            "AF-bearing Production Trace expert routing runtime input must be "
            "not_applicable")
    if execution_provenance["launch_inputs"]["launch_policy"] != (
            expected_simulation["launch_policy"]):
        raise ValueError(
            "Production Trace sweep launch provenance disagrees with "
            "command manifest")
    if execution_provenance["prediction_method"] != expected_simulation[
            "prediction_method"]:
        raise ValueError(
            "Production Trace sweep prediction provenance disagrees with "
            "command manifest")
    expected_hardware_id = expected_simulation["hardware"]["hardware_id"]
    if tuple(result["hardware_ids"]) != (expected_hardware_id,):
        raise ValueError(
            "Production Trace sweep point hardware IDs disagree with "
            "command manifest")
    if expected_simulation["deployment"] == "colocated":
        _validate_colocated_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
    elif expected_simulation["deployment"] == "pd_disagg":
        _validate_pd_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
    elif expected_simulation["deployment"] == "af_disagg":
        _validate_af_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
    elif expected_simulation["deployment"] == "pd_af_disagg":
        _validate_pd_af_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
    else:
        raise ValueError(
            "Production Trace sweep point has an unsupported deployment")


def _validate_colocated_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    if (expected_simulation["pd_disagg"] is not None
            or expected_simulation["af_disagg"] is not None):
        raise ValueError(
            "colocated Production Trace command retained disaggregated "
            "configuration")
    scheduler_runtime = result["scheduler_runtime"]
    if set(scheduler_runtime) != {"main"}:
        raise ValueError(
            "Production Trace sweep must contain exactly the main Scheduler "
            "runtime")
    runtime = scheduler_runtime["main"]
    expected_scheduler = expected_simulation["scheduler"]
    _validate_production_trace_scheduler_runtime(
        runtime,
        expected_scheduler,
        expected_model,
        expected_simulation["mtp"],
        role="colocated",
        tensor_parallel_field="prefill_tp_size",
    )
    parallelism = (
        expected_scheduler["prefill_tp_size"],
        expected_scheduler["prefill_pp_size"],
    )
    if parallelism == (1, 1):
        _validate_colocated_tp1_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
        return
    if parallelism == (2, 1):
        if expected_scheduler["enable_expert_parallel"]:
            _validate_colocated_ep2_production_trace_point(
                expected_simulation,
                expected_model,
                expected_model_path,
                result,
                expected_requests,
                expected_slo_contract,
            )
        else:
            _validate_colocated_tp2_production_trace_point(
                expected_simulation,
                expected_model,
                expected_model_path,
                result,
                expected_requests,
                expected_slo_contract,
            )
    elif parallelism == (1, 2):
        _validate_colocated_pp2_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
        )
    elif parallelism == (2, 2):
        _validate_colocated_tp2_pp2_production_trace_point(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
        )
    else:
        raise AssertionError(
            "validated colocated Production Trace parallelism vanished")


def _validate_production_trace_scheduler_runtime(
    runtime: dict[str, Any],
    expected_scheduler: dict[str, Any],
    expected_model: ModelConfig,
    expected_mtp: dict[str, Any],
    *,
    role: str,
    tensor_parallel_field: str,
) -> None:
    expected_runtime_fields = {
        field.name for field in fields(VLLMSchedulerRuntimeInfo)
    }
    if set(runtime) != expected_runtime_fields:
        raise ValueError(
            "Production Trace sweep Scheduler runtime fields disagree with "
            "the pinned contract")
    scheduler_fields = {
        "max_num_seqs": "max_requests",
        "max_num_batched_tokens": "max_batched_tokens",
        "kv_block_size_tokens": "kv_block_size_tokens",
        "max_model_len": "max_model_len",
        "requested_tensor_parallel_size": tensor_parallel_field,
        "effective_tensor_parallel_size": tensor_parallel_field,
        "requested_pipeline_parallel_size": "prefill_pp_size",
        "effective_pipeline_parallel_size": "prefill_pp_size",
        "requested_expert_parallel": "enable_expert_parallel",
        "effective_expert_parallel": "enable_expert_parallel",
    }
    for runtime_field, manifest_field in scheduler_fields.items():
        if runtime[runtime_field] != expected_scheduler[manifest_field]:
            raise ValueError(
                "Production Trace sweep Scheduler identity disagrees with "
                f"command manifest: {runtime_field}")
    if (expected_scheduler["capacity_mode"]
            == "debug_explicit_block_count"
            and runtime["num_kv_blocks"]
            != expected_scheduler["num_kv_blocks"]):
        raise ValueError(
            "Production Trace sweep Scheduler debug capacity disagrees with "
            "command manifest")
    if (isinstance(runtime["num_kv_blocks"], bool)
            or not isinstance(runtime["num_kv_blocks"], int)
            or runtime["num_kv_blocks"] < 1):
        raise ValueError(
            "Production Trace sweep Scheduler capacity must be positive")
    if runtime["model_config_sha256"] != expected_model.config_sha256:
        raise ValueError(
            "Production Trace sweep Scheduler model identity disagrees with "
            "command manifest")
    source_identity = vllm_scheduler_source_identity()
    actual_source_hashes = tuple(
        tuple(record) for record in runtime["source_hashes"])
    if (runtime["version"] != source_identity["version"]
            or runtime["source_commit"] != source_identity["source_commit"]
            or actual_source_hashes != source_identity["source_hashes"]
            or runtime["package_python_file_count"]
            != source_identity["package_python_file_count"]
            or runtime["package_python_manifest_sha256"]
            != source_identity["package_python_manifest_sha256"]):
        raise ValueError(
            "Production Trace sweep Scheduler source identity disagrees "
            "with the pinned runtime")
    if runtime["backend_role"] != role:
        raise ValueError(
            "Production Trace sweep Scheduler role disagrees with command "
            "manifest")
    if expected_mtp["acceptance_policy"] == "disabled":
        expected_speculative_method = "disabled"
        expected_bootstrap = "version_pinned_scheduler_only_cpu"
    elif expected_mtp["acceptance_policy"] in (
            "fixed_count", "maximum_entropy"):
        expected_speculative_method = "mtp"
        expected_bootstrap = "version_pinned_engine_args_mtp_cpu"
    else:
        raise ValueError(
            "Production Trace command manifest has an unsupported MTP "
            "acceptance policy")
    if runtime["speculative_method"] != expected_speculative_method:
        raise ValueError(
            "Production Trace sweep Scheduler speculative method disagrees "
            "with command manifest")
    expected_tp_size = expected_scheduler[tensor_parallel_field]
    expected_pp_size = expected_scheduler["prefill_pp_size"]
    try:
        expected_kv_heads = expected_model.kv_heads_per_tensor_parallel_rank(
            expected_tp_size)
        expected_kv_bytes = (
            expected_model.kv_bytes_per_token_per_parallel_rank(
                expected_tp_size,
                expected_pp_size,
            ))
    except ValueError as error:
        raise ValueError(
            "Production Trace Scheduler model parallelism is invalid") \
            from error
    expected_local_layers = (
        expected_model.num_hidden_layers // expected_pp_size)
    expected_page_size_bytes = (
        expected_scheduler["kv_block_size_tokens"]
        * expected_kv_bytes // expected_local_layers)
    owners = runtime["scheduler_kv_cache_tensor_owners"]
    if (runtime["platform"] != "cpu"
            or runtime["requested_dtype"] != "bfloat16"
            or runtime["effective_dtype"] != "bfloat16"
            or runtime["device_guard"]
            != "python_tripwire+cuda_state+procfs"
            or runtime["bootstrap"]
            != expected_bootstrap
            or runtime["requested_chunked_prefill"] is not True
            or runtime["effective_chunked_prefill"] is not True
            or runtime["requested_prefix_caching"] is not False
            or runtime["effective_prefix_caching"] is not False
            or runtime["prefix_oracle_mode"]
            != "trace_length_private_nonshared_v1"
            or runtime["scheduler_pipeline_stage_rank"] != 0
            or runtime["scheduler_pipeline_stage_layer_start"] != 0
            or runtime["scheduler_pipeline_stage_layer_end"]
            != expected_local_layers
            or runtime["scheduler_kv_cache_num_heads_per_rank"]
            != expected_kv_heads
            or runtime["scheduler_kv_cache_page_size_bytes"]
            != expected_page_size_bytes
            or len(owners) != expected_local_layers
            or any(not isinstance(owner, list)
                   or len(owner) != 1
                   or not isinstance(owner[0], str)
                   or not owner[0]
                   for owner in owners)
            or runtime["max_num_partial_prefills"] != 1
            or runtime["max_long_partial_prefills"] != 1
            or runtime["long_prefill_token_threshold"] != 0
            or runtime["scheduler_policy"] != "fcfs"
            or runtime["num_lookahead_slots"]
            != expected_mtp["num_speculative_tokens"]
            or runtime["attention_kv_layout"] != "paged_varlen"):
        raise ValueError(
            "Production Trace sweep Scheduler runtime or KV geometry "
            "disagrees with the pinned contract")
    expected_mtp_runtime = {
        "mtp_num_speculative_tokens": expected_mtp[
            "num_speculative_tokens"],
        "mtp_acceptance_policy": expected_mtp["acceptance_policy"],
        "mtp_fixed_accepted_tokens": expected_mtp[
            "fixed_accepted_tokens"],
        "mtp_acceptance_source_id": expected_mtp[
            "acceptance_source_id"],
        "mtp_acceptance_replay_path": None,
        "mtp_acceptance_replay_sha256": None,
        "mtp_acceptance_replay_artifact_id": None,
        "mtp_acceptance_replay_step_count": None,
        "mtp_acceptance_assumption_sha256": None,
        "mtp_request_stream_sha256": None,
    }
    if expected_mtp["acceptance_policy"] == "disabled":
        expected_mtp_runtime.update({
            "mtp_padded_drafter_batch": False,
            "mtp_draft_model_type": "disabled",
            "mtp_num_mtp_layers": 0,
            "mtp_draft_max_model_len": 0,
        })
    elif expected_mtp["acceptance_policy"] in (
            "fixed_count", "maximum_entropy"):
        expected_mtp_runtime.update({
            "mtp_padded_drafter_batch": True,
            "mtp_draft_model_type": "glm4_moe_mtp",
            "mtp_num_mtp_layers": expected_model.num_mtp_layers,
            "mtp_draft_max_model_len": expected_scheduler["max_model_len"],
        })
        if expected_mtp["acceptance_policy"] == "maximum_entropy":
            assumption = expected_mtp["acceptance_assumption"]
            expected_mtp_runtime.update({
                "mtp_acceptance_assumption_sha256": assumption[
                    "policy_descriptor_sha256"],
                "mtp_request_stream_sha256": assumption[
                    "request_stream_sha256"],
            })
    else:
        raise AssertionError("validated MTP acceptance policy vanished")
    if any(
            runtime[field] != value
            for field, value in expected_mtp_runtime.items()):
        raise ValueError(
            "Production Trace sweep Scheduler MTP runtime disagrees with "
            "command manifest")
    if role in ("colocated", "attention"):
        expected_handoff_mode = "not_applicable"
    elif role in (
            "prefill", "decode", "prefill-attention", "decode-attention"):
        expected_handoff_mode = "metadata_only_private_destination_kv"
    else:
        raise ValueError(
            f"unknown Production Trace Scheduler role: {role!r}")
    if runtime["kv_handoff_mode"] != expected_handoff_mode:
        raise ValueError(
            "Production Trace sweep Scheduler KV handoff mode disagrees "
            "with command manifest")


def _explicit_expert_routing_input(
    routing_config: object,
    context: str,
) -> ExpertRoutingInput:
    if not isinstance(routing_config, dict):
        raise ValueError(
            f"{context} command manifest lost explicit expert routing")
    uniform_fields = {"source_kind", "source_id", "root_seed"}
    skew_fields = uniform_fields | {
        "hot_expert_count", "hot_expert_weight"}
    if (set(routing_config) not in (uniform_fields, skew_fields)
            or routing_config["source_kind"] != "explicit_assumption"):
        raise ValueError(
            f"{context} command manifest lost explicit expert routing")
    if set(routing_config) == uniform_fields:
        return ExpertRoutingInput.seeded_assumption(
            routing_config["source_id"],
            routing_config["root_seed"],
        )
    return ExpertRoutingInput.skewed_assumption(
        routing_config["source_id"],
        routing_config["root_seed"],
        hot_expert_count=routing_config["hot_expert_count"],
        hot_expert_weight=routing_config["hot_expert_weight"],
    )


def _maximum_entropy_mtp_assumption(
    scenario: dict[str, Any],
) -> MTPAcceptanceAssumption | None:
    """Reconstruct a canonical assumption for independent replay."""

    if scenario["acceptance_policy"] != "maximum_entropy":
        return None
    manifest = scenario["acceptance_assumption"]
    assumption = MTPAcceptanceAssumption(
        manifest["source_id"],
        manifest["proposal_depth"],
        manifest["mean_acceptance_rate"],
        manifest["root_seed"],
        manifest["request_stream_sha256"],
    )
    if assumption.manifest() != manifest:
        raise ValueError(
            "Production Trace maximum-entropy MTP assumption is not "
            "canonical")
    return assumption


def _validate_colocated_tp1_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    mtp = expected_simulation["mtp"]
    mtp_enabled = mtp["acceptance_policy"] != "disabled"
    replay_identity = expected_simulation["kv_lifecycle_replay"]
    if replay_identity is None:
        kv_lifecycle_replay = None
    else:
        try:
            kv_lifecycle_replay = load_kv_lifecycle_replay(
                replay_identity["path"])
        except (OSError, TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                "TP=PP=1 Host-KV Production Trace replay cannot be "
                "independently reloaded") from error
        if (kv_lifecycle_replay.source_sha256
                != replay_identity["source_sha256"]
                or kv_lifecycle_replay.artifact_id
                != replay_identity["artifact_id"]):
            raise ValueError(
                "TP=PP=1 Host-KV Production Trace replay identity changed")
    if kv_lifecycle_replay is not None:
        context = "TP=PP=1 Host-KV Production Trace"
    elif mtp_enabled:
        context = "TP=PP=1 GLM4-MoE MTP Production Trace"
    else:
        context = "TP=PP=1 Production Trace"
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(),
                kv_cache=(
                    None
                    if kv_lifecycle_replay is None
                    else kv_lifecycle_replay.cache_config
                ),
            ),),
        ),
        num_layers=expected_model.num_hidden_layers,
        num_experts=expected_model.num_experts,
    )
    expected_plan = _json_round_trip(deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            f"{context} worker-pool topology disagrees with command manifest")

    scheduler = expected_simulation["scheduler"]
    runtime = result["scheduler_runtime"]["main"]
    expected_owners = [
        [f"model.layers.{layer_index}.self_attn.attn"]
        for layer_index in range(expected_model.num_hidden_layers)
    ]
    if runtime["scheduler_kv_cache_tensor_owners"] != expected_owners:
        raise ValueError(
            f"{context} Scheduler KV owners disagree with the pinned model")
    _validate_production_trace_capacity(
        result,
        scheduler,
        ("main",),
        context=context,
    )
    expected_kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(1, 1))
    capacity_plan = result["kv_capacity"]["scheduler_pools"][0]
    if capacity_plan["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError(f"{context} KV capacity disagrees with the model")
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["route_decisions"] != []
            or result["transfers"] != []
            or result["num_transfers"] != 0):
        raise ValueError(f"{context} routing or transfer state is invalid")
    if kv_lifecycle_replay is not None:
        _validate_colocated_kv_lifecycle_replay(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
            deployment,
            runtime,
            kv_lifecycle_replay,
            tensor_parallel_size=1,
            context=context,
        )
        return
    if (result["kv_lifecycle_replay"] is not None
            or result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            }):
        raise ValueError(f"{context} unexpectedly enabled KV offload")
    if (result["mtp_scenario"] != mtp
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(
            f"{context} MTP scenario disagrees with command manifest")

    routing_config = expected_simulation["expert_routing"]
    if mtp_enabled:
        expert_routing = _explicit_expert_routing_input(
            routing_config, context)
    else:
        if routing_config is not None:
            raise ValueError(
                f"{context} unexpectedly enabled expert routing")
        expert_routing = ExpertRoutingInput.not_applicable()
    runtime_inputs = RuntimeInputs(
        expert_routing,
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    expected_runtime_inputs = {
        "expert_routing": _json_round_trip(
            expert_routing.manifest()),
        "attention_kv_layout": expected_simulation["attention_kv_layout"],
        "attention_kv_distribution": expected_simulation[
            "attention_kv_distribution"],
        "graph_execution": runtime_inputs.graph_execution.manifest(),
    }
    if result["execution_provenance"]["runtime_inputs"] != (
            expected_runtime_inputs):
        raise ValueError(
            f"{context} runtime inputs disagree with the command manifest")

    artifact = result["execution_artifact"]
    artifact_trace = artifact["schedule_trace"]
    if (artifact_trace["route_decision_version"] != 2
            or artifact_trace["transfer_trace_version"] != 2
            or artifact_trace["route_decisions"] != []
            or artifact_trace["transfers"] != []
            or artifact["transfers"] != []):
        raise ValueError(
            f"{context} execution artifact contains a route or transfer")
    steps = artifact["steps"]
    counts = artifact["counts"]
    if (not steps
            or result["num_steps"] != len(steps)
            or counts["schedule_steps"] != len(steps)
            or counts["simulation_steps"] != len(steps)
            or counts["simulation_transfers"] != 0
            or counts["trace_transfers"] != 0
            or counts["modeled_kv_offload_transfers"] != 0):
        raise ValueError(f"{context} execution artifact counts disagree")
    trace_entries = artifact_trace["entries"]
    entries_by_id = {
        entry["step_id"]: entry for entry in trace_entries
    }
    if (len(entries_by_id) != len(trace_entries)
            or len(entries_by_id) != len(steps)):
        raise ValueError(f"{context} schedule trace coverage disagrees")

    hardware = _load_expected_hardware(
        expected_simulation, hardware_id, context)
    hardware_by_id = {hardware_id: hardware}
    total_call_count = 0
    worker_pool_intervals = []
    all_step_ids = set()
    all_call_ids = set()
    backend_steps_by_id = {}
    simulation_steps = []
    execution = ServingExecutionContext(
        expected_model,
        deployment,
        hardware_by_id,
        expected_simulation["launch_policy"],
        expected_simulation["prediction_method"],
        runtime_inputs,
        execution_mode=result["execution_mode"],
    )
    for step in steps:
        step_id = step["step_id"]
        if (step_id in all_step_ids
                or step_id not in entries_by_id
                or step["pool_name"] != "main"):
            raise ValueError(f"{context} execution step identity is invalid")
        all_step_ids.add(step_id)
        entry = entries_by_id[step_id]
        if (entry["pool_name"] != "main"
                or step["schedule_interval"] != {
                    "start_time_s": entry["start_time_s"],
                    "end_time_s": entry["end_time_s"],
                }):
            raise ValueError(f"{context} schedule interval identity is invalid")
        try:
            backend_step = _backend_step_from_output_manifest(
                step["backend_step"])
            components = execution.execute_step(
                backend_step.schedule,
                kv_cache_events=backend_step.kv_cache_events,
            )
            workload = components.workload
            kernel_plan = components.kernel_plan
            predictions = components.predictions
            timeline = components.execution_timeline
        except (TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                f"{context} step cannot be canonically rematerialized") \
                from error
        backend_steps_by_id[step_id] = backend_step
        expected_components = _json_round_trip(
            execution_step_components_manifest(
                workload,
                kernel_plan,
                predictions,
                timeline,
                origin_time_s=step["schedule_interval"]["start_time_s"],
            ))
        actual_components = {
            field: step[field]
            for field in expected_components
        }
        if actual_components != expected_components:
            raise ValueError(
                f"{context} canonical Scheduler-to-KernelSpec "
                "rematerialization disagrees with the execution artifact")
        _validate_canonical_production_trace_step_duration(
            step,
            timeline.duration_s,
            context=context,
        )
        simulation_steps.append(SimulationStep(
            backend_step,
            workload,
            kernel_plan,
            predictions,
            timeline,
        ))
        calls = step["kernel_plan"]["calls"]
        call_ids = tuple(call["call_id"] for call in calls)
        allowed_kernel_kinds = {"VECTOR", "GEMM", "FLASH_ATTN"}
        if mtp_enabled:
            allowed_kernel_kinds.add("FUSED_MOE")
        if (not calls
                or all_call_ids.intersection(call_ids)
                or {call["rank"] for call in calls} != {0}
                or any(
                    call["pool_name"] != "main"
                    or call["kernel_spec"]["hardware_id"] != hardware_id
                    or call["kernel_spec"]["kind"]
                    not in allowed_kernel_kinds
                    for call in calls
                )):
            raise ValueError(
                f"{context} rank-local compute coverage is invalid")
        if mtp_enabled:
            node_ids = tuple(
                node["node_id"] for node in step["workload_graph"]["nodes"])
            if any(
                    not any(f":mtp-pass-{pass_index}:" in node_id
                            for node_id in node_ids)
                    for pass_index in range(mtp["num_speculative_tokens"])):
                raise ValueError(
                    f"{context} MTP proposer workload coverage is invalid")
        all_call_ids.update(call_ids)
        intervals = step["execution_timeline"]["intervals"]
        intervals_by_id = {
            interval["call_id"]: interval for interval in intervals
        }
        if set(intervals_by_id) != set(call_ids):
            raise ValueError(f"{context} timeline coverage is invalid")
        worker_pool_intervals.extend(
            (
                interval["absolute_start_time_s"],
                interval["absolute_end_time_s"],
            )
            for interval in intervals
        )
        total_call_count += len(calls)
    if (set(entries_by_id) != all_step_ids
            or counts["kernel_calls"] != total_call_count
            or result["num_kernel_calls"] != total_call_count):
        raise ValueError(f"{context} kernel or step counts disagree")

    canonical_schedule_trace = ScheduleTrace(
        ScheduleTraceEntry(
            backend_steps_by_id[entry["step_id"]],
            entry["start_time_s"],
            entry["end_time_s"],
        )
        for entry in trace_entries
    )
    _validate_production_trace_scheduler_binding(
        expected_requests,
        canonical_schedule_trace,
        scheduler["kv_block_size_tokens"],
        result,
        max_requests=scheduler["max_requests"],
        context=context,
    )
    _replay_pinned_parallel_scheduler(
        expected_requests,
        canonical_schedule_trace,
        scheduler,
        expected_model,
        expected_model_path,
        runtime,
        num_kv_blocks=runtime["num_kv_blocks"],
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        expected_mtp=mtp,
        context=context,
    )

    if mtp_enabled:
        expected_mtp_outcomes = {
            "schema_version": 1,
            "source_kind": "explicit_assumption",
            "source_id": mtp["acceptance_source_id"],
            "steps": mtp_acceptance_steps_manifest(
                mtp_acceptance_step_from_backend_step(
                    step_index,
                    backend_steps_by_id[entry["step_id"]],
                )
                for step_index, entry in enumerate(trace_entries)
            ),
        }
    else:
        expected_mtp_outcomes = None
    if result["mtp_acceptance_outcomes"] != expected_mtp_outcomes:
        raise ValueError(
            f"{context} MTP acceptance outcomes disagree with canonical "
            "Scheduler steps")

    canonical_metrics = compute_serving_metrics(
        expected_requests,
        canonical_schedule_trace,
    )
    canonical_result = SimulationResult(
        canonical_schedule_trace,
        canonical_metrics,
        tuple(simulation_steps),
    )
    expected_prefix_alignment = _json_round_trip(prefix_alignment_manifest(
        expected_requests,
        canonical_result,
        scheduler["kv_block_size_tokens"],
    ))
    if result["request_source"]["prefix_alignment"] != (
            expected_prefix_alignment):
        raise ValueError(
            f"{context} Prefix alignment disagrees with the command Trace "
            "and canonical Scheduler")
    expected_artifact = _json_round_trip(
        execution_artifact_manifest(
            canonical_result,
            performance_timing_status=(
                _expected_performance_timing_status(expected_simulation)),
        ))
    if artifact != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")
    expected_metrics = _canonical_production_trace_metrics_manifest(
        deployment,
        expected_requests,
        canonical_result,
        expected_slo_contract,
    )
    if result["metrics"] != expected_metrics:
        raise ValueError(
            f"{context} metrics disagree with canonical Scheduler and "
            "execution trace")

    system_resources = result["metrics"]["system_resources"]
    observation_window = system_resources["observation_window"]
    observation_window_s = observation_window["duration_s"]
    worker_pool_busy_time_s = _interval_union_duration_from_output(
        worker_pool_intervals,
        observation_window_s,
        context=f"{context} worker pool",
    )
    worker_pool_idle_time_s = observation_window_s - worker_pool_busy_time_s
    worker_pool_resources = system_resources["worker_pool_utilization"]
    if set(worker_pool_resources) != {"main"}:
        raise ValueError(
            f"{context} worker-pool utilization coverage is invalid")
    worker_pool = worker_pool_resources["main"]
    if (observation_window["basis"]
            != "simulation_origin_to_last_completion"
            or not math.isclose(
                observation_window_s,
                result["schedule_end_time_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or system_resources["communication_resource_utilization"] != {}
            or system_resources["link_utilization"] != {}
            or set(worker_pool) != {
                "kernel_interval_count", "busy_time_s", "idle_time_s",
                "utilization"}
            or worker_pool["kernel_interval_count"] != total_call_count
            or not math.isclose(
                worker_pool["busy_time_s"],
                worker_pool_busy_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                worker_pool["idle_time_s"],
                worker_pool_idle_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                worker_pool["utilization"],
                worker_pool_busy_time_s / observation_window_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or set(system_resources["scheduler_queue"]["by_pool"])
            != {"main"}):
        raise ValueError(
            f"{context} system resource metrics disagree with the pool")


def _validate_colocated_kv_lifecycle_replay(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
    deployment: DeploymentPlan,
    recorded_runtime: dict[str, Any],
    kv_lifecycle_replay: Any,
    *,
    tensor_parallel_size: int,
    context: str,
) -> None:
    scheduler = expected_simulation["scheduler"]
    if (expected_simulation["mtp"]["acceptance_policy"] != "disabled"
            or expected_simulation["expert_routing"] is not None
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(f"{context} unexpectedly enabled MTP or MoE routing")
    expected_replay_manifest = _json_round_trip(
        kv_lifecycle_replay.manifest())
    if (result["kv_lifecycle_replay"] != expected_replay_manifest
            or kv_lifecycle_replay.context.request_stream_sha256
            != request_stream_sha256(expected_requests)):
        raise ValueError(
            f"{context} replay disagrees with the command manifest or Trace")

    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    expected_runtime_inputs = {
        "expert_routing": _json_round_trip(
            runtime_inputs.expert_routing.manifest()),
        "attention_kv_layout": expected_simulation["attention_kv_layout"],
        "attention_kv_distribution": expected_simulation[
            "attention_kv_distribution"],
        "graph_execution": runtime_inputs.graph_execution.manifest(),
    }
    if result["execution_provenance"]["runtime_inputs"] != (
            expected_runtime_inputs):
        raise ValueError(f"{context} runtime inputs disagree")

    hardware_id = expected_simulation["hardware"]["hardware_id"]
    hardware = _load_expected_hardware(
        expected_simulation, hardware_id, context)
    kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(
            tensor_parallel_size, 1))
    backend_config = BackendConfig(
        "main",
        "colocated",
        scheduler["max_batched_tokens"],
        scheduler["max_requests"],
        scheduler["num_kv_blocks"],
        kv_block_size_tokens=scheduler["kv_block_size_tokens"],
        kv_bytes_per_token=kv_bytes_per_token,
    scheduler_pipeline_depth=scheduler["scheduler_pipeline_depth"])
    try:
        with ExitStack() as construction_lifecycle:
            backend = VLLMBackend.from_model(
                backend_config,
                VLLMSchedulerConfig(
                    expected_model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=1,
                    enable_expert_parallel=False,
                    dtype="bfloat16",
                    max_model_len=scheduler["max_model_len"],
                    enable_chunked_prefill=True,
                    enable_prefix_caching=False,
                    mtp_num_speculative_tokens=0,
                    mtp_acceptance_policy="disabled",
                    mtp_fixed_accepted_tokens=0,
                    mtp_acceptance_source_id="disabled",
                ),
            )
            construction_lifecycle.callback(backend.close)
            canonical_runtime = _json_round_trip(asdict(backend.runtime_info))
            if canonical_runtime != recorded_runtime:
                raise ValueError(
                    f"{context} pinned Scheduler runtime disagrees with the "
                    "recorded runtime")
            runtime_sha256 = vllm_scheduler_runtime_sha256(
                backend.runtime_info)
            canonical_result = Simulation(
                SimulationConfig(
                    expected_model,
                    deployment,
                    expected_requests,
                    expected_simulation["launch_policy"],
                    expected_simulation["prediction_method"],
                    runtime_inputs,
                ),
                backend,
                {hardware_id: hardware},
                kv_cache_tier_manager=KVCacheTierManager(
                    kv_lifecycle_replay.cache_config),
                kv_lifecycle_replay=kv_lifecycle_replay,
                scheduler_runtime_sha256=runtime_sha256,
            ).run()
            # Simulation.run owns the backend after construction and closes it
            # on both success and failure.  Keep the ExitStack callback only
            # for failures before run() assumes that ownership.
            construction_lifecycle.pop_all()
    except (OSError, TypeError, ValueError, NotImplementedError, RuntimeError,
            TimeoutError) as error:
        raise ValueError(
            f"{context} cannot be canonically reproduced from the command "
            "Trace, replay, and pinned Scheduler") from error

    expected_prefix_alignment = _json_round_trip(prefix_alignment_manifest(
        expected_requests,
        canonical_result,
        scheduler["kv_block_size_tokens"],
    ))
    if result["request_source"]["prefix_alignment"] != (
            expected_prefix_alignment):
        raise ValueError(
            f"{context} Prefix alignment disagrees with the command Trace "
            "and canonical Scheduler")
    expected_artifact = _json_round_trip(
        execution_artifact_manifest(
            canonical_result,
            performance_timing_status=(
                _expected_performance_timing_status(expected_simulation)),
        ))
    if result["execution_artifact"] != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")
    modeled_offload = expected_artifact["modeled_kv_offload"]
    expected_availability = (
        "available_modeled_serial"
        if tensor_parallel_size == 1
        else "available_modeled_rank_local"
    )
    if (modeled_offload["availability"] != expected_availability
            or modeled_offload["intent_source"]["kind"]
            != "scheduler_event_replay"
            or modeled_offload["intent_source"]["replay"]["source_sha256"]
            != kv_lifecycle_replay.source_sha256):
        raise ValueError(
            f"{context} modeled lifecycle or final tier state is invalid")
    if tensor_parallel_size == 1:
        _validate_tp1_kv_offload_invariants(
            expected_artifact,
            expected_model,
            expected_hbm_blocks=scheduler["num_kv_blocks"],
            context=context,
        )
    elif tensor_parallel_size == 2:
        _validate_tp2_kv_offload_invariants(
            expected_artifact,
            expected_model,
            context=context,
        )
    else:
        raise AssertionError(
            "formal Host-KV verification supports only TP=1 or TP=2")
    counts = expected_artifact["counts"]
    expected_kernel_calls = sum(
        step.kernel_plan.logical_call_count
        for step in canonical_result.steps)
    if (result["num_steps"] != len(canonical_result.steps)
            or result["num_kernel_calls"] != expected_kernel_calls
            or result["schedule_end_time_s"]
            != canonical_result.schedule_trace.end_time_s
            or counts["modeled_kv_offload_transfers"] != 2):
        raise ValueError(f"{context} counts or completion time disagree")
    expected_metrics = _canonical_production_trace_metrics_manifest(
        deployment,
        expected_requests,
        canonical_result,
        expected_slo_contract,
    )
    if result["metrics"] != expected_metrics:
        raise ValueError(
            f"{context} metrics disagree with canonical Scheduler and "
            "execution trace")
    _validate_host_kv_system_resources(
        expected_metrics["system_resources"],
        pool_name="main",
        ranks=tuple(range(tensor_parallel_size)),
        hardware_id=hardware_id,
        block_bytes=(
            16
            * expected_model.kv_bytes_per_token_per_parallel_rank(
                tensor_parallel_size, 1)
        ),
        context=context,
    )
    request_id = expected_requests.requests[0].request_id
    critical_path_contributions = expected_metrics["critical_path"][
        "requests"][request_id]["e2e"]["contribution_s_by_category"]
    if ("kernel:kv_offload" not in critical_path_contributions
            or critical_path_contributions["kernel:kv_offload"] <= 0):
        raise ValueError(
            f"{context} critical path lacks KV offload contribution")


def _validate_tp1_kv_offload_invariants(
    artifact: dict[str, Any],
    model: ModelConfig,
    *,
    expected_hbm_blocks: int,
    context: str,
) -> None:
    block_size_tokens = 16
    block_bytes = (
        block_size_tokens
        * model.kv_bytes_per_token_per_parallel_rank(1, 1)
    )
    modeled = artifact["modeled_kv_offload"]
    final_state = modeled["final_tier_state"]
    tiers = final_state["tiers"]
    if expected_hbm_blocks == 2:
        expected_hbm_description = "2-HBM"
    elif expected_hbm_blocks == 3:
        expected_hbm_description = "3-HBM"
    else:
        raise AssertionError("unsupported formal Host-KV HBM block count")
    if (artifact["schema_version"] not in (7, 8)
            or modeled["schema_version"] != 4
            or modeled["availability"] != "available_modeled_serial"
            or modeled["mode"] != "modeled_host_kv_serial"
            or modeled["ledger_semantics"]
            != "logical_request_blocks_with_single_rank_capacity"
            or modeled["physical_transfer_count"] != 2
            or modeled["scope"]["host_link_utilization"]
            != "available_in_system_resource_manifest_v5_"
            "host_kv_link_utilization_per_rank"
            or final_state["block_size_tokens"] != block_size_tokens
            or final_state["block_bytes"] != block_bytes
            or final_state["configured_policy"] != "lru"
            or final_state["inflight_ready_events"] != []
            or final_state["residency"] != []
            or set(tiers) != {"hbm", "host"}
            or tiers["hbm"]["capacity_bytes"]
            != expected_hbm_blocks * block_bytes
            or tiers["host"]["capacity_bytes"] != block_bytes
            or any(
                tier[field] != 0
                for tier in tiers.values()
                for field in (
                    "used_bytes", "reserved_bytes", "resident_blocks")
            )):
        raise ValueError(
            f"{context} final tier ledger violates the fixed "
            f"{expected_hbm_description}/1-Host block contract")

    expected_routes = (
        ("evict", "hbm", "host"),
        ("load", "host", "hbm"),
    )
    transfers = modeled["transfers"]
    if len(transfers) != len(expected_routes):
        raise ValueError(f"{context} requires exactly two KV transfers")
    for transfer, expected_route in zip(
            transfers, expected_routes, strict=True):
        action, source_tier, destination_tier = expected_route
        intent = transfer["intent"]
        logical = transfer["logical_completion"]
        physical_transfers = transfer["physical_transfers"]
        if len(physical_transfers) != 1:
            raise ValueError(
                f"{context} TP1 transfer requires one physical owner")
        physical = physical_transfers[0]
        completion = physical["completion"]
        interval = physical["relative_interval"]
        expected_resource_id = (
            f"pool:{physical['pool_name']}:rank:{physical['rank']}:pcie")
        if (intent["action"] != action
                or intent["source_tier"] != source_tier
                or intent["destination_tier"] != destination_tier
                or intent["tier"] != "host"
                or intent["token_start"] != 0
                or intent["token_end"] != block_size_tokens
                or intent["bytes"] != block_bytes
                or logical["action"] != action
                or logical["source_tier"] != source_tier
                or logical["destination_tier"] != destination_tier
                or logical["bytes_per_rank"] != block_bytes
                or logical["physical_transfer_count"] != 1
                or logical["link"] != "pcie"
                or completion["link"] != "pcie"
                or completion["bytes"] != block_bytes
                or physical["resource_id"] != expected_resource_id
                or physical["prediction_total_s"] <= 0
                or not math.isclose(
                    interval["end_time_s"] - interval["start_time_s"],
                    physical["prediction_total_s"],
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )):
            raise ValueError(
                f"{context} KV transfer geometry, route, or canonical "
                "PCIe prediction is invalid")

    kv_offload_calls = []
    for step in artifact["steps"]:
        for call in step["kernel_plan"]["calls"]:
            if call["kernel_spec"]["kind"] == "KV_OFFLOAD":
                kv_offload_calls.append(call)
    if len(kv_offload_calls) != len(expected_routes):
        raise ValueError(
            f"{context} requires exactly two KV_OFFLOAD KernelSpec calls")
    for call, expected_route in zip(
            kv_offload_calls, expected_routes, strict=True):
        action, _source_tier, _destination_tier = expected_route
        spec = call["kernel_spec"]
        physical = spec["transfer"]
        if (spec["hardware_id"] != "nvidia_a800"
                or call["resource"] != "pcie"
                or physical["algorithm"] != "dma"
                or physical["bytes"] != block_bytes
                or physical["direction"] != action
                or physical["link"] != "pcie"
                or physical["tier"] != "host"):
            raise ValueError(
                f"{context} KV_OFFLOAD KernelSpec physical transfer is "
                "invalid")


def _validate_tp2_kv_offload_invariants(
    artifact: dict[str, Any],
    model: ModelConfig,
    *,
    context: str,
) -> None:
    block_size_tokens = 16
    block_bytes = (
        block_size_tokens
        * model.kv_bytes_per_token_per_parallel_rank(2, 1)
    )
    modeled = artifact["modeled_kv_offload"]
    final_state = modeled["final_tier_state"]
    tiers = final_state["tiers"]
    if (artifact["schema_version"] not in (7, 8)
            or modeled["schema_version"] != 4
            or modeled["availability"] != "available_modeled_rank_local"
            or modeled["mode"] != "modeled_host_kv_rank_local_lockstep"
            or modeled["ledger_semantics"]
            != "logical_request_blocks_with_lockstep_per_rank_capacity"
            or modeled["physical_transfer_count"] != 4
            or modeled["scope"]["multi_rank"]
            != "validated_colocated_qwen3_tp2_lockstep"
            or modeled["scope"]["host_link_utilization"]
            != "available_in_system_resource_manifest_v5_"
            "host_kv_link_utilization_per_rank"
            or final_state["block_size_tokens"] != block_size_tokens
            or final_state["block_bytes"] != block_bytes
            or final_state["configured_policy"] != "lru"
            or final_state["inflight_ready_events"] != []
            or final_state["residency"] != []
            or set(tiers) != {"hbm", "host"}
            or tiers["hbm"]["capacity_bytes"] != 2 * block_bytes
            or tiers["host"]["capacity_bytes"] != block_bytes
            or any(
                tier[field] != 0
                for tier in tiers.values()
                for field in (
                    "used_bytes", "reserved_bytes", "resident_blocks")
            )):
        raise ValueError(
            f"{context} rank-local artifact or final tier ledger is invalid")

    expected_routes = (
        ("evict", "hbm", "host"),
        ("load", "host", "hbm"),
    )
    transfers = modeled["transfers"]
    if len(transfers) != len(expected_routes):
        raise ValueError(f"{context} requires exactly two logical transfers")
    steps_by_id = {
        step["step_id"]: step for step in artifact["steps"]
    }
    for transfer, expected_route in zip(
            transfers, expected_routes, strict=True):
        action, source_tier, destination_tier = expected_route
        intent = transfer["intent"]
        logical = transfer["logical_completion"]
        physical_transfers = transfer["physical_transfers"]
        if (intent["action"] != action
                or intent["source_tier"] != source_tier
                or intent["destination_tier"] != destination_tier
                or intent["tier"] != "host"
                or intent["token_start"] != 0
                or intent["token_end"] != block_size_tokens
                or intent["bytes"] != block_bytes
                or logical["action"] != action
                or logical["source_tier"] != source_tier
                or logical["destination_tier"] != destination_tier
                or logical["link"] != "pcie"
                or logical["bytes_per_rank"] != block_bytes
                or logical["physical_transfer_count"] != 2
                or tuple(item["rank"] for item in physical_transfers)
                != (0, 1)):
            raise ValueError(
                f"{context} logical rank-local transfer is invalid")
        step_id = transfer["step_id"]
        if step_id not in steps_by_id:
            raise ValueError(f"{context} transfer references an unknown step")
        step = steps_by_id[step_id]
        calls = {
            call["call_id"]: call for call in step["kernel_plan"]["calls"]
        }
        intervals = {
            interval["call_id"]: interval
            for interval in step["execution_timeline"]["intervals"]
        }
        for physical in physical_transfers:
            rank = physical["rank"]
            call_id = physical["call_id"]
            completion = physical["completion"]
            if call_id not in calls or call_id not in intervals:
                raise ValueError(
                    f"{context} physical transfer lacks a call or interval")
            call = calls[call_id]
            interval = intervals[call_id]
            transfer_config = call["kernel_spec"]["transfer"]
            duration_s = (
                physical["relative_interval"]["end_time_s"]
                - physical["relative_interval"]["start_time_s"]
            )
            if (physical["pool_name"] != "main"
                    or completion["rank"] != rank
                    or completion["pool_name"] != "main"
                    or completion["node_id"] != physical["node_id"]
                    or completion["call_id"] != call_id
                    or completion["bytes"] != block_bytes
                    or completion["link"] != "pcie"
                    or physical["resource_id"]
                    != f"pool:main:rank:{rank}:pcie"
                    or call["rank"] != rank
                    or call["pool_name"] != "main"
                    or call["resource"] != "pcie"
                    or call["kernel_spec"]["kind"] != "KV_OFFLOAD"
                    or transfer_config["algorithm"] != "dma"
                    or transfer_config["bytes"] != block_bytes
                    or transfer_config["direction"] != action
                    or transfer_config["link"] != "pcie"
                    or transfer_config["tier"] != "host"
                    or interval["rank"] != rank
                    or interval["pool_name"] != "main"
                    or interval["resource"] != "pcie"
                    or duration_s <= 0
                    or not math.isclose(
                        duration_s,
                        physical["prediction_total_s"],
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )):
                raise ValueError(
                    f"{context} physical rank transfer is invalid")


def _validate_host_kv_system_resources(
    resources: dict[str, Any],
    *,
    pool_name: str,
    ranks: tuple[int, ...],
    hardware_id: str,
    block_bytes: int,
    context: str,
) -> None:
    expected_resource_ids = {
        f"pool:{pool_name}:rank:{rank}:pcie" for rank in ranks
    }
    host_resources = resources["host_kv_link_utilization"]
    if (resources["schema_version"] != 5
            or resources["host_kv_link_activity_basis"]
            != "per_rank_resource_union_of_canonical_kv_offload_physical_"
            "completion_intervals"
            or set(host_resources) != expected_resource_ids):
        raise ValueError(
            f"{context} Host-KV resource coverage is invalid")
    observation_window_s = resources["observation_window"]["duration_s"]
    for rank in ranks:
        resource_id = f"pool:{pool_name}:rank:{rank}:pcie"
        record = host_resources[resource_id]
        if (record["pool_name"] != pool_name
                or record["rank"] != rank
                or record["hardware_id"] != hardware_id
                or record["link"] != "pcie"
                or record["physical_transfer_count"] != 2
                or record["action_counts"] != {"evict": 1, "load": 1}
                or record["bytes_by_action"]
                != {"evict": block_bytes, "load": block_bytes}
                or record["transferred_bytes"] != 2 * block_bytes
                or record["busy_time_s"] <= 0
                or record["idle_time_s"] < 0
                or not math.isclose(
                    record["busy_time_s"] + record["idle_time_s"],
                    observation_window_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    record["utilization"],
                    record["busy_time_s"] / observation_window_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )):
            raise ValueError(
                f"{context} Host-KV resource {resource_id!r} is invalid")


def _validate_colocated_tp2_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    replay_identity = expected_simulation["kv_lifecycle_replay"]
    if replay_identity is None:
        kv_lifecycle_replay = None
        context = "TP=2 Production Trace"
    else:
        try:
            kv_lifecycle_replay = load_kv_lifecycle_replay(
                replay_identity["path"])
        except (OSError, TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                "TP=2 Host-KV Production Trace replay cannot be "
                "independently reloaded") from error
        if (kv_lifecycle_replay.source_sha256
                != replay_identity["source_sha256"]
                or kv_lifecycle_replay.artifact_id
                != replay_identity["artifact_id"]):
            raise ValueError(
                "TP=2 Host-KV Production Trace replay identity changed")
        context = "TP=2 Host-KV Production Trace"
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    resource_id = "nvlink:main:tensor-fabric"
    collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            resource_id,
            "analytical_roofline_v1",
        )
        for operation in ("all_reduce", "all_gather")
    )
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(tensor_parallel_size=2),
                kv_cache=(
                    None
                    if kv_lifecycle_replay is None
                    else kv_lifecycle_replay.cache_config
                ),
                collectives=collectives,
            ),),
        ),
        num_layers=expected_model.num_hidden_layers,
        num_experts=expected_model.num_experts,
    )
    expected_plan = _json_round_trip(deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            "TP=2 Production Trace worker-pool topology disagrees with "
            "command manifest")

    scheduler = expected_simulation["scheduler"]
    runtime = result["scheduler_runtime"]["main"]
    expected_owners = [
        [f"model.layers.{layer_index}.self_attn.attn"]
        for layer_index in range(expected_model.num_hidden_layers)
    ]
    if runtime["scheduler_kv_cache_tensor_owners"] != expected_owners:
        raise ValueError(
            "TP=2 Production Trace Scheduler KV owners disagree with the "
            "pinned Qwen3 model")
    _validate_production_trace_capacity(
        result,
        scheduler,
        ("main",),
        context="TP=2 Production Trace",
    )
    expected_kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(2, 1))
    capacity_plan = result["kv_capacity"]["scheduler_pools"][0]
    if capacity_plan["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError(
            "TP=2 Production Trace KV capacity disagrees with model "
            "sharding")
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["route_decisions"] != []
            or result["transfers"] != []
            or result["num_transfers"] != 0):
        raise ValueError(
            "TP=2 Production Trace routing or transfer state is invalid")
    if (kv_lifecycle_replay is None
            and (result["kv_lifecycle_replay"] is not None
            or result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            })):
        raise ValueError(
            "TP=2 Production Trace unexpectedly enabled KV offload")
    mtp_scenario = result["mtp_scenario"]
    if (mtp_scenario["num_speculative_tokens"] != 0
            or mtp_scenario["acceptance_policy"] != "disabled"
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(
            "TP=2 Production Trace unexpectedly enabled MTP")
    if kv_lifecycle_replay is not None:
        _validate_colocated_kv_lifecycle_replay(
            expected_simulation,
            expected_model,
            expected_model_path,
            result,
            expected_requests,
            expected_slo_contract,
            deployment,
            runtime,
            kv_lifecycle_replay,
            tensor_parallel_size=2,
            context=context,
        )
        return

    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    expected_runtime_inputs = {
        "expert_routing": _json_round_trip(
            runtime_inputs.expert_routing.manifest()),
        "attention_kv_layout": expected_simulation["attention_kv_layout"],
        "attention_kv_distribution": expected_simulation[
            "attention_kv_distribution"],
        "graph_execution": runtime_inputs.graph_execution.manifest(),
    }
    if result["execution_provenance"]["runtime_inputs"] != (
            expected_runtime_inputs):
        raise ValueError(
            f"{context} runtime inputs disagree with the command manifest")

    artifact = result["execution_artifact"]
    artifact_trace = artifact["schedule_trace"]
    if (artifact_trace["route_decision_version"] != 2
            or artifact_trace["transfer_trace_version"] != 2
            or artifact_trace["route_decisions"] != []
            or artifact_trace["transfers"] != []
            or artifact["transfers"] != []):
        raise ValueError(
            "TP=2 Production Trace execution artifact contains a route or "
            "transfer")
    steps = artifact["steps"]
    counts = artifact["counts"]
    if (not steps
            or result["num_steps"] != len(steps)
            or counts["schedule_steps"] != len(steps)
            or counts["simulation_steps"] != len(steps)
            or counts["simulation_transfers"] != 0
            or counts["trace_transfers"] != 0
            or counts["modeled_kv_offload_transfers"] != 0):
        raise ValueError(
            "TP=2 Production Trace execution artifact counts disagree")
    trace_entries = artifact_trace["entries"]
    entries_by_id = {
        entry["step_id"]: entry for entry in trace_entries
    }
    if (len(entries_by_id) != len(trace_entries)
            or len(entries_by_id) != len(steps)):
        raise ValueError(f"{context} schedule trace coverage disagrees")

    operation_counts = {"all_gather": 0, "all_reduce": 0}
    communication_bytes = 0
    communication_intervals = []
    total_call_count = 0
    all_call_ids = set()
    compute_kinds = {"GEMM", "FLASH_ATTN", "VECTOR"}
    hardware = _load_expected_hardware(
        expected_simulation, hardware_id, context)
    hardware_by_id = {hardware_id: hardware}
    all_step_ids = set()
    backend_steps_by_id = {}
    simulation_steps = []
    execution = ServingExecutionContext(
        expected_model,
        deployment,
        hardware_by_id,
        expected_simulation["launch_policy"],
        expected_simulation["prediction_method"],
        runtime_inputs,
        execution_mode=result["execution_mode"],
    )
    for step in steps:
        step_id = step["step_id"]
        if (step_id in all_step_ids
                or step_id not in entries_by_id
                or step["pool_name"] != "main"):
            raise ValueError(
                "TP=2 Production Trace execution step used an unknown pool")
        all_step_ids.add(step_id)
        entry = entries_by_id[step_id]
        if (entry["pool_name"] != "main"
                or step["schedule_interval"] != {
                    "start_time_s": entry["start_time_s"],
                    "end_time_s": entry["end_time_s"],
                }):
            raise ValueError(f"{context} schedule interval identity is invalid")
        try:
            backend_step = _backend_step_from_output_manifest(
                step["backend_step"])
            components = execution.execute_step(
                backend_step.schedule,
                kv_cache_events=backend_step.kv_cache_events,
            )
            workload_object = components.workload
            kernel_plan_object = components.kernel_plan
            prediction_objects = components.predictions
            timeline_object = components.execution_timeline
        except (TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                f"{context} step cannot be canonically rematerialized") \
                from error
        backend_steps_by_id[step_id] = backend_step
        expected_components = _json_round_trip(
            execution_step_components_manifest(
                workload_object,
                kernel_plan_object,
                prediction_objects,
                timeline_object,
                origin_time_s=step["schedule_interval"]["start_time_s"],
            ))
        actual_components = {
            field: step[field]
            for field in expected_components
        }
        schedule = step["backend_step"]["schedule"]
        kernel_plan = step["kernel_plan"]
        calls = kernel_plan["calls"]
        call_ids = tuple(call["call_id"] for call in calls)
        if (not calls
                or len(call_ids) != len(set(call_ids))
                or kernel_plan["call_count"] != len(calls)
                or kernel_plan["topological_order"] != list(call_ids)):
            raise ValueError(
                "TP=2 Production Trace KernelPlan identity is invalid")
        workload = step["workload_graph"]
        nodes = workload["nodes"]
        nodes_by_id = {node["node_id"]: node for node in nodes}
        if (len(nodes_by_id) != len(nodes)
                or workload["node_count"] != len(nodes)
                or set(nodes_by_id) != set(call_ids)
                or workload["topological_order"] != list(call_ids)
                or workload["edge_count"] != len(workload["edges"])):
            raise ValueError(
                "TP=2 Production Trace WorkloadGraph identity is invalid")
        expected_lowering = [{
            "node_id": node["node_id"],
            "call_ids": [node["node_id"]],
        } for node in nodes]
        lowering = step["lowering"]
        if (lowering["relation"]
                != "workload_nodes_to_kernel_calls_many_to_many_v1"
                or lowering["node_to_call_ids"] != expected_lowering):
            raise ValueError(
                "TP=2 Production Trace lowering identity is invalid")
        if all_call_ids.intersection(call_ids):
            raise ValueError(
                "TP=2 Production Trace contains duplicate cross-step call IDs")
        all_call_ids.update(call_ids)
        predictions = step["predictions"]
        predictions_by_id = {
            prediction["call_id"]: prediction
            for prediction in predictions
        }
        if (len(predictions_by_id) != len(predictions)
                or set(predictions_by_id) != set(call_ids)):
            raise ValueError(
                "TP=2 Production Trace predictions do not cover KernelPlan")
        if ({call["rank"] for call in calls} != {0, 1}
                or any(call["pool_name"] != "main"
                       or call["kernel_spec"]["hardware_id"] != hardware_id
                       for call in calls)):
            raise ValueError(
                "TP=2 Production Trace rank coverage disagrees with the "
                "worker pool")
        call_position = {
            call_id: index for index, call_id in enumerate(call_ids)
        }
        graph_predecessors = {call_id: [] for call_id in call_ids}
        for edge in workload["edges"]:
            if (edge["source"] not in call_position
                    or edge["destination"] not in call_position):
                raise ValueError(
                    "TP=2 Production Trace WorkloadGraph edge is invalid")
            graph_predecessors[edge["destination"]].append(edge["source"])
        calls_by_id = {call["call_id"]: call for call in calls}
        for call in calls:
            call_id = call["call_id"]
            if call["source_node_ids"] != [call_id]:
                raise ValueError(
                    "TP=2 Production Trace call/source-node identity is "
                    "invalid")
            dependencies = call["dependencies"]
            if (len(dependencies) != len(set(dependencies))
                    or set(dependencies) != set(graph_predecessors[call_id])
                    or len(dependencies)
                    != len(graph_predecessors[call_id])
                    or any(dependency not in call_position
                           or call_position[dependency] >= call_position[call_id]
                           for dependency in dependencies)):
                raise ValueError(
                    "TP=2 Production Trace KernelPlan dependency is invalid")
            kernel_kind = call["kernel_spec"]["kind"]
            if kernel_kind == "COMMUNICATION":
                if call_id.endswith(":tp-all-reduce"):
                    base_id = call_id[:-len(":tp-all-reduce")]
                elif call_id.endswith(":tp-all-gather"):
                    base_id = call_id[:-len(":tp-all-gather")]
                else:
                    raise ValueError(
                        "TP=2 Production Trace collective call ID is invalid")
                expected_dependencies = [
                    f"{base_id}:tp-rank-0",
                    f"{base_id}:tp-rank-1",
                ]
                if dependencies != expected_dependencies:
                    raise ValueError(
                        "TP=2 Production Trace collective dependency is "
                        "invalid")
            elif kernel_kind in compute_kinds:
                if call["rank"] == 0:
                    rank_suffix = ":tp-rank-0"
                    other_rank_suffix = ":tp-rank-1"
                elif call["rank"] == 1:
                    rank_suffix = ":tp-rank-1"
                    other_rank_suffix = ":tp-rank-0"
                else:
                    raise ValueError(
                        "TP=2 Production Trace compute call rank is invalid")
                if (not call_id.endswith(rank_suffix)
                        or any(dependency.endswith(other_rank_suffix)
                               for dependency in dependencies)):
                    raise ValueError(
                        "TP=2 Production Trace rank-local DAG is invalid")
            else:
                raise ValueError(
                    "TP=2 Production Trace emitted an unsupported kernel "
                    "kind")
        intervals = step["execution_timeline"]["intervals"]
        intervals_by_id = {
            interval["call_id"]: interval for interval in intervals
        }
        if (len(intervals_by_id) != len(intervals)
                or set(intervals_by_id) != set(call_ids)):
            raise ValueError(
                "TP=2 Production Trace timeline does not cover KernelPlan")
        for call in calls:
            call_id = call["call_id"]
            interval = intervals_by_id[call_id]
            prediction = predictions_by_id[call_id]
            duration_s = interval["end_time_s"] - interval["start_time_s"]
            absolute_duration_s = (
                interval["absolute_end_time_s"]
                - interval["absolute_start_time_s"])
            if (interval["pool_name"] != call["pool_name"]
                    or interval["rank"] != call["rank"]
                    or interval["resource"] != call["resource"]
                    or not math.isclose(
                        duration_s,
                        absolute_duration_s,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                    or not math.isclose(
                        duration_s,
                        prediction["t_total_s"],
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )):
                raise ValueError(
                    "TP=2 Production Trace prediction/timeline identity is "
                    "invalid")
            kernel_kind = call["kernel_spec"]["kind"]
            if kernel_kind == "COMMUNICATION":
                expected_reserved_resources = [
                    f"communication:endpoint:{hardware_id}:0:nvlink",
                    f"communication:endpoint:{hardware_id}:1:nvlink",
                    f"communication:fabric:{resource_id}",
                ]
                communication_intervals.append((
                    interval["absolute_start_time_s"],
                    interval["absolute_end_time_s"],
                ))
            elif kernel_kind in compute_kinds:
                expected_reserved_resources = [
                    f"pool:main:rank:{call['rank']}:compute",
                ]
            else:
                raise ValueError(
                    "TP=2 Production Trace emitted an unsupported kernel "
                    "kind")
            if interval["reserved_resource_ids"] != expected_reserved_resources:
                raise ValueError(
                    "TP=2 Production Trace reserved resources disagree with "
                    "KernelPlan")
        step_operations = set()
        for call in calls:
            kernel_spec = call["kernel_spec"]
            if kernel_spec["kind"] != "COMMUNICATION":
                continue
            transfer = kernel_spec["transfer"]
            operation = transfer["operation"]
            if operation not in operation_counts:
                raise ValueError(
                    "TP=2 Production Trace emitted an unsupported collective")
            if operation == "all_reduce":
                expected_reduction = "sum"
            elif operation == "all_gather":
                expected_reduction = None
            else:
                raise AssertionError(
                    "validated TP=2 collective operation vanished")
            if (call["rank"] != 0
                    or call["resource"] != resource_id
                    or kernel_spec["dtype"] is not None
                    or kernel_spec["shape"] != {}
                    or kernel_spec["attributes"] != {}
                    or kernel_spec["variant"] is not None
                    or kernel_spec["launch"] is not None
                    or transfer["algorithm"] != "ring"
                    or transfer["backend"] != "analytical_roofline_v1"
                    or transfer["backend_config_sha256"] is not None
                    or transfer["bytes_semantics"] != "per_rank_input"
                    or not isinstance(transfer["bytes"], int)
                    or isinstance(transfer["bytes"], bool)
                    or transfer["bytes"] <= 0
                    or transfer["direction"] != "collective"
                    or transfer["link"] != "nvlink"
                    or transfer["topology"] != "ring"
                    or transfer["resource_id"] != resource_id
                    or transfer["participants"] != [0, 1]
                    or transfer["ring_order"] != [0, 1]
                    or transfer["reduction"] != expected_reduction
                    or transfer["source_rank"] is not None
                    or transfer["destination_rank"] is not None
                    or transfer["tier"] is not None):
                raise ValueError(
                    "TP=2 Production Trace collective KernelSpec is invalid")
            expected_bytes = _tp2_collective_bytes_from_schedule_and_model(
                expected_model,
                schedule,
                call,
                calls_by_id,
                nodes_by_id,
            )
            if transfer["bytes"] != expected_bytes:
                raise ValueError(
                    "TP=2 Production Trace collective bytes disagree with "
                    "the Scheduler/model shape")
            prediction = predictions_by_id[call["call_id"]]
            if (prediction["bottleneck_group"] != "communication"
                    or prediction["t_launch_s"] != 0.0
                    or prediction["t_total_s"] <= 0.0
                    or prediction["t_per_group_s"] != {
                        "communication": prediction["t_total_s"]}
                    or prediction["t_per_resource_s"] != {
                        "communication": prediction["t_total_s"]}):
                raise ValueError(
                    "TP=2 Production Trace collective prediction is invalid")
            try:
                canonical_kernel = KernelSpec(
                    kind=KernelKind.COMMUNICATION,
                    hardware_id=hardware_id,
                    transfer=PhysicalTransferConfig(**transfer),
                )
                canonical_prediction = predict(
                    canonical_kernel,
                    hardware,
                    expected_simulation["prediction_method"],
                )
            except (TypeError, ValueError, NotImplementedError) as error:
                raise ValueError(
                    "TP=2 Production Trace collective cannot be repriced") \
                    from error
            if (prediction["t_total_s"] != canonical_prediction.t_total_s
                    or prediction["t_launch_s"]
                    != canonical_prediction.t_launch_s
                    or prediction["bottleneck_group"]
                    != canonical_prediction.bottleneck_group
                    or prediction["t_per_group_s"]
                    != dict(canonical_prediction.t_per_group)
                    or prediction["t_per_resource_s"]
                    != dict(canonical_prediction.t_per_resource)):
                raise ValueError(
                    "TP=2 Production Trace collective prediction disagrees "
                    "with kernel_base.predict")
            operation_counts[operation] += 1
            communication_bytes += transfer["bytes"]
            step_operations.add(operation)
        if step_operations != set(operation_counts):
            raise ValueError(
                "TP=2 Production Trace step is missing a required collective")
        if actual_components != expected_components:
            raise ValueError(
                f"{context} canonical Scheduler-to-KernelSpec "
                "rematerialization disagrees with the execution artifact")
        _validate_canonical_production_trace_step_duration(
            step,
            timeline_object.duration_s,
            context=context,
        )
        simulation_steps.append(SimulationStep(
            backend_step,
            workload_object,
            kernel_plan_object,
            prediction_objects,
            timeline_object,
        ))
        total_call_count += len(calls)
    if (set(entries_by_id) != all_step_ids
            or operation_counts["all_gather"] <= 0
            or operation_counts["all_reduce"] <= 0
            or counts["kernel_calls"] != total_call_count
            or result["num_kernel_calls"] != total_call_count):
        raise ValueError(
            "TP=2 Production Trace kernel or collective counts disagree")

    canonical_schedule_trace = ScheduleTrace(
        ScheduleTraceEntry(
            backend_steps_by_id[entry["step_id"]],
            entry["start_time_s"],
            entry["end_time_s"],
        )
        for entry in trace_entries
    )
    _validate_production_trace_scheduler_binding(
        expected_requests,
        canonical_schedule_trace,
        scheduler["kv_block_size_tokens"],
        result,
        max_requests=scheduler["max_requests"],
        context=context,
    )
    _replay_pinned_parallel_scheduler(
        expected_requests,
        canonical_schedule_trace,
        scheduler,
        expected_model,
        expected_model_path,
        runtime,
        num_kv_blocks=runtime["num_kv_blocks"],
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        expected_mtp=expected_simulation["mtp"],
        context=context,
    )
    canonical_metrics = compute_serving_metrics(
        expected_requests,
        canonical_schedule_trace,
    )
    canonical_result = SimulationResult(
        canonical_schedule_trace,
        canonical_metrics,
        tuple(simulation_steps),
    )
    expected_prefix_alignment = _json_round_trip(prefix_alignment_manifest(
        expected_requests,
        canonical_result,
        scheduler["kv_block_size_tokens"],
    ))
    if result["request_source"]["prefix_alignment"] != (
            expected_prefix_alignment):
        raise ValueError(
            f"{context} Prefix alignment disagrees with the command Trace "
            "and canonical Scheduler")
    expected_artifact = _json_round_trip(
        execution_artifact_manifest(
            canonical_result,
            performance_timing_status=(
                _expected_performance_timing_status(expected_simulation)),
        ))
    if artifact != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")

    system_resources = result["metrics"]["system_resources"]
    communication_resources = system_resources[
        "communication_resource_utilization"]
    if set(communication_resources) != {resource_id}:
        raise ValueError(
            "TP=2 Production Trace communication resources disagree with "
            "the worker pool")
    communication = communication_resources[resource_id]
    if (communication["backend"] != "analytical_roofline_v1"
            or communication["backend_config_sha256"] is not None
            or communication["hardware_id"] != hardware_id
            or communication["link"] != "nvlink"
            or communication["timing_status"]
            != "projected_analytical_roofline"
            or communication["call_count"] != sum(operation_counts.values())
            or communication["operation_counts"] != operation_counts
            or communication["topology_counts"] != {
                "ring": sum(operation_counts.values())}
            or communication["bytes_by_semantics"] != {
                "per_rank_input": communication_bytes}):
        raise ValueError(
            "TP=2 Production Trace communication metrics disagree with "
            "collective calls")
    ordered_intervals = sorted(communication_intervals)
    communication_busy_time_s = 0.0
    previous_end_time_s = None
    for start_time_s, end_time_s in ordered_intervals:
        if (end_time_s <= start_time_s
                or (previous_end_time_s is not None
                    and start_time_s < previous_end_time_s
                    and not math.isclose(
                        start_time_s,
                        previous_end_time_s,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    ))):
            raise ValueError(
                "TP=2 Production Trace communication intervals overlap or "
                "are non-positive")
        communication_busy_time_s += end_time_s - start_time_s
        previous_end_time_s = end_time_s
    observation_window = system_resources["observation_window"]
    observation_window_s = observation_window["duration_s"]
    expected_idle_time_s = observation_window_s - communication_busy_time_s
    if (observation_window["basis"]
            != "simulation_origin_to_last_completion"
            or not math.isclose(
                observation_window_s,
                result["schedule_end_time_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or expected_idle_time_s < 0.0
            or not math.isclose(
                communication["busy_time_s"],
                communication_busy_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                communication["idle_time_s"],
                expected_idle_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                communication["utilization"],
                communication_busy_time_s / observation_window_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )):
        raise ValueError(
            "TP=2 Production Trace communication utilization disagrees "
            "with execution intervals")
    if (system_resources["link_utilization"] != {}
            or set(system_resources["worker_pool_utilization"]) != {"main"}
            or system_resources["worker_pool_utilization"]["main"][
                "kernel_interval_count"] != total_call_count
            or set(system_resources["scheduler_queue"]["by_pool"])
            != {"main"}):
        raise ValueError(
            "TP=2 Production Trace system resource metrics disagree with "
            "the worker pool")

    expected_metrics = _json_round_trip({
        "request_throughput_per_s": (
            canonical_metrics.request_throughput_per_s),
        "output_token_throughput_per_s": (
            canonical_metrics.output_token_throughput_per_s),
        "token_throughput_per_s": canonical_metrics.token_throughput_per_s,
        "latency_distributions": latency_distribution_manifest(
            canonical_metrics.requests),
        "slo_goodput": slo_goodput_manifest(
            expected_requests,
            canonical_metrics,
            canonical_schedule_trace.end_time_s,
            expected_slo_contract,
        ),
        "system_resources": system_resource_manifest(
            deployment,
            expected_requests,
            canonical_result,
        ),
        "critical_path": critical_path_manifest(
            deployment,
            expected_requests,
            canonical_result,
        ),
        "requests": {
            request_id: {
                "ttft_s": metric.ttft_s,
                "tpot_s": metric.tpot_s,
                "latency_s": metric.latency_s,
            }
            for request_id, metric in canonical_metrics.requests.items()
        },
    })
    if result["metrics"] != expected_metrics:
        raise ValueError(
            f"{context} metrics disagree with canonical Scheduler and "
            "execution trace")


def _tp2_collective_bytes_from_schedule_and_model(
    model: ModelConfig,
    schedule: dict[str, Any],
    collective_call: dict[str, Any],
    calls_by_id: dict[str, dict[str, Any]],
    nodes_by_id: dict[str, dict[str, Any]],
) -> int:
    requests = schedule["requests"]
    token_count = sum(
        request["input_tokens"] + request["output_tokens"]
        for request in requests
    )
    if token_count <= 0:
        raise ValueError(
            "TP=2 Production Trace Scheduler step has no token work")
    call_id = collective_call["call_id"]
    transfer = collective_call["kernel_spec"]["transfer"]
    if call_id.endswith(":tp-all-reduce"):
        base_id = call_id[:-len(":tp-all-reduce")]
    elif call_id.endswith(":tp-all-gather"):
        base_id = call_id[:-len(":tp-all-gather")]
    else:
        raise ValueError(
            "TP=2 Production Trace collective call ID is invalid")
    producer_ids = [
        f"{base_id}:tp-rank-0",
        f"{base_id}:tp-rank-1",
    ]
    if collective_call["dependencies"] != producer_ids:
        raise ValueError(
            "TP=2 Production Trace collective producers are invalid")

    if base_id.endswith(":token-embedding"):
        expected_operation = "all_reduce"
        expected_kernel_kind = "VECTOR"
        expected_kernel_variant = "gather"
        element_count = token_count * model.hidden_size
        expected_kernel_shape = {
            "elements": element_count,
            "rows": token_count,
            "hidden": model.hidden_size,
            "dtype_bytes": model.dtype_bytes,
            "element_count": element_count,
        }
        expected_workload_kind = "gather"
        expected_workload_variant = "embedding"
        expected_workload_shape = {
            "elements": element_count,
            "rows": token_count,
            "hidden": model.hidden_size,
        }
        expected_bytes = tp_collective_payload_bytes(
            model,
            expected_operation,
            token_count,
        )
    elif base_id.endswith(":o-proj"):
        expected_operation = "all_reduce"
        expected_kernel_kind = "GEMM"
        expected_kernel_variant = "gemm"
        m, n, k = infer_gemm_dimensions(
            model, "o_proj", token_count, tp_size=2)
        expected_kernel_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
            "dtype_bytes": model.dtype_bytes,
        }
        expected_workload_kind = "linear"
        expected_workload_variant = "gemm"
        expected_workload_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
        }
        expected_bytes = tp_collective_payload_bytes(
            model,
            expected_operation,
            token_count,
        )
    elif base_id.endswith(":down"):
        expected_operation = "all_reduce"
        expected_kernel_kind = "GEMM"
        expected_kernel_variant = "gemm"
        m, n, k = infer_gemm_dimensions(
            model, "down_proj", token_count, tp_size=2)
        expected_kernel_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
            "dtype_bytes": model.dtype_bytes,
        }
        expected_workload_kind = "linear"
        expected_workload_variant = "gemm"
        expected_workload_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
        }
        expected_bytes = tp_collective_payload_bytes(
            model,
            expected_operation,
            token_count,
        )
    elif base_id.endswith(":lm-head"):
        expected_operation = "all_gather"
        expected_kernel_kind = "GEMM"
        target_logit_rows = len(requests)
        m, n, k = infer_gemm_dimensions(
            model, "lm_head", target_logit_rows, tp_size=2)
        if m == 1:
            expected_kernel_variant = "gemv"
        elif m > 1:
            expected_kernel_variant = "gemm"
        else:
            raise ValueError(
                "TP=2 Production Trace LM-head rows must be positive")
        expected_kernel_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
            "dtype_bytes": model.dtype_bytes,
        }
        expected_workload_kind = "lm_head"
        expected_workload_variant = expected_kernel_variant
        expected_workload_shape = {
            "b": 1,
            "m": m,
            "n": n,
            "k": k,
        }
        expected_bytes = tp_collective_payload_bytes(
            model,
            expected_operation,
            target_logit_rows,
            local_output_width=n,
        )
    else:
        raise ValueError(
            "TP=2 Production Trace emitted a collective for an unsupported "
            "operator")
    if transfer["operation"] != expected_operation:
        raise ValueError(
            "TP=2 Production Trace collective operation disagrees with its "
            "producer")

    for producer_id in producer_ids:
        producer = calls_by_id[producer_id]
        kernel_spec = producer["kernel_spec"]
        node = nodes_by_id[producer_id]
        if (producer["resource"] != "compute"
                or kernel_spec["kind"] != expected_kernel_kind
                or kernel_spec["dtype"] != "bf16"
                or kernel_spec["variant"] != expected_kernel_variant
                or kernel_spec["shape"] != expected_kernel_shape
                or kernel_spec["transfer"] is not None
                or node["kind"] != expected_workload_kind
                or node["pool_name"] != producer["pool_name"]
                or node["rank"] != producer["rank"]
                or node["shape"] != expected_workload_shape
                or node["dtype"] != "bf16"
                or node["variant"] != expected_workload_variant):
            raise ValueError(
                "TP=2 Production Trace collective producer shape disagrees "
                "with the Scheduler/model")

    collective_node = nodes_by_id[call_id]
    expected_collective_attributes = {
        "algorithm": transfer["algorithm"],
        "bytes": transfer["bytes"],
        "bytes_semantics": transfer["bytes_semantics"],
        "backend": transfer["backend"],
        "backend_config_sha256": transfer["backend_config_sha256"],
        "direction": transfer["direction"],
        "link": transfer["link"],
        "topology": transfer["topology"],
        "participants": transfer["participants"],
        "operation": transfer["operation"],
        "source_rank": transfer["source_rank"],
        "destination_rank": transfer["destination_rank"],
        "ring_order": transfer["ring_order"],
        "reduction": transfer["reduction"],
        "resource_id": transfer["resource_id"],
    }
    if (collective_node["kind"] != "collective"
            or collective_node["pool_name"] != collective_call["pool_name"]
            or collective_node["rank"] != collective_call["rank"]
            or collective_node["shape"] != {}
            or collective_node["dtype"] is not None
            or collective_node["variant"] is not None
            or collective_node["attributes"]
            != expected_collective_attributes):
        raise ValueError(
            "TP=2 Production Trace collective lowering identity is invalid")
    return expected_bytes


def _validate_colocated_ep2_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    context = "TP=EP=2 Production Trace"
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    resource_id = "nvlink:main:tensor-fabric"
    collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            resource_id,
            "analytical_roofline_v1",
        )
        for operation in ("all_reduce", "all_gather")
    )
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(
                    tensor_parallel_size=2,
                    enable_expert_parallel=True,
                ),
                collectives=collectives,
            ),),
        ),
        num_layers=expected_model.num_hidden_layers,
        num_experts=expected_model.num_experts,
    )
    expected_plan = _json_round_trip(deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            f"{context} worker-pool topology disagrees with command manifest")

    scheduler = expected_simulation["scheduler"]
    runtime = result["scheduler_runtime"]["main"]
    expected_owners = [
        [f"model.layers.{layer_index}.self_attn.attn"]
        for layer_index in range(expected_model.num_hidden_layers)
    ]
    if runtime["scheduler_kv_cache_tensor_owners"] != expected_owners:
        raise ValueError(
            f"{context} Scheduler KV owners disagree with the pinned model")
    _validate_production_trace_capacity(
        result,
        scheduler,
        ("main",),
        context=context,
    )
    expected_kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(2, 1))
    capacity_plan = result["kv_capacity"]["scheduler_pools"][0]
    if capacity_plan["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError(
            f"{context} KV capacity disagrees with model sharding")
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["route_decisions"] != []
            or result["transfers"] != []
            or result["num_transfers"] != 0):
        raise ValueError(f"{context} routing or transfer state is invalid")
    if (result["kv_lifecycle_replay"] is not None
            or result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            }):
        raise ValueError(f"{context} unexpectedly enabled KV offload")
    mtp_scenario = result["mtp_scenario"]
    if (mtp_scenario["num_speculative_tokens"] != 0
            or mtp_scenario["acceptance_policy"] != "disabled"
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(f"{context} unexpectedly enabled MTP")

    routing_config = expected_simulation["expert_routing"]
    expert_routing = _explicit_expert_routing_input(routing_config, context)
    runtime_inputs = RuntimeInputs(
        expert_routing,
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    expected_runtime_inputs = {
        "expert_routing": _json_round_trip(expert_routing.manifest()),
        "attention_kv_layout": expected_simulation["attention_kv_layout"],
        "attention_kv_distribution": expected_simulation[
            "attention_kv_distribution"],
        "graph_execution": runtime_inputs.graph_execution.manifest(),
    }
    if result["execution_provenance"]["runtime_inputs"] != (
            expected_runtime_inputs):
        raise ValueError(
            f"{context} seeded expert-routing identity disagrees with the "
            "command manifest")

    artifact = result["execution_artifact"]
    artifact_trace = artifact["schedule_trace"]
    if (artifact_trace["route_decision_version"] != 2
            or artifact_trace["transfer_trace_version"] != 2
            or artifact_trace["route_decisions"] != []
            or artifact_trace["transfers"] != []
            or artifact["transfers"] != []):
        raise ValueError(
            f"{context} execution artifact contains a route or transfer")
    artifact_body = {
        key: artifact[key]
        for key in artifact
        if key != "artifact_sha256"
    }
    expected_artifact_sha256 = hashlib.sha256(json.dumps(
        artifact_body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    if artifact["artifact_sha256"] != expected_artifact_sha256:
        raise ValueError(f"{context} execution artifact SHA256 is invalid")
    steps = artifact["steps"]
    counts = artifact["counts"]
    if (not steps
            or result["num_steps"] != len(steps)
            or counts["schedule_steps"] != len(steps)
            or counts["simulation_steps"] != len(steps)
            or counts["simulation_transfers"] != 0
            or counts["trace_transfers"] != 0
            or counts["modeled_kv_offload_transfers"] != 0):
        raise ValueError(f"{context} execution artifact counts disagree")
    trace_entries = artifact_trace["entries"]
    entries_by_id = {
        entry["step_id"]: entry for entry in trace_entries
    }
    if (len(entries_by_id) != len(trace_entries)
            or len(entries_by_id) != len(steps)):
        raise ValueError(f"{context} schedule trace coverage disagrees")

    hardware = _load_expected_hardware(
        expected_simulation, hardware_id, context)
    hardware_by_id = {hardware_id: hardware}
    total_call_count = 0
    fused_moe_call_count = 0
    communication_call_count = 0
    communication_bytes = 0
    communication_operation_counts: dict[str, int] = {}
    communication_topology_counts: dict[str, int] = {}
    communication_bytes_by_semantics: dict[str, int] = {}
    communication_intervals = []
    worker_pool_intervals = []
    all_step_ids = set()
    all_call_ids = set()
    backend_steps_by_id = {}
    simulation_steps = []
    execution = ServingExecutionContext(
        expected_model,
        deployment,
        hardware_by_id,
        expected_simulation["launch_policy"],
        expected_simulation["prediction_method"],
        runtime_inputs,
        execution_mode=result["execution_mode"],
    )
    for step in steps:
        step_id = step["step_id"]
        if (step_id in all_step_ids
                or step_id not in entries_by_id
                or step["pool_name"] != "main"):
            raise ValueError(f"{context} execution step identity is invalid")
        all_step_ids.add(step_id)
        entry = entries_by_id[step_id]
        if (entry["pool_name"] != "main"
                or step["schedule_interval"] != {
                    "start_time_s": entry["start_time_s"],
                    "end_time_s": entry["end_time_s"],
                }):
            raise ValueError(f"{context} schedule interval identity is invalid")
        try:
            backend_step = _backend_step_from_output_manifest(
                step["backend_step"])
            components = execution.execute_step(
                backend_step.schedule,
                kv_cache_events=backend_step.kv_cache_events,
            )
            workload = components.workload
            kernel_plan = components.kernel_plan
            predictions = components.predictions
            timeline = components.execution_timeline
        except (TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                f"{context} step cannot be canonically rematerialized") \
                from error
        backend_steps_by_id[step_id] = backend_step
        expected_components = _json_round_trip(
            execution_step_components_manifest(
                workload,
                kernel_plan,
                predictions,
                timeline,
                origin_time_s=step["schedule_interval"]["start_time_s"],
            ))
        simulation_steps.append(SimulationStep(
            backend_step,
            workload,
            kernel_plan,
            predictions,
            timeline,
        ))
        actual_components = {
            field: step[field]
            for field in expected_components
        }
        if actual_components != expected_components:
            raise ValueError(
                f"{context} canonical Scheduler-to-KernelSpec "
                "rematerialization disagrees with the execution artifact")
        _validate_canonical_production_trace_step_duration(
            step,
            timeline.duration_s,
            context=context,
        )

        calls = step["kernel_plan"]["calls"]
        call_ids = tuple(call["call_id"] for call in calls)
        if all_call_ids.intersection(call_ids):
            raise ValueError(
                f"{context} contains duplicate cross-step call IDs")
        all_call_ids.update(call_ids)
        fused_nodes = tuple(
            node for node in step["workload_graph"]["nodes"]
            if node["kind"] == "fused_moe"
        )
        invocations: dict[str, list[dict[str, Any]]] = {}
        for node in fused_nodes:
            attributes = node["attributes"]
            invocation_id = attributes["moe_invocation_id"]
            if invocation_id not in invocations:
                invocations[invocation_id] = []
            invocations[invocation_id].append(node)
            rank = node["rank"]
            global_counts = attributes["tokens_per_expert"]
            local_counts = attributes["local_tokens_per_expert"]
            routing_provenance = attributes["routing_provenance"]
            if (rank not in (0, 1)
                    or attributes["expert_parallel_size"] != 2
                    or attributes["expert_rank"] != rank
                    or attributes["expert_placement_strategy"] != "linear"
                    or attributes["owned_experts"] != [rank]
                    or len(global_counts) != 2
                    or len(local_counts) != 2
                    or sum(global_counts)
                    != node["shape"]["tokens"] * node["shape"]["topk"]
                    or local_counts != [
                        global_counts[0] if rank == 0 else 0,
                        global_counts[1] if rank == 1 else 0,
                    ]
                    or routing_provenance["source_kind"]
                    != "explicit_assumption"
                    or routing_provenance["source_id"]
                    != expert_routing.source_id
                    or routing_provenance["algorithm"]
                    != expert_routing.algorithm
                    or routing_provenance["distribution"]
                    != expert_routing.distribution
                    or routing_provenance["hot_expert_count"]
                    != expert_routing.hot_expert_count
                    or routing_provenance["hot_expert_weight"]
                    != expert_routing.hot_expert_weight
                    or routing_provenance["root_seed"]
                    != expert_routing.root_seed
                    or routing_provenance["policy_descriptor_sha256"]
                    != expert_routing.policy_descriptor_sha256
                    or routing_provenance["mechanism_source"] != "Modeled"
                    or routing_provenance["validation_status"] != "Projected"):
                raise ValueError(
                    f"{context} FUSED_MOE routing or ownership is invalid")
        if (len(invocations) != expected_model.num_hidden_layers
                or any(
                    {node["rank"] for node in nodes} != {0, 1}
                    or len(nodes) != 2
                    for nodes in invocations.values()
                )):
            raise ValueError(
                f"{context} FUSED_MOE invocation coverage is invalid")

        intervals_by_id = {
            interval["call_id"]: interval
            for interval in step["execution_timeline"]["intervals"]
        }
        worker_pool_intervals.extend(
            (
                interval["absolute_start_time_s"],
                interval["absolute_end_time_s"],
            )
            for interval in intervals_by_id.values()
        )
        for call in calls:
            kind = call["kernel_spec"]["kind"]
            if kind == "FUSED_MOE":
                fused_moe_call_count += 1
            elif kind == "COMMUNICATION":
                transfer = call["kernel_spec"]["transfer"]
                if (transfer["operation"] == "all_to_all"
                        or transfer["resource_id"] != resource_id
                        or transfer["backend"] != "analytical_roofline_v1"):
                    raise ValueError(
                        f"{context} emitted an unsupported collective")
                operation = transfer["operation"]
                topology = transfer["topology"]
                semantics = transfer["bytes_semantics"]
                if operation not in communication_operation_counts:
                    communication_operation_counts[operation] = 0
                if topology not in communication_topology_counts:
                    communication_topology_counts[topology] = 0
                if semantics not in communication_bytes_by_semantics:
                    communication_bytes_by_semantics[semantics] = 0
                communication_operation_counts[operation] += 1
                communication_topology_counts[topology] += 1
                communication_bytes_by_semantics[semantics] += transfer["bytes"]
                communication_bytes += transfer["bytes"]
                communication_call_count += 1
                interval = intervals_by_id[call["call_id"]]
                communication_intervals.append((
                    interval["absolute_start_time_s"],
                    interval["absolute_end_time_s"],
                ))
            elif kind in {"VECTOR", "GEMM", "FLASH_ATTN"}:
                pass
            else:
                raise ValueError(
                    f"{context} emitted unsupported kernel kind {kind!r}")
        if not fused_nodes:
            raise ValueError(f"{context} step emitted no FUSED_MOE workload")
        total_call_count += len(calls)
    if (set(entries_by_id) != all_step_ids
            or counts["kernel_calls"] != total_call_count
            or result["num_kernel_calls"] != total_call_count
            or fused_moe_call_count <= 0
            or communication_call_count <= 0
            or communication_bytes <= 0
            or set(communication_operation_counts)
            != {"all_gather", "all_reduce"}):
        raise ValueError(
            f"{context} kernel, MoE, or collective counts disagree")

    canonical_schedule_trace = ScheduleTrace(
        ScheduleTraceEntry(
            backend_steps_by_id[entry["step_id"]],
            entry["start_time_s"],
            entry["end_time_s"],
        )
        for entry in trace_entries
    )
    _validate_production_trace_scheduler_binding(
        expected_requests,
        canonical_schedule_trace,
        scheduler["kv_block_size_tokens"],
        result,
        max_requests=scheduler["max_requests"],
        context=context,
    )
    _replay_pinned_parallel_scheduler(
        expected_requests,
        canonical_schedule_trace,
        scheduler,
        expected_model,
        expected_model_path,
        runtime,
        num_kv_blocks=runtime["num_kv_blocks"],
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        enable_expert_parallel=True,
        expected_mtp=expected_simulation["mtp"],
        context=context,
    )

    canonical_metrics = compute_serving_metrics(
        expected_requests,
        canonical_schedule_trace,
    )
    canonical_result = SimulationResult(
        canonical_schedule_trace,
        canonical_metrics,
        tuple(simulation_steps),
    )
    expected_artifact = _json_round_trip(
        execution_artifact_manifest(
            canonical_result,
            performance_timing_status=(
                _expected_performance_timing_status(expected_simulation)),
        ))
    if artifact != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")
    expected_metrics = _json_round_trip({
        "request_throughput_per_s": (
            canonical_metrics.request_throughput_per_s),
        "output_token_throughput_per_s": (
            canonical_metrics.output_token_throughput_per_s),
        "token_throughput_per_s": canonical_metrics.token_throughput_per_s,
        "latency_distributions": latency_distribution_manifest(
            canonical_metrics.requests),
        "slo_goodput": slo_goodput_manifest(
            expected_requests,
            canonical_metrics,
            canonical_schedule_trace.end_time_s,
            expected_slo_contract,
        ),
        "system_resources": system_resource_manifest(
            deployment,
            expected_requests,
            canonical_result,
        ),
        "critical_path": critical_path_manifest(
            deployment,
            expected_requests,
            canonical_result,
        ),
        "requests": {
            request_id: {
                "ttft_s": metric.ttft_s,
                "tpot_s": metric.tpot_s,
                "latency_s": metric.latency_s,
            }
            for request_id, metric in canonical_metrics.requests.items()
        },
    })
    if result["metrics"] != expected_metrics:
        raise ValueError(
            f"{context} metrics disagree with canonical Scheduler and "
            "execution trace")

    system_resources = result["metrics"]["system_resources"]
    communication_resources = system_resources[
        "communication_resource_utilization"]
    if set(communication_resources) != {resource_id}:
        raise ValueError(
            f"{context} communication resources disagree with the pool")
    communication = communication_resources[resource_id]
    observation_window = system_resources["observation_window"]
    observation_window_s = observation_window["duration_s"]
    communication_busy_time_s = _interval_union_duration_from_output(
        communication_intervals,
        observation_window_s,
        context=f"{context} communication fabric",
    )
    communication_idle_time_s = (
        observation_window_s - communication_busy_time_s)
    if (communication["backend"] != "analytical_roofline_v1"
            or communication["backend_config_sha256"] is not None
            or communication["hardware_id"] != hardware_id
            or communication["link"] != "nvlink"
            or communication["timing_status"]
            != "projected_analytical_roofline"
            or communication["call_count"] != communication_call_count
            or communication["operation_counts"]
            != communication_operation_counts
            or communication["topology_counts"]
            != communication_topology_counts
            or communication["bytes_by_semantics"]
            != communication_bytes_by_semantics
            or not math.isclose(
                communication["busy_time_s"],
                communication_busy_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                communication["idle_time_s"],
                communication_idle_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                communication["utilization"],
                communication_busy_time_s / observation_window_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )):
        raise ValueError(
            f"{context} communication metrics disagree with collective calls")
    worker_pool_busy_time_s = _interval_union_duration_from_output(
        worker_pool_intervals,
        observation_window_s,
        context=f"{context} worker pool",
    )
    worker_pool_idle_time_s = observation_window_s - worker_pool_busy_time_s
    worker_pool_resources = system_resources["worker_pool_utilization"]
    if set(worker_pool_resources) != {"main"}:
        raise ValueError(
            f"{context} worker-pool utilization coverage is invalid")
    worker_pool = worker_pool_resources["main"]
    if (observation_window["basis"]
            != "simulation_origin_to_last_completion"
            or not math.isclose(
                observation_window_s,
                result["schedule_end_time_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or system_resources["link_utilization"] != {}
            or set(worker_pool) != {
                "kernel_interval_count", "busy_time_s", "idle_time_s",
                "utilization"}
            or worker_pool["kernel_interval_count"] != total_call_count
            or not math.isclose(
                worker_pool["busy_time_s"],
                worker_pool_busy_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                worker_pool["idle_time_s"],
                worker_pool_idle_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                worker_pool["utilization"],
                worker_pool_busy_time_s / observation_window_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or set(system_resources["scheduler_queue"]["by_pool"])
            != {"main"}):
        raise ValueError(
            f"{context} system resource metrics disagree with the pool")


def _validate_colocated_pp2_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
) -> None:
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    resource_id = "nvlink:main:pipeline-fabric"
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(pipeline_parallel_size=2),
                pipeline_p2p=(PipelineP2PConfig(
                    0,
                    1,
                    "send_recv",
                    "nvlink",
                    "point_to_point",
                    resource_id,
                    "analytical_roofline_v1",
                ),),
            ),),
        ),
        num_layers=expected_model.num_hidden_layers,
    )
    expected_plan = _json_round_trip(
        deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            "PP=2 Production Trace worker-pool topology disagrees with "
            "command manifest")

    scheduler = expected_simulation["scheduler"]
    runtime = result["scheduler_runtime"]["main"]
    local_layer_count = expected_model.num_hidden_layers // 2
    expected_owners = [
        [f"model.layers.{layer_index}.self_attn.attn"]
        for layer_index in range(local_layer_count)
    ]
    if runtime["scheduler_kv_cache_tensor_owners"] != expected_owners:
        raise ValueError(
            "PP=2 Production Trace Scheduler KV owners disagree with the "
            "pinned Qwen3 stage")
    _validate_production_trace_capacity(
        result,
        scheduler,
        ("main",),
        context="PP=2 Production Trace",
    )
    expected_kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(1, 2))
    capacity_plan = result["kv_capacity"]["scheduler_pools"][0]
    if capacity_plan["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError(
            "PP=2 Production Trace KV capacity disagrees with model "
            "stage sharding")
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["route_decisions"] != []
            or result["transfers"] != []
            or result["num_transfers"] != 0):
        raise ValueError(
            "PP=2 Production Trace routing or transfer state is invalid")
    if (result["kv_lifecycle_replay"] is not None
            or result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            }):
        raise ValueError(
            "PP=2 Production Trace unexpectedly enabled KV offload")
    mtp_scenario = result["mtp_scenario"]
    if (mtp_scenario["num_speculative_tokens"] != 0
            or mtp_scenario["acceptance_policy"] != "disabled"
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(
            "PP=2 Production Trace unexpectedly enabled MTP")

    artifact = result["execution_artifact"]
    artifact_trace = artifact["schedule_trace"]
    if (artifact_trace["route_decision_version"] != 2
            or artifact_trace["transfer_trace_version"] != 2
            or artifact_trace["route_decisions"] != []
            or artifact_trace["transfers"] != []
            or artifact["transfers"] != []):
        raise ValueError(
            "PP=2 Production Trace execution artifact contains a route or "
            "transfer")
    steps = artifact["steps"]
    counts = artifact["counts"]
    if (not steps
            or result["num_steps"] != len(steps)
            or counts["schedule_steps"] != len(steps)
            or counts["simulation_steps"] != len(steps)
            or counts["simulation_transfers"] != 0
            or counts["trace_transfers"] != 0
            or counts["modeled_kv_offload_transfers"] != 0):
        raise ValueError(
            "PP=2 Production Trace execution artifact counts disagree")
    trace_entries = artifact_trace["entries"]
    entries_by_id = {
        entry["step_id"]: entry for entry in trace_entries
    }
    if (len(entries_by_id) != len(trace_entries)
            or len(entries_by_id) != len(steps)):
        raise ValueError(
            "PP=2 Production Trace schedule trace coverage disagrees")

    hardware = _load_expected_hardware(
        expected_simulation, hardware_id, context)
    hardware_by_id = {hardware_id: hardware}
    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    total_call_count = 0
    communication_call_count = 0
    communication_bytes = 0
    communication_intervals = []
    worker_pool_intervals = []
    all_step_ids = set()
    all_call_ids = set()
    backend_steps_by_id = {}
    execution = ServingExecutionContext(
        expected_model,
        deployment,
        hardware_by_id,
        expected_simulation["launch_policy"],
        expected_simulation["prediction_method"],
        runtime_inputs,
        execution_mode=result["execution_mode"],
    )
    for step in steps:
        step_id = step["step_id"]
        if (step_id in all_step_ids
                or step_id not in entries_by_id
                or step["pool_name"] != "main"):
            raise ValueError(
                "PP=2 Production Trace execution step identity is invalid")
        all_step_ids.add(step_id)
        entry = entries_by_id[step_id]
        if (entry["pool_name"] != "main"
                or step["schedule_interval"] != {
                    "start_time_s": entry["start_time_s"],
                    "end_time_s": entry["end_time_s"],
                }):
            raise ValueError(
                "PP=2 Production Trace schedule interval identity is invalid")
        try:
            backend_step = _backend_step_from_output_manifest(
                step["backend_step"])
            components = execution.execute_step(
                backend_step.schedule,
                kv_cache_events=backend_step.kv_cache_events,
            )
            workload = components.workload
            kernel_plan = components.kernel_plan
            predictions = components.predictions
            timeline = components.execution_timeline
        except (TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                "PP=2 Production Trace step cannot be canonically "
                "rematerialized") from error
        backend_steps_by_id[step_id] = backend_step
        expected_components = _json_round_trip(
            execution_step_components_manifest(
                workload,
                kernel_plan,
                predictions,
                timeline,
                origin_time_s=step["schedule_interval"]["start_time_s"],
            ))
        actual_components = {
            field: step[field]
            for field in expected_components
        }
        if actual_components != expected_components:
            raise ValueError(
                "PP=2 Production Trace canonical Scheduler-to-KernelSpec "
                "rematerialization disagrees with the execution artifact")
        _validate_canonical_production_trace_step_duration(
            step,
            timeline.duration_s,
            context="PP=2 Production Trace",
        )
        calls = step["kernel_plan"]["calls"]
        call_ids = tuple(call["call_id"] for call in calls)
        if all_call_ids.intersection(call_ids):
            raise ValueError(
                "PP=2 Production Trace contains duplicate cross-step call IDs")
        all_call_ids.update(call_ids)
        p2p_calls = tuple(
            call for call in calls
            if call["kernel_spec"]["kind"] == "COMMUNICATION"
        )
        if (len(p2p_calls) != 2
                or tuple(
                    call["kernel_spec"]["transfer"]["operation"]
                    for call in p2p_calls
                ) != ("send_recv", "send_recv")):
            raise ValueError(
                "PP=2 Production Trace step must contain the two pipeline "
                "payload transfers")
        total_call_count += len(calls)
        communication_call_count += len(p2p_calls)
        communication_bytes += sum(
            call["kernel_spec"]["transfer"]["bytes"]
            for call in p2p_calls
        )
        intervals_by_id = {
            interval["call_id"]: interval
            for interval in step["execution_timeline"]["intervals"]
        }
        worker_pool_intervals.extend(
            (
                interval["absolute_start_time_s"],
                interval["absolute_end_time_s"],
            )
            for interval in intervals_by_id.values()
        )
        communication_intervals.extend(
            (
                intervals_by_id[call["call_id"]][
                    "absolute_start_time_s"],
                intervals_by_id[call["call_id"]][
                    "absolute_end_time_s"],
            )
            for call in p2p_calls
        )
    if (set(entries_by_id) != all_step_ids
            or counts["kernel_calls"] != total_call_count
            or result["num_kernel_calls"] != total_call_count):
        raise ValueError(
            "PP=2 Production Trace kernel or schedule counts disagree")
    canonical_schedule_trace = ScheduleTrace(
        ScheduleTraceEntry(
            backend_steps_by_id[entry["step_id"]],
            entry["start_time_s"],
            entry["end_time_s"],
        )
        for entry in trace_entries
    )
    _validate_production_trace_scheduler_binding(
        expected_requests,
        canonical_schedule_trace,
        scheduler["kv_block_size_tokens"],
        result,
        max_requests=scheduler["max_requests"],
        context="PP=2 Production Trace",
    )
    _replay_pinned_parallel_scheduler(
        expected_requests,
        canonical_schedule_trace,
        scheduler,
        expected_model,
        expected_model_path,
        runtime,
        num_kv_blocks=runtime["num_kv_blocks"],
        tensor_parallel_size=1,
        pipeline_parallel_size=2,
        enable_expert_parallel=False,
        expected_mtp=expected_simulation["mtp"],
        context="PP=2 Production Trace",
    )

    system_resources = result["metrics"]["system_resources"]
    communication_resources = system_resources[
        "communication_resource_utilization"]
    if set(communication_resources) != {resource_id}:
        raise ValueError(
            "PP=2 Production Trace communication resources disagree with "
            "the pipeline plan")
    communication = communication_resources[resource_id]
    if (communication["backend"] != "analytical_roofline_v1"
            or communication["backend_config_sha256"] is not None
            or communication["hardware_id"] != hardware_id
            or communication["link"] != "nvlink"
            or communication["timing_status"]
            != "projected_analytical_roofline"
            or communication["call_count"] != communication_call_count
            or communication["operation_counts"] != {
                "send_recv": communication_call_count}
            or communication["topology_counts"] != {
                "point_to_point": communication_call_count}
            or communication["bytes_by_semantics"] != {
                "payload": communication_bytes}):
        raise ValueError(
            "PP=2 Production Trace communication metrics disagree with "
            "pipeline P2P calls")
    ordered_intervals = sorted(communication_intervals)
    communication_busy_time_s = 0.0
    previous_end_time_s = None
    for start_time_s, end_time_s in ordered_intervals:
        if (end_time_s <= start_time_s
                or (previous_end_time_s is not None
                    and start_time_s < previous_end_time_s
                    and not math.isclose(
                        start_time_s,
                        previous_end_time_s,
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    ))):
            raise ValueError(
                "PP=2 Production Trace communication intervals overlap or "
                "are non-positive")
        communication_busy_time_s += end_time_s - start_time_s
        previous_end_time_s = end_time_s
    observation_window = system_resources["observation_window"]
    observation_window_s = observation_window["duration_s"]
    expected_idle_time_s = observation_window_s - communication_busy_time_s
    if (observation_window["basis"]
            != "simulation_origin_to_last_completion"
            or not math.isclose(
                observation_window_s,
                result["schedule_end_time_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or expected_idle_time_s < 0.0
            or not math.isclose(
                communication["busy_time_s"],
                communication_busy_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                communication["idle_time_s"],
                expected_idle_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                communication["utilization"],
                communication_busy_time_s / observation_window_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )):
        raise ValueError(
            "PP=2 Production Trace communication utilization disagrees "
            "with execution intervals")
    worker_pool_busy_time_s = _interval_union_duration_from_output(
        worker_pool_intervals,
        observation_window_s,
        context="PP=2 Production Trace worker pool",
    )
    worker_pool_idle_time_s = (
        observation_window_s - worker_pool_busy_time_s)
    worker_pool_resources = system_resources["worker_pool_utilization"]
    if set(worker_pool_resources) != {"main"}:
        raise ValueError(
            "PP=2 Production Trace worker-pool utilization coverage "
            "disagrees with the deployment")
    worker_pool = worker_pool_resources["main"]
    if (system_resources["link_utilization"] != {}
            or set(worker_pool) != {
                "kernel_interval_count", "busy_time_s", "idle_time_s",
                "utilization"}
            or worker_pool["kernel_interval_count"] != total_call_count
            or not math.isclose(
                worker_pool["busy_time_s"],
                worker_pool_busy_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                worker_pool["idle_time_s"],
                worker_pool_idle_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                worker_pool["utilization"],
                worker_pool_busy_time_s / observation_window_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or set(system_resources["scheduler_queue"]["by_pool"])
            != {"main"}):
        raise ValueError(
            "PP=2 Production Trace system resource metrics disagree with "
            "the worker pool")


def _validate_colocated_tp2_pp2_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    context = "TP=2, PP=2 Production Trace"
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    tensor_resource_id = "nvlink:main:tensor-fabric"
    pipeline_resource_id = "nvlink:main:pipeline-fabric"
    collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            tensor_resource_id,
            "analytical_roofline_v1",
        )
        for operation in ("all_reduce", "all_gather")
    )
    deployment = build_colocated_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(
                    tensor_parallel_size=2,
                    pipeline_parallel_size=2,
                ),
                collectives=collectives,
                pipeline_p2p=(PipelineP2PConfig(
                    0,
                    1,
                    "send_recv",
                    "nvlink",
                    "point_to_point",
                    pipeline_resource_id,
                    "analytical_roofline_v1",
                ),),
            ),),
        ),
        num_layers=expected_model.num_hidden_layers,
        num_experts=expected_model.num_experts,
    )
    expected_plan = _json_round_trip(deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            f"{context} worker-pool topology disagrees with command manifest")

    scheduler = expected_simulation["scheduler"]
    runtime = result["scheduler_runtime"]["main"]
    local_layer_count = expected_model.num_hidden_layers // 2
    expected_owners = [
        [f"model.layers.{layer_index}.self_attn.attn"]
        for layer_index in range(local_layer_count)
    ]
    if runtime["scheduler_kv_cache_tensor_owners"] != expected_owners:
        raise ValueError(
            f"{context} Scheduler KV owners disagree with the pinned model")
    _validate_production_trace_capacity(
        result,
        scheduler,
        ("main",),
        context=context,
    )
    expected_kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(2, 2))
    capacity_plan = result["kv_capacity"]["scheduler_pools"][0]
    if capacity_plan["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError(
            f"{context} KV capacity disagrees with TP/PP model sharding")
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["route_decisions"] != []
            or result["transfers"] != []
            or result["num_transfers"] != 0):
        raise ValueError(f"{context} routing or transfer state is invalid")
    if (result["kv_lifecycle_replay"] is not None
            or result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            }):
        raise ValueError(f"{context} unexpectedly enabled KV offload")
    mtp_scenario = result["mtp_scenario"]
    if (mtp_scenario["num_speculative_tokens"] != 0
            or mtp_scenario["acceptance_policy"] != "disabled"
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(f"{context} unexpectedly enabled MTP")

    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    expected_runtime_inputs = {
        "expert_routing": _json_round_trip(
            runtime_inputs.expert_routing.manifest()),
        "attention_kv_layout": expected_simulation["attention_kv_layout"],
        "attention_kv_distribution": expected_simulation[
            "attention_kv_distribution"],
        "graph_execution": runtime_inputs.graph_execution.manifest(),
    }
    if result["execution_provenance"]["runtime_inputs"] != (
            expected_runtime_inputs):
        raise ValueError(
            f"{context} runtime inputs disagree with the command manifest")

    artifact = result["execution_artifact"]
    artifact_trace = artifact["schedule_trace"]
    if (artifact_trace["route_decision_version"] != 2
            or artifact_trace["transfer_trace_version"] != 2
            or artifact_trace["route_decisions"] != []
            or artifact_trace["transfers"] != []
            or artifact["transfers"] != []):
        raise ValueError(f"{context} execution artifact contains a transfer")
    steps = artifact["steps"]
    counts = artifact["counts"]
    if (not steps
            or result["num_steps"] != len(steps)
            or counts["schedule_steps"] != len(steps)
            or counts["simulation_steps"] != len(steps)
            or counts["simulation_transfers"] != 0
            or counts["trace_transfers"] != 0
            or counts["modeled_kv_offload_transfers"] != 0):
        raise ValueError(f"{context} execution artifact counts disagree")
    trace_entries = artifact_trace["entries"]
    entries_by_id = {
        entry["step_id"]: entry for entry in trace_entries
    }
    if (len(entries_by_id) != len(trace_entries)
            or len(entries_by_id) != len(steps)):
        raise ValueError(f"{context} schedule trace coverage disagrees")

    hardware = _load_expected_hardware(
        expected_simulation, hardware_id, context)
    hardware_by_id = {hardware_id: hardware}
    resource_stats = {
        tensor_resource_id: {
            "call_count": 0,
            "operation_counts": {"all_gather": 0, "all_reduce": 0},
            "topology_counts": {"ring": 0},
            "bytes_by_semantics": {"per_rank_input": 0},
            "intervals": [],
        },
        pipeline_resource_id: {
            "call_count": 0,
            "operation_counts": {"send_recv": 0},
            "topology_counts": {"point_to_point": 0},
            "bytes_by_semantics": {"payload": 0},
            "intervals": [],
        },
    }
    total_call_count = 0
    worker_pool_intervals = []
    all_step_ids = set()
    all_call_ids = set()
    backend_steps_by_id = {}
    simulation_steps = []
    execution = ServingExecutionContext(
        expected_model,
        deployment,
        hardware_by_id,
        expected_simulation["launch_policy"],
        expected_simulation["prediction_method"],
        runtime_inputs,
        execution_mode=result["execution_mode"],
    )
    for step in steps:
        step_id = step["step_id"]
        if (step_id in all_step_ids
                or step_id not in entries_by_id
                or step["pool_name"] != "main"):
            raise ValueError(f"{context} execution step identity is invalid")
        all_step_ids.add(step_id)
        entry = entries_by_id[step_id]
        if (entry["pool_name"] != "main"
                or step["schedule_interval"] != {
                    "start_time_s": entry["start_time_s"],
                    "end_time_s": entry["end_time_s"],
                }):
            raise ValueError(f"{context} schedule interval identity is invalid")
        try:
            backend_step = _backend_step_from_output_manifest(
                step["backend_step"])
            components = execution.execute_step(
                backend_step.schedule,
                kv_cache_events=backend_step.kv_cache_events,
            )
            workload = components.workload
            kernel_plan = components.kernel_plan
            predictions = components.predictions
            timeline = components.execution_timeline
        except (TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                f"{context} step cannot be canonically rematerialized") \
                from error
        backend_steps_by_id[step_id] = backend_step
        expected_components = _json_round_trip(
            execution_step_components_manifest(
                workload,
                kernel_plan,
                predictions,
                timeline,
                origin_time_s=step["schedule_interval"]["start_time_s"],
            ))
        actual_components = {
            field: step[field]
            for field in expected_components
        }
        if actual_components != expected_components:
            raise ValueError(
                f"{context} canonical Scheduler-to-KernelSpec "
                "rematerialization disagrees with the execution artifact")
        _validate_canonical_production_trace_step_duration(
            step,
            timeline.duration_s,
            context=context,
        )
        simulation_steps.append(SimulationStep(
            backend_step,
            workload,
            kernel_plan,
            predictions,
            timeline,
        ))

        calls = step["kernel_plan"]["calls"]
        call_ids = tuple(call["call_id"] for call in calls)
        if (not calls
                or all_call_ids.intersection(call_ids)
                or {call["rank"] for call in calls} != {0, 1, 2, 3}
                or any(
                    call["pool_name"] != "main"
                    or call["kernel_spec"]["hardware_id"] != hardware_id
                    for call in calls
                )):
            raise ValueError(f"{context} four-rank coverage is invalid")
        all_call_ids.update(call_ids)
        calls_by_id = {call["call_id"]: call for call in calls}
        communication_calls = tuple(
            call for call in calls
            if call["kernel_spec"]["kind"] == "COMMUNICATION"
        )
        step_operation_counts = {
            "all_gather": 0,
            "all_reduce": 0,
            "send_recv": 0,
        }
        for call in communication_calls:
            operation = call["kernel_spec"]["transfer"]["operation"]
            if operation not in step_operation_counts:
                raise ValueError(
                    f"{context} emitted unsupported communication {operation!r}")
            step_operation_counts[operation] += 1
        expected_all_reduce_count = (
            1 + 2 * expected_model.num_hidden_layers)
        expected_communication_count = (
            expected_all_reduce_count + 3 + 4)
        if (len(communication_calls) != expected_communication_count
                or step_operation_counts != {
                    "all_gather": 3,
                    "all_reduce": expected_all_reduce_count,
                    "send_recv": 4,
                }):
            raise ValueError(
                f"{context} communication operation counts disagree")

        source_frontier_id = (
            f"{step_id}:layer-{local_layer_count - 1}:down:tp-all-reduce")
        expected_lane_pairs = ((0, 2), (1, 3))
        p2p_ids_by_tensor = {}
        for tensor_name in ("hidden_states", "residual"):
            tensor_slug = tensor_name.replace("_", "-")
            p2p_ids = tuple(
                f"{step_id}:pp-0-to-1:{tensor_slug}:tp-rank-{lane_index}"
                for lane_index in range(2)
            )
            p2p_ids_by_tensor[tensor_name] = p2p_ids
            boundary_id = (
                f"{step_id}:pp-0-to-1:{tensor_slug}:tp-all-gather")
            if (any(call_id not in calls_by_id for call_id in p2p_ids)
                    or boundary_id not in calls_by_id):
                raise ValueError(
                    f"{context} pipeline boundary call coverage is invalid")
            for call_id, lane_pair in zip(
                    p2p_ids, expected_lane_pairs, strict=True):
                call = calls_by_id[call_id]
                transfer = call["kernel_spec"]["transfer"]
                if (call["dependencies"] != [source_frontier_id]
                        or transfer["participants"] != list(lane_pair)
                        or transfer["source_rank"] != lane_pair[0]
                        or transfer["destination_rank"] != lane_pair[1]):
                    raise ValueError(
                        f"{context} source-to-P2P lane dependency is invalid")
            boundary_call = calls_by_id[boundary_id]
            boundary_transfer = boundary_call["kernel_spec"]["transfer"]
            if (boundary_call["dependencies"] != list(p2p_ids)
                    or boundary_transfer["participants"] != [2, 3]
                    or boundary_transfer["ring_order"] != [2, 3]):
                raise ValueError(
                    f"{context} P2P-to-all-gather dependency is invalid")
        boundary_ids = tuple(
            f"{step_id}:pp-0-to-1:{tensor_name.replace('_', '-')}:"
            "tp-all-gather"
            for tensor_name in ("hidden_states", "residual")
        )
        destination_layer_index = local_layer_count
        for lane_index in range(2):
            destination_norm_id = (
                f"{step_id}:layer-{destination_layer_index}:input-norm:"
                f"tp-rank-{lane_index}")
            if (destination_norm_id not in calls_by_id
                    or calls_by_id[destination_norm_id]["dependencies"]
                    != list(boundary_ids)):
                raise ValueError(
                    f"{context} boundary-to-destination dependency is invalid")

        token_count = sum(
            request["input_tokens"] + request["output_tokens"]
            for request in step["backend_step"]["schedule"]["requests"]
        )
        if token_count <= 0:
            raise ValueError(f"{context} Scheduler step contains no token work")
        full_tensor_bytes = (
            token_count * expected_model.hidden_size * expected_model.dtype_bytes)
        if full_tensor_bytes % 2:
            raise ValueError(f"{context} boundary bytes do not divide TP size")
        expected_wire_bytes = full_tensor_bytes // 2
        source_transfer = calls_by_id[source_frontier_id]["kernel_spec"][
            "transfer"]
        if (source_transfer["operation"] != "all_reduce"
                or source_transfer["participants"] != [0, 1]
                or source_transfer["bytes"] != full_tensor_bytes):
            raise ValueError(f"{context} source all-reduce bytes are invalid")
        for tensor_name in ("hidden_states", "residual"):
            for call_id in p2p_ids_by_tensor[tensor_name]:
                if calls_by_id[call_id]["kernel_spec"]["transfer"][
                        "bytes"] != expected_wire_bytes:
                    raise ValueError(f"{context} P2P wire bytes are invalid")
            boundary_id = (
                f"{step_id}:pp-0-to-1:"
                f"{tensor_name.replace('_', '-')}:tp-all-gather")
            if calls_by_id[boundary_id]["kernel_spec"]["transfer"][
                    "bytes"] != expected_wire_bytes:
                raise ValueError(
                    f"{context} boundary all-gather bytes are invalid")

        intervals_by_id = {
            interval["call_id"]: interval
            for interval in step["execution_timeline"]["intervals"]
        }
        worker_pool_intervals.extend(
            (
                interval["absolute_start_time_s"],
                interval["absolute_end_time_s"],
            )
            for interval in intervals_by_id.values()
        )
        for call in communication_calls:
            transfer = call["kernel_spec"]["transfer"]
            operation = transfer["operation"]
            resource_id = transfer["resource_id"]
            if resource_id not in resource_stats:
                raise ValueError(f"{context} communication fabric is invalid")
            if (transfer["backend"] != "analytical_roofline_v1"
                    or transfer["backend_config_sha256"] is not None
                    or transfer["link"] != "nvlink"):
                raise ValueError(
                    f"{context} communication backend contract is invalid")
            if resource_id == tensor_resource_id:
                if operation == "all_reduce":
                    expected_reduction = "sum"
                elif operation == "all_gather":
                    expected_reduction = None
                else:
                    raise ValueError(
                        f"{context} tensor collective operation is invalid")
                if (operation not in ("all_reduce", "all_gather")
                        or transfer["algorithm"] != "ring"
                        or transfer["direction"] != "collective"
                        or transfer["topology"] != "ring"
                        or transfer["participants"] not in ([0, 1], [2, 3])
                        or transfer["ring_order"] != transfer["participants"]
                        or transfer["bytes_semantics"] != "per_rank_input"
                        or transfer["source_rank"] is not None
                        or transfer["destination_rank"] is not None
                        or transfer["reduction"] != expected_reduction):
                    raise ValueError(
                        f"{context} tensor collective contract is invalid")
            elif resource_id == pipeline_resource_id:
                if (operation != "send_recv"
                        or transfer["algorithm"] != "send_recv"
                        or transfer["direction"] != "point_to_point"
                        or transfer["topology"] != "point_to_point"
                        or tuple(transfer["participants"])
                        not in expected_lane_pairs
                        or transfer["ring_order"] != []
                        or transfer["bytes_semantics"] != "payload"
                        or transfer["reduction"] is not None):
                    raise ValueError(
                        f"{context} pipeline P2P contract is invalid")
            else:
                raise AssertionError("validated communication fabric vanished")
            stats = resource_stats[resource_id]
            topology = transfer["topology"]
            semantics = transfer["bytes_semantics"]
            stats["call_count"] += 1
            stats["operation_counts"][operation] += 1
            stats["topology_counts"][topology] += 1
            stats["bytes_by_semantics"][semantics] += transfer["bytes"]
            interval = intervals_by_id[call["call_id"]]
            stats["intervals"].append((
                interval["absolute_start_time_s"],
                interval["absolute_end_time_s"],
            ))
        total_call_count += len(calls)

    if (set(entries_by_id) != all_step_ids
            or counts["kernel_calls"] != total_call_count
            or result["num_kernel_calls"] != total_call_count):
        raise ValueError(f"{context} kernel or schedule counts disagree")
    canonical_schedule_trace = ScheduleTrace(
        ScheduleTraceEntry(
            backend_steps_by_id[entry["step_id"]],
            entry["start_time_s"],
            entry["end_time_s"],
        )
        for entry in trace_entries
    )
    _validate_production_trace_scheduler_binding(
        expected_requests,
        canonical_schedule_trace,
        scheduler["kv_block_size_tokens"],
        result,
        max_requests=scheduler["max_requests"],
        context=context,
    )
    _replay_pinned_parallel_scheduler(
        expected_requests,
        canonical_schedule_trace,
        scheduler,
        expected_model,
        expected_model_path,
        runtime,
        num_kv_blocks=runtime["num_kv_blocks"],
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        enable_expert_parallel=False,
        expected_mtp=expected_simulation["mtp"],
        context=context,
    )
    canonical_metrics = compute_serving_metrics(
        expected_requests,
        canonical_schedule_trace,
    )
    canonical_result = SimulationResult(
        canonical_schedule_trace,
        canonical_metrics,
        tuple(simulation_steps),
    )
    expected_prefix_alignment = _json_round_trip(prefix_alignment_manifest(
        expected_requests,
        canonical_result,
        scheduler["kv_block_size_tokens"],
    ))
    if result["request_source"]["prefix_alignment"] != (
            expected_prefix_alignment):
        raise ValueError(
            f"{context} Prefix alignment disagrees with the command Trace "
            "and canonical Scheduler")
    expected_artifact = _json_round_trip(
        execution_artifact_manifest(
            canonical_result,
            performance_timing_status=(
                _expected_performance_timing_status(expected_simulation)),
        ))
    if artifact != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")

    system_resources = result["metrics"]["system_resources"]
    observation_window = system_resources["observation_window"]
    observation_window_s = observation_window["duration_s"]
    communication_resources = system_resources[
        "communication_resource_utilization"]
    if set(communication_resources) != set(resource_stats):
        raise ValueError(
            f"{context} communication resource coverage is invalid")
    for resource_id, stats in resource_stats.items():
        resource = communication_resources[resource_id]
        busy_time_s = _interval_union_duration_from_output(
            stats["intervals"],
            observation_window_s,
            context=f"{context} {resource_id}",
        )
        idle_time_s = observation_window_s - busy_time_s
        if (resource["backend"] != "analytical_roofline_v1"
                or resource["backend_config_sha256"] is not None
                or resource["hardware_id"] != hardware_id
                or resource["link"] != "nvlink"
                or resource["timing_status"]
                != "projected_analytical_roofline"
                or resource["call_count"] != stats["call_count"]
                or resource["operation_counts"]
                != stats["operation_counts"]
                or resource["topology_counts"]
                != stats["topology_counts"]
                or resource["bytes_by_semantics"]
                != stats["bytes_by_semantics"]
                or not math.isclose(
                    resource["busy_time_s"],
                    busy_time_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    resource["idle_time_s"],
                    idle_time_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    resource["utilization"],
                    busy_time_s / observation_window_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )):
            raise ValueError(
                f"{context} communication metrics disagree for {resource_id}")
    worker_pool_busy_time_s = _interval_union_duration_from_output(
        worker_pool_intervals,
        observation_window_s,
        context=f"{context} worker pool",
    )
    worker_pool_idle_time_s = observation_window_s - worker_pool_busy_time_s
    worker_pool_resources = system_resources["worker_pool_utilization"]
    if set(worker_pool_resources) != {"main"}:
        raise ValueError(
            f"{context} worker-pool utilization coverage is invalid")
    worker_pool = worker_pool_resources["main"]
    if (observation_window["basis"]
            != "simulation_origin_to_last_completion"
            or not math.isclose(
                observation_window_s,
                result["schedule_end_time_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or system_resources["link_utilization"] != {}
            or worker_pool["kernel_interval_count"] != total_call_count
            or not math.isclose(
                worker_pool["busy_time_s"],
                worker_pool_busy_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                worker_pool["idle_time_s"],
                worker_pool_idle_time_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                worker_pool["utilization"],
                worker_pool_busy_time_s / observation_window_s,
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or set(system_resources["scheduler_queue"]["by_pool"])
            != {"main"}):
        raise ValueError(f"{context} system resource metrics disagree")
    expected_metrics = _canonical_production_trace_metrics_manifest(
        deployment,
        expected_requests,
        canonical_result,
        expected_slo_contract,
    )
    if result["metrics"] != expected_metrics:
        raise ValueError(
            f"{context} metrics disagree with canonical Scheduler and "
            "execution trace")


def _validate_canonical_production_trace_step_duration(
    step: dict[str, Any],
    canonical_duration_s: float,
    *,
    context: str,
) -> None:
    interval = step["schedule_interval"]
    recorded_duration_s = (
        interval["end_time_s"] - interval["start_time_s"])
    if not math.isclose(
            recorded_duration_s,
            canonical_duration_s,
            rel_tol=1e-12,
            abs_tol=1e-15,
            ):
        raise ValueError(
            f"{context} Scheduler step duration disagrees with the "
            "canonical execution timeline")


def _canonical_production_trace_metrics_manifest(
    deployment: DeploymentPlan,
    requests: RequestStream,
    result: SimulationResult,
    slo_contract: Any,
) -> dict[str, Any]:
    metrics = result.metrics
    return _json_round_trip({
        "request_throughput_per_s": metrics.request_throughput_per_s,
        "output_token_throughput_per_s": (
            metrics.output_token_throughput_per_s),
        "token_throughput_per_s": metrics.token_throughput_per_s,
        "latency_distributions": latency_distribution_manifest(
            metrics.requests),
        "slo_goodput": slo_goodput_manifest(
            requests,
            metrics,
            result.schedule_trace.end_time_s,
            slo_contract,
        ),
        "system_resources": system_resource_manifest(
            deployment,
            requests,
            result,
        ),
        "critical_path": critical_path_manifest(
            deployment,
            requests,
            result,
        ),
        "requests": {
            request_id: {
                "ttft_s": metric.ttft_s,
                "tpot_s": metric.tpot_s,
                "latency_s": metric.latency_s,
            }
            for request_id, metric in metrics.requests.items()
        },
    })


def _backend_step_from_output_manifest(
    raw: dict[str, Any],
) -> BackendStep:
    if not isinstance(raw, dict):
        raise TypeError("serialized BackendStep must be an object")
    if set(raw) != {"schedule", "kv_cache_events", "generated_tokens"}:
        raise ValueError(
            "serialized BackendStep fields disagree with the canonical "
            "contract")
    raw_events = raw["kv_cache_events"]
    if not isinstance(raw_events, list):
        raise TypeError(
            "serialized BackendStep kv_cache_events must be a list")
    generated_tokens = raw["generated_tokens"]
    if not isinstance(generated_tokens, dict):
        raise TypeError(
            "serialized BackendStep generated_tokens must be an object")
    step = BackendStep(
        schedule=_schedule_step_from_output_manifest(raw["schedule"]),
        kv_cache_events=tuple(
            _kv_cache_event_from_output_manifest(event)
            for event in raw_events
        ),
        generated_tokens=generated_tokens,
    )
    expected = {
        "schedule": _json_round_trip(asdict(step.schedule)),
        "kv_cache_events": [
            _json_round_trip(asdict(event))
            for event in step.kv_cache_events
        ],
        "generated_tokens": {
            request_id: step.generated_tokens[request_id]
            for request_id in sorted(step.generated_tokens)
        },
    }
    if raw != expected:
        raise ValueError("serialized BackendStep derived fields disagree")
    return step


def _validate_production_trace_scheduler_binding(
    expected_requests: RequestStream,
    schedule_trace: ScheduleTrace,
    kv_block_size_tokens: int,
    result: dict[str, Any],
    *,
    max_requests: int,
    context: str,
) -> None:
    if (isinstance(max_requests, bool)
            or not isinstance(max_requests, int)
            or max_requests <= 0):
        raise ValueError(
            f"{context} max_requests must be a positive integer")
    requests_by_id = {
        request.request_id: request for request in expected_requests
    }
    first_schedule_ids = set()
    unique_prompt_tokens = {
        request_id: 0 for request_id in requests_by_id
    }
    for entry in schedule_trace.entries:
        schedule = entry.backend_step.schedule
        if len(schedule.requests) > max_requests:
            raise ValueError(
                f"{context} exceeded the configured Scheduler "
                "max_requests")
        if set(schedule.preempted_request_ids) - set(requests_by_id):
            raise ValueError(
                f"{context} preempted an unknown request")
        for scheduled in schedule.requests:
            request_id = scheduled.request_id
            if request_id not in requests_by_id:
                raise ValueError(
                    f"{context} scheduled an unknown request")
            request = requests_by_id[request_id]
            if entry.start_time_s < request.arrival_time_s:
                raise ValueError(
                    f"{context} scheduled a request before its "
                    "Trace arrival")
            if request_id not in first_schedule_ids:
                expected_scheduler_hit = (
                    scheduler_effective_prefix_hit_tokens(
                        request, kv_block_size_tokens))
                if (scheduled.trace_prefix_hit_tokens
                        != request.prefix_hit_tokens
                        or scheduled.scheduler_prefix_hit_tokens
                        != expected_scheduler_hit
                        or scheduled.computed_tokens
                        != expected_scheduler_hit):
                    raise ValueError(
                        f"{context} first Scheduler step "
                        "disagrees with the command Trace request")
                unique_prompt_tokens[request_id] += expected_scheduler_hit
                first_schedule_ids.add(request_id)
            elif (scheduled.trace_prefix_hit_tokens != 0
                  or scheduled.scheduler_prefix_hit_tokens != 0):
                raise ValueError(
                    f"{context} repeated first-step prefix "
                    "metadata")
            recomputed_prompt_tokens = min(
                scheduled.input_tokens,
                scheduled.recomputed_tokens,
            )
            unique_prompt_tokens[request_id] += (
                scheduled.input_tokens - recomputed_prompt_tokens)
    if first_schedule_ids != set(requests_by_id):
        raise ValueError(
            f"{context} Scheduler did not cover every command "
            "Trace request")
    for request_id, request in requests_by_id.items():
        if unique_prompt_tokens[request_id] != request.input_tokens:
            raise ValueError(
                f"{context} Scheduler prompt work disagrees "
                "with the command Trace request")
    try:
        metrics = compute_serving_metrics(
            expected_requests,
            schedule_trace,
        )
    except ValueError as error:
        raise ValueError(
            f"{context} Scheduler output-token ledger "
            "disagrees with the command Trace requests") from error
    actual_metrics = result["metrics"]
    if (not math.isclose(
            schedule_trace.end_time_s,
            result["schedule_end_time_s"],
            rel_tol=1e-12,
            abs_tol=1e-15,
            )
            or not math.isclose(
                metrics.request_throughput_per_s,
                actual_metrics["request_throughput_per_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                metrics.output_token_throughput_per_s,
                actual_metrics["output_token_throughput_per_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )
            or not math.isclose(
                metrics.token_throughput_per_s,
                actual_metrics["token_throughput_per_s"],
                rel_tol=1e-12,
                abs_tol=1e-15,
            )):
        raise ValueError(
            f"{context} metrics disagree with the canonical "
            "Scheduler trace")
    expected_request_metrics = {
        request_id: {
            "ttft_s": metric.ttft_s,
            "tpot_s": metric.tpot_s,
            "latency_s": metric.latency_s,
        }
        for request_id, metric in metrics.requests.items()
    }
    if actual_metrics["requests"] != expected_request_metrics:
        raise ValueError(
            f"{context} request metrics disagree with the "
            "canonical Scheduler trace")


def _replay_pinned_parallel_scheduler(
    expected_requests: RequestStream,
    expected_trace: ScheduleTrace,
    scheduler: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    expected_runtime: dict[str, Any],
    *,
    num_kv_blocks: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    enable_expert_parallel: bool,
    expected_mtp: dict[str, Any],
    context: str,
) -> None:
    backend = None
    replay_error = None
    replay_trace = None
    try:
        mtp_assumption = _maximum_entropy_mtp_assumption(expected_mtp)
        if (mtp_assumption is not None
                and mtp_assumption.request_stream_sha256
                != request_stream_sha256(expected_requests)):
            raise ValueError(
                f"{context} maximum-entropy MTP assumption disagrees with "
                "the replayed request stream")
        backend = VLLMBackend.from_model(
            BackendConfig(
                pool_name="main",
                role="colocated",
                max_batched_tokens=scheduler["max_batched_tokens"],
                max_requests=scheduler["max_requests"],
                num_kv_blocks=num_kv_blocks,
                kv_block_size_tokens=scheduler[
                    "kv_block_size_tokens"],
                kv_bytes_per_token=(
                    expected_model.
                    kv_bytes_per_token_per_parallel_rank(
                        tensor_parallel_size,
                        pipeline_parallel_size,
                    )),
            scheduler_pipeline_depth=scheduler["scheduler_pipeline_depth"]),
            VLLMSchedulerConfig(
                model=expected_model_path,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                enable_expert_parallel=enable_expert_parallel,
                dtype="bfloat16",
                max_model_len=scheduler["max_model_len"],
                enable_chunked_prefill=True,
                enable_prefix_caching=False,
                mtp_num_speculative_tokens=expected_mtp[
                    "num_speculative_tokens"],
                mtp_acceptance_policy=expected_mtp[
                    "acceptance_policy"],
                mtp_fixed_accepted_tokens=expected_mtp[
                    "fixed_accepted_tokens"],
                mtp_acceptance_source_id=expected_mtp[
                    "acceptance_source_id"],
                mtp_acceptance_assumption=mtp_assumption,
                request_stream_sha256=(
                    None
                    if mtp_assumption is None
                    else mtp_assumption.request_stream_sha256
                ),
            ),
        )
        if _json_round_trip(asdict(backend.runtime_info)) != expected_runtime:
            raise ValueError(
                f"{context} pinned Scheduler replay runtime "
                "disagrees with the recorded runtime")
        expected_entries = expected_trace.entries
        replay_step_index = 0

        def validated_step_duration(actual_step: BackendStep) -> float:
            nonlocal replay_step_index
            if replay_step_index >= len(expected_entries):
                raise ValueError(
                    f"{context} pinned Scheduler replay emitted "
                    "an extra step")
            expected_entry = expected_entries[replay_step_index]
            if actual_step != expected_entry.backend_step:
                raise ValueError(
                    f"{context} BackendStep disagrees with an "
                    "independent pinned Scheduler replay")
            replay_step_index += 1
            return expected_entry.end_time_s - expected_entry.start_time_s

        replay_trace = run_single_pool_schedule(
            backend,
            expected_requests,
            validated_step_duration,
        )
        if replay_step_index != len(expected_entries):
            raise ValueError(
                f"{context} pinned Scheduler replay emitted "
                "fewer steps than the execution artifact")
    except (TypeError, ValueError, RuntimeError, TimeoutError) as error:
        replay_error = error
    finally:
        if backend is not None:
            try:
                backend.close()
            except (TypeError, ValueError, RuntimeError, TimeoutError) as error:
                if replay_error is None:
                    replay_error = error
    if replay_error is not None:
        raise ValueError(
            f"{context} cannot be reproduced by the pinned "
            "Scheduler") from replay_error
    if replay_trace is None:
        raise AssertionError("successful pinned Scheduler replay vanished")
    if len(replay_trace.entries) != len(expected_trace.entries):
        raise AssertionError("validated pinned Scheduler step count changed")
    for actual_entry, expected_entry in zip(
            replay_trace.entries, expected_trace.entries, strict=True):
        if (actual_entry.backend_step != expected_entry.backend_step
                or not math.isclose(
                    actual_entry.start_time_s,
                    expected_entry.start_time_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )
                or not math.isclose(
                    actual_entry.end_time_s,
                    expected_entry.end_time_s,
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                )):
            raise ValueError(
                f"{context} pinned Scheduler replay interval "
                "disagrees with the execution artifact")


def _interval_union_duration_from_output(
    intervals: list[tuple[float, float]],
    observation_window_s: float,
    *,
    context: str,
) -> float:
    if not intervals:
        raise ValueError(f"{context} contains no execution intervals")
    ordered = sorted(intervals)
    union_start = None
    union_end = None
    total_s = 0.0
    for start_time_s, end_time_s in ordered:
        if (isinstance(start_time_s, bool)
                or isinstance(end_time_s, bool)
                or not isinstance(start_time_s, (int, float))
                or not isinstance(end_time_s, (int, float))
                or not math.isfinite(start_time_s)
                or not math.isfinite(end_time_s)
                or start_time_s < 0.0
                or end_time_s < start_time_s
                or end_time_s > observation_window_s):
            raise ValueError(f"{context} contains an invalid interval")
        if union_start is None:
            union_start = start_time_s
            union_end = end_time_s
        elif start_time_s <= union_end:
            union_end = max(union_end, end_time_s)
        else:
            total_s += union_end - union_start
            union_start = start_time_s
            union_end = end_time_s
    if union_start is None or union_end is None:
        raise AssertionError("validated non-empty interval set vanished")
    return total_s + union_end - union_start


def _schedule_step_from_output_manifest(
    raw: dict[str, Any],
) -> ScheduleStep:
    if not isinstance(raw, dict):
        raise TypeError("serialized ScheduleStep must be an object")
    schedule_fields = fields(ScheduleStep)
    expected_fields = {field.name for field in schedule_fields}
    if set(raw) != expected_fields:
        raise ValueError(
            "serialized ScheduleStep fields disagree with the canonical "
            "contract")
    raw_requests = raw["requests"]
    if not isinstance(raw_requests, list) or not raw_requests:
        raise TypeError(
            "serialized ScheduleStep requests must be a non-empty list")
    request_fields = fields(ScheduledRequest)
    expected_request_fields = {field.name for field in request_fields}
    requests = []
    for raw_request in raw_requests:
        if (not isinstance(raw_request, dict)
                or set(raw_request) != expected_request_fields):
            raise ValueError(
                "serialized ScheduledRequest fields disagree with the "
                "canonical contract")
        request = ScheduledRequest(**{
            field.name: raw_request[field.name]
            for field in request_fields
            if field.init
        })
        if _json_round_trip(asdict(request)) != raw_request:
            raise ValueError(
                "serialized ScheduledRequest derived fields disagree")
        requests.append(request)
    schedule_kwargs = {}
    for field in schedule_fields:
        if not field.init:
            continue
        if field.name == "requests":
            schedule_kwargs[field.name] = tuple(requests)
        elif field.name in expected_fields - {"requests"}:
            schedule_kwargs[field.name] = raw[field.name]
        else:
            raise AssertionError(
                f"unknown ScheduleStep constructor field: {field.name!r}")
    schedule = ScheduleStep(**schedule_kwargs)
    if _json_round_trip(asdict(schedule)) != raw:
        raise ValueError("serialized ScheduleStep derived fields disagree")
    return schedule


def _kv_cache_event_from_output_manifest(
    raw: dict[str, Any],
) -> KVCacheEvent:
    if not isinstance(raw, dict):
        raise TypeError("serialized KVCacheEvent must be an object")
    event_fields = fields(KVCacheEvent)
    expected_fields = {field.name for field in event_fields}
    if set(raw) != expected_fields:
        raise ValueError(
            "serialized KVCacheEvent fields disagree with the canonical "
            "contract")
    event = KVCacheEvent(**{
        field.name: raw[field.name]
        for field in event_fields
        if field.init
    })
    if _json_round_trip(asdict(event)) != raw:
        raise ValueError("serialized KVCacheEvent derived fields disagree")
    return event


def _route_decision_from_output_manifest(
    raw: dict[str, Any],
) -> RouteDecision:
    if not isinstance(raw, dict):
        raise TypeError("serialized RouteDecision must be an object")
    decision_fields = fields(RouteDecision)
    expected_fields = {field.name for field in decision_fields}
    if set(raw) != expected_fields:
        raise ValueError(
            "serialized RouteDecision fields disagree with the canonical "
            "contract")
    decision = RouteDecision(**{
        field.name: raw[field.name]
        for field in decision_fields
        if field.init
    })
    if _json_round_trip(asdict(decision)) != raw:
        raise ValueError("serialized RouteDecision derived fields disagree")
    return decision


def _json_round_trip(value: Any) -> Any:
    return json.loads(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ))


def _validate_production_trace_capacity(
    result: dict[str, Any],
    scheduler: dict[str, Any],
    expected_pools: tuple[str, ...],
    *,
    context: str,
) -> None:
    capacity = result["kv_capacity"]
    if capacity["schema_version"] != 2:
        raise ValueError(
            f"{context} KV capacity source disagrees with command manifest")
    source = capacity["source"]
    if scheduler["capacity_mode"] == "analytical":
        expected_source_mode = "analytical"
        expected_policy_path = str(scheduler["runtime_memory_policy"])
        if (source["mode"] != expected_source_mode
                or source["runtime_memory_policy_path"] != expected_policy_path
                or not isinstance(
                    source["runtime_memory_policy_sha256"], str)):
            raise ValueError(
                f"{context} analytical KV capacity source disagrees with "
                "command manifest")
    elif scheduler["capacity_mode"] == "debug_explicit_block_count":
        expected_source_mode = "debug_explicit_block_count"
        if source != {
                "mode": expected_source_mode,
                "runtime_memory_policy_path": None,
                "runtime_memory_policy_sha256": None,
        }:
            raise ValueError(
                f"{context} debug KV capacity source disagrees with command "
                "manifest")
    else:
        raise ValueError(f"{context} has an unknown KV capacity mode")
    capacity_plans = capacity["scheduler_pools"]
    plans_by_pool = {
        plan["pool_name"]: plan for plan in capacity_plans
    }
    if (len(plans_by_pool) != len(capacity_plans)
            or set(plans_by_pool) != set(expected_pools)):
        raise ValueError(
            f"{context} KV capacity pools disagree with command manifest")
    for pool_name, plan in plans_by_pool.items():
        runtime = result["scheduler_runtime"][pool_name]
        owner_count = len(runtime["scheduler_kv_cache_tensor_owners"])
        page_size_bytes = runtime["scheduler_kv_cache_page_size_bytes"]
        bytes_numerator = owner_count * page_size_bytes
        if (owner_count <= 0
                or isinstance(page_size_bytes, bool)
                or not isinstance(page_size_bytes, int)
                or page_size_bytes <= 0
                or bytes_numerator % scheduler["kv_block_size_tokens"] != 0):
            raise ValueError(f"{context} Scheduler KV geometry is invalid")
        expected_kv_bytes_per_token = (
            bytes_numerator // scheduler["kv_block_size_tokens"])
        expected_block_bytes = (
            scheduler["kv_block_size_tokens"]
            * expected_kv_bytes_per_token)
        if (plan["source_mode"] != expected_source_mode
                or plan["num_kv_blocks"] != runtime["num_kv_blocks"]
                or plan["kv_block_size_tokens"]
                != scheduler["kv_block_size_tokens"]
                or plan["kv_bytes_per_token"]
                != expected_kv_bytes_per_token
                or plan["kv_block_bytes"] != expected_block_bytes):
            raise ValueError(
                f"{context} KV capacity geometry disagrees with command "
                "manifest")
        if expected_source_mode == "analytical":
            if (plan["runtime_memory_policy"] is None
                    or not plan["rank_budgets"]
                    or plan["native_vllm_allocation"] is None):
                raise ValueError(
                    f"{context} analytical KV capacity evidence is incomplete")
            utilization = plan["runtime_memory_policy"][
                "gpu_memory_utilization"]
            expected_utilization = (
                utilization["numerator"] / utilization["denominator"])
        elif (plan["runtime_memory_policy"] is not None
              or plan["rank_budgets"]
              or plan["native_vllm_allocation"] is not None
              or plan["num_kv_blocks"] != scheduler["num_kv_blocks"]):
            raise ValueError(
                f"{context} debug KV capacity carries analytical evidence")
        else:
            expected_utilization = 0.9
        if (runtime["requested_gpu_memory_utilization"]
                != expected_utilization
                or runtime["effective_gpu_memory_utilization"]
                != expected_utilization):
            raise ValueError(
                f"{context} Scheduler memory utilization disagrees with KV "
                "capacity policy")


def _artifact_af_payload_names_by_transfer_id(
    result: dict[str, Any],
    *,
    context: str,
) -> dict[str, str]:
    """Validate ordered AF tensor pairs already carried by artifact v5."""

    payload_names = {}
    for step in result["execution_artifact"]["steps"]:
        trace_ids_by_template = {}
        workload_graph = step["workload_graph"]
        if "block_repeats" in workload_graph:
            for repeat in workload_graph["block_repeats"]:
                for trace_node in repeat["trace_node_ids"]:
                    trace_ids_by_template[
                        trace_node["template_node_id"]] = tuple(
                            trace_node["logical_node_ids"])
        nodes = tuple(
            node for node in workload_graph["nodes"]
            if node["kind"] == WorkloadKind.AF_ACTIVATION_TRANSFER.value
        )
        if len(nodes) % 2 != 0:
            raise ValueError(
                f"{context} AF artifact must contain complete tensor pairs")
        for pair_index in range(0, len(nodes), 2):
            pair = nodes[pair_index:pair_index + 2]
            tensors = []
            logical_ids_by_node = []
            for node in pair:
                node_id = node["node_id"]
                attributes = node["attributes"]
                if "payload_tensor" not in attributes:
                    raise ValueError(
                        f"{context} AF artifact lacks payload_tensor")
                payload = attributes["payload_tensor"]
                if set(payload) != {"name", "shape", "dtype", "bytes"}:
                    raise ValueError(
                        f"{context} AF artifact payload_tensor fields are "
                        "invalid")
                tensors.append(payload)
                logical_ids_by_node.append(
                    trace_ids_by_template[node_id]
                    if node_id in trace_ids_by_template else (node_id,))
            if tuple(tensor["name"] for tensor in tensors) != (
                    "hidden_states", "residual"):
                raise ValueError(
                    f"{context} AF artifact tensor pair must be ordered "
                    "hidden_states then residual")
            if any(
                    tensors[0][field] != tensors[1][field]
                    for field in ("shape", "dtype", "bytes")):
                raise ValueError(
                    f"{context} AF artifact tensor pair geometry disagrees")
            if len(logical_ids_by_node[0]) != len(logical_ids_by_node[1]):
                raise ValueError(
                    f"{context} AF artifact tensor pair repeat counts disagree")
            for logical_ids, tensor in zip(
                    logical_ids_by_node, tensors, strict=True):
                for logical_id in logical_ids:
                    if logical_id in payload_names:
                        raise ValueError(
                            f"{context} AF artifact repeats transfer node_id")
                    payload_names[logical_id] = tensor["name"]
    activation_transfer_ids = {
        transfer["transfer_id"]
        for transfer in result["transfers"]
        if transfer["payload"] == "activation"
    }
    if set(payload_names) != activation_transfer_ids:
        raise ValueError(
            f"{context} AF artifact nodes and activation Trace disagree")
    return payload_names


def _validate_af_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    replay_identity = expected_simulation["kv_lifecycle_replay"]
    if replay_identity is None:
        kv_lifecycle_replay = None
        context = "AF Production Trace"
    else:
        context = "AF+Host-KV Production Trace"
        try:
            kv_lifecycle_replay = load_kv_lifecycle_replay(
                replay_identity["path"])
        except (OSError, TypeError, ValueError, NotImplementedError) as error:
            raise ValueError(
                f"{context} replay cannot be independently reloaded") \
                from error
        if (kv_lifecycle_replay.source_sha256
                != replay_identity["source_sha256"]
                or kv_lifecycle_replay.artifact_id
                != replay_identity["artifact_id"]):
            raise ValueError(f"{context} replay identity changed")
        expected_replay_manifest = _json_round_trip(
            kv_lifecycle_replay.manifest())
        if (result["kv_lifecycle_replay"] != expected_replay_manifest
                or kv_lifecycle_replay.context.request_stream_sha256
                != request_stream_sha256(expected_requests)):
            raise ValueError(
                f"{context} replay disagrees with the command manifest or "
                "Trace")
    af_disagg = expected_simulation["af_disagg"]
    if (af_disagg is None
            or expected_simulation["pd_disagg"] is not None):
        raise ValueError(f"{context} command deployment config is invalid")
    if (expected_simulation["expert_routing"] is not None
            or expected_simulation["mtp"]["acceptance_policy"]
            != "disabled"):
        raise ValueError(f"{context} command enabled an unsupported feature")
    scheduler_runtime = result["scheduler_runtime"]
    if set(scheduler_runtime) != {"attention"}:
        raise ValueError(
            f"{context} must contain only the Attention Scheduler runtime")
    scheduler = expected_simulation["scheduler"]
    _validate_production_trace_scheduler_runtime(
        scheduler_runtime["attention"],
        scheduler,
        expected_model,
        expected_simulation["mtp"],
        role="attention",
        tensor_parallel_field="prefill_tp_size",
    )
    _validate_production_trace_capacity(
        result,
        scheduler,
        ("attention",),
        context=context,
    )
    if (result["routing_policy"] != "least_queued_tokens"
            or result["kv_handoff_mode"] != "not_applicable"
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["kv_lifecycle_replay"] != (
                None
                if kv_lifecycle_replay is None
                else _json_round_trip(kv_lifecycle_replay.manifest())
            )
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(f"{context} routing or lifecycle state is invalid")
    if (kv_lifecycle_replay is None
            and result["execution_artifact"]["modeled_kv_offload"] != {
                "schema_version": 2,
                "availability": "not_enabled",
                "intent_source": None,
                "transfers": [],
                "final_tier_state": None,
            }):
        raise ValueError(f"{context} unexpectedly enabled KV offload")

    transfer_config = af_disagg["activation_transfer"]
    activation_bytes = expected_model.hidden_size * expected_model.dtype_bytes
    request_id = expected_requests.requests[0].request_id
    route_decisions = result["route_decisions"]
    if len(route_decisions) != 1:
        raise ValueError(
            f"{context} must contain one ingress decision for Attention")
    decision = route_decisions[0]
    if (decision["deployment_id"] != result["deployment_id"]
            or decision["request_id"] != request_id
            or decision["stage"] != "ingress_to_colocated"
            or decision["policy"] != "least_queued_tokens"
            or decision["source_engine_id"] is not None
            or decision["selected_engine_id"] != "attention"
            or decision["candidate_queued_tokens"] != [["attention", 0]]):
        raise ValueError(
            f"{context} ingress route must select the sole Attention "
            "Scheduler from zero visible queued tokens")
    tensor_parallel_size = scheduler["prefill_tp_size"]
    if (tensor_parallel_size not in (1, 2)
            or af_disagg["ffn_tp_size"] != tensor_parallel_size):
        raise ValueError(f"{context} Attention and FFN TP sizes are invalid")
    attention_ranks = tuple(range(tensor_parallel_size))
    ffn_ranks = tuple(range(
        tensor_parallel_size, 2 * tensor_parallel_size))
    steps = result["execution_artifact"]["steps"]
    if tensor_parallel_size == 1:
        request = expected_requests.requests[0]
        expected_scheduled_tokens = (
            request.input_tokens - request.prefix_hit_tokens,
            1,
        )
    elif tensor_parallel_size == 2:
        expected_scheduled_tokens = (2,)
    else:
        raise AssertionError("validated AF TP size vanished")
    actual_scheduled_tokens = tuple(
        sum(
            request["input_tokens"] + request["output_tokens"]
            for request in step["backend_step"]["schedule"]["requests"]
        )
        for step in steps
    )
    if (tuple(step["pool_name"] for step in steps)
            != tuple("attention" for _ in expected_scheduled_tokens)
            or actual_scheduled_tokens != expected_scheduled_tokens):
        raise ValueError(
            f"{context} Scheduler steps disagree with the validated Trace")

    payload_names_by_transfer_id = _artifact_af_payload_names_by_transfer_id(
        result, context=context)
    expected_transfer_identities = []
    for token_count in expected_scheduled_tokens:
        byte_count = token_count * activation_bytes
        for _layer_index in range(expected_model.num_hidden_layers):
            for attention_rank, ffn_rank in zip(
                    attention_ranks, ffn_ranks, strict=True):
                for tensor_name in ("hidden_states", "residual"):
                    expected_transfer_identities.append((
                        (request_id,), "attention", "ffn", byte_count,
                        attention_rank, ffn_rank, tensor_name,
                    ))
            for attention_rank, ffn_rank in zip(
                    attention_ranks, ffn_ranks, strict=True):
                for tensor_name in ("hidden_states", "residual"):
                    expected_transfer_identities.append((
                        (request_id,), "ffn", "attention", byte_count,
                        ffn_rank, attention_rank, tensor_name,
                    ))
    transfers = result["transfers"]
    if (result["num_steps"] != len(expected_scheduled_tokens)
            or result["num_transfers"] != len(expected_transfer_identities)
            or len(transfers) != len(expected_transfer_identities)):
        raise ValueError(
            f"{context} Scheduler step or activation transfer count is "
            "invalid")
    expected_transfer_records = []
    for expected in expected_transfer_identities:
        (request_ids, source_pool, destination_pool, byte_count, source_rank,
         destination_rank, tensor_name) = expected
        expected_reserved_resources = (
            "communication:endpoint:nvidia_a800:"
            f"{source_rank}:{transfer_config['link']}",
            "communication:endpoint:nvidia_a800:"
            f"{destination_rank}:{transfer_config['link']}",
            "communication:fabric:"
            f"{transfer_config['resource_id']}",
        )
        expected_transfer_records.append((
            request_ids,
            source_pool,
            destination_pool,
            byte_count,
            expected_reserved_resources,
            tensor_name,
        ))
    actual_transfer_records = [
        (
            tuple(transfer["request_ids"]),
            transfer["source_pool_id"],
            transfer["destination_pool_id"],
            transfer["byte_count"],
            tuple(transfer["reserved_resource_ids"]),
            payload_names_by_transfer_id[transfer["transfer_id"]],
        )
        for transfer in transfers
    ]
    if sorted(actual_transfer_records) != sorted(expected_transfer_records):
        raise ValueError(
            f"{context} activation transfer disagrees with the canonical "
            "Attention/FFN lane, bytes, or resource reservation")
    for transfer in transfers:
        if (transfer["payload"] != transfer_config["payload"]
                or transfer["resource_id"] != transfer_config["resource_id"]
                or transfer["link"] != transfer_config["link"]
                or transfer["end_time_s"] <= transfer["start_time_s"]):
            raise ValueError(
                f"{context} activation transfer disagrees with the canonical "
                "Attention/FFN route, bytes, or resource reservation")

    communication_calls = tuple(
        call
        for step in result["execution_artifact"]["steps"]
        for call in step["kernel_plan"]["calls"]
        if call["kernel_spec"]["kind"] == KernelKind.COMMUNICATION.value
    )
    lane_calls = tuple(
        call for call in communication_calls
        if call["kernel_spec"]["transfer"]["operation"] == "send_recv"
    )
    collective_calls = tuple(
        call for call in communication_calls
        if call["kernel_spec"]["transfer"]["operation"]
        in ("all_reduce", "all_gather")
    )
    expected_collective_count = (
        0 if tensor_parallel_size == 1
        else 6 * len(expected_scheduled_tokens)
    )
    if (len(lane_calls) != len(transfers)
            or len(collective_calls) != expected_collective_count
            or len(communication_calls)
            != len(transfers) + expected_collective_count):
        raise ValueError(
            f"{context} COMMUNICATION KernelSpec count is invalid")
    lane_calls_by_id = {}
    for call in lane_calls:
        call_id = call["call_id"]
        if call_id in lane_calls_by_id:
            raise ValueError(f"{context} repeated lane transfer call_id")
        lane_calls_by_id[call_id] = call
    if set(lane_calls_by_id) != {
            transfer["transfer_id"] for transfer in transfers}:
        raise ValueError(
            f"{context} lane COMMUNICATION calls disagree with transfers")
    ranks_by_reserved_resources = {
        record[4]: (expected[4], expected[5])
        for record, expected in zip(
            expected_transfer_records,
            expected_transfer_identities,
            strict=True,
        )
    }
    for transfer in transfers:
        call = lane_calls_by_id[transfer["transfer_id"]]
        source_pool = transfer["source_pool_id"]
        byte_count = transfer["byte_count"]
        source_rank, destination_rank = ranks_by_reserved_resources[
            tuple(transfer["reserved_resource_ids"])]
        physical = call["kernel_spec"]["transfer"]
        if (call["pool_name"] != source_pool
                or call["rank"] != source_rank
                or call["resource"] != transfer_config["resource_id"]
                or physical["algorithm"] != transfer_config["algorithm"]
                or physical["bytes"] != byte_count
                or physical["bytes_semantics"] != "payload"
                or physical["operation"] != transfer_config["algorithm"]
                or physical["link"] != transfer_config["link"]
                or physical["topology"] != transfer_config["topology"]
                or physical["backend"] != transfer_config["backend"]
                or physical["resource_id"] != transfer_config["resource_id"]
                or physical["source_rank"] != source_rank
                or physical["destination_rank"] != destination_rank
                or tuple(physical["participants"])
                != (source_rank, destination_rank)):
            raise ValueError(
                f"{context} COMMUNICATION KernelSpec disagrees with the "
                "canonical physical activation transfer")

    resources = result["metrics"]["system_resources"]
    if set(resources["worker_pool_utilization"]) != {"attention", "ffn"}:
        raise ValueError(f"{context} worker utilization pools are invalid")
    if set(resources["scheduler_queue"]["by_pool"]) != {"attention"}:
        raise ValueError(
            f"{context} must expose only the Attention Scheduler queue")
    router = af_disagg["router"]
    router_queue = resources["router_queue"]
    if (router_queue["resource_id"] != router["resource_id"]
            or router_queue["configured_dispatch_service_time_s"]
            != router["dispatch_service_time_s"]):
        raise ValueError(f"{context} Router metrics disagree with the manifest")
    link = resources["link_utilization"]
    if set(link) != {transfer_config["resource_id"]}:
        raise ValueError(f"{context} link utilization resources are invalid")
    link_metrics = link[transfer_config["resource_id"]]
    expected_transferred_bytes = sum(
        transfer["byte_count"] for transfer in transfers)
    if (link_metrics["transfer_count"] != len(transfers)
            or link_metrics["transferred_bytes"]
            != expected_transferred_bytes):
        raise ValueError(f"{context} link utilization totals are invalid")
    communication = resources["communication_resource_utilization"]
    expected_communication_resources = {transfer_config["resource_id"]}
    if tensor_parallel_size == 2:
        expected_communication_resources.update({
            "nvlink:attention:tensor-fabric",
            "nvlink:ffn:tensor-fabric",
        })
    elif tensor_parallel_size == 1:
        pass
    else:
        raise AssertionError("validated AF TP size vanished")
    if set(communication) != expected_communication_resources:
        raise ValueError(
            f"{context} communication utilization resources are invalid")
    communication_metrics = communication[transfer_config["resource_id"]]
    if (communication_metrics["timing_status"]
            != "projected_analytical_roofline"
            or communication_metrics["operation_counts"]
            != {"send_recv": len(transfers)}):
        raise ValueError(
            f"{context} communication timing status or operation counts are "
            "invalid")
    if tensor_parallel_size == 2:
        for resource_id in (
                "nvlink:attention:tensor-fabric",
                "nvlink:ffn:tensor-fabric"):
            fabric = communication[resource_id]
            expected_operation_counts = {}
            for call in collective_calls:
                physical = call["kernel_spec"]["transfer"]
                if physical["resource_id"] != resource_id:
                    continue
                operation = physical["operation"]
                if operation in expected_operation_counts:
                    expected_operation_counts[operation] += 1
                else:
                    expected_operation_counts[operation] = 1
            if (fabric["timing_status"]
                    != "projected_analytical_roofline"
                    or fabric["operation_counts"]
                    != expected_operation_counts):
                raise ValueError(
                    f"{context} tensor fabric metrics are invalid")

    _validate_af_canonical_rematerialization(
        expected_simulation,
        expected_model,
        expected_model_path,
        result,
        expected_requests,
        expected_slo_contract,
        kv_lifecycle_replay,
        context=context,
    )


def _validate_af_canonical_rematerialization(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
    kv_lifecycle_replay: Any,
    *,
    context: str,
) -> None:
    af_disagg = expected_simulation["af_disagg"]
    if af_disagg is None:
        raise AssertionError("validated AF configuration vanished")
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    transfer = af_disagg["activation_transfer"]
    router = af_disagg["router"]
    scheduler = expected_simulation["scheduler"]
    tensor_parallel_size = scheduler["prefill_tp_size"]
    if (tensor_parallel_size not in (1, 2)
            or af_disagg["ffn_tp_size"] != tensor_parallel_size):
        raise AssertionError("validated AF TP configuration vanished")

    def collectives(pool_name: str) -> tuple[CollectiveConfig, ...]:
        if tensor_parallel_size == 1:
            return ()
        if tensor_parallel_size == 2:
            resource_id = f"nvlink:{pool_name}:tensor-fabric"
            return (
                CollectiveConfig(
                    "all_reduce", "ring", "nvlink", "ring", resource_id,
                    "analytical_roofline_v1",
                ),
                CollectiveConfig(
                    "all_gather", "ring", "nvlink", "ring", resource_id,
                    "analytical_roofline_v1",
                ),
            )
        raise AssertionError("validated AF TP size vanished")

    attention_ranks = tuple(range(tensor_parallel_size))
    ffn_ranks = tuple(range(
        tensor_parallel_size, 2 * tensor_parallel_size))
    deployment = build_af_disagg_plan(
        DeploymentConfig(
            "af_disagg",
            (
                WorkerPoolConfig(
                    "attention",
                    "attention",
                    hardware_id,
                    ParallelConfig(
                        tensor_parallel_size=tensor_parallel_size),
                    attention_ranks,
                    kv_cache=(
                        None
                        if kv_lifecycle_replay is None
                        else kv_lifecycle_replay.cache_config
                    ),
                    collectives=collectives("attention"),
                ),
                WorkerPoolConfig(
                    "ffn",
                    "ffn",
                    hardware_id,
                    ParallelConfig(
                        tensor_parallel_size=af_disagg["ffn_tp_size"]),
                    ffn_ranks,
                    collectives=collectives("ffn"),
                ),
            ),
            (
                RouteConfig(
                    "attention",
                    "ffn",
                    transfer["payload"],
                    transfer["algorithm"],
                    transfer["link"],
                    transfer["topology"],
                    resource_id=transfer["resource_id"],
                    backend=transfer["backend"],
                ),
                RouteConfig(
                    "ffn",
                    "attention",
                    transfer["payload"],
                    transfer["algorithm"],
                    transfer["link"],
                    transfer["topology"],
                    resource_id=transfer["resource_id"],
                    backend=transfer["backend"],
                ),
            ),
            RouterConfig(
                router["resource_id"],
                router["dispatch_service_time_s"],
            ),
        ),
        num_layers=expected_model.num_hidden_layers,
        num_experts=expected_model.num_experts,
    )
    expected_plan = _json_round_trip(deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            f"{context} deployment disagrees with canonical rematerialization")

    hardware = _load_expected_hardware(
        expected_simulation, hardware_id, context)
    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(
            tensor_parallel_size, 1))
    try:
        with ExitStack() as construction_lifecycle:
            backend = VLLMBackend.from_model(
                BackendConfig(
                    "attention",
                    "attention",
                    scheduler["max_batched_tokens"],
                    scheduler["max_requests"],
                    scheduler_runtime["attention"]["num_kv_blocks"],
                    kv_block_size_tokens=scheduler["kv_block_size_tokens"],
                    kv_bytes_per_token=kv_bytes_per_token,
                scheduler_pipeline_depth=scheduler["scheduler_pipeline_depth"]),
                VLLMSchedulerConfig(
                    expected_model_path,
                    tensor_parallel_size=tensor_parallel_size,
                    pipeline_parallel_size=1,
                    enable_expert_parallel=False,
                    dtype="bfloat16",
                    max_model_len=scheduler["max_model_len"],
                    enable_chunked_prefill=True,
                    enable_prefix_caching=False,
                    mtp_num_speculative_tokens=0,
                    mtp_acceptance_policy="disabled",
                    mtp_fixed_accepted_tokens=0,
                    mtp_acceptance_source_id="disabled",
                ),
            )
            construction_lifecycle.callback(backend.close)
            expected_runtime = result["scheduler_runtime"]["attention"]
            if _json_round_trip(asdict(backend.runtime_info)) != expected_runtime:
                raise ValueError(
                    f"{context} pinned Attention Scheduler runtime disagrees "
                    "with the recorded runtime")
            runtime_sha256 = (
                None
                if kv_lifecycle_replay is None
                else vllm_scheduler_runtime_sha256(backend.runtime_info)
            )
            canonical_result = MultiSimulation(
                SimulationConfig(
                    expected_model,
                    deployment,
                    expected_requests,
                    expected_simulation["launch_policy"],
                    expected_simulation["prediction_method"],
                    runtime_inputs,
                ),
                {"attention": backend},
                {hardware_id: hardware},
                kv_cache_tier_manager=(
                    None
                    if kv_lifecycle_replay is None
                    else KVCacheTierManager(
                        kv_lifecycle_replay.cache_config)
                ),
                kv_lifecycle_replay=kv_lifecycle_replay,
                scheduler_runtime_sha256=runtime_sha256,
            ).run()
            construction_lifecycle.pop_all()
    except (OSError, TypeError, ValueError, NotImplementedError, RuntimeError,
            TimeoutError) as error:
        raise ValueError(
            f"{context} cannot be canonically reproduced from the command "
            "Trace, lifecycle replay, and pinned Attention Scheduler") \
            from error

    expected_prefix_alignment = _json_round_trip(prefix_alignment_manifest(
        expected_requests,
        canonical_result,
        scheduler["kv_block_size_tokens"],
    ))
    if result["request_source"]["prefix_alignment"] != (
            expected_prefix_alignment):
        raise ValueError(
            f"{context} Prefix alignment disagrees with the command Trace "
            "and canonical Scheduler")
    expected_artifact = _json_round_trip(
        execution_artifact_manifest(
            canonical_result,
            performance_timing_status=(
                _expected_performance_timing_status(expected_simulation)),
        ))
    if result["execution_artifact"] != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")
    if kv_lifecycle_replay is not None:
        modeled_offload = expected_artifact["modeled_kv_offload"]
        if (modeled_offload["availability"] != "available_modeled_serial"
                or modeled_offload["intent_source"]["kind"]
                != "scheduler_event_replay"
                or modeled_offload["intent_source"]["replay"][
                    "source_sha256"] != kv_lifecycle_replay.source_sha256):
            raise ValueError(
                f"{context} modeled lifecycle or final tier state is invalid")
        _validate_tp1_kv_offload_invariants(
            expected_artifact,
            expected_model,
            expected_hbm_blocks=scheduler["num_kv_blocks"],
            context=context,
        )
    expected_route_decisions = expected_artifact["schedule_trace"][
        "route_decisions"]
    expected_transfers = expected_artifact["schedule_trace"]["transfers"]
    expected_kernel_calls = sum(
        step.kernel_plan.logical_call_count
        for step in canonical_result.steps)
    if (result["route_decisions"] != expected_route_decisions
            or result["transfers"] != expected_transfers
            or result["num_steps"] != len(canonical_result.steps)
            or result["num_transfers"]
            != len(canonical_result.schedule_trace.transfers)
            or result["num_kernel_calls"] != expected_kernel_calls
            or result["schedule_end_time_s"]
            != canonical_result.schedule_trace.end_time_s):
        raise ValueError(
            f"{context} counts or ordered activation trace disagree with "
            "canonical rematerialization")
    expected_metrics = _canonical_production_trace_metrics_manifest(
        deployment,
        expected_requests,
        canonical_result,
        expected_slo_contract,
    )
    if result["metrics"] != expected_metrics:
        raise ValueError(
            f"{context} metrics disagree with canonical Scheduler and "
            "execution trace")
    if kv_lifecycle_replay is not None:
        _validate_host_kv_system_resources(
            expected_metrics["system_resources"],
            pool_name="attention",
            ranks=(0,),
            hardware_id=hardware_id,
            block_bytes=(
                16
                * expected_model.kv_bytes_per_token_per_parallel_rank(
                    tensor_parallel_size, 1)
            ),
            context=context,
        )
        request_id = expected_requests.requests[0].request_id
        critical_path_contributions = expected_metrics["critical_path"][
            "requests"][request_id]["e2e"]["contribution_s_by_category"]
        if ("kernel:kv_offload" not in critical_path_contributions
                or critical_path_contributions["kernel:kv_offload"] <= 0):
            raise ValueError(
                f"{context} critical path lacks KV offload contribution")


def _validate_pd_af_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    context = "PD+AF Production Trace"
    pd_disagg = expected_simulation["pd_disagg"]
    af_disagg = expected_simulation["af_disagg"]
    if pd_disagg is None or af_disagg is None:
        raise ValueError(f"{context} command deployment config is invalid")
    instance_pair = (
        pd_disagg["num_prefill_instances"],
        pd_disagg["num_decode_instances"],
    )
    if (instance_pair not in ((1, 1), (1, 2), (2, 1), (2, 2))
            or expected_simulation["kv_lifecycle_replay"] is not None
            or expected_simulation["expert_routing"] is not None
            or expected_simulation["mtp"]["acceptance_policy"]
            != "disabled"):
        raise ValueError(f"{context} command enabled an unsupported feature")

    scheduler_runtime = result["scheduler_runtime"]
    if pd_disagg["num_prefill_instances"] == 1:
        prefill_pools = ("prefill-attention",)
    elif pd_disagg["num_prefill_instances"] == 2:
        prefill_pools = (
            "prefill-attention-0", "prefill-attention-1")
    else:
        raise AssertionError("validated PD+AF prefill instance count vanished")
    if pd_disagg["num_decode_instances"] == 1:
        decode_pools = ("decode-attention",)
    elif pd_disagg["num_decode_instances"] == 2:
        decode_pools = ("decode-attention-0", "decode-attention-1")
    else:
        raise AssertionError("validated PD+AF decode instance count vanished")
    scheduler_pools = (*prefill_pools, *decode_pools)
    if set(scheduler_runtime) != set(scheduler_pools):
        raise ValueError(
            f"{context} must contain exactly the P and D Attention "
            "Scheduler runtimes")
    scheduler = expected_simulation["scheduler"]
    for prefill_pool in prefill_pools:
        _validate_production_trace_scheduler_runtime(
            scheduler_runtime[prefill_pool],
            scheduler,
            expected_model,
            expected_simulation["mtp"],
            role="prefill-attention",
            tensor_parallel_field="prefill_tp_size",
        )
    for decode_pool in decode_pools:
        _validate_production_trace_scheduler_runtime(
            scheduler_runtime[decode_pool],
            scheduler,
            expected_model,
            expected_simulation["mtp"],
            role="decode-attention",
            tensor_parallel_field="decode_tp_size",
        )
    _validate_production_trace_capacity(
        result,
        scheduler,
        scheduler_pools,
        context=context,
    )
    if (result["routing_policy"] != pd_disagg["routing_policy"]
            or result["kv_handoff_mode"] != pd_disagg["kv_handoff_mode"]
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2
            or result["kv_lifecycle_replay"] is not None
            or result["mtp_acceptance_outcomes"] is not None
            or result["mtp_pd_handoff_contract"] is not None):
        raise ValueError(f"{context} routing or lifecycle state is invalid")
    if result["execution_artifact"]["modeled_kv_offload"] != {
            "schema_version": 2,
            "availability": "not_enabled",
            "intent_source": None,
            "transfers": [],
            "final_tier_state": None,
    }:
        raise ValueError(f"{context} unexpectedly enabled KV offload")

    request_ids = tuple(
        request.request_id for request in expected_requests.requests)
    route_decisions = result["route_decisions"]
    if instance_pair == (1, 1):
        request_prefill_pools = (prefill_pools[0],)
        request_decode_pools = (decode_pools[0],)
        expected_decisions = (
            (
                request_ids[0], "ingress_to_prefill", None,
                prefill_pools[0], [[prefill_pools[0], 0]],
            ),
            (
                request_ids[0], "prefill_to_decode", prefill_pools[0],
                decode_pools[0], [[decode_pools[0], 0]],
            ),
        )
    elif instance_pair == (1, 2):
        request_prefill_pools = tuple(
            prefill_pools[0] for _request_id in request_ids)
        request_decode_pools = (
            decode_pools[0], decode_pools[1], decode_pools[0])
        expected_decisions = tuple(
            (
                request_id, "ingress_to_prefill", None, prefill_pools[0],
                [[prefill_pools[0], index]],
            )
            for index, request_id in enumerate(request_ids)
        ) + tuple(
            (
                request_id,
                "prefill_to_decode",
                prefill_pools[0],
                destination,
                [
                    [decode_pools[0], queued_tokens[0]],
                    [decode_pools[1], queued_tokens[1]],
                ],
            )
            for request_id, destination, queued_tokens in zip(
                request_ids,
                request_decode_pools,
                ((0, 0), (1, 0), (1, 1)),
                strict=True,
            )
        )
    elif instance_pair == (2, 2):
        request_prefill_pools = (
            prefill_pools[0], prefill_pools[1])
        request_decode_pools = (decode_pools[0], decode_pools[1])
        expected_decisions = (
            (
                request_ids[0], "ingress_to_prefill", None,
                prefill_pools[0],
                [[prefill_pools[0], 0], [prefill_pools[1], 0]],
            ),
            (
                request_ids[1], "ingress_to_prefill", None,
                prefill_pools[1],
                [[prefill_pools[0], 4], [prefill_pools[1], 0]],
            ),
            (
                request_ids[0], "prefill_to_decode", prefill_pools[0],
                decode_pools[0],
                [[decode_pools[0], 0], [decode_pools[1], 0]],
            ),
            (
                request_ids[1], "prefill_to_decode", prefill_pools[1],
                decode_pools[1],
                [[decode_pools[0], 1], [decode_pools[1], 0]],
            ),
        )
    elif instance_pair == (2, 1):
        request_prefill_pools = (
            prefill_pools[0], prefill_pools[1])
        request_decode_pools = (decode_pools[0], decode_pools[0])
        expected_decisions = (
            (
                request_ids[0], "ingress_to_prefill", None,
                prefill_pools[0],
                [[prefill_pools[0], 0], [prefill_pools[1], 0]],
            ),
            (
                request_ids[1], "ingress_to_prefill", None,
                prefill_pools[1],
                [[prefill_pools[0], 4], [prefill_pools[1], 0]],
            ),
            (
                request_ids[0], "prefill_to_decode", prefill_pools[0],
                decode_pools[0], [[decode_pools[0], 0]],
            ),
            (
                request_ids[1], "prefill_to_decode", prefill_pools[1],
                decode_pools[0], [[decode_pools[0], 1]],
            ),
        )
    else:
        raise AssertionError("validated PD+AF instance pair vanished")
    if len(route_decisions) != len(expected_decisions):
        raise ValueError(
            f"{context} route decision count disagrees with the validated "
            "instance and request slice")
    for decision, expected in zip(
            route_decisions, expected_decisions, strict=True):
        request_id, stage, source, destination, candidates = expected
        if (decision["deployment_id"] != result["deployment_id"]
                or decision["request_id"] != request_id
                or decision["stage"] != stage
                or decision["policy"] != pd_disagg["routing_policy"]
                or decision["source_engine_id"] != source
                or decision["selected_engine_id"] != destination
                or decision["candidate_queued_tokens"] != candidates):
            raise ValueError(
                f"{context} route decision disagrees with the canonical "
                "P/D topology and least-queued-tokens snapshot")

    activation = af_disagg["activation_transfer"]
    kv_transfer = pd_disagg["kv_transfer"]
    activation_bytes = expected_model.hidden_size * expected_model.dtype_bytes
    kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(1, 1))
    if len(prefill_pools) == 1:
        prefill_pairs = ((prefill_pools[0], "prefill-ffn"),)
    elif len(prefill_pools) == 2:
        prefill_pairs = (
            (prefill_pools[0], "prefill-ffn-0"),
            (prefill_pools[1], "prefill-ffn-1"),
        )
    else:
        raise AssertionError("validated PD+AF prefill pool set vanished")
    if len(decode_pools) == 1:
        decode_pairs = ((decode_pools[0], "decode-ffn"),)
    elif len(decode_pools) == 2:
        decode_pairs = (
            (decode_pools[0], "decode-ffn-0"),
            (decode_pools[1], "decode-ffn-1"),
        )
    else:
        raise AssertionError("validated PD+AF decode pool set vanished")
    pool_pairs = (*prefill_pairs, *decode_pairs)
    ffn_by_attention = {
        attention_pool: ffn_pool
        for attention_pool, ffn_pool in pool_pairs
    }
    pool_ranks = {
        pool_name: 2 * pair_index + rank_offset
        for pair_index, pair in enumerate(pool_pairs)
        for rank_offset, pool_name in enumerate(pair)
    }
    payload_names_by_transfer_id = _artifact_af_payload_names_by_transfer_id(
        result, context=context)
    expected_transfer_identities = []
    for request, prefill_pool, decode_pool in zip(
            expected_requests.requests,
            request_prefill_pools,
            request_decode_pools,
            strict=True):
        prefill_ffn_pool = ffn_by_attention[prefill_pool]
        decode_ffn_pool = ffn_by_attention[decode_pool]
        prefill_bytes = request.input_tokens * activation_bytes
        for tensor_name in ("hidden_states", "residual"):
            expected_transfer_identities.append((
                (request.request_id,), "activation", prefill_pool,
                prefill_ffn_pool, prefill_bytes, tensor_name,
            ))
        for tensor_name in ("hidden_states", "residual"):
            expected_transfer_identities.append((
                (request.request_id,), "activation", prefill_ffn_pool,
                prefill_pool, prefill_bytes, tensor_name,
            ))
        expected_transfer_identities.append((
            (request.request_id,), "kv", prefill_pool, decode_pool,
            request.input_tokens * kv_bytes_per_token, None,
        ))
        for _decode_step in range(request.output_tokens - 1):
            for tensor_name in ("hidden_states", "residual"):
                expected_transfer_identities.append((
                    (request.request_id,), "activation", decode_pool,
                    decode_ffn_pool, activation_bytes, tensor_name,
                ))
            for tensor_name in ("hidden_states", "residual"):
                expected_transfer_identities.append((
                    (request.request_id,), "activation", decode_ffn_pool,
                    decode_pool, activation_bytes, tensor_name,
                ))
    transfers = result["transfers"]
    expected_step_count = sum(
        request.output_tokens for request in expected_requests.requests)
    if (result["num_steps"] != expected_step_count
            or result["num_transfers"] != len(expected_transfer_identities)
            or len(transfers) != len(expected_transfer_identities)):
        raise ValueError(
            f"{context} step or transfer count disagrees with the validated "
            "request slice")
    actual_transfer_identities = []
    for transfer in transfers:
        if transfer["payload"] == "activation":
            tensor_name = payload_names_by_transfer_id[
                transfer["transfer_id"]]
        elif transfer["payload"] == "kv":
            tensor_name = None
        else:
            raise ValueError(f"{context} transfer payload is invalid")
        actual_transfer_identities.append((
            tuple(transfer["request_ids"]),
            transfer["payload"],
            transfer["source_pool_id"],
            transfer["destination_pool_id"],
            transfer["byte_count"],
            tensor_name,
        ))
    if sorted(actual_transfer_identities) != sorted(
            expected_transfer_identities):
        raise ValueError(
            f"{context} transfer disagrees with the canonical payload, "
            "endpoint, request, or byte count")
    transfer_resources = {}
    for transfer in transfers:
        payload = transfer["payload"]
        source_pool = transfer["source_pool_id"]
        destination_pool = transfer["destination_pool_id"]
        byte_count = transfer["byte_count"]
        source_rank = pool_ranks[source_pool]
        destination_rank = pool_ranks[destination_pool]
        if payload == "activation":
            config = activation
        elif payload == "kv":
            config = kv_transfer
        else:
            raise ValueError(f"{context} transfer payload is invalid")
        endpoints = tuple(sorted((source_pool, destination_pool)))
        resource_id = f"{config['link']}:{endpoints[0]}--{endpoints[1]}"
        expected_reserved_resources = (
            "communication:endpoint:nvidia_a800:"
            f"{source_rank}:{config['link']}",
            "communication:endpoint:nvidia_a800:"
            f"{destination_rank}:{config['link']}",
            f"communication:fabric:{resource_id}",
        )
        if (transfer["resource_id"] != resource_id
                or transfer["link"] != config["link"]
                or tuple(transfer["reserved_resource_ids"])
                != expected_reserved_resources
                or transfer["end_time_s"] <= transfer["start_time_s"]):
            raise ValueError(
                f"{context} transfer disagrees with the canonical payload, "
                "endpoint, bytes, or resource reservation")
        if resource_id not in transfer_resources:
            transfer_resources[resource_id] = [0, 0]
        transfer_resources[resource_id][0] += 1
        transfer_resources[resource_id][1] += byte_count

    artifact = result["execution_artifact"]
    steps = artifact["steps"]
    kv_artifacts = artifact["transfers"]
    if len(kv_artifacts) != len(expected_requests.requests):
        raise ValueError(f"{context} KV transfer artifact count is invalid")
    expected_prefill_pool_by_request = dict(zip(
        request_ids, request_prefill_pools, strict=True))
    expected_decode_pool_by_request = dict(zip(
        request_ids, request_decode_pools, strict=True))
    kv_artifact_by_request = {}
    for kv_artifact in kv_artifacts:
        request_id = kv_artifact["request_id"]
        if request_id in kv_artifact_by_request:
            raise ValueError(f"{context} repeated one request KV artifact")
        kv_artifact_by_request[request_id] = kv_artifact
    if set(kv_artifact_by_request) != set(request_ids):
        raise ValueError(f"{context} KV transfer requests are invalid")
    for request_id in request_ids:
        kv_artifact = kv_artifact_by_request[request_id]
        kv_trace = kv_artifact["trace_transfer"]
        prefill_steps = tuple(
            step for step in steps
            if step["pool_name"]
            == expected_prefill_pool_by_request[request_id]
            and any(
                request["request_id"] == request_id
                for request in step["backend_step"]["schedule"]["requests"])
        )
        decode_steps = tuple(
            step for step in steps
            if step["pool_name"]
            == expected_decode_pool_by_request[request_id]
            and any(
                request["request_id"] == request_id
                for request in step["backend_step"]["schedule"]["requests"])
        )
        if len(prefill_steps) != 1 or not decode_steps:
            raise ValueError(
                f"{context} request Scheduler step ownership is invalid")
        first_decode_step = min(
            decode_steps,
            key=lambda step: (
                step["schedule_interval"]["start_time_s"], step["step_id"]),
        )
        if (kv_trace["payload"] != "kv"
                or kv_trace["source_pool_id"]
                != expected_prefill_pool_by_request[request_id]
                or kv_trace["destination_pool_id"]
                != expected_decode_pool_by_request[request_id]
                or tuple(kv_trace["request_ids"]) != (request_id,)
                or kv_trace["start_time_s"]
                < prefill_steps[0]["schedule_interval"]["end_time_s"]
                or first_decode_step["schedule_interval"]["start_time_s"]
                < kv_trace["end_time_s"]):
            raise ValueError(
                f"{context} must complete each prefill before KV handoff "
                "and start decode only after that handoff completes")

    step_communication_calls = tuple(
        call
        for step in steps
        for call in step["kernel_plan"]["calls"]
        if call["kernel_spec"]["kind"] == KernelKind.COMMUNICATION.value
    )
    kv_communication_calls = tuple(
        call
        for transfer_artifact in kv_artifacts
        for call in transfer_artifact["kernel_plan"]["calls"]
        if call["kernel_spec"]["kind"] == KernelKind.COMMUNICATION.value
    )
    communication_calls = (*step_communication_calls, *kv_communication_calls)
    if len(communication_calls) != len(transfers):
        raise ValueError(
            f"{context} must lower every activation and KV transfer to one "
            "COMMUNICATION KernelSpec")
    calls_by_id = {}
    for call in communication_calls:
        call_id = call["call_id"]
        if call_id in calls_by_id:
            raise ValueError(f"{context} repeated COMMUNICATION call_id")
        calls_by_id[call_id] = call
    transfer_ids = {transfer["transfer_id"] for transfer in transfers}
    if set(calls_by_id) != transfer_ids:
        raise ValueError(
            f"{context} COMMUNICATION calls disagree with transfer IDs")
    for transfer in transfers:
        call = calls_by_id[transfer["transfer_id"]]
        payload = transfer["payload"]
        source_pool = transfer["source_pool_id"]
        destination_pool = transfer["destination_pool_id"]
        byte_count = transfer["byte_count"]
        source_rank = pool_ranks[source_pool]
        destination_rank = pool_ranks[destination_pool]
        if payload == "activation":
            config = activation
        elif payload == "kv":
            config = kv_transfer
        else:
            raise AssertionError("validated PD+AF transfer payload vanished")
        endpoints = tuple(sorted((source_pool, destination_pool)))
        resource_id = f"{config['link']}:{endpoints[0]}--{endpoints[1]}"
        expected_physical_transfer = {
            "algorithm": config["algorithm"],
            "bytes": byte_count,
            "direction": "point_to_point",
            "link": config["link"],
            "topology": config["topology"],
            "tier": None,
            "participants": [source_rank, destination_rank],
            "operation": config["algorithm"],
            "bytes_semantics": "payload",
            "source_rank": source_rank,
            "destination_rank": destination_rank,
            "ring_order": [],
            "reduction": None,
            "backend": config["backend"],
            "resource_id": resource_id,
            "backend_config_sha256": None,
        }
        expected_kernel_spec = {
            "kind": KernelKind.COMMUNICATION.value,
            "hardware_id": expected_simulation["hardware"]["hardware_id"],
            "shape": {},
            "dtype": None,
            "variant": None,
            "launch": None,
            "transfer": expected_physical_transfer,
            "attributes": {},
        }
        if (call["pool_name"] != source_pool
                or call["rank"] != source_rank
                or call["resource"] != resource_id
                or call["kernel_spec"] != expected_kernel_spec):
            raise ValueError(
                f"{context} COMMUNICATION KernelSpec disagrees with the "
                "canonical physical transfer contract")

    resources = result["metrics"]["system_resources"]
    worker_pools = set(pool_ranks)
    if set(resources["worker_pool_utilization"]) != worker_pools:
        raise ValueError(f"{context} worker utilization pools are invalid")
    if set(resources["scheduler_queue"]["by_pool"]) != set(scheduler_pools):
        raise ValueError(f"{context} Scheduler queue pools are invalid")
    router = pd_disagg["router"]
    router_queue = resources["router_queue"]
    if (router_queue["resource_id"] != router["resource_id"]
            or router_queue["configured_dispatch_service_time_s"]
            != router["dispatch_service_time_s"]):
        raise ValueError(f"{context} Router metrics disagree with the manifest")
    link_metrics = resources["link_utilization"]
    communication_metrics = resources["communication_resource_utilization"]
    expected_link_resources = {
        f"{activation['link']}:"
        f"{min(attention_pool, ffn_pool)}--"
        f"{max(attention_pool, ffn_pool)}"
        for attention_pool, ffn_pool in pool_pairs
    }
    expected_link_resources.update(
        f"{kv_transfer['link']}:"
        f"{min(prefill_pool, decode_pool)}--"
        f"{max(prefill_pool, decode_pool)}"
        for prefill_pool in prefill_pools
        for decode_pool in decode_pools
    )
    if (set(link_metrics) != expected_link_resources
            or set(communication_metrics) != set(transfer_resources)):
        raise ValueError(f"{context} communication resources are invalid")
    for resource_id in expected_link_resources:
        if resource_id in transfer_resources:
            transfer_count, transferred_bytes = transfer_resources[resource_id]
        else:
            transfer_count, transferred_bytes = 0, 0
        if (link_metrics[resource_id]["transfer_count"] != transfer_count
                or link_metrics[resource_id]["transferred_bytes"]
                != transferred_bytes):
            raise ValueError(
                f"{context} link metrics are inconsistent")
    for resource_id, (transfer_count, transferred_bytes) in (
            transfer_resources.items()):
        if (communication_metrics[resource_id]["timing_status"]
                != "projected_analytical_roofline"
                or communication_metrics[resource_id]["operation_counts"]
                != {"send_recv": transfer_count}):
            raise ValueError(
                f"{context} communication metrics are inconsistent")

    _validate_pd_af_canonical_rematerialization(
        expected_simulation,
        expected_model,
        expected_model_path,
        result,
        expected_requests,
        expected_slo_contract,
    )


def _validate_pd_af_canonical_rematerialization(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    context = "PD+AF Production Trace"
    pd_disagg = expected_simulation["pd_disagg"]
    af_disagg = expected_simulation["af_disagg"]
    if pd_disagg is None or af_disagg is None:
        raise AssertionError("validated PD+AF configuration vanished")
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    instance_pair = (
        pd_disagg["num_prefill_instances"],
        pd_disagg["num_decode_instances"],
    )
    if instance_pair not in ((1, 1), (1, 2), (2, 1), (2, 2)):
        raise AssertionError("validated PD+AF instance pair vanished")
    if pd_disagg["num_prefill_instances"] == 1:
        prefill_pairs = (("prefill-attention", "prefill-ffn"),)
    elif pd_disagg["num_prefill_instances"] == 2:
        prefill_pairs = (
            ("prefill-attention-0", "prefill-ffn-0"),
            ("prefill-attention-1", "prefill-ffn-1"),
        )
    else:
        raise AssertionError("validated PD+AF prefill instance count vanished")
    if pd_disagg["num_decode_instances"] == 1:
        decode_pairs = (("decode-attention", "decode-ffn"),)
    elif pd_disagg["num_decode_instances"] == 2:
        decode_pairs = (
            ("decode-attention-0", "decode-ffn-0"),
            ("decode-attention-1", "decode-ffn-1"),
        )
    else:
        raise AssertionError("validated PD+AF decode instance count vanished")
    pool_pairs = (*prefill_pairs, *decode_pairs)
    role_pairs = (
        *((pair, "prefill-attention", "prefill-ffn")
          for pair in prefill_pairs),
        *((pair, "decode-attention", "decode-ffn")
          for pair in decode_pairs),
    )
    pools = tuple(
        (pool_name, role, 2 * pair_index + rank_offset)
        for pair_index, (pair, attention_role, ffn_role) in enumerate(
            role_pairs)
        for rank_offset, (pool_name, role) in enumerate((
            (pair[0], attention_role),
            (pair[1], ffn_role),
        ))
    )
    worker_pools = tuple(WorkerPoolConfig(
        name,
        role,
        hardware_id,
        ParallelConfig(tensor_parallel_size=1),
        (rank,),
    ) for name, role, rank in pools)
    activation = af_disagg["activation_transfer"]
    kv_transfer = pd_disagg["kv_transfer"]

    def route(
        source: str,
        destination: str,
        payload: str,
        config: dict[str, Any],
    ) -> RouteConfig:
        endpoints = tuple(sorted((source, destination)))
        return RouteConfig(
            source,
            destination,
            payload,
            config["algorithm"],
            config["link"],
            config["topology"],
            resource_id=(
                f"{config['link']}:{endpoints[0]}--{endpoints[1]}"),
            backend=config["backend"],
        )

    routes = tuple(
        route(source_pool, destination_pool, "activation", activation)
        for attention_pool, ffn_pool in pool_pairs
        for source_pool, destination_pool in (
            (attention_pool, ffn_pool),
            (ffn_pool, attention_pool),
        )
    ) + tuple(
        route(prefill_pair[0], decode_pair[0], "kv", kv_transfer)
        for prefill_pair in prefill_pairs
        for decode_pair in decode_pairs
    )
    router = pd_disagg["router"]
    deployment = build_pd_disagg_plan(
        DeploymentConfig(
            "pd_af_disagg",
            worker_pools,
            routes,
            RouterConfig(
                router["resource_id"],
                router["dispatch_service_time_s"],
            ),
        ),
        num_layers=expected_model.num_hidden_layers,
        num_experts=expected_model.num_experts,
    )
    expected_plan = _json_round_trip(deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            f"{context} deployment disagrees with canonical rematerialization")

    hardware = _load_expected_hardware(
        expected_simulation, hardware_id, context)
    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    scheduler = expected_simulation["scheduler"]
    kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(1, 1))
    backends = {}
    try:
        with ExitStack() as construction_lifecycle:
            scheduler_roles = (
                *tuple(
                    (prefill_pair[0], "prefill-attention")
                    for prefill_pair in prefill_pairs
                ),
                *tuple(
                    (decode_pair[0], "decode-attention")
                    for decode_pair in decode_pairs
                ),
            )
            for pool_name, role in scheduler_roles:
                backend = VLLMBackend.from_model(
                    BackendConfig(
                        pool_name,
                        role,
                        scheduler["max_batched_tokens"],
                        scheduler["max_requests"],
                        scheduler_runtime[pool_name]["num_kv_blocks"],
                        kv_block_size_tokens=scheduler[
                            "kv_block_size_tokens"],
                        kv_bytes_per_token=kv_bytes_per_token,
                    scheduler_pipeline_depth=scheduler[
                        "scheduler_pipeline_depth"]),
                    VLLMSchedulerConfig(
                        expected_model_path,
                        tensor_parallel_size=1,
                        pipeline_parallel_size=1,
                        enable_expert_parallel=False,
                        dtype="bfloat16",
                        max_model_len=scheduler["max_model_len"],
                        enable_chunked_prefill=True,
                        enable_prefix_caching=False,
                        mtp_num_speculative_tokens=0,
                        mtp_acceptance_policy="disabled",
                        mtp_fixed_accepted_tokens=0,
                        mtp_acceptance_source_id="disabled",
                    ),
                )
                construction_lifecycle.callback(backend.close)
                expected_runtime = result["scheduler_runtime"][pool_name]
                if _json_round_trip(asdict(backend.runtime_info)) != (
                        expected_runtime):
                    raise ValueError(
                        f"{context} pinned Scheduler runtime for "
                        f"{pool_name!r} disagrees with the recorded runtime")
                backends[pool_name] = backend
            canonical_result = MultiSimulation(
                SimulationConfig(
                    expected_model,
                    deployment,
                    expected_requests,
                    expected_simulation["launch_policy"],
                    expected_simulation["prediction_method"],
                    runtime_inputs,
                ),
                backends,
                {hardware_id: hardware},
            ).run()
            construction_lifecycle.pop_all()
    except (OSError, TypeError, ValueError, NotImplementedError, RuntimeError,
            TimeoutError) as error:
        raise ValueError(
            f"{context} cannot be canonically reproduced from the command "
            "Trace and pinned P/D Attention Schedulers") from error

    expected_prefix_alignment = _json_round_trip(prefix_alignment_manifest(
        expected_requests,
        canonical_result,
        scheduler["kv_block_size_tokens"],
    ))
    if result["request_source"]["prefix_alignment"] != (
            expected_prefix_alignment):
        raise ValueError(
            f"{context} Prefix alignment disagrees with the command Trace "
            "and canonical Schedulers")
    expected_artifact = _json_round_trip(
        execution_artifact_manifest(
            canonical_result,
            performance_timing_status=(
                _expected_performance_timing_status(expected_simulation)),
        ))
    if result["execution_artifact"] != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")
    expected_route_decisions = expected_artifact["schedule_trace"][
        "route_decisions"]
    expected_transfers = expected_artifact["schedule_trace"]["transfers"]
    expected_kernel_calls = (
        sum(step.kernel_plan.logical_call_count
            for step in canonical_result.steps)
        + sum(
            transfer.kernel_plan.logical_call_count
            for transfer in canonical_result.transfers
        )
    )
    if (result["route_decisions"] != expected_route_decisions
            or result["transfers"] != expected_transfers
            or result["num_steps"] != len(canonical_result.steps)
            or result["num_transfers"]
            != len(canonical_result.schedule_trace.transfers)
            or result["num_kernel_calls"] != expected_kernel_calls
            or result["schedule_end_time_s"]
            != canonical_result.schedule_trace.end_time_s):
        raise ValueError(
            f"{context} counts or ordered transfer trace disagree with "
            "canonical rematerialization")
    expected_metrics = _canonical_production_trace_metrics_manifest(
        deployment,
        expected_requests,
        canonical_result,
        expected_slo_contract,
    )
    if result["metrics"] != expected_metrics:
        raise ValueError(
            f"{context} metrics disagree with canonical Schedulers and "
            "execution trace")


def _validate_pd_production_trace_point(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
) -> None:
    if expected_simulation["af_disagg"] is not None:
        raise ValueError("PD Production Trace command retained AF config")
    pd_disagg = expected_simulation["pd_disagg"]
    if pd_disagg is None:
        raise ValueError("PD Production Trace command lost PD configuration")
    expected_scheduler = expected_simulation["scheduler"]
    prefill_pools = tuple(
        f"prefill-{index}"
        for index in range(pd_disagg["num_prefill_instances"])
    )
    decode_pools = tuple(
        f"decode-{index}"
        for index in range(pd_disagg["num_decode_instances"])
    )
    expected_pools = prefill_pools + decode_pools
    scheduler_runtime = result["scheduler_runtime"]
    if set(scheduler_runtime) != set(expected_pools):
        raise ValueError(
            "PD Production Trace sweep Scheduler pools disagree with "
            "command manifest")
    for pool_name in prefill_pools:
        _validate_production_trace_scheduler_runtime(
            scheduler_runtime[pool_name],
            expected_scheduler,
            expected_model,
            expected_simulation["mtp"],
            role="prefill",
            tensor_parallel_field="prefill_tp_size",
        )
    for pool_name in decode_pools:
        _validate_production_trace_scheduler_runtime(
            scheduler_runtime[pool_name],
            expected_scheduler,
            expected_model,
            expected_simulation["mtp"],
            role="decode",
            tensor_parallel_field="decode_tp_size",
        )
    _validate_pd_deployment_plan(
        expected_simulation,
        result,
        prefill_pools,
        decode_pools,
    )
    _validate_pd_capacity_and_routes(
        expected_simulation,
        result,
        prefill_pools,
        decode_pools,
    )
    _validate_pd_canonical_rematerialization(
        expected_simulation,
        expected_model,
        expected_model_path,
        result,
        expected_requests,
        expected_slo_contract,
        prefill_pools,
        decode_pools,
    )


def _validate_pd_deployment_plan(
    expected_simulation: dict[str, Any],
    result: dict[str, Any],
    prefill_pools: tuple[str, ...],
    decode_pools: tuple[str, ...],
) -> None:
    pd_disagg = expected_simulation["pd_disagg"]
    if pd_disagg is None:
        raise AssertionError("validated PD configuration vanished")
    plan = result["deployment_plan"]
    if (plan["schema_version"] != 6
            or plan["mode"] != "pd_disagg"
            or plan["deployment_id"] != result["deployment_id"]):
        raise ValueError(
            "PD Production Trace deployment identity disagrees with "
            "command manifest")
    if plan["router"] != pd_disagg["router"]:
        raise ValueError(
            "PD Production Trace router contract disagrees with command "
            "manifest")
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    default_parallelism = {
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "enable_expert_parallel": False,
        "expert_communication_backend": "allgather_reducescatter",
        "enable_eplb": False,
        "sequence_parallel_size": 1,
        "context_parallel_size": 1,
        "data_parallel_size": 1,
    }
    expected_pool_contracts = {}
    for rank, pool_name in enumerate(prefill_pools):
        expected_pool_contracts[pool_name] = {
            "name": pool_name,
            "role": "prefill",
            "hardware_id": hardware_id,
            "ranks": [rank],
            "parallelism": default_parallelism,
            "kv_cache": None,
            "collectives": [],
            "pipeline_p2p": [],
        }
    decode_rank_offset = len(prefill_pools)
    for index, pool_name in enumerate(decode_pools):
        expected_pool_contracts[pool_name] = {
            "name": pool_name,
            "role": "decode",
            "hardware_id": hardware_id,
            "ranks": [decode_rank_offset + index],
            "parallelism": default_parallelism,
            "kv_cache": None,
            "collectives": [],
            "pipeline_p2p": [],
        }
    pools = plan["worker_pools"]
    actual_pool_contracts = {pool["name"]: pool for pool in pools}
    if (len(actual_pool_contracts) != len(pools)
            or actual_pool_contracts != expected_pool_contracts):
        raise ValueError(
            "PD Production Trace worker-pool topology disagrees with "
            "command manifest")
    transfer = pd_disagg["kv_transfer"]
    expected_routes = {}
    for source_pool in prefill_pools:
        for destination_pool in decode_pools:
            endpoints = tuple(sorted((source_pool, destination_pool)))
            resource_id = (
                f"{transfer['link']}:{endpoints[0]}--{endpoints[1]}")
            expected_routes[(source_pool, destination_pool)] = {
                "source_pool": source_pool,
                "destination_pool": destination_pool,
                "payload": transfer["payload"],
                "algorithm": transfer["algorithm"],
                "link": transfer["link"],
                "topology": transfer["topology"],
                "resource_id": resource_id,
                "backend": transfer["backend"],
                "backend_config_sha256": None,
            }
    routes = plan["routes"]
    actual_routes = {
        (route["source_pool"], route["destination_pool"]): route
        for route in routes
    }
    if (len(actual_routes) != len(routes)
            or actual_routes != expected_routes):
        raise ValueError(
            "PD Production Trace transfer routes disagree with command "
            "manifest")


def _validate_pd_capacity_and_routes(
    expected_simulation: dict[str, Any],
    result: dict[str, Any],
    prefill_pools: tuple[str, ...],
    decode_pools: tuple[str, ...],
) -> None:
    pd_disagg = expected_simulation["pd_disagg"]
    if pd_disagg is None:
        raise AssertionError("validated PD configuration vanished")
    if (result["routing_policy"] != pd_disagg["routing_policy"]
            or result["kv_handoff_mode"] != pd_disagg["kv_handoff_mode"]
            or result["route_decision_version"] != 2
            or result["transfer_trace_version"] != 2):
        raise ValueError(
            "PD Production Trace routing contract disagrees with command "
            "manifest")
    if result["kv_lifecycle_replay"] is not None:
        raise ValueError(
            "PD Production Trace sweep unexpectedly enabled KV offload")
    if result["execution_artifact"]["modeled_kv_offload"][
            "availability"] != "not_enabled":
        raise ValueError(
            "PD Production Trace execution unexpectedly modeled KV offload")
    mtp_scenario = result["mtp_scenario"]
    if (mtp_scenario["num_speculative_tokens"] != 0
            or mtp_scenario["acceptance_policy"] != "disabled"
            or result["mtp_acceptance_outcomes"] is not None):
        raise ValueError(
            "PD Production Trace sweep unexpectedly enabled MTP")
    expected_pools = prefill_pools + decode_pools
    expected_pool_set = set(expected_pools)
    _validate_production_trace_capacity(
        result,
        expected_simulation["scheduler"],
        expected_pools,
        context="PD Production Trace",
    )
    capacity_plans = result["kv_capacity"]["scheduler_pools"]
    plans_by_pool = {
        plan["pool_name"]: plan for plan in capacity_plans
    }
    if (len(plans_by_pool) != len(capacity_plans)
            or set(plans_by_pool) != expected_pool_set):
        raise ValueError(
            "PD Production Trace KV capacity pools disagree with command "
            "manifest")
    artifact = result["execution_artifact"]
    artifact_trace = artifact["schedule_trace"]
    if (artifact_trace["route_decision_version"] != 2
            or artifact_trace["transfer_trace_version"] != 2
            or artifact_trace["route_decisions"]
            != result["route_decisions"]
            or artifact_trace["transfers"] != result["transfers"]):
        raise ValueError(
            "PD Production Trace routing records disagree with the execution "
            "artifact")
    artifact_counts = artifact["counts"]
    if (result["num_transfers"] != len(result["transfers"])
            or artifact_counts["trace_transfers"]
            != len(result["transfers"])
            or artifact_counts["simulation_transfers"]
            != len(artifact["transfers"])
            or len(artifact["transfers"]) != len(result["transfers"])):
        raise ValueError(
            "PD Production Trace transfer counts disagree with the execution "
            "artifact")
    trace_transfers_by_id = {
        transfer["transfer_id"]: transfer
        for transfer in result["transfers"]
    }
    if len(trace_transfers_by_id) != len(result["transfers"]):
        raise ValueError(
            "PD Production Trace contains duplicate transfer IDs")
    artifact_transfers_by_id = {
        transfer["trace_transfer"]["transfer_id"]:
        transfer["trace_transfer"]
        for transfer in artifact["transfers"]
    }
    if (len(artifact_transfers_by_id) != len(artifact["transfers"])
            or artifact_transfers_by_id != trace_transfers_by_id):
        raise ValueError(
            "PD Production Trace transfers disagree with the execution "
            "artifact")
    plan_routes = {
        (route["source_pool"], route["destination_pool"]): route
        for route in result["deployment_plan"]["routes"]
    }
    prefill_set = set(prefill_pools)
    decode_set = set(decode_pools)
    prefill_assignment = {}
    decode_assignment = {}
    for step in artifact["steps"]:
        pool_name = step["pool_name"]
        schedule = step["backend_step"]["schedule"]
        if (pool_name not in expected_pool_set
                or schedule["pool_name"] != pool_name):
            raise ValueError(
                "PD Production Trace execution step uses an unknown pool")
        if pool_name in prefill_set:
            assignment = prefill_assignment
        elif pool_name in decode_set:
            assignment = decode_assignment
        else:
            raise AssertionError("validated PD execution pool vanished")
        for request in schedule["requests"]:
            request_id = request["request_id"]
            if (request_id in assignment
                    and assignment[request_id] != pool_name):
                raise ValueError(
                    "PD Production Trace request executed in multiple pools "
                    "for one role")
            assignment[request_id] = pool_name
    ingress_by_request = {}
    handoff_by_request = {}
    for raw_decision in result["route_decisions"]:
        try:
            _route_decision_from_output_manifest(raw_decision)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "PD Production Trace route decision violates the canonical "
                f"RouteDecision contract: {error}") from error
        decision = raw_decision
        if (decision["deployment_id"] != result["deployment_id"]
                or decision["policy"] != pd_disagg["routing_policy"]):
            raise ValueError(
                "PD Production Trace route decision identity disagrees "
                "with command manifest")
        candidate_ids = tuple(
            candidate[0]
            for candidate in decision["candidate_queued_tokens"])
        if decision["stage"] == "ingress_to_prefill":
            if (candidate_ids != tuple(sorted(prefill_pools))
                    or decision["selected_engine_id"] not in prefill_set
                    or decision["source_engine_id"] is not None):
                raise ValueError(
                    "PD Production Trace ingress route disagrees with "
                    "command manifest")
            destination = ingress_by_request
        elif decision["stage"] == "prefill_to_decode":
            if (candidate_ids != tuple(sorted(decode_pools))
                    or decision["selected_engine_id"] not in decode_set
                    or decision["source_engine_id"] not in prefill_set):
                raise ValueError(
                    "PD Production Trace handoff route disagrees with "
                    "command manifest")
            destination = handoff_by_request
        else:
            raise ValueError(
                "PD Production Trace contains an unsupported route stage")
        request_id = decision["request_id"]
        if request_id in destination:
            raise ValueError(
                "PD Production Trace contains duplicate per-request route "
                "decisions")
        destination[request_id] = decision
    arrival_ids = tuple(
        arrival["request_id"] for arrival in result["request_arrivals_s"])
    if (len(arrival_ids) != len(set(arrival_ids))
            or set(arrival_ids) != set(result["metrics"]["requests"])
            or set(arrival_ids) != set(ingress_by_request)
            or set(arrival_ids) != set(prefill_assignment)):
        raise ValueError(
            "PD Production Trace ingress coverage disagrees with requests")
    for request_id, decision in ingress_by_request.items():
        if decision["selected_engine_id"] != prefill_assignment[request_id]:
            raise ValueError(
                "PD Production Trace ingress route disagrees with execution "
                "pool assignment")
    if set(handoff_by_request) != set(decode_assignment):
        raise ValueError(
            "PD Production Trace handoff coverage disagrees with decode "
            "execution")
    for request_id, decision in handoff_by_request.items():
        if (decision["source_engine_id"] != prefill_assignment[request_id]
                or decision["selected_engine_id"]
                != decode_assignment[request_id]):
            raise ValueError(
                "PD Production Trace handoff route disagrees with execution "
                "pool assignment")
    transfers_by_request = {}
    for transfer in result["transfers"]:
        endpoints = (
            transfer["source_pool_id"], transfer["destination_pool_id"])
        if endpoints not in plan_routes:
            raise ValueError(
                "PD Production Trace transfer references an unknown route")
        route = plan_routes[endpoints]
        if (transfer["payload"] != "kv"
                or transfer["resource_id"] != route["resource_id"]
                or transfer["link"] != route["link"]):
            raise ValueError(
                "PD Production Trace transfer disagrees with command "
                "manifest")
        request_ids = transfer["request_ids"]
        if len(request_ids) != 1:
            raise ValueError(
                "PD Production Trace KV transfer must contain one request")
        request_id = request_ids[0]
        if request_id in transfers_by_request:
            raise ValueError(
                "PD Production Trace contains duplicate per-request KV "
                "transfers")
        transfers_by_request[request_id] = transfer
    if set(transfers_by_request) != set(handoff_by_request):
        raise ValueError(
            "PD Production Trace transfer coverage disagrees with handoff "
            "routing")
    for request_id, transfer in transfers_by_request.items():
        decision = handoff_by_request[request_id]
        if (transfer["source_pool_id"] != decision["source_engine_id"]
                or transfer["destination_pool_id"]
                != decision["selected_engine_id"]):
            raise ValueError(
                "PD Production Trace transfer endpoints disagree with "
                "handoff routing")
    system_resources = result["metrics"]["system_resources"]
    if set(system_resources["worker_pool_utilization"]) != expected_pool_set:
        raise ValueError(
            "PD Production Trace worker utilization pools disagree with "
            "command manifest")
    scheduler_queue = system_resources["scheduler_queue"]
    if set(scheduler_queue["by_pool"]) != expected_pool_set:
        raise ValueError(
            "PD Production Trace Scheduler queue pools disagree with command "
            "manifest")
    router_queue = system_resources["router_queue"]
    if (router_queue["resource_id"] != pd_disagg["router"]["resource_id"]
            or router_queue["configured_dispatch_service_time_s"]
            != pd_disagg["router"]["dispatch_service_time_s"]):
        raise ValueError(
            "PD Production Trace Router metrics disagree with command "
            "manifest")


def _validate_pd_canonical_rematerialization(
    expected_simulation: dict[str, Any],
    expected_model: ModelConfig,
    expected_model_path: str,
    result: dict[str, Any],
    expected_requests: RequestStream,
    expected_slo_contract: Any,
    prefill_pools: tuple[str, ...],
    decode_pools: tuple[str, ...],
) -> None:
    context = "PD Production Trace"
    pd_disagg = expected_simulation["pd_disagg"]
    if pd_disagg is None:
        raise AssertionError("validated PD configuration vanished")
    hardware_id = expected_simulation["hardware"]["hardware_id"]
    worker_pools = []
    for rank, pool_name in enumerate(prefill_pools):
        worker_pools.append(WorkerPoolConfig(
            pool_name,
            "prefill",
            hardware_id,
            ParallelConfig(),
            (rank,),
        ))
    decode_rank_offset = len(prefill_pools)
    for index, pool_name in enumerate(decode_pools):
        worker_pools.append(WorkerPoolConfig(
            pool_name,
            "decode",
            hardware_id,
            ParallelConfig(),
            (decode_rank_offset + index,),
        ))
    transfer = pd_disagg["kv_transfer"]
    routes = []
    for source_pool in prefill_pools:
        for destination_pool in decode_pools:
            endpoints = tuple(sorted((source_pool, destination_pool)))
            routes.append(RouteConfig(
                source_pool,
                destination_pool,
                transfer["payload"],
                transfer["algorithm"],
                transfer["link"],
                transfer["topology"],
                resource_id=(
                    f"{transfer['link']}:{endpoints[0]}--{endpoints[1]}"),
                backend=transfer["backend"],
            ))
    router = pd_disagg["router"]
    deployment = build_pd_disagg_plan(
        DeploymentConfig(
            "pd_disagg",
            tuple(worker_pools),
            tuple(routes),
            RouterConfig(
                router["resource_id"],
                router["dispatch_service_time_s"],
            ),
        ),
        num_layers=expected_model.num_hidden_layers,
        num_experts=expected_model.num_experts,
    )
    expected_plan = _json_round_trip(deployment_plan_manifest(deployment))
    if (result["deployment_id"] != deployment.deployment_id
            or result["deployment_plan"] != expected_plan):
        raise ValueError(
            f"{context} deployment disagrees with canonical "
            "rematerialization")
    hardware = _load_expected_hardware(
        expected_simulation, hardware_id, context)
    hardware_by_id = {hardware_id: hardware}
    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        expected_simulation["attention_kv_layout"],
        expected_simulation["attention_kv_distribution"],
    )
    scheduler = expected_simulation["scheduler"]
    kv_bytes_per_token = (
        expected_model.kv_bytes_per_token_per_parallel_rank(1, 1))
    backends = {}
    try:
        with ExitStack() as construction_lifecycle:
            for pool_name in prefill_pools + decode_pools:
                role = (
                    "prefill" if pool_name in prefill_pools else "decode")
                backend = VLLMBackend.from_model(
                    BackendConfig(
                        pool_name,
                        role,
                        scheduler["max_batched_tokens"],
                        scheduler["max_requests"],
                        result["scheduler_runtime"][pool_name][
                            "num_kv_blocks"],
                        kv_block_size_tokens=scheduler[
                            "kv_block_size_tokens"],
                        kv_bytes_per_token=kv_bytes_per_token,
                    scheduler_pipeline_depth=scheduler[
                        "scheduler_pipeline_depth"]),
                    VLLMSchedulerConfig(
                        expected_model_path,
                        tensor_parallel_size=1,
                        pipeline_parallel_size=1,
                        enable_expert_parallel=False,
                        dtype="bfloat16",
                        max_model_len=scheduler["max_model_len"],
                        enable_chunked_prefill=True,
                        enable_prefix_caching=False,
                        mtp_num_speculative_tokens=0,
                        mtp_acceptance_policy="disabled",
                        mtp_fixed_accepted_tokens=0,
                        mtp_acceptance_source_id="disabled",
                    ),
                )
                construction_lifecycle.callback(backend.close)
                expected_runtime = result["scheduler_runtime"][pool_name]
                if _json_round_trip(asdict(backend.runtime_info)) != (
                        expected_runtime):
                    raise ValueError(
                        f"{context} pinned Scheduler runtime for "
                        f"{pool_name!r} disagrees with the recorded runtime")
                backends[pool_name] = backend
            canonical_result = MultiSimulation(
                SimulationConfig(
                    expected_model,
                    deployment,
                    expected_requests,
                    expected_simulation["launch_policy"],
                    expected_simulation["prediction_method"],
                    runtime_inputs,
                ),
                backends,
                hardware_by_id,
            ).run()
            construction_lifecycle.pop_all()
    except (OSError, TypeError, ValueError, NotImplementedError, RuntimeError,
            TimeoutError) as error:
        raise ValueError(
            f"{context} cannot be canonically reproduced from the command "
            "Trace and pinned Scheduler") from error

    expected_prefix_alignment = _json_round_trip(prefix_alignment_manifest(
        expected_requests,
        canonical_result,
        scheduler["kv_block_size_tokens"],
    ))
    if result["request_source"]["prefix_alignment"] != (
            expected_prefix_alignment):
        raise ValueError(
            f"{context} Prefix alignment disagrees with the command Trace "
            "and canonical Scheduler")
    expected_artifact = _json_round_trip(
        execution_artifact_manifest(
            canonical_result,
            performance_timing_status=(
                _expected_performance_timing_status(expected_simulation)),
        ))
    if result["execution_artifact"] != expected_artifact:
        raise ValueError(
            f"{context} execution artifact disagrees with canonical "
            "rematerialization")
    expected_route_decisions = expected_artifact["schedule_trace"][
        "route_decisions"]
    expected_transfers = expected_artifact["schedule_trace"]["transfers"]
    expected_kernel_calls = (
        sum(step.kernel_plan.logical_call_count
            for step in canonical_result.steps)
        + sum(
            transfer.kernel_plan.logical_call_count
            for transfer in canonical_result.transfers
        )
    )
    if (result["route_decisions"] != expected_route_decisions
            or result["transfers"] != expected_transfers
            or result["num_steps"] != len(canonical_result.steps)
            or result["num_transfers"]
            != len(canonical_result.schedule_trace.transfers)
            or result["num_kernel_calls"] != expected_kernel_calls
            or result["schedule_end_time_s"]
            != canonical_result.schedule_trace.end_time_s):
        raise ValueError(
            f"{context} counts or ordered routing trace disagree with "
            "canonical rematerialization")
    expected_metrics = _canonical_production_trace_metrics_manifest(
        deployment,
        expected_requests,
        canonical_result,
        expected_slo_contract,
    )
    if result["metrics"] != expected_metrics:
        raise ValueError(
            f"{context} metrics disagree with canonical Scheduler and "
            "execution trace")


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(
                f"duplicate Production Trace sweep JSON field: {key!r}")
        payload[key] = value
    return payload


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-standard JSON numeric constant: {value!r}")


def _failure_status(
    stage: str,
    result,
    logs_dir: Path,
    command: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stderr = _required_process_stream(result, "stderr").strip()
    dependency_category = _dependency_category(stderr)
    status = "blocked" if dependency_category else "failed"
    failure = {
        "status": status,
        "failed_stage": stage,
        "returncode": result.returncode,
        "stderr_summary": stderr[:500],
        "stdout_log": str(logs_dir / f"{stage}.stdout.log"),
        "stderr_log": str(logs_dir / f"{stage}.stderr.log"),
    }
    if command is not None:
        failure["reproducer_commands"] = {
            stage: list(command["argv"]),
        }
    if dependency_category:
        failure["dependency_category"] = dependency_category
        failure["unblock_checklist"] = _unblock_checklist(dependency_category)
    return failure


def _stage_environment(
    command_manifest: dict[str, Any],
    command: dict[str, Any],
) -> dict[str, str]:
    env = dict(os.environ)
    env.update(command_manifest["environment"])
    if "environment" in command:
        env.update(command["environment"])
    return env


def _missing_executable_result(exc: FileNotFoundError) -> "_SyntheticResult":
    if exc.filename is None:
        executable = "executable"
    elif isinstance(exc.filename, str) and exc.filename:
        executable = exc.filename
    else:
        raise TypeError("FileNotFoundError.filename must be a non-empty string or None")
    return _SyntheticResult(
        returncode=127,
        stdout="",
        stderr=f"dependency executable not found: {executable}",
    )


def _process_result(
    returncode: int,
    stdout_log: Path,
    stderr_log: Path,
) -> "_SyntheticResult":
    return _SyntheticResult(
        returncode=returncode,
        stdout=stdout_log.read_text(),
        stderr=stderr_log.read_text(),
    )


def _reproducer_commands(command_manifest: dict[str, Any]) -> dict[str, list[str]]:
    return {
        name: list(command["argv"])
        for name, command in command_manifest["commands"].items()
    }


def _production_trace_reproducer_commands(
    command_manifest: dict[str, Any],
) -> dict[str, list[str]]:
    return {
        stage: list(command_manifest["commands"][stage]["argv"])
        for stage in (
            "production_trace_simulation",
            "production_trace_report",
        )
    }


def _pd_reproducer_commands(command_manifest: dict[str, Any]) -> dict[str, list[str]]:
    reproducers = _reproducer_commands(command_manifest)
    return {
        name: reproducers[name]
        for name in ("pd_decode_serve", "pd_prefill_serve", "pd_proxy_serve")
        if name in reproducers
    }


class _SyntheticResult:
    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _mock_stages(command_manifest: dict[str, Any]) -> tuple[str, ...]:
    if command_manifest["deployment"] == "disaggregated":
        return (
            "vllm_benchmark",
            "disaggregated_midlayer_simulation",
        )
    if command_manifest["deployment"] == "colocated":
        return (
            "vllm_serve",
            "vllm_benchmark",
            "colocated_midlayer_simulation",
        )
    raise ManifestValidationError(
        f"unknown experiment deployment: {command_manifest['deployment']!r}")


def _dependency_category(stderr: str) -> str | None:
    text = stderr.lower()
    tokens = set(re.findall(r"[a-z0-9_+-]+", text))
    if not text:
        return None
    module_failure = (
        "modulenotfounderror" in tokens
        or "importerror" in tokens
        or "no module named" in text
        or "cannot import" in text
        or "missing package" in text
    )
    backend_failure = (
        "deepep" in tokens
        or "all-to-all" in tokens
        or "all2all" in tokens
        or ("backend" in tokens and ("unsupported" in tokens or "version" in tokens))
        or "not supported" in text
    )
    if module_failure:
        return "dependency"
    if backend_failure:
        return "backend"
    if (
        "model path" in text
        or "failed to load model" in text
        or "model not found" in text
        or "does not exist" in text
        or "no such file" in text
    ):
        return "model"
    if (
        "connection refused" in text
        or "failed to establish" in text
        or "readiness timeout" in text
        or "timed out" in text
        or "timeout" in tokens
    ):
        return "runtime"
    for category in ("nccl", "gdrcopy", "connector", "cuda", "dependency"):
        if category in tokens:
            return category
    if "dependencies" in tokens:
        return "dependency"
    if "port" in tokens or "eaddrinuse" in tokens or "address already in use" in text:
        return "port"
    return None


def _unblock_checklist(category: str) -> list[str]:
    category_specific = {
        "backend": "Verify the requested MoE/all-to-all backend is installed and supported by this vLLM build.",
        "model": "Verify the model path exists on this host and can be loaded by vLLM.",
        "runtime": "Verify the target HTTP endpoint becomes reachable before retrying the experiment.",
        "port": "Verify the configured HTTP/KV ports are free before retrying the experiment.",
        "dependency": "Verify the required runtime package is installed and importable.",
        "nccl": "Verify NCCL is installed and visible to the runtime.",
        "gdrcopy": "Verify GDRCopy is installed and visible to the runtime.",
        "connector": "Verify the requested KV connector is installed and configured.",
        "cuda": "Verify the CUDA driver and runtime are installed and compatible.",
    }
    if category not in category_specific:
        raise ValueError(f"unknown dependency category: {category!r}")
    return [
        category_specific[category],
        "Re-run the dry-run command manifest to confirm command arguments.",
        "Inspect the recorded stderr/stdout logs before retrying the experiment.",
    ]


def _write_placeholder_outputs(command_manifest: dict[str, Any]) -> None:
    if command_manifest["deployment"] == "disaggregated":
        _write_disaggregated_placeholder_outputs(command_manifest)
        return
    output_root = Path(command_manifest["output_root"])
    _write_json(output_root / "ground_truth.json", {
        "status": "placeholder",
        "source": "mock-run",
        "cell_id": command_manifest["cell_id"],
        "workload_hash": command_manifest["workload"]["hash"],
    })
    _write_json(output_root / "prediction.json", {
        "status": "placeholder",
        "source": "mock-run",
        "cell_id": command_manifest["cell_id"],
        "workload_hash": command_manifest["workload"]["hash"],
    })


def _write_disaggregated_placeholder_outputs(command_manifest: dict[str, Any]) -> None:
    output_root = Path(command_manifest["output_root"])
    _write_json(output_root / "ground_truth.json", {
        "status": "placeholder",
        "source": "mock-run",
        "cell_id": command_manifest["cell_id"],
        "workload_hash": command_manifest["workload"]["hash"],
        "comparison_role": "ground_truth",
    })


def _write_cell_manifest(command_manifest: dict[str, Any]) -> None:
    _write_json(
        Path(command_manifest["command_manifest_path"]), command_manifest)




def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _required_process_stream(result, stream: str) -> str:
    if stream == "stdout":
        value = result.stdout
    elif stream == "stderr":
        value = result.stderr
    else:
        raise ValueError(f"unknown process stream: {stream!r}")
    if not isinstance(value, str):
        raise TypeError(
            f"captured subprocess {stream} must be text, got "
            f"{type(value).__name__}")
    return value
    _write_json(output_root / "prediction.json", {
        "status": "placeholder",
        "source": "mock-run",
        "kind": "disaggregated_prediction",
        "cell_id": command_manifest["cell_id"],
        "workload_hash": command_manifest["workload"]["hash"],
        "inputs": {
            "prefill_trace": command_manifest["traces"]["prefill_trace"],
            "decode_trace": command_manifest["traces"]["decode_trace"],
            "prefill_sidecar": command_manifest["sidecars"]["prefill_sidecar"],
            "decode_sidecar": command_manifest["sidecars"]["decode_sidecar"],
        },
        "kv_connector": command_manifest["kv_connector"],
        "kv_transfer": {
            "status": "mock-placeholder",
            "metadata_source": "runner",
        },
        "metrics": [
            {
                "request_id": "mock-placeholder",
                "ttft_disagg_ms": None,
                "tpot_disagg_ms": None,
                "e2e_disagg_ms": None,
                "prefill_time_ms": None,
                "kv_transfer_time_ms": None,
                "decode_time_ms": None,
            }
        ],
    })
