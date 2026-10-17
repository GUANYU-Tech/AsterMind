"""Replay one canonical ShareGPT request schedule against a vLLM server."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Callable, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request

ROOT = Path(__file__).resolve().parents[3]

from evaluation.serving.experiments.experiment_plan import git_code_identity
from src.serving.request_gen import load_canonical_requests
from src.serving.request_gen.request_stream import (
    RequestSpec,
    request_stream_sha256,
)
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS


PROFILER_ENVIRONMENT_KEYS = (
    "CUDA_INJECTION64_PATH",
    "CUDA_TRACE_OUTPUT",
    "CUDA_TRACE_REQUIRE_PROFILER",
    "LD_PRELOAD",
    "NSYS_NVTX_PROFILER_REGISTER_ONLY",
    "VLLM_CUSTOM_SCOPES_FOR_PROFILING",
    "VLLM_NVTX_SCOPES_FOR_PROFILING",
    "VLLM_TORCH_PROFILER_DIR",
    "VLLM_TRACE_FUNCTION",
)

GROUND_TRUTH_SCHEDULER_POLICY = "fcfs"

EXECUTION_ENVIRONMENT_PREFIXES = (
    "ASCEND_",
    "ATB_",
    "CUDA_",
    "HCCL_",
    "MACA_",
    "NCCL_",
    "TASK_QUEUE_",
    "TORCH_",
    "VLLM_",
)
_ASCEND_STEP_TIMESTAMP_ENV = "MIDLAYER_910B_MSPROF_STEP_TIMESTAMP"
_ASCEND_MSPROFTX_STEP_MARK_ENV = "MIDLAYER_910B_MSPROFTX_STEP_MARK"
C500_DEVICE_EVENT_ENVIRONMENT_KEYS = (
    "MIDLAYER_C500_DEVICE_EVENT_JSONL",
    "MIDLAYER_C500_EVENT_POST_SOURCE_SHA256",
    "MIDLAYER_C500_EVENT_PATCH_SHA256",
    "MIDLAYER_C500_EVENT_SOURCE_PATH",
)
SUPPORTED_HARDWARE_IDS = EVALUATED_HARDWARE_IDS
# A hardware without its own canonical ShareGPT cases replays the identical
# request file and arrival schedule of a template hardware's canonical case.
# The template identity is recorded in the result; only the hardware and the
# runtime release change.
REQUEST_TEMPLATE_HARDWARE_BY_TARGET = {"nvidia_h200": "nvidia_a800"}
H200_RUNTIME_VERSIONS = ("0.11.0", "0.30.0")
_SERVER_TERMINATION_TIMEOUT_SECONDS = 30.0
_PROCESS_GROUP_POLL_INTERVAL_SECONDS = 0.05
_SCHEDULER_TRACE_BOOTSTRAP = (
    ROOT
    / "dataset_pipeline"
    / "serving_dataset"
    / "instrumentation_bootstrap"
)
_A800_COMPILATION_COUNTER_FIELDS = (
    "num_models_seen",
    "num_graphs_seen",
    "num_piecewise_graphs_seen",
    "num_piecewise_capturable_graphs_seen",
    "num_backend_compilations",
    "num_gpu_runner_capture_triggers",
    "num_cudagraph_captured",
    "num_inductor_compiles",
    "num_eager_compiles",
    "num_cache_entries_updated",
    "num_compiled_artifacts_saved",
    "dynamo_as_is_count",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _c500_device_event_environment(
    hardware_id: str, runtime: dict[str, Any], case: dict[str, Any]
) -> dict[str, str]:
    present = {
        key for key in C500_DEVICE_EVENT_ENVIRONMENT_KEYS if key in os.environ
    }
    if not present:
        return {}
    if present != set(C500_DEVICE_EVENT_ENVIRONMENT_KEYS):
        missing = sorted(set(C500_DEVICE_EVENT_ENVIRONMENT_KEYS) - present)
        raise ValueError(
            "C500 device Event environment is incomplete; missing "
            + ", ".join(missing)
        )
    if hardware_id != "metax_c500" or runtime["runtime_version"] != "0.24.0":
        raise ValueError("C500 device Event instrumentation requires MetaX vLLM 0.24.0")
    parallel_sizes = (
        runtime["tensor_parallel_size"],
        runtime["pipeline_parallel_size"],
        runtime["expert_parallel_size"],
    )
    if parallel_sizes != (1, 1, 1):
        raise ValueError("C500 device Event instrumentation requires TP1/PP1/EP1")

    environment = {
        key: os.environ[key] for key in C500_DEVICE_EVENT_ENVIRONMENT_KEYS
    }
    if any(not value for value in environment.values()):
        raise ValueError("C500 device Event environment values must be non-empty")
    for key in (
        "MIDLAYER_C500_EVENT_POST_SOURCE_SHA256",
        "MIDLAYER_C500_EVENT_PATCH_SHA256",
    ):
        digest = environment[key]
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError(f"{key} must be a lowercase SHA256 digest")

    output_path = Path(environment["MIDLAYER_C500_DEVICE_EVENT_JSONL"])
    if not output_path.is_absolute():
        raise ValueError("MIDLAYER_C500_DEVICE_EVENT_JSONL must be absolute")
    if output_path.exists():
        raise ValueError(f"refusing to overwrite device Event output: {output_path}")
    if not output_path.parent.is_dir():
        raise ValueError("C500 device Event output parent must already exist")

    source_path = Path(environment["MIDLAYER_C500_EVENT_SOURCE_PATH"])
    if not source_path.is_absolute() or not source_path.is_file():
        raise ValueError("MIDLAYER_C500_EVENT_SOURCE_PATH must be an existing file")
    source_path = source_path.resolve()
    expected_suffix = Path("vllm/v1/worker/gpu_model_runner.py")
    if Path(*source_path.parts[-4:]) != expected_suffix:
        raise ValueError("C500 device Event source path has an unexpected suffix")
    if sha256_file(source_path) != environment[
        "MIDLAYER_C500_EVENT_POST_SOURCE_SHA256"
    ]:
        raise ValueError("C500 device Event source SHA256 differs")
    environment["MIDLAYER_C500_EVENT_SOURCE_PATH"] = str(source_path)

    request_stream = load_canonical_requests(case["requests_path"])
    requests = request_stream.requests
    if len(requests) != case["requests_row_count"]:
        raise ValueError("requests.csv row count disagrees with case manifest")
    if any(request.output_tokens < 1 for request in requests):
        raise ValueError("ground-truth requests must generate at least one token")
    if any(request.prefix_hit_tokens != 0 for request in requests):
        raise ValueError(
            "C500 device Event instrumentation requires zero prefix-hit tokens"
        )
    expected_scheduled_tokens = sum(
        request.input_tokens + request.output_tokens - 1
        for request in requests
    )
    environment[
        "MIDLAYER_C500_DEVICE_EVENT_EXPECTED_SCHEDULED_TOKENS"
    ] = str(expected_scheduled_tokens)

    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        raise ValueError("C500 device Event instrumentation requires PYTHONPATH")
    python_paths = tuple(
        Path(entry).resolve()
        for entry in os.environ["PYTHONPATH"].split(os.pathsep)
        if entry
    )
    overlay_root = source_path.parents[3]
    if overlay_root not in python_paths:
        raise ValueError("PYTHONPATH does not select the C500 device Event overlay")
    environment["PYTHONPATH"] = os.environ["PYTHONPATH"]
    return environment


def _installed_vllm_release() -> str:
    from importlib.metadata import version

    return version("vllm").split("+", 1)[0]


def _vllm_source_identity(release: str) -> str:
    """Identify the installed vLLM source the same way canonical A800 cases do.

    Canonical runtime.source_commit values have the form
    ``vllm=<release>;core=<sha256[:8]>...<sha256[-6:]>;executor=<...>`` over
    ``vllm/v1/engine/core.py`` and ``vllm/v1/executor/multiproc_executor.py``.
    """

    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        raise RuntimeError("installed vLLM package cannot be located")
    package = Path(spec.origin).resolve().parent

    def digest(relative: str) -> str:
        value = sha256_file(package / relative)
        return f"{value[:8]}...{value[-6:]}"

    return (
        f"vllm={release};core={digest('v1/engine/core.py')};"
        f"executor={digest('v1/executor/multiproc_executor.py')}"
    )


def _retarget_case_contract(
    case: dict[str, Any],
    *,
    target_hardware_id: str,
    target_runtime_version: str,
) -> dict[str, Any]:
    """Relabel a template hardware's canonical case for another hardware."""

    if target_hardware_id not in REQUEST_TEMPLATE_HARDWARE_BY_TARGET:
        raise ValueError(
            f"unsupported request-template target hardware: {target_hardware_id!r}"
        )
    template_hardware_id = REQUEST_TEMPLATE_HARDWARE_BY_TARGET[target_hardware_id]
    if case["request_template"] is not None:
        raise ValueError("case contract was already retargeted")
    if case["identity"]["hardware_id"] != template_hardware_id:
        raise ValueError(
            f"{target_hardware_id} replays {template_hardware_id} cases only, got "
            f"{case['identity']['hardware_id']!r}"
        )
    if target_runtime_version not in H200_RUNTIME_VERSIONS:
        raise ValueError(
            f"unsupported H200 runtime version: {target_runtime_version!r}")
    installed = _installed_vllm_release()
    if installed != target_runtime_version:
        raise RuntimeError(
            "requested runtime version disagrees with the installed vLLM: "
            f"{target_runtime_version} != {installed}"
        )
    template_runtime = case["runtime"]
    identity = dict(case["identity"])
    identity["hardware_id"] = target_hardware_id
    runtime = {
        "dtype": template_runtime["dtype"],
        "tensor_parallel_size": template_runtime["tensor_parallel_size"],
        "pipeline_parallel_size": template_runtime["pipeline_parallel_size"],
        "expert_parallel_size": template_runtime["expert_parallel_size"],
        "target_qps": template_runtime["target_qps"],
        "runtime_version": target_runtime_version,
        "source_commit": _vllm_source_identity(installed),
    }
    return {
        **case,
        "identity": identity,
        "runtime": runtime,
        "request_template": {
            "hardware_id": template_hardware_id,
            "case_manifest_path": case["manifest_path"],
            "case_manifest_sha256": case["manifest_sha256"],
            "requests_sha256": case["requests_sha256"],
        },
    }


# The dtypes the production ground-truth runner accepts per hardware: the
# MetaX capture retains a float16 cell as well as bfloat16, the others
# bfloat16 only.
_PRODUCTION_DTYPES = {
    "nvidia_a800": ("bfloat16",),
    "nvidia_h200": ("bfloat16",),
    "metax_c500": ("bfloat16", "float16"),
    "ascend_910b": ("bfloat16",),
}


def _load_case_contract(
    manifest_path: str | Path,
    *,
    model_path: str | Path,
    suite_manifest_path: str | Path,
) -> dict[str, Any]:
    path = Path(manifest_path).resolve(strict=True)
    raw = json.loads(path.read_text())
    identity = raw["identity"]
    requests = raw["requests"]
    runtime = raw["runtime"]
    if identity["source_kind"] != "sharegpt":
        raise ValueError("ground-truth runner requires a ShareGPT case")
    if identity["hardware_id"] not in SUPPORTED_HARDWARE_IDS:
        raise ValueError(
            "unsupported production ground-truth hardware_id: "
            f"{identity['hardware_id']!r}"
        )
    suite_path = Path(suite_manifest_path).resolve(strict=True)
    suite = json.loads(suite_path.read_text())
    dataset_root = (ROOT / suite["dataset_root"]).resolve(strict=True)
    expected_case_path = (
        dataset_root
        / identity["hardware_id"]
        / identity["model_id"]
        / identity["case_id"]
        / "manifest.json"
    ).resolve(strict=True)
    if path != expected_case_path:
        raise ValueError(
            "case manifest path disagrees with suite identity: "
            f"{path} != {expected_case_path}"
        )
    matching_cells = [
        cell
        for cell in suite["cells"]
        if cell["hardware_id"] == identity["hardware_id"]
        and cell["model_id"] == identity["model_id"]
        and cell["case_id"] == identity["case_id"]
    ]
    if len(matching_cells) != 1:
        raise ValueError(
            "suite must contain exactly one matching hardware/model/case cell"
        )
    cell = matching_cells[0]
    manifest_sha256 = sha256_file(path)
    if manifest_sha256 != cell["manifest_sha256"]:
        raise ValueError("case manifest SHA256 disagrees with suite manifest")
    requests_path = (path.parent / requests["path"]).resolve(strict=True)
    requests_sha256 = sha256_file(requests_path)
    if requests_sha256 != requests["sha256"]:
        raise ValueError("requests.csv SHA256 disagrees with case manifest")
    if requests_sha256 != cell["requests_sha256"]:
        raise ValueError("requests.csv SHA256 disagrees with suite manifest")
    matching_models = [
        model
        for model in suite["models"]
        if model["dataset_model_id"] == identity["model_id"]
    ]
    if len(matching_models) != 1:
        raise ValueError("suite must contain exactly one matching model identity")
    model_directory = Path(model_path).resolve(strict=True)
    model_config_path = (model_directory / "config.json").resolve(strict=True)
    model_config_sha256 = sha256_file(model_config_path)
    if model_config_sha256 != matching_models[0]["model_config_sha256"]:
        raise ValueError("local model config SHA256 disagrees with suite manifest")
    tensor_parallel_size = runtime["tensor_parallel_size"]
    if tensor_parallel_size not in (1, 2):
        raise ValueError("production ground-truth runner requires TP=1 or TP=2")
    if runtime["pipeline_parallel_size"] != 1:
        raise ValueError("production ground-truth runner requires PP=1")
    expert_parallel_size = runtime["expert_parallel_size"]
    if expert_parallel_size not in (1, 2):
        raise ValueError("production ground-truth runner requires EP=1 or EP=2")
    if expert_parallel_size == 2 and tensor_parallel_size != 2:
        raise ValueError("production ground-truth EP=2 requires TP=2")
    if identity["hardware_id"] not in _PRODUCTION_DTYPES:
        raise ValueError(
            f"unsupported production hardware: {identity['hardware_id']!r}"
        )
    if runtime["dtype"] not in _PRODUCTION_DTYPES[identity["hardware_id"]]:
        raise ValueError(
            "production ground-truth runner received an unsupported dtype"
        )
    target_qps = runtime["target_qps"]
    if (
        isinstance(target_qps, bool)
        or not isinstance(target_qps, (int, float))
        or not math.isfinite(target_qps)
        or target_qps <= 0
    ):
        raise ValueError("case manifest target_qps must be finite and positive")
    return {
        "manifest_path": str(path),
        "manifest_sha256": manifest_sha256,
        "suite_manifest_path": str(suite_path),
        "suite_manifest_sha256": sha256_file(suite_path),
        "identity": dict(identity),
        "runtime": dict(runtime),
        "model_config_path": str(model_config_path),
        "model_config_sha256": model_config_sha256,
        "requests_path": str(requests_path),
        "requests_sha256": requests_sha256,
        "requests_row_count": requests["row_count"],
        "request_template": None,
    }


# Whether the backend can offload weights above the GPU: the CUDA backends
# have no CPU-offload flag, so their plan passes 0; the MACA and CANN backends
# take no such argument at all, so their plan passes None.
_CPU_OFFLOAD_GB = {
    "nvidia_a800": 0,
    "nvidia_h200": 0,
    "metax_c500": None,
    "ascend_910b": None,
}


# The launch record each backend needs.  The stored key is the *device
# visibility* environment variable for that backend; the rest is the argv and
# resolved plan fields the hand-written branches set.
#
# ``by_runtime_version`` overrides fields for a backend that ships more than
# one release (H200's 0.11.0 vs 0.30.0 renamed the compilation "level" key
# and added --cudagraph-metrics; MetaX's 0.8.5 is the v0 engine).  A backend
# with ``by_runtime_version`` set requires its runtime version to be listed.
_LAUNCH_RECORD_DEFAULTS = {
    "explicit_non_eager_argv": ["--no-enforce-eager"],
    "model_load_argv": [],
    "safetensors_load_strategy": None,
    "safetensors_prefetch_num_threads": None,
    "cpu_offload_argv": [],
}

_SERVER_LAUNCH_RECORDS = {
    "nvidia_a800": {
        "visibility_environment_key": "CUDA_VISIBLE_DEVICES",
        "engine_environment": {"VLLM_USE_V1": "1"},
        "compilation_argv": [
            "--compilation-config",
            json.dumps(
                {"level": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"},
                separators=(",", ":"),
            ),
        ],
        "engine_generation": "v1",
        "compilation_level": 3,
        "graph_mode": "full_and_piecewise_cuda_graph",
        "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
        "async_scheduling": True,
        "scheduling_argv": ["--async-scheduling"],
        "cpu_offload_argv": ["--cpu-offload-gb", "0"],
        "by_runtime_version": None,
        "unsupported_runtime_message": None,
    },
    "nvidia_h200": {
        "visibility_environment_key": "CUDA_VISIBLE_DEVICES",
        # Only the 0.11.0 release sets VLLM_USE_V1; 0.30.0 does not, so this
        # is empty here and the 0.11.0 override below adds it.
        "engine_environment": {},
        "compilation_argv": None,
        "engine_generation": "v1",
        "compilation_level": 3,
        "graph_mode": "full_and_piecewise_cuda_graph",
        "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
        "async_scheduling": True,
        "scheduling_argv": None,
        "cpu_offload_argv": ["--cpu-offload-gb", "0"],
        "by_runtime_version": {
            # Identical to the canonical A800 0.11.0 production contract.
            "0.11.0": {
                "engine_environment": {"VLLM_USE_V1": "1"},
                "compilation_argv": [
                    "--compilation-config",
                    json.dumps(
                        {"level": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"},
                        separators=(",", ":"),
                    ),
                ],
                "scheduling_argv": ["--async-scheduling"],
            },
            # 0.30.0 renamed compilation "level" to "mode" and reports
            # per-step CUDA Graph statistics only with --cudagraph-metrics.
            "0.30.0": {
                "compilation_argv": [
                    "--compilation-config",
                    json.dumps(
                        {"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"},
                        separators=(",", ":"),
                    ),
                ],
                "scheduling_argv": [
                    "--async-scheduling", "--cudagraph-metrics"],
            },
        },
        "unsupported_runtime_message": (
            "unsupported H200 runtime_version: {runtime_version!r}"),
    },
    "metax_c500": {
        "visibility_environment_key": "MACA_VISIBLE_DEVICES",
        "engine_environment": {},
        "compilation_argv": None,
        "engine_generation": None,
        "compilation_level": None,
        "graph_mode": None,
        "configured_cudagraph_mode": None,
        "async_scheduling": None,
        "scheduling_argv": None,
        "cpu_offload_argv": None,
        "by_runtime_version": {
            "0.8.5": {
                "engine_environment": {"VLLM_USE_V1": "0"},
                "explicit_non_eager_argv": [],
                "compilation_argv": [],
                "cpu_offload_argv": [],
                "engine_generation": "v0",
                "compilation_level": 0,
                "graph_mode": "cuda_graph_hybrid",
                "configured_cudagraph_mode": None,
                "async_scheduling": False,
                "scheduling_argv": [],
            },
            "0.24.0": {
                "engine_environment": {
                    "MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE": "0",
                    "USE_VLLM_TRITON_EXPERT": "0",
                },
                "compilation_argv": [
                    "--compilation-config",
                    json.dumps({"mode": 3}, separators=(",", ":")),
                ],
                "model_load_argv": [
                    "--safetensors-load-strategy",
                    "prefetch",
                    "--safetensors-prefetch-num-threads",
                    "1",
                ],
                "safetensors_load_strategy": "prefetch",
                "safetensors_prefetch_num_threads": 1,
                "engine_generation": "v1",
                "compilation_level": 3,
                "graph_mode": "piecewise_cuda_graph",
                "configured_cudagraph_mode": "FULL_AND_PIECEWISE",
                "async_scheduling": True,
                "scheduling_argv": [
                    "--async-scheduling", "--cudagraph-metrics"],
                "cpu_offload_argv": [],
            },
        },
        "unsupported_runtime_message": (
            "unsupported MetaX production ground-truth runtime_version: "
            "{runtime_version!r}"),
    },
    "ascend_910b": {
        "visibility_environment_key": "ASCEND_RT_VISIBLE_DEVICES",
        "engine_environment": {"VLLM_USE_V1": "1"},
        "compilation_argv": [
            "--compilation-config",
            json.dumps(
                {
                    "cudagraph_mode": "FULL",
                    "max_cudagraph_capture_size": 512,
                },
                separators=(",", ":"),
            ),
        ],
        "engine_generation": "v1",
        "compilation_level": None,
        "graph_mode": "acl_graph_full",
        "configured_cudagraph_mode": "FULL",
        "async_scheduling": True,
        "scheduling_argv": ["--async-scheduling", "--cudagraph-metrics"],
        "cpu_offload_argv": [],
        "by_runtime_version": None,
        "unsupported_runtime_message": None,
    },
}


def _launch_record(hardware_id: str, runtime_version: str) -> dict[str, Any]:
    """Return the resolved launch fields ``hardware_id`` needs at its release."""

    if hardware_id not in _SERVER_LAUNCH_RECORDS:
        raise ValueError(
            f"unsupported server hardware_id: {hardware_id!r}")
    record = _SERVER_LAUNCH_RECORDS[hardware_id]
    resolved = dict(_LAUNCH_RECORD_DEFAULTS)
    for key, value in record.items():
        if key not in ("by_runtime_version", "unsupported_runtime_message"):
            resolved[key] = value
    by_runtime_version = record["by_runtime_version"]
    if by_runtime_version is not None:
        if runtime_version not in by_runtime_version:
            raise ValueError(record["unsupported_runtime_message"].format(
                runtime_version=runtime_version))
        resolved.update(by_runtime_version[runtime_version])
    return resolved


def _server_plan(args: argparse.Namespace, case: dict[str, Any]) -> dict[str, Any]:
    runtime = case["runtime"]
    hardware_id = case["identity"]["hardware_id"]
    devices = _validate_devices(args.devices, runtime)
    visible_devices = ",".join(str(device) for device in devices)
    environment = {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key.startswith(EXECUTION_ENVIRONMENT_PREFIXES)
        and key not in {"CUDA_VISIBLE_DEVICES", "MACA_VISIBLE_DEVICES"}
    }
    ascend_step_boundary_names = (
        _ASCEND_STEP_TIMESTAMP_ENV,
        _ASCEND_MSPROFTX_STEP_MARK_ENV,
    )
    enabled_ascend_boundaries = [
        name for name in ascend_step_boundary_names if name in os.environ
    ]
    if len(enabled_ascend_boundaries) > 1:
        raise ValueError(
            "910B profiler step boundary mechanisms are mutually exclusive"
        )
    for name in enabled_ascend_boundaries:
        if hardware_id != "ascend_910b":
            raise ValueError(f"{name} is supported only on ascend_910b")
        if os.environ[name] != "1":
            raise ValueError(f"{name} must be exactly '1'")
        environment[name] = "1"
    launch_record = _launch_record(hardware_id, runtime["runtime_version"])
    environment[launch_record["visibility_environment_key"]] = visible_devices
    environment.update(launch_record["engine_environment"])
    explicit_non_eager_argv = launch_record["explicit_non_eager_argv"]
    compilation_argv = launch_record["compilation_argv"]
    model_load_argv = launch_record["model_load_argv"]
    safetensors_load_strategy = launch_record["safetensors_load_strategy"]
    safetensors_prefetch_num_threads = launch_record[
        "safetensors_prefetch_num_threads"]
    engine_generation = launch_record["engine_generation"]
    compilation_level = launch_record["compilation_level"]
    graph_mode = launch_record["graph_mode"]
    configured_cudagraph_mode = launch_record["configured_cudagraph_mode"]
    async_scheduling = launch_record["async_scheduling"]
    scheduling_argv = launch_record["scheduling_argv"]
    cpu_offload_argv = launch_record["cpu_offload_argv"]
    device_event_environment = _c500_device_event_environment(
        hardware_id, runtime, case
    )
    environment.update(device_event_environment)
    shutdown_timeout_seconds = 30 if device_event_environment else 0
    shutdown_argv = (
        ["--shutdown-timeout", str(shutdown_timeout_seconds)]
        if shutdown_timeout_seconds
        else []
    )
    environment["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    scheduler_trace_output = None
    scheduler_request_add_output = None
    counter_only_output = None
    if (
        "scheduler_trace_output" in vars(args)
        and args.scheduler_trace_output is not None
    ):
        if engine_generation != "v1":
            raise ValueError(
                "Scheduler trace instrumentation currently requires vLLM V1"
            )
        scheduler_trace_output = Path(args.scheduler_trace_output).resolve()
        if scheduler_trace_output.exists():
            raise ValueError(
                "refusing to overwrite Scheduler trace output: "
                f"{scheduler_trace_output}"
            )
        if _uses_worker_dispatch_trace(hardware_id, runtime):
            for rank in range(runtime["tensor_parallel_size"]):
                for path in (
                    Path(f"{scheduler_trace_output}.rank-{rank}.jsonl"),
                    Path(
                        f"{scheduler_trace_output}.compilation-counter.rank-{rank}.jsonl"
                    ),
                ):
                    if path.exists():
                        raise ValueError(
                            "refusing to overwrite A800 Scheduler trace sidecar: "
                            f"{path}"
                        )
    if (
        "scheduler_request_add_output" in vars(args)
        and args.scheduler_request_add_output is not None
    ):
        if scheduler_trace_output is None:
            raise ValueError(
                "Scheduler request-add timing requires Scheduler tracing"
            )
        scheduler_request_add_output = Path(
            args.scheduler_request_add_output
        ).resolve()
        if scheduler_request_add_output.exists():
            raise ValueError(
                "refusing to overwrite Scheduler request-add output: "
                f"{scheduler_request_add_output}"
            )
        stage_sidecars = tuple(
            scheduler_request_add_output.parent.glob(
                f"{scheduler_request_add_output.name}.stages.*.jsonl"
            )
        )
        if stage_sidecars:
            raise ValueError(
                "refusing to reuse Scheduler request-stage sidecars: "
                + ", ".join(str(path) for path in stage_sidecars)
            )
    if (
        "a800_compilation_counter_output" in vars(args)
        and args.a800_compilation_counter_output is not None
    ):
        if hardware_id != "nvidia_a800":
            raise ValueError(
                "A800 compilation-counter instrumentation requires nvidia_a800"
            )
        if scheduler_trace_output is not None:
            raise ValueError(
                "A800 counter-only instrumentation cannot accompany Scheduler trace"
            )
        counter_only_output = Path(
            args.a800_compilation_counter_output
        ).resolve()
        if counter_only_output.exists():
            raise ValueError(
                "refusing to overwrite A800 counter-only output: "
                f"{counter_only_output}"
            )
        for rank in range(runtime["tensor_parallel_size"]):
            path = Path(
                f"{counter_only_output}.compilation-counter.rank-{rank}.jsonl"
            )
            if path.exists():
                raise ValueError(
                    "refusing to overwrite A800 counter-only sidecar: "
                    f"{path}"
                )
    instrumentation_output = (
        scheduler_trace_output
        if scheduler_trace_output is not None
        else counter_only_output
    )
    if instrumentation_output is not None:
        python_paths = [str(_SCHEDULER_TRACE_BOOTSTRAP), str(ROOT)]
        if "PYTHONPATH" in os.environ and os.environ["PYTHONPATH"]:
            python_paths.append(os.environ["PYTHONPATH"])
        environment["PYTHONPATH"] = os.pathsep.join(python_paths)
        environment["MIDLAYER_SCHEDULER_TRACE_SPAWN_WRAPPER"] = str(
            _SCHEDULER_TRACE_BOOTSTRAP / "spawn_with_scheduler_trace.sh"
        )
        environment["MIDLAYER_SCHEDULER_TRACE_SPAWN_PYTHON"] = sys.executable
        environment["MIDLAYER_SCHEDULER_TRACE_SPAWN_PYTHONPATH"] = os.pathsep.join(
            (str(_SCHEDULER_TRACE_BOOTSTRAP), str(ROOT))
        )
        if scheduler_trace_output is not None:
            source_identity = runtime["source_commit"]
            if source_identity is None:
                source_identity = git_code_identity(ROOT)["git_commit"]
            if not isinstance(source_identity, str) or not source_identity:
                raise ValueError(
                    "Scheduler trace requires a non-empty source identity")
            environment["MIDLAYER_SCHEDULER_TRACE_OUTPUT"] = str(
                scheduler_trace_output
            )
            environment["MIDLAYER_SCHEDULER_TRACE_BOOTSTRAP_MARKER"] = str(
                Path(f"{scheduler_trace_output}.bootstrap.jsonl")
            )
            environment["MIDLAYER_SCHEDULER_TRACE_REQUEST_COUNT"] = str(
                case["requests_row_count"]
            )
            environment["MIDLAYER_SCHEDULER_TRACE_SOURCE_IDENTITY"] = (
                source_identity
            )
            if scheduler_request_add_output is not None:
                environment["MIDLAYER_SCHEDULER_REQUEST_ADD_OUTPUT"] = str(
                    scheduler_request_add_output
                )
            if configured_cudagraph_mode is not None:
                environment[
                    "MIDLAYER_SCHEDULER_TRACE_CONFIGURED_CUDAGRAPH_MODE"
                ] = configured_cudagraph_mode
                environment["MIDLAYER_SCHEDULER_TRACE_MAX_IN_FLIGHT"] = str(
                    2 if async_scheduling else 1
                )
        else:
            environment["MIDLAYER_A800_COUNTER_ONLY_OUTPUT"] = str(
                counter_only_output
            )
        # All V1 Scheduler trace installations wrap the executor whenever a
        # configured Graph mode is present.  The bootstrap must therefore know
        # which platform-specific UniProcExecutor implementation to import;
        # this is not an A800-only contract.
        environment["MIDLAYER_SCHEDULER_TRACE_HARDWARE_ID"] = hardware_id
        environment["MIDLAYER_SCHEDULER_TRACE_WORKER_RANK_COUNT"] = str(
            runtime["tensor_parallel_size"]
        )
    if runtime["expert_parallel_size"] == 1:
        expert_parallel_argv = []
    elif runtime["expert_parallel_size"] == 2:
        expert_parallel_argv = ["--enable-expert-parallel"]
    else:
        raise AssertionError("case contract allowed an unsupported EP size")
    argv = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        str(Path(args.model).resolve(strict=True)),
        "--served-model-name",
        case["identity"]["model_id"],
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--dtype",
        runtime["dtype"],
        "--tensor-parallel-size",
        str(runtime["tensor_parallel_size"]),
        "--pipeline-parallel-size",
        str(runtime["pipeline_parallel_size"]),
        *expert_parallel_argv,
        *model_load_argv,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--scheduling-policy",
        GROUND_TRUTH_SCHEDULER_POLICY,
        *scheduling_argv,
        "--enable-prefix-caching",
        *cpu_offload_argv,
        "--block-size",
        str(args.kv_block_size_tokens),
        *explicit_non_eager_argv,
        "--seed",
        str(args.seed),
        *shutdown_argv,
        *compilation_argv,
    ]
    if hardware_id not in _CPU_OFFLOAD_GB:
        raise ValueError(f"unsupported server hardware_id: {hardware_id!r}")
    cpu_offload_gb = _CPU_OFFLOAD_GB[hardware_id]
    if scheduler_trace_output is None:
        scheduler_instrumentation_version = None
    elif _uses_worker_dispatch_trace(hardware_id, runtime):
        scheduler_instrumentation_version = 4
    elif configured_cudagraph_mode is not None:
        scheduler_instrumentation_version = 2
    elif hardware_id in ("metax_c500", "ascend_910b"):
        scheduler_instrumentation_version = 1
    else:
        raise ValueError(f"unsupported scheduler trace hardware: {hardware_id!r}")
    diagnostic_nsys = (
        os.environ["MIDLAYER_A800_NSYS_STEP_NVTX"]
        if "MIDLAYER_A800_NSYS_STEP_NVTX" in os.environ else None)
    if diagnostic_nsys == "1":
        if hardware_id != "nvidia_a800":
            raise ValueError("A800 Nsight step diagnostics require nvidia_a800")
        output_text = (
            os.environ["MIDLAYER_A800_NSYS_OUTPUT"]
            if "MIDLAYER_A800_NSYS_OUTPUT" in os.environ else None)
        if output_text is None:
            raise RuntimeError(
                "MIDLAYER_A800_NSYS_OUTPUT is required for A800 Nsight diagnostics"
            )
        output_path = Path(output_text)
        if not output_path.is_absolute():
            raise ValueError("A800 Nsight output path must be absolute")
        output_path = output_path.resolve()
        if any(path.exists() for path in (
                output_path.with_suffix(".nsys-rep"),
                output_path.with_suffix(".sqlite"))):
            raise FileExistsError("refusing to overwrite A800 Nsight diagnostic output")
        nsys_binary = (
            os.environ["NSYS_BIN"] if "NSYS_BIN" in os.environ else "nsys")
        nsys = shutil.which(nsys_binary)
        if nsys is None:
            raise FileNotFoundError("A800 Nsight diagnostic requires nsys")
        argv = [
            nsys, "profile", "--force-overwrite=true", "--trace=cuda,nvtx,osrt",
            "--cuda-graph-trace=node", "--trace-fork-before-exec=true",
            "--sample=none", "--output", str(output_path), *argv,
        ]
    return {
        "argv": argv,
        "python_executable": sys.executable,
        "environment": environment,
        "hardware_id": hardware_id,
        "devices": list(devices),
        "endpoint": {
            "base_url": f"http://127.0.0.1:{args.port}",
            "health_url": f"http://127.0.0.1:{args.port}/v1/models",
        },
        "engine": {
            "dtype": runtime["dtype"],
            "tensor_parallel_size": runtime["tensor_parallel_size"],
            "pipeline_parallel_size": runtime["pipeline_parallel_size"],
            "expert_parallel_size": runtime["expert_parallel_size"],
            "safetensors_load_strategy": safetensors_load_strategy,
            "safetensors_prefetch_num_threads": safetensors_prefetch_num_threads,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "scheduler_policy": GROUND_TRUTH_SCHEDULER_POLICY,
            "enable_prefix_caching": True,
            "cpu_offload_gb": cpu_offload_gb,
            "kv_block_size_tokens": args.kv_block_size_tokens,
            "enforce_eager": False,
            "vllm_engine_generation": engine_generation,
            "runtime_version": runtime["runtime_version"],
            "compilation_level": compilation_level,
            "graph_mode": graph_mode,
            "configured_cudagraph_mode": configured_cudagraph_mode,
            "cudagraph_metrics": "--cudagraph-metrics" in scheduling_argv,
            "async_scheduling": async_scheduling,
            # This is the source-defined executor setting selected by the
            # explicit ``--async-scheduling`` argv above.  Accepted capture
            # evidence still reads the live Executor property separately;
            # this field only makes the planned depth contract explicit.
            "max_concurrent_batches": 2 if async_scheduling else 1,
            "seed": args.seed,
            "shutdown_timeout_seconds": shutdown_timeout_seconds,
        },
        "profiler": {
            "requested": diagnostic_nsys == "1",
            "forbidden_environment_keys": list(PROFILER_ENVIRONMENT_KEYS),
        },
        "scheduler_trace": {
            "enabled": scheduler_trace_output is not None,
            "output_path": (
                None
                if scheduler_trace_output is None
                else str(scheduler_trace_output)
            ),
            "instrumentation_id": (
                None
                if scheduler_trace_output is None
                else "midlayer-vllm-scheduler-trace"
            ),
            "instrumentation_version": scheduler_instrumentation_version,
            "buffering": (
                None
                if scheduler_trace_output is None
                else "in_memory_append_then_single_write"
            ),
            "worker_rank_count": (
                None
                if scheduler_trace_output is None
                or not _uses_worker_dispatch_trace(hardware_id, runtime)
                else runtime["tensor_parallel_size"]
            ),
            "request_add_output_path": (
                None
                if scheduler_request_add_output is None
                else str(scheduler_request_add_output)
            ),
            "request_stage_output_pattern": (
                None
                if scheduler_request_add_output is None
                else f"{scheduler_request_add_output}.stages.*.jsonl"
            ),
        },
        "compilation_counter_instrumentation": {
            "enabled": instrumentation_output is not None
            and hardware_id == "nvidia_a800",
            "mode": (
                None
                if instrumentation_output is None
                else (
                    "scheduler_trace" if scheduler_trace_output is not None
                    else "counter_only"
                )
            ),
            "output_path": (
                None
                if instrumentation_output is None
                else str(instrumentation_output)
            ),
            "worker_rank_count": (
                None
                if instrumentation_output is None
                or hardware_id != "nvidia_a800"
                else runtime["tensor_parallel_size"]
            ),
        },
        "max_dispatch_lag_ms": args.max_dispatch_lag_ms,
        "arrival_probe_per_1000_tokens_ms": (
            args.arrival_probe_per_1000_tokens_ms),
    }


def _uses_worker_dispatch_trace(
    hardware_id: str, runtime: dict[str, Any]
) -> bool:
    """Whether the Scheduler trace observes graphs through worker dispatch.

    Mirrors ``vllm_scheduler_trace._uses_worker_dispatch_observations``: the
    vLLM 0.11.0 runtimes (A800, H200 0.11.0) have no per-step graph statistics.
    """

    if hardware_id == "nvidia_a800":
        return True
    if hardware_id == "nvidia_h200":
        return runtime["runtime_version"] == "0.11.0"
    if hardware_id in ("metax_c500", "ascend_910b"):
        return False
    raise ValueError(f"unsupported Scheduler trace hardware: {hardware_id!r}")


def _validate_devices(
    devices: Sequence[int], runtime: dict[str, Any]
) -> tuple[int, ...]:
    if not isinstance(devices, Sequence) or isinstance(devices, (str, bytes)):
        raise TypeError("devices must be a sequence of integer device indices")
    normalized = tuple(devices)
    if not normalized:
        raise ValueError("at least one device must be selected")
    if any(isinstance(device, bool) or not isinstance(device, int) for device in normalized):
        raise TypeError("device indices must be integers")
    if any(device < 0 for device in normalized):
        raise ValueError("device indices must be non-negative")
    if len(set(normalized)) != len(normalized):
        raise ValueError("device indices must be unique")
    expected = (
        runtime["tensor_parallel_size"] * runtime["pipeline_parallel_size"]
    )
    if len(normalized) != expected:
        raise ValueError(
            "device count must equal tensor_parallel_size * "
            f"pipeline_parallel_size: expected {expected}, got {len(normalized)}"
        )
    return normalized


def _validate_no_profiler_environment() -> None:
    diagnostic_nsys = (
        os.environ["MIDLAYER_A800_NSYS_STEP_NVTX"]
        if "MIDLAYER_A800_NSYS_STEP_NVTX" in os.environ else None)
    if diagnostic_nsys not in (None, "1"):
        raise ValueError(
            "MIDLAYER_A800_NSYS_STEP_NVTX must be exactly '1' when set"
        )
    nsys_injection_keys = {"CUDA_INJECTION64_PATH", "LD_PRELOAD"}
    c500_graph_launch_context = (
        os.environ["MIDLAYER_C500_GRAPH_LAUNCH_CONTEXT"]
        if "MIDLAYER_C500_GRAPH_LAUNCH_CONTEXT" in os.environ else None
    )
    c500_graph_launch_log = (
        os.environ["MIDLAYER_C500_GRAPH_LAUNCH_LOG"]
        if "MIDLAYER_C500_GRAPH_LAUNCH_LOG" in os.environ else None
    )
    c500_graph_launch_library = (
        os.environ["MIDLAYER_C500_GRAPH_LAUNCH_INTERPOSER"]
        if "MIDLAYER_C500_GRAPH_LAUNCH_INTERPOSER" in os.environ else None
    )
    c500_graph_launch_injection = (
        c500_graph_launch_context is not None
        or c500_graph_launch_log is not None
        or c500_graph_launch_library is not None
    )
    if c500_graph_launch_injection:
        if (
            not c500_graph_launch_context
            or not c500_graph_launch_log
            or not c500_graph_launch_library
        ):
            raise ValueError("C500 Graph launch diagnostic environment is incomplete")
        if not all(
            value.startswith("/")
            for value in (
                c500_graph_launch_context,
                c500_graph_launch_log,
                c500_graph_launch_library,
            )
        ):
            raise ValueError("C500 Graph launch diagnostic paths must be absolute")
        if ("LD_PRELOAD" not in os.environ
                or os.environ["LD_PRELOAD"] != c500_graph_launch_library):
            raise ValueError(
                "C500 Graph launch diagnostic LD_PRELOAD must name its declared interposer"
            )
    active = [
        key
        for key in PROFILER_ENVIRONMENT_KEYS
        if key in os.environ and os.environ[key]
        and not (diagnostic_nsys == "1" and key in nsys_injection_keys)
        and not (c500_graph_launch_injection and key == "LD_PRELOAD")
    ]
    if active:
        raise RuntimeError(
            "production ground-truth run refuses profiler/injection "
            f"environment variables: {active}"
        )


def _assert_port_available(host: str, port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as error:
            raise RuntimeError(
                f"refusing to reuse occupied server endpoint {host}:{port}"
            ) from error


def _wait_for_server(
    process: subprocess.Popen,
    health_url: str,
    *,
    expected_model_id: str,
    timeout_seconds: float,
    poll_interval_seconds: float = 2.0,
) -> dict[str, Any]:
    started = time.monotonic()
    attempts = 0
    last_probe_error = "server has not accepted a readiness connection"
    while True:
        attempts += 1
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"vLLM server exited before readiness with code {returncode}"
            )
        try:
            with urllib_request.urlopen(health_url, timeout=2.0) as response:
                if 200 <= response.status < 300:
                    payload = json.loads(response.read())
                    model_ids = [entry["id"] for entry in payload["data"]]
                    if model_ids == [expected_model_id]:
                        if process.poll() is not None:
                            raise RuntimeError(
                                "owned vLLM server exited during readiness"
                            )
                        return {
                            "attempts": attempts,
                            "elapsed_s": time.monotonic() - started,
                            "model_ids": model_ids,
                        }
                    last_probe_error = (
                        "server model IDs disagree: "
                        f"{model_ids!r} != {[expected_model_id]!r}"
                    )
        except (OSError, urllib_error.URLError, json.JSONDecodeError) as error:
            last_probe_error = f"{type(error).__name__}: {error}"
        elapsed = time.monotonic() - started
        if elapsed >= timeout_seconds:
            raise RuntimeError(
                f"vLLM server readiness timed out after {elapsed:.1f}s; "
                f"last probe error: {last_probe_error}"
            )
        time.sleep(poll_interval_seconds)


# The markers each backend's vLLM release prints when it really captured the
# production CUDA/ACL graph.  Selection is data.
#
# ``single_runtime`` is the marker set for a backend with one release, which
# is selected without consulting the engine generation.
#
# ``by_generation`` is the marker set per engine generation for a backend
# with more than one; an unlisted generation is a caller error.  The message
# differs per backend because the hand-written chain produced two of them:
# MetaX named the generation, while a generation the A800/H200 chain did not
# recognise fell through to the unsupported-hardware message.
_GRAPH_LOG_MARKERS = {
    "nvidia_a800": {
        "single_runtime": (
            "Initializing a V1 LLM engine",
            "torch.compile takes",
            "Graph capturing finished",
        ),
        "by_generation": None,
        "by_generation_error": None,
        "runtime_replay_marker": None,
    },
    "nvidia_h200": {
        "single_runtime": None,
        "by_generation": {
            "v1_0.11.0": (
                "Initializing a V1 LLM engine",
                "torch.compile takes",
                "Graph capturing finished",
            ),
            "v1_0.30.0": (
                "Initializing a V1 LLM engine",
                "torch.compile took",
                "Graph capturing finished",
            ),
        },
        # An unrecognised H200 generation fell through the
        # ``elif hardware_id == "nvidia_h200"`` branch to the generic error.
        "by_generation_error": (
            "unsupported graph-log hardware_id: {hardware_id!r}"),
        "runtime_replay_marker": None,
    },
    "metax_c500": {
        "single_runtime": None,
        "by_generation": {
            "v0": (
                "Capturing CUDA graph shapes",
                "Graph capturing finished",
            ),
            "v1": (
                "Initializing a V1 LLM engine",
                "Capturing CUDA graphs (",
                "Graph capturing finished",
            ),
        },
        "by_generation_error": (
            "MetaX graph-log validation requires a supported "
            "engine_generation, got {engine_generation!r}"),
        "runtime_replay_marker": None,
    },
    "ascend_910b": {
        "single_runtime": (
            "FULL_DECODE_ONLY compilation enabled on NPU",
        ),
        "by_generation": None,
        "by_generation_error": None,
        # The NPU backend replays its aclgraph; the CUDA ones do not.
        "runtime_replay_marker": "Replaying aclgraph",
    },
}


def _graph_log_markers(
    hardware_id: str, engine_generation: str | None, require_runtime_replay: bool,
) -> tuple[str, ...]:
    """Return the markers ``hardware_id``/``engine_generation`` must print."""

    if hardware_id not in _GRAPH_LOG_MARKERS:
        raise ValueError(
            f"unsupported graph-log hardware_id: {hardware_id!r}"
        )
    scope = _GRAPH_LOG_MARKERS[hardware_id]
    single_runtime = scope["single_runtime"]
    if single_runtime is not None:
        required_markers = single_runtime
    else:
        by_generation = scope["by_generation"]
        if engine_generation not in by_generation:
            raise ValueError(scope["by_generation_error"].format(
                hardware_id=hardware_id,
                engine_generation=engine_generation,
            ))
        required_markers = by_generation[engine_generation]
    if require_runtime_replay:
        replay_marker = scope["runtime_replay_marker"]
        if replay_marker is not None:
            required_markers += (replay_marker,)
    return required_markers


def _validate_graph_server_logs(
    stdout_path: Path,
    stderr_path: Path,
    *,
    hardware_id: str,
    engine_generation: str | None = None,
    require_runtime_replay: bool = False,
) -> dict[str, Any]:
    text = stdout_path.read_text(errors="replace") + "\n" + stderr_path.read_text(
        errors="replace"
    )
    required_markers = _graph_log_markers(
        hardware_id, engine_generation, require_runtime_replay)
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        raise RuntimeError(
            "vLLM server logs do not prove production Graph execution; "
            f"missing markers: {missing}"
        )
    return {
        "status": "validated",
        "required_markers": list(required_markers),
    }


def _validate_a800_async_scheduling_logs(
    stdout_path: Path, stderr_path: Path,
) -> dict[str, Any]:
    text = stdout_path.read_text(errors="replace") + "\n" + stderr_path.read_text(
        errors="replace"
    )
    required_markers = (
        "'async_scheduling': True",
        "mp-based distributed executor backend for async scheduling",
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        "Batch queue is enabled with size 2",
    )
    missing = [marker for marker in required_markers if marker not in text]
    if missing:
        raise RuntimeError(
        "vLLM logs do not prove the A800 async Scheduler runtime: "
            f"missing markers: {missing}"
        )
    return {
        "status": "validated",
        "required_markers": list(required_markers),
    }


def _requires_a800_async_scheduling_validation(
        hardware_id: str, async_scheduling: bool) -> bool:
    """Return whether the A800-specific executor-log contract applies."""

    if not isinstance(hardware_id, str):
        raise TypeError("hardware_id must be a string")
    if not isinstance(async_scheduling, bool):
        raise TypeError("async_scheduling must be a bool")
    return hardware_id == "nvidia_a800" and async_scheduling


def _graph_log_engine_generation(server_plan: dict[str, Any]) -> str:
    """Graph-log marker key; H200 markers differ between its two runtimes."""

    engine_generation = server_plan["engine"]["vllm_engine_generation"]
    if server_plan["hardware_id"] != "nvidia_h200":
        return engine_generation
    return f"{engine_generation}_{server_plan['engine']['runtime_version']}"


def _live_process_group_member_pids(process_group_id: int) -> list[int]:
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        raise RuntimeError("process-group quiescence requires Linux /proc")
    live_member_pids = []
    for process_path in proc_root.iterdir():
        if not process_path.name.isdecimal():
            continue
        stat_path = process_path / "stat"
        try:
            stat = stat_path.read_bytes()
        except FileNotFoundError:
            stat = None
        except OSError as error:
            raise RuntimeError(
                f"failed to inspect process state at {stat_path}"
            ) from error
        if stat is None:
            continue
        command_end = stat.rfind(b")")
        if command_end < 0:
            raise RuntimeError(
                f"invalid Linux process stat record at {stat_path}"
            )
        fields = stat[command_end + 1 :].split()
        if len(fields) < 3:
            raise RuntimeError(
                f"invalid Linux process stat record at {stat_path}"
            )
        try:
            member_process_group_id = int(fields[2])
        except ValueError as error:
            raise RuntimeError(f"invalid process group in {stat_path}") from error
        process_state = fields[0]
        if (
            member_process_group_id == process_group_id
            and process_state not in (b"Z", b"X", b"x")
        ):
            live_member_pids.append(int(process_path.name))
    return sorted(live_member_pids)


def _wait_for_process_group_quiescence(
    process_group_id: int,
    *,
    timeout_seconds: float,
) -> list[int]:
    deadline = time.monotonic() + timeout_seconds
    while True:
        live_member_pids = _live_process_group_member_pids(process_group_id)
        if not live_member_pids:
            return []
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            return live_member_pids
        time.sleep(
            min(_PROCESS_GROUP_POLL_INTERVAL_SECONDS, remaining_seconds)
        )


def _signal_process_group(process_group_id: int, signal_number: int) -> None:
    try:
        os.killpg(process_group_id, signal_number)
    except ProcessLookupError:
        if _live_process_group_member_pids(process_group_id):
            raise RuntimeError(
                "process group disappeared during signaling but still has live members: "
                f"pgid={process_group_id}"
            )


def _terminate_process_group(process_group_id: int) -> None:
    if _live_process_group_member_pids(process_group_id):
        _signal_process_group(process_group_id, signal.SIGTERM)
        live_member_pids = _wait_for_process_group_quiescence(
            process_group_id,
            timeout_seconds=_SERVER_TERMINATION_TIMEOUT_SECONDS,
        )
        if live_member_pids:
            _signal_process_group(process_group_id, signal.SIGKILL)
            live_member_pids = _wait_for_process_group_quiescence(
                process_group_id,
                timeout_seconds=_SERVER_TERMINATION_TIMEOUT_SECONDS,
            )
            if live_member_pids:
                raise RuntimeError(
                    "vLLM server process group remained live after SIGKILL: "
                    f"pgid={process_group_id}, pids={live_member_pids}"
                )


def _terminate_server(process: subprocess.Popen) -> int:
    process_group_id = process.pid
    _terminate_process_group(process_group_id)
    if process.returncode is None:
        try:
            process.wait(timeout=_SERVER_TERMINATION_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "vLLM server process group is quiescent but its leader did not "
                f"produce a return code: pid={process.pid}"
            ) from error
    if process.returncode is None:
        raise RuntimeError("vLLM server termination did not produce a return code")
    return process.returncode


def _preflight_scheduler_trace(
    server_plan: dict[str, Any], server_environment: dict[str, str]
) -> None:
    if not server_plan["scheduler_trace"]["enabled"]:
        return
    completed = subprocess.run(
        [
            server_plan["python_executable"],
            "-c",
            (
                "from dataset_pipeline.serving_dataset.vllm_scheduler_trace "
                "import install_from_environment; "
                "assert install_from_environment()"
            ),
        ],
        cwd=str(ROOT),
        env=server_environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0 or "Error in sitecustomize" in completed.stderr:
        raise RuntimeError(
            "Scheduler trace instrumentation preflight failed:\n"
            + completed.stderr.strip()
        )


def _sample_sharegpt(
    *,
    model: str,
    dataset: str,
    seed: int,
    requests: Sequence[RequestSpec],
) -> tuple[Any, list[Any]]:
    from transformers import AutoTokenizer
    from vllm.benchmarks.datasets import ShareGPTDataset

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    samples = ShareGPTDataset(
        dataset_path=dataset,
        random_seed=seed,
    ).sample(
        tokenizer,
        num_requests=len(requests),
        output_len=None,
    )
    if len(samples) != len(requests):
        raise ValueError(
            "ShareGPT sample count disagrees with canonical requests: "
            f"{len(samples)} != {len(requests)}"
        )
    for index, (sample, request) in enumerate(zip(samples, requests)):
        if sample.prompt_len != request.input_tokens:
            raise ValueError(
                f"ShareGPT input length mismatch at request {index}: "
                f"{sample.prompt_len} != {request.input_tokens}"
            )
        if sample.expected_output_len != request.output_tokens:
            raise ValueError(
                f"ShareGPT output length mismatch at request {index}: "
                f"{sample.expected_output_len} != {request.output_tokens}"
            )
    return tokenizer, samples


def _prefix_cache_reuse_evidence(
    tokenizer: Any,
    samples: Sequence[Any],
    *,
    kv_block_size_tokens: int,
) -> dict[str, Any]:
    if (
        isinstance(kv_block_size_tokens, bool)
        or not isinstance(kv_block_size_tokens, int)
        or kv_block_size_tokens < 1
    ):
        raise ValueError("kv_block_size_tokens must be a positive integer")
    token_sequences = []
    for index, sample in enumerate(samples):
        encoded = tokenizer(sample.prompt)
        token_ids = list(encoded.input_ids)
        if len(token_ids) != sample.prompt_len:
            raise ValueError(
                "prefix-cache evidence token count disagrees with sampled "
                f"prompt {index}: {len(token_ids)} != {sample.prompt_len}"
            )
        if any(isinstance(token_id, bool) or not isinstance(token_id, int)
               for token_id in token_ids):
            raise TypeError("tokenizer input_ids must contain integers")
        token_sequences.append(token_ids)

    shared_prefix_block_pairs = 0
    max_shared_prefix_blocks = 0
    for left_index, left in enumerate(token_sequences):
        for right in token_sequences[left_index + 1:]:
            shared_tokens = 0
            for left_token, right_token in zip(left, right):
                if left_token != right_token:
                    break
                shared_tokens += 1
            shared_blocks = shared_tokens // kv_block_size_tokens
            if shared_blocks:
                shared_prefix_block_pairs += 1
                max_shared_prefix_blocks = max(
                    max_shared_prefix_blocks, shared_blocks
                )
    token_stream_bytes = json.dumps(
        token_sequences,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    evidence = {
        "kind": "static_prompt_token_prefix_analysis_v1",
        "logical_prefix_hit_observability": "unavailable_without_instrumentation",
        "enable_prefix_caching": True,
        "kv_block_size_tokens": kv_block_size_tokens,
        "request_count": len(token_sequences),
        "prompt_token_stream_sha256": hashlib.sha256(token_stream_bytes).hexdigest(),
        "shared_prefix_block_pair_count": shared_prefix_block_pairs,
        "max_shared_prefix_blocks": max_shared_prefix_blocks,
        "effective_reuse_representative_tokens": 0,
        "representative_provenance": "scheduler_equivalent_representative",
    }
    if shared_prefix_block_pairs:
        raise ValueError(
            "canonical prompts contain a shared complete prefix-cache block; "
            "a zero effective-reuse representative is not valid"
        )
    return evidence


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values_s: Sequence[float]) -> dict[str, float]:
    if not values_s:
        raise ValueError("latency distribution must not be empty")
    return {
        "mean_ms": sum(values_s) * 1000.0 / len(values_s),
        "p50_ms": _percentile(values_s, 50.0) * 1000.0,
        "p90_ms": _percentile(values_s, 90.0) * 1000.0,
        "p99_ms": _percentile(values_s, 99.0) * 1000.0,
    }


# Highest descriptor the replay client pre-sizes its fd table to (see below).
CLIENT_FD_TABLE_RESERVE = 1023


def _reserve_client_fd_table() -> None:
    """Grow this process's fd table once, before warmup and replay.

    On Linux (5.15 on h200-tmp) the first use of fd 64, 128, 256, ... in a
    multithreaded process expands the fd table and waits for a system-wide RCU
    grace period inside socket().  During replay that froze the asyncio loop
    for 12-338 ms when about 35 requests were first in flight, so every request
    due in that window went out late (dispatch-lag failures on H200).  The
    table never shrinks, so paying the expansion here removes the stall without
    changing the arrival schedule or when connections are opened.
    """

    import resource

    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft_limit != resource.RLIM_INFINITY and soft_limit <= CLIENT_FD_TABLE_RESERVE:
        raise RuntimeError(
            f"RLIMIT_NOFILE soft limit {soft_limit} does not allow fd "
            f"{CLIENT_FD_TABLE_RESERVE}")
    reserve_fd = os.open(os.devnull, os.O_RDONLY)
    os.dup2(reserve_fd, CLIENT_FD_TABLE_RESERVE, inheritable=False)
    os.close(CLIENT_FD_TABLE_RESERVE)
    os.close(reserve_fd)


async def _run_http_benchmark(
    *,
    base_url: str,
    served_model_name: str,
    samples: Sequence[Any],
    requests: Sequence[RequestSpec],
    require_request_id: bool,
    tokenizer: Any | None = None,
    max_model_len: int = 8192,
    benchmark_boundary_callback: Callable[[], None] | None = None,
    arrival_probe_per_token_s: float = 0.0,
) -> dict[str, Any]:
    import aiohttp
    try:
        from vllm.benchmarks.lib.endpoint_request_func import (
            RequestFuncInput,
            async_request_openai_completions,
        )
    except ModuleNotFoundError as error:
        if error.name not in {
            "vllm.benchmarks.lib",
            "vllm.benchmarks.lib.endpoint_request_func",
        }:
            raise
        from vllm.benchmarks.endpoint_request_func import (
            RequestFuncInput,
            async_request_openai_completions,
        )

    request_fields = RequestFuncInput.__dataclass_fields__
    request_function_parameters = inspect.signature(
        async_request_openai_completions
    ).parameters
    supports_request_id = "request_id" in request_fields
    supports_shared_session = "session" in request_function_parameters
    if supports_request_id != supports_shared_session:
        raise RuntimeError(
            "unsupported vLLM benchmark client API: request_id and session "
            "support must change together"
        )
    if require_request_id and not supports_request_id:
        raise RuntimeError(
            "Scheduler trace requires vLLM benchmark request_id support"
        )

    def request_input(
        *,
        prompt: str,
        prompt_len: int,
        output_len: int,
        request_id: str,
    ) -> Any:
        kwargs = {
            "prompt": prompt,
            "api_url": api_url,
            "prompt_len": prompt_len,
            "output_len": output_len,
            "model": served_model_name,
            "model_name": served_model_name,
            "ignore_eos": True,
        }
        if supports_request_id:
            kwargs["request_id"] = request_id
        return RequestFuncInput(**kwargs)

    async def send(request: Any, session: Any) -> Any:
        if supports_shared_session:
            return await async_request_openai_completions(
                request_func_input=request,
                session=session,
            )
        return await async_request_openai_completions(
            request_func_input=request,
        )

    _reserve_client_fd_table()
    connector = aiohttp.TCPConnector(
        limit=0,
        limit_per_host=0,
        ttl_dns_cache=300,
        use_dns_cache=True,
        keepalive_timeout=60,
        enable_cleanup_closed=True,
        force_close=False,
        ssl=False,
    )
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    api_url = f"{base_url.rstrip('/')}/v1/completions"
    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=False,
        timeout=timeout,
    ) as session:
        if tokenizer is None:
            warmup_specs = [("sharegpt-warmup", "Warmup.", 1, 8)]
            mixed_warmup_specs: list[tuple[str, str, int, int]] = []
            isolated_warmup_specs: list[tuple[str, str, int, int]] = []
        else:
            eligible = [
                (index, sample)
                for index, sample in enumerate(samples)
                if sample.prompt_len <= max_model_len - 256
            ]
            if len(eligible) < 8:
                raise RuntimeError(
                    "deterministic warmup requires eight prompts with model-length "
                    "headroom"
                )
            eligible.sort(key=lambda value: value[1].prompt_len)
            positions = [
                round(position * (len(eligible) - 1) / 7)
                for position in range(8)
            ]

            def isolated_warmup_spec(
                wave: str,
                ordinal: int,
                output_len: int,
            ) -> tuple[str, str, int, int]:
                sample_index, sample = eligible[positions[ordinal]]
                prefix = (
                    f"Midlayer isolated {wave} warmup request {ordinal}; "
                    f"canonical sample {sample_index}; no prefix reuse. " * 3
                )
                prompt = prefix + sample.prompt
                prompt_len = len(tokenizer(prompt).input_ids)
                if prompt_len + output_len > max_model_len:
                    raise RuntimeError(
                        "warmup prompt exceeds configured maximum model length"
                    )
                return (
                    f"warmup-{wave}-{ordinal:02d}",
                    prompt,
                    prompt_len,
                    output_len,
                )

            warmup_specs = [
                isolated_warmup_spec("batch", ordinal, 8)
                for ordinal in range(8)
            ]
            mixed_warmup_specs = [
                isolated_warmup_spec(
                    "mixed",
                    ordinal,
                    64 if ordinal == 0 else 8,
                )
                for ordinal in range(8)
            ]
            isolated_warmup_specs = [
                isolated_warmup_spec("isolated", 7, 64)
            ]

        async def send_warmup(
            specification: tuple[str, str, int, int]
        ) -> Any:
            request_id, prompt, prompt_len, output_len = specification
            output = await send(request_input(
                prompt=prompt,
                prompt_len=prompt_len,
                output_len=output_len,
                request_id=request_id,
            ), session)
            if not output.success:
                raise RuntimeError(
                    f"vLLM warmup request {request_id} failed: {output.error}"
                )
            if output.output_tokens != output_len:
                raise RuntimeError(
                    f"vLLM warmup request {request_id} returned "
                    f"{output.output_tokens} tokens instead of {output_len}"
                )
            return output

        warmup_started = time.perf_counter()
        await asyncio.gather(*(
            send_warmup(specification) for specification in warmup_specs
        ))
        if mixed_warmup_specs:
            anchor = asyncio.create_task(send_warmup(mixed_warmup_specs[0]))
            await asyncio.sleep(0.25)
            followers = await asyncio.gather(*(
                send_warmup(specification)
                for specification in mixed_warmup_specs[1:]
            ))
            await anchor
            if len(followers) != 7:
                raise RuntimeError("mixed warmup follower count is inconsistent")
        for specification in isolated_warmup_specs:
            await send_warmup(specification)
        warmup_duration_s = time.perf_counter() - warmup_started
        if benchmark_boundary_callback is not None:
            benchmark_boundary_callback()

        dispatch_start = asyncio.Event()
        all_requests_ready = asyncio.Event()
        ready_request_count = 0
        benchmark_start_monotonic_ns: int | None = None

        async def submit_one(index: int) -> tuple[Any, int]:
            nonlocal ready_request_count
            ready_request_count += 1
            if ready_request_count == len(requests):
                all_requests_ready.set()
            await dispatch_start.wait()
            if benchmark_start_monotonic_ns is None:
                raise RuntimeError("benchmark clock was not initialized")
            request = requests[index]
            # Opt-in controlled probe: delay this request's submission in
            # proportion to its prompt length, which is the only shape of
            # perturbation that changes the relative spacing between arrivals
            # and can therefore reach the Scheduler's batching decisions.  A
            # zero rate reproduces the canonical arrival replay exactly.
            probe_offset_s = arrival_probe_per_token_s * request.input_tokens
            delay_ns = (
                benchmark_start_monotonic_ns
                + round(
                    (request.arrival_time_s + probe_offset_s) * 1_000_000_000)
                - time.perf_counter_ns()
            )
            if delay_ns > 0:
                await asyncio.sleep(delay_ns / 1_000_000_000)
            submitted_at_ns = time.perf_counter_ns()
            completion_request = request_input(
                prompt=samples[index].prompt,
                prompt_len=request.input_tokens,
                output_len=request.output_tokens,
                request_id=request.request_id,
            )
            output = await send(completion_request, session)
            return output, submitted_at_ns

        request_tasks = [
            asyncio.create_task(submit_one(index))
            for index in range(len(requests))
        ]
        await all_requests_ready.wait()
        benchmark_start_monotonic_ns = time.perf_counter_ns()
        dispatch_start.set()
        completed = await asyncio.gather(*request_tasks)
        duration_s = (
            time.perf_counter_ns() - benchmark_start_monotonic_ns
        ) / 1_000_000_000

    if benchmark_start_monotonic_ns is None:
        raise RuntimeError("benchmark clock was not initialized")
    host_boot_id = Path(
        "/proc/sys/kernel/random/boot_id"
    ).read_text(encoding="utf-8").strip()
    if not host_boot_id:
        raise RuntimeError("host boot ID is empty")

    per_request = []
    ttft_s = []
    tpot_s = []
    e2e_s = []
    total_output_tokens = 0
    failures = []
    for index, ((output, submitted_at_ns), request) in enumerate(
            zip(completed, requests)):
        if not output.success:
            failures.append({
                "request_id": request.request_id,
                "error": output.error,
            })
            continue
        if output.output_tokens != request.output_tokens:
            failures.append({
                "request_id": request.request_id,
                "error": "output token count disagrees with canonical request",
                "expected_output_tokens": request.output_tokens,
                "actual_output_tokens": output.output_tokens,
            })
            continue
        request_tpot_s = 0.0
        if output.output_tokens > 1:
            request_tpot_s = (
                output.latency - output.ttft
            ) / (output.output_tokens - 1)
            tpot_s.append(request_tpot_s)
        ttft_s.append(output.ttft)
        e2e_s.append(output.latency)
        total_output_tokens += output.output_tokens
        request_start_s = output.start_time
        if (
            isinstance(request_start_s, bool)
            or not isinstance(request_start_s, (int, float))
            or not math.isfinite(request_start_s)
        ):
            raise TypeError("vLLM benchmark request start time is invalid")
        request_start_ns = round(float(request_start_s) * 1_000_000_000)
        if request_start_ns < submitted_at_ns:
            raise RuntimeError(
                "vLLM benchmark TTFT start predates request submission"
            )
        first_token_ns = request_start_ns + round(
            float(output.ttft) * 1_000_000_000
        )
        submit_offset_s = (
            submitted_at_ns - benchmark_start_monotonic_ns
        ) / 1_000_000_000
        per_request.append({
            "request_id": request.request_id,
            "scheduled_arrival_s": request.arrival_time_s,
            "arrival_probe_offset_s": (
                arrival_probe_per_token_s * request.input_tokens),
            "submit_offset_s": submit_offset_s,
            # Dispatch lag measures how late the client was against the
            # schedule it was asked to hit, so the requested probe offset is
            # part of that schedule and is not counted as lag.
            "dispatch_lag_ms": (
                submit_offset_s
                - request.arrival_time_s
                - arrival_probe_per_token_s * request.input_tokens
            ) * 1000.0,
            "input_tokens": request.input_tokens,
            "output_tokens": output.output_tokens,
            "submit_monotonic_ns": submitted_at_ns,
            "ttft_start_monotonic_ns": request_start_ns,
            "first_token_monotonic_ns": first_token_ns,
            "ttft_ms": output.ttft * 1000.0,
            "tpot_ms": request_tpot_s * 1000.0,
            "e2e_ms": output.latency * 1000.0,
        })
    if failures:
        raise RuntimeError(json.dumps({
            "message": "one or more production Serving requests failed",
            "failures": failures,
        }, sort_keys=True))
    return {
        "warmup": {
            "request_count": (
                len(warmup_specs)
                + len(mixed_warmup_specs)
                + len(isolated_warmup_specs)
            ),
            "batch_wave_request_count": len(warmup_specs),
            "mixed_wave_request_count": len(mixed_warmup_specs),
            "isolated_wave_request_count": len(isolated_warmup_specs),
            "duration_s": warmup_duration_s,
            "request_ids": [
                specification[0]
                for specification in (
                    *warmup_specs,
                    *mixed_warmup_specs,
                    *isolated_warmup_specs,
                )
            ],
            "prompt_tokens": [
                specification[2]
                for specification in (
                    *warmup_specs,
                    *mixed_warmup_specs,
                    *isolated_warmup_specs,
                )
            ],
            "output_tokens": [
                specification[3]
                for specification in (
                    *warmup_specs,
                    *mixed_warmup_specs,
                    *isolated_warmup_specs,
                )
            ],
        },
        "duration_s": duration_s,
        "benchmark_clock": {
            "clock_domain": "shared_host_time.perf_counter_ns",
            "host_boot_id": host_boot_id,
            "benchmark_start_monotonic_ns": benchmark_start_monotonic_ns,
        },
        "completed_requests": len(per_request),
        "total_input_tokens": sum(request.input_tokens for request in requests),
        "total_output_tokens": total_output_tokens,
        "request_throughput_per_s": len(per_request) / duration_s,
        "output_token_throughput_per_s": total_output_tokens / duration_s,
        "ttft": _distribution(ttft_s),
        "tpot": _distribution(tpot_s),
        "e2e": _distribution(e2e_s),
        "per_request": per_request,
    }


def build_ground_truth(
    *,
    model: str,
    dataset: str,
    seed: int,
    case: dict[str, Any],
    server_plan: dict[str, Any],
    dry_run: bool,
    benchmark_boundary_callback: Callable[[], None] | None = None,
) -> dict[str, Any]:
    requests_path = case["requests_path"]
    request_stream = load_canonical_requests(requests_path)
    requests = request_stream.requests
    if requests[0].arrival_time_s != 0:
        raise ValueError("canonical request schedule must start at zero seconds")
    if any(request.output_tokens < 1 for request in requests):
        raise ValueError("ground-truth requests must generate at least one token")
    if any(request.prefix_hit_tokens != 0 for request in requests):
        raise ValueError(
            "ShareGPT ground-truth runner does not support prefix-hit requests"
        )
    tokenizer, samples = _sample_sharegpt(
        model=model,
        dataset=dataset,
        seed=seed,
        requests=requests,
    )
    prefix_cache_evidence = _prefix_cache_reuse_evidence(
        tokenizer,
        samples,
        kv_block_size_tokens=server_plan["engine"]["kv_block_size_tokens"],
    )
    if len(requests) != case["requests_row_count"]:
        raise ValueError("requests.csv row count disagrees with case manifest")
    import vllm

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "dry_run" if dry_run else "running",
        "workload": {
            "kind": "sharegpt_canonical_schedule",
            "request_count": len(requests),
            "expected_qps": case["runtime"]["target_qps"],
            "scheduled_duration_s": requests[-1].arrival_time_s,
            "seed": seed,
            "case_manifest_path": case["manifest_path"],
            "case_manifest_sha256": case["manifest_sha256"],
            "suite_manifest_path": case["suite_manifest_path"],
            "suite_manifest_sha256": case["suite_manifest_sha256"],
            "identity": case["identity"],
            "model_config_path": case["model_config_path"],
            "model_config_sha256": case["model_config_sha256"],
            "requests_path": requests_path,
            "requests_sha256": case["requests_sha256"],
            "request_stream_sha256": request_stream_sha256(request_stream),
            "dataset_path": str(Path(dataset).resolve(strict=True)),
            "dataset_sha256": sha256_file(dataset),
            "prompt_and_output_lengths_validated": True,
            "prefix_cache_reuse_evidence": prefix_cache_evidence,
            "request_template": case["request_template"],
        },
        "server": {
            "model_path": str(Path(model).resolve(strict=True)),
            "client_vllm_version": vllm.__version__,
            **server_plan,
        },
        "provenance": {
            **git_code_identity(ROOT),
            "argv": list(sys.argv),
        },
        "metrics": None,
    }
    if dry_run:
        return result
    result["metrics"] = asyncio.run(_run_http_benchmark(
        base_url=server_plan["endpoint"]["base_url"],
        served_model_name=case["identity"]["model_id"],
        samples=samples,
        requests=requests,
        require_request_id=server_plan["scheduler_trace"]["enabled"],
        tokenizer=tokenizer,
        max_model_len=server_plan["engine"]["max_model_len"],
        benchmark_boundary_callback=benchmark_boundary_callback,
        arrival_probe_per_token_s=(
            server_plan["arrival_probe_per_1000_tokens_ms"] / 1000.0 / 1000.0),
    ))
    max_dispatch_lag_ms = max(
        request["dispatch_lag_ms"]
        for request in result["metrics"]["per_request"]
    )
    result["metrics"]["max_dispatch_lag_ms"] = max_dispatch_lag_ms
    result["status"] = "success"
    return result


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a canonical ShareGPT schedule against vLLM"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--case-manifest", required=True)
    parser.add_argument("--suite-manifest", required=True)
    parser.add_argument("--devices", required=True, nargs="+", type=int)
    parser.add_argument("--port", default=18080, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu-memory-utilization", default=0.8, type=float)
    parser.add_argument("--max-model-len", default=8192, type=int)
    parser.add_argument("--max-num-batched-tokens", default=8192, type=int)
    parser.add_argument("--max-num-seqs", default=256, type=int)
    parser.add_argument("--kv-block-size-tokens", required=True, type=int)
    parser.add_argument("--max-dispatch-lag-ms", default=100.0, type=float)
    parser.add_argument(
        "--arrival-probe-per-1000-tokens-ms",
        default=0.0,
        type=float,
        help=(
            "controlled experiment only: delay each request's submission by "
            "this many ms per 1000 prompt tokens, which perturbs the relative "
            "spacing between arrivals; zero, the default, replays the "
            "canonical schedule unchanged"),
    )
    parser.add_argument("--server-wait-seconds", default=600.0, type=float)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scheduler-trace-output")
    parser.add_argument("--scheduler-request-add-output")
    parser.add_argument("--a800-compilation-counter-output")
    parser.add_argument(
        "--target-hardware-id",
        choices=tuple(REQUEST_TEMPLATE_HARDWARE_BY_TARGET),
        help=(
            "run on this hardware while replaying the --case-manifest request "
            "file and arrival schedule of its template hardware"),
    )
    parser.add_argument(
        "--runtime-version",
        choices=H200_RUNTIME_VERSIONS,
        help="installed vLLM release; required with --target-hardware-id",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.target_hardware_id is None) != (args.runtime_version is None):
        raise ValueError(
            "--target-hardware-id and --runtime-version must be given together")
    if any(device < 0 for device in args.devices):
        raise ValueError("devices must be non-negative")
    if not 0 < args.port < 65536:
        raise ValueError("port must be between 1 and 65535")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    for name in (
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "kv_block_size_tokens",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.max_dispatch_lag_ms <= 0:
        raise ValueError("max_dispatch_lag_ms must be positive")
    if args.server_wait_seconds <= 0:
        raise ValueError("server_wait_seconds must be positive")
    output_path = Path(args.output)
    if output_path.exists():
        raise ValueError(f"refusing to overwrite existing output: {output_path}")
    if args.scheduler_trace_output is not None:
        scheduler_trace_path = Path(args.scheduler_trace_output)
        if scheduler_trace_path.exists():
            raise ValueError(
                "refusing to overwrite existing Scheduler trace: "
                f"{scheduler_trace_path}"
            )
        if scheduler_trace_path.resolve() == output_path.resolve():
            raise ValueError("Scheduler trace output must differ from result output")
    if args.scheduler_request_add_output is not None:
        if args.scheduler_trace_output is None:
            raise ValueError(
                "--scheduler-request-add-output requires --scheduler-trace-output"
            )
        request_add_path = Path(args.scheduler_request_add_output)
        if request_add_path.exists():
            raise ValueError(
                "refusing to overwrite Scheduler request-add output: "
                f"{request_add_path}"
            )
        stage_sidecars = tuple(
            request_add_path.parent.glob(
                f"{request_add_path.name}.stages.*.jsonl"
            )
        )
        if stage_sidecars:
            raise ValueError(
                "refusing to overwrite Scheduler request-stage sidecars: "
                + ", ".join(str(path) for path in stage_sidecars)
            )
        if request_add_path.resolve() == output_path.resolve():
            raise ValueError(
                "Scheduler request-add output must differ from result output"
            )
    if args.a800_compilation_counter_output is not None:
        counter_output_path = Path(args.a800_compilation_counter_output)
        if counter_output_path.exists():
            raise ValueError(
                "refusing to overwrite existing A800 compilation-counter output: "
                f"{counter_output_path}"
            )
        if counter_output_path.resolve() == output_path.resolve():
            raise ValueError(
                "A800 compilation-counter output must differ from result output"
            )
        if args.scheduler_trace_output is not None:
            raise ValueError(
                "A800 counter-only instrumentation cannot accompany Scheduler trace"
            )
    return args


def _write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _metric_summary(result: dict[str, Any]) -> dict[str, Any] | None:
    metrics = result["metrics"]
    if metrics is None:
        return None
    return {
        key: metrics[key]
        for key in (
            "duration_s",
            "completed_requests",
            "request_throughput_per_s",
            "output_token_throughput_per_s",
            "max_dispatch_lag_ms",
            "ttft",
            "tpot",
            "e2e",
        )
    }


def _scheduler_runtime_depth_metadata(trace_path: Path) -> dict[str, Any]:
    """Load the trace sidecar written from the live Executor object."""

    path = Path(f"{trace_path}.runtime.json")
    if not path.is_file():
        raise RuntimeError(
            "Scheduler trace is missing live Executor depth metadata: "
            f"{path}")
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or set(metadata) != {
            "schema_version", "scheduler_pipeline_depth",
            "scheduler_pipeline_depth_evidence"}:
        raise ValueError("Scheduler runtime depth metadata fields are invalid")
    if metadata["schema_version"] != 1:
        raise ValueError("Scheduler runtime depth metadata schema is invalid")
    depth = metadata["scheduler_pipeline_depth"]
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise ValueError("Scheduler runtime depth must be a positive integer")
    evidence = metadata["scheduler_pipeline_depth_evidence"]
    if not isinstance(evidence, dict):
        raise ValueError("Scheduler runtime depth evidence fields are invalid")
    source_pair = (evidence["source_kind"], evidence["source_field"])
    if source_pair == ("runtime_executor_property", "max_concurrent_batches"):
        expected_fields = {
            "source_kind", "source_field", "executor_class",
            "pipeline_parallel_size", "runtime_version", "source_identity"}
    elif source_pair == (
            "runtime_vllm_config_field", "vllm_config.max_concurrent_batches"):
        expected_fields = {
            "source_kind", "source_field", "configured_depth", "executor_class",
            "pipeline_parallel_size", "runtime_version", "source_identity"}
        if evidence["configured_depth"] != depth:
            raise ValueError(
                "Scheduler runtime depth evidence configured depth is invalid")
    else:
        raise ValueError("Scheduler runtime depth evidence source is invalid")
    if set(evidence) != expected_fields:
        raise ValueError("Scheduler runtime depth evidence fields are invalid")
    if any(not isinstance(evidence[field], str) or not evidence[field]
           for field in ("executor_class", "runtime_version")):
        raise ValueError("Scheduler runtime depth evidence text is invalid")
    if source_pair == ("runtime_executor_property", "max_concurrent_batches"):
        if (not isinstance(evidence["source_identity"], str)
                or not evidence["source_identity"]):
            raise ValueError(
                "Scheduler runtime depth evidence source identity is invalid")
    elif source_pair == (
            "runtime_vllm_config_field", "vllm_config.max_concurrent_batches"):
        source_identity = evidence["source_identity"]
        if (not isinstance(source_identity, dict)
                or set(source_identity) != {"module_file", "module_name"}):
            raise ValueError(
                "Scheduler runtime depth evidence source identity is invalid")
        if any(not isinstance(source_identity[field], str)
               or not source_identity[field]
               for field in ("module_file", "module_name")):
            raise ValueError(
                "Scheduler runtime depth evidence source identity is invalid")
    pipeline_parallel_size = evidence["pipeline_parallel_size"]
    if (isinstance(pipeline_parallel_size, bool)
            or not isinstance(pipeline_parallel_size, int)
            or pipeline_parallel_size < 1):
        raise ValueError("Scheduler runtime depth evidence PP is invalid")
    return metadata


def _scheduler_clock_metadata(trace_path: Path) -> dict[str, Any]:
    """Load the explicit host-monotonic origin used by Scheduler rows."""

    path = Path(f"{trace_path}.clock.json")
    if not path.is_file():
        raise RuntimeError(
            f"Scheduler trace is missing clock metadata: {path}"
        )
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or set(metadata) != {
        "schema_version",
        "clock_domain",
        "host_boot_id",
        "trace_origin_monotonic_ns",
    }:
        raise ValueError("Scheduler clock metadata fields are invalid")
    if metadata["schema_version"] != 1:
        raise ValueError("Scheduler clock metadata schema is invalid")
    if metadata["clock_domain"] != "shared_host_time.perf_counter_ns":
        raise ValueError("Scheduler clock domain is invalid")
    if (
        not isinstance(metadata["host_boot_id"], str)
        or not metadata["host_boot_id"]
    ):
        raise ValueError("Scheduler host boot ID is invalid")
    origin = metadata["trace_origin_monotonic_ns"]
    if isinstance(origin, bool) or not isinstance(origin, int) or origin < 0:
        raise ValueError("Scheduler trace origin must be a non-negative integer")
    return metadata


def _scheduler_trace_summary(
    path: str | Path,
    expected_request_count: int,
    *,
    max_result_consumption_gap_ns: int | None = None,
) -> dict[str, Any]:
    if max_result_consumption_gap_ns is not None and (
        isinstance(max_result_consumption_gap_ns, bool)
        or not isinstance(max_result_consumption_gap_ns, int)
        or max_result_consumption_gap_ns <= 0
    ):
        raise ValueError(
            "Scheduler result-consumption gap limit must be a positive integer"
        )
    trace_path = Path(path)
    if not trace_path.is_file():
        raise RuntimeError(
            f"Scheduler trace was not produced by the accepted run: {trace_path}"
        )
    depth_metadata = _scheduler_runtime_depth_metadata(trace_path)
    clock_metadata = _scheduler_clock_metadata(trace_path)
    configured_depth = depth_metadata["scheduler_pipeline_depth"]
    records = []
    for line_index, line in enumerate(trace_path.read_text().splitlines()):
        if not line:
            raise ValueError("Scheduler trace contains an empty JSONL row")
        value = json.loads(line)
        if value["step_index"] != line_index:
            raise ValueError("Scheduler trace step indexes are not contiguous")
        records.append(value)
    if not records:
        raise ValueError("Scheduler trace is empty")
    if records[-1]["cumulative_completed_requests"] != expected_request_count:
        raise ValueError("Scheduler trace did not complete every request")
    summary = {
        "path": str(trace_path.resolve()),
        "sha256": sha256_file(trace_path),
        "row_count": len(records),
        "completed_requests": records[-1]["cumulative_completed_requests"],
        "scheduled_tokens": records[-1]["cumulative_scheduled_tokens"],
        "runtime_metadata_path": str(
            Path(f"{trace_path}.runtime.json").resolve()),
        "runtime_metadata_sha256": sha256_file(
            Path(f"{trace_path}.runtime.json")),
        "scheduler_pipeline_depth": configured_depth,
        "scheduler_pipeline_depth_evidence": depth_metadata[
            "scheduler_pipeline_depth_evidence"],
        "clock_metadata_path": str(
            Path(f"{trace_path}.clock.json").resolve()),
        "clock_metadata_sha256": sha256_file(
            Path(f"{trace_path}.clock.json")),
        "clock": clock_metadata,
    }
    if "rank_observations" in records[0]:
        return _a800_scheduler_trace_summary(
            records, summary, configured_depth)
    if "configured_cudagraph_mode" not in records[0]:
        return summary
    graph_mode_counts = {"FULL": 0, "PIECEWISE": 0, "NONE": 0}
    fallback_counts: dict[str, int] = {}
    max_in_flight = 0
    for record in records:
        scheduled_tokens = record["scheduled_tokens"]
        request_tokens = sum(
            request["scheduled_tokens"]
            for request in record["scheduled_requests"]
        )
        if scheduled_tokens != request_tokens:
            raise ValueError("Scheduler row request tokens are inconsistent")
        if scheduled_tokens != (
            record["prefill_tokens"] + record["decode_tokens"]
        ):
            raise ValueError("Scheduler row prefill/decode tokens are inconsistent")
        if scheduled_tokens != record["num_unpadded_tokens"]:
            raise ValueError("Scheduler row unpadded tokens are inconsistent")
        if record["padded_total_tokens"] != (
            scheduled_tokens + record["padding_tokens"]
        ):
            raise ValueError("Scheduler row padding tokens are inconsistent")
        runtime_mode = record["runtime_cudagraph_mode"]
        if runtime_mode not in graph_mode_counts:
            raise ValueError("Scheduler row has an unsupported runtime Graph mode")
        graph_mode_counts[runtime_mode] += 1
        fallback_reason = record["graph_fallback_reason"]
        if runtime_mode == "NONE":
            if record["capture_bucket_tokens"] is not None:
                raise ValueError("NONE Scheduler row must not have a capture bucket")
            if not isinstance(fallback_reason, str) or not fallback_reason:
                raise ValueError("NONE Scheduler row must have a fallback reason")
            fallback_counts[fallback_reason] = (
                fallback_counts[fallback_reason] + 1
                if fallback_reason in fallback_counts
                else 1
            )
        elif runtime_mode in ("FULL", "PIECEWISE"):
            if record["capture_bucket_tokens"] != record["padded_total_tokens"]:
                raise ValueError("Graph Scheduler row capture bucket is inconsistent")
            if fallback_reason is not None:
                raise ValueError("Graph Scheduler row must not have a fallback reason")
        else:
            raise ValueError(f"unsupported Scheduler runtime Graph mode: {runtime_mode!r}")
        if not (
            record["schedule_start_ns"] <= record["schedule_end_ns"]
            <= record["device_execution_start_ns"]
            <= record["device_execution_end_ns"]
            <= record["scheduler_update_start_ns"]
            <= record["scheduler_update_end_ns"]
        ):
            raise ValueError("Scheduler row stage timestamps are not causal")
        if (
            max_result_consumption_gap_ns is not None
            and record["scheduler_update_start_ns"]
            - record["device_execution_end_ns"]
            > max_result_consumption_gap_ns
        ):
            raise RuntimeError(
                "Scheduler device result was not consumed within the accepted "
                "host-side gap"
            )
        in_flight = record["in_flight_step_count"]
        if isinstance(in_flight, bool) or not isinstance(in_flight, int):
            raise TypeError("Scheduler row in-flight count must be an integer")
        if not 1 <= in_flight <= configured_depth:
            raise ValueError(
                "Scheduler row in-flight count exceeds configured depth")
        max_in_flight = max(max_in_flight, in_flight)
    warmup_path = Path(f"{trace_path}.warmup.json")
    if not warmup_path.is_file():
        raise ValueError("Scheduler runtime warmup evidence is missing")
    warmup = json.loads(warmup_path.read_text())
    if warmup["max_scheduled_request_count"] < 8:
        raise ValueError("warmup did not schedule at least eight requests together")
    if warmup["mixed_prefill_decode_step_count"] < 1:
        raise ValueError("warmup did not exercise mixed prefill/decode")
    summary.update({
        "runtime_cudagraph_mode_counts": graph_mode_counts,
        "graph_fallback_counts": fallback_counts,
        "max_in_flight_step_count": max_in_flight,
        "warmup_path": str(warmup_path.resolve()),
        "warmup_sha256": sha256_file(warmup_path),
        "warmup": warmup,
    })
    return summary


_VLLM_011_MULTIPROC_EXECUTOR_CLASS = (
    "vllm.v1.executor.multiproc_executor.MultiprocExecutor"
)


def _worker_dispatch_reply_rank(
    evidence: dict[str, Any], world_size: int
) -> int:
    """Return the only worker rank whose reply the executor RPC envelope awaits.

    vLLM 0.11.0 ``v1/executor/multiproc_executor.py`` (sha256 ``ad938d34...
    e0cd16``, the file recorded by A800 and H200 0.11.0 captures):
    ``execute_model`` without a KV connector calls ``collective_rpc(...,
    unique_reply_rank=self.output_rank)`` (L179-186); ``collective_rpc``
    dequeues only ``self.workers[unique_reply_rank]`` (L237-238); the worker
    busy loop replies only when ``self.rank == output_rank`` (L674-678); and
    ``_get_output_rank`` returns ``world_size - tensor_parallel_size``
    (L335-345) with ``world_size == TP * PP`` (L62-67).  The ground-truth
    runner never passes ``--kv-transfer-config``, so the all-rank branch
    (``has_connector``) is unreachable here.  A non-reply rank's
    ``Worker.execute_model`` has no happens-before edge to the executor
    completion, so only the reply rank's worker envelope is contained.
    """

    if world_size == 1:
        # One worker: whichever executor runs it must wait for its reply.
        return 0
    if evidence["executor_class"] != _VLLM_011_MULTIPROC_EXECUTOR_CLASS:
        raise ValueError(
            "multi-rank worker-dispatch trace uses an unaudited executor reply "
            f"contract: {evidence['executor_class']!r}"
        )
    if evidence["runtime_version"] != "0.11.0":
        raise ValueError(
            "multi-rank worker-dispatch trace uses an unaudited vLLM release: "
            f"{evidence['runtime_version']!r}"
        )
    pipeline_parallel_size = evidence["pipeline_parallel_size"]
    if world_size % pipeline_parallel_size:
        raise ValueError(
            "worker rank count is not a multiple of pipeline parallel size"
        )
    return world_size - world_size // pipeline_parallel_size


def _a800_scheduler_trace_summary(
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    configured_depth: int,
) -> dict[str, Any]:
    graph_mode_counts = {"FULL": 0, "PIECEWISE": 0, "NONE": 0}
    fallback_counts: dict[str, int] = {}
    max_in_flight = 0
    expected_ranks: set[int] | None = None
    first_observations = records[0]["rank_observations"]
    if not isinstance(first_observations, list) or not first_observations:
        raise TypeError("A800 Scheduler row rank observations are invalid")
    # Every row is checked below to cover exactly ranks 0..world_size-1.
    reply_rank = _worker_dispatch_reply_rank(
        summary["scheduler_pipeline_depth_evidence"], len(first_observations)
    )
    late_non_reply_ranks: list[dict[str, Any]] = []
    for record in records:
        scheduled_tokens = record["scheduled_tokens"]
        request_tokens = sum(
            request["scheduled_tokens"]
            for request in record["scheduled_requests"]
        )
        if scheduled_tokens != request_tokens:
            raise ValueError("Scheduler row request tokens are inconsistent")
        if scheduled_tokens != (
            record["prefill_tokens"] + record["decode_tokens"]
        ):
            raise ValueError("Scheduler row prefill/decode tokens are inconsistent")
        if not (
            record["schedule_start_ns"] <= record["schedule_end_ns"]
            <= record["device_execution_start_ns"]
            <= record["device_execution_end_ns"]
            <= record["scheduler_update_start_ns"]
            <= record["scheduler_update_end_ns"]
        ):
            raise ValueError("Scheduler row stage timestamps are not causal")
        if record["device_execution_clock"] != "time.perf_counter_ns":
            raise ValueError("Scheduler row executor clock is invalid")
        if record["device_execution_source"] != (
            "host_monotonic_executor_rpc_envelope"
        ):
            raise ValueError("Scheduler row executor source is invalid")
        in_flight = record["in_flight_step_count"]
        if isinstance(in_flight, bool) or not isinstance(in_flight, int):
            raise TypeError("Scheduler row in-flight count must be an integer")
        if not 1 <= in_flight <= configured_depth:
            raise ValueError(
                "Scheduler row in-flight count exceeds configured depth")
        max_in_flight = max(max_in_flight, in_flight)
        observations = record["rank_observations"]
        if not isinstance(observations, list) or not observations:
            raise TypeError("A800 Scheduler row rank observations are invalid")
        ranks = set()
        for observation in observations:
            rank = observation["rank"]
            _validate_a800_rank_observation_for_summary(
                observation,
                expected_rank=rank,
                expected_step_id=record["step_id"],
                expected_scheduled_tokens=scheduled_tokens,
            )
            if rank in ranks:
                raise ValueError("A800 Scheduler row repeats a worker rank")
            ranks.add(rank)
            if not (
                record["device_execution_start_ns"]
                <= observation["worker_execution_start_ns"]
                <= observation["worker_execution_end_ns"]
            ):
                raise ValueError(
                    "A800 worker execution envelope is outside executor envelope"
                )
            end_after_executor_ns = (
                observation["worker_execution_end_ns"]
                - record["device_execution_end_ns"]
            )
            if end_after_executor_ns > 0:
                if rank == reply_rank:
                    raise ValueError(
                        "A800 worker execution envelope is outside executor "
                        "envelope"
                    )
                late_non_reply_ranks.append({
                    "step_id": record["step_id"],
                    "rank": rank,
                    "worker_end_after_executor_end_ns": end_after_executor_ns,
                })
            runtime_mode = observation["runtime_cudagraph_mode"]
            graph_mode_counts[runtime_mode] += 1
            fallback_reason = observation["graph_fallback_reason"]
            if runtime_mode == "NONE":
                fallback_counts[fallback_reason] = (
                    fallback_counts[fallback_reason] + 1
                    if fallback_reason in fallback_counts
                    else 1
                )
        if expected_ranks is None:
            expected_ranks = ranks
        elif ranks != expected_ranks:
            raise ValueError("A800 Scheduler row worker-rank coverage changed")
    if expected_ranks is None or expected_ranks != set(range(len(expected_ranks))):
        raise ValueError("A800 Scheduler trace ranks must be contiguous from zero")
    trace_path = Path(summary["path"])
    warmup_path = Path(f"{trace_path}.warmup.json")
    if not warmup_path.is_file():
        raise ValueError("Scheduler runtime warmup evidence is missing")
    warmup = json.loads(warmup_path.read_text())
    if warmup["max_scheduled_request_count"] < 8:
        raise ValueError("warmup did not schedule at least eight requests together")
    if warmup["mixed_prefill_decode_step_count"] < 1:
        raise ValueError("warmup did not exercise mixed prefill/decode")
    summary.update({
        "runtime_cudagraph_mode_counts": graph_mode_counts,
        "graph_fallback_counts": fallback_counts,
        "max_in_flight_step_count": max_in_flight,
        "worker_ranks": sorted(expected_ranks),
        "worker_execution_envelope": {
            "executor_reply_rank": reply_rank,
            "non_reply_rank_end_after_executor_end_count": len(
                late_non_reply_ranks),
            "non_reply_rank_end_after_executor_end_max_ns": (
                max(
                    late["worker_end_after_executor_end_ns"]
                    for late in late_non_reply_ranks
                )
                if late_non_reply_ranks
                else None
            ),
            "non_reply_rank_end_after_executor_end": late_non_reply_ranks,
        },
        "warmup_path": str(warmup_path.resolve()),
        "warmup_sha256": sha256_file(warmup_path),
        "warmup": warmup,
    })
    return summary


def _validate_a800_rank_observation_for_summary(
    observation: Any,
    *,
    expected_rank: int,
    expected_step_id: str,
    expected_scheduled_tokens: int,
) -> None:
    if not isinstance(observation, dict):
        raise TypeError("A800 Scheduler rank observation must be an object")
    required = {
        "schema_version",
        "source",
        "rank",
        "trace_step_id",
        "scheduled_tokens",
        "configured_cudagraph_mode",
        "runtime_cudagraph_mode",
        "dispatch_input_key",
        "selected_graph_key",
        "graph_fallback_reason",
        "num_unpadded_tokens",
        "padded_total_tokens",
        "padding_tokens",
        "worker_execution_start_ns",
        "worker_execution_end_ns",
        "worker_execution_clock",
        "worker_execution_source",
    }
    if set(observation) != required:
        raise ValueError("A800 Scheduler rank observation fields are invalid")
    if observation["schema_version"] != 1:
        raise ValueError("A800 Scheduler rank observation schema is invalid")
    if observation["source"] != "vllm_v1_worker_cudagraph_dispatch":
        raise ValueError("A800 Scheduler rank observation source is invalid")
    if observation["rank"] != expected_rank:
        raise ValueError("A800 Scheduler rank observation rank is invalid")
    if observation["trace_step_id"] != expected_step_id:
        raise ValueError("A800 Scheduler rank observation step is invalid")
    if observation["scheduled_tokens"] != expected_scheduled_tokens:
        raise ValueError("A800 Scheduler rank observation tokens are invalid")
    if observation["configured_cudagraph_mode"] != "FULL_AND_PIECEWISE":
        raise ValueError("A800 Scheduler configured CUDAGraph mode is invalid")
    runtime_mode = observation["runtime_cudagraph_mode"]
    if runtime_mode not in {"FULL", "PIECEWISE", "NONE"}:
        raise ValueError("A800 Scheduler runtime CUDAGraph mode is invalid")
    _validate_a800_graph_key(
        observation["dispatch_input_key"], "A800 dispatch input key"
    )
    selected_key = observation["selected_graph_key"]
    if runtime_mode == "NONE":
        if selected_key is not None:
            raise ValueError("NONE A800 Scheduler observation selected graph key")
        fallback_reason = observation["graph_fallback_reason"]
        if not isinstance(fallback_reason, str) or not fallback_reason:
            raise ValueError("NONE A800 Scheduler observation lacks fallback")
    elif runtime_mode in ("FULL", "PIECEWISE"):
        _validate_a800_graph_key(selected_key, "A800 selected graph key")
        if observation["graph_fallback_reason"] is not None:
            raise ValueError("Graph A800 Scheduler observation has fallback")
    else:
        raise ValueError(f"unsupported A800 runtime Graph mode: {runtime_mode!r}")
    for field in (
        "num_unpadded_tokens",
        "padded_total_tokens",
        "padding_tokens",
        "worker_execution_start_ns",
        "worker_execution_end_ns",
    ):
        value = observation[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TypeError(f"A800 Scheduler rank observation {field} is invalid")
    if observation["num_unpadded_tokens"] != expected_scheduled_tokens:
        raise ValueError("A800 Scheduler unpadded tokens disagree")
    if observation["padded_total_tokens"] != (
        expected_scheduled_tokens + observation["padding_tokens"]
    ):
        raise ValueError("A800 Scheduler padding tokens disagree")
    if observation["dispatch_input_key"]["num_tokens"] != observation[
        "padded_total_tokens"
    ]:
        raise ValueError("A800 Scheduler dispatch input key is inconsistent")
    if observation["worker_execution_clock"] != "time.perf_counter_ns":
        raise ValueError("A800 Scheduler worker clock is invalid")
    if observation["worker_execution_source"] != (
        "vllm_v1_worker_execute_model_envelope"
    ):
        raise ValueError("A800 Scheduler worker source is invalid")


def _validate_a800_graph_key(value: Any, description: str) -> None:
    if not isinstance(value, dict) or set(value) != {
        "num_tokens", "uniform_decode"
    }:
        raise TypeError(f"{description} fields are invalid")
    num_tokens = value["num_tokens"]
    if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 1:
        raise TypeError(f"{description} num_tokens is invalid")
    if not isinstance(value["uniform_decode"], bool):
        raise TypeError(f"{description} uniform_decode is invalid")


def _validate_benchmark_log_window(
    paths_and_offsets: Sequence[tuple[Path, int]],
) -> dict[str, Any]:
    warning = "JIT compilation during inference"
    files = []
    matches = []
    for path, start_offset in paths_and_offsets:
        data = path.read_bytes()
        if not 0 <= start_offset <= len(data):
            raise ValueError(f"invalid benchmark log offset for {path}")
        window = data[start_offset:].decode("utf-8", errors="replace")
        file_matches = [line for line in window.splitlines() if warning in line]
        files.append({
            "path": str(path.resolve()),
            "start_offset_bytes": start_offset,
            "end_offset_bytes": len(data),
            "sha256": sha256_file(path),
        })
        matches.extend(file_matches)
    if matches:
        raise RuntimeError(
            "formal benchmark window contains runtime JIT warnings: "
            + json.dumps(matches, ensure_ascii=False)
        )
    return {
        "status": "validated",
        "jit_warning_pattern": warning,
        "jit_warning_count": 0,
        "files": files,
    }


def _capture_a800_post_benchmark_counters(
    process_group_id: int,
    instrumentation: dict[str, Any],
) -> dict[str, Any]:
    worker_rank_count = instrumentation["worker_rank_count"]
    if (
        isinstance(worker_rank_count, bool)
        or not isinstance(worker_rank_count, int)
        or worker_rank_count < 1
    ):
        raise TypeError("A800 Scheduler trace worker rank count is invalid")
    trace_path = Path(instrumentation["output_path"])
    _signal_process_group(process_group_id, signal.SIGUSR1)
    deadline = time.monotonic() + 10.0
    while True:
        paths = [
            Path(f"{trace_path}.compilation-counter.rank-{rank}.jsonl")
            for rank in range(worker_rank_count)
        ]
        if all(path.is_file() and len(path.read_text().splitlines()) == 2 for path in paths):
            return _validate_a800_compilation_counters(paths)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "A800 worker compilation counters did not arrive after formal benchmark"
            )
        time.sleep(0.05)


def _vllm_server_process_group_for_port(
    port: int,
    *,
    proc_root: Path = Path("/proc"),
) -> int:
    """Resolve one live vLLM serve process group for an exclusive port.

    Nsight's fork-before-exec mode detaches the target server from the profiler
    launcher group. The runner reserves its port before launch, so the port is
    a fail-closed ownership key for both post-run worker signals and cleanup.
    """

    if isinstance(port, bool) or not isinstance(port, int) or not 0 < port < 65536:
        raise ValueError("vLLM server port must be between 1 and 65535")
    if not proc_root.is_dir():
        raise RuntimeError("vLLM server process discovery requires Linux /proc")
    expected_port = str(port)
    candidates: list[tuple[int, int]] = []
    for process_path in proc_root.iterdir():
        if not process_path.name.isdecimal():
            continue
        try:
            cmdline = (process_path / "cmdline").read_bytes().split(b"\0")
            stat = (process_path / "stat").read_bytes()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise RuntimeError(
                f"failed to inspect vLLM process at {process_path}"
            ) from error
        argv = [argument.decode("utf-8", errors="surrogateescape")
                for argument in cmdline if argument]
        if (
            not argv
            or not Path(argv[0]).name.startswith("python")
            or "vllm.entrypoints.cli.main" not in argv
            or "serve" not in argv
            or "--port" not in argv
        ):
            continue
        port_index = argv.index("--port")
        if port_index + 1 >= len(argv) or argv[port_index + 1] != expected_port:
            continue
        command_end = stat.rfind(b")")
        if command_end < 0:
            raise RuntimeError(f"invalid Linux process stat record at {process_path}")
        fields = stat[command_end + 1 :].split()
        if len(fields) < 3:
            raise RuntimeError(f"invalid process group in {process_path}")
        if fields[0] in (b"Z", b"X", b"x"):
            continue
        try:
            process_group_id = int(fields[2])
        except ValueError as error:
            raise RuntimeError(
                f"invalid process group in {process_path}"
            ) from error
        candidates.append((int(process_path.name), process_group_id))
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one live vLLM server for reserved port "
            f"{port}, found {candidates!r}"
        )
    return candidates[0][1]


def _wait_for_scheduler_trace_output(trace_path: str | Path) -> None:
    """Wait for the EngineCore's post-benchmark atomic trace publication."""

    path = Path(trace_path)
    deadline = time.monotonic() + 30.0
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Scheduler trace was not published within 30 seconds after "
                f"the formal benchmark: {path}"
            )
        time.sleep(0.05)


def _validate_a800_compilation_counters(
    paths: Sequence[Path],
) -> dict[str, Any]:
    rank_summaries = []
    for expected_rank, path in enumerate(paths):
        lines = path.read_text().splitlines()
        if len(lines) != 2:
            raise ValueError("A800 compilation counter file must have two snapshots")
        snapshots = [json.loads(line) for line in lines]
        before, after = snapshots
        for value, phase in (
            (before, "after_warmup_before_formal"),
            (after, "after_formal_benchmark"),
        ):
            if value["schema_version"] != 1:
                raise ValueError("A800 compilation counter schema version is invalid")
            if value["source"] != "vllm.compilation.counter.compilation_counter":
                raise ValueError("A800 compilation counter source is invalid")
            if value["rank"] != expected_rank:
                raise ValueError("A800 compilation counter rank disagrees with path")
            if value["phase"] != phase:
                raise ValueError("A800 compilation counter phase order is invalid")
            if not isinstance(value["counter"], dict):
                raise TypeError("A800 compilation counter snapshot must be an object")
        if not isinstance(before["trace_step_id"], str) or not before[
            "trace_step_id"
        ]:
            raise ValueError("A800 warmup counter snapshot lacks formal step identity")
        if after["trace_step_id"] is not None:
            raise ValueError("A800 post-benchmark counter snapshot has step identity")
        if before["timestamp_ns"] > after["timestamp_ns"]:
            raise ValueError("A800 compilation counter timestamps are not causal")
        before_counter = before["counter"]
        after_counter = after["counter"]
        if set(before_counter) != set(after_counter):
            raise ValueError("A800 compilation counter field sets differ")
        delta = {}
        for field in _A800_COMPILATION_COUNTER_FIELDS:
            before_value = before_counter[field]
            after_value = after_counter[field]
            if (
                isinstance(before_value, bool)
                or not isinstance(before_value, int)
                or before_value < 0
                or isinstance(after_value, bool)
                or not isinstance(after_value, int)
                or after_value < 0
            ):
                raise TypeError(
                    "A800 compilation counter value is not a non-negative int: "
                    f"{field}"
                )
            delta[field] = after_value - before_value
        nonzero = {
            field: value for field, value in delta.items() if value != 0
        }
        if nonzero:
            raise RuntimeError(
                "formal benchmark performed runtime compilation/capture: "
                + json.dumps(nonzero, sort_keys=True)
            )
        rank_summaries.append({
            "rank": expected_rank,
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "after_warmup": before_counter,
            "after_formal_benchmark": after_counter,
            "formal_delta": delta,
        })
    return {
        "status": "validated",
        "counter_fields": list(_A800_COMPILATION_COUNTER_FIELDS),
        "rank_snapshots": rank_summaries,
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    case = _load_case_contract(
        args.case_manifest,
        model_path=args.model,
        suite_manifest_path=args.suite_manifest,
    )
    if args.target_hardware_id is not None:
        case = _retarget_case_contract(
            case,
            target_hardware_id=args.target_hardware_id,
            target_runtime_version=args.runtime_version,
        )
    server_plan = _server_plan(args, case)
    result = build_ground_truth(
        model=args.model,
        dataset=args.dataset,
        seed=args.seed,
        case=case,
        server_plan=server_plan,
        dry_run=True,
    )
    output_path = Path(args.output)
    if args.dry_run:
        _write_result(output_path, result)
        return result

    _validate_no_profiler_environment()
    _assert_port_available("127.0.0.1", args.port)
    logs_dir = output_path.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "vllm_server.stdout.log"
    stderr_path = logs_dir / "vllm_server.stderr.log"
    server_environment = dict(os.environ)
    server_environment.update(server_plan["environment"])
    _preflight_scheduler_trace(server_plan, server_environment)
    process = None
    server_process_group_id = None
    stdout_handle = stdout_path.open("w")
    stderr_handle = stderr_path.open("w")
    result["status"] = "starting_server"
    result["server"]["runtime"] = {
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    _write_result(output_path, result)
    try:
        process = subprocess.Popen(
            server_plan["argv"],
            cwd=str(ROOT),
            env=server_environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            start_new_session=True,
        )
        result["server"]["runtime"]["pid"] = process.pid
        result["server"]["runtime"]["readiness"] = _wait_for_server(
            process,
            server_plan["endpoint"]["health_url"],
            expected_model_id=case["identity"]["model_id"],
            timeout_seconds=args.server_wait_seconds,
        )
        server_process_group_id = _vllm_server_process_group_for_port(args.port)
        result["server"]["runtime"]["server_process_group_id"] = (
            server_process_group_id
        )
        stdout_handle.flush()
        stderr_handle.flush()
        result["server"]["runtime"]["graph_validation"] = (
            _validate_graph_server_logs(
                stdout_path,
                stderr_path,
                hardware_id=server_plan["hardware_id"],
                engine_generation=_graph_log_engine_generation(server_plan),
            )
        )
        if _requires_a800_async_scheduling_validation(
                server_plan["hardware_id"],
                server_plan["engine"]["async_scheduling"]) or (
                    server_plan["hardware_id"] == "nvidia_h200"
                    and server_plan["engine"]["runtime_version"] == "0.11.0"):
            # H200 0.11.0 runs the A800 runtime release; its executor logs
            # carry the same async-scheduling markers.
            result["server"]["runtime"]["async_scheduling_validation"] = (
                _validate_a800_async_scheduling_logs(
                    stdout_path,
                    stderr_path,
                )
            )
        _write_result(output_path, result)
        benchmark_log_offsets: list[tuple[Path, int]] = []

        def mark_benchmark_boundary() -> None:
            stdout_handle.flush()
            stderr_handle.flush()
            benchmark_log_offsets.extend((
                (stdout_path, stdout_path.stat().st_size),
                (stderr_path, stderr_path.stat().st_size),
            ))

        measured = build_ground_truth(
            model=args.model,
            dataset=args.dataset,
            seed=args.seed,
            case=case,
            server_plan=server_plan,
            dry_run=False,
            benchmark_boundary_callback=mark_benchmark_boundary,
        )
        measured["server"]["runtime"] = result["server"]["runtime"]
        result = measured
        stdout_handle.flush()
        stderr_handle.flush()
        if len(benchmark_log_offsets) != 2:
            raise RuntimeError("formal benchmark log boundary was not recorded")
        result["server"]["runtime"]["benchmark_log_window"] = (
            _validate_benchmark_log_window(benchmark_log_offsets)
        )
        scheduler_trace = server_plan["scheduler_trace"]
        counter_instrumentation = server_plan[
            "compilation_counter_instrumentation"
        ]
        if counter_instrumentation["enabled"]:
            result["server"]["runtime"]["compilation_counters"] = (
                _capture_a800_post_benchmark_counters(
                    server_process_group_id,
                    counter_instrumentation,
                )
            )
        if scheduler_trace["enabled"]:
            if server_plan["hardware_id"] == "metax_c500":
                max_result_consumption_gap_ns = 1_000_000_000
            elif server_plan["hardware_id"] in (
                "nvidia_a800", "nvidia_h200", "ascend_910b"
            ):
                max_result_consumption_gap_ns = None
            else:
                raise ValueError(
                    "unsupported Scheduler trace hardware: "
                    f"{server_plan['hardware_id']!r}"
                )
            _wait_for_scheduler_trace_output(scheduler_trace["output_path"])
            result["scheduler_trace"] = _scheduler_trace_summary(
                scheduler_trace["output_path"],
                case["requests_row_count"],
                max_result_consumption_gap_ns=max_result_consumption_gap_ns,
            )
            benchmark_clock = result["metrics"]["benchmark_clock"]
            scheduler_clock = result["scheduler_trace"]["clock"]
            if (
                benchmark_clock["clock_domain"]
                != scheduler_clock["clock_domain"]
                or benchmark_clock["host_boot_id"]
                != scheduler_clock["host_boot_id"]
            ):
                raise RuntimeError(
                    "benchmark and Scheduler trace do not share one host clock"
                )
            result["server"]["runtime"]["scheduler_pipeline_depth"] = (
                result["scheduler_trace"]["scheduler_pipeline_depth"]
            )
            result["server"]["runtime"][
                "scheduler_pipeline_depth_evidence"
            ] = result["scheduler_trace"]["scheduler_pipeline_depth_evidence"]
            if (
                server_plan["hardware_id"] == "metax_c500"
                and result["scheduler_trace"]["max_in_flight_step_count"] != 2
            ):
                raise RuntimeError(
                    "C500 formal benchmark never reached async depth=2"
                )
        else:
            result["scheduler_trace"] = None
        if server_plan["hardware_id"] == "ascend_910b":
            stdout_handle.flush()
            stderr_handle.flush()
            result["server"]["runtime"]["graph_validation"] = (
                _validate_graph_server_logs(
                    stdout_path,
                    stderr_path,
                    hardware_id=server_plan["hardware_id"],
                    engine_generation=server_plan["engine"][
                        "vllm_engine_generation"
                    ],
                    require_runtime_replay=True,
                )
            )
        max_dispatch_lag_ms = result["metrics"]["max_dispatch_lag_ms"]
        if max_dispatch_lag_ms > server_plan["max_dispatch_lag_ms"]:
            raise RuntimeError(
                "canonical arrival replay exceeded dispatch-lag limit: "
                f"{max_dispatch_lag_ms:.3f} ms > "
                f"{server_plan['max_dispatch_lag_ms']:.3f} ms"
            )
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        raise
    finally:
        if process is not None:
            exited_before_cleanup = process.poll() is not None
            if (
                server_process_group_id is not None
                and server_process_group_id != process.pid
            ):
                _terminate_process_group(server_process_group_id)
            termination_returncode = _terminate_server(process)
            result["server"]["runtime"]["termination_returncode"] = (
                termination_returncode
            )
            result["server"]["runtime"]["exited_before_cleanup"] = (
                exited_before_cleanup
            )
            if exited_before_cleanup and result["status"] == "success":
                result["status"] = "failed"
                result["failure"] = {
                    "type": "ServerExitedEarly",
                    "message": (
                        "vLLM server exited before runner-managed cleanup with "
                        f"code {termination_returncode}"
                    ),
                }
        stdout_handle.close()
        stderr_handle.close()
        if stdout_path.exists():
            result["server"]["runtime"]["stdout_sha256"] = sha256_file(
                stdout_path
            )
        if stderr_path.exists():
            result["server"]["runtime"]["stderr_sha256"] = sha256_file(
                stderr_path
            )
        _write_result(output_path, result)
    if result["status"] != "success":
        raise RuntimeError(result["failure"]["message"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = _run(args)
    print(json.dumps({
        "status": result["status"],
        "output": args.output,
        "metrics": _metric_summary(result),
    }, indent=2, sort_keys=True))
    return 0
