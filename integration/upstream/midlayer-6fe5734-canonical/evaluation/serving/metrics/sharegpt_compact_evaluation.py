"""First-phase ShareGPT evaluation against validated compact cases.

The compact schema remains owned by :mod:`dataset_pipeline`.  This module only
accepts objects that the CLI has already validated and never imports that
owner.  Observations are consumed only after the current production Forward
has returned. Persisted request-level values are limited to synthetic request
indexes and paired latency scalars needed for evaluation plots.
"""

from __future__ import annotations

import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import resource
import statistics
import subprocess
import time
import types
from typing import (
    Callable,
    get_args,
    get_origin,
    get_type_hints,
    Mapping,
    Protocol,
    Sequence,
)

from evaluation.serving.metrics.compact_step_alignment import align_compact_case
from evaluation.serving.metrics.latency_distribution import (
    latency_distribution_manifest,
    linear_percentile,
)
from evaluation.serving.metrics.sharegpt_evaluation_report import (
    STATISTICS,
    write_sharegpt_evaluation_report,
)
from evaluation.serving.metrics.scheduler_progress import (
    validate_scheduler_progress_manifest,
)
from evaluation.serving.experiments.experiment_manifest import (
    RUNTIME_MEMORY_POLICY_PATH_TEMPLATE,
    resolve_runtime_memory_policy_path,
)
from evaluation.serving.simulation.execution_artifact import (
    DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS,
    performance_timing_status,
    PROJECTED_PERFORMANCE_TIMING_STATUS,
    UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
)
from src.serving.backend.vllm_scheduler_process import (
    VLLMSchedulerRuntimeInfo,
    VLLM_SCHEDULER_SOURCE_COMMIT,
    VLLM_SCHEDULER_VERSION,
    vllm_scheduler_runtime_sha256,
    vllm_scheduler_source_identity,
)
from src.kernel.predictors.protobridge.adapter import (
    development_profile_config_sha256,
)
from src.serving.workload.expert_frequency import (
    load_expert_frequency_profile,
)
from src.serving.workload.graph_padding import GraphExecutionConfig
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import ExpertRoutingInput
from src.serving.metrics.serving_metrics import RequestMetrics
from src.serving.host_timing import (
    SERVING_TIMING_MODES,
    load_host_launch_profile,
    load_host_request_path_profile,
    load_scheduler_host_profile,
    validate_serving_timing_mode,
)
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS

from evaluation.reporting.standard_evaluation_report import (
    MIDLAYER_WALL_TIME_PHASES,
)

from evaluation.json_shape import require_object_value as _mapping


_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SUITE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_CASE_ID = re.compile(
    r"tp(?P<tp>[1-9][0-9]*)(?:_ep(?P<ep>[1-9][0-9]*))?"
    r"_qps(?P<qps>[1-9][0-9]*)"
)
_MAX_CASE_JOBS = 64
_KNOWN_A800_GEMM_LAUNCH_GAP = re.compile(
    r"^uncharacterized NVIDIA A800 GEMM launch key \(.+\)$"
)
_KNOWN_METAX_GEMM_LAUNCH_GAP = re.compile(
    r"^uncharacterized MetaX C500 GEMM launch key \(.+\)$"
)
_KNOWN_ASCEND_LAUNCH_GAP = re.compile(
    r"^uncharacterized Ascend 910B (?:FLASH_ATTN|SwiGLU) workload: .+$"
)
# Hardware the frozen phase-1 contract accepts; an id outside it is a
# contract violation rather than an absent gap.
_PHASE1_HARDWARE_IDS = frozenset({
    "nvidia_a800", "nvidia_h200", "metax_c500", "ascend_910b",
})
_KNOWN_METAX_VECTOR_PREFIX = "MetaX C500 production Vector launch is unsupported for "
_KNOWN_METAX_VECTOR_SUFFIX = (
    ": no MACA 3.7.2 device source or complete versioned launch fact is retained"
)
_KNOWN_METAX_FLASH_ATTN_GAP = (
    "MetaX C500 MACA 3.7.2 production FlashAttention launch is "
    "evidence_pending: the installed library has no auditable "
    "default-solution export, while retained 3.2.1 observations only bind "
    "evaluation grid/block/split; NVIDIA FA2 launch rules cannot be substituted"
)
_KNOWN_A800_FA2_COMBINE_GAP = re.compile(
    r"^FA2 combine dependency evidence has no unique source fragment for "
    r"block_m=[1-9][0-9]*, head_dim=[1-9][0-9]*, "
    r"num_splits=[1-9][0-9]*, max_splits=[1-9][0-9]*$"
)
_KNOWN_A800_ADD_RMS_QUEUED_SOURCE_GAP = (
    "A800 block-256 AddRMSNorm queued CTA source service is unavailable"
)
_KNOWN_EXPERT_PARALLEL_LAUNCH_GAPS = {
    "metax_c500": (
        "MetaX C500 FUSED_MOE expert-parallel launch is not source-backed"
    ),
    "ascend_910b": (
        "Ascend 910B FUSED_MOE expert-parallel launch is not source-backed"
    ),
}
EXECUTION_ARTIFACT_FIELDS = {
    "schema_version",
    "hash_method",
    "performance_timing_status",
    "counts",
    "schedule_trace",
    "steps",
    "transfers",
    "modeled_kv_offload",
    "artifact_sha256",
}
_STREAMING_EXECUTION_ARTIFACT_FIELDS = {
    "schema_version",
    "hash_method",
    "performance_timing_status",
    "counts",
    "streaming_aggregate",
    "artifact_sha256",
}
_EXECUTION_ARTIFACT_COUNT_FIELDS = {
    "schedule_steps",
    "simulation_steps",
    "simulation_transfers",
    "trace_transfers",
    "modeled_kv_offload_transfers",
    "kernel_calls",
}
_EXECUTION_ARTIFACT_HASH_METHOD = "canonical_json_sort_keys_compact_utf8_v1"
STREAMING_EXECUTION_ARTIFACT_HASH_METHODS = {
    9: "ordered_step_manifest_sha256_chain_v1",
    10: "ordered_exact_execution_contract_sha256_chain_v2",
    11: "ordered_exact_execution_contract_sha256_chain_v3",
    12: "ordered_exact_execution_contract_sha256_chain_v4",
    13: "ordered_exact_execution_contract_sha256_chain_v5",
}
_MAIN_CHAIN = (
    "RequestStream -> SchedulerBackend/BackendStep/ScheduleStep "
    "-> build_workload -> WorkloadGraph -> lower_workload -> "
    "KernelSpec -> predict"
)
_MODEL_FIELDS = {
    "dataset_model_id",
    "simulation_model_path",
    "model_config_sha256",
}
_HARDWARE_FIELDS = {
    "hardware_id",
    "hardware_config_path",
    "hardware_config_sha256",
}
_GRAPH_HARDWARE_FIELDS = _HARDWARE_FIELDS | {"graph_execution"}
_GRAPH_EXECUTION_FIELDS = {
    "configured_mode",
    "uniform_decode_query_len",
    "capture_sizes",
    "piecewise_eager_workload_kinds",
}
_PREDICTION_FIELDS = {
    "semantics",
    "backend",
    "scheduler_version",
    "scheduler_source_commit",
    "deployment",
    "compute_dtype",
    "prediction_method",
    "tensor_collective_backend",
    "tensor_all_reduce_profile_sha256",
    "tensor_all_gather_profile_sha256",
    "launch_policy",
    "tensor_parallel_size",
    "pipeline_parallel_size",
    "expert_parallel_size",
    "max_model_len",
    "max_batched_tokens",
    "max_requests",
    "runtime_memory_policy_path",
    "kv_block_size_tokens",
    "attention_kv_layout",
    "attention_kv_distribution",
    "enable_chunked_prefill",
    "enable_prefix_caching",
    "router_resource_id",
    "router_dispatch_service_time_s",
    "mtp_num_speculative_tokens",
    "mtp_acceptance_policy",
    "mtp_fixed_accepted_tokens",
    "mtp_acceptance_source_id",
}
_CONFIG_FIELDS = {
    "schema_version",
    "suite_id",
    "dataset_root",
    "models",
    "hardware",
    "qps_values",
    "cells",
    "prediction_contract",
}
_CELL_FIELDS = {
    "hardware_id",
    "model_id",
    "case_id",
    "manifest_sha256",
    "requests_sha256",
}


_CANONICAL_MODEL_SPECS = {
    "qwen3-4b": (
        "configs/model/qwen3-4b.json",
        "8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a",
    ),
    "qwen3-8b": (
        "configs/model/qwen3-8b.json",
        "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    ),
    "llama-3.1-8b": (
        "configs/model/llama-3.1-8b-instruct.json",
        "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    ),
    "qwen3-30b-a3b": (
        "configs/model/qwen3-30b-a3b.json",
        "2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297",
    ),
    "qwen3-32b": (
        "configs/model/qwen3-32b.json",
        "97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb",
    ),
    "llama-3-70b": (
        "${MODELS_ROOT}/Meta-Llama-3-70B-Instruct",
        "bfd937677ee1c3d5b535ccaeec3223c06eca9452f3ac0dabe2ff87deb0f818a4",
    ),
    "llama-3.1-70b": (
        "${MODELS_ROOT}/Llama-3.1-70B-Instruct",
        "fa6e9124e4621df77aecf96fbfaf7975814013d2d5ab1c972e965000588a9749",
    ),
}
_MODEL_CONFIG_ALTERNATES = {
    "qwen3-32b": (
        (
            "configs/model/qwen3-32b-fp16.json",
            "8150eeaa087d37397bde1f2f128a6d120916e078bc51b1c7e95fc5a6b9905316",
        ),
    ),
}
_LEGACY_MODEL_IDS = {
    "Qwen3-4B": "qwen3-4b",
    "Qwen3-8B": "qwen3-8b",
    "Llama-3.1-8B": "llama-3.1-8b",
    "Qwen3-30B-A3B": "qwen3-30b-a3b",
    "Qwen3-32B": "qwen3-32b",
}
_EXPECTED_MODELS = {
    **_CANONICAL_MODEL_SPECS,
    **{
        legacy: _CANONICAL_MODEL_SPECS[canonical]
        for legacy, canonical in _LEGACY_MODEL_IDS.items()
    },
}
_EXPECTED_HARDWARE = {
    "nvidia_a800": (
        "configs/hardware/nvidia_a800.yaml",
        "4fd8c3cbed54268219dc8dac76ed51d747f53b5d0c7d0fdb2033454979f2bf0c",
    ),
    "metax_c500": (
        "configs/hardware/metax_c500.yaml",
        "3eff3be984ab4c42acce85efb053155760685a95b0b08e6544c479cf5ff8b929",
    ),
    "ascend_910b": (
        "configs/hardware/ascend_910b.yaml",
        "e0fe91381d31955cecc29c97484c959f5a4bf8ce941de4f38c47c0c42b48eff0",
    ),
    "nvidia_h200": (
        "configs/hardware/nvidia_h200.yaml",
        "1b67bb9e4c555976a9d96386967bf50ca36bf11287ad33aedff5adca8ebf406a",
    ),
}
# The frozen 45-cell official matrix and the phase-1 suite cover these three
# hardware; H200 cells join a run only through an explicit vLLM version below.
_OFFICIAL_HARDWARE = EVALUATED_HARDWARE_IDS
_EXPECTED_PREDICTION = {
    "semantics": "current_pinned_vllm_not_capture_replay",
    "backend": "vllm",
    "scheduler_version": VLLM_SCHEDULER_VERSION,
    "scheduler_source_commit": VLLM_SCHEDULER_SOURCE_COMMIT,
    "deployment": "colocated",
    "compute_dtype": "bf16",
    "prediction_method": "midlayer",
    "tensor_collective_backend": "protobridge",
    "tensor_all_reduce_profile_sha256": None,
    "tensor_all_gather_profile_sha256": None,
    "launch_policy": "injected_launch",
    "tensor_parallel_size": 1,
    "pipeline_parallel_size": 1,
    "expert_parallel_size": 1,
    "max_model_len": 8192,
    "max_batched_tokens": 8192,
    "max_requests": 256,
    "runtime_memory_policy_path": RUNTIME_MEMORY_POLICY_PATH_TEMPLATE,
    "kv_block_size_tokens": 16,
    "attention_kv_layout": "paged_varlen",
    "attention_kv_distribution": "uniform",
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
    "router_resource_id": "router_cpu",
    "router_dispatch_service_time_s": 0.0,
    "mtp_num_speculative_tokens": 0,
    "mtp_acceptance_policy": "disabled",
    "mtp_fixed_accepted_tokens": 0,
    "mtp_acceptance_source_id": "disabled",
}
OFFICIAL_METRICS_SUITE_SELECTIONS = (
    (
        "configs/evaluation/experiments/sharegpt_phase1.json",
        ("nvidia_a800", "metax_c500"),
    ),
    (
        "configs/evaluation/experiments/ascend_prefix_cache_phase1_suite.json",
        ("ascend_910b",),
    ),
    ("configs/evaluation/experiments/nvidia_a800_qwen3_30b_a3b_suite.json", None),
    ("configs/evaluation/experiments/metax_c500_qwen3_30b_a3b_suite.json", None),
    ("configs/evaluation/experiments/ascend_qwen3_30b_a3b_suite.json", None),
    ("configs/evaluation/experiments/qwen3_32b_qps1_suite.json", None),
    ("configs/evaluation/experiments/qwen3_32b_qps4_qps16_suite.json", None),
    ("configs/evaluation/experiments/metax_c500_qwen3_32b_suite.json", None),
    ("configs/evaluation/experiments/ascend_qwen3_32b_suite.json", None),
)
# H200 ShareGPT suites per captured vLLM version.  Their cells pin candidate
# datasets outside serving_dataset/, so they are evaluated only when the
# caller selects one version and supplies its candidate dataset root.
NVIDIA_H200_SUITE_SELECTIONS = {
    version: tuple(
        f"configs/evaluation/experiments/nvidia_h200_{tag}_{group}_suite.json"
        for group in ("sharegpt_tp1", "qwen3_30b_a3b", "qwen3_32b")
    )
    for version, tag in (("0.30.0", "vllm030"), ("0.11.0", "vllm011"))
}
# Hardware whose Host timing profiles exist once per captured vLLM version:
# every profile family is bound by the case's recorded runtime version.
_HOST_PROFILE_VARIANTS = {
    "nvidia_h200": {"0.30.0": "nvidia_h200_vllm030", "0.11.0": "nvidia_h200_vllm011"},
}
_HOST_PROFILE_DIRECTORIES = {
    "host_launch": ("host_launch_profiles", ""),
    "scheduler_host": ("scheduler_host_profiles", ""),
    "host_request_path": ("host_request_path_profiles", ""),
    "host_runtime_prepare": ("host_runtime_prepare_profiles", "_eager"),
}
_MOE_MODEL_IDS = {"qwen3-30b-a3b", "Qwen3-30B-A3B"}
_EXPERT_FREQUENCY_MANIFEST = Path(
    "serving_dataset/moe_expert_freq/manifest.json")
_LATENCY_FIELDS = {
    "ttft_ms": "ttft_s",
    "tpot_ms": "tpot_s",
    "e2e_latency_ms": "latency_s",
}
_THROUGHPUT_FIELDS = {
    "request_throughput_per_s": "request_throughput_per_s",
    "output_token_throughput_per_s": "output_token_throughput_per_s",
    "total_token_throughput_per_s": "token_throughput_per_s",
}


class ValidatedCompactCase(Protocol):
    case_dir: Path
    manifest: Mapping[str, object]
    scheduler_pipeline_depth: int
    requests: Sequence[object]
    metrics: Mapping[str, object] | None
    scheduler_steps: Sequence[Mapping[str, object]] | None
    alignment_steps: Mapping[str, object | None]
    operator_signatures: Mapping[str, Mapping[str, object]]
    launch_signatures: Mapping[str, Mapping[str, object]]


def scheduler_pipeline_depth_from_validated_case(
    case: ValidatedCompactCase,
) -> int:
    """Consume Scheduler capacity normalized by the dataset boundary."""

    return case.scheduler_pipeline_depth


def scheduler_async_scheduling_from_validated_case(
    case: ValidatedCompactCase,
) -> bool:
    """Resolve vLLM async scheduling from the observed executor capacity."""

    manifest = _mapping(case.manifest, "compact manifest")
    if manifest["schema_version"] != 5:
        raise ValueError(
            "runtime Scheduler evidence is required to resolve async mode"
        )
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    depth = scheduler_pipeline_depth_from_validated_case(case)
    pipeline_parallel_size = runtime["pipeline_parallel_size"]
    if (isinstance(pipeline_parallel_size, bool)
            or not isinstance(pipeline_parallel_size, int)
            or pipeline_parallel_size < 1):
        raise ValueError(
            "runtime.pipeline_parallel_size must be a positive non-bool integer"
        )
    evidence = _mapping(
        runtime["scheduler_pipeline_depth_evidence"],
        "runtime scheduler_pipeline_depth_evidence",
    )
    if evidence["pipeline_parallel_size"] != pipeline_parallel_size:
        raise ValueError(
            "Scheduler depth evidence pipeline size disagrees with runtime"
        )
    if pipeline_parallel_size != 1:
        raise NotImplementedError(
            "compact runtime must explicitly record async scheduling when "
            "pipeline_parallel_size exceeds one"
        )
    if depth not in (1, 2):
        raise ValueError(
            "observed PP=1 vLLM executor capacity must be one or two"
        )
    return depth == 2


@dataclass(frozen=True)
class ModelEntry:
    dataset_model_id: str
    simulation_model_path: str
    model_config_sha256: str


@dataclass(frozen=True)
class HardwareEntry:
    hardware_id: str
    hardware_config_path: str
    hardware_config_sha256: str
    graph_execution: GraphExecutionConfig


@dataclass(frozen=True)
class PredictionContract:
    values: Mapping[str, object]


@dataclass(frozen=True)
class Phase1Config:
    source_path: Path
    source_sha256: str
    suite_id: str
    dataset_root: str
    models: tuple[ModelEntry, ...]
    hardware: tuple[HardwareEntry, ...]
    qps_values: tuple[int, ...]
    cell_order: tuple[str, ...]
    cell_hashes: Mapping[str, tuple[str, str]]
    prediction: PredictionContract


@dataclass(frozen=True)
class Phase1Cell:
    cell_id: str
    model: ModelEntry
    hardware: HardwareEntry
    qps: int
    case_id: str
    manifest_sha256: str
    requests_sha256: str


def _expected_expert_routing(
    repository_root: str | Path,
    cell: Phase1Cell,
) -> ExpertRoutingInput:
    if cell.model.dataset_model_id in _MOE_MODEL_IDS:
        model_path = _model_config_path(
            repository_root, cell.model.simulation_model_path)
        model = load_model_config(model_path)
        if model.model_id is None:
            executing_model_id = model_path.stem
        else:
            executing_model_id = model.model_id
        profile = load_expert_frequency_profile(
            Path(repository_root).resolve() / _EXPERT_FREQUENCY_MANIFEST,
            cell.model.dataset_model_id,
            expected_model_id=executing_model_id,
            expected_moe_layers=sum(
                model.layer_uses_moe(layer_index)
                for layer_index in range(model.num_hidden_layers)
            ),
            expected_num_experts=model.num_experts,
        )
        return ExpertRoutingInput.profile_observation(profile)
    return ExpertRoutingInput.not_applicable()


def load_phase1_config(path: str | Path) -> Phase1Config:
    """Load a closed compact-suite contract, including the phase-1 suite."""

    source_path = Path(path)
    source_bytes = source_path.read_bytes()
    try:
        payload = json.loads(source_bytes, object_pairs_hook=_unique_json_object)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid phase-1 JSON config: {error}") from error
    root = _mapping(payload, "phase-1 config")
    _exact_fields(root, _CONFIG_FIELDS, "phase-1 config")
    if (type(root["schema_version"]) is not int
            or root["schema_version"] not in (2, 4)):
        raise ValueError("phase-1 config schema_version must equal 2 or 4")
    schema_version = root["schema_version"]
    suite_id = _string(root["suite_id"], "suite_id")
    if _SUITE_ID.fullmatch(suite_id) is None:
        raise ValueError("suite_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    if root["dataset_root"] != "serving_dataset/sharegpt":
        raise ValueError(
            "phase-1 dataset_root must equal 'serving_dataset/sharegpt'"
        )

    raw_models = _sequence(root["models"], "phase-1 models")
    models = []
    for index, raw_model in enumerate(raw_models):
        model = _mapping(raw_model, f"phase-1 model {index}")
        _exact_fields(model, _MODEL_FIELDS, f"phase-1 model {index}")
        model_id = _string(model["dataset_model_id"], "dataset_model_id")
        path_value = _string(
            model["simulation_model_path"], "simulation_model_path"
        )
        digest = _sha256(model["model_config_sha256"], "model config SHA256")
        if model_id not in _EXPECTED_MODELS:
            raise ValueError(f"unknown compact-suite model: {model_id!r}")
        alternates = ()
        if model_id in _MODEL_CONFIG_ALTERNATES:
            alternates = _MODEL_CONFIG_ALTERNATES[model_id]
        expected_model_specs = (_EXPECTED_MODELS[model_id], *alternates)
        if (path_value, digest) not in expected_model_specs:
            raise ValueError(
                f"phase-1 model mapping disagrees with frozen contract: {model_id}"
            )
        models.append(ModelEntry(model_id, path_value, digest))
    if not models:
        raise ValueError("compact suite must contain at least one model")
    if len(models) != len({model.dataset_model_id for model in models}):
        raise ValueError("compact-suite models contain duplicates")

    raw_hardware = _sequence(root["hardware"], "phase-1 hardware")
    hardware_entries = []
    for index, raw_entry in enumerate(raw_hardware):
        entry = _mapping(raw_entry, f"phase-1 hardware {index}")
        expected_hardware_fields = (
            _HARDWARE_FIELDS if schema_version == 2 else
            _GRAPH_HARDWARE_FIELDS)
        _exact_fields(
            entry, expected_hardware_fields, f"phase-1 hardware {index}")
        hardware_id = _string(entry["hardware_id"], "hardware_id")
        config_path = _relative_path(
            entry["hardware_config_path"], "hardware_config_path"
        )
        digest = _sha256(
            entry["hardware_config_sha256"], "hardware config SHA256"
        )
        if hardware_id not in _EXPECTED_HARDWARE:
            raise ValueError(f"unknown phase-1 hardware: {hardware_id!r}")
        if config_path != _EXPECTED_HARDWARE[hardware_id][0]:
            raise ValueError(
                "phase-1 hardware mapping disagrees with frozen contract: "
                f"{hardware_id}"
            )
        if schema_version == 2:
            graph_execution = GraphExecutionConfig.disabled()
        else:
            graph = _mapping(
                entry["graph_execution"],
                f"phase-1 hardware {index} graph_execution",
            )
            _exact_fields(
                graph,
                _GRAPH_EXECUTION_FIELDS,
                f"phase-1 hardware {index} graph_execution",
            )
            graph_execution = GraphExecutionConfig(
                _string(graph["configured_mode"], "configured graph mode"),
                tuple(
                    _integer(value, "graph capture size")
                    for value in _sequence(
                        graph["capture_sizes"], "graph capture sizes")
                ),
                tuple(
                    _string(value, "piecewise eager workload kind")
                    for value in _sequence(
                        graph["piecewise_eager_workload_kinds"],
                        "piecewise eager workload kinds",
                    )
                ),
                _integer(
                    graph["uniform_decode_query_len"],
                    "uniform decode query length",
                ),
            )
        hardware_entries.append(HardwareEntry(
            hardware_id,
            config_path,
            digest,
            graph_execution,
        ))
    if not hardware_entries:
        raise ValueError("compact suite must contain at least one hardware entry")
    if len(hardware_entries) != len(
        {entry.hardware_id for entry in hardware_entries}
    ):
        raise ValueError("compact-suite hardware entries contain duplicates")

    qps_values = _sequence(root["qps_values"], "phase-1 qps_values")
    if any(type(value) is not int for value in qps_values):
        raise TypeError("phase-1 qps_values must contain exact integers")
    if not qps_values or any(value <= 0 for value in qps_values):
        raise ValueError("compact-suite qps_values must be positive")
    if len(qps_values) != len(set(qps_values)):
        raise ValueError("compact-suite qps_values contain duplicates")

    raw_cells = _sequence(root["cells"], "phase-1 cells")
    cell_hashes = {}
    cell_order = []
    for index, raw_cell in enumerate(raw_cells):
        cell = _mapping(raw_cell, f"phase-1 cell {index}")
        _exact_fields(cell, _CELL_FIELDS, f"phase-1 cell {index}")
        hardware_id = _string(cell["hardware_id"], "cell hardware_id")
        model_id = _string(cell["model_id"], "cell model_id")
        case_id = _string(cell["case_id"], "cell case_id")
        if hardware_id not in {entry.hardware_id for entry in hardware_entries}:
            raise ValueError(f"cell uses undeclared hardware: {hardware_id!r}")
        if model_id not in {entry.dataset_model_id for entry in models}:
            raise ValueError(f"cell uses undeclared model: {model_id!r}")
        case_match = _CASE_ID.fullmatch(case_id)
        if case_match is None:
            raise ValueError(f"cell case_id is not canonical: {case_id!r}")
        if int(case_match.group("qps")) not in qps_values:
            raise ValueError(f"cell case_id QPS is absent from qps_values: {case_id!r}")
        cell_id = f"{hardware_id}/{model_id}/{case_id}"
        if cell_id in cell_hashes:
            raise ValueError(f"duplicate phase-1 cell: {cell_id}")
        cell_order.append(cell_id)
        cell_hashes[cell_id] = (
            _sha256(cell["manifest_sha256"], "cell manifest SHA256"),
            _sha256(cell["requests_sha256"], "cell requests SHA256"),
        )

    prediction = _mapping(root["prediction_contract"], "prediction contract")
    _exact_fields(prediction, _PREDICTION_FIELDS, "prediction contract")
    _validate_prediction_contract(prediction)
    collective_backend = prediction["tensor_collective_backend"]
    if collective_backend == "protobridge":
        configured_profiles = (
            prediction["tensor_all_reduce_profile_sha256"],
            prediction["tensor_all_gather_profile_sha256"],
        )
        if (prediction["tensor_parallel_size"] == 1
                and configured_profiles == (None, None)):
            pass
        elif len(hardware_entries) != 1:
            raise ValueError(
                "ProtoBridge compact suite requires exactly one hardware "
                "profile")
        else:
            hardware_id = hardware_entries[0].hardware_id
            expected_profiles = (
                development_profile_config_sha256(hardware_id, "all_reduce"),
                development_profile_config_sha256(hardware_id, "all_gather"),
            )
            if configured_profiles != expected_profiles:
                raise ValueError(
                    "ProtoBridge prediction profiles disagree with the "
                    "declared hardware and operations")
    elif collective_backend == "analytical_roofline_v1":
        pass
    else:
        raise ValueError(
            f"unknown tensor collective backend: {collective_backend!r}")
    cell_qps_values = set()
    for cell_id in cell_order:
        case_id = cell_id.rsplit("/", maxsplit=1)[1]
        _validate_case_id_contract(case_id, prediction)
        case_match = _CASE_ID.fullmatch(case_id)
        if case_match is None:
            raise AssertionError("validated case ID unexpectedly became invalid")
        cell_qps_values.add(int(case_match.group("qps")))
    if cell_qps_values != set(qps_values):
        raise ValueError("cell case_id QPS values disagree with qps_values")
    if not cell_order:
        raise ValueError("compact suite must contain at least one cell")
    if suite_id == "sharegpt_phase1":
        expected_models = ("qwen3-4b", "qwen3-8b", "llama-3.1-8b")
        actual_models = tuple(
            _LEGACY_MODEL_IDS[model.dataset_model_id]
            if model.dataset_model_id in _LEGACY_MODEL_IDS
            else model.dataset_model_id
            for model in models
        )
        if actual_models != expected_models:
            raise ValueError("phase-1 models must contain the frozen ordered model set")
        if tuple(entry.hardware_id for entry in hardware_entries) != (
            _OFFICIAL_HARDWARE
        ):
            raise ValueError("phase-1 hardware must contain the frozen ordered set")
        if tuple(qps_values) != (4, 16, 64):
            raise ValueError("phase-1 qps_values must equal [4, 16, 64]")
        for field, expected in _EXPECTED_PREDICTION.items():
            actual = prediction[field]
            if type(actual) is not type(expected) or actual != expected:
                raise ValueError(
                    "prediction contract disagrees with the frozen baseline at "
                    f"{field}"
                )
        expected_cell_order = [
            f"{hardware.hardware_id}/{model.dataset_model_id}/tp1_qps{qps}"
            for hardware in hardware_entries
            for model in models
            for qps in qps_values
        ]
        if cell_order != expected_cell_order:
            raise ValueError(
                "phase-1 cells must equal the frozen ordered 27-cell matrix"
            )
    return Phase1Config(
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        suite_id=suite_id,
        dataset_root=root["dataset_root"],
        models=tuple(models),
        hardware=tuple(hardware_entries),
        qps_values=tuple(qps_values),
        cell_order=tuple(cell_order),
        cell_hashes=cell_hashes,
        prediction=PredictionContract(dict(prediction)),
    )


def phase1_cells(config: Phase1Config) -> tuple[Phase1Cell, ...]:
    """Return compact-suite cells in their declared stable order."""

    hardware_by_id = {entry.hardware_id: entry for entry in config.hardware}
    model_by_id = {entry.dataset_model_id: entry for entry in config.models}
    cells = []
    for cell_id in config.cell_order:
        hardware_id, model_id, case_id = cell_id.split("/", maxsplit=2)
        case_match = _CASE_ID.fullmatch(case_id)
        if case_match is None:
            raise AssertionError("validated compact-suite case lost its identity")
        hashes = config.cell_hashes[cell_id]
        cells.append(Phase1Cell(
            cell_id=cell_id,
            model=model_by_id[model_id],
            hardware=hardware_by_id[hardware_id],
            qps=int(case_match.group("qps")),
            case_id=case_id,
            manifest_sha256=hashes[0],
            requests_sha256=hashes[1],
        ))
    if config.suite_id == "sharegpt_phase1" and len(cells) != 27:
        raise AssertionError("frozen first-phase matrix must contain 27 cells")
    return tuple(cells)


def official_metrics_suite_runs(
    repository_root: str | Path,
    *,
    nvidia_h200_vllm_version: str | None = None,
) -> tuple[tuple[Phase1Config, tuple[Phase1Cell, ...]], ...]:
    """Return the non-overlapping suite selections covering all 45 truth cells.

    ``nvidia_h200_vllm_version`` additionally selects the 15 H200 cells of one
    captured vLLM version.
    """

    root = Path(repository_root).resolve()
    selections = list(OFFICIAL_METRICS_SUITE_SELECTIONS)
    expected = _official_metrics_case_keys()
    if nvidia_h200_vllm_version is not None:
        if nvidia_h200_vllm_version not in NVIDIA_H200_SUITE_SELECTIONS:
            raise ValueError(
                f"unknown H200 vLLM version: {nvidia_h200_vllm_version!r}")
        selections.extend(
            (path, ("nvidia_h200",))
            for path in NVIDIA_H200_SUITE_SELECTIONS[nvidia_h200_vllm_version])
        expected = expected | _matrix_case_keys(("nvidia_h200",))
    runs = []
    observed = set()
    for relative_path, hardware_ids in selections:
        config = load_phase1_config(root / relative_path)
        cells = tuple(
            cell
            for cell in phase1_cells(config)
            if hardware_ids is None or cell.hardware.hardware_id in hardware_ids
        )
        if not cells:
            raise ValueError(f"official metrics suite selects no cells: {relative_path}")
        for cell in cells:
            key = (
                cell.hardware.hardware_id,
                _canonical_dataset_model_id(cell.model.dataset_model_id),
                cell.case_id,
            )
            if key in observed:
                raise ValueError(f"duplicate official metrics cell: {'/'.join(key)}")
            observed.add(key)
        runs.append((config, cells))
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "official metrics suites do not cover the selected cell matrix; "
            f"missing={missing}, extra={extra}"
        )
    return tuple(runs)


def run_official_metrics_evaluation(
    repository_root: str | Path,
    run_id: str,
    source_commit: str,
    *,
    jobs: int,
    metrics_only: bool,
    compact_alignment_summary: Mapping[str, object] | None,
    isolated_case_worker: Callable[
        [
            str,
            Phase1Config,
            Phase1Cell,
            str,
            bool | Mapping[str, object],
            str,
        ],
        dict[str, object],
    ],
    execution_mode: str = "accelerated",
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Run 45 metrics cells after one shared 45-case alignment gate."""

    _validate_jobs(jobs)
    alignment_by_cell = _official_alignment_inputs(
        compact_alignment_summary,
        source_commit=source_commit,
        metrics_only=metrics_only,
    )
    _validate_execution_mode(execution_mode)
    if not isinstance(debug_analytical_launch, bool):
        raise TypeError("debug_analytical_launch must be boolean")
    metrics_started = time.perf_counter()
    suite_runs = official_metrics_suite_runs(repository_root)
    if host_launch_profile_paths is not None:
        required_hardware = {
            cell.hardware.hardware_id
            for _config, cells in suite_runs
            for cell in cells
        }
        unexpected_hardware = set(host_launch_profile_paths) - required_hardware
        if unexpected_hardware:
            raise ValueError(
                "three-mode official evaluation profiles contain unselected "
                f"hardware: {sorted(unexpected_hardware)}")
    if scheduler_host_profile_paths is not None:
        required_hardware = {
            cell.hardware.hardware_id
            for _config, cells in suite_runs
            for cell in cells
        }
        unexpected_scheduler = (
            set(scheduler_host_profile_paths) - required_hardware)
        if unexpected_scheduler:
            raise ValueError(
                "official evaluation Scheduler Host profiles contain "
                f"unselected hardware: {sorted(unexpected_scheduler)}")
    prepared_runs = []
    parallel_cases = []
    for config, cells in suite_runs:
        output_dir, alignment_inputs = _prepare_phase1_evaluation(
            repository_root,
            config,
            cells,
            f"{run_id}--{config.suite_id}",
            metrics_only=metrics_only,
            precomputed_alignment=(
                None
                if alignment_by_cell is None
                else {
                    _canonical_cell_id(cell): alignment_by_cell[
                        _canonical_cell_id(cell)
                    ]
                    for cell in cells
                }
            ),
        )
        start_index = len(parallel_cases)
        parallel_cases.extend(
            (config, cell, alignment_input)
            for cell, alignment_input in zip(
                cells, alignment_inputs, strict=True
            )
        )
        prepared_runs.append((
            config,
            cells,
            output_dir,
            start_index,
            len(parallel_cases),
        ))
    all_results, completed_at = _run_parallel_case_jobs(
        repository_root,
        parallel_cases,
        source_commit,
        jobs,
        isolated_case_worker,
        execution_mode=execution_mode,
        debug_analytical_launch=debug_analytical_launch,
        host_launch_profile_paths=host_launch_profile_paths,
        scheduler_host_profile_paths=scheduler_host_profile_paths,
    )
    metrics_wall_time_seconds = time.perf_counter() - metrics_started
    summaries = []
    for config, _cells, output_dir, start_index, stop_index in prepared_runs:
        results = all_results[start_index:stop_index]
        suite_wall_time_seconds = (
            max(completed_at[start_index:stop_index]) - metrics_started
        )
        summaries.append(_persist_phase1_evaluation(
            repository_root,
            config,
            source_commit,
            output_dir,
            results,
            jobs=jobs,
            execution_mode=execution_mode,
            evaluation_wall_time_seconds=suite_wall_time_seconds,
            debug_num_kv_blocks=None,
            debug_analytical_launch=debug_analytical_launch,
        ))
    root = Path(repository_root).resolve()
    report_output_dir = create_run_directory(root, run_id)
    report_cells = []
    for summary in summaries:
        result_path = root / summary["artifact_paths"]["json"]
        result = json.loads(result_path.read_text())
        report_cells.extend(_sequence(result["cells"], "suite result cells"))
    report_paths = write_sharegpt_evaluation_report(
        report_output_dir,
        report_cells,
        run_elapsed_wall_time_seconds=metrics_wall_time_seconds,
    )
    if compact_alignment_summary is None:
        alignment_report = {
            "suite_id": "sharegpt_compact_alignment_45",
            "status": "skipped_metrics_only_debug",
            "wall_time_seconds": 0.0,
            "alignment_stage_ran_count": 0,
            "matched_count": 0,
            "mismatched_count": 0,
            "unsupported_count": 0,
            "alignment_checked": False,
            "formal_conclusion_allowed": False,
        }
    else:
        alignment_report = dict(compact_alignment_summary)
    alignment_stage_ran = (
        alignment_report["alignment_stage_ran_count"] == 45
    )
    alignment_checked = alignment_report["alignment_checked"] is True
    formal_conclusion_allowed = (
        alignment_report["formal_conclusion_allowed"] is True
        and all(summary["formal_conclusion_allowed"] is True for summary in summaries)
    )
    alignment_wall_time_seconds = alignment_report["wall_time_seconds"]
    return {
        "schema_version": 1,
        "suite_id": "sharegpt_official_metrics_45",
        "metrics_only_debug": metrics_only,
        "execution_mode": execution_mode,
        "evaluation_wall_time_seconds": (
            alignment_wall_time_seconds + metrics_wall_time_seconds
        ),
        "alignment_wall_time_seconds": alignment_wall_time_seconds,
        "metrics_evaluation_wall_time_seconds": metrics_wall_time_seconds,
        "cell_count": sum(summary["cell_count"] for summary in summaries),
        "completed_count": sum(summary["completed_count"] for summary in summaries),
        "unsupported_count": sum(summary["unsupported_count"] for summary in summaries),
        "alignment_stage_ran": alignment_stage_ran,
        "alignment_checked": alignment_checked,
        "formal_conclusion_allowed": formal_conclusion_allowed,
        "dataset_validation_and_alignment": alignment_report,
        "suite_runs": summaries,
        "run_directory": str(report_output_dir.relative_to(root)),
        "report_artifact_paths": {
            name: str(path.relative_to(root))
            for name, path in report_paths.items()
        },
    }


def _official_alignment_inputs(
    summary_value: Mapping[str, object] | None,
    *,
    source_commit: str,
    metrics_only: bool,
) -> Mapping[str, Mapping[str, object]] | None:
    if metrics_only:
        if summary_value is not None:
            raise ValueError(
                "metrics-only debug evaluation cannot consume alignment results"
            )
        return None
    if summary_value is None:
        raise ValueError(
            "official metrics evaluation requires the complete 45-case alignment report"
        )
    summary = _mapping(summary_value, "compact alignment summary")
    if summary["suite_id"] != "sharegpt_compact_alignment_45":
        raise ValueError("compact alignment summary suite_id is invalid")
    if summary["source_commit"] != source_commit:
        raise ValueError("compact alignment summary source_commit mismatch")
    if summary["case_count"] != 45 or summary["schema_validated_count"] != 45:
        raise ValueError("compact alignment summary must validate exactly 45 cases")
    cases = _sequence(summary["cases"], "compact alignment summary cases")
    if len(cases) != 45:
        raise ValueError("compact alignment summary must contain exactly 45 cases")
    expected = _official_metrics_case_keys()
    alignment_by_cell = {}
    for index, case_value in enumerate(cases):
        case = _mapping(case_value, f"compact alignment case {index}")
        cell_id = _string(case["cell_id"], "compact alignment cell_id")
        if cell_id in alignment_by_cell:
            raise ValueError(f"duplicate compact alignment result: {cell_id}")
        alignment_by_cell[cell_id] = _validated_precomputed_alignment(
            case["alignment"]
        )
    if set(alignment_by_cell) != {"/".join(key) for key in expected}:
        raise ValueError(
            "compact alignment report does not cover the exact 45-case matrix"
        )
    status_counts = {
        status: sum(
            alignment["status"] == status
            for alignment in alignment_by_cell.values()
        )
        for status in ("matched", "mismatched", "unsupported")
    }
    summary_expectations = {
        "alignment_stage_ran_count": 45,
        "matched_count": status_counts["matched"],
        "mismatched_count": status_counts["mismatched"],
        "unsupported_count": status_counts["unsupported"],
        "alignment_checked": status_counts["matched"] == 45,
        "formal_conclusion_allowed": status_counts["matched"] == 45,
    }
    for field, expected_value in summary_expectations.items():
        if summary[field] != expected_value:
            raise ValueError(
                f"compact alignment summary {field} is inconsistent"
            )
    return {
        cell_id: alignment_by_cell[cell_id]
        for cell_id in sorted(alignment_by_cell)
        if tuple(cell_id.split("/", maxsplit=2)) in _official_metrics_case_keys()
    }


def _official_metrics_case_keys() -> set[tuple[str, str, str]]:
    keys = _matrix_case_keys(_OFFICIAL_HARDWARE)
    if len(keys) != 45:
        raise AssertionError("official ShareGPT metric matrix must contain 45 cells")
    return keys


def _matrix_case_keys(hardware_ids: Sequence[str]) -> set[tuple[str, str, str]]:
    keys = {
        (hardware_id, model_id, f"tp1_qps{qps}")
        for hardware_id in hardware_ids
        for model_id in ("qwen3-4b", "qwen3-8b", "llama-3.1-8b")
        for qps in (4, 16, 64)
    }
    keys.update({
        (hardware_id, model_id, f"tp2_ep{ep}_qps{qps}")
        for hardware_id in hardware_ids
        for model_id, ep in (("qwen3-30b-a3b", 2), ("qwen3-32b", 1))
        for qps in (1, 4, 16)
    })
    return keys


def _canonical_cell_id(cell: Phase1Cell) -> str:
    return "/".join((
        cell.hardware.hardware_id,
        _canonical_dataset_model_id(cell.model.dataset_model_id),
        cell.case_id,
    ))


def _validated_precomputed_alignment(
    value: object,
) -> dict[str, object]:
    alignment = _mapping(value, "precomputed compact alignment")
    status = alignment["status"]
    if status not in ("matched", "mismatched", "unsupported"):
        raise ValueError(f"invalid precomputed alignment status: {status!r}")
    expected_success = status == "matched"
    expected_fields = {
        "alignment_stage_ran": True,
        "alignment_checked": expected_success,
        "formal_conclusion_allowed": expected_success,
    }
    for field, expected in expected_fields.items():
        if alignment[field] is not expected:
            raise ValueError(
                f"precomputed alignment {field} is inconsistent with {status}"
            )
    _mapping(alignment["phases"], "precomputed alignment phases")
    return dict(alignment)


def _case_alignment_inputs(
    cells: Sequence[Phase1Cell],
    *,
    metrics_only: bool,
    precomputed_alignment: Mapping[str, Mapping[str, object]] | None,
) -> tuple[bool | Mapping[str, object], ...]:
    if metrics_only:
        if precomputed_alignment is not None:
            raise ValueError(
                "metrics-only debug evaluation cannot consume alignment results"
            )
        return tuple(False for _ in cells)
    if precomputed_alignment is None:
        return tuple(True for _ in cells)
    expected_ids = tuple(_canonical_cell_id(cell) for cell in cells)
    if set(precomputed_alignment) != set(expected_ids):
        raise ValueError(
            "precomputed alignment cells must exactly match selected metric cells"
        )
    return tuple(
        _validated_precomputed_alignment(precomputed_alignment[cell_id])
        for cell_id in expected_ids
    )


def select_phase1_cells(
    config: Phase1Config,
    requested_cell_ids: Sequence[str],
    *,
    all_cases: bool,
) -> tuple[Phase1Cell, ...]:
    """Select explicit cells; an empty CLI never starts all 27 implicitly."""

    cells = phase1_cells(config)
    if all_cases:
        if requested_cell_ids:
            raise ValueError("--all-cases cannot be combined with --case")
        return cells
    if not requested_cell_ids:
        raise ValueError("select at least one --case or pass --all-cases")
    if len(requested_cell_ids) != len(set(requested_cell_ids)):
        raise ValueError("phase-1 case selection contains duplicates")
    by_id = {cell.cell_id: cell for cell in cells}
    unknown = [cell_id for cell_id in requested_cell_ids if cell_id not in by_id]
    if unknown:
        raise ValueError(f"unknown phase-1 cell IDs: {unknown}")
    return tuple(by_id[cell_id] for cell_id in requested_cell_ids)


def compact_case_path(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
) -> Path:
    root = Path(repository_root).resolve()
    dataset_root = _confined_path(root, config.dataset_root, "dataset_root")
    _validate_case_id_contract(cell.case_id, config.prediction.values)
    expected_cell_id = (
        f"{cell.hardware.hardware_id}/{cell.model.dataset_model_id}/{cell.case_id}"
    )
    if cell.cell_id != expected_cell_id:
        raise ValueError("compact cell_id disagrees with its path components")
    canonical_model_id = _canonical_dataset_model_id(cell.model.dataset_model_id)
    case_path = (
        dataset_root
        / cell.hardware.hardware_id
        / canonical_model_id
        / cell.case_id
    ).resolve()
    if dataset_root not in case_path.parents:
        raise ValueError(f"compact case path escapes dataset_root: {cell.cell_id}")
    return case_path


def run_isolated_case(
    repository_root: str,
    config: Phase1Config,
    cell: Phase1Cell,
    source_commit: str,
    alignment_checked: bool | Mapping[str, object],
    execution_mode: str,
    *,
    load_validated_case: Callable[[Path], ValidatedCompactCase],
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
    host_request_path_profile_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Load and evaluate one isolated compact case in the default timing mode."""

    case = load_validated_case(compact_case_path(repository_root, config, cell))
    hardware_id = cell.hardware.hardware_id
    profile_path = None
    if (host_launch_profile_paths is not None
            and hardware_id in host_launch_profile_paths):
        profile_path = host_launch_profile_paths[hardware_id]
    common = {
        "source_commit": source_commit,
        "alignment_checked": alignment_checked,
        "execution_mode": execution_mode,
        "debug_num_kv_blocks": debug_num_kv_blocks,
        "debug_analytical_launch": debug_analytical_launch,
    }
    if profile_path is None:
        if scheduler_host_profile_paths is not None:
            raise ValueError(
                "device-only evaluation cannot consume Scheduler Host "
                "profiles")
        return evaluate_validated_case(
            repository_root,
            config,
            cell,
            case,
            run_simulation,
            serving_timing_mode="device_only",
            **common,
        )
    if (scheduler_host_profile_paths is None
            or hardware_id not in scheduler_host_profile_paths):
        raise ValueError(
            "full-timing evaluation requires a Scheduler Host profile for "
            f"{hardware_id}")
    request_path_override = None
    if (host_request_path_profile_paths is not None
            and hardware_id in host_request_path_profile_paths):
        request_path_override = host_request_path_profile_paths[hardware_id]
    return evaluate_validated_case(
        repository_root,
        config,
        cell,
        case,
        run_simulation,
        serving_timing_mode="device_plus_launch_plus_scheduler",
        host_launch_profile_path=profile_path,
        scheduler_host_profile_path=scheduler_host_profile_paths[hardware_id],
        host_request_path_profile_path=request_path_override,
        **common,
    )


def run_isolated_case_three_modes(
    repository_root: str,
    config: Phase1Config,
    cell: Phase1Cell,
    source_commit: str,
    alignment_checked: bool | Mapping[str, object],
    execution_mode: str,
    *,
    load_validated_case: Callable[[Path], ValidatedCompactCase],
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Load and evaluate one isolated compact case in all timing modes."""

    case = load_validated_case(compact_case_path(repository_root, config, cell))
    hardware_id = cell.hardware.hardware_id
    if host_launch_profile_paths is None or hardware_id not in host_launch_profile_paths:
        raise ValueError(
            "three-mode evaluation requires a Host launch profile for "
            f"{hardware_id}"
        )
    if (scheduler_host_profile_paths is None
            or hardware_id not in scheduler_host_profile_paths):
        raise ValueError(
            "three-mode evaluation requires a Scheduler Host profile for "
            f"{hardware_id}"
        )
    return evaluate_validated_case_three_modes(
        repository_root,
        config,
        cell,
        case,
        run_simulation,
        source_commit=source_commit,
        alignment_checked=alignment_checked,
        execution_mode=execution_mode,
        debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch,
        host_launch_profile_path=host_launch_profile_paths[hardware_id],
        scheduler_host_profile_path=scheduler_host_profile_paths[hardware_id],
    )


def build_simulation_argv(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
    *,
    execution_mode: str = "accelerated",
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    serving_timing_mode: str = "device_only",
    host_launch_profile_path: str | Path | None = None,
    scheduler_host_profile_path: str | Path | None = None,
    host_request_path_profile_path: str | Path | None = None,
    scheduler_pipeline_depth: int | None = None,
    scheduler_async_scheduling: bool | None = None,
) -> list[str]:
    """Build a fully explicit current-vLLM Forward invocation."""

    root = Path(repository_root).resolve()
    _validate_execution_mode(execution_mode)
    _validate_case_binding(config, cell, case)
    if scheduler_pipeline_depth is None:
        resolved_scheduler_pipeline_depth = (
            scheduler_pipeline_depth_from_validated_case(case))
    else:
        if (isinstance(scheduler_pipeline_depth, bool)
                or not isinstance(scheduler_pipeline_depth, int)
                or scheduler_pipeline_depth < 1):
            raise ValueError(
                "explicit scheduler_pipeline_depth must be a positive "
                "non-bool integer")
        resolved_scheduler_pipeline_depth = scheduler_pipeline_depth
    if scheduler_async_scheduling is None:
        async_scheduling = scheduler_async_scheduling_from_validated_case(case)
    elif isinstance(scheduler_async_scheduling, bool):
        async_scheduling = scheduler_async_scheduling
    else:
        raise TypeError("explicit scheduler_async_scheduling must be boolean")
    prediction = config.prediction.values
    model_path = _model_config_path(root, cell.model.simulation_model_path)
    hardware_path = _confined_path(
        root, cell.hardware.hardware_config_path, "hardware config"
    )
    if debug_num_kv_blocks is not None and (
            type(debug_num_kv_blocks) is not int or debug_num_kv_blocks < 1):
        raise ValueError("debug_num_kv_blocks must be a positive integer")
    if not isinstance(debug_analytical_launch, bool):
        raise TypeError("debug_analytical_launch must be boolean")
    validate_serving_timing_mode(serving_timing_mode)
    if serving_timing_mode == "device_only":
        if host_launch_profile_path is not None:
            raise ValueError("device_only cannot consume a Host launch profile")
    elif host_launch_profile_path is None:
        raise ValueError(f"{serving_timing_mode} requires a Host launch profile")
    if serving_timing_mode == "device_plus_launch_plus_scheduler":
        if scheduler_host_profile_path is None:
            raise ValueError(
                f"{serving_timing_mode} requires a Scheduler Host profile")
    elif scheduler_host_profile_path is not None:
        raise ValueError(
            f"{serving_timing_mode} cannot consume a Scheduler Host profile")
    variant_paths = None
    if (cell.hardware.hardware_id in _HOST_PROFILE_VARIANTS
            and serving_timing_mode != "device_only"):
        # Versioned Host profiles follow the runtime the case was captured on.
        case_runtime = _mapping(case.manifest["runtime"], "manifest runtime")
        variant_paths = case_host_profile_paths(
            root, cell.hardware.hardware_id,
            _string(case_runtime["runtime_version"], "case runtime version"))
        if host_request_path_profile_path is None and (
                serving_timing_mode == "device_plus_launch_plus_scheduler"):
            host_request_path_profile_path = variant_paths["host_request_path"]
    request_manifest = _mapping(case.manifest["requests"], "manifest requests")
    requests_path = Path(case.case_dir) / request_manifest["path"]
    argv = [
        "--canonical-requests", str(requests_path),
        "--model", str(model_path),
        "--hardware-config", str(hardware_path),
        "--deployment", str(prediction["deployment"]),
        "--backend", str(prediction["backend"]),
        "--prediction-method", str(prediction["prediction_method"]),
        "--execution-mode", execution_mode,
        "--serving-timing-mode", serving_timing_mode,
        "--collective-backend", str(
            prediction["tensor_collective_backend"]),
        "--launch-policy", (
            "analytical_launch"
            if debug_analytical_launch else str(prediction["launch_policy"])
        ),
        "--attention-kv-layout", str(prediction["attention_kv_layout"]),
        "--attention-kv-distribution", str(
            prediction["attention_kv_distribution"]
        ),
        "--graph-mode", cell.hardware.graph_execution.configured_mode,
        "--graph-uniform-decode-query-len", str(
            cell.hardware.graph_execution.uniform_decode_query_len
        ),
        "--max-batched-tokens", str(prediction["max_batched_tokens"]),
        "--max-requests", str(prediction["max_requests"]),
        "--scheduler-pipeline-depth", str(resolved_scheduler_pipeline_depth),
        (
            "--vllm-async-scheduling"
            if async_scheduling else "--no-vllm-async-scheduling"
        ),
        "--kv-block-size-tokens", str(prediction["kv_block_size_tokens"]),
        "--max-model-len", str(prediction["max_model_len"]),
        "--mtp-num-speculative-tokens", str(
            prediction["mtp_num_speculative_tokens"]
        ),
        "--mtp-acceptance-policy", str(prediction["mtp_acceptance_policy"]),
        "--mtp-fixed-accepted-tokens", str(
            prediction["mtp_fixed_accepted_tokens"]
        ),
        "--mtp-acceptance-source-id", str(
            prediction["mtp_acceptance_source_id"]
        ),
        "--prefill-tp-size", str(prediction["tensor_parallel_size"]),
        "--prefill-pp-size", str(prediction["pipeline_parallel_size"]),
        (
            "--enable-chunked-prefill"
            if prediction["enable_chunked_prefill"]
            else "--no-enable-chunked-prefill"
        ),
        (
            "--enable-prefix-caching"
            if prediction["enable_prefix_caching"]
            else "--no-enable-prefix-caching"
        ),
        "--router-resource-id", str(prediction["router_resource_id"]),
        "--router-dispatch-service-time-s", str(
            prediction["router_dispatch_service_time_s"]
        ),
    ]
    if cell.hardware.graph_execution.capture_sizes:
        argv.extend((
            "--graph-capture-sizes",
            *(str(size) for size in cell.hardware.graph_execution.capture_sizes),
        ))
    if cell.hardware.graph_execution.piecewise_eager_workload_kinds:
        argv.extend((
            "--graph-piecewise-eager-workload-kinds",
            *cell.hardware.graph_execution.piecewise_eager_workload_kinds,
        ))
    if host_launch_profile_path is not None:
        profile_path = resolve_repository_input(
            root, str(host_launch_profile_path), "Host launch profile")
        profile = load_host_launch_profile(profile_path)
        if profile.hardware_id != cell.hardware.hardware_id:
            raise ValueError(
                "Host launch profile hardware disagrees with evaluation cell")
        if (variant_paths is not None
                and profile_path.resolve() != variant_paths["host_launch"]):
            raise ValueError(
                "Host launch profile is not the case runtime version's profile")
        argv.extend(("--host-launch-profile", str(profile_path)))
    if scheduler_host_profile_path is not None:
        scheduler_path = resolve_repository_input(
            root, str(scheduler_host_profile_path), "Scheduler Host profile")
        scheduler_profile = load_scheduler_host_profile(scheduler_path)
        if scheduler_profile.hardware_id != cell.hardware.hardware_id:
            raise ValueError(
                "Scheduler Host profile hardware disagrees with evaluation "
                "cell")
        if (variant_paths is not None
                and scheduler_path.resolve() != variant_paths["scheduler_host"]):
            raise ValueError(
                "Scheduler Host profile is not the case runtime version's "
                "profile")
        argv.extend(("--scheduler-host-profile", str(scheduler_path)))
    if host_request_path_profile_path is not None:
        request_path = resolve_repository_input(
            root,
            str(host_request_path_profile_path),
            "Host request path profile")
        request_path_profile = load_host_request_path_profile(request_path)
        if request_path_profile.hardware_id != cell.hardware.hardware_id:
            raise ValueError(
                "Host request path profile hardware disagrees with evaluation "
                "cell")
        argv.extend(("--host-request-path-profile", str(request_path)))
    if variant_paths is not None:
        argv.extend((
            "--host-runtime-prepare-profile",
            str(variant_paths["host_runtime_prepare"]),
        ))
    if debug_num_kv_blocks is None:
        argv.extend((
            "--runtime-memory-policy",
            _expand_runtime_memory_policy_path(
                root, config, cell, cell.model.dataset_model_id),
        ))
    else:
        argv.extend(("--debug-num-kv-blocks", str(debug_num_kv_blocks)))
    if (prediction["tensor_collective_backend"] == "protobridge"
            and prediction["tensor_parallel_size"] > 1):
        argv.extend((
            "--tensor-all-reduce-profile-sha256",
            str(prediction["tensor_all_reduce_profile_sha256"]),
            "--tensor-all-gather-profile-sha256",
            str(prediction["tensor_all_gather_profile_sha256"]),
        ))
    if prediction["expert_parallel_size"] > 1:
        expert_routing = _expected_expert_routing(root, cell)
        if expert_routing.profile is None:
            raise RuntimeError(
                "profile expert routing lost its canonical profile")
        argv.extend((
            "--enable-expert-parallel",
            "--expert-routing-policy", "profile_observation",
            "--expert-routing-source-id", expert_routing.source_id,
            "--expert-routing-profile-manifest",
            str((root / _EXPERT_FREQUENCY_MANIFEST).resolve()),
        ))
    forbidden_flags = {"--workload", "--production-trace"}
    if any(argument in forbidden_flags for argument in argv):
        raise AssertionError("Forward argv contains an evaluation-only input")
    return argv


def evaluate_validated_case(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
    *,
    source_commit: str,
    alignment_checked: bool | Mapping[str, object] = True,
    execution_mode: str = "accelerated",
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    serving_timing_mode: str = "device_only",
    host_launch_profile_path: str | Path | None = None,
    scheduler_host_profile_path: str | Path | None = None,
    host_request_path_profile_path: str | Path | None = None,
) -> dict[str, object]:
    """Run one current prediction and compare it with compact observations."""

    _validate_execution_mode(execution_mode)
    argv = build_simulation_argv(
        repository_root, config, cell, case, execution_mode=execution_mode,
        debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch,
        serving_timing_mode=serving_timing_mode,
        host_launch_profile_path=host_launch_profile_path,
        scheduler_host_profile_path=scheduler_host_profile_path,
        host_request_path_profile_path=host_request_path_profile_path)
    if type(alignment_checked) is bool:
        alignment_result = (
            validate_compact_alignment(repository_root, config, cell, case)
            if alignment_checked
            else {
                "alignment_stage_ran": False,
                "alignment_checked": False,
                "status": "skipped_metrics_only_debug",
                "formal_conclusion_allowed": False,
                "phases": {},
            }
        )
    else:
        alignment_result = _validated_precomputed_alignment(alignment_checked)
    observed = _observed_metrics(case)
    common = _cell_context(config, cell, case, argv, alignment_result)
    started = time.perf_counter()
    try:
        full_payload = run_simulation(argv)
    except KeyError as error:
        reason = _known_launch_gap_reason(error, cell.hardware.hardware_id)
        if reason is None:
            raise
        elapsed = time.perf_counter() - started
        return {
            **common,
            "execution_mode": execution_mode,
            "serving_timing_mode": serving_timing_mode,
            "evaluation_wall_time_seconds": elapsed,
            "prediction_status": "unsupported_launch_coverage",
            "prediction_reason": reason,
            "performance_timing_status": None,
            "accuracy_claim_status": "not_available",
            "forward_provenance": None,
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
            "timing_components": _timing_components(False, elapsed),
            "metrics": _unavailable_prediction_metrics(observed),
        }
    except NotImplementedError as error:
        reason = _known_not_implemented_launch_gap_reason(
            error, cell.hardware.hardware_id
        )
        if reason is None:
            raise
        elapsed = time.perf_counter() - started
        return {
            **common,
            "execution_mode": execution_mode,
            "serving_timing_mode": serving_timing_mode,
            "evaluation_wall_time_seconds": elapsed,
            "prediction_status": "unsupported_launch_coverage",
            "prediction_reason": reason,
            "performance_timing_status": None,
            "accuracy_claim_status": "not_available",
            "forward_provenance": None,
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
            "timing_components": _timing_components(False, elapsed),
            "metrics": _unavailable_prediction_metrics(observed),
        }
    except ValueError as error:
        reason = _known_value_error_launch_gap_reason(
            error, cell.hardware.hardware_id)
        if reason is None:
            raise
        elapsed = time.perf_counter() - started
        return {
            **common,
            "execution_mode": execution_mode,
            "serving_timing_mode": serving_timing_mode,
            "evaluation_wall_time_seconds": elapsed,
            "prediction_status": "unsupported_launch_coverage",
            "prediction_reason": reason,
            "performance_timing_status": None,
            "accuracy_claim_status": "not_available",
            "forward_provenance": None,
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
            "timing_components": _timing_components(False, elapsed),
            "metrics": _unavailable_prediction_metrics(observed),
        }
    forward_provenance = _validate_completed_payload(
        repository_root,
        config,
        cell,
        case,
        full_payload,
        source_commit,
        execution_mode,
        debug_num_kv_blocks,
        debug_analytical_launch,
        serving_timing_mode,
    )
    comparison = _compare_prediction(observed, full_payload, len(case.requests))
    host_timing = _validated_host_timing_payload(
        full_payload, serving_timing_mode, cell.hardware.hardware_id)
    process_peak_rss_bytes = _integer(
        full_payload["process_peak_rss_bytes"], "process peak RSS bytes")
    if process_peak_rss_bytes <= 0:
        raise ValueError("process peak RSS bytes must be positive")
    del full_payload
    elapsed = time.perf_counter() - started
    return {
        **common,
        "execution_mode": execution_mode,
        "serving_timing_mode": serving_timing_mode,
        "evaluation_wall_time_seconds": elapsed,
        "prediction_status": "completed",
        "prediction_reason": None,
        "performance_timing_status": forward_provenance[
            "performance_timing_status"
        ],
        "accuracy_claim_status": forward_provenance["accuracy_claim_status"],
        "forward_provenance": forward_provenance,
        "timing_components": _timing_components(True, elapsed),
        "process_peak_rss_bytes": process_peak_rss_bytes,
        "host_timing": host_timing,
        "metrics": comparison,
    }


def evaluate_validated_case_three_modes(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
    *,
    source_commit: str,
    host_launch_profile_path: str | Path,
    scheduler_host_profile_path: str | Path,
    alignment_checked: bool | Mapping[str, object] = True,
    execution_mode: str = "accelerated",
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
) -> dict[str, object]:
    """Run three independent simulations and attribute error reductions."""

    if type(alignment_checked) is bool:
        alignment_result = (
            validate_compact_alignment(repository_root, config, cell, case)
            if alignment_checked else {
                "alignment_stage_ran": False,
                "alignment_checked": False,
                "status": "skipped_metrics_only_debug",
                "formal_conclusion_allowed": False,
                "phases": {},
            }
        )
    else:
        alignment_result = _validated_precomputed_alignment(alignment_checked)
    repeated_alignment_input = (
        False if alignment_checked is False else alignment_result)
    results = {}
    for timing_mode in SERVING_TIMING_MODES:
        profile_path_by_mode = {
            "device_only": None,
            "device_plus_launch": host_launch_profile_path,
            "device_plus_launch_plus_scheduler": host_launch_profile_path,
        }
        scheduler_path_by_mode = {
            "device_only": None,
            "device_plus_launch": None,
            "device_plus_launch_plus_scheduler": scheduler_host_profile_path,
        }
        results[timing_mode] = evaluate_validated_case(
            repository_root,
            config,
            cell,
            case,
            run_simulation,
            source_commit=source_commit,
            alignment_checked=repeated_alignment_input,
            execution_mode=execution_mode,
            debug_num_kv_blocks=debug_num_kv_blocks,
            debug_analytical_launch=debug_analytical_launch,
            serving_timing_mode=timing_mode,
            host_launch_profile_path=profile_path_by_mode[timing_mode],
            scheduler_host_profile_path=scheduler_path_by_mode[timing_mode],
        )
    device_only = results["device_only"]
    mode_results = {
        mode: {
            field: result[field]
            for field in (
                "prediction_status", "prediction_reason",
                "performance_timing_status", "accuracy_claim_status",
                "forward_provenance", "timing_components", "metrics",
            )
        } | ({"host_timing": result["host_timing"]}
             if "host_timing" in result else {})
        for mode, result in results.items()
    }
    return {
        **device_only,
        "evaluation_wall_time_seconds": sum(
            result["evaluation_wall_time_seconds"] for result in results.values()),
        "timing_components": {
            "schema_version": 2,
            "independent_simulation_per_mode": True,
            "modes": tuple(SERVING_TIMING_MODES),
        },
        "timing_mode_results": mode_results,
        "host_error_attribution": _host_error_attribution(mode_results),
    }


def _timing_components(
    device_only_available: bool,
    simulator_wall_time_seconds: float,
) -> dict[str, object]:
    """Describe prediction time axes without treating simulator CPU as latency."""

    if not isinstance(device_only_available, bool):
        raise TypeError("device-only availability must be boolean")
    wall_time = _finite_number(
        simulator_wall_time_seconds, "simulator wall time")
    if wall_time < 0.0:
        raise ValueError("simulator wall time must be non-negative")
    return {
        "schema_version": 1,
        "device_only": {
            "availability": (
                "available" if device_only_available else "unavailable"),
            "metrics_field": "metrics" if device_only_available else None,
            "time_axis": "predicted_device_only_virtual_time",
        },
        "cpu_plus_device": {
            "availability": "unavailable",
            "metrics_field": None,
            "cpu_source": None,
            "overlap_rule": None,
            "reason": (
                "no validated production critical-path CPU observation was "
                "supplied; simulator Python wall time is diagnostic only"
            ),
        },
        "simulator_wall_time_seconds": wall_time,
    }


def _validated_host_timing_payload(
    payload: Mapping[str, object],
    expected_mode: str,
    expected_hardware_id: str,
) -> dict[str, object]:
    if payload["serving_timing_mode"] != expected_mode:
        raise ValueError("simulation payload Serving timing mode mismatch")
    timing = _mapping(payload["host_timing"], "Host timing payload")
    _exact_fields(timing, {
        "schema_version", "timing_mode", "scheduler_measurement_boundary",
        "launch_measurement_boundary", "profiles", "runtime_prepare_profiles",
        "aggregate", "steps",
    }, "Host timing payload")
    if timing["schema_version"] != 1 or timing["timing_mode"] != expected_mode:
        raise ValueError("Host timing schema or mode mismatch")
    if timing["scheduler_measurement_boundary"] != (
            "in_scheduler_process_schedule_and_output_update_only"):
        raise ValueError("Host timing Scheduler boundary mismatch")
    if timing["launch_measurement_boundary"] != (
            "host_submit_only_no_device_sync"):
        raise ValueError("Host launch measurement boundary mismatch")
    profiles = _sequence(timing["profiles"], "Host launch profiles")
    expected_profile_count = {
        "device_only": 0,
        "device_plus_launch": 1,
        "device_plus_launch_plus_scheduler": 1,
    }[expected_mode]
    if len(profiles) != expected_profile_count:
        raise ValueError(
            "Host timing profile count disagrees with timing mode")
    if profiles:
        profile = _mapping(profiles[0], "Host launch profile")
        if profile["hardware_id"] != expected_hardware_id:
            raise ValueError("Host timing profile hardware mismatch")
    runtime_prepare_profiles = _sequence(
        timing["runtime_prepare_profiles"], "Host runtime prepare profiles")
    if len(runtime_prepare_profiles) > expected_profile_count:
        raise ValueError("Host runtime prepare profile count disagrees with timing mode")
    if runtime_prepare_profiles:
        profile = _mapping(
            runtime_prepare_profiles[0], "Host runtime prepare profile")
        _exact_fields(profile, {
            "hardware_id", "service_time_s_by_mechanism",
            "lane_service_time_s_by_mechanism", "profile_path", "profile_sha256",
        }, "Host runtime prepare profile")
        if profile["hardware_id"] != expected_hardware_id:
            raise ValueError("Host runtime prepare profile hardware mismatch")
    aggregate = _mapping(timing["aggregate"], "Host timing aggregate")
    expected_aggregate_fields = {
        "predicted_device_s", "launch_cpu_s", "launch_critical_path_s",
        "host_device_overlap_s", "scheduler_plan_cpu_s",
        "scheduler_update_cpu_s", "execution_wall_s", "step_wall_s",
        "eager_dispatch_count", "graph_replay_count", "scheduler_cpu_s",
        "schedule_makespan_s", "step_count",
    }
    _exact_fields(aggregate, expected_aggregate_fields, "Host timing aggregate")
    for field in expected_aggregate_fields - {
            "eager_dispatch_count", "graph_replay_count", "step_count"}:
        if _finite_number(aggregate[field], field) < 0:
            raise ValueError(f"Host timing aggregate {field} must be non-negative")
    for field in ("eager_dispatch_count", "graph_replay_count", "step_count"):
        if _integer(aggregate[field], field) < 0:
            raise ValueError(f"Host timing aggregate {field} must be non-negative")
    if not math.isclose(
            aggregate["scheduler_cpu_s"],
            aggregate["scheduler_plan_cpu_s"]
            + aggregate["scheduler_update_cpu_s"],
            rel_tol=1e-12,
            abs_tol=1e-15):
        raise ValueError("Host timing Scheduler aggregate is inconsistent")
    steps = _sequence(timing["steps"], "Host timing steps")
    if len(steps) != aggregate["step_count"] or not steps:
        raise ValueError("Host timing step count is inconsistent")
    return dict(timing)


def _host_error_attribution(
    mode_results: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Quantify each added Host component without treating it as fitted."""

    if set(mode_results) != set(SERVING_TIMING_MODES):
        raise ValueError("Host attribution requires all three timing modes")
    if any(
            mode_results[mode]["prediction_status"] != "completed"
            for mode in SERVING_TIMING_MODES):
        return {
            "schema_version": 1,
            "status": "unavailable_incomplete_prediction",
            "metrics": {},
        }

    def attribution(rows):
        device = _mapping(rows["device_only"], "device-only comparison")
        launch = _mapping(rows["device_plus_launch"], "launch comparison")
        scheduler = _mapping(
            rows["device_plus_launch_plus_scheduler"],
            "launch-plus-Scheduler comparison",
        )
        observed = _finite_number(device["observed"], "observed metric")
        if launch["observed"] != observed or scheduler["observed"] != observed:
            raise ValueError("timing-mode comparisons use different truth")
        predicted = {
            mode: _finite_number(rows[mode]["predicted"], f"{mode} prediction")
            for mode in SERVING_TIMING_MODES
        }
        errors = {mode: predicted[mode] - observed for mode in SERVING_TIMING_MODES}
        return {
            "observed": observed,
            "device_only": predicted["device_only"],
            "device_plus_launch": predicted["device_plus_launch"],
            "device_plus_launch_plus_scheduler": predicted[
                "device_plus_launch_plus_scheduler"],
            "launch_prediction_delta": (
                predicted["device_plus_launch"] - predicted["device_only"]),
            "scheduler_prediction_delta": (
                predicted["device_plus_launch_plus_scheduler"]
                - predicted["device_plus_launch"]),
            "launch_absolute_error_reduction": (
                abs(errors["device_only"]) - abs(errors["device_plus_launch"])),
            "scheduler_absolute_error_reduction": (
                abs(errors["device_plus_launch"])
                - abs(errors["device_plus_launch_plus_scheduler"])),
            "final_residual": errors["device_plus_launch_plus_scheduler"],
            "final_absolute_residual": abs(
                errors["device_plus_launch_plus_scheduler"]),
        }

    mode_metrics = {
        mode: _mapping(mode_results[mode]["metrics"], f"{mode} metrics")
        for mode in SERVING_TIMING_MODES
    }
    latency = {}
    for metric in _LATENCY_FIELDS:
        latency[metric] = {}
        for statistic in STATISTICS:
            latency[metric][statistic] = attribution({
                mode: _mapping(
                    _mapping(mode_metrics[mode]["latency"], "latency")[metric],
                    metric,
                )[statistic]
                for mode in SERVING_TIMING_MODES
            })
    throughput = {
        metric: attribution({
            mode: _mapping(
                mode_metrics[mode]["throughput"], "throughput")[metric]
            for mode in SERVING_TIMING_MODES
        })
        for metric in _THROUGHPUT_FIELDS
    }
    return {
        "schema_version": 1,
        "status": "available",
        "units": {"latency": "ms", "throughput": "per_second"},
        "latency": latency,
        "throughput": throughput,
    }


def run_phase1_evaluation(
    repository_root: str | Path,
    config: Phase1Config,
    cells: Sequence[Phase1Cell],
    run_id: str,
    source_commit: str,
    load_validated_case: Callable[[Path], ValidatedCompactCase] | None = None,
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]] | None = None,
    *,
    jobs: int = 1,
    metrics_only: bool = False,
    execution_mode: str = "accelerated",
    isolated_case_worker: Callable[
        [
            str,
            Phase1Config,
            Phase1Cell,
            str,
            bool | Mapping[str, object],
            str,
        ],
        dict[str, object],
    ] | None = None,
    precomputed_alignment: Mapping[str, Mapping[str, object]] | None = None,
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Validate inputs, isolate case execution, and persist in parent order."""

    _validate_jobs(jobs)
    _validate_execution_mode(execution_mode)
    if debug_num_kv_blocks is not None:
        if type(debug_num_kv_blocks) is not int or debug_num_kv_blocks < 1:
            raise ValueError("debug_num_kv_blocks must be a positive integer")
        if not metrics_only or len(cells) != 1:
            raise ValueError(
                "debug_num_kv_blocks requires metrics_only and one cell")
    if not isinstance(debug_analytical_launch, bool):
        raise TypeError("debug_analytical_launch must be boolean")
    if host_launch_profile_paths is not None:
        required_hardware = {cell.hardware.hardware_id for cell in cells}
        unexpected_hardware = set(host_launch_profile_paths) - required_hardware
        if unexpected_hardware:
            raise ValueError(
                "three-mode evaluation profiles contain unselected hardware: "
                f"{sorted(unexpected_hardware)}")
    if scheduler_host_profile_paths is not None:
        required_hardware = {cell.hardware.hardware_id for cell in cells}
        unexpected_scheduler = (
            set(scheduler_host_profile_paths) - required_hardware)
        if unexpected_scheduler:
            raise ValueError(
                "evaluation Scheduler Host profiles contain unselected "
                f"hardware: {sorted(unexpected_scheduler)}")
    started = time.perf_counter()
    if isolated_case_worker is None:
        if jobs != 1:
            raise ValueError(
                "parallel compact evaluation requires an isolated case worker"
            )
        if load_validated_case is None or run_simulation is None:
            raise ValueError(
                "serial compact evaluation requires loader and simulator"
            )
    elif load_validated_case is not None or run_simulation is not None:
        raise ValueError(
            "isolated compact evaluation cannot use in-process dependencies"
        )
    output_dir, alignment_inputs = _prepare_phase1_evaluation(
        repository_root,
        config,
        cells,
        run_id,
        metrics_only=metrics_only,
        precomputed_alignment=precomputed_alignment,
    )
    if isolated_case_worker is None:
        assert load_validated_case is not None
        assert run_simulation is not None
        results = [
            _evaluate_one_case(
                repository_root,
                config,
                cell,
                source_commit,
                load_validated_case,
                run_simulation,
                alignment_input,
                execution_mode,
                debug_num_kv_blocks,
                debug_analytical_launch,
                host_launch_profile_paths,
                scheduler_host_profile_paths,
            )
            for cell, alignment_input in zip(cells, alignment_inputs, strict=True)
        ]
    else:
        results = _run_parallel_cases(
            repository_root,
            config,
            cells,
            source_commit,
            jobs,
            isolated_case_worker,
            alignment_inputs=alignment_inputs,
            execution_mode=execution_mode,
            debug_num_kv_blocks=debug_num_kv_blocks,
            debug_analytical_launch=debug_analytical_launch,
            host_launch_profile_paths=host_launch_profile_paths,
            scheduler_host_profile_paths=scheduler_host_profile_paths,
        )
    return _persist_phase1_evaluation(
        repository_root,
        config,
        source_commit,
        output_dir,
        results,
        jobs=jobs,
        execution_mode=execution_mode,
        evaluation_wall_time_seconds=time.perf_counter() - started,
        debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch,
    )


def _prepare_phase1_evaluation(
    repository_root: str | Path,
    config: Phase1Config,
    cells: Sequence[Phase1Cell],
    run_id: str,
    *,
    metrics_only: bool,
    precomputed_alignment: Mapping[str, Mapping[str, object]] | None,
) -> tuple[Path, tuple[bool | Mapping[str, object], ...]]:
    alignment_inputs = _case_alignment_inputs(
        cells,
        metrics_only=metrics_only,
        precomputed_alignment=precomputed_alignment,
    )
    output_dir = create_run_directory(repository_root, run_id)
    for cell in cells:
        case_path = compact_case_path(repository_root, config, cell)
        for path in (case_path / "manifest.json", case_path / "requests.csv"):
            if not path.is_file():
                raise ValueError(f"phase-1 case input is missing: {path}")
    return output_dir, alignment_inputs


def _persist_phase1_evaluation(
    repository_root: str | Path,
    config: Phase1Config,
    source_commit: str,
    output_dir: Path,
    results: Sequence[Mapping[str, object]],
    *,
    jobs: int,
    execution_mode: str,
    evaluation_wall_time_seconds: float,
    debug_num_kv_blocks: int | None,
    debug_analytical_launch: bool,
) -> dict[str, object]:
    alignment_stage_ran = bool(results) and all(
        _mapping(result["alignment"], "cell alignment")["alignment_stage_ran"] is True
        for result in results
    )
    alignment_checked = bool(results) and all(
        _mapping(result["alignment"], "cell alignment")["alignment_checked"] is True
        for result in results
    )
    formal_conclusion_allowed = (
        debug_num_kv_blocks is None
        and not debug_analytical_launch
        and bool(results)
        and all(
        _mapping(result["alignment"], "cell alignment")[
            "formal_conclusion_allowed"
        ] is True
        for result in results
        )
    )
    paths = write_phase1_results(
        output_dir,
        config=config,
        source_commit=source_commit,
        cells=results,
        execution_mode=execution_mode,
        evaluation_wall_time_seconds=evaluation_wall_time_seconds,
        alignment_stage_ran=alignment_stage_ran,
        alignment_checked=alignment_checked,
        formal_conclusion_allowed=formal_conclusion_allowed,
    )
    report_paths = write_sharegpt_evaluation_report(
        output_dir,
        results,
        run_elapsed_wall_time_seconds=evaluation_wall_time_seconds,
    )
    paths = {**paths, **report_paths}
    root = Path(repository_root).resolve()
    return {
        "schema_version": 1,
        "suite_id": config.suite_id,
        "execution_mode": execution_mode,
        "evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        "run_directory": str(output_dir.relative_to(root)),
        "cell_count": len(results),
        "completed_count": sum(
            result["prediction_status"] == "completed" for result in results
        ),
        "unsupported_count": sum(
            result["prediction_status"] == "unsupported_launch_coverage"
            for result in results
        ),
        "jobs": jobs,
        "alignment_stage_ran": alignment_stage_ran,
        "alignment_checked": alignment_checked,
        "formal_conclusion_allowed": formal_conclusion_allowed,
        "artifact_paths": {
            name: str(path.relative_to(root)) for name, path in paths.items()
        },
    }


def _evaluate_one_case(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    source_commit: str,
    load_validated_case: Callable[[Path], ValidatedCompactCase],
    run_simulation: Callable[[Sequence[str]], Mapping[str, object]],
    alignment_checked: bool | Mapping[str, object],
    execution_mode: str,
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> dict[str, object]:
    case_path = compact_case_path(repository_root, config, cell)
    validated_case = load_validated_case(case_path)
    hardware_id = cell.hardware.hardware_id
    profile_path = None
    if (host_launch_profile_paths is not None
            and hardware_id in host_launch_profile_paths):
        profile_path = host_launch_profile_paths[hardware_id]
    evaluator = (
        evaluate_validated_case_three_modes
        if profile_path is not None else evaluate_validated_case)
    extra = (
        {"host_launch_profile_path": profile_path}
        if profile_path is not None else {})
    if profile_path is not None:
        if (scheduler_host_profile_paths is None
                or hardware_id not in scheduler_host_profile_paths):
            raise ValueError(
                "three-mode evaluation requires a Scheduler Host profile "
                f"for {hardware_id}")
        extra["scheduler_host_profile_path"] = (
            scheduler_host_profile_paths[hardware_id])
    elif scheduler_host_profile_paths is not None:
        raise ValueError(
            "device-only evaluation cannot consume Scheduler Host profiles")
    return evaluator(
        repository_root,
        config,
        cell,
        validated_case,
        run_simulation,
        source_commit=source_commit,
        alignment_checked=alignment_checked,
        execution_mode=execution_mode,
        debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch,
        **extra,
    )


def _run_parallel_cases(
    repository_root: str | Path,
    config: Phase1Config,
    cells: Sequence[Phase1Cell],
    source_commit: str,
    jobs: int,
    isolated_case_worker: Callable[
        [
            str,
            Phase1Config,
            Phase1Cell,
            str,
            bool | Mapping[str, object],
            str,
        ],
        dict[str, object],
    ],
    alignment_checked: bool = True,
    *,
    alignment_inputs: Sequence[bool | Mapping[str, object]] | None = None,
    execution_mode: str = "accelerated",
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> list[dict[str, object]]:
    if not cells:
        return []
    if alignment_inputs is None:
        resolved_alignment_inputs = tuple(alignment_checked for _ in cells)
    else:
        resolved_alignment_inputs = tuple(alignment_inputs)
        if len(resolved_alignment_inputs) != len(cells):
            raise ValueError("alignment input count must match selected cells")
    results, _completed_at = _run_parallel_case_jobs(
        repository_root,
        tuple(
            (config, cell, alignment_input)
            for cell, alignment_input in zip(
                cells, resolved_alignment_inputs, strict=True
            )
        ),
        source_commit,
        jobs,
        isolated_case_worker,
        execution_mode=execution_mode,
        debug_num_kv_blocks=debug_num_kv_blocks,
        debug_analytical_launch=debug_analytical_launch,
        host_launch_profile_paths=host_launch_profile_paths,
        scheduler_host_profile_paths=scheduler_host_profile_paths,
    )
    return results


def _run_parallel_case_jobs(
    repository_root: str | Path,
    case_jobs: Sequence[
        tuple[
            Phase1Config,
            Phase1Cell,
            bool | Mapping[str, object],
        ]
    ],
    source_commit: str,
    jobs: int,
    isolated_case_worker: Callable[
        [
            str,
            Phase1Config,
            Phase1Cell,
            str,
            bool | Mapping[str, object],
            str,
        ],
        dict[str, object],
    ],
    *,
    execution_mode: str = "accelerated",
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    host_launch_profile_paths: Mapping[str, str] | None = None,
    scheduler_host_profile_paths: Mapping[str, str] | None = None,
) -> tuple[list[dict[str, object]], tuple[float, ...]]:
    if not case_jobs:
        return [], ()
    result_by_index: dict[int, dict[str, object]] = {}
    completed_at_by_index: dict[int, float] = {}
    executor = ProcessPoolExecutor(max_workers=min(jobs, len(case_jobs)))
    future_to_index = {}
    try:
        for index, (config, cell, alignment_input) in enumerate(case_jobs):
            args = (
                str(Path(repository_root).resolve()),
                config,
                cell,
                source_commit,
                alignment_input,
                execution_mode,
            )
            try:
                optional_args = []
                if debug_num_kv_blocks is not None or debug_analytical_launch:
                    optional_args.extend((
                        debug_num_kv_blocks, debug_analytical_launch))
                if host_launch_profile_paths is not None:
                    if not optional_args:
                        optional_args.extend((None, False))
                    optional_args.append(dict(host_launch_profile_paths))
                if scheduler_host_profile_paths is not None:
                    if not optional_args:
                        optional_args.extend((None, False, None))
                    elif len(optional_args) == 2:
                        optional_args.append(None)
                    optional_args.append(dict(scheduler_host_profile_paths))
                future = (
                    executor.submit(isolated_case_worker, *args, *optional_args)
                )
            except Exception as error:
                for pending in future_to_index:
                    pending.cancel()
                raise RuntimeError(
                    "compact evaluation worker submission failed for "
                    f"{_canonical_cell_id(cell)}"
                ) from error
            future_to_index[future] = index
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result_by_index[index] = future.result()
                completed_at_by_index[index] = time.perf_counter()
            except Exception as error:
                for pending in future_to_index:
                    pending.cancel()
                raise RuntimeError(
                    "compact evaluation worker failed for "
                    f"{_canonical_cell_id(case_jobs[index][1])}"
                ) from error
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return (
        [result_by_index[index] for index in range(len(case_jobs))],
        tuple(completed_at_by_index[index] for index in range(len(case_jobs))),
    )


def write_phase1_results(
    output_dir: str | Path,
    *,
    config: Phase1Config,
    source_commit: str,
    cells: Sequence[Mapping[str, object]],
    execution_mode: str = "accelerated",
    evaluation_wall_time_seconds: float = 0.0,
    alignment_stage_ran: bool = False,
    alignment_checked: bool = False,
    formal_conclusion_allowed: bool = False,
) -> Mapping[str, Path]:
    """Persist only aggregate, request-safe experiment artifacts."""

    destination = Path(output_dir)
    if not destination.is_dir():
        raise ValueError(f"phase-1 output directory must already exist: {destination}")
    if not isinstance(source_commit, str) or _GIT_COMMIT.fullmatch(
        source_commit
    ) is None:
        raise ValueError("source commit must be a full lowercase Git SHA")
    _validate_execution_mode(execution_mode)
    evaluation_wall_time_seconds = _finite_number(
        evaluation_wall_time_seconds, "evaluation wall time"
    )
    if evaluation_wall_time_seconds < 0.0:
        raise ValueError("evaluation wall time must be nonnegative")
    for cell in cells:
        if cell["execution_mode"] != execution_mode:
            raise ValueError(
                "persisted cell execution_mode disagrees with run mode")
    result = {
        "schema_version": 1,
        "suite_id": config.suite_id,
        "interpretation": "observed_vs_current_prediction_not_capture_replay",
        "config_sha256": config.source_sha256,
        "source_commit": source_commit,
        "execution_mode": execution_mode,
        "evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        "cell_count": len(cells),
        "alignment_stage_ran": alignment_stage_ran,
        "alignment_checked": alignment_checked,
        "formal_conclusion_allowed": formal_conclusion_allowed,
        "cells": list(cells),
    }
    paths = {
        "json": destination / "results.json",
        "csv": destination / "results.csv",
        "markdown": destination / "report.md",
        "manifest": destination / "run_manifest.json",
        "hashes": destination / "hashes.json",
    }
    _write_json(paths["json"], result)
    _write_csv(paths["csv"], cells)
    paths["markdown"].write_text(_render_markdown(result))
    manifest = {
        "schema_version": 1,
        "suite_id": config.suite_id,
        "interpretation": "observed_vs_current_prediction_not_capture_replay",
        "config_path": str(config.source_path),
        "config_sha256": config.source_sha256,
        "source_commit": source_commit,
        "execution_mode": execution_mode,
        "evaluation_wall_time_seconds": evaluation_wall_time_seconds,
        "alignment_stage_ran": alignment_stage_ran,
        "alignment_checked": alignment_checked,
        "formal_conclusion_allowed": formal_conclusion_allowed,
        "prediction_contract": dict(config.prediction.values),
        "model_mappings": [
            {
                "dataset_model_id": model.dataset_model_id,
                "simulation_model_path": model.simulation_model_path,
                "model_config_sha256": model.model_config_sha256,
            }
            for model in config.models
        ],
        "hardware_mappings": [
            {
                "hardware_id": hardware.hardware_id,
                "hardware_config_path": hardware.hardware_config_path,
                "hardware_config_sha256": hardware.hardware_config_sha256,
                "graph_execution": hardware.graph_execution.manifest(),
            }
            for hardware in config.hardware
        ],
        "selected_cell_inputs": [
            {
                "cell_id": cell["cell_id"],
                "manifest_sha256": cell["manifest_sha256"],
                "requests_sha256": cell["requests_sha256"],
                "alignment": cell["alignment_files"],
            }
            for cell in cells
        ],
        "persisted_content": (
            "aggregate_plus_synthetic_request_index_paired_latency"
        ),
    }
    _write_json(paths["manifest"], manifest)
    digests = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
        if name != "hashes"
    }
    _write_json(paths["hashes"], {"schema_version": 1, "sha256": digests})
    return paths


def create_run_directory(
    repository_root: str | Path,
    run_id: str,
) -> Path:
    """Create one fresh run below the repository experiment_logs tree."""

    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id must match [A-Za-z0-9][A-Za-z0-9._-]*")
    root = Path(repository_root).resolve()
    output_root = root / "experiment_logs" / "serving" / "sharegpt_phase1"
    destination = (output_root / run_id).resolve()
    if output_root.resolve() not in destination.parents:
        raise ValueError("phase-1 output must remain under experiment_logs")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def resolve_repository_input(
    repository_root: str | Path,
    value: str | Path,
    label: str,
) -> Path:
    """Resolve one CLI input while preventing repository-root escape."""

    root = Path(repository_root).resolve()
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{label} must remain inside the repository")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def resolve_host_launch_profile_paths(
    repository_root: str | Path,
    explicit_paths: Sequence[str],
    selected_hardware_ids: set[str] | None,
) -> Mapping[str, str] | None:
    """Resolve explicit profiles or discover canonical profiles by hardware."""

    root = Path(repository_root).resolve()
    values = tuple(explicit_paths)
    use_defaults = not values
    if use_defaults:
        profile_dir = (
            root / "configs/host_timing/host_launch_profiles")
        values = tuple(str(path) for path in sorted(profile_dir.glob("*.json")))
    profiles = {}
    for value in values:
        path = resolve_repository_input(root, value, "Host launch profile")
        profile = load_host_launch_profile(path)
        if use_defaults and profile.hardware_id in _HOST_PROFILE_VARIANTS:
            continue
        if profile.hardware_id in profiles:
            raise ValueError(
                f"duplicate Host launch profile for {profile.hardware_id}")
        if (use_defaults and selected_hardware_ids is not None
                and profile.hardware_id not in selected_hardware_ids):
            continue
        profiles[profile.hardware_id] = str(path)
    return profiles if profiles else None


def resolve_scheduler_host_profile_paths(
    repository_root: str | Path,
    explicit_paths: Sequence[str],
    selected_hardware_ids: set[str] | None,
) -> Mapping[str, str] | None:
    """Resolve explicit profiles or discover canonical profiles by hardware."""

    root = Path(repository_root).resolve()
    values = tuple(explicit_paths)
    use_defaults = not values
    if use_defaults:
        profile_dir = (
            root / "configs/host_timing/scheduler_host_profiles")
        values = tuple(str(path) for path in sorted(profile_dir.glob("*.json")))
    profiles = {}
    for value in values:
        path = resolve_repository_input(root, value, "Scheduler Host profile")
        profile = load_scheduler_host_profile(path)
        if use_defaults and profile.hardware_id in _HOST_PROFILE_VARIANTS:
            continue
        if profile.hardware_id in profiles:
            raise ValueError(
                f"duplicate Scheduler Host profile for {profile.hardware_id}")
        if (use_defaults and selected_hardware_ids is not None
                and profile.hardware_id not in selected_hardware_ids):
            continue
        profiles[profile.hardware_id] = str(path)
    return profiles if profiles else None


def host_profile_variant_hardware() -> frozenset[str]:
    """Hardware whose Host profiles are bound per captured vLLM version."""

    return frozenset(_HOST_PROFILE_VARIANTS)


def case_host_profile_paths(
    repository_root: str | Path,
    hardware_id: str,
    runtime_version: str,
) -> Mapping[str, Path]:
    """Return the four Host profiles of one versioned hardware case."""

    if hardware_id not in _HOST_PROFILE_VARIANTS:
        raise ValueError(f"{hardware_id} Host profiles are not versioned")
    variants = _HOST_PROFILE_VARIANTS[hardware_id]
    if runtime_version not in variants:
        raise ValueError(
            f"no {hardware_id} Host profiles for vLLM {runtime_version!r}")
    root = Path(repository_root).resolve()
    return {
        family: (
            root / "configs/host_timing" / directory
            / f"{variants[runtime_version]}{suffix}.json"
        ).resolve(strict=True)
        for family, (directory, suffix) in _HOST_PROFILE_DIRECTORIES.items()
    }


def repository_source_commit(repository_root: str | Path) -> str:
    """Return the immutable source identity used for this evaluation run."""

    root = Path(repository_root)
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if _GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("git rev-parse HEAD did not return a full commit SHA")
    return commit


def _validate_case_binding(
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
) -> None:
    manifest = _mapping(case.manifest, "compact manifest")
    identity = _mapping(manifest["identity"], "manifest identity")
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    expected_identity = {
        "hardware_id": cell.hardware.hardware_id,
        "model_id": _canonical_dataset_model_id(cell.model.dataset_model_id),
        "case_id": cell.case_id,
        "source_kind": "sharegpt",
    }
    for field, expected in expected_identity.items():
        if identity[field] != expected:
            raise ValueError(
                f"{cell.cell_id} manifest {field} mismatch: "
                f"{identity[field]!r} != {expected!r}"
            )
    runtime_expected = {
        "tensor_parallel_size": config.prediction.values[
            "tensor_parallel_size"
        ],
        "pipeline_parallel_size": config.prediction.values[
            "pipeline_parallel_size"
        ],
        "expert_parallel_size": config.prediction.values[
            "expert_parallel_size"
        ],
        "target_qps": float(cell.qps),
    }
    for field, expected in runtime_expected.items():
        if runtime[field] != expected:
            raise ValueError(
                f"{cell.cell_id} capture runtime {field} mismatch: "
                f"{runtime[field]!r} != {expected!r}"
            )
    provenance = _mapping(manifest["provenance"], "manifest provenance")
    if case.metrics is None:
        raise ValueError(f"{cell.cell_id} requires available official metrics")
    _sequence(provenance["gaps"], "manifest provenance gaps")


def _canonical_dataset_model_id(model_id: str) -> str:
    if model_id in _LEGACY_MODEL_IDS:
        return _LEGACY_MODEL_IDS[model_id]
    if model_id in _CANONICAL_MODEL_SPECS:
        return model_id
    raise ValueError(f"unsupported compact dataset model_id: {model_id!r}")


def _cell_context(
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
    argv: Sequence[str],
    alignment_result: Mapping[str, object],
) -> dict[str, object]:
    manifest = _mapping(case.manifest, "compact manifest")
    runtime = _mapping(manifest["runtime"], "manifest runtime")
    requests = _mapping(manifest["requests"], "manifest requests")
    alignment = _mapping(manifest["alignment_steps"], "manifest alignment_steps")
    context = {
        "cell_id": cell.cell_id,
        "hardware_id": cell.hardware.hardware_id,
        "model_id": cell.model.dataset_model_id,
        "case_id": cell.case_id,
        "target_qps": cell.qps,
        "request_count": len(case.requests),
        "manifest_sha256": cell.manifest_sha256,
        "requests_sha256": requests["sha256"],
        "capture_contract_status": (
            f"schema_v{manifest['schema_version']}_validated"
        ),
        "capture_runtime_comparison": _capture_runtime_comparison(
            runtime, config.prediction.values, cell.qps
        ),
        "prediction_contract": {
            **dict(config.prediction.values),
            "model_config_sha256": cell.model.model_config_sha256,
            "hardware_config_sha256": cell.hardware.hardware_config_sha256,
        },
        "forward_input": {
            "kind": "validated_manifest_binding_and_requests_csv_only",
            "argv_sha256": _canonical_json_sha256(list(argv)),
            "alignment_consumed_by_forward": False,
        },
        "alignment": dict(alignment_result),
        "alignment_files": {
            "alignment_steps": {
                "status": alignment["status"],
                "sha256": alignment["sha256"],
                "phases": alignment["phases"],
            },
        },
    }
    if manifest["schema_version"] == 3:
        if case.scheduler_steps is None:
            raise ValueError("schema-v3 case requires loaded scheduler steps")
        context["observed_scheduler_progress"] = (
            _observed_scheduler_progress(case.scheduler_steps, len(case.requests))
        )
    return context


def _observed_scheduler_progress(
    steps: Sequence[Mapping[str, object]], request_count: int
) -> Mapping[str, object]:
    if not steps:
        raise ValueError("observed scheduler steps cannot be empty")
    series = []
    for index, value in enumerate(steps):
        step = _mapping(value, f"observed scheduler step {index}")
        series.append({
            "step_index": step["step_index"],
            "step_id": step["step_id"],
            "start_time_s": step["step_start_ns"] / 1_000_000_000.0,
            "end_time_s": step["step_end_ns"] / 1_000_000_000.0,
            "request_count": step["scheduled_request_count"],
            "prefill_request_count": step["prefill_request_count"],
            "decode_request_count": step["decode_request_count"],
            "prefill_tokens": step["prefill_tokens"],
            "decode_tokens": step["decode_tokens"],
            "scheduled_tokens": step["scheduled_tokens"],
            "cumulative_scheduled_tokens": step[
                "cumulative_scheduled_tokens"
            ],
            "cumulative_completed_requests": step[
                "cumulative_completed_requests"
            ],
        })
    final = series[-1]
    return {
        "schema_version": 1,
        "time_axis": {
            "kind": "observed_serving_step_wall",
            "unit": "seconds",
            "start_boundary": "immediately_before_scheduler_schedule",
            "end_boundary": "immediately_after_scheduler_output_update",
        },
        "series": series,
        "summary": {
            "step_count": len(series),
            "request_count": request_count,
            "scheduled_tokens": final["cumulative_scheduled_tokens"],
            "completed_requests": final["cumulative_completed_requests"],
            "end_time_s": final["end_time_s"],
        },
    }


def validate_compact_alignment(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
) -> Mapping[str, object]:
    """Run real ScheduleStep-to-lowering structural alignment."""

    model = load_model_config(str(_model_config_path(
        repository_root, cell.model.simulation_model_path)))
    return align_compact_case(repository_root, config, cell, case, model)


def _capture_runtime_comparison(
    capture: Mapping[str, object],
    prediction: Mapping[str, object],
    target_qps: int,
) -> Mapping[str, object]:
    comparison_specs = (
        ("dtype", "bfloat16", True),
        ("deployment", prediction["deployment"], True),
        ("tensor_parallel_size", prediction["tensor_parallel_size"], True),
        ("pipeline_parallel_size", prediction["pipeline_parallel_size"], True),
        ("expert_parallel_size", prediction["expert_parallel_size"], True),
        ("target_qps", float(target_qps), True),
        ("seed", None, False),
        ("scheduler_config_sha256", None, False),
        ("runtime_version", VLLM_SCHEDULER_VERSION, True),
        ("source_commit", VLLM_SCHEDULER_SOURCE_COMMIT, True),
        (
            "capture_enable_prefix_caching",
            prediction["enable_prefix_caching"],
            True,
        ),
    )
    fields = {}
    for field, prediction_value, comparable in comparison_specs:
        capture_value = capture[field]
        if capture_value is None:
            field_status = "unknown"
        elif not comparable:
            field_status = "known"
        elif capture_value == prediction_value:
            field_status = "equal"
        elif field == "dtype" and {
            capture_value, prediction_value
        } == {"float16", "bfloat16"}:
            field_status = "performance_equivalent"
        else:
            field_status = "mismatch"
        fields[field] = {
            "status": field_status,
            "capture": capture_value,
            "prediction": prediction_value,
        }
    if any(entry["status"] == "mismatch" for entry in fields.values()):
        status = "mismatch"
    else:
        status = "incomplete_capture_contract"
    return {
        "status": status,
        "fields": fields,
        "interpretation": "current_prediction_not_capture_replay",
    }


def _observed_ttft_breakdown(
    case: ValidatedCompactCase,
    by_request: Mapping[int, Mapping[str, object]],
) -> Mapping[str, object] | None:
    if case.scheduler_steps is None:
        return None
    first_schedule_start_ms: list[float | None] = [
        None for _request in case.requests
    ]
    for step_index, step_value in enumerate(case.scheduler_steps):
        step = _mapping(step_value, f"observed Scheduler step {step_index}")
        start_ms = _finite_number(
            step["step_start_ns"], "observed Scheduler step start"
        ) / 1_000_000.0
        for scheduled_value in _sequence(
            step["scheduled_requests"], "observed scheduled requests"
        ):
            scheduled = _mapping(
                scheduled_value, "observed scheduled request"
            )
            request_idx = _integer(
                scheduled["request_idx"], "observed scheduled request_idx"
            )
            if request_idx < 0 or request_idx >= len(case.requests):
                raise ValueError("observed scheduled request index is out of range")
            if first_schedule_start_ms[request_idx] is None:
                first_schedule_start_ms[request_idx] = start_ms
    if any(value is None for value in first_schedule_start_ms):
        raise ValueError(
            "observed Scheduler trace does not first-schedule every request"
        )
    breakdown = {}
    negative_wait_clamp_count = 0
    minimum_raw_wait_ms = 0.0
    for request_idx, request in enumerate(case.requests):
        arrival_ms = _finite_number(
            request.arrival_time_ms, "observed request arrival"
        )
        first_start_value = first_schedule_start_ms[request_idx]
        if first_start_value is None:
            raise AssertionError("first schedule coverage was validated")
        raw_wait_ms = first_start_value - arrival_ms
        minimum_raw_wait_ms = min(minimum_raw_wait_ms, raw_wait_ms)
        if raw_wait_ms < 0.0:
            negative_wait_clamp_count += 1
        wait_ms = max(0.0, raw_wait_ms)
        ttft_ms = _finite_number(
            by_request[request_idx]["ttft_ms"], "observed request TTFT"
        )
        processing_ms = ttft_ms - wait_ms
        if processing_ms < -1e-6:
            raise ValueError(
                "observed first-schedule wait exceeds client-visible TTFT"
            )
        breakdown[request_idx] = {
            "wait_ms": wait_ms,
            "processing_ms": max(0.0, processing_ms),
        }
    return {
        "arrival_basis": "canonical_scheduled_arrival_time",
        "negative_wait_clamp_count": negative_wait_clamp_count,
        "minimum_raw_wait_ms": minimum_raw_wait_ms,
        "records": breakdown,
    }


def _observed_metrics(case: ValidatedCompactCase) -> Mapping[str, object]:
    if case.metrics is None:
        raise ValueError("compact case has no metrics")
    metrics = _mapping(case.metrics, "alignment metrics")
    per_request = _sequence(metrics["per_request"], "observed per_request")
    summary = _mapping(metrics["summary"], "observed summary")
    expected_indices = set(range(len(case.requests)))
    actual_indices = {
        _integer(_mapping(item, "observed request")["request_idx"], "request_idx")
        for item in per_request
    }
    if actual_indices != expected_indices or len(per_request) != len(case.requests):
        raise ValueError("observed request_idx set disagrees with requests.csv")
    latencies = {}
    by_request = {}
    for item in per_request:
        record = _mapping(item, "observed request")
        request_idx = _integer(record["request_idx"], "request_idx")
        by_request[request_idx] = {
            name: record[name] for name in _LATENCY_FIELDS
        }
    for observed_name in _LATENCY_FIELDS:
        values = []
        for item in per_request:
            record = _mapping(item, "observed request")
            value = record[observed_name]
            if value is None:
                raise ValueError(
                    f"first-phase observed metric {observed_name} cannot be null"
                )
            values.append(_finite_number(value, observed_name))
        latencies[observed_name] = _distribution(values)
    throughputs = {
        name: _finite_number(summary[name], name) for name in _THROUGHPUT_FIELDS
    }
    result = {
        "latencies": latencies,
        "throughputs": throughputs,
        "by_request": by_request,
    }
    breakdown = _observed_ttft_breakdown(case, by_request)
    if breakdown is not None:
        result["ttft_breakdown"] = breakdown
    return result


def _validate_completed_payload(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    case: ValidatedCompactCase,
    payload_value: object,
    source_commit: str,
    expected_execution_mode: str,
    debug_num_kv_blocks: int | None = None,
    debug_analytical_launch: bool = False,
    expected_serving_timing_mode: str = "device_only",
) -> Mapping[str, object]:
    payload = _mapping(payload_value, "simulation payload")
    validate_serving_timing_mode(expected_serving_timing_mode)
    if payload["serving_timing_mode"] != expected_serving_timing_mode:
        raise ValueError("simulation payload Serving timing mode mismatch")
    if payload["execution_mode"] != expected_execution_mode:
        raise ValueError("simulation payload execution_mode mismatch")
    midlayer_wall_time = _validate_midlayer_wall_time(payload)
    execution_cache_stats = _validate_execution_cache_stats(
        payload["execution_cache_stats"], expected_execution_mode)
    expected_launch_policy = (
        "analytical_launch"
        if debug_analytical_launch else config.prediction.values["launch_policy"]
    )
    expected_top_level = {
        "backend": config.prediction.values["backend"],
        "deployment": config.prediction.values["deployment"],
        "launch_policy": expected_launch_policy,
        "prediction_method": config.prediction.values["prediction_method"],
        "model_compute_dtype": config.prediction.values["compute_dtype"],
    }
    for field, expected in expected_top_level.items():
        if payload[field] != expected:
            raise ValueError(
                f"simulation payload {field} mismatch: "
                f"{payload[field]!r} != {expected!r}"
            )
    if payload["hardware_ids"] != [cell.hardware.hardware_id]:
        raise ValueError("simulation payload hardware_ids mismatch")

    request_manifest = _mapping(case.manifest["requests"], "manifest requests")
    request_source = _mapping(payload["request_source"], "request source")
    if request_source["kind"] != "canonical_requests":
        raise ValueError("simulation request source must be canonical_requests")
    request_expectations = {
        "source_sha256": cell.requests_sha256,
        "row_count": len(case.requests),
        "request_id_policy": "physical_row_request_id_v1",
        "equal_arrival_order": "physical_row_ascending",
    }
    for field, expected in request_expectations.items():
        if request_source[field] != expected:
            raise ValueError(f"simulation request source {field} mismatch")
    expected_requests_path = (
        Path(case.case_dir) / request_manifest["path"]
    ).resolve()
    if (Path(request_source["input_path"]).resolve() != expected_requests_path
            or Path(request_source["source_path"]).resolve()
            != expected_requests_path):
        raise ValueError("simulation request source path mismatch")
    if request_manifest["sha256"] != cell.requests_sha256:
        raise ValueError("validated manifest requests SHA256 changed from config")
    normalized_request_sha = _sha256(
        request_source["normalized_request_stream_sha256"],
        "normalized request stream SHA256",
    )
    workload_binding = _mapping(
        payload["workload_assumption_binding"], "workload assumption binding"
    )
    if workload_binding["request_stream_sha256"] != normalized_request_sha:
        raise ValueError("workload binding request stream SHA256 mismatch")
    expected_expert_routing = _expected_expert_routing(
        repository_root, cell)
    expected_routing_sha256 = expected_expert_routing.policy_descriptor_sha256
    binding_components = {
        "request_stream_sha256": normalized_request_sha,
        "expert_routing_policy_sha256": expected_routing_sha256,
        "mtp_acceptance_policy_sha256": None,
    }
    if workload_binding["mtp_acceptance_policy_sha256"] is not None:
        raise ValueError("disabled MTP workload binding must be null")
    if workload_binding["composite_sha256"] != _canonical_json_sha256(
        binding_components
    ):
        raise ValueError("workload binding composite SHA256 mismatch")
    arrival_process = _mapping(payload["arrival_process"], "arrival process")
    if (arrival_process["kind"] != "recorded_canonical_arrivals"
            or arrival_process["request_stream_sha256"] != normalized_request_sha):
        raise ValueError("arrival process is not bound to canonical requests")

    model = _mapping(payload["model_config_provenance"], "model provenance")
    if model["compute_dtype"] != config.prediction.values["compute_dtype"]:
        raise ValueError(
            "simulation model provenance compute_dtype differs from the "
            "prediction contract")
    reported_model_id = model["model_id"]
    if reported_model_id is not None and reported_model_id not in (
        cell.model.dataset_model_id,
        Path(cell.model.simulation_model_path).name,
    ):
        raise ValueError("simulation model identity disagrees with phase-1 mapping")

    execution = _mapping(payload["execution_provenance"], "execution provenance")
    expected_timing_status = performance_timing_status(
        expected_launch_policy,
        [cell.hardware.hardware_id],
        uses_development_protobridge=(
            (config.prediction.values["tensor_collective_backend"]
             == "protobridge"
             and config.prediction.values["tensor_parallel_size"] > 1)
            or config.prediction.values["deployment"] in (
                "af_disagg", "pd_disagg", "pd_af_disagg")),
    )
    if execution["performance_timing_status"] != expected_timing_status:
        raise ValueError("execution performance timing status mismatch")
    hardware_configs = _mapping(
        execution["hardware_configs"], "execution hardware configs"
    )
    prefill_hardware = _mapping(
        hardware_configs["prefill"], "prefill hardware provenance"
    )
    expected_hardware_path = (
        Path(repository_root).resolve() / cell.hardware.hardware_config_path
    ).resolve()
    if Path(prefill_hardware["path"]).resolve() != expected_hardware_path:
        raise ValueError("execution hardware config path mismatch")
    if hardware_configs["decode"] is not None:
        raise ValueError("colocated execution cannot carry decode hardware config")
    if tuple(hardware_configs["additional_colocated"]) != ():
        raise ValueError("phase-1 execution requires one colocated instance")
    if tuple(execution["loaded_hardware_ids"]) != (cell.hardware.hardware_id,):
        raise ValueError("execution provenance hardware IDs mismatch")
    if execution["prediction_method"] != config.prediction.values[
        "prediction_method"
    ]:
        raise ValueError("execution provenance prediction method mismatch")
    collective_backend = config.prediction.values[
        "tensor_collective_backend"]
    if (collective_backend == "protobridge"
            and config.prediction.values["tensor_parallel_size"] > 1):
        expected_external_predictor_inputs = {
            "protobridge": {
                "backend": "protobridge",
                "source_kind": "development_only",
                "profile_sha256_by_operation": {
                    "all_reduce": config.prediction.values[
                        "tensor_all_reduce_profile_sha256"],
                    "all_gather": config.prediction.values[
                        "tensor_all_gather_profile_sha256"],
                },
            },
        }
    elif collective_backend in (
            "analytical_roofline_v1", "protobridge"):
        expected_external_predictor_inputs = None
    else:
        raise AssertionError(
            "validated tensor collective backend became unreachable")
    if (execution["external_predictor_inputs"]
            != expected_external_predictor_inputs):
        raise ValueError(
            "execution ProtoBridge backend/profile provenance mismatch")
    launch_inputs = _mapping(execution["launch_inputs"], "launch inputs")
    if launch_inputs["launch_policy"] != expected_launch_policy:
        raise ValueError("execution provenance launch policy mismatch")
    runtime_inputs = _mapping(execution["runtime_inputs"], "runtime inputs")
    expert_routing = _mapping(
        runtime_inputs["expert_routing"], "expert routing input"
    )
    runtime_expectations = {
        "expert_routing": expected_expert_routing.manifest(),
        "attention_kv_layout": config.prediction.values["attention_kv_layout"],
        "attention_kv_distribution": config.prediction.values[
            "attention_kv_distribution"
        ],
        "graph_execution": cell.hardware.graph_execution.manifest(),
    }
    for field, expected in runtime_expectations.items():
        if runtime_inputs[field] != expected:
            raise ValueError(f"execution runtime input {field} mismatch")
    if config.prediction.values["expert_parallel_size"] > 1:
        routing_expectations = expected_expert_routing.manifest()
        for field, expected in routing_expectations.items():
            if expert_routing[field] != expected:
                raise ValueError(
                    f"simulation expert routing {field} mismatch"
                )
        routing_sha = _sha256(
            expert_routing["policy_descriptor_sha256"],
            "expert routing policy descriptor SHA256",
        )
        if workload_binding["expert_routing_policy_sha256"] != routing_sha:
            raise ValueError("workload binding expert routing SHA256 mismatch")
    else:
        if expert_routing["source_kind"] != "not_applicable":
            raise ValueError("dense compact-suite execution requires no routing")
        if workload_binding["expert_routing_policy_sha256"] is not None:
            raise ValueError("dense workload binding expert routing must be null")

    mtp = _mapping(payload["mtp_scenario"], "MTP scenario")
    mtp_expectations = {
        "num_speculative_tokens": 0,
        "acceptance_policy": "disabled",
        "fixed_accepted_tokens": 0,
        "acceptance_source_id": "disabled",
        "acceptance_replay": None,
        "validation_status": "disabled",
    }
    for field, expected in mtp_expectations.items():
        if type(mtp[field]) is not type(expected) or mtp[field] != expected:
            raise ValueError(f"simulation MTP scenario {field} mismatch")

    repository = _mapping(execution["repository"], "execution repository")
    if repository["git_commit"] != source_commit:
        raise ValueError(
            "execution repository commit does not match evaluation source commit"
        )

    scheduler_map = _mapping(payload["scheduler_runtime"], "scheduler runtime")
    if set(scheduler_map) != {"main"}:
        raise ValueError("phase-1 scheduler runtime must contain only main")
    capacity = (
        _validate_analytical_capacity(
            repository_root, payload["kv_capacity"], config, cell)
        if debug_num_kv_blocks is None
        else _validate_debug_capacity(
            repository_root, payload["kv_capacity"], config, cell,
            debug_num_kv_blocks)
    )
    scheduler, scheduler_sha = _validate_scheduler_runtime(
        repository_root, scheduler_map["main"], config, cell,
        capacity["num_kv_blocks"])

    artifact_summary = _validate_execution_artifact(
        payload, expected_timing_status
    )
    scheduler_progress = validate_scheduler_progress_manifest(
        payload["scheduler_progress"])
    expected_time_axis = {
        "device_only": "predicted_device_only_virtual_time",
        "device_plus_launch": "predicted_device_plus_launch_virtual_time",
        "device_plus_launch_plus_scheduler": (
            "predicted_device_plus_launch_plus_scheduler_virtual_time"),
    }[expected_serving_timing_mode]
    if scheduler_progress["time_axis"] != expected_time_axis:
        raise ValueError("Scheduler progress time axis disagrees with timing mode")
    if scheduler_progress["summary"]["step_count"] != artifact_summary[
            "counts"]["schedule_steps"]:
        raise ValueError(
            "scheduler progress step count disagrees with execution artifact")
    if debug_num_kv_blocks is not None:
        accuracy_claim_status = "debug_capacity_not_for_accuracy"
    elif expected_timing_status == PROJECTED_PERFORMANCE_TIMING_STATUS:
        accuracy_claim_status = "supported_for_accuracy_interpretation"
    elif expected_timing_status in (
            UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
            DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS):
        accuracy_claim_status = "unsupported_development_only"
    else:
        raise AssertionError("canonical timing status classification is incomplete")

    return {
        "validation_status": "validated_current_forward",
        "model_identity": {
            "dataset_model_id": cell.model.dataset_model_id,
            "simulation_model_basename": Path(
                cell.model.simulation_model_path
            ).name,
            "reported_model_id": reported_model_id,
            "config_sha256": cell.model.model_config_sha256,
            "binding": "explicit_model_path_alias_and_config_sha256",
        },
        "hardware_config_sha256": cell.hardware.hardware_config_sha256,
        "normalized_request_stream_sha256": normalized_request_sha,
        "runtime_inputs": runtime_expectations,
        "expert_routing": dict(expert_routing),
        "scheduler_runtime": scheduler,
        "kv_capacity": capacity,
        "main_chain_status": "executed",
        "performance_timing_status": expected_timing_status,
        "accuracy_claim_status": accuracy_claim_status,
        "execution_artifact": artifact_summary,
        "scheduler_progress": scheduler_progress,
        "midlayer_wall_time": midlayer_wall_time,
        "execution_mode": expected_execution_mode,
        "execution_cache_stats": execution_cache_stats,
        "execution_provenance_sha256": _canonical_json_sha256(execution),
        "workload_assumption_binding_sha256": _canonical_json_sha256(
            workload_binding
        ),
        "scheduler_runtime_sha256": scheduler_sha,
    }


def _validate_midlayer_wall_time(
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    simulation_wall_time_s = _finite_number(
        payload["simulation_wall_time_s"], "simulation wall time"
    )
    if simulation_wall_time_s < 0.0:
        raise ValueError("simulation wall time must be nonnegative")
    raw_phases = _mapping(
        payload["evaluation_wall_time_breakdown_s"],
        "evaluation wall time breakdown",
    )
    if set(raw_phases) != set(MIDLAYER_WALL_TIME_PHASES):
        raise ValueError("evaluation wall time breakdown fields are invalid")
    phases = {
        name: _finite_number(raw_phases[name], f"{name} wall time")
        for name in MIDLAYER_WALL_TIME_PHASES
    }
    if any(value < 0.0 for value in phases.values()):
        raise ValueError("evaluation wall time breakdown must be nonnegative")
    tolerance = max(1e-9, simulation_wall_time_s * 1e-6)
    if abs(sum(phases.values()) - simulation_wall_time_s) > tolerance:
        raise ValueError(
            "evaluation wall time breakdown does not sum to simulation wall time"
        )
    return {
        "schema_version": 1,
        "scope": "midlayer_simulation_cpu_wall",
        "simulation_wall_time_s": simulation_wall_time_s,
        "phase_wall_time_s": phases,
    }


def _validate_debug_capacity(
    repository_root: str | Path,
    capacity_value: object,
    config: Phase1Config,
    cell: Phase1Cell,
    expected_num_kv_blocks: int,
) -> dict[str, object]:
    """Validate the explicit fixture capacity without claiming HBM evidence."""

    capacity = _mapping(capacity_value, "KV capacity manifest")
    _exact_fields(
        capacity,
        {"schema_version", "source", "scheduler_pools"},
        "KV capacity manifest",
    )
    if capacity["schema_version"] != 2:
        raise ValueError("KV capacity manifest schema_version must equal 2")
    source = _mapping(capacity["source"], "KV capacity source")
    _exact_fields(
        source,
        {
            "mode",
            "runtime_memory_policy_path",
            "runtime_memory_policy_sha256",
        },
        "KV capacity source",
    )
    if source != {
        "mode": "debug_explicit_block_count",
        "runtime_memory_policy_path": None,
        "runtime_memory_policy_sha256": None,
    }:
        raise ValueError("debug evaluation requires explicit debug KV capacity")
    pools = _sequence(capacity["scheduler_pools"], "KV capacity pools")
    if len(pools) != 1:
        raise ValueError("debug evaluation requires exactly one KV capacity pool")
    pool = _mapping(pools[0], "main KV capacity pool")
    _exact_fields(
        pool,
        {
            "pool_name", "source_mode", "hardware_id",
            "model_config_sha256", "worker_role", "tensor_parallel_size",
            "pipeline_parallel_size", "expert_parallel_size",
            "kv_block_size_tokens", "kv_bytes_per_token", "kv_block_bytes",
            "num_kv_blocks", "runtime_memory_policy", "rank_budgets",
            "native_vllm_allocation",
        },
        "main KV capacity pool",
    )
    expected_none_fields = (
        "hardware_id",
        "model_config_sha256",
        "worker_role",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
        "runtime_memory_policy",
        "native_vllm_allocation",
    )
    if (pool["pool_name"] != "main"
            or pool["source_mode"] != "debug_explicit_block_count"
            or any(pool[field] is not None for field in expected_none_fields)
            or tuple(_sequence(
                pool["rank_budgets"], "debug KV capacity rank budgets"))):
        raise ValueError("debug KV capacity carries analytical identity")
    model = load_model_config(str(_model_config_path(
        repository_root, cell.model.simulation_model_path)))
    prediction = config.prediction.values
    expected_kv_bytes_per_token = model.kv_bytes_per_token_per_parallel_rank(
        prediction["tensor_parallel_size"],
        prediction["pipeline_parallel_size"],
    )
    expected_values = {
        "kv_block_size_tokens": prediction["kv_block_size_tokens"],
        "kv_bytes_per_token": expected_kv_bytes_per_token,
        "kv_block_bytes": (
            prediction["kv_block_size_tokens"] * expected_kv_bytes_per_token
        ),
        "num_kv_blocks": expected_num_kv_blocks,
    }
    for field, expected in expected_values.items():
        if pool[field] != expected:
            raise ValueError(f"debug KV capacity {field} mismatch")
    return {
        "source_mode": "debug_explicit_block_count",
        "runtime_memory_policy_path": None,
        "runtime_memory_policy_sha256": None,
        "num_kv_blocks": expected_num_kv_blocks,
        "available_kv_bytes_by_rank": (),
        "allocator_symbol": None,
        "allocator_version": None,
        "allocator_source_commit": None,
    }


def _validate_analytical_capacity(
    repository_root: str | Path,
    capacity_value: object,
    config: Phase1Config,
    cell: Phase1Cell,
) -> dict[str, object]:
    prediction = config.prediction.values
    tensor_parallel_size = prediction["tensor_parallel_size"]
    pipeline_parallel_size = prediction["pipeline_parallel_size"]
    expert_parallel_size = prediction["expert_parallel_size"]
    worker_count = tensor_parallel_size * pipeline_parallel_size
    capacity = _mapping(capacity_value, "KV capacity manifest")
    _exact_fields(
        capacity,
        {"schema_version", "source", "scheduler_pools"},
        "KV capacity manifest",
    )
    if capacity["schema_version"] != 2:
        raise ValueError("KV capacity manifest schema_version must equal 2")
    source = _mapping(capacity["source"], "KV capacity source")
    _exact_fields(
        source,
        {
            "mode",
            "runtime_memory_policy_path",
            "runtime_memory_policy_sha256",
        },
        "KV capacity source",
    )
    if source["mode"] != "analytical":
        raise ValueError("phase-1 Forward requires analytical KV capacity")
    expected_policy_path = Path(_expand_runtime_memory_policy_path(
        repository_root, config, cell, cell.model.dataset_model_id)).resolve()
    if Path(source["runtime_memory_policy_path"]).resolve() != expected_policy_path:
        raise ValueError("KV capacity runtime memory policy path mismatch")
    policy_sha256 = _sha256(
        source["runtime_memory_policy_sha256"],
        "runtime memory policy SHA256",
    )
    pools = _sequence(capacity["scheduler_pools"], "KV capacity pools")
    if len(pools) != 1:
        raise ValueError("phase-1 Forward requires exactly one KV capacity pool")
    pool = _mapping(pools[0], "main KV capacity pool")
    _exact_fields(
        pool,
        {
            "pool_name", "source_mode", "hardware_id",
            "model_config_sha256", "worker_role", "tensor_parallel_size",
            "pipeline_parallel_size", "expert_parallel_size",
            "kv_block_size_tokens", "kv_bytes_per_token", "kv_block_bytes",
            "num_kv_blocks", "runtime_memory_policy", "rank_budgets",
            "native_vllm_allocation",
        },
        "main KV capacity pool",
    )
    expected_identity = {
        "pool_name": "main",
        "source_mode": "analytical",
        "hardware_id": cell.hardware.hardware_id,
        "model_config_sha256": cell.model.model_config_sha256,
        "worker_role": "colocated",
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "kv_block_size_tokens": prediction["kv_block_size_tokens"],
    }
    for field, expected in expected_identity.items():
        if pool[field] != expected:
            raise ValueError(f"KV capacity pool {field} mismatch")
    for field in ("kv_bytes_per_token", "kv_block_bytes", "num_kv_blocks"):
        if _integer(pool[field], f"KV capacity {field}") < 1:
            raise ValueError(f"KV capacity {field} must be positive")
    if pool["kv_block_bytes"] != (
            pool["kv_block_size_tokens"] * pool["kv_bytes_per_token"]):
        raise ValueError("KV capacity block-byte accounting mismatch")
    rank_budgets = _sequence(pool["rank_budgets"], "KV capacity rank budgets")
    if len(rank_budgets) != worker_count:
        raise ValueError("KV capacity rank count must equal TP x PP")
    model = load_model_config(str(_model_config_path(
        repository_root, cell.model.simulation_model_path)))
    if model.num_hidden_layers % pipeline_parallel_size:
        raise ValueError("KV capacity model layers must divide pipeline parallelism")
    local_layers = model.num_hidden_layers // pipeline_parallel_size
    expected_kv_bytes_per_token = (
        model.kv_bytes_per_token_per_parallel_rank(
            tensor_parallel_size,
            pipeline_parallel_size,
        )
    )
    if pool["kv_bytes_per_token"] != expected_kv_bytes_per_token:
        raise ValueError("KV capacity bytes per token disagree with model geometry")
    normalized_rank_budgets = []
    for expected_rank, rank_value in enumerate(rank_budgets):
        rank = _mapping(rank_value, "KV capacity rank budget")
        _exact_fields(
            rank,
            {
                "rank", "tensor_parallel_rank", "pipeline_parallel_rank",
                "expert_parallel_rank", "layer_start", "layer_end",
                "physical_hbm_bytes", "utilized_hbm_bytes",
                "model_weight_resident_bytes", "non_kv_reserve_bytes",
                "available_kv_bytes",
            },
            "KV capacity rank budget",
        )
        for field in rank:
            minimum = 1 if field.endswith("_bytes") else 0
            if _integer(rank[field], f"KV capacity rank {field}") < minimum:
                raise ValueError(
                    f"KV capacity rank {field} must be at least {minimum}")
        expected_tp_rank = expected_rank % tensor_parallel_size
        expected_pp_rank = expected_rank // tensor_parallel_size
        expected_ep_rank = expected_tp_rank if expert_parallel_size > 1 else 0
        expected_coordinates = {
            "rank": expected_rank,
            "tensor_parallel_rank": expected_tp_rank,
            "pipeline_parallel_rank": expected_pp_rank,
            "expert_parallel_rank": expected_ep_rank,
            "layer_start": expected_pp_rank * local_layers,
            "layer_end": (expected_pp_rank + 1) * local_layers,
        }
        if any(
            rank[field] != expected
            for field, expected in expected_coordinates.items()
        ):
            raise ValueError("KV capacity rank coordinates disagree with TP/PP/EP")
        if rank["available_kv_bytes"] != (
                rank["utilized_hbm_bytes"]
                - rank["model_weight_resident_bytes"]
                - rank["non_kv_reserve_bytes"]):
            raise ValueError("KV capacity rank byte accounting mismatch")
        normalized_rank_budgets.append(rank)
    policy = _mapping(
        pool["runtime_memory_policy"], "KV capacity runtime memory policy")
    _exact_fields(
        policy,
        {
            "binding", "gpu_memory_utilization", "non_kv_reserves",
            "non_kv_reserve_bytes", "reserve_scope",
            "completeness_assertion",
        },
        "KV capacity runtime memory policy",
    )
    if policy["completeness_assertion"] != (
            "all_non_kv_reserves_for_bound_runtime"):
        raise ValueError("KV capacity policy completeness assertion mismatch")
    if policy["reserve_scope"] != "verified_uniform_across_pool_ranks":
        raise ValueError("KV capacity policy reserve scope mismatch")
    binding = _mapping(policy["binding"], "KV capacity policy binding")
    expected_binding = {
        "hardware_id": cell.hardware.hardware_id,
        "hardware_config_sha256": cell.hardware.hardware_config_sha256,
        "model_config_sha256": cell.model.model_config_sha256,
        "deployment": prediction["deployment"],
        "worker_role": "colocated",
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "max_model_len": prediction["max_model_len"],
        "max_batched_tokens": prediction["max_batched_tokens"],
        "max_requests": prediction["max_requests"],
        "kv_block_size_tokens": prediction["kv_block_size_tokens"],
    }
    _exact_fields(binding, set(expected_binding), "KV capacity policy binding")
    if any(
        binding[field] != expected
        for field, expected in expected_binding.items()
    ):
        raise ValueError("KV capacity policy binding disagrees with suite contract")
    allocation = _mapping(
        pool["native_vllm_allocation"], "native vLLM KV allocation")
    required_allocation_fields = {
        "version", "source_commit", "package_python_manifest_sha256",
        "allocator_symbol", "scheduler_config_symbol", "kv_spec_type",
        "kv_block_size_tokens", "max_model_len",
        "kv_page_size_bytes_per_layer",
        "num_kv_blocks", "worker_num_kv_blocks",
        "worker_allocated_tensor_bytes", "scheduler_tensor_owners",
    }
    _exact_fields(
        allocation, required_allocation_fields, "native vLLM KV allocation")
    if (allocation["version"] != VLLM_SCHEDULER_VERSION
            or allocation["source_commit"] != VLLM_SCHEDULER_SOURCE_COMMIT):
        raise ValueError("native KV allocator identity mismatch")
    if allocation["allocator_symbol"] != (
            "vllm.v1.core.kv_cache_utils.get_kv_cache_configs"):
        raise ValueError("native KV allocator symbol mismatch")
    if allocation["scheduler_config_symbol"] != (
            "vllm.v1.core.kv_cache_utils.generate_scheduler_kv_cache_config"):
        raise ValueError("native Scheduler KV config symbol mismatch")
    if allocation["num_kv_blocks"] != pool["num_kv_blocks"]:
        raise ValueError("native and analytical KV block counts disagree")
    if allocation["max_model_len"] != config.prediction.values["max_model_len"]:
        raise ValueError("native KV allocator max_model_len mismatch")
    if allocation["kv_block_size_tokens"] != prediction["kv_block_size_tokens"]:
        raise ValueError("native KV allocator block size mismatch")
    expected_page_size = (
        prediction["kv_block_size_tokens"]
        * 2
        * model.kv_heads_per_tensor_parallel_rank(tensor_parallel_size)
        * model.head_dim
        * model.dtype_bytes
    )
    if allocation["kv_page_size_bytes_per_layer"] != expected_page_size:
        raise ValueError("native KV allocator page geometry mismatch")
    worker_blocks = _sequence(
        allocation["worker_num_kv_blocks"],
        "native KV allocation worker block counts",
    )
    worker_tensor_bytes = _sequence(
        allocation["worker_allocated_tensor_bytes"],
        "native KV allocation worker tensor bytes",
    )
    if (
        len(worker_blocks) != worker_count
        or len(worker_tensor_bytes) != worker_count
    ):
        raise ValueError("native KV allocator worker count must equal TP x PP")
    if any(value != pool["num_kv_blocks"] for value in worker_blocks):
        raise ValueError("native KV allocator worker block counts disagree")
    expected_allocated_bytes = pool["num_kv_blocks"] * pool["kv_block_bytes"]
    if any(value != expected_allocated_bytes for value in worker_tensor_bytes):
        raise ValueError("native KV allocator worker tensor bytes disagree")
    scheduler_owners = tuple(
        tuple(_sequence(value, "native KV tensor owners"))
        for value in _sequence(
            allocation["scheduler_tensor_owners"],
            "native KV Scheduler tensor owners",
        )
    )
    expected_owners = tuple(
        (f"model.layers.{index}.self_attn.attn",)
        for index in range(local_layers)
    )
    if scheduler_owners != expected_owners:
        raise ValueError("native KV Scheduler tensor ownership mismatch")
    return {
        "source_mode": "analytical",
        "runtime_memory_policy_path": str(expected_policy_path),
        "runtime_memory_policy_sha256": policy_sha256,
        "num_kv_blocks": pool["num_kv_blocks"],
        "available_kv_bytes_by_rank": tuple(
            item["available_kv_bytes"] for item in normalized_rank_budgets
        ),
        "allocator_symbol": allocation["allocator_symbol"],
        "allocator_version": allocation["version"],
        "allocator_source_commit": allocation["source_commit"],
    }


def _validate_scheduler_runtime(
    repository_root: str | Path,
    scheduler_value: object,
    config: Phase1Config,
    cell: Phase1Cell,
    expected_num_kv_blocks: int,
) -> tuple[dict[str, object], str]:
    scheduler = _mapping(scheduler_value, "main scheduler runtime")
    runtime_fields = {field.name for field in fields(VLLMSchedulerRuntimeInfo)}
    if set(scheduler) != runtime_fields:
        raise ValueError(
            "phase-1 Scheduler runtime fields disagree with the pinned contract"
        )
    annotations = get_type_hints(VLLMSchedulerRuntimeInfo)
    for field_name, annotation in annotations.items():
        if not _runtime_value_matches_type(scheduler[field_name], annotation):
            raise TypeError(
                f"phase-1 Scheduler runtime {field_name} has the wrong type"
            )
    runtime = VLLMSchedulerRuntimeInfo(**scheduler)

    source_identity = vllm_scheduler_source_identity()
    for field_name, expected in source_identity.items():
        if scheduler[field_name] != expected:
            raise ValueError(
                f"effective Scheduler source identity {field_name} mismatch"
            )
    prediction = config.prediction.values
    tensor_parallel_size = prediction["tensor_parallel_size"]
    pipeline_parallel_size = prediction["pipeline_parallel_size"]
    expert_parallel_enabled = prediction["expert_parallel_size"] > 1
    fixed_values = {
        "platform": "cpu",
        "model_config_sha256": cell.model.model_config_sha256,
        "requested_dtype": "bfloat16",
        "effective_dtype": "bfloat16",
        "requested_tensor_parallel_size": tensor_parallel_size,
        "effective_tensor_parallel_size": tensor_parallel_size,
        "requested_pipeline_parallel_size": pipeline_parallel_size,
        "effective_pipeline_parallel_size": pipeline_parallel_size,
        "requested_expert_parallel": expert_parallel_enabled,
        "effective_expert_parallel": expert_parallel_enabled,
        "scheduler_pipeline_stage_rank": 0,
        "scheduler_pipeline_stage_layer_start": 0,
        "device_guard": "python_tripwire+cuda_state+procfs",
        "bootstrap": "version_pinned_scheduler_only_cpu",
        "requested_chunked_prefill": prediction["enable_chunked_prefill"],
        "effective_chunked_prefill": prediction["enable_chunked_prefill"],
        "requested_prefix_caching": prediction["enable_prefix_caching"],
        # The pinned CPU Scheduler deliberately disables vLLM's content-hash
        # cache. Logical reuse comes from the trace-length prefix oracle.
        "effective_prefix_caching": False,
        "prefix_oracle_mode": "trace_length_private_nonshared_v1",
        "speculative_method": "disabled",
        "mtp_num_speculative_tokens": prediction["mtp_num_speculative_tokens"],
        "mtp_acceptance_policy": prediction["mtp_acceptance_policy"],
        "mtp_fixed_accepted_tokens": prediction["mtp_fixed_accepted_tokens"],
        "mtp_acceptance_source_id": prediction["mtp_acceptance_source_id"],
        "mtp_acceptance_replay_path": None,
        "mtp_acceptance_replay_sha256": None,
        "mtp_acceptance_replay_artifact_id": None,
        "mtp_acceptance_replay_step_count": None,
        "mtp_acceptance_assumption_sha256": None,
        "mtp_request_stream_sha256": None,
        "mtp_padded_drafter_batch": False,
        "mtp_draft_model_type": "disabled",
        "mtp_num_mtp_layers": 0,
        "mtp_draft_max_model_len": 0,
        "max_num_batched_tokens": prediction["max_batched_tokens"],
        "max_num_seqs": prediction["max_requests"],
        "max_model_len": prediction["max_model_len"],
        "max_num_partial_prefills": 1,
        "max_long_partial_prefills": 1,
        "long_prefill_token_threshold": 0,
        "kv_block_size_tokens": prediction["kv_block_size_tokens"],
        "num_kv_blocks": expected_num_kv_blocks,
        "scheduler_policy": "fcfs",
        "num_lookahead_slots": 0,
        "attention_kv_layout": prediction["attention_kv_layout"],
        "backend_role": prediction["deployment"],
        "kv_handoff_mode": "not_applicable",
    }
    for field_name, expected in fixed_values.items():
        if scheduler[field_name] != expected:
            raise ValueError(f"effective Scheduler runtime {field_name} mismatch")
    if scheduler["requested_async_scheduling"] != scheduler[
            "effective_async_scheduling"]:
        raise ValueError("effective Scheduler async mode mismatch")

    model = load_model_config(str(_model_config_path(
        repository_root, cell.model.simulation_model_path)))
    if model.attention_layout != "standard_gqa":
        raise ValueError("phase-1 Scheduler KV geometry requires standard GQA")
    if model.num_hidden_layers % pipeline_parallel_size:
        raise ValueError("Scheduler model layers must divide pipeline parallelism")
    local_layer_count = model.num_hidden_layers // pipeline_parallel_size
    kv_head_count = model.kv_heads_per_tensor_parallel_rank(
        tensor_parallel_size)
    expected_page_size = (
        scheduler["kv_block_size_tokens"]
        * 2
        * kv_head_count
        * model.head_dim
        * model.dtype_bytes
    )
    expected_owners = tuple(
        (f"model.layers.{index}.self_attn.attn",)
        for index in range(local_layer_count)
    )
    if (scheduler["scheduler_pipeline_stage_layer_end"]
            != local_layer_count
            or scheduler["scheduler_kv_cache_num_heads_per_rank"]
            != kv_head_count
            or scheduler["scheduler_kv_cache_page_size_bytes"]
            != expected_page_size
            or scheduler["scheduler_kv_cache_tensor_owners"]
            != expected_owners):
        raise ValueError(
            "effective Scheduler KV geometry disagrees with the frozen model"
        )
    return dict(scheduler), vllm_scheduler_runtime_sha256(runtime)


def _runtime_value_matches_type(value: object, annotation: object) -> bool:
    origin = get_origin(annotation)
    if origin is types.UnionType:
        return any(
            _runtime_value_matches_type(value, member)
            for member in get_args(annotation)
        )
    if origin is tuple:
        if type(value) is not tuple:
            return False
        members = get_args(annotation)
        if len(members) == 2 and members[1] is Ellipsis:
            return all(
                _runtime_value_matches_type(item, members[0]) for item in value
            )
        return len(value) == len(members) and all(
            _runtime_value_matches_type(item, member)
            for item, member in zip(value, members)
        )
    if annotation is type(None):
        return value is None
    if annotation is float:
        return type(value) is float and math.isfinite(value)
    if annotation in (bool, int, str):
        return type(value) is annotation
    raise TypeError(f"unsupported Scheduler runtime annotation: {annotation!r}")


def _validate_execution_artifact(
    payload: Mapping[str, object],
    expected_timing_status: str,
) -> dict[str, object]:
    artifact = _mapping(payload["execution_artifact"], "execution artifact")
    if artifact["schema_version"] in STREAMING_EXECUTION_ARTIFACT_HASH_METHODS:
        return _validate_streaming_execution_artifact(
            payload, artifact, expected_timing_status)
    _exact_fields(artifact, EXECUTION_ARTIFACT_FIELDS, "execution artifact")
    schema_version = artifact["schema_version"]
    if type(schema_version) is not int or schema_version not in (5, 7, 8):
        raise ValueError("execution artifact schema_version must be 5, 7, or 8")
    if artifact["hash_method"] != _EXECUTION_ARTIFACT_HASH_METHOD:
        raise ValueError("execution artifact hash_method mismatch")
    timing_status = artifact["performance_timing_status"]
    if (timing_status != expected_timing_status
            or payload["performance_timing_status"] != timing_status):
        raise ValueError("execution artifact performance timing status mismatch")
    counts = _mapping(artifact["counts"], "execution artifact counts")
    _exact_fields(
        counts,
        _EXECUTION_ARTIFACT_COUNT_FIELDS,
        "execution artifact counts",
    )
    if any(type(counts[field]) is not int for field in counts):
        raise TypeError("execution artifact counts must be exact integers")
    if any(counts[field] < 0 for field in counts):
        raise ValueError("execution artifact counts must be non-negative")

    schedule_trace = _mapping(
        artifact["schedule_trace"], "execution artifact schedule trace"
    )
    _exact_fields(
        schedule_trace,
        {
            "route_decision_version",
            "transfer_trace_version",
            "entries",
            "route_decisions",
            "transfers",
        },
        "execution artifact schedule trace",
    )
    if (type(schedule_trace["route_decision_version"]) is not int
            or schedule_trace["route_decision_version"] != 2
            or type(schedule_trace["transfer_trace_version"]) is not int
            or schedule_trace["transfer_trace_version"] != 2):
        raise ValueError("execution artifact trace versions mismatch")
    schedule_entries = _artifact_tuple(
        schedule_trace["entries"], "execution artifact schedule entries"
    )
    _artifact_tuple(
        schedule_trace["route_decisions"],
        "execution artifact route decisions",
    )
    trace_transfers = _artifact_tuple(
        schedule_trace["transfers"], "execution artifact trace transfers"
    )
    steps = _artifact_tuple(artifact["steps"], "execution artifact steps")
    transfers = _artifact_tuple(
        artifact["transfers"], "execution artifact transfers"
    )
    modeled_kv_offload = _mapping(
        artifact["modeled_kv_offload"], "execution artifact modeled KV offload"
    )
    modeled_transfers = _artifact_tuple(
        modeled_kv_offload["transfers"],
        "execution artifact modeled KV transfers",
    )
    availability = _string(
        modeled_kv_offload["availability"],
        "execution artifact modeled KV availability",
    )
    if availability == "not_enabled":
        _exact_fields(
            modeled_kv_offload,
            {
                "schema_version",
                "availability",
                "intent_source",
                "transfers",
                "final_tier_state",
            },
            "disabled modeled KV artifact",
        )
        if (type(modeled_kv_offload["schema_version"]) is not int
                or modeled_kv_offload["schema_version"] != 2
                or schema_version not in (5, 8)
                or modeled_transfers != ()):
            raise ValueError("disabled modeled KV artifact is inconsistent")
    elif availability in (
            "available_modeled_rank_local",
            "available_modeled_serial",
    ):
        _exact_fields(
            modeled_kv_offload,
            {
                "schema_version",
                "availability",
                "mode",
                "validation_status",
                "intent_source",
                "ledger_semantics",
                "physical_transfer_count",
                "scope",
                "transfers",
                "final_tier_state",
            },
            "available modeled KV artifact",
        )
        physical_transfer_count = _integer(
            modeled_kv_offload["physical_transfer_count"],
            "modeled KV physical transfer count",
        )
        if (type(modeled_kv_offload["schema_version"]) is not int
                or modeled_kv_offload["schema_version"] != 4
                or schema_version not in (7, 8)
                or not modeled_transfers
                or physical_transfer_count < len(modeled_transfers)):
            raise ValueError("available modeled KV artifact schema mismatch")
    else:
        raise ValueError(
            f"unknown modeled KV artifact availability: {availability!r}"
        )

    kernel_calls = sum(
        _artifact_call_count(step, "execution artifact step") for step in steps
    ) + sum(
        _artifact_call_count(transfer, "execution artifact transfer")
        for transfer in transfers
    )
    actual_counts = {
        "schedule_steps": len(schedule_entries),
        "simulation_steps": len(steps),
        "simulation_transfers": len(transfers),
        "trace_transfers": len(trace_transfers),
        "modeled_kv_offload_transfers": len(modeled_transfers),
        "kernel_calls": kernel_calls,
    }
    if dict(counts) != actual_counts:
        raise ValueError("execution artifact counts disagree with artifact body")
    top_level_counts = {
        "num_steps": _integer(payload["num_steps"], "simulation num_steps"),
        "num_transfers": _integer(
            payload["num_transfers"], "simulation num_transfers"
        ),
        "num_kernel_calls": _integer(
            payload["num_kernel_calls"], "simulation num_kernel_calls"
        ),
    }
    if (top_level_counts["num_steps"] != actual_counts["schedule_steps"]
            or top_level_counts["num_steps"]
            != actual_counts["simulation_steps"]
            or top_level_counts["num_transfers"]
            != actual_counts["trace_transfers"]
            or top_level_counts["num_transfers"]
            != actual_counts["simulation_transfers"]
            or top_level_counts["num_kernel_calls"] != kernel_calls):
        raise ValueError(
            "simulation top-level counts disagree with execution artifact"
        )

    artifact_sha = _sha256(
        artifact["artifact_sha256"], "execution artifact SHA256"
    )
    body = {
        field: artifact[field]
        for field in EXECUTION_ARTIFACT_FIELDS
        if field != "artifact_sha256"
    }
    if _streaming_canonical_json_sha256(body) != artifact_sha:
        raise ValueError("execution artifact body SHA256 mismatch")

    main_chain = _mapping(payload["main_chain_status"], "main chain status")
    _exact_fields(
        main_chain,
        {"status", "chain", "execution_artifact_sha256"},
        "main chain status",
    )
    if (main_chain["status"] != "executed"
            or main_chain["chain"] != _MAIN_CHAIN
            or main_chain["execution_artifact_sha256"] != artifact_sha):
        raise ValueError("main chain did not execute the canonical artifact")
    return {
        "schema_version": schema_version,
        "performance_timing_status": timing_status,
        "counts": dict(counts),
        "artifact_sha256": artifact_sha,
    }


def _validate_streaming_execution_artifact(
    payload: Mapping[str, object],
    artifact: Mapping[str, object],
    expected_timing_status: str,
) -> dict[str, object]:
    _exact_fields(
        artifact,
        _STREAMING_EXECUTION_ARTIFACT_FIELDS,
        "streaming execution artifact",
    )
    schema_version = artifact["schema_version"]
    if schema_version not in STREAMING_EXECUTION_ARTIFACT_HASH_METHODS:
        raise ValueError(
            "streaming execution artifact schema_version must be 9, 10, 11, 12, or 13")
    if (artifact["hash_method"]
            != STREAMING_EXECUTION_ARTIFACT_HASH_METHODS[schema_version]):
        raise ValueError("streaming execution artifact hash_method mismatch")
    timing_status = artifact["performance_timing_status"]
    if (timing_status != expected_timing_status
            or payload["performance_timing_status"] != timing_status):
        raise ValueError(
            "streaming execution artifact performance timing status mismatch")

    counts = _mapping(
        artifact["counts"], "streaming execution artifact counts")
    _exact_fields(
        counts,
        _EXECUTION_ARTIFACT_COUNT_FIELDS,
        "streaming execution artifact counts",
    )
    if any(type(counts[field]) is not int for field in counts):
        raise TypeError(
            "streaming execution artifact counts must be exact integers")
    if any(counts[field] < 0 for field in counts):
        raise ValueError(
            "streaming execution artifact counts must be non-negative")
    if (counts["schedule_steps"] != counts["simulation_steps"]
            or counts["modeled_kv_offload_transfers"] != 0):
        raise ValueError(
            "streaming execution artifact counts violate the retention scope")

    streaming = _mapping(
        artifact["streaming_aggregate"],
        "streaming execution artifact aggregate",
    )
    chain_field = (
        "ordered_step_manifest_chain_sha256"
        if schema_version == 9
        else "ordered_step_execution_contract_chain_sha256"
    )
    streaming_fields = {
        "retention",
        "representation",
        chain_field,
        "request_level_artifact",
    }
    if schema_version == 13:
        streaming_fields.add("transfer_summary")
    _exact_fields(streaming, streaming_fields,
                  "streaming execution artifact aggregate")
    if streaming["retention"] != "step_ir_consumed_then_released":
        raise ValueError("streaming execution artifact retention mismatch")
    if streaming["representation"] not in (
            "full_step_ir_consumed_without_retention",
            "compact_step_ir_consumed_without_retention"):
        raise ValueError("streaming execution artifact representation mismatch")
    _sha256(
        streaming[chain_field],
        "streaming execution artifact step chain SHA256",
    )
    if streaming["request_level_artifact"] != "not_retained":
        raise ValueError(
            "streaming execution artifact request-level retention mismatch")
    if schema_version == 13:
        transfer_summary = _mapping(
            streaming["transfer_summary"],
            "streaming execution artifact transfer summary",
        )
        _exact_fields(
            transfer_summary,
            {
                "retention",
                "trace_transfer_count",
                "simulation_transfer_count",
                "trace_transfer_manifest_sha256",
            },
            "streaming execution artifact transfer summary",
        )
        if transfer_summary["retention"] != (
                "trace_manifest_digest_and_counts"):
            raise ValueError(
                "streaming execution artifact transfer retention mismatch")
        for field in ("trace_transfer_count", "simulation_transfer_count"):
            if (type(transfer_summary[field]) is not int
                    or transfer_summary[field] < 0):
                raise ValueError(
                    "streaming execution artifact transfer counts must be "
                    "non-negative integers")
        if (transfer_summary["trace_transfer_count"]
                != counts["trace_transfers"]
                or transfer_summary["simulation_transfer_count"]
                != counts["simulation_transfers"]):
            raise ValueError(
                "streaming execution artifact transfer summary disagrees "
                "with counts")
        _sha256(
            transfer_summary["trace_transfer_manifest_sha256"],
            "streaming transfer manifest SHA256",
        )
        if _streaming_canonical_json_sha256(payload["transfers"]) != (
                transfer_summary["trace_transfer_manifest_sha256"]):
            raise ValueError(
                "streaming transfer manifest digest disagrees with payload")
    elif (counts["simulation_transfers"] != 0
          or counts["trace_transfers"] != 0):
        raise ValueError(
            "legacy streaming execution artifact cannot retain transfers")

    top_level_counts = {
        "num_steps": _integer(payload["num_steps"], "simulation num_steps"),
        "num_transfers": _integer(
            payload["num_transfers"], "simulation num_transfers"),
        "num_kernel_calls": _integer(
            payload["num_kernel_calls"], "simulation num_kernel_calls"),
    }
    if (top_level_counts["num_steps"] != counts["schedule_steps"]
            or top_level_counts["num_steps"] != counts["simulation_steps"]
            or (schema_version != 13
                and top_level_counts["num_transfers"] != 0)
            or (schema_version == 13
                and top_level_counts["num_transfers"]
                != counts["trace_transfers"])
            or top_level_counts["num_kernel_calls"] != counts["kernel_calls"]):
        raise ValueError(
            "simulation top-level counts disagree with streaming artifact")

    artifact_sha = _sha256(
        artifact["artifact_sha256"],
        "streaming execution artifact SHA256",
    )
    body = {
        field: artifact[field]
        for field in _STREAMING_EXECUTION_ARTIFACT_FIELDS
        if field != "artifact_sha256"
    }
    if _streaming_canonical_json_sha256(body) != artifact_sha:
        raise ValueError("streaming execution artifact body SHA256 mismatch")

    main_chain = _mapping(payload["main_chain_status"], "main chain status")
    _exact_fields(
        main_chain,
        {"status", "chain", "execution_artifact_sha256"},
        "main chain status",
    )
    if (main_chain["status"] != "executed"
            or main_chain["chain"] != _MAIN_CHAIN
            or main_chain["execution_artifact_sha256"] != artifact_sha):
        raise ValueError("main chain did not execute the streaming artifact")
    return {
        "schema_version": schema_version,
        "performance_timing_status": timing_status,
        "counts": dict(counts),
        "artifact_sha256": artifact_sha,
    }


def _artifact_tuple(value: object, label: str) -> tuple[object, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a tuple from the in-memory producer")
    return value


def _artifact_call_count(value: object, label: str) -> int:
    record = _mapping(value, label)
    kernel_plan = _mapping(record["kernel_plan"], f"{label} kernel plan")
    count_field = (
        "logical_call_count"
        if "logical_call_count" in kernel_plan else "call_count"
    )
    count = _integer(kernel_plan[count_field], f"{label} kernel call count")
    if count < 0:
        raise ValueError(f"{label} kernel call count must be non-negative")
    return count


def _streaming_canonical_json_sha256(payload: object) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    for chunk in encoder.iterencode(payload):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _compare_prediction(
    observed: Mapping[str, object],
    payload_value: object,
    request_count: int,
) -> Mapping[str, object]:
    payload = _mapping(payload_value, "simulation payload")
    if payload["backend"] != "vllm" or payload["deployment"] != "colocated":
        raise ValueError("phase-1 prediction returned the wrong runtime contract")
    metrics = _mapping(payload["metrics"], "simulation metrics")
    predicted_requests = _mapping(metrics["requests"], "predicted requests")
    expected_ids = {f"request-{idx:012d}" for idx in range(request_count)}
    if set(predicted_requests) != expected_ids:
        raise ValueError("predicted request IDs disagree with canonical request_idx")
    predicted_distributions = {}
    paired_errors = {}
    paired_latency = {}
    observed_by_request = _mapping(
        observed["by_request"], "observed requests by index"
    )
    for observed_name, predicted_name in _LATENCY_FIELDS.items():
        values = []
        absolute_errors = []
        percentage_errors = []
        paired_values = []
        for request_idx in range(request_count):
            request_id = f"request-{request_idx:012d}"
            record = _mapping(predicted_requests[request_id], "predicted request")
            value = record[predicted_name]
            if value is None:
                raise ValueError(
                    f"first-phase predicted metric {predicted_name} cannot be null"
                )
            predicted_ms = 1000.0 * _finite_number(value, predicted_name)
            values.append(predicted_ms)
            observed_record = _mapping(
                observed_by_request[request_idx], "observed request metric"
            )
            observed_ms = _finite_number(
                observed_record[observed_name], observed_name
            )
            paired_values.append({
                "request_idx": request_idx,
                "measured_ms": observed_ms,
                "predicted_ms": predicted_ms,
            })
            absolute_error = abs(predicted_ms - observed_ms)
            absolute_errors.append(absolute_error)
            if observed_ms != 0:
                percentage_errors.append(absolute_error / abs(observed_ms) * 100.0)
        predicted_distributions[observed_name] = _distribution(values)
        paired_errors[observed_name] = _paired_error_summary(
            absolute_errors,
            percentage_errors,
            request_count,
        )
        paired_latency[observed_name] = paired_values

    observed_latency = _mapping(observed["latencies"], "observed latencies")
    latency_comparisons = {}
    for metric_name in _LATENCY_FIELDS:
        observed_distribution = _mapping(
            observed_latency[metric_name], f"observed {metric_name}"
        )
        predicted_distribution = predicted_distributions[metric_name]
        latency_comparisons[metric_name] = {
            statistic: _comparison(
                observed_distribution[statistic],
                predicted_distribution[statistic],
            )
            for statistic in STATISTICS
        }

    observed_throughputs = _mapping(
        observed["throughputs"], "observed throughputs"
    )
    throughput_comparisons = {}
    for observed_name, predicted_name in _THROUGHPUT_FIELDS.items():
        throughput_comparisons[observed_name] = _comparison(
            observed_throughputs[observed_name],
            _finite_number(metrics[predicted_name], predicted_name),
        )
    result = {
        "latency": latency_comparisons,
        "latency_quantiles_p0_p100": {
            "comparison_kind": "measured_vs_predicted",
            "measured": _canonical_latency_quantile_projection(
                observed_by_request,
                request_count,
                predicted=False,
            ),
            "predicted": _canonical_latency_quantile_projection(
                predicted_requests,
                request_count,
                predicted=True,
            ),
        },
        "paired_request_error": paired_errors,
        "paired_request_latency": {
            "schema_version": 1,
            "comparison_kind": "measured_vs_predicted_by_request_idx",
            "request_count": request_count,
            "metrics": paired_latency,
        },
        "throughput": throughput_comparisons,
    }
    if "ttft_breakdown" in observed:
        arrivals = _sequence(
            payload["request_arrivals_s"], "predicted request arrivals"
        )
        arrival_by_id = {}
        for arrival_value in arrivals:
            arrival = _mapping(arrival_value, "predicted request arrival")
            request_id = str(arrival["request_id"])
            if request_id in arrival_by_id:
                raise ValueError("predicted request arrival IDs must be unique")
            arrival_by_id[request_id] = _finite_number(
                arrival["arrival_time_s"], "predicted request arrival time"
            )
        if set(arrival_by_id) != expected_ids:
            raise ValueError("predicted request arrivals disagree with requests")
        observed_breakdown_manifest = _mapping(
            observed["ttft_breakdown"], "observed TTFT breakdown"
        )
        observed_breakdown = _mapping(
            observed_breakdown_manifest["records"],
            "observed TTFT breakdown records",
        )
        records = []
        for request_idx in range(request_count):
            request_id = f"request-{request_idx:012d}"
            observed_record = _mapping(
                observed_breakdown[request_idx], "observed TTFT breakdown record"
            )
            predicted_record = _mapping(
                predicted_requests[request_id], "predicted request"
            )
            predicted_first_schedule_s = _finite_number(
                predicted_record["first_schedule_start_s"],
                "predicted first schedule start",
            )
            predicted_wait_ms = 1000.0 * (
                predicted_first_schedule_s - arrival_by_id[request_id]
            )
            if predicted_wait_ms < -1e-6:
                raise ValueError(
                    "predicted first schedule starts before request arrival"
                )
            predicted_wait_ms = max(0.0, predicted_wait_ms)
            predicted_ttft_ms = 1000.0 * _finite_number(
                predicted_record["ttft_s"], "predicted TTFT"
            )
            predicted_processing_ms = predicted_ttft_ms - predicted_wait_ms
            if predicted_processing_ms < -1e-6:
                raise ValueError(
                    "predicted first-schedule wait exceeds predicted TTFT"
                )
            measured_wait_ms = _finite_number(
                observed_record["wait_ms"], "measured first-schedule wait"
            )
            measured_processing_ms = _finite_number(
                observed_record["processing_ms"], "measured TTFT processing"
            )
            records.append({
                "request_idx": request_idx,
                "measured_wait_ms": measured_wait_ms,
                "measured_processing_ms": measured_processing_ms,
                "predicted_wait_ms": predicted_wait_ms,
                "predicted_processing_ms": max(0.0, predicted_processing_ms),
            })
        result["ttft_breakdown"] = {
            "schema_version": 1,
            "boundary": "request_arrival_to_first_schedule_to_first_token",
            "processing_scope": (
                "first_schedule_to_client_visible_first_token"
            ),
            "measured_arrival_basis": observed_breakdown_manifest[
                "arrival_basis"
            ],
            "measured_negative_wait_clamp_count": _integer(
                observed_breakdown_manifest["negative_wait_clamp_count"],
                "measured negative wait clamp count",
            ),
            "measured_minimum_raw_wait_ms": _finite_number(
                observed_breakdown_manifest["minimum_raw_wait_ms"],
                "measured minimum raw wait",
            ),
            "request_count": request_count,
            "records": records,
        }
    return result


def _unavailable_prediction_metrics(
    observed: Mapping[str, object],
) -> Mapping[str, object]:
    observed_latency = _mapping(observed["latencies"], "observed latencies")
    latency = {}
    for metric_name in _LATENCY_FIELDS:
        distribution = _mapping(observed_latency[metric_name], metric_name)
        latency[metric_name] = {
            statistic: _comparison(distribution[statistic], None)
            for statistic in STATISTICS
        }
    observed_throughput = _mapping(
        observed["throughputs"], "observed throughputs"
    )
    throughput = {
        metric_name: _comparison(observed_throughput[metric_name], None)
        for metric_name in _THROUGHPUT_FIELDS
    }
    paired = {
        metric_name: {
            "request_count": len(_mapping(
                observed["by_request"], "observed requests by index"
            )),
            "absolute_percentage_error_eligible_count": None,
            "mae_ms": None,
            "mape_percent": None,
            "absolute_error_ms": {"p50": None, "p90": None, "p99": None},
            "absolute_percentage_error": {
                "p50": None,
                "p90": None,
                "p99": None,
            },
        }
        for metric_name in _LATENCY_FIELDS
    }
    observed_by_request = _mapping(
        observed["by_request"], "observed requests by index"
    )
    return {
        "latency": latency,
        "latency_quantiles_p0_p100": {
            "comparison_kind": "measured_vs_predicted",
            "measured": _canonical_latency_quantile_projection(
                observed_by_request,
                len(observed_by_request),
                predicted=False,
            ),
            "predicted": None,
        },
        "paired_request_error": paired,
        "paired_request_latency": None,
        "ttft_breakdown": None,
        "throughput": throughput,
    }


def _canonical_latency_quantile_projection(
    records_value: object,
    request_count: int,
    *,
    predicted: bool,
) -> Mapping[str, object]:
    records = _mapping(records_value, "request latency records")
    if len(records) != request_count:
        raise ValueError("request latency records do not cover every request")
    request_metrics = {}
    for request_idx in range(request_count):
        request_id = f"request-{request_idx:012d}"
        key = request_id if predicted else request_idx
        record = _mapping(records[key], "request latency record")
        if predicted:
            ttft_s = _finite_number(record["ttft_s"], "predicted ttft_s")
            tpot_s = _finite_number(record["tpot_s"], "predicted tpot_s")
            latency_s = _finite_number(record["latency_s"], "predicted latency_s")
        else:
            ttft_s = _finite_number(record["ttft_ms"], "measured ttft_ms") / 1000.0
            tpot_s = _finite_number(record["tpot_ms"], "measured tpot_ms") / 1000.0
            latency_s = (
                _finite_number(
                    record["e2e_latency_ms"], "measured e2e_latency_ms"
                )
                / 1000.0
            )
        request_metrics[request_id] = RequestMetrics(
            request_id=request_id,
            ttft_s=ttft_s,
            tpot_s=tpot_s,
            latency_s=latency_s,
        )
    manifest = latency_distribution_manifest(request_metrics)
    return {
        "schema_version": 1,
        "source_schema_version": manifest["schema_version"],
        "quantile_representation": manifest["quantile_representation"],
        "percentile_method": manifest["percentile_method"],
        **{
            field: {
                "availability": manifest[field]["availability"],
                "eligible_request_count": manifest[field][
                    "eligible_request_count"
                ],
                "excluded_request_count": manifest[field]["excluded_request_count"],
                "exclusion_reason": manifest[field]["exclusion_reason"],
                "quantiles": manifest[field]["quantiles"],
            }
            for field in ("ttft_s", "tpot_s", "latency_s")
        },
    }


def _paired_error_summary(
    absolute_errors: Sequence[float],
    percentage_errors: Sequence[float],
    request_count: int,
) -> Mapping[str, object]:
    if len(absolute_errors) != request_count:
        raise ValueError("paired latency errors do not cover every request")
    absolute_distribution = _distribution(absolute_errors)
    if percentage_errors:
        percentage_distribution = _distribution(percentage_errors)
        mape = percentage_distribution["mean"]
        percentage_quantiles = {
            statistic: percentage_distribution[statistic]
            for statistic in ("p50", "p90", "p99")
        }
    else:
        mape = None
        percentage_quantiles = {"p50": None, "p90": None, "p99": None}
    return {
        "request_count": request_count,
        "absolute_percentage_error_eligible_count": len(percentage_errors),
        "mae_ms": absolute_distribution["mean"],
        "mape_percent": mape,
        "absolute_error_ms": {
            statistic: absolute_distribution[statistic]
            for statistic in ("p50", "p90", "p99")
        },
        "absolute_percentage_error": percentage_quantiles,
    }


def _distribution(values: Sequence[float]) -> Mapping[str, float]:
    if not values:
        raise ValueError("metric distribution cannot be empty")
    sorted_values = sorted(values)
    return {
        "mean": statistics.fmean(values),
        "p50": linear_percentile(sorted_values, 50.0),
        "p90": linear_percentile(sorted_values, 90.0),
        "p99": linear_percentile(sorted_values, 99.0),
    }


def _comparison(observed: object, predicted: object | None) -> Mapping[str, object]:
    observed_value = _finite_number(observed, "observed metric")
    if predicted is None:
        return {
            "observed": observed_value,
            "predicted": None,
            "error": None,
            "absolute_percentage_error": None,
        }
    predicted_value = _finite_number(predicted, "predicted metric")
    error = predicted_value - observed_value
    ape = None if observed_value == 0 else abs(error) / abs(observed_value) * 100.0
    return {
        "observed": observed_value,
        "predicted": predicted_value,
        "error": error,
        "absolute_percentage_error": ape,
    }


# Each hardware declares which error messages its frozen phase-1 contract
# accepts as a *known* launch gap.  The three functions below differ only in
# which error class carries the message, so one engine reads this table and
# each caller keeps its own error class and unknown-hardware message.
#
def _is_known_metax_vector_gap(message: str) -> bool:
    """Return whether ``message`` is the MetaX Vector unsupported-for-'<op>' gap."""

    if not (message.startswith(_KNOWN_METAX_VECTOR_PREFIX)
            and message.endswith(_KNOWN_METAX_VECTOR_SUFFIX)):
        return False
    operation = message[
        len(_KNOWN_METAX_VECTOR_PREFIX):-len(_KNOWN_METAX_VECTOR_SUFFIX)
    ]
    return (len(operation) >= 3 and operation[0] == "'"
            and operation[-1] == "'")


# The selector-gap message each hardware's launch selector is allowed to raise
# on a case outside its characterized corpus.  A hardware absent from this map
# has no permitted gap, so its KeyError stays a failure.  The pattern must
# match the selector's own wording exactly; it is evidence consumed at the
# evaluation boundary, never a fallback for a missing selection.
# ``matchers`` are evaluated in order; the first one that accepts the message
# returns it.  A matcher is the message itself (exact equality), a compiled
# pattern (full match), or a callable returning whether the message is the
# declared gap.
_LAUNCH_GAP_CONTRACT = {
    "nvidia_a800": {
        "key_error": _KNOWN_A800_GEMM_LAUNCH_GAP,
        "not_implemented": {
            "matchers": (_KNOWN_A800_ADD_RMS_QUEUED_SOURCE_GAP,),
        },
        "value_error": {"matchers": (_KNOWN_A800_FA2_COMBINE_GAP,)},
    },
    "nvidia_h200": {
        "key_error": None,
        "not_implemented": None,
        "value_error": {"matchers": ()},
    },
    "metax_c500": {
        "key_error": _KNOWN_METAX_GEMM_LAUNCH_GAP,
        "not_implemented": {
            "matchers": (
                _KNOWN_METAX_FLASH_ATTN_GAP,
                _is_known_metax_vector_gap,
                _KNOWN_EXPERT_PARALLEL_LAUNCH_GAPS["metax_c500"],
            ),
        },
        "value_error": {"matchers": ()},
    },
    "ascend_910b": {
        "key_error": _KNOWN_ASCEND_LAUNCH_GAP,
        "not_implemented": {
            "matchers": (
                _KNOWN_EXPERT_PARALLEL_LAUNCH_GAPS["ascend_910b"],
            ),
        },
        "value_error": {"matchers": ()},
    },
}


def _known_launch_gap_reason(
    error: KeyError,
    hardware_id: str,
) -> str | None:
    if len(error.args) != 1 or not isinstance(error.args[0], str):
        return None
    message = error.args[0]
    if hardware_id not in _PHASE1_HARDWARE_IDS:
        raise ValueError(f"unknown phase-1 hardware: {hardware_id}")
    pattern = _LAUNCH_GAP_CONTRACT[hardware_id]["key_error"]
    if pattern is None:
        return None
    if pattern.fullmatch(message) is not None:
        return message
    return None


def _known_not_implemented_launch_gap_reason(
    error: NotImplementedError,
    hardware_id: str,
) -> str | None:
    if len(error.args) != 1 or not isinstance(error.args[0], str):
        return None
    return _declared_launch_gap(
        error.args[0], hardware_id, "not_implemented",
        unknown_message=f"unknown compact-suite hardware: {hardware_id}")


def _known_value_error_launch_gap_reason(
    error: ValueError,
    hardware_id: str,
) -> str | None:
    if len(error.args) != 1 or not isinstance(error.args[0], str):
        return None
    return _declared_launch_gap(
        error.args[0], hardware_id, "value_error",
        unknown_message=f"unknown compact-suite hardware: {hardware_id}")


def _declared_launch_gap(
    message: str, hardware_id: str, scope: str, *, unknown_message: str,
) -> str | None:
    """Return ``message`` when the declared contract names it as a gap.

    ``None`` in the row means the hardware has no gaps of this error class:
    the message is not a declared gap.  A hardware id outside the table is a
    contract violation, not an absent gap.
    """

    if hardware_id not in _LAUNCH_GAP_CONTRACT:
        raise ValueError(unknown_message)
    entry = _LAUNCH_GAP_CONTRACT[hardware_id][scope]
    if entry is None:
        return None
    for matcher in entry["matchers"]:
        if callable(matcher):
            if matcher(message):
                return message
        elif hasattr(matcher, "fullmatch"):
            if matcher.fullmatch(message) is not None:
                return message
        elif message == matcher:
            return message
    return None


def _process_peak_rss_bytes() -> int:
    return max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    ) * 1024




def _validate_execution_mode(execution_mode: object) -> str:
    if execution_mode not in ("accelerated", "reference"):
        raise ValueError(
            "execution_mode must be 'accelerated' or 'reference'")
    return str(execution_mode)


def _validate_execution_cache_stats(
    value: object,
    execution_mode: str,
) -> Mapping[str, object]:
    stats = _mapping(value, "execution cache stats")
    cache_names = (
        "block_construction",
        "expert_routing",
        "kernel_lowering",
        "kernel_prediction",
    )
    _exact_fields(
        stats,
        {"execution_mode", *cache_names},
        "execution cache stats",
    )
    if stats["execution_mode"] != execution_mode:
        raise ValueError("execution cache stats mode mismatch")
    normalized: dict[str, object] = {"execution_mode": execution_mode}
    for cache_name in cache_names:
        cache = stats[cache_name]
        if execution_mode == "reference":
            if cache is not None:
                raise ValueError(
                    f"reference execution cache {cache_name} must be null")
            normalized[cache_name] = None
            continue
        cache_stats = _mapping(cache, f"{cache_name} cache stats")
        _exact_fields(
            cache_stats,
            {"entries", "hits", "misses"},
            f"{cache_name} cache stats",
        )
        normalized_cache = {}
        for field in ("entries", "hits", "misses"):
            normalized_cache[field] = _integer(
                cache_stats[field], f"{cache_name} cache {field}")
            if normalized_cache[field] < 0:
                raise ValueError(
                    f"{cache_name} cache {field} must be nonnegative")
        if normalized_cache["entries"] != normalized_cache["misses"]:
            raise ValueError(
                f"{cache_name} cache entries must equal misses")
        normalized[cache_name] = normalized_cache
    return normalized


def _write_csv(path: Path, cells: Sequence[Mapping[str, object]]) -> None:
    fieldnames = (
        "cell_id",
        "hardware_id",
        "model_id",
        "case_id",
        "target_qps",
        "execution_mode",
        "evaluation_wall_time_seconds",
        "prediction_status",
        "performance_timing_status",
        "accuracy_claim_status",
        "capture_contract_status",
        "metric",
        "statistic",
        "observed",
        "predicted",
        "error",
        "absolute_percentage_error",
        "value",
        "eligible_count",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            metrics = _mapping(cell["metrics"], "cell metrics")
            latency = _mapping(metrics["latency"], "cell latency")
            for metric_name in _LATENCY_FIELDS:
                distribution = _mapping(latency[metric_name], metric_name)
                for statistic in STATISTICS:
                    writer.writerow(_csv_row(cell, metric_name, statistic, distribution[statistic]))
            throughput = _mapping(metrics["throughput"], "cell throughput")
            for metric_name in _THROUGHPUT_FIELDS:
                writer.writerow(_csv_row(cell, metric_name, "scalar", throughput[metric_name]))
            paired = _mapping(
                metrics["paired_request_error"], "paired request error"
            )
            for metric_name in _LATENCY_FIELDS:
                summary = _mapping(paired[metric_name], metric_name)
                eligible_count = summary[
                    "absolute_percentage_error_eligible_count"
                ]
                writer.writerow(_paired_csv_row(
                    cell, metric_name, "mae_ms", summary["mae_ms"], None
                ))
                writer.writerow(_paired_csv_row(
                    cell,
                    metric_name,
                    "mape_percent",
                    summary["mape_percent"],
                    eligible_count,
                ))
                writer.writerow(_paired_csv_row(
                    cell,
                    metric_name,
                    "absolute_percentage_error_eligible_count",
                    eligible_count,
                    eligible_count,
                ))
                for statistic in ("p50", "p90", "p99"):
                    writer.writerow(_paired_csv_row(
                        cell,
                        metric_name,
                        f"absolute_error_ms_{statistic}",
                        _mapping(
                            summary["absolute_error_ms"],
                            "paired absolute errors",
                        )[statistic],
                        None,
                    ))
                    writer.writerow(_paired_csv_row(
                        cell,
                        metric_name,
                        f"absolute_percentage_error_{statistic}",
                        _mapping(
                            summary["absolute_percentage_error"],
                            "paired percentage errors",
                        )[statistic],
                        eligible_count,
                    ))


def _csv_row(
    cell: Mapping[str, object],
    metric: str,
    statistic: str,
    comparison_value: object,
) -> Mapping[str, object]:
    comparison = _mapping(comparison_value, "metric comparison")
    return {
        "cell_id": cell["cell_id"],
        "hardware_id": cell["hardware_id"],
        "model_id": cell["model_id"],
        "case_id": cell["case_id"],
        "target_qps": cell["target_qps"],
        "execution_mode": cell["execution_mode"],
        "evaluation_wall_time_seconds": cell[
            "evaluation_wall_time_seconds"],
        "prediction_status": cell["prediction_status"],
        "performance_timing_status": cell["performance_timing_status"],
        "accuracy_claim_status": cell["accuracy_claim_status"],
        "capture_contract_status": cell["capture_contract_status"],
        "metric": metric,
        "statistic": statistic,
        "observed": comparison["observed"],
        "predicted": comparison["predicted"],
        "error": comparison["error"],
        "absolute_percentage_error": comparison["absolute_percentage_error"],
        "value": None,
        "eligible_count": None,
    }


def _paired_csv_row(
    cell: Mapping[str, object],
    metric: str,
    statistic: str,
    value: object,
    eligible_count: object,
) -> Mapping[str, object]:
    return {
        "cell_id": cell["cell_id"],
        "hardware_id": cell["hardware_id"],
        "model_id": cell["model_id"],
        "case_id": cell["case_id"],
        "target_qps": cell["target_qps"],
        "execution_mode": cell["execution_mode"],
        "evaluation_wall_time_seconds": cell[
            "evaluation_wall_time_seconds"],
        "prediction_status": cell["prediction_status"],
        "performance_timing_status": cell["performance_timing_status"],
        "accuracy_claim_status": cell["accuracy_claim_status"],
        "capture_contract_status": cell["capture_contract_status"],
        "metric": metric,
        "statistic": statistic,
        "observed": None,
        "predicted": None,
        "error": None,
        "absolute_percentage_error": None,
        "value": value,
        "eligible_count": eligible_count,
    }


def _render_markdown(result: Mapping[str, object]) -> str:
    if result["formal_conclusion_allowed"] is True:
        conclusion = "FORMAL: every required compact alignment sample matched."
    elif result["alignment_stage_ran"] is False:
        conclusion = (
            "NON-FORMAL DEBUG OUTPUT: the mandatory compact alignment stage "
            "did not run."
        )
    else:
        conclusion = (
            "NON-FORMAL OUTPUT: at least one required compact alignment "
            "sample was mismatched or unsupported."
        )
    lines = [
        "# ShareGPT compact-suite evaluation",
        "",
        f"**{conclusion}**",
        "",
        "This report compares historical compact observations with the current pinned vLLM prediction contract; it is not capture-exact replay.",
        "",
        f"Execution mode: `{result['execution_mode']}`. Evaluation wall time: {float(result['evaluation_wall_time_seconds']):.6f} s.",
        "",
        "| Cell | Prediction | Timing status | Accuracy claim | Capture contract | TTFT MAPE | TPOT MAPE | E2E MAPE |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for cell_value in _sequence(result["cells"], "result cells"):
        cell = _mapping(cell_value, "result cell")
        metrics = _mapping(cell["metrics"], "cell metrics")
        paired = _mapping(metrics["paired_request_error"], "paired errors")
        mape_values = []
        for metric_name in _LATENCY_FIELDS:
            summary = _mapping(paired[metric_name], metric_name)
            mape = summary["mape_percent"]
            if cell["prediction_status"] == "completed" and mape is not None:
                mape_values.append(f"{_finite_number(mape, metric_name):.2f}%")
            else:
                mape_values.append("--")
        lines.append(
            f"| {cell['cell_id']} | {cell['prediction_status']} | "
            f"{cell['performance_timing_status']} | "
            f"{cell['accuracy_claim_status']} | "
            f"{cell['capture_contract_status']} | "
            f"{' | '.join(mape_values)} |"
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _unique_json_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _validate_prediction_contract(prediction: Mapping[str, object]) -> None:
    fixed_fields = {
        field: expected
        for field, expected in _EXPECTED_PREDICTION.items()
        if field not in {
            "launch_policy",
            "tensor_collective_backend",
            "tensor_all_reduce_profile_sha256",
            "tensor_all_gather_profile_sha256",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "expert_parallel_size",
            "enable_prefix_caching",
            "compute_dtype",
        }
    }
    for field, expected in fixed_fields.items():
        actual = prediction[field]
        if field == "router_dispatch_service_time_s":
            if isinstance(actual, bool) or not isinstance(actual, (int, float)):
                raise TypeError(
                    "prediction router_dispatch_service_time_s must be numeric"
                )
            if not math.isfinite(float(actual)) or float(actual) != 0.0:
                raise ValueError(
                    "prediction router_dispatch_service_time_s must equal zero"
                )
        elif type(actual) is not type(expected) or actual != expected:
            raise ValueError(
                f"prediction contract has unsupported {field}: {actual!r}"
            )
    if prediction["compute_dtype"] not in ("bf16", "fp16"):
        raise ValueError("prediction contract has unsupported compute_dtype")
    if prediction["launch_policy"] not in (
        "injected_launch", "analytical_launch"
    ):
        raise ValueError("prediction contract has unknown launch_policy")
    collective_backend = prediction["tensor_collective_backend"]
    profile_values = (
        prediction["tensor_all_reduce_profile_sha256"],
        prediction["tensor_all_gather_profile_sha256"],
    )
    if collective_backend == "analytical_roofline_v1":
        if any(value is not None for value in profile_values):
            raise ValueError(
                "analytical collective prediction requires null profiles")
    elif collective_backend == "protobridge":
        if prediction["tensor_parallel_size"] == 1:
            if profile_values != (None, None):
                raise ValueError(
                    "TP=1 ProtoBridge prediction must not declare unused "
                    "collective profiles")
        else:
            for value in profile_values:
                _sha256(value, "ProtoBridge profile SHA256")
    else:
        raise ValueError(
            f"prediction contract has unknown tensor collective backend: "
            f"{collective_backend!r}")
    integer_fields = (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "expert_parallel_size",
    )
    for field in integer_fields:
        value = prediction[field]
        if type(value) is not int:
            raise ValueError(f"prediction {field} must be an exact integer")
        if value < 1:
            raise ValueError(f"prediction {field} must be positive")
    if prediction["pipeline_parallel_size"] != 1:
        raise ValueError("compact evaluator currently supports PP=1 only")
    if prediction["expert_parallel_size"] not in (
        1, prediction["tensor_parallel_size"]
    ):
        raise ValueError("expert_parallel_size must equal one or TP size")
    if type(prediction["enable_prefix_caching"]) is not bool:
        raise TypeError("prediction enable_prefix_caching must be a boolean")


def _validate_case_id_contract(
    case_id: str,
    prediction: Mapping[str, object],
) -> None:
    match = _CASE_ID.fullmatch(case_id)
    if match is None:
        raise ValueError(f"cell case_id is not canonical: {case_id!r}")
    case_tp = int(match.group("tp"))
    case_ep_text = match.group("ep")
    case_ep = 1 if case_ep_text is None else int(case_ep_text)
    expected_tp = prediction["tensor_parallel_size"]
    expected_ep = prediction["expert_parallel_size"]
    if case_tp != expected_tp:
        raise ValueError(
            f"case_id {case_id!r} TP={case_tp} disagrees with "
            f"prediction TP={expected_tp}"
        )
    if case_ep != expected_ep:
        raise ValueError(
            f"case_id {case_id!r} EP={case_ep} disagrees with "
            f"prediction EP={expected_ep}"
        )
    if case_ep_text is None and (expected_tp, expected_ep) not in (
        (1, 1),
        (4, 1),
    ):
        raise ValueError(
            f"case_id {case_id!r} may omit EP only for TP1 or TP4 with EP1"
        )


def _validate_jobs(jobs: int) -> None:
    if type(jobs) is not int:
        raise TypeError("jobs must be an exact integer")
    if jobs < 1 or jobs > _MAX_CASE_JOBS:
        raise ValueError(f"jobs must be between 1 and {_MAX_CASE_JOBS}")


def _canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _model_config_path(
    repository_root: str | Path,
    value: object,
) -> Path:
    path = _confined_path(
        Path(repository_root).resolve(), value, "simulation model config")
    if not path.is_file():
        raise ValueError(f"simulation model config is not a file: {path}")
    return path


def _expand_runtime_memory_policy_path(
    repository_root: str | Path,
    config: Phase1Config,
    cell: Phase1Cell,
    model_id: str,
) -> str:
    template = config.prediction.values["runtime_memory_policy_path"]
    return str(resolve_runtime_memory_policy_path(
        template,
        repository_root=repository_root,
        hardware_id=cell.hardware.hardware_id,
        model_id=model_id,
    ))


def _confined_path(root: Path, value: object, label: str) -> Path:
    relative = _relative_path(value, label)
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"{label} escapes repository root")
    return path


def _relative_path(value: object, label: str) -> str:
    text = _string(value, label)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text != path.as_posix():
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return text


def _exact_fields(value: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields must exactly equal {sorted(expected)}; "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{label} must be an array")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _sha256(value: object, label: str) -> str:
    text = _string(value, label)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return text


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
