#!/usr/bin/env python3
"""Run the production no-GPU Serving simulation and write canonical metrics.

``vllm`` selects the real vLLM v1 Scheduler adapter.  ``conformance`` is a
small deterministic backend used only by repository architecture/full-eval
gates when a framework package is unavailable; it is never selected by an
experiment manifest.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import copy
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
import resource
import sys
import threading
import time
import warnings
from src.kernel.hardware_ids import EVALUATED_HARDWARE_IDS

ROOT = Path(__file__).resolve().parents[3]

_REQUEST_PATH_PROFILE_BY_HARDWARE = {
    hardware_id: (
        ROOT / "configs/host_timing/host_request_path_profiles/"
        f"{hardware_id}.json"
    )
    for hardware_id in EVALUATED_HARDWARE_IDS
}

_RUNTIME_PREPARE_PROFILE_BY_HARDWARE = {
    hardware_id: (
        ROOT / "configs/host_timing/host_runtime_prepare_profiles/"
        f"{profile_name}"
    )
    for hardware_id, profile_name in (
        ("nvidia_a800", "nvidia_a800_eager.json"),
        ("metax_c500", "metax_c500_eager.json"),
        ("ascend_910b", "ascend_910b_eager.json"),
    )
}

# Hardware whose request-path and runtime-prepare profiles exist once per
# captured vLLM version.  They have no default binding: a Host timing run on
# such hardware must name them explicitly (--host-request-path-profile and
# --host-runtime-prepare-profile), otherwise it fails instead of silently
# running without those Host mechanisms.
_VERSIONED_HOST_PROFILE_HARDWARE = frozenset({"nvidia_h200"})

# EngineCore host-round pacing is registered only for hardware with an
# independently measured EngineCore host-round mechanism profile.  Hardware
# absent from this registry is explicitly schema-gated off: its steps carry
# zero EngineCore Host pacing and full-mode predictions stay bit-identical.
_ENGINE_CORE_HOST_ROUND_PROFILE_BY_HARDWARE = {
    "ascend_910b": (
        ROOT / "configs/host_timing/"
        "engine_core_host_round_profiles/ascend_910b.json"
    ),
}

from evaluation.serving.experiments.experiment_plan import (
    git_code_identity,
    load_workload_artifact,
    vllm_bench_arrival_process,
    vllm_bench_arrival_times,
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
from evaluation.serving.metrics.scheduler_progress import (
    scheduler_progress_manifest,
)
from evaluation.serving.metrics.system_resource_analysis import (
    critical_path_manifest,
    system_resource_manifest,
)
from evaluation.serving.simulation.execution_artifact import (
    DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS,
    execution_artifact_manifest,
    performance_timing_status,
    prefix_alignment_manifest,
    StreamingExecutionArtifactAccumulator,
    UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
)
from src.kernel.hardware_spec import HWSpec
from src.kernel.predictors.protobridge.adapter import (
    development_profile_config_sha256,
    resolve_registered_profile,
)
from src.serving.backend.vllm import (
    VLLMBackend,
    VLLMSchedulerConfig,
    mtp_scenario_manifest,
)
from src.serving.backend.vllm_scheduler_process import (
    vllm_scheduler_runtime_sha256,
)
from src.serving.backend.mtp_acceptance_replay import (
    PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT,
    load_mtp_acceptance_replay,
    mtp_acceptance_replay_manifest,
    mtp_acceptance_step_from_backend_step,
    mtp_acceptance_steps_manifest,
    closed_loop_admission_contract_sha256,
    pd_decode_admission_contract_sha256,
)
from src.serving.backend.sglang_scheduler_mirror import (
    SGLANG_SOURCE_VERSIONS,
    SGLangSchedulerMirrorBackend,
    SGLangSchedulerMirrorConfig,
)
from src.serving.backend.mtp_acceptance_assumption import (
    MTPAcceptanceAssumption,
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
    worker_role_operator_group,
    worker_role_phase,
)
from src.serving.deployment.deployment_plan import deployment_plan_manifest
from src.serving.deployment.pd_disagg import build_pd_disagg_plan
from src.serving.kv_cache.kv_capacity import (
    kv_capacity_manifest,
    load_runtime_memory_policy,
    resolve_analytical_kv_capacity_plan,
    resolve_debug_kv_capacity_plan,
)
from src.serving.kv_cache.kv_cache_events import KVCacheAction
from src.serving.host_timing import (
    load_host_launch_profile,
    load_host_runtime_prepare_profile,
    load_host_request_path_profile,
    load_scheduler_host_profile,
    load_engine_core_host_round_profile,
    schedule_phase,
)
from src.serving.metrics.serving_metrics import (
    engine_clock_first_schedule_start_s,
)
from src.serving.runtime_prepare import runtime_prepare_events
from src.serving.kv_cache.kv_cache_tier_manager import KVCacheTierManager
from src.serving.kv_cache.kv_lifecycle_replay import (
    load_kv_lifecycle_replay,
)
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.parallelism.pp import pipeline_layer_ranges_for_model
from src.serving.request_gen.production_trace import (
    ProductionTraceReaderConfig,
    SuccessfulRequestFilter,
    build_production_trace_arrival_rate_sweep,
    build_production_trace_replay,
    load_production_trace,
    production_trace_manifest,
    production_trace_replay_manifest,
)
from src.serving.request_gen import (
    CANONICAL_EQUAL_ARRIVAL_ORDER,
    CANONICAL_REQUEST_COLUMNS,
    CANONICAL_REQUEST_ID_POLICY,
)
from src.serving.request_gen.canonical_requests import (
    parse_canonical_requests_bytes,
)
from src.serving.request_gen.request_stream import (
    RequestSpec,
    RequestStream,
    request_stream_sha256,
)
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduledRequest,
    ScheduleStep,
)
from src.serving.simulation import (
    MultiSimulation,
    Simulation,
    SimulationConfig,
    StreamingSimulationResult,
    validate_pd_af_mtp_scope,
    validate_standalone_af_ep2_scope,
    validate_standalone_af_mtp_scope,
    validate_workload_emission_parallelism,
)
from src.serving.workload.model_config import (
    DEEPSEEK_V3_L32_CONFIG_SHA256,
    is_validated_qwen3_tp2_host_kv_model_scope,
    load_model_config,
)
from src.serving.workload.expert_frequency import (
    load_expert_frequency_profile,
)
from src.serving.workload.graph_padding import GraphExecutionConfig
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    load_expert_routing_replay,
    RuntimeInputs,
)


def _p2p_resource_id(link: str, source_pool: str, destination_pool: str) -> str:
    endpoints = tuple(sorted((source_pool, destination_pool)))
    return f"{link}:{endpoints[0]}--{endpoints[1]}"


def _protobridge_p2p_profile(hardware_id: str, requested_link: str):
    """Resolve the registered point-to-point profile for one hardware."""

    digest = development_profile_config_sha256(
        hardware_id, "send_recv", link=requested_link)
    profile = resolve_registered_profile(digest)
    if requested_link != profile.link:
        raise ValueError(
            "point-to-point link disagrees with the registered ProtoBridge "
            f"profile: requested={requested_link!r}, profile={profile.link!r}")
    return profile, digest


def _tensor_collective_configs(
    pool_name: str,
    tensor_parallel_size: int,
    hardware_id: str,
    backend: str,
    all_reduce_profile_sha256: str | None,
    all_gather_profile_sha256: str | None,
) -> tuple[CollectiveConfig, ...]:
    if not isinstance(pool_name, str) or not pool_name:
        raise ValueError("tensor collective pool name must be non-empty")
    if (isinstance(tensor_parallel_size, bool)
            or not isinstance(tensor_parallel_size, int)):
        raise TypeError("tensor parallel size must be an integer")
    if tensor_parallel_size == 1:
        return ()
    if tensor_parallel_size in (2, 4, 8):
        if tensor_parallel_size == 4 and backend != "analytical_roofline_v1":
            raise NotImplementedError(
                "TP=4 tensor collectives exist only for the analytical "
                "DeepSeek-V3-L32 H200 path")
        if backend == "analytical_roofline_v1":
            link = "nvlink"
            if (all_reduce_profile_sha256 is not None
                    or all_gather_profile_sha256 is not None):
                raise ValueError(
                    "analytical Roofline collectives do not accept "
                    "ProtoBridge profile SHA256 values")
            digests = (None, None)
            algorithms = ("ring", "ring")
            topologies = ("ring", "ring")
        elif backend == "protobridge":
            if tensor_parallel_size != 2:
                raise NotImplementedError(
                    "ProtoBridge tensor collective profiles are validated "
                    "only for TP=2; TP=8 requires analytical_roofline_v1")
            supplied_digests = (
                all_reduce_profile_sha256,
                all_gather_profile_sha256,
            )
            if any(digest is None for digest in supplied_digests):
                raise ValueError(
                    "ProtoBridge collectives require explicit all-reduce and "
                    "all-gather profile SHA256 values")
            expected_digests = tuple(
                development_profile_config_sha256(
                    hardware_id, operation)
                for operation in ("all_reduce", "all_gather")
            )
            if supplied_digests != expected_digests:
                raise ValueError(
                    "ProtoBridge collective profile SHA256 values do not "
                    "match the registered hardware/operation profiles")
            digests = supplied_digests
            profiles = tuple(
                resolve_registered_profile(digest)
                for digest in digests
                if digest is not None
            )
            links = {profile.link for profile in profiles}
            if len(links) != 1:
                raise ValueError(
                    "ProtoBridge tensor profiles disagree on physical link")
            link = next(iter(links))
            algorithms = tuple(profile.algorithm for profile in profiles)
            topologies = tuple(profile.topology for profile in profiles)
        else:
            raise ValueError(
                f"unknown tensor collective backend: {backend!r}")
        resource_id = f"{link}:{pool_name}:tensor-fabric"
        return (
            CollectiveConfig(
                "all_reduce",
                algorithms[0],
                link,
                topologies[0],
                resource_id,
                backend,
                digests[0],
            ),
            CollectiveConfig(
                "all_gather",
                algorithms[1],
                link,
                topologies[1],
                resource_id,
                backend,
                digests[1],
            ),
        )
    raise NotImplementedError(
        "production tensor collective configuration supports only TP=1 or "
        "TP=2, plus the target-model analytical TP=4/TP=8 paths")


def _bind_tensor_collective_profiles(args, hardware_id: str) -> None:
    supplied = (
        args.tensor_all_reduce_profile_sha256,
        args.tensor_all_gather_profile_sha256,
    )
    if args.collective_backend == "analytical_roofline_v1":
        if any(digest is not None for digest in supplied):
            raise ValueError(
                "analytical Roofline collectives do not accept ProtoBridge "
                "profile SHA256 values")
        return
    if args.collective_backend == "protobridge":
        expected = tuple(
            development_profile_config_sha256(hardware_id, operation)
            for operation in ("all_reduce", "all_gather")
        )
        if supplied == (None, None):
            (
                args.tensor_all_reduce_profile_sha256,
                args.tensor_all_gather_profile_sha256,
            ) = expected
        elif supplied != expected:
            raise ValueError(
                "ProtoBridge collective profile SHA256 values do not match "
                "the registered hardware/operation profiles")
        return
    raise ValueError(
        f"unknown collective backend: {args.collective_backend!r}")


def _uses_development_protobridge(args) -> bool:
    sizes = [args.prefill_tp_size]
    if args.decode_tp_size is not None:
        sizes.append(args.decode_tp_size)
    if args.ffn_tp_size is not None:
        sizes.append(args.ffn_tp_size)
    uses_collectives = (
        args.collective_backend == "protobridge"
        and any(size > 1 for size in sizes))
    # Disaggregated KV/activation routes always consume registered
    # development send_recv profiles regardless of the collective backend.
    uses_p2p = (
        (args.collective_backend == "protobridge"
         and args.prefill_pp_size > 1)
        or args.deployment in ("af_disagg", "pd_disagg", "pd_af_disagg"))
    return uses_collectives or uses_p2p


def _pipeline_p2p_configs(
    pool_name: str,
    pipeline_parallel_size: int,
    *,
    hardware_id: str,
    backend: str = "protobridge",
    tensor_parallel_size: int = 1,
    ranks_per_node: int | None = None,
) -> tuple[PipelineP2PConfig, ...]:
    if not isinstance(pool_name, str) or not pool_name:
        raise ValueError("pipeline P2P pool name must be non-empty")
    if (isinstance(pipeline_parallel_size, bool)
            or not isinstance(pipeline_parallel_size, int)):
        raise TypeError("pipeline parallel size must be an integer")
    if pipeline_parallel_size == 1:
        return ()
    if (isinstance(tensor_parallel_size, bool)
            or not isinstance(tensor_parallel_size, int)
            or tensor_parallel_size < 1):
        raise ValueError("pipeline tensor parallel size must be positive")
    if ranks_per_node is not None:
        if (isinstance(ranks_per_node, bool)
                or not isinstance(ranks_per_node, int)
                or ranks_per_node < 1):
            raise ValueError("pipeline ranks per node must be positive")
    if pipeline_parallel_size >= 2:
        configs = []
        for source_stage in range(pipeline_parallel_size - 1):
            crosses_node = (
                ranks_per_node is not None
                and source_stage * tensor_parallel_size // ranks_per_node
                != ((source_stage + 1) * tensor_parallel_size
                    // ranks_per_node)
            )
            if backend == "analytical_roofline_v1":
                digest = None
                link = "infiniband" if crosses_node else "nvlink"
                if crosses_node:
                    source_node = (
                        source_stage * tensor_parallel_size // ranks_per_node)
                    destination_node = (
                        (source_stage + 1) * tensor_parallel_size
                        // ranks_per_node)
                    resource_id = (
                        f"infiniband:{pool_name}:node-{source_node}-to-"
                        f"node-{destination_node}")
                else:
                    node = (
                        0 if ranks_per_node is None
                        else source_stage * tensor_parallel_size
                        // ranks_per_node)
                    resource_id = f"nvlink:{pool_name}:node-{node}"
            elif backend == "protobridge":
                if crosses_node:
                    raise NotImplementedError(
                        "ProtoBridge pipeline P2P has no registered inter-node "
                        "profile")
                # Intra-node pipeline P2P shares the tensor-collective
                # scale-up fabric, never the scale-out record.
                scale_up_link = resolve_registered_profile(
                    development_profile_config_sha256(
                        hardware_id, "all_reduce")).link
                digest = development_profile_config_sha256(
                    hardware_id, "send_recv", link=scale_up_link)
                profile = resolve_registered_profile(digest)
                link = profile.link
                node = (
                    0 if ranks_per_node is None
                    else source_stage * tensor_parallel_size // ranks_per_node)
                resource_id = f"{link}:{pool_name}:node-{node}"
            else:
                raise ValueError(
                    f"unknown pipeline P2P backend: {backend!r}")
            configs.append(PipelineP2PConfig(
                source_stage,
                source_stage + 1,
                "send_recv",
                link,
                "point_to_point",
                resource_id,
                backend,
                digest,
            ))
        return tuple(configs)
    raise NotImplementedError(
        "production pipeline P2P configuration requires positive PP")


class _ConformanceBackend:
    """Deterministic protocol exerciser; not a serving policy model."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._requests = {}
        self._pending = None
        self._step = 0

    def add_request(self, request: RequestSpec) -> None:
        if request.output_tokens == 0:
            raise NotImplementedError(
                "conformance zero-output request completion is not implemented")
        if request.prefix_hit_tokens != 0:
            raise NotImplementedError(
                "conformance backend prefix injection is not implemented")
        has_transferred_kv = self.config.is_decode_only
        self._requests[request.request_id] = {
            "request": request,
            "prompt_computed": request.input_tokens if has_transferred_kv else 0,
            "output_produced": 1 if has_transferred_kv else 0,
        }

    def has_work(self) -> bool:
        return bool(self._requests)

    def close(self) -> None:
        return None

    def plan_step(self, virtual_time_s: float):
        if self._pending is not None:
            raise RuntimeError("complete the conformance step before planning another")
        budget = self.config.max_batched_tokens
        scheduled = []
        completion = {}
        finished = []
        for request_id, state in list(self._requests.items())[:self.config.max_requests]:
            if budget <= 0:
                break
            completion[request_id] = 0
            request = state["request"]
            prompt_computed = state["prompt_computed"]
            if prompt_computed < request.input_tokens:
                count = min(request.input_tokens - prompt_computed, budget)
                phase = "prefill"
                input_tokens = count
                output_tokens = 0
                if prompt_computed + count == request.input_tokens:
                    completion[request_id] = 1
            else:
                count = 1
                phase = "decode"
                input_tokens = 0
                output_tokens = 1
                completion[request_id] = 1
            budget -= count
            scheduled.append(ScheduledRequest(
                request_id=request_id,
                phase=phase,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                computed_tokens=(
                    prompt_computed
                    + max(state["output_produced"] - 1, 0)
                ),
                raw_model_output_tokens=completion[request_id],
            ))
            output_limit = (
                1 if self.config.is_prefill_only
                else request.output_tokens)
            if state["output_produced"] + completion[request_id] >= output_limit:
                finished.append(request_id)
        if not scheduled:
            raise RuntimeError("conformance backend made no progress")
        step = BackendStep(
            ScheduleStep(
                f"{self.config.pool_name}:conformance:{self._step}",
                self.config.pool_name,
                tuple(scheduled),
                tuple(finished),
            ),
            generated_tokens=completion,
        )
        self._pending = step
        self._step += 1
        return step

    def complete_step(self, step: BackendStep, virtual_time_s: float) -> float:
        if step is not self._pending:
            raise ValueError("completion does not match conformance pending step")
        for scheduled in step.schedule.requests:
            state = self._requests[scheduled.request_id]
            state["prompt_computed"] += scheduled.input_tokens
            state["output_produced"] += step.generated_tokens[
                scheduled.request_id]
        for request_id in step.finished_request_ids:
            self._requests.pop(request_id)
        self._pending = None
        return 0.0


def run(args) -> dict:
    if args.trace_replay_mode == "arrival_rate_sweep":
        payload = _run_arrival_rate_sweep(args)
    elif args.trace_replay_mode in (None, "original", "arrival_rate_scale"):
        payload = _run_single(args)
    else:
        raise ValueError(
            f"unknown production Trace replay mode: "
            f"{args.trace_replay_mode!r}")
    timing_status = payload["execution_provenance"][
        "performance_timing_status"]
    if timing_status in (
            UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
            DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS):
        warnings.warn(
            "\n"
            "============================================================\n"
            "UNSUPPORTED / DEVELOPMENT ONLY: Serving timing inputs\n"
            f"status={timing_status}\n"
            "Do not use these timings for publication performance claims, "
            "accuracy validation, hardware comparison, or DSE conclusions.\n"
            "============================================================",
            RuntimeWarning,
            stacklevel=2,
        )
    return payload


def _run_arrival_rate_sweep(args) -> dict:
    if args.production_trace is None:
        raise ValueError("arrival-rate sweep requires --production-trace")
    if args.trace_arrival_rate_scales is None:
        raise ValueError(
            "arrival-rate sweep requires --trace-arrival-rate-scales")
    if args.interactivity_tgs_config is None:
        interactivity_tgs_contract = None
        analysis_points = None
    elif (isinstance(args.interactivity_tgs_config, str)
          and args.interactivity_tgs_config):
        interactivity_tgs_contract = load_interactivity_tgs_contract(
            args.interactivity_tgs_config)
        _require_expected_sha256(
            "Interactivity--TGS contract",
            args.interactivity_tgs_config_sha256,
            interactivity_tgs_contract.source_sha256,
        )
        analysis_points = []
    else:
        raise ValueError(
            "--interactivity-tgs-config must be a non-empty path")
    trace = load_production_trace(
        args.production_trace,
        _production_trace_reader_config(args),
    )
    _require_expected_sha256(
        "production Trace",
        args.trace_source_sha256,
        trace.manifest.source_sha256,
    )
    replays = build_production_trace_arrival_rate_sweep(
        trace,
        args.trace_arrival_rate_scales,
    )
    trace_manifest = production_trace_manifest(trace)
    points = []
    shared_execution_provenance = None
    for replay in replays:
        point_args = copy(args)
        point_args.trace_replay_mode = "arrival_rate_scale"
        point_args.trace_arrival_rate_scale = replay.arrival_rate_scale
        point_args.trace_arrival_rate_scales = None
        point_args.interactivity_tgs_config = None
        point_args.output = None
        result = _run_single(point_args)
        request_source = result["request_source"]
        if request_source["kind"] != "production_trace":
            raise RuntimeError(
                "arrival-rate sweep point did not use a production Trace")
        if request_source["trace"] != trace_manifest:
            raise RuntimeError(
                "arrival-rate sweep point changed source Trace provenance")
        replay_manifest = production_trace_replay_manifest(replay)
        if request_source["replay"] != replay_manifest:
            raise RuntimeError(
                "arrival-rate sweep point disagrees with canonical replay")
        execution_provenance = result["execution_provenance"]
        if shared_execution_provenance is None:
            shared_execution_provenance = execution_provenance
        elif execution_provenance != shared_execution_provenance:
            raise RuntimeError(
                "arrival-rate sweep execution provenance changed between "
                "points")
        aggregate_result = _production_trace_aggregate_result(result)
        points.append({
            "arrival_rate_scale": replay.arrival_rate_scale,
            "request_stream_sha256": replay.request_stream_sha256,
            "validation_status": {
                "trace_replay": "validated_conservation",
                "scheduler_control_plane": "native_pinned_vllm",
                "performance_timing": result[
                    "performance_timing_status"],
            },
            "metrics": {
                "request_throughput_per_s": result["metrics"][
                    "request_throughput_per_s"],
                "output_token_throughput_per_s": result["metrics"][
                    "output_token_throughput_per_s"],
                "token_throughput_per_s": result["metrics"][
                    "token_throughput_per_s"],
                "latency_distributions": result["metrics"][
                    "latency_distributions"],
                "slo_goodput": result["metrics"]["slo_goodput"],
            },
            "result": aggregate_result,
        })
        if analysis_points is not None:
            analysis_points.append(sweep_metric_point_from_manifest(
                replay.arrival_rate_scale,
                result["metrics"],
                interactivity_tgs_contract,
            ))
    if shared_execution_provenance is None:
        raise RuntimeError("arrival-rate sweep produced no execution points")
    analysis = arrival_rate_sweep_analysis_manifest(
        analysis_points,
        interactivity_tgs_contract,
    )
    payload = {
        "schema_version": 3,
        "kind": "production_trace_arrival_rate_sweep",
        "interpretation": "successful_path_performance_only",
        "interactivity_tgs": analysis["interactivity_tgs"],
        "peak_selection": analysis["peak_selection"],
        "saturation_analysis": analysis["saturation_analysis"],
        "scale_order": "caller_supplied",
        "arrival_transform": (
            "t_anchor + (t - t_anchor) / arrival_rate_scale"),
        "point_count": len(points),
        "arrival_rate_scales": tuple(
            replay.arrival_rate_scale for replay in replays),
        "source_trace": trace_manifest,
        "execution_provenance": shared_execution_provenance,
        "persistence": {
            "schema_version": 1,
            "projection": "aggregate_only_production_trace_v1",
            "request_level_content": "omitted",
        },
        "points": points,
    }
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _serving_progress_reporter(
        total_requests: int,
        every_steps: int = 100,
        heartbeat_seconds: float = 30.0,
):
    """Return a stderr reporter and heartbeat for one CLI simulation run."""

    if (isinstance(total_requests, bool)
            or not isinstance(total_requests, int)
            or total_requests <= 0):
        raise ValueError("Serving progress total_requests must be positive")
    if (isinstance(every_steps, bool)
            or not isinstance(every_steps, int)
            or every_steps <= 0):
        raise ValueError("Serving progress interval must be positive")
    if (isinstance(heartbeat_seconds, bool)
            or not isinstance(heartbeat_seconds, (int, float))
            or not math.isfinite(heartbeat_seconds)
            or heartbeat_seconds <= 0):
        raise ValueError("Serving progress heartbeat must be positive")
    completed_steps = 0
    scheduled_request_rows = 0
    scheduled_tokens = 0
    finished_request_ids = set()
    started_at = time.monotonic()
    stop_event = threading.Event()

    def heartbeat() -> None:
        while not stop_event.wait(float(heartbeat_seconds)):
            peak_rss_bytes = max(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
            ) * 1024
            print(
                "[serving-progress-heartbeat] "
                f"wall_elapsed_s={time.monotonic() - started_at:.1f} "
                f"completed_steps={completed_steps} "
                f"finished_requests={len(finished_request_ids)}/"
                f"{total_requests} "
                f"scheduled_request_rows={scheduled_request_rows} "
                f"scheduled_tokens={scheduled_tokens} "
                f"peak_rss_bytes={peak_rss_bytes}",
                file=sys.stderr,
                flush=True,
            )

    threading.Thread(
        target=heartbeat,
        name="serving-progress-heartbeat",
        daemon=True,
    ).start()

    def report(backend_step: BackendStep, virtual_time_s: float) -> None:
        nonlocal completed_steps, scheduled_request_rows, scheduled_tokens
        if not isinstance(backend_step, BackendStep):
            raise TypeError("Serving progress requires BackendStep")
        if (isinstance(virtual_time_s, bool)
                or not isinstance(virtual_time_s, (int, float))
                or not math.isfinite(virtual_time_s)
                or virtual_time_s < 0):
            raise ValueError(
                "Serving progress virtual time must be finite and non-negative")
        schedule = backend_step.schedule
        completed_steps += 1
        scheduled_request_rows += len(schedule.requests)
        scheduled_tokens += schedule.token_count
        finished_request_ids.update(schedule.finished_request_ids)
        if completed_steps % every_steps:
            return
        print(
            "[serving-progress] "
            f"completed_steps={completed_steps} "
            f"finished_requests={len(finished_request_ids)}/{total_requests} "
            f"scheduled_request_rows={scheduled_request_rows} "
            f"scheduled_tokens={scheduled_tokens} "
            f"virtual_time_s={float(virtual_time_s):.9g} "
            f"pool={schedule.pool_name} step_id={schedule.step_id}",
            file=sys.stderr,
            flush=True,
        )

    def stop() -> None:
        stop_event.set()

    report.stop = stop
    return report


def _executing_model_id(model_id: str | None, model_path: str) -> str:
    """Return the canonical model identity used by workload observations."""

    if model_id is not None:
        return model_id
    return Path(model_path).resolve().stem


def _use_streaming_step_retention(args, request_source) -> bool:
    streaming_scope = (
        request_source is not None
        and args.deployment in ("colocated", "pd_disagg")
        and not args.additional_colocated_hardware_config
        and args.kv_lifecycle_replay is None
    )
    if args.step_retention == "auto":
        return streaming_scope
    elif args.step_retention == "full":
        return False
    elif args.step_retention == "streaming_aggregate":
        if not streaming_scope:
            raise NotImplementedError(
                "streaming aggregate retention requires an external request "
                "source, a colocated or non-MTP PD deployment, no additional "
                "colocated hardware, and no KV lifecycle replay")
        return True
    else:
        raise ValueError(
            f"unknown Serving step retention: {args.step_retention!r}")


def _host_timing_manifest(
    result,
    host_launch_profiles,
    host_runtime_prepare_profiles,
) -> dict[str, object]:
    """Return additive and overlap-aware Host/Device timing evidence."""

    entries = result.schedule_trace.entries
    if not entries:
        raise ValueError("Host timing requires a non-empty schedule trace")
    mode = entries[0].timing_mode
    if any(entry.timing_mode != mode for entry in entries):
        raise ValueError("schedule trace mixes Serving timing modes")
    if isinstance(result, StreamingSimulationResult):
        steps_by_id = None
        timings_by_id = {
            timing.step_id: timing for timing in result.step_timings
        }
    else:
        timings_by_id = None
        steps_by_id = {
            step.backend_step.schedule.step_id: step for step in result.steps
        }
        if set(steps_by_id) != {
                entry.backend_step.schedule.step_id for entry in entries}:
            raise RuntimeError(
                "Host timing steps disagree with schedule trace entries")
    rows = []
    for entry in entries:
        schedule = entry.backend_step.schedule
        if entry.execution_wall_s is None:
            raise RuntimeError("simulation trace lacks execution wall time")
        if steps_by_id is None:
            if mode == "device_only":
                predicted_device_s = entry.execution_wall_s
                launch_cpu_s = 0.0
                launch_critical_path_s = 0.0
                overlap_s = 0.0
                eager_dispatch_count = 0
                graph_replay_count = 0
                launch_kinds = ()
            elif mode in (
                    "device_plus_launch",
                    "device_plus_launch_plus_scheduler"):
                timing = timings_by_id[schedule.step_id]
                predicted_device_s = timing.predicted_device_time_s
                launch_cpu_s = timing.launch_cpu_time_s
                launch_critical_path_s = timing.launch_critical_path_s
                overlap_s = timing.host_device_overlap_s
                eager_dispatch_count = timing.eager_dispatch_count
                graph_replay_count = timing.graph_replay_count
                launch_kinds = timing.launch_kinds
            else:
                raise ValueError(
                    f"unsupported Serving timing mode: {mode!r}")
        else:
            timeline = steps_by_id[schedule.step_id].execution_timeline
            predicted_device_s = timeline.predicted_device_time_s
            launch_cpu_s = timeline.launch_cpu_time_s
            launch_critical_path_s = timeline.launch_critical_path_s
            overlap_s = timeline.host_device_overlap_s
            eager_dispatch_count = timeline.eager_dispatch_count
            graph_replay_count = timeline.graph_replay_count
            launch_kinds = timeline.launch_kinds
        rows.append({
            "step_id": schedule.step_id,
            "pool_name": schedule.pool_name,
            "phase": schedule_phase(schedule),
            "scheduled_request_count": len(schedule.requests),
            "prefill_token_count": sum(
                request.input_tokens for request in schedule.requests),
            "decode_token_count": sum(
                request.output_tokens for request in schedule.requests),
            "predicted_device_s": predicted_device_s,
            "launch_cpu_s": launch_cpu_s,
            "launch_critical_path_s": launch_critical_path_s,
            "host_device_overlap_s": overlap_s,
            "scheduler_plan_cpu_s": entry.scheduler_plan_cpu_s,
            "scheduler_update_cpu_s": entry.scheduler_update_cpu_s,
            "execution_wall_s": entry.execution_wall_s,
            "step_wall_s": entry.end_time_s - entry.start_time_s,
            "eager_dispatch_count": eager_dispatch_count,
            "graph_replay_count": graph_replay_count,
            "launch_kinds": launch_kinds,
        })
    aggregate_fields = (
        "predicted_device_s", "launch_cpu_s", "launch_critical_path_s",
        "host_device_overlap_s", "scheduler_plan_cpu_s",
        "scheduler_update_cpu_s", "execution_wall_s", "step_wall_s",
        "eager_dispatch_count", "graph_replay_count",
    )
    aggregate = {
        field: sum(row[field] for row in rows) for field in aggregate_fields
    }
    aggregate["scheduler_cpu_s"] = (
        aggregate["scheduler_plan_cpu_s"]
        + aggregate["scheduler_update_cpu_s"])
    aggregate["schedule_makespan_s"] = result.schedule_trace.end_time_s
    aggregate["step_count"] = len(rows)
    return {
        "schema_version": 1,
        "timing_mode": mode,
        "scheduler_measurement_boundary": (
            "in_scheduler_process_schedule_and_output_update_only"),
        "launch_measurement_boundary": "host_submit_only_no_device_sync",
        "profiles": (
            [] if host_launch_profiles is None else [
                host_launch_profiles[hardware_id].manifest()
                for hardware_id in sorted(host_launch_profiles)
            ]),
        "runtime_prepare_profiles": (
            [] if host_runtime_prepare_profiles is None else [
                host_runtime_prepare_profiles[hardware_id].manifest()
                for hardware_id in sorted(host_runtime_prepare_profiles)
            ]),
        "aggregate": aggregate,
        "steps": rows,
    }


def _run_single(args) -> dict:
    if args.slo_config is None:
        slo_contract = None
    elif isinstance(args.slo_config, str) and args.slo_config:
        slo_contract = load_slo_contract(args.slo_config)
        _require_expected_sha256(
            "SLO contract",
            args.slo_config_sha256,
            slo_contract.source_sha256,
        )
    else:
        raise ValueError("--slo-config must be a non-empty path")
    if args.runtime_memory_policy is None:
        runtime_memory_policy = None
    elif (isinstance(args.runtime_memory_policy, str)
          and args.runtime_memory_policy):
        runtime_memory_policy = load_runtime_memory_policy(
            args.runtime_memory_policy)
    else:
        raise ValueError("--runtime-memory-policy must be a non-empty path")
    router_config = RouterConfig(
        args.router_resource_id,
        args.router_dispatch_service_time_s,
    )
    model = load_model_config(args.model)
    _require_expected_sha256(
        "model config",
        args.model_config_sha256,
        model.config_sha256,
    )
    if args.kv_lifecycle_replay is not None:
        if (model.model_type == "qwen3"
                and model.num_experts == 0
                and model.attention_layout == "standard_gqa"
                and model.compute_dtype == "bf16"
                and model.weight_dtype == "bf16"):
            if args.mtp_acceptance_policy != "disabled":
                raise NotImplementedError(
                    "dense Qwen3 KV lifecycle replay requires MTP disabled")
        elif (model.model_type == "glm4_moe"
              and model.num_experts > 0
              and model.attention_layout == "standard_gqa"
              and model.compute_dtype == "bf16"
              and model.weight_dtype == "bf16"
              and model.mtp_architecture == "glm4_moe_mtp"
              and model.num_mtp_layers == 1):
            if args.mtp_acceptance_policy != "fixed_count":
                raise NotImplementedError(
                    "GLM4-MoE MTP KV lifecycle replay currently requires "
                    "fixed_count acceptance")
            if model.mtp_layer_uses_moe(0):
                raise NotImplementedError(
                    "KV lifecycle replay does not support a MoE MTP layer")
        else:
            raise NotImplementedError(
                "KV lifecycle replay currently requires dense BF16 Qwen3 "
                "or BF16 GLM4-MoE with one dense MTP layer and standard GQA")
    target_model = model.model_type in ("kimi_k2", "glm_moe_dsa", "deepseek_v3")
    if target_model:
        if args.attention_kv_layout != "mla":
            raise ValueError(
                "Kimi-K2.6, GLM-5.2, and DeepSeek-V3 require "
                "--attention-kv-layout mla")
        if model.model_type == "deepseek_v3":
            # DeepSeek-V3 has no source-backed PP production layout in
            # this repository; the H200 TP8/PP1/EP1 predicates below are
            # its only validated target topologies.
            target_parallelism = None
        else:
            target_parallelism = {
                "kimi_k2": (8, 2),
                "glm_moe_dsa": (8, 4),
            }[model.model_type]
        target_pd_parallelism = (
            args.deployment == "pd_disagg"
            and args.prefill_tp_size == 8
            and args.decode_tp_size == 8
            and args.prefill_pp_size == 1
        )
        target_colocated_parallelism = (
            args.deployment == "colocated"
            and target_parallelism is not None
            and (args.prefill_tp_size, args.prefill_pp_size)
            == target_parallelism
        )
        # InferenceX-style aggregate GLM-5.2 serving uses one TP8 pool with
        # EP=1 and PP=1.  It is distinct from the source-backed PP4/EP8
        # production-trace scope below, so retain an explicit predicate.
        target_h200_aggregate_parallelism = (
            model.model_type == "glm_moe_dsa"
            and args.deployment == "colocated"
            and (args.prefill_tp_size, args.prefill_pp_size) == (8, 1)
            and not args.enable_expert_parallel
            and not args.additional_colocated_hardware_config
        )
        # PD TP8/PP1 with EP=1 shares the EP8 KV transfer contract: the
        # compressed MLA KV and FP8 index cache are TP-replicated
        # single-head caches, and EP only shards MoE expert weights.
        target_h200_pd_ep1_parallelism = (
            model.model_type == "glm_moe_dsa"
            and target_pd_parallelism
            and not args.enable_expert_parallel
        )
        # Kimi-K2.6 H200 InferenceX-style colocated serving uses the
        # standard vLLM DeepSeek-V3-family single-node topology: one TP8
        # pool with EP=TP and PP1.  It is distinct from the source-backed
        # PP2/EP8 production scope, so retain an explicit predicate.
        target_kimi_h200_colocated_parallelism = (
            model.model_type == "kimi_k2"
            and args.deployment == "colocated"
            and (args.prefill_tp_size, args.prefill_pp_size) == (8, 1)
            and args.enable_expert_parallel
            and not args.additional_colocated_hardware_config
        )
        # DeepSeek-V3 H200 InferenceX scopes mirror the GLM-5.2 H200
        # aggregate and PD TP8/PP1/EP1 layouts; its PD transfer payload is
        # the compressed MLA cache only (no DSA index cache).
        target_dsv3_h200_aggregate_parallelism = (
            model.model_type == "deepseek_v3"
            and args.deployment == "colocated"
            and ((args.prefill_tp_size, args.prefill_pp_size) == (8, 1)
                 or ((args.prefill_tp_size, args.prefill_pp_size) == (4, 1)
                     and model.config_sha256 == DEEPSEEK_V3_L32_CONFIG_SHA256))
            and not args.enable_expert_parallel
            and not args.additional_colocated_hardware_config
        )
        target_dsv3_h200_pd_ep1_parallelism = (
            model.model_type == "deepseek_v3"
            and target_pd_parallelism
            and not args.enable_expert_parallel
        )
        if ((not target_pd_parallelism and not target_colocated_parallelism
             and not target_h200_aggregate_parallelism
             and not target_kimi_h200_colocated_parallelism
             and not target_dsv3_h200_aggregate_parallelism
             and not target_dsv3_h200_pd_ep1_parallelism)
                or (not target_h200_aggregate_parallelism
                    and not target_h200_pd_ep1_parallelism
                    and not target_dsv3_h200_aggregate_parallelism
                    and not target_dsv3_h200_pd_ep1_parallelism
                    and not args.enable_expert_parallel)
                or args.additional_colocated_hardware_config):
            raise NotImplementedError(
                "target production traces require the named target scope: "
                "one colocated TP=EP=8 Scheduler pool (Kimi-K2.6 PP2 or "
                "GLM-5.2 PP4), the PD TP8/PP1 deployment, GLM-5.2 "
                "H200 aggregate TP8/PP1/EP1, GLM-5.2 H200 PD "
                "TP8/PP1/EP1, Kimi-K2.6 H200 colocated TP8/PP1/EP8, "
                "DeepSeek-V3 H200 aggregate TP8/PP1/EP1, or DeepSeek-V3 "
                "H200 PD TP8/PP1/EP1")
        if (args.deployment == "colocated"
                and args.pipeline_ranks_per_node != 8):
            raise NotImplementedError(
                "target production traces require the explicit A800 "
                "placement --pipeline-ranks-per-node 8")
        if args.kv_lifecycle_replay is not None:
            raise NotImplementedError(
                "Kimi-K2.6 and GLM-5.2 production traces do not yet support "
                "KV offload replay")
        if (model.model_type == "kimi_k2"
                and args.mtp_acceptance_policy != "disabled"):
            raise NotImplementedError(
                "Kimi-K2.6 fixed text config declares zero MTP layers")
    else:
        if (args.backend == "vllm"
                and args.attention_kv_layout != "paged_varlen"):
            raise ValueError(
                "the standard-GQA vLLM path requires "
                "--attention-kv-layout paged_varlen")
        if model.attention_layout != "standard_gqa":
            raise NotImplementedError(
                "the production runner supports non-standard attention only "
                "for the fixed Kimi-K2.6 and GLM-5.2 contracts")
        if model.weight_dtype != model.compute_dtype:
            raise NotImplementedError(
                "the production runner supports quantized weights only for "
                "the fixed Kimi-K2.6 contract")
    prefill_hardware = HWSpec(args.hardware_config)
    # Resolve provenance dependencies before any scheduler work. H200 uses
    # the source-backed A800 launch package, not an H200 observation dataset.
    provenance_hardware_ids = {prefill_hardware.hardware_id}
    if args.decode_hardware_config is not None:
        provenance_hardware_ids.add(
            HWSpec(args.decode_hardware_config).hardware_id)
    for config_path in args.additional_colocated_hardware_config:
        provenance_hardware_ids.add(HWSpec(config_path).hardware_id)
    _launch_data_provenance(tuple(sorted(provenance_hardware_ids)))
    _require_expected_sha256(
        "hardware config",
        args.hardware_config_sha256,
        _file_provenance(args.hardware_config)["sha256"],
    )
    _bind_tensor_collective_profiles(
        args,
        prefill_hardware.hardware_id,
    )
    host_launch_profiles = None
    host_runtime_prepare_profiles = None
    host_request_path_profiles = None
    scheduler_host_profiles = None
    runtime_prepare_event_builder = None
    if args.serving_timing_mode != "device_only":
        profiles = tuple(
            load_host_launch_profile(path)
            for path in args.host_launch_profile
        )
        host_launch_profiles = {
            profile.hardware_id: profile for profile in profiles
        }
        if len(host_launch_profiles) != len(profiles):
            raise ValueError("duplicate Host launch profile hardware_id")
        if args.serving_timing_mode == "device_plus_launch_plus_scheduler":
            scheduler_profiles = tuple(
                load_scheduler_host_profile(path)
                for path in args.scheduler_host_profile
            )
            scheduler_host_profiles = {
                profile.hardware_id: profile for profile in scheduler_profiles
            }
            if len(scheduler_host_profiles) != len(scheduler_profiles):
                raise ValueError("duplicate Scheduler Host profile hardware_id")
            # The per-request Host request path is Host scheduling-path time,
            # so it joins the Scheduler Host CPU mode rather than the launch
            # mode, which deliberately excludes Host scheduling cost.
            request_path_profile_by_hardware = _host_request_path_profile_paths(
                args.host_request_path_profile)
            if (prefill_hardware.hardware_id in _VERSIONED_HOST_PROFILE_HARDWARE
                    and prefill_hardware.hardware_id
                    not in request_path_profile_by_hardware):
                raise ValueError(
                    f"{prefill_hardware.hardware_id} Host scheduling timing "
                    "requires an explicit --host-request-path-profile")
            if prefill_hardware.hardware_id in request_path_profile_by_hardware:
                request_path_profile = load_host_request_path_profile(
                    request_path_profile_by_hardware[
                        prefill_hardware.hardware_id])
                if (request_path_profile.hardware_id
                        != prefill_hardware.hardware_id):
                    raise ValueError(
                        "Host request path profile hardware disagrees with model")
                host_request_path_profiles = {
                    request_path_profile.hardware_id: request_path_profile,
                }
        runtime_prepare_profile_path = _host_runtime_prepare_profile_path(
            prefill_hardware.hardware_id, args.host_runtime_prepare_profile)
        if runtime_prepare_profile_path is not None:
            runtime_prepare_profile = load_host_runtime_prepare_profile(
                runtime_prepare_profile_path)
            if runtime_prepare_profile.hardware_id != prefill_hardware.hardware_id:
                raise ValueError("runtime prepare profile hardware disagrees with model")
            host_runtime_prepare_profiles = {
                runtime_prepare_profile.hardware_id: runtime_prepare_profile,
            }
            enabled_mechanisms = tuple(
                mechanism for mechanism, _ in
                runtime_prepare_profile.service_time_s_by_mechanism)
            runtime_prepare_event_builder = partial(
                runtime_prepare_events,
                enabled_mechanisms=enabled_mechanisms,
            )
    target_h200_scope = (
        model.model_type in ("kimi_k2", "glm_moe_dsa")
        and prefill_hardware.hardware_id == "nvidia_h200"
        and args.deployment in ("colocated", "pd_disagg"))
    if (args.prefill_pp_size > 1
            and prefill_hardware.hardware_id != "nvidia_a800"
            and not target_h200_scope):
        raise NotImplementedError(
            "pipeline parallel Serving path requires nvidia_a800 or the "
            "explicit H200 target deployment scope")
    if (args.prefill_tp_size, args.prefill_pp_size) == (2, 2) and (
            model.model_type != "qwen3"
            or model.num_experts != 0
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16"):
        raise NotImplementedError(
            "the validated TP=2, PP=2 Serving slice requires dense BF16 "
            "Qwen3 with standard GQA")
    if (not target_model
            and args.prefill_tp_size == 1
            and args.prefill_pp_size > 1) and (
            args.prefill_tp_size != 1
            or model.model_type != "qwen3"
            or model.num_experts != 0
            or model.tie_word_embeddings
            or model.attention_layout != "standard_gqa"
            or model.compute_dtype != "bf16"
            or model.weight_dtype != "bf16"
            or model.num_hidden_layers % args.prefill_pp_size):
        raise NotImplementedError(
            "the validated uniform TP=1 PP Serving path requires untied "
            "dense BF16 Qwen3 with standard GQA and decoder layers "
            "divisible by PP")
    (
        requests,
        arrival_process,
        workload_hash,
        request_source,
        result_version,
    ) = _load_request_source(args)
    use_streaming_aggregate = _use_streaming_step_retention(
        args, request_source)
    streaming_artifact = (
        StreamingExecutionArtifactAccumulator()
        if use_streaming_aggregate else None
    )
    progress_callback = _serving_progress_reporter(len(requests.requests))
    actual_request_stream_sha256 = request_stream_sha256(requests)
    if args.mtp_acceptance_policy == "maximum_entropy":
        args.mtp_acceptance_assumption = MTPAcceptanceAssumption(
            args.mtp_acceptance_source_id,
            args.mtp_num_speculative_tokens,
            args.mtp_mean_acceptance_rate,
            args.mtp_acceptance_seed,
            actual_request_stream_sha256,
        )
    elif args.mtp_acceptance_assumption is not None:
        raise ValueError(
            "non-maximum-entropy MTP cannot carry an acceptance assumption")
    if (args.kv_lifecycle_replay is not None
            and args.prefill_tp_size == 2):
        if not is_validated_qwen3_tp2_host_kv_model_scope(model):
            raise NotImplementedError(
                "the validated TP=2 Host-KV lifecycle requires the exact "
                "source-backed dense BF16 Qwen3 fixture")
        if (len(requests.requests) != 1
                or requests.requests[0].arrival_time_s != 0.0
                or requests.requests[0].input_tokens != 4
                or requests.requests[0].output_tokens != 2
                or requests.requests[0].prefix_hit_tokens != 0):
            raise NotImplementedError(
                "the validated TP=2 Host-KV lifecycle requires one 4->2/H0 "
                "request arriving at time zero")
        expected_bytes_per_token = (
            model.kv_bytes_per_token_per_parallel_rank(2, 1))
        replay = args.kv_lifecycle_replay
        hbm_tier = next(
            tier for tier in replay.cache_config.tiers
            if tier.name == "hbm")
        host_tier = next(
            tier for tier in replay.cache_config.tiers
            if tier.name == "host")
        if (expected_bytes_per_token != 512
                or replay.cache_config.bytes_per_token != 512
                or hbm_tier.block_size_tokens != 16
                or hbm_tier.capacity_bytes != 2 * 8192
                or host_tier.block_size_tokens != 16
                or host_tier.capacity_bytes != 8192
                or host_tier.link != "pcie"):
            raise NotImplementedError(
                "the validated TP=2 Host-KV lifecycle requires 512 B/token "
                "per rank, two 8192-B HBM blocks, and one 8192-B Host block")
        replay_actions = tuple(
            tuple(event.action for event in step.events)
            for step in replay.steps
        )
        if replay_actions != (
                (KVCacheAction.EVICT,), (KVCacheAction.LOAD,)):
            raise NotImplementedError(
                "the validated TP=2 Host-KV lifecycle requires exactly two "
                "Scheduler steps with EVICT then LOAD")
    mtp_decode_admission_sha256 = None
    if (args.mtp_acceptance_assumption is not None
            and args.deployment == "pd_disagg"):
        mtp_decode_admission_sha256 = (
            pd_decode_admission_contract_sha256(requests))
    # A closed-loop client population submits successors at client-driven
    # completion instants, so neither admission order nor arrival stamps
    # match the canonical stream; bind the order- and arrival-insensitive
    # admission contract instead.
    mtp_closed_loop_admission_sha256 = None
    if (args.mtp_acceptance_assumption is not None
            and args.closed_loop_client_count is not None):
        mtp_closed_loop_admission_sha256 = (
            closed_loop_admission_contract_sha256(requests))
        mtp_decode_admission_sha256 = None
    if args.mtp_acceptance_replay is not None:
        replay_context = args.mtp_acceptance_replay.scheduler_context
        if (actual_request_stream_sha256
                != replay_context.ingress_identity_sha256):
            raise ValueError(
                "MTP acceptance replay ingress request stream SHA256 "
                "disagrees with the selected request source")
        if args.mtp_acceptance_replay.schema_version == 2:
            if (args.deployment != "pd_disagg"
                    or replay_context.adapter_contract
                    != PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT):
                raise ValueError(
                    "MTP acceptance replay schema v2 requires PD decode")
            mtp_decode_admission_sha256 = (
                pd_decode_admission_contract_sha256(requests))
            if (mtp_decode_admission_sha256
                    != replay_context.decode_admission_contract_sha256):
                raise ValueError(
                    "MTP acceptance replay decode admission SHA256 "
                    "disagrees with the selected request source")
        elif args.mtp_acceptance_replay.schema_version != 1:
            raise AssertionError("unknown MTP replay schema version")
    if args.kv_lifecycle_replay is not None:
        replay_context = args.kv_lifecycle_replay.context
        if actual_request_stream_sha256 != (
                replay_context.request_stream_sha256):
            raise ValueError(
                "KV lifecycle replay request stream SHA256 disagrees with "
                "the selected request source")
        if model.config_sha256 != replay_context.model_config_sha256:
            raise ValueError(
                "KV lifecycle replay model config SHA256 disagrees with "
                "the selected model")
    scheduler_runtime = {}
    kv_capacity_plans = {}
    if model.num_experts == 0:
        if args.expert_routing_policy is not None:
            raise ValueError(
                "dense model simulation cannot accept expert routing input")
        expert_routing = ExpertRoutingInput.not_applicable()
    elif args.expert_routing_policy == "seeded_assumption":
        if args.expert_routing_hot_expert_count is None:
            expert_routing = ExpertRoutingInput.seeded_assumption(
                args.expert_routing_source_id,
                args.expert_routing_seed,
            )
        else:
            expert_routing = ExpertRoutingInput.skewed_assumption(
                args.expert_routing_source_id,
                args.expert_routing_seed,
                hot_expert_count=args.expert_routing_hot_expert_count,
                hot_expert_weight=args.expert_routing_hot_expert_weight,
            )
    elif args.expert_routing_policy == "replay_observation":
        expert_routing = ExpertRoutingInput.replay_observation(
            args.expert_routing_replay,
            actual_request_stream_sha256,
        )
    elif args.expert_routing_policy == "profile_observation":
        executing_model_id = _executing_model_id(model.model_id, args.model)
        profile = load_expert_frequency_profile(
            args.expert_routing_profile_manifest,
            args.expert_routing_source_id,
            expected_model_id=executing_model_id,
            expected_moe_layers=sum(
                model.layer_uses_moe(layer_index)
                for layer_index in range(model.num_hidden_layers)
            ),
            expected_num_experts=model.num_experts,
        )
        expert_routing = ExpertRoutingInput.profile_observation(
            profile,
            per_moe_layer=(
                args.expert_routing_profile_resolution == "per_moe_layer"),
        )
    elif args.expert_routing_policy is None:
        raise ValueError(
            "MoE model simulation requires --expert-routing-policy")
    else:
        raise ValueError(
            f"unknown expert routing policy: "
            f"{args.expert_routing_policy!r}")
    runtime_inputs = RuntimeInputs(
        expert_routing=expert_routing,
        attention_kv_layout=args.attention_kv_layout,
        attention_kv_distribution=args.attention_kv_distribution,
        graph_execution=args.graph_execution_config,
        serving_framework=(
            "sglang" if args.backend == "sglang_mirror" else "vllm"),
    )

    if model.compute_dtype == "bf16":
        scheduler_dtype = "bfloat16"
    elif model.compute_dtype == "fp16":
        scheduler_dtype = "float16"
    else:
        raise NotImplementedError(
            "the pinned vLLM Scheduler path supports only BF16 and FP16 "
            f"model compute dtype, got {model.compute_dtype!r}")

    def make_backend(
            pool_name, role, hardware, hardware_config_path, tp_size,
            pp_size=1):
        if pool_name in kv_capacity_plans:
            raise RuntimeError(
                f"scheduler backend {pool_name!r} was constructed twice")
        parallelism = ParallelConfig(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=pp_size,
            enable_expert_parallel=args.enable_expert_parallel,
        )
        if runtime_memory_policy is None:
            kv_bytes_per_token = (
                model.kv_bytes_per_token_per_parallel_rank(tp_size, pp_size))
            scheduler_gpu_memory_utilization = 0.9
            capacity_plan = resolve_debug_kv_capacity_plan(
                pool_name,
                kv_block_size_tokens=args.kv_block_size_tokens,
                kv_bytes_per_token=kv_bytes_per_token,
                num_kv_blocks=args.num_kv_blocks,
            )
        else:
            policy = runtime_memory_policy.policy_for_pool(pool_name)
            scheduler_gpu_memory_utilization = (
                policy.gpu_memory_utilization_numerator
                / policy.gpu_memory_utilization_denominator)
            capacity_plan = resolve_analytical_kv_capacity_plan(
                pool_name,
                hardware=hardware,
                hardware_config_sha256=(
                    _file_provenance(hardware_config_path)["sha256"]),
                model=model,
                parallelism=parallelism,
                deployment=args.deployment,
                worker_role=role,
                kv_block_size_tokens=args.kv_block_size_tokens,
                max_model_len=args.max_model_len,
                max_batched_tokens=args.max_batched_tokens,
                max_requests=args.max_requests,
                policy=policy,
                include_mtp=(
                    model.model_type == "glm_moe_dsa"
                    and args.mtp_num_speculative_tokens > 0
                ),
            )
            kv_bytes_per_token = capacity_plan.kv_bytes_per_token
        kv_capacity_plans[pool_name] = capacity_plan
        config = BackendConfig(
            pool_name,
            role,
            args.max_batched_tokens,
            args.max_requests,
            capacity_plan.num_kv_blocks,
            kv_block_size_tokens=args.kv_block_size_tokens,
            kv_bytes_per_token=kv_bytes_per_token,
            scheduler_pipeline_depth=args.scheduler_pipeline_depth,
        )
        if args.serving_timing_mode == "device_plus_launch_plus_scheduler":
            if scheduler_host_profiles is None:
                raise AssertionError("Scheduler Host profiles were not loaded")
            if hardware.hardware_id not in scheduler_host_profiles:
                raise ValueError(
                    "Scheduler Host profiles do not cover hardware: "
                    f"{hardware.hardware_id}")
            scheduler_host_profile = scheduler_host_profiles[
                hardware.hardware_id]
            if hardware.hardware_id in _ENGINE_CORE_HOST_ROUND_PROFILE_BY_HARDWARE:
                engine_core_host_round_profile = load_engine_core_host_round_profile(
                    _ENGINE_CORE_HOST_ROUND_PROFILE_BY_HARDWARE[
                        hardware.hardware_id])
                if (engine_core_host_round_profile.hardware_id
                        != hardware.hardware_id):
                    raise ValueError(
                        "EngineCore host-round profile hardware disagrees "
                        "with model")
            else:
                engine_core_host_round_profile = None
        else:
            scheduler_host_profile = None
            engine_core_host_round_profile = None
        if (args.deployment in ("pd_disagg", "pd_af_disagg")
                and config.is_prefill_only):
            mtp_num_speculative_tokens = 0
            mtp_acceptance_policy = "disabled"
            mtp_fixed_accepted_tokens = 0
            mtp_acceptance_source_id = "disabled"
            mtp_acceptance_replay = None
            mtp_acceptance_assumption = None
            mtp_request_stream_sha256 = None
            mtp_decode_admission_contract_sha256 = None
            mtp_closed_loop_admission_contract_sha256 = None
        else:
            mtp_num_speculative_tokens = args.mtp_num_speculative_tokens
            mtp_acceptance_policy = args.mtp_acceptance_policy
            mtp_fixed_accepted_tokens = args.mtp_fixed_accepted_tokens
            mtp_acceptance_source_id = args.mtp_acceptance_source_id
            mtp_acceptance_replay = args.mtp_acceptance_replay
            mtp_acceptance_assumption = args.mtp_acceptance_assumption
            mtp_request_stream_sha256 = (
                actual_request_stream_sha256
                if (args.mtp_acceptance_replay is not None
                    or args.mtp_acceptance_assumption is not None
                    or (args.deployment == "af_disagg"
                        and args.max_requests == 2
                        and args.mtp_num_speculative_tokens != 0))
                else None
            )
            mtp_decode_admission_contract_sha256 = (
                mtp_decode_admission_sha256
                if (args.mtp_acceptance_replay is not None
                    or args.mtp_acceptance_assumption is not None)
                else None
            )
            mtp_closed_loop_admission_contract_sha256 = (
                mtp_closed_loop_admission_sha256
                if args.mtp_acceptance_assumption is not None
                else None
            )
        if args.backend == "vllm":
            return VLLMBackend.from_model(
                config,
                VLLMSchedulerConfig(
                    args.model,
                    tensor_parallel_size=tp_size,
                    pipeline_parallel_size=pp_size,
                    enable_expert_parallel=args.enable_expert_parallel,
                    dtype=scheduler_dtype,
                    max_model_len=args.max_model_len,
                    gpu_memory_utilization=(
                        scheduler_gpu_memory_utilization),
                    enable_chunked_prefill=args.enable_chunked_prefill,
                    enable_prefix_caching=args.enable_prefix_caching,
                    async_scheduling=args.vllm_async_scheduling,
                    mtp_num_speculative_tokens=(
                        mtp_num_speculative_tokens),
                    mtp_acceptance_policy=mtp_acceptance_policy,
                    mtp_fixed_accepted_tokens=(
                        mtp_fixed_accepted_tokens),
                    mtp_acceptance_source_id=(
                        mtp_acceptance_source_id),
                    mtp_acceptance_replay=mtp_acceptance_replay,
                    mtp_acceptance_assumption=(
                        mtp_acceptance_assumption),
                    request_stream_sha256=mtp_request_stream_sha256,
                    decode_admission_contract_sha256=(
                        mtp_decode_admission_contract_sha256),
                    closed_loop_admission_contract_sha256=(
                        mtp_closed_loop_admission_contract_sha256),
                    authorized_h200_pd_mtp_sweep=(
                        _is_glm_h200_pd_mtp_target_scope(args, True)
                        or _is_dsv3_h200_pd_mtp_target_scope(args)),
                    scheduler_host_profile=scheduler_host_profile,
                    engine_core_host_round_profile=(
                        engine_core_host_round_profile),
                ),
            )
        elif args.backend == "conformance":
            return _ConformanceBackend(config)
        elif args.backend == "sglang_mirror":
            return SGLangSchedulerMirrorBackend(
                config,
                SGLangSchedulerMirrorConfig(
                    source_version=args.sglang_source_version,
                    max_total_num_tokens=args.sglang_max_total_num_tokens,
                    max_total_num_tokens_source=(
                        args.sglang_max_total_num_tokens_source),
                    chunked_prefill_size=args.max_batched_tokens,
                    max_prefill_tokens=args.max_batched_tokens,
                    max_running_requests=args.sglang_max_running_requests,
                    schedule_conservativeness=1.0,
                    mtp_num_speculative_tokens=mtp_num_speculative_tokens,
                    mtp_acceptance_assumption=mtp_acceptance_assumption,
                    mtp_acceptance_source_id=mtp_acceptance_source_id,
                ),
            )
        else:
            raise ValueError(f"unknown serving backend: {args.backend!r}")

    def validate_capacity_pools(pool_names):
        if runtime_memory_policy is not None:
            runtime_memory_policy.validate_scheduler_pools(pool_names)

    simulation_wall_started = time.perf_counter()
    if args.deployment == "colocated":
        if args.decode_hardware_config is not None:
            raise ValueError(
                "--decode-hardware-config is only valid for pd_disagg")
        if args.additional_colocated_hardware_config:
            hardware_config_paths = (
                args.hardware_config,
                *tuple(args.additional_colocated_hardware_config),
            )
            instance_hardware = (
                prefill_hardware,
                *tuple(HWSpec(path) for path in hardware_config_paths[1:]),
            )
            config_path_by_hardware_id = {}
            for config_path, hardware in zip(
                    hardware_config_paths, instance_hardware):
                resolved_path = Path(config_path).resolve()
                hardware_id = hardware.hardware_id
                if hardware_id in config_path_by_hardware_id:
                    previous_path = config_path_by_hardware_id[hardware_id]
                    if resolved_path != previous_path:
                        raise ValueError(
                            "colocated instances cannot bind different "
                            "hardware config files with the same hardware_id: "
                            f"{hardware_id!r} maps to {previous_path} and "
                            f"{resolved_path}; reuse one config path for "
                            "same-profile instances")
                else:
                    config_path_by_hardware_id[hardware_id] = resolved_path
            pool_names = tuple(
                f"colocated-{index}"
                for index in range(len(instance_hardware))
            )
            pool_configs = []
            next_rank = 0
            for pool_name, hardware in zip(
                    pool_names, instance_hardware, strict=True):
                ranks = tuple(range(
                    next_rank,
                    next_rank + args.prefill_tp_size,
                ))
                next_rank += args.prefill_tp_size
                pool_configs.append(WorkerPoolConfig(
                    pool_name,
                    "colocated",
                    hardware.hardware_id,
                    ParallelConfig(
                        tensor_parallel_size=args.prefill_tp_size,
                        enable_expert_parallel=args.enable_expert_parallel,
                    ),
                    ranks,
                    collectives=_tensor_collective_configs(
                        pool_name,
                        args.prefill_tp_size,
                        hardware.hardware_id,
                        args.collective_backend,
                        args.tensor_all_reduce_profile_sha256,
                        args.tensor_all_gather_profile_sha256,
                    ),
                ))
            deployment = build_colocated_plan(
                DeploymentConfig(
                    "colocated",
                    tuple(pool_configs),
                    router=router_config,
                ),
                num_layers=model.num_hidden_layers,
                num_experts=model.num_experts,
            )
            hardware_by_id = {
                hardware.hardware_id: hardware
                for hardware in instance_hardware
            }
            validate_capacity_pools(pool_names)
            backends = {}
            with ExitStack() as construction_lifecycle:
                for pool_name, hardware, hardware_config_path in zip(
                        pool_names, instance_hardware, hardware_config_paths,
                        strict=True):
                    backend = make_backend(
                        pool_name, "colocated", hardware,
                        hardware_config_path, args.prefill_tp_size)
                    construction_lifecycle.callback(backend.close)
                    backends[pool_name] = backend
                if args.backend == "vllm":
                    scheduler_runtime.update({
                        pool_name: asdict(backend.runtime_info)
                        for pool_name, backend in backends.items()
                    })
                elif args.backend == "conformance":
                    pass
                elif args.backend == "sglang_mirror":
                    scheduler_runtime.update({
                        pool_name: backend.runtime_info
                        for pool_name, backend in backends.items()
                    })
                else:
                    raise ValueError(
                        f"unknown serving backend: {args.backend!r}")
                result = MultiSimulation(
                    SimulationConfig(
                        model, deployment, requests, args.launch_policy,
                        args.prediction_method, runtime_inputs,
                        execution_mode=args.execution_mode,
                        timing_mode=args.serving_timing_mode,
                        host_launch_profiles=host_launch_profiles,
                        host_runtime_prepare_profiles=host_runtime_prepare_profiles,
                        host_request_path_profiles=host_request_path_profiles,
                        runtime_prepare_event_builder=runtime_prepare_event_builder,
                        closed_loop_client_count=args.closed_loop_client_count),
                    backends,
                    hardware_by_id,
                    progress_callback=progress_callback,
                ).run()
                construction_lifecycle.pop_all()
        else:
            deployment = build_colocated_plan(
                DeploymentConfig(
                    "colocated",
                    (WorkerPoolConfig(
                        "main",
                        "colocated",
                        prefill_hardware.hardware_id,
                        ParallelConfig(
                            tensor_parallel_size=args.prefill_tp_size,
                            pipeline_parallel_size=args.prefill_pp_size,
                            enable_expert_parallel=(
                                args.enable_expert_parallel),
                        ),
                        collectives=_tensor_collective_configs(
                            "main", args.prefill_tp_size,
                            prefill_hardware.hardware_id,
                            args.collective_backend,
                            args.tensor_all_reduce_profile_sha256,
                            args.tensor_all_gather_profile_sha256),
                        pipeline_p2p=_pipeline_p2p_configs(
                            "main", args.prefill_pp_size,
                            hardware_id=prefill_hardware.hardware_id,
                            backend=args.collective_backend,
                            tensor_parallel_size=args.prefill_tp_size,
                            ranks_per_node=args.pipeline_ranks_per_node),
                        kv_cache=(
                            None
                            if args.kv_lifecycle_replay is None
                            else args.kv_lifecycle_replay.cache_config
                        ),
                    ),),
                    router=router_config,
                ),
                num_layers=model.num_hidden_layers,
                num_experts=model.num_experts,
                pipeline_layer_partition=tuple(
                    layer_end - layer_start
                    for layer_start, layer_end in (
                        pipeline_layer_ranges_for_model(
                            model, args.prefill_pp_size)
                    )
                ),
            )
            hardware_by_id = {prefill_hardware.hardware_id: prefill_hardware}
            validate_capacity_pools(("main",))
            validate_workload_emission_parallelism(
                deployment,
                model,
                allow_single_pool_tp2=True,
                allow_single_pool_ep2=True,
                allow_single_pool_tp1_pp=True,
                allow_single_pool_tp2_pp2=True,
            )
            with ExitStack() as construction_lifecycle:
                backend = make_backend(
                    "main",
                    "colocated",
                    prefill_hardware,
                    args.hardware_config,
                    args.prefill_tp_size,
                    args.prefill_pp_size,
                )
                construction_lifecycle.callback(backend.close)
                if args.backend == "vllm":
                    runtime_info = backend.runtime_info
                    scheduler_runtime["main"] = asdict(runtime_info)
                    if expert_routing.source_kind == "replay_observation":
                        expert_routing = expert_routing.bind_vllm_runtime(
                            version=runtime_info.version,
                            source_commit=runtime_info.source_commit,
                            package_python_manifest_sha256=(
                                runtime_info.package_python_manifest_sha256),
                            model_config_sha256=(
                                runtime_info.model_config_sha256),
                        )
                        runtime_inputs = RuntimeInputs(
                            expert_routing=expert_routing,
                            attention_kv_layout=args.attention_kv_layout,
                            attention_kv_distribution=(
                                args.attention_kv_distribution),
                            graph_execution=args.graph_execution_config,
                        )
                elif args.backend == "conformance":
                    pass
                elif args.backend == "sglang_mirror":
                    scheduler_runtime["main"] = backend.runtime_info
                else:
                    raise ValueError(
                        f"unknown serving backend: {args.backend!r}")
                simulation = Simulation(
                    SimulationConfig(
                        model, deployment, requests, args.launch_policy,
                        args.prediction_method, runtime_inputs,
                        execution_mode=args.execution_mode,
                        step_retention=(
                            "streaming_aggregate"
                            if use_streaming_aggregate else "full"),
                        execution_workers=args.execution_workers,
                        timing_mode=args.serving_timing_mode,
                        host_launch_profiles=host_launch_profiles,
                        host_runtime_prepare_profiles=host_runtime_prepare_profiles,
                        host_request_path_profiles=host_request_path_profiles,
                        runtime_prepare_event_builder=runtime_prepare_event_builder,
                        closed_loop_client_count=args.closed_loop_client_count),
                    backend,
                    hardware_by_id,
                    kv_cache_tier_manager=(
                        None
                        if args.kv_lifecycle_replay is None
                        else KVCacheTierManager(
                            args.kv_lifecycle_replay.cache_config)
                    ),
                    kv_lifecycle_replay=args.kv_lifecycle_replay,
                    scheduler_runtime_sha256=(
                        None
                        if args.kv_lifecycle_replay is None
                        else vllm_scheduler_runtime_sha256(runtime_info)
                    ),
                    progress_callback=progress_callback,
                    streaming_step_consumer=(
                        None
                        if streaming_artifact is None
                        else streaming_artifact.consume
                    ),
                    streaming_step_projector=(
                        None
                        if streaming_artifact is None
                        else streaming_artifact.execution_contract_digest
                    ),
                    streaming_prehashed_step_consumer=(
                        None
                        if streaming_artifact is None
                        else streaming_artifact.consume_prehashed
                    ),
                )
                construction_lifecycle.pop_all()
            result = simulation.run()
    elif args.deployment == "af_disagg":
        _activation_profile, activation_profile_sha256 = (
            _protobridge_p2p_profile(
                prefill_hardware.hardware_id,
                args.activation_transfer_link,
            ))
        attention_ranks = tuple(range(args.prefill_tp_size))
        ffn_ranks = tuple(range(
            args.prefill_tp_size,
            args.prefill_tp_size + args.ffn_tp_size,
        ))
        deployment = build_af_disagg_plan(
            DeploymentConfig(
                "af_disagg",
                (
                    WorkerPoolConfig(
                        "attention",
                        "attention",
                        prefill_hardware.hardware_id,
                        ParallelConfig(
                            tensor_parallel_size=args.prefill_tp_size),
                        attention_ranks,
                        kv_cache=(
                            None
                            if args.kv_lifecycle_replay is None
                            else args.kv_lifecycle_replay.cache_config
                        ),
                        collectives=_tensor_collective_configs(
                            "attention", args.prefill_tp_size,
                            prefill_hardware.hardware_id,
                            args.collective_backend,
                            args.tensor_all_reduce_profile_sha256,
                            args.tensor_all_gather_profile_sha256),
                    ),
                    WorkerPoolConfig(
                        "ffn",
                        "ffn",
                        prefill_hardware.hardware_id,
                        ParallelConfig(
                            tensor_parallel_size=args.ffn_tp_size,
                            enable_expert_parallel=(
                                args.enable_expert_parallel),
                        ),
                        ffn_ranks,
                        collectives=_tensor_collective_configs(
                            "ffn", args.ffn_tp_size,
                            prefill_hardware.hardware_id,
                            args.collective_backend,
                            args.tensor_all_reduce_profile_sha256,
                            args.tensor_all_gather_profile_sha256),
                    ),
                ),
                (
                    RouteConfig(
                        "attention", "ffn", "activation",
                        args.activation_transfer_algorithm,
                        args.activation_transfer_link,
                        "point_to_point",
                        resource_id=_p2p_resource_id(
                            args.activation_transfer_link,
                            "attention",
                            "ffn",
                        ),
                        backend="protobridge",
                        backend_config_sha256=activation_profile_sha256,
                    ),
                    RouteConfig(
                        "ffn", "attention", "activation",
                        args.activation_transfer_algorithm,
                        args.activation_transfer_link,
                        "point_to_point",
                        resource_id=_p2p_resource_id(
                            args.activation_transfer_link,
                            "ffn",
                            "attention",
                        ),
                        backend="protobridge",
                        backend_config_sha256=activation_profile_sha256,
                    ),
                ),
                router=router_config,
            ),
            num_layers=model.num_hidden_layers,
            num_experts=model.num_experts,
        )
        hardware_by_id = {
            prefill_hardware.hardware_id: prefill_hardware,
        }
        validate_capacity_pools(("attention",))
        validate_workload_emission_parallelism(
            deployment,
            model,
            allow_single_pool_tp2=False,
            allow_single_pool_ep2=False,
            allow_single_pool_tp1_pp=False,
            allow_single_pool_tp2_pp2=False,
        )
        simulation_config = SimulationConfig(
            model, deployment, requests, args.launch_policy,
            args.prediction_method, runtime_inputs,
            execution_mode=args.execution_mode,
            step_retention=(
                "streaming_aggregate" if use_streaming_aggregate else "full"),
            timing_mode=args.serving_timing_mode,
            host_launch_profiles=host_launch_profiles,
            host_runtime_prepare_profiles=host_runtime_prepare_profiles,
            host_request_path_profiles=host_request_path_profiles,
            runtime_prepare_event_builder=runtime_prepare_event_builder,
            closed_loop_client_count=args.closed_loop_client_count)
        if args.enable_expert_parallel:
            validate_standalone_af_ep2_scope(simulation_config)
        af_mtp_preemption = (
            args.mtp_num_speculative_tokens != 0
            and args.max_requests == 2)
        if args.mtp_num_speculative_tokens != 0:
            validate_standalone_af_mtp_scope(
                simulation_config,
                allow_preemption=af_mtp_preemption,
            )
        with ExitStack() as construction_lifecycle:
            backend = make_backend(
                "attention", "attention", prefill_hardware,
                args.hardware_config,
                args.prefill_tp_size)
            construction_lifecycle.callback(backend.close)
            if args.backend == "vllm":
                scheduler_runtime["attention"] = asdict(
                    backend.runtime_info)
            elif args.backend == "conformance":
                pass
            else:
                raise ValueError(
                    f"unknown serving backend: {args.backend!r}")
            result = MultiSimulation(
                simulation_config,
                {"attention": backend},
                hardware_by_id,
                kv_cache_tier_manager=(
                    None
                    if args.kv_lifecycle_replay is None
                    else KVCacheTierManager(
                        args.kv_lifecycle_replay.cache_config)
                ),
                kv_lifecycle_replay=args.kv_lifecycle_replay,
                scheduler_runtime_sha256=(
                    None
                    if args.kv_lifecycle_replay is None
                    else vllm_scheduler_runtime_sha256(backend.runtime_info)
                ),
                enable_standalone_af_mtp=(
                    args.mtp_num_speculative_tokens != 0),
                enable_standalone_af_mtp_preemption=(
                    af_mtp_preemption),
                progress_callback=progress_callback,
            ).run()
            construction_lifecycle.pop_all()
    elif args.deployment == "pd_af_disagg":
        _activation_profile, activation_profile_sha256 = (
            _protobridge_p2p_profile(
                prefill_hardware.hardware_id,
                args.activation_transfer_link,
            ))
        _kv_profile, kv_profile_sha256 = _protobridge_p2p_profile(
            prefill_hardware.hardware_id,
            args.transfer_link,
        )
        if (not isinstance(args.decode_hardware_config, str)
                or not args.decode_hardware_config):
            raise ValueError(
                "--decode-hardware-config is required for pd_af_disagg")
        if (Path(args.decode_hardware_config).resolve()
                != Path(args.hardware_config).resolve()):
            raise NotImplementedError(
                "the current PD+AF composition requires every worker pool "
                "to use the same hardware config file")
        def pd_af_pool_names(phase, operator_group, count):
            if count == 1:
                return (f"{phase}-{operator_group}",)
            return tuple(
                f"{phase}-{operator_group}-{index}"
                for index in range(count)
            )

        prefill_attention_names = pd_af_pool_names(
            "prefill", "attention", args.num_prefill_instances)
        prefill_ffn_names = pd_af_pool_names(
            "prefill", "ffn", args.num_prefill_instances)
        decode_attention_names = pd_af_pool_names(
            "decode", "attention", args.num_decode_instances)
        decode_ffn_names = pd_af_pool_names(
            "decode", "ffn", args.num_decode_instances)
        pool_configs = []
        next_rank = 0
        for phase, attention_names, ffn_names in (
                ("prefill", prefill_attention_names, prefill_ffn_names),
                ("decode", decode_attention_names, decode_ffn_names)):
            for attention_name, ffn_name in zip(
                    attention_names, ffn_names, strict=True):
                pool_configs.extend((
                    WorkerPoolConfig(
                        attention_name,
                        f"{phase}-attention",
                        prefill_hardware.hardware_id,
                        ParallelConfig(tensor_parallel_size=1),
                        (next_rank,),
                    ),
                    WorkerPoolConfig(
                        ffn_name,
                        f"{phase}-ffn",
                        prefill_hardware.hardware_id,
                        ParallelConfig(tensor_parallel_size=1),
                        (next_rank + 1,),
                    ),
                ))
                next_rank += 2
        routes = []
        for attention_names, ffn_names in (
                (prefill_attention_names, prefill_ffn_names),
                (decode_attention_names, decode_ffn_names)):
            for attention_name, ffn_name in zip(
                    attention_names, ffn_names, strict=True):
                routes.extend((
                    RouteConfig(
                        attention_name, ffn_name, "activation",
                        args.activation_transfer_algorithm,
                        args.activation_transfer_link,
                        "point_to_point",
                        resource_id=_p2p_resource_id(
                            args.activation_transfer_link,
                            attention_name,
                            ffn_name,
                        ),
                        backend="protobridge",
                        backend_config_sha256=activation_profile_sha256,
                    ),
                    RouteConfig(
                        ffn_name, attention_name, "activation",
                        args.activation_transfer_algorithm,
                        args.activation_transfer_link,
                        "point_to_point",
                        resource_id=_p2p_resource_id(
                            args.activation_transfer_link,
                            ffn_name,
                            attention_name,
                        ),
                        backend="protobridge",
                        backend_config_sha256=activation_profile_sha256,
                    ),
                ))
        for prefill_name in prefill_attention_names:
            for decode_name in decode_attention_names:
                routes.append(RouteConfig(
                    prefill_name, decode_name, "kv",
                    args.transfer_algorithm,
                    args.transfer_link,
                    "point_to_point",
                    resource_id=_p2p_resource_id(
                        args.transfer_link,
                        prefill_name,
                        decode_name,
                    ),
                    backend="protobridge",
                    backend_config_sha256=kv_profile_sha256,
                ))
        deployment = build_pd_disagg_plan(
            DeploymentConfig(
                "pd_af_disagg",
                pool_configs,
                routes,
                router=router_config,
            ),
            num_layers=model.num_hidden_layers,
            num_experts=model.num_experts,
        )
        hardware_by_id = {
            prefill_hardware.hardware_id: prefill_hardware,
        }
        scheduler_pool_names = (
            prefill_attention_names + decode_attention_names)
        validate_capacity_pools(scheduler_pool_names)
        scheduler_roles = tuple(
            (name, "prefill-attention")
            for name in prefill_attention_names
        ) + tuple(
            (name, "decode-attention")
            for name in decode_attention_names
        )
        simulation_config = SimulationConfig(
            model, deployment, requests, args.launch_policy,
            args.prediction_method, runtime_inputs,
            execution_mode=args.execution_mode,
            timing_mode=args.serving_timing_mode,
            host_launch_profiles=host_launch_profiles,
            host_runtime_prepare_profiles=host_runtime_prepare_profiles,
            host_request_path_profiles=host_request_path_profiles,
            runtime_prepare_event_builder=runtime_prepare_event_builder,
            closed_loop_client_count=args.closed_loop_client_count)
        if args.mtp_num_speculative_tokens != 0:
            validate_pd_af_mtp_scope(
                simulation_config,
                fixed_accepted_tokens=args.mtp_fixed_accepted_tokens,
            )
        with ExitStack() as construction_lifecycle:
            backends = {}
            for pool_name, role in scheduler_roles:
                backend = make_backend(
                    pool_name, role, prefill_hardware,
                    args.hardware_config, 1)
                construction_lifecycle.callback(backend.close)
                backends[pool_name] = backend
            if args.backend == "vllm":
                scheduler_runtime.update({
                    pool_name: asdict(backend.runtime_info)
                    for pool_name, backend in backends.items()
                })
            elif args.backend == "conformance":
                pass
            else:
                raise ValueError(
                    f"unknown serving backend: {args.backend!r}")
            result = MultiSimulation(
                simulation_config,
                backends,
                hardware_by_id,
                enable_pd_af_mtp=(
                    args.mtp_num_speculative_tokens != 0),
                pd_af_mtp_fixed_accepted_tokens=(
                    args.mtp_fixed_accepted_tokens
                    if args.mtp_num_speculative_tokens != 0 else None
                ),
                progress_callback=progress_callback,
                streaming_step_consumer=(
                    None
                    if streaming_artifact is None
                    else streaming_artifact.consume
                ),
                streaming_step_projector=(
                    None
                    if streaming_artifact is None
                    else streaming_artifact.execution_contract_digest
                ),
                streaming_prehashed_step_consumer=(
                    None
                    if streaming_artifact is None
                    else streaming_artifact.consume_prehashed
                ),
            ).run()
            construction_lifecycle.pop_all()
    elif args.deployment == "pd_disagg":
        _kv_profile, kv_profile_sha256 = _protobridge_p2p_profile(
            prefill_hardware.hardware_id,
            args.transfer_link,
        )
        if (not isinstance(args.decode_hardware_config, str)
                or not args.decode_hardware_config):
            raise ValueError(
                "--decode-hardware-config is required for pd_disagg")
        decode_hardware = HWSpec(args.decode_hardware_config)
        prefill_hardware_path = Path(args.hardware_config).resolve()
        decode_hardware_path = Path(args.decode_hardware_config).resolve()
        if (prefill_hardware.hardware_id == decode_hardware.hardware_id
                and prefill_hardware_path != decode_hardware_path):
            raise ValueError(
                "pd_disagg cannot bind different hardware config files with "
                "the same hardware_id: "
                f"{prefill_hardware.hardware_id!r} maps to "
                f"{prefill_hardware_path} and {decode_hardware_path}; reuse "
                "one config path or assign distinct canonical hardware IDs")
        prefill_names = tuple(
            f"prefill-{index}"
            for index in range(args.num_prefill_instances)
        )
        decode_names = tuple(
            f"decode-{index}"
            for index in range(args.num_decode_instances)
        )
        pool_configs = []
        next_rank = 0
        for pool_name in prefill_names:
            ranks = tuple(range(
                next_rank,
                next_rank + args.prefill_tp_size,
            ))
            next_rank += args.prefill_tp_size
            pool_configs.append(WorkerPoolConfig(
                pool_name,
                "prefill",
                prefill_hardware.hardware_id,
                ParallelConfig(
                    tensor_parallel_size=args.prefill_tp_size,
                    enable_expert_parallel=args.enable_expert_parallel,
                ),
                ranks,
                collectives=_tensor_collective_configs(
                    pool_name,
                    args.prefill_tp_size,
                    prefill_hardware.hardware_id,
                    args.collective_backend,
                    args.tensor_all_reduce_profile_sha256,
                    args.tensor_all_gather_profile_sha256,
                ),
            ))
        for pool_name in decode_names:
            ranks = tuple(range(
                next_rank,
                next_rank + args.decode_tp_size,
            ))
            next_rank += args.decode_tp_size
            pool_configs.append(WorkerPoolConfig(
                pool_name,
                "decode",
                decode_hardware.hardware_id,
                ParallelConfig(
                    tensor_parallel_size=args.decode_tp_size,
                    enable_expert_parallel=args.enable_expert_parallel,
                ),
                ranks,
                collectives=_tensor_collective_configs(
                    pool_name,
                    args.decode_tp_size,
                    decode_hardware.hardware_id,
                    args.collective_backend,
                    args.tensor_all_reduce_profile_sha256,
                    args.tensor_all_gather_profile_sha256,
                ),
            ))
        routes = tuple(
            RouteConfig(
                source,
                destination,
                "kv",
                args.transfer_algorithm,
                args.transfer_link,
                "point_to_point",
                resource_id=_p2p_resource_id(
                    args.transfer_link,
                    source,
                    destination,
                ),
                backend="protobridge",
                backend_config_sha256=kv_profile_sha256,
            )
            for source in prefill_names
            for destination in decode_names
        )
        deployment = build_pd_disagg_plan(
            DeploymentConfig(
                "pd_disagg",
                tuple(pool_configs),
                routes,
                router=router_config,
            ),
            num_layers=model.num_hidden_layers,
            num_experts=model.num_experts,
        )
        hardware_by_id = {
            prefill_hardware.hardware_id: prefill_hardware,
            decode_hardware.hardware_id: decode_hardware,
        }
        validate_capacity_pools(prefill_names + decode_names)
        backends = {}
        with ExitStack() as construction_lifecycle:
            for pool_name in prefill_names:
                backend = make_backend(
                    pool_name, "prefill", prefill_hardware,
                    args.hardware_config,
                    args.prefill_tp_size)
                construction_lifecycle.callback(backend.close)
                backends[pool_name] = backend
            for pool_name in decode_names:
                backend = make_backend(
                    pool_name, "decode", decode_hardware,
                    args.decode_hardware_config,
                    args.decode_tp_size)
                construction_lifecycle.callback(backend.close)
                backends[pool_name] = backend
            if args.backend == "vllm":
                scheduler_runtime.update({
                    pool_name: asdict(backend.runtime_info)
                    for pool_name, backend in backends.items()
                })
            elif args.backend == "conformance":
                pass
            else:
                raise ValueError(
                    f"unknown serving backend: {args.backend!r}")
            result = MultiSimulation(
                SimulationConfig(
                    model, deployment, requests, args.launch_policy,
                    args.prediction_method, runtime_inputs,
                    authorized_h200_pd_mtp_sweep=(
                        _is_glm_h200_pd_mtp_target_scope(args, True)
                        or _is_dsv3_h200_pd_mtp_target_scope(args)),
                    execution_mode=args.execution_mode,
                    step_retention=(
                        "streaming_aggregate"
                        if use_streaming_aggregate else "full"),
                    timing_mode=args.serving_timing_mode,
                    host_launch_profiles=host_launch_profiles,
                    host_runtime_prepare_profiles=host_runtime_prepare_profiles,
            host_request_path_profiles=host_request_path_profiles,
                    runtime_prepare_event_builder=runtime_prepare_event_builder,
                    closed_loop_client_count=args.closed_loop_client_count),
                backends,
                hardware_by_id,
                progress_callback=progress_callback,
                streaming_step_consumer=(
                    None
                    if streaming_artifact is None
                    else streaming_artifact.consume
                ),
                streaming_step_projector=(
                    None
                    if streaming_artifact is None
                    else streaming_artifact.execution_contract_digest
                ),
                streaming_prehashed_step_consumer=(
                    None
                    if streaming_artifact is None
                    else streaming_artifact.consume_prehashed
                ),
            ).run()
            construction_lifecycle.pop_all()
    else:
        raise ValueError(f"unknown deployment mode: {args.deployment!r}")
    simulation_wall_time_s = time.perf_counter() - simulation_wall_started
    if (request_source is not None
            and request_source["kind"] == "production_trace"):
        request_source["prefix_alignment"] = prefix_alignment_manifest(
            requests,
            result,
            args.kv_block_size_tokens,
        )
    if (args.backend == "vllm"
            and args.deployment in ("pd_disagg", "pd_af_disagg")):
        kv_handoff_mode = "metadata_only_private_destination_kv"
    elif args.deployment in ("pd_disagg", "pd_af_disagg"):
        kv_handoff_mode = (
            "modeled_transfer_without_framework_block_ownership")
    elif args.deployment in ("colocated", "af_disagg"):
        kv_handoff_mode = "not_applicable"
    else:
        raise ValueError(
            f"unsupported serving deployment: {args.deployment!r}")
    slo_goodput = slo_goodput_manifest(
        requests,
        result.metrics,
        result.schedule_trace.end_time_s,
        slo_contract,
    )
    # Reported on the engine clock, the same normalization the observed
    # Scheduler trace carries, so that a predicted and a measured wait cover
    # the same boundary.
    first_schedule_start_s = engine_clock_first_schedule_start_s(
        result.schedule_trace, requests)
    expected_request_ids = {
        request.request_id for request in requests
    }
    if set(first_schedule_start_s) != expected_request_ids:
        raise ValueError(
            "schedule trace does not provide a first schedule boundary for "
            "every request"
        )
    timing_status = performance_timing_status(
        args.launch_policy,
        tuple(sorted(hardware_by_id)),
        uses_development_protobridge=_uses_development_protobridge(args),
    )
    execution_provenance = _execution_provenance(
        args,
        runtime_inputs,
        hardware_by_id,
        timing_status,
    )
    if isinstance(result, StreamingSimulationResult):
        if streaming_artifact is None:
            raise RuntimeError(
                "streaming simulation lost its artifact accumulator")
        execution_artifact = streaming_artifact.manifest(
            result,
            performance_timing_status=timing_status,
        )
        streaming_chain_field = (
            "ordered_step_execution_contract_chain_sha256"
            if execution_artifact["schema_version"] in (10, 11, 12, 13)
            else "ordered_step_manifest_chain_sha256"
        )
        analysis_status = {
            "schema_version": 1,
            "availability": "not_materialized_streaming_aggregate",
            "reason": (
                "full per-step IR is consumed once and released to keep "
                "external-request simulation memory bounded"),
            "ordered_step_manifest_chain_sha256": (
                execution_artifact["streaming_aggregate"][
                    streaming_chain_field]),
        }
        system_resources = dict(analysis_status)
        critical_path = dict(analysis_status)
    else:
        if streaming_artifact is not None:
            raise RuntimeError(
                "full simulation unexpectedly retained a streaming "
                "artifact accumulator")
        # Closed-loop runs stamp each successor at its client's completion
        # instant; downstream per-request analysis must see those injected
        # arrival times, not the canonical stream.
        analysis_requests = (
            requests if result.metrics_requests is None
            else result.metrics_requests)
        system_resources = system_resource_manifest(
            deployment,
            analysis_requests,
            result,
        )
        critical_path = critical_path_manifest(
            deployment,
            analysis_requests,
            result,
        )
        execution_artifact = execution_artifact_manifest(
            result,
            performance_timing_status=timing_status,
        )
    mtp_pd_handoff = _mtp_pd_handoff_contract(
        args,
        requests,
        result,
        deployment,
        scheduler_runtime,
    )
    payload = {
        "version": result_version,
        "backend": args.backend,
        "deployment_id": deployment.deployment_id,
        "hardware_ids": sorted(hardware_by_id),
        "model_compute_dtype": model.compute_dtype,
        "model_config_provenance": {
            "config_sha256": model.config_sha256,
            "source_config_sha256": model.source_config_sha256,
            "canonical_config_sha256": model.canonical_config_sha256,
            "canonicalization": model.config_canonicalization,
            "extraction_path": model.config_extraction_path,
            "extracted_config_sha256": model.extracted_config_sha256,
            "model_id": model.model_id,
            "revision": model.model_revision,
            "source_url": model.model_source_url,
            "scope": model.model_scope,
            "architecture": model.architecture,
            "model_type": model.model_type,
            "attention_layout": model.attention_layout,
            "compute_dtype": model.compute_dtype,
            "weight_dtype": model.weight_dtype,
            "weight_group_size": model.weight_group_size,
            "quantized_weight_targets": model.quantized_weight_targets,
            "quantized_weight_ignore": model.quantized_weight_ignore,
        },
        "deployment": args.deployment,
        "deployment_plan": deployment_plan_manifest(deployment),
        "routing_policy": "least_queued_tokens",
        "route_decision_version": 2,
        "transfer_trace_version": 2,
        "kv_handoff_mode": kv_handoff_mode,
        "kv_capacity": kv_capacity_manifest(
            kv_capacity_plans.values(), runtime_memory_policy),
        "kv_lifecycle_replay": (
            None
            if args.kv_lifecycle_replay is None
            else args.kv_lifecycle_replay.manifest()
        ),
        "launch_policy": args.launch_policy,
        "prediction_method": args.prediction_method,
        "execution_mode": args.execution_mode,
        "serving_timing_mode": args.serving_timing_mode,
        "host_timing": _host_timing_manifest(
            result, host_launch_profiles, host_runtime_prepare_profiles),
        "execution_cache_stats": result.execution_cache_stats,
        "execution_parallelism": {
            "worker_count": result.execution_performance_stats[
                "parallel_worker_count"],
            "parallel_step_count": result.execution_performance_stats[
                "parallel_step_count"],
            "parallel_execution_wall_time_s": (
                result.execution_performance_stats[
                    "parallel_execution_wall_time_s"]),
            "worker_aggregate_phase_s": (
                result.execution_performance_stats[
                    "parallel_worker_aggregate_phase_s"]),
        },
        "performance_timing_status": timing_status,
        "execution_provenance": execution_provenance,
        "workload_assumption_binding": (
            _workload_assumption_binding_manifest(
                actual_request_stream_sha256,
                runtime_inputs,
                args,
            )
        ),
        "main_chain_status": {
            "status": "executed",
            "chain": (
                "RequestStream -> SchedulerBackend/BackendStep/ScheduleStep "
                "-> build_workload -> WorkloadGraph -> lower_workload -> "
                "KernelSpec -> predict"
            ),
            "execution_artifact_sha256": execution_artifact["artifact_sha256"],
        },
        "execution_artifact": execution_artifact,
        "scheduler_runtime": scheduler_runtime,
        "mtp_scenario": mtp_scenario_manifest(VLLMSchedulerConfig(
            args.model,
            tensor_parallel_size=args.prefill_tp_size,
            pipeline_parallel_size=args.prefill_pp_size,
            enable_chunked_prefill=args.enable_chunked_prefill,
            enable_prefix_caching=args.enable_prefix_caching,
            mtp_num_speculative_tokens=args.mtp_num_speculative_tokens,
            mtp_acceptance_policy=args.mtp_acceptance_policy,
            mtp_fixed_accepted_tokens=args.mtp_fixed_accepted_tokens,
            mtp_acceptance_source_id=args.mtp_acceptance_source_id,
            mtp_acceptance_replay=args.mtp_acceptance_replay,
            mtp_acceptance_assumption=args.mtp_acceptance_assumption,
            request_stream_sha256=(
                actual_request_stream_sha256
                if (args.mtp_acceptance_replay is not None
                    or args.mtp_acceptance_assumption is not None)
                else None
            ),
            decode_admission_contract_sha256=(
                mtp_decode_admission_sha256
                if (args.mtp_acceptance_replay is not None
                    or args.mtp_acceptance_assumption is not None)
                else None
            ),
        )),
        "mtp_acceptance_outcomes": _mtp_acceptance_outcomes_manifest(
            args, result, deployment),
        "mtp_pd_handoff_contract": mtp_pd_handoff,
        "arrival_process": arrival_process,
        "request_arrivals_s": [
            {
                "request_id": request.request_id,
                "arrival_time_s": request.arrival_time_s,
            }
            for request in requests
        ],
        "workload_hash": workload_hash,
        "num_steps": (
            result.num_steps
            if isinstance(result, StreamingSimulationResult)
            else len(result.steps)
        ),
        "num_transfers": len(result.schedule_trace.transfers),
        "num_kernel_calls": (
            result.num_kernel_calls
            if isinstance(result, StreamingSimulationResult)
            else (
                sum(
                    step.kernel_plan.logical_call_count
                    for step in result.steps
                )
                + sum(
                    transfer.kernel_plan.logical_call_count
                    for transfer in result.transfers
                )
            )
        ),
        "schedule_end_time_s": result.schedule_trace.end_time_s,
        "simulation_wall_time_s": simulation_wall_time_s,
        "evaluation_wall_time_breakdown_s": {
            **result.execution_performance_stats["phase_wall_time_s"],
            "artifact_generation": (
                0.0
                if streaming_artifact is None
                else streaming_artifact.consume_wall_time_s
            ),
            "scheduler_and_orchestration": max(
                0.0,
                simulation_wall_time_s
                - sum(result.execution_performance_stats[
                    "phase_wall_time_s"].values())
                - (
                    0.0
                    if streaming_artifact is None
                    else streaming_artifact.consume_wall_time_s
                ),
            ),
        },
        "process_peak_rss_bytes": max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        ) * 1024,
        "scheduler_progress": scheduler_progress_manifest(
            requests, result.schedule_trace),
        "route_decisions": [
            asdict(decision)
            for decision in result.schedule_trace.route_decisions
        ],
        "transfers": [
            asdict(transfer)
            for transfer in result.schedule_trace.transfers
        ],
        "metrics": {
            "request_throughput_per_s": result.metrics.request_throughput_per_s,
            "output_token_throughput_per_s": (
                result.metrics.output_token_throughput_per_s),
            "token_throughput_per_s": result.metrics.token_throughput_per_s,
            "latency_distributions": latency_distribution_manifest(
                result.metrics.requests),
            "slo_goodput": slo_goodput,
            "system_resources": system_resources,
            "critical_path": critical_path,
            "requests": {
                request_id: {
                    "ttft_s": metric.ttft_s,
                    "tpot_s": metric.tpot_s,
                    "latency_s": metric.latency_s,
                    "first_schedule_start_s": first_schedule_start_s[
                        request_id
                    ],
                }
                for request_id, metric in result.metrics.requests.items()
            },
        },
    }
    if request_source is not None:
        payload["request_source"] = request_source
    progress_callback.stop()
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        persisted_payload = _persistence_projection(payload)
        path.write_text(
            json.dumps(persisted_payload, indent=2, sort_keys=True) + "\n")
    return payload


def _persistence_projection(payload: dict[str, object]) -> dict[str, object]:
    """Project every external request source before persistence."""

    if "request_source" not in payload:
        return payload
    request_source = payload["request_source"]
    if not isinstance(request_source, dict):
        raise TypeError("simulation request_source must be a dict")
    if request_source["kind"] == "production_trace":
        return _production_trace_aggregate_result(payload)
    if request_source["kind"] == "canonical_requests":
        return _canonical_requests_aggregate_result(payload)
    raise ValueError(
        f"unknown simulation request source kind: "
        f"{request_source['kind']!r}")


def _production_trace_aggregate_result(
    payload: dict[str, object],
) -> dict[str, object]:
    """Project an in-memory Trace result to a request-safe persisted form."""

    if not isinstance(payload, dict):
        raise TypeError("production Trace aggregate projection requires a dict")
    if "request_source" not in payload:
        raise ValueError(
            "production Trace aggregate projection requires request_source")
    request_source = payload["request_source"]
    if not isinstance(request_source, dict):
        raise TypeError("production Trace request_source must be a dict")
    if request_source["kind"] != "production_trace":
        raise ValueError(
            "aggregate projection is reserved for production Trace results")
    prefix_alignment = request_source["prefix_alignment"]
    if not isinstance(prefix_alignment, dict):
        raise TypeError("production Trace prefix alignment must be a dict")
    alignment_entries = prefix_alignment["entries"]
    if not isinstance(alignment_entries, (list, tuple)):
        raise TypeError("production Trace prefix entries must be a sequence")
    alignment_totals = {
        "request_count": len(alignment_entries),
        "trace_prefix_hit_tokens": sum(
            entry["trace_prefix_hit_tokens"] for entry in alignment_entries),
        "scheduler_prefix_hit_tokens": sum(
            entry["scheduler_prefix_hit_tokens"]
            for entry in alignment_entries),
        "prefix_scheduler_delta_tokens": sum(
            entry["prefix_scheduler_delta_tokens"]
            for entry in alignment_entries),
    }
    aggregate_request_source = {
        "schema_version": request_source["schema_version"],
        "kind": request_source["kind"],
        "trace": request_source["trace"],
        "replay": request_source["replay"],
        "zero_output_contract": request_source["zero_output_contract"],
        "prefix_alignment": {
            "schema_version": prefix_alignment["schema_version"],
            "owner": prefix_alignment["owner"],
            "vllm_version": prefix_alignment["vllm_version"],
            "rule": prefix_alignment["rule"],
            "kv_block_size_tokens": (
                prefix_alignment["kv_block_size_tokens"]),
            "aggregate": alignment_totals,
            "request_level_entries_sha256": _canonical_json_sha256(
                alignment_entries),
            "request_level_entries": "omitted",
        },
    }
    return _request_safe_aggregate_result(
        payload,
        result_kind="production_trace_simulation_aggregate",
        projection="aggregate_only_production_trace_v1",
        aggregate_request_source=aggregate_request_source,
    )


def _canonical_requests_aggregate_result(
    payload: dict[str, object],
) -> dict[str, object]:
    """Project canonical requests to a request-safe persisted result."""

    if not isinstance(payload, dict):
        raise TypeError(
            "canonical requests aggregate projection requires a dict")
    if "request_source" not in payload:
        raise ValueError(
            "canonical requests aggregate projection requires request_source")
    request_source = payload["request_source"]
    if not isinstance(request_source, dict):
        raise TypeError("canonical requests request_source must be a dict")
    if request_source["kind"] != "canonical_requests":
        raise ValueError(
            "canonical aggregate projection requires canonical requests")
    aggregate_request_source = {
        "schema_version": request_source["schema_version"],
        "kind": request_source["kind"],
        "input_path": request_source["input_path"],
        "source_path": request_source["source_path"],
        "source_sha256": request_source["source_sha256"],
        "source_size_bytes": request_source["source_size_bytes"],
        "input_schema": request_source["input_schema"],
        "row_count": request_source["row_count"],
        "request_id_policy": request_source["request_id_policy"],
        "equal_arrival_order": request_source["equal_arrival_order"],
        "normalized_request_stream_sha256": (
            request_source["normalized_request_stream_sha256"]),
    }
    return _request_safe_aggregate_result(
        payload,
        result_kind="canonical_requests_simulation_aggregate",
        projection="aggregate_only_canonical_requests_v1",
        aggregate_request_source=aggregate_request_source,
    )


def _request_safe_aggregate_result(
    payload: dict[str, object],
    *,
    result_kind: str,
    projection: str,
    aggregate_request_source: dict[str, object],
) -> dict[str, object]:
    """Omit request-level content from one externally sourced simulation."""

    execution_artifact = payload["execution_artifact"]
    if not isinstance(execution_artifact, dict):
        raise TypeError("simulation execution artifact must be a dict")
    aggregate_artifact = {
        "schema_version": execution_artifact["schema_version"],
        "hash_method": execution_artifact["hash_method"],
        "performance_timing_status": (
            execution_artifact["performance_timing_status"]),
        "counts": execution_artifact["counts"],
        "artifact_sha256": execution_artifact["artifact_sha256"],
        "request_level_artifact": "omitted",
    }
    metrics = payload["metrics"]
    if not isinstance(metrics, dict):
        raise TypeError("simulation metrics must be a dict")
    aggregate_metrics = {
        "request_throughput_per_s": metrics["request_throughput_per_s"],
        "output_token_throughput_per_s": (
            metrics["output_token_throughput_per_s"]),
        "token_throughput_per_s": metrics["token_throughput_per_s"],
        "latency_distributions": metrics["latency_distributions"],
        "slo_goodput": metrics["slo_goodput"],
        "system_resources_sha256": _canonical_json_sha256(
            metrics["system_resources"]),
        "critical_path_sha256": _canonical_json_sha256(
            metrics["critical_path"]),
        "request_level_metrics": "omitted",
    }
    kv_lifecycle = payload["kv_lifecycle_replay"]
    if kv_lifecycle is None:
        aggregate_kv_lifecycle = None
    else:
        aggregate_kv_lifecycle = {
            "manifest_sha256": _canonical_json_sha256(kv_lifecycle),
            "request_level_manifest": "omitted",
        }
    mtp_outcomes = _aggregate_mtp_acceptance_outcomes(
        payload["mtp_acceptance_outcomes"])
    mtp_pd_handoff = _aggregate_mtp_pd_handoff(
        payload["mtp_pd_handoff_contract"])
    full_result_sha256 = _canonical_json_sha256(payload)
    return {
        "schema_version": 1,
        "kind": result_kind,
        "persistence": {
            "projection": projection,
            "request_level_content": "omitted",
            "full_in_memory_result_sha256": full_result_sha256,
        },
        "version": payload["version"],
        "backend": payload["backend"],
        "deployment_id": payload["deployment_id"],
        "hardware_ids": payload["hardware_ids"],
        "model_compute_dtype": payload["model_compute_dtype"],
        "model_config_provenance": payload["model_config_provenance"],
        "deployment": payload["deployment"],
        "deployment_plan": payload["deployment_plan"],
        "routing_policy": payload["routing_policy"],
        "kv_handoff_mode": payload["kv_handoff_mode"],
        "kv_capacity": payload["kv_capacity"],
        "kv_lifecycle_replay": aggregate_kv_lifecycle,
        "launch_policy": payload["launch_policy"],
        "prediction_method": payload["prediction_method"],
        "execution_mode": payload["execution_mode"],
        "execution_cache_stats": payload["execution_cache_stats"],
        "execution_parallelism": payload["execution_parallelism"],
        "serving_timing_mode": payload["serving_timing_mode"],
        "host_timing": payload["host_timing"],
        "performance_timing_status": payload["performance_timing_status"],
        "execution_provenance": payload["execution_provenance"],
        "workload_assumption_binding": (
            payload["workload_assumption_binding"]),
        "main_chain_status": payload["main_chain_status"],
        "execution_artifact": aggregate_artifact,
        "scheduler_progress": payload["scheduler_progress"],
        "scheduler_runtime": payload["scheduler_runtime"],
        "mtp_scenario": payload["mtp_scenario"],
        "mtp_acceptance_outcomes": mtp_outcomes,
        "mtp_pd_handoff_contract": mtp_pd_handoff,
        "arrival_process": payload["arrival_process"],
        "num_steps": payload["num_steps"],
        "num_transfers": payload["num_transfers"],
        "num_kernel_calls": payload["num_kernel_calls"],
        "schedule_end_time_s": payload["schedule_end_time_s"],
        "simulation_wall_time_s": payload["simulation_wall_time_s"],
        "evaluation_wall_time_breakdown_s": (
            payload["evaluation_wall_time_breakdown_s"]),
        "process_peak_rss_bytes": payload["process_peak_rss_bytes"],
        "metrics": aggregate_metrics,
        "request_source": aggregate_request_source,
    }


def _aggregate_mtp_acceptance_outcomes(
    value: object,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("MTP acceptance outcomes must be a dict or null")
    steps = value["steps"]
    if not isinstance(steps, (list, tuple)):
        raise TypeError("MTP acceptance outcome steps must be a sequence")
    totals = {
        "scheduled_request_events": 0,
        "target_model_workload_tokens": 0,
        "target_verification_tokens": 0,
        "proposer_tokens": 0,
        "proposer_workload_tokens": 0,
        "proposed_tokens": 0,
        "verified_tokens": 0,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "emitted_tokens": 0,
    }
    for step in steps:
        requests = step["requests"]
        totals["scheduled_request_events"] += len(requests)
        for request in requests:
            scheduled_tokens = request["input_tokens"] + request["output_tokens"]
            totals["target_model_workload_tokens"] += scheduled_tokens
            totals["target_verification_tokens"] += (
                request["mtp_verified_tokens"])
            totals["proposer_tokens"] += request["mtp_proposer_tokens"]
            if request["mtp_proposer_tokens"]:
                totals["proposer_workload_tokens"] += (
                    scheduled_tokens + request["mtp_proposer_tokens"] - 1)
            totals["proposed_tokens"] += request["mtp_proposed_tokens"]
            totals["verified_tokens"] += request["mtp_verified_tokens"]
            totals["accepted_tokens"] += request["mtp_accepted_tokens"]
            totals["rejected_tokens"] += request["mtp_rejected_tokens"]
            totals["emitted_tokens"] += request["emitted_tokens"]
    if totals["verified_tokens"] != (
            totals["accepted_tokens"] + totals["rejected_tokens"]):
        raise RuntimeError(
            "aggregate MTP accepted/rejected token conservation failed")
    aggregate = {
        "schema_version": value["schema_version"],
        "source_kind": value["source_kind"],
        "source_id": value["source_id"],
        "step_count": len(steps),
        "totals": totals,
        "request_level_outcomes_sha256": _canonical_json_sha256(value),
        "request_level_outcomes": "omitted",
    }
    if value["schema_version"] == 2:
        aggregate["scope"] = value["scope"]
        aggregate["scheduler_pools"] = value["scheduler_pools"]
    elif value["schema_version"] != 1:
        raise ValueError("unknown MTP acceptance outcomes schema version")
    return aggregate


def _aggregate_mtp_pd_handoff(
    value: object,
) -> dict[str, object] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("MTP PD handoff contract must be a dict or null")
    common_fields = {
        "schema_version",
        "contract_id",
        "scheduler_pools",
        "prefill_mtp",
        "decode_admission",
        "kv_handoff_mode",
        "kv_handoff_lanes",
        "decode_preemption_resume",
        "scheduler_reference",
        "decode_mtp",
        "validation_status",
        "requests",
    }
    schema_version = value["schema_version"]
    if schema_version == 1:
        expected_fields = common_fields
    elif schema_version == 2:
        expected_fields = common_fields | {
            "acceptance_replay",
            "ingress_request_stream_sha256",
            "decode_admission_contract_sha256",
            "decode_arrival_time_binding",
        }
    elif schema_version == 3:
        expected_fields = common_fields | {
            "acceptance_assumption",
            "ingress_request_stream_sha256",
            "decode_admission_contract_sha256",
            "decode_arrival_time_binding",
        }
    else:
        raise ValueError("unknown MTP PD handoff schema version")
    if set(value) != expected_fields:
        raise ValueError(
            "MTP PD handoff fields disagree with its schema")
    requests = value["requests"]
    if not isinstance(requests, (list, tuple)):
        raise TypeError("MTP PD handoff requests must be a sequence")
    numeric_fields = (
        "target_output_tokens",
        "prefill_emitted_tokens",
        "decode_emitted_tokens",
        "decode_raw_model_output_tokens",
        "decode_raw_output_steps",
        "decode_mtp_proposed_tokens",
        "decode_mtp_verified_tokens",
        "decode_mtp_accepted_tokens",
        "decode_mtp_rejected_tokens",
        "total_emitted_tokens",
    )
    totals = {
        field: sum(request[field] for request in requests)
        for field in numeric_fields
    }
    if totals["decode_mtp_verified_tokens"] != (
            totals["decode_mtp_accepted_tokens"]
            + totals["decode_mtp_rejected_tokens"]):
        raise RuntimeError("aggregate PD MTP token conservation failed")
    contract = {
        name: value[name]
        for name in expected_fields - {"requests"}
    }
    contract["aggregate_requests"] = {
        "request_count": len(requests),
        "totals": totals,
        "request_level_handoff_sha256": _canonical_json_sha256(value),
        "request_level_handoff": "omitted",
    }
    return contract


def _workload_assumption_binding_manifest(
    request_stream_identity: str,
    runtime_inputs: RuntimeInputs,
    args,
) -> dict[str, object]:
    """Bind independent request, MoE-routing, and MTP policy identities."""

    if (not isinstance(request_stream_identity, str)
            or len(request_stream_identity) != 64
            or any(character not in "0123456789abcdef"
                   for character in request_stream_identity)):
        raise ValueError("workload assumption binding requires request SHA256")
    routing = runtime_inputs.expert_routing.manifest()
    if routing["source_kind"] == "not_applicable":
        routing_identity = None
    elif routing["source_kind"] == "explicit_assumption":
        routing_identity = routing["policy_descriptor_sha256"]
    elif routing["source_kind"] == "profile_observation":
        routing_identity = routing["policy_descriptor_sha256"]
    elif routing["source_kind"] == "replay_observation":
        replay = routing["replay"]
        if not isinstance(replay, dict):
            raise RuntimeError(
                "expert routing replay manifest lost its replay payload")
        routing_identity = replay["canonical_payload_sha256"]
    else:
        raise ValueError(
            "unknown expert routing source kind in workload binding")
    if args.mtp_acceptance_policy == "disabled":
        mtp_identity = None
    elif args.mtp_acceptance_policy == "fixed_count":
        mtp_identity = _canonical_json_sha256({
            "schema_version": 1,
            "acceptance_policy": "fixed_count",
            "num_speculative_tokens": args.mtp_num_speculative_tokens,
            "fixed_accepted_tokens": args.mtp_fixed_accepted_tokens,
            "acceptance_source_id": args.mtp_acceptance_source_id,
        })
    elif args.mtp_acceptance_policy == "maximum_entropy":
        assumption = args.mtp_acceptance_assumption
        if assumption is None:
            raise ValueError(
                "maximum-entropy workload binding requires an assumption")
        mtp_identity = assumption.policy_descriptor_sha256
    elif args.mtp_acceptance_policy == "replay":
        replay = args.mtp_acceptance_replay
        if replay is None:
            raise ValueError("replayed workload binding requires an artifact")
        mtp_identity = replay.source_sha256
    else:
        raise ValueError("unknown MTP policy in workload binding")
    components = {
        "request_stream_sha256": request_stream_identity,
        "expert_routing_policy_sha256": routing_identity,
        "mtp_acceptance_policy_sha256": mtp_identity,
    }
    return {
        "schema_version": 1,
        **components,
        "composite_sha256": _canonical_json_sha256(components),
        "independence_contract": (
            "separate_request_routing_and_acceptance_components_v1"),
    }


def _canonical_json_sha256(value: object) -> str:
    serialized = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _execution_provenance(
    args,
    runtime_inputs: RuntimeInputs,
    hardware_by_id: dict[str, HWSpec],
    timing_status: str,
) -> dict[str, object]:
    if not isinstance(runtime_inputs, RuntimeInputs):
        raise TypeError("execution provenance requires RuntimeInputs")
    if not isinstance(hardware_by_id, dict) or not hardware_by_id:
        raise TypeError(
            "execution provenance requires a non-empty hardware mapping")
    hardware_ids = tuple(sorted(hardware_by_id))
    expected_timing_status = performance_timing_status(
        args.launch_policy,
        hardware_ids,
        uses_development_protobridge=_uses_development_protobridge(args),
    )
    if timing_status != expected_timing_status:
        raise ValueError(
            "execution provenance performance timing status disagrees with "
            "launch policy and loaded hardware")
    for hardware_id in hardware_ids:
        hardware = hardware_by_id[hardware_id]
        if not isinstance(hardware, HWSpec):
            raise TypeError(
                "execution provenance hardware values must be HWSpec")
        if hardware.hardware_id != hardware_id:
            raise ValueError(
                "execution provenance hardware key and HWSpec ID disagree")
    if args.decode_hardware_config is None:
        decode_config = None
    else:
        decode_config = _file_provenance(args.decode_hardware_config)
    hardware_configs = {
        "prefill": _file_provenance(args.hardware_config),
        "decode": decode_config,
        "additional_colocated": tuple(
            _file_provenance(path)
            for path in args.additional_colocated_hardware_config
        ),
    }
    launch_data = _launch_data_provenance(hardware_ids)
    if args.prediction_method in (
            "midlayer", "roofline", "roofline_span_floor"):
        protobridge_profiles = {}
        uses_p2p = (
            args.prefill_pp_size > 1
            or args.deployment in (
                "af_disagg", "pd_disagg", "pd_af_disagg"))
        if uses_p2p:
            if args.deployment in ("pd_disagg", "pd_af_disagg"):
                p2p_link = args.transfer_link
            elif args.deployment == "af_disagg":
                p2p_link = args.activation_transfer_link
            else:
                # Pipeline P2P only: intra-node transfers share the
                # tensor-collective scale-up fabric.
                p2p_link = resolve_registered_profile(
                    development_profile_config_sha256(
                        hardware_ids[0], "all_reduce")).link
            protobridge_profiles["send_recv"] = (
                development_profile_config_sha256(
                    hardware_ids[0], "send_recv", link=p2p_link))
        if args.collective_backend == "analytical_roofline_v1":
            pass
        elif args.collective_backend == "protobridge":
            tensor_parallel_sizes = [args.prefill_tp_size]
            if args.decode_tp_size is not None:
                tensor_parallel_sizes.append(args.decode_tp_size)
            if args.ffn_tp_size is not None:
                tensor_parallel_sizes.append(args.ffn_tp_size)
            if any(size > 1 for size in tensor_parallel_sizes):
                protobridge_profiles.update({
                    "all_reduce": args.tensor_all_reduce_profile_sha256,
                    "all_gather": args.tensor_all_gather_profile_sha256,
                })
        else:
            raise ValueError(
                "unknown collective backend for provenance: "
                f"{args.collective_backend!r}")
        if protobridge_profiles:
            predictor_inputs = {
                "protobridge": {
                    "backend": "protobridge",
                    "source_kind": "development_only",
                    "profile_sha256_by_operation": protobridge_profiles,
                },
            }
        else:
            predictor_inputs = None
    elif args.prediction_method == "pipeweave":
        predictor_inputs = {
            "implementation": _directory_provenance(
                ROOT / "pipeweave",
                allowed_suffixes=(".py",),
            ),
            "configs": _directory_provenance(
                ROOT / "configs" / "baselines" / "pipeweave",
                allowed_suffixes=(".yaml", ".yml", ".json"),
            ),
        }
    else:
        raise ValueError(
            f"unknown prediction method for provenance: "
            f"{args.prediction_method!r}")
    return {
        "schema_version": 3,
        "repository": _repository_provenance(),
        "hardware_configs": hardware_configs,
        "loaded_hardware_ids": hardware_ids,
        "performance_timing_status": timing_status,
        "runtime_inputs": {
            "expert_routing": runtime_inputs.expert_routing.manifest(),
            "attention_kv_layout": runtime_inputs.attention_kv_layout,
            "attention_kv_distribution": (
                runtime_inputs.attention_kv_distribution),
            "graph_execution": runtime_inputs.graph_execution.manifest(),
        },
        "launch_inputs": {
            "launch_policy": args.launch_policy,
            "launch_data": launch_data,
        },
        "microbenchmark_inputs": {
            "runtime_artifact_loaded": False,
            "consumption": "materialized_parameters_in_hardware_config",
        },
        "prediction_method": args.prediction_method,
        "external_predictor_inputs": predictor_inputs,
    }


def _file_provenance(path: str | Path) -> dict[str, object]:
    resolved = Path(path).resolve()
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise RuntimeError(
            f"cannot read provenance input file: {resolved}") from error
    return {
        "path": str(resolved),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _launch_data_provenance(hardware_ids: tuple[str, ...]) -> dict:
    launch_data = {}
    for hardware_id in hardware_ids:
        # Match the explicit projection in select_launch_config.py. Keep the
        # requested hardware as the key and the actual source in the path.
        source_id = "nvidia_a800" if hardware_id == "nvidia_h200" else hardware_id
        launch_data[hardware_id] = _directory_provenance(
            ROOT / "launch_data" / source_id,
            allowed_suffixes=(".json",),
        )
    return launch_data


def _directory_provenance(
    path: str | Path,
    *,
    allowed_suffixes: tuple[str, ...],
) -> dict[str, object]:
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise RuntimeError(
            f"provenance input directory does not exist: {resolved}")
    if (not isinstance(allowed_suffixes, tuple)
            or not allowed_suffixes
            or any(not isinstance(suffix, str) or not suffix
                   for suffix in allowed_suffixes)):
        raise ValueError(
            "provenance allowed_suffixes must be non-empty strings")
    files = tuple(
        candidate
        for candidate in sorted(resolved.rglob("*"))
        if candidate.is_file() and candidate.suffix in allowed_suffixes
    )
    if not files:
        raise RuntimeError(
            f"provenance input directory has no selected files: {resolved}")
    entries = []
    for file_path in files:
        file_manifest = _file_provenance(file_path)
        entries.append({
            "relative_path": file_path.relative_to(resolved).as_posix(),
            "sha256": file_manifest["sha256"],
            "size_bytes": file_manifest["size_bytes"],
        })
    serialized = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "path": str(resolved),
        "tree_sha256": hashlib.sha256(serialized).hexdigest(),
        "file_count": len(entries),
        "files": tuple(entries),
    }


def _repository_provenance() -> dict[str, object]:
    return git_code_identity(ROOT)


def _vllm_step_index(step: BackendStep) -> int:
    if not isinstance(step, BackendStep):
        raise TypeError("vLLM step index requires BackendStep")
    parts = step.schedule.step_id.rsplit(":", 2)
    if len(parts) != 3 or not parts[0] or parts[1] != "vllm":
        raise ValueError(
            "MTP outcome requires canonical <pool>:vllm:<index> step ID")
    try:
        index = int(parts[2])
    except ValueError as error:
        raise ValueError(
            "MTP outcome step index must be an integer") from error
    if str(index) != parts[2] or index < 0:
        raise ValueError(
            "MTP outcome step index must be canonical and non-negative")
    return index


def _mtp_acceptance_outcomes_manifest(args, result, deployment) -> dict | None:
    if args.mtp_acceptance_policy == "disabled":
        return None
    if args.backend == "sglang_mirror":
        # The outcome manifest is the vLLM acceptance-replay schema, whose
        # step identity is a native vLLM Scheduler step.
        return None
    if args.mtp_acceptance_policy in ("fixed_count", "maximum_entropy"):
        source_kind = "explicit_assumption"
    elif args.mtp_acceptance_policy == "replay":
        source_kind = "replay_observation"
    else:
        raise ValueError(
            "unknown MTP acceptance policy for outcomes: "
            f"{args.mtp_acceptance_policy!r}")
    if args.deployment == "colocated":
        scheduler_pools = tuple(sorted(deployment.worker_pools))
        if isinstance(result, StreamingSimulationResult):
            selected_backend_steps = tuple(
                entry.backend_step for entry in result.schedule_trace.entries)
        else:
            selected_backend_steps = tuple(
                simulation_step.backend_step
                for simulation_step in result.steps
            )
        schema_version = 1
        scope = None
    elif args.deployment == "af_disagg":
        scheduler_pools = tuple(sorted(
            pool_name
            for pool_name, pool in deployment.worker_pools.items()
            if pool.role == "attention"
        ))
        if len(scheduler_pools) != 1:
            raise RuntimeError(
                "the validated AF MTP outcome scope requires one Attention "
                "pool")
        selected_backend_steps = tuple(
            simulation_step.backend_step for simulation_step in result.steps)
        if any(
                backend_step.schedule.pool_name
                != scheduler_pools[0]
                for backend_step in selected_backend_steps):
            raise RuntimeError(
                "AF MTP outcomes contain a non-Attention Scheduler step")
        schema_version = 1
        scope = None
    elif args.deployment in ("pd_disagg", "pd_af_disagg"):
        scheduler_pools = tuple(sorted(
            pool_name
            for pool_name, pool in deployment.worker_pools.items()
            if (worker_role_phase(pool.role) == "decode"
                and worker_role_operator_group(pool.role) != "ffn")
        ))
        if len(scheduler_pools) != 1:
            raise RuntimeError(
                "the validated PD MTP outcome scope requires one decode pool")
        if isinstance(result, StreamingSimulationResult):
            selected_backend_steps = tuple(
                entry.backend_step
                for entry in result.schedule_trace.entries
                if entry.backend_step.schedule.pool_name in scheduler_pools
            )
        else:
            selected_backend_steps = tuple(
                simulation_step.backend_step
                for simulation_step in result.steps
                if simulation_step.backend_step.schedule.pool_name
                in scheduler_pools
            )
        schema_version = 2
        scope = "decode_only"
    else:
        raise RuntimeError(
            "enabled MTP outcome reporting does not support deployment "
            f"{args.deployment!r}")
    manifest = {
        "schema_version": schema_version,
        "source_kind": source_kind,
        "source_id": args.mtp_acceptance_source_id,
        "steps": mtp_acceptance_steps_manifest(
            mtp_acceptance_step_from_backend_step(
                _vllm_step_index(backend_step),
                backend_step,
            )
            for backend_step in selected_backend_steps
        ),
    }
    if scope is not None:
        manifest["scope"] = scope
        manifest["scheduler_pools"] = list(scheduler_pools)
    return manifest


def _mtp_pd_handoff_contract(
    args,
    requests: RequestStream,
    result,
    deployment,
    scheduler_runtime: dict[str, dict[str, object]],
) -> dict[str, object] | None:
    if args.mtp_num_speculative_tokens == 0:
        return None
    if (args.deployment not in ("pd_disagg", "pd_af_disagg")
            or args.backend != "vllm"):
        return None
    prefill_pools = tuple(sorted(
        pool_name
        for pool_name, pool in deployment.worker_pools.items()
        if (worker_role_phase(pool.role) == "prefill"
            and worker_role_operator_group(pool.role) != "ffn")
    ))
    decode_pools = tuple(sorted(
        pool_name
        for pool_name, pool in deployment.worker_pools.items()
        if (worker_role_phase(pool.role) == "decode"
            and worker_role_operator_group(pool.role) != "ffn")
    ))
    if len(prefill_pools) != 1 or len(decode_pools) != 1:
        raise RuntimeError(
            "the validated PD MTP handoff contract requires one P and one D")
    prefill_pool = prefill_pools[0]
    decode_pool = decode_pools[0]
    prefill_runtime = scheduler_runtime[prefill_pool]
    decode_runtime = scheduler_runtime[decode_pool]
    if (prefill_runtime["speculative_method"] != "disabled"
            or prefill_runtime["mtp_num_speculative_tokens"] != 0
            or prefill_runtime["mtp_acceptance_policy"] != "disabled"):
        raise RuntimeError(
            "PD MTP requires the prefill Scheduler to disable MTP")
    if (decode_runtime["speculative_method"] != "mtp"
            or decode_runtime["mtp_num_speculative_tokens"]
            != args.mtp_num_speculative_tokens
            or decode_runtime["mtp_acceptance_source_id"]
            != args.mtp_acceptance_source_id):
        raise RuntimeError(
            "PD MTP decode Scheduler runtime disagrees with the requested "
            "scenario")
    if args.mtp_acceptance_policy == "fixed_count":
        if (decode_runtime["mtp_acceptance_policy"] != "fixed_count"
                or decode_runtime["mtp_fixed_accepted_tokens"]
                != args.mtp_fixed_accepted_tokens
                or decode_runtime["mtp_acceptance_replay_path"] is not None
                or args.mtp_acceptance_replay is not None):
            raise RuntimeError(
                "PD MTP decode Scheduler runtime disagrees with the requested "
                "fixed-count scenario")
        replay_manifest = None
        assumption_manifest = None
    elif args.mtp_acceptance_policy == "maximum_entropy":
        assumption = args.mtp_acceptance_assumption
        if (assumption is None
                or decode_runtime["mtp_acceptance_policy"]
                != "maximum_entropy"
                or decode_runtime["mtp_fixed_accepted_tokens"] != 0
                or decode_runtime["mtp_acceptance_replay_path"] is not None
                or decode_runtime["mtp_acceptance_assumption_sha256"]
                != assumption.policy_descriptor_sha256
                or decode_runtime["mtp_request_stream_sha256"]
                != assumption.request_stream_sha256):
            raise RuntimeError(
                "PD MTP decode Scheduler runtime disagrees with the requested "
                "maximum-entropy scenario")
        replay_manifest = None
        assumption_manifest = assumption.manifest()
    elif args.mtp_acceptance_policy == "replay":
        replay = args.mtp_acceptance_replay
        if (replay is None
                or decode_runtime["mtp_acceptance_policy"] != "replay"
                or decode_runtime["mtp_fixed_accepted_tokens"] != 0
                or decode_runtime["mtp_acceptance_replay_path"]
                != replay.source_path
                or decode_runtime["mtp_acceptance_replay_sha256"]
                != replay.source_sha256
                or decode_runtime["mtp_acceptance_replay_artifact_id"]
                != replay.artifact_id
                or decode_runtime["mtp_acceptance_replay_step_count"]
                != len(replay.steps)
                or decode_runtime["mtp_request_stream_sha256"]
                != replay.scheduler_context.ingress_identity_sha256):
            raise RuntimeError(
                "PD MTP decode Scheduler runtime disagrees with the requested "
                "replay scenario")
        replay_manifest = mtp_acceptance_replay_manifest(replay)
        assumption_manifest = None
    else:
        raise RuntimeError(
            "PD MTP handoff received an unsupported acceptance policy")

    requests_by_id = {
        request.request_id: request for request in requests
    }
    accounting = {
        request.request_id: {
            "prefill_emitted_tokens": 0,
            "decode_emitted_tokens": 0,
            "decode_raw_model_output_tokens": 0,
            "decode_raw_output_steps": 0,
            "decode_mtp_proposed_tokens": 0,
            "decode_mtp_verified_tokens": 0,
            "decode_mtp_accepted_tokens": 0,
            "decode_mtp_rejected_tokens": 0,
        }
        for request in requests
    }
    transfers_by_request = {
        request.request_id: [] for request in requests
    }
    for transfer in result.schedule_trace.transfers:
        if (transfer.payload != "kv"
                or transfer.source_pool_id != prefill_pool
                or transfer.destination_pool_id != decode_pool):
            continue
        for request_id in transfer.request_ids:
            if request_id not in transfers_by_request:
                raise RuntimeError(
                    f"PD MTP transferred unknown request {request_id!r}")
            transfers_by_request[request_id].append(transfer)
    decode_entry_starts_by_request = {
        request.request_id: [] for request in requests
    }
    for entry in result.schedule_trace.entries:
        if entry.backend_step.schedule.pool_name != decode_pool:
            continue
        for scheduled in entry.backend_step.schedule.requests:
            request_id = scheduled.request_id
            if request_id not in decode_entry_starts_by_request:
                raise RuntimeError(
                    f"PD MTP decode trace contains unknown request "
                    f"{request_id!r}")
            decode_entry_starts_by_request[request_id].append(
                (entry.start_time_s, entry.end_time_s))
    first_decode_seen: set[str] = set()
    # Decode-pool preemption frees a resident request's KV; the request later
    # resumes by recomputing its prefix locally in the decode pool (the
    # prefill-side KV was released after the handoff, so no re-transfer is
    # possible).  Those resume rows are prefill-phase work in the decode pool
    # and are exempt from the decode-only row contract below; first-admission
    # semantics still apply to every request's initial decode row.
    decode_preempted_ids: set[str] = set()
    if isinstance(result, StreamingSimulationResult):
        # Streaming execution consumes and releases the heavy SimulationStep
        # IR, but the canonical schedule trace retains each BackendStep.  The
        # PD MTP handoff contract only needs scheduler accounting from that
        # BackendStep, so validate the same contract against the trace rather
        # than requiring materialized per-step execution artifacts.
        backend_steps = tuple(
            entry.backend_step for entry in result.schedule_trace.entries)
    else:
        backend_steps = tuple(
            simulation_step.backend_step for simulation_step in result.steps)
    for backend_step in backend_steps:
        pool_name = backend_step.schedule.pool_name
        if pool_name == prefill_pool:
            for scheduled in backend_step.schedule.requests:
                request_id = scheduled.request_id
                if request_id not in accounting:
                    raise RuntimeError(
                        f"PD MTP scheduled unknown request {request_id!r}")
                if scheduled.mtp_acceptance_source_kind != "disabled":
                    raise RuntimeError(
                        "PD MTP prefill Scheduler produced speculative state "
                        f"for {request_id!r}")
                accounting[request_id]["prefill_emitted_tokens"] += (
                    backend_step.generated_tokens[request_id])
        elif pool_name == decode_pool:
            decode_preempted_ids.update(
                backend_step.schedule.preempted_request_ids)
            for scheduled in backend_step.schedule.requests:
                request_id = scheduled.request_id
                if request_id not in accounting:
                    raise RuntimeError(
                        f"PD MTP scheduled unknown request {request_id!r}")
                emitted = backend_step.generated_tokens[request_id]
                if ((scheduled.phase != "decode" or scheduled.input_tokens != 0)
                        and request_id not in decode_preempted_ids):
                    raise RuntimeError(
                        "PD MTP decode pool scheduled non-decode work for "
                        f"{request_id!r}")
                if request_id not in first_decode_seen:
                    request = requests_by_id[request_id]
                    if (scheduled.computed_tokens != request.input_tokens
                            or scheduled.output_tokens != 1
                            or scheduled.raw_model_output_tokens != 1
                            or emitted != 1
                            or scheduled.trace_prefix_hit_tokens != 0
                            or scheduled.scheduler_prefix_hit_tokens != 0
                            or scheduled.resumed_from_preemption
                            or any((
                                scheduled.mtp_proposed_tokens,
                                scheduled.mtp_verified_tokens,
                                scheduled.mtp_accepted_tokens,
                                scheduled.mtp_rejected_tokens,
                            ))):
                        raise RuntimeError(
                            "PD MTP decode admission must start from full prompt "
                            "KV, one terminal seed, and empty draft state for "
                            f"{request_id!r}")
                    first_decode_seen.add(request_id)
                accounting[request_id]["decode_emitted_tokens"] += emitted
                accounting[request_id][
                    "decode_raw_model_output_tokens"] += (
                        scheduled.raw_model_output_tokens)
                if scheduled.raw_model_output_tokens:
                    accounting[request_id]["decode_raw_output_steps"] += 1
                accounting[request_id]["decode_mtp_proposed_tokens"] += (
                    scheduled.mtp_proposed_tokens)
                accounting[request_id]["decode_mtp_verified_tokens"] += (
                    scheduled.mtp_verified_tokens)
                accounting[request_id]["decode_mtp_accepted_tokens"] += (
                    scheduled.mtp_accepted_tokens)
                accounting[request_id]["decode_mtp_rejected_tokens"] += (
                    scheduled.mtp_rejected_tokens)
        else:
            raise RuntimeError(
                f"PD MTP produced a step for unexpected pool {pool_name!r}")

    request_manifests = []
    for request in requests:
        counts = accounting[request.request_id]
        emitted_tokens = (
            counts["prefill_emitted_tokens"]
            + counts["decode_emitted_tokens"])
        if emitted_tokens != request.output_tokens:
            raise RuntimeError(
                "PD MTP emitted-token conservation failed for "
                f"{request.request_id!r}: expected={request.output_tokens}, "
                f"actual={emitted_tokens}")
        if (counts["decode_mtp_verified_tokens"]
                != counts["decode_mtp_accepted_tokens"]
                + counts["decode_mtp_rejected_tokens"]):
            raise RuntimeError(
                "PD MTP accepted/rejected token conservation failed for "
                f"{request.request_id!r}")
        if (counts["decode_raw_model_output_tokens"]
                != counts["decode_mtp_accepted_tokens"]
                + counts["decode_raw_output_steps"]):
            raise RuntimeError(
                "PD MTP raw output does not equal accepted draft prefixes plus "
                f"terminal tokens for {request.request_id!r}")
        if request.request_id in first_decode_seen:
            if counts["prefill_emitted_tokens"] != 1:
                raise RuntimeError(
                    "PD MTP decode admission requires one prefill terminal "
                    f"seed for {request.request_id!r}")
            transfers = transfers_by_request[request.request_id]
            expected_transfer_lanes = len(
                deployment.worker_pools[prefill_pool].ranks)
            if len(transfers) != expected_transfer_lanes:
                raise RuntimeError(
                    "PD MTP requires one P-to-D KV transfer per source rank "
                    f"for {request.request_id!r}: expected="
                    f"{expected_transfer_lanes}, actual={len(transfers)}")
            if len({transfer.transfer_id for transfer in transfers}) != len(
                    transfers):
                raise RuntimeError(
                    "PD MTP P-to-D KV transfer IDs must be unique for "
                    f"{request.request_id!r}")
            decode_entry_intervals = decode_entry_starts_by_request[
                request.request_id]
            latest_transfer_end = max(
                transfer.end_time_s for transfer in transfers)
            if (not decode_entry_intervals
                    or min(
                        start_time for start_time, _ in decode_entry_intervals)
                    < latest_transfer_end):
                raise RuntimeError(
                    "PD MTP decode scheduling started before KV transfer "
                    f"completion for {request.request_id!r}: "
                    f"decode_intervals={decode_entry_intervals!r}, "
                    f"latest_transfer_end={latest_transfer_end!r}")
        elif counts["decode_emitted_tokens"] != 0:
            raise RuntimeError(
                "PD MTP decode accounting lacks an admission step for "
                f"{request.request_id!r}")
        elif transfers_by_request[request.request_id]:
            raise RuntimeError(
                "PD MTP transferred a request that never entered decode for "
                f"{request.request_id!r}")
        request_manifests.append({
            "request_id": request.request_id,
            "target_output_tokens": request.output_tokens,
            **counts,
            "total_emitted_tokens": emitted_tokens,
        })
    common_contract = {
        "scheduler_pools": {
            "prefill": prefill_pool,
            "decode": decode_pool,
        },
        "prefill_mtp": "disabled",
        "decode_admission": {
            "computed_tokens": "full_prompt_length",
            "terminal_seed_tokens": 1,
            "initial_draft_tokens": 0,
            "prefix_hit_reinjection": False,
            "requires_completed_kv_transfer": True,
        },
        "kv_handoff_mode": "metadata_only_private_destination_kv",
        "kv_handoff_lanes": "one_request_scoped_transfer_per_prefill_rank",
        "decode_preemption_resume": (
            "local_prefix_recompute_after_handoff_kv_release"),
        "scheduler_reference": {
            "vllm_version": decode_runtime["version"],
            "vllm_source_commit": decode_runtime["source_commit"],
            "vllm_package_manifest_sha256": (
                decode_runtime["package_python_manifest_sha256"]),
        },
    }
    if assumption_manifest is not None:
        common_contract["scheduler_reference"]["alignment_gate"] = (
            "deterministic_maximum_entropy_0_to_L")
        return {
            "schema_version": 3,
            "contract_id": (
                "vllm_0_11_maximum_entropy_decode_fresh_mtp_pd_v1"),
            **common_contract,
            "decode_mtp": "maximum_entropy_fresh_state",
            "acceptance_assumption": assumption_manifest,
            "ingress_request_stream_sha256": (
                assumption_manifest["request_stream_sha256"]),
            "decode_admission_contract_sha256": (
                pd_decode_admission_contract_sha256(requests)),
            "decode_arrival_time_binding": (
                "projected_transfer_completion_not_assumption_identity"),
            "validation_status": {
                "scheduler_control_plane": (
                    "validated_pinned_vllm_reference"),
                "acceptance": "projected_explicit_assumption",
                "kernel_and_transfer_timing": "projected",
            },
            "requests": request_manifests,
        }
    if replay_manifest is None:
        common_contract["scheduler_reference"]["alignment_gate"] = (
            "seeded_fixed_count_0_1_L")
        return {
            "schema_version": 1,
            "contract_id": "vllm_0_11_seeded_decode_fresh_mtp_pd_v1",
            **common_contract,
            "decode_mtp": "fixed_count_fresh_state",
            "validation_status": {
                "scheduler_control_plane": (
                    "validated_pinned_vllm_reference"),
                "acceptance": "projected_explicit_assumption",
                "kernel_and_transfer_timing": "projected",
            },
            "requests": request_manifests,
        }
    common_contract["scheduler_reference"]["alignment_gate"] = (
        "seeded_decode_replay_contract_0_1_L")
    return {
        "schema_version": 2,
        "contract_id": "vllm_0_11_seeded_decode_replay_mtp_pd_v2",
        **common_contract,
        "decode_mtp": "replay_fresh_state",
        "acceptance_replay": replay_manifest,
        "ingress_request_stream_sha256": (
            args.mtp_acceptance_replay.scheduler_context.
            ingress_request_stream_sha256),
        "decode_admission_contract_sha256": (
            args.mtp_acceptance_replay.scheduler_context.
            decode_admission_contract_sha256),
        "decode_arrival_time_binding": (
            "projected_transfer_completion_not_replay_identity"),
        "validation_status": {
            "scheduler_control_plane": "validated_pinned_vllm_reference",
            "acceptance": "replayed_external_observation",
            "kernel_and_transfer_timing": "projected",
        },
        "requests": request_manifests,
    }


def _load_request_source(args) -> tuple[
    RequestStream,
    dict[str, object],
    str | None,
    dict[str, object] | None,
    int,
]:
    if args.workload is not None:
        artifact = load_workload_artifact(args.workload)
        arrival_times, arrival_process = _arrival_schedule(args, artifact)
        requests = RequestStream(
            RequestSpec(
                arrival_time,
                request_id,
                prompt_length,
                artifact.token_controls["max_tokens"],
                0,
            )
            for arrival_time, request_id, prompt_length in zip(
                arrival_times,
                artifact.request_ids,
                artifact.prompt_lengths,
            )
        )
        return requests, arrival_process, artifact.hash, None, 2
    elif args.production_trace is not None:
        trace = load_production_trace(
            args.production_trace,
            _production_trace_reader_config(args),
        )
        _require_expected_sha256(
            "production Trace",
            args.trace_source_sha256,
            trace.manifest.source_sha256,
        )
        if args.trace_replay_mode == "original":
            replay = build_production_trace_replay(
                trace, arrival_rate_scale=None)
        elif args.trace_replay_mode == "arrival_rate_scale":
            replay = build_production_trace_replay(
                trace,
                arrival_rate_scale=args.trace_arrival_rate_scale,
            )
        else:
            raise ValueError(
                f"unknown production Trace replay mode: "
                f"{args.trace_replay_mode!r}")
        replay_manifest = production_trace_replay_manifest(replay)
        request_source = {
            "schema_version": 1,
            "kind": "production_trace",
            "trace": production_trace_manifest(trace),
            "replay": replay_manifest,
            "zero_output_contract": {
                "mode": "pinned_vllm_raw_terminal_tail_truncation_v1",
                "raw_terminal_tokens": 1,
                "emitted_tokens": 0,
                "prompt_workload": "scheduled",
                "ttft_tpot": "not_applicable_null",
            },
        }
        return replay.requests, replay_manifest, None, request_source, 3
    elif args.canonical_requests is not None:
        source_path = Path(args.canonical_requests)
        try:
            source_bytes = source_path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"cannot read canonical requests source: {source_path}") from error
        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        _require_expected_sha256(
            "canonical requests",
            args.canonical_requests_sha256,
            source_sha256,
        )
        requests = parse_canonical_requests_bytes(
            source_bytes,
            source_path=source_path,
        )
        normalized_sha256 = request_stream_sha256(requests)
        arrival_process = {
            "schema_version": 1,
            "kind": "recorded_canonical_arrivals",
            "request_stream_sha256": normalized_sha256,
            "equal_arrival_order": CANONICAL_EQUAL_ARRIVAL_ORDER,
        }
        if args.closed_loop_client_count is not None:
            if (len(requests) % args.closed_loop_client_count) != 0:
                raise ValueError(
                    "closed-loop requires the canonical request count to be "
                    f"a multiple of the client count: {len(requests)} "
                    f"requests, {args.closed_loop_client_count} clients")
            arrival_process = {
                "schema_version": 1,
                "kind": "closed_loop_zero_think_time",
                "request_stream_sha256": normalized_sha256,
                "client_count": args.closed_loop_client_count,
                "requests_per_client": (
                    len(requests) // args.closed_loop_client_count),
                "chain_assignment": "interleaved",
                "think_time_s": 0.0,
                "equal_arrival_order": CANONICAL_EQUAL_ARRIVAL_ORDER,
            }
        request_source = {
            "schema_version": 1,
            "kind": "canonical_requests",
            "input_path": str(source_path),
            "source_path": str(source_path.resolve()),
            "source_sha256": source_sha256,
            "source_size_bytes": len(source_bytes),
            "input_schema": {
                "schema_version": 1,
                "format": "csv",
                "columns": CANONICAL_REQUEST_COLUMNS,
            },
            "row_count": len(requests),
            "request_id_policy": CANONICAL_REQUEST_ID_POLICY,
            "equal_arrival_order": CANONICAL_EQUAL_ARRIVAL_ORDER,
            "normalized_request_stream_sha256": normalized_sha256,
        }
        return requests, arrival_process, None, request_source, 3
    else:
        raise AssertionError("request source arguments were not normalized")


def _production_trace_reader_config(args) -> ProductionTraceReaderConfig:
    success_filter = SuccessfulRequestFilter(
        mode=args.trace_success_filter_mode,
        field_name=args.trace_success_field,
        accepted_value=args.trace_success_value,
        provenance=args.trace_success_provenance,
    )
    return ProductionTraceReaderConfig(
        source_format=args.trace_format,
        arrival_time_field=args.trace_arrival_field,
        input_tokens_field=args.trace_input_tokens_field,
        output_tokens_field=args.trace_output_tokens_field,
        prefix_hit_tokens_field=args.trace_prefix_hit_tokens_field,
        request_id_field=args.trace_request_id_field,
        input_time_unit=args.trace_time_unit,
        successful_request_filter=success_filter,
    )


def _arrival_schedule(args, artifact) -> tuple[tuple[float, ...], dict]:
    if args.arrival_process == "vllm_bench_gamma":
        arrival_times = vllm_bench_arrival_times(
            num_requests=artifact.num_requests,
            request_rate=args.request_rate,
            burstiness=args.arrival_burstiness,
            seed=args.arrival_seed,
        )
        process = vllm_bench_arrival_process(
            request_rate=args.request_rate,
            seed=args.arrival_seed,
        )
    elif args.arrival_process == "fixed_interval":
        interval = 1.0 / args.request_rate
        arrival_times = tuple(
            index * interval for index in range(artifact.num_requests))
        process = {
            "kind": "fixed_interval",
            "request_rate_per_s": float(args.request_rate),
        }
    elif args.arrival_process == "simultaneous":
        arrival_times = (0.0,) * artifact.num_requests
        process = {"kind": "simultaneous"}
    else:
        raise ValueError(
            f"unknown arrival process: {args.arrival_process!r}")
    return arrival_times, process


def _parse_success_value_json(text: str) -> str | int | bool:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "--trace-success-value-json must be a JSON scalar") from error
    if (value is None
            or isinstance(value, float)
            or not isinstance(value, (str, int, bool))):
        raise argparse.ArgumentTypeError(
            "--trace-success-value-json must decode to string, integer, or bool")
    return value


def _host_request_path_profile_paths(
    overrides: Sequence[str],
) -> dict[str, Path]:
    """Resolve which per-request Host request path profile each hardware uses.

    The published profile of every covered hardware is the default.  An
    override replaces exactly one hardware's entry, keyed by the hardware the
    overriding profile itself declares, so a profile can never be bound to a
    hardware it was not measured on.  Overrides exist for admission-sensitivity
    experiments that need a controlled service time; a run that passes none
    resolves exactly the published mapping.
    """

    paths = dict(_REQUEST_PATH_PROFILE_BY_HARDWARE)
    for override in overrides:
        path = Path(override)
        hardware_id = load_host_request_path_profile(path).hardware_id
        if (hardware_id not in _REQUEST_PATH_PROFILE_BY_HARDWARE
                and hardware_id not in _VERSIONED_HOST_PROFILE_HARDWARE):
            raise ValueError(
                "Host request path profile override names uncovered hardware "
                f"{hardware_id!r}")
        paths[hardware_id] = path
    return paths


def _host_runtime_prepare_profile_path(
    hardware_id: str,
    explicit_path: str | None,
) -> Path | None:
    """Resolve the runtime-prepare profile one Host timing run binds.

    Published hardware binds its profile by default; hardware with versioned
    profiles must name one explicitly.  An explicit profile is accepted only
    for versioned hardware, so the published bindings cannot be replaced.
    """

    if explicit_path is not None:
        if hardware_id not in _VERSIONED_HOST_PROFILE_HARDWARE:
            raise ValueError(
                "--host-runtime-prepare-profile is only for hardware with "
                f"versioned Host profiles, not {hardware_id!r}")
        return Path(explicit_path)
    if hardware_id in _VERSIONED_HOST_PROFILE_HARDWARE:
        raise ValueError(
            f"{hardware_id} Host timing requires an explicit "
            "--host-runtime-prepare-profile")
    if hardware_id in _RUNTIME_PREPARE_PROFILE_BY_HARDWARE:
        return _RUNTIME_PREPARE_PROFILE_BY_HARDWARE[hardware_id]
    return None


def _require_expected_sha256(
    label: str,
    expected: str | None,
    actual: str,
) -> None:
    del label, expected, actual


def _validate_expected_sha256_args(parser, args) -> None:
    bindings = (
        ("--trace-source-sha256", args.trace_source_sha256,
         "--production-trace", args.production_trace),
        ("--canonical-requests-sha256", args.canonical_requests_sha256,
         "--canonical-requests", args.canonical_requests),
        ("--model-config-sha256", args.model_config_sha256,
         "--model", args.model),
        ("--hardware-config-sha256", args.hardware_config_sha256,
         "--hardware-config", args.hardware_config),
        ("--slo-config-sha256", args.slo_config_sha256,
         "--slo-config", args.slo_config),
        ("--interactivity-tgs-config-sha256",
         args.interactivity_tgs_config_sha256,
         "--interactivity-tgs-config", args.interactivity_tgs_config),
    )
    for option, value, source_option, source_value in bindings:
        if value is None:
            continue
        if (not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef"
                       for character in value)):
            parser.error(f"{option} must be a lowercase SHA256")
        if source_value is None:
            parser.error(f"{option} requires {source_option}")


def _validate_request_source_args(parser, args) -> None:
    if args.workload is not None:
        missing = [
            option for option, value in (
                ("--request-rate", args.request_rate),
                ("--arrival-process", args.arrival_process),
            )
            if value is None
        ]
        if missing:
            parser.error("--workload requires: " + ", ".join(missing))
        trace_only = [
            option for option, value in (
                ("--trace-format", args.trace_format),
                ("--trace-time-unit", args.trace_time_unit),
                ("--trace-arrival-field", args.trace_arrival_field),
                ("--trace-input-tokens-field", args.trace_input_tokens_field),
                ("--trace-output-tokens-field", args.trace_output_tokens_field),
                ("--trace-prefix-hit-tokens-field",
                 args.trace_prefix_hit_tokens_field),
                ("--trace-request-id-field", args.trace_request_id_field),
                ("--trace-success-filter-mode",
                 args.trace_success_filter_mode),
                ("--trace-success-field", args.trace_success_field),
                ("--trace-success-value-json", args.trace_success_value),
                ("--trace-success-provenance",
                 args.trace_success_provenance),
                ("--trace-replay-mode", args.trace_replay_mode),
                ("--trace-arrival-rate-scale",
                 args.trace_arrival_rate_scale),
                ("--trace-arrival-rate-scales",
                 args.trace_arrival_rate_scales),
                ("--interactivity-tgs-config",
                 args.interactivity_tgs_config),
            )
            if value is not None
        ]
        if trace_only:
            parser.error(
                "--workload does not accept Production Trace options: "
                + ", ".join(trace_only))
        if not math.isfinite(args.request_rate) or args.request_rate <= 0:
            parser.error("--request-rate must be finite and positive")
        if args.arrival_process == "vllm_bench_gamma":
            if args.backend != "vllm":
                parser.error(
                    "--arrival-process vllm_bench_gamma requires --backend vllm")
            if args.arrival_seed is None or args.arrival_seed < 0:
                parser.error(
                    "vllm_bench_gamma requires a non-negative --arrival-seed")
            if args.arrival_burstiness != 1.0:
                parser.error(
                    "the pinned vLLM benchmark contract requires "
                    "--arrival-burstiness 1.0")
        elif args.arrival_process in ("fixed_interval", "simultaneous"):
            if (args.arrival_seed is not None
                    or args.arrival_burstiness is not None):
                parser.error(
                    f"--arrival-process {args.arrival_process} does not accept "
                    "arrival seed or burstiness")
            if (args.backend == "vllm"
                    and args.arrival_process == "fixed_interval"):
                parser.error(
                    "the vLLM backend cannot use a fixed-interval arrival process")
        else:
            raise AssertionError(
                "argparse allowed unknown arrival process: "
                f"{args.arrival_process!r}")
    elif args.production_trace is not None:
        generated_arrival_options = [
            option for option, value in (
                ("--request-rate", args.request_rate),
                ("--arrival-process", args.arrival_process),
                ("--arrival-seed", args.arrival_seed),
                ("--arrival-burstiness", args.arrival_burstiness),
            )
            if value is not None
        ]
        if generated_arrival_options:
            parser.error(
                "--production-trace uses recorded arrivals and does not accept: "
                + ", ".join(generated_arrival_options))
        required = [
            option for option, value in (
                ("--trace-format", args.trace_format),
                ("--trace-time-unit", args.trace_time_unit),
                ("--trace-arrival-field", args.trace_arrival_field),
                ("--trace-input-tokens-field", args.trace_input_tokens_field),
                ("--trace-output-tokens-field", args.trace_output_tokens_field),
                ("--trace-prefix-hit-tokens-field",
                 args.trace_prefix_hit_tokens_field),
                ("--trace-success-filter-mode",
                 args.trace_success_filter_mode),
                ("--trace-success-provenance",
                 args.trace_success_provenance),
                ("--trace-replay-mode", args.trace_replay_mode),
            )
            if value is None or (isinstance(value, str) and not value.strip())
        ]
        if required:
            parser.error(
                "--production-trace requires: " + ", ".join(required))
        if args.backend != "vllm":
            parser.error(
                "--production-trace requires the pinned vllm Scheduler backend")
        if args.trace_success_filter_mode == "upstream_attested":
            if (args.trace_success_field is not None
                    or args.trace_success_value is not None):
                parser.error(
                    "upstream_attested success filtering does not accept "
                    "--trace-success-field or --trace-success-value-json")
        elif args.trace_success_filter_mode == "field_equals":
            filter_missing = []
            if (args.trace_success_field is None
                    or not args.trace_success_field.strip()):
                filter_missing.append("--trace-success-field")
            if args.trace_success_value is None:
                filter_missing.append("--trace-success-value-json")
            if filter_missing:
                parser.error(
                    "field_equals success filtering requires: "
                    + ", ".join(filter_missing))
        else:
            raise AssertionError(
                "argparse allowed unknown Trace success filter: "
                f"{args.trace_success_filter_mode!r}")
        if args.trace_replay_mode == "original":
            if (args.trace_arrival_rate_scale is not None
                    or args.trace_arrival_rate_scales is not None):
                parser.error(
                    "original Trace replay does not accept "
                    "arrival-rate scale options")
            if args.interactivity_tgs_config is not None:
                parser.error(
                    "original Trace replay does not accept "
                    "--interactivity-tgs-config")
        elif args.trace_replay_mode == "arrival_rate_scale":
            scale = args.trace_arrival_rate_scale
            if (scale is None or not math.isfinite(scale) or scale <= 0):
                parser.error(
                    "arrival_rate_scale replay requires a finite positive "
                    "--trace-arrival-rate-scale")
            if args.trace_arrival_rate_scales is not None:
                parser.error(
                    "arrival_rate_scale replay does not accept "
                    "--trace-arrival-rate-scales")
            if args.interactivity_tgs_config is not None:
                parser.error(
                    "arrival_rate_scale replay does not accept "
                    "--interactivity-tgs-config")
        elif args.trace_replay_mode == "arrival_rate_sweep":
            if args.trace_arrival_rate_scale is not None:
                parser.error(
                    "arrival_rate_sweep replay does not accept "
                    "--trace-arrival-rate-scale")
            scales = args.trace_arrival_rate_scales
            if scales is None:
                parser.error(
                    "arrival_rate_sweep replay requires "
                    "--trace-arrival-rate-scales")
            if any(not math.isfinite(scale) or scale <= 0
                   for scale in scales):
                parser.error(
                    "arrival_rate_sweep scales must be finite and positive")
            if len(scales) != len(set(scales)):
                parser.error("arrival_rate_sweep scales must be unique")
            if (args.interactivity_tgs_config is not None
                    and not args.interactivity_tgs_config):
                parser.error(
                    "--interactivity-tgs-config must be a non-empty path")
            if args.mtp_acceptance_replay is not None:
                parser.error(
                    "arrival_rate_sweep cannot reuse one MTP acceptance "
                    "replay across distinct request streams")
        else:
            raise AssertionError(
                "argparse allowed unknown Trace replay mode: "
                f"{args.trace_replay_mode!r}")
    elif args.canonical_requests is not None:
        incompatible_options = [
            option for option, value in (
                ("--request-rate", args.request_rate),
                ("--arrival-process", args.arrival_process),
                ("--arrival-seed", args.arrival_seed),
                ("--arrival-burstiness", args.arrival_burstiness),
                ("--trace-format", args.trace_format),
                ("--trace-time-unit", args.trace_time_unit),
                ("--trace-arrival-field", args.trace_arrival_field),
                ("--trace-input-tokens-field", args.trace_input_tokens_field),
                ("--trace-output-tokens-field", args.trace_output_tokens_field),
                ("--trace-prefix-hit-tokens-field",
                 args.trace_prefix_hit_tokens_field),
                ("--trace-request-id-field", args.trace_request_id_field),
                ("--trace-success-filter-mode",
                 args.trace_success_filter_mode),
                ("--trace-success-field", args.trace_success_field),
                ("--trace-success-value-json", args.trace_success_value),
                ("--trace-success-provenance",
                 args.trace_success_provenance),
                ("--trace-replay-mode", args.trace_replay_mode),
                ("--trace-arrival-rate-scale",
                 args.trace_arrival_rate_scale),
                ("--trace-arrival-rate-scales",
                 args.trace_arrival_rate_scales),
                ("--interactivity-tgs-config",
                 args.interactivity_tgs_config),
            )
            if value is not None
        ]
        if incompatible_options:
            parser.error(
                "--canonical-requests uses recorded canonical arrivals and "
                "does not accept: " + ", ".join(incompatible_options))
    else:
        raise AssertionError("argparse did not select a request source")


_GLM_PD_MTP_MAX_REQUESTS = frozenset((1, 2, 4, 8, 16, 32, 64, 128))


def _load_pd_scope_configs(args):
    """Load the model and both hardware specs a two-hardware PD scope needs.

    The target-scope predicates all read the same three artifacts and treat
    an unreadable one as "this is not that scope" rather than an error, so an
    unreadable config returns None instead of raising.
    """

    try:
        model = load_model_config(args.model)
        prefill_hardware = HWSpec(args.hardware_config)
        decode_hardware = HWSpec(args.decode_hardware_config)
    except (OSError, TypeError, ValueError):
        return None
    return model, prefill_hardware, decode_hardware


def _is_glm_h200_pd_mtp_target_scope(args, target_tp8_scope: bool) -> bool:
    """Return whether the named GLM/H200 PD MTP sweep scope is selected."""

    if (not target_tp8_scope
            and not _is_glm_h200_pd_tp8_ep1_target_scope(args)):
        return False
    return _matches_target_scope(args, _TARGET_SCOPES["glm_h200_pd_mtp"])


def _load_target_scope_config(args):
    """Load the model and single hardware spec a target scope needs.

    Same contract as the two-hardware variant: an unreadable config means the
    scope is not selected, so it returns None rather than raising.
    """

    try:
        model = load_model_config(args.model)
        hardware = HWSpec(args.hardware_config)
    except (OSError, TypeError, ValueError):
        return None
    return model, hardware


def _is_glm_h200_aggregate_target_scope(args) -> bool:
    """Return whether the exact colocated GLM H200 aggregate scope is selected."""

    return _matches_target_scope(args, _TARGET_SCOPES["glm_h200_aggregate"])


def _is_kimi_h200_colocated_tp8_ep8_target_scope(args) -> bool:
    """Return whether the exact colocated Kimi H200 TP8/PP1/EP8 scope is selected.

    This is the standard vLLM DeepSeek-V3-family single-node topology
    (TP8 with EP=TP, PP1) applied to Kimi-K2.6 W4A16 on H200; it is
    distinct from the source-backed PP2/EP8 A800 production scope.
    """

    return _matches_target_scope(args, _TARGET_SCOPES["kimi_h200_colocated"])


def _is_glm_h200_pd_tp8_ep1_target_scope(args) -> bool:
    """Return whether the named GLM/H200 PD TP8/PP1/EP1 sweep scope is selected.

    The EP1 PD transfer contract is the EP8 one byte-for-byte: the
    compressed MLA KV and FP8 index cache are TP-replicated single-head
    caches, and expert parallelism only shards MoE expert weights, so the
    pinned vLLM REPLICATE rank-pair transfer is unchanged.
    """

    return _matches_target_scope(args, _TARGET_SCOPES["glm_h200_pd_tp8_ep1"])



_DSV3_PD_MTP_MAX_REQUESTS = frozenset((1, 2, 4, 8, 16, 32, 64, 128))


def _is_dsv3_h200_aggregate_target_scope(args) -> bool:
    """Return whether the exact colocated DeepSeek-V3 H200 aggregate scope is selected."""

    return _matches_target_scope(args, _TARGET_SCOPES["dsv3_h200_aggregate"])


def _is_dsv3_h200_pd_tp8_ep1_target_scope(args) -> bool:
    """Return whether the named DeepSeek-V3/H200 PD TP8/PP1/EP1 sweep scope is selected.

    The DeepSeek-V3 PD transfer payload is the compressed MLA kv_c||k_pe
    cache only (no DSA index cache): a TP-replicated single-head cache, so
    the pinned vLLM REPLICATE rank-pair transfer is EP-independent and the
    EP1 contract matches the aggregate layout.
    """

    return _matches_target_scope(args, _TARGET_SCOPES["dsv3_h200_pd_tp8_ep1"])


def _is_dsv3_h200_pd_mtp_target_scope(args) -> bool:
    """Return whether the named DeepSeek-V3/H200 PD MTP sweep scope is selected."""

    return _matches_target_scope(args, _TARGET_SCOPES["dsv3_h200_pd_mtp"])


# Every named target scope is the same decision -- "do these arguments
# describe this exact deployment?" -- over a different set of pinned facts.
# Each row declares those facts; `_matches_target_scope` is the one place that
# evaluates them, so a new scope is a row, not another six-branch body.
#
# ``hardware_roles`` is how many of ``hardware_config`` / ``decode_hardware_config``
# the scope pins (and therefore which loader supplies the hardware specs).
# ``None`` means the field is not constrained by this scope.
_TARGET_SCOPES = {
    "glm_h200_aggregate": {
        "deployment": "colocated",
        "hardware_roles": 1,
        "model_type": "glm_moe_dsa",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (8,),
        "prefill_pp_size": (1,),
        "expert_parallel": False,
        "decode_tp_size": None,
        "require_decode_hardware_config": False,
        "reject_additional_colocated_config": True,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
    "kimi_h200_colocated": {
        "deployment": "colocated",
        "hardware_roles": 1,
        "model_type": "kimi_k2",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (8,),
        "prefill_pp_size": (1,),
        "expert_parallel": True,
        "decode_tp_size": None,
        "require_decode_hardware_config": False,
        "reject_additional_colocated_config": True,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
    "dsv3_h200_aggregate": {
        "deployment": "colocated",
        "hardware_roles": 1,
        "model_type": "deepseek_v3",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (4, 8),
        "prefill_pp_size": (1,),
        "expert_parallel": False,
        "decode_tp_size": None,
        "require_decode_hardware_config": False,
        "reject_additional_colocated_config": True,
        # TP8 is the pinned local variant; the TP4 layout is only the named
        # 32-layer configuration.
        "tp8_bypasses_config_sha256": 8,
        "config_sha256": DEEPSEEK_V3_L32_CONFIG_SHA256,
    },
    "glm_h200_pd_tp8_ep1": {
        "deployment": "pd_disagg",
        "hardware_roles": 2,
        "model_type": "glm_moe_dsa",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (8,),
        "prefill_pp_size": (1,),
        "expert_parallel": False,
        "decode_tp_size": (8,),
        "require_decode_hardware_config": True,
        "reject_additional_colocated_config": True,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
    "dsv3_h200_pd_tp8_ep1": {
        "deployment": "pd_disagg",
        "hardware_roles": 2,
        "model_type": "deepseek_v3",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": (8,),
        "prefill_pp_size": (1,),
        "expert_parallel": False,
        "decode_tp_size": (8,),
        "require_decode_hardware_config": True,
        "reject_additional_colocated_config": True,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
    # The MTP sweep accepts whatever the PD scope accepts and only pins the
    # deployment plus the presence of the decode hardware config.
    "glm_h200_pd_mtp": {
        "deployment": "pd_disagg",
        "hardware_roles": 2,
        "model_type": "glm_moe_dsa",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": None,
        "prefill_pp_size": None,
        "expert_parallel": None,
        "decode_tp_size": None,
        "require_decode_hardware_config": True,
        "reject_additional_colocated_config": False,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
    "dsv3_h200_pd_mtp": {
        "deployment": "pd_disagg",
        "hardware_roles": 2,
        "model_type": "deepseek_v3",
        "hardware_id": "nvidia_h200",
        "prefill_tp_size": None,
        "prefill_pp_size": None,
        "expert_parallel": None,
        "decode_tp_size": None,
        "require_decode_hardware_config": True,
        "reject_additional_colocated_config": False,
        "tp8_bypasses_config_sha256": None,
        "config_sha256": None,
    },
}


def _matches_target_scope(args, scope: Mapping) -> bool:
    """Return whether ``args`` describes exactly the deployment ``scope`` pins.

    The argument guards are checked in the same order the hand-written
    predicates used, so an unreadable config is still not consulted when a
    cheap guard already decided the answer.
    """

    if args.deployment != scope["deployment"]:
        return False
    if scope["require_decode_hardware_config"] and (
            args.decode_hardware_config is None):
        return False
    prefill_tp_size = scope["prefill_tp_size"]
    if prefill_tp_size is not None and (
            args.prefill_tp_size not in prefill_tp_size):
        return False
    prefill_pp_size = scope["prefill_pp_size"]
    if prefill_pp_size is not None and (
            args.prefill_pp_size not in prefill_pp_size):
        return False
    decode_tp_size = scope["decode_tp_size"]
    if decode_tp_size is not None and (
            args.decode_tp_size not in decode_tp_size):
        return False
    expert_parallel = scope["expert_parallel"]
    if expert_parallel is not None and (
            args.enable_expert_parallel is not expert_parallel):
        return False
    if scope["reject_additional_colocated_config"] and (
            args.additional_colocated_hardware_config):
        return False
    if scope["hardware_roles"] == 2:
        loaded = _load_pd_scope_configs(args)
        if loaded is None:
            return False
        model, prefill_hardware, decode_hardware = loaded
        if decode_hardware.hardware_id != scope["hardware_id"]:
            return False
    else:
        loaded = _load_target_scope_config(args)
        if loaded is None:
            return False
        model, prefill_hardware = loaded
    if model.model_type != scope["model_type"]:
        return False
    if prefill_hardware.hardware_id != scope["hardware_id"]:
        return False
    tp8_bypasses = scope["tp8_bypasses_config_sha256"]
    if tp8_bypasses is not None and args.prefill_tp_size == tp8_bypasses:
        return True
    config_sha256 = scope["config_sha256"]
    if config_sha256 is None:
        return True
    return model.config_sha256 == config_sha256


def _pd_mtp_allowed_max_requests(args, target_tp8_scope: bool) -> frozenset[int]:
    if _is_glm_h200_pd_mtp_target_scope(args, target_tp8_scope):
        return _GLM_PD_MTP_MAX_REQUESTS
    if _is_dsv3_h200_pd_mtp_target_scope(args):
        return _DSV3_PD_MTP_MAX_REQUESTS
    return frozenset((1,))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workload")
    source.add_argument("--production-trace")
    source.add_argument("--canonical-requests")
    parser.add_argument("--trace-source-sha256")
    parser.add_argument("--canonical-requests-sha256")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-config-sha256")
    parser.add_argument("--hardware-config", required=True)
    parser.add_argument("--hardware-config-sha256")
    parser.add_argument("--decode-hardware-config")
    parser.add_argument(
        "--additional-colocated-hardware-config",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--deployment",
        choices=("colocated", "af_disagg", "pd_disagg", "pd_af_disagg"),
        required=True,
    )
    parser.add_argument(
        "--backend", choices=("vllm", "conformance", "sglang_mirror"),
        required=True)
    parser.add_argument(
        "--sglang-source-version", choices=SGLANG_SOURCE_VERSIONS,
        default=None)
    parser.add_argument(
        "--sglang-max-total-num-tokens", type=int, default=None)
    parser.add_argument(
        "--sglang-max-total-num-tokens-source", default=None)
    parser.add_argument(
        "--sglang-max-running-requests", type=int, default=None)
    parser.add_argument(
        "--prediction-method",
        choices=("midlayer", "roofline", "roofline_span_floor", "pipeweave"),
        required=True,
    )
    parser.add_argument(
        "--collective-backend",
        choices=("analytical_roofline_v1", "protobridge"),
        default=None,
    )
    parser.add_argument("--tensor-all-reduce-profile-sha256")
    parser.add_argument("--tensor-all-gather-profile-sha256")
    parser.add_argument(
        "--launch-policy", choices=("injected_launch", "analytical_launch"), required=True)
    parser.add_argument(
        "--attention-kv-layout",
        choices=("contiguous", "paged", "paged_varlen", "chunked_prefill", "mla"),
        required=True,
    )
    parser.add_argument(
        "--attention-kv-distribution",
        choices=("uniform", "skew"),
        required=True,
    )
    parser.add_argument(
        "--graph-mode",
        choices=(
            "NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY",
            "FULL_AND_PIECEWISE",
        ),
        default="NONE",
    )
    parser.add_argument(
        "--graph-uniform-decode-query-len", type=int, default=1)
    parser.add_argument("--graph-capture-sizes", type=int, nargs="+")
    parser.add_argument(
        "--graph-piecewise-eager-workload-kinds", nargs="*", default=())
    parser.add_argument("--request-rate", type=float)
    parser.add_argument(
        "--arrival-process",
        choices=("vllm_bench_gamma", "fixed_interval", "simultaneous"),
    )
    parser.add_argument("--arrival-seed", type=int)
    parser.add_argument("--arrival-burstiness", type=float)
    parser.add_argument(
        "--trace-format", choices=("csv", "jsonl", "parquet"))
    parser.add_argument(
        "--trace-time-unit", choices=("s", "ms", "us", "ns"))
    parser.add_argument("--trace-arrival-field")
    parser.add_argument("--trace-input-tokens-field")
    parser.add_argument("--trace-output-tokens-field")
    parser.add_argument("--trace-prefix-hit-tokens-field")
    parser.add_argument("--trace-request-id-field")
    parser.add_argument(
        "--trace-success-filter-mode",
        choices=("upstream_attested", "field_equals"),
    )
    parser.add_argument("--trace-success-field")
    parser.add_argument(
        "--trace-success-value-json",
        dest="trace_success_value",
        type=_parse_success_value_json,
    )
    parser.add_argument("--trace-success-provenance")
    parser.add_argument(
        "--trace-replay-mode",
        choices=("original", "arrival_rate_scale", "arrival_rate_sweep"),
    )
    parser.add_argument("--trace-arrival-rate-scale", type=float)
    parser.add_argument(
        "--trace-arrival-rate-scales",
        nargs="+",
        type=float,
    )
    parser.add_argument("--max-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=1,
        help="scheduler sequence capacity",
    )
    parser.add_argument(
        "--closed-loop-client-count",
        type=int,
        default=None,
        help=("zero-think-time closed-loop clients; the canonical request "
              "stream is chained interleaved across this many clients and "
              "each client's next request is injected at its previous "
              "request's completion (default: static canonical arrivals)"),
    )
    parser.add_argument("--scheduler-pipeline-depth", type=int, required=True)
    parser.add_argument(
        "--vllm-async-scheduling",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    kv_capacity = parser.add_mutually_exclusive_group()
    kv_capacity.add_argument(
        "--debug-num-kv-blocks",
        dest="num_kv_blocks",
        type=int,
        help=(
            "fixture-only explicit Scheduler capacity; forbidden for "
            "Production Trace and canonical Serving dataset requests"),
    )
    kv_capacity.add_argument("--runtime-memory-policy")
    parser.add_argument("--kv-block-size-tokens", type=int, default=16)
    parser.add_argument("--kv-lifecycle-replay")
    parser.add_argument("--kv-lifecycle-source-sha256")
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--mtp-num-speculative-tokens", type=int, default=0)
    parser.add_argument(
        "--mtp-acceptance-policy",
        choices=("disabled", "fixed_count", "maximum_entropy", "replay"),
        default="disabled",
    )
    parser.add_argument("--mtp-fixed-accepted-tokens", type=int, default=0)
    parser.add_argument(
        "--mtp-acceptance-source-id",
        default="disabled",
    )
    parser.add_argument("--mtp-acceptance-replay")
    parser.add_argument("--mtp-mean-acceptance-rate")
    parser.add_argument("--mtp-acceptance-seed", type=int)
    parser.add_argument(
        "--expert-routing-policy",
        choices=(
            "seeded_assumption",
            "profile_observation",
            "replay_observation",
        ),
    )
    parser.add_argument("--expert-routing-source-id")
    parser.add_argument("--expert-routing-seed", type=int)
    parser.add_argument("--expert-routing-hot-expert-count", type=int)
    parser.add_argument("--expert-routing-hot-expert-weight", type=int)
    parser.add_argument("--expert-routing-replay")
    parser.add_argument("--expert-routing-profile-manifest")
    parser.add_argument(
        "--expert-routing-profile-resolution",
        choices=("layer_mean", "per_moe_layer"),
        default="layer_mean",
        help=(
            "profile routing realization: one histogram per step from the "
            "model-level layer-mean row, or one per MoE layer from its own "
            "row"),
    )
    parser.add_argument("--expert-routing-source-sha256")
    parser.add_argument("--expert-routing-canonical-payload-sha256")
    parser.add_argument(
        "--execution-mode",
        choices=("accelerated", "reference"),
        default="accelerated",
    )
    parser.add_argument(
        "--execution-workers",
        type=int,
        default=0,
        help=(
            "streaming tail worker processes; zero selects a bounded "
            "CPU/memory-aware default"),
    )
    parser.add_argument(
        "--serving-timing-mode",
        choices=(
            "device_only",
            "device_plus_launch",
            "device_plus_launch_plus_scheduler",
        ),
        default="device_only",
    )
    parser.add_argument(
        "--host-launch-profile",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--scheduler-host-profile",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--host-request-path-profile",
        action="append",
        default=[],
        help=(
            "replace the automatically bound per-request Host request path "
            "profile for one hardware (debug-only for published hardware; "
            "required for hardware with versioned Host profiles); repeat per "
            "hardware"),
    )
    parser.add_argument(
        "--host-runtime-prepare-profile",
        help=(
            "runtime-prepare profile of hardware with versioned Host profiles "
            "(required there in Host timing modes; rejected elsewhere)"),
    )
    parser.add_argument(
        "--step-retention",
        choices=("auto", "full", "streaming_aggregate"),
        default="auto",
        help=(
            "retain full per-step IR, consume it as a bounded-memory "
            "aggregate, or select streaming automatically for compatible "
            "external Serving requests"),
    )
    parser.add_argument("--prefill-tp-size", type=int, required=True)
    parser.add_argument("--prefill-pp-size", type=int, default=1)
    parser.add_argument("--pipeline-ranks-per-node", type=int)
    parser.add_argument(
        "--enable-expert-parallel",
        action="store_true",
        help="enable the validated vLLM TP=EP=2 colocated slice",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--ffn-tp-size", type=int)
    parser.add_argument("--decode-tp-size", type=int)
    parser.add_argument("--num-prefill-instances", type=int)
    parser.add_argument("--num-decode-instances", type=int)
    parser.add_argument("--transfer-algorithm", choices=("send_recv",))
    parser.add_argument(
        "--transfer-link", choices=("nvlink", "metaxlink", "hccs",
                                    "infiniband"))
    parser.add_argument("--router-resource-id", default="router_cpu")
    parser.add_argument(
        "--router-dispatch-service-time-s",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--activation-transfer-algorithm", choices=("send_recv",))
    parser.add_argument(
        "--activation-transfer-link",
        choices=("nvlink", "metaxlink", "hccs"),
    )
    parser.add_argument("--slo-config")
    parser.add_argument("--slo-config-sha256")
    parser.add_argument("--interactivity-tgs-config")
    parser.add_argument("--interactivity-tgs-config-sha256")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    capture_sizes = (() if args.graph_capture_sizes is None else
                     tuple(args.graph_capture_sizes))
    try:
        args.graph_execution_config = GraphExecutionConfig(
            args.graph_mode,
            capture_sizes,
            tuple(args.graph_piecewise_eager_workload_kinds),
            args.graph_uniform_decode_query_len,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.serving_timing_mode == "device_only":
        if args.host_launch_profile or args.scheduler_host_profile:
            parser.error(
                "Host timing profiles are forbidden for device_only")
    elif not args.host_launch_profile:
        parser.error(
            "Host timing modes require at least one --host-launch-profile")
    elif (args.serving_timing_mode == "device_plus_launch_plus_scheduler"
          and not args.scheduler_host_profile):
        parser.error(
            "device_plus_launch_plus_scheduler requires at least one "
            "--scheduler-host-profile")
    elif (args.serving_timing_mode != "device_plus_launch_plus_scheduler"
          and args.scheduler_host_profile):
        parser.error(
            "--scheduler-host-profile requires "
            "device_plus_launch_plus_scheduler")
    if (args.serving_timing_mode != "device_plus_launch_plus_scheduler"
            and args.host_request_path_profile):
        parser.error(
            "--host-request-path-profile requires "
            "device_plus_launch_plus_scheduler")
    if (args.serving_timing_mode == "device_only"
            and args.host_runtime_prepare_profile is not None):
        parser.error(
            "--host-runtime-prepare-profile requires a Host timing mode")
    _validate_expected_sha256_args(parser, args)
    if args.kv_lifecycle_replay is None:
        if args.kv_lifecycle_source_sha256 is not None:
            parser.error(
                "--kv-lifecycle-source-sha256 requires "
                "--kv-lifecycle-replay")
        kv_lifecycle_replay = None
    elif (not isinstance(args.kv_lifecycle_replay, str)
          or not args.kv_lifecycle_replay):
        parser.error("--kv-lifecycle-replay must be a non-empty path")
    else:
        if (not isinstance(args.kv_lifecycle_source_sha256, str)
                or len(args.kv_lifecycle_source_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in args.kv_lifecycle_source_sha256
                )):
            parser.error(
                "--kv-lifecycle-replay requires an expected lowercase "
                "--kv-lifecycle-source-sha256")
        try:
            kv_lifecycle_replay = load_kv_lifecycle_replay(
                args.kv_lifecycle_replay)
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
        if (kv_lifecycle_replay.source_sha256
                != args.kv_lifecycle_source_sha256):
            parser.error(
                "KV lifecycle replay source SHA256 disagrees with the "
                "expected immutable identity")
    args.kv_lifecycle_replay = kv_lifecycle_replay
    if args.expert_routing_replay is None:
        expert_routing_replay = None
    elif (not isinstance(args.expert_routing_replay, str)
          or not args.expert_routing_replay):
        parser.error("--expert-routing-replay must be a non-empty path")
    else:
        try:
            expert_routing_replay = load_expert_routing_replay(
                args.expert_routing_replay)
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    for field_name in (
            "expert_routing_source_sha256",
            "expert_routing_canonical_payload_sha256"):
        value = getattr(args, field_name)
        if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef"
                       for character in value)):
            parser.error(
                f"--{field_name.replace('_', '-')} must be a lowercase "
                "SHA256")
    if args.expert_routing_policy is None:
        if any((
                args.expert_routing_source_id is not None,
                args.expert_routing_seed is not None,
                args.expert_routing_hot_expert_count is not None,
                args.expert_routing_hot_expert_weight is not None,
                expert_routing_replay is not None,
                args.expert_routing_profile_manifest is not None,
                args.expert_routing_source_sha256 is not None,
                args.expert_routing_canonical_payload_sha256 is not None,
        )):
            parser.error(
                "expert routing source, seed, and replay require "
                "--expert-routing-policy")
    elif args.expert_routing_policy == "seeded_assumption":
        if (not isinstance(args.expert_routing_source_id, str)
                or not args.expert_routing_source_id):
            parser.error(
                "seeded expert routing requires "
                "--expert-routing-source-id")
        if args.expert_routing_seed is None:
            parser.error(
                "seeded expert routing requires --expert-routing-seed")
        if args.expert_routing_seed < 0:
            parser.error("--expert-routing-seed cannot be negative")
        if ((args.expert_routing_hot_expert_count is None)
                != (args.expert_routing_hot_expert_weight is None)):
            parser.error(
                "skewed expert routing requires both "
                "--expert-routing-hot-expert-count and "
                "--expert-routing-hot-expert-weight")
        if (args.expert_routing_hot_expert_count is not None
                and args.expert_routing_hot_expert_count < 1):
            parser.error(
                "--expert-routing-hot-expert-count must be positive")
        if (args.expert_routing_hot_expert_weight is not None
                and args.expert_routing_hot_expert_weight <= 1):
            parser.error(
                "--expert-routing-hot-expert-weight must be greater than one")
        if expert_routing_replay is not None:
            parser.error(
                "seeded expert routing cannot use "
                "--expert-routing-replay")
        if args.expert_routing_profile_manifest is not None:
            parser.error(
                "seeded expert routing cannot use "
                "--expert-routing-profile-manifest")
        if (args.expert_routing_source_sha256 is not None
                or args.expert_routing_canonical_payload_sha256 is not None):
            parser.error(
                "seeded expert routing cannot use replay SHA256 identities")
    elif args.expert_routing_policy == "profile_observation":
        if (not isinstance(args.expert_routing_source_id, str)
                or not args.expert_routing_source_id):
            parser.error(
                "profile expert routing requires --expert-routing-source-id")
        if (not isinstance(args.expert_routing_profile_manifest, str)
                or not args.expert_routing_profile_manifest):
            parser.error(
                "profile expert routing requires "
                "--expert-routing-profile-manifest")
        if (args.expert_routing_seed is not None
                or args.expert_routing_hot_expert_count is not None
                or args.expert_routing_hot_expert_weight is not None
                or expert_routing_replay is not None
                or args.expert_routing_source_sha256 is not None
                or args.expert_routing_canonical_payload_sha256 is not None):
            parser.error(
                "profile expert routing cannot use seed, hot-expert, replay, "
                "or replay SHA256 inputs")
    elif args.expert_routing_policy == "replay_observation":
        if args.execution_mode != "reference":
            parser.error(
                "replayed expert routing requires --execution-mode reference")
        if expert_routing_replay is None:
            parser.error(
                "replayed expert routing requires "
                "--expert-routing-replay")
        if args.expert_routing_seed is not None:
            parser.error("replayed expert routing cannot use a seed")
        if (args.expert_routing_hot_expert_count is not None
                or args.expert_routing_hot_expert_weight is not None):
            parser.error(
                "replayed expert routing cannot use hot-expert parameters")
        if (args.expert_routing_source_id is not None
                and args.expert_routing_source_id
                != expert_routing_replay.artifact_id):
            parser.error(
                "--expert-routing-source-id must match replay artifact_id")
        args.expert_routing_source_id = expert_routing_replay.artifact_id
        if (args.expert_routing_source_sha256 is None
                or args.expert_routing_canonical_payload_sha256 is None):
            parser.error(
                "replayed expert routing requires expected source and "
                "canonical payload SHA256 identities")
        if args.expert_routing_source_sha256 != (
                expert_routing_replay.source_sha256):
            parser.error(
                "expert routing replay source SHA256 disagrees with the "
                "expected immutable identity")
        if args.expert_routing_canonical_payload_sha256 != (
                expert_routing_replay.canonical_payload_sha256):
            parser.error(
                "expert routing replay canonical payload SHA256 disagrees "
                "with the expected immutable identity")
        if args.expert_routing_profile_manifest is not None:
            parser.error(
                "replayed expert routing cannot use "
                "--expert-routing-profile-manifest")
    else:
        parser.error(
            f"unknown expert routing policy: "
            f"{args.expert_routing_policy!r}")
    if (args.expert_routing_profile_resolution != "layer_mean"
            and args.expert_routing_policy != "profile_observation"):
        parser.error(
            "--expert-routing-profile-resolution requires profile routing")
    args.expert_routing_replay = expert_routing_replay
    if args.mtp_acceptance_replay is None:
        replay = None
    elif (not isinstance(args.mtp_acceptance_replay, str)
          or not args.mtp_acceptance_replay):
        parser.error("--mtp-acceptance-replay must be a non-empty path")
    else:
        try:
            replay = load_mtp_acceptance_replay(
                args.mtp_acceptance_replay)
        except (OSError, TypeError, ValueError) as error:
            parser.error(str(error))
    if args.mtp_acceptance_policy == "replay":
        if replay is None:
            parser.error(
                "--mtp-acceptance-policy replay requires "
                "--mtp-acceptance-replay")
        if args.mtp_acceptance_source_id == "disabled":
            args.mtp_acceptance_source_id = replay.artifact_id
    elif replay is not None:
        parser.error(
            "--mtp-acceptance-replay requires "
            "--mtp-acceptance-policy replay")
    args.mtp_acceptance_replay = replay
    if args.mtp_acceptance_policy == "maximum_entropy":
        if args.mtp_mean_acceptance_rate is None:
            parser.error(
                "maximum-entropy MTP requires "
                "--mtp-mean-acceptance-rate")
        if args.mtp_acceptance_seed is None:
            parser.error(
                "maximum-entropy MTP requires --mtp-acceptance-seed")
        if (not isinstance(args.mtp_acceptance_source_id, str)
                or not args.mtp_acceptance_source_id
                or args.mtp_acceptance_source_id == "disabled"):
            parser.error(
                "maximum-entropy MTP requires a non-disabled "
                "--mtp-acceptance-source-id")
        try:
            mtp_acceptance_assumption = MTPAcceptanceAssumption(
                args.mtp_acceptance_source_id,
                args.mtp_num_speculative_tokens,
                args.mtp_mean_acceptance_rate,
                args.mtp_acceptance_seed,
                "0" * 64,
            )
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    elif args.mtp_acceptance_policy in (
            "disabled", "fixed_count", "replay"):
        if (args.mtp_mean_acceptance_rate is not None
                or args.mtp_acceptance_seed is not None):
            parser.error(
                "--mtp-mean-acceptance-rate and --mtp-acceptance-seed "
                "require --mtp-acceptance-policy maximum_entropy")
        mtp_acceptance_assumption = None
    else:
        raise ValueError(
            "unknown MTP acceptance policy: "
            f"{args.mtp_acceptance_policy!r}")
    args.mtp_acceptance_assumption = mtp_acceptance_assumption
    if args.num_kv_blocks is None and args.runtime_memory_policy is None:
        parser.error(
            "production Forward requires --runtime-memory-policy; narrow "
            "fixtures may use --debug-num-kv-blocks")
    if args.num_kv_blocks is not None and args.num_kv_blocks < 1:
        parser.error("--debug-num-kv-blocks must be positive")
    if (args.runtime_memory_policy is not None
            and not args.runtime_memory_policy):
        parser.error("--runtime-memory-policy must be a non-empty path")
    _validate_request_source_args(parser, args)
    if args.workload is None and args.num_kv_blocks is not None:
        parser.error(
            "Serving dataset execution forbids --debug-num-kv-blocks; "
            "use --runtime-memory-policy")
    if args.prefill_tp_size < 1:
        parser.error("--prefill-tp-size must be positive")
    if args.prefill_pp_size < 1:
        parser.error("--prefill-pp-size must be positive")
    if args.scheduler_pipeline_depth < 1:
        parser.error("--scheduler-pipeline-depth must be positive")
    if args.vllm_async_scheduling and args.scheduler_pipeline_depth < 2:
        parser.error(
            "--vllm-async-scheduling requires scheduler pipeline depth >= 2"
        )
    if (args.pipeline_ranks_per_node is not None
            and args.pipeline_ranks_per_node < 1):
        parser.error("--pipeline-ranks-per-node must be positive")
    supplied_collective_profiles = (
        args.tensor_all_reduce_profile_sha256,
        args.tensor_all_gather_profile_sha256,
    )
    if args.collective_backend is None:
        args.collective_backend = "protobridge"
    if args.collective_backend == "analytical_roofline_v1":
        if any(digest is not None for digest in supplied_collective_profiles):
            parser.error(
                "analytical Roofline collectives do not accept ProtoBridge "
                "profile SHA256 values")
    elif args.collective_backend == "protobridge":
        if ((supplied_collective_profiles[0] is None)
                != (supplied_collective_profiles[1] is None)):
            parser.error(
                "ProtoBridge collectives require both tensor profile SHA256 "
                "arguments when either one is supplied")
    else:
        raise ValueError(
            f"unknown collective backend: {args.collective_backend!r}")
    try:
        RouterConfig(
            args.router_resource_id,
            args.router_dispatch_service_time_s,
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.backend != "vllm" and any((
            args.mtp_num_speculative_tokens,
            args.mtp_fixed_accepted_tokens,
            args.mtp_acceptance_policy != "disabled",
            args.mtp_acceptance_source_id != "disabled",
            args.mtp_acceptance_replay is not None,
    )):
        if args.backend != "sglang_mirror":
            parser.error("MTP options require --backend vllm")
        if (args.mtp_acceptance_policy != "maximum_entropy"
                or args.mtp_acceptance_replay is not None
                or args.mtp_fixed_accepted_tokens):
            parser.error(
                "the SGLang mirror requires maximum-entropy EAGLE acceptance")
    sglang_options = (
        args.sglang_source_version,
        args.sglang_max_total_num_tokens,
        args.sglang_max_total_num_tokens_source,
        args.sglang_max_running_requests,
    )
    if args.backend == "sglang_mirror":
        if any(option is None for option in sglang_options):
            parser.error(
                "--backend sglang_mirror requires --sglang-source-version, "
                "--sglang-max-total-num-tokens, "
                "--sglang-max-total-num-tokens-source and "
                "--sglang-max-running-requests")
        if args.deployment != "colocated":
            parser.error("--backend sglang_mirror requires colocated")
    elif any(option is not None for option in sglang_options):
        parser.error("SGLang options require --backend sglang_mirror")
    if args.kv_lifecycle_replay is not None:
        if args.backend != "vllm":
            parser.error("KV lifecycle replay requires --backend vllm")
        if args.deployment not in ("colocated", "af_disagg"):
            parser.error(
                "KV lifecycle replay currently requires colocated or "
                "standalone AF deployment")
        if args.prefill_pp_size != 1:
            parser.error("KV lifecycle replay currently requires PP=1")
        if args.prefill_tp_size == 1:
            pass
        elif args.prefill_tp_size == 2:
            if not (
                args.deployment == "colocated"
                and args.max_batched_tokens == 8
                and args.max_requests == 1
                and args.num_kv_blocks == 2
                and args.runtime_memory_policy is None
                and args.kv_block_size_tokens == 16
                and args.max_model_len == 128
                and args.mtp_num_speculative_tokens == 0
                and args.mtp_acceptance_policy == "disabled"
                and not args.enable_expert_parallel
                and args.launch_policy == "analytical_launch"
                and args.prediction_method in ("midlayer", "roofline")
            ):
                parser.error(
                    "TP=2 KV lifecycle replay requires the exact colocated "
                    "Qwen3 scope with batch=8, max_requests=1, two 16-token "
                    "HBM blocks, max_model_len=128, MTP/EP disabled, "
                    "analytical launch, and the midlayer or roofline "
                    "predictor")
        else:
            parser.error(
                "KV lifecycle replay currently supports TP=1 or the exact "
                "colocated Qwen3 TP=2 scope")
        if args.additional_colocated_hardware_config:
            parser.error(
                "KV lifecycle replay currently requires one colocated "
                "instance")
        if args.deployment == "af_disagg":
            if (args.kv_lifecycle_replay.context.pool_name != "attention"
                    or args.kv_lifecycle_replay.context.backend_role
                    != "attention"):
                parser.error(
                    "standalone AF KV lifecycle replay must bind the "
                    "Attention Scheduler pool")
            if (args.mtp_acceptance_policy == "fixed_count"
                    and (
                        args.max_batched_tokens != 128
                        or args.max_requests != 1
                        or args.num_kv_blocks != 3
                        or args.kv_block_size_tokens != 16
                        or args.max_model_len != 128
                        or args.mtp_num_speculative_tokens != 2
                        or args.mtp_fixed_accepted_tokens != 1
                    )):
                parser.error(
                    "standalone AF Host KV lifecycle with MTP requires "
                    "batch=128, max_requests=1, three 16-token HBM blocks, "
                    "max_model_len=128, MTP depth two, and fixed acceptance "
                    "one")
        if args.mtp_acceptance_policy == "disabled":
            pass
        elif args.mtp_acceptance_policy == "fixed_count":
            if args.mtp_acceptance_replay is not None:
                parser.error(
                    "fixed-count MTP KV lifecycle replay cannot carry an "
                    "acceptance replay artifact")
            if args.max_requests != 1:
                parser.error(
                    "MTP KV lifecycle replay currently requires "
                    "--max-requests 1")
            if args.enable_expert_parallel:
                parser.error(
                    "MTP KV lifecycle replay currently requires EP disabled")
        else:
            parser.error(
                "KV lifecycle replay supports only disabled or fixed_count "
                "MTP acceptance")
            raise AssertionError("argparse parser.error unexpectedly returned")
        hbm_tier = next(
            tier for tier in args.kv_lifecycle_replay.cache_config.tiers
            if tier.name == "hbm")
        if args.kv_block_size_tokens != hbm_tier.block_size_tokens:
            parser.error(
                "--kv-block-size-tokens disagrees with KV lifecycle replay")
    if args.backend != "vllm" and args.prefill_pp_size != 1:
        parser.error("pipeline parallelism currently requires --backend vllm")
    if not args.enable_chunked_prefill:
        parser.error(
            "the pinned current-vLLM Scheduler contract requires "
            "--enable-chunked-prefill")
    try:
        VLLMSchedulerConfig(
            args.model,
            tensor_parallel_size=args.prefill_tp_size,
            pipeline_parallel_size=args.prefill_pp_size,
            enable_expert_parallel=args.enable_expert_parallel,
            max_model_len=args.max_model_len,
            enable_chunked_prefill=args.enable_chunked_prefill,
            enable_prefix_caching=args.enable_prefix_caching,
            mtp_num_speculative_tokens=args.mtp_num_speculative_tokens,
            mtp_acceptance_policy=args.mtp_acceptance_policy,
            mtp_fixed_accepted_tokens=args.mtp_fixed_accepted_tokens,
            mtp_acceptance_source_id=args.mtp_acceptance_source_id,
            mtp_acceptance_replay=args.mtp_acceptance_replay,
            mtp_acceptance_assumption=args.mtp_acceptance_assumption,
            request_stream_sha256=(
                args.mtp_acceptance_assumption.request_stream_sha256
                if args.mtp_acceptance_assumption is not None
                else None
                if args.mtp_acceptance_replay is None
                else args.mtp_acceptance_replay.scheduler_context.
                ingress_identity_sha256
            ),
            decode_admission_contract_sha256=(
                None
                if args.mtp_acceptance_replay is None
                else args.mtp_acceptance_replay.scheduler_context.
                decode_admission_contract_sha256
            ),
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    target_h200_aggregate_scope = _is_glm_h200_aggregate_target_scope(args)
    target_h200_pd_ep1_scope = _is_glm_h200_pd_tp8_ep1_target_scope(args)
    target_kimi_h200_colocated_scope = (
        _is_kimi_h200_colocated_tp8_ep8_target_scope(args))
    target_dsv3_h200_aggregate_scope = (
        _is_dsv3_h200_aggregate_target_scope(args))
    target_dsv3_h200_pd_ep1_scope = (
        _is_dsv3_h200_pd_tp8_ep1_target_scope(args))
    target_tp8_scope = (
        args.prefill_tp_size == 8
        and ((args.deployment == "colocated"
              and args.prefill_pp_size in (2, 4))
             or (args.deployment == "pd_disagg"
                 and args.prefill_pp_size == 1
                 and args.decode_tp_size == 8))
        and args.enable_expert_parallel
        and not args.additional_colocated_hardware_config
    )
    pd_mtp_allowed_max_requests = _pd_mtp_allowed_max_requests(
        args, target_tp8_scope)
    glm_h200_pd_mtp_scope = _is_glm_h200_pd_mtp_target_scope(
        args, target_tp8_scope)
    dsv3_h200_pd_mtp_scope = _is_dsv3_h200_pd_mtp_target_scope(args)
    if args.backend == "vllm":
        if args.deployment not in (
                "colocated", "af_disagg", "pd_disagg", "pd_af_disagg"):
            parser.error(
                "the current vLLM baseline supports only colocated, "
                "af_disagg, pd_disagg, or pd_af_disagg deployment")
        if (args.prefill_tp_size not in (1, 2)
                and not target_tp8_scope
                and not target_h200_aggregate_scope
                and not target_h200_pd_ep1_scope
                and not target_kimi_h200_colocated_scope
                and not target_dsv3_h200_aggregate_scope
                and not target_dsv3_h200_pd_ep1_scope):
            parser.error(
                "the current vLLM baseline supports TP=1 or the validated "
                "single-colocated TP=2 slice, plus the exact Kimi-K2.6 "
                "TP=EP=8/PP2, GLM-5.2 TP=EP=8/PP4, GLM-5.2 H200 "
                "aggregate TP8/PP1/EP1, GLM-5.2 H200 PD TP8/PP1/EP1, "
                "Kimi-K2.6 H200 colocated TP8/PP1/EP8, DeepSeek-V3 H200 "
                "aggregate TP8/PP1/EP1, or DeepSeek-V3 H200 PD "
                "TP8/PP1/EP1 target scope")
        if args.attention_kv_layout not in ("paged_varlen", "mla"):
            parser.error(
                "the pinned vLLM Scheduler path requires "
                "--attention-kv-layout paged_varlen or mla")
    if args.enable_expert_parallel:
        if args.backend != "vllm":
            parser.error("expert parallelism requires --backend vllm")
        colocated_ep2 = (
            args.deployment == "colocated"
            and args.prefill_tp_size == 2
            and args.prefill_pp_size == 1
        )
        pd_ep2 = (
            args.deployment == "pd_disagg"
            and args.prefill_tp_size == 2
            and args.decode_tp_size == 2
            and args.prefill_pp_size == 1
        )
        standalone_af_ep2 = (
            args.deployment == "af_disagg"
            and args.prefill_tp_size == 2
            and args.ffn_tp_size == 2
            and args.prefill_pp_size == 1
            and not args.additional_colocated_hardware_config
            and args.max_batched_tokens == 8
            and args.max_requests == 1
            and args.num_kv_blocks == 8
            and args.runtime_memory_policy is None
            and args.kv_block_size_tokens == 16
            and args.max_model_len == 128
            and args.mtp_num_speculative_tokens == 0
            and args.kv_lifecycle_replay is None
            and args.expert_routing_policy == "seeded_assumption"
            and args.launch_policy == "analytical_launch"
            and args.prediction_method == "midlayer"
        )
        if (not colocated_ep2 and not pd_ep2 and not standalone_af_ep2
                and not target_tp8_scope
                and not target_kimi_h200_colocated_scope):
            parser.error(
                "expert parallelism currently requires either one colocated "
                "or matching PD TP=EP=2 pools with PP=1, the exact "
                "standalone AF+EP2 "
                "seeded-routing scope, the exact Kimi-K2.6/GLM-5.2 "
                "TP=EP=8 target scope, or the exact Kimi-K2.6 H200 "
                "colocated TP8/PP1/EP8 scope")
    if (args.expert_routing_policy == "replay_observation"
            and not args.enable_expert_parallel):
        parser.error(
            "expert routing replay v1 requires the validated "
            "colocated TP=EP=2 slice")
    if args.deployment == "colocated":
        if (args.mtp_acceptance_replay is not None
                and args.mtp_acceptance_replay.schema_version != 1):
            parser.error(
                "colocated MTP replay requires the colocated replay schema v1")
        if (not args.additional_colocated_hardware_config
                and args.router_dispatch_service_time_s != 0.0):
            parser.error(
                "router dispatch service requires a routed multi-pool "
                "deployment")
        if (args.additional_colocated_hardware_config
                and (args.prefill_tp_size not in (1, 2)
                     or args.prefill_pp_size != 1)):
            parser.error(
                "additional colocated instances require TP=1 or TP=2 and "
                "PP=1")
        if (args.additional_colocated_hardware_config
                and args.mtp_acceptance_replay is not None):
            parser.error(
                "one MTP acceptance replay artifact binds exactly one "
                "colocated Scheduler pool; additional instances require "
                "separate per-pool artifacts")
        if ((args.prefill_tp_size != 1 or args.prefill_pp_size != 1)
                and not target_tp8_scope
                and not target_h200_aggregate_scope
                and not target_dsv3_h200_aggregate_scope
                and any((
                    args.mtp_num_speculative_tokens,
                    args.mtp_fixed_accepted_tokens,
                    args.mtp_acceptance_policy != "disabled",
                    args.mtp_acceptance_source_id != "disabled",
                ))):
            parser.error(
                "the validated single-colocated rank-local parallel paths do "
                "not support MTP outside the fixed GLM-5.2 or DeepSeek-V3 "
                "target scope")
        disaggregated_only = {
            "--decode-hardware-config": args.decode_hardware_config,
            "--decode-tp-size": args.decode_tp_size,
            "--num-prefill-instances": args.num_prefill_instances,
            "--num-decode-instances": args.num_decode_instances,
            "--transfer-algorithm": args.transfer_algorithm,
            "--transfer-link": args.transfer_link,
            "--ffn-tp-size": args.ffn_tp_size,
            "--activation-transfer-algorithm": (
                args.activation_transfer_algorithm),
            "--activation-transfer-link": args.activation_transfer_link,
        }
        supplied = [
            option for option, value in disaggregated_only.items()
            if value is not None
        ]
        if supplied:
            parser.error(
                "colocated deployment does not accept disaggregated options: "
                + ", ".join(supplied))
    elif args.deployment == "af_disagg":
        if args.prefill_pp_size != 1:
            parser.error("af_disagg currently requires --prefill-pp-size 1")
        if args.additional_colocated_hardware_config:
            parser.error(
                "af_disagg does not accept "
                "--additional-colocated-hardware-config")
        pd_only = {
            "--decode-hardware-config": args.decode_hardware_config,
            "--decode-tp-size": args.decode_tp_size,
            "--num-prefill-instances": args.num_prefill_instances,
            "--num-decode-instances": args.num_decode_instances,
            "--transfer-algorithm": args.transfer_algorithm,
            "--transfer-link": args.transfer_link,
        }
        supplied = [
            option for option, value in pd_only.items() if value is not None
        ]
        if supplied:
            parser.error(
                "af_disagg deployment does not accept PD options: "
                + ", ".join(supplied))
        required = {
            "--ffn-tp-size": args.ffn_tp_size,
            "--activation-transfer-algorithm": (
                args.activation_transfer_algorithm),
            "--activation-transfer-link": args.activation_transfer_link,
        }
        missing = [
            option for option, value in required.items() if value is None
        ]
        if missing:
            parser.error(
                "af_disagg deployment requires: " + ", ".join(missing))
        af_tp_sizes = (args.prefill_tp_size, args.ffn_tp_size)
        if af_tp_sizes not in ((1, 1), (2, 2)):
            parser.error(
                "the current AF activation contract requires matching "
                "Attention/FFN TP sizes of either 1 or the validated TP=2 "
                "slice")
        af_mtp_requested = any((
                args.mtp_num_speculative_tokens,
                args.mtp_fixed_accepted_tokens,
                args.mtp_acceptance_policy != "disabled",
                args.mtp_acceptance_source_id != "disabled",
        ))
        if af_tp_sizes == (2, 2) and af_mtp_requested:
            parser.error("the validated standalone AF TP=2 slice disables MTP")
        if af_mtp_requested:
            if args.backend != "vllm":
                parser.error("standalone AF MTP requires --backend vllm")
            if (args.mtp_acceptance_policy != "fixed_count"
                    or args.mtp_acceptance_replay is not None):
                parser.error(
                    "standalone AF MTP currently requires fixed_count "
                    "acceptance without a replay artifact")
            if args.max_requests == 1:
                pass
            elif args.max_requests == 2:
                if (args.production_trace is None
                        or args.trace_replay_mode != "original"
                        or args.max_batched_tokens != 64
                        or not (
                            (args.num_kv_blocks == 6
                             and args.runtime_memory_policy is None)
                            or (args.num_kv_blocks is None
                                and args.runtime_memory_policy is not None)
                        )
                        or args.kv_block_size_tokens != 16
                        or args.max_model_len != 128
                        or args.mtp_num_speculative_tokens != 2
                        or args.mtp_fixed_accepted_tokens != 1):
                    parser.error(
                        "standalone AF MTP max_requests=2 requires the exact "
                        "original Production Trace preemption scope: batch=64, "
                        "six 16-token KV blocks, max_model_len=128, MTP depth "
                        "two, and fixed acceptance one")
            else:
                parser.error(
                    "standalone AF MTP currently requires --max-requests 1 "
                    "or the exact max_requests=2 preemption scope")
            if (args.mtp_num_speculative_tokens != 2
                    or args.mtp_fixed_accepted_tokens not in (0, 1, 2)):
                parser.error(
                    "standalone AF MTP currently requires exactly two "
                    "speculative tokens and fixed acceptance in {0, 1, 2}")
            if (args.launch_policy != "analytical_launch"
                    or args.prediction_method not in ("midlayer", "roofline")):
                parser.error(
                    "standalone AF MTP currently requires analytical launch "
                    "and --prediction-method midlayer or roofline")
            if args.enable_expert_parallel:
                parser.error("standalone AF MTP does not support EP")
            if args.expert_routing_policy != "seeded_assumption":
                parser.error(
                    "standalone AF MTP requires seeded-assumption expert "
                    "routing")
    elif args.deployment == "pd_disagg":
        if args.prefill_pp_size != 1:
            parser.error("pd_disagg currently requires --prefill-pp-size 1")
        if args.additional_colocated_hardware_config:
            parser.error(
                "pd_disagg does not accept "
                "--additional-colocated-hardware-config")
        af_only = {
            "--ffn-tp-size": args.ffn_tp_size,
            "--activation-transfer-algorithm": (
                args.activation_transfer_algorithm),
            "--activation-transfer-link": args.activation_transfer_link,
        }
        supplied = [
            option for option, value in af_only.items() if value is not None
        ]
        if supplied:
            parser.error(
                "pd_disagg deployment does not accept AF options: "
                + ", ".join(supplied))
        required = {
            "--decode-hardware-config": args.decode_hardware_config,
            "--decode-tp-size": args.decode_tp_size,
            "--num-prefill-instances": args.num_prefill_instances,
            "--num-decode-instances": args.num_decode_instances,
            "--transfer-algorithm": args.transfer_algorithm,
            "--transfer-link": args.transfer_link,
        }
        missing = [option for option, value in required.items() if value is None]
        if missing:
            parser.error(
                "pd_disagg deployment requires: " + ", ".join(missing))
        if args.decode_tp_size < 1:
            parser.error("--decode-tp-size must be positive")
        if args.num_prefill_instances < 1:
            parser.error("--num-prefill-instances must be positive")
        if args.num_decode_instances < 1:
            parser.error("--num-decode-instances must be positive")
        if args.backend == "vllm":
            if ((args.prefill_tp_size not in (1, 2)
                 or args.decode_tp_size not in (1, 2))
                    and not target_tp8_scope
                    and not target_h200_pd_ep1_scope
                    and not target_dsv3_h200_pd_ep1_scope):
                parser.error(
                    "the current vLLM PD baseline requires independently "
                    "selected prefill/decode TP sizes in {1, 2}")
            if (Path(args.decode_hardware_config).resolve()
                    != Path(args.hardware_config).resolve()):
                parser.error(
                    "the current vLLM PD baseline requires the same hardware "
                    "config file for P and D")
            if args.mtp_num_speculative_tokens != 0:
                if ((args.prefill_tp_size != 1
                     or args.decode_tp_size != 1)
                        and not target_tp8_scope
                        and not target_h200_pd_ep1_scope
                        and not target_dsv3_h200_pd_ep1_scope):
                    parser.error(
                        "the current vLLM PD MTP baseline requires P/D TP=1")
                if args.mtp_acceptance_policy in (
                        "fixed_count", "maximum_entropy"):
                    pass
                elif args.mtp_acceptance_policy == "replay":
                    replay = args.mtp_acceptance_replay
                    if (replay is None
                            or replay.schema_version != 2
                            or replay.scheduler_context.adapter_contract
                            != PD_DECODE_MTP_REPLAY_ADAPTER_CONTRACT):
                        parser.error(
                            "the current vLLM PD MTP replay baseline requires "
                            "the PD decode replay schema v2 contract")
                else:
                    parser.error(
                        "the current vLLM PD MTP baseline supports only "
                        "fixed_count, maximum_entropy, or replay acceptance")
                    raise AssertionError("argparse.error unexpectedly returned")
                if (args.num_prefill_instances != 1
                        or args.num_decode_instances != 1):
                    parser.error(
                        "the current vLLM PD MTP baseline requires exactly "
                        "one prefill and one decode instance")
                if args.max_requests not in pd_mtp_allowed_max_requests:
                    if glm_h200_pd_mtp_scope:
                        parser.error(
                            "the GLM-5.2 PD MTP sweep scope "
                            "(model_type=glm_moe_dsa, deployment=pd_disagg, "
                            "P/D hardware=nvidia_h200, TP8/PP=1 with EP8 "
                            "or EP1) allows --max-requests in "
                            "{1,2,4,8,16,32,64,128}")
                    if dsv3_h200_pd_mtp_scope:
                        parser.error(
                            "the DeepSeek-V3 PD MTP sweep scope "
                            "(model_type=deepseek_v3, deployment=pd_disagg, "
                            "P/D hardware=nvidia_h200, TP8/PP=1 with EP1) "
                            "allows --max-requests in "
                            "{1,2,4,8,16,32,64,128}")
                    parser.error(
                        "the current vLLM PD MTP baseline requires "
                        "--max-requests 1 outside the named GLM-5.2 and "
                        "DeepSeek-V3 H200 PD sweep scopes")
    elif args.deployment == "pd_af_disagg":
        if args.prefill_pp_size != 1:
            parser.error(
                "pd_af_disagg currently requires --prefill-pp-size 1")
        if args.additional_colocated_hardware_config:
            parser.error(
                "pd_af_disagg does not accept "
                "--additional-colocated-hardware-config")
        required = {
            "--decode-hardware-config": args.decode_hardware_config,
            "--decode-tp-size": args.decode_tp_size,
            "--num-prefill-instances": args.num_prefill_instances,
            "--num-decode-instances": args.num_decode_instances,
            "--transfer-algorithm": args.transfer_algorithm,
            "--transfer-link": args.transfer_link,
            "--ffn-tp-size": args.ffn_tp_size,
            "--activation-transfer-algorithm": (
                args.activation_transfer_algorithm),
            "--activation-transfer-link": args.activation_transfer_link,
        }
        missing = [option for option, value in required.items() if value is None]
        if missing:
            parser.error(
                "pd_af_disagg deployment requires: " + ", ".join(missing))
        if args.num_prefill_instances < 1:
            parser.error("--num-prefill-instances must be positive")
        if args.num_decode_instances < 1:
            parser.error("--num-decode-instances must be positive")
        if (args.prefill_tp_size != 1
                or args.decode_tp_size != 1
                or args.ffn_tp_size != 1):
            parser.error(
                "the current PD+AF composition requires one rank per role")
        if (Path(args.decode_hardware_config).resolve()
                != Path(args.hardware_config).resolve()):
            parser.error(
                "the current PD+AF composition requires every worker pool "
                "to use the same hardware config file")
        pd_af_mtp_requested = any((
                args.mtp_num_speculative_tokens,
                args.mtp_fixed_accepted_tokens,
                args.mtp_acceptance_policy != "disabled",
                args.mtp_acceptance_source_id != "disabled",
        ))
        if pd_af_mtp_requested:
            if args.backend != "vllm":
                parser.error("PD+AF MTP requires --backend vllm")
            if (args.mtp_num_speculative_tokens != 2
                    or args.mtp_acceptance_policy != "fixed_count"
                    or args.mtp_fixed_accepted_tokens not in (0, 1, 2)
                    or args.mtp_acceptance_replay is not None):
                parser.error(
                    "PD+AF MTP currently requires depth two, fixed_count "
                    "acceptance zero, one, or two, and no replay artifact")
            if (args.num_prefill_instances != 1
                    or args.num_decode_instances != 1
                    or args.max_requests != 1
                    or args.max_batched_tokens != 128
                    or not (
                        (args.num_kv_blocks == 8
                         and args.runtime_memory_policy is None)
                        or (args.num_kv_blocks is None
                            and args.runtime_memory_policy is not None)
                    )
                    or args.kv_block_size_tokens != 16
                    or args.max_model_len != 128):
                parser.error(
                    "PD+AF MTP currently requires 1P x 1D, batch=128, "
                    "max_requests=1, eight 16-token KV blocks, and "
                    "max_model_len=128")
            if (args.launch_policy != "analytical_launch"
                    or args.prediction_method != "midlayer"):
                parser.error(
                    "PD+AF MTP currently requires analytical launch and "
                    "--prediction-method midlayer")
            if args.attention_kv_distribution != "uniform":
                parser.error(
                    "PD+AF MTP currently requires "
                    "--attention-kv-distribution uniform")
            if args.enable_expert_parallel:
                parser.error("PD+AF MTP does not support EP")
            if args.expert_routing_policy != "seeded_assumption":
                parser.error(
                    "PD+AF MTP requires seeded-assumption expert routing")
    else:
        raise AssertionError(
            f"argparse allowed unknown deployment: {args.deployment!r}")
    return args


def _run_from_cli_args(argv=None) -> tuple[argparse.Namespace, dict]:
    args = _parse_args(argv)
    return args, run(args)


def run_cli_args(argv=None) -> dict:
    """Run one simulation from CLI arguments and return its full payload."""

    _, payload = _run_from_cli_args(argv)
    return payload


def _ensure_deterministic_hash_seed() -> None:
    if os.environ.get("PYTHONHASHSEED") is not None:
        return
    os.environ["PYTHONHASHSEED"] = "0"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


def main(argv=None) -> int:
    if argv is None:
        _ensure_deterministic_hash_seed()
    args, payload = _run_from_cli_args(argv)
    if args.output:
        sys.stdout.write(Path(args.output).read_text())
    else:
        printed_payload = _persistence_projection(payload)
        print(json.dumps(printed_payload, indent=2, sort_keys=True))
    return 0
