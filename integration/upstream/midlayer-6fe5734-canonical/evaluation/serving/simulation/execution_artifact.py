"""Versioned, auditable Serving execution-IR serialization."""

from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import hashlib
import json
import math
import struct
import time
from collections.abc import Mapping

from src.kernel.kernel_spec import (
    KernelKind,
    KernelPrediction,
    KernelSpec,
    PhysicalTransferConfig,
)
from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.serving.kernel_lowering.execution_timeline import (
    call_reserved_resource_ids,
    call_host_lanes,
    ExecutionInterval,
    ExecutionTimeline,
    HostExecutionInterval,
)
from src.serving.kernel_lowering.kernel_plan import KernelPlan
from src.serving.kernel_lowering.kernel_predict import freeze_contract_value
from src.serving.kernel_lowering.kv_offload_completion import (
    build_kv_cache_transfer_completion,
)
from src.serving.kv_cache.kv_cache_events import KVCacheAction
from src.serving.request_gen.request_stream import RequestStream
from src.serving.schedule.schedule_step import BackendStep, ScheduleStep
from src.serving.schedule.schedule_trace import ScheduleTraceEntry
from src.serving.simulation import (
    SimulationResult,
    SimulationKVCacheTransfer,
    SimulationStep,
    SimulationTransfer,
    StreamingSimulationResult,
)
from src.serving.workload.workload_graph import WorkloadGraph, WorkloadKind


_HASH_METHOD = "canonical_json_sort_keys_compact_utf8_v1"
_STREAMING_HASH_METHOD = "ordered_exact_execution_contract_sha256_chain_v5"
PROJECTED_PERFORMANCE_TIMING_STATUS = "projected_by_midlayer"
DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS = (
    "projected_with_protobridge_development_only"
)
UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS = (
    "unsupported_a800_analytical_launch"
)
_PERFORMANCE_TIMING_STATUSES = {
    PROJECTED_PERFORMANCE_TIMING_STATUS,
    DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS,
    UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS,
}


def _timestamp_subtraction_tolerance_s(
    start_time_s: float,
    end_time_s: float,
) -> float:
    timestamp_scale_s = max(abs(start_time_s), abs(end_time_s))
    return 8.0 * math.ulp(max(1.0, timestamp_scale_s))


def performance_timing_status(
    launch_policy: str,
    hardware_ids: tuple[str, ...] | list[str] | set[str],
    *,
    uses_development_protobridge: bool = False,
) -> str:
    """Classify whether a Serving timing result may carry projected status."""

    if launch_policy not in ("injected_launch", "analytical_launch"):
        raise ValueError(
            f"unknown Serving launch policy: {launch_policy!r}")
    if (not isinstance(hardware_ids, (tuple, list, set))
            or not hardware_ids):
        raise TypeError(
            "Serving performance timing requires non-empty hardware IDs")
    normalized_hardware_ids = tuple(hardware_ids)
    if any(
            not isinstance(hardware_id, str) or not hardware_id
            for hardware_id in normalized_hardware_ids):
        raise TypeError(
            "Serving performance timing hardware IDs must be non-empty "
            "strings")
    if not isinstance(uses_development_protobridge, bool):
        raise TypeError(
            "uses_development_protobridge must be boolean")
    if uses_development_protobridge:
        return DEVELOPMENT_PROTOBRIDGE_TIMING_STATUS
    if (launch_policy == "analytical_launch"
            and "nvidia_a800" in normalized_hardware_ids):
        return UNSUPPORTED_A800_ANALYTICAL_LAUNCH_STATUS
    return PROJECTED_PERFORMANCE_TIMING_STATUS


def _require_performance_timing_status(value: object) -> str:
    if value not in _PERFORMANCE_TIMING_STATUSES:
        raise ValueError(
            f"unsupported Serving performance timing status: {value!r}")
    return value


def structural_lowering_manifest(
    schedule: ScheduleStep,
    graph: WorkloadGraph,
    plan: KernelPlan,
) -> dict[str, object]:
    """Serialize schedule, logical graph, and lowering without timing data."""

    if not isinstance(schedule, ScheduleStep):
        raise TypeError("structural lowering manifest requires ScheduleStep")
    if not isinstance(graph, WorkloadGraph):
        raise TypeError("structural lowering manifest requires WorkloadGraph")
    if not isinstance(plan, KernelPlan):
        raise TypeError("structural lowering manifest requires KernelPlan")
    schedule_payload = _json_value(asdict(schedule), "schedule step")
    schedule_bytes = json.dumps(
        schedule_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    lowering = _lowering_manifest(graph, plan)
    call_ids_by_node = {
        entry["node_id"]: entry["call_ids"]
        for entry in lowering["node_to_call_ids"]
    }
    if any(len(call.source_node_ids) != 1 for call in plan.calls):
        raise ValueError(
            "structural lowering requires one source node per KernelCall")
    edge_mappings = tuple({
        "source_node_id": edge.source,
        "source_call_ids": call_ids_by_node[edge.source],
        "destination_node_id": edge.destination,
        "destination_call_ids": call_ids_by_node[edge.destination],
        "edge_kind": edge.kind,
    } for edge in graph.edges)
    cache_side_effects = tuple({
        "node_id": node.node_id,
        "call_ids": call_ids_by_node[node.node_id],
        "variant": node.variant,
        "shape": _json_value(node.shape, "KV cache update shape"),
        "dtype": node.dtype,
        "attributes": _json_value(
            node.attributes, "KV cache update attributes"),
    } for node in graph.nodes if node.kind is WorkloadKind.KV_UPDATE)
    body = {
        "schema_version": 1,
        "hash_method": _HASH_METHOD,
        "schedule_step": {
            "sha256": hashlib.sha256(schedule_bytes).hexdigest(),
            "payload": schedule_payload,
        },
        "workload_graph": _workload_graph_manifest(graph),
        "lowering": {
            **lowering,
            "typed_edge_mappings": edge_mappings,
            "cache_side_effects": cache_side_effects,
        },
        "kernel_plan": _kernel_plan_manifest(
            plan, include_execution_support=True),
    }
    serialized = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **body,
        "artifact_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def prefix_alignment_manifest(
    requests: RequestStream,
    result: SimulationResult | StreamingSimulationResult,
    kv_block_size_tokens: int,
) -> dict[str, object]:
    """Bind Trace prefix intent to the first canonical Scheduler step."""

    if not isinstance(requests, RequestStream):
        raise TypeError("prefix alignment requires a RequestStream")
    if not isinstance(result, (SimulationResult, StreamingSimulationResult)):
        raise TypeError(
            "prefix alignment requires a full or streaming SimulationResult")
    if (isinstance(kv_block_size_tokens, bool)
            or not isinstance(kv_block_size_tokens, int)):
        raise TypeError("prefix alignment KV block size must be an integer")
    if kv_block_size_tokens <= 0:
        raise ValueError("prefix alignment KV block size must be positive")
    first_schedule_by_id = {}
    for entry in result.schedule_trace.entries:
        for scheduled in entry.backend_step.schedule.requests:
            request_id = scheduled.request_id
            if request_id not in first_schedule_by_id:
                first_schedule_by_id[request_id] = scheduled
    request_ids = {request.request_id for request in requests}
    scheduled_ids = set(first_schedule_by_id)
    if scheduled_ids != request_ids:
        raise ValueError(
            "production Trace prefix alignment does not cover every request: "
            f"missing={sorted(request_ids - scheduled_ids)}, "
            f"unexpected={sorted(scheduled_ids - request_ids)}")
    entries = []
    for request in requests:
        scheduled = first_schedule_by_id[request.request_id]
        if scheduled.trace_prefix_hit_tokens != request.prefix_hit_tokens:
            raise ValueError(
                "Scheduler first-step prefix oracle disagrees with the Trace: "
                f"request={request.request_id!r}, "
                f"trace={request.prefix_hit_tokens}, "
                f"scheduled={scheduled.trace_prefix_hit_tokens}")
        entries.append({
            "request_id": request.request_id,
            "trace_prefix_hit_tokens": scheduled.trace_prefix_hit_tokens,
            "scheduler_prefix_hit_tokens": (
                scheduled.scheduler_prefix_hit_tokens),
            "prefix_scheduler_delta_tokens": (
                scheduled.prefix_scheduler_delta_tokens),
        })
    return {
        "schema_version": 1,
        "owner": "pinned_vllm_scheduler_adapter",
        "vllm_version": "0.11.0",
        "rule": "floor(min(H, P - 1) / B) * B",
        "kv_block_size_tokens": kv_block_size_tokens,
        "entries": entries,
    }


def execution_artifact_manifest(
    result: SimulationResult,
    *,
    performance_timing_status: str,
) -> dict[str, object]:
    """Serialize every execution layer without re-running any predictor."""

    if not isinstance(result, SimulationResult):
        raise TypeError("execution artifact requires a SimulationResult")
    performance_timing_status = _require_performance_timing_status(
        performance_timing_status)
    entries_by_step_id = {}
    schedule_entries = []
    for entry in result.schedule_trace.entries:
        step_id = entry.backend_step.schedule.step_id
        if step_id in entries_by_step_id:
            raise ValueError(
                f"execution artifact found duplicate schedule step: "
                f"{step_id!r}")
        entries_by_step_id[step_id] = entry
        schedule_entries.append({
            "step_id": step_id,
            "pool_name": entry.backend_step.schedule.pool_name,
            "start_time_s": float(entry.start_time_s),
            "end_time_s": float(entry.end_time_s),
        })
    steps = tuple(
        _step_manifest(step, entries_by_step_id)
        for step in result.steps
    )
    simulation_step_ids = tuple(step["step_id"] for step in steps)
    if len(simulation_step_ids) != len(set(simulation_step_ids)):
        raise ValueError(
            "execution artifact found duplicate SimulationStep IDs")
    if (len(entries_by_step_id) != len(simulation_step_ids)
            or set(entries_by_step_id) != set(simulation_step_ids)):
        raise ValueError(
            "execution artifact schedule trace and simulation steps disagree")

    trace_transfers_by_id = {}
    trace_transfers = []
    for transfer in result.schedule_trace.transfers:
        if transfer.transfer_id in trace_transfers_by_id:
            raise ValueError(
                "execution artifact found duplicate trace transfer ID: "
                f"{transfer.transfer_id!r}")
        manifest = _json_value(asdict(transfer), "transfer trace")
        trace_transfers_by_id[transfer.transfer_id] = manifest
        trace_transfers.append(manifest)
    transfers = tuple(
        _transfer_manifest(transfer, trace_transfers_by_id)
        for transfer in result.transfers
    )
    _validate_trace_transfer_coverage(
        result,
        entries_by_step_id,
        trace_transfers_by_id,
        transfers,
    )
    _validate_global_reserved_resource_intervals(steps, transfers)
    modeled_kv_offload = _modeled_kv_offload_manifest(
        result, entries_by_step_id)
    compact_execution = any(
        "logical_call_count" in step["kernel_plan"]
        for step in steps
    )
    if compact_execution:
        artifact_schema_version = 8
    elif modeled_kv_offload["schema_version"] == 4:
        artifact_schema_version = 7
    elif (modeled_kv_offload["schema_version"] == 2
            and modeled_kv_offload["availability"] == "not_enabled"):
        artifact_schema_version = 5
    else:
        raise AssertionError(
            "execution artifact received an unversioned KV offload contract")

    body = {
        "schema_version": artifact_schema_version,
        "hash_method": _HASH_METHOD,
        "performance_timing_status": performance_timing_status,
        "counts": {
            "schedule_steps": len(schedule_entries),
            "simulation_steps": len(steps),
            "simulation_transfers": len(transfers),
            "trace_transfers": len(trace_transfers),
            "modeled_kv_offload_transfers": (
                len(result.kv_offload.transfers)
                if result.kv_offload is not None else 0
            ),
            "kernel_calls": sum(
                (
                    step["kernel_plan"]["logical_call_count"]
                    if "logical_call_count" in step["kernel_plan"]
                    else step["kernel_plan"]["call_count"]
                )
                for step in steps
            ) + sum(
                (
                    transfer["kernel_plan"]["logical_call_count"]
                    if "logical_call_count" in transfer["kernel_plan"]
                    else transfer["kernel_plan"]["call_count"]
                )
                for transfer in transfers
            ),
        },
        "schedule_trace": {
            "route_decision_version": 2,
            "transfer_trace_version": 2,
            "entries": tuple(schedule_entries),
            "route_decisions": tuple(
                _json_value(asdict(decision), "route decision")
                for decision in result.schedule_trace.route_decisions
            ),
            "transfers": tuple(trace_transfers),
        },
        "steps": steps,
        "transfers": transfers,
        "modeled_kv_offload": modeled_kv_offload,
    }
    serialized = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **body,
        "artifact_sha256": hashlib.sha256(serialized).hexdigest(),
    }


class StreamingExecutionArtifactAccumulator:
    """Consume full step IR once and retain only an auditable hash chain."""

    _MAX_CONTRACT_DIGEST_CACHE_ENTRIES = 262_144

    def __init__(self) -> None:
        self._chain = hashlib.sha256()
        self._step_count = 0
        self._kernel_call_count = 0
        self._all_steps_compact = True
        self._seen_step_ids = set()
        self._contract_digest_by_identity = {}
        self._consume_wall_time_s = 0.0

    @staticmethod
    def _transfer_trace_digest(transfers: tuple[object, ...]) -> str:
        """Hash retained scheduler transfer traces without retaining IR."""

        payload = tuple(
            _json_value(asdict(transfer), "streaming transfer trace")
            for transfer in transfers
        )
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    @property
    def consume_wall_time_s(self) -> float:
        return self._consume_wall_time_s

    @staticmethod
    def _contract_digest(value: object) -> bytes:
        frozen = freeze_contract_value(value)
        return hashlib.sha256(repr(frozen).encode("utf-8")).digest()

    def _cached_contract_digest(self, value: object) -> bytes:
        identity = id(value)
        if identity in self._contract_digest_by_identity:
            retained, digest = self._contract_digest_by_identity[identity]
            if retained is not value:
                raise RuntimeError(
                    "streaming artifact contract identity collision")
            return digest
        digest = self._contract_digest(value)
        self._contract_digest_by_identity[identity] = (value, digest)
        return digest

    def _backend_step_contract_digest(self, step: BackendStep) -> bytes:
        """Hash deterministic Scheduler output, excluding measured CPU time."""

        if not isinstance(step, BackendStep):
            raise TypeError("streaming artifact requires a BackendStep")
        return self._contract_digest((
            step.schedule,
            step.kv_cache_events,
            step.generated_tokens,
        ))

    @staticmethod
    def _update_contract_value(
        digest: object,
        value: object,
    ) -> None:
        encoded = repr(freeze_contract_value(value)).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    @staticmethod
    def _update_text_hash(digest: object, value: str) -> None:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)

    @classmethod
    def _update_string_sequence_hash(
        cls,
        digest: object,
        values: tuple[str, ...],
    ) -> None:
        digest.update(len(values).to_bytes(8, "big"))
        for value in values:
            cls._update_text_hash(digest, value)

    def _timeline_contract_digest(
        self,
        timeline: ExecutionTimeline,
    ) -> bytes:
        digest = hashlib.sha256(b"execution_timeline_contract_v1")

        def update_text(value: str) -> None:
            self._update_text_hash(digest, value)

        def update_int(value: int) -> None:
            update_text(str(value))

        def update_float(value: float) -> None:
            digest.update(struct.pack("!d", value))

        update_float(timeline.duration_s)
        update_float(timeline.predicted_device_time_s)
        update_float(timeline.launch_cpu_time_s)
        update_float(timeline.launch_critical_path_s)
        update_float(timeline.host_device_overlap_s)
        update_float(timeline.host_exposed_critical_path_s)
        update_int(timeline.eager_dispatch_count)
        update_int(timeline.graph_replay_count)
        self._update_string_sequence_hash(digest, timeline.launch_kinds)
        update_text(
            "" if timeline.runtime_graph_mode is None
            else timeline.runtime_graph_mode)
        for interval in timeline.intervals:
            update_text(interval.call_id)
            update_text(interval.pool_name)
            update_int(interval.rank)
            update_text(interval.resource)
            update_float(interval.start_time_s)
            update_float(interval.end_time_s)
            update_int(len(interval.reserved_resource_ids))
            for resource_id in interval.reserved_resource_ids:
                update_text(resource_id)
        for repeat in timeline.block_repeats:
            update_text(repeat.block_id)
            update_int(repeat.repeat_count)
            update_float(repeat.first_iteration_start_time_s)
            update_float(repeat.last_iteration_end_time_s)
            for call_id, intervals in repeat.call_intervals:
                update_text(call_id)
                update_int(len(intervals))
                digest.update(b"".join(
                    struct.pack("!dd", start_time_s, end_time_s)
                    for start_time_s, end_time_s in intervals
                ))
            for call_id, interval_sha256 in repeat.call_interval_sha256:
                update_text(call_id)
                update_int(repeat.repeat_count)
                update_text(interval_sha256)
        for activities in (
                timeline.pool_activity,
                timeline.reserved_resource_activity):
            update_int(len(activities))
            for activity in activities:
                update_text(activity.activity_id)
                update_int(activity.interval_count)
                update_int(len(activity.intervals))
                digest.update(b"".join(
                    struct.pack("!dd", start_time_s, end_time_s)
                    for start_time_s, end_time_s in activity.intervals
                ))
        for call_id, count in timeline.call_occurrence_counts:
            update_text(call_id)
            update_int(count)
        for category, duration_s in timeline.critical_path_contributions:
            update_text(category)
            update_float(duration_s)
        update_int(timeline.critical_call_count)
        digest.update(b"\x01" if timeline.critical_path_unique else b"\x00")
        update_int(len(timeline.host_intervals))
        for event in timeline.host_intervals:
            update_text(event.event_id)
            update_text(event.event_kind)
            update_text(event.pool_name)
            update_int(event.rank)
            update_text(event.call_id)
            update_int(event.call_occurrence_index)
            update_int(len(event.covered_call_occurrences))
            for call_id, occurrence_index in event.covered_call_occurrences:
                update_text(call_id)
                update_int(occurrence_index)
            update_int(len(event.device_predecessor_occurrences))
            for call_id, occurrence_index in event.device_predecessor_occurrences:
                update_text(call_id)
                update_int(occurrence_index)
            update_text(
                "" if event.host_predecessor_event_id is None
                else event.host_predecessor_event_id)
            update_text(event.runtime_graph_mode)
            update_text(
                "" if event.graph_segment_id is None
                else event.graph_segment_id)
            update_float(event.start_time_s)
            update_float(event.end_time_s)
            update_float(event.exposed_time_s)
            update_float(event.overlapped_time_s)
        return digest.digest()

    def execution_contract_digest(
        self,
        step: SimulationStep,
    ) -> bytes:
        if (len(self._contract_digest_by_identity)
                >= self._MAX_CONTRACT_DIGEST_CACHE_ENTRIES):
            self._contract_digest_by_identity.clear()
        digest = hashlib.sha256()

        def update(label: str, value: bytes) -> None:
            encoded_label = label.encode("utf-8")
            digest.update(len(encoded_label).to_bytes(4, "big"))
            digest.update(encoded_label)
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)

        update(
            "workload_graph",
            self._contract_digest((
                tuple(
                    (
                        node.node_id,
                        node.kind.value,
                        node.pool_name,
                        node.rank,
                    )
                    for node in step.workload.nodes
                ),
                step.workload.edges,
                step.workload.block_repeats,
                step.workload.logical_node_count,
                step.workload.logical_edge_count,
            )),
        )
        plan_digest = hashlib.sha256()
        for call in step.kernel_plan.calls:
            self._update_text_hash(plan_digest, call.call_id)
            self._update_text_hash(plan_digest, call.pool_name)
            self._update_text_hash(plan_digest, str(call.rank))
            self._update_text_hash(plan_digest, call.resource)
            self._update_string_sequence_hash(
                plan_digest, call.source_node_ids)
            self._update_string_sequence_hash(
                plan_digest, call.dependencies)
            plan_digest.update(
                self._cached_contract_digest(call.kernel_spec))
        plan_digest.update(
            self._contract_digest(step.kernel_plan.block_repeats))
        update("kernel_plan", plan_digest.digest())

        prediction_digest = hashlib.sha256()
        call_ids = tuple(
            call.call_id for call in step.kernel_plan.calls)
        if set(step.predictions) != set(call_ids):
            raise ValueError(
                "streaming artifact predictions must cover KernelPlan exactly")
        for call_id in call_ids:
            self._update_text_hash(prediction_digest, call_id)
            prediction_digest.update(
                self._cached_contract_digest(step.predictions[call_id]))
        update("predictions", prediction_digest.digest())
        update(
            "execution_timeline",
            self._timeline_contract_digest(step.execution_timeline),
        )
        return digest.digest()

    def _step_contract_digest(
        self,
        step: SimulationStep,
        entry: ScheduleTraceEntry,
    ) -> bytes:
        digest = hashlib.sha256()

        def update(label: str, value: bytes) -> None:
            encoded_label = label.encode("utf-8")
            digest.update(len(encoded_label).to_bytes(4, "big"))
            digest.update(encoded_label)
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)

        update(
            "backend_step",
            self._backend_step_contract_digest(step.backend_step),
        )
        update(
            "schedule_interval",
            self._contract_digest((entry.start_time_s, entry.end_time_s)),
        )
        update("execution_contract", self.execution_contract_digest(step))
        return digest.digest()

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def kernel_call_count(self) -> int:
        return self._kernel_call_count

    def consume(
        self,
        step: SimulationStep,
        entry: ScheduleTraceEntry,
    ) -> None:
        if not isinstance(step, SimulationStep):
            raise TypeError(
                "streaming execution artifact requires SimulationStep")
        if not isinstance(entry, ScheduleTraceEntry):
            raise TypeError(
                "streaming execution artifact requires ScheduleTraceEntry")
        step_id = step.backend_step.schedule.step_id
        if step_id in self._seen_step_ids:
            raise ValueError(
                f"streaming execution artifact found duplicate step: "
                f"{step_id!r}")
        if entry.backend_step != step.backend_step:
            raise ValueError(
                f"streaming artifact BackendStep disagrees for {step_id!r}")
        schedule_duration_s = entry.end_time_s - entry.start_time_s
        if not math.isclose(
                schedule_duration_s,
                step.execution_timeline.duration_s,
                rel_tol=1e-12,
                abs_tol=_timestamp_subtraction_tolerance_s(
                    entry.start_time_s, entry.end_time_s)):
            raise ValueError(
                f"streaming artifact timeline duration disagrees for "
                f"{step_id!r}")
        started = time.perf_counter()
        step_digest = self._step_contract_digest(step, entry)
        self._record_prehashed(
            step_id,
            step_digest,
            step.kernel_plan.logical_call_count,
            bool(step.kernel_plan.block_repeats),
        )
        self._consume_wall_time_s += time.perf_counter() - started

    def consume_prehashed(
        self,
        backend_step: object,
        entry: ScheduleTraceEntry,
        execution_contract_digest: bytes,
        logical_kernel_call_count: int,
        compact: bool,
    ) -> None:
        if not isinstance(entry, ScheduleTraceEntry):
            raise TypeError(
                "streaming execution artifact requires ScheduleTraceEntry")
        if entry.backend_step != backend_step:
            raise ValueError("streaming artifact BackendStep disagrees")
        if not isinstance(execution_contract_digest, bytes):
            raise TypeError(
                "streaming execution contract digest must be bytes")
        step_id = backend_step.schedule.step_id
        if step_id in self._seen_step_ids:
            raise ValueError(
                f"streaming execution artifact found duplicate step: "
                f"{step_id!r}")
        started = time.perf_counter()
        digest = hashlib.sha256()

        def update(label: str, value: bytes) -> None:
            encoded_label = label.encode("utf-8")
            digest.update(len(encoded_label).to_bytes(4, "big"))
            digest.update(encoded_label)
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)

        update(
            "backend_step",
            self._backend_step_contract_digest(backend_step),
        )
        update(
            "schedule_interval",
            self._contract_digest((entry.start_time_s, entry.end_time_s)),
        )
        update("execution_contract", execution_contract_digest)
        self._record_prehashed(
            step_id,
            digest.digest(),
            logical_kernel_call_count,
            compact,
        )
        self._consume_wall_time_s += time.perf_counter() - started

    def _record_prehashed(
        self,
        step_id: str,
        step_digest: bytes,
        logical_kernel_call_count: int,
        compact: bool,
    ) -> None:
        if (isinstance(logical_kernel_call_count, bool)
                or not isinstance(logical_kernel_call_count, int)
                or logical_kernel_call_count <= 0):
            raise ValueError(
                "streaming logical kernel call count must be positive")
        if not isinstance(compact, bool):
            raise TypeError("streaming compact marker must be boolean")
        self._chain.update(len(step_digest).to_bytes(8, "big"))
        self._chain.update(step_digest)
        self._seen_step_ids.add(step_id)
        self._step_count += 1
        self._kernel_call_count += logical_kernel_call_count
        self._all_steps_compact = (
            self._all_steps_compact
            and compact
        )

    def manifest(
        self,
        result: StreamingSimulationResult,
        *,
        performance_timing_status: str,
    ) -> dict[str, object]:
        if not isinstance(result, StreamingSimulationResult):
            raise TypeError(
                "streaming execution artifact requires its result type")
        performance_timing_status = _require_performance_timing_status(
            performance_timing_status)
        if (result.num_steps != self._step_count
                or result.num_kernel_calls != self._kernel_call_count):
            raise ValueError(
                "streaming execution artifact counts disagree with result")
        trace_transfer_count = len(result.schedule_trace.transfers)
        if result.num_simulation_transfers != trace_transfer_count:
            raise ValueError(
                "streaming execution artifact transfer counts disagree")
        if not self._all_steps_compact:
            representation = "full_step_ir_consumed_without_retention"
        else:
            representation = "compact_step_ir_consumed_without_retention"
        body = {
            "schema_version": 13,
            "hash_method": _STREAMING_HASH_METHOD,
            "performance_timing_status": performance_timing_status,
            "counts": {
                "schedule_steps": result.num_steps,
                "simulation_steps": result.num_steps,
                "simulation_transfers": result.num_simulation_transfers,
                "trace_transfers": trace_transfer_count,
                "modeled_kv_offload_transfers": 0,
                "kernel_calls": result.num_kernel_calls,
            },
            "streaming_aggregate": {
                "retention": "step_ir_consumed_then_released",
                "representation": representation,
                "ordered_step_execution_contract_chain_sha256": (
                    self._chain.hexdigest()),
                "request_level_artifact": "not_retained",
                "transfer_summary": {
                    "retention": "trace_manifest_digest_and_counts",
                    "trace_transfer_count": trace_transfer_count,
                    "simulation_transfer_count": (
                        result.num_simulation_transfers),
                    "trace_transfer_manifest_sha256": (
                        self._transfer_trace_digest(
                            result.schedule_trace.transfers)),
                },
            },
        }
        serialized = json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return {
            **body,
            "artifact_sha256": hashlib.sha256(serialized).hexdigest(),
        }


def _validate_global_reserved_resource_intervals(
    steps: tuple[dict[str, object], ...],
    transfers: tuple[dict[str, object], ...],
) -> None:
    intervals_by_resource = {}
    owners = tuple(
        (f"step:{step['step_id']}", step["execution_timeline"])
        for step in steps
    ) + tuple(
        (
            f"transfer:{transfer['request_id']}",
            transfer["execution_timeline"],
        )
        for transfer in transfers
    )
    for owner_id, timeline in owners:
        if "reserved_resource_activity" in timeline:
            for activity in timeline["reserved_resource_activity"]:
                resource_id = activity["resource_id"]
                if resource_id not in intervals_by_resource:
                    intervals_by_resource[resource_id] = []
                intervals_by_resource[resource_id].extend(
                    (
                        interval["absolute_start_time_s"],
                        interval["absolute_end_time_s"],
                        owner_id,
                        f"compact-activity:{resource_id}",
                    )
                    for interval in activity["intervals"]
                )
            continue
        for interval in timeline["intervals"]:
            for resource_id in interval["reserved_resource_ids"]:
                if resource_id not in intervals_by_resource:
                    intervals_by_resource[resource_id] = []
                intervals_by_resource[resource_id].append((
                    interval["absolute_start_time_s"],
                    interval["absolute_end_time_s"],
                    owner_id,
                    interval["call_id"],
                ))
    for resource_id, intervals in intervals_by_resource.items():
        ordered = sorted(intervals)
        for previous, current in zip(ordered, ordered[1:]):
            if (current[0] < previous[1]
                    and not math.isclose(
                        current[0], previous[1],
                        rel_tol=1e-12,
                        abs_tol=_timestamp_subtraction_tolerance_s(
                            current[0], previous[1]))):
                raise ValueError(
                    "execution artifact global intervals overlap on reserved "
                    f"resource {resource_id!r}: "
                    f"{previous[2]}:{previous[3]} and "
                    f"{current[2]}:{current[3]}")


def _modeled_kv_offload_manifest(
    result: SimulationResult,
    entries_by_step_id: Mapping[str, ScheduleTraceEntry],
) -> dict[str, object]:
    steps_by_id = {
        step.backend_step.schedule.step_id: step
        for step in result.steps
    }
    transfer_events_by_key = {}
    for step in result.steps:
        step_id = step.backend_step.schedule.step_id
        for event in step.backend_step.kv_cache_events:
            if event.action not in (
                    KVCacheAction.LOAD, KVCacheAction.EVICT):
                continue
            key = (step_id, event.ready_event)
            if key in transfer_events_by_key:
                raise ValueError(
                    "execution artifact found duplicate KV offload intent")
            transfer_events_by_key[key] = event
    if result.kv_offload is None:
        if transfer_events_by_key:
            raise ValueError(
                "execution artifact found KV offload intents without "
                "lifecycle completion state")
        return {
            "schema_version": 2,
            "availability": "not_enabled",
            "intent_source": None,
            "transfers": (),
            "final_tier_state": None,
        }

    records_by_key = {}
    for record in result.kv_offload.transfers:
        if not isinstance(record, SimulationKVCacheTransfer):
            raise TypeError(
                "execution artifact KV offload records must be canonical")
        key = (record.step_id, record.event.ready_event)
        if key in records_by_key:
            raise ValueError(
                "execution artifact found duplicate KV offload completion")
        records_by_key[key] = record
    if set(records_by_key) != set(transfer_events_by_key):
        raise ValueError(
            "execution artifact KV offload intent/completion coverage "
            "disagrees")
    physical_counts = {
        len(record.completion.physical_transfers)
        for record in records_by_key.values()
    }
    if not physical_counts or any(count < 1 for count in physical_counts):
        raise ValueError(
            "execution artifact KV offload requires physical completions")
    multi_rank = physical_counts != {1}
    if multi_rank and physical_counts != {2}:
        raise NotImplementedError(
            "execution artifact v7 currently supports exactly two physical "
            "rank transfers per logical KV event")

    transfers = []
    for key in sorted(records_by_key):
        step_id, ready_event = key
        if step_id not in steps_by_id or step_id not in entries_by_step_id:
            raise ValueError(
                "execution artifact KV offload references unknown step")
        record = records_by_key[key]
        event = transfer_events_by_key[key]
        if record.event != event:
            raise ValueError(
                "execution artifact KV offload intent changed after schedule")
        step = steps_by_id[step_id]
        owner_nodes = tuple(
            node for node in step.workload.nodes
            if (node.kind is WorkloadKind.KV_OFFLOAD
                and node.attributes["ready_event"] == ready_event)
        )
        if (not owner_nodes
                or any(node.rank is None for node in owner_nodes)):
            raise ValueError(
                "execution artifact KV offload owner ranks are incomplete")
        expected_pool_name = step.backend_step.schedule.pool_name
        expected_ranks = tuple(sorted(node.rank for node in owner_nodes))
        expected_completion = build_kv_cache_transfer_completion(
            event,
            step.workload,
            step.kernel_plan,
            step.predictions,
            step.execution_timeline,
            expected_pool_name=expected_pool_name,
            expected_ranks=expected_ranks,
        )
        if record.completion != expected_completion:
            raise ValueError(
                "execution artifact KV offload completion disagrees with "
                "the canonical timeline")
        entry = entries_by_step_id[step_id]
        calls_by_id = {
            call.call_id: call for call in step.kernel_plan.calls
        }
        intervals_by_id = {
            interval.call_id: interval
            for interval in step.execution_timeline.intervals
        }
        physical_transfers = []
        for physical in record.completion.physical_transfers:
            if (physical.call_id not in calls_by_id
                    or physical.call_id not in intervals_by_id):
                raise ValueError(
                    "execution artifact physical KV offload completion lacks "
                    "its canonical call or interval")
            call = calls_by_id[physical.call_id]
            interval = intervals_by_id[physical.call_id]
            reserved_resource_ids = call_reserved_resource_ids(call)
            if (len(reserved_resource_ids) != 1
                    or interval.reserved_resource_ids
                    != reserved_resource_ids):
                raise ValueError(
                    "execution artifact physical KV offload interval lacks "
                    "one canonical rank-local resource")
            physical_transfers.append({
                "resource_id": reserved_resource_ids[0],
                "rank": physical.rank,
                "pool_name": physical.pool_name,
                "node_id": physical.node_id,
                "call_id": physical.call_id,
                "relative_interval": {
                    "start_time_s": physical.start_time_s,
                    "end_time_s": physical.end_time_s,
                },
                "absolute_interval": {
                    "start_time_s": (
                        entry.start_time_s + physical.start_time_s),
                    "end_time_s": (
                        entry.start_time_s + physical.end_time_s),
                },
                "prediction_total_s": step.predictions[
                    physical.call_id].t_total_s,
                "completion": _json_value(
                    asdict(physical),
                    "physical KV offload completion",
                ),
            })
        transfers.append({
            "step_id": step_id,
            "ready_event": ready_event,
            "intent": _json_value(asdict(event), "KV offload intent"),
            "logical_completion": {
                "ready_event": record.completion.ready_event,
                "action": record.completion.action.value,
                "bytes_per_rank": record.completion.bytes,
                "source_tier": record.completion.source_tier,
                "destination_tier": record.completion.destination_tier,
                "link": record.completion.link,
                "physical_transfer_count": len(physical_transfers),
                "relative_interval": {
                    "start_time_s": record.completion.start_time_s,
                    "end_time_s": record.completion.end_time_s,
                },
                "absolute_interval": {
                    "start_time_s": (
                        entry.start_time_s
                        + record.completion.start_time_s),
                    "end_time_s": (
                        entry.start_time_s
                        + record.completion.end_time_s),
                },
            },
            "physical_transfers": tuple(physical_transfers),
        })

    final_tier_state = _json_value(
        result.kv_offload.final_tier_state,
        "KV offload final tier state",
    )
    if final_tier_state["inflight_ready_events"] != ():
        raise ValueError(
            "execution artifact KV offload final state is not quiescent")
    for tier_name in final_tier_state["tiers"]:
        if final_tier_state["tiers"][tier_name]["reserved_bytes"] != 0:
            raise ValueError(
                "execution artifact KV offload final state retains "
                "reserved capacity")
    if result.kv_offload.lifecycle_replay is None:
        intent_source = {
            "kind": "backend_emitted",
            "replay": None,
        }
    else:
        replay_context = result.kv_offload.lifecycle_replay["context"]
        if not isinstance(replay_context, Mapping):
            raise TypeError(
                "KV lifecycle replay context must be a mapping")
        replay_pool_name = replay_context["pool_name"]
        if (not isinstance(replay_pool_name, str)
                or not replay_pool_name):
            raise ValueError(
                "KV lifecycle replay pool name must be non-empty")
        for record in result.kv_offload.transfers:
            step = steps_by_id[record.step_id]
            if step.backend_step.schedule.pool_name != replay_pool_name:
                raise ValueError(
                    "KV offload Scheduler step does not belong to the replay "
                    "owner pool")
            for physical in record.completion.physical_transfers:
                nodes = tuple(
                    node for node in step.workload.nodes
                    if node.node_id == physical.node_id
                )
                calls = tuple(
                    call for call in step.kernel_plan.calls
                    if call.call_id == physical.call_id
                )
                if len(nodes) != 1 or len(calls) != 1:
                    raise ValueError(
                        "physical KV offload completion must resolve to one "
                        "node and call")
                if (nodes[0].pool_name != replay_pool_name
                        or calls[0].pool_name != replay_pool_name):
                    raise ValueError(
                        "KV offload node/call does not belong to the replay "
                        "owner pool")
        intent_source = {
            "kind": "scheduler_event_replay",
            "replay": _json_value(
                result.kv_offload.lifecycle_replay,
                "KV lifecycle replay manifest"),
        }
    return {
        "schema_version": 4,
        "availability": (
            "available_modeled_rank_local"
            if multi_rank else "available_modeled_serial"),
        "mode": (
            "modeled_host_kv_rank_local_lockstep"
            if multi_rank else result.kv_offload.mode),
        "validation_status": result.kv_offload.validation_status,
        "intent_source": intent_source,
        "ledger_semantics": (
            "logical_request_blocks_with_lockstep_per_rank_capacity"
            if multi_rank
            else "logical_request_blocks_with_single_rank_capacity"),
        "physical_transfer_count": sum(
            len(record.completion.physical_transfers)
            for record in records_by_key.values()),
        "scope": {
            "vllm_native_offload": "unsupported",
            "ramulator": "unavailable",
            "host_dram_timing": "unavailable",
            "host_dram_queue": "unavailable",
            "host_link_utilization": (
                "available_in_system_resource_manifest_v5_"
                "host_kv_link_utilization_per_rank"),
            "prefetch": "unsupported_demand_load_only",
            "compute_overlap": "unsupported_serial_dependencies",
            "multi_pool": "unsupported",
            "multi_rank": (
                "validated_colocated_qwen3_tp2_lockstep"
                if multi_rank else "unsupported"),
        },
        "transfers": tuple(transfers),
        "final_tier_state": final_tier_state,
    }


def _step_manifest(
    step: SimulationStep,
    entries_by_step_id: Mapping[str, ScheduleTraceEntry],
) -> dict[str, object]:
    if not isinstance(step, SimulationStep):
        raise TypeError("execution artifact steps must be SimulationStep objects")
    step_id = step.backend_step.schedule.step_id
    if step_id not in entries_by_step_id:
        raise ValueError(
            f"execution artifact step has no schedule entry: {step_id!r}")
    entry = entries_by_step_id[step_id]
    if entry.backend_step != step.backend_step:
        raise ValueError(
            f"execution artifact BackendStep disagrees for {step_id!r}")
    schedule_duration_s = entry.execution_wall_s
    if schedule_duration_s is None:
        raise ValueError(
            f"execution artifact schedule timing is absent for {step_id!r}")
    if not math.isclose(
            schedule_duration_s,
            step.execution_timeline.duration_s,
            rel_tol=1e-12,
            abs_tol=_timestamp_subtraction_tolerance_s(
                entry.start_time_s, entry.end_time_s)):
        raise ValueError(
            f"execution artifact timeline duration disagrees for {step_id!r}")
    manifest = {
        "step_id": step_id,
        "pool_name": step.backend_step.schedule.pool_name,
        "schedule_interval": {
            "start_time_s": float(entry.start_time_s),
            "end_time_s": float(entry.end_time_s),
        },
        "timing": {
            "mode": entry.timing_mode,
            "scheduler_plan_cpu_s": entry.scheduler_plan_cpu_s,
            "scheduler_update_cpu_s": entry.scheduler_update_cpu_s,
            "execution_wall_s": entry.execution_wall_s,
            "step_wall_s": entry.end_time_s - entry.start_time_s,
        },
        "backend_step": _backend_step_manifest(step.backend_step),
    }
    manifest.update(execution_step_components_manifest(
        step.workload,
        step.kernel_plan,
        step.predictions,
        step.execution_timeline,
        origin_time_s=(
            entry.start_time_s
            + entry.scheduler_plan_cpu_s
            + entry.device_queue_wait_s),
    ))
    return manifest


def execution_step_components_manifest(
    workload: WorkloadGraph,
    kernel_plan: KernelPlan,
    predictions: Mapping[str, KernelPrediction],
    execution_timeline: ExecutionTimeline,
    *,
    origin_time_s: float,
) -> dict[str, object]:
    """Serialize the canonical five-layer execution identity for one step."""

    return {
        "workload_graph": _workload_graph_manifest(workload),
        "lowering": _lowering_manifest(workload, kernel_plan),
        "kernel_plan": _kernel_plan_manifest(kernel_plan),
        "predictions": _predictions_manifest(
            kernel_plan, predictions),
        "execution_timeline": _execution_timeline_manifest(
            kernel_plan,
            execution_timeline,
            predictions,
            origin_time_s=origin_time_s,
        ),
    }


def _transfer_manifest(
    transfer: SimulationTransfer,
    trace_transfers_by_id: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(transfer, SimulationTransfer):
        raise TypeError(
            "execution artifact transfers must be SimulationTransfer objects")
    lowering = _lowering_manifest(
        transfer.workload, transfer.kernel_plan)
    workload = _workload_graph_manifest(transfer.workload)
    node_ids = tuple(node["node_id"] for node in workload["nodes"])
    matching_ids = tuple(
        node_id for node_id in node_ids if node_id in trace_transfers_by_id)
    if len(matching_ids) != 1:
        raise ValueError(
            "execution artifact SimulationTransfer must match exactly one "
            f"trace transfer, found {matching_ids}")
    trace_transfer = trace_transfers_by_id[matching_ids[0]]
    trace_duration_s = (
        trace_transfer["end_time_s"] - trace_transfer["start_time_s"])
    if not math.isclose(
            trace_duration_s,
            transfer.execution_timeline.duration_s,
            rel_tol=1e-12,
            abs_tol=_timestamp_subtraction_tolerance_s(
                trace_transfer["start_time_s"],
                trace_transfer["end_time_s"])):
        raise ValueError(
            "execution artifact transfer trace and timeline durations "
            "disagree")
    _validate_pd_transfer(
        transfer,
        trace_transfer,
    )
    return {
        "request_id": transfer.request_id,
        "source_engine_id": transfer.source_engine_id,
        "destination_engine_id": transfer.destination_engine_id,
        "trace_transfer": trace_transfer,
        "workload_graph": workload,
        "lowering": lowering,
        "kernel_plan": _kernel_plan_manifest(transfer.kernel_plan),
        "predictions": _predictions_manifest(
            transfer.kernel_plan, transfer.predictions),
        "execution_timeline": _execution_timeline_manifest(
            transfer.kernel_plan,
            transfer.execution_timeline,
            transfer.predictions,
            origin_time_s=trace_transfer["start_time_s"],
        ),
    }


def _validate_pd_transfer(
    transfer: SimulationTransfer,
    trace_transfer: Mapping[str, object],
) -> None:
    if trace_transfer["payload"] != "kv":
        raise ValueError("SimulationTransfer must match a KV trace transfer")
    if trace_transfer["request_ids"] != (transfer.request_id,):
        raise ValueError(
            "SimulationTransfer request disagrees with its trace transfer")
    if trace_transfer["source_pool_id"] != transfer.source_engine_id:
        raise ValueError(
            "SimulationTransfer source disagrees with its trace transfer")
    if (trace_transfer["destination_pool_id"]
            != transfer.destination_engine_id):
        raise ValueError(
            "SimulationTransfer destination disagrees with its trace transfer")
    nodes = tuple(
        node for node in transfer.workload.nodes
        if node.kind is WorkloadKind.PD_KV_TRANSFER
    )
    if len(nodes) != 1 or len(transfer.workload.nodes) != 1:
        raise ValueError(
            "SimulationTransfer requires exactly one PD KV workload node")
    node = nodes[0]
    if node.node_id != trace_transfer["transfer_id"]:
        raise ValueError(
            "PD KV workload node ID disagrees with trace transfer ID")
    if node.pool_name != transfer.source_engine_id:
        raise ValueError(
            "PD KV workload source disagrees with SimulationTransfer")
    if node.attributes["destination_pool"] != transfer.destination_engine_id:
        raise ValueError(
            "PD KV workload destination disagrees with SimulationTransfer")
    for field, trace_field in (
            ("bytes", "byte_count"),
            ("link", "link"),
            ("resource_id", "resource_id")):
        if node.attributes[field] != trace_transfer[trace_field]:
            raise ValueError(
                f"PD KV workload {field} disagrees with trace transfer")
    calls = tuple(
        call for call in transfer.kernel_plan.calls
        if node.node_id in call.source_node_ids
    )
    if len(calls) != 1 or len(transfer.kernel_plan.calls) != 1:
        raise ValueError(
            "PD KV workload must lower to exactly one communication call")
    call = calls[0]
    spec = call.kernel_spec
    if spec.kind is not KernelKind.COMMUNICATION or spec.transfer is None:
        raise ValueError(
            "PD KV workload must lower to a COMMUNICATION KernelSpec")
    physical = spec.transfer
    expected_physical = _communication_transfer_from_node(node)
    actual_physical = _physical_transfer_manifest(physical)
    if actual_physical != expected_physical:
        raise ValueError(
            "PD KV physical transfer disagrees with workload attributes")
    if call.resource != trace_transfer["resource_id"]:
        raise ValueError(
            "PD KV KernelCall resource disagrees with trace transfer")
    intervals = tuple(
        interval for interval in transfer.execution_timeline.intervals
        if interval.call_id == call.call_id
    )
    if len(intervals) != 1:
        raise ValueError(
            "PD KV communication call requires exactly one interval")
    interval = intervals[0]
    if interval.reserved_resource_ids != trace_transfer[
            "reserved_resource_ids"]:
        raise ValueError(
            "PD KV communication reserved resources disagree with trace")
    absolute_start = trace_transfer["start_time_s"] + interval.start_time_s
    absolute_end = trace_transfer["start_time_s"] + interval.end_time_s
    if (not math.isclose(
            absolute_start, trace_transfer["start_time_s"],
            rel_tol=1e-12,
            abs_tol=_timestamp_subtraction_tolerance_s(
                absolute_start, trace_transfer["start_time_s"]))
            or not math.isclose(
                absolute_end, trace_transfer["end_time_s"],
                rel_tol=1e-12,
                abs_tol=_timestamp_subtraction_tolerance_s(
                    absolute_end, trace_transfer["end_time_s"]))):
        raise ValueError(
            "PD KV communication interval disagrees with trace transfer")


def _validate_trace_transfer_coverage(
    result: SimulationResult,
    entries_by_step_id: Mapping[str, ScheduleTraceEntry],
    trace_transfers_by_id: Mapping[str, Mapping[str, object]],
    transfers: tuple[dict[str, object], ...],
) -> None:
    kv_trace_ids = {
        transfer_id
        for transfer_id, transfer in trace_transfers_by_id.items()
        if transfer["payload"] == "kv"
    }
    simulation_transfer_ids = tuple(
        transfer["trace_transfer"]["transfer_id"]
        for transfer in transfers
    )
    if len(simulation_transfer_ids) != len(set(simulation_transfer_ids)):
        raise ValueError(
            "execution artifact found duplicate SimulationTransfer trace "
            "IDs")
    if (len(kv_trace_ids) != len(simulation_transfer_ids)
            or kv_trace_ids != set(simulation_transfer_ids)):
        raise ValueError(
            "execution artifact KV trace and SimulationTransfer coverage "
            f"disagree: trace={sorted(kv_trace_ids)}, "
            f"simulation={sorted(simulation_transfer_ids)}")
    _validate_pd_scheduler_chain(
        result,
        entries_by_step_id,
        trace_transfers_by_id,
    )

    activation_nodes_by_id = {}
    step_by_activation_id = {}
    relative_interval_by_activation_id = {}
    for step in result.steps:
        _lowering_manifest(step.workload, step.kernel_plan)
        repeated_template_node_ids = {
            template_node_id
            for repeat in step.workload.block_repeats
            for template_node_id, _ in repeat.trace_node_ids
        }
        intervals_by_call_id = {
            interval.call_id: (
                interval.start_time_s, interval.end_time_s)
            for interval in step.execution_timeline.intervals
        }
        for node in step.workload.nodes:
            if node.kind is not WorkloadKind.AF_ACTIVATION_TRANSFER:
                continue
            if node.node_id in repeated_template_node_ids:
                continue
            if node.node_id in activation_nodes_by_id:
                raise ValueError(
                    "execution artifact found duplicate activation transfer "
                    f"node ID: {node.node_id!r}")
            activation_nodes_by_id[node.node_id] = node
            step_by_activation_id[node.node_id] = step
            relative_interval_by_activation_id[node.node_id] = (
                intervals_by_call_id[node.node_id])
        timeline_repeats_by_id = {
            repeat.block_id: repeat
            for repeat in step.execution_timeline.block_repeats
        }
        for workload_repeat in step.workload.block_repeats:
            if not workload_repeat.trace_node_ids:
                continue
            if workload_repeat.block_id not in timeline_repeats_by_id:
                raise ValueError(
                    "execution artifact activation repeat lacks timeline")
            call_intervals = dict(
                timeline_repeats_by_id[
                    workload_repeat.block_id].call_intervals)
            for template_node_id, logical_node_ids in (
                    workload_repeat.trace_node_ids):
                node = step.workload.node(template_node_id)
                calls = tuple(
                    call for call in step.kernel_plan.calls
                    if template_node_id in call.source_node_ids
                )
                if len(calls) != 1 or calls[0].call_id not in call_intervals:
                    raise ValueError(
                        "execution artifact repeated activation transfer "
                        "lacks one call interval run")
                occurrence_intervals = call_intervals[calls[0].call_id]
                for logical_node_id, interval_times in zip(
                        logical_node_ids,
                        occurrence_intervals,
                        strict=True):
                    if logical_node_id in activation_nodes_by_id:
                        raise ValueError(
                            "execution artifact found duplicate logical "
                            "activation transfer ID")
                    activation_nodes_by_id[logical_node_id] = node
                    step_by_activation_id[logical_node_id] = step
                    relative_interval_by_activation_id[
                        logical_node_id] = interval_times
    activation_trace_ids = {
        transfer_id
        for transfer_id, transfer in trace_transfers_by_id.items()
        if transfer["payload"] == "activation"
    }
    if activation_trace_ids != set(activation_nodes_by_id):
        raise ValueError(
            "execution artifact activation trace and workload coverage "
            f"disagree: trace={sorted(activation_trace_ids)}, "
            f"workload={sorted(activation_nodes_by_id)}")
    known_trace_ids = kv_trace_ids | activation_trace_ids
    if known_trace_ids != set(trace_transfers_by_id):
        raise ValueError(
            "execution artifact found unsupported trace transfer payload")
    for transfer_id in sorted(activation_trace_ids):
        trace = trace_transfers_by_id[transfer_id]
        node = activation_nodes_by_id[transfer_id]
        step = step_by_activation_id[transfer_id]
        entry = entries_by_step_id[step.backend_step.schedule.step_id]
        expected_requests = tuple(
            request.request_id
            for request in step.backend_step.schedule.requests
        )
        if trace["request_ids"] != expected_requests:
            raise ValueError(
                "activation trace requests disagree with its Scheduler step")
        if trace["source_pool_id"] != node.pool_name:
            raise ValueError(
                "activation trace source disagrees with workload node")
        if (trace["destination_pool_id"]
                != node.attributes["destination_pool"]):
            raise ValueError(
                "activation trace destination disagrees with workload node")
        if trace["resource_id"] != node.attributes["resource_id"]:
            raise ValueError(
                "activation trace resource disagrees with workload node")
        if trace["link"] != node.attributes["link"]:
            raise ValueError(
                "activation trace link disagrees with workload node")
        if trace["byte_count"] != node.attributes["bytes"]:
            raise ValueError(
                "activation trace bytes disagree with workload node")
        calls = tuple(
            call for call in step.kernel_plan.calls
            if node.node_id in call.source_node_ids
        )
        if len(calls) != 1:
            raise ValueError(
                "activation workload must lower to exactly one call")
        call = calls[0]
        spec = call.kernel_spec
        if spec.kind is not KernelKind.COMMUNICATION or spec.transfer is None:
            raise ValueError(
                "activation workload must lower to COMMUNICATION")
        physical = spec.transfer
        expected_physical = _communication_transfer_from_node(node)
        actual_physical = _physical_transfer_manifest(physical)
        if actual_physical != expected_physical:
            raise ValueError(
                "activation physical transfer disagrees with workload")
        if call.resource != trace["resource_id"]:
            raise ValueError(
                "activation KernelCall resource disagrees with trace")
        if call_reserved_resource_ids(call) != trace["reserved_resource_ids"]:
            raise ValueError(
                "activation reserved resources disagree with trace")
        relative_start, relative_end = relative_interval_by_activation_id[
            transfer_id]
        absolute_start = entry.start_time_s + relative_start
        absolute_end = entry.start_time_s + relative_end
        if (not math.isclose(
                absolute_start, trace["start_time_s"],
                rel_tol=1e-12,
                abs_tol=_timestamp_subtraction_tolerance_s(
                    absolute_start, trace["start_time_s"]))
                or not math.isclose(
                    absolute_end, trace["end_time_s"],
                    rel_tol=1e-12,
                    abs_tol=_timestamp_subtraction_tolerance_s(
                        absolute_end, trace["end_time_s"]))):
            raise ValueError(
                "activation interval disagrees with trace transfer")


def _validate_pd_scheduler_chain(
    result: SimulationResult,
    entries_by_step_id: Mapping[str, ScheduleTraceEntry],
    trace_transfers_by_id: Mapping[str, Mapping[str, object]],
) -> None:
    kv_traces = tuple(
        transfer
        for transfer in trace_transfers_by_id.values()
        if transfer["payload"] == "kv"
    )
    pd_decisions = tuple(
        decision
        for decision in result.schedule_trace.route_decisions
        if decision.stage == "prefill_to_decode"
    )
    trace_keys = tuple(
        (
            transfer["request_ids"][0],
            transfer["source_pool_id"],
            transfer["destination_pool_id"],
        )
        for transfer in kv_traces
    )
    decision_keys = tuple(
        (
            decision.request_id,
            decision.source_engine_id,
            decision.selected_engine_id,
        )
        for decision in pd_decisions
    )
    if len(decision_keys) != len(set(decision_keys)):
        raise ValueError(
            "execution artifact found duplicate prefill_to_decode routes")
    if set(trace_keys) != set(decision_keys):
        raise ValueError(
            "execution artifact PD KV transfers and prefill_to_decode routes "
            "disagree")

    entries = tuple(entries_by_step_id.values())
    decisions_by_key = {
        key: decision for key, decision in zip(decision_keys, pd_decisions)
    }
    traces_by_key = {}
    for transfer, key in zip(kv_traces, trace_keys):
        if key not in traces_by_key:
            traces_by_key[key] = []
        traces_by_key[key].append(transfer)
    for key in decision_keys:
        request_id, source_pool_id, destination_pool_id = key
        decision = decisions_by_key[key]
        source_completions = tuple(
            entry for entry in entries
            if (entry.backend_step.schedule.pool_name == source_pool_id
                and request_id
                in entry.backend_step.schedule.finished_request_ids)
        )
        if len(source_completions) != 1:
            raise ValueError(
                "PD KV transfer requires exactly one source Scheduler "
                "completion")
        source_completion = source_completions[0]
        if not math.isclose(
                source_completion.end_time_s,
                decision.decision_time_s,
                rel_tol=1e-12,
                abs_tol=_timestamp_subtraction_tolerance_s(
                    source_completion.end_time_s,
                    decision.decision_time_s)):
            raise ValueError(
                "PD KV route decision disagrees with source Scheduler "
                "completion")
        physical_traces = tuple(traces_by_key[key])
        for transfer in physical_traces:
            if (decision.dispatch_time_s > transfer["start_time_s"]
                    and not math.isclose(
                        decision.dispatch_time_s,
                        transfer["start_time_s"],
                        rel_tol=1e-12,
                        abs_tol=_timestamp_subtraction_tolerance_s(
                            decision.dispatch_time_s,
                            transfer["start_time_s"]))):
                raise ValueError(
                    "PD KV transfer starts before its route dispatch")
            if (source_completion.end_time_s > transfer["start_time_s"]
                    and not math.isclose(
                        source_completion.end_time_s,
                        transfer["start_time_s"],
                        rel_tol=1e-12,
                        abs_tol=_timestamp_subtraction_tolerance_s(
                            source_completion.end_time_s,
                            transfer["start_time_s"]))):
                raise ValueError(
                    "PD KV transfer starts before source Scheduler completion")
        destination_entries = tuple(
            entry for entry in entries
            if (entry.backend_step.schedule.pool_name == destination_pool_id
                and any(
                    request.request_id == request_id
                    for request in entry.backend_step.schedule.requests))
        )
        if not destination_entries:
            raise ValueError(
                "PD KV transfer has no destination Scheduler step")
        destination_first = min(
            destination_entries, key=lambda entry: entry.start_time_s)
        transfer_completion_s = max(
            transfer["end_time_s"] for transfer in physical_traces)
        if (destination_first.start_time_s < transfer_completion_s
                and not math.isclose(
                    destination_first.start_time_s,
                    transfer_completion_s,
                    rel_tol=1e-12,
                    abs_tol=_timestamp_subtraction_tolerance_s(
                        destination_first.start_time_s,
                        transfer_completion_s))):
            raise ValueError(
                "destination Scheduler step starts before PD KV transfer "
                "completion")


def _backend_step_manifest(step: BackendStep) -> dict[str, object]:
    if not isinstance(step, BackendStep):
        raise TypeError("execution artifact requires BackendStep objects")
    return {
        "schedule": _json_value(asdict(step.schedule), "schedule step"),
        "kv_cache_events": tuple(
            _json_value(asdict(event), "KV cache event")
            for event in step.kv_cache_events
        ),
        "generated_tokens": {
            request_id: step.generated_tokens[request_id]
            for request_id in sorted(step.generated_tokens)
        },
    }


def _workload_graph_manifest(graph: WorkloadGraph) -> dict[str, object]:
    if not isinstance(graph, WorkloadGraph):
        raise TypeError("execution artifact requires WorkloadGraph objects")
    manifest = {
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
        "topological_order": graph.topological_order(),
        "nodes": tuple({
            "node_id": node.node_id,
            "kind": node.kind.value,
            "pool_name": node.pool_name,
            "shape": _json_value(node.shape, "workload shape"),
            "dtype": node.dtype,
            "variant": node.variant,
            "rank": node.rank,
            "attributes": _json_value(
                node.attributes, "workload attributes"),
        } for node in graph.nodes),
        "edges": tuple({
            "source": edge.source,
            "destination": edge.destination,
            "kind": edge.kind,
        } for edge in graph.edges),
    }
    if graph.block_repeats:
        manifest.update({
            "representation": "materialized_templates_with_logical_repeats_v1",
            "logical_node_count": graph.logical_node_count,
            "logical_edge_count": graph.logical_edge_count,
            "block_repeats": tuple({
                "block_id": repeat.block_id,
                "node_ids": repeat.node_ids,
                "repeat_count": repeat.repeat_count,
                "feedback_edges": tuple({
                    "source": edge.source,
                    "destination": edge.destination,
                    "kind": edge.kind,
                } for edge in repeat.feedback_edges),
                "trace_node_ids": tuple({
                    "template_node_id": template_node_id,
                    "logical_node_ids": logical_node_ids,
                } for template_node_id, logical_node_ids
                    in repeat.trace_node_ids),
            } for repeat in graph.block_repeats),
        })
    return manifest


def _lowering_manifest(
    graph: WorkloadGraph,
    plan: KernelPlan,
) -> dict[str, object]:
    if not isinstance(graph, WorkloadGraph):
        raise TypeError("execution artifact lowering requires WorkloadGraph")
    if not isinstance(plan, KernelPlan):
        raise TypeError("execution artifact lowering requires KernelPlan")
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    known_node_ids = set(nodes_by_id)
    calls_by_node_id = {
        node.node_id: [] for node in graph.nodes
    }
    calls_by_id = {call.call_id: call for call in plan.calls}
    for call in plan.calls:
        unknown = set(call.source_node_ids) - known_node_ids
        if unknown:
            raise ValueError(
                "execution artifact KernelCall references unknown workload "
                f"nodes: {sorted(unknown)}")
        for node_id in call.source_node_ids:
            node = nodes_by_id[node_id]
            if call.pool_name != node.pool_name:
                raise ValueError(
                    "execution artifact KernelCall pool disagrees with its "
                    f"source node: call={call.call_id!r}, node={node_id!r}")
            if call.rank != node.rank:
                raise ValueError(
                    "execution artifact KernelCall rank disagrees with its "
                    f"source node: call={call.call_id!r}, node={node_id!r}")
            calls_by_node_id[node_id].append(call.call_id)
    uncovered = [
        node_id for node_id, call_ids in calls_by_node_id.items()
        if not call_ids
    ]
    if uncovered:
        raise ValueError(
            "execution artifact workload nodes lack lowered KernelCalls: "
            f"{uncovered}")

    for edge in graph.edges:
        source_calls = calls_by_node_id[edge.source]
        destination_calls = calls_by_node_id[edge.destination]
        if set(source_calls).intersection(destination_calls):
            continue
        source_final_call_id = source_calls[-1]
        destination_first_call = calls_by_id[destination_calls[0]]
        if source_final_call_id not in destination_first_call.dependencies:
            raise ValueError(
                "execution artifact workload dependency is absent from "
                f"KernelPlan: {edge.source!r} -> {edge.destination!r}")
    return {
        "relation": "workload_nodes_to_kernel_calls_many_to_many_v1",
        "node_to_call_ids": tuple({
            "node_id": node.node_id,
            "call_ids": tuple(calls_by_node_id[node.node_id]),
        } for node in graph.nodes),
    }


def _kernel_plan_manifest(
    plan: KernelPlan,
    *,
    include_execution_support: bool = False,
) -> dict[str, object]:
    if not isinstance(plan, KernelPlan):
        raise TypeError("execution artifact requires KernelPlan objects")
    manifest = {
        "call_count": len(plan.calls),
        "topological_order": tuple(
            call.call_id for call in plan.topological_order()),
        "calls": tuple({
            "call_id": call.call_id,
            "pool_name": call.pool_name,
            "rank": call.rank,
            "resource": call.resource,
            "source_node_ids": call.source_node_ids,
            "dependencies": call.dependencies,
            "kernel_spec": _kernel_spec_manifest(
                call.kernel_spec,
                include_execution_support=include_execution_support,
            ),
        } for call in plan.calls),
    }
    if plan.block_repeats:
        manifest.update({
            "representation": "materialized_templates_with_logical_repeats_v1",
            "logical_call_count": plan.logical_call_count,
            "block_repeats": tuple({
                "block_id": repeat.block_id,
                "call_ids": repeat.call_ids,
                "repeat_count": repeat.repeat_count,
                "feedback_dependencies": tuple({
                    "destination_call_id": destination,
                    "source_call_ids": sources,
                } for destination, sources in repeat.feedback_dependencies),
            } for repeat in plan.block_repeats),
        })
    return manifest


def _kernel_spec_manifest(
    spec: KernelSpec,
    *,
    include_execution_support: bool = False,
) -> dict[str, object]:
    if not isinstance(spec, KernelSpec):
        raise TypeError("execution artifact requires KernelSpec objects")
    if spec.launch is None:
        launch = None
    else:
        launch = _launch_manifest(spec.launch)
    if spec.transfer is None:
        transfer = None
    else:
        transfer = _physical_transfer_manifest(spec.transfer)
    manifest = {
        "kind": spec.kind.value,
        "hardware_id": spec.hardware_id,
        "shape": _json_value(spec.shape, "kernel shape"),
        "dtype": spec.dtype,
        "variant": spec.variant,
        "launch": launch,
        "transfer": transfer,
        "attributes": _json_value(spec.attributes, "kernel attributes"),
    }
    if include_execution_support:
        manifest.update({
            "execution_support": spec.execution_support,
            "unsupported_reason": spec.unsupported_reason,
        })
    return manifest


def _launch_manifest(launch: LaunchConfig) -> dict[str, object]:
    if not isinstance(launch, LaunchConfig):
        raise TypeError("execution artifact requires LaunchConfig objects")
    if launch.tiling is None:
        tiling = None
    else:
        tiling = _tiling_manifest(launch.tiling)
    return {
        "algorithm": launch.algorithm,
        "grid": launch.grid,
        "block": launch.block,
        "tiling": tiling,
        "cluster": launch.cluster,
        "parameters": _json_value(
            launch.parameters, "launch parameters"),
        "execution_variant": launch.execution_variant,
    }


def _tiling_manifest(tiling: TilingSpec) -> dict[str, object]:
    if not isinstance(tiling, TilingSpec):
        raise TypeError("execution artifact requires TilingSpec objects")
    return {
        "tile_shape": _json_value(tiling.tile_shape, "tile shape"),
        "pipeline_depth": tiling.pipeline_depth,
        "split_k": tiling.split_k,
        "num_kv_splits": tiling.num_kv_splits,
    }


def _physical_transfer_manifest(
    transfer: PhysicalTransferConfig,
) -> dict[str, object]:
    if not isinstance(transfer, PhysicalTransferConfig):
        raise TypeError(
            "execution artifact requires PhysicalTransferConfig objects")
    return {
        "algorithm": transfer.algorithm,
        "bytes": transfer.bytes,
        "direction": transfer.direction,
        "link": transfer.link,
        "topology": transfer.topology,
        "tier": transfer.tier,
        "participants": transfer.participants,
        "operation": transfer.operation,
        "bytes_semantics": transfer.bytes_semantics,
        "source_rank": transfer.source_rank,
        "destination_rank": transfer.destination_rank,
        "ring_order": transfer.ring_order,
        "reduction": transfer.reduction,
        "backend": transfer.backend,
        "resource_id": transfer.resource_id,
        "backend_config_sha256": transfer.backend_config_sha256,
    }


def _communication_transfer_from_node(node) -> dict[str, object]:
    attributes = node.attributes
    return {
        "algorithm": attributes["algorithm"],
        "bytes": attributes["bytes"],
        "direction": attributes["direction"],
        "link": attributes["link"],
        "topology": attributes["topology"],
        "tier": None,
        "participants": tuple(attributes["participants"]),
        "operation": attributes["operation"],
        "bytes_semantics": attributes["bytes_semantics"],
        "source_rank": attributes["source_rank"],
        "destination_rank": attributes["destination_rank"],
        "ring_order": tuple(attributes["ring_order"]),
        "reduction": attributes["reduction"],
        "backend": attributes["backend"],
        "resource_id": attributes["resource_id"],
        "backend_config_sha256": attributes["backend_config_sha256"],
    }


def _predictions_manifest(
    plan: KernelPlan,
    predictions: Mapping[str, KernelPrediction],
) -> tuple[dict[str, object], ...]:
    if not isinstance(predictions, Mapping):
        raise TypeError("execution artifact predictions must be a mapping")
    call_ids = tuple(call.call_id for call in plan.calls)
    if set(predictions) != set(call_ids):
        raise ValueError(
            "execution artifact predictions must cover KernelPlan exactly")
    manifests = []
    for call_id in call_ids:
        prediction = predictions[call_id]
        if not isinstance(prediction, KernelPrediction):
            raise TypeError(
                "execution artifact predictions must be KernelPrediction "
                "objects")
        manifests.append({
            "call_id": call_id,
            "t_total_s": float(prediction.t_total_s),
            "t_launch_s": float(prediction.t_launch_s),
            "bottleneck_group": prediction.bottleneck_group,
            "t_per_group_s": _json_value(
                prediction.t_per_group, "prediction group times"),
            "t_per_resource_s": _json_value(
                prediction.t_per_resource, "prediction resource times"),
        })
    return tuple(manifests)


def _execution_timeline_manifest(
    plan: KernelPlan,
    timeline: ExecutionTimeline,
    predictions: Mapping[str, KernelPrediction],
    *,
    origin_time_s: float,
) -> dict[str, object]:
    if not isinstance(timeline, ExecutionTimeline):
        raise TypeError(
            "execution artifact requires ExecutionTimeline objects")
    if any(not isinstance(interval, ExecutionInterval)
           for interval in timeline.intervals):
        raise TypeError(
            "execution artifact timeline requires ExecutionInterval objects")
    call_ids = {call.call_id for call in plan.calls}
    interval_ids = tuple(interval.call_id for interval in timeline.intervals)
    if len(interval_ids) != len(set(interval_ids)):
        raise ValueError(
            "execution artifact timeline contains duplicate call intervals")
    if set(interval_ids) != call_ids:
        raise ValueError(
            "execution artifact timeline must cover KernelPlan exactly")
    _predictions_manifest(plan, predictions)
    if (isinstance(timeline.duration_s, bool)
            or not isinstance(timeline.duration_s, (int, float))
            or not math.isfinite(timeline.duration_s)
            or timeline.duration_s < 0):
        raise ValueError(
            "execution artifact timeline duration must be finite and "
            "non-negative")
    calls_by_id = {call.call_id: call for call in plan.calls}
    intervals_by_id = {}
    intervals_by_resource = {}
    for interval in timeline.intervals:
        call = calls_by_id[interval.call_id]
        if (interval.pool_name != call.pool_name
                or interval.rank != call.rank
                or interval.resource != call.resource):
            raise ValueError(
                "execution artifact interval placement disagrees with "
                f"KernelCall: {call.call_id!r}")
        expected_reserved_resources = call_reserved_resource_ids(call)
        if interval.reserved_resource_ids != expected_reserved_resources:
            raise ValueError(
                "execution artifact interval reserved resources disagree "
                f"with KernelCall: {call.call_id!r}")
        for name, value in (
                ("start", interval.start_time_s),
                ("end", interval.end_time_s)):
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0):
                raise ValueError(
                    f"execution artifact interval {name} must be finite "
                    "and non-negative")
        if interval.end_time_s < interval.start_time_s:
            raise ValueError(
                "execution artifact interval end precedes start")
        if interval.end_time_s > timeline.duration_s and not math.isclose(
                interval.end_time_s,
                timeline.duration_s,
                rel_tol=1e-12,
                abs_tol=_timestamp_subtraction_tolerance_s(
                    interval.end_time_s, timeline.duration_s)):
            raise ValueError(
                "execution artifact interval exceeds timeline duration")
        interval_duration_s = (
            interval.end_time_s - interval.start_time_s)
        expected_duration_s = predictions[call.call_id].t_total_s
        duration_tolerance_s = max(
            _timestamp_subtraction_tolerance_s(
                interval.start_time_s, interval.end_time_s),
            _timestamp_subtraction_tolerance_s(
                origin_time_s + interval.start_time_s,
                origin_time_s + interval.end_time_s),
        )
        if not math.isclose(
                interval_duration_s,
                expected_duration_s,
                rel_tol=1e-12,
                abs_tol=duration_tolerance_s):
            raise ValueError(
                "execution artifact interval duration disagrees with "
                f"prediction: {call.call_id!r}; "
                f"actual={interval_duration_s!r}, "
                f"expected={expected_duration_s!r}, "
                f"abs_tol={duration_tolerance_s!r}")
        intervals_by_id[call.call_id] = interval
        for resource_id in interval.reserved_resource_ids:
            if resource_id not in intervals_by_resource:
                intervals_by_resource[resource_id] = []
            intervals_by_resource[resource_id].append(interval)
    maximum_end_s = max(
        interval.end_time_s for interval in timeline.intervals)
    if not math.isclose(
            maximum_end_s,
            timeline.duration_s,
            rel_tol=1e-12,
            abs_tol=_timestamp_subtraction_tolerance_s(
                maximum_end_s, timeline.duration_s)):
        raise ValueError(
            "execution artifact timeline duration disagrees with intervals")
    for call in plan.calls:
        interval = intervals_by_id[call.call_id]
        for dependency in call.dependencies:
            dependency_interval = intervals_by_id[dependency]
            if (interval.start_time_s < dependency_interval.end_time_s
                    and not math.isclose(
                        interval.start_time_s,
                        dependency_interval.end_time_s,
                        rel_tol=1e-12,
                        abs_tol=_timestamp_subtraction_tolerance_s(
                            interval.start_time_s,
                            dependency_interval.end_time_s))):
                raise ValueError(
                    "execution artifact interval begins before dependency "
                    f"completion: {call.call_id!r}")
    for resource_intervals in intervals_by_resource.values():
        ordered = sorted(
            resource_intervals,
            key=lambda interval: (
                interval.start_time_s,
                interval.end_time_s,
                interval.call_id,
            ),
        )
        for previous, current in zip(ordered, ordered[1:]):
            if (current.start_time_s < previous.end_time_s
                    and not math.isclose(
                        current.start_time_s,
                        previous.end_time_s,
                        rel_tol=1e-12,
                        abs_tol=_timestamp_subtraction_tolerance_s(
                            current.start_time_s,
                            previous.end_time_s))):
                raise ValueError(
                    "execution artifact intervals overlap on one resource")
    if (isinstance(origin_time_s, bool)
            or not isinstance(origin_time_s, (int, float))
            or not math.isfinite(origin_time_s)
            or origin_time_s < 0):
        raise ValueError(
            "execution artifact timeline origin must be finite and "
            "non-negative")
    origin_time_s = float(origin_time_s)
    if any(
            not isinstance(event, HostExecutionInterval)
            for event in timeline.host_intervals):
        raise TypeError(
            "execution artifact Host timeline requires HostExecutionInterval "
            "objects")
    if timeline.runtime_graph_mode is not None and timeline.runtime_graph_mode not in (
            "NONE", "FULL", "PIECEWISE"):
        raise ValueError(
            "execution artifact timeline runtime Graph mode is invalid")
    event_runtime_modes = {
        event.runtime_graph_mode for event in timeline.host_intervals
    }
    if event_runtime_modes and event_runtime_modes != {
            timeline.runtime_graph_mode}:
        raise ValueError(
            "execution artifact Host event runtime modes disagree with summary")
    event_ids = tuple(event.event_id for event in timeline.host_intervals)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("execution artifact Host event IDs must be unique")
    eager_event_count = sum(
        event.event_kind == "eager_submit"
        for event in timeline.host_intervals)
    graph_event_count = sum(
        event.event_kind == "graph_replay"
        for event in timeline.host_intervals)
    if eager_event_count != timeline.eager_dispatch_count:
        raise ValueError(
            "execution artifact eager Host event count disagrees with summary")
    if graph_event_count != timeline.graph_replay_count:
        raise ValueError(
            "execution artifact Graph replay count disagrees with summary")
    expected_launch_kinds = tuple(sorted({
        ("eager" if event.event_kind == "eager_submit"
         else "runtime_prepare" if event.event_kind == "runtime_prepare"
         else "graph_replay")
        for event in timeline.host_intervals
    }))
    if timeline.launch_kinds != expected_launch_kinds:
        raise ValueError(
            "execution artifact Host event kinds disagree with summary")
    represented_launch_cpu_s = sum(
        event.end_time_s - event.start_time_s
        for event in timeline.host_intervals)
    accumulated_timestamp_tolerance_s = max(
        _timestamp_subtraction_tolerance_s(
            represented_launch_cpu_s, timeline.launch_cpu_time_s),
        len(timeline.host_intervals) * 4.0 * math.ulp(
            origin_time_s + timeline.duration_s),
    )
    if not math.isclose(
            represented_launch_cpu_s,
            timeline.launch_cpu_time_s,
            rel_tol=1e-12,
            abs_tol=accumulated_timestamp_tolerance_s):
        raise ValueError(
            "execution artifact Host event durations disagree with summary")
    repeated_call_ids = {
        call_id
        for repeat in plan.block_repeats
        for call_id in repeat.call_ids
    }
    occurrence_count_by_call = dict(timeline.call_occurrence_counts)
    valid_occurrences = {
        (call.call_id, occurrence_index)
        for call in plan.calls
        for occurrence_index in (
            range(occurrence_count_by_call[call.call_id])
            if call.call_id in repeated_call_ids else (-1,)
        )
    }
    previous_event_by_lane = {}
    for event in timeline.host_intervals:
        if event.end_time_s > timeline.duration_s and not math.isclose(
                event.end_time_s,
                timeline.duration_s,
                rel_tol=1e-12,
                abs_tol=_timestamp_subtraction_tolerance_s(
                    event.end_time_s, timeline.duration_s)):
            raise ValueError(
                "execution artifact Host event exceeds timeline duration")
        trigger = (event.call_id, event.call_occurrence_index)
        if trigger not in valid_occurrences:
            raise ValueError(
                "execution artifact Host event references an invalid call "
                "occurrence")
        lane = (event.pool_name, event.rank)
        if lane not in call_host_lanes(calls_by_id[event.call_id]):
            raise ValueError(
                "execution artifact Host event lane disagrees with KernelCall")
        for occurrence in (
                event.covered_call_occurrences
                + event.device_predecessor_occurrences):
            if occurrence not in valid_occurrences:
                raise ValueError(
                    "execution artifact Host event references an invalid "
                    "covered or predecessor occurrence")
        for call_id, _occurrence_index in event.covered_call_occurrences:
            if lane not in call_host_lanes(calls_by_id[call_id]):
                raise ValueError(
                    "execution artifact Host event covers a call on another "
                    "Host lane")
        previous_event = (
            previous_event_by_lane[lane]
            if lane in previous_event_by_lane else None)
        expected_predecessor = (
            None if previous_event is None else previous_event.event_id)
        if event.host_predecessor_event_id != expected_predecessor:
            raise ValueError(
                "execution artifact Host predecessor chain is inconsistent")
        if (previous_event is not None
                and event.start_time_s < previous_event.end_time_s
                and not math.isclose(
                    event.start_time_s,
                    previous_event.end_time_s,
                    rel_tol=1e-12,
                    abs_tol=_timestamp_subtraction_tolerance_s(
                        event.start_time_s, previous_event.end_time_s))):
            raise ValueError(
                "execution artifact Host events overlap on one lane")
        previous_event_by_lane[lane] = event
    manifest = {
        "schema_version": 2,
        "time_basis": "relative_to_origin_with_absolute_projection",
        "origin_time_s": origin_time_s,
        "duration_s": float(timeline.duration_s),
        "predicted_device_time_s": float(timeline.predicted_device_time_s),
        "launch_cpu_time_s": float(timeline.launch_cpu_time_s),
        "launch_critical_path_s": float(timeline.launch_critical_path_s),
        "host_device_overlap_s": float(timeline.host_device_overlap_s),
        "host_exposed_critical_path_s": float(
            timeline.host_exposed_critical_path_s),
        "eager_dispatch_count": timeline.eager_dispatch_count,
        "graph_replay_count": timeline.graph_replay_count,
        "launch_kinds": timeline.launch_kinds,
        "runtime_graph_mode": timeline.runtime_graph_mode,
        "host_events": tuple({
            "event_id": event.event_id,
            "event_kind": event.event_kind,
            "pool_name": event.pool_name,
            "rank": event.rank,
            "call_id": event.call_id,
            "call_occurrence_index": event.call_occurrence_index,
            "covered_call_occurrences": tuple({
                "call_id": call_id,
                "occurrence_index": occurrence_index,
            } for call_id, occurrence_index in event.covered_call_occurrences),
            "device_predecessor_occurrences": tuple({
                "call_id": call_id,
                "occurrence_index": occurrence_index,
            } for call_id, occurrence_index in (
                event.device_predecessor_occurrences)),
            "host_predecessor_event_id": event.host_predecessor_event_id,
            "runtime_graph_mode": event.runtime_graph_mode,
            "graph_segment_id": event.graph_segment_id,
            "clock_domain": "predicted_simulation_relative_s",
            "timing_source": "predicted",
            "exposure_scope": "lane_local_delay_before_call",
            "start_time_s": float(event.start_time_s),
            "end_time_s": float(event.end_time_s),
            "absolute_start_time_s": (
                origin_time_s + float(event.start_time_s)),
            "absolute_end_time_s": (
                origin_time_s + float(event.end_time_s)),
            "exposed_time_s": float(event.exposed_time_s),
            "overlapped_time_s": float(event.overlapped_time_s),
        } for event in timeline.host_intervals),
        "intervals": tuple({
            "call_id": interval.call_id,
            "pool_name": interval.pool_name,
            "rank": interval.rank,
            "resource": interval.resource,
            "reserved_resource_ids": interval.reserved_resource_ids,
            "start_time_s": float(interval.start_time_s),
            "end_time_s": float(interval.end_time_s),
            "absolute_start_time_s": (
                origin_time_s + float(interval.start_time_s)),
            "absolute_end_time_s": (
                origin_time_s + float(interval.end_time_s)),
        } for interval in timeline.intervals),
    }
    if plan.block_repeats:
        repeats_by_id = {
            repeat.block_id: repeat for repeat in plan.block_repeats
        }
        timeline_repeat_ids = {
            repeat.block_id for repeat in timeline.block_repeats
        }
        if set(repeats_by_id) != timeline_repeat_ids:
            raise ValueError(
                "execution timeline block repeats disagree with KernelPlan")
        timeline_repeats_by_id = {
            repeat.block_id: repeat for repeat in timeline.block_repeats
        }
        for block_id, plan_repeat in repeats_by_id.items():
            timeline_repeat = timeline_repeats_by_id[block_id]
            call_intervals = dict(timeline_repeat.call_intervals)
            if set(call_intervals) != set(plan_repeat.call_ids):
                raise ValueError(
                    "execution block repeat intervals do not cover its calls")
            for call_id in plan_repeat.call_ids:
                for start_time_s, end_time_s in call_intervals[call_id]:
                    if not math.isclose(
                            end_time_s - start_time_s,
                            predictions[call_id].t_total_s,
                            rel_tol=1e-12,
                            abs_tol=max(
                                _timestamp_subtraction_tolerance_s(
                                    start_time_s, end_time_s),
                                _timestamp_subtraction_tolerance_s(
                                    origin_time_s + start_time_s,
                                    origin_time_s + end_time_s))):
                        raise ValueError(
                            "execution block repeat interval duration "
                            "disagrees with prediction")
                representative = intervals_by_id[call_id]
                if call_intervals[call_id][-1] != (
                        representative.start_time_s,
                        representative.end_time_s):
                    raise ValueError(
                        "execution block repeat representative interval is "
                        "not its final occurrence")
        occurrence_counts = dict(timeline.call_occurrence_counts)
        if set(occurrence_counts) != call_ids:
            raise ValueError(
                "execution timeline occurrence counts must cover KernelPlan")
        if sum(occurrence_counts.values()) != plan.logical_call_count:
            raise ValueError(
                "execution timeline occurrence counts disagree with logical "
                "call count")
        manifest.update({
            "representation": "materialized_templates_with_logical_repeats_v1",
            "logical_interval_count": plan.logical_call_count,
            "call_occurrence_counts": tuple({
                "call_id": call_id,
                "occurrence_count": count,
            } for call_id, count in timeline.call_occurrence_counts),
            "pool_activity": tuple({
                "pool_name": activity.activity_id,
                "interval_count": activity.interval_count,
                "intervals": tuple({
                    "start_time_s": float(start_time_s),
                    "end_time_s": float(end_time_s),
                    "absolute_start_time_s": (
                        origin_time_s + float(start_time_s)),
                    "absolute_end_time_s": (
                        origin_time_s + float(end_time_s)),
                } for start_time_s, end_time_s in activity.intervals),
            } for activity in timeline.pool_activity),
            "reserved_resource_activity": tuple({
                "resource_id": activity.activity_id,
                "interval_count": activity.interval_count,
                "intervals": tuple({
                    "start_time_s": float(start_time_s),
                    "end_time_s": float(end_time_s),
                    "absolute_start_time_s": (
                        origin_time_s + float(start_time_s)),
                    "absolute_end_time_s": (
                        origin_time_s + float(end_time_s)),
                } for start_time_s, end_time_s in activity.intervals),
            } for activity in timeline.reserved_resource_activity),
            "block_repeats": tuple({
                "block_id": repeat.block_id,
                "repeat_count": repeat.repeat_count,
                "first_iteration_start_time_s": float(
                    repeat.first_iteration_start_time_s),
                "last_iteration_end_time_s": float(
                    repeat.last_iteration_end_time_s),
                "call_intervals": tuple({
                    "call_id": call_id,
                    "intervals": tuple({
                        "start_time_s": float(start_time_s),
                        "end_time_s": float(end_time_s),
                        "absolute_start_time_s": (
                            origin_time_s + float(start_time_s)),
                        "absolute_end_time_s": (
                            origin_time_s + float(end_time_s)),
                    } for start_time_s, end_time_s in call_intervals),
                } for call_id, call_intervals in repeat.call_intervals),
            } for repeat in timeline.block_repeats),
        })
    elif timeline.block_repeats:
        raise ValueError(
            "execution timeline has block repeats absent from KernelPlan")
    return manifest


def _json_value(value: object, path: str) -> object:
    if isinstance(value, Enum):
        return _json_value(value.value, path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        invalid_keys = [
            key for key in value if not isinstance(key, str) or not key
        ]
        if invalid_keys:
            raise TypeError(
                f"{path} mapping keys must be non-empty strings: "
                f"{invalid_keys}")
        return {
            key: _json_value(value[key], f"{path}.{key}")
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return tuple(
            _json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} contains a non-JSON value of type "
        f"{type(value).__name__}")
