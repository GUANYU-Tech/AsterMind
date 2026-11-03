"""Terrestrial multi-region runtime models used by OrbitServe baselines.

The SkyWalker path reproduces the published control logic (regional load
balancers, local-first routing, pending-request selective pushing, two-layer
locality-aware routing, and cross-region Internet delay).  Parameters that the
paper does not publish remain explicit deterministic surrogates.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any

from ...common import to_jsonable
from .contracts import GroundRequestRecord as RuntimeRequestRecord


TERRESTRIAL_NETWORK_TRACE = "terrestrial_network_surrogate"
TERRESTRIAL_GROUND_SCOPE = "skywalker_ground_reproduction_surrogate"


@dataclass(frozen=True)
class GroundRuntimeSimulationResult:
    records: tuple[RuntimeRequestRecord, ...]
    decision_traces: tuple[dict[str, Any], ...]
    state_summary: dict[str, Any]
    warnings: tuple[str, ...]
    runtime_metrics: dict[str, Any] | None = None
    communication_metrics: dict[str, Any] | None = None
    energy_metrics: dict[str, Any] | None = None
    session_metrics: dict[str, Any] | None = None
    region_metrics: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] | None = None
    evaluated: bool = True
    scope: str = TERRESTRIAL_GROUND_SCOPE
    trace_model: str = TERRESTRIAL_NETWORK_TRACE
    valid_for_paper_main_results: bool = False


@dataclass(frozen=True)
class _RegionConfig:
    region_id: str
    replica_count: int
    local_rtt_s: float
    lat_deg: float | None = None
    lon_deg: float | None = None


@dataclass
class _ReplicaJob:
    request_id: str
    enqueue_s: float
    start_s: float
    end_s: float
    batch_tokens: int


@dataclass
class _CacheEntry:
    tokens: int
    available_at_s: float
    prefix_signature: tuple[str, ...]
    signature_kind: str


@dataclass
class _ReplicaState:
    replica_id: str
    region_id: str
    jobs: list[_ReplicaJob] = field(default_factory=list)
    # Routing metadata becomes visible when the regional LB dispatches a
    # request.  The realized KV cache is separate and only becomes reusable
    # when the corresponding request completes.
    routing_index: OrderedDict[str, _CacheEntry] = field(default_factory=OrderedDict)
    cache: OrderedDict[str, _CacheEntry] = field(default_factory=OrderedDict)

    def active_jobs(self, time_s: float) -> list[_ReplicaJob]:
        return [job for job in self.jobs if job.start_s <= time_s < job.end_s]

    def pending_jobs(self, time_s: float) -> list[_ReplicaJob]:
        return [job for job in self.jobs if job.enqueue_s <= time_s < job.start_s]

    def outstanding_count(self, time_s: float) -> int:
        return len(self.active_jobs(time_s)) + len(self.pending_jobs(time_s))

    def active_tokens(self, time_s: float) -> int:
        return sum(job.batch_tokens for job in self.active_jobs(time_s))

    def next_state_change_after(self, time_s: float) -> float | None:
        candidates = [
            value
            for job in self.jobs
            for value in (job.start_s, job.end_s)
            if value > time_s + 1e-12
        ]
        return min(candidates) if candidates else None


@dataclass(frozen=True)
class _ServiceProfile:
    """Execution parameters selected for one ground replica.

    The profile is an explicitly analytical contract: it can be calibrated
    from a measurement, but it does not turn this discrete-event simulator
    into an inference-engine replay.  Keeping that distinction in the data
    model makes the resulting provenance auditable.
    """

    profile_id: str
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    first_token_overhead_s: float
    continuous_batch_capacity_tokens: int
    max_batch_size: int
    batch_contention_alpha: float
    calibration_source: str
    is_surrogate: bool
    batch_slowdown_source: str


@dataclass(frozen=True)
class _WanLatencyProfile:
    """Fixed or deterministically sampled WAN RTT for one directed edge."""

    profile_id: str
    mode: str
    fixed_rtt_s: float | None
    samples_s: tuple[float, ...]
    calibration_source: str
    is_surrogate: bool


@dataclass(frozen=True)
class _GroundConfig:
    regions: tuple[_RegionConfig, ...]
    wan_rtt_s: dict[str, dict[str, float]]
    wan_latency_profiles: dict[str, dict[str, _WanLatencyProfile]]
    probe_interval_s: float
    remote_queue_buffer_requests: int
    routing_policy: str
    selective_pushing: str
    prefill_tokens_per_s: float
    decode_tokens_per_s: float
    continuous_batch_capacity_tokens: int
    kv_cache_capacity_tokens: int
    max_batch_size: int
    max_replica_queue_requests: int
    max_lb_queue_wait_s: float | None
    prefix_cache_max_entries: int
    max_outstanding_requests: int
    propagation_speed_km_s: float
    internet_path_stretch: float
    internet_router_overhead_s: float
    arrival_time_scale: float
    batch_contention_alpha: float
    admission_output_reservation_tokens: int
    hash_virtual_nodes_per_target: int
    hash_seed: str
    prefix_affinity_min_similarity: float | None
    prefix_low_affinity_fallback: str
    explicit_regions: bool
    default_service_profile: _ServiceProfile
    service_profiles_by_region: dict[str, _ServiceProfile]
    service_profiles_by_replica: dict[str, _ServiceProfile]
    raw: dict[str, Any]


@dataclass
class _GroundQueueContext:
    request: dict[str, Any]
    origin_region_id: str
    client_to_lb_s: float
    first_lb_arrival_s: float
    current_lb_arrival_s: float
    accumulated_lb_wait_s: float = 0.0
    source_lb_wait_s: float = 0.0
    remote_lb_wait_s: float = 0.0
    forwarded: bool = False
    target_region_id: str | None = None
    wan_rtt_s: float = 0.0
    wan_latency_profile: dict[str, Any] | None = None
    source_decision_time_s: float | None = None
    source_audit: dict[str, Any] = field(default_factory=dict)
    current_queue_interval: list[float] | None = None
    finished: bool = False


@dataclass(frozen=True)
class _ClientProgram:
    """One client-visible program, which may contain parallel requests."""

    client_id: str
    program_id: str
    sequence_index: int
    request_ids: tuple[str, ...]
    request_release_offsets_s: dict[str, float]
    request_dependencies: dict[str, tuple[str, ...]]
    initial_release_s: float


@dataclass(frozen=True)
class _ClientHarness:
    """Fail-closed implementation of the paper's one-program-per-client rule."""

    inter_program_think_time_s: float
    program_execution_model: str
    programs_by_client: dict[str, tuple[_ClientProgram, ...]]




def run_ground_runtime(
    *,
    requests: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    runtime_strategy: Any,
    regions: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    gateways: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    policy: str | None = None,
) -> GroundRuntimeSimulationResult:
    """Run the causal, global-event terrestrial SkyWalker reproduction.

    Client arrivals and inter-region forwarded arrivals share one event heap.
    Every regional load balancer owns one FCFS queue, so a forwarded request
    cannot reserve a future position ahead of a request that reaches the
    destination earlier.  This ordering is the central difference from the
    retired sequential implementation above.
    """

    policy_name = str(
        policy or getattr(runtime_strategy, "policy", "skywalker_like_ground")
    )
    if policy_name not in {"skywalker_like_ground", "ground_regional"}:
        raise ValueError(f"unsupported terrestrial baseline policy: {policy_name!r}")
    raw_requests = tuple(dict(item) for item in requests if isinstance(item, dict))
    config = _resolve_config(runtime_strategy, regions=regions, requests=raw_requests)
    region_by_id = {item.region_id: item for item in config.regions}
    gateway_to_region = _gateway_region_map(gateways)
    normalized = tuple(
        _normalize_request(item, index, gateway_to_region=gateway_to_region)
        for index, item in enumerate(raw_requests)
    )
    client_harness = _resolve_client_harness(
        config.raw,
        normalized,
        arrival_time_scale=config.arrival_time_scale,
    )
    if (
        policy_name == "skywalker_like_ground"
        and config.routing_policy in {"consistent_hash", "skywalker_ch", "ch"}
        and _profile_bool(config.raw.get("require_hash_identity"), False)
        and any(item["hash_identity_source"] == "prefix_key_fallback" for item in normalized)
    ):
        raise ValueError(
            "require_hash_identity=true but one or more requests lack explicit user_id/hashed_user_id or session_id"
        )
    if normalized and abs(config.arrival_time_scale - 1.0) > 1e-12:
        first_arrival_s = min(item["arrival_time_s"] for item in normalized)
        normalized = tuple(
            {
                **item,
                "arrival_time_s": first_arrival_s
                + (item["arrival_time_s"] - first_arrival_s)
                * config.arrival_time_scale,
                "original_arrival_time_s": item["arrival_time_s"],
            }
            for item in normalized
        )
    unknown_regions = sorted(
        {
            item["origin_region_id"]
            for item in normalized
            if item["origin_region_id"] not in region_by_id
        }
    )
    if unknown_regions:
        if config.explicit_regions:
            raise ValueError(
                "ground request references region(s) absent from ground_config.regions: "
                + ", ".join(unknown_regions)
            )
        config = _config_with_inferred_regions(config, unknown_regions)
        region_by_id = {item.region_id: item for item in config.regions}

    normalized = tuple(
        _resolve_ground_ingress(item, config=config, region_by_id=region_by_id)
        for item in normalized
    )

    replicas_by_region = {
        region.region_id: [
            _ReplicaState(
                replica_id=f"ground:{region.region_id}:replica-{index}",
                region_id=region.region_id,
            )
            for index in range(region.replica_count)
        ]
        for region in config.regions
    }
    if not any(replicas_by_region.values()):
        raise ValueError("ground runtime has no configured replicas")

    lb_queues: dict[str, list[_GroundQueueContext]] = {
        region.region_id: [] for region in config.regions
    }
    # Mutable [arrival, exit] intervals make probe-time LB queue snapshots
    # auditable even when the probe tick precedes the current event time.
    lb_queue_intervals: dict[str, list[list[float]]] = {
        region.region_id: [] for region in config.regions
    }
    remote_snapshots: dict[str, dict[str, OrderedDict[str, _CacheEntry]]] = {
        source.region_id: {
            target.region_id: OrderedDict()
            for target in config.regions
            if target.region_id != source.region_id
        }
        for source in config.regions
    }
    round_robin_index = {region.region_id: 0 for region in config.regions}

    # Heap tuple: time, priority (all arrivals before dispatch), stable key,
    # insertion sequence, event kind, region, payload.
    events: list[tuple[float, int, str, int, str, str, Any]] = []
    event_sequence = 0
    dispatch_generation = {region.region_id: 0 for region in config.regions}
    scheduled_dispatch: dict[str, tuple[float, int] | None] = {
        region.region_id: None for region in config.regions
    }
    outcomes: list[
        tuple[tuple[float, str], RuntimeRequestRecord, dict[str, Any]]
    ] = []
    client_program_by_request: dict[str, _ClientProgram] = {}
    client_program_position: dict[tuple[str, int], int] = {}
    client_program_remaining: dict[tuple[str, int], int] = {}
    client_program_terminal_s: dict[tuple[str, int], list[float]] = {}
    client_program_request_terminal_s: dict[str, float] = {}
    client_program_request_scheduled: set[str] = set()
    client_program_start_s: dict[tuple[str, int], float] = {}
    client_active_programs: dict[str, int] = {}
    client_peak_inflight_programs: dict[str, int] = {}
    client_released_program_count = 0

    def push_event(
        time_s: float,
        priority: int,
        stable_key: str,
        kind: str,
        region_id: str,
        payload: Any,
    ) -> None:
        nonlocal event_sequence
        event_sequence += 1
        heapq.heappush(
            events,
            (
                float(time_s),
                int(priority),
                stable_key,
                event_sequence,
                kind,
                region_id,
                payload,
            ),
        )

    def schedule_dispatch(region_id: str, time_s: float) -> None:
        if not lb_queues[region_id]:
            return
        current = scheduled_dispatch[region_id]
        if current is not None and current[0] <= time_s + 1e-12:
            return
        dispatch_generation[region_id] += 1
        generation = dispatch_generation[region_id]
        scheduled_dispatch[region_id] = (float(time_s), generation)
        push_event(
            time_s,
            1,
            region_id,
            "dispatch",
            region_id,
            generation,
        )

    def close_head(region_id: str, context: _GroundQueueContext, time_s: float) -> None:
        queue = lb_queues[region_id]
        if not queue or queue[0] is not context:
            raise RuntimeError("ground LB FCFS queue head changed unexpectedly")
        if context.current_queue_interval is not None:
            context.current_queue_interval[1] = float(time_s)
        queue.pop(0)
        scheduled_dispatch[region_id] = None
        if queue:
            schedule_dispatch(region_id, time_s)

    def append_outcome(
        context: _GroundQueueContext,
        record: RuntimeRequestRecord,
        trace: dict[str, Any],
        *,
        terminal_event_time_s: float,
    ) -> None:
        nonlocal client_released_program_count
        context.finished = True
        if client_harness is not None:
            program = client_program_by_request[context.request["request_id"]]
            program_key = (program.client_id, program.sequence_index)
            terminal_s = (
                record.arrival_time_s + record.e2e_s
                if record.completed and record.e2e_s is not None
                else terminal_event_time_s
            )
            client_program_remaining[program_key] -= 1
            client_program_terminal_s[program_key].append(float(terminal_s))
            client_program_request_terminal_s[context.request["request_id"]] = float(
                terminal_s
            )
            record = replace(
                record,
                provenance={
                    **record.provenance,
                    "client_harness": {
                        "mode": "one_program_per_client",
                        "program_execution_model": client_harness.program_execution_model,
                        "client_id": program.client_id,
                        "program_id": program.program_id,
                        "program_sequence_index": program.sequence_index,
                        "program_request_release_offset_s": program.request_release_offsets_s[
                            context.request["request_id"]
                        ],
                        "terminal_time_s": terminal_s,
                    },
                },
            )
            trace = {
                **trace,
                "client_harness": record.provenance["client_harness"],
            }
            release_ready_program_requests(program)
            if client_program_remaining[program_key] == 0:
                client_active_programs[program.client_id] -= 1
                next_position = client_program_position[program_key] + 1
                programs = client_harness.programs_by_client[program.client_id]
                if next_position < len(programs):
                    release_time_s = (
                        max(client_program_terminal_s[program_key])
                        + client_harness.inter_program_think_time_s
                    )
                    release_program(program.client_id, next_position, release_time_s)
        outcomes.append(
            (
                (context.request["arrival_time_s"], context.request["request_id"]),
                record,
                trace,
            )
        )

    contexts: list[_GroundQueueContext] = []
    contexts_by_request_id: dict[str, _GroundQueueContext] = {}
    for request in sorted(
        normalized, key=lambda item: (item["arrival_time_s"], item["request_id"])
    ):
        origin = request["origin_region_id"]
        ingress = request["ingress_region_id"]
        client_to_lb_s = _client_to_ground_ingress_one_way_s(
            config,
            region_by_id[origin],
            region_by_id[ingress],
            request_key=request["request_id"],
            dns_resolution_delay_s=request["dns_resolution_delay_s"],
        )
        arrival_s = request["arrival_time_s"] + client_to_lb_s
        context = _GroundQueueContext(
            request=request,
            origin_region_id=origin,
            client_to_lb_s=client_to_lb_s,
            first_lb_arrival_s=arrival_s,
            current_lb_arrival_s=arrival_s,
        )
        contexts.append(context)
        contexts_by_request_id[request["request_id"]] = context

    def schedule_client_request(
        context: _GroundQueueContext,
        *,
        client_release_s: float,
        release_offset_s: float,
    ) -> None:
        request_arrival_s = client_release_s + release_offset_s
        context.request["arrival_time_s"] = request_arrival_s
        context.first_lb_arrival_s = request_arrival_s + context.client_to_lb_s
        context.current_lb_arrival_s = context.first_lb_arrival_s
        push_event(
            context.first_lb_arrival_s,
            0,
            context.request["request_id"],
            "arrival",
            context.request["ingress_region_id"],
            context,
        )

    def release_ready_program_requests(program: _ClientProgram) -> None:
        program_key = (program.client_id, program.sequence_index)
        program_start_s = client_program_start_s[program_key]
        for request_id in program.request_ids:
            if request_id in client_program_request_scheduled:
                continue
            dependencies = program.request_dependencies[request_id]
            if any(
                dependency not in client_program_request_terminal_s
                for dependency in dependencies
            ):
                continue
            release_base_s = (
                max(
                    client_program_request_terminal_s[dependency]
                    for dependency in dependencies
                )
                if dependencies
                else program_start_s
            )
            schedule_client_request(
                contexts_by_request_id[request_id],
                client_release_s=release_base_s,
                release_offset_s=program.request_release_offsets_s[request_id],
            )
            client_program_request_scheduled.add(request_id)

    def release_program(client_id: str, position: int, release_time_s: float) -> None:
        nonlocal client_released_program_count
        if client_harness is None:
            raise RuntimeError("client program release called without client harness")
        program = client_harness.programs_by_client[client_id][position]
        program_key = (client_id, program.sequence_index)
        if program_key in client_program_remaining:
            raise RuntimeError("client program released more than once")
        client_program_position[program_key] = position
        client_program_remaining[program_key] = len(program.request_ids)
        client_program_terminal_s[program_key] = []
        client_program_start_s[program_key] = release_time_s
        client_active_programs[client_id] = client_active_programs.get(client_id, 0) + 1
        client_peak_inflight_programs[client_id] = max(
            client_peak_inflight_programs.get(client_id, 0),
            client_active_programs[client_id],
        )
        if client_active_programs[client_id] != 1:
            raise RuntimeError("client harness released overlapping programs for one client")
        client_released_program_count += 1
        release_ready_program_requests(program)

    if client_harness is None:
        for context in contexts:
            push_event(
                context.first_lb_arrival_s,
                0,
                context.request["request_id"],
                "arrival",
                context.request["ingress_region_id"],
                context,
            )
    else:
        for programs in client_harness.programs_by_client.values():
            for program in programs:
                for request_id in program.request_ids:
                    client_program_by_request[request_id] = program
        for client_id, programs in client_harness.programs_by_client.items():
            if programs:
                release_program(client_id, 0, programs[0].initial_release_s)

    while events:
        (
            time_s,
            _priority,
            _stable_key,
            _sequence,
            kind,
            region_id,
            payload,
        ) = heapq.heappop(events)
        if kind == "arrival":
            context = payload
            if context.finished:
                continue
            context.current_lb_arrival_s = time_s
            interval = [time_s, math.inf]
            context.current_queue_interval = interval
            lb_queue_intervals[region_id].append(interval)
            queue = lb_queues[region_id]
            queue.append(context)
            if len(queue) == 1:
                schedule_dispatch(region_id, time_s)
            continue

        generation = int(payload)
        scheduled = scheduled_dispatch.get(region_id)
        if scheduled is None or scheduled != (time_s, generation):
            continue
        scheduled_dispatch[region_id] = None
        queue = lb_queues[region_id]
        if not queue:
            continue
        context = queue[0]
        if context.finished:
            close_head(region_id, context, time_s)
            continue
        request = context.request
        observed_s = _probe_observation_time(time_s, config.probe_interval_s)
        local_candidates = _available_replicas(
            replicas_by_region[region_id],
            observed_s,
            config,
            enforce_queue_capacity_at_s=time_s,
        )
        local_candidate_ids = tuple(
            replica.replica_id for replica in sorted(
                local_candidates, key=lambda item: item.replica_id
            )
        )
        current_wait_s = max(0.0, time_s - context.current_lb_arrival_s)
        total_lb_wait_s = context.accumulated_lb_wait_s + current_wait_s
        audit: dict[str, Any] = {
            "event_ordering": "global_arrival_before_dispatch",
            "lb_region_id": region_id,
            "lb_queue_arrival_s": context.current_lb_arrival_s,
            "lb_queue_depth_at_dispatch": len(queue),
            "observed_probe_time_s": observed_s,
            "probe_interval_s": config.probe_interval_s,
            "available_local_replica_ids": local_candidate_ids,
            "remote_queue_buffer_requests_tau": config.remote_queue_buffer_requests,
            "prefix_key_hash": _stable_key_hash(request["prefix_key"]),
            "prefix_signature_kind": request["signature_kind"],
        }

        if local_candidates:
            selected_replica = _select_replica(
                local_candidates,
                request,
                time_s=time_s,
                config=config,
                round_robin_index=round_robin_index,
                all_replicas=replicas_by_region[region_id],
            )
            service_profile = _service_profile_for_replica(config, selected_replica)
            prefix_affinity = (
                _prefix_affinity_decision(
                    local_candidates, request, at_s=time_s, config=config
                )
                if config.routing_policy in {"prefix_tree", "prefix_tree_snapshot", "skywalker"}
                else None
            )
            reservation_tokens = (
                request["max_output_tokens"]
                if request.get("max_output_tokens") is not None
                else config.admission_output_reservation_tokens
            )
            output_reservation_exceeded = (
                request["output_tokens"] > max(0, int(reservation_tokens))
            )
            admission_tokens = max(
                1, request["prompt_tokens"] + max(0, int(reservation_tokens))
            )
            if admission_tokens > service_profile.continuous_batch_capacity_tokens:
                reason = "ground_request_exceeds_continuous_batch_capacity"
                network = _ground_network_delays(
                    config, context, serving_region_id=region_id
                )
                record = _reject_record(
                    request,
                    reason=reason,
                    lb_queue_wait_s=total_lb_wait_s,
                    policy=policy_name,
                    serving_region_id=region_id,
                    cross_region=context.forwarded,
                    network_delay_s=network["total_network_s"],
                    selected_path=_ground_path(
                        origin=context.origin_region_id,
                        serving_region=region_id,
                        replica_id=selected_replica.replica_id,
                        local_rtt_s=region_by_id[context.origin_region_id].local_rtt_s,
                        wan_rtt_s=context.wan_rtt_s,
                        ingress_region=network["ingress_region_id"],
                        client_to_ingress_s=network["client_to_ingress_s"],
                        ingress_to_serving_s=network["ingress_to_serving_s"],
                        response_to_client_s=network["response_to_client_s"],
                        dns_resolution_delay_s=request["dns_resolution_delay_s"],
                    ),
                )
                trace_audit = context.source_audit or audit
                trace = _decision_trace(
                    request=request,
                    record=record,
                    decision_time_s=time_s,
                    policy=policy_name,
                    candidate_count=len(local_candidates),
                    local_candidate_count=len(local_candidates),
                    remote_candidate_count=0,
                    selected_action={
                        "action_type": "ground_reject_drop",
                        "target_region_id": region_id,
                        "admission_tokens": admission_tokens,
                        "continuous_batch_capacity_tokens": service_profile.continuous_batch_capacity_tokens,
                        "service_profile": _service_profile_provenance(service_profile),
                    },
                    selection_reason=reason,
                    audit={**trace_audit, "target_lb_audit": audit},
                )
                close_head(region_id, context, time_s)
                append_outcome(
                    context,
                    record,
                    trace,
                    terminal_event_time_s=(
                        time_s
                        + network["response_to_client_s"]
                    ),
                )
                continue

            service_start_s = _earliest_replica_start(
                selected_replica,
                time_s,
                batch_tokens=admission_tokens,
                config=config,
                service_profile=service_profile,
            )
            if not math.isfinite(service_start_s):
                raise RuntimeError("ground admission accepted an unschedulable request")
            replica_pending_wait_s = max(0.0, service_start_s - time_s)
            active_at_start = selected_replica.active_jobs(service_start_s)
            contention_factor = 1.0 + service_profile.batch_contention_alpha * len(
                active_at_start
            )
            _compact_cache_as_of(
                selected_replica,
                at_s=time_s,
                max_entries=config.prefix_cache_max_entries,
                max_tokens=config.kv_cache_capacity_tokens,
            )
            hit_tokens, hit_ratio = _prefix_hit(
                selected_replica,
                request,
                at_s=service_start_s,
                max_entries=config.prefix_cache_max_entries,
                max_tokens=config.kv_cache_capacity_tokens,
            )
            miss_tokens = max(0, request["prompt_tokens"] - hit_tokens)
            prefill_service_s = (
                service_profile.first_token_overhead_s
                + miss_tokens / service_profile.prefill_tokens_per_s * contention_factor
            )
            decode_service_s = (
                request["output_tokens"]
                / service_profile.decode_tokens_per_s
                * contention_factor
            )
            first_token_ready_s = service_start_s + prefill_service_s
            completion_s = first_token_ready_s + decode_service_s
            selected_replica.jobs.append(
                _ReplicaJob(
                    request_id=request["request_id"],
                    enqueue_s=time_s,
                    start_s=service_start_s,
                    end_s=completion_s,
                    batch_tokens=admission_tokens,
                )
            )
            _update_routing_index(
                selected_replica,
                request,
                visible_at_s=time_s,
                max_entries=config.prefix_cache_max_entries,
            )
            _update_prefix_cache(
                selected_replica,
                request,
                available_at_s=completion_s,
                max_entries=config.prefix_cache_max_entries,
                max_tokens=config.kv_cache_capacity_tokens,
            )
            context.remote_lb_wait_s = current_wait_s if context.forwarded else 0.0
            if not context.forwarded:
                context.source_lb_wait_s = current_wait_s
            origin_config = region_by_id[context.origin_region_id]
            network = _ground_network_delays(
                config, context, serving_region_id=region_id
            )
            response_network_s = network["response_to_client_s"]
            network_delay_s = network["total_network_s"]
            ttft_s = (
                first_token_ready_s
                + response_network_s
                - request["arrival_time_s"]
            )
            e2e_s = completion_s + response_network_s - request["arrival_time_s"]
            timeout, timeout_reason = _timeout_status(
                ttft_s,
                e2e_s,
                request["ttft_deadline_s"],
                request["e2e_deadline_s"],
            )
            action_type = (
                "ground_cross_region_forward"
                if context.forwarded
                else "ground_local_compute"
            )
            selection_reason = (
                "local_unavailable_remote_available"
                if context.forwarded
                else "local_available"
            )
            selected_path = _ground_path(
                origin=context.origin_region_id,
                serving_region=region_id,
                replica_id=selected_replica.replica_id,
                local_rtt_s=origin_config.local_rtt_s,
                wan_rtt_s=context.wan_rtt_s,
                ingress_region=network["ingress_region_id"],
                client_to_ingress_s=network["client_to_ingress_s"],
                ingress_to_serving_s=network["ingress_to_serving_s"],
                response_to_client_s=network["response_to_client_s"],
                dns_resolution_delay_s=request["dns_resolution_delay_s"],
            )
            record = RuntimeRequestRecord(
                request_id=request["request_id"],
                arrival_time_s=request["arrival_time_s"],
                deployment_id=selected_replica.replica_id,
                action_type=action_type,
                accepted=True,
                completed=True,
                served=not timeout,
                rejected=False,
                timeout=timeout,
                timeout_reason=timeout_reason,
                prompt_tokens=request["prompt_tokens"],
                output_tokens=request["output_tokens"],
                ttft_deadline_s=request["ttft_deadline_s"],
                e2e_deadline_s=request["e2e_deadline_s"],
                prefill_queue_wait_s=total_lb_wait_s + replica_pending_wait_s,
                decode_queue_wait_s=0.0,
                link_queue_wait_s=0.0,
                link_transmission_s=network_delay_s,
                prefill_service_s=prefill_service_s,
                decode_service_s=decode_service_s,
                ttft_s=ttft_s,
                e2e_s=e2e_s,
                kv_hit_tokens=hit_tokens,
                kv_miss_tokens=miss_tokens,
                context_recompute_tokens=miss_tokens,
                session_action=(
                    "sticky_kv_owner" if hit_tokens > 0 else "recompute_context"
                ),
                selected_path=selected_path,
                ground_origin_region_id=context.origin_region_id,
                ground_serving_region_id=region_id,
                ground_replica_id=selected_replica.replica_id,
                ground_cross_region=context.forwarded,
                ground_network_delay_s=network_delay_s,
                ground_load_balancer_queue_wait_s=total_lb_wait_s,
                ground_replica_pending_wait_s=replica_pending_wait_s,
                ground_prefix_cache_hit_ratio=hit_ratio,
                bottleneck_reason=_bottleneck_reason(
                    lb_queue_wait_s=total_lb_wait_s,
                    replica_pending_wait_s=replica_pending_wait_s,
                    network_delay_s=network_delay_s,
                    prefill_service_s=prefill_service_s,
                    decode_service_s=decode_service_s,
                ),
                scope=TERRESTRIAL_GROUND_SCOPE,
                trace_model=TERRESTRIAL_NETWORK_TRACE,
                fallback_used=False,
                valid_for_paper_main_results=False,
                provenance={
                    "baseline_policy": policy_name,
                    "routing_policy": config.routing_policy,
                    "selective_pushing": config.selective_pushing,
                    "selection_reason": selection_reason,
                    "probe_interval_s": config.probe_interval_s,
                    "remote_queue_buffer_requests": config.remote_queue_buffer_requests,
                    "admission_output_reservation_tokens": reservation_tokens,
                    "admission_tokens": admission_tokens,
                    "realized_output_tokens_used_for_admission": False,
                    "realized_output_exceeded_admission_reservation": output_reservation_exceeded,
                    "service_profile": _service_profile_provenance(service_profile),
                    "batch_contention_alpha": service_profile.batch_contention_alpha,
                    "batch_contention_factor": contention_factor,
                    "wan_latency_profile": context.wan_latency_profile,
                    "ingress_resolution_source": request["ingress_resolution_source"],
                    "ingress_region_id": network["ingress_region_id"],
                    "ingress_to_serving_latency_profile": network[
                        "ingress_to_serving_profile"
                    ],
                    "response_latency_profile": network["response_to_client_profile"],
                    "active_sequences_at_service_start": len(active_at_start),
                    "routing_metadata_visible_at_dispatch": True,
                    "kv_reuse_visible_at_completion": True,
                    "prefix_key_hash": _stable_key_hash(request["prefix_key"]),
                    "hash_identity_source": request["hash_identity_source"],
                    "prefix_affinity": prefix_affinity,
                    "leo_access_links_used": False,
                    "leo_isl_links_used": False,
                    "mechanism_reproduction": True,
                    "paper_exact_numerical_reproduction": False,
                },
            )
            source_audit = context.source_audit or audit
            trace = _decision_trace(
                request=request,
                record=record,
                decision_time_s=time_s,
                policy=policy_name,
                candidate_count=(
                    int(source_audit.get("candidate_action_count", len(local_candidates)))
                ),
                local_candidate_count=(
                    int(source_audit.get("local_candidate_count", len(local_candidates)))
                ),
                remote_candidate_count=(
                    int(source_audit.get("remote_candidate_count", 0))
                ),
                selected_action={
                    "action_type": action_type,
                    "origin_region_id": context.origin_region_id,
                    "ingress_region_id": network["ingress_region_id"],
                    "serving_region_id": region_id,
                    "replica_id": selected_replica.replica_id,
                    "cross_region": context.forwarded,
                    "wan_rtt_s": context.wan_rtt_s,
                    "source_lb_queue_wait_s": context.source_lb_wait_s,
                    "remote_lb_queue_wait_s": context.remote_lb_wait_s,
                    "replica_pending_wait_s": replica_pending_wait_s,
                    "prefix_cache_hit_ratio": hit_ratio,
                    "prefix_affinity": prefix_affinity,
                    "admission_tokens": admission_tokens,
                    "batch_contention_factor": contention_factor,
                    "service_profile_id": service_profile.profile_id,
                    "first_token_overhead_s": service_profile.first_token_overhead_s,
                    "wan_latency_profile": context.wan_latency_profile,
                },
                selection_reason=selection_reason,
                audit={
                    **source_audit,
                    "target_lb_audit": audit if context.forwarded else None,
                    "selected_replica_id": selected_replica.replica_id,
                    "source_decision_time_s": context.source_decision_time_s,
                },
            )
            close_head(region_id, context, time_s)
            append_outcome(
                context,
                record,
                trace,
                terminal_event_time_s=(
                    time_s
                    + network["response_to_client_s"]
                ),
            )
            continue

        remote_regions: list[tuple[str, list[_ReplicaState]]] = []
        remote_candidate_count = 0
        remote_queue_depths: dict[str, int] = {}
        available_remote_replica_counts: dict[str, int] = {}
        remote_with_available_replicas: list[str] = []
        tau_filtered_regions: list[str] = []
        if not context.forwarded and policy_name == "skywalker_like_ground":
            (
                remote_regions,
                remote_queue_depths,
                available_remote_replica_counts,
                remote_with_available_replicas,
                tau_filtered_regions,
            ) = _remote_probe_candidates(
                source_region_id=region_id,
                replicas_by_region=replicas_by_region,
                lb_queue_intervals=lb_queue_intervals,
                observed_s=observed_s,
                config=config,
            )
            remote_candidate_count = len(remote_regions)
        audit.update(
            {
                "remote_lb_queue_depths_at_probe": remote_queue_depths,
                "available_remote_replica_counts_by_region": available_remote_replica_counts,
                "remote_regions_with_available_replicas": tuple(
                    remote_with_available_replicas
                ),
                "tau_filtered_remote_region_ids": tuple(tau_filtered_regions),
                "eligible_remote_region_ids": tuple(
                    region for region, _candidates in remote_regions
                ),
                "local_candidate_count": len(local_candidates),
                "remote_candidate_count": remote_candidate_count,
                "candidate_action_count": len(local_candidates)
                + remote_candidate_count,
            }
        )
        if remote_regions:
            remote_prefix_affinity = _remote_prefix_affinity_decision(
                remote_regions,
                request,
                origin_region=region_id,
                remote_snapshots=remote_snapshots,
                time_s=time_s,
                config=config,
            )
            target_region = _select_remote_region(
                remote_regions,
                request,
                origin_region=region_id,
                remote_snapshots=remote_snapshots,
                time_s=time_s,
                config=config,
                remote_lb_queue_depths=remote_queue_depths,
            )
            wan_rtt_s, wan_latency_profile = _wan_latency(
                config,
                region_by_id[region_id],
                region_by_id[target_region],
                request_key=request["request_id"],
            )
            context.source_decision_time_s = time_s
            context.source_lb_wait_s = current_wait_s
            context.accumulated_lb_wait_s += current_wait_s
            context.forwarded = True
            context.target_region_id = target_region
            context.wan_rtt_s = wan_rtt_s
            context.wan_latency_profile = wan_latency_profile
            context.source_audit = {
                **audit,
                "selected_remote_region_id": target_region,
                "remote_prefix_affinity": remote_prefix_affinity,
                "wan_latency_profile": wan_latency_profile,
                "selected_remote_snapshot_hit_ratio": _snapshot_hit_ratio(
                    remote_snapshots[region_id][target_region],
                    request,
                    at_s=time_s,
                ),
            }
            # The source LBSnapshotTrie records the routing decision now.  It
            # does not claim that any target-replica KV is complete.
            _update_remote_snapshot(
                remote_snapshots[region_id][target_region],
                request,
                available_at_s=time_s,
                max_entries=config.prefix_cache_max_entries,
            )
            close_head(region_id, context, time_s)
            remote_arrival_s = time_s + wan_rtt_s / 2.0
            push_event(
                remote_arrival_s,
                0,
                request["request_id"],
                "arrival",
                target_region,
                context,
            )
            continue

        wait_limit_reached = (
            config.max_lb_queue_wait_s is not None
            and total_lb_wait_s >= config.max_lb_queue_wait_s - 1e-12
        )
        if wait_limit_reached:
            reason = (
                "remote_ground_lb_queue_wait_limit"
                if context.forwarded
                else "ground_lb_queue_wait_limit"
            )
            network = _ground_network_delays(
                config, context, serving_region_id=region_id
            )
            record = _reject_record(
                request,
                reason=reason,
                lb_queue_wait_s=total_lb_wait_s,
                policy=policy_name,
                serving_region_id=region_id if context.forwarded else None,
                cross_region=context.forwarded,
                network_delay_s=network["total_network_s"],
                selected_path=(
                    _ground_path(
                        origin=context.origin_region_id,
                        serving_region=region_id,
                        replica_id=f"ground:{region_id}:unassigned",
                        local_rtt_s=region_by_id[context.origin_region_id].local_rtt_s,
                        wan_rtt_s=context.wan_rtt_s,
                        ingress_region=network["ingress_region_id"],
                        client_to_ingress_s=network["client_to_ingress_s"],
                        ingress_to_serving_s=network["ingress_to_serving_s"],
                        response_to_client_s=network["response_to_client_s"],
                        dns_resolution_delay_s=request["dns_resolution_delay_s"],
                    )
                    if context.forwarded
                    else ()
                ),
            )
            source_audit = context.source_audit or audit
            trace = _decision_trace(
                request=request,
                record=record,
                decision_time_s=time_s,
                policy=policy_name,
                candidate_count=int(source_audit.get("candidate_action_count", 0)),
                local_candidate_count=int(source_audit.get("local_candidate_count", 0)),
                remote_candidate_count=int(source_audit.get("remote_candidate_count", 0)),
                selected_action={
                    "action_type": "ground_reject_drop",
                    "target_region_id": region_id if context.forwarded else None,
                },
                selection_reason=reason,
                audit={**source_audit, "target_lb_audit": audit},
            )
            close_head(region_id, context, time_s)
            append_outcome(
                context,
                record,
                trace,
                terminal_event_time_s=time_s + network["response_to_client_s"],
            )
            continue

        next_probe_s = _next_probe_time(time_s, config.probe_interval_s)
        if config.max_lb_queue_wait_s is not None:
            remaining_wait_s = max(
                0.0,
                config.max_lb_queue_wait_s - context.accumulated_lb_wait_s,
            )
            wait_deadline_s = context.current_lb_arrival_s + remaining_wait_s
            next_probe_s = min(next_probe_s, wait_deadline_s)
        schedule_dispatch(region_id, next_probe_s)

    if len(outcomes) != len(contexts):
        unfinished = sorted(
            context.request["request_id"]
            for context in contexts
            if not context.finished
        )
        raise RuntimeError(
            "ground event heap ended with unfinished requests: " + ", ".join(unfinished)
        )

    outcomes.sort(key=lambda item: item[0])
    record_tuple = tuple(item[1] for item in outcomes)
    decision_traces = tuple(item[2] for item in outcomes)
    applied_profiles = _applied_profile_summaries(record_tuple)
    effective_horizon_s = _effective_horizon_s(record_tuple)
    client_harness_metrics: dict[str, Any] = {
        "evaluated": client_harness is not None,
        "mode": "trace_open_loop" if client_harness is None else "one_program_per_client",
    }
    if client_harness is not None:
        if any(value != 0 for value in client_active_programs.values()):
            raise RuntimeError("client harness ended with an in-flight program")
        program_count = sum(
            len(programs) for programs in client_harness.programs_by_client.values()
        )
        completed_program_count = sum(
            1
            for key, remaining in client_program_remaining.items()
            if remaining == 0 and client_program_terminal_s.get(key)
        )
        client_harness_metrics.update(
            {
                "client_count": len(client_harness.programs_by_client),
                "program_execution_model": client_harness.program_execution_model,
                "program_count": program_count,
                "released_program_count": client_released_program_count,
                "completed_program_count": completed_program_count,
                "program_completion_throughput_per_s": completed_program_count
                / max(1e-9, effective_horizon_s),
                "inter_program_think_time_s": client_harness.inter_program_think_time_s,
                "max_inflight_programs_per_client": max(
                    client_peak_inflight_programs.values(), default=0
                ),
                "inflight_program_constraint_satisfied": all(
                    value <= 1 for value in client_peak_inflight_programs.values()
                ),
                "next_program_release": "previous_program_all_requests_terminal_plus_configured_think_time",
            }
        )
    replica_metrics = _replica_metrics(
        replicas_by_region,
        effective_horizon_s=effective_horizon_s,
        config=config,
    )
    runtime_metrics = {
        **_ground_runtime_metrics(record_tuple),
        **_ground_completed_throughput_metrics(
            record_tuple, effective_horizon_s=effective_horizon_s
        ),
        **_replica_imbalance_metrics(replica_metrics),
        "ground_hash_identity_fallback_count": len(
            [
                item
                for item in normalized
                if item["hash_identity_source"] == "prefix_key_fallback"
            ]
        ),
        **({"client_program_completion_throughput_per_s": client_harness_metrics.get("program_completion_throughput_per_s")} if client_harness is not None else {}),
    }
    region_metrics = _region_metrics(record_tuple, config.regions)
    communication_metrics = {
        "evaluated": True,
        "model": "configurable_terrestrial_internet_rtt",
        "cross_region_request_fraction": runtime_metrics[
            "ground_cross_region_request_fraction"
        ],
        "network_delay_p50_s": runtime_metrics["ground_network_delay_p50_s"],
        "network_delay_p95_s": runtime_metrics["ground_network_delay_p95_s"],
        "network_delay_p99_s": runtime_metrics["ground_network_delay_p99_s"],
        "wan_rtt_source": "ground_config.wan_latency_profiles_or_wan_rtt_s_or_geometry_surrogate",
        "configured_wan_latency_profiles": _configured_execution_profiles(config)[
            "wan_latency_profiles"
        ],
        "applied_wan_latency_profiles": applied_profiles["wan_latency_profiles"],
        "warnings": (
            "Internet delay excludes packet loss, jitter, congestion, DNS, TLS, and egress pricing",
        ),
    }
    session_metrics = {
        "evaluated": True,
        "kv_hit_tokens": sum(record.kv_hit_tokens or 0 for record in record_tuple),
        "kv_miss_tokens": sum(record.kv_miss_tokens or 0 for record in record_tuple),
        "kv_hit_rate_tokens": runtime_metrics[
            "ground_prefix_cache_hit_rate_tokens"
        ],
        "model": "routing_metadata_at_dispatch_kv_reuse_at_completion",
        "client_harness": client_harness_metrics,
    }
    state_summary = {
        "request_count": len(record_tuple),
        "served_count": len([record for record in record_tuple if record.served]),
        "completed_count": len(
            [record for record in record_tuple if record.completed]
        ),
        "rejected_count": len([record for record in record_tuple if record.rejected]),
        "timeout_count": len([record for record in record_tuple if record.timeout]),
        "module_status": {
            "queue_model": "global_event_heap_regional_lb_fcfs_plus_replica_pending_queue",
            "link_model": "terrestrial_internet_rtt_surrogate",
            "energy_model": "not_applicable_ground_baseline",
            "kv_state": "routing_metadata_dispatch_kv_cache_completion",
            "routing_model": (
                "skywalker_two_layer_local_first"
                if policy_name == "skywalker_like_ground"
                else "ground_region_local_only"
            ),
            "batch_service_model": "profiled_admission_time_contention_surrogate",
        },
        "regions": [to_jsonable(item) for item in config.regions],
        "replicas": replica_metrics,
        "client_harness": client_harness_metrics,
        "remote_snapshot_entry_counts": {
            source: {target: len(snapshot) for target, snapshot in targets.items()}
            for source, targets in remote_snapshots.items()
        },
    }
    fidelity_warnings = [
        "SkyWalker mechanism reproduction uses a calibrated analytical token-service and deterministic Internet-RTT profile, not an inference-engine or cloud-network replay",
        "paper does not publish tau, exact WAN matrix, batching capacity, trie/vnode parameters, or trace preprocessing",
        "global event ordering preserves causal FCFS arrival order at every regional load balancer",
        "batch contention is a per-profile admission-time deterministic slowdown, not SGLang token-level continuous batching",
        "valid_for_paper_main_results=false",
    ]
    if any(
        bool(profile.get("is_surrogate", True))
        for profile in applied_profiles["service_profiles"]
        + applied_profiles["wan_latency_profiles"]
    ):
        fidelity_warnings.append(
            "one or more applied execution/WAN profiles are marked surrogate; inspect record provenance before interpreting absolute performance"
        )
    if any(
        profile.get("mode") in {"deterministic_samples", "trace_samples"}
        for profile in applied_profiles["wan_latency_profiles"]
    ):
        fidelity_warnings.append(
            "WAN sample profiles use deterministic request-key selection and do not model temporal congestion correlation"
        )
    if any(not item.get("explicit_prefix_key") for item in normalized):
        fidelity_warnings.append(
            "prompt token IDs/prefix hierarchy are unavailable for some requests; session/user exact keys provide a bounded prefix-trie surrogate"
        )
    if client_harness is not None:
        fidelity_warnings.append(
            "client_harness enforces at most one in-flight program per explicit client; program-internal request offsets remain concurrent"
        )
        if client_harness.inter_program_think_time_s == 0.0:
            fidelity_warnings.append(
                "client_harness uses configured zero inter-program think time; the paper does not publish a think-time distribution"
            )
    reservation_exceeded_count = sum(
        1
        for record in record_tuple
        if bool(
            record.provenance.get(
                "realized_output_exceeded_admission_reservation", False
            )
        )
    )
    if reservation_exceeded_count:
        fidelity_warnings.append(
            f"{reservation_exceeded_count} completed request(s) exceeded their declared/default output admission reservation; increase admission_output_reservation_tokens or provide max_output_tokens"
        )
    return GroundRuntimeSimulationResult(
        records=record_tuple,
        decision_traces=decision_traces,
        state_summary=state_summary,
        warnings=tuple(dict.fromkeys(fidelity_warnings)),
        runtime_metrics=runtime_metrics,
        communication_metrics=communication_metrics,
        energy_metrics={
            "evaluated": False,
            "not_applicable": True,
            "value": None,
            "warnings": (
                "orbital energy/SOC is not applicable to the terrestrial baseline",
            ),
        },
        session_metrics=session_metrics,
        region_metrics=region_metrics,
        provenance={
            "runtime_engine": "skywalker_ground_discrete_event_surrogate",
            "event_ordering": "single_global_heap_arrival_before_dispatch",
            "policy": policy_name,
            "paper": "SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference",
            "mechanism_reproduction": True,
            "numerical_reproduction": False,
            "supplied_ground_config": to_jsonable(config.raw),
            "effective_ground_config": _effective_ground_config(config),
            "execution_calibration": {
                **_configured_execution_profiles(config),
                "applied_profiles": applied_profiles,
            },
            "client_harness": client_harness_metrics,
            "effective_time_horizon_s": effective_horizon_s,
            "leo_trace_consumed": False,
            "leo_cost_consumed": False,
            "valid_for_paper_main_results": False,
        },
    )


def _profile_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _service_profile_from_payload(
    payload: Any,
    *,
    fallback: _ServiceProfile,
    default_profile_id: str,
) -> _ServiceProfile:
    """Merge a calibration payload over a profile without changing legacy defaults."""

    if payload is None:
        return fallback
    if not isinstance(payload, dict):
        raise ValueError("ground service profile entries must be mappings")
    data = dict(payload)
    slowdown_payload = data.get("batch_slowdown")
    if isinstance(slowdown_payload, dict):
        slowdown_value = _first_present(
            slowdown_payload.get("per_active_request_multiplier"),
            slowdown_payload.get("linear_active_request_multiplier"),
            data.get("batch_contention_alpha"),
            fallback.batch_contention_alpha,
        )
        slowdown_source = "batch_slowdown.per_active_request_multiplier"
    elif slowdown_payload is not None:
        slowdown_value = slowdown_payload
        slowdown_source = "batch_slowdown"
    else:
        slowdown_value = _first_present(
            data.get("batch_contention_alpha"), fallback.batch_contention_alpha
        )
        slowdown_source = (
            "batch_contention_alpha"
            if data.get("batch_contention_alpha") is not None
            else fallback.batch_slowdown_source
        )
    return _ServiceProfile(
        profile_id=str(data.get("profile_id") or default_profile_id),
        prefill_tokens_per_s=_positive_float(
            _first_present(data.get("prefill_tokens_per_s"), fallback.prefill_tokens_per_s),
            "service_profile.prefill_tokens_per_s",
        ),
        decode_tokens_per_s=_positive_float(
            _first_present(data.get("decode_tokens_per_s"), fallback.decode_tokens_per_s),
            "service_profile.decode_tokens_per_s",
        ),
        first_token_overhead_s=_nonnegative_float(
            _first_present(
                data.get("first_token_overhead_s"),
                data.get("fixed_first_token_overhead_s"),
                fallback.first_token_overhead_s,
            ),
            "service_profile.first_token_overhead_s",
        ),
        continuous_batch_capacity_tokens=_positive_int(
            _first_present(
                data.get("continuous_batch_capacity_tokens"),
                fallback.continuous_batch_capacity_tokens,
            ),
            fallback.continuous_batch_capacity_tokens,
        ),
        max_batch_size=_positive_int(
            _first_present(data.get("max_batch_size"), fallback.max_batch_size),
            fallback.max_batch_size,
        ),
        batch_contention_alpha=_nonnegative_float(
            slowdown_value, "service_profile.batch_slowdown"
        ),
        calibration_source=str(
            _first_present(
                data.get("calibration_source"), data.get("source"), fallback.calibration_source
            )
        ),
        is_surrogate=_profile_bool(
            _first_present(data.get("is_surrogate"), data.get("surrogate")),
            fallback.is_surrogate,
        ),
        batch_slowdown_source=slowdown_source,
    )


def _service_profile_provenance(profile: _ServiceProfile) -> dict[str, Any]:
    return {
        "profile_id": profile.profile_id,
        "prefill_tokens_per_s": profile.prefill_tokens_per_s,
        "decode_tokens_per_s": profile.decode_tokens_per_s,
        "first_token_overhead_s": profile.first_token_overhead_s,
        "continuous_batch_capacity_tokens": profile.continuous_batch_capacity_tokens,
        "max_batch_size": profile.max_batch_size,
        "batch_slowdown_per_active_request_multiplier": profile.batch_contention_alpha,
        "batch_slowdown_source": profile.batch_slowdown_source,
        "calibration_source": profile.calibration_source,
        "is_surrogate": profile.is_surrogate,
    }


def _wan_latency_profile_from_payload(
    payload: Any,
    *,
    default_profile_id: str,
    default_source: str,
) -> _WanLatencyProfile:
    """Parse a fixed RTT or deterministic samples without introducing RNG."""

    if isinstance(payload, (int, float)):
        return _WanLatencyProfile(
            profile_id=default_profile_id,
            mode="fixed_rtt",
            fixed_rtt_s=_nonnegative_float(payload, "wan_rtt_s"),
            samples_s=(),
            calibration_source=default_source,
            is_surrogate=True,
        )
    if not isinstance(payload, dict):
        raise ValueError("WAN latency profile entries must be numbers or mappings")
    data = dict(payload)
    distribution = data.get("distribution")
    if distribution is not None and not isinstance(distribution, dict):
        raise ValueError("WAN latency profile distribution must be a mapping")
    distribution = distribution or {}
    samples_payload = _first_present(
        data.get("trace_samples_s"),
        data.get("samples_s"),
        data.get("rtt_samples_s"),
        distribution.get("samples_s"),
        distribution.get("values_s"),
    )
    if samples_payload is not None:
        if not isinstance(samples_payload, (list, tuple)) or not samples_payload:
            raise ValueError("WAN latency samples must be a non-empty list")
        samples = tuple(
            _nonnegative_float(item, "wan_latency_profile.samples_s")
            for item in samples_payload
        )
        mode = str(data.get("mode") or ("trace_samples" if data.get("trace_samples_s") is not None else "deterministic_samples"))
        return _WanLatencyProfile(
            profile_id=str(data.get("profile_id") or default_profile_id),
            mode=mode,
            fixed_rtt_s=None,
            samples_s=samples,
            calibration_source=str(
                _first_present(data.get("calibration_source"), data.get("source"), default_source)
            ),
            is_surrogate=_profile_bool(
                _first_present(data.get("is_surrogate"), data.get("surrogate")), True
            ),
        )
    fixed = _first_present(data.get("rtt_s"), data.get("fixed_rtt_s"), data.get("value_s"))
    if fixed is None:
        raise ValueError("WAN latency profile requires rtt_s or deterministic samples")
    return _WanLatencyProfile(
        profile_id=str(data.get("profile_id") or default_profile_id),
        mode=str(data.get("mode") or "fixed_rtt"),
        fixed_rtt_s=_nonnegative_float(fixed, "wan_latency_profile.rtt_s"),
        samples_s=(),
        calibration_source=str(
            _first_present(data.get("calibration_source"), data.get("source"), default_source)
        ),
        is_surrogate=_profile_bool(
            _first_present(data.get("is_surrogate"), data.get("surrogate")), True
        ),
    )


def _resolve_config(
    runtime_strategy: Any,
    *,
    regions: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    requests: tuple[dict[str, Any], ...],
) -> _GroundConfig:
    strategy_payload = runtime_strategy if isinstance(runtime_strategy, dict) else {}
    raw = dict(
        strategy_payload.get("ground_config", {})
        if strategy_payload
        else getattr(runtime_strategy, "ground_config", {}) or {}
    )
    supplied_regions = tuple(item for item in raw.get("regions", ()) if isinstance(item, dict))
    explicit_regions = bool(supplied_regions)
    scenario_regions = {
        str(item.get("region_id")): dict(item)
        for item in regions
        if isinstance(item, dict) and item.get("region_id") is not None
    }
    region_rows = supplied_regions or tuple(scenario_regions.values())
    if not region_rows:
        inferred_ids = sorted(
            {
                str(
                    _first_present(
                        item.get("origin_region_id"),
                        item.get("user_region_id"),
                        item.get("region_id"),
                        item.get("ingress_id"),
                    )
                )
                for item in requests
                if _first_present(
                    item.get("origin_region_id"),
                    item.get("user_region_id"),
                    item.get("region_id"),
                    item.get("ingress_id"),
                )
                is not None
            }
        )
        region_rows = tuple({"region_id": region_id} for region_id in inferred_ids or ["ground_default"])
    seen: set[str] = set()
    resolved_regions: list[_RegionConfig] = []
    replicas_default = _positive_int(raw.get("replicas_per_region"), 4)
    replicas_mapping = raw.get("replicas_per_region") if isinstance(raw.get("replicas_per_region"), dict) else {}
    for row in region_rows:
        region_id = str(row.get("region_id") or "")
        if not region_id:
            raise ValueError("ground_config.regions entries require region_id")
        if region_id in seen:
            raise ValueError(f"duplicate ground region_id: {region_id}")
        seen.add(region_id)
        scenario_row = scenario_regions.get(region_id, {})
        replica_count = _positive_int(
            _first_present(row.get("replica_count"), replicas_mapping.get(region_id), replicas_default),
            replicas_default,
        )
        local_rtt_s = _nonnegative_float(
            _first_present(row.get("local_rtt_s"), raw.get("local_rtt_s"), 0.020),
            "local_rtt_s",
        )
        resolved_regions.append(
            _RegionConfig(
                region_id=region_id,
                replica_count=replica_count,
                local_rtt_s=local_rtt_s,
                lat_deg=_optional_float(_first_present(row.get("lat_deg"), scenario_row.get("lat_deg"))),
                lon_deg=_optional_float(_first_present(row.get("lon_deg"), scenario_row.get("lon_deg"))),
            )
        )
    wan_rtt_s: dict[str, dict[str, float]] = {}
    wan_latency_profiles: dict[str, dict[str, _WanLatencyProfile]] = {}
    for source, targets in (raw.get("wan_rtt_s") or {}).items():
        if not isinstance(targets, dict):
            raise ValueError("ground_config.wan_rtt_s must be a nested mapping")
        for target, value in targets.items():
            profile = _wan_latency_profile_from_payload(
                value,
                default_profile_id=f"wan:{source}->{target}",
                default_source="ground_config.wan_rtt_s",
            )
            source_id, target_id = str(source), str(target)
            wan_latency_profiles.setdefault(source_id, {})[target_id] = profile
            if profile.fixed_rtt_s is not None:
                wan_rtt_s.setdefault(source_id, {})[target_id] = profile.fixed_rtt_s
    configured_wan_profiles = raw.get("wan_latency_profiles") or {}
    if not isinstance(configured_wan_profiles, dict):
        raise ValueError("ground_config.wan_latency_profiles must be a nested mapping")
    for source, targets in configured_wan_profiles.items():
        if not isinstance(targets, dict):
            raise ValueError("ground_config.wan_latency_profiles must be a nested mapping")
        for target, value in targets.items():
            profile = _wan_latency_profile_from_payload(
                value,
                default_profile_id=f"wan:{source}->{target}",
                default_source="ground_config.wan_latency_profiles",
            )
            source_id, target_id = str(source), str(target)
            wan_latency_profiles.setdefault(source_id, {})[target_id] = profile
            if profile.fixed_rtt_s is not None:
                wan_rtt_s.setdefault(source_id, {})[target_id] = profile.fixed_rtt_s

    service = dict(raw.get("replica_token_service") or raw.get("service") or {})
    base_profile = _ServiceProfile(
        profile_id="default_service_profile",
        prefill_tokens_per_s=_positive_float(
            _first_present(service.get("prefill_tokens_per_s"), raw.get("prefill_tokens_per_s"), 512.0 / 0.300),
            "prefill_tokens_per_s",
        ),
        decode_tokens_per_s=_positive_float(
            _first_present(service.get("decode_tokens_per_s"), raw.get("decode_tokens_per_s"), 80.0),
            "decode_tokens_per_s",
        ),
        first_token_overhead_s=_nonnegative_float(
            _first_present(
                service.get("first_token_overhead_s"),
                raw.get("first_token_overhead_s"),
                0.0,
            ),
            "first_token_overhead_s",
        ),
        continuous_batch_capacity_tokens=_positive_int(
            _first_present(
                service.get("continuous_batch_capacity_tokens"),
                raw.get("continuous_batch_capacity_tokens"),
                131072,
            ),
            131072,
        ),
        max_batch_size=_positive_int(raw.get("max_batch_size"), 64),
        batch_contention_alpha=_nonnegative_float(
            raw.get("batch_contention_alpha", 1.0), "batch_contention_alpha"
        ),
        calibration_source="ground_config.legacy_token_service_defaults",
        is_surrogate=True,
        batch_slowdown_source="batch_contention_alpha",
    )
    service_profile_payload = raw.get("service_profile")
    if service_profile_payload is not None and not isinstance(service_profile_payload, dict):
        raise ValueError("ground_config.service_profile must be a mapping")
    default_profile = _service_profile_from_payload(
        service_profile_payload,
        fallback=base_profile,
        default_profile_id="default_service_profile",
    )
    service_profiles = raw.get("service_profiles") or {}
    if not isinstance(service_profiles, dict):
        raise ValueError("ground_config.service_profiles must be a mapping")
    default_profile = _service_profile_from_payload(
        service_profiles.get("default"),
        fallback=default_profile,
        default_profile_id=default_profile.profile_id,
    )
    region_profile_rows: dict[str, Any] = {}
    for profile_map in (
        raw.get("region_service_profiles"),
        service_profiles.get("regions"),
        service_profiles.get("by_region"),
    ):
        if profile_map is None:
            continue
        if not isinstance(profile_map, dict):
            raise ValueError("region service profiles must be a mapping")
        region_profile_rows.update({str(key): value for key, value in profile_map.items()})
    for row in region_rows:
        region_id = str(row.get("region_id") or "")
        if isinstance(row.get("service_profile"), dict):
            region_profile_rows[region_id] = row["service_profile"]
    replica_profile_rows: dict[str, Any] = {}
    for profile_map in (
        raw.get("replica_service_profiles"),
        service_profiles.get("replicas"),
        service_profiles.get("by_replica"),
    ):
        if profile_map is None:
            continue
        if not isinstance(profile_map, dict):
            raise ValueError("replica service profiles must be a mapping")
        replica_profile_rows.update({str(key): value for key, value in profile_map.items()})
    service_profiles_by_region = {
        region_id: _service_profile_from_payload(
            profile_payload,
            fallback=default_profile,
            default_profile_id=f"region:{region_id}",
        )
        for region_id, profile_payload in region_profile_rows.items()
    }
    service_profiles_by_replica = {
        replica_id: _service_profile_from_payload(
            profile_payload,
            fallback=service_profiles_by_region.get(
                replica_id.split(":")[1] if replica_id.startswith("ground:") and ":replica-" in replica_id else "",
                default_profile,
            ),
            default_profile_id=f"replica:{replica_id}",
        )
        for replica_id, profile_payload in replica_profile_rows.items()
    }
    kv_capacity = _positive_int(
        _first_present(service.get("kv_cache_capacity_tokens"), raw.get("kv_cache_capacity_tokens"), 131072),
        131072,
    )
    max_lb_queue_wait = _optional_float(raw.get("max_lb_queue_wait_s"))
    if max_lb_queue_wait is not None and max_lb_queue_wait < 0:
        raise ValueError("max_lb_queue_wait_s must be non-negative")
    prefix_affinity_min_similarity = _optional_float(
        raw.get("prefix_affinity_min_similarity")
    )
    if prefix_affinity_min_similarity is not None and not 0.0 <= prefix_affinity_min_similarity <= 1.0:
        raise ValueError("prefix_affinity_min_similarity must be between 0 and 1")
    prefix_low_affinity_fallback = str(
        raw.get("prefix_low_affinity_fallback", "none")
    ).strip().lower()
    if prefix_low_affinity_fallback not in {"none", "least_outstanding"}:
        raise ValueError(
            "prefix_low_affinity_fallback must be none or least_outstanding"
        )
    return _GroundConfig(
        regions=tuple(resolved_regions),
        wan_rtt_s=wan_rtt_s,
        wan_latency_profiles=wan_latency_profiles,
        probe_interval_s=_positive_float(raw.get("probe_interval_s", 0.100), "probe_interval_s"),
        remote_queue_buffer_requests=_nonnegative_int(raw.get("remote_queue_buffer_requests"), 1),
        routing_policy=str(raw.get("routing_policy", "prefix_tree")),
        selective_pushing=str(raw.get("selective_pushing", "pending_requests")),
        prefill_tokens_per_s=default_profile.prefill_tokens_per_s,
        decode_tokens_per_s=default_profile.decode_tokens_per_s,
        continuous_batch_capacity_tokens=default_profile.continuous_batch_capacity_tokens,
        kv_cache_capacity_tokens=kv_capacity,
        max_batch_size=default_profile.max_batch_size,
        max_replica_queue_requests=_positive_int(raw.get("max_replica_queue_requests"), 256),
        max_lb_queue_wait_s=max_lb_queue_wait,
        prefix_cache_max_entries=_positive_int(
            _first_present(raw.get("prefix_cache_max_entries"), raw.get("prefix_tree_max_records")),
            100000,
        ),
        max_outstanding_requests=_positive_int(raw.get("max_outstanding_requests"), 32),
        propagation_speed_km_s=_positive_float(raw.get("propagation_speed_km_s", 200000.0), "propagation_speed_km_s"),
        internet_path_stretch=_positive_float(raw.get("internet_path_stretch", 1.5), "internet_path_stretch"),
        internet_router_overhead_s=_nonnegative_float(raw.get("internet_router_overhead_s", 0.010), "internet_router_overhead_s"),
        arrival_time_scale=_positive_float(raw.get("arrival_time_scale", 1.0), "arrival_time_scale"),
        batch_contention_alpha=default_profile.batch_contention_alpha,
        admission_output_reservation_tokens=_nonnegative_int(
            raw.get("admission_output_reservation_tokens"), 256
        ),
        hash_virtual_nodes_per_target=_positive_int(
            raw.get("hash_virtual_nodes_per_target"), 128
        ),
        hash_seed=str(raw.get("hash_seed", "skywalker-ground-v1")),
        prefix_affinity_min_similarity=prefix_affinity_min_similarity,
        prefix_low_affinity_fallback=prefix_low_affinity_fallback,
        explicit_regions=explicit_regions,
        default_service_profile=default_profile,
        service_profiles_by_region=service_profiles_by_region,
        service_profiles_by_replica=service_profiles_by_replica,
        raw=raw,
    )


def _config_with_inferred_regions(config: _GroundConfig, region_ids: list[str]) -> _GroundConfig:
    inferred = tuple(
        _RegionConfig(region_id=region_id, replica_count=4, local_rtt_s=0.020)
        for region_id in region_ids
    )
    return _GroundConfig(**{**config.__dict__, "regions": (*config.regions, *inferred)})


def _normalize_request(
    raw: dict[str, Any],
    index: int,
    *,
    gateway_to_region: dict[str, str],
) -> dict[str, Any]:
    ingress = _optional_str(_first_present(raw.get("ingress_id"), raw.get("gateway_id")))
    origin = _optional_str(
        _first_present(
            raw.get("origin_region_id"),
            raw.get("user_region_id"),
            raw.get("region_id"),
        )
    )
    if origin is None and ingress is not None:
        origin = gateway_to_region.get(ingress, ingress)
    origin = origin or "ground_default"
    prompt_tokens = max(0, _int(_first_present(raw.get("prompt_tokens"), raw.get("input_tokens")), 0))
    output_tokens = max(0, _int(_first_present(raw.get("output_tokens"), raw.get("decode_tokens")), 0))
    session_id = _optional_str(raw.get("session_id"))
    user_id = _optional_str(_first_present(raw.get("user_id"), raw.get("hashed_user_id")))
    explicit_prefix_key = _optional_str(_first_present(raw.get("prefix_key"), raw.get("prompt_prefix_key")))
    prefix_key = explicit_prefix_key or session_id or user_id or f"request:{raw.get('request_id') or index}"
    hash_key = (
        f"user:{user_id or ''}|session:{session_id or ''}"
        if user_id is not None or session_id is not None
        else prefix_key
    )
    hash_identity_source = (
        "user_and_session"
        if user_id is not None and session_id is not None
        else "user_id"
        if user_id is not None
        else "session_id"
        if session_id is not None
        else "prefix_key_fallback"
    )
    raw_token_ids = raw.get("prompt_token_ids")
    raw_prefix_path = raw.get("prefix_path")
    if isinstance(raw_token_ids, (list, tuple)) and raw_token_ids:
        prefix_signature = tuple(str(item) for item in raw_token_ids)
        signature_kind = "prompt_token_ids"
    elif isinstance(raw_prefix_path, (list, tuple)) and raw_prefix_path:
        prefix_signature = tuple(str(item) for item in raw_prefix_path)
        signature_kind = "metadata_prefix_path"
    else:
        prefix_signature = (prefix_key,)
        signature_kind = "exact_key"
    return {
        "request_id": str(raw.get("request_id") or f"ground-req-{index:06d}"),
        "arrival_time_s": _float(_first_present(raw.get("arrival_time_s"), raw.get("time_s")), 0.0),
        "origin_region_id": origin,
        "resolved_ingress_region_id": _optional_str(
            _first_present(raw.get("resolved_ingress_region_id"), raw.get("lb_region_id"))
        ),
        "client_lat_deg": _optional_float(raw.get("client_lat_deg")),
        "client_lon_deg": _optional_float(raw.get("client_lon_deg")),
        "dns_eligible_region_ids": tuple(
            str(item)
            for item in (raw.get("dns_eligible_region_ids") or ())
            if str(item).strip()
        )
        if isinstance(raw.get("dns_eligible_region_ids"), (tuple, list))
        else (),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "max_output_tokens": _optional_int(
            _first_present(raw.get("max_output_tokens"), raw.get("output_token_budget"))
        ),
        "new_user_tokens": _optional_int(raw.get("new_user_tokens")),
        "context_tokens_before_turn": _optional_int(raw.get("context_tokens_before_turn")),
        "ttft_deadline_s": _optional_float(_first_present(raw.get("ttft_deadline_s"), raw.get("deadline_ttft_s"))),
        "e2e_deadline_s": _optional_float(_first_present(raw.get("e2e_deadline_s"), raw.get("deadline_s"))),
        "session_id": session_id,
        "turn_index": _optional_int(raw.get("turn_index")),
        "user_id": user_id,
        # ``client_id`` is deliberately not inferred from user/session ID:
        # the paper's client program model is an offered-load contract, not a
        # cache-affinity hint.  Closed-loop mode must fail closed without it.
        "client_id": _optional_str(raw.get("client_id")),
        "program_id": _optional_str(raw.get("program_id")),
        "program_sequence_index": _optional_int(
            _first_present(raw.get("program_sequence_index"), raw.get("program_index"))
        ),
        "program_release_offset_s": _optional_float(
            _first_present(raw.get("program_release_offset_s"), raw.get("release_offset_s"))
        ),
        "program_request_dependencies": _normalized_request_dependencies(raw),
        "prefix_key": prefix_key,
        "hash_key": hash_key,
        "hash_identity_source": hash_identity_source,
        "prefix_signature": prefix_signature,
        "signature_kind": signature_kind,
        "explicit_prefix_key": explicit_prefix_key,
        "prefix_tokens": _optional_int(raw.get("prefix_tokens")),
    }


def _resolve_client_harness(
    raw_config: dict[str, Any],
    requests: tuple[dict[str, Any], ...],
    *,
    arrival_time_scale: float,
) -> _ClientHarness | None:
    """Validate an optional one-program-per-client arrival contract.

    The original paper says that each client issues one *program* at a time.
    A program may be a concurrent tree (for example ToT siblings), so this
    helper serializes programs per client but never serializes requests inside
    one program.  It intentionally does not guess a client from ``user_id``
    or invent a think-time distribution.
    """

    payload = raw_config.get("client_harness")
    if payload is None or payload is False:
        return None
    if not isinstance(payload, dict):
        raise ValueError("ground_config.client_harness must be a mapping when enabled")
    mode = str(payload.get("mode") or "").strip().lower()
    if mode not in {"one_program_per_client", "closed_loop_program"}:
        raise ValueError(
            "ground_config.client_harness.mode must be one_program_per_client"
        )
    execution_model = str(payload.get("program_execution_model") or "").strip().lower()
    if execution_model not in {"parallel_bag", "dependency_dag"}:
        raise ValueError(
            "client_harness.program_execution_model must be parallel_bag or dependency_dag; "
            "do not infer dialogue/ToT dependencies from timestamps"
        )
    if abs(arrival_time_scale - 1.0) > 1e-12:
        raise ValueError(
            "client_harness is incompatible with arrival_time_scale != 1; "
            "provide an uncompressed client/program trace"
        )
    think_time_s = _nonnegative_float(
        _first_present(
            payload.get("inter_program_think_time_s"),
            payload.get("think_time_s"),
            0.0,
        ),
        "client_harness.inter_program_think_time_s",
    )
    by_client: dict[str, dict[int, list[dict[str, Any]]]] = {}
    program_ids: dict[tuple[str, int], str] = {}
    for request in requests:
        client_id = _optional_str(request.get("client_id"))
        program_id = _optional_str(request.get("program_id"))
        sequence_index = request.get("program_sequence_index")
        if client_id is None or program_id is None or sequence_index is None:
            raise ValueError(
                "client_harness requires explicit client_id, program_id, and "
                "program_sequence_index on every request; user_id/session_id "
                "cannot be used as a substitute"
            )
        if int(sequence_index) < 0:
            raise ValueError("client_harness program_sequence_index must be non-negative")
        key = (client_id, int(sequence_index))
        existing = program_ids.get(key)
        if existing is not None and existing != program_id:
            raise ValueError(
                "client_harness maps one client/program_sequence_index to multiple program_ids: "
                f"{client_id!r}, {sequence_index!r}"
            )
        program_ids[key] = program_id
        by_client.setdefault(client_id, {}).setdefault(int(sequence_index), []).append(request)

    programs_by_client: dict[str, tuple[_ClientProgram, ...]] = {}
    for client_id, by_sequence in sorted(by_client.items()):
        programs: list[_ClientProgram] = []
        for sequence_index, members in sorted(by_sequence.items()):
            first_release_s = min(float(item["arrival_time_s"]) for item in members)
            offsets: dict[str, float] = {}
            member_ids = {item["request_id"] for item in members}
            dependencies: dict[str, tuple[str, ...]] = {}
            for item in members:
                explicit_offset = item.get("program_release_offset_s")
                offset_s = (
                    float(explicit_offset)
                    if explicit_offset is not None
                    else float(item["arrival_time_s"]) - first_release_s
                )
                if offset_s < -1e-12:
                    raise ValueError("client_harness program request release offset must be non-negative")
                offsets[item["request_id"]] = max(0.0, offset_s)
                deps = tuple(item.get("program_request_dependencies") or ())
                if execution_model == "parallel_bag" and deps:
                    raise ValueError(
                        "parallel_bag client harness cannot include request dependencies; "
                        "use program_execution_model=dependency_dag"
                    )
                unknown_deps = sorted(set(deps) - member_ids)
                if unknown_deps:
                    raise ValueError(
                        "client_harness dependency must refer to a request in the same program: "
                        + ", ".join(unknown_deps)
                    )
                if item["request_id"] in deps:
                    raise ValueError("client_harness request cannot depend on itself")
                dependencies[item["request_id"]] = tuple(sorted(set(deps)))
            if execution_model == "dependency_dag":
                _validate_program_dependency_dag(dependencies)
            programs.append(
                _ClientProgram(
                    client_id=client_id,
                    program_id=program_ids[(client_id, sequence_index)],
                    sequence_index=sequence_index,
                    request_ids=tuple(sorted(item["request_id"] for item in members)),
                    request_release_offsets_s=offsets,
                    request_dependencies=dependencies,
                    initial_release_s=first_release_s,
                )
            )
        programs_by_client[client_id] = tuple(programs)
    return _ClientHarness(
        inter_program_think_time_s=think_time_s,
        program_execution_model=execution_model,
        programs_by_client=programs_by_client,
    )


def _normalized_request_dependencies(raw: dict[str, Any]) -> tuple[str, ...]:
    payload = _first_present(
        raw.get("program_request_dependencies"),
        raw.get("depends_on_request_ids"),
        raw.get("parent_request_id"),
    )
    if payload is None:
        return ()
    if isinstance(payload, str):
        values = (payload,)
    elif isinstance(payload, (tuple, list)):
        values = tuple(payload)
    else:
        raise ValueError("program request dependencies must be a request_id or list of request_ids")
    result = tuple(str(item).strip() for item in values if str(item).strip())
    if len(set(result)) != len(result):
        raise ValueError("program request dependencies must not repeat a request_id")
    return result


def _resolve_ground_ingress(
    request: dict[str, Any],
    *,
    config: _GroundConfig,
    region_by_id: dict[str, _RegionConfig],
) -> dict[str, Any]:
    """Resolve the first regional LB under an explicit GeoDNS surrogate.

    ``pre_resolved_ingress`` preserves the older workload-supplied origin
    mapping.  ``nearest_healthy_region`` is deliberately fail-closed: it
    requires client (or origin-region) coordinates and explicit healthy LB
    candidates, instead of silently claiming Route53 behavior from a region
    label alone.
    """

    dns = config.raw.get("dns_ingress") or {}
    if not isinstance(dns, dict):
        raise ValueError("ground_config.dns_ingress must be a mapping")
    mode = str(dns.get("mode") or "pre_resolved_ingress").strip().lower()
    origin_id = request["origin_region_id"]
    if mode == "pre_resolved_ingress":
        ingress_id = request.get("resolved_ingress_region_id") or origin_id
        if ingress_id not in region_by_id:
            raise ValueError(
                f"pre-resolved ingress {ingress_id!r} is not a configured ground region"
            )
        source = "workload_pre_resolved_ingress"
    elif mode == "nearest_healthy_region":
        configured_healthy = dns.get("healthy_region_ids")
        healthy = (
            {str(item) for item in configured_healthy}
            if isinstance(configured_healthy, (tuple, list))
            else set(region_by_id)
        )
        eligible = set(request.get("dns_eligible_region_ids") or ()) or set(region_by_id)
        candidates = sorted(healthy & eligible & set(region_by_id))
        if not candidates:
            raise ValueError("GeoDNS has no healthy eligible regional LB")
        origin = region_by_id[origin_id]
        client_lat = request.get("client_lat_deg")
        client_lon = request.get("client_lon_deg")
        if client_lat is None or client_lon is None:
            client_lat, client_lon = origin.lat_deg, origin.lon_deg
        if client_lat is None or client_lon is None:
            raise ValueError(
                "nearest_healthy_region requires request client_lat_deg/client_lon_deg "
                "or coordinates for its origin region"
            )
        missing_coordinates = [
            region_id
            for region_id in candidates
            if region_by_id[region_id].lat_deg is None
            or region_by_id[region_id].lon_deg is None
        ]
        if missing_coordinates:
            raise ValueError(
                "nearest_healthy_region requires lat_deg/lon_deg for every eligible LB: "
                + ", ".join(missing_coordinates)
            )
        ingress_id = min(
            candidates,
            key=lambda region_id: (
                _great_circle_km(
                    float(client_lat),
                    float(client_lon),
                    float(region_by_id[region_id].lat_deg),
                    float(region_by_id[region_id].lon_deg),
                ),
                region_id,
            ),
        )
        source = "geometry_nearest_healthy_region_surrogate"
    else:
        raise ValueError(
            "dns_ingress.mode must be pre_resolved_ingress or nearest_healthy_region"
        )
    resolution_delay_s = _nonnegative_float(
        dns.get("resolution_delay_s", 0.0), "dns_ingress.resolution_delay_s"
    )
    return {
        **request,
        "ingress_region_id": str(ingress_id),
        "dns_resolution_delay_s": resolution_delay_s,
        "ingress_resolution_source": source,
    }


def _validate_program_dependency_dag(dependencies: dict[str, tuple[str, ...]]) -> None:
    # Kahn's algorithm also handles very long multi-parent client programs.
    from collections import deque
    degree = {request: len(parents) for request, parents in dependencies.items()}
    successors = {request: [] for request in dependencies}
    for request, parents in dependencies.items():
        for parent in parents:
            if parent not in successors:
                raise ValueError('client_harness program dependency references an unknown request')
            successors[parent].append(request)
    ready = deque(request for request, count in degree.items() if count == 0)
    visited = 0
    while ready:
        visited += 1
        for child in successors[ready.popleft()]:
            degree[child] -= 1
            if degree[child] == 0:
                ready.append(child)
    if visited != len(dependencies):
        raise ValueError('client_harness program dependency graph contains a cycle')


def _gateway_region_map(gateways: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in gateways or ():
        if not isinstance(item, dict):
            continue
        gateway_id = _optional_str(item.get("gateway_id"))
        served = tuple(item.get("served_region_ids") or ())
        if gateway_id and served:
            result[gateway_id] = str(served[0])
    return result


def _service_profile_for_replica(
    config: _GroundConfig, replica: _ReplicaState
) -> _ServiceProfile:
    return config.service_profiles_by_replica.get(
        replica.replica_id,
        config.service_profiles_by_region.get(
            replica.region_id, config.default_service_profile
        ),
    )


def _available_replicas(
    replicas: Any,
    observed_s: float,
    config: _GroundConfig,
    *,
    enforce_queue_capacity_at_s: float | None = None,
) -> list[_ReplicaState]:
    result: list[_ReplicaState] = []
    for replica in replicas:
        if (
            enforce_queue_capacity_at_s is not None
            and len(replica.pending_jobs(enforce_queue_capacity_at_s))
            >= config.max_replica_queue_requests
        ):
            continue
        pending = len(replica.pending_jobs(observed_s))
        outstanding = replica.outstanding_count(observed_s)
        if pending >= config.max_replica_queue_requests:
            continue
        mode = config.selective_pushing
        if mode == "pending_requests":
            available = pending == 0
        elif mode in {"max_outstanding", "outstanding_requests"}:
            available = outstanding < config.max_outstanding_requests
        elif mode in {"blind", "blind_pushing"}:
            available = True
        else:
            raise ValueError(f"unsupported ground selective_pushing mode: {mode!r}")
        if available:
            result.append(replica)
    return result


def _remote_probe_candidates(
    *,
    source_region_id: str,
    replicas_by_region: dict[str, list[_ReplicaState]],
    lb_queue_intervals: dict[str, list[list[float]]],
    observed_s: float,
    config: _GroundConfig,
    admission_tokens: int | None = None,
) -> tuple[
    list[tuple[str, list[_ReplicaState]]],
    dict[str, int],
    dict[str, int],
    list[str],
    list[str],
]:
    eligible: list[tuple[str, list[_ReplicaState]]] = []
    queue_depths: dict[str, int] = {}
    available_counts: dict[str, int] = {}
    regions_with_available_replicas: list[str] = []
    tau_filtered: list[str] = []
    for target_region in sorted(replicas_by_region):
        if target_region == source_region_id:
            continue
        candidates = _available_replicas(
            replicas_by_region[target_region], observed_s, config
        )
        if admission_tokens is not None:
            candidates = [
                replica
                for replica in candidates
                if admission_tokens
                <= _service_profile_for_replica(
                    config, replica
                ).continuous_batch_capacity_tokens
            ]
        queue_depth = _lb_queue_depth(
            lb_queue_intervals.get(target_region, ()), observed_s
        )
        queue_depths[target_region] = queue_depth
        available_counts[target_region] = len(candidates)
        if candidates:
            regions_with_available_replicas.append(target_region)
            if queue_depth <= config.remote_queue_buffer_requests:
                eligible.append((target_region, candidates))
            else:
                tau_filtered.append(target_region)
    return (
        eligible,
        queue_depths,
        available_counts,
        regions_with_available_replicas,
        tau_filtered,
    )


def _select_replica(
    replicas: list[_ReplicaState],
    request: dict[str, Any],
    *,
    time_s: float,
    config: _GroundConfig,
    round_robin_index: dict[str, int],
    all_replicas: list[_ReplicaState] | None = None,
) -> _ReplicaState:
    ordered = sorted(replicas, key=lambda replica: replica.replica_id)
    policy = config.routing_policy
    if policy in {"consistent_hash", "skywalker_ch", "ch"}:
        all_targets = sorted(
            (all_replicas or replicas), key=lambda replica: replica.replica_id
        )
        target_id = _consistent_hash_target(
            request["hash_key"],
            all_target_ids=[replica.replica_id for replica in all_targets],
            available_target_ids={replica.replica_id for replica in ordered},
            config=config,
        )
        return next(replica for replica in all_targets if replica.replica_id == target_id)
    if policy in {"round_robin", "rr"}:
        region_id = ordered[0].region_id
        index = round_robin_index.get(region_id, 0) % len(ordered)
        round_robin_index[region_id] = index + 1
        return ordered[index]
    if policy in {"least_load", "ll"}:
        return min(ordered, key=lambda replica: (replica.outstanding_count(time_s), replica.replica_id))
    if policy not in {"prefix_tree", "prefix_tree_snapshot", "skywalker"}:
        raise ValueError(f"unsupported ground routing_policy: {policy!r}")
    affinity = _prefix_affinity_decision(ordered, request, at_s=time_s, config=config)
    if affinity["fallback_used"]:
        return min(
            ordered,
            key=lambda replica: (
                replica.outstanding_count(time_s),
                replica.replica_id,
            ),
        )
    return min(
        ordered,
        key=lambda replica: (
            *_prefix_routing_sort_key(
                replica.routing_index, request, at_s=time_s
            ),
            replica.outstanding_count(time_s),
            replica.replica_id,
        ),
    )


def _prefix_affinity_decision(
    replicas: list[_ReplicaState],
    request: dict[str, Any],
    *,
    at_s: float,
    config: _GroundConfig,
) -> dict[str, Any]:
    """Make the low-affinity load-balance heuristic explicit and auditable.

    Algorithm 1 publishes maximum-prefix routing but does not publish a fixed
    similarity threshold.  The paper's Figure 8 discussion gives ``<50%`` as
    an explanatory example.  Consequently the strict default is ``none``;
    opting into a threshold is an experiment sensitivity, not a paper fact.
    """

    similarities = [
        _snapshot_prefix_score(replica.routing_index, request, at_s=at_s)[1]
        for replica in replicas
    ]
    best_similarity = max(similarities, default=0.0)
    threshold = config.prefix_affinity_min_similarity
    fallback_used = bool(
        threshold is not None
        and best_similarity < threshold
        and config.prefix_low_affinity_fallback == "least_outstanding"
    )
    return {
        "best_similarity": best_similarity,
        "threshold": threshold,
        "fallback_policy": config.prefix_low_affinity_fallback,
        "fallback_used": fallback_used,
    }


def _select_remote_region(
    remote_regions: list[tuple[str, list[_ReplicaState]]],
    request: dict[str, Any],
    *,
    origin_region: str,
    remote_snapshots: dict[str, dict[str, OrderedDict[str, _CacheEntry]]],
    time_s: float,
    config: _GroundConfig,
    remote_lb_queue_depths: dict[str, int] | None = None,
) -> str:
    if config.routing_policy in {"consistent_hash", "skywalker_ch", "ch"}:
        return _consistent_hash_target(
            request["hash_key"],
            all_target_ids=[region.region_id for region in config.regions],
            available_target_ids={region_id for region_id, _ in remote_regions},
            config=config,
            excluded_target_ids={origin_region},
        )
    remote_affinity = _remote_prefix_affinity_decision(
        remote_regions,
        request,
        origin_region=origin_region,
        remote_snapshots=remote_snapshots,
        time_s=time_s,
        config=config,
    )
    if remote_affinity["fallback_used"]:
        return min(
            remote_regions,
            key=lambda item: (
                int((remote_lb_queue_depths or {}).get(item[0], 0)),
                -len(item[1]),
                item[0],
            ),
        )[0]
    # Source LBs only consume their partial LBSnapshotTrie plus the published
    # remote probe payload (LB queue depth and available-replica count).  Do
    # not peek at exact remote replica outstanding counts here.
    return min(
        remote_regions,
        key=lambda item: (
            *_prefix_routing_sort_key(
                remote_snapshots.get(origin_region, {}).get(
                    item[0], OrderedDict()
                ),
                request,
                at_s=time_s,
            ),
            int((remote_lb_queue_depths or {}).get(item[0], 0)),
            -len(item[1]),
            item[0],
        ),
    )[0]


def _remote_prefix_affinity_decision(
    remote_regions: list[tuple[str, list[_ReplicaState]]],
    request: dict[str, Any],
    *,
    origin_region: str,
    remote_snapshots: dict[str, dict[str, OrderedDict[str, _CacheEntry]]],
    time_s: float,
    config: _GroundConfig,
) -> dict[str, Any]:
    similarities = [
        _snapshot_prefix_score(
            remote_snapshots.get(origin_region, {}).get(region_id, OrderedDict()),
            request,
            at_s=time_s,
        )[1]
        for region_id, _replicas in remote_regions
    ]
    best_similarity = max(similarities, default=0.0)
    threshold = config.prefix_affinity_min_similarity
    return {
        "best_similarity": best_similarity,
        "threshold": threshold,
        "fallback_policy": config.prefix_low_affinity_fallback,
        "fallback_used": bool(
            threshold is not None
            and best_similarity < threshold
            and config.prefix_low_affinity_fallback == "least_outstanding"
        ),
    }


def _earliest_replica_start(
    replica: _ReplicaState,
    enqueue_s: float,
    *,
    batch_tokens: int,
    config: _GroundConfig,
    service_profile: _ServiceProfile | None = None,
) -> float:
    service_profile = service_profile or _service_profile_for_replica(config, replica)
    required_tokens = max(1, batch_tokens)
    if required_tokens > service_profile.continuous_batch_capacity_tokens:
        return math.inf
    pending_ahead = [job for job in replica.pending_jobs(enqueue_s)]
    candidate = max([enqueue_s, *(job.start_s for job in pending_ahead)])
    event_times = sorted(
        {
            candidate,
            *(
                value
                for job in replica.jobs
                for value in (job.start_s, job.end_s)
                if value >= candidate - 1e-12
            ),
        }
    )
    for time_s in event_times:
        active = replica.active_jobs(time_s)
        if (
            len(active) < service_profile.max_batch_size
            and sum(job.batch_tokens for job in active) + required_tokens
            <= service_profile.continuous_batch_capacity_tokens
        ):
            return max(candidate, time_s)
    time_s = max([candidate, *(job.end_s for job in replica.jobs)], default=candidate)
    return time_s


def _prefix_hit(
    replica: _ReplicaState,
    request: dict[str, Any],
    *,
    at_s: float,
    max_entries: int | None = None,
    max_tokens: int | None = None,
) -> tuple[int, float]:
    best_entry: _CacheEntry | None = None
    best_common = 0
    request_signature = tuple(request.get("prefix_signature") or (request["prefix_key"],))
    for entry in _ready_cache_entries(
        replica.cache,
        at_s=at_s,
        max_entries=max_entries,
        max_tokens=max_tokens,
    ):
        common = _common_prefix_length(request_signature, entry.prefix_signature)
        if common > best_common:
            best_common = common
            best_entry = entry
    if best_entry is None or best_common <= 0:
        return 0, 0.0
    desired = _first_present(
        request.get("prefix_tokens"),
        request.get("context_tokens_before_turn"),
        request.get("prompt_tokens"),
    )
    if (
        request.get("signature_kind") == "prompt_token_ids"
        and best_entry.signature_kind == "prompt_token_ids"
    ):
        estimated_hit = best_common
    elif best_common == len(request_signature) == len(best_entry.prefix_signature):
        estimated_hit = max(0, _int(desired, 0))
    else:
        estimated_hit = int(max(0, _int(desired, 0)) * best_common / max(1, len(request_signature)))
    hit_tokens = min(estimated_hit, best_entry.tokens, request["prompt_tokens"])
    ratio = hit_tokens / request["prompt_tokens"] if request["prompt_tokens"] > 0 else 0.0
    return hit_tokens, ratio


def _update_prefix_cache(
    replica: _ReplicaState,
    request: dict[str, Any],
    *,
    available_at_s: float,
    max_entries: int,
    max_tokens: int,
) -> None:
    # Keep temporal versions.  Replacing a ready entry with a newer request's
    # future completion would incorrectly hide reusable KV in the interval
    # between the two completions.
    prefix_key = request["prefix_key"]
    key = f"{prefix_key}|{request['request_id']}|{len(replica.cache)}"
    replica.cache[key] = _CacheEntry(
        tokens=max(request["prompt_tokens"] + request["output_tokens"], request["prompt_tokens"]),
        available_at_s=available_at_s,
        prefix_signature=tuple(request.get("prefix_signature") or (prefix_key,)),
        signature_kind=str(request.get("signature_kind") or "exact_key"),
    )
    replica.cache.move_to_end(key)
    # Capacity is applied lazily by ``_ready_cache_entries`` at query time.
    # A future completion must not evict a currently ready entry early.
    del max_entries, max_tokens


def _ready_cache_entries(
    cache: OrderedDict[str, _CacheEntry],
    *,
    at_s: float,
    max_entries: int | None,
    max_tokens: int | None,
) -> tuple[_CacheEntry, ...]:
    # OrderedDict order is dispatch insertion order.  Capacity eviction drops
    # the earliest inserted ready record, matching SkyWalker's documented
    # bounded-trie rule instead of reordering by completion time.
    ready = tuple(
        (key, entry)
        for key, entry in cache.items()
        if entry.available_at_s <= at_s + 1e-12
    )
    if max_entries is None and max_tokens is None:
        return tuple(entry for _key, entry in ready)
    retained: OrderedDict[str, _CacheEntry] = OrderedDict()
    token_count = 0
    for key, entry in ready:
        retained[key] = entry
        token_count += entry.tokens
        while retained and (
            (max_entries is not None and len(retained) > max_entries)
            or (max_tokens is not None and token_count > max_tokens)
        ):
            _old_key, old_entry = retained.popitem(last=False)
            token_count -= old_entry.tokens
    return tuple(retained.values())


def _compact_cache_as_of(
    replica: _ReplicaState,
    *,
    at_s: float,
    max_entries: int,
    max_tokens: int,
) -> None:
    """Drop only entries already evicted by the current causal time.

    Dispatch may predict a service start in the future, so compaction must use
    the global event time rather than that future start.  Future completions
    remain alongside a bounded ready-cache view.
    """

    retained = {
        id(entry)
        for entry in _ready_cache_entries(
            replica.cache,
            at_s=at_s,
            max_entries=max_entries,
            max_tokens=max_tokens,
        )
    }
    for key, entry in tuple(replica.cache.items()):
        if entry.available_at_s <= at_s + 1e-12 and id(entry) not in retained:
            del replica.cache[key]


def _update_routing_index(
    replica: _ReplicaState,
    request: dict[str, Any],
    *,
    visible_at_s: float,
    max_entries: int,
) -> None:
    """Publish a routing-trie entry without claiming the KV is ready."""

    key = request["prefix_key"]
    replica.routing_index[key] = _CacheEntry(
        tokens=max(request["prompt_tokens"], 0),
        available_at_s=visible_at_s,
        prefix_signature=tuple(request.get("prefix_signature") or (key,)),
        signature_kind=str(request.get("signature_kind") or "exact_key"),
    )
    replica.routing_index.move_to_end(key)
    while len(replica.routing_index) > max_entries:
        replica.routing_index.popitem(last=False)


def _update_remote_snapshot(
    snapshot: OrderedDict[str, _CacheEntry],
    request: dict[str, Any],
    *,
    available_at_s: float,
    max_entries: int,
) -> None:
    key = request["prefix_key"]
    snapshot[key] = _CacheEntry(
        tokens=max(request["prompt_tokens"] + request["output_tokens"], request["prompt_tokens"]),
        available_at_s=available_at_s,
        prefix_signature=tuple(request.get("prefix_signature") or (key,)),
        signature_kind=str(request.get("signature_kind") or "exact_key"),
    )
    snapshot.move_to_end(key)
    while len(snapshot) > max_entries:
        snapshot.popitem(last=False)


def _snapshot_hit_ratio(
    snapshot: OrderedDict[str, _CacheEntry],
    request: dict[str, Any],
    *,
    at_s: float,
) -> float:
    return _snapshot_prefix_score(snapshot, request, at_s=at_s)[1]


def _snapshot_prefix_score(
    snapshot: OrderedDict[str, _CacheEntry],
    request: dict[str, Any],
    *,
    at_s: float,
) -> tuple[int, float]:
    request_signature = tuple(request.get("prefix_signature") or (request["prefix_key"],))
    best_common = 0
    best_similarity = 0.0
    for entry in snapshot.values():
        if entry.available_at_s > at_s + 1e-12:
            continue
        common = _common_prefix_length(request_signature, entry.prefix_signature)
        similarity = common / max(
            1, min(len(request_signature), len(entry.prefix_signature))
        )
        if (common, similarity) > (best_common, best_similarity):
            best_common, best_similarity = common, similarity
    return best_common, best_similarity


def _prefix_routing_sort_key(
    snapshot: OrderedDict[str, _CacheEntry],
    request: dict[str, Any],
    *,
    at_s: float,
) -> tuple[float, float]:
    common, similarity = _snapshot_prefix_score(snapshot, request, at_s=at_s)
    return -float(common), -similarity


def _common_prefix_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def _lb_queue_depth(intervals: Any, time_s: float) -> int:
    return sum(1 for arrival_s, dispatch_s in intervals if arrival_s <= time_s < dispatch_s)


def _next_ground_state_change(
    replicas_by_region: dict[str, list[_ReplicaState]],
    time_s: float,
) -> float | None:
    candidates = [
        value
        for replicas in replicas_by_region.values()
        for replica in replicas
        for value in (replica.next_state_change_after(time_s),)
        if value is not None
    ]
    return min(candidates) if candidates else None


def _wan_latency(
    config: _GroundConfig,
    source: _RegionConfig,
    target: _RegionConfig,
    *,
    request_key: str | None = None,
) -> tuple[float, dict[str, Any]]:
    profile = config.wan_latency_profiles.get(source.region_id, {}).get(target.region_id)
    direction = "configured_direct"
    if profile is None:
        profile = config.wan_latency_profiles.get(target.region_id, {}).get(source.region_id)
        direction = "configured_reverse_symmetric" if profile is not None else ""
    if profile is not None:
        if profile.fixed_rtt_s is not None:
            rtt_s = profile.fixed_rtt_s
            sample_index = None
        else:
            sample_key = (
                f"{config.hash_seed}|wan|{source.region_id}|{target.region_id}|"
                f"{request_key or 'effective-config'}"
            )
            sample_index = _stable_hash_index(sample_key, len(profile.samples_s))
            rtt_s = profile.samples_s[sample_index]
        return rtt_s, {
            "profile_id": profile.profile_id,
            "mode": profile.mode,
            "calibration_source": profile.calibration_source,
            "is_surrogate": profile.is_surrogate,
            "selection": "fixed" if sample_index is None else "deterministic_hash_sample",
            "sample_index": sample_index,
            "sample_count": len(profile.samples_s),
            "rtt_s": rtt_s,
            "direction": direction,
        }
    if None not in {source.lat_deg, source.lon_deg, target.lat_deg, target.lon_deg}:
        distance_km = _great_circle_km(
            float(source.lat_deg),
            float(source.lon_deg),
            float(target.lat_deg),
            float(target.lon_deg),
        )
        rtt_s = (
            2.0 * distance_km * config.internet_path_stretch / config.propagation_speed_km_s
            + config.internet_router_overhead_s
        )
        return rtt_s, {
            "profile_id": "geometry_rtt_surrogate",
            "mode": "geometry_rtt_surrogate",
            "calibration_source": "ground_config.geometry_rtt_surrogate",
            "is_surrogate": True,
            "selection": "geometry",
            "sample_index": None,
            "sample_count": 0,
            "rtt_s": rtt_s,
            "direction": "geometry_fallback",
        }
    return 0.200, {
        "profile_id": "default_200ms_wan_surrogate",
        "mode": "default_fixed_rtt",
        "calibration_source": "ground_runtime.default_wan_rtt",
        "is_surrogate": True,
        "selection": "default",
        "sample_index": None,
        "sample_count": 0,
        "rtt_s": 0.200,
        "direction": "default_fallback",
    }


def _client_to_ground_ingress_one_way_s(
    config: _GroundConfig,
    origin: _RegionConfig,
    ingress: _RegionConfig,
    *,
    request_key: str,
    dns_resolution_delay_s: float,
) -> float:
    """Client-to-first-LB delay for pre-resolved or GeoDNS ingress.

    The local half RTT represents the client-to-nearest regional edge.  A
    non-local chosen LB additionally pays the directional origin-to-ingress
    WAN half RTT; GeoDNS delay is explicit rather than silently zero.
    """

    wan_half_s = 0.0
    if origin.region_id != ingress.region_id:
        wan_rtt_s, _profile = _wan_latency(
            config, origin, ingress, request_key=request_key
        )
        wan_half_s = wan_rtt_s / 2.0
    return origin.local_rtt_s / 2.0 + wan_half_s + dns_resolution_delay_s


def _ground_network_delays(
    config: _GroundConfig,
    context: _GroundQueueContext,
    *,
    serving_region_id: str,
) -> dict[str, Any]:
    """Return the three causal network legs for one regional-ground request.

    A request can first reach a GeoDNS-selected ingress that differs from its
    origin, then be forwarded once by the SkyWalker regional LB.  DNS delay is
    input-only: it must never be re-used as a response delay.  Keeping these
    legs separate makes the event timing, reported network metric, and
    selected-path ledger agree.
    """

    # This helper intentionally accepts only the immutable runtime config and
    # queue context, so look the region table up locally instead of retaining
    # mutable state in the context.
    region_by_id = {region.region_id: region for region in config.regions}
    origin = region_by_id[context.origin_region_id]
    ingress_id = context.request["ingress_region_id"]
    ingress = region_by_id[ingress_id]
    serving = region_by_id[serving_region_id]

    if ingress_id == serving_region_id:
        ingress_to_serving_s = 0.0
        ingress_to_serving_profile = None
    elif (
        context.forwarded
        and context.target_region_id == serving_region_id
        and context.wan_latency_profile is not None
    ):
        ingress_to_serving_s = context.wan_rtt_s / 2.0
        ingress_to_serving_profile = context.wan_latency_profile
    else:
        ingress_rtt_s, ingress_to_serving_profile = _wan_latency(
            config, ingress, serving, request_key=context.request["request_id"]
        )
        ingress_to_serving_s = ingress_rtt_s / 2.0

    if serving_region_id == context.origin_region_id:
        response_to_client_s = origin.local_rtt_s / 2.0
        response_profile = None
    else:
        response_rtt_s, response_profile = _wan_latency(
            config, serving, origin, request_key=context.request["request_id"]
        )
        response_to_client_s = origin.local_rtt_s / 2.0 + response_rtt_s / 2.0

    return {
        "client_to_ingress_s": context.client_to_lb_s,
        "ingress_to_serving_s": ingress_to_serving_s,
        "response_to_client_s": response_to_client_s,
        "total_network_s": (
            context.client_to_lb_s + ingress_to_serving_s + response_to_client_s
        ),
        "ingress_region_id": ingress_id,
        "ingress_to_serving_profile": ingress_to_serving_profile,
        "response_to_client_profile": response_profile,
    }


def _wan_rtt(config: _GroundConfig, source: _RegionConfig, target: _RegionConfig) -> float:
    """Compatibility helper used by summaries and the sequential legacy path."""

    return _wan_latency(config, source, target)[0]


def _great_circle_km(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (lat_a, lon_a, lat_b, lon_b))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    hav = math.sin(delta_lat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    return 6371.0 * 2.0 * math.asin(min(1.0, math.sqrt(hav)))


def _probe_observation_time(time_s: float, interval_s: float) -> float:
    return math.floor((time_s + 1e-12) / interval_s) * interval_s


def _next_probe_time(time_s: float, interval_s: float) -> float:
    tick = math.floor((time_s + 1e-12) / interval_s) + 1
    return tick * interval_s


def _stable_hash_index(key: str, size: int) -> int:
    digest = hashlib.sha256(str(key).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % max(1, size)


def _stable_key_hash(key: str) -> str:
    return hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:16]


def _consistent_hash_target(
    key: str,
    *,
    all_target_ids: list[str],
    available_target_ids: set[str],
    config: _GroundConfig,
    excluded_target_ids: set[str] | None = None,
) -> str:
    """Choose from a fixed vnode ring and skip currently unavailable targets.

    Building the ring from the full configured target set avoids the remapping
    artifact caused by taking ``hash(key) % len(available_targets)``.
    """

    excluded = excluded_target_ids or set()
    targets = sorted(target for target in all_target_ids if target not in excluded)
    available = set(targets) & set(available_target_ids)
    if not targets or not available:
        raise ValueError("consistent-hash selection requires an available target")
    ring: list[tuple[int, str]] = []
    for target in targets:
        for vnode in range(config.hash_virtual_nodes_per_target):
            digest = hashlib.sha256(
                f"{config.hash_seed}|{target}|{vnode}".encode("utf-8")
            ).digest()
            ring.append((int.from_bytes(digest[:8], "big"), target))
    ring.sort(key=lambda item: (item[0], item[1]))
    key_digest = hashlib.sha256(
        f"{config.hash_seed}|key|{key}".encode("utf-8")
    ).digest()
    point = int.from_bytes(key_digest[:8], "big")
    start = 0
    low, high = 0, len(ring)
    while low < high:
        middle = (low + high) // 2
        if ring[middle][0] < point:
            low = middle + 1
        else:
            high = middle
    start = low % len(ring)
    for offset in range(len(ring)):
        target = ring[(start + offset) % len(ring)][1]
        if target in available:
            return target
    raise RuntimeError("consistent-hash ring contains no available target")


def _ground_path(
    *,
    origin: str,
    serving_region: str,
    replica_id: str,
    local_rtt_s: float,
    wan_rtt_s: float,
    ingress_region: str | None = None,
    client_to_ingress_s: float | None = None,
    ingress_to_serving_s: float | None = None,
    response_to_client_s: float | None = None,
    dns_resolution_delay_s: float = 0.0,
) -> tuple[dict[str, Any], ...]:
    ingress = ingress_region or origin
    client_to_ingress_s = (
        local_rtt_s / 2.0
        if client_to_ingress_s is None
        else client_to_ingress_s
    )
    ingress_to_serving_s = (
        wan_rtt_s / 2.0
        if ingress_to_serving_s is None and serving_region != ingress
        else (ingress_to_serving_s or 0.0)
    )
    response_to_client_s = (
        local_rtt_s / 2.0 + (wan_rtt_s / 2.0 if serving_region != origin else 0.0)
        if response_to_client_s is None
        else response_to_client_s
    )
    legs: list[dict[str, Any]] = [
        {
            "relation": "client_to_regional_lb",
            "source": f"client:{origin}",
            "target": f"lb:{ingress}",
            "one_way_delay_s": max(0.0, client_to_ingress_s - dns_resolution_delay_s),
        }
    ]
    if dns_resolution_delay_s > 0.0:
        legs.insert(
            0,
            {
                "relation": "dns_resolution",
                "source": f"client:{origin}",
                "target": f"lb:{ingress}",
                "one_way_delay_s": dns_resolution_delay_s,
            },
        )
    if serving_region != ingress:
        legs.append(
            {
                "relation": "cross_region_lb_forward",
                "source": f"lb:{ingress}",
                "target": f"lb:{serving_region}",
                "one_way_delay_s": ingress_to_serving_s,
            }
        )
    legs.append(
        {
            "relation": "regional_lb_to_replica",
            "source": f"lb:{serving_region}",
            "target": replica_id,
            "one_way_delay_s": 0.0,
        }
    )
    legs.append(
        {
            "relation": "replica_to_client_stream",
            "source": replica_id,
            "target": f"client:{origin}",
            "one_way_delay_s": response_to_client_s,
        }
    )
    return tuple(legs)


def _decision_trace(
    *,
    request: dict[str, Any],
    record: RuntimeRequestRecord,
    decision_time_s: float,
    policy: str,
    candidate_count: int,
    local_candidate_count: int,
    remote_candidate_count: int,
    selected_action: dict[str, Any],
    selection_reason: str,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "request_id": request["request_id"],
        "arrival_time_s": request["arrival_time_s"],
        "decision_time_s": decision_time_s,
        "policy_name": policy,
        "candidate_action_count": candidate_count,
        "local_candidate_count": local_candidate_count,
        "remote_candidate_count": remote_candidate_count,
        "feasible_action_count": 1 if record.accepted else 0,
        "selected_action": selected_action,
        "selection_reason": selection_reason,
        "audit": to_jsonable(audit or {}),
        "bottleneck_reason": record.bottleneck_reason,
        "trace_schema_version": "orbitserve.runtime_decision_trace.v1",
        "warnings": (),
    }


def _reject_record(
    request: dict[str, Any],
    *,
    reason: str,
    lb_queue_wait_s: float | None,
    policy: str,
    serving_region_id: str | None = None,
    cross_region: bool = False,
    network_delay_s: float | None = None,
    selected_path: tuple[dict[str, Any], ...] = (),
) -> RuntimeRequestRecord:
    return RuntimeRequestRecord(
        request_id=request["request_id"],
        arrival_time_s=request["arrival_time_s"],
        deployment_id=None,
        action_type="ground_reject_drop",
        accepted=False,
        completed=False,
        served=False,
        rejected=True,
        timeout=False,
        reject_reason=reason,
        prompt_tokens=request["prompt_tokens"],
        output_tokens=request["output_tokens"],
        ttft_deadline_s=request["ttft_deadline_s"],
        e2e_deadline_s=request["e2e_deadline_s"],
        ground_origin_region_id=request["origin_region_id"],
        ground_serving_region_id=serving_region_id,
        ground_cross_region=cross_region,
        ground_network_delay_s=network_delay_s,
        ground_load_balancer_queue_wait_s=lb_queue_wait_s,
        selected_path=selected_path,
        bottleneck_reason=reason,
        scope=TERRESTRIAL_GROUND_SCOPE,
        trace_model=TERRESTRIAL_NETWORK_TRACE,
        valid_for_paper_main_results=False,
        provenance={
            "baseline_policy": policy,
            "leo_access_links_used": False,
            "leo_isl_links_used": False,
        },
    )


def _timeout_status(
    ttft_s: float,
    e2e_s: float,
    ttft_deadline_s: float | None,
    e2e_deadline_s: float | None,
) -> tuple[bool, str | None]:
    ttft_miss = ttft_deadline_s is not None and ttft_s > ttft_deadline_s + 1e-12
    e2e_miss = e2e_deadline_s is not None and e2e_s > e2e_deadline_s + 1e-12
    if ttft_miss and e2e_miss:
        return True, "ttft_and_e2e_deadline_miss"
    if ttft_miss:
        return True, "ttft_deadline_miss"
    if e2e_miss:
        return True, "e2e_deadline_miss"
    return False, None


def _bottleneck_reason(
    *,
    lb_queue_wait_s: float,
    replica_pending_wait_s: float,
    network_delay_s: float,
    prefill_service_s: float,
    decode_service_s: float,
) -> str:
    terms = {
        "ground_lb_queue": lb_queue_wait_s,
        "ground_replica_pending": replica_pending_wait_s,
        "ground_internet": network_delay_s,
        "ground_prefill": prefill_service_s,
        "ground_decode": decode_service_s,
    }
    return max(terms, key=lambda key: (terms[key], key))


def _ground_runtime_metrics(records: tuple[RuntimeRequestRecord, ...]) -> dict[str, Any]:
    ground = [record for record in records if record.action_type.startswith("ground_")]
    completed = [record for record in ground if record.completed]
    cross_region = [record for record in completed if record.ground_cross_region]
    hit_tokens = sum(record.kv_hit_tokens or 0 for record in ground)
    miss_tokens = sum(record.kv_miss_tokens or 0 for record in ground)
    return {
        "ground_request_count": len(ground),
        "ground_local_request_count": len([record for record in completed if not record.ground_cross_region]),
        "ground_cross_region_request_count": len(cross_region),
        "ground_cross_region_request_fraction": len(cross_region) / len(completed) if completed else None,
        "ground_network_delay_p50_s": _percentile([record.ground_network_delay_s for record in completed], 0.50),
        "ground_network_delay_p95_s": _percentile([record.ground_network_delay_s for record in completed], 0.95),
        "ground_network_delay_p99_s": _percentile([record.ground_network_delay_s for record in completed], 0.99),
        "ground_load_balancer_queue_wait_p95_s": _percentile(
            [record.ground_load_balancer_queue_wait_s for record in completed], 0.95
        ),
        "ground_replica_pending_wait_p95_s": _percentile(
            [record.ground_replica_pending_wait_s for record in completed], 0.95
        ),
        "ground_prefix_cache_hit_tokens": hit_tokens,
        "ground_prefix_cache_miss_tokens": miss_tokens,
        "ground_prefix_cache_hit_rate_tokens": hit_tokens / (hit_tokens + miss_tokens)
        if hit_tokens + miss_tokens > 0
        else None,
        "ground_output_reservation_exceeded_count": len(
            [
                record
                for record in ground
                if bool(
                    record.provenance.get(
                        "realized_output_exceeded_admission_reservation", False
                    )
                )
            ]
        ),
        "valid_for_paper_main_results": False,
    }


def _ground_completed_throughput_metrics(
    records: tuple[RuntimeRequestRecord, ...],
    *,
    effective_horizon_s: float,
) -> dict[str, Any]:
    completed = [record for record in records if record.completed]
    horizon = max(1e-9, effective_horizon_s)
    return {
        "completed_request_throughput_per_s": len(completed) / horizon,
        "completed_token_throughput_per_s": sum(
            record.prompt_tokens + record.output_tokens for record in completed
        )
        / horizon,
        "completed_output_token_throughput_per_s": sum(
            record.output_tokens for record in completed
        )
        / horizon,
        "completed_throughput_token_accounting": "prompt_plus_output_and_output_only_both_reported",
    }


def _region_metrics(
    records: tuple[RuntimeRequestRecord, ...],
    regions: tuple[_RegionConfig, ...],
) -> tuple[dict[str, Any], ...]:
    result: list[dict[str, Any]] = []
    for region in regions:
        ingress = [record for record in records if record.ground_origin_region_id == region.region_id]
        served_here = [record for record in records if record.ground_serving_region_id == region.region_id]
        completed_ingress = [record for record in ingress if record.completed]
        result.append(
            {
                "region_id": region.region_id,
                "replica_count": region.replica_count,
                "ingress_request_count": len(ingress),
                "served_request_count": len([record for record in served_here if record.served]),
                "offloaded_request_count": len(
                    [record for record in completed_ingress if record.ground_cross_region]
                ),
                "offloaded_request_fraction": (
                    len([record for record in completed_ingress if record.ground_cross_region])
                    / len(completed_ingress)
                    if completed_ingress
                    else None
                ),
                "lb_queue_wait_p95_s": _percentile(
                    [record.ground_load_balancer_queue_wait_s for record in completed_ingress], 0.95
                ),
                "ttft_p95_s": _percentile([record.ttft_s for record in completed_ingress], 0.95),
                "e2e_p95_s": _percentile([record.e2e_s for record in completed_ingress], 0.95),
                "valid_for_paper_main_results": False,
            }
        )
    return tuple(result)


def _replica_metrics(
    replicas_by_region: dict[str, list[_ReplicaState]],
    *,
    effective_horizon_s: float,
    config: _GroundConfig,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for region_id, replicas in replicas_by_region.items():
        rows: list[dict[str, Any]] = []
        for replica in sorted(replicas, key=lambda item: item.replica_id):
            replica_end_s = max(
                (job.end_s for job in replica.jobs),
                default=effective_horizon_s,
            )
            _compact_cache_as_of(
                replica,
                at_s=replica_end_s,
                max_entries=config.prefix_cache_max_entries,
                max_tokens=config.kv_cache_capacity_tokens,
            )
            cache_as_of_end = _ready_cache_entries(
                replica.cache,
                at_s=replica_end_s,
                max_entries=config.prefix_cache_max_entries,
                max_tokens=config.kv_cache_capacity_tokens,
            )
            event_times = sorted(
                {
                    value
                    for job in replica.jobs
                    for value in (job.enqueue_s, job.start_s, job.end_s)
                }
            )
            peak_active = max(
                (len(replica.active_jobs(time_s)) for time_s in event_times),
                default=0,
            )
            peak_pending = max(
                (len(replica.pending_jobs(time_s)) for time_s in event_times),
                default=0,
            )
            peak_outstanding = max(
                (replica.outstanding_count(time_s) for time_s in event_times),
                default=0,
            )
            peak_active_tokens = max(
                (replica.active_tokens(time_s) for time_s in event_times),
                default=0,
            )
            intervals = sorted(
                (job.start_s, job.end_s)
                for job in replica.jobs
                if job.end_s > job.start_s
            )
            busy_s = 0.0
            if intervals:
                current_start, current_end = intervals[0]
                for start_s, end_s in intervals[1:]:
                    if start_s <= current_end + 1e-12:
                        current_end = max(current_end, end_s)
                    else:
                        busy_s += current_end - current_start
                        current_start, current_end = start_s, end_s
                busy_s += current_end - current_start
            rows.append(
                {
                    "replica_id": replica.replica_id,
                    "job_count": len(replica.jobs),
                    "admission_token_count": sum(
                        job.batch_tokens for job in replica.jobs
                    ),
                    "peak_active_request_count": peak_active,
                    "peak_pending_request_count": peak_pending,
                    "peak_outstanding_request_count": peak_outstanding,
                    "peak_active_admission_tokens": peak_active_tokens,
                    "busy_time_s": busy_s,
                    "busy_utilization": min(
                        1.0, busy_s / max(1e-9, effective_horizon_s)
                    ),
                    "routing_entry_count": len(replica.routing_index),
                    "kv_cache_entry_count": len(cache_as_of_end),
                    "kv_cache_version_count": len(replica.cache),
                    "kv_cache_token_count": sum(
                        entry.tokens for entry in cache_as_of_end
                    ),
                }
            )
        result[region_id] = rows
    return result


def _replica_imbalance_metrics(
    replica_metrics: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    rows = [row for region_rows in replica_metrics.values() for row in region_rows]

    def max_over_mean(field: str) -> float | None:
        values = [float(row[field]) for row in rows]
        if not values:
            return None
        mean = sum(values) / len(values)
        return max(values) / mean if mean > 0 else 0.0

    return {
        "ground_replica_imbalance_ratio_definition": "max_over_mean_across_all_ground_replicas",
        "ground_replica_job_count_imbalance_ratio": max_over_mean("job_count"),
        "ground_replica_peak_outstanding_imbalance_ratio": max_over_mean(
            "peak_outstanding_request_count"
        ),
        "ground_replica_kv_entry_imbalance_ratio": max_over_mean(
            "kv_cache_entry_count"
        ),
        "ground_replica_peak_active_request_count": max(
            (row["peak_active_request_count"] for row in rows), default=0
        ),
        "ground_replica_peak_pending_request_count": max(
            (row["peak_pending_request_count"] for row in rows), default=0
        ),
        "ground_replica_peak_outstanding_request_count": max(
            (row["peak_outstanding_request_count"] for row in rows), default=0
        ),
    }


def _configured_execution_profiles(config: _GroundConfig) -> dict[str, Any]:
    return {
        "default_service_profile": _service_profile_provenance(
            config.default_service_profile
        ),
        "service_profiles_by_region": {
            region_id: _service_profile_provenance(profile)
            for region_id, profile in sorted(config.service_profiles_by_region.items())
        },
        "service_profiles_by_replica": {
            replica_id: _service_profile_provenance(profile)
            for replica_id, profile in sorted(config.service_profiles_by_replica.items())
        },
        "wan_latency_profiles": {
            source: {
                target: {
                    "profile_id": profile.profile_id,
                    "mode": profile.mode,
                    "fixed_rtt_s": profile.fixed_rtt_s,
                    "samples_s": profile.samples_s,
                    "calibration_source": profile.calibration_source,
                    "is_surrogate": profile.is_surrogate,
                }
                for target, profile in sorted(targets.items())
            }
            for source, targets in sorted(config.wan_latency_profiles.items())
        },
        "execution_model_is_surrogate": True,
        "execution_model_boundary": (
            "calibrated analytical token-service and deterministic WAN profiles; "
            "not a replay of an inference engine or cloud network"
        ),
    }


def _applied_profile_summaries(
    records: tuple[RuntimeRequestRecord, ...]
) -> dict[str, tuple[dict[str, Any], ...]]:
    service: dict[str, dict[str, Any]] = {}
    wan: dict[str, dict[str, Any]] = {}
    for record in records:
        provenance = record.provenance or {}
        service_profile = provenance.get("service_profile")
        if isinstance(service_profile, dict) and service_profile.get("profile_id"):
            service[str(service_profile["profile_id"])] = service_profile
        wan_profile = provenance.get("wan_latency_profile")
        if isinstance(wan_profile, dict) and wan_profile.get("profile_id"):
            wan_key = (
                f"{wan_profile['profile_id']}|{wan_profile.get('mode')}|"
                f"{wan_profile.get('calibration_source')}"
            )
            wan[wan_key] = wan_profile
    return {
        "service_profiles": tuple(service[key] for key in sorted(service)),
        "wan_latency_profiles": tuple(wan[key] for key in sorted(wan)),
    }


def _effective_ground_config(config: _GroundConfig) -> dict[str, Any]:
    region_by_id = {region.region_id: region for region in config.regions}
    return to_jsonable(
        {
            "regions": config.regions,
            "wan_rtt_s": {
                source.region_id: {
                    target.region_id: (
                        0.0
                        if source.region_id == target.region_id
                        else _wan_rtt(config, source, target)
                    )
                    for target in config.regions
                }
                for source in config.regions
            },
            "probe_interval_s": config.probe_interval_s,
            "remote_queue_buffer_requests": config.remote_queue_buffer_requests,
            "routing_policy": config.routing_policy,
            "selective_pushing": config.selective_pushing,
            "replica_token_service": {
                "prefill_tokens_per_s": config.prefill_tokens_per_s,
                "decode_tokens_per_s": config.decode_tokens_per_s,
                "continuous_batch_capacity_tokens": config.continuous_batch_capacity_tokens,
                "kv_cache_capacity_tokens": config.kv_cache_capacity_tokens,
                "batch_contention_alpha": config.batch_contention_alpha,
                "service_semantics": "admission_time_contention_surrogate",
            },
            "execution_profiles": _configured_execution_profiles(config),
            "max_batch_size": config.max_batch_size,
            "max_replica_queue_requests": config.max_replica_queue_requests,
            "max_lb_queue_wait_s": config.max_lb_queue_wait_s,
            "prefix_cache_max_entries": config.prefix_cache_max_entries,
            "max_outstanding_requests": config.max_outstanding_requests,
            "admission_output_reservation_tokens": config.admission_output_reservation_tokens,
            "hash_virtual_nodes_per_target": config.hash_virtual_nodes_per_target,
            "hash_seed": config.hash_seed,
            "prefix_affinity_min_similarity": config.prefix_affinity_min_similarity,
            "prefix_low_affinity_fallback": config.prefix_low_affinity_fallback,
            "arrival_time_scale": config.arrival_time_scale,
            "geometry_rtt_surrogate": {
                "propagation_speed_km_s": config.propagation_speed_km_s,
                "internet_path_stretch": config.internet_path_stretch,
                "internet_router_overhead_s": config.internet_router_overhead_s,
            },
            "region_ids": tuple(region_by_id),
        }
    )


def _percentile(values: Any, quantile: float) -> float | None:
    data = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not data:
        return None
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return data[lower]
    fraction = position - lower
    return data[lower] * (1.0 - fraction) + data[upper] * fraction


def _effective_horizon_s(records: tuple[RuntimeRequestRecord, ...]) -> float:
    if not records:
        return 1.0
    start_s = min(record.arrival_time_s for record in records)
    finishes = [
        record.arrival_time_s + (record.e2e_s or 0.0)
        for record in records
        if record.completed
    ]
    end_s = max(
        max(record.arrival_time_s for record in records),
        max(finishes, default=start_s),
    )
    return max(1e-9, end_s - start_s)


def _positive_float(value: Any, name: str) -> float:
    parsed = _float(value, 0.0)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    parsed = _float(value, 0.0)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _positive_int(value: Any, default: int) -> int:
    parsed = _int(value, default)
    if parsed <= 0:
        raise ValueError("ground integer capacity fields must be positive")
    return parsed


def _nonnegative_int(value: Any, default: int) -> int:
    parsed = _int(value, default)
    if parsed < 0:
        raise ValueError("ground integer queue fields must be non-negative")
    return parsed


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


__all__ = [
    "GroundRuntimeSimulationResult",
    "TERRESTRIAL_GROUND_SCOPE",
    "TERRESTRIAL_NETWORK_TRACE",
    "run_ground_runtime",
]
