"""Centralized and gateway-style terrestrial comparison baselines.

This module deliberately does *not* reuse the regional SkyWalker dispatch
loop.  SkyWalker uses a nearest regional load balancer, local-first routing,
and pending-request selective pushing.  The four ``SkyServe`` comparison
baselines in the SkyWalker paper instead use one global coordinator in the
US.  Treating those as a routing-policy toggle inside the regional loop would
give them SkyWalker's topology for free and invalidate the comparison.

The implementation remains a discrete-event service surrogate.  In
particular, ``central_sgl_like`` describes a global prefix-aware, blind-push
router and is not an implementation of the SGLang Router binary.
"""

from __future__ import annotations

import math
import heapq
from dataclasses import replace
from typing import Any

from ...common import to_jsonable
from .runtime import (
    TERRESTRIAL_GROUND_SCOPE,
    TERRESTRIAL_NETWORK_TRACE,
    GroundRuntimeSimulationResult,
    _ReplicaJob,
    _ReplicaState,
    _compact_cache_as_of,
    _consistent_hash_target,
    _effective_ground_config,
    _effective_horizon_s,
    _ground_completed_throughput_metrics,
    _ground_runtime_metrics,
    _normalize_request,
    _percentile,
    _prefix_hit,
    _prefix_routing_sort_key,
    _region_metrics,
    _replica_imbalance_metrics,
    _replica_metrics,
    _resolve_config,
    _resolve_client_harness,
    _service_profile_for_replica,
    _stable_key_hash,
    _timeout_status,
    _update_prefix_cache,
    _update_routing_index,
    _wan_latency,
    _resolve_ground_ingress,
)
from .contracts import GroundRequestRecord as RuntimeRequestRecord


CENTRALIZED_GROUND_POLICIES = frozenset(
    {
        "ground_central_rr",
        "ground_central_ll",
        "ground_central_ch",
        "ground_central_sgl_like",
        "ground_gke_gateway_like",
    }
)


def is_centralized_ground_policy(policy: str | None) -> bool:
    return str(policy or "") in CENTRALIZED_GROUND_POLICIES


def run_centralized_ground_runtime(
    *,
    requests: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    runtime_strategy: Any,
    regions: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    gateways: tuple[dict[str, Any], ...] | list[dict[str, Any]] = (),
    policy: str | None = None,
) -> GroundRuntimeSimulationResult:
    """Run one paper-comparison terrestrial topology.

    ``ground_central_*`` models a single US global coordinator.  It directly
    blind-pushes to a replica in the global set; there is no local-first gate,
    remote-LB probe, or ``tau`` queue filter.  ``ground_gke_gateway_like``
    keeps a regional ingress gateway but deliberately has no prefix affinity
    or pending-request selective pushing.  Its backend rule is explicitly a
    configurable surrogate, not a claim about an undocumented GKE policy.
    """

    policy_name = str(policy or getattr(runtime_strategy, "policy", ""))
    if policy_name not in CENTRALIZED_GROUND_POLICIES:
        raise ValueError(f"unsupported centralized terrestrial policy: {policy_name!r}")
    raw_requests = tuple(dict(item) for item in requests if isinstance(item, dict))
    config = _resolve_config(runtime_strategy, regions=regions, requests=raw_requests)
    raw_config = dict(config.raw)
    topology = dict(raw_config.get("baseline_topology") or {})
    central_region_id = str(
        topology.get("central_region_id")
        or raw_config.get("central_region_id")
        or "us"
    )
    region_by_id = {region.region_id: region for region in config.regions}
    if policy_name != "ground_gke_gateway_like" and central_region_id not in region_by_id:
        raise ValueError(
            "centralized ground baseline requires ground_config.baseline_topology."
            "central_region_id (or ground_config.central_region_id) to name a configured region"
        )
    gateway_backend_policy = str(
        topology.get("gke_backend_policy")
        or raw_config.get("gke_backend_policy")
        or "round_robin"
    ).lower()
    if policy_name == "ground_gke_gateway_like" and gateway_backend_policy not in {
        "round_robin",
        "least_outstanding",
    }:
        raise ValueError(
            "ground_gke_gateway_like supports gke_backend_policy "
            "round_robin or least_outstanding"
        )

    normalized = tuple(
        _normalize_request(item, index, gateway_to_region={})
        for index, item in enumerate(raw_requests)
    )
    client_harness = _resolve_client_harness(
        config.raw,
        normalized,
        arrival_time_scale=config.arrival_time_scale,
    )
    if (
        policy_name == "ground_central_ch"
        and bool(config.raw.get("require_hash_identity", False))
        and any(item["hash_identity_source"] == "prefix_key_fallback" for item in normalized)
    ):
        raise ValueError(
            "require_hash_identity=true but central CH requires explicit user_id/hashed_user_id or session_id"
        )
    if normalized and abs(config.arrival_time_scale - 1.0) > 1e-12:
        first_arrival_s = min(item["arrival_time_s"] for item in normalized)
        normalized = tuple(
            {
                **item,
                "arrival_time_s": first_arrival_s
                + (item["arrival_time_s"] - first_arrival_s) * config.arrival_time_scale,
                "original_arrival_time_s": item["arrival_time_s"],
            }
            for item in normalized
        )
    unknown_regions = sorted(
        {item["origin_region_id"] for item in normalized if item["origin_region_id"] not in region_by_id}
    )
    if unknown_regions:
        raise ValueError(
            "centralized ground request references region(s) absent from ground_config.regions: "
            + ", ".join(unknown_regions)
        )
    if policy_name == "ground_gke_gateway_like":
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
    all_replicas = [
        replica for region_id in sorted(replicas_by_region)
        for replica in replicas_by_region[region_id]
    ]
    if not all_replicas:
        raise ValueError("centralized ground runtime has no configured replicas")

    # A blind-push router does not wait for pending==0 or use a queue-depth
    # gate.  An artificial admission reservation above batch capacity is still
    # rejected because no inference engine could admit it.
    global_rr_index = 0
    records: list[RuntimeRequestRecord] = []
    decision_traces: list[dict[str, Any]] = []
    # A single heap carries client-to-ingress arrivals and ingress-to-replica
    # arrivals.  Replica admission must be ordered by the latter, not by the
    # time at which the global coordinator made its decision.
    pending_arrivals: list[tuple[float, str, int, str, dict[str, Any]]] = []
    arrival_sequence = 0
    client_program_by_request: dict[str, Any] = {}
    client_program_position: dict[tuple[str, int], int] = {}
    client_program_remaining: dict[tuple[str, int], int] = {}
    client_program_terminal_s: dict[tuple[str, int], list[float]] = {}
    client_program_request_terminal_s: dict[str, float] = {}
    client_program_request_scheduled: set[str] = set()
    client_program_start_s: dict[tuple[str, int], float] = {}
    client_active_programs: dict[str, int] = {}
    client_peak_inflight_programs: dict[str, int] = {}
    client_released_program_count = 0
    coordinator_inflight_dispatches: dict[str, int] = {
        replica.replica_id: 0 for replica in all_replicas
    }
    normalized_by_request_id = {item["request_id"]: item for item in normalized}

    def schedule_request(raw_request: dict[str, Any], arrival_time_s: float) -> None:
        nonlocal arrival_sequence
        arrival_sequence += 1
        request = {**raw_request, "arrival_time_s": float(arrival_time_s)}
        origin = region_by_id[request["origin_region_id"]]
        ingress_region_id = (
            request["ingress_region_id"]
            if policy_name == "ground_gke_gateway_like"
            else central_region_id
        )
        ingress = region_by_id[ingress_region_id]
        client_to_ingress_s, client_ingress_profile = _client_to_ingress_one_way_s(
            config,
            origin,
            ingress,
            request_key=request["request_id"],
            dns_resolution_delay_s=(
                float(request.get("dns_resolution_delay_s", 0.0))
                if policy_name == "ground_gke_gateway_like"
                else 0.0
            ),
        )
        heapq.heappush(
            pending_arrivals,
            (
                float(arrival_time_s) + client_to_ingress_s,
                request["request_id"],
                arrival_sequence,
                "ingress",
                {
                    "request": request,
                    "origin": origin,
                    "ingress": ingress,
                    "client_to_ingress_s": client_to_ingress_s,
                    "client_ingress_profile": client_ingress_profile,
                    "dns_resolution_delay_s": (
                        float(request.get("dns_resolution_delay_s", 0.0))
                        if policy_name == "ground_gke_gateway_like"
                        else 0.0
                    ),
                },
            ),
        )

    def release_ready_program_requests(program: Any) -> None:
        key = (program.client_id, program.sequence_index)
        program_start_s = client_program_start_s[key]
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
            schedule_request(
                normalized_by_request_id[request_id],
                release_base_s + program.request_release_offsets_s[request_id],
            )
            client_program_request_scheduled.add(request_id)

    def release_program(client_id: str, position: int, release_time_s: float) -> None:
        nonlocal client_released_program_count
        if client_harness is None:
            raise RuntimeError("client program release called without client harness")
        program = client_harness.programs_by_client[client_id][position]
        key = (client_id, program.sequence_index)
        if key in client_program_remaining:
            raise RuntimeError("client program released more than once")
        client_program_position[key] = position
        client_program_remaining[key] = len(program.request_ids)
        client_program_terminal_s[key] = []
        client_program_start_s[key] = release_time_s
        client_active_programs[client_id] = client_active_programs.get(client_id, 0) + 1
        client_peak_inflight_programs[client_id] = max(
            client_peak_inflight_programs.get(client_id, 0), client_active_programs[client_id]
        )
        if client_active_programs[client_id] != 1:
            raise RuntimeError("client harness released overlapping programs for one client")
        client_released_program_count += 1
        release_ready_program_requests(program)

    def on_terminal(request: dict[str, Any], record: RuntimeRequestRecord, terminal_time_s: float) -> RuntimeRequestRecord:
        if client_harness is None:
            return record
        program = client_program_by_request[request["request_id"]]
        key = (program.client_id, program.sequence_index)
        client_program_remaining[key] -= 1
        client_program_terminal_s[key].append(float(terminal_time_s))
        client_program_request_terminal_s[request["request_id"]] = float(terminal_time_s)
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
                    "program_request_release_offset_s": program.request_release_offsets_s[request["request_id"]],
                    "terminal_time_s": terminal_time_s,
                },
            },
        )
        release_ready_program_requests(program)
        if client_program_remaining[key] == 0:
            client_active_programs[program.client_id] -= 1
            next_position = client_program_position[key] + 1
            programs = client_harness.programs_by_client[program.client_id]
            if next_position < len(programs):
                release_program(
                    program.client_id,
                    next_position,
                    max(client_program_terminal_s[key])
                    + client_harness.inter_program_think_time_s,
                )
        return record

    if client_harness is None:
        for request in normalized:
            schedule_request(request, request["arrival_time_s"])
    else:
        for programs in client_harness.programs_by_client.values():
            for program in programs:
                for request_id in program.request_ids:
                    client_program_by_request[request_id] = program
        for client_id, programs in client_harness.programs_by_client.items():
            if programs:
                release_program(client_id, 0, programs[0].initial_release_s)

    while pending_arrivals:
        event_time_s, _request_id, _sequence, event_kind, payload = heapq.heappop(
            pending_arrivals
        )
        if event_kind == "ingress":
            request = payload["request"]
            origin = payload["origin"]
            ingress = payload["ingress"]
            client_to_ingress_s = payload["client_to_ingress_s"]
            client_ingress_profile = payload["client_ingress_profile"]
            dns_resolution_delay_s = payload["dns_resolution_delay_s"]
            ingress_decision_s = event_time_s
            selected_replica, global_rr_index = _select_global_replica(
                policy=policy_name,
                backend_policy=gateway_backend_policy,
                replicas=all_replicas,
                request=request,
                dispatch_s=ingress_decision_s,
                config=config,
                round_robin_index=global_rr_index,
                coordinator_inflight_dispatches=coordinator_inflight_dispatches,
            )
            service_profile = _service_profile_for_replica(config, selected_replica)
            reservation_tokens = (
                request["max_output_tokens"]
                if request.get("max_output_tokens") is not None
                else config.admission_output_reservation_tokens
            )
            admission_tokens = max(
                1, request["prompt_tokens"] + max(0, int(reservation_tokens))
            )
            if admission_tokens > service_profile.continuous_batch_capacity_tokens:
                reject_response_s, reject_response_profile = _replica_to_client_one_way_s(
                    config, ingress, origin, request_key=request["request_id"]
                )
                record = RuntimeRequestRecord(
                    request_id=request["request_id"],
                    arrival_time_s=request["arrival_time_s"],
                    deployment_id=None,
                    action_type="ground_reject_drop",
                    accepted=False,
                    completed=False,
                    served=False,
                    rejected=True,
                    timeout=False,
                    reject_reason="ground_request_exceeds_continuous_batch_capacity",
                    prompt_tokens=request["prompt_tokens"],
                    output_tokens=request["output_tokens"],
                    ttft_deadline_s=request["ttft_deadline_s"],
                    e2e_deadline_s=request["e2e_deadline_s"],
                    ground_origin_region_id=origin.region_id,
                    ground_serving_region_id=ingress.region_id,
                    ground_cross_region=False,
                    ground_network_delay_s=client_to_ingress_s + reject_response_s,
                    ground_load_balancer_queue_wait_s=0.0,
                    selected_path=_baseline_path(
                        origin=origin.region_id,
                        ingress=ingress.region_id,
                        serving=ingress.region_id,
                        replica_id=f"ground:{ingress.region_id}:unassigned",
                        client_to_ingress_s=client_to_ingress_s,
                        ingress_to_replica_s=0.0,
                        response_s=reject_response_s,
                        dns_resolution_delay_s=dns_resolution_delay_s,
                    ),
                    bottleneck_reason="ground_request_exceeds_continuous_batch_capacity",
                    scope=TERRESTRIAL_GROUND_SCOPE,
                    trace_model=TERRESTRIAL_NETWORK_TRACE,
                    valid_for_paper_main_results=False,
                    provenance={
                        "baseline_policy": policy_name,
                        "topology": _topology_name(policy_name),
                        "central_region_id": central_region_id
                        if policy_name != "ground_gke_gateway_like"
                        else None,
                        "ingress_region_id": ingress.region_id,
                        "ingress_resolution_source": request.get(
                            "ingress_resolution_source"
                        ),
                        "dns_resolution_delay_s": dns_resolution_delay_s,
                        "reject_response_latency_profile": reject_response_profile,
                        "rejected_before_replica_forward": True,
                        "realized_output_tokens_used_for_admission": False,
                    },
                )
                record = on_terminal(
                    request,
                    record,
                    ingress_decision_s + reject_response_s,
                )
                records.append(record)
                decision_traces.append(
                    _decision_trace(
                        request,
                        record,
                        ingress_decision_s,
                        policy_name,
                        selected_replica,
                        {"event_kind": "ingress_reject"},
                    )
                )
                continue
            ingress_to_replica_s, ingress_replica_profile = _region_one_way_s(
                config,
                ingress,
                region_by_id[selected_replica.region_id],
                request_key=request["request_id"],
            )
            coordinator_inflight_dispatches[selected_replica.replica_id] += 1
            response_s, response_profile = _replica_to_client_one_way_s(
                config,
                region_by_id[selected_replica.region_id],
                origin,
                request_key=request["request_id"],
            )
            arrival_sequence += 1
            heapq.heappush(
                pending_arrivals,
                (
                    ingress_decision_s + ingress_to_replica_s,
                    request["request_id"],
                    arrival_sequence,
                    "replica_arrival",
                    {
                        "request": request,
                        "origin": origin,
                        "ingress": ingress,
                        "selected_replica": selected_replica,
                        "service_profile": service_profile,
                        "reservation_tokens": reservation_tokens,
                        "admission_tokens": admission_tokens,
                        "client_to_ingress_s": client_to_ingress_s,
                        "client_ingress_profile": client_ingress_profile,
                        "dns_resolution_delay_s": dns_resolution_delay_s,
                        "ingress_to_replica_s": ingress_to_replica_s,
                        "ingress_replica_profile": ingress_replica_profile,
                        "response_s": response_s,
                        "response_profile": response_profile,
                        "ingress_decision_s": ingress_decision_s,
                    },
                ),
            )
            continue
        if event_kind != "replica_arrival":
            raise RuntimeError(f"unsupported centralized ground event: {event_kind!r}")
        request = payload["request"]
        origin = payload["origin"]
        ingress = payload["ingress"]
        selected_replica = payload["selected_replica"]
        coordinator_inflight_dispatches[selected_replica.replica_id] -= 1
        if coordinator_inflight_dispatches[selected_replica.replica_id] < 0:
            raise RuntimeError("central coordinator transit accounting underflow")
        service_profile = payload["service_profile"]
        reservation_tokens = payload["reservation_tokens"]
        admission_tokens = payload["admission_tokens"]
        client_to_ingress_s = payload["client_to_ingress_s"]
        client_ingress_profile = payload["client_ingress_profile"]
        dns_resolution_delay_s = payload["dns_resolution_delay_s"]
        ingress_to_replica_s = payload["ingress_to_replica_s"]
        ingress_replica_profile = payload["ingress_replica_profile"]
        response_s = payload["response_s"]
        response_profile = payload["response_profile"]
        ingress_decision_s = payload["ingress_decision_s"]
        replica_arrival_s = event_time_s
        service_start_s = _earliest_start_blind(
            selected_replica, replica_arrival_s, admission_tokens, service_profile
        )
        replica_wait_s = max(0.0, service_start_s - replica_arrival_s)
        _compact_cache_as_of(
            selected_replica,
            at_s=replica_arrival_s,
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
        active = selected_replica.active_jobs(service_start_s)
        contention = 1.0 + service_profile.batch_contention_alpha * len(active)
        prefill_s = (
            service_profile.first_token_overhead_s
            + miss_tokens / service_profile.prefill_tokens_per_s * contention
        )
        decode_s = request["output_tokens"] / service_profile.decode_tokens_per_s * contention
        first_token_ready_s = service_start_s + prefill_s
        completion_s = first_token_ready_s + decode_s
        selected_replica.jobs.append(
            _ReplicaJob(
                request_id=request["request_id"],
                enqueue_s=replica_arrival_s,
                start_s=service_start_s,
                end_s=completion_s,
                batch_tokens=admission_tokens,
            )
        )
        _update_routing_index(
            selected_replica,
            request,
            visible_at_s=replica_arrival_s,
            max_entries=config.prefix_cache_max_entries,
        )
        _update_prefix_cache(
            selected_replica,
            request,
            available_at_s=completion_s,
            max_entries=config.prefix_cache_max_entries,
            max_tokens=config.kv_cache_capacity_tokens,
        )
        network_s = client_to_ingress_s + ingress_to_replica_s + response_s
        ttft_s = first_token_ready_s + response_s - request["arrival_time_s"]
        e2e_s = completion_s + response_s - request["arrival_time_s"]
        timeout, timeout_reason = _timeout_status(
            ttft_s, e2e_s, request["ttft_deadline_s"], request["e2e_deadline_s"]
        )
        action_type = {
            "ground_central_rr": "ground_central_rr",
            "ground_central_ll": "ground_central_ll",
            "ground_central_ch": "ground_central_ch",
            "ground_central_sgl_like": "ground_central_sgl_like",
            "ground_gke_gateway_like": "ground_gke_gateway_like",
        }[policy_name]
        path = _baseline_path(
            origin=origin.region_id,
            ingress=ingress.region_id,
            serving=selected_replica.region_id,
            replica_id=selected_replica.replica_id,
            client_to_ingress_s=client_to_ingress_s,
            ingress_to_replica_s=ingress_to_replica_s,
            response_s=response_s,
            dns_resolution_delay_s=dns_resolution_delay_s,
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
            prefill_queue_wait_s=replica_wait_s,
            decode_queue_wait_s=0.0,
            link_queue_wait_s=0.0,
            link_transmission_s=network_s,
            prefill_service_s=prefill_s,
            decode_service_s=decode_s,
            ttft_s=ttft_s,
            e2e_s=e2e_s,
            kv_hit_tokens=hit_tokens,
            kv_miss_tokens=miss_tokens,
            context_recompute_tokens=miss_tokens,
            session_action="sticky_kv_owner" if hit_tokens > 0 else "recompute_context",
            selected_path=path,
            ground_origin_region_id=origin.region_id,
            ground_serving_region_id=selected_replica.region_id,
            ground_replica_id=selected_replica.replica_id,
            ground_cross_region=selected_replica.region_id != origin.region_id,
            ground_network_delay_s=network_s,
            ground_load_balancer_queue_wait_s=0.0,
            ground_replica_pending_wait_s=replica_wait_s,
            ground_prefix_cache_hit_ratio=hit_ratio,
            bottleneck_reason=_baseline_bottleneck(network_s, replica_wait_s, prefill_s, decode_s),
            scope=TERRESTRIAL_GROUND_SCOPE,
            trace_model=TERRESTRIAL_NETWORK_TRACE,
            fallback_used=False,
            valid_for_paper_main_results=False,
            provenance={
                "baseline_policy": policy_name,
                "topology": _topology_name(policy_name),
                "central_region_id": central_region_id if policy_name != "ground_gke_gateway_like" else None,
                "ingress_region_id": ingress.region_id,
                "ingress_resolution_source": request.get("ingress_resolution_source"),
                "dns_resolution_delay_s": dns_resolution_delay_s,
                "gke_backend_policy": gateway_backend_policy if policy_name == "ground_gke_gateway_like" else None,
                "dispatcher_semantics": _dispatcher_semantics(policy_name, gateway_backend_policy),
                "selective_pushing": "blind_pushing",
                "pending_gate_used": False,
                "remote_tau_gate_used": False,
                "prefix_key_hash": _stable_key_hash(request["prefix_key"]),
                "hash_identity_source": request["hash_identity_source"],
                "admission_tokens": admission_tokens,
                "admission_output_reservation_tokens": reservation_tokens,
                "realized_output_tokens_used_for_admission": False,
                "realized_output_exceeded_admission_reservation": request["output_tokens"] > int(reservation_tokens),
                "service_profile": {
                    "profile_id": service_profile.profile_id,
                    "prefill_tokens_per_s": service_profile.prefill_tokens_per_s,
                    "decode_tokens_per_s": service_profile.decode_tokens_per_s,
                    "first_token_overhead_s": service_profile.first_token_overhead_s,
                    "batch_slowdown_per_active_request_multiplier": service_profile.batch_contention_alpha,
                    "calibration_source": service_profile.calibration_source,
                    "is_surrogate": service_profile.is_surrogate,
                },
                "wan_latency_profiles": {
                    "client_to_ingress": client_ingress_profile,
                    "ingress_to_replica": ingress_replica_profile,
                    "replica_to_client": response_profile,
                },
                "global_coordinator_surrogate": policy_name != "ground_gke_gateway_like",
                "gke_gateway_exact_reproduction": False,
                "s_glang_router_exact_reproduction": False,
                "leo_access_links_used": False,
                "leo_isl_links_used": False,
                "paper_exact_numerical_reproduction": False,
            },
        )
        record = on_terminal(
            request, record, request["arrival_time_s"] + e2e_s
        )
        records.append(record)
        decision_traces.append(_decision_trace(request, record, ingress_decision_s, policy_name, selected_replica, {
            "replica_arrival_s": replica_arrival_s,
            "client_to_ingress_one_way_s": client_to_ingress_s,
            "dns_resolution_delay_s": dns_resolution_delay_s,
            "ingress_to_replica_one_way_s": ingress_to_replica_s,
            "replica_to_client_one_way_s": response_s,
            "service_profile_id": service_profile.profile_id,
            "wan_latency_profiles": {
                "client_to_ingress": client_ingress_profile,
                "ingress_to_replica": ingress_replica_profile,
                "replica_to_client": response_profile,
            },
            "global_replica_count": len(all_replicas),
        }))

    record_tuple = tuple(sorted(records, key=lambda item: (item.arrival_time_s, item.request_id)))
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
                "program_completion_throughput_per_s": completed_program_count / max(1e-9, effective_horizon_s),
                "inter_program_think_time_s": client_harness.inter_program_think_time_s,
                "max_inflight_programs_per_client": max(client_peak_inflight_programs.values(), default=0),
                "inflight_program_constraint_satisfied": all(
                    value <= 1 for value in client_peak_inflight_programs.values()
                ),
                "next_program_release": "previous_program_all_requests_terminal_plus_configured_think_time",
            }
        )
    replica_metrics = _replica_metrics(
        replicas_by_region, effective_horizon_s=effective_horizon_s, config=config
    )
    runtime_metrics = {
        **_ground_runtime_metrics(record_tuple),
        **_ground_completed_throughput_metrics(record_tuple, effective_horizon_s=effective_horizon_s),
        **_replica_imbalance_metrics(replica_metrics),
    }
    runtime_metrics.update(
        {
            "ground_baseline_topology": _topology_name(policy_name),
            "ground_blind_push_request_fraction": 1.0 if record_tuple else None,
            "ground_hash_identity_fallback_count": len(
                [
                    item
                    for item in normalized
                    if item["hash_identity_source"] == "prefix_key_fallback"
                ]
            ),
            **(
                {
                    "client_program_completion_throughput_per_s": client_harness_metrics[
                        "program_completion_throughput_per_s"
                    ]
                }
                if client_harness is not None
                else {}
            ),
        }
    )
    communication_metrics = {
        "evaluated": True,
        "model": "central_or_gateway_terrestrial_internet_rtt_surrogate",
        "network_delay_p50_s": _percentile([record.ground_network_delay_s for record in record_tuple if record.completed], 0.50),
        "network_delay_p95_s": _percentile([record.ground_network_delay_s for record in record_tuple if record.completed], 0.95),
        "network_delay_p99_s": _percentile([record.ground_network_delay_s for record in record_tuple if record.completed], 0.99),
        "warnings": (
            "central/gateway network path is an explicit RTT surrogate; DNS, TLS, proxy retries, jitter, loss, and egress billing are not modeled",
        ),
    }
    return GroundRuntimeSimulationResult(
        records=record_tuple,
        decision_traces=tuple(decision_traces),
        state_summary={
            "request_count": len(record_tuple),
            "completed_count": len([item for item in record_tuple if item.completed]),
            "served_count": len([item for item in record_tuple if item.served]),
            "module_status": {
                "queue_model": "blind_push_replica_admission_queue",
                "routing_model": _topology_name(policy_name),
                "link_model": "terrestrial_internet_rtt_surrogate",
                "energy_model": "not_applicable_ground_baseline",
                "kv_state": "dispatch_routing_metadata_completion_kv_cache",
            },
            "regions": [to_jsonable(item) for item in config.regions],
            "replicas": replica_metrics,
            "client_harness": client_harness_metrics,
        },
        warnings=(
            "central/GKE-like comparison baseline uses a discrete-event execution and Internet-RTT surrogate",
            "central RR/LL/CH/SGL-like policies blind-push and do not inherit SkyWalker local-first, pending, or tau gates",
            "central_sgl_like_global_prefix_blind is not an exact SGLang Router reproduction",
            "gke_gateway_like backend selection is configured surrogate behavior, not an exact GKE Gateway reproduction",
            "valid_for_paper_main_results=false",
        ),
        runtime_metrics=runtime_metrics,
        communication_metrics=communication_metrics,
        energy_metrics={"evaluated": False, "not_applicable": True, "value": None},
        session_metrics={
            "evaluated": True,
            "kv_hit_tokens": sum(item.kv_hit_tokens or 0 for item in record_tuple),
            "kv_miss_tokens": sum(item.kv_miss_tokens or 0 for item in record_tuple),
            "kv_hit_rate_tokens": runtime_metrics.get("ground_prefix_cache_hit_rate_tokens"),
            "client_harness": client_harness_metrics,
        },
        region_metrics=_region_metrics(record_tuple, config.regions),
        provenance={
            "runtime_engine": "skywalker_paper_comparison_topology_surrogate",
            "policy": policy_name,
            "topology": _topology_name(policy_name),
            "central_region_id": central_region_id if policy_name != "ground_gke_gateway_like" else None,
            "gke_backend_policy": gateway_backend_policy if policy_name == "ground_gke_gateway_like" else None,
            "effective_ground_config": _effective_ground_config(config),
            "client_harness": client_harness_metrics,
            "effective_time_horizon_s": effective_horizon_s,
            "leo_trace_consumed": False,
            "leo_cost_consumed": False,
            "numerical_reproduction": False,
            "valid_for_paper_main_results": False,
        },
    )


def _select_global_replica(
    *,
    policy: str,
    backend_policy: str,
    replicas: list[_ReplicaState],
    request: dict[str, Any],
    dispatch_s: float,
    config: Any,
    round_robin_index: int,
    coordinator_inflight_dispatches: dict[str, int] | None = None,
) -> tuple[_ReplicaState, int]:
    ordered = sorted(replicas, key=lambda item: item.replica_id)
    mode = {
        "ground_central_rr": "round_robin",
        "ground_central_ll": "least_outstanding",
        "ground_central_ch": "consistent_hash",
        "ground_central_sgl_like": "global_prefix",
        "ground_gke_gateway_like": backend_policy,
    }[policy]
    if mode == "round_robin":
        selected = ordered[round_robin_index % len(ordered)]
        return selected, round_robin_index + 1
    if mode == "least_outstanding":
        in_transit = coordinator_inflight_dispatches or {}
        return min(
            ordered,
            key=lambda item: (
                item.outstanding_count(dispatch_s)
                + int(in_transit.get(item.replica_id, 0)),
                item.replica_id,
            ),
        ), round_robin_index
    if mode == "consistent_hash":
        target_id = _consistent_hash_target(
            request["hash_key"],
            all_target_ids=[item.replica_id for item in ordered],
            available_target_ids={item.replica_id for item in ordered},
            config=config,
        )
        return next(item for item in ordered if item.replica_id == target_id), round_robin_index
    return min(
        ordered,
        key=lambda item: (
            *_prefix_routing_sort_key(item.routing_index, request, at_s=dispatch_s),
            item.outstanding_count(dispatch_s),
            item.replica_id,
        ),
    ), round_robin_index


def _earliest_start_blind(replica: _ReplicaState, enqueue_s: float, batch_tokens: int, service_profile: Any) -> float:
    if batch_tokens > service_profile.continuous_batch_capacity_tokens:
        return math.inf
    # A direct blind push still has to wait for a continuous-batch slot.  The
    # queue is held by the replica rather than the SkyWalker regional LB.
    candidate = enqueue_s
    event_times = sorted(
        {
            candidate,
            *(value for job in replica.jobs for value in (job.start_s, job.end_s) if value >= candidate - 1e-12),
        }
    )
    for time_s in event_times:
        active = replica.active_jobs(time_s)
        if len(active) < service_profile.max_batch_size and sum(job.batch_tokens for job in active) + batch_tokens <= service_profile.continuous_batch_capacity_tokens:
            return max(candidate, time_s)
    return max([candidate, *(job.end_s for job in replica.jobs)], default=candidate)


def _client_to_ingress_one_way_s(
    config: Any,
    origin: Any,
    ingress: Any,
    *,
    request_key: str,
    dns_resolution_delay_s: float = 0.0,
) -> tuple[float, dict[str, Any] | None]:
    wan_s, profile = _region_one_way_s(config, origin, ingress, request_key=request_key)
    return origin.local_rtt_s / 2.0 + wan_s + dns_resolution_delay_s, profile


def _replica_to_client_one_way_s(
    config: Any, serving: Any, origin: Any, *, request_key: str
) -> tuple[float, dict[str, Any] | None]:
    """Return a client-visible stream leg, including local return RTT/2.

    A same-region replica still has to stream its response over the local
    client-to-LB/region path; returning zero here would undercount every
    local request's TTFT and E2E by half the configured local RTT.
    """

    wan_s, profile = _region_one_way_s(config, serving, origin, request_key=request_key)
    return origin.local_rtt_s / 2.0 + wan_s, profile


def _region_one_way_s(config: Any, source: Any, target: Any, *, request_key: str) -> tuple[float, dict[str, Any] | None]:
    if source.region_id == target.region_id:
        return 0.0, None
    rtt_s, profile = _wan_latency(config, source, target, request_key=request_key)
    return rtt_s / 2.0, profile


def _baseline_path(
    *,
    origin: str,
    ingress: str,
    serving: str,
    replica_id: str,
    client_to_ingress_s: float,
    ingress_to_replica_s: float,
    response_s: float,
    dns_resolution_delay_s: float = 0.0,
) -> tuple[dict[str, Any], ...]:
    legs: list[dict[str, Any]] = []
    if dns_resolution_delay_s > 0.0:
        legs.append(
            {
                "relation": "dns_resolution",
                "source": f"client:{origin}",
                "target": f"lb:{ingress}",
                "one_way_delay_s": dns_resolution_delay_s,
            }
        )
    legs.extend((
        {"relation": "client_to_baseline_ingress", "source": f"client:{origin}", "target": f"lb:{ingress}", "one_way_delay_s": max(0.0, client_to_ingress_s - dns_resolution_delay_s)},
        {"relation": "baseline_ingress_to_replica", "source": f"lb:{ingress}", "target": replica_id, "one_way_delay_s": ingress_to_replica_s},
        {"relation": "replica_to_client_stream", "source": replica_id, "target": f"client:{origin}", "one_way_delay_s": response_s},
    ))
    return tuple(legs)


def _decision_trace(request: dict[str, Any], record: RuntimeRequestRecord, dispatch_s: float, policy: str, replica: _ReplicaState, audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": request["request_id"],
        "arrival_time_s": request["arrival_time_s"],
        "decision_time_s": dispatch_s,
        "policy_name": policy,
        "candidate_action_count": 1,
        "local_candidate_count": 0,
        "remote_candidate_count": 0,
        "feasible_action_count": 1 if record.accepted else 0,
        "selected_action": {
            "action_type": record.action_type,
            "replica_id": replica.replica_id,
            "serving_region_id": replica.region_id,
            "blind_push": True,
        },
        "selection_reason": "configured_global_or_gateway_dispatcher",
        "audit": to_jsonable(audit),
        "bottleneck_reason": record.bottleneck_reason,
        "trace_schema_version": "orbitserve.runtime_decision_trace.v1",
        "warnings": (),
    }


def _topology_name(policy: str) -> str:
    if policy == "ground_gke_gateway_like":
        return "regional_ingress_gateway_global_backend_surrogate"
    return "single_us_global_coordinator_blind_push"


def _dispatcher_semantics(policy: str, backend_policy: str) -> str:
    return {
        "ground_central_rr": "global_replica_round_robin_blind_push",
        "ground_central_ll": "global_replica_least_outstanding_blind_push",
        "ground_central_ch": "global_replica_consistent_hash_blind_push",
        "ground_central_sgl_like": "global_prefix_router_blind_push_surrogate",
        "ground_gke_gateway_like": f"regional_gateway_global_{backend_policy}_surrogate",
    }[policy]


def _baseline_bottleneck(network_s: float, replica_wait_s: float, prefill_s: float, decode_s: float) -> str:
    return max(
        {
            "ground_internet": network_s,
            "ground_replica_pending": replica_wait_s,
            "ground_prefill": prefill_s,
            "ground_decode": decode_s,
        },
        key=lambda key: ({
            "ground_internet": network_s,
            "ground_replica_pending": replica_wait_s,
            "ground_prefill": prefill_s,
            "ground_decode": decode_s,
        }[key], key),
    )
