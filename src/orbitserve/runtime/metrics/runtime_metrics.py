"""Aggregate runtime request records into OrbitServe metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ...common import to_jsonable
from ...common import PerformanceMetrics, SLOMetrics
from .runtime_records import RuntimeRequestRecord
from ...orbit import (
    CIRCULAR_GEOMETRY_TRACE,
    SURROGATE_ORBIT_TRACE,
    TRACE_BACKED_RUNTIME_SURROGATE_SCOPE,
    normalize_trace_scope,
)


@dataclass(frozen=True)
class RuntimeAggregateMetrics:
    runtime_metrics: dict[str, Any]
    performance: PerformanceMetrics
    slo: SLOMetrics
    bottleneck_breakdown: dict[str, Any]
    communication_metrics: dict[str, Any]
    energy_metrics: dict[str, Any]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def aggregate_runtime_records(
    records: tuple[RuntimeRequestRecord, ...],
    *,
    time_horizon_s: float | None,
    trace_model: str,
    scope: str,
) -> RuntimeAggregateMetrics:
    scope = normalize_trace_scope(scope)
    total = len(records)
    if total == 0:
        warnings = ("runtime evaluator has no request records to aggregate",)
        return RuntimeAggregateMetrics(
            runtime_metrics=_empty_runtime_metrics(trace_model=trace_model, scope=scope),
            performance=PerformanceMetrics(warnings=warnings),
            slo=SLOMetrics(warnings=warnings),
            bottleneck_breakdown={},
            communication_metrics=_unevaluated_group("link model is not implemented in compute-only runtime"),
            energy_metrics=_unevaluated_group("energy model is not implemented in compute-only runtime"),
            warnings=warnings,
        )

    completed = [record for record in records if record.completed]
    uncompleted = [record for record in records if not record.completed]
    fallback = [record for record in records if getattr(record, "fallback_used", False)]
    terrestrial = [record for record in records if _is_ground_record(record)]
    cross_region_ground = [
        record for record in terrestrial if getattr(record, "ground_cross_region", False)
    ]
    deferred = [record for record in records if getattr(record, "deferred", False)]
    energy_violations = [record for record in records if getattr(record, "energy_violation", False)]
    session_action_counts = _session_action_counts(records)
    kv_serve_action_counts = _kv_serve_action_counts(records)
    kv_post_turn_action_counts = _kv_post_turn_action_counts(records)
    kv_hit_tokens_total = _sum_numeric(getattr(record, "kv_hit_tokens", None) for record in records)
    kv_miss_tokens_total = _sum_numeric(getattr(record, "kv_miss_tokens", None) for record in records)
    kv_pages_required_total = _sum_numeric(getattr(record, "kv_pages_required", None) for record in records)
    kv_pages_hit_total = _sum_numeric(getattr(record, "kv_pages_hit", None) for record in records)
    kv_transfer_attempted_total = _sum_numeric(getattr(record, "kv_transfer_attempted_bytes", None) for record in records)
    kv_transfer_useful_total = _sum_numeric(getattr(record, "kv_transfer_useful_bytes", None) for record in records)
    horizon = _effective_horizon(records, time_horizon_s)
    completed_tokens = sum(
        record.prompt_tokens + record.output_tokens for record in completed
    )
    completed_output_tokens = sum(record.output_tokens for record in completed)
    bottlenecks: dict[str, int] = {}
    for record in records:
        reason = record.bottleneck_reason or "unknown"
        bottlenecks[reason] = bottlenecks.get(reason, 0) + 1
    action_type_counts = _action_type_counts(records)
    shadow_dvfs_count = len(
        [
            record
            for record in records
            if record.action_type == "shadow_dvfs"
            or "shadow_dvfs" in tuple(getattr(record, "shadow_strategies", ()) or ())
        ]
    )
    shadow_offload_count = len(
        [
            record
            for record in records
            if record.action_type == "shadow_offload"
            or "shadow_offload" in tuple(getattr(record, "shadow_strategies", ()) or ())
        ]
    )
    shadow_records = [
        record
        for record in records
        if getattr(record, "shadow_strategy", None)
        or getattr(record, "compute_sunlit", None) is False
        or getattr(record, "access_sunlit", None) is False
    ]
    shadow_completed = [record for record in shadow_records if record.completed]
    uncompleted_stage_breakdown = _value_counts(uncompleted, "uncompleted_stage")
    uncompleted_reason_breakdown = _value_counts(uncompleted, "uncompleted_reason")

    runtime_metrics = {
        "total_request_count": total,
        "completed_request_count": len(completed),
        "uncompleted_request_count": len(uncompleted),
        "completion_ratio": len(completed) / total,
        "uncompleted_ratio": len(uncompleted) / total,
        "uncompleted_stage_breakdown": uncompleted_stage_breakdown,
        "uncompleted_reason_breakdown": uncompleted_reason_breakdown,
        "fallback_request_fraction": len(fallback) / total,
        "deferred_request_fraction": len(deferred) / total,
        "leo_completion_ratio": len(
            [
                record
                for record in completed
                if not getattr(record, "fallback_used", False) and not _is_ground_record(record)
            ]
        )
        / total,
        "terrestrial_completion_ratio": len([record for record in completed if _is_ground_record(record)]) / total,
        "ground_cross_region_request_count": len(cross_region_ground),
        "ground_cross_region_request_fraction": (
            len(cross_region_ground) / len(terrestrial) if terrestrial else None
        ),
        "ground_network_delay_p50_s": _percentile(
            [getattr(record, "ground_network_delay_s", None) for record in terrestrial], 0.50
        ),
        "ground_network_delay_p95_s": _percentile(
            [getattr(record, "ground_network_delay_s", None) for record in terrestrial], 0.95
        ),
        "ground_network_delay_p99_s": _percentile(
            [getattr(record, "ground_network_delay_s", None) for record in terrestrial], 0.99
        ),
        "ground_load_balancer_queue_wait_p95_s": _percentile(
            [getattr(record, "ground_load_balancer_queue_wait_s", None) for record in terrestrial], 0.95
        ),
        "ground_replica_pending_wait_p95_s": _percentile(
            [getattr(record, "ground_replica_pending_wait_s", None) for record in terrestrial], 0.95
        ),
        "action_type_counts": action_type_counts,
        "shadow_affected_request_count": len(shadow_records),
        "shadow_affected_request_fraction": len(shadow_records) / total,
        "shadow_completion_ratio": len(shadow_completed) / len(shadow_records) if shadow_records else None,
        "shadow_dvfs_request_count": shadow_dvfs_count,
        "shadow_dvfs_request_fraction": shadow_dvfs_count / total,
        "shadow_offload_request_count": shadow_offload_count,
        "shadow_offload_request_fraction": shadow_offload_count / total,
        "shadow_local_compute_request_count": action_type_counts.get("shadow_local_compute", 0),
        "shadow_local_compute_request_fraction": action_type_counts.get("shadow_local_compute", 0) / total,
        "cross_satellite_forward_request_fraction": action_type_counts.get("cross_satellite_forward", 0) / total,
        "ttft_p50_s": _percentile([record.ttft_s for record in completed], 0.50),
        "ttft_p95_s": _percentile([record.ttft_s for record in completed], 0.95),
        "ttft_p99_s": _percentile([record.ttft_s for record in completed], 0.99),
        "tpot_p50_s": _percentile(_tpot_values(completed), 0.50),
        "tpot_p95_s": _percentile(_tpot_values(completed), 0.95),
        "tpot_p99_s": _percentile(_tpot_values(completed), 0.99),
        "e2e_p50_s": _percentile([record.e2e_s for record in completed], 0.50),
        "e2e_p95_s": _percentile([record.e2e_s for record in completed], 0.95),
        "e2e_p99_s": _percentile([record.e2e_s for record in completed], 0.99),
        "useful_request_throughput_per_s": len(completed) / horizon if horizon > 0 else None,
        "useful_token_throughput_per_s": completed_tokens / horizon if horizon > 0 else None,
        "completed_request_throughput_per_s": len(completed) / horizon if horizon > 0 else None,
        "completed_token_throughput_per_s": completed_tokens / horizon if horizon > 0 else None,
        "completed_output_token_throughput_per_s": (
            completed_output_tokens / horizon if horizon > 0 else None
        ),
        "completed_throughput_token_accounting": "prompt_plus_output_and_output_only_both_reported",
        "prefill_queue_wait_p95_s": _percentile([record.prefill_queue_wait_s for record in completed], 0.95),
        "decode_queue_wait_p95_s": _percentile([record.decode_queue_wait_s for record in completed], 0.95),
        "sticky_kv_owner_requests": kv_serve_action_counts.get("sticky_kv_owner", session_action_counts.get("sticky_kv_owner", 0)),
        "recompute_context_requests": kv_serve_action_counts.get("recompute_context", session_action_counts.get("recompute_context", 0)),
        "migrate_kv_requests": kv_serve_action_counts.get("migrate_kv", session_action_counts.get("migrate_kv", 0)),
        "replicate_kv_requests": kv_post_turn_action_counts.get("replicate_kv", session_action_counts.get("replicate_kv", 0)),
        "kv_serve_action_counts": kv_serve_action_counts,
        "kv_post_turn_action_counts": kv_post_turn_action_counts,
        "kv_hit_tokens_total": kv_hit_tokens_total,
        "kv_miss_tokens_total": kv_miss_tokens_total,
        "kv_hit_rate_tokens": (
            kv_hit_tokens_total / (kv_hit_tokens_total + kv_miss_tokens_total)
            if (kv_hit_tokens_total + kv_miss_tokens_total) > 0
            else None
        ),
        "kv_pages_required_total": kv_pages_required_total,
        "kv_pages_hit_total": kv_pages_hit_total,
        "kv_hit_rate_pages": (
            kv_pages_hit_total / kv_pages_required_total
            if kv_pages_required_total > 0
            else None
        ),
        "kv_reuse_bytes_total": _sum_numeric(getattr(record, "kv_reused_bytes", None) for record in records),
        "kv_generated_bytes_total": _sum_numeric(getattr(record, "kv_generated_bytes", None) for record in records),
        "kv_migration_bytes_total": _sum_numeric(getattr(record, "kv_migration_bytes", None) for record in records),
        "kv_replication_bytes_total": _sum_numeric(getattr(record, "kv_replication_bytes", None) for record in records),
        "kv_transfer_attempted_bytes_total": kv_transfer_attempted_total,
        "kv_transfer_useful_bytes_total": kv_transfer_useful_total,
        "kv_transfer_waste_ratio": (
            max(0.0, kv_transfer_attempted_total - kv_transfer_useful_total) / kv_transfer_attempted_total
            if kv_transfer_attempted_total > 0
            else None
        ),
        "kv_transfer_interruption_count": len([record for record in records if getattr(record, "kv_transfer_interrupted", False)]),
        "kv_transfer_expiry_count": len([record for record in records if getattr(record, "kv_transfer_expired", False)]),
        "kv_eviction_count": int(_sum_numeric(getattr(record, "kv_pages_evicted", None) for record in records)),
        "post_turn_replication_request_fraction": (
            kv_post_turn_action_counts.get("replicate_kv", 0) / total
            if total > 0
            else None
        ),
        "replication_event_count": kv_post_turn_action_counts.get("replicate_kv", 0),
        "hbm_occupancy_p50": _percentile([getattr(record, "hbm_occupancy_after", None) for record in records], 0.50),
        "hbm_occupancy_p95": _percentile([getattr(record, "hbm_occupancy_after", None) for record in records], 0.95),
        "hbm_occupancy_max": _max_numeric(getattr(record, "hbm_occupancy_after", None) for record in records),
        "context_recompute_tokens_total": _context_recompute_tokens_total(records),
        "kv_capacity_reject_count": _kv_capacity_reject_count(records),
        "link_queue_wait_p95_s": (
            _percentile(_link_queue_waits(completed), 0.95)
            if _uses_trace_backed_surrogate_model(trace_model, scope)
            else _percentile(
                [getattr(record, "ground_network_delay_s", None) for record in completed],
                0.95,
            )
            if _uses_terrestrial_ground_model(trace_model, scope)
            else None
        ),
        "access_wait_p95_s": _percentile([record.access_wait_s for record in completed], 0.95),
        "energy_violation_count": (
            len(energy_violations)
            if _uses_trace_backed_surrogate_model(trace_model, scope)
            else None
        ),
        "sgl_congestion_p95": (
            _percentile([getattr(record, "sgl_queue_wait_s", None) for record in completed], 0.95)
            if _uses_trace_backed_surrogate_model(trace_model, scope)
            else None
        ),
        "isl_congestion_p95": (
            _percentile([getattr(record, "isl_queue_wait_s", None) for record in completed], 0.95)
            if _uses_trace_backed_surrogate_model(trace_model, scope)
            else None
        ),
        "eta_serving": None,
        "eta_prefill_capacity": None,
        "eta_decode_capacity": None,
        "eta_compute_useful": None,
        "scope": scope,
        "trace_model": trace_model,
        "valid_for_paper_main_results": False,
    }
    warnings = _warnings_for_scope(trace_model=trace_model, scope=scope)
    performance = PerformanceMetrics(
        evaluated=True,
        request_throughput_per_s=runtime_metrics["useful_request_throughput_per_s"],
        token_throughput_per_s=runtime_metrics["useful_token_throughput_per_s"],
        ttft_p95_s=runtime_metrics["ttft_p95_s"],
        ttft_p99_s=runtime_metrics["ttft_p99_s"],
        e2e_p95_s=runtime_metrics["e2e_p95_s"],
        e2e_p99_s=runtime_metrics["e2e_p99_s"],
        queue_delay_p95_s=max(
            runtime_metrics["prefill_queue_wait_p95_s"] or 0.0,
            runtime_metrics["decode_queue_wait_p95_s"] or 0.0,
        ),
        link_delay_p95_s=(
            runtime_metrics["ground_network_delay_p95_s"]
            if _uses_terrestrial_ground_model(trace_model, scope)
            else runtime_metrics["link_queue_wait_p95_s"]
        ),
        feasible=True,
        bottleneck=max(bottlenecks, key=bottlenecks.get) if bottlenecks else None,
        details={
            "scope": scope,
            "trace_model": trace_model,
            "valid_for_paper_main_results": False,
            "bottleneck_breakdown": bottlenecks,
            "action_type_counts": action_type_counts,
        },
        warnings=warnings,
    )
    slo = SLOMetrics(
        evaluated=False,
        warnings=(*warnings, "SLO evaluation is intentionally outside Runtime"),
    )
    return RuntimeAggregateMetrics(
        runtime_metrics=runtime_metrics,
        performance=performance,
        slo=slo,
        bottleneck_breakdown=bottlenecks,
        communication_metrics=_communication_metrics(records, trace_model=trace_model, scope=scope),
        energy_metrics=_energy_metrics(records, trace_model=trace_model, scope=scope),
        warnings=warnings,
    )


def _empty_runtime_metrics(*, trace_model: str, scope: str) -> dict[str, Any]:
    return {
        "total_request_count": 0,
        "completed_request_count": 0,
        "uncompleted_request_count": 0,
        "completion_ratio": None,
        "uncompleted_ratio": None,
        "uncompleted_stage_breakdown": {},
        "uncompleted_reason_breakdown": {},
        "terrestrial_completion_ratio": None,
        "ground_cross_region_request_count": 0,
        "ground_cross_region_request_fraction": None,
        "ground_network_delay_p50_s": None,
        "ground_network_delay_p95_s": None,
        "ground_network_delay_p99_s": None,
        "ground_load_balancer_queue_wait_p95_s": None,
        "ground_replica_pending_wait_p95_s": None,
        "action_type_counts": {},
        "shadow_affected_request_count": 0,
        "shadow_affected_request_fraction": None,
        "shadow_completion_ratio": None,
        "shadow_dvfs_request_count": 0,
        "shadow_dvfs_request_fraction": None,
        "shadow_offload_request_count": 0,
        "shadow_offload_request_fraction": None,
        "shadow_local_compute_request_count": 0,
        "shadow_local_compute_request_fraction": None,
        "cross_satellite_forward_request_fraction": None,
        "ttft_p95_s": None,
        "e2e_p95_s": None,
        "useful_request_throughput_per_s": None,
        "useful_token_throughput_per_s": None,
        "completed_request_throughput_per_s": None,
        "completed_token_throughput_per_s": None,
        "completed_output_token_throughput_per_s": None,
        "completed_throughput_token_accounting": "prompt_plus_output_and_output_only_both_reported",
        "sticky_kv_owner_requests": 0,
        "recompute_context_requests": 0,
        "migrate_kv_requests": 0,
        "replicate_kv_requests": 0,
        "kv_serve_action_counts": {},
        "kv_post_turn_action_counts": {},
        "kv_hit_tokens_total": 0.0,
        "kv_miss_tokens_total": 0.0,
        "kv_hit_rate_tokens": None,
        "kv_pages_required_total": 0.0,
        "kv_pages_hit_total": 0.0,
        "kv_hit_rate_pages": None,
        "kv_reuse_bytes_total": 0.0,
        "kv_generated_bytes_total": 0.0,
        "kv_migration_bytes_total": 0.0,
        "kv_replication_bytes_total": 0.0,
        "kv_transfer_attempted_bytes_total": 0.0,
        "kv_transfer_useful_bytes_total": 0.0,
        "kv_transfer_waste_ratio": None,
        "kv_transfer_interruption_count": 0,
        "kv_transfer_expiry_count": 0,
        "kv_eviction_count": 0,
        "post_turn_replication_request_fraction": None,
        "replication_event_count": 0,
        "hbm_occupancy_p50": None,
        "hbm_occupancy_p95": None,
        "hbm_occupancy_max": None,
        "context_recompute_tokens_total": 0.0,
        "kv_capacity_reject_count": 0,
        "scope": scope,
        "trace_model": trace_model,
        "valid_for_paper_main_results": False,
    }


def _unevaluated_group(reason: str) -> dict[str, Any]:
    return {"evaluated": False, "value": None, "warnings": (reason,)}


def _communication_metrics(
    records: tuple[RuntimeRequestRecord, ...],
    *,
    trace_model: str,
    scope: str,
) -> dict[str, Any]:
    scope = normalize_trace_scope(scope)
    if _uses_terrestrial_ground_model(trace_model, scope):
        completed = [record for record in records if record.completed]
        terrestrial = [record for record in completed if _is_ground_record(record)]
        cross_region = [
            record for record in terrestrial if getattr(record, "ground_cross_region", False)
        ]
        return {
            "evaluated": True,
            "model": "terrestrial_internet_rtt_surrogate",
            "ground_network_delay_p50_s": _percentile(
                [getattr(record, "ground_network_delay_s", None) for record in terrestrial], 0.50
            ),
            "ground_network_delay_p95_s": _percentile(
                [getattr(record, "ground_network_delay_s", None) for record in terrestrial], 0.95
            ),
            "ground_network_delay_p99_s": _percentile(
                [getattr(record, "ground_network_delay_s", None) for record in terrestrial], 0.99
            ),
            "ground_cross_region_request_fraction": (
                len(cross_region) / len(terrestrial) if terrestrial else None
            ),
            "warnings": (
                "terrestrial Internet delay is a configurable RTT/geometry surrogate, not a packet-level trace",
            ),
        }
    if not _uses_trace_backed_surrogate_model(trace_model, scope):
        return _unevaluated_group("link model is not implemented in compute-only runtime")
    completed = [record for record in records if record.completed]
    return {
        "evaluated": True,
        "link_queue_wait_p95_s": _percentile(_link_queue_waits(completed), 0.95),
        "sgl_congestion_p95": _percentile([getattr(record, "sgl_queue_wait_s", None) for record in completed], 0.95),
        "isl_congestion_p95": _percentile([getattr(record, "isl_queue_wait_s", None) for record in completed], 0.95),
        "warnings": ("SGL/ISL congestion values are queue-wait proxies, not calibrated congestion",),
    }


def _energy_metrics(
    records: tuple[RuntimeRequestRecord, ...],
    *,
    trace_model: str,
    scope: str,
) -> dict[str, Any]:
    scope = normalize_trace_scope(scope)
    if _uses_terrestrial_ground_model(trace_model, scope):
        return {
            "evaluated": False,
            "not_applicable": True,
            "value": None,
            "warnings": ("orbital energy/SOC is not applicable to the terrestrial baseline",),
        }
    if not _uses_trace_backed_surrogate_model(trace_model, scope):
        return _unevaluated_group("energy model is not implemented in compute-only runtime")
    soc_values = [
        getattr(record, "soc_after", None)
        for record in records
        if getattr(record, "soc_after", None) is not None
    ]
    return {
        "evaluated": True,
        "energy_violation_count": len([record for record in records if getattr(record, "energy_violation", False)]),
        "min_observed_soc": min(soc_values) if soc_values else None,
        "shadow_min_observed_soc": min(
            (
                float(getattr(record, "soc_after"))
                for record in records
                if getattr(record, "soc_after", None) is not None
                and (
                    getattr(record, "shadow_strategy", None)
                    or getattr(record, "compute_sunlit", None) is False
                    or getattr(record, "access_sunlit", None) is False
                )
            ),
            default=None,
        ),
        "warnings": ("energy/SOC values are deterministic surrogate metrics",),
    }


def _warnings_for_scope(*, trace_model: str, scope: str) -> tuple[str, ...]:
    scope = normalize_trace_scope(scope)
    if _uses_terrestrial_ground_model(trace_model, scope):
        return (
            "scope=skywalker_ground_reproduction_surrogate",
            "trace_model=terrestrial_network_surrogate",
            "valid_for_paper_main_results=false",
            "SkyWalker control logic is reproduced, while unpublished workload/network/engine parameters remain explicit surrogates",
            "ground cost_per_served_* remains null unless a separate terrestrial cost model is configured",
        )
    if trace_model == SURROGATE_ORBIT_TRACE and scope == TRACE_BACKED_RUNTIME_SURROGATE_SCOPE:
        return (
            f"scope={TRACE_BACKED_RUNTIME_SURROGATE_SCOPE}",
            f"trace_model={SURROGATE_ORBIT_TRACE}",
            "valid_for_paper_main_results=false",
            "SGL/ISL congestion values are queue-wait proxies, not calibrated congestion",
            "cost_per_served_* is intentionally null for preliminary surrogate runtime analysis",
        )
    if trace_model == CIRCULAR_GEOMETRY_TRACE and scope == "geometry_trace_surrogate":
        return (
            "scope=geometry_trace_surrogate",
            f"trace_model={CIRCULAR_GEOMETRY_TRACE}",
            "valid_for_paper_main_results=false",
            "SGL/ISL congestion values are queue-wait proxies, not calibrated congestion",
            "energy/SOC values are deterministic surrogate metrics",
            "cost_per_served_* is intentionally null for circular geometry runtime surrogate analysis",
        )
    return (
        "scope=idealized_compute_only",
        "trace_model=ideal_full_visibility",
        "valid_for_paper_main_results=false",
        "cost_per_served_* is intentionally null for Phase 1a",
    )


def _uses_trace_backed_surrogate_model(trace_model: str, scope: str) -> bool:
    scope = normalize_trace_scope(scope)
    return (
        trace_model == SURROGATE_ORBIT_TRACE
        and scope == TRACE_BACKED_RUNTIME_SURROGATE_SCOPE
    ) or (
        trace_model == CIRCULAR_GEOMETRY_TRACE
        and scope == "geometry_trace_surrogate"
    )


def _uses_terrestrial_ground_model(trace_model: str, scope: str) -> bool:
    return (
        str(trace_model) == "terrestrial_network_surrogate"
        and str(scope) == "skywalker_ground_reproduction_surrogate"
    )


def _is_ground_record(record: RuntimeRequestRecord) -> bool:
    return str(getattr(record, "action_type", "")).startswith("ground_") or getattr(
        record, "ground_serving_region_id", None
    ) is not None


def _effective_horizon(records: tuple[RuntimeRequestRecord, ...], configured_horizon_s: float | None) -> float:
    if configured_horizon_s is not None and configured_horizon_s > 0:
        return configured_horizon_s
    finish_times = [
        record.arrival_time_s + (record.e2e_s or 0.0)
        for record in records
        if record.completed and record.e2e_s is not None
    ]
    if finish_times:
        return max(finish_times) - min(record.arrival_time_s for record in records)
    return max(1.0, max(record.arrival_time_s for record in records) - min(record.arrival_time_s for record in records))


def _tpot_values(records: list[RuntimeRequestRecord]) -> list[float | None]:
    values: list[float | None] = []
    for record in records:
        if record.output_tokens <= 1 or record.e2e_s is None or record.ttft_s is None:
            values.append(None)
        else:
            values.append(max(0.0, record.e2e_s - record.ttft_s) / (record.output_tokens - 1))
    return values


def _link_queue_waits(records: list[RuntimeRequestRecord]) -> list[float | None]:
    values: list[float | None] = []
    for record in records:
        if record.link_queue_wait_s is not None:
            values.append(record.link_queue_wait_s)
        else:
            sgl = getattr(record, "sgl_queue_wait_s", None) or 0.0
            isl = getattr(record, "isl_queue_wait_s", None) or 0.0
            values.append(sgl + isl)
    return values



def _session_action_counts(records: tuple[RuntimeRequestRecord, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        action = getattr(record, "session_action", None)
        if action is None:
            continue
        key = str(action)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _value_counts(records: list[RuntimeRequestRecord], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = getattr(record, field_name, None)
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _kv_serve_action_counts(records: tuple[RuntimeRequestRecord, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        action = getattr(record, "kv_serve_action", None)
        if action is None:
            continue
        key = str(action)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _kv_post_turn_action_counts(records: tuple[RuntimeRequestRecord, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        action = getattr(record, "kv_post_turn_action", None)
        if action is None or str(action) == "none":
            continue
        key = str(action)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _action_type_counts(records: tuple[RuntimeRequestRecord, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        action = getattr(record, "action_type", None)
        if action is None:
            continue
        key = str(action)
        counts[key] = counts.get(key, 0) + 1
    return counts



def _context_recompute_tokens_total(records: tuple[RuntimeRequestRecord, ...]) -> float:
    total = 0.0
    for record in records:
        direct = _optional_number(getattr(record, "context_recompute_tokens", None))
        if direct is not None:
            total += direct
            continue
        provenance = getattr(record, "provenance", {}) or {}
        total += _first_numeric_from_mappings(
            provenance,
            (
                "context_recompute_tokens",
                "recompute_context_tokens",
                "context_recompute_tokens_total",
            ),
        )
    return total


def _kv_capacity_reject_count(records: tuple[RuntimeRequestRecord, ...]) -> int:
    rejected_values = {
        "fail",
        "failed",
        "false",
        "reject",
        "rejected",
        "deny",
        "denied",
        "capacity_reject",
        "kv_capacity_reject",
    }
    count = 0
    for record in records:
        admission = getattr(record, "kv_capacity_admission", None)
        if admission is not None and str(admission).lower() in rejected_values:
            count += 1
            continue
        if not record.completed and getattr(record, "uncompleted_reason", None) == "kv_capacity_unavailable":
            count += 1
    return count


def _max_numeric(values: Any) -> float | None:
    numeric = [
        number
        for number in (_optional_number(value) for value in values)
        if number is not None
    ]
    return max(numeric) if numeric else None


def _first_numeric_from_mappings(mapping: dict[str, Any], keys: tuple[str, ...]) -> float:
    for key in keys:
        value = _nested_lookup(mapping, key)
        number = _optional_number(value)
        if number is not None:
            return number
    return 0.0


def _nested_lookup(value: Any, key: str) -> Any:
    if not isinstance(value, dict):
        return None
    if key in value:
        return value[key]
    for item in value.values():
        found = _nested_lookup(item, key)
        if found is not None:
            return found
    return None


def _sum_numeric(values: Any) -> float:
    return sum(_optional_number(value) or 0.0 for value in values)


def _optional_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not number == number or number in (float("inf"), float("-inf")):
        return None
    return number


def _percentile(values: list[float | None], percentile: float) -> float | None:
    clean = sorted(float(value) for value in values if value is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * percentile
    lower = int(position)
    upper = min(len(clean) - 1, lower + 1)
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


__all__ = ["RuntimeAggregateMetrics", "aggregate_runtime_records"]
