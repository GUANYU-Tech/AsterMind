"""Workload-level aggregation and the stable Runtime result contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..common import to_jsonable
from .metrics.records import RuntimeRequestRecord, records_to_jsonable


@dataclass(frozen=True)
class RuntimePerformanceResult:
    evaluated: bool
    scenario: dict[str, Any]
    strategy: dict[str, Any]
    workload_metrics: dict[str, Any] = field(default_factory=dict)
    energy_metrics: dict[str, Any] = field(default_factory=dict)
    kv_metrics: dict[str, Any] = field(default_factory=dict)
    region_metrics: tuple[dict[str, Any], ...] = ()
    slo_metrics: dict[str, Any] = field(default_factory=dict)
    request_records: tuple[dict[str, Any], ...] = ()
    state_summary: dict[str, Any] = field(default_factory=dict)
    decision_trace: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: str = "orbitserve.runtime_performance"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def build_runtime_performance_result(*, simulation_spec: Any, engine_result: Any) -> RuntimePerformanceResult:
    records = tuple(engine_result.records)
    completed = tuple(record for record in records if record.completed)
    uncompleted = tuple(record for record in records if not record.completed)
    horizon_s = _horizon(records, getattr(simulation_spec, "time_horizon_s", None))
    completed_tokens = sum(record.prompt_tokens + record.output_tokens for record in completed)
    workload_metrics = {
        "total_request_count": len(records),
        "completed_request_count": len(completed),
        "uncompleted_request_count": len(uncompleted),
        "completion_ratio": len(completed) / len(records) if records else None,
        "uncompleted_ratio": len(uncompleted) / len(records) if records else None,
        "uncompleted_reason_breakdown": _counts(record.uncompleted_reason or "unknown" for record in uncompleted),
        "ttft_p50_s": _percentile([record.ttft_s for record in completed], 0.50),
        "ttft_p95_s": _percentile([record.ttft_s for record in completed], 0.95),
        "ttft_p99_s": _percentile([record.ttft_s for record in completed], 0.99),
        "e2e_p50_s": _percentile([record.e2e_s for record in completed], 0.50),
        "e2e_p95_s": _percentile([record.e2e_s for record in completed], 0.95),
        "e2e_p99_s": _percentile([record.e2e_s for record in completed], 0.99),
        "request_throughput_per_s": len(completed) / horizon_s if horizon_s > 0.0 else None,
        "token_throughput_per_s": completed_tokens / horizon_s if horizon_s > 0.0 else None,
        "prefill_queue_wait_p95_s": _percentile([record.prefill_queue_wait_s for record in completed], 0.95),
        "decode_queue_wait_p95_s": _percentile([record.decode_queue_wait_s for record in completed], 0.95),
    }
    energy_metrics = {
        "compute_energy_j": sum(float(record.compute_energy_j or 0.0) for record in records),
        "communication_energy_j": sum(float(record.communication_energy_j or 0.0) for record in records),
        "minimum_soc": min((float(record.soc_after) for record in records if record.soc_after is not None), default=None),
        "energy_violation_count": sum(1 for record in records if record.energy_violation),
    }
    hit_tokens = sum(float(record.kv_hit_tokens or 0.0) for record in records)
    miss_tokens = sum(float(record.kv_miss_tokens or 0.0) for record in records)
    kv_metrics = {
        "kv_hit_tokens": hit_tokens,
        "kv_miss_tokens": miss_tokens,
        "kv_hit_ratio": hit_tokens / (hit_tokens + miss_tokens) if hit_tokens + miss_tokens > 0.0 else None,
        "kv_evicted_bytes": sum(float(record.kv_evicted_bytes or 0.0) for record in records),
    }
    return RuntimePerformanceResult(
        evaluated=True,
        scenario={
            "scenario_id": simulation_spec.scenario_id,
            "design_id": simulation_spec.design_id,
            "time_horizon_s": simulation_spec.time_horizon_s,
            "schema_version": simulation_spec.schema_version,
        },
        strategy=simulation_spec.strategy.to_dict(),
        workload_metrics=workload_metrics,
        energy_metrics=energy_metrics,
        kv_metrics=kv_metrics,
        region_metrics=_region_metrics(records),
        slo_metrics={
            "evaluated": False,
            "slo_attainment_ratio": None,
            "slo_violation_ratio": None,
            "slo_class_breakdown": {},
            "warnings": ("SLO evaluation is reserved for future Runtime development",),
        },
        request_records=records_to_jsonable(records),
        state_summary=engine_result.state_summary,
        decision_trace=tuple(engine_result.decision_traces),
        provenance={**dict(simulation_spec.provenance or {}), **dict(engine_result.provenance or {})},
        warnings=tuple(dict.fromkeys((*simulation_spec.warnings, *engine_result.warnings))),
    )


def _region_metrics(records: tuple[RuntimeRequestRecord, ...]) -> tuple[dict[str, Any], ...]:
    by_region: dict[str, list[RuntimeRequestRecord]] = {}
    for record in records:
        by_region.setdefault(str(record.request.get("region_id") or "unknown"), []).append(record)
    return tuple(
        {
            "region_id": region,
            "request_count": len(rows),
            "completion_ratio": sum(1 for row in rows if row.completed) / len(rows),
        }
        for region, rows in sorted(by_region.items())
    )


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


def _horizon(records: tuple[RuntimeRequestRecord, ...], configured: Any) -> float:
    if configured is not None and float(configured) > 0.0:
        return float(configured)
    if not records:
        return 0.0
    starts = [float(record.arrival_time_s) for record in records]
    finishes = [float(record.arrival_time_s) + float(record.e2e_s or 0.0) for record in records if record.completed]
    return max(1.0, (max(finishes) if finishes else max(starts)) - min(starts))


def _percentile(values: list[float | None], percentile: float) -> float | None:
    numeric = sorted(float(value) for value in values if value is not None)
    if not numeric:
        return None
    if len(numeric) == 1:
        return numeric[0]
    position = percentile * (len(numeric) - 1)
    low = int(position)
    high = min(low + 1, len(numeric) - 1)
    fraction = position - low
    return numeric[low] * (1.0 - fraction) + numeric[high] * fraction


__all__ = ["RuntimePerformanceResult", "build_runtime_performance_result"]
