"""Metric containers used by experiment, simulation, and DSE skeletons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .serialization import to_jsonable




@dataclass(frozen=True)
class PerformanceMetrics:
    evaluated: bool = False
    request_throughput_per_s: float | None = None
    token_throughput_per_s: float | None = None
    ttft_p95_s: float | None = None
    ttft_p99_s: float | None = None
    e2e_p95_s: float | None = None
    e2e_p99_s: float | None = None
    queue_delay_p95_s: float | None = None
    link_delay_p95_s: float | None = None
    prefill_tokens_per_s: float | None = None
    decode_tokens_per_s: float | None = None
    model_weight_memory_gb: float | None = None
    kv_cache_capacity_tokens: int | None = None
    satellite_count_used: int | None = None
    feasible: bool | None = None
    bottleneck: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ("performance model is not implemented yet",)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

@dataclass(frozen=True)
class SLOMetrics:
    evaluated: bool = False
    served_ratio: float | None = None
    rejected_ratio: float | None = None
    timeout_ratio: float | None = None
    energy_violation_count: int | None = None
    isl_congestion_p95: float | None = None
    sgl_congestion_p95: float | None = None
    warnings: tuple[str, ...] = ("SLO accounting is not implemented yet",)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class DSEMetrics:
    evaluated: bool = False
    pareto_frontier_evaluated: bool = False
    non_separability_regret: float | None = None
    dominated_design_count: int | None = None
    rank_disagreement_count: int | None = None
    warnings: tuple[str, ...] = ("DSE insight metrics are not implemented yet",)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


