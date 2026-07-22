"""Shared event-engine result contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..metrics.records import RuntimeRequestRecord

@dataclass(frozen=True)
class RuntimeEngineResult:
    records: tuple[RuntimeRequestRecord, ...]
    decision_traces: tuple[dict[str, Any], ...]
    state_summary: dict[str, Any]
    warnings: tuple[str, ...]
    evaluated: bool = True
    scope: str = "trace_backed_runtime_surrogate"
    trace_model: str = "surrogate_orbit_trace"
    runtime_metrics: dict[str, Any] | None = None
    communication_metrics: dict[str, Any] | None = None
    energy_metrics: dict[str, Any] | None = None
    session_metrics: dict[str, Any] | None = None
    region_metrics: tuple[dict[str, Any], ...] | None = None
    provenance: dict[str, Any] | None = None
    valid_for_paper_main_results: bool = False

