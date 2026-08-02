"""Internal Runtime engine facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..metrics.records import RuntimeRequestRecord


@dataclass(frozen=True)
class RuntimeEngineResult:
    records: tuple[RuntimeRequestRecord, ...]
    decision_traces: tuple[dict[str, Any], ...]
    state_summary: dict[str, Any]
    warnings: tuple[str, ...] = ()
    provenance: dict[str, Any] | None = None


__all__ = ["RuntimeEngineResult"]
