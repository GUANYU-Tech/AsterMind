"""Aggregate input contract consumed by the Runtime evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common import to_jsonable
from ..contracts import RuntimeStrategySpec, ServingCapacityProfile


@dataclass(frozen=True)
class RuntimeSimulationSpec:
    """Fully assembled, immutable input boundary for one Runtime evaluation."""

    strategy: RuntimeStrategySpec
    scenario_id: str = "runtime_simulation"
    design_id: str = "runtime_design"
    time_horizon_s: float | None = None
    workload_bundle: dict[str, Any] = field(default_factory=dict)
    requests: tuple[dict[str, Any], ...] = ()
    sessions: tuple[dict[str, Any], ...] = ()
    regions: tuple[dict[str, Any], ...] = ()
    gateways: tuple[dict[str, Any], ...] = ()
    simulation_trace_ref: dict[str, Any] = field(default_factory=dict)
    constellation: dict[str, Any] = field(default_factory=dict)
    serving_profile: ServingCapacityProfile | None = None
    energy_profile: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: str = "orbitserve.runtime_simulation_spec.v1"
    time_mode: str = "event_driven"
    time_step_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


__all__ = ["RuntimeSimulationSpec"]
