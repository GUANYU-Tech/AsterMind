"""Aggregate input contract consumed by the Runtime evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import math

from ...common import to_jsonable
from ..contracts import RuntimeStrategySpec, ServingCapacityProfile
from .network_spec import NetworkSpec


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
    environment_state: Any = None
    constellation: dict[str, Any] = field(default_factory=dict)
    serving_profile: ServingCapacityProfile | None = None
    energy_profile: dict[str, Any] = field(default_factory=dict)
    network: NetworkSpec | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: str = "orbitserve.runtime_simulation_spec.v1"
    time_mode: str = "event_driven"
    time_step_s: float | None = None
    physical_state_ref: str | None = None

    def __post_init__(self) -> None:
        if self.time_mode != "event_driven":
            raise ValueError(
                "Runtime time_mode must be event_driven; topology spacing belongs to Orbit"
            )
        if self.time_horizon_s is not None and (
            not math.isfinite(self.time_horizon_s) or self.time_horizon_s <= 0
        ):
            raise ValueError("Runtime time_horizon_s must be finite and positive")
        object.__setattr__(
            self, "strategy", RuntimeStrategySpec.from_value(self.strategy)
        )
        object.__setattr__(
            self,
            "serving_profile",
            ServingCapacityProfile.from_value(self.serving_profile),
        )
        object.__setattr__(
            self,
            "network",
            NetworkSpec.from_value(self.network),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


__all__ = ["RuntimeSimulationSpec"]
