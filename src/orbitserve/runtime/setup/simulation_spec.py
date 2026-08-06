"""Aggregate input contract consumed by the Runtime evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common import to_jsonable
from ..contracts import RuntimeStrategySpec, ServingCapacityProfile


@dataclass(frozen=True)
class CommunicationIngressProfile:
    """Normalized Runtime view of an abstract satellite-internet ingress."""

    altitude_km: float | None = None
    uplink_bandwidth_bps: float | None = None
    downlink_bandwidth_bps: float | None = None

    @property
    def configured(self) -> bool:
        return bool(
            self.altitude_km is not None
            and self.altitude_km > 0.0
            and self.uplink_bandwidth_bps is not None
            and self.uplink_bandwidth_bps > 0.0
            and self.downlink_bandwidth_bps is not None
            and self.downlink_bandwidth_bps > 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_value(cls, value: Any) -> "CommunicationIngressProfile":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise TypeError("communication_ingress must be a mapping")
        return cls(
            altitude_km=_optional_float(value.get("altitude_km")),
            uplink_bandwidth_bps=_optional_float(value.get("uplink_bandwidth_bps")),
            downlink_bandwidth_bps=_optional_float(value.get("downlink_bandwidth_bps")),
        )


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
    orbit_trace_bundle: Any = None
    constellation: dict[str, Any] = field(default_factory=dict)
    serving_profile: ServingCapacityProfile | None = None
    energy_profile: dict[str, Any] = field(default_factory=dict)
    communication_ingress: CommunicationIngressProfile = field(
        default_factory=CommunicationIngressProfile
    )
    provenance: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    schema_version: str = "orbitserve.runtime_simulation_spec.v1"
    time_mode: str = "event_driven"
    time_step_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy", RuntimeStrategySpec.from_value(self.strategy))
        object.__setattr__(
            self,
            "serving_profile",
            ServingCapacityProfile.from_value(self.serving_profile),
        )
        object.__setattr__(
            self,
            "communication_ingress",
            CommunicationIngressProfile.from_value(self.communication_ingress),
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


__all__ = ["CommunicationIngressProfile", "RuntimeSimulationSpec"]


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
