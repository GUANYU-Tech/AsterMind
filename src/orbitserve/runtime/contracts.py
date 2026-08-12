"""Stable input contracts for deterministic Runtime simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..common import to_jsonable


RUNTIME_RESULT_FIELDS = frozenset(
    {
        "assigned_satellite",
        "assigned_satellite_id",
        "access_satellite_id",
        "access_compute_node_id",
        "assigned_compute_node_id",
        "selected_path",
        "completed",
        "completion_ratio",
        "uncompleted_reason",
        "ttft_s",
        "e2e_s",
        "soc_before",
        "soc_after",
        "dvfs_scale",
    }
)


INGRESS_SELECTIONS = frozenset(
    {
        "communication_satellite",
        "nearest_compute_satellite",
        "longest_visible_compute_satellite",
        "highest_energy_compute_satellite",
    }
)


@dataclass(frozen=True)
class RuntimeStrategySpec:
    """Complete scheduling rule fixed for one Runtime simulation."""

    ingress_selection: str = "communication_satellite"
    latency_weight: float = 0.25
    kv_reuse_weight: float = 0.25
    energy_weight: float = 0.25
    compute_load_weight: float = 0.25
    shadow_dvfs_enabled: bool = False
    shadow_dvfs_scale: float = 0.5
    kv_cache_copy_enabled: bool = False

    def __post_init__(self) -> None:
        if self.ingress_selection not in INGRESS_SELECTIONS:
            known = ", ".join(sorted(INGRESS_SELECTIONS))
            raise ValueError(f"ingress_selection must be one of: {known}")
        weights = self.weights
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("Runtime strategy weights must be finite and non-negative")
        if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Runtime strategy weights must sum to 1")
        if not math.isfinite(self.shadow_dvfs_scale) or not 0.0 < self.shadow_dvfs_scale <= 1.0:
            raise ValueError("shadow_dvfs_scale must be in (0, 1]")
        if type(self.kv_cache_copy_enabled) is not bool:
            raise ValueError("kv_cache_copy_enabled must be a bool")

    @property
    def weights(self) -> dict[str, float]:
        return {
            "latency": float(self.latency_weight),
            "kv_reuse": float(self.kv_reuse_weight),
            "energy": float(self.energy_weight),
            "compute_load": float(self.compute_load_weight),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RuntimeStrategySpec":
        return cls.from_value(data)

    @classmethod
    def from_value(cls, value: Any = None) -> "RuntimeStrategySpec":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        payload = _mapping_from_result(value)
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError("Unsupported RuntimeStrategySpec fields: " + ", ".join(unknown))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass(frozen=True)
class ServingCapacityProfile:
    """Runtime's normalized read-only view of Serving output."""

    serving_spec: dict[str, Any] = field(default_factory=dict)
    throughput_estimate: dict[str, Any] = field(default_factory=dict)
    evaluated: bool = False
    feasible: bool | None = None
    prefill_time_table: tuple[dict[str, Any], ...] = ()
    decode_time_table: tuple[dict[str, Any], ...] = ()
    kv_cache_bytes_per_token: float | None = None
    kv_cache_capacity_tokens: int | None = None
    max_context_tokens: int | None = None
    internal_comm_summary: dict[str, Any] = field(default_factory=dict)
    internal_comm_accounting: str = "embedded"
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_value(cls, value: Any) -> "ServingCapacityProfile":
        payload = _mapping_from_result(value)
        details = _dict(payload.get("details"))
        throughput = {**details, **{key: item for key, item in payload.items() if item is not None}}
        return cls(
            serving_spec=_dict(payload.get("serving_spec")),
            throughput_estimate=throughput,
            evaluated=bool(throughput.get("evaluated", False)),
            feasible=_optional_bool(throughput.get("feasible")),
            prefill_time_table=_table(throughput.get("prefill_time_table")),
            decode_time_table=_table(throughput.get("decode_time_table")),
            kv_cache_bytes_per_token=_optional_float(throughput.get("kv_cache_bytes_per_token")),
            kv_cache_capacity_tokens=_optional_int(throughput.get("kv_cache_capacity_tokens")),
            max_context_tokens=_optional_int(throughput.get("max_context_tokens")),
            internal_comm_summary=_dict(throughput.get("internal_comm_summary")),
            internal_comm_accounting=str(throughput.get("internal_comm_accounting", "embedded")),
            warnings=_warnings_from_payload(throughput),
        )

    @property
    def has_compute_capacity(self) -> bool:
        return bool(
            self.evaluated
            and self.feasible is not False
            and bool(self.prefill_time_table)
            and bool(self.decode_time_table)
        )

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def _mapping_from_result(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        return dict(converted) if isinstance(converted, dict) else {}
    raise TypeError(f"Expected a mapping-compatible value, got {type(value).__name__}")


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _table(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise TypeError("Serving time table must be a sequence of mappings")
    return tuple(dict(row) for row in value)


def _warnings_from_payload(payload: dict[str, Any]) -> tuple[str, ...]:
    raw = payload.get("warnings", ())
    if raw is None:
        return ()
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


__all__ = ["INGRESS_SELECTIONS", "RUNTIME_RESULT_FIELDS", "RuntimeStrategySpec", "ServingCapacityProfile"]
