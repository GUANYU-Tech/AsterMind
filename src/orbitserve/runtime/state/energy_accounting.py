"""Token/byte energy accounting for Runtime event debits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EnergyIntensity:
    """Baseline-exclusive bus-side energy coefficients."""

    compute_standby_power_w: float = 0.0
    communication_standby_power_w: float = 0.0
    compute_energy_j_per_token: float = 0.5
    link_energy_j_per_byte: float = 1e-6

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_nonnegative_finite(value, name)

    @property
    def standby_power_w(self) -> float:
        return self.compute_standby_power_w + self.communication_standby_power_w

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicEnergyBreakdown:
    compute_j: float
    link_j: float

    @property
    def total_j(self) -> float:
        return self.compute_j + self.link_j

    def to_dict(self) -> dict[str, float]:
        return {
            "compute_j": self.compute_j,
            "link_j": self.link_j,
            "total_j": self.total_j,
        }


def compute_dynamic_energy_j(
    *,
    prompt_tokens: float,
    output_tokens: float,
    link_bytes: float,
    intensity: EnergyIntensity,
) -> DynamicEnergyBreakdown:
    """Return baseline-exclusive compute and physical-link bus energy."""

    prompt = _require_nonnegative_finite(prompt_tokens, "prompt_tokens")
    output = _require_nonnegative_finite(output_tokens, "output_tokens")
    physical_bytes = _require_nonnegative_finite(link_bytes, "link_bytes")
    return DynamicEnergyBreakdown(
        compute_j=(prompt + output) * intensity.compute_energy_j_per_token,
        link_j=physical_bytes * intensity.link_energy_j_per_byte,
    )


def _require_nonnegative_finite(value: Any, name: str) -> float:
    parsed = _number(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _require_fraction(value: Any, name: str) -> float:
    parsed = _require_nonnegative_finite(value, name)
    if parsed > 1:
        raise ValueError(f"{name} must be <= 1")
    return parsed


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


__all__ = [
    "DynamicEnergyBreakdown",
    "EnergyIntensity",
    "compute_dynamic_energy_j",
]
