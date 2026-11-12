from __future__ import annotations
import math
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict

class ReferenceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

def _number(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is outside the supported numeric range") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


class InterLinkUnavailableError(ValueError):
    """A positive message cannot be delivered over an unavailable link."""


@dataclass(frozen=True)
class InterLinkState:
    """One directed link; capacity already includes ISLSim's FEC deduction.

    propagation_ns is pure one-way light travel time. carrier_efficiency is
    applied later and must only account for overhead absent from this capacity.
    An infeasible link may carry zero capacity. A feasible link must be positive.
    """

    capacity_gbps: float
    propagation_ns: float
    feasible: bool = True

    def __post_init__(self) -> None:
        capacity = _number(self.capacity_gbps, "capacity_gbps")
        propagation = _number(self.propagation_ns, "propagation_ns")
        if not isinstance(self.feasible, bool):
            raise TypeError("feasible must be a boolean")
        if self.feasible and capacity <= 0:
            raise ValueError("A feasible link requires positive capacity_gbps")
        object.__setattr__(self, "capacity_gbps", capacity)
        object.__setattr__(self, "propagation_ns", propagation)

    @classmethod
    def from_islsim(
        cls,
        *,
        total_bw_gbps: float,
        latency_ms: float,
        feasible: bool = True,
    ) -> InterLinkState:
        """Explicit unit boundary; no import or re-execution of ISLSim."""
        milliseconds = _number(latency_ms, "latency_ms")
        return cls(total_bw_gbps, milliseconds * 1_000_000, feasible)

