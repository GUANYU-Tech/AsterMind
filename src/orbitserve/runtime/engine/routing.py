"""Route facts shared by dynamic network execution and compute selection."""

from dataclasses import dataclass
from typing import Any

from ...common import to_jsonable
from ..state.network import NetworkUnavailable

RouteUnavailable = NetworkUnavailable


@dataclass(frozen=True)
class AccessCandidate:
    access_type: str
    ingress_id: str
    satellite_id: str
    start_s: float
    end_s: float
    propagation_delay_s: float


@dataclass(frozen=True)
class RoutePlan:
    access_type: str
    access_satellite_id: str
    compute_satellite_id: str
    inbound_legs: tuple[dict[str, Any], ...]
    outbound_legs: tuple[dict[str, Any], ...] = ()

    @property
    def path_id(self) -> str:
        return "->".join([self.inbound_legs[0]["source_node_id"],
                          *(leg["target_node_id"] for leg in self.inbound_legs)])

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)
