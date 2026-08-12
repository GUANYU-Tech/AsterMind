"""Compute-target eligibility contracts for trace-backed runtime evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common import to_jsonable


_NON_COMPUTE_ROLES = frozenset({"relay", "communication", "communications", "gateway", "non_compute", "none"})
_FAILED_STATES = frozenset({"failed", "faulted", "offline", "disabled", "unavailable", "decommissioned"})


@dataclass(frozen=True)
class ComputeEligibilitySet:
    """One authoritative separation between route nodes and compute targets.

    ``capacity_member_ids`` are hardware-capable members used to
    derive nominal per-node capacity. ``eligible_compute_node_ids`` are
    the subset currently allowed to receive work. Relay nodes deliberately do
    not constrain the routing graph.
    """

    capacity_member_ids: tuple[str, ...]
    eligible_compute_satellite_ids: tuple[str, ...]
    relay_satellite_ids: tuple[str, ...]
    excluded_reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)
    evaluated_at_s: float | None = None

    @property
    def capacity_member_count(self) -> int:
        return len(self.capacity_member_ids)

    @property
    def empty(self) -> bool:
        return not self.eligible_compute_satellite_ids

    @property
    def eligible_compute_node_ids(self) -> tuple[str, ...]:
        return self.eligible_compute_satellite_ids

    @property
    def relay_compute_node_ids(self) -> tuple[str, ...]:
        return self.relay_satellite_ids

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def build_compute_eligibility_set(
    *,
    satellite_catalog: tuple[dict[str, Any], ...],
    runtime_exclusions: dict[str, str | tuple[str, ...]] | None = None,
    evaluated_at_s: float | None = None,
) -> ComputeEligibilitySet:
    catalog_by_id = {
        str(item.get("satellite_id")): item
        for item in satellite_catalog
        if item.get("satellite_id") is not None
    }
    members = tuple(catalog_by_id)
    excluded: dict[str, tuple[str, ...]] = {}
    capacity_members: list[str] = []
    eligible: list[str] = []
    runtime_exclusions = runtime_exclusions or {}

    for satellite_id in members:
        reasons: list[str] = []
        catalog_row = catalog_by_id[satellite_id]
        reasons.extend(_hardware_reasons(catalog_row))
        if not reasons:
            capacity_members.append(satellite_id)
            reasons.extend(_fault_reasons(catalog_row))
            runtime_reason = runtime_exclusions.get(satellite_id)
            if isinstance(runtime_reason, str) and runtime_reason:
                reasons.append(runtime_reason)
            elif runtime_reason:
                reasons.extend(str(item) for item in runtime_reason)
        if reasons:
            excluded[satellite_id] = tuple(dict.fromkeys(reasons))
        else:
            eligible.append(satellite_id)

    capacity_set = set(capacity_members)
    relay_ids = tuple(satellite_id for satellite_id in catalog_by_id if satellite_id not in capacity_set)
    return ComputeEligibilitySet(
        capacity_member_ids=tuple(capacity_members),
        eligible_compute_satellite_ids=tuple(eligible),
        relay_satellite_ids=relay_ids,
        excluded_reasons=excluded,
        evaluated_at_s=evaluated_at_s,
    )


def _hardware_reasons(row: dict[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    if row.get("compute_capable") is False or row.get("accelerator_present") is False:
        reasons.append("hardware_role_not_compute_capable")
    role = str(row.get("hardware_role") or row.get("role") or "").strip().lower()
    roles = {str(item).strip().lower() for item in row.get("hardware_roles", ()) or ()}
    if role in _NON_COMPUTE_ROLES or (roles and roles.isdisjoint({"compute", "accelerator", "colocated"})):
        reasons.append("hardware_role_not_compute_capable")
    return tuple(dict.fromkeys(reasons))


def _fault_reasons(row: dict[str, Any]) -> tuple[str, ...]:
    state = str(row.get("failure_state") or row.get("operational_state") or row.get("status") or "").strip().lower()
    if row.get("failed") is True or row.get("operational") is False or state in _FAILED_STATES:
        return ("satellite_fault_or_unavailable",)
    return ()


__all__ = ["ComputeEligibilitySet", "build_compute_eligibility_set"]
