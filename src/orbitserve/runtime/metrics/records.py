"""Structured request-level facts emitted by Runtime engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common import to_jsonable


_SECTION_DEFAULTS: dict[str, dict[str, Any]] = {
    "request": {
        "request_id": None, "arrival_time_s": 0.0, "prompt_tokens": 0,
        "output_tokens": 0, "session_id": None,
    },
    "decision": {
        "strategy": None, "action_type": "reject_drop", "action_components": (),
        "deployment_id": None, "access_satellite_id": None,
        "serving_satellite_id": None, "dvfs_scale": None,
        "fallback_used": False, "deferred": False, "defer_until_s": None,
        "session_action": None, "shadow_strategy": None, "shadow_strategies": (),
        "access_action_type": None, "prefill_action_type": None,
        "decode_action_type": None,
    },
    "placement": {
        "compute_owner_satellite_id": None, "compute_owner_satellite_ids": (),
    },
    "timing": {
        "access_wait_s": None, "link_queue_wait_s": None,
        "sgl_queue_wait_s": None, "isl_queue_wait_s": None,
        "link_transmission_s": None, "propagation_s": None,
        "prefill_queue_wait_s": None, "prefill_service_s": None,
        "decode_queue_wait_s": None, "decode_service_s": None,
        "uplink_s": None, "kv_transfer_s": None, "first_token_decode_s": None,
        "return_path_s": None, "first_token_return_s": None,
        "ttft_breakdown_s": {}, "ttft_s": None, "e2e_s": None,
        "response_streaming_model": None,
    },
    "routing": {
        "ingress": None, "input_route_nodes": (), "output_route_nodes": (),
        "route_node_ids": (), "route_links": (), "hop_count": 0,
        "selected_path": (),
    },
    "energy": {
        "soc_before": None, "soc_after": None, "compute_energy_j": None,
        "communication_energy_j": None, "energy_violation": None,
        "compute_sunlit": None, "access_sunlit": None, "energy_ledger": (),
    },
    "kv": {
        "kv_capacity_admission": None, "kv_serve_action": None,
        "kv_hit_tokens": None, "kv_miss_tokens": None,
        "context_recompute_tokens": None, "kv_reused_bytes": None,
        "kv_generated_bytes": None, "kv_evicted_bytes": None,
        "kv_evicted_entry_count": None,
        "kv_pages_required": None, "kv_pages_hit": None,
        "kv_pages_evicted": None, "hbm_admission_result": None,
        "hbm_occupancy_before": None, "hbm_occupancy_after": None,
    },
    "outcome": {
        "completed": False, "uncompleted_stage": None,
        "uncompleted_reason": None, "bottleneck_reason": None,
    },
    "provenance": {
        "scope": "trace_backed_runtime_surrogate", "trace_model": "surrogate_orbit_trace",
        "valid_for_paper_main_results": False,
    },
}

_FIELD_SECTION = {
    key: section
    for section, defaults in _SECTION_DEFAULTS.items()
    for key in defaults
}


@dataclass(frozen=True, init=False)
class RuntimeRequestRecord:
    request: dict[str, Any] = field(default_factory=dict)
    decision: dict[str, Any] = field(default_factory=dict)
    placement: dict[str, Any] = field(default_factory=dict)
    timing: dict[str, Any] = field(default_factory=dict)
    routing: dict[str, Any] = field(default_factory=dict)
    energy: dict[str, Any] = field(default_factory=dict)
    kv: dict[str, Any] = field(default_factory=dict)
    outcome: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **values: Any) -> None:
        sections = {
            name: _copy_defaults(defaults)
            for name, defaults in _SECTION_DEFAULTS.items()
        }
        for section_name in _SECTION_DEFAULTS:
            supplied = values.pop(section_name, None)
            if supplied is not None:
                if not isinstance(supplied, dict):
                    raise TypeError(f"{section_name} must be a mapping")
                sections[section_name].update(supplied)
        detail_provenance = values.pop("provenance", None)
        if detail_provenance is not None:
            if not isinstance(detail_provenance, dict):
                raise TypeError("provenance must be a mapping")
            sections["provenance"].update(detail_provenance)
        for key, value in values.items():
            section_name = _FIELD_SECTION.get(key)
            if section_name is None:
                raise TypeError(f"unexpected RuntimeRequestRecord field: {key}")
            sections[section_name][key] = value

        path = tuple(sections["routing"].get("selected_path") or ())
        if path:
            sections["routing"]["route_links"] = tuple(
                str(leg["link_id"]) for leg in path
                if isinstance(leg, dict) and leg.get("link_id") is not None
            )
            sections["routing"]["hop_count"] = sum(
                1 for leg in path
                if isinstance(leg, dict) and str(leg.get("kind", "")).lower() == "isl"
            )
            sections["timing"]["propagation_s"] = _sum_path_value(path, "propagation_delay_s")
        ledger = tuple(sections["energy"].get("energy_ledger") or ())
        if ledger:
            sections["energy"]["compute_energy_j"] = _sum_ledger_value(ledger, "compute_bus_j")
            sections["energy"]["communication_energy_j"] = _sum_ledger_value(ledger, "link_bus_j")
        if sections["decision"].get("strategy") is None:
            sections["decision"]["strategy"] = sections["provenance"].get("policy")
        if sections["request"].get("request_id") is None:
            raise TypeError("request_id is required")
        # TTFT and E2E describe a delivered response.  An incomplete request
        # may retain stage-level timing for diagnosis, but has no end-user
        # latency value.
        if not sections["outcome"].get("completed", False):
            sections["timing"]["ttft_s"] = None
            sections["timing"]["e2e_s"] = None
        for name, payload in sections.items():
            object.__setattr__(self, name, payload)

    def __getattr__(self, name: str) -> Any:
        section_name = _FIELD_SECTION.get(name)
        if section_name is not None:
            return object.__getattribute__(self, section_name).get(name)
        raise AttributeError(name)

    def with_updates(self, **values: Any) -> "RuntimeRequestRecord":
        payload = self.to_dict()
        for key, value in values.items():
            section_name = _FIELD_SECTION.get(key)
            if section_name is None:
                raise TypeError(f"unexpected RuntimeRequestRecord field: {key}")
            payload[section_name][key] = value
        return RuntimeRequestRecord(**payload)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable({
            "request": self.request, "decision": self.decision,
            "placement": self.placement, "timing": self.timing,
            "routing": self.routing, "energy": self.energy, "kv": self.kv,
            "outcome": self.outcome, "provenance": self.provenance,
        })


def _copy_defaults(defaults: dict[str, Any]) -> dict[str, Any]:
    return {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in defaults.items()
    }


def _sum_path_value(path: tuple[Any, ...], key: str) -> float:
    return sum(float(leg.get(key, 0.0) or 0.0) for leg in path if isinstance(leg, dict))


def _sum_ledger_value(ledger: tuple[Any, ...], key: str) -> float:
    return sum(float(row.get(key, 0.0) or 0.0) for row in ledger if isinstance(row, dict))


def records_to_jsonable(records: tuple[RuntimeRequestRecord, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(record.to_dict() for record in records)


__all__ = ["RuntimeRequestRecord", "records_to_jsonable"]
