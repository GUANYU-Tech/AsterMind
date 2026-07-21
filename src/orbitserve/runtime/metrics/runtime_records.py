"""Request-level runtime records for OrbitServe event simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common import to_jsonable


@dataclass(frozen=True)
class RuntimeRequestRecord:
    request_id: str
    arrival_time_s: float
    deployment_id: str | None
    action_type: str
    completed: bool
    uncompleted_stage: str | None = None
    uncompleted_reason: str | None = None
    action_components: tuple[str, ...] = ()
    prompt_tokens: int = 0
    output_tokens: int = 0
    prefill_queue_wait_s: float | None = None
    decode_queue_wait_s: float | None = None
    link_queue_wait_s: float | None = None
    access_wait_s: float | None = None
    sgl_queue_wait_s: float | None = None
    isl_queue_wait_s: float | None = None
    link_transmission_s: float | None = None
    ground_origin_region_id: str | None = None
    ground_serving_region_id: str | None = None
    ground_replica_id: str | None = None
    ground_cross_region: bool | None = None
    ground_network_delay_s: float | None = None
    ground_load_balancer_queue_wait_s: float | None = None
    ground_replica_pending_wait_s: float | None = None
    ground_prefix_cache_hit_ratio: float | None = None
    prefill_service_s: float | None = None
    decode_service_s: float | None = None
    uplink_s: float | None = None
    kv_transfer_s: float | None = None
    first_token_decode_s: float | None = None
    return_path_s: float | None = None
    ttft_breakdown_s: dict[str, Any] = field(default_factory=dict)
    ttft_s: float | None = None
    e2e_s: float | None = None
    assigned_satellite_id: str | None = None
    compute_owner_satellite_id: str | None = None
    compute_owner_satellite_ids: tuple[str, ...] = ()
    access_satellite_id: str | None = None
    first_token_return_s: float | None = None
    session_kv_migration_bytes: float | None = None
    session_kv_visible_s: float | None = None
    kv_capacity_admission: str | None = None
    kv_serve_action: str | None = None
    kv_post_turn_action: str | None = None
    kv_background_event_count: int | None = None
    kv_hit_tokens: int | None = None
    kv_miss_tokens: int | None = None
    context_recompute_tokens: int | None = None
    kv_reused_bytes: float | None = None
    kv_generated_bytes: float | None = None
    kv_migration_bytes: float | None = None
    kv_replication_bytes: float | None = None
    kv_transfer_attempted_bytes: float | None = None
    kv_transfer_delivered_bytes: float | None = None
    kv_transfer_committed_page_bytes: float | None = None
    kv_transfer_useful_bytes: float | None = None
    kv_transfer_interrupted: bool | None = None
    kv_transfer_expired: bool | None = None
    kv_transfer_interruption_reason: str | None = None
    kv_transfer_expiry_reason: str | None = None
    kv_pages_required: int | None = None
    kv_pages_hit: int | None = None
    kv_pages_migrated: int | None = None
    kv_pages_replicated: int | None = None
    kv_pages_evicted: int | None = None
    hbm_admission_result: str | None = None
    hbm_occupancy_before: float | None = None
    hbm_occupancy_after: float | None = None
    visible_kv_delay_s: float | None = None
    hidden_kv_delay_s: float | None = None
    response_streaming_model: str | None = None
    route_node_ids: tuple[str, ...] = ()
    selected_path: tuple[dict[str, Any], ...] = ()
    soc_before: float | None = None
    soc_after: float | None = None
    energy_violation: bool | None = None
    compute_sunlit: bool | None = None
    access_sunlit: bool | None = None
    dvfs_multiplier: float | None = None
    shadow_strategy: str | None = None
    shadow_strategies: tuple[str, ...] = ()
    access_action_type: str | None = None
    prefill_action_type: str | None = None
    decode_action_type: str | None = None
    energy_ledger: tuple[dict[str, Any], ...] = ()
    fallback_used: bool = False
    deferred: bool = False
    defer_until_s: float | None = None
    session_action: str | None = None
    bottleneck_reason: str | None = None
    scope: str = "idealized_compute_only"
    trace_model: str = "ideal_full_visibility"
    valid_for_paper_main_results: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def records_to_jsonable(records: tuple[RuntimeRequestRecord, ...]) -> tuple[dict[str, Any], ...]:
    return tuple(record.to_dict() for record in records)


__all__ = ["RuntimeRequestRecord", "records_to_jsonable"]
