"""Mutable state for OrbitServe runtime event simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...common import to_jsonable


@dataclass
class ComputeQueueState:
    prefill_available_s: float = 0.0
    decode_available_s: float = 0.0

    def preview_prefill(self, arrival_s: float, service_s: float) -> tuple[float, float, float]:
        start_s = max(arrival_s, self.prefill_available_s)
        finish_s = start_s + service_s
        wait_s = max(0.0, start_s - arrival_s)
        return start_s, finish_s, wait_s

    def preview_decode(self, ready_s: float, service_s: float) -> tuple[float, float, float]:
        start_s = max(ready_s, self.decode_available_s)
        finish_s = start_s + service_s
        wait_s = max(0.0, start_s - ready_s)
        return start_s, finish_s, wait_s

    def reserve_prefill(self, arrival_s: float, service_s: float) -> tuple[float, float, float]:
        start_s, finish_s, wait_s = self.preview_prefill(arrival_s, service_s)
        self.prefill_available_s = finish_s
        return start_s, finish_s, wait_s

    def reserve_decode(self, ready_s: float, service_s: float) -> tuple[float, float, float]:
        start_s, finish_s, wait_s = self.preview_decode(ready_s, service_s)
        self.decode_available_s = finish_s
        return start_s, finish_s, wait_s

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class LinkState:
    link_id: str
    available_s: float = 0.0

    def preview(
        self,
        start_s: float,
        payload_bytes: float,
        *,
        bandwidth_bps: float,
        propagation_delay_s: float,
        setup_delay_s: float = 0.0,
    ) -> dict[str, float]:
        if bandwidth_bps <= 0:
            transmission_s = float("inf")
        else:
            transmission_s = max(0.0, payload_bytes) * 8.0 / bandwidth_bps
        begin_s = max(start_s, self.available_s)
        finish_transmit_s = begin_s + max(0.0, setup_delay_s) + transmission_s
        return {
            "queue_wait_s": max(0.0, begin_s - start_s),
            "transmission_s": transmission_s,
            "setup_delay_s": max(0.0, setup_delay_s),
            "propagation_delay_s": propagation_delay_s,
            "arrival_s": finish_transmit_s + propagation_delay_s,
            "finish_transmit_s": finish_transmit_s,
        }

    def reserve(
        self,
        start_s: float,
        payload_bytes: float,
        *,
        bandwidth_bps: float,
        propagation_delay_s: float,
        setup_delay_s: float = 0.0,
    ) -> dict[str, float]:
        result = self.preview(
            start_s,
            payload_bytes,
            bandwidth_bps=bandwidth_bps,
            propagation_delay_s=propagation_delay_s,
            setup_delay_s=setup_delay_s,
        )
        self.available_s = result["finish_transmit_s"]
        return result

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class SatelliteState:
    satellite_id: str
    compute_queue: ComputeQueueState = field(default_factory=ComputeQueueState)
    soc: float = 0.8
    initial_soc: float = 0.8
    last_energy_update_s: float = 0.0
    sunlit: bool = True
    dvfs_multiplier: float = 1.0
    energy_events: tuple[dict[str, Any], ...] = ()
    energy_ledger: tuple[dict[str, Any], ...] = ()
    materialized_energy_event_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "satellite_id": self.satellite_id,
            "compute_queue": self.compute_queue.to_dict(),
            "soc": self.soc,
            "initial_soc": self.initial_soc,
            "last_energy_update_s": self.last_energy_update_s,
            "sunlit": self.sunlit,
            "dvfs_multiplier": self.dvfs_multiplier,
            "energy_events": to_jsonable(self.energy_events),
            "energy_ledger": to_jsonable(self.energy_ledger),
            "materialized_energy_event_ids": self.materialized_energy_event_ids,
        }


@dataclass
class SessionState:
    session_id: str
    owner_satellite_id: str | None = None
    owner_deployment_id: str | None = None
    owner_role: str | None = None
    owner_satellite_ids: tuple[str, ...] = ()
    paired_prefill_deployment_id: str | None = None
    context_tokens: int = 0
    kv_cache_tokens: int = 0
    kv_cache_bytes: float = 0.0
    kv_pages: int = 0
    context_version: int = 0
    last_seen_s: float = 0.0
    expires_at_s: float | None = None
    replica_deployment_ids: tuple[str, ...] = ()
    last_session_action: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class KVCacheCardState:
    card_id: str
    satellite_id: str | None = None
    shard_id: str | None = None
    capacity_bytes: float | None = None
    used_bytes_by_kind: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class KVCacheDeploymentState:
    deployment_id: str
    hbm_model: str = "deployment_token_capacity_surrogate"
    aggregate_capacity_bytes: float | None = None
    aggregate_capacity_tokens: int | None = None
    aggregate_used_bytes_by_kind: dict[str, float] = field(default_factory=dict)
    aggregate_used_tokens_by_kind: dict[str, int] = field(default_factory=dict)
    cards: dict[str, KVCacheCardState] = field(default_factory=dict)
    page_placement_policy: str = "session_level_page_accounting"
    page_size_tokens: int | None = None
    metadata_bytes_per_page: float = 0.0
    fragmentation_reserve_fraction: float = 0.0
    card_level_hbm_admission: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "hbm_model": self.hbm_model,
            "aggregate_capacity_bytes": self.aggregate_capacity_bytes,
            "aggregate_capacity_tokens": self.aggregate_capacity_tokens,
            "aggregate_used_bytes_by_kind": to_jsonable(self.aggregate_used_bytes_by_kind),
            "aggregate_used_tokens_by_kind": to_jsonable(self.aggregate_used_tokens_by_kind),
            "cards": {card_id: card.to_dict() for card_id, card in self.cards.items()},
            "page_placement_policy": self.page_placement_policy,
            "page_size_tokens": self.page_size_tokens,
            "metadata_bytes_per_page": self.metadata_bytes_per_page,
            "fragmentation_reserve_fraction": self.fragmentation_reserve_fraction,
            "card_level_hbm_admission": self.card_level_hbm_admission,
        }


@dataclass
class RuntimeState:
    compute_queues: dict[str, ComputeQueueState] = field(default_factory=dict)
    satellites: dict[str, SatelliteState] = field(default_factory=dict)
    links: dict[str, LinkState] = field(default_factory=dict)
    sessions: dict[str, SessionState] = field(default_factory=dict)
    kv_cache_deployments: dict[str, KVCacheDeploymentState] = field(default_factory=dict)
    request_records: list[Any] = field(default_factory=list)
    decision_traces: list[dict[str, Any]] = field(default_factory=list)
    module_status: dict[str, str] = field(
        default_factory=lambda: {
                "queue_model": "fifo_work_seconds",
                "link_model": "not_implemented",
                "energy_model": "not_implemented",
                "kv_state": "not_implemented",
            }
    )

    def queue_for(self, deployment_id: str) -> ComputeQueueState:
        if deployment_id not in self.compute_queues:
            self.compute_queues[deployment_id] = ComputeQueueState()
        return self.compute_queues[deployment_id]

    def satellite_for(self, satellite_id: str, *, initial_soc: float = 0.8) -> SatelliteState:
        if satellite_id not in self.satellites:
            self.satellites[satellite_id] = SatelliteState(
                satellite_id=satellite_id,
                soc=initial_soc,
                initial_soc=initial_soc,
            )
        return self.satellites[satellite_id]

    def link_for(self, link_id: str) -> LinkState:
        if link_id not in self.links:
            self.links[link_id] = LinkState(link_id=link_id)
        return self.links[link_id]

    def session_for(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id)
        return self.sessions[session_id]

    def kv_cache_for(self, deployment_id: str) -> KVCacheDeploymentState:
        if deployment_id not in self.kv_cache_deployments:
            self.kv_cache_deployments[deployment_id] = KVCacheDeploymentState(deployment_id=deployment_id)
        return self.kv_cache_deployments[deployment_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_queues": {
                deployment_id: queue.to_dict()
                for deployment_id, queue in self.compute_queues.items()
            },
            "satellites": {
                satellite_id: satellite.to_dict()
                for satellite_id, satellite in self.satellites.items()
            },
            "links": {
                link_id: link.to_dict()
                for link_id, link in self.links.items()
            },
            "sessions": {
                session_id: session.to_dict()
                for session_id, session in self.sessions.items()
            },
            "kv_cache_deployments": {
                deployment_id: deployment.to_dict()
                for deployment_id, deployment in self.kv_cache_deployments.items()
            },
            "request_count": len(self.request_records),
            "decision_trace_count": len(self.decision_traces),
            "module_status": to_jsonable(self.module_status),
        }


__all__ = [
    "ComputeQueueState",
    "KVCacheCardState",
    "KVCacheDeploymentState",
    "LinkState",
    "SatelliteState",
    "SessionState",
    "RuntimeState",
]
