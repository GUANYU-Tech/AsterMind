"""Aggregate mutable state for one Runtime simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .resources import ComputeConcurrencyState, ComputeNodeState

if TYPE_CHECKING:
    from .kv_cache import KVCacheState, SessionState


@dataclass
class RuntimeState:
    compute_nodes: dict[str, ComputeNodeState] = field(default_factory=dict)
    sessions: dict[str, SessionState] = field(default_factory=dict)
    kv_caches: dict[str, KVCacheState] = field(default_factory=dict)

    def compute_node_for(
        self,
        compute_node_id: str,
        *,
        initial_soc: float = 0.8,
        max_concurrency: int | None = None,
    ) -> ComputeNodeState:
        if compute_node_id not in self.compute_nodes:
            self.compute_nodes[compute_node_id] = ComputeNodeState(
                compute_node_id=compute_node_id,
                compute_queue=ComputeConcurrencyState(max_concurrency or 1),
                soc=initial_soc,
                initial_soc=initial_soc,
            )
        elif max_concurrency is not None:
            queue = self.compute_nodes[compute_node_id].compute_queue
            if queue.max_concurrency != max_concurrency:
                if queue.reservations:
                    raise ValueError("cannot change max_concurrency after reservations exist")
                queue.max_concurrency = max_concurrency
        return self.compute_nodes[compute_node_id]

    def satellite_for(self, satellite_id: str, *, initial_soc: float = 0.8) -> ComputeNodeState:
        """Compatibility adapter for geometry code that still uses satellite IDs."""

        return self.compute_node_for(satellite_id, initial_soc=initial_soc)

    @property
    def satellites(self) -> dict[str, ComputeNodeState]:
        return self.compute_nodes

    def session_for(self, session_id: str) -> SessionState:
        from .kv_cache import SessionState
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id)
        return self.sessions[session_id]

    def kv_cache_for(
        self,
        satellite_id: str,
        *,
        capacity_tokens: int | None = None,
    ) -> KVCacheState:
        from .kv_cache import KVCacheState
        if satellite_id not in self.kv_caches:
            self.kv_caches[satellite_id] = KVCacheState(
                satellite_id=satellite_id,
                capacity_tokens=max(0, int(capacity_tokens or 0)),
            )
        elif capacity_tokens is not None:
            self.kv_caches[satellite_id].capacity_tokens = max(0, int(capacity_tokens))
        return self.kv_caches[satellite_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_nodes": {
                compute_node_id: compute_node.to_dict()
                for compute_node_id, compute_node in self.compute_nodes.items()
            },
            "sessions": {
                session_id: session.to_dict()
                for session_id, session in self.sessions.items()
            },
            "kv_caches": {
                compute_node_id: cache.to_dict()
                for compute_node_id, cache in self.kv_caches.items()
            },
        }


__all__ = ["RuntimeState"]
