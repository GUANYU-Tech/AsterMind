"""Aggregate mutable state for one Runtime simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...common import to_jsonable
from .resources import SatelliteState

if TYPE_CHECKING:
    from .kv_cache import KVCacheState, SessionState


@dataclass
class RuntimeState:
    satellites: dict[str, SatelliteState] = field(default_factory=dict)
    sessions: dict[str, SessionState] = field(default_factory=dict)
    kv_caches: dict[str, KVCacheState] = field(default_factory=dict)

    def satellite_for(self, satellite_id: str, *, initial_soc: float = 0.8) -> SatelliteState:
        if satellite_id not in self.satellites:
            self.satellites[satellite_id] = SatelliteState(
                satellite_id=satellite_id,
                soc=initial_soc,
                initial_soc=initial_soc,
            )
        return self.satellites[satellite_id]

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
            "satellites": {
                satellite_id: satellite.to_dict()
                for satellite_id, satellite in self.satellites.items()
            },
            "sessions": {
                session_id: session.to_dict()
                for session_id, session in self.sessions.items()
            },
            "kv_caches": {
                satellite_id: cache.to_dict()
                for satellite_id, cache in self.kv_caches.items()
            },
        }


__all__ = ["RuntimeState"]
