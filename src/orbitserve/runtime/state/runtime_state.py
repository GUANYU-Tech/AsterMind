"""Aggregate mutable state for one Runtime simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...common import to_jsonable
from .resources import LinkState, SatelliteState

if TYPE_CHECKING:
    from .kv_cache import KVCacheSatelliteState, SessionState


@dataclass
class RuntimeState:
    satellites: dict[str, SatelliteState] = field(default_factory=dict)
    links: dict[str, LinkState] = field(default_factory=dict)
    sessions: dict[str, SessionState] = field(default_factory=dict)
    kv_caches: dict[str, KVCacheSatelliteState] = field(default_factory=dict)

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
        from .kv_cache import SessionState
        if session_id not in self.sessions:
            self.sessions[session_id] = SessionState(session_id=session_id)
        return self.sessions[session_id]

    def kv_cache_for(self, satellite_id: str) -> KVCacheSatelliteState:
        from .kv_cache import KVCacheSatelliteState
        if satellite_id not in self.kv_caches:
            self.kv_caches[satellite_id] = KVCacheSatelliteState(satellite_id=satellite_id)
        return self.kv_caches[satellite_id]

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "kv_caches": {
                satellite_id: cache.to_dict()
                for satellite_id, cache in self.kv_caches.items()
            },
        }


__all__ = ["RuntimeState"]
