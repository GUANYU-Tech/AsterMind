"""Session and token-level KV-cache runtime mechanics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...common import to_jsonable

if TYPE_CHECKING:
    from .runtime_state import RuntimeState


@dataclass
class SessionState:
    session_id: str
    context_tokens: int = 0
    last_seen_s: float = 0.0
    last_session_action: str = "none"
    last_kv_event: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class KVCacheState:
    satellite_id: str
    capacity_tokens: int
    used_tokens: int = 0
    entries: dict[str, "KVCacheEntryState"] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "satellite_id": self.satellite_id,
            "capacity_tokens": self.capacity_tokens,
            "used_tokens": self.used_tokens,
            "entries": {key: entry.to_dict() for key, entry in self.entries.items()},
        }


@dataclass
class KVCacheEntryState:
    session_id: str
    cached_tokens: int
    last_access_s: float

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


def kv_locations_for_session(
    state: RuntimeState,
    session_id: str | None,
) -> tuple[str, ...]:
    if not session_id:
        return ()
    return tuple(
        satellite_id
        for satellite_id, cache in sorted(state.kv_caches.items())
        if str(session_id) in cache.entries
    )


def _admit_session_kv(
    state: RuntimeState,
    *,
    session_id: str,
    satellite_id: str,
    cached_tokens: int,
    now_s: float,
) -> dict[str, Any]:
    """Admit one whole session entry and evict other entries by LRU if needed."""

    cache = state.kv_cache_for(satellite_id)
    required_tokens = max(0, int(cached_tokens))
    current = cache.entries.get(session_id)
    occupancy_tokens_before = cache.used_tokens
    other_entries = [
        entry for key, entry in cache.entries.items() if key != session_id
    ]
    used_tokens = sum(entry.cached_tokens for entry in other_entries)
    eviction_candidates = sorted(
        other_entries,
        key=lambda entry: (entry.last_access_s, entry.session_id),
    )
    evicted: list[KVCacheEntryState] = []
    while used_tokens + required_tokens > cache.capacity_tokens and eviction_candidates:
        victim = eviction_candidates.pop(0)
        evicted.append(victim)
        used_tokens -= victim.cached_tokens

    if used_tokens + required_tokens > cache.capacity_tokens:
        return {
            "admitted": False,
            "admission": "capacity_unavailable",
            "evicted_session_ids": (),
            "evicted_tokens": 0,
            "entry": current.to_dict() if current is not None else None,
            "capacity_tokens": cache.capacity_tokens,
            "occupancy_tokens_before": occupancy_tokens_before,
            "occupancy_tokens_after": occupancy_tokens_before,
        }

    for victim in evicted:
        cache.entries.pop(victim.session_id, None)
    entry = KVCacheEntryState(
        session_id=session_id,
        cached_tokens=required_tokens,
        last_access_s=float(now_s),
    )
    cache.entries[session_id] = entry
    cache.used_tokens = sum(item.cached_tokens for item in cache.entries.values())
    return {
        "admitted": True,
        "admission": "hit" if current is not None else "admitted",
        "evicted_session_ids": tuple(victim.session_id for victim in evicted),
        "evicted_tokens": sum(victim.cached_tokens for victim in evicted),
        "entry": entry.to_dict(),
        "capacity_tokens": cache.capacity_tokens,
        "occupancy_tokens_before": occupancy_tokens_before,
        "occupancy_tokens_after": cache.used_tokens,
    }


def local_kv_reuse_tokens(
    state: RuntimeState,
    *,
    satellite_id: str,
    session_id: str | None,
    required_context_tokens: int,
) -> int:
    if not session_id or required_context_tokens <= 0:
        return 0
    cache = state.kv_caches.get(str(satellite_id))
    entry = cache.entries.get(str(session_id)) if cache is not None else None
    return (
        min(required_context_tokens, max(0, int(entry.cached_tokens)))
        if entry is not None
        else 0
    )


def touch_kv_entry(
    state: RuntimeState,
    *,
    satellite_id: str | None,
    session_id: str | None,
    now_s: float,
) -> None:
    if not satellite_id or not session_id:
        return
    cache = state.kv_caches.get(str(satellite_id))
    entry = cache.entries.get(str(session_id)) if cache is not None else None
    if entry is not None:
        entry.last_access_s = float(now_s)


def commit_session_kv(
    state: RuntimeState,
    *,
    session_id: str | None,
    satellite_id: str,
    context_tokens: int,
    now_s: float,
    copied_from: str | None = None,
) -> dict[str, Any]:
    if not session_id:
        return {"action": "none", "hit_tokens": 0, "miss_tokens": 0}
    session = state.session_for(str(session_id))
    session.context_tokens = context_tokens
    session.last_seen_s = now_s
    local_hit = local_kv_reuse_tokens(
        state,
        satellite_id=str(satellite_id),
        session_id=str(session_id),
        required_context_tokens=context_tokens,
    )
    admission = _admit_session_kv(
        state,
        session_id=str(session_id),
        satellite_id=str(satellite_id),
        cached_tokens=context_tokens,
        now_s=now_s,
    )
    if not admission["admitted"]:
        session.last_session_action = "capacity_unavailable"
        session.last_kv_event = dict(admission)
        return {
            "action": "capacity_unavailable",
            "hit_tokens": local_hit,
            "miss_tokens": max(0, context_tokens - local_hit),
            **admission,
        }
    if copied_from and copied_from != satellite_id:
        action = "copied"
    elif local_hit > 0:
        action = "local_hit"
    else:
        action = "recompute_context"
    session.last_session_action = action
    session.last_kv_event = {
        **dict(admission),
        "action": action,
        "hit_tokens": local_hit,
        "miss_tokens": max(0, context_tokens - local_hit),
        "copied_from": copied_from,
    }
    return dict(session.last_kv_event)


__all__ = [
    "KVCacheEntryState",
    "KVCacheState",
    "SessionState",
    "commit_session_kv",
    "kv_locations_for_session",
    "local_kv_reuse_tokens",
    "touch_kv_entry",
]
