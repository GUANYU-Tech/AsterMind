"""Prefill/decode session and KV-cache runtime mechanics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...common import to_jsonable

if TYPE_CHECKING:
    from .runtime_state import RuntimeState


@dataclass
class SessionState:
    session_id: str
    satellite_ids: list[str] = field(default_factory=list)
    context_tokens: int = 0
    context_version: int = 0
    last_seen_s: float = 0.0
    last_session_action: str = "none"
    last_kv_event: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)


@dataclass
class KVCacheSatelliteState:
    satellite_id: str
    hbm_model: str = "satellite_token_capacity"
    capacity_bytes: float | None = None
    capacity_tokens: int | None = None
    used_bytes: float = 0.0
    used_tokens: int = 0
    page_placement_policy: str = "session_level_page_accounting"
    page_size_tokens: int | None = None
    metadata_bytes_per_page: float = 0.0
    fragmentation_reserve_fraction: float = 0.0
    entries: dict[str, "KVCacheEntryState"] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "satellite_id": self.satellite_id,
            "hbm_model": self.hbm_model,
            "capacity_bytes": self.capacity_bytes,
            "capacity_tokens": self.capacity_tokens,
            "used_bytes": self.used_bytes,
            "used_tokens": self.used_tokens,
            "page_placement_policy": self.page_placement_policy,
            "page_size_tokens": self.page_size_tokens,
            "metadata_bytes_per_page": self.metadata_bytes_per_page,
            "fragmentation_reserve_fraction": self.fragmentation_reserve_fraction,
            "entries": {key: entry.to_dict() for key, entry in self.entries.items()},
        }


@dataclass
class KVCacheEntryState:
    session_id: str
    satellite_id: str
    tokens: int
    bytes: float
    last_access_s: float
    reference_count: int = 0
    evictable: bool = True

    @property
    def in_use(self) -> bool:
        return self.reference_count > 0

    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

def _kv_page_size_tokens(kv_policy: dict[str, Any]) -> int | None:
    value = _optional_int(kv_policy.get("kv_page_size_tokens"))
    return value if value is not None and value > 0 else None

def _kv_required_pages(tokens: int | float, page_size_tokens: int | None) -> int:
    token_count = max(0, int(math.ceil(float(tokens or 0))))
    if page_size_tokens is None or page_size_tokens <= 0:
        return 1 if token_count > 0 else 0
    return int(math.ceil(token_count / page_size_tokens))


def _kv_accounted_bytes(
    tokens: int | float,
    *,
    kv_bytes_per_token: float,
    kv_policy: dict[str, Any],
) -> float:
    token_count = max(0, int(math.ceil(float(tokens or 0))))
    if token_count <= 0:
        return 0.0
    page_size_tokens = _kv_page_size_tokens(kv_policy)
    if page_size_tokens is None or page_size_tokens <= 0:
        return token_count * max(0.0, float(kv_bytes_per_token or 0.0))
    pages = _kv_required_pages(token_count, page_size_tokens)
    metadata_bytes = float(_optional_float(kv_policy.get("kv_page_metadata_bytes")) or 0.0)
    fragmentation = max(0.0, float(_optional_float(kv_policy.get("kv_fragmentation_reserve_fraction")) or 0.0))
    return (
        pages * page_size_tokens * max(0.0, float(kv_bytes_per_token or 0.0))
        + pages * metadata_bytes
    ) * (1.0 + fragmentation)

def _sync_kv_cache_state_for_satellite(
    state: RuntimeState,
    satellite_id: str,
    *,
    capacity_tokens: int | None,
    capacity_bytes: float | None,
    kv_policy: dict[str, Any],
) -> None:
    if not satellite_id:
        return
    cache = state.kv_cache_for(satellite_id)
    cache.capacity_bytes = capacity_bytes
    cache.hbm_model = "satellite_hbm_bytes" if capacity_bytes is not None else "satellite_token_capacity"
    cache.capacity_tokens = capacity_tokens
    cache.page_size_tokens = _kv_page_size_tokens(kv_policy)
    cache.metadata_bytes_per_page = float(_optional_float(kv_policy.get("kv_page_metadata_bytes")) or 0.0)
    cache.fragmentation_reserve_fraction = float(_optional_float(kv_policy.get("kv_fragmentation_reserve_fraction")) or 0.0)
    cache.used_tokens = sum(max(0, int(entry.tokens)) for entry in cache.entries.values())
    cache.used_bytes = sum(max(0.0, float(entry.bytes)) for entry in cache.entries.values())

def _admit_session_kv(
    state: RuntimeState,
    *,
    capacity_tokens: int | None,
    capacity_bytes: float | None,
    kv_policy: dict[str, Any],
    session_id: str,
    satellite_id: str,
    tokens: int,
    kv_bytes_per_token: float,
    now_s: float,
) -> dict[str, Any]:
    """Admit one session on one satellite, evicting only under capacity pressure."""

    _sync_kv_cache_state_for_satellite(
        state,
        satellite_id,
        capacity_tokens=capacity_tokens,
        capacity_bytes=capacity_bytes,
        kv_policy=kv_policy,
    )
    cache = state.kv_cache_for(satellite_id)
    accounted_bytes = _kv_accounted_bytes(
        tokens,
        kv_bytes_per_token=kv_bytes_per_token,
        kv_policy=kv_policy,
    )
    current = cache.entries.get(session_id)
    occupancy_tokens_before = sum(entry.tokens for entry in cache.entries.values())
    occupancy_bytes_before = sum(entry.bytes for entry in cache.entries.values())
    other_entries = [entry for key, entry in cache.entries.items() if key != session_id]
    used_tokens = sum(entry.tokens for entry in other_entries)
    used_bytes = sum(entry.bytes for entry in other_entries)
    required_tokens = max(0, int(tokens))

    def fits(extra_tokens: int, extra_bytes: float) -> bool:
        tokens_ok = cache.capacity_tokens is None or extra_tokens <= cache.capacity_tokens
        bytes_ok = cache.capacity_bytes is None or extra_bytes <= cache.capacity_bytes
        return tokens_ok and bytes_ok

    eviction_candidates = sorted(
        (entry for entry in other_entries if entry.evictable and not entry.in_use),
        key=lambda entry: (entry.last_access_s, entry.session_id),
    )
    evicted: list[KVCacheEntryState] = []
    while not fits(used_tokens + required_tokens, used_bytes + accounted_bytes) and eviction_candidates:
        victim = eviction_candidates.pop(0)
        evicted.append(victim)
        used_tokens -= victim.tokens
        used_bytes -= victim.bytes

    if not fits(used_tokens + required_tokens, used_bytes + accounted_bytes):
        return {
            "admitted": False,
            "admission": "capacity_unavailable",
            "evicted_session_ids": (),
            "entry": current.to_dict() if current is not None else None,
            "evicted_bytes": 0.0,
            "occupancy_tokens_before": occupancy_tokens_before,
            "occupancy_tokens_after": occupancy_tokens_before,
            "occupancy_bytes_before": occupancy_bytes_before,
            "occupancy_bytes_after": occupancy_bytes_before,
        }

    for victim in evicted:
        cache.entries.pop(victim.session_id, None)
        victim_session = state.sessions.get(victim.session_id)
        if victim_session is not None and satellite_id in victim_session.satellite_ids:
            victim_session.satellite_ids = sorted(
                item for item in victim_session.satellite_ids if item != satellite_id
            )
            victim_session.last_session_action = "evicted"

    entry = KVCacheEntryState(
        session_id=session_id,
        satellite_id=str(satellite_id),
        tokens=required_tokens,
        bytes=accounted_bytes,
        last_access_s=float(now_s),
        reference_count=current.reference_count if current is not None else 0,
        evictable=current.evictable if current is not None else True,
    )
    cache.entries[session_id] = entry
    _sync_kv_cache_state_for_satellite(
        state,
        satellite_id,
        capacity_tokens=capacity_tokens,
        capacity_bytes=capacity_bytes,
        kv_policy=kv_policy,
    )
    return {
        "admitted": True,
        "admission": "hit" if current is not None else "admitted",
        "evicted_session_ids": tuple(victim.session_id for victim in evicted),
        "entry": entry.to_dict(),
        "evicted_bytes": sum(victim.bytes for victim in evicted),
        "occupancy_tokens_before": occupancy_tokens_before,
        "occupancy_tokens_after": sum(item.tokens for item in cache.entries.values()),
        "occupancy_bytes_before": occupancy_bytes_before,
        "occupancy_bytes_after": sum(item.bytes for item in cache.entries.values()),
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
    return min(required_context_tokens, max(0, int(entry.tokens))) if entry is not None else 0


def commit_session_kv(
    state: RuntimeState,
    *,
    session_id: str | None,
    satellite_id: str,
    context_tokens: int,
    capacity_tokens: int | None,
    kv_bytes_per_token: float,
    now_s: float,
    copied_from: str | None = None,
) -> dict[str, Any]:
    if not session_id:
        return {"action": "none", "hit_tokens": 0, "miss_tokens": 0}
    session = state.session_for(str(session_id))
    local_hit = local_kv_reuse_tokens(
        state,
        satellite_id=str(satellite_id),
        session_id=str(session_id),
        required_context_tokens=context_tokens,
    )
    admission = _admit_session_kv(
        state,
        capacity_tokens=capacity_tokens,
        capacity_bytes=None,
        kv_policy={},
        session_id=str(session_id),
        satellite_id=str(satellite_id),
        tokens=context_tokens,
        kv_bytes_per_token=kv_bytes_per_token,
        now_s=now_s,
    )
    if not admission["admitted"]:
        session.last_session_action = "capacity_unavailable"
        session.last_kv_event = dict(admission)
        session.last_seen_s = now_s
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
    session.satellite_ids = sorted({*session.satellite_ids, str(satellite_id)})
    session.context_tokens = context_tokens
    session.context_version += 1
    session.last_seen_s = now_s
    session.last_session_action = action
    session.last_kv_event = {
        **dict(admission),
        "action": action,
        "hit_tokens": local_hit,
        "miss_tokens": max(0, context_tokens - local_hit),
        "copied_from": copied_from,
    }
    return dict(session.last_kv_event)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


__all__ = [
    "KVCacheEntryState",
    "KVCacheSatelliteState",
    "SessionState",
    "commit_session_kv",
    "local_kv_reuse_tokens",
]
