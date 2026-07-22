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
    owner_satellite_id: str | None = None
    owner_deployment_id: str | None = None
    owner_satellite_ids: tuple[str, ...] = ()
    context_tokens: int = 0
    kv_cache_tokens: int = 0
    kv_cache_bytes: float = 0.0
    kv_pages: int = 0
    context_version: int = 0
    last_seen_s: float = 0.0
    replica_deployment_ids: tuple[str, ...] = ()
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

def _kv_policy(runtime_strategy: Any) -> dict[str, Any]:
    if runtime_strategy is None:
        return {}
    policy = _dict(getattr(runtime_strategy, "kv_policy", {}))
    overrides = _dict(getattr(runtime_strategy, "hard_gate_overrides", {}))
    override_policy = _dict(overrides.get("kv_policy"))
    merged = {**policy, **override_policy}
    for key in (
        "kv_serve_actions_enabled",
        "kv_page_size_tokens",
        "kv_page_metadata_bytes",
        "kv_fragmentation_reserve_fraction",
        "eviction_policy",
        "hbm_model",
        "cards_per_satellite",
        "hbm_capacity_gb_per_card",
        "hbm_usable_fraction",
        "model_weight_hbm_gb_per_card",
        "activation_reserve_gb_per_card",
        "runtime_workspace_gb_per_card",
    ):
        if key in overrides and key not in merged:
            merged[key] = overrides[key]
    return merged


def _kv_page_size_tokens(kv_policy: dict[str, Any]) -> int | None:
    value = _optional_int(kv_policy.get("kv_page_size_tokens"))
    return value if value is not None and value > 0 else None

def _kv_required_pages(tokens: int | float, page_size_tokens: int | None) -> int:
    token_count = max(0, int(math.ceil(float(tokens or 0))))
    if page_size_tokens is None or page_size_tokens <= 0:
        return 1 if token_count > 0 else 0
    return int(math.ceil(token_count / page_size_tokens))


def _satellite_hbm_capacity_bytes(kv_policy: dict[str, Any]) -> float | None:
    capacity_gb = _optional_float(kv_policy.get("hbm_capacity_gb_per_card"))
    cards_per_sat = _optional_int(kv_policy.get("cards_per_satellite"))
    if capacity_gb is None or cards_per_sat is None or cards_per_sat <= 0:
        return None
    usable = _optional_float(kv_policy.get("hbm_usable_fraction"))
    usable = usable if usable is not None else 1.0
    model_gb = _optional_float(kv_policy.get("model_weight_hbm_gb_per_card")) or 0.0
    activation_gb = _optional_float(kv_policy.get("activation_reserve_gb_per_card")) or 0.0
    workspace_gb = _optional_float(kv_policy.get("runtime_workspace_gb_per_card")) or 0.0
    per_card_available_gb = max(0.0, capacity_gb * usable - model_gb - activation_gb - workspace_gb)
    return per_card_available_gb * cards_per_sat * (1024.0 ** 3)

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
    deployment: Any,
    runtime_strategy: Any,
) -> None:
    if not satellite_id:
        return
    kv_policy = _kv_policy(runtime_strategy)
    cache = state.kv_cache_for(satellite_id)
    cache.capacity_bytes = _satellite_hbm_capacity_bytes(kv_policy)
    cache.hbm_model = "satellite_hbm_bytes" if cache.capacity_bytes is not None else "satellite_token_capacity"
    cache.capacity_tokens = _optional_int(getattr(deployment, "kv_cache_capacity_tokens", None))
    cache.page_size_tokens = _kv_page_size_tokens(kv_policy)
    cache.metadata_bytes_per_page = float(_optional_float(kv_policy.get("kv_page_metadata_bytes")) or 0.0)
    cache.fragmentation_reserve_fraction = float(_optional_float(kv_policy.get("kv_fragmentation_reserve_fraction")) or 0.0)
    cache.used_tokens = sum(max(0, int(entry.tokens)) for entry in cache.entries.values())
    cache.used_bytes = sum(max(0.0, float(entry.bytes)) for entry in cache.entries.values())

def _admit_session_kv(
    state: RuntimeState,
    deployment: Any,
    runtime_strategy: Any,
    *,
    session_id: str,
    satellite_id: str,
    tokens: int,
    kv_bytes_per_token: float,
    now_s: float,
) -> dict[str, Any]:
    """Admit one session on one satellite, evicting only under capacity pressure."""

    kv_policy = _kv_policy(runtime_strategy)
    _sync_kv_cache_state_for_satellite(state, satellite_id, deployment, runtime_strategy)
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
        if victim_session is not None and victim_session.owner_satellite_id == satellite_id:
            victim_session.owner_satellite_id = None
            victim_session.owner_deployment_id = None
            victim_session.owner_satellite_ids = ()
            victim_session.kv_cache_tokens = 0
            victim_session.kv_cache_bytes = 0.0
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
    _sync_kv_cache_state_for_satellite(state, satellite_id, deployment, runtime_strategy)
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


def _remove_session_kv(state: RuntimeState, satellite_id: str, session_id: str) -> bool:
    """Remove a stale session entry when execution moves to another satellite."""

    cache = state.kv_caches.get(satellite_id)
    if cache is None or cache.entries.pop(session_id, None) is None:
        return False
    cache.used_tokens = sum(max(0, int(entry.tokens)) for entry in cache.entries.values())
    cache.used_bytes = sum(max(0.0, float(entry.bytes)) for entry in cache.entries.values())
    return True


def _token_context_contract_error(request: dict[str, Any]) -> str | None:
    new_user_tokens = request.get("new_user_tokens")
    context_tokens_before_turn = request.get("context_tokens_before_turn")
    if (
        request.get("prompt_tokens_declared")
        and new_user_tokens is not None
        and context_tokens_before_turn is not None
        and int(request["prompt_tokens"]) != int(new_user_tokens) + int(context_tokens_before_turn)
    ):
        return "token_context_contract_ambiguous"
    return None


def _deployment_entry_satellite(deployment: Any) -> str:
    entries = tuple(getattr(deployment, "entry_satellite_ids", ()) or ())
    if entries:
        return str(entries[0])
    members = tuple(getattr(deployment, "member_satellites", ()) or ())
    return str(members[0]) if members else str(getattr(deployment, "deployment_id", ""))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


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


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (tuple, list, set)):
        return tuple(str(item) for item in value if item is not None)
    return (str(value),)
