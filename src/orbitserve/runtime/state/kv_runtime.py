"""Prefill/decode session and KV-cache runtime mechanics."""

from __future__ import annotations

import math
from typing import Any

from .runtime_state import KVCacheEntryState, RuntimeState


def _kv_policy(runtime_strategy: Any) -> dict[str, Any]:
    if runtime_strategy is None:
        return {}
    policy = _dict(getattr(runtime_strategy, "kv_policy", {}))
    overrides = _dict(getattr(runtime_strategy, "hard_gate_overrides", {}))
    override_policy = _dict(overrides.get("kv_policy"))
    merged = {**policy, **override_policy}
    for key in (
        "kv_serve_actions_enabled",
        "kv_post_turn_actions_enabled",
        "replica_count_limit",
        "replica_target_deployment_id",
        "kv_page_size_tokens",
        "kv_page_metadata_bytes",
        "kv_fragmentation_reserve_fraction",
        "kv_transfer_chunk_bytes",
        "max_kv_transfer_hop_count",
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


def _deployment_hbm_capacity_bytes(deployment: Any, kv_policy: dict[str, Any]) -> float | None:
    capacity_gb = _optional_float(kv_policy.get("hbm_capacity_gb_per_card"))
    cards_per_sat = _optional_int(kv_policy.get("cards_per_satellite"))
    if capacity_gb is None or cards_per_sat is None or cards_per_sat <= 0:
        return None
    usable = _optional_float(kv_policy.get("hbm_usable_fraction"))
    usable = usable if usable is not None else 1.0
    model_gb = _optional_float(kv_policy.get("model_weight_hbm_gb_per_card")) or 0.0
    activation_gb = _optional_float(kv_policy.get("activation_reserve_gb_per_card")) or 0.0
    workspace_gb = _optional_float(kv_policy.get("runtime_workspace_gb_per_card")) or 0.0
    member_count = len(tuple(getattr(deployment, "member_satellites", ()) or ())) or 1
    per_card_available_gb = max(0.0, capacity_gb * usable - model_gb - activation_gb - workspace_gb)
    return per_card_available_gb * cards_per_sat * member_count * (1024.0 ** 3)

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

def _card_level_hbm_enabled(deployment: Any, kv_policy: dict[str, Any]) -> bool:
    del deployment
    del kv_policy
    return False

def _sync_kv_cache_state_for_deployment(state: RuntimeState, deployment: Any, runtime_strategy: Any) -> None:
    deployment_id = str(getattr(deployment, "deployment_id", ""))
    if not deployment_id:
        return
    kv_policy = _kv_policy(runtime_strategy)
    cache = state.kv_cache_for(deployment_id)
    cache.hbm_model = "aggregate_hbm_bytes" if _deployment_hbm_capacity_bytes(deployment, kv_policy) is not None else "deployment_token_capacity_surrogate"
    cache.aggregate_capacity_bytes = _deployment_hbm_capacity_bytes(deployment, kv_policy)
    cache.aggregate_capacity_tokens = _optional_int(getattr(deployment, "kv_cache_capacity_tokens", None))
    cache.page_size_tokens = _kv_page_size_tokens(kv_policy)
    cache.metadata_bytes_per_page = float(_optional_float(kv_policy.get("kv_page_metadata_bytes")) or 0.0)
    cache.fragmentation_reserve_fraction = float(_optional_float(kv_policy.get("kv_fragmentation_reserve_fraction")) or 0.0)
    cache.card_level_hbm_admission = False
    primary_tokens = 0
    replica_tokens = 0
    primary_bytes = 0.0
    replica_bytes = 0.0
    for entry in cache.entries.values():
        primary_tokens += max(0, int(entry.tokens))
        primary_bytes += max(0.0, float(entry.bytes))
    cache.aggregate_used_tokens_by_kind = {"primary": primary_tokens, "replica": replica_tokens}
    cache.aggregate_used_bytes_by_kind = {"primary": primary_bytes, "replica": replica_bytes}

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
    """Admit or refresh one session entry, evicting by capacity-pressure LRU."""

    deployment_id = str(getattr(deployment, "deployment_id", ""))
    kv_policy = _kv_policy(runtime_strategy)
    _sync_kv_cache_state_for_deployment(state, deployment, runtime_strategy)
    cache = state.kv_cache_for(deployment_id)
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
        tokens_ok = cache.aggregate_capacity_tokens is None or extra_tokens <= cache.aggregate_capacity_tokens
        bytes_ok = cache.aggregate_capacity_bytes is None or extra_bytes <= cache.aggregate_capacity_bytes
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
        if victim_session is not None and victim_session.owner_deployment_id == deployment_id:
            victim_session.owner_satellite_id = None
            victim_session.owner_deployment_id = None
            victim_session.owner_satellite_ids = ()
            victim_session.kv_cache_tokens = 0
            victim_session.kv_cache_bytes = 0.0
            victim_session.last_session_action = "evicted"

    entry = KVCacheEntryState(
        session_id=session_id,
        satellite_id=str(satellite_id),
        deployment_id=deployment_id,
        tokens=required_tokens,
        bytes=accounted_bytes,
        last_access_s=float(now_s),
        reference_count=current.reference_count if current is not None else 0,
        evictable=current.evictable if current is not None else True,
    )
    cache.entries[session_id] = entry
    _sync_kv_cache_state_for_deployment(state, deployment, runtime_strategy)
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
