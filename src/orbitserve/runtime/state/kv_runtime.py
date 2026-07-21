"""Prefill/decode session and KV-cache runtime mechanics."""

from __future__ import annotations

import math
from typing import Any

from ..scenario.deployment_contracts import RuntimeDeploymentCatalog
from .runtime_state import RuntimeState

def _pd_prefill_context(state: RuntimeState, request: dict[str, Any]) -> dict[str, Any]:
    session_id = request.get("session_id")
    session = state.sessions.get(str(session_id)) if session_id else None
    new_user_tokens = request.get("new_user_tokens")
    if request.get("prompt_tokens_declared"):
        prefill_tokens_effective = int(request["prompt_tokens"])
        context_tokens_before_turn = int(request["prompt_tokens"])
        token_context_contract = "full_prompt"
    elif new_user_tokens is not None:
        context_tokens_before_turn = _first_present(
            request.get("context_tokens_before_turn"),
            getattr(session, "context_tokens", None),
            0,
        )
        context_tokens_before_turn = int(context_tokens_before_turn or 0)
        prefill_tokens_effective = context_tokens_before_turn + int(new_user_tokens or 0)
        token_context_contract = "incremental_turn_recompute_context"
    else:
        prefill_tokens_effective = int(request["prompt_tokens"])
        context_tokens_before_turn = int(request["prompt_tokens"])
        token_context_contract = "legacy_prompt_tokens"
    return {
        "prefill_tokens_effective": max(0, prefill_tokens_effective),
        "context_tokens_before_turn": max(0, context_tokens_before_turn),
        "new_user_tokens": new_user_tokens,
        "token_context_contract": token_context_contract,
    }

def _pd_prefill_context_for_action(
    state: RuntimeState,
    request: dict[str, Any],
    decode_deployment: Any,
    runtime_strategy: Any,
    *,
    kv_serve_action: str,
) -> dict[str, Any]:
    session_id = request.get("session_id")
    session = state.sessions.get(str(session_id)) if session_id else None
    base = _pd_prefill_context(state, request)
    kv_bytes_per_token = float(getattr(decode_deployment, "kv_cache_bytes_per_token", None) or 0.0)
    context_tokens_before_turn = int(base.get("context_tokens_before_turn") or 0)
    new_user_tokens = base.get("new_user_tokens")
    if request.get("prompt_tokens_declared") and new_user_tokens is None:
        context_tokens_before_turn = 0
        historical_recompute_tokens = 0
        new_input_tokens_this_turn = int(request.get("prompt_tokens") or 0)
        token_contract = "full_prompt_no_delta"
    elif new_user_tokens is not None:
        new_input_tokens_this_turn = int(new_user_tokens or 0)
        historical_recompute_tokens = context_tokens_before_turn
        token_contract = "incremental_turn_recompute_context"
    else:
        historical_recompute_tokens = 0
        new_input_tokens_this_turn = int(request.get("prompt_tokens") or 0)
        token_contract = "legacy_prompt_tokens_no_delta"

    reusable_tokens = 0
    kv_source_satellite_id = None
    kv_source_deployment_id = None
    if session is not None and int(getattr(session, "kv_cache_tokens", 0) or 0) > 0:
        reusable_tokens = min(context_tokens_before_turn, int(session.kv_cache_tokens))
        kv_source_satellite_id = getattr(session, "owner_satellite_id", None)
        kv_source_deployment_id = getattr(session, "owner_deployment_id", None)

    can_hit_here = _pd_session_kv_available_on_decode(session, decode_deployment)
    if kv_serve_action == "sticky_kv_owner" and can_hit_here:
        historical_recompute_tokens = max(0, context_tokens_before_turn - reusable_tokens)
        token_contract = "incremental_turn_sticky_kv_reuse"
    elif kv_serve_action == "migrate_kv" and reusable_tokens > 0:
        historical_recompute_tokens = max(0, context_tokens_before_turn - reusable_tokens)
        token_contract = "incremental_turn_migrate_kv_reuse"
    elif kv_serve_action == "recompute_context":
        reusable_tokens = 0
        kv_source_satellite_id = None
        kv_source_deployment_id = None
        if new_user_tokens is not None:
            token_contract = "incremental_turn_recompute_context"

    current_prefill_tokens = max(0, historical_recompute_tokens + new_input_tokens_this_turn)
    kv_hit_tokens = max(0, context_tokens_before_turn - historical_recompute_tokens) if kv_serve_action in {"sticky_kv_owner", "migrate_kv"} else 0
    kv_hit_tokens = min(kv_hit_tokens, reusable_tokens)
    kv_miss_tokens = current_prefill_tokens
    kv_generated_bytes = kv_miss_tokens * kv_bytes_per_token
    kv_migration_bytes = kv_hit_tokens * kv_bytes_per_token if kv_serve_action == "migrate_kv" else 0.0
    page_size_tokens = _kv_page_size_tokens(_kv_policy(runtime_strategy))
    pages_required = _kv_required_pages(context_tokens_before_turn + new_input_tokens_this_turn, page_size_tokens)
    pages_hit = _kv_required_pages(kv_hit_tokens, page_size_tokens)
    return {
        "prefill_tokens_effective": current_prefill_tokens,
        "context_tokens_before_turn": max(0, context_tokens_before_turn),
        "new_user_tokens": new_user_tokens,
        "new_input_tokens_this_turn": max(0, new_input_tokens_this_turn),
        "historical_recompute_tokens": max(0, historical_recompute_tokens),
        "current_prefill_tokens": current_prefill_tokens,
        "reusable_context_tokens_before_turn": reusable_tokens,
        "kv_hit_tokens": kv_hit_tokens,
        "kv_miss_tokens": kv_miss_tokens,
        "kv_reused_bytes": kv_hit_tokens * kv_bytes_per_token,
        "kv_generated_bytes": kv_generated_bytes,
        "kv_migration_bytes": kv_migration_bytes,
        "kv_source_satellite_id": kv_source_satellite_id,
        "kv_source_deployment_id": kv_source_deployment_id,
        "kv_pages_required": pages_required,
        "kv_pages_hit": pages_hit,
        "kv_pages_migrated": pages_hit if kv_serve_action == "migrate_kv" else 0,
        "kv_pages_replicated": 0,
        "token_context_contract": token_contract,
    }

def _pd_context_tokens_after(
    request: dict[str, Any],
    *,
    prefill_context: dict[str, Any],
) -> int:
    if "new_input_tokens_this_turn" in prefill_context:
        context_tokens = (
            int(prefill_context.get("context_tokens_before_turn") or 0)
            + int(prefill_context.get("new_input_tokens_this_turn") or 0)
            + int(request.get("output_tokens") or 0)
        )
    else:
        context_input_tokens = int(prefill_context.get("prefill_tokens_effective") or 0)
        context_tokens = context_input_tokens + int(request.get("output_tokens") or 0)
    context_window_tokens = request.get("context_window_tokens")
    if context_window_tokens is not None:
        context_tokens = min(context_tokens, max(0, int(context_window_tokens)))
    return max(0, context_tokens)

def _pd_context_recompute_tokens(
    prefill_context: dict[str, Any],
    prefill_tokens_effective: int,
    runtime_strategy: Any,
    prefill_action_type: str,
    decode_action_type: str,
    *,
    session_action: str,
    sticky_prefill_kv_hit: bool,
) -> int:
    if sticky_prefill_kv_hit:
        return 0
    token_contract = str(prefill_context.get("token_context_contract") or "")
    if session_action == "recompute_context" or token_contract == "incremental_turn_recompute_context":
        return max(0, int(prefill_tokens_effective or 0))
    return 0

def _pd_session_action_preview(state: RuntimeState, session_id: str | None, decode_deployment: Any) -> str:
    if not session_id:
        return "none"
    session = state.sessions.get(str(session_id))
    decode_entry = _deployment_entry_satellite(decode_deployment)
    if session is None:
        return "sticky_kv_owner"
    if (
        session.owner_deployment_id is None
        and session.owner_satellite_id is None
        and session.kv_cache_tokens <= 0
    ):
        return "sticky_kv_owner"
    if session.owner_deployment_id == getattr(decode_deployment, "deployment_id", None):
        return "sticky_kv_owner"
    if session.owner_satellite_id == decode_entry:
        return "sticky_kv_owner"
    return "recompute_context"

def _pd_kv_serve_actions_for_decode(
    state: RuntimeState,
    request: dict[str, Any],
    decode_deployment: Any,
    runtime_strategy: Any,
) -> tuple[str, ...]:
    explicit_enabled = bool(_kv_policy(runtime_strategy).get("kv_serve_actions_enabled") is not None)
    enabled = _string_tuple(_kv_policy(runtime_strategy).get("kv_serve_actions_enabled"))
    if not enabled:
        enabled = tuple(
            action
            for action in _string_tuple(getattr(runtime_strategy, "session_actions_enabled", ()))
            if action in {"sticky_kv_owner", "migrate_kv", "recompute_context"}
        )
    if not enabled:
        enabled = ("recompute_context",)
    session_id = request.get("session_id")
    session = state.sessions.get(str(session_id)) if session_id else None
    has_valid_kv = session is not None and int(getattr(session, "kv_cache_tokens", 0) or 0) > 0
    actions: list[str] = []
    legacy_allows_sticky = (
        not explicit_enabled
        and str(getattr(runtime_strategy, "policy", "")) != "session_recompute_only"
        and has_valid_kv
        and _pd_session_kv_available_on_decode(session, decode_deployment)
    )
    if legacy_allows_sticky and "sticky_kv_owner" not in enabled:
        enabled = ("sticky_kv_owner", *enabled)
    for action in enabled:
        if action == "sticky_kv_owner":
            if has_valid_kv and _pd_session_kv_available_on_decode(session, decode_deployment):
                actions.append(action)
        elif action == "migrate_kv":
            if has_valid_kv and not _pd_session_kv_available_on_decode(session, decode_deployment):
                actions.append(action)
        elif action == "recompute_context":
            actions.append(action)
    if not actions and (not explicit_enabled or not has_valid_kv):
        actions.append("recompute_context")
    return tuple(dict.fromkeys(actions))

def _pd_session_kv_available_on_decode(session: Any, decode_deployment: Any) -> bool:
    if session is None:
        return False
    decode_id = getattr(decode_deployment, "deployment_id", None)
    decode_entry = _deployment_entry_satellite(decode_deployment)
    replicas = tuple(str(item) for item in getattr(session, "replica_deployment_ids", ()) or ())
    return bool(
        getattr(session, "owner_deployment_id", None) == decode_id
        or getattr(session, "owner_satellite_id", None) == decode_entry
        or (decode_id is not None and str(decode_id) in replicas)
    )

def _pd_kv_post_turn_candidates(
    state: RuntimeState,
    request: dict[str, Any],
    decode_deployment: Any,
    deployment_catalog: RuntimeDeploymentCatalog,
    runtime_strategy: Any,
) -> tuple[dict[str, Any], ...]:
    del state, request
    policy = _kv_policy(runtime_strategy)
    explicit_enabled = bool(policy.get("kv_post_turn_actions_enabled") is not None)
    enabled = _string_tuple(policy.get("kv_post_turn_actions_enabled"))
    if not enabled:
        enabled = tuple(
            action
            for action in _string_tuple(getattr(runtime_strategy, "session_actions_enabled", ()))
            if action in {"replicate_kv", "evict_session"}
        )
    candidates: list[dict[str, Any]] = []
    if not explicit_enabled or "none" in enabled:
        candidates.append({"action": "none", "target_deployment": None})
    replica_limit = _optional_int(policy.get("replica_count_limit"))
    if replica_limit is None:
        replica_limit = _optional_int(_dict(getattr(runtime_strategy, "hard_gate_overrides", {})).get("replica_count_limit"))
    if "replicate_kv" in enabled and (replica_limit is None or replica_limit > 0):
        target = _pd_replica_target_deployment(decode_deployment, deployment_catalog, policy)
        if target is not None:
            candidates.append({"action": "replicate_kv", "target_deployment": target})
    if "evict_session" in enabled:
        candidates.append({"action": "evict_session", "target_deployment": None})
    return tuple(candidates) or ({"action": "none", "target_deployment": None},)

def _pd_replica_target_deployment(
    decode_deployment: Any,
    deployment_catalog: RuntimeDeploymentCatalog,
    kv_policy: dict[str, Any],
) -> Any | None:
    decode_id = getattr(decode_deployment, "deployment_id", None)
    target_id = _optional_str(kv_policy.get("replica_target_deployment_id"))
    if target_id is not None:
        for candidate in tuple(getattr(deployment_catalog, "decode_deployments", ()) or ()):
            if str(getattr(candidate, "deployment_id", None)) == target_id and target_id != str(decode_id):
                return candidate
        return None
    for candidate in tuple(getattr(deployment_catalog, "decode_deployments", ()) or ()):
        if getattr(candidate, "deployment_id", None) != decode_id:
            return candidate
    return None

def _pd_session_action_compat(kv_serve_action: str, kv_post_turn_action: str | None) -> str:
    del kv_post_turn_action
    return kv_serve_action

def _pd_migration_start_s(state: RuntimeState, request: dict[str, Any], arrival_s: float) -> float:
    session_id = request.get("session_id")
    session = state.sessions.get(str(session_id)) if session_id else None
    if session is None:
        return arrival_s
    return min(arrival_s, max(0.0, float(getattr(session, "last_seen_s", arrival_s) or 0.0)))

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
        "session_ttl_s",
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

def _pd_max_kv_hop_count(runtime_strategy: Any, default: int | None) -> int | None:
    value = _optional_int(_kv_policy(runtime_strategy).get("max_kv_transfer_hop_count"))
    return value if value is not None else default

def _kv_transfer_chunk_bytes(kv_policy: dict[str, Any]) -> float | None:
    value = _optional_float(kv_policy.get("kv_transfer_chunk_bytes"))
    return value if value is not None and value > 0 else None

def _kv_page_size_tokens(kv_policy: dict[str, Any]) -> int | None:
    value = _optional_int(kv_policy.get("kv_page_size_tokens"))
    return value if value is not None and value > 0 else None

def _kv_required_pages(tokens: int | float, page_size_tokens: int | None) -> int:
    token_count = max(0, int(math.ceil(float(tokens or 0))))
    if page_size_tokens is None or page_size_tokens <= 0:
        return 1 if token_count > 0 else 0
    return int(math.ceil(token_count / page_size_tokens))

def _request_kv_deadline_s(request: dict[str, Any], arrival_s: float) -> float | None:
    deadlines = []
    for key in ("ttft_deadline_s", "e2e_deadline_s"):
        value = _optional_float(request.get(key))
        if value is not None:
            deadlines.append(arrival_s + value)
    return min(deadlines) if deadlines else None

def _session_expires_at(state: RuntimeState, session_id: str | None) -> float | None:
    session = state.sessions.get(str(session_id)) if session_id else None
    return _optional_float(getattr(session, "expires_at_s", None)) if session is not None else None

def _session_ttl_deadline(runtime_strategy: Any, now_s: float) -> float | None:
    ttl_s = _session_ttl_s(runtime_strategy)
    return now_s + ttl_s if ttl_s is not None else None

def _pd_kv_capacity_result(
    state: RuntimeState,
    request: dict[str, Any],
    decode_deployment: Any,
    context_tokens_after: int,
    *,
    runtime_strategy: Any = None,
    kv_bytes_per_token: float | None = None,
    kv_post_turn_action: str | None = None,
    replica_target_deployment: Any | None = None,
) -> dict[str, Any]:
    kv_policy = _kv_policy(runtime_strategy)
    kv_bytes_per_token = float(kv_bytes_per_token or getattr(decode_deployment, "kv_cache_bytes_per_token", None) or 0.0)
    decode_capacity_bytes = _deployment_hbm_capacity_bytes(decode_deployment, kv_policy)
    required_pages = _kv_required_pages(context_tokens_after, _kv_page_size_tokens(kv_policy))
    required_bytes = _kv_accounted_bytes(
        context_tokens_after,
        kv_bytes_per_token=kv_bytes_per_token,
        kv_policy=kv_policy,
    )
    existing_bytes = _pd_existing_kv_bytes_for_decode(
        state,
        decode_deployment,
        exclude_session_id=request.get("session_id"),
    )
    replica_bytes = 0.0
    replica_existing_bytes = 0.0
    replica_capacity_bytes = None
    if kv_post_turn_action == "replicate_kv" and replica_target_deployment is not None:
        replica_bytes = _kv_accounted_bytes(
            context_tokens_after,
            kv_bytes_per_token=kv_bytes_per_token,
            kv_policy=kv_policy,
        )
        replica_existing_bytes = _pd_existing_kv_bytes_for_decode(
            state,
            replica_target_deployment,
            exclude_session_id=request.get("session_id"),
        )
        replica_capacity_bytes = _deployment_hbm_capacity_bytes(replica_target_deployment, kv_policy)
    if decode_capacity_bytes is not None or replica_capacity_bytes is not None:
        projected_decode_bytes = existing_bytes + required_bytes
        projected_replica_bytes = replica_existing_bytes + replica_bytes
        decode_passed = decode_capacity_bytes is None or projected_decode_bytes <= decode_capacity_bytes
        replica_passed = replica_capacity_bytes is None or projected_replica_bytes <= replica_capacity_bytes
        passed_bytes = decode_passed and replica_passed
        occupancy_before = existing_bytes / decode_capacity_bytes if decode_capacity_bytes and decode_capacity_bytes > 0 else None
        occupancy_after = projected_decode_bytes / decode_capacity_bytes if decode_capacity_bytes and decode_capacity_bytes > 0 else None
        replica_occupancy_before = (
            replica_existing_bytes / replica_capacity_bytes
            if replica_capacity_bytes and replica_capacity_bytes > 0
            else None
        )
        replica_occupancy_after = (
            projected_replica_bytes / replica_capacity_bytes
            if replica_capacity_bytes and replica_capacity_bytes > 0
            else None
        )
        return {
            "passed": passed_bytes,
            "reason": None if passed_bytes else "kv_capacity",
            "admission": "pass" if passed_bytes else "fail",
            "context_tokens_after": context_tokens_after,
            "existing_kv_bytes": existing_bytes,
            "projected_kv_bytes": projected_decode_bytes,
            "capacity_bytes": decode_capacity_bytes,
            "required_pages": required_pages,
            "required_bytes": required_bytes,
            "replica_projected_bytes": replica_bytes,
            "replica_existing_bytes": replica_existing_bytes,
            "replica_target_projected_bytes": projected_replica_bytes,
            "replica_target_capacity_bytes": replica_capacity_bytes,
            "replica_target_deployment_id": (
                getattr(replica_target_deployment, "deployment_id", None)
                if replica_target_deployment is not None
                else None
            ),
            "decode_capacity_passed": decode_passed,
            "replica_capacity_passed": replica_passed,
            "occupancy_before": occupancy_before,
            "occupancy_after": occupancy_after,
            "replica_occupancy_before": replica_occupancy_before,
            "replica_occupancy_after": replica_occupancy_after,
            "hbm_model": "aggregate_hbm_bytes",
            "card_level_hbm_admission": False,
        }
    capacity_tokens = getattr(decode_deployment, "kv_cache_capacity_tokens", None)
    existing_tokens = _pd_existing_kv_tokens_for_decode(
        state,
        decode_deployment,
        exclude_session_id=request.get("session_id"),
    )
    replica_tokens = context_tokens_after if kv_post_turn_action == "replicate_kv" and replica_target_deployment is not None else 0
    replica_existing_tokens = (
        _pd_existing_kv_tokens_for_decode(
            state,
            replica_target_deployment,
            exclude_session_id=request.get("session_id"),
        )
        if replica_tokens and replica_target_deployment is not None
        else 0
    )
    replica_capacity_tokens = (
        getattr(replica_target_deployment, "kv_cache_capacity_tokens", None)
        if replica_target_deployment is not None
        else None
    )
    replica_capacity_tokens = int(replica_capacity_tokens) if replica_capacity_tokens is not None else None
    projected_tokens = existing_tokens + context_tokens_after
    projected_replica_tokens = replica_existing_tokens + replica_tokens
    if capacity_tokens is None:
        replica_passed = replica_capacity_tokens is None or projected_replica_tokens <= replica_capacity_tokens
        return {
            "passed": replica_passed,
            "reason": None if replica_passed else "kv_capacity",
            "admission": "unknown" if replica_passed else "fail",
            "context_tokens_after": context_tokens_after,
            "existing_kv_tokens": existing_tokens,
            "projected_kv_tokens": projected_tokens,
            "capacity_tokens": None,
            "replica_projected_tokens": replica_tokens,
            "replica_existing_tokens": replica_existing_tokens,
            "replica_target_projected_tokens": projected_replica_tokens,
            "replica_target_capacity_tokens": replica_capacity_tokens,
            "replica_target_deployment_id": (
                getattr(replica_target_deployment, "deployment_id", None)
                if replica_target_deployment is not None
                else None
            ),
            "decode_capacity_passed": True,
            "replica_capacity_passed": replica_passed,
            "occupancy_before": None,
            "occupancy_after": None,
            "replica_occupancy_before": (
                replica_existing_tokens / replica_capacity_tokens
                if replica_capacity_tokens is not None and replica_capacity_tokens > 0
                else None
            ),
            "replica_occupancy_after": (
                projected_replica_tokens / replica_capacity_tokens
                if replica_capacity_tokens is not None and replica_capacity_tokens > 0
                else None
            ),
            "hbm_model": "deployment_token_capacity_surrogate",
            "card_level_hbm_admission": False,
        }
    capacity_tokens = int(capacity_tokens)
    decode_passed = projected_tokens <= capacity_tokens
    replica_passed = replica_capacity_tokens is None or projected_replica_tokens <= replica_capacity_tokens
    passed = decode_passed and replica_passed
    return {
        "passed": passed,
        "reason": None if passed else "kv_capacity",
        "admission": "pass" if passed else "fail",
        "context_tokens_after": context_tokens_after,
        "existing_kv_tokens": existing_tokens,
        "projected_kv_tokens": projected_tokens,
        "capacity_tokens": capacity_tokens,
        "replica_projected_tokens": replica_tokens,
        "replica_existing_tokens": replica_existing_tokens,
        "replica_target_projected_tokens": projected_replica_tokens,
        "replica_target_capacity_tokens": replica_capacity_tokens,
        "replica_target_deployment_id": (
            getattr(replica_target_deployment, "deployment_id", None)
            if replica_target_deployment is not None
            else None
        ),
        "decode_capacity_passed": decode_passed,
        "replica_capacity_passed": replica_passed,
        "occupancy_before": existing_tokens / capacity_tokens if capacity_tokens > 0 else None,
        "occupancy_after": projected_tokens / capacity_tokens if capacity_tokens > 0 else None,
        "replica_occupancy_before": (
            replica_existing_tokens / replica_capacity_tokens
            if replica_capacity_tokens is not None and replica_capacity_tokens > 0
            else None
        ),
        "replica_occupancy_after": (
            projected_replica_tokens / replica_capacity_tokens
            if replica_capacity_tokens is not None and replica_capacity_tokens > 0
            else None
        ),
        "hbm_model": "deployment_token_capacity_surrogate",
        "card_level_hbm_admission": False,
    }

def _pd_existing_kv_tokens_for_decode(
    state: RuntimeState,
    decode_deployment: Any,
    *,
    exclude_session_id: str | None,
) -> int:
    decode_id = getattr(decode_deployment, "deployment_id", None)
    total = 0
    for session_id, session in state.sessions.items():
        if exclude_session_id is not None and str(session_id) == str(exclude_session_id):
            continue
        if getattr(session, "owner_deployment_id", None) == decode_id:
            total += max(0, int(getattr(session, "kv_cache_tokens", 0) or 0))
        if decode_id in tuple(str(item) for item in getattr(session, "replica_deployment_ids", ()) or ()):
            total += max(0, int(getattr(session, "kv_cache_tokens", 0) or 0))
    return total

def _pd_existing_kv_bytes_for_decode(
    state: RuntimeState,
    decode_deployment: Any,
    *,
    exclude_session_id: str | None,
) -> float:
    decode_id = getattr(decode_deployment, "deployment_id", None)
    total = 0.0
    for session_id, session in state.sessions.items():
        if exclude_session_id is not None and str(session_id) == str(exclude_session_id):
            continue
        if getattr(session, "owner_deployment_id", None) == decode_id:
            total += max(0.0, float(getattr(session, "kv_cache_bytes", 0.0) or 0.0))
        if decode_id in tuple(str(item) for item in getattr(session, "replica_deployment_ids", ()) or ()):
            total += max(0.0, float(getattr(session, "kv_cache_bytes", 0.0) or 0.0))
    return total

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
    for session in state.sessions.values():
        if getattr(session, "owner_deployment_id", None) == deployment_id:
            primary_tokens += max(0, int(getattr(session, "kv_cache_tokens", 0) or 0))
            primary_bytes += max(0.0, float(getattr(session, "kv_cache_bytes", 0.0) or 0.0))
        if deployment_id in tuple(str(item) for item in getattr(session, "replica_deployment_ids", ()) or ()):
            replica_tokens += max(0, int(getattr(session, "kv_cache_tokens", 0) or 0))
            replica_bytes += max(0.0, float(getattr(session, "kv_cache_bytes", 0.0) or 0.0))
    cache.aggregate_used_tokens_by_kind = {"primary": primary_tokens, "replica": replica_tokens}
    cache.aggregate_used_bytes_by_kind = {"primary": primary_bytes, "replica": replica_bytes}

def _update_pd_session_state(
    *,
    state: RuntimeState,
    request: dict[str, Any],
    runtime_strategy: Any,
    prefill: Any,
    decode: Any,
    session_action: str,
    prefill_context: dict[str, Any],
    context_tokens_after: int,
    kv_post_turn_action: str | None = None,
    replica_target_deployment: Any | None = None,
    now_s: float,
) -> None:
    session_id = request.get("session_id")
    if not session_id:
        return
    session = state.session_for(str(session_id))
    decode_entry = _deployment_entry_satellite(decode)
    kv_cache_bytes_per_token = float(
        _first_present(
            getattr(decode, "kv_cache_bytes_per_token", None),
            getattr(prefill, "kv_cache_bytes_per_token", None),
            0.0,
        )
    )
    session.owner_satellite_id = decode_entry
    session.owner_deployment_id = getattr(decode, "deployment_id", None)
    session.owner_role = "decode"
    session.owner_satellite_ids = tuple(str(item) for item in getattr(decode, "member_satellites", ()) or (decode_entry,))
    session.paired_prefill_deployment_id = getattr(prefill, "deployment_id", None)
    session.context_tokens = context_tokens_after
    session.kv_cache_tokens = context_tokens_after
    session.kv_cache_bytes = _kv_accounted_bytes(
        context_tokens_after,
        kv_bytes_per_token=kv_cache_bytes_per_token,
        kv_policy=_kv_policy(runtime_strategy),
    )
    session.kv_pages = _kv_required_pages(context_tokens_after, _kv_page_size_tokens(_kv_policy(runtime_strategy)))
    session.context_version += 1
    session.last_seen_s = now_s
    ttl_s = _session_ttl_s(runtime_strategy)
    session.expires_at_s = now_s + ttl_s if ttl_s is not None else None
    session.last_session_action = session_action
    if kv_post_turn_action == "evict_session":
        session.owner_satellite_id = None
        session.owner_deployment_id = None
        session.owner_role = None
        session.owner_satellite_ids = ()
        session.paired_prefill_deployment_id = None
        session.kv_cache_tokens = 0
        session.kv_cache_bytes = 0.0
        session.kv_pages = 0
        session.expires_at_s = None
        session.replica_deployment_ids = ()
    elif kv_post_turn_action == "replicate_kv" and replica_target_deployment is not None:
        session.replica_deployment_ids = (str(getattr(replica_target_deployment, "deployment_id")),)
    else:
        session.replica_deployment_ids = ()

def _expire_session_kv(state: RuntimeState, now_s: float) -> None:
    for session in state.sessions.values():
        expires_at_s = getattr(session, "expires_at_s", None)
        if expires_at_s is None or now_s < expires_at_s:
            continue
        session.owner_satellite_id = None
        session.owner_deployment_id = None
        session.owner_role = None
        session.owner_satellite_ids = ()
        session.paired_prefill_deployment_id = None
        session.kv_cache_tokens = 0
        session.kv_cache_bytes = 0.0
        session.expires_at_s = None
        session.replica_deployment_ids = ()
        session.last_session_action = "expired"

def _session_ttl_s(runtime_strategy: Any) -> float | None:
    overrides = _dict(getattr(runtime_strategy, "hard_gate_overrides", {}))
    kv_policy = _kv_policy(runtime_strategy)
    value = _first_present(
        kv_policy.get("session_ttl_s"),
        overrides.get("session_ttl_s"),
        overrides.get("kv_session_ttl_s"),
        getattr(runtime_strategy, "session_ttl_s", None),
    )
    ttl_s = _optional_float(value)
    return ttl_s if ttl_s is not None and ttl_s >= 0 else None

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

def _pd_catalog_externalizes_kv(deployment_catalog: RuntimeDeploymentCatalog) -> bool:
    values = _pd_internal_comm_accounting_values(deployment_catalog)
    return bool(values) and all(value == "externalized_pd_kv" for value in values)

def _pd_catalog_has_kv_bytes(deployment_catalog: RuntimeDeploymentCatalog) -> bool:
    deployments = tuple(getattr(deployment_catalog, "prefill_deployments", ()) or ()) + tuple(
        getattr(deployment_catalog, "decode_deployments", ()) or ()
    )
    return bool(deployments) and all(
        _optional_float(getattr(deployment, "kv_cache_bytes_per_token", None)) is not None
        and (_optional_float(getattr(deployment, "kv_cache_bytes_per_token", None)) or 0.0) > 0
        for deployment in deployments
    )

def _pd_internal_comm_accounting_values(deployment_catalog: RuntimeDeploymentCatalog) -> tuple[str, ...]:
    deployments = tuple(getattr(deployment_catalog, "prefill_deployments", ()) or ()) + tuple(
        getattr(deployment_catalog, "decode_deployments", ()) or ()
    )
    return tuple(str(getattr(deployment, "internal_comm_accounting", "embedded")) for deployment in deployments)

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

