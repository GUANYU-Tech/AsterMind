"""Runtime-owned KV-cache capacity derivation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


BYTES_PER_TB = 1_000_000_000_000


def derive_kv_cache_capacity_tokens(
    constellation: Mapping[str, Any] | None,
    serving_profile: Any,
) -> int:
    """Derive uniform per-satellite KV capacity from architecture and model data."""

    direct_capacity = getattr(serving_profile, "kv_cache_capacity_tokens", None)
    if direct_capacity is not None:
        capacity = int(direct_capacity)
        if capacity < 0:
            raise ValueError("runtime_kv_capacity_invalid")
        return capacity

    payload = dict(constellation or {})
    architecture = payload.get("architecture")
    if not isinstance(architecture, Mapping):
        raise ValueError("runtime_kv_architecture_missing")
    raw_storage = architecture.get("storage_capacity_tb")
    if raw_storage is None:
        raise ValueError("runtime_kv_storage_capacity_missing")
    storage_capacity_tb = float(raw_storage)
    if not math.isfinite(storage_capacity_tb) or storage_capacity_tb < 0.0:
        raise ValueError("runtime_kv_storage_capacity_invalid")

    raw_bytes_per_token = getattr(serving_profile, "kv_cache_bytes_per_token", None)
    if raw_bytes_per_token is None:
        raise ValueError("runtime_kv_bytes_per_token_missing")
    kv_cache_bytes_per_token = float(raw_bytes_per_token)
    if not math.isfinite(kv_cache_bytes_per_token) or kv_cache_bytes_per_token <= 0.0:
        raise ValueError("runtime_kv_bytes_per_token_invalid")

    return math.floor(
        storage_capacity_tb * BYTES_PER_TB / kv_cache_bytes_per_token
    )


__all__ = ["BYTES_PER_TB", "derive_kv_cache_capacity_tokens"]
