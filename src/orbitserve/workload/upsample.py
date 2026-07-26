"""Lazy deterministic session upsampling for materialized workloads (engine runtime).

``stream_upsampled_sessions`` replicates each session of a loaded bundle
``factor`` times with deterministic replica ids and a bin-local arrival
jitter, streaming one (session, requests) pair at a time so even factor×
expansion of very large workloads never materializes in memory.
"""

from __future__ import annotations

import math
from typing import Any, Iterator

from .contract import WorkloadGenerationResult, stable_hex, stable_seed

DEFAULT_TIME_BIN_S = 300.0


def _group_requests(requests: tuple[dict[str, Any], ...]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in requests:
        session_id = row.get("session_id")
        if session_id is not None:
            grouped.setdefault(str(session_id), []).append(row)
    return grouped


def stream_upsampled_sessions(
    workload: WorkloadGenerationResult,
    factor: int,
    *,
    seed: int | str,
) -> Iterator[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Yield ``(session, [requests])`` with every session replicated ``factor`` times.

    Sessions stream in (session_start_time_s, session_id) order. Each replica
    derives its ids deterministically via contract.stable_hex/stable_seed over
    ``(seed, original session_id, copy index)`` and shifts all of its arrivals
    by one bin-local jitter (bin width = manifest ``time.time_bin_s``, default
    300 s) drawn from the copy-index hash and clamped so the replica's first
    turn stays inside its original bin and its last turn stays inside the
    workload horizon. Turn structure (turn_index, session_turn_count,
    inter_turn_gap_s) is preserved verbatim; request rows keep the exact
    REQUEST_KEYS field set. ``factor=1`` yields the original sessions
    unchanged. The factor× expansion is never materialized as a list.
    """
    if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
        raise ValueError(f"factor must be a positive integer, got {factor!r}")
    sessions = sorted(
        workload.sessions,
        key=lambda row: (float(row["session_start_time_s"]), str(row["session_id"])),
    )
    by_session = _group_requests(workload.requests)
    if factor == 1:
        for session in sessions:
            yield session, list(by_session.get(str(session["session_id"]), []))
        return
    manifest = workload.manifest if isinstance(workload.manifest, dict) else {}
    time_cfg = manifest.get("time") if isinstance(manifest.get("time"), dict) else {}
    bin_s = float(time_cfg.get("time_bin_s") or DEFAULT_TIME_BIN_S)
    horizon = float(time_cfg.get("time_horizon_s") or 0.0)
    seed_key = str(seed)
    for session in sessions:
        base_session_id = str(session["session_id"])
        rows = sorted(by_session.get(base_session_id, []), key=lambda row: int(row["turn_index"]))
        first = float(rows[0]["arrival_time_s"]) if rows else float(session["session_start_time_s"])
        last = float(rows[-1]["arrival_time_s"]) if rows else first
        bin_start = math.floor(first / bin_s) * bin_s
        bin_end = min(bin_start + bin_s, horizon) if horizon > 0 else bin_start + bin_s
        low = bin_start - first
        high = bin_end - 1e-6 - last
        for copy_index in range(factor):
            tag = stable_hex(seed_key, base_session_id, str(copy_index), length=16)
            new_session_id = f"{base_session_id}_up{copy_index}_{tag}"
            # One jitter per replica, hashed from the copy index, in [-bin_s, bin_s).
            unit = stable_seed(seed_key, base_session_id, str(copy_index), "jitter") / float((1 << 63) - 1)
            jitter = (2.0 * unit - 1.0) * bin_s
            shift = 0.0 if high < low else min(max(jitter, low), high)
            new_session = dict(session)
            new_session["session_id"] = new_session_id
            new_session["session_start_time_s"] = round(float(session["session_start_time_s"]) + shift, 6)
            new_session["seed"] = stable_seed(seed_key, new_session_id)
            new_rows: list[dict[str, Any]] = []
            for row in rows:
                request_tag = stable_hex(seed_key, str(row["request_id"]), str(copy_index), length=16)
                new_request_id = f"{row['request_id']}_up{copy_index}_{request_tag}"
                new_row = dict(row)
                new_row["request_id"] = new_request_id
                new_row["session_id"] = new_session_id
                new_row["arrival_time_s"] = round(float(row["arrival_time_s"]) + shift, 6)
                new_row["seed"] = stable_seed(seed_key, new_request_id)
                new_rows.append(new_row)
            yield new_session, new_rows


__all__ = ["stream_upsampled_sessions"]
