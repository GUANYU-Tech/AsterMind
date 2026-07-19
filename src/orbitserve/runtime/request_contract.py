"""Strict adapter from canonical workload requests to Runtime inputs."""

from __future__ import annotations

from typing import Any


REQUIRED_RUNTIME_REQUEST_FIELDS = (
    "request_id",
    "arrival_time_s",
    "prompt_tokens",
    "output_tokens",
    "ingress_id",
)


def normalize_runtime_request(raw_request: dict[str, Any], index: int = 0) -> dict[str, Any]:
    missing = tuple(key for key in REQUIRED_RUNTIME_REQUEST_FIELDS if raw_request.get(key) is None)
    if missing:
        raise ValueError(
            "runtime requests must use the canonical workload contract; "
            f"missing fields: {', '.join(missing)}"
        )
    prompt_tokens = max(0, _int(raw_request["prompt_tokens"], 0))
    output_tokens = max(0, _int(raw_request["output_tokens"], 0))
    return {
        "request_id": str(raw_request["request_id"]),
        "arrival_time_s": _float(raw_request["arrival_time_s"], 0.0),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "input_bytes": _float(raw_request.get("input_bytes"), prompt_tokens * 4.0),
        "output_bytes": _float(raw_request.get("output_bytes"), output_tokens * 4.0),
        "ttft_deadline_s": _optional_float(raw_request.get("ttft_deadline_s")),
        "e2e_deadline_s": _optional_float(raw_request.get("e2e_deadline_s")),
        "session_id": _optional_str(raw_request.get("session_id")),
        "ingress_id": _optional_str(raw_request["ingress_id"]),
        "prompt_tokens_declared": True,
        "new_user_tokens": _optional_int(raw_request.get("new_user_tokens")),
        "context_tokens_before_turn": _optional_int(raw_request.get("context_tokens_before_turn")),
        "context_window_tokens": _optional_int(raw_request.get("context_window_tokens")),
    }


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return _float(value, 0.0)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return _int(value, 0)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = ["REQUIRED_RUNTIME_REQUEST_FIELDS", "normalize_runtime_request"]
