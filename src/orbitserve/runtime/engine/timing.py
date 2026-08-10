"""Shared TTFT stage accounting for colocated and PD runtime paths."""

from __future__ import annotations

from typing import Any


TTFT_DEFINITION = "gateway_request_arrival_to_first_output_token_at_exit_gateway"


def ttft_breakdown(
    *,
    access_wait_s: float,
    uplink_s: float,
    prefill_queue_s: float,
    prefill_compute_s: float,
    kv_transfer_s: float,
    decode_queue_s: float,
    first_token_decode_s: float,
    return_path_s: float,
) -> dict[str, Any]:
    stages = {
        "access_wait_s": max(0.0, access_wait_s),
        "uplink_s": max(0.0, uplink_s),
        "prefill_queue_s": max(0.0, prefill_queue_s),
        "prefill_compute_s": max(0.0, prefill_compute_s),
        "kv_transfer_s": max(0.0, kv_transfer_s),
        "decode_queue_s": max(0.0, decode_queue_s),
        "first_token_decode_s": max(0.0, first_token_decode_s),
        "return_path_s": max(0.0, return_path_s),
    }
    return {
        **stages,
        "ttft_s": sum(stages.values()),
        "definition": TTFT_DEFINITION,
    }


__all__ = ["TTFT_DEFINITION", "ttft_breakdown"]
