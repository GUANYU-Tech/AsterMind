import threading

import pytest

from evaluation.serving.experiments import request_router


class _Handler:
    def __init__(self):
        self.headers = {"X-Request-Id": "request-000000000059"}


def _state():
    return {
        "model": "qwen3-8b",
        "prefill_base_url": "http://127.0.0.1:31400",
        "decode_base_url": "http://127.0.0.1:31401",
        "handoff_transport": "p2p_request_id",
        "prefill_kv_ip": "127.0.0.1",
        "prefill_kv_port": 31410,
        "decode_kv_ip": "127.0.0.1",
        "decode_kv_port": 31420,
        "handoff_log": None,
        "run_id": "pd-test-run",
        "run_origin_monotonic_ns": 0,
        "host_boot_id": "boot-test",
        "request_sequence": 0,
        "lock": threading.Lock(),
    }


def test_router_records_prefill_failure_before_decode(monkeypatch):
    records = []

    def fail_prefill(*args, **kwargs):
        raise RuntimeError("upstream request failed for Prefill: connection reset")

    monkeypatch.setattr(request_router, "_request_json", fail_prefill)
    monkeypatch.setattr(
        request_router,
        "_append_handoff_record",
        lambda state, record: records.append(record),
    )

    with pytest.raises(RuntimeError, match="connection reset"):
        request_router._route_streaming_completion(
            _Handler(), _state(), {"model": "qwen3-8b", "stream": True}
        )

    assert len(records) == 1
    record = records[0]
    assert record["request_idx"] == 59
    assert record["router_status"] == "failed"
    assert record["producer_control_response"] is None
    assert record["consumer_request_sent"] is None
    assert record["failure"] == (
        "RuntimeError: upstream request failed for Prefill: connection reset"
    )
