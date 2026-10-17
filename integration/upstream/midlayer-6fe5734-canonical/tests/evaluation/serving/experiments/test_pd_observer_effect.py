import json

import pytest

from evaluation.serving.experiments import pd_observer_effect


def _run(trace_enabled, ttft_ms):
    return {
        "schema_version": 1,
        "status": "success",
        "workload": {
            "request_count": 1,
            "expected_qps": 4.0,
            "seed": 42,
            "identity": {
                "hardware_id": "nvidia_a800",
                "model_id": "qwen3-8b",
                "case_id": "tp1_qps4",
            },
            "requests_sha256": "requests",
            "request_stream_sha256": "stream",
            "dataset_sha256": "dataset",
        },
        "server": {
            "profiler_attached": False,
            "scheduler_trace": {"enabled": trace_enabled},
        },
        "metrics": {
            "request_throughput_per_s": 10.0,
            "output_token_throughput_per_s": 20.0,
            "ttft": {"mean_ms": ttft_ms},
            "tpot": {"mean_ms": 5.0},
            "e2e": {"mean_ms": 20.0},
            "per_request": [{
                "request_id": "request-0",
                "scheduled_arrival_s": 0.0,
                "input_tokens": 10,
                "output_tokens": 2,
            }],
        },
    }


def test_observer_effect_accepts_sub_three_percent_pair(tmp_path):
    off = tmp_path / "off.json"
    on = tmp_path / "on.json"
    off.write_text(json.dumps(_run(False, 10.0)))
    on.write_text(json.dumps(_run(True, 10.2)))

    report = pd_observer_effect.compare_pd_observer_effect(off, on)

    assert report["passed"] is True
    assert report["metric_delta_percent"]["ttft"] == pytest.approx(2.0)
