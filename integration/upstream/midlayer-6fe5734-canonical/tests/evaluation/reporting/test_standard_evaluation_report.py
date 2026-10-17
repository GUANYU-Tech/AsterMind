from evaluation.reporting.standard_evaluation_report import (
    _kernel_table,
    _kernel_traffic_exclusion_note,
    _kernel_traffic_table,
    _kernel_unsupported_note,
    _serving_case_table,
    _serving_summary_table,
)


def test_kernel_table_renders_explicit_forward_error():
    payload = {
        "kernel_matrix": {
            "nvidia_a800": {
                "status": "PASS",
                "rows": [["ALL", 10, None, None, None, None]],
            },
            "metax_c500": {
                "status": "PASS",
                "rows": [["ALL", 10, None, [1.0, 2.0, 3.0], None, None]],
            },
            "ascend_910b": {
                "status": "PASS",
                "rows": [["ALL", 10, None, None, None, None]],
            },
        },
    }

    table = _kernel_table(payload)

    assert table[-1] == (
        "| ALL | 10 | Err | Err | Err | "
        "10 | 1.00% | 2.00% | 3.00% | 10 | Err | Err | Err |"
    )


def test_serving_tables_render_explicit_unsupported_case():
    cell = {
        "hardware_id": "ascend_910b",
        "model_id": "llama-3.1-8b",
        "case_id": "tp1_qps16",
        "prediction_status": "unsupported_launch_coverage",
        "prediction_reason": "typed source program unavailable",
    }

    assert _serving_summary_table((cell,))[-1] == (
        "| Ascend 910B | 0 / 1 | 0 | 1 | Err | Err | Err |"
    )
    assert _serving_case_table((cell,))[-1] == (
        "| `llama-3.1-8b / tp1_qps16` | Unsupported: typed source program "
        "unavailable | Err | Err | Err | Err | Err | — |"
    )


def test_serving_tables_render_dev_only_claim_and_capture_delta():
    cell = {
        "hardware_id": "metax_c500",
        "model_id": "qwen3-30b-a3b",
        "case_id": "tp2_ep2_qps4",
        "prediction_status": "completed",
        "prediction_reason": None,
        "accuracy_claim_status": "unsupported_development_only",
        "capture_runtime_comparison": {
            "status": "mismatch",
            "fields": {
                "dtype": {"status": "equal"},
                "runtime_version": {"status": "mismatch"},
                "source_commit": {"status": "mismatch"},
            },
        },
        "metrics": {
            "throughput": {
                "request_throughput_per_s": {
                    "observed": 4.0,
                    "predicted": 4.4,
                    "absolute_percentage_error": 10.0,
                },
            },
            "latency": {
                "ttft_ms": {"mean": {"observed": 100.0, "predicted": 110.0}},
                "tpot_ms": {"mean": {"observed": 50.0, "predicted": 40.0}},
            },
            "paired_request_error": {
                "ttft_ms": {"mape_percent": 12.5},
                "tpot_ms": {"mape_percent": 21.0},
            },
        },
    }

    assert _serving_summary_table((cell,))[-1] == (
        "| MetaX C500 | 1 / 1 | 1 | 0 | 10.00% | 12.50% | 21.00% |"
    )
    assert _serving_case_table((cell,))[-1] == (
        "| `qwen3-30b-a3b / tp2_ep2_qps4` | Dev-only | 1.100× | "
        "+10.00% | -20.00% | 12.50% | 21.00% | runtime_version |"
    )


def _traffic_entry(hardware_id, l2, hbm, fa_capture_ids=()):
    return {
        "hardware_id": hardware_id,
        "fa_serving_capture_case_ids": list(fa_capture_ids),
        "metrics": {"l2": l2, "hbm": hbm},
    }


def test_kernel_traffic_table_renders_all_hardware():
    payload = {
        "hardware": [
            _traffic_entry(
                "nvidia_a800",
                {"case_count": 10, "mape_percent": 12.345,
                 "median_ape_percent": 6.789,
                 "mean_prediction_over_measured": 1.05},
                {"case_count": 9, "mape_percent": 8.0,
                 "median_ape_percent": 5.5,
                 "mean_prediction_over_measured": 0.98},
            ),
            _traffic_entry(
                "metax_c500",
                {"case_count": 11, "mape_percent": 20.0,
                 "median_ape_percent": 15.0,
                 "mean_prediction_over_measured": 1.2},
                {"case_count": 11, "mape_percent": 21.5,
                 "median_ape_percent": 16.5,
                 "mean_prediction_over_measured": 1.1},
            ),
            _traffic_entry(
                "ascend_910b",
                {"case_count": 12, "mape_percent": 30.25,
                 "median_ape_percent": 25.75,
                 "mean_prediction_over_measured": 0.85},
                {"case_count": 12, "mape_percent": 31.0,
                 "median_ape_percent": 26.0,
                 "mean_prediction_over_measured": 0.9},
            ),
        ],
    }

    table = _kernel_traffic_table(payload)

    assert table[0] == (
        "| Hardware | L2 cases | L2 MAPE | L2 MedAPE | L2 mean P/M | "
        "HBM cases | HBM MAPE | HBM MedAPE | HBM mean P/M |"
    )
    assert table[2] == (
        "| NVIDIA A800 | 10 | 12.35% | 6.79% | 1.050× | "
        "9 | 8.00% | 5.50% | 0.980× |"
    )
    assert table[-1] == (
        "| Ascend 910B | 12 | 30.25% | 25.75% | 0.850× | "
        "12 | 31.00% | 26.00% | 0.900× |"
    )


def test_kernel_traffic_table_rejects_missing_hardware():
    payload = {
        "hardware": [
            _traffic_entry(
                "nvidia_a800",
                {"case_count": 1, "mape_percent": 1.0,
                 "median_ape_percent": 1.0,
                 "mean_prediction_over_measured": 1.0},
                {"case_count": 1, "mape_percent": 1.0,
                 "median_ape_percent": 1.0,
                 "mean_prediction_over_measured": 1.0},
            ),
        ],
    }

    try:
        _kernel_traffic_table(payload)
    except ValueError as error:
        assert "hardware set" in str(error)
    else:
        raise AssertionError("incomplete hardware set must fail")


def test_kernel_traffic_exclusion_note_lists_only_present_hardware():
    stats = {
        "case_count": 1, "mape_percent": 1.0, "median_ape_percent": 1.0,
        "mean_prediction_over_measured": 1.0,
    }
    payload = {
        "hardware": [
            _traffic_entry("nvidia_a800", stats, stats,
                           fa_capture_ids=("row-a", "row-b")),
            _traffic_entry("metax_c500", stats, stats),
            _traffic_entry("ascend_910b", stats, stats,
                           fa_capture_ids=("row-c",)),
        ],
    }

    note = _kernel_traffic_exclusion_note(payload)

    assert note == [
        "FA serving capture 行只有 op-level 证据，不提供 per-kernel traffic，未参与本图（"
        "NVIDIA A800 2 个、Ascend 910B 1 个）；"
        "按 metric 聚合时被过滤的 case 明细见结构化结果。",
        "",
    ]
    payload["hardware"][0]["fa_serving_capture_case_ids"] = []
    payload["hardware"][2]["fa_serving_capture_case_ids"] = []
    assert _kernel_traffic_exclusion_note(payload) == []


def test_kernel_table_renders_candidate_hardware_scored_subset():
    rows = [["ALL", 10, None, [1.0, 2.0, 3.0], None, None]]
    payload = {"kernel_matrix": {
        hardware_id: {"status": "PASS", "summary": "", "rows": rows}
        for hardware_id in ("nvidia_a800", "metax_c500", "ascend_910b")
    }}
    payload["kernel_matrix"]["nvidia_h200"] = {
        "status": "PASS", "summary": "", "rows": rows,
        "forward_coverage": {"scored": [[0, 7, 10]],
                             "unsupported_cases": {"ALL": 3, "FLASHATTN": 3}},
    }

    table = _kernel_table(payload)

    assert table[0].startswith("| Bucket | A800 N |")
    assert "| H200 N | H200 MAPE |" in table[0]
    assert table[-1] == (
        "| ALL | 10 | 1.00% | 2.00% | 3.00% | 7/10 | 1.00% | 2.00% | 3.00% "
        "| 10 | 1.00% | 2.00% | 3.00% | 10 | 1.00% | 2.00% | 3.00% |")
    note = _kernel_unsupported_note(payload)
    assert "共 3 个（FLASHATTN 3）" in note[0]
