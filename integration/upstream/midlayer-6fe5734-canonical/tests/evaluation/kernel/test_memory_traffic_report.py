import pytest

import evaluation.kernel.memory_traffic_report as memory_traffic_report


def _mechanism(ordinal, l2_cache, dram, measured_l2, measured_hbm):
    return {
        "mechanisms": {
            "kernel_ordinal": ordinal,
            "kernel_name": f"kernel_{ordinal}",
            "analytical_input": {
                "demand_per_resource": {"l2_cache": l2_cache, "dram": dram},
                # metax_c500's predicted_l2_bytes reads the directional
                # read-path ledger (see memory_traffic_report._predicted_l2_bytes);
                # putting the whole demand on a single "read" path keeps this
                # fixture's read-only extraction equal to the aggregate demand
                # used by every other hardware, so it does not need its own
                # assertions below.
                "memory_paths": [
                    {
                        "access_kind": "read",
                        "nodes": [
                            {"resource_role": "l2_cache", "bytes": l2_cache},
                            {"resource_role": "dram", "bytes": dram},
                        ],
                    },
                ],
            },
            "observation": {
                "l2_source_interface_bytes": measured_l2,
                "downstream_global_bytes": measured_hbm,
            },
        },
    }


def _result(task_key, kernels, forward_pair=(2000.0, 1000.0)):
    return (
        task_key, {"op_type": "GEMM"}, None, forward_pair, None, None,
        kernels,
    )


def test_collect_hardware_traffic_extracts_kernel_points(monkeypatch):
    monkeypatch.setattr(
        memory_traffic_report,
        "_hardware_work",
        lambda hardware_id: ("hardware.yaml", [
            (f"{hardware_id}-case", {"packed": True}),
            (f"{hardware_id}-fa-row", {"capture_evidence": {}}),
        ]),
    )

    def process(packed):
        return _result(packed[0], [
            _mechanism(0, 100.0, 40.0, 120.0, 50.0),
            _mechanism(1, 10.0, 4.0, None, 5.0),
        ])

    monkeypatch.setattr(memory_traffic_report, "_process_op_star", process)

    traffic = memory_traffic_report.collect_hardware_traffic(jobs=1)

    assert set(traffic) == {"nvidia_a800", "metax_c500", "ascend_910b"}
    assert traffic["nvidia_a800"]["dataset_case_count"] == 1
    assert traffic["nvidia_a800"]["fa_serving_capture_case_ids"] == [
        "nvidia_a800-fa-row"]
    assert traffic["nvidia_a800"]["hardware_config"] == "hardware.yaml"
    assert traffic["nvidia_a800"]["kernel_points"] == [
        {
            "case_id": "nvidia_a800-case",
            "family": "GEMM",
            "op_type": "GEMM",
            "kernel_ordinal": 0,
            "kernel_name": "kernel_0",
            "predicted_l2_bytes": 100.0,
            "predicted_hbm_bytes": 40.0,
            "measured_l2_bytes": 120.0,
            "measured_hbm_bytes": 50.0,
        },
        {
            "case_id": "nvidia_a800-case",
            "family": "GEMM",
            "op_type": "GEMM",
            "kernel_ordinal": 1,
            "kernel_name": "kernel_1",
            "predicted_l2_bytes": 10.0,
            "predicted_hbm_bytes": 4.0,
            "measured_l2_bytes": None,
            "measured_hbm_bytes": 5.0,
        },
    ]


def test_collect_hardware_traffic_validates_jobs():
    with pytest.raises(ValueError, match="jobs must be positive"):
        memory_traffic_report.collect_hardware_traffic(jobs=0)
    with pytest.raises(TypeError, match="jobs must be an integer"):
        memory_traffic_report.collect_hardware_traffic(jobs=True)


def test_kernel_traffic_points_requires_forward_result():
    with pytest.raises(RuntimeError, match="no Forward result"):
        memory_traffic_report._kernel_traffic_points(
            "nvidia_a800", _result("case", [], forward_pair="ERR"))


def test_aggregate_case_traffic_sums_matched_kernels():
    points = [
        {
            "case_id": "a", "family": "GEMM",
            "measured_l2_bytes": 100.0, "predicted_l2_bytes": 110.0,
            "measured_hbm_bytes": 50.0, "predicted_hbm_bytes": 40.0,
        },
        {
            "case_id": "a", "family": "GEMM",
            "measured_l2_bytes": None, "predicted_l2_bytes": 10.0,
            "measured_hbm_bytes": 5.0, "predicted_hbm_bytes": 6.0,
        },
    ]

    cases, excluded, unmatched = memory_traffic_report.aggregate_case_traffic(
        points, "l2")

    assert unmatched == 1
    assert excluded == []
    assert cases == [{
        "case_id": "a",
        "family": "GEMM",
        "measured_bytes": 100.0,
        "predicted_bytes": 110.0,
        "matched_kernel_count": 1,
        "total_kernel_count": 2,
    }]

    cases, excluded, unmatched = memory_traffic_report.aggregate_case_traffic(
        points, "hbm")

    assert unmatched == 0
    assert excluded == []
    assert cases[0]["measured_bytes"] == 55.0
    assert cases[0]["predicted_bytes"] == 46.0
    assert cases[0]["matched_kernel_count"] == 2


def test_aggregate_case_traffic_excludes_cases_without_positive_traffic():
    points = [
        {
            "case_id": "none-case", "family": "Vector",
            "measured_l2_bytes": None, "predicted_l2_bytes": None,
            "measured_hbm_bytes": None, "predicted_hbm_bytes": None,
        },
        {
            "case_id": "zero-case", "family": "Vector",
            "measured_l2_bytes": 0.0, "predicted_l2_bytes": 0.0,
            "measured_hbm_bytes": 0.0, "predicted_hbm_bytes": 0.0,
        },
    ]

    cases, excluded, unmatched = memory_traffic_report.aggregate_case_traffic(
        points, "l2")

    assert cases == []
    assert unmatched == 1
    assert excluded == [
        {
            "case_id": "none-case",
            "family": "Vector",
            "reason": "no_matched_kernel_traffic",
            "matched_kernel_count": 0,
            "total_kernel_count": 1,
        },
        {
            "case_id": "zero-case",
            "family": "Vector",
            "reason": "non_positive_case_traffic",
            "matched_kernel_count": 1,
            "total_kernel_count": 1,
        },
    ]


def test_aggregate_case_traffic_rejects_negative_bytes():
    points = [{
        "case_id": "bad", "family": "GEMM",
        "measured_l2_bytes": -1.0, "predicted_l2_bytes": 1.0,
        "measured_hbm_bytes": 1.0, "predicted_hbm_bytes": 1.0,
    }]

    with pytest.raises(ValueError, match="negative l2 traffic"):
        memory_traffic_report.aggregate_case_traffic(points, "l2")


def test_aggregate_case_traffic_rejects_unknown_metric():
    with pytest.raises(ValueError, match="unknown traffic metric"):
        memory_traffic_report.aggregate_case_traffic([], "l3")


def test_summarize_cases_reports_mape_and_bias():
    summary = memory_traffic_report._summarize_cases([
        {"measured_bytes": 100.0, "predicted_bytes": 110.0},
        {"measured_bytes": 200.0, "predicted_bytes": 100.0},
    ])

    assert summary["case_count"] == 2
    assert summary["mape_percent"] == pytest.approx(30.0)
    assert summary["median_ape_percent"] == pytest.approx(30.0)
    assert summary["mean_prediction_over_measured"] == pytest.approx(0.8)


def test_summarize_cases_requires_points():
    with pytest.raises(ValueError, match="at least one case"):
        memory_traffic_report._summarize_cases([])


def test_build_traffic_summary_reports_per_hardware_metrics():
    traffic = {
        hardware_id: {
            "hardware_config": "hardware.yaml",
            "dataset_case_count": 1,
            "fa_serving_capture_case_ids": [],
            "kernel_points": [{
                "case_id": f"{hardware_id}-case",
                "family": "GEMM",
                "measured_l2_bytes": 100.0,
                "predicted_l2_bytes": 110.0,
                "measured_hbm_bytes": 40.0,
                "predicted_hbm_bytes": 50.0,
            }],
        }
        for hardware_id in ("nvidia_a800", "metax_c500", "ascend_910b")
    }

    aggregated = memory_traffic_report.aggregate_traffic(traffic)
    document = memory_traffic_report.build_traffic_summary(traffic, aggregated)

    assert document["schema_version"] == 1
    assert [entry["hardware_id"] for entry in document["hardware"]] == [
        "nvidia_a800", "metax_c500", "ascend_910b"]
    entry = document["hardware"][0]
    assert entry["kernel_count"] == 1
    assert entry["metrics"]["l2"]["mape_percent"] == pytest.approx(10.0)
    assert (
        entry["metrics"]["hbm"]["mean_prediction_over_measured"]
        == pytest.approx(1.25))
    assert entry["metrics"]["l2"]["excluded_cases"] == []
    assert entry["metrics"]["l2"]["unmatched_kernel_count"] == 0
