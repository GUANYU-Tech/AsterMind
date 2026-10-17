import evaluation.kernel.latency_scatter_report as latency_scatter_report


def test_hardware_points_omit_explicitly_unsupported_forward(monkeypatch):
    monkeypatch.setattr(
        latency_scatter_report,
        "load_kernel_evaluation_dataset",
        lambda _hardware_id: {"supported": {}, "unsupported": {}},
    )
    monkeypatch.setattr(
        latency_scatter_report,
        "get_hardware_config_path",
        lambda _hardware_id: "hardware.yaml",
    )

    def process(packed):
        task_key = packed[0]
        forward_pair = (2000.0, 1000.0) if task_key == "supported" else "ERR"
        return task_key, {"op_type": "GEMM"}, None, forward_pair, None, None, []

    monkeypatch.setattr(latency_scatter_report, "_process_op_star", process)

    hardware_id, points = latency_scatter_report._hardware_points(
        "nvidia_a800", jobs=2)

    assert hardware_id == "nvidia_a800"
    assert points == [(1.0, 2.0, 0)]
