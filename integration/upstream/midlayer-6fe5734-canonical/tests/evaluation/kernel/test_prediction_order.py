"""Fast contracts for the kernel prediction partial-order evaluator."""

from evaluation.kernel.prediction_order import (
    DEFAULT_EPSILON_NS,
    _canonical_gemm_pairs,
    _diagnostic_change_summary,
    _extents,
    _gemm_launch_m_values,
    _is_fail_closed_unsupported,
    _predict_case,
    _trend,
    _unsupported_records,
    _unsupported_summaries,
    build_scan_plan,
    compare_ordered_points,
)


def _matching_cases(plan, predicate):
    return [case for case in plan["cases"] if predicate(case)]


def _row(axis_value, total_ns, *, launch_algorithm="same"):
    return {
        "sequence_id": "sequence", "physical_label": "0:GEMM:gemm",
        "status": "success", "axis_value": axis_value,
        "hardware_id": "nvidia_a800", "family": "gemm",
        "model_name": "synthetic", "axis": "m",
        "tensor_parallel_size": 1, "expert_parallel_size": 1,
        "kernel_spec": {
            "kind": "GEMM", "dtype": "bf16",
            "shape": {"m": axis_value},
        },
        "launch": {
            "algorithm": launch_algorithm, "grid": [1, 1, 1],
            "grid_size": 1, "block": [1, 1, 1], "block_size": 1,
            "tiling": None, "cluster": None, "parameters": {},
            "execution_variant": "gemm", "source_program_type": None,
            "source_program": None,
        },
        "prediction": {
            "total_ns": total_ns, "launch_dispatch_ns": 1.0,
            "time_per_group_ns": {}, "time_per_resource_ns": {},
            "compute_time_by_role_ns": {}, "memory": [], "pipeline": None,
            "bottleneck_group": "synthetic", "critical_path": [],
            "support_level": "complete", "limitations": [],
        },
    }


def test_order_comparator_allows_monotone_plateau_and_numeric_noise():
    rows = (
        _row(1, 10.0),
        _row(2, 10.0),
        _row(3, 10.0 - DEFAULT_EPSILON_NS / 2),
        _row(4, 11.0),
    )
    assert compare_ordered_points(rows) == []


def test_order_comparator_retains_real_drop_and_launch_cause():
    rows = (
        _row(1, 10.0, launch_algorithm="before"),
        _row(2, 9.0, launch_algorithm="after"),
    )
    violations = compare_ordered_points(rows)
    assert len(violations) == 1
    violation = violations[0]
    assert violation["delta_ns"] == -1.0
    assert violation["drop_pct"] == 10.0
    assert violation["before_launch"]["algorithm"] == "before"
    assert violation["after_launch"]["algorithm"] == "after"
    assert violation["prediction_diff"]["total_ns"] == {"before": 10.0, "after": 9.0}
    assert "compute_time_by_role_ns" in violation["prediction_diff"]
    assert violation["root_cause"]["classification"] == "launch transition"
    assert "algorithm" in violation["launch_diff"]


def test_order_comparator_reports_source_program_transition():
    before = _row(1, 10.0)
    after = _row(2, 9.0)
    before["launch"]["source_program"] = {"m_tiles": 1}
    after["launch"]["source_program"] = {"m_tiles": 2}
    violation = compare_ordered_points((before, after))[0]
    assert violation["root_cause"] == {
        "classification": "source program transition",
        "trigger_fields": ["launch.source_program"],
        "source_function": (
            "src/kernel/launch_config/nvidia_a800/"
            "gemm_launch.py:select_gemm_launch"),
        "mechanism": "the production source program changed",
    }
    assert _diagnostic_change_summary(violation) == (
        "launch source program changed")


def test_pipeline_cause_cites_predictor_and_summarizes_component_change():
    before = _row(1, 10.0)
    after = _row(2, 9.0)
    before["prediction"]["pipeline"] = {
        "execution_time_ns": 9.0, "source_schedule": []}
    after["prediction"]["pipeline"] = {
        "execution_time_ns": 8.0, "source_schedule": []}
    violation = compare_ordered_points((before, after))[0]
    assert violation["root_cause"]["classification"] == "pipeline"
    assert (
        violation["root_cause"]["source_function"]
        == "src/kernel/predictors/gemm.py:GEMMPredictor.midlayer")
    assert _diagnostic_change_summary(violation) == (
        "pipeline execution: 9.000000→8.000000 ns")


def test_source_schedule_change_precedes_pipeline_duration_classification():
    before = _row(32, 88.0)
    after = _row(33, 62.0)
    before["prediction"]["pipeline"] = {
        "execution_time_ns": 88.0,
        "source_schedule": [{
            "task_name": "split_fuse_group0_compressed_program",
            "worker": 0,
            "repeat_count": 1,
        }],
    }
    after["prediction"]["pipeline"] = {
        "execution_time_ns": 62.0,
        "source_schedule": [
            {
                "task_name": f"split_fuse_group{worker}_compressed_program",
                "worker": worker,
                "repeat_count": 1,
            }
            for worker in range(2)
        ],
    }
    violation = compare_ordered_points((before, after))[0]
    assert violation["root_cause"] == {
        "classification": "source program transition",
        "trigger_fields": ["diagnostics.pipeline.source_schedule"],
        "source_function": (
            "src/kernel/predictors/gemm.py:GEMMPredictor.midlayer"),
        "mechanism": "the source-derived physical worker schedule changed",
    }
    assert _diagnostic_change_summary(violation) == (
        "source worker programs: 1→2")


def test_unsupported_records_preserve_input_and_aggregate_reason():
    unsupported = {
        "sequence_id": "unsupported", "physical_label": "unsupported",
        "status": "unsupported", "axis_value": 1,
        "hardware_id": "metax_c500", "family": "fused_moe",
        "model_name": "synthetic", "axis": "tokens",
        "tensor_parallel_size": 1, "expert_parallel_size": 1,
        "controls": {"dtype": "int4"},
        "recipe": {"shape": {"tokens": 1}},
        "unsupported_reason": "source-backed selector unavailable",
    }
    records = _unsupported_records((unsupported, _row(1, 1.0)))
    assert records == [unsupported]
    assert _unsupported_summaries(records) == [{
        "hardware_id": "metax_c500", "family": "fused_moe",
        "reason": "source-backed selector unavailable", "count": 1,
    }]


def test_gemm_m_domain_uses_only_exact_canonical_launch_data():
    values = _gemm_launch_m_values(
        "ascend_910b", "bf16", 1, 19456, 2560)
    assert values[0] == 1
    assert values[-1] == 8192
    assert 4096 in values
    assert _gemm_launch_m_values(
        "ascend_910b", "bf16", 1, 6144, 2048) == ()


def test_gemm_scan_plan_selects_injected_launch_for_exact_domain_only():
    plan = build_scan_plan(
        hardware_ids=("ascend_910b",), families=("gemm",),
        model_names=("qwen3-4b",), scan_size="smoke")
    assert plan["cases"]
    for case in plan["cases"]:
        attributes = case["recipe"]["selection_attributes"]
        assert attributes["launch_policy"] == "injected_launch"
        assert attributes["a_layout"] == "row_major"
        assert attributes["b_layout"] == "row_major"
        assert attributes["c_layout"] == "row_major"
        assert attributes["trans_a"] is False
        assert attributes["trans_b"] is True
        assert attributes["epilogue"] == "none"
        assert attributes["has_bias"] is False


def test_ascend_injected_gemm_descriptor_boundary_keeps_physical_direction():
    plan = build_scan_plan(
        hardware_ids=("ascend_910b",), families=("gemm",),
        model_names=("qwen3-4b",), scan_size="full")
    cases = _matching_cases(
        plan,
        lambda case: (
            case["recipe"]["shape"].get("k") == 2560
            and case["recipe"]["shape"].get("n") == 19456
            and case["axis_value"] in (480, 488)),
    )
    assert [case["axis_value"] for case in cases] == [480, 488]
    rows = [_predict_case(case)[0] for case in cases]
    assert [row["status"] for row in rows] == ["success", "success"]
    assert rows[1]["prediction"]["total_ns"] > rows[0]["prediction"]["total_ns"]


def test_ascend_low_q_split_fuse_fails_closed_for_unmeasured_handoff_span():
    plan = build_scan_plan(
        hardware_ids=("ascend_910b",), families=("flash_attn",),
        model_names=("qwen3-30b-a3b",), scan_size="full")
    cases = _matching_cases(
        plan,
        lambda case: (
            case["controls"].get("variant") == "prefill"
            and case["axis_value"] == 2),
    )
    assert len(cases) == 1
    row = _predict_case(cases[0])[0]
    assert row["status"] == "success"
    assert row["prediction"]["support_level"] == "partial"
    assert "cann_split_fuse_handoff_span_domain_unavailable" in (
        row["prediction"]["limitations"])


def test_full_extent_source_includes_required_m_values_and_smoke_is_bounded():
    full = _extents("full", (24, 48, 512))
    assert set((1, 17, 33, 65, 129, 257, 4096, 24, 48)) <= set(full)
    assert _extents("smoke", (24, 48, 512)) == (1, 2, 16, 17, 64, 65, 128)


def test_known_capacity_domain_error_is_unsupported_but_other_value_errors_fail():
    assert _is_fail_closed_unsupported(
        ValueError("CANN GEMM output bytes exhaust the operand DRAM path"))
    assert not _is_fail_closed_unsupported(ValueError("unexpected implementation defect"))


def test_canonical_trend_labels_preserve_direction_and_plateau():
    assert _trend(1.0) == "increases"
    assert _trend(-1.0) == "decreases"
    assert _trend(0.0) == "plateaus"


def test_exact_canonical_gemm_points_are_reported_only_at_evaluation_boundary():
    before = _row(1, 10.0)
    after = _row(2048, 20.0)
    for row in (before, after):
        row["hardware_id"] = "metax_c500"
        row["kernel_spec"] = {
            "kind": "GEMM", "dtype": "bf16",
            "shape": {"b": 1, "m": row["axis_value"], "k": 4096, "n": 4096},
        }
    pairs = _canonical_gemm_pairs((before, after))
    assert len(pairs) == 1
    assert pairs[0]["canonical_trend"] == "increases"
    assert pairs[0]["before"]["canonical"]["case_id"] == "GEMM_B1_M1_K4096_N4096"
