import copy
import json
import math
from pathlib import Path

import pytest

import evaluation.kernel.evaluation_runner as evaluation_runner
import evaluation.kernel.observation_normalizer as observation_normalizer
from evaluation.kernel.dataset import (
    get_hardware_config_path,
    kernel_dataset_l2_execution_context,
    load_kernel_dataset,
    parse_kernel_evaluation_timing,
)
from evaluation.kernel.evaluation_runner import (
    PIPEWEAVE_OUT_OF_DOMAIN,
    _process_op,
    compute_keyed_tail_metrics,
    compute_metrics,
    define_subsets,
    predict_forward_op_total,
)
from evaluation.kernel.fa_serving_coverage_report import (
    build_fa_serving_specs,
    validate_fa_serving_capture_record,
)
from evaluation.kernel.observation_normalizer import normalize_kernel_spec
from src.kernel.hardware_spec import HWSpec
from src.kernel.dataset_storage import load_kernel_dataset_document
from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.predictors.gemm import GEMMPredictor, _model as build_gemm_resource_input
from src.kernel.predictors.vector import VectorPredictor
from src.kernel.predictors.vector import forward_analytical_resource_input
from src.kernel.predictors.flash_attn import FlashAttentionPredictor, _model as build_flash_attention_input
from src.kernel.predictors.common import prediction_from_ns, roofline_ns


def _attention_meta(
        kv_length, executed_splits, *, query_length=1,
        layout="contiguous", max_launch_waves=1):
    return {
        "op_type": "FLASHATTN",
        "query_length": query_length,
        "kv_length": kv_length,
        "executed_attention_splits": executed_splits,
        "attention_layout": layout,
        "max_launch_waves": max_launch_waves,
    }


def test_metric_summary_reports_distribution_tail():
    metrics = compute_metrics([(90.0, 100.0), (120.0, 100.0)])

    assert metrics == {
        "status": "ok",
        "N": 2,
        "scored": 2,
        "unsupported": 0,
        "out_of_domain": 0,
        "MAPE": pytest.approx(15.0),
        "MedAPE": pytest.approx(15.0),
        "P90APE": pytest.approx(19.0),
        "MaxAPE": pytest.approx(20.0),
        "WMAE": pytest.approx(15.0),
    }


def test_out_of_domain_cases_are_excluded_and_counted():
    """A method scored on a subset must report the subset, never imply N."""

    metrics = compute_metrics(
        [(90.0, 100.0), PIPEWEAVE_OUT_OF_DOMAIN, (120.0, 100.0)])

    assert metrics["status"] == "ok"
    assert metrics["N"] == 3
    assert metrics["scored"] == 2
    assert metrics["out_of_domain"] == 1
    assert metrics["MAPE"] == pytest.approx(15.0)


def test_a_subset_with_nothing_left_to_score_reports_no_error():
    metrics = compute_metrics([PIPEWEAVE_OUT_OF_DOMAIN] * 3)

    assert metrics == {
        "status": "out_of_domain",
        "N": 3,
        "out_of_domain": 3,
    }


def test_a_hard_error_still_voids_the_subset():
    """Out-of-domain is not a way to launder a predictor that raised."""

    metrics = compute_metrics([(90.0, 100.0), "ERR", PIPEWEAVE_OUT_OF_DOMAIN])

    assert metrics["status"] == "unsupported"
    assert metrics["unsupported"] == 1


def test_forward_tables_exclude_and_count_explicitly_unsupported_cases():
    pairs = [(90.0, 100.0), "ERR", (120.0, 100.0)]

    metrics = compute_metrics(pairs, exclude_unsupported=True)

    assert metrics["status"] == "ok"
    assert (metrics["N"], metrics["scored"], metrics["unsupported"]) == (
        3, 2, 1)
    assert metrics["MAPE"] == pytest.approx(15.0)
    assert compute_metrics(pairs)["status"] == "unsupported"
    assert compute_metrics(
        ["ERR", "ERR"], exclude_unsupported=True)["status"] == "unsupported"
    tail = compute_keyed_tail_metrics(
        {"a": (90.0, 100.0), "b": "ERR"}, exclude_unsupported=True)
    assert tail["worst_key"] == "a"


def test_keyed_tail_summary_reports_deterministic_worst_case():
    metrics = compute_keyed_tail_metrics({
        "case_b": (80.0, 100.0),
        "case_a": (120.0, 100.0),
        "case_c": (95.0, 100.0),
    })

    assert metrics["P90APE"] == pytest.approx(20.0)
    assert metrics["MaxAPE"] == pytest.approx(20.0)
    assert metrics["worst_key"] == "case_a"


def _timing_result(task_key="case"):
    return (
        task_key, {}, None, (1.0, 1.0), None, None, [],
    )


def _timing_artifact(samples=(1.0,), hardware_config_sha256="config-a"):
    return evaluation_runner._build_forward_timing_artifact(
        hardware_id="nvidia_a800",
        jobs=1,
        case_count=1,
        dataset_source="kernel_dataset/nvidia_a800",
        dataset_sha256="dataset-a",
        hardware_config_source="configs/hardware/nvidia_a800.yaml",
        hardware_config_sha256=hardware_config_sha256,
        warmups=1,
        samples=list(samples),
        forward_error_count=0,
    )


def test_forward_timing_measures_only_case_evaluation(monkeypatch):
    calls = []

    def process(packed, jobs, hardware_id):
        calls.append((packed, jobs, hardware_id))
        return [_timing_result()]

    clock = iter((10.0, 10.2, 20.0, 20.4))
    monkeypatch.setattr(
        evaluation_runner, "process_kernel_evaluation_cases", process)
    monkeypatch.setattr(
        evaluation_runner.time, "perf_counter", lambda: next(clock))

    results, samples = evaluation_runner._run_forward_timing(
        [("packed",)], 1, "nvidia_a800", warmups=1, repetitions=2)

    assert results == [_timing_result()]
    assert samples == pytest.approx([0.2, 0.4])
    assert calls == [
        ([("packed",)], 1, "nvidia_a800"),
        ([("packed",)], 1, "nvidia_a800"),
        ([("packed",)], 1, "nvidia_a800"),
    ]


def test_case_timing_records_one_executor_task(monkeypatch):
    result = _timing_result("case-a")[:-1] + (["first", "second"],)
    monkeypatch.setattr(
        evaluation_runner, "_process_op_star", lambda packed: result)
    monkeypatch.setattr(evaluation_runner.os, "getpid", lambda: 4123)
    clock = iter((10.0, 10.25))
    monkeypatch.setattr(
        evaluation_runner.time, "perf_counter", lambda: next(clock))

    results, case_timing = evaluation_runner.process_kernel_evaluation_cases_with_timing(
        [("case-a", {"op_info": {"Type": "NORM"}})],
        1, "nvidia_a800")

    assert results == [result]
    assert case_timing == [{
        "case_id": "case-a",
        "worker_pid": 4123,
        "elapsed_seconds": pytest.approx(0.25),
        "kernel_prediction_count": 2,
    }]


def test_semantic_artifact_preserves_every_case_result_without_timing():
    artifact = evaluation_runner._build_semantic_artifact(
        hardware_id="nvidia_a800",
        mode="standard",
        jobs=45,
        dataset_source="kernel_dataset/nvidia_a800",
        dataset_sha256="dataset-a",
        hardware_config_source="configs/hardware/nvidia_a800.yaml",
        hardware_config_sha256="hardware-a",
        results=[(
            "case-a",
            {"op_type": "NORM"},
            (10.0, 11.0),
            "ERR",
            None,
            None,
            [{"task_key": "case-a", "forward_unavailable_reason": "none"}],
        )],
    )

    assert artifact["scope"] == "kernel_evaluation_semantic_result"
    assert artifact["case_count"] == 1
    assert artifact["cases"] == [{
        "case_id": "case-a",
        "report_meta": {"op_type": "NORM"},
        "oracle": (10.0, 11.0),
        "forward": "ERR",
        "roofline": None,
        "pipeweave": None,
        "kernel_results": [
            {"task_key": "case-a", "forward_unavailable_reason": "none"},
        ],
    }]


def test_case_workers_preload_gemm_selector_before_execution(monkeypatch):
    calls = []
    predictor_kinds = []
    monkeypatch.setattr(
        evaluation_runner, "preload_gemm_launch_selection",
        lambda hardware_id: calls.append(hardware_id))
    monkeypatch.setattr(
        evaluation_runner, "create_predictor",
        lambda kind: predictor_kinds.append(kind))
    monkeypatch.setattr(
        evaluation_runner, "_process_op_star",
        lambda packed: _timing_result("case-a"))

    results = evaluation_runner.process_kernel_evaluation_cases(
        [("case-a", {"op_info": {"Type": "GEMM"}})],
        1, "ascend_910b")

    assert results == [_timing_result("case-a")]
    assert calls == ["ascend_910b"]
    assert predictor_kinds == [KernelKind.GEMM, KernelKind.VECTOR]


def test_case_timing_artifact_records_work_and_worker_mapping():
    artifact = evaluation_runner._build_case_timing_artifact(
        hardware_id="nvidia_a800",
        mode="standard",
        jobs=45,
        dataset_source="kernel_dataset/nvidia_a800",
        dataset_sha256="dataset-a",
        hardware_config_source="configs/hardware/nvidia_a800.yaml",
        hardware_config_sha256="config-a",
        case_timing=[{
            "case_id": "case-a",
            "worker_pid": 4123,
            "elapsed_seconds": 0.25,
            "kernel_prediction_count": 2,
        }],
    )

    assert artifact["scope"] == "kernel_case_evaluation_timing"
    assert artifact["jobs"] == 45
    assert artifact["timing"] == {
        "timed_region": "_process_op",
        "excluded_phases": [
            "dataset_loading",
            "hardware_spec_loading",
            "case_packing",
            "executor_queue_wait",
            "metric_aggregation",
            "table_rendering",
            "counterfactual_evaluation",
            "plotting",
        ],
        "worker_count": 1,
        "total_case_execution_seconds": pytest.approx(0.25),
    }
    assert artifact["work_count"] == {"kernel_prediction_count": 2}


def test_forward_timing_artifact_has_exact_scope_and_median():
    artifact = _timing_artifact(samples=(0.3, 0.1, 0.2))

    assert set(artifact) == {
        "schema_version", "scope", "hardware_id", "mode", "jobs",
        "case_count", "dataset", "hardware_config", "timing",
        "forward_error_count", "gate",
    }
    assert artifact["scope"] == "kernel_forward_evaluation"
    assert artifact["mode"] == "forward"
    assert artifact["timing"] == {
        "warmup_runs": 1,
        "measured_runs": 3,
        "elapsed_seconds": [0.3, 0.1, 0.2],
        "median_seconds": pytest.approx(0.2),
        "timed_region": "process_kernel_evaluation_cases",
        "excluded_phases": [
            "dataset_loading",
            "hardware_spec_loading",
            "case_packing",
            "metric_aggregation",
            "table_rendering",
            "counterfactual_evaluation",
            "plotting",
        ],
    }


def test_forward_timing_gate_uses_jobs_one_median_ratio():
    baseline = _timing_artifact(samples=(1.0,), hardware_config_sha256="old")
    current = _timing_artifact(samples=(1.05,), hardware_config_sha256="new")

    gate = evaluation_runner._forward_timing_gate(
        current, baseline, "baseline.json", 1.10)

    assert gate == {
        "baseline_path": "baseline.json",
        "baseline_median_seconds": 1.0,
        "max_ratio": 1.10,
        "ratio": pytest.approx(1.05),
        "passed": True,
    }


def test_forward_timing_gate_fails_loudly_after_artifact_is_writable(tmp_path):
    baseline = _timing_artifact(samples=(1.0,))
    current = _timing_artifact(samples=(1.2,))
    current["gate"] = evaluation_runner._forward_timing_gate(
        current, baseline, "baseline.json", 1.10)
    output = tmp_path / "timing.json"

    evaluation_runner._write_forward_timing_artifact(output, current)
    with pytest.raises(RuntimeError, match="Forward timing gate failed"):
        evaluation_runner._require_forward_timing_gate_passed(current)

    written = json.loads(output.read_text())
    assert written["gate"]["passed"] is False
    assert written["gate"]["ratio"] == pytest.approx(1.2)


def test_forward_timing_gate_rejects_dataset_identity_mismatch():
    baseline = _timing_artifact()
    current = _timing_artifact()
    current["dataset"]["content_sha256"] = "dataset-b"

    with pytest.raises(ValueError, match="dataset.content_sha256"):
        evaluation_runner._forward_timing_gate(
            current, baseline, "baseline.json", 1.10)


def test_forward_timing_cli_requires_forward_and_jobs_one_for_gate():
    with pytest.raises(ValueError, match="requires --mode forward"):
        evaluation_runner._validate_forward_timing_cli(
            "standard", Path("current.json"), None, 1)
    with pytest.raises(ValueError, match="requires --jobs 1"):
        evaluation_runner._validate_forward_timing_cli(
            "forward", Path("current.json"), Path("baseline.json"), 8)
    with pytest.raises(ValueError, match="requires --forward-timing-output"):
        evaluation_runner._validate_forward_timing_cli(
            "forward", None, Path("baseline.json"), 1)


def test_public_forward_op_adapter_reuses_production_evaluation_path():
    hardware_id = "nvidia_a800"
    case_id = "VECTOR_RMSNORM_T1_H2048_B1"
    entry = load_kernel_dataset(hardware_id, ["vector"])[case_id]

    result = predict_forward_op_total(
        task_key=case_id, entry=entry, hardware_id=hardware_id)

    assert result["status"] == "supported"
    assert result["prediction_ns"] > 0.0
    assert result["canonical_dataset_total_ns"] > 0.0


def test_oracle_predictor_unsupported_is_recorded_per_kernel(monkeypatch):
    hardware_id = "nvidia_a800"
    task_key = "GEMM_B1_M1_K13824_N5120"
    entry = _target_invariance_entry(hardware_id, "gemm", task_key)
    production_predict = evaluation_runner.predict_production

    def reject_oracle(kernel_spec, hardware, method="midlayer", **kwargs):
        if method == "oracle":
            raise NotImplementedError("synthetic unsupported Oracle")
        return production_predict(
            kernel_spec, hardware, method=method, **kwargs)

    monkeypatch.setattr(
        evaluation_runner, "predict_production", reject_oracle)

    result = _process_op(
        task_key, entry, get_hardware_config_path(hardware_id), True, True)

    assert result[2] == "ERR"
    assert result[3] != "ERR"
    assert result[-1]
    for row in result[-1]:
        assert "pred_oracle_ns" not in row
        assert row["oracle_unavailable_reason"] == (
            "synthetic unsupported Oracle")


_TARGET_INVARIANCE_CASES = (
    pytest.param(
        "nvidia_a800", "gemm", "GEMM_B1_M1_K13824_N5120",
        id="nvidia_a800-GEMM"),
    pytest.param(
        "nvidia_a800", "fused_moe",
        "FUSEDMOE_TOK1_E8_TOPK2_H4096_INT14336",
        id="nvidia_a800-FUSED_MOE"),
    pytest.param(
        "nvidia_a800", "flash_attn",
        "FLASHATTN_B1_S512_HQ32_HKV8_D128_C0",
        id="nvidia_a800-FLASH_ATTN"),
    pytest.param(
        "nvidia_a800", "vector", "VECTOR_RMSNORM_T1_H2048_B1",
        id="nvidia_a800-VECTOR"),
    pytest.param(
        "ascend_910b", "gemm", "GEMM_B1_M1_K13824_N5120",
        id="ascend_910b-GEMM"),
    pytest.param(
        "ascend_910b", "fused_moe",
        "FUSEDMOE_TOK1_E8_TOPK2_H4096_INT14336",
        id="ascend_910b-FUSED_MOE"),
    pytest.param(
        "ascend_910b", "flash_attn",
        "FLASHATTN_B1_S512_HQ32_HKV8_D128_C0",
        id="ascend_910b-FLASH_ATTN"),
    pytest.param(
        "ascend_910b", "vector", "VECTOR_RMSNORM_T1_H2048_B1",
        id="ascend_910b-VECTOR"),
    pytest.param(
        "metax_c500", "gemm", "GEMM_B1_M1_K13824_N5120",
        id="metax_c500-GEMM"),
    pytest.param(
        "metax_c500", "fused_moe",
        "FUSEDMOE_TOK1_E8_TOPK2_H4096_INT14336",
        id="metax_c500-FUSED_MOE"),
    pytest.param(
        "metax_c500", "flash_attn",
        "FLASHATTN_B1_S512_HQ32_HKV8_D128_C0",
        id="metax_c500-FLASH_ATTN"),
    pytest.param(
        "metax_c500", "vector", "VECTOR_RMSNORM_T1_H2048_B1",
        id="metax_c500-VECTOR"),
)


@pytest.mark.parametrize(
    "hardware_id,family,task_key", _TARGET_INVARIANCE_CASES)
def test_target_forward_exposes_structured_diagnostics(
        hardware_id, family, task_key, monkeypatch):
    recorded = []
    production_predict = evaluation_runner.predict_production

    def record_prediction(kernel_spec, hardware, method="midlayer", **kwargs):
        prediction = production_predict(
            kernel_spec, hardware, method=method, **kwargs)
        if method == "midlayer":
            recorded.append((kernel_spec.kind, prediction))
        return prediction

    monkeypatch.setattr(
        evaluation_runner, "predict_production", record_prediction)
    entry = _target_invariance_entry(hardware_id, family, task_key)
    _process_op(
        task_key, entry, get_hardware_config_path(hardware_id),
        False, True)

    expected_kind = {
        "gemm": KernelKind.GEMM,
        "vector": KernelKind.VECTOR,
        "flash_attn": KernelKind.FLASH_ATTN,
        "fused_moe": KernelKind.FUSED_MOE,
    }[family]
    predictions = [
        prediction for kind, prediction in recorded
        if kind is expected_kind
    ]
    assert predictions
    if ((hardware_id == "nvidia_a800"
         and family in {"gemm", "flash_attn"})
            or (hardware_id == "metax_c500"
                and family in {"gemm", "flash_attn"})
            or (hardware_id == "ascend_910b"
                and family in {
                    "gemm", "vector", "flash_attn", "fused_moe"})):
        expected_support = "partial"
    elif (family in {"vector", "flash_attn", "fused_moe"}
          or hardware_id == "ascend_910b"
          or (hardware_id == "metax_c500" and family == "gemm")):
        expected_support = "aggregate_only"
    else:
        raise AssertionError(f"unexpected target hardware {hardware_id!r}")
    expect_pipeline = (
        (expected_support == "partial"
         and not (hardware_id == "metax_c500" and family == "gemm"))
        or (hardware_id == "metax_c500" and family == "vector")
        or (hardware_id == "nvidia_a800" and family == "vector")
        or (hardware_id == "nvidia_a800" and family == "fused_moe"))
    for prediction in predictions:
        diagnostics = prediction.diagnostics
        assert diagnostics is not None
        assert diagnostics.support_level == expected_support
        assert diagnostics.limitations
        assert diagnostics.memory
        assert diagnostics.critical_path
        for memory in diagnostics.memory:
            assert memory.lambda_initial_per_s > 0.0
            assert memory.lambda_capacity_per_s > 0.0
            assert memory.nodes
            if hardware_id == "ascend_910b":
                assert memory.rate_unit == "bytes"
                assert memory.initial_transactions is None
                assert memory.first_request_latency_s is None
            else:
                assert memory.rate_unit == "transactions"
                assert memory.initial_transactions > 0.0
                assert memory.first_request_latency_s is not None
            for node in memory.nodes:
                assert node.bytes > 0.0
                assert node.service_capacity_per_s > 0.0
                assert 0.0 < node.rho <= 1.0
                assert node.capacity_time_s > 0.0
                if hardware_id == "ascend_910b":
                    assert node.transactions is None
                    assert node.request_capacity_per_s is None
                    assert node.queue_wait_s == 0.0
                elif node.saturated:
                    assert node.transactions > 0.0
                    assert node.queue_wait_s is None
                else:
                    assert node.transactions > 0.0
                    assert node.queue_wait_s is not None
        if expect_pipeline:
            assert diagnostics.pipeline is not None
            assert diagnostics.pipeline.execution_time_s > 0.0
            assert diagnostics.pipeline.worker_makespan_s
            assert diagnostics.pipeline.stage_service_s
            assert diagnostics.pipeline.critical_path
        else:
            assert diagnostics.pipeline is None


@pytest.mark.parametrize(
    "hardware_id,expected_cases,expected_kernels,expected_sources",
    (
        ("nvidia_a800", 1027, 2330,
         {"nsys_nvtx_kernel_median"}),
        ("metax_c500", 1027, 2634,
         {"device_event_op_median"}),
        ("ascend_910b", 1027, 2600,
         {"msprof_timeline_op_median"}),
        ("samsung_hbm_pim", 999, 999,
         {"unindp_simulator_latency"}),
        ("groq_lpu", 557, 557,
         {"timeloop_lpu_scaled_hybrid_latency"}),
        ("ucb_gemmini", 703, 703,
         {"timeloop_mapper_simulated_latency"}),
    ),
)
def test_canonical_production_datasets_satisfy_timing_contract(
        hardware_id, expected_cases, expected_kernels, expected_sources):
    case_count = 0
    kernel_count = 0
    sources = set()
    dataset_dir = Path("kernel_dataset") / hardware_id
    paths = sorted(dataset_dir.glob("*.json.zst"))
    for path in paths:
        payload = load_kernel_dataset_document(path)
        for entry in payload.values():
            if "capture_evidence" in entry:
                validate_fa_serving_capture_record(entry, hardware_id)
                sources.add(
                    entry["op_info"]["evaluation"]["timing"]["source"])
                kernel_count += len(build_fa_serving_specs(
                    entry["op_info"]["Attr"], hardware_id,
                    HWSpec(get_hardware_config_path(hardware_id))))
                case_count += 1
                continue
            kernels = _kernel_records(entry)
            timing = parse_kernel_evaluation_timing(
                entry, kernels, hardware_id)
            sources.add(timing.source)
            case_count += 1
            kernel_count += len(kernels)

    assert case_count == expected_cases
    assert kernel_count == expected_kernels
    assert sources == expected_sources


def test_canonical_ascend_flash_attention_retains_variant_and_task_ratio():
    payload = load_kernel_dataset_document(
        "kernel_dataset/ascend_910b/flash_attn.json.zst")
    expected_variant_counts = {
        "103000000010022221": 55,
        "103000000010222221": 40,
        "103000000010422221": 33,
        "103000000010522221": 12,
        "103000000010622221": 44,
        "105000000000222220": 5,
        "105000000000322220": 11,
        "5000000000000200203": 31,
        "5000000000010200203": 63,
    }
    variant_counts = {
        variant: 0 for variant in expected_variant_counts}
    symbol_prefix = (
        "FusedInferAttentionScore_3b093497fc536d61a77a7a3293a524da_")
    symbol_suffix = "_mix_aic"

    for entry in payload.values():
        if "capture_evidence" in entry:
            continue
        kernels = _kernel_records(entry)
        assert len(kernels) == 1
        kernel = kernels[0]
        symbol = kernel["Metadata"]["Kernel Name"]
        assert symbol.startswith(symbol_prefix)
        assert symbol.endswith(symbol_suffix)
        runtime_variant = symbol[len(symbol_prefix):-len(symbol_suffix)]
        assert runtime_variant in expected_variant_counts
        variant_counts[runtime_variant] += 1

        ascend = kernel["Ascend"]
        assert ascend["block_dim"] > 0
        assert ascend["mix_block_dim"] == 2 * ascend["block_dim"]
        assert kernel["Metadata"]["Op Type"] == "mix"

    assert variant_counts == expected_variant_counts


def test_c500_split_k_uses_fp32_partial_and_exact_reducer_launch():
    task_key = "GEMM_B2_M1_K4096_N4096"
    entry = _target_invariance_entry("metax_c500", "gemm", task_key)

    result = _process_op(
        task_key, entry, get_hardware_config_path("metax_c500"),
        False, True, True)
    _, metadata, _, forward_pair, roofline_pair, _, kernel_results = result

    assert metadata["split"] is True
    assert forward_pair != "ERR"
    assert roofline_pair != "ERR"
    assert len(kernel_results) == 2
    assert all(
        "forward_unavailable_reason" not in row
        and "roofline_unavailable_reason" not in row
        for row in kernel_results)


def _kernel_records(entry):
    kernels = entry["kernels"]
    if isinstance(kernels, dict):
        return list(kernels.values())
    elif isinstance(kernels, list):
        return kernels
    else:
        raise TypeError("test dataset kernels must be a mapping or list")


@pytest.mark.parametrize(
    "hardware_id", ("nvidia_a800", "metax_c500", "ascend_910b"))
def test_canonical_fa_serving_capture_uses_production_forward(hardware_id):
    dataset = load_kernel_dataset(hardware_id, ["flash_attn"])
    task_key, entry = next(
        (key, value) for key, value in sorted(dataset.items())
        if "capture_evidence" in value)

    result = _process_op(
        task_key, entry, get_hardware_config_path(hardware_id),
        True, True, True)
    _, metadata, oracle_pair, forward_pair, roofline_pair, _, rows = result

    assert metadata["actual_granularity"] == "op"
    assert oracle_pair == "ERR"
    assert forward_pair != "ERR"
    assert roofline_pair != "ERR"
    assert rows
    assert all(row["actual_ns"] is None for row in rows)


def _with_a800_timing(entry):
    # Synthetic timing is sufficient for runner contract tests; the canonical
    # dataset is never rewritten or treated as the source of this fixture.
    timed = copy.deepcopy(entry)
    timing_kernels = [
        {
            "kernel_idx": kernel_idx,
            "kernel_name": kernel["Metadata"]["Kernel Name"],
            "duration_ns": float(1000 + kernel_idx),
        }
        for kernel_idx, kernel in enumerate(_kernel_records(timed))
    ]
    timed["op_info"]["evaluation"] = {
        "timing": {
            "granularity": "kernel",
            "source": "nsys_nvtx_kernel_median",
            "total_duration_ns": math.fsum(
                target["duration_ns"] for target in timing_kernels),
            "kernels": timing_kernels,
        },
    }
    return timed


def _target_invariance_entry(hardware_id, family, task_key):
    path = Path(f"kernel_dataset/{hardware_id}/{family}.json.zst")
    entry = copy.deepcopy(load_kernel_dataset_document(path)[task_key])
    if hardware_id not in {"nvidia_a800", "metax_c500", "ascend_910b"}:
        raise ValueError(
            f"unsupported target-invariance hardware: {hardware_id!r}")
    return entry


def _change_evaluation_target(
        entry, hardware_id, baseline_kernel_rows, factor):
    changed = copy.deepcopy(entry)
    if hardware_id not in {"nvidia_a800", "metax_c500", "ascend_910b"}:
        raise ValueError(
            f"unsupported target-invariance hardware: {hardware_id!r}")
    timing = changed["op_info"]["evaluation"]["timing"]
    if timing["granularity"] == "kernel":
        targets = timing["kernels"]
        if len(targets) != len(baseline_kernel_rows):
            raise ValueError("test fixture timing count mismatch")
        for target, row in zip(targets, baseline_kernel_rows, strict=True):
            target["duration_ns"] = row["actual_ns"] * factor
        timing["total_duration_ns"] = math.fsum(
            target["duration_ns"] for target in targets)
    elif timing["granularity"] == "op":
        timing["total_duration_ns"] *= factor
    else:
        raise ValueError("test fixture has unsupported timing granularity")
    return changed


def test_attention_report_uses_executed_split_and_kv_range_buckets():
    metadata = {
        "FLASHATTN_B1_S512": _attention_meta(512, 1),
        "FLASHATTN_PAGED_A": _attention_meta(1000, 2),
        "FLASHATTN_PAGED_B": _attention_meta(2048, 4),
        "FLASHATTN_PAGED_C": _attention_meta(4093, 6),
        "FLASHATTN_PAGED_D": _attention_meta(8192, 8),
        "FLASHATTN_PAGED_E": _attention_meta(16384, 11),
        "FLASHATTN_PAGED_F": _attention_meta(32768, 22),
        # CANN does not expose an executed split count. It still belongs to a
        # KV-length range, but must not be fabricated into the SP=1 bucket.
        "FLASHATTN_PAGED_CANN": _attention_meta(1024, None),
    }

    rows = {
        label: keys
        for label, keys, _indent, _is_last in define_subsets(metadata)
    }

    assert rows["SP=1"] == ["FLASHATTN_B1_S512"]
    assert rows["SP=2–4"] == [
        "FLASHATTN_PAGED_A", "FLASHATTN_PAGED_B"]
    assert rows["SP=5–8"] == [
        "FLASHATTN_PAGED_C", "FLASHATTN_PAGED_D"]
    assert rows["SP≥9"] == [
        "FLASHATTN_PAGED_E", "FLASHATTN_PAGED_F"]
    assert "FLASHATTN_PAGED_CANN" not in {
        key
        for label in ("SP=1", "SP=2–4", "SP=5–8", "SP≥9")
        for key in rows[label]
    }

    assert rows["short KV (≤512)"] == ["FLASHATTN_B1_S512"]
    assert rows["medium KV (513–2048)"] == [
        "FLASHATTN_PAGED_A", "FLASHATTN_PAGED_B",
        "FLASHATTN_PAGED_CANN"]
    assert rows["long KV (2049–8192)"] == [
        "FLASHATTN_PAGED_C", "FLASHATTN_PAGED_D"]
    assert rows["very-long KV (≥8193)"] == [
        "FLASHATTN_PAGED_E", "FLASHATTN_PAGED_F"]
    assert not any(label.startswith("S=") for label in rows)


def test_shape_layout_and_launch_wave_report_buckets():
    metadata = {
        "GEMM_LOW": {
            "op_type": "GEMM", "eqm": 16, "split": False,
            "gemm_m": 16, "gemm_n": 4096, "gemm_k": 8192,
            "max_launch_waves": 1,
        },
        "GEMM_HIGH": {
            "op_type": "GEMM", "eqm": 2048, "split": True,
            "gemm_m": 2048, "gemm_n": 16385, "gemm_k": 16385,
            "max_launch_waves": 3,
        },
        "MOE_LOW": {
            "op_type": "FUSEDMOE", "eqm": 16, "moe_experts": 32,
            "moe_max_tokens_per_expert": 16, "moe_topk": 4,
            "max_launch_waves": 16,
        },
        "MOE_HIGH": {
            "op_type": "UNFUSEDMOE", "eqm": 2048,
            "moe_experts": 257, "moe_max_tokens_per_expert": 257,
            "moe_topk": 9, "max_launch_waves": 17,
        },
        "ATTN_SHORT": _attention_meta(
            512, 1, query_length=256, layout="paged",
            max_launch_waves=4),
        "ATTN_LONG": _attention_meta(
            16384, None, query_length=4096, layout="mla",
            max_launch_waves=5),
        "VECTOR_LOW": {
            "op_type": "NORM", "vector_tokens": 16,
            "max_launch_waves": 1,
        },
        "VECTOR_HIGH": {
            "op_type": "ELEMENTWISE", "vector_tokens": 2048,
            "max_launch_waves": 17,
        },
    }

    rows = {
        label: keys
        for label, keys, _indent, _is_last in define_subsets(metadata)
    }

    assert rows["N≤4096"] == ["GEMM_LOW"]
    assert rows["N≥16385"] == ["GEMM_HIGH"]
    assert rows["4097≤K≤8192"] == ["GEMM_LOW"]
    assert rows["K≥16385"] == ["GEMM_HIGH"]
    assert rows["experts≤32"] == ["MOE_LOW"]
    assert rows["experts≥257"] == ["MOE_HIGH"]
    assert rows["max tokens/expert≤16"] == ["MOE_LOW"]
    assert rows["max tokens/expert≥257"] == ["MOE_HIGH"]
    assert rows["top-k≤4"] == ["MOE_LOW"]
    assert rows["top-k≥9"] == ["MOE_HIGH"]
    assert rows["short Q (2–512)"] == ["ATTN_SHORT"]
    assert rows["long Q (≥2049)"] == ["ATTN_LONG"]
    assert rows["layout=paged"] == ["ATTN_SHORT"]
    assert rows["layout=mla"] == ["ATTN_LONG"]
    assert rows["tokens≤16"] == ["VECTOR_LOW"]
    assert rows["tokens≥2048"] == ["VECTOR_HIGH"]
    assert rows["1 wave"] == ["GEMM_LOW", "VECTOR_LOW"]
    assert rows["2–4 waves"] == ["GEMM_HIGH", "ATTN_SHORT"]
    assert rows["5–16 waves"] == ["MOE_LOW", "ATTN_LONG"]
    assert rows["≥17 waves"] == ["MOE_HIGH", "VECTOR_HIGH"]


@pytest.mark.parametrize(
    "family,field,value,label",
    (
        ("gemm", "gemm_m", 16, "tiny (B×M≤16)"),
        ("gemm", "gemm_m", 17, "small (17≤B×M≤64)"),
        ("gemm", "gemm_m", 64, "small (17≤B×M≤64)"),
        ("gemm", "gemm_m", 65, "med (65≤B×M≤2047)"),
        ("gemm", "gemm_m", 2047, "med (65≤B×M≤2047)"),
        ("gemm", "gemm_m", 2048, "prefill (B×M≥2048)"),
        ("gemm", "gemm_n", 4096, "N≤4096"),
        ("gemm", "gemm_n", 4097, "4097≤N≤8192"),
        ("gemm", "gemm_n", 8192, "4097≤N≤8192"),
        ("gemm", "gemm_n", 8193, "8193≤N≤16384"),
        ("gemm", "gemm_n", 16384, "8193≤N≤16384"),
        ("gemm", "gemm_n", 16385, "N≥16385"),
        ("gemm", "gemm_k", 4096, "K≤4096"),
        ("gemm", "gemm_k", 4097, "4097≤K≤8192"),
        ("gemm", "gemm_k", 8192, "4097≤K≤8192"),
        ("gemm", "gemm_k", 8193, "8193≤K≤16384"),
        ("gemm", "gemm_k", 16384, "8193≤K≤16384"),
        ("gemm", "gemm_k", 16385, "K≥16385"),
        ("moe", "moe_experts", 32, "experts≤32"),
        ("moe", "moe_experts", 33, "33≤experts≤128"),
        ("moe", "moe_experts", 128, "33≤experts≤128"),
        ("moe", "moe_experts", 129, "129≤experts≤256"),
        ("moe", "moe_experts", 256, "129≤experts≤256"),
        ("moe", "moe_experts", 257, "experts≥257"),
        ("moe", "moe_max_tokens_per_expert", 16,
         "max tokens/expert≤16"),
        ("moe", "moe_max_tokens_per_expert", 17,
         "17≤max tokens/expert≤64"),
        ("moe", "moe_max_tokens_per_expert", 64,
         "17≤max tokens/expert≤64"),
        ("moe", "moe_max_tokens_per_expert", 65,
         "65≤max tokens/expert≤256"),
        ("moe", "moe_max_tokens_per_expert", 256,
         "65≤max tokens/expert≤256"),
        ("moe", "moe_max_tokens_per_expert", 257,
         "max tokens/expert≥257"),
        ("moe", "moe_topk", 4, "top-k≤4"),
        ("moe", "moe_topk", 5, "5≤top-k≤8"),
        ("moe", "moe_topk", 8, "5≤top-k≤8"),
        ("moe", "moe_topk", 9, "top-k≥9"),
        ("attention", "query_length", 1, "Q=1"),
        ("attention", "query_length", 2, "short Q (2–512)"),
        ("attention", "query_length", 512, "short Q (2–512)"),
        ("attention", "query_length", 513, "medium Q (513–2048)"),
        ("attention", "query_length", 2048, "medium Q (513–2048)"),
        ("attention", "query_length", 2049, "long Q (≥2049)"),
        ("vector", "vector_tokens", 16, "tokens≤16"),
        ("vector", "vector_tokens", 17, "17≤tokens≤64"),
        ("vector", "vector_tokens", 64, "17≤tokens≤64"),
        ("vector", "vector_tokens", 65, "65≤tokens≤2047"),
        ("vector", "vector_tokens", 2047, "65≤tokens≤2047"),
        ("vector", "vector_tokens", 2048, "tokens≥2048"),
        ("wave", "max_launch_waves", 1, "1 wave"),
        ("wave", "max_launch_waves", 2, "2–4 waves"),
        ("wave", "max_launch_waves", 4, "2–4 waves"),
        ("wave", "max_launch_waves", 5, "5–16 waves"),
        ("wave", "max_launch_waves", 16, "5–16 waves"),
        ("wave", "max_launch_waves", 17, "≥17 waves"),
    ),
)
def test_report_bucket_boundaries(family, field, value, label):
    if family == "gemm":
        metadata = {
            "CASE": {
                "op_type": "GEMM", "eqm": 1, "split": False,
                "gemm_m": 1, "gemm_n": 1, "gemm_k": 1,
                "max_launch_waves": 1,
            },
        }
    elif family == "moe":
        metadata = {
            "CASE": {
                "op_type": "FUSEDMOE", "eqm": 1,
                "moe_experts": 1, "moe_max_tokens_per_expert": 1,
                "moe_topk": 1, "max_launch_waves": 1,
            },
        }
    elif family == "attention":
        metadata = {
            "CASE": _attention_meta(1, 1),
        }
    elif family in ("vector", "wave"):
        metadata = {
            "CASE": {
                "op_type": "NORM", "vector_tokens": 1,
                "max_launch_waves": 1,
            },
        }
    else:
        raise AssertionError(f"unexpected test family {family!r}")
    metadata["CASE"][field] = value

    rows = {
        row_label: keys
        for row_label, keys, _indent, _is_last in define_subsets(metadata)
    }

    assert rows[label] == ["CASE"]


@pytest.mark.parametrize(
    "hardware_id,family,task_key,expected_domain,expected_waves",
    (
        ("nvidia_a800", "gemm", "GEMM_B8_M4096_K4096_N11008",
         "generic", 408),
        ("ascend_910b", "gemm", "GEMM_B1_M1_K4096_N4096", "aic", 1),
        ("ascend_910b", "vector", "VECTOR_RMSNORM_T1_H2048_B1", "aiv", 1),
        ("ascend_910b", "flash_attn",
         "FLASHATTN_B1_S512_HQ32_HKV8_D128_C0", "mixed", 1),
    ),
)
def test_launch_wave_report_uses_the_static_execution_domain(
        hardware_id, family, task_key, expected_domain, expected_waves,
        monkeypatch):
    recorded_domains = []
    execution_units_for_task = (
        evaluation_runner.HWSpec.execution_units_for_task)

    def record_execution_domain(hardware, launch_task_kind):
        recorded_domains.append(launch_task_kind)
        return execution_units_for_task(hardware, launch_task_kind)

    monkeypatch.setattr(
        evaluation_runner.HWSpec,
        "execution_units_for_task",
        record_execution_domain,
    )
    entry = _target_invariance_entry(hardware_id, family, task_key)
    result = _process_op(
        task_key, entry, get_hardware_config_path(hardware_id),
        False, False)

    assert recorded_domains == [expected_domain]
    assert result[1]["max_launch_waves"] == expected_waves


def test_evaluation_metadata_records_shape_routing_and_layout_dimensions():
    cases = (
        ("nvidia_a800", "gemm", "GEMM_B1_M1_K1536_N5120",
         {"gemm_m": 1, "gemm_n": 5120, "gemm_k": 1536}),
        ("metax_c500", "fused_moe",
         "FUSEDMOE_TOK1_E8_TOPK2_H4096_INT14336",
         {"moe_experts": 8, "moe_topk": 2,
          "moe_max_tokens_per_expert": 1}),
        ("ascend_910b", "flash_attn",
         "FLASHATTN_B1_S512_HQ32_HKV8_D128_C0",
         {"attention_layout": "contiguous"}),
        ("nvidia_a800", "vector", "VECTOR_RMSNORM_T1_H2048_B1",
         {"vector_tokens": 1}),
    )

    for hardware_id, family, task_key, expected in cases:
        entry = _target_invariance_entry(hardware_id, family, task_key)
        result = _process_op(
            task_key, entry, get_hardware_config_path(hardware_id),
            False, False)
        for name, value in expected.items():
            assert result[1][name] == value
        assert result[1]["max_launch_waves"] >= 1


@pytest.mark.parametrize(
    "mutation,message",
    (
        (lambda attr: attr["tok_per_expert"].__setitem__(0, -1),
         "entries must be non-negative integers"),
        (lambda attr: attr["tok_per_expert"].__setitem__(0, 2),
         "sums to"),
        (lambda attr: attr["tok_per_expert"].pop(),
         "entries, expected"),
    ),
)
def test_moe_report_metadata_rejects_invalid_routing(mutation, message):
    task_key = "FUSEDMOE_TOK1_E8_TOPK2_H4096_INT14336"
    entry = _target_invariance_entry("nvidia_a800", "fused_moe", task_key)
    mutation(entry["op_info"]["Attr"])

    with pytest.raises(ValueError, match=message):
        _process_op(
            task_key, entry, get_hardware_config_path("nvidia_a800"),
            False, False)


def test_attention_report_metadata_rejects_unknown_layout():
    task_key = "FLASHATTN_B1_S512_HQ32_HKV8_D128_C0"
    entry = _target_invariance_entry("nvidia_a800", "flash_attn", task_key)
    entry["op_info"]["Attr"]["kv_layout"] = "mystery"

    with pytest.raises(ValueError, match="kv_layout must be one of"):
        _process_op(
            task_key, entry, get_hardware_config_path("nvidia_a800"),
            False, False)


def test_attention_metadata_records_only_externally_observed_splits():
    a800_key = "FLASHATTN_PAGED_B2_S512_HQ32_HKV8_D128_C0"
    a800 = _with_a800_timing(load_kernel_dataset_document(
        "kernel_dataset/nvidia_a800/flash_attn.json.zst")[a800_key])
    a800_result = _process_op(
        a800_key, a800, get_hardware_config_path("nvidia_a800"),
        False, False)
    assert a800_result[1]["executed_attention_splits"] == 4

    ascend_key = "FLASHATTN_B1_S512_HQ32_HKV8_D128_C0"
    ascend = load_kernel_dataset_document(
        "kernel_dataset/ascend_910b/flash_attn.json.zst")[ascend_key]
    ascend_result = _process_op(
        ascend_key, ascend, get_hardware_config_path("ascend_910b"),
        False, False)
    assert ascend_result[1]["executed_attention_splits"] is None


def test_explicit_timing_target_is_separate_from_profiler_duration():
    kernel = {"Metadata": {"Kernel Name": "kernel"}}
    timing = parse_kernel_evaluation_timing(
        {"op_info": {
            "Type": "GEMM",
            "Attr": {},
            "evaluation": {
                "timing": {
                    "granularity": "kernel",
                    "source": "nsys_nvtx_kernel_median",
                    "total_duration_ns": 3328.0,
                    "kernels": [{
                        "kernel_idx": 0,
                        "kernel_name": "kernel",
                        "duration_ns": 3328.0,
                    }],
                },
            },
        }},
        [kernel],
        "nvidia_a800",
    )

    assert timing.durations_ns == (3328.0,)
    assert timing.granularity == "kernel"
    assert timing.source == "nsys_nvtx_kernel_median"


def test_c500_vector_uses_op_level_device_event_timing():
    task_key = "VECTOR_RMSNORM_T1_H2048_B1"
    entry = _target_invariance_entry("metax_c500", "vector", task_key)
    timing = parse_kernel_evaluation_timing(
        entry, _kernel_records(entry), "metax_c500")

    assert timing.granularity == "op"
    assert timing.source == "device_event_op_median"
    assert timing.durations_ns == (timing.total_duration_ns,)

    result = _process_op(
        task_key, entry, get_hardware_config_path("metax_c500"),
        True, True, capture_mechanisms=True)
    assert result[1]["actual_granularity"] == "op"
    assert result[-1][0]["actual_granularity"] == "op"
    assert result[-1][0]["mechanisms"]["timing_target"] == {
        "availability": "unobserved",
        "duration_ns": None,
        "source": None,
        "unavailable_reason": "canonical_target_is_op_level",
    }


@pytest.mark.parametrize(
    "hardware_id", ("metax_c500", "ascend_910b"))
def test_norm_dataset_and_serving_share_canonical_launch_identity(hardware_id):
    task_key = "VECTOR_RMSNORM_T1_H2048_B1"
    entry = _target_invariance_entry(hardware_id, "vector", task_key)
    hardware = HWSpec(get_hardware_config_path(hardware_id))
    dataset_spec = normalize_kernel_spec(
        entry["op_info"], _kernel_records(entry)[0], hardware_id, hardware)
    serving_launch = select_launch_config(
        hardware_id, KernelKind.VECTOR, dataset_spec.shape,
        dataset_spec.dtype, dataset_spec.variant,
        {**dataset_spec.attributes, "launch_policy": "injected_launch"},
        hardware)

    assert dataset_spec.launch == serving_launch


def test_ascend_fused_qknorm_rope_dataset_and_serving_share_contract():
    dataset = load_kernel_dataset("ascend_910b", ["vector"])
    entry = copy.deepcopy(next(
        value for value in dataset.values()
        if value["op_info"]["Type"] == "ROTARY"))
    attributes = entry["op_info"]["Attr"]
    attributes.update({
        "op": "qkv_rmsnorm_rope",
        "tok": 2220,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rotary_dim": 128,
        "element_count": 2220 * (32 + 2 * 8) * 128,
        "bias": False,
        "is_neox": True,
    })
    record = entry["kernels"][0]
    record["Metadata"].update({
        "Kernel Name": "split_qkv_rmsnorm_rope_kernel",
        "Grid Size": "48,1,1",
        "Block Size": "1,1,1",
    })
    record["Sections"]["Launch Statistics"]["Header"]["Block Size"] = 1
    hardware = HWSpec(get_hardware_config_path("ascend_910b"))

    dataset_spec = normalize_kernel_spec(
        entry["op_info"], record, "ascend_910b", hardware)
    duration_changed_record = copy.deepcopy(record)
    duration_changed_record["Sections"][
        "GPU Speed Of Light Throughput"]["Header"][
            "Duration [nsecond]"] *= 1000
    assert normalize_kernel_spec(
        entry["op_info"], duration_changed_record,
        "ascend_910b", hardware) == dataset_spec
    serving_launch = select_launch_config(
        "ascend_910b",
        KernelKind.VECTOR,
        dataset_spec.shape,
        dataset_spec.dtype,
        dataset_spec.variant,
        {**dataset_spec.attributes, "launch_policy": "injected_launch"},
        hardware,
    )

    assert dataset_spec.variant == "qkv_rmsnorm_rope"
    assert dataset_spec.launch == serving_launch
    assert dataset_spec.shape == {
        "seq_len": 2220,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rotary_dim": 128,
        "element_count": 2220 * (32 + 2 * 8) * 128,
        "elements": 2220 * (32 + 2 * 8) * 128,
        "dtype_bytes": 2,
    }


def test_a800_vector_uses_kernel_level_nsys_timing():
    dataset = load_kernel_dataset("nvidia_a800", ["vector"])

    for entry in dataset.values():
        timing = parse_kernel_evaluation_timing(
            entry, _kernel_records(entry), "nvidia_a800")
        assert timing.granularity == "kernel"
        assert timing.source == "nsys_nvtx_kernel_median"
        assert timing.total_duration_ns > 0.0


@pytest.mark.parametrize(
    "family,task_key",
    (
        ("gemm", "GEMM_B1_M1_K4096_N4096"),
        ("flash_attn", "FLASHATTN_PAGED_PREFILL_B128_S4096_HQ32_HKV8_D128_C1"),
        ("fused_moe", "FUSEDMOE_TOK1_E8_TOPK2_H4096_INT14336"),
    ),
)
def test_c500_stable_graph_cases_use_op_level_device_event_timing(
        family, task_key):
    entry = _target_invariance_entry("metax_c500", family, task_key)
    timing = parse_kernel_evaluation_timing(
        entry, _kernel_records(entry), "metax_c500")

    assert timing.granularity == "op"
    assert timing.source == "device_event_op_median"
    assert timing.total_duration_ns > 0.0


def test_c500_event_publication_uses_op_level_targets():
    expected_op_case_counts = {
        "gemm": 221,
        "flash_attn": 334,
        "fused_moe": 176,
    }

    for family, expected_count in expected_op_case_counts.items():
        dataset = load_kernel_dataset("metax_c500", [family])
        op_case_count = sum(
            entry["op_info"]["evaluation"]["timing"]["granularity"] == "op"
            for entry in dataset.values())
        assert op_case_count == expected_count
        assert len(dataset) == op_case_count


def test_op_level_timing_does_not_fabricate_per_kernel_targets():
    entry = _target_invariance_entry(
        "metax_c500", "vector", "VECTOR_RMSNORM_T1_H2048_B1")
    kernels = _kernel_records(entry)

    timing = parse_kernel_evaluation_timing(
        entry, [kernels[0], copy.deepcopy(kernels[0])], "metax_c500")

    assert timing.granularity == "op"
    assert timing.durations_ns == ()


def test_ascend_multi_kernel_op_timing_scores_only_the_logical_op():
    task_key = "FUSEDMOE_TOK1_E8_TOPK2_H4096_INT14336"
    entry = copy.deepcopy(
        load_kernel_dataset("ascend_910b", ["fused_moe"])[task_key])
    target_ns = 123_456.0
    entry["op_info"]["evaluation"]["timing"] = {
        "granularity": "op",
        "source": "msprof_timeline_op_median",
        "total_duration_ns": target_ns,
        "provenance": {
            "baseline_commit": "a" * 40,
            "measurement_mode": "msprof_timeline_task_start_and_duration",
            "profiler": "msprof",
            "runner_sha256": "b" * 64,
            "scope": "logical_public_op",
            "source_artifact_sha256": "c" * 64,
            "source_artifact_tree_sha256": "d" * 64,
            "source_snapshot_sha256": "e" * 64,
            "statistic": "median_of_per_call_op_elapsed_duration",
        },
    }

    timing = parse_kernel_evaluation_timing(
        entry, _kernel_records(entry), "ascend_910b")
    result = _process_op(
        task_key, entry, get_hardware_config_path("ascend_910b"),
        True, True, capture_mechanisms=True)

    assert timing.durations_ns == ()
    assert result[3][1] == target_ns
    assert all(row["actual_ns"] is None for row in result[-1])
    assert all("err_forward" not in row for row in result[-1])
    for row in result[-1]:
        timing_target = row["mechanisms"]["timing_target"]
        assert timing_target == {
            "availability": "unobserved",
            "duration_ns": None,
            "source": None,
            "unavailable_reason": "canonical_target_is_op_level",
        }


def test_ascend_public_fused_moe_forward_uses_public_source_recipe(
        monkeypatch):
    task_key = "FUSEDMOE_TOK1_E8_TOPK2_H4096_INT14336"
    entry = copy.deepcopy(
        load_kernel_dataset("ascend_910b", ["fused_moe"])[task_key])
    recorded = []
    production_predict = evaluation_runner.predict_production

    def record_prediction(kernel_spec, hardware, method="midlayer", **kwargs):
        if method == "midlayer":
            recorded.append(kernel_spec.variant)
        return production_predict(
            kernel_spec, hardware, method=method, **kwargs)

    monkeypatch.setattr(
        evaluation_runner, "predict_production", record_prediction)
    result = _process_op(
        task_key, entry, get_hardware_config_path("ascend_910b"),
        False, True)

    assert result[3][0] > 0.0
    assert recorded[-7:] == [
        "moe_gating_topk", "moe_init_routing_v3", "w1",
        "moe_swiglu", "w2", "moe_abs", "moe_token_unpermute",
    ]
    assert recorded.count("w1") == 2
    assert recorded.count("w2") == 2


def test_explicit_timing_target_does_not_change_oracle_prediction():
    task_key = "GEMM_B1_M1_K1536_N5120"
    entry = _with_a800_timing(load_kernel_dataset_document(
        "kernel_dataset/nvidia_a800/gemm.json.zst")[task_key])
    baseline = _process_op(
        task_key,
        entry,
        get_hardware_config_path("nvidia_a800"),
        True,
        False,
    )
    baseline_kernels = baseline[-1]
    assert len(baseline_kernels) == 1

    changed_entry = copy.deepcopy(entry)
    changed_timing = changed_entry["op_info"]["evaluation"]["timing"]
    changed_timing["kernels"][0]["duration_ns"] = (
        baseline_kernels[0]["actual_ns"] * 3.0)
    changed_timing["total_duration_ns"] = (
        baseline_kernels[0]["actual_ns"] * 3.0)
    changed = _process_op(
        task_key,
        changed_entry,
        get_hardware_config_path("nvidia_a800"),
        True,
        False,
    )
    changed_kernels = changed[-1]

    assert changed_kernels[0]["actual_ns"] == (
        baseline_kernels[0]["actual_ns"] * 3.0)
    assert changed_kernels[0]["actual_source"] == (
        "nsys_nvtx_kernel_median")
    assert changed_kernels[0]["pred_oracle_ns"] == (
        baseline_kernels[0]["pred_oracle_ns"])


@pytest.mark.parametrize(
    "hardware_id,family,task_key", _TARGET_INVARIANCE_CASES)
def test_oracle_prediction_and_provenance_are_target_invariant(
        hardware_id, family, task_key, monkeypatch):
    observations = []
    normalize = evaluation_runner.normalize_oracle_observation

    def record_observation(kernel, profiler_record, hardware, **kwargs):
        observation = normalize(kernel, profiler_record, hardware, **kwargs)
        observations.append(observation)
        return observation

    monkeypatch.setattr(
        evaluation_runner, "normalize_oracle_observation",
        record_observation)
    entry = _target_invariance_entry(hardware_id, family, task_key)
    hardware_config = get_hardware_config_path(hardware_id)

    baseline = _process_op(
        task_key, entry, hardware_config, True, False)
    baseline_rows = baseline[-1]
    baseline_observations = tuple(observations)
    observations.clear()

    target_factor = 3.0
    changed_entry = _change_evaluation_target(
        entry, hardware_id, baseline_rows, target_factor)
    changed = _process_op(
        task_key, changed_entry, hardware_config, True, False)
    changed_rows = changed[-1]
    changed_observations = tuple(observations)

    baseline_supported_rows = tuple(
        row for row in baseline_rows if "pred_oracle_ns" in row)
    changed_supported_rows = tuple(
        row for row in changed_rows if "pred_oracle_ns" in row)
    assert len(baseline_observations) == len(baseline_supported_rows)
    assert len(changed_observations) == len(changed_supported_rows)
    timing_granularity = entry["op_info"]["evaluation"]["timing"][
        "granularity"]
    if timing_granularity == "kernel":
        assert [row["actual_ns"] for row in changed_rows] == pytest.approx([
            row["actual_ns"] * target_factor for row in baseline_rows
        ])
    elif timing_granularity == "op":
        if len(baseline_rows) == 1:
            assert changed_rows[0]["actual_ns"] == pytest.approx(
                baseline_rows[0]["actual_ns"] * target_factor)
        else:
            assert all(row["actual_ns"] is None for row in baseline_rows)
            assert all(row["actual_ns"] is None for row in changed_rows)
    else:
        raise AssertionError(
            f"unexpected timing granularity: {timing_granularity!r}")
    assert changed[1]["actual_ns"] == pytest.approx(
        baseline[1]["actual_ns"] * target_factor)
    assert changed[2][0] == baseline[2][0]
    assert [row["pred_oracle_ns"] for row in changed_supported_rows] == [
        row["pred_oracle_ns"] for row in baseline_supported_rows
    ]
    assert [row["bottleneck"] for row in changed_supported_rows] == [
        row["bottleneck"] for row in baseline_supported_rows
    ]

    for before, after in zip(
            baseline_observations, changed_observations, strict=True):
        assert before.target_derived_paths == ()
        assert after.target_derived_paths == ()
        assert after.analytical_input == before.analytical_input
        assert after.prediction_input == before.prediction_input
        assert after.field_provenance == before.field_provenance
        assert after.unavailable_fields == before.unavailable_fields
        assert after.coverage == before.coverage
        assert after.backend == before.backend

    for before, after in zip(baseline_rows, changed_rows, strict=True):
        if "pred_oracle_ns" in before:
            assert "pred_oracle_ns" in after
            assert after["oracle_backend"] == before["oracle_backend"]
            assert after["oracle_target_derived_paths"] == []
            assert after["oracle_unavailable_fields"] == (
                before["oracle_unavailable_fields"])
            assert after["oracle_coverage"] == before["oracle_coverage"]
        elif "oracle_unavailable_reason" in before:
            assert "pred_oracle_ns" not in after
            assert after["oracle_unavailable_reason"] == (
                before["oracle_unavailable_reason"])
        else:
            raise AssertionError(
                "Oracle row is neither supported nor explicitly unavailable")


@pytest.mark.parametrize("task_key", (
    "VECTOR_ELEMENTWISE_T256_H5120_B1_ADD",
    "VECTOR_ADD_RMSNORM_T1024_H8192_B1",
))
def test_ascend_vector_compact_oracle_envelope_is_evaluation_equivalent(
        task_key, monkeypatch):
    entry = _target_invariance_entry("ascend_910b", "vector", task_key)
    hardware_config = get_hardware_config_path("ascend_910b")

    compact = _process_op(
        task_key, entry, hardware_config, True, True, True)
    monkeypatch.setattr(
        VectorPredictor, "oracle_resource_input", VectorPredictor.resource_input)
    expanded = _process_op(
        task_key, entry, hardware_config, True, True, True)

    assert compact == expanded


@pytest.mark.parametrize("hardware_id,task_key", (
    ("nvidia_a800", "VECTOR_RMSNORM_T1_H2048_B1"),
    ("metax_c500", "VECTOR_RMSNORM_T1_H2048_B1"),
    ("ascend_910b", "VECTOR_ADD_RMSNORM_T1024_H8192_B1"),
))
def test_vector_current_forward_input_cache_is_evaluation_equivalent(
        hardware_id, task_key, monkeypatch):
    entry = _target_invariance_entry(hardware_id, "vector", task_key)


@pytest.mark.parametrize("hardware_id", (
    "nvidia_a800", "metax_c500", "ascend_910b",
))
def test_gemm_current_resource_input_cache_is_evaluation_equivalent(
        hardware_id, monkeypatch):
    task_key = "GEMM_B1_M1_K4096_N4096"
    entry = _target_invariance_entry(hardware_id, "gemm", task_key)
    hardware_config = get_hardware_config_path(hardware_id)

    cached = _process_op(
        task_key, entry, hardware_config, True, True, True)

    def fresh_forward_input(_, kernel, hardware):
        return forward_analytical_resource_input(kernel, hardware)

    monkeypatch.setattr(
        VectorPredictor, "_forward_resource_input", fresh_forward_input)
    fresh = _process_op(
        task_key, entry, hardware_config, True, True, True)

    assert cached == fresh


@pytest.mark.parametrize("hardware_id", (
    "nvidia_a800", "metax_c500", "ascend_910b",
))
def test_gemm_current_resource_input_cache_is_evaluation_equivalent(
        hardware_id, monkeypatch):
    task_key = "GEMM_B1_M1_K4096_N4096"
    entry = _target_invariance_entry(hardware_id, "gemm", task_key)
    hardware_config = get_hardware_config_path(hardware_id)

    cached = _process_op(
        task_key, entry, hardware_config, True, True, True)

    def expanded_resource_input(_, kernel, hardware):
        return build_gemm_resource_input(kernel, hardware)

    monkeypatch.setattr(
        GEMMPredictor, "resource_input", expanded_resource_input)
    expanded = _process_op(
        task_key, entry, hardware_config, True, True, True)

    assert cached == expanded


@pytest.mark.parametrize("hardware_id,task_key", (
    ("nvidia_a800", "FLASHATTN_CHUNKED_UNIFORM_PREFILL_Q8192_B8_S4096_HQ32_HKV8_D128_C1"),
    ("metax_c500", "FLASHATTN_PAGED_PREFILL_B128_S4096_HQ32_HKV8_D128_C1"),
    ("ascend_910b", "FLASHATTN_PAGED_PREFILL_B128_S4096_HQ32_HKV8_D128_C1"),
))
def test_flash_attention_roofline_reuses_complete_input_equivalently(
        hardware_id, task_key, monkeypatch):
    entry = _target_invariance_entry(hardware_id, "flash_attn", task_key)
    hardware_config = get_hardware_config_path(hardware_id)

    reused = _process_op(
        task_key, entry, hardware_config, True, True, True)

    def dedicated_roofline(_, kernel, hardware):
        resource_input, compulsory = build_flash_attention_input(
            kernel, hardware, attach_pipeline=False)
        if kernel.variant.lower() in ("combine", "split_combine"):
            compulsory = resource_input.demand_per_resource["l2_cache"]
        return prediction_from_ns(
            roofline_ns(
                resource_input.demand_per_resource, hardware, compulsory),
            "roofline")

    monkeypatch.setattr(
        FlashAttentionPredictor, "roofline", dedicated_roofline)
    dedicated = _process_op(
        task_key, entry, hardware_config, True, True, True)

    assert reused == dedicated


def _gemm_entry(hardware_id, task_key="GEMM_B1_M1_K1536_N5120"):
    return copy.deepcopy(load_kernel_dataset_document(
        f"kernel_dataset/{hardware_id}/gemm.json.zst")[task_key])


def test_ascend_gemm_dataset_does_not_invent_predecessor_dirty_state():
    task_key = "GEMM_B1_M1_K5120_N13824"
    entry = _gemm_entry("ascend_910b", task_key)
    hardware = HWSpec(get_hardware_config_path("ascend_910b"))

    dataset_spec = normalize_kernel_spec(
        entry["op_info"], _kernel_records(entry)[0],
        "ascend_910b", hardware)

    # The canonical Timeline capture is an ordinary warm public call with no
    # pre-call eviction sweep.  A standalone KernelSpec therefore carries no
    # anonymous cross-op dirty state; Serving may supply explicit predecessor
    # state without selecting a different GEMM launch or predictor formula.
    assert dataset_spec.anonymous_l2_dirty_bytes == 0


def test_ascend_gemm_dataset_uses_its_observed_launch_not_serving_lut(
        monkeypatch):
    task_key = "GEMM_B128_M1_K11008_N4096"
    entry = _gemm_entry("ascend_910b", task_key)
    hardware = HWSpec(get_hardware_config_path("ascend_910b"))

    def reject_serving_lookup(*_args, **_kwargs):
        raise AssertionError("Kernel Dataset loaded the Serving launch LUT")

    monkeypatch.setattr(
        observation_normalizer, "select_launch_config", reject_serving_lookup)
    spec = normalize_kernel_spec(
        entry["op_info"], _kernel_records(entry)[0],
        "ascend_910b", hardware)

    assert spec.launch.algorithm == "ascend_cann_matmul"
    assert spec.launch.grid == (22, 1, 1)
    assert spec.launch.source_program.aic_workers == 22
    assert spec.launch.source_program.single_core_k == 11008


def test_ascend_gemm_dataset_binds_observed_split_program():
    task_key = "GEMM_B128_M1_K28672_N8192"
    entry = _gemm_entry("ascend_910b", task_key)
    hardware = HWSpec(get_hardware_config_path("ascend_910b"))

    spec = normalize_kernel_spec(
        entry["op_info"], _kernel_records(entry)[0],
        "ascend_910b", hardware)

    assert spec.launch.algorithm == "cann_matmul_v3_deterministic_split_k"
    assert spec.launch.grid == (24, 1, 1)
    assert spec.launch.source_program.program_kind == (
        "cann_gemm_deterministic_split_k_v1")


def test_ascend_gemm_timeline_context_requires_the_bound_runner_source():
    entry = _gemm_entry("ascend_910b", "GEMM_B1_M1_K8192_N8192")

    context = kernel_dataset_l2_execution_context(
        entry["op_info"], "ascend_910b")

    assert context.same_program_operand_replays == 5

    invalid = copy.deepcopy(entry["op_info"])
    invalid["evaluation"]["timing"]["provenance"]["runner_sha256"] = (
        "0" * 64)
    with pytest.raises(ValueError, match="runner provenance"):
        kernel_dataset_l2_execution_context(invalid, "ascend_910b")


def test_legacy_top_level_timing_cannot_replace_evaluation_contract():
    task_key = "GEMM_B1_M1_K1536_N5120"
    entry = _gemm_entry("nvidia_a800", task_key)
    entry["evaluation"] = entry["op_info"].pop("evaluation")

    with pytest.raises(ValueError, match="legacy top-level evaluation"):
        _process_op(
            task_key,
            entry,
            get_hardware_config_path("nvidia_a800"),
            False,
            False,
        )


def test_missing_evaluation_timing_fails_instead_of_using_raw_header():
    task_key = "GEMM_B1_M1_K1536_N5120"
    entry = _gemm_entry("nvidia_a800", task_key)
    del entry["op_info"]["evaluation"]

    with pytest.raises(ValueError, match="missing required op_info.evaluation.timing"):
        _process_op(
            task_key,
            entry,
            get_hardware_config_path("nvidia_a800"),
            False,
            False,
        )


def test_legacy_overall_duration_is_rejected():
    entry = _gemm_entry("nvidia_a800")
    entry["op_info"]["Overall Duration [nsecond]"] = 12345.0

    with pytest.raises(ValueError, match="legacy op_info Overall Duration"):
        parse_kernel_evaluation_timing(
            entry, _kernel_records(entry), "nvidia_a800")


def test_unknown_explicit_timing_source_fails_loudly():
    entry = _gemm_entry("nvidia_a800")
    entry["op_info"]["evaluation"]["timing"]["source"] = "mystery"

    with pytest.raises(ValueError, match="source for nvidia_a800 must be"):
        parse_kernel_evaluation_timing(
            entry, _kernel_records(entry), "nvidia_a800")


@pytest.mark.parametrize(
    "total,error_type,message",
    (
        (0.0, ValueError, "finite and positive"),
        (-1.0, ValueError, "finite and positive"),
        (float("nan"), ValueError, "finite and positive"),
        (True, TypeError, "must be numeric"),
    ),
)
def test_invalid_total_timing_target_fails_loudly(
        total, error_type, message):
    entry = _gemm_entry("nvidia_a800")
    entry["op_info"]["evaluation"]["timing"][
        "total_duration_ns"] = total

    with pytest.raises(error_type, match=message):
        parse_kernel_evaluation_timing(
            entry, _kernel_records(entry), "nvidia_a800")


def test_total_timing_must_equal_kernel_duration_sum():
    entry = _gemm_entry("nvidia_a800")
    timing = entry["op_info"]["evaluation"]["timing"]
    timing["total_duration_ns"] += 1.0

    with pytest.raises(ValueError, match="does not equal the sum"):
        parse_kernel_evaluation_timing(
            entry, _kernel_records(entry), "nvidia_a800")


def test_per_kernel_timing_count_must_match_profiled_launches():
    entry = _gemm_entry("nvidia_a800", "GEMM_B16_M1_K5120_N13824")
    del entry["op_info"]["evaluation"]["timing"]["kernels"][-1]

    with pytest.raises(ValueError, match="count does not match"):
        parse_kernel_evaluation_timing(
            entry, _kernel_records(entry), "nvidia_a800")


@pytest.mark.parametrize(
    "duration,error_type,message",
    (
        (0.0, ValueError, "finite and positive"),
        (-1.0, ValueError, "finite and positive"),
        (float("nan"), ValueError, "finite and positive"),
        (True, TypeError, "must be numeric"),
    ),
)
def test_invalid_kernel_timing_target_fails_loudly(
        duration, error_type, message):
    entry = _gemm_entry("nvidia_a800")
    entry["op_info"]["evaluation"]["timing"]["kernels"][0][
        "duration_ns"] = duration

    with pytest.raises(error_type, match=message):
        parse_kernel_evaluation_timing(
            entry, _kernel_records(entry), "nvidia_a800")


def test_kernel_timing_indices_must_be_contiguous():
    entry = _gemm_entry("nvidia_a800")
    entry["op_info"]["evaluation"]["timing"]["kernels"][0][
        "kernel_idx"] = 1

    with pytest.raises(ValueError, match="contiguous and ordered"):
        parse_kernel_evaluation_timing(
            entry, _kernel_records(entry), "nvidia_a800")


def test_kernel_timing_name_must_match_raw_kernel():
    entry = _gemm_entry("nvidia_a800")
    entry["op_info"]["evaluation"]["timing"]["kernels"][0][
        "kernel_name"] = "wrong"

    with pytest.raises(ValueError, match="kernel name mismatch"):
        parse_kernel_evaluation_timing(
            entry, _kernel_records(entry), "nvidia_a800")


def test_kernel_timing_rejects_unknown_fields():
    entry = _gemm_entry("nvidia_a800")
    entry["op_info"]["evaluation"]["timing"]["kernels"][0][
        "raw_duration_ns"] = 1.0

    with pytest.raises(ValueError, match="fields are invalid"):
        parse_kernel_evaluation_timing(
            entry, _kernel_records(entry), "nvidia_a800")


@pytest.mark.parametrize(
    "hardware_id,task_key,expected_source",
    (
        ("nvidia_a800", "GEMM_B1_M1_K1536_N5120",
         "nsys_nvtx_kernel_median"),
        ("metax_c500", "GEMM_B1_M1_K1536_N5120",
         "device_event_op_median"),
        ("ascend_910b", "GEMM_B1_M1_K1536_N5120",
         "msprof_timeline_op_median"),
        ("samsung_hbm_pim", "GEMM_B1_M1_K4096_N4096",
         "unindp_simulator_latency"),
        ("groq_lpu", "GEMM_B1_M1_K4096_N11008",
         "timeloop_lpu_scaled_hybrid_latency"),
        ("ucb_gemmini", "GEMM_B1_M1_K11008_N4096",
         "timeloop_mapper_simulated_latency"),
    ),
)
def test_raw_header_duration_is_never_the_evaluation_target(
        hardware_id, task_key, expected_source):
    entry = _gemm_entry(hardware_id, task_key)
    baseline = _process_op(
        task_key, entry, get_hardware_config_path(hardware_id),
        False, False)
    changed = copy.deepcopy(entry)
    for kernel in _kernel_records(changed):
        header = kernel["Sections"]["GPU Speed Of Light Throughput"]["Header"]
        header["Duration [nsecond]"] = float(
            str(header["Duration [nsecond]"]).replace(",", "")) * 100
    result = _process_op(
        task_key, changed, get_hardware_config_path(hardware_id),
        False, False)

    assert result[1]["actual_ns"] == baseline[1]["actual_ns"]
    assert result[1]["actual_source"] == expected_source


def test_c500_evaluation_rejects_mcprofiler_replay_as_timing_target():
    task_key = "GEMM_B1_M1_K4096_N4096"
    entry = _gemm_entry("metax_c500", task_key)
    entry["kernels"][0]["MetaX"]["dur_source"] = "mcProfiler"

    with pytest.raises(ValueError, match="raw C500 duration must come from mcTracer"):
        _process_op(
            task_key,
            entry,
            get_hardware_config_path("metax_c500"),
            False,
            False,
        )


def test_c500_evaluation_reports_graph_event_timing_provenance():
    task_key = "GEMM_B1_M1_K13824_N5120"
    entry = _target_invariance_entry(
        "metax_c500", "gemm", task_key)

    result = _process_op(
        task_key,
        entry,
        get_hardware_config_path("metax_c500"),
        False,
        True,
    )

    assert result[1]["actual_source"] == "device_event_op_median"
    assert {row["actual_source"] for row in result[-1]} == {
        "device_event_op_median"}
