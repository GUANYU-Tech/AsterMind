from dataclasses import replace

import pytest

from evaluation.serving.simulation.execution_artifact import (
    structural_lowering_manifest,
)
from evaluation.serving.simulation.model_semantics_validation import (
    KernelCheckpoint,
    OperatorCheckpoint,
    RequestMetricCheckpoint,
    ServingChainReference,
    ServingValidationError,
    compare_serving_chain,
    target_model_support_matrix,
)
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_spec import KernelKind
from src.serving.deployment.deployment_config import (
    DeploymentConfig,
    WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.kernel_lowering.kernel_predict import predict_kernel_plan
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.metrics.serving_metrics import RequestMetrics, ServingMetrics
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduleStep,
    ScheduledRequest,
)
from src.serving.workload.build_workload import build_workload
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import ExpertRoutingInput, RuntimeInputs
from src.serving.workload.workload_graph import WorkloadKind


_PROXY_SEQUENCE = (
    "validation:0:token-embedding",
    "validation:0:layer-0:input-norm",
    "validation:0:layer-0:qkv",
    "validation:0:layer-0:q-norm",
    "validation:0:layer-0:k-norm",
    "validation:0:layer-0:rotary",
    "validation:0:layer-0:kv-update",
    "validation:0:layer-0:attention",
    "validation:0:layer-0:o-proj",
    "validation:0:layer-0:post-attn-norm",
    "validation:0:layer-0:gate-up",
    "validation:0:layer-0:activation",
    "validation:0:layer-0:down",
    "validation:0:final-norm",
    "validation:0:lm-head",
)


def _proxy_chain():
    model = load_model_config("tests/serving/fixtures/qwen3_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig(
            "colocated",
            (WorkerPoolConfig("main", "colocated", "nvidia_a800"),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    requests = RequestStream((RequestSpec(0.0, "request", 4, 0),))
    schedule = ScheduleStep(
        "validation:0",
        "main",
        (ScheduledRequest("request", "prefill", 4, 0),),
        finished_request_ids=("request",),
        attention_kv_layout="paged_varlen",
    )
    backend_step = BackendStep(schedule, generated_tokens={"request": 0})
    workload = build_workload(
        schedule,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.not_applicable(),
            "paged_varlen",
            "uniform",
        ),
    )
    hardware = {
        "nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml"),
    }
    plan = lower_workload(workload, deployment, hardware)
    predictions = predict_kernel_plan(plan, hardware)
    metrics = ServingMetrics(
        requests={
            "request": RequestMetrics("request", None, None, 0.01),
        },
        request_throughput_per_s=100.0,
        output_token_throughput_per_s=0.0,
        token_throughput_per_s=400.0,
    )
    return requests, backend_step, workload, plan, predictions, metrics


def test_target_model_support_matrix_separates_six_stages():
    kimi = target_model_support_matrix(load_model_config(
        "tests/serving/fixtures/kimi_k2_6_official"))
    glm = target_model_support_matrix(load_model_config(
        "tests/serving/fixtures/glm_5_2_official"))

    assert kimi["schema_version"] == 3
    assert glm["schema_version"] == 3
    assert tuple(row["stage"] for row in kimi["stages"]) == (
        "configuration_parse",
        "scheduler_step",
        "logical_workload",
        "kernel_spec",
        "hardware_launch",
        "hardware_validation",
    )
    assert tuple(row["status"] for row in kimi["stages"]) == (
        "supported",
        "supported_scheduler_only",
        "supported_static",
        "supported_structural",
        "unsupported",
        "supported_transformers_tiny",
    )
    assert tuple(row["status"] for row in glm["stages"]) == (
        "supported",
        "supported_scheduler_only",
        "supported_logical",
        "supported_structural",
        "unsupported",
        "supported_transformers_tiny",
    )
    assert {
        (row["phase"], row["layer_type"])
        for row in kimi["cases"]
    } == {
        (phase, layer_type)
        for phase in ("prefill", "decode", "mixed")
        for layer_type in ("dense", "moe")
    }
    assert tuple(
        row["attention_variant"] for row in glm["cases"]
    ) == (
        "dsa_full_indexer", "dsa_shared_indexer",
        "dsa_full_indexer", "dsa_shared_indexer",
        "dsa_full_indexer", "dsa_shared_indexer",
    )
    for matrix in (kimi, glm):
        for row in matrix["cases"]:
            levels = {
                level["level"]: level["status"]
                for level in row["validation_levels"]
            }
            assert levels["fixed_upstream_source"] == "supported"
            assert levels["production_shape_logical_workload"] == "supported"
            assert levels["tiny_transformers_cuda_control_flow"] == (
                "supported")
            assert levels["kernel_lowering"] == "supported_structural"
            assert levels["kernel_prediction"] == "unsupported"


def test_target_model_support_matrix_rejects_unpinned_identity():
    model = load_model_config(
        "tests/serving/fixtures/kimi_k2_6_official")

    with pytest.raises(ValueError, match="fixed Kimi-K2.6 and GLM-5.2"):
        target_model_support_matrix(replace(model, config_sha256="0" * 64))
