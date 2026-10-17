from dataclasses import replace

from evaluation.serving.simulation.execution_artifact import (
    PROJECTED_PERFORMANCE_TIMING_STATUS,
    StreamingExecutionArtifactAccumulator,
)
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor, predict
from src.kernel.kernel_spec import KernelKind, KernelSpec, PAGED_KV_CACHE_ATTRIBUTES
from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.serving.deployment.af_disagg import build_af_disagg_plan
from src.serving.deployment.deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    PipelineP2PConfig,
    RouteConfig,
    WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.deployment.pd_disagg import build_pd_disagg_plan
from src.serving.host_timing import load_host_launch_profile
from src.serving.kernel_lowering.execution_timeline import (
    build_execution_timeline,
)
from src.serving.kernel_lowering import kernel_predict as kernel_predict_module
from src.serving.kernel_lowering.kernel_predict import (
    KernelPredictionMemo,
    predict_kernel_plan,
)
from src.serving.kernel_lowering.kernel_plan import KernelCall, KernelPlan
from src.kernel.predictors.fused_moe import canonical_expert_axis
from src.serving.kernel_lowering.lower_workload import (
    KernelLoweringMemo,
    _fused_moe_routing_relabelling,
    _fused_moe_specs,
    _gemm_specs,
    lower_workload,
)
from src.serving.kv_cache import KVCacheConfig, KVCacheTierConfig
from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.kv_cache.model_residency import (
    _linear_weight_bytes,
    _target_moe_weight_bytes,
)
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.request_gen.request_stream import RequestSpec, RequestStream
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.schedule.schedule_step import (
    BackendStep,
    ScheduleStep,
    ScheduledRequest,
)
from src.serving.simulation import (
    MultiSimulation,
    Simulation,
    SimulationConfig,
    validate_workload_emission_parallelism,
)
from src.serving.workload.build_workload import (
    _collective_ring_order,
    build_pd_kv_transfer_workload,
    build_workload,
    is_validated_standalone_af_ep2_scope,
)
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    ExpertRoutingReplayContext,
    ExpertRoutingReplayProvenance,
    ExpertRoutingReplayRecord,
    expert_routing_replay_payload_sha256,
    load_expert_routing_replay,
    RuntimeInputs,
    seeded_expert_token_counts,
)
from src.serving.workload.graph_padding import GraphExecutionConfig
from src.serving.workload.shape_inference import (
    infer_gemm_dimensions,
    padded_vocab_size,
)


from src.serving.workload.workload_graph import (
    WorkloadEdge, WorkloadGraph, WorkloadKind, WorkloadNode,
)
from tests.serving.vllm_workload_reference import (
    compare_reference_to_emission,
    compare_tp2_reference_to_emission,
    kimi_k2_6_mla_projection_reference,
    llama_af_tp2_forward_reference,
    llama_tp2_forward_reference,
    pinned_kimi_mla_source_evidence,
    pinned_llama_source_evidence,
    pinned_qwen3_moe_ep_source_evidence,
    pinned_qwen3_pp_source_evidence,
    pinned_qwen3_source_evidence,
    qwen3_moe_tp2_attention_reference,
    qwen3_pp2_forward_reference,
    qwen3_pp3_forward_reference,
    qwen3_single_rank_forward_reference,
    qwen3_tp2_forward_reference,
    qwen3_tp2_pp2_forward_reference,
    qwen3_uniform_pp_forward_reference,
)


@pytest.mark.parametrize("algorithm", ("fc2", "mesh", "tree"))
def test_non_ring_collectives_do_not_emit_a_ring_order(algorithm):
    assert _collective_ring_order(algorithm, (0, 1, 2, 3)) == ()


def test_ring_collective_preserves_participant_order():
    assert _collective_ring_order("ring", (3, 1, 2, 0)) == (3, 1, 2, 0)


def test_collective_ring_order_rejects_unknown_algorithm():
    with pytest.raises(
            ValueError, match="unsupported Serving collective algorithm"):
        _collective_ring_order("unknown", (0, 1))


@pytest.mark.parametrize(
    "operation,hardware_id,algorithm,link,topology,backend,backend_sha,"
    "ring_order",
    (
        (
            "all_reduce", "nvidia_a800", "ring", "nvlink", "ring",
            "analytical_roofline_v1", None, (5, 3, 7),
        ),
        (
            "all_gather", "nvidia_a800", "ring", "nvlink", "ring",
            "analytical_roofline_v1", None, (5, 3, 7),
        ),
        (
            "all_reduce", "metax_c500", "fc2", "metaxlink",
            "fully_connected", "protobridge", "a" * 64, (),
        ),
        (
            "all_gather", "metax_c500", "fc2", "metaxlink",
            "fully_connected", "protobridge", "a" * 64, (),
        ),
        (
            "all_reduce", "ascend_910b", "mesh", "hccs",
            "fully_connected", "protobridge", "b" * 64, (),
        ),
        (
            "all_gather", "ascend_910b", "mesh", "hccs",
            "fully_connected", "protobridge", "b" * 64, (),
        ),
        (
            "all_reduce", "nvidia_a800", "tree", "nvlink", "tree",
            "protobridge", "c" * 64, (),
        ),
    ),
)
def test_collective_lowering_validates_algorithm_specific_ring_order(
        operation, hardware_id, algorithm, link, topology, backend,
        backend_sha, ring_order):
    participants = (5, 3, 7)
    reductions = {
        "all_reduce": "sum",
        "all_gather": None,
    }
    collective = CollectiveConfig(
        operation,
        algorithm,
        link,
        topology,
        "main-collective-fabric",
        backend,
        backend_sha,
    )
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(tensor_parallel_size=3),
                ranks=participants,
                collectives=(collective,),
            ),
        )),
        num_layers=1,
    )
    node = WorkloadNode(
        operation,
        WorkloadKind.COLLECTIVE,
        "main",
        {},
        rank=participants[0],
        attributes={
            "algorithm": algorithm,
            "backend": backend,
            "backend_config_sha256": backend_sha,
            "bytes": 1024,
            "bytes_semantics": "per_rank_input",
            "direction": "collective",
            "link": link,
            "operation": operation,
            "participants": participants,
            "source_rank": None,
            "destination_rank": None,
            "ring_order": ring_order,
            "reduction": reductions[operation],
            "topology": topology,
            "resource_id": "main-collective-fabric",
        },
    )

    plan = lower_workload(
        WorkloadGraph((node,)), deployment, {hardware_id: object()})

    call, = plan.calls
    assert call.kernel_spec.transfer.ring_order == ring_order


def _deployment_for(hardware_id):
    return build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", hardware_id),)),
        num_layers=1)


@pytest.mark.parametrize(
    "phase,tokens,mixed,kv_heads",
    (("prefill", 1, True, 8), ("prefill", 1, False, 8),
     ("prefill", 2, True, 8), ("decode", 1, True, 8),
     ("prefill", 1, True, 32)),
)
def test_single_token_prefill_preserves_launch_head_groups(
        phase, tokens, mixed, kv_heads):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    model = replace(
        load_model_config("configs/model/qwen3-4b.json"),
        num_hidden_layers=1, num_dense_layers=1, num_key_value_heads=kv_heads)
    requests = (
        ScheduledRequest("B", phase, tokens if phase == "prefill" else 0,
                         tokens if phase == "decode" else 0,
                         computed_tokens=127),)
    if mixed:
        requests = (
            ScheduledRequest("A", "decode", 0, 1, computed_tokens=128),
            *requests)
    deployment = _deployment_for(hardware.hardware_id)
    graph = build_workload(
        ScheduleStep("single-token", "main", requests,
                     attention_kv_layout="paged_varlen"),
        model, deployment, launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption("test_seeded_routing_v1", 0),
            "paged_varlen", "uniform"))
    attention = next(
        node for node in graph.nodes if node.kind is WorkloadKind.FLASH_ATTN)
    plan = lower_workload(
        WorkloadGraph((attention,)), deployment,
        {hardware.hardware_id: hardware})
    kernel = next(call.kernel_spec for call in plan.calls
                  if call.kernel_spec.kind is KernelKind.FLASH_ATTN)
    assert kernel.variant == phase
    assert kernel.shape["q_len"] == tokens
    assert predict(kernel, hardware).t_total_s > 0
    if phase == "prefill":
        assert kernel.launch.grid == (1, len(requests), model.num_attention_heads)
        # Each prefill CTA processes one Q head, even for a single active row.
        # Full operand panels are issued for every CTA; logical KV sharing
        # cannot reduce either its padded Tensor work or its L2 reads.
        tile = kernel.launch.tiling.tile_shape
        panels = sum(
            (length + tile["kv"] - 1) // tile["kv"]
            for length in kernel.attributes["kv_lens"])
        resource, _ = create_predictor(KernelKind.FLASH_ATTN).resource_input(
            kernel, hardware)
        assert resource.demand_per_resource["compute_tensor"] == (
            4 * model.num_attention_heads * model.head_dim
            * tile["q"] * tile["kv"] * panels)
        q_bytes = (sum(kernel.attributes["q_lens"])
                   * model.num_attention_heads * model.head_dim * 2)
        assert resource.demand_per_resource["l2_cache"] == (
            2 * q_bytes + 2 * model.num_attention_heads * model.head_dim
            * 2 * tile["kv"] * panels)


def _test_moe_attributes(tokens, experts, topk):
    invocation_id = "test-step:layer-0:moe"
    resolution = ExpertRoutingInput.seeded_assumption(
        "test_seeded_routing_v1", 0).resolve(
            step_id="test-step",
            layer_index=0,
            moe_invocation_id=invocation_id,
            schedule_step_sha256=hashlib.sha256(
                b"test-schedule-step-v1").hexdigest(),
            tokens=tokens,
            num_experts=experts,
            topk=topk,
            model_config_sha256=hashlib.sha256(
                b"test-model-config-v1").hexdigest(),
            pool_name="main",
            tensor_parallel_size=1,
            expert_parallel_size=1,
            expert_rank_group=(0,),
            expert_placement_strategy="linear",
        )
    return {
        "tokens_per_expert": resolution.tokens_per_expert,
        "expert_parallel_size": 1,
        "moe_invocation_id": invocation_id,
        "routing_provenance": dict(resolution.routing_provenance),
    }


def _test_gemm_attributes(
        position="qkv_proj", *, tensor_parallel_size=1,
        expert_parallel_size=1, launch_policy="analytical_launch"):
    return {
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "none",
        "has_bias": False,
        "position": position,
        "tensor_parallel_size": tensor_parallel_size,
        "expert_parallel_size": expert_parallel_size,
        "launch_policy": launch_policy,
    }


def _expert_routing_replay_parts(*, counts=(1, 0), tokens=1, topk=None):
    model = load_model_config(
        "tests/serving/fixtures/qwen3_moe_ep2_scheduler")
    provenance = ExpertRoutingReplayProvenance(
        producer="vllm_model_runner_observation",
        vllm_version="0.11.0",
        vllm_source_commit="a" * 40,
        vllm_package_manifest_sha256="1" * 64,
        model_config_sha256=model.config_sha256,
    )
    context = ExpertRoutingReplayContext(
        model_config_sha256=model.config_sha256,
        request_stream_sha256="2" * 64,
        scheduler_config_sha256="3" * 64,
        pool_name="main",
        tensor_parallel_size=2,
        expert_parallel_size=2,
        expert_rank_group=(91, 7),
        expert_placement_strategy="linear",
        num_experts=model.num_experts,
        topk=(model.num_experts_per_tok if topk is None else topk),
    )
    records = tuple(
        ExpertRoutingReplayRecord(
            step_id="main:vllm:0",
            layer_index=layer_index,
            moe_invocation_id=(
                f"main:vllm:0:layer-{layer_index}:moe"),
            schedule_step_sha256="4" * 64,
            tokens=tokens,
            tokens_per_expert=counts,
        )
        for layer_index in range(model.num_hidden_layers)
    )
    return model, provenance, context, records


def test_observed_gemv_without_tile_does_not_require_split_k_reducer():
    launch = LaunchConfig(
        algorithm="observed_gemv",
        grid=(16, 1, 1),
        block=(128, 1, 1),
        tiling=None,
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 0.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
        },
        execution_variant="gemv",
    )
    main = KernelSpec(
        kind=KernelKind.GEMM,
        hardware_id="nvidia_a800",
        shape={"b": 1, "m": 1, "n": 128, "k": 2048},
        dtype="bf16",
        variant="gemv",
        launch=launch,
        attributes={},
    )

    assert _gemm_specs(
        main, main.shape, "gemm", {}, "nvidia_a800", None) == (main,)


def _tp2_collectives():
    return (
        CollectiveConfig(
            "all_reduce", "ring", "nvlink", "ring",
            "main-nvlink-fabric", "analytical_roofline_v1"),
        CollectiveConfig(
            "all_gather", "ring", "nvlink", "ring",
            "main-nvlink-fabric", "analytical_roofline_v1"),
    )


def _named_tp2_collectives(pool_name):
    return tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            f"{pool_name}-nvlink-fabric",
            "analytical_roofline_v1",
        )
        for operation in ("all_reduce", "all_gather")
    )


def _standalone_af_tp2_deployment(model, *, ffn_kv_cache=None):
    routes = tuple(
        RouteConfig(
            source,
            destination,
            "activation",
            "send_recv",
            "nvlink",
            "point_to_point",
            resource_id="attention-ffn-nvlink",
            backend="analytical_roofline_v1",
        )
        for source, destination in (
            ("attention", "ffn"),
            ("ffn", "attention"),
        )
    )
    return build_af_disagg_plan(
        DeploymentConfig(
            "af_disagg",
            (
                WorkerPoolConfig(
                    "attention",
                    "attention",
                    "nvidia_a800",
                    ParallelConfig(tensor_parallel_size=2),
                    ranks=(91, 7),
                    collectives=_named_tp2_collectives("attention"),
                ),
                WorkerPoolConfig(
                    "ffn",
                    "ffn",
                    "nvidia_a800",
                    ParallelConfig(tensor_parallel_size=2),
                    ranks=(18, 4),
                    kv_cache=ffn_kv_cache,
                    collectives=_named_tp2_collectives("ffn"),
                ),
            ),
            routes,
        ),
        num_layers=model.num_hidden_layers,
    )


def _standalone_af_ep2_deployment(
    model,
    *,
    attention_ranks=(0, 1),
    ffn_ranks=(2, 3),
):
    routes = tuple(
        RouteConfig(
            source,
            destination,
            "activation",
            "send_recv",
            "nvlink",
            "point_to_point",
            resource_id="attention-ffn-nvlink",
            backend="analytical_roofline_v1",
        )
        for source, destination in (
            ("attention", "ffn"),
            ("ffn", "attention"),
        )
    )
    return build_af_disagg_plan(
        DeploymentConfig(
            "af_disagg",
            (
                WorkerPoolConfig(
                    "attention",
                    "attention",
                    "nvidia_a800",
                    ParallelConfig(tensor_parallel_size=2),
                    ranks=attention_ranks,
                    collectives=_named_tp2_collectives("attention"),
                ),
                WorkerPoolConfig(
                    "ffn",
                    "ffn",
                    "nvidia_a800",
                    ParallelConfig(
                        tensor_parallel_size=2,
                        enable_expert_parallel=True,
                    ),
                    ranks=ffn_ranks,
                    collectives=_named_tp2_collectives("ffn"),
                ),
            ),
            routes,
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )


def _standalone_af_tp1_deployment(model):
    routes = tuple(
        RouteConfig(
            source,
            destination,
            "activation",
            "send_recv",
            "nvlink",
            "point_to_point",
            resource_id="attention-ffn-nvlink",
            backend="analytical_roofline_v1",
        )
        for source, destination in (
            ("attention", "ffn"),
            ("ffn", "attention"),
        )
    )
    return build_af_disagg_plan(
        DeploymentConfig(
            "af_disagg",
            (
                WorkerPoolConfig(
                    "attention", "attention", "nvidia_a800", ranks=(0,)),
                WorkerPoolConfig(
                    "ffn", "ffn", "nvidia_a800", ranks=(1,)),
            ),
            routes,
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )


def _uniform_pipeline_p2p(pipeline_size):
    return tuple(
        PipelineP2PConfig(
            source_stage,
            source_stage + 1,
            "send_recv",
            "nvlink",
            "point_to_point",
            "main-pipeline-fabric",
            "analytical_roofline_v1",
        )
        for source_stage in range(pipeline_size - 1)
    )


def _pp2_pipeline_p2p():
    return _uniform_pipeline_p2p(2)


def _pp3_pipeline_p2p():
    return _uniform_pipeline_p2p(3)


def _ep2_deployment(
    hardware_id="nvidia_a800",
    collectives=None,
    model=None,
):
    if model is None:
        model = load_model_config(
            "tests/serving/fixtures/qwen3_moe_ep2_scheduler")
    selected_collectives = (
        _tp2_collectives() if collectives is None else collectives)
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(
                    tensor_parallel_size=2,
                    enable_expert_parallel=True,
                ),
                ranks=(91, 7),
                collectives=selected_collectives,
            ),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    return model, deployment


def _qwen3_30b_a3b_test_model():
    """Keep the production MoE geometry with two layers for a fast test."""

    fixture = load_model_config(
        "tests/serving/fixtures/qwen3_moe_ep2_scheduler")
    return replace(
        fixture,
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=32,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=151936,
        max_position_embeddings=40960,
        num_experts=128,
        num_experts_per_tok=8,
        moe_intermediate_size=768,
    )


def _ep2_workload(
    step_id,
    scheduled_requests,
    expert_token_counts,
    *,
    hardware_id="nvidia_a800",
    collectives=None,
):
    model, deployment = _ep2_deployment(hardware_id, collectives)
    counts = tuple(
        expert_token_counts[expert_id]
        for expert_id in range(model.num_experts)
    )
    source_id = f"test-seeded-routing-{step_id}"
    schedule = ScheduleStep(
        step_id,
        "main",
        scheduled_requests,
        attention_kv_layout="paged_varlen",
    )
    for candidate_seed in range(10000):
        graph = build_workload(
            schedule,
            model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.seeded_assumption(
                    source_id, candidate_seed),
                "paged_varlen",
                "uniform",
            ),
        )
        emitted_counts = {
            tuple(node.attributes["tokens_per_expert"])
            for node in graph.nodes
            if node.kind is WorkloadKind.FUSED_MOE
        }
        if emitted_counts == {counts}:
            return model, deployment, graph
    raise AssertionError("test could not find a seeded routing realization")


@pytest.mark.parametrize(
    "hardware_id",
    ("metax_c500", "ascend_910b"),
)
def test_ep2_workload_supports_cross_hardware_emission(hardware_id):
    _, deployment, graph = _ep2_workload(
        "ep-cross-hardware",
        (ScheduledRequest("r0", "decode", 0, 1, 4),),
        {0: 1, 1: 0},
        hardware_id=hardware_id,
    )

    assert deployment.worker_pools["main"].hardware_id == hardware_id
    assert any(node.kind is WorkloadKind.FUSED_MOE for node in graph.nodes)


def test_ep2_lowering_preserves_explicit_protobridge_backend():
    protobridge_collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            "main-nvlink-fabric",
            "protobridge",
            "0" * 64,
        )
        for operation in ("all_reduce", "all_gather")
    )
    _, deployment, graph = _ep2_workload(
        "ep-protobridge",
        (ScheduledRequest("r0", "decode", 0, 1, 4),),
        {0: 1, 1: 0},
        collectives=protobridge_collectives,
    )

    plan = lower_workload(
        graph,
        deployment,
        {"nvidia_a800": HWSpec(
            "configs/hardware/nvidia_a800.yaml")},
    )

    assert any(
        call.kernel_spec.kind is KernelKind.COMMUNICATION
        and call.kernel_spec.transfer.backend == "protobridge"
        for call in plan.calls
    )


def test_ep2_lowering_rejects_graph_with_all_ep_nodes_removed():
    _, deployment, graph = _ep2_workload(
        "ep-truncated",
        (ScheduledRequest("r0", "decode", 0, 1, 4),),
        {0: 1, 1: 0},
    )
    removed = {
        node.node_id
        for node in graph.nodes
        if node.kind is WorkloadKind.FUSED_MOE
    }
    truncated = WorkloadGraph(
        tuple(node for node in graph.nodes if node.node_id not in removed),
        tuple(
            edge for edge in graph.edges
            if edge.source not in removed and edge.destination not in removed
        ),
    )

    with pytest.raises(ValueError, match="does not contain EP FUSED_MOE"):
        lower_workload(
            truncated,
            deployment,
            {"nvidia_a800": HWSpec(
                "configs/hardware/nvidia_a800.yaml")},
        )


def test_ep_scheduler_scope_admits_validated_rank_local_kernel_emission():
    model = load_model_config(
        "tests/serving/fixtures/qwen3_moe_ep2_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(
                    tensor_parallel_size=2,
                    enable_expert_parallel=True,
                ),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )

    validate_workload_emission_parallelism(
        deployment,
        model,
        allow_single_pool_tp2=True,
        allow_single_pool_ep2=True,
        allow_single_pool_tp1_pp=True,
        allow_single_pool_tp2_pp2=True,
    )


def test_ep2_workload_and_lowering_support_canonical_128_expert_geometry():
    fixture = load_model_config(
        "tests/serving/fixtures/qwen3_moe_ep2_scheduler")
    model = replace(
        fixture,
        num_experts=128,
        num_experts_per_tok=8,
    )
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(
                    tensor_parallel_size=2,
                    enable_expert_parallel=True,
                ),
                ranks=(0, 1),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    validate_workload_emission_parallelism(
        deployment,
        model,
        allow_single_pool_tp2=True,
        allow_single_pool_ep2=True,
        allow_single_pool_tp1_pp=True,
        allow_single_pool_tp2_pp2=True,
    )

    graph = build_workload(
        ScheduleStep(
            "canonical-ep2",
            "main",
            (ScheduledRequest("request", "prefill", 4, 0, 0),),
            attention_kv_layout="paged_varlen",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "canonical-qwen3-30b-a3b-routing-v1", 0),
            "paged_varlen",
            "uniform",
        ),
    )
    ep_nodes = tuple(
        node for node in graph.nodes if node.kind is WorkloadKind.FUSED_MOE)
    assert len(ep_nodes) == model.num_hidden_layers * 2
    assert all(len(node.attributes["owned_experts"]) == 64 for node in ep_nodes)

    plan = lower_workload(
        graph,
        deployment,
        {"nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml")},
    )
    assert any(
        call.kernel_spec.kind is KernelKind.FUSED_MOE for call in plan.calls)


def test_ep_scheduler_scope_rejects_non_qwen3_moe_model():
    model = load_model_config(
        "tests/serving/fixtures/glm4_moe_mtp_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(
                    tensor_parallel_size=2,
                    enable_expert_parallel=True,
                ),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )

    with pytest.raises(
            NotImplementedError,
            match="only the validated BF16 Qwen3MoE TP=EP=2"):
        validate_workload_emission_parallelism(
            deployment,
            model,
            allow_single_pool_tp2=True,
            allow_single_pool_ep2=True,
            allow_single_pool_tp1_pp=True,
            allow_single_pool_tp2_pp2=True,
        )


def test_llama_tp2_scheduler_scope_admits_rank_local_kernel_emission():
    model = load_model_config("tests/serving/fixtures/llama_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(tensor_parallel_size=2),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )

    validate_workload_emission_parallelism(
        deployment,
        model,
        allow_single_pool_tp2=True,
        allow_single_pool_ep2=True,
        allow_single_pool_tp1_pp=True,
        allow_single_pool_tp2_pp2=True,
    )

    drifted_model = replace(model, config_sha256="0" * 64)
    with pytest.raises(NotImplementedError, match="unsupported rank-local"):
        validate_workload_emission_parallelism(
            deployment,
            drifted_model,
            allow_single_pool_tp2=True,
            allow_single_pool_ep2=True,
            allow_single_pool_tp1_pp=True,
            allow_single_pool_tp2_pp2=True,
        )
    with pytest.raises(
            NotImplementedError, match="exact source-backed Llama fixture"):
        build_workload(
            ScheduleStep(
                "drifted-llama",
                "main",
                (ScheduledRequest("request", "prefill", 4, 0, 0),),
            ),
            drifted_model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.not_applicable(),
                "paged_varlen",
                "uniform",
            ),
        )
    with pytest.raises(
            NotImplementedError, match="Llama fixture in BF16"):
        build_workload(
            ScheduleStep(
                "fp16-llama",
                "main",
                (ScheduledRequest("request", "prefill", 4, 0, 0),),
            ),
            replace(model, compute_dtype="fp16", weight_dtype="fp16"),
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.not_applicable(),
                "paged_varlen",
                "uniform",
            ),
        )


@pytest.mark.parametrize(
    "scheduled_request,forward_scope,query_lengths,kv_lengths,"
    "expected_payload_bytes",
    (
        (
            ScheduledRequest("request", "prefill", 4, 0, 0),
            "prefill",
            (4,),
            (4,),
            1024,
        ),
        (
            ScheduledRequest("request", "decode", 0, 1, 2),
            "decode",
            (1,),
            (3,),
            256,
        ),
    ),
)
def test_llama_tp2_scope_admits_exact_standalone_af(
        scheduled_request,
        forward_scope,
        query_lengths,
        kv_lengths,
        expected_payload_bytes):
    model = load_model_config("tests/serving/fixtures/llama_scheduler")
    deployment = _standalone_af_tp2_deployment(model)
    validate_workload_emission_parallelism(
        deployment,
        model,
        allow_single_pool_tp2=False,
        allow_single_pool_ep2=False,
        allow_single_pool_tp1_pp=False,
        allow_single_pool_tp2_pp2=False,
    )
    graph = build_workload(
        ScheduleStep(
            "llama-af-tp2",
            "attention",
            (scheduled_request,),
            attention_kv_layout="paged_varlen",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.not_applicable(),
            "paged_varlen",
            "uniform",
        ),
    )
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    plan = lower_workload(
        graph,
        deployment,
        {"nvidia_a800": hardware},
    )
    reference = llama_af_tp2_forward_reference(
        Path("tests/serving/fixtures/llama_scheduler"),
        forward_scope=forward_scope,
        step_id="llama-af-tp2",
        query_lengths=query_lengths,
        kv_lengths=kv_lengths,
        attention_ranks=(91, 7),
        ffn_ranks=(18, 4),
        attention_resource_id="attention-nvlink-fabric",
        ffn_resource_id="ffn-nvlink-fabric",
        activation_resource_id="attention-ffn-nvlink",
    )
    report = compare_tp2_reference_to_emission(reference, graph, plan)

    assert report["validation_status"] == (
        "validated_llama_af_tp2_static_source_semantics_projected_latency"
    ), (report["mismatched"], report["topology_mismatched"])
    # Decode keeps the token-embedding gather plus its all-reduce; prefill
    # drops both (A800 measured-zero embedding law) and re-anchors the
    # first residual transfer.  Both scopes add the two
    # logits-boundary-gather rank nodes.
    if forward_scope == "decode":
        expected_nodes = 40
        expected_collectives = 12
    else:
        expected_nodes = 37
        expected_collectives = 11
    assert report["expected_node_count"] == expected_nodes
    assert report["actual_node_count"] == expected_nodes
    assert len(report["matched"]) == expected_nodes
    assert report["missing"] == ()
    assert report["extra"] == ()
    assert report["mismatched"] == ()
    assert report["topology_mismatched"] == ()
    communication_calls = tuple(
        call for call in plan.calls
        if call.kernel_spec.kind is KernelKind.COMMUNICATION
    )
    transfer_calls = tuple(
        call for call in communication_calls
        if call.kernel_spec.transfer.operation == "send_recv"
    )
    assert len(communication_calls) == expected_collectives
    assert len(transfer_calls) == 8
    if forward_scope == "decode":
        expected_participants = (
            (91, 18), (7, 4), (91, 18), (7, 4),
            (18, 91), (4, 7), (18, 91), (4, 7),
        )
    else:
        # Prefill emits both tensors of one lane together at the pool
        # transition (the residual transfer no longer precedes the lane's
        # compute because its embedding producer is suppressed).
        expected_participants = (
            (91, 18), (91, 18), (7, 4), (7, 4),
            (18, 91), (4, 7), (18, 91), (4, 7),
        )
    assert tuple(
        call.kernel_spec.transfer.participants for call in transfer_calls
    ) == expected_participants
    assert all(
        call.kernel_spec.transfer.bytes == expected_payload_bytes
        for call in transfer_calls
    )
    predictions = predict_kernel_plan(
        plan,
        {"nvidia_a800": hardware},
        method="midlayer",
    )
    timeline = build_execution_timeline(plan, predictions)
    assert set(predictions) == {call.call_id for call in plan.calls}
    assert {interval.call_id for interval in timeline.intervals} == set(
        predictions)


def test_llama_tp2_scope_rejects_drifted_standalone_af_fixture():
    model = load_model_config("tests/serving/fixtures/llama_scheduler")
    drifted_model = replace(model, config_sha256="0" * 64)
    deployment = _standalone_af_tp2_deployment(drifted_model)

    with pytest.raises(NotImplementedError, match="unsupported rank-local"):
        validate_workload_emission_parallelism(
            deployment,
            drifted_model,
            allow_single_pool_tp2=True,
            allow_single_pool_ep2=True,
            allow_single_pool_tp1_pp=True,
            allow_single_pool_tp2_pp2=True,
        )
    with pytest.raises(
            NotImplementedError,
            match="one colocated pool or the validated standalone AF"):
        build_workload(
            ScheduleStep(
                "drifted-llama-af-tp2",
                "attention",
                (ScheduledRequest("request", "prefill", 4, 0, 0),),
            ),
            drifted_model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.not_applicable(),
                "paged_varlen",
                "uniform",
            ),
        )


def test_llama_tp2_scope_rejects_ffn_owned_kv_cache():
    model = load_model_config("tests/serving/fixtures/llama_scheduler")
    ffn_kv_cache = KVCacheConfig(
        (KVCacheTierConfig("hbm", 4096, 16),),
        bytes_per_token=256,
    )
    deployment = _standalone_af_tp2_deployment(
        model,
        ffn_kv_cache=ffn_kv_cache,
    )

    with pytest.raises(NotImplementedError, match="unsupported rank-local"):
        validate_workload_emission_parallelism(
            deployment,
            model,
            allow_single_pool_tp2=True,
            allow_single_pool_ep2=True,
            allow_single_pool_tp1_pp=True,
            allow_single_pool_tp2_pp2=True,
        )


def test_tp2_pp2_scheduler_scope_admits_exact_rank_local_emission():
    model = load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(
                    tensor_parallel_size=2,
                    pipeline_parallel_size=2,
                ),
                ranks=(0, 1, 2, 3),
                collectives=_tp2_collectives(),
                pipeline_p2p=_pp2_pipeline_p2p(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )

    validate_workload_emission_parallelism(
        deployment,
        model,
        allow_single_pool_tp2=True,
        allow_single_pool_ep2=True,
        allow_single_pool_tp1_pp=True,
        allow_single_pool_tp2_pp2=True,
    )


@pytest.mark.parametrize("hardware_id", ("metax_c500", "ascend_910b"))
def test_qwen3_tp2_scheduler_and_workload_are_hardware_independent(
        hardware_id):
    model = load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(tensor_parallel_size=2),
                ranks=(0, 1),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )

    validate_workload_emission_parallelism(
        deployment,
        model,
        allow_single_pool_tp2=True,
        allow_single_pool_ep2=True,
        allow_single_pool_tp1_pp=True,
        allow_single_pool_tp2_pp2=True,
    )
    graph = build_workload(
        ScheduleStep(
            "cross-hardware-qwen3-tp2",
            "main",
            (ScheduledRequest("r", "decode", 0, 1, 2),),
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.not_applicable(),
            "paged_varlen", "uniform"),
    )

    assert {
        node.rank for node in graph.nodes
        if node.node_id.startswith(
            "cross-hardware-qwen3-tp2:layer-0:qkv:tp-rank-")
    } == {0, 1}
    assert all(
        node.attributes["participants"] == (0, 1)
        for node in graph.nodes
        if node.kind is WorkloadKind.COLLECTIVE
    )


def test_qwen3_fp16_tp2_scheduler_scope_is_dtype_preserving():
    model = load_model_config("configs/model/qwen3-32b-fp16.json")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "metax_c500",
                ParallelConfig(tensor_parallel_size=2),
                ranks=(0, 1),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )

    validate_workload_emission_parallelism(
        deployment,
        model,
        allow_single_pool_tp2=True,
        allow_single_pool_ep2=True,
        allow_single_pool_tp1_pp=True,
        allow_single_pool_tp2_pp2=True,
    )

    assert model.compute_dtype == "fp16"
    assert model.weight_dtype == "fp16"


@pytest.mark.parametrize("compute_dtype", ("bf16", "fp16"))
def test_qwen3_tp2_rank_local_workload_accepts_supported_dtypes(
        compute_dtype):
    model = replace(
        load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler"),
        compute_dtype=compute_dtype,
        weight_dtype=compute_dtype,
    )
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "metax_c500",
                ParallelConfig(tensor_parallel_size=2),
                ranks=(0, 1),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )

    graph = build_workload(
        ScheduleStep(
            f"qwen3-{compute_dtype}-tp2",
            "main",
            (ScheduledRequest("request", "prefill", 4, 0, 0),),
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.not_applicable(),
            "paged_varlen",
            "uniform",
        ),
    )

    assert {
        node.dtype for node in graph.nodes if node.dtype is not None
    } == {compute_dtype}


def test_qwen3_tp2_rank_local_workload_rejects_weight_dtype_mismatch():
    model = replace(
        load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler"),
        weight_dtype="mixed_int4_bf16",
        weight_group_size=128,
        quantized_weight_targets=("Linear",),
        quantized_weight_ignore=("lm_head",),
    )
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "metax_c500",
                ParallelConfig(tensor_parallel_size=2),
                ranks=(0, 1),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )

    with pytest.raises(
            NotImplementedError, match="quantized model weights"):
        build_workload(
            ScheduleStep(
                "qwen3-mismatched-weight-tp2",
                "main",
                (ScheduledRequest("request", "prefill", 4, 0, 0),),
            ),
            model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.not_applicable(),
                "paged_varlen",
                "uniform",
            ),
        )


@pytest.mark.parametrize("hardware_id", ("metax_c500", "ascend_910b"))
def test_llama_tp2_keeps_its_a800_source_boundary(hardware_id):
    model = load_model_config("tests/serving/fixtures/llama_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(tensor_parallel_size=2),
                ranks=(0, 1),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )

    with pytest.raises(NotImplementedError, match="unsupported rank-local"):
        validate_workload_emission_parallelism(
            deployment,
            model,
            allow_single_pool_tp2=True,
            allow_single_pool_ep2=True,
            allow_single_pool_tp1_pp=True,
            allow_single_pool_tp2_pp2=True,
        )
    with pytest.raises(
            NotImplementedError,
            match="non-Qwen3 models currently requires nvidia_a800"):
        build_workload(
            ScheduleStep(
                "cross-hardware-llama-tp2",
                "main",
                (ScheduledRequest("r", "decode", 0, 1, 2),),
            ),
            model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.not_applicable(),
                "paged_varlen", "uniform"),
        )


@pytest.mark.parametrize(
    "step_id,scheduled_requests,global_counts,expected_local_counts",
    (
        (
            "ep-balanced",
            (
                ScheduledRequest("r0", "decode", 0, 1, 4),
                ScheduledRequest("r1", "decode", 0, 1, 4),
            ),
            {0: 1, 1: 1},
            ((1, 0), (0, 1)),
        ),
        (
            "ep-empty-shard",
            (ScheduledRequest("r0", "decode", 0, 1, 4),),
            {0: 1, 1: 0},
            ((1, 0), (0, 0)),
        ),
    ),
)
def test_ep2_workload_preserves_full_input_and_local_routed_rows(
        step_id, scheduled_requests, global_counts, expected_local_counts):
    model, deployment, graph = _ep2_workload(
        step_id,
        scheduled_requests,
        global_counts,
    )
    token_count = len(scheduled_requests)
    expected_global_counts = tuple(
        global_counts[index] for index in range(model.num_experts))

    for layer in range(model.num_hidden_layers):
        invocation_id = f"{step_id}:layer-{layer}:moe"
        router_nodes = tuple(
            graph.node(f"{step_id}:layer-{layer}:router:tp-rank-{rank}")
            for rank in range(2)
        )
        assert tuple(node.rank for node in router_nodes) == (91, 7)
        assert all(
            dict(node.shape) == {
                "b": 1,
                "m": token_count,
                "n": model.num_experts,
                "k": model.hidden_size,
            }
            for node in router_nodes
        )
        moe_nodes = tuple(
            graph.node(f"{invocation_id}:ep-rank-{rank}")
            for rank in range(2)
        )
        assert tuple(node.rank for node in moe_nodes) == (91, 7)
        assert all(node.shape["tokens"] == token_count for node in moe_nodes)
        assert all(
            tuple(node.attributes["tokens_per_expert"])
            == expected_global_counts
            for node in moe_nodes
        )
        assert tuple(
            tuple(node.attributes["local_tokens_per_expert"])
            for node in moe_nodes
        ) == expected_local_counts
        assert tuple(
            tuple(node.attributes["owned_experts"])
            for node in moe_nodes
        ) == ((0,), (1,))
        assert tuple(
            graph.predecessors(node.node_id) for node in moe_nodes
        ) == (
            (f"{step_id}:layer-{layer}:router:tp-rank-0",),
            (f"{step_id}:layer-{layer}:router:tp-rank-1",),
        )
        collective = graph.node(
            f"{invocation_id}:ep-output-all-reduce")
        assert collective.attributes["operation"] == "all_reduce"
        assert collective.attributes["participants"] == (91, 7)
        assert collective.attributes["bytes"] == (
            token_count * model.hidden_size * model.dtype_bytes)
        assert set(graph.predecessors(collective.node_id)) == {
            node.node_id for node in moe_nodes}

    assert all(
        node.kind is not WorkloadKind.COLLECTIVE
        or node.attributes["operation"] != "all_to_all"
        for node in graph.nodes
    )
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    plan = lower_workload(
        graph,
        deployment,
        {"nvidia_a800": hardware},
    )
    predictions = predict_kernel_plan(
        plan, {"nvidia_a800": hardware}, method="midlayer")
    assert set(predictions) == {call.call_id for call in plan.calls}

    for layer in range(model.num_hidden_layers):
        invocation_id = f"{step_id}:layer-{layer}:moe"
        final_rank_calls = []
        for expert_rank, expected_counts in enumerate(expected_local_counts):
            node_id = f"{invocation_id}:ep-rank-{expert_rank}"
            calls = tuple(
                call for call in plan.calls
                if call.source_node_ids == (node_id,)
            )
            assert tuple(
                call.kernel_spec.variant for call in calls
            ) == (
                "moe_topk",
                "moe_align_small_batch",
                "w1",
                "moe_act_and_mul",
                "w2",
                "moe_reduce",
            )
            assert all(
                call.kernel_spec.shape["tokens"] == token_count
                for call in calls)
            projections = tuple(
                call for call in calls
                if call.kernel_spec.kind is KernelKind.FUSED_MOE)
            assert tuple(
                call.kernel_spec.variant for call in projections
            ) == ("w1", "w2")
            assert all(
                tuple(call.kernel_spec.attributes["tokens_per_expert"])
                == expected_global_counts
                for call in projections)
            assert all(
                tuple(call.kernel_spec.attributes[
                    "local_tokens_per_expert"])
                == expected_counts
                for call in projections)
            if sum(expected_counts) == 0:
                for call in projections:
                    resource_input, compulsory = create_predictor(
                        KernelKind.FUSED_MOE).resource_input(
                            call.kernel_spec, hardware)
                    assert compulsory > 0
                    assert resource_input.demand_per_resource[
                        "compute_tensor"] == 0
                    assert resource_input.demand_per_resource[
                        "l2_cache"] == compulsory
                    assert resource_input.demand_per_resource[
                        "dram"] == compulsory
            final_rank_calls.append(calls[-1].call_id)

        collective_id = f"{invocation_id}:ep-output-all-reduce"
        collective_call = next(
            call for call in plan.calls if call.call_id == collective_id)
        assert collective_call.dependencies == tuple(final_rank_calls)
        assert collective_call.kernel_spec.transfer.operation == "all_reduce"
        assert collective_call.kernel_spec.transfer.participants == (91, 7)


@pytest.mark.parametrize(
    "hardware_id", ("nvidia_a800", "metax_c500", "ascend_910b"))
def test_qwen3_30b_a3b_ep2_emits_deterministic_local_expert_work(
        hardware_id):
    model = _qwen3_30b_a3b_test_model()
    _, deployment = _ep2_deployment(hardware_id, model=model)
    validate_workload_emission_parallelism(
        deployment,
        model,
        allow_single_pool_tp2=True,
        allow_single_pool_ep2=True,
        allow_single_pool_tp1_pp=True,
        allow_single_pool_tp2_pp2=True,
    )
    step = ScheduleStep(
        "qwen3-30b-a3b",
        "main",
        (ScheduledRequest("request", "prefill", 4, 0, 0),),
        attention_kv_layout="paged_varlen",
    )

    def emit():
        return build_workload(
            step,
            model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.seeded_assumption(
                    "qwen3_30b_a3b_ep2_test_v1", 19),
                "paged_varlen",
                "uniform",
            ),
        )

    graph = emit()
    replay = emit()
    assert deployment.worker_pools["main"].parallel_plan.expert_to_shard == (
        (0,) * 64 + (1,) * 64)
    assert deployment.worker_pools["main"].parallel_plan.communication_rules == (
        "row_parallel_all_reduce",
        "vocab_parallel_all_gather",
        "expert_output_all_reduce",
    )

    for layer_index in range(model.num_hidden_layers):
        invocation_id = f"qwen3-30b-a3b:layer-{layer_index}:moe"
        moe_nodes = tuple(
            graph.node(f"{invocation_id}:ep-rank-{expert_rank}")
            for expert_rank in range(2)
        )
        replay_nodes = tuple(
            replay.node(f"{invocation_id}:ep-rank-{expert_rank}")
            for expert_rank in range(2)
        )
        global_counts = tuple(moe_nodes[0].attributes["tokens_per_expert"])
        assert len(global_counts) == 128
        assert sum(global_counts) == 4 * 8
        assert all(
            tuple(node.attributes["tokens_per_expert"]) == global_counts
            for node in moe_nodes
        )
        assert tuple(
            tuple(node.attributes["tokens_per_expert"])
            for node in moe_nodes
        ) == tuple(
            tuple(node.attributes["tokens_per_expert"])
            for node in replay_nodes
        )
        assert tuple(
            tuple(node.attributes["owned_experts"])
            for node in moe_nodes
        ) == (tuple(range(64)), tuple(range(64, 128)))
        for expert_rank, node in enumerate(moe_nodes):
            local_counts = tuple(
                node.attributes["local_tokens_per_expert"])
            assert local_counts == tuple(
                count if owner == expert_rank else 0
                for count, owner in zip(
                    global_counts,
                    deployment.worker_pools[
                        "main"].parallel_plan.expert_to_shard,
                )
            )
            assert graph.predecessors(node.node_id) == (
                f"qwen3-30b-a3b:layer-{layer_index}:router:"
                f"tp-rank-{expert_rank}",
            )
        aggregation = graph.node(
            f"{invocation_id}:ep-output-all-reduce")
        assert aggregation.attributes["operation"] == "all_reduce"
        assert aggregation.attributes["participants"] == (91, 7)
        assert set(graph.predecessors(aggregation.node_id)) == {
            node.node_id for node in moe_nodes}

    assert all(
        node.kind is not WorkloadKind.COLLECTIVE
        or node.attributes["operation"] != "all_to_all"
        for node in graph.nodes
    )


def test_ascend_ep2_omits_grouped_matmul_for_zero_routed_rank():
    model = _qwen3_30b_a3b_test_model()
    _, deployment = _ep2_deployment("ascend_910b", model=model)
    graph = build_workload(
        ScheduleStep(
            "empty-rank",
            "main",
            (ScheduledRequest("request", "prefill", 1, 0, 0),),
            attention_kv_layout="paged_varlen",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test-empty-rank", 76),
            "paged_varlen",
            "uniform",
        ),
    )
    zero_rank_nodes = tuple(
        node for node in graph.nodes
        if (node.kind is WorkloadKind.FUSED_MOE
            and sum(node.attributes["local_tokens_per_expert"]) == 0)
    )
    assert zero_rank_nodes

    plan = lower_workload(
        graph,
        deployment,
        {"ascend_910b": HWSpec("configs/hardware/ascend_910b.yaml")},
    )
    for node in zero_rank_nodes:
        calls = tuple(
            call for call in plan.calls
            if call.source_node_ids == (node.node_id,)
        )
        assert calls
        assert all(
            call.kernel_spec.kind is not KernelKind.FUSED_MOE
            for call in calls
        )
        assert all(
            call.kernel_spec.variant != "moe_swiglu"
            for call in calls
        )
        collective_id = (
            f"{node.attributes['moe_invocation_id']}:ep-output-all-reduce")
        collective = next(
            call for call in plan.calls if call.call_id == collective_id)
        assert calls[-1].call_id in collective.dependencies


def test_qwen3_30b_a3b_ep2_lowers_rank_local_a800_fused_moe():
    model = _qwen3_30b_a3b_test_model()
    _, deployment = _ep2_deployment(model=model)
    graph = build_workload(
        ScheduleStep(
            "qwen3-30b-a3b-lower",
            "main",
            (ScheduledRequest("request", "decode", 0, 1, 8),),
            attention_kv_layout="paged_varlen",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "qwen3_30b_a3b_ep2_lowering_v1", 23),
            "paged_varlen",
            "uniform",
        ),
    )
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    plan = lower_workload(
        graph,
        deployment,
        {"nvidia_a800": hardware},
    )

    for layer_index in range(model.num_hidden_layers):
        invocation_id = f"qwen3-30b-a3b-lower:layer-{layer_index}:moe"
        global_counts = tuple(
            graph.node(
                f"{invocation_id}:ep-rank-0"
            ).attributes["tokens_per_expert"]
        )
        final_rank_calls = []
        for expert_rank in range(2):
            node_id = f"{invocation_id}:ep-rank-{expert_rank}"
            node = graph.node(node_id)
            calls = tuple(
                call for call in plan.calls
                if call.source_node_ids == (node_id,)
            )
            assert tuple(call.kernel_spec.variant for call in calls) == (
                "moe_topk",
                "moe_align",
                "moe_count_and_sort",
                "w1",
                "moe_act_and_mul",
                "w2",
                "moe_reduce",
            )
            projections = tuple(
                call for call in calls
                if call.kernel_spec.kind is KernelKind.FUSED_MOE
            )
            assert tuple(
                call.kernel_spec.attributes["owned_experts"]
                for call in projections
            ) == (node.attributes["owned_experts"],) * 2
            # Lowering emits the canonical member of the node routing's
            # physical class: the shard keeps its populated counts in launch
            # order and its empty experts move behind them, so rows, shard
            # ownership and tokens*topk are conserved exactly.
            canonical_counts, canonical_local = canonical_expert_axis(
                global_counts, node.attributes["owned_experts"])
            assert sum(canonical_counts) == sum(global_counts)
            assert sorted(canonical_counts) == sorted(global_counts)
            assert canonical_local == tuple(
                count if expert in node.attributes["owned_experts"] else 0
                for expert, count in enumerate(canonical_counts)
            )
            assert sorted(canonical_local) == sorted(
                node.attributes["local_tokens_per_expert"])
            assert all(
                tuple(call.kernel_spec.attributes["tokens_per_expert"])
                == canonical_counts
                for call in projections
            )
            assert all(
                tuple(call.kernel_spec.attributes["local_tokens_per_expert"])
                == canonical_local
                for call in projections
            )
            final_rank_calls.append(calls[-1].call_id)
        aggregation_id = f"{invocation_id}:ep-output-all-reduce"
        aggregation = next(
            call for call in plan.calls if call.call_id == aggregation_id)
        assert aggregation.dependencies == tuple(final_rank_calls)
        assert aggregation.kernel_spec.transfer.operation == "all_reduce"

    predictions = predict_kernel_plan(
        plan, {"nvidia_a800": hardware}, method="midlayer")
    assert set(predictions) == {call.call_id for call in plan.calls}


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("local_count", "local routing counts disagree"),
        ("ownership", "owned_experts disagrees"),
        ("declared_ep_size", "disagrees with the deployment plan"),
        ("missing_zero_shard", "requires one node per rank"),
        ("missing_collective_predecessor", "wait for every expert rank"),
    ),
)
def test_ep2_lowering_rejects_forged_workload_contract(mutation, match):
    _, deployment, graph = _ep2_workload(
        "ep-forged",
        (ScheduledRequest("r0", "decode", 0, 1, 4),),
        {0: 1, 1: 0},
    )
    target_id = "ep-forged:layer-0:moe:ep-rank-1"
    collective_id = "ep-forged:layer-0:moe:ep-output-all-reduce"
    nodes = list(graph.nodes)
    edges = list(graph.edges)
    target = graph.node(target_id)
    if mutation == "local_count":
        attributes = dict(target.attributes)
        attributes["local_tokens_per_expert"] = (1, 0)
        nodes[nodes.index(target)] = replace(target, attributes=attributes)
    elif mutation == "ownership":
        attributes = dict(target.attributes)
        attributes["owned_experts"] = (0,)
        nodes[nodes.index(target)] = replace(target, attributes=attributes)
    elif mutation == "declared_ep_size":
        nodes = [
            replace(
                node,
                attributes={
                    **dict(node.attributes),
                    "expert_parallel_size": 1,
                },
            )
            if node.kind is WorkloadKind.FUSED_MOE else node
            for node in nodes
        ]
    elif mutation == "missing_zero_shard":
        nodes.remove(target)
        edges = [
            edge for edge in edges
            if edge.source != target_id and edge.destination != target_id
        ]
    elif mutation == "missing_collective_predecessor":
        edges = [
            edge for edge in edges
            if not (
                edge.source == target_id
                and edge.destination == collective_id
            )
        ]
    else:
        raise AssertionError(f"unknown EP contract mutation: {mutation!r}")
    forged = WorkloadGraph(nodes, edges)

    with pytest.raises((ValueError, NotImplementedError), match=match):
        lower_workload(
            forged,
            deployment,
            {"nvidia_a800": HWSpec(
                "configs/hardware/nvidia_a800.yaml")},
        )


def test_ep2_lowering_rejects_duplicate_expert_routing_histogram():
    _, deployment, graph = _ep2_workload(
        "ep-duplicate-route",
        (ScheduledRequest("r0", "prefill", 2, 0, 0),),
        {0: 1, 1: 1},
    )
    nodes = []
    for node in graph.nodes:
        if node.kind is not WorkloadKind.FUSED_MOE:
            nodes.append(node)
            continue
        expert_rank = node.attributes["expert_rank"]
        attributes = {
            **dict(node.attributes),
            "tokens_per_expert": (3, 1),
            "local_tokens_per_expert": (
                (3, 0) if expert_rank == 0 else (0, 1)),
        }
        nodes.append(replace(
            node,
            shape={**dict(node.shape), "topk": 2},
            attributes=attributes,
        ))

    with pytest.raises(ValueError, match="cannot exceed input tokens"):
        lower_workload(
            WorkloadGraph(nodes, graph.edges),
            deployment,
            {"nvidia_a800": HWSpec(
                "configs/hardware/nvidia_a800.yaml")},
        )


def test_tp_qkv_shape_preserves_replicated_kv_heads_through_lowering():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    model = replace(
        load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler"),
        num_key_value_heads=1,
    )
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(tensor_parallel_size=4),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    m, n, k = infer_gemm_dimensions(
        model, "qkv_proj", 3, tp_size=4)
    qkv = WorkloadNode(
        "tp4-replicated-kv:qkv:tp-rank-0",
        WorkloadKind.LINEAR,
        "main",
        {"b": 1, "m": m, "n": n, "k": k},
        dtype="bf16",
        variant="gemm",
        rank=0,
        attributes=_test_gemm_attributes(
            "qkv_proj", tensor_parallel_size=4),
    )
    # vLLM QKVParallelLinear shards four Q heads over four ranks but
    # replicates the single KV head: (1 Q + 2 * 1 KV) * 64 = 192.
    assert qkv.shape == {"b": 1, "m": 3, "n": 192, "k": 256}

    plan = lower_workload(
        WorkloadGraph((qkv,)),
        deployment,
        {"nvidia_a800": hardware},
    )
    qkv_spec = next(
        call.kernel_spec for call in plan.calls
        if call.kernel_spec.kind is KernelKind.GEMM
    )
    assert qkv_spec.shape == {
        "b": 1,
        "m": 3,
        "n": 192,
        "k": 256,
        "dtype_bytes": 2,
    }


def test_tp_qkv_shape_supports_regular_kv_head_sharding():
    model = load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler")

    # Q=4, KV=2, TP=2: (2 local Q + 2 * 1 local KV) * 64 = 256.
    assert infer_gemm_dimensions(
        model, "qkv_proj", 3, tp_size=2) == (3, 256, 256)


def test_tp_lm_head_uses_pinned_vllm_padded_vocab_partition():
    model = load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler")

    assert padded_vocab_size(64) == 64
    assert padded_vocab_size(65) == 128
    assert infer_gemm_dimensions(
        model, "lm_head", 1, tp_size=2) == (1, 32, 256)

    with pytest.raises(ValueError, match="divisible by tp_size"):
        infer_gemm_dimensions(model, "lm_head", 1, tp_size=3)


def test_tp2_pp2_reference_pins_lane_sends_and_destination_all_gathers():
    reference = qwen3_tp2_pp2_forward_reference(
        Path("tests/serving/fixtures/qwen3_tp2_scheduler"),
        forward_scope="prefill",
        step_id="tp2-pp2-reference",
        query_lengths=(2,),
        kv_lengths=(2,),
        physical_rank_groups=((91, 7), (18, 4)),
        tensor_resource_id="main-nvlink-fabric",
        pipeline_resource_id="main-pipeline-fabric",
    )
    p2p = tuple(
        item for item in reference["nodes"]
        if item["workload_kind"] == "p2p"
    )
    boundary_collectives = tuple(
        item for item in reference["nodes"]
        if (item["workload_kind"] == "collective"
            and ":pp-0-to-1:" in item["node_id"])
    )

    assert reference["validation_scope"] == (
        "qwen3_bf16_tp2_pp2_vllm_v1")
    assert reference["physical_tensor_sends_per_step"] == 4
    assert reference["destination_all_gathers_per_step"] == 2
    assert reference["pipeline_wire_elements_per_rank"] == 256
    assert reference["pipeline_wire_bytes_per_rank"] == 512
    assert tuple(
        item["workload_attributes"]["participants"] for item in p2p
    ) == ((91, 18), (7, 4), (91, 18), (7, 4))
    assert all(
        item["workload_attributes"]["payload_tensor"]["shape"] == (256,)
        for item in p2p)
    assert all(
        item["workload_attributes"]["participants"] == (18, 4)
        for item in boundary_collectives)
    for item in boundary_collectives:
        expected_p2p = tuple(
            candidate["node_id"] for candidate in p2p
            if candidate["workload_attributes"]["payload_tensor"]["name"]
            == item["node_id"].split(":")[-2].replace("-", "_")
        )
        actual_predecessors = tuple(
            source for source, destination, kind in reference["edges"]
            if destination == item["node_id"] and kind == "barrier"
        )
        assert actual_predecessors == expected_p2p


@pytest.mark.parametrize(
    "scheduled_request,expected_wire_elements,expected_wire_bytes",
    (
        (ScheduledRequest("r", "prefill", 2, 0, 0), 256, 512),
        (ScheduledRequest("r", "decode", 0, 1, 2), 128, 256),
    ),
)
def test_tp2_pp2_qwen3_emits_stage_local_compute_and_exact_boundary(
    scheduled_request,
    expected_wire_elements,
    expected_wire_bytes,
):
    model = load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(
                    tensor_parallel_size=2,
                    pipeline_parallel_size=2,
                ),
                ranks=(91, 7, 18, 4),
                collectives=_tp2_collectives(),
                pipeline_p2p=_pp2_pipeline_p2p(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    graph = build_workload(
        ScheduleStep(
            "tp2-pp2",
            "main",
            (scheduled_request,),
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.not_applicable(),
            "paged_varlen",
            "uniform",
        ),
    )

    stage0_compute = tuple(
        node for node in graph.nodes
        if (node.node_id.startswith("tp2-pp2:token-embedding:")
            or node.node_id.startswith("tp2-pp2:layer-0:"))
        and node.kind not in (WorkloadKind.COLLECTIVE, WorkloadKind.P2P)
    )
    stage1_compute = tuple(
        node for node in graph.nodes
        if (node.node_id.startswith("tp2-pp2:layer-1:")
            or node.node_id.startswith("tp2-pp2:final-norm:")
            or node.node_id.startswith("tp2-pp2:lm-head:"))
        and node.kind not in (WorkloadKind.COLLECTIVE, WorkloadKind.P2P)
    )
    assert {node.rank for node in stage0_compute} == {91, 7}
    assert {node.rank for node in stage1_compute} == {18, 4}

    p2p = tuple(
        node for node in graph.nodes if node.kind is WorkloadKind.P2P)
    assert tuple(node.node_id for node in p2p) == (
        "tp2-pp2:pp-0-to-1:hidden-states:tp-rank-0",
        "tp2-pp2:pp-0-to-1:hidden-states:tp-rank-1",
        "tp2-pp2:pp-0-to-1:residual:tp-rank-0",
        "tp2-pp2:pp-0-to-1:residual:tp-rank-1",
    )
    assert tuple(node.attributes["participants"] for node in p2p) == (
        (91, 18), (7, 4), (91, 18), (7, 4),
    )
    assert all(node.attributes["bytes"] == expected_wire_bytes for node in p2p)
    assert all(node.attributes["payload_tensor"]["shape"] == (
        expected_wire_elements,)
               for node in p2p)
    stage0_frontier = "tp2-pp2:layer-0:down:tp-all-reduce"
    assert all(graph.predecessors(node.node_id) == (stage0_frontier,)
               for node in p2p)

    boundary_collectives = tuple(
        node for node in graph.nodes
        if (node.kind is WorkloadKind.COLLECTIVE
            and ":pp-0-to-1:" in node.node_id)
    )
    assert tuple(node.node_id for node in boundary_collectives) == (
        "tp2-pp2:pp-0-to-1:hidden-states:tp-all-gather",
        "tp2-pp2:pp-0-to-1:residual:tp-all-gather",
    )
    assert all(node.rank == 18 for node in boundary_collectives)
    assert all(node.attributes["participants"] == (18, 4)
               for node in boundary_collectives)
    assert all(node.attributes["bytes"] == expected_wire_bytes
               for node in boundary_collectives)
    hidden_collective, residual_collective = boundary_collectives
    assert graph.predecessors(hidden_collective.node_id) == tuple(
        node.node_id for node in p2p[:2])
    assert graph.predecessors(residual_collective.node_id) == tuple(
        node.node_id for node in p2p[2:])
    assert graph.predecessors(
        "tp2-pp2:layer-1:input-norm:tp-rank-0") == (
            hidden_collective.node_id,
            residual_collective.node_id,
        )
    lm_head_collective = graph.node(
        "tp2-pp2:lm-head:tp-all-gather")
    assert lm_head_collective.rank == 18
    assert lm_head_collective.attributes["participants"] == (18, 4)

    plan = lower_workload(
        graph,
        deployment,
        {"nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml")},
    )
    reference = qwen3_tp2_pp2_forward_reference(
        Path("tests/serving/fixtures/qwen3_tp2_scheduler"),
        forward_scope=scheduled_request.phase,
        step_id="tp2-pp2",
        query_lengths=(scheduled_request.scheduled_tokens,),
        kv_lengths=(
            scheduled_request.computed_tokens
            + scheduled_request.scheduled_tokens,
        ),
        physical_rank_groups=((91, 7), (18, 4)),
        tensor_resource_id="main-nvlink-fabric",
        pipeline_resource_id="main-pipeline-fabric",
    )
    report = compare_tp2_reference_to_emission(reference, graph, plan)
    source = pinned_qwen3_pp_source_evidence()
    assert report["validation_status"] == (
        "validated_tp2_pp2_static_source_semantics_projected_latency"), report
    # Prefill drops the token-embedding rank pair plus its all-reduce
    # (A800 measured-zero observation) and adds the two
    # logits-boundary-gather rank nodes; decode keeps the embedding.
    expected_reference_nodes = (
        65 if scheduled_request.phase == "prefill" else 68)
    assert report["expected_node_count"] == expected_reference_nodes
    assert report["actual_node_count"] == expected_reference_nodes
    assert len(report["matched"]) == expected_reference_nodes
    assert report["missing"] == ()
    assert report["extra"] == ()
    assert report["mismatched"] == ()
    assert report["topology_mismatched"] == ()
    assert report["node_coverage"] == 1.0
    assert source["pipeline_intermediate_tensor_fields"] == (
        "hidden_states", "residual")
    assert source["pipeline_tp_wire_partition"] == (
        "reshape(tp_world_size,-1)[tp_rank]")
    assert source["pipeline_tp_receiver_reassembly"] == (
        "tp_group.all_gather(dim=0)->original_shape")
    assert source["pipeline_residual_all_gather_when_sp_disabled"] is True

    predictions = predict_kernel_plan(
        plan,
        {"nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml")},
        method="midlayer",
    )
    timeline = build_execution_timeline(plan, predictions)
    intervals = {
        interval.call_id: interval for interval in timeline.intervals
    }
    for left, right in zip(p2p, p2p[1:]):
        assert intervals[right.node_id].start_time_s >= intervals[
            left.node_id].end_time_s
    assert intervals[hidden_collective.node_id].start_time_s >= max(
        intervals[p2p[0].node_id].end_time_s,
        intervals[p2p[1].node_id].end_time_s,
    )
    assert intervals[residual_collective.node_id].start_time_s >= max(
        intervals[p2p[2].node_id].end_time_s,
        intervals[p2p[3].node_id].end_time_s,
    )
    first_stage1 = intervals[
        "tp2-pp2:layer-1:input-norm:tp-rank-0"]
    assert first_stage1.start_time_s >= max(
        intervals[hidden_collective.node_id].end_time_s,
        intervals[residual_collective.node_id].end_time_s,
    )

    target = p2p[0]
    crossed_lane = {
        **target.attributes,
        "participants": (91, 4),
        "destination_rank": 4,
    }
    matrix_shaped_payload = {
        **target.attributes,
        "payload_tensor": {
            **target.attributes["payload_tensor"],
            "shape": (128, 2),
        },
    }
    wrong_wire_bytes = {
        **target.attributes,
        "bytes": expected_wire_bytes // 2,
        "payload_tensor": {
            **target.attributes["payload_tensor"],
            "shape": (expected_wire_elements // 2,),
            "bytes": expected_wire_bytes // 2,
        },
    }
    for replacement, message in (
        (crossed_lane, "disagrees with deployment"),
        (matrix_shaped_payload, "shape rank disagrees"),
        (wrong_wire_bytes, "wire shape/bytes disagree"),
    ):
        forged = WorkloadGraph(
            tuple(
                replace(node, attributes=replacement)
                if node.node_id == target.node_id else node
                for node in graph.nodes
            ),
            graph.edges,
        )
        with pytest.raises((TypeError, ValueError), match=message):
            lower_workload(
                forged,
                deployment,
                {"nvidia_a800": HWSpec(
                    "configs/hardware/nvidia_a800.yaml")},
            )

    removed_lane_id = p2p[1].node_id
    missing_lane = WorkloadGraph(
        tuple(node for node in graph.nodes if node.node_id != removed_lane_id),
        tuple(
            edge for edge in graph.edges
            if edge.source != removed_lane_id
            and edge.destination != removed_lane_id
        ),
    )
    with pytest.raises(ValueError, match="exactly four P2P lane nodes"):
        lower_workload(
            missing_lane,
            deployment,
            {"nvidia_a800": HWSpec(
                "configs/hardware/nvidia_a800.yaml")},
        )

    destination_norm_ids = {
        "tp2-pp2:layer-1:input-norm:tp-rank-0",
        "tp2-pp2:layer-1:input-norm:tp-rank-1",
    }
    boundary_collective_ids = {
        hidden_collective.node_id,
        residual_collective.node_id,
    }
    coherently_shrunk_nodes = []
    for node in graph.nodes:
        if node.node_id in destination_norm_ids:
            coherently_shrunk_nodes.append(replace(
                node,
                shape={
                    **node.shape,
                    "hidden": node.shape["hidden"] // 2,
                },
            ))
        elif node.kind is WorkloadKind.P2P:
            coherently_shrunk_nodes.append(replace(
                node,
                attributes={
                    **node.attributes,
                    "bytes": expected_wire_bytes // 2,
                    "payload_tensor": {
                        **node.attributes["payload_tensor"],
                        "shape": (expected_wire_elements // 2,),
                        "bytes": expected_wire_bytes // 2,
                    },
                },
            ))
        elif node.node_id in boundary_collective_ids:
            coherently_shrunk_nodes.append(replace(
                node,
                attributes={
                    **node.attributes,
                    "bytes": expected_wire_bytes // 2,
                },
            ))
        else:
            coherently_shrunk_nodes.append(node)
    coherently_shrunk = WorkloadGraph(
        tuple(coherently_shrunk_nodes),
        graph.edges,
    )
    with pytest.raises(
            ValueError, match="norm shape disagrees with.*QKV successor"):
        lower_workload(
            coherently_shrunk,
            deployment,
            {"nvidia_a800": HWSpec(
                "configs/hardware/nvidia_a800.yaml")},
        )

    orphan_all_gather = replace(
        hidden_collective,
        node_id="tp2-pp2:pp-0-to-1:orphan:tp-all-gather",
    )
    with pytest.raises(
            ValueError, match="exactly two boundary all-gathers"):
        lower_workload(
            WorkloadGraph(graph.nodes + (orphan_all_gather,), graph.edges),
            deployment,
            {"nvidia_a800": HWSpec(
                "configs/hardware/nvidia_a800.yaml")},
        )


@pytest.mark.parametrize(
    "scheduled_request,expected_tokens,expected_tensor_bytes",
    (
        (ScheduledRequest("r", "prefill", 2, 0, 0), 2, 256),
        (ScheduledRequest("r", "decode", 0, 1, 2), 1, 128),
    ),
)
def test_uniform_pp4_qwen3_emits_every_ordered_tensor_boundary(
    scheduled_request,
    expected_tokens,
    expected_tensor_bytes,
):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    fixture = Path("tests/serving/fixtures/qwen3_pp2_scheduler")
    model = load_model_config(str(fixture))
    physical_ranks = (11, 7, 29, 31)
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(pipeline_parallel_size=4),
                ranks=physical_ranks,
                pipeline_p2p=_uniform_pipeline_p2p(4),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    validate_workload_emission_parallelism(
        deployment,
        model,
        allow_single_pool_tp2=True,
        allow_single_pool_ep2=True,
        allow_single_pool_tp1_pp=True,
        allow_single_pool_tp2_pp2=True,
    )
    graph = build_workload(
        ScheduleStep("pp4", "main", (scheduled_request,)),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen",
            "uniform",
        ),
    )

    # A800 prefill carries the authorized measured-zero token-embedding
    # emission observation (the step-head gather is a launch-placement
    # divergence at the vendor dispatch layer), while decode/mixed emit it;
    # the §2.3 logits boundary gather is emitted in both phases.
    assert not tuple(
        node for node in graph.nodes if node.kind is WorkloadKind.COLLECTIVE)
    assert {node.rank for node in graph.nodes} == set(physical_ranks)
    for stage, rank in enumerate(physical_ranks):
        layer_nodes = tuple(
            node for node in graph.nodes
            if node.node_id.startswith(f"pp4:layer-{stage}:")
        )
        assert layer_nodes
        assert {node.rank for node in layer_nodes} == {rank}
    if scheduled_request.phase == "prefill":
        assert len(graph.nodes) == 57
        assert not any(
            node.kind is WorkloadKind.GATHER and node.variant == "embedding"
            for node in graph.nodes)
    else:
        assert len(graph.nodes) == 58
        assert graph.node("pp4:token-embedding").rank == physical_ranks[0]
    assert graph.node("pp4:final-norm").rank == physical_ranks[-1]
    assert graph.node("pp4:lm-head").rank == physical_ranks[-1]

    p2p_nodes = tuple(
        node for node in graph.nodes if node.kind is WorkloadKind.P2P)
    expected_ids = tuple(
        f"pp4:pp-{source}-to-{source + 1}:{tensor}"
        for source in range(3)
        for tensor in ("hidden-states", "residual")
    )
    assert tuple(node.node_id for node in p2p_nodes) == expected_ids
    assert tuple(
        (node.attributes["source_rank"],
         node.attributes["destination_rank"])
        for node in p2p_nodes
    ) == tuple(
        pair
        for source, destination in zip(physical_ranks, physical_ranks[1:])
        for pair in ((source, destination), (source, destination))
    )
    assert tuple(
        node.attributes["payload_tensor"]["name"]
        for node in p2p_nodes
    ) == ("hidden_states", "residual") * 3
    assert all(
        node.attributes["payload_tensor"]["shape"]
        == (expected_tokens, 64)
        for node in p2p_nodes)
    assert all(
        node.attributes["bytes"] == expected_tensor_bytes
        for node in p2p_nodes)
    assert sum(node.attributes["bytes"] for node in p2p_nodes) == (
        6 * expected_tensor_bytes)
    for source_stage in range(3):
        hidden_id = (
            f"pp4:pp-{source_stage}-to-{source_stage + 1}:hidden-states")
        residual_id = (
            f"pp4:pp-{source_stage}-to-{source_stage + 1}:residual")
        source_id = f"pp4:layer-{source_stage}:down"
        destination_id = f"pp4:layer-{source_stage + 1}:input-norm"
        assert graph.predecessors(hidden_id) == (source_id,)
        assert graph.predecessors(residual_id) == (hidden_id,)
        assert graph.predecessors(destination_id) == (residual_id,)

    plan = lower_workload(
        graph, deployment, {"nvidia_a800": hardware})
    communication_calls = tuple(
        call for call in plan.calls
        if call.kernel_spec.kind is KernelKind.COMMUNICATION)
    qk_materialization_calls = tuple(
        call for call in plan.calls
        if (call.kernel_spec.kind is KernelKind.VECTOR
            and call.kernel_spec.variant == "elementwise"
            and call.kernel_spec.attributes["operation"] == "copy")
    )
    attention_output_fill_calls = tuple(
        call for call in plan.calls
        if (call.kernel_spec.kind is KernelKind.VECTOR
            and call.kernel_spec.variant == "elementwise"
            and call.kernel_spec.attributes["operation"] == "fill")
    )
    assert len(qk_materialization_calls) == 8
    assert len(attention_output_fill_calls) == 4
    assert len(plan.calls) == (
        len(graph.nodes) + len(qk_materialization_calls)
        + len(attention_output_fill_calls))
    assert tuple(call.call_id for call in communication_calls) == expected_ids
    predictions = predict_kernel_plan(
        plan, {"nvidia_a800": hardware}, method="midlayer")
    timeline = build_execution_timeline(plan, predictions)
    intervals = {
        interval.call_id: interval for interval in timeline.intervals
    }
    for source_stage in range(3):
        transfer_id = (
            f"pp4:pp-{source_stage}-to-{source_stage + 1}:residual")
        consumer_id = f"pp4:layer-{source_stage + 1}:input-norm"
        assert intervals[consumer_id].start_time_s >= intervals[
            transfer_id].end_time_s

    reference = qwen3_uniform_pp_forward_reference(
        fixture,
        forward_scope=scheduled_request.phase,
        step_id="pp4",
        query_lengths=(scheduled_request.scheduled_tokens,),
        kv_lengths=(
            scheduled_request.computed_tokens
            + scheduled_request.scheduled_tokens,
        ),
        physical_ranks=physical_ranks,
        resource_id="main-pipeline-fabric",
    )
    report = compare_tp2_reference_to_emission(reference, graph, plan)
    source = pinned_qwen3_pp_source_evidence()
    assert source["vllm_version"] == "0.11.0"
    assert source["pipeline_intermediate_tensor_fields"] == (
        "hidden_states", "residual")
    assert reference["high_level_send_tensor_dict_calls_per_step"] == 3
    assert reference["physical_tensor_sends_per_step"] == 6
    assert report["validation_status"] == (
        "validated_uniform_pp_static_source_semantics_projected_latency"
    ), report
    expected_reference_nodes = 57 if scheduled_request.phase == "prefill" else 58
    assert report["expected_node_count"] == expected_reference_nodes
    assert report["actual_node_count"] == expected_reference_nodes
    assert len(report["matched"]) == expected_reference_nodes
    assert report["missing"] == ()
    assert report["extra"] == ()
    assert report["mismatched"] == ()
    assert report["topology_mismatched"] == ()
    assert report["node_coverage"] == 1.0


def test_pp3_lowering_rejects_a_missing_boundary_residual():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    model = load_model_config(
        "tests/serving/fixtures/qwen3_pp3_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(pipeline_parallel_size=3),
                ranks=(0, 1, 2),
                pipeline_p2p=_pp3_pipeline_p2p(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    graph = build_workload(
        ScheduleStep(
            "pp3-missing-residual",
            "main",
            (ScheduledRequest("r", "prefill", 2, 0, 0),),
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen",
            "uniform",
        ),
    )
    residual_id = "pp3-missing-residual:pp-1-to-2:residual"
    hidden_id = "pp3-missing-residual:pp-1-to-2:hidden-states"
    destination_id = "pp3-missing-residual:layer-4:input-norm"
    forged = WorkloadGraph(
        tuple(node for node in graph.nodes if node.node_id != residual_id),
        tuple(
            edge for edge in graph.edges
            if edge.source != residual_id and edge.destination != residual_id
        ) + (WorkloadEdge(hidden_id, destination_id),),
    )

    with pytest.raises(
            ValueError, match="requires hidden_states and residual"):
        lower_workload(
            forged,
            deployment,
            {"nvidia_a800": hardware},
        )


def test_pp3_lowering_rejects_a_nonfinal_source_frontier():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    model = load_model_config(
        "tests/serving/fixtures/qwen3_pp3_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(pipeline_parallel_size=3),
                ranks=(0, 1, 2),
                pipeline_p2p=_pp3_pipeline_p2p(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    graph = build_workload(
        ScheduleStep(
            "pp3-early-frontier",
            "main",
            (ScheduledRequest("r", "prefill", 2, 0, 0),),
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen",
            "uniform",
        ),
    )
    hidden_id = "pp3-early-frontier:pp-0-to-1:hidden-states"
    early_source_id = "pp3-early-frontier:layer-1:o-proj"
    forged = WorkloadGraph(
        graph.nodes,
        tuple(
            edge for edge in graph.edges
            if edge.destination != hidden_id
        ) + (WorkloadEdge(early_source_id, hidden_id, "ready"),),
    )

    with pytest.raises(ValueError, match="final source-stage frontier"):
        lower_workload(
            forged,
            deployment,
            {"nvidia_a800": hardware},
        )


def test_uniform_pp_scope_rejects_tied_embeddings_before_emission():
    model = load_model_config(
        "tests/serving/fixtures/qwen3_pp3_scheduler")
    tied_model = replace(model, tie_word_embeddings=True)
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(pipeline_parallel_size=3),
                ranks=(0, 1, 2),
                pipeline_p2p=_pp3_pipeline_p2p(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )

    with pytest.raises(NotImplementedError, match="uniform TP=1 PP"):
        validate_workload_emission_parallelism(
            deployment,
            tied_model,
            allow_single_pool_tp2=True,
            allow_single_pool_ep2=True,
            allow_single_pool_tp1_pp=True,
            allow_single_pool_tp2_pp2=True,
        )
    with pytest.raises(NotImplementedError, match="untied dense BF16 Qwen3"):
        build_workload(
            ScheduleStep(
                "pp3-tied",
                "main",
                (ScheduledRequest("r", "prefill", 2, 0, 0),),
            ),
            tied_model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.seeded_assumption(
                    "test_seeded_routing_v1", 0),
                "paged_varlen",
                "uniform",
            ),
        )


def test_pp2_lowering_rejects_forged_endpoint_resource_and_payload_bytes():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    model = load_model_config("tests/serving/fixtures/qwen3_pp2_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(pipeline_parallel_size=2),
                ranks=(91, 7),
                pipeline_p2p=_pp2_pipeline_p2p(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    graph = build_workload(
        ScheduleStep(
            "pp2-forge",
            "main",
            (ScheduledRequest("r", "prefill", 2, 0, 0),),
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen", "uniform"),
    )
    target = graph.node("pp2-forge:pp-0-to-1:hidden-states")

    for replacement, message in (
        ({**target.attributes, "participants": (7, 91)},
         "disagrees with deployment"),
        ({**target.attributes, "resource_id": "forged-fabric"},
         "disagrees with deployment"),
        ({**target.attributes, "bytes": 255},
         "bytes disagree with shape"),
        ({
            **target.attributes,
            "payload_tensor": {
                **target.attributes["payload_tensor"],
                "name": "activation",
            },
        }, "name must be hidden_states or residual"),
    ):
        forged_nodes = tuple(
            replace(node, attributes=replacement)
            if node.node_id == target.node_id else node
            for node in graph.nodes
        )
        with pytest.raises(ValueError, match=message):
            lower_workload(
                WorkloadGraph(forged_nodes, graph.edges),
                deployment,
                {"nvidia_a800": hardware},
            )

    forged_rank_nodes = tuple(
        replace(node, rank=7) if node.node_id == target.node_id else node
        for node in graph.nodes
    )
    with pytest.raises(ValueError, match="physical source rank"):
        lower_workload(
            WorkloadGraph(forged_rank_nodes, graph.edges),
            deployment,
            {"nvidia_a800": hardware},
        )


def test_standalone_af_ep2_scope_rejects_unvalidated_ranks_and_model_shape():
    from src.serving.backend.vllm import VLLMBackend

    model = load_model_config(
        "tests/serving/fixtures/qwen3_moe_ep2_scheduler")
    deployment = _standalone_af_ep2_deployment(model)
    assert is_validated_standalone_af_ep2_scope(deployment, model)

    alternate_ranks = _standalone_af_ep2_deployment(
        model,
        attention_ranks=(91, 7),
        ffn_ranks=(18, 4),
    )
    assert not is_validated_standalone_af_ep2_scope(
        alternate_ranks, model)
    assert not is_validated_standalone_af_ep2_scope(
        deployment,
        replace(model, num_hidden_layers=4),
    )

    simulation_config = SimulationConfig(
        model,
        deployment,
        RequestStream((RequestSpec(0.0, "af-ep2", 4, 2, 0),)),
        "analytical_launch",
        "midlayer",
        RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen",
            "uniform",
        ),
    )
    wrong_capacity_backend = VLLMBackend(
        BackendConfig(
            "attention",
            "attention",
            max_batched_tokens=128,
            max_requests=4,
            num_kv_blocks=128,
            kv_block_size_tokens=32,
            kv_bytes_per_token=1,
        scheduler_pipeline_depth=1),
        object(),
    )
    with pytest.raises(
            NotImplementedError, match="Scheduler/KV capacity"):
        MultiSimulation(
            simulation_config,
            {"attention": wrong_capacity_backend},
            {"nvidia_a800": HWSpec(
                "configs/hardware/nvidia_a800.yaml")},
        )

    class NonVLLMBackend:
        config = BackendConfig(
            "attention",
            "attention",
            max_batched_tokens=8,
            max_requests=1,
            num_kv_blocks=8,
            kv_block_size_tokens=16,
            kv_bytes_per_token=512,
        scheduler_pipeline_depth=1)

    with pytest.raises(
            NotImplementedError, match="pinned vLLM Scheduler backend"):
        MultiSimulation(
            simulation_config,
            {"attention": NonVLLMBackend()},
            {"nvidia_a800": HWSpec(
                "configs/hardware/nvidia_a800.yaml")},
        )


def test_standalone_af_tp2_preserves_vllm_tp_compute_and_emits_tensor_pairs():
    source = pinned_qwen3_source_evidence()
    assert source["residual_rmsnorm_boundaries"] is True
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    model = load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler")
    deployment = _standalone_af_tp2_deployment(model)
    validate_workload_emission_parallelism(
        deployment,
        model,
        allow_single_pool_tp2=False,
        allow_single_pool_ep2=False,
        allow_single_pool_tp1_pp=False,
        allow_single_pool_tp2_pp2=False,
    )
    scheduled_request = ScheduledRequest("r", "prefill", 2, 0, 0)
    step = ScheduleStep(
        "af-tp2",
        "attention",
        (scheduled_request,),
        attention_kv_layout="paged_varlen",
    )
    runtime_inputs = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        "paged_varlen",
        "uniform",
    )
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=runtime_inputs,
    )

    colocated = build_deployment_plan(
        DeploymentConfig(
            "colocated",
            (
                WorkerPoolConfig(
                    "attention",
                    "colocated",
                    "nvidia_a800",
                    ParallelConfig(tensor_parallel_size=2),
                    ranks=(91, 7),
                    collectives=_named_tp2_collectives("attention"),
                ),
            ),
        ),
        num_layers=model.num_hidden_layers,
    )
    colocated_graph = build_workload(
        step,
        model,
        colocated,
        launch_policy="analytical_launch",
        runtime_inputs=runtime_inputs,
    )
    communication_kinds = {
        WorkloadKind.COLLECTIVE,
        WorkloadKind.AF_ACTIVATION_TRANSFER,
    }
    af_compute = {
        node.node_id: node for node in graph.nodes
        if node.kind not in communication_kinds
    }
    colocated_compute = {
        node.node_id: node for node in colocated_graph.nodes
        if node.kind not in communication_kinds
    }
    assert set(af_compute) == set(colocated_compute)
    for node_id, node in af_compute.items():
        reference = colocated_compute[node_id]
        assert (
            node.kind,
            dict(node.shape),
            node.dtype,
            node.variant,
            dict(node.attributes),
        ) == (
            reference.kind,
            dict(reference.shape),
            reference.dtype,
            reference.variant,
            dict(reference.attributes),
        )

    af_collectives = {
        node.node_id: node for node in graph.nodes
        if node.kind is WorkloadKind.COLLECTIVE
    }
    colocated_collectives = {
        node.node_id: node for node in colocated_graph.nodes
        if node.kind is WorkloadKind.COLLECTIVE
    }
    assert set(af_collectives) == set(colocated_collectives)
    # A800 prefill suppresses the token-embedding gather and its TP
    # all-reduce through the authorized measured-zero emission observation,
    # in both the AF and the colocated reference graph.
    assert len(af_collectives) == 5
    for node_id, node in af_collectives.items():
        reference = colocated_collectives[node_id]
        assert (
            node.attributes["operation"],
            node.attributes["bytes"],
            node.attributes["bytes_semantics"],
        ) == (
            reference.attributes["operation"],
            reference.attributes["bytes"],
            reference.attributes["bytes_semantics"],
        )
        expected_ranks = deployment.worker_pools[node.pool_name].ranks
        assert tuple(node.attributes["participants"]) == expected_ranks

    transfers = tuple(
        node for node in graph.nodes
        if node.kind is WorkloadKind.AF_ACTIVATION_TRANSFER
    )
    assert len(transfers) == 16
    assert tuple(node.attributes["participants"] for node in transfers) == (
        (91, 18), (91, 18), (7, 4), (7, 4),
        (18, 91), (18, 91), (4, 7), (4, 7),
        (91, 18), (91, 18), (7, 4), (7, 4),
        (18, 91), (18, 91), (4, 7), (4, 7),
    )
    assert tuple(
        node.attributes["payload_tensor"]["name"] for node in transfers
    ) == ("hidden_states", "residual") * 8
    expected_hidden_bytes = 2 * model.hidden_size * model.dtype_bytes
    assert all(
        node.attributes["bytes"] == expected_hidden_bytes
        and node.attributes["payload_tensor"] == {
            "name": node.attributes["payload_tensor"]["name"],
            "shape": (2, model.hidden_size),
            "dtype": "bf16",
            "bytes": expected_hidden_bytes,
        }
        for node in transfers
    )
    for node in transfers:
        predecessor_id, = graph.predecessors(node.node_id)
        predecessor = graph.node(predecessor_id)
        if node.attributes["payload_tensor"]["name"] == "hidden_states":
            assert predecessor.kind is WorkloadKind.COLLECTIVE
            assert predecessor.attributes["operation"] == "all_reduce"
            assert predecessor.attributes["bytes"] == expected_hidden_bytes
        elif ":af-transfer-0:residual:" in node.node_id:
            # A800 prefill suppresses the token-embedding gather (and its
            # TP all-reduce) through the authorized measured-zero emission
            # observation, so the initial residual transfer anchors its
            # ready edge on the attention pool's last in-window producer,
            # the layer-0 o-proj all-reduce.
            assert predecessor.node_id == (
                "af-tp2:layer-0:o-proj:tp-all-reduce")
        else:
            assert predecessor.kind is WorkloadKind.NORM
            assert predecessor.variant == "add_rmsnorm"
            assert predecessor.shape == {
                "rows": 2,
                "hidden": model.hidden_size,
            }
        outgoing_edge, = tuple(
            edge for edge in graph.edges if edge.source == node.node_id)
        consumer = graph.node(outgoing_edge.destination)
        assert consumer.pool_name == node.attributes["destination_pool"]
        assert consumer.rank == node.attributes["destination_rank"]

    plan = lower_workload(
        graph,
        deployment,
        {"nvidia_a800": hardware},
    )
    communication_calls = tuple(
        call for call in plan.calls
        if call.kernel_spec.kind is KernelKind.COMMUNICATION
    )
    # 5 TP collectives + 16 AF transfer calls: the token-embedding
    # all-reduce is gone with the observation-suppressed prefill embedding.
    assert len(communication_calls) == 21
    transfer_calls = tuple(
        call for call in communication_calls
        if call.kernel_spec.transfer.operation == "send_recv"
    )
    assert len(transfer_calls) == 16
    assert all(
        call.kernel_spec.transfer.bytes == expected_hidden_bytes
        and call.kernel_spec.transfer.bytes_semantics == "payload"
        and call.kernel_spec.transfer.backend == "analytical_roofline_v1"
        for call in transfer_calls
    )
    predictions = predict_kernel_plan(
        plan,
        {"nvidia_a800": hardware},
        method="midlayer",
    )
    timeline = build_execution_timeline(plan, predictions)
    transfer_intervals = tuple(
        interval for interval in timeline.intervals
        if interval.call_id.startswith("af-tp2:af-transfer-")
    )
    assert len(transfer_intervals) == 16
    assert all(
        "communication:fabric:attention-ffn-nvlink"
        in interval.reserved_resource_ids
        for interval in transfer_intervals
    )

    forged = transfers[0]
    forged_nodes = tuple(
        replace(
            node,
            attributes={
                **node.attributes,
                "destination_rank": 4,
            },
        )
        if node.node_id == forged.node_id else node
        for node in graph.nodes
    )
    with pytest.raises(ValueError, match="deployment route/lane"):
        lower_workload(
            WorkloadGraph(forged_nodes, graph.edges),
            deployment,
            {"nvidia_a800": hardware},
        )

    missing_lane = transfers[0]
    missing_lane_graph = WorkloadGraph(
        tuple(
            node for node in graph.nodes
            if node.node_id != missing_lane.node_id
        ),
        tuple(
            edge for edge in graph.edges
            if (edge.source != missing_lane.node_id
                and edge.destination != missing_lane.node_id)
        ),
    )
    with pytest.raises(
            ValueError, match="hidden_states/residual activation pair"):
        lower_workload(
            missing_lane_graph,
            deployment,
            {"nvidia_a800": hardware},
        )

    missing_pair_ids = {transfers[0].node_id, transfers[1].node_id}
    missing_pair_graph = WorkloadGraph(
        tuple(
            node for node in graph.nodes
            if node.node_id not in missing_pair_ids
        ),
        tuple(
            edge for edge in graph.edges
            if (edge.source not in missing_pair_ids
                and edge.destination not in missing_pair_ids)
        ),
    )
    with pytest.raises(
            ValueError, match="hidden_states/residual activation pair"):
        lower_workload(
            missing_pair_graph,
            deployment,
            {"nvidia_a800": hardware},
        )

    missing_boundary_ids = {
        node.node_id for node in transfers
        if ":af-transfer-0:" in node.node_id
    }
    bypass_edges = []
    for hidden_transfer in transfers:
        if (hidden_transfer.node_id not in missing_boundary_ids
                or hidden_transfer.attributes["payload_tensor"]["name"]
                != "hidden_states"):
            continue
        source_id, = graph.predecessors(hidden_transfer.node_id)
        destination_edge, = tuple(
            edge for edge in graph.edges
            if edge.source == hidden_transfer.node_id)
        bypass_edges.append(WorkloadEdge(
            source_id, destination_edge.destination, "data"))
    missing_boundary_graph = WorkloadGraph(
        tuple(
            node for node in graph.nodes
            if node.node_id not in missing_boundary_ids
        ),
        tuple(
            edge for edge in graph.edges
            if (edge.source not in missing_boundary_ids
                and edge.destination not in missing_boundary_ids)
        ) + tuple(bypass_edges),
    )
    with pytest.raises(
            ValueError, match="cross-pool dependencies must pass through"):
        lower_workload(
            missing_boundary_graph,
            deployment,
            {"nvidia_a800": hardware},
        )

    hidden_transfer, residual_transfer = transfers[:2]
    swapped_payloads = {
        hidden_transfer.node_id: residual_transfer.attributes["payload_tensor"],
        residual_transfer.node_id: hidden_transfer.attributes["payload_tensor"],
    }
    swapped_nodes = tuple(
        replace(
            node,
            attributes={
                **{
                    name: value
                    for name, value in node.attributes.items()
                    # drop the observation marker with the residual payload
                    # it licenses, so the forged pair differs from the
                    # canonical contract only in payload ordering
                    if name != "residual_anchor"
                },
                "payload_tensor": swapped_payloads[node.node_id],
            },
        )
        if node.node_id in swapped_payloads else node
        for node in graph.nodes
    )
    with pytest.raises(
            ValueError, match="ordered hidden_states/residual payload pair"):
        lower_workload(
            WorkloadGraph(swapped_nodes, graph.edges),
            deployment,
            {"nvidia_a800": hardware},
        )

    bad_bytes_nodes = tuple(
        replace(
            node,
            attributes={
                **node.attributes,
                "payload_tensor": {
                    **node.attributes["payload_tensor"],
                    "bytes": node.attributes["bytes"] + 2,
                },
            },
        )
        if node.node_id == hidden_transfer.node_id else node
        for node in graph.nodes
    )
    with pytest.raises(ValueError, match="bytes disagree with shape"):
        lower_workload(
            WorkloadGraph(bad_bytes_nodes, graph.edges),
            deployment,
            {"nvidia_a800": hardware},
        )

    float_bytes_nodes = tuple(
        replace(
            node,
            attributes={
                **node.attributes,
                "payload_tensor": {
                    **node.attributes["payload_tensor"],
                    "bytes": float(node.attributes["bytes"]),
                },
            },
        )
        if node.node_id == hidden_transfer.node_id else node
        for node in graph.nodes
    )
    with pytest.raises(TypeError, match="bytes must be a positive integer"):
        lower_workload(
            WorkloadGraph(float_bytes_nodes, graph.edges),
            deployment,
            {"nvidia_a800": hardware},
        )

    bad_shape_nodes = tuple(
        replace(
            node,
            attributes={
                **node.attributes,
                "payload_tensor": {
                    **node.attributes["payload_tensor"],
                    "shape": (
                        model.hidden_size,
                        scheduled_request.scheduled_tokens,
                    ),
                },
            },
        )
        if node.node_id == residual_transfer.node_id else node
        for node in graph.nodes
    )
    with pytest.raises(
            ValueError, match="shape disagrees with its add_rmsnorm consumer"):
        lower_workload(
            WorkloadGraph(bad_shape_nodes, graph.edges),
            deployment,
            {"nvidia_a800": hardware},
        )

    hidden_producer, = graph.predecessors(hidden_transfer.node_id)
    # The observation-suppressed prefill embedding makes the hidden-state
    # producer the legitimate anchor for the transfer-0 residual, so a wrong
    # producer here is a future residual norm rather than the hidden
    # producer itself.
    wrong_residual_edges = tuple(
        WorkloadEdge(
            "af-tp2:layer-0:qkv:tp-rank-0"
            if edge.destination == residual_transfer.node_id
            else edge.source,
            edge.destination,
            edge.kind,
        )
        for edge in graph.edges
    )
    with pytest.raises(
            ValueError, match=(
                "lost its input producer without the "
                "observation-suppression anchor")):
        lower_workload(
            WorkloadGraph(graph.nodes, wrong_residual_edges),
            deployment,
            {"nvidia_a800": hardware},
        )

    stale_residual_ids = {
        node.node_id for node in transfers
        if (":af-transfer-3:residual:" in node.node_id)
    }
    stale_residual_edges = tuple(
        WorkloadEdge(
            edge.source.replace(
                "layer-1:post-attn-norm", "layer-0:post-attn-norm"),
            edge.destination,
            edge.kind,
        )
        if edge.destination in stale_residual_ids else edge
        for edge in graph.edges
    )
    with pytest.raises(
            ValueError, match="current boundary residual producer"):
        lower_workload(
            WorkloadGraph(graph.nodes, stale_residual_edges),
            deployment,
            {"nvidia_a800": hardware},
        )

    stale_source_ids = {
        "af-tp2:layer-0:post-attn-norm:tp-rank-0",
        "af-tp2:layer-0:post-attn-norm:tp-rank-1",
    }
    reordered_nodes = tuple(
        node for node in graph.nodes
        if node.node_id not in stale_source_ids
    ) + tuple(
        graph.node(node_id) for node_id in sorted(stale_source_ids)
    )
    with pytest.raises(
            ValueError, match="current boundary residual producer"):
        lower_workload(
            WorkloadGraph(reordered_nodes, stale_residual_edges),
            deployment,
            {"nvidia_a800": hardware},
        )


def test_standalone_af_tp1_initial_residual_anchors_on_pool_producer_when_embedding_suppressed():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    model = load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler")
    deployment = _standalone_af_tp1_deployment(model)
    graph = build_workload(
        ScheduleStep(
            "af-tp1",
            "attention",
            (ScheduledRequest("request", "prefill", 2, 0, 0),),
            attention_kv_layout="paged_varlen",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.not_applicable(),
            "paged_varlen",
            "uniform",
        ),
    )

    # A800 prefill carries the authorized measured-zero token-embedding
    # emission observation, so no in-graph producer owns the initial
    # residual: the first residual transfer still launches and anchors its
    # ready edge on the attention pool's last in-window producer.
    first_residual = graph.node("af-tp1:af-transfer-0:residual")
    assert first_residual.attributes["payload_tensor"]["name"] == "residual"
    assert graph.predecessors(first_residual.node_id) == (
        "af-tp1:layer-0:o-proj",
    )
    carried_residual = graph.node("af-tp1:af-transfer-1:residual")
    carried_source_id, = graph.predecessors(carried_residual.node_id)
    carried_source = graph.node(carried_source_id)
    assert carried_source.node_id == "af-tp1:layer-0:post-attn-norm"
    assert carried_source.variant == "add_rmsnorm"
    lower_workload(graph, deployment, {"nvidia_a800": hardware})


def test_standalone_af_mtp_initial_residual_uses_merge_projection_input():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    model = load_model_config(
        "tests/serving/fixtures/glm4_moe_mtp_scheduler")
    deployment = _standalone_af_tp1_deployment(model)
    step = ScheduleStep(
        "af-mtp",
        "attention",
        (ScheduledRequest(
            "request",
            "decode",
            0,
            3,
            16,
            raw_model_output_tokens=2,
            mtp_proposed_tokens=2,
            mtp_verified_tokens=2,
            mtp_accepted_tokens=1,
            mtp_proposer_tokens=2,
            mtp_next_proposed_tokens=2,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="fixed-1",
        ),),
        attention_kv_layout="paged_varlen",
    )
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen",
            "uniform",
        ),
    )

    mtp_consumer_id = "af-mtp:mtp-pass-0:decoder:post-attn-norm"
    mtp_residuals = tuple(
        node for node in graph.nodes
        if (node.kind is WorkloadKind.AF_ACTIVATION_TRANSFER
            and node.attributes["payload_tensor"]["name"] == "residual"
            and any(
                edge.source == node.node_id
                and edge.destination == mtp_consumer_id
                for edge in graph.edges))
    )
    mtp_residual, = mtp_residuals
    assert graph.predecessors(mtp_residual.node_id) == (
        "af-mtp:mtp-pass-0:merge-proj",
    )
    lower_workload(graph, deployment, {"nvidia_a800": hardware})


def test_tp2_lm_head_emission_uses_nontrivial_padded_vocab_partition():
    model = replace(
        load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler"),
        vocab_size=65,
    )
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(tensor_parallel_size=2),
                ranks=(0, 1),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    graph = build_workload(
        ScheduleStep(
            "padded-vocab",
            "main",
            (ScheduledRequest("r", "prefill", 2, 0),),
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen", "uniform"),
    )

    lm_head_nodes = tuple(
        node for node in graph.nodes
        if node.kind is WorkloadKind.LM_HEAD
    )
    assert len(lm_head_nodes) == 2
    assert all(node.shape["n"] == 64 for node in lm_head_nodes)
    collective = graph.node(
        "padded-vocab:lm-head:tp-all-gather")
    assert collective.attributes["bytes"] == 128
    plan = lower_workload(
        graph,
        deployment,
        {"nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml")},
    )
    call = next(
        call for call in plan.calls
        if call.call_id == collective.node_id
    )
    assert call.kernel_spec.transfer.bytes == 128
    assert call.kernel_spec.transfer.operation == "all_gather"


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("participants", "participants disagree"),
        ("resource_id", "disagrees with deployment config"),
        ("leader", "leader must be logical tensor rank zero"),
        ("bytes", "payload bytes disagree with producer shape and dtype"),
    ),
)
def test_tp2_collective_lowering_rejects_forged_physical_contract(
        mutation, match):
    model = load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(tensor_parallel_size=2),
                ranks=(91, 7),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    graph = build_workload(
        ScheduleStep(
            "forged",
            "main",
            (ScheduledRequest("r", "decode", 0, 1, 2),),
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen", "uniform"),
    )
    collective = next(
        node for node in graph.nodes
        if node.kind is WorkloadKind.COLLECTIVE
    )
    if mutation == "participants":
        attributes = dict(collective.attributes)
        attributes["participants"] = (91, 8)
        forged = replace(collective, attributes=attributes)
    elif mutation == "resource_id":
        attributes = dict(collective.attributes)
        attributes["resource_id"] = "forged-fabric"
        forged = replace(collective, attributes=attributes)
    elif mutation == "leader":
        forged = replace(collective, rank=7)
    elif mutation == "bytes":
        attributes = dict(collective.attributes)
        attributes["bytes"] += 8192
        forged = replace(collective, attributes=attributes)
    else:
        raise AssertionError(f"unknown collective mutation: {mutation!r}")
    forged_graph = WorkloadGraph(
        tuple(
            forged if node.node_id == collective.node_id else node
            for node in graph.nodes
        ),
        graph.edges,
    )

    with pytest.raises(ValueError, match=match):
        lower_workload(
            forged_graph,
            deployment,
            {"nvidia_a800": HWSpec(
                "configs/hardware/nvidia_a800.yaml")},
        )


@pytest.mark.parametrize(
    "mutation,match",
    (
        ("edge_kind", "producer edges must be barriers"),
        ("producer_kind", "producers must have one consistent kind"),
        ("producer_shape", "must have the same shape, dtype, and variant"),
        ("producer_variant", "must have the same shape, dtype, and variant"),
        ("operation", "operation disagrees with producer kind"),
    ),
)
def test_tp2_collective_lowering_rejects_forged_producer_contract(
        mutation, match):
    model = load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(tensor_parallel_size=2),
                ranks=(91, 7),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    graph = build_workload(
        ScheduleStep(
            "forged-producer",
            "main",
            (ScheduledRequest("r", "decode", 0, 1, 2),),
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen", "uniform"),
    )
    if mutation == "producer_variant":
        collective_operation = "all_gather"
        expected_producer_kind = WorkloadKind.LM_HEAD
    elif mutation in (
            "edge_kind", "producer_kind", "producer_shape", "operation"):
        collective_operation = "all_reduce"
        expected_producer_kind = WorkloadKind.GATHER
    else:
        raise AssertionError(f"unknown producer mutation: {mutation!r}")
    collective = next(
        node for node in graph.nodes
        if (node.kind is WorkloadKind.COLLECTIVE
            and node.attributes["operation"] == collective_operation)
    )
    predecessor_ids = graph.predecessors(collective.node_id)
    assert len(predecessor_ids) == 2
    assert all(
        graph.node(node_id).kind is expected_producer_kind
        for node_id in predecessor_ids)

    nodes = graph.nodes
    edges = graph.edges
    if mutation == "edge_kind":
        edges = tuple(
            replace(edge, kind="data")
            if (edge.source == predecessor_ids[0]
                and edge.destination == collective.node_id)
            else edge
            for edge in graph.edges
        )
    elif mutation == "producer_kind":
        nodes = tuple(
            replace(
                node,
                kind=WorkloadKind.LINEAR,
                shape={"b": 1, "m": 1, "n": 256, "k": 256},
                variant="gemm",
                attributes=_test_gemm_attributes("qkv_proj"),
            )
            if node.node_id == predecessor_ids[0]
            else node
            for node in graph.nodes
        )
    elif mutation == "producer_shape":
        nodes = tuple(
            replace(
                node,
                shape={"elements": 256, "rows": 2, "hidden": 128},
            )
            if node.node_id == predecessor_ids[0]
            else node
            for node in graph.nodes
        )
    elif mutation == "producer_variant":
        assert graph.node(predecessor_ids[0]).variant == "gemv"
        nodes = tuple(
            replace(node, variant="gemm")
            if node.node_id == predecessor_ids[0]
            else node
            for node in graph.nodes
        )
    elif mutation == "operation":
        attributes = dict(collective.attributes)
        attributes["operation"] = "all_gather"
        attributes["reduction"] = None
        forged = replace(collective, attributes=attributes)
        nodes = tuple(
            forged if node.node_id == collective.node_id else node
            for node in graph.nodes
        )
    else:
        raise AssertionError(f"unknown producer mutation: {mutation!r}")

    with pytest.raises(ValueError, match=match):
        lower_workload(
            WorkloadGraph(nodes, edges),
            deployment,
            {"nvidia_a800": HWSpec(
                "configs/hardware/nvidia_a800.yaml")},
        )


def test_configured_all_to_all_collective_lowers_without_dense_tp_contract():
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(tensor_parallel_size=3),
                ranks=(0, 1, 2),
                collectives=(CollectiveConfig(
                    "all_to_all",
                    "pairwise",
                    "nvlink",
                    "fully_connected",
                    "main-all-to-all-fabric",
                    "analytical_roofline_v1",
                ),),
            ),
        )),
        num_layers=1,
        num_experts=0,
    )
    node = WorkloadNode(
        "all-to-all",
        WorkloadKind.COLLECTIVE,
        "main",
        {},
        rank=0,
        attributes={
            "algorithm": "pairwise",
            "bytes": 1026,
            "bytes_semantics": "per_rank_input",
            "backend": "analytical_roofline_v1",
            "backend_config_sha256": None,
            "direction": "collective",
            "link": "nvlink",
            "topology": "fully_connected",
            "participants": (0, 1, 2),
            "operation": "all_to_all",
            "source_rank": None,
            "destination_rank": None,
            "ring_order": (),
            "reduction": None,
            "resource_id": "main-all-to-all-fabric",
        },
    )

    plan = lower_workload(
        WorkloadGraph((node,)),
        deployment,
        {"nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml")},
    )

    assert len(plan.calls) == 1
    transfer = plan.calls[0].kernel_spec.transfer
    assert transfer.operation == "all_to_all"
    assert transfer.algorithm == "pairwise"
    assert transfer.ring_order == ()
    assert transfer.bytes == 1026


def test_tp2_workload_rejects_any_enabled_mtp_semantic_state():
    model = load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(tensor_parallel_size=2),
                ranks=(0, 1),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    request = ScheduledRequest(
        "mtp-state",
        "decode",
        0,
        1,
        computed_tokens=2,
        mtp_proposed_tokens=1,
        mtp_proposer_mode="padded",
        mtp_acceptance_source_kind="explicit_assumption",
        mtp_acceptance_source_id="tp2-must-reject",
    )

    with pytest.raises(NotImplementedError, match="does not yet support MTP"):
        build_workload(
            ScheduleStep("mtp-state", "main", (request,)),
            model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.seeded_assumption(
                    "test_seeded_routing_v1", 0),
                "paged_varlen", "uniform"),
        )


def test_qkv_shape_rejects_invalid_tensor_parallel_size():
    model = load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler")

    with pytest.raises(TypeError, match="tp_size must be an integer"):
        infer_gemm_dimensions(model, "qkv_proj", 1, tp_size=True)
    with pytest.raises(ValueError, match="tp_size must be positive"):
        infer_gemm_dimensions(model, "qkv_proj", 1, tp_size=0)
    with pytest.raises(ValueError, match="query heads must be divisible"):
        infer_gemm_dimensions(model, "qkv_proj", 1, tp_size=3)

    incompatible_kv_model = replace(
        model,
        hidden_size=384,
        num_attention_heads=6,
        num_key_value_heads=3,
    )
    with pytest.raises(
            ValueError, match="must divide one another exactly"):
        infer_gemm_dimensions(
            incompatible_kv_model, "qkv_proj", 1, tp_size=2)


def test_schedule_step_rejects_duplicate_request_ids():
    duplicate = ScheduledRequest("request", "decode", 0, 1, 4)

    with pytest.raises(ValueError, match="request IDs must be unique"):
        ScheduleStep("duplicate", "main", (duplicate, duplicate))


@pytest.mark.parametrize(
    "rank,error",
    [
        (None, "requires an explicit rank"),
        (7, "rank 7 is not assigned to pool"),
    ],
)
def test_lowering_rejects_invalid_rank_before_hardware_lookup(rank, error):
    deployment = _deployment_for("nvidia_a800")
    node = WorkloadNode(
        "invalid-rank",
        WorkloadKind.LINEAR,
        "main",
        {"b": 1, "m": 1, "n": 64, "k": 64},
        dtype="bf16",
        variant="gemm",
        rank=rank,
        attributes=_test_gemm_attributes("qkv_proj"),
    )

    with pytest.raises(ValueError, match=error):
        lower_workload(
            WorkloadGraph((node,)),
            deployment,
            {},
        )


def test_preemption_contract_rejects_inconsistent_state():
    with pytest.raises(ValueError, match="cannot exceed scheduled"):
        ScheduledRequest(
            "request", "prefill", 4, 0, 0,
            recomputed_tokens=5,
            resumed_from_preemption=True,
        )
    with pytest.raises(TypeError, match="must be boolean"):
        ScheduledRequest(
            "request", "prefill", 4, 0, 0,
            recomputed_tokens=4,
            resumed_from_preemption=1,
        )
    with pytest.raises(ValueError, match="cannot exceed the trace"):
        ScheduledRequest(
            "request", "prefill", 4, 0, 2,
            trace_prefix_hit_tokens=1,
            scheduler_prefix_hit_tokens=2,
        )
    with pytest.raises(ValueError, match="cannot exceed computed"):
        ScheduledRequest(
            "request", "prefill", 4, 0, 1,
            trace_prefix_hit_tokens=2,
            scheduler_prefix_hit_tokens=2,
        )
    with pytest.raises(ValueError, match="cannot overlap"):
        ScheduledRequest(
            "request", "prefill", 4, 0, 0,
            recomputed_tokens=3,
            prefix_replay_tokens=2,
        )

    request = ScheduledRequest("request", "decode", 0, 1, 4)
    with pytest.raises(ValueError, match="cannot execute"):
        ScheduleStep(
            "preempted-and-scheduled",
            "main",
            (request,),
            preempted_request_ids=("request",),
        )
    with pytest.raises(ValueError, match="must be disjoint"):
        ScheduleStep(
            "preempted-and-finished",
            "main",
            (request,),
            finished_request_ids=("request",),
            preempted_request_ids=("request",),
        )


def test_mtp_schedule_contract_rejects_inconsistent_rounds():
    with pytest.raises(ValueError, match="verified tokens cannot exceed"):
        ScheduledRequest(
            "request", "decode", 0, 3, 16,
            mtp_proposed_tokens=1,
            mtp_verified_tokens=2,
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="scenario",
        )
    with pytest.raises(ValueError, match="accepted tokens cannot exceed"):
        ScheduledRequest(
            "request", "decode", 0, 3, 16,
            mtp_proposed_tokens=2,
            mtp_verified_tokens=2,
            mtp_accepted_tokens=3,
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="scenario",
        )
    with pytest.raises(ValueError, match="disabled MTP source"):
        ScheduledRequest(
            "request", "decode", 0, 3, 16,
            mtp_proposer_tokens=2,
        )
    with pytest.raises(ValueError, match="must produce a raw model output"):
        ScheduledRequest(
            "request", "decode", 0, 3, 16,
            mtp_proposed_tokens=2,
            mtp_verified_tokens=2,
            mtp_accepted_tokens=1,
            mtp_proposer_tokens=2,
            mtp_next_proposed_tokens=2,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="scenario",
        )


@pytest.mark.parametrize("accepted", [0, 1, 2])
def test_glm_mtp_emits_validated_proposer_shapes_and_predictor_chain(accepted):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    model = load_model_config(
        "tests/serving/fixtures/glm4_moe_mtp_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    step = ScheduleStep(
        f"mtp-{accepted}",
        "main",
        (ScheduledRequest(
            "request",
            "decode",
            0,
            3,
            16,
            raw_model_output_tokens=accepted + 1,
            mtp_proposed_tokens=2,
            mtp_verified_tokens=2,
            mtp_accepted_tokens=accepted,
            mtp_proposer_tokens=2,
            mtp_next_proposed_tokens=2,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id=f"fixed-{accepted}",
        ),),
        attention_kv_layout="paged_varlen",
    )
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen", "uniform"),
    )

    assert model.model_type == "glm4_moe"
    assert model.num_hidden_layers == 2
    assert model.num_dense_layers == 1
    assert model.num_experts == 2
    assert model.num_mtp_layers == 1
    assert model.mtp_architecture == "glm4_moe_mtp"
    assert model.num_shared_experts == 1
    assert model.qk_norm is False
    assert model.routed_scaling_factor == 1.0
    assert model.layer_uses_moe(0) is False
    assert model.layer_uses_moe(1) is True
    assert model.mtp_layer_uses_moe(0) is False
    assert any(node.node_id.endswith("layer-0:gate-up")
               for node in graph.nodes)
    assert any(node.node_id.endswith("layer-1:moe")
               for node in graph.nodes)
    routed_moe = graph.node(f"mtp-{accepted}:layer-1:moe")
    shared_gate_up = graph.node(
        f"mtp-{accepted}:layer-1:shared-experts:gate-up")
    assert sum(routed_moe.attributes["tokens_per_expert"]) == (
        routed_moe.shape["tokens"] * routed_moe.shape["topk"])
    assert shared_gate_up.shape["m"] == routed_moe.shape["tokens"]
    assert shared_gate_up.shape["n"] == (
        2 * model.moe_intermediate_size * model.num_shared_experts)
    assert graph.node(f"mtp-{accepted}:token-embedding").shape["rows"] == 3
    assert graph.node(f"mtp-{accepted}:lm-head").shape["m"] == 3
    assert not any("mtp-pass-0:decoder:moe" in node.node_id
                   for node in graph.nodes)
    assert graph.node(
        f"mtp-{accepted}:layer-0:input-norm").variant == "rmsnorm"
    assert graph.node(
        f"mtp-{accepted}:layer-0:post-attn-norm").variant == "add_rmsnorm"
    assert graph.node(
        f"mtp-{accepted}:layer-1:input-norm").variant == "add_rmsnorm"
    assert graph.node(
        f"mtp-{accepted}:layer-1:post-attn-norm").variant == "add_rmsnorm"
    assert graph.node(f"mtp-{accepted}:final-norm").variant == "add_rmsnorm"

    initial_rows = 3
    first_merge = graph.node(f"mtp-{accepted}:mtp-pass-0:merge-proj")
    assert first_merge.shape == {
        "b": 1, "m": initial_rows, "n": 64, "k": 128,
    }
    first_attention = graph.node(
        f"mtp-{accepted}:mtp-pass-0:decoder:attention")
    assert first_attention.attributes["q_lens"] == (initial_rows,)
    assert first_attention.attributes["kv_lens"] == (19,)
    assert first_attention.shape["q_len"] == initial_rows
    assert first_attention.shape["kv_len"] == 19
    first_head = graph.node(f"mtp-{accepted}:mtp-pass-0:head")
    assert first_head.shape == {"b": 1, "m": 1, "n": 64, "k": 64}
    first_mask = graph.node(
        f"mtp-{accepted}:mtp-pass-0:position-zero-mask")
    assert first_mask.shape == {"elements": initial_rows * 64}
    assert first_mask.attributes == {
        "reads": 1,
        "writes": 1,
        "operation": "copy",
        "launch_policy": "analytical_launch",
    }
    assert graph.node(
        f"mtp-{accepted}:mtp-pass-0:embedding-norm").variant == "rmsnorm"
    assert graph.node(
        f"mtp-{accepted}:mtp-pass-0:hidden-norm").variant == "rmsnorm"
    assert graph.node(
        f"mtp-{accepted}:mtp-pass-0:decoder:input-norm").variant == "rmsnorm"
    assert graph.node(
        f"mtp-{accepted}:mtp-pass-0:decoder:post-attn-norm"
    ).variant == "add_rmsnorm"
    assert graph.node(
        f"mtp-{accepted}:mtp-pass-0:head-norm").variant == "rmsnorm"

    second_merge = graph.node(f"mtp-{accepted}:mtp-pass-1:merge-proj")
    assert second_merge.shape == {"b": 1, "m": 1, "n": 64, "k": 128}
    second_attention = graph.node(
        f"mtp-{accepted}:mtp-pass-1:decoder:attention")
    assert second_attention.variant == "decode"
    assert second_attention.attributes["q_lens"] == (1,)
    assert second_attention.attributes["kv_lens"] == (20,)
    assert graph.node(
        f"mtp-{accepted}:mtp-pass-1:position-zero-mask"
    ).shape == {"elements": 64}

    plan = lower_workload(
        graph,
        deployment,
        {"nvidia_a800": hardware},
    )
    mtp_calls = tuple(
        call for call in plan.calls if ":mtp-pass-" in call.call_id)
    assert {call.kernel_spec.kind for call in mtp_calls} >= {
        KernelKind.GEMM,
        KernelKind.FLASH_ATTN,
        KernelKind.VECTOR,
    }
    assert any(
        call.kernel_spec.kind is KernelKind.VECTOR
        and call.kernel_spec.variant == "add_rmsnorm"
        for call in mtp_calls
    )
    mask_calls = tuple(
        call for call in mtp_calls
        if ":position-zero-mask" in call.call_id)
    assert len(mask_calls) == 2
    assert all(
        call.kernel_spec.kind is KernelKind.VECTOR
        and call.kernel_spec.variant == "elementwise"
        and call.kernel_spec.attributes == {
            "reads": 1, "writes": 1, "operation": "copy"}
        for call in mask_calls
    )
    predictions = predict_kernel_plan(
        plan,
        {"nvidia_a800": hardware},
        method="midlayer",
    )
    assert set(predictions) == {call.call_id for call in plan.calls}
    assert all(prediction.t_total_ns > 0
               for prediction in predictions.values())


def test_mtp_padded_workload_preserves_partial_prefill_rows_and_logit_selection():
    model = load_model_config(
        "tests/serving/fixtures/glm4_moe_mtp_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    step = ScheduleStep(
        "mtp-mixed",
        "main",
        (
            ScheduledRequest(
                "decode", "decode", 0, 3, 16,
                raw_model_output_tokens=2,
                mtp_proposed_tokens=2,
                mtp_verified_tokens=2,
                mtp_accepted_tokens=1,
                mtp_proposer_tokens=2,
                mtp_next_proposed_tokens=2,
                mtp_proposer_mode="padded",
                mtp_acceptance_source_kind="explicit_assumption",
                mtp_acceptance_source_id="mixed-v1",
            ),
            ScheduledRequest(
                "prefill", "prefill", 5, 0, 0,
                mtp_proposer_tokens=2,
                mtp_next_proposed_tokens=2,
                mtp_proposer_mode="padded",
                mtp_acceptance_source_kind="explicit_assumption",
                mtp_acceptance_source_id="mixed-v1",
            ),
        ),
        attention_kv_layout="paged_varlen",
    )

    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen", "uniform"),
    )

    assert graph.node("mtp-mixed:token-embedding").shape["rows"] == 8
    assert graph.node("mtp-mixed:lm-head").shape["m"] == 4
    first_attention = graph.node("mtp-mixed:mtp-pass-0:decoder:attention")
    assert first_attention.attributes["q_lens"] == (3, 5)
    assert first_attention.attributes["kv_lens"] == (19, 5)
    assert graph.node("mtp-mixed:mtp-pass-0:merge-proj").shape["m"] == 8
    assert graph.node("mtp-mixed:mtp-pass-0:head").shape["m"] == 2
    assert graph.node(
        "mtp-mixed:mtp-pass-0:position-zero-mask").shape == {
            "elements": 8 * model.hidden_size,
        }

    BackendStep(step, generated_tokens={"decode": 2, "prefill": 0})
    with pytest.raises(ValueError, match="cannot exceed raw"):
        BackendStep(step, generated_tokens={"decode": 3, "prefill": 0})
    with pytest.raises(ValueError, match="complete raw model output"):
        BackendStep(step, generated_tokens={"decode": 1, "prefill": 0})


def test_recompute_emits_compute_without_turning_capacity_into_offload():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    model = load_model_config("tests/serving/fixtures/qwen3_scheduler")
    deployment = _deployment_for("nvidia_a800")
    step = ScheduleStep(
        "recompute-step",
        "main",
        (ScheduledRequest(
            "request",
            "prefill",
            16,
            1,
            0,
            recomputed_tokens=16,
            resumed_from_preemption=True,
        ),),
        finished_request_ids=("request",),
        attention_kv_layout="paged_varlen",
    )
    cache_events = (
        KVCacheEvent(
            "request", KVCacheAction.ALLOCATE, "hbm", 0, 32, 1024),
        KVCacheEvent(
            "request", KVCacheAction.MISS, "hbm", 0, 16, 512),
        KVCacheEvent(
            "request", KVCacheAction.RELEASE, "hbm", 0, 32, 1024),
    )
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        kv_cache_events=cache_events,
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen", "uniform"),
    )

    assert all(node.kind is not WorkloadKind.KV_OFFLOAD for node in graph.nodes)
    qkv = next(
        node for node in graph.nodes
        if node.kind is WorkloadKind.LINEAR and node.node_id.endswith(":qkv"))
    assert qkv.shape["m"] == 17
    attention = next(
        node for node in graph.nodes
        if node.kind is WorkloadKind.FLASH_ATTN)
    assert attention.attributes["q_lens"] == (17,)
    assert attention.attributes["kv_lens"] == (17,)

    plan = lower_workload(
        graph, deployment, {"nvidia_a800": hardware})
    attention_spec = next(
        call.kernel_spec for call in plan.calls
        if call.kernel_spec.kind is KernelKind.FLASH_ATTN)
    assert attention_spec.attributes["q_lens"] == (17,)
    assert attention_spec.attributes["kv_lens"] == (17,)
    predictions = predict_kernel_plan(
        plan, {"nvidia_a800": hardware}, method="midlayer")
    assert set(predictions) == {call.call_id for call in plan.calls}

    catchup_step = ScheduleStep(
        "decode-catchup-step",
        "main",
        (ScheduledRequest(
            "request",
            "decode",
            0,
            2,
            16,
            recomputed_tokens=1,
        ),),
        attention_kv_layout="paged_varlen",
    )
    catchup_graph = build_workload(
        catchup_step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "paged_varlen", "uniform"),
    )
    catchup_attention = next(
        node for node in catchup_graph.nodes
        if node.kind is WorkloadKind.FLASH_ATTN)
    assert catchup_attention.variant == "prefill"
    assert catchup_attention.attributes["q_lens"] == (2,)
    assert catchup_attention.attributes["kv_lens"] == (18,)
    catchup_plan = lower_workload(
        catchup_graph, deployment, {"nvidia_a800": hardware})
    catchup_attention_spec = next(
        call.kernel_spec for call in catchup_plan.calls
        if call.kernel_spec.kind is KernelKind.FLASH_ATTN)
    assert catchup_attention_spec.attributes["q_lens"] == (2,)
    assert catchup_attention_spec.attributes["kv_lens"] == (18,)
    catchup_predictions = predict_kernel_plan(
        catchup_plan, {"nvidia_a800": hardware}, method="midlayer")
    assert set(catchup_predictions) == {
        call.call_id for call in catchup_plan.calls}


def test_mtp_step_tolerates_decode_replay_rows_from_preemption():
    # A preempted-then-resumed request replays already-generated tokens as
    # decode-phase rows with no verified drafts (the waiting-queue path never
    # schedules spec tokens).  Such replay rows are not verification rows and
    # must not trip the MTP terminal-token invariant; genuine verify rows in
    # the same step are still checked.
    model = load_model_config(
        "tests/serving/fixtures/glm_5_2_official")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    step = ScheduleStep(
        "mtp-with-replay",
        "main",
        (
            ScheduledRequest(
                "verify-request",
                "decode",
                0,
                3,
                16,
                raw_model_output_tokens=3,
                mtp_proposed_tokens=2,
                mtp_verified_tokens=2,
                mtp_accepted_tokens=2,
                mtp_proposer_tokens=2,
                mtp_next_proposed_tokens=2,
                mtp_proposer_mode="padded",
                mtp_acceptance_source_kind="explicit_assumption",
                mtp_acceptance_source_id="fixed-2",
            ),
            ScheduledRequest(
                "replay-request",
                "decode",
                0,
                4,
                20,
                recomputed_tokens=4,
                resumed_from_preemption=True,
            ),
        ),
        attention_kv_layout="mla",
    )
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )
    assert len(graph.nodes) > 0
    assert any("attention" in node.node_id or "attn" in node.node_id
               for node in graph.nodes)

    bad_step = ScheduleStep(
        "mtp-bad-verify",
        "main",
        (ScheduledRequest(
            "verify-request",
            "decode",
            0,
            4,
            16,
            raw_model_output_tokens=3,
            mtp_proposed_tokens=2,
            mtp_verified_tokens=2,
            mtp_accepted_tokens=2,
            mtp_proposer_tokens=2,
            mtp_next_proposed_tokens=2,
            mtp_proposer_mode="padded",
            mtp_acceptance_source_kind="explicit_assumption",
            mtp_acceptance_source_id="fixed-2",
        ),),
        attention_kv_layout="mla",
    )
    with pytest.raises(ValueError, match="terminal token"):
        build_workload(
            bad_step,
            model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.seeded_assumption(
                    "test_seeded_routing_v1", 0),
                "mla", "uniform"),
        )


def test_workload_rejects_scheduler_attention_layout_mismatch():
    model = load_model_config("tests/serving/fixtures/qwen3_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
    )
    step = ScheduleStep(
        "layout-step",
        "main",
        (ScheduledRequest("request", "prefill", 4, 0, 0),),
        attention_kv_layout="paged_varlen",
    )

    with pytest.raises(ValueError, match="layout disagrees"):
        build_workload(
            step,
            model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.seeded_assumption(
                    "test_seeded_routing_v1", 0),
                "contiguous", "uniform"),
        )


def test_every_workload_kind_has_explicit_lowering():
    hw = HWSpec("configs/hardware/nvidia_a800.yaml")
    # This is an exhaustive lowering test.  Communication success uses a
    # declared synthetic setup latency; production A800 remains fail-loud
    # because its independent NCCL setup measurement is absent.
    hw.interconnect["nccl_alpha_ns"] = 100.0
    hw.interconnect["pcie_latency_ns"] = 100.0
    compute = {
        WorkloadKind.LINEAR: ({"b": 1, "m": 4, "n": 4096, "k": 4096}, "gemm"),
        WorkloadKind.LM_HEAD: ({"b": 1, "m": 1, "n": 4096, "k": 4096}, "gemv"),
        WorkloadKind.ROUTER: ({"b": 1, "m": 4, "n": 4096, "k": 4096}, "gemm"),
        WorkloadKind.FLASH_ATTN: ({"batch": 1, "q_len": 1, "kv_len": 64,
            "head_dim": 64, "num_query_heads": 8, "num_kv_heads": 8}, "decode"),
        WorkloadKind.EAGER_ATTN: ({"batch": 1, "q_len": 1, "kv_len": 64,
            "head_dim": 64, "num_query_heads": 8, "num_kv_heads": 8}, "decode"),
        WorkloadKind.FUSED_MOE: ({"tokens": 4, "experts": 8, "topk": 2,
            "hidden": 64, "intermediate": 128}, "full"),
        WorkloadKind.NORM: ({"rows": 4, "hidden": 64}, "rmsnorm"),
    }
    vector_kinds = (
        WorkloadKind.ACTIVATION, WorkloadKind.ROTARY, WorkloadKind.ELEMENTWISE,
        WorkloadKind.REDUCE, WorkloadKind.GATHER, WorkloadKind.VECTOR,
        WorkloadKind.KV_UPDATE,
    )

    nodes = []
    for kind, (shape, variant) in compute.items():
        if kind in (WorkloadKind.LINEAR, WorkloadKind.ROUTER):
            attributes = _test_gemm_attributes("qkv_proj")
        elif kind is WorkloadKind.LM_HEAD:
            attributes = _test_gemm_attributes("lm_head")
        elif kind is WorkloadKind.EAGER_ATTN:
            attributes = {
                "launch_policy": "analytical_launch",
                "tensor_parallel_size": 1,
                "expert_parallel_size": 1,
            }
        elif kind is WorkloadKind.FLASH_ATTN:
                attributes = {
                    "causal": False,
                    "kv_layout": "contiguous",
                    "kv_dist": "uniform",
                "q_lens": (1,),
                "kv_lens": (64,),
                    "fa_ksteps": None,
                    "fa_eff_mfrac": 1.0,
                    "split_policy": "serving",
                }
        elif kind is WorkloadKind.FUSED_MOE:
            attributes = _test_moe_attributes(
                shape["tokens"], shape["experts"], shape["topk"])
        elif kind is WorkloadKind.NORM:
            attributes = {}
        else:
            raise AssertionError(f"unhandled compute workload kind: {kind!r}")
        nodes.append(WorkloadNode(
            kind.value, kind, "main", shape, "bf16", variant,
            rank=0, attributes=attributes))
    for kind in vector_kinds:
        if kind in (
                WorkloadKind.ELEMENTWISE, WorkloadKind.VECTOR):
            attributes = {
                "reads": 1, "writes": 1, "operation": "copy"}
        elif kind is WorkloadKind.REDUCE:
            attributes = {"reduction_factor": 1}
        elif kind is WorkloadKind.GATHER:
            attributes = {
                "gather_factor": 1,
                "index_count": 64,
                "index_bytes": 8,
                "scattered": True,
            }
        elif kind is WorkloadKind.KV_UPDATE:
            attributes = PAGED_KV_CACHE_ATTRIBUTES
        elif kind in (
                WorkloadKind.ACTIVATION,
                WorkloadKind.ROTARY,
        ):
            attributes = {}
        else:
            raise AssertionError(f"unhandled vector workload kind: {kind!r}")
        if kind is WorkloadKind.ACTIVATION:
            shape = {"elements": 64, "tok": 1, "intermediate": 64}
        elif kind is WorkloadKind.ROTARY:
            shape = {
                "elements": 64, "seq_len": 1, "num_query_heads": 1,
                "num_kv_heads": 1, "head_dim": 32,
            }
        elif kind is WorkloadKind.KV_UPDATE:
            shape = {
                "elements": 64, "seq_len": 1,
                "num_kv_heads": 1, "head_dim": 64,
            }
        elif kind in (
                WorkloadKind.ELEMENTWISE, WorkloadKind.REDUCE,
                WorkloadKind.GATHER, WorkloadKind.VECTOR):
            shape = {"elements": 64}
        else:
            raise AssertionError(f"unhandled vector workload kind: {kind!r}")
        if kind is WorkloadKind.ACTIVATION:
            variant = "silu"
        elif kind is WorkloadKind.KV_UPDATE:
            variant = "paged_kv_cache"
        else:
            variant = kind.value
        nodes.append(WorkloadNode(
            kind.value, kind, "main", shape, "bf16", variant,
            rank=0, attributes=attributes))
    collective_producer_shape = {"b": 1, "m": 4, "n": 128, "k": 128}
    collective_producer_attributes = _test_gemm_attributes(
        "qkv_proj", tensor_parallel_size=2)
    nodes.extend(
        WorkloadNode(
            f"collective-producer-{rank}",
            WorkloadKind.LINEAR,
            "main",
            collective_producer_shape,
            "bf16",
            "gemm",
            rank=rank,
            attributes=collective_producer_attributes,
        )
        for rank in (0, 1)
    )
    for kind in (
            WorkloadKind.COLLECTIVE,
            WorkloadKind.AF_ACTIVATION_TRANSFER,
    ):
        if kind is WorkloadKind.COLLECTIVE:
            direction = "collective"
            algorithm = "ring"
            topology = "ring"
            operation = "all_reduce"
            bytes_semantics = "per_rank_input"
            source_rank = None
            destination_rank = None
            ring_order = (0, 1)
            reduction = "sum"
        elif kind is WorkloadKind.AF_ACTIVATION_TRANSFER:
            direction = "point_to_point"
            algorithm = "send_recv"
            topology = "point_to_point"
            operation = "send_recv"
            bytes_semantics = "payload"
            source_rank = 0
            destination_rank = 1
            ring_order = ()
            reduction = None
        else:
            raise AssertionError(f"unhandled transfer workload kind: {kind!r}")
        nodes.append(WorkloadNode(kind.value, kind, "main", {}, rank=0, attributes={
                "algorithm": algorithm,
                "bytes": 1024,
                "bytes_semantics": bytes_semantics,
                "backend": "analytical_roofline_v1",
                "backend_config_sha256": None,
                "link": "nvlink",
                "direction": direction,
                "topology": topology,
                "participants": (0, 1),
                "operation": operation,
                "source_rank": source_rank,
                "destination_rank": destination_rank,
                "ring_order": ring_order,
                "reduction": reduction,
                "resource_id": "main-nvlink-fabric",
            }))
    nodes.append(WorkloadNode(
        "memcpy", WorkloadKind.MEMCPY, "main", {}, rank=0, attributes={
        "algorithm": "dma", "bytes": 1024, "direction": "d2d", "link": "dram"}))
    nodes.append(WorkloadNode(
        "kv-offload", WorkloadKind.KV_OFFLOAD, "main", {}, rank=0, attributes={
        "algorithm": "dma", "bytes": 1024, "direction": "load", "link": "pcie",
        "source_tier": "host", "destination_tier": "hbm"}))
    communication_deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(tensor_parallel_size=2),
                ranks=(0, 1),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=1,
    )
    plan = lower_workload(
        WorkloadGraph(nodes, (
            WorkloadEdge("collective-producer-0", "collective", "barrier"),
            WorkloadEdge("collective-producer-1", "collective", "barrier"),
        )), communication_deployment,
        {"nvidia_a800": hw})
    p2p_deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(pipeline_parallel_size=2),
                ranks=(0, 1),
                pipeline_p2p=_pp2_pipeline_p2p(),
            ),
        )),
        num_layers=2,
    )
    p2p_hidden = WorkloadNode(
        "p2p-hidden",
        WorkloadKind.P2P,
        "main",
        {},
        rank=0,
        attributes={
            "algorithm": "send_recv",
            "bytes": 1024,
            "bytes_semantics": "payload",
            "backend": "analytical_roofline_v1",
            "backend_config_sha256": None,
            "link": "nvlink",
            "direction": "point_to_point",
            "topology": "point_to_point",
            "participants": (0, 1),
            "operation": "send_recv",
            "source_rank": 0,
            "destination_rank": 1,
            "ring_order": (),
            "reduction": None,
            "resource_id": "main-pipeline-fabric",
            "source_stage": 0,
            "destination_stage": 1,
            "payload_tensor": {
                "name": "hidden_states",
                "shape": (1, 512),
                "dtype": "bf16",
                "bytes": 1024,
            },
        },
    )
    p2p_residual = replace(
        p2p_hidden,
        node_id="p2p-residual",
        attributes={
            **p2p_hidden.attributes,
            "payload_tensor": {
                **p2p_hidden.attributes["payload_tensor"],
                "name": "residual",
            },
        },
    )
    p2p_source = WorkloadNode(
        "p2p-source",
        WorkloadKind.LINEAR,
        "main",
        {"b": 1, "m": 1, "n": 512, "k": 512},
        "bf16",
        "gemm",
        rank=0,
        attributes=_test_gemm_attributes("qkv_proj"),
    )
    p2p_destination = WorkloadNode(
        "p2p-destination",
        WorkloadKind.NORM,
        "main",
        {"rows": 1, "hidden": 512},
        "bf16",
        "rmsnorm",
        rank=1,
        attributes={},
    )
    p2p_plan = lower_workload(
        WorkloadGraph(
            (p2p_source, p2p_hidden, p2p_residual, p2p_destination),
            (
                WorkloadEdge("p2p-source", "p2p-hidden", "ready"),
                WorkloadEdge("p2p-hidden", "p2p-residual", "ready"),
                WorkloadEdge("p2p-residual", "p2p-destination"),
            ),
        ),
        p2p_deployment,
        {"nvidia_a800": hw},
    )
    assert tuple(
        call.kernel_spec.kind for call in p2p_plan.calls
        if call.kernel_spec.kind is KernelKind.COMMUNICATION
    ) == (KernelKind.COMMUNICATION, KernelKind.COMMUNICATION)
    pd_deployment = build_deployment_plan(
        DeploymentConfig(
            "pd_disagg",
            (
                WorkerPoolConfig(
                    "prefill", "prefill", "nvidia_a800", ranks=(0,)),
                WorkerPoolConfig(
                    "decode", "decode", "nvidia_a800", ranks=(1,)),
            ),
            (RouteConfig(
                "prefill",
                "decode",
                "kv",
                "send_recv",
                "nvlink",
                resource_id="pd-kv-link",
                backend="analytical_roofline_v1",
            ),),
        ),
        num_layers=1,
    )
    pd_node = WorkloadNode(
        "pd-kv-transfer",
        WorkloadKind.PD_KV_TRANSFER,
        "prefill",
        {},
        rank=0,
        attributes={
            "algorithm": "send_recv",
            "bytes": 1024,
            "bytes_semantics": "payload",
            "backend": "analytical_roofline_v1",
            "backend_config_sha256": None,
            "link": "nvlink",
            "direction": "point_to_point",
            "topology": "point_to_point",
            "participants": (0, 1),
            "operation": "send_recv",
            "source_rank": 0,
            "destination_rank": 1,
            "ring_order": (),
            "reduction": None,
            "resource_id": "pd-kv-link",
            "destination_pool": "decode",
            "rank_mapping": "kv_head_overlap",
            "source_logical_rank": 0,
            "destination_logical_rank": 0,
            "source_tensor_parallel_size": 1,
            "destination_tensor_parallel_size": 1,
            "kv_head_start": 0,
            "kv_head_end": 1,
            "num_key_value_heads": 1,
        },
    )
    pd_plan = lower_workload(
        WorkloadGraph((pd_node,)),
        pd_deployment,
        {"nvidia_a800": hw},
    )
    assert pd_plan.calls[0].kernel_spec.kind is KernelKind.COMMUNICATION
    # EAGER_ATTN is the aggregate observation kind used by decomposed accelerator
    # datasets. Serving lowers the logical eager operator into its three real
    # kernel calls instead of routing it through either aggregate EAGER_ATTN or
    # fused FLASH_ATTN prediction.
    assert {call.kernel_spec.kind for call in plan.calls} == (
        set(KernelKind) - {
            KernelKind.EAGER_ATTN,
            KernelKind.SPARSE_ATTN_INDEXER,
            KernelKind.SPARSE_ATTN,
            KernelKind.UNFUSED_MOE,
        })
    norm_call = next(call for call in plan.calls if call.call_id == "norm")
    assert norm_call.kernel_spec.kind is KernelKind.VECTOR
    assert norm_call.kernel_spec.variant == "rmsnorm"
    activation_call = next(
        call for call in plan.calls if call.call_id == "activation")
    assert activation_call.kernel_spec.kind is KernelKind.VECTOR
    assert activation_call.kernel_spec.variant == "silu"
    eager_calls = [
        call for call in plan.calls
        if call.call_id.startswith("eager_attn:")]
    assert [call.kernel_spec.kind for call in eager_calls] == [
        KernelKind.GEMM, KernelKind.VECTOR, KernelKind.GEMM]
    assert [call.kernel_spec.variant for call in eager_calls] == [
        "gemm", "softmax", "gemm"]
    supported_calls = tuple(
        call for call in plan.calls
        if call.kernel_spec.kind is not KernelKind.MEMCPY)
    supported_plan = KernelPlan(supported_calls)
    assert len(predict_kernel_plan(
        supported_plan, {"nvidia_a800": hw})) == len(supported_calls)
    with pytest.raises(
            KeyError,
            match="MEMCPY requires memory_subsystem.memcpy_dtod_floor_ns"):
        predict_kernel_plan(plan, {"nvidia_a800": hw})


def test_elementwise_lowering_rejects_unknown_semantics():
    node = WorkloadNode(
        "typo",
        WorkloadKind.ELEMENTWISE,
        "main",
        {"elements": 64},
        "bf16",
        "masked-fill-zreo",
        rank=0,
        attributes={"reads": 1, "writes": 1},
    )

    with pytest.raises(ValueError, match="unsupported elementwise"):
        lower_workload(
            WorkloadGraph((node,)),
            _deployment_for("nvidia_a800"),
            {"nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml")},
        )


def test_kernel_lowering_memo_reuses_exact_kernel_spec_source_contracts(
        monkeypatch):
    lowering_module = importlib.import_module(
        "src.serving.kernel_lowering.lower_workload")
    deployment = _deployment_for("nvidia_a800")
    first_hardware = HWSpec("configs/hardware/nvidia_a800.yaml")

    def make_graph(prefix, edge_kind="data"):
        source = WorkloadNode(
            f"{prefix}-source",
            WorkloadKind.NORM,
            "main",
            {"rows": 4, "hidden": 64},
            "bf16",
            "rmsnorm",
            rank=0,
        )
        destination = WorkloadNode(
            f"{prefix}-destination",
            WorkloadKind.NORM,
            "main",
            {"rows": 5, "hidden": 64},
            "bf16",
            "rmsnorm",
            rank=0,
        )
        return WorkloadGraph(
            (source, destination),
            (WorkloadEdge(
                source.node_id,
                destination.node_id,
                edge_kind,
            ),),
        )

    first_graph = make_graph("first")
    repeated_graph = make_graph("repeated")
    repeated_without_memo = lower_workload(
        repeated_graph,
        deployment,
        {"nvidia_a800": first_hardware},
    )
    original_compute_spec = lowering_module._compute_spec
    compute_calls = []

    def counted_compute_spec(node, hardware_id, hardware):
        compute_calls.append((node, hardware_id, hardware))
        return original_compute_spec(node, hardware_id, hardware)

    monkeypatch.setattr(
        lowering_module,
        "_compute_spec",
        counted_compute_spec,
    )
    memo = KernelLoweringMemo()
    lower_workload(
        first_graph,
        deployment,
        {"nvidia_a800": first_hardware},
        memo,
    )
    repeated_with_memo = lower_workload(
        repeated_graph,
        deployment,
        {"nvidia_a800": first_hardware},
        memo,
    )

    assert repeated_with_memo == repeated_without_memo
    assert tuple(call.call_id for call in repeated_with_memo.calls) == (
        "repeated-source",
        "repeated-destination",
    )
    assert repeated_with_memo.calls[1].dependencies == (
        "repeated-source",
    )
    assert len(compute_calls) == 2
    assert memo.entry_count == 2
    assert memo.miss_count == 2
    assert memo.hit_count == 2

    lower_workload(
        make_graph("changed-edge", "ready"),
        deployment,
        {"nvidia_a800": first_hardware},
        memo,
    )
    assert len(compute_calls) == 2
    assert memo.entry_count == 2
    assert memo.miss_count == 2
    assert memo.hit_count == 4

    second_hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    lower_workload(
        repeated_graph,
        deployment,
        {"nvidia_a800": second_hardware},
        memo,
    )
    assert len(compute_calls) == 4
    assert memo.entry_count == 4
    assert memo.miss_count == 4
    assert memo.hit_count == 4


def test_kernel_lowering_source_contract_ignores_rank_but_not_semantics():
    lowering_module = importlib.import_module(
        "src.serving.kernel_lowering.lower_workload")
    rank_zero = WorkloadNode(
        "rank-zero", WorkloadKind.NORM, "main",
        {"rows": 4, "hidden": 64}, "bf16", "rmsnorm", rank=0)
    rank_seven = WorkloadNode(
        "rank-seven", WorkloadKind.NORM, "main",
        {"rows": 4, "hidden": 64}, "bf16", "rmsnorm", rank=7)
    different = WorkloadNode(
        "rank-seven-different", WorkloadKind.NORM, "main",
        {"rows": 5, "hidden": 64}, "bf16", "rmsnorm", rank=7)

    assert lowering_module._logical_node_contract(rank_zero) == (
        lowering_module._logical_node_contract(rank_seven))
    assert lowering_module._logical_node_contract(rank_zero) != (
        lowering_module._logical_node_contract(different))


def test_kernel_lowering_memo_does_not_refreeze_source_contract(monkeypatch):
    lowering_module = importlib.import_module(
        "src.serving.kernel_lowering.lower_workload")
    node = WorkloadNode(
        "destination",
        WorkloadKind.NORM,
        "main",
        {"rows": 5, "hidden": 64},
        "bf16",
        "rmsnorm",
        rank=0,
    )
    logical_node_contract = lowering_module._logical_node_contract(node)
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")

    def reject_dependency_refreeze(*unused_args, **unused_kwargs):
        raise AssertionError("pre-frozen contract was serialized again")

    monkeypatch.setattr(
        lowering_module.pickle,
        "dumps",
        reject_dependency_refreeze,
    )
    specs = (lowering_module._compute_spec(
        node,
        "nvidia_a800",
        hardware,
    ),)
    memo = KernelLoweringMemo()

    assert memo.lower(
        node,
        logical_node_contract,
        hardware,
        lambda: specs,
    ) is specs
    assert memo.lower(
        node,
        logical_node_contract,
        hardware,
        lambda: (_ for _ in ()).throw(
            AssertionError("memoized spec was rebuilt")),
    ) is specs
    assert memo.entry_count == 1
    assert memo.miss_count == 1
    assert memo.hit_count == 1


def test_kernel_prediction_memo_reuses_only_exact_run_local_contracts(
        monkeypatch):
    first = WorkloadNode(
        "first",
        WorkloadKind.NORM,
        "main",
        {"rows": 4, "hidden": 64},
        "bf16",
        "rmsnorm",
        rank=0,
    )
    second = WorkloadNode(
        "second",
        WorkloadKind.NORM,
        "main",
        {"rows": 4, "hidden": 64},
        "bf16",
        "rmsnorm",
        rank=0,
    )
    deployment = _deployment_for("nvidia_a800")
    first_hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    plan = lower_workload(
        WorkloadGraph((first, second)),
        deployment,
        {"nvidia_a800": first_hardware},
    )
    original_predict = kernel_predict_module.predict
    calls = []

    def counted_predict(kernel_spec, hardware, method):
        calls.append((kernel_spec, hardware, method))
        return original_predict(kernel_spec, hardware, method)

    monkeypatch.setattr(kernel_predict_module, "predict", counted_predict)
    memo = KernelPredictionMemo()
    predictions = predict_kernel_plan(
        plan,
        {"nvidia_a800": first_hardware},
        memo=memo,
    )

    assert len(calls) == 1
    assert memo.entry_count == 1
    assert memo.miss_count == 1
    assert memo.hit_count == 1
    assert predictions["first"] is predictions["second"]

    repeated = predict_kernel_plan(
        plan,
        {"nvidia_a800": first_hardware},
        memo=memo,
    )
    assert len(calls) == 1
    assert memo.entry_count == 1
    assert memo.miss_count == 1
    assert memo.hit_count == 3
    assert repeated == predictions

    assert (
        kernel_predict_module._contract_cache_key(False)
        != kernel_predict_module._contract_cache_key(0)
    )

    predict_kernel_plan(
        plan,
        {"nvidia_a800": first_hardware},
        method="roofline",
        memo=memo,
    )
    assert len(calls) == 2
    assert memo.entry_count == 2
    assert memo.miss_count == 2
    assert memo.hit_count == 4

    changed = WorkloadNode(
        "changed",
        WorkloadKind.NORM,
        "main",
        {"rows": 5, "hidden": 64},
        "bf16",
        "rmsnorm",
        rank=0,
    )
    changed_plan = lower_workload(
        WorkloadGraph((changed,)),
        deployment,
        {"nvidia_a800": first_hardware},
    )
    predict_kernel_plan(
        changed_plan,
        {"nvidia_a800": first_hardware},
        memo=memo,
    )
    assert len(calls) == 3
    assert memo.entry_count == 3
    assert memo.miss_count == 3
    assert memo.hit_count == 4

    second_hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    predict_kernel_plan(
        plan,
        {"nvidia_a800": second_hardware},
        memo=memo,
    )
    assert len(calls) == 4
    assert memo.entry_count == 4
    assert memo.miss_count == 4
    assert memo.hit_count == 5


def test_kernel_prediction_memo_freezes_each_spec_once_per_plan(monkeypatch):
    node = WorkloadNode(
        "norm",
        WorkloadKind.NORM,
        "main",
        {"rows": 4, "hidden": 64},
        "bf16",
        "rmsnorm",
        rank=0,
    )
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    plan = lower_workload(
        WorkloadGraph((node,)),
        _deployment_for("nvidia_a800"),
        {"nvidia_a800": hardware},
    )
    original_contract_cache_key = kernel_predict_module._contract_cache_key
    key_calls = []

    def counted_contract_cache_key(kernel_spec):
        key_calls.append(kernel_spec)
        return original_contract_cache_key(kernel_spec)

    monkeypatch.setattr(
        kernel_predict_module,
        "_contract_cache_key",
        counted_contract_cache_key,
    )
    memo = KernelPredictionMemo()

    first = predict_kernel_plan(
        plan,
        {"nvidia_a800": hardware},
        memo=memo,
    )
    second = predict_kernel_plan(
        plan,
        {"nvidia_a800": hardware},
        memo=memo,
    )

    assert second == first
    assert key_calls == [plan.calls[0].kernel_spec]
    assert memo.entry_count == 1
    assert memo.miss_count == 1
    assert memo.hit_count == 1


def test_kernel_prediction_groups_exact_rank_specs_and_splits_one_difference(
        monkeypatch):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    deployment = _deployment_for("nvidia_a800")

    def spec_for(rows):
        node = WorkloadNode(
            f"norm-{rows}",
            WorkloadKind.NORM,
            "main",
            {"rows": rows, "hidden": 64},
            "bf16",
            "rmsnorm",
            rank=0,
        )
        return lower_workload(
            WorkloadGraph((node,)),
            deployment,
            {"nvidia_a800": hardware},
        ).calls[0].kernel_spec

    shared = spec_for(4)
    different = spec_for(5)
    calls = tuple(
        KernelCall(
            f"rank-{rank}",
            replace(shared),
            "main",
            rank,
            f"compute:main:rank-{rank}",
            (f"node-{rank}",),
        )
        for rank in range(8)
    ) + (
        KernelCall(
            "rank-8",
            different,
            "main",
            8,
            "compute:main:rank-8",
            ("node-8",),
        ),
    )
    original_predict = kernel_predict_module.predict
    predictor_calls = []

    def counted_predict(kernel_spec, hw, method):
        predictor_calls.append(kernel_spec)
        return original_predict(kernel_spec, hw, method)

    monkeypatch.setattr(kernel_predict_module, "predict", counted_predict)
    memo = KernelPredictionMemo()
    predictions = predict_kernel_plan(
        KernelPlan(calls), {"nvidia_a800": hardware}, memo=memo)

    assert len(predictions) == 9
    assert len(predictor_calls) == 2
    assert memo.entry_count == 2
    assert memo.miss_count == 2
    assert memo.hit_count == 7
    assert len({id(predictions[f"rank-{rank}"]) for rank in range(8)}) == 1
    assert predictions["rank-8"] is not predictions["rank-0"]


def test_simulation_creates_one_prediction_memo_per_top_level_run():
    class OneStepPerRequestBackend:
        config = BackendConfig("main", "colocated", 128, 2, 128, scheduler_pipeline_depth=1)

        def __init__(self):
            self.requests = []
            self.pending = None
            self.step_index = 0

        def add_request(self, request):
            self.requests.append(request)

        def has_work(self):
            return bool(self.requests)

        def plan_step(self, virtual_time_s):
            if self.pending is not None:
                raise RuntimeError("test backend has an incomplete step")
            request = self.requests[0]
            scheduled = ScheduledRequest(
                request.request_id,
                "prefill",
                request.input_tokens,
                1,
                raw_model_output_tokens=1,
            )
            step = BackendStep(
                ScheduleStep(
                    f"main:test:{self.step_index}",
                    "main",
                    (scheduled,),
                    (request.request_id,),
                    attention_kv_layout="paged_varlen",
                ),
                generated_tokens={request.request_id: 1},
            )
            self.pending = step
            self.step_index += 1
            return step

        def complete_step(self, step, virtual_time_s):
            if step is not self.pending:
                raise ValueError("test backend completion does not match")
            self.requests.pop(0)
            self.pending = None

        def close(self):
            return None

    model = load_model_config("tests/serving/fixtures/qwen3_scheduler")
    deployment = _deployment_for("nvidia_a800")
    requests = RequestStream((
        RequestSpec(0.0, "request-0", 2, 1),
        RequestSpec(0.0, "request-1", 2, 1),
    ))
    config = SimulationConfig(
        model,
        deployment,
        requests,
        "analytical_launch",
        "midlayer",
        RuntimeInputs(
            ExpertRoutingInput.not_applicable(),
            "paged_varlen",
            "uniform",
        ),
    )
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")

    first_result = Simulation(
        config,
        OneStepPerRequestBackend(),
        {"nvidia_a800": hardware},
    ).run()
    second_result = Simulation(
        config,
        OneStepPerRequestBackend(),
        {"nvidia_a800": hardware},
    ).run()

    assert len(first_result.steps) == 2
    assert len(second_result.steps) == 2
    assert first_result.execution_cache_stats["execution_mode"] == "accelerated"
    assert first_result.execution_cache_stats["kernel_prediction"]["misses"] > 0
    assert first_result.execution_cache_stats["kernel_lowering"]["misses"] > 0
    assert second_result.execution_cache_stats == first_result.execution_cache_stats


def test_multi_simulation_shares_run_local_memo_with_pd_transfer():
    class OneStepPDBackend:
        def __init__(self, pool_name, role):
            self.config = BackendConfig(pool_name, role, 128, 1, 128, scheduler_pipeline_depth=1)
            self.requests = []
            self.pending = None
            self.step_index = 0

        def add_request(self, request):
            self.requests.append(request)

        def has_work(self):
            return bool(self.requests)

        def plan_step(self, virtual_time_s):
            if self.pending is not None:
                raise RuntimeError("test PD backend has an incomplete step")
            request = self.requests[0]
            if self.config.is_prefill_only:
                phase = "prefill"
                input_tokens = request.input_tokens
                output_tokens = 0
                computed_tokens = 0
            elif self.config.is_decode_only:
                phase = "decode"
                input_tokens = 0
                output_tokens = 1
                computed_tokens = request.input_tokens
            else:
                raise AssertionError("test PD backend has an invalid role")
            scheduled = ScheduledRequest(
                request.request_id,
                phase,
                input_tokens,
                output_tokens,
                computed_tokens=computed_tokens,
                raw_model_output_tokens=1,
            )
            step = BackendStep(
                ScheduleStep(
                    f"{self.config.pool_name}:test:{self.step_index}",
                    self.config.pool_name,
                    (scheduled,),
                    (request.request_id,),
                    attention_kv_layout="paged_varlen",
                ),
                generated_tokens={request.request_id: 1},
            )
            self.pending = step
            self.step_index += 1
            return step

        def complete_step(self, step, virtual_time_s):
            if step is not self.pending:
                raise ValueError("test PD backend completion does not match")
            self.requests.pop(0)
            self.pending = None

        def close(self):
            return None

    model = load_model_config("tests/serving/fixtures/qwen3_scheduler")
    deployment = build_pd_disagg_plan(
        DeploymentConfig(
            "pd_disagg",
            (
                WorkerPoolConfig(
                    "prefill", "prefill", "nvidia_a800", ranks=(0,)),
                WorkerPoolConfig(
                    "decode", "decode", "nvidia_a800", ranks=(1,)),
            ),
            (
                RouteConfig(
                    "prefill",
                    "decode",
                    "kv",
                    "send_recv",
                    "nvlink",
                    "point_to_point",
                    resource_id="pd-test-nvlink",
                    backend="analytical_roofline_v1",
                ),
            ),
        ),
        num_layers=model.num_hidden_layers,
    )
    requests = RequestStream((RequestSpec(0.0, "request", 2, 2),))
    config = SimulationConfig(
        model,
        deployment,
        requests,
        "analytical_launch",
        "midlayer",
        RuntimeInputs(
            ExpertRoutingInput.not_applicable(),
            "paged_varlen",
            "uniform",
        ),
        timing_mode="device_plus_launch",
        host_launch_profiles={
            "nvidia_a800": load_host_launch_profile(
                "configs/host_timing/host_launch_profiles/"
                "nvidia_a800.json"
            ),
        },
    )
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")

    def run_once():
        return MultiSimulation(
            config,
            {
                "prefill": OneStepPDBackend("prefill", "prefill"),
                "decode": OneStepPDBackend("decode", "decode"),
            },
            {"nvidia_a800": hardware},
        ).run()

    first_result = run_once()
    second_result = run_once()

    assert len(first_result.steps) == 2
    assert len(first_result.transfers) == 1
    assert len(second_result.steps) == 2
    assert len(second_result.transfers) == 1
    transfer_timeline = first_result.transfers[0].execution_timeline
    assert transfer_timeline.eager_dispatch_count == 2
    assert transfer_timeline.graph_replay_count == 0
    assert tuple(
        event.runtime_graph_mode
        for event in transfer_timeline.host_intervals
    ) == ("NONE", "NONE")
    first_stats = first_result.execution_cache_stats
    second_stats = second_result.execution_cache_stats
    assert first_stats["execution_mode"] == "accelerated"
    assert first_stats["kernel_prediction"]["misses"] > 0
    assert first_stats["kernel_lowering"]["misses"] > 0
    assert second_stats == first_stats

    streaming_config = replace(config, step_retention="streaming_aggregate")
    streaming_accumulator = StreamingExecutionArtifactAccumulator()
    streaming_result = MultiSimulation(
        streaming_config,
        {
            "prefill": OneStepPDBackend("prefill", "prefill"),
            "decode": OneStepPDBackend("decode", "decode"),
        },
        {"nvidia_a800": hardware},
        streaming_step_consumer=streaming_accumulator.consume,
    ).run()
    assert streaming_result.schedule_trace == first_result.schedule_trace
    assert streaming_result.metrics == first_result.metrics
    assert streaming_result.num_steps == len(first_result.steps)
    assert streaming_result.num_simulation_transfers == len(first_result.transfers)
    streaming_artifact = streaming_accumulator.manifest(
        streaming_result,
        performance_timing_status=PROJECTED_PERFORMANCE_TIMING_STATUS,
    )
    assert streaming_artifact["schema_version"] == 13
    assert streaming_artifact["counts"]["trace_transfers"] == 1
    assert streaming_artifact["counts"]["simulation_transfers"] == 1
    assert streaming_artifact["streaming_aggregate"]["transfer_summary"][
        "trace_transfer_count"] == 1


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("attention_kv_layout", "future_layout", "attention_kv_layout"),
        ("attention_kv_distribution", "future_distribution",
         "attention_kv_distribution"),
    ],
)
def test_runtime_inputs_reject_unknown_attention_categories(
        field, value, message):
    values = {
        "expert_routing": ExpertRoutingInput.seeded_assumption(
            "test_seeded_routing_v1", 0),
        "attention_kv_layout": "contiguous",
        "attention_kv_distribution": "uniform",
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        RuntimeInputs(**values)


def test_seeded_expert_routing_is_invocation_stable_and_auditable():
    model, _, context, _ = _expert_routing_replay_parts()
    routing = ExpertRoutingInput.seeded_assumption(
        "unit_seeded_uniform_v1", 17)
    arguments = {
        "step_id": "main:vllm:0",
        "layer_index": 0,
        "moe_invocation_id": "main:vllm:0:layer-0:moe",
        "schedule_step_sha256": "4" * 64,
        "tokens": 97,
        "num_experts": model.num_experts,
        "topk": model.num_experts_per_tok,
        "model_config_sha256": model.config_sha256,
        "pool_name": "main",
        "tensor_parallel_size": 2,
        "expert_parallel_size": 2,
        "expert_rank_group": context.expert_rank_group,
        "expert_placement_strategy": "linear",
    }

    first = routing.resolve(**arguments)
    second = routing.resolve(**arguments)
    other = routing.resolve(**{
        **arguments,
        "layer_index": 1,
        "moe_invocation_id": "main:vllm:0:layer-1:moe",
    })

    assert first == second
    assert sum(first.tokens_per_expert) == 97
    assert first.routing_provenance["mechanism_source"] == "Modeled"
    assert first.routing_provenance["validation_status"] == "Projected"
    assert first.routing_provenance["root_seed"] == 17
    assert first.routing_provenance["distribution"] == (
        "maximum_entropy_uniform_topk_v1")
    assert first.routing_provenance["hot_expert_count"] is None
    assert first.routing_provenance["hot_expert_weight"] is None
    assert len(first.routing_provenance["policy_descriptor_sha256"]) == 64
    assert (first.routing_provenance["record_sha256"]
            != other.routing_provenance["record_sha256"])


def test_seeded_expert_routing_skew_is_reproducible_and_conservative():
    uniform = seeded_expert_token_counts(
        4096,
        16,
        4,
        source_id="unit-routing-sensitivity-v1",
        root_seed=23,
        moe_invocation_id="main:vllm:0:layer-0:moe",
    )
    skewed = ExpertRoutingInput.skewed_assumption(
        "unit-routing-sensitivity-v1",
        23,
        hot_expert_count=2,
        hot_expert_weight=16,
    )
    arguments = {
        "step_id": "main:vllm:0",
        "layer_index": 0,
        "moe_invocation_id": "main:vllm:0:layer-0:moe",
        "schedule_step_sha256": "4" * 64,
        "tokens": 4096,
        "num_experts": 16,
        "topk": 4,
        "model_config_sha256": "5" * 64,
        "pool_name": "main",
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
        "expert_rank_group": (0,),
        "expert_placement_strategy": "linear",
    }

    first = skewed.resolve(**arguments)
    second = skewed.resolve(**arguments)

    assert first == second
    assert sum(uniform) == sum(first.tokens_per_expert) == 4096 * 4
    assert max(uniform) <= 4096
    assert max(first.tokens_per_expert) <= 4096
    assert sum(sorted(first.tokens_per_expert, reverse=True)[:2]) > (
        2 * sum(first.tokens_per_expert) // 16)
    assert first.routing_provenance["source_kind"] == "explicit_assumption"
    assert first.routing_provenance["distribution"] == (
        "stable_hotset_skew_sensitivity_v1")
    assert first.routing_provenance["hot_expert_count"] == 2
    assert first.routing_provenance["hot_expert_weight"] == 16
    assert first.routing_provenance["schema_version"] == 2


@pytest.mark.parametrize(
    "hot_expert_count,hot_expert_weight,message",
    [
        (0, 2, "hot_expert_count"),
        (1, 1, "hot_expert_weight"),
        (16, 2, "smaller than num_experts"),
    ],
)
def test_seeded_expert_routing_rejects_invalid_skew_points(
        hot_expert_count, hot_expert_weight, message):
    if hot_expert_count == 0 or hot_expert_weight == 1:
        with pytest.raises((TypeError, ValueError), match=message):
            ExpertRoutingInput.skewed_assumption(
                "invalid-skew-v1",
                0,
                hot_expert_count=hot_expert_count,
                hot_expert_weight=hot_expert_weight,
            )
        return
    routing = ExpertRoutingInput.skewed_assumption(
        "invalid-skew-v1",
        0,
        hot_expert_count=hot_expert_count,
        hot_expert_weight=hot_expert_weight,
    )
    with pytest.raises(ValueError, match=message):
        routing.resolve(
            step_id="main:vllm:0",
            layer_index=0,
            moe_invocation_id="main:vllm:0:layer-0:moe",
            schedule_step_sha256="4" * 64,
            tokens=1,
            num_experts=16,
            topk=4,
            model_config_sha256="5" * 64,
            pool_name="main",
            tensor_parallel_size=1,
            expert_parallel_size=1,
            expert_rank_group=(0,),
            expert_placement_strategy="linear",
        )


def test_expert_routing_replay_loads_exact_bytes_and_resolves_one_layer(
        tmp_path):
    model, provenance, context, records = _expert_routing_replay_parts()
    artifact_id = "unit-routing-replay-v1"
    canonical_sha256 = expert_routing_replay_payload_sha256(
        artifact_id, provenance, context, records)
    payload = {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "canonical_payload_sha256": canonical_sha256,
        "provenance": provenance.manifest(),
        "context": context.manifest(),
        "records": tuple(record.manifest() for record in records),
    }
    path = tmp_path / "routing-replay.json"
    serialized = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(serialized)

    replay = load_expert_routing_replay(path)
    routing = ExpertRoutingInput.replay_observation(
        replay, context.request_stream_sha256)
    with pytest.raises(RuntimeError, match="effective vLLM runtime"):
        routing.bind_vllm_runtime(
            version=provenance.vllm_version,
            source_commit=provenance.vllm_source_commit,
            package_python_manifest_sha256="5" * 64,
            model_config_sha256=model.config_sha256,
        )
    with pytest.raises(RuntimeError, match="must bind"):
        routing.resolve(
            step_id="main:vllm:0",
            layer_index=0,
            moe_invocation_id="main:vllm:0:layer-0:moe",
            schedule_step_sha256="4" * 64,
            tokens=1,
            num_experts=model.num_experts,
            topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256,
            pool_name="main",
            tensor_parallel_size=2,
            expert_parallel_size=2,
            expert_rank_group=(91, 7),
            expert_placement_strategy="linear",
        )
    routing = routing.bind_vllm_runtime(
        version=provenance.vllm_version,
        source_commit=provenance.vllm_source_commit,
        package_python_manifest_sha256=(
            provenance.vllm_package_manifest_sha256),
        model_config_sha256=model.config_sha256,
    )
    resolution = routing.resolve(
        step_id="main:vllm:0",
        layer_index=0,
        moe_invocation_id="main:vllm:0:layer-0:moe",
        schedule_step_sha256="4" * 64,
        tokens=1,
        num_experts=model.num_experts,
        topk=model.num_experts_per_tok,
        model_config_sha256=model.config_sha256,
        pool_name="main",
        tensor_parallel_size=2,
        expert_parallel_size=2,
        expert_rank_group=(91, 7),
        expert_placement_strategy="linear",
    )

    assert replay.source_sha256 == hashlib.sha256(serialized).hexdigest()
    assert replay.canonical_payload_sha256 == canonical_sha256
    assert resolution.tokens_per_expert == (1, 0)
    assert resolution.routing_provenance["mechanism_source"] == "Captured"
    assert resolution.routing_provenance["validation_status"] == "Validated"
    routing.assert_replay_consumed(tuple(
        record.moe_invocation_id for record in records))
    with pytest.raises(RuntimeError, match="coverage disagrees"):
        routing.assert_replay_consumed((records[0].moe_invocation_id,))
    with pytest.raises(ValueError, match="missing the exact step/layer"):
        routing.resolve(
            step_id="main:vllm:1",
            layer_index=0,
            moe_invocation_id="main:vllm:1:layer-0:moe",
            schedule_step_sha256="4" * 64,
            tokens=1,
            num_experts=model.num_experts,
            topk=model.num_experts_per_tok,
            model_config_sha256=model.config_sha256,
            pool_name="main",
            tensor_parallel_size=2,
            expert_parallel_size=2,
            expert_rank_group=(91, 7),
            expert_placement_strategy="linear",
        )


@pytest.mark.parametrize(
    "counts,tokens,topk,expected_message",
    (
        ((1, 0, 0), 1, 1, "count length"),
        ((0, 0), 1, 1, r"tokens \* top-k"),
        ((3, 1), 2, 2, "cannot exceed tokens"),
    ),
)
def test_expert_routing_replay_rejects_invalid_expert_histograms(
        tmp_path, counts, tokens, topk, expected_message):
    _, provenance, context, records = _expert_routing_replay_parts(
        counts=counts, tokens=tokens, topk=topk)
    artifact_id = "invalid-routing-replay-v1"

    path = tmp_path / "invalid-routing-replay.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "artifact_id": artifact_id,
        "canonical_payload_sha256": expert_routing_replay_payload_sha256(
            artifact_id, provenance, context, records),
        "provenance": provenance.manifest(),
        "context": context.manifest(),
        "records": tuple(record.manifest() for record in records),
    }))
    with pytest.raises(ValueError, match=expected_message):
        load_expert_routing_replay(path)


def test_expert_routing_resolution_rejects_duplicate_expert_histogram(
        monkeypatch):
    monkeypatch.setattr(
        "src.serving.workload.runtime_inputs.seeded_expert_token_counts",
        lambda *_args, **_kwargs: (3, 1),
    )
    routing = ExpertRoutingInput.seeded_assumption(
        "invalid-duplicate-expert-routing-v1", 0)

    with pytest.raises(ValueError, match="cannot exceed tokens"):
        routing.resolve(
            step_id="main:vllm:0",
            layer_index=0,
            moe_invocation_id="main:vllm:0:layer-0:moe",
            schedule_step_sha256="4" * 64,
            tokens=2,
            num_experts=2,
            topk=2,
            model_config_sha256="5" * 64,
            pool_name="main",
            tensor_parallel_size=1,
            expert_parallel_size=1,
            expert_rank_group=(0,),
            expert_placement_strategy="linear",
        )


def test_expert_routing_resolution_accepts_valid_topk_histogram():
    resolution = ExpertRoutingInput.seeded_assumption(
        "valid-topk-routing-v1", 0).resolve(
            step_id="main:vllm:0",
            layer_index=0,
            moe_invocation_id="main:vllm:0:layer-0:moe",
            schedule_step_sha256="4" * 64,
            tokens=2,
            num_experts=2,
            topk=2,
            model_config_sha256="5" * 64,
            pool_name="main",
            tensor_parallel_size=1,
            expert_parallel_size=1,
            expert_rank_group=(0,),
            expert_placement_strategy="linear",
        )

    assert resolution.tokens_per_expert == (2, 2)


def test_expert_routing_replay_rejects_hash_and_step_identity_mismatch(
        tmp_path):
    _, provenance, context, records = _expert_routing_replay_parts()
    hash_path = tmp_path / "hash-mismatch.json"
    hash_path.write_text(json.dumps({
        "schema_version": 1,
        "artifact_id": "hash-mismatch-v1",
        "canonical_payload_sha256": "4" * 64,
        "provenance": provenance.manifest(),
        "context": context.manifest(),
        "records": tuple(record.manifest() for record in records),
    }))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_expert_routing_replay(hash_path)

    wrong_step_records = tuple(
        replace(
            record,
            step_id="other:vllm:0",
            moe_invocation_id=(
                f"other:vllm:0:layer-{record.layer_index}:moe"),
        )
        for record in records
    )
    artifact_id = "step-mismatch-v1"
    step_path = tmp_path / "step-mismatch.json"
    step_path.write_text(json.dumps({
        "schema_version": 1,
        "artifact_id": artifact_id,
        "canonical_payload_sha256": expert_routing_replay_payload_sha256(
            artifact_id, provenance, context, wrong_step_records),
        "provenance": provenance.manifest(),
        "context": context.manifest(),
        "records": tuple(
            record.manifest() for record in wrong_step_records),
    }))
    with pytest.raises(ValueError, match="Scheduler pool"):
        load_expert_routing_replay(step_path)


@pytest.mark.parametrize(
    "hardware_id,expected_variants",
    [
        ("nvidia_a800", [
            "moe_topk", "moe_align", "moe_count_and_sort", "w1",
            "moe_act_and_mul", "w2", "moe_reduce",
        ]),
        ("metax_c500", [
            "moe_router_logit_cast", "moe_topk",
            "moe_routing_weight_norm",
            "moe_routing_weight_scale", "moe_routed_id_widen",
            "moe_routed_id_extent",
            "moe_routed_id_extent", "moe_fill", "moe_fill", "moe_fill", "moe_count_and_sort", "moe_align_opt_128",
            "moe_expert_map_lookup", "w1",
            "moe_act_and_mul", "w2", "moe_sum",
        ]),
        ("ascend_910b", [
            "moe_gating_topk", "moe_init_routing_v2", "moe_cast",
            "w1", "moe_swiglu", "w2", "moe_abs",
            "moe_token_unpermute",
        ]),
    ],
)
def test_fused_moe_uses_complete_hardware_recipe(
        hardware_id, expected_variants):
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    routing_attributes = _test_moe_attributes(16, 128, 8)
    node = WorkloadNode(
        "moe", WorkloadKind.FUSED_MOE, "main",
        {"tokens": 16, "experts": 128, "topk": 8,
         "hidden": 2048, "intermediate": 768},
        "bf16", "full", rank=0, attributes={
            **routing_attributes,
            "launch_policy": "analytical_launch",
        })
    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hardware})

    assert [call.kernel_spec.variant for call in plan.calls] == expected_variants
    assert [call.call_id for call in plan.calls] == [
        f"moe:{index}" for index in range(len(expected_variants))]
    assert all(
        call.dependencies == (() if index == 0 else (f"moe:{index - 1}",))
        for index, call in enumerate(plan.calls))
    if hardware_id == "ascend_910b":
        casts = tuple(
            call.kernel_spec for call in plan.calls
            if call.kernel_spec.variant == "moe_cast")
        assert tuple(spec.attributes["cast_role"] for spec in casts) == (
            "expert_tokens",)
        swiglu, = tuple(
            call.kernel_spec for call in plan.calls
            if call.kernel_spec.variant == "moe_swiglu")
        assert swiglu.attributes["routed_rows"] == 16 * 8


def test_ascend_fused_moe_public_v3_recipe_has_no_legacy_casts():
    hardware_id = "ascend_910b"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    routing_attributes = _test_moe_attributes(16, 512, 10)
    node = WorkloadNode(
        "moe", WorkloadKind.FUSED_MOE, "main",
        {"tokens": 16, "experts": 512, "topk": 10,
         "hidden": 2048, "intermediate": 512},
        "bf16", "full", rank=0, attributes={
            **routing_attributes,
            "launch_policy": "analytical_launch",
        })
    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hardware})

    assert [call.kernel_spec.variant for call in plan.calls] == [
        "moe_gating_topk", "moe_init_routing_v3", "w1", "moe_swiglu",
        "w2", "moe_abs", "moe_token_unpermute",
    ]
    predictions = predict_kernel_plan(plan, {hardware_id: hardware})
    assert all(prediction.t_total_ns > 0 for prediction in predictions.values())


@pytest.mark.parametrize("intermediate", (14336, 16384))
def test_metax_large_fused_moe_lowers_one_fused_activation(intermediate):
    """The deployed activation is one launch however large the projection."""

    hardware_id = "metax_c500"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    tokens, experts, topk = 16384, 16, 10
    node = WorkloadNode(
        "moe", WorkloadKind.FUSED_MOE, "main",
        {
            "tokens": tokens,
            "experts": experts,
            "topk": topk,
            "hidden": 4096 if intermediate == 14336 else 6144,
            "intermediate": intermediate,
        },
        "bf16", "full", rank=0, attributes={
            **_test_moe_attributes(tokens, experts, topk),
            "launch_policy": "analytical_launch",
        })

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hardware})

    variants = [call.kernel_spec.variant for call in plan.calls]
    # vllm::act_and_mul_kernel is a hand-written launch, not an ATen
    # TensorIterator, so no index split applies to it at any size.
    assert variants.count("moe_act_and_mul") == 1
    assert "moe_silu" not in variants and "moe_mul" not in variants
    activation, = tuple(
        call.kernel_spec for call in plan.calls
        if call.kernel_spec.variant == "moe_act_and_mul")
    assert activation.launch.grid_size > 0


def test_a800_large_fused_moe_lowers_ordered_reduce_shards():
    hardware_id = "nvidia_a800"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    tokens = 32768
    experts = 256
    topk = 8
    node = WorkloadNode(
        "moe", WorkloadKind.FUSED_MOE, "main",
        {
            "tokens": tokens,
            "experts": experts,
            "topk": topk,
            "hidden": 7168,
            "intermediate": 2048,
        },
        "bf16", "full", rank=0, attributes={
            **_test_moe_attributes(tokens, experts, topk),
            "launch_policy": "analytical_launch",
        })

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hardware})
    reductions = tuple(
        call for call in plan.calls
        if call.kernel_spec.variant == "moe_reduce")

    assert len(reductions) == 2
    assert tuple(
        call.kernel_spec.attributes["work_fraction"]
        for call in reductions) == (0.5, 0.5)
    assert tuple(call.kernel_spec.launch.grid for call in reductions) == (
        (229376, 1, 1),
        (229376, 1, 1),
    )
    assert tuple(call.kernel_spec.launch.block for call in reductions) == (
        (32, 4, 1),
        (32, 4, 1),
    )
    assert reductions[1].dependencies == (reductions[0].call_id,)


def test_metax_fused_moe_serving_binds_three_fill_roles_in_source_order(
        monkeypatch):
    physical_specs_module = importlib.import_module(
        "src.kernel.fused_moe_physical_specs")
    hardware_id = "metax_c500"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    routing_attributes = _test_moe_attributes(16, 128, 8)
    node = WorkloadNode(
        "moe", WorkloadKind.FUSED_MOE, "main",
        {"tokens": 16, "experts": 128, "topk": 8,
         "hidden": 2048, "intermediate": 768},
        "bf16", "full", rank=0, attributes={
            **routing_attributes,
            "launch_policy": "analytical_launch",
        })

    monkeypatch.setattr(
        physical_specs_module, "select_fused_moe_recipe",
        lambda *_args, **_kwargs: (
            "moe_fill", "moe_fill", "moe_fill"))

    def _fill_launch(_hardware_id, _kind, _shape, _dtype, variant,
                     _attributes, *, hardware):
        del hardware
        return LaunchConfig(
            algorithm=variant,
            grid=(1, 1, 1),
            block=(256, 1, 1),
            execution_variant=variant,
        )

    monkeypatch.setattr(
        physical_specs_module, "select_launch_config", _fill_launch)

    specs = _fused_moe_specs(node, hardware_id, hardware)

    assert [spec.attributes["fill_role"] for spec in specs] == [
        "sorted_token_ids",
        "expert_ids",
        "num_tokens_post_padded",
    ]


def test_metax_large_moe_splits_sum_work_across_two_invocations():
    hardware_id = "metax_c500"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    experts = 256
    tokens = 16384
    topk = 10
    routing_attributes = _test_moe_attributes(tokens, experts, topk)
    node = WorkloadNode(
        "moe", WorkloadKind.FUSED_MOE, "main",
        {
            "tokens": tokens,
            "experts": experts,
            "topk": topk,
            "hidden": 7168,
            "intermediate": 2048,
        },
        "bf16", "full", rank=0, attributes={
            **routing_attributes,
            "launch_policy": "analytical_launch",
        })

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hardware})
    # topk, fill x3, the routed-token histogram, the alignment kernel, W1,
    # the one fused activation, W2 and the split moe_sum pair.
    assert len(plan.calls) == 18
    assert sum(
        call.kernel_spec.variant == "moe_sum" for call in plan.calls) == 2
    assert all(
        call.kernel_spec.launch.grid_size > 0 for call in plan.calls)


@pytest.mark.parametrize("m, expected_variant", [(1, "gemv"), (16, "gemm")])
def test_metax_single_token_gemm_lowers_to_measured_execution_variant(
        m, expected_variant):
    hw = HWSpec("configs/hardware/metax_c500.yaml")
    node = WorkloadNode(
        "linear", WorkloadKind.LINEAR, "main",
        {"b": 1, "m": m, "n": 6144, "k": 4096}, "bf16", "gemm",
        rank=0, attributes=_test_gemm_attributes(
            "qkv_proj", launch_policy="injected_launch"))
    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for("metax_c500"),
        {"metax_c500": hw})
    assert plan.calls[0].kernel_spec.variant == expected_variant
    predictions = predict_kernel_plan(plan, {"metax_c500": hw})
    assert [call.call_id for call in plan.calls] == ["linear"]
    serving_prediction = predictions["linear"]
    direct_prediction = predict(plan.calls[0].kernel_spec, hw)
    assert serving_prediction == direct_prediction
    assert serving_prediction.t_total_s > 0


@pytest.mark.parametrize(
    ("hardware_id", "m", "expected_tile", "expected_grid"),
    (
        ("metax_c500", 12, (32, 256, 128), (1, 76, 1)),
        ("ascend_910b", 16, (16, 256, 64), (19, 1, 1)),
    ),
)
def test_qwen3_4b_gate_up_uses_one_gemm_launch_contract(
        hardware_id, m, expected_tile, expected_grid):
    """Serving and direct Kernel evaluation normalize one physical launch."""

    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    shape = {"b": 1, "m": m, "n": 19456, "k": 2560}
    attributes = _test_gemm_attributes(
        "gate_up_proj", launch_policy="injected_launch")
    node = WorkloadNode(
        "gate_up", WorkloadKind.LINEAR, "main", shape, "bf16", "gemm",
        rank=0, attributes=attributes)

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hardware})
    serving_call, = plan.calls
    direct_launch = select_launch_config(
        hardware_id, KernelKind.GEMM, shape, "bf16", "gemm", attributes,
        hardware=hardware)
    direct_spec = KernelSpec(
        kind=KernelKind.GEMM,
        hardware_id=hardware_id,
        shape=shape,
        dtype="bf16",
        variant=direct_launch.execution_variant,
        launch=direct_launch,
        attributes={},
    )

    assert serving_call.kernel_spec == direct_spec
    assert direct_launch.tiling is not None
    assert tuple(direct_launch.tiling.tile_shape[axis]
                 for axis in ("m", "n", "k")) == expected_tile
    assert direct_launch.grid == expected_grid
    assert type(create_predictor(serving_call.kernel_spec.kind)) is type(
        create_predictor(direct_spec.kind))
    assert predict(serving_call.kernel_spec, hardware) == predict(
        direct_spec, hardware)

    if hardware_id == "ascend_910b":
        program = direct_launch.source_program
        assert program is not None
        assert program.aic_workers == 19
        assert (program.single_core_m, program.single_core_n,
                program.single_core_k) == (16, 1024, 2560)
        assert (program.depth_a1, program.depth_b1) == (1, 2)
        assert program.traversal_order == "persistent_n_then_m"


def _split_k_reducer_test_launch(*, algorithm, dram_bytes):
    return LaunchConfig(
        algorithm=algorithm,
        grid=(7, 3, 1),
        block=(128, 1, 1),
        parameters={
            "split_k_partial_bytes_per_element": 4,
            "split_k_partial_dram_bytes_per_element": dram_bytes,
        },
        execution_variant="gemm_reduce",
    )


def test_metax_split_k_lowering_serializes_main_then_reducer(monkeypatch):
    lowering_module = importlib.import_module(
        "src.serving.kernel_lowering.lower_workload")
    hardware_id = "metax_c500"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    shape = {"b": 1, "m": 8, "n": 32, "k": 64}
    attributes = _test_gemm_attributes(
        "qkv_proj", launch_policy="injected_launch")

    def main_selector(_hardware_id, kind, selected_shape, dtype, variant,
                      selected_attributes, *, hardware):
        del dtype, hardware
        assert _hardware_id == hardware_id
        assert kind is KernelKind.GEMM
        assert selected_shape == shape
        assert variant == "gemm"
        assert selected_attributes == attributes
        return LaunchConfig(
            algorithm="observed_mcblas_gemm:gemm",
            grid=(7, 3, 1),
            block=(128, 1, 1),
            tiling=TilingSpec(
                tile_shape={"m": 8, "n": 16, "k": 32}, split_k=2),
            parameters={
                "compulsory_traffic": 0,
                "weight_tile_rows": 0,
                "mte2_tile_bytes": 0,
                "aiv_mixed": 0,
                "n_major_traffic": 0,
                "split_k_partial_bytes_per_element": 4,
                "split_k_partial_dram_bytes_per_element": 4,
            },
            execution_variant="gemm",
        )

    reducer_attributes = []

    def reducer_selector(_hardware_id, selected_shape, dtype, variant,
                         selected_attributes, *, hardware):
        del dtype, hardware
        assert _hardware_id == hardware_id
        assert selected_shape == shape
        assert variant == "gemm"
        reducer_attributes.append(dict(selected_attributes))
        return _split_k_reducer_test_launch(
            algorithm="observed_mcblas_gemm_aux:gemm_reduce", dram_bytes=4)

    monkeypatch.setattr(lowering_module, "select_launch_config", main_selector)
    monkeypatch.setattr(
        lowering_module, "select_gemm_reducer_launch", reducer_selector)
    node = WorkloadNode(
        "linear", WorkloadKind.LINEAR, "main", shape, "bf16", "gemm",
        rank=0, attributes=attributes)

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hardware})

    assert [call.call_id for call in plan.calls] == ["linear:0", "linear:1"]
    assert plan.calls[1].dependencies == ("linear:0",)
    reducer = plan.calls[1].kernel_spec
    assert reducer.kind is KernelKind.VECTOR
    assert reducer.shape["element_count"] == 256
    assert reducer.attributes == {"split_k": 2}
    assert reducer.launch.parameters == {
        "split_k_partial_bytes_per_element": 4.0,
        "split_k_partial_dram_bytes_per_element": 4.0,
    }
    resource_input, _ = create_predictor(
        reducer.kind).resource_input(reducer, hardware)
    assert resource_input.demand_per_resource["dram"] > 0
    assert reducer_attributes == [attributes]


def test_eager_attention_passes_full_qk_pv_attributes_to_split_k_reducers(
        monkeypatch):
    lowering_module = importlib.import_module(
        "src.serving.kernel_lowering.lower_workload")
    hardware_id = "nvidia_a800"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    common_attributes = {
        "launch_policy": "injected_launch",
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
    }
    selected = []

    def main_selector(_hardware_id, kind, shape, dtype, variant, attributes,
                      *, hardware):
        del dtype, hardware
        assert _hardware_id == hardware_id
        selected.append((kind, variant, dict(attributes)))
        if kind is KernelKind.VECTOR:
            return LaunchConfig(
                algorithm="softmax", grid=(1, 1, 1), block=(32, 1, 1),
                execution_variant="softmax")
        return LaunchConfig(
            algorithm="observed_main", grid=(11, 1, 1), block=(128, 1, 1),
            tiling=TilingSpec(
                tile_shape={"m": 8, "n": 16, "k": 32}, split_k=2),
            parameters={
                "compulsory_traffic": 0,
                "weight_tile_rows": 0,
                "mte2_tile_bytes": 0,
                "aiv_mixed": 0,
                "n_major_traffic": 0,
                "split_k_partial_bytes_per_element": 4,
                "split_k_partial_dram_bytes_per_element": 0,
            }, execution_variant="gemm")

    reduced = []

    def reducer_selector(_hardware_id, shape, dtype, variant, attributes,
                         *, hardware):
        del dtype, hardware
        assert _hardware_id == hardware_id
        reduced.append((shape, variant, dict(attributes)))
        return _split_k_reducer_test_launch(
            algorithm="observed_gemm_aux:gemm_reduce", dram_bytes=0)

    monkeypatch.setattr(lowering_module, "select_launch_config", main_selector)
    monkeypatch.setattr(
        lowering_module, "select_gemm_reducer_launch", reducer_selector)
    node = WorkloadNode(
        "eager", WorkloadKind.EAGER_ATTN, "main",
        {"batch": 1, "q_len": 2, "kv_len": 4, "head_dim": 8,
         "num_query_heads": 3, "num_kv_heads": 3},
        "bf16", "decode", rank=0, attributes=common_attributes)

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hardware})

    assert [call.kernel_spec.variant for call in plan.calls] == [
        "gemm", "gemm_reduce", "softmax", "gemm", "gemm_reduce"]
    assert [call.dependencies for call in plan.calls] == [
        (), ("eager:0",), ("eager:1",), ("eager:2",), ("eager:3",)]
    assert [entry[2]["position"] for entry in reduced] == [
        "attention_qk", "attention_pv"]
    selected_gemms = [
        entry for entry in selected if entry[0] is KernelKind.GEMM]
    assert [entry[2] for entry in selected_gemms] == [
        entry[2] for entry in reduced]
    assert reduced[0][2]["trans_b"] is True
    assert reduced[1][2]["trans_b"] is False
    assert all(
        all(name in entry[2] for name in (
            "a_layout", "b_layout", "c_layout", "trans_a", "trans_b",
            "epilogue", "has_bias", "position", "tensor_parallel_size",
            "expert_parallel_size", "launch_policy"))
        for entry in reduced)
    for call in (plan.calls[1], plan.calls[4]):
        resource_input, _ = create_predictor(
            call.kernel_spec.kind).resource_input(call.kernel_spec, hardware)
        assert resource_input.demand_per_resource["l2_cache"] > 0


def test_workload_graph_rejects_cycles():
    nodes = (
        WorkloadNode("a", WorkloadKind.VECTOR, "main", {"elements": 1}),
        WorkloadNode("b", WorkloadKind.VECTOR, "main", {"elements": 1}),
    )
    with pytest.raises(ValueError, match="acyclic"):
        WorkloadGraph(nodes, (WorkloadEdge("a", "b"), WorkloadEdge("b", "a")))


def test_workload_graph_and_kernel_plan_reject_empty_inputs():
    with pytest.raises(ValueError, match="at least one node"):
        WorkloadGraph(())
    with pytest.raises(ValueError, match="at least one kernel call"):
        KernelPlan(())


def test_workload_node_rejects_mapping_lookalikes():
    with pytest.raises(TypeError, match="shape must be a mapping"):
        WorkloadNode("bad", WorkloadKind.VECTOR, "main", [("elements", 1)])


@pytest.mark.parametrize("hardware_id", ("nvidia_a800", "metax_c500"))
def test_split_flash_attention_lowers_main_then_combine(hardware_id):
    hw = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    node = WorkloadNode(
        "attention", WorkloadKind.FLASH_ATTN, "main",
        {"batch": 1, "q_len": 1, "kv_len": 8192, "head_dim": 128,
         "num_query_heads": 32, "num_kv_heads": 8},
        "bf16", "decode", rank=0, attributes={
            "causal": False,
            "kv_layout": "contiguous",
            "kv_dist": "uniform",
            "q_lens": (1,),
            "kv_lens": (8192,),
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
            "launch_policy": "analytical_launch",
            "split_policy": "serving",
        })
    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hw})
    if hardware_id == "metax_c500":
        assert [call.kernel_spec.kind for call in plan.calls] == [
            KernelKind.VECTOR, KernelKind.FLASH_ATTN, KernelKind.MEMCPY]
        output_fill = plan.calls[0].kernel_spec
        assert output_fill.attributes["operation"] == "fill"
        assert output_fill.shape["element_count"] == 1 * 1 * 32 * 128
        assert output_fill.launch.grid == (1, 1, 1)
        assert output_fill.launch.block == (1024, 1, 1)
        assert plan.calls[1].kernel_spec.launch.tiling.num_kv_splits == 1
        output_copy = plan.calls[2].kernel_spec
        assert output_copy.transfer.direction == "d2d"
        assert output_copy.transfer.bytes == 1 * 32 * 128 * 2
        inputs = [
            create_predictor(plan.calls[1].kernel_spec.kind).resource_input(
                plan.calls[1].kernel_spec, hw)[0]
        ]
        assert inputs[0].pipeline_plan is None
        assert inputs[0].support_level == "aggregate_only"
        return
    assert [call.kernel_spec.variant for call in plan.calls] == [
        "elementwise", "decode", "combine"]
    output_fill = plan.calls[0].kernel_spec
    assert output_fill.attributes["operation"] == "fill"
    assert output_fill.shape["element_count"] == 1 * 1 * 32 * 128
    assert output_fill.launch.grid == (4, 1, 1)
    assert output_fill.launch.block == (128, 1, 1)
    assert all(
        call.source_node_ids == (node.node_id,) for call in plan.calls)
    assert plan.calls[1].dependencies == (plan.calls[0].call_id,)
    splits = plan.calls[2].kernel_spec.launch.tiling.num_kv_splits
    assert splits > 1
    inputs = [
        create_predictor(call.kernel_spec.kind).resource_input(
            call.kernel_spec, hw)[0]
        for call in plan.calls
    ]
    scratch = 32 * splits * (128 + 1) * 4
    main_write = next(
        path for path in inputs[1].memory_paths
        if path.name == "global_write")
    combine_read = next(
        path for path in inputs[2].memory_paths
        if path.name == "global_read")
    if hardware_id == "nvidia_a800":
        assert (
            "flash_attention_split_scratch_dram_lifecycle_unavailable"
            not in inputs[1].limitations)
        assert (
            "flash_attention_split_scratch_dram_lifecycle_unavailable"
            not in inputs[2].limitations)
        assert [node.resource_role for node in main_write.nodes] == [
            "l2_cache", "dram"]
        assert [node.resource_role for node in combine_read.nodes] == [
            "l2_cache", "dram"]
    else:
        raise AssertionError("unexpected split FlashAttention hardware")


def test_metax_injected_paged_decode_lowers_and_predicts_split_combine():
    hardware_id = "metax_c500"
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    node = WorkloadNode(
        "attention", WorkloadKind.FLASH_ATTN, "main",
        {"batch": 1, "q_len": 1, "kv_len": 8192, "head_dim": 128,
         "num_query_heads": 32, "num_kv_heads": 8},
        "bf16", "decode", rank=0, attributes={
            "causal": True,
            "kv_layout": "paged_varlen",
            "kv_dist": "uniform",
            "q_lens": (1,),
            "kv_lens": (8192,),
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
            "launch_policy": "injected_launch",
            "split_policy": "serving",
        })

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hardware})

    assert [call.kernel_spec.kind for call in plan.calls] == [
        KernelKind.VECTOR, KernelKind.FLASH_ATTN, KernelKind.MEMCPY]
    output_fill, main, output_copy = (call.kernel_spec for call in plan.calls)
    assert output_fill.attributes["operation"] == "fill"
    assert output_fill.shape["element_count"] == 1 * 32 * 128
    assert main.launch.algorithm == (
        "metax_flash_attention_trait_direct:decode")
    assert main.launch.tiling.num_kv_splits == 1
    assert main.launch.grid == (8, 1, 1)
    assert main.launch.block == (64, 1, 1)
    assert dict(main.launch.tiling.tile_shape) == {
        "q": 16, "kv": 16, "head": 128,
        "query_heads_per_workgroup": 4}
    assert main.launch.parameters["page_table_bytes_per_kv_panel"] == 0.0
    assert output_copy.kind is KernelKind.MEMCPY
    assert output_copy.transfer.algorithm == "native_d2d_flash_attention_copy"
    assert output_copy.transfer.direction == "d2d"
    assert output_copy.transfer.link == "dram"
    assert output_copy.transfer.bytes == 1 * 32 * 128 * 2
    assert all(
        predict(spec, hardware).t_total_ns > 0
        for spec in (output_fill, main, output_copy))


def test_metax_split_forward_zero_fills_the_whole_call_once():
    hardware_id = "metax_c500"
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    q_lens = (1, 1, 9)
    node = WorkloadNode(
        "attention", WorkloadKind.FLASH_ATTN, "main",
        {"batch": 3, "q_len": 9, "kv_len": 512, "head_dim": 128,
         "num_query_heads": 32, "num_kv_heads": 8},
        "bf16", "prefill", rank=0, attributes={
            "causal": True,
            "kv_layout": "paged_varlen",
            "kv_dist": "uniform",
            "q_lens": q_lens,
            "kv_lens": (300, 200, 9),
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
            "launch_policy": "injected_launch",
            "split_policy": "serving",
        })

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hardware})

    specs = [call.kernel_spec for call in plan.calls]
    fills = [spec for spec in specs
             if spec.kind is KernelKind.VECTOR and spec.launch.algorithm
             == "observed_flash_attention_output_fill:elementwise"]
    # Both pathways share one output buffer, zero-filled once up front.
    assert len(fills) == 1
    assert fills[0].shape["element_count"] == sum(q_lens) * 32 * 128
    # Only the prefill pathway is preceded by the CUB prefix scan pair.
    scans = [spec.launch.algorithm for spec in specs
             if spec.kind is KernelKind.VECTOR
             and spec.launch.algorithm.startswith("observed_cub_device_scan")]
    assert scans == ["observed_cub_device_scan:scan_init",
                     "observed_cub_device_scan:scan"]
    assert specs.index(fills[0]) < next(
        index for index, spec in enumerate(specs)
        if spec.kind is KernelKind.FLASH_ATTN)
    assert sum(
        call.kernel_spec.kind is KernelKind.FLASH_ATTN
        and call.kernel_spec.variant in ("prefill", "decode")
        for call in plan.calls) == 2


def test_metax_full_graph_decode_lowers_native_splitkv_body_and_combine():
    hardware_id = "metax_c500"
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    node = WorkloadNode(
        "attention", WorkloadKind.FLASH_ATTN, "main",
        {"batch": 56, "q_len": 1, "kv_len": 839, "head_dim": 128,
         "num_query_heads": 32, "num_kv_heads": 8},
        "bf16", "decode", rank=0, attributes={
            "causal": True,
            "kv_layout": "paged_varlen",
            "kv_dist": "skew",
            "q_lens": (1,) * 56,
            "kv_lens": (
                745, 23, 123, 71, 33, 69, 48, 88, 677, 236, 15, 386,
                408, 24, 750, 252, 27, 409, 331, 67, 418, 94, 186, 31,
                33, 193, 512, 254, 389, 839, 390, 9, 683, 29, 745, 388,
                61, 335, 14, 40, 164, 791, 32, 60, 26, 337, 60, 275, 7,
                50, 159, 307, 773, 316, 21, 76),
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
            "launch_policy": "injected_launch",
            "split_policy": "serving",
            "graph_runtime_mode": "FULL",
        })

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hardware})

    assert [call.kernel_spec.kind for call in plan.calls] == [
        KernelKind.VECTOR, KernelKind.FLASH_ATTN, KernelKind.FLASH_ATTN,
        KernelKind.MEMCPY]
    output_fill, main, combine, output_copy = (
        call.kernel_spec for call in plan.calls)
    assert output_fill.shape["element_count"] == 56 * 32 * 128
    assert output_fill.launch.grid == (28, 1, 1)
    assert main.launch.grid == (1, 448, 5)
    assert combine.launch.grid == (224, 1, 1)
    assert combine.launch.tiling.num_kv_splits == 5
    assert output_copy.kind is KernelKind.MEMCPY
    assert output_copy.transfer.bytes == 56 * 32 * 128 * 2


def test_a800_paged_mha_decode_lowers_without_combine():
    hardware_id = "nvidia_a800"
    hw = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    node = WorkloadNode(
        "attention", WorkloadKind.FLASH_ATTN, "main",
        {"batch": 2, "q_len": 1, "kv_len": 8192, "head_dim": 128,
         "num_query_heads": 8, "num_kv_heads": 8},
        "bf16", "decode", rank=0, attributes={
            "causal": False,
            "kv_layout": "paged",
            "kv_dist": "uniform",
            "q_lens": (1, 1),
            "kv_lens": (8192, 8192),
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
            "launch_policy": "analytical_launch",
            "split_policy": "serving",
        })

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hw})

    assert [call.kernel_spec.variant for call in plan.calls] == [
        "elementwise", "decode"]
    output_fill, main = (call.kernel_spec for call in plan.calls)
    assert output_fill.attributes["operation"] == "fill"
    assert output_fill.shape["element_count"] == 2 * 1 * 8 * 128
    launch = main.launch
    assert launch.tiling.num_kv_splits == 1
    assert launch.grid == (1, 2, 8)


def test_a800_paged_gqa_decode_lowers_three_runtime_copies():
    hardware_id = "nvidia_a800"
    hw = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    node = WorkloadNode(
        "attention", WorkloadKind.FLASH_ATTN, "main",
        {"batch": 2, "q_len": 1, "kv_len": 512, "head_dim": 128,
         "num_query_heads": 32, "num_kv_heads": 8},
        "bf16", "decode", rank=0, attributes={
            "causal": False,
            "kv_layout": "paged",
            "kv_dist": "uniform",
            "q_lens": (1, 1),
            "kv_lens": (512, 512),
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
            "launch_policy": "analytical_launch",
            "split_policy": "serving",
        })

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hw})

    assert [call.kernel_spec.variant for call in plan.calls] == [
        "elementwise", "elementwise", "decode", "combine",
        "elementwise", "elementwise"]
    output_fill, input_copy, _, _, output_copy, lse_copy = (
        call.kernel_spec for call in plan.calls)
    assert output_fill.attributes["operation"] == "fill"
    assert output_fill.shape["element_count"] == 2 * 1 * 32 * 128
    assert output_fill.launch.grid == (8, 1, 1)
    assert output_fill.launch.block == (128, 1, 1)
    assert input_copy.launch.grid == (16, 1, 1)
    assert output_copy.launch.grid == (16, 1, 1)
    assert lse_copy.launch.grid == (1, 1, 1)
    assert input_copy.dtype == output_copy.dtype == "bf16"
    assert lse_copy.dtype == "float32"
    assert all(
        predict(call.kernel_spec, hw).t_total_ns > 0
        for call in plan.calls)


def test_ascend_decode_lowering_keeps_causal_prediction_semantics():
    hardware_id = "ascend_910b"
    hw = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    node = WorkloadNode(
        "attention", WorkloadKind.FLASH_ATTN, "main",
        {"batch": 4, "q_len": 1, "kv_len": 4096, "head_dim": 128,
         "num_query_heads": 32, "num_kv_heads": 8},
        "bf16", "decode", rank=0, attributes={
            "causal": True,
            "kv_layout": "paged_varlen",
            "kv_dist": "uniform",
            "q_lens": (1, 1, 1, 1),
            "kv_lens": (1024, 2048, 3072, 4096),
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
            "launch_policy": "injected_launch",
        })

    plan = lower_workload(
        WorkloadGraph((node,)), _deployment_for(hardware_id),
        {hardware_id: hw})

    assert len(plan.calls) == 1
    spec = plan.calls[0].kernel_spec
    assert spec.launch.algorithm == (
        "cann_flash_attention_score:103000000010622221")
    assert spec.launch.grid == (23, 1, 1)
    assert spec.attributes["causal"] is True
    predictions = predict_kernel_plan(plan, {hardware_id: hw})
    assert predictions["attention"].t_total_s > 0
    assert spec.attributes["causal"] is True


def _write_model_config(tmp_path, config):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config))
    return model_dir


def _dense_model_config(model_type, architecture):
    return {
        "model_type": model_type,
        "architectures": [architecture],
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 32,
        "vocab_size": 128256,
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": False,
        "attention_bias": False,
        "max_position_embeddings": 8192,
    }


def test_qwen_model_schema_defines_absent_mlp_bias(tmp_path):
    config = _dense_model_config("qwen3", "Qwen3ForCausalLM")
    config["head_dim"] = 128

    model = load_model_config(str(_write_model_config(tmp_path, config)))

    assert model.head_dim == 128
    assert model.mlp_bias is False
    assert model.tie_word_embeddings is False


def test_model_config_requires_explicit_tied_embedding_contract(tmp_path):
    config = _dense_model_config("qwen3", "Qwen3ForCausalLM")
    config["head_dim"] = 128
    del config["tie_word_embeddings"]

    with pytest.raises(ValueError, match="requires tie_word_embeddings"):
        load_model_config(str(_write_model_config(tmp_path, config)))


def test_llama_model_schema_derives_absent_head_dim_and_mlp_bias(tmp_path):
    config = _dense_model_config("llama", "LlamaForCausalLM")

    model = load_model_config(str(_write_model_config(tmp_path, config)))

    assert model.head_dim == 128
    assert model.mlp_bias is False

    with pytest.raises(ValueError, match="derived head_dim"):
        replace(model, head_dim=64)


def test_llama_model_schema_rejects_conflicting_explicit_head_dim(tmp_path):
    config = _dense_model_config("llama", "LlamaForCausalLM")
    config["head_dim"] = 64

    with pytest.raises(ValueError, match="disagrees with derived value"):
        load_model_config(str(_write_model_config(tmp_path, config)))


def test_model_dtype_preserves_bfloat16_semantics(tmp_path):
    config = _dense_model_config("qwen3", "Qwen3ForCausalLM")
    config["head_dim"] = 128

    model = load_model_config(str(_write_model_config(tmp_path, config)))

    assert model.compute_dtype == "bf16"
    assert model.dtype_bytes == 2


def test_model_dtype_does_not_rewrite_float16_as_bfloat16(tmp_path):
    config = _dense_model_config("qwen3", "Qwen3ForCausalLM")
    config["head_dim"] = 128
    config["torch_dtype"] = "float16"

    model = load_model_config(str(_write_model_config(tmp_path, config)))

    assert model.compute_dtype == "fp16"
    assert model.dtype_bytes == 2


def test_float16_propagates_to_kernel_spec_then_fails_unsupported_attention(
        tmp_path):
    config = _dense_model_config("qwen3", "Qwen3ForCausalLM")
    config["head_dim"] = 128
    config["torch_dtype"] = "float16"
    config["num_hidden_layers"] = 1
    model = load_model_config(str(_write_model_config(tmp_path, config)))
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
    )
    graph = build_workload(
        ScheduleStep(
            "fp16-step",
            "main",
            (ScheduledRequest("request", "prefill", 1, 0, 0),),
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "contiguous", "uniform"),
    )

    assert {node.dtype for node in graph.nodes} == {"fp16"}
    qkv = graph.node("fp16-step:layer-0:qkv")
    qkv_plan = lower_workload(
        WorkloadGraph((qkv,)),
        deployment,
        {"nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml")},
    )
    assert len(qkv_plan.calls) == 1
    assert qkv_plan.calls[0].kernel_spec.dtype == "fp16"
    assert qkv_plan.calls[0].kernel_spec.shape["dtype_bytes"] == 2

    with pytest.raises(ValueError, match="unsupported launch dtype 'fp16'"):
        lower_workload(
            graph,
            deployment,
            {"nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml")},
        )


def test_model_dtype_accepts_matching_modern_and_legacy_fields(tmp_path):
    config = _dense_model_config("qwen3", "Qwen3ForCausalLM")
    config["head_dim"] = 128
    config["dtype"] = "bfloat16"

    model = load_model_config(str(_write_model_config(tmp_path, config)))

    assert model.compute_dtype == "bf16"


def test_model_dtype_rejects_conflicting_fields(tmp_path):
    config = _dense_model_config("qwen3", "Qwen3ForCausalLM")
    config["head_dim"] = 128
    config["dtype"] = "float16"

    with pytest.raises(ValueError, match="dtype and torch_dtype disagree"):
        load_model_config(str(_write_model_config(tmp_path, config)))


def test_qwen_explicit_head_dim_can_expand_attention_projection_width(
        tmp_path):
    config = _dense_model_config("qwen3", "Qwen3ForCausalLM")
    config.update({
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "head_dim": 128,
        "num_hidden_layers": 1,
        "vocab_size": 151936,
    })
    model_path = _write_model_config(tmp_path, config)

    model = load_model_config(str(model_path))
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
    )
    step = ScheduleStep(
        "real-qwen3-shape",
        "main",
        (ScheduledRequest("request", "prefill", 729, 0),),
        attention_kv_layout="paged_varlen",
    )
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.not_applicable(),
            "paged_varlen",
            "uniform",
        ),
    )
    plan = lower_workload(
        graph,
        deployment,
        {"nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml")},
    )
    reference = qwen3_single_rank_forward_reference(
        model_path,
        forward_scope="prefill",
        step_id=step.step_id,
        query_lengths=(729,),
        kv_lengths=(729,),
    )
    report = compare_reference_to_emission(reference, graph, plan)

    assert model.hidden_size == 2560
    assert model.num_attention_heads * model.head_dim == 4096
    assert infer_gemm_dimensions(model, "qkv_proj", 729) == (
        729, 6144, 2560)
    assert infer_gemm_dimensions(model, "o_proj", 729) == (
        729, 2560, 4096)
    assert graph.node("real-qwen3-shape:layer-0:q-norm").shape == {
        "rows": 729 * 32,
        "hidden": 128,
    }
    assert graph.node("real-qwen3-shape:layer-0:k-norm").shape == {
        "rows": 729 * 8,
        "hidden": 128,
    }
    assert report["validation_status"] == (
        "validated_shapes_with_projected_attention_decomposition")
    assert report["missing"] == ()
    assert report["extra"] == ()
    assert report["mismatched"] == ()
    assert report["topology_mismatched"] == ()
    assert report["comparison_field_coverage"] == 1.0


def _grouped_qk_norm_node(
        node_id, rows_per_group, *, launch_policy="analytical_launch"):
    outer_groups = 12
    hidden = 128
    return WorkloadNode(
        node_id,
        WorkloadKind.NORM,
        "main",
        {"rows": outer_groups * rows_per_group, "hidden": hidden},
        "bf16",
        "rmsnorm",
        rank=0,
        attributes={
            "launch_policy": launch_policy,
            "input_materialization": "grouped_strided_to_contiguous",
            "input_outer_groups": outer_groups,
            "input_rows_per_group": rows_per_group,
            "input_group_stride_elements": 48 * hidden,
            "input_row_stride_elements": hidden,
        },
    )


def test_c500_grouped_qk_norm_lowers_copy_then_norm():
    hardware_id = "metax_c500"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    q_norm = _grouped_qk_norm_node(
        "q-norm", 32, launch_policy="injected_launch")
    k_norm = _grouped_qk_norm_node(
        "k-norm", 8, launch_policy="injected_launch")
    graph = WorkloadGraph(
        (q_norm, k_norm),
        (WorkloadEdge("q-norm", "k-norm", "data"),),
    )

    plan = lower_workload(
        graph,
        _deployment_for(hardware_id),
        {hardware_id: hardware},
    )

    assert tuple(call.call_id for call in plan.calls) == (
        "q-norm:0", "q-norm:1", "k-norm:0", "k-norm:1")
    q_copy, q_main, k_copy, k_main = plan.calls
    assert tuple(call.kernel_spec.variant for call in plan.calls) == (
        "elementwise", "rmsnorm", "elementwise", "rmsnorm")
    assert q_copy.kernel_spec.launch.algorithm == (
        "pytorch_2_4_metax_3_7_2_bf16_grouped_strided_copy")
    assert q_copy.kernel_spec.launch.grid == (96, 1, 1)
    assert k_copy.kernel_spec.launch.grid == (24, 1, 1)
    assert q_copy.kernel_spec.launch.block == (64, 1, 1)
    assert k_copy.kernel_spec.launch.block == (64, 1, 1)
    assert q_copy.kernel_spec.attributes == {
        "reads": 1,
        "writes": 1,
        "operation": "copy",
        "input_materialization": "grouped_strided_to_contiguous",
        "input_outer_groups": 12,
        "input_rows_per_group": 32,
        "input_group_stride_elements": 48 * 128,
        "input_row_stride_elements": 128,
    }
    assert q_main.dependencies == (q_copy.call_id,)
    assert k_copy.dependencies == (q_main.call_id,)
    assert k_main.dependencies == (k_copy.call_id,)


def test_910b_qk_norm_rope_lowers_to_separate_norm_and_rope_programs():
    """Q/K Norm and RoPE stay three programs, not one fused variant.

    The canonical launch observation carries no split_qkv_rmsnorm_rope
    signature on any captured NPU serving case, so the fusion matcher is
    inert and each source node keeps its own physical program.  The lowered
    RoPE spec must still equal the dataset-style spec for the same shape and
    predict identically, which is the invariant this test guards.
    """

    hardware_id = "ascend_910b"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    q_norm = _grouped_qk_norm_node("q-norm", 32)
    k_norm = _grouped_qk_norm_node("k-norm", 8)
    rotary = WorkloadNode(
        "rotary",
        WorkloadKind.ROTARY,
        "main",
        {
            "elements": 12 * (32 + 8) * 128,
            "seq_len": 12,
            "num_query_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        "bf16",
        "rotary",
        rank=0,
        attributes={
            "launch_policy": "analytical_launch",
            # §2.1 class1b: the V materialization count the versioned
            # emission observation resolves; zero in this fixture's phase
            "qkv_v_materialization_count": 0,
        },
    )
    graph = WorkloadGraph(
        (q_norm, k_norm, rotary),
        (
            WorkloadEdge("q-norm", "k-norm", "data"),
            WorkloadEdge("k-norm", "rotary", "data"),
        ),
    )

    plan = lower_workload(
        graph,
        _deployment_for(hardware_id),
        {hardware_id: hardware},
    )

    # Each norm keeps its own input materialization copy and its own source
    # identity; RoPE is a separate program chained after the last norm.
    assert [call.call_id for call in plan.calls] == [
        "q-norm:0", "q-norm:1", "k-norm:0", "k-norm:1", "rotary"]
    assert [call.source_node_ids for call in plan.calls] == [
        ("q-norm",), ("q-norm",), ("k-norm",), ("k-norm",), ("rotary",)]
    assert [call.kernel_spec.variant for call in plan.calls] == [
        "elementwise", "rmsnorm", "elementwise", "rmsnorm", "rotary"]
    assert [
        call.kernel_spec.shape["element_count"]
        for call in plan.calls if call.kernel_spec.variant == "elementwise"
    ] == [12 * 32 * 128, 12 * 8 * 128]
    assert [call.dependencies for call in plan.calls[1:]] == [
        ("q-norm:0",), ("q-norm:1",), ("k-norm:0",), ("k-norm:1",)]

    call = plan.calls[-1]
    assert call.kernel_spec.launch.algorithm == "aclnn_rotary"
    direct_shape = {
        "seq_len": 12,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "element_count": 12 * (32 + 8) * 128,
    }
    direct_launch = select_launch_config(
        hardware_id,
        KernelKind.VECTOR,
        direct_shape,
        "bf16",
        "rotary",
        {"launch_policy": "analytical_launch"},
        hardware=hardware,
    )
    dataset_style_spec = KernelSpec(
        KernelKind.VECTOR,
        hardware_id,
        direct_shape,
        dtype="bf16",
        variant="rotary",
        launch=direct_launch,
        attributes={},
    )
    assert call.kernel_spec == dataset_style_spec
    predictor = create_predictor(KernelKind.VECTOR)
    assert predictor.resource_input(call.kernel_spec, hardware) == (
        predictor.resource_input(dataset_style_spec, hardware))
    assert predict(call.kernel_spec, hardware) == predict(
        dataset_style_spec, hardware)


def test_norm_without_materialization_remains_one_call():
    hardware_id = "metax_c500"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    node = WorkloadNode(
        "input-norm",
        WorkloadKind.NORM,
        "main",
        {"rows": 12, "hidden": 2560},
        "bf16",
        "rmsnorm",
        rank=0,
        attributes={"launch_policy": "analytical_launch"},
    )

    plan = lower_workload(
        WorkloadGraph((node,)),
        _deployment_for(hardware_id),
        {hardware_id: hardware},
    )

    assert tuple(call.call_id for call in plan.calls) == ("input-norm",)
    assert plan.calls[0].kernel_spec.variant == "rmsnorm"


def test_norm_materialization_contract_fails_closed_when_incomplete():
    hardware_id = "metax_c500"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    node = WorkloadNode(
        "q-norm",
        WorkloadKind.NORM,
        "main",
        {"rows": 384, "hidden": 128},
        "bf16",
        "rmsnorm",
        rank=0,
        attributes={
            "launch_policy": "analytical_launch",
            "input_materialization": "grouped_strided_to_contiguous",
        },
    )

    with pytest.raises(ValueError, match="contract is incomplete"):
        lower_workload(
            WorkloadGraph((node,)),
            _deployment_for(hardware_id),
            {hardware_id: hardware},
        )


def test_910b_grouped_qk_copy_exposes_missing_tiler_payload():
    hardware_id = "ascend_910b"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    node = _grouped_qk_norm_node("q-norm", 32)
    plan = lower_workload(
        WorkloadGraph((node,)),
        _deployment_for(hardware_id),
        {hardware_id: hardware},
    )
    copy_call, norm_call = plan.calls
    prediction = predict(copy_call.kernel_spec, hardware)

    assert copy_call.kernel_spec.variant == "elementwise"
    assert norm_call.kernel_spec.variant == "rmsnorm"
    assert prediction.diagnostics.support_level == "aggregate_only"
    assert "cann_contiguous_slice_tiling_payload_unavailable" in (
        prediction.diagnostics.limitations)


def _project_qwen3_moe_attention_scope(
        graph, plan, *, step_id, layer_index):
    layer_prefix = f"{step_id}:layer-{layer_index}:"
    # The §2.2 routing-mask helper launches (a800 expert-map-index, 910B
    # router-mask index/mul) belong to the MoE/FFN scope and chain after the
    # MoE block's output all-reduce, so they stay outside the attention
    # projection together with the router and fused-MoE nodes.
    ffn_scope_fragments = (
        ":router:",
        ":moe:",
        ":expert-map-index",
        ":router-mask-index",
        ":router-mask-mul",
    )
    projected_nodes = tuple(
        node for node in graph.nodes
        if (node.node_id.startswith(layer_prefix)
            and not any(
                fragment in node.node_id
                for fragment in ffn_scope_fragments))
    )
    projected_node_ids = {node.node_id for node in projected_nodes}
    projected_graph = WorkloadGraph(
        projected_nodes,
        tuple(
            edge for edge in graph.edges
            if (edge.source in projected_node_ids
                and edge.destination in projected_node_ids)
        ),
    )
    mixed_scope_calls = tuple(
        call.call_id for call in plan.calls
        if (set(call.source_node_ids).intersection(projected_node_ids)
            and not set(call.source_node_ids).issubset(projected_node_ids))
    )
    if mixed_scope_calls:
        raise ValueError(
            "Qwen3MoE attention projection found mixed-scope KernelCalls: "
            f"{mixed_scope_calls!r}"
        )
    projected_calls = tuple(
        call for call in plan.calls
        if (call.source_node_ids
            and set(call.source_node_ids).issubset(projected_node_ids))
    )
    projected_call_ids = {call.call_id for call in projected_calls}
    embedding_boundary = f"{step_id}:token-embedding:tp-all-reduce"
    # The A800 prefill step-head embedding is a versioned measured zero, so
    # its boundary collective is absent and input-norm carries no external
    # dependency; decode/mixed keep the embedding boundary.
    embedding_boundary_present = any(
        node.node_id == embedding_boundary for node in graph.nodes)
    input_norm_node_ids = {
        f"{step_id}:layer-{layer_index}:input-norm:tp-rank-{rank}"
        for rank in range(2)
    }
    for call in projected_calls:
        external_dependencies = tuple(
            dependency for dependency in call.dependencies
            if dependency not in projected_call_ids
        )
        expected_external_dependencies = (
            (embedding_boundary,)
            if embedding_boundary_present
            and call.source_node_ids[0] in input_norm_node_ids
            else ()
        )
        if external_dependencies != expected_external_dependencies:
            raise ValueError(
                "Qwen3MoE attention projection found unexpected external "
                f"dependencies for {call.call_id!r}: expected "
                f"{expected_external_dependencies!r}, got "
                f"{external_dependencies!r}"
            )
    projected_plan = KernelPlan(tuple(
        replace(
            call,
            dependencies=tuple(
                dependency for dependency in call.dependencies
                if dependency in projected_call_ids
            ),
        )
        for call in projected_calls
    ))
    return projected_graph, projected_plan


def test_qwen3_moe_explicit_attention_width_reaches_tp_workload_and_kv(
        tmp_path):
    config = _dense_model_config("qwen3_moe", "Qwen3MoeForCausalLM")
    config.update({
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "head_dim": 128,
        "num_hidden_layers": 2,
        "vocab_size": 151936,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 1024,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
    })
    model_path = _write_model_config(tmp_path, config)
    model = load_model_config(str(model_path))
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(
                    tensor_parallel_size=2,
                    enable_expert_parallel=True,
                ),
                ranks=(0, 1),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    graph = build_workload(
        ScheduleStep(
            "explicit-moe-tp2",
            "main",
            (ScheduledRequest("request", "prefill", 729, 0),),
            attention_kv_layout="paged_varlen",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_explicit_moe_head_dim_v1", 0),
            "paged_varlen",
            "uniform",
        ),
    )
    hardware = {"nvidia_a800": HWSpec(
        "configs/hardware/nvidia_a800.yaml")}
    plan = lower_workload(graph, deployment, hardware)
    predictions = predict_kernel_plan(plan, hardware, method="midlayer")
    source = pinned_qwen3_moe_ep_source_evidence()
    assert source["attention_head_dim_source"] == (
        "config.head_dim_or_hidden_size_div_total_heads")
    assert source["attention_o_projection_input"] == (
        "total_num_heads_times_head_dim")
    reference = qwen3_moe_tp2_attention_reference(
        model_path,
        step_id="explicit-moe-tp2",
        layer_index=0,
        query_lengths=(729,),
        kv_lengths=(729,),
        physical_ranks=(0, 1),
        resource_id="main-nvlink-fabric",
    )
    projected_graph, projected_plan = _project_qwen3_moe_attention_scope(
        graph, plan, step_id="explicit-moe-tp2", layer_index=0)
    report = compare_tp2_reference_to_emission(
        reference, projected_graph, projected_plan)

    assert model.hidden_size == 2560
    assert model.num_attention_heads * model.head_dim == 4096
    assert model.kv_bytes_per_token_per_parallel_rank(2, 1) == 4096
    assert graph.node(
        "explicit-moe-tp2:layer-0:qkv:tp-rank-0").shape["n"] == 3072
    assert graph.node(
        "explicit-moe-tp2:layer-0:o-proj:tp-rank-0").shape["k"] == 2048
    assert any(node.kind is WorkloadKind.FUSED_MOE for node in graph.nodes)
    assert set(predictions) == {call.call_id for call in plan.calls}
    assert report["validation_status"] == (
        "validated_qwen3_moe_tp2_attention_static_source_semantics_"
        "projected_latency")
    assert report["missing"] == ()
    assert report["extra"] == ()
    assert report["mismatched"] == ()
    assert report["topology_mismatched"] == ()
    assert report["node_coverage"] == 1.0

    o_projection_node_id = (
        "explicit-moe-tp2:layer-0:o-proj:tp-rank-0")
    # The A800 prefill token-embedding all-reduce is suppressed by the
    # versioned measured-zero observation, so the forged external
    # dependency is a helper call sourced on the router node (outside the
    # attention projection, emitted before the o-proj call).
    router_node_id = next(
        node.node_id for node in graph.nodes
        if ":router:" in node.node_id
    )
    o_projection_call = next(
        call for call in plan.calls
        if call.source_node_ids == (o_projection_node_id,)
    )
    forged_helper_call_id = f"{o_projection_node_id}:forged-helper"
    unexpected_dependency_plan = KernelPlan((
        replace(
            o_projection_call,
            call_id=forged_helper_call_id,
            source_node_ids=(router_node_id,),
            dependencies=(),
        ),
        *(
            replace(
                call,
                dependencies=call.dependencies + (
                    forged_helper_call_id,),
            ) if call.source_node_ids == (o_projection_node_id,) else call
            for call in plan.calls
        ),
    ))
    with pytest.raises(
            ValueError, match="unexpected external dependencies"):
        _project_qwen3_moe_attention_scope(
            graph,
            unexpected_dependency_plan,
            step_id="explicit-moe-tp2",
            layer_index=0,
        )

    moe_node_id = next(
        node.node_id for node in graph.nodes
        if node.kind is WorkloadKind.FUSED_MOE
    )
    qkv_full_plan_call = next(
        call for call in plan.calls
        if call.source_node_ids == (
            "explicit-moe-tp2:layer-0:qkv:tp-rank-0",)
    )
    mixed_scope_plan = KernelPlan((*plan.calls, replace(
        qkv_full_plan_call,
        call_id="explicit-moe-tp2:mixed-scope:unexpected-gemm",
        source_node_ids=(o_projection_node_id, moe_node_id),
        dependencies=(),
    )))
    with pytest.raises(ValueError, match="mixed-scope KernelCalls"):
        _project_qwen3_moe_attention_scope(
            graph,
            mixed_scope_plan,
            step_id="explicit-moe-tp2",
            layer_index=0,
        )

    post_norm_node_id = (
        "explicit-moe-tp2:layer-0:post-attn-norm:tp-rank-0")
    qkv_call = next(
        call for call in projected_plan.calls
        if call.source_node_ids == (
            "explicit-moe-tp2:layer-0:qkv:tp-rank-0",)
    )
    extra_call_plan = KernelPlan((*projected_plan.calls, replace(
        qkv_call,
        call_id=f"{post_norm_node_id}:unexpected-gemm",
        source_node_ids=(post_norm_node_id,),
        dependencies=(post_norm_node_id,),
    )))
    extra_call_report = compare_tp2_reference_to_emission(
        reference, projected_graph, extra_call_plan)
    extra_call_differences = tuple(
        difference
        for mismatch in extra_call_report["mismatched"]
        if mismatch["node_id"] == post_norm_node_id
        for difference in mismatch["differences"]
    )
    assert {
        "field": "kernel_call_count",
        "expected": 1,
        "actual": 2,
    } in extra_call_differences

    wrong_rank_plan = KernelPlan(tuple(
        replace(call, rank=99)
        if call.source_node_ids == (o_projection_node_id,) else call
        for call in projected_plan.calls
    ))
    wrong_rank_report = compare_tp2_reference_to_emission(
        reference, projected_graph, wrong_rank_plan)
    wrong_rank_differences = tuple(
        difference
        for mismatch in wrong_rank_report["mismatched"]
        if mismatch["node_id"] == o_projection_node_id
        for difference in mismatch["differences"]
    )
    assert {
        "field": "kernel_rank",
        "expected": 0,
        "actual": 99,
    } in wrong_rank_differences

    renamed_o_projection_call_id = f"{o_projection_node_id}:renamed"
    renamed_call_plan = KernelPlan(tuple(
        replace(
            call,
            call_id=(
                renamed_o_projection_call_id
                if call.source_node_ids == (o_projection_node_id,)
                else call.call_id
            ),
            dependencies=tuple(
                renamed_o_projection_call_id
                if dependency == o_projection_node_id else dependency
                for dependency in call.dependencies
            ),
        )
        for call in projected_plan.calls
    ))
    renamed_call_report = compare_tp2_reference_to_emission(
        reference, projected_graph, renamed_call_plan)
    renamed_call_differences = tuple(
        difference
        for mismatch in renamed_call_report["mismatched"]
        if mismatch["node_id"] == o_projection_node_id
        for difference in mismatch["differences"]
    )
    assert {
        "field": "kernel_call_id",
        "expected": o_projection_node_id,
        "actual": renamed_o_projection_call_id,
    } in renamed_call_differences


def test_qwen_explicit_attention_width_survives_tp2_and_pp2(
        tmp_path):
    config = _dense_model_config("qwen3", "Qwen3ForCausalLM")
    config.update({
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "head_dim": 128,
        "num_hidden_layers": 2,
        "vocab_size": 151936,
    })
    model_path = _write_model_config(tmp_path, config)
    model = load_model_config(str(model_path))
    scheduled = ScheduledRequest("request", "prefill", 729, 0)
    runtime = RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        "paged_varlen",
        "uniform",
    )
    hardware = {"nvidia_a800": HWSpec(
        "configs/hardware/nvidia_a800.yaml")}
    hidden_bytes = 729 * 2560 * 2

    tp2_deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(tensor_parallel_size=2),
                ranks=(0, 1),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    tp2_graph = build_workload(
        ScheduleStep(
            "explicit-tp2",
            "main",
            (scheduled,),
            attention_kv_layout="paged_varlen",
        ),
        model,
        tp2_deployment,
        launch_policy="analytical_launch",
        runtime_inputs=runtime,
    )
    tp2_plan = lower_workload(tp2_graph, tp2_deployment, hardware)
    tp2_reference = qwen3_tp2_forward_reference(
        model_path,
        forward_scope="prefill",
        step_id="explicit-tp2",
        query_lengths=(729,),
        kv_lengths=(729,),
        physical_ranks=(0, 1),
        resource_id="main-nvlink-fabric",
    )
    tp2_report = compare_tp2_reference_to_emission(
        tp2_reference, tp2_graph, tp2_plan)

    assert tp2_graph.node(
        "explicit-tp2:layer-0:qkv:tp-rank-0").shape["n"] == 3072
    assert tp2_graph.node(
        "explicit-tp2:layer-0:o-proj:tp-rank-0").shape["k"] == 2048
    assert tp2_graph.node(
        "explicit-tp2:layer-0:o-proj:tp-all-reduce"
    ).attributes["bytes"] == hidden_bytes
    assert tp2_report["mismatched"] == ()
    assert tp2_report["topology_mismatched"] == ()
    assert tp2_report["node_coverage"] == 1.0

    tp2_pp2_deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(
                    tensor_parallel_size=2,
                    pipeline_parallel_size=2,
                ),
                ranks=(0, 1, 2, 3),
                collectives=_tp2_collectives(),
                pipeline_p2p=_pp2_pipeline_p2p(),
            ),
        )),
        num_layers=model.num_hidden_layers,
    )
    tp2_pp2_graph = build_workload(
        ScheduleStep(
            "explicit-tp2-pp2",
            "main",
            (scheduled,),
            attention_kv_layout="paged_varlen",
        ),
        model,
        tp2_pp2_deployment,
        launch_policy="analytical_launch",
        runtime_inputs=runtime,
    )
    tp2_pp2_plan = lower_workload(
        tp2_pp2_graph, tp2_pp2_deployment, hardware)
    tp2_pp2_reference = qwen3_tp2_pp2_forward_reference(
        model_path,
        forward_scope="prefill",
        step_id="explicit-tp2-pp2",
        query_lengths=(729,),
        kv_lengths=(729,),
        physical_rank_groups=((0, 1), (2, 3)),
        tensor_resource_id="main-nvlink-fabric",
        pipeline_resource_id="main-pipeline-fabric",
    )
    tp2_pp2_report = compare_tp2_reference_to_emission(
        tp2_pp2_reference, tp2_pp2_graph, tp2_pp2_plan)

    assert tp2_pp2_graph.node(
        "explicit-tp2-pp2:layer-1:o-proj:tp-rank-0"
    ).shape["k"] == 2048
    assert model.kv_bytes_per_token_per_parallel_rank(2, 2) == 2048
    pipeline_calls = tuple(
        node for node in tp2_pp2_graph.nodes
        if node.kind is WorkloadKind.P2P)
    assert len(pipeline_calls) == 4
    assert all(
        node.attributes["bytes"] == hidden_bytes // 2
        for node in pipeline_calls
    )
    assert tp2_pp2_report["mismatched"] == ()
    assert tp2_pp2_report["topology_mismatched"] == ()
    assert tp2_pp2_report["node_coverage"] == 1.0


def test_standard_attention_rejects_invalid_gqa_ratio(tmp_path):
    config = _dense_model_config("qwen3", "Qwen3ForCausalLM")
    config["head_dim"] = 128
    config["num_key_value_heads"] = 7

    with pytest.raises(ValueError, match="must be divisible"):
        load_model_config(str(_write_model_config(tmp_path, config)))


def test_direct_model_config_cannot_bypass_gqa_ratio_invariant():
    model = load_model_config("tests/serving/fixtures/qwen3_scheduler")

    with pytest.raises(ValueError, match="num_attention_heads must be"):
        replace(model, num_key_value_heads=model.num_attention_heads + 1)


def test_unknown_model_schema_fails_loudly(tmp_path):
    config = _dense_model_config("future_model", "FutureForCausalLM")

    with pytest.raises(ValueError, match="unsupported model_type"):
        load_model_config(str(_write_model_config(tmp_path, config)))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("moe_router_dtype", "fp16", "router dtype"),
        ("moe_score_function", "softmax", "score function"),
        ("moe_topk_method", "greedy", "top-k method"),
        ("rope_layout", "half_split", "main RoPE layout"),
        ("rope_theta", 0, "RoPE theta"),
    ),
)
def test_target_model_semantics_reject_source_contract_drift(
        field, value, message):
    model = load_model_config(
        "tests/serving/fixtures/glm_5_2_official")
    assert model.target_semantics is not None

    with pytest.raises(ValueError, match=message):
        replace(model.target_semantics, **{field: value})


def test_glm_target_semantics_reject_dsa_skip_rule_drift():
    model = load_model_config(
        "tests/serving/fixtures/glm_5_2_official")
    assert model.target_semantics is not None

    with pytest.raises(ValueError, match="index top-k pattern"):
        replace(
            model.target_semantics,
            index_topk_pattern=("F", "unknown"),
        )


def test_kimi_k2_6_mla_projection_shapes_match_vllm_0_27_1_source():
    model = load_model_config(
        "tests/serving/fixtures/kimi_k2_6_official")

    assert infer_gemm_dimensions(
        model, "fused_qkv_a_proj", 7) == (7, 2_112, 7_168)
    assert infer_gemm_dimensions(
        model, "q_b_proj", 7) == (7, 12_288, 1_536)
    assert infer_gemm_dimensions(
        model, "kv_b_proj", 7) == (7, 16_384, 512)
    assert infer_gemm_dimensions(
        model, "o_proj", 7) == (7, 7_168, 8_192)

    # fused_qkv_a_proj is explicitly replicated (disable_tp=True), while the
    # B projections and O projection use vLLM column/row parallel semantics.
    assert infer_gemm_dimensions(
        model, "fused_qkv_a_proj", 1, tp_size=2) == (1, 2_112, 7_168)
    assert infer_gemm_dimensions(
        model, "q_b_proj", 1, tp_size=2) == (1, 6_144, 1_536)
    assert infer_gemm_dimensions(
        model, "kv_b_proj", 1, tp_size=2) == (1, 8_192, 512)
    assert infer_gemm_dimensions(
        model, "o_proj", 1, tp_size=2) == (1, 7_168, 4_096)


def test_glm_dsa_exposes_vllm_fused_projection_shapes():
    model = load_model_config(
        "tests/serving/fixtures/glm_5_2_official")

    assert infer_gemm_dimensions(
        model, "fused_qkv_a_proj", 1) == (1, 2_624, 6_144)
    assert infer_gemm_dimensions(
        model, "indexer_wk_weights_proj", 1) == (1, 160, 6_144)
    assert infer_gemm_dimensions(
        model, "o_proj", 1) == (1, 6_144, 16_384)


def test_pd_transfer_emits_and_lowers_matching_tp2_head_ownership_edges():
    model = load_model_config(
        "tests/serving/fixtures/qwen3_tp2_scheduler")
    deployment = build_pd_disagg_plan(
        DeploymentConfig(
            "pd_disagg",
            (
                WorkerPoolConfig(
                    "prefill", "prefill", "nvidia_a800",
                    ParallelConfig(tensor_parallel_size=2), (0, 1)),
                WorkerPoolConfig(
                    "decode", "decode", "nvidia_a800",
                    ParallelConfig(tensor_parallel_size=2), (2, 3)),
            ),
            (RouteConfig(
                "prefill", "decode", "kv", "send_recv", "nvlink",
                resource_id="pd-kv-link",
                backend="analytical_roofline_v1",
            ),),
        ),
        num_layers=model.num_hidden_layers,
    )

    graph = build_pd_kv_transfer_workload(
        "request-0", 4, model, deployment, "prefill", "decode")

    assert len(graph.nodes) == 2
    expected_bytes = (
        4 * model.kv_bytes_per_token_per_parallel_rank(2, 1))
    assert tuple(
        (
            node.node_id,
            node.rank,
            node.attributes["source_logical_rank"],
            node.attributes["destination_logical_rank"],
            (
                node.attributes["kv_head_start"],
                node.attributes["kv_head_end"],
            ),
            tuple(node.attributes["participants"]),
            node.attributes["bytes"],
        )
        for node in graph.nodes
    ) == (
        (
            "request-0:pd-kv-transfer:prefill-to-decode:kv-edge-0",
            0, 0, 0, (0, 1), (0, 2), expected_bytes,
        ),
        (
            "request-0:pd-kv-transfer:prefill-to-decode:kv-edge-1",
            1, 1, 1, (1, 2), (1, 3), expected_bytes,
        ),
    )
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    plan = lower_workload(
        graph, deployment, {"nvidia_a800": hardware})
    assert len(plan.calls) == 2
    assert {
        call.kernel_spec.transfer.participants for call in plan.calls
    } == {(0, 2), (1, 3)}


@pytest.mark.parametrize(("prefill_tp", "decode_tp", "participants"), [
    (1, 2, ((0, 1), (0, 2))),
    (2, 1, ((0, 2), (1, 2))),
])
def test_pd_transfer_maps_mixed_tp_by_kv_head_overlap(
        prefill_tp, decode_tp, participants):
    model = load_model_config(
        "tests/serving/fixtures/qwen3_tp2_scheduler")
    deployment = build_pd_disagg_plan(
        DeploymentConfig(
            "pd_disagg",
            (
                WorkerPoolConfig(
                    "prefill", "prefill", "nvidia_a800",
                    ParallelConfig(tensor_parallel_size=prefill_tp),
                    tuple(range(prefill_tp))),
                WorkerPoolConfig(
                    "decode", "decode", "nvidia_a800",
                    ParallelConfig(tensor_parallel_size=decode_tp),
                    tuple(range(prefill_tp, prefill_tp + decode_tp))),
            ),
            (RouteConfig(
                "prefill", "decode", "kv", "send_recv", "nvlink",
                resource_id="pd-kv-link",
                backend="analytical_roofline_v1",
            ),),
        ),
        num_layers=model.num_hidden_layers,
    )

    graph = build_pd_kv_transfer_workload(
        "request-0", 4, model, deployment, "prefill", "decode")

    assert tuple(
        tuple(node.attributes["participants"]) for node in graph.nodes
    ) == participants
    assert tuple(
        (node.attributes["kv_head_start"], node.attributes["kv_head_end"])
        for node in graph.nodes
    ) == ((0, 1), (1, 2))
    expected_edge_bytes = (
        4 * 2 * model.num_hidden_layers * model.head_dim
        * model.dtype_bytes)
    assert tuple(node.attributes["bytes"] for node in graph.nodes) == (
        expected_edge_bytes, expected_edge_bytes)

    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    plan = lower_workload(
        graph, deployment, {"nvidia_a800": hardware})
    assert tuple(
        call.kernel_spec.transfer.participants for call in plan.calls
    ) == participants


def test_pd_transfer_rejects_replicated_kv_head_source_selection():
    model = replace(
        load_model_config("tests/serving/fixtures/qwen3_tp2_scheduler"),
        num_key_value_heads=1,
    )
    deployment = build_pd_disagg_plan(
        DeploymentConfig(
            "pd_disagg",
            (
                WorkerPoolConfig(
                    "prefill", "prefill", "nvidia_a800",
                    ParallelConfig(tensor_parallel_size=2), (0, 1)),
                WorkerPoolConfig(
                    "decode", "decode", "nvidia_a800", ranks=(2,)),
            ),
            (RouteConfig(
                "prefill", "decode", "kv", "send_recv", "nvlink",
                resource_id="pd-kv-link",
                backend="analytical_roofline_v1"),),
        ),
        num_layers=model.num_hidden_layers,
    )

    with pytest.raises(
            NotImplementedError, match="canonical source replica"):
        build_pd_kv_transfer_workload(
            "request-0", 4, model, deployment, "prefill", "decode")


def test_target_model_pd_transfer_uses_compressed_mla_cache_contract(
):
    model = load_model_config(
        "tests/serving/fixtures/kimi_k2_6_official")
    deployment = build_deployment_plan(
        DeploymentConfig(
            "pd_disagg",
            (
                WorkerPoolConfig(
                    "prefill", "prefill", "nvidia_a800", ranks=(0,)),
                WorkerPoolConfig(
                    "decode", "decode", "nvidia_a800", ranks=(1,)),
            ),
            (RouteConfig(
                "prefill",
                "decode",
                "kv",
                "send_recv",
                "nvlink",
                "point_to_point",
                resource_id="target-model-kv-link",
                backend="analytical_roofline_v1",
            ),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )

    graph = build_pd_kv_transfer_workload(
        "target-model",
        3,
        model,
        deployment,
        "prefill",
        "decode",
    )
    node, = graph.nodes
    assert node.kind is WorkloadKind.PD_KV_TRANSFER
    assert node.attributes["bytes"] == 3 * 70_272

    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    plan = lower_workload(
        graph, deployment, {"nvidia_a800": hardware})
    call, = plan.calls
    assert call.kernel_spec.kind is KernelKind.COMMUNICATION
    assert call.kernel_spec.transfer.bytes == 3 * 70_272
    predictions = predict_kernel_plan(plan, {"nvidia_a800": hardware})
    assert predictions[call.call_id].t_total_s > 0

    tampered_values = (
        ("algorithm", "ring"),
        ("bytes_semantics", "per_rank_input"),
        ("backend", "protobridge"),
        ("backend_config_sha256", "a" * 64),
        ("direction", "collective"),
        ("link", "pcie"),
        ("topology", "ring"),
        ("participants", (1, 0)),
        ("operation", "all_reduce"),
        ("source_rank", 1),
        ("destination_rank", 0),
        ("ring_order", (0, 1)),
        ("reduction", "sum"),
            ("resource_id", "other-link"),
            ("rank_mapping", "repartition"),
            ("source_logical_rank", 1),
            ("destination_logical_rank", 1),
            ("source_tensor_parallel_size", 2),
            ("destination_tensor_parallel_size", 2),
            ("kv_head_start", 1),
            ("kv_head_end", 1),
            ("num_key_value_heads", 32),
    )
    for field, value in tampered_values:
        attributes = dict(node.attributes)
        attributes[field] = value
        with pytest.raises(
                ValueError, match="disagrees with deployment config"):
            lower_workload(
                WorkloadGraph((replace(node, attributes=attributes),)),
                deployment,
                {"nvidia_a800": hardware},
            )

    with pytest.raises(ValueError, match="canonical contract"):
        lower_workload(
            WorkloadGraph((replace(
                node,
                attributes={**node.attributes, "unexpected": True},
            ),)),
            deployment,
            {"nvidia_a800": hardware},
        )
    with pytest.raises(ValueError, match="source must have a prefill role"):
        lower_workload(
            WorkloadGraph((replace(node, rank=1, pool_name="decode"),)),
            deployment,
            {"nvidia_a800": hardware},
        )
    with pytest.raises(ValueError, match="unknown destination pool"):
        lower_workload(
            WorkloadGraph((replace(
                node,
                attributes={
                    **node.attributes,
                    "destination_pool": "missing",
                },
            ),)),
            deployment,
            {"nvidia_a800": hardware},
        )
    with pytest.raises(TypeError, match="bytes must be a positive integer"):
        lower_workload(
            WorkloadGraph((replace(
                node,
                attributes={**node.attributes, "bytes": 0},
            ),)),
            deployment,
            {"nvidia_a800": hardware},
        )
    with pytest.raises(ValueError, match="shape must be empty"):
        lower_workload(
            WorkloadGraph((replace(node, shape={"tokens": 3}),)),
            deployment,
            {"nvidia_a800": hardware},
        )
    with pytest.raises(
            ValueError, match="dtype and variant must both be null"):
        lower_workload(
            WorkloadGraph((replace(
                node,
                dtype="bf16",
                variant="silently-discarded",
            ),)),
            deployment,
            {"nvidia_a800": hardware},
        )

    ffn_deployment = build_deployment_plan(
        DeploymentConfig(
            "pd_af_disagg",
            (
                WorkerPoolConfig(
                    "prefill-ffn",
                    "prefill_ffn",
                    "nvidia_a800",
                    ranks=(0,),
                ),
                WorkerPoolConfig(
                    "decode-ffn",
                    "decode_ffn",
                    "nvidia_a800",
                    ranks=(1,),
                ),
            ),
            (RouteConfig(
                "prefill-ffn",
                "decode-ffn",
                "kv",
                "send_recv",
                "nvlink",
                resource_id="ffn-kv-link",
                backend="analytical_roofline_v1",
            ),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    ffn_attributes = {
        **node.attributes,
        "participants": (0, 1),
        "source_rank": 0,
        "destination_rank": 1,
        "resource_id": "ffn-kv-link",
        "destination_pool": "decode-ffn",
    }
    with pytest.raises(ValueError, match="not FFN pools"):
        lower_workload(
            WorkloadGraph((replace(
                node,
                pool_name="prefill-ffn",
                rank=0,
                attributes=ffn_attributes,
            ),)),
            ffn_deployment,
            {"nvidia_a800": hardware},
        )


def test_glm_5_2_pd_transfer_fails_without_combined_cache_contract():
    model = load_model_config(
        "tests/serving/fixtures/glm_5_2_official")
    deployment = build_deployment_plan(
        DeploymentConfig(
            "pd_disagg",
            (
                WorkerPoolConfig(
                    "prefill", "prefill", "nvidia_a800", ranks=(0,)),
                WorkerPoolConfig(
                    "decode", "decode", "nvidia_a800", ranks=(1,)),
            ),
            (RouteConfig(
                "prefill",
                "decode",
                "kv",
                "send_recv",
                "nvlink",
                "point_to_point",
                resource_id="target-model-kv-link",
                backend="analytical_roofline_v1",
            ),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )

    with pytest.raises(
            NotImplementedError,
            match="compressed MLA and FP8 index-cache transfer contract"):
        build_pd_kv_transfer_workload(
            "target-model",
            3,
            model,
            deployment,
            "prefill",
            "decode",
        )


def test_glm_5_2_pd_transfer_tp8_ep1_replicated_cache_contract():
    model = load_model_config(
        "tests/serving/fixtures/glm_5_2_official")
    deployment = build_deployment_plan(
        DeploymentConfig(
            "pd_disagg",
            (
                WorkerPoolConfig(
                    "prefill", "prefill", "nvidia_h200",
                    ParallelConfig(tensor_parallel_size=8),
                    tuple(range(8))),
                WorkerPoolConfig(
                    "decode", "decode", "nvidia_h200",
                    ParallelConfig(tensor_parallel_size=8),
                    tuple(range(8, 16))),
            ),
            (RouteConfig(
                "prefill",
                "decode",
                "kv",
                "send_recv",
                "nvlink",
                "point_to_point",
                resource_id="target-model-kv-link",
                backend="analytical_roofline_v1",
            ),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )

    graph = build_pd_kv_transfer_workload(
        "target-model", 3, model, deployment, "prefill", "decode")

    # 78 layers * (512 + 64) * 2 B compressed MLA + 21 full-indexer layers
    # * (128 B FP8 + 4 B scale); both caches are single-head TP-replicated,
    # so the per-rank payload is identical to the TP=EP=8 contract and
    # independent of expert parallelism.
    assert model.kv_bytes_per_token_per_parallel_rank(8, 1) == 92_628
    expected_bytes = 3 * 92_628
    assert tuple(
        (
            node.attributes["source_logical_rank"],
            node.attributes["destination_logical_rank"],
            node.attributes["kv_head_start"],
            node.attributes["kv_head_end"],
            tuple(node.attributes["participants"]),
            node.attributes["bytes"],
        )
        for node in graph.nodes
    ) == tuple(
        (rank, rank, 8 * rank, 8 * rank + 8, (rank, rank + 8),
         expected_bytes)
        for rank in range(8)
    )


def _dsv3_h200_pd_ep1_deployment(model, *, enable_expert_parallel=False):
    return build_deployment_plan(
        DeploymentConfig(
            "pd_disagg",
            (
                WorkerPoolConfig(
                    "prefill", "prefill", "nvidia_h200",
                    ParallelConfig(
                        tensor_parallel_size=8,
                        enable_expert_parallel=enable_expert_parallel),
                    tuple(range(8))),
                WorkerPoolConfig(
                    "decode", "decode", "nvidia_h200",
                    ParallelConfig(
                        tensor_parallel_size=8,
                        enable_expert_parallel=enable_expert_parallel),
                    tuple(range(8, 16))),
            ),
            (RouteConfig(
                "prefill",
                "decode",
                "kv",
                "send_recv",
                "nvlink",
                "point_to_point",
                resource_id="dsv3-kv-link",
                backend="analytical_roofline_v1",
            ),),
        ),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )


def test_dsv3_pd_transfer_tp8_ep1_mla_only_replicated_cache_contract():
    model = load_model_config("configs/model/deepseek-v3.json")
    deployment = _dsv3_h200_pd_ep1_deployment(model)

    graph = build_pd_kv_transfer_workload(
        "target-model", 3, model, deployment, "prefill", "decode")

    # 61 layers * (512 + 64) * 2 B compressed MLA kv_c||k_pe only;
    # DeepSeek-V3 has no DSA index cache.  The cache is single-head
    # TP-replicated, so the per-rank payload is EP-independent and the EP1
    # contract matches the aggregate layout.  The logical head range covers
    # the 128 attention heads sharded over TP8 (16 per rank).
    assert model.kv_bytes_per_token_per_parallel_rank(8, 1) == 70_272
    expected_bytes = 3 * 70_272
    assert tuple(
        (
            node.attributes["source_logical_rank"],
            node.attributes["destination_logical_rank"],
            node.attributes["kv_head_start"],
            node.attributes["kv_head_end"],
            tuple(node.attributes["participants"]),
            node.attributes["bytes"],
        )
        for node in graph.nodes
    ) == tuple(
        (rank, rank, 16 * rank, 16 * rank + 16, (rank, rank + 8),
         expected_bytes)
        for rank in range(8)
    )


def test_dsv3_pd_transfer_rejects_ep8_pools():
    model = load_model_config("configs/model/deepseek-v3.json")
    deployment = _dsv3_h200_pd_ep1_deployment(
        model, enable_expert_parallel=True)

    with pytest.raises(
            NotImplementedError,
            match="DeepSeek-V3 PD transfer requires a compressed MLA"):
        build_pd_kv_transfer_workload(
            "target-model", 3, model, deployment, "prefill", "decode")


def test_dsv3_h200_colocated_tp8_pp1_ep1_emits_logical_workload():
    model = load_model_config("configs/model/deepseek-v3.json")
    collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            "main-collective-fabric",
            "analytical_roofline_v1",
            None,
        )
        for operation in ("all_reduce", "all_gather"))
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_h200",
                ParallelConfig(tensor_parallel_size=8),
                tuple(range(8)),
                collectives=collectives,
            ),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )

    graph = _target_prefill_step_workload(model, deployment)

    assert graph.node("target-model:token-embedding").attributes[
        "model_semantics_contract"] == "deepseek_v3_vllm_0_27_1_logical_v1"
    assert any(
        node.kind is WorkloadKind.COLLECTIVE for node in graph.nodes)


def _dsv3_h200_colocated_mtp_step(phases, serving_framework,
                                   graph_execution=None):
    model = load_model_config("configs/model/deepseek-v3.json")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main", "colocated", "nvidia_h200",
                ParallelConfig(tensor_parallel_size=8), tuple(range(8)),
                collectives=tuple(
                    CollectiveConfig(
                        operation, "ring", "nvlink", "ring",
                        "main-collective-fabric", "analytical_roofline_v1",
                        None)
                    for operation in ("all_reduce", "all_gather"))),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    mtp = {
        "mtp_proposer_tokens": 2,
        "mtp_next_proposed_tokens": 2,
        "mtp_proposer_mode": "padded",
        "mtp_acceptance_source_kind": "explicit_assumption",
        "mtp_acceptance_source_id": "fixed-1",
    }
    verify = {
        "raw_model_output_tokens": 2,
        "mtp_proposed_tokens": 2,
        "mtp_verified_tokens": 2,
        "mtp_accepted_tokens": 1,
    }
    requests = tuple(
        ScheduledRequest(f"request-{index}", "prefill", 4, 0, 0, **mtp)
        if phase == "prefill"
        else ScheduledRequest(
            f"request-{index}", "decode", 0, 3, 16, **verify, **mtp)
        for index, phase in enumerate(phases))
    return build_workload(
        ScheduleStep("step", "main", requests, attention_kv_layout="mla"),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform",
            **({} if graph_execution is None
               else {"graph_execution": graph_execution}),
            serving_framework=serving_framework),
    )


def test_sglang_eagle_verify_replays_the_target_graph_only():
    # Two 3-token verify requests padded to the 12-token capture size:
    # target nodes carry the graph domain, draft and EAGLE glue stay eager.
    graph_execution = GraphExecutionConfig(
        "FULL_DECODE_ONLY", (3, 12), (), 3)
    graph = _dsv3_h200_colocated_mtp_step(
        ("decode", "decode"), "sglang", graph_execution)
    attention = graph.node("step:layer-3:decode-mla-attention")
    assert attention.attributes["graph_runtime_mode"] == "FULL"
    assert attention.attributes["graph_model_input_tokens"] == 12
    assert attention.attributes["graph_logical_tokens"] == 6
    assert not any(
        "graph_runtime_mode" in node.attributes
        for node in graph.nodes
        if ":mtp-pass-" in node.node_id or ":sglang-eagle:" in node.node_id)
    with pytest.raises(NotImplementedError, match="MTP is not source-closed"):
        _dsv3_h200_colocated_mtp_step(
            ("decode", "decode"), "vllm", graph_execution)


def test_sglang_router_declares_its_cublas_runner_and_vllm_does_not():
    sglang = _dsv3_h200_colocated_mtp_step(("decode", "decode"), "sglang")
    vllm = _dsv3_h200_colocated_mtp_step(("decode", "decode"), "vllm")
    assert (sglang.node("step:layer-3:router:tp-rank-0").attributes[
        "router_linear_runner"] == "sglang_linear_bf16_fp32_cublas")
    assert "router_linear_runner" not in vllm.node(
        "step:layer-3:router:tp-rank-0").attributes


def test_sglang_eagle_prefill_draft_head_is_last_row_and_single_pass():
    # SGLang's prefill-batch draft extend prunes to each request's last row
    # before the lm_head and runs no draft decode pass; vLLM keeps its
    # proposer passes.
    sglang = _dsv3_h200_colocated_mtp_step(("prefill", "prefill"), "sglang")
    assert sglang.node("step:mtp-pass-0:head").shape["m"] == 2
    assert sglang.node("step:mtp-pass-0:head-norm").shape["rows"] == 2
    assert not any("mtp-pass-1" in node.node_id for node in sglang.nodes)
    vllm = _dsv3_h200_colocated_mtp_step(("prefill", "prefill"), "vllm")
    assert any("mtp-pass-1" in node.node_id for node in vllm.nodes)


def test_sglang_eagle_mixed_batch_fails_closed():
    with pytest.raises(NotImplementedError, match="all-prefill or all-decode"):
        _dsv3_h200_colocated_mtp_step(("prefill", "decode"), "sglang")


def test_dsv3_colocated_tp8_pp1_ep8_scope_remains_rejected():
    model = load_model_config("configs/model/deepseek-v3.json")
    deployment = _h200_colocated_tp8_ep8_deployment(model)

    with pytest.raises(
            NotImplementedError,
            match="H200 DeepSeek-V3 TP8/PP1/EP1 aggregate"):
        _target_prefill_step_workload(model, deployment)


def test_glm_kimi_dsv3_h200_ep1_aggregate_scopes_stay_model_gated():
    # The GLM-5.2 and Kimi-K2.6 gates must not admit DeepSeek-V3, and the
    # DeepSeek-V3 gate must not admit them.
    for fixture in (
            "configs/model/glm-5.2-fp8.json",
            "configs/model/kimi-k2.6.json"):
        model = load_model_config(fixture)
        deployment = build_deployment_plan(
            DeploymentConfig("colocated", (
                WorkerPoolConfig(
                    "main",
                    "colocated",
                    "nvidia_h200",
                    ParallelConfig(tensor_parallel_size=8),
                    tuple(range(8)),
                    collectives=tuple(
                        CollectiveConfig(
                            operation,
                            "ring",
                            "nvlink",
                            "ring",
                            "main-collective-fabric",
                            "analytical_roofline_v1",
                            None,
                        )
                        for operation in ("all_reduce", "all_gather")),
                ),
            )),
            num_layers=model.num_hidden_layers,
            num_experts=model.num_experts,
        )
        if model.model_type == "glm_moe_dsa":
            # GLM-5.2 owns the EP1 aggregate scope; it stays admitted.
            graph = _target_prefill_step_workload(model, deployment)
            assert graph.node("target-model:token-embedding").attributes[
                "model_semantics_contract"] == (
                    "glm_5_2_vllm_0_27_1_logical_v1")
        else:
            with pytest.raises(NotImplementedError):
                _target_prefill_step_workload(model, deployment)


@pytest.mark.parametrize(("fixture", "layout", "contract"), [
    (
        "tests/serving/fixtures/glm_5_2_official",
        "mla",
        "glm_5_2_vllm_0_27_1_logical_v1",
    ),
    (
        "configs/model/glm-5.2-fp8.json",
        "mla",
        "glm_5_2_vllm_0_27_1_logical_v1",
    ),
    (
        "tests/serving/fixtures/kimi_k2_6_official",
        "mla",
        "kimi_k2_6_vllm_0_27_1_logical_v1",
    ),
    (
        "configs/model/deepseek-v3.json",
        "mla",
        "deepseek_v3_vllm_0_27_1_logical_v1",
    ),
])
def test_target_models_emit_pinned_logical_workload(
        fixture, layout, contract):
    model = load_model_config(fixture)
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )

    graph = build_workload(
        ScheduleStep(
            "target-model",
            "main",
            (ScheduledRequest("request", "prefill", 1, 0, 0),),
            attention_kv_layout=layout,
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        kv_cache_events=(
            KVCacheEvent(
                "request", KVCacheAction.ALLOCATE, "hbm", 0, 16, 1024),
            KVCacheEvent(
                "request", KVCacheAction.MISS, "hbm", 0, 1, 64),
            KVCacheEvent(
                "request", KVCacheAction.HIT, "hbm", 0, 1, 64),
            KVCacheEvent(
                "request", KVCacheAction.RELEASE, "hbm", 0, 16, 1024),
        ),
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            layout, "uniform"),
    )

    assert graph.node("target-model:token-embedding").attributes[
        "model_semantics_contract"] == contract
    assert all(
        node.kind is not WorkloadKind.KV_OFFLOAD for node in graph.nodes)


def _h200_colocated_tp8_ep1_deployment(model):
    collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            "main-collective-fabric",
            "analytical_roofline_v1",
            None,
        )
        for operation in ("all_reduce", "all_gather"))
    return build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_h200",
                ParallelConfig(tensor_parallel_size=8),
                tuple(range(8)),
                collectives=collectives,
            ),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )


@pytest.mark.parametrize(("fixture", "ep_enabled"), [
    ("configs/model/deepseek-v3.json", False),
    ("configs/model/glm-5.2-fp8.json", False),
    ("configs/model/kimi-k2.6.json", True),
])
def test_moe_forward_weight_bytes_match_residency(fixture, ep_enabled):
    # The forward FUSED_MOE shape and the analytical residency model must
    # quote the same per-rank MoE weight bytes: N/EP owned experts whose
    # linears are sharded across the TP group (model_residency.py's TP8/EP1
    # comment), never the full unsharded expert tensor per rank.
    model = load_model_config(fixture)
    if ep_enabled:
        deployment = _h200_colocated_tp8_ep8_deployment(model)
    else:
        deployment = _h200_colocated_tp8_ep1_deployment(model)
    graph = build_workload(
        ScheduleStep(
            "target-model",
            "main",
            (ScheduledRequest("request", "prefill", 64, 0, 0),),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        kv_cache_events=(),
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )
    layer_index = next(
        index for index in range(model.num_hidden_layers)
        if model.layer_uses_moe(index))
    prefix = f"target-model:layer-{layer_index}"
    parallelism = deployment.worker_pools["main"].parallel_plan.config
    tp_size = parallelism.tensor_parallel_size
    ep_size = parallelism.expert_parallel_size
    hidden = model.hidden_size
    router_bytes = _linear_weight_bytes(
        model,
        out_features=model.num_experts,
        in_features=hidden,
        module_path=f"model.layers.{layer_index}.mlp.gate",
        module_type="GateLinear",
    )
    residency_bytes = _target_moe_weight_bytes(
        model, parallelism, layer_index)
    # The noaux_tc correction bias is a routing-table buffer that never
    # enters a forward GEMM, so the forward weight sum excludes it.
    bias_bytes = (
        model.num_experts * 4
        if model.target_semantics.moe_topk_method == "noaux_tc" else 0)
    expert_group = deployment.worker_pools[
        "main"].parallel_plan.rank_groups.for_rank("expert", 0)
    for expert_rank in range(len(expert_group)):
        rank_suffix = f"tp-rank-{expert_rank}"
        routed = graph.node(f"{prefix}:routed-experts:{rank_suffix}")
        local_intermediate = routed.shape["intermediate"]
        assert local_intermediate == (
            model.moe_intermediate_size * ep_size // tp_size)
        routed_bytes = 0
        for expert_id in tuple(routed.attributes["owned_experts"]):
            expert = (
                f"model.layers.{layer_index}.mlp.experts.{expert_id}")
            routed_bytes += _linear_weight_bytes(
                model,
                out_features=2 * local_intermediate,
                in_features=hidden,
                module_path=f"{expert}.gate_proj",
            )
            routed_bytes += _linear_weight_bytes(
                model,
                out_features=hidden,
                in_features=local_intermediate,
                module_path=f"{expert}.down_proj",
            )
        shared_gate_up = graph.node(
            f"{prefix}:shared-experts:gate-up:{rank_suffix}")
        shared_down = graph.node(
            f"{prefix}:shared-experts:down-proj:{rank_suffix}")
        shared_prefix = (
            f"model.layers.{layer_index}.mlp.shared_experts")
        shared_bytes = (
            _linear_weight_bytes(
                model,
                out_features=shared_gate_up.shape["n"],
                in_features=shared_gate_up.shape["k"],
                module_path=f"{shared_prefix}.gate_proj",
            )
            + _linear_weight_bytes(
                model,
                out_features=shared_down.shape["n"],
                in_features=shared_down.shape["k"],
                module_path=f"{shared_prefix}.down_proj",
            ))
        forward_bytes = routed_bytes + shared_bytes + router_bytes
        assert forward_bytes == residency_bytes - bias_bytes


def test_fp8_routed_expert_operands_stream_one_byte_elements():
    # Pinned vLLM quantizes both W1 and W2 inputs before the use_fp8_w8a8
    # grouped GEMM, so an FP8 routed expert streams one-byte A/B operands
    # while its output cache stays in the BF16 compute dtype.
    from src.kernel.predictors.fused_moe import _gpu_traffic_components

    model = load_model_config("configs/model/deepseek-v3.json")
    deployment = _h200_colocated_tp8_ep1_deployment(model)
    hardware = HWSpec("configs/hardware/nvidia_h200.yaml")
    graph = build_workload(
        ScheduleStep(
            "target-model",
            "main",
            (ScheduledRequest("request", "prefill", 64, 0, 0),),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        kv_cache_events=(),
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )
    plan = lower_workload(graph, deployment, {"nvidia_h200": hardware})
    routed = [
        call.kernel_spec for call in plan.calls
        if call.kernel_spec.kind is KernelKind.FUSED_MOE
        and call.kernel_spec.variant in ("w1", "w2")]
    assert {spec.variant for spec in routed} == {"w1", "w2"}
    for spec in routed:
        assert spec.dtype == "bf16"
        assert spec.attributes["routed_weight_dtype"] == "fp8"
        hidden = spec.shape["hidden"]
        intermediate = spec.shape["intermediate"]
        k, n = ((hidden, 2 * intermediate) if spec.variant == "w1"
                else (intermediate, hidden))
        active_experts = sum(
            1 for count in spec.attributes["local_tokens_per_expert"]
            if count > 0)
        routed_rows = sum(spec.attributes["local_tokens_per_expert"])
        fp8 = _gpu_traffic_components(spec, hardware)
        assert fp8.weight_compulsory_read_bytes == active_experts * k * n
        assert fp8.output_dram_write_bytes == routed_rows * n * 2
        bf16 = _gpu_traffic_components(
            replace(spec, attributes={
                **spec.attributes, "routed_weight_dtype": "bf16"}),
            hardware)
        assert bf16.weight_compulsory_read_bytes == (
            2 * fp8.weight_compulsory_read_bytes)
        assert bf16.activation_compulsory_read_bytes == (
            2 * fp8.activation_compulsory_read_bytes)
        assert bf16.output_dram_write_bytes == fp8.output_dram_write_bytes
        # The complete Forward input must resolve for FP8 operands too.
        for method in ("roofline", "midlayer"):
            assert predict(spec, hardware, method).t_total_s > 0.0


def test_roofline_span_floor_bounds_roofline_by_measured_launch_floor():
    # The floored method keeps Roofline as the baseline and composes the
    # published memory-touching launch span floor by maximum; transfers carry
    # no GPU launch and keep their Roofline price.
    from src.kernel.formula import small_kernel_span_floor_s
    from src.kernel.predictors.common import graph_replayed_launch

    model = load_model_config("configs/model/deepseek-v3.json")
    deployment = _h200_colocated_tp8_ep1_deployment(model)
    hardware = HWSpec("configs/hardware/nvidia_h200.yaml")
    graph = build_workload(
        ScheduleStep(
            "target-model",
            "main",
            (ScheduledRequest("request", "prefill", 64, 0, 0),),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        kv_cache_events=(),
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )
    plan = lower_workload(graph, deployment, {"nvidia_h200": hardware})
    floored_count = 0
    for call in plan.calls:
        spec = call.kernel_spec
        roofline = predict(spec, hardware, "roofline").t_total_s
        floored = predict(spec, hardware, "roofline_span_floor").t_total_s
        if spec.transfer is not None:
            assert floored == roofline
            continue
        floor = small_kernel_span_floor_s(
            hardware, spec.launch.grid_size, spec.variant,
            graph_replayed_launch(spec))
        assert floored == pytest.approx(max(roofline, floor))
        floored_count += roofline < floor
    assert floored_count > 0


def test_glm_target_workload_rejects_unregistered_config_identity():
    model = replace(
        load_model_config("configs/model/glm-5.2-fp8.json"),
        config_sha256="0" * 64,
    )
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )

    with pytest.raises(ValueError, match="fixed official model identity"):
        build_workload(
            ScheduleStep(
                "target-model",
                "main",
                (ScheduledRequest("request", "prefill", 1, 0, 0),),
                attention_kv_layout="mla",
            ),
            model,
            deployment,
            launch_policy="analytical_launch",
            kv_cache_events=(
                KVCacheEvent(
                    "request", KVCacheAction.ALLOCATE, "hbm", 0, 16, 1024),
                KVCacheEvent(
                    "request", KVCacheAction.MISS, "hbm", 0, 1, 64),
                KVCacheEvent(
                    "request", KVCacheAction.HIT, "hbm", 0, 1, 64),
                KVCacheEvent(
                    "request", KVCacheAction.RELEASE, "hbm", 0, 16, 1024),
            ),
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.seeded_assumption(
                    "test_seeded_routing_v1", 0),
                "mla", "uniform"),
        )


def _h200_colocated_tp8_ep8_deployment(model, hardware_id="nvidia_h200"):
    collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            "main-collective-fabric",
            "analytical_roofline_v1",
            None,
        )
        for operation in ("all_reduce", "all_gather"))
    return build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                hardware_id,
                ParallelConfig(
                    tensor_parallel_size=8,
                    enable_expert_parallel=True,
                ),
                tuple(range(8)),
                collectives=collectives,
            ),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )


def _target_prefill_step_workload(model, deployment):
    return build_workload(
        ScheduleStep(
            "target-model",
            "main",
            (ScheduledRequest("request", "prefill", 4, 0, 0),),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        kv_cache_events=(
            KVCacheEvent(
                "request", KVCacheAction.ALLOCATE, "hbm", 0, 16, 1024),
            KVCacheEvent(
                "request", KVCacheAction.MISS, "hbm", 0, 1, 64),
            KVCacheEvent(
                "request", KVCacheAction.HIT, "hbm", 0, 1, 64),
            KVCacheEvent(
                "request", KVCacheAction.RELEASE, "hbm", 0, 16, 1024),
        ),
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )


def test_kimi_k2_6_h200_colocated_tp8_pp1_ep8_emits_logical_workload():
    model = load_model_config("configs/model/kimi-k2.6.json")
    deployment = _h200_colocated_tp8_ep8_deployment(model)

    graph = _target_prefill_step_workload(model, deployment)

    assert graph.node("target-model:token-embedding").attributes[
        "model_semantics_contract"] == "kimi_k2_6_vllm_0_27_1_logical_v1"
    assert any(
        node.kind is WorkloadKind.COLLECTIVE for node in graph.nodes)


def test_kimi_k2_6_colocated_tp8_pp1_ep8_scope_requires_h200():
    model = load_model_config("configs/model/kimi-k2.6.json")
    deployment = _h200_colocated_tp8_ep8_deployment(
        model, hardware_id="nvidia_a800")

    with pytest.raises(
            NotImplementedError,
            match="H200 Kimi-K2.6 colocated TP8/PP1/EP8"):
        _target_prefill_step_workload(model, deployment)


def test_glm_5_2_colocated_tp8_pp1_ep8_scope_remains_rejected():
    model = load_model_config("configs/model/glm-5.2-fp8.json")
    deployment = _h200_colocated_tp8_ep8_deployment(model)

    with pytest.raises(
            NotImplementedError,
            match="PP4 for GLM-5.2"):
        _target_prefill_step_workload(model, deployment)


@pytest.mark.parametrize(("fixture", "expected_variants"), [
    (
        "tests/serving/fixtures/kimi_k2_6_official",
        {
            "batched_gemm",
            "mla_prefill",
            "mla_decode",
            "combine",
            "mla_rope",
            "moe_router",
            "mla_kv_cache",
            "routed_experts",
        },
    ),
    (
        "tests/serving/fixtures/glm_5_2_official",
        {
            "batched_gemm",
            "layernorm",
            "mla_rope",
            "moe_router",
            "mla_kv_cache",
            "dsa_fp8_index_cache",
            "dsa_indexer_mixed",
            "dsa_index_reuse",
            "dsa_sparse_prefill",
            "dsa_sparse_decode",
        },
    ),
])
def test_target_model_compute_launches_reselect_from_source_abi(
        fixture, expected_variants):
    """Every target-only compute launch is reproducible from its source ABI.

    The mixed step deliberately emits prefill and decode paths together.  It
    covers Kimi MLA's split decode combine and GLM DSA's full/shared indexer
    paths, in addition to the target GEMM, cache, vector, and MoE variants.
    """

    model = load_model_config(fixture)
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    graph = build_workload(
        ScheduleStep(
            "target-launch-audit",
            "main",
            (
                ScheduledRequest("prefill", "prefill", 4, 0, 0),
                ScheduledRequest("decode", "decode", 1, 0, 1024),
            ),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    plan = lower_workload(
        graph, deployment, {"nvidia_a800": hardware})
    target_calls = tuple(
        call for call in plan.calls
        if call.kernel_spec.variant in expected_variants)
    assert {call.kernel_spec.variant for call in target_calls} == (
        expected_variants)

    for call in target_calls:
        spec = call.kernel_spec
        source_node_ids = call.source_node_ids
        assert len(source_node_ids) == 1
        source = graph.node(source_node_ids[0])
        selection_attributes = dict(source.attributes)
        if spec.variant == "combine":
            assert source.variant == "mla_decode"
            selection_attributes.update({
                "fa_ksteps": None,
                "fa_eff_mfrac": 1.0,
                "num_kv_splits": spec.launch.tiling.num_kv_splits,
            })
        expected = select_launch_config(
            "nvidia_a800",
            spec.kind,
            spec.shape,
            spec.dtype,
            spec.variant,
            selection_attributes,
            hardware=hardware,
        )
        assert spec.launch == expected

    # Target graphs also retain ordinary FFN, residual, embedding, norm, and
    # logits operators.  Exercise their real target-model shapes rather than
    # relying only on the generic synthetic-kind lowering test.
    def vector_shape(node):
        shape = dict(node.shape)
        shape["element_count"] = shape.pop("elements")
        return shape

    direct_cases = {
        WorkloadKind.LINEAR: (KernelKind.GEMM, lambda node: node.variant,
                              lambda node: dict(node.shape)),
        WorkloadKind.LM_HEAD: (KernelKind.GEMM, lambda node: node.variant,
                               lambda node: dict(node.shape)),
        WorkloadKind.NORM: (KernelKind.VECTOR, lambda node: node.variant,
                            lambda node: dict(node.shape)),
        WorkloadKind.ACTIVATION: (
            KernelKind.VECTOR, lambda node: node.variant, vector_shape),
        WorkloadKind.ELEMENTWISE: (
            KernelKind.VECTOR,
            lambda node: "elementwise",
            vector_shape,
        ),
        WorkloadKind.GATHER: (
            KernelKind.VECTOR,
            lambda node: "gather",
            vector_shape,
        ),
    }
    observed_direct_kinds = set()
    for call in plan.calls:
        spec = call.kernel_spec
        if len(call.source_node_ids) != 1:
            continue
        source = graph.node(call.source_node_ids[0])
        if source.kind not in direct_cases:
            continue
        expected_kind, variant_for, shape_for = direct_cases[source.kind]
        expected_variant = variant_for(source)
        if (spec.kind is not expected_kind
                or spec.variant != expected_variant):
            continue
        expected = select_launch_config(
            "nvidia_a800",
            expected_kind,
            shape_for(source),
            source.dtype,
            expected_variant,
            source.attributes,
            hardware=hardware,
        )
        assert spec.launch == expected
        observed_direct_kinds.add(source.kind)
    assert observed_direct_kinds == set(direct_cases)


def test_glm_dsa_indexer_accepts_single_token_prefill_tail_chunk():
    # A 1-token chunked-prefill/resume tail row keeps its prefill phase in
    # the DSA indexer batch while sharing it with a q_len=1 decode row --
    # the GLM PD decode-pool shape that must lower instead of tripping the
    # indexer phase/q_len check.
    model = load_model_config("tests/serving/fixtures/glm_5_2_official")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    graph = build_workload(
        ScheduleStep(
            "target-dsa-prefill-tail",
            "main",
            (
                ScheduledRequest("prefill-tail", "prefill", 1, 0, 1023),
                ScheduledRequest("decode", "decode", 1, 0, 1024),
            ),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )
    indexer_nodes = tuple(
        node for node in graph.nodes
        if node.variant == "dsa_indexer_mixed")
    assert indexer_nodes
    for node in indexer_nodes:
        phases = node.attributes["request_phases"]
        q_lens = node.attributes["q_lens"]
        assert "prefill" in phases and "decode" in phases
        assert all(
            phase == "prefill" or q_len == 1
            for phase, q_len in zip(phases, q_lens, strict=True))

    plan = lower_workload(
        graph,
        deployment,
        {"nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml")},
    )
    indexer_specs = tuple(
        call.kernel_spec for call in plan.calls
        if call.kernel_spec.kind is KernelKind.SPARSE_ATTN_INDEXER
        and call.kernel_spec.variant == "dsa_indexer_mixed")
    assert indexer_specs
    for spec in indexer_specs:
        phases = spec.attributes["request_phases"]
        q_lens = spec.attributes["q_lens"]
        assert all(
            phase == "prefill" or q_len == 1
            for phase, q_len in zip(phases, q_lens, strict=True))


def _colocated_target_deployment(model):
    return build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )


def test_kimi_mla_decode_verify_window_takes_decode_attention_path():
    # A decode-phase multi-token row is an MTP/EAGLE verify window: real
    # backends run it through the MLA decode (absorbed) path, so the layer
    # must emit the decode q-side batched GEMMs and an mla_decode attention
    # kernel with q_len > 1 -- never the prefill full-prefix kv_b_proj /
    # mla_prefill pair.
    model = load_model_config("tests/serving/fixtures/kimi_k2_6_official")
    deployment = _colocated_target_deployment(model)
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    graph = build_workload(
        ScheduleStep(
            "target-mla-verify-window",
            "main",
            (
                ScheduledRequest("verify", "decode", 0, 4, 1024),
                ScheduledRequest("decode", "decode", 0, 1, 1024),
            ),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )
    layer0 = "target-mla-verify-window:layer-0"
    decode_attention = graph.node(f"{layer0}:decode-mla-attention")
    assert decode_attention.variant == "mla_decode"
    assert decode_attention.shape["q_len"] == 4
    assert decode_attention.attributes["q_lens"] == (4, 1)
    assert decode_attention.attributes["kv_lens"] == (1028, 1025)
    assert graph.node(f"{layer0}:decode-q-down-proj").shape["m"] == 5
    assert graph.node(f"{layer0}:decode-v-up-proj").shape["m"] == 5
    assert not any(
        node.node_id.endswith(":prefill-kv-b-proj") for node in graph.nodes)
    assert not any(node.variant == "mla_prefill" for node in graph.nodes)

    plan = lower_workload(graph, deployment, {"nvidia_a800": hardware})
    decode_specs = tuple(
        call.kernel_spec for call in plan.calls
        if call.kernel_spec.variant == "mla_decode")
    assert decode_specs
    for spec in decode_specs:
        assert spec.launch.grid[0] == sum(spec.attributes["q_lens"])
    predictions = predict_kernel_plan(
        plan, {"nvidia_a800": hardware}, method="midlayer")
    assert all(
        prediction.t_total_ns > 0 for prediction in predictions.values())


def test_kimi_chunked_prefill_row_keeps_prefill_attention_path():
    # A genuine chunked-prefill row (still consuming its prompt) keeps the
    # prefill attention path -- including the full-prefix kv_b_proj -- while
    # the q_len=1 decode row in the same step takes the decode path.
    model = load_model_config("tests/serving/fixtures/kimi_k2_6_official")
    deployment = _colocated_target_deployment(model)
    graph = build_workload(
        ScheduleStep(
            "target-mla-chunked-prefill",
            "main",
            (
                ScheduledRequest("chunk", "prefill", 3, 0, 1021),
                ScheduledRequest("decode", "decode", 0, 1, 1024),
            ),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )
    layer0 = "target-mla-chunked-prefill:layer-0"
    kv_b = graph.node(f"{layer0}:prefill-kv-b-proj")
    assert kv_b.shape["m"] == 1024
    prefill_attention = graph.node(f"{layer0}:prefill-mla-attention")
    assert prefill_attention.variant == "mla_prefill"
    assert prefill_attention.attributes["q_lens"] == (3,)
    assert prefill_attention.attributes["kv_lens"] == (1024,)
    decode_attention = graph.node(f"{layer0}:decode-mla-attention")
    assert decode_attention.variant == "mla_decode"
    assert decode_attention.attributes["q_lens"] == (1,)
    assert decode_attention.attributes["kv_lens"] == (1025,)


def test_glm_mtp_verify_window_and_proposer_take_decode_attention_path():
    # The GLM MTP verify row (q_len=2) and the proposer pass-0 row both run
    # through the sparse-MLA decode path in real backends (the V3.2 indexer
    # admits decode-group q_len=2), so the workload must not emit any
    # prefill-kv-b-proj and the indexer must take its decode variant.
    model = load_model_config("tests/serving/fixtures/glm_5_2_official")
    deployment = _colocated_target_deployment(model)
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    graph = build_workload(
        ScheduleStep(
            "target-dsa-mtp-verify",
            "main",
            (ScheduledRequest(
                "verify",
                "decode",
                0,
                2,
                1024,
                raw_model_output_tokens=2,
                mtp_proposed_tokens=1,
                mtp_verified_tokens=1,
                mtp_accepted_tokens=1,
                mtp_proposer_tokens=1,
                mtp_next_proposed_tokens=1,
                mtp_proposer_mode="padded",
                mtp_acceptance_source_kind="explicit_assumption",
                mtp_acceptance_source_id="fixed-1",
            ),),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )
    assert not any(
        node.node_id.endswith(":prefill-kv-b-proj") for node in graph.nodes)
    layer0 = "target-dsa-mtp-verify:layer-0"
    indexer = graph.node(f"{layer0}:indexer-score-topk")
    assert indexer.variant == "dsa_indexer_decode"
    assert indexer.attributes["request_phases"] == ("decode",)
    assert indexer.attributes["q_lens"] == (2,)
    decode_attention = graph.node(f"{layer0}:decode-dsa-attention")
    assert decode_attention.variant == "dsa_sparse_decode"
    assert decode_attention.attributes["q_lens"] == (2,)
    assert graph.node(f"{layer0}:decode-q-down-proj").shape["m"] == 2
    proposer = "target-dsa-mtp-verify:mtp-pass-0:decoder"
    proposer_attention = graph.node(f"{proposer}:decode-dsa-attention")
    assert proposer_attention.variant == "dsa_sparse_decode"
    assert proposer_attention.attributes["q_lens"] == (2,)
    assert graph.node(f"{proposer}:decode-q-down-proj").shape["m"] == 2

    plan = lower_workload(graph, deployment, {"nvidia_a800": hardware})
    predictions = predict_kernel_plan(
        plan, {"nvidia_a800": hardware}, method="midlayer")
    assert all(
        prediction.t_total_ns >= 0 for prediction in predictions.values())
    attention_predictions = tuple(
        predictions[call.call_id] for call in plan.calls
        if call.kernel_spec.variant == "dsa_sparse_decode")
    assert attention_predictions
    assert all(
        prediction.t_total_ns > 0 for prediction in attention_predictions)


def test_target_routed_experts_is_not_a_cross_hardware_fallback():
    """The Kimi Marlin aggregate has no C500/910B source launch contract."""

    model = load_model_config("tests/serving/fixtures/kimi_k2_6_official")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    graph = build_workload(
        ScheduleStep(
            "target-routed-experts",
            "main",
            (ScheduledRequest("decode", "decode", 1, 0, 1024),),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla", "uniform"),
    )
    source = next(
        node for node in graph.nodes
        if (node.kind is WorkloadKind.FUSED_MOE
            and node.attributes["routed_weight_dtype"] == "int4"))
    for hardware_id in ("metax_c500", "ascend_910b"):
        hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
        with pytest.raises(
                NotImplementedError,
                match=(
                    f"no source-backed {hardware_id} launch selector for "
                    "FUSED_MOE/routed_experts")):
            select_launch_config(
                hardware_id,
                KernelKind.FUSED_MOE,
                source.shape,
                source.dtype,
                "routed_experts",
                source.attributes,
                hardware=hardware,
            )


def test_target_models_reject_actual_kv_offload_events():
    model = load_model_config(
        "tests/serving/fixtures/kimi_k2_6_official")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    event = KVCacheEvent(
        "request",
        KVCacheAction.EVICT,
        "host",
        0,
        16,
        1024,
        source_tier="hbm",
        destination_tier="host",
        ready_event="evict-request-0",
    )

    with pytest.raises(
            NotImplementedError,
            match="does not yet support KV offload"):
        build_workload(
            ScheduleStep(
                "target-model",
                "main",
                (ScheduledRequest("request", "prefill", 1, 0, 0),),
                attention_kv_layout="mla",
            ),
            model,
            deployment,
            launch_policy="analytical_launch",
            kv_cache_events=(event,),
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.seeded_assumption(
                    "test_seeded_routing_v1", 0),
                "mla", "uniform"),
        )


def test_target_model_requires_fixed_revision_provenance(tmp_path):
    config = json.loads(Path(
        "tests/serving/fixtures/glm_5_2_official/config.json"
    ).read_text())

    with pytest.raises(ValueError, match="requires fixed model provenance"):
        load_model_config(str(_write_model_config(tmp_path, config)))


def test_target_model_cannot_self_attest_a_modified_config(tmp_path):
    source = Path("tests/serving/fixtures/glm_5_2_official")
    config = json.loads((source / "config.json").read_text())
    config["rope_parameters"]["rope_theta"] = 12345
    model_dir = _write_model_config(tmp_path, config)
    provenance = json.loads(
        (source / "model_provenance.json").read_text())
    provenance["fixture_config_sha256"] = hashlib.sha256(
        (model_dir / "config.json").read_bytes()).hexdigest()
    (model_dir / "model_provenance.json").write_text(
        json.dumps(provenance))

    with pytest.raises(ValueError, match="fixed target revision"):
        load_model_config(str(model_dir))


def _glm_model_config():
    return json.loads(Path(
        "tests/serving/fixtures/glm4_moe_mtp_scheduler/config.json"
    ).read_text())


def test_glm_model_schema_normalizes_source_default_routed_scale(tmp_path):
    config = _glm_model_config()
    del config["routed_scaling_factor"]

    model = load_model_config(str(_write_model_config(tmp_path, config)))

    assert model.routed_scaling_factor == 1.0


def test_glm_nonidentity_routed_scale_fails_without_launch_evidence(tmp_path):
    config = _glm_model_config()
    config["routed_scaling_factor"] = 1.25
    model = load_model_config(str(_write_model_config(tmp_path, config)))
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    step = ScheduleStep(
        "scaled-glm",
        "main",
        (ScheduledRequest("request", "prefill", 1, 0),),
    )

    with pytest.raises(NotImplementedError, match="will not invent"):
        build_workload(
            step,
            model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=RuntimeInputs(
                ExpertRoutingInput.seeded_assumption(
                    "test_seeded_routing_v1", 0),
                "contiguous", "uniform"),
        )


def test_moe_routing_relabelling_is_withheld_for_an_order_sensitive_recipe(
        monkeypatch):
    """A recipe reading the axis as a prefix stream keeps its exact routing."""

    _, deployment, graph = _ep2_workload(
        "ep-relabel-gate",
        (ScheduledRequest("r0", "decode", 0, 1, 4),),
        {0: 0, 1: 1},
    )
    del deployment
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    node = next(
        node for node in graph.nodes
        if node.kind is WorkloadKind.FUSED_MOE)
    relabelling = _fused_moe_routing_relabelling(
        node, "nvidia_a800", hardware)
    assert relabelling is not None
    assert sorted(relabelling[0]) == sorted(
        node.attributes["tokens_per_expert"])

    lower_workload_module = sys.modules[
        "src.serving.kernel_lowering.lower_workload"]
    original = lower_workload_module.select_fused_moe_recipe

    def order_sensitive_recipe(hardware_id, shape, hardware_spec):
        return original(hardware_id, shape, hardware_spec) + (
            "moe_compute_expert_tokens",)

    monkeypatch.setattr(
        lower_workload_module, "select_fused_moe_recipe",
        order_sensitive_recipe)
    assert _fused_moe_routing_relabelling(
        node, "nvidia_a800", hardware) is None


def test_dsv3_fp8_dense_projections_lower_to_h200_block_fp8_gemms():
    # The DeepSeek-V3 FP8 checkpoint quantises every dense projection with
    # 128x128 blocks; lowering declares it and only the H200 block-FP8
    # selector consumes the declaration (A800 launches are unchanged).
    from src.kernel.launch_config.block_fp8_gemm import (
        DEEPGEMM_1D2D_ALGORITHM,
        TRTLLM_SWAP_AB_ALGORITHM,
    )

    model = load_model_config("configs/model/deepseek-v3-l32.json")
    deployment = _h200_colocated_tp8_ep1_deployment(model)
    hardware = HWSpec("configs/hardware/nvidia_h200.yaml")
    graph = _target_prefill_step_workload(model, deployment)
    plan = lower_workload(graph, deployment, {"nvidia_h200": hardware})
    gemms = [call for call in plan.calls
             if call.kernel_spec.kind is KernelKind.GEMM
             and call.rank == 0]
    positions = {}
    for call in gemms:
        position = graph.node(call.source_node_ids[0]).attributes["position"]
        if position not in positions:
            positions[position] = set()
        positions[position].add(call.kernel_spec.launch.algorithm)
    block_fp8 = {TRTLLM_SWAP_AB_ALGORITHM, DEEPGEMM_1D2D_ALGORITHM}
    for position in ("fused_qkv_a_proj", "q_b_proj", "kv_b_proj",
                     "o_proj", "gate_up_proj", "down_proj"):
        assert positions[position] <= block_fp8, position
    for position in ("router", "lm_head"):
        assert not positions[position] & block_fp8, position
