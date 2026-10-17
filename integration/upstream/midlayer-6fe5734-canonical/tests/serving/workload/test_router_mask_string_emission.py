"""Router-mask string emission mirror for the 910B CANN compare/cast pair.

Every test here mirrors one rule of helper contract §2.2 as it lands in the
workload graph: the 910B card emits the NotEqual/Cast ELEMENTWISE nodes and
the router run stamp, the other cards keep their existing emission shape,
and any count that leaves the 0/1-per-layer or shared-string contract fails
closed instead of degrading the emission.
"""

import pytest

from src.serving.deployment.deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.schedule.schedule_step import (
    ScheduleStep,
    ScheduledRequest,
)
from src.serving.workload import build_workload as build_workload_module
from src.serving.workload.build_workload import build_workload
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    RuntimeInputs,
)
from src.serving.workload.workload_graph import WorkloadKind


def _tp2_collectives():
    return (
        CollectiveConfig(
            "all_reduce", "ring", "nvlink", "ring",
            "main-nvlink-fabric", "analytical_roofline_v1"),
        CollectiveConfig(
            "all_gather", "ring", "nvlink", "ring",
            "main-nvlink-fabric", "analytical_roofline_v1"),
    )


def _ep2_deployment(hardware_id):
    model = load_model_config(
        "tests/serving/fixtures/qwen3_moe_ep2_scheduler")
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main", "colocated", hardware_id,
                ParallelConfig(
                    tensor_parallel_size=2,
                    enable_expert_parallel=True,
                ),
                ranks=(91, 7),
                collectives=_tp2_collectives(),
            ),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    return model, deployment


def _ep2_graph(hardware_id, token_count=4):
    model, deployment = _ep2_deployment(hardware_id)
    schedule = ScheduleStep(
        "mask-string", "main",
        (ScheduledRequest("r0", "decode", 0, 1, token_count),),
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
                    f"mask-string-{candidate_seed}", candidate_seed),
                "paged_varlen",
                "uniform",
            ),
        )
        if any(node.kind is WorkloadKind.FUSED_MOE for node in graph.nodes):
            return model, graph
    raise AssertionError("test could not find a seeded routing realization")


def _nodes(graph, suffix):
    return [
        node for node in graph.nodes
        if f":router-mask-{suffix}" in node.node_id
    ]


def test_910b_emits_the_compare_and_cast_nodes_with_the_string_stamp():
    model, graph = _ep2_graph("ascend_910b")

    notequal = _nodes(graph, "notequal")
    cast = _nodes(graph, "cast")
    index = _nodes(graph, "index")
    mul = _nodes(graph, "mul")
    assert notequal and cast and index and mul
    # Every member of the string shares the same entry count per launch
    # (tokens x top-k), so the compare/cast nodes are priced on exactly the
    # entries the trace-side string kernel processes.
    entry_counts = {
        tuple(node.shape.items())
        for node in notequal + cast + index + mul}
    assert len(entry_counts) == 1
    elements = next(iter(entry_counts))[0][1]
    routers = [
        node for node in graph.nodes
        if node.kind is WorkloadKind.ROUTER]
    assert routers
    assert all(
        node.shape["m"] * model.num_experts_per_tok == elements
        for node in routers)
    for node in notequal + cast:
        assert node.kind is WorkloadKind.ELEMENTWISE
        assert node.attributes["reads"] == 1
        assert node.attributes["writes"] == 1
    assert all(
        node.attributes["operation"] == "notequal_int32_bool"
        for node in notequal)
    assert all(
        node.attributes["helper_launch"] == {"class": "class2_router_notequal"}
        for node in notequal)
    assert all(
        node.attributes["operation"] == "cast_bool_bf16"
        for node in cast)
    assert all(
        node.attributes["helper_launch"] == {"class": "class2_router_mask_cast"}
        for node in cast)
    # The whole four-kernel string carries one router run per MoE layer.
    routers = [
        node for node in graph.nodes
        if node.kind is WorkloadKind.ROUTER]
    assert routers
    assert all(
        node.attributes["helper_launch"] == {"class": "class2_router_run"}
        for node in routers)


def test_other_cards_keep_their_existing_mask_emission_shape():
    # A800 stays on the single expert-map-index gather and never claims a
    # string run; C500 emits no routing-mask helper at all.
    _, a800_graph = _ep2_graph("nvidia_a800")
    assert not _nodes(a800_graph, "notequal")
    assert not _nodes(a800_graph, "cast")
    assert not _nodes(a800_graph, "index")
    assert [
        node for node in a800_graph.nodes
        if ":expert-map-index" in node.node_id]
    assert not any(
        "helper_launch" in node.attributes
        for node in a800_graph.nodes
        if node.kind is WorkloadKind.ROUTER)

    _, c500_graph = _ep2_graph("metax_c500")
    assert not _nodes(c500_graph, "notequal")
    assert not _nodes(c500_graph, "cast")
    assert not _nodes(c500_graph, "index")
    assert not [
        node for node in c500_graph.nodes
        if ":expert-map-index" in node.node_id]
    assert not any(
        "helper_launch" in node.attributes
        for node in c500_graph.nodes
        if node.kind is WorkloadKind.ROUTER)


def _patched_count(original, overrides):
    def count(hardware_id, call_point, phase, *, source_count, model_type):
        if call_point in overrides:
            return overrides[call_point]
        return original(
            hardware_id, call_point, phase,
            source_count=source_count, model_type=model_type)
    return count


def test_emission_count_outside_zero_or_one_fails_closed(monkeypatch):
    monkeypatch.setattr(
        build_workload_module, "helper_emission_count",
        _patched_count(
            build_workload_module.helper_emission_count,
            {"router_mask_cast": 2}))
    with pytest.raises(ValueError, match="must be 0 or 1"):
        _ep2_graph("ascend_910b")


def test_mixed_string_counts_fail_instead_of_claiming_a_run(monkeypatch):
    # A partial emitter (one call point at zero) must not resolve a shared
    # string count silently.
    monkeypatch.setattr(
        build_workload_module, "helper_emission_count",
        _patched_count(
            build_workload_module.helper_emission_count,
            {"router_mask_notequal": 0}))
    with pytest.raises(ValueError, match="shared per-layer count"):
        _ep2_graph("ascend_910b")
