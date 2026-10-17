from dataclasses import replace
import importlib

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.predictors.common import graph_replayed_launch
from src.kernel.kernel_spec import KernelKind
from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.launch_config.select_launch_config import (
    normalize_flash_attn_execution_attributes,
    select_launch_config,
)
from src.serving.deployment.af_disagg import build_af_disagg_plan
from src.serving.deployment.colocated import build_colocated_plan
from src.serving.deployment.deployment_config import (
    CollectiveConfig,
    DeploymentConfig,
    PipelineP2PConfig,
    RouteConfig,
    WorkerPoolConfig,
)
from src.serving.deployment.deployment_plan import build_deployment_plan
from src.serving.parallelism.parallel_config import ParallelConfig
from src.serving.kernel_lowering.lower_workload import lower_workload
from src.serving.kv_cache import KVCacheConfig, KVCacheTierConfig
from src.serving.kv_cache.kv_cache_events import KVCacheAction, KVCacheEvent
from src.serving.schedule.schedule_step import ScheduleStep, ScheduledRequest
from src.serving.workload.build_workload import build_workload
from src.serving.workload.graph_padding import (
    GraphExecutionConfig,
    GraphTokenShape,
)
from src.serving.workload.model_config import load_model_config
from src.serving.workload.runtime_inputs import (
    ExpertRoutingInput,
    RuntimeInputs,
)
from src.serving.workload.workload_graph import (
    WorkloadEdge,
    WorkloadGraph,
    WorkloadKind,
    WorkloadNode,
)


_CAPTURE_SIZES = (1, 2, 4, *range(8, 513, 8))
_PIECEWISE_EAGER_KINDS = ("flash_attn", "kv_update")


def _graph_config(
    configured_mode="FULL",
    capture_sizes=_CAPTURE_SIZES,
):
    return GraphExecutionConfig(
        configured_mode,
        capture_sizes,
        (
            _PIECEWISE_EAGER_KINDS
            if "PIECEWISE" in configured_mode else ()
        ),
    )


def _llama_case(hardware_id="ascend_910b"):
    model = load_model_config("tests/serving/fixtures/llama_scheduler")
    deployment = build_colocated_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", hardware_id),
        )),
        num_layers=model.num_hidden_layers,
    )
    step = ScheduleStep(
        "vllm:120",
        "main",
        (
            ScheduledRequest("decode", "decode", 0, 1, 32),
            ScheduledRequest("prefill", "prefill", 10, 0, 0),
        ),
        attention_kv_layout="paged_varlen",
    )
    return model, deployment, step


def _runtime(graph_execution=None):
    if graph_execution is None:
        return RuntimeInputs(
            ExpertRoutingInput.not_applicable(),
            "paged_varlen",
            "uniform",
        )
    return RuntimeInputs(
        ExpertRoutingInput.not_applicable(),
        "paged_varlen",
        "uniform",
        graph_execution,
    )


def _target_padded_decode(fixture):
    model = load_model_config(fixture)
    deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    return build_workload(
        ScheduleStep(
            "target-graph-padding",
            "main",
            tuple(
                ScheduledRequest(f"decode-{index}", "decode", 0, 1, 32)
                for index in range(3)),
            attention_kv_layout="mla",
        ),
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption(
                "test_seeded_routing_v1", 0),
            "mla",
            "uniform",
            _graph_config("FULL", (4, 8)),
        ),
    )


def test_dsa_target_model_fails_loud_on_graph_padding():
    """DSA indexer nodes have no padded-row contract; never drop them."""

    with pytest.raises(
            NotImplementedError, match="cannot represent CUDA-graph padding"):
        _target_padded_decode("tests/serving/fixtures/glm_5_2_official")


def test_mla_target_model_emits_padded_rows_and_decode_domain():
    graph = _target_padded_decode("tests/serving/fixtures/kimi_k2_6_official")
    by_suffix = {node.node_id.split(":", 1)[1]: node for node in graph.nodes}
    norm = by_suffix["layer-0:input-norm"]
    assert norm.shape["rows"] == 4
    assert norm.attributes["graph_logical_tokens"] == 3
    assert norm.attributes["graph_model_input_tokens"] == 4
    assert norm.attributes["graph_runtime_mode"] == "FULL"
    # The FULL-graph MLA op runs its absorption bmm on the padded rows.
    q_down = by_suffix["layer-0:decode-q-down-proj"]
    assert q_down.shape["m"] == 4
    # Attention keeps the logical requests plus the replay domain; padded
    # requests (query 0, seq_len 0) carry no attention work.
    attention = by_suffix["layer-0:decode-mla-attention"]
    assert attention.shape["batch"] == 3
    assert attention.attributes["graph_model_input_tokens"] == 4
    # lm_head consumes hidden_states[logits_indices] outside the graph.
    lm_head = by_suffix["lm-head"]
    assert lm_head.shape["m"] == 3
    assert "graph_runtime_mode" not in lm_head.attributes


def test_graph_dispatch_resolves_bucket_hit_padding_and_fallback():
    config = _graph_config()

    padded = config.resolve(11, uniform_decode=False)
    exact = config.resolve(16, uniform_decode=True)
    fallback = config.resolve(513, uniform_decode=False)
    eager_747 = config.resolve(747, uniform_decode=False)

    assert padded.runtime_mode == "FULL"
    assert padded.logical_tokens == 11
    assert padded.model_input_tokens == 16
    assert padded.attention_active_tokens == 11
    assert padded.kv_write_tokens == 11
    assert exact.model_input_tokens == 16
    assert exact.padding_tokens == 0
    assert fallback.runtime_mode == "NONE"
    assert fallback.model_input_tokens == 513
    assert eager_747.runtime_mode == "NONE"
    assert eager_747.logical_tokens == eager_747.model_input_tokens == 747
    assert eager_747.padding_tokens == 0


def test_graph_attention_domains_expand_only_the_captured_decode_slots():
    config = GraphExecutionConfig(
        "FULL_AND_PIECEWISE", (40, 72, 512), _PIECEWISE_EAGER_KINDS)
    a800_shape = config.resolve(66, uniform_decode=True)
    c500_shape = config.resolve(36, uniform_decode=True)
    eager_shape = config.resolve(747, uniform_decode=False)

    a800_attention = a800_shape.attention_shape(
        (1,) * 66,
        (512,) * 66,
        uniform_decode_query_len=1,
    )
    c500_attention = c500_shape.attention_shape(
        (1,) * 36,
        (955,) * 36,
        uniform_decode_query_len=1,
    )
    eager_attention = eager_shape.attention_shape(
        (747,),
        (747,),
        uniform_decode_query_len=1,
    )

    assert a800_shape.runtime_mode == "FULL"
    assert a800_attention.logical_tokens == 66
    assert a800_attention.model_input_tokens == 72
    assert a800_attention.logical_q_lens == (1,) * 66
    assert a800_attention.model_input_q_lens == (1,) * 72
    assert a800_attention.logical_kv_lens == (512,) * 66
    assert a800_attention.model_input_kv_lens == (512,) * 66 + (1,) * 6

    assert c500_shape.runtime_mode == "FULL"
    assert c500_attention.logical_tokens == 36
    assert c500_attention.model_input_tokens == 40
    assert c500_attention.logical_q_lens == (1,) * 36
    assert c500_attention.model_input_q_lens == (1,) * 40
    assert c500_attention.logical_kv_lens == (955,) * 36
    assert c500_attention.model_input_kv_lens == (955,) * 36 + (1,) * 4

    assert eager_shape.runtime_mode == "NONE"
    assert eager_attention.logical_q_lens == eager_attention.model_input_q_lens
    assert eager_attention.logical_kv_lens == eager_attention.model_input_kv_lens


def test_graph_attention_domains_pad_piecewise_tensor_extent_not_active_work():
    token_shape = GraphExecutionConfig(
        "FULL_AND_PIECEWISE", (16,), _PIECEWISE_EAGER_KINDS,
    ).resolve(11, uniform_decode=False)
    attention = token_shape.attention_shape(
        (1, 10),
        (33, 10),
        uniform_decode_query_len=1,
    )

    assert token_shape.runtime_mode == "PIECEWISE"
    assert attention.logical_q_lens == (1, 10)
    assert attention.logical_kv_lens == (33, 10)
    assert attention.model_input_q_lens == (1, 15)
    assert attention.model_input_kv_lens == (33, 15)


def _graph_attention_node(
    node_id,
    *,
    logical_q_lens,
    logical_kv_lens,
    model_input_q_lens,
    model_input_kv_lens,
    query_heads,
    kv_heads,
):
    return WorkloadNode(
        node_id,
        WorkloadKind.FLASH_ATTN,
        "main",
        {
            "batch": len(model_input_q_lens),
            "q_len": max(model_input_q_lens),
            "kv_len": max(model_input_kv_lens),
            "head_dim": 128,
            "num_query_heads": query_heads,
            "num_kv_heads": kv_heads,
        },
        "bf16",
        "decode",
        rank=0,
        attributes={
            "causal": True,
            "kv_layout": "paged_varlen",
            "kv_dist": "uniform",
            "q_lens": logical_q_lens,
            "kv_lens": logical_kv_lens,
            "model_input_q_lens": model_input_q_lens,
            "model_input_kv_lens": model_input_kv_lens,
            "graph_runtime_mode": "FULL",
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
            "launch_policy": "injected_launch",
            "split_policy": "serving",
        },
    )


def test_a800_selected_graph_attention_uses_72_slots_but_66_active_rows():
    attention = _graph_attention_node(
        "a800-attention",
        logical_q_lens=(1,) * 66,
        logical_kv_lens=(512,) * 66,
        model_input_q_lens=(1,) * 72,
        model_input_kv_lens=(512,) * 66 + (1,) * 6,
        query_heads=16,
        kv_heads=4,
    )
    deployment = build_colocated_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "nvidia_a800"),
        )),
        num_layers=1,
    )
    plan = lower_workload(
        WorkloadGraph((attention,)),
        deployment,
        {"nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml")},
    )
    main, = tuple(
        call.kernel_spec for call in plan.calls
        if call.kernel_spec.kind is KernelKind.FLASH_ATTN
        and call.kernel_spec.variant == "decode")

    assert main.launch.grid == (1, 72, 4)
    assert main.shape["batch"] == 72
    assert main.attributes["q_lens"] == (1,) * 66
    assert main.attributes["kv_lens"] == (512,) * 66
    assert main.attributes["model_input_q_lens"] == (1,) * 72
    assert main.attributes["model_input_kv_lens"] == (512,) * 66 + (1,) * 6
    assert predict(main, HWSpec("configs/hardware/nvidia_a800.yaml")).t_total_ns > 0


def test_c500_selected_graph_attention_uses_40_slots_but_36_active_rows():
    logical_kv_lens = (
        793, 699, 551, 644, 607, 453, 361, 382, 320, 309, 304, 955,
        289, 479, 237, 227, 651, 356, 310, 188, 230, 179, 170, 176,
        157, 104, 416, 80, 831, 130, 64, 104, 80, 60, 68, 760,
    )
    attention = _graph_attention_node(
        "c500-attention",
        logical_q_lens=(1,) * 36,
        logical_kv_lens=logical_kv_lens,
        model_input_q_lens=(1,) * 40,
        model_input_kv_lens=logical_kv_lens + (1,) * 4,
        query_heads=16,
        kv_heads=2,
    )
    deployment = build_colocated_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "metax_c500"),
        )),
        num_layers=1,
    )
    plan = lower_workload(
        WorkloadGraph((attention,)),
        deployment,
        {"metax_c500": HWSpec("configs/hardware/metax_c500.yaml")},
    )
    main, combine = tuple(
        call.kernel_spec for call in plan.calls
        if call.kernel_spec.kind is KernelKind.FLASH_ATTN)

    assert main.launch.grid == (1, 80, 9)
    assert combine.launch.grid == (80, 1, 1)
    assert main.shape["batch"] == 40
    assert main.attributes["q_lens"] == (1,) * 36
    assert main.attributes["kv_lens"] == logical_kv_lens
    assert main.attributes["model_input_q_lens"] == (1,) * 40
    assert main.attributes["model_input_kv_lens"] == logical_kv_lens + (1,) * 4
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    assert predict(main, hardware).t_total_ns > 0
    assert predict(combine, hardware).t_total_ns > 0


def test_910b_padded_decode_does_not_reuse_real_slot_worker_schedule():
    logical_kv_lens = (256,) * 33
    attention = _graph_attention_node(
        "ascend-padded-attention",
        logical_q_lens=(1,) * 33,
        logical_kv_lens=logical_kv_lens,
        model_input_q_lens=(1,) * 40,
        model_input_kv_lens=logical_kv_lens + (1,) * 7,
        query_heads=32,
        kv_heads=4,
    )
    deployment = build_colocated_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "ascend_910b"),
        )),
        num_layers=1,
    )
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    plan = lower_workload(
        WorkloadGraph((attention,)), deployment, {"ascend_910b": hardware})
    main, = tuple(
        call.kernel_spec for call in plan.calls
        if call.kernel_spec.kind is KernelKind.FLASH_ATTN)

    assert "cann_source_task_count" in main.launch.parameters
    assert main.launch.source_program is not None
    prediction = predict(main, hardware)
    assert prediction.t_total_ns > 0


@pytest.mark.parametrize(
    ("copy_plan", "expected_elements"),
    (
        (("physical",), (4 * 4 * 128,)),
        (("logical", "physical"), (2 * 4 * 128, 4 * 4 * 128)),
    ),
)
def test_910b_attention_output_copy_plan_uses_source_row_domains(
        copy_plan, expected_elements):
    """Materialize the source FIA TensorMoves with logical/physical rows."""

    node = _graph_attention_node(
        "ascend-output-copies",
        logical_q_lens=(1, 1),
        logical_kv_lens=(8, 9),
        model_input_q_lens=(1, 1, 1, 1),
        model_input_kv_lens=(8, 9, 1, 1),
        query_heads=4,
        kv_heads=2,
    )
    node = replace(
        node,
        attributes={
            **node.attributes,
            "launch_policy": "analytical_launch",
            "source_output_copy_plan": copy_plan,
        },
    )
    deployment = build_colocated_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "ascend_910b"),
        )),
        num_layers=1,
    )
    plan = lower_workload(
        WorkloadGraph((node,)), deployment,
        {"ascend_910b": HWSpec("configs/hardware/ascend_910b.yaml")},
    )
    copies = tuple(
        call.kernel_spec for call in plan.calls
        if call.kernel_spec.kind is KernelKind.VECTOR
        and call.kernel_spec.attributes["operation"] == "copy"
    )
    assert tuple(spec.shape["element_count"] for spec in copies) == (
        expected_elements)


@pytest.mark.parametrize("hardware_id", ("nvidia_a800", "metax_c500"))
def test_graph_padding_fa_copy_uses_physical_execution_abi(
        hardware_id, monkeypatch):
    """Keep every FA helper on captured slots, not logical request lengths."""

    node = _graph_attention_node(
        f"{hardware_id}-copy-domain",
        logical_q_lens=(1, 1),
        logical_kv_lens=(8, 9),
        model_input_q_lens=(1, 1, 1, 1),
        model_input_kv_lens=(8, 9, 1, 1),
        query_heads=4,
        kv_heads=2,
    )
    node = replace(
        node,
        attributes={
            **node.attributes,
            "launch_policy": "analytical_launch",
        },
    )
    deployment = build_colocated_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", hardware_id),
        )),
        num_layers=1,
    )
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    lowering_module = importlib.import_module(
        "src.serving.kernel_lowering.lower_workload")
    original_selector = lowering_module.select_flash_attn_copy_launches
    selected_attributes = []

    def copy_selector(
            selected_hardware_id, shape, attributes, selected_hardware):
        assert selected_hardware_id == hardware_id
        assert {
            name: value for name, value in shape.items()
            if name != "dtype_bytes"
        } == dict(node.shape)
        assert shape["dtype_bytes"] == 2
        assert selected_hardware is hardware
        selected_attributes.append(dict(attributes))
        return original_selector(
            selected_hardware_id, shape, attributes, selected_hardware)

    monkeypatch.setattr(
        lowering_module, "select_flash_attn_copy_launches", copy_selector)
    plan = lower_workload(
        WorkloadGraph((node,)), deployment, {hardware_id: hardware})

    assert selected_attributes
    assert all(
        attributes["q_lens"] == node.attributes["model_input_q_lens"]
        and attributes["kv_lens"] == node.attributes["model_input_kv_lens"]
        and attributes["is_decode"] is True
        for attributes in selected_attributes
    )
    assert any(call.kernel_spec.kind in (KernelKind.VECTOR, KernelKind.MEMCPY)
               for call in plan.calls)


def test_graph_padding_propagates_only_across_source_proven_boundaries():
    model, deployment, step = _llama_case()
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(_graph_config()),
    )

    assert graph.node("vllm:120:token-embedding").shape["rows"] == 16
    assert graph.node("vllm:120:layer-0:input-norm").shape["rows"] == 16
    assert graph.node("vllm:120:layer-0:qkv").shape["m"] == 16
    assert graph.node("vllm:120:layer-0:rotary").shape["seq_len"] == 16
    assert graph.node("vllm:120:layer-0:kv-update").shape["seq_len"] == 11
    attention = graph.node("vllm:120:layer-0:attention")
    assert attention.attributes["q_lens"] == (1, 10)
    assert attention.attributes["kv_lens"] == (33, 10)
    assert attention.attributes["graph_runtime_mode"] == "FULL"
    assert graph.node("vllm:120:layer-0:o-proj").shape["m"] == 16
    assert graph.node("vllm:120:final-norm").shape["rows"] == 16
    assert graph.node("vllm:120:lm-head").shape["m"] == 2

    plan = lower_workload(
        graph,
        deployment,
        {"ascend_910b": HWSpec("configs/hardware/ascend_910b.yaml")},
    )
    qkv = next(
        call.kernel_spec for call in plan.calls
        if call.call_id == "vllm:120:layer-0:qkv")
    kv_update = next(
        call.kernel_spec for call in plan.calls
        if call.call_id == "vllm:120:layer-0:kv-update")
    lm_head = next(
        call.kernel_spec for call in plan.calls
        if call.call_id == "vllm:120:lm-head")
    assert qkv.kind is KernelKind.GEMM
    assert qkv.shape["m"] == 11
    assert qkv.attributes["model_input_shape"]["m"] == 16
    assert kv_update.kind is KernelKind.KV_CACHE
    assert kv_update.shape["seq_len"] == 11
    assert lm_head.kind is KernelKind.GEMM
    assert lm_head.shape["m"] == 2
    assert all(
        predict(call.kernel_spec, HWSpec("configs/hardware/ascend_910b.yaml"))
        .t_total_ns > 0
        for call in plan.calls
    )


@pytest.mark.parametrize(
    ("hardware_id", "configured_mode", "expected_runtime_mode"),
    (
        ("nvidia_a800", "FULL", "FULL"),
        ("nvidia_a800", "FULL_AND_PIECEWISE", "PIECEWISE"),
        ("metax_c500", "FULL", "FULL"),
        ("metax_c500", "FULL_AND_PIECEWISE", "PIECEWISE"),
        ("ascend_910b", "FULL", "FULL"),
        ("ascend_910b", "FULL_AND_PIECEWISE", "PIECEWISE"),
    ),
)
def test_graph_padding_dense_plan_predictions_keep_work_logical(
    hardware_id,
    configured_mode,
    expected_runtime_mode,
):
    model, deployment, step = _llama_case(hardware_id)
    if hardware_id == "metax_c500":
        # The C500 standard-GQA FA selector is source-backed only for D=128.
        model = replace(
            model,
            hidden_size=256,
            intermediate_size=512,
            head_dim=128,
        )
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(GraphExecutionConfig(
            configured_mode,
            (16,),
            (_PIECEWISE_EAGER_KINDS
             if "PIECEWISE" in configured_mode else ()),
        )),
    )
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    plan = lower_workload(graph, deployment, {hardware_id: hardware})

    def spec_for(node_suffix, kind):
        node_id = f"vllm:120:{node_suffix}"
        matches = tuple(
            call.kernel_spec for call in plan.calls
            if call.source_node_ids == (node_id,)
            and call.kernel_spec.kind is kind)
        assert len(matches) == 1
        return matches[0]

    def assert_domain(spec):
        assert spec.attributes["graph_logical_tokens"] == 11
        assert spec.attributes["graph_model_input_tokens"] == 16
        assert spec.attributes["graph_runtime_mode"] == expected_runtime_mode
        assert all(
            spec.attributes["model_input_shape"][name] >= value
            for name, value in spec.shape.items()
            if name in spec.attributes["model_input_shape"])

    embedding = spec_for("token-embedding", KernelKind.VECTOR)
    assert_domain(embedding)
    assert embedding.shape["element_count"] == 11 * model.hidden_size
    assert embedding.attributes["model_input_shape"]["element_count"] == (
        16 * model.hidden_size)
    assert embedding.attributes["index_count"] == 11

    input_norm = spec_for("layer-0:input-norm", KernelKind.VECTOR)
    assert_domain(input_norm)
    assert input_norm.shape["rows"] == 11
    assert input_norm.attributes["model_input_shape"]["rows"] == 16

    qkv = spec_for("layer-0:qkv", KernelKind.GEMM)
    o_proj = spec_for("layer-0:o-proj", KernelKind.GEMM)
    gate_up = spec_for("layer-0:gate-up", KernelKind.GEMM)
    down = spec_for("layer-0:down", KernelKind.GEMM)
    for spec in (qkv, o_proj, gate_up, down):
        assert_domain(spec)
        assert spec.shape["m"] == 11
        assert spec.attributes["model_input_shape"]["m"] == 16

    rotary = spec_for("layer-0:rotary", KernelKind.VECTOR)
    assert_domain(rotary)
    assert rotary.shape["seq_len"] == 11
    assert rotary.attributes["model_input_shape"]["seq_len"] == 16

    kv_update = spec_for("layer-0:kv-update", KernelKind.KV_CACHE)
    assert kv_update.attributes["graph_logical_tokens"] == 11
    assert kv_update.attributes["graph_model_input_tokens"] == 11
    assert kv_update.shape["seq_len"] == 11
    assert kv_update.attributes["model_input_shape"]["seq_len"] == 11

    activation = spec_for("layer-0:activation", KernelKind.VECTOR)
    assert_domain(activation)
    assert activation.shape["tok"] == 11
    assert activation.attributes["model_input_shape"]["tok"] == 16

    if hardware_id == "metax_c500":
        # vLLM-MetaX split forward: one pathway per request class, each on
        # the request ABI the backend call sees (padded only inside FULL).
        attention_specs = tuple(
            call.kernel_spec for call in plan.calls
            if call.source_node_ids == ("vllm:120:layer-0:attention",)
            and call.kernel_spec.kind is KernelKind.FLASH_ATTN
            and call.kernel_spec.variant in ("prefill", "decode"))
        prefill_rows = 15 if expected_runtime_mode == "FULL" else 10
        assert sorted(
            (spec.variant, spec.attributes["q_lens"],
             spec.attributes["kv_lens"])
            for spec in attention_specs) == [
                ("decode", (1,), (33,)),
                ("prefill", (prefill_rows,), (prefill_rows,)),
            ]
    else:
        attention = spec_for("layer-0:attention", KernelKind.FLASH_ATTN)
        assert attention.attributes["q_lens"] == (1, 10)
        assert attention.attributes["kv_lens"] == (33, 10)
        assert attention.attributes["model_input_q_lens"] == (1, 15)
        assert attention.attributes["model_input_kv_lens"] == (33, 15)
        assert attention.attributes["graph_runtime_mode"] == (
            expected_runtime_mode)

    final_norm = spec_for("final-norm", KernelKind.VECTOR)
    assert_domain(final_norm)
    assert final_norm.shape["rows"] == 11
    lm_head = spec_for("lm-head", KernelKind.GEMM)
    assert lm_head.shape["m"] == 2
    assert "graph_logical_tokens" not in lm_head.attributes
    assert all(predict(call.kernel_spec, hardware).t_total_ns > 0
               for call in plan.calls)


@pytest.mark.parametrize(
    "hardware_id",
    ("nvidia_a800", "metax_c500", "ascend_910b"),
)
def test_graph_padding_dense_launches_reselect_from_physical_abi(hardware_id):
    """Every dense graph launch must be reproducible from captured inputs."""

    model, deployment, step = _llama_case(hardware_id)
    if hardware_id == "metax_c500":
        model = replace(
            model,
            hidden_size=256,
            intermediate_size=512,
            head_dim=128,
        )
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(GraphExecutionConfig(
            "FULL_AND_PIECEWISE", (16,), _PIECEWISE_EAGER_KINDS)),
    )
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    plan = lower_workload(graph, deployment, {hardware_id: hardware})

    def vector_shape(node):
        shape = dict(node.shape)
        shape["element_count"] = shape.pop("elements")
        return shape

    def node_shape(node):
        return dict(node.shape)

    def spec_for(node_suffix, kind):
        node_id = f"vllm:120:{node_suffix}"
        matches = tuple(
            call.kernel_spec for call in plan.calls
            if (call.source_node_ids == (node_id,)
                and call.kernel_spec.kind is kind))
        assert len(matches) == 1
        return graph.node(node_id), matches[0]

    selector_cases = (
        ("token-embedding", KernelKind.VECTOR, vector_shape, "gather"),
        ("layer-0:input-norm", KernelKind.VECTOR, node_shape, None),
        ("layer-0:qkv", KernelKind.GEMM, node_shape, None),
        ("layer-0:rotary", KernelKind.VECTOR, vector_shape, None),
        ("layer-0:kv-update", KernelKind.KV_CACHE, vector_shape,
         "paged_kv_cache"),
        ("layer-0:o-proj", KernelKind.GEMM, node_shape, None),
        ("layer-0:post-attn-norm", KernelKind.VECTOR, node_shape,
         None),
        ("layer-0:gate-up", KernelKind.GEMM, node_shape, None),
        ("layer-0:activation", KernelKind.VECTOR, vector_shape, None),
        ("layer-0:down", KernelKind.GEMM, node_shape, None),
        ("final-norm", KernelKind.VECTOR, node_shape, None),
    )
    for suffix, kind, physical_shape, variant in selector_cases:
        node, spec = spec_for(suffix, kind)
        if variant is None:
            variant = node.variant
        expected_launch = select_launch_config(
            hardware_id,
            kind,
            physical_shape(node),
            node.dtype,
            variant,
            node.attributes,
            hardware=hardware,
        )
        if (hardware_id == "ascend_910b"
                and expected_launch.algorithm == "cann_matmul"):
            # KernelSpec normalizes this CANN algorithm identity at its
            # boundary; grid, tiling, source program, and parameters remain
            # the exact selector result.
            expected_launch = replace(
                expected_launch, algorithm="ascend_cann_matmul")
        assert spec.launch == expected_launch, suffix

    if hardware_id == "metax_c500":
        # The split forward reselects each pathway from its own requests:
        # PIECEWISE attention is unpadded, so these are the logical ones.
        attention = graph.node("vllm:120:layer-0:attention")
        attention_specs = {
            call.kernel_spec.variant: call.kernel_spec
            for call in plan.calls
            if call.source_node_ids == ("vllm:120:layer-0:attention",)
            and call.kernel_spec.kind is KernelKind.FLASH_ATTN
            and call.kernel_spec.variant in ("prefill", "decode")}
        for variant, role, q_lens, kv_lens in (
                ("prefill", "prefill", (10,), (10,)),
                ("decode", "decode", (1,), (33,))):
            part_attributes = {
                name: value for name, value in attention.attributes.items()
                if name not in ("model_input_q_lens", "model_input_kv_lens")}
            part_attributes.update({
                "q_lens": q_lens, "kv_lens": kv_lens,
                "flash_attn_forward_role": role})
            part_attributes = dict(normalize_flash_attn_execution_attributes(
                hardware_id, attention.shape, part_attributes))
            part_attributes["is_decode"] = variant == "decode"
            assert attention_specs[variant].launch == select_launch_config(
                hardware_id,
                KernelKind.FLASH_ATTN,
                {**attention.shape, "batch": 1, "q_len": max(q_lens),
                 "kv_len": max(kv_lens)},
                attention.dtype,
                variant,
                part_attributes,
                hardware=hardware,
            )
        return
    attention, attention_spec = spec_for("layer-0:attention",
                                         KernelKind.FLASH_ATTN)
    execution_attributes = dict(attention.attributes)
    execution_attributes["q_lens"] = tuple(
        execution_attributes["model_input_q_lens"])
    execution_attributes["kv_lens"] = tuple(
        execution_attributes["model_input_kv_lens"])
    execution_attributes = dict(normalize_flash_attn_execution_attributes(
        hardware_id, attention.shape, execution_attributes))
    execution_attributes["is_decode"] = attention.variant == "decode"
    assert attention_spec.launch == select_launch_config(
        hardware_id,
        KernelKind.FLASH_ATTN,
        attention.shape,
        attention.dtype,
        attention.variant,
        execution_attributes,
        hardware=hardware,
    )


@pytest.mark.parametrize("hardware_id", ("metax_c500", "ascend_910b"))
def test_graph_padding_split_k_reducer_reselects_physical_output(
        hardware_id, monkeypatch):
    """A split-K reducer must use the captured GEMM output, not active M."""

    node = WorkloadNode(
        "graph-gemm",
        WorkloadKind.LINEAR,
        "main",
        {"b": 1, "m": 16, "n": 128, "k": 128},
        "bf16",
        "gemm",
        rank=0,
        attributes={
            "has_bias": False,
            "position": "qkv_proj",
            "launch_policy": "analytical_launch",
            "graph_logical_tokens": 11,
            "graph_model_input_tokens": 16,
            "graph_runtime_mode": "PIECEWISE",
        },
    )
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    deployment = build_colocated_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", hardware_id),
        )),
        num_layers=1,
    )
    lowering_module = importlib.import_module(
        "src.serving.kernel_lowering.lower_workload")
    selected_shapes = []

    def main_selector(
            selected_hardware_id, kind, shape, dtype, variant, attributes,
            *, hardware):
        assert selected_hardware_id == hardware_id
        assert kind is KernelKind.GEMM
        assert shape == node.shape
        assert dtype == node.dtype
        assert variant == node.variant
        assert attributes == node.attributes
        assert hardware is not None
        return LaunchConfig(
            algorithm="cublas_analytical_gemm",
            grid=(1, 1, 2),
            block=(256, 1, 1),
            tiling=TilingSpec(
                tile_shape={"m": 32, "n": 128, "k": 64},
                pipeline_depth=2,
                split_k=2,
            ),
            parameters={
                "compulsory_traffic": 0.0,
                "weight_tile_rows": 0.0,
                "mte2_tile_bytes": 0.0,
                "aiv_mixed": 0.0,
                "n_major_traffic": 0.0,
                "split_k_partial_bytes_per_element": 2.0,
                "split_k_partial_dram_bytes_per_element": 0.0,
            },
            execution_variant="gemm",
        )

    def reducer_selector(
            selected_hardware_id, shape, dtype, variant, attributes,
            *, hardware):
        assert selected_hardware_id == hardware_id
        assert dtype == node.dtype
        assert variant == node.variant
        assert attributes == node.attributes
        assert hardware is not None
        selected_shapes.append(dict(shape))
        return LaunchConfig(
            algorithm="logical_gemm_aux:gemm_reduce",
            grid=(8, 1, 1),
            block=(256, 1, 1),
            parameters={
                "split_k_partial_bytes_per_element": 2.0,
                "split_k_partial_dram_bytes_per_element": 0.0,
            },
            execution_variant="gemm_reduce",
        )

    monkeypatch.setattr(lowering_module, "select_launch_config", main_selector)
    monkeypatch.setattr(
        lowering_module, "select_gemm_reducer_launch", reducer_selector)
    plan = lower_workload(
        WorkloadGraph((node,)), deployment, {hardware_id: hardware})

    assert selected_shapes == [dict(node.shape)]
    main, reducer = tuple(call.kernel_spec for call in plan.calls)
    assert main.shape["m"] == 11
    assert main.attributes["model_input_shape"]["m"] == 16
    assert reducer.kind is KernelKind.VECTOR
    assert reducer.shape["element_count"] == 16 * 128
    assert reducer.attributes == {
        "split_k": 2,
        "graph_logical_tokens": 11,
        "graph_model_input_tokens": 16,
        "graph_runtime_mode": "PIECEWISE",
        "model_input_shape": {"element_count": 16 * 128},
    }


def test_graph_padding_norm_materialization_reselects_physical_copy():
    """Strided Q/K materialization remains a physical captured-tensor copy."""

    physical_tokens = 8
    logical_tokens = 5
    rows_per_group = 2
    hidden = 128
    node = WorkloadNode(
        "graph-q-norm",
        WorkloadKind.NORM,
        "main",
        {"rows": physical_tokens * rows_per_group, "hidden": hidden},
        "bf16",
        "rmsnorm",
        rank=0,
        attributes={
            "launch_policy": "analytical_launch",
            "input_materialization": "grouped_strided_to_contiguous",
            "input_outer_groups": physical_tokens,
            "input_rows_per_group": rows_per_group,
            "input_group_stride_elements": 4 * hidden,
            "input_row_stride_elements": hidden,
            "graph_logical_tokens": logical_tokens,
            "graph_model_input_tokens": physical_tokens,
            "graph_runtime_mode": "FULL",
        },
    )
    hardware_id = "nvidia_a800"
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    deployment = build_colocated_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", hardware_id),
        )),
        num_layers=1,
    )
    plan = lower_workload(
        WorkloadGraph((node,)), deployment, {hardware_id: hardware})
    copy, norm = tuple(call.kernel_spec for call in plan.calls)
    copy_attributes = {
        "operation": "copy",
        "reads": 1,
        "writes": 1,
        "launch_policy": node.attributes["launch_policy"],
        "input_materialization": node.attributes["input_materialization"],
        "input_outer_groups": physical_tokens,
        "input_rows_per_group": rows_per_group,
        "input_group_stride_elements": 4 * hidden,
        "input_row_stride_elements": hidden,
    }
    physical_copy_shape = {"element_count": physical_tokens * rows_per_group
                           * hidden}
    assert copy.shape["element_count"] == physical_copy_shape["element_count"]
    assert copy.launch == select_launch_config(
        hardware_id,
        KernelKind.VECTOR,
        physical_copy_shape,
        node.dtype,
        "elementwise",
        copy_attributes,
        hardware=hardware,
    )
    assert copy.attributes["graph_logical_tokens"] == logical_tokens
    assert copy.attributes["graph_model_input_tokens"] == physical_tokens
    assert copy.attributes["graph_runtime_mode"] == "FULL"
    assert copy.attributes["model_input_shape"] == physical_copy_shape
    assert norm.shape["rows"] == logical_tokens * rows_per_group
    assert norm.attributes["model_input_shape"]["rows"] == (
        physical_tokens * rows_per_group)
    stripped_copy = replace(
        copy,
        attributes={
            name: value
            for name, value in copy.attributes.items()
            if name not in (
                "graph_logical_tokens", "graph_model_input_tokens",
                "graph_runtime_mode", "model_input_shape",
            )
        },
    )
    assert predict(copy, hardware).t_total_ns == (
        predict(stripped_copy, hardware).t_total_ns)


def test_graph_padding_910b_qk_norm_rope_uses_logical_work():
    logical_tokens = 11
    model_input_tokens = 16
    query_heads = 2
    kv_heads = 1
    head_dim = 128
    domain = {
        "graph_logical_tokens": logical_tokens,
        "graph_model_input_tokens": model_input_tokens,
        "graph_runtime_mode": "PIECEWISE",
    }
    grouped_attributes = {
        "launch_policy": "analytical_launch",
        "input_materialization": "grouped_strided_to_contiguous",
        "input_outer_groups": model_input_tokens,
        "input_group_stride_elements": (
            (query_heads + 2 * kv_heads) * head_dim),
        "input_row_stride_elements": head_dim,
        **domain,
    }
    q_norm = WorkloadNode(
        "q-norm",
        WorkloadKind.NORM,
        "main",
        {"rows": model_input_tokens * query_heads, "hidden": head_dim},
        "bf16",
        "rmsnorm",
        rank=0,
        attributes={
            **grouped_attributes,
            "input_rows_per_group": query_heads,
        },
    )
    k_norm = WorkloadNode(
        "k-norm",
        WorkloadKind.NORM,
        "main",
        {"rows": model_input_tokens * kv_heads, "hidden": head_dim},
        "bf16",
        "rmsnorm",
        rank=0,
        attributes={
            **grouped_attributes,
            "input_rows_per_group": kv_heads,
        },
    )
    rotary = WorkloadNode(
        "rotary",
        WorkloadKind.ROTARY,
        "main",
        {
            "elements": model_input_tokens * (query_heads + kv_heads)
            * head_dim,
            "seq_len": model_input_tokens,
            "num_query_heads": query_heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        "bf16",
        "rotary",
        rank=0,
        attributes={
            "launch_policy": "analytical_launch",
            # §2.1 class1b: the vendor dispatch layer launches one extra V
            # materialization copy per attention layer in prefill; the count
            # comes from the versioned emission observation.
            "qkv_v_materialization_count": 1,
            **domain,
        },
    )
    graph = WorkloadGraph(
        (q_norm, k_norm, rotary),
        (
            WorkloadEdge("q-norm", "k-norm", "data"),
            WorkloadEdge("k-norm", "rotary", "data"),
        ),
    )
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    deployment = build_colocated_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "ascend_910b"),
        )),
        num_layers=1,
    )

    plan = lower_workload(graph, deployment, {"ascend_910b": hardware})

    # Q/K Norm and RoPE stay separate programs (the fused
    # split_qkv_rmsnorm_rope signature is absent from every captured launch
    # observation), each preceded by its own input materialization copy.
    assert [call.call_id for call in plan.calls] == [
        "q-norm:0", "q-norm:1", "k-norm:0", "k-norm:1", "rotary"]
    assert [call.kernel_spec.variant for call in plan.calls] == [
        "elementwise", "rmsnorm", "elementwise", "rmsnorm", "rotary"]
    assert [call.dependencies for call in plan.calls[1:]] == [
        ("q-norm:0",), ("q-norm:1",), ("k-norm:0",), ("k-norm:1",)]
    assert [call.source_node_ids for call in plan.calls] == [
        ("q-norm",), ("q-norm",), ("k-norm",), ("k-norm",), ("rotary",)]

    copy_calls = [
        call for call in plan.calls
        if call.kernel_spec.variant == "elementwise"]
    # The materialization copies move the whole padded input: their element
    # counts stay on model_input_tokens.
    assert [
        (call.kernel_spec.shape["element_count"],
         call.kernel_spec.attributes["operation"])
        for call in copy_calls
    ] == [
        (model_input_tokens * query_heads * head_dim, "copy"),
        (model_input_tokens * kv_heads * head_dim, "copy"),
    ]
    for call in plan.calls:
        assert call.kernel_spec.attributes["graph_logical_tokens"] == (
            logical_tokens)
        assert predict(call.kernel_spec, hardware).t_total_ns > 0
    for call in copy_calls:
        assert call.kernel_spec.attributes["model_input_shape"] == {
            "element_count": call.kernel_spec.shape["element_count"]}

    # The priced work is logical while the recorded model input stays padded:
    # that is the graph-padding invariant, and it holds per program.
    norm_calls = [
        call for call in plan.calls
        if call.kernel_spec.variant == "rmsnorm"]
    assert [call.kernel_spec.shape["rows"] for call in norm_calls] == [
        logical_tokens * query_heads, logical_tokens * kv_heads]
    assert [
        call.kernel_spec.attributes["model_input_shape"]["rows"]
        for call in norm_calls
    ] == [model_input_tokens * query_heads, model_input_tokens * kv_heads]

    spec = plan.calls[-1].kernel_spec
    assert spec.variant == "rotary"
    assert spec.shape["seq_len"] == logical_tokens
    assert spec.shape["element_count"] == (
        logical_tokens * (query_heads + kv_heads) * head_dim)
    assert spec.attributes["model_input_shape"] == {
        "seq_len": model_input_tokens,
        "num_query_heads": query_heads,
        "num_kv_heads": kv_heads,
        "head_dim": head_dim,
        "element_count": (
            model_input_tokens * (query_heads + kv_heads) * head_dim),
    }
    expected_launch = select_launch_config(
        "ascend_910b",
        KernelKind.VECTOR,
        spec.attributes["model_input_shape"],
        "bf16",
        "rotary",
        {"launch_policy": "analytical_launch"},
        hardware=hardware,
    )
    assert spec.launch == expected_launch
    assert predict(spec, hardware).t_total_ns > 0

    forged_rotary = replace(
        rotary,
        attributes={
            **rotary.attributes,
            "graph_model_input_tokens": logical_tokens,
        },
    )
    # A forged graph token domain on one node of the chain is still rejected;
    # with the nodes lowering separately the RoPE program fails closed on its
    # own physical extent rather than on the group's shared domain.
    with pytest.raises(
            ValueError, match="not an integral graph-token multiple"):
        lower_workload(
            WorkloadGraph(
                (q_norm, k_norm, forged_rotary),
                (
                    WorkloadEdge("q-norm", "k-norm", "data"),
                    WorkloadEdge("k-norm", "rotary", "data"),
                ),
            ),
            deployment,
            {"ascend_910b": hardware},
        )


def test_piecewise_graph_padding_keeps_launch_physical_and_work_logical():
    model, deployment, step = _llama_case()
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(GraphExecutionConfig(
            "FULL_AND_PIECEWISE", (16,), _PIECEWISE_EAGER_KINDS)),
    )
    attention = graph.node("vllm:120:layer-0:attention")
    assert attention.attributes["graph_runtime_mode"] == "PIECEWISE"
    assert attention.attributes["q_lens"] == (1, 10)
    assert attention.attributes["model_input_q_lens"] == (1, 15)

    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    plan = lower_workload(graph, deployment, {"ascend_910b": hardware})
    qkv = next(
        call.kernel_spec for call in plan.calls
        if call.call_id == "vllm:120:layer-0:qkv")
    assert qkv.shape["m"] == 11
    assert qkv.attributes["graph_runtime_mode"] == "PIECEWISE"
    assert qkv.attributes["model_input_shape"]["m"] == 16


def test_graph_token_domain_rejects_incomplete_or_shrinking_contracts():
    model, deployment, step = _llama_case()
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(_graph_config()),
    )
    qkv_node = graph.node("vllm:120:layer-0:qkv")
    incomplete_attributes = dict(qkv_node.attributes)
    del incomplete_attributes["graph_model_input_tokens"]
    incomplete_qkv = replace(qkv_node, attributes=incomplete_attributes)
    with pytest.raises(ValueError, match="incomplete graph token domains"):
        lower_workload(
            WorkloadGraph((incomplete_qkv,)),
            deployment,
            {"ascend_910b": HWSpec("configs/hardware/ascend_910b.yaml")},
        )

    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    plan = lower_workload(graph, deployment, {"ascend_910b": hardware})
    qkv = next(
        call.kernel_spec for call in plan.calls
        if call.call_id == "vllm:120:layer-0:qkv")
    shrinking_attributes = dict(qkv.attributes)
    model_input_shape = dict(shrinking_attributes["model_input_shape"])
    model_input_shape["m"] = 10
    shrinking_attributes["model_input_shape"] = model_input_shape
    with pytest.raises(ValueError, match="model_input_shape cannot shrink"):
        replace(qkv, attributes=shrinking_attributes)


def test_disabled_graph_execution_is_exactly_the_legacy_workload():
    model, deployment, step = _llama_case()
    default_graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(),
    )
    explicit_disabled_graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(GraphExecutionConfig.disabled()),
    )

    assert explicit_disabled_graph == default_graph
    assert default_graph.node("vllm:120:layer-0:qkv").shape["m"] == 11


def test_exact_bucket_workload_keeps_its_graph_domain():
    """An exact-size capture is still a replayed graph, not eager execution.

    ``bisect_left`` returns the bucket itself when the logical token count is
    one of the capture sizes, so ``padding_tokens == 0``.  That says the
    replayed tensor needs no padding rows; it does not say the runtime
    launched the call eagerly.  Every node that declares a graph domain when
    the same step is rounded up must still declare one here, with the two
    token extents equal, and every physical shape must equal the eager
    baseline's.
    """

    model, deployment, _ = _llama_case()
    exact_step = ScheduleStep(
        "exact-bucket",
        "main",
        (
            ScheduledRequest("decode", "decode", 0, 1, 32),
            ScheduledRequest("prefill", "prefill", 15, 0, 0),
        ),
        attention_kv_layout="paged_varlen",
    )
    # 16 logical tokens: a capture size, so the exact bucket pads by zero.
    assert sum(
        request.scheduled_tokens for request in exact_step.requests) == 16
    assert 16 in _CAPTURE_SIZES

    def build(runtime_inputs):
        return build_workload(
            exact_step,
            model,
            deployment,
            launch_policy="analytical_launch",
            runtime_inputs=runtime_inputs,
        )

    baseline = build(_runtime())
    exact_bucket = build(_runtime(_graph_config(capture_sizes=_CAPTURE_SIZES)))
    # The same step rounded up to 24 defines which nodes own a graph domain.
    padded = build(_runtime(_graph_config(capture_sizes=(1, 2, 4, 8, 24))))

    def domain_owners(graph):
        return {
            node.node_id for node in graph.nodes
            if "graph_logical_tokens" in node.attributes
        }

    assert domain_owners(exact_bucket) == domain_owners(padded)
    assert domain_owners(exact_bucket)
    assert not domain_owners(baseline)

    assert tuple(node.node_id for node in exact_bucket.nodes) == tuple(
        node.node_id for node in baseline.nodes)
    for graph_node, baseline_node in zip(
            exact_bucket.nodes, baseline.nodes, strict=True):
        graph_attributes = dict(graph_node.attributes)
        baseline_attributes = dict(baseline_node.attributes)
        if "graph_logical_tokens" in graph_attributes:
            logical = graph_attributes.pop("graph_logical_tokens")
            model_input = graph_attributes.pop("graph_model_input_tokens")
            # Exact capture: no padded rows, so the two extents coincide.
            assert logical == model_input == 16
            assert graph_attributes.pop("graph_runtime_mode") == "FULL"
            assert "graph_logical_tokens" not in baseline_attributes
        if graph_node.kind is WorkloadKind.FLASH_ATTN:
            # FlashAttention declares its runtime mode unconditionally.
            assert baseline_attributes.pop("graph_runtime_mode") == "NONE"
        assert "graph_runtime_mode" not in baseline_attributes
        assert graph_attributes == baseline_attributes
        # Physical shapes are unchanged: only the declared domain is added.
        assert replace(
            graph_node, attributes=baseline_node.attributes
        ) == baseline_node


def test_uncapturable_workload_remains_unchanged():
    model, deployment, fallback_step = _llama_case()
    fallback_baseline = build_workload(
        fallback_step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(),
    )
    fallback_graph_enabled = build_workload(
        fallback_step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(_graph_config(capture_sizes=(1, 2, 4, 8))),
    )
    assert fallback_graph_enabled == fallback_baseline


@pytest.mark.parametrize(
    "hardware_id",
    ("nvidia_a800", "metax_c500", "ascend_910b"),
)
def test_graph_dispatch_rule_is_hardware_independent(hardware_id):
    model, deployment, step = _llama_case(hardware_id)
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(_graph_config()),
    )
    assert graph.node("vllm:120:layer-0:qkv").shape["m"] == 16
    assert graph.node("vllm:120:layer-0:kv-update").shape["seq_len"] == 11
    attention = graph.node("vllm:120:layer-0:attention")
    assert attention.attributes["graph_runtime_mode"] == "FULL"


def test_graph_padding_keeps_moe_routing_logical_with_physical_rows():
    model, deployment, step = _llama_case()
    moe_model = replace(
        model,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=model.intermediate_size,
        num_dense_layers=0,
    )
    graph = build_workload(
        step,
        moe_model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption("test-routing-v1", 0),
            "paged_varlen",
            "uniform",
            _graph_config(),
        ),
    )

    router = graph.node("vllm:120:layer-0:router")
    moe = graph.node("vllm:120:layer-0:moe")
    assert router.shape["m"] == 16
    assert moe.shape["tokens"] == 16
    assert moe.attributes["graph_logical_tokens"] == 11
    assert moe.attributes["graph_model_input_tokens"] == 16
    assert sum(moe.attributes["tokens_per_expert"]) == 11

    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    plan = lower_workload(graph, deployment, {"ascend_910b": hardware})
    moe_specs = tuple(
        call.kernel_spec for call in plan.calls
        if call.call_id.startswith("vllm:120:layer-0:moe")
    )
    assert moe_specs
    assert all(spec.shape["tokens"] == 11 for spec in moe_specs)
    assert all(
        spec.attributes["model_input_shape"]["tokens"] == 16
        for spec in moe_specs
    )
    assert all(predict(spec, hardware).t_total_ns > 0 for spec in moe_specs)


@pytest.mark.parametrize(
    "hardware_id",
    ("nvidia_a800", "metax_c500", "ascend_910b"),
)
def test_piecewise_graph_padding_keeps_moe_and_router_work_logical(
        hardware_id):
    model, deployment, step = _llama_case(hardware_id)
    if hardware_id == "metax_c500":
        model = replace(
            model,
            hidden_size=256,
            intermediate_size=512,
            head_dim=128,
        )
    moe_model = replace(
        model,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=model.intermediate_size,
        num_dense_layers=0,
    )
    graph = build_workload(
        step,
        moe_model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption("test-routing-v1", 0),
            "paged_varlen",
            "uniform",
            GraphExecutionConfig(
                "FULL_AND_PIECEWISE", (16,), _PIECEWISE_EAGER_KINDS),
        ),
    )
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    plan = lower_workload(graph, deployment, {hardware_id: hardware})

    router = next(
        call.kernel_spec for call in plan.calls
        if call.source_node_ids == ("vllm:120:layer-0:router",)
        and call.kernel_spec.kind is KernelKind.GEMM)
    assert router.shape["m"] == 11
    assert router.attributes["model_input_shape"]["m"] == 16
    assert router.attributes["graph_runtime_mode"] == "PIECEWISE"

    moe_specs = tuple(
        call.kernel_spec for call in plan.calls
        if call.source_node_ids == ("vllm:120:layer-0:moe",))
    assert moe_specs
    assert all(spec.shape["tokens"] == 11 for spec in moe_specs)
    assert all(
        spec.attributes["model_input_shape"]["tokens"] == 16
        for spec in moe_specs)
    assert all(
        sum(spec.attributes["tokens_per_expert"]) == 11
        for spec in moe_specs
        if "tokens_per_expert" in spec.attributes)
    assert all(predict(call.kernel_spec, hardware).t_total_ns > 0
               for call in plan.calls)


@pytest.mark.parametrize(
    "hardware_id",
    ("nvidia_a800", "metax_c500", "ascend_910b"),
)
def test_graph_padding_moe_launches_reselect_from_physical_abi(hardware_id):
    """Router and every physical MoE helper retain captured launch inputs."""

    model, deployment, step = _llama_case(hardware_id)
    if hardware_id == "metax_c500":
        model = replace(
            model,
            hidden_size=256,
            intermediate_size=512,
            head_dim=128,
        )
    moe_model = replace(
        model,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=model.intermediate_size,
        num_dense_layers=0,
    )
    graph = build_workload(
        step,
        moe_model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption("test-routing-v1", 0),
            "paged_varlen",
            "uniform",
            GraphExecutionConfig(
                "FULL_AND_PIECEWISE", (16,), _PIECEWISE_EAGER_KINDS),
        ),
    )
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    plan = lower_workload(graph, deployment, {hardware_id: hardware})
    router_id = "vllm:120:layer-0:router"
    router = graph.node(router_id)
    router_spec, = tuple(
        call.kernel_spec for call in plan.calls
        if (call.source_node_ids == (router_id,)
            and call.kernel_spec.kind is KernelKind.GEMM))
    expected_router_launch = select_launch_config(
        hardware_id,
        KernelKind.GEMM,
        router.shape,
        router.dtype,
        router.variant,
        router.attributes,
        hardware=hardware,
    )
    if (hardware_id == "ascend_910b"
            and expected_router_launch.algorithm == "cann_matmul"):
        expected_router_launch = replace(
            expected_router_launch, algorithm="ascend_cann_matmul")
    assert router_spec.launch == expected_router_launch

    moe_id = "vllm:120:layer-0:moe"
    moe = graph.node(moe_id)
    selection_attributes = dict(moe.attributes)
    del selection_attributes["routing_provenance"]
    moe_specs = tuple(
        call.kernel_spec for call in plan.calls
        if call.source_node_ids == (moe_id,))
    assert moe_specs
    for spec in moe_specs:
        launch_attributes = dict(selection_attributes)
        for name, value in spec.attributes.items():
            if name in launch_attributes:
                assert launch_attributes[name] == value
            launch_attributes[name] = value
        assert spec.launch == select_launch_config(
            hardware_id,
            spec.kind,
            moe.shape,
            moe.dtype,
            spec.variant,
            launch_attributes,
            hardware=hardware,
        )


def test_graph_padding_propagates_physical_rows_across_tp2_ep2():
    model = load_model_config(
        "tests/serving/fixtures/qwen3_moe_ep2_scheduler")
    _, _, step = _llama_case()
    collectives = tuple(
        CollectiveConfig(
            operation,
            "ring",
            "nvlink",
            "ring",
            "main-nvlink-fabric",
            "analytical_roofline_v1",
        )
        for operation in ("all_reduce", "all_gather")
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
                collectives=collectives,
            ),
        )),
        num_layers=model.num_hidden_layers,
        num_experts=model.num_experts,
    )
    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        runtime_inputs=RuntimeInputs(
            ExpertRoutingInput.seeded_assumption("test-routing-v1", 0),
            "paged_varlen",
            "uniform",
            _graph_config(),
        ),
    )

    routers = tuple(
        node for node in graph.nodes if node.kind is WorkloadKind.ROUTER)
    experts = tuple(
        node for node in graph.nodes if node.kind is WorkloadKind.FUSED_MOE)
    assert all(node.shape["m"] == 16 for node in routers)
    assert all(node.shape["tokens"] == 16 for node in experts)
    assert all(
        sum(node.attributes["tokens_per_expert"])
        == 11 * model.num_experts_per_tok
        for node in experts
    )
    output_collective = graph.node(
        "vllm:120:layer-0:moe:ep-output-all-reduce")
    assert output_collective.attributes["bytes"] == (
        16 * model.hidden_size * model.dtype_bytes)
    assert output_collective.attributes["graph_logical_tokens"] == 11
    assert output_collective.attributes["graph_model_input_tokens"] == 16

    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    plan = lower_workload(graph, deployment, {"nvidia_a800": hardware})
    output_transfer = next(
        call.kernel_spec for call in plan.calls
        if call.call_id == "vllm:120:layer-0:moe:ep-output-all-reduce"
    )
    assert output_transfer.kind is KernelKind.COMMUNICATION
    assert output_transfer.transfer.bytes == (
        16 * model.hidden_size * model.dtype_bytes)


def test_graph_padding_preserves_domain_across_pp_and_af_transfers():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    graph_config = GraphExecutionConfig("FULL", (16,), ())

    pp_model = load_model_config("tests/serving/fixtures/qwen3_pp2_scheduler")
    pp_deployment = build_deployment_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main",
                "colocated",
                "nvidia_a800",
                ParallelConfig(pipeline_parallel_size=2),
                ranks=(0, 1),
                pipeline_p2p=(PipelineP2PConfig(
                    0,
                    1,
                    "send_recv",
                    "nvlink",
                    "point_to_point",
                    "main-pipeline-fabric",
                    "analytical_roofline_v1",
                ),),
            ),
        )),
        num_layers=pp_model.num_hidden_layers,
    )
    step = ScheduleStep(
        "graph-boundary",
        "main",
        (
            ScheduledRequest("decode", "decode", 0, 1, 32),
            ScheduledRequest("prefill", "prefill", 10, 0, 0),
        ),
        attention_kv_layout="paged_varlen",
    )
    pp_graph = build_workload(
        step,
        pp_model,
        pp_deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(graph_config),
    )
    pp_boundaries = tuple(
        node for node in pp_graph.nodes
        if node.kind is WorkloadKind.P2P
        or (node.kind is WorkloadKind.COLLECTIVE
            and ":pp-0-to-1:" in node.node_id))
    assert pp_boundaries
    assert all(
        "graph_logical_tokens" in node.attributes
        and node.attributes["graph_logical_tokens"] == 11
        and node.attributes["graph_model_input_tokens"] == 16
        and node.attributes["graph_runtime_mode"] == "FULL"
        for node in pp_boundaries)
    pp_plan = lower_workload(
        pp_graph,
        pp_deployment,
        {"nvidia_a800": hardware},
    )
    assert all(predict(call.kernel_spec, hardware).t_total_ns > 0
               for call in pp_plan.calls)

    af_model = load_model_config("tests/serving/fixtures/llama_scheduler")
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
        ))
    af_deployment = build_af_disagg_plan(
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
        num_layers=af_model.num_hidden_layers,
    )
    af_graph = build_workload(
        ScheduleStep(
            "graph-af-boundary",
            "attention",
            step.requests,
            attention_kv_layout="paged_varlen",
        ),
        af_model,
        af_deployment,
        launch_policy="analytical_launch",
        runtime_inputs=_runtime(graph_config),
    )
    af_boundaries = tuple(
        node for node in af_graph.nodes
        if node.kind is WorkloadKind.AF_ACTIVATION_TRANSFER)
    assert af_boundaries
    assert all(
        "graph_logical_tokens" in node.attributes
        and node.attributes["graph_logical_tokens"] == 11
        and node.attributes["graph_model_input_tokens"] == 16
        and node.attributes["graph_runtime_mode"] == "FULL"
        for node in af_boundaries)
    af_plan = lower_workload(
        af_graph,
        af_deployment,
        {"nvidia_a800": hardware},
    )
    assert all(predict(call.kernel_spec, hardware).t_total_ns > 0
               for call in af_plan.calls)

    forged_transfer = af_boundaries[0]
    forged_attributes = dict(forged_transfer.attributes)
    del forged_attributes["graph_logical_tokens"]
    del forged_attributes["graph_model_input_tokens"]
    del forged_attributes["graph_runtime_mode"]
    forged_nodes = tuple(
        replace(node, attributes=forged_attributes)
        if node.node_id == forged_transfer.node_id else node
        for node in af_graph.nodes)
    with pytest.raises(ValueError, match="omitted its producer token domain"):
        lower_workload(
            WorkloadGraph(forged_nodes, af_graph.edges),
            af_deployment,
            {"nvidia_a800": hardware},
        )


def test_explicit_sparse_capture_sizes_and_piecewise_mode():
    config = GraphExecutionConfig(
        "FULL_AND_PIECEWISE", (32, 1, 16, 3), _PIECEWISE_EAGER_KINDS)
    small_shape = config.resolve(2, uniform_decode=True)
    shape = config.resolve(17, uniform_decode=False)
    assert config.capture_sizes == (1, 3, 16, 32)
    assert small_shape.model_input_tokens == 3
    assert small_shape.runtime_mode == "FULL"
    assert shape.runtime_mode == "PIECEWISE"
    assert shape.model_input_tokens == 32
    assert shape.padding_tokens == 15


def test_disabled_non_uniform_dispatch_does_not_pad():
    config = GraphExecutionConfig("FULL_DECODE_ONLY", _CAPTURE_SIZES, ())
    decode_shape = config.resolve(11, uniform_decode=True)
    mixed_shape = config.resolve(11, uniform_decode=False)
    assert decode_shape.runtime_mode == "FULL"
    assert decode_shape.model_input_tokens == 16
    assert mixed_shape.runtime_mode == "NONE"
    assert mixed_shape.model_input_tokens == 11


def test_graph_padding_preserves_logical_kv_offload_block_range():
    model = load_model_config("tests/serving/fixtures/llama_scheduler")
    bytes_per_token = model.kv_bytes_per_token_per_parallel_rank(1, 1)
    block_tokens = 16
    block_bytes = block_tokens * bytes_per_token
    cache = KVCacheConfig(
        tiers=(
            KVCacheTierConfig("hbm", block_bytes, block_tokens),
            KVCacheTierConfig("host", block_bytes, block_tokens, "pcie"),
        ),
        bytes_per_token=bytes_per_token,
        offload_policy="lru",
    )
    deployment = build_colocated_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig(
                "main", "colocated", "nvidia_a800", kv_cache=cache),
        )),
        num_layers=model.num_hidden_layers,
    )
    step = ScheduleStep(
        "offload",
        "main",
        (ScheduledRequest("request", "prefill", 3, 0, 16),),
        attention_kv_layout="paged_varlen",
    )
    event = KVCacheEvent(
        "request",
        KVCacheAction.EVICT,
        "host",
        0,
        block_tokens,
        block_bytes,
        source_tier="hbm",
        destination_tier="host",
        ready_event="evict-request-0",
    )

    graph = build_workload(
        step,
        model,
        deployment,
        launch_policy="analytical_launch",
        kv_cache_events=(event,),
        runtime_inputs=_runtime(_graph_config()),
    )

    # A800 prefill carries the authorized measured-zero token-embedding
    # emission observation (a launch-placement divergence at the vendor
    # dispatch layer, not a source predicate branch), so the step-head
    # gather is not part of the workload graph; decode/mixed still emit it.
    assert not any(
        node.kind is WorkloadKind.GATHER and node.variant == "embedding"
        for node in graph.nodes)
    offload = next(
        node for node in graph.nodes if node.kind is WorkloadKind.KV_OFFLOAD)
    assert offload.attributes["token_start"] == 0
    assert offload.attributes["token_end"] == block_tokens
    assert offload.attributes["bytes"] == block_bytes


@pytest.mark.parametrize(
    ("configured_mode", "expected_modes"),
    (
        ("NONE", ("NONE", "NONE")),
        ("PIECEWISE", ("PIECEWISE", "PIECEWISE")),
        ("FULL", ("FULL", "FULL")),
        ("FULL_DECODE_ONLY", ("FULL", "NONE")),
        ("FULL_AND_PIECEWISE", ("FULL", "PIECEWISE")),
    ),
)
def test_configured_mode_is_parsed_by_the_uniform_dispatch_table(
    configured_mode, expected_modes
):
    capture_sizes_by_mode = {
        "NONE": (),
        "PIECEWISE": _CAPTURE_SIZES,
        "FULL": _CAPTURE_SIZES,
        "FULL_DECODE_ONLY": _CAPTURE_SIZES,
        "FULL_AND_PIECEWISE": _CAPTURE_SIZES,
    }
    capture_sizes = capture_sizes_by_mode[configured_mode]
    config = GraphExecutionConfig(
        configured_mode,
        capture_sizes,
        _PIECEWISE_EAGER_KINDS if "PIECEWISE" in configured_mode else (),
    )
    assert (config.uniform_decode_mode, config.non_uniform_mode) == expected_modes


@pytest.mark.parametrize(
    "constructor,error",
    (
        (lambda: GraphExecutionConfig("future", (1,), ()), ValueError),
        (lambda: GraphExecutionConfig("NONE", (1,), ()), ValueError),
        (lambda: GraphExecutionConfig("FULL", (), ()), ValueError),
        (lambda: GraphExecutionConfig("FULL", (1, 1), ()), ValueError),
        (lambda: GraphExecutionConfig("FULL", (0, 1), ()), ValueError),
        (lambda: GraphExecutionConfig("FULL", (1,), (), 0), ValueError),
        (lambda: GraphExecutionConfig(
            "FULL_AND_PIECEWISE", (1,), ()), ValueError),
        (lambda: GraphExecutionConfig(
            "FULL", (1,), _PIECEWISE_EAGER_KINDS), ValueError),
    ),
)
def test_graph_contract_rejects_unknown_or_invalid_configuration(
        constructor, error):
    with pytest.raises(error):
        constructor()


def _ascend_qk_norm_rope_graph(domain, *, tokens):
    """Build the three-op chain the vLLM-Ascend QKNormRope pass matches."""

    query_heads = 2
    kv_heads = 1
    head_dim = 128
    grouped_attributes = {
        "launch_policy": "analytical_launch",
        "input_materialization": "grouped_strided_to_contiguous",
        "input_outer_groups": tokens,
        "input_group_stride_elements": (
            (query_heads + 2 * kv_heads) * head_dim),
        "input_row_stride_elements": head_dim,
        **domain,
    }
    q_norm = WorkloadNode(
        "q-norm", WorkloadKind.NORM, "main",
        {"rows": tokens * query_heads, "hidden": head_dim},
        "bf16", "rmsnorm", rank=0,
        attributes={
            **grouped_attributes, "input_rows_per_group": query_heads},
    )
    k_norm = WorkloadNode(
        "k-norm", WorkloadKind.NORM, "main",
        {"rows": tokens * kv_heads, "hidden": head_dim},
        "bf16", "rmsnorm", rank=0,
        attributes={
            **grouped_attributes, "input_rows_per_group": kv_heads},
    )
    rotary = WorkloadNode(
        "rotary", WorkloadKind.ROTARY, "main",
        {
            "elements": tokens * (query_heads + kv_heads) * head_dim,
            "seq_len": tokens,
            "num_query_heads": query_heads,
            "num_kv_heads": kv_heads,
            "head_dim": head_dim,
        },
        "bf16", "rotary", rank=0,
        attributes={
            "launch_policy": "analytical_launch",
            # no observed V materialization in the phases these fixtures
            # model; the contract attribute is still mandatory
            "qkv_v_materialization_count": 0,
            **domain,
        },
    )
    return WorkloadGraph(
        (q_norm, k_norm, rotary),
        (
            WorkloadEdge("q-norm", "k-norm", "data"),
            WorkloadEdge("k-norm", "rotary", "data"),
        ),
    )


def _ascend_qkv_rmsnorm_rope_prediction(domain, *, tokens):
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    deployment = build_colocated_plan(
        DeploymentConfig("colocated", (
            WorkerPoolConfig("main", "colocated", "ascend_910b"),
        )),
        num_layers=1,
    )
    plan = lower_workload(
        _ascend_qk_norm_rope_graph(domain, tokens=tokens),
        deployment,
        {"ascend_910b": hardware},
    )
    # Q/K Norm and RoPE lower as separate programs: the canonical launch
    # observation carries no fused split_qkv_rmsnorm_rope signature, so each
    # norm gets its input materialization copy and RoPE stands on its own.
    # The graph-padding regime invariants below ride on the RoPE call.
    assert [call.kernel_spec.variant for call in plan.calls] == [
        "elementwise", "rmsnorm", "elementwise", "rmsnorm", "rotary"]
    rotary_call = plan.calls[-1]
    return rotary_call.kernel_spec, predict(
        rotary_call.kernel_spec, hardware, "midlayer")


def _limitations(prediction):
    if prediction.diagnostics is None:
        return ()
    return prediction.diagnostics.limitations


def test_exact_size_full_graph_qkv_rmsnorm_rope_is_not_priced_as_eager():
    """The G1 regression: zero bucket padding is still a graph replay.

    ``VectorPredictor.midlayer`` reads the graph-domain attributes to decide
    whether this call is a replayed fused program or the capture-limit eager
    fallback.  A capture size that happens to equal the logical token count
    carries no padding rows, so the workload builder used to drop the domain.
    The eager deployment was later falsified as a three-kernel sequence
    (msprof shows one fused split_qkv_rmsnorm_rope kernel in both lanes;
    vllm-ascend fuses QKNorm+RoPE at torchair compile time), so both regimes
    now price the same fused envelope and only the regime metadata differs.
    """

    tokens = 16
    exact_spec, exact = _ascend_qkv_rmsnorm_rope_prediction(
        {
            "graph_logical_tokens": tokens,
            "graph_model_input_tokens": tokens,
            "graph_runtime_mode": "FULL",
        },
        tokens=tokens,
    )
    eager_spec, eager = _ascend_qkv_rmsnorm_rope_prediction({}, tokens=tokens)

    assert graph_replayed_launch(exact_spec)
    assert not graph_replayed_launch(eager_spec)
    # Same logical and physical work; only the execution regime differs.
    assert exact_spec.shape == eager_spec.shape
    # Both regimes price one fused envelope: reintroducing a phantom serial
    # eager split would make the fallback regime cost more than the replay.
    assert exact.t_total_ns == eager.t_total_ns


def test_exact_size_full_graph_qkv_matches_the_padded_graph_regime():
    """Zero padding must not change which execution path is priced."""

    tokens = 16
    exact_spec, exact = _ascend_qkv_rmsnorm_rope_prediction(
        {
            "graph_logical_tokens": tokens,
            "graph_model_input_tokens": tokens,
            "graph_runtime_mode": "FULL",
        },
        tokens=tokens,
    )
    padded_spec, padded = _ascend_qkv_rmsnorm_rope_prediction(
        {
            "graph_logical_tokens": tokens - 5,
            "graph_model_input_tokens": tokens,
            "graph_runtime_mode": "FULL",
        },
        tokens=tokens,
    )
    assert graph_replayed_launch(padded_spec)
    assert padded_spec.launch == exact_spec.launch
    # Same physical launch, strictly more logical work in the exact case:
    # the replayed program cannot get cheaper by doing more valid rows.
    assert exact.t_total_ns >= padded.t_total_ns
