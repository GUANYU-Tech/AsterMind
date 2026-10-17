import pytest

from evaluation.serving.metrics.diagnostics.ascend_910b_compiled_fx import (
    ascend_910b_compiled_fx_backend_events,
    ascend_910b_compiled_fx_reshape_events,
)
from src.serving.compiled_fx import (
    ascend_910b_compiled_fx_dense_metadata_events,
    ascend_910b_compiled_fx_rotary_events,
)
from src.kernel.kernel_spec import KernelKind, KernelSpec, PhysicalTransferConfig
from src.kernel.launch_config import LaunchConfig
from src.serving.kernel_lowering.kernel_plan import (
    KernelBlockRepeat,
    KernelCall,
    KernelPlan,
)
from src.serving.runtime_prepare import RuntimePrepareEvent, runtime_prepare_events


def _vector_spec():
    return KernelSpec(
        KernelKind.VECTOR,
        "ascend_910b",
        {"element_count": 1},
        dtype="bf16",
        variant="elementwise",
        attributes={"reads": 1, "writes": 1, "operation": "copy"},
        launch=LaunchConfig(
            "fixture", (1, 1, 1), (1, 1, 1),
            execution_variant="elementwise"),
    )


def _collective_spec(participants=(0, 1)):
    return KernelSpec(
        KernelKind.COMMUNICATION,
        "ascend_910b",
        transfer=PhysicalTransferConfig(
            algorithm="ring",
            bytes=4096,
            direction="collective",
            link="hccs",
            topology="ring",
            participants=participants,
            operation="all_reduce",
            bytes_semantics="per_rank_input",
            ring_order=participants,
            reduction="sum",
            backend="protobridge",
            backend_config_sha256="0" * 64,
            resource_id="hccs:fixture",
        ),
    )


def _call(call_id, rank, spec):
    return KernelCall(call_id, spec, "main", rank, "compute", (call_id,))


def _decoder_layer(layer, physical_ranks=(0, 1)):
    calls = []
    for rank in physical_ranks:
        for name in (
                "input-norm", "qkv", "q-norm", "attention", "o-proj",
                "post-attn-norm", "router"):
            calls.append(_call(
                f"fixture:layer-{layer}:{name}:tp-rank-{rank}", rank,
                _vector_spec()))
        calls.append(_call(
            f"fixture:layer-{layer}:moe:ep-rank-{rank}:0", rank,
            _vector_spec()))
    calls.extend((
        _call(
            f"fixture:layer-{layer}:o-proj:tp-all-reduce", physical_ranks[0],
            _collective_spec(physical_ranks)),
        _call(
            f"fixture:layer-{layer}:moe:ep-output-all-reduce", physical_ranks[0],
            _collective_spec(physical_ranks)),
    ))
    return calls


def test_compiled_fx_descriptor_has_674_source_entries_per_rank():
    layer_zero = _decoder_layer(0)
    layer_one = _decoder_layer(1)
    endpoints = [
        _call(f"fixture:token-embedding:tp-rank-{rank}", rank, _vector_spec())
        for rank in (0, 1)
    ] + [
        _call(f"fixture:final-norm:tp-rank-{rank}", rank, _vector_spec())
        for rank in (0, 1)
    ]
    plan = KernelPlan(
        tuple(endpoints + layer_zero + layer_one),
        (KernelBlockRepeat(
            "fixture:layer-1", tuple(call.call_id for call in layer_one), 47,
            ()),),
    )

    events = ascend_910b_compiled_fx_backend_events(plan)

    assert {
        rank: sum(
            event.count
            for (pool, lane, _, _), entry in events.items()
            if pool == "main" and lane == rank
            for event in entry
        )
        for rank in (0, 1)
    } == {0: 674, 1: 674}

    production_events = runtime_prepare_events(
        plan, enabled_mechanisms=("compiled_fx_backend_entry",))
    assert {
        rank: sum(
            event.count
            for (pool, lane, _, _), entry in production_events.items()
            if pool == "main" and lane == rank
            for event in entry
        )
        for rank in (0, 1)
    } == {0: 674, 1: 674}


def test_compiled_fx_reshape_descriptor_has_288_nodes_per_rank():
    layer_zero = _decoder_layer(0)
    layer_one = _decoder_layer(1)
    endpoints = [
        _call(f"fixture:token-embedding:tp-rank-{rank}", rank, _vector_spec())
        for rank in (0, 1)
    ] + [
        _call(f"fixture:final-norm:tp-rank-{rank}", rank, _vector_spec())
        for rank in (0, 1)
    ]
    plan = KernelPlan(
        tuple(endpoints + layer_zero + layer_one),
        (KernelBlockRepeat(
            "fixture:layer-1", tuple(call.call_id for call in layer_one), 47,
            ()),),
    )

    events = ascend_910b_compiled_fx_reshape_events(plan)

    assert {
        rank: sum(
            event.count
            for (pool, lane, _, _), entry in events.items()
            if pool == "main" and lane == rank
            for event in entry
        )
        for rank in (0, 1)
    } == {0: 288, 1: 288}

    production_events = runtime_prepare_events(
        plan, enabled_mechanisms=("compiled_fx_aten_reshape_metadata",))
    assert {
        rank: sum(
            event.count
            for (pool, lane, _, _), entry in production_events.items()
            if pool == "main" and lane == rank
            for event in entry
        )
        for rank in (0, 1)
    } == {0: 288, 1: 288}


def test_production_compiled_fx_accepts_remapped_ep_physical_ranks():
    physical_ranks = (91, 7)
    layer_zero = _decoder_layer(0, physical_ranks)
    layer_one = _decoder_layer(1, physical_ranks)
    endpoints = [
        _call(f"fixture:token-embedding:tp-rank-{rank}", rank, _vector_spec())
        for rank in physical_ranks
    ] + [
        _call(f"fixture:final-norm:tp-rank-{rank}", rank, _vector_spec())
        for rank in physical_ranks
    ]
    plan = KernelPlan(
        tuple(endpoints + layer_zero + layer_one),
        (KernelBlockRepeat(
            "fixture:layer-1", tuple(call.call_id for call in layer_one), 47,
            ()),),
    )

    events = runtime_prepare_events(
        plan,
        enabled_mechanisms=(
            "compiled_fx_backend_entry", "compiled_fx_moe_wrapper_control"),
    )

    assert {
        rank: sum(
            event.count
            for (pool, lane, _, _), entry in events.items()
            if pool == "main" and lane == rank
            for event in entry
        )
        for rank in physical_ranks
    } == {91: 674 + 48, 7: 674 + 48}


def test_production_dense_node_dispatch_charges_entry_and_copy_nodes():
    """The dense decoder is one unsplit FX graph: one graph entry per
    forward plus per-node class dispatch, never per-call backend entries."""
    layer_zero = [
        _call(f"fixture:layer-0:{name}", 0, _vector_spec())
        for name in (
            "input-norm", "qkv", "rotary-q-helper", "q-norm", "kv-update", "attention:0",
            "attention:1", "o-proj", "post-attn-norm", "gate-up",
            "activation", "down",
        )
    ]
    layer_one = [
        _call(f"fixture:layer-1:{name}", 0, _vector_spec())
        for name in (
            "input-norm", "qkv", "q-norm", "kv-update", "attention:0",
            "attention:1", "attention:2", "o-proj", "post-attn-norm",
            "gate-up", "activation", "down",
        )
    ]
    endpoints = [
        _call("fixture:token-embedding", 0, _vector_spec()),
        _call("fixture:final-norm", 0, _vector_spec()),
        _call("fixture:lm-head", 0, _vector_spec()),
    ]
    plan = KernelPlan(
        tuple(endpoints + layer_zero + layer_one),
        (KernelBlockRepeat(
            "fixture:layer-1", tuple(call.call_id for call in layer_one), 3,
            ()),),
    )

    events = runtime_prepare_events(
        plan,
        enabled_mechanisms=(
            "compiled_fx_graph_entry",
            "compiled_fx_aten_copy_node_dispatch",
        ),
    )

    node_counts = sum(
        event.count for entry in events.values() for event in entry
        if event.mechanism == "compiled_fx_aten_copy_node_dispatch")
    entry_counts = sum(
        event.count for entry in events.values() for event in entry
        if event.mechanism == "compiled_fx_graph_entry")
    # Every dense call is one copy-class node; Llama-style rotary helper
    # children are folded into their parent source call and skipped.
    assert node_counts == len(endpoints) + len(layer_zero) - 1 + 3 * len(layer_one)
    assert entry_counts == 1
    assert ("main", 0, "fixture:token-embedding", -1) in events
    embedding_events = events[("main", 0, "fixture:token-embedding", -1)]
    assert [event.mechanism for event in embedding_events] == [
        "compiled_fx_aten_copy_node_dispatch", "compiled_fx_graph_entry"]


def test_production_dense_node_dispatch_fails_closed_on_unknown_variant():
    layer_zero = [
        _call(f"fixture:layer-0:{name}", 0, _vector_spec())
        for name in ("input-norm", "qkv", "o-proj")
    ]
    # A valid variant outside the dense decoder node-class table: the
    # mapping must fail closed instead of borrowing a neighbouring class.
    unmapped = KernelCall(
        "fixture:layer-0:unmapped-op",
        KernelSpec(
            KernelKind.VECTOR,
            "ascend_910b",
            {
                "tokens": 16,
                "experts": 16,
                "topk": 2,
                "hidden": 64,
                "intermediate": 128,
            },
            dtype="bf16",
            variant="moe_gating_topk",
            attributes={},
            launch=LaunchConfig(
                "fixture", (1, 1, 1), (1, 1, 1),
                execution_variant="moe_gating_topk"),
        ),
        "main", 0, "compute", ("fixture:layer-0:unmapped-op",),
    )
    plan = KernelPlan(tuple(layer_zero + [unmapped]))

    with pytest.raises(ValueError, match="no node class"):
        runtime_prepare_events(
            plan,
            enabled_mechanisms=("compiled_fx_graph_entry",),
        )


def test_production_dense_node_dispatch_skips_tp_collective_nodes():
    """Dense TP collective nodes stay on the collective-prepare lane.

    The dense node-class table is pinned by the single-rank qwen3-8b
    census, which contains no communication nodes.  A dense TP plan's
    all-reduce Host lane is priced by the collective prepare mechanisms,
    so the class mapping skips communication calls instead of failing
    the mapping or borrowing a compute class (a double charge at the
    same source site)."""
    layer_zero = [
        _call(f"fixture:layer-0:{name}", 0, _vector_spec())
        for name in ("input-norm", "qkv", "o-proj")
    ]
    endpoints = [
        _call(f"fixture:token-embedding:tp-rank-{rank}", rank,
              _vector_spec())
        for rank in (0, 1)
    ]
    collective = _call(
        "fixture:token-embedding:tp-all-reduce", 0, _collective_spec((0, 1)))
    plan = KernelPlan(tuple(endpoints + layer_zero + [collective]))

    events = runtime_prepare_events(
        plan,
        enabled_mechanisms=(
            "compiled_fx_graph_entry",
            "compiled_fx_aten_copy_node_dispatch",
            "collective_all_reduce_prepare",
        ),
    )

    for rank in (0, 1):
        collective_events = events[
            ("main", rank, "fixture:token-embedding:tp-all-reduce", -1)]
        assert [
            event.mechanism for event in collective_events
        ] == ["collective_all_reduce_prepare"]
    assert [
        event.mechanism for event in events[
            ("main", 0, "fixture:token-embedding:tp-rank-0", -1)]
    ] == ["compiled_fx_aten_copy_node_dispatch", "compiled_fx_graph_entry"]
    assert [
        event.mechanism for event in events[
            ("main", 1, "fixture:token-embedding:tp-rank-1", -1)]
    ] == ["compiled_fx_aten_copy_node_dispatch", "compiled_fx_graph_entry"]


def test_dense_reshape_contract_covers_tp1_split_attention_children():
    """Dense fallback graphs consume the same static metadata mechanism.

    The source contract has three attention and two output-projection reshape
    nodes per layer.  Lowering may expose attention as numeric split children;
    those children must partition the declared count without changing the
    layer/repeat occurrence domain.
    """
    layer_zero = [
        _call(f"fixture:layer-0:attention:tp-rank-0:{index}", 0,
              _vector_spec())
        for index in range(3)
    ] + [
        _call("fixture:layer-0:o-proj:tp-rank-0", 0, _vector_spec()),
    ]
    layer_one = [
        _call(f"fixture:layer-1:attention:tp-rank-0:{index}", 0,
              _vector_spec())
        for index in range(2)
    ] + [
        _call("fixture:layer-1:o-proj:tp-rank-0", 0, _vector_spec()),
    ]
    plan = KernelPlan(
        tuple(layer_zero + layer_one),
        (KernelBlockRepeat(
            "fixture:layer-1", tuple(call.call_id for call in layer_one), 3,
            ()),),
    )
    events = runtime_prepare_events(
        plan, enabled_mechanisms=("compiled_fx_aten_reshape_metadata",))
    assert sum(event.count for entry in events.values() for event in entry) == (
        5 + 3 * 5)
    assert sum(
        event.count for key, entry in events.items()
        if key[2].startswith("fixture:layer-1")
        for event in entry
    ) == 15


def test_dense_fallback_metadata_contract_counts_qwen3_source_nodes():
    layer_zero = [
        _call(f"fixture:layer-0:{name}", 0, _vector_spec())
        for name in ("qkv", "q-norm", "attention:0", "attention:1",
                     "attention:2", "o-proj")
    ]
    layer_one = [
        _call(f"fixture:layer-1:{name}", 0, _vector_spec())
        for name in ("qkv", "q-norm", "attention:0", "attention:1",
                     "attention:2", "o-proj")
    ]
    plan = KernelPlan(
        tuple(layer_zero + layer_one),
        (KernelBlockRepeat(
            "fixture:layer-1", tuple(call.call_id for call in layer_one), 2,
            ()),),
    )
    events = ascend_910b_compiled_fx_dense_metadata_events(plan)
    assert sum(event.count for entry in events.values() for event in entry
               if event.mechanism == "compiled_fx_aten_view_metadata") == 27
    assert sum(event.count for entry in events.values() for event in entry
               if event.mechanism == "compiled_fx_aten_empty_metadata") == 3


def test_dense_fallback_metadata_contract_counts_llama_source_nodes():
    layer_zero = [
        _call(f"fixture:layer-0:{name}", 0, _vector_spec())
        for name in ("qkv", "attention:0", "attention:1", "attention:2",
                     "o-proj")
    ]
    layer_one = [
        _call(f"fixture:layer-1:{name}", 0, _vector_spec())
        for name in ("qkv", "attention:0", "attention:1", "attention:2",
                     "o-proj")
    ]
    plan = KernelPlan(
        tuple(layer_zero + layer_one),
        (KernelBlockRepeat(
            "fixture:layer-1", tuple(call.call_id for call in layer_one), 2,
            ()),),
    )
    events = ascend_910b_compiled_fx_dense_metadata_events(plan)
    assert sum(event.count for entry in events.values() for event in entry
               if event.mechanism == "compiled_fx_aten_view_metadata") == 15
    assert sum(event.count for entry in events.values() for event in entry
               if event.mechanism == "compiled_fx_aten_empty_metadata") == 3


def test_dense_rotary_fallback_contract_charges_one_source_helper_per_layer():
    layer_zero = [
        _call(f"fixture:layer-0:{name}", 0, _vector_spec())
        for name in (
            "rotary-cos-sin-gather",
            "rotary-q-x1-cos-operand-slice",
            "rotary-q-x1-cos-mul",
            "rotary-q-o1-sub",
            "rotary-q-rotate-concat",
        )
    ]
    layer_one = [
        _call(f"fixture:layer-1:{name}", 0, _vector_spec())
        for name in (
            "rotary-cos-sin-gather",
            "rotary-k-x1-cos-operand-slice",
            "rotary-k-x1-cos-mul",
            "rotary-k-o1-sub",
            "rotary-k-rotate-concat",
        )
    ]
    plan = KernelPlan(
        tuple(layer_zero + layer_one),
        (KernelBlockRepeat(
            "fixture:layer-1", tuple(call.call_id for call in layer_one), 3,
            ()),),
    )
    events = ascend_910b_compiled_fx_rotary_events(plan)
    assert sum(
        event.count for entry in events.values() for event in entry
        if event.mechanism == "compiled_fx_rotary_composite_entry"
    ) == 4
    production_events = runtime_prepare_events(
        plan, enabled_mechanisms=("compiled_fx_rotary_composite_entry",))
    assert sum(
        event.count for entry in production_events.values() for event in entry
        if event.mechanism == "compiled_fx_rotary_composite_entry"
    ) == 4


def test_production_model_runner_outer_contract_uses_embedding_and_lm_head():
    plan = KernelPlan((
        _call("fixture:token-embedding", 0, _vector_spec()),
        _call("fixture:lm-head", 0, _vector_spec()),
    ))

    events = runtime_prepare_events(
        plan,
        enabled_mechanisms=(
            "model_runner_input_prepare",
            "model_runner_forward_metadata_prepare",
            "model_runner_logits_prepare",
        ),
    )

    assert events == {
        ("main", 0, "fixture:token-embedding", -1): (
            RuntimePrepareEvent(
                "model_runner_input_prepare", graph_capturable=False),
            RuntimePrepareEvent(
                "model_runner_forward_metadata_prepare", graph_capturable=False),
        ),
        ("main", 0, "fixture:lm-head", -1): (
            RuntimePrepareEvent(
                "model_runner_logits_prepare", graph_capturable=False),
        ),
    }


def test_production_model_runner_outer_contract_accepts_ranked_anchors():
    physical_ranks = (91, 7)
    plan = KernelPlan(tuple(
        _call(f"fixture:{anchor}:tp-rank-{logical_rank}", physical_rank,
              _vector_spec())
        for logical_rank, physical_rank in enumerate(physical_ranks)
        for anchor in ("token-embedding", "lm-head")
    ))

    events = runtime_prepare_events(
        plan,
        enabled_mechanisms=("model_runner_logits_prepare",),
    )

    assert set(events) == {
        ("main", physical_rank,
         f"fixture:lm-head:tp-rank-{logical_rank}", -1)
        for logical_rank, physical_rank in enumerate(physical_ranks)
    }
    assert all(
        entry[0] == RuntimePrepareEvent(
            "model_runner_logits_prepare", graph_capturable=False)
        for entry in events.values()
    )


def test_production_model_runner_outer_contract_rejects_logical_rank_drift():
    plan = KernelPlan((
        _call("fixture:token-embedding:tp-rank-0", 91, _vector_spec()),
        _call("fixture:lm-head:tp-rank-1", 91, _vector_spec()),
    ))

    with pytest.raises(ValueError, match="logical ranks differ"):
        runtime_prepare_events(
            plan,
            enabled_mechanisms=("model_runner_logits_prepare",),
        )


def test_production_model_runner_outer_contract_rejects_missing_lm_head():
    plan = KernelPlan((_call("fixture:token-embedding", 0, _vector_spec()),))

    with pytest.raises(ValueError, match="matching token-embedding"):
        runtime_prepare_events(
            plan,
            enabled_mechanisms=("model_runner_input_prepare",),
        )
