from types import SimpleNamespace

import pytest

import evaluation.serving.metrics.diagnostics.ascend_910b_host_prepare as prepare_module
from evaluation.serving.metrics.diagnostics.ascend_910b_host_prepare import (
    ascend_910b_fia_source_wrapper_events,
    ascend_910b_model_runner_outer_events,
    ascend_910b_rotary_cache_update_events,
    ascend_910b_moe_wrapper_control_events,
    ascend_910b_qkv_triton_composite_events,
    ascend_910b_runtime_prepare_events,
    ascend_910b_unquant_mlp_wrapper_events,
)
from src.kernel.kernel_spec import KernelKind, KernelSpec, PhysicalTransferConfig
from src.kernel.hardware_spec import HWSpec
from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.serving.host_timing import RuntimePrepareEvent
from src.serving.kernel_lowering.kernel_plan import (
    KernelBlockRepeat,
    KernelCall,
    KernelPlan,
)
from src.serving.runtime_prepare import (
    runtime_prepare_events as production_runtime_prepare_events,
)


def _spec(kind, variant, hardware_id="ascend_910b"):
    shape = (
        {
            "tokens": 16,
            "experts": 16,
            "topk": 2,
            "hidden": 64,
            "intermediate": 128,
        }
        if variant.startswith("moe_")
        else {
            "seq_len": 16,
            "num_query_heads": 8,
            "num_kv_heads": 2,
            "head_dim": 128,
            "rotary_dim": 128,
            "element_count": 16 * 12 * 128,
        }
        if variant == "qkv_rmsnorm_rope"
        else {
            "batch": 1,
            "q_len": 1,
            "kv_len": 16,
            "head_dim": 128,
            "num_query_heads": 8,
            "num_kv_heads": 2,
        }
        if kind is KernelKind.FLASH_ATTN
        else {"element_count": 16}
    )
    attributes = (
        {"routed_rows": 16}
        if variant == "moe_swiglu"
        else
        {"reads": 2, "writes": 1, "operation": "add"}
        if variant == "elementwise"
        else {
            "causal": False,
            "kv_layout": "contiguous",
            "kv_dist": "uniform",
            "q_lens": (1,),
            "kv_lens": (16,),
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
        }
        if kind is KernelKind.FLASH_ATTN
        else {"bias": False, "is_neox": True}
        if variant == "qkv_rmsnorm_rope"
        else {}
    )
    parameters = (
        {
            "unpadded_flops": 1.0,
            "mma_smem_bytes_per_flop": 0.0,
            "split_scratch_dram_bytes_per_element": 0.0,
            "page_table_bytes_per_kv_panel": 0.0,
        }
        if kind is KernelKind.FLASH_ATTN
        else
        {"cann_moe_init_routing_mix_aiv_1_0": 1}
        if variant == "moe_init_routing_v2"
        else {
            "source_vllm_ascend_split_qkv_rmsnorm_rope_program": 1.0,
            "qk_batch_rows_per_tile": 1.0,
            "v_batch_rows_per_tile": 42.0,
            "source_ub_bytes": 87040.0,
        }
        if variant == "qkv_rmsnorm_rope"
        else {}
    )
    algorithms = {
        "moe_gating_topk": "aclnn_moe_gating_topk",
        "moe_init_routing_v2": "aclnn_moe_init_routing_v2",
        "qkv_rmsnorm_rope": "vllm_ascend_split_qkv_rmsnorm_rope",
    }
    return KernelSpec(
        kind,
        hardware_id,
        shape,
        dtype="bf16",
        variant=variant,
        attributes=attributes,
        launch=LaunchConfig(
            algorithm=algorithms.get(variant, "fixture"),
            grid=(1, 1, 1),
            block=(1, 1, 1),
            parameters=parameters,
            execution_variant=variant,
        ),
    )


def test_prepare_occurrences_follow_source_variants_and_repeats():
    gating = KernelCall(
        "gating",
        _spec(KernelKind.VECTOR, "moe_gating_topk"),
        "main",
        1,
        "compute",
        ("moe",),
    )
    routing = KernelCall(
        "routing",
        _spec(KernelKind.VECTOR, "moe_init_routing_v2"),
        "main",
        1,
        "compute",
        ("moe",),
    )
    plan = KernelPlan(
        (gating, routing),
        (KernelBlockRepeat("layers", ("gating", "routing"), 3, ()),),
    )

    events = ascend_910b_runtime_prepare_events(
        plan,
        enabled_mechanisms=(
            "moe_gating_prepare",
            "moe_token_dispatch_prepare",
        ),
    )

    assert events == {
        ("main", 1, "gating", occurrence): (
            RuntimePrepareEvent("moe_gating_prepare"),
        )
        for occurrence in range(3)
    } | {
        ("main", 1, "routing", occurrence): (
            RuntimePrepareEvent("moe_token_dispatch_prepare"),
        )
        for occurrence in range(3)
    }


def test_prepare_occurrences_reject_non_910b_plan():
    plan = KernelPlan((KernelCall(
        "gating",
        _spec(KernelKind.VECTOR, "elementwise", "nvidia_a800"),
        "main",
        0,
        "compute",
        ("moe",),
    ),))

    with pytest.raises(ValueError, match="Ascend-only"):
        ascend_910b_runtime_prepare_events(plan)


def test_prepare_occurrences_filter_unmeasured_mechanisms():
    gating = KernelCall(
        "gating",
        _spec(KernelKind.VECTOR, "moe_gating_topk"),
        "main",
        0,
        "compute",
        ("moe",),
    )
    plan = KernelPlan((gating,))

    events = ascend_910b_runtime_prepare_events(
        plan,
        enabled_mechanisms=("python_argument_prepare",),
    )

    assert events == {}


def test_prepare_occurrences_map_triton_dynamic_arguments():
    split_qkv = KernelCall(
        "split_qkv",
        _spec(KernelKind.VECTOR, "qkv_rmsnorm_rope"),
        "main",
        0,
        "compute",
        ("attention",),
    )
    plan = KernelPlan((split_qkv,))

    events = ascend_910b_runtime_prepare_events(
        plan,
        enabled_mechanisms=("triton_dynamic_argument_prepare",),
    )

    assert events == {
        ("main", 0, "split_qkv", -1): (
            RuntimePrepareEvent("triton_dynamic_argument_prepare"),
        )
    }


def test_qkv_triton_composite_replaces_dynamic_prepare_occurrence():
    split_qkv = KernelCall(
        "split_qkv",
        _spec(KernelKind.VECTOR, "qkv_rmsnorm_rope"),
        "main",
        0,
        "compute",
        ("attention",),
    )
    plan = KernelPlan((split_qkv,))

    events = ascend_910b_qkv_triton_composite_events(plan)

    assert events == {
        ("main", 0, "split_qkv", -1): (
            RuntimePrepareEvent("triton_three_output_composite_prepare"),
        )
    }

    # The wrapper-prepare contract describes the split-piece MoE executor;
    # it fires only when the plan carries the MoE decoder signature.
    moe_signature = KernelCall(
        "req:layer-0:moe:ep-rank-0:0",
        _spec(KernelKind.VECTOR, "elementwise"),
        "main",
        0,
        "compute",
        ("moe",),
    )
    assert production_runtime_prepare_events(
        KernelPlan((split_qkv, moe_signature)),
        enabled_mechanisms=("triton_three_output_composite_prepare",),
    ) == events
    assert production_runtime_prepare_events(
        plan,
        enabled_mechanisms=("triton_three_output_composite_prepare",),
    ) == {}


def test_production_qkv_prepare_rejects_dynamic_and_composite_double_count():
    split_qkv = KernelCall(
        "split_qkv",
        _spec(KernelKind.VECTOR, "qkv_rmsnorm_rope"),
        "main",
        0,
        "compute",
        ("attention",),
    )

    with pytest.raises(ValueError, match="overlap"):
        production_runtime_prepare_events(
            KernelPlan((split_qkv,)),
            enabled_mechanisms=(
                "triton_dynamic_argument_prepare",
                "triton_three_output_composite_prepare",
            ),
        )


def test_production_w1_anchor_charges_routing_index_mask_mul():
    shape = {
        "tokens": 16,
        "experts": 128,
        "topk": 8,
        "hidden": 2048,
        "intermediate": 768,
    }
    launch_attributes = {
        "tokens_per_expert": (1,) * 128,
        "local_tokens_per_expert": (1,) * 128,
        "expert_parallel_size": 1,
        "expert_rank": 0,
        "owned_experts": tuple(range(128)),
        "expert_weight_tile_rows": None,
        "launch_policy": "analytical_launch",
    }
    attributes = {
        name: value for name, value in launch_attributes.items()
        if name != "launch_policy"
    }
    spec = KernelSpec(
        KernelKind.FUSED_MOE,
        "ascend_910b",
        shape,
        dtype="bf16",
        variant="w1",
        attributes=attributes,
        launch=select_launch_config(
            "ascend_910b",
            KernelKind.FUSED_MOE,
            shape,
            "bf16",
            "w1",
            launch_attributes,
            hardware=HWSpec("configs/hardware/ascend_910b.yaml"),
        ),
    )
    call = KernelCall(
        "step:layer-0:moe:w1",
        spec,
        "main",
        0,
        "compute",
        ("moe",),
    )
    events = production_runtime_prepare_events(
        KernelPlan((call,)),
        enabled_mechanisms=(
            "moe_grouped_gemm_prepare",
            "moe_routing_index_mask_mul_prepare",
        ),
    )
    assert events == {
        ("main", 0, "step:layer-0:moe:w1", -1): (
            RuntimePrepareEvent("moe_grouped_gemm_prepare"),
            RuntimePrepareEvent("moe_routing_index_mask_mul_prepare"),
        ),
    }


def test_fia_source_wrapper_replaces_raw_fia_prepare_occurrence():
    flash_attention = KernelCall(
        "flash_attention",
        _spec(KernelKind.FLASH_ATTN, "decode"),
        "main",
        0,
        "compute",
        ("attention",),
    )
    plan = KernelPlan((flash_attention,))

    events = ascend_910b_fia_source_wrapper_events(plan)

    assert events == {
        ("main", 0, "flash_attention", -1): (
            RuntimePrepareEvent("fused_attention_source_wrapper_prepare"),
        )
    }

    # The wrapper-prepare contract describes the split-piece MoE executor;
    # it fires only when the plan carries the MoE decoder signature.
    moe_signature = KernelCall(
        "req:layer-0:moe:ep-rank-0:0",
        _spec(KernelKind.VECTOR, "elementwise"),
        "main",
        0,
        "compute",
        ("moe",),
    )
    assert production_runtime_prepare_events(
        KernelPlan((flash_attention, moe_signature)),
        enabled_mechanisms=("fused_attention_source_wrapper_prepare",),
    ) == events
    assert production_runtime_prepare_events(
        plan,
        enabled_mechanisms=("fused_attention_source_wrapper_prepare",),
    ) == {}


def test_hardware_profile_applies_generic_mechanisms_for_moe_signature_plans():
    """Dense decoders are priced by per-node dispatch instead, so the
    generic wrapper-prepare mechanisms fire only for MoE-signature plans."""
    flash_attention = KernelCall(
        "dense:attention",
        _spec(KernelKind.FLASH_ATTN, "decode"),
        "main",
        0,
        "compute",
        ("attention",),
    )
    split_qkv = KernelCall(
        "dense:qkv",
        _spec(KernelKind.VECTOR, "qkv_rmsnorm_rope"),
        "main",
        0,
        "compute",
        ("attention",),
    )
    moe_signature = KernelCall(
        "req:layer-0:moe:ep-rank-0:0",
        _spec(KernelKind.VECTOR, "elementwise"),
        "main",
        0,
        "compute",
        ("moe",),
    )

    dense_events = production_runtime_prepare_events(
        KernelPlan((flash_attention, split_qkv)),
        enabled_mechanisms=(
            "triton_three_output_composite_prepare",
            "fused_attention_source_wrapper_prepare",
            "moe_unquant_mlp_wrapper_prepare",
        ),
    )
    assert dense_events == {}

    events = production_runtime_prepare_events(
        KernelPlan((flash_attention, split_qkv, moe_signature)),
        enabled_mechanisms=(
            "triton_three_output_composite_prepare",
            "fused_attention_source_wrapper_prepare",
            "moe_unquant_mlp_wrapper_prepare",
        ),
    )

    assert events == {
        ("main", 0, "dense:attention", -1): (
            RuntimePrepareEvent("fused_attention_source_wrapper_prepare"),
        ),
        ("main", 0, "dense:qkv", -1): (
            RuntimePrepareEvent("triton_three_output_composite_prepare"),
        ),
    }


def test_wrapper_mechanisms_without_source_anchor_emit_no_event():
    gating = KernelCall(
        "dense:gating",
        _spec(KernelKind.VECTOR, "moe_gating_topk"),
        "main",
        0,
        "compute",
        ("moe",),
    )

    events = production_runtime_prepare_events(
        KernelPlan((gating,)),
        enabled_mechanisms=(
            "triton_three_output_composite_prepare",
            "fused_attention_source_wrapper_prepare",
        ),
    )

    assert events == {}


def test_model_runner_outer_events_follow_input_and_logits_source_boundaries():
    calls = tuple(
        KernelCall(
            f"step:token-embedding:tp-rank-{rank}",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            rank,
            "compute",
            ("input",),
        )
        for rank in (0, 1)
    ) + tuple(
            KernelCall(
                f"step:lm-head:tp-rank-{rank}",
                _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            rank,
            "compute",
            ("logits",),
        )
        for rank in (0, 1)
    )

    assert ascend_910b_model_runner_outer_events(KernelPlan(calls)) == {
        ("main", rank, f"step:token-embedding:tp-rank-{rank}", -1): (
            RuntimePrepareEvent("model_runner_input_prepare"),
            RuntimePrepareEvent("model_runner_forward_metadata_prepare"),
        )
        for rank in (0, 1)
    } | {
        ("main", rank, f"step:lm-head:tp-rank-{rank}", -1): (
            RuntimePrepareEvent("model_runner_logits_prepare"),
        )
        for rank in (0, 1)
    }


def test_model_runner_outer_events_map_logical_rank_labels_to_physical_lanes():
    physical_ranks = (91, 7)
    calls = tuple(
        KernelCall(
            f"step:token-embedding:tp-rank-{logical_rank}",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            physical_rank,
            "compute",
            ("input",),
        )
        for logical_rank, physical_rank in enumerate(physical_ranks)
    ) + tuple(
        KernelCall(
            f"step:lm-head:tp-rank-{logical_rank}",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            physical_rank,
            "compute",
            ("logits",),
        )
        for logical_rank, physical_rank in enumerate(physical_ranks)
    )

    events = ascend_910b_model_runner_outer_events(KernelPlan(calls))

    assert set(events) == {
        ("main", physical_rank,
         f"step:lm-head:tp-rank-{logical_rank}", -1)
        for logical_rank, physical_rank in enumerate(physical_ranks)
    } | {
        ("main", physical_rank,
         f"step:token-embedding:tp-rank-{logical_rank}", -1)
        for logical_rank, physical_rank in enumerate(physical_ranks)
    }


def test_model_runner_outer_events_reject_incomplete_rank_anchors():
    plan = KernelPlan((
        KernelCall(
            "step:token-embedding:tp-rank-0",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            0,
            "compute",
            ("input",),
        ),
        KernelCall(
            "step:lm-head:tp-rank-1",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            1,
            "compute",
            ("logits",),
        ),
    ))

    with pytest.raises(ValueError, match="matching token-embedding"):
        ascend_910b_model_runner_outer_events(plan)


def test_model_runner_outer_events_reject_mixed_ranked_anchor_forms():
    calls = (
        KernelCall(
            "step:token-embedding:tp-rank-0",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main", 0, "compute", ("input0",)),
        KernelCall(
            "step:token-embedding",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main", 1, "compute", ("input1",)),
        KernelCall(
            "step0:lm-head",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main", 0, "compute", ("logits0",)),
        KernelCall(
            "step1:lm-head",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main", 1, "compute", ("logits1",)),
    )
    with pytest.raises(ValueError, match="uniformly ranked"):
        ascend_910b_model_runner_outer_events(KernelPlan(calls))


def test_rotary_cache_update_follows_qwen3_model_runner_source_anchor():
    calls = tuple(
        KernelCall(
            f"step:token-embedding:tp-rank-{rank}",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            rank,
            "compute",
            ("input",),
        )
        for rank in (0, 1)
    ) + tuple(
        KernelCall(
            f"step:qkv:tp-rank-{rank}",
            _spec(KernelKind.VECTOR, "qkv_rmsnorm_rope"),
            "main",
            rank,
            "compute",
            ("attention",),
        )
        for rank in (0, 1)
    )

    assert ascend_910b_rotary_cache_update_events(KernelPlan(calls)) == {
        ("main", rank, f"step:token-embedding:tp-rank-{rank}", -1): (
            RuntimePrepareEvent(
                "rotary_cache_update_prepare", graph_capturable=False),
        )
        for rank in (0, 1)
    }


def test_production_rotary_cache_update_closes_source_anchor_contract():
    calls = tuple(
        KernelCall(
            f"step:token-embedding:tp-rank-{rank}",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            rank,
            "compute",
            ("input",),
        )
        for rank in (0, 1)
    ) + tuple(
        KernelCall(
            f"step:qkv:tp-rank-{rank}",
            _spec(KernelKind.VECTOR, "qkv_rmsnorm_rope"),
            "main",
            rank,
            "compute",
            ("attention",),
        )
        for rank in (0, 1)
    ) + (KernelCall(
        "step:rotary",
        _spec(KernelKind.VECTOR, "rotary"),
        "main", 0, "compute", ("rotary",)),)

    assert production_runtime_prepare_events(
        KernelPlan(calls),
        enabled_mechanisms=("rotary_cache_update_prepare",),
    ) == {
        ("main", rank, f"step:token-embedding:tp-rank-{rank}", -1): (
            RuntimePrepareEvent(
                "rotary_cache_update_prepare", graph_capturable=False),
        )
        for rank in (0, 1)
    }


def test_production_rotary_cache_update_accepts_unranked_embedding_source_ids():
    calls = tuple(
        KernelCall(
            f"step:rank-{rank}:token-embedding",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            rank,
            "compute",
            ("input",),
        )
        for rank in (0, 1)
    ) + tuple(
        KernelCall(
            f"step:qkv:tp-rank-{rank}",
            _spec(KernelKind.VECTOR, "qkv_rmsnorm_rope"),
            "main",
            rank,
            "compute",
            ("attention",),
        )
        for rank in (0, 1)
    ) + (KernelCall(
        "step:rotary",
        _spec(KernelKind.VECTOR, "rotary"),
        "main", 0, "compute", ("rotary",)),)

    assert production_runtime_prepare_events(
        KernelPlan(calls),
        enabled_mechanisms=("rotary_cache_update_prepare",),
    ) == {
        ("main", rank, f"step:rank-{rank}:token-embedding", -1): (
            RuntimePrepareEvent(
                "rotary_cache_update_prepare", graph_capturable=False),
        )
        for rank in (0, 1)
    }


def test_production_rotary_cache_update_accepts_non_fused_rope_plan():
    call = KernelCall(
        "step:token-embedding",
        _spec(KernelKind.VECTOR, "elementwise"),
        "main",
        0,
        "compute",
        ("input",),
    )
    rotary = KernelCall(
        "step:rotary", _spec(KernelKind.VECTOR, "rotary"),
        "main", 0, "compute", ("rotary",), (call.call_id,))
    assert production_runtime_prepare_events(
        KernelPlan((call, rotary)),
        enabled_mechanisms=("rotary_cache_update_prepare",),
    ) == {
        ("main", 0, "step:token-embedding", -1): (
            RuntimePrepareEvent(
                "rotary_cache_update_prepare", graph_capturable=False),
        ),
    }


def test_production_rotary_cache_update_rejects_plan_without_rope_call():
    call = KernelCall(
        "step:token-embedding",
        _spec(KernelKind.VECTOR, "elementwise"),
        "main", 0, "compute", ("input",),
    )
    with pytest.raises(ValueError, match="explicit RoPE source call"):
        production_runtime_prepare_events(
            KernelPlan((call,)),
            enabled_mechanisms=("rotary_cache_update_prepare",),
        )


def test_production_rotary_cache_update_rejects_mixed_anchor_forms():
    calls = (
        KernelCall("step:token-embedding:tp-rank-0", _spec(KernelKind.VECTOR, "elementwise"),
                   "main", 0, "compute", ("input0",)),
        KernelCall("step:rank-1:token-embedding", _spec(KernelKind.VECTOR, "elementwise"),
                   "main", 1, "compute", ("input1",)),
        KernelCall("step:rotary", _spec(KernelKind.VECTOR, "rotary"),
                   "main", 0, "compute", ("rotary",)),
    )
    with pytest.raises(ValueError, match="uniformly ranked"):
        production_runtime_prepare_events(
            KernelPlan(calls),
            enabled_mechanisms=("rotary_cache_update_prepare",),
        )


def test_production_rotary_cache_update_rejects_mla_plan():
    calls = (
        KernelCall("step:rank-0:token-embedding", _spec(KernelKind.VECTOR, "elementwise"),
                   "main", 0, "compute", ("input",)),
        KernelCall("step:main-rotary", _spec(KernelKind.VECTOR, "rotary"),
                   "main", 0, "compute", ("rotary",)),
    )
    with pytest.raises(ValueError, match="MLA"):
        production_runtime_prepare_events(
            KernelPlan(calls),
            enabled_mechanisms=("rotary_cache_update_prepare",),
        )


def test_model_runner_outer_events_can_restrict_to_observed_worker_lane():
    calls = tuple(
        KernelCall(
            f"step:token-embedding:tp-rank-{rank}",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            rank,
            "compute",
            ("input",),
        )
        for rank in (0, 1)
    ) + tuple(
        KernelCall(
            f"step:lm-head:tp-rank-{rank}",
            _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            rank,
            "compute",
            ("logits",),
        )
        for rank in (0, 1)
    )

    assert ascend_910b_model_runner_outer_events(
        KernelPlan(calls), enabled_ranks=(1,)) == {
            ("main", 1, "step:token-embedding:tp-rank-1", -1): (
                RuntimePrepareEvent("model_runner_input_prepare"),
                RuntimePrepareEvent("model_runner_forward_metadata_prepare"),
            ),
            ("main", 1, "step:lm-head:tp-rank-1", -1): (
                RuntimePrepareEvent("model_runner_logits_prepare"),
            ),
        }


def test_prepare_occurrences_map_source_moe_mlp_calls(monkeypatch):
    class FakePlan:
        def __init__(self, calls):
            self.calls = calls
            self.block_repeats = ()

    def call(call_id, kind, variant):
        return SimpleNamespace(
            call_id=call_id,
            rank=0,
            kernel_spec=SimpleNamespace(
                hardware_id="ascend_910b", kind=kind, variant=variant,
                shape={"experts": 128}),
        )

    plan = FakePlan((
        call("w1", KernelKind.FUSED_MOE, "w1"),
        call("swiglu", KernelKind.VECTOR, "moe_swiglu"),
        call("w2", KernelKind.FUSED_MOE, "w2"),
        call("combine", KernelKind.VECTOR, "moe_token_unpermute"),
    ))
    monkeypatch.setattr(prepare_module, "KernelPlan", FakePlan)
    monkeypatch.setattr(
        prepare_module, "call_host_lanes", lambda call: (("main", call.rank),))

    events = ascend_910b_runtime_prepare_events(
        plan,
        enabled_mechanisms=(
            "moe_grouped_gemm_prepare",
            "moe_swiglu_prepare",
            "moe_token_combine_prepare",
        ),
    )

    assert events == {
        ("main", 0, "w1", -1): (
            RuntimePrepareEvent("moe_grouped_gemm_prepare"),
        ),
        ("main", 0, "swiglu", -1): (
            RuntimePrepareEvent("moe_swiglu_prepare"),),
        ("main", 0, "w2", -1): (
            RuntimePrepareEvent("moe_grouped_gemm_prepare"),
        ),
        ("main", 0, "combine", -1): (
            RuntimePrepareEvent("moe_token_combine_prepare"),),
    }


def test_unquant_mlp_wrapper_uses_only_w1_source_anchor(monkeypatch):
    class FakePlan:
        def __init__(self, calls):
            self.calls = calls
            self.block_repeats = ()

    def call(call_id, variant):
        return SimpleNamespace(
            call_id=call_id,
            rank=1,
            kernel_spec=SimpleNamespace(
                hardware_id="ascend_910b",
                kind=KernelKind.FUSED_MOE,
                variant=variant,
            ),
        )

    plan = FakePlan((call("w1", "w1"), call("w2", "w2")))
    monkeypatch.setattr(prepare_module, "KernelPlan", FakePlan)
    monkeypatch.setattr(
        prepare_module, "call_host_lanes", lambda call: (("main", call.rank),))

    events = ascend_910b_unquant_mlp_wrapper_events(plan)

    assert events == {
        ("main", 1, "w1", -1): (
            RuntimePrepareEvent("moe_unquant_mlp_wrapper_prepare"),)
    }


def test_prepare_occurrences_map_unquantized_gemm_backend(monkeypatch):
    class FakePlan:
        def __init__(self, calls):
            self.calls = calls
            self.block_repeats = ()

    gemm = SimpleNamespace(
        call_id="qkv",
        rank=1,
        kernel_spec=SimpleNamespace(
            hardware_id="ascend_910b",
            kind=KernelKind.GEMM,
            variant="gemm",
        ),
    )
    monkeypatch.setattr(prepare_module, "KernelPlan", FakePlan)
    monkeypatch.setattr(
        prepare_module, "call_host_lanes", lambda call: (("main", call.rank),))

    events = ascend_910b_runtime_prepare_events(
        FakePlan((gemm,)),
        enabled_mechanisms=("unquantized_gemm_prepare",),
    )

    assert events == {
        ("main", 1, "qkv", -1): (
            RuntimePrepareEvent("unquantized_gemm_prepare"),),
    }


def test_prepare_occurrences_map_source_moe_stream_events(monkeypatch):
    class FakePlan:
        def __init__(self, calls):
            self.calls = calls
            self.block_repeats = ()

    def call(call_id, variant):
        return SimpleNamespace(
            call_id=call_id,
            rank=0,
            kernel_spec=SimpleNamespace(
                hardware_id="ascend_910b", kind=KernelKind.VECTOR,
                variant=variant),
        )

    plan = FakePlan((
        call("gating", "moe_gating_topk"),
        call("dispatch", "moe_init_routing_v2"),
        call("combine", "moe_token_unpermute"),
    ))
    monkeypatch.setattr(prepare_module, "KernelPlan", FakePlan)
    monkeypatch.setattr(
        prepare_module, "call_host_lanes", lambda call: (("main", call.rank),))

    events = ascend_910b_runtime_prepare_events(
        plan,
        enabled_mechanisms=("stream_record_event_prepare",),
    )

    assert events == {
        ("main", 0, "gating", -1): (
            RuntimePrepareEvent("stream_record_event_prepare"),),
        ("main", 0, "dispatch", -1): (
            RuntimePrepareEvent("stream_record_event_prepare"),),
        ("main", 0, "combine", -1): (
            RuntimePrepareEvent("stream_record_event_prepare"),),
    }


def test_prepare_occurrences_map_collective_call_stack():
    all_reduce = KernelCall(
        "all-reduce",
        KernelSpec(
            KernelKind.COMMUNICATION,
            "ascend_910b",
            transfer=PhysicalTransferConfig(
                algorithm="ring",
                bytes=4096,
                direction="collective",
                link="infiniband",
                topology="ring",
                participants=(0, 1),
                ring_order=(0, 1),
                operation="all_reduce",
                bytes_semantics="per_rank_input",
                reduction="sum",
                backend="analytical_roofline_v1",
                resource_id="hccs:fixture",
            ),
        ),
        "main",
        0,
        "hccs:fixture",
        ("collective",),
    )
    plan = KernelPlan((all_reduce,))

    events = ascend_910b_runtime_prepare_events(
        plan,
        enabled_mechanisms=("collective_all_reduce_prepare",),
    )

    assert events == {
        ("main", 0, "all-reduce", -1): (
            RuntimePrepareEvent("collective_all_reduce_prepare"),
        ),
        ("main", 1, "all-reduce", -1): (
            RuntimePrepareEvent("collective_all_reduce_prepare"),
        ),
    }


def test_moe_wrapper_events_follow_compact_layer_anchor_repeats():
    calls = tuple(
        KernelCall(
            f"step:layer-{layer}:moe:ep-rank-{rank}:0",
            # The descriptor is keyed by the compact MoE source anchor, not
            # by the child kernel variant.  Keep this fixture on the generic
            # vector contract so it does not duplicate CANN MoeSwiGLU launch
            # validation details unrelated to anchor placement.
            _spec(KernelKind.VECTOR, "elementwise"),
            "main",
            rank,
            "compute",
            ("moe",),
        )
        for layer in (0, 1)
        for rank in (0, 1)
    )
    plan = KernelPlan(
        calls,
        (KernelBlockRepeat(
            "layer-one",
            tuple(call.call_id for call in calls if ":layer-1:" in call.call_id),
            3,
            (),
        ),),
    )

    events = ascend_910b_moe_wrapper_control_events(plan)

    assert events == {
        ("main", rank, f"step:layer-0:moe:ep-rank-{rank}:0", -1): (
            RuntimePrepareEvent("compiled_fx_moe_wrapper_control"),)
        for rank in (0, 1)
    } | {
        ("main", rank, f"step:layer-1:moe:ep-rank-{rank}:0", occurrence): (
            RuntimePrepareEvent("compiled_fx_moe_wrapper_control"),)
        for rank in (0, 1)
        for occurrence in range(3)
    }
