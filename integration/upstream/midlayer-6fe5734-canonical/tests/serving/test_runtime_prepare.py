import pytest

from collections import OrderedDict
from functools import partial

from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    PAGED_KV_CACHE_ATTRIBUTES,
    PhysicalTransferConfig,
)
from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.serving.execution import _cached_runtime_prepare_events
from src.serving.host_timing import RuntimePrepareEvent
from src.serving.kernel_lowering.execution_timeline import (
    _common_id_prefix,
    _normalize_id,
    _plan_structure_key,
)
from src.serving.kernel_lowering.kernel_plan import (
    KernelBlockRepeat,
    KernelCall,
    KernelPlan,
)
from src.serving.runtime_prepare import runtime_prepare_events


def _spec(kind, variant, hardware_id="nvidia_a800"):
    shape = (
        {
            "tokens": 16,
            "experts": 16,
            "topk": 2,
            "hidden": 64,
            "intermediate": 128,
        }
        if kind in (KernelKind.FUSED_MOE,) or variant.startswith("moe_")
        else {
            "b": 1,
            "m": 16,
            "n": 64,
            "k": 64,
        }
        if kind is KernelKind.GEMM
        else {
            "seq_len": 16,
            "num_kv_heads": 2,
            "head_dim": 128,
        }
        if kind is KernelKind.KV_CACHE
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
        dict(PAGED_KV_CACHE_ATTRIBUTES)
        if kind is KernelKind.KV_CACHE
        else {
            "gather_factor": 1,
            "index_bytes": 8,
            "index_count": 16,
            "scattered": False,
        }
        if variant == "gather"
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
        else {
            "tokens_per_expert": (2,) * 16,
            "local_tokens_per_expert": (2,) * 16,
            "expert_parallel_size": 1,
            "expert_rank": 0,
            "owned_experts": tuple(range(16)),
            "expert_weight_tile_rows": (1,) * 16,
        }
        if kind is KernelKind.FUSED_MOE
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
        else {
            "group_size_m": 1.0,
            "padded_rows": 32.0,
            "mte2_tile_bytes": 1.0,
            "weight_tile_rows": 1.0,
            "operand_replay_a": 1.0,
            "operand_replay_b": 1.0,
        }
        if kind is KernelKind.FUSED_MOE
        else {
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 0.0,
            "split_k_partial_dram_bytes_per_element": 0.0,
        }
        if kind is KernelKind.GEMM
        else {}
    )
    tiling = (
        TilingSpec(tile_shape={"m": 16, "n": 16, "k": 16})
        if kind is KernelKind.FUSED_MOE
        else None
    )
    return KernelSpec(
        kind,
        hardware_id,
        shape,
        dtype="bf16",
        variant=variant,
        attributes=attributes,
        launch=LaunchConfig(
            algorithm="fixture",
            grid=(1, 1, 1),
            block=(1, 1, 1),
            tiling=tiling,
            parameters=parameters,
            execution_variant=variant,
        ),
    )


def _call(call_id, kind, variant, hardware_id="nvidia_a800", rank=0):
    return KernelCall(
        call_id,
        _spec(kind, variant, hardware_id),
        "main",
        rank,
        "compute",
        (call_id,),
    )


def test_runtime_prepare_prepare_occurrences_follow_source_variants_and_repeats():
    gating = _call("gating", KernelKind.VECTOR, "moe_gating_topk")
    experts = _call("experts", KernelKind.FUSED_MOE, "w1")
    plan = KernelPlan(
        (gating, experts),
        (KernelBlockRepeat("layers", ("gating", "experts"), 3, ()),),
    )

    events = runtime_prepare_events(
        plan,
        enabled_mechanisms=(
            "compiled_fx_backend_entry",
            "moe_gating_prepare",
            "compiled_fx_moe_wrapper_control",
            "moe_unquant_mlp_wrapper_prepare",
        ),
    )

    assert events == {
        ("main", 0, "gating", occurrence): (
            RuntimePrepareEvent("compiled_fx_backend_entry"),
            RuntimePrepareEvent("moe_gating_prepare"),
        )
        for occurrence in range(3)
    } | {
        ("main", 0, "experts", occurrence): (
            RuntimePrepareEvent("compiled_fx_moe_wrapper_control"),
            RuntimePrepareEvent("moe_unquant_mlp_wrapper_prepare"),
        )
        for occurrence in range(3)
    }


def test_runtime_prepare_embedding_anchors_first_compiled_piece_entry():
    plan = KernelPlan((
        _call("req:token-embedding:tp-rank-0", KernelKind.VECTOR, "gather"),
        _call("gating-0", KernelKind.VECTOR, "moe_gating_topk"),
        _call("gating-1", KernelKind.VECTOR, "moe_gating_topk"),
        _call("indexer-topk-reuse", KernelKind.VECTOR, "gather"),
    ))

    events = runtime_prepare_events(
        plan, enabled_mechanisms=("compiled_fx_backend_entry",))

    assert events == {
        ("main", 0, "req:token-embedding:tp-rank-0", -1): (
            RuntimePrepareEvent("compiled_fx_backend_entry"),),
        ("main", 0, "gating-0", -1): (
            RuntimePrepareEvent("compiled_fx_backend_entry"),),
        ("main", 0, "gating-1", -1): (
            RuntimePrepareEvent("compiled_fx_backend_entry"),),
    }


def test_runtime_prepare_prepare_occurrences_cover_gemm_kv_and_attention():
    plan = KernelPlan((
        _call("qkv", KernelKind.GEMM, "gemm"),
        _call("kv", KernelKind.KV_CACHE, "paged_kv_cache"),
        _call("prefill", KernelKind.FLASH_ATTN, "prefill"),
        _call("decode", KernelKind.FLASH_ATTN, "decode"),
    ))

    events = runtime_prepare_events(
        plan,
        enabled_mechanisms=(
            "unquantized_gemm_prepare",
            "reshape_cache_prepare",
            "fused_attention_source_wrapper_prepare",
        ),
    )

    assert events == {
        ("main", 0, "qkv", -1): (
            RuntimePrepareEvent("unquantized_gemm_prepare"),),
        ("main", 0, "kv", -1): (
            RuntimePrepareEvent("reshape_cache_prepare"),),
        ("main", 0, "prefill", -1): (
            RuntimePrepareEvent("fused_attention_source_wrapper_prepare"),),
        ("main", 0, "decode", -1): (
            RuntimePrepareEvent("fused_attention_source_wrapper_prepare"),),
    }


def test_runtime_prepare_moe_w2_carries_no_wrapper_event():
    plan = KernelPlan((
        _call("w1", KernelKind.FUSED_MOE, "w1"),
        _call("w2", KernelKind.FUSED_MOE, "w2"),
    ))

    events = runtime_prepare_events(
        plan,
        enabled_mechanisms=(
            "compiled_fx_moe_wrapper_control",
            "moe_unquant_mlp_wrapper_prepare",
        ),
    )

    assert events == {
        ("main", 0, "w1", -1): (
            RuntimePrepareEvent("compiled_fx_moe_wrapper_control"),
            RuntimePrepareEvent("moe_unquant_mlp_wrapper_prepare"),
        ),
    }


def test_runtime_prepare_prepare_occurrences_filter_unmeasured_mechanisms():
    plan = KernelPlan((
        _call("gating", KernelKind.VECTOR, "moe_gating_topk"),
    ))

    events = runtime_prepare_events(
        plan,
        enabled_mechanisms=("moe_gating_prepare",),
    )

    assert events == {
        ("main", 0, "gating", -1): (
            RuntimePrepareEvent("moe_gating_prepare"),),
    }


def test_runtime_prepare_occurrences_reject_hardware_without_rule_set():
    plan = KernelPlan((
        _call("gating", KernelKind.VECTOR, "moe_gating_topk", "groq_lpu"),
    ))

    with pytest.raises(ValueError, match="no rule set"):
        runtime_prepare_events(
            plan,
            enabled_mechanisms=("moe_gating_prepare",),
        )


def test_runtime_prepare_prepare_occurrences_reject_mixed_hardware():
    plan = KernelPlan((
        _call("gating", KernelKind.VECTOR, "moe_gating_topk", "nvidia_a800"),
        _call("qkv", KernelKind.GEMM, "gemm", "metax_c500"),
    ))

    with pytest.raises(ValueError, match="one hardware"):
        runtime_prepare_events(
            plan,
            enabled_mechanisms=("moe_gating_prepare",),
        )


def test_runtime_prepare_prepare_occurrences_support_metax_c500():
    plan = KernelPlan((
        _call("gating", KernelKind.VECTOR, "moe_gating_topk", "metax_c500"),
    ))

    events = runtime_prepare_events(
        plan,
        enabled_mechanisms=("moe_gating_prepare",),
    )

    assert events == {
        ("main", 0, "gating", -1): (
            RuntimePrepareEvent("moe_gating_prepare"),),
    }


def _collective_call(operation, hardware_id="nvidia_a800"):
    return KernelCall(
        operation,
        KernelSpec(
            KernelKind.COMMUNICATION,
            hardware_id,
            transfer=PhysicalTransferConfig(
                algorithm="ring",
                bytes=4096,
                direction="collective",
                link="nvlink",
                topology="ring",
                participants=(0, 1),
                ring_order=(0, 1),
                operation=operation,
                bytes_semantics="per_rank_input",
                reduction="sum" if operation == "all_reduce" else None,
                backend="analytical_roofline_v1",
                resource_id="nvlink:fixture",
            ),
        ),
        "main",
        0,
        "nvlink:fixture",
        ("collective",),
    )


@pytest.mark.parametrize(
    "operation,mechanism",
    (
        ("all_reduce", "collective_all_reduce_prepare"),
        ("all_gather", "collective_all_gather_prepare"),
    ),
)
def test_runtime_prepare_prepare_occurrences_map_collective_call_stack(
        operation, mechanism):
    plan = KernelPlan((_collective_call(operation),))

    events = runtime_prepare_events(
        plan,
        enabled_mechanisms=(mechanism,),
    )

    assert events == {
        ("main", rank, operation, -1): (RuntimePrepareEvent(mechanism),)
        for rank in (0, 1)
    }


def _step_prefixed_plan(prefix, lm_head_variant):
    return KernelPlan((
        _call(f"{prefix}qkv", KernelKind.GEMM, "gemm"),
        _call(f"{prefix}lm-head", KernelKind.GEMM, lm_head_variant),
    ))


def _structure_key(plan):
    prefix = _common_id_prefix(plan)
    return _plan_structure_key(
        plan.topological_order(), plan.block_repeats, prefix)


def test_runtime_prepare_events_cache_distinguishes_variant_across_prefixes():
    """Identical plan structure with a gemv->gemm flip under a new step-id
    prefix must miss the events cache and rebuild."""
    builder = partial(
        runtime_prepare_events,
        enabled_mechanisms=("unquantized_gemm_prepare",))
    gemv_plan = _step_prefixed_plan("main:vllm:7:", "gemv")
    gemm_plan = _step_prefixed_plan("main:vllm:8:", "gemm")
    gemv_events = builder(gemv_plan)
    gemm_events = builder(gemm_plan)
    assert gemv_events != gemm_events
    assert _structure_key(gemv_plan) != _structure_key(gemm_plan)

    cache = OrderedDict()
    events, normalized, token = _cached_runtime_prepare_events(
        cache, builder, gemv_plan, True)
    assert events == gemv_events
    assert normalized is None and token is None
    events, normalized, token = _cached_runtime_prepare_events(
        cache, builder, gemm_plan, True)
    assert events == gemm_events
    assert normalized is None and token is None


def test_runtime_prepare_events_cache_remaps_identical_structure_across_prefixes():
    """Same normalized structure under a new step-id prefix hits the cache
    and remaps to the current prefix, matching a fresh builder call."""
    builder = partial(
        runtime_prepare_events,
        enabled_mechanisms=("unquantized_gemm_prepare",))
    first = _step_prefixed_plan("main:vllm:7:", "gemm")
    second = _step_prefixed_plan("main:vllm:8:", "gemm")
    assert _structure_key(first) == _structure_key(second)

    cache = OrderedDict()
    events_first, _, _ = _cached_runtime_prepare_events(
        cache, builder, first, True)
    assert events_first == builder(first)
    events_second, _, _ = _cached_runtime_prepare_events(
        cache, builder, second, True)
    assert events_second == builder(second)

    _, normalized_first, token_first = _cached_runtime_prepare_events(
        cache, builder, first, False)
    _, normalized_second, token_second = _cached_runtime_prepare_events(
        cache, builder, second, False)
    prefix = _common_id_prefix(first)
    assert normalized_first == {
        ("main", 0, _normalize_id(call_id, prefix), -1): mechanisms
        for (_, _, call_id, _), mechanisms in builder(first).items()
    }
    assert normalized_second is normalized_first
    assert token_second is token_first
