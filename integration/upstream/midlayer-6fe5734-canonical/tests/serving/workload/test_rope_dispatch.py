"""Source-pinned vLLM RoPE dispatch and its native-composite expansion."""

import collections

import pytest

from src.serving.workload.model_config import load_model_config
from src.serving.workload.rope_dispatch import (
    CAT_SINGLE_INPUT_TENSOR_MOVE_NUMEL,
    ROPE_PROGRAM_FUSED,
    ROPE_PROGRAM_NATIVE_COMPOSITE,
    modelled_rope_program,
    native_composite_rope_calls,
    rope_composite_semantics,
    vllm_rope_program,
)

ACLNN = "aclnn_materialized_operands"
ATEN = "aten_tensor_iterator"


def _aclnn(tokens, query_heads=32, kv_heads=8, head_dim=128):
    return native_composite_rope_calls(
        tokens, query_heads, kv_heads, head_dim, ACLNN)


def _aten(tokens, query_heads=32, kv_heads=8, head_dim=128):
    return native_composite_rope_calls(
        tokens, query_heads, kv_heads, head_dim, ATEN)


def test_out_of_tree_registry_decides_the_rope_program_not_the_model():
    # vllm-ascend registers "RotaryEmbedding" but not "Llama3RotaryEmbedding",
    # and CustomOp.__new__ keys op_registry_oot on the exact class name.
    assert vllm_rope_program("ascend_910b", "default") == ROPE_PROGRAM_FUSED
    assert vllm_rope_program("ascend_910b", "yarn") == ROPE_PROGRAM_FUSED
    assert vllm_rope_program("ascend_910b", "deepseek_yarn") == (
        ROPE_PROGRAM_FUSED)
    assert vllm_rope_program("ascend_910b", "llama3") == (
        ROPE_PROGRAM_NATIVE_COMPOSITE)
    # vllm-metax registers XDRotaryEmbedding instead of the YaRN class, so the
    # two out-of-tree runtimes genuinely disagree for the same model.
    assert vllm_rope_program("metax_c500", "xdrope") == ROPE_PROGRAM_FUSED
    assert vllm_rope_program("metax_c500", "yarn") == (
        ROPE_PROGRAM_NATIVE_COMPOSITE)
    # In-tree CUDA reaches forward_cuda through inheritance, never a registry.
    for scaling_type in ("default", "llama3", "yarn", "linear", "dynamic"):
        assert vllm_rope_program("nvidia_a800", scaling_type) == (
            ROPE_PROGRAM_FUSED)


def test_unknown_hardware_and_scaling_type_fail_closed():
    with pytest.raises(ValueError):
        vllm_rope_program("unknown_hardware", "default")
    with pytest.raises(ValueError):
        vllm_rope_program("ascend_910b", "not_a_vllm_rope_type")


def test_both_out_of_tree_runtimes_lower_the_composite_they_run():
    assert modelled_rope_program("metax_c500", "llama3") == (
        ROPE_PROGRAM_NATIVE_COMPOSITE, None)
    assert modelled_rope_program("ascend_910b", "llama3") == (
        ROPE_PROGRAM_NATIVE_COMPOSITE, None)
    assert rope_composite_semantics("ascend_910b") == ACLNN
    assert rope_composite_semantics("metax_c500") == ATEN
    assert modelled_rope_program("ascend_910b", "default") == (
        ROPE_PROGRAM_FUSED, None)


def _families(calls):
    counts = collections.Counter()
    for call in calls:
        if call.operation == "gather":
            counts["gather"] += 1
        elif call.name.endswith("-mul"):
            counts["mul"] += 1
        elif call.name.endswith("-sub"):
            counts["sub"] += 1
        elif call.name.endswith("-add"):
            counts["add"] += 1
        elif call.name.endswith("-rotate-concat"):
            counts["concat"] += 1
        elif call.name.endswith("-passthrough-concat"):
            counts["cat_tensor_move"] += 1
        elif call.name.endswith("-operand-slice"):
            counts["slice_qkv"] += 1
        else:
            counts["slice_cos_sin"] += 1
    return counts


def test_composite_matches_the_captured_per_call_operator_manifest():
    # Llama-3.1-8B TP1 geometry; counts are the ones reconciled against the
    # three ShareGPT ascend_910b op_profile captures.
    assert _families(_aclnn(16)) == collections.Counter({
        "gather": 1, "slice_qkv": 8, "slice_cos_sin": 8, "mul": 8,
        "sub": 2, "add": 2, "concat": 2,
    })
    # ATen's TensorIterator consumes the chunk views in place, so the same
    # source program is 15 kernels rather than 31 and carries no Slice.
    assert _families(_aten(16)) == collections.Counter({
        "gather": 1, "mul": 8, "sub": 2, "add": 2, "concat": 2,
    })
    assert _families(_aten(1)) == _families(_aten(8192))


def test_single_token_step_drops_the_contiguous_cos_sin_slices():
    # cos_sin.chunk(2, -1) is already contiguous when the only outer dimension
    # has extent one, so CANN launches no Slice for it.
    counts = _families(_aclnn(1))
    assert counts["slice_cos_sin"] == 0
    assert counts["slice_qkv"] == 8
    assert counts["mul"] == 8


def test_empty_passthrough_concat_appears_only_above_the_cann_threshold():
    head_dim = 128
    for heads in (32, 8):
        boundary = CAT_SINGLE_INPUT_TENSOR_MOVE_NUMEL // (heads * head_dim)
        below = _families(_aclnn(boundary, head_dim=head_dim))
        above = _families(_aclnn(boundary + 1, head_dim=head_dim))
        assert above["cat_tensor_move"] == below["cat_tensor_move"] + 1
    assert _families(_aclnn(8192, head_dim=head_dim))["cat_tensor_move"] == 2
    # ATen returns the already materialized rotate output, at every size.
    assert _families(_aten(8192, head_dim=head_dim))["cat_tensor_move"] == 0


def test_composite_elements_follow_the_rotate_half_shapes():
    tokens, query_heads, kv_heads, head_dim = 7, 32, 8, 128
    calls = {
        call.name: call
        for call in _aclnn(tokens, query_heads, kv_heads, head_dim)
    }
    assert calls["cos-sin-gather"].elements == tokens * head_dim
    assert calls["q-x1-cos-mul"].elements == (
        tokens * query_heads * head_dim // 2)
    assert calls["k-x1-cos-mul"].elements == tokens * kv_heads * head_dim // 2
    assert calls["q-x1-cos-cos-slice"].elements == tokens * head_dim // 2
    assert calls["q-rotate-concat"].elements == tokens * query_heads * head_dim


def test_composite_rope_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        _aclnn(0)
    with pytest.raises(ValueError):
        _aclnn(8, head_dim=127)
    with pytest.raises(TypeError):
        _aclnn(True)
    with pytest.raises(ValueError):
        native_composite_rope_calls(8, 32, 8, 128, "not_a_runtime_semantics")


def test_repository_model_configs_carry_the_source_scaling_type():
    llama = load_model_config("configs/model/llama-3.1-8b-instruct.json")
    assert llama.rope_scaling_type == "llama3"
    qwen = load_model_config("configs/model/qwen3-8b.json")
    assert qwen.rope_scaling_type == "default"


def test_c500_rotate_half_multiply_launches_follow_the_pinned_maca_chain():
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.launch_config.select_launch_config import (
        select_launch_config,
    )
    from src.kernel.hardware_spec import HWSpec

    hardware = HWSpec("configs/hardware/metax_c500.yaml")

    def launch(tokens, rows, inner):
        return select_launch_config(
            "metax_c500", KernelKind.VECTOR,
            {"element_count": tokens * rows * inner},
            "bf16", "elementwise",
            {
                "operation": "mul",
                "reads": 2,
                "writes": 1,
                "launch_policy": "injected_launch",
                "operand_layout": "dim0_contiguous_dim1_broadcast",
                "iteration_tokens": tokens,
                "iteration_rows": rows,
                "iteration_inner": inner,
            },
            hardware=hardware,
        )

    # numel >= 5120 takes launch_legacy_kernel_maca_3_2_dim0_contiguous with
    # nt=128 and vec=4, because the innermost extent is 64 (< 128).
    assert launch(11, 8, 64).grid == (11, 1, 1)
    assert launch(11, 32, 64).grid == (44, 1, 1)
    assert launch(11, 8, 64).block == (128, 1, 1)
    # numel < 5120 falls through to the dim1-broadcast launcher, whose vec is
    # pinned to 4 and whose grid divides by nt * vec * vec.
    assert launch(4, 8, 64).grid == (1, 1, 1)
    # A single token collapses the iterator to two dimensions.
    single = launch(1, 32, 64)
    assert single.grid == (1, 32, 1)
    assert single.block == (64, 1, 1)


def test_c500_rotate_half_concat_uses_the_cat_array_program():
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.launch_config.select_launch_config import (
        select_launch_config,
    )
    from src.kernel.hardware_spec import HWSpec

    launch = select_launch_config(
        "metax_c500", KernelKind.VECTOR,
        {"element_count": 23 * 8 * 128},
        "bf16", "elementwise",
        {
            "operation": "copy",
            "reads": 1,
            "writes": 1,
            "launch_policy": "injected_launch",
            "input_materialization": "concatenated_last_dim",
        },
        hardware=HWSpec("configs/hardware/metax_c500.yaml"),
    )
    assert launch.grid == (23, 1, 1)
    assert launch.block == (128, 1, 1)
