import ast
import dataclasses
import importlib
import inspect
from pathlib import Path

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.hardware_ids import canonical_hardware_id
from src.kernel.kernel_base import create_predictor, predict, registered_kernel_kinds
from src.kernel.kernel_spec import (
    KernelKind,
    KernelSpec,
    PAGED_KV_CACHE_ATTRIBUTES,
)
from src.kernel.launch_config import LaunchConfig
from src.kernel.launch_config import TilingSpec
from src.kernel.launch_config.select_launch_config import (
    select_launch_config as _select_launch_config,
)
from src.kernel.launch_config.select_launch_config import (
    _HARDWARE_PACKAGES,
    _KERNEL_SELECTORS,
)
from src.kernel.model_contract import (
    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
    PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS,
    SOURCE_PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_PARAMETER,
    SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER,
    SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER,
    SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER,
    SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER,
    SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER,
    TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR,
    WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR,
)
from src.kernel.predictors.vector import cann_vector_observed_aiv_floor_ns


def select_launch_config(
        hardware_id, kernel_kind, shape, dtype, variant, attributes,
        hardware):
    """Use injected launches for source-identity predictor unit tests."""

    normalized_attributes = dict(attributes)
    if "launch_policy" not in normalized_attributes:
        normalized_attributes["launch_policy"] = "injected_launch"
    if (kernel_kind in ("GEMM", KernelKind.GEMM)
            and normalized_attributes):
        if "position" not in normalized_attributes:
            normalized_attributes["position"] = "qkv_proj"
        elif normalized_attributes["position"] == "generic":
            normalized_attributes["position"] = "qkv_proj"
        if "has_bias" not in normalized_attributes:
            normalized_attributes["has_bias"] = False
        has_bias = normalized_attributes["has_bias"]
        if "tensor_parallel_size" not in normalized_attributes:
            normalized_attributes["tensor_parallel_size"] = 1
        if "expert_parallel_size" not in normalized_attributes:
            normalized_attributes["expert_parallel_size"] = 1
        normalized_attributes.update({
            "a_layout": "row_major",
            "b_layout": "row_major",
            "c_layout": "row_major",
            "trans_a": False,
            "trans_b": True,
            "epilogue": "bias" if has_bias else "none",
        })
    return _select_launch_config(
        hardware_id, kernel_kind, shape, dtype, variant,
        normalized_attributes, hardware)


def test_every_kind_has_exactly_one_registered_predictor():
    assert registered_kernel_kinds() == frozenset(KernelKind)
    classes = [type(create_predictor(kind)) for kind in KernelKind]
    assert len(set(classes)) == len(KernelKind)


def test_unsupported_kernel_fails_before_performance_modeling():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    kernel = KernelSpec(
        KernelKind.KV_CACHE,
        hardware.hardware_id,
        {"element_count": 30, "seq_len": 3, "compressed_dim": 10},
        dtype="bf16",
        variant="mla_kv_cache",
        attributes={
            "cache_layout": "paged_mla",
            "cache_write_semantics": "append",
        },
        execution_support="unsupported",
        unsupported_reason="launch_selector_unavailable",
    )

    with pytest.raises(
            NotImplementedError,
            match=(
                "KV_CACHE/mla_kv_cache unsupported: "
                "launch_selector_unavailable")):
        predict(kernel, hardware, "midlayer")


def test_kimi_mla_decode_predictor_counts_triton_split_scratch():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {
        "batch": 1,
        "q_len": 1,
        "kv_len": 4096,
        "num_query_heads": 64,
        "qk_head_dim": 576,
        "v_head_dim": 512,
        "kv_lora_rank": 512,
    }
    attributes = {
        "causal": True,
        "kv_layout": "mla",
        "kv_dist": "uniform",
        "q_lens": (1,),
        "kv_lens": (4096,),
        "full_cuda_graphs": False,
        "max_cudagraph_capture_size": 0,
    }
    launch = select_launch_config(
        hardware.hardware_id,
        KernelKind.FLASH_ATTN,
        shape,
        "bf16",
        "mla_decode",
        attributes,
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.FLASH_ATTN,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="mla_decode",
        launch=launch,
        attributes=attributes,
    )

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")

    scratch_bytes = 64 * 8 * (512 + 1) * 4
    input_bytes = (64 * 576 + 4096 * 576) * 2
    output_bytes = 64 * 512 * 2
    assert launch.tiling.num_kv_splits == 8
    assert resource_input.demand_per_resource["l2_cache"] == (
        input_bytes + scratch_bytes)
    assert resource_input.demand_per_resource["l2_cache"] != (
        input_bytes + output_bytes + 2 * scratch_bytes)
    assert resource_input.demand_per_resource["compute_tensor"] == (
        2 * 64 * 4096 * (576 + 512))
    assert prediction.t_total_s > 0.0


def test_kimi_routed_experts_predictor_keeps_packed_int4_and_bf16_scales():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {
        "tokens": 1,
        "experts": 384,
        "topk": 8,
        "hidden": 7168,
        "intermediate": 2048,
    }
    counts = (1,) * 8 + (0,) * (384 - 8)
    attributes = {
        "tokens_per_expert": counts,
        "local_tokens_per_expert": counts,
        "expert_parallel_size": 1,
        "expert_rank": 0,
        "owned_experts": tuple(range(384)),
        "expert_weight_tile_rows": None,
        "routed_scaling_factor": 2.827,
        "routed_weight_dtype": "int4",
        "weight_group_size": 32,
    }
    launch = select_launch_config(
        hardware.hardware_id,
        KernelKind.FUSED_MOE,
        shape,
        "bf16",
        "routed_experts",
        attributes,
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.FUSED_MOE,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="routed_experts",
        launch=launch,
        attributes=attributes,
    )

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")
    weight_path, = tuple(
        path for path in resource_input.memory_paths
        if path.name == "weight_read")
    l2_node, = tuple(
        node for node in weight_path.nodes
        if node.resource_role == "l2_cache")

    packed_weight_bytes = 3 * 7168 * 2048 * 4 / 8
    bf16_scale_bytes = (
        2 * 2048 * (7168 // 32) + 7168 * (2048 // 32)) * 2
    assert l2_node.bytes == 8 * (packed_weight_bytes + bf16_scale_bytes)
    assert resource_input.demand_per_resource["compute_tensor"] == (
        6 * 64 * 7168 * 2048)
    assert prediction.t_total_s > 0.0


def test_kimi_routed_experts_predictor_accepts_rank_local_ep8_rows():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {
        "tokens": 2,
        "experts": 384,
        "topk": 8,
        "hidden": 7168,
        "intermediate": 2048,
    }
    global_counts = (
        (1,) * 8 + (0,) * 40 + (1,) * 8 + (0,) * (384 - 56))
    local_counts = (1,) * 8 + (0,) * (384 - 8)
    attributes = {
        "tokens_per_expert": global_counts,
        "local_tokens_per_expert": local_counts,
        "expert_parallel_size": 8,
        "expert_rank": 0,
        "owned_experts": tuple(range(48)),
        "expert_weight_tile_rows": None,
        "routed_scaling_factor": 2.827,
        "routed_weight_dtype": "int4",
        "weight_group_size": 32,
    }
    launch = select_launch_config(
        hardware.hardware_id,
        KernelKind.FUSED_MOE,
        shape,
        "bf16",
        "routed_experts",
        attributes,
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.FUSED_MOE,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="routed_experts",
        launch=launch,
        attributes=attributes,
    )

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert launch.parameters["expert_parallel_size"] == 8.0
    assert resource_input.demand_per_resource["compute_tensor"] > 0
    assert predict(kernel, hardware, "midlayer").t_total_s > 0.0


def test_unknown_contract_discriminants_fail_loudly():
    with pytest.raises(ValueError, match="unknown kernel kind"):
        KernelKind.parse("future_accelerator_kernel")
    with pytest.raises(ValueError, match="unknown hardware_id"):
        canonical_hardware_id("future_vendor_x1")

    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR,
        {"rows": 1, "hidden": 4096}, "bf16", "rmsnorm",
        {},
        hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id,
        {"rows": 1, "hidden": 4096}, dtype="bf16",
        variant="rmsnorm", launch=launch)
    with pytest.raises(ValueError, match="unknown prediction method"):
        predict(kernel, hardware, "future_method")


def test_hardware_launch_selectors_share_one_module_and_call_signature():
    expected_parameters = ("shape", "dtype", "variant", "attributes", "hardware")
    for package in _HARDWARE_PACKAGES.values():
        for module_name, function_name in _KERNEL_SELECTORS.values():
            module = importlib.import_module(f"{package}.{module_name}")
            selector = getattr(module, function_name)
            assert tuple(inspect.signature(selector).parameters) == expected_parameters


@pytest.mark.parametrize(
    ("hardware_id", "observed_algorithm"),
    (
        ("nvidia_a800", "observed_gemm:gemm"),
        ("metax_c500", "observed_mcblas_gemm:gemm"),
        ("ascend_910b", "observed_gemm:gemm"),
    ),
)
def test_gemm_prediction_is_identical_for_physical_launch_aliases(
        hardware_id, observed_algorithm):
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    shape = {"b": 1, "m": 128, "n": 4096, "k": 4096}
    attributes = {
        "has_bias": False,
        "position": "qkv_proj",
        "launch_policy": "injected_launch",
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "none",
    }
    selected = _select_launch_config(
        hardware_id, KernelKind.GEMM, shape, "bf16", "gemm", attributes,
        hardware)
    source_kernel = KernelSpec(
        KernelKind.GEMM, hardware_id, shape, dtype="bf16", variant="gemm",
        launch=selected)
    observed_kernel = KernelSpec(
        KernelKind.GEMM, hardware_id, shape, dtype="bf16", variant="gemm",
        launch=dataclasses.replace(selected, algorithm=observed_algorithm))

    assert source_kernel.launch == observed_kernel.launch
    assert predict(source_kernel, hardware, "midlayer") == predict(
        observed_kernel, hardware, "midlayer")


@pytest.mark.parametrize(
    "hardware_id", ("nvidia_a800", "metax_c500", "ascend_910b"))
def test_vector_launch_identity_is_not_inferred_from_an_observed_label(
        hardware_id):
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    shape = {"rows": 1, "hidden": 4096}
    selected = _select_launch_config(
        hardware_id, KernelKind.VECTOR, shape, "bf16", "rmsnorm",
        {"launch_policy": "injected_launch"}, hardware)
    source_kernel = KernelSpec(
        KernelKind.VECTOR, hardware_id, shape, dtype="bf16",
        variant="rmsnorm", launch=selected)
    observed_kernel = KernelSpec(
        KernelKind.VECTOR, hardware_id, shape, dtype="bf16",
        variant="rmsnorm", launch=dataclasses.replace(
            selected, algorithm="observed_vector:rmsnorm"))

    assert source_kernel.launch.algorithm == selected.algorithm
    assert observed_kernel.launch.algorithm == "observed_vector:rmsnorm"
    assert source_kernel.launch != observed_kernel.launch


def test_metax_flash_attention_prediction_is_identical_for_launch_aliases():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    shape = {
        "batch": 1,
        "q_len": 1024,
        "kv_len": 1024,
        "head_dim": 128,
        "num_query_heads": 32,
        "num_kv_heads": 8,
    }
    attributes = {
        "is_decode": False,
        "causal": True,
        "kv_layout": "contiguous",
        "kv_dist": "uniform",
        "q_lens": (1024,),
        "kv_lens": (1024,),
        "fa_ksteps": None,
        "fa_eff_mfrac": 1.0,
        "launch_policy": "injected_launch",
        "split_policy": "serving",
    }
    selected = _select_launch_config(
        hardware.hardware_id, KernelKind.FLASH_ATTN, shape, "bf16",
        "prefill", attributes, hardware)
    prediction_attributes = {
        name: attributes[name]
        for name in (
            "causal", "kv_layout", "kv_dist", "fa_ksteps",
            "fa_eff_mfrac", "q_lens", "kv_lens")
    }
    source_kernel = KernelSpec(
        KernelKind.FLASH_ATTN, hardware.hardware_id, shape, dtype="bf16",
        variant="prefill", launch=selected,
        attributes=prediction_attributes)
    observed_kernel = KernelSpec(
        KernelKind.FLASH_ATTN, hardware.hardware_id, shape, dtype="bf16",
        variant="prefill", launch=dataclasses.replace(
            selected,
            algorithm=(
                "observed_metax_flash_attention_trait_direct:"
                "prefill_q128_full_tile")),
        attributes=prediction_attributes)

    assert source_kernel.launch == observed_kernel.launch
    assert predict(source_kernel, hardware, "midlayer") == predict(
        observed_kernel, hardware, "midlayer")


@pytest.mark.parametrize(
    ("variant", "shape", "launch_attributes"),
    (
        (
            "prefill",
            {
                "batch": 1,
                "q_len": 128,
                "kv_len": 256,
                "head_dim": 64,
                "num_query_heads": 16,
                "num_kv_heads": 16,
            },
            {
                "is_decode": False,
                "causal": True,
                "kv_layout": "contiguous",
                "kv_dist": "uniform",
                "q_lens": (128,),
                "kv_lens": (256,),
                "split_policy": "serving",
            },
        ),
        (
            "prefill",
            {
                "batch": 1,
                "q_len": 128,
                "kv_len": 256,
                "head_dim": 128,
                "num_query_heads": 16,
                "num_kv_heads": 16,
            },
            {
                "is_decode": False,
                "causal": True,
                "kv_layout": "contiguous",
                "kv_dist": "uniform",
                "q_lens": (128,),
                "kv_lens": (256,),
                "split_policy": "serving",
            },
        ),
        (
            "prefill",
            {
                "batch": 1,
                "q_len": 128,
                "kv_len": 256,
                "head_dim": 192,
                "num_query_heads": 16,
                "num_kv_heads": 16,
            },
            {
                "is_decode": False,
                "causal": True,
                "kv_layout": "contiguous",
                "kv_dist": "uniform",
                "q_lens": (128,),
                "kv_lens": (256,),
                "split_policy": "serving",
            },
        ),
        (
            "decode",
            {
                "batch": 1,
                "q_len": 1,
                "kv_len": 256,
                "head_dim": 192,
                "num_query_heads": 16,
                "num_kv_heads": 4,
            },
            {
                "is_decode": True,
                "causal": False,
                "kv_layout": "contiguous",
                "kv_dist": "uniform",
                "q_lens": (1,),
                "kv_lens": (256,),
                "split_policy": "forced",
                "num_kv_splits": 1,
            },
        ),
    ),
)
def test_metax_binary_flash_attention_kernel_spec_direct_smoke(
        variant, shape, launch_attributes):
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    launch_attributes = {
        **launch_attributes,
        "launch_policy": "injected_launch",
    }
    launch = _select_launch_config(
        hardware.hardware_id,
        KernelKind.FLASH_ATTN,
        shape,
        "bf16",
        variant,
        launch_attributes,
        hardware,
    )
    prediction_attributes = {
        name: launch_attributes[name]
        for name in ("causal", "kv_layout", "kv_dist", "q_lens", "kv_lens")
    }
    prediction_attributes.update({"fa_ksteps": None, "fa_eff_mfrac": 1.0})
    kernel = KernelSpec(
        KernelKind.FLASH_ATTN,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=variant,
        launch=launch,
        attributes=prediction_attributes,
    )

    prediction = predict(kernel, hardware, "midlayer")
    assert prediction.t_total_s > 0.0


def test_a800_full_graph_paged_splitkv_reducer_uses_captured_slots():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {
        "batch": 16,
        "q_len": 1,
        "kv_len": 888,
        "head_dim": 128,
        "num_query_heads": 32,
        "num_kv_heads": 8,
    }
    model_input_q_lens = (1,) * 16
    model_input_kv_lens = (888,) * 12 + (1,) * 4
    launch_attributes = {
        "is_decode": True,
        "causal": False,
        "kv_layout": "paged_varlen",
        "kv_dist": "uniform",
        "q_lens": model_input_q_lens,
        "kv_lens": model_input_kv_lens,
        "launch_policy": "analytical_launch",
        "split_policy": "serving",
    }
    main_launch = _select_launch_config(
        hardware.hardware_id, KernelKind.FLASH_ATTN, shape, "bf16",
        "decode", launch_attributes, hardware)
    combine_launch = _select_launch_config(
        hardware.hardware_id, KernelKind.FLASH_ATTN, shape, "bf16",
        "combine", {
            **launch_attributes,
            "num_kv_splits": main_launch.tiling.num_kv_splits,
        }, hardware)
    logical_attributes = {
        "causal": False,
        "kv_layout": "paged_varlen",
        "kv_dist": "uniform",
        "q_lens": (1,) * 12,
        "kv_lens": (888,) * 12,
        "model_input_q_lens": model_input_q_lens,
        "model_input_kv_lens": model_input_kv_lens,
        "graph_runtime_mode": "FULL",
        "fa_ksteps": None,
        "fa_eff_mfrac": 1.0,
    }
    main = KernelSpec(
        KernelKind.FLASH_ATTN, hardware.hardware_id, shape, dtype="bf16",
        variant="decode", launch=main_launch, attributes=logical_attributes)
    combine = KernelSpec(
        KernelKind.FLASH_ATTN, hardware.hardware_id, shape, dtype="bf16",
        variant="combine", launch=combine_launch,
        attributes=logical_attributes)

    main_input, _ = create_predictor(main.kind).resource_input(main, hardware)
    combine_input, _ = create_predictor(
        combine.kind).resource_input(combine, hardware)

    assert main_launch.grid == (1, 3, 128)
    assert combine_launch.grid == (128, 1, 1)
    assert main_input.pipeline_plan is None
    assert combine_input.pipeline_plan is not None


def test_metax_q64_ragged_direct_prefill_uses_source_pipeline():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    shape = {
        "batch": 2,
        "q_len": 80,
        "kv_len": 512,
        "head_dim": 128,
        "num_query_heads": 16,
        "num_kv_heads": 2,
    }
    launch_attributes = {
        "is_decode": False,
        "causal": True,
        "kv_layout": "paged_varlen",
        "kv_dist": "explicit",
        "q_lens": (1, 80),
        "kv_lens": (512, 96),
        "launch_policy": "injected_launch",
        "split_policy": "serving",
        "graph_runtime_mode": "NONE",
    }
    launch = _select_launch_config(
        hardware.hardware_id, KernelKind.FLASH_ATTN, shape, "bf16",
        "prefill", launch_attributes, hardware)
    kernel = KernelSpec(
        KernelKind.FLASH_ATTN, hardware.hardware_id, shape, dtype="bf16",
        variant="prefill", launch=launch,
        attributes={
            "causal": True,
            "kv_layout": "paged_varlen",
            "kv_dist": "explicit",
            "q_lens": (1, 80),
            "kv_lens": (512, 96),
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
        })

    prediction = predict(kernel, hardware, "midlayer")

    assert launch.algorithm == "metax_flash_attention_trait_direct:prefill"
    assert launch.grid == (16, 2, 2)
    assert prediction.t_total_ns > 0
    assert prediction.diagnostics.support_level == "partial"


def test_complete_gemm_spec_predicts_without_observation_data():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"b": 1, "m": 128, "n": 4096, "k": 4096}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.GEMM, shape, "bf16", "gemm",
        {"has_bias": False, "position": "generic",
         "launch_policy": "analytical_launch"}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.GEMM,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="gemm",
        launch=launch,
    )
    assert predict(kernel, hardware, "midlayer").t_total_ns > 0
    assert predict(kernel, hardware, "roofline").t_total_ns > 0


def test_ascend_fused_qkv_rmsnorm_rope_uses_one_streaming_envelope():
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    shape = {
        "seq_len": 2220,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rotary_dim": 128,
    }
    attributes = {"bias": False, "is_neox": True}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
        "qkv_rmsnorm_rope", attributes, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="qkv_rmsnorm_rope",
        launch=launch,
        attributes=attributes,
    )

    resource_input, compulsory_bytes = create_predictor(
        kernel.kind).resource_input(kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")

    input_bytes = 2220 * (32 + 2 * 8) * 128 * 2
    assert resource_input.demand_per_resource["dram"] == (
        input_bytes + 2 * 128 * 2 + 2220 * 8 + 2220 * 128 * 2
        + input_bytes)
    assert resource_input.demand_per_resource["compute_vector"] == (
        6 * 2220 * (32 + 8) * 128
        + 3 * 2220 * (32 + 8) * 128
        + 3 * 2220 * (32 + 8))
    assert compulsory_bytes == resource_input.demand_per_resource["dram"]
    assert resource_input.phase_ledger is None
    assert prediction.t_total_ns > 0.0


def test_c500_gemm_without_complete_pipeline_evidence_is_explicitly_partial():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    shape = {"b": 1, "m": 128, "n": 6144, "k": 4096}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.GEMM, shape, "bf16", "gemm",
        {"has_bias": False, "position": "generic",
         "launch_policy": "injected_launch"}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.GEMM,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="gemm",
        launch=launch,
    )

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")

    assert resource_input.pipeline_plan is None
    assert resource_input.support_level == "partial"
    assert "gemm_pipeline_overlap_evidence_unavailable" not in (
        resource_input.limitations)
    assert resource_input.dependency_paths
    assert prediction.diagnostics.pipeline is None


def test_c500_lm_head_gemv_without_pipeline_evidence_is_aggregate_only():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    shape = {"b": 1, "m": 1, "n": 128256, "k": 4096}
    attributes = {
        "has_bias": False,
        "position": "lm_head",
        "launch_policy": "injected_launch",
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
    }
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.GEMM, shape, "bf16", "gemv",
        attributes, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.GEMM,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=launch.execution_variant,
        launch=launch,
    )

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert launch.tiling is None
    assert resource_input.pipeline_plan is None
    assert resource_input.support_level == "aggregate_only"
    assert (
        "gemm_pipeline_overlap_evidence_unavailable"
        in resource_input.limitations)


def test_a800_analytical_gemv_uses_steady_dependency_path():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"b": 1, "m": 1, "n": 4096, "k": 4096}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.GEMM, shape, "bf16", "gemv",
        {"has_bias": False, "position": "generic",
         "launch_policy": "analytical_launch"}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.GEMM,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=launch.execution_variant,
        launch=launch,
    )

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert launch.tiling.pipeline_depth is not None
    assert resource_input.pipeline_plan is None
    path, = resource_input.dependency_paths
    assert path.name == "gpu_gemv_load_reduce_store"
    assert path.concurrent_service_groups == ((
        "memory:global_read", "compute:compute_fma"),)
    assert resource_input.support_level == "partial"
    assert resource_input.eta == 1.0
    assert resource_input.wave_efficiency < 1.0


def test_a800_injected_depth_two_gemm_uses_direct_k_tile_pipeline():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"b": 1, "m": 16, "n": 2560, "k": 2048}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.GEMM, shape, "bf16", "gemm",
        {
            "has_bias": False,
            "position": "fused_moe",
            "tensor_parallel_size": 2,
            "expert_parallel_size": 2,
        },
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.GEMM,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=launch.execution_variant,
        launch=launch,
    )

    prediction = predict(kernel, hardware, "midlayer")

    assert launch.tiling is not None
    assert launch.tiling.pipeline_depth == 2
    assert prediction.t_total_s > 0.0
    assert prediction.diagnostics.pipeline is not None


_VECTOR_FORWARD_CASES = (
    (KernelKind.VECTOR, "rmsnorm", {"rows": 1, "hidden": 2048}, {}),
    (KernelKind.VECTOR, "add_rmsnorm", {
        "rows": 1024, "hidden": 16384}, {}),
    (KernelKind.VECTOR, "rotary", {
        "seq_len": 1024, "num_query_heads": 128,
        "num_kv_heads": 8, "head_dim": 128,
    }, {}),
    (KernelKind.VECTOR, "activation", {
        "tok": 1024, "intermediate": 53248}, {}),
    (KernelKind.VECTOR, "elementwise", {"element_count": 16777216}, {
        "reads": 2, "writes": 1, "operation": "add",
    }),
    (KernelKind.KV_CACHE, "paged_kv_cache", {
        "seq_len": 1024, "num_kv_heads": 8, "head_dim": 128,
    }, PAGED_KV_CACHE_ATTRIBUTES),
)

_GPU_VECTOR_LIMITATIONS = {
    "configs/hardware/nvidia_a800.yaml": (
        "vector_cta_to_sm_residency_assignment_unavailable",
        "vector_load_compute_store_buffer_depth_unavailable",
        "l1_and_local_queue_evidence_unavailable",
        "noncoalesced_transaction_geometry_unavailable",
        "workload_producer_independence_unavailable",
        "hardware_source_ceiling_evidence_unavailable",
        "request_frontend_capacity_evidence_unavailable",
        "dram_direction_switch_schedule_unavailable",
        "dram_turnaround_unmodeled",
    ),
    "configs/hardware/metax_c500.yaml": (
        "vector_cta_to_sm_residency_assignment_unavailable",
        "vector_load_compute_store_buffer_depth_unavailable",
        "local_queue_evidence_unavailable",
        "noncoalesced_transaction_geometry_unavailable",
        "workload_producer_independence_unavailable",
        "hardware_source_ceiling_evidence_unavailable",
        "request_frontend_capacity_evidence_unavailable",
        "dram_direction_switch_schedule_unavailable",
        "dram_turnaround_unmodeled",
    ),
}


@pytest.mark.parametrize(
    "hardware_path",
    [
        "configs/hardware/nvidia_a800.yaml",
        "configs/hardware/metax_c500.yaml",
    ],
)
@pytest.mark.parametrize(
    "kind,variant,shape,attributes", _VECTOR_FORWARD_CASES)
def test_gpu_vector_forward_uses_analytical_work_without_duration_calibration(
        hardware_path, kind, variant, shape, attributes):
    hardware = HWSpec(hardware_path)
    launch = select_launch_config(
        hardware.hardware_id, kind, shape, "bf16", variant,
        attributes, hardware=hardware)
    kernel = KernelSpec(
        kind, hardware.hardware_id, shape, dtype="bf16",
        variant=launch.execution_variant, launch=launch,
        attributes=attributes)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")

    has_source_pipeline = (
        (hardware.hardware_id == "metax_c500"
         and variant in (
             "rmsnorm", "add_rmsnorm", "activation", "rotary"))
        or SOURCE_BOUND_GPU_NORM_CTA_PIPELINE_PARAMETER
        in launch.parameters)
    if has_source_pipeline:
        assert resource_input.pipeline_plan is not None
    else:
        assert resource_input.pipeline_plan is None
    assert resource_input.support_level == "aggregate_only"
    expected_limitations = _GPU_VECTOR_LIMITATIONS[hardware_path]
    contract = WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR
    generic_elementwise_add_mul = (
        hardware.hardware_id == "nvidia_a800"
        and kind is KernelKind.VECTOR
        and variant == "elementwise"
        and attributes["operation"] in ("add", "mul"))
    if (not generic_elementwise_add_mul
            and hardware.has_task_duration_floor_contract(contract)
            and not hardware.task_duration_floor_contract_supports_launch(
                    contract, launch.grid_size, launch.block_size,
                    launch.grid, launch.block, hardware.nominal_sm_freq_hz)):
        expected_limitations += (
            "memory_dependency_task_floor_outside_support",)
    if "warp_instructions_lower_bound" in launch.parameters:
        expected_limitations += (
            "warp_instruction_demand_is_static_lower_bound",)
    if SOURCE_BOUND_WARP_SLOT2_INSTRUCTIONS_PARAMETER in launch.parameters:
        expected_limitations += (
            "warp_slot2_instruction_demand_is_static_lower_bound",)
    if hardware.hardware_id == "nvidia_a800" and variant == "activation":
        expected_limitations += (
            "silu_scalar_dependency_launch_outside_support",)
    if (hardware.hardware_id == "nvidia_a800"
            and variant == "rmsnorm"
            and "source_bound_block_reduction" not in launch.parameters):
        expected_limitations += (
            "norm_reduction_dependency_service_unavailable",)
    block256_source_rounds = (
        hardware.add_rmsnorm_block256_source_round_pipeline)
    if (hardware.hardware_id == "nvidia_a800"
            and variant == "add_rmsnorm"
            and block256_source_rounds is not None
            and launch.block_size == block256_source_rounds["threads_per_cta"]
            and shape["rows"] > block256_source_rounds["maximum_grid_blocks"]):
        expected_limitations += (
            "a800_block256_add_rms_queued_ctas_use_static_source_pipeline",
            "a800_block256_add_rms_queued_cta_typed_scalar_service_unavailable",
        )
    if (SOURCE_PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_PARAMETER
            in launch.parameters
            and not hardware.runtime_task_class_floor_supports(
                PYTORCH_BF16_BINARY_TENSOR_ITERATOR_TASK_CLASS,
                shape["element_count"])):
        expected_limitations += (
            "runtime_task_class_floor_outside_support",)
    assert resource_input.limitations == expected_limitations
    assert resource_input.demand_per_resource["l2_cache"] > 0.0
    assert resource_input.demand_per_resource["dram"] > 0.0
    assert {path.access_kind for path in resource_input.memory_paths} == {
        "read", "write"}
    if has_source_pipeline:
        assert prediction.diagnostics.pipeline is not None
    else:
        assert prediction.diagnostics.pipeline is None
    assert prediction.diagnostics.memory
    assert resource_input.include_dispatch_overhead
    has_runtime_task_floor = (
        resource_input.runtime_task_class is not None
        and resource_input.runtime_task_elements is not None
        and hardware.runtime_task_class_floor_supports(
            resource_input.runtime_task_class,
            resource_input.runtime_task_elements))
    # A card that publishes a measured CTA dispatch cost pays it on every
    # launch whose grid is wide enough for the measurement to be non-zero,
    # since a CTA cannot begin before it is dispatched.
    dispatch_ns = hardware.cta_dispatch_cost_ns(resource_input.grid_size)
    if (has_source_pipeline
            or (hardware.hardware_id == "metax_c500"
                and has_runtime_task_floor)
            or dispatch_ns > 0.0):
        assert prediction.t_launch_s > 0.0
    else:
        assert prediction.t_launch_s == 0.0
    empty_floor = hardware.empty_kernel_duration_for_launch(
        resource_input.grid_size, resource_input.block_size,
        resource_input.launch_grid, resource_input.launch_block,
    )
    assert prediction.t_total_ns >= empty_floor
    if hardware.hardware_id == "nvidia_a800":
        assert empty_floor > 0.0
        assert resource_input.demand_per_resource["l1_cache"] > 0.0
    elif hardware.hardware_id == "metax_c500":
        # MACA 3.7.2 uses the independent register-only non-empty task
        # scheduler mechanism in the mcTracer GPU-event domain required of
        # the version-matched production candidate.
        assert empty_floor > 0.0
        assert resource_input.demand_per_resource["l1_cache"] == 0.0
    else:
        raise AssertionError(
            f"unexpected GPU hardware: {hardware.hardware_id}")


def test_a800_small_grid_rms_reduction_uses_only_active_sms():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"rows": 64, "hidden": 4096}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16", "rmsnorm",
        {}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant="rmsnorm", launch=launch)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")
    reduction_demand = next(
        demand for demand in resource_input.compute_demands
        if demand.resource_role == "compute_reduction")
    service_demand, = reduction_demand.service_demands
    reduction = hardware.resource_by_role("compute_reduction")
    assert reduction.service_rates_per_unit_s is not None
    expected_time_s = (
        max(service_demand.operations_by_worker)
        / reduction.service_rates_per_unit_s[service_demand.service]
    )

    assert resource_input.active_units == 64
    assert service_demand.service == "block_reduce_first_1024"
    assert service_demand.operations == 64.0
    assert all(
        operations == 1.0
        for operations in service_demand.operations_by_worker)
    assert all(
        demand.resource_role != "compute_sfu"
        for demand in resource_input.compute_demands)
    assert prediction.t_per_resource[reduction.name] == pytest.approx(
        expected_time_s)
    assert prediction.t_total_s >= prediction.t_per_resource[reduction.name]


def test_c500_norm_publishes_versioned_non_duration_vector_cycle_floor():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    shape = {"rows": 64, "hidden": 4096}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16", "rmsnorm",
        {}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant="rmsnorm", launch=launch)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")
    issue = hardware.resource_by_role("warp_issue")

    assert resource_input.active_units == 64
    assert "warp_instructions" not in launch.parameters
    issue_demand, = tuple(
        demand for demand in resource_input.compute_demands
        if demand.resource_role == "warp_issue")
    expected_cycles_per_cta = 1317.0 + 624.0 * (shape["hidden"] // 1024)
    assert issue_demand.operations == (
        5.0 * shape["rows"] * expected_cycles_per_cta)
    assert prediction.t_per_resource[issue.name] > 0.0


def test_c500_rope_shared_divmod_publishes_one_integer_service():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    shape = {
        "seq_len": 4,
        "num_query_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "rotary_dim": 128,
    }
    attributes = {"is_neox": True, "key_present": True}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16", "rope",
        attributes, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant="rope", launch=launch, attributes=attributes)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    vector_demand = next(
        demand for demand in resource_input.compute_demands
        if demand.resource_role == "compute_vector")
    services = {demand.service for demand in vector_demand.service_demands}

    assert services == {
        "integer_rem_u32_dependency",
        "address_multiply_add_u64_dependency",
        "bf16_roundtrip_pair_dependency",
    }
    assert "integer_div_u32_dependency" not in services


def test_a800_copy_uses_generic_memory_dependency_task_without_pipeline():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"element_count": 8192}
    attributes = {"reads": 1, "writes": 1, "operation": "copy"}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
        "elementwise", attributes, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant=launch.execution_variant, launch=launch,
        attributes=attributes)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")

    assert resource_input.pipeline_plan is None
    assert resource_input.support_level == "aggregate_only"
    assert resource_input.limitations == _GPU_VECTOR_LIMITATIONS[
        "configs/hardware/nvidia_a800.yaml"]
    # The 8192-element copy's complete operand and result footprint is
    # L2-resident, so neither its loads nor its stores reach DRAM.
    assert resource_input.demand_per_resource["dram"] == 0.0
    assert resource_input.demand_per_resource["l2_cache"] > 0.0
    assert resource_input.demand_per_resource["compute_fma"] == 0.0
    assert launch.parameters == {}
    # The launch pays the generic dispatch overhead plus the card's measured
    # CTA dispatch cost for its grid.
    assert prediction.t_launch_s == pytest.approx(
        hardware.dispatch_overhead_ns * 1e-9
        + hardware.cta_dispatch_cost_ns(launch.grid_size) * 1e-9)
    assert resource_input.task_duration_floor_contract == (
        WARM_DEFAULT_CACHE_MEMORY_DEPENDENCY_TASK_FLOOR)
    assert prediction.t_total_ns > hardware.empty_kernel_duration_for_launch(
        resource_input.grid_size, resource_input.block_size)
    assert prediction.diagnostics.pipeline is None


@pytest.mark.parametrize(
    ("variant", "shape", "attributes"),
    (
        ("rmsnorm", {"rows": 2, "hidden": 4096}, {}),
        ("rope", {
            "seq_len": 2, "num_query_heads": 32, "num_kv_heads": 4,
            "head_dim": 128, "rotary_dim": 128,
        }, {"is_neox": True, "key_present": True}),
        ("silu", {"tok": 2, "intermediate": 6144}, {}),
        ("elementwise_pytorch_vec4", {"element_count": 2048}, {
            "reads": 2, "writes": 1, "operation": "add",
        }),
    ),
)
def test_a800_source_bound_vector_paths_consume_memory_dependency_task_floor(
        variant, shape, attributes):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16", variant,
        attributes, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant=variant, launch=launch, attributes=attributes)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")
    contract = "warm_default_cache_memory_dependency_task_v1"
    expected_floor_ns = hardware.memory_dependency_task_duration_ns(
        contract,
        launch.grid_size,
        launch.block_size,
        launch.grid,
        launch.block,
        resource_input.sm_freq_hz,
    )

    assert resource_input.task_duration_floor_contract == contract
    assert resource_input.include_dispatch_overhead
    if resource_input.pipeline_plan is None:
        assert prediction.t_launch_s == 0.0
    else:
        assert prediction.t_launch_s > 0.0
    assert prediction.t_total_ns >= expected_floor_ns


def test_a800_grid256_silu_uses_compact_segmented_cta_pipeline():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {
        "tok": 256,
        "intermediate": 53248,
        "element_count": 256 * 53248,
    }
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16", "silu", {},
        hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant="silu", launch=launch)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")

    assert launch.parameters[
        SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER] == 1.0
    assert resource_input.resident_cta_cohort_refills == 2
    plan = resource_input.pipeline_plan
    assert plan is not None
    assert plan.worker_count == 108
    assert len(plan.tasks) == 2
    # Each worker retires one source-loop round at a time: 52 rounds per CTA
    # for three or two CTAs.  The source has no multi-round prefetch, so each
    # round is its own task carrying its own exposed operand access.
    assert tuple(
        (program.worker_multiplicity, len(program.order))
        for program in plan.tasks) == ((40, 3 * 52), (68, 2 * 52))
    assert tuple(
        worker
        for program in plan.tasks
        for worker in program.physical_workers) == tuple(range(108))
    assert all(
        tuple(stage.name for stage in pattern.tasks[0].stages) == (
            "load", "source_silu", "source_silu_issue",
            "gate_multiply", "store",
        )
        for program in plan.tasks
        for pattern in program.patterns
    )
    assert all(
        pattern.tasks[0].stages[0].include_first_request_latency
        for program in plan.tasks
        for pattern in program.patterns
    )
    # The round store drains asynchronously and does not gate the round.
    assert all(
        pattern.tasks[0].stages[-1].predecessors == ()
        for program in plan.tasks
        for pattern in program.patterns
    )
    vector_demand, = (
        demand for demand in resource_input.compute_demands
        if demand.resource_role == "compute_vector")
    assert {demand.service for demand in vector_demand.service_demands} == {
        f"silu_nonrecurrent_{regime}_{component}_block1024"
        for regime in ("first", "resident", "queued")
        for component in ("entry", "round")
    }
    assert vector_demand.operations == 256 * (1 + 52)
    assert prediction.t_total_ns > 0


def test_a800_full_sm_multiwave_silu_uses_source_program():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {
        "tok": 3301,
        "intermediate": 14336,
        "element_count": 3301 * 14336,
    }
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16", "silu", {},
        hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant="silu", launch=launch)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")

    assert launch.parameters[
        SOURCE_BOUND_SILU_NONRECURRENT_CTA_PIPELINE_PARAMETER] == 1.0
    assert resource_input.pipeline_plan is not None
    assert any(
        demand.resource_role == "compute_vector"
        for demand in resource_input.compute_demands)
    assert prediction.t_total_ns > 0


def test_a800_single_cta_rmsnorm_does_not_repeat_complete_task_body():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")

    for rows in (1, 2):
        shape = {"rows": rows, "hidden": 2048}
        launch = select_launch_config(
            hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
            "rmsnorm", {}, hardware=hardware)
        kernel = KernelSpec(
            KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
            variant="rmsnorm", launch=launch)
        resource_input, _ = create_predictor(kernel.kind).resource_input(
            kernel, hardware)
        plan = resource_input.pipeline_plan
        assert plan is not None
        assert plan.worker_count == rows
        assert len(plan.tasks) == 1
        assert resource_input.dependency_paths == ()
        program, = plan.tasks
        assert program.worker == 0
        assert program.worker_multiplicity == rows
        assert program.physical_workers == tuple(range(rows))
        assert program.repeat_count == 1
        assert len(program.patterns) == 1
        sequence = program.patterns[0]
        assert sequence.repeat_count == 1
        assert len(sequence.tasks) == 1
        assert sequence.tasks[0].repeat_count == 1



def _memory_touching_span_floor_ns(hardware, grid_size):
    """The published floor for a launch that performs any global access."""

    profile = hardware.small_kernel_span_floor
    if profile is None:
        return 0.0
    resolved = profile["floor_ns_by_grid_ctas"][0][1]
    for grid, floor_ns in profile["floor_ns_by_grid_ctas"]:
        if grid <= grid_size:
            resolved = floor_ns
    return resolved

def test_a800_moe_act_and_mul_does_not_spoof_memory_dependency_task_floor():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {
        "tokens": 8,
        "experts": 128,
        "topk": 8,
        "hidden": 2048,
        "intermediate": 768,
    }
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
        "moe_act_and_mul", {}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant=launch.execution_variant, launch=launch, attributes={})

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")

    assert launch.grid == (shape["tokens"] * shape["topk"], 1, 1)
    assert resource_input.task_duration_floor_contract is None
    # Still the launch floor and nothing spoofed on top of it -- but a launch
    # that performs a global access does not reach the empty-task duration, so
    # the measured memory-touching floor is the larger of the two.
    assert prediction.t_total_ns == pytest.approx(max(
        hardware.empty_kernel_duration_for_launch(
            launch.grid_size, launch.block_size, launch.grid, launch.block),
        _memory_touching_span_floor_ns(hardware, launch.grid_size)))


def test_a800_generic_fp16_elementwise_does_not_claim_complete_task_floor():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"element_count": 56 * 1024}
    attributes = {"reads": 2, "writes": 1, "operation": "add"}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "fp16",
        "elementwise", attributes, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="fp16",
        variant=launch.execution_variant, launch=launch, attributes=attributes)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert launch.parameters == {}
    assert resource_input.task_duration_floor_contract is None


def test_a800_fp16_rmsnorm_preserves_single_cta_completion_across_rows():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    predictions = []
    floors = []
    for rows in (1, 2):
        shape = {"rows": rows, "hidden": 5120}
        launch = select_launch_config(
            hardware.hardware_id, KernelKind.VECTOR, shape, "fp16",
            "rmsnorm", {}, hardware=hardware)
        kernel = KernelSpec(
            KernelKind.VECTOR, hardware.hardware_id, shape, dtype="fp16",
            variant=launch.execution_variant, launch=launch, attributes={})
        resource_input, _ = create_predictor(kernel.kind).resource_input(
            kernel, hardware)
        predictions.append(predict(kernel, hardware, "midlayer").t_total_ns)
        floors.append(resource_input.execution_time_floor_ns)

    assert floors[0] is not None
    assert floors[1] == pytest.approx(floors[0])
    assert predictions[1] >= predictions[0]


def test_a800_kv_cache_semantics_consume_memory_dependency_task_floor():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"seq_len": 2, "num_kv_heads": 4, "head_dim": 128}
    attributes = {
        "cache_layout": "paged",
        "cache_head_layout": "nhd",
        "kv_cache_dtype": "auto",
        "cache_storage_matches_input": True,
        "slot_mapping_all_valid": True,
        "vector_copy_base_alignment_bytes": 16,
    }
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.KV_CACHE, shape, "bf16",
        "paged_kv_cache",
        attributes, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.KV_CACHE, hardware.hardware_id, shape, dtype="bf16",
        variant="paged_kv_cache", launch=launch, attributes=attributes)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert resource_input.task_duration_floor_contract == (
        "warm_default_cache_memory_dependency_task_v1")


def test_a800_h2_kv_cache_requires_explicit_resident_cta_binding():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    attributes = {
        "cache_layout": "paged",
        "cache_head_layout": "nhd",
        "kv_cache_dtype": "auto",
        "cache_storage_matches_input": True,
        "slot_mapping_all_valid": True,
        "vector_copy_base_alignment_bytes": 16,
    }
    predictions = []
    for seq_len in (320, 328):
        shape = {
            "seq_len": seq_len,
            "num_kv_heads": 2,
            "head_dim": 128,
        }
        launch = select_launch_config(
            hardware.hardware_id, KernelKind.KV_CACHE, shape, "bf16",
            "paged_kv_cache", attributes, hardware=hardware)
        kernel = KernelSpec(
            KernelKind.KV_CACHE, hardware.hardware_id, shape, dtype="bf16",
            variant=launch.execution_variant, launch=launch,
            attributes=attributes)
        resource_input, _ = create_predictor(kernel.kind).resource_input(
            kernel, hardware)
        read_path = next(
            path for path in resource_input.memory_paths
            if path.name == "global_read")

        assert "source_bound_resident_ctas_per_unit" not in launch.parameters
        assert read_path.source.producers_per_unit == 1.0
        predictions.append(predict(kernel, hardware, "midlayer").t_total_ns)

    assert predictions[1] >= predictions[0]


def test_c500_vectorized_kv_source_rounds_extend_only_h8_read_depth():
    """The static scalar loop adds a read chain per 512-element source round."""

    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    serial_counts = {}
    for num_kv_heads in (4, 8):
        shape = {
            "seq_len": 1024,
            "num_kv_heads": num_kv_heads,
            "head_dim": 128,
        }
        launch = select_launch_config(
            hardware.hardware_id, KernelKind.KV_CACHE, shape, "bf16",
            "paged_kv_cache", PAGED_KV_CACHE_ATTRIBUTES, hardware=hardware)
        kernel = KernelSpec(
            KernelKind.KV_CACHE, hardware.hardware_id, shape, dtype="bf16",
            variant=launch.execution_variant, launch=launch,
            attributes=PAGED_KV_CACHE_ATTRIBUTES)
        resource_input, _ = create_predictor(kernel.kind).resource_input(
            kernel, hardware)
        dependency_path, = resource_input.dependency_paths
        serial_counts[num_kv_heads] = dict(
            dependency_path.serial_service_counts)[
                "memory_latency:global_read"]

    # 1024 CTAs form three resident-CTA batches (104 active APs x 4
    # architecturally resident CTAs per AP = 416 concurrent slots; ceil(1024
    # / 416) = 3), the same _source_bound_cta_batches every other
    # source-bound copy path on this hardware uses.  H4D128 has one
    # 512-element scalar-loop round and H8D128 has two; aggregate traffic is
    # unchanged.
    assert serial_counts == {4: 3, 8: 6}


@pytest.mark.parametrize(
    ("variant", "shape", "attributes", "expected_transactions"),
    (
        ("rmsnorm", {"rows": 2, "hidden": 4096}, {}, 2.0),
        ("silu", {"tok": 2, "intermediate": 6144}, {}, 4.0),
        ("rope", {
            "seq_len": 2, "num_query_heads": 32,
            "num_kv_heads": 8, "head_dim": 128,
            "rotary_dim": 128,
        }, {"is_neox": True, "key_present": True}, 8.0),
        ("elementwise_pytorch_vec4", {"element_count": 2048}, {
            "reads": 2, "writes": 1, "operation": "add",
        }, 32.0),
    ),
)
def test_a800_typed_vector_publishes_only_static_l2_read_sources(
        variant, shape, attributes, expected_transactions):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16", variant,
        attributes, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant=variant, launch=launch, attributes=attributes)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    paths = {path.name: path for path in resource_input.memory_paths}

    assert launch.parameters[
        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
    ] == expected_transactions
    assert paths["global_read"].source.independent_requests_per_producer == (
        expected_transactions)
    assert (
        paths["global_write"].source.independent_requests_per_producer
        is None)


def test_a800_vec4_half_tail_keeps_read_source_independence_unknown():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"element_count": 2560}
    attributes = {"reads": 2, "writes": 1, "operation": "add"}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
        "elementwise_pytorch_vec4", attributes, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant="elementwise_pytorch_vec4", launch=launch,
        attributes=attributes)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert (
        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
        not in launch.parameters)
    assert all(
        path.source.independent_requests_per_producer is None
        for path in resource_input.memory_paths)


def test_a800_rmsnorm_producer_tail_keeps_source_independence_unknown():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR,
        {"rows": 1, "hidden": 2080}, "bf16", "rmsnorm", {},
        hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id,
        {"rows": 1, "hidden": 2080}, dtype="bf16",
        variant="rmsnorm", launch=launch)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert (
        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER
        not in launch.parameters)
    assert all(
        path.source.independent_requests_per_producer is None
        for path in resource_input.memory_paths)


def test_a800_silu_tail_uses_full_warp_read_source():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"tok": 1, "intermediate": 2080}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16", "silu", {},
        hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant="silu", launch=launch)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    read_path = next(
        path for path in resource_input.memory_paths
        if path.name == "global_read")

    assert launch.parameters[
        SOURCE_BOUND_L2_READ_TRANSACTIONS_PER_WARP_PARAMETER] == 4.0
    assert read_path.source.producers_per_unit == 32.0
    assert read_path.source.independent_requests_per_producer == 4.0


def test_a800_exact_add_rmsnorm_consumes_memory_dependency_task_floor():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"rows": 2, "hidden": 4096}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
        "add_rmsnorm", {}, hardware=hardware)
    parameters = dict(launch.parameters)
    parameters["source_bound_memory_dependency_task"] = 1.0
    launch = dataclasses.replace(
        launch,
        parameters=parameters,
    )
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant="add_rmsnorm", launch=launch)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert resource_input.task_duration_floor_contract == (
        "warm_default_cache_memory_dependency_task_v1")


def test_a800_tensor_iterator_copy_floor_precedes_semantic_memory_floor():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    kernel = KernelSpec(
        KernelKind.VECTOR,
        hardware.hardware_id,
        {"element_count": 1024},
        dtype="bf16",
        variant="elementwise",
        launch=LaunchConfig(
            "observed_flash_attention_copy:elementwise",
            (2, 1, 1),
            (128, 1, 1),
            parameters={
                SOURCE_BOUND_TENSOR_ITERATOR_COPY_TASK_PARAMETER: 4.0,
            },
            execution_variant="elementwise",
        ),
        attributes={"reads": 1, "writes": 1, "operation": "copy"},
    )

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert resource_input.task_duration_floor_contract == (
        TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR)


def test_a800_norm_input_materialization_copy_binds_tensor_iterator_floor():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    elements = 16 * 32 * 128
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, {"element_count": elements},
        "bf16", "elementwise",
        {
            "operation": "copy",
            "reads": 1,
            "writes": 1,
            "launch_policy": "injected_launch",
            "input_materialization": "grouped_strided_to_contiguous",
            "input_outer_groups": 16,
            "input_rows_per_group": 32,
            "input_group_stride_elements": 6144,
            "input_row_stride_elements": 128,
        },
        hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, {"element_count": elements},
        dtype="bf16", variant=launch.execution_variant, launch=launch,
        attributes={
            "reads": 1,
            "writes": 1,
            "operation": "copy",
            "input_materialization": "grouped_strided_to_contiguous",
            "input_outer_groups": 16,
            "input_rows_per_group": 32,
            "input_group_stride_elements": 6144,
            "input_row_stride_elements": 128,
        })

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert resource_input.task_duration_floor_contract == (
        TENSOR_ITERATOR_BF16_RANK4_COPY_TASK_FLOOR)



def test_a800_memory_dependency_task_floor_does_not_extrapolate_or_leak():
    a800 = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"rows": 109, "hidden": 4096}
    launch = select_launch_config(
        a800.hardware_id, KernelKind.VECTOR, shape, "bf16", "rmsnorm", {},
        hardware=a800)
    kernel = KernelSpec(
        KernelKind.VECTOR, a800.hardware_id, shape, dtype="bf16",
        variant="rmsnorm", launch=launch)
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, a800)

    assert resource_input.task_duration_floor_contract is None
    assert (
        "memory_dependency_task_floor_outside_support"
        in resource_input.limitations)

    kv_shape = {"seq_len": 2, "num_kv_heads": 4, "head_dim": 128}
    kv_launch = select_launch_config(
        a800.hardware_id, KernelKind.KV_CACHE, kv_shape, "bf16",
        "paged_kv_cache",
        PAGED_KV_CACHE_ATTRIBUTES, hardware=a800)
    kv_kernel = KernelSpec(
        KernelKind.KV_CACHE, a800.hardware_id, kv_shape, dtype="bf16",
        variant="paged_kv_cache", launch=kv_launch,
        attributes=PAGED_KV_CACHE_ATTRIBUTES)
    kv_input, _ = create_predictor(kv_kernel.kind).resource_input(
        kv_kernel, a800)

    assert kv_input.task_duration_floor_contract == (
        "warm_default_cache_memory_dependency_task_v1")

    c500 = HWSpec("configs/hardware/metax_c500.yaml")
    c500_shape = {"rows": 2, "hidden": 4096}
    c500_launch = select_launch_config(
        c500.hardware_id, KernelKind.VECTOR, c500_shape, "bf16", "rmsnorm",
        {}, hardware=c500)
    c500_kernel = KernelSpec(
        KernelKind.VECTOR, c500.hardware_id, c500_shape, dtype="bf16",
        variant="rmsnorm", launch=c500_launch)
    c500_input, _ = create_predictor(c500_kernel.kind).resource_input(
        c500_kernel, c500)

    assert c500_input.task_duration_floor_contract is None
    assert (
        "memory_dependency_task_floor_outside_support"
        not in c500_input.limitations)


@pytest.mark.parametrize(
    ("operation", "reads", "writes", "expected_fma"),
    (
        ("add", 2, 1, 8192),
        ("mul", 2, 1, 8192),
        ("copy", 1, 1, 0),
        ("fill", 0, 1, 0),
        ("read", 1, 0, 0),
    ),
)
def test_elementwise_operation_identity_controls_arithmetic(
        operation, reads, writes, expected_fma):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"element_count": 8192}
    attributes = {
        "reads": reads, "writes": writes, "operation": operation}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
        "elementwise", attributes, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant="elementwise", launch=launch, attributes=attributes)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert resource_input.demand_per_resource["compute_fma"] == expected_fma




@pytest.mark.parametrize(
    "kind,variant,shape",
    [
        (KernelKind.VECTOR, "rmsnorm", {
            "rows": 1024, "hidden": 16384}),
        (KernelKind.VECTOR, "rotary", {
            "seq_len": 1024, "num_query_heads": 128,
            "num_kv_heads": 8, "head_dim": 128,
        }),
        (KernelKind.KV_CACHE, "paged_kv_cache", {
            "seq_len": 1024, "num_kv_heads": 8, "head_dim": 128,
            "element_count": 1024 * 8 * 128,
        }),
    ],
)
def test_ascend_vector_forward_is_byte_only_with_retained_launch_floor(
        kind, variant, shape):
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    if variant == "paged_kv_cache":
        attributes = PAGED_KV_CACHE_ATTRIBUTES
    elif variant in ("rmsnorm", "rotary"):
        attributes = {}
    else:
        raise AssertionError(f"unhandled test variant: {variant!r}")
    launch = select_launch_config(
        hardware.hardware_id, kind, shape, "bf16", variant,
        attributes, hardware=hardware)
    kernel = KernelSpec(
        kind, hardware.hardware_id, shape, dtype="bf16",
        variant=launch.execution_variant, launch=launch,
        attributes=attributes)

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")

    assert resource_input.launch_task_kind == "aiv"
    assert resource_input.include_dispatch_overhead
    if resource_input.task_duration_floor_contract == (
            AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM):
        expected_launch_ns = hardware.empty_kernel_duration_for_task(
            "aiv", launch.grid_size, 1, launch.grid, launch.block)
    else:
        expected_launch_ns = (
            hardware.aiv_nonempty_task_latency_model["submit_ns"])
    if variant in ("rmsnorm", "rotary", "paged_kv_cache"):
        assert resource_input.task_prologue_contract is not None
        expected_launch_ns += (
            resource_input.task_prologue_repetitions
            * hardware.task_prologue_duration_ns(
                resource_input.task_prologue_contract,
                resource_input.active_units,
            ))
    if "source_cann_rope_index_select_count" in launch.parameters:
        assert prediction.t_launch_s * 1e9 > expected_launch_ns
        assert "public_op_index_select_sequence" in prediction.t_per_group
    else:
        assert prediction.t_launch_s * 1e9 == pytest.approx(expected_launch_ns)
    assert prediction.t_total_ns >= hardware.task_duration_floor_for_task(
        "aiv", launch.grid_size, 1,
        task_duration_floor_contract=(
            resource_input.task_duration_floor_contract))
    assert (
        "launch_task_duration_evidence_unavailable"
        not in resource_input.limitations)
    assert resource_input.demand_per_resource["l2_cache"] > 0.0
    assert resource_input.demand_per_resource["dram"] > 0.0
    assert resource_input.active_units == min(launch.grid_size, 48)
    assert resource_input.demand_per_resource["compute_vector"] > 0.0
    assert resource_input.demand_per_resource["mte3"] > 0.0
    read_path = next(
        path for path in resource_input.memory_paths
        if path.name == "mte2_read")
    assert read_path.rate_unit == "bytes"
    assert read_path.initial_transactions is None
    assert read_path.nodes[0].transactions is None
    assert read_path.nodes[0].transaction_bytes is None
    assert {path.name for path in resource_input.memory_paths} == {
        "mte2_read", "mte3_write"}
    has_source_pipeline = variant in (
        "rmsnorm", "rotary", "paged_kv_cache")
    expected_support = "partial" if has_source_pipeline else "aggregate_only"
    assert (resource_input.pipeline_plan is not None) is has_source_pipeline
    assert resource_input.support_level == expected_support
    if variant in ("rmsnorm", "rotary", "paged_kv_cache"):
        expected_prediction_support = expected_support
        expect_prediction_pipeline = has_source_pipeline
    else:
        raise AssertionError(f"unhandled test variant: {variant!r}")
    assert prediction.diagnostics.support_level == expected_prediction_support
    assert (
        prediction.diagnostics.pipeline is not None
    ) is expect_prediction_pipeline
    assert all(
        path.source.independent_requests_per_producer is None
        for path in resource_input.memory_paths)
    for role in ("compute_fma", "compute_sfu", "compute_alu"):
        assert resource_input.demand_per_resource[role] == 0.0


@pytest.mark.parametrize(
    ("variant", "block_dim", "expected_ns"),
    [
        ("rmsnorm", 8, (50.0 + 0.630) * 1000.0),
        ("add_rmsnorm", 8, (50.0 + 0.630) * 1000.0),
        ("rotary", 8, 50_000.0),
        ("paged_kv_cache", 8, 50_000.0),
    ],
)
def test_ascend_vector_oracle_replaces_only_observed_aiv_component(
        variant, block_dim, expected_ns):
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")

    observed_floor_ns = cann_vector_observed_aiv_floor_ns(
        variant, 50_000.0, block_dim, hardware)

    assert observed_floor_ns == pytest.approx(expected_ns)


@pytest.mark.parametrize("variant", ["silu", "activation", "elementwise"])
def test_ascend_vector_oracle_rejects_aiv_for_non_aiv_floor_variants(variant):
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")

    assert cann_vector_observed_aiv_floor_ns(
        variant, 50_000.0, 8, hardware) is None


def test_vector_forward_has_no_duration_derived_calibration_hooks():
    source = Path("src/kernel/predictors/vector.py").read_text()
    forbidden = {
        "vector_latency_calibration",
        "_VECTOR_AIVTIME",
        "_VECTOR_DUR_ADDITIVE",
        "mcblas_splitk_reduce_fixed_ns",
        "cublas_splitk_reduce_floor_ns",
    }

    assert forbidden.isdisjoint(source)


@pytest.mark.parametrize(
    "hardware_path,variant,shape,attributes",
    [
        (
            "configs/hardware/metax_c500.yaml",
            "moe_align_small_batch",
            {
                "tokens": 1024, "experts": 128, "topk": 8,
                "hidden": 4096, "intermediate": 14336,
            },
            {},
        ),
    ],
)
def test_vector_auxiliary_forward_has_no_fitted_pipeline_floor(
        hardware_path, variant, shape, attributes):
    hardware = HWSpec(hardware_path)
    kernel = KernelSpec(
        KernelKind.VECTOR,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=variant,
        launch=LaunchConfig(
            algorithm="test_vector_auxiliary",
            grid=(1, 1, 1),
            block=(128, 1, 1),
            execution_variant=variant,
        ),
        attributes=attributes,
    )

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert resource_input.pipeline_plan is None
    assert resource_input.support_level == "aggregate_only"
    assert resource_input.limitations == _GPU_VECTOR_LIMITATIONS[hardware_path]
    assert resource_input.launch_task_kind == "generic"
    assert resource_input.demand_per_resource["dram"] > 0.0


@pytest.mark.parametrize("split_k,k", ((2, 128), (3, 144), (4, 256), (5, 320)))
def test_split_k_gemm_prices_fp32_partials_at_l2_and_dram(split_k, k):
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    m, n = 32, 64
    launch = LaunchConfig(
        algorithm="mcblas_bf16_gemm",
        grid=(1, 1, split_k),
        block=(256, 1, 1),
        tiling=TilingSpec(
            tile_shape={"m": m, "n": n, "k": 16},
            split_k=split_k),
        parameters={
            "compulsory_traffic": 0.0,
            "weight_tile_rows": 0.0,
            "mte2_tile_bytes": 0.0,
            "aiv_mixed": 0.0,
            "n_major_traffic": 0.0,
            "split_k_partial_bytes_per_element": 4.0,
            "split_k_partial_dram_bytes_per_element": 4.0,
        },
        execution_variant="gemm",
    )
    kernel = KernelSpec(
        KernelKind.GEMM,
        hardware.hardware_id,
        {"b": 1, "m": m, "n": n, "k": k},
        dtype="bf16",
        variant="gemm",
        launch=launch,
    )

    resource_input, compulsory = create_predictor(
        kernel.kind).resource_input(kernel, hardware)

    input_bytes = float((m * k + k * n) * 2)
    partial_bytes = float(split_k * m * n * 4)
    assert resource_input.demand_per_resource["l2_cache"] == (
        input_bytes + partial_bytes)
    assert resource_input.demand_per_resource["dram"] == (
        input_bytes + partial_bytes)
    assert compulsory == input_bytes + partial_bytes
    paths = {path.name: path for path in resource_input.memory_paths}
    assert tuple(node.bytes for node in paths["global_read"].nodes) == (
        input_bytes, input_bytes)
    assert tuple(node.bytes for node in paths["global_write"].nodes) == (
        partial_bytes, partial_bytes)
    assert "split_k_partial_dram_residency_unavailable" not in (
        resource_input.limitations)


def test_gemm_without_tail_contract_rejects_non_divisible_k():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    shape = {"b": 1, "m": 32, "n": 64, "k": 130}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.GEMM, shape, "bf16", "gemm",
        {"has_bias": False, "position": "generic",
         "launch_policy": "analytical_launch"}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.GEMM,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=launch.execution_variant,
        launch=launch,
    )

    with pytest.raises(ValueError, match="no source-backed tail-tile semantics"):
        create_predictor(kernel.kind).resource_input(kernel, hardware)


@pytest.mark.parametrize("split_k", (2, 3, 4, 5))
def test_split_k_reducer_rejects_unproven_lifecycle(split_k):
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    elements = 8192
    kernel = KernelSpec(
        KernelKind.VECTOR,
        hardware.hardware_id,
        {"element_count": elements},
        dtype="bf16",
        variant="gemm_reduce",
        launch=LaunchConfig(
            algorithm="mcblas_gemm_reduce_unresolved_abi",
            grid=(8, 1, 1),
            block=(128, 1, 1),
            parameters={
                "split_k_partial_bytes_per_element": 4.0,
                "split_k_partial_dram_bytes_per_element": 4.0,
            },
            execution_variant="gemm_reduce",
        ),
        attributes={"split_k": split_k},
    )

    with pytest.raises(NotImplementedError, match="exact reducer launch geometry"):
        create_predictor(kernel.kind).resource_input(kernel, hardware)


def test_ascend_analytical_mixed_gemm_exposes_split_k_decomposition():
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    shape = {"b": 1, "m": 128, "n": 8192, "k": 28672}
    launch = select_launch_config(
        hardware.hardware_id,
        KernelKind.GEMM,
        shape,
        "bf16",
        "gemm",
        {
            "has_bias": False,
            "position": "generic",
            "launch_policy": "analytical_launch",
        },
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.GEMM,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="gemm",
        launch=launch,
    )

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)

    assert launch.parameters["aiv_mixed"] == 1.0
    assert resource_input.launch_task_kind == "mixed"
    assert resource_input.support_level == "partial"
    assert (
        "aiv_mixed_operation_decomposition_unavailable"
        not in resource_input.limitations)
    assert (
        "aic_aiv_collective_barrier_lifecycle_unavailable"
        not in resource_input.limitations)
    assert resource_input.demand_per_resource["compute_vector"] > 0.0
    assert resource_input.include_dispatch_overhead
    assert (
        "launch_task_duration_evidence_unavailable"
        not in resource_input.limitations)
    assert "l2_cache" not in resource_input.resource_time_overrides_ns


def test_ascend_pure_aic_kernels_consume_the_aic_empty_task_floor():
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    gemm_shape = {"b": 1, "m": 16, "n": 6144, "k": 4096}
    gemm_launch = select_launch_config(
        hardware.hardware_id, KernelKind.GEMM, gemm_shape, "bf16", "gemm",
        {"launch_policy": "injected_launch"}, hardware=hardware)
    gemm = KernelSpec(
        KernelKind.GEMM, hardware.hardware_id, gemm_shape, dtype="bf16",
        variant="gemm", launch=gemm_launch)

    moe_shape = {
        "tokens": 2, "experts": 2, "topk": 1,
        "hidden": 16, "intermediate": 32,
    }
    moe_launch = select_launch_config(
        hardware.hardware_id, KernelKind.FUSED_MOE, moe_shape, "bf16", "w1",
        {"launch_policy": "injected_launch", "expert_parallel_size": 1},
        hardware=hardware)
    moe = KernelSpec(
        KernelKind.FUSED_MOE, hardware.hardware_id, moe_shape, dtype="bf16",
        variant="w1", launch=moe_launch,
        attributes={
            "tokens_per_expert": (1, 1),
            "local_tokens_per_expert": (1, 1),
            "expert_parallel_size": 1,
            "expert_rank": 0,
            "owned_experts": (0, 1),
            "expert_weight_tile_rows": None,
        })

    for kernel in (gemm, moe):
        resource_input, _ = create_predictor(
            kernel.kind).resource_input(kernel, hardware)
        prediction = predict(kernel, hardware, "midlayer")

        assert resource_input.launch_task_kind == "aic"
        assert resource_input.include_dispatch_overhead
        assert prediction.t_total_ns >= hardware.aic_empty_kernel_duration_ns(
            kernel.launch.grid_size)
        assert (
            "launch_task_duration_evidence_unavailable"
            not in resource_input.limitations)


def test_ascend_fused_moe_predicts_source_backed_expert_parallel_launch():
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    shape = {
        "tokens": 2, "experts": 128, "topk": 1,
        "hidden": 16, "intermediate": 32,
    }
    global_counts = (1,) + (0,) * 63 + (1,) + (0,) * 63
    local_counts = (1,) + (0,) * 127
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.FUSED_MOE, shape, "bf16", "w1",
        {
            "launch_policy": "analytical_launch",
            "expert_parallel_size": 2,
            "local_tokens_per_expert": local_counts,
        },
        hardware=hardware)
    kernel = KernelSpec(
        KernelKind.FUSED_MOE,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="w1",
        launch=launch,
        attributes={
            "tokens_per_expert": global_counts,
            "local_tokens_per_expert": local_counts,
            "expert_parallel_size": 2,
            "expert_rank": 0,
            "owned_experts": tuple(range(64)),
            "expert_weight_tile_rows": None,
        },
    )

    resource_input, _ = create_predictor(
        kernel.kind).resource_input(kernel, hardware)
    assert resource_input.demand_per_resource["compute_tensor"] > 0
    assert predict(kernel, hardware, "midlayer").t_total_ns > 0


@pytest.mark.parametrize("variant", ("w1", "w2"))
def test_metax_fused_moe_predicts_source_backed_expert_parallel_launch(variant):
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    shape = {
        "tokens": 2, "experts": 128, "topk": 1,
        "hidden": 64, "intermediate": 64,
    }
    global_counts = (1,) + (0,) * 63 + (1,) + (0,) * 63
    local_counts = (1,) + (0,) * 127
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.FUSED_MOE, shape, "bf16", variant,
        {
            "launch_policy": "analytical_launch",
            "expert_parallel_size": 2,
        }, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.FUSED_MOE,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=variant,
        launch=launch,
        attributes={
            "tokens_per_expert": global_counts,
            "local_tokens_per_expert": local_counts,
            "expert_parallel_size": 2,
            "expert_rank": 0,
            "owned_experts": tuple(range(64)),
            "expert_weight_tile_rows": None,
        },
    )

    resource_input, _ = create_predictor(
        kernel.kind).resource_input(kernel, hardware)
    assert resource_input.demand_per_resource["compute_tensor"] > 0
    assert launch.tiling.tile_shape["k"] == 128
    assert "moe_gather_service_class_not_separated" in (
        resource_input.limitations)
    assert all(
        not path.access_class_demands for path in resource_input.memory_paths)
    assert predict(kernel, hardware, "midlayer").t_total_ns > 0


@pytest.mark.parametrize("hardware_id", ("ascend_910b", "metax_c500"))
def test_fused_moe_rejects_forged_expert_parallel_launch(hardware_id):
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    shape = {
        "tokens": 2, "experts": 128, "topk": 1,
        "hidden": 64, "intermediate": 64,
    }
    global_counts = (1,) + (0,) * 63 + (1,) + (0,) * 63
    local_counts = (1,) + (0,) * 127
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.FUSED_MOE, shape, "bf16", "w1",
        {"expert_parallel_size": 1}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.FUSED_MOE,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="w1",
        launch=launch,
        attributes={
            "tokens_per_expert": global_counts,
            "local_tokens_per_expert": local_counts,
            "expert_parallel_size": 2,
            "expert_rank": 0,
            "owned_experts": tuple(range(64)),
            "expert_weight_tile_rows": None,
        },
    )

    with pytest.raises(NotImplementedError, match="not source-backed"):
        create_predictor(kernel.kind).resource_input(kernel, hardware)


def test_ascend_flash_attn_exact_launch_consumes_mixed_empty_task_floor():
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    shape = {
        "batch": 1,
        "q_len": 1,
        "kv_len": 4096,
        "head_dim": 128,
        "num_query_heads": 32,
        "num_kv_heads": 8,
    }
    launch_attributes = {
        "is_decode": True,
        "causal": False,
        "kv_layout": "contiguous",
        "kv_dist": "skew",
        "q_lens": (1,),
        "kv_lens": (4096,),
    }
    launch = select_launch_config(
        hardware.hardware_id,
        KernelKind.FLASH_ATTN,
        shape,
        "bf16",
        "decode",
        launch_attributes,
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.FLASH_ATTN,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="decode",
        launch=launch,
        attributes={
            "causal": False,
            "kv_layout": "contiguous",
            "kv_dist": "skew",
            "fa_ksteps": None,
            "fa_eff_mfrac": 1.0,
            "q_lens": (1,),
            "kv_lens": (4096,),
        },
    )

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    prediction = predict(kernel, hardware, "midlayer")

    assert launch.grid == (8, 1, 1)
    assert resource_input.launch_task_kind == "mixed"
    assert resource_input.active_units == 8
    assert resource_input.support_level == "partial"
    assert prediction.t_total_ns >= hardware.mixed_empty_kernel_duration_ns(8)
    assert (
        "launch_task_duration_evidence_unavailable"
        not in resource_input.limitations)


@pytest.mark.parametrize(
    ("variant", "expected_alu", "expected_fma", "expected_bytes"),
    (
        ("moe_align", 8 * 4 + 16, 0, 16 * 4 * 2 + 8 * 4 * 4 * 2 * 2),
        (
            "moe_count_and_sort",
            8 * 4,
            0,
            16 * 4 * 2 + 8 * 4 * 4 * 2 * 2,
        ),
        (
            "moe_reduce",
            0,
            8 * 32 * (4 - 1),
            (8 * 4 * 32 + 8 * 32) * 2,
        ),
    ),
)
def test_moe_auxiliary_work_comes_from_semantic_iteration_counts(
        variant, expected_alu, expected_fma, expected_bytes):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {
        "tokens": 8,
        "experts": 16,
        "topk": 4,
        "hidden": 32,
        "intermediate": 64,
    }
    if variant == "moe_reduce":
        attributes = {"work_fraction": 1.0}
    elif variant in ("moe_align", "moe_count_and_sort"):
        attributes = {}
    else:
        raise AssertionError(f"unexpected test variant: {variant!r}")
    kernel = KernelSpec(
        KernelKind.VECTOR,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant=variant,
        launch=LaunchConfig(
            algorithm="test_semantic_moe_auxiliary",
            grid=(1, 1, 1),
            block=(128, 1, 1),
            execution_variant=variant,
        ),
        attributes=attributes,
    )

    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    demand = resource_input.demand_per_resource

    assert demand["compute_alu"] == expected_alu
    assert demand["compute_fma"] == expected_fma
    assert demand["l2_cache"] == expected_bytes
    assert demand["dram"] == expected_bytes


@pytest.mark.parametrize(
    ("cast_role", "expected_elements"),
    (
        ("topk_ids", 32768 * 2),
        ("expert_tokens", 8),
    ),
)
def test_ascend_moe_cast_work_follows_source_role(
        cast_role, expected_elements):
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    shape = {
        "tokens": 32768,
        "experts": 8,
        "topk": 2,
        "hidden": 4096,
        "intermediate": 14336,
    }
    attributes = {"cast_role": cast_role}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape,
        "bf16", "moe_cast", attributes, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR,
        hardware.hardware_id,
        shape,
        dtype="bf16",
        variant="moe_cast",
        launch=launch,
        attributes=attributes,
    )

    resource_input, total = create_predictor(
        kernel.kind).resource_input(kernel, hardware)

    expected_bytes = float(expected_elements * 4)
    assert total == 2 * expected_bytes
    assert resource_input.demand_per_resource["compute_vector"] == (
        expected_elements)
    assert resource_input.demand_per_resource["l2_cache"] == 2 * expected_bytes
    assert resource_input.demand_per_resource["dram"] == 2 * expected_bytes
    paths = {path.name: path for path in resource_input.memory_paths}
    assert set(paths) == {"mte2_read", "mte3_write"}
    assert all(
        node.bytes == expected_bytes
        for node in paths["mte2_read"].nodes)
    assert all(
        node.bytes == expected_bytes
        for node in paths["mte3_write"].nodes)


@pytest.mark.parametrize(
    ("fill_role", "expected_bytes"),
    (
        ("sorted_token_ids", 272 * 4),
        ("expert_ids", 17 * 4),
        ("num_tokens_post_padded", 4),
    ),
)
def test_metax_moe_fill_is_typed_int32_write_only(
        fill_role, expected_bytes):
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    variant = "moe_fill"
    kernel = KernelSpec(
        KernelKind.VECTOR,
        hardware.hardware_id,
        {
            "tokens": 8,
            "experts": 16,
            "topk": 4,
            "hidden": 32,
            "intermediate": 64,
        },
        dtype="bf16",
        variant=variant,
        launch=LaunchConfig(
            algorithm="observed_moe_aux",
            grid=(1, 1, 1),
            block=(256, 1, 1),
            execution_variant=variant,
        ),
        attributes={"fill_role": fill_role},
    )

    resource_input, total = create_predictor(
        kernel.kind).resource_input(kernel, hardware)

    assert total == expected_bytes
    assert resource_input.demand_per_resource["l2_cache"] == expected_bytes
    assert resource_input.demand_per_resource["dram"] == expected_bytes
    assert resource_input.demand_per_resource["compute_alu"] == 0
    assert resource_input.demand_per_resource["compute_fma"] == 0
    assert resource_input.demand_per_resource["compute_sfu"] == 0
    assert len(resource_input.memory_paths) == 1
    write_path = resource_input.memory_paths[0]
    assert write_path.name == "global_write"
    assert write_path.access_kind == "write"
    assert all(node.bytes == expected_bytes for node in write_path.nodes)


def test_predictor_rejects_hardware_mismatch():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, {"rows": 1, "hidden": 4096},
        "fp16", "rmsnorm", {}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, "metax_c500", {"rows": 1, "hidden": 4096},
        dtype="fp16", variant="rmsnorm", launch=launch)
    with pytest.raises(ValueError, match="does not match"):
        predict(kernel, hardware)


def test_predictor_import_purity():
    predictor_dir = Path("src/kernel/predictors")
    forbidden_import_parts = (
        "launch_config.select_launch_config",
        "launch_config.nvidia_a800",
        "launch_config.metax_c500",
        "launch_config.ascend_910b",
        ".validation",
        "serve_trace",
    )
    forbidden_names = {"runtime_name", "kernel_name", "duration_ns"}
    forbidden_modules = {"re", "json"}
    forbidden_calls = {
        "open", "load_json", "select_launch_config", "parse_gemm_tile",
        "parse_complete_gemm_tile", "parse_flash_attention_tile",
    }
    for path in predictor_dir.glob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = ["" if node.module is None else node.module]
            else:
                imports = []
            assert not any(part in imported for imported in imports for part in forbidden_import_parts), path
            assert not any(imported.split(".")[0] in forbidden_modules
                           for imported in imports), path
            if isinstance(node, ast.Name):
                assert node.id not in forbidden_names, path
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden_names, path
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in forbidden_names, path
            if isinstance(node, ast.Call):
                called = (
                    node.func.id if isinstance(node.func, ast.Name) else
                    node.func.attr if isinstance(node.func, ast.Attribute) else "")
                assert called not in forbidden_calls, path
