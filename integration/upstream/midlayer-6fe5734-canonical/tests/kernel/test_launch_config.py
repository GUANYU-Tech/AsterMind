import dataclasses
import hashlib
import json
import struct
from pathlib import Path

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_spec import PAGED_KV_CACHE_ATTRIBUTES
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config import FiaCrossDomainLaunchProgram
from src.kernel.launch_config.gemm_launch_selection import (
    CublasCtaRasterSelection,
    GemmLaunchSelection,
    gemm_launch_hardware_contract,
    gemm_launch_hardware_contract_sha256,
    load_gemm_launch_selection,
)
from src.kernel.launch_config.gemm_workload import (
    gemm_launch_selection_key,
    gemm_launch_selection_key_from_workload_key,
    gemm_workload_key,
    parse_cublaslt_stages,
)
from src.kernel.launch_config.ascend_910b import (
    fused_moe_launch as ascend_fused_moe_launch,
    flash_attn_launch as ascend_flash_attn_launch,
    gemm_launch as ascend_gemm_launch,
    vector_launch as ascend_vector_launch,
)
from src.kernel.launch_config.metax_c500 import (
    fused_moe_launch as metax_fused_moe_launch,
    gemm_launch as metax_c500_gemm_launch,
)
from src.kernel.launch_config.nvidia_a800 import (
    gemm_launch as nvidia_a800_gemm_launch,
)
from src.kernel.launch_config.ascend_910b import (
    vector_launch as ascend_vector_launch,
)
from src.kernel.launch_config.nvidia_a800 import (
    fused_moe_launch as nvidia_fused_moe_launch,
)
from src.kernel.launch_config.nvidia_a800.fused_moe_launch import (
    pytorch_2_8_moe_reduce_output_element_slices,
)
from src.kernel.launch_config.select_launch_config import (
    select_fused_moe_cast_roles,
    select_fused_moe_recipe,
    select_gemm_reducer_launch as _select_gemm_reducer_launch,
    select_launch_config as _select_launch_config,
)


def select_launch_config(
        hardware_id, kernel_kind, shape, dtype, variant, attributes,
        hardware):
    """Use the source-bound mode for legacy launch-unit expectations."""

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


@pytest.fixture(scope="module")
def hardware():
    return {
        "nvidia_a800": HWSpec("configs/hardware/nvidia_a800.yaml"),
        "metax_c500": HWSpec("configs/hardware/metax_c500.yaml"),
        "ascend_910b": HWSpec("configs/hardware/ascend_910b.yaml"),
    }


def test_gemm_launch_hardware_contract_tracks_only_launch_resources():
    scheduling = {
        "parallel_unit": "warp",
        "parallel_width": 1,
        "num_compute_units": 24,
        "max_warps_per_unit": 1,
        "max_blocks_per_unit": 1,
        "max_regs_per_unit": 65536,
        "max_smem_per_unit_bytes": 524288,
        "max_smem_per_block_bytes": 524288,
        "max_threads_per_unit": 1,
    }
    config = {
        "hardware_id": "ascend_910b",
        "scheduling": scheduling,
        "components": [
            {"role": "l1_cache", "size_bytes": 524288},
            {"role": "l2_cache", "size_bytes": 201326592},
        ],
        "memory_subsystem": {"dram_protocol_efficiency": 1.0},
    }
    baseline = gemm_launch_hardware_contract(
        config, hardware_id="ascend_910b")
    pricing_only = gemm_launch_hardware_contract(
        {
            **config,
            "memory_subsystem": {"dram_protocol_efficiency": 0.99},
        },
        hardware_id="ascend_910b",
    )
    changed_resources = gemm_launch_hardware_contract(
        {
            **config,
            "scheduling": {**scheduling, "num_compute_units": 23},
        },
        hardware_id="ascend_910b",
    )

    assert pricing_only == baseline
    assert gemm_launch_hardware_contract_sha256(pricing_only) == (
        gemm_launch_hardware_contract_sha256(baseline))
    assert gemm_launch_hardware_contract_sha256(changed_resources) != (
        gemm_launch_hardware_contract_sha256(baseline))


def test_production_launch_config_excludes_compiler_resource_usage(hardware):
    launch = select_launch_config(
        "nvidia_a800", "GEMM", {"b": 1, "m": 32, "n": 64, "k": 64},
        "bf16", "gemm", {
            "has_bias": False,
            "launch_policy": "analytical_launch",
        },
        hardware=hardware["nvidia_a800"])

    assert not hasattr(launch, "shared_memory_bytes")
    assert not hasattr(launch, "registers_per_thread")


def test_ascend_910b_selector_returns_typed_source_program(hardware):
    shape = {
        "batch": 1,
        "q_len": 1,
        "kv_len": 4096,
        "head_dim": 128,
        "num_query_heads": 32,
        "num_kv_heads": 32,
    }
    attributes = {
        "is_decode": True,
        "causal": False,
        "kv_layout": "contiguous",
        "kv_dist": "skew",
        "q_lens": (1,),
        "kv_lens": (4096,),
    }
    launch = select_launch_config(
        "ascend_910b", "FLASH_ATTN", shape, "bf16", "decode",
        attributes, hardware=hardware["ascend_910b"])

    assert isinstance(launch.source_program, FiaCrossDomainLaunchProgram)
    assert len(launch.source_program.aic_groups) == launch.grid_size == 16
    assert {len(group.tasks) for group in launch.source_program.aic_groups} == {2}


def test_ascend_910b_source_schedule_rejects_endpoint_tamper():
    shape = {
        "batch": 1,
        "q_len": 1,
        "kv_len": 1024,
        "head_dim": 128,
        "num_query_heads": 32,
        "num_kv_heads": 8,
    }
    attributes = {
        "is_decode": True,
        "causal": False,
        "kv_layout": "contiguous",
        "kv_dist": "skew",
        "q_lens": (1,),
        "kv_lens": (1024,),
    }
    launch = ascend_flash_attn_launch.select_cann_fia_launch(
        shape, "bf16", "decode", attributes)
    schedule = json.loads(json.dumps(launch["source_schedule"]))
    schedule["worker_endpoints"][0] = [0, 0, 0]
    entry = {
        "runtime_variant": launch["runtime_variant"],
        "block_dim": launch["block_dim"],
    }
    with pytest.raises(ValueError, match="strictly ordered"):
        ascend_flash_attn_launch._source_schedule_parameters(
            schedule, shape, attributes, entry)


def test_ascend_910b_source_schedule_rejects_gqa_group_split():
    shape = {
        "batch": 1,
        "q_len": 1,
        "kv_len": 1024,
        "head_dim": 128,
        "num_query_heads": 32,
        "num_kv_heads": 8,
    }
    attributes = {
        "is_decode": True,
        "causal": False,
        "kv_layout": "contiguous",
        "kv_dist": "skew",
        "q_lens": (1,),
        "kv_lens": (1024,),
    }
    launch = ascend_flash_attn_launch.select_cann_fia_launch(
        shape, "bf16", "decode", attributes)
    schedule = json.loads(json.dumps(launch["source_schedule"]))
    schedule["tile"][0] = 1
    entry = {
        "runtime_variant": launch["runtime_variant"],
        "block_dim": launch["block_dim"],
    }

    with pytest.raises(
            NotImplementedError, match="splits one GQA group"):
        ascend_flash_attn_launch._source_schedule_parameters(
            schedule, shape, attributes, entry)


def test_ascend_910b_paged_long_decode_accepts_source_flashdecode_variant(
        hardware):
    """Derive the FlashDecode contract from the source tile flags."""

    shape = {
        "batch": 1,
        "q_len": 1,
        "kv_len": 16384,
        "head_dim": 128,
        "num_query_heads": 32,
        "num_kv_heads": 8,
    }
    attributes = {
        "is_decode": True,
        "causal": False,
        "kv_layout": "paged_varlen",
        "kv_dist": "uniform",
        "q_lens": (1,),
        "kv_lens": (16384,),
    }

    launch = select_launch_config(
        "ascend_910b", "FLASH_ATTN", shape, "bf16", "decode",
        attributes, hardware=hardware["ascend_910b"])

    assert launch.algorithm == "cann_flash_attention_score:103000000010722221"
    assert launch.parameters[
        "cann_source_flash_decode_terminal_read_bytes"] > 0.0


@pytest.mark.parametrize(
    "variant,shape,expected_grid,expected_block",
    [
        ("rmsnorm", {"rows": 64, "hidden": 4096},
         64, 1024),
        ("add_rmsnorm", {"rows": 64, "hidden": 4096},
         64, 1024),
        ("add_rmsnorm", {"rows": 256, "hidden": 4096},
         256, 256),
        ("rotary", {
            "seq_len": 1024, "num_query_heads": 32,
            "num_kv_heads": 4, "head_dim": 128,
            "element_count": 4718592,
        }, 1024, 512),
        ("activation", {
            "tok": 1024, "intermediate": 6144,
            "element_count": 6291456,
        }, 1024, 1024),
        ("paged_kv_cache", {
            "seq_len": 1024, "num_kv_heads": 4, "head_dim": 128,
            "element_count": 524288,
        }, 1024, 512),
        ("elementwise", {"element_count": 16777216},
         16384, 128),
    ],
)
def test_nvidia_a800_vector_launch_matches_vllm(
        hardware, variant, shape, expected_grid, expected_block):
    if variant == "elementwise":
        attributes = {"operation": "add"}
        kind = "VECTOR"
    elif variant == "paged_kv_cache":
        attributes = PAGED_KV_CACHE_ATTRIBUTES
        kind = "KV_CACHE"
    else:
        attributes = {}
        kind = "VECTOR"
    launch = select_launch_config(
        "nvidia_a800", kind, shape, "bf16", variant,
        attributes,
        hardware=hardware["nvidia_a800"])
    assert launch.grid == (expected_grid, 1, 1)
    assert launch.block == (expected_block, 1, 1)
    expected_parameters = {}
    if variant in ("rmsnorm", "add_rmsnorm"):
        expected_parameters["input_reread_count"] = 1.0
        expected_parameters["norm_reduction_tree_flops"] = float(
            expected_grid * (expected_block - 1))
        expected_parameters["norm_epsilon_add_flops"] = float(expected_grid)
        if variant == "rmsnorm":
            expected_parameters["source_bound_memory_dependency_task"] = 1.0
            expected_parameters["source_bound_block_reduction"] = 1.0
            expected_parameters["norm_weight_l2_copies"] = float(
                expected_grid)
            if (expected_grid == 64
                    and 2048 <= shape["hidden"] <= 16384):
                expected_parameters[
                    "source_bound_gpu_norm_cta_pipeline"] = 1.0
            expected_parameters[
                "source_bound_l2_read_transactions_per_warp"] = 2.0
        else:
            expected_parameters["source_bound_memory_dependency_task"] = 1.0
            expected_parameters["source_bound_block_reduction"] = 1.0
            expected_parameters[
                "source_bound_gpu_norm_cta_pipeline"] = 1.0
            expected_parameters[
                "source_bound_l2_read_transactions_per_warp"] = 32.0
            expected_parameters[
                "source_bound_l2_read_warps_per_cta"] = float(
                    (min(expected_block, (shape["hidden"] + 7) // 8) + 31)
                    // 32)
            expected_parameters[
                "source_bf16_conversion_operations"] = float(
                    6 * expected_grid * shape["hidden"])
    elif variant == "rotary":
        expected_parameters["rotary_cache_path_bytes"] = float(
            shape["element_count"] * 2)
        expected_parameters["rotary_position_path_bytes"] = float(
            expected_grid * (expected_block // 32) * 32)
    elif variant == "paged_kv_cache":
        expected_parameters["kv_cache_slot_l1_bytes"] = float(
            expected_grid * (expected_block // 32) * 32)
        expected_parameters[
            "source_bound_l2_read_transactions_per_warp"] = 16.0
        expected_parameters["source_bound_l2_read_warps_per_cta"] = 2.0
        expected_parameters["source_bound_resident_ctas_per_unit"] = 2.0
        expected_parameters["source_bound_memory_dependency_task"] = 1.0
    assert launch.parameters == expected_parameters


def test_nvidia_a800_partial_elementwise_block_uses_generic_resources(hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"element_count": 2560}, "bf16",
        "elementwise", {"operation": "add"},
        hardware=hardware["nvidia_a800"])

    assert launch.grid == (3, 1, 1)
    assert launch.parameters == {}


def test_nvidia_a800_rmsnorm_tail_keeps_source_read_contract(hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"rows": 16, "hidden": 2560}, "bf16",
        "rmsnorm", {}, hardware=hardware["nvidia_a800"])

    assert launch.parameters[
        "source_bound_l2_read_transactions_per_warp"] == 2.0
    assert launch.parameters["source_bound_block_reduction"] == 1.0


def test_nvidia_a800_multiwave_rmsnorm_publishes_source_cta_pipeline(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"rows": 256, "hidden": 4096}, "bf16",
        "rmsnorm", {}, hardware=hardware["nvidia_a800"])

    assert launch.parameters[
        "source_bound_gpu_norm_cta_pipeline"] == 1.0
    assert launch.parameters["norm_weight_l2_copies"] == 148.0


@pytest.mark.parametrize(
    "hidden,expected",
    ((1024, True), (2048, True), (16384, True), (16640, True)),
)
def test_nvidia_a800_grid64_rmsnorm_publishes_static_source_program(
        hardware, hidden, expected):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"rows": 64, "hidden": hidden},
        "bf16", "rmsnorm", {}, hardware=hardware["nvidia_a800"])

    assert (
        "source_bound_gpu_norm_cta_pipeline" in launch.parameters
    ) is expected


@pytest.mark.parametrize(
    ("rows", "expected_copies"),
    ((1, 1), (108, 108), (109, 108), (216, 108), (217, 109),
     (1024, 540)),
)
def test_nvidia_a800_rmsnorm_gamma_copies_follow_resident_cta_groups(
        hardware, rows, expected_copies):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"rows": rows, "hidden": 16384},
        "bf16", "rmsnorm", {}, hardware=hardware["nvidia_a800"])

    assert launch.parameters["norm_weight_l2_copies"] == float(
        expected_copies)


def test_nvidia_a800_rmsnorm_gamma_residency_stops_beyond_evidence(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"rows": 256, "hidden": 16640},
        "bf16", "rmsnorm", {}, hardware=hardware["nvidia_a800"])

    assert "norm_weight_l2_copies" not in launch.parameters


@pytest.mark.parametrize("element_count", (2048, 2560, 16777216))
@pytest.mark.parametrize("operation", ("add", "mul"))
def test_nvidia_a800_typed_pytorch_vec4_publishes_analytical_task(
        hardware, element_count, operation):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"element_count": element_count},
        "bf16", "elementwise_pytorch_vec4", {"operation": operation},
        hardware=hardware["nvidia_a800"])

    expected_parameters = {
        "source_bound_memory_dependency_task": 1.0,
        "warp_instructions": (
            200.0 * (element_count // 1024)
            + (700.0 if element_count % 1024 else 0.0)),
    }
    if element_count % 1024 == 0:
        expected_parameters[
            "source_bound_l2_read_transactions_per_warp"] = 32.0
        expected_parameters["source_bound_l2_read_warps_per_cta"] = 4.0
    assert launch.parameters == expected_parameters


def test_nvidia_a800_typed_pytorch_vec4_unknown_tail_keeps_issue_absent(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"element_count": 2049}, "bf16",
        "elementwise_pytorch_vec4", {"operation": "add"},
        hardware=hardware["nvidia_a800"])

    assert launch.parameters == {}


def test_nvidia_a800_typed_pytorch_vec4_fp16_does_not_borrow_bf16_cfg(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"element_count": 2048}, "fp16",
        "elementwise_pytorch_vec4", {"operation": "add"},
        hardware=hardware["nvidia_a800"])

    assert launch.parameters == {}


def test_nvidia_a800_typed_pytorch_vec4_rejects_memory_only_operation(
        hardware):
    with pytest.raises(
            ValueError, match="typed PyTorch vec4 elementwise requires"):
        select_launch_config(
            "nvidia_a800", "VECTOR", {"element_count": 2048}, "bf16",
            "elementwise_pytorch_vec4", {"operation": "copy"},
            hardware=hardware["nvidia_a800"])


def test_nvidia_a800_typed_rope_publishes_analytical_task(hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR",
        {
            "seq_len": 1,
            "num_query_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "rotary_dim": 128,
        },
        "bf16", "rope", {"is_neox": True, "key_present": True},
        hardware=hardware["nvidia_a800"])

    assert launch.grid == (1, 1, 1)
    assert launch.block == (512, 1, 1)
    assert launch.parameters == {
        "rotary_cache_path_bytes": 10240.0,
        "rotary_position_path_bytes": 512.0,
        "source_bound_memory_dependency_task": 1.0,
        "source_bound_l2_read_transactions_per_warp": 8.0,
    }


@pytest.mark.parametrize(
    "shape,attributes",
    (
        (
            {
                "seq_len": 1, "num_query_heads": 32,
                "num_kv_heads": 8, "head_dim": 128, "rotary_dim": 64,
            },
            {"is_neox": True, "key_present": True},
        ),
        (
            {
                "seq_len": 1, "num_query_heads": 32,
                "num_kv_heads": 8, "head_dim": 128, "rotary_dim": 128,
            },
            {"is_neox": True, "key_present": False},
        ),
    ),
)
def test_nvidia_a800_typed_rope_does_not_borrow_sass_outside_fixed_binary(
        hardware, shape, attributes):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", shape, "bf16", "rope", attributes,
        hardware=hardware["nvidia_a800"])

    assert "warp_instructions" not in launch.parameters
    assert "lowered_sfu_operations" not in launch.parameters


@pytest.mark.parametrize(
    ("tokens", "experts", "topk", "expected"),
    (
        (
            512,
            2,
            1,
            (
                "moe_topk", "moe_align_small_batch", "w1",
                "moe_act_and_mul", "w2", "moe_reduce",
            ),
        ),
        (
            1024,
            2,
            1,
            (
                "moe_topk", "moe_align", "moe_count_and_sort", "w1",
                "moe_act_and_mul", "w2", "moe_reduce",
            ),
        ),
        (
            512,
            128,
            2,
            (
                "moe_topk", "moe_align", "moe_count_and_sort", "w1",
                "moe_act_and_mul", "w2", "moe_sum",
            ),
        ),
    ),
)
def test_nvidia_a800_fused_moe_recipe_matches_vllm_0110_source(
        hardware, tokens, experts, topk, expected):
    shape = {
        "tokens": tokens,
        "experts": experts,
        "topk": topk,
        "hidden": 4096,
        "intermediate": 14336,
    }

    assert select_fused_moe_recipe(
        "nvidia_a800", shape, hardware["nvidia_a800"]) == expected


@pytest.mark.parametrize(
    ("variant", "topk", "tokens", "hidden", "intermediate",
     "expected_grid", "expected_block"),
    (
        ("moe_act_and_mul", 4, 8, 32, 64, (32, 1, 1), (64, 1, 1)),
        ("moe_sum", 4, 8, 32, 64, (8, 1, 1), (32, 1, 1)),
        ("moe_reduce", 1, 8, 128, 64, (2, 1, 1), (128, 1, 1)),
        ("moe_reduce", 8, 1, 2048, 64, (4, 1, 1), (32, 4, 1)),
    ),
)
def test_nvidia_a800_moe_auxiliary_launch_matches_source_geometry(
        hardware, variant, topk, tokens, hidden, intermediate,
        expected_grid, expected_block):
    shape = {
        "tokens": tokens,
        "experts": 128,
        "topk": topk,
        "hidden": hidden,
        "intermediate": intermediate,
    }

    if variant == "moe_reduce":
        attributes = {"work_fraction": 1.0}
    elif variant in ("moe_sum", "moe_act_and_mul"):
        attributes = {}
    else:
        raise AssertionError(f"unexpected test variant: {variant!r}")
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", shape, "bf16", variant, attributes,
        hardware=hardware["nvidia_a800"])

    assert launch.grid == expected_grid
    assert launch.block == expected_block
    assert launch.execution_variant == variant


@pytest.mark.parametrize(
    ("tokens", "topk", "hidden", "expected"),
    (
        (16384, 16, 4096, ((0, 16384 * 4096),)),
        (
            16385, 16, 4096,
            ((0, 8192 * 4096), (8192 * 4096, 8193 * 4096)),
        ),
        (
            32768, 8, 7168,
            ((0, 16384 * 7168), (16384 * 7168, 16384 * 7168)),
        ),
        (
            32769, 16, 4096,
            (
                (0, 16384 * 4096),
                (16384 * 4096, 8192 * 4096),
                (24576 * 4096, 8193 * 4096),
            ),
        ),
    ),
)
def test_pytorch_2_8_moe_reduce_uses_ordered_output_element_slices(
        tokens, topk, hidden, expected):
    shape = {
        "tokens": tokens,
        "experts": 256,
        "topk": topk,
        "hidden": hidden,
        "intermediate": 2048,
    }

    assert pytorch_2_8_moe_reduce_output_element_slices(shape) == expected


def test_pytorch_2_8_topk_one_splits_coalesced_flat_output_elements():
    shape = {
        "tokens": 357_913_942,
        "experts": 256,
        "topk": 1,
        "hidden": 3,
        "intermediate": 2048,
    }

    # TensorIterator coalesces topk=1 into one flat dimension.  Its midpoint
    # is deliberately not a token boundary (three output elements per token).
    assert pytorch_2_8_moe_reduce_output_element_slices(shape) == (
        (0, 536_870_913),
        (536_870_913, 536_870_913),
    )


def test_nvidia_a800_moe_reduce_block_y_reduction_launches_all_outputs(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR",
        {
            "tokens": 1,
            "experts": 128,
            "topk": 64,
            "hidden": 4096,
            "intermediate": 2048,
        },
        "bf16", "moe_reduce", {"work_fraction": 1.0},
        hardware=hardware["nvidia_a800"])

    assert launch.grid == (32, 1, 1)
    assert launch.block == (32, 4, 1)


def test_nvidia_a800_moe_reduce_output_vector_width_respects_hidden_extent(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR",
        {
            "tokens": 2,
            "experts": 128,
            "topk": 8,
            "hidden": 2,
            "intermediate": 64,
        },
        "bf16", "moe_reduce", {"work_fraction": 1.0},
        hardware=hardware["nvidia_a800"])

    assert launch.grid == (1, 1, 1)
    assert launch.block == (2, 8, 1)


def test_nvidia_a800_large_moe_recipe_repeats_each_physical_reduce(
        hardware):
    shape = {
        "tokens": 32768,
        "experts": 256,
        "topk": 8,
        "hidden": 7168,
        "intermediate": 2048,
    }

    recipe = select_fused_moe_recipe(
        "nvidia_a800", shape, hardware["nvidia_a800"])

    assert recipe[-2:] == ("moe_reduce", "moe_reduce")


def test_nvidia_a800_fused_moe_recipe_rejects_unimplemented_topk_family(
        hardware):
    shape = {
        "tokens": 8,
        "experts": 3,
        "topk": 1,
        "hidden": 64,
        "intermediate": 16,
    }

    with pytest.raises(NotImplementedError, match="topk_softmax"):
        select_fused_moe_recipe(
            "nvidia_a800", shape, hardware["nvidia_a800"])


@pytest.mark.parametrize("experts", (192, 320, 448, 576))
def test_nvidia_a800_fused_moe_recipe_does_not_advertise_invalid_topk_warps(
        hardware, experts):
    shape = {
        "tokens": 8,
        "experts": experts,
        "topk": 1,
        "hidden": 64,
        "intermediate": 16,
    }

    with pytest.raises(NotImplementedError, match="topk_softmax"):
        select_fused_moe_recipe(
            "nvidia_a800", shape, hardware["nvidia_a800"])


def _c500_qwen3_fused_moe_shape(tokens):
    return {
        "tokens": tokens,
        "experts": 128,
        "topk": 8,
        "hidden": 2048,
        "intermediate": 768,
    }


@pytest.mark.parametrize(
    ("variant", "expected_grid"),
    (("w1", 3216), ("w2", 4288)),
)
def test_metax_c500_fused_moe_resolves_selected_deployment_launch(
        hardware, variant, expected_grid):
    shape = {
        "tokens": 40,
        "experts": 128,
        "topk": 8,
        "hidden": 2048,
        "intermediate": 768,
    }

    launch = select_launch_config(
        "metax_c500",
        "FUSED_MOE",
        shape,
        "bf16",
        variant,
        {"expert_parallel_size": 2, "launch_policy": "analytical_launch"},
        hardware=hardware["metax_c500"],
    )

    assert launch.algorithm == f"triton_fused_moe_{variant}"
    assert launch.grid == (expected_grid, 1, 1)
    assert launch.block == (256, 1, 1)
    assert launch.tiling.tile_shape == {"m": 32, "n": 64, "k": 128}
    assert launch.parameters["expert_parallel_size"] == 2.0
    assert launch.parameters["group_size_m"] == 1.0
    assert launch.parameters["padded_rows"] == 4288.0
    assert launch.parameters["dynamic_shared_memory_bytes"] == 49152.0


@pytest.mark.parametrize(
    ("experts", "expert_parallel_size", "hidden"),
    (
        (64, 1, 640),
        (128, 2, 1280),
        (160, 1, 192),
        (256, 1, 64),
    ),
)
def test_nvidia_a800_fused_moe_rejects_unrepresented_bundled_tuned_config(
        hardware, experts, expert_parallel_size, hidden):
    with pytest.raises(NotImplementedError, match="bundled tuned config"):
        select_launch_config(
            "nvidia_a800",
            "FUSED_MOE",
            {
                "tokens": 2,
                "experts": experts,
                "topk": 2,
                "hidden": hidden,
                "intermediate": 4096,
            },
            "bf16",
            "w1",
            {
                "expert_parallel_size": expert_parallel_size,
                "launch_policy": "analytical_launch",
            },
            hardware=hardware["nvidia_a800"],
        )


def test_metax_c500_fused_moe_selected_launch_matches_retained_raw(hardware):
    evidence = json.loads(Path(
        "microbench/metax_c500/references/"
        "fused_moe_runtime_config_vllm_metax_0_24_0_g436274.json"
    ).read_text())
    observed = evidence["selected_launch_observation"]
    shape = evidence["selected_deployment_input"]
    selector_shape = {
        "tokens": shape["graph_physical_tokens"],
        "experts": shape["global_experts"],
        "topk": shape["topk"],
        "hidden": shape["hidden_size"],
        "intermediate": shape["moe_intermediate_size"],
    }
    attributes = {
        "expert_parallel_size": shape["expert_parallel_size"],
        "launch_policy": "analytical_launch",
    }

    for projection in ("w1", "w2"):
        launch = select_launch_config(
            "metax_c500", "FUSED_MOE", selector_shape, "bf16", projection,
            attributes, hardware=hardware["metax_c500"])
        assert {
            "grid": launch.grid[0],
            "block": launch.block[0],
            "dynamic_shared_memory_bytes": launch.parameters[
                "dynamic_shared_memory_bytes"],
        } == observed[projection]


@pytest.mark.parametrize(
    ("tokens", "expected_w1_grid", "expected_w2_grid", "expected_block"),
    (
        (12, 2304, 3072, 256),
        (21, 3144, 4192, 256),
        (195, 1812, 2416, 512),
        (223, 1848, 2464, 512),
        (356, 2052, 2736, 512),
        (729, 2076, 2768, 512),
    ),
)
def test_metax_c500_target_triton_default_reuses_w1_routing_buffer(
        hardware, tokens, expected_w1_grid, expected_w2_grid, expected_block):
    attributes = {
        "expert_parallel_size": 2,
        "launch_policy": "analytical_launch",
    }
    shape = _c500_qwen3_fused_moe_shape(tokens)
    w1 = select_launch_config(
        "metax_c500", "FUSED_MOE", shape, "bf16", "w1", attributes,
        hardware=hardware["metax_c500"])
    w2 = select_launch_config(
        "metax_c500", "FUSED_MOE", shape, "bf16", "w2", attributes,
        hardware=hardware["metax_c500"])
    assert w1.grid == (expected_w1_grid, 1, 1)
    assert w2.grid == (expected_w2_grid, 1, 1)
    assert w1.block == (expected_block, 1, 1)
    assert w2.block == (expected_block, 1, 1)


@pytest.mark.parametrize(
    ("tokens", "expected_tile", "expected_shared_memory"),
    (
        (32, {"m": 16, "n": 64, "k": 128}, 40960.0),
        (33, {"m": 32, "n": 64, "k": 128}, 49152.0),
        (65, {"m": 32, "n": 128, "k": 64}, 40960.0),
        (97, {"m": 64, "n": 128, "k": 64}, 49152.0),
        (129, {"m": 64, "n": 128, "k": 64}, 49152.0),
        (513, {"m": 128, "n": 128, "k": 64}, 65536.0),
    ),
)
def test_metax_c500_fused_moe_default_rule_tracks_source_boundaries(
        hardware, tokens, expected_tile, expected_shared_memory):
    launch = select_launch_config(
        "metax_c500", "FUSED_MOE", _c500_qwen3_fused_moe_shape(tokens),
        "bf16", "w1",
        {"expert_parallel_size": 2, "launch_policy": "analytical_launch"},
        hardware=hardware["metax_c500"],
    )

    assert launch.tiling.tile_shape == expected_tile
    assert launch.parameters["dynamic_shared_memory_bytes"] == (
        expected_shared_memory)


@pytest.mark.parametrize(
    ("tokens", "expected_tile", "expected_shared_memory"),
    (
        (1, {"m": 16, "n": 64, "k": 128}, 40960.0),
        (32, {"m": 16, "n": 64, "k": 128}, 40960.0),
        (33, {"m": 32, "n": 64, "k": 128}, 49152.0),
        (64, {"m": 32, "n": 64, "k": 128}, 49152.0),
        (65, {"m": 32, "n": 128, "k": 64}, 40960.0),
        (96, {"m": 32, "n": 128, "k": 64}, 40960.0),
        (97, {"m": 64, "n": 128, "k": 64}, 49152.0),
        (128, {"m": 64, "n": 128, "k": 64}, 49152.0),
        (129, {"m": 64, "n": 128, "k": 64}, 49152.0),
        (512, {"m": 64, "n": 128, "k": 64}, 49152.0),
        (513, {"m": 128, "n": 128, "k": 64}, 65536.0),
    ),
)
@pytest.mark.parametrize("projection", ("w1", "w2"))
def test_metax_c500_fused_moe_generic_no_config_tracks_source_default(
        hardware, tokens, expected_tile, expected_shared_memory, projection):
    launch = select_launch_config(
        "metax_c500", "FUSED_MOE",
        {
            "tokens": tokens,
            "experts": 128,
            "topk": 8,
            "hidden": 64,
            "intermediate": 768,
        },
        "bf16", projection,
        {"expert_parallel_size": 2, "launch_policy": "analytical_launch"},
        hardware=hardware["metax_c500"],
    )

    assert launch.tiling.tile_shape == expected_tile
    assert launch.parameters["dynamic_shared_memory_bytes"] == (
        expected_shared_memory)


@pytest.mark.parametrize(
    ("tokens", "expected_group_size"),
    (
        (128, 1.0),
        (129, 16.0),
        (512, 16.0),
    ),
)
@pytest.mark.parametrize("projection", ("w1", "w2"))
def test_metax_c500_fused_moe_default_group_size_tracks_local_experts(
        hardware, tokens, expected_group_size, projection):
    """Keep C500's ``M // local_E > 128`` source branch explicit."""

    launch = select_launch_config(
        "metax_c500", "FUSED_MOE",
        {
            "tokens": tokens,
            "experts": 1,
            "topk": 1,
            "hidden": 64,
            "intermediate": 768,
        },
        "bf16", projection,
        {"expert_parallel_size": 1, "launch_policy": "analytical_launch"},
        hardware=hardware["metax_c500"],
    )

    assert launch.tiling.tile_shape == {"m": 64, "n": 128, "k": 64}
    assert launch.parameters["group_size_m"] == expected_group_size
    # BLOCK_SIZE_M=64 maps to 512 threads in the compiler launch records.
    assert launch.block == (512, 1, 1)


@pytest.mark.parametrize("projection", ("w1", "w2"))
def test_metax_c500_fused_moe_launch_policy_uses_runtime_resolver(
        hardware, projection):
    shape = {
        "tokens": 40,
        "experts": 128,
        "topk": 8,
        "hidden": 64,
        "intermediate": 768,
    }
    runtime_launch = select_launch_config(
        "metax_c500", "FUSED_MOE", shape, "bf16", projection,
        {"expert_parallel_size": 2, "launch_policy": "analytical_launch"},
        hardware=hardware["metax_c500"])
    injected_launch = select_launch_config(
        "metax_c500", "FUSED_MOE", shape, "bf16", projection,
        {"expert_parallel_size": 2, "launch_policy": "injected_launch"},
        hardware=hardware["metax_c500"])

    assert runtime_launch.tiling.tile_shape == {"m": 32, "n": 64, "k": 128}
    assert injected_launch == runtime_launch


@pytest.mark.parametrize(
    ("experts", "expert_parallel_size", "hidden", "intermediate"),
    (
        # H=2560's staged W1/W2 JSON differs from the generic formula.
        (128, 2, 2560, 768),
        # A second H-scoped file exercises a distinct H/E/N domain.
        (128, 1, 2048, 192),
        # This is the package-default config, not an H-scoped file.
        (256, 1, 64, 256),
    ),
)
def test_metax_c500_fused_moe_rejects_matching_installed_tuned_config(
        hardware, experts, expert_parallel_size, hidden, intermediate):
    with pytest.raises(NotImplementedError, match="matching installed"):
        select_launch_config(
            "metax_c500",
            "FUSED_MOE",
            {
                "tokens": 40,
                "experts": experts,
                "topk": 8,
                "hidden": hidden,
                "intermediate": intermediate,
            },
            "bf16",
            "w1",
            {
                "expert_parallel_size": expert_parallel_size,
                "launch_policy": "analytical_launch",
            },
            hardware=hardware["metax_c500"],
        )


def test_metax_c500_fused_moe_resolver_obeys_override_and_nearest_tuned_key():
    override = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "num_warps": 4,
        "num_stages": 4,
    }
    tuned = {
        32: {
            "stage1": override,
            "stage2": {
                "BLOCK_SIZE_M": 32,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 3,
            },
        },
        128: {
            "stage1": override,
            "stage2": {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 128,
                "BLOCK_SIZE_K": 64,
                "GROUP_SIZE_M": 1,
                "num_warps": 8,
                "num_stages": 3,
            },
        },
    }

    override_selected = metax_fused_moe_launch.resolve_fused_moe_runtime_config(
        batch_tokens=100,
        local_experts=64,
        projection="w1",
        batch_invariant=False,
        override_config=override,
        tuned_configs=tuned,
    )
    nearest_tuned = metax_fused_moe_launch.resolve_fused_moe_runtime_config(
        batch_tokens=100,
        local_experts=64,
        projection="w2",
        batch_invariant=False,
        override_config=None,
        tuned_configs=tuned,
    )
    staged_w1 = metax_fused_moe_launch.resolve_fused_moe_runtime_config(
        batch_tokens=32,
        local_experts=64,
        projection="w1",
        batch_invariant=False,
        override_config=None,
        tuned_configs=tuned,
    )
    staged_w2 = metax_fused_moe_launch.resolve_fused_moe_runtime_config(
        batch_tokens=32,
        local_experts=64,
        projection="w2",
        batch_invariant=False,
        override_config=None,
        tuned_configs=tuned,
    )

    assert (override_selected.source, override_selected.batch_key,
            override_selected.block_size_m) == ("override", None, 16)
    assert (nearest_tuned.source, nearest_tuned.batch_key,
            nearest_tuned.block_size_m) == ("tuned", 128, 64)
    assert (staged_w1.batch_key, staged_w1.block_size_m) == (32, 16)
    assert (staged_w2.batch_key, staged_w2.block_size_m) == (32, 32)


@pytest.mark.parametrize(
    ("override_config", "batch_invariant", "error", "message"),
    (
        ({
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
            "SPLIT_K": 2,
        }, False, NotImplementedError, "SPLIT_K != 1"),
        ({
            "BLOCK_SIZE_M": 7,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
        }, False, NotImplementedError, "no source-bound compiler resource"),
        ({
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1,
            "num_warps": 4,
            "num_stages": 3,
            "stale_runtime_key": 1,
        }, False, ValueError, "unknown keys"),
        (None, True, NotImplementedError, "batch-invariant"),
    ),
)
def test_metax_c500_fused_moe_resolver_rejects_unclosed_runtime_paths(
        override_config, batch_invariant, error, message):
    with pytest.raises(error, match=message):
        metax_fused_moe_launch.resolve_fused_moe_runtime_config(
            batch_tokens=40,
            local_experts=64,
            projection="w1",
            batch_invariant=batch_invariant,
            override_config=override_config,
            tuned_configs=None,
        )


@pytest.mark.parametrize(
    ("hardware_id", "message"),
    (("metax_c500", "only EP1 or EP2"),
     ("ascend_910b", "only EP1 or EP2")),
)
def test_fused_moe_launch_rejects_expert_parallel_size_outside_source_scope(
        hardware, hardware_id, message):
    attributes = {
        "expert_parallel_size": 3,
        "launch_policy": "analytical_launch",
    }
    if hardware_id == "ascend_910b":
        attributes["local_tokens_per_expert"] = (1, 0, 0, 0, 0, 0)
    with pytest.raises(NotImplementedError, match=message):
        select_launch_config(
            hardware_id,
            "FUSED_MOE",
            {
                "tokens": 1,
                "experts": 6,
                "topk": 1,
                "hidden": 64,
                "intermediate": 64,
            },
            "bf16",
            "w1",
            attributes,
            hardware=hardware[hardware_id],
        )


@pytest.mark.parametrize(
    ("hardware_id", "message"),
    (("metax_c500", "requires 128 experts"),
     ("ascend_910b", "requires 128 experts")),
)
def test_fused_moe_ep2_launch_rejects_expert_count_outside_source_scope(
        hardware, hardware_id, message):
    attributes = {
        "expert_parallel_size": 2,
        "launch_policy": "analytical_launch",
    }
    if hardware_id == "ascend_910b":
        attributes["local_tokens_per_expert"] = (1,) + (0,) * 7
    with pytest.raises(NotImplementedError, match=message):
        select_launch_config(
            hardware_id,
            "FUSED_MOE",
            {
                "tokens": 1,
                "experts": 8,
                "topk": 1,
                "hidden": 64,
                "intermediate": 64,
            },
            "bf16",
            "w1",
            attributes,
            hardware=hardware[hardware_id],
        )


@pytest.mark.parametrize("projection", ("w1", "w2"))
def test_metax_c500_fused_moe_rejects_multi_chunk_default_launch(
        hardware, projection):
    with pytest.raises(NotImplementedError, match="multi-chunk"):
        select_launch_config(
            "metax_c500", "FUSED_MOE", _c500_qwen3_fused_moe_shape(16385),
            "bf16", projection,
            {"expert_parallel_size": 2, "launch_policy": "analytical_launch"},
            hardware=hardware["metax_c500"],
        )


@pytest.mark.parametrize(
    ("variant", "shape", "expected_grid", "expected_block"),
    (
        (
            "moe_topk",
            {"tokens": 129, "experts": 2, "topk": 1,
             "hidden": 64, "intermediate": 16},
            2,
            128,
        ),
        (
            "moe_topk",
            {"tokens": 9, "experts": 128, "topk": 2,
             "hidden": 64, "intermediate": 16},
            3,
            128,
        ),
        (
            "moe_align_small_batch",
            {"tokens": 512, "experts": 2, "topk": 1,
             "hidden": 64, "intermediate": 16},
            1,
            32,
        ),
        (
            "moe_align",
            {"tokens": 1024, "experts": 2, "topk": 1,
             "hidden": 64, "intermediate": 16},
            1,
            1024,
        ),
        (
            "moe_count_and_sort",
            {"tokens": 20_000_000, "experts": 2, "topk": 1,
             "hidden": 64, "intermediate": 16},
            65_535,
            256,
        ),
        (
            "moe_sum",
            {"tokens": 7, "experts": 128, "topk": 2,
             "hidden": 4096, "intermediate": 14336},
            7,
            1024,
        ),
    ),
)
def test_nvidia_a800_moe_aux_launch_matches_vllm_0110_source(
        hardware, variant, shape, expected_grid, expected_block):
    launch = select_launch_config(
        "nvidia_a800",
        "VECTOR",
        shape,
        "bf16",
        variant,
        {},
        hardware=hardware["nvidia_a800"],
    )

    assert launch.grid == (expected_grid, 1, 1)
    if variant == "moe_topk":
        assert launch.block == (32, 4, 1)
        assert launch.block_size == expected_block
    else:
        assert launch.block == (expected_block, 1, 1)


@pytest.mark.parametrize(("tokens", "expected_group_size"), ((1, 1), (2, 8)))
def test_nvidia_a800_default_fused_moe_group_size_uses_local_expert_count(
        hardware, tokens, expected_group_size):
    shape = {
        "tokens": tokens,
        "experts": 2,
        "topk": 1,
        "hidden": 64,
        "intermediate": 16,
    }
    launch = select_launch_config(
        "nvidia_a800",
        "FUSED_MOE",
        shape,
        "bf16",
        "w1",
        {"expert_parallel_size": 2},
        hardware=hardware["nvidia_a800"],
    )

    assert launch.parameters["group_size_m"] == expected_group_size


@pytest.mark.parametrize(
    "variant,shape,expected_block,expected_parameters",
    [
        ("rmsnorm", {"rows": 64, "hidden": 4096},
         1024, {
             "input_reread_count": 1.0,
             "lowered_sfu_operations": 64.0,
             "norm_reduction_tree_flops": 65472.0,
             "norm_epsilon_add_flops": 64.0,
             "source_bf16_conversion_operations": 1572864.0,
             "source_bound_l2_read_transactions_per_warp": 1.0,
             "source_bound_l2_read_warps_per_cta": 16.0,
             "source_bound_block_reduction": 1.0,
             "source_bound_gpu_norm_cta_pipeline": 1.0,
             "source_rmsnorm_kernel_vllm_0_8_5_maca_3_7_2": 1.0,
         }),
        ("add_rmsnorm", {"rows": 64, "hidden": 4096},
         1024, {
             "input_reread_count": 1.0,
             "lowered_sfu_operations": 64.0,
             "norm_reduction_tree_flops": 65472.0,
             "norm_epsilon_add_flops": 64.0,
             "source_bf16_conversion_operations": 786432.0,
             "source_bound_l2_read_transactions_per_warp": 16.0,
             "source_bound_l2_read_warps_per_cta": 8.0,
             "source_bound_block_reduction": 1.0,
             "source_bound_gpu_norm_cta_pipeline": 1.0,
         }),
        ("add_rmsnorm", {"rows": 256, "hidden": 4096},
         256, {
             "input_reread_count": 1.0,
             "lowered_sfu_operations": 256.0,
             "norm_reduction_tree_flops": 65280.0,
             "norm_epsilon_add_flops": 256.0,
             "source_bf16_conversion_operations": 3145728.0,
             "source_bound_l2_read_transactions_per_warp": 16.0,
             "source_bound_l2_read_warps_per_cta": 4.0,
             "source_bound_block_reduction": 1.0,
             "source_bound_gpu_norm_cta_pipeline": 1.0,
         }),
        ("activation", {
            "tok": 16, "intermediate": 6144,
        }, 512, {
            "warp_instructions_lower_bound": 50560.0,
            "source_bound_warp_slot2_instructions": 47616.0,
            "source_vector_elements_per_thread": 8.0,
            "source_bf16_conversion_operations": 294912.0,
        }),
    ],
)
def test_metax_c500_vector_launch_keeps_only_source_closed_facts(
        hardware, variant, shape, expected_block, expected_parameters):
    launch = select_launch_config(
        "metax_c500", "VECTOR", shape, "bf16", variant, {},
        hardware=hardware["metax_c500"])

    if variant in ("rmsnorm", "add_rmsnorm"):
        expected_grid = (shape["rows"], 1, 1)
    elif variant == "activation":
        expected_grid = (
            (shape["intermediate"] + 4095) // 4096,
            shape["tok"], 1)
    else:
        raise AssertionError("unexpected C500 Vector test variant")
    assert launch.grid == expected_grid
    assert launch.block == (expected_block, 1, 1)
    assert launch.parameters == expected_parameters


def test_metax_c500_irregular_norm_width_keeps_source_cub_schedule(hardware):
    launch = select_launch_config(
        "metax_c500", "VECTOR", {"rows": 1, "hidden": 4097},
        "bf16", "rmsnorm", {}, hardware=hardware["metax_c500"])

    assert launch.parameters == {
        "input_reread_count": 1.0,
        "lowered_sfu_operations": 1.0,
        "norm_reduction_tree_flops": 1023.0,
        "norm_epsilon_add_flops": 1.0,
        "source_bf16_conversion_operations": 24582.0,
        "source_bound_l2_read_transactions_per_warp": 1.0,
        "source_bound_l2_read_warps_per_cta": 16.0,
        "source_rmsnorm_kernel_vllm_0_8_5_maca_3_7_2": 1.0,
    }


def test_metax_c500_h128_rmsnorm_binds_legacy_block128_program(hardware):
    launch = select_launch_config(
        "metax_c500", "VECTOR", {"rows": 188064, "hidden": 128},
        "bf16", "rmsnorm", {}, hardware=hardware["metax_c500"])

    assert launch.grid == (188064, 1, 1)
    assert launch.block == (128, 1, 1)
    assert launch.parameters[
        "source_rmsnorm_kernel_vllm_0_8_5_maca_3_7_2"] == 1.0
    assert launch.parameters["source_bound_block_reduction"] == 1.0
    assert launch.parameters["source_bound_gpu_norm_cta_pipeline"] == 1.0


@pytest.mark.parametrize(
    "rows_per_group,expected_grid",
    ((32, (96, 1, 1)), (8, (24, 1, 1))),
)
def test_metax_legacy_grouped_qk_copy_uses_source_geometry(
        hardware, rows_per_group, expected_grid):
    outer_groups = 12
    hidden = 128
    elements = outer_groups * rows_per_group * hidden
    launch = select_launch_config(
        "metax_c500",
        "VECTOR",
        {"element_count": elements},
        "bf16",
        "elementwise",
        {
            "operation": "copy",
            "reads": 1,
            "writes": 1,
            "input_materialization": "grouped_strided_to_contiguous",
            "input_outer_groups": outer_groups,
            "input_rows_per_group": rows_per_group,
            "input_group_stride_elements": 48 * hidden,
            "input_row_stride_elements": hidden,
        },
        hardware=hardware["metax_c500"],
    )

    assert launch.algorithm == (
        "pytorch_2_4_metax_3_7_2_bf16_grouped_strided_copy")
    assert launch.grid == expected_grid
    assert launch.block == (64, 1, 1)
    assert launch.parameters == {
        "source_metax_bf16_grouped_strided_copy": 1.0,
        "source_bound_l2_read_transactions_per_warp": 8.0,
        "source_bound_l2_read_warps_per_cta": 1.0,
    }


@pytest.mark.parametrize(
    "outer_groups,rows_per_group,expected_grid",
    ((16, 32, (128, 1, 1)), (16, 8, (32, 1, 1))),
)
def test_nvidia_a800_norm_input_copy_binds_tensor_iterator_task_floor(
        hardware, outer_groups, rows_per_group, expected_grid):
    hidden = 128
    elements = outer_groups * rows_per_group * hidden
    launch = select_launch_config(
        "nvidia_a800",
        "VECTOR",
        {"element_count": elements},
        "bf16",
        "elementwise",
        {
            "operation": "copy",
            "reads": 1,
            "writes": 1,
            "input_materialization": "grouped_strided_to_contiguous",
            "input_outer_groups": outer_groups,
            "input_rows_per_group": rows_per_group,
            "input_group_stride_elements": 48 * hidden,
            "input_row_stride_elements": hidden,
        },
        hardware=hardware["nvidia_a800"],
    )

    assert launch.algorithm == (
        "observed_norm_input_materialization_copy:elementwise")
    assert launch.grid == expected_grid
    assert launch.block == (128, 1, 1)
    assert launch.parameters == {
        "source_bound_tensor_iterator_copy_task": 4.0,
    }


def test_nvidia_a800_norm_input_copy_rejects_unknown_materialization(
        hardware):
    with pytest.raises(NotImplementedError, match="grouped-strided"):
        select_launch_config(
            "nvidia_a800", "VECTOR", {"element_count": 4096}, "bf16",
            "elementwise",
            {
                "operation": "copy",
                "reads": 1,
                "writes": 1,
                "input_materialization": "dense_contiguous",
            },
            hardware=hardware["nvidia_a800"])


def test_nvidia_a800_plain_copy_keeps_generic_geometry(hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"element_count": 65536}, "bf16",
        "elementwise", {"operation": "copy", "reads": 1, "writes": 1},
        hardware=hardware["nvidia_a800"])

    assert launch.algorithm == "elementwise"
    assert launch.grid == (128, 1, 1)
    assert launch.block == (128, 1, 1)
    assert launch.parameters == {}


def test_nvidia_a800_scalar_tail_rmsnorm_binds_memory_dependency_task(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"rows": 512, "hidden": 128}, "bf16",
        "rmsnorm", {}, hardware=hardware["nvidia_a800"])

    assert launch.parameters["source_bound_memory_dependency_task"] == 1.0
    assert "source_bound_block_reduction" not in launch.parameters
    assert "source_bound_gpu_norm_cta_pipeline" not in launch.parameters
    assert "source_bound_l2_read_transactions_per_warp" not in (
        launch.parameters)

    # Tail-free widths below the 1024-thread source program lack the typed
    # reduction/read-round services and stay unmarked.
    tail_free = select_launch_config(
        "nvidia_a800", "VECTOR", {"rows": 512, "hidden": 512}, "bf16",
        "rmsnorm", {}, hardware=hardware["nvidia_a800"])
    assert "source_bound_memory_dependency_task" not in tail_free.parameters


@pytest.mark.parametrize(
    "batch,expected_grid",
    ((12, (48, 1, 1)), (16, (64, 1, 1)), (40, (160, 1, 1))),
)
def test_nvidia_a800_flash_attention_output_fill_matches_pytorch_geometry(
        hardware, batch, expected_grid):
    from src.kernel.launch_config.select_launch_config import (
        select_flash_attn_output_fill_launches)
    selections = select_flash_attn_output_fill_launches(
        "nvidia_a800",
        {"batch": batch, "q_len": 1, "kv_len": 512, "head_dim": 128,
         "num_query_heads": 32, "num_kv_heads": 8},
        "bf16",
        {},
        hardware["nvidia_a800"],
    )

    (dtype, elements, launch), = selections
    assert dtype == "bf16"
    assert elements == batch * 32 * 128
    assert launch.algorithm == (
        "observed_flash_attention_output_fill:elementwise")
    assert launch.grid == expected_grid
    assert launch.block == (128, 1, 1)
    assert launch.execution_variant == "elementwise"


@pytest.mark.parametrize(
    "tokens,expected_grid",
    # Canonical qwen3-4b tp1_qps4 alignment: decode 19, mixed 52, prefill 729.
    ((19, (10, 1, 1)), (52, (26, 1, 1)), (729, (365, 1, 1))),
)
def test_metax_c500_flash_attention_output_fill_matches_observed_geometry(
        hardware, tokens, expected_grid):
    from src.kernel.launch_config.select_launch_config import (
        select_flash_attn_output_fill_launches)
    selections = select_flash_attn_output_fill_launches(
        "metax_c500",
        {"batch": tokens, "q_len": 1, "kv_len": 512, "head_dim": 128,
         "num_query_heads": 32, "num_kv_heads": 8},
        "bf16",
        {},
        hardware["metax_c500"],
    )

    (dtype, elements, launch), = selections
    assert dtype == "bf16"
    assert elements == tokens * 32 * 128
    assert launch.grid == expected_grid
    assert launch.block == (1024, 1, 1)
    assert launch.execution_variant == "elementwise"


def test_ascend_910b_flash_attention_output_fill_is_unsupported(hardware):
    from src.kernel.launch_config.select_launch_config import (
        select_flash_attn_output_fill_launches)
    shape = {"batch": 2, "q_len": 1, "kv_len": 512, "head_dim": 128,
             "num_query_heads": 32, "num_kv_heads": 8}
    assert select_flash_attn_output_fill_launches(
        "ascend_910b", shape, "bf16", {}, hardware["ascend_910b"]) == ()



def test_metax_c500_swiglu_rejects_unproven_small_or_unaligned_branch(
        hardware):
    with pytest.raises(ValueError, match="intermediate >= 4096"):
        select_launch_config(
            "metax_c500", "VECTOR", {"tok": 1, "intermediate": 4100},
            "bf16", "activation", {}, hardware=hardware["metax_c500"])


def test_metax_c500_kv_cache_uses_flash_layout_source_launch(hardware):
    shape = {"seq_len": 1, "num_kv_heads": 3, "head_dim": 127}
    launch = select_launch_config(
        "metax_c500", "KV_CACHE", shape, "bf16", "paged_kv_cache",
        PAGED_KV_CACHE_ATTRIBUTES,
        hardware=hardware["metax_c500"])

    assert launch.grid == (1, 1, 1)
    assert launch.block == (381, 1, 1)
    assert launch.parameters == {}


def test_metax_c500_kv_cache_publishes_scalar_two_tensor_source_path(hardware):
    """Read producers are the block's threads, with key and value in flight.

    The pinned vLLM-MetaX 0.8.5 reshape_and_cache_flash body is one
    block-strided scalar loop, so all 512 threads (8 warps of 64) read, each
    warp's 64 contiguous BF16 loads fill exactly one 128-byte L2 transaction,
    and one iteration issues that once for the key and once for the value.
    """

    shape = {"seq_len": 16, "num_kv_heads": 4, "head_dim": 128}
    launch = select_launch_config(
        "metax_c500", "KV_CACHE", shape, "bf16", "paged_kv_cache",
        PAGED_KV_CACHE_ATTRIBUTES,
        hardware=hardware["metax_c500"])

    assert launch.grid == (16, 1, 1)
    assert launch.block == (512, 1, 1)
    assert launch.parameters == {
        "source_vectorized_kv_copy": 1.0,
        "source_bound_l2_read_transactions_per_warp": 2.0,
        "source_bound_l2_read_warps_per_cta": 8.0,
    }


@pytest.mark.parametrize(
    "field,value,message",
    (
        ("cache_layout", "contiguous", "paged cache layout"),
        ("kv_cache_dtype", "fp8", "kv_cache_dtype='auto'"),
        ("cache_storage_matches_input", False, "match the input dtype"),
        ("slot_mapping_all_valid", False, "non-negative"),
    ),
)
def test_metax_c500_kv_cache_rejects_unmodeled_runtime_contract(
        hardware, field, value, message):
    attributes = dict(PAGED_KV_CACHE_ATTRIBUTES)
    attributes[field] = value

    with pytest.raises(NotImplementedError, match=message):
        select_launch_config(
            "metax_c500", "KV_CACHE",
            {"seq_len": 1, "num_kv_heads": 4, "head_dim": 128},
            "bf16", "paged_kv_cache", attributes,
            hardware=hardware["metax_c500"])


@pytest.mark.parametrize(
    "variant,shape,expected_grid,expected_block",
    [
        ("rmsnorm", {"rows": 3, "hidden": 128}, 3, 128),
        ("add_rmsnorm", {"rows": 3, "hidden": 128}, 3, 128),
        ("add_rmsnorm", {"rows": 256, "hidden": 128}, 256, 128),
        ("rotary", {
            "seq_len": 4, "num_query_heads": 2,
            "num_kv_heads": 1, "head_dim": 128,
        }, 4, 128),
        ("activation", {"tok": 5, "intermediate": 64}, 5, 64),
        ("paged_kv_cache", {
            "seq_len": 6, "num_kv_heads": 1, "head_dim": 64,
        }, 6, 64),
    ],
)
def test_nvidia_a800_small_vector_launch_uses_source_block_limit(
        hardware, variant, shape, expected_grid, expected_block):
    if variant == "paged_kv_cache":
        attributes = PAGED_KV_CACHE_ATTRIBUTES
        kind = "KV_CACHE"
    elif variant in ("rmsnorm", "add_rmsnorm", "rotary", "activation"):
        attributes = {}
        kind = "VECTOR"
    else:
        raise AssertionError(f"unhandled test variant: {variant!r}")
    launch = select_launch_config(
        "nvidia_a800", kind, shape, "bf16", variant, attributes,
        hardware=hardware["nvidia_a800"])

    assert launch.grid == (expected_grid, 1, 1)
    assert launch.block == (expected_block, 1, 1)
    # The launch package publishes source semantics and geometry only.
    if variant in ("rmsnorm", "add_rmsnorm"):
        expected_parameters = {
            "input_reread_count": 1.0,
            "norm_reduction_tree_flops": float(
                expected_grid * (expected_block - 1)),
            "norm_epsilon_add_flops": float(expected_grid),
        }
        if variant == "rmsnorm" and (
                expected_block == 1024 or shape["hidden"] % 256 != 0):
            expected_parameters[
                "source_bound_memory_dependency_task"] = 1.0
        assert launch.parameters == expected_parameters
    elif variant == "rotary":
        assert launch.parameters == {
            "rotary_cache_path_bytes": 3072.0,
            "rotary_position_path_bytes": 512.0,
        }
    elif variant == "paged_kv_cache":
        assert launch.parameters == {
            "kv_cache_slot_l1_bytes": 384.0,
        }
    else:
        assert launch.parameters == {}


def test_nvidia_a800_irregular_kv_row_counts_one_runtime_rcp_per_thread(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "KV_CACHE",
        {"seq_len": 2, "num_kv_heads": 1, "head_dim": 7},
        "bf16", "paged_kv_cache", PAGED_KV_CACHE_ATTRIBUTES,
        hardware=hardware["nvidia_a800"])

    assert launch.grid == (2, 1, 1)
    assert launch.block == (7, 1, 1)
    assert launch.parameters == {
        "kv_cache_slot_l1_bytes": 64.0,
    }


def test_nvidia_a800_full_block_kv_tail_counts_one_runtime_rcp_per_thread(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "KV_CACHE",
        {"seq_len": 2, "num_kv_heads": 1, "head_dim": 513},
        "bf16", "paged_kv_cache", PAGED_KV_CACHE_ATTRIBUTES,
        hardware=hardware["nvidia_a800"])

    assert launch.grid == (2, 1, 1)
    assert launch.block == (512, 1, 1)
    assert launch.parameters == {
        "kv_cache_slot_l1_bytes": 1024.0,
    }


@pytest.mark.parametrize(
    "field,value",
    (
        ("cache_head_layout", "hnd"),
        ("vector_copy_base_alignment_bytes", 8),
    ),
)
def test_nvidia_a800_kv_cache_does_not_borrow_aligned_nhd_sass(
        hardware, field, value):
    attributes = dict(PAGED_KV_CACHE_ATTRIBUTES)
    attributes[field] = value
    launch = select_launch_config(
        "nvidia_a800", "KV_CACHE",
        {"seq_len": 1, "num_kv_heads": 8, "head_dim": 128},
        "bf16", "paged_kv_cache", attributes,
        hardware=hardware["nvidia_a800"])

    assert launch.parameters == {
        "kv_cache_slot_l1_bytes": 512.0,
    }


def test_nvidia_a800_kv_cache_does_not_extrapolate_512_thread_residency(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "KV_CACHE",
        {"seq_len": 1, "num_kv_heads": 2, "head_dim": 128},
        "bf16", "paged_kv_cache", PAGED_KV_CACHE_ATTRIBUTES,
        hardware=hardware["nvidia_a800"])

    assert launch.block == (256, 1, 1)
    assert launch.parameters == {
        "kv_cache_slot_l1_bytes": 256.0,
        "source_bound_l2_read_transactions_per_warp": 16.0,
        "source_bound_l2_read_warps_per_cta": 1.0,
        "source_bound_memory_dependency_task": 1.0,
    }


def test_kv_cache_launch_accepts_legacy_kind_only_at_parse_boundary(hardware):
    shape = {"seq_len": 1, "num_kv_heads": 8, "head_dim": 128}
    launch = select_launch_config(
        "nvidia_a800", "KV_CACHE_UPDATE", shape, "bf16",
        "paged_kv_cache", PAGED_KV_CACHE_ATTRIBUTES,
        hardware=hardware["nvidia_a800"])

    assert launch.execution_variant == "paged_kv_cache"


@pytest.mark.parametrize(
    "kind,variant",
    (("VECTOR", "kv_cache"), ("VECTOR", "kvcache"),
     ("KV_CACHE", "kv_cache"), ("KV_CACHE", "kvcache")),
)
def test_kv_cache_launch_rejects_legacy_public_kind_or_variant(
        hardware, kind, variant):
    with pytest.raises(ValueError, match="KV cache launch selection requires"):
        select_launch_config(
            "nvidia_a800", kind,
            {"seq_len": 1, "num_kv_heads": 8, "head_dim": 128},
            "bf16", variant, PAGED_KV_CACHE_ATTRIBUTES,
            hardware=hardware["nvidia_a800"])


def test_nvidia_a800_fp16_kv_cache_does_not_borrow_bf16_rcp_lowering(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "KV_CACHE",
        {"seq_len": 2, "num_kv_heads": 1, "head_dim": 128},
        "fp16", "paged_kv_cache", PAGED_KV_CACHE_ATTRIBUTES,
        hardware=hardware["nvidia_a800"])

    assert launch.parameters == {"kv_cache_slot_l1_bytes": 256.0}


def test_nvidia_a800_rope_rejects_non_gqa_head_ratio(hardware):
    with pytest.raises(
            ValueError,
            match="num_query_heads to be divisible by num_kv_heads"):
        select_launch_config(
            "nvidia_a800", "VECTOR",
            {"seq_len": 2, "num_query_heads": 3,
             "num_kv_heads": 2, "head_dim": 128},
            "bf16", "rotary", {}, hardware=hardware["nvidia_a800"])


@pytest.mark.parametrize("variant", ("rmsnorm", "add_rmsnorm"))
def test_nvidia_a800_fp16_norm_does_not_borrow_bf16_sass_lowering(
        hardware, variant):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"rows": 3, "hidden": 128}, "fp16",
        variant, {}, hardware=hardware["nvidia_a800"])

    assert launch.parameters == {
        "input_reread_count": 1.0,
        "norm_reduction_tree_flops": 381.0,
        "norm_epsilon_add_flops": 3.0,
    }


@pytest.mark.parametrize("variant", ("rmsnorm", "add_rmsnorm"))
def test_nvidia_a800_norm_publishes_source_cub_reduction(
        hardware, variant):
    shape = {"rows": 64, "hidden": 4096}
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", shape, "bf16", variant, {},
        hardware=hardware["nvidia_a800"])

    assert launch.parameters["norm_reduction_tree_flops"] == 64 * 1023
    assert launch.parameters["norm_epsilon_add_flops"] == 64
    assert "warp_instructions" not in launch.parameters
    assert "warp_instructions_lower_bound" not in launch.parameters
    assert launch.parameters["source_bound_block_reduction"] == 1.0
    if variant == "rmsnorm":
        assert launch.parameters[
            "source_bound_memory_dependency_task"] == 1.0


@pytest.mark.parametrize("variant", ("rmsnorm", "add_rmsnorm"))
def test_nvidia_a800_irregular_norm_cta_does_not_borrow_sass_issue(
        hardware, variant):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"rows": 3, "hidden": 512}, "bf16",
        variant, {}, hardware=hardware["nvidia_a800"])

    assert "warp_instructions" not in launch.parameters
    assert "warp_instructions_lower_bound" not in launch.parameters


def test_nvidia_a800_silu_publishes_analytical_task_contract(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"tok": 1, "intermediate": 12288},
        "bf16", "silu", {}, hardware=hardware["nvidia_a800"])

    assert launch.parameters == {
        "warp_instructions_lower_bound": 18528.0,
        "source_bound_memory_dependency_task": 1.0,
        "source_bound_l2_read_transactions_per_warp": 4.0,
    }


def test_nvidia_a800_full_sm_silu_publishes_segmented_pipeline(hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"tok": 256, "intermediate": 53248},
        "bf16", "silu", {}, hardware=hardware["nvidia_a800"])

    assert launch.grid == (256, 1, 1)
    assert launch.block == (1024, 1, 1)
    assert launch.parameters == {
        "warp_instructions_lower_bound": 20144128.0,
        "source_bound_memory_dependency_task": 1.0,
        "source_bound_silu_nonrecurrent_cta_pipeline": 1.0,
        "source_bound_l2_read_transactions_per_warp": 4.0,
    }

    queued = select_launch_config(
        "nvidia_a800", "VECTOR", {"tok": 1024, "intermediate": 53248},
        "bf16", "silu", {}, hardware=hardware["nvidia_a800"])
    assert queued.parameters[
        "source_bound_silu_nonrecurrent_cta_pipeline"] == 1.0


def test_nvidia_a800_rmsnorm_producer_tail_keeps_read_source_unknown(
        hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"rows": 1, "hidden": 2080}, "bf16",
        "rmsnorm", {},
        hardware=hardware["nvidia_a800"])

    assert launch.parameters["source_bound_memory_dependency_task"] == 1.0
    assert (
        "source_bound_l2_read_transactions_per_warp"
        not in launch.parameters)
    assert "source_bound_l2_read_warps_per_cta" not in launch.parameters
    assert "source_bound_block_reduction" not in launch.parameters
    assert "source_bound_gpu_norm_cta_pipeline" not in launch.parameters


def test_nvidia_a800_silu_tail_keeps_full_warp_read_source(hardware):
    launch = select_launch_config(
        "nvidia_a800", "VECTOR", {"tok": 1, "intermediate": 2080},
        "bf16", "silu", {}, hardware=hardware["nvidia_a800"])

    assert launch.parameters[
        "source_bound_memory_dependency_task"] == 1.0
    assert launch.parameters[
        "source_bound_l2_read_transactions_per_warp"] == 4.0


@pytest.mark.parametrize(
    "variant,shape,expected_grid,expected_block,expected_parameters",
    [
        ("rmsnorm", {"rows": 64, "hidden": 4096}, (64, 1, 1), 1024,
         {
             "input_reread_count": 1.0,
             "lowered_sfu_operations": 64.0,
             "norm_reduction_tree_flops": 65472.0,
             "norm_epsilon_add_flops": 64.0,
             "source_bf16_conversion_operations": 1572864.0,
             "source_bound_l2_read_transactions_per_warp": 1.0,
             "source_bound_l2_read_warps_per_cta": 16.0,
             "source_bound_block_reduction": 1.0,
             "source_bound_gpu_norm_cta_pipeline": 1.0,
             "source_rmsnorm_kernel_vllm_0_8_5_maca_3_7_2": 1.0,
         }),
        ("add_rmsnorm", {"rows": 255, "hidden": 4096},
         (255, 1, 1), 1024,
         {
             "input_reread_count": 1.0,
             "lowered_sfu_operations": 255.0,
             "norm_reduction_tree_flops": 260865.0,
             "norm_epsilon_add_flops": 255.0,
             "source_bf16_conversion_operations": 3133440.0,
             "source_bound_l2_read_transactions_per_warp": 16.0,
             "source_bound_l2_read_warps_per_cta": 8.0,
             "source_bound_block_reduction": 1.0,
             "source_bound_gpu_norm_cta_pipeline": 1.0,
         }),
        ("add_rmsnorm", {"rows": 256, "hidden": 4096},
         (256, 1, 1), 256,
         {
             "input_reread_count": 1.0,
             "lowered_sfu_operations": 256.0,
             "norm_reduction_tree_flops": 65280.0,
             "norm_epsilon_add_flops": 256.0,
             "source_bf16_conversion_operations": 3145728.0,
             "source_bound_l2_read_transactions_per_warp": 16.0,
             "source_bound_l2_read_warps_per_cta": 4.0,
             "source_bound_block_reduction": 1.0,
             "source_bound_gpu_norm_cta_pipeline": 1.0,
         }),
        ("activation", {
            "tok": 16, "intermediate": 9728,
            "element_count": 155648,
        }, (3, 16, 1), 512, {
            "warp_instructions_lower_bound": 79840.0,
            "source_bound_warp_slot2_instructions": 75328.0,
            "source_vector_elements_per_thread": 8.0,
            "source_bf16_conversion_operations": 466944.0,
        }),
        ("rope", {
            "seq_len": 4, "num_query_heads": 32,
            "num_kv_heads": 8, "head_dim": 128,
            "rotary_dim": 128,
        }, (4, 1, 1), 512, {
            "rotary_cache_path_bytes": 40960.0,
            "source_metax_rope_vllm_0_8_5_abi10_program": 1.0,
            "source_qk_rope_loop": 1.0,
            "source_bf16_conversion_operations": 61440.0,
            "source_bound_l2_read_transactions_per_warp": 4.0,
            "source_bound_l2_read_warps_per_cta": 8.0,
        }),
        ("paged_kv_cache", {
            "seq_len": 6, "num_kv_heads": 8, "head_dim": 128,
        }, (6, 1, 1), 512, {
            "source_vectorized_kv_copy": 1.0,
            "source_bound_l2_read_transactions_per_warp": 2.0,
            "source_bound_l2_read_warps_per_cta": 8.0,
        }),
        ("elementwise", {"element_count": 16777216},
         (16384, 1, 1), 256, {}),
    ],
)
def test_metax_c500_vector_launch_matches_serving_runtime(
        hardware, variant, shape, expected_grid, expected_block,
        expected_parameters):
    if variant == "elementwise":
        attributes = {"operation": "add"}
        kind = "VECTOR"
    elif variant == "rope":
        attributes = {"is_neox": True, "key_present": True}
        kind = "VECTOR"
    elif variant == "paged_kv_cache":
        attributes = PAGED_KV_CACHE_ATTRIBUTES
        kind = "KV_CACHE"
    elif variant in ("rmsnorm", "add_rmsnorm", "activation"):
        attributes = {}
        kind = "VECTOR"
    else:
        raise AssertionError(f"unexpected Vector variant: {variant!r}")
    launch = select_launch_config(
        "metax_c500", kind, shape, "bf16", variant, attributes,
        hardware=hardware["metax_c500"])

    assert launch.grid == expected_grid
    assert launch.block == (expected_block, 1, 1)
    assert launch.parameters == expected_parameters


def test_metax_c500_full_dimension_rotary_uses_source_launch(hardware):
    launch = select_launch_config(
        "metax_c500", "VECTOR",
        {
            "seq_len": 4,
            "num_query_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
        },
        "bf16", "rotary", {}, hardware=hardware["metax_c500"])

    assert launch.grid == (4, 1, 1)
    assert launch.block == (512, 1, 1)
    assert launch.parameters == {
        "rotary_cache_path_bytes": 40960.0,
        "source_metax_rope_vllm_0_9_1_abi11_program": 1.0,
        "source_qk_rope_loop": 1.0,
        "source_bf16_conversion_operations": 61440.0,
        # Four independent per-pair loads, each one 128-byte C500 transaction
        # (64 lanes x BF16), issued by the eight wavefronts of the 512-thread
        # CTA that the Q loop keeps active.
        "source_bound_l2_read_transactions_per_warp": 4.0,
        "source_bound_l2_read_warps_per_cta": 8.0,
    }


@pytest.mark.parametrize(
    "attributes,message",
    [
        ({"is_neox": False, "key_present": True}, "NeoX template"),
        ({"is_neox": True, "key_present": False}, "Q\\+K path"),
    ],
)
def test_metax_c500_typed_rope_rejects_unbound_template_or_missing_key(
        hardware, attributes, message):
    with pytest.raises(NotImplementedError, match=message):
        select_launch_config(
            "metax_c500", "VECTOR",
            {
                "seq_len": 1,
                "num_query_heads": 32,
                "num_kv_heads": 8,
                "head_dim": 128,
                "rotary_dim": 128,
            },
            "bf16", "rope", attributes,
            hardware=hardware["metax_c500"])


def test_metax_c500_elementwise_keeps_only_static_geometry(hardware):
    launch = select_launch_config(
        "metax_c500", "VECTOR", {"element_count": 2560}, "bf16",
        "elementwise", {"operation": "add"},
        hardware=hardware["metax_c500"])

    assert launch.grid == (3, 1, 1)
    assert launch.block == (256, 1, 1)
    assert launch.parameters == {}


def test_metax_c500_elementwise_rejects_unclosed_remainder_schedule(
        hardware):
    launch = select_launch_config(
        "metax_c500", "VECTOR", {"element_count": 2561}, "bf16",
        "elementwise", {"operation": "add"},
        hardware=hardware["metax_c500"])

    assert launch.parameters == {}


@pytest.mark.parametrize("operation", ("copy", "fill", "read"))
def test_metax_c500_elementwise_without_source_bound_launch_fails_loudly(
        hardware, operation):
    with pytest.raises(NotImplementedError, match="source-bound only"):
        select_launch_config(
            "metax_c500", "VECTOR", {"element_count": 2560}, "bf16",
            "elementwise", {"operation": operation},
            hardware=hardware["metax_c500"])


@pytest.mark.parametrize(
    ("fill_role", "expected_grid", "expected_block", "expected_algorithm"),
    (
        (
            "sorted_token_ids", (18, 1, 1), (1024, 1, 1),
            "pytorch_2_4_metax_int32_fill_nullary_opt",
        ),
        (
            "expert_ids", (2, 1, 1), (256, 1, 1),
            "pytorch_2_4_metax_int32_fill",
        ),
        (
            "num_tokens_post_padded", (1, 1, 1), (256, 1, 1),
            "pytorch_2_4_metax_int32_fill",
        ),
    ),
)
def test_metax_c500_moe_fill_launch_matches_fixed_torch_source(
        hardware, fill_role, expected_grid, expected_block,
        expected_algorithm):
    launch = select_launch_config(
        "metax_c500", "VECTOR",
        {
            "tokens": 4096,
            "experts": 128,
            "topk": 16,
            "hidden": 2048,
            "intermediate": 768,
        },
        "bf16", "moe_fill", {"fill_role": fill_role},
        hardware=hardware["metax_c500"])

    assert launch.grid == expected_grid
    assert launch.block == expected_block
    assert launch.algorithm == expected_algorithm
    assert launch.execution_variant == "moe_fill"
    assert launch.parameters == {}


def test_metax_c500_moe_fill_nullary_vector_width_follows_source(hardware):
    launch = select_launch_config(
        "metax_c500", "VECTOR",
        {
            "tokens": 65348,
            "experts": 3,
            "topk": 1,
            "hidden": 2048,
            "intermediate": 768,
        },
        "bf16", "moe_fill", {"fill_role": "sorted_token_ids"},
        hardware=hardware["metax_c500"])

    assert launch.grid == (65, 1, 1)
    assert launch.block == (1024, 1, 1)


@pytest.mark.parametrize(
    ("variant", "expected_algorithm"),
    (
        ("moe_silu", "pytorch_2_4_metax_bf16_silu_elementwise"),
        ("moe_mul", "pytorch_2_4_metax_bf16_mul_elementwise"),
    ),
)
def test_metax_c500_replacement_moe_elementwise_launch_matches_source(
        hardware, variant, expected_algorithm):
    launch = select_launch_config(
        "metax_c500", "VECTOR",
        {
            "tokens": 2,
            "experts": 128,
            "topk": 4,
            "hidden": 2880,
            "intermediate": 2880,
        },
        "bf16", variant, {
            "launch_policy": "injected_launch",
            "work_fraction": 1.0,
        }, hardware=hardware["metax_c500"])

    assert launch.grid == (23, 1, 1)
    assert launch.block == (128, 1, 1)
    assert launch.algorithm == expected_algorithm
    assert launch.execution_variant == variant
    assert launch.parameters == {}


@pytest.mark.parametrize("variant", ("moe_silu", "moe_mul"))
def test_metax_c500_replacement_moe_elementwise_rejects_unvectorized_width(
        hardware, variant):
    with pytest.raises(NotImplementedError, match="divisible by 8"):
        select_launch_config(
            "metax_c500", "VECTOR",
            {
                "tokens": 2,
                "experts": 128,
                "topk": 4,
                "hidden": 2880,
                "intermediate": 2879,
            },
            "bf16", variant, {
                "launch_policy": "injected_launch",
                "work_fraction": 1.0,
            }, hardware=hardware["metax_c500"])


@pytest.mark.parametrize("variant", ("moe_silu", "moe_mul"))
def test_metax_c500_replacement_moe_elementwise_split_matches_source(
        hardware, variant):
    launch = select_launch_config(
        "metax_c500", "VECTOR",
        {
            "tokens": 32768,
            "experts": 8,
            "topk": 2,
            "hidden": 4096,
            "intermediate": 14336,
        },
        "bf16", variant, {
            "launch_policy": "injected_launch",
            "work_fraction": 0.5,
        }, hardware=hardware["metax_c500"])

    assert launch.grid == (458752, 1, 1)
    assert launch.block == (128, 1, 1)


@pytest.mark.parametrize(
    ("tokens", "topk", "hidden", "work_fraction", "expected_grid"),
    (
        (1, 2, 4096, 1.0, (4, 1, 1)),
        (1, 8, 2048, 1.0, (8, 1, 1)),
        (2, 2, 4096, 1.0, (4, 1, 1)),
        (16384, 8, 2048, 1.0, (16384, 1, 1)),
        (32768, 8, 7168, 0.5, (57344, 1, 1)),
    ),
)
def test_metax_c500_moe_sum_launch_matches_tensor_iterator_source(
        hardware, tokens, topk, hidden, work_fraction, expected_grid):
    launch = select_launch_config(
        "metax_c500", "VECTOR",
        {
            "tokens": tokens,
            "experts": 128,
            "topk": topk,
            "hidden": hidden,
            "intermediate": 768,
        },
        "bf16", "moe_sum", {"work_fraction": work_fraction},
        hardware=hardware["metax_c500"])

    assert launch.grid == expected_grid
    assert launch.block == (512, 1, 1)
    assert launch.algorithm == "pytorch_2_4_metax_bf16_sum_imcontinuous_reduce"
    assert launch.execution_variant == "moe_sum"
    assert launch.parameters == {}


def test_metax_c500_moe_sum_rejects_wrong_tensor_iterator_fraction(hardware):
    with pytest.raises(ValueError, match="expected 0.5"):
        select_launch_config(
            "metax_c500", "VECTOR",
            {
                "tokens": 32768,
                "experts": 128,
                "topk": 8,
                "hidden": 7168,
                "intermediate": 768,
            },
            "bf16", "moe_sum", {"work_fraction": 1.0},
            hardware=hardware["metax_c500"])


def test_metax_c500_known_gemm_entry(hardware):
    launch = select_launch_config(
        "metax_c500", "GEMM", {"b": 1, "m": 128, "n": 6144, "k": 4096},
        "bf16", "gemm", {"has_bias": False, "position": "generic",
                           "launch_policy": "injected_launch"},
        hardware=hardware["metax_c500"])
    assert launch.algorithm == "mcblas_injected_gemm:gemm"
    assert launch.grid == (1, 48, 2)
    assert launch.block == (256, 1, 1)
    assert dict(launch.tiling.tile_shape) == {"m": 128, "n": 128, "k": 128}
    assert launch.tiling.pipeline_depth is None
    assert launch.tiling.split_k == 2


def test_metax_c500_opaque_stage_label_is_not_numeric_pipeline_evidence(
        hardware):
    launch = select_launch_config(
        "metax_c500", "GEMM",
        {"b": 1, "m": 100, "n": 6144, "k": 4096},
        "bf16", "gemm", {
            "has_bias": False,
            "position": "generic",
            "launch_policy": "injected_launch",
        }, hardware=hardware["metax_c500"])

    # The frozen LUT calls this runtime family ``4stage_*``, but no active
    # compiler/IR evidence maps that opaque identity to buffer-slot semantics.
    assert launch.tiling.pipeline_depth is None


def test_metax_c500_lm_head_parallelism(hardware):
    expected = {
        (128256, 4096, 1, 1): ((8016, 1, 1), (256, 1, 1)),
        (151936, 2560, 1, 1): ((9496, 1, 1), (256, 1, 1)),
        (151936, 4096, 1, 1): ((9496, 1, 1), (256, 1, 1)),
        (75968, 5120, 2, 1): ((4748, 1, 1), (256, 1, 1)),
        (75968, 2048, 2, 2): ((4748, 1, 1), (256, 1, 1)),
    }
    for (n, k, tp, ep), (grid, block) in expected.items():
        launch = select_launch_config(
            "metax_c500", "GEMM", {"b": 1, "m": 1, "n": n, "k": k},
            "bf16", "gemv", {
                "position": "lm_head",
                "tensor_parallel_size": tp,
                "expert_parallel_size": ep,
            }, hardware["metax_c500"])
        assert launch.grid == grid
        assert launch.algorithm == "mcblas_injected_gemm:gemm"
        assert launch.block == block
        assert launch.tiling is None


def test_metax_c500_multitoken_lm_head_uses_tensor_gemm(hardware):
    launch = select_launch_config(
        "metax_c500", "GEMM", {"b": 1, "m": 16, "n": 151936, "k": 2560},
        "bf16", "gemm", {
            "position": "lm_head",
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
            "launch_policy": "injected_launch",
        }, hardware["metax_c500"])
    assert launch.algorithm == "mcblas_injected_gemm:gemm"
    assert launch.grid == (1, 2374, 1)
    assert dict(launch.tiling.tile_shape) == {"m": 64, "n": 64, "k": 64}


@pytest.mark.parametrize("n", (128, 4096))
def test_metax_c500_gemv_outside_exact_domain_fails_loudly(hardware, n):
    with pytest.raises(KeyError, match="uncharacterized MetaX C500 GEMM"):
        select_launch_config(
            "metax_c500", "GEMM", {"b": 1, "m": 1, "n": n, "k": 4096},
            "bf16", "gemv", {"has_bias": False, "position": "generic"},
            hardware=hardware["metax_c500"])


def test_ascend_910b_launch_does_not_consume_traffic_derived_tiles(hardware):
    launch = select_launch_config(
        "ascend_910b", "GEMM", {"b": 1, "m": 128, "n": 6144, "k": 4096},
        "bf16", "gemm", {"has_bias": False, "position": "generic",
                           "launch_policy": "injected_launch"},
        hardware=hardware["ascend_910b"])
    assert launch.grid == (24, 1, 1)
    assert launch.parameters["mte2_tile_bytes"] == 0.0
    assert launch.parameters["weight_tile_rows"] == 0.0


def test_ascend_910b_v2_tiling_decoder_tracks_dynamic_field_order():
    """The source's ordered cache-tiling variables occupy payload words 6..37."""

    words = (
        64, 16, 64, 4, 1, 4,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 4, 1, 4, 1, 1, 1,
        1, 1, 4, 4, 1, 1, 1, 1, 1,
        0, 0, 1, 0, 0, 0, 0, 0,
    )
    payload = struct.pack("<38i", *words) + bytes(8)
    decoded = ascend_gemm_launch.decode_matmul_v2_tiling_payload(
        payload, {"k": 64, "m": 16, "n": 64}, 1,
        "MatMulV2_ND_ND_FP16_FP16_false_true_all_98304")
    program = decoded["source_program"]

    assert (
        program.aic_workers, program.single_core_m, program.single_core_n,
        program.single_core_k, program.base_m, program.base_n,
        program.base_k, program.depth_a1, program.depth_b1,
        program.a_l1_k_tiles, program.b_l1_k_tiles,
    ) == (1, 16, 64, 64, 16, 64, 64, 1, 1, 1, 1)
    source_artifacts = ascend_gemm_launch.matmul_v2_tiling_abi_provenance()[
        "source_artifacts"]
    dynamic_source = next(
        artifact for artifact in source_artifacts
        if artifact["path"].endswith("dynamic/batch_matmul_v2.py"))
    assert len(dynamic_source["sha256"]) == 64


@pytest.mark.parametrize(
    ("shape", "expected_grid", "expected_tile"),
    (
        ({"b": 256, "m": 1, "n": 5120, "k": 1536},
         (24, 1, 1), {"m": 128, "n": 112, "k": 128}),
        ({"b": 272, "m": 1, "n": 8192, "k": 8192},
         (24, 1, 1), {"m": 128, "n": 256, "k": 64}),
    ),
)
def test_ascend_910b_analytical_gemm_bounds_double_l0c_program(
        hardware, shape, expected_grid, expected_tile):
    launch = select_launch_config(
        "ascend_910b", "GEMM", shape, "bf16", "gemm",
        {"launch_policy": "analytical_launch"},
        hardware=hardware["ascend_910b"])

    assert launch.grid == expected_grid
    assert dict(launch.tiling.tile_shape) == expected_tile
    assert launch.tiling.pipeline_depth == 2
    assert launch.tiling.split_k == 1


@pytest.mark.parametrize(
    ("shape", "expected_program"),
    (
        (
            {"b": 1, "m": 1, "n": 6144, "k": 4096},
            (24, 16, 256, 64, 16, 256, 1, 2, 64, 4),
        ),
        (
            {"b": 1, "m": 1, "n": 4096, "k": 11008},
            (22, 16, 96, 64, 16, 192, 2, 2, 8, 8),
        ),
        (
            {"b": 1, "m": 1, "n": 5120, "k": 13824},
            (23, 16, 112, 64, 16, 224, 2, 2, 8, 8),
        ),
    ),
)
def test_ascend_910b_analytical_narrow_gemm_uses_l1_residency_program(
        hardware, shape, expected_program):
    """Narrow MatMulV2 programs select L1 residency before tile replay."""

    launch = select_launch_config(
        "ascend_910b", "GEMM", shape, "bf16", "gemm",
        {"launch_policy": "analytical_launch"},
        hardware=hardware["ascend_910b"])
    source = launch.source_program
    assert source is not None
    assert (
        source.aic_workers,
        source.base_m,
        source.base_n,
        source.base_k,
        source.single_core_m,
        source.single_core_n,
        source.depth_a1,
        source.depth_b1,
        source.a_l1_k_tiles,
        source.b_l1_k_tiles,
    ) == expected_program


@pytest.mark.parametrize(
    ("seq_len", "expected_grid"),
    ((1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (64, 32),
     (256, 43), (1024, 47)),
)
def test_ascend_910b_rope_launch_uses_48_aiv_cores(
        hardware, seq_len, expected_grid):
    launch = select_launch_config(
        "ascend_910b", "VECTOR",
        {
            "seq_len": seq_len,
            "num_query_heads": 128,
            "num_kv_heads": 8,
            "head_dim": 128,
            "element_count": seq_len * (128 + 8) * 128,
        },
        "bf16", "rotary", {}, hardware=hardware["ascend_910b"])

    assert launch.grid == (expected_grid, 1, 1)


def test_ascend_910b_split_qkv_rmsnorm_rope_uses_source_tiling(hardware):
    launch = select_launch_config(
        "ascend_910b", "VECTOR",
        {
            "seq_len": 12,
            "num_query_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "rotary_dim": 128,
        },
        "bf16", "qkv_rmsnorm_rope",
        {"bias": False, "is_neox": True},
        hardware=hardware["ascend_910b"])

    assert launch.grid == (48, 1, 1)
    assert launch.block == (1, 1, 1)
    assert launch.parameters == {
        "source_vllm_ascend_split_qkv_rmsnorm_rope_program": 1.0,
        "qk_batch_rows_per_tile": 1.0,
        "v_batch_rows_per_tile": 42.0,
        "source_ub_bytes": 87_040.0,
    }


@pytest.mark.parametrize(
    ("seq_len", "expected_key", "expected_slice"),
    ((16, 22.0, 4096.0), (64, 24.0, 2048.0)),
)
def test_ascend_910b_rope_publishes_atb_bf16_source_program(
        hardware, seq_len, expected_key, expected_slice):
    shape = {
        "seq_len": seq_len,
        "num_query_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
    }
    launch = select_launch_config(
        "ascend_910b", "VECTOR", shape, "bf16", "rotary", {},
        hardware=hardware["ascend_910b"])
    elements = seq_len * (64 + 8) * 128
    cos_sin_casts = 2 * seq_len * min(64 * 128, int(expected_slice))
    worker_count = launch.grid_size
    base_tokens, remainder = divmod(seq_len, worker_count)
    second_mask_workers = (
        worker_count
        if expected_key == 24.0 and base_tokens > 1
        else (remainder if expected_key == 24.0 else 0))
    scalar_mask_writes = 128 * (worker_count + second_mask_workers)
    query_heads_per_slice = min(64 * 128, int(expected_slice)) // 128
    mask_copy_operations = (
        (worker_count + second_mask_workers)
        * (query_heads_per_slice - 1) * 128 * 4)
    cos_sin_copy_operations = (
        seq_len * 2 * (query_heads_per_slice - 1) * 128 * 2)
    half_rotation_copy_operations = 2 * elements

    assert launch.parameters == {
        "cann_tiling_key": expected_key,
        "lowered_vector_operations": float(
            5 * elements + cos_sin_casts + scalar_mask_writes
            + mask_copy_operations + cos_sin_copy_operations
            + half_rotation_copy_operations),
        "source_bound_atb_bf16_rope_dag": 1.0,
        "atb_rope_slice_elements": expected_slice,
        "atb_rope_rotary_coefficient": 2.0,
        "atb_rope_cos_format": 0.0,
        "source_cann_rope_index_select_count": 2.0,
        "source_cann_rope_gather_v3_program": 10330.0,
        "cann_rope_gather_v3_real_core_num": float(min(seq_len, 48)),
        "cann_rope_gather_v3_selected_row_bytes": float(seq_len * 128 * 2),
        "cann_rope_gather_v3_tiling_payload_bytes": 192.0,
    }


@pytest.mark.parametrize("variant", ("rmsnorm", "add_rmsnorm"))
@pytest.mark.parametrize(
    ("rows", "expected_grid"),
    ((1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (64, 32),
     (256, 43), (1024, 47)),
)
def test_ascend_910b_norm_launch_uses_48_aiv_cores(
        hardware, variant, rows, expected_grid):
    launch = select_launch_config(
        "ascend_910b", "VECTOR",
        {"rows": rows, "hidden": 8192},
        "bf16", variant, {}, hardware=hardware["ascend_910b"])

    assert launch.grid == (expected_grid, 1, 1)


@pytest.mark.parametrize(
    ("seq_len", "expected_grid"),
    ((1, 1), (2, 2), (4, 4), (8, 8), (16, 16),
     (64, 48), (256, 48), (1024, 48)),
)
def test_ascend_910b_kv_cache_launch_uses_48_aiv_cores(
        hardware, seq_len, expected_grid):
    launch = select_launch_config(
        "ascend_910b", "KV_CACHE",
        {
            "seq_len": seq_len,
            "num_kv_heads": 8,
            "head_dim": 128,
            "element_count": seq_len * 8 * 128,
        },
        "bf16", "paged_kv_cache", PAGED_KV_CACHE_ATTRIBUTES,
        hardware=hardware["ascend_910b"])

    assert launch.grid == (expected_grid, 1, 1)


@pytest.mark.parametrize(
    ("seq_len", "num_kv_heads", "head_dim", "expected_key"),
    (
        (1, 4, 128, 200_000_000.0),
        (999, 4, 128, 200_000_000.0),
        (1000, 4, 128, 202_000_000.0),
        (8, 1, 7, 202_000_000.0),
    ),
)
def test_ascend_910b_kv_cache_selects_source_tiling_key(
        hardware, seq_len, num_kv_heads, head_dim, expected_key):
    launch = select_launch_config(
        "ascend_910b", "KV_CACHE",
        {
            "seq_len": seq_len,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
        },
        "bf16", "paged_kv_cache", PAGED_KV_CACHE_ATTRIBUTES,
        hardware=hardware["ascend_910b"])

    assert launch.parameters == {"cann_tiling_key": expected_key}


@pytest.mark.parametrize(
    ("variant", "shape", "attributes"),
    (
        ("softmax", {"element_count": 4096}, {}),
        ("gemm_reduce", {"element_count": 4096}, {}),
        (
            "moe_swiglu",
            {
                "tokens": 8,
                "experts": 128,
                "topk": 8,
                "hidden": 4096,
                "intermediate": 14336,
            },
            {"routed_rows": 64},
        ),
    ),
)
def test_ascend_910b_general_vector_launch_uses_48_aiv_cores(
        hardware, variant, shape, attributes):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", shape, "bf16", variant, attributes,
        hardware=hardware["ascend_910b"])

    assert launch.grid == (48, 1, 1)


@pytest.mark.parametrize(
    ("tokens", "experts", "topk", "expected_key"),
    (
        (12_000, 256, 1, 1001.0),
        (12_001, 256, 1, 1002.0),
        (8_192, 8, 8, 1001.0),
        (65_536, 9, 1, 1002.0),
        (12_000, 257, 1, 1003.0),
        (12_001, 257, 1, 1003.0),
        (1 << 24, 256, 1, 1002.0),
        (1 << 24, 2_048, 1, 1003.0),
    ),
)
def test_ascend_910b_moe_compute_expert_tokens_selects_source_program(
        hardware, tokens, experts, topk, expected_key):
    launch = select_launch_config(
        "ascend_910b", "VECTOR",
        {
            "tokens": tokens,
            "experts": experts,
            "topk": topk,
            "hidden": 4096,
            "intermediate": 14336,
        },
        "bf16", "moe_compute_expert_tokens", {},
        hardware=hardware["ascend_910b"])

    assert launch.grid == (48, 1, 1)
    assert launch.block == (1, 1, 1)
    assert launch.parameters == {
        "cann_tiling_key": expected_key,
        "cann_moe_compute_expert_tokens_mix_aiv_1_0": 1.0,
    }


@pytest.mark.parametrize(
    ("variant", "attributes", "expected_grid"),
    (
        ("moe_cast", {"cast_role": "topk_ids"}, (47, 1, 1)),
        ("moe_cast", {"cast_role": "expert_tokens"}, (1, 1, 1)),
        ("moe_init_routing", {}, (48, 1, 1)),
        ("moe_finalize_routing", {}, (48, 1, 1)),
        ("moe_range", {}, (48, 1, 1)),
        ("moe_transpose", {}, (48, 1, 1)),
    ),
)
def test_ascend_910b_moe_auxiliary_launch_matches_cann_source(
        hardware, variant, attributes, expected_grid):
    launch = select_launch_config(
        "ascend_910b", "VECTOR",
        {
            "tokens": 32768,
            "experts": 8,
            "topk": 2,
            "hidden": 4096,
            "intermediate": 14336,
        },
        "bf16", variant, attributes,
        hardware=hardware["ascend_910b"])

    assert launch.grid == expected_grid
    assert launch.block == (1, 1, 1)
    if variant == "moe_init_routing":
        assert launch.parameters == {
            "cann_moe_init_routing_mix_aiv_1_0": 1.0}
    elif variant == "moe_finalize_routing":
        assert launch.parameters == {"cann_tiling_key": 20036.0}
    else:
        assert launch.parameters == {}


@pytest.mark.parametrize(
    ("elements", "expected_grid"),
    (
        (1_024, 1),
        (1_280, 2),
        (2_560, 3),
        (65_536, 47),
        (163_840, 48),
    ),
)
def test_ascend_910b_public_moe_abs_uses_its_own_source_tiler(
        hardware, elements, expected_grid):
    """Abs has no generic Add small-work launch branch."""

    launch = select_launch_config(
        "ascend_910b", "VECTOR",
        {
            "tokens": elements,
            "experts": 8,
            "topk": 1,
            "hidden": 2048,
            "intermediate": 512,
        },
        "bf16", "moe_abs", {}, hardware=hardware["ascend_910b"])

    assert launch.grid == (expected_grid, 1, 1)
    assert launch.block == (1, 1, 1)
    assert launch.parameters == {}


@pytest.mark.parametrize(
    ("shape", "expected_variant", "expected_cast_roles", "expected_grids"),
    (
        (
            {"tokens": 16, "experts": 128, "topk": 8,
             "hidden": 2048, "intermediate": 768},
            "moe_init_routing_v2", ("expert_tokens",),
            {"moe_gating_topk": (16, 1, 1),
             "moe_init_routing_v2": (48, 1, 1),
             "moe_cast": (1, 1, 1), "moe_abs": (1, 1, 1),
             "moe_token_unpermute": (16, 1, 1)},
        ),
        (
            {"tokens": 64, "experts": 512, "topk": 10,
             "hidden": 2048, "intermediate": 512},
            "moe_init_routing_v3", (),
            {"moe_gating_topk": (32, 1, 1),
             "moe_init_routing_v3": (48, 1, 1),
             "moe_abs": (1, 1, 1),
             "moe_token_unpermute": (48, 1, 1)},
        ),
    ),
)
def test_ascend_910b_public_fused_moe_recipe_uses_source_abi_topology(
        hardware, shape, expected_variant, expected_cast_roles,
        expected_grids):
    hw = hardware["ascend_910b"]
    recipe = select_fused_moe_recipe("ascend_910b", shape, hw)
    assert recipe[0:2] == ("moe_gating_topk", expected_variant)
    assert recipe[-2:] == ("moe_abs", "moe_token_unpermute")
    assert select_fused_moe_cast_roles(
        "ascend_910b", shape, hw) == expected_cast_roles
    for variant, expected_grid in expected_grids.items():
        attributes = (
            {"cast_role": expected_cast_roles[0]}
            if variant == "moe_cast" else {})
        launch = select_launch_config(
            "ascend_910b", "VECTOR", shape, "bf16", variant,
            attributes, hardware=hw)
        assert launch.grid == expected_grid
        assert launch.block == (1, 1, 1)


@pytest.mark.parametrize(
    ("tokens", "experts", "message"),
    (
        ((1 << 24) + 1, 256, "exceeds the source limit of 2\\*\\*24"),
        (1, 2_049, "expert count exceeds the source limit of 2048"),
    ),
)
def test_ascend_910b_moe_compute_expert_tokens_rejects_source_overflow(
        hardware, tokens, experts, message):
    with pytest.raises(ValueError, match=message):
        select_launch_config(
            "ascend_910b", "VECTOR",
            {
                "tokens": tokens,
                "experts": experts,
                "topk": 1,
                "hidden": 4096,
                "intermediate": 14336,
            },
            "bf16", "moe_compute_expert_tokens", {},
            hardware=hardware["ascend_910b"])


@pytest.mark.parametrize(
    ("operation", "expected_parameters"),
    (
        ("add", {
            "lowered_vector_operations": 12288.0,
            "cann_tiling_key": 210_000_000.0,
            "cann_tiling_payload_bytes": 56.0,
            "elementwise_block_factor": 1024.0,
            "elementwise_ub_factor": 1024.0,
            "elementwise_input_queue_depth": 1.0,
            "elementwise_input_ping_pong_slots": 2.0,
            "elementwise_output_queue_depth": 1.0,
            "source_cann_bf16_elementwise_dag": 1.0,
        }),
        ("mul", {
            "lowered_vector_operations": 12288.0,
            "cann_tiling_key": 210_020_000.0,
            "cann_tiling_payload_bytes": 56.0,
            "elementwise_block_factor": 1024.0,
            "elementwise_ub_factor": 1024.0,
            "elementwise_input_queue_depth": 4.0,
            "elementwise_input_ping_pong_slots": 2.0,
            "elementwise_output_queue_depth": 1.0,
            "source_cann_bf16_elementwise_dag": 1.0,
        }),
    ),
)
def test_ascend_910b_elementwise_declares_source_cast_passes(
        hardware, operation, expected_parameters):
    elements = 4096
    launch = select_launch_config(
        "ascend_910b", "VECTOR", {"element_count": elements}, "bf16",
        "elementwise", {"operation": operation},
        hardware=hardware["ascend_910b"])

    assert launch.grid == (4, 1, 1)
    assert launch.parameters == expected_parameters


def test_ascend_910b_analytical_elementwise_add_keeps_source_cast_passes(
        hardware):
    elements = 4096
    analytical = _select_launch_config(
        "ascend_910b", "VECTOR", {"element_count": elements}, "bf16",
        "elementwise",
        {"operation": "add", "launch_policy": "analytical_launch"},
        hardware=hardware["ascend_910b"])
    injected = _select_launch_config(
        "ascend_910b", "VECTOR", {"element_count": elements}, "bf16",
        "elementwise",
        {"operation": "add", "launch_policy": "injected_launch"},
        hardware=hardware["ascend_910b"])

    assert analytical == injected
    assert analytical.algorithm == "aclnn_elementwise"
    assert analytical.grid == (4, 1, 1)
    assert analytical.parameters == {
        "lowered_vector_operations": float(3 * elements),
        "cann_tiling_key": 210_000_000.0,
        "cann_tiling_payload_bytes": 56.0,
        "elementwise_block_factor": 1024.0,
        "elementwise_ub_factor": 1024.0,
        "elementwise_input_queue_depth": 1.0,
        "elementwise_input_ping_pong_slots": 2.0,
        "elementwise_output_queue_depth": 1.0,
        "source_cann_bf16_elementwise_dag": 1.0,
    }


@pytest.mark.parametrize(
    ("elements", "operation", "expected"),
    (
        (1535, "add", {
            "block_dim": 1, "block_factor": 1535, "ub_factor": 1535,
            "tiling_key": 210_000_000, "input_queue_depth": 1,
            "input_ping_pong_slots": 2,
            "output_queue_depth": 1,
        }),
        (1536, "add", {
            "block_dim": 2, "block_factor": 1024, "ub_factor": 1024,
            "tiling_key": 210_000_000, "input_queue_depth": 1,
            "input_ping_pong_slots": 2,
            "output_queue_depth": 1,
        }),
        (48 * 12288, "add", {
            "block_dim": 48, "block_factor": 12288, "ub_factor": 4096,
            "tiling_key": 210_010_000, "input_queue_depth": 2,
            "input_ping_pong_slots": 2,
            "output_queue_depth": 1,
        }),
        (1536, "mul", {
            "block_dim": 2, "block_factor": 1024, "ub_factor": 1024,
            "tiling_key": 210_020_000, "input_queue_depth": 4,
            "input_ping_pong_slots": 2,
            "output_queue_depth": 1,
        }),
        (48 * 9729, "mul", {
            "block_dim": 48, "block_factor": 9856, "ub_factor": 4992,
            "tiling_key": 210_020_000, "input_queue_depth": 4,
            "input_ping_pong_slots": 2,
            "output_queue_depth": 1,
        }),
    ),
)
def test_ascend_910b_elementwise_tiling_source_boundaries(
        elements, operation, expected):
    assert ascend_vector_launch.select_cann_elementwise_tiling(
        elements, operation, 48) == expected


def test_ascend_910b_elementwise_tiling_marks_unclosed_source_domain():
    assert ascend_vector_launch.select_cann_elementwise_tiling(
        16_777_217, "add", 48) is None


def test_ascend_910b_silu_source_rule_accepts_shape_outside_deleted_lut(
        hardware):
    launch = select_launch_config(
        "ascend_910b", "VECTOR",
        {"tok": 8193, "intermediate": 2080,
         "element_count": 8193 * 2080},
        "bf16", "silu", {"launch_policy": "injected_launch"},
        hardware=hardware["ascend_910b"])

    assert launch.grid == (48, 1, 1)
    assert launch.parameters["swiglu_base_row_len"] == 38.0
    assert launch.parameters["swiglu_base_col_len"] == 256.0
    assert launch.parameters["swiglu_total_tiles"] == 1944.0
    assert launch.parameters["swiglu_queue_depth"] == 2.0


@pytest.mark.parametrize(
    ("variant", "shape", "expected_grid", "expected_parameters"),
    (
        ("rmsnorm", {"rows": 1, "hidden": 12288}, (1, 1, 1),
         {
             "source_bound_rmsnorm_single_row_dag": 1.0,
             "lowered_vector_input_operations": float(12288),
             "lowered_vector_output_operations": float(2 * 12288),
             "norm_weight_l2_copies": 1.0,
             "rstd_output_bytes": 4.0,
             "cann_rmsnorm_rstd_transfer_bytes": 32.0,
             "norm_epsilon_add_flops": 1.0,
             "norm_block_factor": 1.0,
             "cann_tiling_payload_bytes": 28.0,
             "cann_tiling_key": 0.0,
         }),
        ("rmsnorm", {"rows": 256, "hidden": 12289}, (43, 1, 1),
         {
             "source_bound_rmsnorm_split_d_dag": 1.0,
             "input_reread_count": 1.0,
             "lowered_vector_input_operations": float(256 * 12289),
             "lowered_vector_output_operations": float(3 * 256 * 12289),
             "norm_weight_l2_copies": 43.0,
             "rstd_output_bytes": 1024.0,
             "cann_rmsnorm_rstd_transfer_bytes": 1376.0,
             "norm_epsilon_add_flops": 256.0,
             "norm_block_factor": 6.0,
             "cann_tiling_payload_bytes": 28.0,
             "cann_tiling_key": 1.0,
         }),
        ("add_rmsnorm", {"rows": 1, "hidden": 12288}, (1, 1, 1),
         {
             "source_bound_add_rmsnorm_single_n_dag": 1.0,
             "lowered_vector_input_operations": float(3 * 12288),
             "lowered_vector_reduce_operations": float(3 * 12288),
             "lowered_vector_output_operations": float(4 * 12288),
             "norm_weight_l2_copies": 1.0,
             "rstd_output_bytes": 4.0,
             "norm_epsilon_add_flops": 1.0,
             "norm_block_factor": 1.0,
             "cann_tiling_payload_bytes": 80.0,
             "cann_tiling_key": 33.0,
         }),
        ("add_rmsnorm", {"rows": 256, "hidden": 12289}, (43, 1, 1),
         {
             "source_bound_add_rmsnorm_split_d_dag": 1.0,
             "input_reread_count": 1.0,
             "lowered_vector_input_operations": float(3 * 256 * 12289),
             "lowered_vector_reduce_operations": float(256 * 12289 + 256),
             "lowered_vector_output_operations": float(5 * 256 * 12289),
             "norm_weight_l2_copies": 43.0,
             "rstd_output_bytes": 1024.0,
             "norm_epsilon_add_flops": 256.0,
             "norm_block_factor": 6.0,
             "cann_tiling_payload_bytes": 80.0,
             "cann_tiling_key": 31.0,
         }),
    ),
)
def test_ascend_910b_norm_split_d_declares_source_reread(
        hardware, variant, shape, expected_grid, expected_parameters):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", shape, "bf16", variant, {},
        hardware=hardware["ascend_910b"])

    assert launch.grid == expected_grid
    assert launch.parameters == expected_parameters


def test_ascend_910b_norm_tiling_payload_contract_fails_closed(hardware):
    """A CANN normalization entry cannot inherit another entry's prologue."""
    shape = {"rows": 64, "hidden": 16384}
    launch = select_launch_config(
        "ascend_910b", "VECTOR", shape, "bf16", "add_rmsnorm", {},
        hardware=hardware["ascend_910b"])

    missing_payload = dict(launch.parameters)
    del missing_payload["cann_tiling_payload_bytes"]
    with pytest.raises(ValueError, match="launch parameter schema mismatch"):
        KernelSpec(
            KernelKind.VECTOR, "ascend_910b", shape, dtype="bf16",
            variant=launch.execution_variant,
            launch=dataclasses.replace(launch, parameters=missing_payload))

    wrong_payload = dict(launch.parameters)
    wrong_payload["cann_tiling_payload_bytes"] = 28.0
    with pytest.raises(ValueError, match="tiling payload disagrees"):
        KernelSpec(
            KernelKind.VECTOR, "ascend_910b", shape, dtype="bf16",
            variant=launch.execution_variant,
            launch=dataclasses.replace(launch, parameters=wrong_payload))


@pytest.mark.parametrize(
        ("variant", "rows", "hidden", "extra_passes", "rstd_init_rows"),
        (
            ("rmsnorm", 48, 12288, 3, False),
            ("rmsnorm", 49, 12289, 4, False),
            ("add_rmsnorm", 48, 12288, 10, False),
            ("add_rmsnorm", 49, 12288, 8, False),
            # Split-D key-31 initializes its VECOUT rstd tensor once per
            # row before the Former loop; this is Vector work, not another
            # elementwise pass.
            ("add_rmsnorm", 49, 12289, 9, True),
        ),
    )
def test_ascend_910b_norm_declares_source_lowered_vector_passes(
        hardware, variant, rows, hidden, extra_passes, rstd_init_rows):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", {"rows": rows, "hidden": hidden},
        "bf16", variant, {}, hardware=hardware["ascend_910b"])

    phase_operations = sum(
        launch.parameters[name]
        for name in (
            "lowered_vector_input_operations",
            "lowered_vector_reduce_operations",
            "lowered_vector_output_operations",
        )
        if name in launch.parameters)
    assert phase_operations == float(
        extra_passes * rows * hidden + (rows if rstd_init_rows else 0))


@pytest.mark.parametrize("rows", (1, 2, 4, 8, 16))
@pytest.mark.parametrize("hidden", (2048, 2560, 4096, 5120, 8192))
def test_ascend_910b_add_rms_single_n_key33_selector_coverage(
        hardware, rows, hidden):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", {"rows": rows, "hidden": hidden},
        "bf16", "add_rmsnorm", {}, hardware=hardware["ascend_910b"])

    assert launch.grid == (rows, 1, 1)
    assert launch.block == (1, 1, 1)
    assert launch.parameters["cann_tiling_key"] == 33.0
    assert launch.parameters["norm_block_factor"] == 1.0
    assert launch.parameters[
        "source_bound_add_rmsnorm_single_n_dag"] == 1.0


@pytest.mark.parametrize(
    ("rows", "hidden"),
    ((3, 128), (17, 8192)),
)
def test_ascend_910b_add_rms_single_n_noncanonical_source_domain(
        hardware, rows, hidden):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", {"rows": rows, "hidden": hidden},
        "bf16", "add_rmsnorm", {}, hardware=hardware["ascend_910b"])

    assert launch.grid == (rows, 1, 1)
    assert launch.parameters["cann_tiling_key"] == 33.0
    assert launch.parameters[
        "source_bound_add_rmsnorm_single_n_dag"] == 1.0


@pytest.mark.parametrize(
    ("rows", "hidden", "expected_key"),
    ((49, 12288, 30.0), (48, 12289, 31.0)),
)
def test_ascend_910b_add_rms_single_n_source_domain_boundaries(
        hardware, rows, hidden, expected_key):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", {"rows": rows, "hidden": hidden},
        "bf16", "add_rmsnorm", {}, hardware=hardware["ascend_910b"])

    assert launch.parameters["cann_tiling_key"] == expected_key
    assert "source_bound_add_rmsnorm_single_n_dag" not in launch.parameters
    if expected_key == 30.0:
        assert launch.parameters[
            "source_bound_add_rmsnorm_row_group_dag"] == 1.0
    else:
        assert (
            "source_bound_add_rmsnorm_row_group_dag"
            not in launch.parameters)
        assert launch.parameters[
            "source_bound_add_rmsnorm_split_d_dag"] == 1.0


@pytest.mark.parametrize(
    ("rows", "hidden", "expected_grid", "expected_block_factor"),
    (
        (49, 2001, 25, 2),
        (64, 2048, 32, 2),
        (256, 8192, 43, 6),
        (1024, 12288, 47, 22),
    ),
)
def test_ascend_910b_add_rms_key30_source_domain(
        hardware, rows, hidden, expected_grid, expected_block_factor):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", {"rows": rows, "hidden": hidden},
        "bf16", "add_rmsnorm", {}, hardware=hardware["ascend_910b"])

    assert launch.grid == (expected_grid, 1, 1)
    assert launch.block == (1, 1, 1)
    assert launch.parameters["cann_tiling_key"] == 30.0
    assert launch.parameters["norm_block_factor"] == float(
        expected_block_factor)
    assert launch.parameters[
        "source_bound_add_rmsnorm_row_group_dag"] == 1.0


def test_ascend_910b_add_rms_rejects_unmodeled_small_hidden_multi_row(
        hardware):
    with pytest.raises(ValueError, match="unsupported MergeN/MultiN"):
        select_launch_config(
            "ascend_910b", "VECTOR", {"rows": 49, "hidden": 2000},
            "bf16", "add_rmsnorm", {}, hardware=hardware["ascend_910b"])


# Columns at or below SMALL_REDUCE_NUM select MERGE_N on this SoC
# (rms_norm_tiling.cpp:524-526), so the key-0 domains below stay above that
# bound; test_vector_rmsnorm_merge_n_program.py owns the key-2 domain.
@pytest.mark.parametrize(
    ("rows", "hidden"),
    ((1, 2048), (3, 8192), (48, 12288)),
)
def test_ascend_910b_rms_single_row_source_domain(
        hardware, rows, hidden):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", {"rows": rows, "hidden": hidden},
        "bf16", "rmsnorm", {}, hardware=hardware["ascend_910b"])

    assert launch.grid == (rows, 1, 1)
    assert launch.parameters["cann_tiling_key"] == 0.0
    assert launch.parameters["norm_block_factor"] == 1.0
    assert launch.parameters["cann_rmsnorm_rstd_transfer_bytes"] == (
        float(32 * rows))
    assert launch.parameters[
        "source_bound_rmsnorm_single_row_dag"] == 1.0


@pytest.mark.parametrize(
    ("rows", "hidden", "expected_key"),
    ((49, 12288, 0.0), (48, 12289, 1.0)),
)
def test_ascend_910b_rms_single_row_source_domain_boundaries(
        hardware, rows, hidden, expected_key):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", {"rows": rows, "hidden": hidden},
        "bf16", "rmsnorm", {}, hardware=hardware["ascend_910b"])

    assert launch.parameters["cann_tiling_key"] == expected_key
    assert "source_bound_rmsnorm_single_row_dag" not in launch.parameters
    if expected_key == 0.0:
        assert launch.parameters[
            "source_bound_rmsnorm_row_group_dag"] == 1.0
    else:
        assert "source_bound_rmsnorm_row_group_dag" not in launch.parameters
        assert launch.parameters[
            "source_bound_rmsnorm_split_d_dag"] == 1.0


@pytest.mark.parametrize(
    ("rows", "hidden", "expected_grid", "expected_block_factor"),
    (
        (1, 12289, 1, 1),
        (256, 16384, 43, 6),
        (3072, 16384, 48, 64),
        (6145, 16384, 48, 129),
    ),
)
def test_ascend_910b_rms_key1_split_d_source_domain(
        hardware, rows, hidden, expected_grid, expected_block_factor):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", {"rows": rows, "hidden": hidden},
        "bf16", "rmsnorm", {}, hardware=hardware["ascend_910b"])

    assert launch.grid == (expected_grid, 1, 1)
    assert launch.block == (1, 1, 1)
    assert launch.parameters["cann_tiling_key"] == 1.0
    assert launch.parameters["norm_block_factor"] == float(
        expected_block_factor)
    assert launch.parameters[
        "source_bound_rmsnorm_split_d_dag"] == 1.0


@pytest.mark.parametrize(
    ("rows", "hidden", "expected_grid", "expected_block_factor",
     "expected_rstd_transfer_bytes"),
    (
        (49, 2048, 25, 2, 800),
        (64, 2048, 32, 2, 1024),
        (256, 8192, 43, 6, 1376),
        (4096, 12288, 48, 86, 16768),
    ),
)
def test_ascend_910b_rms_key0_row_group_source_domain(
        hardware, rows, hidden, expected_grid, expected_block_factor,
        expected_rstd_transfer_bytes):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", {"rows": rows, "hidden": hidden},
        "bf16", "rmsnorm", {}, hardware=hardware["ascend_910b"])

    assert launch.grid == (expected_grid, 1, 1)
    assert launch.block == (1, 1, 1)
    assert launch.parameters["cann_tiling_key"] == 0.0
    assert launch.parameters["norm_block_factor"] == float(
        expected_block_factor)
    assert launch.parameters["cann_rmsnorm_rstd_transfer_bytes"] == float(
        expected_rstd_transfer_bytes)
    assert launch.parameters[
        "source_bound_rmsnorm_row_group_dag"] == 1.0


@pytest.mark.parametrize(
    ("rows", "hidden", "expected_copies"),
    (
        (1, 8192, 1),
        (64, 8192, 32),
        (1024, 8192, 47),
        (256, 16384, 43),
        (4096, 16384, 95),
    ),
)
@pytest.mark.parametrize("variant", ("rmsnorm", "add_rmsnorm"))
def test_ascend_910b_norm_declares_source_gamma_copy_count(
        hardware, variant, rows, hidden, expected_copies):
    launch = select_launch_config(
        "ascend_910b", "VECTOR", {"rows": rows, "hidden": hidden},
        "bf16", variant, {}, hardware=hardware["ascend_910b"])

    assert launch.parameters["norm_weight_l2_copies"] == float(
        expected_copies)


@pytest.mark.parametrize("variant", ("rmsnorm", "add_rmsnorm"))
def test_ascend_910b_norm_launch_requires_hidden(hardware, variant):
    with pytest.raises(ValueError, match="missing launch shape fields: hidden"):
        select_launch_config(
            "ascend_910b", "VECTOR", {"rows": 1}, "bf16", variant, {},
            hardware=hardware["ascend_910b"])


@pytest.mark.parametrize(
    ("hardware_id", "hidden", "has_reread"),
    (
        ("nvidia_a800", 8192, True),
        ("metax_c500", 8192, True),
        ("ascend_910b", 8192, False),
        ("ascend_910b", 16384, True),
    ),
)
def test_norm_reread_launch_fact_is_hardware_local(
        hardware, hardware_id, hidden, has_reread):
    launch = select_launch_config(
        hardware_id, "VECTOR", {"rows": 2, "hidden": hidden}, "bf16",
        "rmsnorm", {}, hardware=hardware[hardware_id])

    assert ("input_reread_count" in launch.parameters) is has_reread
    if has_reread:
        assert launch.parameters["input_reread_count"] == 1.0


def test_unregistered_hardware_fails_loudly():
    with pytest.raises(NotImplementedError, match="no production launch selector"):
        select_launch_config(
            "ucb_gemmini", "GEMM", {"b": 1, "m": 1, "n": 1, "k": 1},
            "bf16", "gemm", {}, None)


@pytest.mark.parametrize(
    "hardware_id", ("nvidia_a800", "metax_c500", "ascend_910b"))
def test_unknown_normalization_vector_variant_fails_loudly(hardware, hardware_id):
    with pytest.raises(
            ValueError, match="unsupported (GPU|CANN) VECTOR launch variant"):
        select_launch_config(
            hardware_id,
            "NORM",
            {"rows": 1, "hidden": 128},
            "bf16",
            "batchnorm",
            {},
            hardware=hardware[hardware_id],
        )


def test_nvidia_a800_analytical_launch_does_not_load_injected_lut(
        hardware, monkeypatch):
    def fail_if_loaded():
        raise AssertionError("analytical_launch loaded the injected tile LUT")

    monkeypatch.setattr(nvidia_a800_gemm_launch, "_tiling_lut", fail_if_loaded)
    launch = select_launch_config(
        "nvidia_a800",
        "GEMM",
        {"b": 1, "m": 111, "n": 6144, "k": 2560},
        "bf16",
        "gemm",
        {"has_bias": False, "position": "generic",
         "launch_policy": "analytical_launch"},
        hardware=hardware["nvidia_a800"],
    )

    assert dict(launch.tiling.tile_shape) == {"m": 128, "n": 128, "k": 64}


@pytest.mark.parametrize(
    ("stages", "expected"),
    (("8x1", (8, 1)), ("32x10", (32, 10)), ("128xAUTO", (128, None))),
)
def test_nvidia_a800_cublaslt_stage_projection_is_exact(stages, expected):
    assert parse_cublaslt_stages(stages) == expected


@pytest.mark.parametrize("stages", ("64x0", "64x11", "64xauto", "undefined"))
def test_nvidia_a800_cublaslt_stage_projection_rejects_unknown_enum(stages):
    with pytest.raises(ValueError, match="cuBLASLt stages must match"):
        parse_cublaslt_stages(stages)


def _nvidia_a800_exact_test_lut():
    workload_specs = (
        ("gemm", 1, 128, 64, "qkv_proj", 1, 1),
        ("gemv", 1, 256, 64, "lm_head", 1, 1),
        ("gemm", 8192, 128, 64, "qkv_proj", 1, 1),
        ("gemm", 128, 64, 64, "qkv_proj", 2, 1),
        ("gemm", 64, 32, 64, "router", 2, 2),
        ("gemm", 64, 32, 64, "qkv_proj", 1, 1),
    )
    workloads = {}
    operations = {}
    for variant, m, n, k, position, tp, ep in workload_specs:
        attributes = {
            "a_layout": "row_major",
            "b_layout": "row_major",
            "c_layout": "row_major",
            "trans_a": False,
            "trans_b": True,
            "epilogue": "none",
            "has_bias": False,
            "position": position,
            "tensor_parallel_size": tp,
            "expert_parallel_size": ep,
        }
        semantic_key = gemm_workload_key(
            {"b": 1, "m": m, "n": n, "k": k},
            "bf16", variant, attributes)
        operation_key = json.dumps(
            [m, n, k, "bf16", False], separators=(",", ":"))
        workloads[semantic_key] = operation_key
        if operation_key not in operations:
            algorithm = {
                "rank": 0,
                "algorithm_id": 31,
                "tile_id": 15,
                "tile": "64x64",
                "logical_tile_m": 64,
                "logical_tile_n": 64,
                "stages_id": 17,
                "stages": "64x5",
                "inner_shape_id": 0,
                "split_k": 1,
                "reduction_scheme": 0,
                "cta_swizzle": 1,
                "custom_option": 0,
                "workspace_bytes": 0,
                "waves_count": 1.0,
            }
            operations[operation_key] = {
                "operation": {
                    "m": m,
                    "n": n,
                    "k": k,
                    "dtype": "bf16",
                    "layout": "row_major",
                    "trans_a": False,
                    "trans_b": True,
                    "has_bias": False,
                },
                "queried_device": "0",
                "algorithms": [algorithm],
                "main_launch": {
                    "api": "cuLaunchKernel",
                    "symbol": "cutlass_main",
                    "grid": [1, 1, 1],
                    "block": [128, 1, 1],
                    "shared_memory_bytes": 32768,
                },
                "reducer_launch": None,
                "device_execution": {
                    "evidence_kind": "cupti_activity_kernel",
                    "evidence_version": 1,
                    "activity_record_count": 1,
                    "executed_symbols": ["midlayer_noop"],
                    "all_replacements_observed": True,
                    "dropped_record_count": 0,
                },
                "selection": {
                    "contract": "vendor_rank_0_heuristic",
                    "selected_rank": 0,
                    "reason": "cublaslt_rank_0_under_captured_preference",
                },
                "projection": {
                    "execution_variant": "gemm",
                    "logical_tile_m": 64,
                    "logical_tile_n": 64,
                    "tile_k": 64,
                    "pipeline_depth": 5,
                    "split_k": 1,
                    "reduction_scheme": 0,
                    "workspace_bytes": 0,
                },
            }
    workload_keys = sorted(workloads)
    domain_sha256 = hashlib.sha256(json.dumps(
        workload_keys, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8")).hexdigest()
    model_specs = [
        ("Qwen3-4B", "qwen3", 1, 1, "${MODELS_ROOT}/Qwen3-4B"),
        ("Qwen3-8B", "qwen3", 1, 1, "${MODELS_ROOT}/Qwen3-8B"),
        ("Llama-3.1-8B", "llama", 1, 1,
         "${MODELS_ROOT}/Llama-3.1-8B-Instruct"),
        ("Qwen3-32B", "qwen3", 2, 1, "${MODELS_ROOT}/Qwen3-32B"),
        ("Qwen3-30B-A3B", "qwen3_moe", 2, 2,
         "${MODELS_ROOT}/Qwen3-30B-A3B"),
    ]
    hardware_ids = ["nvidia_a800", "metax_c500", "ascend_910b"]

    def position_domains(model_type):
        if model_type == "qwen3_moe":
            positions = ("qkv_proj", "o_proj", "router", "lm_head")
        elif model_type in ("qwen3", "llama"):
            positions = (
                "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "lm_head")
        else:
            raise AssertionError(f"unexpected fixture model type {model_type}")
        result = []
        for position in positions:
            if position == "lm_head":
                m_max = 256
                variant_contract = "m1_gemv_else_gemm"
            else:
                m_max = 8192
                variant_contract = "gemm"
            result.append({
                "position": position,
                "m_min": 1,
                "m_max": m_max,
                "semantic_key_count": m_max,
                "n": 64,
                "k": 64,
                "dtype": "bf16",
                "has_bias": False,
                "variant_contract": variant_contract,
            })
        return result

    def model_key_count(model_type):
        if model_type == "qwen3_moe":
            return 24832
        if model_type in ("qwen3", "llama"):
            return 33024
        raise AssertionError(f"unexpected fixture model type {model_type}")

    manifest = {
        "schema_version": 2,
        "artifact_type": "gemm_workload_candidate",
        "source_config": (
            "configs/evaluation/experiments/gemm_launch_domain.json"),
        "source_config_sha256": "5" * 64,
        "duration_consumed": False,
        "workload_key_fields": list(
            nvidia_a800_gemm_launch.GEMM_WORKLOAD_KEY_FIELDS),
        "physical_operation_fields": [
            "m", "n", "k", "dtype", "has_bias"],
        "scheduler_contract": {
            "backend": "vllm",
            "scheduler_version": "0.11.0",
            "scheduler_source_commit": "6" * 40,
            "max_batched_tokens": 8192,
            "max_requests": 256,
            "enable_chunked_prefill": True,
            "mtp_num_speculative_tokens": 0,
        },
        "coverage_scope": {
            "kind": "static_complete_legal_launch_selector_domain",
            "request_trajectory_consumed": False,
            "scheduler_alignment_consumed": False,
            "operator_trace_consumed": False,
            "timing_consumed": False,
            "fused_moe_included": False,
            "serving_end_to_end_executability_claimed": False,
        },
        "limitations": [{
            "scope": "serving_workload_lowering",
            "affected_hardware_ids": ["metax_c500", "ascend_910b"],
            "affected_model_ids": ["Qwen3-32B"],
            "reason": "fixture limitation",
        }],
        "dataset_model_inventory": {
            hardware_id: sorted(item[0] for item in model_specs)
            for hardware_id in hardware_ids
        },
        "model_domains": [{
            "model_id": model_id,
            "model_path": model_path,
            "model_config_sha256": {
                "Qwen3-4B": "a" * 64,
                "Qwen3-8B": "b" * 64,
                "Llama-3.1-8B": "c" * 64,
                "Qwen3-32B": "d" * 64,
                "Qwen3-30B-A3B": "e" * 64,
            }[model_id],
            "model_type": model_type,
            "tensor_parallel_size": tp,
            "expert_parallel_size": ep,
            "positions": position_domains(model_type),
            "semantic_key_count": model_key_count(model_type),
            "physical_operation_count": model_key_count(model_type),
        } for model_id, model_type, tp, ep, model_path in model_specs],
        "workload_keys": workload_keys,
        "semantic_key_count": len(workloads),
        "physical_operation_count": len(operations),
        "hardware_domains": {
            hardware_id: {
                "status": "exact",
                "domain_sha256": domain_sha256,
                "semantic_key_count": len(workloads),
                "physical_operation_count": len(operations),
            }
            for hardware_id in hardware_ids
        },
        "cross_hardware": {
            "status": "exact_equal",
            "hardware_ids": hardware_ids,
            "domain_sha256": domain_sha256,
            "semantic_key_count": len(workloads),
            "physical_operation_count": len(operations),
        },
    }
    expanded = {
        "schema_version": 3,
        "artifact_type": "nvidia_a800_gemm_launch_facts",
        "hardware_id": "nvidia_a800",
        "software_revision": "cuda_12.2",
        "cublaslt_version": 120205,
        "source": "candidate.json",
        "generated_by": "scripts/launch_data/nvidia_a800_launch_generator.py",
        "candidate_sha256": "1" * 64,
        "capture_baseline_commit": "2" * 40,
        "runner_sha256": "3" * 64,
        "query_source_sha256": "4" * 64,
        "workload_manifest_sha256": hashlib.sha256(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(
                "utf-8")).hexdigest(),
        "workload_manifest": manifest,
        "workload_key_fields": list(
            nvidia_a800_gemm_launch.GEMM_WORKLOAD_KEY_FIELDS),
        "selector_contract": "vendor_rank_0_heuristic",
        "duration_consumed": False,
        "matmul_executed": False,
        "production_kernel_submitted": False,
        "max_workspace_bytes": 32 * 1024 * 1024,
        "semantic_key_count": len(workloads),
        "physical_operation_count": len(operations),
        "workloads": workloads,
        "operations": operations,
    }
    reduction_names = {
        0: "none", 1: "in_place", 2: "fp32_external",
        4: "output_dtype_external",
    }
    entries = {}
    for workload_key, operation_key in workloads.items():
        projection = operations[operation_key]["projection"]
        if projection["execution_variant"] == "gemm":
            tile = [
                projection["logical_tile_m"],
                projection["logical_tile_n"],
                projection["tile_k"],
            ]
        elif projection["execution_variant"] == "gemv":
            tile = None
        else:
            raise AssertionError("unexpected test execution variant")
        entries[workload_key] = {
            "tile": tile,
            "pipeline_depth": projection["pipeline_depth"],
            "split_k": projection["split_k"],
            "reduction": reduction_names[projection["reduction_scheme"]],
        }
    canonical = {
        "schema_version": 4,
        "artifact_type": "nvidia_a800_gemm_launch_selection",
        "hardware_id": "nvidia_a800",
        "software_revision": "cuda_12.2",
        "cublaslt_version": 120205,
        "generated_by": "scripts/launch_data/nvidia_a800_launch_generator.py",
        "evidence_sha256": "0" * 64,
        "candidate_sha256": expanded["candidate_sha256"],
        "capture_baseline_commit": expanded["capture_baseline_commit"],
        "duration_consumed": False,
        "domain_sha256": "f" * 64,
        "workload_key_fields": list(
            nvidia_a800_gemm_launch.GEMM_WORKLOAD_KEY_FIELDS),
        "entry_count": len(entries),
        "entries": entries,
    }
    physical = {}
    for semantic_key, entry in canonical["entries"].items():
        tile = entry["tile"]
        tile_values = (None, None, None) if tile is None else tuple(tile)
        physical[gemm_launch_selection_key_from_workload_key(semantic_key)] = (
            GemmLaunchSelection(
                *tile_values,
                entry["pipeline_depth"],
                entry["split_k"],
                entry["reduction"],
                None,
                None if tile is None else CublasCtaRasterSelection(
                    "cublaslt_m_group", 1, True, entry["split_k"]),
            )
        )
    return physical, workload_specs


def test_missing_nvidia_a800_gemm_lut_entry_fails_loudly(
        hardware, monkeypatch):
    canonical, _ = _nvidia_a800_exact_test_lut()
    monkeypatch.setattr(
        nvidia_a800_gemm_launch, "_EXPECTED_LAUNCH_ROW_COUNT",
        len(canonical))
    monkeypatch.setattr(
        nvidia_a800_gemm_launch, "load_gemm_launch_selection",
        lambda *args, **kwargs: canonical)
    nvidia_a800_gemm_launch._tiling_lut.cache_clear()
    try:
        with pytest.raises(KeyError, match="uncharacterized NVIDIA A800 GEMM"):
            select_launch_config(
                "nvidia_a800",
                "GEMM",
                {"b": 1, "m": 7, "n": 11, "k": 13},
                "bf16",
                "gemm",
                {"has_bias": False, "position": "qkv_proj",
                 "launch_policy": "injected_launch"},
                hardware=hardware["nvidia_a800"],
            )
    finally:
        nvidia_a800_gemm_launch._tiling_lut.cache_clear()


@pytest.mark.parametrize(
    "hardware_id",
    ("nvidia_a800", "metax_c500", "ascend_910b"),
)
def test_injected_gemm_miss_is_fail_closed_on_every_hardware(
        hardware, hardware_id):
    attributes = {
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "none",
        "has_bias": False,
        "launch_policy": "injected_launch",
    }

    with pytest.raises(KeyError, match="uncharacterized .* GEMM launch key"):
        select_launch_config(
            hardware_id,
            "GEMM",
            {"b": 1, "m": 7, "n": 11, "k": 13},
            "bf16",
            "gemm",
            attributes,
            hardware=hardware[hardware_id],
        )


def test_nvidia_a800_injected_lookup_uses_physical_domain(
        hardware, monkeypatch):
    canonical, workload_specs = _nvidia_a800_exact_test_lut()
    monkeypatch.setattr(
        nvidia_a800_gemm_launch, "_EXPECTED_LAUNCH_ROW_COUNT",
        len(canonical))
    monkeypatch.setattr(
        nvidia_a800_gemm_launch, "load_gemm_launch_selection",
        lambda *args, **kwargs: canonical)
    nvidia_a800_gemm_launch._tiling_lut.cache_clear()
    try:
        for variant, m, n, k, position, tp, ep in workload_specs:
            launch = nvidia_a800_gemm_launch.select_gemm_launch(
                {"b": 1, "m": m, "n": n, "k": k},
                "bf16",
                variant,
                {
                    "launch_policy": "injected_launch",
                    "a_layout": "row_major",
                    "b_layout": "row_major",
                    "c_layout": "row_major",
                    "trans_a": False,
                    "trans_b": True,
                    "epilogue": "none",
                    "has_bias": False,
                    "position": position,
                    "tensor_parallel_size": tp,
                    "expert_parallel_size": ep,
                },
                hardware["nvidia_a800"],
            )
            assert dict(launch.tiling.tile_shape) == {
                "m": 64, "n": 64, "k": 64}
            assert launch.tiling.pipeline_depth == 5
            assert launch.tiling.split_k == 1
            assert launch.algorithm == "cublaslt_injected_gemm"
            assert launch.grid == (
                ((m + 63) // 64) * ((n + 63) // 64), 1, 1)
            assert launch.block == (128, 1, 1)
    finally:
        nvidia_a800_gemm_launch._tiling_lut.cache_clear()


@pytest.mark.parametrize(
    ("tile_m", "tile_n", "expected_threads"),
    ((16, 16, 32), (128, 128, 128), (128, 256, 256), (256, 128, 256)),
)
def test_nvidia_a800_gemm_block_threads_follow_selected_tile(
        hardware, monkeypatch, tile_m, tile_n, expected_threads):
    shape = {"b": 1, "m": 3301, "n": 28672, "k": 4096}
    attributes = {
        "launch_policy": "injected_launch",
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "none",
        "has_bias": False,
        "position": "qkv_proj",
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
    }
    physical_key = gemm_launch_selection_key(
        shape, "bf16", "gemm", attributes)
    monkeypatch.setattr(
        nvidia_a800_gemm_launch,
        "_tiling_lut",
        lambda: {
            physical_key: GemmLaunchSelection(
                tile_m, tile_n, 64, 3, 1, "none", None,
                CublasCtaRasterSelection("cublaslt_m_group", 1, True, 1))
        },
    )

    launch = nvidia_a800_gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", attributes, hardware["nvidia_a800"])

    assert launch.block == (expected_threads, 1, 1)
    assert launch.grid == (
        ((shape["m"] + tile_m - 1) // tile_m)
        * ((shape["n"] + tile_n - 1) // tile_n),
        1,
        1,
    )


def test_nvidia_a800_observed_gemv_keeps_launch_without_invented_tile(
        hardware, monkeypatch):
    shape = {"b": 1, "m": 1, "n": 128, "k": 2048}
    attributes = {
        "launch_policy": "injected_launch",
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "none",
        "has_bias": False,
        "position": "qkv_proj",
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
    }
    workload_key = gemm_launch_selection_key(
        shape, "bf16", "gemm", attributes)
    injected = GemmLaunchSelection(
        tile_m=None,
        tile_n=None,
        tile_k=None,
        pipeline_depth=None,
        split_k=1,
        reduction="none",
    )
    monkeypatch.setattr(
        nvidia_a800_gemm_launch, "_tiling_lut",
        lambda: {workload_key: injected})

    launch = nvidia_a800_gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", attributes, hardware["nvidia_a800"])

    assert launch.execution_variant == "gemv"
    assert launch.tiling is None
    assert launch.algorithm == "cublaslt_injected_gemv"
    assert launch.grid == (128, 1, 1)


@pytest.mark.parametrize(
    ("reduction", "partial_element_bytes"),
    (("fp32_external", 4), ("output_dtype_external", 2)),
)
def test_nvidia_a800_split_rank_zero_uses_logical_main_and_reducer_geometry(
        hardware, monkeypatch, reduction, partial_element_bytes):
    canonical, workload_specs = _nvidia_a800_exact_test_lut()
    variant, m, n, k, position, tp, ep = workload_specs[0]
    attributes = {
        "launch_policy": "injected_launch",
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "none",
        "has_bias": False,
        "position": position,
        "tensor_parallel_size": tp,
        "expert_parallel_size": ep,
    }
    semantic_key = gemm_launch_selection_key(
        {"b": 1, "m": m, "n": n, "k": k}, "bf16", variant, attributes)
    current = canonical[semantic_key]
    canonical[semantic_key] = GemmLaunchSelection(
        current.tile_m, current.tile_n, current.tile_k,
        current.pipeline_depth, 3, reduction, None,
        CublasCtaRasterSelection("cublaslt_m_group", 1, True, 3))
    monkeypatch.setattr(
        nvidia_a800_gemm_launch, "_EXPECTED_LAUNCH_ROW_COUNT",
        len(canonical))
    monkeypatch.setattr(
        nvidia_a800_gemm_launch, "load_gemm_launch_selection",
        lambda *args, **kwargs: canonical)
    nvidia_a800_gemm_launch._tiling_lut.cache_clear()
    try:
        shape = {"b": 1, "m": m, "n": n, "k": k}
        main = nvidia_a800_gemm_launch.select_gemm_launch(
            shape, "bf16", variant, attributes, hardware["nvidia_a800"])
        reducer = _select_gemm_reducer_launch(
            "nvidia_a800", shape, "bf16", variant, attributes,
            hardware["nvidia_a800"])
    finally:
        nvidia_a800_gemm_launch._tiling_lut.cache_clear()

    assert main.algorithm == "cublaslt_injected_gemm"
    assert main.grid == (((m + 63) // 64) * ((n + 63) // 64), 1, 3)
    assert main.block == (128, 1, 1)
    assert main.tiling.split_k == 3
    assert main.parameters[
        "split_k_partial_bytes_per_element"] == partial_element_bytes
    assert reducer is not None
    assert reducer.algorithm == "logical_gemm_aux:gemm_reduce"
    if reduction == "output_dtype_external":
        assert reducer.grid == ((n + 31) // 32, (m + 15) // 16, 1)
        assert reducer.block == (32, 16, 1)
        output_warp_programs = m * ((n + 31) // 32)
        assert reducer.parameters[
            "source_bound_l2_read_transactions_per_warp"] == 6.0
        assert reducer.parameters[
            "source_bound_l2_read_warps_per_cta"] == float(min(m, 16))
        assert reducer.parameters["warp_instructions_lower_bound"] == float(
            output_warp_programs * 82)
    else:
        assert reducer.grid == (((m * n) + 255) // 256, 1, 1)
        assert reducer.block == (256, 1, 1)
    assert reducer.parameters[
        "split_k_partial_bytes_per_element"] == partial_element_bytes


@pytest.mark.parametrize(
    ("split_k", "expected_warp_instructions"),
    ((2, 70.0), (3, 82.0), (4, 76.0), (5, 79.0)),
)
def test_nvidia_a800_output_dtype_reducer_uses_exact_split_three_path(
        split_k, expected_warp_instructions):
    parameters = (
        nvidia_a800_gemm_launch.
        cublaslt_output_dtype_reducer_source_parameters(
            element_count=512,
            split_k=split_k,
            grid=(1, 1, 1),
            block=(32, 16, 1),
        ))

    assert parameters["warp_instructions_lower_bound"] == (
        16 * expected_warp_instructions)


def test_nvidia_a800_in_place_split_k_has_no_separate_reducer(
        hardware, monkeypatch):
    shape = {"b": 1, "m": 16, "n": 4096, "k": 4096}
    attributes = {
        "launch_policy": "injected_launch",
        "a_layout": "row_major",
        "b_layout": "row_major",
        "c_layout": "row_major",
        "trans_a": False,
        "trans_b": True,
        "epilogue": "none",
        "has_bias": False,
        "position": "qkv_proj",
        "tensor_parallel_size": 1,
        "expert_parallel_size": 1,
    }
    workload_key = gemm_launch_selection_key(
        shape, "bf16", "gemm", attributes)
    injected = GemmLaunchSelection(
        tile_m=64,
        tile_n=128,
        tile_k=64,
        pipeline_depth=3,
        split_k=2,
        reduction="in_place",
        cta_raster=CublasCtaRasterSelection(
            "cublaslt_m_group", 1, True, 2),
    )
    monkeypatch.setattr(
        nvidia_a800_gemm_launch, "_tiling_lut",
        lambda: {workload_key: injected})

    main = nvidia_a800_gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", attributes, hardware["nvidia_a800"])
    reducer = _select_gemm_reducer_launch(
        "nvidia_a800", shape, "bf16", "gemm", attributes,
        hardware["nvidia_a800"])

    assert main.tiling.split_k == 2
    assert main.parameters["split_k_partial_bytes_per_element"] == 0.0
    assert reducer is None


@pytest.mark.parametrize("legacy_policy", ("measured", "analytical"))
def test_legacy_launch_policy_names_fail_loudly(hardware, legacy_policy):
    with pytest.raises(ValueError, match="unknown launch_policy"):
        select_launch_config(
            "nvidia_a800",
            "GEMM",
            {"b": 1, "m": 128, "n": 4096, "k": 4096},
            "bf16",
            "gemm",
            {"has_bias": False, "position": "generic",
             "launch_policy": legacy_policy},
            hardware=hardware["nvidia_a800"],
        )


def test_missing_metax_c500_gemm_lut_entry_fails_loudly(hardware):
    with pytest.raises(KeyError, match="uncharacterized MetaX C500 GEMM"):
        select_launch_config(
            "metax_c500",
            "GEMM",
            {"b": 1, "m": 7, "n": 11, "k": 13},
            "bf16",
            "gemm",
            {"has_bias": False, "position": "generic",
             "launch_policy": "injected_launch"},
            hardware=hardware["metax_c500"],
        )


def test_metax_c500_analytical_gemm_declares_supported_tile_order(hardware):
    launch = select_launch_config(
        "metax_c500",
        "GEMM",
        {"b": 1, "m": 8192, "n": 8192, "k": 8192},
        "bf16",
        "gemm",
        {"has_bias": False, "position": "generic",
         "launch_policy": "analytical_launch"},
        hardware=hardware["metax_c500"],
    )

    assert launch.grid == (64, 64, 1)
    assert launch.parameters["n_major_traffic"] == 0.0


def test_metax_c500_analytical_launch_does_not_load_injected_lut(
        hardware, monkeypatch):
    def fail_if_loaded():
        raise AssertionError("analytical_launch loaded the injected tile LUT")

    monkeypatch.setattr(metax_c500_gemm_launch, "_data", fail_if_loaded)
    launch = select_launch_config(
        "metax_c500",
        "GEMM",
        {"b": 1, "m": 111, "n": 6144, "k": 2560},
        "bf16",
        "gemm",
        {"has_bias": False, "position": "generic",
         "launch_policy": "analytical_launch"},
        hardware=hardware["metax_c500"],
    )

    assert dict(launch.tiling.tile_shape) == {"m": 128, "n": 128, "k": 64}


def test_metax_c500_source_flash_attn_supports_d64_trait(hardware):
    launch = select_launch_config(
        "metax_c500",
        "FLASH_ATTN",
        {
            "batch": 1,
            "q_len": 1,
            "kv_len": 128,
            "head_dim": 64,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "causal": False,
            "kv_layout": "contiguous",
            "kv_dist": "uniform",
            "q_lens": (1,),
            "kv_lens": (128,),
            "launch_policy": "injected_launch",
            "split_policy": "serving",
        },
        hardware=hardware["metax_c500"],
    )

    assert launch.grid == (1, 8, 1)
    assert launch.block == (64, 1, 1)
    assert dict(launch.tiling.tile_shape) == {
        "q": 16, "kv": 16, "head": 64,
        "query_heads_per_workgroup": 4}


def test_metax_c500_binary_d192_prefill_trait(hardware):
    launch = select_launch_config(
        "metax_c500",
        "FLASH_ATTN",
        {
            "batch": 1,
            "q_len": 128,
            "kv_len": 256,
            "head_dim": 192,
            "num_query_heads": 16,
            "num_kv_heads": 16,
        },
        "bf16",
        "prefill",
        {
            "is_decode": False,
            "causal": True,
            "kv_layout": "contiguous",
            "kv_dist": "uniform",
            "q_lens": (128,),
            "kv_lens": (256,),
            "launch_policy": "injected_launch",
            "split_policy": "serving",
        },
        hardware=hardware["metax_c500"],
    )

    assert launch.block == (512, 1, 1)
    assert dict(launch.tiling.tile_shape) == {
        "q": 128, "kv": 64, "head": 192}
    assert launch.tiling.num_kv_splits == 1


def test_metax_c500_binary_get_tile_size_d192_decode(hardware):
    launch = select_launch_config(
        "metax_c500",
        "FLASH_ATTN",
        {
            "batch": 1,
            "q_len": 1,
            "kv_len": 256,
            "head_dim": 192,
            "num_query_heads": 16,
            "num_kv_heads": 4,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "causal": False,
            "kv_layout": "contiguous",
            "kv_dist": "uniform",
            "q_lens": (1,),
            "kv_lens": (256,),
            "launch_policy": "injected_launch",
            "split_policy": "forced",
            "num_kv_splits": 1,
        },
        hardware=hardware["metax_c500"],
    )

    assert dict(launch.tiling.tile_shape) == {
        "q": 64, "kv": 64, "head": 192}
    assert launch.block == (256, 1, 1)


@pytest.mark.parametrize(
    ("kv_layout", "expected_algorithm", "expected_grid", "expected_tile",
     "expected_page_table_bytes"),
    (
        ("contiguous", "metax_flash_attention_trait_direct:prefill",
         (32, 3, 1),
         {"q": 128, "kv": 64, "head": 128}, 0.0),
        ("paged_varlen", "metax_flash_attention_trait_direct:prefill",
         (32, 2, 3),
         {"q": 64, "kv": 64, "head": 128}, 0.0),
        ("chunked_prefill",
         "metax_flash_attention_trait_packed:prefill_q64_full_tile",
         (32, 2, 3),
         {"q": 64, "kv": 64, "head": 128,
          "query_heads_per_workgroup": 1}, 0.0),
    ),
)
def test_metax_c500_source_prefill_uses_xcore1000_d128_dispatch(
        hardware, kv_layout, expected_algorithm, expected_grid, expected_tile,
        expected_page_table_bytes):
    launch = select_launch_config(
        "metax_c500",
        "FLASH_ATTN",
        {
            "batch": 3,
            "q_len": 128,
            "kv_len": 256,
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "prefill",
        {
            "is_decode": False,
            "causal": True,
            "kv_layout": kv_layout,
            "kv_dist": "uniform",
            "q_lens": (128, 96, 64),
            "kv_lens": (256, 192, 128),
            "launch_policy": "injected_launch",
            "split_policy": "serving",
        },
        hardware=hardware["metax_c500"],
    )

    assert launch.algorithm == expected_algorithm
    assert launch.grid == expected_grid
    assert launch.block == (256, 1, 1)
    assert dict(launch.tiling.tile_shape) == expected_tile
    assert launch.tiling.num_kv_splits == 1
    assert launch.parameters["unpadded_flops"] == 0.0
    assert launch.parameters["page_table_bytes_per_kv_panel"] == (
        expected_page_table_bytes)


def test_metax_c500_source_paged_decode_uses_runtime_unsplit_dispatch(hardware):
    launch = select_launch_config(
        "metax_c500",
        "FLASH_ATTN",
        {
            "batch": 1,
            "q_len": 1,
            "kv_len": 129,
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "causal": True,
            "kv_layout": "paged_varlen",
            "kv_dist": "uniform",
            "q_lens": (1,),
            "kv_lens": (129,),
            "launch_policy": "injected_launch",
            "split_policy": "serving",
        },
        hardware=hardware["metax_c500"],
    )

    assert launch.grid == (8, 1, 1)
    assert launch.tiling.num_kv_splits == 1
    assert dict(launch.tiling.tile_shape) == {
        "q": 16, "kv": 16, "head": 128,
        "query_heads_per_workgroup": 4}
    assert launch.parameters[
        "split_scratch_dram_bytes_per_element"] == 0.0
    assert launch.parameters["page_table_bytes_per_kv_panel"] == 0.0


def test_metax_c500_source_decode_uses_unsplit_grid_at_full_ap_load(hardware):
    launch = select_launch_config(
        "metax_c500",
        "FLASH_ATTN",
        {
            "batch": 96,
            "q_len": 1,
            "kv_len": 2048,
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "causal": True,
            "kv_layout": "paged_varlen",
            "kv_dist": "uniform",
            "q_lens": (1,) * 96,
            "kv_lens": (2048,) * 96,
            "launch_policy": "injected_launch",
            "split_policy": "serving",
        },
        hardware=hardware["metax_c500"],
    )

    assert launch.tiling.num_kv_splits == 1
    assert launch.grid == (8, 96, 1)
    assert launch.parameters[
        "split_scratch_dram_bytes_per_element"] == 0.0
    assert launch.parameters["page_table_bytes_per_kv_panel"] == 0.0


def test_metax_c500_full_graph_decode_uses_native_fa2_splitkv_heuristic(
        hardware):
    """Graph replay keeps FA2's source-selected SplitKV body and combine."""

    shape = {
        "batch": 56,
        "q_len": 1,
        "kv_len": 839,
        "head_dim": 128,
        "num_query_heads": 32,
        "num_kv_heads": 8,
    }
    attributes = {
        "is_decode": True,
        "causal": True,
        "kv_layout": "paged_varlen",
        "kv_dist": "skew",
        "q_lens": (1,) * 56,
        "kv_lens": (
            745, 23, 123, 71, 33, 69, 48, 88, 677, 236, 15, 386, 408,
            24, 750, 252, 27, 409, 331, 67, 418, 94, 186, 31, 33, 193,
            512, 254, 389, 839, 390, 9, 683, 29, 745, 388, 61, 335, 14,
            40, 164, 791, 32, 60, 26, 337, 60, 275, 7, 50, 159, 307,
            773, 316, 21, 76,
        ),
        "launch_policy": "injected_launch",
        "split_policy": "serving",
        "graph_runtime_mode": "FULL",
    }
    main = select_launch_config(
        "metax_c500", "FLASH_ATTN", shape, "bf16", "decode",
        attributes, hardware=hardware["metax_c500"])
    combine_attributes = dict(attributes)
    combine_attributes["num_kv_splits"] = main.tiling.num_kv_splits
    combine = select_launch_config(
        "metax_c500", "FLASH_ATTN", shape, "bf16", "combine",
        combine_attributes, hardware=hardware["metax_c500"])

    assert main.tiling.num_kv_splits == 5
    assert main.grid == (1, 448, 5)
    assert main.block == (64, 1, 1)
    assert combine.grid == (224, 1, 1)
    assert combine.block == (256, 1, 1)


def test_metax_c500_source_contiguous_serving_decode_is_direct(hardware):
    launch = select_launch_config(
        "metax_c500",
        "FLASH_ATTN",
        {
            "batch": 2,
            "q_len": 1,
            "kv_len": 8192,
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "causal": False,
            "kv_layout": "contiguous",
            "kv_dist": "uniform",
            "q_lens": (1, 1),
            "kv_lens": (8192, 8192),
            "launch_policy": "injected_launch",
            "split_policy": "serving",
        },
        hardware=hardware["metax_c500"],
    )

    assert launch.tiling.num_kv_splits == 1
    assert launch.grid == (8, 2, 1)
    assert launch.parameters["page_table_bytes_per_kv_panel"] == 0.0


def test_metax_c500_source_decode_split_count_is_grid_y(hardware):
    shape = {
        "batch": 2,
        "q_len": 1,
        "kv_len": 4096,
        "head_dim": 128,
        "num_query_heads": 32,
        "num_kv_heads": 8,
    }
    attributes = {
        "is_decode": True,
        "causal": True,
        "kv_layout": "contiguous",
        "kv_dist": "skew",
        "q_lens": (1, 1),
        "kv_lens": (2048, 4096),
        "launch_policy": "injected_launch",
        "split_policy": "forced",
        "num_kv_splits": 3,
    }
    main = select_launch_config(
        "metax_c500", "FLASH_ATTN", shape, "bf16", "decode",
        attributes, hardware=hardware["metax_c500"])
    combine = select_launch_config(
        "metax_c500", "FLASH_ATTN", shape, "bf16", "combine",
        attributes, hardware=hardware["metax_c500"])

    assert main.grid == (1, 16, 3)
    assert main.tiling.num_kv_splits == 3
    assert combine.algorithm == "metax_flash_attn_lut"
    assert combine.grid == (8, 1, 1)
    assert combine.block == (256, 1, 1)
    assert combine.tiling.num_kv_splits == 3
    assert combine.parameters["page_table_bytes_per_kv_panel"] == 0.0


def test_metax_c500_source_decode_combine_selects_block_m_16(hardware):
    shape = {
        "batch": 33,
        "q_len": 1,
        "kv_len": 1024,
        "head_dim": 128,
        "num_query_heads": 128,
        "num_kv_heads": 8,
    }
    attributes = {
        "is_decode": True,
        "causal": True,
        "kv_layout": "paged",
        "kv_dist": "uniform",
        "q_lens": (1,) * 33,
        "kv_lens": (1024,) * 33,
        "launch_policy": "injected_launch",
        "split_policy": "forced",
        "num_kv_splits": 3,
    }
    main = select_launch_config(
        "metax_c500", "FLASH_ATTN", shape, "bf16", "decode",
        attributes, hardware=hardware["metax_c500"])
    combine_attributes = dict(attributes)
    combine_attributes["num_kv_splits"] = main.tiling.num_kv_splits
    combine = select_launch_config(
        "metax_c500", "FLASH_ATTN", shape, "bf16", "combine",
        combine_attributes, hardware=hardware["metax_c500"])

    assert main.tiling.num_kv_splits == 3
    assert combine.grid == (264, 1, 1)


def test_metax_c500_flash_attn_unknown_policy_fails_loudly(hardware):
    with pytest.raises(ValueError, match="unknown launch_policy"):
        select_launch_config(
            "metax_c500",
            "FLASH_ATTN",
            {
                "batch": 1,
                "q_len": 1,
                "kv_len": 8,
                "head_dim": 128,
                "num_query_heads": 32,
                "num_kv_heads": 8,
            },
            "bf16",
            "decode",
            {
                "is_decode": True,
                "causal": True,
                "kv_layout": "contiguous",
                "kv_dist": "uniform",
                "q_lens": (1,),
                "kv_lens": (8,),
                "launch_policy": "unknown",
                "split_policy": "serving",
            },
            hardware=hardware["metax_c500"],
        )


def test_metax_c500_analytical_vector_gather_is_shape_derived(hardware):
    launch = select_launch_config(
        "metax_c500", "VECTOR",
        {"element_count": 257, "rows": 1, "hidden": 257},
        "bf16", "gather",
        {
            "gather_factor": 1,
            "index_bytes": 8,
            "index_count": 1,
            "launch_policy": "analytical_launch",
            "scattered": True,
        },
        hardware=hardware["metax_c500"],
    )

    assert launch.grid == (2, 1, 1)
    assert launch.block == (256, 1, 1)
    assert launch.parameters == {}


@pytest.mark.parametrize(
    ("rows", "hidden", "expected_algorithm", "expected_grid"),
    (
        (1, 2048, "aten_index_select_small_index", (16, 1, 1)),
        (16, 5120, "aten_index_select_small_index", (40, 1, 1)),
        (17, 4096, "aten_index_select_large_index", (544, 1, 1)),
        (8192, 2560, "aten_index_select_large_index", (6656, 1, 1)),
    ),
)
def test_metax_c500_injected_embedding_gather_uses_source_bound_index_select(
        hardware, rows, hidden, expected_algorithm, expected_grid):
    launch = select_launch_config(
        "metax_c500", "VECTOR",
        {"element_count": rows * hidden, "rows": rows, "hidden": hidden},
        "bf16", "gather",
        {
            "gather_factor": 1,
            "index_bytes": 8,
            "index_count": rows,
            "launch_policy": "injected_launch",
            "scattered": True,
        },
        hardware=hardware["metax_c500"],
    )

    assert launch.algorithm == expected_algorithm
    assert launch.grid == expected_grid
    assert launch.block == (128, 1, 1)


def test_metax_c500_injected_embedding_gather_rejects_other_semantics(hardware):
    with pytest.raises(NotImplementedError, match="token-embedding index_select"):
        select_launch_config(
            "metax_c500", "VECTOR",
            {"element_count": 2560, "rows": 1, "hidden": 2560},
            "bf16", "gather",
            {
                "gather_factor": 2,
                "index_bytes": 8,
                "index_count": 1,
                "launch_policy": "injected_launch",
                "scattered": True,
            },
            hardware=hardware["metax_c500"],
        )


def test_metax_c500_injected_gather_rejects_rows_beyond_the_proven_cap(
        hardware):
    with pytest.raises(NotImplementedError, match="row domain"):
        select_launch_config(
            "metax_c500", "VECTOR",
            {"element_count": 8193 * 2560, "rows": 8193, "hidden": 2560},
            "bf16", "gather",
            {
                "gather_factor": 1,
                "index_bytes": 8,
                "index_count": 8193,
                "launch_policy": "injected_launch",
                "scattered": True,
            },
            hardware=hardware["metax_c500"],
        )


def test_metax_c500_injected_gather_covers_any_slice_width(hardware):
    # The pinned binary proof records that host launch construction has no
    # hidden-specific branch: hidden enters only through sliceSize (small) and
    # rows * hidden (large).  A rope cos/sin table is therefore resolved by the
    # same geometry as a token embedding.
    launch = select_launch_config(
        "metax_c500", "VECTOR",
        {"element_count": 8 * 128, "rows": 8, "hidden": 128},
        "bf16", "gather",
        {
            "gather_factor": 1,
            "index_bytes": 8,
            "index_count": 8,
            "launch_policy": "injected_launch",
            "scattered": True,
        },
        hardware=hardware["metax_c500"],
    )
    assert launch.algorithm == "aten_index_select_small_index"
    assert launch.grid == (1, 1, 1)
    assert launch.block == (128, 1, 1)


def test_ascend_910b_non_gemm_policies_select_same_source_launch(hardware):
    attributes = {
        "is_decode": False,
        "causal": True,
        "kv_layout": "paged_varlen",
        "kv_dist": "uniform",
        "q_lens": (721,),
        "kv_lens": (721,),
    }
    flash_shape = {
        "batch": 1,
        "q_len": 721,
        "kv_len": 721,
        "head_dim": 128,
        "num_query_heads": 32,
        "num_kv_heads": 8,
    }
    vector_shape = {
        "tok": 721,
        "intermediate": 14336,
        "element_count": 10_336_256,
    }

    for kind, shape, variant, base_attributes in (
            ("FLASH_ATTN", flash_shape, "prefill", attributes),
            ("VECTOR", vector_shape, "silu", {})):
        launches = tuple(
            select_launch_config(
                "ascend_910b", kind, shape, "bf16", variant,
                {**base_attributes, "launch_policy": policy},
                hardware=hardware["ascend_910b"],
            )
            for policy in ("analytical_launch", "injected_launch")
        )
        assert launches[0] == launches[1]


def test_ascend_910b_flash_attn_does_not_publish_unproven_tiles(hardware):
    launch = select_launch_config(
        "ascend_910b",
        "FLASH_ATTN",
        {
            "batch": 1,
            "q_len": 1,
            "kv_len": 4096,
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "causal": False,
            "kv_layout": "contiguous",
            "kv_dist": "skew",
            "q_lens": (1,),
            "kv_lens": (4096,),
        },
        hardware=hardware["ascend_910b"],
    )

    assert launch.grid == (8, 1, 1)
    assert launch.block == (1, 1, 1)
    assert launch.algorithm == (
        "cann_flash_attention_score:103000000010422221")
    assert launch.execution_variant == "decode"
    assert dict(launch.tiling.tile_shape) == {}
    assert launch.tiling.num_kv_splits == 1
    assert launch.parameters["unpadded_flops"] == 1.0


@pytest.mark.parametrize(
    ("kv_layout", "runtime_variant"),
    (
        ("contiguous", "5000000000000200203"),
        ("paged", "5000000000010200203"),
    ),
)
def test_ascend_910b_split_fuse_publishes_source_tile_marker(
        hardware, kv_layout, runtime_variant):
    launch = select_launch_config(
        "ascend_910b",
        "FLASH_ATTN",
        {
            "batch": 1,
            "q_len": 512,
            "kv_len": 512,
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "prefill",
        {
            "is_decode": False,
            "causal": True,
            "kv_layout": kv_layout,
            "kv_dist": "skew",
            "q_lens": (512,),
            "kv_lens": (512,),
        },
        hardware=hardware["ascend_910b"],
    )

    assert launch.grid == (24, 1, 1)
    assert launch.block == (1, 1, 1)
    assert launch.algorithm == f"cann_flash_attention_score:{runtime_variant}"
    assert dict(launch.tiling.tile_shape) == {
        "q": 128,
        "kv_stack_ceiling": 512,
    }
    assert launch.tiling.num_kv_splits == 1
    assert launch.parameters["unpadded_flops"] == 1.0


@pytest.mark.parametrize(
    ("tokens", "topk", "expected_tile_m"),
    (
        (1, 2, 16),
        (10, 2, 32),
        (20, 2, 48),
        (16, 4, 64),
        (17, 4, 80),
        (64, 2, 128),
        (1024, 8, 128),
    ),
)
@pytest.mark.parametrize("projection", ("w1", "w2"))
def test_ascend_910b_grouped_matmul_publishes_cann_base_tiles(
        hardware, tokens, topk, expected_tile_m, projection):
    launch = select_launch_config(
        "ascend_910b",
        "FUSED_MOE",
        {
            "tokens": tokens,
            "experts": 8,
            "topk": topk,
            "hidden": 2048,
            "intermediate": 768,
        },
        "bf16",
        projection,
        {"launch_policy": "analytical_launch", "expert_parallel_size": 1},
        hardware=hardware["ascend_910b"],
    )
    injected = select_launch_config(
        "ascend_910b",
        "FUSED_MOE",
        {
            "tokens": tokens,
            "experts": 8,
            "topk": topk,
            "hidden": 2048,
            "intermediate": 768,
        },
        "bf16",
        projection,
        {"launch_policy": "injected_launch", "expert_parallel_size": 1},
        hardware=hardware["ascend_910b"],
    )

    assert launch == injected
    assert launch.grid == (24, 1, 1)
    assert launch.block == (1, 1, 1)
    assert launch.algorithm == f"cann_grouped_matmul_{projection}"
    assert dict(launch.tiling.tile_shape) == {
        "m": expected_tile_m,
        "n": 256,
        "k": 64,
    }
    program_names = (
        "cann_depth_a1", "cann_depth_b1",
        "cann_step_ka", "cann_step_kb",
        "cann_step_m", "cann_step_n",
        "cann_db_l0a", "cann_db_l0b", "cann_db_l0c",
    )
    expected_programs = {
        16: (120, 8, 60, 4, 1, 1, 2, 2, 1),
        32: (56, 8, 28, 4, 1, 1, 2, 2, 1),
        64: (24, 8, 12, 4, 1, 1, 2, 2, 1),
        80: (24, 8, 12, 4, 1, 1, 2, 2, 1),
        128: (8, 8, 4, 4, 1, 1, 2, 2, 1),
    }
    if expected_tile_m == 48:
        assert tuple(
            launch.parameters[name] for name in program_names
        ) == (40.0, 8.0, 20.0, 4.0, 1.0, 1.0, 2.0, 2.0, 1.0)
    elif expected_tile_m in expected_programs:
        assert tuple(
            launch.parameters[name] for name in program_names
        ) == expected_programs[expected_tile_m]
    else:
        assert not any(name in launch.parameters for name in program_names)


def test_ascend_910b_grouped_matmul_uses_rank_local_expert_rows(hardware):
    shape = {
        "tokens": 64,
        "experts": 128,
        "topk": 2,
        "hidden": 2048,
        "intermediate": 768,
    }
    launch = select_launch_config(
        "ascend_910b",
        "FUSED_MOE",
        shape,
        "bf16",
        "w1",
        {
            "launch_policy": "analytical_launch",
            "expert_parallel_size": 2,
            "local_tokens_per_expert": (20, 12) + (0,) * 126,
        },
        hardware=hardware["ascend_910b"],
    )

    assert dict(launch.tiling.tile_shape) == {
        "m": 32,
        "n": 256,
        "k": 64,
    }
    assert launch.parameters["expert_parallel_size"] == 2.0


def test_ascend_910b_grouped_matmul_rejects_invalid_rank_local_counts(
        hardware):
    with pytest.raises(ValueError, match="127 entries, expected 128"):
        select_launch_config(
            "ascend_910b",
            "FUSED_MOE",
            {
                "tokens": 64,
                "experts": 128,
                "topk": 2,
                "hidden": 2048,
                "intermediate": 768,
            },
            "bf16",
            "w1",
            {
                "launch_policy": "analytical_launch",
                "expert_parallel_size": 2,
                "local_tokens_per_expert": (1,) * 127,
            },
            hardware=hardware["ascend_910b"],
        )


def test_ascend_910b_grouped_matmul_zero_local_rows_remains_unsupported(
        hardware):
    with pytest.raises(
            NotImplementedError, match="legal base-M is not source-backed"):
        select_launch_config(
            "ascend_910b",
            "FUSED_MOE",
            {
                "tokens": 1,
                "experts": 128,
                "topk": 1,
                "hidden": 2048,
                "intermediate": 768,
            },
            "bf16",
            "w1",
            {
                "launch_policy": "analytical_launch",
                "expert_parallel_size": 2,
                "local_tokens_per_expert": (0,) * 128,
            },
            hardware=hardware["ascend_910b"],
        )


def test_ascend_910b_decode_keeps_causal_semantics_out_of_launch_selection(
        hardware):
    attributes = {
        "is_decode": True,
        "causal": True,
        "kv_layout": "contiguous",
        "kv_dist": "skew",
        "q_lens": (1,),
        "kv_lens": (4096,),
    }
    launch = select_launch_config(
        "ascend_910b",
        "FLASH_ATTN",
        {
            "batch": 1,
            "q_len": 1,
            "kv_len": 4096,
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        attributes,
        hardware=hardware["ascend_910b"],
    )

    assert launch.algorithm == (
        "cann_flash_attention_score:103000000010422221")
    assert attributes["causal"] is True


def test_ascend_910b_flash_attn_source_rule_accepts_new_lengths(hardware):
    launch = select_launch_config(
        "ascend_910b",
        "FLASH_ATTN",
        {
            "batch": 2,
            "q_len": 1,
            "kv_len": 4096,
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "causal": True,
            "kv_layout": "paged_varlen",
            "kv_dist": "skew",
            "q_lens": (1, 1),
            "kv_lens": (2048, 4096),
        },
        hardware=hardware["ascend_910b"],
    )

    assert launch.algorithm == (
        "cann_flash_attention_score:103000000010622221")
    assert launch.grid == (12, 1, 1)


def _cann_fia_replay_abi_program(hardware, q_lens, kv_lens, **extra):
    """Select one 910B FIA decode launch and return its typed program."""

    launch = select_launch_config(
        "ascend_910b",
        "FLASH_ATTN",
        {
            "batch": len(q_lens),
            "q_len": 1,
            "kv_len": max(kv_lens),
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "causal": True,
            "kv_layout": "paged_varlen",
            "kv_dist": "skew",
            "q_lens": q_lens,
            "kv_lens": kv_lens,
            **extra,
        },
        hardware=hardware["ascend_910b"],
    )
    assert isinstance(launch.source_program, FiaCrossDomainLaunchProgram)
    return launch, launch.source_program


def _cann_fia_task_keys(program):
    return [
        tuple(
            (task.b_n2, task.g_s1, task.s2, task.actual_m, task.actual_s2,
             task.is_head_first, task.is_head_last, task.is_s2_split_core)
            for task in group.tasks)
        for group in program.aic_groups]


def test_ascend_910b_padded_replay_split_keeps_exact_real_grouping(
        hardware):
    """The replay-time tiler consumes zero-KV padded slots, not dummy work.

    vLLM-Ascend re-derives the FIA split at every FULL-graph replay from the
    runtime ``seq_lens_list`` whose padded slots are zero-filled
    (``model_runner_v1.py:732-734``).  With the logical lengths published the
    real requests must keep their exact-side greedy grouping and the dummy
    slots must only append zero-cost tail tasks.
    """

    real_kv_lens = tuple((100 + 37 * index) % 1900 + 60 for index in range(28))
    exact_launch, exact_program = _cann_fia_replay_abi_program(
        hardware, (1,) * 28, real_kv_lens)
    exact_groups = _cann_fia_task_keys(exact_program)
    exact_keys = {key for group in exact_groups for key in group}

    _, padded_program = _cann_fia_replay_abi_program(
        hardware, (1,) * 32, real_kv_lens + (1,) * 4,
        graph_logical_q_lens=(1,) * 28,
        graph_logical_kv_lens=real_kv_lens)
    padded_groups = _cann_fia_task_keys(padded_program)

    assert len(padded_groups) == len(exact_groups)
    for padded_group, exact_group in zip(padded_groups, exact_groups):
        assert tuple(
            key for key in padded_group if key in exact_keys) == exact_group
    assert sum(
        1 for group in padded_groups for key in group
        if key not in exact_keys) == 32


def test_ascend_910b_padded_replay_split_without_logical_keys_regroups(
        hardware):
    """Without the logical lengths the split prices dummy KV work.

    This pins the pre-fix semantics: model-input slot arrays with dummy
    KV=1 entries perturb the greedy ``core_limit`` division and re-group
    real requests, which is exactly the negative-discount artifact the
    replay ABI fix removes.
    """

    real_kv_lens = tuple((100 + 37 * index) % 1900 + 60 for index in range(28))
    _, exact_program = _cann_fia_replay_abi_program(
        hardware, (1,) * 28, real_kv_lens)
    exact_groups = _cann_fia_task_keys(exact_program)

    _, padded_program = _cann_fia_replay_abi_program(
        hardware, (1,) * 32, real_kv_lens + (1,) * 4)
    padded_groups = _cann_fia_task_keys(padded_program)
    exact_keys = {key for group in exact_groups for key in group}

    regrouped = any(
        tuple(key for key in padded_group if key in exact_keys)
        != exact_group
        for padded_group, exact_group in zip(
            padded_groups, exact_groups))
    assert regrouped


def test_ascend_910b_padded_replay_logical_keys_are_inert_when_exact(
        hardware):
    """Logical lengths equal to the slot arrays leave the launch unchanged."""

    real_kv_lens = (2048, 4096)
    baseline_launch, baseline_program = _cann_fia_replay_abi_program(
        hardware, (1, 1), real_kv_lens)
    keyed_launch, keyed_program = _cann_fia_replay_abi_program(
        hardware, (1, 1), real_kv_lens,
        graph_logical_q_lens=(1, 1),
        graph_logical_kv_lens=real_kv_lens)

    assert keyed_launch.algorithm == baseline_launch.algorithm
    assert keyed_launch.grid == baseline_launch.grid
    assert keyed_launch.block == baseline_launch.block
    assert keyed_launch.parameters == baseline_launch.parameters
    assert _cann_fia_task_keys(keyed_program) == (
        _cann_fia_task_keys(baseline_program))


@pytest.mark.parametrize(
    ("attributes_extra", "match"),
    (
        (
            {"graph_logical_q_lens": (1,) * 28},
            "must be complete",
        ),
        (
            {
                "graph_logical_q_lens": (1,) * 33,
                "graph_logical_kv_lens": (100,) * 33,
            },
            "disagree with the captured slot arrays",
        ),
        (
            {
                "graph_logical_q_lens": (1,) * 28,
                "graph_logical_kv_lens": (2048,) + ((100,) * 27),
            },
            "not the appending slot shape",
        ),
    ),
)
def test_ascend_910b_padded_replay_logical_keys_fail_closed(
        hardware, attributes_extra, match):
    """Partial, mismatched, or non-appending logical fields fail closed."""

    real_kv_lens = tuple((100 + 37 * index) % 1900 + 60 for index in range(28))
    with pytest.raises(ValueError, match=match):
        _cann_fia_replay_abi_program(
            hardware, (1,) * 32, real_kv_lens + (1,) * 4, **attributes_extra)


@pytest.mark.parametrize(
    ("hardware_id", "variant", "q_len", "kv_len", "expected_grid"),
    (
        ("nvidia_a800", "prefill", 64, 512, (1, 2, 32)),
        ("nvidia_a800", "decode", 1, 8, (1, 2, 8)),
    ),
)
def test_gpu_flash_attn_grid_uses_variant_head_axis(
        hardware, hardware_id, variant, q_len, kv_len, expected_grid):
    launch = select_launch_config(
        hardware_id,
        "FLASH_ATTN",
        {
            "batch": 2,
            "q_len": q_len,
            "kv_len": kv_len,
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        variant,
        {
            "is_decode": variant == "decode",
            "kv_layout": "contiguous",
            "launch_policy": "analytical_launch",
            "split_policy": "serving",
        },
        hardware=hardware[hardware_id],
    )

    assert launch.grid == expected_grid
    if hardware_id == "nvidia_a800":
        expected_smem_ratio = 0.125
        assert launch.algorithm == "flash_attn_dense"
        assert dict(launch.tiling.tile_shape) == {
            "q": 128, "kv": 64, "head": 128}
    else:
        raise AssertionError("unexpected FlashAttention hardware fixture")
    assert launch.parameters["mma_smem_bytes_per_flop"] == expected_smem_ratio


@pytest.mark.parametrize(
    ("variant", "shape", "q_lens", "kv_lens", "algorithm", "grid", "splits"),
    (
        (
            "mla_prefill",
            {
                "batch": 1, "q_len": 3, "kv_len": 3,
                "num_query_heads": 64, "qk_head_dim": 192,
                "v_head_dim": 128, "kv_lora_rank": 512,
            },
            (3,),
            (3,),
            "vllm_flash_attn_v2_mla_prefill",
            (1, 1, 64),
            1,
        ),
        (
            "mla_decode",
            {
                "batch": 1, "q_len": 1, "kv_len": 4096,
                "num_query_heads": 64, "qk_head_dim": 576,
                "v_head_dim": 512, "kv_lora_rank": 512,
            },
            (1,),
            (4096,),
            "vllm_triton_grouped_mla_decode",
            (1, 4, 8),
            8,
        ),
    ),
)
def test_nvidia_a800_kimi_mla_launch_matches_pinned_vllm_source(
        hardware, variant, shape, q_lens, kv_lens, algorithm, grid, splits):
    launch = select_launch_config(
        "nvidia_a800",
        KernelKind.FLASH_ATTN,
        shape,
        "bf16",
        variant,
        {
            "causal": True,
            "kv_layout": "mla",
            "kv_dist": "uniform",
            "q_lens": q_lens,
            "kv_lens": kv_lens,
        },
        hardware=hardware["nvidia_a800"],
    )

    assert launch.algorithm == algorithm
    assert launch.grid == grid
    assert launch.tiling.num_kv_splits == splits


def test_nvidia_a800_kimi_int4_moe_launch_uses_marlin_block_rule(hardware):
    counts = (1,) * 8 + (0,) * (384 - 8)
    launch = select_launch_config(
        "nvidia_a800",
        KernelKind.FUSED_MOE,
        {
            "tokens": 1,
            "experts": 384,
            "topk": 8,
            "hidden": 7168,
            "intermediate": 2048,
        },
        "bf16",
        "routed_experts",
        {
            "tokens_per_expert": counts,
            "expert_parallel_size": 1,
            "routed_weight_dtype": "int4",
            "weight_group_size": 32,
        },
        hardware=hardware["nvidia_a800"],
    )

    assert launch.algorithm == "vllm_marlin_moe_w4a16_aggregate"
    assert dict(launch.tiling.tile_shape) == {"m": 8, "n": 128, "k": 128}
    assert launch.parameters["padded_rows"] == 64.0
    assert launch.grid == (704, 1, 1)


@pytest.mark.parametrize(
    ("kv_layout", "batch", "q_len", "kv_len", "head_dim",
     "expected_algorithm", "expected_tile", "expected_smem",
     "expected_splits"),
    (
        ("contiguous", 32, 1, 512, 128, "flash_attn_dense",
         {"q": 128, "kv": 64, "head": 128}, 64 * 1024, 1),
        ("contiguous", 1, 1, 1024, 128, "flash_attn_splitkv",
         {"q": 64, "kv": 128, "head": 128}, 80 * 1024, 8),
        ("paged", 1, 1, 512, 128, "vllm_flash_attn_splitkv",
         {"q": 64, "kv": 128, "head": 128}, 80 * 1024, 4),
        ("contiguous", 32, 1, 512, 64, "flash_attn_dense",
         {"q": 128, "kv": 128, "head": 64}, 48 * 1024, 1),
        ("contiguous", 1, 1, 1024, 64, "flash_attn_splitkv",
         {"q": 64, "kv": 256, "head": 64}, 72 * 1024, 4),
        ("paged", 1, 512, 512, 128, "vllm_flash_attn_splitkv",
         {"q": 64, "kv": 128, "head": 128}, 80 * 1024, 1),
    ),
)
def test_nvidia_a800_flash_attn_selects_exact_source_family(
        hardware, kv_layout, batch, q_len, kv_len, head_dim,
        expected_algorithm, expected_tile, expected_smem, expected_splits):
    variant = "decode" if q_len == 1 else "prefill"
    launch = select_launch_config(
        "nvidia_a800",
        "FLASH_ATTN",
        {
            "batch": batch,
            "q_len": q_len,
            "kv_len": kv_len,
            "head_dim": head_dim,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        variant,
        {
            "is_decode": variant == "decode",
            "kv_layout": kv_layout,
            "launch_policy": "analytical_launch",
            "split_policy": "serving",
        },
        hardware=hardware["nvidia_a800"],
    )

    assert launch.algorithm == expected_algorithm
    assert dict(launch.tiling.tile_shape) == expected_tile
    assert launch.tiling.num_kv_splits == expected_splits
    assert launch.tiling.pipeline_depth is None
    assert launch.parameters["page_table_bytes_per_kv_panel"] == (
        4.0 if kv_layout in ("paged", "paged_varlen")
        and expected_splits > 1 else 0.0)


@pytest.mark.parametrize(
    "kv_layout", ("paged", "paged_varlen", "chunked_prefill"))
def test_nvidia_a800_paged_mha_decode_uses_unsplit_source_dispatch(
        hardware, kv_layout):
    launch = select_launch_config(
        "nvidia_a800",
        "FLASH_ATTN",
        {
            "batch": 2,
            "q_len": 1,
            "kv_len": 8192,
            "head_dim": 128,
            "num_query_heads": 8,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "kv_layout": kv_layout,
            "launch_policy": "analytical_launch",
            "split_policy": "serving",
        },
        hardware=hardware["nvidia_a800"],
    )

    assert launch.algorithm == "vllm_flash_attn_splitkv"
    assert launch.tiling.num_kv_splits == 1
    assert launch.grid == (1, 2, 8)
    assert launch.parameters["page_table_bytes_per_kv_panel"] == 0.0


def test_nvidia_a800_paged_gqa_decode_preserves_analytical_splits(hardware):
    launch = select_launch_config(
        "nvidia_a800",
        "FLASH_ATTN",
        {
            "batch": 1,
            "q_len": 1,
            "kv_len": 8192,
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "kv_layout": "paged",
            "launch_policy": "analytical_launch",
            "split_policy": "serving",
        },
        hardware=hardware["nvidia_a800"],
    )

    assert launch.algorithm == "vllm_flash_attn_splitkv"
    assert launch.tiling.num_kv_splits > 1
    assert launch.grid == (1, launch.tiling.num_kv_splits, 8)
    assert launch.parameters["page_table_bytes_per_kv_panel"] == 4.0


def test_nvidia_a800_paged_mha_decode_preserves_forced_observed_splits(
        hardware):
    launch = select_launch_config(
        "nvidia_a800",
        "FLASH_ATTN",
        {
            "batch": 2,
            "q_len": 1,
            "kv_len": 8192,
            "head_dim": 128,
            "num_query_heads": 8,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "kv_layout": "paged",
            "launch_policy": "analytical_launch",
            "split_policy": "forced",
            "num_kv_splits": 3,
        },
        hardware=hardware["nvidia_a800"],
    )

    assert launch.algorithm == "vllm_flash_attn_splitkv"
    assert launch.tiling.num_kv_splits == 3
    assert launch.grid == (1, 3, 16)


@pytest.mark.parametrize(
    ("kv_len", "forced_splits", "expected_algorithm", "expected_grid"),
    (
        (512, 1, "flash_attn_dense", (1, 1, 8)),
        (1024, 2, "flash_attn_splitkv", (1, 2, 8)),
    ),
)
def test_nvidia_a800_contiguous_decode_preserves_forced_split_contract(
        hardware, kv_len, forced_splits, expected_algorithm, expected_grid):
    launch = select_launch_config(
        "nvidia_a800",
        "FLASH_ATTN",
        {
            "batch": 1,
            "q_len": 1,
            "kv_len": kv_len,
            "head_dim": 128,
            "num_query_heads": 32,
            "num_kv_heads": 8,
        },
        "bf16",
        "decode",
        {
            "is_decode": True,
            "kv_layout": "contiguous",
            "launch_policy": "analytical_launch",
            "split_policy": "forced",
            "num_kv_splits": forced_splits,
        },
        hardware=hardware["nvidia_a800"],
    )

    assert launch.algorithm == expected_algorithm
    assert launch.grid == expected_grid
    assert launch.tiling.num_kv_splits == forced_splits


def test_nvidia_a800_mla_rejects_generic_fa2_selector(hardware):
    with pytest.raises(NotImplementedError, match="two-stage Triton launch"):
        select_launch_config(
            "nvidia_a800",
            "FLASH_ATTN",
            {
                "batch": 1,
                "q_len": 1,
                "kv_len": 512,
                "head_dim": 512,
                "num_query_heads": 64,
                "num_kv_heads": 1,
            },
            "bf16",
            "decode",
            {
                "is_decode": True,
                "kv_layout": "mla",
                "launch_policy": "analytical_launch",
                "split_policy": "serving",
            },
            hardware=hardware["nvidia_a800"],
        )


@pytest.mark.parametrize(
    ("variant", "q_len", "kv_len", "expected_grid"),
    (
        ("prefill", 64, 512, (1, 32, 2)),
        ("decode", 1, 8, (1, 8, 2)),
        ("decode", 1, 1024, (16, 4, 1)),
    ),
)
def test_metax_c500_d64_flash_attn_uses_runtime_axis_order(
        hardware, variant, q_len, kv_len, expected_grid):
    del expected_grid
    with pytest.raises(
            NotImplementedError, match="head_dim 128"):
        select_launch_config(
            "metax_c500",
            "FLASH_ATTN",
            {
                "batch": 2,
                "q_len": q_len,
                "kv_len": kv_len,
                "head_dim": 64,
                "num_query_heads": 32,
                "num_kv_heads": 8,
            },
            "bf16",
            variant,
            {
                "is_decode": variant == "decode",
                "causal": variant == "prefill",
                "kv_layout": "contiguous",
                "kv_dist": "uniform",
                "q_lens": (q_len, q_len),
                "kv_lens": (kv_len, kv_len),
                "launch_policy": "analytical_launch",
            },
            hardware=hardware["metax_c500"],
        )


@pytest.mark.parametrize(
    ("hardware_id", "error"),
    (("nvidia_a800", "supports head_dim 64 or 128"),),
)
def test_gpu_flash_attn_rejects_unsupported_head_dimension(
        hardware, hardware_id, error):
    with pytest.raises(ValueError, match=error):
        select_launch_config(
            hardware_id,
            "FLASH_ATTN",
            {
                "batch": 1,
                "q_len": 1,
                "kv_len": 128,
                "head_dim": 512,
                "num_query_heads": 1,
                "num_kv_heads": 1,
            },
            "bf16",
            "decode",
            {"is_decode": True, "kv_layout": "contiguous"},
            hardware=hardware[hardware_id],
        )


def test_cann_vector_launch_rejects_gpu_runtime_variant(hardware):
    with pytest.raises(ValueError, match="unsupported CANN VECTOR"):
        select_launch_config(
            "ascend_910b", "VECTOR", {"element_count": 128}, "bf16",
            "moe_align", {}, hardware=hardware["ascend_910b"])


@pytest.mark.parametrize("hardware_id", ("nvidia_a800", "metax_c500"))
def test_gpu_vector_launch_rejects_cann_runtime_variant(
        hardware, hardware_id):
    with pytest.raises(ValueError, match="unsupported GPU VECTOR"):
        select_launch_config(
            hardware_id, "VECTOR", {"element_count": 128}, "bf16",
            "moe_swiglu", {}, hardware=hardware[hardware_id])


def test_metax_c500_moe_act_and_mul_follows_the_source_gate_launch():
    """One block per routed row, one thread per 16-byte output vector."""

    from src.kernel.hardware_spec import HWSpec
    from src.kernel.kernel_spec import KernelKind
    from src.kernel.launch_config.select_launch_config import (
        select_launch_config)

    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    attributes = {"launch_policy": "injected_launch"}
    shape = {
        "tokens": 35, "experts": 128, "topk": 8,
        "hidden": 2048, "intermediate": 768,
    }
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
        "moe_act_and_mul", attributes, hardware=hardware)
    # LAUNCH_ACTIVATION_GATE_KERNEL: grid(num_tokens),
    # block(min(d / (16 / element_size), 1024)) on the vector path.
    assert launch.grid == (35 * 8, 1, 1)
    assert launch.block == (768 // 8, 1, 1)

    # A projection half that does not divide the 16-byte vector falls to the
    # scalar branch, one thread per output element.
    scalar_shape = {**shape, "intermediate": 100}
    scalar = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, scalar_shape, "bf16",
        "moe_act_and_mul", attributes, hardware=hardware)
    assert scalar.block == (100, 1, 1)

    # The 1024-thread cap binds for a wide projection.
    wide = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR,
        {**shape, "intermediate": 16384}, "bf16", "moe_act_and_mul",
        attributes, hardware=hardware)
    assert wide.block == (1024, 1, 1)


def test_new_hardware_runs_shared_entry_from_declarations_only(tmp_path):
    """G3: a hardware added as data drives the shared entry point.

    The dispatcher and hardware-spec tables are derived from the hardware
    declarations at import time, so the test registers the synthetic hardware
    directly in the derived tables: one entry in the canonical-id set, one
    timer-domain entry in
    ``hardware_spec_module._SMALL_KERNEL_SPAN_FLOOR_TIMER_DOMAIN``, and the
    dispatcher rows ``dispatcher._HARDWARE_PACKAGES`` (projected package),
    ``dispatcher._OWN_PACKAGES`` and ``dispatcher._OWN_PACKAGE_SELECTORS``
    (owned package and its selectors).  Its resources come from a re-keyed
    copy of the A800 hardware config.  It adds no implementation module, and
    the launch selectors it reuses are the ones already shipped.
    """

    import shutil
    import yaml

    from src.kernel import hardware_ids
    from src.kernel import hardware_spec as hardware_spec_module
    from src.kernel.launch_config import select_launch_config as dispatcher

    synthetic_id = "nvidia_synthetic_probe"
    if synthetic_id in hardware_ids.CANONICAL_HARDWARE_IDS:
        pytest.skip("synthetic hardware id is already registered")

    # Build a self-consistent hardware declaration tree: a copy of the A800
    # config and its measured evidence, re-keyed to the synthetic id.  This is
    # data, not a new implementation path.
    config_text = Path("configs/hardware/nvidia_a800.yaml").read_text()
    config = yaml.safe_load(config_text.replace("nvidia_a800", synthetic_id))

    def _rekey(node):
        if isinstance(node, dict):
            return {key: _rekey(value) for key, value in node.items()}
        if isinstance(node, list):
            return [_rekey(value) for value in node]
        if isinstance(node, str):
            return node.replace("nvidia_a800", synthetic_id)
        return node

    evidence_dir = tmp_path / "microbench" / synthetic_id
    shutil.copytree(
        Path("microbench") / "nvidia_a800", evidence_dir)
    repository_root = Path(__file__).resolve().parents[2]
    for path in evidence_dir.rglob("*.json"):
        try:
            document = json.loads(path.read_text())
        except ValueError:
            continue
        path.write_text(json.dumps(_rekey(document), indent=2) + "\n")

    def _reanchor(node):
        if isinstance(node, dict):
            if "reference_path" in node and "reference_sha256" in node:
                reference = Path(node["reference_path"])
                if reference.is_file():
                    node["reference_sha256"] = hashlib.sha256(
                        reference.read_bytes()).hexdigest()
            for value in node.values():
                _reanchor(value)
        elif isinstance(node, list):
            for value in node:
                _reanchor(value)

    # The repository-root-relative reference_path values must resolve, so the
    # synthetic evidence tree replaces the tracked one for this probe only.
    original_evidence = Path("microbench") / synthetic_id
    created_evidence = not original_evidence.exists()
    if created_evidence:
        shutil.copytree(evidence_dir, original_evidence)
    try:
        _reanchor(config)
        config_path = tmp_path / f"{synthetic_id}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))

        hardware_ids.CANONICAL_HARDWARE_IDS = (
            hardware_ids.CANONICAL_HARDWARE_IDS | {synthetic_id})
        hardware_spec_module._SMALL_KERNEL_SPAN_FLOOR_TIMER_DOMAIN[
            synthetic_id] = "nsys_gpu_kernel"
        # Declare the hardware: project A800 and own one FA selector.
        dispatcher._HARDWARE_PACKAGES[synthetic_id] = (
            "src.kernel.launch_config.nvidia_a800")
        dispatcher._OWN_PACKAGES[synthetic_id] = (
            "src.kernel.launch_config.nvidia_h200")
        dispatcher._OWN_PACKAGE_SELECTORS[synthetic_id] = frozenset(
            {("flash_attn_launch", "select_flash_attn_launch")})

        spec = HWSpec(str(config_path))
        assert spec.hardware_id == synthetic_id
        assert spec.resources

        # The shared entry point serves the new hardware with no new module.
        policy = {"launch_policy": "analytical_launch"}
        gemm = dispatcher.select_launch_config(
            synthetic_id, KernelKind.GEMM,
            {"b": 1, "m": 128, "n": 256, "k": 512}, "bf16", "gemm",
            policy, hardware=spec)
        assert gemm.execution_variant == "gemm"
        norm = dispatcher.select_launch_config(
            synthetic_id, KernelKind.VECTOR, {"rows": 2, "hidden": 1024},
            "bf16", "rmsnorm", policy, hardware=spec)
        assert norm.execution_variant == "rmsnorm"
    finally:
        hardware_ids.CANONICAL_HARDWARE_IDS = (
            hardware_ids.CANONICAL_HARDWARE_IDS - {synthetic_id})
        hardware_spec_module._SMALL_KERNEL_SPAN_FLOOR_TIMER_DOMAIN.pop(
            synthetic_id, None)
        dispatcher._HARDWARE_PACKAGES.pop(synthetic_id, None)
        dispatcher._OWN_PACKAGES.pop(synthetic_id, None)
        dispatcher._OWN_PACKAGE_SELECTORS.pop(synthetic_id, None)
        if created_evidence:
            shutil.rmtree(original_evidence, ignore_errors=True)


def test_undeclared_hardware_id_is_rejected_by_the_canonical_registry():
    """G5: an id that no hardware declares is refused, never defaulted."""

    from src.kernel.hardware_ids import canonical_hardware_id

    with pytest.raises(ValueError, match="unknown hardware_id"):
        canonical_hardware_id("nvidia_not_a_card")


def test_hardware_spec_missing_a_required_field_is_refused(tmp_path):
    """G5: a hardware config missing a declared field fails closed."""

    import yaml

    config = yaml.safe_load(
        Path("configs/hardware/nvidia_a800.yaml").read_text())
    del config["features"]["kernel_memory_model"]
    config_path = tmp_path / "nvidia_a800.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(
            ValueError, match="features schema mismatch.*kernel_memory_model"):
        HWSpec(str(config_path))


def test_hardware_spec_with_an_undescribed_reference_is_refused(tmp_path):
    """G5: measured evidence that no longer matches its record fails closed."""

    import yaml

    config = yaml.safe_load(
        Path("configs/hardware/nvidia_a800.yaml").read_text())
    config["scheduling"]["small_kernel_span_floor"]["reference_sha256"] = (
        "0" * 64)
    config_path = tmp_path / "nvidia_a800_sha.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

    with pytest.raises(ValueError, match="reference SHA256 mismatch"):
        HWSpec(str(config_path))


def test_uncovered_launch_domain_is_refused_not_defaulted():
    """G5: a domain no hardware covers raises instead of falling back."""

    with pytest.raises(
            NotImplementedError,
            match="no source-backed .* launch selector"):
        select_launch_config(
            "metax_c500", KernelKind.GEMM,
            {"b": 1, "m": 128, "n": 256, "k": 512}, "bf16",
            "batched_gemm", {}, hardware=HWSpec(
                "configs/hardware/metax_c500.yaml"))
