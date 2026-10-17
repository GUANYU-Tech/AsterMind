"""H200 SGLang 0.5.12 fused-MoE (shared expert fused): launch and pricing contract."""

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.nvidia_h200.fused_moe_launch import (
    sglang_expanded_rows,
)
from src.kernel.launch_config.select_launch_config import (
    select_launch_config,
)

_H200 = HWSpec("configs/hardware/nvidia_h200.yaml")


def _counts(tokens):
    counts = [0] * 256
    for route in range(tokens * 8):
        counts[(route * 37) % 256] += 1
    return tuple(counts)


def _spec(tokens, variant, fused=True):
    shape = {"tokens": tokens, "experts": 256, "topk": 8, "hidden": 7168,
             "intermediate": 256}
    counts = _counts(tokens)
    attributes = {
        "tokens_per_expert": counts, "local_tokens_per_expert": counts,
        "expert_parallel_size": 1, "expert_rank": 0,
        "owned_experts": tuple(range(256)), "expert_weight_tile_rows": None,
        "routed_scaling_factor": 2.5, "routed_weight_dtype": "fp8",
        "weight_group_size": 0,
    }
    if fused:
        attributes["fused_shared_experts"] = 1
    launch = select_launch_config(
        "nvidia_h200", KernelKind.FUSED_MOE, shape, "bf16", variant,
        attributes, hardware=_H200)
    return KernelSpec(kind=KernelKind.FUSED_MOE, hardware_id="nvidia_h200",
                      shape=shape, dtype="bf16", variant=variant,
                      launch=launch, attributes=attributes)


@pytest.mark.parametrize(
    "tokens, variant, algorithm, tile",
    [
        # up config key 4: BM16 BN64 s5; down key 4: TMA BM16 BN128 BK128 s2
        (4, "w1", "sglang_triton_fused_moe_fp8_block128_aligned_w1",
         (16, 64, 128)),
        (4, "w2", "sglang_triton_fused_moe_fp8_block128_aligned_tma_w2",
         (16, 128, 128)),
        # down key 32: TMA BM16 BN256 BK64 s3
        (30, "w2", "sglang_triton_fused_moe_fp8_block128_aligned_tma_w2",
         (16, 256, 64)),
    ],
)
def test_sglang_config_and_grid(tokens, variant, algorithm, tile):
    spec = _spec(tokens, variant)
    assert spec.launch.algorithm == algorithm
    shape = spec.launch.tiling.tile_shape
    assert (shape["m"], shape["n"], shape["k"]) == tile
    # moe_align_block_size over E = 257 experts and top-k 9.
    rows = sglang_expanded_rows(tokens, 9, 257, 16)
    n = 512 if variant == "w1" else 7168
    assert spec.launch.grid_size == -(-rows // tile[0]) * -(-n // tile[1])
    assert predict(spec, _H200, "midlayer").t_total_s > 0


def test_expanded_rows_follow_moe_align_block_size():
    assert sglang_expanded_rows(4, 9, 257, 16) == 36 * 16
    assert sglang_expanded_rows(64, 9, 257, 16) == 576 + 258 * 15


def test_fused_shared_expert_adds_work():
    fused = predict(_spec(16, "w1"), _H200, "midlayer").t_total_s
    assert fused > 0


def test_fused_shared_experts_must_be_one():
    spec = _spec(4, "w1")
    attributes = dict(spec.attributes, fused_shared_experts=2)
    with pytest.raises(ValueError, match="fused_shared_experts"):
        KernelSpec(kind=spec.kind, hardware_id=spec.hardware_id,
                   shape=spec.shape, dtype=spec.dtype, variant=spec.variant,
                   launch=spec.launch, attributes=attributes)


def test_vllm_launch_is_unchanged_without_fusion():
    spec = _spec(4, "w1", fused=False)
    assert spec.launch.algorithm.startswith(
        "vllm_triton_fused_moe_fp8_block128")
