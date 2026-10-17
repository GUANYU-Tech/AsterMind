"""H200 target-model BF16 GEMMs: MLA strided-batched GEMM and the DSV3 router kernel."""

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import (
    select_launch_config,
)

_H200 = HWSpec("configs/hardware/nvidia_h200.yaml")
_A800 = HWSpec("configs/hardware/nvidia_a800.yaml")
_BMM_ATTRIBUTES = {
    "launch_policy": "analytical_launch", "batch_layout": "strided",
    "weight_dtype": "bf16", "weight_group_size": 0,
}
_ROUTER_ATTRIBUTES = {
    "launch_policy": "analytical_launch", "score_function": "sigmoid",
    "topk_method": "noaux_tc", "expert_group_count": 8,
    "topk_group_count": 4, "normalize_topk_probability": True,
    "weight_dtype": "bf16", "weight_group_size": 0,
}


def _launch(variant, shape, attributes, hardware=_H200):
    return select_launch_config(
        hardware.hardware_id, KernelKind.GEMM, shape, "bf16", variant,
        attributes, hardware=hardware)


@pytest.mark.parametrize("m, n, k", [(1, 128, 512), (4, 512, 128), (64, 128, 512)])
def test_batched_gemm_launches_every_batch_problem(m, n, k):
    shape = {"b": 16, "m": m, "n": n, "k": k}
    launch = _launch("batched_gemm", shape, _BMM_ATTRIBUTES)
    tile = launch.tiling.tile_shape
    per_problem = -(-m // tile["m"]) * -(-n // tile["n"])
    # One output tile can never span two batch problems (distinct operands).
    assert launch.grid == (per_problem, 16, 1)
    assert tile["m"] == (32 if m <= 32 else 64)
    # The A800 package keeps its own projection.
    a800 = _launch("batched_gemm", shape, _BMM_ATTRIBUTES, _A800)
    assert a800.grid[1] == 1


def test_router_uses_dsv3_kernel_only_inside_the_vllm_tier():
    small = _launch("moe_router", {"b": 1, "m": 16, "n": 256, "k": 7168},
                    _ROUTER_ATTRIBUTES)
    assert (small.algorithm, small.grid, small.block, small.tiling) == (
        "vllm_dsv3_router_gemm_float_out", (256, 1, 1), (128, 1, 1), None)
    large = _launch("moe_router", {"b": 1, "m": 17, "n": 256, "k": 7168},
                    _ROUTER_ATTRIBUTES)
    assert large.algorithm != "vllm_dsv3_router_gemm_float_out"
    unsupported = _launch("moe_router", {"b": 1, "m": 4, "n": 128, "k": 7168},
                          _ROUTER_ATTRIBUTES)
    assert unsupported.algorithm != "vllm_dsv3_router_gemm_float_out"


def _predict_us(variant, shape, attributes):
    spec_attributes = {
        name: value for name, value in attributes.items()
        if name != "launch_policy"}
    if variant == "batched_gemm":
        spec_attributes.update(
            a_batch_stride_elements=shape["m"] * shape["k"],
            b_batch_stride_elements=shape["k"] * shape["n"],
            c_batch_stride_elements=shape["m"] * shape["n"])
    spec = KernelSpec(
        kind=KernelKind.GEMM, hardware_id=_H200.hardware_id,
        shape={**shape, "dtype_bytes": 2}, dtype="bf16", variant=variant,
        launch=_launch(variant, shape, attributes), attributes=spec_attributes)
    return predict(spec, _H200, "midlayer").t_total_s * 1e6


def test_router_simt_prediction_grows_with_tokens():
    times = [_predict_us("moe_router", {"b": 1, "m": m, "n": 256, "k": 7168},
                         _ROUTER_ATTRIBUTES) for m in (1, 4, 16)]
    assert 0 < times[0] <= times[1] <= times[2]


def test_batched_gemm_prediction_does_not_serialize_batches():
    one = _predict_us("batched_gemm", {"b": 1, "m": 4, "n": 128, "k": 512},
                      _BMM_ATTRIBUTES)
    sixteen = _predict_us("batched_gemm", {"b": 16, "m": 4, "n": 128, "k": 512},
                          _BMM_ATTRIBUTES)
    assert sixteen < 16 * one
