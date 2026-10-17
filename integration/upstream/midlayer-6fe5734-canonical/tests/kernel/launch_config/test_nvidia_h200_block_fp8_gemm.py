"""H200 block-FP8 dense GEMM: source-derived launch and template contract."""

from dataclasses import replace

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import predict
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.nvidia_h200.gemm_launch import (
    select_block_fp8_gemm_config,
    select_gemm_launch,
)
from src.kernel.launch_config.block_fp8_gemm import (
    BLOCK_FP8_WEIGHT_QUANTIZATION,
    DEEPGEMM_1D2D_ALGORITHM,
    TRTLLM_SWAP_AB_ALGORITHM,
)
from src.kernel.launch_config.select_launch_config import (
    select_launch_config,
)

_H200 = HWSpec("configs/hardware/nvidia_h200.yaml")
_ATTRIBUTES = {
    "launch_policy": "analytical_launch",
    "weight_dtype": "fp8",
    "weight_quantization": BLOCK_FP8_WEIGHT_QUANTIZATION,
}


def _spec(m, n, k, attributes=_ATTRIBUTES, hardware=_H200):
    shape = {"b": 1, "m": m, "n": n, "k": k}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.GEMM, shape, "bf16", "gemm",
        attributes, hardware=hardware)
    return KernelSpec(
        kind=KernelKind.GEMM, hardware_id=hardware.hardware_id, shape=shape,
        dtype="bf16", variant="gemm", launch=launch, attributes={})


@pytest.mark.parametrize(
    "m, n, k, expected",
    [
        # (algorithm, tile_m, tile_n, stages, cluster, threads, smem bytes)
        (1, 2112, 7168, ("swap_ab", 16, 128, 8, 1, 384, 152928)),
        (16, 6144, 1536, ("swap_ab", 16, 128, 8, 1, 384, 152752)),
        (64, 9216, 7168, ("1d2d", 64, 80, 11, 1, 256, 216512)),
        (64, 6144, 1536, ("1d2d", 64, 48, 15, 1, 256, 225376)),
        (29492, 7168, 4096, ("1d2d", 256, 128, 3, 2, 384, 216448)),
    ],
)
def test_source_heuristics_reproduce_vllm_0_24_kernels(m, n, k, expected):
    config = select_block_fp8_gemm_config(m, n, k, 132)
    algorithm = {"swap_ab": TRTLLM_SWAP_AB_ALGORITHM,
                 "1d2d": DEEPGEMM_1D2D_ALGORITHM}[expected[0]]
    assert (config.algorithm, config.tile_m, config.tile_n, config.stages,
            config.cluster, 128 + config.math_threads,
            config.smem_bytes) == (algorithm, *expected[1:])
    launch = _spec(m, n, k).launch
    assert launch.algorithm == algorithm
    assert launch.grid == (132, 1, 1)
    assert launch.cluster == (config.cluster, 1, 1)
    assert launch.tiling.split_k == 1


def test_block_fp8_requires_flashinfer_eligible_shape():
    with pytest.raises(NotImplementedError):
        select_block_fp8_gemm_config(1, 2100, 7168, 132)


def test_weight_quantization_does_not_change_a800_launch():
    a800 = HWSpec("configs/hardware/nvidia_a800.yaml")
    plain = {"launch_policy": "analytical_launch", "weight_dtype": "fp8"}
    for m, n, k in ((1, 2112, 7168), (29492, 7168, 4096)):
        assert (_spec(m, n, k, _ATTRIBUTES, a800).launch
                == _spec(m, n, k, plain, a800).launch)


def test_h200_without_declaration_keeps_projected_bf16_launch():
    plain = {"launch_policy": "analytical_launch", "weight_dtype": "fp8"}
    assert _spec(1, 2112, 7168, plain).launch.algorithm == (
        "cublas_analytical_gemm")


def test_single_wave_template_is_ring_bound_and_monotone_in_k():
    short = predict(_spec(1, 7168, 512), _H200, "midlayer")
    long = predict(_spec(1, 7168, 4096), _H200, "midlayer")
    assert 0.0 < short.t_total_s < long.t_total_s
    assert "trtllm_scale_1x128" in long.t_per_group
    assert long.bottleneck_group == "tma_ring_stages"
    decode = predict(_spec(64, 7168, 4096), _H200, "midlayer")
    assert "trtllm_scale_1x128" not in decode.t_per_group


def test_multi_wave_template_is_tensor_bound_for_long_k():
    prediction = predict(_spec(29492, 9216, 7168), _H200, "midlayer")
    assert prediction.bottleneck_group == "fp8_tensor"
    flops = 2.0 * 29492 * 9216 * 7168
    # The busiest-CTA tensor time cannot beat the public FP8 rate.
    assert prediction.t_total_s > flops / 1.9788e15


def test_template_requires_published_ring_service():
    spec = _spec(1, 2112, 7168)
    hardware = HWSpec("configs/hardware/nvidia_h200.yaml")
    hardware.block_fp8_gemm_tma_ring_service = None
    with pytest.raises(ValueError, match="TMA-ring"):
        predict(spec, hardware, "midlayer")


def test_block_fp8_launch_exposes_no_resource_formula_input():
    from src.kernel.kernel_base import create_predictor

    with pytest.raises(NotImplementedError):
        create_predictor(KernelKind.GEMM).resource_input(
            _spec(1, 2112, 7168), _H200)
    with pytest.raises(ValueError, match="not a block-FP8"):
        from src.kernel.predictors.gemm_block_fp8 import (
            predict_block_fp8_gemm)
        spec = _spec(1, 2112, 7168)
        predict_block_fp8_gemm(
            replace(spec, launch=replace(spec.launch, algorithm="other")),
            _H200)


def test_h200_selector_has_no_rule_outside_its_declared_domain():
    plain = {"launch_policy": "analytical_launch", "weight_dtype": "fp8"}
    with pytest.raises(ValueError, match="only owns"):
        select_gemm_launch(
            {"b": 1, "m": 1, "n": 2112, "k": 7168}, "bf16", "gemm", plain,
            hardware=_H200)


def test_input_quantiser_follows_the_selected_block_fp8_path():
    from src.kernel.launch_config.select_launch_config import (
        select_block_fp8_input_quant_launch,
    )

    # DeepGEMM 1d2d (M >= 32): vLLM's separate per_token_group_quant_8bit,
    # 16 groups of 128 per CTA when the group count allows it.
    deepgemm = _spec(64, 7168, 2048).launch
    quant = select_block_fp8_input_quant_launch(
        _H200.hardware_id, deepgemm, 64 * 2048, _H200)
    assert (quant.execution_variant, quant.grid, quant.block) == (
        "fp8_group_quant", (64, 1, 1), (256, 1, 1))
    # Swap-AB (M < 32) quantises inside its own call.
    swap_ab = _spec(16, 7168, 2048).launch
    assert select_block_fp8_input_quant_launch(
        _H200.hardware_id, swap_ab, 16 * 2048, _H200) is None
    spec = KernelSpec(
        kind=KernelKind.VECTOR, hardware_id=_H200.hardware_id,
        shape={"element_count": 64 * 2048}, dtype="bf16",
        variant="fp8_group_quant", launch=quant,
        attributes={"group_size": 128})
    assert predict(spec, _H200, "midlayer").t_total_s > 0
