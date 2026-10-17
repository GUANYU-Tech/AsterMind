"""Test suite 3: kernel name parsing regression tests."""
import pytest

from evaluation.kernel.observation_parser import (
    is_non_tensor,
    parse_flash_attention_trait,
    parse_flash_attention_tile as _parse_fa_template,
    parse_gemm_tile as _parse_tile_dims,
)
from src.kernel.launch_config.tiling_spec import parse_complete_gemm_tile


class TestParseTileDims:
    def test_cutlass_3d(self):
        # Cutlass "128x64x32" → (Mb=64, Nb=128, Kb=32)
        mb, nb, kb = _parse_tile_dims("ampere_fp16_s16816gemm_fp16_128x64x32_tn_align8")
        assert mb == 64
        assert nb == 128
        assert kb == 32

    def test_unparseable_returns_none_triple(self):
        mb, nb, kb = _parse_tile_dims("unknown_kernel_no_dims")
        assert mb is None
        assert nb is None
        assert kb is None

    def test_dims_outside_range_not_parsed(self):
        # 8x8x8 — all below 32 threshold; should not be parsed
        mb, nb, kb = _parse_tile_dims("something_8x8x8_end")
        assert mb is None

    def test_triton_2d_with_stages(self):
        # Triton "128x64" + "stages_32x1" → Mb=64, Nb=128, Kb=32
        name = "triton_gemm_128x64_stages_32x1"
        mb, nb, kb = _parse_tile_dims(name)
        assert mb == 64
        assert nb == 128
        assert kb == 32

    def test_cutlass_depth_six_suffix_is_a_complete_tile(self):
        name = (
            "void cutlass::Kernel2<cutlass_80_tensorop_f16_"
            "s16816gemm_relu_f16_128x64_64x6_tn_align8>(T1::Params)")

        assert parse_complete_gemm_tile(name) == (64, 128, 64, 6)

    def test_cutlass_k32_depth_six_suffix_is_a_complete_tile(self):
        name = (
            "void cutlass::Kernel2<cutlass_80_tensorop_f16_"
            "s16816gemm_relu_f16_128x256_32x6_tn_align8>(T1::Params)")

        assert parse_complete_gemm_tile(name) == (256, 128, 32, 6)

    def test_other_cutlass_k32_depth_six_suffix_remains_opaque(self):
        name = (
            "void cutlass::Kernel2<cutlass_80_tensorop_f16_"
            "s16816gemm_relu_f16_128x128_32x6_tn_align8>(T1::Params)")

        assert parse_complete_gemm_tile(name) == (128, 128, 32, None)

    def test_audited_cutlass_depth_three_suffix_is_a_complete_tile(self):
        name = (
            "void cutlass::Kernel2<cutlass_80_tensorop_f16_"
            "s16816gemm_relu_f16_256x128_64x3_tn_align8>(T1::Params)")

        assert parse_complete_gemm_tile(name) == (128, 256, 64, 3)

    def test_unvalidated_cutlass_stage_suffix_remains_opaque(self):
        name = (
            "void cutlass::Kernel2<cutlass_80_tensorop_f16_"
            "s16816gemm_relu_f16_128x128_32x5_tn_align8>(T1::Params)")

        assert parse_complete_gemm_tile(name) == (128, 128, 32, None)


class TestParseFATemplate:
    def test_valid_fa_kernel(self):
        name = "void flash::flash_fwd_kernel_traits<(int)128, (int)64, (int)128, ..."
        result = _parse_fa_template(name)
        assert result is not None
        kBM, kBN = result
        assert kBM == 64
        assert kBN == 128

    def test_invalid_returns_none(self):
        assert _parse_fa_template("unknown_fa_kernel") is None

    def test_combine_kernel_unparseable(self):
        # combine kernel names don't match the FA tensor template pattern
        name = "void flash::flash_fwd_splitkv_combine_kernel"
        assert _parse_fa_template(name) is None

    def test_vllm_ampere_mla_grouped_tile(self):
        # BLOCK_H is a query-head tile, not the parser's Q-row tile axis.
        assert _parse_fa_template("_fwd_grouped_kernel_stage1") is None

    def test_metax_trait_retains_head_tile_and_warp_positions(self):
        name = (
            "void flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, "
            "4, true, true, mctlass::bfloat16_t>>")
        assert parse_flash_attention_trait(name) == (128, 128, 64, 4)


def test_vllm_ampere_mla_stage2_is_a_non_tensor_reduction():
    assert not is_non_tensor("FLASHATTN", "_fwd_grouped_kernel_stage1")
    assert is_non_tensor("FLASHATTN", "_fwd_kernel_stage2")
