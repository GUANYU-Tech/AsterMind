"""Compact single-wave timing for H200 V4 FP8 projections."""

from functools import lru_cache
from math import ceil

from src.kernel.formula import small_kernel_span_floor_s
from src.kernel.launch_config.block_fp8_gemm import TRTLLM_SCALE_1X128_CTAS_PER_SM
from src.kernel.launch_config.nvidia_h200.gemm_launch import select_block_fp8_gemm_config
from src.kernel.predictors.gemm_block_fp8 import ring_stage_interval_ns


# Tile selection is deterministic for a matrix shape and SM count.
_projection_tile = lru_cache(maxsize=256)(select_block_fp8_gemm_config)


class SmallFp8Gemm:
    def __init__(self, spec, hardware):
        self.hardware = hardware
        self.sms = spec.num_compute_units
        self.ring = dict(spec.block_fp8_gemm_tma_ring_service["ring"])
        self.ring["dram_stream_bytes_per_ns"] = hardware.hbm_bytes * hardware.hbm_efficiency * 1e-9
        self.gemm_floor = small_kernel_span_floor_s(spec, self.sms, "gemm", False)
        self.quant_floor = small_kernel_span_floor_s(spec, TRTLLM_SCALE_1X128_CTAS_PER_SM * self.sms, "gemm", False)
        self.deepgemm_floors = {}

    def __call__(self, rows, out_features, in_features, *, groups=1, prequantized=False):
        """Estimate one projection without instantiating a kernel or memory plan."""
        weights = groups * out_features * in_features
        io = groups * rows * (in_features + out_features) * 2
        base = self.hardware.roofline(2 * rows * weights, weights * (1 + 4 / 128**2) + io, fp8=True)
        if rows < 32:
            wave = self._one_wave(rows, out_features, in_features, groups)
            if not prequantized:
                quant_bytes = rows * in_features * (3 + 4 / 128)
                wave += max(self.quant_floor, quant_bytes / (self.ring["dram_stream_bytes_per_ns"] * 1e9))
            return max(base, wave)

        # At the provider switch, take the first DeepGEMM wave as a fixed
        # launch floor per projection shape. Larger inputs still grow with Roofline.
        shape = (out_features, in_features, groups)
        if shape not in self.deepgemm_floors:
            self.deepgemm_floors[shape] = self._one_wave(32, *shape)
        return max(base, self.deepgemm_floors[shape])

    def _one_wave(self, rows, out_features, in_features, groups):
        tile = _projection_tile(rows, out_features, in_features, self.sms)
        column_tiles = groups * ceil(out_features / tile.tile_n)
        if ceil(rows / tile.tile_m) * column_tiles > self.sms:
            return self.gemm_floor

        stage_ns = ring_stage_interval_ns(self.ring, tile.tile_n * 128, column_tiles)
        tensor_s = (2 * tile.tile_m * tile.tile_n * 128 * self.sms
                    / (self.hardware.fp8_flops * self.hardware.tensor_efficiency))
        fill_ns = self.ring["fill_base_ns"] + self.ring["fill_per_store_byte_ns"] * 2 * tile.tile_m * tile.tile_n
        wave = fill_ns * 1e-9 + ceil(in_features / 128) * max(stage_ns * 1e-9, tensor_s)
        return max(self.gemm_floor, wave)
