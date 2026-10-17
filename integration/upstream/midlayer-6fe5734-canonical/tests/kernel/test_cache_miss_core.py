"""Contract tests for the shared L2-to-DRAM miss core and its access orders."""

import math

import numpy as np
import pytest

from src.kernel.launch_config.launch_program import GpuCtaRasterProgram
from src.kernel.lru_cache import (
    simulate_gemm_tile_program,
    simulate_moe_tile_program,
)


def _dense(**overrides):
    arguments = {
        "m_extent": 512,
        "n_extent": 512,
        "k_extent": 512,
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 64,
        "a_element_bytes": 2,
        "b_element_bytes": 2,
        "c_element_bytes": 2,
        "split_k": 1,
        "accumulate_in_place": False,
        "wave_size": 4,
        "capacity_bytes": 8 * 1024 * 1024,
    }
    arguments.update(overrides)
    return arguments


def _schedule(m_tiles, n_tiles, slices=1):
    return np.array(
        [(m, n, z) for z in range(slices)
         for n in range(n_tiles) for m in range(m_tiles)],
        dtype=np.int32)


class TestMissCoreReplacement:
    def test_a_resident_program_serves_only_its_compulsory_bytes(self):
        traffic = simulate_gemm_tile_program(
            _schedule(4, 4), **_dense())

        compulsory = 512 * 512 * 2
        assert traffic.a_dram_bytes == compulsory
        assert traffic.b_dram_bytes == compulsory
        assert traffic.output_dram_bytes == compulsory
        assert traffic.l2_bytes > traffic.dram_bytes

    def test_a_capacity_below_one_wave_step_misses_every_operand(self):
        """One wave touches four A and four B tiles of 16 KiB per K step."""

        arguments = _dense(capacity_bytes=64 * 1024)
        traffic = simulate_gemm_tile_program(_schedule(4, 4), **arguments)

        # No A tile recurs inside a wave, so every A request is a miss.  The
        # four programs of a wave share one B tile at each K step, so B keeps
        # exactly the first of its four requests.
        assert traffic.a_dram_bytes == traffic.a_l2_bytes
        assert traffic.b_dram_bytes * 4 == traffic.b_l2_bytes

    def test_the_step_profile_is_the_capacity_step_it_replaces(self):
        """An explicit step profile at the capacity is the default rule."""

        capacity = 1024 * 1024
        default = simulate_gemm_tile_program(
            _schedule(4, 4), **_dense(capacity_bytes=capacity))
        explicit = simulate_gemm_tile_program(
            _schedule(4, 4), **_dense(capacity_bytes=capacity),
            residency_profile=((capacity, 0.0), (capacity + 1, 1.0)))

        assert explicit == default

    def test_a_graded_profile_serves_a_fraction_of_a_reused_object(self):
        """A profile that half-serves even a short reuse is not the step."""

        capacity = 1024 * 1024
        graded = simulate_gemm_tile_program(
            _schedule(4, 4), **_dense(capacity_bytes=capacity),
            residency_profile=((1, 0.5), (1 << 40, 0.5)))
        default = simulate_gemm_tile_program(
            _schedule(4, 4), **_dense(capacity_bytes=capacity))

        assert graded.dram_bytes > default.dram_bytes
        assert graded.dram_bytes < graded.l2_bytes

    @pytest.mark.parametrize("profile", [
        ((0, 0.0),),
        ((16, 0.0), (8, 1.0)),
        ((16, 0.5), (32, 0.25)),
        ((16, 1.5),),
    ])
    def test_an_invalid_residency_profile_is_rejected(self, profile):
        with pytest.raises(ValueError):
            simulate_gemm_tile_program(
                _schedule(2, 2), **_dense(), residency_profile=profile)

    def test_an_in_place_k_split_reads_its_output_tile_back(self):
        arguments = _dense(split_k=2, accumulate_in_place=True)
        in_place = simulate_gemm_tile_program(
            _schedule(4, 4, slices=2), **arguments)
        separate = simulate_gemm_tile_program(
            _schedule(4, 4, slices=2),
            **_dense(split_k=2, accumulate_in_place=False))

        # Two slices publish two output tiles; the second also reads its
        # predecessor's partial back before updating it.
        assert in_place.output_l2_bytes == 3 * separate.output_l2_bytes // 2

    def test_a_schedule_that_is_not_a_tile_program_is_rejected(self):
        with pytest.raises(TypeError):
            simulate_gemm_tile_program(
                np.array([[0, 0]], dtype=np.int32), **_dense())


class TestCtaRasterProgram:
    @pytest.mark.parametrize("convention,log_group,serpentine", [
        ("none", 0, False),
        ("cublaslt_m_group", 3, True),
        ("cublaslt_m_group", 0, True),
        ("cutlass_n_group", 3, False),
        ("cutlass_m_group", 1, False),
    ])
    def test_every_map_covers_its_output_tiles_exactly_once(
            self, convention, log_group, serpentine):
        program = GpuCtaRasterProgram(convention, log_group, serpentine, 2)

        schedule = program.schedule(37, 11)

        assert len(schedule) == 37 * 11 * 2
        assert len(set(schedule)) == len(schedule)
        assert {tile[2] for tile in schedule} == {0, 1}

    def test_the_cublaslt_band_walks_m_fastest_inside_one_band(self):
        program = GpuCtaRasterProgram("cublaslt_m_group", 3, False, 1)

        schedule = program.schedule(16, 4)

        assert schedule[:9] == (
            (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (4, 0, 0),
            (5, 0, 0), (6, 0, 0), (7, 0, 0), (0, 1, 0))

    def test_a_band_that_would_pass_the_grid_edge_halves(self):
        """Six M tiles cannot host an eight-row band; it shrinks to four."""

        program = GpuCtaRasterProgram("cublaslt_m_group", 3, False, 1)

        schedule = program.schedule(6, 4)

        assert len(schedule) == 24
        assert schedule[:5] == (
            (0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0), (0, 1, 0))

    def test_an_exact_three_row_remainder_keeps_a_base_three_band(self):
        program = GpuCtaRasterProgram("cublaslt_m_group", 2, False, 1)

        schedule = program.schedule(7, 3)

        assert len(schedule) == 21
        assert {tile[0] for tile in schedule[12:]} == {4, 5, 6}

    def test_the_serpentine_reverses_n_on_alternate_bands(self):
        direct = GpuCtaRasterProgram("cublaslt_m_group", 0, False, 1)
        serpentine = GpuCtaRasterProgram("cublaslt_m_group", 0, True, 1)

        assert direct.schedule(4, 5)[5] == (1, 0, 0)
        assert serpentine.schedule(4, 5)[5] == (1, 4, 0)

    def test_the_grouped_grid_states_its_own_group(self):
        assert GpuCtaRasterProgram(
            "cutlass_n_group", 3, False, 1).grid(64, 40) == (512, 5, 1)
        assert GpuCtaRasterProgram(
            "cutlass_m_group", 1, False, 1).grid(64, 43) == (86, 32, 1)
        assert GpuCtaRasterProgram(
            "cublaslt_m_group", 3, True, 2).grid(256, 86) == (86, 256, 2)

    @pytest.mark.parametrize("convention,log_group,serpentine", [
        ("none", 1, False),
        ("none", 0, True),
        ("cutlass_n_group", 2, True),
        ("unknown_group", 0, False),
    ])
    def test_an_unsupported_raster_contract_is_rejected(
            self, convention, log_group, serpentine):
        with pytest.raises(ValueError):
            GpuCtaRasterProgram(convention, log_group, serpentine, 1)


class TestMoETileProgram:
    def _arguments(self, **overrides):
        arguments = {
            "m_tiles": 8,
            "n_tiles": 4,
            "group_size_m": 1,
            "k_extent": 512,
            "block_k": 64,
            "n_extent": 256,
            "block_n": 64,
            "element_bytes": 2,
            "c_row_bytes": 128,
            "block_m": 64,
            "wave_size": 8,
            "tokens_per_expert": [128, 128, 128, 128],
            "capacity_bytes": 8 * 1024 * 1024,
        }
        arguments.update(overrides)
        return arguments

    def test_a_resident_routing_serves_each_expert_matrix_once(self):
        traffic = simulate_moe_tile_program(**self._arguments())

        assert traffic.b_dram_bytes == 4 * 512 * 256 * 2

    def test_padding_rows_beyond_the_routing_never_reach_memory(self):
        populated = simulate_moe_tile_program(
            **self._arguments(m_tiles=8, tokens_per_expert=[128, 128, 0, 0]))
        compact = simulate_moe_tile_program(
            **self._arguments(m_tiles=4, tokens_per_expert=[128, 128, 0, 0]))

        assert populated == compact

    def test_a_tail_n_tile_carries_only_its_logical_width(self):
        traffic = simulate_moe_tile_program(
            **self._arguments(n_tiles=3, n_extent=130, block_n=64))

        assert traffic.b_dram_bytes == 4 * 512 * 130 * 2

    def test_a_routing_with_no_populated_expert_is_rejected(self):
        with pytest.raises(ValueError):
            simulate_moe_tile_program(
                **self._arguments(tokens_per_expert=[0, 0, 0, 0]))
