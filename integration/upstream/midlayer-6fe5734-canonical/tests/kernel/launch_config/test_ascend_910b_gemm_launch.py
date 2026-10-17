from types import SimpleNamespace

import pytest

from src.kernel.launch_config.ascend_910b import gemm_launch
from src.kernel.launch_config.cann_gemm_schedule import (
    cann_gemm_persistent_descriptor_trace,
    cann_gemm_source_tile_order,
)
from src.kernel.launch_config.gemm_launch_selection import (
    CannGemmLaunchProgramSelection,
    GemmLaunchSelection,
)
from src.kernel.launch_config.launch_program import CannGemmLaunchProgram
from src.kernel.launch_config.gemm_workload import gemm_launch_selection_key


@pytest.fixture(autouse=True)
def _small_exact_domain(monkeypatch):
    monkeypatch.setattr(gemm_launch, "GEMM_EXACT_SEMANTIC_KEY_COUNT", 1)


def _attributes(policy="injected_launch"):
    return {
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
        "launch_policy": policy,
    }


def _selection(workload_key, tile=(256, 128, 64)):
    return {workload_key: GemmLaunchSelection(
        *tile, pipeline_depth=None, split_k=1, reduction="none")}


def _hardware():
    return SimpleNamespace(
        num_compute_units=24,
        l2_size_bytes=201_326_592,
        aic_mte2_cold_small_m_stride_service=None,
        resource_by_role=lambda role: SimpleNamespace(size_bytes=524_288),
    )


def _exact_program():
    program = CannGemmLaunchProgram(
        program_kind="cann_gemm_launch_v1",
        aic_workers=24,
        worker_grid_m=4,
        worker_grid_n=6,
        worker_grid_k=1,
        single_core_m=128,
        single_core_n=896,
        single_core_k=13824,
        base_m=128,
        base_n=224,
        base_k=64,
        depth_a1=2,
        depth_b1=2,
        a_l1_k_tiles=8,
        b_l1_k_tiles=4,
        step_m=1,
        step_n=1,
        m_l2_regions=1,
        n_l2_regions=1,
        m_l2_block=4,
        n_l2_block=6,
        traversal_order="persistent_n_then_m",
        a_uses_l2=True,
        b_uses_l2=True,
        split_k=1,
    )
    return program


def _selection_with_program(workload_key, program):
    fields = CannGemmLaunchProgramSelection._fields
    return {workload_key: GemmLaunchSelection(
        program.base_m, program.base_n, program.base_k, None,
        program.split_k, "none",
        CannGemmLaunchProgramSelection(
            **{field: getattr(program, field) for field in fields}))}


def test_cann_source_order_covers_tail_regions_in_declared_order():
    assert cann_gemm_source_tile_order(
        m_tiles=3,
        n_tiles=5,
        m_l2_regions=2,
        n_l2_regions=2,
        m_l2_block=2,
        n_l2_block=3,
        traversal_order="m_then_n_snake",
        aic_workers=4,
    ) == (
        (0, 0), (1, 1), (0, 2), (1, 0), (0, 1), (1, 2),
        (1, 3), (0, 3), (1, 4), (0, 4),
        (2, 3), (2, 4), (2, 2), (2, 0), (2, 1),
    )


def test_injected_gemm_uses_exact_semantic_key_and_compact_tile(monkeypatch):
    shape = {"b": 1, "m": 729, "n": 6144, "k": 2560}
    attributes = _attributes()
    workload_key = gemm_launch_selection_key(
        shape, "bf16", "gemm", attributes)
    monkeypatch.setattr(
        gemm_launch, "load_gemm_launch_selection",
        lambda *args, **kwargs: _selection(workload_key))
    gemm_launch._launch_facts.cache_clear()

    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", attributes, _hardware())

    assert launch.algorithm == "cann_injected_matmul"
    assert launch.grid == (24, 1, 1)
    assert launch.block == (1, 1, 1)
    assert dict(launch.tiling.tile_shape) == {"m": 256, "n": 128, "k": 64}
    assert launch.tiling.pipeline_depth == 2
    assert launch.tiling.split_k == 1
    assert launch.source_program is not None
    assert gemm_launch.select_gemm_reducer_launch(
        shape, "bf16", "gemm", attributes,
        _hardware()) is None
    gemm_launch._launch_facts.cache_clear()


def test_exact_program_is_shared_by_analytical_and_injected_paths(monkeypatch):
    shape = {"b": 1, "m": 512, "n": 5120, "k": 13824}
    attributes = _attributes()
    workload_key = gemm_launch_selection_key(
        shape, "bf16", "gemm", attributes)
    program = _exact_program()
    monkeypatch.setattr(
        gemm_launch, "_launch_facts",
        lambda: _selection_with_program(workload_key, program))

    injected = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", attributes, _hardware())
    analytical = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", _attributes("analytical_launch"),
        _hardware())

    assert injected == analytical
    assert injected.algorithm == "cann_matmul"
    assert injected.grid == (24, 1, 1)
    assert dict(injected.tiling.tile_shape) == {
        "m": 128, "n": 224, "k": 64,
    }
    assert injected.source_program == program


def test_exact_gemv_uses_the_captured_cann_gemm_program(monkeypatch):
    shape = {"b": 1, "m": 1, "n": 151936, "k": 4096}
    attributes = {
        **_attributes(),
        "position": "lm_head",
    }
    workload_key = gemm_launch_selection_key(
        shape, "bf16", "gemv", attributes)
    program = CannGemmLaunchProgram(
        program_kind="cann_gemm_launch_v1",
        aic_workers=24,
        worker_grid_m=1,
        worker_grid_n=24,
        worker_grid_k=1,
        single_core_m=16,
        single_core_n=6400,
        single_core_k=4096,
        base_m=16,
        base_n=256,
        base_k=64,
        depth_a1=1,
        depth_b1=2,
        a_l1_k_tiles=64,
        b_l1_k_tiles=4,
        step_m=1,
        step_n=1,
        m_l2_regions=1,
        n_l2_regions=1,
        m_l2_block=1,
        n_l2_block=24,
        traversal_order="persistent_n_then_m",
        a_uses_l2=False,
        b_uses_l2=False,
        split_k=1,
    )
    monkeypatch.setattr(
        gemm_launch, "_launch_facts",
        lambda: _selection_with_program(workload_key, program))

    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemv", attributes, _hardware())

    assert launch.algorithm == "cann_matmul"
    assert launch.execution_variant == "gemv"
    assert launch.source_program == program


def test_injected_gemm_missing_physical_key_fails_loudly(monkeypatch):
    shape = {"b": 1, "m": 729, "n": 6144, "k": 2560}
    attributes = _attributes()
    workload_key = gemm_launch_selection_key(
        shape, "bf16", "gemm", attributes)
    monkeypatch.setattr(
        gemm_launch, "load_gemm_launch_selection",
        lambda *args, **kwargs: _selection(workload_key))
    gemm_launch._launch_facts.cache_clear()

    with pytest.raises(KeyError, match="uncharacterized Ascend 910B GEMM"):
        gemm_launch.select_gemm_launch(
            {**shape, "m": 730}, "bf16", "gemm", attributes,
            _hardware())
    gemm_launch._launch_facts.cache_clear()


def test_semantic_context_does_not_change_physical_lookup(monkeypatch):
    shape = {"b": 1, "m": 729, "n": 6144, "k": 2560}
    attributes = _attributes()
    workload_key = gemm_launch_selection_key(
        shape, "bf16", "gemm", attributes)
    monkeypatch.setattr(
        gemm_launch, "load_gemm_launch_selection",
        lambda *args, **kwargs: _selection(workload_key))
    gemm_launch._launch_facts.cache_clear()

    changed_attributes = {
        **attributes,
        "position": "down_proj",
        "tensor_parallel_size": 2,
        "expert_parallel_size": 2,
    }
    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", changed_attributes, _hardware())

    assert dict(launch.tiling.tile_shape) == {"m": 256, "n": 128, "k": 64}
    gemm_launch._launch_facts.cache_clear()


def test_analytical_gemm_derives_capacity_bound_macro_tile():
    shape = {"b": 1, "m": 128, "n": 8192, "k": 28672}
    launch = gemm_launch.select_gemm_launch(
        shape, "bf16", "gemm", {"launch_policy": "analytical_launch"},
        SimpleNamespace(
            num_compute_units=24,
            l2_size_bytes=201_326_592,
        ))

    assert launch.algorithm == "cann_matmul_v3_deterministic_split_k"
    assert launch.grid == (24, 1, 1)
    assert launch.block == (1, 1, 1)
    assert dict(launch.tiling.tile_shape) == {"m": 128, "n": 128, "k": 128}
    assert launch.tiling.pipeline_depth == 2
    assert launch.tiling.split_k == 24
    assert launch.parameters["aiv_mixed"] == 1.0
    assert launch.parameters["split_k_partial_bytes_per_element"] == 4.0
    program = launch.source_program
    assert program is not None
    assert (
        program.single_core_m,
        program.single_core_n,
        program.single_core_k,
    ) == (384, 4096, 384)
    assert (program.depth_a1, program.depth_b1) == (9, 6)
    assert (program.step_m, program.step_n) == (3, 1)
    assert (program.aic_workers, program.aiv_workers) == (24, 48)


@pytest.mark.parametrize(
    "rows,expected",
    (
        (256, (22, 128, 768, 512, 8, 8, 1, 1)),
        (512, (22, 256, 768, 512, 8, 8, 2, 1)),
        (2048, (24, 256, 2816, 512, 8, 8, 2, 1)),
        (4096, (24, 512, 2816, 384, 9, 6, 3, 1)),
        (8192, (24, 1024, 2816, 384, 9, 6, 3, 1)),
        (16384, (24, 2048, 2816, 384, 9, 6, 3, 1)),
        (32768, (24, 4096, 2816, 384, 9, 6, 3, 1)),
    ),
)
def test_analytical_gemm_preserves_captured_single_core_split_k_geometry(
        rows, expected):
    launch = gemm_launch.select_gemm_launch(
        {"b": 1, "m": rows, "n": 8192, "k": 28672},
        "bf16", "gemm", {"launch_policy": "analytical_launch"},
        SimpleNamespace(
            num_compute_units=24,
            l2_size_bytes=201_326_592,
        ))

    program = launch.source_program
    assert launch.algorithm == "cann_matmul_v3_single_core_split_k"
    assert launch.grid == (expected[0], 1, 1)
    assert launch.tiling.split_k == 1
    assert launch.parameters["split_k_partial_bytes_per_element"] == 0.0
    assert program.program_kind == "cann_gemm_single_core_split_k_v1"
    assert (
        program.aic_workers,
        program.single_core_m,
        program.single_core_n,
        program.single_core_k,
        program.depth_a1,
        program.depth_b1,
        program.step_m,
        program.step_n,
    ) == expected
    assert program.scheduler_policy == "single_core_split_k"
    assert program.traversal_order == "n_then_m"
    assert (
        program.a1_buffer_slots,
        program.b1_buffer_slots,
        program.l0a_buffer_slots,
        program.l0b_buffer_slots,
    ) == (3 if program.step_m == 3 else 2, 2, 2, 2)


def test_analytical_gemm_saturates_aics_for_single_row_without_lut(
        monkeypatch):
    monkeypatch.setattr(
        gemm_launch, "_launch_facts",
        lambda: (_ for _ in ()).throw(AssertionError("LUT was read")))
    launch = gemm_launch.select_gemm_launch(
        {"b": 1, "m": 1, "n": 4096, "k": 4096},
        "bf16", "gemm", {"launch_policy": "analytical_launch"},
        SimpleNamespace(
            num_compute_units=24,
            l2_size_bytes=201_326_592,
            aic_mte2_cold_small_m_stride_service=None,
            resource_by_role=lambda role: SimpleNamespace(size_bytes=524288),
        ))

    assert launch.grid == (24, 1, 1)
    assert dict(launch.tiling.tile_shape) == {"m": 16, "n": 176, "k": 64}


def test_asw_macro_tile_orientation_uses_strict_half_l1_capacity():
    assert gemm_launch.derive_cann_matmul_v3_asw_macro_tile(
        1792, 64, 524288) == (256, 128, 64)
    assert gemm_launch.derive_cann_matmul_v3_asw_macro_tile(
        2048, 64, 524288) == (128, 256, 64)


@pytest.mark.parametrize(
    ("k", "expected"),
    (
        (1536, (1, 1, 16, 20)),
        (5120, (1, 1, 16, 20)),
        (8192, (4, 1, 4, 20)),
        (13824, (2, 2, 8, 10)),
    ),
)
def test_matmul_v3_l2_region_schedule_matches_dry_launches(k, expected):
    assert gemm_launch.derive_cann_matmul_v3_l2_region_schedule(
        2048, 5120, k, 128, 256, 24, 201_326_592) == expected


def test_matmul_v3_l2_region_schedule_uses_exact_large_m_coverage():
    assert gemm_launch.derive_cann_matmul_v3_l2_region_schedule(
        32768, 28672, 8192, 128, 256, 24, 201_326_592) == (
            32, 8, 8, 14)


def test_analytical_gemm_uses_capacity_selected_asw_orientation():
    launch = gemm_launch.select_gemm_launch(
        {"b": 1, "m": 2048, "n": 1536, "k": 5120},
        "bf16", "gemm", {"launch_policy": "analytical_launch"},
        SimpleNamespace(
            num_compute_units=24,
            l2_size_bytes=201_326_592,
            aic_mte2_cold_small_m_stride_service=None,
            resource_by_role=lambda role: SimpleNamespace(size_bytes=524288),
        ))

    assert dict(launch.tiling.tile_shape) == {"m": 256, "n": 128, "k": 64}


def test_analytical_gemm_keeps_two_l0c_tiles_for_one_m_tile():
    launch = gemm_launch.select_gemm_launch(
        {"b": 1, "m": 128, "n": 14336, "k": 4096},
        "bf16", "gemm", {"launch_policy": "analytical_launch"},
        SimpleNamespace(
            num_compute_units=24,
            l2_size_bytes=201_326_592,
            aic_mte2_cold_small_m_stride_service=None,
            resource_by_role=lambda role: SimpleNamespace(size_bytes=524288),
        ))

    assert launch.grid == (23, 1, 1)
    assert dict(launch.tiling.tile_shape) == {"m": 128, "n": 128, "k": 128}
    assert launch.source_program.worker_grid_m == 1
    assert launch.source_program.worker_grid_n == 23


@pytest.mark.parametrize(
    ("m_tiles", "n_tiles", "expected"),
    (
        (1, 24, (1, 24)),
        (1, 23, (1, 23)),
        (1, 32, (1, 16)),
        (1, 46, (1, 23)),
        (1, 54, (1, 18)),
        (1, 64, (1, 22)),
        (1, 86, (1, 22)),
        (1, 108, (1, 22)),
        (1, 224, (1, 23)),
        (2, 46, (2, 12)),
        (2, 64, (2, 11)),
        (2, 86, (2, 11)),
        (2, 108, (2, 12)),
        (2, 224, (2, 12)),
        (3, 32, (3, 8)),
        (4, 35, (4, 6)),
        (4, 54, (4, 6)),
        (4, 112, (4, 6)),
        (9, 32, (3, 8)),
    ),
)
def test_persistent_worker_grid_matches_retained_v2_dry_launches(
        m_tiles, n_tiles, expected):
    assert gemm_launch._persistent_worker_grid(
        m_tiles, n_tiles, 24) == expected


def test_persistent_trace_keeps_shifted_redundant_m_tail_worker():
    program = gemm_launch.derive_cann_persistent_launch_program(
        1712, 5120, 4096, 128, 256, 64, _hardware())

    assert (program.worker_grid_m, program.worker_grid_n) == (8, 3)
    assert program.single_core_m == 256
    assert program.single_core_m * (program.worker_grid_m - 1) >= 1712
    trace = cann_gemm_persistent_descriptor_trace(
        program, 1712, 5120, 4096, element_bytes=2)
    assert trace.descriptor_count > 0


@pytest.mark.parametrize(
    ("shape", "tile", "expected_geometry"),
    (
        (
            {"b": 1, "m": 512, "n": 5120, "k": 5120},
            (256, 112, 64),
            (24, 2, 12, 256, 448, 4, 4),
        ),
        (
            {"b": 1, "m": 2220, "n": 4096, "k": 12288},
            (256, 128, 64),
            (24, 3, 8, 768, 512, 4, 4),
        ),
    ),
)
def test_exact_physical_key_normalizes_both_policies_to_one_program(
        monkeypatch, shape, tile, expected_geometry):
    attributes = _attributes()
    workload_key = gemm_launch_selection_key(
        shape, "bf16", "gemm", attributes)
    monkeypatch.setattr(
        gemm_launch, "load_gemm_launch_selection",
        lambda *args, **kwargs: _selection(workload_key, tile))
    gemm_launch._launch_facts.cache_clear()

    launches = tuple(
        gemm_launch.select_gemm_launch(
            shape, "bf16", "gemm", {**attributes, "launch_policy": policy},
            _hardware())
        for policy in ("analytical_launch", "injected_launch")
    )

    assert launches[0] == launches[1]
    program = launches[0].source_program
    assert program is not None
    assert (
        program.aic_workers,
        program.worker_grid_m,
        program.worker_grid_n,
        program.single_core_m,
        program.single_core_n,
        program.a_l1_k_tiles,
        program.b_l1_k_tiles,
    ) == expected_geometry
    assert program.traversal_order == "persistent_n_then_m"
    assert program.a_uses_l2 is True
    assert program.b_uses_l2 is True
    gemm_launch._launch_facts.cache_clear()
