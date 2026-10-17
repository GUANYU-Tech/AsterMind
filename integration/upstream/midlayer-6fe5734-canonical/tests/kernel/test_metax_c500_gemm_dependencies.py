"""Focused MetaX C500 GEMM dependency-path tests."""

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config import LaunchConfig, TilingSpec
from src.kernel.model_contract import (
    PipelineTaskProgramDemand,
    PipelineWorkPlan,
)


METAX_HARDWARE = HWSpec("configs/hardware/metax_c500.yaml")


def _single_buffer_partial_m_kernel(*, b: int, k: int) -> KernelSpec:
    return KernelSpec(
        KernelKind.GEMM,
        METAX_HARDWARE.hardware_id,
        {"b": b, "m": 1, "n": 5120, "k": k, "dtype_bytes": 2},
        dtype="bf16",
        variant="gemm",
        launch=LaunchConfig(
            algorithm="observed_mcblas_gemm:gemm",
            grid=(1, 80, 1),
            block=(256, 1, 1),
            tiling=TilingSpec(
                tile_shape={"m": 32, "n": 64, "k": 256},
                split_k=1,
            ),
            parameters={
                "compulsory_traffic": 0.0,
                "weight_tile_rows": 0.0,
                "mte2_tile_bytes": 0.0,
                "aiv_mixed": 0.0,
                "n_major_traffic": 0.0,
                "split_k_partial_bytes_per_element": 0.0,
                "split_k_partial_dram_bytes_per_element": 0.0,
            },
            execution_variant="gemm",
        ),
    )


@pytest.mark.parametrize("b", (2, 8))
def test_short_partial_m_tile_drains_both_operand_streams(b):
    resource_input, _ = create_predictor(
        KernelKind.GEMM).resource_input(
            _single_buffer_partial_m_kernel(b=b, k=1536), METAX_HARDWARE)

    assert resource_input.dependency_paths == ()
    assert isinstance(resource_input.pipeline_plan, PipelineWorkPlan)
    program, = resource_input.pipeline_plan.tasks
    assert isinstance(program, PipelineTaskProgramDemand)
    sequence, = program.patterns
    first, steady, store = sequence.tasks
    assert first.repeat_count + steady.repeat_count == 4 * 1536 // 256
    assert tuple(stage.service for stage in first.stages) == (
        "memory:global_read",
        "resource:smem",
        "compute:compute_tensor",
    )
    assert store.stages[0].service == "memory:global_write"


@pytest.mark.parametrize(
    ("b", "k"),
    ((2, 5120), (8, 13824)),
)
def test_deep_partial_tile_uses_rolling_single_buffer_pipeline(b, k):
    resource_input, _ = create_predictor(
        KernelKind.GEMM).resource_input(
            _single_buffer_partial_m_kernel(b=b, k=k), METAX_HARDWARE)

    assert resource_input.dependency_paths == ()
    assert isinstance(resource_input.pipeline_plan, PipelineWorkPlan)
    program, = resource_input.pipeline_plan.tasks
    assert isinstance(program, PipelineTaskProgramDemand)
    sequence, = program.patterns
    first, steady, store = sequence.tasks
    assert first.repeat_count + steady.repeat_count == 4 * k // 256
    assert tuple(stage.service for stage in first.stages) == (
        "memory:global_read",
        "resource:smem",
        "compute:compute_tensor",
    )
    assert store.stages[0].service == "memory:global_write"


@pytest.mark.parametrize("k", (1536, 13824))
def test_complete_32x64x256_tile_uses_rolling_single_buffer_pipeline(k):
    resource_input, _ = create_predictor(
        KernelKind.GEMM).resource_input(
            _single_buffer_partial_m_kernel(b=32, k=k), METAX_HARDWARE)

    assert resource_input.dependency_paths == ()
    assert isinstance(resource_input.pipeline_plan, PipelineWorkPlan)
    program, = resource_input.pipeline_plan.tasks
    assert isinstance(program, PipelineTaskProgramDemand)
    sequence, = program.patterns
    first, steady, store = sequence.tasks
    assert first.repeat_count + steady.repeat_count == 4 * k // 256
    assert tuple(stage.service for stage in first.stages) == (
        "memory:global_read",
        "resource:smem",
        "compute:compute_tensor",
    )
    assert store.stages[0].service == "memory:global_write"
