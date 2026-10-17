"""CANN 8.5.1 RMSNorm MERGE_N (tiling key 2) launch and worker contract."""

import math

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.model_contract import (
    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
    AIV_WORKER_START_FILL,
)


HEAD_DIM = 128
VECTOR_CORES = 48


def _hardware():
    return HWSpec("configs/hardware/ascend_910b.yaml")


def _grouped_attributes(outer_groups, rows_per_group):
    return {
        "input_materialization": "grouped_strided_to_contiguous",
        "input_outer_groups": outer_groups,
        "input_rows_per_group": rows_per_group,
        "input_group_stride_elements": 48 * HEAD_DIM,
        "input_row_stride_elements": HEAD_DIM,
        "launch_policy": "injected_launch",
    }


def _launch(hardware, rows, hidden, attributes):
    return select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR,
        {"rows": rows, "hidden": hidden}, "bf16", "rmsnorm",
        attributes, hardware=hardware)


def _resource_input(hardware, rows, hidden, attributes):
    launch = _launch(hardware, rows, hidden, attributes)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id,
        {"rows": rows, "hidden": hidden}, dtype="bf16",
        variant=launch.execution_variant, launch=launch, attributes={})
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    return launch, resource_input


def test_small_reduce_columns_select_merge_n_and_wide_columns_do_not():
    """``rms_norm_tiling.cpp:524-526`` gates MERGE_N on numColAlign <= 2000."""

    hardware = _hardware()
    merge_n = _launch(
        hardware, 2442 * 32, HEAD_DIM, _grouped_attributes(2442, 32))
    assert merge_n.parameters["cann_tiling_key"] == 2.0
    assert merge_n.parameters["source_bound_rmsnorm_merge_n_dag"] == 1.0
    wide = _launch(
        hardware, 2442, 4096, {"launch_policy": "injected_launch"})
    assert wide.parameters["cann_tiling_key"] == 0.0
    assert "source_bound_rmsnorm_merge_n_dag" not in wide.parameters


@pytest.mark.parametrize(
    ("hidden", "expected_key"),
    (
        (1992, 2.0),   # numColAlign 2000, exactly at SMALL_REDUCE_NUM
        (2000, 2.0),   # numColAlign 2000
        (2001, 0.0),   # numColAlign 2016, one block past the bound
        (12289, 1.0),  # above the full-row UB bound, Split-D
    ),
)
def test_merge_n_selection_boundary_is_the_aligned_column_count(
        hidden, expected_key):
    """``rms_norm_tiling.cpp:524`` compares numColAlign, not numCol."""

    hardware = _hardware()
    launch = _launch(
        hardware, 4096, hidden, {"launch_policy": "injected_launch"})
    assert launch.parameters["cann_tiling_key"] == expected_key


def test_merge_n_row_factor_follows_the_host_tiler_ub_division():
    """``rms_norm_tiling.cpp:543`` over the ``UB_USED`` remainder of UB."""

    hardware = _hardware()
    unified_buffer_bytes = int(hardware.resource_by_role("ub").size_bytes)
    expected = (unified_buffer_bytes - 1024) // (HEAD_DIM * 16 + 260)
    launch = _launch(
        hardware, 2442 * 32, HEAD_DIM, _grouped_attributes(2442, 32))
    assert launch.parameters["norm_merge_n_row_factor"] == float(expected)


@pytest.mark.parametrize(
    ("outer_groups", "rows_per_group", "performance"),
    (
        (2442, 32, 0.0),   # second dimension above eight
        (2442, 8, 0.0),    # leading dimension above 512
        (4, 8, 1.0),       # both source bounds satisfied
        (512, 8, 1.0),     # leading dimension exactly at the bound
        (513, 8, 0.0),     # one row past the bound
    ),
)
def test_merge_n_performance_flag_follows_the_source_extent_predicate(
        outer_groups, rows_per_group, performance):
    """``rms_norm_tiling.cpp:503-507`` reads only rank and leading extents."""

    hardware = _hardware()
    launch = _launch(
        hardware, outer_groups * rows_per_group, HEAD_DIM,
        _grouped_attributes(outer_groups, rows_per_group))
    assert launch.parameters["norm_merge_n_performance_flag"] == performance


def test_merge_n_launch_geometry_matches_the_host_tiler():
    """``rms_norm_tiling.cpp:510-516`` blockFactor/useCoreNum division."""

    hardware = _hardware()
    for outer_groups, rows_per_group in ((1, 32), (63, 32), (2442, 8)):
        rows = outer_groups * rows_per_group
        launch = _launch(
            hardware, rows, HEAD_DIM,
            _grouped_attributes(outer_groups, rows_per_group))
        block_factor = math.ceil(rows / VECTOR_CORES)
        assert launch.parameters["norm_block_factor"] == float(block_factor)
        assert launch.grid == (math.ceil(rows / block_factor), 1, 1)


def test_merge_n_rstd_transfer_is_charged_once_per_row_group():
    """``CopyOutRstd`` transfers one group, not one AIV extent."""

    hardware = _hardware()
    rows = 2442 * 32
    launch = _launch(
        hardware, rows, HEAD_DIM, _grouped_attributes(2442, 32))
    row_factor = int(launch.parameters["norm_merge_n_row_factor"])
    block_factor = int(launch.parameters["norm_block_factor"])
    workers = launch.grid[0]
    rows_by_worker = (
        (block_factor,) * (workers - 1)
        + (rows - (workers - 1) * block_factor,))
    expected = 0
    for worker_rows in rows_by_worker:
        row_loop = math.ceil(worker_rows / row_factor)
        tail = worker_rows - (row_loop - 1) * row_factor
        for group_rows in (row_factor,) * (row_loop - 1) + (tail,):
            expected += math.ceil(group_rows / 8) * 32
    assert launch.parameters["cann_rmsnorm_rstd_transfer_bytes"] == float(
        expected)


@pytest.mark.parametrize(
    ("outer_groups", "rows_per_group"), ((2442, 32), (2442, 8), (4, 8)))
def test_merge_n_worker_program_conserves_and_closes_its_queues(
        outer_groups, rows_per_group):
    """The typed services conserve the aggregate and the DAG is attached."""

    hardware = _hardware()
    rows = outer_groups * rows_per_group
    _, resource_input = _resource_input(
        hardware, rows, HEAD_DIM,
        _grouped_attributes(outer_groups, rows_per_group))
    assert resource_input.pipeline_plan is not None
    assert {buffer.name for buffer in resource_input.pipeline_plan.buffers} == {
        "rms_merge_n_x_queue",
        "rms_merge_n_y_queue",
        "rms_merge_n_rstd_queue",
    }
    assert resource_input.task_duration_floor_contract == (
        AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM)
    assert resource_input.aiv_worker_start_contract == AIV_WORKER_START_FILL
    vector_demand, = tuple(
        demand for demand in resource_input.compute_demands
        if demand.resource_role == "compute_vector")
    typed = math.fsum(
        demand.operations for demand in vector_demand.service_demands)
    exact = math.fsum(
        demand.operations
        for demand in vector_demand.exact_request_demands)
    assert typed + exact == vector_demand.operations


def test_merge_n_reduces_the_vector_demand_of_the_performance_branch():
    """``is_performance`` removes one full-length output pass and gamma
    broadcast (``rms_norm_merge_n.h:88-90``, ``:163-170``)."""

    hardware = _hardware()
    general, _ = _resource_input(
        hardware, 513 * 8, HEAD_DIM, _grouped_attributes(513, 8))
    performance, _ = _resource_input(
        hardware, 512 * 8, HEAD_DIM, _grouped_attributes(512, 8))
    assert general.parameters["norm_merge_n_performance_flag"] == 0.0
    assert performance.parameters["norm_merge_n_performance_flag"] == 1.0
    assert (
        performance.parameters["lowered_vector_output_operations"]
        < general.parameters["lowered_vector_output_operations"])
