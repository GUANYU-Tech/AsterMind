"""Exact finite-source envelope coverage for CANN BF16 RMSNorm key 0."""

import pytest

from src.kernel.formula import PerformanceFormula
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.model_contract import (
    AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM,
    AIV_WORKER_START_FILL,
)


def _resource_input(rows, hidden):
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    shape = {"rows": rows, "hidden": hidden}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16", "rmsnorm",
        {"launch_policy": "injected_launch"}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant=launch.execution_variant, launch=launch, attributes={})
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    return hardware, launch, resource_input


@pytest.mark.parametrize(
    ("rows", "hidden", "workers"),
    (
        (2, 8192, 2),
        (4, 8192, 4),
    ),
)
def test_key0_single_row_exact_domain_keeps_source_envelope_as_floor(
        rows, hidden, workers):
    hardware, launch, resource_input = _resource_input(rows, hidden)

    assert launch.grid == (workers, 1, 1)
    assert launch.parameters["cann_tiling_key"] == 0.0
    assert resource_input.task_duration_floor_contract == (
        AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM)
    assert resource_input.aiv_worker_start_contract == AIV_WORKER_START_FILL
    assert resource_input.competing_pipeline_floor is not None
    program, = resource_input.competing_pipeline_floor.tasks
    assert program.worker_multiplicity == workers
    assert program.physical_workers == tuple(range(workers))
    task, = program.patterns[0].tasks
    finite_stage = next(
        stage for stage in task.stages
        if stage.name == "finite_source_envelope")
    completion, = finite_stage.memory_exact_request_demands
    assert completion.service == "aiv_rmsnorm_key0_finite_source_envelope"
    assert completion.request_sizes_by_worker == ((workers,),)
    prediction = PerformanceFormula(hardware).predict_kernel(resource_input)
    assert prediction.diagnostics.pipeline.execution_time_s * 1e9 >= (
        hardware.resource_by_role("mte3").exact_request_service_time_s(
            completion.service, workers) * 1e9)
    assert prediction.t_total_ns > 0.0


@pytest.mark.parametrize(("rows", "hidden"), ((64, 4096), (8, 8192)))
def test_key0_row_group_program_does_not_use_finite_launch_floor(
        rows, hidden):
    _, _, resource_input = _resource_input(rows, hidden)

    assert resource_input.task_duration_floor_contract == (
        AIV_EMPTY_TASK_SHELL_PLUS_SOURCE_PROGRAM)
    assert resource_input.aiv_worker_start_contract == AIV_WORKER_START_FILL
    assert resource_input.competing_pipeline_floor is None
    assert all(
        stage.name != "finite_source_envelope"
        for task in resource_input.pipeline_plan.tasks
        for stage in task.stages)
