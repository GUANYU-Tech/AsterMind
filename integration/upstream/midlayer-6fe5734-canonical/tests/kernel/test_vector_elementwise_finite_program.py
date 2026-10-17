"""Exact finite-program coverage for CANN BF16 elementwise Mul."""

from src.kernel.formula import PerformanceFormula
from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config


def _resource_input(elements):
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    shape = {"element_count": elements}
    attributes = {"reads": 2, "writes": 1, "operation": "mul"}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
        "elementwise", {**attributes, "launch_policy": "injected_launch"},
        hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant=launch.execution_variant, launch=launch,
        attributes=attributes)
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    return hardware, launch, resource_input


def test_q4q1_five_tile_worker_classes_use_exact_finite_program():
    hardware, launch, resource_input = _resource_input(256 * 8192)

    assert launch.grid == (48, 1, 1)
    assert launch.parameters["cann_tiling_key"] == 210_020_000.0
    assert launch.parameters["elementwise_block_factor"] == 43_776.0
    assert launch.parameters["elementwise_ub_factor"] == 8_832.0
    program, = resource_input.pipeline_plan.tasks
    task, = program.patterns[0].tasks
    finite_stage = next(
        stage for stage in task.stages
        if stage.name == "finite_memory_program")
    exact_completion, = finite_stage.memory_exact_request_demands
    assert exact_completion.service == (
        "aiv_mul_q4q1_finite_worker_completion_48_workers_17664b_x5")
    assert exact_completion.request_sizes_by_worker == ((1,),)

    prediction = PerformanceFormula(hardware).predict_kernel(resource_input)
    assert prediction.t_total_s > 0.0
    assert prediction.t_launch_s * 1e9 >= 1809.5782470703125


def test_q4q1_five_tile_service_requires_exact_tail_classes():
    _, launch, resource_input = _resource_input(256 * 8192 + 32)

    assert launch.parameters["elementwise_ub_factor"] == 8_832.0
    assert all(
        stage.name != "finite_memory_program"
        for program in resource_input.pipeline_plan.tasks
        for pattern in program.patterns
        for task in pattern.tasks
        for stage in task.stages)
