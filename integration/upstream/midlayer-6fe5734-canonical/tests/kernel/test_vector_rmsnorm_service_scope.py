import json
from pathlib import Path

from src.kernel.hardware_spec import HWSpec
from src.kernel.kernel_base import create_predictor
from src.kernel.kernel_spec import KernelKind, KernelSpec
from src.kernel.launch_config.select_launch_config import select_launch_config
from src.kernel.predictors.vector import (
    _has_typed_rmsnorm_scalar_round_services,
)


def _rmsnorm_input(hardware_id: str, rows: int, hidden: int):
    hardware = HWSpec(f"configs/hardware/{hardware_id}.yaml")
    shape = {"rows": rows, "hidden": hidden}
    launch = select_launch_config(
        hardware_id, KernelKind.VECTOR, shape, "bf16", "rmsnorm",
        {"launch_policy": "injected_launch"},
        hardware=hardware,
    )
    kernel = KernelSpec(
        KernelKind.VECTOR,
        hardware_id,
        shape,
        dtype="bf16",
        variant=launch.execution_variant,
        launch=launch,
        attributes={},
    )
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    return resource_input, _has_typed_rmsnorm_scalar_round_services(
        kernel, hardware)


def _add_rmsnorm_input(rows: int, hidden: int):
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    shape = {"rows": rows, "hidden": hidden}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
        "add_rmsnorm", {}, hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant=launch.execution_variant, launch=launch, attributes={})
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    return resource_input


def _c500_add_rmsnorm_input(rows: int, hidden: int):
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    shape = {"rows": rows, "hidden": hidden}
    launch = select_launch_config(
        hardware.hardware_id, KernelKind.VECTOR, shape, "bf16",
        "add_rmsnorm", {"launch_policy": "injected_launch"},
        hardware=hardware)
    kernel = KernelSpec(
        KernelKind.VECTOR, hardware.hardware_id, shape, dtype="bf16",
        variant=launch.execution_variant, launch=launch, attributes={})
    resource_input, _ = create_predictor(kernel.kind).resource_input(
        kernel, hardware)
    return resource_input


def _vector_services(resource_input):
    demand = next(
        (demand for demand in resource_input.compute_demands
         if demand.resource_role == "compute_vector"),
        None,
    )
    if demand is None:
        return set()
    return {service.service for service in demand.service_demands}


def test_a800_grid64_composite_probe_is_not_a_production_stage_service():
    a800_input, a800_typed = _rmsnorm_input(
        "nvidia_a800", rows=64, hidden=16384)
    c500_input, c500_typed = _rmsnorm_input(
        "metax_c500", rows=64, hidden=8192)
    a800_services = _vector_services(a800_input)
    c500_services = _vector_services(c500_input)

    assert not a800_typed
    assert c500_typed
    assert not any("rmsnorm_" in service for service in a800_services)
    assert not any("phase_entry" in service for service in c500_services)


def test_a800_small_gamma_persists_for_the_complete_launch():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    for hidden, expected_copies in ((2048, 108.0), (8192, 540.0)):
        shape = {"rows": 1024, "hidden": hidden}
        launch = select_launch_config(
            hardware.hardware_id,
            KernelKind.VECTOR,
            shape,
            "bf16",
            "rmsnorm",
            {"launch_policy": "injected_launch"},
            hardware=hardware,
        )
        assert launch.parameters["norm_weight_l2_copies"] == expected_copies


def test_a800_add_rmsnorm_grid_capture_is_validation_only():
    service = "add_rmsnorm_h4096_grid64_cold_source_chain_entry"
    exact = _add_rmsnorm_input(rows=64, hidden=4096)
    other_hidden = _add_rmsnorm_input(rows=64, hidden=5120)
    other_grid = _add_rmsnorm_input(rows=16, hidden=4096)

    assert service not in _vector_services(exact)
    assert service not in _vector_services(other_hidden)
    assert service not in _vector_services(other_grid)


def test_c500_block256_add_rmsnorm_uses_scalar_input_source_depth_only():
    scalar_services = {
        "add_rmsnorm_scalar_input_add_square_first_256",
        "add_rmsnorm_scalar_input_add_square_resident_256",
    }
    exact = _c500_add_rmsnorm_input(rows=256, hidden=2048)
    other_block = _c500_add_rmsnorm_input(rows=64, hidden=2048)
    queued_grid = _c500_add_rmsnorm_input(rows=1024, hidden=2048)

    assert scalar_services <= _vector_services(exact)
    assert not scalar_services & _vector_services(other_block)
    assert not scalar_services & _vector_services(queued_grid)
    demand = next(
        demand for demand in exact.compute_demands
        if demand.resource_role == "compute_vector")
    scalar_operations = sum(
        item.operations for item in demand.service_demands
        if item.service in scalar_services)
    assert scalar_operations == 256.0 * 8.0
    assert "bf16_scale_multiply_first_256" in _vector_services(exact)
    assert not any(
        name.endswith("phase_chain_complete_first_1024")
        for name in _vector_services(exact))


def test_c500_generic_block1024_add_rmsnorm_prices_static_prologue_once():
    block1024 = _c500_add_rmsnorm_input(rows=64, hidden=5120)
    block256 = _c500_add_rmsnorm_input(rows=256, hidden=2048)

    issue, = (
        demand for demand in block1024.compute_demands
        if demand.resource_role == "warp_issue")
    program, = block1024.pipeline_plan.tasks
    schedule, = program.patterns[0].tasks
    prologue, = (
        stage for stage in schedule.stages
        if stage.name == "cta0_source_prologue")

    assert issue.operations == 64.0 * 16.0 * 46.0
    assert prologue.work_fraction == 1.0 / 64.0
    assert not any(
        demand.resource_role == "warp_issue"
        for demand in block256.compute_demands)


def test_c500_grid64_add_rmsnorm_uses_exact_fixed_input_loop_floor():
    for depth in (4, 5):
        service = (
            f"add_rmsnorm_grid64_fixed_input_loop_depth{depth}_1024")
        exact = _c500_add_rmsnorm_input(rows=64, hidden=depth * 1024)
        services = _vector_services(exact)
        assert service in services
        assert not any(name.startswith("bf16_add_square_first_")
                       for name in services)

        program, = exact.pipeline_plan.tasks
        schedule, = program.patterns[0].tasks
        fixed_input, = (
            stage for stage in schedule.stages
            if stage.name == "cta0_fixed_input_loop")
        prologue, = (
            stage for stage in schedule.stages
            if stage.name == "cta0_source_prologue")
        reduction, = (
            stage for stage in schedule.stages
            if stage.name == "cta0_block_reduction")
        assert fixed_input.predecessors == ()
        assert prologue.predecessors == ()
        assert set(reduction.predecessors) == {
            fixed_input.name, prologue.name}


def test_c500_grid64_add_rmsnorm_uses_complete_source_phase_chain():
    service = "add_rmsnorm_phase_chain_complete_first_1024"
    fixed_prefix = "add_rmsnorm_grid64_fixed_input_loop_depth"
    for depth in (8, 16, 32):
        exact = _c500_add_rmsnorm_input(rows=64, hidden=depth * 1024)
        services = _vector_services(exact)
        assert service in services
        assert not any(name.startswith(fixed_prefix) for name in services)
        assert not any("phase_chain_excess" in name for name in services)

        program, = exact.pipeline_plan.tasks
        schedule, = program.patterns[0].tasks
        stages = {stage.name: stage for stage in schedule.stages}
        complete = stages["cta0_complete_source_phase_chain"]
        reduction = stages["cta0_block_reduction"]
        assert complete.compute_service_demands[0].service == service
        assert complete.compute_service_demands[0].operations == float(depth)
        assert complete.predecessors == ("cta0_source_prologue",)
        assert reduction.predecessors == (complete.name,)


def test_c500_complete_source_phase_chain_stays_in_measured_domain():
    service = "add_rmsnorm_phase_chain_complete_first_1024"
    shallow = _c500_add_rmsnorm_input(rows=64, hidden=7 * 1024)
    deep = _c500_add_rmsnorm_input(rows=64, hidden=33 * 1024)
    queued_grid = _c500_add_rmsnorm_input(rows=105, hidden=16 * 1024)

    assert service not in _vector_services(shallow)
    assert service not in _vector_services(deep)
    assert service not in _vector_services(queued_grid)


def test_c500_grid64_fixed_input_loop_stays_in_measured_domain():
    service_prefix = "add_rmsnorm_grid64_fixed_input_loop_depth"
    other_grid = _c500_add_rmsnorm_input(rows=32, hidden=16384)
    other_depth = _c500_add_rmsnorm_input(rows=64, hidden=12288)

    assert not any(name.startswith(service_prefix)
                   for name in _vector_services(other_grid))
    assert not any(name.startswith(service_prefix)
                   for name in _vector_services(other_depth))


def test_c500_grid64_fixed_input_loop_rates_match_reference():
    reference = json.loads(Path(
        "microbench/metax_c500/references/"
        "add_rmsnorm_grid64_fixed_input_loop_maca_3_7_2.json"
    ).read_text())
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    resource = hardware.resource_by_role("compute_vector")

    assert not reference["production_or_evaluation_duration_consumed"]
    assert reference["stability_gate"]["passed"]
    for selected in reference["selected_services"]:
        service = (
            "add_rmsnorm_grid64_fixed_input_loop_depth"
            f"{selected['source_depth']}_1024")
        assert resource.service_rate_for_one_unit(service) == (
            selected["service_rate_per_unit_s"])
