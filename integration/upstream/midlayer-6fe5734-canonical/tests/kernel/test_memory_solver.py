import math

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.memory_model import (
    memory_hierarchy_for_path,
    solve_memory_access_classes,
    solve_memory_arbitration,
    solve_memory_path,
)
from src.kernel.model_contract import (
    MemoryAccessClassDemand,
    MemoryAccessClassService,
    MemoryAccessClassSignature,
    MemoryArbitrationDiagnostic,
    MemoryHierarchy,
    MemoryNodeDemand,
    MemoryPathDemand,
    MemoryService,
    MemorySourceDemand,
    QueueCapacity,
    QueueUse,
)


def _access_signature(*, tile_m=64, tile_k=32, row_bytes=64):
    return MemoryAccessClassSignature(
        access_class="indirect_row_gather",
        dtype="bf16",
        tile_m=tile_m,
        tile_k=tile_k,
        logical_row_bytes=row_bytes,
    )


def _access_path(name, logical_bytes, *, signature=None, ingress_bytes=1000.0):
    if signature is None:
        signature = _access_signature()
    return MemoryPathDemand(
        name=name,
        access_kind="read",
        initial_transactions=ingress_bytes / 128.0,
        nodes=(MemoryNodeDemand(
            "l2_cache", ingress_bytes / 128.0, ingress_bytes, 128.0),),
        source=MemorySourceDemand(
            active_units=1,
            producers_per_unit=1.0,
            independent_requests_per_producer=None,
            frontend_rate_per_unit_s=None,
        ),
        access_class_demands=(MemoryAccessClassDemand(
            service="indirect_row_gather",
            ingress_resource_role="l2_cache",
            logical_bytes=logical_bytes,
            signature=signature,
        ),),
    )


def _access_service(*, signature=None, capacity=100.0):
    if signature is None:
        signature = _access_signature()
    return MemoryAccessClassService(
        service="indirect_row_gather",
        access_kind="read",
        ingress_resource_role="l2_cache",
        signature=signature,
        scope="full_chip",
        logical_byte_capacity_per_s=capacity,
        evidence_ref="independent_mechanism.tsv#signature",
    )


def test_access_class_solver_uses_exact_signature_and_full_chip_capacity():
    paths = (
        _access_path("read_a", 400.0),
        _access_path("read_b", 600.0),
    )

    diagnostics = solve_memory_access_classes(
        paths, (_access_service(capacity=100.0),))

    assert tuple(value.service_time_s for value in diagnostics) == (4.0, 6.0)
    assert math.fsum(
        value.service_time_s for value in diagnostics) == 10.0
    assert all(value.scope == "full_chip" for value in diagnostics)


def test_access_class_capacity_is_read_at_the_launch_grid():
    """A swept capacity is read at the launch's grid, never extrapolated."""

    service = MemoryAccessClassService(
        service="indirect_row_gather",
        access_kind="read",
        ingress_resource_role="l2_cache",
        signature=_access_signature(),
        scope="full_chip",
        logical_byte_capacity_per_s=100.0,
        evidence_ref="independent_mechanism.tsv#signature",
        logical_byte_capacity_per_s_by_grid_ctas=(
            (108, 100.0), (216, 180.0), (1024, 200.0)),
    )
    paths = (_access_path("read_a", 400.0),)

    def capacity(grid):
        diagnostic, = solve_memory_access_classes(
            paths, (service,), grid_size=grid)
        return diagnostic.logical_byte_capacity_per_s

    # Below the first measured grid the entry capacity stands; at or above a
    # measured grid that measurement does; past the sweep the last point is
    # held rather than projected.
    assert capacity(None) == 100.0
    assert capacity(64) == 100.0
    assert capacity(108) == 100.0
    assert capacity(215) == 100.0
    assert capacity(216) == 180.0
    assert capacity(1023) == 180.0
    assert capacity(1024) == 200.0
    assert capacity(4096) == 200.0
    # The service time follows the resolved capacity, not the entry one.
    diagnostic, = solve_memory_access_classes(
        paths, (service,), grid_size=1024)
    assert diagnostic.service_time_s == 2.0


def test_access_class_measured_capacity_table_must_be_ordered_and_positive():
    for table in (
            ((216, 180.0), (108, 100.0)),   # descending grids
            ((108, 100.0), (108, 180.0)),   # repeated grid
            ((108, 0.0),),                  # non-positive capacity
            (),                             # empty
    ):
        with pytest.raises((ValueError, TypeError)):
            MemoryAccessClassService(
                service="indirect_row_gather",
                access_kind="read",
                ingress_resource_role="l2_cache",
                signature=_access_signature(),
                scope="full_chip",
                logical_byte_capacity_per_s=100.0,
                evidence_ref="independent_mechanism.tsv#signature",
                logical_byte_capacity_per_s_by_grid_ctas=table,
            )


def test_access_class_solver_rejects_signature_extrapolation():
    demand = _access_path("read", 400.0)
    different_signature = _access_signature(
        tile_m=16, tile_k=64, row_bytes=128)

    with pytest.raises(ValueError, match="no exact hardware signature"):
        solve_memory_access_classes(
            (demand,), (_access_service(signature=different_signature),))


def test_access_class_demand_must_be_a_subset_of_the_path_ingress():
    with pytest.raises(ValueError, match="cannot exceed ingress bytes"):
        _access_path("read", 1001.0, ingress_bytes=1000.0)


def test_access_class_demand_must_bind_the_first_path_node():
    signature = _access_signature()
    with pytest.raises(ValueError, match="must bind the first ingress"):
        MemoryPathDemand(
            name="read",
            access_kind="read",
            initial_transactions=1.0,
            nodes=(MemoryNodeDemand("l2_cache", 1.0, 128.0, 128.0),),
            source=MemorySourceDemand(1, 1.0, None, None),
            access_class_demands=(MemoryAccessClassDemand(
                service="indirect_row_gather",
                ingress_resource_role="dram",
                logical_bytes=64.0,
                signature=signature,
            ),),
        )


def test_memory_node_parallel_streams_require_and_respect_capacity_ceiling():
    with pytest.raises(ValueError, match="capacity ceiling"):
        MemoryNodeDemand(
            "l2_cache", 1.0, 128.0, 128.0,
            independent_service_streams=2,
        )

    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    node = MemoryNodeDemand(
        "l2_cache", 1024.0, 32_768.0, 32.0,
        independent_service_streams=2,
        capacity_ceiling_resource_role="xbar",
    )
    path = MemoryPathDemand(
        name="parallel_l2_read",
        access_kind="read",
        initial_transactions=1024.0,
        nodes=(node,),
        source=MemorySourceDemand(108, 1.0, None, None),
    )

    hierarchy = memory_hierarchy_for_path(
        hardware,
        path,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        dram_capacity_multiplier=1.0,
    )

    l2_capacity = hardware.resource_by_role("l2_cache").effective_capacity(
        108,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
    )
    xbar_capacity = hardware.resource_by_role("xbar").effective_capacity(
        108,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
    )
    service, = hierarchy.services
    assert service.byte_capacity_per_s == min(2 * l2_capacity, xbar_capacity)


def test_npu_memory_node_parallel_streams_respect_capacity_ceiling():
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    node = MemoryNodeDemand(
        "mte2", None, 32_768.0, None,
        independent_service_streams=2,
        capacity_ceiling_resource_role="l2_cache",
    )
    path = MemoryPathDemand(
        name="parallel_mte2_read",
        access_kind="read",
        initial_transactions=None,
        nodes=(node,),
        source=MemorySourceDemand(24, 1.0, None, None),
    )

    hierarchy = memory_hierarchy_for_path(
        hardware,
        path,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        dram_capacity_multiplier=1.0,
    )

    mte2_capacity = hardware.resource_by_role("mte2").effective_capacity(
        24,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
    )
    l2_capacity = hardware.resource_by_role("l2_cache").effective_capacity(
        24,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
    )
    mte2_units = hardware.resource_by_role("mte2").total_units
    active_mte2_capacity = (
        mte2_capacity * min(24, mte2_units) / mte2_units)
    service, = hierarchy.services
    assert service.byte_capacity_per_s == min(
        2 * active_mte2_capacity, l2_capacity)


def test_access_class_path_accepts_one_service_with_distinct_signatures():
    first = _access_signature(tile_m=64, tile_k=32, row_bytes=4160)
    second = _access_signature(tile_m=64, tile_k=32, row_bytes=8192)
    path = MemoryPathDemand(
        name="read",
        access_kind="read",
        initial_transactions=1000.0 / 128.0,
        nodes=(MemoryNodeDemand(
            "l2_cache", 1000.0 / 128.0, 1000.0, 128.0),),
        source=MemorySourceDemand(1, 1.0, None, None),
        access_class_demands=(
            MemoryAccessClassDemand(
                service="indirect_row_gather",
                ingress_resource_role="l2_cache",
                logical_bytes=400.0,
                signature=first,
            ),
            MemoryAccessClassDemand(
                service="indirect_row_gather",
                ingress_resource_role="l2_cache",
                logical_bytes=600.0,
                signature=second,
            ),
        ),
    )
    diagnostics = solve_memory_access_classes(
        (path,),
        (
            _access_service(signature=first, capacity=100.0),
            _access_service(signature=second, capacity=200.0),
        ),
    )

    assert tuple(value.service_time_s for value in diagnostics) == (4.0, 3.0)


def test_access_class_path_rejects_repeated_exact_identity():
    signature = _access_signature()
    demand = MemoryAccessClassDemand(
        service="indirect_row_gather",
        ingress_resource_role="l2_cache",
        logical_bytes=400.0,
        signature=signature,
    )
    with pytest.raises(ValueError, match="repeat an exact"):
        MemoryPathDemand(
            name="read",
            access_kind="read",
            initial_transactions=1000.0 / 128.0,
            nodes=(MemoryNodeDemand(
                "l2_cache", 1000.0 / 128.0, 1000.0, 128.0),),
            source=MemorySourceDemand(1, 1.0, None, None),
            access_class_demands=(demand, demand),
        )


def _path(*, source_slots=200.0, queues=()):
    node = MemoryNodeDemand(
        resource_role="l2_cache",
        transactions=1000.0,
        bytes=128000.0,
        transaction_bytes=128.0,
    )
    return MemoryPathDemand(
        name="read",
        access_kind="read",
        initial_transactions=1000.0,
        nodes=(node,),
        source=MemorySourceDemand(
            active_units=1,
            producers_per_unit=1.0,
            independent_requests_per_producer=source_slots,
            frontend_rate_per_unit_s=None,
        ),
        queues=queues,
    )


def _hierarchy(*, source_slots=200.0, request_capacity=None,
               queue_model="none", service_time=None, queues=()):
    return MemoryHierarchy(
        services=(MemoryService(
            resource_role="l2_cache",
            byte_capacity_per_s=128e9,
            request_capacity_per_s=request_capacity,
            unloaded_latency_s=100e-9,
            queue_model=queue_model,
            service_time_s=service_time,
        ),),
        frontend_rate_per_unit_s=None,
        source_slots_per_unit=source_slots,
        queues=queues,
    )


def test_capacity_branch_returns_exact_service_capacity_without_epsilon():
    result = solve_memory_path(_path(), _hierarchy())

    assert result.capacity_bound
    assert result.lambda_capacity_per_s == 1e9
    assert result.lambda_initial_per_s == result.lambda_capacity_per_s
    assert result.steady_time_s == pytest.approx(1e-6)
    assert result.nodes[0].rho == 1.0
    assert result.nodes[0].saturated
    assert result.nodes[0].queue_wait_s is None


def test_finite_source_root_is_below_service_capacity():
    result = solve_memory_path(
        _path(source_slots=1.0),
        _hierarchy(source_slots=1.0),
    )

    assert not result.capacity_bound
    assert result.lambda_initial_per_s == pytest.approx(10e6, rel=1e-9)
    assert result.source_rate_capacity_per_s == pytest.approx(
        result.lambda_initial_per_s, rel=1e-9)
    assert result.nodes[0].rho == pytest.approx(0.01, rel=1e-9)
    assert result.nodes[0].queue_wait_s == 0.0
    assert result.bottleneck == "source"


def test_unknown_workload_independence_omits_source_constraint_explicitly():
    result = solve_memory_path(
        _path(source_slots=None),
        _hierarchy(source_slots=200.0),
    )

    assert result.capacity_bound
    assert result.bottleneck == "service:l2_cache"
    assert result.workload_slots_per_unit is None
    assert result.hardware_source_slots_per_unit == 200.0
    assert result.source_slots_per_unit is None
    assert result.source_rate_capacity_per_s is None


def test_md1_wait_closes_the_source_rate_without_rho_clamp():
    low_slots = solve_memory_path(
        _path(source_slots=1.0),
        _hierarchy(
            source_slots=1.0, queue_model="md1", service_time=1e-9),
    )
    high_slots = solve_memory_path(
        _path(source_slots=20.0),
        _hierarchy(
            source_slots=20.0, queue_model="md1", service_time=1e-9),
    )

    assert 0.0 < low_slots.lambda_initial_per_s < high_slots.lambda_initial_per_s
    assert high_slots.lambda_initial_per_s < high_slots.lambda_capacity_per_s
    assert 0.0 < high_slots.nodes[0].rho < 1.0
    assert math.isfinite(high_slots.nodes[0].queue_wait_s)


def test_local_queue_credit_can_bind_independently_of_source_slots():
    queue_use = QueueUse(
        name="miss_entries",
        allocations_per_initial_request=0.5,
        lifetime_weights=(("l2_cache", 1.0),),
    )
    result = solve_memory_path(
        _path(queues=(queue_use,)),
        _hierarchy(queues=(QueueCapacity("miss_entries", 2.0),)),
    )

    assert not result.capacity_bound
    assert result.lambda_initial_per_s == pytest.approx(40e6, rel=1e-9)
    assert result.bottleneck == "queue:miss_entries"
    assert result.queues[0].used_entries == pytest.approx(2.0, rel=1e-9)


def test_request_rate_capacity_can_bind_before_byte_capacity():
    result = solve_memory_path(
        _path(),
        _hierarchy(request_capacity=250e6),
    )

    assert result.capacity_bound
    assert result.lambda_initial_per_s == 250e6
    assert result.nodes[0].service_capacity_per_s == 250e6
    assert result.nodes[0].capacity_time_s == pytest.approx(4e-6)


def test_capacity_branch_reports_wait_for_a_sub_saturated_md1_node():
    path = MemoryPathDemand(
        name="two_level",
        access_kind="read",
        initial_transactions=1000.0,
        nodes=(
            MemoryNodeDemand("l2_cache", 1000.0, 128_000.0, 128.0),
            MemoryNodeDemand("dram", 1000.0, 128_000.0, 128.0),
        ),
        source=MemorySourceDemand(
            active_units=1,
            producers_per_unit=1000.0,
            independent_requests_per_producer=1.0,
            frontend_rate_per_unit_s=None,
        ),
    )
    hierarchy = MemoryHierarchy(
        services=(
            MemoryService(
                "l2_cache", 256e9, None, 10e-9, "md1", 0.5e-9),
            MemoryService(
                "dram", 128e9, None, 100e-9, "none", None),
        ),
        frontend_rate_per_unit_s=None,
        source_slots_per_unit=1000.0,
    )

    result = solve_memory_path(path, hierarchy)

    assert result.capacity_bound
    assert result.nodes[0].rho == pytest.approx(0.5)
    assert result.nodes[0].queue_wait_s == pytest.approx(0.25e-9)
    assert result.nodes[1].rho == 1.0
    assert result.nodes[1].queue_wait_s is None


def test_path_and_hierarchy_must_name_the_same_physical_structures():
    with pytest.raises(ValueError, match="path/service schema mismatch"):
        solve_memory_path(
            _path(),
            MemoryHierarchy(
                services=(MemoryService(
                    "dram", 1e12, None, 1e-7, "none", None),),
                frontend_rate_per_unit_s=None,
                source_slots_per_unit=1.0,
            ),
        )


def test_a800_l2_without_partition_evidence_uses_byte_capacity_only():
    hardware = HWSpec("configs/hardware/nvidia_a800.yaml")
    demand = _path()
    demand = MemoryPathDemand(
        name=demand.name,
        access_kind=demand.access_kind,
        initial_transactions=demand.initial_transactions,
        nodes=demand.nodes,
        source=MemorySourceDemand(
            active_units=hardware.num_compute_units,
            producers_per_unit=1.0,
            independent_requests_per_producer=1.0,
            frontend_rate_per_unit_s=None,
        ),
    )
    hierarchy = memory_hierarchy_for_path(
        hardware,
        demand,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        dram_capacity_multiplier=1.0,
    )
    service = hierarchy.services[0]

    assert service.queue_model == "none"
    assert service.request_capacity_per_s is None
    assert service.service_time_s is None


def test_gpu_l2_without_partition_evidence_uses_byte_capacity_only():
    hardware = HWSpec("configs/hardware/metax_c500.yaml")
    demand = _path()
    demand = MemoryPathDemand(
        name=demand.name,
        access_kind=demand.access_kind,
        initial_transactions=demand.initial_transactions,
        nodes=demand.nodes,
        source=MemorySourceDemand(
            active_units=hardware.num_compute_units,
            producers_per_unit=1.0,
            independent_requests_per_producer=1.0,
            frontend_rate_per_unit_s=None,
        ),
    )

    hierarchy = memory_hierarchy_for_path(
        hardware,
        demand,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        dram_capacity_multiplier=1.0,
    )
    service = hierarchy.services[0]

    assert service.request_capacity_per_s is None
    assert service.queue_model == "none"
    assert service.service_time_s is None


@pytest.mark.parametrize(
    ("active_units", "expected_mte2_capacity"),
    ((1, 16 * 1.8e9), (48, 768 * 1.8e9)),
)
def test_ascend_aiv_mte2_capacity_scales_with_active_vector_cores(
        active_units, expected_mte2_capacity):
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    byte_count = 4096.0
    demand = MemoryPathDemand(
        name="mte2_read",
        access_kind="read",
        initial_transactions=None,
        nodes=(
            MemoryNodeDemand("mte2", None, byte_count, None),
            MemoryNodeDemand("l2_cache", None, byte_count, None),
            MemoryNodeDemand("dram", None, byte_count, None),
        ),
        source=MemorySourceDemand(active_units, 1.0, None, None),
    )

    hierarchy = memory_hierarchy_for_path(
        hardware,
        demand,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        dram_capacity_multiplier=1.0,
    )
    services = {
        service.resource_role: service for service in hierarchy.services}

    assert services["mte2"].byte_capacity_per_s == pytest.approx(
        expected_mte2_capacity)
    assert services["dram"].byte_capacity_per_s == pytest.approx(
        768 * hardware.nominal_dram_freq_hz)


@pytest.mark.parametrize(
    ("role", "bytes_per_cycle"),
    (("l0a_buffer", 512), ("l0b_buffer", 256)),
)
def test_ascend_aic_local_operand_paths_use_active_l0_capacity(
        role, bytes_per_cycle):
    hardware = HWSpec("configs/hardware/ascend_910b.yaml")
    active_units = 22
    byte_count = 131_072.0
    demand = MemoryPathDemand(
        name=f"{role}_read",
        access_kind="read",
        initial_transactions=None,
        nodes=(MemoryNodeDemand(role, None, byte_count, None),),
        source=MemorySourceDemand(active_units, 1.0, None, None),
    )

    hierarchy = memory_hierarchy_for_path(
        hardware,
        demand,
        sm_freq_hz=hardware.nominal_sm_freq_hz,
        dram_freq_hz=hardware.nominal_dram_freq_hz,
        dram_capacity_multiplier=1.0,
    )
    service, = hierarchy.services

    assert service.resource_role == role
    assert service.byte_capacity_per_s == pytest.approx(
        active_units * bytes_per_cycle * hardware.nominal_sm_freq_hz)


def _dram_path(name, access_kind, byte_count):
    return MemoryPathDemand(
        name=name,
        access_kind=access_kind,
        initial_transactions=byte_count / 128.0,
        nodes=(MemoryNodeDemand(
            "dram", byte_count / 128.0, byte_count, 128.0),),
        source=MemorySourceDemand(
            active_units=1,
            producers_per_unit=1.0,
            independent_requests_per_producer=None,
            frontend_rate_per_unit_s=None,
        ),
    )


def _solve_dram_path(path, capacity):
    return solve_memory_path(
        path,
        MemoryHierarchy(
            services=(MemoryService(
                "dram", capacity, None, 100e-9, "none", None),),
            frontend_rate_per_unit_s=None,
            source_slots_per_unit=None,
        ),
    )


def test_directional_arbitration_is_exact_half_duplex_service_sum():
    read = _dram_path("read", "read", 3000.0)
    write = _dram_path("write", "write", 2000.0)
    read_capacity = 1000.0
    write_capacity = 500.0
    result = solve_memory_arbitration(
        (read, write),
        (
            _solve_dram_path(read, read_capacity),
            _solve_dram_path(write, write_capacity),
        ),
        resource_role="dram",
        arbitration_mode="shared_directional_capacity",
        read_capacity_per_s=read_capacity,
        write_capacity_per_s=write_capacity,
    )

    assert result.read_service_time_s == 3.0
    assert result.write_service_time_s == 4.0
    assert result.analytical_time_s == 7.0
    assert result.effective_time_s == 7.0
    assert result.switch_count is None
    assert result.read_to_write_turnaround_s is None
    assert result.write_to_read_turnaround_s is None
    assert result.bound_kind == "lower_bound"
    assert result.schedule_status == "unavailable"


def test_independent_directional_arbitration_overlaps_read_and_write():
    read = _dram_path("read", "read", 3000.0)
    write = _dram_path("write", "write", 2000.0)
    read_capacity = 1000.0
    write_capacity = 500.0
    result = solve_memory_arbitration(
        (read, write),
        (
            _solve_dram_path(read, read_capacity),
            _solve_dram_path(write, write_capacity),
        ),
        resource_role="dram",
        arbitration_mode="independent_directional_capacity",
        read_capacity_per_s=read_capacity,
        write_capacity_per_s=write_capacity,
    )

    assert result.read_service_time_s == 3.0
    assert result.write_service_time_s == 4.0
    assert result.analytical_time_s == 4.0
    assert result.effective_time_s == 4.0
    assert result.bound_kind == "lower_bound"
    assert result.schedule_status == "unavailable"


def test_common_arbitration_is_total_bytes_over_one_capacity():
    read = _dram_path("read", "read", 3000.0)
    write = _dram_path("write", "write", 2000.0)
    result = solve_memory_arbitration(
        (read, write),
        (
            _solve_dram_path(read, 1000.0),
            _solve_dram_path(write, 1000.0),
        ),
        resource_role="dram",
        arbitration_mode="shared_common_capacity",
        read_capacity_per_s=1000.0,
        write_capacity_per_s=1000.0,
    )

    assert result.analytical_time_s == 5.0


def test_arbitration_rejects_a_path_solved_with_the_wrong_capacity():
    read = _dram_path("read", "read", 3000.0)

    with pytest.raises(ValueError, match="did not use the declared"):
        solve_memory_arbitration(
            (read,),
            (_solve_dram_path(read, 900.0),),
            resource_role="dram",
            arbitration_mode="shared_directional_capacity",
            read_capacity_per_s=1000.0,
            write_capacity_per_s=500.0,
        )


def test_common_arbitration_rejects_unequal_directional_capacities():
    read = _dram_path("read", "read", 3000.0)

    with pytest.raises(ValueError, match="requires equal read/write"):
        solve_memory_arbitration(
            (read,),
            (_solve_dram_path(read, 1000.0),),
            resource_role="dram",
            arbitration_mode="shared_common_capacity",
            read_capacity_per_s=1000.0,
            write_capacity_per_s=500.0,
        )


def test_arbitration_diagnostic_enforces_common_capacity_at_schema_boundary():
    with pytest.raises(ValueError, match="requires equal read/write"):
        MemoryArbitrationDiagnostic(
            resource_role="dram",
            arbitration_mode="shared_common_capacity",
            read_bytes=100.0,
            write_bytes=50.0,
            read_capacity_per_s=100.0,
            write_capacity_per_s=50.0,
            read_service_time_s=1.0,
            write_service_time_s=1.0,
            analytical_time_s=2.0,
            effective_time_s=2.0,
            switch_count=None,
            read_to_write_turnaround_s=None,
            write_to_read_turnaround_s=None,
            bound_kind="lower_bound",
            schedule_status="unavailable",
            overridden=False,
        )
