"""Ascend aggregate-byte memory contract tests."""

import pytest

from src.kernel.hardware_spec import HWSpec
from src.kernel.memory_model import memory_hierarchy_for_path, solve_memory_path
from src.kernel.model_contract import (
    MemoryHierarchy,
    MemoryNodeDemand,
    MemoryPathDemand,
    MemoryService,
    MemorySourceDemand,
)


HARDWARE = HWSpec("configs/hardware/ascend_910b.yaml")


def _demand(byte_count, active_cores):
    return MemoryPathDemand(
        name="mte2_read",
        access_kind="read",
        initial_transactions=None,
        nodes=tuple(
            MemoryNodeDemand(
                resource_role=role,
                transactions=None,
                bytes=float(byte_count),
                transaction_bytes=None,
            )
            for role in ("l2_cache", "dram")
        ),
        source=MemorySourceDemand(
            active_units=active_cores,
            producers_per_unit=1.0,
            independent_requests_per_producer=None,
            frontend_rate_per_unit_s=None,
        ),
    )


def _solve(byte_count, active_cores):
    demand = _demand(byte_count, active_cores)
    hierarchy = memory_hierarchy_for_path(
        HARDWARE,
        demand,
        sm_freq_hz=HARDWARE.nominal_sm_freq_hz,
        dram_freq_hz=HARDWARE.nominal_dram_freq_hz,
        dram_capacity_multiplier=1.0,
    )
    return demand, hierarchy, solve_memory_path(demand, hierarchy)


def test_910b_read_path_uses_only_aggregate_byte_capacity():
    byte_count = 64 * 1024 * 1024
    _, hierarchy, result = _solve(byte_count, active_cores=24)
    expected_capacity = min(
        service.byte_capacity_per_s for service in hierarchy.services)

    assert result.rate_unit == "bytes"
    assert result.initial_transactions is None
    assert result.lambda_initial_per_s == pytest.approx(expected_capacity)
    assert result.steady_time_s == pytest.approx(
        byte_count / expected_capacity)
    assert result.first_request_latency_s is None
    assert result.workload_slots_per_unit is None
    assert result.hardware_source_slots_per_unit is None
    assert result.queues == ()
    assert all(node.transactions is None for node in result.nodes)


def test_910b_byte_path_does_not_consume_request_or_latency_mechanisms():
    _, hierarchy, _ = _solve(4096, active_cores=1)

    assert hierarchy.frontend_rate_per_unit_s is None
    assert hierarchy.source_slots_per_unit is None
    assert hierarchy.queues == ()
    assert all(
        service.request_capacity_per_s is None
        and service.queue_model == "none"
        and service.unloaded_latency_s == 0.0
        for service in hierarchy.services)


def test_910b_hierarchy_rejects_fabricated_transaction_geometry():
    byte_count = 4096.0
    demand = MemoryPathDemand(
        name="mte2_read",
        access_kind="read",
        initial_transactions=1.0,
        nodes=(
            MemoryNodeDemand("l2_cache", 1.0, byte_count, byte_count),
            MemoryNodeDemand("dram", 1.0, byte_count, byte_count),
        ),
        source=MemorySourceDemand(1, 1.0, 1.0, None),
    )

    with pytest.raises(ValueError, match="requires byte-only demand"):
        memory_hierarchy_for_path(
            HARDWARE,
            demand,
            sm_freq_hz=HARDWARE.nominal_sm_freq_hz,
            dram_freq_hz=HARDWARE.nominal_dram_freq_hz,
            dram_capacity_multiplier=1.0,
        )


def test_byte_only_path_rejects_request_source_slots():
    with pytest.raises(ValueError, match="independent request slots"):
        MemoryPathDemand(
            name="mte2_read",
            access_kind="read",
            initial_transactions=None,
            nodes=(MemoryNodeDemand("l2_cache", None, 4096.0, None),),
            source=MemorySourceDemand(1, 1.0, 1.0, None),
        )


@pytest.mark.parametrize(
    ("service", "source_slots", "message"),
    [
        (MemoryService("l2_cache", 1e12, 1e9, 0.0, "none", None),
         None, "request-rate capacity"),
        (MemoryService("l2_cache", 1e12, None, 1e-7, "none", None),
         None, "first-request latency"),
        (MemoryService("l2_cache", 1e12, None, 0.0, "none", None),
         4.0, "request-source slots"),
    ],
)
def test_byte_only_solver_rejects_request_mechanisms(
        service, source_slots, message):
    demand = MemoryPathDemand(
        name="mte2_read",
        access_kind="read",
        initial_transactions=None,
        nodes=(MemoryNodeDemand("l2_cache", None, 4096.0, None),),
        source=MemorySourceDemand(1, 1.0, None, None),
    )
    hierarchy = MemoryHierarchy(
        services=(service,),
        frontend_rate_per_unit_s=None,
        source_slots_per_unit=source_slots,
    )

    with pytest.raises(ValueError, match=message):
        solve_memory_path(demand, hierarchy)
