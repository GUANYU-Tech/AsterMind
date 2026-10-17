import math

import pytest

from src.serving.kv_cache import (
    HostDramAccess,
    HostDramAccessPattern,
    HostDramAnalyticalBackend,
    HostDramCapability,
    HostDramDirection,
    HostDramIsolatedAccessBackend,
    HostDramMechanismParameter,
    HostDramModelKind,
    HostDramOperatingPoint,
    HostDramParameterProvenance,
    HostDramPath,
    HostDramPrediction,
    HostDramPredictionProvenance,
    KVCacheConfig,
    KVCachePhysicalTransferCompletion,
    KVCacheTierConfig,
    KVCacheTierManager,
    KVCacheTransferCompletion,
)


def _parameter(name, value, unit):
    return HostDramMechanismParameter(
        name=name,
        value=value,
        unit=unit,
        provenance=HostDramParameterProvenance(
            source="synthetic unit-test fixture",
            measurement_method="closed-form test input",
            physical_meaning=f"test-only {name.replace('_', ' ')}",
            applicability="tests/serving/test_host_dram.py only",
        ),
    )


def _capability(
    direction=HostDramDirection.READ,
    *,
    hardware_id="nvidia_a800",
    path=HostDramPath.LOCAL,
    initiator_numa_node=0,
    memory_numa_node=0,
    initiator_socket_id=0,
    memory_socket_id=0,
    concurrency=1,
    read_ratio=(1, 1),
    controller_resource_id="host-dram-controller:numa-0:0",
):
    return HostDramCapability(
        hardware_id=hardware_id,
        direction=direction,
        access_pattern=HostDramAccessPattern.STREAMING,
        initiator_numa_node=initiator_numa_node,
        memory_numa_node=memory_numa_node,
        initiator_socket_id=initiator_socket_id,
        memory_socket_id=memory_socket_id,
        path=path,
        concurrency=concurrency,
        controller_resource_id=controller_resource_id,
        read_ratio_numerator=read_ratio[0],
        read_ratio_denominator=read_ratio[1],
        min_bytes=64,
        max_bytes=8192,
    )


def _point(capability, *, latency_ns=100.0, bandwidth_bytes_s=1024.0):
    return HostDramOperatingPoint(
        capability=capability,
        fixed_access_latency=_parameter(
            "fixed_access_latency", latency_ns, "ns"),
        streaming_bandwidth=_parameter(
            "streaming_bandwidth", bandwidth_bytes_s, "bytes/s"),
    )


def _access(
    direction=HostDramDirection.READ,
    *,
    hardware_id="nvidia_a800",
    bytes_=1024,
    read_bytes=None,
    write_bytes=None,
    path=HostDramPath.LOCAL,
    initiator_numa_node=0,
    memory_numa_node=0,
    initiator_socket_id=0,
    memory_socket_id=0,
    concurrency=1,
    controller_resource_id="host-dram-controller:numa-0:0",
):
    if read_bytes is None:
        read_bytes = bytes_ if direction is HostDramDirection.READ else 0
    if write_bytes is None:
        write_bytes = bytes_ if direction is HostDramDirection.WRITE else 0
    return HostDramAccess(
        hardware_id=hardware_id,
        direction=direction,
        bytes=bytes_,
        read_bytes=read_bytes,
        write_bytes=write_bytes,
        access_pattern=HostDramAccessPattern.STREAMING,
        initiator_numa_node=initiator_numa_node,
        memory_numa_node=memory_numa_node,
        initiator_socket_id=initiator_socket_id,
        memory_socket_id=memory_socket_id,
        path=path,
        concurrency=concurrency,
        controller_resource_id=controller_resource_id,
    )


def test_analytical_backend_preserves_units_and_parameter_provenance():
    backend = HostDramAnalyticalBackend((_point(_capability()),))
    access = _access()

    prediction = backend.predict(access)

    assert prediction.access is access
    assert prediction.t_fixed_access_s == pytest.approx(100e-9)
    assert prediction.t_streaming_s == pytest.approx(1.0)
    assert prediction.t_queue_s == 0.0
    assert prediction.t_total_s == pytest.approx(1.0 + 100e-9)
    assert prediction.controller_resource_id == (
        "host-dram-controller:numa-0:0")
    assert prediction.provenance.model_kind is HostDramModelKind.ANALYTICAL
    assert prediction.provenance.backend_name == backend.backend_name
    assert tuple(
        (parameter.name, parameter.unit)
        for parameter in prediction.provenance.parameters
    ) == (
        ("fixed_access_latency", "ns"),
        ("streaming_bandwidth", "bytes/s"),
    )
    assert all(
        parameter.provenance.source == "synthetic unit-test fixture"
        for parameter in prediction.provenance.parameters)


def test_direction_numa_mix_and_concurrency_select_exact_operating_points():
    write_capability = _capability(
        HostDramDirection.WRITE,
        path=HostDramPath.CROSS_SOCKET,
        initiator_numa_node=0,
        memory_numa_node=1,
        initiator_socket_id=0,
        memory_socket_id=1,
        read_ratio=(0, 1),
        controller_resource_id="host-dram-controller:numa-1:0",
    )
    mixed_capability = _capability(
        HostDramDirection.MIXED,
        path=HostDramPath.CROSS_SOCKET,
        initiator_numa_node=0,
        memory_numa_node=1,
        initiator_socket_id=0,
        memory_socket_id=1,
        concurrency=4,
        read_ratio=(3, 4),
        controller_resource_id="host-dram-controller:numa-1:0",
    )
    backend = HostDramAnalyticalBackend((
        _point(write_capability, bandwidth_bytes_s=512.0),
        _point(mixed_capability, bandwidth_bytes_s=256.0),
    ))
    write = _access(
        HostDramDirection.WRITE,
        path=HostDramPath.CROSS_SOCKET,
        initiator_numa_node=0,
        memory_numa_node=1,
        initiator_socket_id=0,
        memory_socket_id=1,
        controller_resource_id="host-dram-controller:numa-1:0",
    )
    mixed = _access(
        HostDramDirection.MIXED,
        read_bytes=768,
        write_bytes=256,
        path=HostDramPath.CROSS_SOCKET,
        initiator_numa_node=0,
        memory_numa_node=1,
        initiator_socket_id=0,
        memory_socket_id=1,
        concurrency=4,
        controller_resource_id="host-dram-controller:numa-1:0",
    )

    assert backend.predict(write).t_streaming_s == pytest.approx(2.0)
    assert backend.predict(mixed).t_streaming_s == pytest.approx(4.0)
    assert write_capability.supports(write)
    assert mixed_capability.supports(mixed)
    assert backend.capabilities == (write_capability, mixed_capability)


@pytest.mark.parametrize(
    "access,error",
    (
        (
            lambda: _access(read_bytes=512, write_bytes=512),
            "direction disagrees",
        ),
        (
            lambda: _access(
                path=HostDramPath.LOCAL,
                initiator_numa_node=0,
                memory_numa_node=1,
                initiator_socket_id=0,
                memory_socket_id=1),
            "local.*same socket",
        ),
        (
            lambda: _access(
                path=HostDramPath.CROSS_SOCKET,
                initiator_numa_node=0,
                memory_numa_node=0,
                initiator_socket_id=0,
                memory_socket_id=1),
            "NUMA node cannot map to multiple sockets",
        ),
    ),
)
def test_access_contract_rejects_inconsistent_inputs(access, error):
    with pytest.raises(ValueError, match=error):
        access()


@pytest.mark.parametrize(
    "capability,error",
    (
        (
            lambda: _capability(
                HostDramDirection.MIXED, read_ratio=(1, 1)),
            "direction disagrees",
        ),
        (
            lambda: _capability(read_ratio=(2, 2)),
            "must be reduced",
        ),
        (
            lambda: _capability(
                path=HostDramPath.CROSS_SOCKET,
                initiator_numa_node=0,
                memory_numa_node=0,
                initiator_socket_id=0,
                memory_socket_id=1),
            "NUMA node cannot map to multiple sockets",
        ),
    ),
)
def test_capability_rejects_ambiguous_scope(capability, error):
    with pytest.raises(ValueError, match=error):
        capability()


def test_operating_point_requires_exact_units_and_unique_capability():
    capability = _capability()
    with pytest.raises(ValueError, match="unit 'ns'"):
        HostDramOperatingPoint(
            capability,
            _parameter("fixed_access_latency", 100.0, "us"),
            _parameter("streaming_bandwidth", 1024.0, "bytes/s"),
        )
    point = _point(capability)
    with pytest.raises(ValueError, match="cannot overlap"):
        HostDramAnalyticalBackend((point, point))


@pytest.mark.parametrize(
    "access",
    (
        _access(hardware_id="metax_c500"),
        _access(concurrency=2),
        _access(bytes_=16_384),
        _access(
            HostDramDirection.MIXED,
            read_bytes=512,
            write_bytes=512),
    ),
)
def test_backend_fails_loud_without_an_exact_operating_point(access):
    backend = HostDramAnalyticalBackend((_point(_capability()),))
    with pytest.raises(NotImplementedError, match="no exact"):
        backend.predict(access)


def test_backend_protocol_is_limited_to_isolated_access_service_demand():
    backend = HostDramAnalyticalBackend((_point(_capability()),))

    assert isinstance(backend, HostDramIsolatedAccessBackend)
    prediction = backend.predict(_access())
    assert prediction.t_queue_s == 0.0
    assert "isolated_access_service_demand_only" in (
        prediction.provenance.limitations)
    assert "execution_timeline_owns_queueing_and_contention" in (
        prediction.provenance.limitations)
    assert "no_address_or_channel_timing_contract" in (
        prediction.provenance.limitations)


def test_backend_rejects_conflicting_numa_to_socket_capabilities():
    local = _capability()
    conflicting = _capability(
        path=HostDramPath.CROSS_SOCKET,
        initiator_numa_node=0,
        memory_numa_node=1,
        initiator_socket_id=1,
        memory_socket_id=0,
        controller_resource_id="host-dram-controller:numa-1:0",
    )

    with pytest.raises(
            ValueError,
            match="NUMA node cannot map to multiple sockets across"):
        HostDramAnalyticalBackend((_point(local), _point(conflicting)))


def test_backend_rejects_access_that_remaps_a_known_numa_node():
    backend = HostDramAnalyticalBackend((_point(_capability()),))
    contradictory_access = _access(
        path=HostDramPath.CROSS_SOCKET,
        initiator_numa_node=0,
        memory_numa_node=1,
        initiator_socket_id=1,
        memory_socket_id=0,
        controller_resource_id="host-dram-controller:numa-1:0",
    )

    with pytest.raises(ValueError, match="known NUMA node"):
        backend.predict(contradictory_access)


def test_host_dram_prediction_does_not_commit_residency_before_completion():
    block_tokens = 16
    block_bytes = 1024
    manager = KVCacheTierManager(KVCacheConfig(
        tiers=(
            KVCacheTierConfig("hbm", block_bytes, block_tokens),
            KVCacheTierConfig(
                "host", block_bytes, block_tokens, "pcie"),
        ),
        bytes_per_token=block_bytes // block_tokens,
        offload_policy="lru",
    ))
    manager.allocate("request", 0, block_tokens)
    event = manager.prepare_evict(pool_name="main", ranks=(0,))
    before_prediction = manager.manifest()
    backend = HostDramAnalyticalBackend((
        _point(_capability(HostDramDirection.WRITE, read_ratio=(0, 1))),
    ))

    prediction = backend.predict(_access(HostDramDirection.WRITE))

    assert manager.manifest() == before_prediction
    call_id = f"step:kv-offload:{event.ready_event}"
    completed = KVCacheTransferCompletion(
        ready_event=event.ready_event,
        action=event.action,
        bytes=event.bytes,
        source_tier=event.source_tier,
        destination_tier=event.destination_tier,
        link="pcie",
        physical_transfers=(KVCachePhysicalTransferCompletion(
            node_id=call_id,
            call_id=call_id,
            pool_name="main",
            rank=0,
            bytes=event.bytes,
            link="pcie",
            start_time_s=0.0,
            end_time_s=prediction.t_total_s,
        ),),
    )
    manager.complete_transfer(completed)
    after_completion = manager.manifest()
    assert after_completion["tiers"]["hbm"]["used_bytes"] == 0
    assert after_completion["tiers"]["host"]["used_bytes"] == block_bytes
    assert after_completion["residency"][0]["tier"] == "host"


def test_prediction_rejects_an_inconsistent_backend_breakdown():
    access = _access()
    point = _point(_capability())
    provenance = HostDramPredictionProvenance(
        backend_name="fixture",
        backend_version="1",
        model_kind=HostDramModelKind.ANALYTICAL,
        source_reference="unit test",
        parameters=(
            point.fixed_access_latency,
            point.streaming_bandwidth,
        ),
        limitations=(),
    )
    with pytest.raises(ValueError, match="total disagrees"):
        HostDramPrediction(
            access=access,
            t_fixed_access_s=1.0,
            t_streaming_s=2.0,
            t_queue_s=0.0,
            t_total_s=4.0,
            controller_resource_id=access.controller_resource_id,
            provenance=provenance,
        )
    with pytest.raises(ValueError, match="cannot include queue time"):
        HostDramPrediction(
            access=access,
            t_fixed_access_s=1.0,
            t_streaming_s=2.0,
            t_queue_s=3.0,
            t_total_s=6.0,
            controller_resource_id=access.controller_resource_id,
            provenance=provenance,
        )
    assert math.isfinite(
        HostDramAnalyticalBackend((_point(_capability()),))
        .predict(access).t_total_s)
