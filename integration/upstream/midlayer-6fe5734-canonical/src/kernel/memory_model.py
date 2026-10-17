"""Hierarchical byte/request capacity and finite-source memory solver."""

import math

from src.kernel.model_contract import (
    MemoryAccessClassDiagnostic,
    MemoryAccessClassService,
    MemoryArbitrationDiagnostic,
    MemoryDiagnostics,
    MemoryHierarchy,
    MemoryNodeDiagnostic,
    MemoryPathDemand,
    MemoryService,
    QueueDiagnostic,
)


def memory_hierarchy_for_path(
        hardware, demand: MemoryPathDemand, *, sm_freq_hz: float,
        dram_freq_hz: float, dram_capacity_multiplier: float) -> MemoryHierarchy:
    """Instantiate one path's hardware services from HWSpec capabilities."""

    if not isinstance(demand, MemoryPathDemand):
        raise TypeError(
            "memory_hierarchy_for_path demand must be MemoryPathDemand")
    for name, value in (
            ("sm_freq_hz", sm_freq_hz),
            ("dram_freq_hz", dram_freq_hz),
            ("dram_capacity_multiplier", dram_capacity_multiplier)):
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    memory_model = hardware.kernel_memory_model
    services = []
    if memory_model == "gpu_mshr":
        l2_latency_s = hardware.l2_latency_ns * 1e-9
        dram_increment_s = (
            hardware.dram_latency_ns - hardware.l2_latency_ns) * 1e-9
        if l2_latency_s <= 0 or dram_increment_s < 0:
            raise ValueError(
                "GPU memory hierarchy requires a non-decreasing latency path")
        for node in demand.nodes:
            resource = hardware.resource_by_role(node.resource_role)
            byte_capacity = resource.effective_capacity(
                demand.source.active_units,
                sm_freq_hz=sm_freq_hz,
                dram_freq_hz=dram_freq_hz)
            if node.resource_role == "dram":
                byte_capacity = hardware.dram_capacity_for_access(
                    demand.access_kind,
                    active_units=demand.source.active_units,
                    sm_freq_hz=sm_freq_hz,
                    dram_freq_hz=dram_freq_hz,
                )
            byte_capacity *= node.independent_service_streams
            if node.capacity_ceiling_resource_role is not None:
                ceiling_resource = hardware.resource_by_role(
                    node.capacity_ceiling_resource_role)
                ceiling_capacity = ceiling_resource.effective_capacity(
                    demand.source.active_units,
                    sm_freq_hz=sm_freq_hz,
                    dram_freq_hz=dram_freq_hz)
                byte_capacity = min(byte_capacity, ceiling_capacity)
            if node.resource_role == "l2_cache":
                configured_service_cycles = hardware.memory_subsystem[
                    "l2_partition_service_cycles_bw"]
                if configured_service_cycles is None:
                    # A byte-capacity-only service is the only defensible model
                    # when partition count or transaction issue interval is not
                    # independently known.  Do not manufacture an M/D/1 queue
                    # from an estimated partition count.
                    services.append(MemoryService(
                        resource_role=node.resource_role,
                        byte_capacity_per_s=byte_capacity,
                        request_capacity_per_s=None,
                        unloaded_latency_s=l2_latency_s,
                        queue_model="none",
                        service_time_s=None,
                    ))
                else:
                    service_cycles = float(configured_service_cycles)
                    partitions = int(resource.num_partitions)
                    if service_cycles <= 0 or partitions <= 0:
                        raise ValueError(
                            "GPU L2 service cycles and partitions must be "
                            "positive when request queueing is enabled")
                    request_capacity = (
                        partitions * sm_freq_hz / service_cycles)
                    service_capacity = min(
                        byte_capacity / node.transaction_bytes,
                        request_capacity,
                    )
                    services.append(MemoryService(
                        resource_role=node.resource_role,
                        byte_capacity_per_s=byte_capacity,
                        request_capacity_per_s=request_capacity,
                        unloaded_latency_s=l2_latency_s,
                        queue_model="md1",
                        service_time_s=partitions / service_capacity,
                    ))
            elif node.resource_role == "dram":
                services.append(MemoryService(
                    resource_role=node.resource_role,
                    byte_capacity_per_s=(
                        byte_capacity * dram_capacity_multiplier),
                    request_capacity_per_s=None,
                    unloaded_latency_s=dram_increment_s,
                    queue_model="none",
                    service_time_s=None,
                ))
            else:
                raise ValueError(
                    "gpu_mshr path supports only l2_cache and dram, got "
                    f"{node.resource_role!r}")
        configured_frontend_rate = hardware.memory_subsystem[
            "lsu_peak_issue_rate_per_sm"]
        frontend_rate = (
            None if configured_frontend_rate is None else
            float(configured_frontend_rate) * sm_freq_hz)
        source_slots = hardware.source_slots_per_unit
    elif memory_model == "npu_tile":
        if demand.rate_unit != "bytes":
            raise ValueError(
                "npu_tile memory requires byte-only demand until physical "
                "MTE transaction geometry is independently established")
        if demand.access_kind == "read":
            allowed_roles = {
                "mte2", "l0a_buffer", "l0b_buffer", "l2_cache", "dram"}
        elif demand.access_kind == "write":
            allowed_roles = {"mte3", "l2_cache", "dram"}
        else:
            raise AssertionError(
                f"unhandled NPU access kind: {demand.access_kind!r}")
        for node in demand.nodes:
            if node.resource_role not in allowed_roles:
                raise ValueError(
                    f"npu_tile {demand.access_kind} path does not support "
                    f"role {node.resource_role!r}")
            resource = hardware.resource_by_role(node.resource_role)
            byte_capacity = resource.effective_capacity(
                demand.source.active_units,
                sm_freq_hz=sm_freq_hz,
                dram_freq_hz=dram_freq_hz)
            if node.resource_role == "dram":
                byte_capacity = hardware.dram_capacity_for_access(
                    demand.access_kind,
                    active_units=demand.source.active_units,
                    sm_freq_hz=sm_freq_hz,
                    dram_freq_hz=dram_freq_hz,
                )
            if node.resource_role in ("mte2", "mte3"):
                if resource.total_units <= 0:
                    raise ValueError(
                        f"{node.resource_role.upper()} resource has no "
                        "physical units")
                byte_capacity *= (
                    min(demand.source.active_units, resource.total_units)
                    / resource.total_units)
            elif node.resource_role == "dram":
                byte_capacity *= dram_capacity_multiplier
            elif node.resource_role == "l2_cache":
                pass
            elif node.resource_role in ("l0a_buffer", "l0b_buffer"):
                pass
            else:
                raise AssertionError(
                    f"unhandled validated NPU role: {node.resource_role!r}")
            byte_capacity *= node.independent_service_streams
            if node.capacity_ceiling_resource_role is not None:
                ceiling_resource = hardware.resource_by_role(
                    node.capacity_ceiling_resource_role)
                ceiling_capacity = ceiling_resource.effective_capacity(
                    demand.source.active_units,
                    sm_freq_hz=sm_freq_hz,
                    dram_freq_hz=dram_freq_hz)
                byte_capacity = min(byte_capacity, ceiling_capacity)
            services.append(MemoryService(
                resource_role=node.resource_role,
                byte_capacity_per_s=byte_capacity,
                request_capacity_per_s=None,
                # Retained 910B evidence establishes only aggregate INI byte
                # capacities.  Unknown first-request latency must not be
                # silently reconstructed from an unproven transfer quantum.
                unloaded_latency_s=0.0,
                queue_model="none",
                service_time_s=None,
            ))
        frontend_rate = None
        # No independent MTE2/MTE3 descriptor or request-source capacity is
        # available for the byte-only aggregate path.
        source_slots = None
    elif memory_model in ("pim_bank", "systolic_scratchpad"):
        raise NotImplementedError(
            f"structured memory hierarchy is not defined for {memory_model!r}")
    else:
        raise ValueError(f"unknown kernel_memory_model: {memory_model!r}")
    return MemoryHierarchy(
        services=tuple(services),
        frontend_rate_per_unit_s=frontend_rate,
        source_slots_per_unit=source_slots,
        queues=(),
    )


def _memory_state(demand, hierarchy, services, queue_capacities, initial_rate,
                  *, include_open_queue_wait):
    """Evaluate one candidate initial request rate.

    ``include_open_queue_wait=False`` is used only for the exact capacity
    sufficiency test.  It asks whether finite source/credit populations cover
    the unloaded bandwidth-delay product; it never manufactures a finite wait
    at rho=1.
    """

    node_state = {}
    for node in demand.nodes:
        service = services[node.resource_role]
        gamma = node.transactions / demand.initial_transactions
        request_capacity = service.request_capacity_per_s
        byte_request_capacity = (
            service.byte_capacity_per_s / node.transaction_bytes)
        if request_capacity is None:
            service_capacity = byte_request_capacity
        else:
            service_capacity = min(byte_request_capacity, request_capacity)
        arrival_rate = gamma * initial_rate
        rho = arrival_rate / service_capacity
        if rho > 1.0 and not math.isclose(
                rho, 1.0, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"candidate rate exceeds {node.resource_role!r} capacity")
        rho = min(rho, 1.0)
        if not include_open_queue_wait or service.queue_model == "none":
            queue_wait = 0.0
        elif rho >= 1.0:
            queue_wait = math.inf
        elif service.queue_model == "md1":
            queue_wait = (
                rho / (2.0 * (1.0 - rho)) * service.service_time_s)
        else:
            raise AssertionError(
                f"unhandled queue model: {service.queue_model!r}")
        node_state[node.resource_role] = {
            "gamma": gamma,
            "service_capacity": service_capacity,
            "arrival_rate": arrival_rate,
            "rho": rho,
            "queue_wait": queue_wait,
            "increment_latency": service.unloaded_latency_s + queue_wait,
        }

    source_lifetime = sum(
        state["gamma"] * state["increment_latency"]
        for state in node_state.values())
    workload_slots = demand.source.workload_slots_per_unit
    hardware_slots = hierarchy.source_slots_per_unit
    if workload_slots is None:
        source_slots = None
    elif hardware_slots is None:
        source_slots = workload_slots
    else:
        source_slots = min(workload_slots, hardware_slots)
    if source_slots is None:
        source_capacity = None
    elif source_lifetime == 0.0:
        source_capacity = math.inf
    else:
        source_capacity = (
            demand.source.active_units * source_slots / source_lifetime)

    frontend_candidates = []
    if demand.source.frontend_rate_per_unit_s is not None:
        frontend_candidates.append(
            demand.source.frontend_rate_per_unit_s
            * demand.source.active_units)
    if hierarchy.frontend_rate_per_unit_s is not None:
        frontend_candidates.append(
            hierarchy.frontend_rate_per_unit_s * demand.source.active_units)
    frontend_capacity = (
        min(frontend_candidates) if frontend_candidates else None)

    queue_state = {}
    for queue_use in demand.queues:
        capacity = queue_capacities[queue_use.name]
        lifetime = sum(
            weight * node_state[role]["increment_latency"]
            for role, weight in queue_use.lifetime_weights)
        allocation_ratio = queue_use.allocations_per_initial_request
        if lifetime == 0.0:
            initial_capacity = math.inf
        else:
            initial_capacity = (
                capacity.max_entries / (allocation_ratio * lifetime))
        allocation_rate = allocation_ratio * initial_rate
        queue_state[queue_use.name] = {
            "allocation_rate": allocation_rate,
            "lifetime": lifetime,
            "used_entries": allocation_rate * lifetime,
            "initial_capacity": initial_capacity,
        }

    limits = []
    if source_capacity is not None:
        limits.append(("source", source_capacity))
    if frontend_capacity is not None:
        limits.append(("frontend", frontend_capacity))
    limits.extend(
        (f"queue:{name}", state["initial_capacity"])
        for name, state in queue_state.items())
    if limits:
        bottleneck, phi = min(limits, key=lambda item: item[1])
    else:
        bottleneck, phi = None, math.inf
    return {
        "nodes": node_state,
        "workload_slots": workload_slots,
        "hardware_slots": hardware_slots,
        "source_slots": source_slots,
        "source_lifetime": source_lifetime,
        "source_capacity": source_capacity,
        "frontend_capacity": frontend_capacity,
        "queues": queue_state,
        "phi": phi,
        "bottleneck": bottleneck,
    }


def _solve_byte_only_path(
        demand: MemoryPathDemand, hierarchy: MemoryHierarchy, services
        ) -> MemoryDiagnostics:
    """Solve a path whose physical transaction geometry is unknown."""

    if demand.rate_unit != "bytes":
        raise ValueError("byte-only solver requires a byte-rate demand")
    if hierarchy.frontend_rate_per_unit_s is not None:
        raise ValueError(
            "byte-only paths cannot consume a request frontend ceiling")
    if hierarchy.source_slots_per_unit is not None:
        raise ValueError(
            "byte-only paths cannot consume hardware request-source slots")
    if hierarchy.queues:
        raise ValueError("byte-only paths cannot consume request queues")

    initial_bytes = demand.nodes[0].bytes
    initial_capacity_by_role = {}
    gamma_by_role = {}
    for node in demand.nodes:
        service = services[node.resource_role]
        if service.request_capacity_per_s is not None:
            raise ValueError(
                "byte-only paths cannot consume a request-rate capacity")
        if service.queue_model != "none" or service.service_time_s is not None:
            raise ValueError(
                "byte-only paths cannot consume a request queue model")
        if service.unloaded_latency_s != 0.0:
            raise ValueError(
                "byte-only paths cannot consume first-request latency")
        gamma = node.bytes / initial_bytes
        gamma_by_role[node.resource_role] = gamma
        initial_capacity_by_role[node.resource_role] = (
            service.byte_capacity_per_s / gamma)

    lambda_capacity = min(initial_capacity_by_role.values())
    if not math.isfinite(lambda_capacity) or lambda_capacity <= 0:
        raise ValueError("byte-only memory path produced a non-positive capacity")
    saturated_roles = {
        role for role, value in initial_capacity_by_role.items()
        if math.isclose(value, lambda_capacity, rel_tol=1e-12, abs_tol=0.0)
    }
    node_diagnostics = []
    for node in demand.nodes:
        role = node.resource_role
        service = services[role]
        arrival_rate = gamma_by_role[role] * lambda_capacity
        rho = arrival_rate / service.byte_capacity_per_s
        node_diagnostics.append(MemoryNodeDiagnostic(
            resource_role=role,
            transactions=None,
            bytes=node.bytes,
            byte_capacity_per_s=service.byte_capacity_per_s,
            request_capacity_per_s=None,
            service_capacity_per_s=service.byte_capacity_per_s,
            arrival_rate_per_s=arrival_rate,
            rho=min(rho, 1.0),
            queue_wait_s=0.0,
            capacity_time_s=node.bytes / service.byte_capacity_per_s,
            saturated=role in saturated_roles,
            unloaded_latency_s=service.unloaded_latency_s,
        ))
    bottleneck = "+".join(
        f"service:{role}" for role in sorted(saturated_roles))
    return MemoryDiagnostics(
        path_name=demand.name,
        access_kind=demand.access_kind,
        rate_unit="bytes",
        initial_transactions=None,
        lambda_initial_per_s=lambda_capacity,
        lambda_capacity_per_s=lambda_capacity,
        workload_slots_per_unit=None,
        hardware_source_slots_per_unit=None,
        source_slots_per_unit=None,
        source_lifetime_s=0.0,
        source_rate_capacity_per_s=None,
        frontend_rate_capacity_per_s=None,
        steady_time_s=initial_bytes / lambda_capacity,
        first_request_latency_s=None,
        bottleneck=bottleneck,
        capacity_bound=True,
        nodes=tuple(node_diagnostics),
        queues=(),
    )


def solve_memory_path(
        demand: MemoryPathDemand,
        hierarchy: MemoryHierarchy) -> MemoryDiagnostics:
    """Solve the bounded scalar initial-request rate for one memory path.

    The service-capacity result is exact: no ``rho=0.99`` clamp and no
    ``(1-epsilon) * capacity`` prediction are used.  Open M/D/1 waiting is
    reported only for a finite-source/credit root strictly below capacity.
    """

    if not isinstance(demand, MemoryPathDemand):
        raise TypeError("solve_memory_path demand must be MemoryPathDemand")
    if not isinstance(hierarchy, MemoryHierarchy):
        raise TypeError("solve_memory_path hierarchy must be MemoryHierarchy")
    services = {value.resource_role: value for value in hierarchy.services}
    demand_roles = tuple(node.resource_role for node in demand.nodes)
    missing_services = set(demand_roles) - set(services)
    extra_services = set(services) - set(demand_roles)
    if missing_services or extra_services:
        raise ValueError(
            "memory path/service schema mismatch: "
            f"missing={sorted(missing_services)}, "
            f"extra={sorted(extra_services)}")
    queue_capacities = {value.name: value for value in hierarchy.queues}
    demand_queues = {value.name for value in demand.queues}
    missing_queues = demand_queues - set(queue_capacities)
    extra_queues = set(queue_capacities) - demand_queues
    if missing_queues or extra_queues:
        raise ValueError(
            "memory queue schema mismatch: "
            f"missing={sorted(missing_queues)}, extra={sorted(extra_queues)}")

    if demand.rate_unit == "bytes":
        return _solve_byte_only_path(demand, hierarchy, services)
    if demand.rate_unit != "transactions":
        raise AssertionError(f"unknown memory-path rate unit: {demand.rate_unit}")

    service_capacity_by_role = {}
    initial_capacity_by_role = {}
    for node in demand.nodes:
        service = services[node.resource_role]
        byte_capacity = (
            service.byte_capacity_per_s / node.transaction_bytes)
        if service.request_capacity_per_s is None:
            service_capacity = byte_capacity
        else:
            service_capacity = min(
                byte_capacity, service.request_capacity_per_s)
        gamma = node.transactions / demand.initial_transactions
        service_capacity_by_role[node.resource_role] = service_capacity
        initial_capacity_by_role[node.resource_role] = service_capacity / gamma
    lambda_capacity = min(initial_capacity_by_role.values())
    if not math.isfinite(lambda_capacity) or lambda_capacity <= 0:
        raise ValueError("memory path produced a non-positive capacity")

    # A closed finite population reaches service saturation exactly when its
    # unloaded bandwidth-delay product fits.  This is the physical capacity
    # branch; evaluating open M/D/1 at rho=1 would instead diverge.
    capacity_state = _memory_state(
        demand, hierarchy, services, queue_capacities, lambda_capacity,
        include_open_queue_wait=False)
    if capacity_state["phi"] >= lambda_capacity:
        solved_rate = lambda_capacity
        capacity_bound = True
        solved_state = capacity_state
    else:
        capacity_bound = False
        low = 0.0
        high = math.nextafter(lambda_capacity, 0.0)
        low_state = _memory_state(
            demand, hierarchy, services, queue_capacities, low,
            include_open_queue_wait=True)
        high_state = _memory_state(
            demand, hierarchy, services, queue_capacities, high,
            include_open_queue_wait=True)
        low_value = low - low_state["phi"]
        high_value = high - high_state["phi"]
        if not low_value < 0.0:
            raise RuntimeError("memory fixed point has no lower bracket")
        if not high_value > 0.0:
            raise RuntimeError("memory fixed point has no upper bracket")
        for _ in range(256):
            midpoint = (low + high) / 2.0
            midpoint_state = _memory_state(
                demand, hierarchy, services, queue_capacities, midpoint,
                include_open_queue_wait=True)
            midpoint_value = midpoint - midpoint_state["phi"]
            if midpoint_value > 0.0:
                high = midpoint
                high_state = midpoint_state
            else:
                low = midpoint
                low_state = midpoint_state
            numerical_tolerance = max(
                math.ulp(lambda_capacity) * 8.0,
                lambda_capacity * 1e-12)
            if high - low <= numerical_tolerance:
                break
        else:
            raise RuntimeError("memory fixed-point bisection did not converge")
        solved_rate = (low + high) / 2.0
        solved_state = _memory_state(
            demand, hierarchy, services, queue_capacities, solved_rate,
            include_open_queue_wait=True)

    saturated_roles = {
        role for role, value in initial_capacity_by_role.items()
        if capacity_bound and math.isclose(
            value, lambda_capacity, rel_tol=1e-12, abs_tol=0.0)
    }
    node_diagnostics = []
    for node in demand.nodes:
        role = node.resource_role
        service = services[role]
        state = solved_state["nodes"][role]
        if role in saturated_roles:
            queue_wait = None
            rho = 1.0
        elif service.queue_model == "md1":
            # The exact-capacity sufficiency branch intentionally evaluates
            # finite populations with unloaded lifetimes.  That does not make
            # a different, sub-saturated M/D/1 node queue-free: its diagnostic
            # still follows the analytical queue law at the solved arrival
            # rate.  Only the saturated open queue has undefined wait.
            rho = state["rho"]
            queue_wait = (
                rho / (2.0 * (1.0 - rho)) * service.service_time_s)
        else:
            queue_wait = state["queue_wait"]
            rho = state["rho"]
        node_diagnostics.append(MemoryNodeDiagnostic(
            resource_role=role,
            transactions=node.transactions,
            bytes=node.bytes,
            byte_capacity_per_s=service.byte_capacity_per_s,
            request_capacity_per_s=service.request_capacity_per_s,
            service_capacity_per_s=service_capacity_by_role[role],
            arrival_rate_per_s=state["arrival_rate"],
            rho=rho,
            queue_wait_s=queue_wait,
            capacity_time_s=(
                node.transactions / service_capacity_by_role[role]),
            saturated=role in saturated_roles,
            unloaded_latency_s=service.unloaded_latency_s,
        ))
    queue_diagnostics = tuple(
        QueueDiagnostic(
            name=name,
            allocation_rate_per_s=state["allocation_rate"],
            lifetime_s=state["lifetime"],
            used_entries=state["used_entries"],
            max_entries=queue_capacities[name].max_entries,
            initial_request_rate_capacity_per_s=state["initial_capacity"],
        )
        for name, state in solved_state["queues"].items()
    )
    first_request_latency = sum(
        (node.transactions / demand.initial_transactions)
        * services[node.resource_role].unloaded_latency_s
        for node in demand.nodes)
    if capacity_bound:
        if saturated_roles:
            bottleneck = "+".join(
                f"service:{role}" for role in sorted(saturated_roles))
        else:
            raise AssertionError(
                "capacity-bound memory path has no saturated service")
    else:
        bottleneck = solved_state["bottleneck"]
    return MemoryDiagnostics(
        path_name=demand.name,
        access_kind=demand.access_kind,
        rate_unit="transactions",
        initial_transactions=demand.initial_transactions,
        lambda_initial_per_s=solved_rate,
        lambda_capacity_per_s=lambda_capacity,
        workload_slots_per_unit=solved_state["workload_slots"],
        hardware_source_slots_per_unit=solved_state["hardware_slots"],
        source_slots_per_unit=solved_state["source_slots"],
        source_lifetime_s=solved_state["source_lifetime"],
        source_rate_capacity_per_s=solved_state["source_capacity"],
        frontend_rate_capacity_per_s=solved_state["frontend_capacity"],
        steady_time_s=demand.initial_transactions / solved_rate,
        first_request_latency_s=first_request_latency,
        bottleneck=bottleneck,
        capacity_bound=capacity_bound,
        nodes=tuple(node_diagnostics),
        queues=queue_diagnostics,
    )


def _access_class_capacity(service, grid_size):
    """Return the measured capacity at or below this launch's CTA count.

    The table is measured, ascending in grid, and never extrapolated: a launch
    smaller than the first measured grid keeps the entry capacity, and one
    larger than the last keeps the last measured point rather than projecting
    past the sweep.
    """

    table = service.logical_byte_capacity_per_s_by_grid_ctas
    if table is None or grid_size is None:
        return service.logical_byte_capacity_per_s
    resolved = None
    for grid, capacity in table:
        if grid <= grid_size:
            resolved = capacity
    if resolved is None:
        return service.logical_byte_capacity_per_s
    return resolved


def solve_memory_access_classes(
        demands, services, *,
        grid_size=None) -> tuple[MemoryAccessClassDiagnostic, ...]:
    """Resolve exact full-chip typed access-class capacities.

    These logical-byte services are independent lower bounds.  They do not
    mutate physical hierarchy traffic, consume DRAM arbitration capacity, or
    scale with the active execution-domain fraction.
    """

    if not isinstance(demands, tuple):
        raise TypeError("memory access-class demands must be a tuple")
    if any(not isinstance(path, MemoryPathDemand) for path in demands):
        raise TypeError(
            "memory access-class demands must contain MemoryPathDemand")
    if not isinstance(services, tuple):
        raise TypeError("memory access-class services must be a tuple")
    if any(not isinstance(service, MemoryAccessClassService)
           for service in services):
        raise TypeError(
            "memory access-class services must contain "
            "MemoryAccessClassService")

    service_by_identity = {
        (
            service.service,
            service.access_kind,
            service.ingress_resource_role,
            service.signature,
        ): service
        for service in services
    }
    if len(service_by_identity) != len(services):
        raise ValueError(
            "memory access-class services repeat an exact signature")
    service_names = {service.service for service in services}
    diagnostics = []
    for path in demands:
        for demand in path.access_class_demands:
            if demand.service not in service_names:
                raise ValueError(
                    "memory path declares access-class service without a "
                    f"hardware capability: {demand.service!r}")
            identity = (
                demand.service,
                path.access_kind,
                demand.ingress_resource_role,
                demand.signature,
            )
            if identity not in service_by_identity:
                raise ValueError(
                    "memory access-class demand has no exact hardware "
                    "signature: "
                    f"service={demand.service!r}, "
                    f"access_kind={path.access_kind!r}, "
                    f"ingress={demand.ingress_resource_role!r}, "
                    f"signature={demand.signature!r}")
            service = service_by_identity[identity]
            capacity = _access_class_capacity(service, grid_size)
            service_time = demand.logical_bytes / capacity
            diagnostics.append(MemoryAccessClassDiagnostic(
                path_name=path.name,
                service=demand.service,
                access_kind=path.access_kind,
                ingress_resource_role=demand.ingress_resource_role,
                signature=demand.signature,
                scope=service.scope,
                logical_bytes=demand.logical_bytes,
                logical_byte_capacity_per_s=capacity,
                service_time_s=service_time,
                evidence_ref=service.evidence_ref,
            ))
    return tuple(diagnostics)


def solve_memory_arbitration(
        demands, diagnostics, *, resource_role, arbitration_mode,
        read_capacity_per_s, write_capacity_per_s):
    """Price directional paths once under the declared hardware contract.

    Shared modes are half duplex and therefore add directional service times.
    An independent-directional endpoint overlaps read and write service, so
    its lower bound is their maximum.  This resolver intentionally has no
    direction-switch input.
    """

    if not isinstance(demands, tuple):
        raise TypeError("memory arbitration demands must be a tuple")
    if not isinstance(diagnostics, tuple):
        raise TypeError("memory arbitration diagnostics must be a tuple")
    if any(not isinstance(value, MemoryPathDemand) for value in demands):
        raise TypeError(
            "memory arbitration demands must contain MemoryPathDemand")
    if any(not isinstance(value, MemoryDiagnostics)
           for value in diagnostics):
        raise TypeError(
            "memory arbitration diagnostics must contain MemoryDiagnostics")
    role = resource_role.strip() if isinstance(resource_role, str) else ""
    if not role:
        raise ValueError(
            "memory arbitration resource_role must be a non-empty string")
    if arbitration_mode not in (
            "shared_common_capacity", "shared_directional_capacity",
            "independent_directional_capacity"):
        raise ValueError(
            f"unknown memory arbitration mode: {arbitration_mode!r}")
    capacities = {
        "read": read_capacity_per_s,
        "write": write_capacity_per_s,
    }
    for access_kind, capacity in capacities.items():
        if (isinstance(capacity, bool)
                or not isinstance(capacity, (int, float))
                or not math.isfinite(capacity)
                or capacity <= 0):
            raise ValueError(
                f"{access_kind} arbitration capacity must be finite and "
                "positive")
        capacities[access_kind] = float(capacity)
    if (arbitration_mode == "shared_common_capacity"
            and not math.isclose(
                capacities["read"], capacities["write"],
                rel_tol=1e-12, abs_tol=0.0)):
        raise ValueError(
            "shared_common_capacity requires equal read/write capacity")

    demand_names = tuple(value.name for value in demands)
    diagnostic_names = tuple(value.path_name for value in diagnostics)
    if (len(demand_names) != len(set(demand_names))
            or len(diagnostic_names) != len(set(diagnostic_names))):
        raise ValueError("memory arbitration path names must be unique")
    if set(demand_names) != set(diagnostic_names):
        raise ValueError(
            "memory arbitration demand/diagnostic path names disagree")
    diagnostic_by_path = {
        value.path_name: value for value in diagnostics}
    bytes_by_access = {"read": 0.0, "write": 0.0}
    time_by_access = {"read": 0.0, "write": 0.0}
    found = False
    for demand in demands:
        diagnostic = diagnostic_by_path[demand.name]
        if diagnostic.access_kind != demand.access_kind:
            raise ValueError(
                f"memory path {demand.name!r} access kind changed during "
                "solving")
        demand_nodes = tuple(
            value for value in demand.nodes
            if value.resource_role == role)
        diagnostic_nodes = tuple(
            value for value in diagnostic.nodes
            if value.resource_role == role)
        if len(demand_nodes) != len(diagnostic_nodes):
            raise ValueError(
                f"memory path {demand.name!r} has inconsistent {role!r} "
                "demand and diagnostics")
        if not demand_nodes:
            continue
        if len(demand_nodes) != 1:
            raise ValueError(
                f"memory path {demand.name!r} must visit shared resource "
                f"{role!r} exactly once")
        found = True
        demand_node = demand_nodes[0]
        diagnostic_node = diagnostic_nodes[0]
        if not math.isclose(
                demand_node.bytes, diagnostic_node.bytes,
                rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(
                f"memory path {demand.name!r} changed {role!r} bytes during "
                "solving")
        expected_capacity = capacities[demand.access_kind]
        if not math.isclose(
                diagnostic_node.byte_capacity_per_s, expected_capacity,
                rel_tol=1e-12, abs_tol=0.0):
            raise ValueError(
                f"memory path {demand.name!r} did not use the declared "
                f"{demand.access_kind} {role!r} capacity")
        expected_time = demand_node.bytes / expected_capacity
        if not math.isclose(
                diagnostic_node.capacity_time_s, expected_time,
                rel_tol=1e-12, abs_tol=1e-18):
            raise ValueError(
                f"memory path {demand.name!r} has inconsistent {role!r} "
                "capacity time")
        bytes_by_access[demand.access_kind] += demand_node.bytes
        time_by_access[demand.access_kind] += expected_time
    if not found:
        raise ValueError(
            f"memory arbitration has no demand for resource {role!r}")
    if arbitration_mode == "independent_directional_capacity":
        analytical_time = max(
            time_by_access["read"], time_by_access["write"])
    elif arbitration_mode in (
            "shared_common_capacity", "shared_directional_capacity"):
        analytical_time = (
            time_by_access["read"] + time_by_access["write"])
    else:
        raise AssertionError(
            f"unhandled memory arbitration mode: {arbitration_mode!r}")
    return MemoryArbitrationDiagnostic(
        resource_role=role,
        arbitration_mode=arbitration_mode,
        read_bytes=bytes_by_access["read"],
        write_bytes=bytes_by_access["write"],
        read_capacity_per_s=capacities["read"],
        write_capacity_per_s=capacities["write"],
        read_service_time_s=time_by_access["read"],
        write_service_time_s=time_by_access["write"],
        analytical_time_s=analytical_time,
        effective_time_s=analytical_time,
        switch_count=None,
        read_to_write_turnaround_s=None,
        write_to_read_turnaround_s=None,
        bound_kind="lower_bound",
        schedule_status="unavailable",
        overridden=False,
    )
