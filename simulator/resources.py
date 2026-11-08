"""Validate serving placements and endpoints against the compiled physical topology."""
from network_sim.physical.resources import build_physical_topology, expand_presets as expand_presets


def compile_topology(config):
    topology = build_physical_topology(
        config.physical, orbit=config.orbit, ground_owners=config.ground_owners,
        horizon_ns=config.horizon_ns, link_parameters=config.link_parameters,
    )
    ground = set(config.ground_owners)
    if not ground <= set(topology.satellites):
        raise ValueError("unknown ground owner")
    if ground & set(config.satellite_roles):
        raise ValueError("ground owner cannot have a satellite role")
    gpu_devices = set()
    for satellite, role in config.satellite_roles.items():
        if satellite not in topology.satellites:
            raise ValueError(f"unknown satellite role: {satellite}")
        devices = [d for d in topology.devices.values() if d.satellite == satellite]
        computes = any(d.kind == "gpu" for d in devices)
        external = any(d.external_access for d in devices)
        if (role == "communication" and computes) or (role in ("compute", "mixed") and not computes):
            raise ValueError(f"satellite {satellite}: role does not match GPU resources")
        if role == "mixed" and not external:
            raise ValueError(f"satellite {satellite}: role does not match external access")
    for cell in config.cells:
        for placement in cell.placements:
            if placement.device in gpu_devices:
                raise ValueError("two Cells cannot occupy the same physical GPU")
            gpu_devices.add(placement.device)
            d = topology.devices.get(placement.device)
            if d is None or d.kind != "gpu" or topology.satellite_to_cell[d.satellite] != cell.id:
                raise ValueError(f"placement is outside physical Cell {cell.id}")
        local = {d.id for d in topology.devices.values() if topology.satellite_to_cell[d.satellite] == cell.id}
        if cell.ingress_device not in local or cell.output_device not in local:
            raise ValueError("Cell ingress/output must belong to its physical resources")
        placement_by_rank = {p.rank: p.device for p in cell.placements}
        if cell.ingress_device != placement_by_rank[0]:
            raise ValueError("first serving entry point delivers input to first-stage rank zero")
        output_rank = cell.parallel.tp * (cell.parallel.pp - 1)
        if cell.deployment == "pd_disagg":
            output_rank += cell.parallel.workers
        if cell.output_device != placement_by_rank[output_rank]:
            raise ValueError("Cell output must originate at the final pipeline stage leader")
        terminals = []
        for identifier in cell.external_terminals:
            d = topology.devices.get(identifier)
            if d is None or identifier not in local or d.kind != "terminal" or not d.external_access:
                raise ValueError("Cell egress needs an external terminal in that Cell")
            terminals.append(d)
        boundary_terminals = set()
        for link in topology.links:
            a, b = topology.devices[link.source], topology.devices[link.target]
            if topology.satellite_to_cell[a.satellite] == topology.satellite_to_cell[b.satellite]:
                continue
            for device in (a, b):
                if device.id in local:
                    boundary_terminals.add(device.id)
        if not boundary_terminals <= set(cell.external_terminals):
            raise ValueError("all physical Cell exits must be declared in external_terminals")
        satellites = {d.satellite for d in terminals}
        if cell.egress_mode == "single_gateway" and len(satellites) != 1:
            raise ValueError("single gateway requires one egress satellite")
        if cell.egress_mode == "multi_gateway" and len(satellites) < 2:
            raise ValueError("multiple gateways require at least two egress satellites")
        if cell.egress_mode == "distributed":
            compute_satellites = {topology.devices[p.device].satellite for p in cell.placements}
            if not compute_satellites <= satellites:
                raise ValueError("distributed egress requires access on every occupied compute satellite")
    for request in config.requests:
        if request.user_device not in topology.devices:
            raise ValueError("request user endpoint is unknown")
        if topology.devices[request.user_device].satellite not in ground:
            raise ValueError("user endpoint must belong to a declared ground owner")
    return topology
