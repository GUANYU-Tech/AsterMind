"""Physical presets and resource validation, independent of model parallelism."""

import copy

from astermind.builder.topology import PhysicalTopology
from astermind.builder.models import Scenario


def expand_presets(physical):
    data = copy.deepcopy(physical)
    templates = []
    for preset in data.get("templates", []):
        if preset["kind"] != "mesh":
            templates.append(preset)
            continue
        rows, columns = preset["rows"], preset["columns"]
        members = preset["devices"]
        if (
            type(rows) is not int
            or type(columns) is not int
            or min(rows, columns) < 1
            or rows * columns != len(members)
        ):
            raise ValueError("Mesh dimensions must exactly cover its devices")
        if len(set(members)) != len(members):
            raise ValueError("Mesh devices must be distinct")
        unknown = set(preset) - {
            "id",
            "kind",
            "devices",
            "rows",
            "columns",
            "bandwidth_gbps",
            "latency_ns",
        }
        if unknown:
            raise ValueError(f"unknown Mesh fields: {unknown}")
        for row in range(rows):
            for column in range(columns):
                here = row * columns + column
                neighbors = ([here + 1] if column + 1 < columns else []) + (
                    [here + columns] if row + 1 < rows else []
                )
                for other in neighbors:
                    data.setdefault("links", []).append(
                        {
                            "id": f"{preset['id']}.{here}.{other}",
                            "source": members[here],
                            "target": members[other],
                            "bandwidth_gbps": preset["bandwidth_gbps"],
                            "latency_ns": preset.get("latency_ns", 0),
                        }
                    )
    data["templates"] = templates
    return data


def compile_topology(config):
    physical = expand_presets(config.physical)
    physical["dynamic_external"] = bool(
        config.physical_dataset or config.physical_network
    )
    topology = PhysicalTopology(Scenario.model_validate(physical))
    topology.environment_report = {
        "source": "verified_physical_dataset"
        if physical["dynamic_external"]
        else "declared_static_resources"
    }
    topology.orbit_grid = None
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
        if (role == "communication" and computes) or (
            role in ("compute", "mixed") and not computes
        ):
            raise ValueError(
                f"satellite {satellite}: role does not match GPU resources"
            )
        if role == "mixed" and not external:
            raise ValueError(
                f"satellite {satellite}: role does not match external access"
            )
    for cell in config.cells:
        if any(p.pool != "main" for p in cell.placements):
            raise ValueError("first serving entry point requires the main pool")
        for placement in cell.placements:
            if placement.device in gpu_devices:
                raise ValueError("two Cells cannot occupy the same physical GPU")
            gpu_devices.add(placement.device)
            d = topology.devices.get(placement.device)
            if (
                d is None
                or d.kind != "gpu"
                or topology.satellite_to_cell[d.satellite] != cell.id
            ):
                raise ValueError(f"placement is outside physical Cell {cell.id}")
        local = {
            d.id
            for d in topology.devices.values()
            if topology.satellite_to_cell[d.satellite] == cell.id
        }
        if cell.ingress_device not in local or cell.output_device not in local:
            raise ValueError(
                "Cell ingress/output must belong to its physical resources"
            )
        placement_by_rank = {p.rank: p.device for p in cell.placements}
        if cell.ingress_device != placement_by_rank[0]:
            raise ValueError(
                "first serving entry point delivers input to first-stage rank zero"
            )
        output_rank = cell.parallel.tp * (cell.parallel.pp - 1)
        if cell.output_device != placement_by_rank[output_rank]:
            raise ValueError(
                "Cell output must originate at the final pipeline stage leader"
            )
        terminals = []
        for identifier in cell.external_terminals:
            d = topology.devices.get(identifier)
            if (
                d is None
                or identifier not in local
                or d.kind != "terminal"
                or not d.external_access
            ):
                raise ValueError("Cell egress needs an external terminal in that Cell")
            terminals.append(d)
        boundary_terminals = set()
        for link in topology.links:
            a, b = topology.devices[link.source], topology.devices[link.target]
            if (
                topology.satellite_to_cell[a.satellite]
                == topology.satellite_to_cell[b.satellite]
            ):
                continue
            for device in (a, b):
                if device.id in local:
                    boundary_terminals.add(device.id)
        if not boundary_terminals <= set(cell.external_terminals):
            raise ValueError(
                "all physical Cell exits must be declared in external_terminals"
            )
        satellites = {d.satellite for d in terminals}
        if cell.egress_mode == "single_gateway" and len(satellites) != 1:
            raise ValueError("single gateway requires one egress satellite")
        if cell.egress_mode == "multi_gateway" and len(satellites) < 2:
            raise ValueError("multiple gateways require at least two egress satellites")
        if cell.egress_mode == "distributed":
            compute_satellites = {
                topology.devices[p.device].satellite for p in cell.placements
            }
            if not compute_satellites <= satellites:
                raise ValueError(
                    "distributed egress requires access on every occupied compute satellite"
                )
    for request in config.requests:
        if request.user_device not in topology.devices:
            raise ValueError("request user endpoint is unknown")
        if topology.devices[request.user_device].satellite not in ground:
            raise ValueError("user endpoint must belong to a declared ground owner")
    return topology
