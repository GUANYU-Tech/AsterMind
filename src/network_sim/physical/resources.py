"""Compile the physical topology independently of serving placement."""
import copy

from .environment import prepare_environment
from .model import Scenario
from .topology import PhysicalTopology


def expand_presets(physical):
    data = copy.deepcopy(physical)
    templates = []
    for preset in data.get("templates", []):
        if preset["kind"] != "mesh":
            templates.append(preset)
            continue
        rows, columns = preset["rows"], preset["columns"]
        members = preset["devices"]
        if type(rows) is not int or type(columns) is not int or min(rows, columns) < 1 or rows * columns != len(members):
            raise ValueError("Mesh dimensions must exactly cover its devices")
        if len(set(members)) != len(members):
            raise ValueError("Mesh devices must be distinct")
        unknown = set(preset) - {"id", "kind", "devices", "rows", "columns", "bandwidth_gbps", "latency_ns"}
        if unknown:
            raise ValueError(f"unknown Mesh fields: {unknown}")
        for row in range(rows):
            for column in range(columns):
                here = row * columns + column
                neighbors = ([here + 1] if column + 1 < columns else []) + ([here + columns] if row + 1 < rows else [])
                for other in neighbors:
                    data.setdefault("links", []).append(dict(
                        id=f"{preset['id']}.{here}.{other}", source=members[here], target=members[other],
                        bandwidth_gbps=preset["bandwidth_gbps"], latency_ns=preset.get("latency_ns", 0)))
    data["templates"] = templates
    return data


def build_physical_topology(physical, *, orbit, ground_owners, horizon_ns, link_parameters):
    scenario, environment, grid = prepare_environment(
        expand_presets(physical), orbit=orbit, ground_owners=ground_owners,
        horizon_ns=horizon_ns, link_parameters=link_parameters,
    )
    topology = PhysicalTopology(Scenario.model_validate(scenario))
    topology.environment_report, topology.orbit_grid = environment, grid
    return topology
