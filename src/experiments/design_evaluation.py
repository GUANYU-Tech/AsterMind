"""Shared current-contract evaluator used by fixed designs and DSE."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from orbitserve.architecture import resolve_architecture
from orbitserve.orbit import load_orbit_trace
from orbitserve.runtime import NetworkSpec, candidate_links, run_simulation
from orbitserve.serving import estimate_serving_throughput

from .constellation_cost import estimate_constellation_cost
from .network import TERMINAL_ALLOCATION_MODEL, run_network_experiment
from .network_inputs import materialize_network_input
from .simulation_builder import build_simulation_spec


@dataclass(frozen=True)
class MaterializedDesignEvaluation:
    physical_input: dict
    architecture: dict
    serving: object
    network: dict
    runtime: object
    cost: dict
    runtime_spec: object


def evaluate_materialized_design(
    spec,
    *,
    physical_network,
    workload_generation,
    output_dir,
    strategy=None,
    progress=None,
    _workers=None,
):
    root = Path(output_dir)
    if spec.physical_dataset:
        from .physical_dataset import dataset_physical_input
        concrete = dataset_physical_input(spec.physical_dataset, physical_network, spec.base_dir or '.')
    else:
        concrete = materialize_network_input(physical_network, spec.base_dir or ".")
    compute = resolve_architecture(spec.hardware)
    # A distinct role profile is explicit; otherwise identical hardware is shared.
    communication = (
        resolve_architecture(spec.communication_hardware)
        if spec.communication_hardware
        else compute
    )
    role_profiles = {"compute": compute, "communication": communication}
    network_spec = NetworkSpec.from_value(concrete["network"])
    for node in network_spec.nodes:
        if node.role == "ground":
            continue
        profile = role_profiles[node.role].hardware_profile
        if node.isl_terminals > profile.laser_terminal_count:
            raise ValueError(
                f"{node.node_id}: network ISL terminals exceed Architecture laser_terminal_count"
            )
    node_catalog = {n.node_id: n for n in network_spec.nodes}
    for edge in candidate_links(network_spec):
        profile = network_spec.link_profiles[edge.get("profile_id", edge["kind"])]
        for endpoint in (edge["endpoint_a"], edge["endpoint_b"]):
            node = node_catalog[endpoint]
            if node.role == "ground":
                continue
            hardware = role_profiles[node.role].hardware_profile
            cap = (
                hardware.ground_link_bandwidth_gbps
                if edge["kind"] == "sgl"
                else hardware.laser_terminal_bandwidth_gbps
            ) * 1e9
            directional = [network_spec.link_profiles[name] for name in
                           (profile.uplink_profile, profile.downlink_profile) if name]
            if any(p.bandwidth_bps > cap + 1e-6 for p in (profile, *directional)):
                raise ValueError(
                    f"{edge['link_id']}: network rate exceeds Architecture terminal bandwidth"
                )
    cells = {}
    for node in network_spec.nodes:
        if node.role == "compute":
            cells.setdefault(node.cell_id or node.node_id, []).append(node)
    sizes = {len(members) for members in cells.values()}
    if sizes != {spec.serving.formation.member_count}:
        raise ValueError(
            "Serving formation.member_count must match every physical compute cell"
        )
    serving = estimate_serving_throughput(
        replace(spec.serving, hardware=None), hardware=compute.hardware_profile
    )
    if not serving.evaluated or not serving.feasible:
        raise ValueError(
            "Serving model is not feasible on the configured Architecture/formation"
        )
    requests = workload_generation.get(
        "session_requests", workload_generation.get("requests", ())
    )
    for request in requests:
        if request.get("model_family") != spec.serving.model.name:
            raise ValueError("Workload model_family must match Serving model.name")
    # Cached geometry must exactly match the resolved input, not just the file name.
    network_dir = root / "network"
    if spec.physical_dataset:
        from .physical_dataset import reuse_dataset
        if progress:
            progress("Loading complete physical dataset (no physical recomputation)")
        network = reuse_dataset(spec.physical_dataset, network_dir, concrete, base_dir=spec.base_dir or '.', include_exports=False)
    elif (network_dir / "network_result.json").exists():
        from .physical_dataset import load_dataset
        _, _, network = load_dataset(network_dir, verify_exports=False)
        stored = json.loads((network_dir / "materialized_input.json").read_text(encoding='utf-8'))
        if stored != json.loads(json.dumps(concrete)):
            raise ValueError("cached network input differs from current experiment")
        if network.get("provenance", {}).get("between_samples") != "sample_hold":
            raise ValueError(
                "cached network uses obsolete interpolation semantics; choose a new output directory"
            )
        if network.get('provenance', {}).get('terminal_allocation_model') != TERMINAL_ALLOCATION_MODEL:
            raise ValueError('cached network uses obsolete terminal/capacity semantics; choose a new output directory')
    else:
        network = run_network_experiment(
            physical_network,
            network_dir,
            base_dir=spec.base_dir or ".",
            progress=progress,
            _workers=_workers,
            export_mode='core',
        )
    if progress:
        progress("Loading sampled Orbit trace")
    trace = load_orbit_trace(network_dir / "orbit_trace/trace_manifest.json")
    energy = compute.energy_profile.to_dict()
    energy["node_profiles"] = {
        n.node_id: role_profiles[n.role].energy_profile.to_dict()
        for n in network_spec.nodes
        if n.role != "ground"
    }
    # Ground coordinates in the Orbit input are authoritative; workload IDs must match.
    ground = {g["ingress_id"]: g for g in concrete["ground_ingresses"]}
    for station in workload_generation.get("gateways", ()):
        station_id = station.get("gateway_id")
        if station_id in ground:
            for coordinate in ("lat_deg", "lon_deg"):
                if not math.isclose(
                    float(station[coordinate]),
                    float(ground[station_id][coordinate]),
                    abs_tol=1e-5,
                ):
                    raise ValueError(
                        f"Workload and physical network disagree on {station_id} {coordinate}"
                    )
    requests = workload_generation.get(
        "session_requests", workload_generation.get("requests", ())
    )
    for request in requests:
        if request["ingress_id"] not in ground:
            raise ValueError(
                f"workload ingress is absent from physical network: {request['ingress_id']}"
            )
    runtime_spec = build_simulation_spec(
        scenario_id=spec.experiment_id,
        design_id=spec.experiment_id,
        workload_generation={**workload_generation, "gateways": tuple(ground.values())},
        constellation_architecture=compute.hardware_profile,
        serving_profile=serving,
        simulation_result=trace,
        simulation_config={
            "network": network_spec.to_dict(),
            "step_s": concrete["time"]["step_s"],
        },
        energy_profile=energy,
        physical_state_ref=str(network_dir / 'physical_state.sqlite')
        if (network_dir / 'physical_state.sqlite').is_file() else None,
        strategy=strategy or spec.runtime_strategy,
    )
    if progress:
        progress("Chronological Runtime request execution")
    runtime = run_simulation(runtime_spec)
    if progress:
        summary = runtime.request_summary
        progress(
            "Runtime completed: "
            + str(summary["completed_request_count"])
            + "/"
            + str(summary["total_request_count"])
        )
    compute_cost = estimate_constellation_cost(
        {"compute_node_count": len(cells)},
        spec.costing,
        resolved_architecture=compute,
        formation_member_count=spec.serving.formation.member_count,
    )
    communication_count = sum(n.role == "communication" for n in network_spec.nodes)
    comm_cost = communication.cost_profile
    cost = {
        "compute": compute_cost.to_dict(),
        "communication_satellite_count": communication_count,
        "communication_single_satellite": comm_cost.to_dict(),
        "total_cost_usd": compute_cost.total_cost_usd
        + communication_count * comm_cost.total_cost_usd,
        "annualized_cost_usd_per_year": compute_cost.annualized_cost_usd_per_year
        + communication_count * comm_cost.annualized_cost_usd_per_year,
    }
    return MaterializedDesignEvaluation(
        concrete,
        {role: value.to_dict() for role, value in role_profiles.items()},
        serving,
        network,
        runtime,
        cost,
        runtime_spec,
    )


def safe_trace_path_component(value: Any) -> str:
    """Return one deterministic path component for a trace identity.

    Design and experiment identifiers are user input.  Keeping path separators
    or ``..`` in them would let a DSE row escape its trace root; replacing an
    unsafe identifier without a digest would also make distinct identifiers
    collide.  Safe short identifiers stay readable, while normalized or long
    identifiers receive a stable digest suffix.
    """

    text = str(value or "trace").strip()
    normalized = "".join(
        character if character.isalnum() or character in {"-", "_", "."} else "_"
        for character in text
    ).strip("._")
    if not normalized:
        normalized = "trace"
    unsafe = (
        normalized != text
        or text in {".", ".."}
        or len(normalized.encode("utf-8")) > 120
    )
    if not unsafe:
        return normalized
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    readable = _truncate_utf8(normalized, 96).rstrip("._-") or "trace"
    return f"{readable}.hash-{digest}"


def _truncate_utf8(value: str, max_bytes: int) -> str:
    byte_count = 0
    characters: list[str] = []
    for character in value:
        encoded_size = len(character.encode("utf-8"))
        if byte_count + encoded_size > max_bytes:
            break
        characters.append(character)
        byte_count += encoded_size
    return "".join(characters)
