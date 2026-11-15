"""Build GPU device resources over an existing, verified orbital dataset."""

from collections import defaultdict
import json
from astermind.builder.config import SimulationConfig


from astermind.builder.config.deployment import SceneDeployment


def build_scene(
    dataset, deployment, *, requests=(), migrations=(), runtime_network=None, routing=None,
    cell_selection='least_loaded'
):
    from astermind.builder.physical.physical_dataset import load_dataset
    from astermind.engine import NetworkSpec, candidate_links
    from astermind.engine.state.physical_dataset import PhysicalAttributes

    deployment = SceneDeployment.model_validate(deployment)
    root, _, _ = load_dataset(dataset)
    concrete = json.loads(
        (root / "materialized_input.json").read_text(encoding="utf-8")
    )
    spec = NetworkSpec.from_value(concrete["network"])
    groups, grounds = defaultdict(list), []
    for node in spec.nodes:
        if node.role == "compute":
            groups[node.cell_id or node.node_id].append(node.node_id)
        elif node.role == "ground":
            grounds.append(node.node_id)
    if not groups or not grounds:
        raise ValueError("原生场景需要计算 Cell 和地面接入节点")
    # The verified export already prunes adaptive pairs with no contact in the
    # horizon. Re-enumerating the combinatorial candidate graph would create
    # device links that the physical Runtime intentionally does not contain.
    attributes = PhysicalAttributes(root / "physical_state.sqlite")
    physical_ids = set(attributes.metadata["link_frames"])
    edges = [edge for edge in candidate_links(spec) if edge["link_id"] in physical_ids]
    degree = defaultdict(int)
    for edge in edges:
        degree[edge["endpoint_a"]] += 1
        degree[edge["endpoint_b"]] += 1
    devices, links, cells = [], [], []
    compute_members = {member for members in groups.values() for member in members}
    for node in spec.nodes:
        owner = node.node_id
        # Terminal concurrency is allocated by physical NetworkState. This
        # aggregate endpoint must not add a second, conflicting allocator.
        devices.append(
            dict(
                id=owner + "/terminal",
                satellite=owner,
                kind="terminal",
                external_access=True,
                max_external_links=max(1, degree[owner]),
            )
        )
        if owner not in compute_members:
            continue
        devices.append(
            dict(
                id=owner + "/fabric",
                satellite=owner,
                kind="switch",
                capacity_gbps=deployment.gpus_per_satellite
                * deployment.gpu_link_bandwidth_gbps,
            )
        )
        links.append(
            dict(
                id=owner + "/egress",
                source=owner + "/fabric",
                target=owner + "/terminal",
                bandwidth_gbps=deployment.gpu_link_bandwidth_gbps,
            )
        )
        for slot in range(deployment.gpus_per_satellite):
            gpu = f"{owner}/gpu{slot}"
            devices.append(dict(id=gpu, satellite=owner, kind="gpu"))
            links.append(
                dict(
                    id=gpu + "/port",
                    source=gpu,
                    target=owner + "/fabric",
                    bandwidth_gbps=deployment.gpu_link_bandwidth_gbps,
                    latency_ns=deployment.gpu_link_latency_ns,
                )
            )
    bindings = {}
    for index, edge in enumerate(edges):
        identifier = f"orbital-link-{index}"
        profile = spec.link_profiles[edge["profile_id"]]
        links.append(
            dict(
                id=identifier,
                source=edge["endpoint_a"] + "/terminal",
                target=edge["endpoint_b"] + "/terminal",
                bandwidth_gbps=profile.bandwidth_bps / 1e9,
            )
        )
        bindings[identifier] = edge["link_id"]
    for identity, members in groups.items():
        slots = [
            f"{member}/gpu{slot}"
            for member in sorted(members)
            for slot in range(deployment.gpus_per_satellite)
        ]
        workers = deployment.tp * deployment.pp
        if workers > len(slots):
            raise ValueError(
                f"Cell {identity} 的 TP×PP={workers} 超过可用 GPU 数 {len(slots)}"
            )
        cells.append(
            dict(
                id=identity,
                model=deployment.model,
                hardware=deployment.hardware,
                hardware_id=deployment.hardware_id,
                parallel=dict(tp=deployment.tp, pp=deployment.pp),
                native_expert_parallel=deployment.expert_parallel,
                expert_routing_seed=deployment.expert_routing_seed,
                placements=[
                    dict(rank=rank, device=gpu)
                    for rank, gpu in enumerate(slots[:workers])
                ],
                ingress_device=slots[0],
                output_device=slots[deployment.tp * (deployment.pp - 1)],
                max_model_len=deployment.max_model_len,
                max_batched_tokens=deployment.max_batched_tokens,
                max_requests=deployment.max_requests,
                num_kv_blocks=deployment.num_kv_blocks,
                egress_mode="distributed",
                external_terminals=[s + "/terminal" for s in members],
            )
        )
    outside = [n.node_id for n in spec.nodes if n.node_id not in compute_members]
    physical = dict(
        cells=[
            dict(id=identity, satellites=members)
            for identity, members in groups.items()
        ]
        + [dict(id="scene-network-resources", satellites=outside)],
        satellites=[dict(id=n.node_id) for n in spec.nodes],
        devices=devices,
        links=links,
    )
    value = dict(
        schema_version="orbitserve.native_cells.v1",
        physical=physical,
        physical_dataset=str(root),
        physical_link_bindings=bindings,
        ground_owners=grounds,
        satellite_roles={
            n.node_id: "compute" if n.role == "compute" else "communication"
            for n in spec.nodes
            if n.role != "ground"
        },
        cells=cells,
        requests=list(requests),
        migrations=list(migrations),
        compute_mode=deployment.compute_mode,
        communication=deployment.communication,
        routing=routing or {},
        cell_selection=cell_selection,
        horizon_ns=round(concrete["time"]["duration_s"] * 1e9),
        runtime_network=runtime_network or {},
    )
    return SimulationConfig.model_validate(value)
