"""Build GPU device resources over an existing, verified orbital dataset."""

from collections import defaultdict
from dataclasses import dataclass
import json
from astermind.builder.config import SimulationConfig


from astermind.builder.config.deployment import SceneDeployment, ServingCell


@dataclass(frozen=True)
class CellLayout:
    deployment: SceneDeployment
    satellite_gpus: dict[str, int]
    rank_devices: tuple[str, ...] | None = None
    scale_up_groups: tuple[int, ...] = ()
    scale_out: tuple[float, int] | None = None
    su_limit: int | None = None
    serving_cell: ServingCell | None = None


def build_scene(
    dataset, deployment, *, requests=(), migrations=(), runtime_network=None, routing=None,
    cell_selection='least_loaded', cell_layouts=None,
):
    from astermind.builder.physical.physical_dataset import load_dataset
    from astermind.builder.catalog import fast_kv_block_size
    from astermind.engine import NetworkSpec, candidate_links
    from astermind.engine.state.physical_dataset import PhysicalAttributes

    deployment = SceneDeployment.model_validate(deployment)
    batch = deployment.compute_mode == "fast_mode"
    communication = dict(deployment.communication)
    if batch:
        communication.setdefault("intra_mode", "surrogate")
        communication.setdefault("default_hardware_id", deployment.hardware_id)
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
    overrides = cell_layouts or {}
    if set(overrides) - groups.keys():
        raise ValueError("部署必须选择拓扑中已有的计算 Cell")
    cell_layouts = {
        identity: overrides[identity] if identity in overrides else
        CellLayout(deployment, dict.fromkeys(members, deployment.gpus_per_satellite))
        for identity, members in groups.items()
    }
    inventories = {}
    for identity, layout in cell_layouts.items():
        options = layout.deployment
        if options.compute_mode != deployment.compute_mode:
            raise ValueError("同一原生场景中的 Cell 必须使用相同计算模式")
        if set(layout.satellite_gpus) != set(groups[identity]):
            raise ValueError(f"Cell {identity} 的 GPU 清单必须覆盖且仅覆盖其卫星")
        if any(type(count) is not int or not 1 <= count <= 64 for count in layout.satellite_gpus.values()):
            raise ValueError(f"Cell {identity} 的每星 GPU 数必须为 1 到 64")
        for owner, count in layout.satellite_gpus.items():
            inventories[owner] = layout, count
    if batch or overrides:
        hardware = {identity: layout.deployment.hardware_id for identity, layout in cell_layouts.items()}
        if any(value != communication.get("default_hardware_id") for value in hardware.values()):
            policies = communication.setdefault("cell_policies", {})
            for identity, value in hardware.items():
                policies.setdefault(identity, {}).setdefault("hardware_id", value)
        if any(layout.deployment.dp > 1 for layout in cell_layouts.values()):
            communication.setdefault("ep_strategy", "direct")
        for identity, layout in overrides.items():
            policies = communication.setdefault("cell_policies", {})
            cell_policy = layout.deployment.communication.get("cell_policies", {}).get(identity, {})
            policies.setdefault(identity, {}).update(cell_policy)
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
    intra_hierarchies = {}
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
        if owner not in inventories:
            continue
        layout, gpu_count = inventories[owner]
        options = layout.deployment
        limit = layout.su_limit or gpu_count
        group_sizes = layout.scale_up_groups or tuple(min(limit, gpu_count - start) for start in range(0, gpu_count, limit))
        if any(type(n) is not int or n < 1 for n in group_sizes) or sum(group_sizes) != gpu_count:
            raise ValueError(f"{owner} 的 SU 分组必须覆盖且仅覆盖其 GPU")
        if len(group_sizes) > 1 and layout.scale_out is None:
            raise ValueError(f"{owner} 的多 SU 互联需要域间带宽与时延")
        switch_ids, gpu_groups = [], []
        offset = 0
        for index, size in enumerate(group_sizes):
            fabric = owner + "/fabric" if index == 0 else f"{owner}/su-fabric-{index}"
            bw, latency = options.gpu_link_bandwidth_gbps, options.gpu_link_latency_ns
            devices.append(dict(id=fabric, satellite=owner, kind="switch", capacity_gbps=size * bw))
            # Each SU has its own uplink to the shared satellite egress.
            # A transfer between SUs uses the source and destination uplinks.
            target = owner + "/terminal"
            uplink_bw, uplink_latency = layout.scale_out if len(group_sizes) > 1 else (bw, 0)
            links.append(dict(id=owner + "/egress" if index == 0 else f"{owner}/scale-out-{index}",
                              source=fabric, target=target, bandwidth_gbps=uplink_bw, latency_ns=uplink_latency))
            gpu_group = []
            for slot in range(offset, offset + size):
                gpu = f"{owner}/gpu{slot}"
                devices.append(dict(id=gpu, satellite=owner, kind="gpu"))
                links.append(dict(id=gpu + "/port", source=gpu, target=fabric,
                                  bandwidth_gbps=bw, latency_ns=latency))
                gpu_group.append(gpu)
            offset += size
            switch_ids.append(fabric)
            gpu_groups.append(gpu_group)
        if batch or layout.su_limit is not None:
            intra_hierarchies[owner] = dict(su_groups=gpu_groups, su_switches=switch_ids)
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
        if identity not in cell_layouts:
            continue
        layout = cell_layouts[identity]
        options = layout.deployment
        if layout.serving_cell is not None:
            if layout.serving_cell.id != identity:
                raise ValueError(f'Cell {identity} 的部署标识不匹配')
            cells.append(layout.serving_cell.model_dump(mode='json'))
            continue
        slots = [
            f"{member}/gpu{slot}"
            for member in sorted(members)
            for slot in range(layout.satellite_gpus[member])
        ]
        workers = options.tp * options.pp * options.dp
        if workers > len(slots):
            raise ValueError(
                f"Cell {identity} 的 TP×PP×DP={workers} 超过可用 GPU 数 {len(slots)}"
            )
        ranks = tuple(slots[:workers]) if layout.rank_devices is None else layout.rank_devices
        if len(ranks) != workers or len(set(ranks)) != workers or not set(ranks) <= set(slots):
            raise ValueError(f"Cell {identity} 的 rank 必须逐一映射到本 Cell 内不同的 GPU")
        parallel = dict(tp=options.tp, pp=options.pp, dp=options.dp)
        if batch and options.expert_parallel:
            parallel["ep"] = options.tp * options.dp
        cells.append(
            dict(
                id=identity,
                model=options.model,
                hardware=options.hardware,
                hardware_id=options.hardware_id,
                parallel=parallel,
                scheduler_adapter="vllm_0_30" if batch else options.scheduler_adapter,
                detail_prediction_method=None if batch else options.detail_prediction_method,
                native_expert_parallel=options.expert_parallel,
                wide_ep_enabled=options.wide_ep_enabled,
                fast_glm_dsa_score=options.fast_glm_dsa_score, fast_roofline=options.fast_roofline,
                fast_kv_pool_blocks=options.fast_kv_pool_blocks if batch else None,
                fast_link_contention=options.fast_link_contention,
                expert_routing_seed=options.expert_routing_seed,
                placements=[
                    dict(rank=rank, device=gpu)
                    for rank, gpu in enumerate(ranks)
                ],
                ingress_device=ranks[0],
                output_device=ranks[options.tp * (options.pp - 1)],
                max_model_len=options.max_model_len,
                max_batched_tokens=options.max_batched_tokens,
                max_requests=options.max_requests,
                num_kv_blocks=options.num_kv_blocks,
                kv_block_size_tokens=fast_kv_block_size(options) if batch else options.kv_block_size_tokens,
                prefix_caching=options.prefix_caching,
                enable_chunked_prefill=options.enable_chunked_prefill,
                long_prefill_token_threshold=options.long_prefill_token_threshold,
                memory_policy_file=options.memory_policy_file if batch else None,
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
    if intra_hierarchies:
        physical["intra_hierarchies"] = intra_hierarchies
    value = dict(
        schema_version="orbitserve.native_cells.v1",
        physical=physical,
        physical_dataset=str(root),
        physical_link_bindings=bindings,
        ground_owners=grounds,
        satellite_roles={
            n.node_id: "compute" if n.role == "compute" else "communication"
            for n in spec.nodes
            if n.role != "ground" and (n.role != "compute" or n.node_id in inventories)
        },
        cells=cells,
        requests=list(requests),
        migrations=list(migrations),
        compute_mode=deployment.compute_mode,
        detail_prediction_method=deployment.detail_prediction_method,
        communication=communication,
        routing=routing or {},
        cell_selection=cell_selection,
        horizon_ns=round(concrete["time"]["duration_s"] * 1e9),
        runtime_network=runtime_network or {},
    )
    return SimulationConfig.model_validate(value)
