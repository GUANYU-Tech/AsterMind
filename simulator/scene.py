"""Build GPU device resources over an existing, verified orbital dataset."""

from collections import defaultdict
import json
from typing import Annotated, Literal

from pydantic import Field

from network_sim.model import Contract
from .config import Positive, SimulationConfig


class SceneDeployment(Contract):
    timeline_mode: Literal['orbitserve', 'simgrid'] = 'orbitserve'
    model: str = "configs/model/qwen3-8b.json"
    hardware: str = "configs/hardware/nvidia_a800.yaml"
    hardware_id: str = "nvidia_a800"
    gpus_per_satellite: Annotated[int, Field(strict=True, ge=1, le=64)] = 1
    tp: Positive = 2
    pp: Positive = 2
    expert_parallel: bool = False
    expert_routing_seed: Annotated[int, Field(strict=True, ge=0)] | None = None
    max_requests: Annotated[int, Field(strict=True, ge=1, le=1024)] = 4
    max_model_len: Annotated[int, Field(strict=True, ge=2, le=131072)] = 128
    max_batched_tokens: Positive = 32
    num_kv_blocks: Positive = 64
    gpu_link_bandwidth_gbps: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1600
    gpu_link_latency_ns: Annotated[int, Field(strict=True, ge=0)] = 200
    communication_mode: Literal[
        "flowsim", "protobridge_satellite", "protobridge_cell"
    ] = "protobridge_cell"
    protobridge_startup_us: Annotated[float, Field(ge=0, allow_inf_nan=False)] = 0


def build_scene(
    dataset, deployment, *, requests=(), migrations=(), runtime_network=None
):
    from orbitserve.experiments.physical_dataset import load_dataset
    from orbitserve.runtime import NetworkSpec, candidate_links
    from orbitserve.runtime.state.physical_dataset import PhysicalAttributes

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
        backend="simgrid" if deployment.timeline_mode == 'simgrid' else "flowsim",
        timeline_mode=deployment.timeline_mode,
        communication_mode=deployment.communication_mode,
        protobridge_startup_us=deployment.protobridge_startup_us,
        horizon_ns=round(concrete["time"]["duration_s"] * 1e9),
        runtime_network=runtime_network or {},
    )
    return SimulationConfig.model_validate(value)


def validate_scene_config(value):
    """Validate hand-edited native JSON, including dataset/device correspondence."""
    from .physics import load_physics
    from .network import DeviceRuntimeHost

    config = SimulationConfig.model_validate(value)
    if config.backend not in {"flowsim", "simgrid"}:
        raise ValueError("此场景入口要求 Cell 外部通信使用 FlowSim 或 SimGrid")
    if config.backend == 'simgrid':
        from orbitserve.runtime.engine.simgrid_transport import require_simgrid
        require_simgrid()
    if not config.requests:
        raise ValueError("请添加至少一个业务请求")
    if any(r.arrival_ns >= config.horizon_ns for r in config.requests):
        raise ValueError("请求到达时间必须在仿真时域内")
    by_cell = {c.id: c for c in config.cells}
    by_request = {r.id: r for r in config.requests}
    histories = {}
    for request in sorted(config.requests, key=lambda r: r.arrival_ns):
        if request.prefix_hit_tokens and not request.session_id:
            raise ValueError("前缀复用需要 session_id 和对应历史")
        if request.session_id:
            identity = (request.session_id, request.context_id or request.id)
            if identity in histories:
                raise ValueError("会话 context_id 重复")
            if request.parent_context_id:
                parent = histories.get((request.session_id, request.parent_context_id))
                if (
                    parent is None
                    or request.input_tokens < parent.input_tokens + parent.output_tokens
                ):
                    raise ValueError(
                        "会话续接必须引用更早的完整历史，且不能截断已有 token"
                    )
            histories[identity] = request
    for migration in config.migrations:
        trigger = by_request[migration.after_request]
        if (
            trigger.cell != migration.source_cell
            or trigger.session_id != migration.session_id
            or (trigger.context_id or trigger.id) != migration.context_id
        ):
            raise ValueError(
                "KV 迁移必须与触发请求的指定源 Cell、session_id、context_id 一致"
            )
        a, b = by_cell[migration.source_cell], by_cell[migration.target_cell]
        if (a.model, a.parallel.tp, a.parallel.pp, a.kv_block_size_tokens) != (
            b.model,
            b.parallel.tp,
            b.parallel.pp,
            b.kv_block_size_tokens,
        ):
            raise ValueError("KV 迁移的两端必须使用相同模型、TP/PP 分区与 KV 块格式")
    topology = config.topology()
    # GPU ports cannot silently become disconnected from their satellite's
    # terminal. Traverse compiled ports, preserving endpoint-only GPU semantics.
    for cell in config.cells:
        for placement in cell.placements:
            device = topology.devices[placement.device]
            local = {
                n
                for n, d in topology.node_device.items()
                if topology.devices[d].satellite == device.satellite
            }
            terminals = [
                d.id
                for d in topology.devices.values()
                if d.satellite == device.satellite and d.kind == "terminal"
            ]

            def reachable(source, targets):
                seen, todo = {source}, [source]
                while todo:
                    here = todo.pop()
                    for edge in topology.native_edges:
                        if (
                            edge.source == here
                            and edge.target in local
                            and edge.target not in seen
                        ):
                            seen.add(edge.target)
                            todo.append(edge.target)
                return bool(seen & targets)

            if not reachable(
                topology.egress[device.id], {topology.ingress[t] for t in terminals}
            ) or not any(
                reachable(topology.egress[t], {topology.ingress[device.id]})
                for t in terminals
            ):
                raise ValueError(
                    f"GPU {device.id} 与本卫星终端不连通，请恢复 GPU 接入链路"
                )
    physical, lifetime = load_physics(config)
    host = None
    try:
        host = DeviceRuntimeHost(topology, config, physical)
    finally:
        if host is not None:
            host.close()
        if lifetime is not None:
            lifetime.cleanup()
    return config
