"""Native scene capability checks; called explicitly by scene clients."""
from astermind.builder.config import SimulationConfig


def validate_scene_config(value):
    """Validate hand-edited native JSON, including dataset/device correspondence."""
    from astermind.builder.physical.loading import load_physics
    from astermind.engine.network_host import DeviceRuntimeHost

    config = SimulationConfig.model_validate(value)
    if config.backend not in {"flowsim", "simgrid"}:
        raise ValueError("此场景入口要求 Cell 外部通信使用 FlowSim 或 SimGrid")
    if config.backend == 'simgrid':
        from astermind.engine.simgrid_transport import require_simgrid
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
