"""Authoritative topology commands on the paused Runtime engine thread."""

from ..setup.radio_capacity import hop_pools
from ...orbit import satellite_line_of_sight


def _candidate(net, a, b):
    if a not in net.nodes or b not in net.nodes or a == b:
        raise ValueError('请选择两个存在且不同的节点')
    first, second = net.nodes[a], net.nodes[b]
    for edge in net.edges:
        if {edge['endpoint_a'], edge['endpoint_b']} == {a, b}:
            return edge, False
    if 'ground' in {first.role, second.role}:
        raise ValueError('新增连接目前限卫星之间；星地连接须使用已准备的候选链路')
    if first.cell_id != second.cell_id and any(not net.spec.external_member(n) for n in (first, second)):
        raise ValueError('该成员不符合 cell 对外出口策略')
    if first.role == second.role == 'communication':
        if first.shell_id != second.shell_id:
            raise ValueError('不同通信壳层之间不允许直接建链')
        kind = 'communication_isl'
    elif first.role != second.role:
        kind = 'cross_layer'
    elif first.cell_id and first.cell_id == second.cell_id:
        kind = 'cell_isl'
    elif net.spec.inter_cell_links:
        kind = 'inter_cell'
    else:
        raise ValueError('当前策略不允许计算 cell 之间直接连接')
    a, b = sorted((a, b))
    shell = next((s for s in net.spec.communication_shells if s['shell_id'] == first.shell_id), {})
    profile = shell.get('isl_profile', kind) if kind == 'communication_isl' else kind
    if profile not in net.spec.link_profiles:
        raise ValueError('缺少此类连接的物理链路配置')
    edge = dict(link_id=f'{kind}:{a}:{b}', endpoint_a=a, endpoint_b=b, kind=kind, profile_id=profile)
    hop_pools(net.nodes, a, b)
    hop_pools(net.nodes, b, a)
    return edge, True


def _add_physics(net, edge, now):
    """Prepare only the new pair from saved positions; never rewrite base files."""
    key = edge['endpoint_a'], edge['endpoint_b']
    windows = []
    for a, b in zip(net.environment_times, net.environment_times[1:]):
        if b <= now:
            continue
        if satellite_line_of_sight(net.position(key[0], a), net.position(key[1], a)):
            if windows and windows[-1]['end_s'] == a:
                windows[-1]['end_s'] = b
            else:
                windows.append(dict(start_s=max(a, now), end_s=b))
    net.windows[key] = windows
    net.edges = (*net.edges, edge)
    net.added_edge_ids.add(edge['link_id'])
    net.candidate_edge_ids = net.candidate_edge_ids | {edge['link_id']}


def apply_topology_command(engine, command):
    """Validate and publish atomically at the engine's current simulation time."""
    if not isinstance(command, dict) or not isinstance(command.get('action'), str):
        raise ValueError('拓扑操作必须包含字符串 action')
    net, now = engine.net, engine.now
    if not net.start_s <= now < engine.end:
        raise ValueError('只能在仿真时域内修改拓扑')
    if net.spec.fault_timing == 'boundary' and now not in net.environment_times:
        raise ValueError('边界故障模式只能在时间片起点修改拓扑')
    action = command.get('action')
    target = command.get('target_id')
    if action in {'node_fail', 'node_restore', 'link_fail', 'link_restore'}:
        if not isinstance(target, str):
            raise ValueError('故障对象 ID 必须是字符串')
        kind = action.split('_')[0]
        if target not in (net.nodes if kind == 'node' else net.edge_by_id):
            raise ValueError('故障对象不存在')
        identity = f'manual:{kind}:{target}'
        if action.endswith('restore'):
            if identity not in net.active_faults:
                raise ValueError('该对象没有可解除的人为故障；自然或定制故障按各自条件恢复')
            engine.on_fault_end(None, (identity, net.active_faults[identity]))
        else:
            if identity in net.active_faults:
                raise ValueError('该对象已处于人为故障状态')
            fault = dict(target_type=kind, target_id=target, start_s=now, end_s=net.end_s, source='manual')
            engine.on_fault_start(None, (identity, fault))
        message = '人为故障已解除' if action.endswith('restore') else '人为故障已生效'
    elif action == 'disconnect':
        if not isinstance(target, str):
            raise ValueError('链路 ID 必须是字符串')
        if target not in net.edge_by_id:
            raise ValueError('链路不存在')
        if target in net.manual_disabled_links:
            raise ValueError('链路已经断开')
        net.manual_disabled_links.add(target)
        net.revision += 1
        net._compile(from_s=now)
        net.frame_times = [f['time_s'] for f in net.frames]
        engine.schedule_reallocation()
        engine.interrupt_unavailable('manual_link_disconnected')
        engine.wake_retries()
        message = '链路已断开，后续时间片持续生效'
    elif action == 'connect':
        if not all(isinstance(command.get(k), str) for k in ('endpoint_a', 'endpoint_b')):
            raise ValueError('两个连接端点 ID 必须是字符串')
        edge, added = _candidate(net, command.get('endpoint_a'), command.get('endpoint_b'))
        target = edge['link_id']
        trial = net.fork()
        if added:
            _add_physics(trial, edge, now)
        trial.manual_disabled_links.discard(target)
        attr = trial._geometry_attributes(edge, now)
        if attr['reason'] is not None or trial.faulted(edge):
            reasons = {'geometry_blocked':'节点当前物理不可见', 'out_of_range':'超过最大连接距离',
                       'sun_exclusion':'太阳避让条件不满足', 'injected_fault':'节点或链路仍处于故障状态'}
            raise ValueError(reasons.get(attr['reason'], '节点或链路仍处于故障状态'))
        directions = trial.directions(edge, now, geometry=attr)
        if any(d['bandwidth_bps'] <= 0 for d in directions):
            raise ValueError('双向物理链路容量必须大于零')
        if not added and target not in net.manual_disabled_links and net.link_at(target, now)['state'] in {'UP', 'SETTING_UP'}:
            raise ValueError('该链路已经连接或正在建链')
        trial.revision += 1
        trial._compile(from_s=now)
        trial.frame_times = [f['time_s'] for f in trial.frames]
        row = trial.link_at(target, now)
        if row['state'] == 'DOWN':
            raise ValueError('两端终端资源不足，或连接不满足当前分配条件')
        if row['state'] == 'SETTING_UP' and (row['ready_s'] >= net.end_s or trial.link_at(target, row['ready_s'])['state'] != 'UP'):
            raise ValueError('剩余物理可用窗口不足以完成建链')
        if added and engine.transport and hasattr(engine.transport, 'register_edge'):
            engine.transport.register_edge(edge)
        # Publish only after the complete physical/resource check succeeds.
        net.__dict__.update(trial.__dict__)
        engine.schedule_reallocation()
        engine.interrupt_unavailable('topology_reallocated')
        engine.wake_retries()
        message = '连接成功' if row['state'] == 'UP' else '建链中，等待捕获完成'
    else:
        raise ValueError('不支持的拓扑操作')
    event = dict(time_s=now, kind='topology_command', action=action, target_id=target,
                 source='manual', message=message, topology_revision=net.revision)
    engine.log.append(event)
    return event
