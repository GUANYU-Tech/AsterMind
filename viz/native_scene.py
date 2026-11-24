"""The restored scene editor calls only V1 builder contracts."""
from copy import deepcopy
from pathlib import Path


def validate_document(root, document):
    from runtime_environment import activate
    activate()
    from astermind.builder.config import SimulationConfig
    from studio import physical_data
    value = deepcopy(document)
    experiment = value.get('experiment')
    if not isinstance(experiment, dict):
        raise ValueError('experiment 必须是完整 V1 配置')
    retired = {'communication_mode', 'timeline_mode', 'protobridge_startup_us'} & experiment.keys()
    if retired:
        raise ValueError('这是旧执行配置；V1 使用 compute_mode 与 communication，请重新生成配置：' + ', '.join(sorted(retired)))
    if value.get('physical_model_id'):
        data = physical_data(root, value['physical_model_id'])
        if data.get('reason'):
            raise ValueError(data['reason'])
        if not data['ready'] or Path(data['physical_dataset']).resolve() != Path(experiment.get('physical_dataset', '')).resolve():
            raise ValueError('场景引用的物理数据与所选已完成模型不一致')
    elif experiment.get('physical_dataset'):
        from maintenance import dataset_islsim_problem
        problem = dataset_islsim_problem(experiment['physical_dataset'])
        if problem:
            raise ValueError(problem)
    config = SimulationConfig.model_validate(experiment)
    if config.compute_mode == 'fast_mode':
        from compute_options import resolve_memory_policy, validate_fast_su_placement
        cells = tuple(cell.model_copy(update={
            'memory_policy_file': resolve_memory_policy(cell.memory_policy_file),
        }) for cell in config.cells)
        for cell in cells:
            validate_fast_su_placement(cell.hardware_id, (p.device for p in cell.placements),
                                       config.physical.get('intra_hierarchies', {}))
        config = config.model_copy(update={'cells': cells})
    from astermind.engine.policies.routing import get_routing_policy
    get_routing_policy(config.routing.policy)
    config.topology()
    value['experiment'] = config.model_dump(mode='json')
    return value, value['experiment']


def generate(root, payload):
    from runtime_environment import enable_device_native
    upstream = enable_device_native()
    from astermind.builder.scene import build_scene
    from compute_options import prepare_deployment
    from studio import physical_data
    if set(payload) - {'physical_model_id', 'deployment', 'cell_deployments', 'requests', 'migrations',
                       'runtime_network', 'routing', 'cell_selection', 'refresh_workloads'}:
        raise ValueError('生成配置包含未知字段')
    data = physical_data(root, payload['physical_model_id'])
    if not data['ready']:
        raise ValueError('请先生成完整物理数据包')
    arguments = dict(requests=payload.get('requests', ()), migrations=payload.get('migrations', ()),
                     runtime_network=payload.get('runtime_network'), routing=payload.get('routing'),
                     cell_selection=payload.get('cell_selection', 'least_loaded'))
    if 'cell_deployments' in payload:
        return build_cell_deployments(data['physical_dataset'], payload, upstream, arguments)
    options = deepcopy(payload.get('deployment', {}))
    if options.get('compute_mode') not in ('fast_mode', 'batch_roofline'):
        options.setdefault('communication', {}).setdefault('intra_mode', 'IntraSatelliteRoofline')
    deployment, hardware = prepare_deployment(options, upstream)
    defaults = {key: hardware[key] for key in ('gpu_link_bandwidth_gbps','gpu_link_latency_ns')
                if key not in deployment.model_fields_set and hardware[key] is not None}
    deployment = deployment.model_copy(update=defaults)
    return build_scene(data['physical_dataset'], deployment, **arguments).model_dump(mode='json')


def build_cell_deployments(dataset, payload, upstream, arguments):
    from astermind.builder.catalog import cell_members
    from astermind.builder.config import SimulationConfig
    from astermind.builder.config.deployment import SceneDeployment
    from astermind.builder.scene import CellLayout, build_scene
    from compute_options import (collective_policy, compute_catalog, detail_prediction_method,
                                 prepare_deployment, resolve_scheduling, scheduler_adapter, su_groups,
                                 validate_fast_su_placement)

    members = cell_members(dataset)
    submitted = payload['cell_deployments']
    if not isinstance(submitted, dict) or set(submitted) != set(members):
        raise ValueError('部署清单必须覆盖物理数据包中的全部 Cell')
    base = SceneDeployment.model_validate(payload.get('deployment', {}))
    catalog = compute_catalog(upstream)
    models = {row['id']: row for row in catalog['models']}
    hardware = {row['id']: row for row in catalog['hardware']}
    from scene_cell_config import editor_layout
    from cell_adapter import capabilities
    editor_catalog = None
    layouts = {}
    for identity, satellites in members.items():
        try:
            row = submitted[identity]
            if isinstance(row, dict) and 'configuration' in row:
                editor_catalog = editor_catalog or capabilities()
                layouts[identity] = editor_layout(identity, satellites, row, base, arguments['requests'], editor_catalog)
                continue
            if not isinstance(row, dict) or set(row) - {
                'deployment', 'satellite_gpus', 'rank_devices', 'scale_up_groups', 'scale_out', 'collective_policies',
            }:
                raise ValueError('无效的逐 Cell 部署字段')
            options = dict(row['deployment'])
            model = models.get(options.get('model'))
            profile = hardware.get(options.get('hardware_id'))
            if model is None or profile is None:
                raise ValueError('请选择后端目录中的模型和 GPU 卡型')
            options.update(hardware=profile['file'], compute_mode=base.compute_mode)
            ep = options.get('expert_parallel', False)
            dp = options.get('dp', 1)
            if type(dp) is not int or dp < 1 or (dp > 1 and not ep) or (ep and not model['experts']):
                raise ValueError('DP 大于 1 需要 MoE 模型和专家并行')
            options.setdefault('tp', 1)
            options.setdefault('pp', 1)
            options.update(dp=dp, expert_parallel=ep)
            options.update(wide_ep_enabled=dp > 1, scheduler_adapter=scheduler_adapter(base.compute_mode, Path(model['id']).stem, dp))
            options.setdefault('expert_routing_seed', 0 if model['experts'] else None)
            options.setdefault('max_requests', 4)
            if base.compute_mode != 'fast_mode' and options.get('detail_prediction_method') in (None, 'auto'):
                options['detail_prediction_method'] = detail_prediction_method(model['id'], profile['id'], upstream)
            elif base.compute_mode == 'fast_mode':
                options['detail_prediction_method'] = None
            counts = row.get('satellite_gpus', dict.fromkeys(satellites, options.get('gpus_per_satellite', 1)))
            if not isinstance(counts, dict) or set(counts) != set(satellites) or any(
                    type(count) is not int or not 1 <= count <= 64 for count in counts.values()):
                raise ValueError('每星 GPU 清单必须覆盖 Cell 成员，且每星有 1 到 64 张 GPU')
            limit = profile['fast_gpu_limit_per_satellite']
            groups = row.get('scale_up_groups', [])
            if groups and len(set(counts.values())) != 1:
                raise ValueError('统一 SU 分组要求各星 GPU 数相同')
            sizes = [su_groups(groups, count, limit) for count in counts.values()]
            scale_out = row.get('scale_out')
            if any(len(su) > 1 for su in sizes):
                if not isinstance(scale_out, dict) or set(scale_out) != {'bandwidth_gbps', 'latency_ns'}:
                    raise ValueError('多个 SU 组需要域间带宽和时延')
                bandwidth, latency = scale_out['bandwidth_gbps'], scale_out['latency_ns']
                if type(bandwidth) not in (int, float) or not 0 < bandwidth < float('inf') or type(latency) is not int or latency < 0:
                    raise ValueError('域间带宽必须为正，时延为非负整数纳秒')
                scale_out = (bandwidth, latency)
            elif scale_out is not None:
                raise ValueError('单 SU 组不使用域间链路参数')
            options.update(gpus_per_satellite=max(counts.values()))
            options.setdefault('gpu_link_bandwidth_gbps', profile['gpu_link_bandwidth_gbps'] or 1600)
            options.setdefault('gpu_link_latency_ns', profile['gpu_link_latency_ns'])
            longest = max((r.get('input_tokens', 0) + r.get('output_tokens', 0) for r in arguments['requests']
                           if r.get('cell') in (None, identity)), default=0)
            options.update(resolve_scheduling(options, model, longest, upstream))
            algorithms = collective_policy(row.get('collective_policies', []))
            options['communication'] = {'cell_policies': {identity: {
                'hardware_id': profile['id'], 'intra_mode': 'surrogate', 'algorithms': algorithms,
            }}}
            deployment, _ = prepare_deployment(options, upstream, largest_su=max(map(max, sizes)), native_detail=True)
            ranks = row.get('rank_devices')
            if ranks is not None and (not isinstance(ranks, list) or any(not isinstance(device, str) for device in ranks)):
                raise ValueError('Rank 映射必须是按 Rank 排列的 GPU ID')
            layouts[identity] = CellLayout(deployment, counts, None if ranks is None else tuple(ranks),
                                           tuple(groups), scale_out, limit)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Cell {identity} · {error}') from error
    scene = build_scene(dataset, base, cell_layouts=layouts, **arguments)
    if scene.compute_mode == 'fast_mode':
        for cell in scene.cells:
            validate_fast_su_placement(cell.hardware_id, (p.device for p in cell.placements),
                                       scene.physical.get('intra_hierarchies', {}))
    scene.topology()
    return SimulationConfig.model_validate(scene.model_dump(mode='json')).model_dump(mode='json')
