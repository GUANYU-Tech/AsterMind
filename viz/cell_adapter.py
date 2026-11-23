"""Translate the preserved Cell editor into the V1 public configuration contract.

No historical engine is imported. Unsupported semantics fail before submission.
The submitted and effective configurations are both retained for inspection.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
from types import SimpleNamespace

from paths import VIZ, RUNTIME, owned
from runtime_environment import activate, identity


def capabilities():
    activate()
    from astermind.builder.catalog import scene_catalog
    catalog = scene_catalog()
    from runtime_environment import enable_device_native
    import yaml
    upstream=enable_device_native()
    value = json.loads((VIZ / 'cell/presentation.json').read_text())
    value.update(models=[Path(m['id']).stem for m in catalog['models'] if m['available']],
                 hardware=[h['id'] for h in catalog['hardware']], backend=identity())
    value['models_by_hardware'] = {
        h['id']: {Path(m['id']).stem: m['choices'][h['id']] for m in catalog['models'] if m['choices'].get(h['id'])}
        for h in catalog['hardware']
    }
    value['hardware_details']={h['id']:yaml.safe_load((upstream/h['file']).read_text()) for h in catalog['hardware']}
    config = value['default_configuration']
    scheduler = config['scheduler']['main']
    scheduler.update(framework='vllm', sglang=None, vllm=dict(
        model=config['model'], dtype='bfloat16', max_model_len=128, gpu_memory_utilization=.9,
        trust_remote_code=False, enable_chunked_prefill=True, enable_prefix_caching=False,
        async_scheduling=False, dummy_token_id=42, mtp_num_speculative_tokens=0,
        mtp_acceptance_policy='disabled', mtp_fixed_accepted_tokens=0, mtp_acceptance_source_id='disabled',
        mtp_acceptance_replay=None, mtp_acceptance_assumption=None, request_stream_sha256=None,
        decode_admission_contract_sha256=None, closed_loop_admission_contract_sha256=None))
    config['flowsim']['acknowledgement']['enabled'] = False
    for pool in config['deployment']['worker_pools']:
        for collective in pool['collectives']:
            collective['backend'] = 'runtime_comm'
    definitions = value['schema']['$defs']
    definitions['HardwareConfig']['properties']['default_hardware_id']['enum'] = value['hardware']
    definitions['SchedulerConfig']['properties']['framework']['enum'] = ['vllm']
    definitions['CollectiveConfig']['properties']['backend']['enum'] = ['runtime_comm']
    value['compatibility'] = {'unsupported': ['sglang', 'af_disagg', 'pd_af_disagg',
        'FlowSim ACK', 'SimGrid', 'MTP', 'KV offload replay', 'closed-loop admission'],
        'simulation_horizon_s': 60, 'client_boundary': 'explicit zero-payload ground endpoint; no access time model'}
    return value


def schema_validate(value, network_only=False):
    from jsonschema import Draft202012Validator
    schema = deepcopy(capabilities()['schema'])
    if network_only:
        schema['required'] = ['hardware', 'topology', 'constellation', 'fso']
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(e.path))
    if errors:
        raise ValueError('; '.join('.'.join(map(str, e.path)) + ': ' + e.message for e in errors[:5]))


def geometry(config):
    ids = config['hardware']['satellite_ids']
    c = config['constellation']
    if config['topology']['source'] == 'formation':
        f = c['formation']
        count, radius, spacing = len(ids), 6371.0 + f['altitude_km'], f['spacing_km']
        if not 2 <= count <= 35:
            raise ValueError('静态编队需要 2–35 颗卫星')
        if f['layout'] == 'ring':
            r = spacing / (2 * math.sin(math.pi / count))
            radial = math.sqrt(radius ** 2 - r ** 2)
            positions = [(radial, r * math.cos(2*math.pi*i/count-math.pi/2),
                          r * math.sin(2*math.pi*i/count-math.pi/2)) for i in range(count)]
        else:
            step = 2*math.asin(spacing/(2*radius))
            positions = [(radius*math.cos((i-(count-1)/2)*step),
                          radius*math.sin((i-(count-1)/2)*step), 0) for i in range(count)]
        return [dict(satellite_id=s, ecef_position_km=p) for s, p in zip(ids, positions)]
    if config['topology']['source'] == 'orbit':
        from astermind.builder.orbit import tle_elements, propagate_sgp4_teme, transform_sgp4_teme_to_itrs
        from datetime import datetime, timezone
        stamp=datetime.fromisoformat(c['timestamp'].replace('Z','+00:00'))
        if stamp.tzinfo is None:raise ValueError('TLE 时间需要明确 UTC 时区，例如 2026-10-21T00:00:00Z')
        epoch=stamp.astimezone(timezone.utc).isoformat(timespec='microseconds').replace('+00:00','Z')
        bundle = tle_elements(owned(c['tle_file']).read_text(), name='cell-upload')
        result = propagate_sgp4_teme(bundle, simulation_epoch_utc=epoch, sample_times_s=(0.,))
        fixed = transform_sgp4_teme_to_itrs(result)
        rows = {s.satellite_id: s for s in fixed.samples}
        return [dict(satellite_id=s, ecef_position_km=rows[s].position_itrs_km) for s in ids]
    raise ValueError('V1 Cell 接口支持编队及 TLE；文件拓扑请使用场景编辑器的 V1 资源 JSON')


def topology(config):
    activate()
    schema_validate(config, network_only=True)
    h, t = config['hardware'], config['topology']
    ids, n = h['satellite_ids'], h['accelerators_per_satellite']
    if len(set(ids)) != len(ids) or len(ids) != h['satellite_count']:
        raise ValueError('卫星数量与唯一 ID 数量不一致')
    if not 0 <= t['intra_satellite']['gateway_local_index'] < n:
        raise ValueError('网关卡索引超出星内卡数量')
    states = geometry(config)
    positions = {s['satellite_id']: s['ecef_position_km'] for s in states}
    devices, links, optical = [], [], []
    def add_link(source, target, bw, latency, **extra):
        links.append(dict(id=f'link-{len(links)}', source=source, target=target,
                          bandwidth_gbps=bw, latency_ns=round(latency), **extra))
    for sat in ids:
        devices.extend(dict(id=f'{sat}.gpu{i}', satellite=sat, kind='gpu') for i in range(n))
        devices.append(dict(id=f'{sat}.exit', satellite=sat, kind='terminal',
                            external_access=True, max_external_links=max(1, len(ids))))
        shape = t['intra_satellite']['type']
        bw, latency = h['intra_satellite_bandwidth_gbps'], h['intra_satellite_latency_ns']
        if shape == 'switch':
            devices.append(dict(id=f'{sat}.switch', satellite=sat, kind='switch', capacity_gbps=bw))
            for i in range(n):
                add_link(f'{sat}.gpu{i}', f'{sat}.switch', bw, latency)
            gateway = f'{sat}.switch'
        else:
            if shape not in ('fully_connected', 'ring'):
                raise ValueError(f'V1 适配层尚未支持星内拓扑 {shape}')
            pairs = [(i,j) for i in range(n) for j in range(i+1,n)
                     if shape == 'fully_connected' or j == i+1 or (shape == 'ring' and i == 0 and j == n-1)]
            for i,j in pairs:
                add_link(f'{sat}.gpu{i}', f'{sat}.gpu{j}', bw, latency)
            gateway = f'{sat}.gpu{t["intra_satellite"]["gateway_local_index"]}'
        add_link(gateway, f'{sat}.exit', bw, latency)
    from astermind.builder.orbit import satellite_line_of_sight
    for edge in t['inter_satellite']['links']:
        a,b = edge['source'], edge['target']
        if a not in positions or b not in positions or a == b:
            raise ValueError('星间链路端点必须是两个已有卫星')
        distance = math.dist(positions[a], positions[b])
        visible = satellite_line_of_sight(positions[a], positions[b])
        feasible = distance <= t['inter_satellite']['max_range_km'] and visible
        reason = '' if feasible else '超出距离或地球遮挡'
        bandwidth = edge['bandwidth_gbps']
        if edge['mode'] == 'islsim':
            from astermind.engine.setup.islsim import resolve_config, calculate
            params, source = resolve_config(config['fso'])
            cap = params['rx']['terminal_cap_gbps']
            profile = SimpleNamespace(_islsim_binding=None, islsim_source_sha256=source,
                islsim_config=params, bandwidth_bps=math.inf if cap is None else cap * 1e9,
                parameter_sources={}, assumptions=[])
            budget = calculate(profile, distance)
            bandwidth = budget['budget_bandwidth_bps']/1e9
            feasible = feasible and budget['link_feasible']
            reason = reason or budget['limiting_reason'] or ''
        latency = distance/299792.458*1e9
        optical.append(dict(source_satellite=a,target_satellite=b,distance_km=distance,
                            bandwidth_gbps=bandwidth,latency_ns=latency,feasible=feasible,limiting_reason=reason))
        if feasible:
            add_link(f'{a}.exit',f'{b}.exit',bandwidth,latency)
        elif not t['inter_satellite']['drop_infeasible_links']:
            raise ValueError(f'{a}–{b}: {reason}')
    physical = dict(cells=[dict(id='cell',satellites=ids)],
                    satellites=[dict(id=s) for s in ids],devices=devices,links=links)
    owner = {d['id']: d['satellite'] for d in devices}
    preview = dict(topology=dict(nodes=[dict(d,satellite_id=d['satellite']) for d in devices],
        links=[dict(e,enabled=True,level='intra_satellite' if owner[e['source']]==owner[e['target']] else 'inter_satellite') for e in links],
        satellites=states), geometry=dict(satellites=states),isl_links=optical,backend=identity())
    return physical, preview


def request_stream(config):
    activate()
    from src.serving.request_gen.synthetic_requests import SyntheticRequestConfig, generate_synthetic_requests
    from src.serving.request_gen.request_stream import RequestSpec, RequestStream
    if config.get('closed_loop_client_count') is not None:
        raise ValueError('V1 输入契约没有旧前端的 closed-loop admission，不能等价转换')
    if config['source'] == 'synthetic':
        stream = generate_synthetic_requests(SyntheticRequestConfig(**config['synthetic']))
    elif config['source'] == 'canonical':
        from src.serving.request_gen.canonical_requests import load_canonical_requests
        stream=load_canonical_requests(owned(config['file']))
    elif config['source']=='production':
        from src.serving.request_gen.production_trace import ProductionTraceReaderConfig,SuccessfulRequestFilter,load_production_trace
        options=deepcopy(config['production'])
        options['successful_request_filter']=SuccessfulRequestFilter(**options['successful_request_filter'])
        stream=load_production_trace(owned(config['file']),ProductionTraceReaderConfig(**options)).requests
    else:
        raise ValueError('生产数据适配请先转换成 canonical 请求 JSON；本接口不会猜测数据字段')
    scale = config.get('arrival_rate_scale', 1)
    if scale <= 0:
        raise ValueError('arrival_rate_scale 必须为正')
    return RequestStream(RequestSpec(**{**asdict(r),'arrival_time_s':r.arrival_time_s/scale}) for r in stream.requests)


def convert(value):
    activate()
    schema_validate(value)
    physical, preview = topology(value)
    deployment, hw = value['deployment'], value['hardware']
    if deployment['mode'] not in ('colocated','pd_disagg'):
        raise ValueError('V1 ServingCell 仅支持 colocated / pd_disagg；AF 与 PD+AF 需要后端扩展')
    pools = deployment['worker_pools']
    expected = ['colocated'] if deployment['mode']=='colocated' else ['prefill','decode']
    if [p['role'] for p in pools] != expected:
        raise ValueError('V1 需要一个 colocated pool 或依次 prefill、decode 两个 pool')
    if hw['accelerator_overrides']:
        raise ValueError('V1 ServingCell 的 hardware 是 Cell 级设置，不能静默忽略逐卡硬件覆盖')
    if value['runtime']['timing_mode'] != 'device_only' or value['kv_offload']['enabled']:
        raise ValueError('旧 Host 时序和 KV offload replay 没有等价 V1 配置入口')
    if value['flowsim']['acknowledgement']['enabled']:
        raise ValueError('V1 Runtime Comm 没有 FlowSim ACK 参数，请关闭 ACK 后运行')
    if value['flowsim'].get('routing_seed',0) != 0:
        raise ValueError('V1 路由不使用旧 FlowSim routing_seed')
    if value['runtime_inputs']['graph_execution']['configured_mode'] != 'NONE':
        raise ValueError('V1 暂不接受旧 CUDA Graph 执行模式')
    if deployment['router']['dispatch_service_time_s'] != 0:
        raise ValueError('V1 不接受旧 Router CPU 服务时间')
    parallel = pools[0]['parallelism']
    if any(p['parallelism'] != parallel for p in pools):
        raise ValueError('V1 PD 要求 prefill/decode 使用相同并行形状')
    if parallel['context_parallel_size'] != 1 or parallel['sequence_parallel_size'] != 1 or parallel['data_parallel_size'] != 1 or parallel['enable_eplb']:
        raise ValueError('此 V1 适配器暂不支持 CP / SP / DP / EPLB；不会忽略这些参数')
    settings = [value['scheduler'][p['name']] for p in pools]
    if any(s['framework'] != 'vllm' for s in settings):
        raise ValueError('V1 使用官方 vLLM Scheduler 适配器，不支持 SGLang')
    if any(s != settings[0] for s in settings):
        raise ValueError('V1 PD 共享 Scheduler 配置，两个 pool 的参数必须一致')
    s = settings[0]
    v = s['vllm']
    fixed={'gpu_memory_utilization':.9,'trust_remote_code':False,'dummy_token_id':42,
           'mtp_fixed_accepted_tokens':0,'mtp_acceptance_policy':'disabled'}
    for key,default in fixed.items():
        if v.get(key)!=default:
            raise ValueError(f'V1 不提供旧调度字段 {key} 的等价设置；请保留默认值 {default}')
    if value['runtime_inputs']['attention_kv_layout']!='paged' or value['runtime_inputs']['attention_kv_distribution']!='uniform':
        raise ValueError('V1 当前原生 Cell 使用 paged / uniform KV 布局')
    if deployment['routes']:
        raise ValueError('V1 PD 使用原生 prefill→decode 契约；不能忽略自定义跨 pool routes')
    if v.get('mtp_num_speculative_tokens',0) or v.get('async_scheduling') or s['scheduler_pipeline_depth'] != 1:
        raise ValueError('V1 ServingCell 没有 MTP / async / scheduler pipeline depth 配置入口')
    if v['dtype'] != 'bfloat16' or not v['enable_chunked_prefill']:
        raise ValueError('V1 原生 Cell 使用 BF16 和 chunked prefill')
    if value['launch_policy'] != 'analytical_launch':
        raise ValueError('V1 当前适配器只接受 analytical_launch')
    tp,pp = parallel['tensor_parallel_size'],parallel['pipeline_parallel_size']
    placements=[]; algorithms={}
    for pool_index,pool in enumerate(pools):
        if pool.get('kv_cache') is not None:
            raise ValueError('V1 当前适配器不接受旧 pool KV 分层覆盖；请使用 Scheduler KV 参数')
        if pool.get('pipeline_p2p'):
            raise ValueError('V1 当前适配器使用原生 PP 邻接通信；暂不接受自定义 pipeline_p2p 覆盖')
        for row in deployment['placements']:
            if row['worker_pool'] != pool['name']:
                continue
            c=row['parallel_coordinate']
            rank=pool_index*tp*pp+c['pp']*tp+c['tp']
            placements.append(dict(rank=rank,device=f"{row['satellite_id']}.gpu{row['local_index']}",
                                   pool='main' if len(pools)==1 else pool['role']))
        for collective in pool['collectives']:
            if collective['backend'] != 'runtime_comm':
                raise ValueError('V1 原生通信后端为 runtime_comm，不能运行旧 FlowSim / ns-3')
            op=collective['operation']; algorithm={'mesh':'fullmesh'}.get(collective['algorithm'],collective['algorithm'])
            if op in algorithms and algorithms[op]!=algorithm:
                raise ValueError('V1 通信算法按 Cell 配置，不支持同一 Cell 的 pool 使用冲突算法')
            algorithms[op]=algorithm
    by_rank={p['rank']:p['device'] for p in placements}
    stream=request_stream(value['requests'])
    if any(r.prompt_token_ids for r in stream.requests):
        raise ValueError('旧请求的 token identity 不能转换成 V1 的会话契约')
    routing=value['runtime_inputs']['expert_routing']
    seed=routing.get('root_seed') if parallel['enable_expert_parallel'] else None
    cell=dict(id='cell',model=f"configs/model/{value['model']}.json",hardware_id=hw['default_hardware_id'],
        hardware=f"configs/hardware/{hw['default_hardware_id']}.yaml",deployment=deployment['mode'],
        parallel=dict(tp=tp,pp=pp,ep=tp if parallel['enable_expert_parallel'] else 1),placements=placements,
        ingress_device=by_rank.get(0,''),output_device=by_rank.get((len(pools)-1)*tp*pp+tp*(pp-1),''),
        external_terminals=[f'{sat}.exit' for sat in hw['satellite_ids']],egress_mode='distributed',max_model_len=v['max_model_len'] or 128,
        max_batched_tokens=s['max_batched_tokens'],max_requests=s['max_requests'],num_kv_blocks=s['num_kv_blocks'],
        kv_block_size_tokens=s['kv_block_size_tokens'],prefix_caching=v['enable_prefix_caching'],
        native_expert_parallel=parallel['enable_expert_parallel'],expert_routing_seed=seed)
    # V1 requires a ground owner. This declared zero-payload boundary models the
    # old device-only request source, not a measured satellite-ground transfer.
    physical['cells'].append(dict(id='client-resources',satellites=['client']))
    physical['satellites'].append(dict(id='client'))
    physical['devices'].append(dict(id='client.endpoint',satellite='client',kind='terminal',external_access=True))
    entry_sat=by_rank.get(0,'').rsplit('.gpu',1)[0]
    physical['links'].append(dict(id='client-boundary',source='client.endpoint',target=f'{entry_sat}.exit',
                                  bandwidth_gbps=hw['intra_satellite_bandwidth_gbps'],latency_ns=0))
    requests=[dict(id=r.request_id,arrival_ns=round(r.arrival_time_s*1e9),user_device='client.endpoint',
        cell='cell',input_tokens=r.input_tokens,output_tokens=r.output_tokens,prefix_hit_tokens=r.prefix_hit_tokens,
        input_bytes=0,output_bytes_per_token=0) for r in stream.requests]
    from astermind.builder.config import SimulationConfig
    config=SimulationConfig.model_validate(dict(physical=physical,ground_owners=['client'],cells=[cell],requests=requests,
        horizon_ns=max(60_000_000_000,max(r['arrival_ns'] for r in requests)+60_000_000_000),
        communication={'intra_mode':'IntraSatelliteRoofline','cell_policies':{'cell':{
            'algorithms':{scope:algorithms for scope in ('single_su','intra_cross_su','inter_satellite')}}}}))
    config.topology()
    return config.model_dump(mode='json'),preview
