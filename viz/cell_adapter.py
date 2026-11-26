"""Translate the preserved Cell editor into the V1 public configuration contract.

No historical engine is imported. Unsupported semantics fail before submission.
The submitted and effective configurations are both retained for inspection.
"""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
from types import SimpleNamespace

from paths import VIZ, RUNTIME, owned
from runtime_environment import activate, identity


def capabilities():
    from compute_options import compute_catalog
    from runtime_environment import enable_device_native
    import yaml
    upstream=enable_device_native()
    catalog = compute_catalog(upstream)
    value = json.loads((VIZ / 'cell/presentation.json').read_text())
    value.update(models=[Path(m['id']).stem for m in catalog['models'] if m['available']],
                 hardware=[h['id'] for h in catalog['hardware']], backend=identity())
    value['models_by_hardware'] = {
        h['id']: {Path(m['id']).stem: m['choices'][h['id']] for m in catalog['models'] if m['choices'].get(h['id'])}
        for h in catalog['hardware']
    }
    value['model_details'] = {Path(m['id']).stem: m for m in catalog['models']}
    for key in ('compute_modes', 'detail_prediction_methods', 'memory_policies', 'collective_algorithms'):
        value[key] = catalog[key]
    value['fast_roofline_defaults'] = {'tensor_percent':70, 'hbm_percent':96}
    value['fast_gpu_limits'] = {h['id']: h['fast_gpu_limit_per_satellite'] for h in catalog['hardware']}
    value['su_profiles'] = {h['id']: h for h in catalog['hardware']}
    from astermind.engine.compute.fast_mode.hardware import Hardware
    from src.kernel.hardware_spec import HWSpec
    for row in catalog['hardware']:
        profile = Hardware.from_hardware_spec(HWSpec(upstream / row['file']))
        row['roofline_peaks'] = dict(bf16_tflops=profile.bf16_flops/1e12,
                                    hbm_tb_s=profile.nominal_hbm_bytes(row['id'])/1e12)
    value['hardware_details']={h['id']:yaml.safe_load((upstream/h['file']).read_text()) for h in catalog['hardware']}
    config = value['default_configuration']
    config.update(compute_mode='detail_mode', detail_prediction_method='auto',
                  memory_policy_file=None, fast_kv_pool_blocks=None, fast_glm_dsa_score=None, expert_routing_seed=0,
                  ep_strategy='auto', collective_policies=[])
    config['hardware']['su_group_sizes'] = []
    config['topology']['intra_satellite']['type'] = 'profile_su'
    scheduler = config['scheduler']['main']
    scheduler.update(framework='vllm', sglang=None, vllm=dict(
        model=config['model'], dtype='bfloat16', max_model_len=None, gpu_memory_utilization=.9,
        trust_remote_code=False, enable_chunked_prefill=True, enable_prefix_caching=False,
        async_scheduling=False, dummy_token_id=42, mtp_num_speculative_tokens=0,
        mtp_acceptance_policy='disabled', mtp_fixed_accepted_tokens=0, mtp_acceptance_source_id='disabled',
        mtp_acceptance_replay=None, mtp_acceptance_assumption=None, request_stream_sha256=None,
        decode_admission_contract_sha256=None, closed_loop_admission_contract_sha256=None))
    scheduler.update(max_batched_tokens=None, num_kv_blocks=None, kv_block_size_tokens=None)
    scheduler['vllm']['long_prefill_token_threshold'] = 0
    config['flowsim']['acknowledgement']['enabled'] = False
    for pool in config['deployment']['worker_pools']:
        for collective in pool['collectives']:
            collective['backend'] = 'runtime_comm'
    definitions = value['schema']['$defs']
    value['schema']['properties'].update(
        fast_roofline={'type':'object','additionalProperties':False,'required':['tensor_percent','hbm_percent'],
                       'properties':{key:{'type':'number','exclusiveMinimum':0,'maximum':100}
                                     for key in ('tensor_percent','hbm_percent')}},
        fast_link_contention={'type':'boolean'},
        compute_mode={'enum': [mode['id'] for mode in value['compute_modes']]},
        detail_prediction_method={'anyOf': [
            {'type': 'string', 'enum': ['auto', *value['detail_prediction_methods']]}, {'type': 'null'},
        ]},
        memory_policy_file={'anyOf': [{'type': 'string'}, {'type': 'null'}]},
        fast_kv_pool_blocks={'anyOf': [{'type': 'integer', 'minimum': 2}, {'type': 'null'}]},
        fast_glm_dsa_score={'enum': ['', 'bf16_timing_proxy', None]},
        expert_routing_seed={'type': 'integer', 'minimum': 0},
    )
    definitions['VLLMOptions']['properties']['long_prefill_token_threshold'] = {'type': 'integer', 'minimum': 0}
    for key in ('max_batched_tokens', 'num_kv_blocks', 'kv_block_size_tokens'):
        definitions['SchedulerConfig']['properties'][key] = {
            'anyOf': [{'type': 'integer', 'minimum': 1}, {'type': 'null'}], 'default': None,
        }
        if key in definitions['SchedulerConfig']['required']:
            definitions['SchedulerConfig']['required'].remove(key)
    definitions['FormationConfig']['properties']['layout']['enum'].append('fully_connected')
    definitions['IntraSatelliteTopologyConfig']['properties']['type']['enum'].append('profile_su')
    definitions['HardwareConfig']['properties']['su_group_sizes'] = {
        'type': 'array', 'items': {'type': 'integer', 'minimum': 1}, 'default': [],
    }
    value['schema']['properties']['ep_strategy'] = {'enum': ['auto', 'allgather', 'direct']}
    value['schema']['properties']['collective_policies'] = {
        'type': 'array',
        'items': {'type': 'object', 'required': ['scope', 'operation', 'algorithm'], 'additionalProperties': False,
                  'properties': {key: {'type': 'string'} for key in ('scope', 'operation', 'algorithm')}},
    }
    definitions['HardwareConfig']['properties']['default_hardware_id']['enum'] = value['hardware']
    definitions['SchedulerConfig']['properties']['framework']['enum'] = ['vllm']
    definitions['CollectiveConfig']['properties']['backend']['enum'] = ['runtime_comm']
    value['compatibility'] = {'unsupported': ['sglang', 'af_disagg', 'pd_af_disagg',
        'FlowSim ACK', 'SimGrid', 'MTP', 'KV offload replay', 'closed-loop admission'],
        'simulation_horizon_s': None, 'client_boundary': 'explicit zero-payload ground endpoint; no access time model'}
    return value


def schema_validate(value, network_only=False, schema=None):
    from jsonschema import Draft202012Validator
    schema = schema or capabilities()['schema']
    if network_only:
        schema = dict(schema)
        schema['required'] = ['hardware', 'topology', 'constellation', 'fso']
    errors = sorted(Draft202012Validator(schema).iter_errors(value), key=lambda e: str(e.path))
    if errors:
        raise ValueError('; '.join('.'.join(map(str, e.path)) + ': ' + e.message for e in errors[:5]))


def geometry(config):
    ids = config['hardware']['satellite_ids']
    c = config['constellation']
    if config['topology']['source'] == 'formation':
        f = c['formation']
        from astermind.builder.physical.formation_geometry import static_formation_offsets
        if not 2 <= len(ids) <= 35:
            raise ValueError('静态编队需要 2–35 颗卫星')
        offsets = static_formation_offsets(len(ids), f['layout'], f['spacing_km'], f['altitude_km'])
        radius = 6371.0 + f['altitude_km']
        positions = [(radius + radial, along, across) for radial, along, across in offsets]
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
    hierarchies = {}
    shape = t['intra_satellite']['type']
    if shape == 'profile_su':
        from itertools import accumulate
        from compute_options import su_groups

        profile = capabilities()['su_profiles'][h['default_hardware_id']]
        sizes = su_groups(h.get('su_group_sizes', []), n, profile['fast_gpu_limit_per_satellite'])
        stops = list(accumulate(sizes, initial=0))
        su_bw = profile['gpu_link_bandwidth_gbps'] or h['intra_satellite_bandwidth_gbps']
        su_latency = profile['gpu_link_latency_ns']
    elif h.get('su_group_sizes'):
        raise ValueError('自定义 SU 分组时，请选择「SU 分域互联（按卡型）」')
    def add_link(source, target, bw, latency, **extra):
        links.append(dict(id=f'link-{len(links)}', source=source, target=target,
                          bandwidth_gbps=bw, latency_ns=round(latency), **extra))
    for sat in ids:
        devices.extend(dict(id=f'{sat}.gpu{i}', satellite=sat, kind='gpu') for i in range(n))
        devices.append(dict(id=f'{sat}.exit', satellite=sat, kind='terminal',
                            external_access=True, max_external_links=max(1, len(ids))))
        bw, latency = h['intra_satellite_bandwidth_gbps'], h['intra_satellite_latency_ns']
        if shape == 'profile_su':
            gpu_groups = [[f'{sat}.gpu{i}' for i in range(start, stop)] for start, stop in zip(stops, stops[1:])]
            switches = [f'{sat}.su{i}' for i in range(len(gpu_groups))]
            hierarchies[sat] = dict(su_groups=gpu_groups, su_switches=switches)
            for group, switch in zip(gpu_groups, switches, strict=True):
                devices.append(dict(id=switch, satellite=sat, kind='switch', capacity_gbps=su_bw * len(group)))
                for gpu in group:
                    add_link(gpu, switch, su_bw, su_latency)
                add_link(switch, f'{sat}.exit', bw if len(gpu_groups) > 1 else su_bw, latency if len(gpu_groups) > 1 else 0)
            continue
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
    inter_links = t['inter_satellite']['links']
    if config['topology']['source'] == 'formation' and config['constellation']['formation']['layout'] == 'fully_connected':
        expected = {frozenset((a, b)) for i, a in enumerate(ids) for b in ids[i+1:]}
        if {frozenset((edge['source'], edge['target'])) for edge in inter_links} != expected or len(inter_links) != len(expected):
            raise ValueError('全连接编队需要每对卫星之间各一条链路')
    for edge in inter_links:
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
    if hierarchies:
        physical['intra_hierarchies'] = hierarchies
    owner = {d['id']: d['satellite'] for d in devices}
    preview = dict(topology=dict(nodes=[dict(d,satellite_id=d['satellite']) for d in devices],
        links=[dict(e,enabled=True,level='intra_satellite' if owner[e['source']]==owner[e['target']] else 'inter_satellite') for e in links],
        satellites=states), geometry=dict(satellites=states),isl_links=optical,backend=identity())
    return physical, preview


def request_stream(config):
    activate()
    from src.serving.request_gen.synthetic_requests import SyntheticRequestConfig, generate_synthetic_requests
    from src.serving.request_gen.request_stream import RequestStream
    if config.get('closed_loop_client_count') is not None:
        raise ValueError('V1 输入契约没有旧前端的 closed-loop admission，不能等价转换')
    if config['source'] == 'synthetic':
        synthetic = config['synthetic']
        if synthetic.get('arrival_distribution') == 'simultaneous':
            # Reuse the pinned generator for request IDs, lengths and prefix hits.
            source = SyntheticRequestConfig(**{**synthetic, 'arrival_distribution': 'constant', 'request_rate': 1.0})
            return RequestStream(replace(request, arrival_time_s=0.0)
                                 for request in generate_synthetic_requests(source))
        stream = generate_synthetic_requests(SyntheticRequestConfig(**synthetic))
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
    return RequestStream(replace(request, arrival_time_s=request.arrival_time_s / scale)
                         for request in stream.requests)


def compile_serving_cell(value, physical, longest_request, *, cell_id='cell', scene=False, catalog=None):
    """Compile the editor's pools, ranks, scheduler and collectives onto an existing physical graph."""
    deployment, hw = value['deployment'], value['hardware']
    compute_mode = value.get('compute_mode', 'detail_mode')
    fast = compute_mode == 'fast_mode'
    if fast and value.get('detail_prediction_method') not in (None, 'auto'):
        raise ValueError('详细预测方法仅在 Detail Mode 可用')
    if not fast and (value.get('memory_policy_file') or value.get('fast_glm_dsa_score')):
        raise ValueError('显存策略和 GLM 计时设置仅在 Fast Mode 可用')
    if deployment['mode'] not in ('colocated','pd_disagg'):
        raise ValueError('V1 ServingCell 仅支持 colocated / pd_disagg；AF 与 PD+AF 需要后端扩展')
    if fast and deployment['mode'] != 'colocated':
        raise ValueError('Fast Mode 需要同池部署（colocated）')
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
    if parallel['context_parallel_size'] != 1 or parallel['sequence_parallel_size'] != 1 or parallel['enable_eplb']:
        raise ValueError('此 V1 适配器要求 CP / SP 为 1，并关闭 EPLB')
    tp = parallel['tensor_parallel_size']
    pp = parallel['pipeline_parallel_size']
    dp = parallel['data_parallel_size']
    ep = parallel['enable_expert_parallel']
    if dp > 1 and not ep:
        raise ValueError('DP 大于 1 需要开启专家并行（WideEP）')
    catalog = catalog or capabilities()
    model = catalog['model_details'].get(value['model'])
    hardware = hw['default_hardware_id']
    if model is None:
        raise ValueError('请选择后端目录中的模型')
    if ep and not model['experts']:
        raise ValueError('专家并行（EP）需要 MoE 模型')

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
    if v['dtype'] not in ('bfloat16', model['dtype']):
        raise ValueError('调度精度与所选模型档案不一致')
    if v['model'] != value['model']:
        raise ValueError('调度器型号与所选模型不一致')
    chunked = v.get('enable_chunked_prefill', True)
    long_prefill = v.get('long_prefill_token_threshold', 0)
    if long_prefill and not chunked:
        raise ValueError('Prefill 分块上限需要开启分块 Prefill')
    if deployment['mode'] == 'pd_disagg' and (not chunked or long_prefill):
        raise ValueError('PD 使用 vLLM 0.11，需要开启分块 Prefill 并将Prefill 分块上限设为 0')
    if value['launch_policy'] != 'analytical_launch':
        raise ValueError('V1 当前适配器只接受 analytical_launch')
    workers = tp * pp * dp
    gpu_count = len(hw['satellite_ids']) * hw['accelerators_per_satellite']
    rows = deployment['placements']
    if workers * len(pools) > gpu_count or len(rows) != workers * len(pools):
        raise ValueError(f'部署需要完整映射 {workers * len(pools)} 个 Rank，目前资源有 {gpu_count} 张 GPU')
    placements=[]; algorithms={}
    for pool_index,pool in enumerate(pools):
        if pool.get('kv_cache') is not None:
            raise ValueError('V1 当前适配器不接受旧 pool KV 分层覆盖；请使用 Scheduler KV 参数')
        if pool.get('pipeline_p2p'):
            raise ValueError('V1 当前适配器使用原生 PP 邻接通信；暂不接受自定义 pipeline_p2p 覆盖')
        for row in rows:
            if row['worker_pool'] != pool['name']:
                continue
            c=row['parallel_coordinate']
            if (c['cp'] != 0 or not (0 <= c['dp'] < dp and 0 <= c['pp'] < pp and 0 <= c['tp'] < tp)
                    or row['satellite_id'] not in hw['satellite_ids']
                    or type(row['local_index']) is not int or not 0 <= row['local_index'] < hw['accelerators_per_satellite']):
                raise ValueError('Rank 坐标或卡映射超出资源范围')
            rank = pool_index * workers + (c['dp'] * pp + c['pp']) * tp + c['tp']
            separator = '/' if scene else '.'
            placements.append(dict(rank=rank,device=f"{row['satellite_id']}{separator}gpu{row['local_index']}",
                                   pool='main' if len(pools)==1 else pool['role']))
        for collective in pool['collectives']:
            if collective['backend'] != 'runtime_comm':
                raise ValueError('V1 原生通信后端为 runtime_comm，不能运行旧 FlowSim / ns-3')
            op=collective['operation']; algorithm={'mesh':'fullmesh'}.get(collective['algorithm'],collective['algorithm'])
            if op in algorithms and algorithms[op]!=algorithm:
                raise ValueError('V1 通信算法按 Cell 配置，不支持同一 Cell 的 pool 使用冲突算法')
            algorithms[op]=algorithm
    by_rank={p['rank']:p['device'] for p in placements}
    if len(by_rank) != len(rows) or len(set(by_rank.values())) != len(rows):
        raise ValueError('每个 Rank 必须映射到唯一的 GPU')
    from compute_options import (collective_policy, detail_prediction_method, prepare_deployment,
                                 resolve_scheduling, scheduler_adapter, validate_fast_su_placement)
    from runtime_environment import enable_device_native

    policy_rows = collective_policy(value.get('collective_policies', []))
    scoped_algorithms = {scope: {**algorithms, **policy_rows.get(scope, {})}
                         for scope in ('single_su', 'intra_cross_su', 'inter_satellite')}
    routing = value['runtime_inputs']['expert_routing']
    seed = value.get('expert_routing_seed', routing.get('root_seed')) if model['experts'] else None
    adapter = scheduler_adapter(compute_mode, value['model'], dp, pd=deployment['mode'] == 'pd_disagg')
    upstream = enable_device_native()
    method = value.get('detail_prediction_method')
    if not fast and method in (None, 'auto'):
        method = detail_prediction_method(model['id'], hardware, upstream)
    options = dict(model=model['id'], hardware=f'configs/hardware/{hardware}.yaml', hardware_id=hardware,
                   gpus_per_satellite=hw['accelerators_per_satellite'], tp=tp, pp=pp, dp=dp,
                   expert_parallel=ep, wide_ep_enabled=dp > 1, expert_routing_seed=seed,
                   compute_mode=compute_mode, detail_prediction_method=None if fast else method,
                   scheduler_adapter=adapter, max_requests=s['max_requests'], max_model_len=v.get('max_model_len'),
                   max_batched_tokens=s.get('max_batched_tokens'), num_kv_blocks=s.get('num_kv_blocks'),
                   kv_block_size_tokens=s.get('kv_block_size_tokens'), prefix_caching=v['enable_prefix_caching'],
                   enable_chunked_prefill=chunked, long_prefill_token_threshold=long_prefill,
                   memory_policy_file=value.get('memory_policy_file') or None,
                   fast_kv_pool_blocks=value.get('fast_kv_pool_blocks') if fast else None,
                   fast_glm_dsa_score=value.get('fast_glm_dsa_score') or None,
                   fast_roofline=value.get('fast_roofline') if fast else None,
                   fast_link_contention=fast and value.get('fast_link_contention',False))
    options.update(resolve_scheduling(options, model, longest_request, upstream))
    if adapter == 'vllm_0_11' and (not chunked or long_prefill):
        raise ValueError('vLLM 0.11 需要开启分块 Prefill 并将Prefill 分块上限设为 0')
    hierarchy = physical.get('intra_hierarchies', {})
    largest_su = max((len(group) for row in hierarchy.values() for group in row['su_groups']), default=hw['accelerators_per_satellite'])
    if deployment['mode'] == 'colocated':
        checked, _ = prepare_deployment(options, upstream, largest_su=largest_su, native_detail=True)
        options['memory_policy_file'] = checked.memory_policy_file
    elif (value['model'], hardware, tp, pp, dp, ep) != ('kimi-k2.6', 'nvidia_h200', 8, 1, 1, True):
        raise ValueError('PD 支持 H200 + Kimi K2.6，每池 TP8 / EP8、PP1 / DP1')
    if fast and not hierarchy:
        physical['intra_hierarchies'] = {
            sat: dict(su_groups=[[f'{sat}.gpu{i}' for i in range(hw['accelerators_per_satellite'])]],
                      su_switches=[f'{sat}.switch'] if value['topology']['intra_satellite']['type'] == 'switch' else [])
            for sat in hw['satellite_ids']
        }
    if fast:
        validate_fast_su_placement(hardware, by_rank.values(), physical.get('intra_hierarchies', {}))
    communication = {
        'intra_mode': 'surrogate' if dp > 1 or physical.get('intra_hierarchies') else 'IntraSatelliteRoofline',
        'default_hardware_id': hardware,
        'cell_policies': {cell_id: {'algorithms': scoped_algorithms, 'hardware_id':hardware}},
    }
    terminal = '/terminal' if scene else '.exit'
    cell = dict(id=cell_id, model=model['id'], hardware_id=hardware, hardware=options['hardware'],
                deployment=deployment['mode'], parallel=dict(tp=tp, pp=pp, dp=dp, ep=tp*dp if ep else 1),
                placements=placements, ingress_device=by_rank[0], output_device=by_rank[(len(pools)-1)*workers+tp*(pp-1)],
                external_terminals=[sat + terminal for sat in hw['satellite_ids']], egress_mode='distributed',
                native_expert_parallel=ep, **{key: options[key] for key in (
                    'max_model_len', 'max_batched_tokens', 'max_requests', 'num_kv_blocks', 'kv_block_size_tokens',
                    'prefix_caching', 'enable_chunked_prefill', 'long_prefill_token_threshold', 'scheduler_adapter',
                    'expert_routing_seed', 'wide_ep_enabled', 'memory_policy_file', 'fast_glm_dsa_score',
                    'fast_roofline', 'fast_kv_pool_blocks', 'fast_link_contention',
                )})
    return cell, communication, None if fast else method


def editor_requests(settings):
    stream = request_stream(settings)
    if any(r.prompt_token_ids for r in stream.requests):
        raise ValueError('旧请求的 token identity 不能转换成 V1 的会话契约')
    return stream.requests


def convert(value):
    activate()
    schema_validate(value)
    physical, preview = topology(value)
    stream = editor_requests(value['requests'])
    longest = max((r.input_tokens + r.output_tokens for r in stream), default=0)
    cell, communication, method = compile_serving_cell(value, physical, longest)
    # V1 requires a ground owner. This declared zero-payload boundary models the
    # old device-only request source, not a measured satellite-ground transfer.
    physical['cells'].append(dict(id='client-resources',satellites=['client']))
    physical['satellites'].append(dict(id='client'))
    physical['devices'].append(dict(id='client.endpoint',satellite='client',kind='terminal',external_access=True))
    entry_sat=cell['ingress_device'].rsplit('.gpu',1)[0]
    physical['links'].append(dict(id='client-boundary',source='client.endpoint',target=f'{entry_sat}.exit',
                                  bandwidth_gbps=value['hardware']['intra_satellite_bandwidth_gbps'],latency_ns=0))
    requests=[dict(id=r.request_id,arrival_ns=round(r.arrival_time_s*1e9),user_device='client.endpoint',
        cell='cell',input_tokens=r.input_tokens,output_tokens=r.output_tokens,prefix_hit_tokens=r.prefix_hit_tokens,
        input_bytes=0,output_bytes_per_token=0) for r in stream]
    from astermind.builder.config import SimulationConfig
    config=SimulationConfig.model_validate(dict(physical=physical,ground_owners=['client'],cells=[cell],requests=requests,
        horizon_ns=None,
        compute_mode=value.get('compute_mode', 'detail_mode'), detail_prediction_method=method,
        communication=communication))
    config.topology()
    return config.model_dump(mode='json'),preview
