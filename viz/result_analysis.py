"""Read-only analysis of one saved experiment, never launches a simulation."""
from collections import Counter, defaultdict
from functools import lru_cache
import json
import gzip
import math
from pathlib import Path


def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf-8') as stream:return json.load(stream)
    from astermind.common.result_storage import restore_result
    return restore_result(json.loads(path.read_text(encoding='utf-8')))


def duration(end, start):
    if end is None or start is None:
        return None
    return max(0, (end-start)/1e9)


def statistics(values):
    values=sorted(v for v in values if isinstance(v,(int,float)) and math.isfinite(v))
    if not values:return dict(count=0,average=None,p50=None,p95=None,p99=None)
    def percentile(p):
        at=(len(values)-1)*p
        lo=math.floor(at);hi=math.ceil(at)
        return values[lo]+(values[hi]-values[lo])*(at-lo)
    return dict(count=len(values),average=sum(values)/len(values),p50=percentile(.5),p95=percentile(.95),p99=percentile(.99))


def summarize(rows):
    completed=[r for r in rows if r['state']=='completed']
    span=max((r.get('completed_s') or 0 for r in rows),default=0)
    return dict(total=len(rows),completed=len(completed),
        latency=statistics(r['latency_s'] for r in completed),
        ttft=statistics(r['ttft_s'] for r in rows),tpot=statistics(r['tpot_s'] for r in completed),
        observation_s=span,requests_per_s=len(completed)/span if span else None,
        tokens_per_s=sum(r['tokens'] for r in completed)/span if span and all(r.get('tokens') is not None for r in completed) else None)


def native_analysis(result, identity):
    rows=[]
    for key,r in result['requests'].items():
        tokens=r.get('received_tokens')
        gap=duration(r.get('last_token_received_ns'),r.get('first_token_received_ns'))
        rows.append(dict(id=key,cell=r.get('cell_id',r.get('cell')),state=r['status'],
            arrival_s=r.get('arrival_ns',0)/1e9,completed_s=None if r.get('completed_ns') is None else r['completed_ns']/1e9,
            latency_s=duration(r.get('completed_ns'),r.get('arrival_ns')) if r['status']=='completed' else None,
            ttft_s=duration(r.get('first_token_received_ns'),r.get('arrival_ns')),
            tpot_s=gap/(tokens-1) if gap is not None and tokens is not None and tokens>1 else None,tokens=tokens,
            input_s=duration(r.get('cell_ready_ns'),r.get('arrival_ns')),
            cell_s=duration(r.get('compute_complete_ns'),r.get('cell_ready_ns')),
            tail_s=duration(r.get('completed_ns'),r.get('compute_complete_ns')),failure=r.get('failure')))
    operators=defaultdict(lambda:dict(count=0,resource_s=0))
    for batch in result.get('batches',[]):
        for call in batch['calls']:
            kind=call.get('communication_backend',result['network'].get('backend',result['configuration'].get('backend','runtime_comm'))) if call['communication'] else call['kind']
            label='通信 / '+kind if call['communication'] else kind
            row=operators[label];row['count']+=1
            if call.get('end_time_ns') is not None and call.get('start_time_ns') is not None:
                row['resource_s']+=(call['end_time_ns']-call['start_time_ns'])/1e9
    flows=result['network'].get('flows',[])
    services=Counter(e.get('cell_transfer_id') for e in result['network'].get('event_log',[]) if e['kind']=='bandwidth_update')
    transport=defaultdict(lambda:dict(count=0,bytes=0,completed=0,service_intervals=0))
    for flow in flows:
        backend=flow.get('backend',result['network'].get('backend',result['configuration'].get('backend','runtime_comm')))
        row=transport[(flow.get('operation','unknown'),backend)];row['count']+=1
        row['bytes']+=flow.get('size_bytes',0);row['completed']+=flow['kind']=='completed'
        row['service_intervals']+=services[flow.get('id')]
        if backend=='runtime_comm':row['service_intervals']=0
    batches=result.get('batches',[])
    compute_count=communication_count=0
    ranks=set();methods=Counter();operations=Counter();batch_calls=[]
    for batch in batches:
        batch_compute=batch_comm=0
        for call in batch['calls']:
            if call['communication']:
                batch_comm+=1
                methods[call.get('communication_backend',result['network'].get('backend',result['configuration'].get('backend','runtime_comm')))]+=1
                operations[call.get('surrogate',{}).get('operation','逐流通信')]+=1
            else:
                batch_compute+=1
                ranks.add((batch['cell_id'],call['rank']))
        compute_count+=batch_compute;communication_count+=batch_comm
        batch_calls.append(dict(cell=batch['cell_id'],id=batch['step_id'],compute=batch_compute,communication=batch_comm))
    execution=dict(input_tokens=sum(r['input_tokens'] for r in result['configuration'].get('requests',[])),
        generated_tokens=sum(r.get('received_tokens') or 0 for r in result['requests'].values()),
        batches=len(batches),compute_calls=compute_count,communication_calls=communication_count,
        active_gpu_ranks=len(ranks),
        communication_methods=dict(methods),
        communication_operations=dict(operations),
        transfer_methods=dict(Counter(f.get('backend',result['network'].get('backend',result['configuration'].get('backend','runtime_comm'))) for f in flows)),
        service_intervals=sum(services.values()),
        batch_calls=batch_calls)
    return dict(id=identity,kind='native',requests=rows,summary=summarize(rows),
        execution=execution,
        communication_mode=result['configuration'].get('communication_mode','runtime_comm'),measurement=result.get('measurement',{}),
        timeline_mode='orbitserve',backend=result['network'].get('backend','runtime_comm'),
        operators=[dict(kind=k,**v) for k,v in operators.items()],
        transfers=[dict(operation=k[0],backend=k[1],**{**v,'service_intervals':None if k[1]=='runtime_comm' else v['service_intervals']}) for k,v in transport.items()],
        migrations=[{**v,'id':k} for k,v in result.get('migrations',{}).items()],audit=result.get('audit'),
        gpu_count=sum(d['kind']=='gpu' for d in result['configuration']['physical']['devices']),
        raw_url=f'/runs/{identity}/native-result')


@lru_cache(maxsize=8)
def _native(path, stamp, identity):
    return native_analysis(read(path),identity)


def analysis(root, identity):
    from studio import key
    identity=key(identity);folder=Path(root)/'local/results/network'/identity
    cached=folder/'analysis.json'
    if cached.is_file():
        return read(cached)
    path=folder/'native_result.json'
    if path.is_file():
        if path.stat().st_size>64*1024*1024:
            raise ValueError('大型实验需要已保存的分析摘要，请先完成结果整理')
        stat=path.stat();return _native(str(path),(stat.st_size,stat.st_mtime_ns),identity)
    index=folder/'runtime_playback/index.json'
    if not index.is_file():
        return dict(id=identity,kind='physical',requests=[],summary=summarize([]))
    manifest=read(index);rows=[]
    def artifact(name):
        target=(index.parent/name).resolve()
        if not target.is_relative_to(index.parent.resolve()):raise ValueError('无效的结果文件路径')
        return read(target)
    for page in manifest.get('requests',[]):
        for entry in artifact(page['artifact']):
            r=artifact(entry['artifact'])['record'];timing=r.get('timing',{});request=r['request']
            complete=r.get('outcome',{}).get('completed',False)
            arrival=request.get('arrival_time_s',0);latency=timing.get('e2e_s') if complete else None
            rows.append(dict(id=request['request_id'],cell=r.get('placement',{}).get('compute_owner_compute_node_id') or r.get('decision',{}).get('serving_compute_node_id'),
                state='completed' if complete else 'incomplete',arrival_s=arrival,completed_s=arrival+latency if latency is not None else None,
                latency_s=latency,ttft_s=timing.get('ttft_s'),tpot_s=timing.get('tpot_s'),tokens=None,
                input_s=None,cell_s=None,tail_s=None,failure=r.get('outcome',{}).get('reason')))
    return dict(id=identity,kind='legacy',requests=rows,summary=summarize(rows),operators=[],transfers=[],migrations=[],
        raw_url=f'/runs/{identity}/runtime-data/index.json',communication_mode='legacy')


@lru_cache(maxsize=2)
def _scene_events(path, stamp):
    from event_catalog import annotate, request_milestones
    from native_environment import enable_device_native
    enable_device_native()
    from astermind.builder.config import SimulationConfig
    from astermind.engine.state.device_network import DeviceNetworkState
    result=read(path);config=SimulationConfig.model_validate(result['configuration'])
    net=DeviceNetworkState(config.topology(),config.horizon_ns/1e9,config.backend)
    bindings=config.physical_link_bindings
    links={k:bindings.get(meta.get('device_link_id')) for k,meta in net._meta.items()}
    transfers={r['id']:r.get('correlation_id') for r in result['network'].get('flows',[])}
    surrogates={r['id']:r for r in result['network'].get('surrogate_operations',[])}
    rows=[]
    for raw in result['network']['event_log']:
        row=dict(raw)
        if row['kind']=='surrogate_complete':
            source=surrogates.get(row.get('surrogate_id'),{})
            row.update({key:source[key] for key in ('correlation_id','operation','data_bytes','devices') if key in source})
        for key in ('source_node_id','target_node_id','resume_node_id','original_source_node_id'):
            if key in row:row[key]=net.owner.get(row[key],row[key])
        if row.get('link_id') in links:
            if links[row['link_id']] is None:continue
            row['link_id']=links[row['link_id']]
        if row.get('target_id') in net.owner:row['target_id']=net.owner[row['target_id']]
        if row.get('request_id','') and row['request_id'].startswith('__cell_transfer__/'):
            transfer=row.get('cell_transfer_id') or row['request_id'].removeprefix('__cell_transfer__/')
            row['request_id']=transfers.get(transfer) or transfer
        if row['kind']=='topology_command' and row.get('scope')=='gpu_fabric':continue
        rows.append(row)
    present = {(row['kind'], row.get('request_id')) for row in rows}
    rows.extend(row for row in request_milestones(result) if (row['kind'], row['request_id']) not in present)
    rows.sort(key=lambda row: row['time_s'])
    rows = [annotate(dict(row, event_order=i)) for i, row in enumerate(rows)]
    return dict(events=rows,end_s=config.horizon_ns/1e9,disabled_links=list(config.disabled_physical_links),
        category_counts=dict(Counter(e['category'] for e in rows)),
        event_counts=dict(Counter(e['kind'] for e in rows)),initial_events=sum(e['time_s']==0 for e in rows))


def scene_events(root, identity):
    from studio import key
    path=Path(root)/'local/results/network'/key(identity)/'native_result.json'
    stat=path.stat()
    if stat.st_size>64*1024*1024:
        raise ValueError('请从场景回放入口按请求查询已归档传输，无需读取全部原始事件')
    result=read(path)
    if result.get('schema_version')==2:
        from runtime_events import result_events
        return result_events(result)
    return _scene_events(str(path),(stat.st_size,stat.st_mtime_ns))
