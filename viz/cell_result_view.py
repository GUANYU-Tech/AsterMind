"""Bounded, visual Cell reports derived from saved V1 observations.

Only presentation data is derived here. No execution or prediction is changed.
Interval totals are unions, so concurrent kernels are never added as wall time.
"""
from collections import defaultdict
from functools import lru_cache
import json
import math


def number(value, fallback=0):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else fallback


def intervals(items):
    merged = []
    for start, end in sorted(items):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return merged


def duration(items):
    return sum(end - start for start, end in intervals(items))


def build_view(native, submitted):
    ranks, calls, kinds = {}, [], defaultdict(lambda: dict(count=0, duration_ns=0))
    configs = {cell['id']: cell for cell in native.get('configuration', {}).get('cells', [])}
    memory = []
    for cell_id, cell_config in configs.items():
        for placement in cell_config.get('placements', []):
            device = placement.get('device', '')
            satellite, _, local = device.rpartition('.gpu')
            rank = placement['rank']
            ranks[(cell_id, rank)] = dict(id=f'{cell_id}/{rank}', cell=cell_id, rank=rank,
                satellite=satellite or device, local_index=int(local) if local.isdigit() else None,
                pool=placement.get('pool', 'main'), compute=[], communication=[], memory=[], calls=0,
                weight_bytes=None, reserved_kv_bytes=None)
    for batch in native.get('batches', []):
        cell_id = batch.get('cell_id', 'cell')
        for call in batch.get('calls', []):
            start, end = number(call.get('start_time_ns')), number(call.get('end_time_ns'))
            if end < start:
                continue
            rank = call.get('rank', 0)
            key = (cell_id, rank)
            if key not in ranks:
                ranks[key] = dict(id=f'{cell_id}/{rank}', cell=cell_id, rank=rank, satellite='未记录',
                    local_index=None, pool=batch.get('pool_name', 'main'), compute=[], communication=[],
                    memory=[], calls=0, weight_bytes=None, reserved_kv_bytes=None)
            kind = call.get('kind', 'OTHER')
            lane = 'communication' if call.get('communication') else 'memory' if kind == 'KV_CACHE' else 'compute'
            ranks[key][lane].append((start, end))
            ranks[key]['calls'] += 1
            item = dict(cell=cell_id, pool=batch.get('pool_name', 'main'), step=batch.get('step_id', ''),
                rank=rank, rank_id=f'{cell_id}/{rank}', name=call.get('call_id', ''), kind=kind,
                communication=bool(call.get('communication')), start_ns=start, end_ns=end,
                duration_ns=end-start, predicted_ns=None if call.get('native_prediction_s') is None else number(call['native_prediction_s'])*1e9,
                operation=call.get('communication_operation'))
            calls.append(item)
            kinds[kind]['count'] += 1
            kinds[kind]['duration_ns'] += end-start
    for cell_id, cell in native.get('cells', {}).items():
        runtime = cell.get('scheduler_runtime', {})
        runtimes = [runtime] if 'capacity' in runtime else [item for item in runtime.values() if isinstance(item, dict)]
        for scheduler in runtimes:
            capacity = scheduler.get('capacity', {})
            for geometry in capacity.get('rank_geometry', []):
                row = ranks.get((cell_id, geometry.get('rank')))
                if row:
                    row['weight_bytes'] = geometry.get('weight_resident_bytes')
                    row['reserved_kv_bytes'] = capacity.get('allocated_kv_tensor_bytes', capacity.get('available_kv_bytes'))
        used, samples, peak, allocations, releases = 0, [[0, 0]], 0, 0, 0
        for step in sorted(cell.get('steps', []), key=lambda item: item.get('end_time_s', 0)):
            events = step.get('step', {}).get('kv_cache_events', [])
            for event in events:
                if event.get('tier') != 'hbm':
                    continue
                size = max(0, number(event.get('bytes')))
                if event.get('action') == 'allocate':
                    used += size
                    allocations += 1
                    peak = max(peak, used)
                elif event.get('action') == 'release':
                    used = max(0, used-size)
                    releases += 1
            samples.append([number(step.get('end_time_s'))*1e9, used])
        memory.append(dict(cell=cell_id, samples=samples, peak_kv_bytes=peak,
            allocation_events=allocations, release_events=releases, available=len(samples)>1))
    all_spans = []
    for row in ranks.values():
        compute = row['compute'] + row['memory']
        comm = row['communication']
        spans = compute + comm
        row['start_ns'] = min((start for start, _ in spans), default=0)
        row['end_ns'] = max((end for _, end in spans), default=0)
        row['wall_ns'] = row['end_ns']-row['start_ns']
        row['compute_ns'], row['communication_ns'] = duration(compute), duration(comm)
        row['overlap_ns'] = max(0, row['compute_ns']+row['communication_ns']-duration(spans))
        row['exposed_communication_ns'] = max(0, row['communication_ns']-row['overlap_ns'])
        row['waiting_ns'] = max(0, row['wall_ns']-duration(spans))
        for lane in ('compute', 'memory', 'communication'):
            row[lane] = intervals(row[lane])
        all_spans.extend(spans)
    rank_rows = sorted(ranks.values(), key=lambda row: (row['cell'], row['rank']))
    critical = max(rank_rows, key=lambda row: row['end_ns'], default=None)
    for row in rank_rows:
        row['critical'] = row is critical and row['calls'] > 0
    requests = []
    for key, row in native.get('requests', {}).items():
        metrics = row.get('metrics', {})
        requests.append(dict(id=key, status=row.get('status'), arrival_ns=row.get('arrival_ns'),
            completed_ns=row.get('completed_ns'), output_tokens=row.get('received_tokens', 0),
            ttft_ns=metrics.get('ttft_ns'), tpot_ns=metrics.get('tpot_ns'), latency_ns=metrics.get('e2e_ns')))
    finished = [row for row in requests if row['status']=='completed' and row['completed_ns'] is not None]
    first_arrival = min((number(row['arrival_ns']) for row in requests), default=0)
    completion = max((row['completed_ns'] for row in finished), default=None)
    elapsed = None if completion is None else max(0, completion-first_arrival)
    admissions, services, concurrency = {}, defaultdict(lambda: dict(bytes=0, duration_ns=0)), []
    for event in native.get('communication_events', []):
        key = (event.get('id'), event.get('attempt_id'))
        if event.get('kind') == 'admit':
            path = event.get('path', [])
            admissions[key] = dict(path=path, start_ns=number(event.get('time_ns')), end_ns=None)
            concurrency.append((number(event.get('time_ns')), 1))
        elif event.get('kind') == 'service':
            services[key]['bytes'] += max(0, number(event.get('application_bytes')))
            services[key]['duration_ns'] += max(0, number(event.get('end_ns'))-number(event.get('start_ns')))
        elif event.get('kind') == 'p2p_complete':
            if key in admissions:
                admissions[key]['end_ns'] = number(event.get('time_ns'))
                concurrency.append((number(event.get('time_ns')), -1))
    paths = {}
    for key, row in admissions.items():
        path = row['path']
        pair = (path[0], path[-1]) if path else ('未记录', '未记录')
        if pair not in paths:
            paths[pair] = dict(source=pair[0], target=pair[1], attempts=0, bytes=0, service_ns=0)
        aggregate = paths[pair]
        aggregate['attempts'] += 1
        aggregate['bytes'] += services[key]['bytes']
        aggregate['service_ns'] += services[key]['duration_ns']
    for row in paths.values():
        row['service_gbps'] = row['bytes']*8/row['service_ns'] if row['service_ns'] else None
    active, concurrent_samples = 0, [[0, 0]]
    deltas = defaultdict(int)
    for time, delta in concurrency:
        deltas[time] += delta
    for time, delta in sorted(deltas.items()):
        active = max(0, active+delta)
        concurrent_samples.append([time, active])
    native_stats = native.get('communication_runtime_stats', {})
    summary = dict(status=native.get('status'), model=submitted.get('model'),
        satellite_count=submitted.get('hardware', {}).get('satellite_count'),
        hardware=submitted.get('hardware', {}).get('default_hardware_id'),
        parallelism=[pool.get('parallelism', {}) for pool in submitted.get('deployment', {}).get('worker_pools', [])],
        completion_ns=elapsed, simulation_ns=native.get('measurement', {}).get('simulated_time_ns'),
        completed=len(finished), failed=sum(row['status']=='failed' for row in requests), requests=len(requests),
        output_tokens=sum(number(row['output_tokens']) for row in finished),
        throughput_rps=len(finished)*1e9/elapsed if elapsed else None,
        kernel_calls=len(calls), scheduler_steps=sum(len(cell.get('steps', [])) for cell in native.get('cells', {}).values()),
        execution_start_ns=min((start for start, _ in all_spans), default=0),
        execution_end_ns=max((end for _, end in all_spans), default=0),
        ranks=[{key:value for key,value in row.items() if key not in ('compute','memory','communication')} for row in rank_rows],
        operator_types=[dict(kind=kind, **value) for kind, value in sorted(kinds.items(), key=lambda item: -item[1]['duration_ns'])],
        memory=memory, request_rows=requests,
        network=dict(attempts=len(admissions), completed=sum(row['end_ns'] is not None for row in admissions.values()),
            application_bytes=sum(row['bytes'] for row in services.values()),
            peak_active=native_stats.get('peak_active_flows'), retries=native_stats.get('retries'),
            paths=sorted(paths.values(), key=lambda row:-row['bytes']), concurrency=concurrent_samples,
            intra_mode=native.get('configuration', {}).get('communication', {}).get('intra_mode')))
    return dict(summary=summary, calls=sorted(calls, key=lambda row:row['start_ns']), ranks=ranks)


@lru_cache(maxsize=2)
def _load(folder, native_stamp, config_stamp):
    from pathlib import Path
    root = Path(folder)
    return build_view(json.loads((root/'native_result.json').read_text(encoding='utf-8')),
                      json.loads((root/'configuration.json').read_text(encoding='utf-8')))


def report(folder, query):
    native, config = folder/'native_result.json', folder/'configuration.json'
    if not native.is_file():
        raise ValueError('此任务尚无完整执行报告；可在产物与明细中查看现有文件。')
    view = _load(str(folder), native.stat().st_mtime_ns, config.stat().st_mtime_ns)
    section = query.get('section', ['summary'])[0]
    if section == 'summary':
        return view['summary']
    if section == 'timeline':
        rank = query.get('rank', [''])[0]
        row = next((row for row in view['ranks'].values() if row['id']==rank), None)
        if row is None:
            raise ValueError('请选择有效 Rank')
        # A long run can contain millions of intervals. Return an explicitly
        # identified window, rather than sampling away short operations.
        offset = max(0, int(query.get('offset', ['0'])[0]))
        limit = max(1, min(500, int(query.get('limit', ['500'])[0])))
        lanes = {}
        for lane in ('compute', 'memory', 'communication'):
            lanes[lane] = dict(items=row[lane][offset:offset+limit], total=len(row[lane]))
        return dict(rank=rank, start_ns=row['start_ns'], end_ns=row['end_ns'], lanes=lanes, offset=offset,
                    has_more=any(offset+limit<len(row[lane]) for lane in lanes))
    if section == 'operators':
        rank, kind, search = (query.get(key, [''])[0] for key in ('rank','kind','search'))
        rows = [row for row in view['calls'] if (not rank or row['rank_id']==rank)
                and (not kind or row['kind']==kind) and (not search or search.casefold() in row['name'].casefold())]
        offset = max(0, int(query.get('offset', ['0'])[0]))
        limit = max(1, min(100, int(query.get('limit', ['25'])[0])))
        return dict(items=rows[offset:offset+limit], total=len(rows), offset=offset, has_more=offset+limit<len(rows))
    raise ValueError('未知结果栏目')
