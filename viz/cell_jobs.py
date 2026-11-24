"""Persistent frontend jobs and usage records; every output lives under viz."""
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import uuid

from paths import VIZ, RUNTIME, owned
from runtime_environment import identity

ROOT = RUNTIME / 'cell-jobs'
LOCK = threading.Lock()


def write(path, value):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2))
    temporary.replace(path)


def folder(job):
    if not re.fullmatch(r'AM-\d{8}-[0-9a-f]{32}', job):
        raise ValueError('无效任务 ID')
    return owned(ROOT/job)


def status(job):
    path=folder(job)/'status.json'
    if not path.is_file():
        raise ValueError('任务不存在')
    return json.loads(path.read_text())


def start(config, client):
    from cell_adapter import convert
    job='AM-'+datetime.now().strftime('%Y%m%d')+'-'+uuid.uuid4().hex
    destination=folder(job)
    destination.mkdir(parents=True)
    write(destination/'configuration.json',config)
    record=dict(id=job,status='queued',phase='queued',progress=0,
        created_at=datetime.now(timezone.utc).isoformat(),model=config.get('model'),
        satellite_count=config.get('hardware',{}).get('satellite_count'),ip=client,backend=identity())
    try:
        effective,_ = convert(config)
        if effective['compute_mode'] == 'fast_mode':
            from compute_options import require_fast_scheduler
            require_fast_scheduler()
    except Exception as error:
        write(destination/'status.json',{**record,'status':'rejected','phase':'validation','error':str(error)})
        raise
    write(destination/'effective_configuration.json',effective)
    write(destination/'status.json',record)
    def run():
        try:
            with (destination/'execution.log').open('w') as log:
                process=subprocess.Popen([sys.executable,'-B',str(VIZ/'cell_worker.py'),job],
                    cwd=VIZ,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                process.wait()
            if process.returncode and status(job)['status'] not in ('failed','cancelled'):
                raise RuntimeError(f'V1 worker 退出码 {process.returncode}；详见执行日志')
        except Exception as error:
            write(destination/'status.json',{**status(job),'status':'failed','error':str(error)})
    threading.Thread(target=run,name=job,daemon=True).start()
    return record


def project(result):
    requests={key:dict(request_id=key,status=row['status'],
        **{target: None if row.get('metrics',{}).get(source) is None else row['metrics'][source]/1e9
           for source,target in [('ttft_ns','ttft_s'),('tpot_ns','tpot_s'),('e2e_ns','latency_s')]})
        for key,row in result['requests'].items()}
    finished=[r.get('completed_ns') for r in result['requests'].values() if r.get('completed_ns') is not None]
    steps=sum(len(c.get('steps',[])) for c in result.get('cells',{}).values())
    calls=sum(len(b.get('calls',[])) for b in result.get('batches',[]) if isinstance(b,dict))
    return dict(status=result['status'],simulator='AsterMind V1.0',network_backend='runtime_comm',
        versions=identity(),metrics=dict(serving=dict(requests=requests,
            completed=sum(r['status']=='completed' for r in result['requests'].values()),
            failed=sum(r['status']=='failed' for r in result['requests'].values())),
            simulation=dict(wall_time_ns=result.get('measurement',{}).get('simulated_time_ns'),
                scheduler_steps=steps,kernel_calls=calls,flow_count=sum(e['kind']=='admit' for e in result.get('communication_events',[])),
                timing_mode='device_only'), network_by_level={'runtime_comm':result.get('communication_runtime_stats',{})},host={},kv_offload={}),
        warnings=['数据来自 V1 原始报告；旧引擎的逐层链路字节及 Host/KV 指标没有等价统计时显示不可用。'])


def artifacts(job):
    priority=['requests.json','result.json','placement.json','communication_events.json','configuration.json','effective_configuration.json']
    return dict(items=[dict(key=p.name,size=p.stat().st_size) for p in sorted(folder(job).iterdir(),key=lambda p:(priority.index(p.name) if p.name in priority else 99,p.name))
                       if p.is_file() and p.suffix in ('.json','.log') and not p.name.endswith('.tmp')])


def artifact(job,key,offset=0,limit=100):
    path=folder(job)/key
    if key not in {i['key'] for i in artifacts(job)['items']}:
        raise ValueError('产物不存在')
    value=json.loads(path.read_text()) if path.suffix=='.json' else path.read_text()
    def bounded(v,depth=0):
        if depth>4: return '[嵌套数据；完整内容请下载]' if isinstance(v,(dict,list)) else v
        if isinstance(v,list):
            return [bounded(x,depth+1) for x in v[:8]] + (['… 完整数组请下载'] if len(v)>8 else [])
        if isinstance(v,dict):return {k:bounded(x,depth+1) for k,x in list(v.items())[:40]}
        if isinstance(v,str) and len(v)>2000:return v[:2000]+'… 完整内容请下载'
        return v
    if isinstance(value,list):
        return dict(items=[bounded(x) for x in value[offset:offset+limit]],has_more=offset+limit<len(value),preview_only=True)
    return dict(value=bounded(value),has_more=False,preview_only=True)


def usage(query):
    records=[]; skipped=0
    for p in sorted(ROOT.glob('AM-*/status.json'),reverse=True):
        try: records.append(json.loads(p.read_text()))
        except (ValueError,OSError): skipped+=1
    term=query.get('query',[''])[0].lower(); state=query.get('status',[''])[0]
    records=[r for r in records if (not state or r['status']==state) and
             (not term or term in json.dumps(r,ensure_ascii=False).lower())]
    offset=max(0,int(query.get('offset',[0])[0])); limit=max(1,min(100,int(query.get('limit',[25])[0])))
    return dict(items=records[offset:offset+limit],total=len(records),skipped=skipped)


def detail(job):
    p=folder(job)
    record=status(job)
    return {**record,'client':{'ip':record.get('ip')},'artifacts':artifacts(job)['items'],
            'submitted_configuration':json.loads((p/'configuration.json').read_text()),
            'effective_configuration':json.loads((p/'effective_configuration.json').read_text()) if (p/'effective_configuration.json').exists() else None}
