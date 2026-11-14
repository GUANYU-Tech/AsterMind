"""Joint collective scenarios served by the original local OrbitServe console."""
import json
from pathlib import Path
import subprocess
import sys
import threading

from paths import WORKSPACE as ROOT
JOBS={}
LOCK=threading.Lock()

def key(mode,backend):
    if mode not in ('ring4','multi') or backend not in ('flowsim','ns3'):
        raise ValueError('unknown collective scenario/backend')
    return mode+'-'+backend

def data(mode,backend):
    path=ROOT/'local/results/collectives'/(key(mode,backend)+'.json')
    if not path.is_file(): raise ValueError('该场景尚未运行，请点击重新仿真')
    return json.loads(path.read_text(encoding='utf8'))

def status(mode,backend):
    identity=key(mode,backend)
    with LOCK: value=dict(JOBS.get(identity,{}))
    if not value:value={'state':'completed' if (ROOT/'local/results/collectives'/(identity+'.json')).is_file() else 'missing'}
    return value

def start(mode,backend):
    raise ValueError('此历史预设依赖旧 FlowSim/ns-3 或旧部署契约，V1 不支持等价运行。请在业务与部署页面生成 V1 原生场景。')
    identity=key(mode,backend)
    with LOCK:
        if JOBS.get(identity,{}).get('state')=='running':raise ValueError('该场景正在运行')
        JOBS[identity]={'state':'running'}
    def work():
        try:
            folder=ROOT/'local/results/collectives';folder.mkdir(parents=True,exist_ok=True)
            temporary=folder/(identity+'.pending.json')
            from native_environment import inference_runtime
            python,environment=inference_runtime()
            process=subprocess.run([python,'-B',str(ROOT/'examples/astermind_joint/collectives.py'),
                '--mode',mode,'--backend',backend,'--output',str(temporary)],cwd=ROOT,env=environment,capture_output=True,text=True)
            if process.returncode:raise ValueError((process.stderr or process.stdout)[-3000:])
            temporary.replace(folder/(identity+'.json'))
            with LOCK:JOBS[identity]={'state':'completed'}
        except Exception as error:
            with LOCK:JOBS[identity]={'state':'failed','error':str(error)}
    threading.Thread(target=work,daemon=True).start()
    return {'state':'running'}
