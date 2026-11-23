"""Run the existing Qwen TP/PP preset through the shared live controller."""
from datetime import datetime
from dataclasses import replace
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import threading

from live import LiveSession, LOCK, SESSIONS, SessionStopped

PRESET = 'qwen3-4b-tp2pp2'
DEPLOYMENT = dict(model='qwen3-4b', tp=2, pp=2, replicas=2, communication_planes=72,
                  satellites_per_plane=22, transport_model='flowsim', users=4)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


class NativeSession(LiveSession):
    def _capture(self, engine, state='paused'):
        super()._capture(engine, state)
        self.payload.update(native_preset=PRESET, deployment=DEPLOYMENT)


def physical_export(output, engine, source):
    """Export actual network snapshots in the existing viewer's frame contract."""
    from astermind.engine.setup.sun_geometry import SUN_GEOMETRY_VERSION
    frames = output/'frames'
    frames.mkdir(exist_ok=True)
    descriptors = []
    for i, time in enumerate(engine.net.environment_times):
        snapshot = engine.net.snapshot(time)
        name = f'frames/frame-{i:06d}.json.gz'
        with gzip.open(output/name, 'wt', encoding='utf-8') as stream:
            json.dump(snapshot, stream, ensure_ascii=False, allow_nan=False)
        descriptors.append(dict(time_s=time, artifact=name))
    network = engine.net.spec.to_dict()
    write_json(output/'network_result.json', dict(
        schema_version='orbitserve.network_result.v1', trace_id=engine.net.trace.trace_id,
        time=source['trace']['trace_manifest']['time'], network=network,
        sun_positions=source['sun_positions'], artifacts=dict(frames=descriptors),
        boundary_sample_times_s=list(engine.net.environment_times),
        candidate_links=list(engine.net.edges), routes=[], events=[], snapshots=[],
        warnings=source['trace'].get('warnings', []),
        provenance=dict(between_samples='sample_hold', sun_geometry_version=SUN_GEOMETRY_VERSION,
                        source='actual_native_preset_runtime'),
        physical_counts=dict(physical_satellites=sum(n.role!='ground' for n in engine.net.nodes.values())),
    ))


def start(root, preset=PRESET):
    raise ValueError('此历史预设依赖旧 FlowSim/ns-3 或旧部署契约，V1 不支持等价运行。请在业务与部署页面生成 V1 原生场景。')
    import console
    from native_environment import ROOT, enable_native_inprocess
    if preset != PRESET:
        raise ValueError('未知的原生并行推理预设')
    root = Path(root)
    with LOCK:
        if any(s.payload['state'] in {'preparing','paused','advancing','finalizing'} for s in SESSIONS.values()):
            raise ValueError('已有交互会话，请先结束当前会话')
        identity = 'qwen3-4b-72x22-flowsim-tp2pp2-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        output = root/'local/results/network'/identity
        output.mkdir(parents=True, exist_ok=False)
        inputs = root/'local/inputs/network'/identity
        inputs.mkdir(parents=True, exist_ok=False)
        write_json(inputs/'manifest.json', dict(schema_version='orbitserve.native_preset.v1',
                   native_preset=PRESET, deployment=DEPLOYMENT))
        write_json(output/'run_metadata.json', dict(title='Qwen3-4B 72×22 TP2×PP2 FlowSim'))
        session = NativeSession(identity, output)
        session.payload.update(native_preset=PRESET, deployment=DEPLOYMENT,
                               message='正在加载 Qwen3-4B、72×22 星座与 FlowSim…')
        SESSIONS[identity] = session
    job = dict(id=identity, state='running', input=str((inputs/'manifest.json').relative_to(root)),
               output=str(output), interactive=True, native_preset=PRESET, runtime_url=f'/live/{identity}')
    console.JOBS[identity] = job
    console.status(job, 'running')

    def worker():
        try:
            enable_native_inprocess()
            from astermind.engine import runtime_control
            path = ROOT/'examples/astermind_joint/full_serving.py'
            spec = importlib.util.spec_from_file_location('local_native_'+identity.replace('-','_'), path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            source = module.physical('tp_pp')
            if (source['shell_satellites'], source['ranks_per_replica'], source['replicas']) != (1584,4,2):
                raise ValueError('默认物理场景与 72×22 / TP2×PP2 预设不一致')
            run_simulation = module.run_simulation
            captured = {}

            def execute(*args, **kwargs):
                simulation = args[0]
                simulation = replace(simulation, network=replace(simulation.network, fault_timing='continuous'))
                result = run_simulation(simulation, *args[1:], **kwargs)
                captured['result'] = result.to_dict()
                return result

            module.run_simulation = execute
            with (output/'execution.log').open('a', encoding='utf-8') as log:
                log.write('Qwen3-4B 72×22 TP2×PP2 2 replicas FlowSim\n')
            with runtime_control(session):
                data = module.run('multi', 'flowsim', 'tp_pp')
            write_json(output/'native_serving.json', data)
            write_json(output/'experiment_result.json', dict(runtime=captured['result'],
                       native_preset=PRESET, deployment=DEPLOYMENT))
            physical_export(output, session.engine, source)
            console.export_runtime_views(output)
            manifest = output/'runtime_playback/index.json'
            index = json.loads(manifest.read_text(encoding='utf-8'))
            index.update(native_preset=PRESET, deployment=DEPLOYMENT)
            write_json(manifest, index)
            files = {p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in output.rglob('*') if p.is_file() and p.name not in {'status.json','execution.log'}}
            write_json(output/'native_run_manifest.json', dict(schema_version='orbitserve.native_preset_run.v1',
                       preset=PRESET, deployment=DEPLOYMENT, files=files))
            console.status(job, 'completed', result_url=f'/runs/{identity}/result')
            with session.condition:
                session.version += 1
                session.payload.update(state='completed', version=session.version,
                                       message='Qwen3-4B 原生推理已完成，执行日志与回放已保存')
                session.condition.notify_all()
        except Exception as error:
            with session.condition:
                session.version += 1
                session.payload.update(state='cancelled' if isinstance(error,SessionStopped) else 'failed',
                                       message=str(error), version=session.version)
                if session.engine is not None:
                    with gzip.open(output/'runtime_events.jsonl.gz','wt',encoding='utf-8') as stream:
                        for event in session.engine.log:
                            stream.write(json.dumps(event,ensure_ascii=False)+'\n')
                for _, result, done in session.commands:
                    result['error'] = str(error)
                    done.set()
                session.commands.clear()
                session.condition.notify_all()
            console.status(job, session.payload['state'], error=str(error))
    threading.Thread(target=worker, name='native-'+identity, daemon=True).start()
    return dict(id=identity, url=f'/live/{identity}', state='preparing')
