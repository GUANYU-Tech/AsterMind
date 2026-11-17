"""Run editable GPU-level scenes through the existing live controller."""
from datetime import datetime
import json
from pathlib import Path
import threading
from types import SimpleNamespace

from live import LiveSession, LOCK, SESSIONS, SessionStopped

LABELS = {
    'flowsim': '全部通信使用 FlowSim',
    'protobridge_satellite': '卫星内通信使用 ProtoBridge',
    'protobridge_cell': 'Cell 内通信使用 ProtoBridge',
}

def execution_label(config):
    return 'V1.0 Runtime Comm · ' + config.compute_mode


class PhysicalCommandView:
    def __init__(self, engine):
        self.engine, self.net = engine, engine.net.physical

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def schedule_reallocation(self):
        self.engine.net._project()
        self.engine.schedule_reallocation()


class DeviceLiveSession(LiveSession):
    def __init__(self, identity, output, config, automatic):
        super().__init__(identity, output)
        self.config, self.automatic = config, automatic
        from live_paths import FlowPathWindow
        self.flow_paths=FlowPathWindow(archive=output)
        self.projected_events=[]
        self.event_cursors={}

    def command_engine(self, engine):
        return PhysicalCommandView(engine)

    def apply_command(self, engine, command):
        if not command.get('action','').startswith('device_'):
            return super().apply_command(engine, command)
        from astermind.engine.device_controls import apply_device_command
        return apply_device_command(engine, command)

    def record_view(self, engine):
        physical = engine.net.physical
        from runtime_events import project
        streams={'host':engine.log,'comm':engine.communication_policy.trace}
        if hasattr(self,'simulation'):
            streams['requests']=self.simulation.request_state_events
        fresh=[]
        owners={**engine.net.owner,**{device['id']:device['satellite'] for device in self.config.physical['devices']}}
        for name,rows in streams.items():
            fresh.extend(project(e,owners) for e in rows[self.event_cursors.get(name,0):])
            self.event_cursors[name]=len(rows)
        fresh.sort(key=lambda e:int(e.get('time_ticks') or round(e['time_s']*1e18)))
        self.projected_events.extend(fresh)
        def project(event):
            row = dict(event)
            for key in ('source_node_id', 'target_node_id', 'resume_node_id', 'original_source_node_id', 'compute_node_id', 'flow_source_node_id', 'flow_target_node_id'):
                if key in row:
                    row[key] = engine.net.owner.get(row[key], row[key])
            edge = engine.net.edge_by_id.get(row.get('link_id'))
            if edge:
                row['link_id'] = engine.net.metadata(edge).get('physical_link_id', row['link_id'])
            return row
        def project_event(event):
            row = project(event)
            if row.get('path'):
                row['path'] = [project(leg) if isinstance(leg,dict) else leg for leg in row['path']]
            if row.get('kind') == 'surrogate_complete' and row.get('devices'):
                row['compute_node_id'] = engine.net.owner.get(row['devices'][0])
            return row
        from astermind.engine.record_store import RecordView
        return SimpleNamespace(net=physical, now=engine.now, end=engine.end,
                               log=RecordView(self.projected_events, range(len(self.projected_events)), project_event), jobs=self.config.requests,
                               env=SimpleNamespace(time=physical.trace.time))

    def persist_paths(self, view):
        for order in range(self.flow_paths.cursor,len(view.log)):
            self.flow_paths.observe(view.log[order],order)
        if self.output is not None:
            self.flow_paths.save(self.output)

    def _capture(self, engine, state='paused'):
        view = self.record_view(engine)
        physical = engine.net.physical
        super()._capture(view, state)
        self.persist_paths(view)
        for link in self.payload['links']:
            link['configuration_disabled'] = (link['manual_disconnected']
                and link['link_id'] in self.config.disabled_physical_links)
        from astermind.engine.clock import CLOCK
        self.payload.update(communication_mode='runtime_comm',
                            clock=CLOCK,
                            time_ticks=str(engine.now.ticks),
                            communication_label=execution_label(self.config),timeline_mode=self.config.timeline_mode,
                            backend=self.config.backend,timeline_authority='simgrid_engine' if self.config.timeline_mode=='simgrid' else 'orbitserve_runtime',
                            gpu_count=sum(d.kind == 'gpu' for d in engine.net.topology.devices.values()),
                            execution_model='native_gpu_operator_graph')
        self.payload['device_links'] = [dict(id=l.id,source=l.source,target=l.target,
            state='DOWN' if not l.enabled or l.id in engine.net.disabled_device_links or l.id in engine.net.failed_device_links
            or any(f['target_type']=='node' and f['target_id']==engine.net.topology.devices[l.source].satellite
                   for f in physical.active_faults.values()) else 'UP')
            for l in engine.net.topology.links
            if engine.net.topology.devices[l.source].satellite==engine.net.topology.devices[l.target].satellite]

    def checkpoint(self, engine, force=False):
        if self.automatic:
            self.engine = engine
            # Batch runs export their final state once; building a 3D snapshot
            # at every physical sample would erase part of the solver speedup.
            return
        return super().checkpoint(engine, force)

    def _event_ready(self, engine):
        view=self.record_view(engine)
        from event_catalog import is_playback_event
        ready=any(is_playback_event(e) for e in view.log[self.event_scan:])
        self.event_scan=len(view.log)
        return ready


def start(root, value, *, automatic=False):
    import console
    from native_environment import enable_device_native
    from native_scene import validate_document
    enable_device_native()
    _, value = validate_document(root, dict(name='运行配置', experiment=value))
    from astermind.builder.config import SimulationConfig
    config = SimulationConfig.model_validate(value)
    if not config.physical_dataset and not config.physical_network:
        raise ValueError('三维场景需要物理数据包；静态设备计算请使用计算 Cell 工作区')
    root = Path(root)
    with LOCK:
        if any(s.payload['state'] in {'preparing','paused','advancing','finalizing'} for s in SESSIONS.values()):
            raise ValueError('已有在线仿真，请先结束当前会话')
        identity = 'native-scene-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        output = root/'local/results/network'/identity
        inputs = root/'local/inputs/network'/identity
        output.mkdir(parents=True, exist_ok=False)
        inputs.mkdir(parents=True, exist_ok=False)
        (inputs/'manifest.json').write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        (output/'run_metadata.json').write_text(json.dumps(dict(title=execution_label(config),timeline_mode=config.timeline_mode)), encoding='utf-8')
        session = DeviceLiveSession(identity, output, config, automatic)
        SESSIONS[identity] = session
    job = dict(id=identity, state='running', interactive=True, native_scene=True,
               input=str((inputs/'manifest.json').relative_to(root)), output=str(output),
               runtime_url=f'/live/{identity}')
    console.JOBS[identity] = job
    console.status(job, 'running')
    from current_experiment import select
    select(root, identity)

    def worker():
        try:
            from astermind.engine import runtime_control
            from astermind.engine.simulation import Simulation
            from astermind.__main__ import json_default
            from astermind.engine.audit import audit
            from native_live import physical_export
            def progress(stage):
                label={'bootstrap':'加载执行环境','physics':'读取已生成的物理拓扑','network':'建立通信资源与全局时间轴','ready':'初始化完成，准备推进仿真'}.get(stage['phase'])
                if stage['phase']=='cell':
                    label=f"初始化计算 Cell {stage['index']}/{stage['total']}：计算模型、调度器与 KV 状态"
                if stage['phase']=='cells':
                    mode='并行初始化' if stage['workers']>1 else '初始化'
                    label=f"{mode}计算 Cell · 已就绪 {stage['completed']}/{stage['total']} · 同时最多 {stage['workers']} 个"
                if stage['phase']=='physics' and config.physical_dataset is None:
                    label='准备物理拓扑' if config.physical_network is not None else '建立设备拓扑'
                with session.condition:
                    if session.stopped:
                        raise SessionStopped('用户停止了交互仿真')
                    session.version += 1
                    session.payload.update(version=session.version,message=label,startup_stage=stage)
                    session.condition.notify_all()
                console.status(job,'running',phase=label)
            with runtime_control(session), Simulation(config,startup_progress=progress,physical_output_dir=output/'physical') as simulation:
                session.simulation = simulation
                console.status(job, 'running', phase='执行全局事件时间轴')
                result = simulation.run()
                from native_result_store import save_audited_result
                save_audited_result(output/'native_result.json', result,
                                    auditor=audit, default=json_default)
                from scene_outputs import export
                export(output,result,config,simulation.physical)
            console.status(job, 'completed', result_url=f'/runs/{identity}/result')
            with session.condition:
                session.version += 1
                session.payload.update(state='completed', version=session.version, message='仿真完成，算子与通信记录已保存')
                (output/'live_state.json').write_text(json.dumps(session.payload, ensure_ascii=False), encoding='utf-8')
                session.engine = None
                session.condition.notify_all()
        except Exception as error:
            import traceback
            (output/'execution.log').write_text(traceback.format_exc(), encoding='utf-8')
            state = 'cancelled' if isinstance(error, SessionStopped) else 'failed'
            console.status(job, state, error=str(error))
            with session.condition:
                session.version += 1
                session.payload.update(state=state, version=session.version, message=str(error))
                (output/'live_state.json').write_text(json.dumps(session.payload, ensure_ascii=False), encoding='utf-8')
                session.engine = None
                for _, result, done in session.commands:
                    result['error'] = str(error)
                    done.set()
                session.commands.clear()
                session.condition.notify_all()
    threading.Thread(target=worker, name=identity, daemon=True).start()
    return dict(id=identity, url=f'/live/{identity}', state='preparing')
