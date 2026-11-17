"""Controllable continuous-time sessions using the production Runtime engine."""

from collections import deque
from copy import deepcopy
import json
import bisect
import math
from pathlib import Path
import threading
import time

SESSIONS = {}
LOCK = threading.Lock()


class SessionStopped(RuntimeError):
    pass


class LiveSession:
    def __init__(self, identity, output=None):
        self.id = identity
        self.output = Path(output) if output is not None else None
        self.condition = threading.Condition()
        self.commands = deque()
        self.engine = None
        self.stopped = False
        self.version = 0
        self.persisted_events = 0
        self.counted_events = 0
        self.completed_requests = self.failed_requests = 0
        self.target_s = None
        self.advance_mode = 'event'
        self.event_start = 0
        self.event_scan = 0
        self.pause_requested = False
        self.stream_at = time.monotonic()
        self.record_lock = threading.RLock()
        from event_catalog import EventWindow
        self.event_window = EventWindow()
        from live_replay import LiveReplayWindow
        self.replay = LiveReplayWindow()
        self.payload = dict(id=identity, state='preparing', time_s=0, preparation_started_at=time.time(), message='正在初始化仿真环境…')

    def record_view(self, engine):
        return engine

    def persist_paths(self, view):
        pass

    def _persist_events(self, engine):
        """Publish completed evidence without pausing or constructing a 3D frame."""
        if self.output is None:
            return
        import gzip
        import shutil
        with self.record_lock:
            folder = self.output/'runtime-chunks'
            folder.mkdir(exist_ok=True)
            target = self.output/'runtime_events.jsonl.gz'
            while self.persisted_events < len(engine.log):
                end = min(len(engine.log), self.persisted_events + 4096)
                part = folder/f'{self.persisted_events:012d}-{end:012d}.jsonl.gz'
                temporary = part.with_suffix('.tmp')
                # Compression finishes before any reader sees this member.
                # Immutable chunks are the live source; the concatenated file
                # remains compatible with existing result/export readers.
                with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=1) as stream:
                    for event in engine.log[self.persisted_events:end]:
                        stream.write(json.dumps(event, ensure_ascii=False)+'\n')
                original_size = target.stat().st_size if target.exists() else 0
                try:
                    with temporary.open('rb') as source, target.open('ab') as stream:
                        shutil.copyfileobj(source, stream)
                    temporary.replace(part)
                except BaseException:
                    # An interrupted append cannot poison subsequent members.
                    with target.open('r+b') as stream:
                        stream.truncate(original_size)
                    raise
                self.persisted_events = end

    def _capture(self, engine, state='paused'):
        for event in engine.log[self.counted_events:]:
            self.completed_requests += event['kind']=='request_complete'
            self.failed_requests += event['kind']=='request_failed'
        self.counted_events = len(engine.log)
        self._persist_events(engine)
        frame = engine.net.snapshot(engine.now)
        faults = list(engine.net.active_faults.values())
        def sources(kind, identity, endpoints=()):
            return sorted({f.get('source', 'scheduled') for f in faults
                           if (f['target_type'] == kind and f['target_id'] == identity)
                           or (f['target_type'] == 'node' and f['target_id'] in endpoints)})
        nodes = [dict(**{key: n.get(key) for key in ('plane', 'slot', 'shell_id', 'cell_ingress', 'compute_target', 'isl_terminals', 'sgl_terminals')},
                      node_id=n['node_id'], role=n['role'], cell_id=n.get('cell_id'),
                      state=n['state'], position_itrs_km=n['position_itrs_km'],
                      sources=sources('node', n['node_id'])) for n in frame['nodes']]
        links = [dict(link_id=e['link_id'], endpoint_a=e['endpoint_a'], endpoint_b=e['endpoint_b'],
                      kind=e['kind'], state=e['state'], reason=e.get('reason'), ready_s=e.get('ready_s'),
                      bandwidth_bps=e['bandwidth_bps'], distance_km=e['distance_km'],
                      propagation_delay_s=e.get('propagation_delay_s', 0),
                      manual_disconnected=e['link_id'] in engine.net.manual_disabled_links,
                      sources=sources('link', e['link_id'], (e['endpoint_a'], e['endpoint_b'])))
                 for e in frame['links']]
        self.version += 1
        replay = self.replay.capture(engine.log, dict(time_s=engine.now, nodes=nodes, links=links),
                                     preserve_from=self.event_start if self.advance_mode == 'event' else None)
        self.payload = dict(id=self.id, state=state, time_s=engine.now, end_s=engine.end,
                            step_s=engine.env.time.get('step_s'), version=self.version,
                            geometry_sample_s=engine.net.sample_time(engine.now),
                            topology_revision=engine.net.revision, nodes=nodes, links=links,
                            events=[dict(e, event_order=i) for i, e in enumerate(engine.log[-300:], max(0, len(engine.log)-300))],
                            event_count=len(engine.log), event_groups=self.event_window.capture(engine.log),
                            request_count=len(engine.jobs),
                            completed=self.completed_requests,
                            failed=self.failed_requests,
                            backend=engine.net.spec.transport_model,
                            sun_positions=[dict(time_s=t, position_itrs_km=p) for t, p in engine.net.sun_positions],
                            replay=replay, inter_cell_links=engine.net.spec.inter_cell_links,
                            message='当前仿真时刻已暂停；操作由后端校验并执行' if state=='paused' else '仿真执行完成，日志已保留')
        from astermind.engine.clock import Time
        if isinstance(engine.now, Time) and getattr(engine,'transport_provenance',{}).get('clock'):
            self.payload.update(clock=engine.transport_provenance['clock'],time_ticks=str(engine.now.ticks))

    def poll(self, engine):
        with self.condition:
            if self.stopped:
                raise SessionStopped('用户停止了交互仿真')
            pause = self.pause_requested
            event_ready = (self.payload['state'] == 'advancing' and self.advance_mode == 'event'
                           and self._event_ready(engine))
        if pause or event_ready:
            self.checkpoint(engine, force=True)
        elif self.payload['state'] == 'advancing' and time.monotonic() >= self.stream_at:
            # A slow look-ahead batch must not withhold all communication until
            # its boundary. Publish on the solver thread, at completed events.
            self.stream_at = time.monotonic() + .5
            view = self.record_view(engine)
            self._persist_events(view)
            self.persist_paths(view)

    def _event_ready(self, engine):
        from event_catalog import is_playback_event
        ready = any(is_playback_event(e) for e in engine.log[self.event_scan:])
        self.event_scan = len(engine.log)
        return ready

    def checkpoint(self, engine, force=False):
        with self.condition:
            if not force and self.payload['state'] == 'advancing' and self.advance_mode == 'event':
                if not self._event_ready(engine):
                    return
            if (not force and self.payload['state'] == 'advancing'
                    and self.target_s is not None and engine.now < self.target_s):
                return
            self.engine = engine
            self.pause_requested = False
            self._capture(engine)
            self.condition.notify_all()
            while True:
                if self.stopped:
                    raise SessionStopped('用户停止了交互仿真')
                if self.commands:
                    command, result, done = self.commands.popleft()
                    try:
                        if command.get('topology_revision') != engine.net.revision:
                            raise ValueError('拓扑状态已变化，请刷新后重试')
                        command_engine = self.command_engine(engine)
                        result['result'] = self.apply_command(engine, command)
                        self._save_overlay(command_engine, command, result['result'])
                    except (ValueError, RuntimeError) as error:
                        result['error'] = str(error)
                    self._capture(engine)
                    done.set()
                    continue
                if self.payload['state'] == 'advancing':
                    target = self.target_s
                    if target is not None and target < engine.end and not any(
                            event[0] == target and event[3] == 'control_boundary' for event in engine.heap):
                        engine.push(target, 5, 'control_boundary')
                    return
                self.condition.wait()

    def command_engine(self, engine):
        return engine

    def apply_command(self, engine, command):
        from astermind.engine.state.interventions import apply_topology_command
        return apply_topology_command(self.command_engine(engine), command)

    def _save_overlay(self, engine, command, result):
        if command['action'] == 'connect':
            # Preserve the actual new resource for later 3D replay as well.
            result['edge'] = dict(engine.net.edge_by_id[result['target_id']])
            result['link_state'] = dict(engine.net.link_at(result['target_id'], engine.now))
        if self.output is None:
            return
        from astermind.common import write_json
        with (self.output/'topology_commands.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(dict(time_s=engine.now, command=command, result=result), ensure_ascii=False)+'\n')
        write_json(self.output/'topology_overlay.json', dict(
            schema_version='orbitserve.topology_overlay.v1',
            trace_id=getattr(engine.net.trace, 'trace_id', None),
            time_s=engine.now, revision=engine.net.revision,
            disabled_links=sorted(engine.net.manual_disabled_links),
            manual_faults=[dict(f) for f in engine.net.active_faults.values() if f.get('source')=='manual'],
            added_links=[dict(e) for e in engine.net.edges if e['link_id'] in engine.net.added_edge_ids],
        ))

    def finished(self, engine):
        with self.condition:
            self._capture(engine, 'finalizing')
            for _, result, done in self.commands:
                result['error'] = '仿真已经结束，此操作未执行'
                done.set()
            self.commands.clear()
            self.condition.notify_all()

    def state(self, after=None):
        with self.condition:
            if after is not None and after == self.payload.get('version'):
                return dict(unchanged=True, state=self.payload['state'], version=after)
            return deepcopy(self.payload)

    def advance(self, until_s=None, mode=None):
        with self.condition:
            if self.payload['state'] != 'paused' or self.commands:
                raise ValueError('请等待当前推进完成或先暂停')
            now = self.engine.now
            mode = mode or ('time' if until_s is not None else 'event')
            if mode not in {'event', 'time'} or (mode == 'event' and until_s is not None):
                raise ValueError('选择逐事件推进，或指定仿真目标时刻')
            if mode == 'time' and until_s is None:
                times = self.engine.net.environment_times
                until_s = times[min(bisect.bisect_right(times, now), len(times)-1)]
            if mode == 'time' and (isinstance(until_s, bool) or not isinstance(until_s, (int, float))
                    or not math.isfinite(until_s) or not now < until_s <= self.engine.end):
                raise ValueError('目标时刻必须晚于当前仿真时刻且不超出仿真时域')
            self.advance_mode = mode
            self.event_start = len(self.engine.log)
            self.event_scan = self.event_start
            self.target_s = float(until_s) if until_s is not None else None
            self.version += 1
            self.payload.update(state='advancing', version=self.version,
                                message='正在执行下一事件…' if mode == 'event' else f'正在执行至 {until_s:g} s…')
            self.condition.notify_all()
            return dict(id=self.id, state='advancing', version=self.version)

    def pause(self):
        with self.condition:
            if self.payload['state'] not in {'paused', 'advancing'}:
                raise ValueError('当前会话不能暂停')
            self.pause_requested = self.payload['state'] == 'advancing'
            self.condition.notify_all()
            return dict(id=self.id, state=self.payload['state'], message='已请求在当前事件处理完成后暂停')

    def command(self, command):
        result, done = {}, threading.Event()
        with self.condition:
            if self.payload['state'] not in {'paused', 'advancing'}:
                raise ValueError('当前会话不能修改拓扑')
            if command.get('topology_revision') != self.engine.net.revision:
                raise ValueError('拓扑状态已变化，请刷新后重试')
            self.commands.append((dict(command), result, done))
            if self.payload['state'] == 'advancing':
                self.pause_requested = True
            self.condition.notify_all()
        if not done.wait(60):
            raise RuntimeError('后端仍在处理此操作，请等待状态更新，勿重复提交')
        if 'error' in result:
            raise ValueError(result['error'])
        return result['result']

    def stop(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        return dict(id=self.id, state='stopping')


def get(identity):
    from studio import key
    with LOCK:
        if key(identity) not in SESSIONS:
            # Terminal native scenes remain inspectable after a console restart.
            root = __import__('paths').WORKSPACE
            output = root/'local/results/network'/key(identity)
            snapshot = output/'live_state.json'
            if not snapshot.is_file():
                raise ValueError('在线会话不存在或服务已重启；已完成日志可从实验结果查看')
            payload = json.loads(snapshot.read_text(encoding='utf-8'))
            if payload.get('state') not in {'completed','cancelled','failed'} or payload.get('id') != identity:
                raise ValueError('已保存的场景状态无效')
            session = LiveSession(identity, output)
            if 'event_groups' not in payload and (output/'native_result.json').is_file():
                from result_analysis import scene_events
                payload['event_groups'] = session.event_window.capture(scene_events(root, identity)['events'])
            session.payload = payload
            session.version = payload['version']
            SESSIONS[identity] = session
        return SESSIONS[identity]


def start(root, value):
    import console
    from orbitserve.experiments import load_experiment_spec, run_experiment
    from astermind.experiments.output import experiment_output_payload
    from astermind.engine import runtime_control
    if value.get('dse', {}).get('enabled'):
        raise ValueError('交互会话只运行一个固定设计；DSE 请使用批量运行')
    with LOCK:
        if any(s.payload['state'] in {'preparing', 'paused', 'advancing'} for s in SESSIONS.values()):
            raise ValueError('已有交互会话，请先结束当前会话')
        identity, path = console.save_input(value)
        # Capture the exact validated input before scheduling a worker. Editing
        # the audit JSON afterwards cannot alter the accepted execution.
        spec = load_experiment_spec(path)
        output = Path(root) / 'local/results/network' / identity
        output.mkdir(parents=True, exist_ok=False)
        session = LiveSession(identity, output)
        SESSIONS[identity] = session
    job = dict(id=identity, state='running', input=str(path.relative_to(root)), output=str(output),
               interactive=True, runtime_url=f'/live/{identity}')
    console.JOBS[identity] = job
    console.status(job, 'running')

    def worker():
        try:
            def progress(message):
                with session.condition:
                    session.payload['message'] = '正在准备物理场景与计算资源…'
                    with (output/'execution.log').open('a', encoding='utf-8') as stream:
                        stream.write(message+'\n')
                    if session.stopped:
                        raise SessionStopped('用户停止了交互仿真')
            with runtime_control(session):
                result = run_experiment(spec, output_dir=output, progress=progress, _workers=1)
            summary = experiment_output_payload(result, artifact_dir=output, output_path=output/'summary.json')
            (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            console.export_runtime_views(output)
            console.status(job, 'completed', result_url=f'/runs/{identity}/result')
            with session.condition:
                session.version += 1
                session.payload.update(state='completed', version=session.version, message='仿真执行完成，完整日志已保存')
                session.condition.notify_all()
        except Exception as error:
            with session.condition:
                session.version += 1
                session.payload.update(state='cancelled' if isinstance(error, SessionStopped) else 'failed',
                                       message=str(error), version=session.version)
                if session.engine is not None:
                    import gzip
                    with gzip.open(output/'runtime_events.jsonl.gz', 'wt', encoding='utf-8') as stream:
                        for event in session.engine.log:
                            stream.write(json.dumps(event, ensure_ascii=False)+'\n')
                session.condition.notify_all()
                for _, pending, done in session.commands:
                    pending['error'] = str(error)
                    done.set()
                session.commands.clear()
            console.status(job, session.payload['state'], error=str(error))
    threading.Thread(target=worker, name=f'live-{identity}', daemon=True).start()
    return dict(id=identity, url=f'/live/{identity}', state='preparing')
