"""Personal local network console. Kept outside the distributed source tree."""

import argparse
import errno
import json
import os
import platform
import re
import subprocess
import sys
import sysconfig
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from zoneinfo import ZoneInfo

# Workers import console to register jobs. Running this file as a script must
# share that same module and job registry with the HTTP handlers.
if __name__ == '__main__':
    sys.modules['console'] = sys.modules[__name__]

from viewer import write_network_viewer
from playback import export_runtime_views

from paths import REPOSITORY, WORKSPACE as ROOT
from native_environment import enable_local_flowsim
enable_local_flowsim()
JOBS = {}
LOCK = threading.Lock()
LAUNCH_LOCK = threading.Lock()
COMPARISONS = {}
PHYSICAL_LOCK = threading.Lock()

HOMEPAGE = Path(__file__).parent / 'homepage' / 'dist'
HOMEPAGE_FILES = {
    '/': ('index.html', 'text/html'),
    '/index.html': ('index.html', 'text/html'),
    '/en/': ('en/index.html', 'text/html'),
    '/en/index.html': ('en/index.html', 'text/html'),
    '/style.css': ('style.css', 'text/css'),
    '/language.js': ('language.js', 'text/javascript'),
    '/app.js': ('app.js', 'text/javascript'),
    '/assets/astermind-mark.png': ('assets/astermind-mark.png', 'image/png'),
    '/assets/guanyu-logo.png': ('assets/guanyu-logo.png', 'image/png'),
}


def prepare_physical(identity, *, independent=False):
    from studio import key, physical_data, physical_input
    from astermind.builder.physical.network_inputs import snapshot_network_input
    from astermind.builder.physical.physical_dataset import input_fingerprint, validate_dataset_input

    with PHYSICAL_LOCK:
        data = physical_data(ROOT, identity)
        if data['state'] == 'running':
            return {**run_status(data['run_id']), 'physical_model_id': identity}
        physical = physical_input(ROOT, identity)
        if data['ready']:
            try:
                validate_dataset_input(data['physical_dataset'], physical, ROOT)
            except (ValueError, OSError):
                pass  # Explicit step-one generation repairs stale or damaged data.
            else:
                return {**run_status(data['run_id']), 'physical_model_id': identity}
        fingerprint = input_fingerprint(snapshot_network_input(physical, ROOT))
        reused = None
        indexes = [] if independent else [*(ROOT / 'local/results/network').glob('*/physical_dataset.json'),
                   *(ROOT / 'local/results/network').glob('*/network/physical_dataset.json')]
        for index in indexes:
            try:
                metadata = json.loads(index.read_text(encoding='utf-8'))
                if fingerprint not in {metadata.get('source_sha256'), metadata.get('input_sha256')}:
                    continue
                source_run = index.parent.parent.name if index.parent.name=='network' else index.parent.name
                previous = run_status(source_run)
                if previous['state'] != 'completed':
                    continue
                validate_dataset_input(str(index.parent), physical, ROOT)
            except (ValueError, OSError):
                continue
            reused = {**previous, 'physical_model_id': identity,
                      'preview_url': f'/runs/{source_run}/viewer'}
            break
        job = reused or start(physical)
        if reused is None:
            status(job, job['state'], physical_model_id=identity, preview_url=f"/runs/{job['id']}/viewer")
        target = ROOT / 'local/inputs/physical-data' / (key(identity) + '.json')
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'run_id': job['id']}), encoding='utf-8')
        temporary.replace(target)
        return job


def require_linux():
    if sys.platform != 'linux':
        raise RuntimeError('仿真服务需要 Linux，请参阅 viz/README.md。')


def template(name):
    if name not in {"starlink_550", "sharing_demo", "fault_recovery"}:
        raise ValueError("unknown example")
    path = Path(__file__).with_name("templates") / (name + ".json")
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value.get("elements"),str):
        value["elements"] = json.loads((path.parent / value["elements"]).read_text(encoding="utf-8"))
    return value


def prepare_input(value):
    from copy import deepcopy
    return deepcopy(value)


def replay_root(output):
    return output / "network" if (output / "network").is_dir() else output


def result_index(output):
    folder = replay_root(output)
    current = folder / "network_result.json"
    return current if current.exists() else folder / "network_replay.json"

def has_replay_frames(folder):
    return ((folder / 'state/index.json').is_file()
            or (folder / 'frames/frame-000000.json.gz').is_file())


def saved_run_input(run_id):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", run_id):
        raise ValueError("invalid run ID")
    path = ROOT / "local/inputs/network" / run_id / "manifest.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if "workload" in value:
        value["workload"]["bundle_dir"] = str(
            (path.parent / value["workload"]["bundle_dir"]).resolve()
        )
    return value


def interactive_run_input(run_id):
    """New execution from a saved input; completed business logs stay immutable."""
    value = saved_run_input(run_id)
    if value.get('schema_version') in (1, 'orbitserve.native_cells.v1'):
        return value
    if 'physical_network' not in value:
        raise ValueError('在线交互需要包含 Workload 和 Serving 的完整实验配置')
    folder = replay_root(ROOT / 'local/results/network' / run_id)
    if (folder / 'physical_dataset.json').is_file():
        from astermind.builder.physical.physical_dataset import validate_dataset_input
        validate_dataset_input(str(folder), value['physical_network'], ROOT)
        value['physical_dataset'] = str(folder)
    return value


def save_input(value):
    from datetime import datetime
    import uuid
    from astermind.builder.physical.network_inputs import materialize_network_input
    if 'physical_network' in value or 'cells' in value:
        raise ValueError('计算场景请使用 V1 场景启动入口')
    concrete = materialize_network_input(value, ROOT)
    identity = 'physical-' + datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    folder = ROOT / 'local/inputs/network' / identity
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / 'manifest.json'
    path.write_text(json.dumps(concrete, ensure_ascii=False, indent=2))
    return identity, path


def command_for(path, output):
    return [sys.executable, '-B', str(Path(__file__).with_name('worker.py')), str(path), '--out', str(output)]


def status(job, state, **extra):
    from tasks import write_status
    with LOCK:
        write_status(job, state, **extra)


def run_status(run_id):
    from tasks import processes, same_process
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', run_id):
        raise ValueError('invalid run ID')
    output = ROOT / 'local/results/network' / run_id
    if not output.is_dir():
        raise FileNotFoundError('实验不存在或已删除')
    if run_id in JOBS:
        job = dict(JOBS[run_id])
    elif (output / 'status.json').is_file():
        try:
            job = json.loads((output / 'status.json').read_text(encoding='utf-8'))
            if not isinstance(job,dict) or not isinstance(job.get('state'),str):
                raise ValueError('invalid task status')
        except ValueError:
            job = {'id':run_id,'state':'unknown','error':'状态记录损坏；可查看日志或删除此任务。'}
        from live import SESSIONS
        if job['state'] == 'running' and run_id not in SESSIONS:
            job.update(state='interrupted', error='控制台重启前的任务未记录完成，请检查日志；已落盘文件仍保留。')
    elif result_index(output).is_file():
        job = {'id': run_id, 'state': 'completed'}
    else:
        raise FileNotFoundError('实验不存在')
    live = processes(ROOT, run_id)
    if job['state']=='interrupted' and job.get('owner_pid') and job.get('owner_stamp') and same_process(job['owner_pid'],job['owner_stamp']):
        job.update(state='running',error=None)
    if live:
        job.update(state='running', error=None)
    if (output / 'stop_requested.json').is_file():
        job.update(state='stopping' if live else 'cancelled', error=None)
    job['can_stop'] = bool(live)
    job['can_rerun'] = job['state'] not in {'running', 'stopping'}
    job['can_delete'] = job['state'] not in {'running', 'stopping'} and not live
    job['can_stop_and_delete'] = bool(live)
    if job.get('interactive'):
        from live import SESSIONS
        session = SESSIONS.get(run_id)
        state = session.payload['state'] if session else None
        active = state in {'preparing', 'paused', 'advancing', 'finalizing', 'stopping'}
        if state == 'stopping' and job['state'] == 'running':
            job['state'] = 'stopping'
        job.update(can_stop=active and job['state'] not in {'completed','stopping'}, can_rerun=not active, can_delete=not active,
                   can_stop_and_delete=False)
    if job.get('native_preset'):
        job['can_rerun'] = False  # The native preset is restarted from its dedicated entry.
    log = output / 'execution.log'
    if log.is_file():
        with log.open('rb') as stream:
            offset = max(0, log.stat().st_size - 12000)
            stream.seek(offset)
            if offset:
                stream.readline()  # The byte tail can begin inside a JSON string or UTF-8 character.
            job['log_tail'] = stream.read().decode('utf-8', errors='replace')
            job['log_tail_truncated'] = bool(offset)
    job['result_url'] = f'/runs/{run_id}/result'
    return job


def task_catalog():
    rows = []
    for output in sorted((ROOT/'local/results/network').glob('*'), key=lambda p:p.stat().st_mtime, reverse=True):
        if not output.is_dir() or not ((output/'status.json').is_file() or result_index(output).is_file()):
            continue
        try:
            job = run_status(output.name)
            metadata = output/'run_metadata.json'
            title = json.loads(metadata.read_text(encoding='utf-8')).get('title') if metadata.is_file() else None
            rows.append(dict(id=output.name, title=title or output.name, state=job['state']))
        except (ValueError, OSError):
            rows.append(dict(id=output.name, title=output.name, state='unknown'))
    return rows


def comparison_status(item, **changes):
    item.update(changes)
    folder = ROOT / 'local/results/comparisons' / item['id']
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'status.json').write_text(json.dumps(item, ensure_ascii=False), encoding='utf-8')


def start(value, comparison=False):
    with LAUNCH_LOCK:
        if not comparison and any(c['state']=='running' for c in COMPARISONS.values()):
            raise ValueError('两模式对照运行中，请等待完成')
        return _start(value)


def start_comparison(value):
    raise ValueError('V1.0 不提供旧 queue / SimGrid 双引擎对照；请分别运行 V1 场景并查看真实结果。')
    from copy import deepcopy
    require_linux()
    with LAUNCH_LOCK:
        if any(c['state']=='running' for c in COMPARISONS.values()) or any(j['state']=='running' for j in JOBS.values()):
            raise ValueError('已有实验运行中，请等待完成')
        run_id,path=save_input(value)
        prepared=saved_run_input(run_id)
        if 'physical_network' not in prepared:raise ValueError('对照需要完整系统运行')
        item={'id':run_id,'state':'running','phase':'queue'}
        COMPARISONS[run_id]=item
        comparison_status(item)
    def worker():
        try:
            runs={}
            for mode in ('queue','simgrid'):
                payload=deepcopy(prepared);payload['physical_network']['network']['transport_model']=mode
                comparison_status(item, phase=mode)
                job=start(payload,comparison=True);runs[mode]=job['output']
                comparison_status(item, active_run_id=job['id'])
                while job['state']=='running':time.sleep(.25)
                if job['state']!='completed':raise RuntimeError(job.get('error','对照运行失败'))
            from comparison import write_comparison
            evidence=write_comparison(runs['queue'],runs['simgrid'])
            comparison_status(item, state='completed', result_url='/runs/'+evidence['simgrid']['run_id']+'/result')
        except Exception as error:comparison_status(item, state='failed',error=str(error))
    threading.Thread(target=worker,daemon=True).start()
    return item


def _start(value):
    require_linux()
    run_id, path = save_input(value)
    output = ROOT / "local/results/network" / run_id
    output.mkdir(parents=True, exist_ok=False)
    job = dict(id=run_id, state="running", input=str(path.relative_to(ROOT)), output=str(output),
               command=command_for(path, output))
    if Path('/proc/self/stat').is_file():
        job.update(owner_pid=os.getpid(), owner_stamp=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19])
    JOBS[run_id] = job
    status(job, "running")

    def worker():
        try:
            with (output / "execution.log").open("w", encoding="utf-8") as log:
                env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
                process = subprocess.Popen(job["command"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                code = process.wait()
            if (output/'stop_requested.json').is_file():
                status(job, 'cancelled')
            elif code:
                status(job, "failed", error=(output / "execution.log").read_text(encoding="utf-8")[-6000:])
            else:
                export_runtime_views(output)
                replay = json.loads(result_index(output).read_text(encoding="utf-8"))
                write_network_viewer(replay, output / "network_replay.html", runtime_path=output/"runtime_viewer.json")
                status(job, "completed", checks=replay["checks"], viewer=f"/runs/{run_id}/viewer")
        except Exception as error:
            status(job, "failed", error=str(error))
    threading.Thread(target=worker, daemon=True).start()
    return job


class Handler(BaseHTTPRequestHandler):
    def respond_file(self, path, kind='application/json', attachment=False):
        """Send large immutable artifacts without allocating their entire contents."""
        with Path(path).open('rb') as stream:
            self.send_response(200)
            self.send_header('Content-Type', kind if kind in ('image/png', 'application/pdf') else kind + '; charset=utf-8')
            self.send_header('Content-Length',str(Path(path).stat().st_size))
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-store')
            if attachment:self.send_header('Content-Disposition','attachment; filename="'+Path(path).name+'"')
            self.end_headers()
            try:
                while block:=stream.read(512*1024):self.wfile.write(block)
            except (BrokenPipeError,ConnectionResetError):pass

    def allowed_authority(self):
        host = self.headers.get("Host", "").lower()
        allowed = getattr(self.server, "allowed_hosts", {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        })
        return host if host in allowed else None

    def respond(self, payload, kind="application/json", code=200, encoding=None):
        content = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        if kind == 'text/html' and not encoding:
            from workspace import decorate_document
            content = decorate_document(content, urlparse(self.path).path)
            if parse_qs(urlparse(self.path).query).get('embedded') == ['1']:
                content = content.replace(b'</head>', b'<link rel="stylesheet" href="/assets/analysis_embed.css"><script src="/assets/analysis_embed.js" defer></script></head>', 1)
            navigation = b'<script src="/assets/navigation.js" defer></script>'
            content = content.replace(b'</html>', navigation + b'</html>', 1) if b'</html>' in content else content + navigation
        if kind == 'application/json' and not encoding and len(content) >= 64 * 1024:
            for entry in self.headers.get('Accept-Encoding', '').split(','):
                parts = [part.strip().lower() for part in entry.split(';')]
                if parts[0] != 'gzip':
                    continue
                try:
                    quality = next((float(part.split('=', 1)[1]) for part in parts[1:] if part.startswith('q=')), 1.)
                except ValueError:
                    quality = 0
                if 0 < quality <= 1:
                    import gzip
                    content = gzip.compress(content, compresslevel=1)
                    encoding = 'gzip'
                break
        self.send_response(code)
        self.send_header("Content-Type", kind + "; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        if encoding:
            self.send_header("Content-Encoding", encoding)
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        try:
            self.end_headers()
            self.wfile.write(content)
        except (BrokenPipeError, ConnectionResetError):
            # The viewer may cancel an obsolete poll or leave a replay page.
            # A closed socket cannot receive a second 400 response.
            return

    def do_GET(self):
        if not self.allowed_authority():
            return self.respond({"error": "Host not allowed"}, code=403)
        path = urlparse(self.path).path
        try:
            if path == '/assets/AsterMind-V1-Tech-Report.pdf':
                name = 'AsterMind-V1-Tech-Report.pdf'
                report = REPOSITORY / 'doc' / name
                if not report.is_file():
                    report = Path(sysconfig.get_path('data')) / 'share' / 'astermind' / 'doc' / name
                return self.respond_file(report, 'application/pdf')
            if path in HOMEPAGE_FILES:
                name, kind = HOMEPAGE_FILES[path]
                return self.respond_file(HOMEPAGE / name, kind)
            if path in {'/zh', '/zh/', '/zh/index.html', '/en', '/platform/'}:
                canonical = {'/en': '/en/', '/platform/': '/platform'}.get(path, '/')
                query = urlparse(self.path).query
                if path.startswith('/zh'):
                    query = (query + '&' if query else '') + 'lang=zh-CN'
                self.send_response(302)
                self.send_header('Location', canonical + ('?' + query if query else ''))
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if path == "/platform":
                return self.respond(Path(__file__).with_name("index.html").read_bytes(), "text/html")
            if path in {'/workspace', '/configuration', '/results'}:
                page = 'workbench' if path == '/workspace' else path[1:]
                return self.respond(Path(__file__).with_name(page+'.html').read_bytes(), 'text/html')
            if path == '/native-cells':
                self.send_response(302)
                self.send_header('Location', '/native-cells/')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if path.startswith('/native-cells/'):
                import gzip
                name = path.removeprefix('/native-cells/') or 'index.html'
                if not re.fullmatch(r'[a-zA-Z0-9_-]+\.(?:html|css|js|json)', name):
                    return self.respond({'error': 'not found'}, code=404)
                directory = (ROOT / 'local/results/native-cells-viewer').resolve()
                file = (directory / name).resolve()
                if not file.is_relative_to(directory) or not file.is_file():
                    return self.respond({'error': 'export native replay first'}, code=404)
                kind = {'html': 'text/html', 'js': 'text/javascript', 'css': 'text/css', 'json': 'application/json'}[file.suffix[1:]]
                if file.suffix == '.json':
                    return self.respond(gzip.compress(file.read_bytes(), compresslevel=1), kind, encoding='gzip')
                return self.respond(file.read_bytes(), kind)
            if path in {'/collectives', '/serving'}:
                self.send_response(302)
                self.send_header('Location', '/configuration')
                self.send_header('Content-Length', '0')
                self.end_headers()
                return
            if path.startswith('/serving/'):
                import serving_runs
                query=parse_qs(urlparse(self.path).query)
                users=query.get('users',['single'])[0];parallel=query.get('parallel',['tp_pp'])[0];backend=query.get('backend',['flowsim'])[0]
                if path=='/serving/status':return self.respond(serving_runs.status(users,parallel,backend))
                if path=='/serving/compare':return self.respond(serving_runs.compare(users,parallel))
                if path=='/serving/physical':return self.respond(serving_runs.index(parallel))
                if path=='/serving/frame':return self.respond(serving_runs.frame(parallel,int(query.get('frame',['0'])[0])),encoding='gzip')
                if path=='/serving/data':
                    import gzip
                    return self.respond(gzip.compress(json.dumps(serving_runs.result(users,parallel,backend)).encode(),compresslevel=1),encoding='gzip')
            if path == '/collectives/status':
                import collective_runs
                query=parse_qs(urlparse(self.path).query)
                return self.respond(collective_runs.status(query.get('mode',['ring4'])[0], query.get('backend',['flowsim'])[0]))
            if path.startswith('/collectives/data/'):
                import collective_runs
                name=path.rsplit('/',1)[1]
                match=re.fullmatch(r'(ring4|multi)-(flowsim|ns3)[.]json',name)
                if not match:raise ValueError('unknown collective dataset')
                return self.respond(collective_runs.data(*match.groups()))
            if path in {'/physical', '/scene'}:
                return self.respond(Path(__file__).with_name(path[1:]+'.html').read_bytes(), 'text/html')
            if path == '/live' or (path.startswith('/live/') and len(path.strip('/').split('/')) == 2):
                return self.respond(Path(__file__).with_name('live.html').read_bytes(), 'text/html')
            if path.startswith('/live/') and path.endswith('/state'):
                from live import get
                query = parse_qs(urlparse(self.path).query)
                after = int(query['after'][0]) if 'after' in query else None
                return self.respond(get(path.strip('/').split('/')[1]).state(after))
            if path == '/radio-profiles':
                from astermind.engine.setup.radio_capacity import STARLINK_RADIO_PROFILES
                return self.respond(STARLINK_RADIO_PROFILES)
            if path == '/islsim-defaults':
                from astermind.engine import resolve_islsim_config
                parameters, identity = resolve_islsim_config()
                return self.respond({'parameters': parameters, 'source_sha256': identity})
            if path == '/tle-samples':
                from tle_import import samples
                return self.respond(samples(ROOT))
            if path.startswith('/tle-samples/'):
                from tle_import import sample
                return self.respond(sample(ROOT,path.rsplit('/',1)[-1]))
            if path.startswith('/tle-files/'):
                from tle_import import sample_paths
                name=path.rsplit('/',1)[-1]
                if not name.endswith('.tle'):
                    raise ValueError('TLE filename required')
                _,_,file=sample_paths(ROOT,name[:-4])
                return self.respond(file.read_bytes(),'text/plain')
            if path == '/physical-catalog':
                from studio import physical_catalog
                return self.respond(physical_catalog(ROOT))
            if path == '/scene-catalog':
                from studio import catalog
                return self.respond(catalog(ROOT, 'scenes'))
            if path == '/physical-datasets':
                from studio import catalog, physical_data
                rows = []
                for item in catalog(ROOT, 'physical'):
                    data = physical_data(ROOT, item['id'])
                    if data['ready']:
                        rows.append({**item, **data})
                return self.respond(rows)
            if path == '/native-scene-catalog':
                from native_environment import enable_device_native
                from astermind.builder.catalog import cell_members
                from compute_options import compute_catalog
                result = compute_catalog(enable_device_native())
                identity = parse_qs(urlparse(self.path).query).get('physical_model_id', [None])[0]
                if identity:
                    from studio import physical_data
                    data = physical_data(ROOT, identity)
                    if not data['ready']:
                        raise ValueError('物理数据尚未就绪')
                    result['cell_satellites'] = cell_members(data['physical_dataset'])
                    result['cell_members'] = {identity: len(members) for identity, members in result['cell_satellites'].items()}
                return self.respond(result)
            if path.startswith('/physical-data/'):
                from studio import physical_data
                return self.respond(physical_data(ROOT, path.rsplit('/', 1)[-1]))
            if path.startswith('/documents/'):
                from studio import load
                pieces=path.strip('/').split('/')
                if len(pieces)!=3:raise ValueError('invalid document URL')
                return self.respond(load(ROOT,pieces[1],pieces[2]))
            if path == '/experiments':
                return self.respond(Path(__file__).with_name('catalog.html').read_bytes(), 'text/html')
            if path == '/current-experiment':
                from current_experiment import current
                return self.respond(current(ROOT))
            if path.startswith("/assets/"):
                name=path.removeprefix("/assets/")
                if "/" in name or "\\" in name or name.startswith("."):
                    return self.respond({"error":"not found"},code=404)
                asset=Path(__file__).with_name("assets")/name
                kind="text/plain" if name.endswith(".tle") else "image/jpeg" if name.endswith(".jpg") else 'text/css' if name.endswith('.css') else "text/javascript"
                return self.respond(asset.read_bytes(),kind)
            if path=="/run-catalog":
                rows=[]
                for output in sorted((ROOT/"local/results/network").iterdir(),key=lambda p:p.stat().st_mtime,reverse=True):
                    if not result_index(output).is_file():continue
                    metadata=output/"run_metadata.json"
                    title=json.loads(metadata.read_text()).get("title") if metadata.is_file() else None
                    if not title and result_index(output).stat().st_size>10_000_000:
                        title="旧版内嵌大文件结果（请重跑） "+output.name
                    if not title:
                        replay=json.loads(result_index(output).read_text())
                        ns=replay.get("network",{})
                        shells=ns.get("communication_shells",[])
                        name=" / ".join(str(s["shell_id"])+" "+s["topology"] for s in shells) or ns.get("communication_topology","物理网络")
                        title=name+" "+str(replay.get("time",{}).get("duration_s","?"))+" s 间隔 "+str(replay.get("time",{}).get("step_s","?"))+" s"
                    rows.append({"id":output.name,"title":title})
                return self.respond(rows)
            if path == '/task-catalog':
                return self.respond(task_catalog())
            if path=="/runs":
                runs=[p.name for p in sorted((ROOT/ "local/results/network").iterdir(),key=lambda p:p.stat().st_mtime,reverse=True) if result_index(p).is_file()]
                return self.respond(runs)
            if path.startswith('/comparisons/'):
                pieces=path.strip('/').split('/')
                if len(pieces)!=3 or not re.fullmatch(r'[a-zA-Z0-9_-]+',pieces[1]):
                    return self.respond({'error':'not found'},code=404)
                if pieces[2]=='progress':
                    return self.respond(Path(__file__).with_name('progress.html').read_bytes(),'text/html')
                if pieces[2]!='status':return self.respond({'error':'not found'},code=404)
                key=pieces[1]
                item=dict(COMPARISONS.get(key,{}))
                saved=ROOT/'local/results/comparisons'/key/'status.json'
                if not item and saved.is_file():
                    item=json.loads(saved.read_text(encoding='utf-8'))
                    if item['state']=='running':item.update(state='interrupted',error='控制台已重启，对照任务中断。')
                if item.get('active_run_id'):
                    active=run_status(item['active_run_id'])
                    item['log_tail']=active.get('log_tail','')
                return self.respond(item or {'state':'failed','error':'对照任务不存在'})
            if path.startswith("/templates/"):
                return self.respond(template(path.rsplit("/", 1)[-1]))
            parts = path.strip("/").split("/")
            if len(parts)==4 and parts[0]=="runs" and re.fullmatch(r"[a-zA-Z0-9_-]+",parts[1]) and parts[2]=="runtime-data":
                if not re.fullmatch(r"(index|requests-[0-9]{5})[.]json|(?:topology|[a-f0-9]{64}|(?:events|context|topology)-[0-9]{5})[.]json[.]gz",parts[3]):
                    return self.respond({"error":"not found"},code=404)
                artifact=ROOT/"local/results/network"/parts[1]/"runtime_playback"/parts[3]
                return self.respond(artifact.read_bytes(),encoding="gzip" if parts[3].endswith(".gz") else None)
            if len(parts)==3 and parts[0]=="runs" and re.fullmatch(r"[a-zA-Z0-9_-]+",parts[1]) and parts[2]=="runtime":
                output=ROOT/"local/results/network"/parts[1]
                if not (output/"runtime_playback/index.json").is_file():
                    return self.respond({"error":"本结果未导出事件回放数据。请使用当前程序重新运行完整系统实验。"},code=404)
                return self.respond(Path(__file__).with_name("runtime.html").read_bytes(),"text/html")
            if len(parts)==4 and parts[0]=="runs" and re.fullmatch(r"[a-zA-Z0-9_-]+",parts[1]) and parts[2]=="frames" and re.fullmatch(r"frame-[0-9]{6}.json.gz",parts[3]):
                folder=replay_root(ROOT/ "local/results/network"/parts[1])
                artifact=folder/"frames"/parts[3]
                if artifact.is_file() and not (folder/'state/index.json').is_file():
                    content=artifact.read_bytes()
                else:
                    from physical_frames import frame_bytes
                    content=frame_bytes(folder,int(parts[3][6:12]))
                return self.respond(content, encoding="gzip")
            if (len(parts) == 4 and parts[0] == 'runs' and parts[2] == 'motion'
                    and re.fullmatch(r'[a-zA-Z0-9_-]+', parts[1])
                    and re.fullmatch(r'frame-\d{6}\.json\.gz', parts[3])):
                from motion_frames import motion_bytes
                folder = replay_root(ROOT/'local/results/network'/parts[1])
                return self.respond(motion_bytes(str(folder), int(parts[3][6:12])), encoding='gzip')
            if len(parts)==4 and parts[0]=='runs' and parts[2]=='native-replay' and re.fullmatch(r'[a-zA-Z0-9_-]+',parts[1]) and re.fullmatch(r'[a-zA-Z0-9_.-]+\.(?:html|js|css|json)',parts[3]):
                artifact=ROOT/'local/results/network'/parts[1]/'native-replay'/parts[3]
                from replay_pages import LIMIT, catalog as page_catalog, batch as batch_page
                source=artifact.parent/'native_result.json'
                if source.is_file() and source.stat().st_size>LIMIT:
                    if parts[3]=='catalog.json':return self.respond(page_catalog(source))
                    match=re.fullmatch(r'batch-(\d+)\.json',parts[3])
                    if match:return self.respond(batch_page(source,int(match[1])))
                if not artifact.is_file():return self.respond({'error':'回放文件不存在'},code=404)
                # Update the analysis UI while keeping each saved experiment's data immutable.
                if parts[3] in {'index.html','replay.js','replay.css'}:
                    artifact=Path(__file__).with_name('native_replay.html' if parts[3]=='index.html' else parts[3])
                kind={'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json'}[artifact.suffix]
                if artifact.stat().st_size>LIMIT:return self.respond_file(artifact,kind)
                return self.respond(artifact.read_bytes(),kind)
            if len(parts)==3 and parts[0]=="runs" and re.fullmatch(r"[a-zA-Z0-9_-]+",parts[1]):
                output=ROOT/ "local/results/network"/parts[1]
                if parts[2] in {'progress','result'}:
                    page = 'results' if parts[2] == 'result' else 'progress'
                    return self.respond(Path(__file__).with_name(page+'.html').read_bytes(),'text/html')
                if parts[2]=='analysis':
                    from result_analysis import analysis
                    return self.respond(analysis(ROOT,parts[1]))
                if parts[2]=='flow-paths':
                    from flow_archive import search_paths
                    query=parse_qs(urlparse(self.path).query)
                    return self.respond(search_paths(output,query.get('query',[''])[0],int(query.get('through',['9223372036854775807'])[0])))
                if parts[2]=='scene-events':
                    from result_analysis import scene_events
                    return self.respond(scene_events(ROOT,parts[1]))
                if parts[2]=='scene-replay':
                    source=output/'native_result.json'
                    if source.is_file() and source.stat().st_size>64*1024*1024 and (output/'live_state.json').is_file():
                        self.send_response(302)
                        self.send_header('Location',f'/live/{parts[1]}?observe=1&embedded=1&recorded=1')
                        self.send_header('Content-Length','0')
                        self.end_headers()
                        return
                    return self.respond(Path(__file__).with_name('scene_replay.html').read_bytes(),'text/html')
                if parts[2]=='native-summary':
                    return self.respond(json.loads((output/'native_summary.json').read_text(encoding='utf-8')))
                if parts[2]=='native-result':
                    return self.respond_file(output/'native_result.json',attachment=True)
                if parts[2]=='status':
                    return self.respond(run_status(parts[1]))
                if parts[2]=="input":
                    return self.respond(saved_run_input(parts[1]))
                if parts[2]=='cell-network-export':
                    return self.respond((replay_root(output)/'astermind_network.jsonl.gz').read_bytes(), 'application/x-ndjson', encoding='gzip')
                if parts[2]=='comparison':
                    source=output/'comparison.json'
                    return self.respond(json.loads(source.read_text()) if source.is_file() else {'available':False})
                if parts[2]=='fault-reference':
                    source=output/'fault_reference.json'
                    return self.respond(json.loads(source.read_text()) if source.is_file() else {'available':False})
                if parts[2]=="environment-index":
                    artifact=result_index(output)
                    if not artifact.is_file():return self.respond({"error":"environment index missing"},code=404)
                    if artifact.stat().st_size>10_000_000 and not has_replay_frames(replay_root(output)):return self.respond({"error":"legacy environment must be regenerated"},code=400)
                    from physical_frames import environment_index
                    return self.respond(environment_index(replay_root(output)))
                if parts[2]=="viewer" and result_index(output).is_file():
                    if result_index(output).stat().st_size > 10_000_000 and not has_replay_frames(replay_root(output)):
                        raise ValueError("旧版结果为整小时大文件。请选择新版按帧实验，避免浏览器内存过高。")
                    from physical_frames import environment_index
                    replay=environment_index(replay_root(output))
                    write_network_viewer(replay,output/"network_replay.html",runtime_path=output/"runtime_viewer.json")
                    return self.respond((output/"network_replay.html").read_bytes(),"text/html")
            self.respond({"error": "not found"}, code=404)
        except Exception as error:
            self.respond({"error": str(error)}, code=400)

    def do_POST(self):
        try:
            host = self.allowed_authority()
            if not host:
                return self.respond({"error":"Host not allowed"}, code=403)
            origin = self.headers.get("Origin")
            if origin and origin.lower() != f"http://{host}":
                return self.respond({"error":"same origin required"}, code=403)
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 10_000_000 or not self.headers.get("Content-Type", "").startswith("application/json"):
                raise ValueError("expected JSON input up to 10 MB")
            from scene_validation import strict_json
            payload = strict_json(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError('请求必须是 JSON 对象')
            parts = self.path.strip('/').split('/')
            if self.path == '/collectives/run':
                import collective_runs
                return self.respond(collective_runs.start(payload.get('mode','ring4'), payload.get('backend','flowsim')), code=202)
            if self.path == '/serving/run':
                import serving_runs
                return self.respond(serving_runs.start(payload.get('users','single'),payload.get('parallel','tp_pp'),payload.get('backend','flowsim')),code=202)
            if self.path == '/live/start':
                from live import start as start_live
                from studio import scene_input
                if payload.get('native_preset'):
                    if set(payload) != {'native_preset'}:
                        raise ValueError('原生预设使用已固定的模型、FlowSim 与 TP/PP 参数')
                    from native_live import start as start_native
                    job = start_native(ROOT, payload['native_preset'])
                    from current_experiment import select
                    select(ROOT, job['id'])
                    return self.respond(job, code=202)
                if sum(bool(payload.get(key)) for key in ('demo', 'scene_id', 'run_id')) != 1:
                    raise ValueError('请选择一个交互示例、已保存场景或原运行配置')
                if payload.get('demo'):
                    from demo_v1 import configuration
                    value = configuration(ROOT,payload)
                else:
                    if {'random_faults', 'transport_model'}.intersection(payload):
                        raise ValueError('已保存场景的故障与传输参数请在第二步修改后保存，运行入口不接受覆盖参数')
                    value = interactive_run_input(payload['run_id']) if payload.get('run_id') else scene_input(ROOT, payload['scene_id'])
                if value.get('schema_version') in (1, 'orbitserve.native_cells.v1'):
                    from native_scene_live import start as start_native_scene
                    return self.respond(start_native_scene(ROOT, value, scene_id=payload.get('scene_id')), code=202)
                from studio import validate_model
                validate_model(value['physical_network'], ROOT)
                job = start_live(ROOT, value)
                from current_experiment import select
                select(ROOT, job['id'])
                return self.respond(job, code=202)
            if len(parts) == 3 and parts[0] == 'live':
                from live import get
                session = get(parts[1])
                if parts[2] == 'advance':
                    return self.respond(session.advance(payload.get('until_s'), payload.get('mode')), code=202)
                if parts[2] == 'pause':
                    return self.respond(session.pause(), code=202)
                if parts[2] == 'command':
                    return self.respond(session.command(payload))
                if parts[2] == 'stop':
                    return self.respond(session.stop(), code=202)
            if len(parts)==3 and parts[0]=='runs' and parts[2] in {'stop','rerun','delete'}:
                from tasks import stop
                run_id=parts[1]
                if parts[2]=='delete':
                    from task_cleanup import delete_files, deletion_paths
                    with LAUNCH_LOCK:
                        current=run_status(run_id)
                        if not current['can_delete']:
                            if payload.get('stop_running') is not True or not current['can_stop_and_delete']:
                                raise ValueError('任务仍在运行；请选择停止并删除，或等待任务结束')
                            # Check dependencies before stopping a live job.
                            deletion_paths(ROOT,run_id)
                            stop(ROOT,run_id)
                        deleted=delete_files(ROOT,run_id)
                        with LOCK:
                            JOBS.pop(run_id,None)
                        from live import LOCK as session_lock, SESSIONS
                        with session_lock:
                            SESSIONS.pop(run_id,None)
                    return self.respond(deleted)
                current=run_status(run_id)
                if parts[2]=='stop':
                    if current.get('interactive'):
                        from live import get
                        get(run_id).stop()
                    else:
                        stop(ROOT,run_id)
                    return self.respond(run_status(run_id))
                if not current['can_rerun']:
                    raise ValueError('请先停止当前任务，或等待完成，再重新运行')
                saved=saved_run_input(run_id)
                if saved.get('schema_version') in (1, 'orbitserve.native_cells.v1'):
                    from native_scene_live import start as start_native
                    job=start_native(ROOT,saved,automatic=True,scene_id=current.get('scene_id'))
                    return self.respond(run_status(job['id']),code=202)
                else:
                    job=start(saved)
                if current.get('physical_model_id'):
                    from studio import key
                    identity=key(current['physical_model_id'])
                    status(job,job['state'],physical_model_id=identity,preview_url=f"/runs/{job['id']}/viewer")
                    reference=ROOT/'local/inputs/physical-data'/(identity+'.json')
                    reference.parent.mkdir(parents=True,exist_ok=True)
                    temporary=reference.with_suffix('.json.tmp')
                    temporary.write_text(json.dumps({'run_id':job['id']}),encoding='utf-8')
                    temporary.replace(reference)
                elif current.get('scene_id'):
                    status(job,job['state'],scene_id=current['scene_id'],runtime_url=f"/runs/{job['id']}/runtime")
                return self.respond(job,code=202)
            if self.path == '/inspect-tle':
                from tle_import import inspect
                return self.respond(inspect(payload.get('tle_text')))
            if self.path in {'/save-physical', '/save-scene'}:
                from studio import save
                return self.respond(save(ROOT,'physical' if self.path=='/save-physical' else 'scenes',payload))
            if self.path == '/validate-scene':
                from scene_validation import validate_scene
                if 'experiment_json' in payload:
                    payload['experiment'] = strict_json(payload.pop('experiment_json'))
                    context = payload.pop('resource_context', None)
                    if context is not None:
                        allowed = {'physical', 'physical_link_bindings', 'ground_owners', 'satellite_roles'}
                        if not isinstance(context, dict) or set(context) != allowed or payload['experiment'].get('schema_version') not in (1, 'orbitserve.native_cells.v1'):
                            raise ValueError('资源上下文必须完整且只能用于原生部署草稿')
                        payload['experiment'] = {**context, **payload['experiment']}
                document, _ = validate_scene(ROOT, payload)
                return self.respond(dict(valid=True, document=document))
            if self.path == '/native-scene-config':
                from native_scene import generate
                return self.respond(generate(ROOT, payload))
            if self.path == '/native-scene-requests':
                from native_environment import enable_device_native
                enable_device_native()
                from astermind.builder.request_batch import prepare_batch
                return self.respond(prepare_batch(payload))
            if self.path == '/edit-scene-topology':
                from native_environment import enable_device_native
                enable_device_native()
                from astermind.builder.topology_edit import edit_topology
                if set(payload) != {'experiment','edit'}:
                    raise ValueError('拓扑修改需要场景和操作')
                return self.respond(edit_topology(payload['experiment'],payload['edit']))
            if self.path=='/launch-scene':
                from studio import scene_input
                value = scene_input(ROOT,payload['id'])
                if value.get('schema_version') in (1, 'orbitserve.native_cells.v1'):
                    from native_scene_live import start as start_native_scene
                    return self.respond(start_native_scene(ROOT, value, automatic=True, scene_id=payload['id']), code=202)
                job=start(value)
                status(job,job['state'],scene_id=payload['id'],runtime_url=f"/runs/{job['id']}/runtime")
                from current_experiment import select
                select(ROOT, job['id'])
                return self.respond(job,code=202)
            if self.path in {'/launch-physical', '/prepare-physical'}:
                return self.respond(prepare_physical(payload['id'], independent=payload.get('independent') is True), code=202)
            if self.path == '/physical-route':
                from studio import key
                from physical_preview import probe_routes
                folder = replay_root(ROOT / 'local/results/network' / key(payload['run_id']))
                return self.respond(probe_routes(folder, payload))
            if self.path=='/compare':
                return self.respond(start_comparison(payload),code=202)
            if self.path == "/run":
                return self.respond(start(payload), code=202)
            if self.path == "/save":
                run_id, path = save_input(payload)
                return self.respond({"id":run_id, "input":str(path.relative_to(ROOT)),
                    "command":command_for(path,ROOT/ "local/results/network"/run_id)})
            self.respond({"error":"not found"}, code=404)
        except Exception as error:
            self.respond({"error":str(error)}, code=400)


if __name__ == "__main__":
    require_linux()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766, help="HTTP port (default: 8766)")
    parser.add_argument("--host", default="127.0.0.1", help="Listening address; use 0.0.0.0 for network access")
    parser.add_argument("--allow-host", action="append", default=[], metavar="HOST:PORT",
                        help="Additional allowed HTTP authority, repeat for each remote address")
    parser.add_argument('--preserve-assets', action='store_true', help='保留已有输入和结果，跳过启动时的自动归档')
    args = parser.parse_args()
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as error:
        if error.errno == errno.EADDRINUSE:
            parser.exit(1, f"端口 {args.port} 已被占用。若已有 OrbitServe 服务，请直接访问 "
                        f"http://localhost:{args.port}/；需要重启时，先结束旧服务，再执行本命令。\n")
        raise
    server.allowed_hosts = {f"127.0.0.1:{server.server_port}", f"localhost:{server.server_port}",
                            *(name.lower() for name in args.allow_host)}
    if args.host != "0.0.0.0":
        server.allowed_hosts.add(f"{args.host.lower()}:{server.server_port}")
    print(f"Network console listening on {args.host}:{server.server_port}; allowed hosts: "
          + ", ".join(sorted(server.allowed_hosts)), flush=True)
    if not args.preserve_assets:
        from maintenance import reconcile
        try:
            cleanup = reconcile(ROOT,apply=True)
            if cleanup['entries']:
                print('Archived obsolete local assets: '+json.dumps(cleanup['counts']),flush=True)
        except (ValueError,OSError) as error:
            print('Local asset cleanup deferred: '+str(error),flush=True)
    print(f"Open http://127.0.0.1:{server.server_port}/", flush=True)
    from live_communication import watch as watch_communications
    threading.Thread(target=watch_communications, args=(ROOT,), daemon=True).start()
    server.serve_forever()
