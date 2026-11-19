"""Prepare the V1 backend for the frontend and its workers."""
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from paths import VIZ, REPOSITORY, RUNTIME, initialize

_backend = None


@contextmanager
def snapshot_lock(root):
    """Serialize preparation across frontend/worker processes on Linux."""
    import fcntl
    root.parent.mkdir(parents=True, exist_ok=True)
    with root.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def extract_source(raw, root):
    """Validate each archive directory once, including existing symlinks."""
    checked = set()
    resolved_root = root.resolve()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for entry in archive.infolist():
            target = root / entry.filename
            parent = target.parent
            if parent not in checked:
                if not parent.resolve().is_relative_to(resolved_root):
                    raise ValueError('Invalid source archive path')
                checked.add(parent)
            if target.is_symlink() or '..' in Path(entry.filename).parts or Path(entry.filename).is_absolute():
                raise ValueError('Invalid source archive path')
            archive.extract(entry, root)


def source_identity():
    """Fingerprint local backend edits and additions without touching Git state."""
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip()
    changed = subprocess.check_output(['git', 'diff', '--name-only', '-z', 'HEAD', '--',
        'astermind', 'workloads', 'examples'], cwd=REPOSITORY).decode().split('\0')
    changed += subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z', '--',
        'astermind', 'workloads', 'examples'], cwd=REPOSITORY).decode().split('\0')
    overlays = {}
    for name in sorted(set(filter(None, changed))):
        path = REPOSITORY / name
        if path.is_symlink() or not path.resolve().is_relative_to(REPOSITORY.resolve()):
            raise ValueError('Backend snapshot requires regular repository files')
        content = path.read_bytes() if path.is_file() else None
        if content is not None and path.suffix in {'.py', '.json', '.md', '.toml', '.yaml', '.yml', '.txt'}:
            content = content.replace(b'\r\n', b'\n')
        overlays[name] = content
    hashes = {name: hashlib.sha256(data).hexdigest() if data is not None else None
              for name, data in overlays.items()}
    fingerprint = hashlib.sha256(json.dumps({'commit': head, 'files': hashes}, sort_keys=True).encode()).hexdigest()
    return dict(commit=head, source_fingerprint=fingerprint, backend_modified=bool(overlays),
                modified_files=hashes), overlays


def prepare_native_libraries(root):
    """Build the two pinned C/C++ dependencies only inside the viz snapshot."""
    upstream=root/'astermind/engine/compute/upstream'
    entries=[(root/'astermind/engine/comm/native_allocator.cc',root/'astermind/engine/build/libcomm_allocator.so',os.environ.get('CXX','g++'),['-fPIC','-std=c++17','-O2','-shared']),
             (upstream/'src/kernel/csrc/lru_sim.c',upstream/'build/lru_sim.so',os.environ.get('CC','cc'),['-O3','-shared','-fPIC'])]
    for source,target,compiler,flags in entries:
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        marker=target.with_suffix('.source.json')
        expected=dict(source_sha256=digest,compiler=compiler,flags=flags)
        if target.is_file() and marker.is_file() and json.loads(marker.read_text())==expected:continue
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_suffix('.'+str(os.getpid())+'.tmp')
        subprocess.run([compiler,*flags,str(source),'-o',str(temporary)],check=True,cwd=root)
        temporary.replace(target)
        marker.write_text(json.dumps(expected))


def activate():
    global _backend
    if _backend is not None:
        return _backend
    if sys.platform != 'linux':
        raise RuntimeError('The AsterMind frontend requires Linux. See viz/README.md.')
    initialize()
    sys.dont_write_bytecode = True
    if importlib.util.find_spec('pydantic') is None:
        raise RuntimeError('当前 Python 环境缺少 pydantic，请按 viz/README.md 安装项目依赖。')
    provenance, overlays = source_identity()
    head = provenance['commit']
    suffix = '-worktree-' + provenance['source_fingerprint'] if overlays else ''
    root = RUNTIME / 'sources' / (head + suffix)
    marker = root / '.frontend-source.json'
    with snapshot_lock(root):
        if not marker.is_file():
            root.mkdir(parents=True, exist_ok=True)
            raw = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip', head,
                                           'astermind', 'workloads', 'examples'], cwd=REPOSITORY)
            extract_source(raw, root)
            for name, content in overlays.items():
                target = root / name
                if content is None:
                    target.unlink(missing_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
            marker.write_text(json.dumps(provenance, indent=2))
        elif json.loads(marker.read_text()) != provenance:
            # Older clean snapshots predate the provenance fields.
            marker.write_text(json.dumps(provenance, indent=2))
    # Canonical HEAD bytes keep reference data intact without changing checkout.
    loaded = sys.modules.get('astermind')
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError('A different AsterMind backend was imported before frontend initialization')
    paths = [VIZ, root, REPOSITORY]
    for path in reversed(paths):
        if str(path) in sys.path:
            sys.path.remove(str(path))
        sys.path.insert(0, str(path))
    os.environ['PYTHONPATH'] = os.pathsep.join(str(p) for p in paths)
    os.environ['ORBITServe_MIDLAYER_ROOT'] = str(root / 'astermind/engine/compute/upstream')
    os.environ['ORBITSERVE_MIDLAYER_ROOT'] = os.environ['ORBITServe_MIDLAYER_ROOT']
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    # Read the existing local dependency locations; never modify these checkouts.
    settings = REPOSITORY / 'local/config/communication-modules.json'
    if settings.is_file():
        configured = json.loads(settings.read_text())
        for module in ('islsim', 'gslsim'):
            for field, suffix in (('root','ROOT'),('config','CONFIG'),('rf_config','RF_CONFIG'),('optical_config','OPTICAL_CONFIG')):
                value = configured.get(module, {}).get(field)
                if value:
                    if len(value)>2 and value[1]==':':
                        value='/mnt/'+value[0].lower()+'/'+value[3:].replace('\\','/')
                    os.environ.setdefault('ORBITSERVE_'+module.upper()+'_'+suffix, value)
    with snapshot_lock(root):
        prepare_native_libraries(root)
    from astermind.common.bootstrap import activate as bootstrap
    bootstrap()
    os.environ['PYTHONPATH']=os.pathsep.join(sys.path)
    _backend = root
    (RUNTIME / 'source.json').write_text(json.dumps({**provenance, 'root': str(root),
                                                   'python': sys.executable}, indent=2))
    return root


def enable_device_native():
    return activate() / 'astermind/engine/compute/upstream'


def enable_local_flowsim():
    """Legacy frontend hook name; initializes only V1 Runtime Comm."""
    activate()


def enable_native_inprocess():
    return activate()


def inference_runtime():
    activate()
    return sys.executable, dict(os.environ)


def identity():
    root = activate()
    provenance = json.loads((root / '.frontend-source.json').read_text())
    return {'backend': 'AsterMind V1.0', **provenance, 'source': str(root),
            'python': sys.executable, 'platform': sys.platform, 'old_backend': False}
