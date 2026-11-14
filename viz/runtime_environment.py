"""Run the current V1 backend from OS Python; never load the old Cell engine."""
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from paths import VIZ, REPOSITORY, RUNTIME, initialize

_backend = None


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
        raise RuntimeError('Start this frontend in WSL Linux with the OS Python environment.')
    initialize()
    sys.dont_write_bytecode = True
    candidates = [os.environ.get('ASTERMIND_FRONTEND_DEPS'), str(RUNTIME / 'python'),
                  '/tmp/orbitserve-native-scene-031-517',
                  str(REPOSITORY / 'local/validation/native-scene-deps'),
                  str(REPOSITORY / 'local/validation/astermind-demo/venv/lib/python3.12/site-packages')]
    dependencies = [Path(p) for p in candidates if p and Path(p).is_dir()]
    for path in reversed(dependencies):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if importlib.util.find_spec('pydantic') is None:
        raise RuntimeError('OS 缺少 pydantic；请将 V1 依赖安装到 viz/.runtime/python 或设置 ASTERMIND_FRONTEND_DEPS。')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPOSITORY, text=True).strip()
    root = RUNTIME / 'sources' / head
    marker = root / '.frontend-source.json'
    if not marker.is_file():
        root.mkdir(parents=True, exist_ok=True)
        raw = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'archive', '--format=zip', head,
                                       'astermind', 'workloads', 'examples'], cwd=REPOSITORY)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for entry in archive.infolist():
                target = (root / entry.filename).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError('Invalid source archive path')
                archive.extract(entry, root)
        marker.write_text(json.dumps({'commit': head, 'backend_modified': False}))
    # Canonical HEAD bytes keep reference data intact without changing checkout.
    loaded = sys.modules.get('astermind')
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError('A different AsterMind backend was imported before frontend initialization')
    paths = [VIZ, root, *dependencies, REPOSITORY]
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
    prepare_native_libraries(root)
    from astermind.common.bootstrap import activate as bootstrap
    bootstrap()
    _backend = root
    (RUNTIME / 'source.json').write_text(json.dumps({'commit': head, 'root': str(root),
                                                   'python': sys.executable, 'backend_modified': False}, indent=2))
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
    return {'backend': 'AsterMind V1.0', 'commit': root.name, 'source': str(root),
            'python': sys.executable, 'platform': sys.platform, 'old_backend': False}
