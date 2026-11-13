"""Build an sdist and its wheel; smoke-test the installed artifact outside source."""
import os
import argparse
from contextlib import nullcontext
from pathlib import Path, PurePosixPath
import subprocess
import shutil
import sys
import tarfile
from tempfile import TemporaryDirectory
import zipfile


SMOKE = r'''
import hashlib
from importlib import import_module
from importlib.resources import files
import json
from pathlib import Path
import pkgutil
import sys

target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
from astermind.common.bootstrap import activate
activate()
import astermind
assert Path(astermind.__file__).is_relative_to(target), astermind.__file__
modules = list(pkgutil.walk_packages(astermind.__path__, astermind.__name__ + '.'))
for module in modules:
    loaded = import_module(module.name)
    assert Path(loaded.__file__).is_relative_to(target), loaded.__file__
from astermind.builder.orbit.sgp4 import frames
for name, expected in (
    ('finals2000A.all', frames.PINNED_IERS_A_SHA256),
    ('ReadMe.finals2000A', frames.PINNED_IERS_README_SHA256),
    ('Leap_Second.dat', frames.PINNED_LEAP_SECONDS_SHA256),
):
    resource = files('astermind.builder.orbit').joinpath('data/iers/0.2026.5.11.1.8.52', name)
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == expected, name
for name in ('state_corpus.json', 'geometry_corpus.json'):
    json.loads(files('astermind.builder.orbit').joinpath('data/stk_validation', name).read_text())
from astermind.experiments.islsim_cli import main
output = Path.cwd() / 'islsim.json'
assert main(['--out', str(output)]) == 0
assert set(json.loads(output.read_text())) == {'tx', 'rx', 'channel'}
for name in ('astermind.builder.models', 'astermind.builder.topology', 'astermind.builder.config',
             'astermind.engine.network_host', 'astermind.engine.simulation', 'astermind.engine.compute'):
    loaded = import_module(name)
    assert Path(loaded.__file__).is_relative_to(target), loaded.__file__
from astermind.builder.config import SimulationConfig
identity = json.loads(files('astermind.engine.compute').joinpath('scheduler_runtime/vllm_0_31_identity.json').read_text())
assert identity, 'native Scheduler identity must be packaged'
assert files('astermind.engine.compute').joinpath('scheduler_runtime/requirements-vllm-0.31.lock').is_file()
scene = files('examples.native').joinpath('dynamic_tp2_pp2.json')
config = SimulationConfig.load(scene)
assert config.cells[0].parallel.workers == 4
assert len(config.topology().devices) == 11
assert Path(config.physical_network).is_file()
for name in ('index.html', 'replay.js', 'replay.css'):
    assert files('viz').joinpath(name).is_file(), name
from viz.export import audit_checks
assert callable(audit_checks)
from workloads.main_composite_chat import adapter
assert Path(adapter.__file__).is_relative_to(target)
assert adapter.load_countries(adapter.DEFAULT_COUNTRIES)
assert adapter.gateways()
for profile, day_type in adapter.PROFILE_PATTERNS.values():
    assert adapter.load_regional_hour_weights(profile, day_type)

print(f'Installed wheel: {len(modules)} modules imported; reference hashes and portable CLI verified.')
'''


def check_paths(names):
    for name in names:
        path = PurePosixPath(name)
        if (path.is_absolute() or '..' in path.parts
                or {'local', '.git', '__pycache__', 'AGENTS.md'}.intersection(path.parts)
                or path.name.startswith('.env') and path.name != '.env.example'
                or path.suffix == '.pyc'):
            raise ValueError(f'Private/generated path in distribution: {name}')


def copy_source(root, destination):
    """Build current Git inputs without ignored local data or stale build/lib."""
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root,
    ).decode('utf-8').split('\0')
    for name in set(filter(None, names)):
        source = root / name
        if not source.is_file():
            continue  # Files deleted in the working tree must stay deleted.
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError(f'Distribution source must stay inside repository: {name}')
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def main():
    root = Path(__file__).resolve().parents[2]
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                   'ORBITSERVE_ISLSIM_ROOT': '/nonexistent/orbitserve-islsim-package-check'}
    environment.pop('PYTHONPATH', None)
    environment.pop('ORBITServe_MIDLAYER_ROOT', None)
    environment.pop('ORBITSERVE_MIDLAYER_ROOT', None)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, help="retain build/install/replay evidence in a new directory")
    args = parser.parse_args()
    if args.work_dir:
        args.work_dir.mkdir(parents=True, exist_ok=False)
    context = nullcontext(str(args.work_dir.resolve())) if args.work_dir else TemporaryDirectory(prefix='astermind-distribution-')
    with context as folder:
        workspace = Path(folder)
        source_root = workspace / 'source'
        copy_source(root, source_root)
        artifacts = workspace / 'artifacts'
        # Default build creates an sdist, then builds the wheel from that sdist.
        subprocess.run([sys.executable, '-m', 'build', '--outdir', str(artifacts), str(source_root)],
                       check=True, env=environment)
        wheels, sources = list(artifacts.glob('*.whl')), list(artifacts.glob('*.tar.gz'))
        if len(wheels) != 1 or len(sources) != 1:
            raise ValueError('Expected one wheel and one source distribution')
        with zipfile.ZipFile(wheels[0]) as wheel:
            check_paths(wheel.namelist())
        with tarfile.open(sources[0]) as source:
            check_paths(source.getnames())
        installed = workspace / 'installed'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--target',
                        str(installed), str(wheels[0])], check=True, env=environment)
        native_env = {**environment, 'PYTHONPATH': str(installed)}
        subprocess.run([sys.executable, '-B', '-m', 'astermind.common.build_native'],
                       cwd=workspace,check=True,env=native_env)
        subprocess.run([sys.executable, '-I', '-B', '-c', SMOKE, str(installed)],
                       cwd=workspace, check=True, env=environment)
        # Child native workers must import the installed package too.
        native_env = {**environment, 'PYTHONPATH': str(installed)}
        report = workspace / 'native_result.json'
        subprocess.run([sys.executable, '-B', '-m', 'astermind',
                        '--scenario', str(installed / 'examples/native/two_cells.json'),
                        '--output', str(report)], cwd=workspace, check=True, env=native_env)
        subprocess.run([sys.executable, '-B', '-m', 'astermind.engine.audit', str(report),
                        '--output', str(workspace / 'audit.json')],
                       cwd=workspace,check=True,env=native_env)
        if environment.get('ORBITServe_VLLM030_SOURCE'):
            # Exact tick objects cross the native subprocess boundary outside source.
            sys.path.insert(0,str(installed))
            from examples.native.compute_comm_matrix import scene as native_scene
            import json
            scene030=workspace/'native030.input.json'
            scene030.write_text(json.dumps(native_scene('DS03')))
            report030=workspace/'native030.report.json'
            subprocess.run([sys.executable,'-B','-m','astermind','--scenario',str(scene030),
                            '--output',str(report030)],cwd=workspace,check=True,env=native_env)
            subprocess.run([sys.executable,'-B','-m','astermind.engine.audit',str(report030),
                            '--output',str(workspace/'native030.audit.json')],cwd=workspace,check=True,env=native_env)
        # New modes use raw schema2; unchanged viz is checked on its historical schema.
        sys.path.insert(0,str(root))
        from tests.unit.runtime.test_native_replay import report as historical_report
        historical=workspace/'historical.json'
        historical.write_text(__import__('json').dumps(historical_report()))
        subprocess.run([sys.executable, '-B', '-m', 'viz.export', str(historical),
                        '--out', str(workspace / 'replay')],
                       cwd=workspace, check=True, env=native_env)


if __name__ == '__main__':
    main()
