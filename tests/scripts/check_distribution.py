"""Build an sdist and its wheel; smoke-test the installed artifact outside source."""
import os
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
import orbitserve
assert Path(orbitserve.__file__).is_relative_to(target), orbitserve.__file__
modules = list(pkgutil.walk_packages(orbitserve.__path__, orbitserve.__name__ + '.'))
for module in modules:
    loaded = import_module(module.name)
    assert Path(loaded.__file__).is_relative_to(target), loaded.__file__
from orbitserve.orbit.sgp4 import frames
for name, expected in (
    ('finals2000A.all', frames.PINNED_IERS_A_SHA256),
    ('ReadMe.finals2000A', frames.PINNED_IERS_README_SHA256),
    ('Leap_Second.dat', frames.PINNED_LEAP_SECONDS_SHA256),
):
    resource = files('orbitserve.orbit').joinpath('data/iers/0.2026.5.11.1.8.52', name)
    assert hashlib.sha256(resource.read_bytes()).hexdigest() == expected, name
for name in ('state_corpus.json', 'geometry_corpus.json'):
    json.loads(files('orbitserve.orbit').joinpath('data/stk_validation', name).read_text())
from orbitserve.experiments.islsim_cli import main
output = Path.cwd() / 'islsim.json'
assert main(['--out', str(output)]) == 0
assert set(json.loads(output.read_text())) == {'tx', 'rx', 'channel'}
for name in ('network_sim', 'network_sim.topology', 'simulator.config',
             'simulator.network', 'simulator.engine', 'integration'):
    loaded = import_module(name)
    assert Path(loaded.__file__).is_relative_to(target), loaded.__file__
from simulator.config import SimulationConfig
identity = json.loads(files('integration').joinpath('runtime/vllm_0_31_identity.json').read_text())
assert identity, 'native Scheduler identity must be packaged'
assert files('integration').joinpath('requirements-scheduler-latest.lock').is_file()
scene = files('simulator').joinpath('examples/dynamic_tp2_pp2.json')
config = SimulationConfig.load(scene)
assert config.cells[0].parallel.workers == 4
assert len(config.topology().devices) == 11
assert Path(config.physical_network).is_file()
for name in ('index.html', 'replay.js', 'replay.css'):
    assert files('simulator').joinpath('web', name).is_file(), name
from simulator.replay import audit_checks
assert callable(audit_checks)
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
    with TemporaryDirectory(prefix='orbitserve-distribution-') as folder:
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
        subprocess.run([sys.executable, '-I', '-B', '-c', SMOKE, str(installed)],
                       cwd=workspace, check=True, env=environment)


if __name__ == '__main__':
    main()
