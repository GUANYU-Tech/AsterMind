"""Frontend-owned paths. All mutable state stays below viz/.runtime."""
from pathlib import Path
import os
import tempfile

VIZ = Path(__file__).resolve().parent
REPOSITORY = VIZ.parent
RUNTIME = VIZ / '.runtime'
WORKSPACE = RUNTIME / 'workspace'


def initialize():
    for path in (RUNTIME, WORKSPACE, RUNTIME / 'logs', RUNTIME / 'tmp', RUNTIME / 'cache',
                 RUNTIME / 'verification', WORKSPACE / 'local/inputs', WORKSPACE / 'local/results'):
        path.mkdir(parents=True, exist_ok=True)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    # Python may have cached its system temp directory before the server starts.
    # Keep both Python and native workers on the volume holding the experiment.
    scratch = str((RUNTIME / 'tmp').resolve())
    for key in ('TMPDIR', 'TEMP', 'TMP', 'SQLITE_TMPDIR'):
        os.environ[key] = scratch
    tempfile.tempdir = scratch
    for key in ('XDG_CACHE_HOME', 'HF_HOME', 'TORCH_HOME', 'TRITON_CACHE_DIR', 'MPLCONFIGDIR', 'NUMBA_CACHE_DIR'):
        path = RUNTIME / 'cache' / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    return WORKSPACE


def owned(path):
    path = Path(path).resolve()
    if not path.is_relative_to(RUNTIME.resolve()):
        raise ValueError('Frontend output must stay inside viz/.runtime')
    return path
