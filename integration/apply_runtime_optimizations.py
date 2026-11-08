"""Apply the reviewable performance patch to the local AsterMind checkout."""

import argparse
from pathlib import Path
import shutil
import subprocess


def apply_local_optimizations(*, check_only=False):
    root = Path(__file__).resolve().parents[1]
    patch = root / 'integration/patches/astermind-runtime-performance.patch'
    owned = root / 'src/orbitserve/runtime/prediction_cache.py'
    target = root / 'third_party/astermind/third_party/midlayer/midlayer/kernel/prediction_cache.py'
    if target.exists() and target.read_bytes() != owned.read_bytes():
        raise RuntimeError('local prediction_cache.py differs; preserve and reconcile that change first')

    def accepts(reverse=False):
        command = ['git', 'apply', '--check']
        if reverse:
            command.append('--reverse')
        return subprocess.run(command + [str(patch)], cwd=root, capture_output=True).returncode == 0

    if accepts(reverse=True):
        state = 'already applied'
    elif accepts():
        state = 'ready to apply'
    else:
        raise RuntimeError('local dependency sources differ from the patch; no files were overwritten')
    if not check_only:
        if state == 'ready to apply':
            subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True)
        if not target.exists():
            shutil.copyfile(owned, target)
        state = 'applied'
    print('AsterMind runtime performance patch:', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='inspect applicability without changing files')
    apply_local_optimizations(check_only=parser.parse_args().check)
