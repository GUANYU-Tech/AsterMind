"""Build the local FlowSim sources without changing the active installation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sysconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    import pybind11

    root = Path(__file__).resolve().parents[1]
    source = root / 'third_party/astermind/third_party/flowsim'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve() / ('_flowsim' + sysconfig.get_config_var('EXT_SUFFIX'))
    sources = [source / name for name in (
        'python/bindings.cc', 'src/event_simulator.cc', 'src/topology.cc')]
    command = [os.environ.get('CXX', 'c++'), '-O2', '-std=c++17', '-shared', '-fPIC',
               '-I' + sysconfig.get_paths()['include'], '-I' + pybind11.get_include(),
               '-I' + str(source / 'include'), *(str(p) for p in sources), '-o', str(output)]
    subprocess.run(command, check=True)
    inputs = sources + [source / 'include/flowsim/flowsim.hh']
    record = {'command': command, 'source_sha256': {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'binary_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    (args.output_dir / 'build.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
