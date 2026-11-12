"""Build the locally patched FlowSim without tracking vendor sources/binaries.

Apply flowsim_clock.patch inside the pinned FlowSim source before building.
Run this with the Python interpreter used by the simulation service.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "third_party/astermind/third_party/flowsim",
    )
    parser.add_argument("--output", type=Path, default=root / "local/native")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    header = source / "include/flowsim/flowsim.hh"
    if "advance_until_ticks" not in header.read_text():
        parser.error(
            "apply astermind/engine/adapters/flowsim_clock.patch inside the pinned FlowSim directory first"
        )
    output.mkdir(parents=True, exist_ok=True)
    paths = [
        source / name
        for name in ("python/bindings.cc", "src/event_simulator.cc", "src/topology.cc")
    ]
    includes = shlex.split(
        subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--includes"], text=True
        )
    )
    binary = output / ("_flowsim" + sysconfig.get_config_var("EXT_SUFFIX"))
    temporary = output / (binary.name + ".building")
    command = [
        "g++",
        "-O2",
        "-std=c++17",
        "-shared",
        "-fPIC",
        *includes,
        "-I" + str(source / "include"),
        *map(str, paths),
        "-o",
        str(temporary),
    ]
    subprocess.run(command, check=True)
    temporary.replace(binary)
    evidence = dict(
        command=command,
        clock="orbitserve.clock.v1",
        sources={
            str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*paths, header]
        },
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
    )
    (output / "flowsim-clock-build.json").write_text(json.dumps(evidence, indent=2))
    print(binary)


if __name__ == "__main__":
    main()
