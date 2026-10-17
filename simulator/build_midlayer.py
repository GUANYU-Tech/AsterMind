"""Build the pinned M-L native LRU helper without modifying its sources."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

from .bootstrap import DEFAULT_ML


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midlayer-root", type=Path, default=DEFAULT_ML)
    parser.add_argument("--compiler")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("build the native M-L helper in Linux/WSL")
    root = args.midlayer_root.resolve()
    source = root / "src/kernel/csrc/lru_sim.c"
    if not source.is_file():
        parser.error("initialize the M-L submodule before building")
    compiler = args.compiler or shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        parser.error("a C compiler is required")
    output = root / "build/lru_sim.so"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([compiler, "-O3", "-shared", "-fPIC", "-o", str(output), str(source)], check=True)
    print(output)


if __name__ == "__main__":
    main()
