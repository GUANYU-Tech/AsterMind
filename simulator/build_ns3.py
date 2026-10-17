"""Build the small ns-3 C ABI using locally installed ns-3 development files."""
from pathlib import Path
import shlex
import subprocess
import hashlib
import json
import os


def main():
    here = Path(__file__).parent
    flags = subprocess.check_output(["pkg-config", "--cflags", "--libs", "ns3-core", "ns3-network", "ns3-point-to-point"], text=True)
    version = subprocess.check_output(["pkg-config", "--modversion", "ns3-core"], text=True).strip()
    target = here / "build/libserving_ns3.so"
    target.parent.mkdir(exist_ok=True)
    temporary = target.with_name(f"libserving_ns3.{os.getpid()}.so")
    command = ["g++", "-std=c++17", "-O2", "-fPIC", "-shared", f'-DSERVING_NS3_VERSION="{version}"',
               str(here / "ns3_bridge.cc"), "-o", str(temporary), *shlex.split(flags)]
    subprocess.run(command, check=True)
    temporary.replace(target)
    (target.parent/"ns3_build.json").write_text(json.dumps(dict(ns3_version=version, command=command,
        source_sha256=hashlib.sha256((here/"ns3_bridge.cc").read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(target.read_bytes()).hexdigest()), indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
