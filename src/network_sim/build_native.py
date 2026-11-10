"""Build the vendored FlowSim adapter and optionally run original regressions."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compiler", help="C++ compiler executable (g++, clang++, or zig)")
    parser.add_argument("--output-dir", type=Path, help="build into a separate directory")
    parser.add_argument("--upstream-tests", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output_dir or root / "build"
    checkout = root.parents[1] if root.parent.name == "src" else root.parent
    bridge = root / "comm/native/bridge.cc"
    output.mkdir(parents=True, exist_ok=True)
    compiler = args.compiler or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        candidate = checkout / "integration/deps/ziglang/zig.exe"
        if candidate.exists():
            compiler = str(candidate)
        else:
            parser.error("provide --compiler; no supported compiler was found")
    prefix = [compiler]
    if Path(compiler).stem == "zig":
        prefix += ["c++"]
    env = os.environ.copy()
    env.setdefault("ZIG_GLOBAL_CACHE_DIR", str(checkout / "tmp/zig-global"))
    env.setdefault("ZIG_LOCAL_CACHE_DIR", str(checkout / "tmp/zig-local"))
    vendor = root / "comm/_vendor/flowsim"
    flags = ["-std=c++17", "-O2", f"-I{vendor / 'include'}"]
    sources = [vendor / f"src/{name}.cc" for name in ("topology", "event_simulator")]
    binary = output / ("network_native.dll" if sys.platform == "win32" else "libnetwork_native.so")
    command = prefix + flags + ["-shared"]
    if sys.platform != "win32":
        command += ["-fPIC"]
    command += [str(p) for p in sources] + [str(bridge), "-o", str(binary)]
    subprocess.run(command, check=True, env=env)
    validation = None
    if args.upstream_tests:
        tests = output / ("upstream_tests.exe" if sys.platform == "win32" else "upstream_tests")
        subprocess.run(prefix + flags + [str(p) for p in sources] +
                       [str(vendor / "src/simulator.cc"), str(vendor / "tests/test_flowsim.cc"),
                        "-o", str(tests)], check=True, env=env)
        result = subprocess.run([str(tests)], check=True, text=True, capture_output=True)
        validation = result.stdout
        print(validation, end="")
    record = {"compiler": str(compiler), "binary": binary.name,
              "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
              "upstream_tests": validation,
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sources + [bridge, vendor / "include/flowsim/flowsim.hh"]}}
    (output / "build_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Built {binary}")


if __name__ == "__main__":
    main()
