"""Build an external shared-runtime FlowSim without changing its checkout.

The advance_until guard mirrors the native run() guard: a positive sub-ULP
residual must advance to the next representable long-double instant.
"""

import argparse
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def guarded_source(text):
    start = text.index("void advance_until(")
    end = text.index("void run(", start)
    body = text[start:end]
    if "std::nextafter(" in body:
        return text, False
    pattern = (
        r"next_time\s*=\s*std::min\(\s*next_time,\s*current_time_\s*\+"
        r"\s*flow\.remaining_bytes\s*/\s*flow\.rate_bytes_per_ns\s*\);"
    )
    replacement = """auto completion_time = current_time_ + flow.remaining_bytes / flow.rate_bytes_per_ns;
        if (completion_time <= current_time_ && flow.remaining_bytes > kByteEpsilon) {
          completion_time = std::nextafter(current_time_, std::numeric_limits<long double>::infinity());
        }
        next_time = std::min(next_time, completion_time);"""
    body, count = re.subn(pattern, replacement, body)
    if count != 1:
        raise ValueError(
            "unrecognized FlowSim advance_until implementation; review this source version"
        )
    return text[:start] + body + text[end:], True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("local/native"))
    parser.add_argument("--compiler")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("build native FlowSim in Linux/WSL")
    source = args.source_root.resolve()
    output = args.output_dir.resolve()
    compiler = args.compiler or shutil.which("c++")
    if compiler is None:
        parser.error("a C++17 compiler is required")
    files = [
        source / name
        for name in ("python/bindings.cc", "src/event_simulator.cc", "src/topology.cc")
    ]
    if not all(p.is_file() for p in files):
        parser.error("complete external shared-runtime FlowSim sources are required")
    original = files[1].read_text(encoding="utf8")
    fixed, applied = guarded_source(original)
    output.mkdir(parents=True, exist_ok=True)
    compiled = output / "event_simulator.cc"
    compiled.write_text(fixed, encoding="utf8")
    binary = output / ("_flowsim" + sysconfig.get_config_var("EXT_SUFFIX"))
    includes = shlex.split(
        subprocess.check_output(
            [sys.executable, "-m", "pybind11", "--includes"], text=True
        )
    )
    command = [
        compiler,
        "-O2",
        "-std=c++17",
        "-shared",
        "-fPIC",
        *includes,
        "-I" + str(source / "include"),
        str(files[0]),
        str(compiled),
        str(files[2]),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True)
    metadata = {
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "advance_until_guard_applied": applied,
        "compiled_source_sha256": hashlib.sha256(compiled.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "command": command,
    }
    (output / "flowsim_build.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf8"
    )
    print(binary)


if __name__ == "__main__":
    main()
