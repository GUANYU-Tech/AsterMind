"""
src.serve_trace package.

Modules:
- fake_cuda:  build fake CUDA libs, write LD_PRELOAD environment
- trace_io:   read and summarize kernel trace files
- classifier: classify kernel names into categories
- kv_transfer: parse KV transfer ops from disaggregated trace
"""

from src.serve_trace.fake_cuda import (
    build,
    get_env,
    EMULATION_DIR,
    CSRC_DIR,
    BUILD_DIR,
    FAKE_CUDA_DIR,
    HW_CONFIGS_DIR,
)
from src.serve_trace.trace_io import parse_trace, summarize_trace

__all__ = [
    "build",
    "get_env",
    "parse_trace",
    "summarize_trace",
    "EMULATION_DIR",
    "BUILD_DIR",
    "FAKE_CUDA_DIR",
    "HW_CONFIGS_DIR",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.serve_trace <command> [args]")
        print("Commands: build, env [hw_config], trace <path>, summary <path>")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "env":
        hw = sys.argv[2] if len(sys.argv) > 2 else "A800-SXM4-80GB"
        env = get_env(hw)
        for k, v in env.items():
            print(f"export {k}={v}")
    elif cmd == "trace":
        entries = parse_trace(sys.argv[2])
        summarize_trace(entries)
    elif cmd == "summary":
        entries = parse_trace(sys.argv[2])
        summarize_trace(entries)
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
