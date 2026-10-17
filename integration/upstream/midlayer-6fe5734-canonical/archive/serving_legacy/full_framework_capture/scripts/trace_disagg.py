#!/usr/bin/env python3
"""
Run vLLM disaggregated serving under CUDA emulation: prefill + decode instances.

Launches two independent vLLM engine processes concurrently:
  - Prefill instance: kv_role=kv_producer, kv_rank=0
  - Decode instance: kv_role=kv_consumer, kv_rank=1

Each instance is a fully independent vLLM engine with its own GPU allocation,
trace capture, and scheduler sidecar. All vLLM native parallelism parameters
(TP, PP, DP, EP, context parallel) are transparently passed through via JSON.

Usage:
    python scripts/serve_trace/trace_disagg.py \
        --model /share_data/public_models/Qwen3-0.6B \
        --prefill-gpus 0 --decode-gpus 1 \
        --kv-connector P2pNcclConnector \
        --num-requests 3 --prompt-len 512 --max-tokens 8

    # With per-instance engine args:
    python scripts/serve_trace/trace_disagg.py \
        --model /share_data/public_models/Qwen3-0.6B \
        --prefill-gpus 0,1 --decode-gpus 2 \
        --prefill-engine-args '{"tensor_parallel_size": 2}' \
        --decode-engine-args '{"tensor_parallel_size": 1}' \
        --kv-connector P2pNcclConnector

    # With JSON config file (no CLI GPU args needed):
    python scripts/serve_trace/trace_disagg.py --config disagg_config.json
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.serve_predict.parallel_utils import (  # noqa: E402
    parallel_worker_count,
    validate_gpu_capacity,
    validate_parallel_semantics,
    ParallelConfigError,
)

PYTHON = sys.executable
INSTANCE_SCRIPT = str(ROOT / "scripts" / "serve_trace" / "_disagg_instance.py")


def parse_gpu_list(gpu_str):
    """Parse comma-separated GPU IDs into a list of ints."""
    return [int(g.strip()) for g in gpu_str.split(",") if g.strip()]


def validate_config(prefill_gpus_str, decode_gpus_str, kv_connector,
                    prefill_engine_args_str, decode_engine_args_str):
    """Validate disaggregated deployment configuration.

    Returns (prefill_gpus, decode_gpus, prefill_engine, decode_engine).
    Raises ValueError on invalid configuration.
    """
    if not kv_connector:
        raise ValueError(
            "KV connector is required for disaggregated deployment. "
            "Use --kv-connector (e.g., P2pNcclConnector)."
        )

    if not prefill_gpus_str:
        raise ValueError("Prefill GPU list is required (--prefill-gpus or config).")
    if not decode_gpus_str:
        raise ValueError("Decode GPU list is required (--decode-gpus or config).")

    prefill_gpus = parse_gpu_list(prefill_gpus_str)
    decode_gpus = parse_gpu_list(decode_gpus_str)

    overlap = set(prefill_gpus) & set(decode_gpus)
    if overlap:
        raise ValueError(
            f"Prefill and decode GPU sets must not overlap. "
            f"Overlapping GPU IDs: {sorted(overlap)}"
        )

    prefill_engine = json.loads(prefill_engine_args_str) if prefill_engine_args_str else {}
    decode_engine = json.loads(decode_engine_args_str) if decode_engine_args_str else {}

    try:
        validate_gpu_capacity("prefill", prefill_gpus, prefill_engine)
        validate_gpu_capacity("decode", decode_gpus, decode_engine)
        validate_parallel_semantics(prefill_engine, "prefill")
        validate_parallel_semantics(decode_engine, "decode")
    except ParallelConfigError as e:
        raise ValueError(str(e)) from e

    return prefill_gpus, decode_gpus, prefill_engine, decode_engine


def build_instance_env(gpus, trace_base, hw_config, tp_size=1, mode="real",
                       nsys_profiled=False):
    """Build environment variables for one engine instance.

    When `nsys_profiled` is True (this instance is wrapped by `nsys profile` to
    capture a matched ground-truth, see `build_nsys_prefix`), the engine is run
    in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so a `cudaProfilerStart/Stop`
    range issued by the instance reaches the same process the kernels launch in
    (the EngineCore child cannot be reached by a parent-process profiler — see
    BL-20260603). The instance reads PD_NSYS_PROFILE to gate its inference range.
    """
    env = os.environ.copy()
    if nsys_profiled:
        # Run the engine in-process so cudaProfilerStart/Stop gating works, and
        # tell the instance to wrap ONLY its real generate() in the profiler range
        # (warmup/init stays outside the captured window).
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env["PD_NSYS_PROFILE"] = "1"
    if mode == "plugin":
        # approach-D: real GPU, operator capture; no LD_PRELOAD / fake CUDA.
        # Strip emulation env vars that may be inherited from the parent shell.
        for key in ("LD_PRELOAD", "CUDA_EMULATE_CONFIG", "VLLM_NCCL_SO_PATH"):
            env.pop(key, None)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        env["CUDA_TRACE_OUTPUT"] = trace_base
        return env
    from src.serve_trace import get_env
    emu_env = get_env(hw_config, trace_output=trace_base, tp_size=tp_size)
    env.update(emu_env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    env["CUDA_TRACE_OUTPUT"] = trace_base
    return env


def build_nsys_prefix(nsys_out):
    """Return the `nsys profile ...` command prefix for a matched GT capture.

    Uses cudaProfilerApi gating (paired with the instance's cudaProfilerStart/Stop
    around real inference) so warmup/init is excluded — the same convention as
    scripts/serve_trace/capture_nsys.py. `nsys` must be on PATH; the caller is
    responsible for only enabling this when a capture is requested.
    """
    return [
        "nsys", "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--trace=cuda,nvtx",
        "--export=sqlite",
        "-f", "true",
        "-o", nsys_out,
    ]


def build_instance_command(role, model, engine_args, kv_connector, kv_port,
                           kv_ip, http_port, peer_kv_port,
                           num_requests, prompt_len, max_tokens,
                           seq_len, trace_base, hw_config, prompts_file=None,
                           seed=42, timeout=300, mode="real"):
    """Build command for _run_vllm_disagg_instance.py."""
    cmd = [
        PYTHON, INSTANCE_SCRIPT,
        "--role", role,
        "--model", model,
        "--engine-args-json", json.dumps(engine_args),
        "--kv-connector", kv_connector,
        "--kv-port", str(kv_port),
        "--kv-ip", kv_ip,
        "--http-port", str(http_port),
        "--peer-kv-port", str(peer_kv_port),
        "--num-requests", str(num_requests),
        "--prompt-len", str(prompt_len),
        "--max-tokens", str(max_tokens),
        "--seq-len", str(seq_len),
        "--trace-output", trace_base,
        "--hw-config", hw_config,
        "--seed", str(seed),
        "--timeout", str(timeout),
        "--mode", mode,
    ]
    if prompts_file:
        cmd.extend(["--prompts-file", prompts_file])
    return cmd


def _terminate_process(proc):
    """Terminate a process with kill fallback if SIGTERM is ignored."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _wait_for_disagg_processes(prefill_proc, decode_proc, timeout_s,
                               now_fn=None, sleep_fn=None, terminate_fn=None):
    """Poll both processes until completion, failure, or deadline.

    Returns one of: "success", "prefill_failed", "decode_failed", "timeout".
    """
    if now_fn is None:
        now_fn = time.time
    if sleep_fn is None:
        sleep_fn = time.sleep
    if terminate_fn is None:
        terminate_fn = _terminate_process

    deadline = now_fn() + timeout_s
    while True:
        prefill_rc = prefill_proc.poll()
        decode_rc = decode_proc.poll()

        if prefill_rc is not None and decode_rc is not None:
            if prefill_rc == 0 and decode_rc == 0:
                return "success"
            elif prefill_rc != 0:
                return "prefill_failed"
            else:
                return "decode_failed"

        if prefill_rc is not None and prefill_rc != 0:
            terminate_fn(decode_proc)
            return "prefill_failed"

        if decode_rc is not None and decode_rc != 0:
            terminate_fn(prefill_proc)
            return "decode_failed"

        if now_fn() > deadline:
            terminate_fn(prefill_proc)
            terminate_fn(decode_proc)
            return "timeout"

        sleep_fn(1)


def merge_instance_traces(trace_base, role):
    """Merge multi-process trace files for a single instance.

    Returns True on success, False on failure.
    """
    from src.serve_trace import parse_trace
    try:
        merged = parse_trace(trace_base, merge_all=True)
        if not merged:
            print(f"[{role}] WARNING: no trace entries after merge", file=sys.stderr)
            return False
        import json
        with open(trace_base, "w") as f:
            json.dump(merged, f)
        print(f"[{role}] Traces merged: {trace_base}")
        return True
    except Exception as e:
        print(f"[{role}] merge failed: {e}", file=sys.stderr)
        return False


def _cleanup_pid_trace_files(trace_base):
    base = Path(trace_base)
    removed = 0
    for path in base.parent.glob(base.name + ".*"):
        try:
            int(path.name.rsplit(".", 1)[-1])
        except ValueError:
            continue
        path.unlink()
        removed += 1
    return removed


def validate_nsys_outputs(nsys_roles, role_sqlite_paths):
    """Fail closed if a profiled role's nsys sqlite is missing or empty.

    `nsys_roles` is the set of roles for which a ground-truth capture was
    requested (e.g. {"decode"}); `role_sqlite_paths` maps role -> expected
    sqlite path. Returns a list of human-readable problems (empty == OK).

    A profiled role MUST yield a non-empty sqlite with at least one kernel row;
    a silently-missing or zero-kernel capture would let a PD run report success
    while the role's ground truth never existed. Callers treat a non-empty list
    as fatal (sys.exit(1)) BEFORE writing success provenance.
    """
    import sqlite3

    problems = []
    for role in sorted(nsys_roles):
        path = role_sqlite_paths.get(role)
        if not path:
            problems.append(f"{role}: no nsys sqlite path resolved")
            continue
        p = Path(path)
        if not p.exists():
            problems.append(f"{role}: nsys sqlite missing ({path}); "
                            f"nsys capture did not produce output")
            continue
        if p.stat().st_size == 0:
            problems.append(f"{role}: nsys sqlite is empty ({path})")
            continue
        # A file can exist yet contain no kernel rows (e.g. capture range never
        # opened). Require the CUPTI kernel table to exist and be non-empty.
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            try:
                n = conn.execute(
                    "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
            finally:
                conn.close()
        except sqlite3.Error as e:
            problems.append(f"{role}: nsys sqlite unreadable or has no kernel "
                            f"table ({path}): {e}")
            continue
        if n == 0:
            problems.append(f"{role}: nsys sqlite has 0 kernel rows ({path}); "
                            f"cudaProfiler capture range likely never opened")
    return problems


def _compute_workload_hash(args):
    """Compute a stable hash from workload-defining fields."""
    import hashlib
    identity = f"{args.model}|{args.num_requests}|{args.prompt_len}|{args.max_tokens}|{args.seed}"
    return hashlib.sha256(identity.encode()).hexdigest()[:16]


def resolve_workload_provenance(args):
    """Resolve canonical workload path/hash for run provenance."""
    workload_path = getattr(args, "workload_artifact", None) or getattr(args, "prompts_file", None)
    workload_hash = getattr(args, "workload_hash", None)
    if workload_hash:
        return {
            "workload_path": workload_path,
            "workload_hash": workload_hash,
        }
    if workload_path and Path(workload_path).exists():
        try:
            with open(workload_path) as f:
                workload = json.load(f)
        except (OSError, json.JSONDecodeError):
            workload = {}
        candidate = workload.get("hash")
        if isinstance(candidate, str) and candidate:
            return {
                "workload_path": workload_path,
                "workload_hash": candidate,
            }
    return {
        "workload_path": workload_path,
        "workload_hash": _compute_workload_hash(args),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run vLLM disaggregated serving under CUDA emulation"
    )
    parser.add_argument("--model", default="/share_data/public_models/Qwen3-0.6B")
    parser.add_argument("--prefill-gpus", default=None,
                        help="Comma-separated GPU IDs for prefill instance")
    parser.add_argument("--decode-gpus", default=None,
                        help="Comma-separated GPU IDs for decode instance")
    parser.add_argument("--prefill-engine-args", default=None,
                        help="JSON string of vLLM engine args for prefill")
    parser.add_argument("--decode-engine-args", default=None,
                        help="JSON string of vLLM engine args for decode")
    parser.add_argument("--kv-connector", default=None,
                        help="KV connector class (e.g., P2pNcclConnector)")
    parser.add_argument("--kv-port", type=int, default=14579)
    parser.add_argument("--prefill-kv-port", type=int, default=None,
                        help="ZMQ port for prefill instance (default: kv-port)")
    parser.add_argument("--decode-kv-port", type=int, default=None,
                        help="ZMQ port for decode instance (default: kv-port + prefill_workers)")
    parser.add_argument("--kv-ip", default=None,
                        help="P2P host IP (default: auto-detect via vllm.utils.get_ip())")
    parser.add_argument("--num-requests", type=int, default=3)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--prefill-hw-config", default="configs/hardware/nvidia_a800.yaml")
    parser.add_argument("--decode-hw-config", default="configs/hardware/nvidia_a800.yaml")
    parser.add_argument("--trace-output", default="/tmp/vllm_disagg_trace")
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--workload-artifact", default=None,
                        help="Canonical workload artifact path for provenance")
    parser.add_argument("--workload-hash", default=None,
                        help="Canonical workload artifact hash for provenance")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=None,
                        help="JSON config file (overrides CLI args)")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--mode", choices=["real", "plugin"], default="real",
                        help="real = LD_PRELOAD CUDA emulation; plugin = approach-D "
                             "operator capture on real GPUs (no LD_PRELOAD).")
    parser.add_argument("--nsys-role", choices=["none", "prefill", "decode", "both"],
                        default="none",
                        help="Capture a matched nsys ground truth for the selected "
                             "instance(s) by wrapping that subprocess in `nsys "
                             "profile` (cudaProfilerApi-gated to real inference). "
                             "Used to validate the synthesized PD trace per role "
                             "against role-matched nsys (e.g. decode-only). Requires "
                             "`nsys` on PATH. Default none = no profiling.")
    parser.add_argument("--nsys-out-dir", default=None,
                        help="Directory for nsys sqlite output (default: alongside "
                             "--trace-output). One <role>_nsys.sqlite per profiled role.")
    args = parser.parse_args()

    # Load config file if provided (merges over CLI defaults)
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        for key, val in cfg.items():
            key_arg = key.replace("-", "_")
            if hasattr(args, key_arg):
                if key_arg in ("prefill_gpus", "decode_gpus") and isinstance(val, list):
                    setattr(args, key_arg, ",".join(str(g) for g in val))
                elif isinstance(val, dict):
                    setattr(args, key_arg, json.dumps(val))
                else:
                    setattr(args, key_arg, val)

    # Auto-detect P2P host IP if not specified
    if args.kv_ip is None:
        try:
            from vllm.utils import get_ip
            args.kv_ip = get_ip()
        except ImportError:
            import socket
            args.kv_ip = socket.gethostbyname(socket.gethostname())

    # Validate configuration
    try:
        prefill_gpus, decode_gpus, prefill_engine, decode_engine = validate_config(
            args.prefill_gpus, args.decode_gpus, args.kv_connector,
            args.prefill_engine_args, args.decode_engine_args,
        )
    except ValueError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    if args.nsys_role != "none" and args.mode != "plugin":
        print("[ERROR] --nsys-role requires --mode plugin (cudaProfiler gating "
              "only works with real GPU execution, not LD_PRELOAD emulation).",
              file=sys.stderr)
        sys.exit(1)

    trace_base = args.trace_output
    prefill_trace = f"{trace_base}_prefill"
    decode_trace = f"{trace_base}_decode"

    # Per-instance P2P ports with range safety
    def _worker_count(engine_args):
        return (engine_args.get("tensor_parallel_size", 1) *
                engine_args.get("pipeline_parallel_size", 1) *
                engine_args.get("data_parallel_size", 1))

    prefill_workers = _worker_count(prefill_engine)
    decode_workers = _worker_count(decode_engine)
    prefill_kv_port = args.prefill_kv_port if args.prefill_kv_port is not None else args.kv_port
    decode_kv_port = args.decode_kv_port if args.decode_kv_port is not None else prefill_kv_port + prefill_workers

    prefill_range = set(range(prefill_kv_port, prefill_kv_port + prefill_workers))
    decode_range = set(range(decode_kv_port, decode_kv_port + decode_workers))
    overlap = prefill_range & decode_range
    if overlap:
        print(f"[ERROR] Prefill port range [{prefill_kv_port}, {prefill_kv_port + prefill_workers - 1}] "
              f"overlaps with decode range [{decode_kv_port}, {decode_kv_port + decode_workers - 1}]",
              file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  vLLM Disaggregated Deployment Emulation")
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  Prefill GPUs: {prefill_gpus} (engine: {prefill_engine or 'default'})")
    print(f"  Decode GPUs: {decode_gpus} (engine: {decode_engine or 'default'})")
    print(f"  KV Connector: {args.kv_connector}")
    print(f"  KV Transfer: {args.kv_ip}:{args.kv_port}")
    print(f"  Prefill KV port: {prefill_kv_port} ({prefill_workers} workers)")
    print(f"  Decode KV port: {decode_kv_port} ({decode_workers} workers)")
    print()

    # HTTP ports for P2pNcclConnector metadata (separate from ZMQ data port)
    prefill_http_port = args.kv_port + 1000
    decode_http_port = args.kv_port + 1001

    # Build environment and commands for each instance
    _cleanup_pid_trace_files(prefill_trace)
    _cleanup_pid_trace_files(decode_trace)
    prefill_nsys = args.nsys_role in ("prefill", "both")
    decode_nsys = args.nsys_role in ("decode", "both")
    prefill_env = build_instance_env(prefill_gpus, prefill_trace, args.prefill_hw_config,
                                     tp_size=prefill_engine.get("tensor_parallel_size", 1),
                                     mode=args.mode, nsys_profiled=prefill_nsys)
    decode_env = build_instance_env(decode_gpus, decode_trace, args.decode_hw_config,
                                    tp_size=decode_engine.get("tensor_parallel_size", 1),
                                    mode=args.mode, nsys_profiled=decode_nsys)

    prefill_cmd = build_instance_command(
        "prefill", args.model, prefill_engine, args.kv_connector,
        prefill_kv_port, args.kv_ip, prefill_http_port, decode_kv_port,
        args.num_requests, args.prompt_len,
        args.max_tokens, args.seq_len, prefill_trace, args.prefill_hw_config,
        args.prompts_file, args.seed, args.timeout, mode=args.mode,
    )
    decode_cmd = build_instance_command(
        "decode", args.model, decode_engine, args.kv_connector,
        decode_kv_port, args.kv_ip, decode_http_port, prefill_kv_port,
        args.num_requests, args.prompt_len,
        args.max_tokens, args.seq_len, decode_trace, args.decode_hw_config,
        args.prompts_file, args.seed, args.timeout, mode=args.mode,
    )

    # Wrap the selected instance(s) in `nsys profile` to capture a role-matched
    # ground truth (decode-only nsys for the decode role, etc.).
    nsys_out_dir = Path(args.nsys_out_dir) if args.nsys_out_dir else Path(trace_base).parent
    if prefill_nsys or decode_nsys:
        nsys_out_dir.mkdir(parents=True, exist_ok=True)
    prefill_nsys_out = str(nsys_out_dir / "prefill_nsys")
    decode_nsys_out = str(nsys_out_dir / "decode_nsys")
    if prefill_nsys:
        prefill_cmd = build_nsys_prefix(prefill_nsys_out) + prefill_cmd
    if decode_nsys:
        decode_cmd = build_nsys_prefix(decode_nsys_out) + decode_cmd

    # Launch both instances concurrently with file-based logging to avoid pipe deadlock.
    prefill_log = f"{prefill_trace}.log"
    decode_log = f"{decode_trace}.log"

    print("[disagg] Starting decode (consumer) instance...")
    decode_log_f = open(decode_log, "w")
    decode_proc = subprocess.Popen(
        decode_cmd, env=decode_env,
        stdout=decode_log_f, stderr=subprocess.STDOUT,
    )

    time.sleep(2)

    print("[disagg] Starting prefill (producer) instance...")
    prefill_log_f = open(prefill_log, "w")
    prefill_proc = subprocess.Popen(
        prefill_cmd, env=prefill_env,
        stdout=prefill_log_f, stderr=subprocess.STDOUT,
    )

    # Wait for both processes with shared deadline
    try:
        result = _wait_for_disagg_processes(prefill_proc, decode_proc, args.timeout)
        if result == "timeout":
            print(f"[FATAL] Deadline exceeded ({args.timeout}s). Both terminated.",
                  file=sys.stderr)
    finally:
        prefill_log_f.close()
        decode_log_f.close()

    # Report results
    print(f"\n[prefill] Exit code: {prefill_proc.returncode} (log: {prefill_log})")
    print(f"[decode] Exit code: {decode_proc.returncode} (log: {decode_log})")

    if prefill_proc.returncode != 0 or decode_proc.returncode != 0:
        print("\n[FATAL] One or both instances failed.", file=sys.stderr)
        sys.exit(1)

    # Merge traces per instance (fatal on failure). An nsys-profiled instance is
    # a GROUND-TRUTH capture: it runs the engine in-process (so cudaProfiler gating
    # reaches the kernels), which disables the .pth-based operator hook used for
    # the synthesized trace. Such an instance produces the nsys sqlite, not a synth
    # trace, so its synth-trace merge is skipped (the synth trace for that role is
    # produced by a separate, non-profiled run on the same workload).
    if not prefill_nsys:
        if not merge_instance_traces(prefill_trace, "prefill"):
            print("[FATAL] Prefill trace merge failed.", file=sys.stderr)
            sys.exit(1)
    else:
        print("[prefill] nsys ground-truth capture (synth-trace merge skipped)")
        _stale = Path(prefill_trace)
        if _stale.exists():
            _stale.unlink()
            print(f"[prefill] removed stale synth trace: {prefill_trace}")
    if not decode_nsys:
        if not merge_instance_traces(decode_trace, "decode"):
            print("[FATAL] Decode trace merge failed.", file=sys.stderr)
            sys.exit(1)
    else:
        print("[decode] nsys ground-truth capture (synth-trace merge skipped)")
        _stale = Path(decode_trace)
        if _stale.exists():
            _stale.unlink()
            print(f"[decode] removed stale synth trace: {decode_trace}")

    # Validate nsys outputs BEFORE writing success provenance (fail-closed).
    nsys_roles = set()
    role_sqlite_paths = {}
    if prefill_nsys:
        nsys_roles.add("prefill")
        role_sqlite_paths["prefill"] = prefill_nsys_out + ".sqlite"
    if decode_nsys:
        nsys_roles.add("decode")
        role_sqlite_paths["decode"] = decode_nsys_out + ".sqlite"
    if nsys_roles:
        problems = validate_nsys_outputs(nsys_roles, role_sqlite_paths)
        if problems:
            for p in problems:
                print(f"[FATAL] nsys validation: {p}", file=sys.stderr)
            sys.exit(1)

    # Write disaggregated provenance
    workload_provenance = resolve_workload_provenance(args)
    provenance = {
        "mode": "disaggregated",
        "capture_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": args.model,
        "prefill_gpus": prefill_gpus,
        "decode_gpus": decode_gpus,
        "prefill_engine_args": prefill_engine,
        "decode_engine_args": decode_engine,
        "kv_connector": args.kv_connector,
        "kv_port": args.kv_port,
        "kv_ip": args.kv_ip,
        "prefill_kv_port": prefill_kv_port,
        "decode_kv_port": decode_kv_port,
        "prefill_kv_port_range": [prefill_kv_port, prefill_kv_port + prefill_workers - 1],
        "decode_kv_port_range": [decode_kv_port, decode_kv_port + decode_workers - 1],
        "prefill_hw_config": args.prefill_hw_config,
        "decode_hw_config": args.decode_hw_config,
        "prefill_trace": prefill_trace,
        "decode_trace": decode_trace,
        "prefill_sidecar": f"{prefill_trace}_scheduler.jsonl",
        "decode_sidecar": f"{decode_trace}_scheduler.jsonl",
        "num_requests": args.num_requests,
        "prompt_len": args.prompt_len,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "workload_path": workload_provenance["workload_path"],
        "workload_hash": workload_provenance["workload_hash"],
    }
    if prefill_nsys:
        provenance["prefill_nsys_sqlite"] = prefill_nsys_out + ".sqlite"
    if decode_nsys:
        provenance["decode_nsys_sqlite"] = decode_nsys_out + ".sqlite"
    prov_path = f"{trace_base}_provenance.json"
    with open(prov_path, "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"\n[disagg] Provenance: {prov_path}")
    print("SUCCESS: Disaggregated emulation completed!")


if __name__ == "__main__":
    sys.exit(main())
