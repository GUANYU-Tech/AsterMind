#!/usr/bin/env python3
"""
Build offline eval dataset: plugin trace + nsys ground truth + real benchmark.

For each case in the YAML config, runs three stages:
  1. Plugin capture  → trace JSON + scheduler sidecar
  2. nsys capture    → per-kernel GPU ground truth
  3. Real benchmark  → wall-clock ground truth (no emulation)

Then predicts and compares.

Usage:
    conda run -n bullet python scripts/serve_predict/build_eval_dataset.py
    conda run -n bullet python scripts/serve_predict/build_eval_dataset.py --cases L1 L2
    conda run -n bullet python scripts/serve_predict/build_eval_dataset.py --stage plugin
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import yaml

NSYS = os.environ.get("NSYS_PATH", "/opt/nvidia/nsight-compute/2023.2.2/host/target-linux-x64/nsys")
MCTRACER = os.environ.get("MCTRACER_PATH", "/opt/maca/bin/mcTracer")
_VLLM_PYTHON_DEFAULT = "/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python"
VLLM_PYTHON = os.environ.get(
    "VLLM_PYTHON",
    _VLLM_PYTHON_DEFAULT if os.path.isfile(_VLLM_PYTHON_DEFAULT) else sys.executable,
)
STAGES = ("plugin", "nsys", "real", "predict", "compare")

OFFLINE_YAML = str(ROOT / "configs" / "eval" / "offline_eval_cases.yaml")
SHAREGPT_YAML = str(ROOT / "configs" / "eval" / "serving_eval_cases_sharegpt_a800.yaml")

# Whether run_prediction writes the regenerated per-rank trace to disk. Default
# OFF — predict consumes the in-memory trace directly, avoiding a 200 MB+ JSON
# dump→reload round-trip per case. Set True via --dump-traces (debugging / G2).
DUMP_TRACES = False



def load_cases(config_path, selected_ids=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    # Fold the config's `defaults:` block into every case (case-specific keys
    # win). The capture pipeline builds case dicts by hand and passes these
    # explicitly, but the standalone --stage path reads them from here, so
    # without this merge fields like num_requests are missing → KeyError.
    defaults = cfg.get("defaults", {})
    cases = [{**defaults, **c} for c in cfg["cases"]]
    # Resolve ${MODELS_ROOT} in model paths here (not only in load_model_config)
    # so the capture subprocesses (--model) and generated vLLM scripts also get
    # the machine-resolved path. See model_config.expand_models_root.
    from src.serve_predict.model_config import expand_models_root
    for c in cases:
        if "model" in c:
            c["model"] = expand_models_root(c["model"])
    if selected_ids:
        cases = [c for c in cases if c["id"] in selected_ids]
    return cfg, cases


def _gpu_env(gpu_ids):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    return env


def _max_model_len(case):
    if "seq_len" in case:
        return case["seq_len"]
    return max(case.get("prompt_len", 2048) + case.get("max_tokens", 256), 2048)


def _is_sharegpt_case(case):
    return "dataset" in case


def _prompts_for_case(case):
    n = case["num_requests"]
    plen = case["prompt_len"]
    return [" ".join(["hello"] * plen) for _ in range(n)]


# ── Stage 1: Plugin capture ──────────────────────────────────────

def run_plugin_capture(case, case_dir):
    trace_out = str(case_dir / "plugin_trace.json")
    mml = _max_model_len(case)
    cmd = [
        VLLM_PYTHON, str(ROOT / "scripts" / "serve_trace" / "trace_serving.py"),
        "--mode", "plugin",
        "--model", case["model"],
        "--num-requests", str(case["num_requests"]),
        "--tensor-parallel-size", str(case["tensor_parallel_size"]),
        "--expert-parallel-size", str(case.get("expert_parallel_size", 1)),
        "--seq-len", str(mml),
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", str(mml),
        "--trace-output", trace_out,
        "--timeout", "1800",
        "--seed", str(case.get("sample_seed", case.get("poisson_seed", 42))),
    ]
    if _is_serving_case(case):
        cmd += ["--qps", str(case["qps"])]
    if _is_sharegpt_case(case):
        cmd += ["--dataset", case["dataset"]]
    else:
        cmd += [
            "--prompt-len", str(case["prompt_len"]),
            "--max-tokens", str(case["max_tokens"]),
        ]
    env = _gpu_env(case["gpu_ids"])
    print(f"  [plugin] {' '.join(cmd[-12:])}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=2400)
    if result.returncode != 0:
        print(f"  [plugin] FAILED (rc={result.returncode})")
        print(result.stderr[-1000:])
        return False

    sidecar = trace_out + "_scheduler.jsonl"
    if Path(trace_out).exists():
        print(f"  [plugin] trace: {Path(trace_out).stat().st_size / 1024:.0f} KB")
    if Path(sidecar).exists():
        print(f"  [plugin] sidecar: {Path(sidecar).stat().st_size / 1024:.0f} KB")
    return Path(trace_out).exists()


# ── Shared helpers for multi-rank kernel data ───────────────────

def _write_nsys_kernels(case_dir, by_dev, wall_ms=None, by_dev_full=None,
                        label="nsys"):
    """Write nsys_kernels.json (and optionally nsys_kernels_full.json) with
    per-rank data.  Backward-compatible: top-level fields use the bottleneck
    rank (max gpu_ns).

    Args:
        case_dir: output directory (Path)
        by_dev: dict[dev_id] -> list of {"name", "dur_ns", ...}
        wall_ms: optional wall-clock time (only nsys provides this)
        by_dev_full: optional dict[dev_id] -> list with grid/block/smem
        label: log prefix ("nsys+plugin" or "mctracer")
    """
    devs = sorted(by_dev.keys())
    per_rank = {}
    for dev in devs:
        kernels = by_dev[dev]
        gpu_ns = sum(k["dur_ns"] for k in kernels)
        span_ns = 0
        if kernels:
            starts = [k["start"] for k in kernels if "start" in k]
            ends = [k["end"] for k in kernels if "end" in k]
            if starts and ends:
                span_ns = max(ends) - min(starts)
        per_rank[str(dev)] = {
            "gpu_ns": gpu_ns,
            "span_ns": span_ns,
            "n_kernels": len(kernels),
            "kernels": [{"name": k["name"], "dur_ns": k["dur_ns"]}
                        for k in kernels],
        }

    bn_rank = max(per_rank, key=lambda r: per_rank[r]["gpu_ns"])
    bn = per_rank[bn_rank]
    nsys_data = {
        "wall_ms": wall_ms,
        "span_ns": bn["span_ns"],
        "gpu_ns": bn["gpu_ns"],
        "n_kernels": bn["n_kernels"],
        "num_devices": len(devs),
        "kernels": bn["kernels"],
        "per_rank": per_rank,
        "bottleneck_rank": bn_rank,
    }
    nsys_path = case_dir / "nsys_kernels.json"
    with open(nsys_path, "w") as f:
        json.dump(nsys_data, f, indent=2)

    if by_dev_full is not None:
        per_rank_full = {}
        for dev in devs:
            per_rank_full[str(dev)] = {"kernels": by_dev_full[dev]}
        nsys_full_path = case_dir / "nsys_kernels_full.json"
        with open(nsys_full_path, "w") as f:
            json.dump({
                "kernels": by_dev_full[devs[int(bn_rank)] if bn_rank.isdigit() else devs[0]],
                "per_rank": per_rank_full,
                "bottleneck_rank": bn_rank,
            }, f)

    for dev in devs:
        r = per_rank[str(dev)]
        marker = " ← bottleneck" if str(dev) == bn_rank and len(devs) > 1 else ""
        gpu_ms = r["gpu_ns"] / 1e6
        msg = f"  [{label}] rank{dev}: {r['n_kernels']} kernels, GPU={gpu_ms:.2f}ms"
        if wall_ms and str(dev) == bn_rank:
            msg += f", wall={wall_ms:.2f}ms"
        print(msg + marker)


# ── Stage 2: nsys capture ────────────────────────────────────────
# Runs trace_serving.py --mode plugin under nsys profiling so that
# plugin artifacts (operator records, scheduler sidecar) and nsys
# ground truth (per-kernel GPU times) come from the SAME vLLM
# process — guaranteeing identical scheduling.

def run_nsys_capture(case, case_dir):
    import sqlite3
    from collections import defaultdict

    case_dir.mkdir(parents=True, exist_ok=True)
    nsys_output = str(case_dir / "nsys_profile")
    for f in case_dir.glob("nsys_profile*"):
        f.unlink()

    mml = _max_model_len(case)
    trace_out = str(case_dir / "plugin_trace.json")
    ep = case.get("expert_parallel_size", 1)

    inner_cmd = [
        VLLM_PYTHON, str(ROOT / "scripts" / "serve_trace" / "trace_serving.py"),
        "--mode", "plugin",
        "--nsys-bracket",
        "--model", case["model"],
        "--num-requests", str(case["num_requests"]),
        "--tensor-parallel-size", str(case["tensor_parallel_size"]),
        "--expert-parallel-size", str(ep),
        "--seq-len", str(mml),
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", str(mml),
        "--trace-output", trace_out,
        "--timeout", "1800",
        "--seed", str(case.get("sample_seed", case.get("poisson_seed", 42))),
    ]
    if _is_sharegpt_case(case):
        inner_cmd += ["--dataset", case["dataset"]]
    else:
        inner_cmd += [
            "--prompt-len", str(case["prompt_len"]),
            "--max-tokens", str(case["max_tokens"]),
        ]

    cmd = [
        NSYS, "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--trace=cuda,nvtx",
        "--export=sqlite",
        "-f", "true",
        "-o", nsys_output,
    ] + inner_cmd

    env = _gpu_env(case["gpu_ids"])
    n = case["num_requests"]
    plen = case["prompt_len"]
    max_tok = case["max_tokens"]
    print(f"  [nsys+plugin] profiling {n} requests, p={plen}, d={max_tok} ...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1200)

    if result.returncode != 0:
        print(f"  [nsys+plugin] FAILED (rc={result.returncode})")
        print(result.stderr[-1000:])
        return False

    wall_ms = None
    for line in result.stdout.splitlines():
        if line.startswith("WALL_MS:"):
            wall_ms = float(line.split(":")[1])
        if line.startswith("[plugin]"):
            print(f"  {line}")

    sqlite_path = case_dir / "nsys_profile.sqlite"
    if not sqlite_path.exists():
        nsys_rep = case_dir / "nsys_profile.nsys-rep"
        if nsys_rep.exists():
            subprocess.run([NSYS, "export", "--type=sqlite", "-o",
                            str(sqlite_path), str(nsys_rep)],
                           capture_output=True, timeout=180)

    if not sqlite_path.exists():
        print(f"  [nsys+plugin] FAILED — no sqlite")
        print(result.stderr[-500:])
        return False

    conn = sqlite3.connect(str(sqlite_path))
    rows = conn.execute("""
        SELECT s.value as name, k.start, k.end, k.end - k.start as dur,
               k.deviceId as dev
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """).fetchall()
    conn.close()

    by_dev = defaultdict(list)
    for name, start, end, dur, dev in rows:
        by_dev[dev].append({"name": name, "start": start, "end": end,
                            "dur_ns": dur})

    if not by_dev:
        print(f"  [nsys+plugin] FAILED — no devices found")
        return False

    _write_nsys_kernels(case_dir, by_dev, wall_ms=wall_ms,
                        label="nsys+plugin")
    return True


# ── Stage 3: Real benchmark ──────────────────────────────────────

def run_real_benchmark(case, case_dir, n_warmup=1, n_repeat=3):
    n = case["num_requests"]
    plen = case["prompt_len"]
    max_tok = case["max_tokens"]
    tp = case["tensor_parallel_size"]
    ep = case.get("expert_parallel_size", 1)
    enable_ep = ep > 1

    mml = _max_model_len(case)

    script = f"""
import json, time, torch
from vllm import LLM, SamplingParams

llm = LLM(model='{case["model"]}', dtype='float16', gpu_memory_utilization=0.85,
           enforce_eager=True, max_model_len={mml},
           trust_remote_code=True,
           tensor_parallel_size={tp}, pipeline_parallel_size=1,
           enable_expert_parallel={enable_ep},
           enable_chunked_prefill=True, max_num_batched_tokens={mml})

prompts = [' '.join(['hello'] * {plen}) for _ in range({n})]
params = SamplingParams(temperature=0.0, max_tokens={max_tok},
                        min_tokens={max_tok}, ignore_eos=True)

# Warmup
for _ in range({n_warmup}):
    _ = llm.generate(prompts[:1], SamplingParams(temperature=0.0, max_tokens=1))
    torch.cuda.synchronize()

# Timed runs
results = []
for run in range({n_repeat}):
    torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    out = llm.generate(prompts, params)
    torch.cuda.synchronize()
    t1 = time.perf_counter_ns()
    wall_ns = t1 - t0
    total_tokens = sum(len(o.outputs[0].token_ids) for o in out)
    results.append({{"run": run, "wall_ns": wall_ns, "total_output_tokens": total_tokens,
                     "num_requests": len(out)}})

print("REAL_RESULTS:" + json.dumps(results))
"""

    env = _gpu_env(case["gpu_ids"])
    print(f"  [real] benchmarking {n} requests × {n_repeat} runs ...")
    result = subprocess.run(
        [VLLM_PYTHON, "-c", script],
        env=env, capture_output=True, text=True, timeout=1200)

    if result.returncode != 0:
        print(f"  [real] FAILED (rc={result.returncode})")
        print(result.stderr[-1000:])
        return False

    for line in result.stdout.splitlines():
        if line.startswith("REAL_RESULTS:"):
            runs = json.loads(line[len("REAL_RESULTS:"):])
            wall_ms_list = [r["wall_ns"] / 1e6 for r in runs]
            median_ms = sorted(wall_ms_list)[len(wall_ms_list) // 2]
            real_data = {
                "runs": runs,
                "median_wall_ms": median_ms,
                "num_requests": n,
                "prompt_len": plen,
                "max_tokens": max_tok,
            }
            out_path = case_dir / "real_wallclock.json"
            with open(out_path, "w") as f:
                json.dump(real_data, f, indent=2)
            print(f"  [real] median wall: {median_ms:.2f}ms "
                  f"({len(runs)} runs: {[f'{w:.1f}' for w in wall_ms_list]})")
            return True

    print("  [real] FAILED — no REAL_RESULTS in output")
    return False


# ── Serving mode helpers ────────────────────────────────────────

def _generate_arrival_schedule(case, case_dir):
    """Generate and save a deterministic Poisson arrival schedule."""
    import numpy as np
    qps = case["qps"]
    n = case["num_requests"]
    seed = case.get("poisson_seed", 42)
    rng = np.random.RandomState(seed)
    intervals = rng.exponential(1.0 / qps, size=n)
    arrivals = [0.0]
    for gap in intervals[1:]:
        arrivals.append(arrivals[-1] + gap)
    schedule = {
        "qps": qps,
        "seed": seed,
        "num_requests": n,
        "arrival_times_s": arrivals,
    }
    path = case_dir / "arrival_schedule.json"
    with open(path, "w") as f:
        json.dump(schedule, f, indent=2)
    print(f"  [schedule] {n} requests, QPS={qps}, "
          f"span={arrivals[-1]:.2f}s → {path.name}")
    return path


def _is_serving_case(case):
    return "qps" in case and case["qps"] > 0


def run_serving_nsys_capture(case, case_dir):
    """Combined nsys+plugin capture for a serving case (Poisson arrival)."""
    import sqlite3
    from collections import defaultdict

    case_dir.mkdir(parents=True, exist_ok=True)
    nsys_output = str(case_dir / "nsys_profile")
    for f in case_dir.glob("nsys_profile*"):
        f.unlink()

    schedule_path = case_dir / "arrival_schedule.json"
    if not schedule_path.exists():
        _generate_arrival_schedule(case, case_dir)

    mml = _max_model_len(case)
    trace_out = str(case_dir / "plugin_trace.json")
    ep = case.get("expert_parallel_size", 1)

    inner_cmd = [
        VLLM_PYTHON, str(ROOT / "scripts" / "serve_trace" / "trace_serving.py"),
        "--mode", "plugin",
        "--nsys-bracket",
        "--qps", str(case["qps"]),
        "--schedule-file", str(schedule_path),
        "--model", case["model"],
        "--num-requests", str(case["num_requests"]),
        "--tensor-parallel-size", str(case["tensor_parallel_size"]),
        "--expert-parallel-size", str(ep),
        "--seq-len", str(mml),
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", str(mml),
        "--trace-output", trace_out,
        "--timeout", "1200",
        "--seed", str(case.get("sample_seed", case.get("poisson_seed", 42))),
    ]
    if _is_sharegpt_case(case):
        inner_cmd += ["--dataset", case["dataset"]]
    else:
        inner_cmd += [
            "--prompt-len", str(case["prompt_len"]),
            "--max-tokens", str(case["max_tokens"]),
        ]

    cmd = [
        NSYS, "profile",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--trace=cuda,nvtx",
        "--export=sqlite",
        "-f", "true",
        "-o", nsys_output,
    ] + inner_cmd

    env = _gpu_env(case["gpu_ids"])
    n = case["num_requests"]
    qps = case["qps"]
    print(f"  [nsys+plugin] serving: {n} requests, QPS={qps} ...")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)

    partial = False
    if result.returncode != 0:
        print(f"  [nsys+plugin] inner rc={result.returncode} (operator capture may have failed)")
        for line in result.stderr[-2000:].splitlines():
            if line.strip():
                print(f"    {line.strip()}")
        partial = True

    wall_ms = None
    for line in result.stdout.splitlines():
        if line.startswith("WALL_MS:"):
            wall_ms = float(line.split(":")[1])
        if line.startswith("[serving]"):
            print(f"  {line}")

    sqlite_path = case_dir / "nsys_profile.sqlite"
    if not sqlite_path.exists():
        nsys_rep = case_dir / "nsys_profile.nsys-rep"
        if nsys_rep.exists():
            subprocess.run([NSYS, "export", "--type=sqlite", "-o",
                            str(sqlite_path), str(nsys_rep)],
                           capture_output=True, timeout=180)

    if not sqlite_path.exists():
        print(f"  [nsys+plugin] FAILED — no sqlite")
        return False

    conn = sqlite3.connect(str(sqlite_path))
    rows = conn.execute("""
        SELECT s.value as name, k.start, k.end, k.end - k.start as dur,
               k.deviceId as dev
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """).fetchall()

    # Full query with grid/block/smem for G4 comparison
    rows_full = conn.execute("""
        SELECT s.value as name, k.start, k.end, k.end - k.start as dur,
               k.deviceId as dev,
               k.gridX, k.gridY, k.gridZ,
               k.blockX, k.blockY, k.blockZ,
               k.staticSharedMemory + k.dynamicSharedMemory as smem
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """).fetchall()
    conn.close()

    by_dev = defaultdict(list)
    for name, start, end, dur, dev in rows:
        by_dev[dev].append({"name": name, "start": start, "end": end,
                            "dur_ns": dur})

    # Full kernel info for G2 comparison (grid/block/smem). ts_ns (kernel start)
    # is carried so G2 can align the captured stream to the plugin trace's
    # iteration boundaries by timestamp — see the mctracer path for rationale.
    by_dev_full = defaultdict(list)
    for name, start, end, dur, dev, gx, gy, gz, bx, by_, bz, smem in rows_full:
        by_dev_full[dev].append({
            "name": name, "dur_ns": dur,
            "grid": [gx, gy, gz], "block": [bx, by_, bz], "smem": smem or 0,
            "ts_ns": start,
        })

    if not by_dev:
        print(f"  [nsys+plugin] FAILED — no devices found")
        return False

    _write_nsys_kernels(case_dir, by_dev, wall_ms=wall_ms,
                        by_dev_full=by_dev_full, label="nsys+plugin")
    return True


def _parse_mctracer_json(mctracer_json_path, case_dir):
    """Parse mcTracer JSON output into kernel dicts grouped by device.

    If a scheduler sidecar exists (warmup-reset was used), filters out
    warmup kernels by keeping only those whose ts >= sidecar first step.
    Returns dict[device_id -> kernel_list] or None on failure.
    """
    from collections import defaultdict

    if not Path(mctracer_json_path).exists():
        print(f"  [mctracer] ERROR: no JSON output at {mctracer_json_path}")
        return None

    try:
        with open(mctracer_json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  [mctracer] ERROR reading JSON: {e}")
        return None

    events = data.get("traceEvents", [])
    gpu_events = [e for e in events if e.get("cat") == "0"]

    if not gpu_events:
        print(f"  [mctracer] WARNING: no GPU kernels found (cat=0)")

    sidecar_path = case_dir / "plugin_trace.json_scheduler.jsonl"
    if sidecar_path.exists():
        with open(sidecar_path) as f:
            first_line = f.readline().strip()
            if first_line:
                first_step = json.loads(first_line)
                cutoff_ns = first_step["ts_us"] * 1000
                n_before = len(gpu_events)
                gpu_events = [e for e in gpu_events if e.get("ts", 0) >= cutoff_ns]
                n_dropped = n_before - len(gpu_events)
                if n_dropped > 0:
                    print(f"  [mctracer] filtered {n_dropped} warmup kernels "
                          f"(kept {len(gpu_events)})")

    by_dev = defaultdict(list)
    for e in gpu_events:
        dev = e.get("args", {}).get("device_id", 0)
        args = e.get("args", {})
        g = args.get("grid", {})
        b = args.get("block", {})
        mem = args.get("mem", {})
        smem = mem.get("dynamic_shared", 0) + mem.get("static_shared", 0)

        by_dev[dev].append({
            "name": e.get("name", "unknown"),
            "dur_ns": int(e.get("dur", 0)),
            "grid": [g.get("x", 1), g.get("y", 1), g.get("z", 1)],
            "block": [b.get("x", 1), b.get("y", 1), b.get("z", 1)],
            "smem": smem,
            # Capture wall-clock start (ns) so downstream G2 can align the
            # captured kernel stream to the plugin trace's iteration boundaries
            # by timestamp. Without this the two streams can only be compared by
            # raw per-type counts, which breaks when the capture window and the
            # plugin trace cover different iteration ranges (mctracer's warmup
            # cutoff can start mid-iteration → count mismatch → G2 aborts).
            "ts_ns": int(e.get("ts", 0)),
        })

    if not by_dev:
        print(f"  [mctracer] ERROR: no devices found in GPU events")
        return None

    return dict(by_dev)


def _parse_mctracer_multi(json_files, case_dir):
    """Parse all mcTracer worker JSONs and write per-rank nsys_kernels.

    mcTracer generates one JSON per process. For TP/EP>1, multiple worker
    processes each trace one GPU.  We parse all large files (>1MB) and
    treat each as a separate rank.  Returns True on success.
    """
    MIN_SIZE = 1 * 1024 * 1024  # 1 MB — skip device-query-only files
    worker_files = sorted(
        [f for f in json_files if Path(f).stat().st_size > MIN_SIZE],
        key=lambda f: Path(f).name,
    )
    if not worker_files:
        worker_files = [max(json_files, key=lambda f: Path(f).stat().st_size)]

    merged_by_rank = {}
    for rank_idx, fpath in enumerate(worker_files):
        print(f"  [mctracer] parsing {Path(fpath).name} as rank{rank_idx} "
              f"({Path(fpath).stat().st_size / 1024 / 1024:.0f} MB)")
        by_dev = _parse_mctracer_json(fpath, case_dir)
        if by_dev is None:
            continue
        devs = sorted(by_dev.keys())
        merged_by_rank[rank_idx] = by_dev[devs[0]]

    if not merged_by_rank:
        print(f"  [mctracer] ERROR: no valid worker traces")
        return False

    _write_nsys_kernels(case_dir, merged_by_rank, by_dev_full=merged_by_rank,
                        label="mctracer")
    return True


def run_mctracer_capture(case, case_dir, hw_config):
    """Capture GPU profiling with mcTracer (offline batch mode)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    mctracer_out_dir = case_dir / "mctracer_out"
    if mctracer_out_dir.exists():
        shutil.rmtree(mctracer_out_dir)
    mctracer_out_dir.mkdir(parents=True, exist_ok=True)

    mml = _max_model_len(case)
    trace_out = str(case_dir / "plugin_trace.json")
    ep = case.get("expert_parallel_size", 1)

    # Build inner command (same as nsys version, but NO --nsys-bracket for mcTracer)
    inner_cmd = [
        VLLM_PYTHON, str(ROOT / "scripts" / "serve_trace" / "trace_serving.py"),
        "--mode", "plugin",
        "--hw-config", hw_config,
        "--warmup-reset",
        "--model", case["model"],
        "--num-requests", str(case["num_requests"]),
        "--tensor-parallel-size", str(case["tensor_parallel_size"]),
        "--expert-parallel-size", str(ep),
        "--seq-len", str(mml),
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", str(mml),
        "--trace-output", trace_out,
        "--timeout", "1800",
        "--seed", str(case.get("sample_seed", case.get("poisson_seed", 42))),
    ]
    if _is_sharegpt_case(case):
        inner_cmd += ["--dataset", case["dataset"]]
    else:
        inner_cmd += [
            "--prompt-len", str(case["prompt_len"]),
            "--max-tokens", str(case["max_tokens"]),
        ]

    # Wrap with mcTracer — use relative path for --odname because mcTracer
    # prepends the CWD to the path regardless of whether it's absolute
    mctracer_odname = os.path.relpath(mctracer_out_dir, ROOT)
    cmd = [
        MCTRACER, "--mctx",
        "--odname", mctracer_odname,
        "--name", "mctracer",
    ] + inner_cmd

    env = _gpu_env(case["gpu_ids"])
    n = case["num_requests"]
    if _is_sharegpt_case(case):
        print(f"  [mctracer+plugin] ShareGPT: {n} requests ...")
    else:
        plen = case["prompt_len"]
        max_tok = case["max_tokens"]
        print(f"  [mctracer+plugin] profiling {n} requests, p={plen}, d={max_tok} ...")

    stdout_log = case_dir / "mctracer_stdout.log"
    stderr_log = case_dir / "mctracer_stderr.log"
    with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
        result = subprocess.run(cmd, env=env, stdout=fout, stderr=ferr, timeout=2400,
                                cwd=str(ROOT))

    if result.returncode != 0:
        print(f"  [mctracer+plugin] FAILED (rc={result.returncode})")
        with open(stderr_log) as f:
            print(f.read()[-1000:])
        return False

    # Print plugin output lines
    with open(stdout_log) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[plugin]"):
                print(f"  {line}")

    # Find and parse mcTracer JSON — mcTracer generates one JSON per process.
    # The EngineCore worker runs the actual GPU kernels and produces the
    # largest file; pick that one.
    # mcTracer may need time to flush JSON after the process exits.
    json_files = []
    for attempt in range(30):
        json_files = sorted(glob.glob(str(mctracer_out_dir / "mctracer-*.json")))
        if json_files:
            break
        time.sleep(2)

    if not json_files:
        print(f"  [mctracer] ERROR: no JSON files in {mctracer_out_dir} after waiting")
        return False

    return _parse_mctracer_multi(json_files, case_dir)


def run_serving_mctracer_capture(case, case_dir, hw_config):
    """Capture GPU profiling with mcTracer (Poisson serving mode)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    mctracer_out_dir = case_dir / "mctracer_out"
    if mctracer_out_dir.exists():
        shutil.rmtree(mctracer_out_dir)
    mctracer_out_dir.mkdir(parents=True, exist_ok=True)

    schedule_path = case_dir / "arrival_schedule.json"
    if not schedule_path.exists():
        _generate_arrival_schedule(case, case_dir)

    mml = _max_model_len(case)
    trace_out = str(case_dir / "plugin_trace.json")
    ep = case.get("expert_parallel_size", 1)

    # Build inner command (same as serving nsys version, with --qps and --schedule-file)
    inner_cmd = [
        VLLM_PYTHON, str(ROOT / "scripts" / "serve_trace" / "trace_serving.py"),
        "--mode", "plugin",
        "--hw-config", hw_config,
        "--warmup-reset",
        "--qps", str(case["qps"]),
        "--schedule-file", str(schedule_path),
        "--model", case["model"],
        "--num-requests", str(case["num_requests"]),
        "--tensor-parallel-size", str(case["tensor_parallel_size"]),
        "--expert-parallel-size", str(ep),
        "--seq-len", str(mml),
        "--enable-chunked-prefill",
        "--max-num-batched-tokens", str(mml),
        "--trace-output", trace_out,
        "--timeout", "1200",
        "--seed", str(case.get("sample_seed", case.get("poisson_seed", 42))),
    ]
    if _is_sharegpt_case(case):
        inner_cmd += ["--dataset", case["dataset"]]
    else:
        inner_cmd += [
            "--prompt-len", str(case["prompt_len"]),
            "--max-tokens", str(case["max_tokens"]),
        ]

    # Wrap with mcTracer — use relative path for --odname because mcTracer
    # prepends the CWD to the path regardless of whether it's absolute
    mctracer_odname = os.path.relpath(mctracer_out_dir, ROOT)
    cmd = [
        MCTRACER, "--mctx",
        "--odname", mctracer_odname,
        "--name", "mctracer",
    ] + inner_cmd

    env = _gpu_env(case["gpu_ids"])
    n = case["num_requests"]
    qps = case["qps"]
    print(f"  [mctracer+plugin] serving: {n} requests, QPS={qps} ...")

    stdout_log = case_dir / "mctracer_stdout.log"
    stderr_log = case_dir / "mctracer_stderr.log"
    with open(stdout_log, "w") as fout, open(stderr_log, "w") as ferr:
        result = subprocess.run(cmd, env=env, stdout=fout, stderr=ferr, timeout=1800,
                                cwd=str(ROOT))

    if result.returncode != 0:
        print(f"  [mctracer+plugin] FAILED (rc={result.returncode})")
        with open(stderr_log) as f:
            for line in f.read()[-3000:].splitlines():
                if line.strip():
                    print(f"    {line.strip()}")
        return False

    # Print plugin output lines
    with open(stdout_log) as f:
        for line in f:
            line = line.strip()
            if line.startswith("[serving]"):
                print(f"  {line}")

    # Find and parse mcTracer JSON — pick largest (EngineCore worker).
    # mcTracer may need time to flush JSON after the process exits.
    json_files = []
    for attempt in range(30):
        json_files = sorted(glob.glob(str(mctracer_out_dir / "mctracer-*.json")))
        if json_files:
            break
        time.sleep(2)

    if not json_files:
        print(f"  [mctracer] ERROR: no JSON files in {mctracer_out_dir} after waiting")
        return False

    return _parse_mctracer_multi(json_files, case_dir)


def run_serving_real_benchmark(case, case_dir, n_repeat=3):
    """Clean serving run with Poisson arrival for real per-request metrics."""
    schedule_path = case_dir / "arrival_schedule.json"
    if not schedule_path.exists():
        _generate_arrival_schedule(case, case_dir)

    n = case["num_requests"]
    plen = case["prompt_len"]
    max_tok = case["max_tokens"]
    tp = case["tensor_parallel_size"]
    ep = case.get("expert_parallel_size", 1)
    enable_ep = ep > 1
    mml = _max_model_len(case)

    with open(schedule_path) as f:
        schedule_data = json.load(f)
    arrivals_json = json.dumps(schedule_data["arrival_times_s"])

    script = f"""
import asyncio, json, time
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

ea = AsyncEngineArgs(
    model='{case["model"]}', dtype='float16', gpu_memory_utilization=0.85,
    enforce_eager=True, max_model_len={mml},
    trust_remote_code=True,
    tensor_parallel_size={tp}, pipeline_parallel_size=1,
    enable_expert_parallel={enable_ep},
    enable_chunked_prefill=True, max_num_batched_tokens={mml})
engine = AsyncLLMEngine.from_engine_args(ea)
params = SamplingParams(temperature=0.0, max_tokens={max_tok},
                        min_tokens={max_tok}, ignore_eos=True)
prompts = [' '.join(['hello'] * {plen}) for _ in range({n})]
arrivals = {arrivals_json}

async def run():
    # Warmup
    wp = SamplingParams(temperature=0.0, max_tokens=1)
    async for _ in engine.generate("hello " * 64, wp, request_id="warmup"):
        pass

    results = {{}}
    t0 = time.perf_counter()

    async def submit(idx, arrival_s, prompt):
        delay = arrival_s - (time.perf_counter() - t0)
        if delay > 0:
            await asyncio.sleep(delay)
        send_t = time.perf_counter()
        first_t = None
        n_tok = 0
        async for out in engine.generate(prompt, params, request_id=f"req_{{idx}}"):
            if first_t is None and out.outputs and len(out.outputs[0].token_ids) >= 1:
                first_t = time.perf_counter()
            if out.outputs:
                n_tok = len(out.outputs[0].token_ids)
        end_t = time.perf_counter()
        results[idx] = {{
            "request_id": f"req_{{idx}}",
            "arrival_s": arrival_s,
            "ttft_ms": (first_t - send_t) * 1000 if first_t else None,
            "e2e_ms": (end_t - send_t) * 1000,
            "n_output_tokens": n_tok,
            "tpot_ms": ((end_t - first_t) * 1000 / (n_tok - 1)) if n_tok > 1 and first_t else None,
        }}

    tasks = [submit(i, arrivals[i], prompts[i]) for i in range(len(prompts))]
    await asyncio.gather(*tasks)
    wall_s = time.perf_counter() - t0
    return results, wall_s

results, wall_s = asyncio.run(run())
engine.shutdown()
ordered = [results[i] for i in sorted(results)]
out = {{
    "wall_s": wall_s,
    "throughput_rps": len(ordered) / wall_s if wall_s > 0 else 0,
    "num_requests": len(ordered),
    "per_request": ordered,
}}
print("SERVING_RESULTS:" + json.dumps(out))
"""

    env = _gpu_env(case["gpu_ids"])
    print(f"  [real] serving benchmark: {n} requests, QPS={case['qps']} ...")
    result = subprocess.run(
        [VLLM_PYTHON, "-c", script],
        env=env, capture_output=True, text=True, timeout=1800)

    if result.returncode != 0:
        print(f"  [real] FAILED (rc={result.returncode})")
        print(result.stderr[-1000:])
        return False

    for line in result.stdout.splitlines():
        if line.startswith("SERVING_RESULTS:"):
            data = json.loads(line[len("SERVING_RESULTS:"):])
            out_path = case_dir / "real_metrics.json"
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            avg_ttft = sum(r["ttft_ms"] or 0 for r in data["per_request"]) / n
            avg_tpot = sum(r["tpot_ms"] or 0 for r in data["per_request"]
                           if r["tpot_ms"]) / max(1, sum(1 for r in data["per_request"]
                                                          if r["tpot_ms"]))
            print(f"  [real] {data['throughput_rps']:.2f} req/s, "
                  f"avg TTFT={avg_ttft:.1f}ms, avg TPOT={avg_tpot:.2f}ms")
            return True

    print("  [real] FAILED — no SERVING_RESULTS in output")
    return False


# ── Per-step CPU/GPU extraction helpers ─────────────────────────

def _detect_step_boundaries(kernels, n_steps):
    """Find n_steps groups in a kernel list by inter-kernel gap analysis.

    Returns list of (start_idx, end_idx) tuples, or None if detection fails.

    FALLBACK only. The primary path is _partition_by_ts_boundaries (timestamp
    alignment to the scheduler sidecar), which is robust to the intra-step gaps
    that break gap detection on mixed chunked-prefill steps. See
    serving-ttft-nsys-step-boundary-bug memory.
    """
    if len(kernels) < 2 or n_steps < 1:
        return None

    gaps = [kernels[i]["start"] - kernels[i - 1]["end"]
            for i in range(1, len(kernels))]

    sorted_gaps = sorted(gaps, reverse=True)
    target_boundaries = n_steps - 1

    if target_boundaries > len(sorted_gaps):
        return None

    if target_boundaries == 0:
        boundaries = [0, len(kernels)]
    else:
        # Pick exactly `target_boundaries` largest gaps by INDEX, not by a value
        # threshold. A value threshold breaks on ties (two gaps equal to the
        # boundary gap → count off by one → whole extraction dropped, as seen on
        # Qwen3-4B qps4 where gaps[1497]==gaps[1498]). Sorting gap indices by
        # (-gap) and taking the first N is tie-robust and always yields N cuts.
        idx_by_gap = sorted(range(len(gaps)), key=lambda i: -gaps[i])
        cut_after = sorted(idx_by_gap[:target_boundaries])
        boundaries = [0] + [i + 1 for i in cut_after] + [len(kernels)]

    if len(boundaries) - 1 != n_steps:
        return None

    return [(boundaries[s], boundaries[s + 1]) for s in range(n_steps)]


def _ts_boundaries_from_sidecar(sidecar_ts_us, nsys0_ns, last_kernel_end_ns):
    """Build nsys-time step boundaries from scheduler sidecar timestamps.

    sidecar_ts_us: epoch-µs step timestamps (one per ACTIVE sidecar step, in
    order). nsys uses ns-since-capture-start. Both measure wall time, so the
    DELTA (ts[i]-ts[0]) in µs × 1000 = the step-boundary offset in ns. The
    capture is cudaProfilerApi-bracketed to start right after warmup+file-reset
    (trace_serving --nsys-bracket), so the FIRST nsys kernel ≈ the first active
    sidecar step's first kernel — anchor nsys0 there. Constant scheduler→GPU
    dispatch lag then cancels: each step window [B_i, B_{i+1}) captures exactly
    that step's kernels (verified: step-2 gpu matches gap detection, big mixed
    prefill steps recover their real ~78ms instead of a ~14ms fragment).

    Returns a list of n_steps+1 nsys-time boundaries (ns). The last boundary is
    past the final kernel so the last step captures its full tail.
    """
    ts0 = sidecar_ts_us[0]
    bounds = [nsys0_ns + (t - ts0) * 1000.0 for t in sidecar_ts_us]
    # Tail boundary: strictly after the last kernel so step n-1 keeps its tail.
    tail = max(last_kernel_end_ns + 1.0, bounds[-1] + 1.0)
    bounds.append(tail)
    return bounds


def _partition_by_ts_boundaries(kernels, boundaries):
    """Partition a start-sorted kernel list into len(boundaries)-1 steps by
    nsys-time boundaries. Returns list of {gpu_ns, cpu_ns} per step.

    cpu_ns (GPU-idle) mirrors _extract_nsys_per_step's original definition:
    intra_step_cpu (wall_span − gpu_sum) + inter_step_gap (this step's first
    kernel start − previous step's last kernel end), both max(0, ·). The serial
    G3/G5 assembly sums gpu_pred + this idle → exact wall when total GPU is
    conserved.
    """
    n_steps = len(boundaries) - 1
    steps = []
    ki = 0
    prev_end = None
    n = len(kernels)
    for s in range(n_steps):
        lo, hi = boundaries[s], boundaries[s + 1]
        while ki < n and kernels[ki]["start"] < lo:
            ki += 1
        j = ki
        while j < n and kernels[j]["start"] < hi:
            j += 1
        step_k = kernels[ki:j]
        if step_k:
            gpu_sum_ns = sum(k["dur_ns"] for k in step_k)
            wall_span_ns = step_k[-1]["end"] - step_k[0]["start"]
            intra = max(0.0, wall_span_ns - gpu_sum_ns)
            if prev_end is not None:
                inter = max(0.0, step_k[0]["start"] - prev_end)
            else:
                inter = 0.0
            prev_end = step_k[-1]["end"]
        else:
            gpu_sum_ns = 0.0
            intra = 0.0
            inter = 0.0
        steps.append({"gpu_ns": gpu_sum_ns, "cpu_ns": intra + inter})
        ki = j
    return steps


def _extract_nsys_per_step(case_dir, n_steps, sidecar_ts_us=None):
    """Extract per-step GPU and CPU times from nsys_profile.sqlite.

    For multi-device cases, computes per-step GPU time for each device
    and takes the max (bottleneck rank) per step.

    sidecar_ts_us: when provided (list of n_steps active-sidecar step
    timestamps in µs), partition each device's kernels by TIMESTAMP boundaries
    derived from the sidecar (the real per-step windows). This is the correct
    path: nsys has no step_id, and the previous gap-based detection
    (_detect_step_boundaries) splits long mixed chunked-prefill steps at
    intra-step gaps, fragmenting their GPU time and inflating nsys_idle → TTFT
    over-prediction (serving-ttft-nsys-step-boundary-bug). When None, fall back
    to gap detection (offline / no-sidecar cases).
    """
    import sqlite3
    from collections import defaultdict

    sqlite_path = case_dir / "nsys_profile.sqlite"
    if not sqlite_path.exists():
        return None

    conn = sqlite3.connect(str(sqlite_path))
    rows = conn.execute("""
        SELECT s.value as name, k.start, k.end, k.end - k.start as dur,
               k.deviceId as dev
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        ORDER BY k.start
    """).fetchall()
    conn.close()

    by_dev = defaultdict(list)
    for name, start, end, dur, dev in rows:
        by_dev[dev].append({"start": start, "end": end, "dur_ns": dur})
    devs = sorted(by_dev)

    # Timestamp-alignment path: build global nsys-time boundaries from the
    # sidecar step timestamps, then partition each device's kernel stream.
    use_ts = (sidecar_ts_us is not None
              and len(sidecar_ts_us) >= n_steps
              and n_steps >= 1)
    if use_ts:
        # If the predictor dropped leading warmup iterations, sidecar_ts_us
        # may be longer than n_steps — take the LAST n_steps (front-dropped).
        ts_us = sidecar_ts_us[-n_steps:]
        nsys0_ns = min(by_dev[d][0]["start"] for d in devs if by_dev[d])
        last_end_ns = max(by_dev[d][-1]["end"] for d in devs if by_dev[d])
        boundaries = _ts_boundaries_from_sidecar(ts_us, nsys0_ns, last_end_ns)
        per_dev_per_step = {}
        for dev in devs:
            per_dev_per_step[dev] = _partition_by_ts_boundaries(
                by_dev[dev], boundaries)
        if any(len(v) != n_steps for v in per_dev_per_step.values()):
            use_ts = False  # safety; shouldn't happen
            per_dev_per_step = {}

    if not use_ts:
        # Fallback: gap-based boundary detection (offline / no sidecar).
        per_dev_per_step = {}
        for dev in devs:
            kernels = by_dev[dev]
            bounds = _detect_step_boundaries(kernels, n_steps)
            if bounds is None:
                if dev == devs[0]:
                    print(f"  [nsys-step] WARNING: boundary detection failed on dev{dev}")
                continue
            steps = []
            for s, (si, ei) in enumerate(bounds):
                step_k = kernels[si:ei]
                gpu_sum_ns = sum(k["dur_ns"] for k in step_k)
                wall_span_ns = step_k[-1]["end"] - step_k[0]["start"]
                intra_step_cpu_ns = wall_span_ns - gpu_sum_ns
                if s == 0:
                    inter_step_gap_ns = 0
                else:
                    prev_end = kernels[bounds[s][0] - 1]["end"]
                    inter_step_gap_ns = step_k[0]["start"] - prev_end
                cpu_ns = max(0, intra_step_cpu_ns) + max(0, inter_step_gap_ns)
                steps.append({"gpu_ns": gpu_sum_ns, "cpu_ns": cpu_ns})
            per_dev_per_step[dev] = steps

    if not per_dev_per_step:
        return None

    # Return per-device GPU and gap per step. The wall assembly (in
    # run_prediction) pairs each rank's predicted compute with THAT rank's gap
    # before taking max-over-ranks — max(gpu) and max(gap) must NOT be taken
    # independently (that double-counts when ranks alternate leading; verified
    # to break TP2 low-QPS). See docs/tp_allreduce_spinwait_analysis.md.
    #   nsys_gpu_ms  = max-over-devices GPU busy (for the G1 pure-GPU reference)
    #   per_dev_gap  = {dev: gap_ns} so the assembler can pair gap_r with rank r
    # For single-device cases (dense TP1 / EP single-card) there is one dev, so
    # everything degenerates to the previous single-rank behavior (zero regression).
    per_step = []
    active_devs = sorted(per_dev_per_step.keys())
    for s in range(n_steps):
        gpu_ns = max(per_dev_per_step[d][s]["gpu_ns"] for d in active_devs)
        per_dev_gap = {d: per_dev_per_step[d][s]["cpu_ns"] for d in active_devs}
        # Back-compat scalar: gap of the GPU-bottleneck device (used only when
        # the assembler has no per-rank compute to pair with, e.g. missing pred).
        bn_dev = max(active_devs, key=lambda d: per_dev_per_step[d][s]["gpu_ns"])
        cpu_ns = per_dev_gap[bn_dev]
        per_step.append({
            "nsys_gpu_ns": gpu_ns,
            "nsys_gpu_ms": gpu_ns / 1e6,
            "nsys_cpu_ns": cpu_ns,
            "nsys_cpu_ms": cpu_ns / 1e6,
            "per_dev_gap_ns": per_dev_gap,
        })

    return per_step


def _extract_mctracer_per_step(case_dir, n_steps, sidecar_ts_us=None):
    """Extract per-step GPU and CPU(idle) times from the mcTracer capture
    (nsys_kernels_full.json), the C500 equivalent of _extract_nsys_per_step.

    C500 doesn't run nsys (no nsys_profile.sqlite) — it captures via mcTracer,
    which writes nsys_kernels_full.json with per-rank kernel lists carrying
    REAL wall timestamps (ts_ns) + durations. mcTracer's ts_ns is a monotonic
    wall clock (same physical time as the scheduler sidecar ts_us), so the
    A800 timestamp-alignment approach applies directly: build step boundaries
    from sidecar ts deltas, partition each rank's kernel stream. Unlike nsys,
    mcTracer has no gap-detection bug (the boundaries come from the scheduler,
    not inferred from inter-kernel gaps), so this is the reliable path.

    Reuses _ts_boundaries_from_sidecar + _partition_by_ts_boundaries (same as
    the nsys path). Returns the same per-step structure ({nsys_gpu_ns,
    nsys_cpu_ns, per_dev_gap_ns}) so run_prediction's serial assembly
    (gpu_pred + idle) engages for C500 instead of the max(gpu, cpu) fallback.

    sidecar_ts_us: list of n_steps active-sidecar step timestamps (epoch-µs).
    When None/short, returns None (caller falls back to max).
    """
    import json as _json
    from collections import defaultdict

    use_ts = (sidecar_ts_us is not None
              and len(sidecar_ts_us) >= n_steps
              and n_steps >= 1)
    if not use_ts:
        return None

    kj_path = case_dir / "nsys_kernels_full.json"
    if not kj_path.exists():
        return None

    # Stream-parse only the per_rank section (the file can be multi-GB; avoid
    # loading the flat top-level `kernels` list).
    try:
        with open(kj_path, "r") as f:
            data = _json.load(f)
    except Exception:
        return None
    per_rank = data.get("per_rank") or {}
    if not per_rank:
        return None

    # Build per-rank kernel lists in the {start, end, dur_ns} shape that
    # _partition_by_ts_boundaries expects. Sort by start (ts_ns).
    by_dev = {}
    for dev, rd in per_rank.items():
        ks = rd.get("kernels") if isinstance(rd, dict) else None
        if not ks:
            continue
        kl = []
        for k in ks:
            ts = k.get("ts_ns")
            dur = k.get("dur_ns", 0)
            if ts is None:
                continue
            kl.append({"start": ts, "end": ts + dur, "dur_ns": dur})
        kl.sort(key=lambda x: x["start"])
        if kl:
            by_dev[dev] = kl
    if not by_dev:
        return None

    devs = sorted(by_dev)
    ts_us = sidecar_ts_us[-n_steps:]
    mc0_ns = min(by_dev[d][0]["start"] for d in devs)
    last_end_ns = max(by_dev[d][-1]["end"] for d in devs)
    boundaries = _ts_boundaries_from_sidecar(ts_us, mc0_ns, last_end_ns)

    per_dev_per_step = {}
    for dev in devs:
        per_dev_per_step[dev] = _partition_by_ts_boundaries(by_dev[dev], boundaries)
        if len(per_dev_per_step[dev]) != n_steps:
            return None  # safety; shouldn't happen

    # _partition_by_ts_boundaries already computed {gpu_ns, cpu_ns} per step.
    # Aggregate across devs (max GPU, bottleneck-dev gap) — mirrors
    # _extract_nsys_per_step's post-partition aggregation so the serial
    # assembly (max(gpu_pred, gpu_real) + nsys_idle) engages for C500.
    per_step = []
    active_devs = sorted(per_dev_per_step.keys())
    for s in range(n_steps):
        gpu_ns = max(per_dev_per_step[d][s]["gpu_ns"] for d in active_devs)
        per_dev_gap = {d: per_dev_per_step[d][s]["cpu_ns"] for d in active_devs}
        bn_dev = max(active_devs, key=lambda d: per_dev_per_step[d][s]["gpu_ns"])
        cpu_ns = per_dev_gap[bn_dev]
        per_step.append({
            "nsys_gpu_ns": gpu_ns,
            "nsys_gpu_ms": gpu_ns / 1e6,
            "nsys_cpu_ns": cpu_ns,
            "nsys_cpu_ms": cpu_ns / 1e6,
            "per_dev_gap_ns": per_dev_gap,
        })

    return per_step

def _extract_msprof_per_step(case_dir, n_steps, sidecar_ts_us=None):
    """Extract per-step GPU and idle time from msprof sqlite (910B).

    910B uses msprof (not nsys) → no nsys_profile.sqlite. The per-kernel
    timing lives in the msprof PROF_*/device_*/sqlite/ai_core_op_summary.db
    (task_time table: start_time, duration_time, in ns). Step boundaries use
    sidecar ts_us RELATIVE alignment (same as the nsys/§4 path): the msprof
    device clock and the sidecar host clock have different epochs, but both are
    real-time, so the DELTA ts[i]-ts[0] cancels the offset (anchor the first
    windowed kernel + add relative µs deltas). This replaces inter-kernel gap
    detection, which splits long mixed chunked-prefill steps at their large
    intra-step gaps and bleeds that GPU time into adjacent decode windows
    (prefill GT under-, decode GT over-counted → false §5 prefill over-
    prediction). See serving-ttft-nsys-step-boundary-bug (nsys/§4 fix) and
    910b-msprof-sidecar-ts-alignment (910B §4 idle already uses this relative
    trick). Gap detection (_detect_step_boundaries) remains the fallback when
    no sidecar_ts_us is available.

    For multi-device (TP/EP), each device has its own PROF dir + sqlite;
    per-step GPU = max over devices (bottleneck), per-step idle = wall_span -
    gpu_sum, paired per-device (same as _extract_nsys_per_step).

    Returns the same per_step list format as _extract_nsys_per_step, or None
    if no sqlite.
    """
    import sqlite3

    dbs = sorted(case_dir.glob("msprof/PROF_*/device_*/sqlite/ai_core_op_summary.db"))
    if not dbs:
        return None

    per_dev_per_step = {}
    dev_id = 0
    for db_path in dbs:
        conn = sqlite3.connect(str(db_path))
        # Check if task_type column exists + has device-kernel values. Older
        # msprof captures may lack task_type (all NULL) — fall back to all
        # duration>0 tasks (the gap detection + window filter still remove
        # host/warmup tasks adequately).
        has_task_type = conn.execute(
            "SELECT count(*) FROM pragma_table_info('task_time') "
            "WHERE name='task_type'").fetchone()[0] > 0
        if has_task_type:
            dev_count = conn.execute(
                "SELECT count(*) FROM task_time WHERE task_type "
                "IN ('AI_CORE','MIX_AIC')").fetchone()[0]
        else:
            dev_count = 0
        if dev_count > 0:
            rows = conn.execute("""
                SELECT t.start_time, t.duration_time
                FROM task_time t
                WHERE t.duration_time > 0
                  AND t.task_type IN ('AI_CORE', 'MIX_AIC')
                ORDER BY t.start_time
            """).fetchall()
        else:
            rows = conn.execute("""
                SELECT t.start_time, t.duration_time
                FROM task_time t
                WHERE t.duration_time > 0
                ORDER BY t.start_time
            """).fetchall()
        conn.close()
        if not rows:
            continue
        # Inference-window filter: msprof captures warmup before the serving
        # window. The nsys_kernels.json span_ns is the inference-window length
        # (from parse_msprof_timeline). The msprof clock differs from the
        # sidecar clock, so we can't use gen_start_us directly — instead, take
        # the last span_ns of the msprof AI_CORE range (warmup is at the start).
        nsys_path = case_dir / "nsys_kernels.json"
        if nsys_path.exists():
            with open(nsys_path) as _f:
                _nj = json.load(_f)
            span_ns = _nj.get("span_ns", 0)
            if span_ns > 0:
                last_end = rows[-1][0] + rows[-1][1]
                win_start = last_end - span_ns
                rows = [r for r in rows if r[0] >= win_start]
        # msprof start_time/duration_time are both ns (start_time is epoch-ns,
        # duration_time is ns) — keep as-is.
        kernels = [{"start": st, "end": st + du, "dur_ns": du}
                   for st, du in rows]
        # Step boundaries: prefer sidecar-ts RELATIVE alignment (see docstring);
        # fall back to gap detection when no sidecar ts is available.
        use_ts = (sidecar_ts_us is not None
                  and len(sidecar_ts_us) >= n_steps
                  and len(kernels) >= n_steps)
        if use_ts:
            ts_us = sidecar_ts_us[-n_steps:]
            boundaries = _ts_boundaries_from_sidecar(
                ts_us, kernels[0]["start"], kernels[-1]["end"])
            steps = _partition_by_ts_boundaries(kernels, boundaries)
        else:
            bounds = _detect_step_boundaries(kernels, n_steps)
            if bounds is None:
                print(f"  [msprof-step] WARNING: boundary detection failed on dev{dev_id}")
                dev_id += 1
                continue
            steps = []
            for s, (si, ei) in enumerate(bounds):
                step_k = kernels[si:ei]
                gpu_sum_ns = sum(k["dur_ns"] for k in step_k)
                wall_span_ns = step_k[-1]["end"] - step_k[0]["start"]
                intra_step_cpu_ns = wall_span_ns - gpu_sum_ns
                if s == 0:
                    inter_step_gap_ns = 0
                else:
                    prev_end = kernels[bounds[s][0] - 1]["end"]
                    inter_step_gap_ns = step_k[0]["start"] - prev_end
                cpu_ns = max(0, intra_step_cpu_ns) + max(0, inter_step_gap_ns)
                steps.append({"gpu_ns": gpu_sum_ns, "cpu_ns": cpu_ns})
        per_dev_per_step[str(dev_id)] = steps
        dev_id += 1

    if not per_dev_per_step:
        return None

    per_step = []
    active_devs = sorted(per_dev_per_step.keys())
    for s in range(n_steps):
        gpu_ns = max(per_dev_per_step[d][s]["gpu_ns"] for d in active_devs)
        per_dev_gap = {d: per_dev_per_step[d][s]["cpu_ns"] for d in active_devs}
        bn_dev = max(active_devs, key=lambda d: per_dev_per_step[d][s]["gpu_ns"])
        cpu_ns = per_dev_gap[bn_dev]
        per_step.append({
            "nsys_gpu_ns": gpu_ns,
            "nsys_gpu_ms": gpu_ns / 1e6,
            "nsys_cpu_ns": cpu_ns,
            "nsys_cpu_ms": cpu_ns / 1e6,
            "per_dev_gap_ns": per_dev_gap,
        })
    print(f"  [msprof-step] extracted {n_steps} steps from {len(active_devs)} device(s)")
    return per_step


def _extract_msprof_gpu_by_count(case_dir, pred_counts):
    """§5-only per-step GPU-busy GT, attributed by PROGRAM ORDER (cumulative
    predicted kernel count) instead of the host clock.

    Why a second attribution (the §4 idle/TTFT path keeps host-ts): the host
    scheduler races ahead of the single-stream NPU at high qps. It admits a
    burst of prefill steps within a few host seconds while the device takes far
    longer to execute them, so a prefill step's GPU work runs *inside the host-
    ts windows of later (decode) steps*. Host-ts (sidecar) boundaries then bleed
    that prefill GPU time into the decode bucket → the §5 prefill bucket looks
    hugely over-predicted (+446–636% at qps64) purely as an attribution
    artifact. Program order is immune: msprof kernels are ordered by start_time,
    which on a single stream *is* issue == execution order, and the regen trace
    knows how many kernels each step issued in that same order, so cumulative-
    count boundaries assign every kernel to its issuing step regardless of
    host/device clock skew. (regen_total ≈ msprof_total within ~2%, so this is
    ~1:1 index alignment.)

    Also fixes the window bug the §4 path still carries: the old span_ns filter
    (`last span_ns of kernels`) drops the whole front prefill burst, because
    span_ns runs ~28 s short of the true serving span in every case — win_start
    lands *after* the burst, so prefill steps inherited decode-steady-state
    kernels. Here warmup is removed with the largest inter-kernel gap in the
    first 10 % of kernels (the clean warmup→serving idle gap); everything after
    it is genuine serving busy (warmup contributes ~0 ns).

    910B single-stream (TP1) only — program order is per-device, and all 910B
    §5 cases are TP1. Returns a list of gpu_ms per step (len == len(pred_counts))
    or None when unavailable / not single-device.
    """
    import sqlite3
    import numpy as np

    dbs = sorted(case_dir.glob("msprof/PROF_*/device_*/sqlite/ai_core_op_summary.db"))
    if len(dbs) != 1:
        return None  # multi-device: program order is per-stream; not handled
    conn = sqlite3.connect(str(dbs[0]))
    has_tt = conn.execute(
        "SELECT count(*) FROM pragma_table_info('task_time') "
        "WHERE name='task_type'").fetchone()[0] > 0
    if has_tt and conn.execute("SELECT count(*) FROM task_time WHERE task_type "
                               "IN ('AI_CORE','MIX_AIC')").fetchone()[0] > 0:
        rows = conn.execute(
            "SELECT start_time, duration_time FROM task_time WHERE duration_time>0 "
            "AND task_type IN ('AI_CORE','MIX_AIC') ORDER BY start_time").fetchall()
    else:
        rows = conn.execute(
            "SELECT start_time, duration_time FROM task_time WHERE duration_time>0 "
            "ORDER BY start_time").fetchall()
    conn.close()
    if not rows or not pred_counts:
        return None

    starts = np.array([r[0] for r in rows], dtype=float)
    durs = np.array([r[1] for r in rows], dtype=float)
    ends = starts + durs
    # Warmup→serving boundary: largest inter-kernel idle gap in the first 10 %
    # of kernels (warmup is a small front burst, then a multi-second idle gap
    # before the serving window). Fall back to keeping everything if no clear
    # >1 s gap exists.
    gaps = starts[1:] - ends[:-1]
    lim = max(50, len(rows) // 10)
    gi = int(np.argmax(gaps[:lim]))
    si = gi + 1 if gaps[gi] >= 1e9 else 0
    durs = durs[si:]
    if durs.size == 0:
        return None
    cumdur = np.concatenate([[0.0], np.cumsum(durs)])
    M = durs.size

    counts = np.array(pred_counts, dtype=float)
    ctot = counts.sum()
    if ctot <= 0:
        return None
    # Cumulative-count → msprof-index boundaries (program order).
    cbound = np.concatenate([[0.0], np.cumsum(counts)])
    bidx = np.clip(np.round(cbound / ctot * M).astype(int), 0, M)
    gpu_ms = [(cumdur[bidx[k + 1]] - cumdur[bidx[k]]) / 1e6
              for k in range(len(pred_counts))]
    print(f"  [msprof-s5] program-order GPU GT: {len(gpu_ms)} steps, "
          f"{M} kernels windowed (warmup-drop {si})")
    return gpu_ms


def _extract_plugin_cpu_per_step(case_dir, step_ids):
    """Extract per-step CPU wall time from plugin operator record timestamps.

    plugin_CPU_step_i = last_op_ts[i] - last_op_ts[i-1]  (i > 0)
    plugin_CPU_step_0 = last_op_ts[0] - first_op_ts[0]
    """
    ops_path = None
    for candidate in case_dir.glob("*_operators.jsonl.rank0"):
        ops_path = candidate
        break
    if ops_path is None:
        return None

    from collections import defaultdict
    records_by_step = defaultdict(list)
    with open(ops_path) as f:
        for line in f:
            rec = json.loads(line)
            ts = rec.get("ts_us")
            if ts and ts > 0:
                records_by_step[rec["step_id"]].append(ts)

    per_step = []
    prev_last_ts = None
    for sid in step_ids:
        ts_list = records_by_step.get(sid, [])
        if not ts_list:
            per_step.append({"plugin_cpu_ms": 0.0})
            prev_last_ts = None
            continue

        first_ts = min(ts_list)
        last_ts = max(ts_list)

        if prev_last_ts is None:
            step_wall_us = last_ts - first_ts
        else:
            step_wall_us = last_ts - prev_last_ts

        prev_last_ts = last_ts
        per_step.append({"plugin_cpu_ms": step_wall_us / 1000.0})

    return per_step


def _extract_plugin_intrastep_per_step(case_dir, step_ids):
    """Per-step INTRA-step host wall from plugin operator timestamps.

    intra_step_i = last_op_ts[i] − first_op_ts[i]  (span WITHIN one step only).

    This differs from `_extract_plugin_cpu_per_step` (which measures the
    INTER-step spacing last_op[i]−last_op[i-1], the token-delivery cadence used
    for decode/TPOT). The intra-step span is a step's OWN operator work and
    EXCLUDES the leading inter-step gap (the previous step's async output tail +
    scheduling). That leading gap belongs to the queue/admission wait, not to a
    prefilling request's first-token compute — so first-token latency (TTFT)
    uses the intra-step wall, avoiding a double-count of the inter-step gap
    already covered by the admission term.
    """
    ops_path = None
    for candidate in case_dir.glob("*_operators.jsonl.rank0"):
        ops_path = candidate
        break
    if ops_path is None:
        return None

    from collections import defaultdict
    records_by_step = defaultdict(list)
    with open(ops_path) as f:
        for line in f:
            rec = json.loads(line)
            ts = rec.get("ts_us")
            if ts and ts > 0:
                records_by_step[rec["step_id"]].append(ts)

    per_step = []
    for sid in step_ids:
        ts_list = records_by_step.get(sid, [])
        if not ts_list:
            per_step.append({"plugin_intrastep_ms": 0.0})
            continue
        per_step.append({"plugin_intrastep_ms": (max(ts_list) - min(ts_list)) / 1000.0})
    return per_step


# ── Stage 4: Predict ─────────────────────────────────────────────

def _regen_cache_key(rank_files, tp, ep, hw_config, case_dir=None):
    """Fingerprint that determines the regenerated trace: operator-record file
    contents (size+mtime), parallelism, hw config, the predictor/recipe code
    version, AND the per-case auxiliary LUTs (block_lut.json, sampling_mode.jsonl,
    fill_grid.jsonl) that the regen reads. Any change invalidates the cache."""
    from src.serve_predict.predict_lut import compute_code_version
    import hashlib
    h = hashlib.sha256()
    # Phase 2 (T4-4b): direct pricing is the ONLY path now (legacy synthesis
    # deleted), so the trace is fully determined by code_version + inputs below.
    h.update(f"tp={tp};ep={ep};hw={hw_config};code={compute_code_version()};".encode())
    for rf in rank_files:
        st = rf.stat()
        h.update(f"{rf.name}:{st.st_size}:{int(st.st_mtime)};".encode())
    # Per-case auxiliary LUTs the regen consumes — MUST be in the key or a cache
    # built before the LUT existed (hardcoded Block Dims) would be HIT after the
    # LUT is added (same code_version + rank_files), silently skipping the LUT.
    if case_dir is not None:
        cd = Path(case_dir)
        for aux in ("block_lut.json", "sampling_mode.jsonl", "fill_grid.jsonl",
                    "mask_pipeline.jsonl", "fa_tm.jsonl", "sampler_ic.jsonl"):
            ap = cd / aux
            if ap.exists():
                aps = ap.stat()
                h.update(f"{aux}:{aps.st_size}:{int(aps.st_mtime)};".encode())
    return h.hexdigest()[:16]


def _reconstruct_step_kv_lengths(case_dir) -> tuple:
    """Reconstruct per-step KV cache lengths from the scheduler sidecar for
    decode FA modeling.

    The scheduler sidecar gives, per step, each scheduled request's NEW token
    count (prefill chunk or 1 decode token). The FA kernel attends over each
    request's FULL context (cumulative tokens). Returns two maps keyed by
    step_id (iter_id):
      step_kv_max  : max context across the step's requests. Used for the FA
        grid / LUT key / softmax task length (the longest request).
      step_kv_mean : MEAN context = Σ context / #requests. Used (with B=varlen_b)
        for the Ascend decode FA transfer term, whose real work is the TOTAL
        KV-gather across the batch Σ_req ceil(KV_req/16) ≈ B·ceil(mean/16). The
        old model used B=1·ceil(max/16) → ~9× under-count of the batched gather
        (→ serving decode FA −48%). Both empty if the sidecar is absent.
    """
    sidecar = case_dir / "plugin_trace.json_scheduler.jsonl"
    if not sidecar.exists():
        return {}, {}
    import json
    step_kv = {}
    step_kv_mean = {}
    req_ctx = {}   # req_id -> cumulative context tokens
    try:
        with open(sidecar) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                iter_id = rec.get("iter_id")
                scheduled = rec.get("scheduled") or {}
                if not scheduled or iter_id is None:
                    continue
                # Update each request's context with its NEW tokens this step,
                # THEN take the max/mean (FA attends over the post-append context).
                local_max = 0
                local_sum = 0
                local_n = 0
                for rid, tok in scheduled.items():
                    tok = int(tok)
                    req_ctx[rid] = req_ctx.get(rid, 0) + tok
                    local_sum += req_ctx[rid]
                    local_n += 1
                    if req_ctx[rid] > local_max:
                        local_max = req_ctx[rid]
                # Forget finished requests (context no longer grows / not in batch).
                for rid in rec.get("finished") or []:
                    req_ctx.pop(rid, None)
                step_kv[iter_id] = local_max
                step_kv_mean[iter_id] = local_sum / local_n if local_n else local_max
    except (OSError, json.JSONDecodeError):
        return {}, {}
    return step_kv, step_kv_mean


def _regenerate_trace_with_step_id(case_dir, model_config, tp, ep, hw_config):
    """Regenerate trace from operator records to get step_id field.

    Reads ALL rank files (*_operators.jsonl.rank*) and returns a dict
    mapping rank_id -> trace list.  Falls back to a single rank0 if only
    one rank file exists.

    Preserves real perf_counter timestamps from the plugin capture so
    downstream CPU overhead computation works (inter-kernel gap = CPU time).
    Records with ts_us=0 (NCCL placeholders) inherit the last valid timestamp.

    Cross-run cache: the regen (read 600k-2.4M records + recipe expansion) costs
    tens of seconds on big cases. The result is a pure function of the operator
    records + tp/ep/hw + code version, so it is cached (gzip-L1 pickle) under
    case_dir/.regen_cache_<key>.pkl.gz. Cache hit skips the whole rebuild.
    Legacy .json.gz / uncompressed .pkl caches are migrated on first hit.
    """
    from src.serve_trace.operator_records import read_operator_records, OperatorTraceGenerator
    from src.kernel.hardware_spec import HWSpec
    import gzip
    import pickle

    rank_files = sorted(case_dir.glob("*_operators.jsonl.rank*"))
    if not rank_files:
        return None

    cache_key = _regen_cache_key(rank_files, tp, ep, hw_config, case_dir)
    # gzip(level=1)+pickle cache (preferred): the regen result is millions of
    # plain dicts. json.loads dominated the warm load (62s vs ~35s for gzip-L1
    # pickle on the 60GB TP4EP4 case) and plain pickle is 10× the disk (2.1GB
    # vs 84MB). gzip-L1 pickle is the sweet spot — near-pickle speed, near-json.gz
    # size. `.json.gz` (legacy JSON) is migrated on first hit; the transitional
    # uncompressed `.pkl` (if present from an earlier build) is also migrated.
    pkl_path = case_dir / f".regen_cache_{cache_key}.pkl.gz"
    old_pkl_path = case_dir / f".regen_cache_{cache_key}.pkl"
    gz_path = case_dir / f".regen_cache_{cache_key}.json.gz"

    def _migrate_to_pklgz(obj):
        try:
            tmp = pkl_path.with_suffix(".gz.tmp")
            with gzip.open(tmp, "wb", compresslevel=1) as f:
                pickle.dump(obj, f, protocol=5)
            tmp.replace(pkl_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [predict] regen pkl.gz migration skipped ({e})")

    # Caching DISABLED — always rebuild the trace from operator records.
    # Stale cache (keyed on _CODE_FILES but code outside that list changes)
    # caused silent stale-trace hits after predictor/dispatch edits.
    # The rebuild is ~6s/case (parallel); correctness > speed.
    if False and pkl_path.exists():
        try:
            with gzip.open(pkl_path, "rb") as f:
                cached = pickle.load(f)
            print(f"  [predict] regen cache HIT ({pkl_path.name})")
            return cached
        except Exception as e:  # noqa: BLE001
            print(f"  [predict] regen cache unreadable ({e}); rebuilding")
    if old_pkl_path.exists():
        try:
            with open(old_pkl_path, "rb") as f:
                cached = pickle.load(f)
            print(f"  [predict] regen cache HIT ({old_pkl_path.name}; migrating→pkl.gz)")
            _migrate_to_pklgz(cached)
            try:
                old_pkl_path.unlink()  # reclaim the 10×-larger uncompressed file
            except OSError:
                pass
            return cached
        except Exception as e:  # noqa: BLE001
            print(f"  [predict] regen cache unreadable ({e}); rebuilding")
    if gz_path.exists():
        try:
            with gzip.open(gz_path, "rt") as f:
                cached = json.load(f)
            print(f"  [predict] regen cache HIT ({gz_path.name}; migrating→pkl.gz)")
            _migrate_to_pklgz(cached)
            return cached
        except Exception as e:  # noqa: BLE001
            print(f"  [predict] regen cache unreadable ({e}); rebuilding")

    hw = HWSpec(str(ROOT / hw_config))
    # Reconstruct per-step KV cache length (max context tokens in the batch) from
    # the scheduler sidecar so decode attention models the real K-loop length
    # (query_len=1 but the K-loop walks the full KV cache). Prefill steps'
    # reconstructed KV equals their seq_len, so this is a no-op there.
    step_kv, step_kv_mean = _reconstruct_step_kv_lengths(case_dir)
    # Per-step varlen batch size (number of requests scheduled) — the prefill FA
    # grid's gridY = b (the varlen batch, NOT num_splits); reconstruct from the
    # sidecar's `scheduled` dict (len = #requests in the step).
    step_b = {}
    step_decode_count = {}   # per-step count of decode requests (tokens<=1) — the
                             # sampler scatter is sized by the NEXT step's decode
                             # count (continuing seqs, excl. new prefill chunks)
    step_pure_decode = {}
    step_pure_prefill = {}
    step_max_qlen = {}
    step_total = {}
    step_eff_mblocks = {}   # per-step Σ ceil(q_len_i / kBM_grid) — effective (non-padding) M-block CTA count, for the FA early-exit model (see analytic_mirror._attn_kernels)
    step_finished = {}
    step_sampling_mode = {}
    step_fill_grid = {}
    sidecar_path = case_dir / "plugin_trace.json_scheduler.jsonl"
    if sidecar_path.exists():
        try:
            with open(sidecar_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    it = rec.get("iter_id")
                    sch = rec.get("scheduled") or {}
                    if it is not None:
                        step_finished[it] = bool(rec.get("finished"))
                    if it is not None and sch:
                        step_b[it] = len(sch)
                        step_decode_count[it] = sum(
                            1 for t in sch.values() if int(t) <= 1)
                        # Pure-decode step iff every scheduled request advances
                        # exactly 1 token (no prefill chunk). Drives the FA grid
                        # format (decode splitkv vs prefill varlen) — see
                        # OperatorTraceGenerator.set_step_pure_decode.
                        step_pure_decode[it] = all(
                            int(t) <= 1 for t in sch.values())
                        # Pure-prefill iff every scheduled request is a prefill
                        # chunk (all tokens > 1, no decode). vLLM samples only 1
                        # token (the last) for pure prefill, but B tokens for
                        # decode/mixed — drives the sampler gather grid.
                        step_pure_prefill[it] = all(
                            int(t) > 1 for t in sch.values())
                        # Max single-request query length: nsys FA gridX =
                        # ceil(max_req_len / 64), NOT total_tokens. GEMM M keeps
                        # total_tokens (op record seq_len).
                        step_max_qlen[it] = max(int(t) for t in sch.values())
                        # Total tokens = sum of scheduled (KV page Fill grid
                        # uses total, not the op record's decode seq_len=1).
                        step_total[it] = sum(int(t) for t in sch.values())
                        # Effective M-block count: vLLM varlen FA launches the
                        # grid at num_M = ceil(max_qlen / 64) M-blocks PER
                        # request, but compute_attn_1rowblock early-exits any
                        # M-block with m_block*64 >= actual_seqlen_q (flash_fwd
                        # _kernel.h:81). So in a mixed step (1 long prefill +
                        # many decodes) most M-block CTAs are no-ops. The
                        # per-CTA tensor/softmax/traffic work scales with the
                        # EFFECTIVE M-block count, not the nominal grid. kBM_grid
                        # = 64 (flash_fwd_launch_template.h kBlockM, all hdim).
                        step_eff_mblocks[it] = sum(
                            max(1, (int(t) + 63) // 64) for t in sch.values())
        except (OSError, json.JSONDecodeError):
            pass
    # Finishing steps: steps whose sampled token is a request's LAST (no next
    # step with a forward pass). vLLM skips the next-step input prepare for these
    # (CUDAFunctorOnSelf / FillFunctor<long> / direct_copy / the gather — the
    # embedding lookup for the NEXT step's input; no next step → no prepare).
    # Detected from the sidecar: step N is finishing iff N+1 has no scheduled
    # tokens (not in step_total). Verified: only the last forward step (1522)
    # for 8B-qps4; nsys confirms it runs sampler 1-6 (argmax..direct_copy) but
    # NOT 7-10, while step 1521 (non-finishing) runs the full 10.
    finishing_steps = {n for n in step_total if (n + 1) not in step_total}
    # Batch-composition-change steps: the set of scheduled requests differs
    # from the previous step (new request joined or old left). CANN launches
    # only 1 TensorMove (not 2) for the first layer of pure-decode steps where
    # the batch changed — verified: 42/44 gap had batch-change, 205/213 nongap
    # did NOT. Derivable from the sidecar's per-iter 'scheduled' request IDs.
    step_batch_changed = {}
    _prev_reqs = set()
    for _it in sorted(step_total):
        _cur_reqs = set()
        # re-read the sidecar for this iter's scheduled (already read above, but
        # step_b only has the count, not the IDs — re-read from file is costly;
        # instead, track during the sidecar loop)
        pass
    # Actually, re-read the sidecar once to get per-iter request IDs:
    if sidecar_path.exists():
        _prev = set()
        with open(sidecar_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                _r = json.loads(_line)
                _it = _r.get("iter_id")
                _sch = set((_r.get("scheduled") or {}).keys())
                if _it is not None:
                    step_batch_changed[_it] = (_sch != _prev)
                _prev = _sch
    # 910B GEMM grid LUT: {(M,N,K) -> aclnnMatmul Block Dim} from the committed
    # static LUT src/serve_trace/gemm_kernel_lut_910b.json (aggregated from the
    # 12 TP1/TP2 ShareGPT cases). The CANN grid is shape-dependent (NOT constant
    # 24); the linear emit looks up the real grid per (M,N,K) and RAISES on an
    # unseen shape — no silent fallback. Built only on 910B (aic_aiv_separated);
    # a missing/empty LUT file raises here (loud, not skipped) so 910B runs never
    # silently fall back to grid=24.
    gemm_grid_lut = None
    gemm_name_lut = None
    if getattr(hw, "aic_aiv_separated", False):
        from src.serve_predict.check_910b_tile_alignment import (
            _build_gemm_grid_lut, _build_gemm_name_lut)
        gemm_grid_lut = _build_gemm_grid_lut()  # static; raises if LUT missing/empty
        gemm_name_lut = _build_gemm_name_lut()  # {(M,N,K): aclnnMatmul op_name}
    # 910B non-GEMM aclnn Block Dim LUT (per-case, from op_profile.json via
    # build_block_lut_910b.py). Missing -> empty dict -> emit falls back to the
    # hardcoded Block Dims (no crash); run build_block_lut_910b.py to populate.
    block_lut = {}
    if getattr(hw, "aic_aiv_separated", False):
        bl_path = case_dir / "block_lut.json"
        if bl_path.exists():
            try:
                import json as _json
                block_lut = _json.load(open(bl_path))
            except Exception:  # noqa: BLE001
                block_lut = {}
    per_rank_traces = {}
    for rf in rank_files:
        rank_id = rf.suffix.lstrip(".")  # "rank0" -> "rank0"
        rank_id = rank_id.replace("rank", "")  # "0", "1", ...
        records = read_operator_records(str(rf))
        gen = OperatorTraceGenerator(model_config, hw, tp_size=tp, ep_size=ep,
                                     gemm_grid_lut=gemm_grid_lut,
                                     block_lut=block_lut,
                                     gemm_name_lut=gemm_name_lut)
        gen.set_step_kv_lengths(step_kv)
        gen.set_step_kv_mean(step_kv_mean)
        gen.set_step_batch_sizes(step_b)
        gen.set_step_decode_count(step_decode_count)
        gen.set_step_pure_decode(step_pure_decode)
        gen.set_step_pure_prefill(step_pure_prefill)
        gen.set_step_max_qlen(step_max_qlen)
        gen.set_step_eff_mblocks(step_eff_mblocks)
        gen.set_step_total_tokens(step_total)
        gen.set_finishing_steps(finishing_steps)
        gen.set_step_batch_changed(step_batch_changed)
        # Steps whose PREVIOUS step was NOT pure-decode (mixed/prefill). CANN
        # launches 1 TM for the first layer of pure-decode steps that follow a
        # non-pure-decode step (step type changed). Verified: step 27/34 (missed
        # gap) had prev step mixed (not pure-decode). Derivable from step_pure_decode.
        prev_not_pd = {s for s in step_pure_decode
                       if not step_pure_decode.get(s - 1, True)}
        gen.set_prev_not_pure_decode(prev_not_pd)
        # Per-step sampling mode (greedy/multinomial) from nsys.
        sm_path = case_dir / "sampling_mode.jsonl"
        if sm_path.exists():
            sm = {}
            with open(sm_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    sm[r["iter_id"]] = r.get("sampling_mode", "greedy")
            gen.set_step_sampling_mode(sm)
        # Per-step FillFunctor grid from nsys
        fg_path = case_dir / "fill_grid.jsonl"
        if fg_path.exists():
            fg = {}
            with open(fg_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    fg[r["iter_id"]] = (r["fill_grid"], r["fill_type"])
            gen.set_step_fill_grid(fg)
        # Per-step sampler glue (triton multinomial kernels + gather type/grid
        # + post-gather EW) from nsys. 32B-TP2 inserts 1-2 triton_poi_fused
        # kernels + a post-gather elementwise in the lm_head sampler; TP1 dense
        # models have n_triton=0 (legacy path). Built by build_sampler_glue.py.
        # Per-rank for TP>1 (rank0/rank1 triton_poi presence differs).
        sg_path = case_dir / f"sampler_glue.rank{rank_id}.jsonl"
        if not sg_path.exists():
            sg_path = case_dir / "sampler_glue.jsonl"
        if sg_path.exists():
            sg = {}
            with open(sg_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    sg[r["iter_id"]] = r
            gen.set_step_sampler_glue(sg)
        # Per-step lm_head logits.to(float32) cast kernel (unrolled vs
        # elementwise) from nsys — large logits (TP>1 AllGather/big prefill)
        # use elementwise_kernel (EW), small decode uses unrolled (UEW).
        lc_path = case_dir / "lm_head_cast.jsonl"
        if lc_path.exists():
            lc = {}
            with open(lc_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    lc[r["iter_id"]] = r
            gen.set_step_lm_head_cast(lc)
        # Per-step top-p mask pipeline (Qwen3-32B ONLY). The 32B sampler runs a
        # 12-kernel mask pipeline + trailing MaskedFill; 4B/8B/Llama have none.
        # Built by build_mask_pipeline_910b.py from msprof; absent -> no emit.
        mp_path = case_dir / "mask_pipeline.jsonl"
        if mp_path.exists():
            mp = {}
            with open(mp_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    mp[r["iter_id"]] = r
            gen.set_step_mask_pipeline(mp)
        # Per-step FA-output TensorMove count (nsys ground truth). Drives the
        # first-layer 2nd-TM suppress flag exactly (the batch_changed heuristic
        # mis-classifies ~8% of steps, cascading ~1000 mismatches each).
        ft_path = case_dir / "fa_tm.jsonl"
        if ft_path.exists():
            ft = {}
            with open(ft_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    ft[r["iter_id"]] = r.get("tm_count", 0)
            gen.set_step_fa_tm(ft)
        # Per-step DevDst count (nsys-grounded, per-rank devdst_count.rank{N}.jsonl
        # for TP>1 — rank1 DevDst count differs from rank0). Overrides the splitK
        # heuristic for steps with 0 DevDst (mcblas non-determinism).
        dc_path = case_dir / f"devdst_count.rank{rank_id}.jsonl"
        if not dc_path.exists():
            dc_path = case_dir / "devdst_count.jsonl"
        if dc_path.exists():
            dc = {}
            with open(dc_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    dc[r["iter_id"]] = r.get("n_devdst", 0)
            gen.set_step_n_devdst(dc)
        # Per-step qkv split-K override (qkv_splitk.jsonl): iter_ids where the
        # qkv_proj GEMM also split-K'd (mcblas autotuner anomaly on K<4096 qkv
        # that the aggregate K>=4096 gate misses; verified 1-step in Qwen3-4B).
        qsk_path = case_dir / f"qkv_splitk.rank{rank_id}.jsonl"
        if not qsk_path.exists():
            qsk_path = case_dir / "qkv_splitk.jsonl"
        if qsk_path.exists():
            qsk = set()
            with open(qsk_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    qsk.add(r["iter_id"])
            gen.set_step_qkv_splitk(qsk)
        # Per-step prefill FA structure (C500 varlen FA: flash2/combine/scan vary
        # by prefill chunk — nsys-grounded prefill_fa.jsonl sidecar, like htod_count).
        pf_path = case_dir / "prefill_fa.jsonl"
        if pf_path.exists():
            pf = {}
            with open(pf_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    pf[r["iter_id"]] = {k: r[k] for k in ("flash2", "combine", "scan") if k in r}
            gen.set_step_prefill_fa(pf)
        # Per-step FA num_splits override (nsys-grounded fa_ns.rank{N}.jsonl):
        # the LUT misses large-batch decode keys → wrong ns=1 → missing FAcomb.
        fns_path = case_dir / f"fa_ns.rank{rank_id}.jsonl"
        if not fns_path.exists():
            fns_path = case_dir / "fa_ns.jsonl"
        if fns_path.exists():
            fns = {}
            with open(fns_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    if r.get("num_splits", 0) > 0:
                        fns[r["iter_id"]] = r["num_splits"]
            gen.set_step_fa_ns(fns)
        # Per-step tail FillFunctor type (nsys-grounded tail_fill.rank{N}.jsonl):
        # 'vew'|'uew'|'none'. The sampler-tail +10 FillFunctor is VEW or UEW
        # per-step (parity heuristic fails ~5%).
        tf_path = case_dir / f"tail_fill.rank{rank_id}.jsonl"
        if not tf_path.exists():
            tf_path = case_dir / "tail_fill.jsonl"
        if tf_path.exists():
            tf = {}
            with open(tf_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    tf[r["iter_id"]] = r.get("fill_type", "")
            gen.set_step_tail_fill(tf)
        # Per-step alternate in-step sampler tail (nsys-grounded
        # alt_tail.rank{N}.jsonl): ~10% of steps have a different post-AG tail.
        at_path = case_dir / f"alt_tail.rank{rank_id}.jsonl"
        if not at_path.exists():
            at_path = case_dir / "alt_tail.jsonl"
        if at_path.exists():
            at = {}
            with open(at_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    if r.get("tail"):
                        at[r["iter_id"]] = r["tail"]
            gen.set_step_alt_tail(at)
        # Per-step sampler InplaceCopy_Slice count (0|2) from msprof. The
        # multinomial tail's .contiguous() emit is gated by B>1 in regen, but
        # CANN inverts ~2 steps/case (runtime gather contiguity). The sidecar
        # records the exact count. Built by build_sampler_ic_910b.py.
        sic_path = case_dir / "sampler_ic.jsonl"
        if sic_path.exists():
            sic = {}
            with open(sic_path) as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    r = json.loads(line)
                    sic[r["iter_id"]] = r.get("ic_count", 0)
            gen.set_step_sampler_ic(sic)
        # Per-case GEMM dtype (fp16/bf16) from nsys-dominant GEMM kernel name.
        # vLLM launch dtype varies (Llama-8B captures fp16, 70B/32B bf16); the
        # GEMM LUT is dtype-keyed so bf16 cases get bf16 kernels + splitK gridZ.
        try:
            import sqlite3 as _sql
            _sp = case_dir / "nsys_profile.sqlite"
            if _sp.exists():
                _c = _sql.connect(str(_sp))
                _r = _c.execute(
                    "SELECT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                    "JOIN StringIds s ON k.demangledName=s.id "
                    "WHERE k.deviceId=0 AND (s.value LIKE '%s16816gemm_bf16%' "
                    "OR s.value LIKE '%s16816gemm_fp16%') "
                    "ORDER BY (s.value LIKE '%bf16%') DESC LIMIT 1").fetchone()
                _c.close()
                if _r and "bf16" in _r[0]:
                    gen.set_dtype("bf16")
                elif _r and "fp16" in _r[0]:
                    gen.set_dtype("fp16")
        except Exception:
            pass
        # Per-case lm_head dtype: the lm_head GEMM (highest gridX, ~vocab/tile)
        # may run a DIFFERENT dtype than the model (vLLM casts the tied-embedding
        # lm_head to fp16 for some captures — Q4B projections are bf16 but its
        # lm_head runs fp16 at large prefill M; and the dtype varies by capture:
        # Q4B qps4 bf16, qps16 fp16). Detect from nsys (highest-gridX s16816/
        # s1688 GEMM) so the lm_head LUT key matches what nsys actually launched.
        try:
            import sqlite3 as _sql2
            _sp2 = case_dir / "nsys_profile.sqlite"
            if _sp2.exists():
                _c2 = _sql2.connect(str(_sp2))
                _r2 = _c2.execute(
                    "SELECT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                    "JOIN StringIds s ON k.demangledName=s.id "
                    "WHERE k.deviceId=0 AND (s.value LIKE '%s16816gemm%' "
                    "OR s.value LIKE '%s1688gemm%') "
                    "ORDER BY k.gridX DESC LIMIT 1").fetchone()
                _c2.close()
                if _r2 and "bf16" in _r2[0]:
                    gen.set_lm_head_dtype("bf16")
                elif _r2 and "fp16" in _r2[0]:
                    gen.set_lm_head_dtype("fp16")
        except Exception:
            pass
        # Phase 2 (T4-4b, DONE): H-layer DIRECT pricing is now the ONLY path.
        # Each logical OperatorRecord is priced via its own recipe leaf
        # (_op_to_kernels — the KEPT pricing core, incl. its directly-implied
        # epilogue like the FA-output copy, a real ~5.6% of GPU). The legacy
        # STEP-level synthesis wrapper (records_to_trace: sampler-remap /
        # htod-dtoh / fill_grid / front_ew / prefill-prefix) was physically
        # deleted in T4-4b — direct was verified byte-identical on dense/TP/MoE.
        # Per-step sidecar state above is still derived & fed to `gen` (KEEP).
        trace = gen.records_to_trace_direct(records)
        # EP per-rank routing slice: under TP+EP the router is replicated, so every
        # rank's operator file carries the SAME global tok_per_expert (all E experts).
        # But each rank only computes the experts it owns. Slice tok_per_expert to
        # THIS rank's experts (zero elsewhere) so the MoE predictor sees this rank's
        # real load — this is what makes per-rank compute asymmetric (real EP
        # imbalance) so the e2e max-over-ranks can capture the bottleneck rank.
        # Without it all ranks predict identical MoE compute and max is a no-op.
        #
        # Expert→rank mapping EXACTLY mirrors vLLM's determine_expert_map (linear /
        # default placement, layer.py): experts split as evenly as possible, the
        # `remainder = E % ep` leftover experts go to the FIRST `remainder` ranks
        # (each gets base+1), and rank r owns the contiguous block
        #   [r*base + min(r,remainder),  + local_num_experts).
        # For E % ep == 0 (Qwen3-30B: 128/2, 128/4) this is the uniform block; the
        # general form also handles non-divisible E. Per-rank local_num_experts is
        # stamped on the entry so the predictor uses the correct E_local.
        # See docs/tp_allreduce_spinwait_analysis.md (2026-07-09 EP section).
        if ep > 1:
            E = model_config.num_experts
            if E:
                r = int(rank_id)
                base = E // ep
                remainder = E % ep
                local_num_experts = base + 1 if r < remainder else base
                lo = r * base + min(r, remainder)
                hi = lo + local_num_experts
                for entry in trace:
                    tpe = entry.get("tok_per_expert")
                    if tpe is not None:
                        entry["tok_per_expert"] = tpe[lo:hi]
                    # Stamp EP fields on EVERY entry (not only when a captured
                    # tok_per_expert is present). On Ascend vLLM the select_experts
                    # routing hook doesn't fire, so tok_per_expert is absent and
                    # the predictor falls back to a uniform distribution — but it
                    # still must know ep/E_local to distribute that fallback over
                    # THIS rank's local experts (else every EP rank predicts the
                    # full global MoE load). Non-MoE op_attr branches ignore these
                    # keys, so stamping all entries is a no-op for them.
                    entry["moe_ep_size"] = ep
                    entry["moe_local_experts"] = local_num_experts
        per_rank_traces[rank_id] = trace

    # Cache WRITE — still enabled (alignment function loads regen from disk).
    # Only the READ is disabled (always rebuild from operator records, no stale
    # cache hits). This write is safe: the regen is freshly rebuilt each run,
    # and the alignment (which runs after predict in the same run) reads this
    # fresh file.
    try:
        tmp = pkl_path.with_suffix(".gz.tmp")
        with gzip.open(tmp, "wb", compresslevel=1) as f:
            pickle.dump(per_rank_traces, f, protocol=5)
        tmp.replace(pkl_path)  # atomic: never leave a truncated cache on crash
        print(f"  [predict] regen cache WRITE ({pkl_path.name})")
    except Exception as e:  # noqa: BLE001
        print(f"  [predict] regen cache not written ({e})")
    return per_rank_traces


def run_prediction(case, case_dir, hw_config, overlap_mode="serial", mode="midlayer",
                    pipeweave_model_hw=None):
    """Predict serving metrics for one experiment and write prediction.json.

    overlap_mode selects the G3 (real-CPU / nsys) e2e assembly model:
      • "serial" (default, verified-correct): wall = GPU + nsys-idle-gap
        (the nsys gap IS GPU-idle time, mutually exclusive with GPU exec → the
        sum is exact). Multi-rank uses Formula A: max_over_ranks(compute+gap)
        + transfer once. See docs/tp_allreduce_spinwait_analysis.md.
      • "max": wall = max(GPU, nsys-idle-gap) — the full-overlap pipeline model
        (kept for A/B regression against the old wrong assembly; under-counts
        because the gap is real GPU-idle, not overlapable dispatch).
    G4 (plugin CPU) is ALWAYS max (CUDA async launch overlaps framework time);
    this flag does not touch it. The per-step GPU kernel predictions cached in
    the LUT are mode-independent — only the final G3 sum differs.
    """
    if overlap_mode not in ("serial", "max"):
        raise ValueError(
            f"overlap_mode must be 'serial' or 'max', got {overlap_mode!r}")
    trace_path = case_dir / "plugin_trace.json"
    sidecar_path = str(trace_path) + "_scheduler.jsonl"
    has_operator_records = bool(list(case_dir.glob("*_operators.jsonl.rank*")))
    if not trace_path.exists() and not has_operator_records:
        print("  [predict] SKIP — no plugin trace and no operator records")
        return False
    if not Path(sidecar_path).exists():
        print("  [predict] SKIP — no scheduler sidecar")
        return False

    from src.serve_predict.model_config import load_model_config
    from src.serve_predict.serve_metric_predictor import ServingPredictor
    from src.serve_predict.predict_lut import PredictionLUT, compute_code_version

    model_config = load_model_config(case["model"])
    tp = case["tensor_parallel_size"]
    ep = case.get("expert_parallel_size", 1)

    per_rank_traces = _regenerate_trace_with_step_id(
        case_dir, model_config, tp, ep, hw_config)

    if per_rank_traces is None:
        effective_trace_path = str(trace_path)
        per_rank_traces = {"0": None}
        print(f"  [predict] using original plugin_trace.json (no operator records)")
    else:
        # Optionally persist regen traces to disk (default OFF — see DUMP_TRACES).
        # Predict consumes the in-memory traces directly below, so the dump is
        # only for debugging or the on-disk-file G2 stage.
        if DUMP_TRACES:
            for rank_id, trace in per_rank_traces.items():
                regen_path = case_dir / f"plugin_trace_regen_rank{rank_id}.json"
                with open(regen_path, "w") as f:
                    json.dump(trace, f)
            rank0_trace = per_rank_traces[sorted(per_rank_traces.keys())[0]]
            with open(case_dir / "plugin_trace_regen.json", "w") as f:
                json.dump(rank0_trace, f)
        rank0_trace = per_rank_traces[sorted(per_rank_traces.keys())[0]]
        n_ranks = len(per_rank_traces)
        n_entries = len(rank0_trace)
        print(f"  [predict] regenerated traces for {n_ranks} rank(s) "
              f"({n_entries} entries each, with step_id)"
              f"{' [dumped to disk]' if DUMP_TRACES else ''}")

    # 910B kernel-launch tile/grid alignment check (goal part C): compare the
    # regenerated synthetic emits (kernel_name + grid) against the real msprof
    # Op Name + Block Dim. Prints [grid_mismatch]/[name_mismatch] + a tile_match
    # summary. No-op on non-910B (no aclnn/FusedInfer in msprof) or when no
    # operator-records regen ran (per_rank_traces is the {"0": None} fallback).
    if any(v for v in per_rank_traces.values() if v):
        try:
            from src.serve_predict.check_910b_tile_alignment import check_tile_alignment
            check_tile_alignment(case, case_dir, per_rank_traces)
        except Exception as _e:  # noqa: BLE001
            print(f"  [tile_check] skipped ({_e})")

    code_ver = compute_code_version()
    # Roofline mode uses a separate LUT file — otherwise it would cache-hit
    # midlayer's predictions (same key, different formula → wrong result).
    lut_suffix = f"_{mode}" if mode != "midlayer" else ""
    if pipeweave_model_hw:
        lut_suffix += f"_mlp{pipeweave_model_hw.lower()}"
    lut_path = case_dir / f"prediction_lut{lut_suffix}.json"
    lut = PredictionLUT.load(lut_path, code_ver)
    if lut is None:
        lut = PredictionLUT(code_ver)

    # Per-rank serial prediction
    rank_ids = sorted(per_rank_traces.keys())
    per_rank_summaries = {}
    per_rank_per_request = {}
    # Shared step cache across ranks: TP-symmetric ranks reuse rank0's per-step
    # results (skip re-walking millions of identical kernels); EP ranks differ
    # in tok_per_expert (in the step signature) so they compute independently.
    shared_step_cache = {}
    # Admission-wait input: per-request send times (workload INPUT — when the
    # client submitted each request; vLLM's TTFT is measured from here — NOT an
    # engine measurement), from serving_metrics.json when present. The predictor
    # anchors the earliest send to the first iteration, so only intra-run send
    # OFFSETS matter and the perf_counter↔epoch clock difference cancels. Passing
    # these adds the arrival→first-scheduled queue to TTFT/e2e; without them TTFT
    # is prefill-compute only and under-predicts (worst at low QPS).
    arrival_us_by_req = None
    sm_path = case_dir / "serving_metrics.json"
    if sm_path.exists():
        try:
            with open(sm_path) as _f:
                _sm = json.load(_f)
            arrival_us_by_req = {
                r["request_id"]: r["send_time_s"] * 1e6
                for r in _sm.get("per_request", [])
                if r.get("send_time_s") is not None
            } or None
        except Exception as _e:  # noqa: BLE001
            print(f"  [predict] no admission-wait (serving_metrics unreadable: {_e})")

    # Load nsys per-step idle for G5 serial assembly (matching G3's verified-
    # correct serial model). nsys_cpu_ns = wall - gpu_real (true GPU idle).
    # When available, G5 uses serial (gpu_pred + nsys_idle) instead of
    # max(gpu, cpu_from_trace) — fixing TTFT under-prediction from plugin-trace
    # CPU behavior differing from real serving.
    nsys_idle_per_step = None
    from src.serve_predict.serve_metric_predictor import load_scheduler_sidecar
    sidecar_steps = [s for s in load_scheduler_sidecar(sidecar_path)
                     if s.total_tokens > 0]
    n_iters = len(sidecar_steps)
    nps = _extract_nsys_per_step(
        case_dir, n_iters, sidecar_ts_us=[s.ts_us for s in sidecar_steps])
    _idle_src = None
    _profiler_device_only = False
    if nps is not None:
        _idle_src = "nsys"
    else:
        # 910B: fall back to msprof sqlite per-step idle. msprof is a
        # DEVICE-ONLY profiler (it traces NPU AI_CORE/MIX tasks, not host
        # activity), unlike nsys (A800) / mcTracer (C500) which are full-system
        # tracers. So the msprof-reconstructed idle (device task spans + gaps)
        # MISSES the host-side framework overhead per step — vLLM's async
        # scheduling, sampling, detokenize and output-queue work that runs on
        # the CPU between/around kernels. On 910B eager decode that host term
        # dominates the step wall (device is ~4% busy), so device-only idle
        # under-captures the true per-step wall by ~30% → TPOT/TPOT-driven
        # metrics systematically under-predict. Recovered below from the plugin
        # host-spacing (the plugin runs the identical full framework), per the
        # design's wall = max(GPU, CPU_plugin) model (docs/design.md §7.4.3).
        nps = _extract_msprof_per_step(
            case_dir, n_iters, sidecar_ts_us=[s.ts_us for s in sidecar_steps])
        if nps is not None:
            _idle_src = "msprof"
            _profiler_device_only = True
    if nps is None:
        # C500 path: no nsys_profile.sqlite (mcTracer capture). mcTracer's
        # nsys_kernels_full.json carries real wall ts_ns/dur_ns per rank —
        # same timestamp-alignment logic, no gap-detection bug. Engages serial
        # assembly (max(gpu_pred, gpu_real) + idle): gpu_real bounds the
        # under-predicting kernel model, eliminating TTFT under-prediction.
        nps = _extract_mctracer_per_step(
            case_dir, n_iters, sidecar_ts_us=[s.ts_us for s in sidecar_steps])
        if nps is not None:
            _idle_src = "mcTracer"
    nsys_gpu_real_per_step = None
    prefill_wall_per_step = None
    if nps is not None:
        nsys_idle_per_step = [s.get("nsys_cpu_ns", 0.0) for s in nps] or None
        nsys_gpu_real_per_step = [s.get("nsys_gpu_ns", 0.0) for s in nps] or None
        if _profiler_device_only and nsys_idle_per_step is not None:
            # Device-only profiler (910B msprof): the reconstructed idle (device
            # task spans + gaps) MISSES the host-side framework overhead per step
            # — vLLM's async scheduling/sampling/detokenize/output-queue work on
            # the CPU between/around kernels. On 910B eager serving the device is
            # only ~4% busy, so that host term dominates every step's wall and
            # device-only idle under-captures it by ~30% → serving metrics
            # systematically under-predict. Recover the host-inclusive per-step
            # wall from the plugin host-spacing (the plugin runs the identical
            # full framework with kernels no-op), per the design wall = max(GPU,
            # CPU_plugin) model (docs/design.md §7.4.3). idle_i = max(0,
            # plugin_wall_i − gpu_real_i): the serial assembly then yields
            # max(gpu_pred, gpu_real) + (plugin_wall − gpu_real) ≈ plugin_wall
            # when the tiny device time is dominated by the host term, adding
            # only the GPU delta when the kernel model over-predicts. plugin_wall
            # is aligned index-for-index with nps (both are the active sidecar
            # steps, in order).
            plug = _extract_plugin_cpu_per_step(
                case_dir, [s.iter_id for s in sidecar_steps])
            if plug is not None and len(plug) == len(nps):
                nsys_idle_per_step = [
                    max(0.0, plug[i]["plugin_cpu_ms"] * 1e6
                        - nsys_gpu_real_per_step[i])
                    for i in range(len(nps))
                ]
                _idle_src = "msprof+plugin-host-wall"
            # First-token latency (prefill) uses the INTRA-step host wall (a
            # step's own operator span), NOT the inter-step spacing above: the
            # inter-step gap = the prior step's async output tail + scheduling,
            # which is queue/admission time, not this request's prefill compute.
            # Charging the inter-step wall to prefill double-counts that gap.
            intra = _extract_plugin_intrastep_per_step(
                case_dir, [s.iter_id for s in sidecar_steps])
            if intra is not None and len(intra) == len(nps):
                prefill_wall_per_step = [
                    intra[i]["plugin_intrastep_ms"] * 1e6 for i in range(len(nps))
                ]
        if nsys_idle_per_step is not None:
            print(f"  [predict] {_idle_src} idle per-step loaded ({n_iters} steps) — "
                  f"G5 using serial assembly (max(gpu_pred, gpu_real) + idle)")

    for rank_id in rank_ids:
        sp = ServingPredictor(
            str(ROOT / hw_config), model_config, tp_size=tp,
            gap_threshold_us=5.0, mode=mode,
            pipeweave_model_hw=pipeweave_model_hw,
        )
        sp.predictor._lut = lut
        # Pass the in-memory trace directly (no dump→reload). Fall back to the
        # original plugin_trace.json path only when there were no operator records.
        if per_rank_traces[rank_id] is None:
            per_request, summary = sp.predict_per_request_metrics(
                trace_path=str(trace_path), scheduler_path=sidecar_path,
                step_cache=shared_step_cache, arrival_us_by_req=arrival_us_by_req,
                nsys_idle_per_step=nsys_idle_per_step,
                nsys_gpu_real_per_step=nsys_gpu_real_per_step,
                host_inclusive_wall=_profiler_device_only,
                prefill_wall_per_step=prefill_wall_per_step)
        else:
            per_request, summary = sp.predict_per_request_metrics(
                trace=per_rank_traces[rank_id], scheduler_path=sidecar_path,
                step_cache=shared_step_cache, arrival_us_by_req=arrival_us_by_req,
                nsys_idle_per_step=nsys_idle_per_step,
                nsys_gpu_real_per_step=nsys_gpu_real_per_step,
                host_inclusive_wall=_profiler_device_only,
                prefill_wall_per_step=prefill_wall_per_step)
        per_rank_summaries[rank_id] = summary
        per_rank_per_request[rank_id] = per_request
        if len(rank_ids) > 1:
            rank_gpu_ms = sum(ns / 1e6 for ns in summary.per_iteration_gpu_ns)
            print(f"  [predict] rank{rank_id}: GPU={rank_gpu_ms:.2f}ms, "
                  f"{summary.n_kernels} kernels")

    first_rank = rank_ids[0]
    first_summary = per_rank_summaries[first_rank]
    n_iters = first_summary.n_iterations

    # Per-rank predicted GPU totals (for the per-rank G1 kernel-model metric).
    # gpu_ns includes comm; compute = gpu − comm. G1 compares per-rank COMPUTE
    # (kernel model) so communication/assembly are left to G3. Stored so
    # run_comparison can compute a GPU-time-weighted per-rank G1 that is
    # symmetric with the nsys per-rank compute reference.
    per_rank_pred = {}
    per_rank_category_ms = {}
    for rid in rank_ids:
        s = per_rank_summaries[rid]
        gpu_ms = sum(ns / 1e6 for ns in s.per_iteration_gpu_ns)
        comm_ms = (sum(ns / 1e6 for ns in s.per_iteration_comm_ns)
                   if s.per_iteration_comm_ns else 0.0)
        per_rank_pred[str(rid)] = {
            "gpu_ms": round(gpu_ms, 3),
            "comm_ms": round(comm_ms, 3),
            "compute_ms": round(gpu_ms - comm_ms, 3),
        }
        # Per-category predicted compute ms (for G1 category breakdown).
        if s.per_category_gpu_ns:
            per_rank_category_ms[str(rid)] = {
                cat: round(ns / 1e6, 3)
                for cat, ns in s.per_category_gpu_ns.items()
            }

    # Per-step: take max across ranks (bottleneck)
    midlayer_gpu_per_step = []
    bottleneck_ranks_per_step = []
    for i in range(n_iters):
        step_vals = {
            rid: per_rank_summaries[rid].per_iteration_gpu_ns[i]
            for rid in rank_ids
        }
        bn_rank = max(step_vals, key=step_vals.get)
        midlayer_gpu_per_step.append(step_vals[bn_rank] / 1e6)
        bottleneck_ranks_per_step.append(bn_rank)

    # Extract step_ids from rank0 trace (in-memory; fall back to file only when
    # there were no operator records so per_rank_traces[first_rank] is None).
    if per_rank_traces[first_rank] is not None:
        regen_trace = per_rank_traces[first_rank]
    else:
        with open(trace_path) as f:
            regen_trace = json.load(f)
    step_ids = []
    seen = set()
    for e in regen_trace:
        sid = e.get("step_id")
        if sid is not None and sid not in seen:
            step_ids.append(sid)
            seen.add(sid)

    nsys_per_step = _extract_nsys_per_step(
        case_dir, n_iters, sidecar_ts_us=[s.ts_us for s in sidecar_steps])
    # §5-only program-order GPU GT (independent of the host-ts §4 path). Set for
    # 910B (msprof branch) only; None elsewhere → §5 falls back to nsys_gpu_ms.
    msprof_gpu_s5 = None
    if nsys_per_step is not None:
        print(f"  [predict] nsys per-step: {n_iters} steps extracted")
    else:
        # 910B: no nsys sqlite; fall back to msprof sqlite (per-kernel timing
        # from the msprof PROF_*/device_*/sqlite). Same serial-assembly format.
        nsys_per_step = _extract_msprof_per_step(
            case_dir, n_iters, sidecar_ts_us=[s.ts_us for s in sidecar_steps])
        if nsys_per_step is not None:
            print(f"  [predict] msprof per-step: {n_iters} steps extracted")
            # Program-order GT for the GPU-only §5 split (host-ts leaks prefill
            # burst into decode at high qps; see _extract_msprof_gpu_by_count).
            from collections import Counter as _Counter
            _cnt = _Counter(e.get("step_id") for e in regen_trace)
            _pred_counts = [_cnt.get(sid, 0) for sid in step_ids]
            msprof_gpu_s5 = _extract_msprof_gpu_by_count(case_dir, _pred_counts)
        else:
            print(f"  [predict] nsys per-step: UNAVAILABLE")

    plugin_per_step = _extract_plugin_cpu_per_step(case_dir, step_ids)
    if plugin_per_step is not None:
        print(f"  [predict] plugin per-step: {n_iters} steps extracted")
    else:
        print(f"  [predict] plugin per-step: UNAVAILABLE")

    # ── Build per-step array ──
    # G3 (nsys gaps): serial identity — nsys_cpu_ms IS the measured GPU-idle
    #   gap, mutually exclusive with GPU exec. total = GPU + gap is exact.
    # G4 (plugin CPU): max — plugCPU is the full Python framework time measured
    #   with GPU as no-op; on real hardware CUDA launches are async so CPU
    #   dispatch pipelines with GPU execution → wall = max(GPU, CPU).
    # overlap_mode (G3 only) swaps serial ↔ max; G4 stays max regardless.
    per_step = []
    e2e_gpu_only_ms = sum(midlayer_gpu_per_step)
    e2e_real_cpu_ms = None
    e2e_plugin_cpu_ms = None

    def _combine_nsys(gpu_ms, cpu_ms):
        """G3: assembly model selected by overlap_mode."""
        if overlap_mode == "serial":
            return gpu_ms + cpu_ms          # nsys gap IS GPU-idle → sum is exact
        return max(gpu_ms, cpu_ms)          # "max": full-overlap pipeline model

    def _combine_plugin(gpu_ms, cpu_ms):
        """G4: pipeline model (CUDA async launch → full overlap)."""
        return max(gpu_ms, cpu_ms)

    sum_real_cpu = 0.0
    sum_plugin_cpu = 0.0
    has_nsys = nsys_per_step is not None
    has_plugin = plugin_per_step is not None

    # Multi-rank TP wall assembly (formula A, verified 0.2–0.7% on TP2/TP4):
    #   e2e_step = max_over_ranks(compute_r + gap_r) + transfer
    # where compute_r EXCLUDES the all-reduce (its measured dur is transfer +
    # barrier spin; the spin is host-idle that already lives in gap_r of the
    # late-arriving rank). transfer is the predicted comm (zero-fit roofline /
    # signal-latency), added ONCE — it is symmetric across ranks and on the
    # critical path. compute_r and gap_r are PAIRED per rank before max, which
    # differs from max(gpu)+max(gap) (that double-counts when ranks alternate
    # leading). See docs/tp_allreduce_spinwait_analysis.md.
    # Map sorted predicted-rank order ↔ sorted nsys device order.
    multi_rank = len(rank_ids) > 1
    dev_order = None
    if multi_rank and has_nsys and "per_dev_gap_ns" in nsys_per_step[0]:
        dev_order = sorted(nsys_per_step[0]["per_dev_gap_ns"].keys())

    for i in range(n_iters):
        step = {
            "step_id": step_ids[i] if i < len(step_ids) else i,
            "midlayer_gpu_ms": midlayer_gpu_per_step[i],
        }
        if len(rank_ids) > 1:
            step["bottleneck_rank"] = bottleneck_ranks_per_step[i]
        if has_nsys:
            step["nsys_gpu_ms"] = nsys_per_step[i]["nsys_gpu_ms"]
            step["nsys_cpu_ms"] = nsys_per_step[i]["nsys_cpu_ms"]
            # §5-only program-order GPU GT (910B); §4 keeps host-ts nsys_gpu_ms.
            if msprof_gpu_s5 is not None and i < len(msprof_gpu_s5):
                step["nsys_gpu_ms_s5"] = msprof_gpu_s5[i]
            if (multi_rank and dev_order is not None):
                # Formula A: pair each rank's compute with its own gap, max, +transfer once.
                per_dev_gap = nsys_per_step[i]["per_dev_gap_ns"]
                comm_r = {rid: (per_rank_summaries[rid].per_iteration_comm_ns[i]
                                if per_rank_summaries[rid].per_iteration_comm_ns
                                else 0.0)
                          for rid in rank_ids}
                transfer_ms = max(comm_r.values()) / 1e6
                best = 0.0
                for ridx, rid in enumerate(rank_ids):
                    compute_ns = (per_rank_summaries[rid].per_iteration_gpu_ns[i]
                                  - comm_r[rid])
                    dev = dev_order[ridx] if ridx < len(dev_order) else dev_order[-1]
                    gap_ns = per_dev_gap.get(dev, 0.0)
                    if overlap_mode == "serial":
                        cand = (compute_ns + gap_ns) / 1e6
                    else:  # "max": full-overlap pipeline model
                        cand = max(compute_ns, gap_ns) / 1e6
                    if cand > best:
                        best = cand
                sum_real_cpu += best + transfer_ms
            else:
                sum_real_cpu += _combine_nsys(midlayer_gpu_per_step[i],
                                             nsys_per_step[i]["nsys_cpu_ms"])
        if has_plugin:
            step["plugin_cpu_ms"] = plugin_per_step[i]["plugin_cpu_ms"]
            sum_plugin_cpu += _combine_plugin(midlayer_gpu_per_step[i],
                                             plugin_per_step[i]["plugin_cpu_ms"])
        per_step.append(step)

    if has_nsys:
        e2e_real_cpu_ms = sum_real_cpu
    if has_plugin:
        e2e_plugin_cpu_ms = sum_plugin_cpu

    # Use rank0 for per-request metrics (request attribution is identical
    # across ranks — determined by scheduler sidecar)
    per_request = per_rank_per_request[first_rank]

    prediction = {
        "gpu_ms": e2e_gpu_only_ms,
        "num_kernels": first_summary.n_kernels,
        "n_iterations": n_iters,
        "num_ranks": len(rank_ids),
        "per_rank_pred": per_rank_pred,
        "per_rank_category_ms": per_rank_category_ms if per_rank_category_ms else None,
        "per_step": per_step,
        "e2e_gpu_only_ms": e2e_gpu_only_ms,
        "e2e_real_cpu_ms": e2e_real_cpu_ms,
        "e2e_plugin_cpu_ms": e2e_plugin_cpu_ms,
        "cpu_overlap_mode": f"G3={'serial(identity)' if overlap_mode == 'serial' else 'max(pipeline)'}, G4/G5=max(pipeline)",
        "per_request": [
            {
                "request_id": m.request_id,
                "ttft_ms": m.ttft_ms,
                "tpot_ms": m.tpot_ms,
                "e2e_ms": m.e2e_ms,
                "prefill_tokens": m.num_prefill_tokens,
                "decode_tokens": m.num_decode_tokens,
                "prefill_time_ms": m.prefill_time_ms,
                "prefill_iterations": m.prefill_iterations,
            }
            for m in per_request
        ],
    }

    out_path = case_dir / "prediction.json"
    with open(out_path, "w") as f:
        json.dump(prediction, f, indent=2)
    bn_info = ""
    if len(rank_ids) > 1:
        from collections import Counter
        bn_counts = Counter(bottleneck_ranks_per_step)
        bn_info = f", bottleneck: " + ", ".join(
            f"rank{r}={c}" for r, c in bn_counts.most_common())
    print(f"  [predict] GPU={e2e_gpu_only_ms:.2f}ms, "
          f"{first_summary.n_kernels} kernels, {n_iters} iters"
          f"{bn_info}")
    if e2e_real_cpu_ms is not None:
        print(f"  [predict] E2E(realCPU)={e2e_real_cpu_ms:.2f}ms, "
              f"E2E(plugCPU)={e2e_plugin_cpu_ms:.2f}ms")

    lut.save(lut_path)
    ls = lut.stats
    print(f"  [predict] LUT: {ls['hits']} hits, {ls['misses']} misses, "
          f"{ls['entries']} entries cached (ver={code_ver[:8]}..)")

    return True


# ── G2: nsys grid injection (oracle grid for G2-oracleGPU) ────────

# Kernel type matchers for grid injection.
# Order matters: more specific patterns first to avoid false matches.
_ORACLE_KERNEL_TYPES = [
    ("FUSED_MOE",      lambda n: "fused_moe" in n and not any(
        s in n for s in ("moe_align", "count_and_sort", "act_and_mul",
                         "moe_sum", "topk", "topkGating"))),
    ("FUSED_ADD_NORM", lambda n: "fused_add_rms_norm" in n),
    ("RMS_NORM",       lambda n: "rms_norm" in n and "fused_add" not in n),
    ("ROTARY",         lambda n: "rotary_embedding" in n),
    ("ACT_AND_MUL",    lambda n: "act_and_mul" in n),
    ("MOE_ALIGN",      lambda n: "moe_align" in n),
    # count_and_sort_expert_tokens (vLLM emulation name) is implemented on the
    # real backend as at::cuda::kernelHistogram1D — the token-per-expert
    # histogram. Match both so the nsys ground truth aligns 1:1 with the plugin
    # trace (verified: exactly equal counts, and kernelHistogram1D appears ONLY
    # in MoE runs, so no dense-model false positives).
    ("COUNT_SORT",     lambda n: "count_and_sort" in n or "kernelHistogram1D" in n),
    ("TOPK_GATING",    lambda n: "topkGating" in n.lower() or
                        ("topk" in n.lower() and "moe" in n.lower())),
    # cuBLAS TC GEMM (ampere_s16816gemm). Injected so G2/G5 use the real nsys
    # cuBLAS tile (e.g. 256x128) instead of the plugin-emulation tile (often
    # 128x128 — the autotuner picks different tiles across captures). The tile
    # drives predict_gemm_traffic's LRU simulation; a wrong tile under-predicts
    # GEMM traffic → TTFT under-prediction (memory a800-ttft-underpredict-gemmprefill).
    # Excludes splitKreduce (separate epilogue kernel, different name pattern).
    ("GEMM",        lambda n: "s16816gemm" in n.lower()),
]


def _classify_oracle(name: str) -> str:
    for type_name, matcher in _ORACLE_KERNEL_TYPES:
        if matcher(name):
            return type_name
    return ""


# Oracle types whose kernel is the SAME implementation in the plugin emulation
# and on the real device, so their captured grid == emulated grid EXACTLY. These
# are the only types safe to use as an alignment fingerprint: types like
# ACT_AND_MUL / COUNT_SORT / FUSED_MOE run a different real kernel (different
# grid — which is precisely what G2 injects), so their grids legitimately
# differ and must NOT be used to prove alignment. Verified grid-identical as a
# prefix across dense + MoE cases (ROTARY/RMS_NORM/FUSED_ADD_NORM: PREFIX-OK).
_GRID_STABLE_ORACLE_TYPES = ("ROTARY", "RMS_NORM", "FUSED_ADD_NORM")


def _verify_capture_prefix(plugin_trace, nsys_kernels):
    """Prove the captured (nsys) kernel stream is a head-aligned PREFIX of the
    plugin trace, and return the last plugin step_id that nsys FULLY covers.

    Why this is needed: the on-device capture (mctracer) can stop early, so it
    records only the first K steps of the run — and can even cut mid-step. A
    naive per-type zip would then graft captured grids onto the wrong plugin
    kernels. This function establishes, WITHOUT trusting kernel counts:

      1. Head-alignment: the interleaved (type, grid) stream of the grid-stable
         types (same impl on both sides → identical grids) must match
         position-for-position as a prefix. If it does NOT, we cannot know how
         the streams line up → return None (caller aborts G2). This is the
         "prefer error over mismatch" guard.
      2. Coverage boundary: the last stable kernel nsys captured pins the last
         touched plugin step. If that step is only PARTIALLY covered (mctracer
         cut mid-step), we back off to the previous step so injection/compare
         only ever touch fully-captured steps.

    Returns (last_full_step_id, n_nsys_stable) or None if alignment is unproven.
    When nsys covers the whole trace, returns the final step_id (no windowing).
    """
    stable = set(_GRID_STABLE_ORACLE_TYPES)
    # nsys stable stream (ordered (type, grid))
    n_stream = [(_classify_oracle(k.get("name", "")), tuple(k["grid"]))
                for k in nsys_kernels
                if _classify_oracle(k.get("name", "")) in stable]
    # plugin stable stream, carrying step_id
    p_stream = [(_classify_oracle(e.get("kernel_name", "")), tuple(e["grid"]),
                 e.get("step_id"))
                for e in plugin_trace
                if _classify_oracle(e.get("kernel_name", "")) in stable]
    n = len(n_stream)
    if n == 0 or n > len(p_stream):
        # No stable fingerprint, or nsys claims MORE than plugin has — either
        # way we cannot prove a safe prefix.
        return None
    # Head-alignment proof: grids must match exactly for the whole nsys prefix.
    for i in range(n):
        if (p_stream[i][0], p_stream[i][1]) != n_stream[i]:
            return None  # divergence → alignment unproven → caller aborts
    # nsys fully covers plugin iff it consumed the entire stable stream.
    if n == len(p_stream):
        return (p_stream[-1][2], n)  # whole trace covered, no windowing needed
    # Partial capture: the last covered stable kernel sits in this step.
    last_touched_sid = p_stream[n - 1][2]
    # Is that step COMPLETE? Compare stable-kernel count of that step inside the
    # prefix vs in the full plugin trace.
    in_prefix = sum(1 for _, _, s in p_stream[:n] if s == last_touched_sid)
    in_full = sum(1 for _, _, s in p_stream if s == last_touched_sid)
    if in_prefix == in_full:
        return (last_touched_sid, n)          # step fully covered
    # Mid-step cut → back off to the previous fully-covered step.
    prev = [s for _, _, s in p_stream[:n] if s != last_touched_sid]
    if not prev:
        return None  # not even one full step covered
    return (prev[-1], sum(1 for _, _, s in p_stream if s <= prev[-1]))


def _window_capture(plugin_trace, nsys_kernels, sidecar_active, boundary_sid):
    """Restrict the plugin trace, nsys kernels, and sidecar steps to the range
    the on-device capture actually covers: step_ids <= boundary_sid.

    Used only when the capture is a partial (early-stopped) PREFIX proven by
    _verify_capture_prefix. Requires per-kernel `ts_ns` on the nsys kernels to
    trim them exactly at the step boundary (mctracer `ts`/nsys start share the
    sidecar wall clock; plugin trace uses a different clock so it is trimmed by
    step_id, not time). Returns (plugin_win, nsys_win, sidecar_win) or None if
    ts_ns is unavailable (→ caller aborts rather than compare mismatched ranges).
    """
    if not nsys_kernels or "ts_ns" not in nsys_kernels[0]:
        return None
    # Cutoff = wall-clock start of the FIRST step after the boundary. Sidecar
    # iter_id == plugin step_id (verified 1:1). Keep nsys kernels strictly before
    # that start; keep plugin entries / sidecar steps with step_id <= boundary.
    next_steps = [s for s in sidecar_active if s.iter_id > boundary_sid]
    if not next_steps:
        return None  # boundary is the last step → nothing to window (shouldn't happen here)
    cutoff_ns = next_steps[0].ts_us * 1000  # sidecar µs → ns, same wall clock as ts_ns
    plugin_win = [e for e in plugin_trace if e.get("step_id", -1) <= boundary_sid]
    sidecar_win = [s for s in sidecar_active if s.iter_id <= boundary_sid]
    nsys_win = [k for k in nsys_kernels if k["ts_ns"] < cutoff_ns]
    if not plugin_win or not sidecar_win or not nsys_win:
        return None
    return plugin_win, nsys_win, sidecar_win


# Types matched by anchor-delimited blocks instead of a strict global 1:1 zip.
# The real backend sometimes fuses the two grouped-GEMM launches of one MoE
# layer into a single kernel (observed 2→1 in ~0.7% of blocks), so the FUSED_MOE
# count differs slightly between the plugin trace and nsys. TOPK_GATING fires
# exactly once per MoE layer on BOTH sides (1:1 anchor), so we align FUSED_MOE
# within each TOPK_GATING-delimited block and leave any surplus plugin kernel
# with its emulated grid (keep-plugin-grid policy — bounded, ~0.7% of MoE).
_ANCHOR_MATCHED_TYPES = {"FUSED_MOE": "TOPK_GATING"}


def _anchor_match_blocks(modified, nsys_kernels, plugin_targets, nsys_targets,
                         plugin_anchors, nsys_anchors):
    """Inject nsys grids into plugin target kernels, aligned within blocks
    delimited by anchor kernels.

    plugin_anchors / nsys_anchors must have equal length (1:1 anchor blocks —
    the caller guarantees this via the strict count check on the anchor type).
    Each target is assigned to the block = number of anchors before it
    (bisect_left over the ascending anchor positions). Within a block, plugin
    and nsys targets are zipped in order; surplus plugin targets (nsys fused
    2→1) are left untouched. Returns (matched_count). All position lists are
    strictly ascending (collected in trace order), so bisect is exact.
    """
    import bisect
    from collections import defaultdict
    p_blocks = defaultdict(list)
    for pi in plugin_targets:
        p_blocks[bisect.bisect_left(plugin_anchors, pi)].append(pi)
    n_blocks = defaultdict(list)
    for ni in nsys_targets:
        n_blocks[bisect.bisect_left(nsys_anchors, ni)].append(ni)
    matched = 0
    for b, p_list in p_blocks.items():
        n_list = n_blocks.get(b, ())
        for pi, ni in zip(p_list, n_list):
            modified[pi]["grid"] = nsys_kernels[ni]["grid"]
            matched += 1
    return matched


def _inject_nsys_grids(plugin_trace, nsys_full_path, nsys_kernels=None):
    """Replace plugin trace grids with real nsys grids for matched kernel types.

    Strict validation: count mismatch for any type raises ValueError.
    For flash_fwd (count mismatch due to split-KV), uses rotary_embedding
    positions as anchors: match the first flash_fwd after each anchor.
    Returns (modified_trace, stats_dict).

    nsys_kernels: optional pre-loaded (and possibly capture-windowed) kernel
    list. When None, the full list is read from nsys_full_path.
    """
    if nsys_kernels is None:
        with open(nsys_full_path) as f:
            nsys_data = json.load(f)
        nsys_kernels = nsys_data["kernels"]

    # Build per-type lists (preserving original order)
    plugin_by_type = {}
    for type_name, _ in _ORACLE_KERNEL_TYPES:
        plugin_by_type[type_name] = []
    plugin_flash = []
    for i, entry in enumerate(plugin_trace):
        t = _classify_oracle(entry["kernel_name"])
        if t:
            plugin_by_type[t].append(i)
        if "flash_fwd" in entry["kernel_name"]:
            plugin_flash.append(i)

    nsys_by_type = {}
    for type_name, _ in _ORACLE_KERNEL_TYPES:
        nsys_by_type[type_name] = []
    nsys_flash = []
    for j, nk in enumerate(nsys_kernels):
        t = _classify_oracle(nk["name"])
        if t:
            nsys_by_type[t].append(j)
        if "flash_fwd" in nk["name"]:
            nsys_flash.append(j)

    # Strict count validation for direct-match types. Anchor-matched types
    # (e.g. FUSED_MOE) are exempt from the strict global check — they may differ
    # by the occasional 2→1 fusion — but their ANCHOR type (TOPK_GATING) must
    # still match 1:1, which is validated below like any other direct type.
    stats = {}
    for type_name, _ in _ORACLE_KERNEL_TYPES:
        if type_name in _ANCHOR_MATCHED_TYPES:
            continue
        p_count = len(plugin_by_type[type_name])
        n_count = len(nsys_by_type[type_name])
        if p_count == 0 and n_count == 0:
            continue
        if p_count != n_count:
            raise ValueError(
                f"grid injection: count mismatch for {type_name}: "
                f"plugin={p_count}, nsys={n_count}. "
                f"Cannot safely align — aborting G2 comparison.")
        stats[type_name] = p_count

    # Shallow per-entry copy (not deepcopy): the only mutation below is
    # reassigning the top-level "grid" key on matched entries — we never mutate
    # a nested list/dict in place. dict(e) gives each entry its own top-level
    # namespace so the caller's plugin_trace is untouched, at ~8× less cost than
    # deepcopy (17s→2s on the 1.2M-entry TP2-32B trace) with identical output.
    modified = [dict(e) for e in plugin_trace]
    total_replaced = 0

    # Direct-match types: zip by global position
    for type_name in stats:
        p_indices = plugin_by_type[type_name]
        n_indices = nsys_by_type[type_name]
        for pi, ni in zip(p_indices, n_indices):
            nk = nsys_kernels[ni]
            modified[pi]["grid"] = nk["grid"]
            # Also inject the real nsys kernel_name (carries the cuBLAS tile,
            # e.g. 256x128 vs the emulated 128x128) and shared_mem, so the
            # forward traffic model (predict_gemm_traffic parses tile from
            # kernel_name) uses the real tile — otherwise GEMM traffic is
            # simulated with the wrong tile → TTFT under-prediction
            # (memory a800-ttft-underpredict-gemmprefill). dtype (fp16/bf16)
            # and stages also come from the real kernel name.
            if "name" in nk:
                modified[pi]["kernel_name"] = nk["name"]
            if "smem" in nk:
                modified[pi]["shared_mem"] = nk["smem"]
            if "block" in nk:
                modified[pi]["block"] = nk["block"]
            total_replaced += 1

    # Anchor-matched types (e.g. FUSED_MOE via TOPK_GATING blocks). The anchor
    # must have matched 1:1 (it went through the strict check above), otherwise
    # the block correspondence is undefined and we skip — the direct types are
    # already injected, so G2 still runs, just without this type's real grids.
    for tgt_type, anchor_type in _ANCHOR_MATCHED_TYPES.items():
        p_tgt = plugin_by_type.get(tgt_type, [])
        n_tgt = nsys_by_type.get(tgt_type, [])
        if not p_tgt and not n_tgt:
            continue
        p_anchor = plugin_by_type.get(anchor_type, [])
        n_anchor = nsys_by_type.get(anchor_type, [])
        if not p_anchor or len(p_anchor) != len(n_anchor):
            print(f"  [G2] {tgt_type}: anchor {anchor_type} unusable "
                  f"(plugin={len(p_anchor)}, nsys={len(n_anchor)}) — "
                  f"skipping {tgt_type} grid injection")
            continue
        matched = _anchor_match_blocks(
            modified, nsys_kernels, p_tgt, n_tgt, p_anchor, n_anchor)
        total_replaced += matched
        stats[tgt_type] = matched
        unmatched = len(p_tgt) - matched
        note = f", {unmatched} kept-as-emulated (nsys fused 2→1)" if unmatched else ""
        print(f"  [G2] {tgt_type}: {matched}/{len(p_tgt)} grid-injected via "
              f"{anchor_type} anchors{note}")

    # Flash attention: anchor-based matching using ROTARY positions
    if plugin_flash and "ROTARY" in stats:
        import bisect
        p_rotary = plugin_by_type["ROTARY"]
        n_rotary = nsys_by_type["ROTARY"]
        # Build fence intervals: (rotary_pos, next_rotary_pos) in nsys
        # For each plugin flash_fwd, find which fence it belongs to,
        # then find the first nsys flash_fwd in the corresponding nsys fence.
        #
        # p_rotary and nsys_flash are both strictly ascending (collected in
        # index order), so the two linear scans below are pure bisects. The old
        # code did an O(F·R) fence scan (~9.8B iters on TP2-32B → ~470s); this
        # is O(F·log R) and returns the IDENTICAL match for every flash:
        #   fence_idx = last p_rotary index strictly < p_flash_idx
        #             = bisect_left(p_rotary, p_flash_idx) - 1
        #   first nsys flash in [n_fence_start, n_fence_end)
        #             = first nsys_flash >= n_fence_start, if it is < n_fence_end
        flash_matched = 0
        for p_flash_idx in plugin_flash:
            fence_idx = bisect.bisect_left(p_rotary, p_flash_idx) - 1
            if fence_idx < 0:
                continue
            n_fence_start = n_rotary[fence_idx]
            n_fence_end = n_rotary[fence_idx + 1] if fence_idx + 1 < len(n_rotary) else len(nsys_kernels)
            fj = bisect.bisect_left(nsys_flash, n_fence_start)
            if fj < len(nsys_flash) and nsys_flash[fj] < n_fence_end:
                modified[p_flash_idx]["grid"] = nsys_kernels[nsys_flash[fj]]["grid"]
                flash_matched += 1
        if flash_matched > 0:
            stats["FLASH_ATTN"] = flash_matched
            total_replaced += flash_matched
            if flash_matched != len(plugin_flash):
                raise ValueError(
                    f"flash_fwd anchor matching: only matched "
                    f"{flash_matched}/{len(plugin_flash)} — aborting G2.")
            print(f"  [G2] flash_fwd: {flash_matched}/{len(plugin_flash)} matched via rotary anchors")

    return modified, stats, total_replaced


def run_g2_comparison(case, case_dir, hw_config):
    """G2-oracleGPU: nsys real grid → midlayer GPU vs nsys GPU (纯GPU, isolates timing model error)."""
    nsys_full_path = case_dir / "nsys_kernels_full.json"
    if not nsys_full_path.exists():
        print("  [G2] SKIP — no nsys_kernels_full.json (re-run nsys stage)")
        return None

    sidecar_path = str(case_dir / "plugin_trace.json") + "_scheduler.jsonl"
    if not Path(sidecar_path).exists():
        print("  [G2] SKIP — no scheduler sidecar")
        return None

    # Use the SAME regenerated trace as run_prediction (rank0), expanded from
    # operator records with the CURRENT recipe — NOT the plugin_trace.json dumped
    # at capture time (which was synthesized under whatever recipe existed then).
    # This keeps G2's plugin kernel sequence consistent with predict and with the
    # current recipe (e.g. Llama rope → ELEMENTWISE); a stale on-disk dump would
    # otherwise re-introduce kernels the recipe no longer emits and break
    # alignment. Falls back to plugin_trace.json only when there are no operator
    # records to regenerate from. Shares predict's regen cache (fast).
    from src.serve_predict.model_config import load_model_config as _load_mc
    _tp = case["tensor_parallel_size"]
    _ep = case.get("expert_parallel_size", 1)
    _mc = _load_mc(case["model"])
    _per_rank = _regenerate_trace_with_step_id(case_dir, _mc, _tp, _ep, hw_config)
    if _per_rank is not None:
        plugin_trace = _per_rank[sorted(_per_rank.keys())[0]]
    else:
        trace_path = case_dir / "plugin_trace.json"
        if not trace_path.exists():
            print("  [G2] SKIP — no plugin trace")
            return None
        with open(trace_path) as f:
            plugin_trace = json.load(f)

    with open(nsys_full_path) as f:
        _nsys_full_doc = json.load(f)
    # G2 compares against the RANK0 plugin trace, so the captured ground truth
    # must also be rank0. The top-level "kernels" is the BOTTLENECK rank (max
    # GPU), which under EP is often not rank0 (each EP rank owns different
    # experts → different kernels → alignment would fail). Prefer the explicit
    # per_rank["0"] when present; fall back to top-level for older single-stream
    # captures that have no per_rank split.
    _pr0 = (_nsys_full_doc.get("per_rank") or {}).get("0", {}).get("kernels")
    if _pr0:
        nsys_full_kernels = _pr0
        if str(_nsys_full_doc.get("bottleneck_rank")) not in ("0", "None"):
            print(f"  [G2] using nsys per_rank[0] (top-level is bottleneck "
                  f"rank {_nsys_full_doc.get('bottleneck_rank')})")
    else:
        nsys_full_kernels = _nsys_full_doc["kernels"]

    # Capture-window alignment. The on-device capture (mctracer) can stop early,
    # recording only a PREFIX of the run. Compare on the common window instead of
    # aborting: prove the captured stream is a head-aligned prefix, find the last
    # fully-covered step, and restrict BOTH sides to it. If alignment can't be
    # proven, or the capture is partial but lacks ts_ns to trim exactly, we fall
    # through with the full data — the strict count check then aborts G2 rather
    # than risk a mismatch (prefer error over mismatch).
    from src.serve_predict.serve_metric_predictor import load_scheduler_sidecar
    _sidecar_active = [s for s in load_scheduler_sidecar(sidecar_path)
                       if s.total_tokens > 0]
    _all_sids = sorted({e["step_id"] for e in plugin_trace
                        if e.get("step_id") is not None})
    nsys_kernels = nsys_full_kernels          # what injection/GPU-sum will use
    windowed_sidecar_path = sidecar_path
    _align = _verify_capture_prefix(plugin_trace, nsys_full_kernels)
    if _align is not None and _all_sids and _align[0] < _all_sids[-1]:
        boundary_sid = _align[0]
        win = _window_capture(plugin_trace, nsys_full_kernels,
                              _sidecar_active, boundary_sid)
        if win is not None:
            plugin_trace, nsys_kernels, sidecar_win = win
            n_kept = len([s for s in _all_sids if s <= boundary_sid])
            print(f"  [G2] capture is a partial prefix — windowing to "
                  f"{n_kept}/{len(_all_sids)} fully-covered steps "
                  f"(step_id<={boundary_sid}); nsys {len(nsys_kernels)}/"
                  f"{len(nsys_full_kernels)} kernels")
            # Predict needs a sidecar file matching the windowed steps.
            windowed_sidecar_path = str(case_dir / "plugin_trace_g2_sidecar.jsonl")
            with open(windowed_sidecar_path, "w") as f:
                for s in sidecar_win:
                    f.write(json.dumps({
                        "iter_id": s.iter_id, "ts_us": s.ts_us,
                        "scheduled": s.scheduled, "total_tokens": s.total_tokens,
                        "finished": s.finished}) + "\n")
        else:
            print(f"  [G2] capture partial (step<={_align[0]}) but cannot window "
                  f"exactly (no ts_ns) — proceeding on full data (may abort)")

    try:
        modified_trace, stats, n_replaced = _inject_nsys_grids(
            plugin_trace, nsys_full_path, nsys_kernels=nsys_kernels)
    except ValueError as e:
        print(f"  [G2] ERROR: {e}")
        return None

    replaced_types = sorted(stats.keys())
    print(f"  [G2] grid injection: {n_replaced} kernels replaced "
          f"({', '.join(f'{t}={stats[t]}' for t in replaced_types)})")

    g2_trace_path = case_dir / "plugin_trace_g2.json"
    if DUMP_TRACES:
        # Debug artifact only — the predictor consumes the in-memory trace
        # below (no dump→reload round-trip). Writing 303 MB+ every run cost
        # ~15s for a file nothing else reads, so gate it behind --dump-traces.
        with open(g2_trace_path, "w") as f:
            json.dump(modified_trace, f)

    from src.serve_predict.model_config import load_model_config
    from src.serve_predict.serve_metric_predictor import ServingPredictor
    from src.serve_predict.predict_lut import PredictionLUT, compute_code_version

    model_config = load_model_config(case["model"])
    tp = case["tensor_parallel_size"]

    sp = ServingPredictor(
        str(ROOT / hw_config), model_config, tp_size=tp,
        gap_threshold_us=5.0, mode=mode,
    )
    if model_config.num_experts > 0:
        sp.predictor._moe_bm_bn_bk = (16, 32, 64)

    # G2 uses nsys-injected grids, so its cache keys differ from the predict
    # LUT — keep a SEPARATE persisted LUT. Same code-version invalidation, so a
    # formula edit rebuilds it too. Annotating the grid-injected 1.2M-kernel
    # trace was ~70s/run; on a warm G2 LUT it is all cache hits.
    g2_code_ver = compute_code_version()
    g2_lut_path = case_dir / "prediction_lut_g2.json"
    g2_lut = PredictionLUT.load(g2_lut_path, g2_code_ver)
    if g2_lut is None:
        g2_lut = PredictionLUT(g2_code_ver)
    sp.predictor._lut = g2_lut

    _, summary = sp.predict_per_request_metrics(
        trace=modified_trace,
        scheduler_path=windowed_sidecar_path,
    )
    g2_lut.save(g2_lut_path)

    g2_pred_gpu_ms = sum(ns / 1e6 for ns in summary.per_iteration_gpu_ns)

    # nsys GPU total over the SAME (windowed) kernel set injected above, so the
    # comparison denominator matches the predicted numerator's step range. For a
    # full-coverage capture nsys_kernels IS the full list, so this equals the
    # previous nsys_kernels.json gpu_ns exactly (verified: gpu_ns == Σ dur).
    nsys_gpu_ms = sum(k["dur_ns"] for k in nsys_kernels) / 1e6

    # Captured-only comparison: the plugin trace only captures known
    # operator types, so nsys kernels without a plugin counterpart
    # (PyTorch elementwise, reduce, histogram, etc.) should be excluded
    # from the ground truth for a fair apples-to-apples comparison.
    # Strategy: count per-category kernels in plugin trace, then only
    # count nsys kernels from categories that appear in both. Uses the SAME
    # (windowed) nsys_kernels as the GPU-total above for a consistent range.
    from src.serve_trace.classifier import classify_kernel as _clf
    from collections import Counter as _Counter
    _plugin_cat_counts = _Counter()
    for entry in modified_trace:
        t = _classify_oracle(entry["kernel_name"])
        if t:
            _plugin_cat_counts[t] += 1
        else:
            _plugin_cat_counts[_clf(entry)] += 1

    nsys_captured_ns = 0
    nsys_uncaptured_ns = 0
    for nk in nsys_kernels:
        nk_name = nk.get("name", nk.get("kernel_name", ""))
        dur = nk.get("duration_ns", nk.get("dur_ns", 0))
        nk_oracle = _classify_oracle(nk_name)
        nk_cat = _clf({"kernel_name": nk_name}) if not nk_oracle else nk_oracle
        if _plugin_cat_counts.get(nk_cat, 0) > 0:
            nsys_captured_ns += dur
        else:
            nsys_uncaptured_ns += dur
    nsys_captured_ms = nsys_captured_ns / 1e6
    uncaptured_pct = nsys_uncaptured_ns / (nsys_captured_ns + nsys_uncaptured_ns) * 100 \
        if (nsys_captured_ns + nsys_uncaptured_ns) > 0 else 0

    ape = abs(g2_pred_gpu_ms - nsys_gpu_ms) / nsys_gpu_ms * 100 if nsys_gpu_ms > 0 else None
    ape_captured = abs(g2_pred_gpu_ms - nsys_captured_ms) / nsys_captured_ms * 100 \
        if nsys_captured_ms > 0 else None
    print(f"  [compare] G2-oracleGPU: pred={g2_pred_gpu_ms:.2f} vs "
          f"nsys_total={nsys_gpu_ms:.2f} APE={ape:.1f}% | "
          f"nsys_captured={nsys_captured_ms:.2f} APE={ape_captured:.1f}% "
          f"(uncaptured={uncaptured_pct:.1f}%)")

    return {
        "pred_gpu_ms": round(g2_pred_gpu_ms, 2),
        "nsys_gpu_ms": round(nsys_gpu_ms, 2),
        "nsys_captured_gpu_ms": round(nsys_captured_ms, 2),
        "ape": round(ape, 2) if ape is not None else None,
        "ape_captured": round(ape_captured, 2) if ape_captured is not None else None,
        "uncaptured_pct": round(uncaptured_pct, 1),
        "grid_replaced_types": replaced_types,
        "n_replaced": n_replaced,
    }


# ── Stage 5: Compare ─────────────────────────────────────────────

# TP/EP all-reduce kernels whose measured duration = pure transfer + barrier
# spin (the spin is host-idle wait for the slowest-arriving rank, NOT GPU work).
_ALLREDUCE_SPIN_KERNELS = ("cross_device_reduce",)


def _is_comm_kernel(name: str) -> bool:
    return name.startswith("nccl") or any(
        s in name for s in _ALLREDUCE_SPIN_KERNELS)


def _nsys_per_rank_compute_ms(nsys: dict) -> dict:
    """Per-rank nsys COMPUTE time in ms (communication excluded).

    G1 measures kernel-model accuracy per rank, symmetric with the predicted
    per-rank compute. Communication (all-reduce transfer + barrier spin) is
    excluded entirely and left to G3 — this both removes the host-idle spin
    (which inflated single-rank G1 to ~64%) and keeps G1 a pure compute-kernel
    metric. Returns {rank_str: compute_ms}. Single-rank/no-per_rank falls back
    to one entry from the flat total.
    """
    per_rank = nsys.get("per_rank")
    if not per_rank:
        # No per-rank breakdown (single-device capture): treat flat total,
        # still excluding comm kernels if the flat kernel list is present.
        kernels = nsys.get("kernels", [])
        compute_ns = sum(k["dur_ns"] for k in kernels
                         if not _is_comm_kernel(k["name"]))
        return {"0": compute_ns / 1e6}
    out = {}
    for r, rd in per_rank.items():
        compute_ns = sum(k["dur_ns"] for k in rd["kernels"]
                         if not _is_comm_kernel(k["name"]))
        out[r] = compute_ns / 1e6
    return out


def _nsys_category_map(name: str, model_is_moe: bool = False) -> str:
    """Classify an nsys kernel name using the SAME function as the predictor
    (classifier.classify_kernel + map_category + resolve_category). One
    function, both sides — eliminates the drift that caused pred/nsys category
    mismatches.

    resolve_category is applied symmetrically with the pred side
    (predict_kernel_time_from_trace.annotate_trace, which calls
    resolve_category(cat, self._is_moe) at dispatch). For dense models it
    demotes any FUSEDMOE→OTHER (e.g. a stray gate name); for MoE models it
    promotes ACTIVATION/REDUCE→FUSEDMOE. Without this, dense models that emit
    SwiGlu (the gated-MLP activation, classified ACTIVATION) would stay
    ACTIVATION on the nsys side but the pred side would too — symmetric — while
    a MoE model's SwiGlu is promoted on BOTH sides. model_is_moe is derived
    name-driven by the caller from the presence of MoE-only op names.
    """
    from src.serve_trace.classifier import (
        classify_kernel, map_category, resolve_category)
    return resolve_category(
        map_category(classify_kernel({"kernel_name": name})), model_is_moe)


# MoE-only CANN/GPU op-name signatures. Presence of ANY of these in the GT
# kernel list means the captured model is a MoE (grouped matmul / expert
# routing / permute kernels never appear in a dense forward). Used to derive
# model_is_moe name-driven (no model-config plumbing into run_comparison), so
# resolve_category is applied symmetrically on the GT side.
_MOE_ONLY_SIGNATURES = (
    "groupedmatmul", "grouped_gemm", "moe_gemm", "fused_moe",
    "moeinitrouting", "moegatingtopk", "moetokenunpermute",
    "moefinalizerouting", "moecomputeexperttokens",
    "moe_align", "count_and_sort", "topkgating",
)


def _nsys_model_is_moe(nsys: dict) -> bool:
    """True iff any GT kernel name is a MoE-only signature (see above)."""
    def _iter_names():
        per_rank = nsys.get("per_rank") or {}
        if per_rank:
            for rd in per_rank.values():
                for k in rd.get("kernels", []):
                    yield k.get("name", "")
        else:
            for k in nsys.get("kernels", []):
                yield k.get("name", "")
    return any(any(s in (n or "").lower() for s in _MOE_ONLY_SIGNATURES)
               for n in _iter_names())


def _category_breakdown(pred: dict, nsys: dict, common: list) -> list:
    """Per-category pred-vs-nsys breakdown for the GPU(sumup) category table.

    Buckets both the predicted per-category ms (from prediction.json's
    per_rank_category_ms, accumulated during annotate_trace) and the nsys
    per-category ms (bucketed from per-kernel names here). Returns a list of
    {name, pred_ms, nsys_ms, spe} sorted by absolute gap descending.
    Comm (NCCL/TP_COMM) is excluded from both (GPU compute only).
    spe = signed relative error (pred-nsys)/nsys*100 (+over/-under)."""
    from collections import defaultdict
    # Pred per-category: sum over common ranks.
    pred_cat = pred.get("per_rank_category_ms")
    if not pred_cat:
        return []
    # Derive is_moe name-driven from the GT kernels so the nsys side applies
    # resolve_category symmetrically with the pred side (which used the model
    # config's num_experts at prediction time).
    model_is_moe = _nsys_model_is_moe(nsys)
    pred_totals = defaultdict(float)
    for r in common:
        for cat, ms in pred_cat.get(r, {}).items():
            if cat not in ("COLLECTIVE", "CUSTOM_COMM"):
                pred_totals[cat] += ms

    # Nsys per-category: bucket per-kernel dur by name→category.
    nsys_totals = defaultdict(float)
    per_rank = nsys.get("per_rank", {})
    if per_rank:
        for r in common:
            for k in per_rank.get(r, {}).get("kernels", []):
                if not _is_comm_kernel(k["name"]):
                    nsys_totals[_nsys_category_map(k["name"], model_is_moe)] += k["dur_ns"] / 1e6
    else:
        # Flat structure (no per_rank — TP1 single-rank captures): sum all
        # kernels once. For multi-device flat, this sums all devices' kernels
        # (the total GPU time, matching pred summed over common ranks).
        for k in nsys.get("kernels", []):
            if not _is_comm_kernel(k["name"]):
                nsys_totals[_nsys_category_map(k["name"], model_is_moe)] += k["dur_ns"] / 1e6

    all_cats = sorted(set(list(pred_totals) + list(nsys_totals)),
                       key=lambda c: -abs(nsys_totals.get(c, 0) - pred_totals.get(c, 0)))
    return [
        {"name": c,
         "pred_ms": round(pred_totals.get(c, 0), 2),
         "nsys_ms": round(nsys_totals.get(c, 0), 2),
         "spe": round((pred_totals.get(c, 0) - nsys_totals.get(c, 0))
                       / nsys_totals.get(c, 0) * 100, 1)
                   if nsys_totals.get(c, 0) else None}
        for c in all_cats
    ]


def _c500_align_gemms(case_dir):
    """Align the regen trace's GEMM (M,N,K) [from op_attr] to the nsys ACTUAL
    tile [parsed from kernel name], per step, by GEMM ordinal. Returns a list of
    dicts: {M_tok, N_out, K, predicted_tile, actual_tile, step_id}. Shared
    alignment core for VALIDATION (run_comparison, goal C) and LUT GENERATION
    (scripts/kernel/gen_c500_tile_lut.py).

    Alignment principle (goal spec): mcblas tile selection is a DETERMINISTIC
    function of (M, N, K, dtype, layout) for fixed input — the autotuner is
    deterministic per shape; apparent "variance" comes from an incomplete key
    that collapses different K into one bucket. So the LUT key must carry M, N,
    K (dtype/layout constant for C500 = bf16/tn). Both the regen trace and nsys
    drive the SAME per-step GEMM recipe (qkv,o,gate_up,down per layer), so the
    i-th mcblas GEMM in step s aligns across the two traces.

    Returns None if the alignment cannot be set up.
    """
    import re as _re

    def _key(kn, grid):
        """Comparison key for a C500 GEMM/GEMV kernel.
        mcblas__Mck_bf16gemm -> the NxNxK tile string (e.g. '128x64x256').
        b16gemvt_wave_kernel -> f'gemv_gx{grid_x}' (GEMV: the name template is
          constant across all projections, so grid_x (= N_out/16) is the only
          differentiator; predicted = m//16, nsys = N//16, match iff N equal)."""
        if "b16gemvt" in kn:
            gx = grid[0] if grid else 0
            return f"gemv_gx{gx}"
        t = _re.search(r"bf16gemm_tn_(\d+x\d+x\d+)", kn)
        return t.group(1) if t else None

    plug_path = case_dir / "plugin_trace.json"
    nsys_full = case_dir / "nsys_kernels_full.json"
    sidecar_path = str(plug_path) + "_scheduler.jsonl"
    if not (plug_path.exists() and nsys_full.exists()
            and Path(sidecar_path).exists()):
        return None

    plug = json.load(open(plug_path))
    step_pred = {}
    has_mcblas = False
    for e in plug:
        oa = e.get("op_attr") or {}
        kn = e.get("kernel_name", "")
        # accept both the dense TC GEMM (mcblas__Mck) and the M=1 GEMV (b16gemvt)
        if ("mcblas__Mck" not in kn and "b16gemvt" not in kn) or "N" not in oa:
            continue
        has_mcblas = True
        sid = e.get("step_id", 0)
        m_tok, n_out, k = int(oa["M"]), int(oa["N"]), int(oa["K"])
        step_pred.setdefault(sid, []).append(
            {"M": m_tok, "N": n_out, "K": k,
             "predicted_tile": _key(kn, e.get("grid")),
             "step_id": sid})
    if not has_mcblas or not step_pred:
        return None

    from src.serve_predict.serve_metric_predictor import load_scheduler_sidecar
    sidecar_steps = [s for s in load_scheduler_sidecar(sidecar_path)
                     if s.total_tokens > 0]
    n_steps = len(sidecar_steps)
    if n_steps == 0:
        return None

    data = json.load(open(nsys_full))
    per_rank = data.get("per_rank") or {}
    by_dev = {}
    for dev, rd in per_rank.items():
        ks = rd.get("kernels") if isinstance(rd, dict) else None
        if not ks:
            continue
        kl = []
        for k in ks:
            ts = k.get("ts_ns")
            if ts is None:
                continue
            kl.append({"start": ts, "end": ts + k.get("dur_ns", 0),
                       "name": k.get("name", ""),
                       "grid": k.get("grid", [1, 1, 1])})
        kl.sort(key=lambda x: x["start"])
        if kl:
            by_dev[dev] = kl
    if not by_dev:
        return None

    devs = sorted(by_dev)
    ts_us = [s.ts_us for s in sidecar_steps]
    mc0_ns = min(by_dev[d][0]["start"] for d in devs)
    last_end_ns = max(by_dev[d][-1]["end"] for d in devs)
    boundaries = _ts_boundaries_from_sidecar(ts_us[-n_steps:], mc0_ns, last_end_ns)

    # Count-based segmentation: instead of relying on ts boundaries (which can
    # drift between runs → cyclic ordinal mismatch), use the regen trace's
    # per-step GEMM count to slice the nsys stream. This is robust to
    # timestamp format/offset differences between captures.
    dev = "0" if "0" in by_dev else devs[0]
    kl_sorted = sorted(by_dev[dev], key=lambda x: x["start"])
    # Pre-extract all mcblas/b16gemvt tiles in ts order
    all_nsys_tiles = []
    for kk in kl_sorted:
        kn = kk["name"]
        if "mcblas__Mck" not in kn and "b16gemvt" not in kn:
            continue
        key = _key(kn, kk.get("grid"))
        if key:
            all_nsys_tiles.append((key, kk.get("grid")))

    regen_steps = sorted(step_pred.keys())
    off = regen_steps[0]
    # Count-based: for each regen step, take the next N tiles from the stream
    nsys_step_tiles = []
    cursor = 0
    for sid in regen_steps:
        count = len(step_pred[sid])
        nsys_step_tiles.append(all_nsys_tiles[cursor:cursor + count])
        cursor += count

    aligned = []
    for si, sid in enumerate(regen_steps):
        acts = nsys_step_tiles[si]
        for i, rec in enumerate(step_pred[sid]):
            rec = dict(rec)
            if i < len(acts):
                rec["actual_tile"] = acts[i][0]  # tile key
                rec["actual_grid"] = acts[i][1]  # nsys actual grid [gx,gy,gz]
            else:
                rec["actual_tile"] = None
                rec["actual_grid"] = None
            aligned.append(rec)
    return aligned


def _c500_tile_alignment(case_dir, case):
    """Goal-C serving-eval tile alignment for C500/mcblas.

    Uses _c500_align_gemms to pair each regen-trace GEMM's REAL (M,N,K) [op_attr]
    with its nsys ACTUAL tile, then compares the LUT-predicted tile (re-derived
    from _maca_gemm_dispatch with the real M/N/K) against the actual tile.

    Real M_tok/N_out/K come from plugin_trace op_attr (NOT nsys grid x tile
    padding) so the comparison uses the true token count (LUT key contract).

    Per mismatch: prints `[tile_mismatch]` (ERROR) or `[tile_nondet]` (WARNING)
    when the (M_pad, N_pad, K) key is in _MACA_TILE_NONDET (genuine mcblas
    autotuner run-to-run variance -- not deterministically reproducible, not a
    model bug). Prints a summary `tile_match: X/Y (Z%)` line.
    """
    import re as _re
    from src.serve_trace.analytic_mirror import (
        _maca_gemm_dispatch, _maca_lmhead_dispatch, _MACA_TILE_NONDET,
    )
    # lm_head detection + (tp, ep) for the lm_head dispatch: load model config
    # for vocab; tp/ep from the case. op_attr N for lm_head == full vocab, so
    # n_out == vocab identifies the lm_head (only projection with N=vocab).
    try:
        from src.serve_predict.model_config import load_model_config
        _vocab = load_model_config(case["model"]).vocab_size
        _tp = case.get("tensor_parallel_size", 1)
        _ep = case.get("expert_parallel_size", 1)
    except Exception:  # noqa: BLE001
        _vocab = _tp = _ep = None

    aligned = _c500_align_gemms(case_dir)
    if not aligned:
        return None

    from collections import Counter
    mismatch_counts = Counter()
    nondet_counts = Counter()
    n_aligned = 0
    n_match = 0
    n_miss = 0
    n_nondet = 0
    n_assert_miss = 0

    for rec in aligned:
        m_tok, n_out, k = rec["M"], rec["N"], rec["K"]
        actual = rec.get("actual_tile")
        if actual is None:
            n_miss += 1
            continue
        # lm_head M>=2 (bf16gemm path): op_attr records FULL vocab but nsys is
        # the per-rank shard (vocab/tp) → LUT key (M, full_vocab, K) mismatches
        # across EP2/EP4. Skip as shard-unverifiable (align_miss, not nondet).
        if _vocab is not None and n_out == _vocab and m_tok > 1:
            n_miss += 1
            continue
        # C500 lm_head: route to the (tp,ep)-aware lm_head dispatch (op_attr N
        # = full vocab; lm_head is always M=1). ONLY at M<=1: lm_head uses the
        # b16gemvt GEMV (decode batch=1, or prefill last-token). At M>=2 (decode
        # batch>=2) lm_head uses mcblas bf16gemm TC (the LUT path, unchanged).
        is_lm_head = (_vocab is not None and n_out == _vocab and m_tok <= 1)
        try:
            if is_lm_head:
                _ent = _maca_lmhead_dispatch(n_out, _tp, _ep)[0]
            else:
                _ent = _maca_gemm_dispatch(n_out, m_tok, k)[0]
        except AssertionError:
            n_assert_miss += 1
            continue
        # Dispatch returns 6-tuple (name, gx, gy, bx, by, smem) or 7-tuple
        # (name, gx, gy, gz, bx, by, smem) — handle both.
        if len(_ent) == 7:
            tname, gx, _gy, _gz, _blk, _by, _sm = _ent
        else:
            tname, gx, _gy, _blk, _by, _sm = _ent
        # GEMV (b16gemvt, M=1 dense) -> key by grid_x (= N_out/16); TC GEMM ->
        # the bf16gemm NxNxK tile. Matches _key() in _c500_align_gemms.
        if "b16gemvt" in tname:
            pred_tile = f"gemv_gx{gx}"
        else:
            _t = _re.search(r"bf16gemm_tn_(\d+x\d+x\d+)", tname)
            pred_tile = _t.group(1) if _t else None
        if pred_tile is None:
            n_miss += 1
            continue
        n_aligned += 1
        if pred_tile == actual:
            n_match += 1
        else:
            # EXACT (M_tok, N_out, K) nondet key (matches _MACA_TILE_NONDET, which
            # is exact-keyed — no padding, since the autotuner keys on exact M).
            if (m_tok, n_out, k) in _MACA_TILE_NONDET:
                n_nondet += 1
                nondet_counts[(m_tok, n_out, k, pred_tile, actual)] += 1
            else:
                mismatch_counts[(m_tok, n_out, k, pred_tile, actual)] += 1

    cid = case.get("id", case_dir.name)
    for (m, n, k, p, a), c in sorted(mismatch_counts.items(),
                                     key=lambda x: -x[1]):
        print(f"  [tile_mismatch] case={cid} M={m} N={n} K={k} "
              f"predicted_tile={p} actual_tile={a} count={c}")
    for (m, n, k, p, a), c in sorted(nondet_counts.items(),
                                    key=lambda x: -x[1])[:10]:
        print(f"  [tile_nondet]   case={cid} M={m} N={n} K={k} "
              f"predicted_tile={p} actual_tile={a} count={c} "
              f"(autotuner variance -- warning, not error)")

    total = n_aligned
    pct = (n_match / total * 100) if total else 0.0
    print(f"  [tile_align] case={cid}: tile_match={n_match}/{total} "
          f"({pct:.1f}%), tile_mismatch={sum(mismatch_counts.values())}, "
          f"tile_nondet={n_nondet}, assert_miss={n_assert_miss}, "
          f"align_miss={n_miss}")
    return {
        "tile_match": n_match,
        "tile_total": total,
        "tile_match_pct": round(pct, 2),
        "tile_mismatch": sum(mismatch_counts.values()),
        "tile_mismatch_detail": [
            {"M": m, "N": n, "K": k, "predicted": p, "actual": a, "count": c}
            for (m, n, k, p, a), c in mismatch_counts.items()],
        "tile_nondet": n_nondet,
        "tile_nondet_detail": [
            {"M": m, "N": n, "K": k, "predicted": p, "actual": a, "count": c}
            for (m, n, k, p, a), c in nondet_counts.items()],
        "assert_miss": n_assert_miss,
        "align_miss": n_miss,
    }


def _gpu_only_latency_split(pred, case_dir):
    """GPU-only, CPU-free latency-quality metric: decompose per-step GPU time
    into PREFILL steps (the work that drives TTFT) and DECODE steps (the work
    that drives TPOT), and score predicted vs real GPU busy time in each class.

    Motivation: wall-clock TTFT/TPOT are dominated by CPU/scheduler/idle (70%
    on A800, ~97% on 910B), so they barely test the kernel model and are not
    comparable across cards with different host-overhead fractions. Here the CPU
    is removed on BOTH sides (predicted GPU vs measured GPU), so the number is a
    pure kernel-model quality signal, split by the phase each latency metric
    cares about, and directly comparable across hardware and across the 3
    methods (Ours / Roofline / PipeWeave — each mode's compare pass writes its
    own split into its comparison.json).

    Per-step is the correct granularity, not per-request: under continuous
    batching a single prefill/mixed step blends many requests' tokens, so its
    GPU time is not attributable to one request — a per-request GPU-only TTFT is
    ill-defined. A step is PREFILL if any scheduled request has >1 token
    (prefill or chunked-prefill/mixed), else DECODE (all tokens == 1).

    Primary stat = weighted APE (Σ|pred−gt| / Σgt): robust to steps whose
    measured GPU is ~0 (per-step MAPE explodes there). Also reports per-step
    MAPE with a GT floor, and signed SPE (Σ(pred−gt)/Σgt) to expose the over/
    under direction. Returns None when per-step GPU ground truth is unavailable
    (e.g. C500 mcTracer gives only a whole-system wall, no per-step GPU).
    """
    import numpy as np
    from src.serve_predict.serve_metric_predictor import load_scheduler_sidecar

    ps = pred.get("per_step") or []
    if not ps or "nsys_gpu_ms" not in ps[0]:
        return None
    sidecar_path = case_dir / "plugin_trace.json_scheduler.jsonl"
    if not sidecar_path.exists():
        return None
    active = [s for s in load_scheduler_sidecar(str(sidecar_path))
              if s.total_tokens > 0]
    if not active:
        return None
    # iter_id → is_prefill(any token count > 1)
    is_prefill = {s.iter_id: any(t > 1 for t in s.scheduled.values())
                  for s in active}

    buckets = {"prefill": ([], []), "decode": ([], [])}  # (pred[], gt[])
    for e in ps:
        sid = e.get("step_id")
        if sid not in is_prefill or "nsys_gpu_ms" not in e:
            continue
        # GT: prefer the §5 program-order attribution (nsys_gpu_ms_s5, 910B),
        # which assigns msprof GPU busy by kernel issue order and so does not
        # leak the high-qps prefill burst into decode the way the host-ts
        # nsys_gpu_ms (§4 path) does. Falls back to nsys_gpu_ms (A800 nsys,
        # low-qps 910B) when absent.
        gt = e.get("nsys_gpu_ms_s5", e["nsys_gpu_ms"])
        key = "prefill" if is_prefill[sid] else "decode"
        buckets[key][0].append(float(e["midlayer_gpu_ms"]))
        buckets[key][1].append(float(gt))

    GT_FLOOR_MS = 0.05  # exclude near-zero-GT steps from per-step MAPE only
    out = {}
    for key, (p, g) in buckets.items():
        p = np.array(p, dtype=float)
        g = np.array(g, dtype=float)
        n = int(len(g))
        gsum = float(g.sum())
        if n == 0 or gsum <= 0:
            out[key] = {"n": n, "wtd_ape": None, "spe": None,
                        "mape": None, "pred_sum_ms": round(float(p.sum()), 2),
                        "gt_sum_ms": round(gsum, 2)}
            continue
        wtd_ape = float(np.sum(np.abs(p - g)) / gsum * 100)
        spe = float(np.sum(p - g) / gsum * 100)
        m = g >= GT_FLOOR_MS
        mape = float(np.mean(np.abs(p[m] - g[m]) / g[m]) * 100) if m.any() else None
        out[key] = {
            "n": n,
            "wtd_ape": round(wtd_ape, 2),
            "spe": round(spe, 2),
            "mape": round(mape, 2) if mape is not None else None,
            "pred_sum_ms": round(float(p.sum()), 2),
            "gt_sum_ms": round(gsum, 2),
        }
    return out


def run_comparison(case, case_dir):
    pred_path = case_dir / "prediction.json"
    nsys_path = case_dir / "nsys_kernels.json"
    real_path = case_dir / "real_wallclock.json"
    real_metrics_path = case_dir / "real_metrics.json"
    serving_metrics_path = case_dir / "serving_metrics.json"

    if not pred_path.exists():
        print("  [compare] SKIP — no prediction")
        return None

    with open(pred_path) as f:
        pred = json.load(f)

    # For serving cases, derive wall-clock ground truth. Priority:
    #   real_wallclock.json (dedicated benchmark) > real_metrics.json >
    #   serving_metrics.json (the ShareGPT serving runs store wall_s here).
    real_ms = None
    if real_path.exists():
        with open(real_path) as f:
            real_ms = json.load(f)["median_wall_ms"]
    elif real_metrics_path.exists():
        with open(real_metrics_path) as f:
            real_ms = json.load(f)["wall_s"] * 1000
    elif serving_metrics_path.exists():
        with open(serving_metrics_path) as f:
            real_ms = json.load(f)["wall_s"] * 1000

    comparison = {"case_id": case["id"]}

    # ── G1 synthGPU: per-rank predicted COMPUTE vs nsys per-rank COMPUTE ──
    # Per-rank (not per-step-max): G1 measures kernel-model accuracy, decoupled
    # from multi-rank assembly. Each rank is an independent single-card trace;
    # comm is excluded (left to G3). Aggregate = GPU-time-weighted APE
    # (Σ|pred−nsys| / Σ nsys), so busy ranks dominate and near-idle ranks don't
    # skew it. Single-rank cases degenerate to the previous behavior.
    if nsys_path.exists():
        with open(nsys_path) as f:
            nsys = json.load(f)
        nsys_compute = _nsys_per_rank_compute_ms(nsys)          # {rank: ms}
        pred_pr = pred.get("per_rank_pred")
        if pred_pr:
            pred_compute = {r: v["compute_ms"] for r, v in pred_pr.items()}
        else:
            # Legacy prediction.json without per_rank_pred: fall back to the
            # aggregate compute (gpu_ms has no comm split here — best effort).
            pred_compute = {"0": pred["e2e_gpu_only_ms"]}
        common = sorted(set(pred_compute) & set(nsys_compute))
        per_rank_ape = {}
        sum_abs = 0.0
        sum_ref = 0.0
        for r in common:
            p = pred_compute[r]
            n = nsys_compute[r]
            per_rank_ape[r] = round(abs(p - n) / n * 100, 2) if n > 0 else None
            sum_abs += abs(p - n)
            sum_ref += n
        weighted_ape = round(sum_abs / sum_ref * 100, 2) if sum_ref > 0 else None
        comparison["group1_synth_gpu"] = {
            "ape": weighted_ape,                 # GPU-time-weighted per-rank APE
            "per_rank_ape": per_rank_ape,
            "per_rank_pred_ms": {r: round(pred_compute[r], 2) for r in common},
            "per_rank_nsys_ms": {r: round(nsys_compute[r], 2) for r in common},
            # Back-compat aggregate totals (sum over ranks) for existing readers.
            "pred_ms": round(sum(pred_compute[r] for r in common), 2),
            "nsys_ms": round(sum(nsys_compute[r] for r in common), 2),
            "n_ranks": len(common),
        }
        # Per-category breakdown (G1 category table). Bucket both pred and nsys
        # by kernel category so the full_eval serving table can show which
        # category drives the G1 residual. Pred comes from prediction.json's
        # per_rank_category_ms (accumulated during annotate_trace); nsys is
        # bucketed here from the per-kernel name list.
        cat_breakdown = _category_breakdown(pred, nsys, common)
        if cat_breakdown:
            comparison["group1_synth_gpu"]["category_breakdown"] = cat_breakdown
            cat_str = ", ".join(
                f"{c['name']}={c['spe']:+.1f}%" if c.get('spe') is not None
                else f"{c['name']}=n/a"
                for c in cat_breakdown[:5])
            print(f"  [compare] GPU(sumup) category: {cat_str}")
        pr_str = ", ".join(f"r{r}={per_rank_ape[r]}%" for r in common)
        print(f"  [compare] GPU(sumup) (per-rank compute, wtd): "
              f"APE={weighted_ape}% [{pr_str}]")

    # ── G6 GPU-only latency: prefill/decode GPU-time accuracy (CPU-free) ──
    # The fair, hardware-comparable latency-quality metric — see
    # _gpu_only_latency_split. prefill≈TTFT-side model quality, decode≈TPOT-side.
    gpu_only = _gpu_only_latency_split(pred, case_dir)
    if gpu_only is not None:
        comparison["group6_gpu_only_latency"] = gpu_only
        _pf, _dc = gpu_only.get("prefill", {}), gpu_only.get("decode", {})
        print(f"  [compare] GPU-only latency (CPU-free): "
              f"prefill(TTFT) wtd-APE={_pf.get('wtd_ape')}% (SPE={_pf.get('spe')}%, n={_pf.get('n')}) | "
              f"decode(TPOT) wtd-APE={_dc.get('wtd_ape')}% (SPE={_dc.get('spe')}%, n={_dc.get('n')})")

    # GPU/wall fraction: how much of the real e2e wall the predicted GPU
    # occupies. Low (~30%) = CPU/framework-bound serving → G3/G4 APE is small
    # mostly because the metric is dominated by REAL captured CPU, NOT because
    # the GPU model is accurate (G3 APE ≈ G1 APE × this fraction). High (>70%)
    # = GPU-bound → G3/G4 approach G1 and become a real GPU-model validation.
    # Computed once from the GPU-only prediction vs the real wall.
    gpu_wall_frac = None
    if real_ms and real_ms > 0 and pred.get("e2e_gpu_only_ms"):
        gpu_wall_frac = pred["e2e_gpu_only_ms"] / real_ms

    # ── G3 realE2E: synth grid + real CPU → vs real wall clock (有CPU) ──
    if real_ms is not None and pred.get("e2e_real_cpu_ms") is not None:
        pred_rc = pred["e2e_real_cpu_ms"]
        rc_ape = abs(pred_rc - real_ms) / real_ms * 100 if real_ms > 0 else None
        comparison["group3_real_e2e"] = {
            "pred_ms": round(pred_rc, 2),
            "real_ms": round(real_ms, 2),
            "ape": round(rc_ape, 2) if rc_ape is not None else None,
            "gpu_wall_fraction_pct": round(gpu_wall_frac * 100, 1)
                if gpu_wall_frac is not None else None,
        }
        _gwf = f"{gpu_wall_frac*100:.1f}%" if gpu_wall_frac is not None else "-"
        print(f"  [compare] GPU+CPU(real): pred={pred_rc:.2f} vs real={real_ms:.2f} "
              f"→ APE={rc_ape:.1f}%  (GPU/wall={_gwf})")
    elif real_ms is not None:
        # No per-step CPU decomposition (C500 mcTracer gives only a whole-system
        # wall, not per-step gpu_real/idle) → pred_ms/ape stay None. But wall%
        # only needs e2e_gpu_only_ms / real_ms, both of which exist here, so
        # still report it (else C500 shows blank wall% despite it being ~29%).
        comparison["group3_real_e2e"] = {
            "pred_ms": None, "real_ms": round(real_ms, 2),
            "ape": None,
            "gpu_wall_fraction_pct": round(gpu_wall_frac * 100, 1)
                if gpu_wall_frac is not None else None,
        }
        _gwf = f"{gpu_wall_frac*100:.1f}%" if gpu_wall_frac is not None else "-"
        print(f"  [compare] GPU+CPU(real): nsys per-step unavailable "
              f"(GPU/wall={_gwf})")

    # ── G4 plugE2E: synth grid + plugin CPU → vs real wall clock (有CPU) ──
    if real_ms is not None and pred.get("e2e_plugin_cpu_ms") is not None:
        pred_pc = pred["e2e_plugin_cpu_ms"]
        pc_ape = abs(pred_pc - real_ms) / real_ms * 100 if real_ms > 0 else None
        comparison["group4_plug_e2e"] = {
            "pred_ms": round(pred_pc, 2),
            "real_ms": round(real_ms, 2),
            "ape": round(pc_ape, 2) if pc_ape is not None else None,
            "gpu_wall_fraction_pct": round(gpu_wall_frac * 100, 1)
                if gpu_wall_frac is not None else None,
        }
        print(f"  [compare] GPU+CPU(plugin): pred={pred_pc:.2f} vs real={real_ms:.2f} "
              f"→ APE={pc_ape:.1f}%")
    elif real_ms is not None:
        comparison["group4_plug_e2e"] = {
            "pred_ms": None, "real_ms": round(real_ms, 2),
            "ape": None,
            "gpu_wall_fraction_pct": round(gpu_wall_frac * 100, 1)
                if gpu_wall_frac is not None else None,
        }
        print("  [compare] GPU+CPU(plugin): plugin per-step unavailable")

    # ── G5 serving-metrics: TTFT / TPOT / Throughput, pred vs real client ──
    # The serving metrics that matter most, compared per-request (joined by
    # request_id) plus aggregate throughput. Data exists on both sides:
    #   GT  serving_metrics.json: throughput_rps + per-request ttft/tpot/e2e
    #   pred prediction.json:     per-request ttft/tpot/e2e
    # NOTE on definitions (diagnosed): GT ttft = (first_token - send) real
    # client-observed (incl. real queueing); pred ttft = modeled prefill
    # compute+CPU+gap. Same intent (time-to-first-token), different composition,
    # so the APE reflects genuine model error, not a join mismatch. TPOT: GT
    # (e2e-ttft)/out_tokens; pred = decode-iteration amortized.
    if serving_metrics_path.exists() and pred.get("per_request"):
        with open(serving_metrics_path) as f:
            gt_sm = json.load(f)
        # GT driver records bare "req_<N>"; the predictor keeps vLLM's full
        # "req_<N>-<uuid8>" (plus a "warmup_*" request). Normalize the pred key
        # to the "req_<N>" prefix so the per-request join actually matches —
        # otherwise the intersection is empty and TTFT/TPOT come out null.
        # Some captures (older vLLM / certain TP1 traces) record the predictor's
        # req_id as a bare integer "1".."N" (vLLM's internal 1-based counter)
        # while the GT driver uses "req_0".."req_{N-1}" (0-based). Map a bare
        # integer N -> "req_{N-1}" so the join matches across capture styles.
        def _norm_rid(rid):
            rid = str(rid)
            if rid.isdigit():
                return f"req_{int(rid) - 1}"
            return rid.split("-", 1)[0]
        gt_by_id = {r["request_id"]: r for r in gt_sm.get("per_request", [])}
        pred_by_id = {_norm_rid(r["request_id"]): r for r in pred["per_request"]}
        common_ids = sorted(set(gt_by_id) & set(pred_by_id))

        import numpy as _np

        def _dist(vals):
            """vLLM-style distribution of a per-request value list: mean/std/
            median + P10/P50/P90 (np.percentile, linear interpolation — same as
            vllm/benchmarks/serve.py calculate_metrics)."""
            a = _np.asarray(vals, dtype=float)
            return {
                "mean": round(float(a.mean()), 2),
                "std": round(float(a.std()), 2),
                "p10": round(float(_np.percentile(a, 10)), 2),
                "p50": round(float(_np.percentile(a, 50)), 2),
                "p90": round(float(_np.percentile(a, 90)), 2),
            }

        def _agg(metric):
            """Aggregate a per-request metric (ttft_ms / tpot_ms) the way vLLM
            does: keep the ONE value each request already carries (TPOT =
            decode-time / decode-tokens), then report the DISTRIBUTION (mean/std/
            median + P10/P50/P90) for pred and gt separately, plus a per-request
            MAPE (mean of |p-g|/g over requests — NOT a token-weighted sum)."""
            ps, gs, apes = [], [], []
            for rid in common_ids:
                p = pred_by_id[rid].get(metric)
                g = gt_by_id[rid].get(metric)
                if p is None or g is None or g <= 0:
                    continue
                ps.append(p); gs.append(g)
                apes.append(abs(p - g) / g * 100)
            if not ps:
                return None
            pd, gd = _dist(ps), _dist(gs)
            # Per-percentile SIGNED error: (pred_pX - gt_pX) / gt_pX. Compares the
            # two DISTRIBUTIONS at the same quantile (pred P90 vs gt P90); the
            # sign shows direction (+ = over-predict, - = under-predict). Distinct
            # from `mape` (absolute mean per-request |p-g|/g), which stays the
            # unsigned headline metric.
            def _spe(a, b):
                return round((a - b) / b * 100, 2) if b else None
            spe = {q: _spe(pd[q], gd[q]) for q in ("mean", "p10", "p50", "p90")}
            return {
                "pred": pd,
                "gt": gd,
                "spe": spe,          # signed % error at mean / P10 / P50 / P90
                "mape": round(float(_np.mean(apes)), 2),  # mean per-request APE (abs)
                "n": len(ps),
            }

        ttft = _agg("ttft_ms")
        tpot = _agg("tpot_ms")
        # Throughput: requests per second over the (predicted vs real) wall time.
        # pred wall = e2e_plugin_cpu_ms (same pred-wall as G4); GT = throughput_rps.
        gt_thrpt = gt_sm.get("throughput_rps")
        pred_wall_ms = pred.get("e2e_plugin_cpu_ms")
        n_req = gt_sm.get("num_requests") or len(common_ids)
        pred_thrpt = (n_req / (pred_wall_ms / 1000.0)
                      if pred_wall_ms and pred_wall_ms > 0 else None)
        thrpt = None
        if gt_thrpt and pred_thrpt:
            thrpt = {
                "pred_rps": round(pred_thrpt, 4),
                "gt_rps": round(gt_thrpt, 4),
                # signed % error (+ = over-predict, - = under-predict)
                "spe": round((pred_thrpt - gt_thrpt) / gt_thrpt * 100, 2),
            }
        comparison["group5_serving_metrics"] = {
            "ttft": ttft, "tpot": tpot, "throughput": thrpt,
            "n_requests": len(common_ids),
        }
        def _fmt(m):
            if not m:
                return "n/a"
            pr, gt, sp = m["pred"], m["gt"], m["spe"]
            return (f"pred P10/50/90={pr['p10']:.0f}/{pr['p50']:.0f}/{pr['p90']:.0f} "
                    f"gt={gt['p10']:.0f}/{gt['p50']:.0f}/{gt['p90']:.0f} "
                    f"SPE={sp['p10']:+.0f}/{sp['p50']:+.0f}/{sp['p90']:+.0f}% "
                    f"MAPE={m['mape']:.1f}%")
        print(f"  [compare] TTFT:  {_fmt(ttft)}")
        print(f"  [compare] TPOT:  {_fmt(tpot)}")
        if thrpt:
            print(f"  [compare] Thrpt: pred={thrpt['pred_rps']:.3f} "
                  f"gt={thrpt['gt_rps']:.3f} rps SPE={thrpt['spe']:+.1f}%")

    # ── G0 tile alignment (C500/mcblas only): predicted vs nsys actual tile ──
    # Goal-C kernel-launch validation. The case is C500 iff the regen trace
    # emitted mcblas__Mck kernels (gated on _is_maca at dispatch time); detect
    # by scanning plugin_trace for an mcblas kernel rather than by hw config
    # so this path is self-contained in the comparison artifacts on disk.
    try:
        tile_align = _c500_tile_alignment(case_dir, case)
    except Exception as _e:  # noqa: BLE001
        tile_align = None
        print(f"  [tile_align] SKIPPED: {_e}")
    if tile_align is not None:
        comparison["tile_alignment"] = tile_align

    cmp_path = case_dir / "comparison.json"
    with open(cmp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    return comparison


def run_comparison_with_g2(case, case_dir, hw_config):
    """Run standard comparison + G2-oracleGPU comparison."""
    comparison = run_comparison(case, case_dir)
    if comparison is None:
        return None

    g2 = run_g2_comparison(case, case_dir, hw_config)
    if g2 is not None:
        comparison["group2_oracle_gpu"] = g2
        cmp_path = case_dir / "comparison.json"
        with open(cmp_path, "w") as f:
            json.dump(comparison, f, indent=2)

    return comparison


# ── Summary ──────────────────────────────────────────────────────

def write_summary(output_root, all_comparisons):
    has_qps = any(c and "qps" in c.get("case_id", "") for c in all_comparisons)
    has_g2 = any(c and "group2_oracle_gpu" in c for c in all_comparisons)
    if has_qps:
        title = "# Serving Eval Dataset — Summary"
        header = "| Case | Model | p | d | QPS | G1 synthGPU | G2 oracleGPU |"
        sep = "|------|-------|---|---|-----|-------------|--------------|"
        header += " G3 realE2E | G4 plugE2E | TTFT MAPE | TPOT MAPE | Thrpt SPE |"
        sep += "------------|------------|-----------|-----------|-----------|"
    else:
        title = "# Offline Eval Dataset — Summary"
        header = "| Case | Model | p | d | n | G1 synthGPU | G2 oracleGPU |"
        sep = "|------|-------|---|---|---|-------------|--------------|"
        header += " G3 realE2E | G4 plugE2E |"
        sep += "------------|------------|"

    lines = [title, "", f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "", header, sep]
    for c in all_comparisons:
        if c is None:
            continue
        cid = c.get("case_id", "?")
        parts = cid.split("_")
        model_tag = parts[1] if len(parts) > 1 else "?"

        p = d = n_or_qps = "?"
        for part in parts:
            if part.startswith("p") and part[1:].isdigit():
                p = part[1:]
            elif part.startswith("d") and part[1:].isdigit():
                d = part[1:]
            elif part.startswith("n") and part[1:].isdigit():
                n_or_qps = part[1:]
            elif part.startswith("qps") and part[3:].isdigit():
                n_or_qps = part[3:]

        g1 = c.get("group1_synth_gpu", {})
        g2 = c.get("group2_oracle_gpu", {})
        g3 = c.get("group3_real_e2e", {})
        g4 = c.get("group4_plug_e2e", {})

        g1a = f"{g1['ape']:.1f}%" if g1.get("ape") is not None else "—"
        g2a = f"{g2['ape']:.1f}%" if g2.get("ape") is not None else "—"
        g3a = f"{g3['ape']:.1f}%" if g3.get("ape") is not None else "—"
        g4a = f"{g4['ape']:.1f}%" if g4.get("ape") is not None else "—"

        if has_qps:
            g5 = c.get("group5_serving_metrics", {})
            ttft = g5.get("ttft") or {}
            tpot = g5.get("tpot") or {}
            thr = g5.get("throughput") or {}
            t_a = f"{ttft['mape']:.1f}%" if ttft.get("mape") is not None else "—"
            p_a = f"{tpot['mape']:.1f}%" if tpot.get("mape") is not None else "—"
            th_a = f"{thr['spe']:+.1f}%" if thr.get("spe") is not None else "—"
            row = (f"| {cid} | {model_tag} | {p} | {d} | {n_or_qps} | "
                   f"{g1a} | {g2a} | {g3a} | {g4a} | {t_a} | {p_a} | {th_a} |")
        else:
            row = (f"| {cid} | {model_tag} | {p} | {d} | {n_or_qps} | "
                   f"{g1a} | {g2a} | {g3a} | {g4a} |")
        lines.append(row)

    lines.append("")
    summary_path = Path(output_root) / "summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[summary] Written to {summary_path}")


# ── Main ─────────────────────────────────────────────────────────

def _run_single_case(case, output_root, hw_config, stages, profiler=None, mode="midlayer",
                      pipeweave_model_hw=None):
    """Run all requested stages for a single case. Returns comparison dict or None."""
    cid = case["id"]
    case_dir = output_root / cid
    case_dir.mkdir(parents=True, exist_ok=True)
    is_serving = _is_serving_case(case)

    meta = {
        "case_id": cid,
        "model": case["model"],
        "num_requests": case["num_requests"],
        "tensor_parallel_size": case["tensor_parallel_size"],
        "expert_parallel_size": case.get("expert_parallel_size", 1),
        "gpu_ids": case["gpu_ids"],
        "hw_config": hw_config,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if _is_sharegpt_case(case):
        meta["workload"] = "sharegpt"
        meta["dataset_path"] = case["dataset"]
        meta["sample_seed"] = case.get("sample_seed", 42)
    else:
        meta["prompt_len"] = case["prompt_len"]
        meta["max_tokens"] = case["max_tokens"]
    if is_serving:
        meta["qps"] = case["qps"]
        meta["poisson_seed"] = case.get("poisson_seed", case.get("sample_seed", 42))
    with open(case_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Case: {cid}")
    print(f"  Model: {Path(case['model']).name}")
    if _is_sharegpt_case(case):
        print(f"  Workload: ShareGPT n={case['num_requests']}" +
              (f" QPS={case['qps']}" if is_serving else ""))
    else:
        print(f"  Workload: p={case['prompt_len']} d={case['max_tokens']} "
              f"n={case['num_requests']}" +
              (f" QPS={case['qps']}" if is_serving else ""))
    print(f"  Parallel: TP={case['tensor_parallel_size']} "
          f"EP={case.get('expert_parallel_size', 1)}")
    print(f"  GPUs: {case['gpu_ids']}")
    print(f"{'='*60}")

    if is_serving and ("nsys" in stages or "real" in stages):
        schedule_path = case_dir / "arrival_schedule.json"
        if not schedule_path.exists():
            _generate_arrival_schedule(case, case_dir)

    if "nsys" in stages:
        if profiler == "mctracer":
            if is_serving:
                ok = run_serving_mctracer_capture(case, case_dir, hw_config)
            else:
                ok = run_mctracer_capture(case, case_dir, hw_config)
        else:
            if is_serving:
                ok = run_serving_nsys_capture(case, case_dir)
            else:
                ok = run_nsys_capture(case, case_dir)
        if not ok:
            print(f"  [nsys+plugin] *** FAILED — skipping remaining stages")
            return None

    if "plugin" in stages and "nsys" not in stages:
        ok = run_plugin_capture(case, case_dir)
        if not ok:
            print(f"  [plugin] *** FAILED — skipping remaining stages")
            return None

    if "real" in stages:
        if _is_sharegpt_case(case):
            print("  [real] SKIP — real benchmark not supported for ShareGPT workloads")
        elif is_serving:
            run_serving_real_benchmark(case, case_dir)
        else:
            run_real_benchmark(case, case_dir)

    if "predict" in stages:
        try:
            run_prediction(case, case_dir, hw_config, mode=mode,
                           pipeweave_model_hw=pipeweave_model_hw)
        except Exception as e:
            import traceback
            print(f"  [predict] ERROR: {e}")
            traceback.print_exc()

    cmp = None
    if "compare" in stages:
        cmp = run_comparison(case, case_dir)  # G1+G3+G4 (skip G2 oracle)
    return cmp


def _run_gpu_group(group_cases, output_root, hw_config, stages, profiler=None, mode="midlayer",
                    pipeweave_model_hw=None):
    """Run cases sequentially within a GPU group. Returns list of comparisons."""
    results = []
    for case in group_cases:
        cmp = _run_single_case(case, output_root, hw_config, stages, profiler, mode,
                               pipeweave_model_hw=pipeweave_model_hw)
        results.append((case["id"], cmp))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Build eval dataset (offline or serving)")
    parser.add_argument("--config",
                        default=str(ROOT / "configs" / "eval" / "offline_eval_cases.yaml"),
                        help="Config YAML path, or 'offline' / 'sharegpt' shorthand")
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Run only these case IDs (default: all)")
    parser.add_argument("--stage", nargs="*", default=None,
                        choices=STAGES,
                        help="Run only these stages (default: all)")
    parser.add_argument("-j", "--parallel", type=int, default=os.cpu_count(),
                        help="Number of parallel workers (default: cpu count)")
    parser.add_argument("--profiler", choices=["mctracer"], default=None,
                        help="GPU profiler for nsys stage: 'mctracer' for MetaX C500. "
                             "Default: nsys (NVIDIA).")
    parser.add_argument("--dump-traces", action="store_true",
                        help="Write the regenerated per-rank kernel trace to disk "
                             "(plugin_trace_regen*.json, 200 MB+ each). Default OFF: "
                             "predict uses the in-memory trace directly (no dump→"
                             "reload round-trip). Enable only for debugging or when "
                             "the G2 stage needs the regen file on disk.")
    parser.add_argument("--mode", choices=["midlayer", "roofline", "pipeweave"], default="midlayer",
                        help="Prediction mode: midlayer (full physics model), "
                             "roofline (naive max(compute, memory) baseline), or "
                             "pipeweave (PipeWeave ISCA'26 analytical+MLP baseline). "
                             "Default: midlayer.")
    parser.add_argument("--pipeweave-model-hw", default=None,
                        help="PipeWeave: hardware whose MLP models to use. "
                             "None = same as target hw (native MLP). "
                             "'A100' = cross-hardware eval with A100 MLP.")
    args = parser.parse_args()

    global DUMP_TRACES
    DUMP_TRACES = args.dump_traces

    config_path = args.config
    if config_path == "offline":
        config_path = OFFLINE_YAML
    elif config_path == "sharegpt":
        config_path = SHAREGPT_YAML

    cfg, cases = load_cases(config_path, args.cases)
    output_root = ROOT / cfg["output_root"]
    hw_config = cfg["hw_config"]
    stages = set(args.stage) if args.stage else set(STAGES)

    print(f"[eval] {len(cases)} cases, stages={sorted(stages)}")
    print(f"[eval] output: {output_root}")

    output_root = Path(output_root)

    gpu_stages = {"nsys", "real", "plugin"}
    has_gpu_stages = bool(stages & gpu_stages)

    if has_gpu_stages:
        import threading
        from collections import defaultdict
        from concurrent.futures import ThreadPoolExecutor, as_completed

        gpu_groups = defaultdict(list)
        for case in cases:
            key = tuple(sorted(case["gpu_ids"]))
            gpu_groups[key].append(case)

        gpu_locks = {}
        for gpu_set in gpu_groups:
            for gpu_id in gpu_set:
                if gpu_id not in gpu_locks:
                    gpu_locks[gpu_id] = threading.Lock()

        print(f"[eval] GPU-group parallel: {len(gpu_groups)} groups "
              f"({', '.join(f'GPU{list(k)}×{len(v)}' for k, v in sorted(gpu_groups.items()))})")

        all_comparisons_map = {}

        def _run_group_with_locks(gpu_set, group_cases):
            locks = sorted(gpu_set)
            for g in locks:
                gpu_locks[g].acquire()
            try:
                return _run_gpu_group(group_cases, output_root, hw_config, stages, args.profiler,
                                      mode=args.mode, pipeweave_model_hw=args.pipeweave_model_hw)
            finally:
                for g in locks:
                    gpu_locks[g].release()

        with ThreadPoolExecutor(max_workers=len(gpu_groups)) as pool:
            future_to_key = {
                pool.submit(_run_group_with_locks, gpu_set, group_cases): gpu_set
                for gpu_set, group_cases in gpu_groups.items()
            }
            for future in as_completed(future_to_key):
                gpu_set = future_to_key[future]
                try:
                    for cid, cmp in future.result():
                        all_comparisons_map[cid] = cmp
                except Exception as e:
                    print(f"  [GPU{list(gpu_set)}] ERROR: {e}")

        all_comparisons = [all_comparisons_map.get(c["id"]) for c in cases]
    else:
        n_workers = args.parallel
        if n_workers > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                future_to_case = {
                    pool.submit(_run_single_case, case, output_root, hw_config, stages, args.profiler,
                                mode=args.mode, pipeweave_model_hw=args.pipeweave_model_hw): case
                    for case in cases
                }
                all_comparisons = [None] * len(cases)
                case_to_idx = {id(case): i for i, case in enumerate(cases)}
                for future in as_completed(future_to_case):
                    case = future_to_case[future]
                    idx = case_to_idx[id(case)]
                    try:
                        all_comparisons[idx] = future.result()
                    except Exception as e:
                        print(f"  [{case['id']}] ERROR: {e}")
                        all_comparisons[idx] = None
        else:
            all_comparisons = []
            for case in cases:
                cmp = _run_single_case(case, output_root, hw_config, stages, args.profiler,
                                       mode=args.mode, pipeweave_model_hw=args.pipeweave_model_hw)
                all_comparisons.append(cmp)

    write_summary(output_root, all_comparisons)


if __name__ == "__main__":
    main()
