"""Update the canonical Kernel/full-Serving report from existing evaluators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]

from evaluation.reporting.standard_evaluation_report import (
    KERNEL_HARDWARE_ORDER,
    REQUIRED_KERNEL_HARDWARE,
    timestamp_now,
    write_kernel_evaluation_report,
    write_serving_evaluation_report,
)

from evaluation.json_shape import write_json_atomic as _write_json_atomic


DEFAULT_OUTPUT = ROOT / "experiment_logs/evaluation/standard_report"
SERVING_DATASET_ROOT = ROOT / "serving_dataset/sharegpt"
DEFAULT_SERVING_PYTHON = Path(
    "/home/yangxinhao/miniconda3/envs/r2r_qwen3_next/bin/python"
)
_KERNEL_TIMING_LINE = re.compile(
    r"^\[kernel\]\s+(?P<hardware>[a-z0-9_]+):\s+"
    r"(?P<status>PASS|FAIL|SKIPPED)\s+\((?P<seconds>[0-9]+(?:\.[0-9]+)?)s\)$"
)


def _arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--kernel-only", action="store_true")
    mode.add_argument("--serving-only", action="store_true")
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--jobs", type=int, choices=range(1, 65), default=45,
        metavar="1..64", help="parallel full Serving case workers (default: 45)",
    )
    parser.add_argument(
        "--kernel-jobs", type=int, choices=range(1, 65), default=16,
        metavar="1..64",
    )
    parser.add_argument(
        "--case", action="append", default=[],
        help="representative Serving hardware/model/case_id; repeat as needed",
    )
    parser.add_argument(
        "--reuse-serving-results",
        action="append",
        type=Path,
        default=[],
        help="completed exact-cell results to reuse in a Serving-only update",
    )
    parser.add_argument(
        "--nvidia-h200-vllm-version", choices=("0.30.0", "0.11.0", "none"),
        default="0.30.0",
        help=("H200 Serving cells of this captured vLLM version (canonical: "
              "0.30.0); 'none' leaves H200 out"),
    )
    parser.add_argument(
        "--nvidia-h200-dataset-root", type=Path,
        help=(
            "non-canonical H200 candidate root (<root>/nvidia_h200/<model>/"
            "<case>) of --nvidia-h200-vllm-version, e.g. the vLLM 0.11.0 "
            "capture; default: serving_dataset/sharegpt"),
    )
    parser.add_argument(
        "--serving-python",
        type=Path,
        default=DEFAULT_SERVING_PYTHON,
        help="Python executable containing the pinned vLLM runtime",
    )
    return parser.parse_args(argv)


def _run_checked(command: list[str], log_path: Path) -> float:
    started = time.perf_counter()
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "0"
    with log_path.open("w") as log_file, subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as process:
        if process.stdout is None:
            raise RuntimeError("evaluation subprocess stdout is unavailable")
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    elapsed = time.perf_counter() - started
    if returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {returncode}; see {log_path}"
        )
    return elapsed


def _run_kernel(output_dir: Path, stamp: str, jobs: int) -> dict[str, object]:
    component_dir = output_dir / "components" / "kernel" / stamp
    component_dir.mkdir(parents=True, exist_ok=False)
    result_path = component_dir / "kernel_matrix.json"
    evaluation_log = component_dir / "evaluation.log"
    evaluation_wall = _run_checked([
        sys.executable,
        "scripts/evaluation/run_full_evaluation.py",
        "--part", "kernel",
        # Keep cards serial: each card receives the entire fixed CPU set for
        # its own 45-case-worker measurement instead of three cards competing
        # with 135 processes.  The standard report's --kernel-jobs therefore
        # controls case workers, matching the public report contract.
        "--kernel-jobs", "1",
        "--kernel-case-jobs", str(jobs),
        "--kernel-case-timing-dir", str(component_dir / "case_timing"),
        "--kernel-semantic-output-dir", str(component_dir / "semantic"),
        "--dump-json", str(result_path),
    ], evaluation_log)
    hardware_wall_times = {}
    for line in evaluation_log.read_text().splitlines():
        match = _KERNEL_TIMING_LINE.fullmatch(line)
        if match is None:
            continue
        hardware_id = match.group("hardware")
        if hardware_id in hardware_wall_times:
            raise ValueError(f"duplicate Kernel timing row for {hardware_id}")
        hardware_wall_times[hardware_id] = float(match.group("seconds"))
    if not set(REQUIRED_KERNEL_HARDWARE) <= set(hardware_wall_times):
        raise ValueError("Kernel evaluation log is missing hardware wall times")
    reported_hardware = sorted(
        hardware_id for hardware_id in KERNEL_HARDWARE_ORDER
        if hardware_id in hardware_wall_times)
    overall_plot = component_dir / "kernel_latency_overall.png"
    detailed_plot = component_dir / "kernel_latency_by_bucket.png"
    plot_wall = _run_checked([
        sys.executable,
        "scripts/evaluation/kernel/kernel_latency_scatter_report.py",
        "--jobs", str(jobs),
        "--output", str(overall_plot),
        "--detailed-output", str(detailed_plot),
    ], component_dir / "plot.log")
    traffic_plot = component_dir / "kernel_memory_traffic.png"
    traffic_summary = component_dir / "kernel_memory_traffic.json"
    traffic_wall = _run_checked([
        sys.executable,
        "scripts/evaluation/kernel/kernel_memory_traffic_report.py",
        "--jobs", str(jobs),
        "--output", str(traffic_plot),
        "--summary-output", str(traffic_summary),
    ], component_dir / "traffic_plot.log")
    return {
        "generated_at": timestamp_now(),
        "source_commit": _source_commit(),
        "result_path": str(result_path.resolve()),
        "overall_plot_path": str(overall_plot.resolve()),
        "detailed_plot_path": str(detailed_plot.resolve()),
        "traffic_plot_path": str(traffic_plot.resolve()),
        "traffic_summary_path": str(traffic_summary.resolve()),
        "evaluation_wall_time_s": evaluation_wall,
        "hardware_evaluation_wall_time_s": {
            hardware_id: hardware_wall_times[hardware_id]
            for hardware_id in reported_hardware
        },
        "semantic_artifact_paths": {
            hardware_id: str((component_dir / "semantic" / f"{hardware_id}.json").resolve())
            for hardware_id in reported_hardware
            if (component_dir / "semantic" / f"{hardware_id}.json").is_file()
        },
        "plot_wall_time_s": plot_wall,
        "traffic_plot_wall_time_s": traffic_wall,
    }


def _run_serving(
    output_dir: Path,
    stamp: str,
    jobs: int,
    cases: tuple[str, ...],
    reuse_results: tuple[Path, ...],
    serving_python: Path,
    nvidia_h200_vllm_version: str | None,
    nvidia_h200_dataset_root: Path | None,
) -> dict[str, object]:
    component_dir = output_dir / "components" / "serving" / stamp
    component_dir.mkdir(parents=True, exist_ok=False)
    run_id = f"standard-report-{stamp}"
    command = [
        str(serving_python.resolve(strict=True)),
        "scripts/evaluation/serving/diagnostics/run_scheduler_depth_breakdown.py",
        "--run-id", run_id,
        "--jobs", str(jobs),
        "--full-serving-report-only",
        "--dataset-root", str(SERVING_DATASET_ROOT.resolve(strict=True)),
    ]
    if cases:
        for cell_id in cases:
            command.extend(("--case", cell_id))
    else:
        pass
    for result_path in reuse_results:
        command.extend(("--reuse-results", str(result_path.resolve(strict=True))))
    if nvidia_h200_vllm_version == "none":
        nvidia_h200_vllm_version = None
    if nvidia_h200_dataset_root is not None and nvidia_h200_vllm_version is None:
        raise ValueError(
            "--nvidia-h200-dataset-root requires --nvidia-h200-vllm-version")
    if nvidia_h200_vllm_version is not None:
        command.extend(("--nvidia-h200-vllm-version", nvidia_h200_vllm_version))
    if nvidia_h200_dataset_root is not None:
        h200_root = nvidia_h200_dataset_root.resolve(strict=True)
        if (ROOT / "serving_dataset") in (h200_root, *h200_root.parents):
            raise ValueError(
                "an H200 candidate root must not live in serving_dataset/")
        command.extend(("--nvidia-h200-dataset-root", str(h200_root)))
    command_wall = _run_checked(command, component_dir / "evaluation.log")
    result_dir = ROOT / "experiment_logs/serving/sharegpt_full" / run_id
    result_path = result_dir / "results.json"
    if not result_path.is_file():
        raise RuntimeError(f"Serving evaluator did not write {result_path}")
    result = json.loads(result_path.read_text())
    if not isinstance(result, dict):
        raise TypeError("Serving result must be a JSON object")
    evaluation_wall = result["evaluation_wall_time_seconds"]
    if isinstance(evaluation_wall, bool) or not isinstance(
        evaluation_wall, (int, float)
    ):
        raise TypeError("Serving evaluation wall time must be numeric")
    plot_timing = json.loads(
        (result_dir / "report_generation_timing.json").read_text()
    )
    if not isinstance(plot_timing, dict):
        raise TypeError("Serving plot timing must be a JSON object")
    return {
        "generated_at": timestamp_now(),
        "source_commit": str(result["source_commit"]),
        "result_path": str(result_path.resolve()),
        "evaluation_wall_time_s": float(evaluation_wall),
        "plot_wall_time_s": float(plot_timing["plot_wall_time_seconds"]),
        "command_wall_time_s": command_wall,
        "case_count": int(result["cell_count"]),
        "nvidia_h200_vllm_version": nvidia_h200_vllm_version,
        "nvidia_h200_dataset_root": (
            None if nvidia_h200_dataset_root is None
            else str(nvidia_h200_dataset_root.resolve(strict=True))),
    }


def _load_state(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(
            f"{path} does not exist; use --all for the first standard report run"
        )
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value["schema_version"] != 1:
        raise ValueError("standard report state is invalid")
    return value




def _source_commit() -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT,
        check=True, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def run(argv=None) -> tuple[Path, ...]:
    args = _arguments(argv)
    output_dir = args.output_dir.resolve()
    if output_dir != ROOT and ROOT not in output_dir.parents:
        raise ValueError("standard report output must remain inside the repository")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if args.all:
        # Both component runners inherit the caller's fixed CPU affinity.
        # Running their 45-worker pools together makes the Kernel card timing
        # depend on the concurrent Serving mix, so it is not a valid
        # cross-hardware timing comparison.  Each component still evaluates
        # its independent cases in parallel internally.
        kernel = _run_kernel(output_dir, stamp, args.kernel_jobs)
        serving = _run_serving(
            output_dir,
            stamp,
            args.jobs,
            tuple(args.case),
            tuple(args.reuse_serving_results),
            args.serving_python,
            args.nvidia_h200_vllm_version,
            args.nvidia_h200_dataset_root,
        )
        update_mode = "all"
    else:
        if state_path.is_file():
            previous = _load_state(state_path)
        elif args.serving_only:
            previous = {"kernel": None}
        else:
            previous = _load_state(state_path)
        if args.kernel_only:
            kernel = _run_kernel(output_dir, stamp, args.kernel_jobs)
            serving = previous["serving"]
            update_mode = "kernel-only"
        elif args.serving_only:
            kernel = previous["kernel"]
            serving = _run_serving(
                output_dir,
                stamp,
                args.jobs,
                tuple(args.case),
                tuple(args.reuse_serving_results),
                args.serving_python,
                args.nvidia_h200_vllm_version,
                args.nvidia_h200_dataset_root,
            )
            update_mode = "serving-only"
        else:
            raise AssertionError("one standard report mode must be selected")
    state = {
        "schema_version": 1,
        "source_commit": _source_commit(),
        "generated_at": timestamp_now(),
        "update_mode": update_mode,
        "kernel": kernel,
        "serving": serving,
    }
    _write_json_atomic(state_path, state)
    report_paths = []
    assembly_wall_times = {}
    if update_mode in {"all", "kernel-only"}:
        kernel_report = output_dir / "kernel_report.md"
        assembly_wall_times["kernel_report.md"] = write_kernel_evaluation_report(
            kernel_report, state
        )
        report_paths.append(kernel_report)
    if update_mode in {"all", "serving-only"}:
        serving_report = output_dir / "serving_report.md"
        assembly_wall_times["serving_report.md"] = write_serving_evaluation_report(
            serving_report, state
        )
        report_paths.append(serving_report)
    legacy_report = output_dir / "report.md"
    if legacy_report.exists():
        legacy_report.unlink()
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            output_dir / "kernel_report.md",
            output_dir / "serving_report.md",
        )
        if path.is_file()
    }
    hashes["state.json"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
    _write_json_atomic(output_dir / "hashes.json", {
        "schema_version": 2,
        "report_assembly_wall_time_s": assembly_wall_times,
        "sha256": hashes,
    })
    return tuple(report_paths)


def main(argv=None) -> int:
    report_paths = run(argv)
    for report_path in report_paths:
        print(report_path)
    return 0
