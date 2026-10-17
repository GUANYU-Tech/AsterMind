#!/usr/bin/env python3
"""Collect every data input behind the CAL'26 paper figures and tables.

One run writes experiment_logs/paper_plot/cal26/<stamp>/ with:

  standard_report/        Kernel + full ShareGPT Serving standard report
                          (Table 1, Table 2 and the Fig. 4 request CDFs)
  memory_traffic.json     measured vs predicted L2/HBM traffic (Fig. 2)
  mechanism_ablation.json cumulative mechanism ladder, 910B Qwen3-4B QPS4 (Fig. 3)
  manifest.json           source commit and the path of every input

The DeepSeek-V3 H200 sweep using the R1 recipe for Fig. 5 is written under
<run-dir>/dsr1_sweep/ and referenced from the manifest.
The four collections are independent and
run in parallel; each keeps its log under logs/.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PARENT = ROOT / "experiment_logs/paper_plot/cal26"
ABLATION_CASE = "ascend_910b/qwen3-4b/tp1_qps4"
DSR1_RUNNER = (ROOT / "scripts/evaluation/serving/"
               "run_inferencex_deepseek_v3_8k1024_closedloop_comparison.py")
# The published suite binds the eager vLLM recipe and immutable inputs.
# Original C=1/2 inputs remain available only through the paper tag.
DSR1_ARGS = (
    "--settings", "vllm_colocated_mtp_off", "vllm_colocated_mtp_on",
    "vllm_pd_mtp_off", "vllm_pd_mtp_on",
    "--concurrencies", "4", "8", "16", "32", "64",
    "--max-workers", "20",
)
INFERENCEX_RAW = ROOT / "experiment_logs/inferencex_alignment/raw/dsr1_all_20260924.json"
STEPS = ("report", "traffic", "ablation", "dsr1")


def _git(*args: str) -> str:
    return subprocess.run(("git", "-C", str(ROOT), *args), check=True,
                          capture_output=True, text=True).stdout.strip()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def collect_traffic(output: Path, jobs: int) -> None:
    from evaluation.kernel.memory_traffic_report import (
        HARDWARE,
        aggregate_case_traffic,
        collect_hardware_traffic,
    )

    # Counter comparability (A800 Vector cold DRAM envelope, C500 read-only L2
    # path sum) is owned by memory_traffic_report, not re-derived here.
    traffic = collect_hardware_traffic(jobs)
    hardware = []
    for hardware_id, _title in HARDWARE:
        entry = traffic[hardware_id]
        metrics = {}
        for metric in ("l2", "hbm"):
            cases, excluded, unmatched = aggregate_case_traffic(
                entry["kernel_points"], metric)
            metrics[metric] = {
                "cases": cases,
                "excluded_case_count": len(excluded),
                "unmatched_kernel_count": unmatched,
            }
        hardware.append({
            "hardware_id": hardware_id,
            "hardware_config": entry["hardware_config"],
            "dataset_case_count": entry["dataset_case_count"],
            "fa_serving_capture_case_count": len(
                entry["fa_serving_capture_case_ids"]),
            "kernel_count": len(entry["kernel_points"]),
            "metrics": metrics,
        })
    _write_json(output, {"schema_version": 2, "hardware": hardware})


def collect_ablation(output: Path) -> None:
    from dataset_pipeline.serving_dataset import load_compact_case
    from evaluation.serving.metrics.diagnostics.mechanism_ablation import (
        run_mechanism_ablation,
    )

    _write_json(output, run_mechanism_ablation(ROOT, ABLATION_CASE,
                                               load_compact_case))


def _commands(run_dir: Path, jobs: int) -> dict[str, list[str]]:
    this = str(Path(__file__).resolve())
    return {
        "report": [sys.executable,
                   str(ROOT / "scripts/evaluation/run_standard_evaluation_report.py"),
                   "--all", "--output-dir", str(run_dir / "standard_report")],
        "traffic": [sys.executable, this, "--step", "traffic",
                    "--output-dir", str(run_dir), "--jobs", str(jobs)],
        "ablation": [sys.executable, this, "--step", "ablation",
                     "--output-dir", str(run_dir)],
        "dsr1": [sys.executable, str(DSR1_RUNNER), *DSR1_ARGS,
                 "--output-dir", str(run_dir / "dsr1_sweep")],
    }


def _run_logged(command: list[str], log: Path) -> int:
    with log.open("w", encoding="utf-8") as stream:
        return subprocess.run(command, cwd=ROOT, stdout=stream,
                              stderr=subprocess.STDOUT).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="run directory (default: "
                             "experiment_logs/paper_plot/cal26/<stamp>)")
    parser.add_argument("--jobs", type=int, default=16,
                        help="traffic collection workers")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate the Fig. 5 suite invocation without running evaluations")
    parser.add_argument("--step", choices=("traffic", "ablation"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.validate_only:
        if args.step is not None:
            parser.error("--validate-only cannot be combined with --step")
        return subprocess.run(
            [sys.executable, str(DSR1_RUNNER), *DSR1_ARGS, "--validate-only"],
            cwd=ROOT, check=False).returncode

    if args.step is not None:
        if args.output_dir is None:
            parser.error("--step requires --output-dir")
        if args.step == "traffic":
            collect_traffic(args.output_dir / "memory_traffic.json", args.jobs)
        else:
            collect_ablation(args.output_dir / "mechanism_ablation.json")
        return 0

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.output_dir or DEFAULT_PARENT / stamp).resolve()
    (run_dir / "logs").mkdir(parents=True, exist_ok=False)
    source_commit = _git("rev-parse", "HEAD")
    source_tree_clean = _git("status", "--porcelain") == ""
    commands = _commands(run_dir, args.jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(STEPS)) as pool:
        futures = {pool.submit(_run_logged, commands[name],
                               run_dir / "logs" / f"{name}.log"): name
                   for name in STEPS}
        failed = []
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            code = future.result()
            print(f"{name}: exit {code}", flush=True)
            if code != 0:
                failed.append(name)
    if failed:
        print(f"failed: {failed}; see {run_dir / 'logs'}", file=sys.stderr)
        return 1

    state = json.loads((run_dir / "standard_report/state.json").read_text(
        encoding="utf-8"))
    _write_json(run_dir / "manifest.json", {
        "schema_version": 2,
        "created_at": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
        "source_commit": source_commit,
        "source_tree_clean": source_tree_clean,
        "kernel_report": str(run_dir / "standard_report/kernel_report.md"),
        "kernel_results": state["kernel"]["result_path"],
        "serving_report": str(run_dir / "standard_report/serving_report.md"),
        "serving_results": state["serving"]["result_path"],
        "memory_traffic": str(run_dir / "memory_traffic.json"),
        "mechanism_ablation": str(run_dir / "mechanism_ablation.json"),
        "dsr1_sweep": str(run_dir / "dsr1_sweep"),
        "inferencex_raw": {
            "path": str(INFERENCEX_RAW),
            "sha256": hashlib.sha256(INFERENCEX_RAW.read_bytes()).hexdigest(),
        },
    })
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
