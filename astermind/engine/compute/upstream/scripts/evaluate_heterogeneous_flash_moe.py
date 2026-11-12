#!/usr/bin/env python3
"""Run the small metadata-only Flash GPU/FPGA MoE sweep."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
from evaluation.serving.metrics.heterogeneous_flash_moe import evaluate_heterogeneous_flash_moe, render_report

DEFAULT_GPU_HARDWARE_CONFIG = str(
    REPOSITORY_ROOT / "configs/hardware/hygon_bw1100.yaml"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--compare-request-sweeps", nargs=2, metavar=("BASELINE", "SCENARIO"),
        help="Summarize the two request sweeps together; unfinished results remain incomplete")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-vector-memory-only", action="store_true",
        help="Explicit Roofline approximation: GPU vectors price logical memory only; GPU matrix and FPGA compute remain")
    parser.add_argument("--scope", choices=("moe", "decode_layer", "decode_model", "prefill_model", "request", "request_sweep"), default="moe")
    parser.add_argument("--gpu-utilization", type=float, default=1.0,
        help="Request sweep effective GPU compute and local memory rate fraction; network/delays unchanged")
    parser.add_argument("--gpu-hardware-config", default=DEFAULT_GPU_HARDWARE_CONFIG,
        help="Request-sweep GPU aggregate Roofline configuration; preserves its actual hardware identity")
    parser.add_argument("--fpga-utilization", type=float, default=1.0,
        help="Request sweep effective FPGA MM/vector/HBM/DDR rate fraction; clocks/network/delays unchanged")
    parser.add_argument("--workers", type=int, default=8,
        help="Maximum CPU worker processes; inherited CPU affinity and thread settings are preserved")
    parser.add_argument("--reuse-completed-from",
        help="Explicitly reuse completed identical request cases after metadata-cache-only changes; retain original snapshot")
    parser.add_argument("--layer-count", type=int, default=1)
    parser.add_argument("--kv-length", type=int, default=1)
    parser.add_argument("--concurrencies", type=int, nargs="+")
    parser.add_argument("--microbatches", type=int, nargs="+")
    parser.add_argument("--fpga-cards", type=int, nargs="+")
    parser.add_argument("--output-tokens", type=int, default=3)
    parser.add_argument("--scheduler-kv-bytes-per-rank", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--sdk-python", help="vLLM 0.30 SDK Python; required for non-sweep scopes")
    parser.add_argument("--sdk-source", help="vLLM 0.30 source root; required for non-sweep scopes")
    args = parser.parse_args()
    if args.compare_request_sweeps is not None:
        from evaluation.serving.metrics.heterogeneous_flash_request_sweep import compare_request_sweeps
        result = compare_request_sweeps(*args.compare_request_sweeps, args.output_dir)
        print(json.dumps({"status": result["status"], "report": str(Path(args.output_dir) / "report.md")}))
        return
    if args.scope == "request_sweep":
        from evaluation.serving.metrics.heterogeneous_flash_request_sweep import evaluate_request_sweep
        snapshot = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "uncommitted_changes": subprocess.check_output(["git", "status", "--short"], text=True).splitlines()}
        result = evaluate_request_sweep(output_dir=args.output_dir,
            concurrencies=tuple(args.concurrencies) if args.concurrencies is not None else (1, 4, 16, 32),
            microbatches=tuple(args.microbatches) if args.microbatches is not None else (1, 2, 3),
            fpga_cards=tuple(args.fpga_cards) if args.fpga_cards is not None else (8, 16),
            prompt_tokens=args.kv_length, output_tokens=args.output_tokens, seed=args.seed,
            gpu_vector_memory_only=args.gpu_vector_memory_only, workers=args.workers, snapshot=snapshot,
            reuse_completed_from=args.reuse_completed_from,
            gpu_hardware_config=args.gpu_hardware_config,
            gpu_utilization=args.gpu_utilization, fpga_utilization=args.fpga_utilization)
        print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
            "workers": result["workers"], "report": str(Path(args.output_dir) / "report.md"), "GPU_used": False}))
        if result["status"] != "completed":
            raise SystemExit(1)
        return
    if (args.gpu_utilization, args.fpga_utilization) != (1.0, 1.0):
        parser.error("utilization options require --scope request_sweep")
    if args.gpu_hardware_config != DEFAULT_GPU_HARDWARE_CONFIG:
        parser.error("GPU hardware configuration selection requires --scope request_sweep")
    if args.sdk_python is None or args.sdk_source is None:
        parser.error("--sdk-python and --sdk-source are required for non-sweep scopes")
    request = args.scope == "request"
    concurrencies = tuple(args.concurrencies) if args.concurrencies is not None else ((1,) if request else (1, 4, 16, 32))
    microbatches = tuple(args.microbatches) if args.microbatches is not None else ((1,) if request else (1, 2))
    fpga_cards = tuple(args.fpga_cards) if args.fpga_cards is not None else ((8,) if request else (8, 16))
    result = evaluate_heterogeneous_flash_moe(seed=args.seed, scope=args.scope, layer_count=args.layer_count,
        kv_length=args.kv_length, concurrencies=concurrencies, microbatches=microbatches,
        fpga_cards=fpga_cards, output_tokens=args.output_tokens,
        sdk_python=args.sdk_python, sdk_source=args.sdk_source, workers=args.workers,
        scheduler_kv_bytes_per_rank=args.scheduler_kv_bytes_per_rank,
        gpu_vector_memory_only=args.gpu_vector_memory_only)
    result["snapshot"] = {"commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
        "uncommitted_changes": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPOSITORY_ROOT, text=True).splitlines()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (output / "report.md").write_text(render_report(result))
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
        "report": str(output / "report.md"), "GPU_used": False}))


if __name__ == "__main__":
    main()
