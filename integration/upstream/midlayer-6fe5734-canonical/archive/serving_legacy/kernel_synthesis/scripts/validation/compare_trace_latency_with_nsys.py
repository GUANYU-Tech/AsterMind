"""
Legacy trace-pricing comparison against nsys ground truth.

Usage:
    python scripts/serving/validation/compare_trace_latency_with_nsys.py \\
        --nsys /tmp/e2e_comparison/nsys_data.json \\
        --trace /tmp/e2e_comparison/emu_trace.json
"""
import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.kernel.validation.dataset_io import get_emulation_output_dir

MODEL = "/share_data/public_models/Qwen3-0.6B"
SEQ_LEN = 128
MAX_TOKENS = 8


def extract_inference_trace(full_trace):
    """Extract inference-only portion (after last big timestamp gap)."""
    big_gap_idx = 0
    for i in range(1, len(full_trace)):
        gap = full_trace[i]["timestamp_us"] - full_trace[i - 1]["timestamp_us"]
        if gap > 100000:  # > 100ms
            big_gap_idx = i
    return full_trace[big_gap_idx:]


def predict_and_compare(nsys_data, trace, model=MODEL, seq_len=SEQ_LEN,
                        max_tokens=MAX_TOKENS, hw_config=None):
    """Predict GPU time per kernel, compare with nsys."""
    from src.serve_predict.predict_kernel_time_from_trace import TracePredictor
    from src.serving.workload.model_config import load_model_config

    if hw_config is None:
        hw_config = str(ROOT / "configs/hardware/nvidia_a800.yaml")

    model_config = load_model_config(model)
    predictor = TracePredictor(hw_config, model_config)
    annotations = predictor.annotate_trace(trace, seq_len, batch_size=1)

    gpu_pred_ns = sum(a.predicted_ns for a in annotations)
    cpu_pred_ns = sum(a.cpu_overhead_ns for a in annotations)
    e2e_pred_ns = gpu_pred_ns + cpu_pred_ns

    nsys_e2e_ns = nsys_data["span_ns"]
    nsys_gpu_ns = nsys_data["gpu_ns"]
    nsys_cpu_ns = nsys_e2e_ns - nsys_gpu_ns

    e2e_pred_ms = e2e_pred_ns / 1e6
    gpu_pred_ms = gpu_pred_ns / 1e6
    cpu_pred_ms = cpu_pred_ns / 1e6
    nsys_e2e_ms = nsys_e2e_ns / 1e6
    nsys_gpu_ms = nsys_gpu_ns / 1e6
    nsys_cpu_ms = nsys_cpu_ns / 1e6

    gpu_err = abs(gpu_pred_ms - nsys_gpu_ms) / nsys_gpu_ms * 100
    e2e_err = abs(e2e_pred_ms - nsys_e2e_ms) / nsys_e2e_ms * 100

    cat_pred = defaultdict(lambda: {"count": 0, "ns": 0.0})
    for a in annotations:
        cat_pred[a.category]["count"] += 1
        cat_pred[a.category]["ns"] += a.predicted_ns

    cat_nsys = nsys_data["cat"]
    all_cats = sorted(
        set(list(cat_pred.keys()) + list(cat_nsys.keys())),
        key=lambda c: cat_nsys[c]["ns"] if c in cat_nsys else 0,
        reverse=True,
    )

    print()
    print("=" * 72)
    print("  End-to-End Latency: Midlayer Prediction vs nsys Ground Truth")
    print("=" * 72)
    print("  Model:    %s" % model.split("/")[-1])
    print("  Hardware:  NVIDIA A800-SXM4-80GB, TP=1")
    print("  Framework: vLLM 0.11.0, enforce_eager=True")
    print("  Workload:  prefill %d tokens + decode %d tokens" % (seq_len, max_tokens))
    print("  nsys e2e:  first_kernel_start -> last_kernel_end")
    print("=" * 72)
    print()
    print("  %-35s %12s %12s %8s" % ("", "Predicted", "nsys", "Error"))
    print("  %-35s %12s %12s %8s" % ("-" * 35, "-" * 12, "-" * 12, "-" * 8))
    print("  %-35s %10.2fms %10.2fms %6.1f%%" % (
        "E2E latency", e2e_pred_ms, nsys_e2e_ms, e2e_err))
    print("  %-35s %10.2fms %10.2fms %6.1f%%" % (
        "  GPU kernel time", gpu_pred_ms, nsys_gpu_ms, gpu_err))
    print("  %-35s %10.2fms %10.2fms" % (
        "  CPU framework overhead", cpu_pred_ms, nsys_cpu_ms))
    print()

    print("  Per-Category GPU Breakdown:")
    print("  %-15s %6s %6s %10s %10s %8s" % (
        "Category", "#pred", "#nsys", "pred(ms)", "nsys(ms)", "error"))
    print("  %-15s %6s %6s %10s %10s %8s" % (
        "-" * 15, "-" * 6, "-" * 6, "-" * 10, "-" * 10, "-" * 8))
    for cat in all_cats:
        p_ns = cat_pred[cat]["ns"] if cat in cat_pred else 0
        n_ns = cat_nsys[cat]["ns"] if cat in cat_nsys else 0
        pc = cat_pred[cat]["count"] if cat in cat_pred else 0
        nc = cat_nsys[cat]["count"] if cat in cat_nsys else 0
        p_ms = p_ns / 1e6
        n_ms = n_ns / 1e6
        err = abs(p_ms - n_ms) / n_ms * 100 if n_ms > 0 else 0
        print("  %-15s %6d %6d %10.2f %10.2f %7.1f%%" % (
            cat, pc, nc, p_ms, n_ms, err))

    print()
    print("  Kernel count: predicted %d, nsys %d" % (
        len(annotations), nsys_data["n_kernels"]))
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Compare emulation prediction vs nsys ground truth")
    _emu_dir = get_emulation_output_dir()
    parser.add_argument("--nsys",
                        default=os.path.join(_emu_dir, "nsys_data.json"),
                        help="Path to nsys_data.json from capture_nsys.py")
    parser.add_argument("--trace",
                        default=os.path.join(_emu_dir, "eager_trace.json"),
                        help="Path to emulation trace JSON")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--hw-config", default="configs/hardware/nvidia_a800.yaml")
    parser.add_argument("--inference-only", action="store_true", default=True,
                        help="Extract inference-only portion of trace (default: True)")
    args = parser.parse_args()

    with open(args.nsys) as f:
        nsys_data = json.load(f)
    with open(args.trace) as f:
        full_trace = json.load(f)

    trace = extract_inference_trace(full_trace) if args.inference_only else full_trace
    print(f"[trace] Full: {len(full_trace)}, inference: {len(trace)} kernels")

    predict_and_compare(
        nsys_data, trace,
        model=args.model, seq_len=args.seq_len, max_tokens=args.max_tokens,
        hw_config=str(ROOT / args.hw_config),
    )


if __name__ == "__main__":
    main()
