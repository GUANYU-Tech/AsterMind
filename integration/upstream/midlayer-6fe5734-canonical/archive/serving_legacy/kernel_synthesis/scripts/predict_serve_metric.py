#!/usr/bin/env python3
"""
Predict per-request serving metrics (TTFT, TPOT, e2e) from emulation trace.

Requires:
  - Merged trace JSON (from run_vllm_serving_emulation.py + merge_traces.py)
  - Scheduler sidecar JSONL (from scheduler monkey-patch)

Usage:
    python scripts/serve_predict/predict_serve_metric.py \
        --trace /tmp/vllm_serving_trace.json \
        --model /share_data/public_models/Qwen3-0.6B

    # With explicit scheduler sidecar path
    python scripts/serve_predict/predict_serve_metric.py \
        --trace /tmp/vllm_serving_trace.json \
        --scheduler /tmp/vllm_serving_trace.json_scheduler.jsonl \
        --model /share_data/public_models/Qwen3-0.6B
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.kernel.validation.dataset_io import get_emulation_output_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict per-request serving metrics from emulation trace"
    )
    _emu_dir = get_emulation_output_dir()
    parser.add_argument("--trace",
                        default=os.path.join(_emu_dir, "serving_trace.json"),
                        help="Path to merged trace JSON file")
    parser.add_argument("--scheduler", default=None,
                        help="Path to scheduler sidecar JSONL (default: {trace}_scheduler.jsonl)")
    parser.add_argument("--model", default="/share_data/public_models/Qwen3-0.6B",
                        help="Model path for config extraction")
    parser.add_argument("--hw-config", default="configs/hardware/nvidia_a800.yaml",
                        help="Hardware config YAML")
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallel size")
    parser.add_argument("--gap-threshold-us", type=float, default=1000.0,
                        help="Gap threshold for iteration segmentation")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON instead of table")
    parser.add_argument("--output", default=None,
                        help="Write JSON output to this path when --json is set")
    args = parser.parse_args(argv)

    # Derive scheduler path if not given
    scheduler_path = args.scheduler
    if scheduler_path is None:
        scheduler_path = args.trace + "_scheduler.jsonl"
        if not Path(scheduler_path).exists():
            # Try provenance to find it
            prov_path = args.trace + "_provenance.json"
            if Path(prov_path).exists():
                with open(prov_path) as f:
                    prov = json.load(f)
                if "scheduler_sidecar" in prov:
                    scheduler_path = prov["scheduler_sidecar"]

    if not Path(scheduler_path).exists():
        print(f"[ERROR] Scheduler sidecar not found: {scheduler_path}", file=sys.stderr)
        print("Run run_vllm_serving_emulation.py to generate it.", file=sys.stderr)
        sys.exit(1)

    # Load inference window from provenance if available
    inference_window_us = None
    prov_path = args.trace + "_provenance.json"
    if Path(prov_path).exists():
        with open(prov_path) as f:
            prov = json.load(f)
        if "inference_window_us" in prov:
            inference_window_us = tuple(prov["inference_window_us"])

    from src.serve_predict.serve_metric_predictor import ServingPredictor
    from src.serve_predict.model_config import load_model_config

    model_config = load_model_config(args.model)
    sp = ServingPredictor(
        hw_config_path=args.hw_config,
        model_config=model_config,
        tp_size=args.tp_size,
        gap_threshold_us=args.gap_threshold_us,
    )

    metrics, _ = sp.predict_per_request_metrics(
        trace_path=args.trace,
        scheduler_path=scheduler_path,
        inference_window_us=inference_window_us,
    )

    if args.json:
        output = []
        for m in metrics:
            output.append({
                "request_id": m.request_id,
                "ttft_ms": round(m.ttft_ms, 3),
                "tpot_ms": round(m.tpot_ms, 3),
                "e2e_ms": round(m.e2e_ms, 3),
                "start_time_ms": round(m.start_time_ms, 3),
                "end_time_ms": round(m.end_time_ms, 3),
                "num_prefill_tokens": m.num_prefill_tokens,
                "num_decode_tokens": m.num_decode_tokens,
                "prefill_iterations": m.prefill_iterations,
                "decode_iterations": m.decode_iterations,
            })
        serialized = json.dumps(output, indent=2)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(serialized + "\n")
        else:
            print(serialized)
    else:
        print(f"Per-request serving metrics ({len(metrics)} requests)")
        print(f"{'='*80}")
        print(f"{'Req ID':>20s} {'TTFT(ms)':>10s} {'TPOT(ms)':>10s} "
              f"{'E2E(ms)':>10s} {'Prefill':>8s} {'Decode':>7s}")
        print(f"{'-'*80}")
        for m in metrics:
            req_short = m.request_id[-12:] if len(m.request_id) > 12 else m.request_id
            print(f"{req_short:>20s} {m.ttft_ms:>10.2f} {m.tpot_ms:>10.2f} "
                  f"{m.e2e_ms:>10.2f} {m.num_prefill_tokens:>8d} "
                  f"{m.num_decode_tokens:>7d}")

        # Summary statistics
        if metrics:
            avg_ttft = sum(m.ttft_ms for m in metrics) / len(metrics)
            avg_tpot = sum(m.tpot_ms for m in metrics) / len(metrics)
            avg_e2e = sum(m.e2e_ms for m in metrics) / len(metrics)
            print(f"{'-'*80}")
            print(f"{'AVERAGE':>20s} {avg_ttft:>10.2f} {avg_tpot:>10.2f} {avg_e2e:>10.2f}")


if __name__ == "__main__":
    main()
