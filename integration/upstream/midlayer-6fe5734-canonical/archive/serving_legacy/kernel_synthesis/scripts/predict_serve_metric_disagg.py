#!/usr/bin/env python3
"""
Predict per-request disaggregated serving metrics (TTFT_disagg, TPOT_disagg, e2e).

Combines prefill instance trace + decode instance trace + KV transfer latency.

Usage:
    python scripts/serve_predict/predict_serve_metric_disagg.py \
        --prefill-trace /tmp/vllm_disagg_trace_prefill \
        --decode-trace /tmp/vllm_disagg_trace_decode \
        --model /share_data/public_models/Qwen3-0.6B

    # With separate HW configs:
    python scripts/serve_predict/predict_serve_metric_disagg.py \
        --prefill-trace /tmp/trace_prefill \
        --decode-trace /tmp/trace_decode \
        --prefill-hw-config configs/hardware/nvidia_a800.yaml \
        --decode-hw-config configs/hardware/nvidia_h100.yaml \
        --model /share_data/public_models/Qwen3-0.6B
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Predict per-request disaggregated serving metrics"
    )
    parser.add_argument("--prefill-trace", required=True)
    parser.add_argument("--decode-trace", required=True)
    parser.add_argument("--prefill-sidecar", default=None,
                        help="Scheduler JSONL for prefill (default: {trace}_scheduler.jsonl)")
    parser.add_argument("--decode-sidecar", default=None,
                        help="Scheduler JSONL for decode (default: {trace}_scheduler.jsonl)")
    parser.add_argument("--model", default="/share_data/public_models/Qwen3-0.6B")
    parser.add_argument("--prefill-hw-config", default="configs/hardware/nvidia_a800.yaml")
    parser.add_argument("--decode-hw-config", default="configs/hardware/nvidia_a800.yaml")
    parser.add_argument("--tp-size", type=int, default=None,
                        help="Tensor parallel size for both instances")
    parser.add_argument("--prefill-tp-size", type=int, default=None,
                        help="Tensor parallel size for prefill instance")
    parser.add_argument("--decode-tp-size", type=int, default=None,
                        help="Tensor parallel size for decode instance")
    parser.add_argument("--kv-bw-gb-s", type=float, default=50.0,
                        help="Unidirectional KV transfer bandwidth in GB/s")
    parser.add_argument("--nccl-alpha-ns", type=float, default=1000.0,
                        help="NCCL protocol overhead in nanoseconds")
    parser.add_argument("--gap-threshold-us", type=float, default=2000.0,
                        help="Gap threshold for iteration segmentation (default 2000 for disagg with NCCL gaps)")
    parser.add_argument("--workload-artifact", default=None,
                        help="Path to canonical workload artifact used for request alignment")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Fixed output-token control recorded with workload artifact")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", default=None,
                        help="Write JSON output to this path; requires --json")
    args = parser.parse_args()

    if args.output and not args.json:
        parser.error("--output requires --json")

    if (args.tp_size is not None
            and ((args.prefill_tp_size is not None and args.prefill_tp_size != args.tp_size)
                 or (args.decode_tp_size is not None and args.decode_tp_size != args.tp_size))):
        parser.error(
            "--tp-size conflicts with explicit --prefill-tp-size or --decode-tp-size")

    base_tp_size = args.tp_size if args.tp_size is not None else 1
    prefill_tp_size = (
        args.prefill_tp_size if args.prefill_tp_size is not None else base_tp_size)
    decode_tp_size = (
        args.decode_tp_size if args.decode_tp_size is not None else base_tp_size)

    # Derive sidecar paths and inference windows from provenance
    prefill_sidecar = args.prefill_sidecar
    decode_sidecar = args.decode_sidecar
    prefill_window = None
    decode_window = None

    prefill_prov_path = f"{args.prefill_trace}_provenance.json"
    if Path(prefill_prov_path).exists():
        with open(prefill_prov_path) as f:
            prov = json.load(f)
        if prefill_sidecar is None and "scheduler_sidecar" in prov:
            prefill_sidecar = prov["scheduler_sidecar"]
        if "inference_window_us" in prov:
            prefill_window = tuple(prov["inference_window_us"])

    decode_prov_path = f"{args.decode_trace}_provenance.json"
    if Path(decode_prov_path).exists():
        with open(decode_prov_path) as f:
            prov = json.load(f)
        if decode_sidecar is None and "scheduler_sidecar" in prov:
            decode_sidecar = prov["scheduler_sidecar"]
        if "inference_window_us" in prov:
            decode_window = tuple(prov["inference_window_us"])

    if prefill_sidecar is None:
        prefill_sidecar = f"{args.prefill_trace}_scheduler.jsonl"
    if decode_sidecar is None:
        decode_sidecar = f"{args.decode_trace}_scheduler.jsonl"

    for path, label in [(args.prefill_trace, "prefill trace"),
                        (args.decode_trace, "decode trace"),
                        (prefill_sidecar, "prefill sidecar"),
                        (decode_sidecar, "decode sidecar")]:
        if not Path(path).exists():
            print(f"[ERROR] {label} not found: {path}", file=sys.stderr)
            sys.exit(1)

    from src.serve_predict.model_config import load_model_config
    from src.serve_predict.serve_metric_predictor import predict_disaggregated_metrics
    from src.kernel.validation.legacy_ops.comm import predict_kv_send_ops_ns as predict_kv_send_ops_time
    from src.serve_trace.kv_transfer import (
        parse_kv_transfers,
        predict_kv_transfer,
    )

    model_config = load_model_config(args.model)
    kv_bw_bytes_s = args.kv_bw_gb_s * 1e9

    # Parse and validate observed KV bytes from prefill trace
    import json as _json
    with open(args.prefill_trace) as f:
        _prefill_trace_data = _json.load(f)
    _prefill_kv_trace_data = _filter_trace_by_window(_prefill_trace_data, prefill_window)
    _kv_ops = parse_kv_transfers(_prefill_kv_trace_data)
    _kv_sends = [op for op in _kv_ops if op.op_type == "send"]
    observed_kv_bytes = sum(op.bytes for op in _kv_sends) if _kv_sends else None
    observed_kv_transfer_time_ns = (
        predict_kv_send_ops_time(_kv_sends, args.nccl_alpha_ns, kv_bw_bytes_s)
        if _kv_sends else None
    )

    # KV validation is computed after metrics (needs sidecar-derived request info)

    metrics = predict_disaggregated_metrics(
        prefill_trace_path=args.prefill_trace,
        prefill_sidecar_path=prefill_sidecar,
        prefill_hw_config=args.prefill_hw_config,
        decode_trace_path=args.decode_trace,
        decode_sidecar_path=decode_sidecar,
        decode_hw_config=args.decode_hw_config,
        model_config=model_config,
        kv_transfer_bw_bytes_s=kv_bw_bytes_s,
        nccl_alpha_ns=args.nccl_alpha_ns,
        tp_size=base_tp_size,
        prefill_tp_size=prefill_tp_size,
        decode_tp_size=decode_tp_size,
        gap_threshold_us=args.gap_threshold_us,
        prefill_inference_window_us=prefill_window,
        decode_inference_window_us=decode_window,
        observed_kv_bytes=observed_kv_bytes,
        observed_kv_transfer_time_ns=observed_kv_transfer_time_ns,
    )

    # KV validation using sidecar-derived per-request prompt lengths
    kv_prediction = None
    if _kv_sends and metrics:
        prompt_lens = [m.num_prefill_tokens for m in metrics]
        kv_prediction = predict_kv_transfer(
            _prefill_kv_trace_data, model_config.num_hidden_layers,
            model_config.num_key_value_heads, model_config.head_dim,
            prompt_lens[0], model_config.dtype_bytes, prefill_tp_size,
            args.nccl_alpha_ns, kv_bw_bytes_s,
            num_requests=len(metrics), prompt_lens=prompt_lens)

    if args.json:
        metrics_list = []
        for m in metrics:
            metrics_list.append({
                "request_id": m.request_id,
                "ttft_disagg_ms": round(m.ttft_disagg_ms, 3),
                "tpot_disagg_ms": round(m.tpot_disagg_ms, 3),
                "e2e_disagg_ms": round(m.e2e_disagg_ms, 3),
                "prefill_time_ms": round(m.prefill_time_ms, 3),
                "kv_transfer_time_ms": round(m.kv_transfer_time_ms, 3),
                "decode_time_ms": round(m.decode_time_ms, 3),
                "num_prefill_tokens": m.num_prefill_tokens,
                "num_decode_tokens": m.num_decode_tokens,
            })
        output = {
            "inputs": {
                "prefill_trace": args.prefill_trace,
                "decode_trace": args.decode_trace,
                "prefill_sidecar": prefill_sidecar,
                "decode_sidecar": decode_sidecar,
                "prefill_hw_config": args.prefill_hw_config,
                "decode_hw_config": args.decode_hw_config,
                "tp_size": base_tp_size,
                "prefill_tp_size": prefill_tp_size,
                "decode_tp_size": decode_tp_size,
                "kv_bw_gb_s": args.kv_bw_gb_s,
                "nccl_alpha_ns": args.nccl_alpha_ns,
                "gap_threshold_us": args.gap_threshold_us,
                "prefill_inference_window_us": list(prefill_window) if prefill_window else None,
                "decode_inference_window_us": list(decode_window) if decode_window else None,
                "workload_artifact": args.workload_artifact,
                "max_tokens": args.max_tokens,
            },
            "kv_transfer": {
                "observed_kv_bytes": observed_kv_bytes,
                "observed_kv_transfer_time_ms": round(observed_kv_transfer_time_ns / 1e6, 6) if observed_kv_transfer_time_ns is not None else None,
                "kv_byte_source": "nccl_send_trace_ops",
                "kv_send_op_count": len(_kv_sends) if _kv_sends else 0,
                "prefill_tp_size": prefill_tp_size,
                "decode_tp_size": decode_tp_size,
                "expected_kv_bytes": kv_prediction.expected_bytes if kv_prediction else None,
                "observed_over_expected_ratio": round(kv_prediction.observed_over_expected_ratio, 3) if kv_prediction and kv_prediction.observed_over_expected_ratio else None,
                "byte_mismatch_ratio": kv_prediction.byte_mismatch_ratio if kv_prediction else None,
                "max_overhead_ratio": kv_prediction.max_overhead_ratio if kv_prediction else None,
            },
            "metrics": metrics_list,
        }
        output_text = json.dumps(output, indent=2)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text)
        print(output_text)
    else:
        print(f"Disaggregated per-request metrics ({len(metrics)} requests)")
        print("=" * 90)
        print(f"{'Req ID':>14s} {'TTFT_d(ms)':>11s} {'TPOT_d(ms)':>11s} "
              f"{'E2E_d(ms)':>10s} {'Prefill':>8s} {'KV_xfer':>8s} {'Decode':>8s}")
        print("-" * 90)
        for m in metrics:
            req_short = m.request_id[-12:] if len(m.request_id) > 12 else m.request_id
            print(f"{req_short:>14s} {m.ttft_disagg_ms:>11.2f} {m.tpot_disagg_ms:>11.2f} "
                  f"{m.e2e_disagg_ms:>10.2f} {m.prefill_time_ms:>8.2f} "
                  f"{m.kv_transfer_time_ms:>8.2f} {m.decode_time_ms:>8.2f}")

        if metrics:
            avg_ttft = sum(m.ttft_disagg_ms for m in metrics) / len(metrics)
            avg_tpot = sum(m.tpot_disagg_ms for m in metrics) / len(metrics)
            avg_e2e = sum(m.e2e_disagg_ms for m in metrics) / len(metrics)
            print("-" * 90)
            print(f"{'AVERAGE':>14s} {avg_ttft:>11.2f} {avg_tpot:>11.2f} {avg_e2e:>10.2f}")


def _filter_trace_by_window(trace_entries, window):
    if window is None:
        return trace_entries
    start_us, end_us = window
    return [
        entry for entry in trace_entries
        if isinstance(entry.get("timestamp_us"), (int, float))
        and start_us <= entry["timestamp_us"] <= end_us
    ]


if __name__ == "__main__":
    main()
