"""Capture an independently reproducible native-vLLM Scheduler CPU profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from statistics import median
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from src.serving.backend.vllm import VLLMBackend, VLLMSchedulerConfig
from src.serving.backend.vllm_scheduler_process import (
    VLLM_SCHEDULER_SOURCE_COMMIT,
    VLLM_SCHEDULER_VERSION,
)
from src.serving.schedule.backend_protocol import BackendConfig
from src.serving.workload.model_config import load_model_config


def _quantile(samples: tuple[int, ...], numerator: int, denominator: int) -> int:
    if not samples:
        raise ValueError("Scheduler service samples cannot be empty")
    ordered = tuple(sorted(samples))
    index = (len(ordered) - 1) * numerator // denominator
    return ordered[index]


def _summary(samples: tuple[int, ...]) -> dict[str, int]:
    return {
        "count": len(samples),
        "min_ns": min(samples),
        "median_ns": int(median(samples)),
        "p95_ns": _quantile(samples, 95, 100),
        "max_ns": max(samples),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-id", required=True)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=512)
    parser.add_argument("--max-batched-tokens", type=int, default=512)
    parser.add_argument("--max-requests", type=int, default=16)
    parser.add_argument("--num-kv-blocks", type=int, default=4096)
    parser.add_argument("--kv-block-size-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=8)
    return parser


def main() -> None:
    args = _parser().parse_args()
    for name in (
            "sample_count", "max_batched_tokens", "max_requests",
            "num_kv_blocks", "kv_block_size_tokens", "max_model_len",
            "prompt_tokens", "output_tokens"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.prompt_tokens > args.max_batched_tokens:
        raise ValueError("--prompt-tokens cannot exceed --max-batched-tokens")
    model_path = args.model_config.resolve()
    raw_model_config = model_path.read_bytes()
    model = load_model_config(model_path)
    backend = VLLMBackend.from_model(
        BackendConfig(
            "scheduler-microbench",
            "colocated",
            args.max_batched_tokens,
            args.max_requests,
            args.num_kv_blocks,
            kv_block_size_tokens=args.kv_block_size_tokens,
            kv_bytes_per_token=model.kv_bytes_per_token_per_parallel_rank(1, 1),
            scheduler_pipeline_depth=1,
        ),
        VLLMSchedulerConfig(
            str(model_path),
            max_model_len=args.max_model_len,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
        ),
    )
    try:
        samples = backend.measure_scheduler_service(
            sample_count=args.sample_count,
            requests_per_batch=args.max_requests,
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
        )
        runtime = backend.runtime_info
    finally:
        backend.close()
    payload = {
        "schema_version": 1,
        "hardware_id": args.hardware_id,
        "host": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "scheduler": {
            "vllm_version": VLLM_SCHEDULER_VERSION,
            "source_commit": VLLM_SCHEDULER_SOURCE_COMMIT,
            "runtime_manifest_sha256": hashlib.sha256(
                json.dumps(runtime.__dict__, sort_keys=True, default=list).encode(
                    "utf-8")).hexdigest(),
        },
        "measurement": {
            "boundary": "in_scheduler_process_schedule_and_output_update_only",
            "sample_count": args.sample_count,
            "control_workload": {
                "max_batched_tokens": args.max_batched_tokens,
                "max_requests": args.max_requests,
                "num_kv_blocks": args.num_kv_blocks,
                "kv_block_size_tokens": args.kv_block_size_tokens,
                "max_model_len": args.max_model_len,
                "requests_per_batch": args.max_requests,
                "prompt_tokens": args.prompt_tokens,
                "output_tokens": args.output_tokens,
            },
        },
        "service_ns": {
            "schedule": _summary(samples["schedule_ns"]),
            "update_from_output": _summary(samples["update_ns"]),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
