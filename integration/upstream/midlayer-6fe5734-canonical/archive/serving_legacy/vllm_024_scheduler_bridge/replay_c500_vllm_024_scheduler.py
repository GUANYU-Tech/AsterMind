#!/usr/bin/env python3
"""Replay a request prefix through C500's real vLLM 0.24 Scheduler bridge.

This is a diagnostic control-plane probe, not a Serving measurement: it
exports only Scheduler decisions and never transmits or records a device
duration.  Arrivals are admitted before the corresponding virtual arrival
frontier; completions are deliberately FIFO and immediate so the requested
prefix isolates vLLM 0.24 request/KV/async semantics from Forward timing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
BRIDGE_PATH = REPOSITORY_ROOT / "src/serving/backend/metax_c500_vllm_024_scheduler.py"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", required=True, type=int)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", required=True, type=int)
    parser.add_argument("--pipeline-parallel-size", required=True, type=int)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--max-model-len", required=True, type=int)
    parser.add_argument("--max-num-batched-tokens", required=True, type=int)
    parser.add_argument("--max-num-seqs", required=True, type=int)
    parser.add_argument("--block-size", required=True, type=int)
    parser.add_argument("--num-kv-blocks", required=True, type=int)
    parser.add_argument("--dummy-token-id", default=42, type=int)
    return parser.parse_args()


def _require_positive(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_requests(path: Path, limit: int) -> list[dict[str, object]]:
    _require_positive(limit, "limit")
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        expected = (
            "Query Time Stamp [ms]",
            "In Token [tokens]",
            "Out Token [tokens]",
            "Cached hit [tokens]",
        )
        if reader.fieldnames != list(expected):
            raise ValueError("request CSV columns differ from canonical contract")
        rows = []
        for request_index, row in enumerate(reader):
            if request_index == limit:
                break
            if row is None or set(row) != set(expected):
                raise ValueError("request CSV row fields differ from canonical contract")
            if row["Cached hit [tokens]"] != "0":
                raise NotImplementedError("0.24 probe supports only no-prefix-cache rows")
            try:
                arrival_ms = float(row["Query Time Stamp [ms]"])
                input_tokens = int(row["In Token [tokens]"])
                output_tokens = int(row["Out Token [tokens]"])
            except (TypeError, ValueError) as error:
                raise ValueError("request CSV has invalid numeric values") from error
            if not arrival_ms >= 0.0 or input_tokens < 1 or output_tokens < 0:
                raise ValueError("request CSV values are out of supported range")
            rows.append({
                "request_id": f"request-{request_index}",
                "arrival_time_s": arrival_ms / 1000.0,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            })
    if not rows:
        raise ValueError("request CSV prefix is empty")
    return rows


class _BridgeClient:
    def __init__(self, init: dict[str, object]) -> None:
        # The Scheduler process is inside the C500 serving container.  Copy
        # through the remote host explicitly; ``docker cp`` itself cannot
        # resolve a local checkout path on muxi.
        subprocess.run([
            "scp", str(BRIDGE_PATH),
            "muxi:/tmp/midlayer_c500_vllm024_scheduler.py",
        ], check=True)
        copy_command = [
            "ssh", "muxi", "docker", "cp",
            "/tmp/midlayer_c500_vllm024_scheduler.py",
            "yxh_vllm_serving_c500:/tmp/midlayer_c500_vllm024_scheduler.py",
        ]
        subprocess.run(copy_command, check=True)
        self._process = subprocess.Popen(
            [
                "ssh", "muxi",
                "docker exec -i yxh_vllm_serving_c500 "
                "python /tmp/midlayer_c500_vllm024_scheduler.py "
                "2>/tmp/midlayer_c500_vllm024_scheduler_bridge.log",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # vLLM/plugin bootstrap emits several KiB of informational logs.
            # This probe consumes only the bridge's JSONL stdout; leaving
            # stderr on a PIPE without a concurrent reader can deadlock before
            # the ready record is emitted.
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("could not open C500 Scheduler bridge pipes")
        self._request(init)
        ready = self._read()
        if ready.get("kind") != "ready":
            raise RuntimeError(f"C500 Scheduler bridge did not become ready: {ready!r}")
        self.ready = ready

    def _request(self, record: dict[str, object]) -> None:
        if self._process.stdin is None:
            raise RuntimeError("C500 Scheduler bridge stdin is unavailable")
        self._process.stdin.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._process.stdin.flush()

    def _read(self) -> dict[str, object]:
        if self._process.stdout is None:
            raise RuntimeError("C500 Scheduler bridge stdout is unavailable")
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("C500 Scheduler bridge exited before response")
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise RuntimeError("C500 Scheduler bridge returned non-object JSON")
        return parsed

    def call(self, record: dict[str, object]) -> dict[str, object]:
        self._request(record)
        return self._read()

    def close(self) -> None:
        response = self.call({"command": "close"})
        if response.get("kind") != "closed":
            raise RuntimeError("C500 Scheduler bridge close failed")
        return_code = self._process.wait(timeout=10)
        if return_code != 0:
            raise RuntimeError(f"C500 Scheduler bridge exit={return_code}")


def main() -> None:
    args = _arguments()
    requests = _read_requests(args.requests, args.limit)
    init = {
        "command": "init",
        "model": args.model,
        "tensor_parallel_size": _require_positive(
            args.tensor_parallel_size, "tensor_parallel_size"),
        "pipeline_parallel_size": _require_positive(
            args.pipeline_parallel_size, "pipeline_parallel_size"),
        "enable_expert_parallel": args.enable_expert_parallel,
        "max_model_len": _require_positive(args.max_model_len, "max_model_len"),
        "max_num_batched_tokens": _require_positive(
            args.max_num_batched_tokens, "max_num_batched_tokens"),
        "max_num_seqs": _require_positive(args.max_num_seqs, "max_num_seqs"),
        "block_size": _require_positive(args.block_size, "block_size"),
        "num_kv_blocks": _require_positive(args.num_kv_blocks, "num_kv_blocks"),
        "dummy_token_id": args.dummy_token_id,
    }
    client = _BridgeClient(init)
    decisions: list[dict[str, object]] = []
    try:
        for request in requests:
            response = client.call({"command": "add", **request})
            if response.get("kind") != "ok":
                raise RuntimeError(f"C500 Scheduler admission failed: {response!r}")
        while True:
            response = client.call({"command": "plan"})
            kind = response.get("kind")
            decisions.append(response)
            if kind == "step":
                step_id = response.get("step_id")
                if not isinstance(step_id, str) or not step_id:
                    raise RuntimeError("C500 Scheduler step has no ID")
                completed = client.call({"command": "complete", "step_id": step_id})
                if completed.get("kind") != "ok":
                    raise RuntimeError("C500 Scheduler completion failed")
            elif kind == "wait":
                raise RuntimeError(
                    "immediate-completion probe reached async wait; use the "
                    "depth-aware Forward replay rather than this prefix probe")
            elif kind == "finished":
                continue
            elif kind == "idle":
                break
            else:
                raise RuntimeError(f"unknown C500 Scheduler response: {response!r}")
        has_work = client.call({"command": "has_work"})
        if has_work != {"kind": "has_work", "value": False}:
            raise RuntimeError("C500 Scheduler still has work after idle")
    finally:
        client.close()
    payload = {
        "schema_version": 1,
        "kind": "c500_vllm_0_24_scheduler_prefix_probe",
        "duration_observation_consumed": False,
        "request_count": len(requests),
        "runtime": client.ready["runtime"],
        "effective": client.ready["effective"],
        "decisions": decisions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
