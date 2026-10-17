"""
KV Transfer: parse and predict P2P KV transfer latency from trace.

In vLLM disaggregated deployment, the prefill instance sends KV cache to the
decode instance via P2pNcclConnector. These transfers appear in the trace as
nccl_send (prefill) and nccl_recv (decode) kernel entries.

Trace encoding (from nccl_emulate.c):
  - kernel_name: "nccl_send" or "nccl_recv"
  - grid[0] = nranks, grid[1] = rank, grid[2] = op_id
  - block[0] = bytes & 0xFFFFFFFF, block[1] = bytes >> 32

Prediction functions live in src/kernel/ops/comm; this module handles trace
parsing and orchestration.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from src.kernel.validation.legacy_ops.comm import (
    compute_expected_kv_bytes,
    predict_kv_transfer_ns,
    predict_kv_send_ops_ns,
)


@dataclass
class KVTransferOp:
    """A single KV transfer operation parsed from trace."""
    op_type: str       # "send" or "recv"
    bytes: int         # total message bytes
    nranks: int        # number of ranks in communicator
    rank: int          # rank of this process
    op_id: int         # operation sequence ID
    timestamp_us: int  # trace timestamp


@dataclass
class KVTransferPrediction:
    """Predicted KV transfer latency."""
    total_bytes: int
    transfer_time_ns: float
    expected_bytes: Optional[int] = None
    byte_mismatch_ratio: Optional[float] = None
    observed_over_expected_ratio: Optional[float] = None
    max_overhead_ratio: float = 4.0


def parse_kv_transfers(trace_entries: List[dict]) -> List[KVTransferOp]:
    """Extract nccl_send and nccl_recv operations from trace.

    Returns list of KVTransferOp, sorted by timestamp.
    """
    ops = []
    for entry in trace_entries:
        name = entry.get("kernel_name", "")
        if name == "nccl_send" or name == "nccl_recv":
            msg_bytes = entry["block"][0] + (entry["block"][1] << 32)
            ops.append(KVTransferOp(
                op_type=name.replace("nccl_", ""),
                bytes=msg_bytes,
                nranks=entry["grid"][0],
                rank=entry["grid"][1],
                op_id=entry["grid"][2],
                timestamp_us=entry.get("timestamp_us", 0),
            ))
    ops.sort(key=lambda x: x.timestamp_us)
    return ops


def predict_kv_transfer(trace_entries: List[dict], num_layers: int,
                        num_kv_heads: int, head_dim: int, prompt_len: int,
                        dtype_bytes: int, tp_size: int,
                        nccl_alpha_ns: float,
                        nvlink_bw_bytes_s: float,
                        num_requests: int = 1,
                        max_overhead_ratio: float = 4.0,
                        prompt_lens: Optional[List[int]] = None) -> KVTransferPrediction:
    """Full KV transfer prediction: parse trace, compute latency, validate bytes.

    Args:
        trace_entries: Prefill instance trace (merged)
        prompt_lens: If provided, compute exact expected bytes per request.
        num_requests: Number of requests (for uniform prompt_len validation).
        max_overhead_ratio: Max accepted observed/expected ratio.
    """
    ops = parse_kv_transfers(trace_entries)
    send_ops = [op for op in ops if op.op_type == "send"]

    if prompt_lens is not None:
        expected_total = sum(
            compute_expected_kv_bytes(num_layers, num_kv_heads, head_dim, pl, dtype_bytes, tp_size)
            for pl in prompt_lens)
    else:
        expected_per_req = compute_expected_kv_bytes(
            num_layers, num_kv_heads, head_dim, prompt_len, dtype_bytes, tp_size)
        expected_total = expected_per_req * num_requests

    if not send_ops:
        import sys
        print("[kv_transfer] WARNING: No nccl_send operations found in prefill trace. "
              "KV transfer may not have occurred.", file=sys.stderr)
        return KVTransferPrediction(
            total_bytes=0,
            transfer_time_ns=0.0,
            expected_bytes=expected_total,
            byte_mismatch_ratio=None,
        )

    total_send_bytes = sum(op.bytes for op in send_ops)

    mismatch_ratio = None
    if expected_total > 0:
        if total_send_bytes < expected_total:
            mismatch_ratio = (expected_total - total_send_bytes) / expected_total
        elif total_send_bytes > expected_total * max_overhead_ratio:
            mismatch_ratio = (total_send_bytes - expected_total) / expected_total
        else:
            mismatch_ratio = 0.0

    over_expected_ratio = total_send_bytes / expected_total if expected_total > 0 else None

    transfer_ns = predict_kv_send_ops_ns(
        send_ops, nccl_alpha_ns, nvlink_bw_bytes_s)

    return KVTransferPrediction(
        total_bytes=total_send_bytes,
        transfer_time_ns=transfer_ns,
        expected_bytes=expected_total,
        byte_mismatch_ratio=mismatch_ratio,
        observed_over_expected_ratio=over_expected_ratio,
        max_overhead_ratio=max_overhead_ratio,
    )
