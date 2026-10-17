"""Integration test for EP collective capture using a REAL captured operator
trace from a TP2/EP2 Qwen3-30B-A3B plugin run on 2x A800 (approach D).

The fixture qwen3moe_tp2ep2_ops_rank0.jsonl is the first two forward steps
(prefill + first decode) of the rank-0 operator capture. It is REAL data, not
synthesized.

Key invariant (mechanistically grounded + nsys-confirmed): the naive EP backend
that A800 uses does NOT issue all-to-all. vLLM places a local expert subset via
expert_map and reduces across ranks with the post-FFN all-reduce
(maybe_all_reduce_tensor_model_parallel). So a real TP2/EP2 capture contains REAL
`all_reduce` records and ZERO `all_to_all` records -- and nsys for the same run
shows zero all-to-all kernels. The adapter must therefore turn this into
all-reduce collective entries (vllm::cross_device_reduce on A800 NVLink) and
must NOT fabricate any all-to-all kernel.
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.serve_trace.operator_records import (
    OperatorTraceGenerator, read_operator_records)
from src.serve_predict.model_config import load_model_config
from src.kernel.hardware_spec import HWSpec

_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                        "qwen3moe_tp2ep2_ops_rank0.jsonl")
_MODEL = "/share_data/public_models/Qwen3-30B-A3B"
_HW = "configs/hardware/nvidia_a800.yaml"


def test_captured_tp2ep2_has_real_allreduce_no_all_to_all():
    """The captured operator trace itself (BEFORE the adapter runs) carries real
    all_reduce records and no fabricated all_to_all -- the contract's pre-adapter
    invariant, corrected to the evidence (naive EP issues all-reduce, not a2a)."""
    recs = read_operator_records(_FIXTURE)
    kinds = Counter(r.op_type for r in recs)
    assert kinds["all_reduce"] > 0, "real TP/EP all-reduce must be captured"
    assert kinds["all_to_all"] == 0, (
        "naive EP backend issues no all-to-all; none must be captured/fabricated")
    # Captured all_reduce carries REAL message bytes (not reconstructed).
    ar = [r for r in recs if r.op_type == "all_reduce"]
    assert all(r.attrs.get("msg_bytes", 0) > 0 for r in ar)


def test_adapter_emits_nccl_allreduce_not_all_to_all():
    """End to end: the real captured TP2/EP2 trace -> priced kernels contains
    one cross-device (all-reduce) collective per captured all_reduce record and
    zero all-to-all kernel names. On A800 NVLink the small per-layer messages use
    vLLM custom_all_reduce (vllm::cross_device_reduce), not ncclDevKernel_AllReduce
    — assert the collective-emit contract by op_type, not a specific kernel name."""
    if not os.path.exists(_MODEL):
        import pytest
        pytest.skip(f"model {_MODEL} not present")
    recs = read_operator_records(_FIXTURE)
    m = load_model_config(_MODEL)
    hw = HWSpec(_HW)
    gen = OperatorTraceGenerator(m, hw, tp_size=2, ep_size=2)
    trace = gen.records_to_trace_direct(recs)
    names = [e["kernel_name"] for e in trace]
    # one cross-device reduce (custom_all_reduce or NCCL ring fallback) per
    # captured all_reduce record; the naive EP backend fabricates no all-to-all.
    def _is_allreduce(n):
        return "cross_device_reduce" in n or "AllReduce" in n
    assert any(_is_allreduce(n) for n in names), "expected all-reduce collective kernels"
    assert not any("AllToAll" in n for n in names), (
        "must not fabricate all-to-all kernels for the naive EP backend")
    n_ar_records = sum(1 for r in recs if r.op_type == "all_reduce")
    n_allreduce = sum(1 for n in names if _is_allreduce(n))
    assert n_allreduce == n_ar_records
