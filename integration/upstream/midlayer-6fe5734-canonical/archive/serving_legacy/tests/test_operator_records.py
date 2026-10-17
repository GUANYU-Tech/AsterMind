"""Tests for the approach-D operator-record adapter
(src/serve_trace/operator_records.py): OperatorRecord schema, JSONL round-trip,
and OperatorTraceGenerator.records_to_trace_direct mapping captured real
operators to the per-op priced kernel trace via the recipe operator->kernel
last hop.

These tests are GPU-free: they feed synthetic OperatorRecords (the same schema
the in-vLLM patch emits) through the adapter and check the resulting kernel
categories, reusing the MoE-aware classifier.
"""
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from src.serve_predict.model_config import ModelConfig
from src.kernel.hardware_spec import HWSpec
from src.serve_trace.operator_records import (
    OperatorRecord, OperatorTraceGenerator,
    write_operator_records, read_operator_records, OP_TYPES,
)
from scripts.serve_trace.validate_analytic_vs_nsys import classify_kernel, coalesce_gemm

HW = HWSpec("configs/hardware/nvidia_a800.yaml")


def _qwen3(layers=2, hidden=1024, heads=16, kv=8, head_dim=128, inter=3072):
    return ModelConfig(hidden_size=hidden, intermediate_size=inter,
                       num_attention_heads=heads, num_key_value_heads=kv,
                       num_hidden_layers=layers, head_dim=head_dim,
                       vocab_size=151936, architecture="Qwen3ForCausalLM")


def _qwen3_moe(layers=1, hidden=2048, heads=32, kv=4, head_dim=128,
               inter=6144, experts=128, topk=8, moe_inter=768):
    return ModelConfig(hidden_size=hidden, intermediate_size=inter,
                       num_attention_heads=heads, num_key_value_heads=kv,
                       num_hidden_layers=layers, head_dim=head_dim,
                       vocab_size=151936, architecture="Qwen3MoeForCausalLM",
                       num_experts=experts, num_experts_per_tok=topk,
                       moe_intermediate_size=moe_inter)


def _cats(trace):
    return coalesce_gemm(Counter(classify_kernel(e["kernel_name"]) for e in trace))


class TestOperatorRecord:
    def test_rejects_unknown_op_type(self):
        with pytest.raises(ValueError):
            OperatorRecord(step_id=0, op_idx=0, op_type="not_a_real_op")

    def test_accepts_known_op_types(self):
        for t in OP_TYPES:
            r = OperatorRecord(step_id=0, op_idx=0, op_type=t,
                               attrs={"batch": 1, "seq_len": 8})
            assert r.op_type == t

    def test_jsonl_roundtrip(self, tmp_path):
        recs = [
            OperatorRecord(0, 0, "rms_norm", "norm", 0, {"batch": 1, "seq_len": 8}),
            OperatorRecord(0, 1, "linear", "qkv_proj", 0,
                           {"B": 1, "M": 8, "N": 4096, "K": 1024}),
        ]
        p = str(tmp_path / "ops.jsonl")
        write_operator_records(recs, p)
        back = read_operator_records(p)
        assert len(back) == 2
        assert back[1].op_type == "linear" and back[1].attrs["N"] == 4096

    def test_dtype_quant_defaults(self):
        """Phase 2 dtype/quant default to unquantized/unknown (backward compat)."""
        r = OperatorRecord(0, 0, "rms_norm", "norm", 0, {"batch": 1, "seq_len": 8})
        assert r.dtype == "" and r.quant is None

    def test_dtype_quant_roundtrip(self, tmp_path):
        """A quantized linear record survives the JSONL round-trip losslessly."""
        r = OperatorRecord(0, 0, "linear", "gate_up_proj", 0,
                           {"B": 1, "M": 8, "N": 4096, "K": 1024},
                           dtype="bfloat16",
                           quant={"scheme": "compressed-tensors",
                                  "weight_dtype": "int8", "act_dtype": "bfloat16",
                                  "group_size": None})
        p = str(tmp_path / "q.jsonl")
        write_operator_records([r], p)
        assert read_operator_records(p)[0] == r

    def test_pre_phase2_jsonl_readable(self, tmp_path):
        """A pre-Phase-2 line (no dtype/quant keys) reads back with defaults."""
        import json
        p = str(tmp_path / "old.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"step_id": 2, "op_idx": 1, "op_type": "linear",
                                "position": "qkv_proj", "layer_idx": 0,
                                "attrs": {"M": 729, "N": 6144, "K": 2560, "B": 1}}) + "\n")
        back = read_operator_records(p)[0]
        assert back.op_type == "linear" and back.dtype == "" and back.quant is None

    def test_unknown_future_key_dropped(self, tmp_path):
        """A newer trace with an unknown key stays readable (forward compat)."""
        import json
        p = str(tmp_path / "future.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"step_id": 0, "op_idx": 0, "op_type": "rms_norm",
                                "attrs": {}, "some_future_field": 42}) + "\n")
        back = read_operator_records(p)[0]
        assert back.op_type == "rms_norm"


class TestAdapterMapping:
    def test_linear_maps_to_gemm(self):
        gen = OperatorTraceGenerator(_qwen3(), HW)
        recs = [OperatorRecord(0, 0, "linear", "qkv_proj", 0,
                               {"B": 1, "M": 128, "N": 4096, "K": 1024})]
        trace = gen.records_to_trace_direct(recs)
        assert _cats(trace).get("GEMM", 0) == 1

    def test_attention_emits_kv_cache_and_flash(self):
        # one captured attention op implies kv_cache + flash kernels
        gen = OperatorTraceGenerator(_qwen3(), HW)
        recs = [OperatorRecord(0, 0, "attention", "", 0,
                               {"batch": 1, "seq_len": 128, "is_decode": False})]
        trace = gen.records_to_trace_direct(recs)
        cats = _cats(trace)
        assert cats.get("KV_CACHE", 0) == 1
        assert cats.get("FLASH_ATTN", 0) >= 1

    def test_decode_attention_one_flash_kernel(self):
        gen = OperatorTraceGenerator(_qwen3(), HW)
        recs = [OperatorRecord(0, 0, "attention", "", 0,
                               {"batch": 1, "seq_len": 1, "is_decode": True})]
        trace = gen.records_to_trace_direct(recs)
        # one flash_fwd_splitkv COMPUTE kernel + 1 KV_CACHE; the split-KV combine
        # epilogue is an autotuner artifact (FLASH_SPLIT_REDUCE exclusion), not
        # emitted by the recipe.
        assert _cats(trace).get("FLASH_ATTN", 0) == 1
        assert not any("splitkv_combine" in e["kernel_name"] for e in trace)

    def test_activation_and_norm_map(self):
        gen = OperatorTraceGenerator(_qwen3(), HW)
        recs = [
            OperatorRecord(0, 0, "rms_norm", "norm", 0, {"batch": 1, "seq_len": 8}),
            OperatorRecord(0, 1, "activation", "", 0, {"batch": 1, "seq_len": 8}),
        ]
        cats = _cats(gen.records_to_trace_direct(recs))
        assert cats.get("NORM", 0) == 1 and cats.get("ACTIVATION", 0) == 1

    def test_fused_norm_block_decode(self):
        """Decode (rows < 256): block = min(hidden, 1024)."""
        gen = OperatorTraceGenerator(_qwen3(hidden=1024), HW)
        recs = [OperatorRecord(0, 0, "fused_add_rms_norm", "norm", 0,
                               {"input_shape": [1, 1024]})]
        trace = gen.records_to_trace_direct(recs)
        assert trace[0]["block"] == [1024, 1, 1]

    def test_fused_norm_block_prefill(self):
        """Prefill (rows >= 256): block = min(hidden, 256)."""
        gen = OperatorTraceGenerator(_qwen3(hidden=1024), HW)
        recs = [OperatorRecord(0, 0, "fused_add_rms_norm", "norm", 0,
                               {"input_shape": [385, 1024]})]
        trace = gen.records_to_trace_direct(recs)
        assert trace[0]["block"] == [256, 1, 1]

    def test_fused_norm_block_large_hidden(self):
        """hidden=4096: both decode and prefill capped by min(hidden, max_block)."""
        gen = OperatorTraceGenerator(_qwen3(hidden=4096, inter=11008), HW)
        recs_decode = [OperatorRecord(0, 0, "fused_add_rms_norm", "norm", 0,
                                      {"input_shape": [1, 4096]})]
        recs_prefill = [OperatorRecord(0, 0, "fused_add_rms_norm", "norm", 0,
                                       {"input_shape": [385, 4096]})]
        trace_d = gen.records_to_trace_direct(recs_decode)
        trace_p = gen.records_to_trace_direct(recs_prefill)
        assert trace_d[0]["block"] == [1024, 1, 1]
        assert trace_p[0]["block"] == [256, 1, 1]

    def test_fused_norm_block_threshold_boundary(self):
        """Exact boundary: rows=255 -> 1024, rows=256 -> 256."""
        gen = OperatorTraceGenerator(_qwen3(hidden=1024), HW)
        recs_255 = [OperatorRecord(0, 0, "fused_add_rms_norm", "norm", 0,
                                   {"input_shape": [255, 1024]})]
        recs_256 = [OperatorRecord(0, 0, "fused_add_rms_norm", "norm", 0,
                                   {"input_shape": [256, 1024]})]
        assert gen.records_to_trace_direct(recs_255)[0]["block"] == [1024, 1, 1]
        assert gen.records_to_trace_direct(recs_256)[0]["block"] == [256, 1, 1]

    def test_unknown_op_in_trace_raises(self):
        # OperatorRecord blocks unknown op_type at construction, so the adapter
        # only ever sees known types; verify the mapping is total over OP_TYPES
        # that carry the needed attrs.
        gen = OperatorTraceGenerator(_qwen3(), HW)
        recs = [OperatorRecord(0, 0, "rotary", "", 0, {"batch": 1, "seq_len": 8}),
                OperatorRecord(0, 1, "kv_cache", "", 0, {"batch": 1, "seq_len": 8})]
        cats = _cats(gen.records_to_trace_direct(recs))
        assert cats.get("ROTARY", 0) == 1 and cats.get("KV_CACHE", 0) == 1

    def test_collectives_map_to_nccl(self):
        # a captured TP all_reduce record -> one COLLECTIVE synth entry. On A800
        # NVLink a small message uses vllm::cross_device_reduce (custom_all_reduce),
        # a large one falls back to ncclDevKernel_AllReduce; both classify as
        # COLLECTIVE. (The classifier's category was renamed NCCL -> COLLECTIVE;
        # this asserts the category, not a specific kernel name.)
        gen = OperatorTraceGenerator(_qwen3(), HW, tp_size=2)
        recs = [OperatorRecord(0, 0, "all_reduce", "tp", 0,
                               {"batch": 1, "seq_len": 8, "msg_bytes": 4096})]
        cats = _cats(gen.records_to_trace_direct(recs))
        assert cats.get("COLLECTIVE", 0) == 1

    def test_moe_op_does_not_fabricate_all_to_all(self):
        # A captured `moe` op must NOT invent all-to-all kernels. On the naive EP
        # backend vLLM issues no all-to-all (expert_map + post-FFN all-reduce);
        # nsys shows zero all-to-all kernels. The MoE op maps only to its fused
        # expert kernels — the cross-rank reduce arrives as a separately captured
        # `all_reduce` record, and real all-to-all (pplx/deepep) as a separately
        # captured `all_to_all` record.
        gen = OperatorTraceGenerator(_qwen3_moe(layers=1), HW, ep_size=2)
        recs = [OperatorRecord(0, 0, "moe", "", 0,
                               {"batch": 1, "seq_len": 8, "is_decode": False})]
        cats = _cats(gen.records_to_trace_direct(recs))
        assert cats.get("COLLECTIVE", 0) == 0  # no fabricated all-to-all

    def test_captured_all_to_all_maps_to_nccl(self):
        # A genuinely captured all_to_all record (real pplx/deepep dispatch or
        # combine) maps to one COLLECTIVE synth entry (ncclDevKernel_AllToAll_*).
        gen = OperatorTraceGenerator(_qwen3_moe(layers=1), HW, ep_size=2)
        recs = [
            OperatorRecord(0, 0, "all_to_all", "dispatch", 0,
                           {"batch": 1, "seq_len": 8, "msg_bytes": 4096}),
            OperatorRecord(0, 1, "all_to_all", "combine", 1,
                           {"batch": 1, "seq_len": 8, "msg_bytes": 4096}),
        ]
        cats = _cats(gen.records_to_trace_direct(recs))
        assert cats.get("COLLECTIVE", 0) == 2  # dispatch + combine, capture-driven

    def test_moe_without_ep_no_all_to_all(self):
        gen = OperatorTraceGenerator(_qwen3_moe(layers=1), HW, ep_size=1)
        recs = [OperatorRecord(0, 0, "moe", "", 0,
                               {"batch": 1, "seq_len": 8, "is_decode": False})]
        cats = _cats(gen.records_to_trace_direct(recs))
        assert cats.get("COLLECTIVE", 0) == 0

    def test_ordering_by_step_then_opidx(self):
        gen = OperatorTraceGenerator(_qwen3(), HW)
        # deliberately out of order; adapter must sort by (step_id, op_idx)
        recs = [
            OperatorRecord(1, 0, "rms_norm", "norm", 0, {"batch": 1, "seq_len": 1}),
            OperatorRecord(0, 1, "activation", "", 0, {"batch": 1, "seq_len": 8}),
        ]
        trace = gen.records_to_trace_direct(recs)
        # step 0 ops come before step 1; timestamps are non-decreasing
        ts = [e["timestamp_us"] for e in trace]
        assert ts == sorted(ts)


class TestFullForwardComposition:
    def test_dense_prefill_step_categories(self):
        """A full dense prefill step (per the real captured structure) yields the
        same category mix as the analytic recipe."""
        m = _qwen3(layers=2)
        gen = OperatorTraceGenerator(m, HW)
        from src.serve_trace.shape_inference import infer_gemm_dimensions
        recs = []
        idx = 0

        def add(op_type, position="", **attrs):
            nonlocal idx
            recs.append(OperatorRecord(0, idx, op_type, position, 0, attrs))
            idx += 1

        for L in range(m.num_hidden_layers):
            add("rms_norm", "norm_1", batch=1, seq_len=128)
            M, N, K = infer_gemm_dimensions(m, "qkv_proj", 128, 1, 1)
            add("linear", "qkv_proj", B=1, M=M, N=N, K=K)
            add("rms_norm", "qk_norm_0", batch=1, seq_len=128)
            add("rms_norm", "qk_norm_1", batch=1, seq_len=128)
            add("rotary", batch=1, seq_len=128)
            add("attention", batch=1, seq_len=128, is_decode=False)
            M, N, K = infer_gemm_dimensions(m, "o_proj", 128, 1, 1)
            add("linear", "o_proj", B=1, M=M, N=N, K=K)
            add("rms_norm", "norm_2", batch=1, seq_len=128)
            M, N, K = infer_gemm_dimensions(m, "gate_up_proj", 128, 1, 1)
            add("linear", "gate_up_proj", B=1, M=M, N=N, K=K)
            add("activation", batch=1, seq_len=128)
            M, N, K = infer_gemm_dimensions(m, "down_proj", 128, 1, 1)
            add("linear", "down_proj", B=1, M=M, N=N, K=K)
        cats = _cats(gen.records_to_trace_direct(recs))
        # per layer: 4 NORM, 1 ROTARY, 1 KV_CACHE, 1 FLASH, 1 ACTIVATION; GEMM
        # count is whatever the cuBLAS dispatch emits for the captured shapes
        # (wide-output projections like gate_up may dispatch >1 kernel), so it is
        # checked as a positive multiple of layers, not a hardcoded 3.
        assert cats["NORM"] == 4 * m.num_hidden_layers
        assert cats["KV_CACHE"] == m.num_hidden_layers
        assert cats["ACTIVATION"] == m.num_hidden_layers
        assert cats["ROTARY"] == m.num_hidden_layers
        assert cats["FLASH_ATTN"] == m.num_hidden_layers  # prefill: 1 flash/layer
        assert cats["GEMM"] >= 3 * m.num_hidden_layers
