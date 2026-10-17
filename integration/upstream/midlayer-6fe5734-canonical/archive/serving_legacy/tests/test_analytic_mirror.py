"""Tests for the analytic vLLM-scheduler mirror (src/serve_trace/analytic_mirror).

Validates the scheduling mirror, the cuBLAS GEMM dispatch port, and that the
generated trace matches the real on-disk schema and the expected per-category
kernel structure (the structure that is validated kernel-for-kernel against
nsys ground truth in scripts; here we lock the invariants as unit tests).
"""
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest

from src.serve_predict.model_config import ModelConfig
from src.kernel.hardware_spec import HWSpec
from src.serve_trace.analytic_mirror import (
    MirrorRequest, mirror_schedule, AnalyticTraceGenerator,
    write_analytic_trace, _cublas_gemm_dispatch, _num_qk_norm,
)

HW = HWSpec("configs/hardware/nvidia_a800.yaml")


def _qwen3(layers=2, hidden=1024, heads=16, kv=8, head_dim=128, inter=3072):
    return ModelConfig(hidden_size=hidden, intermediate_size=inter,
                       num_attention_heads=heads, num_key_value_heads=kv,
                       num_hidden_layers=layers, head_dim=head_dim,
                       vocab_size=151936, architecture="Qwen3ForCausalLM")


def _llama(layers=2, hidden=4096, heads=32, kv=8, head_dim=128, inter=14336):
    return ModelConfig(hidden_size=hidden, intermediate_size=inter,
                       num_attention_heads=heads, num_key_value_heads=kv,
                       num_hidden_layers=layers, head_dim=head_dim,
                       vocab_size=128256, architecture="LlamaForCausalLM")


def _qwen3_moe(layers=2, hidden=2048, heads=32, kv=4, head_dim=128,
               inter=6144, experts=128, topk=8, moe_inter=768):
    return ModelConfig(hidden_size=hidden, intermediate_size=inter,
                       num_attention_heads=heads, num_key_value_heads=kv,
                       num_hidden_layers=layers, head_dim=head_dim,
                       vocab_size=151936, architecture="Qwen3MoeForCausalLM",
                       num_experts=experts, num_experts_per_tok=topk,
                       moe_intermediate_size=moe_inter)


class TestMirrorSchedule:
    def test_prefill_completes_first_token(self):
        # output_len=8 => 1 prefill step (produces token 1) + 7 decode steps.
        steps = mirror_schedule([MirrorRequest("0", 128, 8)],
                                enable_chunked_prefill=False)
        assert len(steps) == 8
        assert steps[0].prefill_tokens == 128 and steps[0].decode_tokens == 0
        assert all(s.decode_tokens == 1 and s.prefill_tokens == 0
                   for s in steps[1:])
        assert steps[-1].finished == ["0"]

    def test_chunked_prefill_splits_long_prompt(self):
        steps = mirror_schedule([MirrorRequest("0", 1000, 1)],
                                max_num_batched_tokens=256,
                                enable_chunked_prefill=True)
        # 1000 prompt tokens / 256 budget => 4 prefill chunks (256,256,256,232);
        # the 4th chunk completes prefill and produces the only output token.
        prefill_steps = [s for s in steps if s.prefill_tokens > 0]
        assert sum(s.prefill_tokens for s in prefill_steps) == 1000
        assert all(s.prefill_tokens <= 256 for s in prefill_steps)

    def test_continuous_batching_mixes(self):
        # Two requests, small budget: decode of one co-batches with prefill of
        # another in the same step (continuous batching).
        steps = mirror_schedule(
            [MirrorRequest("0", 4, 4), MirrorRequest("1", 4, 4)],
            max_num_batched_tokens=4, enable_chunked_prefill=True)
        mixed = [s for s in steps if s.prefill_tokens > 0 and s.decode_tokens > 0]
        assert mixed, "expected at least one mixed prefill+decode step"

    def test_empty_requests(self):
        assert mirror_schedule([]) == []


class TestCublasDispatch:
    def test_decode_narrow_output_gemv(self):
        # out<=2048, 1 token -> internal_gemvx
        e = _cublas_gemm_dispatch(m=1024, n=1, k=2048)
        assert e[0][0] == "internal_gemvx_kernel"

    def test_decode_moderate_output_small_k_gemv2t(self):
        # 2048<out<=4096 and K<=2048 -> gemv2T (Qwen3-0.6B qkv: out=4096,K=1024)
        e = _cublas_gemm_dispatch(m=4096, n=1, k=1024)
        assert e[0][0] == "gemv2T_kernel_val"

    def test_decode_moderate_output_large_k_tc(self):
        # 2048<out<=4096 but K large -> TC (Qwen3-4B/Llama o_proj decode)
        e = _cublas_gemm_dispatch(m=4096, n=1, k=4096)
        assert "s16816gemm" in e[0][0]

    def test_decode_wide_output_tc(self):
        # out>4096 (gate_up, lm_head) -> TC even at 1 token
        e = _cublas_gemm_dispatch(m=6144, n=1, k=1024)
        assert "s16816gemm" in e[0][0]

    def test_prefill_tc(self):
        e = _cublas_gemm_dispatch(m=4096, n=128, k=1024)
        assert "s16816gemm" in e[0][0]

    def test_no_phantom_splitk(self):
        # split-K reduce is a cuBLAS autotuner artifact, not emitted analytically
        for e in [_cublas_gemm_dispatch(1024, 1, 3072),
                  _cublas_gemm_dispatch(2560, 1, 9728)]:
            assert all("splitKreduce" not in name for name, *_ in e)


class TestQKNorm:
    def test_qwen3_has_qk_norm(self):
        assert _num_qk_norm(_qwen3()) == 2

    def test_llama_no_qk_norm(self):
        assert _num_qk_norm(_llama()) == 0


class TestTraceGeneration:
    def test_trace_schema(self):
        gen = AnalyticTraceGenerator(_qwen3(layers=2), HW)
        trace, sched = gen.generate([MirrorRequest("0", 16, 2)])
        assert trace and sched
        for e in trace:
            assert set(e) >= {"timestamp_us", "kernel_name", "grid", "block",
                              "shared_mem", "stream"}
            assert len(e["grid"]) == 3 and len(e["block"]) == 3

    def test_qwen3_norm_count_includes_qk_norm(self):
        # Qwen3: 4 norms/layer (input, post-attn, q_norm, k_norm) + 1 final/step.
        m = _qwen3(layers=2)
        gen = AnalyticTraceGenerator(m, HW)
        # 1 request, output_len=2 -> 2 steps (1 prefill + 1 decode)
        trace, sched = gen.generate([MirrorRequest("0", 8, 2)])
        from scripts.serve_trace.capture_nsys import classify_kernel
        norms = sum(1 for e in trace if classify_kernel(e["kernel_name"]) == "NORM")
        # per step: 4*layers + 1 final = 4*2+1 = 9; 2 steps => 18
        assert norms == (4 * m.num_hidden_layers + 1) * len(sched)

    def test_llama_norm_count_no_qk_norm(self):
        m = _llama(layers=2)
        gen = AnalyticTraceGenerator(m, HW)
        trace, sched = gen.generate([MirrorRequest("0", 8, 2)])
        from scripts.serve_trace.capture_nsys import classify_kernel
        norms = sum(1 for e in trace if classify_kernel(e["kernel_name"]) == "NORM")
        # per step: 2*layers + 1 final
        assert norms == (2 * m.num_hidden_layers + 1) * len(sched)

    def test_attention_is_one_flash_kernel_per_forward(self):
        # Both prefill and decode emit exactly one flash_fwd_splitkv COMPUTE kernel
        # per layer per step. The split-KV combine epilogue is an autotuner
        # artifact (FLASH_SPLIT_REDUCE exclusion), not emitted by the recipe.
        m = _qwen3(layers=1)
        gen = AnalyticTraceGenerator(m, HW)
        trace, sched = gen.generate([MirrorRequest("0", 8, 2)])
        from scripts.serve_trace.capture_nsys import classify_kernel
        flash = [e for e in trace if classify_kernel(e["kernel_name"]) == "FLASH_ATTN"]
        # 1/layer/step over 2 steps (1 prefill + 1 decode), 1 layer = 2
        assert len(flash) == 2
        assert not any("splitkv_combine" in e["kernel_name"] for e in trace)

    def test_write_roundtrip(self, tmp_path):
        gen = AnalyticTraceGenerator(_qwen3(layers=1), HW)
        trace, sched = gen.generate([MirrorRequest("0", 8, 2)])
        out = str(tmp_path / "t.json")
        tp, sp = write_analytic_trace(trace, sched, out)
        import json
        assert json.load(open(tp))  # valid JSON list
        lines = [l for l in open(sp) if l.strip()]
        assert len(lines) == len(sched)
        rec = json.loads(lines[0])
        assert set(rec) >= {"iter_id", "ts_us", "scheduled", "total_tokens", "finished"}


class TestMoE:
    """MoE FFN recipe: router gate GEMM + topkGatingSoftmax + align/sort +
    2 fused_moe GEMMs + SiLU, replacing the dense gate_up/down GEMMs."""

    def _cls(self):
        from scripts.serve_trace.validate_analytic_vs_nsys import classify_kernel
        return classify_kernel

    def test_router_gate_gemm_present_each_moe_layer_step(self):
        # The router gate (hidden->num_experts) is a TC_GEMM emitted once per MoE
        # layer per step, in BOTH prefill and decode (it does not degrade to GEMV
        # at M=1). Regression for the missing-gate bug.
        cls = self._cls()
        m = _qwen3_moe(layers=2)
        gen = AnalyticTraceGenerator(m, HW)
        trace, sched = gen.generate([MirrorRequest("0", 8, 3)])  # 3 steps
        # FUSEDMOE = 2 per layer per step (w1, w2).
        fused = sum(1 for e in trace if cls(e["kernel_name"]) == "FUSEDMOE")
        assert fused == 2 * m.num_hidden_layers * len(sched)
        # The gate GEMM is the s16816gemm 64x64 emitted inside the MoE block; it
        # appears once per layer per step regardless of prefill/decode.
        gate = sum(1 for e in trace
                   if "s16816gemm_fp16_64x64_ldg8_f2f_stages_64x5" in e["kernel_name"])
        assert gate >= m.num_hidden_layers * len(sched)

    def test_moe_routing_kernels(self):
        cls = self._cls()
        m = _qwen3_moe(layers=2)
        gen = AnalyticTraceGenerator(m, HW)
        trace, sched = gen.generate([MirrorRequest("0", 8, 2)])
        reduce_ = sum(1 for e in trace if cls(e["kernel_name"]) == "REDUCE")
        gather = sum(1 for e in trace if cls(e["kernel_name"]) == "GATHER")
        # 1 router softmax (REDUCE) + 2 index kernels (GATHER) per layer per step.
        assert reduce_ == m.num_hidden_layers * len(sched)
        assert gather == 2 * m.num_hidden_layers * len(sched)

    def test_moe_has_no_dense_ffn_silu_only_moe_act(self):
        # MoE replaces the dense gate_up/down GEMMs; ACTIVATION is the per-layer
        # MoE SiLU (act_and_mul), one per layer per step.
        cls = self._cls()
        m = _qwen3_moe(layers=2)
        gen = AnalyticTraceGenerator(m, HW)
        trace, sched = gen.generate([MirrorRequest("0", 8, 2)])
        act = sum(1 for e in trace if cls(e["kernel_name"]) == "ACTIVATION")
        assert act == m.num_hidden_layers * len(sched)


class TestParallel:
    def test_tp_emits_two_allreduce_per_layer(self):
        # TP>1 inserts one NCCL all-reduce after o_proj and one after down_proj.
        from scripts.serve_trace.validate_analytic_vs_nsys import classify_kernel
        m = _llama(layers=2)
        gen = AnalyticTraceGenerator(m, HW, tp_size=2)
        trace, sched = gen.generate([MirrorRequest("0", 8, 2)])
        nccl = sum(1 for e in trace if classify_kernel(e["kernel_name"]) == "NCCL")
        assert nccl == 2 * m.num_hidden_layers * len(sched)

    def test_pp_partitions_layers_and_drops_lm_head(self):
        # PP models one rank: layers/pp of the layers, and (non-last stage) no
        # final norm / lm_head. Compare layer-NORM count to pp=1.
        from scripts.serve_trace.capture_nsys import classify_kernel
        m = _llama(layers=4)
        full = AnalyticTraceGenerator(m, HW, pp_size=1)
        half = AnalyticTraceGenerator(m, HW, pp_size=2)
        tf, sf = full.generate([MirrorRequest("0", 8, 2)])
        th, sh = half.generate([MirrorRequest("0", 8, 2)])
        norm_full = sum(1 for e in tf if classify_kernel(e["kernel_name"]) == "NORM")
        norm_half = sum(1 for e in th if classify_kernel(e["kernel_name"]) == "NORM")
        # pp=2 rank runs 2 of 4 layers and omits the final norm: per step
        # full=2*4+1=9, half=2*2=4 (no final norm on a non-last stage).
        assert norm_full == 9 * len(sf)
        assert norm_half == 4 * len(sh)

    def test_pp_must_divide_layers(self):
        with pytest.raises(ValueError):
            AnalyticTraceGenerator(_llama(layers=3), HW, pp_size=2)

    def test_ep_emits_post_ffn_allreduce_not_all2all(self):
        # The naive EP backend (A800) issues NO all-to-all: it uses expert_map +
        # the SAME post-FFN all-reduce that TP uses (vLLM reduces when tp>1 OR
        # ep>1). nsys confirms zero all-to-all kernels. With ep=2, tp=1 only the
        # post-FFN all-reduce fires -> 1 NCCL per layer per step, and no kernel
        # name contains "AllToAll".
        from scripts.serve_trace.validate_analytic_vs_nsys import classify_kernel
        m = _qwen3_moe(layers=2)
        gen = AnalyticTraceGenerator(m, HW, ep_size=2)
        trace, sched = gen.generate([MirrorRequest("0", 8, 2)])
        nccl = sum(1 for e in trace if classify_kernel(e["kernel_name"]) == "NCCL")
        assert nccl == 1 * m.num_hidden_layers * len(sched)
        assert not any("AllToAll" in e["kernel_name"] for e in trace)


class TestGemmCoalescing:
    def test_coalesce_merges_gemv_and_tc(self):
        from scripts.serve_trace.validate_analytic_vs_nsys import coalesce_gemm
        out = coalesce_gemm({"GEMV": 3, "TC_GEMM": 5, "NORM": 2})
        assert out == {"GEMM": 8, "NORM": 2}

    def test_coalesce_tc_only(self):
        from scripts.serve_trace.validate_analytic_vs_nsys import coalesce_gemm
        assert coalesce_gemm({"TC_GEMM": 4})["GEMM"] == 4
